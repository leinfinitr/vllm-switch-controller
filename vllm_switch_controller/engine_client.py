import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx

from vllm_switch_controller.backends import (
    ControlRequest,
    EngineControlAdapter,
    EngineControlError,
    VllmControlAdapter,
    require_success,
)
from vllm_switch_controller.config import ModelSpec

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def filter_end_to_end_headers(
    headers: Mapping[str, str] | None,
    *,
    rebuilding_body: bool,
) -> dict[str, str]:
    """Drop RFC hop-by-hop headers and fields named by Connection."""
    if not headers:
        return {}
    connection_tokens: set[str] = set()
    for key, value in headers.items():
        if key.lower() == "connection":
            connection_tokens.update(token.strip().lower() for token in value.split(","))
    excluded = HOP_BY_HOP_HEADERS | connection_tokens | {"host"}
    if rebuilding_body:
        # JSON is re-encoded and httpx response bytes are decoded, so original
        # representation metadata would be incorrect downstream.
        excluded |= {"content-length", "content-encoding"}
    return {key: value for key, value in headers.items() if key.lower() not in excluded}


class EngineClient:
    """Engine-adapted lifecycle control and OpenAI-compatible proxying."""

    def __init__(
        self,
        models: Mapping[str, ModelSpec],
        request_timeout_s: float = 600,
        switch_timeout_s: float = 600,
        *,
        timeout_s: float | None = None,
        adapters: Mapping[str, EngineControlAdapter] | None = None,
    ) -> None:
        self.models = dict(models)
        self.adapters: dict[str, EngineControlAdapter] = {"vllm": VllmControlAdapter()}
        if adapters:
            self.adapters.update(adapters)
        for spec in self.models.values():
            if spec.engine not in self.adapters:
                raise ValueError(f"unknown engine adapter: {spec.engine}")
        self.sleep_levels = {name: spec.sleep_level for name, spec in self.models.items()}
        # Keep the old keyword temporarily for callers outside this repository.
        # Also used by test_vllm_client.py to override both timeouts at once.
        if timeout_s is not None:
            request_timeout_s = timeout_s
            switch_timeout_s = timeout_s
        self.timeout = httpx.Timeout(request_timeout_s, connect=30.0)
        self.switch_timeout = httpx.Timeout(switch_timeout_s, connect=30.0)
        self._switch_timeout_s = switch_timeout_s
        # Backend control-plane URLs are explicit and commonly loopback/private.
        # Environment proxies can bypass test transports and misroute local vLLM
        # sleep/wake requests, so do not inherit them here.
        self._client = httpx.AsyncClient(timeout=self.timeout, trust_env=False)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def gpu_residency(self, model: str) -> dict[str, Any]:
        if self._spec(model).engine != "vllm":
            raise EngineControlError("partial GPU eviction requires an adapter capability")
        response = await self._control_request(model)("GET", "/gpu_residency")
        require_success(response, "GPU residency")
        try:
            data = response.json()
        except ValueError as exc:
            raise EngineControlError("invalid GPU residency JSON") from exc
        if not isinstance(data, dict):
            raise EngineControlError("GPU residency must be an object")
        if (
            data.get("schema_version") != 1
            or data.get("capability") != "partial-gpu-sleep-v1"
            or data.get("world_size") != 1
            or data.get("checkpoint_pending")
            or data.get("state") not in {"awake", "sleeping"}
        ):
            raise EngineControlError("unsupported or unsafe GPU residency response")
        for key in [
            "resident_bytes",
            "missing_bytes",
            "free_device_bytes",
            "total_device_bytes",
            "pid",
        ]:
            if type(data.get(key)) is not int or data[key] < 0:
                raise EngineControlError(f"invalid GPU residency field: {key}")
        if not data.get("device_uuid"):
            raise EngineControlError("missing GPU device identity")
        if data["pid"] == 0 or data["free_device_bytes"] > data["total_device_bytes"]:
            raise EngineControlError("invalid GPU residency identity or capacity")
        return data

    async def sleep_partial_and_wait(self, model: str, release_bytes: int, timeout_s: float):
        async def transition():
            start = time.perf_counter()
            response = await self._control_request(model, timeout_s)(
                "POST", "/sleep_partial", params={"release_bytes": release_bytes}
            )
            require_success(response, "partial GPU sleep")
            return time.perf_counter() - start

        return await self._transition_and_wait(
            transition, model, expected=True, timeout_s=timeout_s
        )

    async def wake_partial_and_wait(self, model: str, timeout_s: float):
        async def transition():
            start = time.perf_counter()
            response = await self._control_request(model, timeout_s)("POST", "/wake_partial")
            require_success(response, "partial GPU wake")
            return time.perf_counter() - start

        return await self._transition_and_wait(
            transition, model, expected=False, timeout_s=timeout_s
        )

    def _spec(self, model: str) -> ModelSpec:
        try:
            return self.models[model]
        except KeyError as exc:
            raise EngineControlError(f"unknown model: {model}") from exc

    def _control_request(self, model: str, timeout_s: float | None = None) -> ControlRequest:
        spec = self._spec(model)

        async def request(method: str, path: str, **kwargs: Any) -> httpx.Response:
            return await self._request(
                method,
                f"{spec.backend_url}{path}",
                timeout=self.switch_timeout if timeout_s is None else timeout_s,
                **kwargs,
            )

        return request

    def _adapter(self, model: str) -> EngineControlAdapter:
        return self.adapters[self._spec(model).engine]

    async def health(self, model: str) -> bool:
        return await self._adapter(model).health(self._control_request(model))

    async def sleep(self, model: str, level: int, *, timeout_s: float | None = None) -> float:
        start = time.perf_counter()
        await self._adapter(model).sleep(self._control_request(model, timeout_s), level)
        self.sleep_levels[model] = level
        return time.perf_counter() - start

    async def sleep_and_wait(self, model: str, level: int) -> tuple[float, float]:
        return await self.sleep_and_wait_with_timeout(model, level, self._switch_timeout_s)

    async def sleep_and_wait_with_timeout(
        self, model: str, level: int, timeout_s: float
    ) -> tuple[float, float]:
        return await self._transition_and_wait(
            lambda: self.sleep(model, level, timeout_s=timeout_s),
            model,
            expected=True,
            timeout_s=timeout_s,
        )

    async def wake_up(
        self, model: str, tags: list[str] | None = None, *, timeout_s: float | None = None
    ) -> float:
        start = time.perf_counter()
        async with asyncio.timeout(self._switch_timeout_s if timeout_s is None else timeout_s):
            await self._adapter(model).resume(
                self._control_request(model, timeout_s),
                self.sleep_levels[model],
                tags,
            )
        return time.perf_counter() - start

    async def wake_up_and_wait(
        self, model: str, tags: list[str] | None = None
    ) -> tuple[float, float]:
        return await self.wake_up_and_wait_with_timeout(model, tags, self._switch_timeout_s)

    async def wake_up_and_wait_with_timeout(
        self, model: str, tags: list[str] | None, timeout_s: float
    ) -> tuple[float, float]:
        return await self._transition_and_wait(
            lambda: self.wake_up(model, tags, timeout_s=timeout_s),
            model,
            expected=False,
            timeout_s=timeout_s,
        )

    async def _transition_and_wait(
        self,
        transition: Callable[[], Awaitable[float]],
        model: str,
        *,
        expected: bool,
        timeout_s: float,
    ) -> tuple[float, float]:
        """Run one lifecycle request and its post-condition under one deadline."""
        state = "sleeping" if expected else "awake"
        latency: float | None = None
        try:
            async with asyncio.timeout(timeout_s):
                latency = await transition()
                probe_latency = await self.wait_until_sleeping(
                    model, expected=expected, timeout_s=timeout_s
                )
        except TimeoutError as exc:
            raise EngineControlError(
                f"timed out waiting for {model} to become {state}",
                transition_latency_s=latency,
            ) from exc
        except EngineControlError as exc:
            if latency is not None and exc.transition_latency_s is None:
                exc.transition_latency_s = latency
            raise
        return latency, probe_latency

    async def is_sleeping(self, model: str, *, timeout_s: float | None = None) -> bool:
        return await self._adapter(model).is_sleeping(self._control_request(model, timeout_s))

    async def wait_until_sleeping(
        self,
        model: str,
        *,
        expected: bool,
        timeout_s: float | None = None,
        poll_interval_s: float = 0.1,
    ) -> float:
        start = time.perf_counter()
        deadline = start + (self._switch_timeout_s if timeout_s is None else timeout_s)
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                state = "sleeping" if expected else "awake"
                raise EngineControlError(f"timed out waiting for {model} to become {state}")
            try:
                async with asyncio.timeout(remaining):
                    sleeping = await self.is_sleeping(model)
            except TimeoutError as exc:
                state = "sleeping" if expected else "awake"
                raise EngineControlError(
                    f"timed out waiting for {model} to become {state}"
                ) from exc
            if sleeping is expected:
                return time.perf_counter() - start
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                state = "sleeping" if expected else "awake"
                raise EngineControlError(f"timed out waiting for {model} to become {state}")
            await asyncio.sleep(min(poll_interval_s, remaining))

    async def proxy_json(
        self,
        model: str,
        path: str,
        body: dict[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        spec = self._spec(model)
        backend_body = {**body, "model": spec.served_model_name}
        response = await self._request(
            "POST",
            f"{spec.backend_url}{path}",
            json=backend_body,
            headers=filter_end_to_end_headers(headers, rebuilding_body=True),
        )
        return response.status_code, dict(response.headers), response.content

    @asynccontextmanager
    async def proxy_stream(
        self,
        model: str,
        path: str,
        body: dict[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[httpx.Response]:
        spec = self._spec(model)
        backend_body = {**body, "model": spec.served_model_name}
        try:
            async with self._client.stream(
                "POST",
                f"{spec.backend_url}{path}",
                json=backend_body,
                headers=filter_end_to_end_headers(headers, rebuilding_body=True),
            ) as response:
                yield response
        except httpx.HTTPError as exc:
            raise EngineControlError(f"vLLM proxy stream failed: {exc}") from exc

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise EngineControlError(f"vLLM request failed: {exc}") from exc

    @staticmethod
    def _raise_for_response(response: httpx.Response, action: str) -> None:
        require_success(response, action)
