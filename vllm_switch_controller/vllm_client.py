import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx

from vllm_switch_controller.config import ModelSpec


class VLLMClientError(RuntimeError):
    """Raised when a vLLM management or proxy call fails."""

    def __init__(self, message: str, *, transition_latency_s: float | None = None):
        super().__init__(message)
        self.transition_latency_s = transition_latency_s


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


class VLLMClient:
    """HTTP client for vLLM management endpoints and OpenAI-compatible proxying."""

    def __init__(
        self,
        models: Mapping[str, ModelSpec],
        request_timeout_s: float = 600,
        switch_timeout_s: float = 600,
        *,
        timeout_s: float | None = None,
    ) -> None:
        self.models = dict(models)
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

    def _spec(self, model: str) -> ModelSpec:
        try:
            return self.models[model]
        except KeyError as exc:
            raise VLLMClientError(f"unknown model: {model}") from exc

    async def health(self, model: str) -> bool:
        spec = self._spec(model)
        try:
            response = await self._client.get(f"{spec.backend_url}/health")
            return 200 <= response.status_code < 300
        except httpx.HTTPError:
            return False

    async def sleep(self, model: str, level: int, *, timeout_s: float | None = None) -> float:
        spec = self._spec(model)
        start = time.perf_counter()
        response = await self._request(
            "POST",
            f"{spec.backend_url}/sleep",
            params={"level": level},
            timeout=self.switch_timeout if timeout_s is None else timeout_s,
        )
        self._raise_for_response(response, f"sleep {model}")
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
        self,
        model: str,
        tags: list[str] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> float:
        spec = self._spec(model)
        start = time.perf_counter()
        params: list[tuple[str, str]] | None = None
        if tags:
            params = [("tags", tag) for tag in tags]
        response = await self._request(
            "POST",
            f"{spec.backend_url}/wake_up",
            params=params,
            timeout=self.switch_timeout if timeout_s is None else timeout_s,
        )
        self._raise_for_response(response, f"wake_up {model}")
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
            raise VLLMClientError(
                f"timed out waiting for {model} to become {state}",
                transition_latency_s=latency,
            ) from exc
        except VLLMClientError as exc:
            if latency is not None and exc.transition_latency_s is None:
                exc.transition_latency_s = latency
            raise
        return latency, probe_latency

    async def is_sleeping(self, model: str, *, timeout_s: float | None = None) -> bool:
        spec = self._spec(model)
        response = await self._request(
            "GET",
            f"{spec.backend_url}/is_sleeping",
            timeout=self.switch_timeout if timeout_s is None else timeout_s,
        )
        self._raise_for_response(response, f"is_sleeping {model}")
        try:
            value = response.json()["is_sleeping"]
        except (ValueError, KeyError, TypeError) as exc:
            raise VLLMClientError(
                f"vLLM is_sleeping {model} did not return boolean is_sleeping"
            ) from exc
        if not isinstance(value, bool):
            raise VLLMClientError(f"vLLM is_sleeping {model} did not return boolean is_sleeping")
        return value

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
                raise VLLMClientError(f"timed out waiting for {model} to become {state}")
            try:
                async with asyncio.timeout(remaining):
                    sleeping = await self.is_sleeping(model)
            except TimeoutError as exc:
                state = "sleeping" if expected else "awake"
                raise VLLMClientError(f"timed out waiting for {model} to become {state}") from exc
            if sleeping is expected:
                return time.perf_counter() - start
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                state = "sleeping" if expected else "awake"
                raise VLLMClientError(f"timed out waiting for {model} to become {state}")
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
            raise VLLMClientError(f"vLLM proxy stream failed: {exc}") from exc

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise VLLMClientError(f"vLLM request failed: {exc}") from exc

    @staticmethod
    def _raise_for_response(response: httpx.Response, action: str) -> None:
        if not 200 <= response.status_code < 300:
            raise VLLMClientError(
                f"vLLM {action} failed with HTTP {response.status_code}: {response.text[:500]}"
            )
