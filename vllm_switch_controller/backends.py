"""Engine control contracts. Implementations exchange management metadata only."""

from typing import Any, Protocol

import httpx


class EngineControlError(RuntimeError):
    def __init__(self, message: str, *, transition_latency_s: float | None = None):
        super().__init__(message)
        self.transition_latency_s = transition_latency_s


class ControlRequest(Protocol):
    async def __call__(self, method: str, path: str, **kwargs: Any) -> httpx.Response: ...


class EngineControlAdapter(Protocol):
    async def health(self, request: ControlRequest) -> bool: ...

    async def sleep(self, request: ControlRequest, level: int) -> None: ...

    async def resume(self, request: ControlRequest, level: int, tags: list[str] | None) -> None: ...

    async def is_sleeping(self, request: ControlRequest) -> bool: ...


def require_success(response: httpx.Response, action: str) -> None:
    if not 200 <= response.status_code < 300:
        raise EngineControlError(
            f"{action} failed with HTTP {response.status_code}: {response.text[:500]}"
        )


class VllmControlAdapter:
    async def health(self, request: ControlRequest) -> bool:
        try:
            response = await request("GET", "/health")
            return 200 <= response.status_code < 300
        except EngineControlError:
            return False

    async def sleep(self, request: ControlRequest, level: int) -> None:
        require_success(await request("POST", "/sleep", params={"level": level}), "sleep")

    async def resume(self, request: ControlRequest, level: int, tags: list[str] | None) -> None:
        if level == 2:
            # Weight mappings must exist before reconstruction; KV follows it.
            if tags is not None and not {"weights", "kv_cache"}.issubset(tags):
                raise EngineControlError("L2 resume requires both weights and kv_cache")
            require_success(
                await request("POST", "/wake_up", params=[("tags", "weights")]), "wake weights"
            )
            require_success(
                await request("POST", "/collective_rpc", json={"method": "reload_weights"}),
                "reload checkpoint",
            )
            tags = [tag for tag in tags if tag != "weights"] if tags is not None else None
        params = [("tags", tag) for tag in tags] if tags else None
        require_success(await request("POST", "/wake_up", params=params), "wake")

    async def is_sleeping(self, request: ControlRequest) -> bool:
        response = await request("GET", "/is_sleeping")
        require_success(response, "is_sleeping")
        try:
            value = response.json()["is_sleeping"]
        except (ValueError, KeyError, TypeError) as exc:
            raise EngineControlError("backend did not return boolean is_sleeping") from exc
        if not isinstance(value, bool):
            raise EngineControlError("backend did not return boolean is_sleeping")
        return value
