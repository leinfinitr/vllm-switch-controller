import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum


class ModelState(StrEnum):
    UNKNOWN = "unknown"
    AWAKE = "awake"
    SLEEPING = "sleeping"
    WAKING = "waking"
    SLEEPING_IN_PROGRESS = "sleeping_in_progress"
    ERROR = "error"


class UnknownModelError(KeyError):
    """Raised when a request references a model not present in config."""


@dataclass
class ControllerState:
    """Mutable runtime state guarded by switch_lock for model transitions."""

    active_model: str | None
    model_states: dict[str, ModelState]
    switch_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    startup_reconciled: bool = False
    _request_condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    _active_requests: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_models(
        cls, model_names: list[str] | tuple[str, ...], startup_awake_model: str | None
    ) -> "ControllerState":
        states = {name: ModelState.UNKNOWN for name in model_names}
        if startup_awake_model is not None:
            if startup_awake_model not in states:
                raise UnknownModelError(startup_awake_model)
            states = {name: ModelState.SLEEPING for name in model_names}
            states[startup_awake_model] = ModelState.AWAKE
        return cls(active_model=startup_awake_model, model_states=states)

    def require_model(self, model: str) -> None:
        if model not in self.model_states:
            raise UnknownModelError(model)

    def mark_awake(self, model: str) -> None:
        self.require_model(model)
        self.model_states[model] = ModelState.AWAKE
        self.active_model = model

    def mark_sleeping(self, model: str) -> None:
        self.require_model(model)
        self.model_states[model] = ModelState.SLEEPING
        if self.active_model == model:
            self.active_model = None

    def mark_waking(self, model: str) -> None:
        self.require_model(model)
        self.model_states[model] = ModelState.WAKING

    def mark_sleeping_in_progress(self, model: str) -> None:
        self.require_model(model)
        self.model_states[model] = ModelState.SLEEPING_IN_PROGRESS

    def mark_error(self, model: str) -> None:
        self.require_model(model)
        self.model_states[model] = ModelState.ERROR
        if self.active_model == model:
            self.active_model = None

    async def active_requests_snapshot(self) -> dict[str, int]:
        async with self._request_condition:
            return dict(self._active_requests)

    @asynccontextmanager
    async def track_request(self, model: str) -> AsyncIterator[None]:
        self.require_model(model)
        async with self._request_condition:
            self._active_requests[model] = self._active_requests.get(model, 0) + 1
        try:
            yield
        finally:
            async with self._request_condition:
                active = self._active_requests[model] - 1
                if active:
                    self._active_requests[model] = active
                else:
                    del self._active_requests[model]
                self._request_condition.notify_all()

    async def wait_for_other_model_requests_to_finish(self, model: str) -> None:
        self.require_model(model)
        async with self._request_condition:
            await self._request_condition.wait_for(
                lambda: all(
                    active_model == model or active_count == 0
                    for active_model, active_count in self._active_requests.items()
                )
            )
