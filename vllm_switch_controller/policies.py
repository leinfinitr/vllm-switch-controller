from typing import NamedTuple, Protocol

from vllm_switch_controller.state import ModelState


class SwitchDecision(NamedTuple):
    sleep_models: list[str]
    wake_model: str | None
    wait_for_active_requests: bool = False


class SwitchingPolicy(Protocol):
    def decide(
        self,
        current_active: str | None,
        target_model: str,
        states: dict[str, ModelState],
    ) -> SwitchDecision: ...


class AlwaysSleepPreviousPolicy:
    """Only one model is awake; sleep previous on every switch."""

    def decide(
        self,
        current_active: str | None,
        target_model: str,
        states: dict[str, ModelState],
    ) -> SwitchDecision:
        if target_model not in states:
            raise KeyError(target_model)
        if current_active == target_model and states[target_model] == ModelState.AWAKE:
            return SwitchDecision([], None)
        if current_active is None or current_active == target_model:
            return SwitchDecision([], target_model)
        return SwitchDecision(
            [current_active],
            target_model,
            wait_for_active_requests=True,
        )


class AlwaysAwakePreviousPolicy:
    """Keep previous model awake and switch after its pending requests finish."""

    def decide(
        self,
        current_active: str | None,
        target_model: str,
        states: dict[str, ModelState],
    ) -> SwitchDecision:
        # We can't get if the previous has pending requests,
        # use AlwaysSleepPreviousPolicy for now
        return AlwaysSleepPreviousPolicy().decide(current_active, target_model, states)


def make_policy(name: str) -> SwitchingPolicy:
    if name == "always_sleep_previous":
        return AlwaysSleepPreviousPolicy()
    if name == "always_awake_previous":
        return AlwaysAwakePreviousPolicy()
    raise ValueError(f"unknown switching policy: {name}")
