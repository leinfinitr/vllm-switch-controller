from vllm_switch_controller.policies import (
    AlwaysSleepPreviousPolicy,
)
from vllm_switch_controller.state import ModelState


def test_same_target_model_does_not_switch():
    policy = AlwaysSleepPreviousPolicy()

    decision = policy.decide(
        current_active="a",
        target_model="a",
        states={"a": ModelState.AWAKE},
    )

    assert decision.sleep_models == []
    assert decision.wake_model is None


def test_different_target_sleeps_previous_and_wakes_target():
    policy = AlwaysSleepPreviousPolicy()

    decision = policy.decide(
        current_active="a",
        target_model="b",
        states={"a": ModelState.AWAKE, "b": ModelState.SLEEPING},
    )

    assert decision.sleep_models == ["a"]
    assert decision.wake_model == "b"
    assert decision.wait_for_active_requests is True


def test_no_active_model_wakes_target():
    policy = AlwaysSleepPreviousPolicy()

    decision = policy.decide(
        current_active=None,
        target_model="a",
        states={"a": ModelState.SLEEPING},
    )

    assert decision.sleep_models == []
    assert decision.wake_model == "a"
