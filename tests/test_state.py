import pytest

from vllm_switch_controller.state import ControllerState, ModelState, UnknownModelError


def test_initial_state_marks_startup_model_awake_and_others_sleeping():
    state = ControllerState.from_models(["a", "b"], startup_awake_model="a")

    assert state.active_model == "a"
    assert state.model_states == {"a": ModelState.AWAKE, "b": ModelState.SLEEPING}


def test_initial_state_without_startup_model_marks_all_unknown():
    state = ControllerState.from_models(["a", "b"], startup_awake_model=None)

    assert state.active_model is None
    assert state.model_states == {"a": ModelState.UNKNOWN, "b": ModelState.UNKNOWN}


def test_require_model_rejects_unknown_model():
    state = ControllerState.from_models(["a"], startup_awake_model=None)

    with pytest.raises(UnknownModelError):
        state.require_model("missing")


def test_mark_switch_updates_active_and_sleeping_models():
    state = ControllerState.from_models(["a", "b"], startup_awake_model="a")

    state.mark_sleeping("a")
    state.mark_awake("b")

    assert state.active_model == "b"
    assert state.model_states["a"] == ModelState.SLEEPING
    assert state.model_states["b"] == ModelState.AWAKE


def test_mark_error_clears_inconsistent_active_model():
    state = ControllerState.from_models(["a"], startup_awake_model="a")

    state.mark_error("a")

    assert state.active_model is None
    assert state.model_states["a"] == ModelState.ERROR
