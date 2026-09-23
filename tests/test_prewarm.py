import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from vllm_switch_controller.config import ControllerConfig
from vllm_switch_controller.main import create_app
from vllm_switch_controller.state import ModelState


@pytest.fixture
def setup(tmp_path):
    app = create_app(
        ControllerConfig.model_validate(
            {
                "models": {
                    name: {"backend_url": f"http://{name}", "served_model_name": name}
                    for name in ("a", "b")
                },
                "controller": {
                    "startup_awake_model": "a",
                    "metrics_path": str(tmp_path / "events.jsonl"),
                },
            }
        )
    )
    state = app.state.controller_state
    state.startup_reconciled = True
    calls = []

    async def sleeping(model, **kwargs):
        return state.model_states[model] == ModelState.SLEEPING

    async def sleep(model, level, timeout):
        calls.append(f"sleep:{model}")
        return 0.01, {}

    async def wake(model, tags, timeout):
        calls.append(f"wake:{model}")
        return 0.02, {}

    app.state.vllm_client.is_sleeping = sleeping
    app.state.vllm_client.sleep_and_wait_with_timeout = sleep
    app.state.vllm_client.wake_up_and_wait_with_timeout = wake
    return app, state, calls


def hint(**kwargs):
    return {
        "model": "b",
        "task_id": "task",
        "stage_id": "tool",
        "hint_id": "h1",
        "source": "exact",
        "ttl_ms": 1000,
        **kwargs,
    }


@pytest.mark.asyncio
async def test_prewarm_ready_is_idempotent_and_does_not_reserve(setup, tmp_path):
    app, state, calls = setup
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as c:
        first = await c.post("/admin/prewarm", json=hint())
        duplicate = await c.post("/admin/prewarm", json=hint())
        another = await c.post("/admin/prewarm", json=hint(hint_id="h2"))
        conflict = await c.post("/admin/prewarm", json=hint(model="a"))
    assert first.json()["status"] == another.json()["status"] == "ready"
    assert duplicate.json()["duplicate"]
    assert conflict.status_code == 409
    assert calls == ["sleep:a", "wake:b"]
    assert await state.active_requests_snapshot() == {}
    rows = [json.loads(s) for s in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert rows[0]["task_id"] == "task"
    assert rows[0]["hint_status"] == "ready"
    assert rows[0]["switch_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", ["active", "queued", "lock", "expired"])
async def test_prewarm_never_waits_for_busy_work(setup, busy):
    app, state, calls = setup
    tracker = state.track_request("a")
    if busy == "active":
        await tracker.__aenter__()
    if busy == "queued":
        state.pending_demands = 1
    if busy == "lock":
        await state.switch_lock.acquire()
    try:
        async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as c:
            async with asyncio.timeout(1):
                response = await c.post(
                    "/admin/prewarm", json=hint(ttl_ms=0 if busy == "expired" else 1000)
                )
        assert response.json()["status"] == "ignored"
        assert response.json()["reason"] == ("expired" if busy == "expired" else "busy")
        assert calls == []
    finally:
        if busy == "active":
            await tracker.__aexit__(None, None, None)
        if busy == "lock":
            state.switch_lock.release()
        state.pending_demands = 0


@pytest.mark.asyncio
async def test_inflight_hint_cancellation_finishes_before_demand(setup):
    app, state, calls = setup
    started, finish = asyncio.Event(), asyncio.Event()

    async def wake(model, tags, timeout):
        calls.append("wake:start")
        started.set()
        await finish.wait()
        calls.append("wake:done")
        return 0.02, {}

    app.state.vllm_client.wake_up_and_wait_with_timeout = wake
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as c:
        task = asyncio.create_task(c.post("/admin/prewarm", json=hint()))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert state.switch_lock.locked()
        assert not task.done()
        demand = asyncio.create_task(c.post("/admin/switch/a"))
        await asyncio.sleep(0)
        assert not demand.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await demand).status_code == 200
    assert calls.index("wake:done") < calls.index("sleep:b")
    assert state.active_model == "a"
    assert state.pending_demands == 0


@pytest.mark.asyncio
async def test_hint_failure_keeps_unknown_transition_fail_closed(setup):
    app, state, calls = setup

    async def broken_wake(model, tags, timeout):
        raise RuntimeError("transport lost after command")

    app.state.vllm_client.wake_up_and_wait_with_timeout = broken_wake
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as c:
        response = await c.post("/admin/prewarm", json=hint())
        invalid = await c.post("/admin/prewarm", json=hint(model="missing"))
    assert response.json()["status"] == "failed"
    assert state.model_states["b"] == ModelState.ERROR
    assert state.active_model is None
    assert invalid.status_code == 404


@pytest.mark.asyncio
async def test_concurrent_duplicate_joins_original_transaction(setup):
    app, state, calls = setup
    started, finish = asyncio.Event(), asyncio.Event()

    async def wake(model, tags, timeout):
        calls.append("wake:b")
        started.set()
        await finish.wait()
        return 0.01, {}

    app.state.vllm_client.wake_up_and_wait_with_timeout = wake
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as c:
        first = asyncio.create_task(c.post("/admin/prewarm", json=hint()))
        await started.wait()
        second = asyncio.create_task(c.post("/admin/prewarm", json=hint()))
        await asyncio.sleep(0)
        finish.set()
        results = await asyncio.gather(first, second)
    assert [r.json()["duplicate"] for r in results] == [False, True]
    assert calls == ["sleep:a", "wake:b"]
