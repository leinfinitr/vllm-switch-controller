import pytest
from httpx import ASGITransport, AsyncClient

from vllm_switch_controller.config import ControllerConfig
from vllm_switch_controller.main import create_app


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "capacity", "restart", "device"])
async def test_partial_policy_budget_and_physical_admission(tmp_path, failure):
    config = ControllerConfig.model_validate(
        {
            "models": {
                name: {"backend_url": f"http://{name}", "served_model_name": name}
                for name in ["a", "b"]
            },
            "controller": {
                "startup_awake_model": "a",
                "partial_gpu_eviction": True,
                "gpu_memory_margin_bytes": 100,
                "metrics_path": str(tmp_path / "events"),
            },
        }
    )
    app = create_app(config)
    app.state.controller_state.startup_reconciled = True
    client = app.state.vllm_client
    calls = []
    slept = False

    async def residency(model):
        return {
            "pid": 99 if failure == "restart" and slept else 1,
            "device_uuid": model if failure == "device" else "gpu",
            "missing_bytes": 1000 if model == "b" else 0,
            "free_device_bytes": 200 if not slept or failure == "capacity" else 1200,
        }

    async def sleep(model, release_bytes, timeout_s):
        nonlocal slept
        calls.append(("sleep", model, release_bytes))
        slept = True
        return 0.1, 0

    async def wake(model, timeout_s):
        calls.append(("wake", model))
        return 0.1, 0

    client.gpu_residency = residency
    client.sleep_partial_and_wait = sleep
    client.wake_partial_and_wait = wake
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as c:
        # Admin switches use the same readiness transaction; lifecycle failures
        # propagate as EngineControlError without attempting a competing wake.
        from vllm_switch_controller.backends import EngineControlError

        if failure:
            with pytest.raises(EngineControlError):
                await c.post("/admin/switch/b")
            assert not any(x[0] == "wake" for x in calls)
        else:
            response = await c.post("/admin/switch/b")
            assert response.status_code == 200
            assert calls == [("sleep", "a", 900), ("wake", "b")]
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["[]", "not JSON", '{"schema_version": 2}'])
async def test_residency_protocol_rejects_malformed_response(body):
    import httpx

    from vllm_switch_controller.config import ModelSpec
    from vllm_switch_controller.engine_client import EngineClient, EngineControlError

    client = EngineClient({"a": ModelSpec(backend_url="http://a", served_model_name="a")})
    client._client._transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    try:
        with pytest.raises(EngineControlError):
            await client.gpu_residency("a")
    finally:
        await client.aclose()
