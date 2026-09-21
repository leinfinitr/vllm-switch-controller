import asyncio

import httpx
import pytest

from vllm_switch_controller.backends import EngineControlError
from vllm_switch_controller.config import ModelSpec
from vllm_switch_controller.engine_client import EngineClient


@pytest.mark.parametrize("fail_reload", [False, True])
async def test_l2_resume_reconstructs_before_ready(fail_reload):
    calls = []

    def handle(request):
        calls.append((request.url.path, request.url.query.decode()))
        if request.url.path == "/collective_rpc":
            assert request.content == b'{"method":"reload_weights"}'
            if fail_reload:
                return httpx.Response(500)
        return httpx.Response(200, json={"is_sleeping": False})

    client = EngineClient(
        {"a": ModelSpec(backend_url="http://backend", served_model_name="a", sleep_level=2)}
    )
    client._client._transport = httpx.MockTransport(handle)
    try:
        if fail_reload:
            with pytest.raises(EngineControlError, match="reload checkpoint"):
                await client.wake_up_and_wait("a")
            assert [path for path, _ in calls] == ["/wake_up", "/collective_rpc"]
        else:
            await client.wake_up_and_wait("a")
            assert calls == [
                ("/wake_up", "tags=weights"),
                ("/collective_rpc", ""),
                ("/wake_up", ""),
                ("/is_sleeping", ""),
            ]
    finally:
        await client.aclose()


async def test_l2_reload_is_inside_transition_deadline():
    async def handle(request):
        if request.url.path == "/collective_rpc":
            await asyncio.sleep(10)
        return httpx.Response(200, json={"is_sleeping": False})

    client = EngineClient(
        {"a": ModelSpec(backend_url="http://backend", served_model_name="a", sleep_level=2)},
        switch_timeout_s=0.03,
    )
    client._client._transport = httpx.MockTransport(handle)
    try:
        with pytest.raises(EngineControlError, match="timed out"):
            await client.wake_up_and_wait("a")
    finally:
        await client.aclose()


async def test_second_engine_adapter_uses_no_vllm_management_endpoints():
    class AlternativeEngine:
        sleeping = False

        async def health(self, request):
            return True

        async def sleep(self, request, level):
            self.sleeping = True

        async def resume(self, request, level, tags):
            self.sleeping = False

        async def is_sleeping(self, request):
            return self.sleeping

    client = EngineClient(
        {
            "a": ModelSpec(
                backend_url="http://unreachable", served_model_name="a", engine="alternative"
            )
        },
        adapters={"alternative": AlternativeEngine()},
    )
    try:
        assert await client.health("a")
        await client.sleep_and_wait("a", 1)
        assert await client.is_sleeping("a")
        await client.wake_up_and_wait("a")
        assert not await client.is_sleeping("a")
    finally:
        await client.aclose()


def test_unknown_adapter_fails_before_network():
    with pytest.raises(ValueError, match="unknown engine"):
        EngineClient(
            {
                "a": ModelSpec(
                    backend_url="http://unreachable", served_model_name="a", engine="unknown"
                )
            }
        )
