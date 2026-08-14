import asyncio

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from httpx import ASGITransport

from vllm_switch_controller.config import ModelSpec
from vllm_switch_controller.vllm_client import (
    VLLMClient,
    VLLMClientError,
    filter_end_to_end_headers,
)


def make_client() -> VLLMClient:
    return VLLMClient(
        {"a": ModelSpec(backend_url="http://testserver", served_model_name="a")},
        request_timeout_s=5,
        switch_timeout_s=0.1,
    )


def test_header_filter_removes_standard_and_connection_named_hops():
    headers = {
        "Connection": "keep-alive, X-Internal",
        "Keep-Alive": "timeout=5",
        "X-Internal": "private",
        "Transfer-Encoding": "chunked",
        "Content-Encoding": "gzip",
        "Content-Length": "123",
        "Authorization": "Bearer test",
    }

    assert filter_end_to_end_headers(headers, rebuilding_body=True) == {
        "Authorization": "Bearer test"
    }


def test_management_endpoint_rejects_redirect_status():
    response = httpx.Response(
        status_code=307,
        request=httpx.Request("POST", "http://backend/sleep"),
    )

    with pytest.raises(VLLMClientError, match="HTTP 307"):
        VLLMClient._raise_for_response(response, "sleep a")


@pytest.mark.asyncio
async def test_health_requires_success_status():
    app = FastAPI()

    @app.get("/health")
    async def health():
        return Response(status_code=404)

    client = VLLMClient(
        {"a": ModelSpec(backend_url="http://testserver", served_model_name="a")},
        timeout_s=5,
    )
    client._client._transport = ASGITransport(app=app)
    assert await client.health("a") is False
    await client.aclose()


@pytest.mark.asyncio
async def test_sleep_calls_vllm_sleep_endpoint_with_level():
    seen = {}
    app = FastAPI()

    @app.post("/sleep")
    async def sleep(request: Request):
        seen["level"] = request.query_params["level"]
        return {"ok": True}

    transport = ASGITransport(app=app)
    client = VLLMClient(
        {"a": ModelSpec(backend_url="http://testserver", served_model_name="a")},
        timeout_s=5,
    )
    client._client._transport = transport  # test-only in-process transport

    latency = await client.sleep("a", level=1)
    await client.aclose()

    assert seen == {"level": "1"}
    assert latency >= 0


@pytest.mark.asyncio
async def test_is_sleeping_requires_boolean_json_field():
    app = FastAPI()

    @app.get("/is_sleeping")
    async def is_sleeping():
        return {"is_sleeping": "yes"}

    client = make_client()
    client._client._transport = ASGITransport(app=app)

    with pytest.raises(VLLMClientError, match="boolean is_sleeping"):
        await client.is_sleeping("a")
    await client.aclose()


@pytest.mark.asyncio
async def test_wait_until_sleeping_polls_until_expected_state():
    app = FastAPI()
    probes = 0

    @app.get("/is_sleeping")
    async def is_sleeping():
        nonlocal probes
        probes += 1
        return {"is_sleeping": probes >= 3}

    client = make_client()
    client._client._transport = ASGITransport(app=app)

    latency = await client.wait_until_sleeping("a", expected=True, poll_interval_s=0)
    await client.aclose()

    assert probes == 3
    assert latency >= 0


@pytest.mark.asyncio
async def test_wait_until_sleeping_stops_at_switch_deadline():
    app = FastAPI()

    @app.get("/is_sleeping")
    async def is_sleeping():
        await asyncio.sleep(0)
        return {"is_sleeping": False}

    client = make_client()
    client._switch_timeout_s = 0.01
    client._client._transport = ASGITransport(app=app)

    with pytest.raises(VLLMClientError, match="timed out waiting for a to become sleeping"):
        await client.wait_until_sleeping("a", expected=True, poll_interval_s=0)
    await client.aclose()


@pytest.mark.asyncio
async def test_wait_until_sleeping_uses_one_deadline_for_slow_probes():
    app = FastAPI()

    @app.get("/is_sleeping")
    async def is_sleeping():
        await asyncio.sleep(1)
        return {"is_sleeping": False}

    client = make_client()
    client._switch_timeout_s = 0.02
    client.switch_timeout = httpx.Timeout(0.02)
    client._client._transport = ASGITransport(app=app)

    with pytest.raises(VLLMClientError, match="timed out waiting for a to become sleeping"):
        await client.wait_until_sleeping("a", expected=True, poll_interval_s=0)
    await client.aclose()


@pytest.mark.asyncio
async def test_sleep_and_probe_share_one_transition_deadline():
    client = make_client()
    client._switch_timeout_s = 0.05
    client.switch_timeout = httpx.Timeout(0.05)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sleep":
            await asyncio.sleep(0.04)
            return httpx.Response(200, request=request)
        if request.url.path == "/is_sleeping":
            await asyncio.sleep(0.04)
            return httpx.Response(200, json={"is_sleeping": True}, request=request)
        raise AssertionError(request.url.path)

    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(VLLMClientError, match="timed out"):
        await client.sleep_and_wait("a", 1)
    await client.aclose()


@pytest.mark.asyncio
async def test_wake_up_sends_repeated_tags_query_params():
    seen = {}
    app = FastAPI()

    @app.post("/wake_up")
    async def wake_up(request: Request):
        seen["tags"] = request.query_params.getlist("tags")
        return {"ok": True}

    transport = ASGITransport(app=app)
    client = VLLMClient(
        {"a": ModelSpec(backend_url="http://testserver", served_model_name="a")},
        timeout_s=5,
    )
    client._client._transport = transport

    await client.wake_up("a", tags=["weights", "kv_cache"])
    await client.aclose()

    assert seen == {"tags": ["weights", "kv_cache"]}


@pytest.mark.asyncio
async def test_wake_up_without_tags_requests_all_sleeping_allocations():
    seen = {}
    app = FastAPI()

    @app.post("/wake_up")
    async def wake_up(request: Request):
        seen["query"] = str(request.url.query)
        return {"ok": True}

    client = make_client()
    client._client._transport = ASGITransport(app=app)

    await client.wake_up("a", tags=None)
    await client.aclose()

    assert seen == {"query": ""}


@pytest.mark.asyncio
async def test_proxy_rewrites_route_alias_to_backend_model_name():
    seen = {}
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        seen.update(body)
        return {"ok": True}

    client = VLLMClient(
        {
            "route-alias": ModelSpec(
                backend_url="http://testserver",
                served_model_name="backend-model",
            )
        },
        timeout_s=5,
    )
    client._client._transport = ASGITransport(app=app)

    status, _, _ = await client.proxy_json(
        "route-alias",
        "/v1/chat/completions",
        {"model": "route-alias", "messages": []},
    )
    await client.aclose()

    assert status == 200
    assert seen["model"] == "backend-model"
