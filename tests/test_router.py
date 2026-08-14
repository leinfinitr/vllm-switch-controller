import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient

import vllm_switch_controller.router as router_module
from vllm_switch_controller.config import ControllerConfig
from vllm_switch_controller.main import create_app
from vllm_switch_controller.metrics import MetricsRecorder
from vllm_switch_controller.router import CancellationResistantCleanup, CleanupStreamingResponse
from vllm_switch_controller.state import ModelState
from vllm_switch_controller.vllm_client import VLLMClientError


def make_backend(label: str, events: list[str]) -> FastAPI:
    app = FastAPI()
    sleeping = label != "a"

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/sleep")
    async def sleep():
        nonlocal sleeping
        sleeping = True
        events.append(f"sleep:{label}")
        return {"ok": True}

    @app.post("/wake_up")
    async def wake_up():
        nonlocal sleeping
        sleeping = False
        events.append(f"wake:{label}")
        return {"ok": True}

    @app.get("/is_sleeping")
    async def is_sleeping():
        return {"is_sleeping": sleeping}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        events.append(f"chat:{label}")
        return {"model": body["model"], "choices": [{"message": {"content": label}}]}

    return app


@pytest.mark.asyncio
async def test_cleanup_is_cached_across_concurrent_callers():
    started = asyncio.Event()
    finish = asyncio.Event()
    cleanup_count = 0

    async def close_once():
        nonlocal cleanup_count
        cleanup_count += 1
        started.set()
        await finish.wait()

    cleanup = CancellationResistantCleanup(close_once)
    first = asyncio.create_task(cleanup())
    await started.wait()
    second = asyncio.create_task(cleanup())
    finish.set()
    await asyncio.gather(first, second)

    assert cleanup_count == 1


@pytest.mark.asyncio
async def test_cleanup_error_precedes_waiter_cancellation():
    started = asyncio.Event()
    finish = asyncio.Event()

    async def fail_cleanup():
        started.set()
        await finish.wait()
        raise RuntimeError("synthetic cleanup failure")

    cleanup = CancellationResistantCleanup(fail_cleanup)
    waiter = asyncio.create_task(cleanup())
    await started.wait()
    waiter.cancel()
    finish.set()

    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        await waiter
    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        await cleanup()


@pytest.mark.asyncio
async def test_controller_switches_by_model_and_proxies_json(tmp_path):
    events: list[str] = []
    config = ControllerConfig.model_validate(
        {
            "models": {
                "a": {"backend_url": "http://a", "served_model_name": "a"},
                "b": {"backend_url": "http://b", "served_model_name": "b"},
            },
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    controller_app = create_app(config)
    backend_a = make_backend("a", events)
    backend_b = make_backend("b", events)

    # monkeypatch by replacing the httpx client transport with a simple router below
    class RouterTransport(ASGITransport):
        async def handle_async_request(self, request):
            if request.url.host == "a":
                return await ASGITransport(backend_a).handle_async_request(request)
            return await ASGITransport(backend_b).handle_async_request(request)

    controller_app.state.vllm_client._client._transport = RouterTransport(
        make_backend("unused", events)
    )

    async with AsyncClient(
        transport=ASGITransport(controller_app),
        base_url="http://controller",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "b", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "b"
    assert events == ["sleep:a", "wake:b", "chat:b"]


@pytest.mark.asyncio
async def test_unknown_proxy_model_returns_404_without_context_error(tmp_path):
    config = ControllerConfig.model_validate(
        {
            "models": {
                "a": {"backend_url": "http://a", "served_model_name": "a"},
            },
            "controller": {"metrics_path": str(tmp_path / "events.jsonl")},
        }
    )
    app = create_app(config)

    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "missing", "messages": []},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "unknown model: missing"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "detail"),
    [
        (b"{not-json", "request body must be valid JSON"),
        (b"[]", "request JSON must be an object"),
        (b"null", "request JSON must be an object"),
    ],
)
async def test_proxy_rejects_malformed_or_non_object_json_with_400(tmp_path, content, detail):
    config = ControllerConfig.model_validate(
        {
            "models": {
                "a": {"backend_url": "http://a", "served_model_name": "a"},
            },
            "controller": {"metrics_path": str(tmp_path / "events.jsonl")},
        }
    )
    app = create_app(config)

    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        response = await client.post(
            "/v1/chat/completions",
            content=content,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == detail


@pytest.mark.asyncio
async def test_proxy_reuses_and_forwards_client_request_id(tmp_path):
    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    seen_headers = {}

    async def observed_awake(_model, *, timeout_s=None):
        return False

    async def fake_json(_model, _path, _body, headers=None):
        seen_headers.update(headers or {})
        return 200, {"content-type": "application/json"}, b"{}"

    app.state.vllm_client.is_sleeping = observed_awake
    app.state.vllm_client.proxy_json = fake_json
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"x-request-id": "client-r1"},
            json={"model": "a", "messages": []},
        )

    assert response.status_code == 200
    assert seen_headers["x-request-id"] == "client-r1"
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["request_id"] == "client-r1"


@pytest.mark.asyncio
async def test_switch_failure_does_not_exit_unentered_request_tracker(tmp_path, monkeypatch):
    config = ControllerConfig.model_validate(
        {
            "models": {
                "a": {"backend_url": "http://a", "served_model_name": "a"},
                "b": {"backend_url": "http://b", "served_model_name": "b"},
            },
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)

    async def observed_state(model, *, timeout_s=None):
        return model == "b"

    async def fail_sleep(*_args, **_kwargs):
        raise VLLMClientError("synthetic sleep failure")

    def fail_record(_self, _metrics):
        raise OSError("synthetic metrics failure")

    app.state.vllm_client.sleep = fail_sleep
    app.state.vllm_client.is_sleeping = observed_state
    monkeypatch.setattr(MetricsRecorder, "record", fail_record)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "b", "messages": []},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "synthetic sleep failure"
    assert app.state.controller_state._active_requests == {}
    assert app.state.controller_state.model_states["a"] == ModelState.ERROR


@pytest.mark.asyncio
async def test_cancelled_switch_marks_transition_state_error(tmp_path):
    config = ControllerConfig.model_validate(
        {
            "models": {
                "a": {"backend_url": "http://a", "served_model_name": "a"},
                "b": {"backend_url": "http://b", "served_model_name": "b"},
            },
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)

    async def cancel_sleep(*_args, **_kwargs):
        raise asyncio.CancelledError

    async def observed_state(model, *, timeout_s=None):
        return model == "b"

    app.state.vllm_client.sleep = cancel_sleep
    app.state.vllm_client.is_sleeping = observed_state
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        with pytest.raises(asyncio.CancelledError):
            await client.post(
                "/v1/chat/completions",
                json={"model": "b", "messages": []},
            )

    assert app.state.controller_state.model_states["a"] == ModelState.ERROR
    assert app.state.controller_state._active_requests == {}


@pytest.mark.asyncio
async def test_cancelled_wake_marks_transition_state_error(tmp_path):
    config = ControllerConfig.model_validate(
        {
            "models": {
                "a": {"backend_url": "http://a", "served_model_name": "a"},
            },
            "controller": {"metrics_path": str(tmp_path / "events.jsonl")},
        }
    )
    app = create_app(config)
    app.state.controller_state.model_states["a"] = ModelState.SLEEPING
    app.state.controller_state.startup_reconciled = True

    async def cancel_wake(*_args, **_kwargs):
        raise asyncio.CancelledError

    app.state.vllm_client.wake_up_and_wait_with_timeout = cancel_wake
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        with pytest.raises(asyncio.CancelledError):
            await client.post(
                "/v1/chat/completions",
                json={"model": "a", "messages": []},
            )

    assert app.state.controller_state.model_states["a"] == ModelState.ERROR
    assert app.state.controller_state._active_requests == {}


@pytest.mark.asyncio
async def test_unknown_lifecycle_outcome_blocks_later_wake(tmp_path):
    config = ControllerConfig.model_validate(
        {
            "models": {
                "a": {"backend_url": "http://a", "served_model_name": "a"},
                "b": {"backend_url": "http://b", "served_model_name": "b"},
            },
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    wake_calls = []
    sleeping_calls = []

    sleep_attempts = 0

    async def uncertain_sleep(*_args, **_kwargs):
        nonlocal sleep_attempts
        sleep_attempts += 1
        if sleep_attempts == 1:
            raise VLLMClientError("sleep outcome unknown")
        return (0.01, 0.01)

    async def is_sleeping(model, *, timeout_s=None):
        sleeping_calls.append(model)
        return model != "a"

    async def wake(*args, **kwargs):
        wake_calls.append(args[0])
        return (0.01, 0.01)

    async def proxy(*_args, **_kwargs):
        return (200, {"content-type": "application/json"}, b"{}")

    app.state.vllm_client.sleep_and_wait_with_timeout = uncertain_sleep
    app.state.vllm_client.is_sleeping = is_sleeping
    app.state.vllm_client.wake_up_and_wait_with_timeout = wake
    app.state.vllm_client.proxy_json = proxy
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        first = await client.post("/v1/chat/completions", json={"model": "b", "messages": []})
        second = await client.post("/v1/chat/completions", json={"model": "b", "messages": []})

    assert first.status_code == 502
    assert second.status_code == 200
    assert wake_calls == ["b"]
    assert app.state.controller_state.model_states["a"] == ModelState.SLEEPING


@pytest.mark.asyncio
async def test_non_stream_double_cancel_waits_for_tracker_exit(tmp_path):
    backend_started = asyncio.Event()
    tracker_exit_started = asyncio.Event()
    allow_tracker_exit = asyncio.Event()
    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    state = app.state.controller_state
    state.startup_reconciled = True
    original_track_request = state.track_request

    @asynccontextmanager
    async def blocking_tracker(model):
        tracker = original_track_request(model)
        await tracker.__aenter__()
        try:
            yield
        finally:
            tracker_exit_started.set()
            await allow_tracker_exit.wait()
            await tracker.__aexit__(None, None, None)

    async def blocked_proxy(*_args, **_kwargs):
        backend_started.set()
        await asyncio.Event().wait()

    state.track_request = blocking_tracker
    app.state.vllm_client.proxy_json = blocked_proxy
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        request_task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={"model": "a", "messages": []},
            )
        )
        await backend_started.wait()
        request_task.cancel()
        await tracker_exit_started.wait()
        request_task.cancel()
        allow_tracker_exit.set()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    assert state._active_requests == {}


@pytest.mark.asyncio
async def test_tracker_enter_cancellation_completes_enter_then_exits(tmp_path):
    tracker_enter_started = asyncio.Event()
    allow_tracker_enter = asyncio.Event()
    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    state = app.state.controller_state
    state.startup_reconciled = True
    original_track_request = state.track_request

    @asynccontextmanager
    async def blocking_enter_tracker(model):
        tracker = original_track_request(model)
        await tracker.__aenter__()
        tracker_enter_started.set()
        await allow_tracker_enter.wait()
        try:
            yield
        finally:
            await tracker.__aexit__(None, None, None)

    state.track_request = blocking_enter_tracker
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        request_task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={"model": "a", "messages": []},
            )
        )
        await tracker_enter_started.wait()
        request_task.cancel()
        await asyncio.sleep(0)
        request_task.cancel()
        allow_tracker_enter.set()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    assert state._active_requests == {}


@pytest.mark.asyncio
async def test_stream_response_cleanup_runs_before_body_iteration():
    cleanup_count = 0
    iterator_started = False

    async def body():
        nonlocal iterator_started
        iterator_started = True
        yield b"unused"

    async def cleanup():
        nonlocal cleanup_count
        cleanup_count += 1

    response = CleanupStreamingResponse(body(), cleanup=cleanup)

    async def receive():
        return {"type": "http.disconnect"}

    async def fail_start(_message):
        raise RuntimeError("synthetic send failure")

    scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
    with pytest.raises(RuntimeError, match="synthetic send failure"):
        await response(scope, receive, fail_start)

    assert iterator_started is False
    assert cleanup_count == 1


@pytest.mark.asyncio
async def test_stream_response_cleanup_runs_once_after_normal_iteration():
    cleanup_count = 0
    messages = []

    async def body():
        yield b"one"

    async def cleanup():
        nonlocal cleanup_count
        cleanup_count += 1

    response = CleanupStreamingResponse(body(), cleanup=cleanup)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
    await response(scope, receive, send)

    assert cleanup_count == 1
    assert any(message.get("body") == b"one" for message in messages)


@pytest.mark.asyncio
async def test_stream_setup_failure_preserves_error_and_releases_tracker(tmp_path, monkeypatch):
    backend = FastAPI()

    @backend.post("/v1/chat/completions")
    async def chat():
        return StreamingResponse(iter([b"unused"]))

    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    app.state.controller_state.startup_reconciled = True
    app.state.vllm_client._client._transport = ASGITransport(backend)

    def fail_headers(*_args, **_kwargs):
        raise ValueError("synthetic response setup failure")

    monkeypatch.setattr(router_module, "filter_end_to_end_headers", fail_headers)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        with pytest.raises(ValueError, match="synthetic response setup failure"):
            await client.post(
                "/v1/chat/completions",
                json={"model": "a", "messages": [], "stream": True},
            )

    assert app.state.controller_state._active_requests == {}


@pytest.mark.asyncio
async def test_stream_enter_failure_releases_transferred_tracker(tmp_path):
    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    app.state.controller_state.startup_reconciled = True

    @asynccontextmanager
    async def fail_stream(*_args, **_kwargs):
        raise VLLMClientError("synthetic stream enter failure")
        yield  # pragma: no cover - required to define an async generator

    app.state.vllm_client.proxy_stream = fail_stream
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "a", "messages": [], "stream": True},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "synthetic stream enter failure"
    assert app.state.controller_state._active_requests == {}


@pytest.mark.asyncio
async def test_stream_context_factory_failure_releases_transferred_tracker(tmp_path):
    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    app.state.controller_state.startup_reconciled = True

    def fail_stream_factory(*_args, **_kwargs):
        raise VLLMClientError("synthetic stream factory failure")

    app.state.vllm_client.proxy_stream = fail_stream_factory
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "a", "messages": [], "stream": True},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "synthetic stream factory failure"
    assert app.state.controller_state._active_requests == {}


@pytest.mark.asyncio
async def test_stream_enter_returning_none_still_exits_context_and_tracker(tmp_path):
    stream_exit_count = 0
    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    app.state.controller_state.startup_reconciled = True

    @asynccontextmanager
    async def none_stream(*_args, **_kwargs):
        nonlocal stream_exit_count
        try:
            yield None
        finally:
            stream_exit_count += 1

    app.state.vllm_client.proxy_stream = none_stream
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        with pytest.raises(AttributeError):
            await client.post(
                "/v1/chat/completions",
                json={"model": "a", "messages": [], "stream": True},
            )

    assert stream_exit_count == 1
    assert app.state.controller_state._active_requests == {}


@pytest.mark.asyncio
async def test_stream_body_cancellation_closes_upstream_and_releases_tracker(tmp_path, monkeypatch):
    started = asyncio.Event()
    block_body = asyncio.Event()
    upstream_exit_started = asyncio.Event()
    allow_upstream_exit = asyncio.Event()
    upstream_exit_count = 0
    metrics_count = 0

    class FakeUpstream:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            started.set()
            await block_body.wait()
            yield b"unused"

    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    app.state.controller_state.startup_reconciled = True

    def count_metrics(_self, _metrics):
        nonlocal metrics_count
        metrics_count += 1

    monkeypatch.setattr(MetricsRecorder, "record", count_metrics)

    @asynccontextmanager
    async def fake_stream(*_args, **_kwargs):
        nonlocal upstream_exit_count
        try:
            yield FakeUpstream()
        finally:
            upstream_exit_started.set()
            await allow_upstream_exit.wait()
            upstream_exit_count += 1

    app.state.vllm_client.proxy_stream = fake_stream
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        request_task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={"model": "a", "messages": [], "stream": True},
            )
        )
        await started.wait()
        request_task.cancel()
        await upstream_exit_started.wait()
        request_task.cancel()
        allow_upstream_exit.set()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    assert upstream_exit_count == 1
    assert metrics_count == 1
    assert app.state.controller_state._active_requests == {}


@pytest.mark.asyncio
async def test_default_switch_metrics_include_queue_and_drain_times(tmp_path, monkeypatch):
    release_a = asyncio.Event()
    a_started = asyncio.Event()
    recorded_metrics = []

    async def fake_json(model, *_args, **_kwargs):
        if model == "a":
            a_started.set()
            await release_a.wait()
        return 200, {"content-type": "application/json"}, b"{}"

    config = ControllerConfig.model_validate(
        {
            "models": {
                "a": {"backend_url": "http://a", "served_model_name": "a"},
                "b": {"backend_url": "http://b", "served_model_name": "b"},
            },
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    app.state.controller_state.startup_reconciled = True
    app.state.vllm_client.proxy_json = fake_json
    app.state.vllm_client.sleep = lambda *_args, **_kwargs: asyncio.sleep(0, result=0.01)
    app.state.vllm_client.wake_up = lambda *_args, **_kwargs: asyncio.sleep(0, result=0.02)
    app.state.vllm_client.wait_until_sleeping = lambda *_args, **_kwargs: asyncio.sleep(
        0, result=0.0
    )

    def capture(_self, metrics):
        recorded_metrics.append(metrics)

    monkeypatch.setattr(MetricsRecorder, "record", capture)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        request_a = asyncio.create_task(
            client.post("/v1/chat/completions", json={"model": "a", "messages": []})
        )
        await a_started.wait()
        request_b = asyncio.create_task(
            client.post("/v1/chat/completions", json={"model": "b", "messages": []})
        )
        await asyncio.sleep(0)
        release_a.set()
        await asyncio.gather(request_a, request_b)

    by_model = {metric.model: metric for metric in recorded_metrics}
    assert by_model["a"].route_class == "steady_resident"
    assert by_model["a"].switch_id is None
    assert by_model["b"].route_class == "switch_owner"
    assert by_model["b"].switch_id is not None
    assert by_model["b"].queue_wait_ms is not None
    assert by_model["b"].request_drain_ms is not None
    assert by_model["b"].request_drain_ms >= 0


@pytest.mark.asyncio
async def test_streaming_proxy_preserves_backend_status_and_model_alias(tmp_path, monkeypatch):
    backend = FastAPI()
    seen = {}
    recorded_metrics = []

    @backend.post("/v1/chat/completions")
    async def chat(body: dict):
        seen.update(body)
        return StreamingResponse(
            iter([b'{"error":"busy"}']),
            status_code=429,
            media_type="application/json",
            headers={"x-backend": "test"},
        )

    config = ControllerConfig.model_validate(
        {
            "models": {
                "alias": {
                    "backend_url": "http://backend",
                    "served_model_name": "real-model",
                }
            },
            "controller": {
                "startup_awake_model": "alias",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    controller_app = create_app(config)
    controller_app.state.controller_state.startup_reconciled = True
    controller_app.state.vllm_client._client._transport = ASGITransport(backend)

    def record_once(_self, metrics):
        recorded_metrics.append(metrics)

    monkeypatch.setattr(MetricsRecorder, "record", record_once)

    async with AsyncClient(
        transport=ASGITransport(controller_app), base_url="http://controller"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "alias", "messages": [], "stream": True},
        )

    assert response.status_code == 429
    assert response.headers["x-backend"] == "test"
    assert seen["model"] == "real-model"
    assert len(recorded_metrics) == 1
    assert recorded_metrics[0].status_code == 429


@pytest.mark.asyncio
async def test_stream_metrics_failure_does_not_fail_response_or_leak_tracker(tmp_path, monkeypatch):
    backend = FastAPI()

    @backend.post("/v1/chat/completions")
    async def chat():
        return StreamingResponse(iter([b"ok"]), media_type="text/event-stream")

    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    app.state.controller_state.startup_reconciled = True
    app.state.vllm_client._client._transport = ASGITransport(backend)

    def fail_record(_self, _metrics):
        raise OSError("synthetic metrics failure")

    monkeypatch.setattr(MetricsRecorder, "record", fail_record)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "a", "messages": [], "stream": True},
        )

    assert response.status_code == 200
    assert response.content == b"ok"
    assert app.state.controller_state._active_requests == {}


@pytest.mark.asyncio
async def test_json_metrics_failure_does_not_change_success_response(tmp_path, monkeypatch):
    backend = make_backend("a", [])
    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    app.state.controller_state.startup_reconciled = True
    app.state.vllm_client._client._transport = ASGITransport(backend)

    def fail_record(_self, _metrics):
        raise OSError("synthetic metrics failure")

    monkeypatch.setattr(MetricsRecorder, "record", fail_record)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "a", "messages": []},
        )

    assert response.status_code == 200
    assert app.state.controller_state._active_requests == {}


@pytest.mark.asyncio
async def test_metrics_failure_does_not_mask_unknown_model_404(tmp_path, monkeypatch):
    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {"metrics_path": str(tmp_path / "events.jsonl")},
        }
    )
    app = create_app(config)

    def fail_record(_self, _metrics):
        raise OSError("synthetic metrics failure")

    monkeypatch.setattr(MetricsRecorder, "record", fail_record)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "missing", "messages": []},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "unknown model: missing"


@pytest.mark.asyncio
async def test_metrics_failure_does_not_mask_upstream_exit_error(tmp_path, monkeypatch):
    class FakeUpstream:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            yield b"ok"

    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    app.state.controller_state.startup_reconciled = True

    @asynccontextmanager
    async def failing_exit_stream(*_args, **_kwargs):
        yield FakeUpstream()
        raise RuntimeError("synthetic upstream exit failure")

    def fail_record(_self, _metrics):
        raise OSError("synthetic metrics failure")

    app.state.vllm_client.proxy_stream = failing_exit_stream
    monkeypatch.setattr(MetricsRecorder, "record", fail_record)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        with pytest.raises(RuntimeError, match="synthetic upstream exit failure"):
            await client.post(
                "/v1/chat/completions",
                json={"model": "a", "messages": [], "stream": True},
            )

    assert app.state.controller_state._active_requests == {}


@pytest.mark.asyncio
async def test_tracker_exit_error_precedes_metrics_and_releases_reservation(tmp_path, monkeypatch):
    backend = FastAPI()
    metrics_count = 0

    @backend.post("/v1/chat/completions")
    async def chat():
        return StreamingResponse(iter([b"ok"]), media_type="text/event-stream")

    config = ControllerConfig.model_validate(
        {
            "models": {"a": {"backend_url": "http://backend", "served_model_name": "a"}},
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
            },
        }
    )
    app = create_app(config)
    state = app.state.controller_state
    state.startup_reconciled = True
    original_track_request = state.track_request
    app.state.vllm_client._client._transport = ASGITransport(backend)

    @asynccontextmanager
    async def failing_exit_tracker(model):
        tracker = original_track_request(model)
        await tracker.__aenter__()
        try:
            yield
        finally:
            await tracker.__aexit__(None, None, None)
            raise RuntimeError("synthetic tracker exit failure")

    def count_metrics(_self, _metrics):
        nonlocal metrics_count
        metrics_count += 1

    state.track_request = failing_exit_tracker
    monkeypatch.setattr(MetricsRecorder, "record", count_metrics)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://controller") as client:
        with pytest.raises(RuntimeError, match="synthetic tracker exit failure"):
            await client.post(
                "/v1/chat/completions",
                json={"model": "a", "messages": [], "stream": True},
            )

    assert metrics_count == 0
    assert state._active_requests == {}
