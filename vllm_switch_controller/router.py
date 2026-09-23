import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from vllm_switch_controller.backup_pool import BackupPoolState
from vllm_switch_controller.config import ControllerConfig
from vllm_switch_controller.engine_client import (
    EngineClient,
    EngineControlError,
    filter_end_to_end_headers,
)
from vllm_switch_controller.memory_pressure import MemoryPressureMonitor
from vllm_switch_controller.metrics import RequestMetrics
from vllm_switch_controller.policies import SwitchingPolicy
from vllm_switch_controller.schemas import (
    CPU_BACKUP_CAPABILITIES,
    CPU_BACKUP_PROTOCOL_VERSION,
    BackupRegisterRequest,
    BackupReleaseRequest,
    BackupUsageRequest,
    OpenAIModel,
    OpenAIModelsResponse,
    PrewarmRequest,
)
from vllm_switch_controller.state import ControllerState, ModelState, UnknownModelError

logger = logging.getLogger(__name__)


class CleanupStreamingResponse(StreamingResponse):
    """Run async cleanup even if body iteration never starts or is cancelled."""

    def __init__(self, *args, cleanup: Callable[[], Awaitable[None]], **kwargs):
        super().__init__(*args, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._cleanup()


class CancellationResistantCleanup:
    """Run one async cleanup to completion before propagating cancellation."""

    def __init__(self, cleanup: Callable[[], Coroutine[Any, Any, None]]) -> None:
        self._cleanup = cleanup
        self._task: asyncio.Task[None] | None = None

    async def __call__(self) -> None:
        # Task creation is atomic until the first await on one event loop. Every
        # caller therefore observes the same teardown without a second lock that
        # could itself be interrupted before retrieving the cached task.
        if self._task is None:
            self._task = asyncio.create_task(self._cleanup())
        task = self._task
        cancellation = await wait_task_resisting_cancellation(task)
        # Retrieve teardown errors and avoid an unobserved task exception. A
        # cleanup failure takes precedence because ownership may remain unsafe.
        task.result()
        if cancellation is not None:
            raise cancellation


async def wait_task_resisting_cancellation(
    task: asyncio.Task[Any],
) -> asyncio.CancelledError | None:
    """Wait for task completion without forwarding caller cancellation to it."""
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
    return cancellation


def make_router(
    config: ControllerConfig,
    state: ControllerState,
    policy: SwitchingPolicy,
    vllm_client: EngineClient,
    metrics_recorder,
    backup_pool: BackupPoolState | None = None,
    memory_pressure: MemoryPressureMonitor | None = None,
) -> APIRouter:
    router = APIRouter()
    backup_pool = backup_pool or BackupPoolState()
    # Entries live only for their original TTL; a full cache rejects new hints.
    # Never evict an in-flight transaction to make room for advisory work.
    hints: dict[str, tuple[PrewarmRequest, float, asyncio.Task[dict[str, Any]]]] = {}

    def record_metrics_best_effort(metrics: RequestMetrics) -> None:
        try:
            metrics_recorder.record(metrics)
        except Exception:
            # Metrics are observability, not part of the backend state or proxy
            # result. Never replace the primary HTTP/control outcome with a
            # local JSONL write failure.
            logger.exception("failed to record request metrics")

    @router.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "active_model": state.active_model,
            "states": state.model_states,
            "active_requests": await state.active_requests_snapshot(),
        }

    @router.get("/admin/state")
    async def admin_state() -> dict[str, Any]:
        return {
            "active_model": state.active_model,
            "states": state.model_states,
            "active_requests": await state.active_requests_snapshot(),
        }

    @router.post("/admin/cpu-backup/register")
    async def cpu_backup_register(body: BackupRegisterRequest) -> dict[str, Any]:
        try:
            record = backup_pool.register_client(
                body.client_id,
                protocol_version=body.protocol_version,
                capabilities=body.capabilities,
                pid=body.pid,
                engine=body.engine,
                model_id=body.model_id,
                gpu_uuid=body.gpu_uuid,
                metadata=body.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "ok": True,
            "protocol_version": CPU_BACKUP_PROTOCOL_VERSION,
            "controller_capabilities": sorted(CPU_BACKUP_CAPABILITIES),
            "request_epoch": backup_pool.request_epoch,
            "client": record.snapshot(),
        }

    @router.post("/admin/cpu-backup/usage")
    async def cpu_backup_usage(body: BackupUsageRequest) -> dict[str, Any]:
        try:
            record = backup_pool.report_usage(
                client_id=body.client_id,
                protocol_version=body.protocol_version,
                capabilities=body.capabilities,
                pid=body.pid,
                engine=body.engine,
                model_id=body.model_id,
                gpu_uuid=body.gpu_uuid,
                total_bytes=body.total_bytes,
                released_bytes_total=body.released_bytes_total,
                required_for_restore_bytes=body.required_for_restore_bytes,
                cache_only_bytes=body.cache_only_bytes,
                invalid_bytes=body.invalid_bytes,
                free_local_bytes=body.free_local_bytes,
                disk_backup_current_bytes=body.disk_backup_current_bytes,
                disk_backup_reserved_bytes=body.disk_backup_reserved_bytes,
                ram_reclaimable_with_disk_bytes=(body.ram_reclaimable_with_disk_bytes),
                metadata=body.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        queued = backup_pool.maybe_enqueue_release_requests()
        return {
            "ok": True,
            "protocol_version": CPU_BACKUP_PROTOCOL_VERSION,
            "controller_capabilities": sorted(CPU_BACKUP_CAPABILITIES),
            "client": record.snapshot(),
            "queued_release_requests": queued,
        }

    @router.post("/admin/cpu-backup/release")
    async def cpu_backup_release(body: BackupReleaseRequest) -> dict[str, Any]:
        queued = backup_pool.request_release(body.client_id, body.target_free_bytes)
        return {"ok": True, "queued_bytes": queued}

    @router.get("/admin/cpu-backup/release-requests/{client_id}")
    async def cpu_backup_release_requests(client_id: str) -> dict[str, Any]:
        queued = backup_pool.maybe_enqueue_release_requests()
        requested_total, pending_bytes = backup_pool.release_request_snapshot(client_id)
        return {
            "ok": True,
            "protocol_version": CPU_BACKUP_PROTOCOL_VERSION,
            "controller_capabilities": sorted(CPU_BACKUP_CAPABILITIES),
            "request_epoch": backup_pool.request_epoch,
            "requested_release_bytes_total": requested_total,
            "pending_release_bytes": pending_bytes,
            "queued_release_requests": queued,
        }

    @router.get("/admin/cpu-backup/stats")
    async def cpu_backup_stats() -> dict[str, Any]:
        pressure = memory_pressure.snapshot() if memory_pressure is not None else None
        return {"ok": True, **backup_pool.snapshot(), "memory_pressure": pressure}

    @router.post("/admin/switch/{model}")
    async def admin_switch(model: str) -> dict[str, Any]:
        metrics = RequestMetrics.new(model=model, path=f"/admin/switch/{model}")
        await ensure_model_ready(model, metrics)
        record_metrics_best_effort(metrics)
        return {"active_model": state.active_model, "states": state.model_states}

    @router.post("/admin/prewarm")
    async def prewarm(body: PrewarmRequest) -> dict[str, Any]:
        received = time.perf_counter()
        if body.model not in state.model_states:
            raise HTTPException(status_code=404, detail=f"unknown model: {body.model}")
        for key, (_, expiry, task) in list(hints.items()):
            if expiry <= received and task.done():
                del hints[key]
        existing = hints.get(body.hint_id)
        if existing is not None:
            original, _, task = existing
            if original != body:
                raise HTTPException(
                    status_code=409, detail="hint_id already has different contents"
                )
            cancellation = await wait_task_resisting_cancellation(task)
            result = task.result()
            if cancellation is not None:
                raise cancellation
            return {**result, "duplicate": True}

        metrics = RequestMetrics.new(model=body.model, path="/admin/prewarm")
        metrics.task_id = body.task_id
        metrics.stage_id = body.stage_id
        metrics.hint_id = body.hint_id
        metrics.hint_source = body.source

        def finish(status: str, reason: str | None = None) -> dict[str, Any]:
            metrics.hint_status = status
            metrics.hint_reason = reason
            metrics.e2e_latency_ms = (time.perf_counter() - received) * 1000
            metrics.status_code = 200
            record_metrics_best_effort(metrics)
            return {
                "status": status,
                "reason": reason,
                "hint_id": body.hint_id,
                "model": body.model,
                "active_model": state.active_model,
                "elapsed_ms": metrics.e2e_latency_ms,
                "switch_id": metrics.switch_id,
                "duplicate": False,
            }

        expiry = received + body.ttl_ms / 1000

        async def execute() -> dict[str, Any]:
            # An unlocked asyncio.Lock is acquired without yielding. Recheck
            # demand and reservations inside it before starting any lifecycle IO.
            if state.switch_lock.locked() or state.pending_demands:
                return finish("ignored", "busy")
            async with state.switch_lock:
                active = await state.active_requests_snapshot()
                if state.pending_demands or any(active.values()):
                    return finish("ignored", "busy")
                if time.perf_counter() >= expiry:
                    return finish("ignored", "expired")
                if memory_pressure is not None and memory_pressure.enabled:
                    if memory_pressure.state != "normal" or memory_pressure.last_error:
                        return finish("ignored", "memory_pressure")
                try:
                    await ensure_model_ready_locked(body.model, metrics)
                except Exception as exc:
                    metrics.error = f"{type(exc).__name__}: {exc}"
                    return finish("failed", "lifecycle_error")
                return finish("ready")

        if len(hints) >= 1024:
            return finish("ignored", "capacity")
        task = asyncio.create_task(execute())
        hints[body.hint_id] = (body, expiry, task)
        # Caller cancellation must not release switch_lock while a mutating
        # sleep/wake request can still finish remotely.
        cancellation = await wait_task_resisting_cancellation(task)
        result = task.result()
        if cancellation is not None:
            raise cancellation
        return result

    @router.get("/v1/models")
    async def list_models() -> OpenAIModelsResponse:
        return OpenAIModelsResponse(data=[OpenAIModel(id=name) for name in config.models])

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await handle_openai_proxy(request, "/v1/chat/completions")

    @router.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await handle_openai_proxy(request, "/v1/completions")

    async def handle_openai_proxy(request: Request, path: str) -> Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(
                status_code=400,
                detail="request body must be valid JSON",
            ) from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request JSON must be an object")
        target_model = body.get("model")
        if not isinstance(target_model, str):
            raise HTTPException(
                status_code=400,
                detail="request JSON must contain string field 'model'",
            )

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        metrics = RequestMetrics.new(model=target_model, path=path, request_id=request_id)
        forwarded_headers = dict(request.headers)
        forwarded_headers["x-request-id"] = request_id
        request_start = time.perf_counter()
        request_tracker = None
        tracker_cleanup = None
        try:
            # Validate before constructing the async context manager. If
            # validation failed after construction but before __aenter__, the
            # finally block could call __aexit__ on an unentered generator and
            # mask the intended 404 response.
            state.require_model(target_model)
            # Reserve the request before releasing switch_lock. Otherwise a
            # competing model request can begin sleeping this backend in the
            # gap between readiness and track_request().
            queue_started = time.perf_counter()
            with state.demand():
                await state.switch_lock.acquire()
            try:
                metrics.queue_wait_ms = (time.perf_counter() - queue_started) * 1000
                await ensure_model_ready_locked(target_model, metrics)
                request_tracker = state.track_request(target_model)
                enter_task = asyncio.create_task(request_tracker.__aenter__())
                enter_cancellation = await wait_task_resisting_cancellation(enter_task)
                # An enter failure takes precedence; by the async-context-manager
                # contract it did not transfer ownership to the caller.
                enter_task.result()

                async def exit_request_tracker(tracker=request_tracker) -> None:
                    await tracker.__aexit__(None, None, None)

                tracker_cleanup = CancellationResistantCleanup(exit_request_tracker)
                if enter_cancellation is not None:
                    await tracker_cleanup()
                    raise enter_cancellation
            finally:
                state.switch_lock.release()
            backend_start = time.perf_counter()
            if body.get("stream") is True:
                # Transfer ownership before awaiting stream setup. This avoids
                # a cancellation gap where both caller and response believe
                # they own the reservation, or neither closes the upstream.
                stream_tracker_cleanup = tracker_cleanup
                tracker_cleanup = None
                response = await stream_response(
                    target_model,
                    path,
                    body,
                    request,
                    forwarded_headers,
                    metrics,
                    request_start,
                    backend_start,
                    stream_tracker_cleanup,
                )
                # The streaming iterator owns the request reservation until the
                # upstream body is consumed or the downstream disconnects.
                request_tracker = None
                return response

            try:
                status, headers, content = await vllm_client.proxy_json(
                    target_model, path, body, headers=forwarded_headers
                )
            finally:
                await tracker_cleanup()
                tracker_cleanup = None
                request_tracker = None
            metrics.status_code = status
            metrics.e2e_latency_ms = (time.perf_counter() - request_start) * 1000
            record_metrics_best_effort(metrics)
            return Response(
                content=content,
                status_code=status,
                headers=filter_end_to_end_headers(headers, rebuilding_body=True),
                media_type=headers.get("content-type", "application/json"),
            )
        except (UnknownModelError, KeyError) as exc:
            metrics.error = str(exc)
            metrics.status_code = 404
            record_metrics_best_effort(metrics)
            raise HTTPException(status_code=404, detail=f"unknown model: {target_model}") from exc
        except EngineControlError as exc:
            metrics.error = str(exc)
            metrics.status_code = 502
            record_metrics_best_effort(metrics)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            if tracker_cleanup is not None:
                await tracker_cleanup()

    async def ensure_model_ready(target_model: str, metrics: RequestMetrics) -> None:
        state.require_model(target_model)
        with state.demand():
            await state.switch_lock.acquire()
        try:
            await ensure_model_ready_locked(target_model, metrics)
        finally:
            state.switch_lock.release()

    async def ensure_model_ready_locked(target_model: str, metrics: RequestMetrics) -> None:
        """Transition models while the caller holds state.switch_lock."""
        state.require_model(target_model)
        switch_deadline = time.perf_counter() + config.controller.switch_timeout_s

        def remaining_switch_s() -> float:
            remaining = switch_deadline - time.perf_counter()
            if remaining <= 0:
                raise EngineControlError("model switch exceeded the configured end-to-end timeout")
            return remaining

        async def observe_sleeping(model: str) -> bool:
            remaining = remaining_switch_s()
            task = asyncio.create_task(vllm_client.is_sleeping(model, timeout_s=remaining))
            done, _pending = await asyncio.wait({task}, timeout=remaining)
            if task in done:
                return task.result()
            # A deliberately non-cooperative probe can suppress cancellation.
            # Do not await it here: is_sleeping is read-only, and waiting would
            # let a broken transport extend the externally visible deadline.
            task.cancel()

            def consume_probe_result(completed: asyncio.Task[bool]) -> None:
                try:
                    completed.result()
                except (BaseException, asyncio.CancelledError):
                    pass

            task.add_done_callback(consume_probe_result)
            raise EngineControlError("model switch exceeded the configured end-to-end timeout")

        # The configured startup state is a desired launcher contract, not a
        # backend observation. Reconcile the full configured pool before the
        # first route, and reject multiple observed-awake engines.
        if not state.startup_reconciled:
            observed_awake = []
            for model in sorted(state.model_states):
                sleeping = await observe_sleeping(model)
                if sleeping:
                    state.mark_sleeping(model)
                else:
                    state.mark_awake(model)
                    observed_awake.append(model)
            if len(observed_awake) > 1:
                for model in observed_awake:
                    state.mark_error(model)
                raise EngineControlError(
                    "multiple awake backends observed during startup reconciliation: "
                    + ", ".join(observed_awake)
                )
            state.startup_reconciled = True
        unsafe_models = [
            model
            for model, model_state in state.model_states.items()
            if model_state
            in {
                ModelState.ERROR,
                ModelState.UNKNOWN,
                ModelState.WAKING,
                ModelState.SLEEPING_IN_PROGRESS,
            }
        ]
        if unsafe_models:
            observed_awake = []
            for model in sorted(unsafe_models):
                sleeping = await observe_sleeping(model)
                if sleeping:
                    state.mark_sleeping(model)
                else:
                    state.mark_awake(model)
                    observed_awake.append(model)
            if len(observed_awake) > 1:
                for model in observed_awake:
                    state.mark_error(model)
                raise EngineControlError(
                    "multiple awake backends observed during lifecycle reconciliation: "
                    + ", ".join(observed_awake)
                )
        decision = policy.decide(state.active_model, target_model, state.model_states)
        metrics.previous_model = state.active_model
        metrics.switch_needed = bool(decision.sleep_models or decision.wake_model)
        if not metrics.switch_needed:
            return

        metrics.route_class = "switch_owner"
        metrics.switch_id = str(uuid.uuid4())

        if decision.wait_for_active_requests:
            drain_start = time.perf_counter()
            try:
                async with asyncio.timeout(remaining_switch_s()):
                    await state.wait_for_other_model_requests_to_finish(target_model)
            except TimeoutError as exc:
                raise EngineControlError(
                    "timed out draining active requests before model switch"
                ) from exc
            metrics.request_drain_ms = (time.perf_counter() - drain_start) * 1000
        switch_start = time.perf_counter()
        sleep_total = 0.0
        wake_total = 0.0
        try:
            for model in decision.sleep_models:
                state.mark_sleeping_in_progress(model)
                try:
                    remaining = remaining_switch_s()
                    async with asyncio.timeout(remaining):
                        latency, _ = await vllm_client.sleep_and_wait_with_timeout(
                            model,
                            config.models[model].sleep_level,
                            remaining,
                        )
                except BaseException as exc:
                    completed_latency = getattr(exc, "transition_latency_s", None)
                    if isinstance(completed_latency, (int, float)):
                        sleep_total += completed_latency
                        metrics.sleep_latency_ms = sleep_total * 1000
                    state.mark_error(model)
                    if isinstance(exc, TimeoutError):
                        raise EngineControlError(
                            "model switch exceeded the configured end-to-end timeout"
                        ) from exc
                    raise
                sleep_total += latency
                metrics.sleep_latency_ms = sleep_total * 1000
                state.mark_sleeping(model)
            if decision.wake_model is not None:
                state.mark_waking(decision.wake_model)
                spec = config.models[decision.wake_model]
                try:
                    remaining = remaining_switch_s()
                    async with asyncio.timeout(remaining):
                        wake_total, _ = await vllm_client.wake_up_and_wait_with_timeout(
                            decision.wake_model, spec.wake_tags, remaining
                        )
                except BaseException as exc:
                    completed_latency = getattr(exc, "transition_latency_s", None)
                    if isinstance(completed_latency, (int, float)):
                        wake_total += completed_latency
                        metrics.wake_latency_ms = wake_total * 1000
                    state.mark_error(decision.wake_model)
                    if isinstance(exc, TimeoutError):
                        raise EngineControlError(
                            "model switch exceeded the configured end-to-end timeout"
                        ) from exc
                    raise
                metrics.wake_latency_ms = wake_total * 1000
                state.mark_awake(decision.wake_model)
        finally:
            metrics.switch_latency_ms = (time.perf_counter() - switch_start) * 1000

    async def stream_response(
        target_model: str,
        path: str,
        body: dict[str, Any],
        request: Request,
        forwarded_headers: dict[str, str],
        metrics: RequestMetrics,
        request_start: float,
        backend_start: float,
        request_tracker_cleanup: CancellationResistantCleanup,
    ) -> StreamingResponse:
        stream_context = None
        upstream = None
        stream_entered = False
        record_metrics = False
        status_code: int | None = None

        async def close_resources() -> None:
            try:
                if stream_entered and stream_context is not None:
                    await stream_context.__aexit__(None, None, None)
            finally:
                await request_tracker_cleanup()
                if record_metrics and status_code is not None:
                    metrics.status_code = status_code
                    metrics.e2e_latency_ms = (time.perf_counter() - request_start) * 1000
                    record_metrics_best_effort(metrics)

        cleanup = CancellationResistantCleanup(close_resources)

        try:
            # Context construction itself may synchronously reject the request.
            # Ownership has already moved here, so it must be inside the same
            # cleanup boundary as async enter and response setup.
            stream_context = vllm_client.proxy_stream(
                target_model, path, body, headers=forwarded_headers
            )
            upstream = await stream_context.__aenter__()
            stream_entered = True
            status_code = upstream.status_code
            response_headers = filter_end_to_end_headers(upstream.headers, rebuilding_body=True)

            async def iterator():
                first_chunk_seen = False
                try:
                    async for chunk in upstream.aiter_bytes():
                        if chunk and not first_chunk_seen:
                            first_chunk_seen = True
                            first_chunk_ts = time.perf_counter()
                            metrics.response_body_first_byte_ms = (
                                first_chunk_ts - backend_start
                            ) * 1000
                            metrics.e2e_response_body_first_byte_ms = (
                                first_chunk_ts - request_start
                            ) * 1000
                        yield chunk
                except BaseException as exc:
                    metrics.error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    await cleanup()

            response = CleanupStreamingResponse(
                iterator(),
                status_code=status_code,
                headers=response_headers,
                media_type=upstream.headers.get("content-type", "text/event-stream"),
                cleanup=cleanup,
            )
            record_metrics = True
            return response
        except BaseException:
            await cleanup()
            raise

    return router
