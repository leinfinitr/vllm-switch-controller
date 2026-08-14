import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from vllm_switch_controller.backup_pool import BackupPoolState
from vllm_switch_controller.config import ControllerConfig
from vllm_switch_controller.main import create_app
from vllm_switch_controller.schemas import BackupUsageRequest


def test_backup_pool_state_tracks_aggregate_usage_and_release_bytes():
    state = BackupPoolState(global_cap_bytes=2048)
    state.report_usage(
        client_id="client-a",
        pid=123,
        engine="vllm",
        model_id="model-a",
        total_bytes=4096,
        required_for_restore_bytes=1024,
        cache_only_bytes=2048,
        invalid_bytes=512,
        free_local_bytes=512,
    )

    stats = state.stats()
    assert stats["client_count"] == 1
    assert stats["total_bytes"] == 4096
    assert stats["required_for_restore_bytes"] == 1024
    assert stats["evictable_bytes"] == 3072
    assert stats["over_cap_bytes"] == 2048

    queued = state.maybe_enqueue_release_requests()
    assert queued == {"client-a": 2048}
    assert state.release_request_snapshot("client-a") == (2048, 2048)

    state.report_usage(
        client_id="client-a",
        pid=123,
        engine="vllm",
        model_id="model-a",
        total_bytes=3072,
        required_for_restore_bytes=512,
        cache_only_bytes=1536,
        invalid_bytes=512,
        free_local_bytes=512,
    )
    assert state.stats()["pending_release_bytes"] == 1024
    assert state.maybe_enqueue_release_requests() == {}


def test_process_incarnation_rejects_pid_reuse_for_same_client_id():
    state = BackupPoolState()
    state.register_client("worker-incarnation", pid=100, model_id="model-a")

    with pytest.raises(ValueError, match="process incarnation"):
        state.register_client("worker-incarnation", pid=101, model_id="model-a")


@pytest.mark.asyncio
async def test_protocol_metadata_and_incarnation_conflicts_are_explicit(tmp_path):
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
        registered = await client.post(
            "/admin/cpu-backup/register",
            json={
                "protocol_version": 1,
                "capabilities": [
                    "cumulative-release-v1",
                    "process-incarnation-v1",
                    "released-bytes-total-v1",
                ],
                "client_id": "worker-incarnation",
                "pid": 100,
                "engine": "vllm",
                "model_id": "a",
            },
        )
        conflict = await client.post(
            "/admin/cpu-backup/register",
            json={
                "protocol_version": 1,
                "capabilities": [
                    "cumulative-release-v1",
                    "process-incarnation-v1",
                    "released-bytes-total-v1",
                ],
                "client_id": "worker-incarnation",
                "pid": 101,
                "engine": "vllm",
                "model_id": "a",
            },
        )
        unsupported = await client.post(
            "/admin/cpu-backup/register",
            json={
                "protocol_version": 2,
                "client_id": "future-worker",
                "pid": 102,
            },
        )

    payload = registered.json()
    assert registered.status_code == 200
    assert payload["protocol_version"] == 1
    assert "process-incarnation-v1" in payload["controller_capabilities"]
    assert conflict.status_code == 409
    assert "process incarnation" in conflict.json()["detail"]
    assert unsupported.status_code == 422


def test_exact_disk_accounting_requires_declared_capability():
    with pytest.raises(ValidationError, match="exact-disk-accounting-v1"):
        BackupUsageRequest.model_validate(
            {
                "protocol_version": 1,
                "capabilities": [
                    "cumulative-release-v1",
                    "process-incarnation-v1",
                    "released-bytes-total-v1",
                ],
                "client_id": "worker-incarnation",
                "total_bytes": 1024,
                "released_bytes_total": 0,
                "required_for_restore_bytes": 1024,
                "disk_backup_current_bytes": 1024,
            }
        )


def test_hard_cap_reclaims_all_evictable_bytes_when_required_exceeds_cap():
    state = BackupPoolState(global_cap_bytes=4096)
    state.report_usage(
        client_id="client-a",
        pid=1,
        engine="vllm",
        model_id="a",
        total_bytes=10_240,
        required_for_restore_bytes=5120,
        cache_only_bytes=5120,
        invalid_bytes=0,
        free_local_bytes=0,
    )

    # The cap applies to total backup usage. Required storage is protected, so
    # the best feasible action is to release every evictable byte.
    assert state.maybe_enqueue_release_requests() == {"client-a": 5120}
    assert state.stats()["over_cap_bytes"] == 1024


def test_disk_backed_required_bytes_participate_in_aggregate_reclaim():
    state = BackupPoolState()
    state.report_usage(
        client_id="disk-current",
        pid=1,
        engine="vllm",
        model_id="cold-model",
        total_bytes=4096,
        required_for_restore_bytes=4096,
        cache_only_bytes=0,
        invalid_bytes=0,
        free_local_bytes=0,
        disk_backup_current_bytes=4096,
        disk_backup_reserved_bytes=8192,
        ram_reclaimable_with_disk_bytes=4096,
    )
    state.report_usage(
        client_id="disk-reserved",
        pid=2,
        engine="vllm",
        model_id="cold-model",
        total_bytes=2048,
        required_for_restore_bytes=2048,
        cache_only_bytes=0,
        invalid_bytes=0,
        free_local_bytes=0,
        disk_backup_current_bytes=0,
        disk_backup_reserved_bytes=2048,
        ram_reclaimable_with_disk_bytes=2048,
    )

    queued = state.request_release_bytes(5120)

    assert queued == {"disk-current": 4096, "disk-reserved": 1024}
    stats = state.stats()
    assert stats["required_for_restore_bytes"] == 6144
    assert stats["ram_reclaimable_with_disk_bytes"] == 6144
    assert stats["evictable_bytes"] == 6144
    assert stats["disk_backup_current_bytes"] == 4096
    assert stats["disk_backup_reserved_bytes"] == 10_240
    assert stats["disk_backup_client_count"] == 2


def test_disk_configuration_without_reported_reclaimability_cannot_release_required_bytes():
    state = BackupPoolState()
    state.report_usage(
        client_id="configured-only",
        total_bytes=4096,
        required_for_restore_bytes=4096,
        cache_only_bytes=0,
        invalid_bytes=0,
        free_local_bytes=0,
        disk_backup_current_bytes=0,
        disk_backup_reserved_bytes=8192,
        ram_reclaimable_with_disk_bytes=0,
    )

    assert state.request_release_bytes(4096) == {}
    assert state.stats()["evictable_bytes"] == 0


def test_disk_reclaimable_bytes_must_be_a_subset_of_required_ram():
    with pytest.raises(ValidationError, match="cannot exceed"):
        BackupUsageRequest.model_validate(
            {
                "protocol_version": 1,
                "capabilities": [
                    "cumulative-release-v1",
                    "exact-disk-accounting-v1",
                    "process-incarnation-v1",
                    "released-bytes-total-v1",
                ],
                "client_id": "invalid",
                "released_bytes_total": 0,
                "total_bytes": 1024,
                "required_for_restore_bytes": 0,
                "cache_only_bytes": 1024,
                "ram_reclaimable_with_disk_bytes": 1,
            }
        )


def test_disk_reclaimable_bytes_must_have_a_reported_disk_source():
    with pytest.raises(ValidationError, match="reported disk source"):
        BackupUsageRequest.model_validate(
            {
                "protocol_version": 1,
                "capabilities": [
                    "cumulative-release-v1",
                    "exact-disk-accounting-v1",
                    "process-incarnation-v1",
                    "released-bytes-total-v1",
                ],
                "client_id": "invalid",
                "released_bytes_total": 0,
                "total_bytes": 1024,
                "required_for_restore_bytes": 1024,
                "ram_reclaimable_with_disk_bytes": 1024,
            }
        )


def test_backup_pool_rejects_disk_reclaim_without_reported_source():
    state = BackupPoolState()

    with pytest.raises(ValueError, match="reported disk source"):
        state.report_usage(
            client_id="invalid",
            total_bytes=1024,
            required_for_restore_bytes=1024,
            cache_only_bytes=0,
            invalid_bytes=0,
            free_local_bytes=0,
            ram_reclaimable_with_disk_bytes=1024,
        )


def test_disk_reclaim_uses_priority_before_age_and_size():
    state = BackupPoolState(model_priorities={"cold": 0, "hot": 10})
    for client_id, model_id, size_bytes in (
        ("cold-client", "cold", 1024),
        ("hot-client", "hot", 4096),
    ):
        state.report_usage(
            client_id=client_id,
            model_id=model_id,
            total_bytes=size_bytes,
            required_for_restore_bytes=size_bytes,
            cache_only_bytes=0,
            invalid_bytes=0,
            free_local_bytes=0,
            disk_backup_current_bytes=size_bytes,
            disk_backup_reserved_bytes=size_bytes,
            ram_reclaimable_with_disk_bytes=size_bytes,
        )
    state.clients["hot-client"].updated_at = 1.0
    state.clients["cold-client"].updated_at = 2.0

    assert state.request_release_bytes(1024) == {"cold-client": 1024}


def test_disk_reclaim_pending_survives_required_source_transition_until_ack():
    state = BackupPoolState()
    usage = {
        "client_id": "client-a",
        "total_bytes": 1024,
        "required_for_restore_bytes": 1024,
        "cache_only_bytes": 0,
        "invalid_bytes": 0,
        "free_local_bytes": 0,
        "disk_backup_current_bytes": 1024,
        "disk_backup_reserved_bytes": 1024,
    }
    state.report_usage(
        **usage,
        ram_reclaimable_with_disk_bytes=1024,
        released_bytes_total=0,
    )
    assert state.request_release("client-a", 1024) == 1024

    state.report_usage(
        **usage,
        ram_reclaimable_with_disk_bytes=0,
        released_bytes_total=0,
    )
    assert state.stats()["pending_release_bytes"] == 1024
    assert state.request_release_bytes(1024) == {}

    state.report_usage(
        **usage,
        ram_reclaimable_with_disk_bytes=1024,
        released_bytes_total=1024,
    )
    assert state.stats()["pending_release_bytes"] == 0


def test_backup_pool_release_prefers_lower_model_priority():
    state = BackupPoolState(
        global_cap_bytes=4096,
        model_priorities={"cold-model": 0, "hot-model": 10},
    )
    state.report_usage(
        client_id="client-a",
        pid=123,
        engine="vllm",
        model_id="cold-model",
        total_bytes=1024,
        required_for_restore_bytes=0,
        cache_only_bytes=1024,
        invalid_bytes=0,
        free_local_bytes=0,
    )
    state.report_usage(
        client_id="client-b",
        pid=124,
        engine="vllm",
        model_id="hot-model",
        total_bytes=4096,
        required_for_restore_bytes=0,
        cache_only_bytes=4096,
        invalid_bytes=0,
        free_local_bytes=0,
    )
    state.clients["client-b"].updated_at = 1.0
    state.clients["client-a"].updated_at = 2.0

    queued = state.maybe_enqueue_release_requests()

    assert queued == {"client-a": 1024}
    assert state.release_request_snapshot("client-a") == (1024, 1024)
    assert state.release_request_snapshot("client-b") == (0, 0)


def test_pending_release_survives_non_evictable_state_transition():
    state = BackupPoolState()
    usage = {
        "client_id": "client-a",
        "total_bytes": 1024,
        "invalid_bytes": 0,
        "free_local_bytes": 0,
    }
    state.report_usage(
        **usage,
        required_for_restore_bytes=0,
        cache_only_bytes=1024,
    )
    assert state.request_release("client-a", 1024) == 1024
    assert state.release_request_snapshot("client-a") == (1024, 1024)
    # GET is idempotent; losing a response does not consume the command.
    assert state.release_request_snapshot("client-a") == (1024, 1024)

    # Wake makes the same storage temporarily non-evictable. This is not a
    # physical release acknowledgement and must not cancel/reissue the request.
    state.report_usage(
        **usage,
        required_for_restore_bytes=1024,
        cache_only_bytes=0,
    )
    assert state.stats()["pending_release_bytes"] == 1024
    state.report_usage(
        **usage,
        required_for_restore_bytes=0,
        cache_only_bytes=1024,
    )
    assert state.request_release_bytes(1024) == {}

    # This legacy-client path has no cumulative counter, so an observed
    # total_bytes drop is its compatibility acknowledgement.
    state.report_usage(
        client_id="client-a",
        total_bytes=0,
        required_for_restore_bytes=0,
        cache_only_bytes=0,
        invalid_bytes=0,
        free_local_bytes=0,
    )
    assert state.stats()["pending_release_bytes"] == 0


def test_release_counter_ack_survives_latest_wins_reallocation():
    state = BackupPoolState()
    usage = {
        "client_id": "client-a",
        "total_bytes": 1024,
        "required_for_restore_bytes": 0,
        "cache_only_bytes": 1024,
        "invalid_bytes": 0,
        "free_local_bytes": 0,
    }
    state.report_usage(**usage, released_bytes_total=0)
    assert state.request_release("client-a", 1024) == 1024

    # The worker released 1 KiB and immediately reused newly allocated 1 KiB.
    # A latest-wins footprint is unchanged, but the monotonic counter preserves
    # the real allocator release acknowledgement.
    state.report_usage(**usage, released_bytes_total=1024)
    assert state.stats()["pending_release_bytes"] == 0

    with pytest.raises(ValueError, match="must be monotonic"):
        state.report_usage(**usage, released_bytes_total=512)

    with pytest.raises(ValueError, match="presence must remain stable"):
        state.report_usage(**usage)


@pytest.mark.asyncio
async def test_cpu_backup_admin_api_records_aggregate_usage(tmp_path):
    config = ControllerConfig.model_validate(
        {
            "models": {
                "a": {"backend_url": "http://a", "served_model_name": "a"},
            },
            "controller": {
                "startup_awake_model": "a",
                "metrics_path": str(tmp_path / "events.jsonl"),
                "cpu_backup_global_cap_bytes": 0,
                "cpu_backup_default_model_priority": 1,
                "cpu_backup_model_priorities": {"model-a": 5},
            },
        }
    )
    app = create_app(config)

    async with AsyncClient(
        transport=ASGITransport(app),
        base_url="http://controller",
    ) as client:
        response = await client.post(
            "/admin/cpu-backup/register",
            json={
                "protocol_version": 1,
                "capabilities": [
                    "cumulative-release-v1",
                    "exact-disk-accounting-v1",
                    "process-incarnation-v1",
                    "released-bytes-total-v1",
                ],
                "client_id": "client-a",
                "pid": 123,
                "engine": "vllm",
                "model_id": "model-a",
            },
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True

        response = await client.post(
            "/admin/cpu-backup/usage",
            json={
                "client_id": "client-a",
                "pid": 123,
                "engine": "vllm",
                "model_id": "model-a",
                "protocol_version": 1,
                "capabilities": [
                    "cumulative-release-v1",
                    "exact-disk-accounting-v1",
                    "process-incarnation-v1",
                    "released-bytes-total-v1",
                ],
                "total_bytes": 5120,
                "released_bytes_total": 0,
                "required_for_restore_bytes": 1024,
                "cache_only_bytes": 4096,
                "invalid_bytes": 0,
                "free_local_bytes": 0,
                "disk_backup_current_bytes": 1024,
                "disk_backup_reserved_bytes": 2048,
                "ram_reclaimable_with_disk_bytes": 1024,
            },
        )
        assert response.status_code == 200
        assert response.json()["queued_release_requests"] == {"client-a": 5120}

        stats = (await client.get("/admin/cpu-backup/stats")).json()
        assert stats["ok"] is True
        assert stats["stats"]["model_priorities"] == {"model-a": 5}
        assert stats["stats"]["default_model_priority"] == 1
        assert stats["stats"]["required_for_restore_bytes"] == 1024
        assert stats["stats"]["ram_reclaimable_with_disk_bytes"] == 1024
        assert stats["stats"]["evictable_bytes"] == 5120
        assert stats["stats"]["disk_backup_current_bytes"] == 1024
        assert stats["stats"]["disk_backup_reserved_bytes"] == 2048
        assert stats["stats"]["disk_backup_client_count"] == 1
        assert "backups" not in stats

        release_request = (await client.get("/admin/cpu-backup/release-requests/client-a")).json()
        assert release_request["ok"] is True
        assert release_request["requested_release_bytes_total"] == 5120
        assert release_request["pending_release_bytes"] == 5120

        response = await client.post(
            "/admin/cpu-backup/usage",
            json={
                "client_id": "client-a",
                "pid": 123,
                "engine": "vllm",
                "model_id": "model-a",
                "protocol_version": 1,
                "capabilities": [
                    "cumulative-release-v1",
                    "exact-disk-accounting-v1",
                    "process-incarnation-v1",
                    "released-bytes-total-v1",
                ],
                "total_bytes": 1024,
                "released_bytes_total": 5120,
                "required_for_restore_bytes": 1024,
                "cache_only_bytes": 0,
                "invalid_bytes": 0,
                "free_local_bytes": 0,
                "disk_backup_current_bytes": 1024,
                "disk_backup_reserved_bytes": 2048,
                "ram_reclaimable_with_disk_bytes": 0,
            },
        )
        assert response.status_code == 200
        stats = (await client.get("/admin/cpu-backup/stats")).json()
        assert stats["stats"]["evictable_bytes"] == 0

        stale = await client.post(
            "/admin/cpu-backup/usage",
            json={
                "protocol_version": 1,
                "capabilities": [
                    "cumulative-release-v1",
                    "process-incarnation-v1",
                    "released-bytes-total-v1",
                ],
                "client_id": "client-a",
                "total_bytes": 1024,
                "released_bytes_total": 2048,
                "required_for_restore_bytes": 1024,
                "cache_only_bytes": 0,
                "invalid_bytes": 0,
                "free_local_bytes": 0,
            },
        )
        assert stale.status_code == 409
