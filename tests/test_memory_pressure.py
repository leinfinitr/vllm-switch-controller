import pytest

from vllm_switch_controller.backup_pool import BackupPoolState
from vllm_switch_controller.config import ControllerConfig
from vllm_switch_controller.memory_pressure import MemoryPressureMonitor, SystemMemorySnapshot


def usage(
    state: BackupPoolState,
    *,
    client_id: str,
    model_id: str,
    evictable_bytes: int,
) -> None:
    state.report_usage(
        client_id=client_id,
        pid=1,
        engine="vllm",
        model_id=model_id,
        total_bytes=evictable_bytes,
        required_for_restore_bytes=0,
        cache_only_bytes=evictable_bytes,
        invalid_bytes=0,
        free_local_bytes=0,
    )


def test_memory_pressure_uses_hysteresis_and_model_priority():
    state = BackupPoolState(
        model_priorities={"cold": 0, "hot": 10},
    )
    usage(state, client_id="cold-client", model_id="cold", evictable_bytes=80)
    usage(state, client_id="hot-client", model_id="hot", evictable_bytes=80)

    snapshots = iter(
        [
            SystemMemorySnapshot(1000, 90, 1.0),
            SystemMemorySnapshot(1000, 90, 2.0),
            SystemMemorySnapshot(1000, 150, 3.0),
            SystemMemorySnapshot(1000, 250, 4.0),
        ]
    )
    monitor = MemoryPressureMonitor(
        state,
        reclaim_available_ratio=0.10,
        recovery_available_ratio=0.20,
        reclaim_available_bytes=0,
        recovery_available_bytes=0,
        poll_interval_s=0.5,
        consecutive_samples=2,
        reclaim_cooldown_s=0,
        probe=lambda: next(snapshots),
    )

    assert monitor.evaluate_once(now_monotonic=1.0) == {}
    assert monitor.snapshot()["state"] == "normal"
    assert monitor.evaluate_once(now_monotonic=2.0) == {
        "cold-client": 80,
        "hot-client": 30,
    }
    assert monitor.snapshot()["state"] == "reclaiming"

    # Between low and high watermarks, reclaiming remains active but existing
    # pending requests prevent duplicate requests.
    assert monitor.evaluate_once(now_monotonic=3.0) == {}
    assert monitor.snapshot()["state"] == "reclaiming"
    assert monitor.evaluate_once(now_monotonic=4.0) == {}
    assert monitor.snapshot()["state"] == "normal"


def test_memory_pressure_reports_unresolved_bytes_without_evictable_backups():
    monitor = MemoryPressureMonitor(
        BackupPoolState(),
        reclaim_available_ratio=0.10,
        recovery_available_ratio=0.20,
        reclaim_available_bytes=0,
        recovery_available_bytes=0,
        poll_interval_s=0.5,
        consecutive_samples=1,
        reclaim_cooldown_s=0,
        probe=lambda: SystemMemorySnapshot(1000, 50, 1.0),
    )

    assert monitor.evaluate_once(now_monotonic=1.0) == {}
    assert monitor.snapshot()["unresolved_pressure_bytes"] == 150


def test_memory_pressure_reclaims_explicitly_disk_backed_required_ram():
    state = BackupPoolState()
    state.report_usage(
        client_id="disk-client",
        model_id="cold",
        total_bytes=200,
        required_for_restore_bytes=200,
        cache_only_bytes=0,
        invalid_bytes=0,
        free_local_bytes=0,
        disk_backup_current_bytes=200,
        disk_backup_reserved_bytes=200,
        ram_reclaimable_with_disk_bytes=200,
    )
    monitor = MemoryPressureMonitor(
        state,
        reclaim_available_ratio=0.10,
        recovery_available_ratio=0.20,
        reclaim_available_bytes=0,
        recovery_available_bytes=0,
        poll_interval_s=0.5,
        consecutive_samples=1,
        reclaim_cooldown_s=0,
        probe=lambda: SystemMemorySnapshot(1000, 50, 1.0),
    )

    assert monitor.evaluate_once(now_monotonic=1.0) == {"disk-client": 150}
    assert monitor.snapshot()["unresolved_pressure_bytes"] == 0


def test_memory_pressure_config_rejects_reversed_watermarks():
    with pytest.raises(ValueError, match="recovery ratio"):
        ControllerConfig.model_validate(
            {
                "models": {
                    "a": {
                        "backend_url": "http://a",
                        "served_model_name": "a",
                    }
                },
                "controller": {
                    "cpu_memory_reclaim_available_ratio": 0.20,
                    "cpu_memory_recovery_available_ratio": 0.10,
                },
            }
        )


def test_memory_pressure_clears_current_error_after_probe_recovers():
    snapshots = iter(
        [
            RuntimeError("probe failed"),
            SystemMemorySnapshot(1000, 500, 2.0),
        ]
    )

    def probe():
        value = next(snapshots)
        if isinstance(value, Exception):
            raise value
        return value

    monitor = MemoryPressureMonitor(
        BackupPoolState(),
        reclaim_available_ratio=0.10,
        recovery_available_ratio=0.20,
        reclaim_available_bytes=0,
        recovery_available_bytes=0,
        poll_interval_s=0.5,
        consecutive_samples=1,
        reclaim_cooldown_s=0,
        probe=probe,
    )

    assert monitor.evaluate_once() == {}
    assert monitor.snapshot()["last_error"] is not None
    assert monitor.evaluate_once() == {}
    assert monitor.snapshot()["last_error"] is None
    assert monitor.snapshot()["probe_errors"] == 1
