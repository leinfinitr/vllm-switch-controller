from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from vllm_switch_controller.backup_pool import BackupPoolState


@dataclass(frozen=True)
class SystemMemorySnapshot:
    total_bytes: int
    available_bytes: int
    captured_at: float

    @property
    def available_ratio(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return self.available_bytes / self.total_bytes


def read_system_memory() -> SystemMemorySnapshot:
    """Read the kernel's estimate of memory usable without swapping."""
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, raw = line.split(":", 1)
        fields = raw.split()
        if fields:
            values[name] = int(fields[0]) * 1024
    return SystemMemorySnapshot(
        total_bytes=values["MemTotal"],
        available_bytes=values["MemAvailable"],
        captured_at=time.time(),
    )


class MemoryPressureMonitor:
    """Translate host MemAvailable pressure into aggregate release requests."""

    def __init__(
        self,
        backup_pool: BackupPoolState,
        *,
        reclaim_available_ratio: float | None,
        recovery_available_ratio: float | None,
        reclaim_available_bytes: int,
        recovery_available_bytes: int,
        poll_interval_s: float,
        consecutive_samples: int,
        reclaim_cooldown_s: float,
        probe: Callable[[], SystemMemorySnapshot] = read_system_memory,
    ) -> None:
        self.backup_pool = backup_pool
        self.reclaim_available_ratio = reclaim_available_ratio
        self.recovery_available_ratio = recovery_available_ratio
        self.reclaim_available_bytes = reclaim_available_bytes
        self.recovery_available_bytes = recovery_available_bytes
        self.poll_interval_s = poll_interval_s
        self.consecutive_samples = consecutive_samples
        self.reclaim_cooldown_s = reclaim_cooldown_s
        self.probe = probe
        self.state = "disabled" if not self.enabled else "normal"
        self.low_samples = 0
        self.last_snapshot: SystemMemorySnapshot | None = None
        self.reclaim_watermark_bytes = 0
        self.recovery_watermark_bytes = 0
        self.target_release_bytes = 0
        self.last_queued: dict[str, int] = {}
        self.total_requested_bytes = 0
        self.unresolved_pressure_bytes = 0
        self.probe_errors = 0
        self.last_error: str | None = None
        self._last_request_monotonic = float("-inf")

    @property
    def enabled(self) -> bool:
        return self.reclaim_available_ratio is not None or self.reclaim_available_bytes > 0

    def _watermarks(self, total_bytes: int) -> tuple[int, int]:
        low = self.reclaim_available_bytes
        if self.reclaim_available_ratio is not None:
            low = max(low, int(total_bytes * self.reclaim_available_ratio))
        high = self.recovery_available_bytes
        if self.recovery_available_ratio is not None:
            high = max(high, int(total_bytes * self.recovery_available_ratio))
        return low, max(high, low)

    def evaluate_once(self, now_monotonic: float | None = None) -> dict[str, int]:
        if not self.enabled:
            return {}
        try:
            snapshot = self.probe()
        except Exception as exc:
            # Probe failures retain the previous state and never issue a release.
            # Clearing last_error on the next successful sample distinguishes a
            # current fault from historical probe_errors in the admin snapshot.
            self.probe_errors += 1
            self.last_error = repr(exc)
            return {}

        self.last_error = None
        self.last_snapshot = snapshot
        low, high = self._watermarks(snapshot.total_bytes)
        self.reclaim_watermark_bytes = low
        self.recovery_watermark_bytes = high

        if snapshot.available_bytes < low:
            self.low_samples = min(
                self.low_samples + 1,
                self.consecutive_samples,
            )
            if self.low_samples >= self.consecutive_samples:
                self.state = "reclaiming"
        elif snapshot.available_bytes >= high:
            self.low_samples = 0
            self.state = "normal"
        elif self.state != "reclaiming":
            self.low_samples = 0

        if self.state != "reclaiming":
            self.target_release_bytes = 0
            self.unresolved_pressure_bytes = 0
            self.last_queued = {}
            return {}

        self.target_release_bytes = max(high - snapshot.available_bytes, 0)
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if now - self._last_request_monotonic < self.reclaim_cooldown_s:
            return {}

        pool_stats = self.backup_pool.stats()
        pending_before = pool_stats["pending_release_bytes"]
        additional_bytes = max(self.target_release_bytes - pending_before, 0)
        queued = self.backup_pool.request_release_bytes(additional_bytes)
        self.last_queued = queued
        requested = sum(queued.values())
        self.total_requested_bytes += requested
        pending_bytes = self.backup_pool.stats()["pending_release_bytes"]
        self.unresolved_pressure_bytes = max(
            self.target_release_bytes - pending_bytes,
            0,
        )
        if requested > 0:
            self._last_request_monotonic = now
        return queued

    async def run(self) -> None:
        while True:
            self.evaluate_once()
            await asyncio.sleep(self.poll_interval_s)

    def snapshot(self) -> dict[str, object]:
        data: dict[str, object] = {
            "enabled": self.enabled,
            "state": self.state,
            "reclaim_available_ratio": self.reclaim_available_ratio,
            "recovery_available_ratio": self.recovery_available_ratio,
            "reclaim_available_bytes": self.reclaim_available_bytes,
            "recovery_available_bytes": self.recovery_available_bytes,
            "reclaim_watermark_bytes": self.reclaim_watermark_bytes,
            "recovery_watermark_bytes": self.recovery_watermark_bytes,
            "low_samples": self.low_samples,
            "target_release_bytes": self.target_release_bytes,
            "last_queued": self.last_queued,
            "total_requested_bytes": self.total_requested_bytes,
            "unresolved_pressure_bytes": self.unresolved_pressure_bytes,
            "probe_errors": self.probe_errors,
            "last_error": self.last_error,
        }
        if self.last_snapshot is not None:
            data["system_memory"] = {
                **asdict(self.last_snapshot),
                "available_ratio": self.last_snapshot.available_ratio,
            }
        return data
