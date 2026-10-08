# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-independent transactional backup runtime for one worker process."""

from __future__ import annotations

import dataclasses
import gc
import logging
import threading
import time
from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .contracts import HostBuffer, MemoryBackend, MemoryRegion, RestoreStream, SavePolicy
from .coordinator import make_cpu_backup_coordinator
from .diagnostics import record_event
from .disk import DiskRestoreSegment, DiskSegmentRef, DiskWriteSegment, ExactDiskBackupStore
from .pool import CpuBackupPool

logger = logging.getLogger(__name__)


def serialized(fn):
    @wraps(fn)
    def call(self, *args, **kwargs):
        with self.lifecycle_lock:
            if self.closed:
                raise RuntimeError("backup runtime is closed")
            return fn(self, *args, **kwargs)

    return call


class CpuBackupState(StrEnum):
    COPYING_D2H = "copying_d2h"
    REQUIRED_FOR_RESTORE = "required_for_restore"
    RESTORING_H2D = "restoring_h2d"
    CACHE_ONLY = "cache_only"
    INVALID = "invalid"


class DiskBackupState(StrEnum):
    WRITING = "writing"
    REQUIRED_FOR_RESTORE = "required_for_restore"
    RESTORING = "restoring"
    CACHE_ONLY = "cache_only"
    INVALID = "invalid"


class ResidencyState(StrEnum):
    AWAKE = "awake"
    SLEEP_PREPARED = "sleep_prepared"
    SLEEPING = "sleeping"
    RECOVERY_REQUIRED = "recovery_required"


@dataclasses.dataclass
class AllocationData:
    region: MemoryRegion
    cpu_backup_buffer: HostBuffer | None = None
    cpu_backup_valid: bool = False
    cpu_backup_state: CpuBackupState | None = None
    disk_backup_ref: DiskSegmentRef | None = None
    disk_backup_state: DiskBackupState | None = None
    resident: bool = True

    @property
    def tag(self) -> str:
        return self.region.tag


class BackupRuntime:
    default_tag = "default"

    def __init__(self, backend: MemoryBackend, config: RuntimeConfig):
        self.backend = backend
        self.config = config
        self.allocations: dict[int, AllocationData] = {}
        self.lifecycle_lock = threading.RLock()
        self.cpu_backup_lock = threading.RLock()
        self.cpu_backup_host_cache_flush_lock = threading.Lock()
        self.closed = False
        self.cpu_backup_pool = CpuBackupPool(backend)
        self.cpu_backup_coordinator = make_cpu_backup_coordinator(config)
        self.disk_backup_store: ExactDiskBackupStore | None = None
        self.disk_staging_buffers: tuple[HostBuffer, ...] | None = None
        self.disk_restore_stream: RestoreStream | None = None
        self.disk_hash_workers = 2
        self.disk_backup_written_bytes_total = 0
        self.disk_backup_read_bytes_total = 0
        self.disk_backup_write_errors = 0
        self.disk_backup_read_errors = 0
        self.cpu_backup_release_count = 0
        self.cpu_backup_release_bytes = 0
        self.cpu_backup_host_cache_flush_count = 0
        self.cpu_backup_host_cache_flush_errors = 0
        self.pending_cpu_backup_release_bytes = 0
        self._cpu_backup_usage_dirty = False
        self._cpu_backup_reclaimable_generation = 0
        self._cpu_backup_last_drain_generation = -1
        self._cpu_backup_poller_stop = threading.Event()
        self._cpu_backup_poller_thread: threading.Thread | None = None
        self.residency_state = ResidencyState.AWAKE
        self.sleeping_tags: set[str] = set()
        self.sleeping_exact_restore_tags: set[str] = set()
        self.partial_sleep = False
        if config.disk_enabled:
            try:
                self.disk_backup_store = ExactDiskBackupStore(
                    config.disk_root,
                    chunk_bytes=config.chunk_bytes,
                    direct_io=config.direct_io and backend.pin_memory,
                )
                self.disk_staging_buffers = tuple(
                    backend.allocate_host(config.chunk_bytes) for _ in range(4)
                )
                self.disk_restore_stream = backend.restore_stream(4)
            except BaseException:
                self.cleanup_exact_disk_backups()
                raise
        self._start_cpu_backup_release_poller()

    def record_event(self, phase: str, **payload: Any) -> None:
        record_event(self.config.profile_path, phase, **payload)

    @serialized
    def register(self, region: MemoryRegion) -> None:
        with self.cpu_backup_lock:
            if self.residency_state in {
                ResidencyState.SLEEP_PREPARED,
                ResidencyState.RECOVERY_REQUIRED,
            }:
                raise RuntimeError("cannot allocate during an unstable lifecycle")
            if region.address in self.allocations:
                raise ValueError("allocation address is already registered")
            if region.tag in self.sleeping_tags:
                raise RuntimeError("cannot allocate into a sleeping tag")
            self.allocations[region.address] = AllocationData(region)

    def unregister(self, address: int) -> None:
        with self.lifecycle_lock, self.cpu_backup_lock:
            if not self.closed and self.residency_state == ResidencyState.SLEEP_PREPARED:
                raise RuntimeError("cannot free an allocation in a prepared sleep transaction")
            data = self.allocations.pop(address)
            if data.cpu_backup_buffer is not None:
                self.cpu_backup_pool.release_to_free_list(data.cpu_backup_buffer)
                data.cpu_backup_buffer = None
                self._cpu_backup_usage_dirty = True
                self._cpu_backup_reclaimable_generation += 1
            self._drop_disk_ref(data)

    def _drop_disk_ref(self, data: AllocationData) -> None:
        ref = data.disk_backup_ref
        data.disk_backup_ref = None
        data.disk_backup_state = None
        if ref is not None and self.disk_backup_store is not None:
            if not any(
                other.disk_backup_ref is not None
                and other.disk_backup_ref.bundle_dir == ref.bundle_dir
                for other in self.allocations.values()
            ):
                self.disk_backup_store.delete_bundle(ref)

    @serialized
    def reset_snapshots(self) -> None:
        """Explicit reconstruction boundary, with all old mappings still awake."""
        self.require_awake("reset snapshots")
        with self.cpu_backup_lock:
            for data in self.allocations.values():
                if data.cpu_backup_buffer is not None:
                    self.cpu_backup_pool.release_to_free_list(data.cpu_backup_buffer)
                    data.cpu_backup_buffer = None
                data.cpu_backup_valid = False
                data.cpu_backup_state = None
                self._drop_disk_ref(data)
            self._cpu_backup_reclaimable_generation += 1
            self._report_cpu_backup_usage_locked()

    def reclaim(self, target_bytes: int) -> dict[str, int | bool]:
        if target_bytes < 0:
            raise ValueError("target bytes must be non-negative")
        with self.cpu_backup_lock:
            released_before = self.cpu_backup_release_bytes
            self.pending_cpu_backup_release_bytes += target_bytes
            released = self._drain_pending_cpu_backup_release_locked()
            released_bytes = self.cpu_backup_release_bytes - released_before
        if released:
            self._flush_cpu_backup_host_cache()
        self.cpu_backup_coordinator.flush()
        if released_bytes and self.disk_backup_store is not None:
            self.record_event(
                "exact_disk_demotion",
                released_bytes=released_bytes,
                source_medium="disk",
                fallback=False,
            )
        return {
            "released": released,
            "requested_bytes": target_bytes,
            "remaining_cpu_backup_bytes": self.cpu_backup_pool.reserved_bytes,
            "released_bytes_total": self.cpu_backup_release_bytes,
            "pending_release_bytes": self.pending_cpu_backup_release_bytes,
        }

    def _flush_cpu_backup_host_cache(self) -> None:
        with self.cpu_backup_host_cache_flush_lock:
            try:
                self.backend.empty_host_cache()
                self.cpu_backup_host_cache_flush_count += 1
            except Exception:
                self.cpu_backup_host_cache_flush_errors += 1
                logger.exception("Failed to flush pinned host memory cache")

    def cleanup_exact_disk_backups(self) -> None:
        with self.cpu_backup_lock:
            if self.disk_restore_stream is not None:
                self.disk_restore_stream.close()
                self.disk_restore_stream = None
            self.disk_staging_buffers = None
            if self.disk_backup_store is not None:
                self.disk_backup_store.cleanup_process_dir()
                self.disk_backup_store = None
            for data in self.allocations.values():
                data.disk_backup_ref = None
                data.disk_backup_state = None

    def close(self) -> None:
        self.shutdown_cpu_backup_release_poller()
        with self.lifecycle_lock:
            if self.closed:
                return
            self.backend.synchronize()
            self.cleanup_exact_disk_backups()
            with self.cpu_backup_lock:
                for data in self.allocations.values():
                    if data.cpu_backup_buffer is not None:
                        self.cpu_backup_pool.discard(data.cpu_backup_buffer)
                        data.cpu_backup_buffer = None
                        data.cpu_backup_valid = False
                        data.cpu_backup_state = None
                self.cpu_backup_pool.discard_free_bytes(self.cpu_backup_pool.reserved_bytes)
                self.closed = True
            self._flush_cpu_backup_host_cache()

    def require_awake(self, operation: str) -> None:
        with self.cpu_backup_lock:
            if self.residency_state != ResidencyState.AWAKE:
                raise RuntimeError(
                    f"cannot {operation}; allocator residency is {self.residency_state.value}"
                )

    def require_tags_awake(self, operation: str, tags: set[str]) -> None:
        """Require selected mappings without forbidding a staged partial wake."""
        with self.cpu_backup_lock:
            if self.residency_state == ResidencyState.RECOVERY_REQUIRED:
                raise RuntimeError(f"cannot {operation}; allocator residency requires recovery")
            still_sleeping = tags.intersection(self.sleeping_tags)
            if still_sleeping:
                raise RuntimeError(
                    f"cannot {operation}; tags are still sleeping: {sorted(still_sleeping)}"
                )

    def _start_cpu_backup_release_poller(self) -> None:
        """Start a best-effort background poller for release-byte requests.

        The poller only records a target byte count and releases local tensors
        under `cpu_backup_lock`. Required-for-restore and in-flight copy tensors
        are never released; unsatisfied bytes remain pending until a later safe
        point makes cache-only storage available.
        """
        if not self.cpu_backup_coordinator.is_enabled():
            return
        if self._cpu_backup_poller_thread is not None and self._cpu_backup_poller_thread.is_alive():
            return
        if self._cpu_backup_poller_stop.is_set():
            self._cpu_backup_poller_stop = threading.Event()
        interval_s = self.config.poll_interval_s
        if interval_s <= 0:
            return
        self._cpu_backup_poller_thread = threading.Thread(
            target=self._cpu_backup_release_poll_loop,
            args=(interval_s,),
            name="vllm-switch-cpu-backup-release-poller",
            daemon=True,
        )
        self._cpu_backup_poller_thread.start()

    def _cpu_backup_release_poll_loop(self, interval_s: float) -> None:
        while not self._cpu_backup_poller_stop.wait(interval_s):
            try:
                self._poll_and_release_cpu_backups()
            except Exception:  # pragma: no cover - defensive daemon path.
                # Rate is bounded by interval_s. A transient allocator/control
                # plane failure must not permanently disable later reclaim.
                logger.exception("CPU backup release poller failed")

    def shutdown_cpu_backup_release_poller(self) -> None:
        """Stop the optional background release poller.

        Tests call this explicitly after enabling a fake coordinator. Production
        workers normally rely on process teardown, because the thread is daemon
        and only performs best-effort control-plane polling.
        """
        self._cpu_backup_poller_stop.set()
        if self._cpu_backup_poller_thread is not None:
            self._cpu_backup_poller_thread.join(timeout=self.config.coordinator_timeout_s * 3 + 1)
            if self._cpu_backup_poller_thread.is_alive():
                raise RuntimeError("backup release poller did not stop")
            self._cpu_backup_poller_thread = None

    def _report_cpu_backup_usage_locked(self) -> None:
        self.cpu_backup_coordinator.report_usage(self._cpu_backup_usage_snapshot_locked())
        self._cpu_backup_usage_dirty = False

    def _cpu_backup_usage_snapshot_locked(self) -> dict[str, int]:
        total_bytes = self.cpu_backup_pool.reserved_bytes
        required_for_restore_bytes = 0
        cache_only_bytes = 0
        invalid_bytes = 0
        for data in self.allocations.values():
            if data.cpu_backup_buffer is None:
                continue
            size_bytes = data.cpu_backup_buffer.size_bytes
            if data.cpu_backup_state in {
                CpuBackupState.COPYING_D2H,
                CpuBackupState.REQUIRED_FOR_RESTORE,
                CpuBackupState.RESTORING_H2D,
            }:
                # All three states are non-evictable. Folding in-flight copies
                # into this aggregate prevents the controller from interpreting
                # them as detached free-list storage.
                required_for_restore_bytes += size_bytes
            elif data.cpu_backup_state == CpuBackupState.INVALID:
                invalid_bytes += size_bytes
            elif data.cpu_backup_state == CpuBackupState.CACHE_ONLY:
                cache_only_bytes += size_bytes
        active_bytes = required_for_restore_bytes + cache_only_bytes + invalid_bytes
        disk_exact_bytes = 0
        disk_required_for_restore_bytes = 0
        disk_cache_only_bytes = 0
        disk_write_inflight_bytes = 0
        ram_reclaimable_with_disk_bytes = 0
        for data in self.allocations.values():
            if data.disk_backup_state == DiskBackupState.WRITING:
                disk_write_inflight_bytes += data.region.size_bytes
            if not self._disk_backup_is_current(data):
                continue
            disk_exact_bytes += data.region.size_bytes
            if data.disk_backup_state == DiskBackupState.REQUIRED_FOR_RESTORE:
                disk_required_for_restore_bytes += data.region.size_bytes
            elif data.disk_backup_state == DiskBackupState.CACHE_ONLY:
                disk_cache_only_bytes += data.region.size_bytes
            if (
                data.cpu_backup_buffer is not None
                and data.cpu_backup_state == CpuBackupState.REQUIRED_FOR_RESTORE
            ):
                ram_reclaimable_with_disk_bytes += data.region.size_bytes
        return {
            "total_bytes": total_bytes,
            # Monotonic acknowledgement survives latest-wins coalescing when
            # released capacity is reallocated before the next HTTP flush.
            "released_bytes_total": self.cpu_backup_release_bytes,
            "required_for_restore_bytes": required_for_restore_bytes,
            "cache_only_bytes": cache_only_bytes,
            "invalid_bytes": invalid_bytes,
            "free_local_bytes": max(total_bytes - active_bytes, 0),
            # Controller protocol fields. Only required CPU bytes need a disk
            # restore source; cache-only/invalid/free-local bytes are already
            # represented by their ordinary reclaimable RAM buckets.
            "disk_backup_current_bytes": disk_exact_bytes,
            # This process-local store has no preallocated disk reservation;
            # every committed byte is both current and physically retained.
            "disk_backup_reserved_bytes": disk_exact_bytes,
            "ram_reclaimable_with_disk_bytes": ram_reclaimable_with_disk_bytes,
            # Worker-local telemetry retained in coordinator metadata/profile.
            "disk_exact_bytes": disk_exact_bytes,
            "disk_required_for_restore_bytes": disk_required_for_restore_bytes,
            "disk_cache_only_bytes": disk_cache_only_bytes,
            "disk_write_inflight_bytes": disk_write_inflight_bytes,
            "disk_persisted_bytes_total": self.disk_backup_written_bytes_total,
            "disk_restore_bytes_total": self.disk_backup_read_bytes_total,
            "disk_write_errors_total": self.disk_backup_write_errors,
            "disk_read_errors_total": self.disk_backup_read_errors,
        }

    def _drain_pending_cpu_backup_release_locked(self) -> bool:
        """Release daemon-requested bytes only from locally safe states."""
        target_bytes = self.pending_cpu_backup_release_bytes
        if target_bytes <= 0:
            return False

        # The release order is a policy choice: detached/stale storage has no
        # reuse value, so preserve clean cache-only snapshots until last.
        released_bytes = self.cpu_backup_pool.discard_free_bytes(target_bytes)
        target_bytes -= released_bytes

        # Invalid backups are stale and not useful for reuse/restore, so release
        # them before valid cache-only backups.
        for state in (CpuBackupState.INVALID, CpuBackupState.CACHE_ONLY):
            if target_bytes <= 0:
                break
            for data in self.allocations.values():
                if target_bytes <= 0:
                    break
                if data.cpu_backup_state != state or data.cpu_backup_buffer is None:
                    continue
                bytes_released = self.cpu_backup_pool.discard(data.cpu_backup_buffer)
                data.cpu_backup_buffer = None
                data.cpu_backup_valid = False

                data.cpu_backup_state = None
                released_bytes += bytes_released
                target_bytes -= bytes_released

        # Current exact disk copies make even sleeping CPU-required storage
        # reclaimable: the restore responsibility is atomically transferred to
        # DISK_REQUIRED before the pinned tensor is detached.
        if self.disk_backup_store is not None and target_bytes > 0:
            needs_disk = [
                data
                for data in self.allocations.values()
                if data.cpu_backup_buffer is not None
                and data.cpu_backup_state
                in {CpuBackupState.REQUIRED_FOR_RESTORE, CpuBackupState.CACHE_ONLY}
                and not self._disk_backup_is_current(data)
            ]
            if needs_disk:
                try:
                    self.prepare_disk_backup(tuple(sorted({data.tag for data in needs_disk})))
                except Exception:
                    # A failed spill must preserve the CPU exact source and the
                    # pending obligation. Serving correctness takes precedence
                    # over satisfying host-pressure accounting immediately.
                    logger.exception("Failed to demote exact CPU backups to disk; retaining RAM")
                    self.pending_cpu_backup_release_bytes = max(target_bytes, 0)
                    self._cpu_backup_last_drain_generation = self._cpu_backup_reclaimable_generation
                    if released_bytes > 0:
                        # Earlier FREE_LOCAL/INVALID/CACHE_ONLY victims were
                        # already detached. Acknowledge those real releases even
                        # though the required-source demotion failed.
                        self.cpu_backup_release_count += 1
                        self.cpu_backup_release_bytes += released_bytes
                        self._report_cpu_backup_usage_locked()
                        return True
                    return False
            for data in self.allocations.values():
                if target_bytes <= 0:
                    break
                if (
                    data.cpu_backup_buffer is None
                    or data.cpu_backup_state
                    not in {
                        CpuBackupState.REQUIRED_FOR_RESTORE,
                        CpuBackupState.CACHE_ONLY,
                    }
                    or not self._disk_backup_is_current(data)
                ):
                    continue
                previous_state = data.cpu_backup_state
                bytes_released = self.cpu_backup_pool.discard(data.cpu_backup_buffer)
                data.cpu_backup_buffer = None
                data.cpu_backup_valid = False

                data.cpu_backup_state = None
                data.disk_backup_state = (
                    DiskBackupState.REQUIRED_FOR_RESTORE
                    if previous_state == CpuBackupState.REQUIRED_FOR_RESTORE
                    else DiskBackupState.CACHE_ONLY
                )
                released_bytes += bytes_released
                target_bytes -= bytes_released

        self.pending_cpu_backup_release_bytes = max(target_bytes, 0)
        self._cpu_backup_last_drain_generation = self._cpu_backup_reclaimable_generation
        if released_bytes > 0:
            self.cpu_backup_release_count += 1
            self.cpu_backup_release_bytes += released_bytes
            self._report_cpu_backup_usage_locked()
            return True
        return False

    def _poll_and_release_cpu_backups(self) -> None:
        """Deliver usage, poll once, and drain an obligation only when useful."""
        with self.cpu_backup_lock:
            if self._cpu_backup_usage_dirty:
                self._report_cpu_backup_usage_locked()

        # poll_release_request flushes the newest usage before its GET. Keep all
        # control-plane I/O outside the allocator state lock.
        target_free_bytes = self.cpu_backup_coordinator.poll_release_request()
        with self.cpu_backup_lock:
            self.pending_cpu_backup_release_bytes += target_free_bytes
            state_changed = (
                self._cpu_backup_reclaimable_generation != self._cpu_backup_last_drain_generation
            )
            should_drain = target_free_bytes > 0 or (
                self.pending_cpu_backup_release_bytes > 0 and state_changed
            )
            released = self._drain_pending_cpu_backup_release_locked() if should_drain else False
        if released:
            self._flush_cpu_backup_host_cache()
            # The release counter acknowledges completion. Do not delay it until
            # a later timer tick after storage has already dropped.
            self.cpu_backup_coordinator.flush()

    def _release_reclaimable_cpu_backups(self) -> None:
        self._poll_and_release_cpu_backups()

    @staticmethod
    def _normalize_tags(
        tags: tuple[str, ...] | str | None,
        *,
        default: tuple[str, ...] | None = None,
    ) -> tuple[str, ...] | None:
        if tags is None:
            return default
        if isinstance(tags, str):
            return (tags,)
        return tags

    @staticmethod
    def _cpu_backup_is_current(data: AllocationData) -> bool:
        return data.cpu_backup_buffer is not None and data.cpu_backup_valid

    @staticmethod
    def _disk_backup_is_current(data: AllocationData) -> bool:
        if (
            data.disk_backup_ref is None
            or data.disk_backup_state
            not in {DiskBackupState.CACHE_ONLY, DiskBackupState.REQUIRED_FOR_RESTORE}
            or data.disk_backup_ref.region_id != data.region.region_id
        ):
            return False
        bundle_dir = data.disk_backup_ref.bundle_dir
        return all(
            (bundle_dir / name).is_file() for name in ("data.bin", "manifest.json", "COMMIT")
        )

    def prepare_disk_backup(
        self,
        tags: tuple[str, ...] | str | None = None,
    ) -> dict[str, int | float]:
        """Publish current CPU exact snapshots as one immutable disk bundle."""
        if self.disk_backup_store is None:
            return {
                "disk_backup_written_bytes": 0,
                "disk_backup_write_s": 0.0,
                "disk_backup_reused_bytes": 0,
            }
        tag_set = self._normalize_tags(tags)
        started_at = time.perf_counter()
        candidates: list[tuple[int, AllocationData]] = []
        writes: list[DiskWriteSegment] = []
        reused_bytes = 0
        with self.cpu_backup_lock:
            superseded_bundles: set[Path] = set()
            for ptr, data in self.allocations.items():
                if tag_set is not None and data.tag not in tag_set:
                    continue
                if self._disk_backup_is_current(data):
                    reused_bytes += data.region.size_bytes
                    continue
                if not self._cpu_backup_is_current(data):
                    raise RuntimeError(
                        "cannot publish disk backup without a current CPU exact source"
                    )
                assert data.cpu_backup_buffer is not None

                if data.disk_backup_ref is not None:
                    superseded_bundles.add(data.disk_backup_ref.bundle_dir)
                data.disk_backup_state = DiskBackupState.WRITING
                writes.append(
                    DiskWriteSegment(
                        region_id=data.region.region_id,
                        data=data.cpu_backup_buffer.view(),
                    )
                )
                candidates.append((ptr, data))
            try:
                refs = self.disk_backup_store.write_bundle(writes) if writes else {}
                for _ptr, data in candidates:
                    data.disk_backup_ref = refs[data.region.region_id]

                    data.disk_backup_state = (
                        DiskBackupState.REQUIRED_FOR_RESTORE
                        if data.cpu_backup_state == CpuBackupState.REQUIRED_FOR_RESTORE
                        else DiskBackupState.CACHE_ONLY
                    )
            except BaseException:
                self.disk_backup_write_errors += 1
                for _ptr, data in candidates:
                    data.disk_backup_state = DiskBackupState.INVALID

                    data.disk_backup_ref = None
                raise
            written_bytes = sum(data.region.size_bytes for _ptr, data in candidates)
            self.disk_backup_written_bytes_total += written_bytes
            current_bundle_dirs = {
                data.disk_backup_ref.bundle_dir
                for data in self.allocations.values()
                if data.disk_backup_ref is not None
            }
            for bundle_dir in superseded_bundles - current_bundle_dirs:
                self.disk_backup_store.delete_bundle(bundle_dir)
        elapsed = time.perf_counter() - started_at
        if written_bytes or reused_bytes:
            self.record_event(
                "exact_disk_spill",
                disk_spill_bytes=written_bytes,
                disk_spill_s=elapsed,
                disk_reused_bytes=reused_bytes,
                source_medium="cpu",
                fallback=False,
            )
        return {
            "disk_backup_written_bytes": written_bytes,
            "disk_backup_write_s": elapsed,
            "disk_backup_reused_bytes": reused_bytes,
        }

    @serialized
    def prepare_cpu_backup(
        self,
        tags: tuple[str, ...] | str | None = None,
        *,
        report: bool = True,
    ) -> dict[str, int | float]:
        """Synchronously snapshot mapped allocations without unmapping them.

        This is the prepare half of level-1 sleep. Each snapshot is published
        only after its D2H copy completes. Immutable regions reuse their published
        source, while mutable regions are captured on every preparation.
        """
        tag_set = self._normalize_tags(tags)
        started_at = time.perf_counter()
        prepared_bytes = 0
        reused_bytes = 0
        reused_count = 0
        allocated_bytes = 0
        copy_d2h_s = 0.0

        self.backend.synchronize()
        with self.cpu_backup_lock:
            if self.residency_state != ResidencyState.AWAKE:
                raise RuntimeError(
                    "cannot prepare CPU backups while allocator residency is "
                    f"{self.residency_state.value}"
                )
            try:
                for ptr, data in self.allocations.items():
                    if tag_set is not None and data.tag not in tag_set:
                        continue
                    size_in_bytes = data.region.size_bytes
                    if data.region.policy == SavePolicy.DISCARD:
                        continue
                    if data.region.policy == SavePolicy.SNAPSHOT:
                        data.cpu_backup_valid = False
                        self._drop_disk_ref(data)
                    if self._cpu_backup_is_current(data):
                        reused_count += 1
                        reused_bytes += size_in_bytes
                        continue
                    if self._disk_backup_is_current(data):
                        # A current exact disk source is sufficient for the next
                        # sleep; rebuilding full pinned RAM would defeat demotion.
                        continue

                    if data.cpu_backup_buffer is None:
                        cpu_backup_buffer, _ = self.cpu_backup_pool.acquire(size_in_bytes)
                        data.cpu_backup_buffer = cpu_backup_buffer
                        allocated_bytes += size_in_bytes
                    else:
                        cpu_backup_buffer = data.cpu_backup_buffer

                    data.cpu_backup_valid = False
                    data.cpu_backup_state = CpuBackupState.COPYING_D2H
                    copy_started_at = time.perf_counter()
                    self.backend.copy(cpu_backup_buffer.address, ptr, size_in_bytes)
                    copy_d2h_s += time.perf_counter() - copy_started_at

                    data.cpu_backup_valid = True
                    data.cpu_backup_state = CpuBackupState.CACHE_ONLY
                    prepared_bytes += size_in_bytes
            except BaseException:
                # Publication is transactional at the selected commit-set level.
                # Keep storage, but no subset may remain reusable after failure.
                for data in self.allocations.values():
                    if tag_set is not None and data.tag not in tag_set:
                        continue
                    if data.cpu_backup_buffer is not None:
                        data.cpu_backup_valid = False

                        data.cpu_backup_state = CpuBackupState.INVALID
                self._cpu_backup_reclaimable_generation += 1
                if report:
                    self._report_cpu_backup_usage_locked()
                raise

            if report:
                self._report_cpu_backup_usage_locked()
        if report:
            self.cpu_backup_coordinator.flush()
        return {
            "prepared_bytes": prepared_bytes,
            "reused_count": reused_count,
            "reused_bytes": reused_bytes,
            "allocated_bytes": allocated_bytes,
            "copy_d2h_s": copy_d2h_s,
            "latency_s": time.perf_counter() - started_at,
        }

    @serialized
    def prepare_sleep(
        self, offload_tags: tuple[str, ...] | str | None = None
    ) -> dict[str, int | float]:
        """Snapshot and lease atomically with respect to pressure reclaim."""
        offload_tags = self._normalize_tags(offload_tags, default=(self.default_tag,))
        assert offload_tags is not None
        with self.cpu_backup_lock:
            self.require_awake("prepare sleep")
            stats = self.prepare_cpu_backup(offload_tags, report=False)
            selected = [
                data
                for data in self.allocations.values()
                if data.tag in offload_tags and data.region.policy != SavePolicy.DISCARD
            ]
            if any(
                not self._cpu_backup_is_current(data) and not self._disk_backup_is_current(data)
                for data in selected
            ):
                raise RuntimeError("cannot prepare sleep without a current exact restore source")
            for data in selected:
                if self._cpu_backup_is_current(data):
                    data.cpu_backup_state = CpuBackupState.REQUIRED_FOR_RESTORE
                else:
                    data.disk_backup_state = DiskBackupState.REQUIRED_FOR_RESTORE
            self.residency_state = ResidencyState.SLEEP_PREPARED
            self._report_cpu_backup_usage_locked()
        self.cpu_backup_coordinator.flush()
        self.record_event("allocator_prepare_sleep", **stats)
        return stats

    @serialized
    def abort_sleep_prepare(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """Release a prepared lease while retaining clean pinned storage."""
        offload_tags = self._normalize_tags(offload_tags, default=(BackupRuntime.default_tag,))
        assert isinstance(offload_tags, tuple)
        with self.cpu_backup_lock:
            if self.residency_state != ResidencyState.SLEEP_PREPARED:
                return
            for data in self.allocations.values():
                if (
                    data.tag in offload_tags
                    and data.cpu_backup_state == CpuBackupState.REQUIRED_FOR_RESTORE
                ):
                    data.cpu_backup_state = CpuBackupState.CACHE_ONLY
                if (
                    data.tag in offload_tags
                    and data.disk_backup_state == DiskBackupState.REQUIRED_FOR_RESTORE
                ):
                    data.disk_backup_state = DiskBackupState.CACHE_ONLY
            self.residency_state = ResidencyState.AWAKE
            self._cpu_backup_reclaimable_generation += 1
            self._report_cpu_backup_usage_locked()
        self.cpu_backup_coordinator.flush()

    @serialized
    def sleep(
        self,
        offload_tags: tuple[str, ...] | str | None = None,
        *,
        skip_prepare: bool = False,
        release_bytes: int | None = None,
    ) -> None:
        """
        Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.

        :param offload_tags: The tags of the memory allocation that will be
            offloaded. The rest of the memory allocation will be discarded.
        """
        offload_tags = self._normalize_tags(offload_tags, default=(BackupRuntime.default_tag,))

        assert isinstance(offload_tags, tuple)
        if release_bytes is not None and (release_bytes < 0 or offload_tags != ("weights",)):
            raise ValueError("partial eviction requires a nonnegative L1 weight release budget")

        # Phase 1: make every required backup current while all GPU mappings are
        # still intact. A D2H failure therefore cannot leave a partially slept
        # allocator.
        if skip_prepare:
            with self.cpu_backup_lock:
                if self.residency_state != ResidencyState.SLEEP_PREPARED:
                    raise RuntimeError(
                        "cannot commit sleep before prepare; allocator residency is "
                        f"{self.residency_state.value}"
                    )
                stale = [
                    ptr
                    for ptr, data in self.allocations.items()
                    if data.tag in offload_tags
                    and data.region.policy != SavePolicy.DISCARD
                    and not self._cpu_backup_is_current(data)
                    and not self._disk_backup_is_current(data)
                ]
            if stale:
                raise RuntimeError(
                    f"cannot commit sleep with stale CPU backups for {len(stale)} allocation(s)"
                )
            prepare_stats: dict[str, int | float] = {
                "copy_d2h_s": 0.0,
                "reused_count": 0,
                "reused_bytes": 0,
            }
        else:
            prepare_stats = self.prepare_sleep(offload_tags)

        # Validate the entire commit set before the first unmap. This closes the
        # prepare/commit gap against stale or missing offload backups.
        with self.cpu_backup_lock:
            stale = [
                ptr
                for ptr, data in self.allocations.items()
                if data.tag in offload_tags
                and data.region.policy != SavePolicy.DISCARD
                and not self._cpu_backup_is_current(data)
                and not self._disk_backup_is_current(data)
            ]
            if stale:
                raise RuntimeError(
                    f"cannot commit sleep with stale CPU backups for {len(stale)} allocation(s)"
                )
            if self.residency_state != ResidencyState.SLEEP_PREPARED:
                raise RuntimeError(
                    "cannot commit sleep before prepare; allocator residency is "
                    f"{self.residency_state.value}"
                )

        total_bytes = 0
        backup_bytes = 0

        # Used for profile
        profile = bool(self.config.profile_path)
        started_at = time.perf_counter()
        cpu_backup_alloc_s = 0.0
        cpu_backup_reuse_count = int(prepare_stats["reused_count"])
        cpu_backup_reused_bytes = int(prepare_stats["reused_bytes"])
        cpu_backup_allocated_bytes = int(prepare_stats.get("allocated_bytes", 0))
        copy_d2h_s = float(prepare_stats["copy_d2h_s"])
        unmap_release_s = 0.0
        bytes_by_tag: dict[str, int] = {}
        backup_bytes_by_tag: dict[str, int] = {}
        discard_bytes_by_tag: dict[str, int] = {}

        unmapped_count = 0
        victims = set(self.allocations)
        if release_bytes is not None:
            # Discard KV first. Keep exact weight regions that the next model
            # does not need us to release. Whole regions remain the VMM unit.
            victims = {
                ptr
                for ptr, data in self.allocations.items()
                if data.tag not in offload_tags or data.region.policy == SavePolicy.DISCARD
            }
            freed = sum(self.allocations[ptr].region.size_bytes for ptr in victims)
            candidates = [data for ptr, data in self.allocations.items() if ptr not in victims]
            while freed < release_bytes and candidates:
                remaining = release_bytes - freed
                fitting = [data for data in candidates if data.region.size_bytes <= remaining]
                chosen = (
                    max(fitting, key=lambda d: d.region.size_bytes)
                    if fitting
                    else min(candidates, key=lambda d: d.region.size_bytes)
                )
                candidates.remove(chosen)
                victims.add(chosen.region.address)
                freed += chosen.region.size_bytes
        try:
            for data in self.allocations.values():
                handle = data.region
                if handle.address not in victims:
                    with self.cpu_backup_lock:
                        if data.cpu_backup_state == CpuBackupState.REQUIRED_FOR_RESTORE:
                            data.cpu_backup_state = CpuBackupState.CACHE_ONLY
                        if data.disk_backup_state == DiskBackupState.REQUIRED_FOR_RESTORE:
                            data.disk_backup_state = DiskBackupState.CACHE_ONLY
                    continue
                total_bytes += handle.size_bytes

                if profile:
                    bytes_by_tag[data.tag] = bytes_by_tag.get(data.tag, 0) + handle.size_bytes

                if data.tag in offload_tags and data.region.policy != SavePolicy.DISCARD:
                    backup_bytes += handle.size_bytes
                    size_in_bytes = handle.size_bytes

                    if profile:
                        backup_bytes_by_tag[data.tag] = (
                            backup_bytes_by_tag.get(data.tag, 0) + handle.size_bytes
                        )
                    reused_cpu_backup = False
                    with self.cpu_backup_lock:
                        if self._cpu_backup_is_current(data):
                            data.cpu_backup_state = CpuBackupState.REQUIRED_FOR_RESTORE
                            reused_cpu_backup = True
                        elif self._disk_backup_is_current(data):
                            data.disk_backup_state = DiskBackupState.REQUIRED_FOR_RESTORE
                        else:
                            raise RuntimeError(
                                "cannot sleep without a current exact restore source"
                            )
                    if skip_prepare and reused_cpu_backup:
                        cpu_backup_reuse_count += 1
                        cpu_backup_reused_bytes += size_in_bytes
                elif profile:
                    discard_bytes_by_tag[data.tag] = (
                        discard_bytes_by_tag.get(data.tag, 0) + handle.size_bytes
                    )
                unmap_started_at = time.perf_counter()
                self.backend.unmap(handle)
                data.resident = False
                unmapped_count += 1
                unmap_release_s += time.perf_counter() - unmap_started_at
        except BaseException:
            with self.cpu_backup_lock:
                # Even failure on the first unmap has an unknown VMM outcome.
                self.residency_state = ResidencyState.RECOVERY_REQUIRED
            raise
        with self.cpu_backup_lock:
            self.residency_state = ResidencyState.SLEEPING
            self.partial_sleep = release_bytes is not None
            self.sleeping_tags = {data.tag for data in self.allocations.values()}
            self.sleeping_exact_restore_tags = set(offload_tags)

        # Report once per allocator transition. Building a snapshot scans every
        # allocation, so reporting inside this loop would turn sleep into O(N²).
        with self.cpu_backup_lock:
            self._report_cpu_backup_usage_locked()

        logger.info(
            "BackupRuntime: sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded "
            "directly.",
            total_bytes / 1024**3,
            backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
        )

        gc.collect()
        self.backend.empty_device_cache()
        self.cpu_backup_coordinator.flush()

        if profile:
            stats = self.get_cpu_backup_pool_stats()
            coordinator_stats = self.cpu_backup_coordinator.get_profile_fields()
            self.record_event(
                "allocator_sleep",
                offload_tags=list(offload_tags),
                allocation_count=len(self.allocations),
                evicted_allocation_count=unmapped_count,
                requested_release_bytes=release_bytes,
                retained_gpu_bytes=sum(
                    d.region.size_bytes for d in self.allocations.values() if d.resident
                ),
                total_bytes=total_bytes,
                backup_bytes=backup_bytes,
                discard_bytes=total_bytes - backup_bytes,
                bytes_by_tag=bytes_by_tag,
                backup_bytes_by_tag=backup_bytes_by_tag,
                discard_bytes_by_tag=discard_bytes_by_tag,
                cpu_backup_pin_memory=stats["pin_memory"],
                # Eager prepare owns allocation/D2H timing. The sleep commit
                # does no allocation, so expose byte evidence without inventing
                # an allocation duration or hit/miss count.
                cpu_backup_alloc_s=cpu_backup_alloc_s,
                cpu_backup_allocated_bytes=cpu_backup_allocated_bytes,
                cpu_backup_reuse_count=cpu_backup_reuse_count,
                cpu_backup_reused_bytes=cpu_backup_reused_bytes,
                cpu_backup_pool_reserved_bytes=stats["reserved_bytes"],
                cpu_backup_pool_free_bytes=stats["free_bytes"],
                cpu_backup_release_count=(self.cpu_backup_release_count),
                cpu_backup_release_bytes=(self.cpu_backup_release_bytes),
                cpu_backup_host_cache_flush_count=(self.cpu_backup_host_cache_flush_count),
                cpu_backup_host_cache_flush_errors=(self.cpu_backup_host_cache_flush_errors),
                copy_d2h_s=copy_d2h_s,
                unmap_release_s=unmap_release_s,
                **coordinator_stats,
                latency_s=time.perf_counter() - started_at,
            )

    @staticmethod
    def _require_valid_cpu_backup(data: AllocationData) -> None:
        if not BackupRuntime._cpu_backup_is_current(data):
            raise RuntimeError("cannot restore an allocation from an invalid CPU backup")

    @serialized
    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory.

        :param tags: The tags of the memory allocation that will be loaded
            back to GPU memory. If None, all memory allocation will be loaded
            back to GPU memory.
        """
        profile = bool(self.config.profile_path)
        started_at = time.perf_counter()
        create_map_s = 0.0
        disk_create_map_s = 0.0
        deferred_create_map_s = 0.0
        copy_h2d_s = 0.0
        async_stream = None
        async_restored: list[AllocationData] = []
        copy_enqueue_s = 0.0
        copy_wait_s = 0.0
        async_started_at: float | None = None
        cpu_restore_pipeline_s = 0.0
        bytes_by_tag: dict[str, int] = {}
        cpu_restored_bytes_by_tag: dict[str, int] = {}
        disk_restored_bytes_by_tag: dict[str, int] = {}
        disk_read_s = 0.0
        disk_hash_s = 0.0
        disk_copy_h2d_s = 0.0
        disk_copy_enqueue_s = 0.0
        disk_copy_wait_s = 0.0
        disk_pipeline_wall_s = 0.0
        disk_plan: list[tuple[int, AllocationData, DiskSegmentRef]] = []
        deferred_remap_plan: list[tuple[int, AllocationData]] = []
        restore_source_by_tag: dict[str, str] = {}
        remapped_without_backup_bytes_by_tag: dict[str, int] = {}

        with self.cpu_backup_lock:
            if self.residency_state != ResidencyState.SLEEPING:
                raise RuntimeError(
                    f"cannot wake while allocator residency is {self.residency_state.value}"
                )
            selected_tags = set(self.sleeping_tags) if tags is None else set(tags)
            if not selected_tags.issubset(self.sleeping_tags):
                raise RuntimeError(
                    "cannot wake tags that are not sleeping: "
                    f"{sorted(selected_tags - self.sleeping_tags)}"
                )
            missing_sources = [
                ptr
                for ptr, data in self.allocations.items()
                if data.tag in selected_tags
                and not data.resident
                and data.tag in self.sleeping_exact_restore_tags
                and data.region.policy != SavePolicy.DISCARD
                and not self._cpu_backup_is_current(data)
                and not self._disk_backup_is_current(data)
            ]
            if missing_sources:
                raise RuntimeError(
                    "cannot wake without a current exact restore source for "
                    f"{len(missing_sources)} allocation(s)"
                )
            disk_refs_to_validate = [
                data.disk_backup_ref
                for data in self.allocations.values()
                if data.tag in selected_tags
                and not data.resident
                and data.cpu_backup_buffer is None
                and self._disk_backup_is_current(data)
                and data.disk_backup_ref is not None
            ]

        # Validate every committed manifest before the first VMM remap. Payload
        # hashes are still checked chunk-by-chunk by the pipeline.
        if disk_refs_to_validate:
            if self.disk_backup_store is None:
                raise RuntimeError("exact disk backup restore is not configured")
            try:
                for ref in disk_refs_to_validate:
                    self.disk_backup_store.validate_restore_ref(ref)
            except BaseException:
                self.disk_backup_read_errors += 1
                raise

        try:
            selected_allocations = [
                (ptr, data)
                for ptr, data in self.allocations.items()
                if data.tag in selected_tags and not data.resident
            ]
            if self.config.async_cpu_restore:
                async_stream = self.backend.restore_stream(1)
            exact_disk_restore = any(
                data.cpu_backup_buffer is None and self._disk_backup_is_current(data)
                for _ptr, data in selected_allocations
            )
            # Exact disk wake restores source-backed allocations first. Empty
            # allocations such as KV cache are remapped only after the weight
            # payload is resident, matching the staged L2 ordering and avoiding
            # unnecessary VMM work on the critical path to disk restore.
            if exact_disk_restore:
                selected_allocations.sort(
                    key=lambda item: (
                        item[1].cpu_backup_buffer is None
                        and not self._disk_backup_is_current(item[1])
                    )
                )
            for ptr, data in selected_allocations:
                handle = data.region
                if profile:
                    bytes_by_tag[data.tag] = bytes_by_tag.get(data.tag, 0) + handle.size_bytes
                with self.cpu_backup_lock:
                    cpu_backup_buffer = data.cpu_backup_buffer
                    disk_backup_ref = data.disk_backup_ref
                    has_current_disk_backup = self._disk_backup_is_current(data)
                    if (
                        exact_disk_restore
                        and cpu_backup_buffer is None
                        and not has_current_disk_backup
                    ):
                        deferred_remap_plan.append((ptr, data))
                        continue
                    if cpu_backup_buffer is not None:
                        self._require_valid_cpu_backup(data)
                        # Validation, remap, and the H2D state transition are one
                        # critical section. This prevents reclamation from
                        # interleaving with restore and also
                        # avoids remapping before a known-invalid backup fails.
                        data.cpu_backup_state = CpuBackupState.RESTORING_H2D
                    elif has_current_disk_backup:
                        if self.disk_backup_store is None or disk_backup_ref is None:
                            raise RuntimeError("exact disk backup restore is not configured")
                        data.disk_backup_state = DiskBackupState.RESTORING
                    else:
                        disk_backup_ref = None
                    map_started_at = time.perf_counter()
                    self.backend.map(handle)
                    map_elapsed = time.perf_counter() - map_started_at
                    create_map_s += map_elapsed
                    if has_current_disk_backup:
                        disk_create_map_s += map_elapsed
                    if cpu_backup_buffer is not None:
                        size_in_bytes = cpu_backup_buffer.size_bytes
                        cpu_ptr = cpu_backup_buffer.address
                        copy_started_at = time.perf_counter()
                        if async_stream is None:
                            self.backend.copy(ptr, cpu_ptr, size_in_bytes)
                            copy_h2d_s += time.perf_counter() - copy_started_at
                        else:
                            if async_started_at is None:
                                async_started_at = time.perf_counter()
                            async_stream.submit(0, cpu_backup_buffer.view(), ptr)
                            copy_enqueue_s += time.perf_counter() - copy_started_at
                            async_restored.append(data)
                        # _require_valid_cpu_backup() and the H2D copy run under
                        # the same lock, so successful restore cannot reach an
                        # invalid state here.
                        if async_stream is None:
                            data.cpu_backup_state = CpuBackupState.CACHE_ONLY
                        if profile:
                            cpu_restored_bytes_by_tag[data.tag] = (
                                cpu_restored_bytes_by_tag.get(data.tag, 0) + size_in_bytes
                            )
                    elif disk_backup_ref is not None:
                        disk_plan.append((ptr, data, disk_backup_ref))
                    data.resident = True
            if async_stream is not None:
                wait_started = time.perf_counter()
                async_stream.synchronize()
                copy_wait_s = time.perf_counter() - wait_started
                if async_started_at is not None:
                    cpu_restore_pipeline_s = time.perf_counter() - async_started_at
                with self.cpu_backup_lock:
                    for data in async_restored:
                        data.cpu_backup_state = CpuBackupState.CACHE_ONLY
            if disk_plan:
                if (
                    self.disk_backup_store is None
                    or self.disk_staging_buffers is None
                    or self.disk_restore_stream is None
                ):
                    raise RuntimeError("exact disk backup restore is not configured")
                restore_stream = self.disk_restore_stream
                staging = tuple(tensor.view() for tensor in self.disk_staging_buffers)

                def consume(chunk: memoryview, destination: int, slot: int) -> None:
                    restore_stream.submit(slot, chunk, destination)

                def wait_for_consume(slot: int) -> float:
                    return restore_stream.wait(slot)

                try:
                    disk_stats = self.disk_backup_store.restore_segments_pipelined(
                        [
                            DiskRestoreSegment(ref=ref, destination_ptr=ptr)
                            for ptr, _data, ref in disk_plan
                        ],
                        staging,
                        consume,
                        wait_for_consume,
                        hash_workers=self.disk_hash_workers,
                    )
                except BaseException:
                    self.disk_backup_read_errors += 1
                    # A memcpy may have reached the restore stream even when
                    # recording or synchronizing its completion event failed.
                    # Event state is then insufficient to fence the pinned
                    # staging lifetime, so conservatively drain the whole
                    # stream before propagating RECOVERY_REQUIRED.
                    try:
                        restore_stream.synchronize()
                    except BaseException as fence_error:
                        raise RuntimeError(
                            "failed to fence the exact disk restore stream"
                        ) from fence_error
                    raise
                disk_read_s = float(disk_stats["read_s"])
                disk_hash_s = float(disk_stats["hash_worker_s"])
                disk_copy_h2d_s = float(disk_stats["consume_device_s"])
                disk_copy_enqueue_s = float(disk_stats["consume_s"])
                disk_copy_wait_s = float(disk_stats["consume_wait_s"])
                disk_pipeline_wall_s = float(disk_stats["wall_s"])
                disk_bytes = int(disk_stats["bytes_read"])
                self.disk_backup_read_bytes_total += disk_bytes
                for _ptr, data, ref in disk_plan:
                    data.disk_backup_state = DiskBackupState.CACHE_ONLY
                    disk_restored_bytes_by_tag[data.tag] = (
                        disk_restored_bytes_by_tag.get(data.tag, 0) + ref.size_bytes
                    )
            for _ptr, data in deferred_remap_plan:
                with self.cpu_backup_lock:
                    map_started_at = time.perf_counter()
                    self.backend.map(data.region)
                    data.resident = True
                    map_elapsed = time.perf_counter() - map_started_at
                    create_map_s += map_elapsed
                    deferred_create_map_s += map_elapsed
                if profile:
                    remapped_without_backup_bytes_by_tag[data.tag] = (
                        remapped_without_backup_bytes_by_tag.get(data.tag, 0)
                        + data.region.size_bytes
                    )
        except BaseException:
            with self.cpu_backup_lock:
                self.residency_state = ResidencyState.RECOVERY_REQUIRED
            raise
        finally:
            if async_stream is not None:
                try:
                    # Fence even an uncertain enqueue before exposing buffers to
                    # reclaim. On failure keep RESTORING leases and fail closed.
                    async_stream.close()
                except BaseException:
                    self.residency_state = ResidencyState.RECOVERY_REQUIRED
                    raise
        with self.cpu_backup_lock:
            # Required backups restored above are now cache-only and may satisfy
            # an obligation deferred while wake held them non-evictable.
            self._cpu_backup_reclaimable_generation += 1
            self.sleeping_tags.difference_update(selected_tags)
            self.sleeping_exact_restore_tags.difference_update(selected_tags)
            if not self.sleeping_tags:
                self.residency_state = ResidencyState.AWAKE
                self.partial_sleep = False
            self._report_cpu_backup_usage_locked()
        self.cpu_backup_coordinator.flush()
        self._release_reclaimable_cpu_backups()

        for tag in bytes_by_tag:
            cpu_bytes = cpu_restored_bytes_by_tag.get(tag, 0)
            disk_bytes = disk_restored_bytes_by_tag.get(tag, 0)
            if cpu_bytes and disk_bytes:
                restore_source_by_tag[tag] = "mixed"
            elif cpu_bytes:
                restore_source_by_tag[tag] = "cpu"
            elif disk_bytes:
                restore_source_by_tag[tag] = "disk"

        if profile and disk_restored_bytes_by_tag:
            self.record_event(
                "exact_disk_restore",
                disk_read_bytes=sum(disk_restored_bytes_by_tag.values()),
                disk_read_s=disk_read_s,
                disk_hash_s=disk_hash_s,
                disk_hash_worker_s=disk_hash_s,
                disk_copy_h2d_s=disk_copy_h2d_s,
                disk_copy_enqueue_s=disk_copy_enqueue_s,
                disk_copy_wait_s=disk_copy_wait_s,
                disk_pipeline_wall_s=disk_pipeline_wall_s,
                disk_pipeline_depth=4,
                disk_hash_workers=self.disk_hash_workers,
                source_medium="disk",
                fallback=False,
            )
        if profile:
            coordinator_stats = self.cpu_backup_coordinator.get_profile_fields()
            self.record_event(
                "allocator_wake_up",
                tags=tags,
                allocation_count=sum(
                    1 for data in self.allocations.values() if tags is None or data.tag in tags
                ),
                bytes=sum(bytes_by_tag.values()),
                bytes_by_tag=bytes_by_tag,
                restored_bytes_by_tag=cpu_restored_bytes_by_tag,
                cpu_restored_bytes_by_tag=cpu_restored_bytes_by_tag,
                disk_restored_bytes_by_tag=disk_restored_bytes_by_tag,
                exact_disk_read_s=disk_read_s,
                exact_disk_hash_s=disk_hash_s,
                exact_disk_hash_worker_s=disk_hash_s,
                exact_disk_copy_h2d_s=disk_copy_h2d_s,
                exact_disk_copy_enqueue_s=disk_copy_enqueue_s,
                exact_disk_copy_wait_s=disk_copy_wait_s,
                exact_disk_pipeline_wall_s=disk_pipeline_wall_s,
                exact_disk_create_map_s=disk_create_map_s,
                exact_disk_deferred_create_map_s=deferred_create_map_s,
                exact_disk_pipeline_depth=(4 if disk_restored_bytes_by_tag else 0),
                exact_disk_hash_workers=(
                    self.disk_hash_workers if disk_restored_bytes_by_tag else 0
                ),
                restore_source_by_tag=restore_source_by_tag,
                remapped_without_backup_bytes_by_tag=(remapped_without_backup_bytes_by_tag),
                create_map_s=create_map_s,
                copy_h2d_s=None if self.config.async_cpu_restore else copy_h2d_s,
                async_cpu_restore=self.config.async_cpu_restore,
                cpu_restore_pipeline_s=cpu_restore_pipeline_s,
                cpu_copy_enqueue_s=copy_enqueue_s,
                cpu_copy_wait_s=copy_wait_s,
                cpu_backup_release_count=(self.cpu_backup_release_count),
                cpu_backup_release_bytes=(self.cpu_backup_release_bytes),
                cpu_backup_host_cache_flush_count=(self.cpu_backup_host_cache_flush_count),
                cpu_backup_host_cache_flush_errors=(self.cpu_backup_host_cache_flush_errors),
                **coordinator_stats,
                latency_s=time.perf_counter() - started_at,
            )

    def get_cpu_backup_pool_stats(self) -> dict[str, int | bool]:
        active_backup_bytes = 0
        active_backup_count = 0
        valid_backup_bytes = 0
        valid_backup_count = 0
        with self.cpu_backup_lock:
            for data in self.allocations.values():
                if data.cpu_backup_buffer is None:
                    continue
                active_backup_count += 1
                active_bytes = data.cpu_backup_buffer.size_bytes
                active_backup_bytes += active_bytes
                if data.cpu_backup_valid:
                    valid_backup_count += 1
                    valid_backup_bytes += active_bytes
            free_bytes = sum(
                size * len(tensors) for size, tensors in self.cpu_backup_pool.free_tensors.items()
            )
            free_tensor_count = sum(
                len(tensors) for tensors in self.cpu_backup_pool.free_tensors.values()
            )
            reserved_bytes = self.cpu_backup_pool.reserved_bytes
            pin_memory = self.cpu_backup_pool.pin_memory
        with self.cpu_backup_host_cache_flush_lock:
            host_cache_flush_count = self.cpu_backup_host_cache_flush_count
            host_cache_flush_errors = self.cpu_backup_host_cache_flush_errors
        return {
            "pin_memory": pin_memory,
            "reserved_bytes": reserved_bytes,
            "free_bytes": free_bytes,
            "free_tensor_count": free_tensor_count,
            "active_backup_bytes": active_backup_bytes,
            "active_backup_count": active_backup_count,
            "valid_backup_bytes": valid_backup_bytes,
            "valid_backup_count": valid_backup_count,
            "host_cache_flush_count": host_cache_flush_count,
            "host_cache_flush_errors": host_cache_flush_errors,
        }
