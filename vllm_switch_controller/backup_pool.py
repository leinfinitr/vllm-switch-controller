from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClientBackupUsage:
    """Aggregated pinned CPU backup accounting for one vLLM process.

    The controller intentionally does not track per-tensor backup ids. vLLM owns
    the local tensors and decides which cache-only/invalid/free-local buffers to
    release when the controller asks for a target byte count. Exact disk backup
    content also remains worker-local; the controller trusts only explicit
    aggregate reports that required RAM already has a usable disk restore source.
    """

    client_id: str
    protocol_version: int = 1
    capabilities: tuple[str, ...] = ()
    pid: int | None = None
    engine: str = "unknown"
    model_id: str | None = None
    gpu_uuid: str | None = None
    total_bytes: int = 0
    released_bytes_total: int = 0
    release_counter_enabled: bool | None = None
    required_for_restore_bytes: int = 0
    cache_only_bytes: int = 0
    invalid_bytes: int = 0
    free_local_bytes: int = 0
    disk_backup_current_bytes: int = 0
    disk_backup_reserved_bytes: int = 0
    ram_reclaimable_with_disk_bytes: int = 0
    pending_release_bytes: int = 0
    requested_release_bytes_total: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def evictable_bytes(self) -> int:
        # A configured or reserved disk tier is telemetry only. Required RAM is
        # safe to request only when the worker reports it in the field below.
        return self.ram_reclaimable_without_disk_bytes + self.ram_reclaimable_with_disk_bytes

    @property
    def ram_reclaimable_without_disk_bytes(self) -> int:
        return self.cache_only_bytes + self.invalid_bytes + self.free_local_bytes

    def snapshot(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "protocol_version": self.protocol_version,
            "capabilities": list(self.capabilities),
            "pid": self.pid,
            "engine": self.engine,
            "model_id": self.model_id,
            "gpu_uuid": self.gpu_uuid,
            "total_bytes": self.total_bytes,
            "released_bytes_total": self.released_bytes_total,
            "required_for_restore_bytes": self.required_for_restore_bytes,
            "cache_only_bytes": self.cache_only_bytes,
            "invalid_bytes": self.invalid_bytes,
            "free_local_bytes": self.free_local_bytes,
            "disk_backup_current_bytes": self.disk_backup_current_bytes,
            "disk_backup_reserved_bytes": self.disk_backup_reserved_bytes,
            "ram_reclaimable_without_disk_bytes": self.ram_reclaimable_without_disk_bytes,
            "ram_reclaimable_with_disk_bytes": self.ram_reclaimable_with_disk_bytes,
            "evictable_bytes": self.evictable_bytes,
            "pending_release_bytes": self.pending_release_bytes,
            "requested_release_bytes_total": self.requested_release_bytes_total,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


class BackupPoolState:
    """Metadata-only coordinator for process-local CPU backup pools.

    The controller tracks only per-client aggregate bytes. It never decides which
    local tensor or disk file to release; it only asks a vLLM process to free a
    target number of reclaimable RAM bytes according to global cap and
    model-priority policy.
    """

    def __init__(
        self,
        global_cap_bytes: int | None = None,
        *,
        model_priorities: dict[str, int] | None = None,
        default_model_priority: int = 0,
    ) -> None:
        self.clients: dict[str, ClientBackupUsage] = {}
        self.request_epoch = uuid.uuid4().hex
        self.global_cap_bytes = global_cap_bytes
        self.model_priorities = model_priorities or {}
        self.default_model_priority = default_model_priority

    def model_priority(self, model_id: str | None) -> int:
        if model_id is None:
            return self.default_model_priority
        return self.model_priorities.get(model_id, self.default_model_priority)

    def register_client(
        self,
        client_id: str,
        *,
        protocol_version: int = 1,
        capabilities: list[str] | tuple[str, ...] = (),
        pid: int | None = None,
        engine: str = "unknown",
        model_id: str | None = None,
        gpu_uuid: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ClientBackupUsage:
        now = time.time()
        record = self.clients.get(client_id)
        if record is None:
            record = ClientBackupUsage(
                client_id=client_id,
                protocol_version=protocol_version,
                capabilities=tuple(sorted(capabilities)),
                pid=pid,
                engine=engine,
                model_id=model_id,
                gpu_uuid=gpu_uuid,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
            )
            self.clients[client_id] = record
            return record
        if record.pid is not None and pid is not None and record.pid != pid:
            raise ValueError(
                f"client_id {client_id!r} is already bound to a different process incarnation"
            )
        if record.protocol_version != protocol_version:
            raise ValueError("protocol_version must remain stable per process incarnation")
        incoming_capabilities = tuple(sorted(capabilities))
        if record.capabilities != incoming_capabilities:
            raise ValueError("capabilities must remain stable per process incarnation")
        if pid is not None:
            record.pid = pid
        record.engine = engine
        record.model_id = model_id
        record.gpu_uuid = gpu_uuid
        record.metadata = metadata or {}
        record.updated_at = now
        return record

    def report_usage(
        self,
        *,
        client_id: str,
        protocol_version: int = 1,
        capabilities: list[str] | tuple[str, ...] = (),
        total_bytes: int,
        required_for_restore_bytes: int,
        cache_only_bytes: int,
        invalid_bytes: int,
        free_local_bytes: int,
        disk_backup_current_bytes: int = 0,
        disk_backup_reserved_bytes: int = 0,
        ram_reclaimable_with_disk_bytes: int = 0,
        released_bytes_total: int | None = None,
        pid: int | None = None,
        engine: str = "unknown",
        model_id: str | None = None,
        gpu_uuid: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ClientBackupUsage:
        if ram_reclaimable_with_disk_bytes > required_for_restore_bytes:
            raise ValueError(
                "ram_reclaimable_with_disk_bytes cannot exceed required_for_restore_bytes"
            )
        if (
            ram_reclaimable_with_disk_bytes > 0
            and disk_backup_current_bytes == 0
            and disk_backup_reserved_bytes == 0
        ):
            raise ValueError("ram_reclaimable_with_disk_bytes requires a reported disk source")
        existing = self.clients.get(client_id)
        counter_enabled = released_bytes_total is not None
        if (
            existing is not None
            and existing.release_counter_enabled is not None
            and existing.release_counter_enabled != counter_enabled
        ):
            raise ValueError("released_bytes_total presence must remain stable per client")
        if (
            existing is not None
            and released_bytes_total is not None
            and released_bytes_total < existing.released_bytes_total
        ):
            raise ValueError("released_bytes_total must be monotonic per client")
        record = self.register_client(
            client_id,
            protocol_version=protocol_version,
            capabilities=capabilities,
            pid=pid,
            engine=engine,
            model_id=model_id,
            gpu_uuid=gpu_uuid,
            metadata=metadata,
        )
        if released_bytes_total is None:
            # Compatibility for clients predating the cumulative ack field.
            released_bytes = max(record.total_bytes - total_bytes, 0)
        else:
            # This delta survives an immediate same-sized reallocation before
            # latest-wins usage is flushed.
            released_bytes = released_bytes_total - record.released_bytes_total
        record.pending_release_bytes = max(record.pending_release_bytes - released_bytes, 0)
        record.total_bytes = total_bytes
        if released_bytes_total is not None:
            record.released_bytes_total = released_bytes_total
        if record.release_counter_enabled is None:
            record.release_counter_enabled = counter_enabled
        record.required_for_restore_bytes = required_for_restore_bytes
        record.cache_only_bytes = cache_only_bytes
        record.invalid_bytes = invalid_bytes
        record.free_local_bytes = free_local_bytes
        record.disk_backup_current_bytes = disk_backup_current_bytes
        record.disk_backup_reserved_bytes = disk_backup_reserved_bytes
        record.ram_reclaimable_with_disk_bytes = ram_reclaimable_with_disk_bytes
        # A request remains an outstanding obligation across local state
        # transitions. New clients acknowledge allocator release with the
        # monotonic counter above; legacy clients fall back to an observed
        # footprint drop. Cancelling an obligation when cache-only becomes
        # required would allow duplicate bytes when it becomes evictable again.
        record.updated_at = time.time()
        return record

    def request_release(self, client_id: str, target_free_bytes: int) -> int:
        if target_free_bytes <= 0:
            return 0
        record = self.clients.get(client_id)
        if record is None:
            return 0
        available = max(record.evictable_bytes - record.pending_release_bytes, 0)
        requested = min(target_free_bytes, available)
        if requested <= 0:
            return 0
        record.pending_release_bytes += requested
        # A monotonic total makes GET idempotent. If its HTTP response is lost,
        # the worker can retry and derive the same unseen delta without the
        # controller reissuing (and potentially duplicating) an obligation.
        record.requested_release_bytes_total += requested
        return requested

    def release_request_snapshot(self, client_id: str) -> tuple[int, int]:
        record = self.clients.get(client_id)
        if record is None:
            return 0, 0
        return record.requested_release_bytes_total, record.pending_release_bytes

    def request_release_bytes(self, target_free_bytes: int) -> dict[str, int]:
        """Queue a global target using priority, age, then largest footprint."""
        bytes_to_free = max(target_free_bytes, 0)
        if bytes_to_free <= 0:
            return {}
        queued: dict[str, int] = {}
        candidates = sorted(
            (
                record
                for record in self.clients.values()
                if record.evictable_bytes > record.pending_release_bytes
            ),
            key=lambda record: (
                self.model_priority(record.model_id),
                record.updated_at,
                -(record.evictable_bytes - record.pending_release_bytes),
            ),
        )
        for record in candidates:
            if bytes_to_free <= 0:
                break
            available = record.evictable_bytes - record.pending_release_bytes
            requested = self.request_release(record.client_id, min(bytes_to_free, available))
            if requested > 0:
                queued[record.client_id] = requested
                bytes_to_free -= requested
        return queued

    def maybe_enqueue_release_requests(self) -> dict[str, int]:
        """Queue requests that move total backup usage toward the hard cap."""
        if self.global_cap_bytes is None:
            return {}
        return self.request_release_bytes(self.stats()["over_cap_bytes"])

    def stats(self) -> dict[str, Any]:
        total_bytes = sum(record.total_bytes for record in self.clients.values())
        required_bytes = sum(record.required_for_restore_bytes for record in self.clients.values())
        cache_only_bytes = sum(record.cache_only_bytes for record in self.clients.values())
        invalid_bytes = sum(record.invalid_bytes for record in self.clients.values())
        free_local_bytes = sum(record.free_local_bytes for record in self.clients.values())
        disk_backup_current_bytes = sum(
            record.disk_backup_current_bytes for record in self.clients.values()
        )
        disk_backup_reserved_bytes = sum(
            record.disk_backup_reserved_bytes for record in self.clients.values()
        )
        ram_reclaimable_without_disk_bytes = cache_only_bytes + invalid_bytes + free_local_bytes
        ram_reclaimable_with_disk_bytes = sum(
            record.ram_reclaimable_with_disk_bytes for record in self.clients.values()
        )
        evictable_bytes = ram_reclaimable_without_disk_bytes + ram_reclaimable_with_disk_bytes
        pending_release_bytes = sum(
            record.pending_release_bytes for record in self.clients.values()
        )
        return {
            "client_count": len(self.clients),
            "total_bytes": total_bytes,
            "global_cap_bytes": self.global_cap_bytes,
            "default_model_priority": self.default_model_priority,
            "model_priorities": self.model_priorities,
            "over_cap_bytes": (
                max(total_bytes - self.global_cap_bytes - pending_release_bytes, 0)
                if self.global_cap_bytes is not None
                else 0
            ),
            "required_for_restore_bytes": required_bytes,
            "cache_only_bytes": cache_only_bytes,
            "invalid_bytes": invalid_bytes,
            "free_local_bytes": free_local_bytes,
            "disk_backup_current_bytes": disk_backup_current_bytes,
            "disk_backup_reserved_bytes": disk_backup_reserved_bytes,
            "disk_backup_client_count": sum(
                record.disk_backup_current_bytes > 0 or record.disk_backup_reserved_bytes > 0
                for record in self.clients.values()
            ),
            "ram_reclaimable_without_disk_bytes": ram_reclaimable_without_disk_bytes,
            "ram_reclaimable_with_disk_bytes": ram_reclaimable_with_disk_bytes,
            "evictable_bytes": evictable_bytes,
            "pending_release_bytes": pending_release_bytes,
            "pending_release_request_count": sum(
                record.pending_release_bytes > 0 for record in self.clients.values()
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "request_epoch": self.request_epoch,
            "stats": self.stats(),
            "clients": {client_id: record.snapshot() for client_id, record in self.clients.items()},
        }
