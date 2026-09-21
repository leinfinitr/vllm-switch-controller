# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .config import RuntimeConfig

BACKUP_PROTOCOL_VERSION = 1
BACKUP_CAPABILITIES = (
    "cumulative-release-v1",
    "exact-disk-accounting-v1",
    "process-incarnation-v1",
    "released-bytes-total-v1",
)


class CpuBackupCoordinator:
    """No-op interface for the optional process-local backup coordinator."""

    def report_usage(self, usage: dict[str, int]) -> None:
        pass

    def flush(self) -> None:
        pass

    def poll_release_request(self) -> int:
        return 0

    def is_enabled(self) -> bool:
        return False

    def get_profile_fields(self) -> dict[str, Any]:
        return {"cpu_backup_coordinator_enabled": False}


class HttpCpuBackupCoordinator(CpuBackupCoordinator):
    """Aggregate control-plane client; pinned tensors always remain process-local.

    Usage updates are latest-wins. Both the allocator thread and the background
    release poller can flush, so pending state is protected without holding the
    lock during network I/O. A transient HTTP failure retains an unsent snapshot
    for retry but never disables local sleep/wake behavior.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float,
        client_id: str,
        model_id: str | None,
        engine: str = "vllm-switch",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.client_id = client_id
        self.model_id = model_id
        self.engine = engine
        self.pid = os.getpid()

        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        # Keep extraction and delivery of latest-wins snapshots ordered. The
        # state lock remains free during HTTP so allocator transitions can still
        # replace the pending slot without blocking.
        self._flush_lock = threading.Lock()
        # Response parsing and cumulative epoch state updates form one poll
        # transaction. Serializing only socket I/O would allow an older response
        # to update state after a newer response has already been applied.
        self._poll_lock = threading.Lock()
        self._pending_usage: dict[str, Any] | None = None
        self._requests_succeeded = 0
        self._request_errors = 0
        self._release_polls = 0
        self._release_bytes_received = 0
        self._release_request_epoch: str | None = None
        self._release_bytes_seen = 0
        # Control-plane endpoints are normally loopback/private. Explicitly
        # bypass environment proxies, which can otherwise misroute localhost.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._register()

    def _post_json(self, path: str, payload: Any) -> None:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with (
            self._request_lock,
            self._opener.open(request, timeout=self.timeout_s) as response,
        ):
            response.read()

    def _record_success(self) -> None:
        with self._lock:
            self._requests_succeeded += 1

    def _record_error(self) -> None:
        with self._lock:
            self._request_errors += 1

    def _register(self) -> None:
        try:
            self._post_json(
                "/admin/cpu-backup/register",
                {
                    "protocol_version": BACKUP_PROTOCOL_VERSION,
                    "capabilities": list(BACKUP_CAPABILITIES),
                    "client_id": self.client_id,
                    "pid": self.pid,
                    "engine": self.engine,
                    "model_id": self.model_id,
                    "metadata": {"hostname": socket.gethostname()},
                },
            )
            self._record_success()
        except (OSError, urllib.error.URLError, TimeoutError):
            self._record_error()

    def report_usage(self, usage: dict[str, int]) -> None:
        """Coalesce allocator transitions into the newest aggregate snapshot."""
        controller_fields = {
            "total_bytes",
            "released_bytes_total",
            "required_for_restore_bytes",
            "cache_only_bytes",
            "invalid_bytes",
            "free_local_bytes",
            "disk_backup_current_bytes",
            "disk_backup_reserved_bytes",
            "ram_reclaimable_with_disk_bytes",
        }
        metadata: dict[str, Any] = {
            key: value for key, value in usage.items() if key not in controller_fields
        }
        snapshot: dict[str, Any] = {
            key: value for key, value in usage.items() if key in controller_fields
        }
        if metadata:
            snapshot["metadata"] = metadata
        with self._lock:
            self._pending_usage = snapshot

    def flush(self) -> None:
        with self._flush_lock:
            with self._lock:
                usage = self._pending_usage
                self._pending_usage = None
            if usage is None:
                return

            payload = {
                "protocol_version": BACKUP_PROTOCOL_VERSION,
                "capabilities": list(BACKUP_CAPABILITIES),
                "client_id": self.client_id,
                "pid": self.pid,
                "engine": self.engine,
                "model_id": self.model_id,
                **usage,
            }
            try:
                self._post_json("/admin/cpu-backup/usage", payload)
                self._record_success()
            except (OSError, urllib.error.URLError, TimeoutError):
                self._record_error()
                with self._lock:
                    # Never replace a newer transition reported during this
                    # request; a later ordered flush will deliver that snapshot.
                    if self._pending_usage is None:
                        self._pending_usage = usage

    def poll_release_request(self) -> int:
        """Poll and consume a target number of locally safe bytes to release."""
        with self._poll_lock:
            return self._poll_release_request_once()

    def _poll_release_request_once(self) -> int:
        self.flush()
        client_id = urllib.parse.quote(self.client_id, safe="")
        request = urllib.request.Request(
            f"{self.base_url}/admin/cpu-backup/release-requests/{client_id}",
            method="GET",
        )
        try:
            with (
                self._request_lock,
                self._opener.open(request, timeout=self.timeout_s) as response,
            ):
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("release response must be a JSON object")
            request_epoch = payload.get("request_epoch")
            requested_total = payload.get("requested_release_bytes_total")
            if not isinstance(request_epoch, str) or not request_epoch:
                raise ValueError("request_epoch must be a non-empty string")
            requested_total = int(requested_total or 0)
            if requested_total < 0:
                raise ValueError("requested_release_bytes_total must be non-negative")
        except (
            OSError,
            urllib.error.URLError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            self._record_error()
            return 0

        with self._lock:
            if request_epoch != self._release_request_epoch:
                # A new controller epoch restarts the cumulative sequence.
                self._release_request_epoch = request_epoch
                self._release_bytes_seen = 0
            elif requested_total < self._release_bytes_seen:
                # A cumulative sequence cannot move backwards within one epoch.
                # Ignore stale/malformed responses without replaying bytes.
                self._request_errors += 1
                return 0
            target_free_bytes = requested_total - self._release_bytes_seen
            self._release_bytes_seen = requested_total
            self._requests_succeeded += 1
            self._release_polls += 1
            self._release_bytes_received += target_free_bytes
        return target_free_bytes

    def is_enabled(self) -> bool:
        # Availability is represented by request_errors. Returning True here
        # keeps retries alive when the controller starts after the vLLM worker.
        return True

    def get_profile_fields(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cpu_backup_coordinator_enabled": True,
                "cpu_backup_coordinator_backend": "http",
                "cpu_backup_coordinator_requests_succeeded": self._requests_succeeded,
                "cpu_backup_coordinator_request_errors": self._request_errors,
                "cpu_backup_coordinator_pending_usage": int(self._pending_usage is not None),
                "cpu_backup_coordinator_release_polls": self._release_polls,
                "cpu_backup_coordinator_release_bytes_received": (self._release_bytes_received),
            }


def make_cpu_backup_coordinator(config: RuntimeConfig) -> CpuBackupCoordinator:
    base_url = config.coordinator_url
    mode = config.coordinator_mode.strip().lower()
    if not base_url or mode in {"", "none", "noop", "disabled"}:
        return CpuBackupCoordinator()
    if mode not in {"daemon", "http"}:
        raise ValueError(f"unsupported VLLM_CPU_BACKUP_COORDINATOR mode: {mode!r}")

    timeout_s = config.coordinator_timeout_s
    if timeout_s <= 0:
        raise ValueError("VLLM_CPU_BACKUP_COORDINATOR_TIMEOUT_S must be positive")
    client_id_prefix = config.coordinator_client_id or f"vllm-switch-{socket.gethostname()}"
    # The configured value is a logical prefix, not a reusable process identity.
    # A process-incarnation suffix prevents a restarted worker from inheriting
    # the predecessor's cumulative release counter or pending obligations.
    client_id = f"{client_id_prefix}-{os.getpid()}-{time.time_ns()}"
    model_id = config.model_id
    return HttpCpuBackupCoordinator(
        base_url,
        timeout_s=timeout_s,
        client_id=client_id,
        model_id=model_id,
        engine=config.engine,
    )
