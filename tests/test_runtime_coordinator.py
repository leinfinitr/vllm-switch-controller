# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from switch_runtime.adapters.vllm import config_from_environment
from switch_runtime.coordinator import (
    HttpCpuBackupCoordinator,
    make_cpu_backup_coordinator,
)


class _EventHandler(BaseHTTPRequestHandler):
    posts: list[tuple[str, dict]] = []
    release_requests: dict[str, int] = {}
    request_epoch = "epoch-a"

    def do_POST(self):
        length = int(self.headers["content-length"])
        body = self.rfile.read(length)
        self.posts.append((self.path, json.loads(body)))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def do_GET(self):
        client_id = self.path.rsplit("/", 1)[-1]
        payload = {
            "ok": True,
            "request_epoch": self.request_epoch,
            "requested_release_bytes_total": self.release_requests.get(client_id, 0),
        }
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, format, *args):
        return


def _start_server():
    _EventHandler.posts = []
    _EventHandler.release_requests = {}
    _EventHandler.request_epoch = "epoch-a"
    server = HTTPServer(("127.0.0.1", 0), _EventHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    return server, thread


def test_http_cpu_backup_coordinator_reports_usage_snapshot():
    server, thread = _start_server()
    try:
        coordinator = HttpCpuBackupCoordinator(
            f"http://127.0.0.1:{server.server_port}",
            timeout_s=1.0,
            client_id="client-a",
            model_id="model-a",
        )
        coordinator.report_usage(
            {
                "total_bytes": 4096,
                "released_bytes_total": 1024,
                "required_for_restore_bytes": 1024,
                "cache_only_bytes": 2048,
                "invalid_bytes": 0,
                "free_local_bytes": 1024,
                "disk_backup_current_bytes": 1024,
                "disk_backup_reserved_bytes": 1024,
                "ram_reclaimable_with_disk_bytes": 1024,
                "disk_exact_bytes": 1024,
                "disk_read_errors_total": 2,
            }
        )
        coordinator.flush()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert [path for path, _ in _EventHandler.posts] == [
        "/admin/cpu-backup/register",
        "/admin/cpu-backup/usage",
    ]
    usage = _EventHandler.posts[1][1]
    assert usage["client_id"] == "client-a"
    assert usage["model_id"] == "model-a"
    assert usage["total_bytes"] == 4096
    assert usage["released_bytes_total"] == 1024
    assert usage["cache_only_bytes"] == 2048
    assert usage["disk_backup_current_bytes"] == 1024
    assert usage["ram_reclaimable_with_disk_bytes"] == 1024
    assert usage["metadata"]["disk_exact_bytes"] == 1024
    assert usage["metadata"]["disk_read_errors_total"] == 2
    assert usage["protocol_version"] == 1
    assert "exact-disk-accounting-v1" in usage["capabilities"]
    fields = coordinator.get_profile_fields()
    assert fields["cpu_backup_coordinator_requests_succeeded"] == 2
    assert fields["cpu_backup_coordinator_request_errors"] == 0

    registration = _EventHandler.posts[0][1]
    assert registration["protocol_version"] == 1
    assert "process-incarnation-v1" in registration["capabilities"]
    assert registration["engine"] == "vllm-switch"
    assert usage["engine"] == "vllm-switch"
    assert registration["metadata"]["hostname"]


def test_http_cpu_backup_coordinator_polls_release_bytes():
    server, thread = _start_server()
    try:
        _EventHandler.release_requests = {"client-a": 8192}
        coordinator = HttpCpuBackupCoordinator(
            f"http://127.0.0.1:{server.server_port}",
            timeout_s=1.0,
            client_id="client-a",
            model_id="model-a",
        )
        first = coordinator.poll_release_request()
        duplicate = coordinator.poll_release_request()
        _EventHandler.release_requests["client-a"] = 4096
        rollback = coordinator.poll_release_request()
        _EventHandler.release_requests["client-a"] = 12_288
        delta = coordinator.poll_release_request()
        _EventHandler.request_epoch = "epoch-b"
        _EventHandler.release_requests["client-a"] = 2048
        restarted = coordinator.poll_release_request()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert first == 8192
    assert duplicate == 0
    assert rollback == 0
    assert delta == 4096
    assert restarted == 2048
    fields = coordinator.get_profile_fields()
    assert fields["cpu_backup_coordinator_release_polls"] == 4
    assert fields["cpu_backup_coordinator_release_bytes_received"] == 14_336
    assert fields["cpu_backup_coordinator_request_errors"] == 1


def test_http_cpu_backup_coordinator_keeps_pending_usage_on_flush_error():
    coordinator = HttpCpuBackupCoordinator(
        "http://127.0.0.1:9",
        timeout_s=0.01,
        client_id="client-a",
        model_id="model-a",
    )
    coordinator.report_usage(
        {
            "total_bytes": 4096,
            "required_for_restore_bytes": 0,
            "cache_only_bytes": 4096,
            "invalid_bytes": 0,
            "free_local_bytes": 0,
        }
    )
    coordinator.flush()

    fields = coordinator.get_profile_fields()
    # A controller may start after the worker. Transient failures are visible,
    # but must not permanently stop the background poller from retrying.
    assert fields["cpu_backup_coordinator_enabled"] is True
    assert fields["cpu_backup_coordinator_request_errors"] >= 1
    assert fields["cpu_backup_coordinator_pending_usage"] == 1


def test_http_cpu_backup_coordinator_orders_concurrent_flushes(monkeypatch):
    monkeypatch.setattr(HttpCpuBackupCoordinator, "_register", lambda self: None)
    coordinator = HttpCpuBackupCoordinator(
        "http://unused",
        timeout_s=0.1,
        client_id="client-a",
        model_id="model-a",
    )
    first_started = threading.Event()
    release_first = threading.Event()
    second_lock_attempted = threading.Event()
    second_sent = threading.Event()
    sent_totals: list[int] = []

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()
            self.count_lock = threading.Lock()
            self.enter_count = 0

        def __enter__(self):
            with self.count_lock:
                self.enter_count += 1
                if self.enter_count == 2:
                    second_lock_attempted.set()
            self.lock.acquire()

        def __exit__(self, *_args):
            self.lock.release()

    monkeypatch.setattr(coordinator, "_flush_lock", ObservedLock())

    def post_json(path, payload):
        if path.endswith("/register"):
            return
        if payload["total_bytes"] == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        else:
            second_sent.set()
        sent_totals.append(payload["total_bytes"])

    monkeypatch.setattr(coordinator, "_post_json", post_json)
    coordinator.report_usage({"total_bytes": 1})
    first = threading.Thread(target=coordinator.flush)
    first.start()
    assert first_started.wait(timeout=2)

    coordinator.report_usage({"total_bytes": 2})
    second = threading.Thread(target=coordinator.flush)
    second.start()
    # Without an ordered flush transaction, the second snapshot reaches the
    # transport while the first is still blocked and can overtake it.
    assert second_lock_attempted.wait(timeout=2)
    assert not second_sent.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert sent_totals == [1, 2]


def test_http_cpu_backup_coordinator_orders_complete_poll_transactions(monkeypatch):
    monkeypatch.setattr(HttpCpuBackupCoordinator, "_register", lambda self: None)
    coordinator = HttpCpuBackupCoordinator(
        "http://unused",
        timeout_s=0.1,
        client_id="client-a",
        model_id="model-a",
    )
    first_started = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    call_count = 0

    def poll_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
            return 1
        second_entered.set()
        return 2

    monkeypatch.setattr(coordinator, "_poll_release_request_once", poll_once)
    results: dict[str, int] = {}
    first = threading.Thread(
        target=lambda: results.__setitem__("first", coordinator.poll_release_request())
    )
    second = threading.Thread(
        target=lambda: results.__setitem__("second", coordinator.poll_release_request())
    )
    first.start()
    assert first_started.wait(timeout=2)
    second.start()
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert results == {"first": 1, "second": 2}


def test_factory_scopes_configured_client_id_to_worker_process(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_BACKUP_COORDINATOR", "daemon")
    monkeypatch.setenv("VLLM_CPU_BACKUP_COORDINATOR_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("VLLM_CPU_BACKUP_COORDINATOR_TIMEOUT_S", "0.01")
    monkeypatch.setenv("VLLM_CPU_BACKUP_COORDINATOR_CLIENT_ID", "logical-model-a")
    coordinator = make_cpu_backup_coordinator(config_from_environment())

    assert isinstance(coordinator, HttpCpuBackupCoordinator)
    assert coordinator.client_id.startswith(f"logical-model-a-{os.getpid()}-")


def test_factory_uses_vllm_switch_default_client_prefix(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_BACKUP_COORDINATOR", "daemon")
    monkeypatch.setenv("VLLM_CPU_BACKUP_COORDINATOR_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("VLLM_CPU_BACKUP_COORDINATOR_TIMEOUT_S", "0.01")
    monkeypatch.delenv("VLLM_CPU_BACKUP_COORDINATOR_CLIENT_ID", raising=False)
    coordinator = make_cpu_backup_coordinator(config_from_environment())

    assert isinstance(coordinator, HttpCpuBackupCoordinator)
    assert coordinator.client_id.startswith("vllm-switch-")
