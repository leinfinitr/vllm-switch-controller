import asyncio
import json
from types import SimpleNamespace

import pytest

from scripts import launch_vllm_pool
from vllm_switch_controller.processes import ProcessIdentity


@pytest.mark.asyncio
async def test_post_and_probe_share_one_transition_deadline(monkeypatch):
    async def fake_post(*_args, **_kwargs):
        await asyncio.sleep(0.04)

    async def fake_wait(*_args, **_kwargs):
        await asyncio.sleep(0.04)

    monkeypatch.setattr(launch_vllm_pool.EngineClient, "sleep", fake_post)
    monkeypatch.setattr(launch_vllm_pool.EngineClient, "wait_until_sleeping", fake_wait)
    with pytest.raises(TimeoutError, match="lifecycle transition timed out"):
        await launch_vllm_pool.post_and_wait(
            "http://a",
            "/sleep",
            expected=True,
            timeout_s=0.05,
            params={"level": 1},
        )


@pytest.mark.asyncio
async def test_prepare_pool_sleeps_every_backend_before_waking_startup(tmp_path, monkeypatch):
    events: list[str] = []
    config = SimpleNamespace(
        models={
            "a": SimpleNamespace(
                backend_url="http://a",
                engine="vllm",
                sleep_level=1,
                launch_command=None,
                env={},
                cwd=None,
                wake_tags=["weights", "kv_cache"],
            ),
            "b": SimpleNamespace(
                backend_url="http://b",
                engine="vllm",
                sleep_level=1,
                launch_command=None,
                env={},
                cwd=None,
                wake_tags=None,
            ),
        },
        controller=SimpleNamespace(startup_awake_model="a", switch_timeout_s=5),
    )

    async def fake_health(url, timeout_s, **kwargs):
        events.append(f"health:{url[-1]}")

    transitions = []

    async def fake_post_and_wait(url, path, *, expected, timeout_s=600, params=None, **kwargs):
        transitions.append((url, path, params))
        events.append(f"post:{url[-1]}:{path}")
        events.append(f"probe:{url[-1]}:{expected}")

    monkeypatch.setattr(launch_vllm_pool, "wait_health", fake_health)
    monkeypatch.setattr(launch_vllm_pool, "post_and_wait", fake_post_and_wait)

    pid_file = tmp_path / "nested" / "pids.json"
    await launch_vllm_pool.prepare_pool(
        config,
        pid_file=pid_file,
        skip_launch=True,
    )

    assert events == [
        "health:a",
        "post:a:/sleep",
        "probe:a:True",
        "health:b",
        "post:b:/sleep",
        "probe:b:True",
        "post:a:/wake_up",
        "probe:a:False",
    ]
    assert json.loads(pid_file.read_text()) == {"schema_version": 1, "processes": {}}
    assert transitions[-1] == (
        "http://a",
        "/wake_up",
        [("tags", "weights"), ("tags", "kv_cache")],
    )


@pytest.mark.asyncio
async def test_prepare_pool_cleans_up_started_processes_on_failure(tmp_path, monkeypatch):
    events = []

    class FakeProcess:
        next_pid = 100

        def __init__(self, *_args, **_kwargs):
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1

        def wait(self, timeout=None):
            events.append(f"wait:{self.pid}")

        def poll(self):
            events.append(f"poll:{self.pid}")

    config = SimpleNamespace(
        models={
            "a": SimpleNamespace(
                backend_url="http://a",
                engine="vllm",
                sleep_level=1,
                launch_command=["server-a"],
                env={},
                cwd=None,
                wake_tags=None,
            ),
            "b": SimpleNamespace(
                backend_url="http://b",
                engine="vllm",
                sleep_level=1,
                launch_command=["server-b"],
                env={},
                cwd=None,
                wake_tags=None,
            ),
        },
        controller=SimpleNamespace(startup_awake_model="a", switch_timeout_s=5),
    )

    async def fake_health(url, timeout_s, **kwargs):
        if url == "http://b":
            raise TimeoutError("boom")

    async def fake_transition(*_args, **_kwargs):
        return None

    monkeypatch.setattr(launch_vllm_pool.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        launch_vllm_pool,
        "read_process_identity",
        lambda pid: ProcessIdentity(pid, pid, pid * 10),
    )
    monkeypatch.setattr(
        launch_vllm_pool.os,
        "killpg",
        lambda pid, sig: events.append(f"killpg:{pid}:{sig.name}"),
    )
    monkeypatch.setattr(launch_vllm_pool, "wait_process_group_empty", lambda *_args: True)
    monkeypatch.setattr(launch_vllm_pool, "wait_health", fake_health)
    monkeypatch.setattr(launch_vllm_pool, "post_and_wait", fake_transition)
    pid_file = tmp_path / "pids.json"
    with pytest.raises(TimeoutError, match="boom"):
        await launch_vllm_pool.prepare_pool(config, pid_file=pid_file, skip_launch=False)

    assert events == [
        "killpg:101:SIGTERM",
        "killpg:100:SIGTERM",
        "wait:101",
        "poll:101",
        "wait:100",
        "poll:100",
    ]
    assert not pid_file.exists()


@pytest.mark.asyncio
async def test_prepare_pool_invalidates_stale_success_pid_file_before_failure(
    tmp_path, monkeypatch
):
    config = SimpleNamespace(
        models={
            "a": SimpleNamespace(
                launch_command=None,
                env={},
                cwd=None,
                backend_url="http://a",
                engine="vllm",
                sleep_level=1,
                wake_tags=None,
            )
        },
        controller=SimpleNamespace(startup_awake_model="a", switch_timeout_s=5),
    )

    async def fail_health(_url, timeout_s, **kwargs):
        raise TimeoutError("boom")

    monkeypatch.setattr(launch_vllm_pool, "wait_health", fail_health)
    pid_file = tmp_path / "pids.json"
    pid_file.write_text("STALE-SUCCESS", encoding="utf-8")

    with pytest.raises(TimeoutError, match="boom"):
        await launch_vllm_pool.prepare_pool(config, pid_file=pid_file, skip_launch=True)

    assert not pid_file.exists()
