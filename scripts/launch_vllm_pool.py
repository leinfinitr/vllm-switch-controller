import argparse
import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from controller.config import load_config
from controller.processes import read_process_identity, wait_process_group_empty


async def wait_health(url: str, timeout_s: float = 600) -> None:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        while time.time() < deadline:
            try:
                response = await client.get(f"{url}/health")
                if 200 <= response.status_code < 300:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError(f"backend did not become healthy: {url}")


async def post(
    url: str,
    path: str,
    params: Any = None,
    timeout_s: float = 600,
) -> None:
    async with httpx.AsyncClient(timeout=timeout_s, trust_env=False) as client:
        response = await client.post(f"{url}{path}", params=params)
        response.raise_for_status()


async def wait_sleep_state(
    url: str,
    expected: bool,
    timeout_s: float = 600,
    poll_interval_s: float = 0.1,
) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=min(timeout_s, 30), trust_env=False) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{url}/is_sleeping")
                response.raise_for_status()
                value = response.json().get("is_sleeping")
                if not isinstance(value, bool):
                    raise RuntimeError(f"invalid /is_sleeping response from {url}")
                if value is expected:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(poll_interval_s)
    state = "sleeping" if expected else "awake"
    raise TimeoutError(f"backend did not become {state}: {url}")


async def post_and_wait(
    url: str,
    path: str,
    *,
    expected: bool,
    timeout_s: float,
    params: Any = None,
) -> None:
    """Apply one lifecycle operation and verify its state under one deadline."""
    try:
        async with asyncio.timeout(timeout_s):
            await post(url, path, params, timeout_s=timeout_s)
            await wait_sleep_state(url, expected, timeout_s=timeout_s)
    except TimeoutError as exc:
        raise TimeoutError(f"lifecycle transition timed out: {url}{path}") from exc


async def prepare_pool(config, *, pid_file: str | Path, skip_launch: bool) -> None:
    output = Path(pid_file)
    # A PID file is a success/ownership record. A failed rerun must not leave
    # an older record that appears to describe the failed attempt.
    output.unlink(missing_ok=True)
    output.with_suffix(output.suffix + ".tmp").unlink(missing_ok=True)
    process_records: dict[str, dict[str, int]] = {}
    processes: list[subprocess.Popen] = []
    try:
        for name, spec in config.models.items():
            if spec.launch_command and not skip_launch:
                env = os.environ.copy()
                env.update(spec.env)
                env.setdefault("VLLM_SERVER_DEV_MODE", "1")
                process = subprocess.Popen(
                    spec.launch_command, env=env, cwd=spec.cwd, start_new_session=True
                )
                processes.append(process)
                identity = read_process_identity(process.pid)
                if identity is None:
                    raise RuntimeError(f"cannot read process identity for {name} pid={process.pid}")
                if identity.pgid != process.pid:
                    raise RuntimeError(
                        f"launcher process group mismatch for {name}: "
                        f"pid={process.pid} pgid={identity.pgid}"
                    )
                process_records[name] = {
                    "pid": identity.pid,
                    "pgid": identity.pgid,
                    "start_time_ticks": identity.start_time_ticks,
                }
                print(f"launched {name} pid={process.pid}")
            else:
                print(f"using existing backend for {name}: {spec.backend_url}")
            await wait_health(spec.backend_url, timeout_s=config.controller.switch_timeout_s)
            print(f"sleeping {name}")
            await post_and_wait(
                spec.backend_url,
                "/sleep",
                params={"level": spec.sleep_level},
                expected=True,
                timeout_s=config.controller.switch_timeout_s,
            )

        startup = config.controller.startup_awake_model
        if startup:
            print(f"waking startup model {startup}")
            wake_tags = config.models[startup].wake_tags
            wake_params = [("tags", tag) for tag in wake_tags] if wake_tags is not None else None
            await post_and_wait(
                config.models[startup].backend_url,
                "/wake_up",
                params=wake_params,
                expected=False,
                timeout_s=config.controller.switch_timeout_s,
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"schema_version": 1, "processes": process_records}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(output)
        print(f"wrote pid file {pid_file}")
    except BaseException:
        for process in reversed(processes):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in reversed(processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            if not wait_process_group_empty(process.pid, 5):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                wait_process_group_empty(process.pid, 5)
            process.poll()
        raise


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Launch configured vLLM pool sequentially")
    parser.add_argument("--config", default="configs/models.launcher.example.yaml")
    parser.add_argument("--pid-file", default="pids.json")
    parser.add_argument(
        "--skip-launch",
        action="store_true",
        help="Only sleep/wake already running servers",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    await prepare_pool(config, pid_file=args.pid_file, skip_launch=args.skip_launch)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
