import argparse
import json
import os
import signal
from pathlib import Path
from typing import Any

from vllm_switch_controller.processes import (
    ProcessIdentity,
    read_process_identity,
    wait_process_group_empty,
)


def _parse_pid_file(path: Path) -> dict[str, ProcessIdentity]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read PID file {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("PID file is not a supported launcher ownership record")
    processes = payload.get("processes")
    if not isinstance(processes, dict):
        raise ValueError("PID file processes must be an object")
    records: dict[str, ProcessIdentity] = {}
    for name, record in processes.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError("PID file contains an invalid process record")
        try:
            identity = ProcessIdentity(
                pid=int(record["pid"]),
                pgid=int(record["pgid"]),
                start_time_ticks=int(record["start_time_ticks"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"PID file contains an invalid record for {name}") from exc
        if identity.pid <= 0 or identity.pgid <= 0 or identity.start_time_ticks <= 0:
            raise ValueError(f"PID file contains non-positive identity fields for {name}")
        records[name] = identity
    return records


def _verify_owned_group(name: str, expected: ProcessIdentity) -> bool:
    actual = read_process_identity(expected.pid)
    if actual is None:
        if expected.pgid == os.getpgrp():
            print(f"refusing to signal {name}: process group matches the stop command")
            return False
        # The launcher is the process-group leader. If it exits first, the
        # recorded PGID still owns surviving descendants; PGID reuse is not
        # possible while those members remain.
        return True
    if actual != expected:
        print(
            f"refusing to signal {name}: process identity changed "
            f"(expected {expected}, observed {actual})"
        )
        return False
    if expected.pgid == os.getpgrp():
        print(f"refusing to signal {name}: process group matches the stop command")
        return False
    return True


def stop_owned_processes(pid_file: str | Path, *, timeout_s: float) -> bool:
    """Stop verified launcher-owned process groups.

    The ownership file is retained when any identity cannot be verified so an
    operator can inspect it. Missing processes count as already stopped.
    """

    path = Path(pid_file)
    if not path.exists():
        print(f"pid file does not exist: {path}")
        return True
    try:
        records = _parse_pid_file(path)
    except ValueError as exc:
        print(f"refusing unsafe cleanup: {exc}")
        return False

    verified: list[tuple[str, ProcessIdentity]] = []
    for name, expected in records.items():
        if not _verify_owned_group(name, expected):
            return False
        if read_process_identity(expected.pid) is not None or not wait_process_group_empty(
            expected.pgid, 0
        ):
            verified.append((name, expected))

    success = True
    for name, identity in verified:
        if not _verify_owned_group(name, identity):
            success = False
            continue
        try:
            print(f"terminating {name} process group pgid={identity.pgid}")
            os.killpg(identity.pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    for name, identity in verified:
        if wait_process_group_empty(identity.pgid, timeout_s):
            continue
        # Re-check the leader identity before escalating. If it changed, do not
        # signal a potentially unrelated group that reused the numeric PGID.
        if not _verify_owned_group(name, identity):
            success = False
            continue
        try:
            print(f"killing {name} process group pgid={identity.pgid}")
            os.killpg(identity.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if not wait_process_group_empty(identity.pgid, timeout_s):
            print(f"process group did not exit: {name} pgid={identity.pgid}")
            success = False

    if success:
        path.unlink(missing_ok=True)
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop launcher-owned vLLM process groups")
    parser.add_argument("--pid-file", default="pids.json")
    parser.add_argument("--timeout-s", type=float, default=30)
    args = parser.parse_args()
    if args.timeout_s < 0:
        parser.error("--timeout-s must be non-negative")
    if not stop_owned_processes(args.pid_file, timeout_s=args.timeout_s):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
