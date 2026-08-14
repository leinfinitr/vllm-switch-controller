"""Linux process identity and process-group helpers for launcher ownership."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessIdentity:
    """Identity fields that remain stable for one Linux process incarnation."""

    pid: int
    pgid: int
    start_time_ticks: int


def read_process_identity(pid: int) -> ProcessIdentity | None:
    """Read a Linux process identity from ``/proc/<pid>/stat``.

    PID alone is unsafe for delayed cleanup because the kernel can reuse it. The
    start-time tick identifies the process incarnation, and the PGID identifies
    the launcher-created session that owns its descendants.
    """

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close_paren = stat.rfind(")")
        if close_paren < 0:
            return None
        fields = stat[close_paren + 2 :].split()
        return ProcessIdentity(
            pid=pid,
            pgid=int(fields[2]),
            start_time_ticks=int(fields[19]),
        )
    except (
        FileNotFoundError,
        ProcessLookupError,
        OSError,
        ValueError,
        IndexError,
    ):
        return None


def process_group_has_members(pgid: int) -> bool:
    """Return whether a process group has any non-zombie members."""

    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = path.read_text(encoding="utf-8")
            close_paren = stat.rfind(")")
            if close_paren < 0:
                continue
            fields = stat[close_paren + 2 :].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                return True
        except (
            FileNotFoundError,
            ProcessLookupError,
            OSError,
            ValueError,
            IndexError,
        ):
            continue
    return False


def wait_process_group_empty(pgid: int, timeout_s: float) -> bool:
    """Wait up to ``timeout_s`` for a process group to have no live members."""

    deadline = time.monotonic() + max(timeout_s, 0)
    while time.monotonic() < deadline:
        if not process_group_has_members(pgid):
            return True
        time.sleep(0.05)
    return not process_group_has_members(pgid)
