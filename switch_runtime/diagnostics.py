"""Best-effort diagnostics, retaining the existing benchmark schema."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def record_event(path: Path | None, phase: str, **payload: Any) -> None:
    if path is None:
        return
    event = {
        "schema": "switch.vllm.sleep-backup-profile",
        "schema_version": 1,
        "diagnostic_only": True,
        "ts": time.time(),
        "monotonic_s": time.perf_counter(),
        "pid": os.getpid(),
        "phase": phase,
        **payload,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, sort_keys=True) + "\n")
    except (OSError, TypeError, ValueError):
        logger.warning("Failed to write sleep/backup diagnostic event")
