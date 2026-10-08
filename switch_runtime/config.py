"""Runtime configuration. Engine adapters own environment-variable translation."""

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    disk_enabled: bool = False
    disk_root: Path = Path("~/.cache/vllm/backup")
    chunk_bytes: int = 16 * 1024**2
    direct_io: bool = True
    coordinator_mode: str = ""
    coordinator_url: str | None = None
    coordinator_timeout_s: float = 1.0
    coordinator_client_id: str | None = None
    model_id: str | None = None
    poll_interval_s: float = 0.1
    engine: str = "unknown"
    profile_path: Path | None = None
    async_cpu_restore: bool = False
    weight_slab_bytes: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.weight_slab_bytes <= 1024**3 or self.weight_slab_bytes % (2 * 1024**2):
            raise ValueError("weight_slab_bytes must be a 2 MiB multiple between zero and 1 GiB")
        if self.chunk_bytes <= 0 or self.chunk_bytes % 4096:
            raise ValueError("chunk_bytes must be a positive 4 KiB multiple")
        if not math.isfinite(self.coordinator_timeout_s) or self.coordinator_timeout_s <= 0:
            raise ValueError("coordinator_timeout_s must be finite and positive")
        if not math.isfinite(self.poll_interval_s):
            raise ValueError("poll_interval_s must be finite")
        if self.coordinator_mode not in {"", "none", "noop", "disabled", "http", "daemon"}:
            raise ValueError("unsupported coordinator mode")
