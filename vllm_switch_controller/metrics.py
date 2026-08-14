import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class RequestMetrics:
    request_id: str
    model: str
    path: str
    arrival_ts: float = field(default_factory=time.time)
    previous_model: str | None = None
    switch_id: str | None = None
    route_class: str = "steady_resident"
    queue_wait_ms: float | None = None
    request_drain_ms: float | None = None
    switch_needed: bool = False
    sleep_latency_ms: float | None = None
    wake_latency_ms: float | None = None
    switch_latency_ms: float | None = None
    response_body_first_byte_ms: float | None = None
    e2e_response_body_first_byte_ms: float | None = None
    e2e_latency_ms: float | None = None
    status_code: int | None = None
    error: str | None = None

    @classmethod
    def new(cls, model: str, path: str, request_id: str | None = None) -> "RequestMetrics":
        return cls(request_id=request_id or str(uuid.uuid4()), model=model, path=path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricsRecorder:
    """Append-only JSONL recorder for controller events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def record(self, metrics: RequestMetrics) -> None:
        line = json.dumps(metrics.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
