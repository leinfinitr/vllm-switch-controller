"""Engine-independent memory contracts; addresses never cross the control plane."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class SavePolicy(StrEnum):
    IMMUTABLE = "immutable"
    SNAPSHOT = "snapshot_each_sleep"
    DISCARD = "discard"


@dataclass(frozen=True)
class MemoryRegion:
    region_id: str
    address: int
    size_bytes: int
    device: int
    tag: str
    handle: Any
    policy: SavePolicy = SavePolicy.SNAPSHOT

    def __post_init__(self) -> None:
        if not self.region_id or self.address <= 0 or self.size_bytes <= 0:
            raise ValueError("a memory region needs an identity, address, and positive size")


class HostBuffer(Protocol):
    @property
    def address(self) -> int: ...

    @property
    def size_bytes(self) -> int: ...

    def view(self) -> memoryview: ...


class RestoreStream(Protocol):
    def submit(self, slot: int, source: memoryview, destination: int) -> None: ...

    def wait(self, slot: int) -> float: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...


class MemoryBackend(Protocol):
    """Operations use the engine's existing device context and virtual addresses."""

    pin_memory: bool

    def allocate_host(self, size_bytes: int) -> HostBuffer: ...

    def copy(self, destination: int, source: int, size_bytes: int) -> None: ...

    def map(self, region: MemoryRegion) -> None: ...

    def unmap(self, region: MemoryRegion) -> None: ...

    def synchronize(self) -> None: ...

    def empty_device_cache(self) -> None: ...

    def empty_host_cache(self) -> None: ...

    def restore_stream(self, slots: int) -> RestoreStream: ...
