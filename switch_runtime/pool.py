# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict

from .contracts import HostBuffer, MemoryBackend


class CpuBackupPool:
    """Size-keyed, process-local host storage with explicit logical accounting."""

    def __init__(self, backend: MemoryBackend):
        self.backend = backend
        self.pin_memory = backend.pin_memory
        self.free_tensors: dict[int, list[HostBuffer]] = defaultdict(list)
        self.reserved_bytes = 0

    def acquire(self, size_in_bytes: int) -> tuple[HostBuffer, bool]:
        free_list = self.free_tensors[size_in_bytes]
        if free_list:
            return free_list.pop(), True
        buffer = self.backend.allocate_host(size_in_bytes)
        self.reserved_bytes += buffer.size_bytes
        return buffer, False

    def release_to_free_list(self, buffer: HostBuffer) -> None:
        self.free_tensors[buffer.size_bytes].append(buffer)

    def discard(self, buffer: HostBuffer) -> int:
        self.reserved_bytes -= buffer.size_bytes
        return buffer.size_bytes

    def discard_free_bytes(self, target_bytes: int) -> int:
        released = 0
        for size, buffers in list(self.free_tensors.items()):
            while buffers and released < target_bytes:
                buffers.pop()
                self.reserved_bytes -= size
                released += size
            if not buffers:
                self.free_tensors.pop(size, None)
            if released >= target_bytes:
                break
        return released
