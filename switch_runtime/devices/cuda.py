"""CUDA operations in the engine process; importing this module requires PyTorch."""

import ctypes
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from ..contracts import MemoryRegion


class TorchHostBuffer:
    def __init__(self, size_bytes: int, pin_memory: bool):
        self.tensor = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=pin_memory)

    @property
    def address(self) -> int:
        return self.tensor.data_ptr()

    @property
    def size_bytes(self) -> int:
        return self.tensor.numel()

    def view(self) -> memoryview:
        return memoryview(self.tensor.numpy())


class CudaLibrary:
    def __init__(self) -> None:
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libcudart.so" in line and line.split()[-1].startswith("/")
        }
        if len(paths) != 1:
            raise RuntimeError("expected one PyTorch CUDA runtime library in this process")
        self.lib = ctypes.CDLL(paths.pop())
        ptr = ctypes.c_void_p
        uint = ctypes.c_uint
        signatures = {
            "cudaMemcpy": [ptr, ptr, ctypes.c_size_t, ctypes.c_int],
            "cudaMemcpyAsync": [ptr, ptr, ctypes.c_size_t, ctypes.c_int, ptr],
            "cudaStreamCreateWithFlags": [ctypes.POINTER(ptr), uint],
            "cudaStreamSynchronize": [ptr],
            "cudaStreamDestroy": [ptr],
            "cudaEventCreateWithFlags": [ctypes.POINTER(ptr), uint],
            "cudaEventRecord": [ptr, ptr],
            "cudaEventSynchronize": [ptr],
            "cudaEventDestroy": [ptr],
            "cudaEventElapsedTime": [ctypes.POINTER(ctypes.c_float), ptr, ptr],
        }
        self.functions: dict[str, Any] = {}
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int
            self.functions[name] = fn
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def call(self, name: str, *args: Any) -> None:
        result = self.functions[name](*args)
        if result:
            raise RuntimeError(f"{name}: {self.lib.cudaGetErrorString(result).decode()}")


class CudaRestoreStream:
    def __init__(self, library: CudaLibrary, slots: int):
        self.library = library
        self.stream = ctypes.c_void_p()
        self.events: list[ctypes.c_void_p] = []
        library.call("cudaStreamCreateWithFlags", ctypes.byref(self.stream), 1)
        try:
            for _ in range(slots * 2):
                event = ctypes.c_void_p()
                library.call("cudaEventCreateWithFlags", ctypes.byref(event), 0)
                self.events.append(event)
        except BaseException:
            self.close()
            raise

    def submit(self, slot: int, source: memoryview, destination: int) -> None:
        start, end = self.events[2 * slot : 2 * slot + 2]
        address = ctypes.addressof(ctypes.c_char.from_buffer(source))
        self.library.call("cudaEventRecord", start, self.stream)
        self.library.call("cudaMemcpyAsync", destination, address, len(source), 4, self.stream)
        self.library.call("cudaEventRecord", end, self.stream)

    def wait(self, slot: int) -> float:
        start, end = self.events[2 * slot : 2 * slot + 2]
        self.library.call("cudaEventSynchronize", end)
        elapsed = ctypes.c_float()
        self.library.call("cudaEventElapsedTime", ctypes.byref(elapsed), start, end)
        return elapsed.value / 1000.0

    def synchronize(self) -> None:
        self.library.call("cudaStreamSynchronize", self.stream)

    def close(self) -> None:
        self.synchronize()
        for event in self.events:
            self.library.call("cudaEventDestroy", event)
        self.events.clear()
        self.library.call("cudaStreamDestroy", self.stream)


class CudaMemoryBackend:
    pin_memory = True

    def __init__(
        self,
        map_handle: Callable[[Any], None],
        unmap_handle: Callable[[Any], None],
        device: int,
    ):
        self.map_handle = map_handle
        self.unmap_handle = unmap_handle
        self.device = device
        self.library = CudaLibrary()

    def allocate_host(self, size_bytes: int) -> TorchHostBuffer:
        return TorchHostBuffer(size_bytes, self.pin_memory)

    def copy(self, destination: int, source: int, size_bytes: int) -> None:
        self.library.call("cudaMemcpy", destination, source, size_bytes, 4)

    def map(self, region: MemoryRegion) -> None:
        self.map_handle(region.handle)

    def unmap(self, region: MemoryRegion) -> None:
        self.unmap_handle(region.handle)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def empty_device_cache(self) -> None:
        torch.cuda.empty_cache()

    def empty_host_cache(self) -> None:
        empty_cache = getattr(torch._C, "_host_emptyCache", None)
        if empty_cache is None:
            raise RuntimeError("PyTorch does not expose _host_emptyCache")
        empty_cache()

    def restore_stream(self, slots: int) -> CudaRestoreStream:
        return CudaRestoreStream(self.library, slots)
