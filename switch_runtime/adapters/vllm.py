"""vLLM adapter. All vLLM imports stay in this optional integration module."""

import os
import sys
import uuid
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .. import API_VERSION
from ..config import RuntimeConfig
from ..contracts import MemoryRegion, SavePolicy
from ..runtime import BackupRuntime, ResidencyState


def config_from_environment() -> RuntimeConfig:
    def boolean(name: str, default: str) -> bool:
        value = os.environ.get(name, default).lower().strip()
        if value not in {"0", "1", "false", "true"}:
            raise ValueError(f"{name} must be a boolean")
        return value in {"1", "true"}

    profile = os.environ.get("VLLM_SLEEP_PROFILE_PATH")
    return RuntimeConfig(
        disk_enabled=boolean("VLLM_EXACT_DISK_BACKUP_ENABLED", "0"),
        disk_root=Path(os.environ.get("VLLM_EXACT_DISK_BACKUP_DIR", "~/.cache/vllm/backup")),
        chunk_bytes=int(os.environ.get("VLLM_EXACT_DISK_BACKUP_CHUNK_BYTES", str(16 * 1024**2))),
        direct_io=boolean("VLLM_EXACT_DISK_BACKUP_DIRECT_IO", "1"),
        coordinator_mode=os.environ.get("VLLM_CPU_BACKUP_COORDINATOR", "").strip().lower(),
        coordinator_url=os.environ.get("VLLM_CPU_BACKUP_COORDINATOR_URL"),
        coordinator_timeout_s=float(os.environ.get("VLLM_CPU_BACKUP_COORDINATOR_TIMEOUT_S", "1")),
        coordinator_client_id=os.environ.get("VLLM_CPU_BACKUP_COORDINATOR_CLIENT_ID"),
        model_id=os.environ.get("VLLM_CPU_BACKUP_COORDINATOR_MODEL_ID"),
        poll_interval_s=float(os.environ.get("VLLM_CPU_BACKUP_COORDINATOR_POLL_INTERVAL_S", "0.1")),
        engine="vllm-switch",
        profile_path=Path(profile) if profile else None,
    )


class VllmSleepBackend:
    def __init__(self) -> None:
        import torch
        from vllm.device_allocator.cumem import create_and_map, unmap_and_release

        from ..devices.cuda import CudaMemoryBackend

        device = torch.cuda.current_device()
        self.runtime = BackupRuntime(
            CudaMemoryBackend(create_and_map, unmap_and_release, device),
            config_from_environment(),
        )
        self.checkpoint_pending = False
        self.initialized = False
        self.needs_prebackup = False

    def register(self, handle: Any, tag: str) -> None:
        self.runtime.register(
            MemoryRegion(
                region_id=uuid.uuid4().hex,
                address=handle[2],
                size_bytes=handle[1],
                device=handle[0],
                tag=tag,
                handle=handle,
                policy=SavePolicy.DISCARD if tag == "kv_cache" else SavePolicy.SNAPSHOT,
            )
        )

    def unregister(self, address: int) -> None:
        self.runtime.unregister(address)

    def sleep(self, offload_tags=None) -> None:
        self.runtime.sleep(
            offload_tags,
            skip_prepare=self.runtime.residency_state == ResidencyState.SLEEP_PREPARED,
        )

    def wake_up(self, tags=None) -> None:
        self.runtime.wake_up(tags)

    def freeze(self, worker: Any) -> None:
        """Treat allocations overlapping model buffers conservatively as mutable."""
        model = worker.model_runner.get_model()

        def ranges(tensors):
            result = []
            for tensor in tensors:
                if tensor.device.type != "cuda":
                    continue
                storage = tensor.untyped_storage()
                result.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
            return result

        parameters = ranges(model.parameters())
        buffers = ranges(model.buffers())

        def overlaps(region, spans):
            return any(
                start < region.address + region.size_bytes and end > region.address
                for start, end in spans
            )

        with self.runtime.lifecycle_lock, self.runtime.cpu_backup_lock:
            for data in self.runtime.allocations.values():
                region = data.region
                if region.tag == "weights":
                    policy = SavePolicy.SNAPSHOT
                    if overlaps(region, parameters) and not overlaps(region, buffers):
                        policy = SavePolicy.IMMUTABLE
                    data.region = replace(region, policy=policy)
        self.initialized = True

    def prepare(self, worker: Any) -> None:
        self.freeze(worker)
        self.runtime.prepare_cpu_backup("weights")
        self.runtime.prepare_disk_backup("weights")
        self.needs_prebackup = False


class VllmProvider:
    def validate_config(self, config: Any) -> None:
        if not config.model_config.enable_sleep_mode:
            raise ValueError("switch sleep backend requires --enable-sleep-mode")
        if config.parallel_config.enable_eplb:
            raise ValueError("switch requires fixed weights; EPLB is unsupported")
        if config.weight_transfer_config is not None:
            raise ValueError("switch requires fixed weights; weight transfer is unsupported")
        if config.lora_config is not None:
            raise ValueError("switch does not yet support LoRA storage classification")
        from vllm.platforms import current_platform

        if not current_platform.is_cuda():
            raise ValueError("switch currently supports the CUDA memory backend")

    def create_backend(self, allocator: Any) -> VllmSleepBackend:
        return VllmSleepBackend()

    @staticmethod
    def backend() -> VllmSleepBackend | None:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.instance
        return None if allocator is None else allocator.sleep_backend

    @contextmanager
    def worker_context(self, worker: Any, operation: str, **call):
        if operation == "mutation":
            raise RuntimeError("switch uses fixed weights; this model mutation is unsupported")
        backend = self.backend()
        if operation == "load":
            if backend is not None and backend.initialized:
                raise RuntimeError("switch uses fixed weights; loading a new model is unsupported")
            yield
            return
        if backend is None:
            raise RuntimeError("switch worker has no process-local sleep backend")
        runtime = backend.runtime
        with runtime.lifecycle_lock:
            if runtime.closed or runtime.residency_state == ResidencyState.RECOVERY_REQUIRED:
                raise RuntimeError("switch lifecycle requires engine recovery")
            args, kwargs = call.get("args", ()), call.get("kwargs", {})
            level = kwargs.get("level", args[0] if args else 1)
            if operation == "inference":
                runtime.require_awake("execute inference")
                if backend.checkpoint_pending:
                    raise RuntimeError("L2 checkpoint reconstruction has not completed")
            elif operation == "sleep":
                if backend.checkpoint_pending:
                    raise RuntimeError("cannot sleep before checkpoint reconstruction")
                if level not in (1, 2):
                    raise ValueError("unsupported sleep level")
                if level == 2:
                    runtime.reset_snapshots()
            elif operation == "reload":
                if not backend.checkpoint_pending or args or kwargs:
                    raise RuntimeError("reload is only allowed for the original L2 checkpoint")
                runtime.require_tags_awake("reload checkpoint", {"weights"})
            try:
                yield
                if operation == "sleep" and level == 2:
                    backend.checkpoint_pending = True
                    backend.needs_prebackup = True
                elif operation == "reload":
                    runtime.backend.synchronize()
                    backend.freeze(worker)
                    backend.checkpoint_pending = False
                    if runtime.residency_state == ResidencyState.AWAKE:
                        backend.prepare(worker)
                elif operation == "wake" and not backend.checkpoint_pending:
                    if runtime.residency_state == ResidencyState.AWAKE and backend.needs_prebackup:
                        backend.prepare(worker)
            except BaseException:
                if (
                    operation == "sleep"
                    and runtime.residency_state == ResidencyState.SLEEP_PREPARED
                ):
                    runtime.abort_sleep_prepare("weights")
                elif operation in {"reload", "wake"} or (
                    operation == "sleep" and runtime.residency_state != ResidencyState.AWAKE
                ):
                    runtime.residency_state = ResidencyState.RECOVERY_REQUIRED
                raise

    def worker_event(self, worker: Any, event: str, **kwargs) -> Any:
        backend = self.backend()
        if backend is None:
            if event == "close":
                return None
            raise RuntimeError("switch worker has no process-local sleep backend")
        runtime = backend.runtime
        if backend.checkpoint_pending and event in {
            "ready",
            "prepare",
            "prepare_cpu",
            "prepare_disk",
        }:
            raise RuntimeError("cannot publish a snapshot before checkpoint reconstruction")
        if event == "ready":
            backend.prepare(worker)
        elif event == "prepare":
            if backend.checkpoint_pending:
                raise RuntimeError("cannot prepare sleep before checkpoint reconstruction")
            return runtime.prepare_sleep("weights")
        elif event == "abort":
            runtime.abort_sleep_prepare("weights")
        elif event == "prepare_cpu":
            return runtime.prepare_cpu_backup("weights")
        elif event == "prepare_disk":
            return runtime.prepare_disk_backup("weights")
        elif event == "reclaim":
            target = kwargs.get("target_free_bytes")
            return runtime.reclaim(
                runtime.cpu_backup_pool.reserved_bytes if target is None else target
            )
        elif event == "stats":
            try:
                package_version = version("vllm-switch-controller")
            except PackageNotFoundError:
                package_version = None
            engine_module = sys.modules.get("vllm")
            return {
                "pid": os.getpid(),
                "runtime_module": BackupRuntime.__module__,
                "runtime_identity": {
                    "schema_version": 1,
                    "pid": os.getpid(),
                    "sleep_backend": "switch",
                    "provider_api_version": API_VERSION,
                    "engine_module_path": getattr(engine_module, "__file__", None),
                    "runtime_module_path": sys.modules[BackupRuntime.__module__].__file__,
                    "provider_module_path": __file__,
                    "package_version": package_version,
                    "runtime_config": {
                        "disk_enabled": runtime.config.disk_enabled,
                        "direct_io": runtime.config.direct_io,
                        "chunk_bytes": runtime.config.chunk_bytes,
                        "coordinator_mode": runtime.config.coordinator_mode,
                    },
                },
                **runtime.get_cpu_backup_pool_stats(),
            }
        elif event == "close":
            runtime.close()
        else:
            raise ValueError(f"unknown sleep extension event: {event}")


_provider = VllmProvider()


def register() -> None:
    # General plugins run in API/core/worker processes. Registration allocates no CUDA state.
    # The lightweight controller wheel can also coexist with an unpatched upstream engine.
    if os.environ.get("VLLM_SLEEP_BACKEND", "native") == "native":
        return
    from vllm.device_allocator.sleep_provider import register_sleep_provider

    register_sleep_provider("switch", _provider, API_VERSION)
