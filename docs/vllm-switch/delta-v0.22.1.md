# vllm-switch delta from vLLM v0.22.1

The engine fork adds a versioned provider bridge to the native v0.22.1 sleep allocator.
The backup implementation lives in this repository's `switch_runtime` package and runs
inside each GPU worker. The external controller service sends management requests and
aggregate reclaim targets; it does not execute CUDA copies or hold backup payloads.

This is a guide to the current patch, not a historical performance report. Run these in
the engine checkout when reviewing or rebasing it:

```bash
git diff --name-status v0.22.1 HEAD
git diff v0.22.1 HEAD -- vllm tests
# Include local, uncommitted edits:
git diff v0.22.1 -- vllm tests
```

## What changes relative to upstream

| Area | Native v0.22.1 path | Switch provider capability |
| --- | --- | --- |
| Selection | Built-in CuMemAllocator sleep/wake | Explicit plugin selection with API compatibility checks |
| L1 CPU backup | Allocate/copy at sleep; release the backup tensor after wake | Prepare eagerly and reuse published immutable snapshots while retained |
| Mutable model buffers | Captured with their offloaded allocation at sleep | Mixed/buffer allocations are copied at each preparation |
| Restore storage | CPU backup attached to the allocator | Worker-owned CPU pool and optional committed exact disk sources |
| Preparation | Each worker copies and unmaps in its sleep call | Prepare all workers before issuing sleep; abort leases on preparation failure |
| Reclaim | No aggregate CPU-backup coordination in this path | Local reclaim under metadata-only controller byte targets |
| L2 | Discard weights; caller must reconstruct them | Track the reconstruction obligation and reject inference until reload succeeds |
| Diagnostics | Native lifecycle logging | Named identity/stats RPCs plus optional phase/counter JSONL |
| Weight mutation | Upstream engine behavior | Explicit fixed-weight restrictions make snapshot reuse safe within the supported contract |

Upstream already supplies tagged sleep/wake, CUDA virtual-address preservation, model
loading, and CUDA graph capture. The switch path reuses those primitives. It adds backup
policy, storage tiers, lifecycle checks, and their integration; it does not introduce a
new CUDA allocator extension or a new model runner.

## Source map

Paths in the first column are relative to the engine fork. Runtime links point into this
repository. These are the four production files changed from the tag.

| Engine file and functions | Connection to switch_runtime | Resulting behavior |
| --- | --- | --- |
| `vllm/device_allocator/sleep_provider.py`: `SleepProvider`, `register_sleep_provider`, `get_sleep_provider` | [VllmProvider and register](../../switch_runtime/adapters/vllm.py) via the `vllm.general_plugins` entry point | Select a compatible process-local provider; native is the default, an explicitly missing provider fails |
| Same file: `sleep_provider_context`, `sleep_provider_guard` | `VllmProvider.worker_context` | Wrap existing worker methods with preconditions, lifecycle serialization, and completion/failure handling |
| `vllm/device_allocator/cumem.py`: `__init__`, `_python_malloc_callback`, `_python_free_callback` | `VllmSleepBackend` -> [MemoryRegion](../../switch_runtime/contracts.py) -> [BackupRuntime.register/unregister](../../switch_runtime/runtime.py) | Attach a runtime and track allocation lifetimes using fresh region IDs |
| Same file: `sleep`, `wake_up` | `VllmSleepBackend.sleep/wake_up` -> `BackupRuntime` -> [CudaMemoryBackend](../../switch_runtime/devices/cuda.py) | Copy/reuse/lease exact bytes and unmap/remap existing addresses; restore selected tags from CPU or disk |
| `vllm/v1/executor/abstract.py`: `initialize_from_config`, `sleep` | Named `sleep_extension` RPCs -> `VllmProvider.worker_event` | Send readiness after collective warmup; prepare/abort collectively before L1 unmap |
| `vllm/v1/worker/gpu_worker.py`: initialization, guarded methods, `sleep_extension`, backup helpers, `shutdown` | `VllmProvider.validate_config/worker_context/worker_event` | Enforce the fixed-weight contract, forward local runtime commands, expose identity, and close resources |

There are also three added integration tests, an introductory note in the engine README,
and an external-provider section in its sleep-mode guide. The six fork-specific documents
formerly in the engine repository are consolidated into this documentation tree: the
[fork overview](README.md), this delta, and the four [runtime design guides](../runtime.md#design-guides).
Setup and testing details are maintained in the existing integration and testing pages.

## Registration and allocation

```text
VLLM_SLEEP_BACKEND=switch
  -> get_sleep_provider() -> load_general_plugins()
     -> switch_runtime.adapters.vllm.register()
        -> register_sleep_provider("switch", VllmProvider, API_VERSION)
  -> Worker.__init__ -> VllmProvider.validate_config()
  -> CuMemAllocator.__init__ -> VllmProvider.create_backend()
     -> VllmSleepBackend -> BackupRuntime(CudaMemoryBackend, RuntimeConfig)

CUDA allocation/free callback
  -> CuMemAllocator._python_malloc_callback / _python_free_callback
     -> VllmSleepBackend.register / unregister
        -> BackupRuntime.register(MemoryRegion) / unregister(address)
```

Plugin registration can happen in API, executor, and worker processes. Backend creation
happens at allocator construction in the CUDA worker. `VLLM_PLUGINS`, when allowlisted,
must include `switch_runtime`; configure the selector before startup and keep it fixed.
Provider API version 1 is independent of coordinator protocol 1 and disk manifest schema 2.

Each region identifies one allocation incarnation. Reusing a virtual address for a later
allocation cannot reuse the previous allocation's snapshots. `VllmSleepBackend.freeze`
classifies weight allocations after warmup: parameter storage without overlapping model
buffers becomes `IMMUTABLE`; mixed/unclassified weights remain `SNAPSHOT`; KV is
`DISCARD`. This classification does not detect arbitrary writes to parameter storage.

## Startup and L1 sleep/wake

```text
Executor.initialize_from_config
  -> collective initialize_from_config -> collective compile_or_warm_up_model
  -> collective sleep_extension("ready")
     -> VllmProvider.worker_event -> VllmSleepBackend.prepare
        -> freeze -> prepare_cpu_backup("weights") -> prepare_disk_backup("weights")

Executor.sleep(level=1)
  -> collective sleep_extension("prepare")
     -> BackupRuntime.prepare_sleep("weights"): snapshot/reuse and lease
  -> on preparation failure: collective sleep_extension("abort"), then raise
  -> on success: collective Worker.sleep(level=1)
     -> provider "sleep" context -> CuMemAllocator.sleep(("weights",))
        -> VllmSleepBackend.sleep -> BackupRuntime.sleep(skip_prepare=True)

Executor.wake_up(tags)
  -> collective Worker.wake_up(tags)
     -> provider "wake" context -> CuMemAllocator.wake_up(tags)
        -> VllmSleepBackend.wake_up -> BackupRuntime.wake_up(tags)
     -> worker restores saved model buffers / runs KV wake handling as needed
```

Readiness runs after profiling, warmup, and graph capture so initial snapshots reflect
their effects on storage. The executor waits for every preparation call to succeed before
sending sleep. Prepare captures mutable bytes and leases restore sources atomically
against reclaim; abort returns prepared sources to cache-only use without unmapping.
Collective preparation is not rollback for a partial unmap: uncertain mapping failures
require engine recovery. Multi-rank correctness still needs separate validation.

The sleep timer includes preparation. Diagnostic `allocator_prepare_sleep` work is part
of L1 sleep cost even when the later `allocator_sleep` phase makes no D2H copy. Eager
startup preparation is a separate cost. Retaining an immutable snapshot avoids another
copy only while a usable CPU/disk source survives.

Wake restores at the original virtual addresses, keeping captured graph pointers valid.
Tag selection permits staged weights/KV wake; inference remains blocked until all storage
is awake and any L2 reconstruction obligation is satisfied.

## Worker guards

All operations below reach `VllmProvider.worker_context` through
`sleep_provider_guard`. Native mode uses a no-op context.

| Guard | Worker methods | Switch behavior |
| --- | --- | --- |
| `load` | `load_model` | Allow initial loading; reject replacing the initialized fixed model |
| `sleep` | `sleep` | Check level and reconstruction; reset snapshots for L2 and mark checkpoint reconstruction pending after success |
| `wake` | `wake_up` | Track partial/full residency and prepare fresh backups after completed L2 reconstruction and full wake |
| `reload` | `reload_weights` | Permit only the original L2 checkpoint, with no override arguments and weight mappings awake |
| `inference` | `execute_model`, `sample_tokens` | Require awake residency and completed reconstruction; reject recovery-required state |
| `mutation` | `apply_model`, `update_config`, `add_lora`, `remove_lora`, `pin_lora`, `init_weight_transfer_engine`, `start_weight_update`, `update_weights`, `finish_weight_update`, `elastic_ep_execute` | Reject before touching model state because content changes are not tracked |

The provider's lifecycle lock covers guarded host calls; the memory backend synchronizes
device work where needed. Reclaim separately shares the runtime's CPU backup lock with
snapshot publication and restore-source leasing.

For L2, [VllmControlAdapter.resume](../../vllm_switch_controller/backends.py) orders:

```text
wake(weights) -> collective reload_weights() -> wake(remaining/KV) -> verify awake
```

The worker's native sleep/wake bodies still preserve model buffers. The provider resets
old snapshots, tracks the checkpoint obligation, and blocks inference until reload succeeds.
After reconstruction and full wake, it classifies storage again and prepares new snapshots.
A failed reload or uncertain restore enters `RECOVERY_REQUIRED`.

## Explicit commands and diagnostics

These names are worker RPC methods/events, not new HTTP endpoints. The existing
collective RPC endpoint can address `sleep_extension` by name.

| Worker call | Adapter/runtime operation |
| --- | --- |
| `sleep_extension("identity")` | Inspect the actual allocator; report native identity or the switch adapter's `runtime_identity` |
| `sleep_extension("stats")` | Return runtime counters plus API/package versions, module paths, PID, and effective settings |
| `prepare_weight_cpu_backup()` | `worker_event("prepare_cpu")` -> `BackupRuntime.prepare_cpu_backup("weights")` |
| `prepare_weight_disk_backup()` | `worker_event("prepare_disk")` -> `BackupRuntime.prepare_disk_backup("weights")` |
| `demote_weight_cpu_backup_to_disk(target_free_bytes)` | `worker_event("reclaim")` -> `BackupRuntime.reclaim(target)`; None targets current reserved pool bytes |
| `shutdown()` | `worker_event("close")` -> `BackupRuntime.close()` before model-runner teardown |

Identity reads the instantiated allocator without creating one; an environment selector
alone does not prove which runtime ran. The other extension events return None in native
mode. Named RPCs avoid arbitrary callable serialization and the blocked `apply_model` path.

Reclaim first releases free-local, invalid, then cache-only buffers. If required CPU
snapshots still need to be freed, it can publish missing disk sources before transferring
restore responsibility and releasing RAM. A failed spill preserves required RAM and the
remaining target. Disk preparation itself does not reclaim memory, and reclaim does not
guarantee that every requested byte can be freed. See [Pool and reclaim](../runtime/pinned_cpu_backup_pool.md).

## Runtime implementation and validation

The engine-neutral [BackupRuntime](../../switch_runtime/runtime.py) coordinates
[host pooling](../../switch_runtime/pool.py), [disk bundles](../../switch_runtime/disk.py),
[metadata coordination](../../switch_runtime/coordinator.py), and
[diagnostics](../../switch_runtime/diagnostics.py). Only the
[vLLM adapter](../../switch_runtime/adapters/vllm.py) translates engine-specific objects
and lifecycle calls into those contracts.

The engine adds `tests/v1/executor/test_sleep_cpu_backup.py`,
`tests/v1/worker/test_weight_backup_lifecycle.py`, and
`tests/basic_correctness/test_exact_disk_backup_gpu.py`. They cover RPC ordering,
guard/identity behavior, and real disk restore with CUDA graph address reuse.
The existing `tests/basic_correctness/test_cumem.py` covers native allocator behavior.
See [Testing](testing.md) for commands and the additional GPU/inference checks.
