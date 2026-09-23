# Process-local inference runtime

This repository ships two Python packages in one wheel:

- `vllm_switch_controller`: the external routing and metadata service.
- `switch_runtime`: the backup library imported by each engine GPU worker.

The runtime never starts a backup subprocess. Host buffers, device pointers, CUDA streams,
and disk staging storage remain in the process that owns the model. Optional HTTP carries
aggregate usage, cumulative release targets, and acknowledgements only. The controller
service does not import PyTorch or require a GPU.

## Design guides

This page defines the shared contracts and adapter responsibilities. Detailed mechanisms
are maintained alongside it:

- [CPU backup architecture](runtime/cpu_weight_backup.md): ownership and invariants.
- [Fixed-weight lifecycle](runtime/eager_cpu_weight_backup.md): readiness, reuse, and L2.
- [Pinned CPU pool](runtime/pinned_cpu_backup_pool.md): storage lifetime and reclaim.
- [Exact disk backup](runtime/exact_disk_backup.md): publication, validation, and restore.

For engine functions and their runtime counterparts, see the
[vLLM v0.22.1 delta](vllm-switch/delta-v0.22.1.md). Environment defaults live in the
[integration guide](vllm-switch/integration.md#canonical-environment-variables), and HTTP
accounting is specified in the [coordinator protocol](cpu_backup_coordinator.md).

## Installation and selection

Install this wheel into the engine's existing Python environment as well as the controller
environment. The engine continues to own its PyTorch/CUDA dependencies:

```bash
uv pip install --python /path/to/engine/.venv/bin/python /path/to/vllm-switch-controller
VLLM_SLEEP_BACKEND=switch /path/to/engine/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model --enable-sleep-mode
```

The vLLM fork needs provider API version 1. The `vllm.general_plugins` entry point is named
`switch_runtime`; include it if `VLLM_PLUGINS` is an explicit allowlist. Registration is
idempotent and allocates no device state. Each worker creates the runtime when it creates
its allocator, after CUDA device initialization. An explicitly selected unavailable or
incompatible provider raises an error. `VLLM_SLEEP_BACKEND=native`, or an unset selector,
uses native vLLM sleep behavior without the external runtime.

The vLLM adapter reads existing `VLLM_CPU_BACKUP_COORDINATOR_*`,
`VLLM_EXACT_DISK_BACKUP_*`, and `VLLM_SLEEP_PROFILE_PATH` variables. They are no longer
registered or interpreted inside vLLM. See [Integration](vllm-switch/integration.md).

## Memory and lifecycle contracts

`switch_runtime/contracts.py` defines interfaces without engine or tensor dependencies:

| Contract | Responsibility |
|---|---|
| `MemoryRegion` | Allocation-incarnation identity, process-local address, size, device, tag, opaque engine handle, save policy |
| `HostBuffer` | Host address, byte size, and byte view with a retained storage owner |
| `MemoryBackend` | Mapping, unmapping, synchronous copying, synchronization, host allocation, and cache release |
| `RestoreStream` | Bounded asynchronous transfer slots, completion waiting, stream fencing, and cleanup |
| `BackupRuntime` | Snapshots, prepare/commit/abort, restore-source leases, CPU/disk reclaim, accounting, and diagnostics |

Each allocation has one save policy:

- `IMMUTABLE`: reuse a successfully published snapshot for the allocation's lifetime.
- `SNAPSHOT`: capture current bytes at each sleep preparation.
- `DISCARD`: remap empty storage; the engine reconstructs its contents.

The vLLM adapter marks parameter allocations without overlapping model buffers immutable.
Weight allocations containing buffers or unclassified storage use `SNAPSHOT`; KV storage
uses `DISCARD`. Mixed parameter/buffer allocations are captured conservatively. Engines
must expose all mutable persistent storage; custom models that mutate undeclared parameter
storage violate the fixed-weight contract and are unsupported.

The vLLM bridge retains its existing native allocator and CUDA VMM extension. It notifies
the external backend on allocation/free, delegates sleep/wake, and calls provider guards
at worker boundaries. No runtime code depends on vLLM outside `adapters/vllm.py`.

## Fixed-weight inference

The optimized path freezes model weights after profiling, warmup, and graph capture.
It does not track weight-content versions or intercept writes. The adapter rejects:

- EPLB and weight-transfer configuration at startup;
- LoRA configuration until its separate storage lifetime is supported;
- online weight updates, dynamic LoRA calls, Elastic EP operations, and arbitrary
  `apply_model` callbacks through guarded worker methods;
- reload requests outside reconstruction of the original L2 checkpoint, or with overrides.

These restrictions apply only to the external provider. Diagnostic RPCs such as
`sleep_extension("stats")` remain available without arbitrary model callbacks.

Runtime allocation identity, publication state, checksums, and recovery protection remain.
`INVALID` now describes failed/incomplete snapshot publication, not detected weight changes.

The engine's `sleep_extension("identity")` string RPC reports the instantiated worker
backend, including native mode. For switch workers, `sleep_extension("stats")` also
includes `runtime_identity`: provider API version, PID, engine/runtime/provider module
paths, package version, and effective disk/coordinator configuration. These read-only
queries do not require callable RPC serialization or `apply_model`. Benchmark clients
resolve the returned source paths and record repository identity on the measurement host.

## Transitions and recovery

Startup prepares CPU snapshots and optional disk bundles after collective warmup. L1 sleep
prepares all workers before any worker unmaps. Snapshot capture and restore-source leasing
are atomic with respect to reclaim. A prepare failure leaves mappings intact and aborts
leases on already-prepared workers. Failures after mapping/unmapping can begin enter sticky
`RECOVERY_REQUIRED`; restart the engine before serving again.

L1 wake restores exact bytes, including staged `weights` then `kv_cache` wake. CUDA virtual
addresses remain unchanged, preserving captured graph addresses.

L2 explicitly drops prior snapshots and retains a checkpoint-reconstruction obligation.
Mapping weights alone does not satisfy it. The control adapter performs:

```text
wake weights -> collective reload_weights -> wake remaining/KV storage -> verify awake
```

The worker rejects inference and snapshot publication until reconstruction succeeds.
After all tags wake, fresh snapshots are prepared for the new storage lifecycle. Reload
failure requires engine recovery. L2 uses the configured original checkpoint; online
replacement with different weights is unsupported.

The runtime releases free-local and failed-publication buffers before reusable cache-only
buffers. Required RAM is released only after responsibility transfers to a committed disk
source. Logical release acknowledgement is distinct from physical host-cache reclamation.

## Exact disk

Schema 2 identifies allocation incarnations by opaque region IDs. There are no weight
content-version counters. Each bundle has an immutable identity, payload chunk checksums,
a manifest checksum, and an atomic committed-directory publication. Bundles belong to one
live process incarnation and are not portable checkpoints or upgrade/restart recovery data.

The retained restore pipeline uses four bounded host staging slots, a reader, two checksum
workers, and a CUDA restore stream. Completion fences protect storage until H2D completes.
Manifest validation precedes remapping; payload failure after remapping poisons residency.
`O_DIRECT` remains optional and requires aligned host storage.

## Adding another engine

Implement `MemoryBackend` using the engine's own allocation/context APIs, notify the runtime
when allocations are created/destroyed, and identify immutable, mutable, and discarded
regions. Add readiness, drained sleep, reconstruction, and shutdown boundaries. Preserve
prepare/commit/abort and error propagation across the engine's existing worker transport.

Separately implement `EngineControlAdapter` in `vllm_switch_controller/backends.py` for
health, sleep, resume, and state probes. `EngineClient` owns deadlines and proxy transport;
the adapter translates engine-specific endpoints and reconstruction ordering. The launcher
and request router share this client. The current wheel supplies vLLM; tests also exercise
an independent fake control adapter and a non-PyTorch memory backend.

## Validation and scope

Run the controller's CPU checks, including runtime tests, without installing vLLM:

```bash
uv sync --frozen --dev
uv run python -m pytest tests -q
uv run ruff check vllm_switch_controller switch_runtime scripts tests
uv run ruff format --check vllm_switch_controller switch_runtime scripts tests
uv run mypy --ignore-missing-imports vllm_switch_controller switch_runtime
uv build
```

The engine checkout retains worker/executor bridge tests and a GPU exact-disk/CUDA-graph
test. GPU validation must additionally check inference equality, repeated reuse, staged
wake, L2 reconstruction, and physical reclaim. Benchmark runners and retained evidence
belong in `vllm-switch-bench`.

The current validated integration scope is Linux, CUDA, and single-GPU inference. Shared
interfaces and collective preparation preserve a path to other engines and multiple ranks;
they do not establish validation of those configurations. The runtime uses PyTorch's
private host-cache flush API, so physical reclamation remains version-sensitive and is
reported separately from logical byte accounting.
