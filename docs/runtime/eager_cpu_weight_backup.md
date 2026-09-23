# Eager fixed-weight CPU backup

The external vLLM provider prepares snapshots after model profiling, kernel warmup, and
CUDA graph capture. The library runs in the GPU worker process and keeps host buffers
available for reuse across repeated L1 sleep/wake cycles.

```text
load -> profile / KV initialization -> warmup / graph capture
     -> collective ready event -> external CPU/disk preparation -> engine ready
```

At L1 sleep, snapshot and restore-source leasing are atomic with respect to local reclaim.
All workers prepare before collective sleep starts. A failed prepare aborts leases while
mappings remain intact. Uncertain unmap/remap failures enter `RECOVERY_REQUIRED`.

## Inference contract

The optimized provider supports fixed weights after warmup. It rejects EPLB, online weight
transfer, LoRA configuration, dynamic mutation calls, Elastic EP, and arbitrary `apply_model`
callbacks. These restrictions apply only when `VLLM_SLEEP_BACKEND=switch` is selected.
Known mutable model buffers use a fresh snapshot per sleep; mixed allocations also take
this conservative path. Unsupported custom writes into frozen parameter storage are not
tracked or detected.

L2 explicitly resets snapshots. The worker permits reloading the original checkpoint only
when L2 reconstruction is pending. Inference and snapshot publication remain blocked until
reload succeeds. All tags must be awake before the next readiness snapshot. A failed reload
requires engine recovery.

Snapshot publication state and allocation identity remain necessary even without content
version counters. Disk checksums validate stored bytes, independently of model mutation.

See [Architecture](cpu_weight_backup.md), [Runtime contracts](../runtime.md), and the
[worker guard map](../vllm-switch/delta-v0.22.1.md#worker-guards).
