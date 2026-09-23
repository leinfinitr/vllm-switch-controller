# vllm-switch fork and integration

[vllm-switch](https://github.com/leinfinitr/vllm-switch) is a backup-focused research fork
of upstream vLLM `v0.22.1`. It retains the `vllm` Python package and Apache-2.0 code.
This controller repository maintains its fork documentation alongside the implementation
of `switch_runtime`, the optional backup library loaded inside each GPU worker.

Native vLLM sleep is the default. Installing this package in the engine environment and
selecting `VLLM_SLEEP_BACKEND=switch` enables provider API version 1. The library owns
CPU pools, CUDA copies, exact disk snapshots, and recovery state in the model process.

| Engine worker runtime | Controller service |
| --- | --- |
| Owns pinned buffers and exact snapshot payloads | Owns routing, request drain, and switching policy |
| Tracks publication and allocation identity | Tracks aggregate per-process usage |
| Prepares, restores, and reclaims concrete storage | Sends cumulative byte targets |
| Enforces fixed-weight inference and recovery | Verifies lifecycle post-conditions |

## Capabilities

- Eager CPU snapshots after profiling, warmup, and CUDA graph capture.
- Reusable pinned storage and immutable snapshots across repeated L1 sleep/wake.
- Fresh snapshots for allocations containing mutable model buffers.
- Collective L1 preparation, staged wake, and guarded L2 checkpoint reconstruction.
- Optional metadata-only HTTP coordination and cooperative host-memory reclaim.
- Best-effort PyTorch pinned-cache flushing, measured separately from logical release.
- Optional exact runtime-byte disk bundles with transactional publication, SHA-256
  verification, optional `O_DIRECT`, and bounded pipelined restore.
- Read-only worker identity/stats RPCs and opt-in JSONL backup diagnostics.

The controller service never receives tensors, CUDA pointers, or backup bytes. The runtime
supports fixed weights and rejects EPLB, weight transfer, LoRA, Elastic EP, and online
mutation calls. L2 can reconstruct the originally configured checkpoint.

## Reading guide

- [Integration](integration.md): installation, selectors, environment variables, and endpoints.
- [Delta from v0.22.1](delta-v0.22.1.md): changed functions, runtime counterparts, and call chains.
- [Runtime contracts](../runtime.md): shared interfaces, adapters, and supported scope.
- [CPU backup architecture](../runtime/cpu_weight_backup.md): ownership and invariants.
- [Fixed-weight lifecycle](../runtime/eager_cpu_weight_backup.md): readiness, reuse, and recovery.
- [Pinned pool](../runtime/pinned_cpu_backup_pool.md): local storage and reclaim.
- [Exact disk](../runtime/exact_disk_backup.md): publication, integrity, and restore.
- [Coordinator protocol](../cpu_backup_coordinator.md): aggregate accounting and release targets.
- [Testing](testing.md): engine environment, CPU checks, and GPU verification.
- [Compatibility](../compatibility.md): provider, wire, and disk format versions.

## Scope and support

The validated integration scope is Linux, single-node, single-GPU CUDA inference.
Multi-rank operation, broad model coverage, and production deployment need separate
validation. Disk bundles belong to a live process and are not restartable checkpoints.
A coordinator outage does not weaken local restore protection; a missing or corrupt
required restore source fails closed. Benchmark evidence belongs in `vllm-switch-bench`.

Report engine-bridge issues to the [fork tracker](https://github.com/leinfinitr/vllm-switch/issues)
and runtime/controller issues to the
[controller tracker](https://github.com/leinfinitr/vllm-switch-controller/issues).
If a reproducer also fails with upstream vLLM, report it upstream and link the reports.
Disclose vulnerabilities privately under the [security policy](../../SECURITY.md).
