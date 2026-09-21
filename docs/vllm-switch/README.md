# vllm-switch integration

The companion vLLM fork exposes provider API version 1. The `switch_runtime` package in
this repository implements backups inside each GPU worker. Its CPU pools, CUDA copies,
exact disk snapshots, and recovery state do not live in the controller service.

| Engine worker runtime | Controller service |
|---|---|
| Owns pinned buffers and exact snapshot payloads | Owns routing, request drain, and switching policy |
| Tracks publication and allocation identity | Tracks aggregate per-process usage |
| Prepares, restores, and reclaims concrete storage | Sends cumulative byte targets |
| Enforces fixed-weight inference and recovery | Verifies lifecycle post-conditions |

Select the library with `VLLM_SLEEP_BACKEND=switch` after installing this wheel in the
engine environment. The native engine path remains available without that selector.

The runtime supports eager CPU snapshots, immutable snapshot reuse, per-sleep snapshots
for mutable model buffers, transactional L1 sleep, staged wake, L2 reconstruction of the
original checkpoint, metadata-only coordination, and optional exact disk restore.
It rejects weight-transfer, EPLB, LoRA, and online mutation operations in this mode.

- [Process-local runtime](../runtime.md): contracts, implementation, fixed-weight restrictions.
- [Integration](integration.md): installation and environment configuration.
- [Testing](testing.md): CPU and GPU verification boundaries.
- [Compatibility](../compatibility.md): provider, wire, and disk format versions.

The current tested integration scope is single-node, single-GPU CUDA inference. Interface
support does not establish validation for additional engines, arbitrary custom mutable
models, or multi-rank deployments. Benchmark data belongs in `vllm-switch-bench`.
