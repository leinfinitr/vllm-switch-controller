# Documentation

## Start here

- [Getting started](getting-started.md): install, configure, launch, request, and stop.
- [Compatibility](compatibility.md): protocol and environment contract.
- [Operations and troubleshooting](operations.md): validation, safe cleanup, and failures.

## Reference

- [Configuration](configuration.md): strict YAML fields and defaults.
- [API](api.md): OpenAI-compatible and administrative endpoints.
- [Process-local runtime](runtime.md): memory contracts, engine adapters, and fixed weights.
- [Architecture](architecture.md): request ownership and failure invariants.
- [CPU backup coordinator](cpu_backup_coordinator.md): accounting and reclaim semantics.

## Runtime design

- [CPU backup architecture](runtime/cpu_weight_backup.md): ownership and invariants.
- [Fixed-weight lifecycle](runtime/eager_cpu_weight_backup.md): readiness, reuse, and L2.
- [Pinned CPU pool](runtime/pinned_cpu_backup_pool.md): storage ownership and reclaim.
- [Exact disk backup](runtime/exact_disk_backup.md): publication, integrity, and restore.

## Companion vllm-switch

- [Fork overview](vllm-switch/README.md)
- [Delta from v0.22.1 and call chains](vllm-switch/delta-v0.22.1.md)
- [Controller–vllm-switch integration](vllm-switch/integration.md)
- [Fork testing](vllm-switch/testing.md)
