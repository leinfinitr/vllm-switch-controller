# CPU weight backup architecture

The backup implementation lives in this repository's `switch_runtime` package, imported
into each vLLM GPU worker through the companion fork's provider API.

```text
executor collective readiness / prepare / abort
  -> worker sleep provider boundary
       -> process-local BackupRuntime
            -> CPU pool / exact disk / CUDA memory backend
            -> optional aggregate HTTP coordinator
```

The controller service receives usage and acknowledgements and sends byte targets. It
never owns tensors, device pointers, or snapshot payloads. Each worker owns its runtime
and uses its existing CUDA context and allocator mappings.

## Contracts

`MemoryRegion` describes an allocation incarnation, size, device, tag, and opaque native
handle. Its save policy is immutable reuse, snapshot on every sleep, or discard/rebuild.
`MemoryBackend` supplies allocation, copying, mapping, synchronization, and cache release.
`BackupRuntime` owns publication, leases, transactions, reclaim, and recovery states.

The vLLM adapter classifies parameter allocations without overlapping model buffers as
immutable. Mixed or unclassified weight storage is snapshotted each sleep. KV allocations
are discarded. No weight-content versions or mutation-detection callbacks remain.

## Invariants

1. Snapshots are reusable only after successful publication for the same allocation identity.
2. Every worker prepares before any worker unmaps; prepare failure aborts leases.
3. Required or in-flight storage is protected from reclaim. A committed disk source can
   take responsibility before required RAM is released.
4. Partial mapping/unmapping or reconstruction failures require engine recovery.
5. L2 reconstruction of the original checkpoint finishes before inference and fresh snapshots.

See [Fork setup](../vllm-switch/integration.md), [Fixed-weight lifecycle](eager_cpu_weight_backup.md),
[Pool and coordinator](pinned_cpu_backup_pool.md), and [Exact disk](exact_disk_backup.md).
The shared interfaces are documented in [Runtime contracts](../runtime.md).
For the engine-side calls that reach them, see the
[v0.22.1 delta and call chains](../vllm-switch/delta-v0.22.1.md).
