# vLLM Switch Fork

The companion vLLM Switch repository keeps tensor correctness and data movement inside each
vLLM process; this controller remains a metadata-only external control plane.

It adds five mechanisms:

1. reusable pinned CPU clean backups;
2. eager weight prebackup after engine warmup;
3. lifecycle-wide validity and fail-closed mutation fencing;
4. metadata-only HTTP coordination and dynamic host-memory reclaim;
5. the exact disk backup tier.

See [Integration](integration.md) and [Testing](testing.md).

## Responsibility boundary

| vLLM worker | Controller |
|---|---|
| Owns pinned tensors and disk bundles | Never receives backup bytes or tensor IDs |
| Performs D2H/H2D and CUDA VMM changes | Serializes backend lifecycle calls |
| Tracks content and backup versions | Tracks aggregate per-process usage |
| Selects concrete reclaim victims | Sends cumulative byte targets |
| Fences mutation/recovery failures | Applies host-memory pressure policy |

This boundary avoids a second tensor state machine across the network.

## Mechanism summary

### Reusable clean CPU backup

A valid process-local CPU snapshot remains attached after wake. A later level-1 sleep can
reuse it and skip allocation and D2H. Wake still restores bytes to the GPU, so the feature
must not be described as eliminating both directions of copy.

### Eager prebackup

After weight loading, profiling, kernel warmup, and CUDA graph capture, vLLM synchronously
publishes a clean weight snapshot before readiness. The first level-1 sleep can therefore
reuse it. Mutation paths invalidate snapshots through content versions; they do not trust
stale bytes.

### Transactional lifecycle

Level-1 sleep prepares every worker before any worker unmaps. Prepared backups become
restore-required leases. Prepare failure aborts all workers; uncertainty during commit,
unmap, restore, or a partial mutation enters a sticky recovery-required state.

### Metadata-only coordinator

The worker reports aggregate RAM categories and polls cumulative release targets. The
controller does not choose tensors. The worker acknowledges actual allocator storage drops
through `released_bytes_total`; physical reclaim still requires RSS and `MemAvailable`
evidence.

### Exact disk backup

The optional disk tier stores exact allocator-segment bytes in process-incarnation bundles
with a committed manifest and per-chunk SHA-256 checks. Restore is pipelined through bounded
staging buffers. Validation and failure fencing run before or around GPU remapping so a bad
bundle cannot be treated as a valid restore source.

## Limitations

- CPU and disk backups are process-local and cannot be shared between workers.
- Eager prebackup increases startup time and pinned host memory by one local weight shard.
- Exact disk backup is experimental and depends on local filesystem/direct-I/O behavior.
- The host-cache flush uses a private PyTorch API; logical release may not reduce RSS.
- Out-of-tree model mutations must explicitly participate in backup invalidation.
- A recovery-required transition normally requires engine restart or reload.

Low-level allocator details remain authoritative in the fork's own design documents. This
directory explains the public integration contract from the controller perspective.
