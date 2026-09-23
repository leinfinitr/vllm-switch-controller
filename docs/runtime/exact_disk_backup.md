# Exact runtime-byte disk backup

[`switch_runtime/disk.py`](../../switch_runtime/disk.py) provides process-local
exact-byte snapshots. The store is engine-independent and uses caller-owned staging buffers.
The library and CUDA restore stream run inside the model worker.

## Publication and identity

Each locked process-incarnation directory contains immutable committed bundles. Publication
writes and fsyncs payload bytes, per-chunk SHA-256 values, a canonical manifest, and its
COMMIT checksum, then atomically renames the directory. Only committed references become
usable restoration sources. Startup removes only unlocked stale process directories.

Manifest schema 2 uses opaque allocation-incarnation region IDs. Weight-content version
tracking has been removed. Bundles belong to the current live process; they are not portable
checkpoints and are not reused across upgrades or restarts.

## Restore

All selected manifests are validated before the first remap. A bounded pipeline uses four
host staging slots, a reader, two checksum workers, and a CUDA restore stream. Completion
fences prevent slot reuse while H2D is in flight. Weight-backed allocations restore before
discarded allocations such as KV storage are remapped.

Corrupt payloads, missing files, read failures, or copy/fencing failures propagate. Failures
after remapping may have started enter sticky `RECOVERY_REQUIRED`. There is no silent
fallback to checkpoint reconstruction.

`O_DIRECT` remains optional and requires aligned pinned storage. The vLLM adapter retains
`VLLM_EXACT_DISK_BACKUP_ENABLED`, `VLLM_EXACT_DISK_BACKUP_DIR`,
`VLLM_EXACT_DISK_BACKUP_CHUNK_BYTES`, and `VLLM_EXACT_DISK_BACKUP_DIRECT_IO` configuration.
Enable the runtime with `VLLM_SLEEP_BACKEND=switch`.

See [Fork setup and configuration](../vllm-switch/integration.md),
[Runtime contracts](../runtime.md), and [Verification](../vllm-switch/testing.md).
