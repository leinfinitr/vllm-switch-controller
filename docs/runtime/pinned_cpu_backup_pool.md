# Process-local pinned CPU backup pool

[`switch_runtime/pool.py`](../../switch_runtime/pool.py) owns size-keyed host
buffers. Each engine worker has its own pool; the controller service has no tensor access.
The CUDA memory backend allocates pinned PyTorch storage in the worker's existing environment.

## Ownership and reclaim

Buffers are either copying, required for restoration, restoring, reusable cache-only,
failed-publication storage, or detached free-local storage. Copying, restoring, and required
buffers cannot be reclaimed unless a committed exact disk source can safely take over.
Free-local and failed-publication storage is released before reusable cache-only snapshots.
If the target remains unmet and exact disk is enabled, the runtime can publish missing
disk sources for required CPU snapshots before releasing their RAM. A failed spill keeps
the required CPU source and leaves the unsatisfied byte target pending.

The optional coordinator client exchanges protocol-v1 aggregate byte buckets, process
incarnation identity, capabilities, cumulative release targets, and monotonic release
acknowledgements. Failed requests retain pending usage for retry. Network I/O stays outside
allocator state critical sections during normal transitions and polling.

Dropping storage ownership is logical release. Returning pinned memory to the OS additionally
uses PyTorch's private host-cache flush API. Failure is observable and does not change the
correctness of snapshot transitions. Validate physical reclaim using worker RSS and host
memory observations, separately from released-byte accounting.

## Configuration and shutdown

The external vLLM adapter accepts the existing `VLLM_CPU_BACKUP_COORDINATOR_*` names.
`VLLM_SLEEP_BACKEND=switch` explicitly selects it. A background worker thread polls release
requests; shutdown stops the poller before destroying stream and disk resources.

See [Fork setup](../vllm-switch/integration.md),
[Coordinator protocol](../cpu_backup_coordinator.md), and
[Runtime contracts](../runtime.md) for wire fields, defaults, and memory contracts.
