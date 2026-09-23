# CPU Backup Coordinator

The controller coordinates aggregate usage of process-local pinned CPU backups and
applies a host-memory pressure policy. `switch_runtime` inside each engine worker owns the data plane and correctness
state; the controller issues byte targets only.
See the [worker pool design](runtime/pinned_cpu_backup_pool.md) for concrete storage
ownership and [engine integration](vllm-switch/integration.md) for runtime configuration.

## Responsibility Boundary

| In-process runtime | Controller service |
|---|---|
| Owns pinned tensors | Never receives tensors or backup IDs |
| Performs D2H and H2D copies | Aggregates per-process and per-model bytes |
| Maintains tensor state and validity | Reads host `MemAvailable` |
| Protects restore-required and in-flight storage | Calculates release byte budgets |
| Selects concrete objects to release | Orders clients by policy |

CPU backup is an opportunistic, application-reclaimable cache. Keeping a clean backup
can avoid allocation and D2H on a later sleep. Releasing it under pressure is safe; a
later sleep recreates it. The controller must never require release of content needed to
restore the current GPU mapping.

## Aggregate Accounting Protocol

The coordinator uses protocol version `1`. Registration and usage requests include
`protocol_version` plus an explicit `capabilities` list. The controller rejects unknown
versions/capabilities and extra fields with `422`; identity or monotonic-state conflicts
return `409`.

Defined capabilities are `cumulative-release-v1`, `released-bytes-total-v1`,
`process-incarnation-v1`, and `exact-disk-accounting-v1`.

Workers use these endpoints:

- `POST /admin/cpu-backup/register`
- `POST /admin/cpu-backup/usage`
- `GET /admin/cpu-backup/release-requests/{client_id}`
- `GET /admin/cpu-backup/stats`

Operators can request bytes from one client with `POST /admin/cpu-backup/release`. See
the [API reference](api.md) for request examples.

A usage report has this shape:

```json
{
  "protocol_version": 1,
  "capabilities": [
    "cumulative-release-v1",
    "exact-disk-accounting-v1",
    "process-incarnation-v1",
    "released-bytes-total-v1"
  ],
  "client_id": "run:model-a:12345:incarnation",
  "pid": 12345,
  "engine": "vllm",
  "model_id": "model-a",
  "total_bytes": 3250585600,
  "released_bytes_total": 0,
  "required_for_restore_bytes": 0,
  "cache_only_bytes": 3250585600,
  "invalid_bytes": 0,
  "free_local_bytes": 0,
  "disk_backup_current_bytes": 0,
  "disk_backup_reserved_bytes": 3250585600,
  "ram_reclaimable_with_disk_bytes": 0
}
```

Every report must satisfy:

```text
total_bytes == required_for_restore_bytes
             + cache_only_bytes
             + invalid_bytes
             + free_local_bytes
```

The latter three categories are evictable. Required RAM can additionally be declared
reclaimable only when the exact-disk capability and a usable disk source are reported.
`vllm-switch`'s local release order is:

```text
FREE_LOCAL -> INVALID -> CACHE_ONLY
```

`REQUIRED_FOR_RESTORE`, `COPYING_D2H`, and `RESTORING_H2D` are all represented as
non-evictable bytes in the aggregate protocol.

### Process-Incarnation Identity

The configured client ID is only a logical prefix. Each worker appends its PID and a
high-resolution timestamp, producing a non-reusable process-incarnation ID. The
controller keys records by this complete ID.

A restarted worker must not inherit cumulative commands or pending state from its
predecessor. If old and new workers overlap, both records remain. Replacing a record by
logical prefix would undercount pinned memory still owned by the old process.

### Release Commands and Idempotency

`GET /admin/cpu-backup/release-requests/{client_id}` returns:

- `request_epoch`, regenerated when the controller restarts;
- monotonic `requested_release_bytes_total` within that epoch;
- current `pending_release_bytes`.

A worker executes only the cumulative delta it has not observed in the current epoch.
The GET is therefore idempotent: a lost response can be retried, and a duplicate response
does not release the same bytes twice.

A command is an obligation, not an acknowledgement of completion. `vllm-switch` increments
`released_bytes_total` only when process-local pool `reserved_bytes` actually falls. The
controller acknowledges progress from its monotonic delta:

```text
released = new_released_total - old_released_total
pending  = max(pending - released, 0)
```

This cumulative acknowledgement survives latest-wins usage coalescing. Even if the
worker immediately allocates new backup and returns to its previous footprint, the real
release is not lost. Protocol v1 requires this counter on the wire; the core state object
keeps only a narrow compatibility path for direct in-process tests and old unpublished
callers.

The counter proves an allocator storage decrease, not return of pages to the operating
system. Physical reclaim requires the evidence described below.

A state transition such as `cache_only -> required_for_restore` does not cancel pending
bytes. Otherwise the same storage could become cache-only again and receive a duplicate
release budget. Temporarily unsatisfied bytes remain an outstanding obligation.

## Host-Memory Pressure Policy

The current signal is read on the controller's Linux host:

```text
/proc/meminfo: MemTotal, MemAvailable
```

`MemAvailable` is preferable to `MemFree` because it includes page cache the kernel
expects to reclaim without swapping. Ratio and absolute thresholds combine as:

```text
low  = max(MemTotal * reclaim_ratio, reclaim_bytes)
high = max(MemTotal * recovery_ratio, recovery_bytes)
```

The state machine is:

```text
NORMAL
  N consecutive samples with MemAvailable < low
  -> RECLAIMING

RECLAIMING
  MemAvailable >= high
  -> NORMAL
```

While reclaiming:

```text
target_release   = max(high - MemAvailable, 0)
additional_bytes = max(target_release - pending_release_bytes, 0)
```

Consecutive samples filter short-lived noise, separate low/high watermarks provide
hysteresis, and the cooldown limits command frequency. Existing pending bytes are
subtracted from the target to avoid over-reclamation.

### Victim Order

Clients receive byte budgets in this order:

1. lower model priority;
2. older usage `updated_at` timestamp;
3. larger remaining evictable footprint.

This order chooses only a client and byte budget. `vllm-switch` still selects concrete storage.

### Optional Hard Cap

`cpu_backup_global_cap_bytes` is a safety and debugging guard, not the primary policy:

```text
over_cap = max(total_bytes - cap - pending_release_bytes, 0)
```

The request is bounded by evictable bytes. If required bytes alone exceed the cap, the
controller requests every evictable byte but never violates restore requirements. The
remaining gap stays visible as `over_cap_bytes`.

## Interaction With Request Switching

Lifecycle and request-reservation safety are independent of the aggregate coordinator.
The switch lock serializes model transitions, and a policy that sleeps the old model must
wait for its in-flight requests. The proxy reserves the target while holding that lock.

Coordinator failure cannot change tensor validity or permit unsafe release because `vllm-switch`
enforces all local state checks. Conversely, successful aggregate accounting does not
prove that request switching or physical reclamation succeeded. Each layer needs its own
evidence. See [Architecture](architecture.md) for request cancellation and fail-closed
lifecycle semantics.

## Configuration

```yaml
controller:
  cpu_memory_reclaim_available_ratio: 0.15
  cpu_memory_recovery_available_ratio: 0.20
  cpu_memory_reclaim_available_bytes: 8589934592
  cpu_memory_recovery_available_bytes: 12884901888
  cpu_memory_poll_interval_s: 0.5
  cpu_memory_pressure_consecutive_samples: 3
  cpu_memory_reclaim_cooldown_s: 2.0

  cpu_backup_global_cap_bytes: null
  cpu_backup_default_model_priority: 0
  cpu_backup_model_priorities:
    cold-model: 0
    hot-model: 10
```

Ratio thresholds must appear as a pair, and recovery must not be below reclaim. Byte
thresholds follow the same ordering. See [Configuration Reference](configuration.md) for
defaults and enablement rules.

## Observability and Physical Reclaim

`GET /admin/cpu-backup/stats` returns:

- global `total`, `required`, `cache_only`, `invalid`, `free_local`, `evictable`, and
  `pending` bytes;
- the same accounting per client;
- release commands not yet consumed by clients;
- pressure state, watermarks, consecutive samples, target and unresolved bytes, and
  probe errors.

Important cumulative fields in `vllm-switch` sleep profiles include:

```text
cpu_backup_release_bytes
cpu_backup_host_cache_flush_count
cpu_backup_host_cache_flush_errors
cpu_backup_coordinator_request_errors
```

Benchmarks must calculate step deltas from cumulative counters. A decrease in
`total_bytes` proves application-level logical release only. Physical reclaim requires
correlated worker process-tree RSS and host `MemAvailable` changes.

PyTorch may keep deleted pinned tensors in its host caching allocator. `vllm-switch`
release path makes a best-effort process-wide call to the private
`torch._C._host_emptyCache()` API. Failure does not compromise sleep/wake correctness,
but RSS may remain high. Report flush errors and OS metrics instead of claiming physical
reclaim in that case.

## Validation Standard

Research-quality evidence includes at least two paired conditions:

1. **Pressure:** trigger release; require a positive release delta, no flush errors, a
   meaningful worker RSS decrease, and recovery in `MemAvailable`.
2. **No pressure:** remain in `NORMAL`; require zero release delta, positive backup reuse
   on a later sleep, and zero or timer-resolution D2H time.

On a shared host, raise the configured thresholds above current `MemAvailable` to trigger
the policy safely. Do not validate by consuming most system RAM. Record parameters,
repository commits, model revisions, host/GPU information, raw failures, and automated
assertion results.

## Known Limitations

- `/proc/meminfo` is host-global. Multi-host deployments need one controller or agent per
  host.
- There is no worker lease or heartbeat. Records from abnormally exited workers remain
  conservatively accounted, which can over-request release but does not undercount known
  pinned usage.
- The monitor does not read cgroup v2 `memory.current`, `memory.max`, `memory.events`, or
  memory pressure events.
- PSI and NUMA locality are not used.
- Required or copy-in-flight backup cannot be released; the shortfall remains visible as
  `unresolved_pressure_bytes`.
- The controller cannot independently verify whether host-cache flushing returned pages
  to the OS; worker and host telemetry are required.
