# API Reference

The controller exposes a small OpenAI-compatible data plane and an unauthenticated
administrative API. Keep the listener on a trusted interface.

## Data plane

### `GET /v1/models`

Lists configured aliases. It does not probe backend availability.

### `POST /v1/chat/completions`

### `POST /v1/completions`

Both endpoints require a JSON object containing a string `model`. The controller:

1. validates the alias;
2. serializes and verifies any drain/sleep/wake transition;
3. reserves the target before releasing the switch lock;
4. rewrites `model` to `served_model_name`;
5. proxies the request and response;
6. releases the reservation exactly once.

Malformed JSON and non-object JSON return `400`. A missing or non-string model returns
`400`; an unknown alias returns `404`. Lifecycle or backend transport failures return
`502`. Backend inference status codes are proxied unchanged.

JSON and server-sent events are supported. Hop-by-hop headers and representation metadata
invalidated by body rebuilding are removed. Incoming `x-request-id` is forwarded, or a UUID
is generated.

## Health and lifecycle

### `GET /health`

Returns controller process health, active alias, lifecycle states, and active request
counts. It does not probe backends.

### `GET /admin/state`

Returns the same state snapshot without `ok`.

### `POST /admin/switch/{model}`

Ensures an alias is ready under the configured policy. Inference clients should select a
model in the OpenAI request rather than call this endpoint.

## CPU backup coordinator protocol v1

Worker requests use:

```json
{
  "protocol_version": 1,
  "capabilities": [
    "cumulative-release-v1",
    "exact-disk-accounting-v1",
    "process-incarnation-v1",
    "released-bytes-total-v1"
  ]
}
```

Unsupported versions, unknown capabilities, omitted required protocol metadata, or extra
fields return `422`.

### `POST /admin/cpu-backup/register`

Registers one complete process-incarnation client ID:

```json
{
  "protocol_version": 1,
  "capabilities": [
    "cumulative-release-v1",
    "process-incarnation-v1",
    "released-bytes-total-v1"
  ],
  "client_id": "model-a-host-12345-1720000000000000000",
  "pid": 12345,
  "engine": "vllm",
  "model_id": "model-a",
  "gpu_uuid": null,
  "metadata": {"hostname": "worker-host"}
}
```

The response reports controller protocol metadata, `request_epoch`, and the stored client
snapshot. Reusing the same client ID with another PID, version, or capability set returns
`409`.

### `POST /admin/cpu-backup/usage`

Reports aggregate process-local usage:

```json
{
  "protocol_version": 1,
  "capabilities": [
    "cumulative-release-v1",
    "exact-disk-accounting-v1",
    "process-incarnation-v1",
    "released-bytes-total-v1"
  ],
  "client_id": "model-a-host-12345-1720000000000000000",
  "pid": 12345,
  "engine": "vllm",
  "model_id": "model-a",
  "total_bytes": 3250585600,
  "released_bytes_total": 0,
  "required_for_restore_bytes": 3250585600,
  "cache_only_bytes": 0,
  "invalid_bytes": 0,
  "free_local_bytes": 0,
  "disk_backup_current_bytes": 3250585600,
  "disk_backup_reserved_bytes": 3250585600,
  "ram_reclaimable_with_disk_bytes": 3250585600,
  "metadata": {}
}
```

The four RAM categories must sum exactly to `total_bytes`. Disk bytes are separate
telemetry. Exact-disk fields require `exact-disk-accounting-v1`, and
`released_bytes_total` requires `released-bytes-total-v1`.

### `GET /admin/cpu-backup/release-requests/{client_id}`

Returns:

- `protocol_version` and controller capabilities;
- `request_epoch`;
- monotonic `requested_release_bytes_total`;
- current `pending_release_bytes`;
- release requests enqueued during the poll.

Workers execute only the unseen cumulative delta within one epoch.

### `POST /admin/cpu-backup/release`

Queues a release target for a known client:

```json
{
  "client_id": "model-a-host-12345-1720000000000000000",
  "target_free_bytes": 1073741824
}
```

The response reports bytes that could be queued under current evictable accounting.

### `GET /admin/cpu-backup/stats`

Returns the controller epoch, global and per-client accounting, pending obligations, and
host-memory pressure state.

See [CPU Backup Coordinator](cpu_backup_coordinator.md) for idempotency, incarnation, and
physical-reclaim semantics.

## Advance model hints

`POST /admin/prewarm` accepts `model`, `task_id`, `stage_id`, `hint_id`, `source`
(nonempty strings), and `ttl_ms` (integer, 0–60000). It returns `status` equal to
`ready`, `ignored` (with `reason`), or `failed`. Unknown models return HTTP 404;
reusing an unexpired hint ID with different contents returns HTTP 409. A lifecycle
failure returns `failed` with HTTP 200: clients must inspect `status`.

Submit the hint concurrently with a CPU tool, after the preceding model response
has completed. The endpoint waits for readiness; it does not generate tokens or
reserve a future request. The next inference still goes through the OpenAI proxy.
The application executes its own CPU work, outside the controller process.

Hints are advisory. Active reservations, pending demand, a held lifecycle lock,
observed host pressure, an expired hint, or a full 1024-entry hint cache cause an
immediate skip. TTL starts on receipt using the controller's monotonic clock and
bounds admission, not the duration of a started switch. The configured switch
timeout bounds that transaction. A caller disconnect does not abort a started
sleep/wake operation or release its lock early. Real demand takes priority before
advisory admission; it waits behind a transaction that has already begun.

Concurrent identical hint IDs join one transaction. IDs are retained until their
original TTL expires and any transaction completes. Use a fresh ID for a new stage;
a retry returns the original result, not a new readiness observation or a residency
lease. A different hint for a model already ready performs no additional switch.
The controller does not predict targets, schedule arbitrary DAGs, or expose a
background-job API. On a conditional tool branch, the application may abstain.

Controller JSONL metrics include task/stage/hint correlation, source, decision,
switch ID, and existing sleep/wake/combined timing fields. Failed hints keep the
existing lifecycle reconciliation and fail-closed behavior.
