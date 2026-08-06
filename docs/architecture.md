# Architecture

## Scope

The controller sits in front of multiple long-lived, single-model vLLM processes. It
owns request-driven model selection, lifecycle serialization, OpenAI-compatible proxying,
aggregate CPU backup accounting, and host-memory pressure policy.

It does not execute models or own CPU/GPU backup contents. Those responsibilities stay
inside vLLM so that tensor validity and copy synchronization cannot diverge across a
network boundary.

```text
client
  -> model-switch controller (:9000)
       -> vLLM backend A (:8101)
       -> vLLM backend B (:8102)

vLLM worker -- aggregate usage and acknowledgement --> controller
vLLM worker <-- cumulative target_free_bytes -------- controller
```

## Runtime Components

- `controller/router.py` implements the OpenAI-compatible data plane and `/admin/*`
  management plane.
- `controller/state.py` owns lifecycle state, active-request reservations, draining,
  and the global switch lock.
- `controller/vllm_client.py` calls backend health and lifecycle endpoints and proxies
  inference traffic. Explicit backend traffic does not inherit environment proxies.
- `controller/policies.py` decides which model to sleep or wake for a target alias.
- `controller/metrics.py` records per-request queue, switch, transport first-byte, and
  completion data.
- `controller/processes.py` reads Linux PID/PGID/start-time identity for safe pool cleanup.
- `controller/backup_pool.py` stores only per-process aggregate byte categories,
  priorities, cumulative commands, and release obligations.
- `controller/memory_pressure.py` reads host `MemAvailable` and applies debounce,
  low/high watermarks, and a reclaim cooldown.

## Request Switching

The data path for a configured alias is:

```text
parse request and validate alias
  -> acquire switch_lock
  -> reconcile unsafe lifecycle states with /is_sleeping
  -> wait for requests on the old model to drain, when required
  -> sleep old model and verify /is_sleeping == true
  -> wake target model and verify /is_sleeping == false
  -> reserve target request while still holding switch_lock
  -> release switch_lock
  -> rewrite model to served_model_name
  -> proxy JSON or streaming response
  -> release reservation exactly once
```

The reservation is created before releasing `switch_lock`. Without that ordering, a
second request could begin sleeping the newly ready backend in the gap between readiness
and request accounting.

The default `always_sleep_previous` policy waits for all in-flight requests on the old
model before sleeping it. This is a global first-come transition policy: requests are not
preempted and no speculative wake or request reordering is performed.

### Streaming Ownership

A streaming request holds its reservation until the upstream body completes or the
downstream disconnects. Reservation enter and exit run in cancellation-resistant tasks.
All JSON and streaming cleanup paths wait on one cached teardown operation, which makes
exit exactly once even when cancellation is repeated.

Streaming ownership transfers before upstream setup is awaited. The context factory,
async enter, response construction, body iteration, and downstream-disconnect path share
one cleanup boundary. A failure before iteration starts therefore cannot leak or
double-release a reservation.

Metrics writes are best-effort observability. A local JSONL write failure is logged but
does not replace the primary backend response, stream result, or cancellation outcome.

## Lifecycle State and Failure Behavior

Each alias has one of these controller states:

```text
unknown
awake
sleeping
waking
sleeping_in_progress
error
```

The controller is fail-closed around uncertain transitions:

- A sleep or wake is committed only after a successful management response and matching
  `/is_sleeping` post-condition.
- A failed, timed-out, or cancelled transition marks the affected alias `error`.
- If the failed alias was active, `active_model` is cleared.
- Before a later transition, `unknown`, `error`, or in-progress aliases are reconciled
  against their real `/is_sleeping` state.
- If reconciliation finds multiple awake backends where the policy expects one active
  model, all observed awake aliases are marked `error` and the request fails.

The controller does not implement distributed transactions or automatic rollback across
backend processes. Operators must diagnose backend health when reconciliation cannot
establish a safe state.

## Proxy Contract

The client-facing alias can differ from the backend's `served_model_name`; the controller
rewrites the body before forwarding. Existing `x-request-id` headers are retained, and a
UUID is generated when the caller does not supply one.

The proxy preserves backend HTTP status and end-to-end headers while filtering RFC
hop-by-hop headers, `Host`, and representation metadata invalidated by body rebuilding.
JSON responses are buffered. Streaming responses are forwarded incrementally and keep
their upstream context open for the full downstream lifetime.

## CPU Backup Boundary

The vLLM allocator owns pinned tensors, validity, D2H/H2D, in-flight copy protection, and
the concrete release order. The controller receives aggregate usage and issues
cumulative byte targets only.

Consequently, a controller failure cannot make invalid backup valid and cannot force
release of `REQUIRED_FOR_RESTORE`, `COPYING_D2H`, or `RESTORING_H2D` storage. See
[CPU Backup Coordinator](cpu_backup_coordinator.md) for the wire contract and evidence
required to claim physical host-memory reclamation.

The v0.1 wire contract is explicitly versioned. Workers send `protocol_version: 1` and a
capability set with registration and usage. The controller rejects unknown capabilities,
process-incarnation identity changes, and capability drift instead of silently accepting
a schema it may misinterpret.

## Startup Model

Configuration initializes the controller's expected state, but does not start or inspect
backends. `scripts.launch_vllm_pool` establishes the real initial state by handling each
backend sequentially:

```text
launch or locate backend
  -> wait for /health
  -> sleep and verify
  -> repeat for every backend
  -> wake and verify startup_awake_model
```

Inference must not begin until this preparation completes. The sequential sequence lets
models initialize even when their awake footprints cannot coexist.

`wake_tags: null` emits no query parameter and uses the vLLM default. A configured
list is forwarded as repeated `tags` parameters by both the normal request path and the
startup launcher. This keeps startup and request-driven semantics identical.

Launcher-managed processes run in independent sessions. The ownership file records PID,
PGID, and Linux process start time. Shutdown signals a complete verified process group and
fails closed if a numeric PID/PGID was reused.

## Repository Boundaries

```text
vllm/
  allocator-local backup state, eager snapshots, version invalidation,
  sleep transactions, physical reclamation, coordinator client

vllm-model-switch-controller/
  multi-backend lifecycle, request drain, OpenAI proxy,
  aggregate accounting, host-pressure policy

llm-switch-bench/
  benchmark adapters, raw and curated evidence, plots, reports
```

This implementation repository does not store paper-level aggregate plots. The benchmark
repository does not duplicate allocator correctness logic.

## Current Limitations

- The management API has no authentication, authorization, or TLS.
- State is in memory and there is no worker lease or restart reconciliation loop.
- The switching lock is process-local; running multiple controller replicas is unsafe.
- There is no replica selection, multi-GPU placement, admission control, or preemption.
- OpenAI Responses, Assistants, Batch, and other stateful APIs are not proxied.
- Host pressure is read from host-global `/proc/meminfo`, not cgroup or NUMA signals.
