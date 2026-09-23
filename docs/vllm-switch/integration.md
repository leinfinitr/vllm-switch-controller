# Controller–vllm-switch Integration

Install this controller wheel in both the controller and engine Python environments.
The engine loads `switch_runtime` as a general plugin. Its backup buffers stay in the GPU
worker process. See [Runtime](../runtime.md).

## Runtime installation and selection

From the engine checkout, install the companion package and select its provider:

```bash
uv pip install --python .venv/bin/python /path/to/vllm-switch-controller
export VLLM_SLEEP_BACKEND=switch
```

Start the engine with `--enable-sleep-mode`. If `VLLM_PLUGINS` is an explicit allowlist,
include the `switch_runtime` entry point. An explicitly selected missing or incompatible
provider fails startup. With the selector unset or `native`, upstream sleep remains the
default. Set the selector before starting the engine and keep it fixed for that process.

The switch adapter requires CUDA and fixed weights. EPLB, weight transfer, LoRA, Elastic
EP, and arbitrary model mutations are unsupported. L2 reconstruction of the original
checkpoint is supported. The library can run without the controller service when HTTP
coordination is disabled; public serving through the controller additionally requires
the development endpoints below.

## Backend lifecycle endpoints

The controller uses vLLM development endpoints enabled by:

```text
VLLM_SERVER_DEV_MODE=1
VLLM_SLEEP_BACKEND=switch
--enable-sleep-mode
```

Required endpoints and their contract are listed in
[Compatibility](../compatibility.md). The controller treats a lifecycle POST and its
`/is_sleeping` post-condition as one transition. A timeout or ambiguous outcome fails
closed.

## Wake tags

The `vllm-switch` allocator assigns at least the `weights` and `kv_cache` tags in the supported
worker path. The API accepts repeated query parameters:

```text
POST /wake_up?tags=weights&tags=kv_cache
```

No `tags` parameter means all currently sleeping tags. In controller YAML:

```yaml
wake_tags: null                  # wake all
# wake_tags: [weights, kv_cache] # advanced partial/staged operation
```

The launcher and request-driven path use the same configured value. An empty list,
duplicate tags, and empty tag strings are rejected. A syntactically valid subset can still
be operationally incomplete; inference after a weights-only wake may require a later
KV/scheduling wake.

For L2, `VllmControlAdapter.resume` first wakes weights, invokes the existing collective
RPC endpoint with `reload_weights`, then wakes the remaining tags and verifies readiness.
Explicit L2 wake tags must include both `weights` and `kv_cache`. The worker blocks
inference until reconstruction completes. See the [call chains](delta-v0.22.1.md).

## CPU backup protocol

The controller requires protocol version `1` on registration and usage. Workers must
send a stable capability set and a complete non-reusable process-incarnation `client_id`.
Exact-disk aggregate fields require `exact-disk-accounting-v1`.

Workers without explicit protocol/capability fields are incompatible. Pair this controller
only with an engine checkout that implements the same contract. A
`422` registration/usage response is an integration mismatch, not a reason to weaken
validation.

## Canonical environment variables

The vLLM adapter reads these names into `RuntimeConfig`; they are runtime settings rather
than vLLM compilation configuration. Controller YAML fields are documented separately in
[Configuration](../configuration.md).

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_SLEEP_BACKEND` | `native` | Select `switch` to enable the external worker runtime. |
| `VLLM_EXACT_DISK_BACKUP_ENABLED` | `0` | Enable the process-local exact disk tier. |
| `VLLM_EXACT_DISK_BACKUP_DIR` | `~/.cache/vllm/backup` | Root for process-incarnation bundles. |
| `VLLM_EXACT_DISK_BACKUP_CHUNK_BYTES` | `16777216` | Write/restore chunk size; positive and 4 KiB aligned. |
| `VLLM_EXACT_DISK_BACKUP_DIRECT_IO` | `1` | Request `O_DIRECT`, used with pinned host storage. |
| `VLLM_CPU_BACKUP_COORDINATOR` | empty | `http` or `daemon` enables metadata coordination. |
| `VLLM_CPU_BACKUP_COORDINATOR_URL` | unset | Coordinator base URL. |
| `VLLM_CPU_BACKUP_COORDINATOR_TIMEOUT_S` | `1.0` | Finite positive HTTP timeout. |
| `VLLM_CPU_BACKUP_COORDINATOR_CLIENT_ID` | host-derived | Logical prefix; a process-incarnation suffix is added. |
| `VLLM_CPU_BACKUP_COORDINATOR_MODEL_ID` | unset | Optional policy/model identity. |
| `VLLM_CPU_BACKUP_COORDINATOR_POLL_INTERVAL_S` | `0.1` | Finite interval; non-positive disables the poller. |
| `VLLM_SLEEP_PROFILE_PATH` | unset | Opt-in JSONL diagnostics path. |

Example coordinator configuration:

```text
VLLM_CPU_BACKUP_COORDINATOR=http
VLLM_CPU_BACKUP_COORDINATOR_URL=http://127.0.0.1:9000
VLLM_CPU_BACKUP_COORDINATOR_TIMEOUT_S=1.0
VLLM_CPU_BACKUP_COORDINATOR_CLIENT_ID=<logical-prefix>
VLLM_CPU_BACKUP_COORDINATOR_MODEL_ID=<model-alias>
VLLM_CPU_BACKUP_COORDINATOR_POLL_INTERVAL_S=0.1
```

Example exact disk configuration:

```text
VLLM_EXACT_DISK_BACKUP_ENABLED=1
VLLM_EXACT_DISK_BACKUP_DIR=/path/to/fast-local-backup
VLLM_EXACT_DISK_BACKUP_CHUNK_BYTES=16777216
VLLM_EXACT_DISK_BACKUP_DIRECT_IO=1
```

Optional profiling:

```text
VLLM_SLEEP_PROFILE_PATH=/path/to/profile.jsonl
```

### Diagnostic schema and failure policy

Every JSONL row includes `schema="switch.vllm.sleep-backup-profile"`,
`schema_version=1`, `diagnostic_only=true`, wall/monotonic timestamps, PID, and phase,
followed by phase-specific counters or timings. Diagnostics are disabled by default.
Directory creation, serialization, and append errors are best effort and do not replace
the allocator transition's result. Include `allocator_prepare_sleep` when accounting for
L1 sleep work; eager startup preparation is a separate phase.

Use the worker's named `sleep_extension("identity")` RPC to verify the instantiated native
or switch backend. Switch `sleep_extension("stats")` also reports runtime counters and
identity metadata. See [Explicit commands](delta-v0.22.1.md#explicit-commands-and-diagnostics).

## Startup order

Start the controller first, then prepare engines sequentially:

```text
controller ready
  -> launch A -> health -> sleep A
  -> launch B -> health -> sleep B
  -> wake configured startup alias
```

This permits one GPU to initialize models whose awake footprints cannot coexist. Worker
registration may be retried through later usage, but starting the controller first gives a
clean protocol failure signal.

## Safety expectations

- Keep management traffic on loopback or a private network.
- Never route inference before pool preparation finishes.
- Do not run multiple controller processes against one pool.
- Treat `RECOVERY_REQUIRED` or repeated lifecycle probe failures as an engine restart
  boundary.
- Validate physical CPU reclaim with process-tree RSS and host `MemAvailable`.
