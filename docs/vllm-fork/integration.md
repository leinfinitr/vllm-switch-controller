# Controller–vLLM Integration

## Backend lifecycle endpoints

The controller uses vLLM development endpoints enabled by:

```text
VLLM_SERVER_DEV_MODE=1
--enable-sleep-mode
```

Required endpoints and their contract are listed in
[Compatibility](../compatibility.md). The controller treats a lifecycle POST and its
`/is_sleeping` post-condition as one transition. A timeout or ambiguous outcome fails
closed.

## Wake tags

The vLLM allocator assigns at least the `weights` and `kv_cache` tags in the supported
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

## CPU backup protocol

The controller requires protocol version `1` on registration and usage. Workers must
send a stable capability set and a complete non-reusable process-incarnation `client_id`.
Exact-disk aggregate fields require `exact-disk-accounting-v1`.

Workers without explicit protocol/capability fields are incompatible. Pair this controller
only with an engine checkout that implements the same contract. A
`422` registration/usage response is an integration mismatch, not a reason to weaken
validation.

## Canonical environment variables

Coordinator:

```text
VLLM_CPU_BACKUP_COORDINATOR=http
VLLM_CPU_BACKUP_COORDINATOR_URL=http://127.0.0.1:9000
VLLM_CPU_BACKUP_COORDINATOR_TIMEOUT_S=1.0
VLLM_CPU_BACKUP_COORDINATOR_CLIENT_ID=<logical-prefix>
VLLM_CPU_BACKUP_COORDINATOR_MODEL_ID=<model-alias>
VLLM_CPU_BACKUP_COORDINATOR_POLL_INTERVAL_S=0.1
```

Exact disk:

```text
VLLM_EXACT_DISK_BACKUP_ENABLED=1
VLLM_EXACT_DISK_BACKUP_DIR=/path/to/fast-local-backup
VLLM_EXACT_DISK_BACKUP_CHUNK_BYTES=16777216
VLLM_EXACT_DISK_BACKUP_DIRECT_IO=1
```

Profiling, when needed:

```text
VLLM_SLEEP_PROFILE_PATH=/path/to/profile.jsonl
```

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
