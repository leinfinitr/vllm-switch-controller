# Configuration Reference

The controller reads one YAML document with exactly two top-level keys: `models` and
`controller`. Unknown fields are rejected instead of being ignored, so configuration
typos fail startup.

## Model entries

Each key under `models` is a public alias accepted by the OpenAI-compatible endpoints.

| Field | Required | Default | Description |
|---|---:|---:|---|
| `backend_url` | yes | - | HTTP(S) base URL of one single-model vLLM server. |
| `served_model_name` | yes | - | Non-empty model name written into forwarded bodies. |
| `sleep_level` | no | `1` | vLLM sleep level, `1` or `2`. |
| `wake_tags` | no | `null` | `null` uses the vLLM default; otherwise a non-empty unique string list. |
| `launch_command` | no | `null` | Non-empty argument vector used by `vllm-switch-launch`. |
| `env` | no | `{}` | String environment overrides for a launcher-managed process. |
| `cwd` | no | `null` | Working directory for a launcher-managed process. |

`backend_url` is normalized by removing trailing slashes. `launch_command` is passed
directly to `subprocess.Popen`; no shell is used.

### Exact disk environment contract

The controller does not inject a disk path. The compatible vllm-switch uses only these
canonical variables:

| Variable | Meaning |
|---|---|
| `VLLM_EXACT_DISK_BACKUP_ENABLED` | Enable the process-local exact disk tier. |
| `VLLM_EXACT_DISK_BACKUP_DIR` | Root for process-incarnation bundles. |
| `VLLM_EXACT_DISK_BACKUP_CHUNK_BYTES` | Positive, 4-KiB-aligned chunk size. |
| `VLLM_EXACT_DISK_BACKUP_DIRECT_IO` | Request direct I/O when supported. |

## Controller settings

| Field | Default | Constraint |
|---|---:|---|
| `host` | `127.0.0.1` | Listen address; remote binds require an explicit security decision. |
| `port` | `9000` | `1..65535`. |
| `policy` | `always_sleep_previous` | Known policy name. |
| `startup_awake_model` | `null` | Must name a configured alias when set. |
| `request_timeout_s` | `600` | Positive proxied-request timeout. |
| `switch_timeout_s` | `600` | Positive lifecycle transition deadline. |
| `metrics_path` | `results/controller_events.jsonl` | Per-request JSONL output. |

At least one model is required.

### Switching policies

- `always_sleep_previous`: keep one active model; drain and sleep it before waking a new
  target.
- `always_awake_previous`: keep already-awake models resident and wait for the target's
  pending requests before switching the active route. Use only when GPU capacity permits
  all awake footprints.

## Host-memory pressure settings

| Field | Default | Description |
|---|---:|---|
| `cpu_memory_reclaim_available_ratio` | `null` | Enter below this `MemAvailable / MemTotal` ratio. |
| `cpu_memory_recovery_available_ratio` | `null` | Leave at or above this ratio. |
| `cpu_memory_reclaim_available_bytes` | `0` | Absolute low watermark. |
| `cpu_memory_recovery_available_bytes` | `0` | Absolute high watermark. |
| `cpu_memory_poll_interval_s` | `0.5` | Positive `/proc/meminfo` polling interval. |
| `cpu_memory_pressure_consecutive_samples` | `3` | Positive low-sample debounce. |
| `cpu_memory_reclaim_cooldown_s` | `2.0` | Non-negative interval between rounds. |

Ratio watermarks must appear as a pair. Recovery must be greater than or equal to reclaim
for both ratios and bytes. When both units are configured:

```text
low  = max(MemTotal * reclaim_ratio, reclaim_bytes)
high = max(MemTotal * recovery_ratio, recovery_bytes)
```

## CPU backup policy

| Field | Default | Description |
|---|---:|---|
| `cpu_backup_global_cap_bytes` | `null` | Optional total-backup hard guard. |
| `cpu_backup_default_model_priority` | `0` | Default victim priority. |
| `cpu_backup_model_priorities` | `{}` | Per-model priorities; lower values go first. |

The controller sends byte budgets only. The `vllm-switch` process decides which local allocation is
safe to release. See [CPU Backup Coordinator](cpu_backup_coordinator.md).

## Local files

- Commit reusable examples as `configs/*.example.yaml`.
- Keep paths, model IDs, tokens, and host-specific settings in ignored
  `configs/*.local.yaml` files.
- Put transient metrics under ignored `results/` or `tmp/` paths.
- Put benchmark configurations and evidence in `vllm-switch-bench`, not this repository.

## Engine adapter

Each model accepts `engine: vllm` (the default). The shared client selects an
`EngineControlAdapter`; applications may inject additional adapters when constructing
`EngineClient`. The packaged CLI currently supplies the vLLM adapter. L2 resume restores
weights from the original checkpoint before KV wake and readiness verification.
