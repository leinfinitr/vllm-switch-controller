# Getting Started

This guide starts the controller and a launcher-managed pool of single-model vllm-switch
backends. Read the [compatibility matrix](compatibility.md) first; upstream vLLM does not
implement the CPU-backup coordinator contract.

## 1. Install and verify

Requirements:

- Linux;
- Python 3.11 or newer;
- `uv`;
- a CUDA environment supported by the selected vllm-switch revision;
- enough GPU memory to initialize each model separately.

From the repository root:

```bash
uv sync --frozen --dev
uv run python -m pytest tests -q
uv run ruff check vllm_switch_controller scripts tests
```

The controller package deliberately does not depend on vLLM, PyTorch, NumPy, or plotting
libraries. Backends run in their own environment.

## 2. Create an ignored local configuration

For launcher-managed backends:

```bash
cp configs/models.launcher.example.yaml configs/models.local.yaml
$EDITOR configs/models.local.yaml
```

For already-running backends, copy `configs/models.external.example.yaml` instead. Files
matching `configs/*.local.yaml` are ignored by Git. The launcher example includes
`launch_command`, `env`, and `cwd`; the external example intentionally omits process
ownership and only addresses pre-existing backend URLs.

Each public alias needs a backend URL and the name that backend serves:

```yaml
models:
  model-a:
    backend_url: http://127.0.0.1:8101
    served_model_name: model-a
    sleep_level: 1
    wake_tags: null
  model-b:
    backend_url: http://127.0.0.1:8102
    served_model_name: model-b
    sleep_level: 1
    wake_tags: null

controller:
  host: 127.0.0.1
  port: 9000
  policy: always_sleep_previous
  startup_awake_model: model-a
```

`wake_tags: null` asks vLLM to restore every sleeping allocation. A non-empty list is sent
as repeated `tags` parameters. Use partial wake only when the selected tags restore every
allocation needed by subsequent inference.

## 3. Configure the compatible vllm-switch

Every backend must expose `/health`, `/sleep`, `/wake_up`, and `/is_sleeping`. Launcher
commands need at least:

```text
VLLM_SERVER_DEV_MODE=1
vllm serve /path/to/model --enable-sleep-mode
```

The CPU-backup coordinator additionally uses:

```text
VLLM_CPU_BACKUP_COORDINATOR=http
VLLM_CPU_BACKUP_COORDINATOR_URL=http://127.0.0.1:9000
VLLM_CPU_BACKUP_COORDINATOR_CLIENT_ID=<logical-prefix>
VLLM_CPU_BACKUP_COORDINATOR_MODEL_ID=<model-alias>
```

The logical client prefix is not a process identity. The compatible worker appends PID and
start-time information so a restarted process cannot inherit pending commands.

Exact disk backup is opt-in. Use only:

```text
VLLM_EXACT_DISK_BACKUP_ENABLED=1
VLLM_EXACT_DISK_BACKUP_DIR=/path/to/fast-local-backup
```

## 4. Start the controller

Start the controller before coordinator-enabled workers register:

```bash
uv run vllm-switch-controller --config configs/models.local.yaml
```

The default bind address is loopback. Do not expose the listener directly to an untrusted
network.

## 5. Prepare the backend pool

In another shell:

```bash
uv run vllm-switch-launch \
  --config configs/models.local.yaml \
  --pid-file pids.json
```

The launcher processes models in configuration order:

```text
launch or locate backend
  -> wait for /health
  -> sleep and verify
  -> continue to the next backend
  -> wake startup_awake_model with its configured tags
  -> verify awake
  -> atomically write the ownership file
```

Use `--skip-launch` to prepare externally managed backends. Such processes are not added
to the ownership file and will not be stopped by `vllm-switch-stop`.

## 6. Send requests

```bash
curl -fsS http://127.0.0.1:9000/health
curl -fsS http://127.0.0.1:9000/admin/state
curl -fsS http://127.0.0.1:9000/v1/models
```

```bash
curl -fsS http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "model-b",
    "messages": [{"role": "user", "content": "Reply with one word."}],
    "max_tokens": 8
  }'
```

If another model is active, the controller drains its requests, sleeps it, verifies the
post-condition, wakes `model-b`, reserves the new request, and forwards it.

## 7. Stop the pool

```bash
uv run vllm-switch-stop --pid-file pids.json
```

The command signals launcher-created process groups, not just leaders. Before sending a
signal it validates PID, PGID, and Linux process start time. An unverifiable ownership file
is retained and the command exits nonzero.

Stop the controller separately with `Ctrl-C`.

## Next steps

- [Configuration reference](configuration.md)
- [Operations and troubleshooting](operations.md)
- [vLLM integration and limitations](vllm-switch/README.md)
