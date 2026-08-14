# Operations and Troubleshooting

Follow [Getting Started](getting-started.md) before this guide.

## Startup

```bash
# Shell 1
uv run vllm-switch-controller --config configs/models.local.yaml

# Shell 2
mkdir -p results/tmp/request-switch
uv run vllm-switch-launch \
  --config configs/models.local.yaml \
  --pid-file results/tmp/request-switch/pids.json
```

The launcher starts one process group per `launch_command`, waits for health, verifies each
sleep, and applies the configured startup wake tags. A partial failure terminates and reaps
all process groups created by that invocation. Externally managed backends are never
signalled.

The ownership file is atomically published only after successful pool preparation. It
contains PID, PGID, and Linux process start time for each owned process.

## Health and state

```bash
curl -fsS http://127.0.0.1:9000/health
curl -fsS http://127.0.0.1:9000/admin/state
curl -fsS http://127.0.0.1:9000/v1/models
curl -fsS http://127.0.0.1:9000/admin/cpu-backup/stats
```

`/health` reports controller state but does not probe backends. Query each backend's
`/health` and `/is_sleeping` directly when diagnosing lifecycle uncertainty.

## Request-path smoke test

```bash
uv run vllm-switch-smoke \
  --base-url http://127.0.0.1:9000 \
  --models model-a model-b \
  --output results/tmp/request-switch/smoke.jsonl
```

The verifier requires semantic streaming output and `[DONE]`, exercises A-B-B-A routing,
uses an active-request barrier for a controlled drain overlap, and rejects leaked request
reservations.

## CPU backup pressure validation

```bash
uv run python scripts/validate_backup_pressure.py \
  --base-url http://127.0.0.1:9000 \
  --mode retain \
  --model-a model-a \
  --model-b model-b \
  --output results/tmp/backup-pressure/retain.json

uv run python scripts/validate_backup_pressure.py \
  --base-url http://127.0.0.1:9000 \
  --mode release \
  --model-a model-a \
  --model-b model-b \
  --output results/tmp/backup-pressure/release.json
```

Release validation requires all of the following:

- a positive queued release target;
- monotonic worker acknowledgement;
- the same worker process incarnation remains alive;
- a material process-tree RSS decrease;
- a material host `MemAvailable` increase;
- zero pending release bytes.

Logical accounting alone is not physical-reclaim evidence. On a shared host, trigger the
policy with safe test watermarks; do not create artificial host exhaustion.

## Shutdown

```bash
uv run vllm-switch-stop \
  --pid-file results/tmp/request-switch/pids.json
```

The stop command sends `SIGTERM` to verified process groups and escalates to `SIGKILL`
after the timeout. It refuses to signal when:

- the PID file schema is unknown or malformed;
- PID, PGID, or start time does not match the recorded process incarnation;
- the recorded PGID is the stop command's own group.

A refusal leaves the ownership file in place and exits nonzero. Inspect it and the live
`/proc` state rather than manually trusting a reused PID. Stop the controller separately
with `Ctrl-C`.

## Common failures

### Configuration fails at startup

Unknown keys are forbidden. Use the validation message to locate a typo. Ports must be in
`1..65535`, timeouts and polling intervals must be positive, watermark pairs must be
complete, and `wake_tags` must be null or a non-empty unique list.

### A backend never becomes healthy

Run its configured argument vector directly. Check the vllm-switch revision, model path,
CUDA/PyTorch compatibility, port availability, and that the command uses the backend
virtual environment rather than the controller environment.

### Sleep or wake returns `502`

Verify `VLLM_SERVER_DEV_MODE=1`, `--enable-sleep-mode`, and `/is_sleeping`. The controller
commits a lifecycle state only after the management response and matching probe. A failed,
timed-out, or cancelled transition enters an error barrier and is reconciled before a
later switch.

### A partial wake cannot serve inference

`wake_tags` is an advanced vLLM feature. `null` is the safe default and wakes all sleeping
allocations. If a list is configured, it is sent unchanged as repeated query parameters;
the operator must include all allocations required by the subsequent operation.

### Coordinator requests return `422`

The wire contract requires `protocol_version: 1`, a declared capabilities list, and a
monotonic `released_bytes_total` on usage reports. Exact-disk fields require
`exact-disk-accounting-v1`. This normally means the controller and vllm-switch revisions do
not match; consult [Compatibility](compatibility.md).

### Coordinator requests return `409`

The same complete client ID attempted to change PID, protocol metadata, capabilities, or a
monotonic counter. A restarted worker must use a new process-incarnation ID.

### CPU backup records outlive a crashed worker

The protocol has no lease or heartbeat. Stale records remain conservatively accounted. Confirm the
worker is gone and restart the controller to clear in-memory coordinator state.

### Local control traffic reaches a proxy

Controller-to-backend and launcher traffic bypass environment proxies. Configure external
clients with `NO_PROXY` for loopback/private endpoints if necessary.

## Development checks

```bash
uv sync --frozen --dev
uv run python -m pytest tests -q
uv run ruff check vllm_switch_controller scripts tests
uv run ruff format --check vllm_switch_controller scripts tests
uv run mypy --ignore-missing-imports vllm_switch_controller
uv build
```

Cross-system workloads, resource collection, plots, and curated results belong in
`vllm-switch-bench`.
