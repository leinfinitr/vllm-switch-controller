# vllm-switch-controller

[![CI](https://github.com/leinfinitr/vllm-switch-controller/actions/workflows/ci.yml/badge.svg)](https://github.com/leinfinitr/vllm-switch-controller/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

An experimental external control plane that routes OpenAI-compatible requests across
long-lived, single-model vLLM backends and serializes their sleep/wake lifecycle.

> [!WARNING]
> The controller has no authentication, authorization, or TLS. Its data and management
> APIs share one listener. Keep it on loopback or a trusted management network and place
> an authenticated reverse proxy in front of it when remote access is required.

Unqualified `vLLM` refers to the upstream project and its baseline behavior. The fork-specific
mechanisms in this repository are named `vllm-switch`.

## Capabilities

For upstream [vLLM](https://github.com/vllm-project/vllm) backends, the controller:

- Routes `/v1/chat/completions` and `/v1/completions` by the request's `model` alias.
- Lists configured aliases at `/v1/models`.
- Drains in-flight requests before sleeping their backend.
- Serializes transitions and verifies `/is_sleeping` post-conditions.
- Holds exactly one reservation for complete JSON and streaming request lifetimes.
- Launches a single-GPU backend pool sequentially and stops only verified owned groups.

For [vllm-switch](https://github.com/leinfinitr/vllm-switch) backends, additionally:

- Coordinates aggregate CPU-backup accounting without owning tensors or copies.
- Requests cooperative CPU-backup reclaim from vllm-switch under host-memory pressure.

vllm-switch is useful for:

- Running multiple models on a single GPU with minimal HBM usage.
- Managing sleep/wake cycles across multiple vllm-switch processes.
- Using idle CPU memory for backup storage and reclaiming it when needed.
- Using direct I/O for exact disk snapshots and recovery.

The companion [vllm-switch](https://github.com/leinfinitr/vllm-switch) owns pinned CPU
backups, eager prebackup, D2H/H2D, validity, concrete reclaim, and exact disk snapshots.
The [vllm-switch-bench](https://github.com/leinfinitr/vllm-switch-bench) repository owns
cross-system experiments, results and plots.

## Requirements

- Linux and Python 3.11 or newer.
- [`uv`](https://docs.astral.sh/uv/) for the source workflow.
- The companion vllm-switch for the full backup feature set. Upstream vLLM can supply
  basic sleep endpoints but not this coordinator contract.
- Enough GPU memory to initialize each configured model individually.

See the [compatibility contract](docs/compatibility.md) and
[vLLM fork integration](docs/vllm-fork/README.md).

## Install

From a checkout:

```bash
uv sync --frozen --dev
uv run vllm-switch-controller --help
```

Or install a built wheel:

```bash
uv build
uv tool install dist/*.whl
```

## Quick start

Create a local configuration. Machine paths belong only in ignored `*.local.yaml` files:

```bash
cp configs/models.launcher.example.yaml configs/models.local.yaml
$EDITOR configs/models.local.yaml
```

Start the controller:

```bash
uv run vllm-switch-controller --config configs/models.local.yaml
```

In another shell, launch or prepare each backend sequentially:

```bash
uv run vllm-switch-launch \
  --config configs/models.local.yaml \
  --pid-file pids.json
```

Do not send traffic until pool preparation succeeds. Then list aliases and make a request:

```bash
curl -fsS http://127.0.0.1:9000/v1/models

curl -fsS http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "model-b",
    "messages": [{"role": "user", "content": "Reply with one word."}],
    "max_tokens": 8
  }'
```

Changing only `model` drives any necessary drain, sleep, wake, and routing operation.
Stop launcher-owned backends with:

```bash
uv run vllm-switch-stop --pid-file pids.json
```

The stop command validates PID, process-group ID, and Linux process start time before
signalling the process group. It refuses legacy or stale PID files instead of risking an
unrelated process.

## Documentation

- [Getting started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [Operations and troubleshooting](docs/operations.md)
- [API reference](docs/api.md)
- [Architecture and safety invariants](docs/architecture.md)
- [CPU backup coordinator protocol](docs/cpu_backup_coordinator.md)
- [vLLM fork integration](docs/vllm-fork/README.md)
- [Compatibility matrix](docs/compatibility.md)

## Development

```bash
uv sync --frozen --dev
uv run python -m pytest tests -q
uv run ruff check vllm_switch_controller scripts tests
uv run ruff format --check vllm_switch_controller scripts tests
uv run mypy --ignore-missing-imports vllm_switch_controller
uv build
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the
[Apache-2.0 license](LICENSE).
