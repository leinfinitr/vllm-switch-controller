# vLLM Model Switch Controller

[![CI](https://github.com/leinfinitr/vllm-switch/actions/workflows/ci.yml/badge.svg)](https://github.com/leinfinitr/vllm-switch/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

An experimental external control plane that routes OpenAI-compatible requests across
long-lived, single-model vLLM backends and serializes their sleep/wake lifecycle.

The default branch currently identifies as the PEP 440 development version
`0.2.0.dev0`. The latest published controller release is `v0.1.5`; use exact commits when
combining development checkouts across repositories.

> [!WARNING]
> The controller has no authentication, authorization, or TLS. Its data and management
> APIs share one listener. Keep it on loopback or a trusted management network and place
> an authenticated reverse proxy in front of it when remote access is required.

## Capabilities

For upstream [vLLM](https://github.com/vllm-project/vllm) backends, the controller:

- Routes `/v1/chat/completions` and `/v1/completions` by the request's `model` alias.
- Lists configured aliases at `/v1/models`.
- Drains in-flight requests before sleeping their backend.
- Serializes transitions and verifies `/is_sleeping` post-conditions.
- Holds exactly one reservation for complete JSON and streaming request lifetimes.
- Launches a single-GPU backend pool sequentially and stops only verified owned groups.

For modified [vLLM Switch fork](https://github.com/leinfinitr/vllm) backends, additionally:

- Coordinates aggregate CPU-backup accounting without owning tensors or copies.
- Requests cooperative CPU-backup reclaim from vLLM under host-memory pressure.

vLLM Switch is useful for:

- Running multiple models on a single GPU with minimal HBM usage.
- Managing sleep/wake cycles across multiple vLLM processes.
- Use idle CPU memory for backup storage and reclaiming it when needed.
- Use direct I/O for exact disk snapshots and fast recovery.

The companion [vLLM Switch fork](https://github.com/leinfinitr/vllm) owns pinned CPU
backups, eager prebackup, D2H/H2D, validity, concrete reclaim, and exact disk snapshots.
The [llm-switch-bench](https://github.com/leinfinitr/llm-switch-bench) repository owns
cross-system experiments, results, plots, and phase artifacts.

## Requirements

- Linux and Python 3.11 or newer.
- [`uv`](https://docs.astral.sh/uv/) for the source workflow.
- The compatible vLLM Switch fork for the full backup feature set. Stock vLLM can supply 
  basic sleep endpoints but not this coordinator contract.
- Enough GPU memory to initialize each configured model individually.

See the exact [compatibility contract](docs/compatibility.md), machine-readable
[`v0.1` suite manifest](compatibility/v0.1.yaml), and
[vLLM fork delta](docs/vllm-fork/README.md) before combining revisions.

## Install

From a checkout:

```bash
uv sync --frozen --dev
uv run vllm-switch-controller --help
```

Or install a built wheel:

```bash
uv build
uv tool install dist/vllm_switch_controller-0.2.0.dev0-py3-none-any.whl
vllm-switch-controller --version
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
- [vLLM fork delta and integration](docs/vllm-fork/README.md)
- [Compatibility matrix](docs/compatibility.md)
- [v0.1 release notes](docs/release-notes.md)

## Scope and status

`0.2.0.dev0` is under development after the `v0.1.5` research release. Neither line is a
production gateway. The controller intentionally omits replica scheduling,
multi-controller coordination, durable state, built-in authentication, and automatic
recovery from backend process loss. The supported topology is one controller process
managing trusted, explicitly configured single-model backends.

Benchmark code and historical experiment archives are intentionally absent from this
repository. Use [llm-switch-bench](https://github.com/leinfinitr/llm-switch-bench) for reproducible performance evaluation.

## Development

```bash
uv sync --frozen --dev
uv run python -m pytest tests -q
uv run ruff check controller scripts tests
uv run ruff format --check controller scripts tests
uv run mypy --ignore-missing-imports controller
uv build
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the
[Apache-2.0 license](LICENSE).
