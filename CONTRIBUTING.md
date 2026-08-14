# Contributing

Thank you for improving the vllm-switch-controller. This repository is a research
control plane with correctness-sensitive lifecycle and streaming paths. Small, focused,
well-tested changes are easier to validate and reproduce.

## Before You Start

- Use the issue tracker to describe substantial behavior changes before implementation.
- Report security concerns privately as described in [SECURITY.md](SECURITY.md).
- Read [Architecture](docs/architecture.md) and the repository scope in `AGENTS.md`.
- Put allocator-local tensor behavior in the companion `vllm-switch` repository.
- Put cross-system benchmark harnesses, plots, and reports in `vllm-switch-bench`.

## Development Setup

The documented workflow uses Python 3.11 or newer and `uv`:

```bash
uv sync --frozen --dev
uv run python -m pytest tests -q
uv run ruff check vllm_switch_controller scripts tests
uv run ruff format --check vllm_switch_controller scripts tests
uv run mypy --ignore-missing-imports vllm_switch_controller
uv build
```

GPU hardware and a real vLLM checkout are not required for the unit suite. Tests use fake
HTTP backends for routing, lifecycle, cancellation, and streaming behavior. Clearly label
manual GPU validation separately from automated test results.

## Making a Change

1. Create a topic branch from the current default branch.
2. Keep the change within one ownership boundary and avoid unrelated refactoring.
3. Add or update focused tests for behavior changes.
4. Run the focused tests while iterating, then the full test and lint commands.
5. Update current documentation when configuration, API, lifecycle behavior, telemetry,
   or operating instructions change.
6. Inspect the final diff for machine paths, credentials, generated artifacts, and stale
   links.

### Request-Path Requirements

Changes to routing or lifecycle handling must preserve these contracts:

- Fail closed when a sleep/wake outcome is unknown.
- Verify lifecycle post-conditions before committing controller state.
- Drain a model before a policy sleeps it.
- Reserve the target request before releasing the switch lock.
- Hold one reservation for the complete JSON or streaming request lifetime.
- Release reservations exactly once across completion, disconnect, setup failure, and
  repeated cancellation.
- Preserve backend status codes and end-to-end response headers.
- Keep metrics writes best-effort so observability failures do not replace inference
  outcomes.

Tests for cancellation and streaming ownership should cover both failures before body
iteration and failures during iteration.

### CPU Backup Requirements

The controller is metadata-only. Do not move pinned tensors, D2H/H2D, validity decisions,
or concrete victim selection into this repository. Protocol changes must preserve:

- exact aggregate byte accounting;
- process-incarnation identity;
- monotonic cumulative commands within a controller epoch;
- acknowledgement based on actual allocator storage decrease;
- protection of restore-required and copy-in-flight bytes;
- separate evidence for logical and physical reclaim.

Update [CPU Backup Coordinator](docs/cpu_backup_coordinator.md) with any protocol or
failure-semantics change.

## Documentation and Configuration

- Write project documentation, comments, examples, and user-facing text in English.
- Keep current architecture and operating instructions under `docs/`.
- Keep only current public documentation under `docs/`; Git history preserves completed
  plans and historical reports.
- Keep reusable examples under `configs/*.example.yaml` and machine values only in
  ignored `configs/*.local.yaml` files.
- Use relative repository paths in current documentation.
- Preserve exact machine paths only when they are evidence in an archived report.
- Check that documented commands match CLI `--help` and configuration validation.

## Data and Secrets

Do not commit:

- `configs/*.local.yaml` machine configuration;
- access tokens, private endpoints, or credentials;
- model weights or downloaded datasets;
- PID files, logs, or ad hoc run output;
- raw benchmark runs intended for `vllm-switch-bench`.

Use ignored `results/tmp/` or `tmp/` paths for transient output. Curated evidence must be
small, reproducible, documented, and intentionally reviewed.

## Pull Request Checklist

- [ ] The change has one clear purpose and stays within repository scope.
- [ ] New behavior has focused tests.
- [ ] `uv run python -m pytest tests -q` passes.
- [ ] `uv run ruff check vllm_switch_controller scripts tests` passes.
- [ ] `uv build` and isolated wheel/CLI smoke checks pass.
- [ ] User-facing docs and example configuration are updated in English.
- [ ] Lifecycle and exactly-once reservation contracts are preserved.
- [ ] No secrets, machine-local paths, or transient artifacts are included.
- [ ] Manual GPU checks, skipped tests, and remaining limitations are stated explicitly.

Commit messages should be imperative and scoped, for example:

```text
docs: reorganize operator documentation
fix: preserve reservation during stream setup
test: cover delayed wake post-condition
```
