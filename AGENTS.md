# Project Context

## Scope

This repository owns the external multi-backend control plane: model alias routing,
request reservations and drain, sleep/wake serialization, OpenAI proxying, aggregate
CPU-backup accounting, host-memory pressure policy, and safe launcher-owned process-group
lifecycle. This repository also owns `switch_runtime`, an engine-independent library
loaded inside each inference worker. It owns backup publication, D2H/H2D, exact disk
bundles, and concrete reclamation. Backup bytes never enter the controller service.

## Repository conventions

- Keep only current public architecture, integration, and operating instructions under
  `docs/`.
- Do not restore completed plans, historical reports, benchmark results, or performance
  images to the default branch; Git history preserves old material.
- Keep reusable configuration under `configs/*.example.yaml` and machine values only in
  ignored `configs/*.local.yaml` files.
- Keep live output in ignored `results/`, `tmp/`, or operator-selected paths.
- Put benchmark adapters, data collection, raw/curated evidence, plots, and reports in
  `vllm-switch-bench`.
- Keep public text in English and current tracked files free of developer-machine paths.
- Preserve fail-closed lifecycle behavior and exactly-once streaming reservations.
- Treat protocol versions, capabilities, process-incarnation identity, and PID ownership
  records as compatibility contracts; change them with tests and documentation.

## Verification

The fast verification gate is:

```bash
uv sync --frozen --dev
uv run python -m pytest tests -q
uv run ruff check vllm_switch_controller switch_runtime scripts tests
uv run ruff format --check vllm_switch_controller switch_runtime scripts tests
uv run mypy --ignore-missing-imports vllm_switch_controller switch_runtime
uv build
```

Also install the built wheel in an isolated environment and smoke every console entry
point.

Package versions are maintained only in packaging metadata. Do not change version numbers
or create tags or releases without explicit maintainer direction.

## Related repositories

- `../vllm-switch`: the vLLM fork; thin allocator notifications, worker lifecycle guards,
  provider registration, and collective sleep preparation.
- `../vllm-switch-bench`: cross-system benchmark adapters, raw/curated evidence, plots, and
  reports.
