## Summary

<!-- Explain the problem and the focused change. -->

## Ownership boundary

- [ ] Controller-only change
- [ ] Requires a coordinated `vllm-switch` change
- [ ] Requires a benchmark/artifact change

## Correctness checklist

- [ ] Unknown lifecycle outcomes still fail closed.
- [ ] Request reservations remain exactly once across all touched paths.
- [ ] Protocol/configuration changes include compatibility docs and tests.
- [ ] No machine-local paths, credentials, or benchmark output are included.

## Test plan

```text
uv sync --frozen --dev
uv run python -m pytest tests -q
uv run ruff check vllm_switch_controller scripts tests
uv build
```

<!-- Add focused tests and any real backend/GPU verification performed. -->
