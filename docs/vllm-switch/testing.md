# vllm-switch testing

## Engine-independent tests

From this repository, run the fast verification commands in [Runtime](../runtime.md).
The fake memory backend exercises actual byte copying and simulated unmap without vLLM,
PyTorch, or CUDA. Tests cover publication, leases, immutable reuse, mutable buffers,
reclaim concurrency, allocation identity, disk corruption, protocol retry/ordering, and
reconstruction guards. A second fake engine control adapter exercises shared deadlines
without vLLM management endpoints.

## Engine bridge and GPU tests

Install this package in the engine environment, then run from the engine checkout:

```bash
.venv/bin/python -m pytest -q \
  tests/v1/executor/test_sleep_cpu_backup.py \
  tests/v1/worker/test_weight_backup_lifecycle.py \
  tests/basic_correctness/test_exact_disk_backup_gpu.py
```

The GPU test validates exact disk demotion/restoration and CUDA graph address reuse.
Record Python, PyTorch, CUDA, both repository commits and dirty state, and full output.

Additional dedicated-runner validation should exercise:

1. default engine initialization and production graph configuration;
2. L1 snapshot reuse, accounting separately for mutable buffer copies;
3. inference output equality after repeated and staged wake;
4. L2 original-checkpoint reconstruction before inference and fresh snapshots;
5. logical reclaim acknowledgement and independent worker RSS/host-memory evidence;
6. exact disk write/reclaim/restore, corruption, and copy-fencing failures;
7. two-backend request-driven switching through the controller;
8. provider-disabled native allocator behavior.

Benchmark adapters, evidence, and performance comparisons live in `vllm-switch-bench`.
CPU tests and smoke runs do not establish broad model, multi-rank, or production support.

## Integration failures

| Symptom | Likely cause |
|---|---|
| Provider unavailable | Wheel missing in engine environment or plugin omitted from allowlist |
| Provider API mismatch | Engine bridge and runtime use different local interface versions |
| Coordinator `422` | Missing/incompatible protocol version or capabilities |
| Model mutation rejected | Operation violates the fixed-weight inference contract |
| Snapshot blocked after L2 | Checkpoint reconstruction has not completed |
| Logical release without RSS reduction | Host-cache flush unavailable/failed or allocator retained memory |
| Disk corruption or partial restore | Recovery-required state; restart the engine |
