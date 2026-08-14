# vllm-switch Testing

Each engine change must revalidate:

- CuMem allocator tag and sleep semantics;
- worker initialization order and CUDA graph readiness;
- supported model mutation, reload, and EPLB paths;
- management endpoint query parameters and post-conditions;
- coordinator wire schema;
- exact disk manifest and failure fencing;
- CPU RSS and GPU inference correctness after reclaim and restore.

## CPU-focused tests

Run from the companion `vllm-switch` environment:

```bash
.venv/bin/python -m pytest -q \
  tests/basic_correctness/test_cpu_backup_coordinator.py \
  tests/basic_correctness/test_cumem.py \
  tests/basic_correctness/test_exact_disk_backup.py \
  tests/v1/executor/test_sleep_cpu_backup.py \
  tests/v1/worker/test_weight_backup_lifecycle.py \
  tests/v1/worker/test_eplb_cpu_backup.py
```

Record Python, PyTorch, CUDA, the engine commit, and complete output.

## GPU validation

A GPU validation run should additionally exercise:

1. default CUDA-graph or supported production engine initialization;
2. first level-1 sleep using eager prebackup with zero weight D2H;
3. wake followed by output-equality inference;
4. staged `weights` then `kv_cache` wake when that mode is claimed;
5. repeated same-process sleep/wake demonstrating clean backup reuse;
6. controlled reclaim with logical acknowledgement, RSS drop, and `MemAvailable` recovery;
7. post-reclaim sleep rebuilding CPU backup and returning to reuse;
8. exact disk spill, reclaim, and restore with checksum verification;
9. corrupt or missing exact disk data failing closed;
10. two-model request-driven switching through the controller.

GPU checks belong on a dedicated runner or in a recorded validation workflow, not in the
controller's hardware-free unit CI.

## Integration failures

| Symptom | Likely mismatch |
|---|---|
| Coordinator `422` | Protocol version or capabilities are absent or different. |
| Unknown exact-disk variable | Engine does not implement the canonical `VLLM_EXACT_DISK_BACKUP_*` contract. |
| `/is_sleeping` missing or non-boolean | Wrong management API or development mode disabled. |
| Partial wake fails inference | Configured tags do not restore all required allocations. |
| Usage accepted but no physical reclaim | Host-cache flush failed or no bytes were reclaimable. |
| Snapshot reused after an out-of-tree mutation | Mutation path did not invalidate the weights tag. |
