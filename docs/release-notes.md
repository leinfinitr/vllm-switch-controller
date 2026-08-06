# v0.1.5 Release Notes

v0.1.5 is the first public research-preview release of the vLLM Switch controller.

## Highlights

- OpenAI-compatible alias routing for chat and text completions.
- Fail-closed request drain and sleep/wake serialization.
- Exactly-once reservations across JSON, streaming, disconnect, and cancellation paths.
- Metadata-only CPU backup coordination with dynamic host-memory pressure reclaim.
- Explicit CPU backup protocol version and capability declarations.
- Process-incarnation conflict detection.
- Canonical `VLLM_EXACT_DISK_BACKUP_*` configuration contract.
- Safe loopback defaults and strict configuration validation.
- Sequential launcher semantics with `wake_tags` parity.
- PID/PGID/start-time ownership files and verified process-group shutdown.
- Buildable Apache-2.0 Python package with four console entry points.

## Included companion vLLM mechanisms

The coordinated engine release includes reusable pinned CPU clean backups, eager
prebackup, mutation invalidation, transactional level-1 sleep, metadata coordination,
dynamic reclaim, and the supported v0.1 exact disk backup tier.

The coordinated release pins engine tag `aipc2-v0.1.0` at
`71071ce4d0bc65e38acf2da76eb8c6fb05b9454d` and benchmark tag `v0.1.8` at
`e4e388acc33977bee7ca19d72a2959fc736d76ab`. The machine-readable suite binding is
[`compatibility/v0.1.yaml`](../compatibility/v0.1.yaml). Retained run-specific source and
collection identities remain in the benchmark artifact's provenance rather than the
runtime compatibility contract. See [Compatibility](compatibility.md).

## Breaking changes from development checkouts

- The package distribution is named `vllm-switch-controller`.
- The controller defaults to `127.0.0.1`, not `0.0.0.0`.
- Unknown YAML keys now fail validation.
- CPU backup worker requests are versioned and capability-declared.
- Usage reports require `released_bytes_total` under protocol v1.
- Use `VLLM_EXACT_DISK_BACKUP_DIR` when exact disk backup is explicitly enabled.
- Launcher PID files use schema version 1 and cannot be consumed as legacy `{name: pid}`
  maps.
- Controller-local benchmark scripts, archived plans, and historical results moved out of
  this release repository; use `llm-switch-bench`.

## Known limitations

- Research software; no built-in authentication, authorization, TLS, or rate limiting.
- One in-memory controller process per pool; no durable or distributed state.
- No worker lease/heartbeat; crashed-worker coordinator records persist until restart.
- Host pressure reads host-global `/proc/meminfo`.
- Exact disk backup is supported for this Linux/NVIDIA research-preview scope and remains
  sensitive to local filesystem/direct-I/O support.
- The compatible engine remains pinned to upstream vLLM `v0.22.1`.
