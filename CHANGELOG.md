# Changelog

All notable public changes are documented here. This project follows semantic versioning
for the controller package.

## [Unreleased]

- Renamed the launcher-managed and externally managed example configurations so their
  process-ownership semantics are explicit.
- Marked default-branch packages as the `0.2.0.dev0` development version and added
  cross-metadata consistency checks.
- Simplified policy descriptions and removed obsolete policy attributes and tests.
- Expanded the current setup, architecture, compatibility, and operating documentation.

## [0.1.5] - 2026-08-05

- Published controller package `0.1.5`.
- Aligned the coordinated engine release identity and successively pinned the benchmark
  release through `v0.1.8` as that artifact line closed.

## [0.1.4] - 2026-08-05

- Detached cleanup for read-only lifecycle probes that ignored cancellation, preventing a
  non-cooperative transport from extending the configured switch deadline.
- Published controller package `0.1.4`.

## [0.1.3] - 2026-08-05

- Published controller package `0.1.3`; this tag contains the same controller behavior as
  `v0.1.2` and updates release metadata only.

## [0.1.2] - 2026-08-05

- Published controller package `0.1.2`.
- Relaxed wall-clock-only deadline assertions to avoid scheduler-dependent test flakes;
  controller runtime behavior was unchanged.

## [0.1.1] - 2026-08-05

- Reconciled configured backend state from lifecycle probes before routing the first
  request and failed closed on ambiguous lifecycle outcomes.
- Applied one end-to-end deadline across startup reconciliation, request drain, sleep,
  wake, and post-condition probes.
- Invalidated stale launcher PID success records on failed reruns and strengthened
  process-group cleanup when the recorded leader had already exited.
- Documented coordinated patch and evidence identities and published package `0.1.1`.

## [0.1.0] - 2026-08-04

- Added the first public controller research-preview release.
- Added versioned CPU-backup protocol metadata and capability validation.
- Added strict configuration, safe loopback defaults, malformed JSON handling, and
  process-incarnation conflict detection.
- Added launcher wake-tag parity and verified process-group ownership/shutdown.
- Added Python packaging, console scripts, CI, community files, and public integration
  documentation.
- Removed controller-local benchmark/archive assets in favor of `llm-switch-bench`.

[Unreleased]: https://github.com/leinfinitr/vllm-switch/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/leinfinitr/vllm-switch/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/leinfinitr/vllm-switch/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/leinfinitr/vllm-switch/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/leinfinitr/vllm-switch/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/leinfinitr/vllm-switch/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/leinfinitr/vllm-switch/releases/tag/v0.1.0
