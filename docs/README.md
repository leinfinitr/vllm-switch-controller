# Documentation

Current documentation for the `0.2.0.dev0` development version of the vLLM Model Switch
Controller. The latest published release is `v0.1.5`.

## Start here

- [Getting started](getting-started.md): install, configure, launch, request, and stop.
- [Compatibility](compatibility.md): pinned engine base, protocol, and environment contract.
- [Operations and troubleshooting](operations.md): validation, safe cleanup, and failures.

## Reference

- [Configuration](configuration.md): strict YAML fields and defaults.
- [API](api.md): OpenAI-compatible and administrative endpoints.
- [Architecture](architecture.md): request ownership and failure invariants.
- [CPU backup coordinator](cpu_backup_coordinator.md): accounting and reclaim semantics.

## Companion vLLM fork

- [Fork overview](vllm-fork/README.md)
- [Delta from upstream v0.22.1](vllm-fork/delta-v0.22.1.md)
- [Controller–vLLM integration](vllm-fork/integration.md)
- [Fork compatibility and testing](vllm-fork/compatibility.md)

## Release

- [v0.1.5 release notes](release-notes.md)

Development plans, historical experiments, benchmark scripts, and performance results are
not part of this release repository. Git history preserves prior development, while
`llm-switch-bench` owns current reproducible evaluation artifacts.
