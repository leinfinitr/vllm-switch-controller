# Compatibility Contract

Compatibility is defined by wire protocols and explicitly published component
combinations, not by repository names or nearby commit dates. Pin every component in a
released deployment. For automation, use the published
[`compatibility/v0.1.yaml`](../compatibility/v0.1.yaml) manifest.

The controller's current package identity is `0.2.0.dev0`. Development checkouts are not a
published suite combination: record their exact commits and revalidate the contracts below.

## Published suite combinations

| Suite | Controller | Engine | Benchmark artifact | Protocols |
|---|---|---|---|---|
| v0.1 | `v0.1.5` (`8b64ad232c8eba5d0da265abd93bd7b061db0549`) | `aipc2-v0.1.0` (`71071ce4d0bc65e38acf2da76eb8c6fb05b9454d`) | `v0.1.8` (`e4e388acc33977bee7ca19d72a2959fc736d76ab`) | CPU backup v1; exact disk manifest v1 |

The engine in this combination is based on upstream vLLM `v0.22.1` at
`0decac0d96c42b49572498019f0a0e3600f50398`. Tags are independent across repositories;
the engine keeps upstream's tag namespace, so its historical suite tag has an `aipc2-`
prefix.

The benchmark component is listed to identify the published reproducibility bundle. It is
not a runtime dependency of the controller or engine. Data-collection and artifact-closure
commits belong in benchmark provenance, not in this long-term compatibility contract.

## CPU backup protocol v1

Protocol v1 defines these controller/worker capabilities:

```text
cumulative-release-v1
released-bytes-total-v1
process-incarnation-v1
exact-disk-accounting-v1
```

The first three capabilities are required. `exact-disk-accounting-v1` is optional unless a
worker reports exact-disk aggregate fields. Registration and usage requests declare a
stable capability set. Unknown capabilities, an unsupported `protocol_version`, or missing
required metadata fail validation. A client cannot change PID, protocol version, or
capabilities while retaining the same complete process-incarnation ID.

Older engine commits without the explicit handshake receive no valid coordinator usage;
basic OpenAI routing and sleep/wake can still work independently when their management API
satisfies the contract below.

## vLLM management API contract

Every managed backend must provide:

| Endpoint | Required result |
|---|---|
| `GET /health` | 2xx when ready for lifecycle control. |
| `POST /sleep?level=1\|2` | 2xx only when accepted. |
| `POST /wake_up` | No `tags` means wake all; repeated `tags` selects a non-empty subset. |
| `GET /is_sleeping` | JSON object containing boolean `is_sleeping`. |

The controller verifies sleep and wake post-conditions under one transition deadline.

## Exact disk configuration

Only the canonical vLLM variables are supported:

```text
VLLM_EXACT_DISK_BACKUP_ENABLED
VLLM_EXACT_DISK_BACKUP_DIR
VLLM_EXACT_DISK_BACKUP_CHUNK_BYTES
VLLM_EXACT_DISK_BACKUP_DIRECT_IO
```

The controller never invents or injects a disk location.

## Supported Python versions

The controller CI covers Python 3.11 and 3.12. vLLM, CUDA, PyTorch, GPU architecture, and
model compatibility are governed by the pinned engine rather than the lightweight
controller package.

## Non-guarantees

The published v0.1 suite does not claim compatibility with:

- arbitrary upstream vLLM releases after `v0.22.1`;
- stock vLLM coordinator clients (stock vLLM has no such client);
- multiple active controller replicas;
- out-of-tree worker clients that omit process incarnation or monotonic release counters;
- Windows or macOS process management and `/proc` memory monitoring.
