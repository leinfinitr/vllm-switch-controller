# Compatibility Contract

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

The vLLM adapter reads the existing configuration names outside vLLM. Enable the
installed provider with `VLLM_SLEEP_BACKEND=switch`, then configure:

```text
VLLM_EXACT_DISK_BACKUP_ENABLED
VLLM_EXACT_DISK_BACKUP_DIR
VLLM_EXACT_DISK_BACKUP_CHUNK_BYTES
VLLM_EXACT_DISK_BACKUP_DIRECT_IO
```

The controller never invents or injects a disk location.

## Supported Python versions

The controller CI covers Python 3.11 and 3.12. vLLM, CUDA, PyTorch, GPU architecture, and
model compatibility are governed by the companion engine rather than the lightweight
controller package.

## Non-guarantees

The current implementation does not claim compatibility with:

- arbitrary upstream vLLM checkouts;
- upstream vLLM coordinator clients (upstream vLLM has no such client);
- multiple active controller replicas;
- out-of-tree worker clients that omit process incarnation or monotonic release counters;
- Windows or macOS process management and `/proc` memory monitoring.

## Runtime API and fixed weights

The process-local provider API is version 1, independent of coordinator protocol v1.
An explicitly selected missing provider or mismatched API fails startup. The vLLM adapter
requires CUDA and sleep mode. It rejects EPLB, weight-transfer configuration, and LoRA;
runtime mutation APIs are rejected. L2 may reconstruct the original checkpoint.

Disk manifest schema 2 uses allocation-incarnation region IDs, without content-version
tracking. Bundles belong to one live worker; they are not restart checkpoints and are not
reused across upgrades. Checksums and committed manifests remain mandatory.

See [Runtime](runtime.md) for interfaces, mutable buffers, and supported scope.
