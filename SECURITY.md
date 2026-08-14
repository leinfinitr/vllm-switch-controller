# Security Policy

## Project Status

The vllm-switch-controller is an experimental research system. It has not received
a production security audit and does not provide authentication, authorization, rate
limiting, tenant isolation, or TLS termination.

Security fixes are provided on a best-effort basis for the default branch.

## Reporting a Vulnerability

Do not disclose a suspected vulnerability in a public issue, discussion, benchmark log,
or pull request.

Use the repository host's private vulnerability-reporting or security-advisory feature
when available. Otherwise contact the maintainers privately through the project hosting
account before publishing details. Include:

- the affected commit or version;
- deployment assumptions and reachable endpoints;
- reproducible steps or a minimal proof of concept;
- impact and any known mitigations;
- whether the issue has been disclosed elsewhere.

Maintainers should acknowledge the report privately, validate impact, coordinate a fix,
and agree on disclosure timing before a public advisory. Response times are best effort
because this is a research project without a formal security response SLA.

## Deployment Guidance

- Bind the controller to `127.0.0.1` or a private management network.
- Place authentication, authorization, request limits, and TLS at a trusted reverse proxy.
- Restrict direct access to controller `/admin/*` and backend lifecycle endpoints.
- Do not expose vLLM development management endpoints to untrusted clients.
- Keep backend URLs and coordinator traffic on trusted interfaces.
- Inject secrets outside committed YAML files and avoid logging authorization headers.
- Run one controller process per managed pool. The in-memory lock and state are not safe
  across active controller replicas.
- Treat configuration write access as administrative code-execution access because
  `launch_command`, `env`, and `cwd` control child processes.
- Isolate model processes and apply host-level resource limits appropriate to the
  environment.

The controller forwards inference request bodies and most end-to-end headers to the
selected backend. Operators are responsible for input policy, model access control,
content handling, data retention, and backend security.

## Known Security Limitations

- Administrative and OpenAI-compatible endpoints share one unauthenticated listener.
- `/health` and `/admin/state` expose model aliases, lifecycle state, and request counts.
- CPU backup stats expose worker metadata, process IDs, and memory accounting.
- The controller stores lifecycle state in memory and has no durable audit log.
- Worker registration has no identity proof, lease, or heartbeat.
- There is no distributed coordination for multiple controller replicas.
- Example launch configuration can start arbitrary local commands by design.

These limitations are architectural, not vulnerability-reporting substitutes. Report any
unexpected path that crosses the documented trust boundary or weakens the fail-closed
lifecycle and exactly-once reservation guarantees.
