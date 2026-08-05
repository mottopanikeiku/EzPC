# Authenticated two-host deployment boundary

**Date:** 2026-08-04
**Status:** internal/advisor deployment contract; not a concrete-security or publication claim

## Boundary

`scripts/run_two_host_authenticated.sh` is the only launcher that labels a run
`channel=authenticated-ssh`. The FC executable itself accepts the deliberately
non-claiming label `external-loopback-tunnel`; a direct executable invocation
cannot self-certify authentication.

Party 0 runs on the coordinator. Party 1 runs on the pinned SSH peer and connects
to `127.0.0.1:BASE_PORT` and `127.0.0.1:BASE_PORT+1`. A single monitored OpenSSH
master creates two independent remote forwards:

- peer `127.0.0.1:BASE_PORT` to coordinator `127.0.0.1:BASE_PORT` (straight
  SCI stream); and
- peer `127.0.0.1:BASE_PORT+1` to coordinator
  `127.0.0.1:BASE_PORT+1` (reversed SCI stream).

These streams carry the complete SCI `PartyChannel`: IKNP base/extended OT,
Gilboa OLE and Boolean-triple traffic, preflight, public-polynomial exchange,
openings, conversion, publication agreement, and application messages. There is
no application-only MAC or uncovered raw traffic path. The FC source rejects
either stream unless both socket endpoints are IPv4 loopback. Its server socket
still originates in unmodified SCI and wildcard-listens until accept; a
non-loopback connection is rejected before preflight or OT. This leaves a
denial-of-service surface, not an unauthenticated protocol fallback.

OpenSSH uses no user config (`-F /dev/null`), `BatchMode=yes`,
`IdentitiesOnly=yes`, an explicit private identity, an explicit
`UserKnownHostsFile`, `StrictHostKeyChecking=yes`, no global known-hosts file,
no password/keyboard-interactive/hostbased/GSSAPI authentication,
`ExitOnForwardFailure=yes`, loopback-only remote-forward binds, and server-alive
failure detection. Neither party starts until the authenticated master is usable
and both forward requests succeed. The trusted boundary is the two host kernels
and SSH endpoints, rootless Podman, the pinned container image, and the
peer-private executor. Endpoint compromise, malicious-party security, denial of
service, and side channels remain out of scope.

The existing `run_two_party_fc_preprocess.sh` and
`run_two_party_fc_model_scale.sh` use `local-loopback`; they are local-only
evidence, not authenticated deployments.

## Invocation contract

The coordinator supplies every identity, isolation, freshness, port, and public
work value explicitly:

```text
scripts/run_two_host_authenticated.sh \
  --peer USER@HOST --identity ABS --known-hosts ABS \
  --local-executor ABS --remote-executor ABS \
  --container-image NAME@sha256:HEX --container-binary ABS \
  --local-private-root ABS --remote-private-root ABS \
  --local-party-manifest ABS --remote-party-manifest ABS \
  --remote-peer-manifest ABS --local-export-root ABS --remote-export-root ABS \
  --checker-stage ABS --output-dir ABS \
  --local-container-uid N --remote-container-uid N \
  --local-gpu CDI --remote-gpu CDI \
  --session-id N [--invocation-id 32hex] [--ledger-root ABS] --base-port N \
  --qbits 64|128 --bw N --rows N --inner N --cols N \
  --ole-n N --ole-c N --ole-t N --noise regular|uniform [--timeout N] \
  [--fault-injection none|after-stage|commit-rename|post-commit-ack]
```

`--container-binary` is an absolute path inside the immutable container image;
the executor mounts no source/build tree. Both hosts receive the same image
digest and exact ordered public argument array. Only `--party` and party-owned
output differ. Container output is always
`/run/ringlpn/private/output/key`; a host private-root path never enters the
other party's mount. Before OT, the executable exchanges a canonical preflight
containing the nonzero session ID, 128-bit invocation ID, external-tunnel label,
every workload parameter, and all Conv2D shape fields when applicable. The
numeric session ID remains the compatibility/commit handle. The invocation ID
is the global correlation namespace; if omitted, the launcher draws 16 bytes
from OpenSSL and hex-encodes them. Before SSH startup, persistent mode-0700
coordinator ledgers consume both the session and invocation IDs. Failed attempts
remain consumed. `--ledger-root` defaults to the append-only
`results/deployment/correlation-ledger/coordinator-locks` namespace. Both
private containers separately claim the same invocation namespace inside their
non-shared private roots; coordinator locks never enter the party-claim scan.

All paths must be normalized absolute, distinct and non-nested as applicable.
Output, private, export, and checker roots must be fresh. The SSH identity must
have no group/other permission, known-hosts must contain the requested peer, and
the container image must be a SHA-256 digest reference. Missing or invalid input
fails before startup; deployment identity and public parameters have no
environment-variable defaults.

## Transaction and post-exit handoff

Any tunnel, forwarding, preflight, executor, or party failure invokes
`abort-party` for both labeled session containers, terminates/waits attached
processes, removes party private roots (including temporary or unilateral final
records), deletes partial export/checker staging, stops both forwards, preserves
owner-only executor output/status and isolation manifests outside private roots,
and writes `status=FAIL`. It never invokes a checker and never emits a
`COMMITTED` manifest.

On success, both party PIDs must exit zero before either manifest is sealed or
any peer record is read. Sealed manifests are exchanged over the authenticated
SSH master. `stage-party` requires two successful, sealed, exited manifests with
the same session ID before re-owning/exporting either `output/`. The updated
staged peer manifest and party-1 record then cross authenticated SCP. Local and
remote SHA-256 digests must match.

The coordinator builds a fresh mode-0700 checker-stage tree and writes
`COMMITTED.manifest` only after all gates. Its schema is
`ringlpn-two-host-commit-v1`; it binds state `COMMITTED`, session ID, channel,
both ports, common-public-parameter digest, zero exit codes, relative record
paths and digests, relative isolation-manifest paths and digests, and commit
time. The file is mode 0600 and is published by temporary-file fsync, rename,
directory fsync, atomic checker-stage rename, and parent-directory fsync. The
checker executor rejects absent/malformed commits, digest mismatch, or manifests
that are not successful, sealed, staged, and session-matched. Thus the source's
in-process bilateral rename/ack is only best-effort; the durable coordinator
commit is the consumer-visible transaction boundary.

`--fault-injection` supplies deterministic negative controls at post-stage,
commit rename, and post-commit acknowledgement. Every injected fault follows the
same bilateral abort/record cleanup path and leaves no consumer-visible
`COMMITTED.manifest`.

Each attempt generates `authenticated-boundary.manifest` and
`authenticated-boundary.csv` under `--output-dir`. They record the boundary,
peer, hashes of known-hosts and identity files, pinned image, session,
invocation ID, coordinator-ledger digest, both ports, common-public-parameter
digest, isolation-manifest paths, return codes,
record digests, fault-control point, status, and `security_claim=none`.
