# Authenticated two-host deployment boundary

**Date:** 2026-08-04
**Status:** internal/advisor deployment contract; not a concrete-security or publication claim
**Launcher binding:** updated 2026-09-11 for explicit no-agent SSH, checker ownership handback, physical host/GPU binding, and conservative block-backed ledger admission. Genuine distinct-host execution remains unperformed.

## Boundary

`scripts/run_two_host_authenticated.sh` is the only launcher that labels a run
`channel=authenticated-ssh-plus-mutual-hmac-sha256-v1`. The FC executable
accepts the deliberately non-claiming source label `external-loopback-tunnel`;
the launcher provisions an ephemeral per-run key to mutually authenticate both
party processes before public preflight. A direct executable invocation cannot
self-certify either SSH or process authentication.

Party 0 runs on the coordinator. Party 1 runs on the pinned SSH peer and connects
to `127.0.0.1:BASE_PORT` and `127.0.0.1:BASE_PORT+1`. A single monitored OpenSSH
master creates two independent remote forwards:

- peer `127.0.0.1:BASE_PORT` to coordinator `127.0.0.1:BASE_PORT` (straight
  SCI stream); and
- peer `127.0.0.1:BASE_PORT+1` to coordinator
  `127.0.0.1:BASE_PORT+1` (reversed SCI stream).

These streams carry the complete SCI `PartyChannel`: the mutual HMAC
authentication exchange, IKNP base/extended OT, Gilboa OLE and Boolean-triple
traffic, preflight, jointly exchanged public-vector seeds, openings, conversion,
publication agreement, and application messages. The OpenSSH transport is
restricted to ChaCha20-Poly1305 or AES-GCM and rekeys after at most 1 GiB or
one hour, so all post-handshake bytes receive tunnel confidentiality and
integrity against a network attacker. There is no application-only MAC or
uncovered raw inter-host traffic path. The FC source rejects
either stream unless both socket endpoints are IPv4 loopback. Its server socket
still originates in unmodified SCI and wildcard-listens until accept; a
non-loopback connection is rejected before preflight or OT. This leaves a
denial-of-service surface, not an unauthenticated protocol fallback.

OpenSSH uses no user config (`-F /dev/null`), `BatchMode=yes`,
`IdentitiesOnly=yes`, `IdentityAgent=none`, an explicit private identity, an explicit
`UserKnownHostsFile`, `StrictHostKeyChecking=yes`, no global known-hosts file,
no password/keyboard-interactive/hostbased/GSSAPI authentication, no agent or
X11 forwarding, no proxy command, no compression, AEAD-only ciphers, pinned
modern KEX/host-key algorithm sets, a 1-GiB/one-hour rekey bound,
`ExitOnForwardFailure=yes`, loopback-only remote-forward binds, and server-alive
failure detection. Neither party starts until the authenticated master is
usable and both forward requests succeed. The trusted boundary is the two host
kernels and SSH endpoints, rootless Podman, the pinned container image, and the
peer-private executor. Endpoint compromise, malicious-party security, denial
of service, and side channels remain out of scope.

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
  --container-image NAME@sha256:HEX --container-binary ABS --container-binary-sha256 64hex \
  --local-private-root ABS --remote-private-root ABS \
  --local-party-ledger-root ABS --remote-party-ledger-root ABS \
  --local-party-manifest ABS --remote-party-manifest ABS \
  --remote-peer-manifest ABS --local-export-root ABS --remote-export-root ABS \
  --checker-stage ABS --output-dir ABS \
  --local-container-uid N --remote-container-uid N --checker-container-uid N \
  --local-gpu CDI --remote-gpu CDI --checker-gpu CDI \
  --session-id N [--invocation-id 32hex] --ledger-root ABS --base-port N \
  --qbits 64|128 --bw N --rows N --inner N --cols N \
  --ole-n N --ole-c N --ole-t N --noise regular|uniform [--timeout N] \
  [--fault-injection none|after-stage|prepare-rename|after-checker|cleanup-local-party|cleanup-remote-party|cleanup-checker|deletion-receipt|final-commit]
```

`--container-binary` is an absolute path inside the immutable container image;
the executor mounts no source/build tree. Before either party starts, each host
probes the actual repository digest, image ID, and in-image binary SHA-256 in a
read-only, networkless container; image IDs must match. The remote executor must
be byte-identical to the clean coordinator executor. Both hosts receive the
same image digest and exact ordered public argument array. Only `--party` and
party-owned
output differ. Container output is always
`/run/ringlpn/private/output/key`; a host private-root path never enters the
other party's mount. Before OT, the executable exchanges a canonical preflight
containing the nonzero session ID, 128-bit invocation ID, external-tunnel label,
every workload parameter, and all Conv2D shape fields when applicable. The
numeric session ID remains the compatibility/commit handle. The invocation ID
is the global correlation namespace; if omitted, the launcher draws 16 bytes
from OpenSSL and hex-encodes them. Before SSH startup, persistent mode-0700
coordinator ledgers consume both the session and invocation IDs. Failed attempts
remain consumed. `--ledger-root` is mandatory and must name a pre-existing,
owner-only persistent read-write directory outside private, export, checker, and
evidence roots. Session and invocation claims occupy separate namespaces under
that root. Both private containers separately claim the same invocation
namespace inside their non-shared private roots; coordinator locks never enter
the party-claim scan.

The coordinator ledger and both party ledgers must each be a pre-existing
owner-only read-write **mount boundary on inspectable block-backed ext4 or xfs**.
`peer_private_execution.py ledger-storage --ledger-root ABS` is the shared
admission policy: it binds filesystem UUID, mount source/target, and device
identity, traverses partition parents and device-mapper/MD backing devices,
and rejects unknown, virtual-only, loop, RAM, and tmpfs backing. A lexical path
outside `/tmp`, owner/mode checks, and a successful `fsync` alone never prove
persistence. Btrfs, ZFS, network filesystems, and other storage are conservatively
unsupported until reviewed, not silently treated as unsafe-equivalent ext4.
This policy is **not durability attestation**. The operator still guarantees
that devices honor persistence barriers and that ledger state is never rolled
back, deleted, cloned, or hosted on hidden volatile backing. Storage identity
is captured before each claim, remeasured during cleanup/finalization, and
bound into all three deletion-receipt ledger entries; consumed claims survive
failures. A future hardware/runtime admission broadening requires review.

All paths must be normalized absolute, distinct and non-nested as applicable.
Output, private, export, and checker roots must be fresh. The ledger root must
already exist, be writable by and owned by the coordinator, and deny all
group/other access. The SSH identity must have no group/other permission,
known-hosts must contain the requested peer, and the container image must be a
SHA-256 digest reference. Missing or invalid input fails before startup;
deployment identity and public parameters have no environment-variable
defaults.

## Transaction and post-exit handoff

Any tunnel, forwarding, preflight, executor, party, checker, cleanup, receipt,
or final-commit failure invokes bilateral abort and removes ephemeral channel
keys, private/export roots, checker records, duplicate outputs, and party/checker
containers. The coordinator and both party consume-once ledgers remain durable,
owner-only, and consumed. Failure removes any partial `COMMITTED.manifest`,
writes `status=FAIL`, and cannot leave a consumer-visible PASS result.

On success, both party PIDs must exit zero before either manifest is sealed or
any peer record is read. Sealed manifests are exchanged over the authenticated
SSH master. `stage-party` requires two successful, sealed, exited manifests with
the same session ID before re-owning/exporting either `output/`. The updated
staged peer manifest and party-1 record then cross authenticated SCP. Local and
remote SHA-256 digests must match.

The coordinator first atomically publishes
`checker-stage/PREPARED.manifest` with schema
`ringlpn-two-host-prepare-v1`. It binds the authenticated channel, ports,
public parameters, pinned runtime and executor identities, host and process
identities, zero party exit codes, and the party record/isolation manifests.
Party 0 and the checker share the coordinator host; party 1 is remote.
The three GPU bindings combine stable host identity, requested CDI selector,
GPU UUID, and normalized PCI bus identity. Equal ordinal strings on different
hosts are valid; equal UUID or PCI identities on the same host reject even
through different aliases. These checks bind observations, not a proof that
two machine-id strings necessarily represent physically distinct hosts.
The distinct-UID, distinct-physical-GPU, networkless checker must emit its
bound successful isolation manifest.

Checker mounts use rootless `U=true` ownership changes. After the exact
label-bound container has stopped, the executor restores both input-stage
trees and output tree to the coordinator via `podman unshare`, hardens their
private modes, and verifies ownership **before** coordinator manifest reads
or finalization. Abort stops/removes the exact labeled container first and
reclaims all surviving stage/output roots, including partial-create failure;
ordinary stage deletion must not be attempted while a subordinate UID owns
an unreadable tree.

September 11 controls reproduce the old ownership failure and the corrected
handback using real Linux UID/mode transitions in a networkless container
with a narrow simulated Podman CLI. An actual tmpfs mount that met the old
lexical/owner/mount/fsync checks is rejected by the new policy. Native OpenSSH
client/server/agent controls show that matching ambient-agent authentication
previously succeeded despite `IdentitiesOnly=yes`; `IdentityAgent=none`
rejects it while an explicit unencrypted identity still succeeds. Synthetic
host/GPU boundary regressions exercise remote ordinal reuse and local aliases.
These are focused controls, not a rootless-Podman success, a positive durable
storage test, or an authenticated two-host benchmark. This host lacks native
Podman/user-mapping helpers and an authorized remote identity/host.

After checker success, the launcher deletes both parties' private/export roots,
purges checker records and duplicate outputs while retaining only the checker
log, and emits `deletion-receipt.json` with schema
`ringlpn-two-host-deletion-receipt-v1`. Only then may it atomically publish
`checker-stage/COMMITTED.manifest`. Its schema is
`ringlpn-two-host-final-commit-v2`; it binds the session and invocation IDs,
public-parameter digest, and path/SHA-256 pairs for the prepared manifest,
checker manifest, and deletion receipt. The final
`launcher-result.json` uses `ringlpn-authenticated-launch-result-v2`, records
the cleanup invariants, and digest-binds all four artifacts. The durable
coordinator commit is the consumer-visible transaction boundary.

`--fault-injection` supplies the exact deterministic controls `after-stage`,
`prepare-rename`, `after-checker`, `cleanup-local-party`,
`cleanup-remote-party`, `cleanup-checker`, `deletion-receipt`, and
`final-commit` (or `none`). Every injected failure follows the same fail-closed
cleanup boundary, retains the owner-only consume-once ledgers, and leaves no
consumer-visible `COMMITTED.manifest` or PASS launcher result.

Each attempt generates `authenticated-boundary.manifest` and
`authenticated-boundary.csv` under `--output-dir`. They record the boundary,
peer, hashes of known-hosts and the SSH public identity, authorized and measured
container image/binary identities, local/remote executor hashes, distinct
machine identities, session and invocation IDs, coordinator-ledger digest, both
ports, common-public-parameter digest, isolation-manifest paths, return codes,
record digests, fault-control point, status, and `security_claim=none`.
