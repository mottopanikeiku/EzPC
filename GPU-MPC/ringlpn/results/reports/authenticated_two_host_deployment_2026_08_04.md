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

## Source-only local runtime candidate (September 22 supplement)

`scripts/build_source_runtime_candidate.py` supplies a separate **local image
candidate**, not an authorized registry runtime or publication release. It does
not modify the publication manifest, execute the coordinator, create an approval
object, tag source, push an image, or fill any release-authorization null.
Implementation is supplied here; build/planner/GPU success is not asserted by
this supplement.

The only application-source input is public commit
`afc93c6fb94af01239f51b06d5c29a40c6fe7d84` from
`https://github.com/mottopanikeiku/EzPC.git`. The driver creates a new checkout,
checks exact HEAD/tree and clean index/worktree state, and initializes only the
FC-required CUTLASS gitlink at its parent-pinned commit. The other FC dependencies
(SCI/src, Sytorch, cryptoTools, LLAMA, bitpack) are tracked parent-tree source.
Unused source submodules and the dataset/weight submodules are not initialized.
The build context is reconstructed from Git archive entries: regular source
files in the required include/source roots, the canonical FC build scripts, and
upstream license/notice files. Results, evidence, manuscripts, measurements,
datasets, weights, Git metadata, approval objects and arbitrary caller inputs
are excluded. The three locally supplied packaging recipes are separately
SHA-256 identified; they are not represented as files from the pinned public
commit.

`Dockerfile.runtime` pins the same CUDA 12.6.3-devel Ubuntu 24.04 base digest
as `Dockerfile.reproduction`:
`sha256:badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade`.
The mutable CUDA apt repository is disabled. Ubuntu packages use only
`https://snapshot.ubuntu.com/ubuntu/20260810T000000Z/`, with the reproduction
recipe's exact GCC-13, OpenSSL and other direct package versions. The complete
installed version/architecture inventory is retained. Transitive packages are
resolved from this dated snapshot, not silently from today's archive.
Compilation enters the existing `build_two_party_fc_preprocess.sh` →
`build_component.sh linear-fc` route, including its fixed canonical symlink,
compiler flags and object cleanup. No alternate compiler command is introduced.
The image retains the admitted source/licenses, recipe, compiler environment,
real linked producer/checker ELF and its shared libraries. A missing `ldd`
dependency fails the build. NVIDIA host-driver injection is still required for
GPU execution; driver libraries are not copied from the build host.

Required host inputs: Python 3.11 or newer, Git, Docker CLI with BuildKit, a local
Docker Unix socket accessible to the invoking user, network access to the public
Git repositories/CUDA registry/Ubuntu snapshot, and enough local build/export
storage. No GPU is accessed by the packaging driver. No credentials, SSH agent,
registry login, proxy environment, Podman socket, source override or prebuilt
binary is accepted. Docker uses a fresh empty client configuration. Work/output
roots must be absent, disjoint, normalized absolute paths with existing
non-symlink parents, and outside the repository. They remain consumed even after
failure; choose new roots rather than retrying over them.

From `GPU-MPC`, with the two example paths not already present:

```sh
python3 ringlpn/scripts/build_source_runtime_candidate.py \
  --work-root /tmp/ringlpn-runtime-work-afc93c6-001 \
  --output-root /tmp/ringlpn-runtime-artifacts-afc93c6-001 \
  --docker-host unix:///var/run/docker.sock
```

For rootless Docker, supply that daemon's actual local Unix socket instead.
This is not a rootless-Podman deployment test. No socket enters the build context
or image. Child command failures propagate, interrupted child process groups are
terminated, exact temporary containers are removed, and fresh scratch is deleted
while the work-root tombstone and output logs remain. A failure does not retain
a success `candidate.json` or portable archive. A successfully built image may
remain in the local daemon by its immutable image ID.

Output contracts:

- `source-admission.json`, schema `ringlpn-source-runtime-admission-v1`: public
  URL/commit/tree, required gitlink URL/commit, all admitted source paths/modes/
  SHA-256s, packaging recipe hashes, base digest and package snapshot.
- `provenance/build.json`, schema `ringlpn-source-runtime-build-v1`: exact
  canonical builder, compiler/tool paths/resolved paths/hashes/version output,
  source admission digest, absolute ELF path/SHA-256, architecture and resolved
  shared-library paths/SHA-256s. Adjacent artifacts retain package inventory,
  linker map, dependency files, NUL-delimited build commands/environment,
  `ldd` and ELF dynamic-section output, with hashes bound by `build.json`.
- `candidate.json`, schema `ringlpn-source-runtime-candidate-v1`: source/build
  provenance digests, ELF identity, immutable **local image/config ID**,
  platform/RootFS identity, host Git/Docker versions, and portable archive
  size/SHA-256. `registry_manifest_digest` and `authorized_reference` are null;
  `publication_authorized`, `gpu_execution_performed`, and
  `runtime_planner_executed` are false.
- `runtime-candidate.docker.tar`: portable `docker image save` artifact. The
  driver verifies its single image's config SHA-256 equals the local image ID.
  `image-config.json` preserves those exact config bytes. The extracted
  `test_two_party_fc_preprocess` SHA-256 must equal the in-image build record.
  `commands.json` and numbered logs record host commands and return codes.

The in-image executable is:
`/opt/ringlpn/source/GPU-MPC/ringlpn/bin/test_two_party_fc_preprocess`.
It is the real FC producer and unchanged stock-Orca checker, not a planner-only
stub. Its planner can be exercised without a GPU after a successful build:

```sh
OUT=/tmp/ringlpn-runtime-artifacts-afc93c6-001
IMAGE_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"]["local_id"])' "$OUT/candidate.json")
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev "$IMAGE_ID" \
  --plan --qbits 128 --bw 32 --rows 2 --inner 3 --cols 2
```

Portable import uses `docker image load --input "$OUT/runtime-candidate.docker.tar"`;
compare the loaded ID/config and binary hashes to `candidate.json` before use.
Native producer/checker execution must separately provide explicitly selected
allowed physical GPUs, party-private authenticated channel files, distinct
persistent consume-once ledgers, fresh private output mounts writable by UID
65532, and the producer's public shape/invocation arguments. It is not run by this
packaging command. Do not mount a repository, Docker/Podman socket, secret
credential directory or internal evidence tree into the candidate.

This recipe pins rebuild inputs and reports measured artifact identities; it
does not promise byte-identical Docker-save tars or image IDs across daemon/
BuildKit versions and timestamps. A local config digest is **not** a registry
manifest digest. Registry upload, a resulting registry manifest identity,
authorized owner/reference, two-host admission and native execution evidence
remain separate requirements; the existing coordinator rightly does not accept
this local-candidate record as satisfying them.

### Executed local candidate, not authorized deployment

Main built the recipe from a fresh public checkout on September 22. The
first attempt exposed a missing extensionless `cryptoTools/gsl/span` header
in source admission; the corrected policy admits the tracked GSL header
directory, not arbitrary extensionless data. The next fresh build succeeded:

- Source: `afc93c6fb94af01239f51b06d5c29a40c6fe7d84`; 1,071 admitted source/license files.
- Local image/config ID:
  `sha256:632f3d1e16225968a91c49816e97c234078158e5a57aef756b617bac49f683c8`.
- FC ELF SHA-256:
  `a5630582ebcd27b5c347c1a1ea12172268a884d0fe53c15a40bdba23de0f2283`,
  byte-identical to the existing reference binary.
- Portable archive: `/tmp/ringlpn-runtime-artifacts-afc93c6-002/runtime-candidate.docker.tar`,
  7,754,951,680 bytes, SHA-256
  `3d6d288e555281ab2ef0d11069b9137dc9b05f66509b0a236dd9ea15841baf29`.

The in-image `100x64x10`, q128/bw32 planner passed with the container
read-only, network disabled and default UID/GID 65532. A fresh CPU-only
direct-OT producer pair then generated q128/bw32 `2x3x2` records and
independent mask states. The **in-image unchanged checker** passed all
stock-consumer/dealer-equivalence checks on physical GPU3 only, using
the invoking nonroot UID/GID and a read-only private fixture mount.
All private fixture files and the temporary container were removed.
Reusing the consumed packaging roots rejected before a build.

Packaging metadata remains the immutable build-stage observation; later
planner/GPU-checker evidence is recorded separately in
`autonomous_technical_closure_2026_09_22.json`. No runtime producer pair was
run across hosts, no registry manifest was published, and the publication
manifest's authorization/runtime fields remain unfilled. Docker here is an
ephemeral local build/smoke vehicle, not a bypass of the Podman or host-policy
requirements above.
