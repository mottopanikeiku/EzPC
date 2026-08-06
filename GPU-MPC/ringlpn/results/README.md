# ringlpn results — directory index

Reorganized 2026-06-10. Every run script writes into its artifact directory
below; nothing writes to this top level anymore.

## Current checkpoint (2026-08-06)

The live forward-FC/Conv artifact composes party-local SPFSS, distributed DPF,
SCI/IKNP or opt-in EMP-Silent OT, epoch-zero Gilboa OLE, consume-once
Ring-OLE-output Phase-C bootstrap, GPU Ring-LPN expansion, and exact conversion.
Before OT setup or private-DRBG construction, each party consumes the same
public high-entropy 128-bit invocation and exact fixed-width correlation plan in
its owner-only persistent ledger. Every Ring-OLE/DPF/OT/conversion/use is
separated by layer/kind/direction/limb/ring-batch/tree/phase/primitive/
conversion/output coordinates; version-3 records and preflights bind the
invocation and claim digest. Duplicate/retry/restart, compatibility-ID collision,
truncated/corrupt ledger, tail reuse, and nonpositive bootstrap capacity reject
before publication. Consumed state never rolls back.
The corrected source fixes unsent identity `a0=1`, exchanges/counts exact full-
field shares only for the `(c-1)*n` tail coefficients, reduces DPF correction
words/leaves with one GPU block per tree, reserves `3*c^2*t^2` slots of each
Ring-OLE output for the next DPF Phase C, and exposes only the remainder to the
application. Each party remains a separate process reading only its own private
state. The local loopback runner does not enforce OS-level peer-file isolation;
the authenticated coordinator's sealed digest-bound `COMMITTED.manifest`, not
either raw record, is the consumer gate.

Recorded current live evidence:

- `fc/two_party_fc_preprocess_2026_08_04.csv` — q64/q128,
  regular/uniform, small and q64 multi-batch rows; all six public/key-order/
  current-transcript/bootstrap-pool/unchanged-online contracts pass.
- `fc/two_party_fc_preprocess_controls_2026_08_04.csv` — eleven focused
  consume-once, restart, tail-reuse, invocation-collision, ledger-truncation,
  preflight, stale-output, rename-failure, corrupt-record, swapped-record, and
  nonpositive-capacity controls; every expected rejection passes.
- The `fc/two_party_fc_model_scale_2026_08_04.*` v4 artifact family
  (regenerated 2026-08-06) covers the exact ResNet18 classifier-layer shape
  `1x512x1000`, q128/bw32, `n=8192,c=2,t=8`, one warmup plus ten measured
  trials, 10/10 pass. Median preprocessing is 8.942 s; application traffic is
  159,469,294 bytes; matched stock trusted-dealer keygen is 10.706 ms median;
  unchanged two-share online is 1.097 ms median; final Orca payload is
  4,108,096 bytes per party.
- The aggregate records 11,298 protocol dependency layers, 179,636,224 peak
  host bytes, 27,241,086,976 peak GPU bytes, and 160,618,542 total transport
  bytes. Median Phase B is the largest stage at 3.366 s (37.6%); Phase C is
  0.182 s (2.0%). The exact 276-instance plan accounts per party for 1,536
  epoch-zero and 210,432 PCG-supplied Phase-C products, 210,432 consumed plus
  1,536 terminal-discarded reserved slots, and 1,024 unused application slots.
  This is not a full ResNet18 inference or scale-10 truncation run.
- The runners generate fresh invocation IDs, require a private ledger root,
  and emit invocation/claim-digest columns in current schemas/manifests. Raw
  party key records remain validation inputs, not public evidence.

Proof/evidence boundary:

- The v2.7 TeX source and current security contract contain the canonical
  correlation functionality, persistent consume-once ledger, exact
  correction-word coupling, role-specific correlated-batch simulators, the
  masked-difference bootstrap lemma and noncircular epoch induction, conversion
  simulator, source map, and conditional forward theorem. PDF build/inspection
  status is recorded after the current rebuild.
- `P-FRESH` is source/proof closed only under SHA-256 collision resistance and
  a trusted private persistent filesystem providing one deployment-wide ledger
  namespace, exclusive create, fsync, atomic rename, directory fsync, and no
  adversarial storage cloning/rollback. Its eleven focused controls pass.
- Renewed model-assisted source/proof reviews are current, but they are not
  independent human cryptographic review.
- The exact regular-projection and cancellation law is machine-checkable.
  No reviewed reduction yet maps the structured projected code to a concrete
  advantage bound, and no concrete Ring-LPN parameter is pinned. q64/q128 name
  one/two approximately 62-bit arithmetic limbs, not security levels.
- Current measurements use unauthenticated local loopback. Application bytes
  exclude backend setup and TCP/IP overhead; total transport includes the
  selected backend's recorded setup, but its base-OT subcost is unavailable.
  The selected EMP-Silent revision is pinned but independently unreviewed.
  Results establish executable correctness and cost at feasibility parameters,
  not 128-bit, malicious, WAN, training-layer, full-model, or full-dealerless-
  Orca claims.
- The matched comparison is negative: median live preprocessing is about 831
  times stock dealer keygen. GPU batching, tree-block reduction, executable
  self-bootstrap, and memory/dependency-layer instrumentation are complete.
  Phase-A/B/DMPF optimization, independent silent-backend review, authenticated
  two-host evaluation, every real-model linear layer plus truncation/state
  handoff, a compatible dealerless baseline, clean-clone reproduction, and
  human review remain publication gates.

The source-pinned closest-baseline audit now ranks newly public Reverse Cuckoo /
libOTe first. It supersedes the 2026-07-29 “no public code” statement without
removing that historical record. The pinned stock run is measured but is not
an exact project baseline: process wall was 12.43 s, peak RSS 22,939,444 KiB,
libOTe printed 11 s internally, and local synthetic `setBase` took 446.448 ms;
live `genBaseCors` was excluded. It uses a different field and folded layout,
samples factors internally, and runs on CPU. A separate exact caller-factor
`p0` adapter has now exercised live setup and full-domain differential controls
for the explicitly labelled native 16-folded layout. Raw 31-diagonal timing and
GPU evaluation remain unmeasured and non-comparable; no speedup crosses those
boundaries.

The complete canonical component gate is
`RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 CUDA_VISIBLE_DEVICES=<free-gpu>
PATH=/usr/local/cuda/bin:$PATH ./scripts/run_paper_checkpoint_smoke.sh`.
All source-changing work requires a fresh zero-exit run before checkpointing.
Alp `<fcetin@hawk.iit.edu>` remains the sole paper author by user direction;
inherited code/protocols remain cited, and unresolved ownership/reuse decisions
must not be treated as settled.

Clean-clone reproduction is pinned by
`../scripts/publication_environment_manifest_2026_08_04.json` and
`../scripts/Dockerfile.reproduction`; the fail-closed entry point is
`../scripts/reproduce_publication.sh`. The immutable JSON records the CUDA
image digest, toolchain, `sm_89` flags, recursive external revisions/licenses,
and the checksums of the deliberately excluded dataset/weight snapshots.
Those data are not inputs to the current shape-only gates. Every invocation
emits a redacted runtime JSON manifest. SSH identities, pinned `known_hosts`,
private party records, and private data stay in read-only runtime mounts and
are never copied into the image or either manifest.

## Where to look

| Directory | Contents | Produced by |
|---|---|---|
| `reports/` | **Start here.** Current plans, proposals, baselines, memos, handoffs | hand-written |
| `ntt/` | NTT/PolyMul sweeps: CPU (NFLlib), GPU cheddar q32/q64/q128, legacy | `run_sweep.sh`, `run_cuda_sweep.sh`, `run_cuda_sweep_legacy.sh`, `run_cuda_single.sh` |
| `ole/` | Figure 2 Ring-LPN OLE q64/q128 × uniform/regular component rows; includes independently sampled two-process SPFSS keygen/provenance followed by the existing both-record OLE checker. The current focused row is `ole_two_party_{keygen,keys}_q64_regular_c2_t8_n8192.csv`; neither it nor the `t=8`/`t=64` feasibility rows are security-pinned or a live two-process FC run. | `run_ole_sweep.sh`, `run_ole_two_party_keys.sh` |
| `linear_ole/` | Ring-matrix OLE-to-Beaver (2x2x2, n=8192): q64/q128 × uniform/regular | `run_linear_ole_sweep.sh` |
| `vole/` | Standalone VOLE expansion prototype | `run_vole_sweep.sh` |
| `orca_fc/` | Orca FC artifacts: keywriter demo, ideal-OLE transcript, **real-OLE slot-packed transcript**, Zp bridge | `run_orca_fc_ringlpn_demo.sh`, `run_orca_fc_ideal_ole_transcript.sh`, `run_orca_fc_real_ole_transcript.sh`, `run_orca_zp_bridge_test.sh` |
| `fc/` | **Live two-process forward-FC evidence:** six q64/q128 regular/uniform/multi-batch rows and controls; exact ResNet18-classifier 1-warmup/10-trial CSV, aggregate summary, environment/source/binary digest manifest, matched dealer timer, and unchanged-online timer. Feasibility parameters and local loopback only. | `run_two_party_fc_preprocess.sh`, `run_two_party_fc_model_scale.sh` |
| `secure_convert/` | Two-process evidence for exact `Z_M -> Z_2^bw` conversion using SCI/IKNP-generated edaBits/daBits/Boolean triples; common preflight, bounded bilateral best-effort outputs, corruption controls, and separate transcript counters. The live forward-FC path consumes this API. The wrap bit is never opened; the current security contract gives the hybrid simulator. Transport remains unauthenticated loopback and ripple-depth. |
| `dpf/` | Distributed DPF keygen artifacts: ideal-functionality protocol logic; two-process SCI/IKNP+Gilboa transport with measured bytes/direction switches; full-width four-call GPU AES parity with enforced seed-bit-0 sensitivity; strictly validated GPU-evaluated party keys; offline correctness/corruption/invalid-input controls. Direction switches are not network rounds; security reductions remain open. | `run_distributed_dpf_keygen.sh`, `run_two_party_dpf_keygen.sh`, `run_two_party_gpu_dpf.sh` |
| `profiling/` | VTune hotspot/memory captures | `run_vtune_*.sh` |
| `outreach/` | Abstracts, posters, professor memos/status emails | hand-written |
| `archive/` | Superseded one-offs: early spot checks, `*_regular_patch`, `*_after_linear`, old plan drafts | frozen |
| `security/` | **Start with `security/README.md`.** Current evidence includes `s2_regular_projection_exact_2026_08_04.csv` (1,160 exact-law records), corrected `s2_regular_projection_estimator_sensitivity_2026_08_04.csv` (575 guarded model rows, SHA-256 prefix `ffd335a7...`), `regular_isd_crypto2024_2026_08_04.csv` (50 source-pinned direct-formula/incompatibility rows, SHA-256 prefix `68b8329d...`), and `hybrid_regular_sd_asiacrypt2025_2026_08_04.csv` (20 source-pinned direct-RSD formula/orbit-sensitivity rows, SHA-256 prefix `9a442eec...`). Raw attack-formula costs preserve separate heuristic orbit sensitivities; generic BJMM and projected non-RSD incompatibilities remain explicit. None is an executable attack or a pin. Historical conservative-pin files and the former `c1b9cb53...` estimator artifact are rejected for parameter selection/security claims. | `audit_ringlpn_regular_projection.py`, `audit_regular_isd_crypto2024.py`, `audit_hybrid_rsd_asiacrypt2025.py`, `audit_ringlpn_projection_security.py`, `audit_ringlpn_finite_field_models.py` |
| `pcg/` | Adapted rows from the licensed native-`Z_(2^bw)`/Galois-ring PCG artifact, with patch digest and correctness gate; not a reproduction of the released benchmark | `run_native_ring_pcg_baseline.sh` |
| external evidence directory | Runtime manifest, publication/PDF hashes, and byte-retained copies of otherwise ignored CSV/log/text evidence. This directory must be mounted outside the clone; credentials and private key/noise/record files are excluded. | `reproduce_publication.sh` |

## Reports, newest first

| File | What it is |
|---|---|
| `reports/reverse_cuckoo_p0_baseline_2026_08_04.json` | **COMPLETE EXACT-`p0` NATIVE-FOLDED DISTRIBUTED ROW:** pinned libOTe adapter with caller factors, canonical 62-bit context, live `genBaseCors`, collision accumulation, 16,777,216-position differential check, duplicate and corruption controls. Setup 18,832,990 us / 52,791,184 bytes; online full-domain 2,070,844 us / 1,425,584 bytes; end-to-end including validation 20,948,042 us / 54,216,768 protocol bytes. Native 16-folded CPU layout only—not raw 31-diagonal or GPU timing; speedup/security claims are null. |
| `reports/libote_reverse_cuckoo_stock_baseline_2026_08_04.md` | **MEASURED CLOSEST STOCK DISTRIBUTED BASELINE:** pinned clean libOTe build and corrected `-bench` dispatch at `(2^20,4,16)`; 12.43-s process wall, 22,939,444-KiB peak RSS, 11-s internal total, and 446.448-ms synthetic `setBase`. CPU/local-process/Goldilocks/internal-factor/native-16-folded evidence with live `genBaseCors` excluded—not exact `p0`, raw 31-diagonal, GPU, two-host, or live-setup-inclusive evidence, and not a speedup row. |
| `reports/structured_attack_audit_2026_08_04.md` | **CURRENT STRUCTURED ATTACK AUDIT (internal/advisor; no pin):** exact live iid-uniform-`F_p^*` regular distribution, direct RSD versus projected occupancy/cancellation boundary, source-pinned 2024 regular-ISD and 2025 hybrid-RSD plus 2025/2026 QA/SSD and 2026 sparse-problem dispositions, formal negacyclic/cyclic orbit and stabilizer bound, corrected `d`-element orbit diagnostic, resource/data/success semantics, and explicit reduction/review blockers. Generic-estimator rows and orbit existence are not reviewed concrete Ring-LPN security. |
| `reports/closest_dmpf_baseline_audit_2026_08_04.md` | **CURRENT CLOSEST DMPF BASELINE AUDIT (internal/advisor):** Reverse Cuckoo/libOTe is the newly public rank-1 distributed candidate, not zero-change exact/GPU/setup-inclusive evidence. Complete pinned/license matrix, exact 31-diagonal adaptation, collision normalization, stock and exact-control commands, author-contact gates, and mandatory noncomparability rules. Supersedes the 2026-07-29 no-code statement without deleting history. |
| `reports/native_ring_technology_audit_2026_08_04.md` | **INTERNAL/ADVISOR NO-GO:** source-pinned native-ring QA-SD PCG audit covering arithmetic defects, 2025/2026 attacks, centralized/non-matrix/non-Orca boundaries, SPDZ2k semantic mismatch, and a strictly toy-only future correctness oracle. Not a fallback for either publication track. |
| `reports/two_party_dpf_transport_memo_2026_07_29.md` | **Two-PROCESS keygen on a real transport**: SCI IKNP over TCP, Gilboa `Z_p` OLE, OpenSSL-private-DRBG roots, 369/369 host-reference pairs, and 88 GPU-evaluated full-width-AES pairs. The unchanged host-reference evaluator remains splitmix64 correctness-only; see §3.4 for the deployed four-call full-width GPU AES evidence and the remaining reduction/silent-OT/network-round boundaries. |
| `reports/dealerless_ole_two_party_keys_memo_2026_07_29.md` | **M2 CORE GATE**: independently sampled per-party noise and SPFSS keys generated by two OS processes over real OT drive the Figure 2 OLE engine at q64/q128 with uniform/regular noise; measured keygen direction switches/bytes/time per limb and explicit remaining oracles |
| `reports/session_handoff_2026_07_29_dmpf_comparison.md` | **HISTORICAL/SUPERSEDED** pre-sweep, pre-transport handoff; use `CLAUDE.md` for current catch-up and the measured S2 comparison for final rows |
| `reports/s2_architecture_comparison_2026_07_29.md` | **MEASURED ARCHITECTURE COMPARISON**: 275x/329x under uniform noise, but 0.79x for OKVS and only 2.29x for big-state at the deployed regular layout; dealerless-setup status, native-ring artifact defects, and the decision table. Its §7 wording leaves four owner questions open, but their later binding answers are recorded in `CLAUDE.md` under “2026-07-29 owner route decisions”; do not treat them as open or use the uniform result as this project's deployed-layout result. |
| `reports/s2_parameter_novelty_provenance_audit_2026_07_29.md` | **S2 HARD-STOP REPORT, corrected 2026-08-04**: exact primary-source audit, invalid estimator-call rows, unproved projected-noise/structured-code mapping, implementation-only `n=2^17,c=4,t=34` NO-GO, alternatives/provenance, and no pinned parameters or 128-bit claim. |
| `reports/s2_regular_projection_law_2026_08_04.md` | **CURRENT S2 EXACT-LAW REPORT (internal/advisor):** complete `d<=B`/`d>=B` regular-sampler projection distribution, exact lower-tail and both-prime coefficient-cancellation recurrences, BCG+20 Section 8.2/9.1/Table 1 reconciliation, CRT/PCG advantage composition, machine-checkable theorem, and concrete-pin proof obligations. It makes no bit-security or parameter-pin claim. |
| `reports/s2_professor_decision_request_2026_07_29.md` | **Historical advisor request.** Its unresolved security/provenance questions remain required before claim advancement, but its “before S3 implementation” wording predates the owner's implementation-only S3–S6 gate lift and must not be used to deny the component work that subsequently proceeded. |
| `reports/publication_readiness_plan_2026_07_21.md` | **BINDING PUBLICATION ROADMAP**: integrated dealerless Orca FC thesis; advisor-first report; S1--S10 dependency order, security proof and parameter gates, M1--M6 implementation/evaluation criteria, risks, evidence matrix, per-stage user consultation, and required checkpoint commit |
| `reports/dealerless_orca_fc_security_contract_2026_07_29.md` | **CURRENT FORWARD SECURITY CONTRACT:** exact DPF correction-word coupling, role-specific correlated-batch simulators, conversion simulator, full live source-to-transcript map, conditional forward theorem, obligation table, and explicit concrete-parameter/authentication/training limits. |
| `reports/session_handoff_2026_07_21.md` | **HISTORICAL/SUPERSEDED** corrected-M1/v2.3 checkpoint handoff; current status is in `CLAUDE.md` |
| `reports/distributed_dpf_keygen_memo_2026_07_21.md` | **Corrected M1 host protocol-logic prototype**: party-separated and functionally validated by unchanged evaluator using ideal OT/triple/OLE and non-cryptographic correctness PRG; 2,432 trees, three OLEs/tree, old-sign regression, 5/5 corruptions, 6/6 invalid inputs, ideal-mask-draw and correlation-reuse controls, executable split accounting (1,908 logical / 3,816 meaningful share bits at depth 14); `dpf/distributed_dpf_keygen_prototype.{csv,log}` |
| `reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex` (+`.pdf`) | **LIVE internal v2.7 technical report:** current GPU-batched two-process forward-FC design, exact conditional proof boundary and projection law, consume-once Ring-OLE-output Phase-C bootstrap, matched dealer comparison, current ten-trial ResNet18-classifier-layer evidence, closest-baseline disposition, related work, limitations, and explicit “not conference-ready” assessment. Warning-free, page-inspected 24-page PDF; strong internal/advisor checkpoint, not submission-ready. |
| `reports/session_handoff_2026_07_10.md` | **HISTORICAL** proposal-v2 restructure and explainer rationale; superseded by the 2026-07-21 handoff |
| `reports/dealerless_orca_ringlpn_full_proposal_2026_06_10.tex` | HISTORICAL first proposal draft (M1-M6 milestones) — superseded by v2 |
| `reports/ntt_baseline_comparison_2026_06_10.md` | GPU-NTT external baseline vs cheddar (measured; keep-cheddar decision + revisit triggers) |
| `reports/orca_fc_real_ole_transcript_memo.md` | Real-OLE slot-packed FC transcript (Step 5) + NTT backend changes, 2026-06-10 |
| `reports/baseline_2026_06_10.md` | **HISTORICAL** verified baseline: dated environment, PASS counts, and performance anchors; old prime/status claims superseded |
| `reports/orca_ringlpn_dealerless_results_2026_06_05.tex` | June 5 checkpoint report (4 validated checkpoints, NTT decision) |
| `reports/dealerless_orca_ringlpn_protocol_plan.tex` | Protocol plan separating dealer/oracle demo from dealerless target |
| `reports/orca_ringlpn_linear_integration_plan.md` | Living integration plan (phases 0-8 + dated updates) |
| `reports/ole_figure2_host_results.md` | Host 36/36 OLE validation table (135/57/36 counts) |
| `reports/*_handoff.md`, `*_memo.md`, `cheddar_extract_note.md` | Per-artifact handoffs/design notes |
| `reports/ringlpn_status_report.md`, `paper_execution_next_steps.md` | Older status/roadmap snapshots |

## One-command re-validation

```bash
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 PATH=/usr/local/cuda/bin:$PATH \
  ../scripts/run_paper_checkpoint_smoke.sh
# success criterion: exits 0 and prints "[paper-smoke] ALL GATES PASS"
```

Observed before the identity/freshness changes on 2026-08-04 with
`CUDA_VISIBLE_DEVICES=3`: exit zero, 333.04 s, literal final marker present.
Re-run the focused controls and this gate on current source. The marker
certifies only the exercised feasibility configurations, not concrete security
or a live/model/two-host composition.

## Clean-clone/container reproduction

Initialize a clean recursive clone, create an external evidence directory, and
build the digest-pinned image (the context contains only `scripts/`; no source
tree, dataset, weights, or credentials are copied):

```bash
docker build \
  --build-arg REPRO_UID=\"$(id -u)\" --build-arg REPRO_GID=\"$(id -g)\" \
  -f GPU-MPC/ringlpn/scripts/Dockerfile.reproduction \
  -t ringlpn-repro:2026-08-04 GPU-MPC/ringlpn/scripts
```

The local command rebuilds the current PDF and runs the required-GPU canonical
component gate. Its success marker explicitly says that it is **not**
two-host publication evidence:

```bash
mkdir -p /absolute/external/ringlpn-evidence
docker run --rm --gpus all \
  -v \"$PWD:/work/EzPC\" \
  -v /absolute/external/ringlpn-evidence:/output \
  -e RINGLPN_EVIDENCE_DIR=/output \
  -e RINGLPN_RUNTIME_MANIFEST=/output/runtime-local.json \
  ringlpn-repro:2026-08-04 local-smoke
```

Required publication reproduction uses `two-host-publication`, never the local
mode. It additionally requires the authenticated two-host launcher arguments
after `--`, the two party manifests, the checker manifest, and their bound
durable `COMMITTED.manifest`.
Mount the SSH identity and
the separately pinned `known_hosts` read-only (for example under
`/run/private`) and pass their container paths to the launcher; never bake or
record them. Both clean hosts must first build
`bin/test_two_party_fc_preprocess` with the same pinned image (`remote-build`
is the non-gating remote-host preparation mode). Publication success is only
the literal final marker
`[ringlpn-reproduce] TWO-HOST PUBLICATION GATES PASS`; a missing launcher,
credential, peer manifest, submodule, retained evidence destination,
`libmpfr-dev`, `sm_89` GPU, or clean/pinned environment exits nonzero.

A coordinator invocation for the exact current classifier feasibility shape is
one command (replace only deployment paths, peer, fresh session/port, and
runtime credential files):

```bash
docker run --rm --gpus all \
  -v "$PWD:/work/EzPC" \
  -v /absolute/external/ringlpn-evidence:/output \
  -v /absolute/private-ssh:/run/private:ro \
  -e RINGLPN_EVIDENCE_DIR=/output \
  -e RINGLPN_RUNTIME_MANIFEST=/output/runtime-two-host.json \
  ringlpn-repro:2026-08-04 two-host-publication \
  --p0-isolation-manifest /output/p0-isolation.json \
  --p1-isolation-manifest /output/p1-isolation.json \
  --checker-isolation-manifest /output/checker-isolation.json \
  --commit-isolation-manifest /output/COMMITTED.manifest -- \
  --peer USER@REMOTE_HOST --identity /run/private/id_ed25519 \
  --known-hosts /run/private/known_hosts --session-id 202608040001 \
  --base-port 49000 --remote-root /absolute/remote/EzPC/GPU-MPC/ringlpn \
  --p0-output /output/two-host-p0.fc \
  --p1-output /absolute/remote/evidence/two-host-p1.fc \
  --qbits 128 --bw 32 --rows 1 --inner 512 --cols 1000 \
  --ole-n 8192 --ole-c 2 --ole-t 8 --noise regular
```

`check` performs only fail-closed preflight and emits a runtime manifest; it
does not build or run anything. Ignored CSV/log/text outputs are copied
byte-for-byte to the external evidence mount and indexed before success.
Ignored private key/noise/FC records are deliberately not retained.

Conventions: every run produces a `.csv` (data), usually a `.md` (summary), and
a `.log` (raw stdout + stderr). `validation`/`*_contract` columns must read
`pass`; suites exit non-zero on any failure.

**Staleness convention (binding, see `../CLAUDE.md` documentation contract):**
documents whose claims are no longer current carry a `> **HISTORICAL …**`
banner at the top; `outreach/` and `archive/` are wholly historical (see their
READMEs). A document is current only if it is unbannered and dated
2026-06-10 or later. When your work supersedes a document, banner it in the
same commit.
