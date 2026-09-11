# ringlpn results — directory index

Reorganized 2026-06-10. Evidence-producing runners write into their artifact
directory below. The terminal application gate emits only sanitized public rows
and deletes private scratch; its dated CSV/log are explicit retained summaries.

## Engineering review supplement (2026-09-10)

`fc/sci_duplex_worker_review_2026_09_10.json` retains every final warmup and
measured sample from the counterbalanced SCI sender-worker experiment,
including variability, exact binary/header identities, command/timer scope,
and post-measurement source-export commits. All 20 measured invocations pass;
68 per-party contract/accounting fields match. Process-latency median changes
from 1.0413064545 s to 0.965220348 s; slower-party Phase B median changes from
278.1845 ms to 226.2475 ms. These are shared-host, one-shape feasibility
measurements, not security, GPU, full-model, or matched-dealerless claims.
Historical measurements below retain their original identities; they are
not silently replaced by smoke-test timings.

The internal v2.17 paper now includes the measured scheduling result and
explicit ASPLOS research/evaluation/release gates. Its verdict remains
**not submission-ready**. The terminal integration review additionally
records output-substitution, unsupported-callback, FIFO, signed/wrapping,
and stock-timer coverage, while preserving the caller-owned application
freshness and terminal-only boundaries.

`reports/engineering_review_verification_2026_09_10.json` records the
before/after defects, actual command outcomes, application relocation proof,
macro-off stock build, and final required-GPU checkpoint: `ALL GATES PASS`
in 5,228.406 s. The subsequent source-only worktree rebuild/application gate
and fresh q64/bw29/inner-3 FC/Conv online checks also pass. The focused
`fc/two_party_fc_review_2026_09_10.{csv,log}` and
`fc/two_party_fc_review_controls_2026_09_10.csv` retain five live cases and
sixteen controls; their timings are not a benchmark cohort.

`graph/resnet18_full_graph_review_2026_09_10/` retains the fresh 18-file
known-zero graph summary: full-graph digest `ccd598f8...`, all checker fields
and seven rejection controls pass. Both self-digests and all indexed file
hashes verify. It uses P0/P1/checker GPUs 1/2/3, with TEST-ONLY trusted
nonlinear material. This new bundle remains internal-only and contains
non-secret private-ledger path metadata; it does not replace the original
immutable August checkpoint or authorize external release.

`reports/paper_review_verification_2026_09_10.json` records two independent
byte-identical two-pass PDF builds, font/reference checks, and visual review
of all 34 pages. Fifty previously changed historical smoke artifacts match
their pre-review bytes; current approvals/contracts and new review evidence
are kept separate.

## Current checkpoint (2026-08-24)

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
The corrected source fixes unsent identity `a0=1`, derives each `(c-1)*n`
public tail from one jointly seeded, independently domain-separated SHAKE256
random-oracle stream per Ring-OLE, and exchanges only four seed-share words per
party per layer. Its bounded breadth-first FC caller is integrated: the current
focused suite passes all five q64/q128 cases/controls, every row records
positive P0/P1 breadth-call counts and zero root-to-leaf calls, and all 21
shape plans pass. These counters establish current correctness/path use, not
current Conv0 timing or a breadth-first speedup. Each Ring-OLE reserves exactly
`3*c^2*t^2` output slots for the next
DPF Phase C and exposes only the remainder to the application. Each party is a
separate process reading only its own private state. Each loopback socket first
performs mutual HMAC-SHA256 endpoint/context establishment bound to roles,
direction, invocation, claim digest, and fresh nonces. Subsequent protocol
traffic is plain TCP without per-message integrity. The measured runner does
not enforce OS-level peer-file isolation and uses bilateral best-effort
publication. In the separate, unexecuted two-host publication mode, the
coordinator's sealed digest-bound `COMMITTED.manifest`, not either raw record,
is the consumer gate.

The source configuration uses `(n,c,t)=(8192,2,8)`. The retained known-zero
full-graph execution explicitly overrides it to `(262144,2,8)` so the largest
linear layers fit; neither tuple is a concrete-security pin.

The stable facade now atomically exposes one party's bound linear record and
mask state. A macro-gated role in the real Orca inference source consumes one
terminal FC or Conv2D record through the Sytorch lifecycle and stock Beaver
kernels. It intentionally rejects every multi-layer, truncation, nonlinear,
residual, and invalid-output path.

Recorded current live and retained evidence:

- `application/orca_linear_application_2026_08_24.{csv,log}` and
  `application/orca_linear_application_build_provenance_2026_08_24.json` —
  fresh
  q128/bw32 nonzero terminal FC and Conv2D source-native application runs,
  independent additive input/weight shares, public nonzero biases, clear
  modulo-$2^{32}$ oracles, public record/state API coverage, extracted-helper
  regression, and eight bilateral preflight rejection controls. This is
  terminal functional evidence only, not trained-model, multi-layer,
  deployment, performance, or concrete-security evidence.

- `fc/two_party_fc_preprocess_2026_08_04.csv` — five q64/q128,
  regular/uniform, small and q64 multi-batch live configurations; all public,
  key-order, current-transcript, bootstrap-pool, and unchanged-online validators
  pass.
- `fc/two_party_fc_preprocess_controls_2026_08_04.csv` — sixteen focused
  endpoint/context-authentication, consume-once, restart, tail-reuse,
  invocation-collision, ledger-integrity, preflight, stale-output, capacity,
  rename-failure, corrupt-record, and swapped-record controls; every expected
  rejection passes.
- The current regular-noise Ring-OLE expansion anchors at
  `n=8192,c=2,t=8` are 9.086 ms (q64) and 18.230 ms (q128), each from one
  iteration (`n=1` per configuration), so they are diagnostic rather than
  distribution estimates. The retained q64 `t=64` anchors are 881 ms uniform
  and 61 ms regular, with two timed iterations per mode (`n=2` each; four
  total). The distributions and SPFSS domains differ, so no ratio is claimed.
- `fc/two_party_fc_model_scale_cnn2_cnn3_2026_08_14.csv` and its
  aggregate/summary/environment/A-B-audit/control/log companions are the current
  controlled model-FC matrix. A source manifest selects CNN2 FC4
  `(100x256x128)`, CNN2 FC5 `(100x128x10)`, and CNN3 FC5 `(100x64x10)`, all
  q128/bw32 regular at `(n,c,t)=(8192,2,8)`. Each layer has one warmup plus ten
  measured trials; all 30 measured layer trials pass. Both party GPUs were
  quiescent and locked for each run. After both parties exited, the
  shape/contract-matched stock `gpuKeygenMatmul` comparator ran on the same
  quiescent physical GPU as that sample's slower setup-included party.
- CNN2's two-layer aggregate has 26.8818423365-s mean,
  26.89771076-s median, 0.05096530898-s sample SD, 0.06575381975-s R-7 IQR,
  and 95% Student-`t` mean CI `[26.8453839508,26.9183007222]` s. Its
  shape-matched stock-dealer median is 30.5729 ms, paired-ratio median is
  `879.2970985824659x`, application bytes are 1,302,752,736, total recorded
  transport bytes are 1,302,840,696, mean peak host RSS / process GPU bytes are
  305,696,768 / 431,095,808, and semantic dependency layers are 72,278.
- CNN3 FC5 has 0.903894393-s mean, 0.904909782-s median,
  0.01897115914-s sample SD, 0.02851206825-s R-7 IQR, and CI
  `[0.8903232433,0.9174655427]` s. Its dealer median is 14.96295 ms,
  paired-ratio median is `60.3181599795375x`, application bytes are 39,432,504,
  total recorded transport bytes are 39,476,484, mean peak host RSS / process
  GPU bytes are 137,465,446.4 / 431,095,808, and semantic dependency layers are
  1,663. Dependency layers
  are implementation schedule depth, not packet/network rounds. Fixed
  protocol-then-dealer order can retain order bias. These are controlled,
  strongly negative local feasibility comparisons—not speedup, network,
  full-model, or security-level results.
- The retained `fc/two_party_fc_model_scale_2026_08_04.*` v6 artifact family
  (regenerated 2026-08-10) covers the exact ResNet18 classifier-layer shape
  `1x512x1000`, q128/bw32, `n=8192,c=2,t=8`, one warmup plus ten measured
  trials, 10/10 pass. Post-channel setup-included critical-path time has mean
  4.011203588 s, sample SD 0.036601967373448 s, median 4.0193924415 s, R-7
  IQR 0.03481908225 s, and 95% Student-`t` mean CI
  [3.985020117867291, 4.037387058132709] s. It is the per-layer sum of
  `max(total_us+preflight_us+ot_setup_us)` across parties. It excludes
  `PartyChannel` construction, socket establishment, and channel authentication,
  so it is not true end-to-end time. Application traffic is 182,372,344 bytes;
  shape/contract-matched stock trusted-dealer keygen is 14.73535 ms median;
  unchanged two-share online is 1.14969 ms median; and final Orca payload is
  4,108,096 bytes per party. GPU occupancy was uncontrolled and the comparison
  was not run on the same physical GPU. Its public environment file is an
  explicitly labelled sanitized derivative that omits hostname, GPU UUIDs, and
  absolute workstation paths; it binds retained binary `02eaaac9...`.
- The retained aggregate records 11,023 protocol dependency layers,
  142,542,848 median peak host bytes, and 182,416,324 total transport bytes.
  Its 31,929,597,952-byte GPU peak-used metric is device-wide and includes
  unrelated shared-GPU occupancy; it is not process allocation. Legacy post-OT
  median Phase B is 1.955515 s and Phase C is 0.041723 s. The exact
  276-instance plan accounts per party for 1,536 epoch-zero and 210,432
  PCG-supplied Phase-C products, 210,432 consumed plus 1,536 terminal-discarded
  reserved slots, and 1,024 unused application slots. This is not a full
  ResNet18 inference or scale-10 truncation run.
- The live two-party runners generate fresh invocation IDs, require private
  ledgers, and use owner-only, automatically deleted temporary scratch unless
  the caller explicitly supplies a private debugging `WORKDIR`. Current public
  schemas/manifests retain invocation and claim digests, metrics, and logs—not
  raw party key/state records. Publication postflight rejects surviving private
  scratch.
- `fc/linear_adapter_build_provenance_2026_08_10.json` and its
  `linear_adapter_binary_approval_2026_08_07.json` bind two independent,
  byte-identical FC/Conv builds. Schema v2 replaces every clone and
  build-owned path in commands, working directories, and environment receipts
  with `${REPO}`, `${CANONICAL_SOURCE}`, or `${BUILD_ROOT}`. A second build
  under a different temporary root emits the same provenance bytes; the
  record-set validator's negative control rejects a self-consistently rehashed
  raw `/home/...` path.
  The refreshed approval digest begins `2764ac2a...`; current FC/Conv binaries
  begin `ab282ab6...`/`6a9ae142...`.
- `fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json`
  binds all 20 convolutions and the classifier at q128/bw32,
  `(n,c,t)=(262144,2,8)`, with 6,439 Ring-LPN batches, 25,756 Ring-OLE
  instances, 6,593,536 DPF trees, and 130,774,336 payload bytes per party. Its
  isolated-record scope marks the 62 stock items, 21 truncations, eight
  residuals, and four terminal nodes unexecuted; the separately compiled
  full-graph contract binds and executes those exact rows.
- The consumed incomplete attempt formerly under
  `fc/forward_linear_record_set_2026_08_07/` stopped after Conv0 and never
  produced a checker row or final manifest. Its private records and ledger
  were deleted; never cite or resume it.
- One retained maximum-Conv0 q128/bw32 artifact at `n=262144,c=2,t=8`
  records a 358.085-s legacy post-OT critical path, 445 batches, 1,780 Ring-OLE
  instances, and 455,680 DPF trees; its legacy post-OT comparison row is
  384.816 s. Both bind older binary
  `1db001...`, not current Conv binary `6a9ae142...`, so they do not establish
  a current-source breadth-first speedup. Party metrics, record digests,
  metadata, and archive-time checker revalidations are retained under
  `conv/conv0_breadth_comparison_2026_08_09/`; privacy-sensitive raw key records
  are not retained. This is one older-binary isolated-layer comparison, not
  completed model or current performance evidence.
- `secure_truncate/secure_truncate_check_2026_08_06.csv` validates the
  two-party stochastic-truncation subprotocol in isolation. The full-graph
  checkpoint consumes the same live protocol at every one of the 21 exact
  ResNet18 truncation boundaries.
- `graph/resnet18_full_graph_checkpoint_2026_08_10/INDEX.json` binds a fresh
  source-generated q128/bw32 21-record set to one complete known-zero graph
  run. Its exact 62-item stream passes 21 unchanged Conv2D/FC consumers, 21
  live secure truncations over 2,484,224 values, 19 unchanged stock-key
  consumers, 20 remasks, three projection and five identity residuals, integer
  GlobalAvgPool2D, classifier sign extension, terminal reconstruction,
  source/state/trace/counter contracts, and an independent post-exit checker.
  The one-run sum of legacy post-OT linear-layer critical paths is 4,720.417 s;
  the record-consuming graph critical path is 11.852 s.
- The labelled TEST-ONLY trusted adapter reads both parties' mask states,
  centrally samples/distributes both truncation successor-mask shares,
  computes remask/terminal material, and generated 19 stock nonlinear records
  in 9.356 s, totaling 1,084,582,352 raw stock-key bytes per party. Seven
  fail-closed graph-output controls—forced second rename, reused invocation,
  stale output, party-record swap, truncated nonlinear record, nonlinear
  payload corruption, and digest-valid trace corruption—rejected without
  partial output. The bundle has 17 indexed payload files plus `INDEX.json`;
  private records and ledgers were deleted. Its two manifest-bound historical
  adapter-provenance copies retain the original workstation path and ephemeral
  build-root strings, and the historical linear assignment reused GPU 1 for
  party 1 and its later checker. Those values are non-secret but
  host-identifying/internal-only and no longer satisfy the current pairwise-
  distinct retained-assignment policy. The original bytes are preserved so the
  graph/linear manifests, approvals, provenance self-digests, and INDEX remain
  transitively valid; they must not be externally circulated. A fresh canonical
  three-GPU run with normalized provenance must replace this internal checkpoint
  before release. Manifest digest:
  `fdf51f25902afd94a1e67b8bdffa33f762d89c104836538d913c2d7e392c5395`.
  This is one shared-machine known-zero composition run, not dealerless
  nonlinear preprocessing, private/trained inference, accuracy, deployment,
  a timing distribution, or a full-model performance/security claim.
  Each isolated linear record was first consumed by its omniscient correctness
  checker, so this is not consume-once online-deployment evidence. The older
  `graph/resnet18_graph_prefix_*_2026_08_09.*` files are superseded focused
  regression evidence.
- A final 2026-08-10 same-worktree canonical gate regenerated an ephemeral
  21-record set and complete graph, ended literal `ALL GATES PASS`, and
  reported full-graph digest
  `2588eac6de148910835e6f8e09b11b3fc409fb949acbcfb42197bc485a88ad92`.
  The successful gate deleted its private temporary output. It is
  same-worktree revalidation, not another retained checkpoint, a clean-clone
  run, or two-host publication evidence.
- The 2026-08-24 same-worktree canonical gate retained every prior component
  and full-graph check, included the new terminal application rows, ended
  literal `ALL GATES PASS`, and reported fresh ephemeral full-graph digest
  `ec026fa850dfca7b3f51fd7eaef1e729b17a82d03464976a63c662a662a7b410`.
  Its private output was deleted. Unrelated resident GPU workloads make this
  correctness-only shared-machine evidence.

Proof/evidence boundary:

- The v2.17 TeX source and current security contract contain the canonical
  correlation functionality, persistent consume-once ledger, exact
  correction-word coupling, role-specific correlated-batch simulators, the
  masked-difference bootstrap lemma and noncircular epoch induction, conversion
  simulator, source map, conditional forward theorem, regular-DMPF NO-GO, and
  source-bound full-graph systems route. The first page explicitly marks the
  internal/advisor and unresolved authorship/permission boundary. PDF
  build/inspection status is recorded below.
- `P-FRESH` is source/proof closed only under SHA-256 collision resistance and
  a trusted private persistent filesystem providing one deployment-wide ledger
  namespace, exclusive create, fsync, atomic rename, directory fsync, and no
  adversarial storage cloning/rollback. Its sixteen focused controls pass.
- The public record/state loader and terminal role-2 path close one source-native
  FC/Conv2D ingestion seam. Arbitrary multi-layer state transitions, secure
  truncation dispatch, nonlinear setup, residual composition, and trained
  private inference remain outside this result.
- Renewed model-assisted source/proof reviews are current, but they are not
  independent human cryptographic review.
- The exact regular-projection/cancellation law and 2024 regular-ISD artifact
  remain pinned. The live sampler is freshly rebound at `fbdb56f8...`: the
  source diff from the prior audited `05d2fb62...` revision changes only
  optional Phase-C OLE-source forwarding outside `validate_party_noise` and
  `sample_party_noise`. The self-tested hybrid-RSD formula script/CSV are
  freshly paired at `cbcedaf6...`/`1f671d94...`. No reviewed reduction or
  concrete Ring-LPN parameter is pinned; q64/q128 are arithmetic limbs, not
  security levels.
- The canonical rows and classifier `P-PROC` headline artifact use SCI/IKNP
  over same-host loopback. Mutual HMAC-SHA256 authenticates endpoint roles and
  the invocation/claim/direction context before preflight; local post-handshake
  traffic remains plain loopback. Application bytes exclude backend setup and
  TCP/IP overhead; total transport includes 43,658 base-OT setup bytes.
- A current separate opt-in EMP-Silent rerun is retained as
  `fc/two_party_fc_emp_silent_correctness_2026_08_14.csv`, its sixteen-control
  CSV, log, and sanitized environment binding. Under binary `bf4e4f90...`,
  bridge `435f3be6...`, and EMP-OT revision `2fca139f...`, all five live cases
  and all controls pass, every declared straight/reversed 128-bit-OT inventory
  is consumed exactly, and correlation/adjustment/ciphertext bytes are split by
  direction. One trial per case ran while all four GPUs carried unrelated
  workloads. The custom backend remains independently unreviewed; these rows
  support correctness and exact accounting only, not performance,
  bandwidth-improvement, security, or headline claims.
- Results establish executable correctness/cost at feasibility parameters, not
  128-bit, malicious, WAN, trained/private-model, accuracy, or
  full-dealerless-Orca claims.
- The current controlled CNN2/CNN3 model-FC matrix is strongly negative:
  same-physical-GPU, quiescence-checked median paired preprocessing/dealer
  ratios are `879.2970985824659x` for CNN2 FC4+FC5 and
  `60.3181599795375x` for CNN3 FC5. Fixed protocol-then-dealer order remains a
  possible order bias. The retained ResNet18 classifier ratio
  `268.6769431352700x` used uncontrolled occupancy and different physical GPUs;
  it remains descriptive rather than same-hardware A/B evidence.
  Setup-included time still excludes channel construction, socket
  establishment, and authentication. The integrated breadth-first caller's
  current five-case q64/q128 suite and 21 shape-plan pass establish
  correctness/path use only; every focused row records positive P0/P1 breadth
  calls and zero root-to-leaf calls. GPU batching, executable self-bootstrap,
  public-vector XOF, and memory/dependency instrumentation remain implemented.
  The source-bound known-zero control closes the full graph/state seam using an
  explicitly trusted test-only source for both truncation successor-mask
  shares, remask/terminal material, and nonlinear keys; its original-byte,
  internal-only bundle has 17 indexed payload files plus `INDEX.json`.
  Dealerless nonlinear setup, repeated private-input/trained-model evidence,
  authenticated distinct-host execution, a compatible dealerless baseline,
  competitive performance, and independent review remain systems publication
  gates. Further algorithmic Phase-B work, pinned clean-clone reproduction,
  and silent-backend review also remain open.


The source-pinned closest-baseline audit now ranks newly public Reverse Cuckoo /
libOTe first. It supersedes the 2026-07-29 “no public code” statement without
removing that historical record. The pinned stock run is measured but is not
an exact project baseline: process wall was 12.43 s, peak RSS 22,939,444 KiB,
libOTe printed 11 s internally, and local synthetic `setBase` took 446.448 ms;
live `genBaseCors` was excluded. It uses a different field and folded layout,
samples factors internally, and runs on CPU. A separate exact caller-factor
`p0` adapter has exercised live setup and full-domain differential controls for
the native 16-folded layout: setup 18,523,424 us, online full-domain evaluation
2,116,894 us, and end-to-end including validation 20,688,314 us. Raw
31-diagonal timing and GPU evaluation remain unmeasured and non-comparable; no
speedup crosses those boundaries. Silentium (ePrint 2025/1013) and libOTe's
MIT-licensed `RingLpnTriple`/Reverse-Cuckoo implementation predate this
integration; Agarwal--Raghuraman--Rindal and libOTe provide fully distributed
DMPF prior art; and Rivinius et al. (PoPETs 2023/ePrint 2023/359, source
`618301c...`) provide maliciously secure offline convolution triples. No first
Ring-LPN Beaver-triple, first distributed DMPF, or broad first convolution-
preprocessing claim is made. The candidate contribution is only the exact
Ring-LPN-to-Orca GPU FC/Conv integration under the documented boundaries.

The specialized regular-DMPF design audit is also closed as a NO-GO, not an
impossibility theorem. None of Reverse Cuckoo, dense NTT-slot multiplication,
input-independent oblivious cuckoo, programmable DPF, secure active-path AES,
or Ring-OLE-mask reuse simultaneously preserves the current fixed transcript,
plain Ring-LPN/semi-honest boundary, 7,424 application slots, stock-key ABI,
and exact deployed-shape cost. The selected replacement route has implemented
the 21-layer source-bound plan, shared FC/Conv producer, isolated truncation,
ordered fail-closed record runner, SHAKE256 public-vector XOF, one fresh
complete 21-record set, and the exact 62-item stock graph/state composition.
Current focused evidence validates breadth-first caller integration and path
counters, but supplies no current Conv0 timing or breadth-first speedup claim.
The separate controlled CNN2/CNN3 model-FC matrix supplies current timing and
same-physical-GPU stock-dealer comparisons; its result is strongly negative.
The trusted nonlinear adapter remains the explicit dealer boundary. See
`reports/regular_dmpf_design_no_go_2026_08_06.md` and the canonical
`../CLAUDE.md`; the older systems plan and handoff are historical.

The complete canonical component/full-graph gate is self-contained from a
clean-clone parent directory:
```bash
cd GPU-MPC/ringlpn
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 \
CUDA_VISIBLE_DEVICES=<gpu-a>,<gpu-b>,<gpu-c> \
PATH=/usr/local/cuda/bin:$PATH ./scripts/run_paper_checkpoint_smoke.sh
```
All source-changing work requires a fresh zero-exit run before checkpointing.
Alp `<fcetin@hawk.iit.edu>` remains the sole paper author by user direction;
inherited code/protocols remain cited, and unresolved ownership/reuse decisions
must not be treated as settled.

Clean-clone gate configuration is tracked by
`../scripts/publication_environment_manifest_2026_08_10.json` and
`../scripts/Dockerfile.reproduction`; the fail-closed dispatcher is
`../scripts/reproduce_publication.sh`. Docker is restricted to clean-clone
check/local-smoke/build gates. The 2026-08-04 schema-v1 manifest is historical
and nonselectable; live tooling accepts only the 2026-08-10
`ringlpn-publication-environment/v4` manifest. Two-host publication uses the host coordinator
and each host's native rootless Podman, with no nested Podman and no mounted
Podman socket. The manifest pins required CPU/GPU/toolchain/source identities
and the internal gate-image ID. It still leaves annotated-tag/operator source
authorization and the immutable publication runtime unavailable, so
publication remains blocked. Secret bytes stay out of argv, environment,
images, and evidence; owner-private identity/known-hosts paths are arguments.
Consume-once ledgers remain unpublished owner-private host state.

## Where to look

| Directory | Contents | Produced by |
|---|---|---|
| `reports/` | **Start here.** Current plans, proposals, baselines, memos, handoffs | hand-written |
| `ntt/` | NTT/PolyMul sweeps: CPU (NFLlib), GPU cheddar q32/q64/q128, legacy | `run_sweep.sh`, `run_cuda_sweep.sh`, `run_cuda_sweep_legacy.sh`, `run_cuda_single.sh` |
| `ole/` | Figure 2 Ring-LPN OLE q64/q128 × uniform/regular component rows; includes independently sampled two-process SPFSS keygen/provenance followed by the existing both-record OLE checker. The current focused row is `ole_two_party_{keygen,keys}_q64_regular_c2_t8_n8192.csv`; neither it nor the `t=8`/`t=64` feasibility rows are security-pinned or a live two-process FC run. | `run_ole_sweep.sh`, `run_ole_two_party_keys.sh` |
| `linear_ole/` | Ring-matrix OLE-to-Beaver (2x2x2, n=8192): q64/q128 × uniform/regular | `run_linear_ole_sweep.sh` |
| `vole/` | Standalone VOLE expansion prototype | `run_vole_sweep.sh` |
| `orca_fc/` | Orca FC artifacts: keywriter demo, ideal-OLE transcript, **real-OLE slot-packed transcript**, Zp bridge | `run_orca_fc_ringlpn_demo.sh`, `run_orca_fc_ideal_ole_transcript.sh`, `run_orca_fc_real_ole_transcript.sh`, `run_orca_zp_bridge_test.sh` |
| `application/` | Source-native terminal Orca FC/Conv2D public pass/control rows and sanitized gate log. Private records, states, shares, authentication files, ledgers, and outputs are temporary and are never evidence. | `run_orca_linear_application.sh` |
| `fc/` | **Live two-process forward-linear evidence:** the default SCI/IKNP five-case q64/q128 FC rows and sixteen controls; a separately labeled current EMP-Silent five-case/16-control correctness and exact-inventory/byte-accounting rerun with sanitized environment binding; a controlled source-manifest-selected CNN2 FC4/FC5 and CNN3 FC5 matrix with 30/30 measured layer trials and same-physical-GPU quiescence-checked stock-dealer comparisons; retained ResNet18 classifier trials; source-bound 21-layer baseline/adaptive manifests; and fail-closed record-set machinery. The EMP rows are single-trial, occupied-GPU, independently unreviewed non-headline evidence. The controlled CNN comparisons are strongly negative. The former consumed `forward_linear_record_set_2026_08_07/` attempt was incomplete and its private records were deleted. Feasibility parameters and loopback only. | `run_two_party_fc_preprocess.sh`, `verify_emp_silent_fc_evidence.py`, `run_two_party_fc_model_scale.sh`, `verify_controlled_fc_ab.py`, `run_full_linear_manifest_gate.sh`, `run_full_linear_record_set.py` |
| `conv/` | Focused Conv2D CSV plus the retained maximum-Conv0 breadth/pre-breadth party metrics, record digests, and archive-time checker revalidations. Raw private key records are deliberately excluded. | `run_two_party_conv_preprocess.sh` and the dated isolated Conv0 invocation |
| `graph/` | Internal-only retained source-bound known-zero full-ResNet18 checkpoint plus superseded focused prefix rows. The checkpoint binds all 21 fresh linear records and executes 21 truncations, the exact 62-item stock order, 20 remasks, eight residuals, global pool, sign extension, terminal reconstruction, source/state/trace/counter equality, an independent checker, and seven negative controls. Private records are deleted; the trusted test-only adapter owns both truncation successor-mask shares, remask/terminal material, and nonlinear keys. Its original manifest-bound provenance contains host/build-root strings, so a fresh normalized checkpoint must replace it before external circulation. | `run_resnet18_full_graph.sh`, `run_resnet18_graph_contract_gate.sh` |
| `secure_convert/` | Two-process evidence for exact `Z_M -> Z_2^bw` conversion using SCI/IKNP-generated edaBits/daBits/Boolean triples; common preflight, bounded bilateral best-effort outputs, corruption controls, and separate transcript counters. The live forward-FC path consumes this API. The wrap bit is never opened; the current security contract gives the hybrid simulator. A mutual endpoint/context handshake precedes protocol traffic, but later plain-TCP messages have no per-message integrity; linear-depth ripple also remains. |
| `dpf/` | Distributed DPF keygen artifacts: ideal-functionality protocol logic; two-process SCI/IKNP+Gilboa transport with measured bytes/direction switches; full-width four-call GPU AES parity with enforced seed-bit-0 sensitivity; strictly validated GPU-evaluated party keys; offline correctness/corruption/invalid-input controls. Direction switches are not network rounds; security reductions remain open. | `run_distributed_dpf_keygen.sh`, `run_two_party_dpf_keygen.sh`, `run_two_party_gpu_dpf.sh` |
| `profiling/` | VTune hotspot/memory captures | `run_vtune_*.sh` |
| `outreach/` | Abstracts, posters, professor memos/status emails | hand-written |
| `archive/` | Superseded one-offs: early spot checks, `*_regular_patch`, `*_after_linear`, old plan drafts | frozen |
| `security/` | **Start with `security/README.md`.** Current source-bound mathematical/model evidence includes the exact regular-projection law, the 2024 regular-ISD artifact, and the self-tested 2025 hybrid-RSD formula artifact. These are diagnostics only: no reviewed structured-code reduction, concrete parameter pin, quantum cost, or independent cryptographic review exists. |
| `pcg/` | Adapted rows from the licensed native-`Z_(2^bw)`/Galois-ring PCG artifact, with patch digest and correctness gate; not a reproduction of the released benchmark | `run_native_ring_pcg_baseline.sh` |
| external evidence directories | A fresh same-HEAD local-smoke directory contains its finalized runtime, PDF/gate transcript, and fresh known-zero full-graph bundle. A separate fresh publication mount contains the invocation, deletion receipt, final COMMITTED manifest, finalized runtime, and last-committed evidence manifest. Private key/noise/auth records are excluded. | `reproduce_publication.sh` |
| external ledger directory | Deployment-wide consume-before-release session/invocation claims. This owner-only persistent RW mount must be an ancestor-disjoint mount/source from the clone and both evidence roots and must never be published or rolled back. | `run_two_host_authenticated.sh` |

## Reports, newest first

| File | What it is |
|---|---|
| `reports/chief_of_staff_handoff_2026_09_10.md` | **CURRENT NEXT-MODEL OPERATING BRIEF:** repository-wide review scope, dirty-work classification, terminal-integration invariants, security audit map, staged verification sequence, frequent atomic-commit protocol, non-negotiable claim/evidence boundaries, and required final-report format. `CLAUDE.md` remains the technical authority. |
| `reports/orca_linear_application_integration_2026_08_24.md` | **CURRENT SOURCE-NATIVE TERMINAL LINEAR INTEGRATION:** stable bound record/state ingestion, extracted Orca helpers, real role-2 Sytorch lifecycle, nonzero FC/Conv2D oracles, eight bilateral controls, reproducible build provenance, refreshed source/approval pins, complete canonical PASS, and explicit terminal-only claim boundary. |
| `reports/session_handoff_2026_08_09.md` | **HISTORICAL/SUPERSEDED:** prefix-era handoff; use `CLAUDE.md` and this index for current state. |
| `reports/full_linear_layer_systems_plan_2026_08_06.md` | **HISTORICAL/SUPERSEDED:** plan that led to the completed 21-record/full-graph checkpoint; its prefix-only status statements are obsolete. |
| `reports/regular_dmpf_design_no_go_2026_08_06.md` | **SPECIALIZED REGULAR-DMPF DESIGN NO-GO:** exact functionality/cost ceiling, six candidate dispositions, simulator obligations, source boundary, and explicit selection of the full-linear systems route. Not an impossibility theorem or implementation result. |
| `reports/publication_portfolio_2026_08_04.md` | **CURRENT TWO-TRACK PUBLICATION PORTFOLIO (internal/advisor):** separates the systems and cryptographic theses, preserves the feasibility-only claim boundary, and lists the independent evidence and review gates still blocking either submission. |
| `reports/authenticated_two_host_deployment_2026_08_04.md` | **CURRENT AUTHENTICATED TWO-HOST DEPLOYMENT CONTRACT (internal/advisor; launcher binding updated 2026-08-10):** pinned-SSH plus per-run mutual channel authentication, peer-private rootless containers, durable consume-once ledgers, distinct checker isolation, deletion-receipt-bound final commit v2, and exact fail-closed controls. No executed two-host result or security claim. |
| `reports/reverse_cuckoo_p0_baseline_2026_08_04.json` | **EXACT-`p0` NATIVE-FOLDED DISTRIBUTED ROW:** caller factors, canonical 62-bit context, live `genBaseCors`, collision accumulation, full-domain differential check, duplicate/corruption controls. Setup 18,523,424 us; online 2,116,894 us; end-to-end including validation 20,688,314 us. Native 16-folded CPU layout only—not raw 31-diagonal/GPU timing; speedup/security claims are null. |
| `reports/libote_reverse_cuckoo_stock_baseline_2026_08_04.md` | **MEASURED CLOSEST STOCK DISTRIBUTED BASELINE:** pinned clean libOTe build and corrected `-bench` dispatch at `(2^20,4,16)`; 12.43-s process wall, 22,939,444-KiB peak RSS, 11-s internal total, and 446.448-ms synthetic `setBase`. CPU/local-process/Goldilocks/internal-factor/native-16-folded evidence with live `genBaseCors` excluded—not exact `p0`, raw 31-diagonal, GPU, two-host, or live-setup-inclusive evidence, and not a speedup row. |
| `reports/structured_attack_audit_2026_08_04.md` | **STALE IN PART; NO PIN:** the exact projection law, 2024 regular-ISD calculator, and orbit derivation remain pinned mathematical/model evidence. Current sampler source and hybrid-RSD script/CSV differ from the report pins, so implementation correspondence and hybrid evidence require regeneration. No concrete Ring-LPN security claim follows. |
| `reports/closest_dmpf_baseline_audit_2026_08_04.md` | **CURRENT CLOSEST DMPF BASELINE AUDIT (internal/advisor):** Reverse Cuckoo/libOTe is the newly public rank-1 distributed candidate, not zero-change exact/GPU/setup-inclusive evidence. Complete pinned/license matrix, exact 31-diagonal adaptation, collision normalization, stock and exact-control commands, author-contact gates, and mandatory noncomparability rules. Supersedes the 2026-07-29 no-code statement without deleting history. |
| `reports/native_ring_technology_audit_2026_08_04.md` | **INTERNAL/ADVISOR NO-GO:** source-pinned native-ring QA-SD PCG audit covering arithmetic defects, 2025/2026 attacks, centralized/non-matrix/non-Orca boundaries, SPDZ2k semantic mismatch, and a strictly toy-only future correctness oracle. Not a fallback for either publication track. |
| `reports/two_party_dpf_transport_memo_2026_07_29.md` | **HISTORICAL TRANSPORT COMPONENT CHECKPOINT:** retains dated SCI/IKNP/Gilboa transport, host-reference, and GPU-key validation evidence. Its old downstream one-process/unwired-conversion/no-silent-backend status and historical EMP disposition are superseded; use the current security contract/source map and the separately labeled current EMP correctness/accounting rows above. |
| `reports/dealerless_ole_two_party_keys_memo_2026_07_29.md` | **HISTORICAL/SUPERSEDED M2 COMPONENT CHECKPOINT:** preserves paired-record q64/q128 uniform/regular evidence. Its stated live expansion/conversion/proof gaps are obsolete; the current FC/Conv disposition is in the security contract, with `P-PCG` still blocking. |
| `reports/session_handoff_2026_07_29_dmpf_comparison.md` | **HISTORICAL/SUPERSEDED** pre-sweep, pre-transport handoff; use `CLAUDE.md` for current catch-up and the measured S2 comparison for final rows |
| `reports/s2_architecture_comparison_2026_07_29.md` | **HISTORICAL ARCHITECTURE MEASUREMENTS:** preserves dated 275x/329x uniform, 0.79x regular-OKVS, and 2.29x big-state rows. Its no-public-source/no-frozen-route/no-live-FC status is withdrawn; use the closest-baseline audit and `CLAUDE.md`. The rows remain non-comparable to the live deployed path. |
| `reports/s2_parameter_novelty_provenance_audit_2026_07_29.md` | **S2 HARD-STOP REPORT, corrected 2026-08-04**: exact primary-source audit, invalid estimator-call rows, unproved projected-noise/structured-code mapping, implementation-only `n=2^17,c=4,t=34` NO-GO, alternatives/provenance, and no pinned parameters or 128-bit claim. |
| `reports/s2_regular_projection_law_2026_08_04.md` | **PINNED CURRENT-SOURCE MATHEMATICAL LAW:** exact projection/cancellation recurrences are freshly rebound to the unchanged live sampling functions. This is distribution correspondence only, not bit security or a parameter pin. |
| `reports/s2_professor_decision_request_2026_07_29.md` | **Historical advisor request.** Its unresolved security/provenance questions remain required before claim advancement, but its “before S3 implementation” wording predates the owner's implementation-only S3–S6 gate lift and must not be used to deny the component work that subsequently proceeded. |
| `reports/publication_readiness_plan_2026_07_21.md` | **BINDING PUBLICATION ROADMAP**: integrated dealerless Orca FC thesis; advisor-first report; S1--S10 dependency order, security proof and parameter gates, M1--M6 implementation/evaluation criteria, risks, evidence matrix, per-stage user consultation, and required checkpoint commit |
| `reports/dealerless_orca_fc_security_contract_2026_07_29.md` | **CURRENT FORWARD SECURITY CONTRACT:** exact DPF correction-word coupling, role-specific correlated-batch simulators, conversion simulator, full live source-to-transcript map, conditional forward theorem, obligation table, and explicit concrete-parameter/authentication/training limits. |
| `reports/session_handoff_2026_07_21.md` | **HISTORICAL/SUPERSEDED** corrected-M1/v2.3 checkpoint handoff; current status is in `CLAUDE.md` |
| `reports/distributed_dpf_keygen_memo_2026_07_21.md` | **HISTORICAL COMPONENT PROTOTYPE:** preserves the corrected ideal OT/triple/OLE host logic and 2,432-tree controls. Its old “GPU batching/dependency measurement open” disposition and one-visible-GPU full-gate command are obsolete; use the current source map and the three-GPU canonical command below. |
| `reports/dealerless_orca_ringlpn_proposal_v2_17_2026_08_17.tex` (+ current `.pdf`) | **LIVE internal/advisor v2.17 with September 10 review supplement:** conditional forward proof, historical controlled comparisons, scoped SCI scheduling experiment, source-native terminal checks, and explicit ASPLOS research/AE gates. Two fresh pinned TeX Live 2023 two-pass builds produce the same 34-page PDF, SHA-256 `dd1de27a496a3be4251e60f813f0a82ea658612e47ac2624d0859b99f99dffc1`; no warnings, undefined references, or bad boxes, embedded Type 1 fonts only, and all pages visually inspected. Not a conference submission or external-circulation authorization. |
| `reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.{tex,pdf}` | **HISTORICAL v2.16 advisor predecessor:** frozen after the 2026-08-14 checkpoint; superseded by v2.17 above. |
| `reports/session_handoff_2026_07_10.md` | **HISTORICAL** proposal-v2 restructure and explainer rationale; superseded by the 2026-07-21 handoff |
| `reports/dealerless_orca_ringlpn_full_proposal_2026_06_10.tex` | HISTORICAL first proposal draft (M1-M6 milestones) — superseded by v2 |
| `reports/ntt_baseline_comparison_2026_06_10.md` | GPU-NTT external baseline vs cheddar (measured; keep-cheddar decision + revisit triggers) |
| `reports/orca_fc_real_ole_transcript_memo.md` | **HISTORICAL STEP-5 DIAGNOSTIC:** single-process centralized-keygen/clear-conversion, artifact-local 9/9 evidence only; not the live FC/Conv path. |
| `reports/baseline_2026_06_10.md` | **HISTORICAL** verified baseline: dated environment, PASS counts, and performance anchors; old prime/status claims superseded |
| `reports/orca_ringlpn_dealerless_results_2026_06_05.tex` | June 5 checkpoint report (4 validated checkpoints, NTT decision) |
| `reports/dealerless_orca_ringlpn_protocol_plan.tex` | Protocol plan separating dealer/oracle demo from dealerless target |
| `reports/orca_ringlpn_linear_integration_plan.md` | **HISTORICAL/SUPERSEDED integration plan:** preserves dated phases/paths; centralized-SPFSS and conversion-oracle tasks are resolved in the current source map. |
| `reports/ole_figure2_host_results.md` | Host 36/36 OLE validation table (135/57/36 counts) |
| `reports/*_handoff.md`, `*_memo.md`, `cheddar_extract_note.md` | Per-artifact handoffs/design notes |
| `reports/ringlpn_status_report.md`, `paper_execution_next_steps.md` | Older status/roadmap snapshots |

## One-command re-validation

From a clean-clone parent directory:
```bash
cd GPU-MPC/ringlpn
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 \
CUDA_VISIBLE_DEVICES=<gpu-a>,<gpu-b>,<gpu-c> \
PATH=/usr/local/cuda/bin:$PATH ./scripts/run_paper_checkpoint_smoke.sh
# success criterion: exits 0 and prints "[paper-smoke] ALL GATES PASS"
```

With GPU smoke required, the canonical gate rebuilds the shared FC/Conv
adapters through the fixed private canonical symlink, rejects approval-hash
drift, validates the approved adapters and all 21 shapes, and by default runs
the fresh full-graph control before emitting the literal final marker. The
marker certifies only the exercised feasibility configurations and retained
known-zero/trusted-nonlinear graph boundary; it does not certify concrete
security, dealerless nonlinear preprocessing, private/trained inference, or
authenticated two-host deployment.

The 2026-08-06 Conv2D retry first exposed an artificial integration blocker:
upstream Orca's `initGPUMemPool()` pre-reserved 25 GiB in each process. The
Ring-LPN shared engine now configures the default asynchronous CUDA pool
without that reserve, leaving upstream Orca unchanged. Rebuilt adapters passed
the focused two-distinct-GPU Conv2D runner on GPUs 1 and 3 in 1.36 s and the
six-case FC runner in 55.39 s. These are single-layer contract revalidations,
not full-model, authenticated two-host, or comparison-table measurements.

## Clean-clone and two-host publication reproduction

The pinned Docker image is only for clean-clone `check`, `local-smoke`, and
build gates. It is never a publication coordinator, contains no source or
credentials, and must never receive a Podman socket:

```bash
SOURCE_DATE_EPOCH=1786320000 docker build --no-cache \
  --build-arg SOURCE_DATE_EPOCH=1786320000 \
  -f GPU-MPC/ringlpn/scripts/Dockerfile.reproduction \
  -t ringlpn-repro:2026-08-10 GPU-MPC/ringlpn/scripts
GATE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' ringlpn-repro:2026-08-10)"
```

The image uses a fixed build UID/GID and deterministic epoch; the host
dispatcher always runs it as the invoking UID/GID. The tracked manifest pins
`platform.reproduction_gate_image_id=sha256:63b6d387733d145bd26e6d1ada75c3dea568c5144f0e2831dc3590079ab9bd35`.
The dispatcher measures the installed image and requires exact equality. This
internal authorization does not itself prove that two independent clean builds
produced the same image; no such two-build receipt is retained, so image-build
reproducibility remains an external release check. The image ID is distinct
from the pinned CUDA base-image digest. The required CPU flags are `aes`,
`avx2`, `pclmulqdq`, `rdseed`, and `sse4_1`; every visible CPU must provide all
five before a build or GPU gate starts.

Run the same-HEAD local smoke through the host dispatcher into a fresh
owner-private external mount. The dispatcher measures the final image ID with
native `docker image inspect`, checks the tracked authorization, and injects
that identity itself; direct `docker run` is non-evidence. The source clone is
mounted read-only: the container hashes and scans the original clone, copies it
recursively into an owner-private ephemeral execution workspace, runs the PDF
and source-bound full-graph gates there, retains only sanitized external
evidence, deletes the workspace, and requires the original clone to remain
unchanged even after failure.

```bash
install -d -m 700 /absolute/mount/local-smoke-evidence
RINGLPN_REPRODUCTION_IMAGE=ringlpn-repro:2026-08-10 \
RINGLPN_EVIDENCE_DIR=/absolute/mount/local-smoke-evidence \
  ./GPU-MPC/ringlpn/scripts/reproduce_publication.sh local-smoke
# exact success marker:
# [ringlpn-reproduce] LOCAL SMOKE PASS — NOT TWO-HOST PUBLICATION EVIDENCE
```

GPU roles default to party 0/party 1/checker devices `0/1/2`; all three must be
pairwise distinct, including every explicit linear lane. The sequential
test-only trusted adapter may share a party device. It owns both truncation
successor-mask shares, remask/terminal material, and nonlinear records, so this
smoke is known-zero graph/state-composition evidence, not dealerless nonlinear
preprocessing, private/trained inference, accuracy, deployment, or a new
security claim. Its finalized runtime manifest records the measured gate-image
ID and is byte/hash-bound by the evidence manifest committed last.

Publication itself runs directly on the coordinator host and invokes each
host's native rootless Podman. Running the old container command fails exactly:
`two-host-publication must run on each host's native rootless Podman; invoke
./GPU-MPC/ringlpn/scripts/reproduce_publication.sh two-host-publication directly
on the coordinator host (never via docker run).`

Before SSH or any party secret, the host coordinator rejects a container,
root execution, a dirty or untagged HEAD, missing operator-supplied external
source authorization, executor hash drift, absent/stale same-HEAD local-smoke
evidence, non-rootless Podman, disabled user namespaces, insufficient
`/etc/subuid` or `/etc/subgid` ranges, missing CPU flags, unavailable/non-`sm_89`
GPU, a mutable image
reference, or an in-image FC binary SHA mismatch. Publication evidence,
same-HEAD local-smoke evidence, the coordinator ledger, and the local party
ledger must be owner-only, pairwise non-nested mount points backed by distinct
sources; ledgers must be read-write and are retained, never copied, published,
or rolled back. The remote party ledger has the equivalent remote contract.

The manifest currently pins the internal gate-image ID but keeps the annotated
source/tag authorization and immutable publication runtime digest/in-image
binary/final executor SHA unavailable. Publication therefore fails closed until
the operator supplies those release values and removes their explicit blocking
reasons. The external `ringlpn-source-authorization/v1` object is an
operator-supplied consistency binding, not a cryptographically verified signer
attestation: it contains exactly `schema`, `tag`, `tag_object_id`, `commit`, and
`authorization_digest`; its digest is SHA-256 over canonical sorted-key JSON of
the other four fields, and it is supplied from a read-only mount outside the
clone.

After those release prerequisites are satisfied, use this host command for the
current classifier feasibility shape:

```bash
SESSION="$(date -u +%Y%m%d%H%M%S)"
export RINGLPN_LOCAL_PARTY_LEDGER=/absolute/mount/persistent-party0-ledger
INVOCATION="$(openssl rand -hex 16)"
export RINGLPN_EVIDENCE_DIR=/absolute/mount/publication-evidence
export RINGLPN_LEDGER_DIR=/absolute/mount/persistent-ledger
export RINGLPN_LOCAL_SMOKE_EVIDENCE=/absolute/mount/local-smoke-evidence
export RINGLPN_SOURCE_AUTHORIZATION=/absolute/readonly/source-authorization.json
export RINGLPN_RUNTIME_MANIFEST="$RINGLPN_EVIDENCE_DIR/runtime-$INVOCATION.json"

./GPU-MPC/ringlpn/scripts/reproduce_publication.sh two-host-publication \
  --checker-container-uid 10003 --checker-gpu 2 -- \
  --peer USER@REMOTE_HOST --identity /absolute/private-ssh/id_ed25519 \
  --known-hosts /absolute/private-ssh/known_hosts \
  --remote-executor /absolute/remote/EzPC/GPU-MPC/ringlpn/scripts/peer_private_execution.py \
  --local-private-root "/absolute/private/$INVOCATION-p0" \
  --remote-private-root "/absolute/remote/private/$INVOCATION-p1" \
  --local-party-ledger-root "$RINGLPN_LOCAL_PARTY_LEDGER" \
  --remote-party-ledger-root "/absolute/remote/persistent-party1-ledger" \
  --local-party-manifest "$RINGLPN_EVIDENCE_DIR/$INVOCATION/party0-sealed.json" \
  --remote-party-manifest "/absolute/remote/evidence/$INVOCATION/party1-sealed.json" \
  --remote-peer-manifest "/absolute/remote/evidence/$INVOCATION/party0-peer.json" \
  --local-export-root "/absolute/private/$INVOCATION-p0-export" \
  --remote-export-root "/absolute/remote/private/$INVOCATION-p1-export" \
  --checker-stage "$RINGLPN_EVIDENCE_DIR/$INVOCATION/checker-stage" \
  --output-dir "$RINGLPN_EVIDENCE_DIR/$INVOCATION" \
  --local-container-uid 10001 --remote-container-uid 10002 \
  --local-gpu 0 --remote-gpu 1 \
  --session-id "$SESSION" --invocation-id "$INVOCATION" \
  --ledger-root "$RINGLPN_LEDGER_DIR" --base-port 30000 \
  --qbits 128 --bw 32 --rows 1 --inner 512 --cols 1000 \
  --ole-n 8192 --ole-c 2 --ole-t 8 --noise regular
```

The authenticated launcher provisions its per-run channel-authentication key
only as owner-private files under the two private roots; secret bytes never
enter argv, environment variables, images, logs, or retained evidence. It runs
both parties, then the distinct-UID/distinct-GPU networkless checker/finalizer.
Success is not possible until private/export roots, checker records, channel
keys, and party/checker containers are removed on both hosts. The final
`deletion-receipt.json` and `checker-stage/COMMITTED.manifest` bind cleanup;
cleanup failure is non-PASS. The top-level runtime binds the fresh local-smoke
prerequisite, exact runtime/executor/source identities, and the evidence
manifest committed last inventories every retained public file with relative
paths and byte/SHA-256 bindings.

`check` remains containerized preflight only. It does not build or run a gate.

Conventions: every run produces a `.csv` (data), usually a `.md` (summary), and
a `.log` (raw stdout + stderr). `validation`/`*_contract` columns must read
`pass`; suites exit non-zero on any failure.

**Staleness convention (binding, see `../CLAUDE.md` documentation contract):**
documents whose claims are no longer current carry a `> **HISTORICAL …**`
banner at the top; `outreach/` and `archive/` are wholly historical (see their
READMEs). A document is current only if it is unbannered and dated
2026-06-10 or later. When your work supersedes a document, banner it in the
same commit.
