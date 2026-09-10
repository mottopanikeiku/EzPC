# ringlpn — agent & human catch-up guide

**What this is:** a research subproject building *dealerless* preprocessing for
Orca (the GPU FSS-based secure ML system in this repo) from Ring-LPN
pseudorandom correlation generators (PCGs). Orca's linear layers consume
Beaver-triple keys that a trusted dealer normally produces; this project
replaces the dealer with a two-party protocol: GPU NTT/polynomial arithmetic →
Z_p SPFSS (sum of DPFs) → Figure 2 Ring-LPN OLE → slot-packed Beaver cross
terms → Z_M→Z_2^bw conversion → byte-compatible Orca keys, validated through
Orca's **unchanged** online path (`gpuMatmulBeaver`).

**Status (2026-08-24): the current live source is trusted-dealer-free in the
stated random-oracle model, GPU-batched, Ring-OLE-output-self-bootstrapped, and
a two-process forward-FC/Conv2D
preprocessing path at feasibility parameters. It uses a jointly seeded,
domain-separated SHAKE256 public-vector XOF in an explicit random-oracle model,
a canonical 128-bit invocation/256-bit correlation namespace, and a persistent
consume-once ledger. Focused q64/q128 FC/Conv suites, isolated secure
truncation, and all 21 source-bound ResNet18 linear plans pass. A fresh
complete 21-record set and source-generated graph contract also pass.
A public, move-only record/state loader and macro-gated source-native Orca
terminal backend now pass nonzero FC and Conv2D application runs. The backend
uses the real Sytorch lifecycle and stock Beaver kernels, binds both parties in
a fixed-width preflight, and rejects nonlinear, truncation, residual,
multi-linear, and invalid-output callbacks. This closes terminal linear
ingestion only; arbitrary multi-layer mask-state chaining remains open.
The source configuration uses `(n,c,t)=(8192,2,8)`; the retained full-graph
execution overrides it to `(262144,2,8)` so the largest layers fit. Neither is
security-pinned. The internal/advisor checkpoint is retained under
`results/graph/resnet18_full_graph_checkpoint_2026_08_10/` with manifest digest
`fdf51f25902afd94a1e67b8bdffa33f762d89c104836538d913c2d7e392c5395`.
Its original manifest-bound adapter provenance contains host-identifying
workstation/build-root strings, and its historical linear assignment reused
GPU 1 sequentially for party 1 and checker. Preserve those original bytes for
transitive evidence integrity, but do not externally circulate this checkpoint:
a fresh canonical three-GPU run with normalized provenance must replace it.
It composes the exact 62-item ResNet18 forward stream: every unchanged linear,
MaxPool, and ReLU consumer; every truncation and remask; three projection and
five identity residuals; integer GlobalAvgPool2D; classifier sign extension;
and terminal reconstruction. Exact stock nonlinear keys, both parties'
truncation successor-mask shares, remask material, and terminal material still
come from a TEST-ONLY trusted compatibility adapter that reads both parties'
linear mask states. This closes the full graph/state-composition seam, not
dealerless nonlinear preprocessing, private/trained inference, accuracy, a full-model
performance distribution, deployment, or a new security claim. A
conference/security-level claim remains a NO-GO.**

**Engineering review supplement (2026-09-10).** Persistent producer claims are
now serialized under a directory lock; FC/Conv/scale runners refuse reused
work roots instead of deleting consumption history. Private FIFO inputs reject
without blocking. Conservative q64 no-wrap and stock Conv coordinate bounds,
GPU initialization/expansion lifetimes, empty SCI 128-bit OT, terminal output
allocation binding, unsupported callbacks, and child-process cleanup have
fail-before/pass-after reproductions. The actual terminal FC/Conv application
and its eight rejection controls pass, with signed/wrapping stock-helper clear
oracles. The full-graph runner shares the compiled-contract checker's reviewed
manifest pin; changing approval-bound source still requires a fresh approval.

The SCI duplex sender now uses one lazy channel-owned worker, preserving
synchronous single-caller semantics. After worker creation, its per-job
scheduling handoff allocates no storage and copies no payloads.
In the counterbalanced same-host CNN3 FC5-shaped `100x64x10`
experiment, all 20 measured invocations pass and all 68 per-party
contract/accounting fields match. Process-latency median falls from 1.0413 s
to 0.9652 s (7.31%); slower-party DPF Phase B falls from 278.18 ms to 226.25 ms
(18.67%). Raw samples and limitations are in
`results/fc/sci_duplex_worker_review_2026_09_10.json`. This is one shared-host,
unpinned feasibility point, not a GPU, matched-dealerless-baseline,
full-model, or security claim. Producer freshness does not provide persistent
application-consumer replay prevention; callers must use fresh material.

The application/helper build now also enters a private fixed source path;
host prefix maps alone left checkout-dependent CUDA fatbins. The main and
source-only worktrees produce identical binaries and complete provenance
(`f2d16c75...`), and both provenance verifiers pass after that temporary path
is removed. Dependency-only preprocessing uses the equivalent physical source
view, preserving strict rejection of symlinked source inputs. All ten compiler
dependency groups, environment inputs, and linked archives match the pre-fix
build; only the application build recipe changed.

The current live composition uses thin
`src/test_two_party_{fc,conv}_preprocess.cu` entrypoints over
`src/two_party_linear_preprocess.cuh`, plus:

- `src/correlation_freshness.h`: canonical fixed-width correlation IDs and the
  owner-only append-only consume-before-release ledger;
- `src/two_party_spfss.h`: party-local sparse-noise binding and distributed
  SPFSS key generation;
- `src/two_party_dpf_protocol.h` / `src/two_party_ot.h`: full-width GPU-AES
  DPF semantics over SCI/IKNP or the opt-in EMP-Silent backend, external
  Gilboa OLE for each limb's epoch zero, and consume-once prior Ring-OLE
  correlations for every later Phase-C product;
- `src/ringlpn_ole_party.cuh`: party-local Figure-2 Ring-LPN expansion with
  tree-block GPU correction-word/leaf reductions;
- `src/secure_convert.{h,cpp}`: exact two-process `Z_M -> Z_2^bw`
  conversion;
- `src/graph_mask_state.h`: versioned, digest-bound party-local graph mask
  records tied to an invocation, linear-record digest, layer ordinal, shape,
  and bit widths;
- `src/linear_preprocess.{h,backend.cuh}`: the stable move-only
  `OwnedLayerMaterial` record/state boundary with strict private-file,
  self-digest, plan, invocation, ordinal, identity, and byte-binding checks;
- `src/orca_terminal_linear_backend.cuh` and
  `src/orca_linear_application_entry.cuh`: the one-material terminal FC/Conv2D
  Orca backend and macro-gated role-2 implementation over the real inference
  source;
- `src/test_orca_linear_helpers.cu`: trusted-stock-material regression for the
  protected OrcaBase FC/Conv execution helpers;
- `src/resnet18_graph_contract.h`: the compiled 21-linear/62-stream-item source
  contract, explicit main/shortcut value-source registries, remask edges,
  truncations, residuals, global pool, sign extension, and terminal sizes;
- `src/stock_nonlinear_full_record.h` and
  `src/test_stock_nonlinear_full_keygen.cu`: exact full stock-format
  MaxPool/ReLU/sign-extension records produced by an explicitly TEST-ONLY
  trusted adapter that reads both mask-state records and publishes one private
  key record per party;
- `src/test_resnet18_full_graph.cu`: the party runtime and independent post-exit
  checker for the source-bound known-zero full ResNet18 composition;
- `scripts/check_resnet18_graph_contract.py` and
  `scripts/run_resnet18_graph_contract_gate.sh`: source-manifest-to-compiled-
  contract equality plus topology/symlink negative controls;
- `scripts/run_full_linear_record_set.py`,
  `scripts/run_resnet18_full_graph.py`, and
  `scripts/run_resnet18_full_graph.sh`: fail-closed fresh record-set and
  full-graph orchestration, controls, and policy-filtered checkpoint retention; and
- the unchanged Orca `readGPUMatmulKey` / `gpuMatmulBeaver`,
  `GPUConv2DKey` / `gpuConv2DBeaver`, `gpuMaxPool`, and `gpuReluExtend`
  consumers. Each live graph process reads only its own linear, mask-state,
  nonlinear, and persistent-state records.

Each live party is a separate OS process on a distinct GPU, reads only its own
noise record, and samples private roots/masks/noise with OpenSSL's DRBG. Before
OT setup or DRBG construction it claims the complete high-entropy public
invocation namespace in a private persistent ledger; duplicate/restart,
truncated/colliding state, and tail reuse fail before publication. The claim
digest and invocation ID are bound into preflight and both version-3 records.
The public Ring-LPN vector is exactly `a=(1,a1,...,a_{c-1})`: the identity
polynomial is unsent. Each party exchanges four public-seed-share words once
per layer; the joint seed and unique Ring-OLE scope derive each exact-uniform
tail by SHAKE256 rejection sampling in the explicit random-oracle model. The
runner has no selectable centralized DPF keygen, clear conversion, dealer, or
oracle path. Its
temp-write/rename/peer-ack publication is bilateral best-effort, not
crash-transactional. Canonical local runners use owner-only, automatically
deleted temporary scratch unless the caller explicitly supplies a private
debugging `WORKDIR`; publication postflight rejects surviving private
key/state/ledger/scratch files. Raw records are never public evidence. In the
separate, unexecuted two-host publication mode, only the authenticated
coordinator's sealed two-record, fsynced digest-bound `COMMITTED.manifest`
admits a consumer.

**Executable evidence.**

- `results/application/orca_linear_application_2026_08_24.{csv,log}` and
  `orca_linear_application_build_provenance_2026_08_24.json` bind a
  fresh q128/bw32 source-native terminal FC `(2,3,2)` and Conv2D
  `(1,4,4,1;3x3x1x2,pad=1,stride=1)` run with nonzero clear values,
  independently split additive input/weight shares, public nonzero biases,
  independent clear-ring oracles, the public record/state API, and eight
  fail-closed application controls. Both application parties reject
  bilaterally before output in every control. This is terminal functional
  integration, not trained-model, multi-layer, deployment, performance, or
  security-level evidence.

- `results/fc/two_party_fc_preprocess_2026_08_04.csv`: five current
  q64/q128 regular/uniform/small/multi-batch rows. Every key-order,
  current-transcript, exact bootstrap-pool accounting, record, and unchanged-
  online contract passes.
- `results/fc/two_party_fc_preprocess_controls_2026_08_04.csv`: sixteen current
  endpoint/context-authentication, duplicate/restart/tail-reuse/invocation-
  collision/ledger-integrity/preflight/stale/capacity/rename/record controls;
  every expected rejection passes.
- `results/fc/two_party_fc_emp_silent_correctness_2026_08_14.csv` and its
  controls/environment companions bind a current opt-in EMP-Silent rerun:
  the same five live cases and all sixteen rejection controls pass under binary
  `bf4e4f90...`, bridge `435f3be6...`, and pinned EMP-OT
  `2fca139f...`. Every directional 128-bit OT inventory is consumed exactly;
  retained component counters separate correlation, chosen-message adjustment,
  and ciphertext bytes. All four GPUs had unrelated resident workloads and
  there is one trial per case, so timing/memory columns support no performance
  comparison. The custom backend remains independently unreviewed and supports
  no security or bandwidth-improvement claim.
- `results/fc/two_party_fc_model_scale_cnn2_cnn3_2026_08_14.csv` and its
  aggregate/summary/environment/A-B-audit/control/log companions bind the
  source-manifest-selected CNN2 FC4/FC5 and CNN3 FC5 matrix under current binary
  `bf4e4f90...`, q128/bw32 regular noise, and `(n,c,t)=(8192,2,8)`. Every layer
  passes one warmup plus ten measured trials: 30/30 measured layer trials.
  Party GPUs 1 and 3 were quiescent and locked; after both parties exited, each
  stock `gpuKeygenMatmul` comparator ran on the quiescent physical GPU of that
  sample's slower setup-included party. CNN2's two-layer aggregate has
  26.8818423365-s mean, 26.89771076-s median, 0.06575381975-s R-7 IQR, and
  95% Student-t CI `[26.8453839508,26.9183007222]` s; its stock-dealer median is
  30.5729 ms and paired-ratio median is `879.2970985824659x`. CNN3 FC5 has
  0.903894393-s mean, 0.904909782-s median, 0.02851206825-s IQR, CI
  `[0.8903232433,0.9174655427]` s, 14.96295-ms dealer median, and
  `60.3181599795375x` paired-ratio median. Application/total recorded bytes are
  1,302,752,736/1,302,840,696 (CNN2) and 39,432,504/39,476,484 (CNN3);
  semantic dependency layers are 72,278 and 1,663, not network rounds. Fixed
  protocol-then-dealer order can retain order bias. This is a controlled,
  strongly negative local feasibility comparison—not a speedup, full-model,
  network, or security-level result.
- `results/fc/resnet18_adaptive_degree_linear_execution_manifest_2026_08_07.json`
  binds all 20 convolutions and the classifier at q128/bw32,
  `(n,c,t)=(262144,2,8)`: 1,680,390,912 cross terms, 6,439 batches, 25,756
  Ring-OLE instances, 6,593,536 DPF trees, and 130,774,336 linear-key payload
  bytes per party. It explicitly marks all 62 stock key-stream items, 21
  stochastic truncations, eight residual merges, and four state nodes
  unexecuted by the isolated-record artifact.
- The consumed incomplete attempt formerly under
  `results/fc/forward_linear_record_set_2026_08_07/` stopped after Conv0 and
  never produced a checker row or final manifest. Its private records and
  ledger were deleted; never cite or resume it.
- A retained older-binary q128/bw32 maximum-Conv0 artifact at
  `(n,c,t)=(262144,2,8)` records 445 batches, 1,780 Ring-OLE instances,
  455,680 DPF trees, a 358.085-s legacy post-OT critical path, and 7,702,016
  final payload bytes per party; its legacy post-OT comparison row is 384.816 s.
  Both bind binary `1db001...`, not current Conv binary `6a9ae142...`, so they
  establish no current Conv0 timing or breadth-first speedup.
  Retained metrics and archive-time checker rows are under
  `results/conv/conv0_breadth_comparison_2026_08_09/`;
  private records were deleted. This is older-binary isolated-layer evidence.
- `results/graph/resnet18_full_graph_checkpoint_2026_08_10/INDEX.json`
  binds a fresh q128/bw32, `(n,c,t)=(262144,2,8)` source-generated 21-record
  set to a complete known-zero graph run. The linear set contains 6,439
  batches, 25,756 Ring-OLE instances, 6,593,536 DPF trees, and 130,774,336
  final payload bytes per party; the one-run sum of legacy post-OT layer
  critical paths was 4,720.417 s. The exact 62-item consumer stream passed 21
  unchanged Conv2D/FC operations, 21 live secure truncations over 2,484,224
  values, 19 unchanged stock-key consumers, 20 remasks, three projection and
  five identity residuals, integer GlobalAvgPool2D, classifier sign extension,
  terminal reconstruction, source binding, trace, and counter contracts.
- The record-consuming graph critical path was 11.852 s. Its TEST-ONLY trusted
  adapter generated the 19 stock nonlinear records in 9.356 s and 1,084,582,352
  raw stock-key bytes per party. Seven fail-closed graph-output controls—
  forced second rename, reused invocation, stale output, party-record swap,
  truncated nonlinear record, nonlinear payload corruption, and digest-valid
  trace corruption—rejected without partial output. The retained bundle has 17
  indexed payload files plus `INDEX.json` and contains only metrics, logs,
  digests, approval, provenance, and manifests; all private linear, mask-state,
  nonlinear, ledger, and output records were deleted. Manifest digest:
  `fdf51f25902afd94a1e67b8bdffa33f762d89c104836538d913c2d7e392c5395`.
  This is one shared-machine known-zero systems-composition run, not private or
  trained inference, dealerless nonlinear preprocessing, a performance
  distribution, deployment evidence, or a full-model security claim. The
  older `results/graph/resnet18_graph_prefix_*_2026_08_09.*` rows are
  superseded focused-regression evidence.
- The retained `results/fc/two_party_fc_model_scale_2026_08_04.*` v6 artifact
  family (regenerated 2026-08-10) records the exact ResNet18 classifier layer
  `1x512x1000`, q128/bw32 feasibility `(n,c,t)=(8192,2,8)`, one warmup plus ten
  measured trials, 10/10 pass. Post-channel setup-included critical-path time
  has mean 4.011203588 s, sample SD 0.036601967373448 s, median 4.0193924415 s,
  R-7 IQR 0.03481908225 s, and 95% Student-`t` mean CI
  [3.985020117867291, 4.037387058132709] s. This deterministic metric sums,
  per layer, `max(total_us+preflight_us+ot_setup_us)` across parties; it
  excludes `PartyChannel` construction, socket establishment, and channel
  authentication, so it is not true end-to-end time. Application traffic is
  182,372,344 bytes; shape/contract-matched stock `gpuKeygenMatmul` is
  14.73535 ms median; unchanged online execution is 1.14969 ms median; and
  final payload is 4,108,096 bytes per party. GPU occupancy was uncontrolled
  and the runs were not controlled on the same physical GPU. The environment
  binds retained binary `02eaaac9...`.
- That aggregate records 11,023 dependency layers, 142,542,848 median peak host
  bytes, and a device-wide 31,929,597,952-byte peak-used observation that
  includes unrelated shared-GPU occupancy. Its exact 276-instance plan accounts
  per party for 1,536 epoch-zero and 210,432 PCG-supplied Phase-C products,
  210,432 consumed plus 1,536 terminal-discarded reserved slots, and 1,024
  unused application slots.
- Legacy post-OT median Phase C is 0.041723 s, Phase B is 1.955515 s, Phase A
  is 0.2462935 s, and GPU Ring-LPN expansion is 1.226650 s.

The comparison remains deliberately negative: the median setup-included
shape-matched per-trial preprocessing/dealer ratio is 268.6769431352700. It is
not a same-physical-GPU/occupancy-controlled A/B comparison. It uses SCI/IKNP
on single-host IPv4 loopback. Application bytes exclude setup; total transport
includes 43,658 recorded base-OT setup bytes. The setup-included timing still
excludes channel construction, socket establishment, and authentication.

The canonical and classifier `P-PROC` headline evidence remains SCI/IKNP. A
current separate EMP-Silent correctness/accounting rerun passes the same five
focused q64/q128 cases and sixteen controls with exact directional inventories,
but it is one uncontrolled-occupancy trial per case. It is opt-in,
independently unreviewed, and not performance, bandwidth-improvement, security,
or headline evidence.

**Proof boundary.** The corrected security contract and report contain:

1. exact level-by-level and final-CW coupling to standard DPF generation
   conditioned on party roots;
2. complete role-specific simulators for correlated/repeated DPF batches,
   including the three-OLE Phase C without the removed sign leak;
3. exact ideal-OT wrapper lemmas and a `Z_Q -> Z_2^bw` conversion simulator in
   the edaBit/daBit/Boolean-triple hybrid;
4. a role-indexed Figure-2 simulator that retains the corrupt party's local
   noise/key state and recomputes both
   `X_b=e_(b,0)+sum_(i>=1)a_i e_(b,i)` and its local `Z_b`;
5. a canonical `P-FRESH` functionality matching the fixed-width source tuple,
   consume-once ledger, record binding, and exact SHA-256/durable-filesystem
   assumptions;
6. a masked-difference lemma realizing each post-base Phase-C multiplication
   from a consume-once earlier Ring-OLE slot, plus an epoch-order noncircular
   simulator/induction;
7. an updated live source-to-transcript map; and
8. a conditional static-semi-honest theorem for one forward FC matmul under
   authenticated channels, standard AES/DPF, the selected semi-honest OT/OLE
   realization, SHA-256 and durable no-rollback ledger assumptions, and exact
   decisional module-Ring-LPN for `a=(1,a1,...,a_{c-1})`.

Renewed model-assisted source, composition, and proof audits were run after the
identity, freshness, batching, and transport changes; they do not substitute
for independent human cryptographic review, which remains open. Each live
loopback socket performs mutual HMAC-SHA256 endpoint/context establishment
before preflight, bound to roles, direction, invocation, claim digest, and fresh
nonces. Subsequent SCI/IKNP/application bytes remain plain TCP without
per-message authentication, so the theorem's authenticated-channel integrity
assumption is not realized. The pinned-SSH two-stream, peer-private deployment
boundary is implemented in `scripts/run_two_host_authenticated.sh` and
documented in `results/reports/authenticated_two_host_deployment_2026_08_04.md`;
no authenticated two-host result is claimed until that launcher is run and its
durable digest-bound `COMMITTED.manifest` passes the checker gate.
Publication mode requires its coordinator session/invocation ledger on a
separate owner-only persistent read-write mount outside the clone and retained
evidence; deleting, cloning, publishing, or rolling back that mount is forbidden.

**Hard blockers.** The exact regular-projection/cancellation law is pinned
mathematically and freshly rebound to current `two_party_spfss.h`
(`fbdb56f8...`) after semantic review established that its sampling functions
are unchanged. The 2024 regular-ISD artifact remains pinned; the self-tested
hybrid-RSD script/CSV are freshly paired at
`cbcedaf6...`/`1f671d94...`. No reviewed reduction maps
the structured projected code to a concrete attack advantage, and no
two-CRT-limb composition supports a security level. q64/q128 mean one/two
approximately 62-bit arithmetic limbs, not 64/128-bit security. No
`(n,c,t,p0,p1)` set is pinned.
The source-bound full-graph control carries every party-local state mask,
explicit main/shortcut operand, truncation, remask, residual, pool, sign
extension, and terminal value across all 21 isolated linear records. Its
TEST-ONLY trusted adapter reads both parties' mask states, centrally samples
and distributes both truncation successor-mask shares, computes remask/terminal
material, and generates the stock nonlinear key stream. This closes the
known-zero graph/state-composition seam only. The remaining systems gates are
dealerless nonlinear setup, repeated private-input/trained-model evidence,
authenticated two-host execution, a compatible dealerless baseline,
competitive performance, and independent review. Clean-clone reproduction also
remains open. Training state transitions, malicious security, and full
dealerless Orca remain out of scope.

**2026-08-10 systems-route checkpoint.** The specialized regular-DMPF audit
returned a design NO-GO, not an impossibility theorem. The approved replacement
route now has a shared FC/Conv engine, all 21 source-bound linear plans, an
isolated secure-truncation API, fail-closed plan/record-set machinery, SHAKE256
public-vector generation, and a fresh complete 21-record run. Bounded
breadth-first DPF evaluation is integrated: the current focused FC suite passes
all five q64/q128 cases, every row records positive P0/P1 breadth-call counts
and zero root-to-leaf calls, and all 21 shape plans pass. The current FC/Conv
binaries begin `ab282ab6...`/`6a9ae142...`, and the focused approval digest
begins `2764ac2a...`. These counters are correctness/path evidence only;
no current Conv0 timing or breadth-first speedup is claimed. A separate current
controlled CNN2/CNN3 matrix now supplies model-FC timing and a matched
same-physical-GPU stock-dealer comparison; its result is strongly negative.
A generated compiled contract and trusted
stock-key adapter compose those records through the exact 62-item ResNet18
stream with explicit branch handles and post-exit reconstruction of every
value. The graph seam is therefore closed for one known-zero systems control.
The trusted nonlinear adapter, unpinned parameters, performance gap,
private/trained model evaluation, authenticated two-host run, and independent
review remain open. The superseded prefix-era plan and handoff are retained as
historical reports; this document and `results/README.md` are binding.

The exact S2 audit is
`results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md`.
It invalidates the former “conservative pins”: BCG+20's projection
formulas/prose/Table 1 conflict, several saved estimator calls are
out-of-domain, and surviving finite-field rows lack the necessary reduction.
The `n=2^17,c=4,t=34` implementation NO-GO remains an engineering result, but
its former 257.02-bit label is withdrawn.
The attack inventory is
`results/reports/structured_attack_audit_2026_08_04.md`. Its exact projection
law, 2024 regular-ISD formula artifact, and cyclic-orbit derivation remain
pinned mathematical/model evidence. The live sampler is freshly rebound at
SHA-256 `fbdb56f8...` after a semantic diff established that the audited
sampling functions are unchanged; the self-tested hybrid-RSD formula artifact
is freshly regenerated at script/CSV SHA-256 `cbcedaf6...`/`1f671d94...`.
Generic-estimator rows, modern direct RSD and 2025/2026 QA-SD dispositions, structured-code reductions,
resource/success accounting, and independent human review remain parameter-pin
blockers.

**Paper and publication verdict.** The live v2.17 source now includes the
September 10 engineering/conference-readiness supplement:
`results/reports/dealerless_orca_ringlpn_proposal_v2_17_2026_08_17.tex`.
Two fresh pinned TeX Live 2023 two-pass builds are byte-identical; the
34-page PDF has SHA-256
`dd1de27a496a3be4251e60f813f0a82ea658612e47ac2624d0859b99f99dffc1`.
The final log has no warnings, undefined references, or bad boxes; all fonts are
embedded Type 1, and every page was visually inspected. The manuscript is an
internal/advisor checkpoint, not a submission candidate. Its first-page
boundary leaves authorship, contributor credit, private-project reuse
permission, acknowledgement, and disclosure to an explicit professor/owner
ruling. A cryptography paper still needs a reviewed reduction/parameter result.
The same-hardware stock-dealer comparison is closed for CNN2 FC4/FC5 and
CNN3 FC5, but remains strongly negative. An ASPLOS systems result needs a
generalizable architecture/OS/PL contribution, matched-assumption baselines,
causal ablations and held-out predictions at a reviewed parameter point,
and independent review. Full-model/private-trained claims additionally need
their corresponding nonlinear, state-chaining, and deployment evidence.
A rigorous negative-result route needs new generalizable insight, not
necessarily a speedup. The binding route is
`results/reports/publication_readiness_plan_2026_07_21.md`; the current proof
boundary is
`results/reports/dealerless_orca_fc_security_contract_2026_07_29.md`.

For complete source re-validation, run:
`RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 PATH=/usr/local/cuda/bin:$PATH
./scripts/run_paper_checkpoint_smoke.sh`.
The gate rebuilds both hash-approved FC/Conv adapters twice through a private,
fixed canonical source symlink. Their v2 provenance records normalize every
command, working directory, and build-owned environment path to
`${REPO}`, `${CANONICAL_SOURCE}`, or `${BUILD_ROOT}`; a raw workstation or
ephemeral build root is a hard failure. Thus the provenance bytes and ELF
outputs are identical across absolute clone/container mount paths. Any SHA
drift from `results/fc/linear_adapter_binary_approval_2026_08_07.json` rejects
before record execution. For a repeat build the gate removes only the prior
owner-owned regular ignored graph-provenance output; a symlink, nonregular
entry, or foreign-owned entry rejects. After any adapter source/build change,
run both focused commands separately:
`scripts/run_two_party_fc_preprocess.sh` and
`scripts/run_two_party_conv_preprocess.sh`.
The final 2026-08-10 same-worktree source revalidation regenerated all 21
linear records and the complete graph, ended `ALL GATES PASS`, and reported
fresh ephemeral full-graph digest
`2588eac6de148910835e6f8e09b11b3fc409fb949acbcfb42197bc485a88ad92`;
its private temporary output was not retained and it is not clean-clone or
two-host publication evidence.
The 2026-08-24 same-worktree canonical gate added the source-native terminal
application rows, retained every prior component and full-graph check, ended
`ALL GATES PASS`, and reported fresh ephemeral full-graph digest
`ec026fa850dfca7b3f51fd7eaef1e729b17a82d03464976a63c662a662a7b410`.
Its private output was deleted. Unrelated long-lived workloads occupied GPUs 0
and 2, so this remains correctness-only shared-machine evidence.
Refresh the approval only with owner approval before the canonical gate can
pass. Other focused runners are
`scripts/run_secure_truncate_test.sh`,
`scripts/run_full_linear_manifest_gate.sh`, and
`scripts/run_resnet18_full_graph.sh`.
The full-graph wrapper remains serial by default. Parallel linear preprocessing
is opt-in with explicit, comma-separated exclusive lane descriptors
`P0_GPU:P1_GPU:CHECK_GPU:FIRST_PORT-LAST_PORT`, for example:
`LINEAR_LANES='0:1:2:22000-22085'
./scripts/run_resnet18_full_graph.sh ABS_OUTPUT_ROOT ABS_STATE_ROOT`.
Each lane reserves three pairwise-distinct GPUs and at least 86 ports; checker
execution starts only after both parties exit. GPU ordinals and port ranges
must not overlap across lanes.

**Contribution/provenance boundary.** The paper thesis is the integrated
forward-FC systems path. The per-point distributed DPF, Ring-LPN generator,
conversion primitives, Orca, and GPU polynomial backends are prior/inherited
work, not protocol contributions. The separate private GPU-PCG/PIM stream has
multiple contributors, no repository license, and unresolved ownership,
credit, chronology, reuse, and overlap decisions; do not import or claim it.
Cheddar remains an attributed MIT-licensed dependency under
`extern/Cheddar_{PROVENANCE,MIT_LICENSE}.txt`; GPU-NTT remains an external
cited baseline. Alp `<fcetin@hawk.iit.edu>` is the sole paper/repository author
by user direction, but that does not erase attribution or resolve reuse rights.
Before external circulation, obtain the professor's provenance/credit/reuse
decisions recorded as open in
`results/reports/s2_professor_decision_request_2026_07_29.md`.

The first architecture comparison is measured in
`results/reports/s2_architecture_comparison_2026_07_29.md`. Its headline result
is negative and must not be misquoted. With uniform noise an OKVS-style DMPF
expands the sparse product 275x faster than the then-current sum of point DPFs
at `(n,c,t)=(2^14,4,16)`. With the deployed regular layout
(`spfss_domain=2048`, `log_domain=11`, 31 diagonal groups per pair), the same
encoder is 0.79x (slower), while the big-state candidate wins 2.29x at 37x the
key bytes. This is a microarchitecture result, not an end-to-end result.

Reverse Cuckoo became public on 2026-08-03. The pinned stock libOTe run in
`results/reports/libote_reverse_cuckoo_stock_baseline_2026_08_04.md` completes
in 12.43 s with 22,939,444 KiB peak RSS, but uses Goldilocks, internally
sampled factors, a native 16-folded layout, synthetic preloaded base
correlations, CPU local sockets, and no GPU output. It is the closest
reproducible distributed baseline, not a compatible comparator, and no timing
ratio is claimable. The separate native-ring artifact remains an adapted
diagnostic, never a reproduced dealerless result. There is no first Ring-LPN
Beaver-triple, first fully distributed/dealerless DMPF, or broad first
convolution-preprocessing claim. Silentium (ePrint 2025/1013) and libOTe's
MIT-licensed `RingLpnTriple`/Reverse-Cuckoo code precede this integration;
Agarwal--Raghuraman--Rindal (ePrint 2025/2294) and the same libOTe branch
provide fully distributed DMPF prior art; and Rivinius et al. (PoPETs 2023,
ePrint 2023/359, source `618301c...`) provide maliciously secure offline
convolution triples. The candidate contribution is only the exact
Ring-LPN-to-Orca GPU FC/Conv integration under the documented boundaries.

**2026-08-04 closest-baseline correction (supersedes only the no-code/public-source
sentences above; preserves the 2026-07-29 measurement as history).** Reverse
Cuckoo became public on 2026-08-03. The current source-pinned ranking and claim
boundary are in
`results/reports/closest_dmpf_baseline_audit_2026_08_04.md`: MIT-licensed
`osu-crypto/libOTe:dmpf@edb5d32822eabf2dda9f6844d85d0ce2e402cdd5` is the
rank-1 distributed candidate, with paper source
`ladnir/dmpf@b55bcc4696d10e57bdea8c282a851fdd4fad0c2b`. The stock runner is
not zero-change exact, GPU, or setup-inclusive evidence: it uses a mismatched
field, internally sampled factors, native 16-folded layout, CPU expansion, and
synthetic base correlations. The companion
`results/reports/libote_reverse_cuckoo_stock_baseline_2026_08_04.md` records a
pinned clean run using the required `-bench` dispatcher: 12.43-s process wall,
22,939,444-KiB peak RSS, 11-s printed internal total, and 446.448-ms synthetic
`setBase`. Live `genBaseCors` was excluded. The literal command without
`-bench` only printed help and exited 0. This separately labelled stock row
does not permit a speedup claim. The separate completed
`results/reports/reverse_cuckoo_p0_baseline_2026_08_04.json` exercises exact
caller-factor `p0`, the canonical 62-bit context, live `genBaseCors`, duplicate
accumulation, full-domain differential validation, and corruption rejection for
the explicitly labelled native 16-folded layout. The retained JSON records
18,523,424-us setup, 2,116,894-us online full-domain evaluation, and
20,688,314-us end-to-end including validation. This is not raw 31-diagonal
timing or GPU evidence;
speedup/security claims remain null, and no ratio may cross layout, field,
trust, correlation, execution, or setup boundaries.

**2026-08-04 primary-source parameter correction (supersedes the 2026-07-29
“conservative pin” interpretation).** The 2026-07-29 owner decision still lifts
the S2->S3 ordering gate **for implementation only**: GPU distributed-keygen
and real OT/OLE transport work may proceed without a security or parameter
claim. Its requested “conservative minimum” selection method is not usable.
BCG+20's corrected full version is internally inconsistent: Section 8.2 derives
`c*d*(1-(1-1/d)^t)`, while Section 9.1 uses
`w-c*d+(c*(d-1)+w)*(1-1/d)^(t-1)`; its literal smallest-factor criterion
selects degree 16 for `(c,w)=(4,64)`, while Table 1 reports degree 128. No
published erratum or proof resolves these differences.

The accepted EUROCRYPT 2024 artifact also cannot evaluate every locally
“admitted” row. Its aggregate finite-field function unconditionally calls
formulas containing `C(N-k,t)` and `C(N-k-1,t)`, so a projected row must at
least satisfy `t' <= N-k-1 = d-1`. The artifact's combination helper silently
returns 1 for out-of-range inputs. Consequently the saved 57.293-bit
`(c,t,d)=(4,16,16)`, 218.641-bit `(4,64,64)`, and 257.023-bit
`(4,34,64)` values are invalid estimator calls, not attack costs. For
`c=4,t=16`, the first mechanically defined saved row is degree 64 at 135.12
regular-model bits; BCG's degree-128 row gives 145.85. For `c=4,t=64`, the
first mechanically defined saved row is degree 256 at 470.77. The
`c=2,t=128,d=256` row is mechanically defined and reports 190.53, but all of
these finite-field numbers remain heuristics because no reviewed reduction
maps the dependent projected noise/code to the estimator's exact or regular
model, bounds the lower tail and rounding loss, or composes the two limbs and
PCG advantage.

All `results/security/*conservative_pin*` artifacts and
`s2_conservative_parameter_pin_2026_07_29.csv` are historical failed-rule
transcripts, invalid for parameter selection or security claims.
`s2_projection_estimator_preliminary_2026_07_29.csv` is a raw function
transcript only; rows with `floor(expected) > degree-1` are invalid calls and
all other rows are unproved model outputs. No `(n,c,t,p0,p1)` set is pinned,
no 128-bit classical or quantum claim is made, and S2 remains blocked pending
a reviewed sparse-factor projection/distribution/tail/structured-code and
advantage-composition analysis for both limbs.

**2026-07-29 owner route decisions (recorded after the measured comparison).**
Presented `results/reports/s2_architecture_comparison_2026_07_29.md` §6-§7 to the
owner; the four answers are binding:
1. **Encoder:** keep the per-point DPF. Implementation effort goes to *real*
   silent-OT/OLE transports and a two-process deployment. Rationale is measured,
   not aesthetic: at the deployed regular layout the best DMPF wins only
   2.29x expansion at 37x the key bytes, and OKVS is 0.79x (slower), so the
   encoder is not the pipeline's lever. A dealerless DMPF stays future work.
2. **Parameters:** the requested immediate conservative pin is superseded by
   the 2026-08-04 primary-source correction above. Obtain a reviewed
   projection/distribution/tail/structured-code and two-limb advantage
   reduction before another estimator sweep. Current rows are feasibility-only.
3. **Claim scope:** the paper's headline claim waits for a real two-process
   dealerless FC run. Protocol-logic slices are supporting evidence only, never
   the contribution.
4. **Prior art:** draft (do not send) an artifact/clarification request to the
   Reverse Cuckoo authors for owner approval:
   `results/outreach/reverse_cuckoo_artifact_request_2026_07_29.md`.

**Invalidated parameter-pin transcripts (measured 2026-07-29; corrected
2026-08-04).** The values in
`results/security/ringlpn_conservative_pin_2026_07_29.{csv,log}`,
`ringlpn_conservative_pin_refine_2026_07_29.{csv,log}`,
`ringlpn_conservative_pin_n16_n17_2026_07_29.{csv,log}`, and
`s2_conservative_parameter_pin_2026_07_29.csv` document a failed rule and are
not current evidence. Their `meets_target=yes`, “conservative,” “pin,”
“surviving,” and `t=32 -> 34` projection-eviction interpretations are invalid.
The model values 57.293, 111.244, 218.641, and 257.023 were selected from
out-of-domain aggregate calls. Mechanically defined values such as 135.12,
145.85, 190.53, and 470.77 remain unproved finite-field-model heuristics, not
Ring-LPN security estimates. `scripts/audit_ringlpn_finite_field_models.py`
now rejects undefined tuples, requires both primes, labels outputs as model
diagnostics, and exits nonzero so automation cannot treat them as a pin.

**Real two-party transport (2026-07-29; rerun 2026-08-03).** The frozen keygen
protocol runs as **two OS processes over TCP with real OT**:
`src/test_two_party_dpf_keygen.cpp` + `src/two_party_ot.h`, using this repo's
unmodified SCI IKNP OT extension (header-only, links only OpenSSL), Gilboa
`Z_p` OLE, OT-based Boolean triples, and OpenSSL-private-DRBG party roots.
Keygen is **batched level-synchronously**: the measured direction-switch count
is `6L+6` for depth `L`, independent of batch size. It is not a network-round
count. 369/369 key pairs across ten configurations (depths 4–14, both primes,
batches 1–256) validate through unchanged `dpfEvalAll` in a separate offline
checker with a corrupted-key control. Logical/meaningful-share columns match
the contract's closed forms at every batch size. Setup costs 256 base OTs and
21,829 bytes per party; at `L=11`, batch 1 → 256 lowers per-tree bytes
52,626 → 3,789 (13.9x) and loopback time 11.2 ms → 148 us (75x), with 72
direction switches. IKNP is OT *extension*, not silent OT; splitmix mode is a
host-reference correctness path, not a security claim. See
`results/reports/two_party_dpf_transport_memo_2026_07_29.md`.

**Candidate feasibility result (2026-08-03): NO-GO on the current
implementation.** The measured `n=2^17,c=4,t=34` tuple is not runnable in the
current layouts and has no accepted security estimate. Regular-noise equal
buckets require `t | n`; `34` does not divide a power-of-two ring. Uniform
noise would materialize `7,272,923,136` host slots at 17 bytes each, at least
123.6 GB for one process, before validation and two-party duplication. These
implementation facts remain valid, but the former 257.02-bit and
projection-eviction narrative is withdrawn as an out-of-domain estimator
calculation. No replacement parameter is pinned.

**GPU key compatibility from the two-party protocol (2026-07-29; full-width
PRG correction 2026-08-03).** The deployed Ring-LPN device PRG and its host
twin now use four domain-separated AES calls per node
(`src/gpu_aes_prg_host.h` and `aes_prg_expand` in `src/gpu_spfss_zp.cuh`):
plaintexts 0 and 2 produce full 128-bit child seeds; plaintexts 1 and 3
produce separate control bits. Every GPU run regenerates 16 device vectors,
including low-bit-one seeds; host parity reports zero left/right/tag
mismatches plus a seed-sensitivity control. With `--prg gpu-aes`, 88
two-process key pairs over four configurations (`L=4/8/11`, both primes) pass
batched-SPFSS and per-tree full-domain GPU reconstruction with corrupted-CW
controls firing (`scripts/run_two_party_gpu_dpf.sh`,
`results/dpf/two_party_gpu_dpf_2026_07_29.csv`). This is GPU key compatibility
and GPU-validated correctness, not GPU-side key generation. The 127-bit
encoding defect is removed. The exact coupling now reduces joint-key
distribution/privacy to the standard DPF/PRG theorem, but no concrete
single-key or 128-bit security claim is attached.

**M2 core gate reached (2026-07-29, new): the real Ring-LPN OLE engine runs on
two-party dealerless SPFSS keys.** `build_spfss_keys()` - the pipeline's
centralized-keygen oracle - is now replaceable by two OS processes over real
IKNP OT: `src/test_two_party_spfss_keygen.cpp` (shared protocol in
`src/two_party_dpf_protocol.h`, GPU expansion PRG) plus two env-gated hooks in
`src/bench_ole_ringlpn_cuda.cu` (`RINGLPN_OLE_EXPORT_NOISE`,
`RINGLPN_OLE_SPFSS_KEYS`; with neither set the bench is byte-for-byte its old
self). The engine's own `validation`/`host_validation`/`correct` all pass on
dealerless keys in **all four deployed configurations** (q64/q128 x
uniform/regular), expansion and validation code unmodified:
256 trees/limb, one level-synchronous batch, 89 stages at `L=14` (uniform) and 71
at `L=11` (regular), ~1.0 MB per party per limb, keygen 0.90 s uniform / 0.13 s
regular after a 6.8x host-PRG speedup. Regular noise is ~7x cheaper to key for
the same tree count (domain `2^11` vs `2^14`) - the same effect that collapsed the
DMPF encoder advantage. Wired into the required-GPU gate (`ALL GATES PASS`).
This component checkpoint has since been superseded by the live
`test_two_party_fc_preprocess` composition, which performs party-local
expansion and consumes the exact conversion API. IKNP remains the principal
host/transport bottleneck and is not silent OT.

**Two-process conversion transport validated (2026-08-03).**
`src/test_secure_convert.cpp` and `scripts/run_secure_convert_test.sh` replace
the standalone prototype's labelled edaBit/daBit/AND-triple dealer with the
live two-socket SCI/IKNP path. One exact daBit over `Z_(2^k)` uses one 128-bit
OT; `ell` daBits over `Z_(2^ell)` compose an exact edaBit because the
coefficient-one arithmetic share is uniform independently of the Boolean
shares. Boolean triples use two one-bit OTs each. Independent processes write
only party-local versioned records; the offline checker covers exact
`0,Q-1,Q,2Q-2` boundaries, random inputs, forced wraps, and layer-shaped
inner products, and requires a corruption control to fire. All 76 conversions
in each of four q64/q128/bit-width configurations pass. Per-party rows
separate base-OT setup, correlation, and online bytes/direction switches and
gate `5ell-3` logical / `10ell-6` meaningful-share / `2ell-1` post-mask
accounting. An invalid public-bound control rejects before opening a socket.
The mutual HMAC-SHA256 endpoint/context handshake does not protect subsequent
TCP protocol messages; IKNP rather than silent OT, linear-depth ripple, and
CPU execution remain explicit limits. The live forward-FC path now consumes
this API, and the security contract contains its exact hybrid simulator.

## Catch up in 10 minutes (read in this order)

1. This file.
2. `results/reports/publication_readiness_plan_2026_07_21.md` — binding
   S1--S10 execution order, gates, proof/evaluation requirements, and
   per-stage commit discipline.
3. `results/reports/dealerless_orca_fc_security_contract_2026_07_29.md` —
   S1 functionality, exact DPF/FC transcript, leakage, simulators, and proof
   obligations.
4. `results/reports/s2_parameter_novelty_provenance_audit_2026_07_29.md` and
   `s2_professor_decision_request_2026_07_29.md` — S2 hard stops, current prior
   art, attribution, parameter transcript, and eight advisor decisions.
5. `results/reports/distributed_dpf_keygen_memo_2026_07_21.md` — corrected
   Phase C protocol, executable controls, and regenerated D1 counts.
6. `results/README.md` — where every result/report lives and what produces it.
7. `results/reports/s2_architecture_comparison_2026_07_29.md` — measured
   encoder comparison, dealerless-setup status, artifact defects, and decision
   table.
8. `results/reports/orca_fc_real_ole_transcript_memo.md` — real-OLE
   slot-packed transcript and NTT backend changes.
9. `results/reports/dealerless_orca_ringlpn_proposal_v2_17_2026_08_17.{tex,pdf}` —
   current internal/advisor source with the September review supplement and
   matching deterministic 34-page pinned two-pass PDF (SHA-256
   `dd1de27a496a3be4251e60f813f0a82ea658612e47ac2624d0859b99f99dffc1`).
   Rebuild, visually inspect every page, and refresh its digest/manifest after
   evidence or text changes.
10. `results/reports/baseline_2026_06_10.md` — historical full-GPU
    environment, PASS counts, and performance anchors.

Then re-validate everything with one command (~15 min, needs GPU):

```bash
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 PATH=/usr/local/cuda/bin:$PATH \
  scripts/run_paper_checkpoint_smoke.sh
# must exit 0 and print "[paper-smoke] ALL GATES PASS"
```

Historical 2026-08-09 prefix regression: the separate prefix gate passed at
that checkpoint. It is superseded by the final 2026-08-10 same-worktree
canonical run, which regenerated all 21 records and the complete graph; the
current canonical gate includes the full graph.

## Source map (`src/`)

| File | What it is |
|---|---|
| `bench_ntt_cuda_cheddar.cu` | The GPU NTT backend (substantially Cheddar-derived merged-stage kernels, signed Montgomery, q32/q64/q128-CRT, negacyclic). Upstream MIT notice and reconstructed source/blob pin plus local delta are retained in `extern/Cheddar_MIT_LICENSE.txt` and `extern/Cheddar_PROVENANCE.txt`; cite Cheddar. Included by every GPU bench via `RINGLPN_DISABLE_MAIN`. Contains `run_full_polymul`, `run_polymul_prepared_lhs`, adaptive fused-INTT (`RINGLPN_NTT_NO_FUSE`/`FORCE_FUSE`), `host_polymul_reference` (the host oracle), `kConfig62`/`kConfig62Crt2` (the primes: 2^62−6·2^24+1, 2^62−7·2^24+1). |
| `correlation_freshness.h` | **Live consume-once boundary (2026-08-04).** Fixed-width version/invocation/layer/kind/direction/limb/ring-batch/tree/phase/ordinal/conversion-chunk/output-slot encoding, SHA-256 IDs, compatibility-handle collision rejection, and owner-only immutable claim files written with no-replace creation, fsync, atomic rename, and directory fsync. Duplicate, pending, truncated, corrupt, retry, and restart state fails closed. |
| `gpu_spfss_zp.cuh` | GPU DPF/SPFSS with additive `Z_p` payloads. Its bounded breadth-first path expands nodes level-wise and reduces leaves without per-point root-to-leaf recomputation or atomics, with an exact fallback. The live FC caller is integrated: all five current focused q64/q128 cases/controls record positive P0/P1 breadth-call counts and zero root-to-leaf calls, and all 21 shape plans pass. These are correctness/path counters, not current Conv0 timing or a breadth-first speedup; the separate controlled CNN2/CNN3 matrix supplies model-FC timing. Expansion uses four domain-separated AES calls for full-width seeds/tags, with device/host parity gated. Centralized diagnostics remain correctness-only; the live path uses OpenSSL-private roots. |
| `ringlpn_ole_party.cuh` | **Party-local Figure-2 API (2026-08-06).** Own-party public parameters, noise/key validation and packing, GPU `x`/SPFSS/`z` expansion, and own `X`/`Z` slot shares. The live forward-FC runner calls it separately in each process and reserves exactly `3*c^2*t^2` output slots per instance for the next DPF Phase C; `bench_ole_ringlpn_cuda.cu` remains a both-party diagnostic. Ring-LPN pseudorandomness/parameters remain outside this functional API. |
| `bench_ole_ringlpn_party.cu` | Standalone one-party executable over one noise/key record, retained for focused diagnostics. The canonical live FC path calls the same party API in-process rather than exchanging intermediate slot files. |
| `bench_ole_ringlpn_cuda.cu` | Existing two-party-in-one-process Figure 2 Ring-LPN OLE engine/checker (random ring OLE: z0+z1 = x0·x1 in Z_p[X]/(X^n+1)); now consumes `ringlpn_ole_party.cuh`. It still owns both party states and retains `build_spfss_keys()` as its centralized-keygen fallback, while `RINGLPN_OLE_{NOISE,SPFSS_KEYS}` load the separately generated party records. |
| `bench_linear_ole_ringlpn_cuda.cu` | Ring-polynomial matrix Beaver from two OLEs per ring product. |
| `bench_vole_ringlpn.cu` | Older standalone VOLE expansion prototype. |
| `orca_fc_ringlpn_keywriter.cuh` | Host helpers + dealer/oracle keywriter used by `nn/orca/fc_layer.cu` behind `ORCA_RINGLPN_FC_KEYS` (bw≤32; baseline Orca byte-identical with flag off). Has `exactZmToRingShares` (conversion oracle), CRT/q128 helpers, and a clear value-dependent `dot >= Q` abort; the target replaces that predicate with the public admissibility check `K*2^(2*bw+2)<Q`. |
| `orca_fc_ideal_ole_transcript.cuh` + `bench_orca_fc_ideal_ole_transcript.cu` | Step-1 artifact: dealerless FC transcript with an *ideal* OLE oracle. Kept as reference; superseded by the real-OLE transcript. |
| `bench_orca_fc_real_ole_transcript.cu` | Historical single-process real-generator diagnostic: Ring-LPN expansion, slot packing, per-slot derandomization, Garner lift, clear exact conversion, key write, and unchanged consumer. It retains centralized DPF keygen and O1/O2 boundaries; do not compare its narrow stage timers to live end-to-end results. |
| `bench_orca_fc_ringlpn_demo.cu` | Byte-compatibility demo: forward + dW + dX key contracts at q64/q128. |
| `secure_convert.{h,cpp}` | **Party-local exact conversion API (2026-08-04).** `secure_convert_batch` validates canonical shares and common preflight, generates OT-backed daBits/edaBits and Boolean triples, and reports split transcript counters. The live forward-FC path calls it; the security contract supplies the exact hybrid simulator. |
| `test_secure_convert.cpp` | Standalone two-process conversion harness/checker for exact boundaries, random/forced-wrap and invalid/corrupted cases, bounded bilateral best-effort records, and split counters. The mutual endpoint/context handshake precedes protocol traffic, but later plain-TCP messages have no per-message integrity; SCI/IKNP, linear-depth ripple, and CPU execution remain limits. |
| `test_distributed_dpf_keygen.cpp` | **Corrected M1 host protocol-logic prototype (2026-08-06).** Two-party DPF keygen: secure adder for α's bits (L−1 bit triples), cancellation-lemma level walk (2 string OTs/level), and Phase-C arithmetic-share multiplication (3 scalar OLEs) that opens only standard `finalCW`. Six invalid-input controls, five independent key corruptions, omniscient old-sign regression, per-phase logical/meaningful-share accounting, ideal-mask accounting, and consume-once correlation-ID control pass. Standard keys validate through unchanged `dpfEvalAll`; ideal primitives and splitmix64 make this functional, not computational-security, evidence. |
| `two_party_ot.h` | **Real two-party transport.** SCI/IKNP plus an opt-in, pinned but independently unreviewed EMP-Silent adapter; Gilboa field multiplication is built from the selected OT backend. Each socket mutually authenticates process roles and the invocation/claim/direction context with HMAC-SHA256 and fresh nonces before preflight/OT. Local evidence then uses plain loopback with no application-layer per-message MAC. The unexecuted distinct-host launcher instead carries both complete post-handshake streams through one pinned-host-key, AEAD-only OpenSSH tunnel with bounded rekeying; malicious authenticated endpoints, denial of service, and side channels remain out of scope. The default SCI path overlaps independent straight/reversed Phase-A/B OT batches on disjoint contexts/sockets. `BatchPhaseCOleSource` permits external OLE only at epoch zero and otherwise consumes uniquely reserved prior Ring-OLE shares. Protocol randomness uses buffered `RAND_priv_bytes`; fixed-seed `mt19937_64` is test-only. |
| `emp_silent_{adapter.h,bridge.{h,cpp}}`, `test_emp_silent_loopback.cpp`, `verify_emp_silent_fc_evidence.py` | **Opt-in EMP-Silent route, measured 2026-08-14.** Exact sealed bridge bytes and pinned dependency revisions are enforced; packed 1/62/128-bit chosen-message differential/control tests pass. The current five-case FC rerun plus sixteen controls consumes every declared straight/reversed inventory and records correlation/adjustment/ciphertext bytes. This is correctness and accounting evidence only: one same-host trial per case ran under uncontrolled GPU occupancy, the custom SilentFerret revision remains independently unreviewed, base-OT subaccounting is unavailable, and no performance, bandwidth-improvement, or security claim follows. |
| `two_party_dpf_protocol.h`, `two_party_dpf_gpu.cuh` | Batched party-local DPF protocol and GPU implementation. Phase A/B use the selected real OT backend; Phase C accepts a batch OLE source. GPU tree-block reductions replace per-leaf atomics, retained asynchronous-pool frontier buffers remove repeated synchronous allocation/free, and full-width GPU-AES semantics remain unchanged; host/device parity and unchanged-evaluator gates cover the result. |
| `dpf_key_io.h` | Versioned little-endian `spfss_host::DPFKey` batch serialization (magic `RLPNDPF1`) plus the explicitly TEST-ONLY private-input record the offline checker needs. |
| `test_two_party_dpf_keygen.cpp` | **The two-PROCESS keygen artifact.** Same frozen protocol, but two OS processes over two TCP sockets with real OT/triples/OLE, each party writing only its own key file; gates the contract's closed forms in-process and reports measured wire bytes, direction switches, and setup cost. Primitive self-tests (`--selftest`) open triple/OLE shares in a labelled test-only mode. |
| `two_party_spfss.h` | **Party-local SPFSS API.** Validates/samples local noise, derives public work/group order, binds the full live Ring-OLE correlation-scope ID plus compatibility SID into the v3 public manifest, generates grouped DPF keys, and computes provenance digests. Standalone component baselines may retain an explicitly test-only zero scope; live FC/Conv may not. |
| `test_two_party_spfss_keygen.cpp` | Two-process CLI harness for `two_party_spfss.h`; each process samples or explicitly labels TEST-ONLY external noise, exchanges the manifest/validity state, emits per-party provenance/cost rows, and uses temp writes, a bilateral publishability exchange, and rename to publish only its own versioned noise/key records. This is bilateral best-effort, not crash-transactional. The focused q64 regular evidence is indexed above; the later OLE checker still reads both outputs. |
| `two_party_linear_preprocess.cuh`, `test_two_party_{fc,conv}_preprocess.cu` | **Canonical shared live forward-linear engine and thin public-shape adapters.** The engine claims a high-entropy invocation and exact correlation plan before OT/CSPRNG state, derives independently scoped SHAKE256 public vectors from one four-word joint seed per layer, binds full scope IDs into SPFSS/conversion, generates each limb's first Ring OLE from external Phase-C OLE and later ones from reserved prior output, then derandomizes/converts and publishes version-3 FC or Conv2D records. It rejects nonpositive `n-3*c^2*t^2` and accounts every slot. |
| `linear_preprocess.{h,backend.cuh}` | **Stable public material boundary.** `OwnedLayerMaterial` atomically opens one party's record plus bound mask state, validates exact kind/plan/party/SID/invocation/ordinal/digests/identity/counts/canonical words/input-share equality, exposes `A_i`, `B_i`, `C_i`, and `Y_i` as immutable views, and scrubs every private vector on failure, reset, move assignment, and destruction. |
| `orca_terminal_linear_backend.cuh`, `orca_linear_application_entry.cuh` | **Source-native terminal Orca path (2026-08-24).** One matching FC or Conv2D plus output only; bilateral validity/binding preflight, GPU masked-input/weight reconstruction, direct stock-key views, stock Beaver helper execution, and additive clear-output reconstruction. Every truncation/nonlinear/residual/second-linear/premature-output path rejects. |
| `test_linear_preprocess_api.cpp`, `test_orca_linear_helpers.cu` | Public record/state ABI and private-file rejection suite plus a two-process stock-key regression for the extracted OrcaBase matmul/Conv2D helpers. |
| `graph_mask_state.h` | Fixed-size, versioned, SHA-256-bound party-local linear mask-state record with exact invocation/layer/shape/record binding. The full-record runner publishes one companion state per party and layer. |
| `resnet18_graph_contract.h` | Compiled exact ResNet18 source contract: 21 linear specs, 21 truncations, 19 stock key items, 20 remasks, eight residuals, explicit main/shortcut value-source registries, four terminal state nodes, and 62 ordered stream items. |
| `stock_nonlinear_full_record.h`, `test_stock_nonlinear_full_keygen.cu` | **TEST-ONLY trusted full stock-key compatibility seam.** The adapter reads both source-bound mask states, invokes Orca's stock trusted-dealer MaxPool/ReLU/sign-extension key generation, and bilaterally publishes one digest-bound 62-item private record per party. It is not a distributed DCF protocol. |
| `test_resnet18_full_graph.cu` | Exact source-bound known-zero full ResNet18 runtime and independent checker. Live parties consume only their own records; the checker reconstructs every linear input/output, truncation, remask, residual, pool, sign-extension, and terminal value after both exit. |
| `stock_nonlinear_prefix_record.h`, `test_stock_nonlinear_prefix_keygen.cu`, `test_resnet18_graph_prefix.cu` | Superseded focused Conv0→TR→MaxPool→ReLU→Conv3 regression. Retained for narrow diagnostics; never cite it as current full-graph evidence. |
| `test_two_party_dpf_validate.cpp` | TEST-ONLY offline checker: runs after both parties exit, reads both key files, validates `beta*[x=alpha]` through unchanged `dpfEvalAll`, requires identical public material and differing seeds, and includes a corrupted-`finalCW` negative control. |
| `test_orca_zp_bridge.cpp` | Carry-corrected Z_p→Z_2^bw share export + the bw=32/q62 counterexample (negative control). |
| `bench_ntt_gpu_ntt_baseline.cu` | External baseline: GPU-NTT (Ozcan–Savas) vs cheddar, same prime/psi/operation. Needs external checkout (`GPU_NTT_HOME`, default `/home/fatih/GPU-NTT`); benchmark-only, not in the gate. |
| `spfss_host.{h,cpp}`, `test_spfss.cpp`, `bench_ole_ringlpn_host.cpp`, `verify_figure2_expand.cpp` | Host reference implementations + the 135/57/36 host validation suites. |
| `test_spfss_zp_cuda.cu` | GPU SPFSS payload correctness tests. |
| `orca_globals_stub.cpp` | Defines `OneGB` for standalone benches (instead of linking the comms stack). |

Upstream touches outside this directory are limited to:

- `GPU-MPC/backend/orca_base.h`: behavior-preserving extraction of protected
  stock matmul/Conv2D execution helpers;
- `GPU-MPC/experiments/orca/orca_inference.cu`: a macro-gated role-2 include
  and early return; macro-off roles 0/1 retain their source-bound shape anchors
  and build without the Ring-LPN archive; and
- `GPU-MPC/nn/orca/fc_layer.cu`: the older feature-flagged keygen integration
  (flag off = byte-identical baseline, verified through the two-party
  `tests/nn/orca/fc` test).

## Scripts (`scripts/`)

Every artifact has a `build_*.sh` / `run_*.sh` pair; runners write
`.csv` (data) + usually `.md` (summary) + `.log` (raw stdout+stderr) into
their directory under `results/` (see `results/README.md` for the mapping).
`run_paper_checkpoint_smoke.sh` is the canonical gate: host trio + bridge +
secure convert + GPU smokes (OLE q64/q128 × uniform/regular, linear, demo,
both transcripts), source/compiled graph-contract controls, the source-native
terminal Orca FC/Conv2D application gate, and the fresh 21-record full-graph
run. It ends with `ALL GATES PASS`.

## Validated claims vs. open boundaries

Safe to state (scoped to observed evidence):

- The corrected source is trusted-dealer-free in the stated random-oracle
  model, runs as two party processes on distinct GPUs, and produces party-local
  stock-format keys. Five current q64/q128 regular/uniform/multi-batch
  executions pass the unchanged `gpuMatmulBeaver` contract; sixteen focused
  endpoint/context-authentication, freshness, record, and capacity controls
  reject the intended rogue/replayed/reflected/wrong-secret, duplicate, restart,
  reuse, collision, truncation/corruption, mismatch, stale, capacity, rename,
  corrupt-record, and swapped-record cases.
- The public material API atomically binds one party's linear record to its
  companion mask state and fails closed on every tested metadata, digest,
  content, path-substitution, and permission mismatch. The source-native role-2
  path then produces correct nonzero terminal FC and Conv2D results through the
  real Sytorch module lifecycle and stock Beaver kernels. Eight application
  controls reject bilaterally without publishing a new output.
- The current ResNet18-classifier-layer artifact at `1x512x1000`, q128/bw32
  feasibility `(n,c,t)=(8192,2,8)` passes 10/10 measured trials after one
  warmup. Median post-channel setup-included preprocessing is 4.0193924415 s;
  application traffic is 182,372,344 bytes; shape/contract-matched stock dealer
  keygen is 14.73535 ms; and the unchanged online checker is 1.14969 ms.
  Channel construction/socket/auth time is excluded, so 4.0193924415 s is not
  true end-to-end. Physical-GPU/occupancy state was uncontrolled. Its retained
  binary begins `02eaaac9`.
- Self-sustaining slot packing gives
  `2*limbs*ceil(MKN/(n-3*c^2*t^2))` Ring-OLE instances. The retained classifier
  artifact uses 276 instances and 70,656 DPF trees per party; exact reserved,
  consumed, terminal-discarded, and application-discarded counts pass.
- The source-bound adaptive ResNet18 plan validates 21 isolated linear
  invocations and exact aggregate costs. A fresh run publishes all 21
  digest-bound record pairs and companion mask states, and each unchanged
  Conv2D/FC checker passes. The full-graph control then binds those records to
  the exact source-generated 62-item stream and executes all 21 secure
  truncations, all 19 unchanged stock key consumers (one MaxPool, 17 ReLUs,
  and classifier sign extension), 20 remask edges, three projection and five
  identity residuals, integer GlobalAvgPool2D, and terminal reconstruction.
  The independent checker reconstructs every registered value. Source/topology,
  publication, corrupt-key, and corrupt-output controls reject. The labelled
  TEST-ONLY trusted adapter reads both source mask states, supplies both
  truncation successor-mask shares and remask/terminal material, and generates
  exact stock nonlinear keys; this proves full graph/state composition, not
  dealerless nonlinear preprocessing, private inference, accuracy, or a
  full-model performance/security result.
  The orchestrator first applies each omniscient per-record correctness checker
  and then feeds the same test record to the graph control. This is deliberate
  functional evidence, not a consume-once online deployment trace.
- The standalone real DPF transport validates 369/369 host-reference pairs;
  the four-call full-width AES path matches 16 device vectors and 88
  two-process keys pass GPU evaluation. SCI/IKNP/Gilboa, private OpenSSL roots,
  bytes, direction switches, invalid-input and corruption controls are
  executable evidence.
- The ideal-functionality host reference validates 2,432 DPF pairs and exact
  transcript counts: `2*depth` string OTs, `depth-1` bit triples, three scalar
  OLEs, `2*(depth-1)+130*depth+ceil(log2(p))` logical opened bits, and twice
  that encoded share width.
- The exact DPF correction-word coupling, correlated-batch simulators,
  masked-difference OLE lemma, noncircular epoch induction, conversion
  simulator, and source map yield a conditional static-semi-honest theorem for
  forward FC under the security contract's assumptions. This is a proof
  boundary, not a concrete parameter/security claim.
- The canonical correlation tuple and append-only claim implementation close
  the executable source/proof `P-FRESH` boundary at the explicit SHA-256
  collision-resistance, trusted owner-only persistent filesystem, one
  deployment-wide ledger root, no-replace/fsync/atomic-rename semantics, and
  no storage cloning/rollback assumptions. All sixteen focused controls pass.
- Baseline Orca is byte-identical with the feature flag off.

NOT claimable (never blur these):

- Any concrete security level. The exact projected Ring-LPN
  distribution/structured-code/two-limb reduction is unreviewed and no
  parameter set is pinned.
- A secure network deployment. Local evidence authenticates endpoint/invocation
  context before preflight but then uses plain loopback without
  application-layer per-message integrity. The distinct-host launcher now
  tunnels both complete streams through pinned-host-key AEAD-only OpenSSH with
  bounded rekeying, but no second-host execution exists; the theorem assumes an
  authenticated channel and trusted endpoints.
- A current performance win. The controlled same-physical-GPU stock-dealer
  paired-ratio medians are `879.2970985824659×` for CNN2 FC4+FC5 and
  `60.3181599795375×` for CNN3 FC5: strongly negative. The retained classifier
  artifact's uncontrolled descriptive ratio is `268.6769431352700×`.
- Full-model dealer removal, arbitrary multi-layer record dispatch, dealerless
  nonlinear key generation, private or trained/accuracy ResNet18, a full-model
  performance/security result, stateful training, malicious security, WAN
  behavior, or side-channel resistance.
- Conference readiness. Parameter security, Phase-A/B performance, independent
  review of the silent-OT backend, a compatible dealerless baseline, two-host
  evidence, clean-clone reproduction, venue conversion, and independent human
  cryptographic review remain open.

## Prioritized next-agent runbook

Read this file and `results/README.md` first. The 2026-08-09 handoff and
prefix-era systems plan are historical. A component PASS never permits skipping
a claim gate, and functionality/claim decisions require owner consultation.
For a repository-wide review and version-control pass, use
`results/reports/chief_of_staff_handoff_2026_09_10.md`. It records the dirty
workstream split, audit sequence, commit protocol, and verification commands as
of 2026-09-10. Re-run Git/GPU state before acting; this file remains the
authority for technical status and claim boundaries.

1. **Preserve consumed and retained state.** Treat
   `results/fc/forward_linear_record_set_2026_08_07/` and its ledger as
   permanently consumed and incomplete. The original internal-only full-graph
   checkpoint and final manifest set remain under
   `results/graph/resnet18_full_graph_checkpoint_2026_08_10/`, bound by
   manifest digest
   `fdf51f25902afd94a1e67b8bdffa33f762d89c104836538d913c2d7e392c5395`.
   It contains 17 indexed payload files plus `INDEX.json`; private
   linear/nonlinear records and live ledgers are not publication artifacts.
   Its manifest-bound host-identifying provenance and historical GPU-1
   party/checker reuse make it internal-only. Do not sanitize those bytes in
   place; replace the checkpoint with a fresh normalized-provenance run before
   external circulation. Require the literal canonical `ALL GATES PASS`, then
   rebuild and inspect the PDF after evidence or manuscript changes.
2. **Keep the full-graph boundary exact.** The source-bound known-zero control
   executes the complete 21-linear/62-item graph and all residual/state
   transitions. Its TEST-ONLY trusted adapter sees both parties' mask states,
   supplies both truncation successor-mask shares and remask/terminal material,
   and generates the nonlinear keys. Never call it dealerless nonlinear
   preprocessing, private/trained inference, accuracy, or full-model
   performance/security evidence.
3. **Close the security and performance gates before broad claims.** Obtain an
   independently reviewed structured projected-code reduction and two-limb
   composition before pinning parameters. Reduce Phase B/rounds and remeasure
   the exact end-to-end path at any reviewed tuple. Do not reopen specialized
   DMPF without a source-reviewed fixed-transcript construction below the
   audited cost ceiling.
4. **Replace the remaining dealer only with a reviewed protocol.** The trusted
   stock-nonlinear adapter is the remaining full-graph dealer boundary. A
   dealerless DCF/DMPF replacement needs its own protocol, proof, source audit,
   exact stock ABI gate, and owner-approved contribution/credit decision.
5. **Finish external reproducibility.** Run repeated authenticated LAN and
   controlled-WAN trials on two real isolated hosts, then execute the pinned
   clean-clone/container publication mode and preserve its durable committed
   manifests. Keep Reverse Cuckoo/native-ring rows separately labelled because
   their fields, layouts, setup, and functionality differ.
6. **Handoff discipline.** Keep Alp `<fcetin@hawk.iit.edu>` as sole paper and
   final-history author. Never discard or overwrite unexpected workspace work,
   never reuse consumed ledgers, and do not send outreach or decide
   ownership/credit without the owner.

## Binding execution order (S1–S10 plan; proposal components D1–D5)

S1 froze the functionality/proof contract. S2 remains the hard
security/publication gate for exact Ring-LPN parameters, novelty, and
provenance. The owner explicitly lifted only the implementation-order
dependency, which allowed the feasibility D1–D4 forward path, conditional
proof, and exact classifier-layer evaluation to proceed without a
parameter/security
claim. Those branches are now executable and measured; they do not close S2.

The full source-bound record/graph composition branch is now complete at its
known-zero trusted-adapter boundary. The next order is: replace the trusted
nonlinear adapter with a reviewed dealerless protocol -> independent transport
and cryptographic review -> authenticated repeated two-host matched evaluation
-> pinned clean-clone submission candidate. Parameter/security work and
algorithmic Phase-B reduction proceed in parallel, but neither may borrow a
claim from the completed graph-composition control.
In parallel, a qualified human must review the
structured-code reduction/parameter and transcript arguments. The specialized
regular-DMPF audit remains a design NO-GO and may be reopened only under its
fixed-transcript simulator, capacity, stock-key, and cost gates.

GPU-batched DPF generation, executable Ring-OLE-output bootstrap,
dependency/memory instrumentation, a current opt-in EMP-Silent
correctness/accounting rerun, the pinned mathematical regular-projection
analysis with current-sampler binding and freshly regenerated hybrid evidence,
source-bound 21-layer planning, the fresh 21-record run, exact full-graph/state
composition, a controlled CNN2/CNN3 model-FC matrix, and one mismatched
primitive baseline are complete. The breadth-first FC caller is integrated and
the current default SCI/IKNP focused suite supplies correctness/path-counter
evidence: five q64/q128 cases/controls pass, every row has positive P0/P1
breadth calls and zero root-to-leaf calls, and all 21 shape plans pass. The
separate EMP-Silent rerun passes the same five cases and all sixteen controls
with exact directional inventory exhaustion and split backend byte counters.
Its one-trial, occupied-GPU measurements are not performance or
bandwidth-comparison evidence, and its custom pinned backend remains
independently unreviewed. No current Conv0 timing or breadth speedup is claimed.
The controlled current model-FC matrix covers CNN2 FC4/FC5 and CNN3 FC5; all
30 measured layer trials pass, and the same-physical-GPU stock-dealer
comparisons are strongly negative. None closes the parameter,
dealerless-nonlinear, deployment, private-model, or publication gates. Current
D1 functionality uses real two-process SCI/IKNP or opt-in EMP-Silent OT,
external epoch-zero Gilboa OLE, PCG-supplied later Phase-C correlations,
OpenSSL-private roots, full-width GPU-AES-compatible keys, and measured
bytes/dependency layers/memory. Remaining performance work includes algorithmic
Phase B, Ring-LPN expansion, conversion, reruns at reviewed parameters, and a
compatible dealerless baseline; optional-backend cryptographic review also
remains open. Components remain D1--D5 in the v2.17 report.

## Perf anchors (RTX 5000 Ada, this repo's gate configs)

| Metric | Value |
|---|---|
| OLE expand, n=8192 c=2 t=8 regular | 9.086 ms (q64) / 18.230 ms (q128); one iteration per configuration (`n=1` each), diagnostic only |
| OLE expand, t=64 | 881 ms uniform / 61 ms regular (q64); two timed iterations per mode (`n=2` each; four total); distributions/SPFSS domains differ, so no ratio; diagnostic only |
| Linear OLE-to-Beaver 2×2×2 regular | 143.686 ms (q64) / 289.511 ms (q128); one iteration per configuration (`n=1` each), diagnostic only |
| Cheddar polymul n=8192 batch=64 | ~255–265 µs (q64) |
| Controlled CNN2 all-FC setup-included preprocess | mean 26.8818423365 s; SD 0.05096530898 s; median 26.89771076 s; R-7 IQR 0.06575381975 s; t95 [26.8453839508, 26.9183007222] s (`n=10` aggregate model trials; FC4+FC5) |
| Controlled CNN3 FC5 setup-included preprocess | mean 0.903894393 s; SD 0.01897115914 s; median 0.904909782 s; R-7 IQR 0.02851206825 s; t95 [0.8903232433, 0.9174655427] s (`n=10`) |
| Controlled stock dealer / paired ratio medians | CNN2 all-FC: 30.5729 ms / 879.2970985824659×; CNN3 FC5: 14.96295 ms / 60.3181599795375×; same physical GPU and quiescence checked per sample, fixed protocol-then-dealer order, strongly negative |
| Controlled CNN application / total transport bytes | CNN2 all-FC: 1,302,752,736 / 1,302,840,696; CNN3 FC5: 39,432,504 / 39,476,484 |
| Controlled CNN semantic dependency layers | CNN2 all-FC: 72,278; CNN3 FC5: 1,663; implementation schedule depth, not packet/network rounds |
| Retained ResNet18 classifier post-channel setup-included preprocess | mean 4.011203588 s; SD 0.036601967373448 s; median 4.0193924415 s; R-7 IQR 0.03481908225 s; t95 [3.985020117867291, 4.037387058132709] s (`n=10`; uncontrolled physical GPU/occupancy) |
| Setup-included timing boundary | Includes per-layer preflight + OT setup + legacy total; excludes `PartyChannel` construction, socket establishment, and authentication; never true end-to-end |
| Retained classifier application / total transport bytes | 182,372,344 / 182,416,324 |
| Retained classifier shape/contract-matched stock dealer / unchanged online | 14.73535 ms / 1.14969 ms median; uncontrolled GPU occupancy and not the same physical GPU |
| Retained classifier setup-included preprocessing/dealer ratio | 268.6769431352700× median of shape-matched trial ratios; descriptive, not same-hardware A/B |
| Retained classifier dominant legacy post-OT stage | Phase B: 1.955515 s median; Phase C: 0.041723 s median |
| Retained classifier protocol/memory counters | 11,023 dependency layers; 142,542,848 median host peak bytes; device-wide GPU peak-used metric 31,929,597,952 bytes |
| Retained older-binary maximum Conv0 row | 358.085 s legacy post-OT; binary `1db001...`, not current Conv binary `6a9ae142...`; no current-source breadth speedup claim |
| Retained older-binary Conv0 comparison row | 384.816 s legacy post-OT; one trial, not a current performance result |

NTT decision (measured, `reports/ntt_baseline_comparison_2026_06_10.md`):
keep cheddar — external GPU-NTT merge is 1.2–3.9× faster but cannot run
62-bit primes (Barrett headroom). Revisit at M5 if primes are re-pinned to
the ≤60-bit class.

## Environment & gotchas (will bite you)

- **This is a shared school server; the user is NOT a sudoer.** Never attempt
  privileged operations. The user is in the docker group (root-equivalent in
  principle) — use it only for ephemeral containers and for chown-ing the
  user's OWN files under `/home/fatih`; never touch system paths or other
  users' files. Check `nvidia-smi` before heavy GPU runs and prefer pinning
  with `CUDA_VISIBLE_DEVICES` — others may be working.
- `nvcc` is at `/usr/local/cuda/bin` — NOT in default PATH. `GPU_ARCH=89`
  (4× RTX 5000 Ada). Two-party tests: run party 0 and 1 with
  `CUDA_VISIBLE_DEVICES=0/1`, args `<party> 127.0.0.1`.
- Linux ephemeral client ports are `32768-60999` on the validated hosts.
  Local runner defaults deliberately stay in `20400-29761`; do not override a
  loopback base into the host's `/proc/sys/net/ipv4/ip_local_port_range`.
  A full-graph run at base `57400` reached order 16 and then lost a server bind
  to an unrelated ephemeral socket; that is an environment failure and the
  consume-once run must restart fresh, never resume partial records.
- Source-native role 2 inherits `GpuPeer`'s fixed TCP port 42003, which lies in
  the host ephemeral range. `run_orca_linear_application.py` serializes helper,
  success, and control pairs; never run two instances concurrently, and treat
  an unrelated bind collision as an environment failure.
- Historical builds ran as root in docker (`pcg-accel:*` images; user is in
  the docker group, sudo needs a password). Root-owned files can reappear;
  fix: `docker run --rm -v <dir>:/x ubuntu:22.04 chown -R 1013:1014 /x/...`.
- `bench_ntt` (CPU NFLlib) needs libmpfr-dev — absent on host and in the
  images; build in a container with an ephemeral `apt-get install libmpfr-dev`.
- The local `ringlpn-repro:2026-08-10` tag contains TeX Live 2023, but it is
  not publication provenance: the tracked environment manifest authorizes the
  digest-pinned base image and toolchain, while the final built runtime-image
  digest is still unpublished. Use the local tag only for internal rebuilds;
  publication mode must remain blocked until a tracked manifest authorizes the
  final runtime digest. For internal TeX rebuilds, mount the repository
  read-write as `/work/EzPC` and run the documented two-pass `pdflatex`
  command in `/work/EzPC/GPU-MPC/ringlpn/results/reports`. The image runs as
  the repository owner, so generated files remain host-owned; remove
  `.aux/.log/.out` files after checking the final log. `/bin/sh` in
  Debian-derived containers is dash: no `{a,b}` brace expansion.
- Focused Ring-LPN GPU executables must allocate their explicit buffers only;
  do not call upstream Orca's eager 25-GiB `initGPUMemPool()`. The shared
  FC/Conv engine instead configures the default asynchronous pool without an
  eager reserve. This is a shared-GPU feasibility requirement, not a protocol
  or security optimization.
- Root `.gitignore` ignores `*.csv` globally — committing new result CSVs
  requires `git add -f`.
- `extern/NFLlib` is a registered submodule (quarkslab/NFLlib @ 5cf40ed);
  the mnist/weights data submodules carry deliberate internal renames — leave.
- Env flags: `ORCA_RINGLPN_LINEAR_INTEGRATION` for the source-native terminal
  role-2 build; `ORCA_RINGLPN_FC_KEYS` (+`_QBITS`,`_SEED`) for the historical
  `fc_layer` path; `RINGLPN_NTT_NO_FUSE` / `RINGLPN_NTT_FORCE_FUSE` for
  polymul A/B; `SMOKE=1 QBITS=... NOISE=...` for the sweep runners.

## House rules for new work

1. Every new artifact = source + build script + run script + CSV/MD/log in
   its `results/` subdir + a memo in `results/reports/` + a gate hook if it
   guards a claim. Suites exit non-zero on any failure.
2. Validate against an independent oracle (host reference or unchanged Orca
   online path), not against the code under test.
3. State oracle boundaries in the source header and the memo. Keep the
   "safe to claim / not yet claimable" split current.
4. Don't claim perf wins without an A/B at the consumer's actual shape
   (cf. the fused-INTT adaptive threshold story).
5. **Commit every stage:** a stage is complete only after its mechanical gate
   passes, evidence and current docs are synchronized, and an atomic checkpoint
   commit is created. Preserve completed gate commits; corrections get new
   commits and rerun affected gates. See the publication-readiness plan.

## Documentation contract — BINDING for every agent working here

This file is the single source of truth for project state. Stale documentation
is worse than no documentation: it corrupts the context of whoever reads it
next. Therefore, **before ending any session that changed code, results, or
plans, you must**:

1. **Update this file** — the Status line, the source map (if files were
   added/moved), the claims split (if a boundary moved), the perf anchors
   (if regenerated), and the gotchas (if you hit a new one).
2. **Update `results/README.md`** with a row for any new artifact or report.
3. **Write or refresh the memo** in `results/reports/` for the artifact you
   touched, dated, with reproduction commands.
4. **Mark superseded documents**, never delete-without-trace and never leave
   them unmarked: prepend the standard banner
   (`> **HISTORICAL ...** superseded by CLAUDE.md; statements below may
   describe an older state`) the moment a document's claims stop being
   current. All pre-2026-06-10 reports already carry it; keep the convention.
5. **Never let two live documents disagree.** If you find a contradiction,
   the newer gate-verified statement wins; banner or fix the other on the
   spot, in the same commit.
6. Historical/outreach/archive documents are read-only context: quote them,
   don't trust them. Anything without a banner and dated 2026-06-10 or later,
   plus the gate output, is current; everything else is history.
