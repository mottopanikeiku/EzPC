# Dealerless Orca linear-layer preprocessing — publication readiness plan

**Date:** 2026-07-21  
**Scope:** two-party, semi-honest, dealerless preprocessing for one Orca **forward-FC matmul** from splittable Ring-LPN. Stateful training transitions, nonlinear-layer FSS keys, malicious security, and full-model dealer removal remain out of scope.
**Starting checkpoint:** commit `28f8451` (`ringlpn: add corrected distributed DPF keygen artifact`), with the required GPU gate ending `ALL GATES PASS`.

## Direction decision — 2026-07-29

- **Primary thesis candidate:** an integrated dealerless Orca FC-preprocessing
  system. The corrected per-point DPF is a compatibility artifact/baseline,
  not the candidate protocol contribution, unless advisor review identifies a
  concrete delta from distributed DPF/DMPF prior art. End-to-end two-party GPU
  integration and evaluation must establish the systems contribution. No
  novelty claim is unlocked while S2 remains open.
- **Contribution boundary:** the GPU PCG system is separate work with its own
  forthcoming paper and PIM-architecture comparison. Contributor ownership,
  credit, chronology, and permission to reuse its DPF/GPU code are not yet
  resolved; ask the professor before consuming that implementation. This
  paper's sole author is Alp by user direction, but that does not transfer
  ownership or erase attribution for inherited work. The paper must not
  present GPU PCG design/performance as new. S2 must record the professor's
  reuse/credit decisions and a related-work/overlap table before S3 imports or
  implements overlapping code; external circulation remains blocked until
  then.
- **Publication path:** first produce an advisor-ready technical report at the
  full technical bar in this plan. Lock a venue and its formatting only after
  advisor feedback; do not lower the proof, real-transport, or evaluation gates
  merely because the first deliverable is a report.
- **Code boundary:** work inside `GPU-MPC/ringlpn/` by default. Plan future Orca
  integration explicitly. Any new upstream Orca edit must be minimal,
  feature-flagged where possible, and presented for user review before editing.
  A vetted external cryptographic dependency also requires a design/license
  review before adoption.
- **Consultation:** stop before every S1--S10 stage with current evidence,
  proposed design, alternatives, risks, gate, and intended checkpoint commit.
  Also stop immediately if a security assumption fails, the contribution
  boundary changes, or a stage would require broader upstream modification.


## Execution update — 2026-08-04

- The executable boundary moved materially: the live runner now composes
  party-local noise, real SCI/IKNP/Gilboa distributed DPF key generation,
  full-width GPU-AES Ring-LPN expansion, exact conversion, transactional
  party-local key records, and the unchanged Orca forward-FC consumer across
  two OS processes and distinct GPUs. Six q64/q128 regular/uniform/multi-batch
  cases and their negative controls pass.
- The proof boundary also moved: the current paper contains an exact
  correction-word coupling, complete role-specific correlated-batch
  simulators, an exact conversion simulator, and a conditional forward-FC
  theorem. Independent reviews found no critical proof or live-composition
  defect after the recorded fixes. This is conditional security reasoning,
  not a concrete-security claim.
- One warmup plus ten measured exact ResNet18 classifier-layer trials pass.
  Median preprocessing is 35.735 s and sends 608,860,824 application bytes,
  versus 10.813 ms for matched stock trusted-dealer key generation. This is
  negative performance evidence, not a speedup or a full ResNet18
  inference/truncation measurement.
- S2 remains the hard blocker: no reviewed Ring-LPN parameter pin exists.
  Authentication, two-host/WAN measurement, peak-memory/round instrumentation,
  a compatible closest dealerless baseline, and all-forward-linear-layer model
  coverage also remain open. The work is advisor-ready but not
  conference-submission-ready.
- The selected public-coin design changed from a short commit--open seed to
  exchanging full uniform field-coefficient shares. This preserves the exact
  required public-polynomial distribution; revealing a short PRG seed would
  require a separate computational reduction.

## 1. Definition of publication-ready

The project is publication-ready only when all of the following are true at the same committed revision:

1. **Protocol:** the publication path contains no centralized SPFSS-keygen oracle and no dealer-labelled share-conversion correlations. Ideal implementations remain only as independent test references.
2. **Security:** the paper states a precise functionality and two-party semi-honest theorem, gives simulators for corruption of either party, accounts for every opening, and reduces privacy to named assumptions: AES, the chosen OT/OLE implementation, authenticated channels, exact public-coin sampling, and the pinned splittable Ring-LPN parameters.
3. **Parameters:** a reproducible audit supports the claimed security level for the exact noise distribution and fully split ring used by the implementation. Smoke parameters are never presented as secure parameters.
4. **Implementation:** two OS processes on separate GPUs generate the FC preprocessing, communicate only through the declared transport, serialize byte-compatible keys, and validate through unchanged `gpuMatmulBeaver`.
5. **Evaluation:** trusted-dealer, oracle-backed, fully protocol-backed, and a
   closest dealerless baseline are separated. Time, communication, rounds, key
   bytes, and memory are measured at the exact pinned parameters and at
   model-scale FC shapes.
6. **Artifact:** a clean checkout can regenerate every claim, table, and figure with pinned dependencies and nonzero-on-failure gates.
7. **Paper/provenance:** the formal protocol delta, closest prior art, source
   ownership/license, overlap disclosures, contributor credit/reuse decisions,
   and sole-author boundary are recorded before implementation; the final
   venue-formatted paper, appendix, and artifact documentation survive
   cryptographic, systems, novelty, and reproducibility review without an
   unresolved claim/evidence mismatch.

Passing a component unit test, producing a GPU kernel, or preserving the current 2,432 host passes does not alone satisfy this definition.

## 2. Stage and commit discipline

The user requires a commit at every stage. This is a binding execution rule for the roadmap below.

- A stage is complete only after its mechanical gate passes and its evidence has been regenerated.
- Each completed stage ends in an atomic checkpoint commit. Use the prefix shown below, for example `ringlpn(m5): pin audited splittable parameters`.
- A large stage may use smaller reviewable commits, but it still requires one final gate/evidence commit before the next dependent stage starts.
- Do not amend or squash away a completed gate checkpoint. A later correction gets its own commit and reruns every affected gate.
- Never mix unrelated dependency/submodule state, scratch files, or build products into a Ring-LPN checkpoint.
- Generated CSVs and PDFs are ignored at the repository root; force-add only the canonical evidence explicitly named by the stage.
- Every stage memo records: checkpoint subject, exact commands, hardware/software manifest, pass/fail result, raw-artifact paths, claim unlocked, and claims still blocked. The next status/handoff update records the resulting commit hash without rewriting history.
- Update `CLAUDE.md`, `results/README.md`, the current paper, and the current handoff in the **same stage commit** whenever status, measurements, or claim boundaries move.
- No stage advances while Ring-LPN work from the preceding stage remains uncommitted.

## 3. Dependency order

```text
S1 protocol/proof contract ─> S2 M5 parameters + provenance/novelty pre-gate ─┬─> S3 M1 GPU core ─> S4 M1 real transports ─> S5 M2 integration ─┐
                                                                            └─> S6 M3 protocol-backed conversion ────────────────────────┤
                                                                                                                                          └─> S7 M4 stateful two-process composition
                                                                                                                                                       ↓
                                                                                                  S8 proof + implementation audit ─> S9 M6 evaluation ─> S10 release
```

S2's parameter and provenance workstreams remain hard preconditions for any
security, parameter, performance-at-pinned-parameters, or publication claim.
On 2026-07-29 the owner lifted only the implementation-order gate: S3--S6 may
produce explicitly no-security-claim artifacts in parallel while S2 remains
open. Each branch preserves its own checkpoint commits and gates. S7 is the
first point at which their protocol-backed paths compose.

## 4. Execution stages

### S0 — Corrected host checkpoint and truthful proposal — **complete**

**Commit:** `28f8451`  
**Evidence:** corrected Phase C, old-sign leakage regression, 2,432/2,432 host key pairs, regenerated proposal v2.2, canonical host gate, required GPU gate, warning-free 15-page PDF build and rendered review.

**Claim unlocked:** protocol-logic and host-format functional compatibility using ideal OT/triple/OLE and a non-cryptographic correctness PRG. Nothing stronger.

---

### S1 — Freeze the protocol, functionality, and proof obligations — **complete 2026-07-29**

**Purpose:** prevent implementation choices from outrunning the security argument.

**Work:**

1. Specify the ideal FC-preprocessing functionality: party inputs, public
   dimensions/parameters, party outputs, allowed aborts, exact public
   transcript, and the source-aligned forward/bias/truncation/`dW`/`dX`/
   bias-gradient/weight-and-bias optimizer mask topology.
2. Specify D1 at message level: arithmetic-share adder, level walk, control bits, seed correction words, three-OLE Phase C, and the sole Phase C opening `finalCW`.
3. Specify D2–D4 composition: conversion correlations, exact full-vector
   public-coin exchange, private CSPRNG streams, derandomization openings, key serialization, and two-process transport.
4. Enumerate leakage explicitly: dimensions, parameters, batch sizes, message
   lengths and schedule, public polynomial, DPF correction words, conversion
   openings, key topology, and abort stage. State the excluded packet/timing/
   microarchitectural and active-adversary leakage models.
5. Draft simulators for corrupted party 0 and party 1. In particular, prove that Phase C can simulate each party's OLE shares and the public `finalCW` without revealing the hidden sign, the other payload factor, or the secret point.
6. State all composition assumptions and theorem scope. Separate computational privacy from Ring-LPN pseudorandomness and correctness.
7. Add proof-driven tests for boundary inputs permitted by the functionality:
   point edges; boundary factors `1,p-1`; deterministic product edges;
   randomized legal nonzero factors; zero/noncanonical rejection; wrap/no-wrap
   conversion boundaries; invalid inputs and corrupted-key negative controls;
   transcript length invariants; and no ideal-correlation reuse. The protocol
   proof, not exhaustive testing over `Z_p^*`, covers all legal factors.

**Gate:**

- A protocol transcript table accounts for every sent/opened value.
- Every implementation message maps to one line of the protocol specification.
- Both corruption simulators are complete at the hybrid-model level; no sentence relies on “the sign is random” without conditioning on a party's state.
- Advisor/cryptography review has no unresolved correctness or privacy blocker.

**Evidence:** a proof/specification section in the current paper plus a dated review memo and deterministic transcript tests.

**Checkpoint commit:** `ringlpn(proof): freeze semi-honest protocol and leakage contract`

**Claim unlocked:** none; this is the contract subsequent implementations must meet.

---

### S2 — M5 claim gate: prove and pin splittable Ring-LPN parameters

**Status 2026-08-04:** blocked. The exact primary-source audit found that
BCG+20 Sections 8.2 and 9.1 use different projected-weight formulas and that
Table 1 conflicts with the literal smallest-factor criterion. It also found
that selected local estimator outputs used out-of-range binomial inputs; the
remaining finite-field model values have no reviewed reduction from the
deployed block-conditioned/projected distribution or structured code. No
parameter or 128-bit classical/quantum claim is pinned. Direct 2026 fully
distributed DMPF prior art, alternative PCGs, and private-project
ownership/overlap also remain claim gates. See
`s2_parameter_novelty_provenance_audit_2026_07_29.md`,
`../security/README.md`, and `s2_professor_decision_request_2026_07_29.md`.
The owner permits implementation-only S3--S6 work; none of it lifts S2.

**Purpose:** establish a reviewed security reduction and parameter set before
security, parameter-dependent headline performance, or publication claims.

**Work:**

1. Obtain an author clarification/erratum or independently reviewed lemma that
   resolves BCG+20 Section 8.2 versus Section 9.1 versus Table 1.
2. Derive the projected distribution for the actual block-exact and regular
   samplers over both primes, including coefficient cancellation, a lower-tail
   bound, rounding, and a justified useful-factor criterion.
3. Establish applicability to the fully split quasi-cyclic code, account for
   DOOM/structured-code attacks, and compose distinguishing advantage across
   both CRT limbs, every factor, epoch, and PCG hybrid. State classical and
   quantum scope separately.
4. Only after those proofs, run fail-closed, reproducible estimators over
   candidate `(n,c,t,p0,p1)` sets. Reject every out-of-domain formula input and
   include sensitivity around the chosen point rather than one optimistic row.
5. Recompute the complete epoch budget, not only `3c^2t^2<n`: reserve scalar OLEs for three OLEs per DPF tree, epoch-zero Gilboa bootstrap, conversion correlations, safety margin, and any rejected/unused slots. Prove no slot is reused.
6. Recheck NTT feasibility, prime bit width, `v2(p_i-1)`, CRT correctness, GPU memory, and batch size at the pinned `n`. If primes move to at most 60 bits, reopen the documented cheddar-versus-GPU-NTT decision; if `n` grows, rerun every polynomial and slot-packing gate.
7. Generate machine-readable parameter tables and estimator transcripts from a pinned script/container.
8. In parallel, compare the exact shared-point/multiplicative-payload,
   three-OLE per-point construction against Doerner--shelat, Programmable DPF,
   the 2025 improved DMPF, the 2026 fully distributed DMPF for PCGs, and
   SLAMP-FSS. Compare regular Ring-LPN against Stationary Syndrome Decoding,
   and Ring-LPN/NTT plus conversion against the 2025 direct-`Z_(2^k)` PCG and
   2026 QA-SD/WHT prime-field PCG.
   Record formal deltas, assumptions, asymptotics, implementation availability,
   and what is systems integration rather than protocol novelty.
9. Audit active and candidate source provenance/licenses: local
   cheddar-derived NTT, GPU-NTT baseline, and the separate GPU-PCG/PIM work.
   Obtain the professor's ownership, contributor-credit, chronology,
   reuse-permission, citation, and overlap/disclosure decisions before
   importing overlapping code or externally circulating the paper. Retain Alp
   as the sole paper author as directed by the user.



**Gate:**

- A reviewed projection/distribution/tail/structured-code and two-limb
  advantage reduction supports the claimed level for the exact implemented
  distribution; every estimator call is mechanically in-domain and
  independently reproducible.
- The full bootstrap/consumption budget is positive with a documented margin.
- Both CRT limbs and the required NTT size pass host-reference and GPU tests.
- The formal novelty/overlap table and source/license inventory are complete;
  the professor's required provenance decisions are recorded; no blocked code
  or measurement is imported.

**Hard stops/reversals:**

- No concrete security claim without the reviewed reduction and fail-closed estimator evidence above.
- Grow `n` or change the distribution if the bootstrap budget fails; do not weaken the inequality.
- Re-pin primes and reopen the NTT backend if the selected `n` exceeds current roots-of-unity headroom.
- If provenance or closest prior art invalidates the proposed contribution,
  revise the thesis/protocol before S3; do not defer the problem to submission.

**Evidence:** estimator code, versioned raw transcripts, parameter CSV/MD/log,
security memo, formal protocol-delta/prior-art table, source/license inventory,
recorded professor decisions, and updated paper tables.

**Checkpoint commit:** `ringlpn(m5): pin parameters and contribution boundary`

**Claims unlocked:** concrete security level for the pinned parameter set and
the reviewed contribution/provenance boundary; no end-to-end security claim.

---

### S3 — M1a: cryptographic, GPU-consumable distributed DPF core

**Status 2026-08-04:** functional component complete for the live feasibility
path: full-width GPU-AES-compatible keys, private roots, versioned
serialization, host/GPU evaluation, and corrupted-key controls pass. GPU-side
batched key generation and concrete DPF/PRG security review remain open.


**Route dependency:** the owner selected the per-point DPF as the current
compatibility baseline for M1/M2 implementation, with no novelty or security
claim. DMPF remains future work unless advisor review changes that route.

**Purpose:** replace the splitmix64 host semantics with full-128-bit AES/GPU
key semantics while retaining ideal transports temporarily.

**Primary targets:** `src/test_distributed_dpf_keygen.cpp` as the independent
host reference, `src/gpu_spfss_zp.cuh` with a stable API/callsite but corrected
full-entropy evaluator semantics, and a production distributed-keygen module
under `src/` rather than code embedded in a benchmark.

**Work:**

1. **Partially closed 2026-08-03:** the deployed Ring-LPN expansion now uses
   four domain-separated AES calls: full 128-bit child seeds from plaintexts
   0/2 and separate control tags from 1/3, matching the formal BGI seed/tag
   separation (Boyle--Gilboa--Ishai, CCS 2016, DOI
   `10.1145/2976749.2978429`). Device/host parity, centralized correctness,
   two-process generation, and GPU evaluation are gated. Two-party roots use
   OpenSSL's private DRBG. Still open: the centralized benchmark keygen derives
   roots from one 64-bit `seed_base`; replace that benchmark-only root path
   before treating it as a security realization. The DPF distribution and
   single-key privacy reductions also remain S3/S8 obligations. splitmix64 is
   confined to the labelled host correctness reference.
2. Emit `GPUDPFZpKey`-compatible party keys: seeds, seed correction words,
   separate control correction bits, and final correction words. Define a
   versioned byte serialization with explicit endianness and bounds.
3. GPU-batch every tree level synchronously. Preserve party-owned buffers; no
   kernel or host helper may read both parties' private state outside a named
   ideal transport used at this stage.


4. Implement the known one-string-OT-per-level formulation or retain two only
   if the proof and measured cost justify it. The paper and counters must match
   the implementation exactly.
5. Cross-check the full-128-bit AES implementation against fixed known-answer
   vectors, a separately written CPU AES evaluator, and the pre-change
   functionality on fixtures where the former cleared bit is zero.
6. Test both values of every root-seed LSB, q64/q128 limbs, pinned depths,
   batches from 1 through at least 256, edge points, maximum legal point,
   deterministic replay, malformed serialization, and corrupted correction
   words.

**Gate:**

- For at least 256 trees per pinned configuration, GPU full-domain evaluation
  reconstructs `beta [x=alpha]` through the stable
  `gpuDpfZpFullEvalSum` API with corrected full-128-bit seed semantics.
- Serialized keys round-trip byte-identically and have the same evaluator result before/after transfer.
- CPU reference, corrected GPU evaluator, and centralized/distributed
  generators agree on the functionality but are independently implemented;
  a seed-LSB=1 control fails under the obsolete semantics and passes under the
  corrected one.
- Compute Sanitizer or equivalent bounds checking reports no memory error on the focused suite.
- Production source contains no correctness PRG and no unlabelled cross-party read.

**Evidence:** source/build/run pair, per-case CSV/MD/log, serialization fixtures, AES known-answer log, memo, smoke-gate hook.

**Checkpoint commit:** `ringlpn(m1): add AES GPU distributed DPF core`

**Claim unlocked:** GPU-format functional compatibility under ideal transports; not M1 completion.

---



### S4 — M1b: real OT/OLE/triple transport and self-bootstrapping

**Status 2026-08-04:** real SCI/IKNP string OT, Boolean triples, and Gilboa
scalar OLE are integrated and measured. IKNP is not silent OT, key generation
is host-side, setup is not self-sustaining from PCG output, and true network
rounds are unmeasured; the strict S4 gate remains open.


**Purpose:** remove ideal OT, bit-triple, and scalar-OLE calls from distributed key generation.

**Work:**

1. Select a maintained, reviewed OT-extension implementation with compatible license/build support and documented 128-bit parameters. Do not hand-roll a silent-OT protocol. Record the exact upstream commit and configuration.
2. Instantiate the level-walk string OTs from the selected OT/COT API and implement the ripple-adder triples from real binary correlations.
3. Implement the three scalar OLEs per tree. Epoch zero uses a small explicit OT-based Gilboa source; later epochs reserve output OLE slots from the Ring-LPN factory according to S2's budget.
4. Domain-separate every base OT, epoch, tree, level, direction, limb, layer, and purpose. Add monotonic correlation IDs and fail on reuse.
5. Batch network messages across all trees at the same dependency level. Measure actual payload bytes, framing bytes, setup bytes, rounds, CPU/GPU overlap, and wall-clock time; estimates do not satisfy M1.
6. Retain an ideal-transport build only as a reference mode. It must never share a “protocol-backed” result column.
7. Add transport fault controls: truncated frame, wrong epoch, duplicate correlation ID, inconsistent batch size, and corrupted opening must cause a deterministic nonzero failure.

**Gate:**

- All S3 correctness/serialization cases pass with only real cryptographic transports.
- At least 256 pinned depth trees run in one batch for both CRT limbs.
- No ideal OT/OLE/triple call is reachable in the publication configuration.
- Measured byte counters agree with packet/transport counters; round counts come from actual dependency-separated sends.
- Bootstrap accounting balances for multiple consecutive epochs without slot reuse or hidden external OLE after epoch zero.
- Repeated runs under ASan/UBSan for host transport code and Compute Sanitizer for GPU code are clean.

**Evidence:** transport transcript schema, pcap or equivalent byte-accounting log without secrets, per-epoch budget CSV, benchmark CSV/MD/log, dependency manifest, memo.

**Checkpoint commit:** `ringlpn(m1): replace ideal keygen transports with real OT and OLE`

**Claim unlocked:** M1 complete at the pinned parameters, after independent security review of the selected primitives and composition.



---

### S5 — M2: drive the real Ring-LPN transcript with distributed keys

**Status 2026-08-04:** the live forward-FC runner directly consumes distributed
keys in both Ring-LPN directions/CRT limbs and has no centralized keygen mode.
The old nine-case single-process diagnostic intentionally remains unchanged as
an oracle-labelled reference; its original migration gate is superseded by the
stronger live execution evidence.


**Purpose:** remove `build_spfss_keys()` from the publication path.

**Primary target:** `src/bench_ole_ringlpn_cuda.cu` and `src/bench_orca_fc_real_ole_transcript.cu`; keep centralized keygen only as a labelled test oracle.

**Work:**

1. Feed S4's two-party `GPUDPFZpKey` outputs directly into the existing Figure 2 expansion and slot-packing pipeline.
2. Preserve the unchanged GPU evaluator and `gpuMatmulBeaver` online consumer.
3. Make the centralized `build_spfss_keys()` path impossible to select in the publication runner; a mode flag must fail closed rather than silently falling back.
4. Verify lifecycle and ownership across batches, limbs, directions, and repeated layers. Zeroize transient private seeds after use where the implementation can do so without introducing unsafe behavior.
5. Separate keygen, expand, derandomize, conversion-oracle, write, and online-consumer timings.

**Gate:**

- The existing real-generator transcript suite passes 9/9 at q64/q128 and uniform/regular configurations with distributed keys.
- A build-time or runtime assertion demonstrates that centralized keygen was not called.
- Corrupting one distributed key fails the unchanged consumer contract.
- The only remaining publication-path oracle is share conversion, explicitly labelled for S6.

**Evidence:** regenerated flagship CSV/MD/log with an `spfss_key_source=distributed_real_transport` field, negative-control rows, and updated memo/paper.

**Checkpoint commit:** `ringlpn(m2): drive real OLE transcript with distributed keys`

**Claim unlocked:** linear-layer Beaver keys come from a two-party keygen; conversion remains idealized.

---

### S6 — M3: protocol-backed conversion with PCG correlations and log-round comparison

**Status 2026-08-04:** exact two-process SCI/IKNP conversion is integrated in
the live forward path and has an exact hybrid simulator. PCG-sourced binary
correlations, a logarithmic-depth prefix circuit, silent transport, and
dependency-round measurement remain open; therefore the strict S6 performance
gate is not complete.


**Purpose:** remove the dealer-labelled conversion-correlation source and the exact-carry oracle from the publication path.

**Primary targets:** `src/test_secure_convert.cpp`, `src/orca_fc_ringlpn_keywriter.cuh`, and `src/bench_orca_fc_real_ole_transcript.cu`.

**Work:**

1. Generate binary triples, daBits, and edaBits from the selected binary PCG/OT path; record correlation IDs and enforce one-time consumption.
2. Replace the ripple comparator in the production path with a prefix comparator having `O(log bw)` dependency depth. Retain the ripple implementation as an independent reference/ablation.
3. Define the exact q64 and q128/CRT share-conversion transcript, including wrap handling, openings, aborts, and Garner-lift interaction.
4. Test zero, one, `2^bw-1`, prime/CRT boundaries, wrap/no-wrap pairs, both parties' extreme shares, malformed correlations, duplicate IDs, and corrupted opens.
5. Wire the protocol-backed conversion into the flagship transcript; make `exactZmToRingShares` and dealer-labelled correlations unreachable in the publication runner.
6. Measure preprocessing correlations, online bytes, rounds, CPU/GPU time, and peak memory separately.

**Gate:**

- The conversion suite is bit-exact against the independent oracle for every deterministic edge and randomized case at q64/q128.
- Measured online dependency depth is `O(log bw)` and matches the prefix circuit, not the old ripple count.
- The real-generator transcript passes with `conversion_source=pcg_protocol` and asserts no exact-carry oracle/dealer block was called.
- Correlation generation and consumption balance exactly; duplicate use fails.

**Evidence:** source/build/run pair, conversion and correlation CSV/MD/log, circuit/count derivation, updated flagship artifacts and paper.

**Checkpoint commit:** `ringlpn(m3): source conversion correlations from the PCG`

**Claim unlocked:** the FC transcript has no keygen or conversion oracle; process isolation and seed hygiene remain for S7.

---

### S7 — M4: exact public coins and two-process forward composition — **executable boundary complete 2026-08-04**

**Purpose:** make party separation real rather than a one-process discipline.

**Work:**

1. Instantiate D3 by having each party sample and exchange a full canonical
   uniform field-coefficient vector; their modular sum is the exact public
   Ring-LPN polynomial. Do not replace it with a revealed short PRG seed
   without a separate reduction.
2. Draw DPF roots, masks, and sparse noise only from party-local OpenSSL DRBGs.
   Domain-separate session, party, direction, batch, limb, tree/slot, and
   purpose; reject zero/replayed session identifiers.
3. Run each party as a separate OS process pinned to a distinct GPU. Each
   process reads only its own noise record, creates one party-local record, and
   cannot select centralized keygen or clear conversion.
4. Exchange only protocol-declared OT/OLE messages, public-polynomial shares,
   correction-word shares, derandomization openings, conversion messages, and
   publication handshakes. Validate public preflight parameters before
   correlation setup.
5. Publish records transactionally through a same-directory temporary file and
   atomic rename. Both parties must agree that both records exist before exit;
   the post-exit checker is the only process that reads both.
6. Exercise q64/q128, regular/uniform, multi-batch, malformed/corrupt/swapped
   records, mismatched preflight, stale output, forced rename failure, invalid
   session/port bounds, and unilateral-abort cleanup.
7. Treat this as one forward-FC component. Stateful
   bias/truncation/backward/optimizer transitions and nonlinear keys are
   separate follow-on designs, not hidden requirements for the current claim.

**Gate for the bounded forward artifact:**

- Two processes on distinct GPUs complete keygen -> expand -> derandomize ->
  convert -> transactional key write in all six cases.
- The post-exit checker validates key order and unchanged
  `gpuMatmulBeaver`; corrupt and swapped records fail.
- Source review confirms no undeclared live cross-party file/read path and no
  shared private seed.
- Raw party records are deleted after validation; crash/failure controls do not
  publish a final record.
- Application byte counters and direction switches are labelled with their
  exact scope; direction switches are not called network rounds.
- Authenticated transport, two-host execution, and stateful training remain
  open deployment/scope gates.

**Evidence:** two-party runner, independent P0/P1 CSV/MD/log, transcript-hash report, fault-control results, network manifest, updated canonical gate.

**Checkpoint commit:** `ringlpn(m4): run dealerless FC preprocessing as two processes`

**Claim unlocked:** “Orca FC preprocessing is dealerless in the two-party semi-honest splittable-Ring-LPN model,” contingent on S2 and S8. Never shorten this to “dealerless Orca.”

---

### S8 — Close the proof and audit implementation against it

**Status 2026-08-04:** conditionally closed for the bounded forward artifact.
The paper contains the exact DPF coupling, both batch simulators, conversion
simulator, source-to-transcript map, and conditional theorem. Independent
proof, source, and composition reviews reported no critical defect after
fixes. S2's concrete parameter/reduction gate and authenticated deployment
remain open, and source changes require renewed review.


**Purpose:** turn the design argument into a publication-grade security result after the real transcript is fixed.

**Work:**

1. Complete hybrid proofs for D1–D4 under corruption of either party: seed setup, shared-position adder, tree walk, three-OLE payload correction, bootstrapping, Ring-LPN expansion, derandomization, conversion, and key output.
2. Prove or cite the exact composition theorem used for OT/OLE, AES, commitment, and splittable Ring-LPN hybrids. State selective/adaptive corruption limits and abort semantics.
3. Show that all openings are simulatable: correction words including `finalCW`, per-slot `d/e`, conversion opens, and public coin-toss messages.
4. Prove correlation freshness and epoch separation. Treat reuse as a security failure, not merely an implementation bug.
5. Map every protocol message and randomness label in the implementation to the specification. Remove dead, oracle, or fallback paths from the publication binary; keep references in separate test binaries.
6. Perform independent cryptographic review and independent implementation review. Track findings by severity; all critical/high findings block S9.
7. Add a limitations section covering semi-honest only, linear layers only, side channels outside the model, denial of service/abort, and the stronger splittable assumption.

**Gate:**

- The paper contains a theorem, explicit assumptions, both simulators, and a complete hybrid argument for the actual implementation transcript.
- The corrected multiplicative-payload adaptation has no unresolved proof gap.
- A source-to-protocol checklist accounts for every send/open/read across the party boundary.
- No critical/high review finding remains open; medium findings are fixed or disclosed with a bounded claim.

**Evidence:** proof appendix, review checklists, finding log, source-to-spec matrix, claim audit.

**Checkpoint commit:** `ringlpn(security): close simulation proof and transcript audit`

**Claim unlocked:** computational privacy in the stated semi-honest model at S2's pinned parameters.

---

### S9 — M6: publication-quality evaluation

**Status 2026-08-04:** partially exercised, not complete. The exact
`1x512x1000` ResNet18 classifier-layer row has one warmup and ten passing
measured trials, an environment/binary/source digest manifest, matched
stock-dealer keygen timing, final key/record bytes, and unchanged online
timing. It is not a full inference/truncation run and remains a feasibility
row: no pinned parameters, closest compatible dealerless baseline, peak-memory
or dependency-round metric, two-host network result, or all-forward-linear-layer
model run exists.


**Purpose:** establish usefulness, costs, and bottlenecks without mixing evidence levels.

**Configurations:**

- Pinned secure parameters from S2 are the headline rows.
- `c=2,t=8,n=8192` remains a clearly labelled correctness/smoke row only.
- q64 and q128; supported FC shapes from CNN2 and CNN3; both training and inference FC directions where Orca uses them.
- Trusted-dealer baseline, centralized/oracle diagnostic, and fully protocol-backed system in separate columns.
- At least one closest dealerless preprocessing baseline selected by S2's
  novelty audit, matched to the same security level and FC shapes. If
  compatible code is unavailable, use a reproducible normalized
  communication/round/compute comparison and explain incompatibilities.

**Measurement protocol:**

1. Same commit, compiler flags, GPU architecture, clocks/power state as observed, and input shapes for every A/B comparison.
2. Record driver, CUDA, compiler, container digest, CPU/GPU model, topology, temperature/utilization, and competing processes.
3. Microbenchmarks: at least 5 warmups and 30 measured repetitions; report median, IQR, mean, standard deviation, and 95% confidence interval.
4. Model-scale preprocessing: at least 1 warmup and 10 measured repetitions, matching Orca's existing convention; correctness must pass on every repetition.
5. Report offline wall time, stage GPU/CPU time, communication payload and framing bytes, dependency rounds, throughput, key bytes, peak host/GPU memory, and unchanged online time.
6. Run scaling sweeps over tree count/batch size and layer `M,K,N`; include the pinned `n,c,t` and enough points to expose saturation or memory limits.
7. Ablate slot packing, regular versus uniform noise where both have valid security interpretations, one- versus two-OT-per-level if both remain, prefix versus ripple comparison, and transport/setup versus steady-state epochs.
8. CNN2 and CNN3 may retain dealer-backed nonlinear/other-layer preprocessing, but those costs must be in separate rows/columns. The paper reports dealerless **FC preprocessing**, not a dealerless full model.
9. Loopback results are labelled loopback. Network-latency conclusions require a second host or controlled, documented network; without that prerequisite, restrict claims to bytes, rounds, and local compute.

**Gate:**

- One committed runner regenerates every headline table and figure from raw outputs.
- Fully protocol-backed FC preprocessing completes for all FC layers of at least CNN2; CNN3 supplies the required scale point.
- Every row records parameter security status and oracle/protocol source fields.
- Confidence intervals and raw repetitions are available; no best-of-run timing is reported.
- Consumer output matches trusted-dealer output and the unchanged online path on every case.
- The closest-baseline comparison uses matched assumptions/shapes and
  reproducible source data; trusted-dealer overhead alone is not the systems
  value claim.

**Evidence:** raw per-trial CSV/logs, aggregated tables, plotting scripts, model/layer manifest, baseline comparison, profiling traces for bottlenecks.

**Checkpoint commit:** `ringlpn(m6): add model-scale dealerless FC evaluation`

**Claim unlocked:** measured cost and scalability of dealerless Orca FC preprocessing on the evaluated platform and network only.

---

### S10 — Reproducible artifact and submission candidate

**Purpose:** produce the exact paper/artifact revision that can be submitted after advisor approval.

**Artifact work:**

1. Build from a clean clone with pinned submodules/dependencies and a versioned container image or deterministic setup script. No absolute user paths, root-owned outputs, hidden external checkout, or pre-generated key dependency.
2. Provide one host-only gate, one required-GPU gate, and one two-process publication runner. All fail nonzero when hardware, data, or a security-relevant dependency is missing.
3. Regenerate every paper table/figure from committed raw data. Store commands and checksums; keep raw and summarized artifacts distinct.
4. Have a second person reproduce the artifact from the instructions on a clean environment. Treat undocumented setup intervention as a failed artifact review.
5. Archive the exact source commit, container digest, parameter-estimator version, and paper PDF. Tag the accepted candidate only after all gates pass.
6. Reconcile the final dependency/source/license/citation inventory against
   S2's approved provenance record and every dependency actually shipped.
   Cite Özcan--Savaş GPU-NTT (ePrint 2023/1410 and the applicable published
   version) whenever its merge/four-step algorithms or code are discussed or
   used. Any post-S2 code source or overlap requires the same professor
   ownership/reuse/disclosure review before inclusion.

**Paper work:**

1. With the advisor, lock the target venue, title, disclosure requirements, and
   page/supplement limits. Alp remains the sole author; do not add commit
   co-author trailers or paper co-authors.
2. Convert the v2.5 technical report into a venue-specific results paper only after the parameter and performance gates close: research question, novelty, protocol, theorem, parameter audit, implementation, evaluation, related work, limitations, and reproducibility appendix.
3. Expand related work against the exact distributed-DPF, silent OT/VOLE, Ring-LPN PCG, mixed-circuit conversion, and secure-ML systems baselines. Distinguish inherited primitives from this work's contribution.
4. Remove proposal/future-tense language and any dashed “today” oracle box only when the corresponding gate is genuinely closed.
5. Run three reviews: cryptographic correctness/claims, systems methodology/performance, and artifact reproducibility. Resolve every blocking comment in a committed revision.
6. Build in the venue toolchain with zero LaTeX warnings, inspect every rendered page, verify references/links, and run a final current-document claims-drift audit.

**Final gate:**

```bash
RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 PATH=/usr/local/cuda/bin:$PATH \
  scripts/run_paper_checkpoint_smoke.sh
# plus the committed two-process model-scale runner
```

Both commands must exit 0. The paper's generated tables must match the committed CSVs exactly. The working tree for Ring-LPN must be clean, all stage commits present, and no critical/high review issue open.

**Checkpoint commit:** `ringlpn(paper): prepare reproducible publication candidate`

**Release tag after advisor approval:** `ringlpn-publication-candidate-v1`

## 5. Evidence matrix

| Claim | Required evidence | Gate/stage |
|---|---|---|
| Distributed payload keygen is correct | Independent host and GPU evaluators, deterministic edges, corruption controls | S3 |
| Distributed keygen is private | Real OT/OLE, simulator for each corruption, transcript audit | S4 + S8 |
| Parameters provide at least 128-bit security | Reproducible estimator transcripts for exact splittable distribution | S2 |
| Contribution is novel and reusable code is permitted | Formal protocol delta, closest-prior-art table, source/license inventory, professor decisions | S2 |
| Keygen removes centralized O1 | Live distinct-process absence assertion; q64/q128 uniform/regular/multibatch plus classifier gate | S5 |
| Conversion removes O2 | PCG correlation accounting, no `exactZmToRingShares`, bit-exact suite | S6 |
| Preprocessing is actually two-party | Separate processes/GPUs, independent state/logs, declared network messages only | S7 |
| Online Orca path is unchanged | `gpuMatmulBeaver` as independent consumer; flag-off baseline still byte-identical | S5–S9 |
| Performance claim is reproducible | Raw repetitions, confidence intervals, same-hardware A/B, scripted tables | S9 |
| Publication artifact is reproducible | Clean-clone second-person run, pinned container/dependencies, all gates | S10 |

## 6. Risks and mandatory responses

| Risk | Detection | Mandatory response |
|---|---|---|
| Neither deployed sampler has a reviewed projected-distribution/security reduction | S2 primary-source/proof audit | Resolve the BCG rule, distribution/tail/cancellation, structured-code, and two-limb advantage obligations before selecting any tuple; do not transfer exact/regular finite-field model outputs by analogy. |
| Pinned `t` makes `3c^2t^2` exceed available output | Full epoch budget is non-positive | Increase `n`/change distribution and rerun NTT/memory gates; do not hide bootstrap cost. |
| New `n`/primes invalidate cheddar | Roots-of-unity or modulus-width gate fails | Re-pin primes and reopen the documented NTT decision. |
| Real OT/OLE library is immature, incompatible, or license-blocked | S4 dependency review/bench | Change implementation before integration; do not ship a hand-rolled replacement under a mature-protocol name. |
| Phase C or composition proof fails | S1/S8 simulator cannot reproduce a view | Redesign the transcript before performance work; no privacy claim. |
| GPU batching exceeds memory | S3/S9 scaling sweep | Stream bounded batches and report the schedule; do not silently reduce the publication workload. |
| Conversion dominates latency/rounds | S6/S9 stage counters | Optimize the prefix/batching path or report it honestly as the bottleneck. |
| Only loopback is available | No two-host measurement | Restrict network claims to measured bytes/rounds and label loopback timing; obtain two-host data before claiming LAN/WAN performance. |
| Nonlinear keys remain dealer-generated | Model audit | Keep the title/abstract/claims explicitly scoped to FC/linear preprocessing. |
| Artifact works only in the current dirty workstation | Clean-clone reproduction fails | Fix setup/pinning until a second-person clean run passes; submission candidate remains blocked. |

## 7. Publication exit checklist

Publication readiness is reached only when every box is supported by a committed artifact:

- [x] S1 protocol and proof contract frozen for advisor review after the
  requested model-assisted audit; independent human cryptographic review
  remains an S8 gate.
- [ ] S2 exact splittable parameters have a reviewed reduction, only in-domain independently reproducible estimator evidence, and the claimed security level.
- [ ] S2 formal novelty/overlap, source/license inventory, and professor
  provenance decisions are recorded before overlapping implementation.
- [ ] S3 remains open. The four-call full-width AES/GPU and seed-bit-0
  sensitivity component gate passes, but pinned configurations with at least
  256 trees, sanitizer and serialization evidence, the no-unlabelled-read
  audit, and the stage checkpoint remain unmet.
- [ ] S4 real OT/OLE/triple path passes, bootstraps, and reports measured bytes/rounds.
- [ ] S5 centralized keygen removed from the publication transcript.
- [ ] S6 dealer-labelled conversion correlations and exact-carry oracle removed.
- [ ] S7 complete forward/bias/truncation/`dW`/`dX`/bias-gradient/
  dual-optimizer two-process/two-GPU pipeline passes with exact serialization,
  persistent mask/velocity handoff, coin-tossed public seeds, and private
  CSPRNG streams.
- [ ] S8 both-party simulation proof and implementation audit have no blocking finding.
- [ ] S9 model-scale FC evaluation and the closest-baseline comparison are
  statistically reported with evidence levels separated.
- [ ] S10 clean-clone artifact reproduction, venue paper build, rendered review, and final full gate pass.
- [ ] Every completed stage has its own immutable checkpoint commit and current documentation.
- [ ] Final wording never exceeds: dealerless **linear/FC-layer preprocessing**, two party, semi-honest, pinned splittable Ring-LPN parameters, evaluated environments only.
