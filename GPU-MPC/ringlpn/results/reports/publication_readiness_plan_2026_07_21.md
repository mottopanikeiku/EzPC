# Dealerless Orca linear-layer preprocessing — publication readiness plan

**Date:** 2026-07-21  
**Scope:** two-party, semi-honest, dealerless preprocessing for Orca's **linear/FC layers** from splittable Ring-LPN. Nonlinear-layer FSS keys and malicious security remain out of scope.  
**Starting checkpoint:** commit `28f8451` (`ringlpn: add corrected distributed DPF keygen artifact`), with the required GPU gate ending `ALL GATES PASS`.

## 1. Definition of publication-ready

The project is publication-ready only when all of the following are true at the same committed revision:

1. **Protocol:** the publication path contains no centralized SPFSS-keygen oracle and no dealer-labelled share-conversion correlations. Ideal implementations remain only as independent test references.
2. **Security:** the paper states a precise functionality and two-party semi-honest theorem, gives simulators for corruption of either party, accounts for every opening, and reduces privacy to named assumptions: AES, the chosen OT/OLE implementation, commitments, and the pinned splittable Ring-LPN parameters.
3. **Parameters:** a reproducible audit supports the claimed security level for the exact noise distribution and fully split ring used by the implementation. Smoke parameters are never presented as secure parameters.
4. **Implementation:** two OS processes on separate GPUs generate the FC preprocessing, communicate only through the declared transport, serialize byte-compatible keys, and validate through unchanged `gpuMatmulBeaver`.
5. **Evaluation:** trusted-dealer, oracle-backed, and fully protocol-backed results are separated. Time, communication, rounds, key bytes, and memory are measured at the exact pinned parameters and at model-scale FC shapes.
6. **Artifact:** a clean checkout can regenerate every claim, table, and figure with pinned dependencies and nonzero-on-failure gates.
7. **Paper:** the final venue-formatted paper, appendix, and artifact documentation survive cryptographic, systems, and reproducibility review without an unresolved claim/evidence mismatch.

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
S1 protocol/proof contract ─┬─> S3 M1 cryptographic GPU core ─> S4 M1 real transports ─> S5 M2 integration ─┐
S2 M5 parameter audit ──────┘                                                                                ├─> S7 M4 composition
S6 M3 protocol-backed conversion (independent after S1/S2) ────────────────────────────────────────────────┘
                                                                                                              ↓
                                    S8 proof + implementation audit ─> S9 M6 evaluation ─> S10 release
```

S2 is scheduled before performance freeze even though M5 is logically parallel to M1/M3: it may change `n`, `c`, `t`, the primes, the bootstrap budget, GPU memory, and the NTT backend. S3–S5 and S6 may proceed on separate branches after S1/S2; each branch must preserve its own stage commits and gates. S7 is the first point at which their protocol-backed paths compose.

## 4. Execution stages

### S0 — Corrected host checkpoint and truthful proposal — **complete**

**Commit:** `28f8451`  
**Evidence:** corrected Phase C, old-sign leakage regression, 2,432/2,432 host key pairs, regenerated proposal v2.2, canonical host gate, required GPU gate, warning-free 15-page PDF build and rendered review.

**Claim unlocked:** protocol-logic and host-format functional compatibility using ideal OT/triple/OLE and a non-cryptographic correctness PRG. Nothing stronger.

---

### S1 — Freeze the protocol, functionality, and proof obligations

**Purpose:** prevent implementation choices from outrunning the security argument.

**Work:**

1. Specify the ideal FC-preprocessing functionality: party inputs, public dimensions/parameters, party outputs, allowed aborts, and the exact public transcript.
2. Specify D1 at message level: arithmetic-share adder, level walk, control bits, seed correction words, three-OLE Phase C, and the sole Phase C opening `finalCW`.
3. Specify D2–D4 composition: conversion correlations, coin-tossed public seeds, private AES streams, derandomization openings, key serialization, and two-process transport.
4. Enumerate leakage explicitly: dimensions, parameter set, batch sizes, message lengths, round schedule, public correction words, and abort behavior. State that access patterns and sizes are public if that is the intended model.
5. Draft simulators for corrupted party 0 and party 1. In particular, prove that Phase C can simulate each party's OLE shares and the public `finalCW` without revealing the hidden sign, the other payload factor, or the secret point.
6. State all composition assumptions and theorem scope. Separate computational privacy from Ring-LPN pseudorandomness and correctness.
7. Add proof-driven tests for boundary inputs permitted by the functionality: point edges, all legal payload factors, wrap/no-wrap conversion boundaries, invalid/corrupted messages, transcript length invariants, and no correlation reuse.

**Gate:**

- A protocol transcript table accounts for every sent/opened value.
- Every implementation message maps to one line of the protocol specification.
- Both corruption simulators are complete at the hybrid-model level; no sentence relies on “the sign is random” without conditioning on a party's state.
- Advisor/cryptography review has no unresolved correctness or privacy blocker.

**Evidence:** a proof/specification section in the current paper plus a dated review memo and deterministic transcript tests.

**Checkpoint commit:** `ringlpn(proof): freeze semi-honest protocol and leakage contract`

**Claim unlocked:** none; this is the contract subsequent implementations must meet.

---

### S2 — M5 first: audit and pin splittable Ring-LPN parameters

**Purpose:** establish the parameter set before optimizing or publishing performance.

**Work:**

1. Reproduce the corrected BCG+20 splittable construction and its exact noise distribution. Treat uniform and bucket-regular noise separately; do not transfer an estimate between them.
2. Model the fully split ring's CRT projections and the relevant quasi-abelian/syndrome-decoding attacks. Include classical and, if claimed, quantum work factors; record estimator versions and assumptions.
3. Run reproducible ISD/decoding estimates across candidate `(n,c,t,p0,p1)` sets. Include sensitivity around the chosen point rather than one optimistic row.
4. Audit whether regular noise has a literature-backed reduction/estimator in this setting. If it does not, either use uniform noise for the security claim or name and defend a new structured-noise assumption; performance alone cannot select regular noise.
5. Recompute the complete epoch budget, not only `3c^2t^2<n`: reserve scalar OLEs for three OLEs per DPF tree, epoch-zero Gilboa bootstrap, conversion correlations, safety margin, and any rejected/unused slots. Prove no slot is reused.
6. Recheck NTT feasibility, prime bit width, `v2(p_i-1)`, CRT correctness, GPU memory, and batch size at the pinned `n`. If primes move to at most 60 bits, reopen the documented cheddar-versus-GPU-NTT decision; if `n` grows, rerun every polynomial and slot-packing gate.
7. Generate machine-readable parameter tables and estimator transcripts from a pinned script/container.

**Gate:**

- Conservative estimated security is at least 128 bits for the exact implemented distribution and all published parameter sets.
- The full bootstrap/consumption budget is positive with a documented margin.
- Both CRT limbs and the required NTT size pass host-reference and GPU tests.
- An independent reviewer can rerun the estimator command and obtain the reported table.

**Hard stops/reversals:**

- No concrete security claim if the regular-noise estimate is unsupported.
- Grow `n` or change the distribution if the bootstrap budget fails; do not weaken the inequality.
- Re-pin primes and reopen the NTT backend if the selected `n` exceeds current roots-of-unity headroom.

**Evidence:** estimator code, versioned raw transcripts, parameter CSV/MD/log, security memo, updated paper tables.

**Checkpoint commit:** `ringlpn(m5): pin audited splittable Ring-LPN parameters`

**Claim unlocked:** concrete security level for the pinned parameter set only.

---

### S3 — M1a: cryptographic, GPU-consumable distributed DPF core

**Purpose:** replace the splitmix64 host semantics with the AES/GPU key semantics used by the real expansion path while retaining ideal transports temporarily.

**Primary targets:** `src/test_distributed_dpf_keygen.cpp` as the independent host reference, `src/gpu_spfss_zp.cuh` as the unchanged consumer format/evaluator, and a production distributed-keygen module under `src/` rather than code embedded in a benchmark.

**Work:**

1. Use the existing AES-based GPU DPF expansion semantics. Remove splitmix64 from the production path; keep it only in the labelled host correctness reference.
2. Emit `GPUDPFZpKey`-compatible party keys: seeds, seed correction words, separate control correction bits, and final correction words. Define a versioned byte serialization with explicit endianness and bounds.
3. GPU-batch every tree level synchronously. Preserve party-owned buffers; no kernel or host helper may read both parties' private state outside a named ideal transport used at this stage.
4. Implement the known one-string-OT-per-level formulation or retain two only if the proof and measured cost justify it. The paper and counters must match the implementation exactly.
5. Cross-check the AES implementation against fixed known-answer vectors and a separately written CPU AES evaluator. Keep control bits out of seed LSBs.
6. Test q64/q128 limbs, pinned depths, batches from 1 through at least 256, edge points, maximum legal point, deterministic replay, malformed serialization, and corrupted correction words.

**Gate:**

- For at least 256 trees per pinned configuration, GPU full-domain evaluation reconstructs `beta [x=alpha]` with the existing `gpuDpfZpFullEvalSum` consumer unchanged.
- Serialized keys round-trip byte-identically and have the same evaluator result before/after transfer.
- CPU reference, GPU evaluator, and centralized generator agree on the functionality but are independently implemented.
- Compute Sanitizer or equivalent bounds checking reports no memory error on the focused suite.
- Production source contains no correctness PRG and no unlabelled cross-party read.

**Evidence:** source/build/run pair, per-case CSV/MD/log, serialization fixtures, AES known-answer log, memo, smoke-gate hook.

**Checkpoint commit:** `ringlpn(m1): add AES GPU distributed DPF core`

**Claim unlocked:** GPU-format functional compatibility under ideal transports; not M1 completion.

---

### S4 — M1b: real OT/OLE/triple transport and self-bootstrapping

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

### S7 — M4: coin-tossed seeds and two-process composition

**Purpose:** make party separation real rather than a one-process discipline.

**Work:**

1. Instantiate D3: commit-then-open public seed contributions; verify commitments before XOR/combine. Derive public and private AES streams with explicit domain separation over protocol version, session, party, layer, epoch, direction, limb, tree/slot, and purpose.
2. Draw private root seeds from the OS CSPRNG. Never derive both parties' private randomness from a shared benchmark seed.
3. Run each party as a separate OS process pinned to a distinct GPU, using the existing `SigmaPeer` transport. Ring-LPN code must not communicate through shared memory, shared writable files, or cross-party pointers.
4. Send only protocol-declared messages: commitments/reveals, OT/OLE traffic, correction-word openings, derandomization openings, and conversion messages. Version and length-check every frame.
5. Produce independent per-party logs with matching public transcript hashes but no private seeds, OT choices, or key material.
6. Test loopback for correctness. Any latency/throughput claim also requires two physical hosts or an explicitly documented network environment; otherwise report bytes/rounds and label loopback wall time as such.
7. Exercise disconnect, replayed session ID, mismatched parameters, wrong party role, malformed frame, and peer abort. All must fail without hanging or silently falling back.

**Gate:**

- Two processes on two GPUs complete keygen -> expand -> derandomize -> convert -> key write.
- Unchanged `gpuMatmulBeaver` passes for q64/q128 publication configurations.
- An audit confirms no undeclared cross-party data path and no common private seed.
- Per-party byte counts reconcile with transport totals and transcript hashes.
- Ten consecutive sessions use unique IDs/correlation ranges and pass without resource leaks.

**Evidence:** two-party runner, independent P0/P1 CSV/MD/log, transcript-hash report, fault-control results, network manifest, updated canonical gate.

**Checkpoint commit:** `ringlpn(m4): run dealerless FC preprocessing as two processes`

**Claim unlocked:** “Orca FC preprocessing is dealerless in the two-party semi-honest splittable-Ring-LPN model,” contingent on S2 and S8. Never shorten this to “dealerless Orca.”

---

### S8 — Close the proof and audit implementation against it

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

**Purpose:** establish usefulness, costs, and bottlenecks without mixing evidence levels.

**Configurations:**

- Pinned secure parameters from S2 are the headline rows.
- `c=2,t=8,n=8192` remains a clearly labelled correctness/smoke row only.
- q64 and q128; supported FC shapes from CNN2 and CNN3; both training and inference FC directions where Orca uses them.
- Trusted-dealer baseline, centralized/oracle diagnostic, and fully protocol-backed system in separate columns.

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

**Paper work:**

1. With the advisor, lock the target venue, title, author list/order, disclosure requirements, and page/supplement limits. Technical work before this decision remains venue-neutral.
2. Convert the v2.2 proposal into a results paper: research question, novelty, protocol, theorem, parameter audit, implementation, evaluation, related work, limitations, and reproducibility appendix.
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
| Keygen removes centralized O1 | Flag/assertion that `build_spfss_keys()` is unreachable; 9/9 transcript | S5 |
| Conversion removes O2 | PCG correlation accounting, no `exactZmToRingShares`, bit-exact suite | S6 |
| Preprocessing is actually two-party | Separate processes/GPUs, independent state/logs, declared network messages only | S7 |
| Online Orca path is unchanged | `gpuMatmulBeaver` as independent consumer; flag-off baseline still byte-identical | S5–S9 |
| Performance claim is reproducible | Raw repetitions, confidence intervals, same-hardware A/B, scripted tables | S9 |
| Publication artifact is reproducible | Clean-clone second-person run, pinned container/dependencies, all gates | S10 |

## 6. Risks and mandatory responses

| Risk | Detection | Mandatory response |
|---|---|---|
| Regular noise lacks a defensible security estimate | S2 literature/estimator audit | Use uniform noise for security claims or explicitly introduce and defend a new assumption; never reuse uniform estimates. |
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

- [ ] S1 protocol and proof contract reviewed.
- [ ] S2 exact splittable parameters independently reproducible and at least 128-bit secure.
- [ ] S3 AES/GPU distributed key format passes unchanged GPU evaluator.
- [ ] S4 real OT/OLE/triple path passes, bootstraps, and reports measured bytes/rounds.
- [ ] S5 centralized keygen removed from the publication transcript.
- [ ] S6 dealer-labelled conversion correlations and exact-carry oracle removed.
- [ ] S7 two-process/two-GPU pipeline passes with coin-tossed public seeds and private CSPRNG streams.
- [ ] S8 both-party simulation proof and implementation audit have no blocking finding.
- [ ] S9 model-scale FC evaluation is statistically reported with evidence levels separated.
- [ ] S10 clean-clone artifact reproduction, venue paper build, rendered review, and final full gate pass.
- [ ] Every completed stage has its own immutable checkpoint commit and current documentation.
- [ ] Final wording never exceeds: dealerless **linear/FC-layer preprocessing**, two party, semi-honest, pinned splittable Ring-LPN parameters, evaluated environments only.
