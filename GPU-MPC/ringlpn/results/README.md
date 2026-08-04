# ringlpn results — directory index

Reorganized 2026-06-10. Every run script writes into its artifact directory
below; nothing writes to this top level anymore.

## Current checkpoint (2026-08-04)

The live forward-FC artifact now composes the formerly separate boundaries:
`src/test_two_party_fc_preprocess.cu` calls party-local SPFSS
(`src/two_party_spfss.h`), distributed DPF
(`src/two_party_dpf_protocol.h`), real SCI/IKNP/Gilboa transport
(`src/two_party_ot.h`), party-local GPU Ring-LPN expansion
(`src/ringlpn_ole_party.cuh`), and exact conversion
(`src/secure_convert.{h,cpp}`). Each party is a separate OS process on a
distinct GPU, samples only its own sparse-noise share, and does not read the
peer's file in the live source path. The current single-UID loopback runner
does not enforce OS-level peer-file isolation. The checker runs only after both
processes exit and validates the two records through unchanged
`readGPUMatmulKey`/`gpuMatmulBeaver`.

Current live evidence:

- `fc/two_party_fc_preprocess_2026_08_04.csv` — q64/q128,
  regular/uniform, small and q64 multi-batch rows; all six public/key-order/
  unchanged-online contracts pass.
- `fc/two_party_fc_preprocess_controls_2026_08_04.csv` — malformed,
  corrupt, swapped, mismatched-preflight, stale-output, forced-rename,
  invalid-session/port, and unilateral-failure controls; all reject or clean
  up as specified.
- `fc/two_party_fc_model_scale_2026_08_04.csv` plus summary/environment —
  exact ResNet18 classifier-layer shape `1x512x1000`, q128/bw32,
  `n=8192,c=2,t=8`, one warmup plus ten measured trials, 10/10 pass. This is
  not a full ResNet18 inference or scale-10 truncation run. Median critical-path
  preprocessing is 35.735 s; both parties send 608,860,824 application bytes;
  matched stock trusted-dealer keygen is 10.813 ms median; the unchanged
  two-share online checker is 1.099 ms median. Final Orca payload remains
  4,108,096 bytes per party.
- The live runner and model runner retain canonical aggregate
  CSV/log/manifest evidence plus per-party stdout/checker logs and commit
  markers. Raw party key records are validation inputs, not public evidence,
  and are removed after the post-exit checker.

Proof/evidence boundary:

- The current v2.5 report and security contract contain an exact
  correction-word coupling, both role-specific correlated-batch simulators, an
  exact conversion simulator, a source-to-transcript map, and a conditional
  static-semi-honest theorem for one forward FC matmul.
- Two independent model-assisted proof/source/composition reviews found no
  critical defect after the recorded fixes. This is not independent human
  cryptographic review.
- No concrete Ring-LPN parameter is pinned. The primary-source audit found no
  reviewed reduction from the exact projected/regular distribution and
  structured code to a concrete advantage bound; q64/q128 name one/two
  approximately 62-bit arithmetic limbs, not security levels.
- Current measurements use unauthenticated local loopback and exclude TCP/IP
  framing and base-OT setup from the application-byte counter. They establish
  executable correctness and cost at feasibility parameters, not 128-bit,
  malicious, WAN, training-layer, full-model, or full-dealerless-Orca claims.
- The matched comparison is negative: median live preprocessing is about
  3,286 times stock dealer keygen. GPU-batched DMPF/DPF generation, silent
  correlation expansion, reviewed parameters, authenticated two-host
  evaluation, memory/round instrumentation, and closest compatible
  dealerless-PCG baselines remain publication gates.

The last complete canonical component gate before this final documentation
pass exited zero after 333.04 s and printed `[paper-smoke] ALL GATES PASS`.
All source-changing work requires a fresh run before checkpointing. Alp
`<fcetin@hawk.iit.edu>` remains the sole paper author by user direction;
inherited code/protocols remain cited, and unresolved ownership/reuse decisions
must not be treated as settled.

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
| `secure_convert/` | Two-process evidence for exact `Z_M -> Z_2^bw` conversion using SCI/IKNP-generated edaBits/daBits/Boolean triples; common preflight, bounded transactional outputs, corruption controls, and separate transcript counters. The live forward-FC path consumes this API. The wrap bit is never opened; the current security contract gives the hybrid simulator. Transport remains unauthenticated loopback and ripple-depth. | `run_secure_convert_test.sh` |
| `dpf/` | Distributed DPF keygen artifacts: ideal-functionality protocol logic; two-process SCI/IKNP+Gilboa transport with measured bytes/direction switches; full-width four-call GPU AES parity with enforced seed-bit-0 sensitivity; strictly validated GPU-evaluated party keys; offline correctness/corruption/invalid-input controls. Direction switches are not network rounds; security reductions remain open. | `run_distributed_dpf_keygen.sh`, `run_two_party_dpf_keygen.sh`, `run_two_party_gpu_dpf.sh` |
| `profiling/` | VTune hotspot/memory captures | `run_vtune_*.sh` |
| `outreach/` | Abstracts, posters, professor memos/status emails | hand-written |
| `archive/` | Superseded one-offs: early spot checks, `*_regular_patch`, `*_after_linear`, old plan drafts | frozen |
| `security/` | **Start with `security/README.md`.** The dated conservative-pin files are immutable failed-rule transcripts, invalid for parameter selection/security claims. The raw projection CSV mixes out-of-domain calls with unproved finite-field model outputs. Engineering feasibility/budget rows remain non-security evidence. No parameter or 128-bit classical/quantum claim is pinned. | `audit_ringlpn_projection_security.py`, `audit_ringlpn_finite_field_models.py`, focused benchmark commands |
| `pcg/` | Adapted rows from the licensed native-`Z_(2^bw)`/Galois-ring PCG artifact, with patch digest and correctness gate; not a reproduction of the released benchmark | `run_native_ring_pcg_baseline.sh` |

## Reports, newest first

| File | What it is |
|---|---|
| `reports/two_party_dpf_transport_memo_2026_07_29.md` | **Two-PROCESS keygen on a real transport**: SCI IKNP over TCP, Gilboa `Z_p` OLE, OpenSSL-private-DRBG roots, 369/369 host-reference pairs, and 88 GPU-evaluated full-width-AES pairs. The unchanged host-reference evaluator remains splitmix64 correctness-only; see §3.4 for the deployed four-call full-width GPU AES evidence and the remaining reduction/silent-OT/network-round boundaries. |
| `reports/dealerless_ole_two_party_keys_memo_2026_07_29.md` | **M2 CORE GATE**: independently sampled per-party noise and SPFSS keys generated by two OS processes over real OT drive the Figure 2 OLE engine at q64/q128 with uniform/regular noise; measured keygen direction switches/bytes/time per limb and explicit remaining oracles |
| `reports/session_handoff_2026_07_29_dmpf_comparison.md` | **HISTORICAL/SUPERSEDED** pre-sweep, pre-transport handoff; use `CLAUDE.md` for current catch-up and the measured S2 comparison for final rows |
| `reports/s2_architecture_comparison_2026_07_29.md` | **MEASURED ARCHITECTURE COMPARISON**: 275x/329x under uniform noise, but 0.79x for OKVS and only 2.29x for big-state at the deployed regular layout; dealerless-setup status, native-ring artifact defects, and the decision table. Its §7 wording leaves four owner questions open, but their later binding answers are recorded in `CLAUDE.md` under “2026-07-29 owner route decisions”; do not treat them as open or use the uniform result as this project's deployed-layout result. |
| `reports/s2_parameter_novelty_provenance_audit_2026_07_29.md` | **S2 HARD-STOP REPORT, corrected 2026-08-04**: exact primary-source audit, invalid estimator-call rows, unproved projected-noise/structured-code mapping, implementation-only `n=2^17,c=4,t=34` NO-GO, alternatives/provenance, and no pinned parameters or 128-bit claim. |
| `reports/s2_professor_decision_request_2026_07_29.md` | **Historical advisor request.** Its unresolved security/provenance questions remain required before claim advancement, but its “before S3 implementation” wording predates the owner's implementation-only S3–S6 gate lift and must not be used to deny the component work that subsequently proceeded. |
| `reports/publication_readiness_plan_2026_07_21.md` | **BINDING PUBLICATION ROADMAP**: integrated dealerless Orca FC thesis; advisor-first report; S1--S10 dependency order, security proof and parameter gates, M1--M6 implementation/evaluation criteria, risks, evidence matrix, per-stage user consultation, and required checkpoint commit |
| `reports/dealerless_orca_fc_security_contract_2026_07_29.md` | **CURRENT FORWARD SECURITY CONTRACT:** exact DPF correction-word coupling, role-specific correlated-batch simulators, conversion simulator, full live source-to-transcript map, conditional forward theorem, obligation table, and explicit concrete-parameter/authentication/training limits. |
| `reports/session_handoff_2026_07_21.md` | **HISTORICAL/SUPERSEDED** corrected-M1/v2.3 checkpoint handoff; current status is in `CLAUDE.md` |
| `reports/distributed_dpf_keygen_memo_2026_07_21.md` | **Corrected M1 host protocol-logic prototype**: party-separated and functionally validated by unchanged evaluator using ideal OT/triple/OLE and non-cryptographic correctness PRG; 2,432 trees, three OLEs/tree, old-sign regression, 5/5 corruptions, 6/6 invalid inputs, ideal-mask-draw and correlation-reuse controls, executable split accounting (1,908 logical / 3,816 meaningful share bits at depth 14); `dpf/distributed_dpf_keygen_prototype.{csv,log}` |
| `reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex` (+`.pdf`) | **LIVE internal v2.5 technical report:** composed two-process forward-FC design, exact hybrid proof boundary, matched dealer comparison, ten-trial exact ResNet18-classifier-layer evidence, related work, limitations, and explicit “not conference-ready” assessment. Warning-free 21-page PDF; internal/advisor-ready, not submission-ready. |
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

Observed 2026-08-04 with `CUDA_VISIBLE_DEVICES=3`: exit zero, 333.04 s,
literal final marker present. Re-run after any code, gate, parameter, or
composition change; the marker certifies component correctness at the
exercised feasibility configurations, not concrete security. The separate
live/model runners provide the composed forward-FC evidence.

Conventions: every run produces a `.csv` (data), usually a `.md` (summary), and
a `.log` (raw stdout + stderr). `validation`/`*_contract` columns must read
`pass`; suites exit non-zero on any failure.

**Staleness convention (binding, see `../CLAUDE.md` documentation contract):**
documents whose claims are no longer current carry a `> **HISTORICAL …**`
banner at the top; `outreach/` and `archive/` are wholly historical (see their
READMEs). A document is current only if it is unbannered and dated
2026-06-10 or later. When your work supersedes a document, banner it in the
same commit.
