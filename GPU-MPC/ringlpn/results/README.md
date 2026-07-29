# ringlpn results — directory index

Reorganized 2026-06-10. Every run script writes into its artifact directory
below; nothing writes to this top level anymore.

## Where to look

| Directory | Contents | Produced by |
|---|---|---|
| `reports/` | **Start here.** Current plans, proposals, baselines, memos, handoffs | hand-written |
| `ntt/` | NTT/PolyMul sweeps: CPU (NFLlib), GPU cheddar q32/q64/q128, legacy | `run_sweep.sh`, `run_cuda_sweep.sh`, `run_cuda_sweep_legacy.sh`, `run_cuda_single.sh` |
| `ole/` | Figure 2 Ring-LPN OLE: q64/q128 × uniform/regular; `t=8` correctness smoke and `t=64` per-polynomial performance rows, neither security-pinned | `run_ole_sweep.sh` |
| `linear_ole/` | Ring-matrix OLE-to-Beaver (2x2x2, n=8192): q64/q128 × uniform/regular | `run_linear_ole_sweep.sh` |
| `vole/` | Standalone VOLE expansion prototype | `run_vole_sweep.sh` |
| `orca_fc/` | Orca FC artifacts: keywriter demo, ideal-OLE transcript, **real-OLE slot-packed transcript**, Zp bridge | `run_orca_fc_ringlpn_demo.sh`, `run_orca_fc_ideal_ole_transcript.sh`, `run_orca_fc_real_ole_transcript.sh`, `run_orca_zp_bridge_test.sh` |
| `secure_convert/` | Secure Z_M -> Z_2^bw conversion prototype | `run_secure_convert_test.sh` |
| `dpf/` | Distributed DPF keygen protocol-logic CSV/log: split transcript counters, ideal-functionality mask-draw accounting, consume-once correlation-ID control, invalid-input and independent corruption controls; plus older online-keygen memory-efficiency track | `run_distributed_dpf_keygen.sh`; tests/fss bench (older track, manual) |
| `profiling/` | VTune hotspot/memory captures | `run_vtune_*.sh` |
| `outreach/` | Abstracts, posters, professor memos/status emails | hand-written |
| `archive/` | Superseded one-offs: early spot checks, `*_regular_patch`, `*_after_linear`, old plan drafts | frozen |
| `security/` | S2 preliminary finite-field projection transcript and GPU feasibility rows; not a Ring-LPN security claim | `audit_ringlpn_projection_security.py`, focused benchmark commands |

## Reports, newest first

| File | What it is |
|---|---|
| `reports/s2_parameter_novelty_provenance_audit_2026_07_29.md` | **S2 PRELIMINARY HARD-STOP REPORT**: accepted-estimator transcript, published projection-rule/Table-1 contradiction, unproved projected-noise mapping, incomplete epoch budget, DMPF/SLAMP-FSS/Stationary-SD/direct-ring/QA-SD-WHT alternatives, private-project chronology/license/Phase-C overlap, recorded Cheddar pin/notice/local delta, and no pinned parameters/contribution |
| `reports/s2_professor_decision_request_2026_07_29.md` | Exact eight-question advisor request required before S3: system/protocol thesis, multi-point and PCG-architecture choices, cryptographic review, parameter scale, private-project ownership/overlap, and Cheddar disposition |
| `reports/publication_readiness_plan_2026_07_21.md` | **BINDING PUBLICATION ROADMAP**: integrated dealerless Orca FC thesis; advisor-first report; S1--S10 dependency order, security proof and parameter gates, M1--M6 implementation/evaluation criteria, risks, evidence matrix, per-stage user consultation, and required checkpoint commit |
| `reports/dealerless_orca_fc_security_contract_2026_07_29.md` | **S1 CONTRACT FROZEN FOR ADVISOR REVIEW** after the requested Opus 5 model-assisted audit: two-level `F_FC`/`F_DDPF` functionality, exact three-phase DPF and integrated FC transcripts, complete bias/velocity topology, permitted/prohibited leakage, simulators for either corrupted party, position-distribution and proof obligations; S2 now classifies the per-point DPF as a baseline pending advisor direction |
| `reports/session_handoff_2026_07_21.md` | **START HERE for catch-up**: corrected M1 protocol-logic artifact, fresh verification, v2.3 S1 paper state, security boundaries, open milestones |
| `reports/distributed_dpf_keygen_memo_2026_07_21.md` | **Corrected M1 host protocol-logic prototype**: party-separated and functionally validated by unchanged evaluator using ideal OT/triple/OLE and non-cryptographic correctness PRG; 2,432 trees, three OLEs/tree, old-sign regression, 5/5 corruptions, 6/6 invalid inputs, ideal-mask-draw and correlation-reuse controls, executable split accounting (1,908 logical / 3,816 revealed-share bits at depth 14); `dpf/distributed_dpf_keygen_prototype.{csv,log}` |
| `reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex` (+`.pdf`) | **CURRENT internal proposal (v2.3, 2026-07-29)**: S1 contract plus S2 parameter-reduction, DMPF/direct-ring novelty, provenance, and advisor-decision hard stops; no security or publication-readiness claim; stable filename |
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

Conventions: every run produces a `.csv` (data), usually a `.md` (summary), and
a `.log` (raw stdout + stderr). `validation`/`*_contract` columns must read
`pass`; suites exit non-zero on any failure.

**Staleness convention (binding, see `../CLAUDE.md` documentation contract):**
documents whose claims are no longer current carry a `> **HISTORICAL …**`
banner at the top; `outreach/` and `archive/` are wholly historical (see their
READMEs). A document is current only if it is unbannered and dated
2026-06-10 or later. When your work supersedes a document, banner it in the
same commit.
