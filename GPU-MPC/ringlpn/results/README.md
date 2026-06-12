# ringlpn results — directory index

Reorganized 2026-06-10. Every run script writes into its artifact directory
below; nothing writes to this top level anymore.

## Where to look

| Directory | Contents | Produced by |
|---|---|---|
| `reports/` | **Start here.** Current plans, proposals, baselines, memos, handoffs | hand-written |
| `ntt/` | NTT/PolyMul sweeps: CPU (NFLlib), GPU cheddar q32/q64/q128, legacy | `run_sweep.sh`, `run_cuda_sweep.sh`, `run_cuda_sweep_legacy.sh`, `run_cuda_single.sh` |
| `ole/` | Figure 2 Ring-LPN OLE: q64/q128 × uniform/regular, smoke (t=8) + full (t=64) | `run_ole_sweep.sh` |
| `linear_ole/` | Ring-matrix OLE-to-Beaver (2x2x2, n=8192): q64/q128 × uniform/regular | `run_linear_ole_sweep.sh` |
| `vole/` | Standalone VOLE expansion prototype | `run_vole_sweep.sh` |
| `orca_fc/` | Orca FC artifacts: keywriter demo, ideal-OLE transcript, **real-OLE slot-packed transcript**, Zp bridge | `run_orca_fc_ringlpn_demo.sh`, `run_orca_fc_ideal_ole_transcript.sh`, `run_orca_fc_real_ole_transcript.sh`, `run_orca_zp_bridge_test.sh` |
| `secure_convert/` | Secure Z_M -> Z_2^bw conversion prototype | `run_secure_convert_test.sh` |
| `dpf/` | DPF online-keygen memory-efficiency track (chunked variants) | tests/fss bench (manual) |
| `profiling/` | VTune hotspot/memory captures | `run_vtune_*.sh` |
| `outreach/` | Abstracts, posters, professor memos/status emails | hand-written |
| `archive/` | Superseded one-offs: early spot checks, `*_regular_patch`, `*_after_linear`, old plan drafts | frozen |

## Reports, newest first

| File | What it is |
|---|---|
| `reports/dealerless_orca_ringlpn_full_proposal_2026_06_10.tex` | **Proposal to make the linear-layer pipeline fully dealerless** (M1-M6 milestones) |
| `reports/ntt_baseline_comparison_2026_06_10.md` | GPU-NTT external baseline vs cheddar (measured; keep-cheddar decision + revisit triggers) |
| `reports/orca_fc_real_ole_transcript_memo.md` | Real-OLE slot-packed FC transcript (Step 5) + NTT backend changes, 2026-06-10 |
| `reports/baseline_2026_06_10.md` | Verified baseline: environment, all PASS counts, headline perf |
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
