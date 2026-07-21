# Session handoff — 2026-07-10 (documents-only session)

> **HISTORICAL (superseded by `session_handoff_2026_07_21.md` and
> `CLAUDE.md`).** Statements below may describe an older state. Retained as
> archival context because the proposal-restructure rationale and explainer
> notes remain useful.

**Scope of this session:** no code, no results, no gates touched. Two
deliverables: (1) the restructured **proposal v2**, (2) a defense-depth
upgrade of the personal explainers in `/home/fatih/EzPC/explainers/`. Plus
the documentation-contract updates that go with them. Everything below is
already done; the "open items" section at the end is what's left for next
sessions.

## 1. Why this session happened

The advisor reviewed the 2026-06-10 proposal and asked for:

1. A clearer structure: **(1) problem overview** (current Orca status, why
   not fully dealerless), **(2) motivation** (dealerless + GPU acceleration),
   **(3) proposed design** and how it achieves fully-dealerless on GPU,
   **(4) experiment setup**.
2. Fixing the "Figure 2" confusion: the draft said "Figure 2" everywhere
   (meaning Figure 2 *of BCG+20*) but contained no figures at all.
3. Generally: publication-quality writing.

Separately, the user (who must present and defend this work, learning the
field from zero) asked for the explainers to go deep enough to support a
full defense of the proposal.

## 2. Deliverable A: proposal v2

**File:** `results/reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex`
(+ compiled `.pdf`, 12 pages, 0 overfull boxes, 0 undefined references).
The 2026-06-10 draft got the HISTORICAL banner (as `%` comments at its top)
and stays for the record.

**Structure (matches the advisor's four parts):**
- §1 Problem overview — Orca background, dealer architecture (Figure 1),
  current pipeline status, gaps **G1–G5** (each a code-level statement),
  contributions list.
- §2 Motivation — why dealerless; why PCGs; the **BCG+20 generator**
  walked step-by-step; slot packing; the measured case for GPUs; a notation
  table (composite modulus renamed **Q = p0·p1** to stop colliding with the
  layer dimension M; ring dim n vs layer N disambiguated).
- §3 Design — pipeline figure (Figure 2 — ours, deliberately), threat model,
  components **D1–D5** (D1 distributed SPFSS keygen, D2 PCG-sourced
  conversion, D3 coin-tossed seeds, D4 two-process deployment, D5 parameter
  audit), a composition table (Table 3: G→D→gate→claim), per-layer cost model
  with a worked 16×32×16 example.
- §4 Experiment setup — platform, validation methodology (independent
  oracles, party separation, the ALL-GATES-PASS one-command gate), the
  measured baseline (Table 4 = the 9/9 suite; perf anchors), milestones
  M1–M6 as experiments with mechanical gates (Figure 3 = dependency DAG),
  claims ladder.
- §5 Related work, §6 Summary. Bibliography extended with Beaver'91 and
  BGI'15 (FSS).

**"Figure 2" fix:** every reference to BCG+20's Figure 2 is now "the BCG+20
generator", with one footnote explaining that the codebase names the engine
`figure2` after that paper's figure number. The document contains three real
TikZ figures (dealer architecture / pipeline with oracle boundaries marked /
milestone DAG) — all visually inspected from rendered PNGs.

**Two substantive analytical changes vs the June draft (not just prose):**

1. **Tree-accurate keygen costs (Table 1).** The June draft's regular-noise
   row priced c²(2t−1)=60 SPFSS *instances* × λ·log(2n) = 107,520 OT bits.
   What the code (`build_spfss_keys()`, `spfss_group_count()`) actually does:
   regular noise keeps **t² trees per product pair** (256 at c=2,t=8), grouped
   into 2t−1 diagonal windows over domain 2n/t — trees get *shallower* (depth
   11 vs 14), not fewer. Tree-accurate: 360,448 OT bits at t=8; 16.8M at
   t=64. v2 Table 1 carries these with a footnote recording the correction.
   Regular noise's decisive win is expansion time (881→61 ms measured) and
   key bytes (283→233 kB), not OT volume.
2. **Bootstrap self-sustainability condition (eq. 1): c²t² < n.** Keygen
   consumes one scalar OLE per DPF tree; expansion yields n slot OLEs per
   ring OLE. Epoch bootstrapping (à la BCG+20) works iff c²t² < n — 32×
   surplus at smoke parameters, but *violated* at t=64, n=8192. This couples
   M5 to M1: the parameter audit must grow n alongside t (literature points
   use n ≥ 2^16). M5's gate now includes re-checking eq. 1.

**How to recompile:** no LaTeX exists on host or in any local image. Use the
ephemeral-container recipe now recorded in `CLAUDE.md` (Environment &
gotchas): debian:bookworm + texlive-latex-base/-recommended/-pictures
(~300 MB; do NOT use enumitem — that pulls texlive-latex-extra; the document
avoids it), `pdflatex` twice, clean root-owned aux files via a root
container, chown the PDF to 1013:1014.

## 3. Deliverable B: explainer deep-dive (outside GPU-MPC/, in `EzPC/explainers/`)

These are the user's personal learning documents; they are NOT covered by the
results/README index but ARE kept consistent with the proposal (contract
rule 5). Two files:

- `dealerless_course.html` ("From Zero to Dealerless", 12 units) — the
  defense-prep course. **This session added Unit 11 "The deep end —
  defending every line"**:
  - M1 mechanics for real: the cancellation lemma (off-path nodes identical →
    party-local level aggregates reconstruct every correction word given only
    the secret direction bit → one 1-of-2 OT per level), payload placement
    from ONE scalar OLE + leaf aggregates, the α-bits subtlety (arithmetic vs
    bitwise shares — the one genuine M1 design point), bootstrap/circularity
    with the c²t² < n inequality.
  - **Widget 6: a live distributed-keygen audit** (domain-16 toy DPF):
    rebuilds every per-level correction word from party-local aggregates +
    the secret bit, and the payload word from one simulated OLE, with
    PASS/FAIL verdicts. The logic was verified in Node BEFORE embedding:
    6,000/6,000 exhaustive checks (all α × 200 trials: per-level cwS/cwTL/cwTR
    reconstruction, cwFin-from-OLE, point-function correctness), and the
    embedded page script re-verified with DOM stubs: all 6 widgets render
    PASS, 400/400 fuzz runs of widget 6.
  - Conversion circuit gate-by-gate with every count in closed form
    (124 = 2(ℓ−1) ANDs, 125 = 2ℓ−1 rounds, 375 = 2ℓ + 2·#AND + 1 opened bits
    — the last verified against the counters in `test_secure_convert.cpp`
    lines ~126/244/263).
  - Splittable-Ring-LPN / quasi-abelian story, ISD in plain words, the
    prime-choice chain (v₂(p−1) headroom to n≈2^23, CRT bound K·2^(2bw+2),
    Barrett-headroom reversal condition).
  - A claims ladder and a 12-question hostile-committee Q&A (distinct from
    Unit 10's friendlier 8).
  - New glossary terms: epoch, gilboa, quasiab, genprop.
- `ringlpn_explained.html` ("The Dealerless Book") — reference volume.
  Chapter 12 updated: points at proposal v2, tree-accurate regular-noise
  arithmetic, cross-reference to course Unit 11.

**Sync with v2 done:** Unit 10's section-by-section table now maps the v2
structure (§ numbers, table numbers, figures); the numbers card and both
"regular noise" quiz answers teach the instance-vs-tree history as defense
material ("we caught it ourselves"); stale §-references fixed (old §3.1→§3.3
etc.). Unit 4's quiz Q2 (which repeated the instance-vs-tree error) was
corrected in the first half of this session.

## 4. State of the documentation contract after this session

- `CLAUDE.md`: Status line updated (2026-07-10, documents-only), catch-up
  item 4 → v2, roadmap section → v2 + D-labels + bootstrap note, new gotcha:
  LaTeX-compile recipe.
- `results/README.md`: rows added for proposal v2 (+pdf) and this handoff;
  June proposal row marked HISTORICAL.
- June proposal `.tex`: HISTORICAL banner prepended (as `%` comments).
- This memo: the per-artifact memo for the v2 proposal AND the session
  handoff (one document, deliberately).

## 5. Open items / suggested next steps (in priority order)

1. **User reviews v2 + presents to advisor.** The professor's feedback loop
   is not closed — v2 has not been seen by the professor yet. Expect another
   iteration; keep the v2 file as the live document and banner-don't-delete
   on any v3.
2. **M1 (distributed DPF keygen prototype)** is the critical path and the
   only real protocol-design unknown (α share-format at the walk's input:
   secure adder per tree vs additive-share walk variant). M1 ∥ M3 ∥ M5.
3. **Gates unchanged:** nothing in `src/` or `scripts/` moved this session;
   the 2026-06-10 baseline (all gates passing, HEAD then 87d84d0-era) is
   still the last verified code state. Re-run
   `RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 scripts/run_paper_checkpoint_smoke.sh`
   before/after any code work as usual.
4. **Nothing is committed yet from this session.** New/modified files (see
   §6). CSVs need `git add -f` if any ever get added; none here. Ask the
   user before committing.
5. Possible polish if requested: number the v2 pages against a venue
   template; add a per-figure alt-text pass; extend Related Work.

## 6. File inventory of this session

Modified:
- `GPU-MPC/ringlpn/CLAUDE.md` (status, catch-up, roadmap, gotcha)
- `GPU-MPC/ringlpn/results/README.md` (two new rows, one HISTORICAL mark)
- `GPU-MPC/ringlpn/results/reports/dealerless_orca_ringlpn_full_proposal_2026_06_10.tex` (banner only)
- `explainers/dealerless_course.html` (Unit 11 + widget 6 + v2 sync + Unit 4/10 fixes)
- `explainers/ringlpn_explained.html` (Ch. 12 paragraph)

New:
- `GPU-MPC/ringlpn/results/reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex`
- `GPU-MPC/ringlpn/results/reports/dealerless_orca_ringlpn_proposal_v2_2026_07_10.pdf`
- `GPU-MPC/ringlpn/results/reports/session_handoff_2026_07_10.md` (this file)

Scratch (session-local, disposable): widget verification harnesses under the
session scratchpad and `/tmp/test_dkg_widget.js`; rendered page PNGs.
