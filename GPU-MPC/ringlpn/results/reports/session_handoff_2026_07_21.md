> **HISTORICAL 2026-07-21 checkpoint handoff.** Superseded by `CLAUDE.md`;
> statements below may describe an older state.

# Session handoff — corrected M1 host artifact + proposal v2.3; S1/S2 update 2026-07-29

**Read after `CLAUDE.md`.** This is the current handoff. The 2026-07-10
handoff is historical.

## 0. Current state

The D1 distributed DPF key-generation protocol logic is implemented
party-separated on the host and emits the standard `spfss_host::DPFKey`
format. The unchanged `spfss_host::dpfEvalAll` functionally validates all
2,432 generated pairs across six configurations. The corrected Phase C uses
three scalar OLEs and opens only the standard public `finalCW`; it no longer
opens the sign that leaked one point bit when conditioned on a party's leaf
control-bit vector.

The executable remains a one-process protocol-logic and compatibility
prototype. Its OT/triple/OLE interfaces are ideal functionalities, and the
unchanged evaluator uses a splitmix64 correctness PRG explicitly labelled
non-cryptographic. Proposal
`dealerless_orca_ringlpn_proposal_v2_2026_07_10.{tex,pdf}` keeps its stable
filename and is now **v2.3, 2026-07-29**.

The corrected artifact and publication direction are committed. The S1
protocol/proof contract passed the user-requested Opus 5 model-assisted audit
with no remaining blocker and is frozen **for advisor review**. This is not an
independent human cryptographic review or a security proof.
The user fixed Alp as the sole paper and commit author. Use the configured Git
identity only; never add `Co-Authored-By` or generated-by trailers. This does
not weaken citation, license, provenance, or overlap-disclosure obligations.
S2 is blocked before S3. The preliminary audit found a contradiction between
BCG+20's literal projection rule and Table 1, no proved mapping from dependent
projected noise to the accepted finite-field estimator, and no complete epoch
budget. The preliminary `n=2^14,c=4,t=16` point leaves only 25% raw capacity
after keygen reserve and is not pinned. Direct 2026 fully distributed DMPF and
SLAMP-FSS work plus Stationary-SD, direct-`Z_(2^k)`, and QA-SD/WHT PCGs require
an architecture decision. Private-project ownership/license/overlap remains
unresolved; Cheddar attribution, reconstructed source pin, local delta, MIT
notice, and citation are recorded. No parameter or protocol contribution is
pinned. See `s2_parameter_novelty_provenance_audit_2026_07_29.md` and
`s2_professor_decision_request_2026_07_29.md`.

## 1. Corrected protocol

For a depth-$L$ tree:

1. **Shared position bits.** A ripple adder converts private arithmetic
   summands into XOR shares of $\alpha$. Their non-wrapping sum is the intended
   triangular exponent distribution of the unreduced product. Cost: $L-1$ bit
   triples, $2(L-1)$ logical opened bits, and $4(L-1)$ revealed-share bits.
2. **Level walk.** Per level, each party expands all current nodes. Off-path
   nodes cancel between the two aggregate views. The secret-bit seed-CW MUX
   uses two 128-bit string OTs; seed and control-bit correction words are
   opened because they are standard key material. Cost: $2L$ string OTs,
   $130L$ logical opened bits, and $260L$ revealed-share bits. The walk has
   $L$ sequential level-synchronous batches independent of tree count; this is
   not an end-to-end network-round claim.
3. **Payload.** Signed leaf aggregates satisfy the DPF cancellation
   invariant. One OLE gives
   $\gamma_0+\gamma_1=\beta_0\beta_1=\beta$. Set
   $d_0=\gamma_0-A_0$, $d_1=\gamma_1-A_1$, $s_0=F_0$, and $s_1=F_1$, so
   $d_0+d_1=\beta-A_0-A_1$ and
   $s_0+s_1=F_0+F_1\in\{+1,-1\}$. Two directional OLEs share $d_0s_1$ and
   $s_0d_1$. Local products plus those cross shares yield
   $w_0+w_1=(d_0+d_1)(s_0+s_1)$. Only
   `finalCW = w0 + w1` is opened: one logical 62-bit field element and two
   revealed 62-bit shares.

The former path opened both $F_b$ values. Their reconstructed sign selects
party 0's control-bit class containing $\alpha$ and excludes the opposite
class. Marginal sign independence was therefore insufficient once
conditioned on the party's output key. The old path and its “safe masked
opening” rationale were removed rather than retained as compatibility code.

Control/tag bits stay separate from seed material, following the formal BGI
seed/tag separation. The S1 target does not derive tags from seed LSBs: it
carries 128 secret seed bits and separate tags.

The deployed Ring-LPN GPU expansion reached that seed-format target on
2026-08-03: four domain-separated AES calls produce full 128-bit child seeds
and separate control bits; device/host parity and GPU key evaluation are
gated. The two-party path uses OpenSSL-private-DRBG roots. The centralized
benchmark keygen still derives roots from one 64-bit `seed_base` and is not a
security realization. DPF distribution, CSPRNG state/composition, and
single-key privacy reductions remain open, so there is still no end-to-end
128-bit DPF-security claim.

## 2. Executable gates and regenerated evidence

Each configuration deterministically covers:

- tree 0: $\alpha=0$, factors $1,p-1$, payload $p-1$;
- tree 1: $\alpha=2^L-2$, factors $p-1,p-1$, payload $1$;
- remaining trees: random inputs with nonzero payload factors;
- eight centralized references through the same evaluator;
- five independent root-seed/`sCW`/`tLCW`/`tRCW`/`finalCW` corruptions
  expected to fail;
- six invalid point/payload encodings expected to abort before correlation;
- `old_sign_opening_leak_control=yes`;
- `ideal_mask_draw_accounting=pass` and `correlation_reuse_control=pass`.

The omniscient old-sign control requires the true $\alpha$ in the selected
party-0 control-bit class and at least one leaf outside that class. It is a
regression model demonstrating the old transcript's distinguishing power,
not a proof of security.

Fresh runner output:

| prime | depth | trees | pass | string OTs | bit triples | scalar OLEs | logical open bits | revealed-share bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p0 | 4  | 512 | 512/512 | 8  | 3  | 3 | 588   | 1,176 |
| p0 | 8  | 512 | 512/512 | 16 | 7  | 3 | 1,116 | 2,232 |
| p0 | 11 | 384 | 384/384 | 22 | 10 | 3 | 1,512 | 3,024 |
| p0 | 14 | 256 | 256/256 | 28 | 13 | 3 | 1,908 | 3,816 |
| p1 | 14 | 256 | 256/256 | 28 | 13 | 3 | 1,908 | 3,816 |
| p1 | 8  | 512 | 512/512 | 16 | 7  | 3 | 1,116 | 2,232 |

Per-run timings remain raw observations in the canonical CSV column
`keygen_plus_eval_us_per_tree`, not paper evidence. Depth-14 runs take roughly
2 ms/tree single-threaded including full-domain validation; small-depth
timings varied by more than $2\times$
across repeated gates, so no performance claim is made from them.

Totals: 2,432/2,432 pass. Every CSV row reports
`centralized_ref_pass=8`, `negctrl_expected_fail=yes`,
`corruption_controls=5/5`, `invalid_inputs_rejected=6/6`,
`old_sign_opening_leak_control=yes`, `transcript_accounting=pass`,
`ideal_mask_draw_accounting=pass`, `correlation_reuse_control=pass`, and
`validation=pass`.

Closed forms:

- string OTs: $2L$;
- bit triples: $L-1$;
- scalar OLEs: $3$;
- logical opened bits:
  $2(L-1)+130L+\lceil\log_2 p\rceil$;
- raw revealed-share bits:
  $4(L-1)+260L+2\lceil\log_2 p\rceil$.

At depth 14 these are 1,908 logical opened bits and 3,816 revealed-share bits.
The prior 3,790 mixed those two metrics and is superseded.

The bootstrap requirement is $3c^2t^2<n$. At $(c,t,n)=(2,8,8192)$, keygen
consumes 768 scalar-OLE slots and has an $8192/768=10.67\times$ output/input
surplus. Table 1's silent-OT estimate excludes the still-unmeasured real OLE
transport.

## 3. Verification record

From `GPU-MPC/ringlpn`:

```bash
./scripts/build_distributed_dpf_keygen.sh
./scripts/run_distributed_dpf_keygen.sh
./scripts/run_paper_checkpoint_smoke.sh
```

Observed:

- focused prototype build: exit 0, no prototype compiler warnings;
- six-row sweep: exit 0, all rows pass with the exact counts above;
- canonical host gate: exit 0,
  `[paper-smoke] HOST GATES PASS (GPU smoke skipped)`;
- `nvidia-smi`: GPUs 0–2 had active compute processes; GPU 3 had none;
- required-GPU gate pinned with `CUDA_VISIBLE_DEVICES=3`: exit 0,
  `[paper-smoke] ALL GATES PASS`.

The exact full-gate command was:

```bash
CUDA_VISIBLE_DEVICES=3 RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 \
  PATH=/usr/local/cuda/bin:$PATH ./scripts/run_paper_checkpoint_smoke.sh
```

This freshly revalidates the full package. It does not upgrade the host D1
artifact from a correctness prototype to a computational-security result.

The v2.3 paper was built twice in a clean ephemeral `ubuntu:22.04` container
with `texlive-latex-base`, `texlive-latex-recommended`, and
`texlive-pictures`; `poppler-utils` supplied only the metadata check. Final
`pdflatex` output had no Overfull/Underfull, undefined-reference,
undefined-citation, package, or other LaTeX warnings. `pdfinfo` reports
**18 pages**, letter size (612 × 792 points). Every rendered page was visually
inspected for clipping, overlap, broken tables/figures, and malformed text.
Auxiliaries were removed, the container was removed, and the PDF was chowned
to `1013:1014`.

## 4. Paper v2.3

The paper now:

- gives the corrected Phase C equations and three-OLE count;
- separates the target AES/real-transport threat model from the host
  artifact's ideal-functionality/non-cryptographic-PRG evidence;
- changes the bootstrap bound to $3c^2t^2<n$ and the smoke surplus to
  $10.67\times$;
- adds scalar OLEs to Table 1 and excludes unmeasured real-OLE transport from
  the silent-OT communication estimate;
- records 72 measured direction switches at depth 11 while leaving
  end-to-end network rounds unclaimed;
- replaces the incoherent 3,790-bit figure with 1,908 logical opened bits and
  3,816 meaningful share bits at depth 14, backed by per-phase executable
  counters;
- cites BCG+20's corrected IACR ePrint 2022/1035 full version,
  §5.2/Remark 5.1;
- adds the S1 ideal functionalities, exact DPF transcript, leakage contract,
  both simulator outlines, and open proof-obligation table without upgrading
  the current security claim;
- adds public FC admissibility bounds, exact modulo-$Q$ conversion semantics,
  source-aligned forward/bias/truncation/`dW`/`dX`/bias-gradient/dual-optimizer
  mask and velocity topology, ideal-correlation consume-once identifiers,
  abort-before-output rules, and the GPU benchmark-root/reduction blockers.
- adds the S2 hard stops: exact `w=c*t` parameter mapping, accepted-estimator
  transcript with its unresolved reduction steps, DMPF/direct-ring prior art,
  and Cheddar/private-project provenance boundaries.

The full S1 contract and corrected state topology increased the PDF from 15 to
18 pages. This is accepted rather than compressing text or shrinking figures.

The standalone S1 review artifact is
`dealerless_orca_fc_security_contract_2026_07_29.md`.

## 5. Claims boundary

**Claimable sentence:** “the distributed key-generation protocol logic is
implemented party-separated and functionally validated by the unchanged
evaluator, using ideal OT/triple/OLE functionalities and a non-cryptographic
correctness PRG.”

The artifact does **not** establish:

- computational privacy;
- 128-bit security;
- M1 completion;
- GPU byte compatibility;
- real transports;
- two-process isolation.

Do not market the 2,432 functional passes as privacy evidence.

## 6. Readiness and remaining milestones

S2 now blocks performance implementation. The advisor decision packet is ready
to show the professor as an internal research-progress artifact; the paper is
not publication-ready or cleared for external circulation. The binding order
is the frozen S1 contract, professor resolution of the S2 parameter/
novelty/provenance hard stops, and only then the reviewed implementation route:

- M5/S2 must pin exact splittable Ring-LPN parameters using a reviewed
  sparse-projection/noise-reduction argument, select per-point DPF versus the
  2026 DMPF and prime-plus-conversion versus direct `Z_(2^k)`, complete the
  source/license inventory, and record professor decisions on contributor
  credit, chronology, reuse, and overlap; Alp remains the sole author;
- M1 then needs independent OS-CSPRNG GPU roots, full-128-bit AES seed
  semantics with separate tags, real silent OT/OLE, GPU batching and
  serialization, and measured network bytes/rounds;
- M2 must wire these keys into the real-OLE GPU transcript;
- M3 must source conversion correlations from the PCG;
- M4 must establish the complete forward/bias/truncation/`dW`/`dX`/
  bias-gradient/dual-optimizer mask-and-velocity handoff and two-process
  isolation before the proof audit;
- M6 must provide model-scale measurements and a closest dealerless baseline;
- the multiplicative-payload adaptation still requires the S8
  publication-grade distribution proof, simulator reduction, and
  implementation audit.

### Direction addendum — 2026-07-29

The user selected the integrated dealerless Orca FC-preprocessing system as
the primary paper thesis. The subsequent S2 audit supersedes the earlier
assumption that the corrected distributed DPF is its enabling protocol
contribution: public DPF/DMPF work already covers the protocol class and the
closest 2026 work directly targets Ring-LPN PCGs. Treat the local DPF as a
compatibility artifact/baseline unless the professor identifies a concrete
delta. The private GPU PCG/PIM project's work has multiple contributors, no
repository license, and unresolved overlap; none of it may be imported or
claimed here. The immediate target remains an advisor-ready technical report,
subject to the full proof, real-transport, parameter, evaluation, provenance,
and artifact gates. Consult the user before every S1--S10 stage.

## 7. Repository and operational notes

- `spfss_host.cpp` is intentionally untouched.
- `scripts/run_paper_checkpoint_smoke.sh` contains exactly one DPF artifact
  build/run pair.
- The corrected artifact, canonical evidence, proposal source/PDF, and current
  documentation are committed at `28f8451`; the ignored DPF CSV and proposal
  PDF were force-added deliberately.
- `publication_readiness_plan_2026_07_21.md` is the binding S1--S10 roadmap.
  Every completed stage must pass its gate and end in an atomic checkpoint
  commit.
- Nothing was sent to the professor; authors, venue template, and upstream
  Orca remain unchanged.
- `/bin/sh` in the TeX containers is dash; do not use brace expansion.
- The current memo is
  `results/reports/distributed_dpf_keygen_memo_2026_07_21.md`.
