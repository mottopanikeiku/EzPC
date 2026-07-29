# Session handoff — 2026-07-21 (corrected M1 host artifact + proposal v2.2)

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
filename and is now **v2.2, 2026-07-21**.

No files were staged or committed, and nothing was sent to the professor.

## 1. Corrected protocol

For a depth-$L$ tree:

1. **Shared position bits.** A ripple adder converts private arithmetic
   summands into XOR shares of $\alpha$. Cost: $L-1$ bit triples and
   $2(L-1)$ opened bits.
2. **Level walk.** Per level, each party expands all current nodes. Off-path
   nodes cancel between the two aggregate views. The secret-bit seed-CW MUX
   uses two 128-bit string OTs; seed and control-bit correction words are
   opened because they are standard key material. Cost: $2L$ string OTs and
   $260L$ opened bits. The walk has $L$ sequential level-synchronous batches
   independent of tree count; this is not an end-to-end network-round claim.
3. **Payload.** Signed leaf aggregates satisfy the DPF cancellation
   invariant. One OLE gives
   $\gamma_0+\gamma_1=\beta_0\beta_1=\beta$. Set
   $d_0=\gamma_0-A_0$, $d_1=\gamma_1-A_1$, $s_0=F_0$, and $s_1=F_1$, so
   $d_0+d_1=\beta-A_0-A_1$ and
   $s_0+s_1=F_0+F_1\in\{+1,-1\}$. Two directional OLEs share $d_0s_1$ and
   $s_0d_1$. Local products plus those cross shares yield
   $w_0+w_1=(d_0+d_1)(s_0+s_1)$. Only
   `finalCW = w0 + w1` is opened: 124 bits for two 62-bit shares.

The former path opened both $F_b$ values. Their reconstructed sign selects
party 0's control-bit class containing $\alpha$ and excludes the opposite
class. Marginal sign independence was therefore insufficient once
conditioned on the party's output key. The old path and its “safe masked
opening” rationale were removed rather than retained as compatibility code.

Control/tag bits stay separate from seed material, following the original
BGI-style construction and avoiding the insecure seed-LSB optimization
identified in the corrected BCG+20 full version, IACR ePrint 2022/1035,
§5.2/Remark 5.1.

## 2. Executable gates and regenerated evidence

Each configuration deterministically covers:

- tree 0: $\alpha=0$, factors $1,p-1$, payload $p-1$;
- tree 1: $\alpha=2^L-2$, factors $p-1,p-1$, payload $1$;
- remaining trees: random inputs with nonzero payload factors;
- eight centralized references through the same evaluator;
- one corrupted-CW expected failure;
- `old_sign_opening_leak_control=yes`.

The omniscient old-sign control requires the true $\alpha$ in the selected
party-0 control-bit class and at least one leaf outside that class. It is a
regression model demonstrating the old transcript's distinguishing power,
not a proof of security.

Fresh runner output:

| prime | depth | trees | pass | string OTs | bit triples | scalar OLEs | opened bits | µs/tree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p0 | 4  | 512 | 512/512 | 8  | 3  | 3 | 1,170 | 2.4 |
| p0 | 8  | 512 | 512/512 | 16 | 7  | 3 | 2,218 | 37.2 |
| p0 | 11 | 384 | 384/384 | 22 | 10 | 3 | 3,004 | 223.7 |
| p0 | 14 | 256 | 256/256 | 28 | 13 | 3 | 3,790 | 1,764.7 |
| p1 | 14 | 256 | 256/256 | 28 | 13 | 3 | 3,790 | 1,763.9 |
| p1 | 8  | 512 | 512/512 | 16 | 7  | 3 | 2,218 | 27.8 |

Totals: 2,432/2,432 pass. Every CSV row reports
`centralized_ref_pass=8`, `negctrl_expected_fail=yes`,
`old_sign_opening_leak_control=yes`, and `validation=pass`.

Closed forms:

- string OTs: $2L$;
- bit triples: $L-1$;
- scalar OLEs: $3$;
- opened bits: $2(L-1)+260L+124$.

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

The v2.2 paper was built in the documented ephemeral `debian:bookworm`
container with only `texlive-latex-base`, `texlive-latex-recommended`, and
`texlive-pictures`. Final `pdflatex` output had no Overfull/Underfull,
undefined-reference, undefined-citation, or other LaTeX warnings. `pdfinfo`
reports **15 pages**, letter size (612 × 792 points). Auxiliaries were removed,
the container was removed, and the PDF was chowned to `1013:1014`.

## 4. Paper v2.2

The paper now:

- gives the corrected Phase C equations and three-OLE count;
- separates the target AES/real-transport threat model from the host
  artifact's ideal-functionality/non-cryptographic-PRG evidence;
- changes the bootstrap bound to $3c^2t^2<n$ and the smoke surplus to
  $10.67\times$;
- adds scalar OLEs to Table 1 and excludes unmeasured real-OLE transport from
  the silent-OT communication estimate;
- narrows the round claim to 14 sequential tree-walk batches and leaves
  end-to-end rounds to M1 measurement;
- regenerates Table 5 to 3 OLEs and 1,170/2,218/3,004/3,790 opened bits;
- cites BCG+20's corrected IACR ePrint 2022/1035 full version,
  §5.2/Remark 5.1;
- updates the abstract, contributions, M1 status, claims ladder, summary, and
  bibliography.

The PDF grew from 14 to 15 pages. This is accepted rather than compressing or
shrinking figures to preserve the old count.

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

After the required full GPU gate, clean paper build, rendered review, and
claims-drift search all pass, v2.2 is appropriate to show the professor as a
research-progress artifact and proposal. It is not publication-ready:

- M1 still needs AES/CSPRNG evaluation, real silent OT/OLE, GPU batching and
  serialization, and measured network bytes/rounds;
- M2 must wire these keys into the real-OLE GPU transcript;
- M3 must source conversion correlations from the PCG;
- M4 must establish two-process isolation;
- M5 must audit splittable Ring-LPN parameters;
- M6 must provide model-scale measurements;
- the multiplicative-payload adaptation still lacks a publication-grade
  simulation proof.

### Direction addendum — 2026-07-29

The user selected the integrated dealerless Orca FC-preprocessing system as
the primary paper thesis. The corrected distributed DPF is its enabling
protocol contribution. GPU PCG design/performance belongs to separate
forthcoming work with a PIM-architecture comparison and must not be claimed as
new here. The immediate publication target is an advisor-ready technical
report while retaining the full proof, real-transport, parameter, evaluation,
and artifact gates. Work remains ringlpn-first; proposed external crypto
dependencies and minimal future upstream Orca integration require review.
Consult the user before every S1--S10 stage.

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
