# Distributed DPF keygen — corrected M1 host protocol-logic prototype (2026-07-21)

**Claimable sentence:** “the distributed key-generation protocol logic is
implemented party-separated and functionally validated by the unchanged
evaluator, using ideal OT/triple/OLE functionalities and a non-cryptographic
correctness PRG.”

## Artifact and protocol

Files:

- `src/test_distributed_dpf_keygen.cpp`
- `scripts/build_distributed_dpf_keygen.sh`
- `scripts/run_distributed_dpf_keygen.sh`
- `results/dpf/distributed_dpf_keygen_prototype.{csv,log}`

Two party structs hold disjoint state. Every cross-party value flows through
a counted ideal functionality or an explicit counted opening. For a tree of
depth $L$:

1. **Phase A — shared position bits.** A secure ripple adder converts private
   summands `off0`, `off1` into XOR-shared bits of
   $\alpha=\mathrm{off0}+\mathrm{off1}$. Cost: $L-1$ Beaver bit triples and
   $2(L-1)$ opened bits. This is the concrete integration choice for
   arithmetic position shares, not a claim of ripple-adder novelty.
2. **Phase B — level-synchronous tree walk.** Each party expands every current
   node and XORs its left/right seed and control-bit aggregates. Off-path
   nodes cancel across parties. A secret-bit MUX produces the seed correction
   word using two 128-bit string OTs per level; the parties open only the
   standard seed and control-bit correction words. Cost: $2L$ string OTs and
   $260L$ opened bits. Control/tag bits remain separate from seed material,
   following the original BGI-style construction and avoiding the insecure
   seed-LSB optimization identified in BCG+20's corrected full version,
   IACR ePrint 2022/1035, §5.2/Remark 5.1.
3. **Phase C — payload correction word.** Let the signed leaf aggregates be
   $A_0,A_1,F_0,F_1$. The first of three scalar OLEs produces additive shares
   $\gamma_0+\gamma_1=\beta_0\beta_1=\beta$. Define
   $d_0=\gamma_0-A_0$, $d_1=\gamma_1-A_1$, $s_0=F_0$, and $s_1=F_1$, so
   $d_0+d_1=\beta-A_0-A_1$ and
   $s_0+s_1=F_0+F_1\in\{+1,-1\}$. Two directional scalar OLEs share
   $d_0s_1$ and $s_0d_1$. With the local products, the parties obtain
   $w_0+w_1=(d_0+d_1)(s_0+s_1)$ and open only
   `finalCW = w0 + w1`, which is already present in each standard output key.
   They do not open $d_b$, $s_b$, or the sign.

The predecessor transcript opened both $F_b$ values. Marginal independence
of $F_0+F_1$ from $\alpha$ was insufficient: conditioned on party 0's leaf
control-bit vector, the sign selects the control-bit class containing the
secret point and excludes the opposite class. The corrected protocol removes
that opening.

Output remains the standard `spfss_host::DPFKey` format. The independent
consumer is the **unchanged** `spfss_host::dpfEvalAll`; `spfss_host.cpp` was
not modified.

## Executable controls

Every generated pair is evaluated over the full domain by
`spfss_host::dpfEvalAll` and must sum to $\beta[x=\alpha]`. Each configuration
also requires:

- eight centralized `dpfGen` references to pass through the same evaluator;
- a corrupted correction word to fail;
- deterministic tree 0 with $\alpha=0$, factors $1,p-1$, payload $p-1$;
- deterministic tree 1 with $\alpha=2^L-2$, factors $p-1,p-1$, payload $1$;
- `old_sign_opening_leak_control=yes`.

The old-sign control runs only in the omniscient test harness. It reconstructs
the removed sign, expands party 0's leaf control-bit class, verifies that the
real point is selected, and requires at least one leaf outside the selected
class. It demonstrates that the former transcript had distinguishing power;
it is a regression model, not a security proof.

## Regenerated results

`./scripts/build_distributed_dpf_keygen.sh &&
./scripts/run_distributed_dpf_keygen.sh` completed without prototype compiler
warnings and produced:

| prime | depth | trees | pass | string OTs | bit triples | scalar OLEs | opened bits | µs/tree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p0 | 4  | 512 | 512/512 | 8  | 3  | 3 | 1,170 | 2.4 |
| p0 | 8  | 512 | 512/512 | 16 | 7  | 3 | 2,218 | 37.2 |
| p0 | 11 | 384 | 384/384 | 22 | 10 | 3 | 3,004 | 223.7 |
| p0 | 14 | 256 | 256/256 | 28 | 13 | 3 | 3,790 | 1,764.7 |
| p1 | 14 | 256 | 256/256 | 28 | 13 | 3 | 3,790 | 1,763.9 |
| p1 | 8  | 512 | 512/512 | 16 | 7  | 3 | 2,218 | 27.8 |

Totals: 2,432/2,432 functional passes. Every row has
`centralized_ref_pass=8`, `negctrl_expected_fail=yes`,
`old_sign_opening_leak_control=yes`, and `validation=pass`.

Closed forms per tree:

- string OTs: $2L$;
- bit triples: $L-1$;
- scalar OLEs: $3$;
- opened bits: $2(L-1)+260L+124$.

At depth 14, the opening count is 3,790 bits. The bootstrap condition is
$3c^2t^2<n$; for $(c,t,n)=(2,8,8192)$, 768 scalar-OLE slots are consumed and
the output/input surplus is $8192/768=10.67\times$.

The fresh host-only gate ended
`[paper-smoke] HOST GATES PASS (GPU smoke skipped)`. After confirming GPU 3
had no active compute process, the required-GPU command

```bash
CUDA_VISIBLE_DEVICES=3 RUN_GPU_SMOKE=1 REQUIRE_GPU_SMOKE=1 \
  PATH=/usr/local/cuda/bin:$PATH ./scripts/run_paper_checkpoint_smoke.sh
```

exited 0 and ended `[paper-smoke] ALL GATES PASS`. The full package is
therefore freshly revalidated; this GPU result does not change the host D1
artifact's security boundary.

## Security boundary

The executable uses ideal OT/triple/OLE interfaces and the unchanged
evaluator's splitmix64 correctness PRG, explicitly labelled
non-cryptographic. It proves functional compatibility, edge correctness, and
primitive accounting. It does **not** establish:

- computational privacy;
- 128-bit security;
- M1 completion;
- GPU byte compatibility;
- real OT/OLE transports;
- two-process isolation.

M1 still requires an AES/CSPRNG evaluator, real silent OT/OLE, GPU
level-synchronous batching, the GPU key byte format, and measured network
bytes and rounds.

## Reproduce and paper status

From `GPU-MPC/ringlpn`:

```bash
./scripts/build_distributed_dpf_keygen.sh
./scripts/run_distributed_dpf_keygen.sh
./scripts/run_paper_checkpoint_smoke.sh
```

The build/run pair is wired exactly once into the host section of
`run_paper_checkpoint_smoke.sh`. Proposal
`dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex` keeps its stable filename
and is now v2.2 (2026-07-21), with the corrected Phase C equations, costs,
security boundary, Table 5, M1 status, and claims ladder.

The root `.gitignore` ignores CSV/PDF files. If a later explicit commit is
requested, force-add the generated CSV and PDF; this session does not stage or
commit them.
