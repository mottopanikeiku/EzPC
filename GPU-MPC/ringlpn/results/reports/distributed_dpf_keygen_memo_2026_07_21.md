# Distributed DPF keygen — corrected M1 host protocol-logic prototype (2026-07-21; S1 accounting correction 2026-07-29)

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
a counted ideal functionality or an explicit counted opening. Every ideal call
also carries a tree/phase/ordinal correlation ID recorded in a consume-once
ledger; reuse is rejected before a functionality returns output. For a tree of
depth $L$:

1. **Phase A — shared position bits.** A secure ripple adder converts private
   summands `off0`, `off1` into XOR-shared bits of
   $\alpha=\mathrm{off0}+\mathrm{off1}$. The non-wrapping sum is the intended
   triangular exponent distribution of the unreduced polynomial product:
   uniform-noise positions lie in $[0,n)$ and regular-noise offsets lie in one
   public bucket pair. Cost: $L-1$ Beaver bit triples, $2(L-1)$ logical opened
   bits, and $4(L-1)$ meaningful share bits. This is the concrete integration
   choice for arithmetic position shares, not a claim of ripple-adder novelty.
2. **Phase B — level-synchronous tree walk.** Each party expands every current
   node and XORs its left/right seed and control-bit aggregates. Off-path
   nodes cancel across parties. A secret-bit MUX produces the seed correction
   word using two 128-bit string OTs per level; the parties open only the
   standard seed and control-bit correction words. Cost: $2L$ string OTs,
   $130L$ logical opened bits, and $260L$ meaningful share bits. Control/tag
   bits remain separate from seed material, following the formal BGI seed/tag
   separation. The deployed GPU path reached full 128-bit seeds with separate
   domain-separated tag outputs on 2026-08-03; this ideal host artifact still
   uses its labelled splitmix correctness reference.
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
   This is one logical opened field element and two meaningful field-share
   widths: respectively $\lceil\log_2p\rceil$ and
   $2\lceil\log_2p\rceil$ bits.

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
- five independent corruptions (root seed, `sCW`, `tLCW`, `tRCW`, and
  `finalCW`) to fail;
- six invalid point/payload encodings to abort before consuming correlation;
- deterministic tree 0 with $\alpha=0$, factors $1,p-1$, payload $p-1$;
- deterministic tree 1 with $\alpha=2^L-2$, factors $p-1,p-1$, payload $1$;
- `old_sign_opening_leak_control=yes`;
- `ideal_mask_draw_accounting=pass` and `correlation_reuse_control=pass`.

The old-sign control runs only in the omniscient test harness. It reconstructs
the removed sign, expands party 0's leaf control-bit class, verifies that the
real point is selected, and requires at least one leaf outside the selected
class. It demonstrates that the former transcript had distinguishing power;
it is a regression model, not a security proof.

## Regenerated results

`./scripts/build_distributed_dpf_keygen.sh &&
./scripts/run_distributed_dpf_keygen.sh` completed without prototype compiler
warnings and produced:

| prime | depth | trees | pass | string OTs | bit triples | scalar OLEs | logical open bits | meaningful share bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p0 | 4  | 512 | 512/512 | 8  | 3  | 3 | 588   | 1,176 |
| p0 | 8  | 512 | 512/512 | 16 | 7  | 3 | 1,116 | 2,232 |
| p0 | 11 | 384 | 384/384 | 22 | 10 | 3 | 1,512 | 3,024 |
| p0 | 14 | 256 | 256/256 | 28 | 13 | 3 | 1,908 | 3,816 |
| p1 | 14 | 256 | 256/256 | 28 | 13 | 3 | 1,908 | 3,816 |
| p1 | 8  | 512 | 512/512 | 16 | 7  | 3 | 1,116 | 2,232 |

Per-run timings remain raw observations in the
`keygen_plus_eval_us_per_tree` column of
`results/dpf/distributed_dpf_keygen_prototype.csv`, not paper evidence.
Depth-14 runs take roughly 2 ms/tree single-threaded including full-domain validation;
small-depth timings varied by more than $2\times$ across repeated gates, so no
performance claim is made from them.

Totals: 2,432/2,432 functional passes. Every row has
`centralized_ref_pass=8`, `negctrl_expected_fail=yes`,
`corruption_controls=5/5`, `invalid_inputs_rejected=6/6`,
`old_sign_opening_leak_control=yes`, `transcript_accounting=pass`,
`ideal_mask_draw_accounting=pass`, `correlation_reuse_control=pass`, and
`validation=pass`.

Closed forms per tree:

- string OTs: $2L$;
- bit triples: $L-1$;
- scalar OLEs: $3$;
- logical opened bits:
  $2(L-1)+130L+\lceil\log_2 p\rceil$;
- meaningful share bits:
  $4(L-1)+260L+2\lceil\log_2 p\rceil$.

At depth 14 and either 62-bit prime these are 1,908 logical opened bits and
3,816 meaningful share bits. The earlier 3,790 figure mixed Phase A's logical
openings with Phases B/C's share widths and is superseded. Neither corrected
counter includes byte padding, OT/OLE payloads, or framing; the real transport
artifact separately measures bytes and direction switches.
The bootstrap condition is $3c^2t^2<n$; for $(c,t,n)=(2,8,8192)$, 768
scalar-OLE slots are consumed and the output/input surplus is
$8192/768=10.67\times$.

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

The S1 functionality, exact transcript, leakage contract, simulators, and open
proof obligations are in
`dealerless_orca_fc_security_contract_2026_07_29.md`.

## Security boundary

The executable uses ideal OT/triple/OLE interfaces and the unchanged
evaluator's splitmix64 correctness PRG, explicitly labelled
non-cryptographic. It proves functional compatibility, edge correctness, and
primitive accounting. This ideal host artifact alone does **not** establish:

- computational privacy or 128-bit security;
- M1 completion;
- GPU compatibility;
- real transport; or
- two-process isolation.

**Current-status note (2026-08-04):** the separate live forward-FC path now
consumes full-width four-call GPU-AES-compatible distributed keys generated
over real SCI/IKNP/Gilboa transport with OpenSSL-private roots, then performs
party-local Ring-LPN expansion and exact conversion across two processes and
GPUs. The security contract gives the exact correction-word coupling and
role-specific hybrid simulators. Still open are silent OT, GPU-side batched
key generation, concrete DPF/PRG and Ring-LPN parameter review, authenticated
deployment, and actual dependency-round measurement.

## Reproduce and paper status

From `GPU-MPC/ringlpn`:

```bash
./scripts/build_distributed_dpf_keygen.sh
./scripts/run_distributed_dpf_keygen.sh
./scripts/run_paper_checkpoint_smoke.sh
```

The build/run pair is wired exactly once into the host section of
`run_paper_checkpoint_smoke.sh`. Proposal source
`dealerless_orca_ringlpn_proposal_v2_2026_07_10.tex` keeps its stable filename
and is now v2.5 (2026-08-04). The checked-in PDF is the matching warning-free,
page-inspected 21-page rendering.

The root `.gitignore` ignores CSV/PDF files; checkpoint commits must
deliberately force-add regenerated evidence that is part of the documented
claim.
