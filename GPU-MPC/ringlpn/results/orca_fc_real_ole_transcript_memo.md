# Real-OLE slot-packed FC transcript (Step 5) — 2026-06-10

`bench_orca_fc_real_ole_transcript` replaces the ideal-OLE oracle of the
Step-1 transcript with the **real Figure 2 Ring-LPN OLE engine**
(`bench_ole_ringlpn_cuda.cu`, included with `RINGLPN_OLE_DISABLE_MAIN`), and
resolves the constant-polynomial packing objection via **slot packing**: the
deployed primes fully split `X^N+1`, so the forward negacyclic NTT is a ring
isomorphism `R_p -> Z_p^N` and one ring OLE yields up to `N` independent
scalar OLEs. Each Beaver cross term is derandomized against one slot
(`P0` opens `d = a - X0[s]`, `P1` opens `e = b - X1[s]`;
`u0 = de + e*X0[s] + Z0[s]`, `u1 = d*X1[s] + Z1[s]`; `u0+u1 = ab mod p`).

Suite: `orca_fc_real_ole_transcript_transcript_suite.csv` — 9/9 pass
(q64 bw<=16 and q128 bw=32 via two CRT limbs with per-party Garner lift;
uniform and regular noise), every case validated through the **unchanged**
`gpuMatmulBeaver` online path. Headline rows:

| case | ring OLEs | ideal-OLE equiv | slots used | opened Z_p words |
|---|---|---|---|---|
| q64 16x32x16 bw16 | 2 | 16,384 | 8,192 (full) | 32,768 |
| q128 16x16x16 bw32 | 4 | 8,192 | 4,096 | 32,768 |

Ring-OLE count is `2 * limbs * ceil(MKN/n)` — independent of layer size up to
slot capacity. Internal checks: engine `z0+z1 == x0*x1` (GPU + host oracle),
slot identity on used slots, per-cross-term identity, mask consistency, key
byte order.

Remaining oracle boundaries (unchanged, stated in the source header):
centralized SPFSS keygen; `exactZmToRingShares` conversion; c=2/t=8 are
correctness parameters. The removal plan for all three is
`dealerless_orca_ringlpn_full_proposal_2026_06_10.tex`.

## NTT backend changes landed with this artifact

- **Adaptive fused-INTT polymul**: the Hadamard product folds into the INTT
  phase-1 load when `batch*primes <= 16` (one fewer launch + one fewer full
  coefficient-vector round trip). Measured on RTX 5000 Ada at n=8192: ~2–8%
  faster at the small batches the OLE expand uses; the unfused path is kept
  for large batches where the pointwise round trip stays in L2 (q64 batch=64
  was ~3% faster unfused). Overrides: `RINGLPN_NTT_NO_FUSE=1`,
  `RINGLPN_NTT_FORCE_FUSE=1`.
- **Cached forward NTTs in the OLE engine**: `NTT(a)` and `NTT(a_i*a_j)` are
  computed once per instance; x/z phases use `run_polymul_prepared_lhs`,
  halving forward-NTT work per expand iteration.
- **Measured outcome**: polymul validation passes everywhere
  (2^13–2^20, q32/64/128, both fusion modes); OLE expand time is unchanged
  (13.3 ms q64 / 26.8 ms q128 smoke; 881 ms t=64 uniform; 61 ms t=64
  regular) — confirming the documented bottleneck: SPFSS full-domain
  evaluation dominates, NTT is <1% of expand. This is why the dealerless
  proposal spends its budget on the SPFSS/OT side.

Reproduce:

```bash
bash scripts/build_orca_fc_real_ole_transcript.sh
bash scripts/run_orca_fc_real_ole_transcript.sh
```
