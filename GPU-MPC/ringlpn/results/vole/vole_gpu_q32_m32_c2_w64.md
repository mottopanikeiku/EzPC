# Ring-LPN VOLE GPU Sweep (Requested q=32)

Generated: 2026-04-20 08:18 UTC

## Results

| n | log2(n) | q req | q actual | m | c | noise wt | validate | iters | x mean (us) | y mean (us) | z mean (us) | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | 32 | 2 | 64 | pass | 200 | 59.087 | 59.076 | 59.079 | 191.485 | 5.984 | 167114.92 | 1002689.51 |
| 16384 | 14 | 32 | 30 | 32 | 2 | 64 | pass | 200 | 112.068 | 112.147 | 112.082 | 350.944 | 10.967 | 91182.64 | 547095.83 |
| 32768 | 15 | 32 | 30 | 32 | 2 | 64 | pass | 200 | 222.655 | 222.491 | 221.462 | 681.873 | 21.309 | 46929.56 | 281577.36 |
| 65536 | 16 | 32 | 30 | 32 | 2 | 64 | pass | 100 | 203.034 | 202.637 | 203.440 | 626.172 | 19.568 | 51104.17 | 306625.02 |
| 131072 | 17 | 32 | 30 | 32 | 2 | 64 | pass | 100 | 985.359 | 983.248 | 982.725 | 2966.810 | 92.713 | 10786.00 | 64715.97 |
| 262144 | 18 | 32 | 30 | 32 | 2 | 64 | pass | 40 | 2768.370 | 2749.860 | 2750.620 | 8286.620 | 258.957 | 3861.65 | 23169.88 |
| 524288 | 19 | 32 | 30 | 32 | 2 | 64 | pass | 20 | 5335.770 | 5330.620 | 5333.980 | 16016.100 | 500.503 | 1997.99 | 11987.94 |
| 1048576 | 20 | 32 | 30 | 32 | 2 | 64 | pass | 10 | 10717.700 | 10705.500 | 10705.200 | 32144.700 | 1004.522 | 995.50 | 5972.99 |

## Notes

- This sweep covers requested qbits 32 and realizes them with actual qbits 30 on the promoted single-prime GPU path.
- Input mode for this benchmark is synthetic_mpvole; the harness synthesizes MPVOLE-consistent inputs locally and validates the relation z = y + x * Delta coefficient-wise.
- These runs use m in {32}, c in {2}, and noise weight in {64}.
- Full expand mean is the end-to-end batch latency for computing x, y, and z across all m outputs for one sampled Delta.
- Per-output expand divides Full expand mean by m. Outputs/s measures correlated output polynomials produced per second.
- Pair PolyMuls/s is a work proxy derived from 3 * m * c polynomial multiplications per full expand batch.
- This benchmark isolates the algebraic expansion layer. SPFSS key generation and evaluation are still external to this harness.
