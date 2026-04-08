# Ring-LPN VOLE GPU Sweep (Requested q=64)

Generated: 2026-04-08 08:40 UTC

## Results

| n | log2(n) | q req | q actual | m | c | noise wt | validate | iters | x mean (us) | y mean (us) | z mean (us) | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 64 | 62 | 32 | 2 | 64 | pass | 200 | 252.398 | 252.324 | 252.217 | 772.324 | 24.135 | 41433.39 | 248600.33 |
| 16384 | 14 | 64 | 62 | 32 | 2 | 64 | pass | 200 | 336.157 | 335.917 | 336.126 | 1024.820 | 32.026 | 31225.00 | 187349.97 |
| 32768 | 15 | 64 | 62 | 32 | 2 | 64 | pass | 200 | 506.628 | 506.723 | 506.608 | 1537.150 | 48.036 | 20817.75 | 124906.48 |
| 65536 | 16 | 64 | 62 | 32 | 2 | 64 | pass | 100 | 782.340 | 781.145 | 779.558 | 2359.950 | 73.748 | 13559.61 | 81357.66 |
| 131072 | 17 | 64 | 62 | 32 | 2 | 64 | pass | 100 | 2686.700 | 2672.940 | 2663.110 | 8064.960 | 252.030 | 3967.78 | 23806.69 |
| 262144 | 18 | 64 | 62 | 32 | 2 | 64 | pass | 40 | 5536.370 | 5537.040 | 5540.300 | 16634.400 | 519.825 | 1923.72 | 11542.35 |
| 524288 | 19 | 64 | 62 | 32 | 2 | 64 | pass | 20 | 11101.700 | 11119.500 | 11110.200 | 33358.700 | 1042.459 | 959.27 | 5755.62 |
| 1048576 | 20 | 64 | 62 | 32 | 2 | 64 | pass | 10 | 22503.700 | 22451.700 | 22524.700 | 67531.700 | 2110.366 | 473.85 | 2843.11 |

## Notes

- This sweep covers requested qbits 64 and realizes them with actual qbits 62 on the promoted single-prime GPU path.
- Input mode for this benchmark is synthetic_mpvole; the harness synthesizes MPVOLE-consistent inputs locally and validates the relation z = y + x * Delta coefficient-wise.
- These runs use m in {32}, c in {2}, and noise weight in {64}.
- Full expand mean is the end-to-end batch latency for computing x, y, and z across all m outputs for one sampled Delta.
- Per-output expand divides Full expand mean by m. Outputs/s measures correlated output polynomials produced per second.
- Pair PolyMuls/s is a work proxy derived from 3 * m * c polynomial multiplications per full expand batch.
- This benchmark isolates the algebraic expansion layer. SPFSS key generation and evaluation are still external to this harness.
