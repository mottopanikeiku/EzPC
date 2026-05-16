# Ring-LPN VOLE GPU Sweep (Requested q=128)

Generated: 2026-05-16 20:22 UTC

## Results

| n | log2(n) | q req | q actual | m | c | noise wt | validate | iters | x mean (us) | y mean (us) | z mean (us) | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 128 | 124 | 2 | 2 | 8 | pass | 200 | 42.996 | 42.960 | 42.978 | 143.642 | 71.821 | 13923.50 | 83541.03 |
| 16384 | 14 | 128 | 124 | 2 | 2 | 8 | pass | 200 | 49.335 | 49.423 | 49.325 | 162.659 | 81.329 | 12295.66 | 73773.97 |
| 32768 | 15 | 128 | 124 | 2 | 2 | 8 | pass | 200 | 62.958 | 62.876 | 62.906 | 203.215 | 101.608 | 9841.79 | 59050.76 |
| 65536 | 16 | 128 | 124 | 2 | 2 | 8 | pass | 100 | 70.540 | 70.350 | 70.371 | 225.832 | 112.916 | 8856.14 | 53136.85 |
| 131072 | 17 | 128 | 124 | 2 | 2 | 8 | pass | 100 | 187.603 | 187.431 | 187.446 | 577.727 | 288.863 | 3461.84 | 20771.06 |
| 262144 | 18 | 128 | 124 | 2 | 2 | 8 | pass | 40 | 395.687 | 396.707 | 396.414 | 1205.050 | 602.525 | 1659.68 | 9958.09 |
| 524288 | 19 | 128 | 124 | 2 | 2 | 8 | pass | 20 | 796.387 | 799.370 | 799.394 | 2410.550 | 1205.275 | 829.69 | 4978.12 |
| 1048576 | 20 | 128 | 124 | 2 | 2 | 8 | pass | 10 | 2062.160 | 2072.120 | 2074.370 | 6223.630 | 3111.815 | 321.36 | 1928.14 |

## Notes

- This sweep covers requested qbits 128 and realizes them with actual qbits 124 on the promoted Cheddar GPU path; q=128 uses two q62 CRT prime limbs.
- Input mode for this benchmark is synthetic_mpvole; the harness synthesizes MPVOLE-consistent inputs locally and validates the relation z = y + x * Delta coefficient-wise.
- These runs use m in {2}, c in {2}, and noise weight in {8}.
- Full expand mean is the end-to-end batch latency for computing x, y, and z across all m outputs for one sampled Delta.
- Per-output expand divides Full expand mean by m. Outputs/s measures correlated output polynomials produced per second.
- Pair PolyMuls/s is a work proxy derived from 3 * m * c polynomial multiplications per full expand batch.
- This benchmark isolates the algebraic expansion layer. SPFSS key generation and evaluation are still external to this harness.
