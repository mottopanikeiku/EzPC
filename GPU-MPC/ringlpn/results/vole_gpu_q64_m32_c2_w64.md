# Ring-LPN VOLE GPU Sweep (Requested q=64)

Generated: 2026-04-20 08:19 UTC

## Results

| n | log2(n) | q req | q actual | m | c | noise wt | validate | iters | x mean (us) | y mean (us) | z mean (us) | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 64 | 62 | 32 | 2 | 64 | pass | 200 | 178.396 | 178.413 | 178.432 | 549.802 | 17.181 | 58202.77 | 349216.63 |
| 16384 | 14 | 64 | 62 | 32 | 2 | 64 | pass | 200 | 228.715 | 229.491 | 229.230 | 702.911 | 21.966 | 45524.97 | 273149.80 |
| 32768 | 15 | 64 | 62 | 32 | 2 | 64 | pass | 200 | 384.284 | 383.792 | 384.338 | 1169.510 | 36.547 | 27361.89 | 164171.32 |
| 65536 | 16 | 64 | 62 | 32 | 2 | 64 | pass | 100 | 574.454 | 572.316 | 572.054 | 1735.660 | 54.239 | 18436.79 | 110620.74 |
| 131072 | 17 | 64 | 62 | 32 | 2 | 64 | pass | 100 | 2020.030 | 2010.170 | 2000.340 | 6047.390 | 188.981 | 5291.54 | 31749.23 |
| 262144 | 18 | 64 | 62 | 32 | 2 | 64 | pass | 40 | 4187.240 | 4196.770 | 4204.290 | 12604.900 | 393.903 | 2538.70 | 15232.17 |
| 524288 | 19 | 64 | 62 | 32 | 2 | 64 | pass | 20 | 8413.860 | 8433.120 | 8431.390 | 25295.400 | 790.481 | 1265.05 | 7590.31 |
| 1048576 | 20 | 64 | 62 | 32 | 2 | 64 | pass | 10 | 16934.000 | 17001.500 | 16999.300 | 50952.700 | 1592.272 | 628.03 | 3768.20 |

## Notes

- This sweep covers requested qbits 64 and realizes them with actual qbits 62 on the promoted single-prime GPU path.
- Input mode for this benchmark is synthetic_mpvole; the harness synthesizes MPVOLE-consistent inputs locally and validates the relation z = y + x * Delta coefficient-wise.
- These runs use m in {32}, c in {2}, and noise weight in {64}.
- Full expand mean is the end-to-end batch latency for computing x, y, and z across all m outputs for one sampled Delta.
- Per-output expand divides Full expand mean by m. Outputs/s measures correlated output polynomials produced per second.
- Pair PolyMuls/s is a work proxy derived from 3 * m * c polynomial multiplications per full expand batch.
- This benchmark isolates the algebraic expansion layer. SPFSS key generation and evaluation are still external to this harness.
