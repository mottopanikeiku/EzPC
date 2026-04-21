# Ring-LPN VOLE GPU Sweep (Requested q=32)

Generated: 2026-04-20 08:24 UTC

## Results

| n | log2(n) | q req | q actual | m | c | noise wt | validate | iters | x mean (us) | y mean (us) | z mean (us) | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | 64 | 2 | 64 | pass | 200 | 101.808 | 101.740 | 101.720 | 319.993 | 5.000 | 200004.38 | 1200026.25 |
| 16384 | 14 | 32 | 30 | 64 | 2 | 64 | pass | 200 | 218.534 | 218.379 | 217.760 | 670.203 | 10.472 | 95493.45 | 572960.73 |
| 32768 | 15 | 32 | 30 | 64 | 2 | 64 | pass | 200 | 468.334 | 467.335 | 468.665 | 1421.340 | 22.208 | 45027.93 | 270167.59 |
| 65536 | 16 | 32 | 30 | 64 | 2 | 64 | pass | 100 | 461.861 | 462.433 | 461.221 | 1400.850 | 21.888 | 45686.55 | 274119.28 |
| 131072 | 17 | 32 | 30 | 64 | 2 | 64 | pass | 100 | 2574.300 | 2558.190 | 2558.720 | 7709.040 | 120.454 | 8301.94 | 49811.65 |
| 262144 | 18 | 32 | 30 | 64 | 2 | 64 | pass | 40 | 5568.210 | 5562.250 | 5564.710 | 16712.000 | 261.125 | 3829.58 | 22977.50 |
| 524288 | 19 | 32 | 30 | 64 | 2 | 64 | pass | 20 | 10610.100 | 10620.200 | 10614.300 | 31860.800 | 497.825 | 2008.74 | 12052.43 |
| 1048576 | 20 | 32 | 30 | 64 | 2 | 64 | pass | 10 | 21371.100 | 21345.900 | 21343.500 | 64076.700 | 1001.198 | 998.80 | 5992.82 |

## Notes

- This sweep covers requested qbits 32 and realizes them with actual qbits 30 on the promoted single-prime GPU path.
- Input mode for this benchmark is synthetic_mpvole; the harness synthesizes MPVOLE-consistent inputs locally and validates the relation z = y + x * Delta coefficient-wise.
- These runs use m in {64}, c in {2}, and noise weight in {64}.
- Full expand mean is the end-to-end batch latency for computing x, y, and z across all m outputs for one sampled Delta.
- Per-output expand divides Full expand mean by m. Outputs/s measures correlated output polynomials produced per second.
- Pair PolyMuls/s is a work proxy derived from 3 * m * c polynomial multiplications per full expand batch.
- This benchmark isolates the algebraic expansion layer. SPFSS key generation and evaluation are still external to this harness.
