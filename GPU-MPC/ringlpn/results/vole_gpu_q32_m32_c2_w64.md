# Ring-LPN VOLE GPU Sweep (Requested q=32)

Generated: 2026-04-08 08:36 UTC

## Results

| n | log2(n) | q req | q actual | m | c | noise wt | validate | iters | x mean (us) | y mean (us) | z mean (us) | Full expand mean (us) | Per-output expand (us) | Outputs/s | Pair PolyMuls/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | 32 | 2 | 64 | pass | 200 | 84.642 | 84.626 | 84.634 | 269.484 | 8.421 | 118745.45 | 712472.73 |
| 16384 | 14 | 32 | 30 | 32 | 2 | 64 | pass | 200 | 162.765 | 162.879 | 162.869 | 504.242 | 15.758 | 63461.59 | 380769.55 |
| 32768 | 15 | 32 | 30 | 32 | 2 | 64 | pass | 200 | 316.019 | 315.774 | 315.135 | 963.231 | 30.101 | 33221.52 | 199329.13 |
| 65536 | 16 | 32 | 30 | 32 | 2 | 64 | pass | 100 | 247.008 | 246.283 | 246.503 | 757.046 | 23.658 | 42269.56 | 253617.35 |
| 131072 | 17 | 32 | 30 | 32 | 2 | 64 | pass | 100 | 1397.880 | 1396.440 | 1395.760 | 4214.560 | 131.705 | 7592.73 | 45556.36 |
| 262144 | 18 | 32 | 30 | 32 | 2 | 64 | pass | 40 | 3692.160 | 3670.150 | 3670.790 | 11058.700 | 345.584 | 2893.65 | 17361.90 |
| 524288 | 19 | 32 | 30 | 32 | 2 | 64 | pass | 20 | 7240.820 | 7229.130 | 7233.720 | 21721.100 | 678.784 | 1473.22 | 8839.33 |
| 1048576 | 20 | 32 | 30 | 32 | 2 | 64 | pass | 10 | 14468.400 | 14450.800 | 14438.200 | 43392.000 | 1356.000 | 737.46 | 4424.78 |

## Notes

- This sweep covers requested qbits 32 and realizes them with actual qbits 30 on the promoted single-prime GPU path.
- Input mode for this benchmark is synthetic_mpvole; the harness synthesizes MPVOLE-consistent inputs locally and validates the relation z = y + x * Delta coefficient-wise.
- These runs use m in {32}, c in {2}, and noise weight in {64}.
- Full expand mean is the end-to-end batch latency for computing x, y, and z across all m outputs for one sampled Delta.
- Per-output expand divides Full expand mean by m. Outputs/s measures correlated output polynomials produced per second.
- Pair PolyMuls/s is a work proxy derived from 3 * m * c polynomial multiplications per full expand batch.
- This benchmark isolates the algebraic expansion layer. SPFSS key generation and evaluation are still external to this harness.
