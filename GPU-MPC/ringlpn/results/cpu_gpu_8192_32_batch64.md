# CPU vs GPU NTT Comparison (n=8192, q req CPU/GPU=32/32)

Generated: 2026-03-30 21:36 UTC

## Comparison

| Impl | q actual | batch | validation | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | Correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU (NFLLib) | 30 | 1 | pass | 57.2021 | 61.8469 | 180.594 | 180.594 | n/a |
| GPU (CUDA) | 30 | 64 | pass | 41.7984 | 45.8986 | 129.509 | 2.024 | 1 |

## Speedups

- Forward NTT speedup per polynomial: 87.59x
- Full PolyMul speedup per polynomial: 89.24x
