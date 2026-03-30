# CPU vs GPU NTT Comparison (n=8192, q req CPU/GPU=32/32)

Generated: 2026-03-30 21:11 UTC

## Comparison

| Impl | q actual | batch | validation | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | Correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU (NFLLib) | 30 | 1 | pass | 57.1827 | 62.1121 | 180.335 | 180.335 | n/a |
| GPU (CUDA) | 30 | 64 | pass | 41.7958 | 45.8784 | 129.444 | 2.023 | 1 |

## Speedups

- Forward NTT speedup per polynomial: 87.56x
- Full PolyMul speedup per polynomial: 89.16x
