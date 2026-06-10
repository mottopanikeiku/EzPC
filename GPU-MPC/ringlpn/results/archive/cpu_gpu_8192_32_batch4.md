# CPU vs GPU NTT Comparison (n=8192, q req CPU/GPU=32/32)

Generated: 2026-04-02 08:48 UTC

## Comparison

| Impl | q actual | batch | validation | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | Correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU (NFLLib) | 30 | 1 | pass | 51.7008 | 56.3666 | 168.777 | 168.777 | n/a |
| GPU (CUDA) | 30 | 4 | pass | 6.90256 | 6.71837 | 19.421 | 4.855 | 1 |

## Speedups

- Forward NTT speedup per polynomial: 29.96x
- Full PolyMul speedup per polynomial: 34.76x
