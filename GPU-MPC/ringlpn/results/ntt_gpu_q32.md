# Ring-LPN GPU NTT Sweep (Requested q=32)

Generated: 2026-04-02 10:16 UTC

## Results

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | 64 | pass | 400 | 26.777 | 25.925 | 79.628 | 1.244 | 803741.42 | 52.67 |
| 16384 | 14 | 32 | 30 | 64 | pass | 400 | 51.862 | 50.483 | 155.992 | 2.437 | 410277.45 | 53.78 |
| 32768 | 15 | 32 | 30 | 64 | pass | 400 | 100.302 | 99.889 | 306.232 | 4.785 | 208991.88 | 54.79 |
| 65536 | 16 | 32 | 30 | 16 | pass | 200 | 14.006 | 13.266 | 42.260 | 2.641 | 378613.09 | 198.50 |
| 131072 | 17 | 32 | 30 | 16 | pass | 200 | 96.612 | 99.303 | 298.463 | 18.654 | 53607.98 | 56.21 |
| 262144 | 18 | 32 | 30 | 8 | pass | 80 | 110.216 | 112.384 | 337.991 | 42.249 | 23669.27 | 49.64 |
| 524288 | 19 | 32 | 30 | 4 | pass | 30 | 105.022 | 104.251 | 320.102 | 80.025 | 12496.02 | 52.41 |
| 1048576 | 20 | 32 | 30 | 2 | pass | 10 | 109.050 | 109.437 | 332.762 | 166.381 | 6010.30 | 50.42 |

## Notes

- This CUDA path currently covers requested qbits 32 and realizes them with actual qbits 30 using a single prime per run.
- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.
- Est. coeff GB/s uses bytes_per_op = batch_size * n * coeff_bytes * 2 as a rough traffic proxy, with coeff_bytes = 4 for q actual <= 32 and 8 otherwise.
- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.
- The selected single-prime parameter sets support n up to 2^20, so these sweeps intentionally extend past the CPU NFLLib uint32_t cutoff.
