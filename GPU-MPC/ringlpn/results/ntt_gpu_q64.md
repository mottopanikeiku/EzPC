# Ring-LPN GPU NTT Sweep (Requested q=64)

Generated: 2026-06-10 10:19 UTC

## Results

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 64 | 62 | 64 | pass | 400 | 90.260 | 84.149 | 265.267 | 4.145 | 241266.35 | 31.62 |
| 16384 | 14 | 64 | 62 | 64 | pass | 400 | 109.373 | 102.865 | 327.876 | 5.123 | 195195.74 | 51.17 |
| 32768 | 15 | 64 | 62 | 64 | pass | 400 | 162.621 | 148.338 | 478.716 | 7.480 | 133690.96 | 70.09 |
| 65536 | 16 | 64 | 62 | 16 | pass | 200 | 47.554 | 41.074 | 140.307 | 8.769 | 114035.65 | 119.58 |
| 131072 | 17 | 64 | 62 | 16 | pass | 200 | 157.978 | 147.822 | 469.906 | 29.369 | 34049.36 | 71.41 |
| 262144 | 18 | 64 | 62 | 8 | pass | 80 | 158.158 | 147.150 | 470.942 | 58.868 | 16987.23 | 71.25 |
| 524288 | 19 | 64 | 62 | 4 | pass | 30 | 159.607 | 145.351 | 470.598 | 117.650 | 8499.82 | 71.30 |
| 1048576 | 20 | 64 | 62 | 2 | pass | 10 | 167.626 | 152.941 | 491.939 | 245.970 | 4065.54 | 68.21 |

## Notes

- This CUDA path covers requested qbits 64 and realizes them with actual qbits 62; q=128 uses two q62 CRT prime limbs in one flattened Cheddar launch schedule.
- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.
- Est. coeff GB/s uses bytes_per_op = batch_size * n * coeff_bytes * 2 as a rough traffic proxy, with coeff_bytes = 4 for q actual <= 32 and 8 bytes per 64-bit CRT limb otherwise.
- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.
- The selected NTT prime sets support n up to 2^20, so these sweeps intentionally extend past the CPU NFLLib uint32_t cutoff.
