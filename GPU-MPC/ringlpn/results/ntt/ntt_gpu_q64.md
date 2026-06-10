# Ring-LPN GPU NTT Sweep (Requested q=64)

Generated: 2026-06-10 10:54 UTC

## Results

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 64 | 62 | 64 | pass | 400 | 88.166 | 82.080 | 259.095 | 4.048 | 247013.64 | 32.38 |
| 16384 | 14 | 64 | 62 | 64 | pass | 400 | 109.153 | 103.274 | 328.031 | 5.125 | 195103.51 | 51.15 |
| 32768 | 15 | 64 | 62 | 64 | pass | 400 | 161.938 | 147.802 | 477.469 | 7.460 | 134040.12 | 70.28 |
| 65536 | 16 | 64 | 62 | 16 | pass | 200 | 46.148 | 40.349 | 133.091 | 8.318 | 120218.50 | 126.06 |
| 131072 | 17 | 64 | 62 | 16 | pass | 200 | 155.959 | 147.932 | 456.040 | 28.503 | 35084.64 | 73.58 |
| 262144 | 18 | 64 | 62 | 8 | pass | 80 | 156.666 | 147.324 | 455.702 | 56.963 | 17555.33 | 73.63 |
| 524288 | 19 | 64 | 62 | 4 | pass | 30 | 157.981 | 145.505 | 456.362 | 114.091 | 8764.97 | 73.53 |
| 1048576 | 20 | 64 | 62 | 2 | pass | 10 | 166.483 | 153.613 | 477.402 | 238.701 | 4189.34 | 70.29 |

## Notes

- This CUDA path covers requested qbits 64 and realizes them with actual qbits 62; q=128 uses two q62 CRT prime limbs in one flattened Cheddar launch schedule.
- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.
- Est. coeff GB/s uses bytes_per_op = batch_size * n * coeff_bytes * 2 as a rough traffic proxy, with coeff_bytes = 4 for q actual <= 32 and 8 bytes per 64-bit CRT limb otherwise.
- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.
- The selected NTT prime sets support n up to 2^20, so these sweeps intentionally extend past the CPU NFLLib uint32_t cutoff.
