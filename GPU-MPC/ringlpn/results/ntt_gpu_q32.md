# Ring-LPN GPU NTT Sweep (Requested q=32)

Generated: 2026-03-30 21:10 UTC

## Results

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | 64 | pass | 400 | 44.158 | 48.400 | 136.929 | 2.140 | 467395.51 | 30.63 |
| 16384 | 14 | 32 | 30 | 64 | pass | 400 | 76.342 | 81.484 | 235.912 | 3.686 | 271287.60 | 35.56 |
| 32768 | 15 | 32 | 30 | 64 | pass | 400 | 141.391 | 149.982 | 436.852 | 6.826 | 146502.71 | 38.40 |
| 65536 | 16 | 32 | 30 | 16 | pass | 200 | 81.481 | 86.272 | 251.545 | 15.722 | 63606.91 | 33.35 |
| 131072 | 17 | 32 | 30 | 16 | pass | 200 | 155.899 | 164.717 | 480.382 | 30.024 | 33306.83 | 34.92 |
| 262144 | 18 | 32 | 30 | 8 | pass | 80 | 159.171 | 167.559 | 490.554 | 61.319 | 16308.09 | 34.20 |
| 524288 | 19 | 32 | 30 | 4 | pass | 30 | 173.855 | 182.873 | 536.010 | 134.002 | 7462.55 | 31.30 |
| 1048576 | 20 | 32 | 30 | 2 | pass | 10 | 181.773 | 189.533 | 563.117 | 281.558 | 3551.66 | 29.79 |

## Notes

- This CUDA path currently targets requested qbits=32 and realizes it with a single 30-bit prime, so q actual is 30 in every supported run.
- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.
- Est. coeff GB/s uses bytes_per_op = batch_size * n * 4 * 2 as a rough traffic proxy, not a hardware counter.
- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.
- The selected 30-bit prime supports n up to 2^20, so this sweep intentionally extends past the CPU NFLLib uint32_t cutoff.
