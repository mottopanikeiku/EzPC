# Ring-LPN GPU NTT Sweep (Requested q=64)

Generated: 2026-04-02 10:15 UTC

## Results

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 64 | 62 | 64 | pass | 400 | 86.117 | 82.512 | 253.654 | 3.963 | 252312.20 | 33.07 |
| 16384 | 14 | 64 | 62 | 64 | pass | 400 | 103.020 | 100.905 | 311.081 | 4.861 | 205734.20 | 53.93 |
| 32768 | 15 | 64 | 62 | 64 | pass | 400 | 157.157 | 146.257 | 465.788 | 7.278 | 137401.56 | 72.04 |
| 65536 | 16 | 64 | 62 | 16 | pass | 200 | 47.803 | 40.438 | 140.357 | 8.772 | 113995.03 | 119.53 |
| 131072 | 17 | 64 | 62 | 16 | pass | 200 | 155.839 | 144.325 | 460.117 | 28.757 | 34773.76 | 72.93 |
| 262144 | 18 | 64 | 62 | 8 | pass | 80 | 153.630 | 143.456 | 456.098 | 57.012 | 17540.09 | 73.57 |
| 524288 | 19 | 64 | 62 | 4 | pass | 30 | 156.621 | 143.510 | 462.271 | 115.568 | 8652.93 | 72.59 |
| 1048576 | 20 | 64 | 62 | 2 | pass | 10 | 167.523 | 151.149 | 488.858 | 244.429 | 4091.17 | 68.64 |

## Notes

- This CUDA path currently covers requested qbits 64 and realizes them with actual qbits 62 using a single prime per run.
- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.
- Est. coeff GB/s uses bytes_per_op = batch_size * n * coeff_bytes * 2 as a rough traffic proxy, with coeff_bytes = 4 for q actual <= 32 and 8 otherwise.
- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.
- The selected single-prime parameter sets support n up to 2^20, so these sweeps intentionally extend past the CPU NFLLib uint32_t cutoff.
