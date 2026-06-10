# Ring-LPN GPU NTT Sweep (Requested q=32)

Generated: 2026-04-02 08:49 UTC

## Results

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | 64 | pass | 400 | 44.300 | 48.589 | 137.286 | 2.145 | 466180.09 | 30.55 |
| 16384 | 14 | 32 | 30 | 64 | pass | 400 | 72.162 | 76.882 | 223.449 | 3.491 | 286418.82 | 37.54 |
| 32768 | 15 | 32 | 30 | 64 | pass | 400 | 140.741 | 149.373 | 435.097 | 6.798 | 147093.64 | 38.56 |
| 65536 | 16 | 32 | 30 | 16 | pass | 200 | 81.294 | 85.991 | 250.858 | 15.679 | 63781.10 | 33.44 |
| 131072 | 17 | 32 | 30 | 16 | pass | 200 | 155.098 | 163.815 | 477.836 | 29.865 | 33484.29 | 35.11 |
| 262144 | 18 | 32 | 30 | 8 | pass | 80 | 160.976 | 169.532 | 495.323 | 61.915 | 16151.08 | 33.87 |
| 524288 | 19 | 32 | 30 | 4 | pass | 30 | 174.178 | 182.994 | 537.118 | 134.280 | 7447.15 | 31.24 |
| 1048576 | 20 | 32 | 30 | 2 | pass | 10 | 181.648 | 190.138 | 564.374 | 282.187 | 3543.75 | 29.73 |

## Notes

- This CUDA path currently targets requested qbits=32 and realizes it with a single 30-bit prime, so q actual is 30 in every supported run.
- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.
- Est. coeff GB/s uses bytes_per_op = batch_size * n * 4 * 2 as a rough traffic proxy, not a hardware counter.
- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.
- The selected 30-bit prime supports n up to 2^20, so this sweep intentionally extends past the CPU NFLLib uint32_t cutoff.
