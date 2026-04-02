# Ring-LPN GPU NTT Sweep (Requested q=32)

Generated: 2026-04-02 08:47 UTC

## Results

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | 64 | pass | 400 | 26.641 | 25.960 | 79.351 | 1.240 | 806543.08 | 52.86 |
| 16384 | 14 | 32 | 30 | 64 | pass | 400 | 51.857 | 50.381 | 155.850 | 2.435 | 410651.27 | 53.82 |
| 32768 | 15 | 32 | 30 | 64 | pass | 400 | 95.569 | 95.292 | 292.693 | 4.573 | 218659.14 | 57.32 |
| 65536 | 16 | 32 | 30 | 16 | pass | 200 | 14.053 | 13.263 | 42.196 | 2.637 | 379182.86 | 198.80 |
| 131072 | 17 | 32 | 30 | 16 | pass | 200 | 96.666 | 99.361 | 298.689 | 18.668 | 53567.42 | 56.17 |
| 262144 | 18 | 32 | 30 | 8 | pass | 80 | 104.180 | 106.416 | 320.880 | 40.110 | 24931.44 | 52.29 |
| 524288 | 19 | 32 | 30 | 4 | pass | 30 | 105.341 | 104.503 | 320.509 | 80.127 | 12480.15 | 52.35 |
| 1048576 | 20 | 32 | 30 | 2 | pass | 10 | 109.642 | 109.827 | 332.160 | 166.080 | 6021.19 | 50.51 |

## Notes

- This CUDA path currently targets requested qbits=32 and realizes it with a single 30-bit prime, so q actual is 30 in every supported run.
- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.
- Est. coeff GB/s uses bytes_per_op = batch_size * n * 4 * 2 as a rough traffic proxy, not a hardware counter.
- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.
- The selected 30-bit prime supports n up to 2^20, so this sweep intentionally extends past the CPU NFLLib uint32_t cutoff.
