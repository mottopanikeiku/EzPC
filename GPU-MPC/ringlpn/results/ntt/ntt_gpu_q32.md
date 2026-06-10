# Ring-LPN GPU NTT Sweep (Requested q=32)

Generated: 2026-06-10 10:53 UTC

## Results

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | 64 | pass | 400 | 27.660 | 27.094 | 83.009 | 1.297 | 770998.88 | 50.53 |
| 16384 | 14 | 32 | 30 | 64 | pass | 400 | 51.656 | 50.565 | 156.632 | 2.447 | 408601.05 | 53.56 |
| 32768 | 15 | 32 | 30 | 64 | pass | 400 | 95.170 | 96.113 | 295.231 | 4.613 | 216779.40 | 56.83 |
| 65536 | 16 | 32 | 30 | 16 | pass | 200 | 14.435 | 13.678 | 39.478 | 2.467 | 405286.97 | 212.49 |
| 131072 | 17 | 32 | 30 | 16 | pass | 200 | 96.843 | 98.512 | 292.068 | 18.254 | 54781.76 | 57.44 |
| 262144 | 18 | 32 | 30 | 8 | pass | 80 | 106.903 | 107.910 | 321.580 | 40.197 | 24877.17 | 52.17 |
| 524288 | 19 | 32 | 30 | 4 | pass | 30 | 105.142 | 104.927 | 315.105 | 78.776 | 12694.18 | 53.24 |
| 1048576 | 20 | 32 | 30 | 2 | pass | 10 | 109.475 | 109.210 | 327.034 | 163.517 | 6115.57 | 51.30 |

## Notes

- This CUDA path covers requested qbits 32 and realizes them with actual qbits 30; q=128 uses two q62 CRT prime limbs in one flattened Cheddar launch schedule.
- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.
- Est. coeff GB/s uses bytes_per_op = batch_size * n * coeff_bytes * 2 as a rough traffic proxy, with coeff_bytes = 4 for q actual <= 32 and 8 bytes per 64-bit CRT limb otherwise.
- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.
- The selected NTT prime sets support n up to 2^20, so these sweeps intentionally extend past the CPU NFLLib uint32_t cutoff.
