# Ring-LPN GPU NTT Sweep (Requested q=32)

Generated: 2026-06-10 10:19 UTC

## Results

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | 64 | pass | 400 | 27.514 | 26.886 | 82.340 | 1.287 | 777265.00 | 50.94 |
| 16384 | 14 | 32 | 30 | 64 | pass | 400 | 51.606 | 50.612 | 156.844 | 2.451 | 408048.76 | 53.48 |
| 32768 | 15 | 32 | 30 | 64 | pass | 400 | 95.301 | 96.265 | 295.577 | 4.618 | 216525.64 | 56.76 |
| 65536 | 16 | 32 | 30 | 16 | pass | 200 | 14.462 | 13.527 | 44.255 | 2.766 | 361544.34 | 189.55 |
| 131072 | 17 | 32 | 30 | 16 | pass | 200 | 96.833 | 98.206 | 300.983 | 18.811 | 53159.15 | 55.74 |
| 262144 | 18 | 32 | 30 | 8 | pass | 80 | 105.232 | 106.288 | 326.188 | 40.773 | 24525.73 | 51.43 |
| 524288 | 19 | 32 | 30 | 4 | pass | 30 | 105.637 | 105.408 | 326.110 | 81.528 | 12265.80 | 51.45 |
| 1048576 | 20 | 32 | 30 | 2 | pass | 10 | 109.910 | 110.035 | 338.064 | 169.032 | 5916.04 | 49.63 |

## Notes

- This CUDA path covers requested qbits 32 and realizes them with actual qbits 30; q=128 uses two q62 CRT prime limbs in one flattened Cheddar launch schedule.
- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.
- Est. coeff GB/s uses bytes_per_op = batch_size * n * coeff_bytes * 2 as a rough traffic proxy, with coeff_bytes = 4 for q actual <= 32 and 8 bytes per 64-bit CRT limb otherwise.
- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.
- The selected NTT prime sets support n up to 2^20, so these sweeps intentionally extend past the CPU NFLLib uint32_t cutoff.
