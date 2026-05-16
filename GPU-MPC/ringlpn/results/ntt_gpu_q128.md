# Ring-LPN GPU NTT Sweep (Requested q=128)

Generated: 2026-05-16 20:02 UTC

## Results

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 128 | 124 | 64 | pass | 400 | 163.775 | 157.663 | 491.715 | 7.683 | 130156.70 | 34.12 |
| 16384 | 14 | 128 | 124 | 64 | pass | 400 | 221.010 | 210.248 | 664.390 | 10.381 | 96328.96 | 50.50 |
| 32768 | 15 | 128 | 124 | 64 | pass | 400 | 318.596 | 308.855 | 1076.000 | 16.812 | 59479.55 | 62.37 |
| 65536 | 16 | 128 | 124 | 16 | pass | 200 | 115.266 | 73.120 | 278.679 | 17.417 | 57413.73 | 120.41 |
| 131072 | 17 | 128 | 124 | 16 | pass | 200 | 307.621 | 303.166 | 1052.880 | 65.805 | 15196.41 | 63.74 |
| 262144 | 18 | 128 | 124 | 8 | pass | 80 | 310.968 | 303.587 | 1058.820 | 132.352 | 7555.58 | 63.38 |
| 524288 | 19 | 128 | 124 | 4 | pass | 30 | 297.849 | 287.600 | 1016.860 | 254.215 | 3933.68 | 66.00 |
| 1048576 | 20 | 128 | 124 | 2 | pass | 10 | 307.693 | 295.248 | 1047.250 | 523.625 | 1909.76 | 64.08 |

## Notes

- This CUDA path covers requested qbits 128 and realizes them with actual qbits 124; q=128 uses two q62 CRT prime limbs in one flattened Cheddar launch schedule.
- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.
- Est. coeff GB/s uses bytes_per_op = batch_size * n * coeff_bytes * 2 as a rough traffic proxy, with coeff_bytes = 4 for q actual <= 32 and 8 bytes per 64-bit CRT limb otherwise.
- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.
- The selected NTT prime sets support n up to 2^20, so these sweeps intentionally extend past the CPU NFLLib uint32_t cutoff.
