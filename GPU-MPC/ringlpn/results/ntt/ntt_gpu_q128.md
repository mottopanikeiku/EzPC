# Ring-LPN GPU NTT Sweep (Requested q=128)

Generated: 2026-06-10 10:54 UTC

## Results

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 128 | 124 | 64 | pass | 400 | 162.784 | 157.132 | 489.003 | 7.641 | 130878.54 | 34.31 |
| 16384 | 14 | 128 | 124 | 64 | pass | 400 | 215.725 | 207.176 | 651.881 | 10.186 | 98177.43 | 51.47 |
| 32768 | 15 | 128 | 124 | 64 | pass | 400 | 319.555 | 306.123 | 1071.420 | 16.741 | 59733.81 | 62.64 |
| 65536 | 16 | 128 | 124 | 16 | pass | 200 | 120.168 | 77.182 | 287.539 | 17.971 | 55644.63 | 116.70 |
| 131072 | 17 | 128 | 124 | 16 | pass | 200 | 310.035 | 302.886 | 1051.000 | 65.688 | 15223.60 | 63.85 |
| 262144 | 18 | 128 | 124 | 8 | pass | 80 | 305.221 | 304.798 | 980.189 | 122.524 | 8161.69 | 68.47 |
| 524288 | 19 | 128 | 124 | 4 | pass | 30 | 309.732 | 304.119 | 991.207 | 247.802 | 4035.48 | 67.70 |
| 1048576 | 20 | 128 | 124 | 2 | pass | 10 | 306.950 | 296.621 | 988.307 | 494.154 | 2023.66 | 67.90 |

## Notes

- This CUDA path covers requested qbits 128 and realizes them with actual qbits 124; q=128 uses two q62 CRT prime limbs in one flattened Cheddar launch schedule.
- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.
- Est. coeff GB/s uses bytes_per_op = batch_size * n * coeff_bytes * 2 as a rough traffic proxy, with coeff_bytes = 4 for q actual <= 32 and 8 bytes per 64-bit CRT limb otherwise.
- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.
- The selected NTT prime sets support n up to 2^20, so these sweeps intentionally extend past the CPU NFLLib uint32_t cutoff.
