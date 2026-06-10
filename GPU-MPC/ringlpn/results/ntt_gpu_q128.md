# Ring-LPN GPU NTT Sweep (Requested q=128)

Generated: 2026-06-10 10:19 UTC

## Results

| n | log2(n) | q req | q actual | batch | validate | iters | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) | PolyMul polys/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 128 | 124 | 64 | pass | 400 | 161.545 | 155.680 | 485.153 | 7.581 | 131917.15 | 34.58 |
| 16384 | 14 | 128 | 124 | 64 | pass | 400 | 215.521 | 206.431 | 650.433 | 10.163 | 98395.99 | 51.59 |
| 32768 | 15 | 128 | 124 | 64 | pass | 400 | 319.099 | 304.777 | 1068.420 | 16.694 | 59901.54 | 62.81 |
| 65536 | 16 | 128 | 124 | 16 | pass | 200 | 119.818 | 76.609 | 286.734 | 17.921 | 55800.85 | 117.02 |
| 131072 | 17 | 128 | 124 | 16 | pass | 200 | 307.878 | 300.564 | 1042.940 | 65.184 | 15341.25 | 64.35 |
| 262144 | 18 | 128 | 124 | 8 | pass | 80 | 304.973 | 304.960 | 1041.080 | 130.135 | 7684.33 | 64.46 |
| 524288 | 19 | 128 | 124 | 4 | pass | 30 | 308.637 | 303.283 | 1050.280 | 262.570 | 3808.51 | 63.90 |
| 1048576 | 20 | 128 | 124 | 2 | pass | 10 | 307.466 | 297.261 | 1043.610 | 521.805 | 1916.42 | 64.30 |

## Notes

- This CUDA path covers requested qbits 128 and realizes them with actual qbits 124; q=128 uses two q62 CRT prime limbs in one flattened Cheddar launch schedule.
- The benchmark batches independent polynomials in each launch; Full PolyMul mean is the batch latency, while Per-poly PolyMul divides by batch size.
- Est. coeff GB/s uses bytes_per_op = batch_size * n * coeff_bytes * 2 as a rough traffic proxy, with coeff_bytes = 4 for q actual <= 32 and 8 bytes per 64-bit CRT limb otherwise.
- Full PolyMul is measured directly as NTT(a) + NTT(b) + pointwise multiply + INTT across the full batch.
- The selected NTT prime sets support n up to 2^20, so these sweeps intentionally extend past the CPU NFLLib uint32_t cutoff.
