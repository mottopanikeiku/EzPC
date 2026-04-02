# Ring-LPN CPU Baseline (NFLLib)

Generated: 2026-03-30 21:54 UTC

## Results

| n | log2(n) | q req | q actual | limb | validate | iters | NTT mean (us) | INTT mean (us) | Pointwise est. (us) | Full PolyMul mean (us) | Ops/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | u32/30 | pass | 800 | 57.424 | 61.441 | 4.939 | 181.228 | 5517.91 | 0.36 |
| 8192 | 13 | 64 | 62 | u64/62 | pass | 800 | 58.059 | 64.737 | 9.391 | 190.245 | 5256.38 | 0.69 |
| 8192 | 13 | 128 | 124 | u64/62 | pass | 800 | 120.729 | 133.568 | 12.127 | 387.153 | 2582.96 | 0.68 |
| 16384 | 14 | 32 | 30 | u32/30 | pass | 800 | 122.671 | 129.392 | 15.801 | 390.535 | 2560.59 | 0.34 |
| 16384 | 14 | 64 | 62 | u64/62 | pass | 800 | 134.660 | 143.056 | -2.005 | 410.371 | 2436.82 | 0.64 |
| 16384 | 14 | 128 | 124 | u64/62 | pass | 800 | 247.499 | 290.537 | 45.547 | 831.082 | 1203.25 | 0.63 |
| 32768 | 15 | 32 | 30 | u32/30 | pass | 800 | 254.265 | 286.147 | 23.183 | 817.860 | 1222.70 | 0.32 |
| 32768 | 15 | 64 | 62 | u64/62 | pass | 800 | 267.046 | 306.363 | 33.974 | 874.429 | 1143.60 | 0.60 |
| 32768 | 15 | 128 | 124 | u64/62 | pass | 800 | 514.267 | 611.040 | 140.456 | 1780.030 | 561.79 | 0.59 |
| 65536 | 16 | 64 | 62 | u64/62 | pass | 200 | 554.246 | 642.903 | 87.195 | 1838.590 | 543.90 | 0.57 |
| 65536 | 16 | 128 | 124 | u64/62 | pass | 200 | 1124.700 | 1381.600 | 154.760 | 3785.760 | 264.15 | 0.55 |
| 131072 | 17 | 64 | 62 | u64/62 | pass | 200 | 1173.570 | 1366.900 | 280.440 | 3994.480 | 250.35 | 0.53 |
| 131072 | 17 | 128 | 124 | u64/62 | pass | 200 | 2497.300 | 3031.290 | 527.500 | 8553.390 | 116.91 | 0.49 |
| 262144 | 18 | 64 | 62 | u64/62 | pass | 50 | 2614.040 | 4202.420 | 580.700 | 10011.200 | 99.89 | 0.42 |
| 262144 | 18 | 128 | 124 | u64/62 | pass | 50 | 5516.830 | 8937.270 | 909.370 | 20880.300 | 47.89 | 0.40 |
| 524288 | 19 | 64 | 62 | u64/62 | pass | 50 | 5926.840 | 10603.500 | 1255.720 | 23712.900 | 42.17 | 0.35 |
| 524288 | 19 | 128 | 124 | u64/62 | pass | 50 | 13838.000 | 22340.400 | 915.600 | 50932.000 | 19.63 | 0.33 |
| 1048576 | 20 | 64 | 62 | u64/62 | pass | 12 | 14339.700 | 23156.500 | 1849.900 | 53685.800 | 18.63 | 0.31 |
| 1048576 | 20 | 128 | 124 | u64/62 | pass | 12 | 32329.800 | 48085.500 | -1038.100 | 111707.000 | 8.95 | 0.30 |

## Unsupported Requested Points

| n | log2(n) | q req | status | reason |
| --- | --- | --- | --- | --- |
| 65536 | 16 | 32 | unsupported | Unsupported config: requested qbits=32 maps to actual qbits=30; which is limited to n <= 32768 in NFLLib uint32_t mode |
| 131072 | 17 | 32 | unsupported | Unsupported config: requested qbits=32 maps to actual qbits=30; which is limited to n <= 32768 in NFLLib uint32_t mode |
| 262144 | 18 | 32 | unsupported | Unsupported config: requested qbits=32 maps to actual qbits=30; which is limited to n <= 32768 in NFLLib uint32_t mode |
| 524288 | 19 | 32 | unsupported | Unsupported config: requested qbits=32 maps to actual qbits=30; which is limited to n <= 32768 in NFLLib uint32_t mode |
| 1048576 | 20 | 32 | unsupported | Unsupported config: requested qbits=32 maps to actual qbits=30; which is limited to n <= 32768 in NFLLib uint32_t mode |

## Notes

- NFLLib only supports aggregated modulus sizes that are multiples of 30 bits in uint32_t mode or 62 bits in uint64_t mode.
- Requested qbits 32, 64, and 128 therefore resolve to actual qbits 30, 62, and 124 when the requested point is feasible.
- Requested qbits=32 is only feasible up to n=32768 because NFLLib uint32_t mode caps the degree there.
- Est. coeff GB/s uses bytes_per_op = n * limb_bytes * nb_moduli * 2 and is a rough throughput proxy, not a hardware counter.
- Full PolyMul is measured directly in the benchmark as NTT(a) + NTT(b) + pointwise multiply + INTT.
- Pointwise est. is back-computed as Full PolyMul - 2*NTT - INTT and is shown only as a sanity check.
- Ops/s = 1e6 / Full PolyMul_mean_us.
