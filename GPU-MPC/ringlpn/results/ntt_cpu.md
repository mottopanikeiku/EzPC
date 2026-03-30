# Ring-LPN CPU Baseline (NFLLib)

Generated: 2026-03-30 08:17 UTC

## Results

| n | log2(n) | q req | q actual | limb | validate | iters | NTT mean (us) | INTT mean (us) | Pointwise est. (us) | Full PolyMul mean (us) | Ops/s | Est. coeff GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 13 | 32 | 30 | u32/30 | pass | 800 | 57.623 | 64.525 | 3.896 | 183.667 | 5444.64 | 0.36 |
| 8192 | 13 | 64 | 62 | u64/62 | pass | 800 | 63.169 | 70.680 | 3.779 | 200.798 | 4980.13 | 0.65 |
| 8192 | 13 | 128 | 124 | u64/62 | pass | 800 | 129.529 | 134.951 | 1.923 | 395.932 | 2525.69 | 0.66 |
| 16384 | 14 | 32 | 30 | u32/30 | pass | 800 | 121.067 | 132.578 | 10.633 | 385.345 | 2595.08 | 0.34 |
| 16384 | 14 | 64 | 62 | u64/62 | pass | 800 | 123.307 | 143.097 | 21.941 | 411.652 | 2429.24 | 0.64 |
| 16384 | 14 | 128 | 124 | u64/62 | pass | 800 | 253.046 | 292.494 | 45.117 | 843.703 | 1185.25 | 0.62 |
| 32768 | 15 | 32 | 30 | u32/30 | pass | 800 | 256.393 | 288.655 | 24.124 | 825.565 | 1211.29 | 0.32 |
| 32768 | 15 | 64 | 62 | u64/62 | pass | 800 | 264.513 | 315.081 | 33.883 | 877.990 | 1138.97 | 0.60 |
| 32768 | 15 | 128 | 124 | u64/62 | pass | 800 | 534.061 | 628.781 | 90.117 | 1787.020 | 559.59 | 0.59 |
| 65536 | 16 | 64 | 62 | u64/62 | pass | 200 | 557.355 | 650.194 | 154.646 | 1919.550 | 520.96 | 0.55 |
| 65536 | 16 | 128 | 124 | u64/62 | pass | 200 | 1156.080 | 1401.370 | 56.540 | 3770.070 | 265.25 | 0.56 |
| 131072 | 17 | 64 | 62 | u64/62 | pass | 200 | 1195.020 | 1406.840 | 147.120 | 3944.000 | 253.55 | 0.53 |
| 131072 | 17 | 128 | 124 | u64/62 | pass | 200 | 2524.500 | 3489.300 | 122.820 | 8661.120 | 115.46 | 0.48 |
| 262144 | 18 | 64 | 62 | u64/62 | pass | 50 | 2640.390 | 4245.850 | 597.170 | 10123.800 | 98.78 | 0.41 |
| 262144 | 18 | 128 | 124 | u64/62 | pass | 50 | 5556.950 | 9178.220 | 680.380 | 20972.500 | 47.68 | 0.40 |
| 524288 | 19 | 64 | 62 | u64/62 | pass | 50 | 6146.530 | 10971.900 | 915.240 | 24180.200 | 41.36 | 0.35 |
| 524288 | 19 | 128 | 124 | u64/62 | pass | 50 | 12936.800 | 22503.500 | 3157.900 | 51535.000 | 19.40 | 0.33 |
| 1048576 | 20 | 64 | 62 | u64/62 | pass | 12 | 13388.900 | 23578.500 | 3664.600 | 54020.900 | 18.51 | 0.31 |
| 1048576 | 20 | 128 | 124 | u64/62 | pass | 12 | 27604.600 | 49102.700 | 7171.100 | 111483.000 | 8.97 | 0.30 |

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
