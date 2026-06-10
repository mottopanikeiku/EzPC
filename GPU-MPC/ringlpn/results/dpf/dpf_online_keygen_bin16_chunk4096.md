# DPF Online Key Generation Sweep (bin=16)

Generated: 2026-04-20 08:22 UTC

## Results

| N | bin | chunk | validate | iters | Full pair key (MiB) | Partial peak pair key (MiB) | Peak reduction | Total bytes multiplier | Full pair keygen mean (us) | Partial pipeline mean (us) | Time overhead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 16 | 4096 | pass | 100 | 2.81 | 1.41 | 2.00x | 1.000x | 271.480 | 429.790 | 1.583x |
| 16384 | 16 | 4096 | pass | 100 | 5.63 | 1.41 | 4.00x | 1.000x | 378.120 | 822.400 | 2.175x |
| 32768 | 16 | 4096 | pass | 100 | 11.25 | 1.41 | 8.00x | 1.000x | 691.320 | 1664.270 | 2.407x |
| 65536 | 16 | 4096 | pass | 50 | 22.50 | 1.41 | 16.00x | 1.000x | 1234.980 | 3297.320 | 2.670x |
| 131072 | 16 | 4096 | pass | 50 | 45.00 | 1.41 | 32.00x | 1.000x | 2412.360 | 6590.140 | 2.732x |
| 262144 | 16 | 4096 | pass | 20 | 90.00 | 1.41 | 64.00x | 1.000x | 4677.850 | 13201.200 | 2.822x |
| 524288 | 16 | 4096 | pass | 10 | 180.00 | 1.41 | 128.00x | 1.000x | 9176.100 | 26310.000 | 2.867x |
| 1048576 | 16 | 4096 | pass | 3 | 360.00 | 1.41 | 255.99x | 1.000x | 18242.700 | 53662.000 | 2.942x |

## Notes

- This sweep measures standalone DPF online key generation with eval-all keys for bin 16 and chunk size 4096.
- Full pair key is the total key material generated at once for both parties. Partial peak pair key is the maximum per-chunk key material when keys are generated only for the current chunk.
- Peak reduction quantifies the reduction in peak key footprint from partial online key generation. Total bytes multiplier captures the total key material generated across all chunks relative to the one-shot offline baseline.
- Full pair keygen mean measures one-shot generation for both parties. Partial pipeline mean measures generating all chunks for both parties.
- Validation checks key serialization layout and parsed key metadata for both full and chunked modes. This sweep is a key-generation systems benchmark, not an end-to-end FSS evaluation benchmark.
