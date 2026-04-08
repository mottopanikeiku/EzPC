# DPF Online Key Generation Sweep (bin=16)

Generated: 2026-04-08 10:18 UTC

## Results

| N | bin | chunk | validate | iters | Full pair key (MiB) | Partial peak pair key (MiB) | Peak reduction | Total bytes multiplier | Full pair keygen mean (us) | Partial pipeline mean (us) | Time overhead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 16 | 8192 | pass | 100 | 2.81 | 2.81 | 1.00x | 1.000x | 274.140 | 277.090 | 1.011x |
| 16384 | 16 | 8192 | pass | 100 | 5.63 | 2.81 | 2.00x | 1.000x | 385.630 | 532.340 | 1.380x |
| 32768 | 16 | 8192 | pass | 100 | 11.25 | 2.81 | 4.00x | 1.000x | 698.300 | 1082.750 | 1.551x |
| 65536 | 16 | 8192 | pass | 50 | 22.50 | 2.81 | 8.00x | 1.000x | 1249.100 | 2161.340 | 1.730x |
| 131072 | 16 | 8192 | pass | 50 | 45.00 | 2.81 | 16.00x | 1.000x | 2411.800 | 4311.740 | 1.788x |
| 262144 | 16 | 8192 | pass | 20 | 90.00 | 2.81 | 32.00x | 1.000x | 4700.700 | 8661.300 | 1.843x |
| 524288 | 16 | 8192 | pass | 10 | 180.00 | 2.81 | 64.00x | 1.000x | 9237.600 | 17298.600 | 1.873x |
| 1048576 | 16 | 8192 | pass | 3 | 360.00 | 2.81 | 128.00x | 1.000x | 18244.300 | 34390.300 | 1.885x |

## Notes

- This sweep measures standalone DPF online key generation with eval-all keys for bin 16 and chunk size 8192.
- Full pair key is the total key material generated at once for both parties. Partial peak pair key is the maximum per-chunk key material when keys are generated only for the current chunk.
- Peak reduction quantifies the reduction in peak key footprint from partial online key generation. Total bytes multiplier captures the total key material generated across all chunks relative to the one-shot offline baseline.
- Full pair keygen mean measures one-shot generation for both parties. Partial pipeline mean measures generating all chunks for both parties.
- Validation checks key serialization layout and parsed key metadata for both full and chunked modes. This sweep is a key-generation systems benchmark, not an end-to-end FSS evaluation benchmark.
