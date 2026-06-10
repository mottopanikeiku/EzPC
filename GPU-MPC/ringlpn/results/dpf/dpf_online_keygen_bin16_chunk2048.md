# DPF Online Key Generation Sweep (bin=16)

Generated: 2026-04-20 08:20 UTC

## Results

| N | bin | chunk | validate | iters | Full pair key (MiB) | Partial peak pair key (MiB) | Peak reduction | Total bytes multiplier | Full pair keygen mean (us) | Partial pipeline mean (us) | Time overhead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 16 | 2048 | pass | 100 | 2.81 | 0.70 | 4.00x | 1.000x | 269.840 | 738.720 | 2.738x |
| 16384 | 16 | 2048 | pass | 100 | 5.63 | 0.70 | 8.00x | 1.000x | 377.880 | 1421.810 | 3.763x |
| 32768 | 16 | 2048 | pass | 100 | 11.25 | 0.70 | 16.00x | 1.000x | 690.840 | 2844.310 | 4.117x |
| 65536 | 16 | 2048 | pass | 50 | 22.50 | 0.70 | 32.00x | 1.000x | 1236.620 | 5672.160 | 4.587x |
| 131072 | 16 | 2048 | pass | 50 | 45.00 | 0.70 | 64.00x | 1.000x | 2403.340 | 11386.100 | 4.738x |
| 262144 | 16 | 2048 | pass | 20 | 90.00 | 0.70 | 127.99x | 1.000x | 4666.100 | 22696.300 | 4.864x |
| 524288 | 16 | 2048 | pass | 10 | 180.00 | 0.70 | 255.98x | 1.000x | 9181.000 | 45335.200 | 4.938x |
| 1048576 | 16 | 2048 | pass | 3 | 360.00 | 0.70 | 511.97x | 1.000x | 18239.300 | 90744.000 | 4.975x |

## Notes

- This sweep measures standalone DPF online key generation with eval-all keys for bin 16 and chunk size 2048.
- Full pair key is the total key material generated at once for both parties. Partial peak pair key is the maximum per-chunk key material when keys are generated only for the current chunk.
- Peak reduction quantifies the reduction in peak key footprint from partial online key generation. Total bytes multiplier captures the total key material generated across all chunks relative to the one-shot offline baseline.
- Full pair keygen mean measures one-shot generation for both parties. Partial pipeline mean measures generating all chunks for both parties.
- Validation checks key serialization layout and parsed key metadata for both full and chunked modes. This sweep is a key-generation systems benchmark, not an end-to-end FSS evaluation benchmark.
