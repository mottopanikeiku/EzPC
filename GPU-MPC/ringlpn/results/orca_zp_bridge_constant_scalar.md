# Orca Zp-to-Z2k Bridge Smoke

This host-only smoke validates the carry-corrected share conversion needed when a `Z_p` OLE/Beaver share is exported toward Orca's `Z_{2^bw}` linear-layer ring.

| requested_qbits | actual_qbits | bw | rows | inner | cols | value_bound | naive_share_failures | corrected_share_failures | no_modulus_wrap_bound | constant_scalar_matmul_validation | counterexample_found |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 64 | 62 | 16 | 2 | 2 | 2 | 255 | 621 | 0 | 1 | pass | 0 |
| 64 | 62 | 32 | 1 | 1 | 1 | 4294967295 | 621 | 0 | 0 | not_claimed | 1 |
| 128 | 124 | 32 | 2 | 2 | 2 | 4294967295 | 598 | 0 | 1 | pass | 0 |

Interpretation:

- The corrected conversion subtracts the hidden prime carry `m*p` from one output share before reducing to `Z_{2^bw}`.
- Constant-polynomial scalar packing is valid only under the explicit no-wrap bound `inner * value_bound^2 < modulus`.
- The q62 `bw=32` full-range row is intentionally not claimed; it records a counterexample showing why q62 is insufficient for unrestricted 32-bit scalar products.
- The q128 row uses `M = p0*p1` and validates the bounded full-32-bit scalar case under the same dealer/oracle carry correction.
