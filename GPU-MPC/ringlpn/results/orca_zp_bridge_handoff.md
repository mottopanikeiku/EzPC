# Orca Zp-to-Z2k Bridge Handoff

Updated: 2026-05-15

## Status

This checkpoint adds a host-only correctness harness for the first Orca-facing arithmetic bridge after the ring-polynomial OLE-to-Beaver artifact.

The implemented test is:

- `src/test_orca_zp_bridge.cpp`
- `scripts/build_orca_zp_bridge_test.sh`
- `scripts/run_orca_zp_bridge_test.sh`
- `scripts/run_paper_checkpoint_smoke.sh`
- `results/orca_zp_bridge_constant_scalar.csv`
- `results/orca_zp_bridge_constant_scalar.md`
- `results/paper_execution_next_steps.md`

It validates two narrow facts needed before any Orca FC integration claim:

1. A `Z_p` additive sharing cannot be converted to a `Z_{2^bw}` additive sharing by reducing both shares independently.
2. A conservative constant-polynomial scalar packing model is correct under an explicit no-prime-wrap bound.

## Share Conversion

Let `z0, z1 in [0, p)` be additive shares of a value `v` over `Z_p`, so:

`v = z0 + z1 mod p`.

Write:

`m = floor((z0 + z1) / p)`, where `m in {0, 1}` for two canonical shares.

The exact dealer/oracle conversion to `Z_{2^bw}` is:

- `r0 = z0 mod 2^bw`,
- `r1 = z1 - m*p mod 2^bw`.

Then:

`r0 + r1 = v mod 2^bw`.

The correction term is necessary because the current prime is odd, so `p mod 2^bw != 0`. The smoke test records hundreds of failures for naive per-share reduction and zero failures for the corrected conversion.

## Constant-Polynomial Scalar Packing

The harness also validates a deliberately conservative scalar packing model:

- encode scalar `s` as the constant polynomial `s + 0X + ... + 0X^{N-1}`,
- use one ring polynomial per scalar entry,
- convert the resulting constant coefficient from `Z_p` shares to `Z_{2^bw}` shares using the carry-corrected rule above.

For a scalar matrix product with inner dimension `K` and unsigned values bounded by `B`, this is sound when:

`K * B^2 < p`.

Under that bound, the `Z_p` scalar dot product does not wrap the prime, so reducing the corrected output to `Z_{2^bw}` matches the Orca-ring scalar product.

This is not an efficient final packing scheme. It is a mathematically conservative first bridge for correctness and integration tests.

## Current Smoke Result

| bw | rows | inner | cols | value bound | naive share failures | corrected share failures | no-prime-wrap bound | scalar validation | counterexample |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 16 | 2 | 2 | 2 | 255 | 633 | 0 | yes | pass | no |
| 32 | 1 | 1 | 1 | 4294967295 | 633 | 0 | no | not claimed | yes |

The second row is an intentional negative control: single-prime q62 is not sufficient for unrestricted 32-bit scalar products. It demonstrates why q128/CRT and/or tighter value bounds are required before paper-comparable Orca linear-layer claims.

## Reproduction

Run from the repository root or from `GPU-MPC/ringlpn`:

```bash
GPU-MPC/ringlpn/scripts/build_orca_zp_bridge_test.sh
GPU-MPC/ringlpn/scripts/run_orca_zp_bridge_test.sh
```

The host binary is written to `GPU-MPC/ringlpn/host_bin/test_orca_zp_bridge` because `GPU-MPC/ringlpn/bin` is normally container-owned by the CUDA benchmark builds.

For the full current checkpoint smoke:

```bash
GPU-MPC/ringlpn/scripts/run_paper_checkpoint_smoke.sh
```

Inside the container from `/home/ringlpn`, set `RUN_GPU_SMOKE=1` on that script to include the CUDA OLE and linear smokes.

## Scientific Boundary

What is valid now:

- the exact prime-carry correction for dealer/oracle conversion from `Z_p` shares to `Z_{2^bw}` shares is implemented and tested,
- constant-polynomial scalar packing is validated under an explicit no-prime-wrap bound,
- the harness includes a q62/full-32-bit counterexample to prevent an invalid claim.

What is still not valid to claim:

- no secure distributed conversion protocol is implemented for parties that do not know both `Z_p` shares,
- no high-density scalar packing into ring-polynomial slots is implemented,
- no q128/CRT bridge exists yet,
- no Orca `gpuMatmulBeaver` key writer consumes these converted shares yet.

## Next Steps

1. Add q128/CRT support or prove the concrete Orca value bounds make q62 sufficient for the targeted layer.
2. Wire a constant-polynomial scalar bridge into a tiny FC-only key writer for correctness comparison against baseline Beaver triples.
3. Replace the conservative one-scalar-per-polynomial packing with a denser packing scheme only after the scalar conversion boundary is locked.
4. If trusted-dealer removal is still the claim, implement or cite a secure conversion from `Z_p` shares to `Z_{2^bw}` shares; the current correction is a dealer/oracle operation.
