# Ring-LPN Linear OLE Handoff

Updated: 2026-05-04

## Status

The standalone Ring-LPN linear-layer artifact is implemented and validated for a small GPU smoke case.

This artifact applies the standard two-OLE-to-Beaver conversion to a matrix multiplication whose entries are ring polynomials in `Z_p[X]/(X^N+1)`.

For one ring-product term:

- OLE 1 gives shares of `A_0 * B_1`,
- OLE 2 gives shares of `A_1 * B_0`,
- party 0 locally adds `A_0 * B_0`,
- party 1 locally adds `A_1 * B_1`,
- the resulting shares satisfy `C_0 + C_1 = (A_0 + A_1) * (B_0 + B_1)`.

For a matrix product, the benchmark sums those ring-product shares across the inner dimension.

## Source Map

| Path | Role |
| --- | --- |
| `src/bench_linear_ole_ringlpn_cuda.cu` | Ring-polynomial matrix Beaver artifact built from two Figure 2 OLE instances per ring product |
| `src/bench_ole_ringlpn_cuda.cu` | Existing OLE artifact, now reusable via `RINGLPN_OLE_DISABLE_MAIN` |
| `scripts/build_linear_ole_bench.sh` | Builds `bin/bench_linear_ole_ringlpn_cuda` |
| `scripts/run_linear_ole_sweep.sh` | Runs the smoke/default linear-layer artifact and writes CSV/Markdown |
| `scripts/summarize_linear_ole_results.py` | Summarizes the linear artifact CSV |
| `results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md` | Current smoke result summary |

## Reproduction

Run inside the `orca-dev` container from `/home/ringlpn`:

```bash
./scripts/build_linear_ole_bench.sh
./scripts/run_linear_ole_sweep.sh
```

The default smoke is:

- `N=8192`,
- matrix shape `rows=2`, `inner=2`, `cols=2`,
- `c=2`,
- `t=8`,
- requested `qbits=64`, actual `qbits=62`.

## Current Result

| rows | inner | cols | n | c | t | validation | ring products | OLE instances | pair key bytes | keygen us | linear expand mean us |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 2 | 8192 | 2 | 8 | pass | 8 | 16 | 2,264,064 | 6,594 | 222,355 |

The validation checks coefficientwise that `C_0 + C_1` equals the clear matrix product `(A_0 + A_1) * (B_0 + B_1)` over `Z_p[X]/(X^N+1)`.

## Scientific Boundary

This is a real OLE-to-Beaver linear-layer step, but it is still a ring-polynomial linear layer, not an Orca FC layer.

What is valid:

- two Figure 2 Ring-LPN OLE instances are converted into one Beaver ring product,
- those ring products are summed into a matrix multiplication layer,
- GPU validation passes for the current bounded smoke case.

What is not valid to claim yet:

- no Orca `gpuMatmulBeaver` integration exists yet,
- no scalar packing from Orca tensor elements into Ring-LPN polynomial slots exists yet,
- no `Z_p -> Z_{2^bw}` share conversion exists yet,
- no regular-noise or CRT q128 lift exists yet,
- no trusted-dealer removal claim for Orca is justified yet.

## Recommended Next Steps

1. Decide the scalar packing model from Orca tensors into ring-polynomial entries.
2. Specify and implement `Z_p -> Z_{2^bw}` share conversion.
3. Add an Orca-compatible triple writer that produces the same `(A, B, C)` key shape consumed by `gpuMatmulBeaver`.
4. Run a tiny FC-layer-only Orca validation against baseline Beaver triples.
5. Only after that, measure P-LeNet/P-AlexNet.
