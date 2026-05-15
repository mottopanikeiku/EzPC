# Ring-LPN Linear OLE Handoff

Updated: 2026-05-15

## Status

The standalone Ring-LPN linear-layer artifact is implemented and validated for small GPU smoke cases under both uniform and regular sparse noise. The artifact now uses true matrix semantics: every shared `A[row,k]` operand and every shared `B[k,col]` operand is sampled once and reused across all products that reference it.

This artifact applies the standard two-OLE-to-Beaver conversion to a matrix multiplication whose entries are ring polynomials in `Z_p[X]/(X^N+1)`.

Follow-up bridge status: `results/orca_zp_bridge_handoff.md` records a host-only constant-polynomial scalar packing smoke and the exact carry-corrected `Z_p -> Z_{2^bw}` dealer/oracle share conversion. The new `results/orca_fc_ringlpn_demo_memo.md` records a tiny forward-only Orca FC key-writer demo that exercises `gpuMatmulBeaver` unchanged. This is still a bounded q62 correctness demo, not q128/CRT, dense packing, secure distributed conversion, or full Orca training integration.

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
| `src/test_orca_zp_bridge.cpp` | Host-only scalar bridge test for constant-polynomial packing and exact `Z_p -> Z_{2^bw}` share conversion |
| `scripts/run_orca_zp_bridge_test.sh` | Runs the scalar bridge smoke and negative q62/full-32-bit counterexample |
| `src/bench_orca_fc_ringlpn_demo.cu` | Tiny Orca FC key-writer demo for bounded q62 constant-polynomial masks |
| `scripts/build_orca_fc_ringlpn_demo.sh` | Builds `bin/bench_orca_fc_ringlpn_demo` |
| `scripts/run_orca_fc_ringlpn_demo.sh` | Runs bounded FC demo cases, Orca baseline comparison, deterministic replay, and second-seed checks |
| `results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md` | Current smoke result summary |
| `results/linear_ole_gpu_q64_regular_r2_k2_c2_n8192_t8.md` | Current regular-noise smoke result summary |
| `results/orca_zp_bridge_handoff.md` | Current Orca-facing scalar bridge handoff |
| `results/orca_fc_ringlpn_demo_memo.md` | Professor-facing FC demo memo with proof sketch and paper gaps |

## Reproduction

Run inside the `orca-dev` container from `/home/ringlpn`:

```bash
./scripts/build_linear_ole_bench.sh
./scripts/run_linear_ole_sweep.sh

NOISE=regular ./scripts/run_linear_ole_sweep.sh

./scripts/build_orca_fc_ringlpn_demo.sh
./scripts/run_orca_fc_ringlpn_demo.sh
```

The default smoke is:

- `N=8192`,
- matrix shape `rows=2`, `inner=2`, `cols=2`,
- `c=2`,
- `t=8`,
- requested `qbits=64`, actual `qbits=62`.

## Current Result

| noise | rows | inner | cols | n | c | t | validation | shared operands | ring products | OLE instances | pair key bytes | keygen us | linear expand mean us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 2 | 2 | 2 | 8192 | 2 | 8 | pass | 1 | 8 | 16 | 2,264,064 | 6,587 | 223,667 |
| regular | 2 | 2 | 2 | 8192 | 2 | 8 | pass | 1 | 8 | 16 | 1,864,704 | 81,582 | 114,825 |

The validation checks coefficientwise that `C_0 + C_1` equals the clear matrix product `(A_0 + A_1) * (B_0 + B_1)` over `Z_p[X]/(X^N+1)`. The `shared operands` column is a regression check that would fail if products independently resampled `A[row,k]` or `B[k,col]` instead of reusing matrix operands.

The scalar bridge smoke separately validates:

| bw | rows | inner | cols | value bound | naive share failures | corrected share failures | no-prime-wrap bound | scalar validation | counterexample |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 16 | 2 | 2 | 2 | 255 | 633 | 0 | yes | pass | no |
| 32 | 1 | 1 | 1 | 4294967295 | 633 | 0 | no | not claimed | yes |

This records the exact prime-carry correction needed when exporting `Z_p` shares to Orca's `Z_{2^bw}` ring, and it shows that q62 is insufficient for unrestricted 32-bit scalar products.

The FC demo separately validates:

| seed | second seed | shape | bw | value bound | key bytes per party | baseline bytes per party | carry conversion | replay | second seed | online contract | baseline | validation |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | 2 | 2x2x2 | 16 | 255 | 96 | 96 | 1 | 1 | 1 | pass | pass | pass |
| 3 | 4 | 2x3x2 | 16 | 255 | 128 | 128 | 1 | 1 | 1 | pass | pass | pass |
| 5 | 6 | 3x2x2 | 16 | 255 | 128 | 128 | 1 | 1 | 1 | pass | pass | pass |
| 7 | 8 | 2x2x3 | 32 | 255 | 128 | 128 | 1 | 1 | 1 | pass | pass | pass |

This demo writes raw `A`, `B`, `C_masked` party buffers and reconstructs `clear FC output + output_mask` through the unchanged Orca `gpuMatmulBeaver` implementation. The baseline column is generated with Orca's `gpuKeygenMatmul` under the same masks and deterministic P0/P1 share stream, and it matches the Ring-LPN-style writer in every current case.

## Scientific Boundary

This is a real OLE-to-Beaver linear-layer step. It is still a ring-polynomial artifact plus a tiny Orca FC correctness demo, not a full Orca FC replacement.

What is valid:

- two Figure 2 Ring-LPN OLE instances are converted into one Beaver ring product,
- those ring products are summed into a matrix multiplication layer,
- GPU validation passes for the current uniform and regular smoke cases,
- matrix operand reuse is explicitly checked for `A[row,k]` and `B[k,col]`,
- a host-only dealer/oracle conversion from `Z_p` shares to `Z_{2^bw}` shares is implemented and tested with the required prime-carry correction,
- constant-polynomial scalar packing is validated under the explicit bound `inner * value_bound^2 < p`,
- a tiny forward-only Orca FC demo writes key buffers in the expected order and validates the unchanged `gpuMatmulBeaver` online contract.

What is not valid to claim yet:

- no full Orca model/training integration exists yet,
- no high-density scalar packing from Orca tensor elements into Ring-LPN polynomial slots exists yet,
- no secure distributed `Z_p -> Z_{2^bw}` share conversion exists yet for parties that do not know both prime-field shares,
- no bounded regular-noise or CRT q128 lift exists yet,
- no trusted-dealer removal claim for Orca is justified yet.

## Recommended Next Steps

1. Add q128/CRT support or prove concrete layer-wise bounds that make q62 sufficient.
2. Replace the one-scalar-per-polynomial packing with a denser packing scheme only after the conversion boundary is locked.
3. If the claim is trusted-dealer removal, implement or cite a secure distributed `Z_p -> Z_{2^bw}` conversion protocol.
4. Only after that, measure P-LeNet/P-AlexNet.
