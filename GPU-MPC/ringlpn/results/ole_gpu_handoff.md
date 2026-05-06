# GPU Figure 2 OLE Handoff

Updated: 2026-05-04

## Status

The standalone GPU Figure 2 Ring-LPN OLE artifact is implemented and validated for the first-pass configuration:

- modulus: requested `qbits=64`, actual single 62-bit prime `p = 4611686018326724609`,
- noise: uniform-position `t`-sparse noise over `[0, N)` with nonzero values in `Z_p`,
- SPFSS domain: `[0, 2N)`, folded to degree `< N` using `X^N = -1`,
- benchmark scope: OLE only, with no Orca Beaver-triple conversion or nonlinear FSS integration.

The validated claim is:

`z_0 + z_1 == x_0 * x_1` in `Z_p[X]/(X^N + 1)`.

This is a correctness and systems artifact for Figure 2's SPFSS-based OLE assembly. It is not yet a paper-parameter reproduction, a trusted-dealer removal for Orca, or an end-to-end linear-layer integration.

Follow-up status: `results/linear_ole_handoff.md` now records the first OLE-to-Beaver linear-layer artifact over ring-polynomial matrix entries. That follow-up still does not constitute Orca FC integration because scalar packing and `Z_p -> Z_{2^bw}` share conversion are not implemented yet.

## Source Map

| Path | Role |
| --- | --- |
| `src/gpu_spfss_zp.cuh` | GPU DPF/SPFSS path with additive `uint64_t` payload shares modulo `p` |
| `src/test_spfss_zp_cuda.cu` | GPU SPFSS payload tests |
| `src/bench_ole_ringlpn_cuda.cu` | Standalone Figure 2 SPFSS/OLE GPU benchmark |
| `scripts/build_ole_cuda_bench.sh` | Builds the OLE benchmark and SPFSS test |
| `scripts/run_ole_sweep.sh` | Runs smoke or bounded OLE sweep and writes CSV/Markdown |
| `scripts/summarize_ole_results.py` | Summarizes OLE CSV into Markdown |
| `results/ole_gpu_q64_uniform_c2_t8_smoke.md` | Smoke result summary |
| `results/ole_gpu_q64_uniform_c2_t64.md` | Bounded result summary |
| `results/linear_ole_handoff.md` | OLE-to-Beaver ring-polynomial linear-layer follow-up |

The new arithmetic SPFSS path is intentionally separate from the existing packed one-bit `gpu_dpf.cu` path, so current ReLU, DCF, LUT, and bit-output callers are unchanged.

## Reproduction

Run inside the `orca-dev` container from `/home/ringlpn`:

```bash
./scripts/build_ole_cuda_bench.sh
./bin/test_spfss_zp_cuda

SMOKE=1 ./scripts/run_ole_sweep.sh
./scripts/run_ole_sweep.sh
```

Preserved host-side checks for the Figure 2 oracle and host SPFSS path:

```bash
./scripts/build_ole_host.sh
./host_bin/verify_figure2_expand --n 128 --c 2 --t 16 --seed 1
./host_bin/test_spfss --log-domain 10 --m 16 --seed 1 --trials 5
./host_bin/bench_ole_ringlpn_host --n 64 --c 2 --t 8 --seed 1
```

## Current Results

| Run | n | c | t | Validation | Host validation | Pair key bytes | Keygen us | OLE expand mean us |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| smoke | 8192 | 2 | 8 | pass | pass | 141,504 | 456 | 13,224 |
| bounded | 8192 | 2 | 64 | pass | pass | 9,044,160 | 4,797 | 865,253 |
| bounded | 16384 | 2 | 64 | pass | skipped | 9,633,984 | 5,296 | 1,830,210 |

The host oracle validation is enabled for the small bounded case and intentionally skipped at `n=16384` to keep the sweep bounded.

## Scientific Caveats

- The current modulus is a single 62-bit prime, reported as requested `qbits=64`; it does not match the paper's `log p ~= 128` parameter.
- The current noise is uniform-position sparse noise. The paper-comparable key-size configuration is regular noise, which reduces the per-point SPFSS domain from `[0, 2N)` to a bucketed domain.
- The direct OLE benchmark stops at OLE. The follow-up linear artifact converts OLEs into ring-polynomial Beaver products, but it does not yet produce Orca-compatible scalar Beaver triples.
- There is no `Z_p -> Z_{2^bw}` share conversion yet, so this is not ready for `gpuMatmulBeaver`.
- The current OLE benchmark uses full SPFSS evaluation for clarity and validation. It is correctness-first, not the final optimized scheduling path.
- SPFSS tree expansion uses the AES PRG path, while initial key seeds are deterministically derived from the benchmark seed for reproducible experiments.

## Recommended Next Steps

1. Add regular-noise indexing and oracle support, then rerun the bounded OLE sweep for paper-comparable key-size numbers.
2. Lift the modulus path to dual-prime CRT for requested `qbits=128`.
3. Extend the new ring-polynomial OLE-to-Beaver artifact with a scalar packing model for Orca tensor entries.
4. Add a written `Z_p -> Z_{2^bw}` share-conversion argument.
5. Integrate the resulting triple source behind Orca's linear-layer keygen path and compare against baseline Beaver triples.
6. Optimize SPFSS scheduling only after the above correctness boundaries are locked.
