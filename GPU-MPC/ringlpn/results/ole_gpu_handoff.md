# GPU Figure 2 OLE Handoff

Updated: 2026-05-15

## Status

The standalone GPU Figure 2 Ring-LPN OLE artifact is implemented and validated for these configurations:

- modulus: requested `qbits=64`, actual single 62-bit prime `p = 4611686018326724609`,
- noise: uniform-position sparse noise and regular sparse noise with one position per bucket,
- SPFSS domain: `[0, 2N)`, folded to degree `< N` using `X^N = -1`,
- benchmark scope: OLE only, with no Orca Beaver-triple conversion or nonlinear FSS integration.

For regular noise, the implementation groups point functions by bucket-sum and evaluates SPFSS over domain size `2N/t`, then scatters the grouped output back into `[0, 2N)` before folding.

The validated claim is:

`z_0 + z_1 == x_0 * x_1` in `Z_p[X]/(X^N + 1)`.

This is a correctness and systems artifact for Figure 2's SPFSS-based OLE assembly. It is not yet a paper-parameter reproduction, a trusted-dealer removal for Orca, or an end-to-end linear-layer integration.

Follow-up status: `results/linear_ole_handoff.md` now records the first OLE-to-Beaver linear-layer artifact over ring-polynomial matrix entries. `results/orca_zp_bridge_handoff.md` records a host-only scalar bridge smoke for constant-polynomial packing and carry-corrected dealer/oracle conversion from `Z_p` shares to `Z_{2^bw}` shares. These follow-ups still do not constitute Orca FC integration because there is no Orca key writer, q128/CRT path, high-density packing, or secure distributed share conversion yet.

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
| `results/ole_gpu_q64_uniform_c2_t64.md` | Bounded uniform-noise result summary |
| `results/ole_gpu_q64_regular_c2_t8_smoke.md` | Regular-noise smoke result summary |
| `results/ole_gpu_q64_regular_c2_t64.md` | Bounded regular-noise result summary |
| `results/linear_ole_handoff.md` | OLE-to-Beaver ring-polynomial linear-layer follow-up |
| `results/orca_zp_bridge_handoff.md` | Orca-facing scalar bridge boundary and q62/full-32-bit counterexample |

The new arithmetic SPFSS path is intentionally separate from the existing packed one-bit `gpu_dpf.cu` path, so current ReLU, DCF, LUT, and bit-output callers are unchanged.

## Reproduction

Run inside the `orca-dev` container from `/home/ringlpn`:

```bash
./scripts/build_ole_cuda_bench.sh
./bin/test_spfss_zp_cuda

SMOKE=1 ./scripts/run_ole_sweep.sh
./scripts/run_ole_sweep.sh
SMOKE=1 NOISE=regular ./scripts/run_ole_sweep.sh
NOISE=regular ./scripts/run_ole_sweep.sh
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
| uniform smoke | 8192 | 2 | 8 | pass | pass | 141,504 | 448 | 13,316 |
| regular smoke | 8192 | 2 | 8 | pass | pass | 116,544 | 5,006 | 6,825 |
| uniform bounded | 8192 | 2 | 64 | pass | pass | 9,044,160 | 4,797 | 865,253 |
| uniform bounded | 16384 | 2 | 64 | pass | skipped | 9,633,984 | 5,296 | 1,830,210 |
| regular bounded | 8192 | 2 | 64 | pass | pass | 5,529,408 | 40,828 | 58,462.5 |
| regular bounded | 16384 | 2 | 64 | pass | skipped | 6,119,232 | 42,331 | 67,733 |

The host oracle validation is enabled for the small bounded case and intentionally skipped at `n=16384` to keep the sweep bounded.

## Scientific Caveats

- The current modulus is a single 62-bit prime, reported as requested `qbits=64`; it does not match the paper's `log p ~= 128` parameter.
- Regular-noise bounded numbers are now saved for `n in {8192, 16384}`, `c=2`, `t=64`, but they still use the single 62-bit prime and are therefore not the paper's CRT-sized modulus setting.
- The direct OLE benchmark stops at OLE. The follow-up linear artifact converts OLEs into ring-polynomial Beaver products, but it does not yet produce Orca-compatible scalar Beaver triples.
- A host-only dealer/oracle `Z_p -> Z_{2^bw}` share conversion smoke now exists, but there is no secure distributed conversion protocol or Orca key writer yet, so this is not ready for `gpuMatmulBeaver`.
- The current OLE benchmark uses full SPFSS evaluation for clarity and validation. It is correctness-first, not the final optimized scheduling path.
- SPFSS tree expansion uses the AES PRG path, while initial key seeds are deterministically derived from the benchmark seed for reproducible experiments.

## Recommended Next Steps

1. Lift the modulus path to dual-prime CRT for requested `qbits=128`.
2. Extend the new ring-polynomial OLE-to-Beaver artifact with an Orca key writer, starting with the conservative constant-polynomial scalar bridge.
3. Replace or justify the dealer/oracle `Z_p -> Z_{2^bw}` conversion with a secure conversion protocol if trusted-dealer removal remains the claim.
4. Integrate the resulting triple source behind Orca's linear-layer keygen path and compare against baseline Beaver triples.
5. Optimize SPFSS scheduling only after the above correctness boundaries are locked.
