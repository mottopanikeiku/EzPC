# Ring-LPN Paper Execution Next Steps

Updated: 2026-05-15

## Current Verified Direction

The repository is in a staged-but-coherent paper path:

1. The promoted cheddar-derived CUDA path is the default single-prime NTT/PolyMul backend for requested `q=32` and `q=64`.
2. The standalone Figure 2 SPFSS/OLE artifact validates `z_0 + z_1 = x_0 * x_1` in `Z_p[X]/(X^N+1)` for single-prime q62, uniform sparse noise, and regular sparse noise.
3. The standalone linear artifact converts two Figure 2 OLEs into a Beaver ring-polynomial matrix product.
4. The new host bridge smoke validates the exact dealer/oracle carry correction for exporting `Z_p` shares to Orca's `Z_{2^bw}` ring under a conservative constant-polynomial scalar packing model.
5. The same bridge smoke intentionally records a q62/full-32-bit counterexample, so the current direction does not overclaim unrestricted Orca scalar products.

## One-Command Smoke

From the host repository root:

```bash
GPU-MPC/ringlpn/scripts/run_paper_checkpoint_smoke.sh
```

This runs:

- shell syntax checks for Ring-LPN scripts,
- the host-only Orca `Z_p -> Z_{2^bw}` bridge build,
- the deterministic bridge smoke that regenerates `results/orca_zp_bridge_constant_scalar.csv` and `.md`.

Inside the `orca-dev` container, run the GPU smoke from `/home/ringlpn`:

```bash
RUN_GPU_SMOKE=1 ./scripts/run_paper_checkpoint_smoke.sh
```

This additionally builds and runs:

- `test_spfss_zp_cuda`,
- uniform and regular Figure 2 OLE smokes,
- uniform and regular linear OLE-to-Beaver smokes.

Set `REQUIRE_GPU_SMOKE=1` if CI should fail when CUDA/NVCC is unavailable.

## Repository Hygiene Notes

- Generated host binaries live under `ringlpn/host_bin/`, which is ignored.
- CUDA binaries under `ringlpn/bin/` are container-owned build artifacts and remain ignored.
- Existing dirty submodule worktrees under Orca datasets/weights, CUTLASS, and NFLlib are dependency/cache state; they are intentionally left untouched by this checkpoint.
- The pushed GitHub fork `origin/master` currently resolves to the latest bridge checkpoint commit.

## Immediate Next Implementation Checkpoints

### Checkpoint 1: Tiny Orca-Compatible Key Writer

Goal: produce `(A, B, C)` key arrays matching `GPUMatmulKey<T>` for a tiny FC-only case while keeping online `gpuMatmulBeaver` unchanged.

Scope:

- start with the constant-polynomial bridge,
- restrict values so `inner * value_bound^2 < p`,
- compare against baseline `gpuKeygenMatmul` on a tiny synthetic FC case,
- keep this as correctness-only, not a performance claim.

### Checkpoint 2: q128/CRT Lift

Goal: remove the current q62/full-32-bit limitation before paper-comparable Orca linear claims.

Scope:

- add dual-prime scheduling to the promoted CUDA path,
- add CRT recomposition or an equivalent two-limb share-export path,
- rerun NTT, OLE, and linear smokes under requested `q=128`,
- update all result tables with requested and actual modulus details.

### Checkpoint 3: Secure Share Conversion Boundary

Goal: replace the current dealer/oracle conversion with a protocol-level argument or implementation if the paper claim is trusted-dealer removal.

Scope:

- specify what each party knows at the conversion boundary,
- implement or cite a secure conversion from `Z_p` additive shares to `Z_{2^bw}` additive shares,
- keep the current carry-correction harness as the oracle reference.

### Checkpoint 4: Dense Packing

Goal: improve efficiency after correctness is fixed.

Scope:

- define a slot packing model for Orca tensors,
- prove the resulting convolution/sign-fold behavior in `Z_p[X]/(X^N+1)`,
- add host oracle tests before GPU benchmarking,
- only then measure model-level layers.
