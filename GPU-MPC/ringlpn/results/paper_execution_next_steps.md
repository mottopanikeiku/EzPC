# Ring-LPN Paper Execution Next Steps

Updated: 2026-05-15

## Current Verified Direction

The repository is in a staged-but-coherent paper path:

1. The promoted cheddar-derived CUDA path is the default single-prime NTT/PolyMul backend for requested `q=32` and `q=64`.
2. The standalone Figure 2 SPFSS/OLE artifact validates `z_0 + z_1 = x_0 * x_1` in `Z_p[X]/(X^N+1)` for single-prime q62, uniform sparse noise, and regular sparse noise.
3. The standalone linear artifact converts two Figure 2 OLEs into a Beaver ring-polynomial matrix product.
4. The new host bridge smoke validates the exact dealer/oracle carry correction for exporting `Z_p` shares to Orca's `Z_{2^bw}` ring under a conservative constant-polynomial scalar packing model.
5. The new tiny Orca FC demo writes raw `A`, `B`, `C_masked` buffers for a bounded small-shape forward suite, validates the unchanged `gpuMatmulBeaver` online contract, and matches Orca's `gpuKeygenMatmul` baseline under the same masks.
6. The same bridge smoke intentionally records a q62/full-32-bit counterexample, so the current direction does not overclaim unrestricted Orca scalar products.

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
- uniform and regular linear OLE-to-Beaver smokes,
- the tiny Orca FC Ring-LPN key-writer demo.

Set `REQUIRE_GPU_SMOKE=1` if CI should fail when CUDA/NVCC is unavailable.

## Repository Hygiene Notes

- Generated host binaries live under `ringlpn/host_bin/`, which is ignored.
- CUDA binaries under `ringlpn/bin/` are container-owned build artifacts and remain ignored.
- Existing dirty submodule worktrees under Orca datasets/weights, CUTLASS, and NFLlib are dependency/cache state; they are intentionally left untouched by this checkpoint.
- Verify GitHub state with `git ls-remote origin refs/heads/master`; local cleanup checkpoints may be ahead if HTTPS credentials are unavailable.

## Latest Verification

On 2026-05-15, the consolidated smoke passed on the host and inside the running `orca-dev` container.

Host command:

```bash
GPU-MPC/ringlpn/scripts/run_paper_checkpoint_smoke.sh
```

Container command:

```bash
docker exec orca-dev bash -lc 'cd /home/ringlpn && RUN_GPU_SMOKE=1 ./scripts/run_paper_checkpoint_smoke.sh'
```

Validated results:

- `test_spfss_zp_cuda`: single point, multiple points, colliding alphas, and edge alphas all passed.
- Uniform OLE smoke: validation `pass`, host validation `pass`, keygen `455 us`, expand mean `13,330 us`.
- Regular OLE smoke: validation `pass`, host validation `pass`, keygen `5,061 us`, expand mean `6,960 us`.
- Uniform linear OLE-to-Beaver smoke: validation `pass`, `shared_operands=1`, 16 OLE instances, keygen `6,587 us`, expand mean `223,667 us`.
- Regular linear OLE-to-Beaver smoke: validation `pass`, `shared_operands=1`, 16 OLE instances, keygen `81,582 us`, expand mean `114,825 us`.
- Orca FC demo bounded suite: four cases pass (`2x2x2 bw16`, `2x3x2 bw16`, `3x2x2 bw16`, `2x2x3 bw32`), each with online contract `pass`, deterministic replay `1`, second-seed validation `1`, baseline `pass`, and `baseline_matches_ringlpn=1`.

The build emitted only existing third-party Eigen/cryptoTools warnings.

## Immediate Next Implementation Checkpoints

### Checkpoint 1: Tiny Orca-Compatible Key Writer

Status: complete for the current bounded small-shape suite and baseline comparison.

Goal: keep this as a regression suite while the next research blockers land.

Scope:

- keep the constant-polynomial bridge,
- restrict values so `inner * value_bound^2 < p`,
- compare future changes against baseline `gpuKeygenMatmul`,
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
