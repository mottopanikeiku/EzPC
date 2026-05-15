# v1 Ring-LPN Linear-Layer Orca FC Demo Memo

Updated: 2026-05-15

## Claim

We now have a complete v1 forward-only Orca FC demo for the conservative Ring-LPN scalar bridge path.

The demo generates correlated party key buffers for a `2x2 * 2x2` FC layer with `bw=16`, `value_bound=255`, `poly_n=8192`, `c=2`, `t=8`, regular-noise label, `tf=None`, and zero bias. The buffers are serialized in the exact `FCLayer::readForwardKey` / `readGPUMatmulKey` order: `A`, `B`, `C_masked`. The online phase calls the existing `gpuMatmulBeaver` implementation unchanged and reconstructs `clear FC output + output_mask` in `Z_{2^16}`.

This is professor-presentable as a correctness demo. It is not yet paper-parameter q128, high-density packing, secure distributed conversion, training/backward integration, or trusted-dealer removal.

## Proof Sketch

For an Orca FC Beaver mask, the online contract is:

- party keys contain additive shares of `A`, `B`, and `C_masked`,
- the clear online operands are `X + A` and `W + B`,
- `C_masked = A * B + output_mask` in the Orca ring,
- the unchanged `gpuMatmulBeaver` algebra reconstructs `X * W + output_mask`.

The q62-to-ring bridge is valid only under the stated no-prime-wrap bound. For this demo, `inner * value_bound^2 = 2 * 255^2 = 130050 < p`, where `p = 4611686018326724609`. Therefore the q62 field dot product equals the integer dot product before reduction, and carry-corrected export from `Z_p` additive shares to `Z_{2^16}` additive shares preserves the target ring value.

The required carry correction is:

- `r0 = z0 mod 2^bw`,
- `r1 = z1 - m*p mod 2^bw`,
- `m = floor((z0 + z1) / p)`.

The bridge smoke keeps the negative q62/full-32-bit row to show that unrestricted 32-bit products are not claimed.

## Exact Command Log

Host bridge:

```bash
docker exec orca-dev bash -lc 'cd /home/ringlpn && scripts/run_orca_zp_bridge_test.sh'
```

Shared matrix artifact:

```bash
docker exec orca-dev bash -lc 'cd /home/ringlpn && scripts/build_linear_ole_bench.sh'
docker exec orca-dev bash -lc 'cd /home/ringlpn && scripts/run_linear_ole_sweep.sh && NOISE=regular scripts/run_linear_ole_sweep.sh'
```

Orca FC demo:

```bash
docker exec orca-dev bash -lc 'cd /home/ringlpn && scripts/build_orca_fc_ringlpn_demo.sh'
docker exec orca-dev bash -lc 'cd /home/ringlpn && scripts/run_orca_fc_ringlpn_demo.sh'
```

Consolidated paper checkpoint:

```bash
docker exec orca-dev bash -lc 'cd /home/ringlpn && RUN_GPU_SMOKE=1 scripts/run_paper_checkpoint_smoke.sh'
```

## Result Table

| Artifact | Parameters | Key result |
| --- | --- | --- |
| Host bridge | `bw=16`, `rows=2`, `inner=2`, `cols=2`, `value_bound=255` | `corrected_share_failures=0`, scalar validation `pass` |
| Host counterexample | `bw=32`, `value_bound=4294967295` | q62/full-32-bit counterexample remains present |
| Shared linear OLE, uniform | `n=8192`, `c=2`, `t=8`, `rows=2`, `inner=2`, `cols=2` | validation `pass`, `shared_operands=1`, 16 OLE instances |
| Shared linear OLE, regular | `n=8192`, `c=2`, `t=8`, `rows=2`, `inner=2`, `cols=2` | validation `pass`, `shared_operands=1`, 16 OLE instances |
| Orca FC demo | `bw=16`, `poly_n=8192`, `c=2`, `t=8`, seeds `1` and `2` | online contract `pass`, replay `1`, second-seed validation `1` |

Primary output files:

- `results/orca_zp_bridge_constant_scalar.md`
- `results/linear_ole_gpu_q64_uniform_r2_k2_c2_n8192_t8.md`
- `results/linear_ole_gpu_q64_regular_r2_k2_c2_n8192_t8.md`
- `results/orca_fc_ringlpn_demo_seed1_seed2.md`

## Correctness Boundary

Valid to say:

- the ring-polynomial linear artifact now validates a true shared matrix product, because each `A[row,k]` and `B[k,col]` operand share is generated once and reused across products,
- the scalar bridge correctly exports bounded q62 Beaver-product shares into `Z_{2^16}`,
- the tiny Orca FC demo writes raw correlated party buffers and uses the unchanged `gpuMatmulBeaver` online path successfully.

Not valid to say yet:

- q128/CRT is implemented,
- unrestricted 32-bit Orca products are supported by the single q62 path,
- dense packing is implemented,
- secure distributed `Z_p -> Z_{2^bw}` conversion is implemented,
- Orca training/backward/optimizer keys are integrated,
- trusted-dealer removal is complete.

## Remaining Paper Gaps

1. Implement q128/CRT or prove tighter layer-wise bounds for every claimed model layer.
2. Replace constant-polynomial one-scalar-per-polynomial packing with a dense packing scheme and host oracle tests.
3. Specify or implement secure distributed `Z_p -> Z_{2^bw}` share conversion for parties that do not know both q62 shares.
4. Extend beyond forward FC to backward, optimizer, and model-level Orca execution.
5. Rerun paper-scale OLE and linear measurements after q128/CRT and packing are in place.
