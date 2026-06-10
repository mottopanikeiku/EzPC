# Orca FC Ring-LPN v1 Demo

Configuration: small FC train+infer suite with q62 and q128 dealer/oracle constant-polynomial masks, `poly_n=8192`, `c=2`, `t=8`, regular-noise label, `tf=None`, and zero bias.

| q req | q actual | seed | second seed | shape | bw | bound | key bytes / party | baseline bytes / party | carry conversion | replay | second seed | forward | dW | dX | baseline | baseline matches | validation |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- |
| 64 | 62 | 1 | 2 | 2x2x2 | 16 | 255 | 96 | 96 | 1 | 1 | 1 | pass | pass | pass | pass | 1 | pass |
| 64 | 62 | 3 | 4 | 2x3x2 | 16 | 255 | 128 | 128 | 1 | 1 | 1 | pass | pass | pass | pass | 1 | pass |
| 64 | 62 | 5 | 6 | 3x2x2 | 16 | 255 | 128 | 128 | 1 | 1 | 1 | pass | pass | pass | pass | 1 | pass |
| 64 | 62 | 7 | 8 | 2x2x3 | 32 | 255 | 128 | 128 | 1 | 1 | 1 | pass | pass | pass | pass | 1 | pass |
| 128 | 124 | 9 | 10 | 2x2x2 | 32 | 4294967295 | 96 | 96 | 1 | 1 | 1 | pass | pass | pass | pass | 1 | pass |

Notes:
- The demo writes raw party buffers in `FCLayer::readForwardKey` order: `A`, `B`, `C_masked`.
- The synthetic training checks exercise `dW` and `dX` with the same C-share writer used by the feature-flagged `FCLayer::genBackwardKey` path.
- `C_masked` is formed by converting q62 or q128 CRT Beaver-product shares to `Z_{2^bw}` with the carry-corrected dealer/oracle bridge and then adding an output mask in the ring.
- The online phase calls the existing `gpuMatmulBeaver` implementation unchanged and reconstructs `clear FC output + output mask`.
- The baseline column is generated with Orca's `gpuKeygenMatmul` using the same masks and deterministic P0/P1 random-share stream, then run through the same online path.
- This is a v1 correctness demo, not high-density packing, secure distributed conversion, or trusted-dealer removal.
