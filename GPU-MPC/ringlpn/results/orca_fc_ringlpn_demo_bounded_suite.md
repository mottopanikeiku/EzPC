# Orca FC Ring-LPN v1 Demo

Configuration: forward-only small FC suite with bounded q62 constant-polynomial masks, `poly_n=8192`, `c=2`, `t=8`, regular-noise label, `tf=None`, and zero bias.

| seed | second seed | shape | bw | bound | key bytes / party | baseline bytes / party | carry conversion | replay | second seed | baseline | baseline matches | validation |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 1 | 2 | 2x2x2 | 16 | 1 | 96 | 96 | 1 | 1 | 1 | pass | 1 | pass |
| 3 | 4 | 2x3x2 | 16 | 1 | 128 | 128 | 1 | 1 | 1 | pass | 1 | pass |
| 5 | 6 | 3x2x2 | 16 | 1 | 128 | 128 | 1 | 1 | 1 | pass | 1 | pass |
| 7 | 8 | 2x2x3 | 32 | 1 | 128 | 128 | 1 | 1 | 1 | pass | 1 | pass |

Notes:
- The demo writes raw party buffers in `FCLayer::readForwardKey` order: `A`, `B`, `C_masked`.
- `C_masked` is formed by converting q62 Beaver-product shares to `Z_{2^bw}` with the carry-corrected bridge and then adding an output mask in the ring.
- The online phase calls the existing `gpuMatmulBeaver` implementation unchanged and reconstructs `clear FC output + output mask`.
- The baseline column is generated with Orca's `gpuKeygenMatmul` using the same masks and deterministic P0/P1 random-share stream, then run through the same online path.
- This is a v1 correctness demo, not q128/CRT, high-density packing, or secure distributed q62-to-ring conversion.
