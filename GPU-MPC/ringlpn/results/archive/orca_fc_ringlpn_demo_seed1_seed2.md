# Orca FC Ring-LPN v1 Demo

Configuration: forward-only `2x2 * 2x2` FC layer, `bw=16`, bounded q62 constant-polynomial masks, `poly_n=8192`, `c=2`, `t=8`, regular-noise label, `tf=None`, zero bias.

| seed | second seed | shape | key bytes / party | carry conversion | replay | second seed | distinct second seed | online contract | validation |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2 | 2x2x2 | 96 | 1 | 1 | 1 | 1 | pass | pass |

Notes:
- The demo writes raw party buffers in `FCLayer::readForwardKey` order: `A`, `B`, `C_masked`.
- `C_masked` is formed by converting q62 Beaver-product shares to `Z_{2^16}` with the carry-corrected bridge and then adding an output mask in the ring.
- The online phase calls the existing `gpuMatmulBeaver` implementation unchanged and reconstructs `clear FC output + output mask`.
- This is a v1 correctness demo, not q128/CRT, high-density packing, or secure distributed q62-to-ring conversion.
