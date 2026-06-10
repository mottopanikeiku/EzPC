# GPU Figure 2 OLE over Ring-LPN/SPFSS

Configuration: single 62-bit prime, noise mode(s): regular, folded into `Z_p[X]/(X^N+1)`.

| n | c | t | noise | SPFSS domain | validation | host validation | key bytes MiB | keygen us | OLE expand mean us | OLE expand std us |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 8192 | 2 | 64 | regular | 256 | pass | pass | 5.27 | 42,571.000 | 60,992.000 | 25.000 |
| 16384 | 2 | 64 | regular | 512 | pass | skipped | 5.84 | 43,702.000 | 69,385.000 | 0.000 |

Notes:
- `requested_qbits=64` maps to the promoted single 62-bit prime.
- This artifact stops at OLE: it validates `z_0 + z_1 == x_0 * x_1`; Orca FC integration is follow-up work.
- Regular noise uses one position per bucket and grouped SPFSS domains of size `2N/t`; CRT q128 remains a separate follow-up.
