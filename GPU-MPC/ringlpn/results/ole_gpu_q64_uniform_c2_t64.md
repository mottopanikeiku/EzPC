# GPU Figure 2 OLE over Ring-LPN/SPFSS

Configuration: single 62-bit prime, noise mode(s): uniform, folded into `Z_p[X]/(X^N+1)`.

| n | c | t | noise | SPFSS domain | validation | host validation | key bytes MiB | keygen us | OLE expand mean us | OLE expand std us |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 8192 | 2 | 64 | uniform | 16384 | pass | pass | 8.63 | 5,034.000 | 867,374.000 | 74.500 |
| 16384 | 2 | 64 | uniform | 32768 | pass | skipped | 9.19 | 5,212.000 | 1,840,560.000 | 0.000 |

Notes:
- `requested_qbits=64` maps to the promoted single 62-bit prime.
- This artifact stops at OLE: it validates `z_0 + z_1 == x_0 * x_1`; Orca FC integration is follow-up work.
- Regular noise uses one position per bucket and grouped SPFSS domains of size `2N/t`; CRT q128 remains a separate follow-up.
