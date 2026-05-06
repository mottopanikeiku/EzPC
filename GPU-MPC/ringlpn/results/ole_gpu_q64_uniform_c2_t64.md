# GPU Figure 2 OLE over Ring-LPN/SPFSS

Configuration: single 62-bit prime, uniform sparse noise, SPFSS domain `[0, 2N)`, folded into `Z_p[X]/(X^N+1)`.

| n | c | t | validation | host validation | key bytes MiB | keygen us | OLE expand mean us | OLE expand std us |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 8192 | 2 | 64 | pass | pass | 8.63 | 4,797.000 | 865,253.000 | 820.000 |
| 16384 | 2 | 64 | pass | skipped | 9.19 | 5,296.000 | 1,830,210.000 | 0.000 |

Notes:
- `requested_qbits=64` maps to the promoted single 62-bit prime.
- This artifact stops at OLE: it validates `z_0 + z_1 == x_0 * x_1`; Beaver triple conversion and Orca FC integration are follow-up work.
- Uniform noise is intentionally the first-pass configuration; regular-noise and CRT lifts are separate follow-ups.
