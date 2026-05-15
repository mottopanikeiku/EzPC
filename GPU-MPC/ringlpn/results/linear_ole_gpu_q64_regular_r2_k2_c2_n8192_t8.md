# Ring-LPN OLE Linear-Layer Beaver Artifact

Configuration: ring-polynomial matrix multiplication over the single 62-bit prime. Noise mode(s): regular. Each ring product uses two Figure 2 OLE instances to form Beaver shares.

| rows | inner | cols | n | c | t | noise | SPFSS domain | validation | OLE instances | key bytes MiB | keygen us | linear expand mean us | linear expand std us |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 2 | 8192 | 2 | 8 | regular | 2048 | pass | 16 | 1.78 | 79,162.000 | 114,014.000 | 0.000 |

Notes:
- This is the two-OLE-to-Beaver conversion applied to a linear layer whose entries are Ring-LPN polynomials.
- It validates Beaver correctness for matrix multiplication over `Z_p[X]/(X^N+1)`.
- It is not yet Orca FC integration: scalar packing and `Z_p -> Z_{2^bw}` share conversion remain follow-up work.
