# Ring-LPN OLE Linear-Layer Beaver Artifact

Configuration: ring-polynomial matrix multiplication over the single 62-bit prime. Noise mode(s): uniform. Each ring product uses two Figure 2 OLE instances to form Beaver shares.

| rows | inner | cols | n | c | t | noise | SPFSS domain | validation | shared operands | OLE instances | key bytes MiB | keygen us | linear expand mean us | linear expand std us |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 2 | 8192 | 2 | 8 | uniform | 16384 | pass | 1 | 16 | 2.16 | 8,491.000 | 229,748.000 | 0.000 |

Notes:
- This is the two-OLE-to-Beaver conversion applied to a linear layer whose entries are Ring-LPN polynomials.
- `shared operands = 1` means every `A[row,k]` and `B[k,col]` sparse operand share was generated once and reused across the matrix product.
- It validates Beaver correctness for matrix multiplication over `Z_p[X]/(X^N+1)`.
- Full Orca integration remains separate; the tiny FC key-writer demo is reported in `orca_fc_ringlpn_demo_*.md`.
