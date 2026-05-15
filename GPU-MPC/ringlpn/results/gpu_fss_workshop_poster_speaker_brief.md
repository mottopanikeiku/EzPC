# Speaker Brief And Technical Backstory

Poster:

**Improving Memory Efficiency of GPU-Accelerated Function Secret Sharing**

Use this as the long-form preparation document for presenting the poster, answering questions, and handing context to a plotting or poster-generation agent. It intentionally contains more background than should appear on the final poster.

The single sentence:

**GPU-MPC/Orca already accelerates PPML on the GPU, but offline key material and correlation movement are becoming first-order bottlenecks; our standalone DPF chunking and Ring-LPN GPU building blocks show a validated path toward reducing peak memory pressure and accelerating PCG-style preprocessing, while full Orca integration remains ongoing.**

## 1. The Story In Plain English

The project starts from a systems observation: privacy-preserving ML protocols often make online execution fast by moving expensive work into an offline preprocessing phase. That works, but it creates a new problem. The offline phase generates a large amount of key material and correlated randomness. During actual training or inference, the system has to store, read, move, and stage that material.

In Orca, a GPU-accelerated FSS-based PPML system, this is already visible. The GPU can be fast enough that key movement becomes comparable to compute. For P-LeNet, the saved local profile shows average key-read time of 109.727 ms and average compute time of 107.727 ms. For P-AlexNet, key read is 104.818 ms and compute is 121.727 ms. This is not because the implementation is completely naive: the code already uses direct I/O, aligned buffers, and read/compute overlap.

The poster then presents two complementary building-block directions:

- For FSS-style non-linear operations, use chunked online DPF key generation so the system does not need to stage a huge eval-all key all at once.
- For additive-secret-sharing linear operations, use GPU Ring-LPN PCG building blocks: fast NTT/PolyMul, then VOLE expansion, then OLE-to-Beaver bridge artifacts.

The current work is not a finished replacement of Orca preprocessing. It is a set of validated standalone systems experiments that support the direction.

## 2. How To Present It

### 30-Second Pitch

“This poster is about memory pressure in GPU-accelerated secure ML. Orca already pushes computation to the GPU, but profiling shows key reads can match compute time for larger local runs. We tested two building blocks toward fixing that: chunked DPF online key generation for FSS keys, which reduces peak staged pair-key footprint by 128x at chunk size 8192 with 1.834x overhead, and GPU Ring-LPN PCG building blocks, where our promoted GPU NTT/PolyMul backend gives 89.24x per-polynomial full-PolyMul speedup over NFLLib at n=8192. VOLE and OLE-to-Beaver prototypes validate the bridge toward linear-layer preprocessing. The honest boundary is that these are standalone validated artifacts; full Orca integration is ongoing.”

### 2-Minute Walkthrough

1. Start with PPML structure: non-linear operations use FSS; linear operations use additive secret sharing.
2. Explain the offline/online split: online is fast because keys/correlations are precomputed, but that means key material gets huge.
3. Point to Orca profiling: P-LeNet key read 109.727 ms vs compute 107.727 ms; P-AlexNet key read 104.818 ms vs compute 121.727 ms.
4. Emphasize that Orca already has direct I/O and overlap, so this is a deeper memory-movement issue.
5. Explain DPF chunking: full mode materializes the whole eval-all pair key; chunked mode materializes one fixed-size chunk at a time.
6. Show the headline: at N=1048576, chunk=8192 keeps peak at 2.81 MiB instead of 360.00 MiB, giving 128.00x peak reduction with 1.834x time overhead.
7. Move to Ring-LPN: PCG-style expansion needs fast polynomial multiplication. The GPU backend validates q=32/q=64 requested sweeps over n=8192 to 1048576.
8. Show the direct CPU/GPU comparison: 87.59x forward NTT and 89.24x full PolyMul speedup at n=8192.
9. Explain VOLE: it validates z = y + x * Delta over the full tested range for requested q=32 and q=64.
10. Close with boundary: Figure 2 OLE and OLE-to-Beaver are bridge artifacts; Orca scalar packing, share conversion, and triple writer are next.

### 5-Minute Deep Dive

Spend roughly one minute per block:

- System motivation: Orca and offline/online preprocessing.
- Profiling evidence: key read, compute, communication, direct I/O context.
- DPF chunking: memory model, tradeoff, validation.
- Ring-LPN backend: NTT/PolyMul, q mapping, speedup evidence.
- VOLE/OLE bridge: algebraic relations, validation, integration roadmap.

If someone is from systems/GPU, emphasize memory movement, batching, direct I/O, and per-polynomial throughput.

If someone is from cryptography, emphasize the relations being validated, the distinction between DPF/FSS and additive sharing, and the current security/parameter boundaries.

If someone is from ML/PPML, emphasize the PPML workload split, why ReLU/comparison and linear layers need different preprocessing, and what is still missing before a real Orca speed or memory claim.

## 3. Poster Panel Script

### Title

“The title is FSS-centered because Orca is a GPU-FSS PPML system, but the project looks at preprocessing pressure across both major PPML paradigms: FSS for non-linear layers and additive sharing for linear layers.”

### Problem Panel

Say:

“The issue is not that GPU compute is slow. The issue is that preprocessing material can become large enough that reading and staging keys is a first-order cost.”

Point to:

- P-LeNet: P0 4.0G, P1 4.0G; key read 109.727 ms; compute 107.727 ms.
- P-AlexNet: P0 3.8G, P1 3.8G; key read 104.818 ms; compute 121.727 ms.

### Method Panel

Say:

“We split the response into two tracks. For FSS non-linear operations, we test chunked online DPF key generation. For linear operations, we build GPU Ring-LPN PCG components.”

Then trace the pipeline:

```text
Orca/FSS workload
  -> profiling bottleneck
  -> chunked DPF online key generation
  -> GPU NTT/PolyMul
  -> Ring-LPN VOLE
  -> OLE-to-Beaver bridge
  -> Orca integration
```

### DPF Results Panel

Say:

“The key distinction is peak staged footprint versus total logical key material. Chunking does not make the logical key disappear. It changes how much has to be materialized at once.”

Main number:

- N=1048576, chunk=8192: full pair key 360.00 MiB, partial peak 2.81 MiB, 128.00x reduction, 1.834x overhead.

If asked about stronger chunking:

- chunk=4096: 255.99x reduction, 2.942x overhead.
- chunk=2048: 511.97x reduction, 4.975x overhead.

### NTT / PolyMul Results Panel

Say:

“For the polynomial backend, the poster uses one GPU implementation as the base GPU path. The comparison is CPU NFLLib versus the promoted GPU backend. We are not making this a three-way implementation comparison.”

Main number:

- Direct n=8192 comparison: 87.59x forward NTT speedup and 89.24x full PolyMul per-polynomial speedup over NFLLib.

Important qualifier:

- Requested q=32 is actual q=30.
- Requested q=64 is actual q=62.
- q=128/CRT is not implemented.

### VOLE / Bridge Panel

Say:

“VOLE is the PCG-style expansion layer. The prototype validates the coefficient-wise relation z = y + x * Delta, but it is not yet a full SPFSS-backed pipeline.”

Main numbers:

- q=32, n=8192: full expand 191.485 us.
- q=32, n=1048576: full expand 32144.700 us.
- q=64, n=8192: full expand 549.802 us.
- q=64, n=1048576: full expand 50952.700 us.

Bridge statement:

“Figure 2 OLE validates z_0 + z_1 == x_0 * x_1 in Z_p[X]/(X^N+1). The linear artifact uses two OLE instances per ring product to form Beaver shares over ring-polynomial matrix entries. It still does not emit Orca-compatible scalar Beaver triples.”

## 4. Experimental Setup

These fields were queried from the current host and running `orca-dev` container on 2026-05-11. The saved benchmark artifacts did not embed a separate environment snapshot for every run, so this should be presented as observed current runtime provenance unless the benchmark suite is re-run with environment capture.

Known setup:

| Item | Value |
| --- | --- |
| Project | `GPU-MPC` |
| Main PPML system | Orca |
| Main benchmark area | `GPU-MPC/ringlpn` |
| Container | `orca-dev` |
| Host GPU-MPC root | `/home/fatih/EzPC/GPU-MPC` |
| Container GPU-MPC root | `/home` |
| Container Ring-LPN workdir | `/home/ringlpn` |
| Current container OS | Ubuntu 22.04.4 LTS |
| Current CUDA toolkit in container | CUDA compilation tools 12.3, V12.3.107 |
| Driver-reported CUDA version | 12.6 |
| Current compiler in container | gcc/g++ 9.5.0 |
| Python in container | Python 3.10.12 |
| README-tested environment | Ubuntu 20.04, CUDA 11.7, CMake 3.27.2, g++-9 |

Observed hardware/software:

| Item | Current poster value |
| --- | --- |
| GPU model | 4x NVIDIA RTX 5000 Ada Generation |
| GPU memory | 32760 MiB per GPU |
| GPU compute capability | 8.9 |
| CPU model | Intel Xeon w5-3435X, 16 cores / 32 threads, 1 socket |
| RAM | 109 GiB system memory, 9 GiB swap |
| NVIDIA driver | 560.35.03 |
| CUDA runtime/toolkit observed | driver-reported CUDA 12.6; container nvcc CUDA 12.3.107 |
| Compiler observed | gcc/g++ 9.5.0; nvcc 12.3.107 |
| Container image tag | `fatih` |
| Container image ID | `sha256:8734209bcc3b2f07fd99f236ba499a4fa7d0e8cda2ee109ddf2ebc9ea6d17b0c` |
| Container ID | `7706f2441465100149beb1c8455bffae73ce00f48efcb34efa1fc645ea9886f8` |

Reproduction commands:

```bash
# NTT / PolyMul core
docker exec -w /home/ringlpn orca-dev bash scripts/run_cuda_sweep.sh
docker exec -e QBITS=64 -w /home/ringlpn orca-dev bash scripts/run_cuda_sweep.sh

# Ring-LPN VOLE baseline sweeps
docker exec -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh
docker exec -e QBITS=64 -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh

# Ring-LPN VOLE sensitivity sweep
docker exec -e M=64 -w /home/ringlpn orca-dev bash scripts/run_vole_sweep.sh

# DPF online key-generation sweeps
docker exec -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py
docker exec -e CHUNK_SIZE=4096 -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py
docker exec -e CHUNK_SIZE=2048 -w /home orca-dev python3 scripts/run_dpf_online_keygen_sweep.py
```

What was measured:

- Orca profiling: key file sizes, average key-read time, average compute time, communication per iteration.
- DPF benchmark: full pair key footprint, partial peak pair-key footprint, peak reduction, total bytes multiplier, full keygen mean, partial pipeline mean, time overhead.
- NTT/PolyMul: NTT mean, INTT mean, full PolyMul mean, per-poly PolyMul, PolyMul polys/s, estimated coefficient GB/s, validation.
- VOLE: full expand mean, per-output expand, outputs/s, pair PolyMuls/s, coefficient-wise validation.
- OLE bridge: key bytes, keygen time, expand time, relation validation.

## 5. Theory And Algorithms

### 5.1 PPML Split: Non-Linear Versus Linear

Privacy-preserving ML protocols often split computation by operation type:

- Non-linear functions such as ReLU, comparison, maxpool, and activation logic are hard under plain additive sharing. FSS/DPF-style tools are useful here.
- Linear layers such as matrix multiplication and convolution are naturally handled with additive shares and Beaver-style preprocessing.

The system wants online execution to be fast. To do that, it precomputes material offline:

- FSS keys for non-linear functions.
- Beaver triples or related correlations for linear functions.

This shifts the cost to storage and movement of correlated randomness.

### 5.2 Additive Secret Sharing

For a ring R, a value x is shared as:

```text
x = x_0 + x_1 mod R
```

Party 0 holds x_0 and party 1 holds x_1. Linear operations are easy:

```text
(x + y)_i = x_i + y_i
(c * x)_i = c * x_i
```

Multiplication is harder because:

```text
x * y != x_0 * y_0 + x_1 * y_1
```

The cross terms matter. This is why linear-layer preprocessing often uses Beaver triples.

### 5.3 Beaver Triples

A Beaver triple is a shared random multiplication:

```text
A, B random
C = A * B
```

Each party has shares:

```text
A = A_0 + A_1
B = B_0 + B_1
C = C_0 + C_1
```

To multiply secret values X and Y online, parties open:

```text
E = X - A
F = Y - B
```

Then compute shares of:

```text
X * Y = C + E * B + F * A + E * F
```

The offline triple makes the online multiplication cheap. But triples must be generated, stored, moved, and consumed.

### 5.4 FSS And DPF Intuition

Function Secret Sharing splits a function f into keys k_0 and k_1. Each party evaluates its key locally:

```text
Eval(k_0, x) + Eval(k_1, x) = f(x)
```

A Distributed Point Function is an FSS for a point function:

```text
f_{alpha,beta}(x) = beta if x = alpha, otherwise 0
```

DPFs are useful building blocks for private lookup, comparison-like operations, and non-linear PPML components.

The problem is eval-all expansion. If the domain has N points, materializing all outputs for both parties can require O(N) staged key/output material. That is exactly where chunking helps.

### 5.5 DPF Chunked Online Key Generation

Full mode:

```text
generate full pair key for all N points
stage both parties' full eval-all key material
validate layout and metadata
```

Chunked mode:

```text
for start in 0..N step chunk_size:
    generate pair key material only for this chunk
    validate chunk layout and metadata
    consume or discard chunk
```

The peak memory goes from roughly O(N) to O(chunk_size). The total logical key material is not eliminated; it is streamed or generated in smaller pieces.

For N=1048576 and chunk_size=8192:

```text
N / chunk_size = 1048576 / 8192 = 128
```

The measured peak reduction is exactly 128.00x:

```text
full pair key:        360.00 MiB
partial peak key:       2.81 MiB
time overhead:          1.834x
```

More aggressive chunking:

```text
chunk 4096 -> 255.99x peak reduction, 2.942x overhead
chunk 2048 -> 511.97x peak reduction, 4.975x overhead
```

Presentation line:

“Chunking is a knob. Smaller chunks save more peak memory, but they repeat more generation/setup work.”

### 5.6 Ring-LPN, PCGs, And Why Polynomial Multiplication Matters

A Pseudorandom Correlation Generator expands compact seed-like material into large correlated randomness. Ring-LPN style constructions operate over a polynomial ring:

```text
Z_p[X] / (X^N + 1)
```

Many operations reduce to polynomial multiplication in this ring. If polynomial multiplication is slow, PCG expansion is slow. If polynomial multiplication is fast on GPU, PCG expansion becomes a plausible online or just-in-time preprocessing path.

Negacyclic polynomial multiplication computes:

```text
c(X) = a(X) * b(X) mod (X^N + 1)
```

The NTT accelerates this by transforming convolution into pointwise multiplication:

```text
A = NTT(a)
B = NTT(b)
C_i = A_i * B_i
c = INTT(C)
```

The benchmark reports the full PolyMul path:

```text
NTT(a) + NTT(b) + pointwise multiply + INTT
```

and then divides by batch size for per-polynomial latency.

### 5.7 Requested q Versus Actual q

The GPU backend currently uses single-prime paths:

| Requested qbits | Actual qbits | Meaning |
| ---: | ---: | --- |
| 32 | 30 | one 30-bit prime |
| 64 | 62 | one 62-bit prime |

q=128 / CRT is not implemented. Do not imply paper-parameter q=128 support.

If asked why:

“The current artifact validates the single-prime backend first. q=128 needs multi-prime CRT scheduling and recomposition, which is the next benchmark-core step.”

### 5.8 VOLE Relation

Vector Oblivious Linear Evaluation produces correlated values satisfying:

```text
z = y + x * Delta
```

where Delta is a secret/global correlation value. In the standalone benchmark:

- inputs are synthesized locally under `synthetic_mpvole`,
- the benchmark computes x, y, and z through batched inner-product phases,
- validation checks the coefficient-wise relation.

Do not claim a CPU-vs-GPU VOLE speedup. There is no CPU VOLE baseline in the saved artifacts.

### 5.9 Figure 2 OLE Relation

The direct OLE artifact validates:

```text
z_0 + z_1 == x_0 * x_1
```

inside:

```text
Z_p[X] / (X^N + 1)
```

Uniform sparse noise evaluates SPFSS over:

```text
[0, 2N)
```

Regular sparse noise uses grouped SPFSS domains:

```text
2N / t
```

For the bounded regular runs:

```text
n=8192,  t=64 -> domain 256
n=16384, t=64 -> domain 512
```

### 5.10 OLE-To-Beaver Bridge

For one ring product:

```text
A = A_0 + A_1
B = B_0 + B_1

A * B =
    A_0 * B_0
  + A_0 * B_1
  + A_1 * B_0
  + A_1 * B_1
```

The bridge uses two OLE instances:

```text
OLE 1 gives shares of A_0 * B_1
OLE 2 gives shares of A_1 * B_0
```

Then each party adds its local product:

```text
party 0 adds A_0 * B_0
party 1 adds A_1 * B_1
```

The result is a shared Beaver product over ring-polynomial matrix entries.

What is still missing for Orca:

- packing Orca tensor scalar values into polynomial slots,
- converting shares from Z_p to Orca's Z_{2^bw} arithmetic,
- writing triples in the exact `(A, B, C)` shape consumed by `gpuMatmulBeaver`.

## 6. What We Tested

### Orca Profiling

Source:

- `GPU-MPC/orca_runner/logs/master.log`

Measured:

- key files per party,
- average key-read time,
- average compute time,
- communication per iteration.

Rows:

| Model | Key files observed | Avg key read (ms) | Avg compute (ms) | Comm per iteration (B) |
| --- | --- | ---: | ---: | ---: |
| P-SecureML | P0 338M, P1 338M | 9.909 | 32.273 | 5,692,170.18 |
| P-LeNet | P0 4.0G, P1 4.0G | 109.727 | 107.727 | 65,572,810.18 |
| P-AlexNet | P0 3.8G, P1 3.8G | 104.818 | 121.727 | 113,913,098.18 |

Interpretation:

- P-SecureML is smaller: read time is below compute time.
- P-LeNet shows the strongest “key read matches compute” point.
- P-AlexNet still has key read close to compute.

### DPF Online Key Generation

Benchmark:

- `GPU-MPC/tests/fss/dpf_online_keygen_bench.cu`
- `GPU-MPC/scripts/run_dpf_online_keygen_sweep.py`

Configuration:

- `bin=16`,
- N from 8192 to 1048576,
- chunk sizes 8192, 4096, 2048,
- validation passes for saved rows.

Main result:

| N | chunk | Full pair key (MiB) | Partial peak pair key (MiB) | Peak reduction | Time overhead |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1048576 | 8192 | 360.00 | 2.81 | 128.00x | 1.834x |
| 1048576 | 4096 | 360.00 | 1.41 | 255.99x | 2.942x |
| 1048576 | 2048 | 360.00 | 0.70 | 511.97x | 4.975x |

### GPU NTT / PolyMul

Use only the promoted GPU backend as the GPU implementation in the poster story.

Direct n=8192 result:

| Impl | q actual | batch | NTT mean (us) | INTT mean (us) | Full PolyMul mean (us) | Per-poly PolyMul (us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU NFLLib | 30 | 1 | 57.2021 | 61.8469 | 180.594 | 180.594 |
| GPU CUDA | 30 | 64 | 41.7984 | 45.8986 | 129.509 | 2.024 |

Speedups:

- forward NTT: 87.59x,
- full PolyMul: 89.24x.

Sweep support:

- q req=32, actual q=30, n=8192 to 1048576, validation pass.
- q req=64, actual q=62, n=8192 to 1048576, validation pass.

### Ring-LPN VOLE

Benchmark:

- `GPU-MPC/ringlpn/src/bench_vole_ringlpn.cu`

Baseline:

- m=32,
- c=2,
- noise weight 64,
- requested q=32 and q=64,
- n=8192 to 1048576.

Selected rows:

| q req | q actual | n | Full expand mean (us) | Per-output expand (us) | Outputs/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 30 | 8192 | 191.485 | 5.984 | 167114.92 |
| 32 | 30 | 1048576 | 32144.700 | 1004.522 | 995.50 |
| 64 | 62 | 8192 | 549.802 | 17.181 | 58202.77 |
| 64 | 62 | 1048576 | 50952.700 | 1592.272 | 628.03 |

### Figure 2 OLE

Configuration:

- requested q=64,
- actual q=62,
- c=2,
- t=64,
- n in 8192 and 16384,
- uniform and regular sparse noise.

Selected rows:

| Noise | n | SPFSS domain | validation | host validation | Key bytes MiB | Keygen us | OLE expand mean us |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| uniform | 8192 | 16384 | pass | pass | 8.63 | 4797.000 | 865253.000 |
| regular | 8192 | 256 | pass | pass | 5.27 | 40828.000 | 58462.500 |
| regular | 16384 | 512 | pass | skipped | 5.84 | 42331.000 | 67733.000 |

Interpretation:

- Uniform has a larger SPFSS domain.
- Regular sparse noise uses grouped smaller domains and is much faster in expand for these bounded rows.
- Keygen can still be higher for regular noise.
- This is correctness-first, not final optimized scheduling.

### Linear OLE-To-Beaver

Configuration:

- rows=2,
- inner=2,
- cols=2,
- n=8192,
- c=2,
- t=8,
- 16 OLE instances.

Rows:

| Noise | SPFSS domain | validation | OLE instances | Key bytes MiB | Keygen us | Linear expand mean us |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| uniform | n/a | pass | 16 | 2.16 | 6594.000 | 222355.000 |
| regular | 2048 | pass | 16 | 1.78 | 82726.000 | 115447.000 |

## 7. What Not To Claim

Do not claim:

- finished end-to-end Orca integration,
- trusted-dealer removal in Orca,
- q=128 / CRT support,
- paper-parameter Figure 2 results,
- CPU-vs-GPU speedup for VOLE itself,
- that chunking reduces total logical key bytes,
- that OLE-to-Beaver already emits Orca-compatible scalar Beaver triples,
- that the local profile proves tens of gigabytes of key material.

Say instead:

- “The local profile reaches several GiB per party; tens of GiB is a scaling motivation, not the maximum observed in this saved local table.”
- “These are validated standalone building blocks.”
- “The current work is toward Orca integration.”
- “Chunking reduces peak staged footprint.”
- “q=32/q=64 are requested widths implemented as actual q=30/q=62 single-prime paths.”

## 8. Likely Questions And Good Answers

### Q1. What is the core problem?

The online phase is fast partly because expensive work is moved offline. But then the system has to store, read, and move large key/correlation material. In Orca, key-read time can already match GPU compute time for moderate local training runs.

### Q2. Is this a compute bottleneck or memory bottleneck?

Both matter, but the profiling motivation is memory and movement. The GPU is fast enough that key I/O and staging become first-order costs.

### Q3. Why is direct I/O mentioned?

Because Orca already uses direct I/O infrastructure: `O_DIRECT | O_LARGEFILE`, aligned buffers, and read/compute overlap. That makes the key-movement bottleneck more structural. It is not just “turn on obvious disk optimization.”

### Q4. Does the DPF chunking reduce total key bytes?

No. It reduces peak staged footprint. The total logical material remains essentially the same. This is why the total bytes multiplier is 1.000x in the saved chunk=8192 sweep.

### Q5. What exactly is the 128x result?

At N=1048576 and chunk=8192, one-shot full-pair generation stages 360.00 MiB. Chunked generation stages only 2.81 MiB at peak. The ratio is 128.00x, with 1.834x time overhead.

### Q6. Why not always use chunk=2048 if it gives 511.97x?

Because the time overhead rises to 4.975x. Chunking is a memory-time knob, not a free win. Chunk=8192 is the clean “under 2x overhead” headline.

### Q7. Is chunked DPF integrated into Orca?

No. It is a standalone key-generation systems benchmark. The next step is wiring chunked generation into real online FSS evaluation and measuring end-to-end memory and latency.

### Q8. What is FSS?

Function Secret Sharing splits a function into two keys. Each party evaluates one key, and the sum of outputs equals the function value. It is useful for non-linear operations in secure computation.

### Q9. What is a DPF?

A Distributed Point Function is FSS for a point function. It outputs beta at one hidden index alpha and zero elsewhere, split across two parties.

### Q10. Why does DPF eval-all create memory pressure?

If the domain has N entries and both parties' expanded material is staged, memory scales with N. Chunking changes the staged working set from O(N) to O(chunk_size).

### Q11. What is Ring-LPN doing in an FSS poster?

The system context is GPU-FSS/Orca, but PPML preprocessing has two sides: FSS handles non-linear work, while additive sharing handles linear work. Ring-LPN PCG building blocks target the linear/correlation side of the same preprocessing bottleneck.

### Q12. What is a PCG?

A Pseudorandom Correlation Generator expands compact seeds into large correlated randomness, ideally reducing storage and movement relative to precomputing all correlations explicitly.

### Q13. What is NTT?

The Number Theoretic Transform is the finite-field analogue of an FFT. It accelerates polynomial multiplication by turning convolution into pointwise multiplication.

### Q14. Why is PolyMul the key primitive?

Ring-LPN PCG and VOLE-style expansion operate over polynomial rings. Fast polynomial multiplication is the core operation needed for practical GPU expansion.

### Q15. What does the 89.24x speedup compare?

It compares per-polynomial full PolyMul at n=8192: CPU NFLLib per-poly 180.594 us versus GPU per-poly 2.024 us in the direct saved artifact.

### Q16. Are the broader speedup bars the same artifact?

No. The direct 87.59x/89.24x n=8192 comparison and the sweep-derived speedups are separate benchmark campaigns. They are complementary and should not be merged into one range.

### Q17. Why requested q=32 but actual q=30?

The backend uses prime moduli that support the NTT. The requested width describes the target class; the actual implementation uses a supported single prime: 30 bits for requested q=32.

### Q18. Why requested q=64 but actual q=62?

Same reason: the single-prime q=64-class path uses an actual 62-bit prime.

### Q19. Is q=128 implemented?

No. q=128 needs dual-prime CRT scheduling and recomposition. It is explicitly future work.

### Q20. Why not show a three-way GPU implementation comparison?

Because the poster's clean story is CPU NFLLib baseline versus the promoted GPU backend. The older GPU implementation is not part of the current poster evidence path.

### Q21. What is VOLE?

Vector Oblivious Linear Evaluation is a correlation where outputs satisfy z = y + x * Delta. It is useful as a building block for generating correlated randomness.

### Q22. Does VOLE have a CPU speedup number?

No. The saved artifacts do not include a CPU VOLE baseline. Only claim validation and GPU throughput/latency for the standalone GPU prototype.

### Q23. What does synthetic MPVOLE mean?

The benchmark locally synthesizes inputs that are consistent with the expected MPVOLE relation. This lets us test the algebraic expansion layer without claiming a full upstream protocol integration.

### Q24. What does the VOLE validator check?

It checks coefficient-wise that z = y + x * Delta over the tested polynomial ring.

### Q25. What is Figure 2 OLE?

It is the standalone GPU artifact validating an OLE relation in the ring: z_0 + z_1 == x_0 * x_1 in Z_p[X]/(X^N+1).

### Q26. What is SPFSS?

Sparse Point Function Secret Sharing. In this context it is used to assemble sparse-noise related OLE components over the polynomial domain.

### Q27. Why is regular sparse noise faster in OLE expand?

The regular mode groups point functions into smaller SPFSS domains of size 2N/t. For n=8192 and t=64, that domain is 256 rather than 16384, which makes expansion much smaller.

### Q28. Why is regular keygen slower even though expand is faster?

The saved artifact is correctness-first and the regular path has more structured grouping/setup work. The important poster claim is validation and the domain-size effect, not final optimized scheduling.

### Q29. What is the OLE-to-Beaver bridge?

It uses two OLE instances per ring product to generate the cross terms A_0*B_1 and A_1*B_0. Local products provide A_0*B_0 and A_1*B_1. Together they form Beaver shares for ring-polynomial matrix multiplication.

### Q30. Is that already Orca FC integration?

No. It is ring-polynomial matrix multiplication, not Orca scalar FC. Missing pieces are scalar packing, Z_p to Z_{2^bw} share conversion, and an Orca-compatible triple writer.

### Q31. What does `Z_p -> Z_{2^bw}` mean?

The Ring-LPN artifacts operate over a prime field/ring with modulus p. Orca linear layers use rings tied to bitwidths, often Z_{2^bw}. Shares must be converted without breaking correctness/security before Orca can consume the triples.

### Q32. What is scalar packing?

It is the mapping from many Orca tensor scalar values into polynomial coefficients or slots. Without a packing model, a ring-polynomial Beaver product does not directly correspond to Orca's matrix multiplication layout.

### Q33. What is the integration plan?

1. Add q=128/CRT support for paper-comparable Ring-LPN parameters.
2. Specify scalar packing from Orca tensors into polynomial slots.
3. Implement Z_p to Z_{2^bw} share conversion.
4. Write Orca-compatible `(A, B, C)` triples.
5. Validate a tiny FC layer against baseline Beaver triples.
6. Measure P-LeNet/P-AlexNet end-to-end.

### Q34. What if someone asks where the tens-of-gigabytes claim is?

Say: “The saved local table included here reaches about 4.0G per party. The tens-of-gigabytes phrasing is scaling motivation for larger models/configurations, not the maximum in this local profile. For this poster, I use the measured several-GiB claim.”

### Q35. Are the benchmark results reproducible?

The artifact records container commands, source result files, and current host/container setup fields. For publication-quality reproducibility, the best next step is to re-run or archive an explicit environment capture with each benchmark output, because the older saved result files did not embed a full per-run environment snapshot.

### Q36. How were hardware and environment fields filled?

They were queried from the current host and running `orca-dev` container on 2026-05-11. The observed machine has 4x NVIDIA RTX 5000 Ada Generation GPUs, an Intel Xeon w5-3435X CPU, 109 GiB RAM, NVIDIA driver 560.35.03, driver-reported CUDA 12.6, and container CUDA toolkit 12.3.107 with gcc/g++ 9.5.0. The caveat is that the original saved benchmark result files did not embed a complete environment snapshot per run, so for final publication-quality reproducibility we should re-run or archive an explicit environment capture alongside the benchmark outputs.

### Q37. What is the security claim?

The poster is mostly systems/prototype evidence. It relies on the standard intended roles of FSS/DPF, additive sharing, Ring-LPN PCG/VOLE-style correlations, and OLE-to-Beaver conversion. It does not prove a new protocol end-to-end or claim trusted-dealer removal in Orca.

### Q38. What is actually validated?

- DPF: serialized key layout and parsed metadata for full and chunked modes.
- NTT/PolyMul: roundtrip/multiplication correctness and validation pass in saved sweeps.
- VOLE: coefficient-wise z = y + x * Delta.
- OLE: z_0 + z_1 == x_0 * x_1.
- Linear bridge: C_0 + C_1 equals clear ring-polynomial matrix product.

### Q39. Why does the poster include both DPF and Ring-LPN?

Because the offline preprocessing bottleneck appears in both PPML paradigms. DPF addresses FSS non-linear key material. Ring-LPN PCG components address additive-sharing correlation generation for linear layers.

### Q40. What is the most honest conclusion?

“We have validated standalone building blocks that support a path toward memory-efficient GPU-FSS preprocessing. The strongest finished claims are the Orca profiling evidence, DPF peak-footprint reduction, GPU NTT/PolyMul speedup, and VOLE/OLE correctness. Full Orca integration is the next stage.”

## 9. Follow-Up Work

Immediate engineering follow-ups:

- collect exact hardware/software environment details,
- implement q=128 / CRT in the promoted GPU polynomial backend,
- add scalar packing for Orca tensors,
- specify and implement Z_p to Z_{2^bw} share conversion,
- write Orca-compatible triples for `gpuMatmulBeaver`,
- run FC-only integration validation,
- measure end-to-end Orca memory and runtime effects,
- evaluate whether chunked DPF generation can overlap with online evaluation,
- add application-level peak memory instrumentation.

Paper/poster follow-ups:

- cite Orca, FSS/DPF, Ring-LPN/PCG, VOLE/OLE, NFLLib, and GPU NTT background sources,
- record exact GPU/CPU/driver/container details,
- include QR code to code/artifact bundle,
- keep “standalone prototype” language visible,
- separate direct CPU/GPU comparison from sweep-derived speedups.

## 10. Final Guardrails For The Presentation

Say:

- “validated standalone building blocks,”
- “toward Orca integration,”
- “peak staged footprint,”
- “requested q=32/q=64 map to actual q=30/q=62,”
- “no q=128/CRT yet,”
- “no CPU VOLE speedup claim,”
- “not yet Orca FC integration.”

Avoid:

- “we replaced Orca preprocessing,”
- “trusted dealer removed,”
- “total keys shrink by 128x,”
- “paper parameters are complete,”
- “VOLE is 100x faster than CPU,”
- “linear OLE-to-Beaver already plugs into Orca.”

Best closing line:

**“The contribution is not that the whole Orca pipeline is solved today; it is that the bottleneck is measured, the memory-footprint knob is quantified, the GPU polynomial core is fast and validated, and the next PCG/OLE-to-Beaver bridge pieces are now concrete enough to integrate.”**
