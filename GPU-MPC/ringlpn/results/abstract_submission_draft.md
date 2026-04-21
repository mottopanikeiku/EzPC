# GPU FSS Abstract Draft

Generated: 2026-04-10

## Recommended Title

Primary title:

- Improving Memory Efficiency of GPU-Accelerated Function Secret Sharing

Alternative titles:

- Reducing Peak Key Footprint in GPU-Accelerated Function Secret Sharing with Online DPF Key Generation and Ring-LPN Acceleration
- Toward Memory-Efficient GPU Function Secret Sharing via Chunked DPF Key Generation and Ring-LPN Online-Phase Acceleration
- Tunable Online DPF Key Generation and Ring-LPN Acceleration for Memory-Efficient GPU Function Secret Sharing

## Submission-Ready Abstract

Function secret sharing (FSS) is promising for privacy-preserving machine learning because much of its online work is parallel and maps well to GPUs. Orca is a GPU-accelerated FSS library for machine-learning workloads, but our profiling shows that large precomputed keys stress storage, I/O, and memory alongside arithmetic. P-LeNet and P-AlexNet use 4.0G and 3.8G key files, and key-read time is close to compute time at 109.73 ms versus 107.73 ms and 104.82 ms versus 121.73 ms. This motivates reducing peak staged key footprint and runtime data movement while preserving throughput.

Ring-LPN-based pseudorandom correlation generation (PCG) offers a path to that goal by replacing large staged correlations with compact seeds and fast online algebraic expansion. We study an Orca-oriented path with three pieces: chunked DPF online key generation to cap staged key memory, GPU Ring-LPN VOLE expansion to regenerate correlations on demand, and a promoted GPU NTT/PolyMul backend. In standalone prototypes, chunked DPF generation at bin 16 caps peak pair-key footprint at 2.81 MiB while one-shot generation grows to 360.00 MiB, giving 128x peak-footprint reduction at n = 1048576 with 1.885x overhead. Our GPU Ring-LPN VOLE prototype validates the target relation across n = 8192 to 1048576 for q = 32 and q = 64, with q = 32 latency from 269.484 us to 43.392 ms and q = 64 from 772.324 us to 67.532 ms. The underlying GPU core reaches 87.59x forward-NTT and 89.24x full-PolyMul speedup over NFLLib at n = 8192, with 33.07 GB/s estimated coefficient throughput. Full Orca/SPFSS-backed integration remains ongoing work.

## Conservative Fallback Abstract

Function secret sharing (FSS) is attractive for privacy-preserving machine learning because much of its online work is parallel and can benefit from GPU execution. Orca is a GPU-accelerated FSS library for machine-learning workloads, but in our current environment large precomputed keys create storage, I/O, and memory pressure during online execution. Profiling shows that P-LeNet and P-AlexNet already use 4.0G and 3.8G key files, and average key-read time is close to compute time at 109.73 ms versus 107.73 ms and 104.82 ms versus 121.73 ms. This motivates reducing the peak staged key footprint rather than optimizing arithmetic alone.

Ring-LPN-based pseudorandom correlation generation is a promising way to address that bottleneck by expanding compact online correlations instead of materializing large precomputed key sets. We therefore study an Orca-relevant design built from chunked DPF online key generation, GPU Ring-LPN VOLE expansion, and a promoted GPU polynomial backend. For eval-all keys at bin 16 and chunk size 8192, chunked DPF generation holds peak pair-key footprint at 2.81 MiB while one-shot generation grows to 360.00 MiB, yielding up to 128x peak-footprint reduction at n = 1048576 with 1.885x time overhead. The GPU Ring-LPN VOLE prototype validates the target relation across n = 8192 to 1048576 for requested q = 32 and q = 64, with full expansion latency from 269.484 us to 43.392 ms and from 772.324 us to 67.532 ms, respectively. The underlying GPU NTT/PolyMul core also has a direct saved n = 8192 comparison showing 87.59x forward-NTT and 89.24x full-PolyMul speedup over NFLLib. Together, these results support a memory-efficient Orca integration path, although full end-to-end Orca/SPFSS-backed integration remains ongoing work.

## One-Line Scope Guard

Use this sentence if the submission form or advisor asks what is already complete:

- The current submission is supported by standalone measured prototypes for chunked DPF online key generation and GPU Ring-LPN VOLE expansion, while full end-to-end Orca/SPFSS-backed integration remains ongoing work.

## Benchmark Highlights For Submission

- Profiling motivation: key read is already close to compute for larger models, including `109.73 ms` versus `107.73 ms` for P-LeNet and `104.82 ms` versus `121.73 ms` for P-AlexNet.
- DPF balanced point: chunk size `8192` holds peak pair-key footprint to `2.81 MiB` and reaches `128x` peak-footprint reduction at `n = 1048576` with `1.885x` time overhead.
- DPF aggressive point: chunk size `4096` lowers peak pair-key footprint to `1.41 MiB` and reaches about `256x` peak-footprint reduction at `n = 1048576` with `2.888x` time overhead.
- DPF ultra-aggressive point: chunk size `2048` lowers peak pair-key footprint to `0.70 MiB` and reaches about `512x` peak-footprint reduction at `n = 1048576` with `4.959x` time overhead.
- NTT/PolyMul core: at the direct `n = 8192` CPU overlap point, the promoted backend reports `87.59x` forward-NTT and `89.24x` full-PolyMul per-polynomial speedup over NFLLib, and the validated q=`64` sweep reports a `33.07 GB/s` estimated coefficient-throughput proxy at the same size.
- VOLE q=32: full expansion latency ranges from `269.484 us` to `43.392 ms`, with `118745` outputs/s at `n = 8192`.
- VOLE q=64: full expansion latency ranges from `772.324 us` to `67.532 ms`, with `41433` outputs/s at `n = 8192`.
- VOLE batching sensitivity: doubling q=`32` output count from `m=32` to `m=64` improves per-output latency at `n = 8192` from `8.421 us` to `7.137 us`, while large-degree per-output cost stays near `1.36 ms` at `n = 1048576`.
- Underlying promoted polynomial backend: q=32 promoted path reaches about `803741` polys/s at `n = 8192`, and q=64 reaches about `252312` polys/s at `n = 8192`.

## Raw Data And Provenance

- Detailed raw tables, exact experiment commands, and code-path provenance are collected in `results/abstract_benchmark_appendix.md`.

## Recommendation

- Use the main abstract if you want the strongest version that is still benchmark-backed.
- If you want the safest possible submission, use the conservative fallback and mention only the `8192` chunk-size DPF result.
- If you want one extra strong sentence without overclaiming, mention that smaller chunking pushes peak-footprint reduction to about `256x` at higher key-generation overhead.

## Claims To Keep

- profiling shows memory movement and key I/O are real bottlenecks in the current GPU FSS setting
- chunked DPF key generation reduces peak staged key footprint substantially
- the Ring-LPN VOLE expansion prototype is implemented, benchmarked, and validated as a standalone GPU component
- full integration is ongoing work

## Claims To Avoid

- claiming that the full OLE-R-LPN or SPFSS-backed pipeline is complete end to end
- claiming end-to-end application memory reduction numbers
- claiming q = 128 support for the current VOLE prototype
- claiming CPU-vs-GPU speedup numbers for the VOLE prototype itself