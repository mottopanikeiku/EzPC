# GPU FSS Memory Efficiency Abstract Outline

Generated: 2026-04-08

## Recommended Title Direction

Working title from the professor's guidance:

- `Improving Memory Efficiency of GPU-Accelerated Function Secret Sharing`

This title fits the current evidence as long as the abstract clearly distinguishes implemented components from proposed future work.

## Extracted Recommendation

The professor's suggested structure is:

1. introduction of the GPU FSS library,
2. profiling,
3. potential technique: online key generation based on DPF,
4. accelerating the online phase of FSS with Ring-LPN,
5. partial-key generation pipeline to reduce memory and storage footprint,
6. possibly direct I/O for optimization.

## How To Map That To The Current Evidence

### 1. Introduction Of GPU FSS Library

Safe message:

- `GPU-MPC` is the current GPU-accelerated secure computation environment.
- It already contains GPU-backed FSS-related components, Orca training and inference, profiling utilities, and the standalone Ring-LPN benchmarking track.

### 2. Profiling

Safe message:

- profiling already shows that large offline key material creates substantial online I/O and movement pressure,
- for larger models, key-read time is already close to compute time,
- the system already overlaps key reading with computation, which means the next bottlenecks are deeper than naive disk throughput.

Concrete supporting numbers from `orca_runner/logs/master.log`:

- `P-SecureML`: key read `9.91 ms`, compute `32.27 ms`,
- `P-LeNet`: key read `109.73 ms`, compute `107.73 ms`,
- `P-AlexNet`: key read `104.82 ms`, compute `121.73 ms`.

Supporting implementation details:

- `utils/gpu_file_utils.cpp` uses `O_DIRECT | O_LARGEFILE`,
- key buffers are 4096-byte aligned,
- `experiments/orca/orca_evaluator.cu` overlaps key reading and computation,
- `nn/orca/fc_layer.cu` still performs repeated runtime `moveToGPU()` calls.

### 3. Online Key Generation Based On DPF

Safe message:

- this is currently a proposal motivated by the profiling evidence,
- the intended idea is to generate partial keys on demand for partial computation rather than materializing all keys offline,
- the goal is reducing storage footprint and online staging pressure.

Unsafe message to avoid:

- claiming that this DPF partial-key pipeline is already implemented or benchmarked.

### 4. Accelerating The Online Phase Of FSS With Ring-LPN

Safe message:

- this is the strongest current implementation result,
- a standalone GPU Ring-LPN VOLE prototype has been implemented and validated,
- it reuses the promoted cheddar-derived GPU polynomial backend,
- it validates the coefficient-wise VOLE relation across the full tested range `n=8192..1048576` for both requested `q=32` and requested `q=64`.

Concrete results:

- q=32 full expansion latency ranges from `269.484 us` to `43.392 ms`,
- q=64 full expansion latency ranges from `772.324 us` to `67.532 ms`,
- all sweep points passed validation.

Underlying acceleration context already supported by prior benchmark artifacts:

- requested `q=32` GPU PolyMul speedups over CPU at overlap points are roughly `146x` to `171x`,
- requested `q=64` GPU PolyMul speedups over CPU range roughly `48x` to `220x`.

### 5. Partial-Key Pipeline For Memory Reduction

Safe message:

- this is a natural systems direction suggested by the profiling data,
- it fits the DPF-based online key-generation story,
- it should currently be described as planned design work rather than completed engineering.

### 6. Direct I/O Optimization

Safe message:

- direct I/O is already partially present in the current system,
- this can be presented as supporting infrastructure or a systems lever already explored,
- it should not be framed as the main new contribution unless new measurements are added.

## Recommended Contribution Framing

The abstract should separate contributions into two groups.

### Implemented And Measured

1. Profiling of the current GPU FSS and Orca pipeline that identifies key I/O and runtime movement as memory-efficiency bottlenecks.
2. Direct I/O and overlap infrastructure already present in the system.
3. A validated GPU Ring-LPN VOLE expansion prototype with saved q=32 and q=64 sweep artifacts.

### Proposed Next Techniques

1. Online key generation based on DPF.
2. Partial-key pipeline generation for partial computation.
3. Full integration of Ring-LPN acceleration into the online FSS path.

## Safe One-Paragraph Abstract Logic

The most defensible logic chain is:

- introduce GPU-accelerated FSS as the systems setting,
- show that profiling reveals memory footprint and key movement as major bottlenecks,
- note that existing direct I/O and overlap reduce but do not remove the problem,
- propose on-demand partial key generation as the memory-footprint reduction direction,
- present the new Ring-LPN GPU VOLE expansion prototype as the concrete online-phase acceleration result,
- close by positioning full DPF-backed and SPFSS-backed integration as ongoing work.

## Claims To Avoid

1. Claiming end-to-end memory-footprint reduction numbers unless new experiments are run.
2. Claiming that the DPF pipeline or partial-key generation path is already implemented.
3. Claiming that the current Ring-LPN prototype is a full end-to-end SPFSS-backed degree-1 correlation system.
4. Claiming CPU-vs-GPU speedup numbers for the VOLE prototype itself without a CPU VOLE baseline.

## Best Next Step

The best next step is to draft the 250-word abstract directly from this outline and `ringlpn_vole_abstract_support.md`, using the professor's storyline but keeping the DPF and partial-key pipeline pieces explicitly future-facing.