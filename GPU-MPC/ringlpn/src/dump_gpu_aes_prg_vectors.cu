// Dumps the GPU DPF expansion PRG's exact input/output vectors.
//
// Purpose: the two-party host keygen must produce keys the UNCHANGED GPU
// evaluator (`ringlpn_spfss_zp::gpuDpfZpFullEvalSum`) accepts, which means its
// host-side expansion has to be bit-identical to the device
// `aes_prg_expand`. Rather than reason about the AES byte order in
// GPU-MPC/fss/gpu_aes_shm.cu, this program emits ground-truth vectors that the
// host implementation is tested against (`test_gpu_aes_prg_parity`).
//
// Output: CSV rows `seed_hi,seed_lo,sl_hi,sl_lo,tl,sr_hi,sr_lo,tr`, all hex.

#define RINGLPN_SPFSS_ZP_NO_MAIN 1
#include "gpu_spfss_zp.cuh"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

__global__ void prg_vector_kernel(const AESBlock *seeds, int count,
                                  AESBlock *out_l, uint8_t *out_tl,
                                  AESBlock *out_r, uint8_t *out_tr,
                                  AESGlobalContext gaes) {
    AESSharedContext saes;
    loadSbox(&gaes, &saes);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    AESBlock sl, sr;
    uint8_t tl, tr;
    ringlpn_spfss_zp::aes_prg_expand(seeds[idx], &saes, sl, tl, sr, tr);
    out_l[idx] = sl;
    out_tl[idx] = tl;
    out_r[idx] = sr;
    out_tr[idx] = tr;
}

}  // namespace

int main(int argc, char **argv) {
    int count = 8;
    if (argc > 1) count = std::atoi(argv[1]);
    if (count < 1) count = 1;

    std::vector<AESBlock> seeds((size_t)count);
    // Deterministic, structurally varied full-width seeds, including both
    // values of the low bit.
    for (int i = 0; i < count; ++i) {
        uint64_t lo = 0x0123456789ABCDEFULL * (uint64_t)(i + 1) + (uint64_t)i;
        uint64_t hi = 0xFEDCBA9876543210ULL ^ ((uint64_t)i << 13);
        if (i == 0) { lo = 0; hi = 0; }
        if (i == 1) { lo = 1; hi = 0; }
        if (i == 2) { lo = ~0ULL; hi = ~0ULL; }
        seeds[(size_t)i] = ringlpn_spfss_zp::make_block(lo, hi);
    }

    AESGlobalContext gaes;
    initAESContext(&gaes);

    AESBlock *d_seeds = nullptr, *d_l = nullptr, *d_r = nullptr;
    uint8_t *d_tl = nullptr, *d_tr = nullptr;
    const size_t block_bytes = (size_t)count * sizeof(AESBlock);
    ringlpn_spfss_zp::cuda_check(cudaMalloc(&d_seeds, block_bytes), "malloc seeds");
    ringlpn_spfss_zp::cuda_check(cudaMalloc(&d_l, block_bytes), "malloc l");
    ringlpn_spfss_zp::cuda_check(cudaMalloc(&d_r, block_bytes), "malloc r");
    ringlpn_spfss_zp::cuda_check(cudaMalloc(&d_tl, (size_t)count), "malloc tl");
    ringlpn_spfss_zp::cuda_check(cudaMalloc(&d_tr, (size_t)count), "malloc tr");
    ringlpn_spfss_zp::cuda_check(
        cudaMemcpy(d_seeds, seeds.data(), block_bytes, cudaMemcpyHostToDevice),
        "copy seeds");

    const int threads = 64;
    const int blocks = (count + threads - 1) / threads;
    prg_vector_kernel<<<blocks, threads>>>(d_seeds, count, d_l, d_tl, d_r, d_tr,
                                           gaes);
    ringlpn_spfss_zp::cuda_check(cudaDeviceSynchronize(), "prg kernel");

    std::vector<AESBlock> hl((size_t)count), hr((size_t)count);
    std::vector<uint8_t> htl((size_t)count), htr((size_t)count);
    ringlpn_spfss_zp::cuda_check(
        cudaMemcpy(hl.data(), d_l, block_bytes, cudaMemcpyDeviceToHost), "copy l");
    ringlpn_spfss_zp::cuda_check(
        cudaMemcpy(hr.data(), d_r, block_bytes, cudaMemcpyDeviceToHost), "copy r");
    ringlpn_spfss_zp::cuda_check(
        cudaMemcpy(htl.data(), d_tl, (size_t)count, cudaMemcpyDeviceToHost),
        "copy tl");
    ringlpn_spfss_zp::cuda_check(
        cudaMemcpy(htr.data(), d_tr, (size_t)count, cudaMemcpyDeviceToHost),
        "copy tr");

    std::printf("seed_hi,seed_lo,sl_hi,sl_lo,tl,sr_hi,sr_lo,tr\n");
    for (int i = 0; i < count; ++i) {
        std::printf("%016llx,%016llx,%016llx,%016llx,%u,%016llx,%016llx,%u\n",
                    (unsigned long long)ringlpn_spfss_zp::block_hi(seeds[(size_t)i]),
                    (unsigned long long)ringlpn_spfss_zp::block_lo(seeds[(size_t)i]),
                    (unsigned long long)ringlpn_spfss_zp::block_hi(hl[(size_t)i]),
                    (unsigned long long)ringlpn_spfss_zp::block_lo(hl[(size_t)i]),
                    (unsigned)htl[(size_t)i],
                    (unsigned long long)ringlpn_spfss_zp::block_hi(hr[(size_t)i]),
                    (unsigned long long)ringlpn_spfss_zp::block_lo(hr[(size_t)i]),
                    (unsigned)htr[(size_t)i]);
    }

    cudaFree(d_seeds);
    cudaFree(d_l);
    cudaFree(d_r);
    cudaFree(d_tl);
    cudaFree(d_tr);
    return 0;
}
