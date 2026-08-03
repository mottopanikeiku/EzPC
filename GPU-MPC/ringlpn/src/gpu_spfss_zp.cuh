#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "fss/gpu_aes_shm.h"

namespace ringlpn_spfss_zp {

using Word = uint64_t;
using U128 = unsigned __int128;

struct GPUDPFZpKey {
    int party = 0;
    int log_domain = 0;
    int count = 0;
    Word modulus = 0;
    std::vector<AESBlock> seeds;
    std::vector<AESBlock> s_cw;
    std::vector<uint8_t> t_l_cw;
    std::vector<uint8_t> t_r_cw;
    std::vector<Word> final_cw;
};

struct GPUSPFSSZpKey {
    int party = 0;
    int log_domain = 0;
    Word modulus = 0;
    GPUDPFZpKey dpf;
};

struct DeviceGPUDPFZpKey {
    int party;
    int log_domain;
    int count;
    Word modulus;
    const AESBlock *seeds;
    const AESBlock *s_cw;
    const uint8_t *t_l_cw;
    const uint8_t *t_r_cw;
    const Word *final_cw;
};

inline void cuda_check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

__host__ __device__ inline Word mod_add(Word a, Word b, Word modulus) {
    Word s = a + b;
    return (s >= modulus || s < a) ? (s - modulus) : s;
}

__host__ __device__ inline Word mod_sub(Word a, Word b, Word modulus) {
    return a >= b ? a - b : modulus - (b - a);
}

__host__ __device__ inline uint64_t splitmix64_stateless(uint64_t x) {
    uint64_t z = x + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__host__ __device__ inline AESBlock make_block(uint64_t lo, uint64_t hi) {
    return static_cast<AESBlock>(lo) | (static_cast<AESBlock>(hi) << 64);
}

__host__ __device__ inline uint64_t block_lo(AESBlock x) {
    return static_cast<uint64_t>(x);
}

__host__ __device__ inline uint64_t block_hi(AESBlock x) {
    return static_cast<uint64_t>(x >> 64);
}

__host__ __device__ inline Word convert_zp(AESBlock s, Word modulus) {
    Word lo = static_cast<Word>(block_lo(s) % modulus);
    Word hi = static_cast<Word>(block_hi(s) % modulus);
    return mod_add(lo, hi, modulus);
}

__device__ inline void aes_prg_expand(AESBlock seed,
                                      AESSharedContext *aes,
                                      AESBlock &s_l,
                                      uint8_t &t_l,
                                      AESBlock &s_r,
                                      uint8_t &t_r) {
    AESBlock left_seed = 0;
    AESBlock left_tag = 0;
    AESBlock right_seed = 0;
    AESBlock right_tag = 0;
    // Four domain-separated AES calls: plaintexts 0/2 produce full 128-bit
    // child seeds; plaintexts 1/3 produce independent control bits.
    applyAESPRGFourTimes(aes, reinterpret_cast<u32 *>(&seed),
                         reinterpret_cast<u32 *>(&left_seed),
                         reinterpret_cast<u32 *>(&left_tag),
                         reinterpret_cast<u32 *>(&right_seed),
                         reinterpret_cast<u32 *>(&right_tag));
    s_l = left_seed;
    s_r = right_seed;
    t_l = static_cast<uint8_t>(left_tag & 1);
    t_r = static_cast<uint8_t>(right_tag & 1);
}

__global__ void keygen_dpf_zp_kernel(int log_domain,
                                     int count,
                                     Word modulus,
                                     const Word *alphas,
                                     const Word *betas,
                                     uint64_t seed_base,
                                     AESBlock *seeds0,
                                     AESBlock *seeds1,
                                     AESBlock *s_cw,
                                     uint8_t *t_l_cw,
                                     uint8_t *t_r_cw,
                                     Word *final_cw,
                                     AESGlobalContext gaes) {
    AESSharedContext saes;
    loadSbox(&gaes, &saes);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    AESBlock s0 = make_block(splitmix64_stateless(seed_base ^ (0xD1B54A32D192ED03ULL + 4ULL * idx)),
                             splitmix64_stateless(seed_base ^ (0x94D049BB133111EBULL + 4ULL * idx)));
    AESBlock s1 = make_block(splitmix64_stateless(seed_base ^ (0x8538ECB5BD456EA3ULL + 4ULL * idx)),
                             splitmix64_stateless(seed_base ^ (0xC6BC279692B5C323ULL + 4ULL * idx)));

    seeds0[idx] = s0;
    seeds1[idx] = s1;

    uint8_t tau0 = 0;
    uint8_t tau1 = 1;
    Word alpha = alphas[idx];
    Word beta = betas[idx];

    for (int level = 0; level < log_domain; ++level) {
        AESBlock s0_l, s0_r, s1_l, s1_r;
        uint8_t t0_l, t0_r, t1_l, t1_r;
        aes_prg_expand(s0, &saes, s0_l, t0_l, s0_r, t0_r);
        aes_prg_expand(s1, &saes, s1_l, t1_l, s1_r, t1_r);

        int bit_idx = log_domain - 1 - level;
        uint8_t alpha_bit = static_cast<uint8_t>((alpha >> bit_idx) & 1ULL);

        AESBlock lose0 = alpha_bit == 0 ? s0_r : s0_l;
        AESBlock lose1 = alpha_bit == 0 ? s1_r : s1_l;
        AESBlock keep0 = alpha_bit == 0 ? s0_l : s0_r;
        AESBlock keep1 = alpha_bit == 0 ? s1_l : s1_r;
        uint8_t t_keep0 = alpha_bit == 0 ? t0_l : t0_r;
        uint8_t t_keep1 = alpha_bit == 0 ? t1_l : t1_r;

        AESBlock scw = lose0 ^ lose1;
        uint8_t tlcw = static_cast<uint8_t>((t0_l ^ t1_l ^ alpha_bit ^ 1) & 1);
        uint8_t trcw = static_cast<uint8_t>((t0_r ^ t1_r ^ alpha_bit) & 1);
        size_t off = static_cast<size_t>(idx) * static_cast<size_t>(log_domain) +
                     static_cast<size_t>(level);
        s_cw[off] = scw;
        t_l_cw[off] = tlcw;
        t_r_cw[off] = trcw;

        uint8_t t_chosen = alpha_bit == 0 ? tlcw : trcw;
        s0 = keep0 ^ (tau0 ? scw : static_cast<AESBlock>(0));
        s1 = keep1 ^ (tau1 ? scw : static_cast<AESBlock>(0));
        tau0 = static_cast<uint8_t>((t_keep0 ^ (tau0 ? t_chosen : 0)) & 1);
        tau1 = static_cast<uint8_t>((t_keep1 ^ (tau1 ? t_chosen : 0)) & 1);
    }

    Word c0 = convert_zp(s0, modulus);
    Word c1 = convert_zp(s1, modulus);
    Word diff = mod_sub(mod_add(beta, c1, modulus), c0, modulus);
    final_cw[idx] = tau1 == 0 ? diff : mod_sub(0, diff, modulus);
}

__device__ inline Word eval_one_dpf_zp(const DeviceGPUDPFZpKey key,
                                       int dpf_idx,
                                       Word x,
                                       AESSharedContext *aes) {
    AESBlock s = key.seeds[dpf_idx];
    uint8_t t = key.party == 0 ? 0 : 1;
    const size_t base = static_cast<size_t>(dpf_idx) *
                        static_cast<size_t>(key.log_domain);

    for (int level = 0; level < key.log_domain; ++level) {
        AESBlock s_l, s_r;
        uint8_t t_l, t_r;
        aes_prg_expand(s, aes, s_l, t_l, s_r, t_r);

        int bit_idx = key.log_domain - 1 - level;
        uint8_t x_bit = static_cast<uint8_t>((x >> bit_idx) & 1ULL);
        size_t off = base + static_cast<size_t>(level);
        if (x_bit == 0) {
            s = s_l ^ (t ? key.s_cw[off] : static_cast<AESBlock>(0));
            t = static_cast<uint8_t>((t_l ^ (t ? key.t_l_cw[off] : 0)) & 1);
        } else {
            s = s_r ^ (t ? key.s_cw[off] : static_cast<AESBlock>(0));
            t = static_cast<uint8_t>((t_r ^ (t ? key.t_r_cw[off] : 0)) & 1);
        }
    }

    Word c = convert_zp(s, key.modulus);
    Word v = t ? mod_add(c, key.final_cw[dpf_idx], key.modulus) : c;
    return key.party == 0 ? v : mod_sub(0, v, key.modulus);
}

__device__ inline void atomic_add_mod(Word *addr, Word value, Word modulus) {
    auto ull_addr = reinterpret_cast<unsigned long long *>(addr);
    unsigned long long old = *ull_addr;
    unsigned long long assumed;
    do {
        assumed = old;
        Word next = mod_add(static_cast<Word>(assumed), value, modulus);
        old = atomicCAS(ull_addr, assumed, static_cast<unsigned long long>(next));
    } while (assumed != old);
}

__global__ void dpf_zp_full_eval_sum_kernel(DeviceGPUDPFZpKey key,
                                            Word domain,
                                            Word *out,
                                            AESGlobalContext gaes) {
    AESSharedContext saes;
    loadSbox(&gaes, &saes);

    size_t total = static_cast<size_t>(key.count) * static_cast<size_t>(domain);
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }

    int dpf_idx = static_cast<int>(tid / static_cast<size_t>(domain));
    Word x = static_cast<Word>(tid - static_cast<size_t>(dpf_idx) * static_cast<size_t>(domain));
    Word share = eval_one_dpf_zp(key, dpf_idx, x, &saes);
    if (share != 0) {
        atomic_add_mod(&out[x], share, key.modulus);
    }
}

inline size_t serializedSizeGPUDPFZpKey(const GPUDPFZpKey &key) {
    return 4 * sizeof(int) + sizeof(Word) +
           key.seeds.size() * sizeof(AESBlock) +
           key.s_cw.size() * sizeof(AESBlock) +
           key.t_l_cw.size() * sizeof(uint8_t) +
           key.t_r_cw.size() * sizeof(uint8_t) +
           key.final_cw.size() * sizeof(Word);
}

template <typename T>
inline void write_bytes(uint8_t **ptr, const T *src, size_t count) {
    size_t bytes = count * sizeof(T);
    if (bytes != 0) {
        std::memcpy(*ptr, src, bytes);
        *ptr += bytes;
    }
}

template <typename T>
inline void read_bytes(uint8_t **ptr, std::vector<T> &dst, size_t count) {
    dst.resize(count);
    size_t bytes = count * sizeof(T);
    if (bytes != 0) {
        std::memcpy(dst.data(), *ptr, bytes);
        *ptr += bytes;
    }
}

inline void writeGPUDPFZpKey(uint8_t **key_as_bytes, const GPUDPFZpKey &key) {
    int meta[4] = {key.party, key.log_domain, key.count, 0};
    write_bytes(key_as_bytes, meta, 4);
    write_bytes(key_as_bytes, &key.modulus, 1);
    write_bytes(key_as_bytes, key.seeds.data(), key.seeds.size());
    write_bytes(key_as_bytes, key.s_cw.data(), key.s_cw.size());
    write_bytes(key_as_bytes, key.t_l_cw.data(), key.t_l_cw.size());
    write_bytes(key_as_bytes, key.t_r_cw.data(), key.t_r_cw.size());
    write_bytes(key_as_bytes, key.final_cw.data(), key.final_cw.size());
}

inline GPUDPFZpKey readGPUDPFZpKey(uint8_t **key_as_bytes) {
    int meta[4];
    std::memcpy(meta, *key_as_bytes, sizeof(meta));
    *key_as_bytes += sizeof(meta);

    GPUDPFZpKey key;
    key.party = meta[0];
    key.log_domain = meta[1];
    key.count = meta[2];
    std::memcpy(&key.modulus, *key_as_bytes, sizeof(Word));
    *key_as_bytes += sizeof(Word);

    size_t count = static_cast<size_t>(key.count);
    size_t levels = static_cast<size_t>(key.log_domain);
    read_bytes(key_as_bytes, key.seeds, count);
    read_bytes(key_as_bytes, key.s_cw, count * levels);
    read_bytes(key_as_bytes, key.t_l_cw, count * levels);
    read_bytes(key_as_bytes, key.t_r_cw, count * levels);
    read_bytes(key_as_bytes, key.final_cw, count);
    return key;
}

inline void copy_to_device(const GPUDPFZpKey &host, DeviceGPUDPFZpKey &dev,
                           AESBlock **d_seeds,
                           AESBlock **d_s_cw,
                           uint8_t **d_t_l_cw,
                           uint8_t **d_t_r_cw,
                           Word **d_final_cw) {
    size_t seed_bytes = host.seeds.size() * sizeof(AESBlock);
    size_t scw_bytes = host.s_cw.size() * sizeof(AESBlock);
    size_t tcw_bytes = host.t_l_cw.size() * sizeof(uint8_t);
    size_t final_bytes = host.final_cw.size() * sizeof(Word);

    cuda_check(cudaMalloc(reinterpret_cast<void **>(d_seeds), seed_bytes),
               "alloc DPF Zp seeds");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(d_s_cw), scw_bytes),
               "alloc DPF Zp s_cw");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(d_t_l_cw), tcw_bytes),
               "alloc DPF Zp t_l_cw");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(d_t_r_cw), tcw_bytes),
               "alloc DPF Zp t_r_cw");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(d_final_cw), final_bytes),
               "alloc DPF Zp final_cw");
    cuda_check(cudaMemcpy(*d_seeds, host.seeds.data(), seed_bytes, cudaMemcpyHostToDevice),
               "copy DPF Zp seeds");
    cuda_check(cudaMemcpy(*d_s_cw, host.s_cw.data(), scw_bytes, cudaMemcpyHostToDevice),
               "copy DPF Zp s_cw");
    cuda_check(cudaMemcpy(*d_t_l_cw, host.t_l_cw.data(), tcw_bytes, cudaMemcpyHostToDevice),
               "copy DPF Zp t_l_cw");
    cuda_check(cudaMemcpy(*d_t_r_cw, host.t_r_cw.data(), tcw_bytes, cudaMemcpyHostToDevice),
               "copy DPF Zp t_r_cw");
    cuda_check(cudaMemcpy(*d_final_cw, host.final_cw.data(), final_bytes, cudaMemcpyHostToDevice),
               "copy DPF Zp final_cw");

    dev.party = host.party;
    dev.log_domain = host.log_domain;
    dev.count = host.count;
    dev.modulus = host.modulus;
    dev.seeds = *d_seeds;
    dev.s_cw = *d_s_cw;
    dev.t_l_cw = *d_t_l_cw;
    dev.t_r_cw = *d_t_r_cw;
    dev.final_cw = *d_final_cw;
}

inline void free_device_key(AESBlock *d_seeds,
                            AESBlock *d_s_cw,
                            uint8_t *d_t_l_cw,
                            uint8_t *d_t_r_cw,
                            Word *d_final_cw) {
    cudaFree(d_seeds);
    cudaFree(d_s_cw);
    cudaFree(d_t_l_cw);
    cudaFree(d_t_r_cw);
    cudaFree(d_final_cw);
}

inline void gpuKeyGenDPFZpPair(const std::vector<Word> &alphas,
                               const std::vector<Word> &betas,
                               int log_domain,
                               Word modulus,
                               uint64_t seed,
                               AESGlobalContext *gaes,
                               GPUDPFZpKey &key0,
                               GPUDPFZpKey &key1) {
    assert(alphas.size() == betas.size());
    assert(log_domain > 0 && log_domain < 63);
    int count = static_cast<int>(alphas.size());
    Word domain = Word(1) << log_domain;
    for (size_t i = 0; i < alphas.size(); ++i) {
        assert(alphas[i] < domain);
        assert(betas[i] < modulus);
    }

    key0.party = 0;
    key1.party = 1;
    key0.log_domain = key1.log_domain = log_domain;
    key0.count = key1.count = count;
    key0.modulus = key1.modulus = modulus;
    key0.seeds.resize(count);
    key1.seeds.resize(count);
    key0.s_cw.resize(static_cast<size_t>(count) * static_cast<size_t>(log_domain));
    key1.s_cw.resize(key0.s_cw.size());
    key0.t_l_cw.resize(key0.s_cw.size());
    key1.t_l_cw.resize(key0.s_cw.size());
    key0.t_r_cw.resize(key0.s_cw.size());
    key1.t_r_cw.resize(key0.s_cw.size());
    key0.final_cw.resize(count);
    key1.final_cw.resize(count);

    Word *d_alphas = nullptr;
    Word *d_betas = nullptr;
    AESBlock *d_seeds0 = nullptr;
    AESBlock *d_seeds1 = nullptr;
    AESBlock *d_s_cw = nullptr;
    uint8_t *d_t_l_cw = nullptr;
    uint8_t *d_t_r_cw = nullptr;
    Word *d_final_cw = nullptr;

    size_t count_bytes = static_cast<size_t>(count) * sizeof(Word);
    size_t level_count = static_cast<size_t>(count) * static_cast<size_t>(log_domain);
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_alphas), count_bytes),
               "alloc DPF Zp alphas");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_betas), count_bytes),
               "alloc DPF Zp betas");
    cuda_check(cudaMemcpy(d_alphas, alphas.data(), count_bytes, cudaMemcpyHostToDevice),
               "copy DPF Zp alphas");
    cuda_check(cudaMemcpy(d_betas, betas.data(), count_bytes, cudaMemcpyHostToDevice),
               "copy DPF Zp betas");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_seeds0),
                          static_cast<size_t>(count) * sizeof(AESBlock)),
               "alloc DPF Zp seed0");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_seeds1),
                          static_cast<size_t>(count) * sizeof(AESBlock)),
               "alloc DPF Zp seed1");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_s_cw),
                          level_count * sizeof(AESBlock)),
               "alloc DPF Zp s_cw");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_t_l_cw),
                          level_count * sizeof(uint8_t)),
               "alloc DPF Zp t_l_cw");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_t_r_cw),
                          level_count * sizeof(uint8_t)),
               "alloc DPF Zp t_r_cw");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_final_cw),
                          static_cast<size_t>(count) * sizeof(Word)),
               "alloc DPF Zp final_cw");

    int block = 128;
    int grid = (count + block - 1) / block;
    keygen_dpf_zp_kernel<<<grid, block>>>(log_domain,
                                          count,
                                          modulus,
                                          d_alphas,
                                          d_betas,
                                          seed,
                                          d_seeds0,
                                          d_seeds1,
                                          d_s_cw,
                                          d_t_l_cw,
                                          d_t_r_cw,
                                          d_final_cw,
                                          *gaes);
    cuda_check(cudaGetLastError(), "launch DPF Zp keygen");
    cuda_check(cudaDeviceSynchronize(), "sync DPF Zp keygen");

    cuda_check(cudaMemcpy(key0.seeds.data(), d_seeds0,
                          static_cast<size_t>(count) * sizeof(AESBlock),
                          cudaMemcpyDeviceToHost),
               "copy DPF Zp seeds0");
    cuda_check(cudaMemcpy(key1.seeds.data(), d_seeds1,
                          static_cast<size_t>(count) * sizeof(AESBlock),
                          cudaMemcpyDeviceToHost),
               "copy DPF Zp seeds1");
    cuda_check(cudaMemcpy(key0.s_cw.data(), d_s_cw, level_count * sizeof(AESBlock),
                          cudaMemcpyDeviceToHost),
               "copy DPF Zp s_cw");
    cuda_check(cudaMemcpy(key0.t_l_cw.data(), d_t_l_cw, level_count * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost),
               "copy DPF Zp t_l_cw");
    cuda_check(cudaMemcpy(key0.t_r_cw.data(), d_t_r_cw, level_count * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost),
               "copy DPF Zp t_r_cw");
    cuda_check(cudaMemcpy(key0.final_cw.data(), d_final_cw,
                          static_cast<size_t>(count) * sizeof(Word),
                          cudaMemcpyDeviceToHost),
               "copy DPF Zp final_cw");
    key1.s_cw = key0.s_cw;
    key1.t_l_cw = key0.t_l_cw;
    key1.t_r_cw = key0.t_r_cw;
    key1.final_cw = key0.final_cw;

    cudaFree(d_alphas);
    cudaFree(d_betas);
    cudaFree(d_seeds0);
    cudaFree(d_seeds1);
    cudaFree(d_s_cw);
    cudaFree(d_t_l_cw);
    cudaFree(d_t_r_cw);
    cudaFree(d_final_cw);
}

inline GPUDPFZpKey gpuKeyGenDPFZp(int party,
                                  const std::vector<Word> &alphas,
                                  const std::vector<Word> &betas,
                                  int log_domain,
                                  Word modulus,
                                  uint64_t seed,
                                  AESGlobalContext *gaes) {
    GPUDPFZpKey k0, k1;
    gpuKeyGenDPFZpPair(alphas, betas, log_domain, modulus, seed, gaes, k0, k1);
    return party == 0 ? k0 : k1;
}

inline void gpuDpfZpFullEvalSum(const GPUDPFZpKey &key,
                                Word *d_out,
                                AESGlobalContext *gaes) {
    assert(key.log_domain > 0 && key.log_domain < 63);
    Word domain = Word(1) << key.log_domain;
    cuda_check(cudaMemset(d_out, 0, static_cast<size_t>(domain) * sizeof(Word)),
               "zero DPF Zp full-eval output");

    DeviceGPUDPFZpKey d_key;
    AESBlock *d_seeds = nullptr;
    AESBlock *d_s_cw = nullptr;
    uint8_t *d_t_l_cw = nullptr;
    uint8_t *d_t_r_cw = nullptr;
    Word *d_final_cw = nullptr;
    copy_to_device(key, d_key, &d_seeds, &d_s_cw, &d_t_l_cw, &d_t_r_cw, &d_final_cw);

    size_t total = static_cast<size_t>(key.count) * static_cast<size_t>(domain);
    int block = 128;
    int grid = static_cast<int>((total + static_cast<size_t>(block) - 1) /
                                static_cast<size_t>(block));
    dpf_zp_full_eval_sum_kernel<<<grid, block>>>(d_key, domain, d_out, *gaes);
    cuda_check(cudaGetLastError(), "launch DPF Zp full eval");
    cuda_check(cudaDeviceSynchronize(), "sync DPF Zp full eval");

    free_device_key(d_seeds, d_s_cw, d_t_l_cw, d_t_r_cw, d_final_cw);
}

}  // namespace ringlpn_spfss_zp
