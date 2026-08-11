#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "fss/gpu_aes_shm.h"

namespace ringlpn_spfss_zp {

using Word = uint64_t;
using U128 = unsigned __int128;

constexpr size_t kMaxBreadthStates = size_t(32) * 1024 * 1024;
constexpr size_t kMaxPreparedBatchKeyCount = 65535;

enum class DpfEvaluationPath {
    kBreadth,
    kRootToLeaf,
};

inline bool checked_size_product(size_t first, size_t second, size_t &out) {
    if (first != 0 &&
        second > std::numeric_limits<size_t>::max() / first) {
        return false;
    }
    out = first * second;
    return true;
}

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
__global__ void dpf_zp_full_eval_sum_batch_kernel(
    const DeviceGPUDPFZpKey *keys, int key_count, Word domain, Word *out,
    AESGlobalContext gaes) {
    const int key_index = static_cast<int>(blockIdx.y);
    if (key_index >= key_count) return;
    const DeviceGPUDPFZpKey key = keys[key_index];
    AESSharedContext saes;
    loadSbox(&gaes, &saes);

    const size_t total =
        static_cast<size_t>(key.count) * static_cast<size_t>(domain);
    const size_t tid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int dpf_index =
        static_cast<int>(tid / static_cast<size_t>(domain));
    const Word x = static_cast<Word>(
        tid - static_cast<size_t>(dpf_index) * static_cast<size_t>(domain));
    const Word share = eval_one_dpf_zp(key, dpf_index, x, &saes);
    if (share != 0) {
        atomic_add_mod(
            out + static_cast<size_t>(key_index) * domain + x,
            share, key.modulus);
    }
}
__global__ void dpf_zp_batch_tree_init_kernel(
    const DeviceGPUDPFZpKey *keys, int key_count, int max_count, Word domain,
    AESBlock *seeds, uint8_t *tags) {
    const size_t lane =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t lane_count =
        static_cast<size_t>(key_count) * static_cast<size_t>(max_count);
    if (lane >= lane_count) return;
    const int key_index = static_cast<int>(lane / max_count);
    const int dpf_index = static_cast<int>(lane % max_count);
    const DeviceGPUDPFZpKey key = keys[key_index];
    if (dpf_index >= key.count) return;
    const size_t state_index = lane * static_cast<size_t>(domain);
    seeds[state_index] = key.seeds[dpf_index];
    tags[state_index] = static_cast<uint8_t>(key.party == 0 ? 0 : 1);
}

__global__ void dpf_zp_batch_tree_expand_kernel(
    const DeviceGPUDPFZpKey *keys, int key_count, int max_count, Word domain,
    int level, AESBlock *input_seeds, uint8_t *input_tags,
    AESBlock *output_seeds, uint8_t *output_tags, AESGlobalContext gaes) {
    AESSharedContext saes;
    loadSbox(&gaes, &saes);
    const size_t nodes = size_t(1) << level;
    const size_t lane_count =
        static_cast<size_t>(key_count) * static_cast<size_t>(max_count);
    const size_t total = lane_count * nodes;
    const size_t tid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const size_t lane = tid / nodes;
    const size_t node = tid % nodes;
    const int key_index = static_cast<int>(lane / max_count);
    const int dpf_index = static_cast<int>(lane % max_count);
    const DeviceGPUDPFZpKey key = keys[key_index];
    if (dpf_index >= key.count) return;

    const size_t state_base = lane * static_cast<size_t>(domain);
    const AESBlock seed = input_seeds[state_base + node];
    const uint8_t tag = input_tags[state_base + node];
    AESBlock left_seed, right_seed;
    uint8_t left_tag, right_tag;
    aes_prg_expand(
        seed, &saes, left_seed, left_tag, right_seed, right_tag);
    const size_t correction =
        static_cast<size_t>(dpf_index) * key.log_domain + level;
    if (tag != 0) {
        left_seed ^= key.s_cw[correction];
        right_seed ^= key.s_cw[correction];
        left_tag ^= key.t_l_cw[correction];
        right_tag ^= key.t_r_cw[correction];
    }
    const size_t child = state_base + 2 * node;
    output_seeds[child] = left_seed;
    output_seeds[child + 1] = right_seed;
    output_tags[child] = static_cast<uint8_t>(left_tag & 1);
    output_tags[child + 1] = static_cast<uint8_t>(right_tag & 1);
}

__global__ void dpf_zp_batch_tree_finish_kernel(
    const DeviceGPUDPFZpKey *keys, int key_count, int max_count, Word domain,
    const AESBlock *seeds, const uint8_t *tags, Word *out) {
    const size_t total =
        static_cast<size_t>(key_count) * static_cast<size_t>(domain);
    const size_t tid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int key_index =
        static_cast<int>(tid / static_cast<size_t>(domain));
    const size_t point = tid % static_cast<size_t>(domain);
    const DeviceGPUDPFZpKey key = keys[key_index];
    Word sum = 0;
    for (int dpf_index = 0; dpf_index < key.count; ++dpf_index) {
        const size_t lane =
            static_cast<size_t>(key_index) * max_count + dpf_index;
        const size_t state_index =
            lane * static_cast<size_t>(domain) + point;
        Word value = convert_zp(seeds[state_index], key.modulus);
        if (tags[state_index] != 0) {
            value = mod_add(value, key.final_cw[dpf_index], key.modulus);
        }
        if (key.party != 0) value = mod_sub(0, value, key.modulus);
        sum = mod_add(sum, value, key.modulus);
    }
    out[tid] = sum;
}

inline void gpuDpfZpFullEvalSumPrepared(const DeviceGPUDPFZpKey &key,
                                        Word *d_out,
                                        AESGlobalContext *gaes) {
    assert(key.log_domain > 0 && key.log_domain < 63);
    assert(key.count > 0);
    const Word domain = Word(1) << key.log_domain;
    cuda_check(cudaMemset(d_out, 0, static_cast<size_t>(domain) * sizeof(Word)),
               "zero prepared DPF Zp full-eval output");
    constexpr int block = 128;
    const size_t total =
        static_cast<size_t>(key.count) * static_cast<size_t>(domain);
    const int grid = static_cast<int>(
        (total + static_cast<size_t>(block) - 1) /
        static_cast<size_t>(block));
    dpf_zp_full_eval_sum_kernel<<<grid, block>>>(
        key, domain, d_out, *gaes);
    cuda_check(cudaGetLastError(), "launch prepared DPF Zp full eval");
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
class DeviceGPUDPFZpKeyBatch {
  public:
    DeviceGPUDPFZpKeyBatch() = default;
    DeviceGPUDPFZpKeyBatch(const DeviceGPUDPFZpKeyBatch &) = delete;
    DeviceGPUDPFZpKeyBatch &operator=(const DeviceGPUDPFZpKeyBatch &) = delete;
    ~DeviceGPUDPFZpKeyBatch() { cleanup(); }

    bool initialize(const std::vector<GPUDPFZpKey> &keys,
                    bool enable_breadth_workspace = true) {
        cleanup();
        if (keys.empty()) return false;
        size_t seed_count = 0;
        size_t level_count = 0;
        const int log_domain = keys.front().log_domain;
        const Word modulus = keys.front().modulus;
        int max_count = 0;
        for (const GPUDPFZpKey &key : keys) {
            if (key.count <= 0 || key.log_domain != log_domain ||
                key.log_domain <= 0 || key.log_domain >= 63 ||
                key.modulus != modulus ||
                key.seeds.size() != static_cast<size_t>(key.count) ||
                key.s_cw.size() !=
                    static_cast<size_t>(key.count) * key.log_domain ||
                key.t_l_cw.size() != key.s_cw.size() ||
                key.t_r_cw.size() != key.s_cw.size() ||
                key.final_cw.size() != static_cast<size_t>(key.count)) {
                return false;
            }
            if (key.seeds.size() >
                    std::numeric_limits<size_t>::max() - seed_count ||
                key.s_cw.size() >
                    std::numeric_limits<size_t>::max() - level_count) {
                return false;
            }
            seed_count += key.seeds.size();
            level_count += key.s_cw.size();
            max_count = std::max(max_count, key.count);
        }

        std::vector<AESBlock> seeds;
        std::vector<AESBlock> s_cw;
        std::vector<uint8_t> t_l_cw;
        std::vector<uint8_t> t_r_cw;
        std::vector<Word> final_cw;
        seeds.reserve(seed_count);
        s_cw.reserve(level_count);
        t_l_cw.reserve(level_count);
        t_r_cw.reserve(level_count);
        final_cw.reserve(seed_count);
        std::vector<size_t> seed_offsets;
        std::vector<size_t> level_offsets;
        seed_offsets.reserve(keys.size());
        level_offsets.reserve(keys.size());
        for (const GPUDPFZpKey &key : keys) {
            seed_offsets.push_back(seeds.size());
            level_offsets.push_back(s_cw.size());
            seeds.insert(seeds.end(), key.seeds.begin(), key.seeds.end());
            s_cw.insert(s_cw.end(), key.s_cw.begin(), key.s_cw.end());
            t_l_cw.insert(t_l_cw.end(),
                          key.t_l_cw.begin(), key.t_l_cw.end());
            t_r_cw.insert(t_r_cw.end(),
                          key.t_r_cw.begin(), key.t_r_cw.end());
            final_cw.insert(final_cw.end(),
                            key.final_cw.begin(), key.final_cw.end());
        }

        allocate_copy(&d_seeds_, seeds, "copy DPF batch seeds");
        allocate_copy(&d_s_cw_, s_cw, "copy DPF batch seed CWs");
        allocate_copy(&d_t_l_cw_, t_l_cw, "copy DPF batch left tag CWs");
        allocate_copy(&d_t_r_cw_, t_r_cw, "copy DPF batch right tag CWs");
        allocate_copy(&d_final_cw_, final_cw, "copy DPF batch final CWs");
        keys_.reserve(keys.size());
        for (size_t index = 0; index < keys.size(); ++index) {
            const GPUDPFZpKey &key = keys[index];
            keys_.push_back(DeviceGPUDPFZpKey{
                key.party, key.log_domain, key.count, key.modulus,
                d_seeds_ + seed_offsets[index],
                d_s_cw_ + level_offsets[index],
                d_t_l_cw_ + level_offsets[index],
                d_t_r_cw_ + level_offsets[index],
                d_final_cw_ + seed_offsets[index],
            });
        }
        allocate_copy(&d_keys_, keys_, "copy DPF batch descriptors");
        log_domain_ = log_domain;
        max_count_ = max_count;
        const size_t domain = size_t(1) << log_domain;
        static std::atomic<bool> breadth_disabled_after_pressure{false};
        if (enable_breadth_workspace &&
            !breadth_disabled_after_pressure.load(std::memory_order_relaxed) &&
            keys.size() <=
                kMaxBreadthStates / static_cast<size_t>(max_count) / domain) {
            const size_t candidate_state_count =
                keys.size() * static_cast<size_t>(max_count) * domain;
            const size_t seed_bytes = candidate_state_count * sizeof(AESBlock);
            const size_t tag_bytes = candidate_state_count * sizeof(uint8_t);
            cudaError_t status = cudaMalloc(
                reinterpret_cast<void **>(&d_breadth_seeds0_), seed_bytes);
            if (status == cudaSuccess) {
                status = cudaMalloc(
                    reinterpret_cast<void **>(&d_breadth_seeds1_), seed_bytes);
            }
            if (status == cudaSuccess) {
                status = cudaMalloc(
                    reinterpret_cast<void **>(&d_breadth_tags0_), tag_bytes);
            }
            if (status == cudaSuccess) {
                status = cudaMalloc(
                    reinterpret_cast<void **>(&d_breadth_tags1_), tag_bytes);
            }
            if (status == cudaSuccess) {
                breadth_state_count_ = candidate_state_count;
            } else {
                cudaFree(d_breadth_seeds0_);
                cudaFree(d_breadth_seeds1_);
                cudaFree(d_breadth_tags0_);
                cudaFree(d_breadth_tags1_);
                d_breadth_seeds0_ = nullptr;
                d_breadth_seeds1_ = nullptr;
                d_breadth_tags0_ = nullptr;
                d_breadth_tags1_ = nullptr;
                breadth_state_count_ = 0;
                cudaGetLastError();
                if (!breadth_disabled_after_pressure.exchange(
                        true, std::memory_order_relaxed)) {
                    std::cerr
                        << "[gpu-spfss] breadth workspace unavailable; using "
                           "exact root-to-leaf fallback for this process\n";
                }
            }
        }
        return true;
    }

    const DeviceGPUDPFZpKey &at(size_t index) const {
        return keys_.at(index);
    }
    const DeviceGPUDPFZpKey *device_keys() const { return d_keys_; }
    size_t size() const { return keys_.size(); }
    int log_domain() const { return log_domain_; }
    int max_count() const { return max_count_; }
    bool breadth_ready() const { return breadth_state_count_ != 0; }
    size_t breadth_state_count() const { return breadth_state_count_; }
    AESBlock *breadth_seeds0() const { return d_breadth_seeds0_; }
    AESBlock *breadth_seeds1() const { return d_breadth_seeds1_; }
    uint8_t *breadth_tags0() const { return d_breadth_tags0_; }
    uint8_t *breadth_tags1() const { return d_breadth_tags1_; }

    void cleanup() {
        cudaFree(d_seeds_);
        cudaFree(d_s_cw_);
        cudaFree(d_t_l_cw_);
        cudaFree(d_t_r_cw_);
        cudaFree(d_final_cw_);
        cudaFree(d_keys_);
        cudaFree(d_breadth_seeds0_);
        cudaFree(d_breadth_seeds1_);
        cudaFree(d_breadth_tags0_);
        cudaFree(d_breadth_tags1_);
        d_seeds_ = nullptr;
        d_s_cw_ = nullptr;
        d_t_l_cw_ = nullptr;
        d_t_r_cw_ = nullptr;
        d_final_cw_ = nullptr;
        d_keys_ = nullptr;
        d_breadth_seeds0_ = nullptr;
        d_breadth_seeds1_ = nullptr;
        d_breadth_tags0_ = nullptr;
        d_breadth_tags1_ = nullptr;
        log_domain_ = 0;
        max_count_ = 0;
        breadth_state_count_ = 0;
        keys_.clear();
    }

  private:
    template <typename T>
    static void allocate_copy(T **destination,
                              const std::vector<T> &source,
                              const char *label) {
        cuda_check(cudaMalloc(reinterpret_cast<void **>(destination),
                              source.size() * sizeof(T)), label);
        cuda_check(cudaMemcpy(*destination, source.data(),
                              source.size() * sizeof(T),
                              cudaMemcpyHostToDevice), label);
    }

    std::vector<DeviceGPUDPFZpKey> keys_;
    AESBlock *d_seeds_ = nullptr;
    AESBlock *d_s_cw_ = nullptr;
    uint8_t *d_t_l_cw_ = nullptr;
    uint8_t *d_t_r_cw_ = nullptr;
    Word *d_final_cw_ = nullptr;
    DeviceGPUDPFZpKey *d_keys_ = nullptr;
    AESBlock *d_breadth_seeds0_ = nullptr;
    AESBlock *d_breadth_seeds1_ = nullptr;
    uint8_t *d_breadth_tags0_ = nullptr;
    uint8_t *d_breadth_tags1_ = nullptr;
    int log_domain_ = 0;
    int max_count_ = 0;
    size_t breadth_state_count_ = 0;
};

inline bool gpuDpfZpFullEvalSumBatchPrepared(
    const DeviceGPUDPFZpKeyBatch &keys, size_t key_offset, size_t key_count,
    Word *d_out, size_t d_out_word_capacity, AESGlobalContext *gaes,
    DpfEvaluationPath *executed_path = nullptr) {
    if (gaes == nullptr || d_out == nullptr || key_count == 0 ||
        key_count > keys.size() || key_offset > keys.size() - key_count ||
        key_count > kMaxPreparedBatchKeyCount ||
        key_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        keys.log_domain() <= 0 || keys.log_domain() >= 63 ||
        keys.max_count() <= 0) {
        return false;
    }
    const size_t domain = size_t(1) << keys.log_domain();
    size_t output_words = 0;
    size_t output_bytes = 0;
    if (!checked_size_product(key_count, domain, output_words) ||
        output_words > d_out_word_capacity ||
        !checked_size_product(output_words, sizeof(Word), output_bytes)) {
        return false;
    }

    constexpr size_t block = 128;
    if (!keys.breadth_ready()) {
        size_t threads = 0;
        if (!checked_size_product(static_cast<size_t>(keys.max_count()),
                                  domain, threads)) {
            return false;
        }
        const size_t grid_x_size = (threads - 1) / block + 1;
        if (grid_x_size == 0 ||
            grid_x_size >
                static_cast<size_t>(std::numeric_limits<unsigned>::max())) {
            return false;
        }
        cuda_check(cudaMemset(d_out, 0, output_bytes),
                   "zero prepared DPF Zp batch output");
        const dim3 grid(static_cast<unsigned>(grid_x_size),
                        static_cast<unsigned>(key_count));
        dpf_zp_full_eval_sum_batch_kernel<<<grid, static_cast<int>(block)>>>(
            keys.device_keys() + key_offset, static_cast<int>(key_count),
            static_cast<Word>(domain), d_out, *gaes);
        cuda_check(cudaGetLastError(),
                   "launch prepared DPF Zp batch fallback eval");
        if (executed_path != nullptr) {
            *executed_path = DpfEvaluationPath::kRootToLeaf;
        }
        return true;
    }

    size_t states_per_key = 0;
    size_t state_offset = 0;
    size_t required_states = 0;
    size_t lanes = 0;
    if (!checked_size_product(static_cast<size_t>(keys.max_count()), domain,
                              states_per_key) ||
        !checked_size_product(key_offset, states_per_key, state_offset) ||
        !checked_size_product(key_count, states_per_key, required_states) ||
        state_offset > keys.breadth_state_count() ||
        required_states > keys.breadth_state_count() - state_offset ||
        !checked_size_product(key_count,
                              static_cast<size_t>(keys.max_count()), lanes)) {
        return false;
    }
    AESBlock *input_seeds = keys.breadth_seeds0() + state_offset;
    AESBlock *output_seeds = keys.breadth_seeds1() + state_offset;
    uint8_t *input_tags = keys.breadth_tags0() + state_offset;
    uint8_t *output_tags = keys.breadth_tags1() + state_offset;
    const size_t init_grid_size = (lanes - 1) / block + 1;
    if (init_grid_size == 0 ||
        init_grid_size >
            static_cast<size_t>(std::numeric_limits<unsigned>::max())) {
        return false;
    }
    dpf_zp_batch_tree_init_kernel<<<static_cast<unsigned>(init_grid_size),
                                    static_cast<int>(block)>>>(
        keys.device_keys() + key_offset, static_cast<int>(key_count),
        keys.max_count(), static_cast<Word>(domain), input_seeds, input_tags);
    cuda_check(cudaGetLastError(), "launch DPF batch breadth init");

    for (int level = 0; level < keys.log_domain(); ++level) {
        const size_t nodes = size_t(1) << level;
        size_t threads = 0;
        if (!checked_size_product(lanes, nodes, threads)) return false;
        const size_t grid_size = (threads - 1) / block + 1;
        if (grid_size == 0 ||
            grid_size >
                static_cast<size_t>(std::numeric_limits<unsigned>::max())) {
            return false;
        }
        dpf_zp_batch_tree_expand_kernel<<<static_cast<unsigned>(grid_size),
                                          static_cast<int>(block)>>>(
            keys.device_keys() + key_offset, static_cast<int>(key_count),
            keys.max_count(), static_cast<Word>(domain), level, input_seeds,
            input_tags, output_seeds, output_tags, *gaes);
        cuda_check(cudaGetLastError(), "launch DPF batch breadth level");
        std::swap(input_seeds, output_seeds);
        std::swap(input_tags, output_tags);
    }

    const size_t finish_grid_size = (output_words - 1) / block + 1;
    if (finish_grid_size == 0 ||
        finish_grid_size >
            static_cast<size_t>(std::numeric_limits<unsigned>::max())) {
        return false;
    }
    dpf_zp_batch_tree_finish_kernel<<<static_cast<unsigned>(finish_grid_size),
                                      static_cast<int>(block)>>>(
        keys.device_keys() + key_offset, static_cast<int>(key_count),
        keys.max_count(), static_cast<Word>(domain), input_seeds, input_tags,
        d_out);
    cuda_check(cudaGetLastError(), "launch DPF batch breadth finish");
    if (executed_path != nullptr) {
        *executed_path = DpfEvaluationPath::kBreadth;
    }
    return true;
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
    assert(key.log_domain > 0 && key.log_domain < 63 && key.count > 0);

    DeviceGPUDPFZpKey d_key;
    AESBlock *d_seeds = nullptr;
    AESBlock *d_s_cw = nullptr;
    uint8_t *d_t_l_cw = nullptr;
    uint8_t *d_t_r_cw = nullptr;
    Word *d_final_cw = nullptr;
    copy_to_device(key, d_key, &d_seeds, &d_s_cw, &d_t_l_cw, &d_t_r_cw, &d_final_cw);

    gpuDpfZpFullEvalSumPrepared(d_key, d_out, gaes);
    cuda_check(cudaDeviceSynchronize(), "sync DPF Zp full eval");

    free_device_key(d_seeds, d_s_cw, d_t_l_cw, d_t_r_cw, d_final_cw);
}

}  // namespace ringlpn_spfss_zp
