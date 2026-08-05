#pragma once

#include "dpf_key_io.h"
#include "gpu_spfss_zp.cuh"

// The deployed CUDA NTT implementation is compiled into each Ring-LPN binary.
// Keep it behind this header so party-local users do not need to know which
// polynomial backend the proven Figure-2 equations use.
#ifndef RINGLPN_OLE_NTT_IMPLEMENTATION_INCLUDED
#define RINGLPN_OLE_NTT_IMPLEMENTATION_INCLUDED 1
#ifndef RINGLPN_DISABLE_MAIN
#define RINGLPN_DISABLE_MAIN 1
#define RINGLPN_OLE_UNDEF_DISABLE_MAIN 1
#endif
#define Stats RingOlePartyNttStats
#include "bench_ntt_cuda_cheddar.cu"
#undef Stats
#ifdef RINGLPN_OLE_UNDEF_DISABLE_MAIN
#undef RINGLPN_OLE_UNDEF_DISABLE_MAIN
#undef RINGLPN_DISABLE_MAIN
#endif
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ringlpn_ole_party {

using Word = uint64_t;
using NoiseRecord = ringlpn_keyio::spfss_groups::NoiseRecord;
using GPUDPFZpKey = ringlpn_spfss_zp::GPUDPFZpKey;

struct RingOlePublicParams {
    int n = 0;
    int c = 0;
    int t = 0;
    int log_domain = 0;
    int direction = 0;
    int limb = 0;
    int slot_batch = 0;
    Word modulus = 0;
    uint64_t public_a_seed = 0;
    bool regular = false;
};

struct RingOlePartyKeys {
    int party = -1;
    int log_domain = 0;
    Word modulus = 0;
    std::vector<uint64_t> noise_binding;
    std::vector<GPUDPFZpKey> grouped;
};

struct RingOlePartyShares {
    std::vector<Word> X_slots;
    std::vector<Word> Z_slots;
};

struct RingOlePartyCounters {
    size_t key_bytes = 0;
    uint64_t trees = 0;
};

inline int log2_exact(int value) {
    if (value <= 0 || (value & (value - 1)) != 0) return -1;
    int result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

inline int group_count(const RingOlePublicParams &p) {
    return p.regular ? 2 * p.t - 1 : 1;
}

inline int bucket_size(const RingOlePublicParams &p) {
    return p.regular ? p.n / p.t : 0;
}

inline int domain_size(const RingOlePublicParams &p) {
    return p.regular ? 2 * bucket_size(p) : 2 * p.n;
}

inline size_t points_in_group(const RingOlePublicParams &p, int group) {
    if (group < 0 || group >= group_count(p)) return 0;
    if (!p.regular) return static_cast<size_t>(p.t) * p.t;
    return static_cast<size_t>(
        std::min(std::min(group + 1, 2 * p.t - 1 - group), p.t));
}

inline bool modulus_config(Word modulus, ModulusConfig<Word> &config) {
    if (modulus == kConfig62.modulus) {
        config = kConfig62;
        return true;
    }
    if (modulus == kConfig62Crt2.modulus) {
        config = kConfig62Crt2;
        return true;
    }
    return false;
}

inline bool validate_public_params(const RingOlePublicParams &p) {
    ModulusConfig<Word> unused;
    if (p.n < kMinDegree || p.n > kMaxDegree || log2_exact(p.n) < 0 ||
        p.c <= 0 || p.t <= 0 || p.t > p.n ||
        (p.direction != 0 && p.direction != 1) || p.limb < 0 ||
        p.slot_batch < 0 || !modulus_config(p.modulus, unused)) {
        return false;
    }
    if (p.regular && (p.n % p.t != 0 || log2_exact(p.t) < 0)) return false;
    const uint64_t c = static_cast<uint64_t>(p.c);
    const uint64_t t = static_cast<uint64_t>(p.t);
    constexpr uint64_t kMaxTrees = 1ULL << 20;
    if (c > std::numeric_limits<uint64_t>::max() / c) return false;
    const uint64_t cc = c * c;
    if (t > std::numeric_limits<uint64_t>::max() / t ||
        cc > kMaxTrees / (t * t)) {
        return false;
    }
    constexpr uint64_t kMaxAggregateWords = 1ULL << 30;
    const uint64_t n = static_cast<uint64_t>(p.n);
    if (cc > std::numeric_limits<uint64_t>::max() / n) return false;
    const uint64_t cc_coeffs = cc * n;
    const uint64_t c_coeffs = c * n;
    if (cc_coeffs > (kMaxAggregateWords - 16 * n) / 12 ||
        c_coeffs >
            (kMaxAggregateWords - 16 * n - 12 * cc_coeffs) / 8) {
        return false;
    }
    return p.log_domain == log2_exact(domain_size(p));
}
inline std::vector<std::vector<Word>> make_public_polynomials(
    const RingOlePublicParams &p, std::mt19937_64 &public_rng) {
    std::vector<std::vector<Word>> a(static_cast<size_t>(p.c));
    a[0].assign(static_cast<size_t>(p.n), 0);
    a[0][0] = 1;
    std::uniform_int_distribution<Word> public_dist(0, p.modulus - 1);
    for (int i = 1; i < p.c; ++i) {
        a[static_cast<size_t>(i)].resize(static_cast<size_t>(p.n));
        for (Word &value : a[static_cast<size_t>(i)]) {
            value = public_dist(public_rng);
        }
    }
    return a;
}
inline bool validate_public_polynomials(
    const RingOlePublicParams &p, const std::vector<Word> &flat_a) {
    if (!validate_public_params(p) ||
        flat_a.size() != static_cast<size_t>(p.c) *
                             static_cast<size_t>(p.n) ||
        flat_a.empty() || flat_a[0] != 1) {
        return false;
    }
    for (int coefficient = 1; coefficient < p.n; ++coefficient) {
        if (flat_a[static_cast<size_t>(coefficient)] != 0) return false;
    }
    for (size_t coefficient = static_cast<size_t>(p.n);
         coefficient < flat_a.size(); ++coefficient) {
        if (flat_a[coefficient] >= p.modulus) return false;
    }
    return true;
}


inline bool validate_party_noise(const RingOlePublicParams &p,
                                 int party,
                                 const NoiseRecord &noise) {
    if (!validate_public_params(p) || (party != 0 && party != 1) ||
        noise.party != party || noise.c != p.c || noise.t != p.t ||
        noise.log_domain != p.log_domain || noise.modulus != p.modulus ||
        noise.regular != p.regular || noise.bucket != bucket_size(p)) {
        return false;
    }
    const size_t terms = static_cast<size_t>(p.c) * p.t;
    if (noise.positions.size() != terms || noise.values.size() != terms) return false;
    const int bucket = bucket_size(p);
    for (int poly = 0; poly < p.c; ++poly) {
        std::unordered_set<uint64_t> seen;
        if (!p.regular) seen.reserve(static_cast<size_t>(p.t) * 2 + 1);
        for (int k = 0; k < p.t; ++k) {
            const size_t idx = static_cast<size_t>(poly) * p.t + k;
            const uint64_t pos = noise.positions[idx];
            const uint64_t value = noise.values[idx];
            const bool position_ok =
                p.regular
                    ? pos >= static_cast<uint64_t>(k) * bucket &&
                          pos < static_cast<uint64_t>(k + 1) * bucket
                    : pos < static_cast<uint64_t>(p.n) && seen.insert(pos).second;
            if (!position_ok || value == 0 || value >= p.modulus) return false;
        }
    }
    return true;
}

inline bool validate_party_keys(const RingOlePublicParams &p,
                                int party,
                                const NoiseRecord &noise,
                                const RingOlePartyKeys &keys,
                                RingOlePartyCounters *counters = nullptr) {
    if (!validate_party_noise(p, party, noise) || keys.party != party ||
        keys.log_domain != p.log_domain || keys.modulus != p.modulus ||
        keys.noise_binding != ringlpn_keyio::spfss_groups::noise_binding(noise)) {
        return false;
    }
    const size_t groups = static_cast<size_t>(group_count(p));
    if (keys.grouped.size() != static_cast<size_t>(p.c) * p.c * groups)
        return false;
    uint64_t trees = 0;
    size_t bytes = 0;
    const size_t levels = static_cast<size_t>(p.log_domain);
    for (size_t idx = 0; idx < keys.grouped.size(); ++idx) {
        const GPUDPFZpKey &key = keys.grouped[idx];
        const size_t expected = points_in_group(p, static_cast<int>(idx % groups));
        if (expected == 0 || expected > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            key.party != party || key.log_domain != p.log_domain ||
            key.modulus != p.modulus || key.count != static_cast<int>(expected) ||
            key.seeds.size() != expected || key.s_cw.size() != expected * levels ||
            key.t_l_cw.size() != expected * levels ||
            key.t_r_cw.size() != expected * levels || key.final_cw.size() != expected) {
            return false;
        }
        for (size_t k = 0; k < expected; ++k) {
            if (key.final_cw[k] >= p.modulus) return false;
            for (size_t level = 0; level < levels; ++level) {
                const size_t off = k * levels + level;
                if (key.t_l_cw[off] > 1 || key.t_r_cw[off] > 1) return false;
            }
        }
        if (trees > std::numeric_limits<uint64_t>::max() - expected) return false;
        trees += expected;
        const size_t key_bytes = ringlpn_spfss_zp::serializedSizeGPUDPFZpKey(key);
        if (bytes > std::numeric_limits<size_t>::max() - key_bytes) return false;
        bytes += key_bytes;
    }
    const uint64_t expected_trees = static_cast<uint64_t>(p.c) * p.c * p.t * p.t;
    if (trees != expected_trees) return false;
    if (counters) {
        counters->trees = trees;
        counters->key_bytes = bytes;
    }
    return true;
}
inline bool pack_gpu_party_keys(
    const RingOlePublicParams &p,
    int party,
    const NoiseRecord &noise,
    const std::vector<uint64_t> &noise_binding,
    const std::vector<std::vector<spfss_host::DPFKey>> &grouped,
    RingOlePartyKeys &out) {
    if (!validate_party_noise(p, party, noise) ||
        noise_binding != ringlpn_keyio::spfss_groups::noise_binding(noise)) {
        return false;
    }
    const size_t groups = static_cast<size_t>(group_count(p));
    const size_t expected_groups =
        static_cast<size_t>(p.c) * p.c * groups;
    if (grouped.size() != expected_groups) return false;
    RingOlePartyKeys packed;
    packed.party = party;
    packed.log_domain = p.log_domain;
    packed.modulus = p.modulus;
    packed.noise_binding = noise_binding;
    packed.grouped.assign(expected_groups, GPUDPFZpKey{});
    const size_t levels = static_cast<size_t>(p.log_domain);
    for (size_t group_idx = 0; group_idx < expected_groups; ++group_idx) {
        const auto &src = grouped[group_idx];
        const size_t expected_points =
            points_in_group(p, static_cast<int>(group_idx % groups));
        if (src.size() != expected_points || expected_points == 0 ||
            expected_points >
                static_cast<size_t>(std::numeric_limits<int>::max())) {
            return false;
        }
        GPUDPFZpKey &key = packed.grouped[group_idx];
        key.party = party;
        key.log_domain = p.log_domain;
        key.count = static_cast<int>(src.size());
        key.modulus = p.modulus;
        key.seeds.resize(src.size());
        key.s_cw.resize(src.size() * levels);
        key.t_l_cw.resize(src.size() * levels);
        key.t_r_cw.resize(src.size() * levels);
        key.final_cw.resize(src.size());
        for (size_t k = 0; k < src.size(); ++k) {
            const spfss_host::DPFKey &host_key = src[k];
            if (host_key.t0 != static_cast<uint8_t>(party) ||
                host_key.log_domain != p.log_domain ||
                host_key.modulus != p.modulus ||
                host_key.sCW.size() != levels ||
                host_key.tLCW.size() != levels ||
                host_key.tRCW.size() != levels ||
                host_key.finalCW >= p.modulus) {
                return false;
            }
            key.seeds[k] = static_cast<AESBlock>(host_key.seed);
            key.final_cw[k] = host_key.finalCW;
            for (size_t level = 0; level < levels; ++level) {
                if (host_key.tLCW[level] > 1 ||
                    host_key.tRCW[level] > 1) {
                    return false;
                }
                const size_t off = k * levels + level;
                key.s_cw[off] = static_cast<AESBlock>(host_key.sCW[level]);
                key.t_l_cw[off] = host_key.tLCW[level];
                key.t_r_cw[off] = host_key.tRCW[level];
            }
        }
    }
    if (!validate_party_keys(p, party, noise, packed)) return false;
    out = std::move(packed);
    return true;
}

__device__ __forceinline__ Word party_mod_add(Word a, Word b, Word modulus) {
    Word sum = a + b;
    return (sum >= modulus || sum < a) ? sum - modulus : sum;
}

__device__ __forceinline__ Word party_mod_sub(Word a, Word b, Word modulus) {
    return a >= b ? a - b : modulus - (b - a);
}
static __global__ void party_reduce_batches_kernel(const Word *batches,
                                                   Word *out,
                                                   int batch_count,
                                                   int n,
                                                   Word modulus) {
    const size_t coeff =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff >= static_cast<size_t>(n)) return;
    Word acc = 0;
    for (int batch = 0; batch < batch_count; ++batch) {
        acc = party_mod_add(
            acc, batches[static_cast<size_t>(batch) * n + coeff], modulus);
    }
    out[coeff] = acc;
}

static __global__ void party_fold_2n_to_n_kernel(const Word *in,
                                                 Word *out,
                                                 int n,
                                                 Word modulus) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(n)) {
        out[idx] =
            party_mod_sub(in[idx], in[idx + static_cast<size_t>(n)], modulus);
    }
}

static __global__ void party_scatter_regular_group_kernel(const Word *group,
                                                          Word *full,
                                                          int group_domain,
                                                          int base,
                                                          Word modulus) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(group_domain)) {
        const Word value = group[idx];
        if (value != 0) {
            const size_t out_idx = static_cast<size_t>(base) + idx;
            full[out_idx] = party_mod_add(full[out_idx], value, modulus);
        }
    }
}

static __global__ void party_accumulate_kernel(Word *out,
                                               const Word *term,
                                               int n,
                                               Word modulus) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(n)) {
        out[idx] = party_mod_add(out[idx], term[idx], modulus);
    }
}

inline void reduce_batches(const Word *d_batches, Word *d_out, int batch_count,
                           int n, Word modulus) {
    dim3 block(256);
    dim3 grid(grid_size(static_cast<size_t>(n), block.x));
    party_reduce_batches_kernel<<<grid, block>>>(d_batches, d_out, batch_count, n, modulus);
    check(cudaGetLastError(), "launch party_reduce_batches_kernel");
}

inline void fold_2n_to_n(const Word *d_in, Word *d_out, int n, Word modulus) {
    dim3 block(256);
    dim3 grid(grid_size(static_cast<size_t>(n), block.x));
    party_fold_2n_to_n_kernel<<<grid, block>>>(d_in, d_out, n, modulus);
    check(cudaGetLastError(), "launch party_fold_2n_to_n_kernel");
}

inline void run_x(const RingOlePublicParams &p,
                  const DeviceTables<Word> &tables,
                  const Word *d_a_ntt,
                  Word *d_e,
                  Word *d_bw,
                  Word *d_cw,
                  Word *d_terms,
                  Word *d_x) {
    run_polymul_prepared_lhs(d_a_ntt, d_e, d_bw, d_cw, d_terms, tables,
                             p.n, p.c, log2_exact(p.n));
    reduce_batches(d_terms, d_x, p.c, p.n, p.modulus);
}

inline void run_spfss(const RingOlePublicParams &p,
                      const std::vector<GPUDPFZpKey> &keys,
                      Word *d_group,
                      Word *d_u2n,
                      Word *d_u,
                      AESGlobalContext *gaes) {
    const int cc = p.c * p.c;
    const int groups = group_count(p);
    for (int idx = 0; idx < cc; ++idx) {
        if (p.regular) {
            const int bucket = bucket_size(p);
            const int group_domain = domain_size(p);
            check(cudaMemset(d_u2n, 0, static_cast<size_t>(2) * p.n * sizeof(Word)),
                  "zero party regular u2n");
            for (int group = 0; group < groups; ++group) {
                const size_t key_idx = static_cast<size_t>(idx) * groups + group;
                ringlpn_spfss_zp::gpuDpfZpFullEvalSum(keys[key_idx], d_group, gaes);
                dim3 block(256);
                dim3 grid(grid_size(static_cast<size_t>(group_domain), block.x));
                party_scatter_regular_group_kernel<<<grid, block>>>(
                    d_group, d_u2n, group_domain, group * bucket, p.modulus);
                check(cudaGetLastError(), "launch party regular scatter");
            }
        } else {
            ringlpn_spfss_zp::gpuDpfZpFullEvalSum(keys[static_cast<size_t>(idx)], d_u2n, gaes);
        }
        fold_2n_to_n(d_u2n, d_u + static_cast<size_t>(idx) * p.n, p.n, p.modulus);
    }
}

inline void run_z(const RingOlePublicParams &p,
                  const DeviceTables<Word> &tables,
                  const Word *d_aa_ntt,
                  Word *d_u,
                  Word *d_bw,
                  Word *d_cw,
                  Word *d_terms,
                  Word *d_z) {
    const int cc = p.c * p.c;
    run_polymul_prepared_lhs(d_aa_ntt, d_u, d_bw, d_cw, d_terms, tables,
                             p.n, cc, log2_exact(p.n));
    reduce_batches(d_terms, d_z, cc, p.n, p.modulus);
}

// Memory-bounded variants used by large regular Ring-LPN instances. They
// preserve the same reductions as run_x/run_spfss/run_z but keep only one
// coefficient polynomial and one a_i*a_j NTT product live at a time.
inline void run_x_streamed(const RingOlePublicParams &p,
                           const DeviceTables<Word> &tables,
                           const Word *d_a_ntt,
                           Word *d_e,
                           Word *d_bw,
                           Word *d_cw,
                           Word *d_term,
                           Word *d_x) {
    check(cudaMemset(d_x, 0, static_cast<size_t>(p.n) * sizeof(Word)),
          "zero streamed party x");
    for (int i = 0; i < p.c; ++i) {
        run_polymul_prepared_lhs(
            d_a_ntt + static_cast<size_t>(i) * p.n,
            d_e + static_cast<size_t>(i) * p.n, d_bw, d_cw, d_term, tables,
            p.n, 1, log2_exact(p.n));
        dim3 block(256);
        dim3 grid(grid_size(static_cast<size_t>(p.n), block.x));
        party_accumulate_kernel<<<grid, block>>>(d_x, d_term, p.n, p.modulus);
        check(cudaGetLastError(), "launch streamed party x accumulate");
    }
}

inline void run_spfss_z_streamed(const RingOlePublicParams &p,
                                 const DeviceTables<Word> &tables,
                                 const Word *d_a_ntt,
                                 const std::vector<GPUDPFZpKey> &keys,
                                 Word *d_group,
                                 Word *d_u2n,
                                 Word *d_u,
                                 Word *d_aa_ntt,
                                 Word *d_bw,
                                 Word *d_cw,
                                 Word *d_term,
                                 Word *d_z,
                                 AESGlobalContext *gaes) {
    check(cudaMemset(d_z, 0, static_cast<size_t>(p.n) * sizeof(Word)),
          "zero streamed party z");
    const int groups = group_count(p);
    dim3 block(256);
    dim3 n_grid(grid_size(static_cast<size_t>(p.n), block.x));
    for (int j = 0; j < p.c; ++j) {
        for (int i = 0; i < p.c; ++i) {
            const int matrix_idx = i + j * p.c;
            pointwise_mul_kernel<Word><<<n_grid, block>>>(
                d_a_ntt + static_cast<size_t>(i) * p.n,
                d_a_ntt + static_cast<size_t>(j) * p.n, d_aa_ntt,
                static_cast<size_t>(p.n), p.n, 1, tables.d_primes,
                tables.d_inv_primes);
            check(cudaGetLastError(), "launch streamed party public product");
            if (p.regular) {
                const int bucket = bucket_size(p);
                const int group_domain = domain_size(p);
                check(cudaMemset(d_u2n, 0,
                                 static_cast<size_t>(2) * p.n * sizeof(Word)),
                      "zero streamed party regular u2n");
                for (int group = 0; group < groups; ++group) {
                    const size_t key_idx =
                        static_cast<size_t>(matrix_idx) * groups + group;
                    ringlpn_spfss_zp::gpuDpfZpFullEvalSum(
                        keys[key_idx], d_group, gaes);
                    dim3 group_grid(
                        grid_size(static_cast<size_t>(group_domain), block.x));
                    party_scatter_regular_group_kernel<<<group_grid, block>>>(
                        d_group, d_u2n, group_domain, group * bucket,
                        p.modulus);
                    check(cudaGetLastError(),
                          "launch streamed party regular scatter");
                }
            } else {
                ringlpn_spfss_zp::gpuDpfZpFullEvalSum(
                    keys[static_cast<size_t>(matrix_idx)], d_u2n, gaes);
            }
            fold_2n_to_n(d_u2n, d_u, p.n, p.modulus);
            run_polymul_prepared_lhs(d_aa_ntt, d_u, d_bw, d_cw, d_term,
                                     tables, p.n, 1, log2_exact(p.n));
            party_accumulate_kernel<<<n_grid, block>>>(
                d_z, d_term, p.n, p.modulus);
            check(cudaGetLastError(), "launch streamed party z accumulate");
        }
    }
}

class RingOlePartyContext {
  public:
    RingOlePartyContext() = default;
    RingOlePartyContext(const RingOlePartyContext &) = delete;
    RingOlePartyContext &operator=(const RingOlePartyContext &) = delete;
    ~RingOlePartyContext() { cleanup(); }

    bool initialize(const RingOlePublicParams &params,
                    int party,
                    const NoiseRecord &own_noise,
                    RingOlePartyKeys own_keys,
                    const std::vector<Word> *public_a = nullptr) {
        cleanup();
        RingOlePartyCounters validated;
        if (!validate_party_keys(params, party, own_noise, own_keys, &validated)) return false;

        params_ = params;
        counters_ = validated;
        keys_ = std::move(own_keys.grouped);
        if (!modulus_config(params.modulus, config_)) return false;
        log_degree_ = log2_exact(params.n);

        HostTables<Word> host_tables;
        compute_cheddar_tables(host_tables, params.n, config_);
        alloc_and_copy(tables_, host_tables);
        compute_reference_vectors(phi_norm_, post_norm_, params.n, config_);

        std::vector<Word> flat_a;
        if (public_a != nullptr) {
            if (!validate_public_polynomials(params, *public_a)) return false;
            flat_a = *public_a;
        } else {
            std::mt19937_64 public_rng(params.public_a_seed);
            std::vector<std::vector<Word>> a =
                make_public_polynomials(params, public_rng);
            flat_a.resize(static_cast<size_t>(params.c) * params.n);
            for (int i = 0; i < params.c; ++i) {
                std::copy(a[static_cast<size_t>(i)].begin(),
                          a[static_cast<size_t>(i)].end(),
                          flat_a.begin() + static_cast<size_t>(i) * params.n);
            }
        }
        std::vector<Word> dense_e(static_cast<size_t>(params.c) * params.n, 0);
        for (int i = 0; i < params.c; ++i) {
            for (int k = 0; k < params.t; ++k) {
                const size_t term = static_cast<size_t>(i) * params.t + k;
                dense_e[static_cast<size_t>(i) * params.n +
                        own_noise.positions[term]] = own_noise.values[term];
            }
        }

        copy_device(&d_a_, flat_a, "party copy public a");
        copy_device(&d_e_, dense_e, "party copy own e");

        const size_t c_coeffs = static_cast<size_t>(params.c) * params.n;
        allocate(&d_a_ntt_, c_coeffs, "party alloc a ntt");
        allocate(&d_aw_, params.n, "party alloc streamed public product");
        allocate(&d_bw_, params.n, "party alloc streamed work b");
        allocate(&d_cw_, params.n, "party alloc streamed work c");
        allocate(&d_terms_, params.n, "party alloc streamed term");
        allocate(&d_x_, params.n, "party alloc x");
        allocate(&d_u_, params.n, "party alloc streamed u");
        allocate(&d_u2n_, static_cast<size_t>(2) * params.n,
                 "party alloc u2n");
        if (params.regular) {
            allocate(&d_group_, domain_size(params), "party alloc group");
        }
        allocate(&d_z_, params.n, "party alloc z");

        run_forward_only(d_a_, d_a_ntt_, tables_, params.n, params.c,
                         log_degree_);
        check(cudaDeviceSynchronize(), "sync party public setup");
        initialized_ = true;
        cudaFree(d_a_);
        d_a_ = nullptr;
        return true;
    }

    bool expand_device(AESGlobalContext *gaes) {
        if (!initialized_ || gaes == nullptr) return false;
        run_x_streamed(params_, tables_, d_a_ntt_, d_e_, d_bw_, d_cw_,
                       d_terms_, d_x_);
        run_spfss_z_streamed(params_, tables_, d_a_ntt_, keys_, d_group_,
                             d_u2n_, d_u_, d_aw_, d_bw_, d_cw_, d_terms_,
                             d_z_, gaes);
        return true;
    }

    bool copy_shares(RingOlePartyShares &out) const {
        if (!initialized_) return false;
        out.X_slots.resize(static_cast<size_t>(params_.n));
        out.Z_slots.resize(static_cast<size_t>(params_.n));
        check(cudaMemcpy(out.X_slots.data(), d_x_, sizeof(Word) * params_.n,
                         cudaMemcpyDeviceToHost), "copy party x slots");
        check(cudaMemcpy(out.Z_slots.data(), d_z_, sizeof(Word) * params_.n,
                         cudaMemcpyDeviceToHost), "copy party z slots");
        host_forward_ntt(out.X_slots, phi_norm_, params_.n, log_degree_, config_);
        host_forward_ntt(out.Z_slots, phi_norm_, params_.n, log_degree_, config_);
        return true;
    }

    const RingOlePublicParams &params() const { return params_; }
    const RingOlePartyCounters &counters() const { return counters_; }
    Word *device_x_coeff() const { return d_x_; }
    Word *device_z_coeff() const { return d_z_; }
    const DeviceTables<Word> &device_tables() const { return tables_; }
    const ModulusConfig<Word> &config() const { return config_; }
    const std::vector<Word> &phi_norm() const { return phi_norm_; }
    const std::vector<Word> &post_norm() const { return post_norm_; }

    void cleanup() {
        free_tables(tables_);
        tables_ = DeviceTables<Word>{};
        cudaFree(d_a_); cudaFree(d_a_ntt_); cudaFree(d_e_);
        cudaFree(d_aw_); cudaFree(d_bw_); cudaFree(d_cw_); cudaFree(d_terms_);
        cudaFree(d_x_); cudaFree(d_u_); cudaFree(d_u2n_); cudaFree(d_group_); cudaFree(d_z_);
        d_a_ = d_a_ntt_ = d_e_ = nullptr;
        d_aw_ = d_bw_ = d_cw_ = d_terms_ = nullptr;
        d_x_ = d_u_ = d_u2n_ = d_group_ = d_z_ = nullptr;
        keys_.clear();
        initialized_ = false;
    }

  private:
    static void allocate(Word **dst, size_t count, const char *label) {
        check(cudaMalloc(reinterpret_cast<void **>(dst), count * sizeof(Word)), label);
    }

    static void copy_device(Word **dst, const std::vector<Word> &src, const char *label) {
        allocate(dst, src.size(), label);
        check(cudaMemcpy(*dst, src.data(), src.size() * sizeof(Word),
                         cudaMemcpyHostToDevice), label);
    }

    RingOlePublicParams params_;
    int log_degree_ = 0;
    bool initialized_ = false;
    ModulusConfig<Word> config_ = kConfig62;
    DeviceTables<Word> tables_;
    std::vector<Word> phi_norm_;
    std::vector<Word> post_norm_;
    std::vector<GPUDPFZpKey> keys_;
    RingOlePartyCounters counters_;

    Word *d_a_ = nullptr;
    Word *d_a_ntt_ = nullptr;
    Word *d_e_ = nullptr;
    Word *d_aw_ = nullptr;
    Word *d_bw_ = nullptr;
    Word *d_cw_ = nullptr;
    Word *d_terms_ = nullptr;
    Word *d_x_ = nullptr;
    Word *d_u_ = nullptr;
    Word *d_u2n_ = nullptr;
    Word *d_group_ = nullptr;
    Word *d_z_ = nullptr;
};

inline bool expand_ring_ole_party(const RingOlePublicParams &params,
                                  int party,
                                  const NoiseRecord &own_noise,
                                  RingOlePartyKeys own_keys,
                                  AESGlobalContext *gaes,
                                  RingOlePartyShares &out,
                                  RingOlePartyCounters &counters,
                                  const std::vector<Word> *public_a = nullptr) {
    RingOlePartyContext context;
    if (!context.initialize(params, party, own_noise, std::move(own_keys),
                            public_a) ||
        !context.expand_device(gaes)) {
        return false;
    }
    check(cudaDeviceSynchronize(), "sync party Ring-OLE expansion");
    if (!context.copy_shares(out)) return false;
    counters = context.counters();
    return true;
}

}  // namespace ringlpn_ole_party
