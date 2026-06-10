#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "fss/gpu_matmul.h"

namespace ringlpn_orca {

using u128 = unsigned __int128;

constexpr uint64_t kPrime62 = 4611686018326724609ULL;
constexpr uint64_t kPrime62Crt2 = 4611686018309947393ULL;
constexpr int kDefaultQbits = 128;
constexpr int kDefaultActualQbits = 124;

inline bool fcKeysEnabled() {
    const char *flag = std::getenv("ORCA_RINGLPN_FC_KEYS");
    return flag && std::strcmp(flag, "0") != 0;
}

inline int requestedQbitsFromEnv() {
    const char *value = std::getenv("ORCA_RINGLPN_FC_QBITS");
    if (!value || !*value) {
        return kDefaultQbits;
    }
    return std::atoi(value);
}

inline u128 q128Modulus() {
    return u128(kPrime62) * u128(kPrime62Crt2);
}

inline u128 modulusForQbits(int qbits) {
    return qbits == 128 ? q128Modulus() : u128(kPrime62);
}

inline int actualQbitsForQbits(int qbits) {
    return qbits == 128 ? kDefaultActualQbits : 62;
}

inline uint64_t ringMask(int bw) {
    return bw == 64 ? UINT64_MAX : ((uint64_t(1) << bw) - 1);
}

inline uint64_t ringReduce(u128 x, int bw) {
    return bw == 64 ? static_cast<uint64_t>(x)
                    : static_cast<uint64_t>(x & ((u128(1) << bw) - 1));
}

inline uint64_t ringAdd(uint64_t a, uint64_t b, int bw) {
    return ringReduce(u128(a) + b, bw);
}

inline uint64_t ringSub(uint64_t a, uint64_t b, int bw) {
    return ringReduce(u128(a) + (u128(1) << bw) - ringReduce(b, bw), bw);
}

inline u128 modSub(u128 a, u128 b, u128 modulus) {
    return a >= b ? a - b : a + modulus - b;
}

inline u128 uniformMod(u128 modulus, std::mt19937_64 &rng) {
    u128 x = (u128(rng()) << 64) ^ u128(rng());
    return x % modulus;
}

inline uint64_t mixSeed(uint64_t seed, uint64_t tag) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL + (tag << 6) + (tag >> 2);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline uint64_t baseSeed() {
    const char *value = std::getenv("ORCA_RINGLPN_FC_SEED");
    return value && *value ? std::strtoull(value, nullptr, 10) : 1;
}

inline void exactZmToRingShares(u128 z0,
                                u128 z1,
                                u128 modulus,
                                int bw,
                                uint64_t &r0,
                                uint64_t &r1) {
    const bool carry = z0 + z1 >= modulus;
    r0 = ringReduce(z0, bw);
    r1 = ringReduce(z1, bw);
    if (carry) {
        r1 = ringSub(r1, ringReduce(modulus, bw), bw);
    }
}

inline std::pair<uint64_t, uint64_t> shareModValueToRing(u128 value,
                                                         int bw,
                                                         u128 modulus,
                                                         std::mt19937_64 &rng) {
    u128 z0 = uniformMod(modulus, rng);
    u128 z1 = modSub(value % modulus, z0, modulus);
    uint64_t r0 = 0;
    uint64_t r1 = 0;
    exactZmToRingShares(z0, z1, modulus, bw, r0, r1);
    return {r0, r1};
}

inline std::pair<uint64_t, uint64_t> shareRingValue(uint64_t value,
                                                   int bw,
                                                   std::mt19937_64 &rng) {
    std::uniform_int_distribution<uint64_t> dist(0, ringMask(bw));
    uint64_t r0 = dist(rng);
    uint64_t r1 = ringSub(value, r0, bw);
    return {r0, r1};
}

inline uint64_t matmulSeed(const MatmulParams &p, uint64_t tag) {
    uint64_t seed = baseSeed();
    seed = mixSeed(seed, static_cast<uint64_t>(p.bw));
    seed = mixSeed(seed, static_cast<uint64_t>(p.M) << 32 ^ static_cast<uint64_t>(p.K));
    seed = mixSeed(seed, static_cast<uint64_t>(p.N) << 32 ^ static_cast<uint64_t>(p.batchSz));
    return mixSeed(seed, tag);
}

template <typename T>
inline bool copyDeviceVector(const T *d_src, size_t count, std::vector<T> &out) {
    out.resize(count);
    cudaError_t err = cudaMemcpy(out.data(), d_src, count * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Ring-LPN FC keywriter cudaMemcpy failed: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }
    return true;
}

template <typename T>
inline void appendRaw(u8 **key_as_bytes, const std::vector<T> &src) {
    std::memcpy(*key_as_bytes, src.data(), src.size() * sizeof(T));
    *key_as_bytes += src.size() * sizeof(T);
}

template <typename T>
inline uint64_t ringValue(T value, int bw) {
    return static_cast<uint64_t>(value) & ringMask(bw);
}

template <typename T>
inline uint64_t matrixValue(const std::vector<T> &m,
                            size_t base,
                            int rows,
                            int cols,
                            bool row_major,
                            int row,
                            int col,
                            int bw) {
    size_t idx = row_major ? static_cast<size_t>(row) * cols + col
                           : static_cast<size_t>(col) * rows + row;
    idx += base;
    return ringValue(m[idx], bw);
}

template <typename T>
inline bool makeSharedVector(int party,
                             const std::vector<T> &values,
                             int bw,
                             int qbits,
                             uint64_t tag,
                             std::vector<T> &shares) {
    if (bw <= 2 || bw > 32 || (qbits != 64 && qbits != 128)) {
        return false;
    }
    u128 modulus = modulusForQbits(qbits);
    std::mt19937_64 rng(mixSeed(baseSeed(), tag));
    shares.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        auto s = shareModValueToRing(ringValue(values[i], bw), bw, modulus, rng);
        shares[i] = static_cast<T>(party == SERVER0 ? s.first : s.second);
    }
    return true;
}

template <typename T>
inline bool shareVectorToKey(u8 **key_as_bytes,
                             int party,
                             const std::vector<T> &values,
                             int bw,
                             int qbits,
                             uint64_t tag) {
    std::vector<T> shares;
    if (!makeSharedVector(party, values, bw, qbits, tag, shares)) {
        return false;
    }
    appendRaw(key_as_bytes, shares);
    return true;
}

template <typename T>
inline bool writeValueShares(u8 **key_as_bytes,
                             int party,
                             int count,
                             const T *d_values,
                             int bw,
                             int qbits,
                             uint64_t tag) {
    std::vector<T> values;
    if (!copyDeviceVector(d_values, static_cast<size_t>(count), values)) {
        return false;
    }
    return shareVectorToKey(key_as_bytes, party, values, bw, qbits, tag);
}

template <typename T>
inline bool buildCShare(const MatmulParams &p,
                        int party,
                        const std::vector<T> &a,
                        const std::vector<T> &b,
                        const std::vector<T> &mask_c,
                        int qbits,
                        uint64_t tag,
                        std::vector<T> &out) {
    if (p.bw <= 2 || p.bw > 32 || (qbits != 64 && qbits != 128)) {
        return false;
    }
    u128 modulus = modulusForQbits(qbits);
    std::mt19937_64 rng(matmulSeed(p, tag));
    out.resize(static_cast<size_t>(p.size_C));

    for (int batch = 0; batch < p.batchSz; ++batch) {
        size_t a_base = static_cast<size_t>(batch) * p.stride_A;
        size_t b_base = static_cast<size_t>(batch) * p.stride_B;
        size_t c_base = static_cast<size_t>(batch) * p.stride_C;
        for (int row = 0; row < p.M; ++row) {
            for (int col = 0; col < p.N; ++col) {
                u128 dot = 0;
                for (int k = 0; k < p.K; ++k) {
                    uint64_t av = matrixValue(a, a_base, p.M, p.K, p.rowMaj_A, row, k, p.bw);
                    uint64_t bv = matrixValue(b, b_base, p.K, p.N, p.rowMaj_B, k, col, p.bw);
                    dot += u128(av) * bv;
                }
                if (dot >= modulus) {
                    return false;
                }
                size_t c_idx = c_base + static_cast<size_t>(row) * p.N + col;
                auto product_shares = shareModValueToRing(dot, p.bw, modulus, rng);
                auto mask_shares = shareRingValue(ringValue(mask_c[c_idx], p.bw), p.bw, rng);
                uint64_t selected = party == SERVER0
                                        ? ringAdd(product_shares.first, mask_shares.first, p.bw)
                                        : ringAdd(product_shares.second, mask_shares.second, p.bw);
                out[c_idx] = static_cast<T>(selected);
            }
        }
    }
    return true;
}

template <typename T>
inline bool writeMatmulCShare(u8 **key_as_bytes,
                              int party,
                              const MatmulParams &p,
                              const T *d_a,
                              const T *d_b,
                              const T *d_mask_c,
                              int qbits,
                              uint64_t tag) {
    std::vector<T> a;
    std::vector<T> b;
    std::vector<T> mask_c;
    if (!copyDeviceVector(d_a, static_cast<size_t>(p.size_A), a) ||
        !copyDeviceVector(d_b, static_cast<size_t>(p.size_B), b) ||
        !copyDeviceVector(d_mask_c, static_cast<size_t>(p.size_C), mask_c)) {
        return false;
    }
    std::vector<T> c_share;
    if (!buildCShare(p, party, a, b, mask_c, qbits, tag, c_share)) {
        return false;
    }
    appendRaw(key_as_bytes, c_share);
    return true;
}

template <typename T>
inline bool writeMatmulKey(u8 **key_as_bytes,
                           int party,
                           const MatmulParams &p,
                           const T *d_a,
                           const T *d_b,
                           const T *d_mask_c,
                           int qbits,
                           uint64_t tag) {
    std::vector<T> a;
    std::vector<T> b;
    std::vector<T> mask_c;
    if (!copyDeviceVector(d_a, static_cast<size_t>(p.size_A), a) ||
        !copyDeviceVector(d_b, static_cast<size_t>(p.size_B), b) ||
        !copyDeviceVector(d_mask_c, static_cast<size_t>(p.size_C), mask_c)) {
        return false;
    }
    std::vector<T> a_shares;
    std::vector<T> b_shares;
    std::vector<T> c_shares;
    if (!makeSharedVector(party, a, p.bw, qbits, tag + 1, a_shares) ||
        !makeSharedVector(party, b, p.bw, qbits, tag + 2, b_shares) ||
        !buildCShare(p, party, a, b, mask_c, qbits, tag + 3, c_shares)) {
        return false;
    }
    appendRaw(key_as_bytes, a_shares);
    appendRaw(key_as_bytes, b_shares);
    appendRaw(key_as_bytes, c_shares);
    return true;
}

}  // namespace ringlpn_orca
