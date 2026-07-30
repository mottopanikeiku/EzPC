// dpf_key_io.h - versioned, explicitly little-endian byte serialization for
// spfss_host::DPFKey batches, plus the TEST-ONLY input record the offline
// validator needs.
//
// Format (all integers little-endian, no padding):
//   magic     8 bytes  "RLPNDPF1"
//   version   u8       = 1
//   party     u8       0 or 1
//   count     u32      number of keys in this file
//   per key:  i32 log_domain, u64 modulus, 16B seed, u8 t0,
//             log_domain * 16B sCW, log_domain * 1B tLCW,
//             log_domain * 1B tRCW, u64 finalCW
//
// The meta file is a separate artifact and is explicitly TEST-ONLY: it holds
// each party's private protocol inputs so an offline checker can recompute
// alpha and beta. It is never read by the protocol and must never exist in a
// deployment.

#pragma once

#include "spfss_host.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace ringlpn_keyio {

constexpr char kMagic[8] = {'R', 'L', 'P', 'N', 'D', 'P', 'F', '1'};
constexpr uint8_t kVersion = 1;

struct TestInput {
    uint64_t off = 0;
    uint64_t beta_factor = 0;
};

namespace detail {

template <typename T>
inline bool put(std::FILE *f, const T &v) {
    return std::fwrite(&v, sizeof(T), 1, f) == 1;
}

template <typename T>
inline bool get(std::FILE *f, T &v) {
    return std::fread(&v, sizeof(T), 1, f) == 1;
}

inline bool put_u128(std::FILE *f, spfss_host::U128 v) {
    uint8_t buf[16];
    std::memcpy(buf, &v, 16);
    return std::fwrite(buf, 1, 16, f) == 16;
}

inline bool get_u128(std::FILE *f, spfss_host::U128 &v) {
    uint8_t buf[16];
    if (std::fread(buf, 1, 16, f) != 16) return false;
    std::memcpy(&v, buf, 16);
    return true;
}

}  // namespace detail

inline bool write_keys(const std::string &path, int party,
                       const std::vector<spfss_host::DPFKey> &keys) {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(kMagic, 1, 8, f) == 8;
    ok = ok && detail::put<uint8_t>(f, kVersion);
    ok = ok && detail::put<uint8_t>(f, (uint8_t)party);
    ok = ok && detail::put<uint32_t>(f, (uint32_t)keys.size());
    for (const spfss_host::DPFKey &K : keys) {
        if (!ok) break;
        const int L = K.log_domain;
        if (L <= 0 || L > 40 || (int)K.sCW.size() != L ||
            (int)K.tLCW.size() != L || (int)K.tRCW.size() != L) {
            ok = false;
            break;
        }
        ok = ok && detail::put<int32_t>(f, (int32_t)L);
        ok = ok && detail::put<uint64_t>(f, (uint64_t)K.modulus);
        ok = ok && detail::put_u128(f, K.seed);
        ok = ok && detail::put<uint8_t>(f, K.t0);
        for (int i = 0; i < L && ok; ++i) ok = detail::put_u128(f, K.sCW[i]);
        for (int i = 0; i < L && ok; ++i) ok = detail::put<uint8_t>(f, K.tLCW[i]);
        for (int i = 0; i < L && ok; ++i) ok = detail::put<uint8_t>(f, K.tRCW[i]);
        ok = ok && detail::put<uint64_t>(f, (uint64_t)K.finalCW);
    }
    if (std::fclose(f) != 0) ok = false;
    return ok;
}

inline bool read_keys(const std::string &path, int &party,
                      std::vector<spfss_host::DPFKey> &keys) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8] = {0};
    bool ok = std::fread(magic, 1, 8, f) == 8 &&
              std::memcmp(magic, kMagic, 8) == 0;
    uint8_t version = 0, party_byte = 0;
    uint32_t count = 0;
    ok = ok && detail::get<uint8_t>(f, version) && version == kVersion;
    ok = ok && detail::get<uint8_t>(f, party_byte) && party_byte <= 1;
    ok = ok && detail::get<uint32_t>(f, count);
    keys.clear();
    for (uint32_t k = 0; ok && k < count; ++k) {
        spfss_host::DPFKey K;
        int32_t L = 0;
        uint64_t modulus = 0, final_cw = 0;
        uint8_t t0 = 0;
        ok = ok && detail::get<int32_t>(f, L) && L > 0 && L <= 40;
        ok = ok && detail::get<uint64_t>(f, modulus) && modulus > 1;
        ok = ok && detail::get_u128(f, K.seed);
        ok = ok && detail::get<uint8_t>(f, t0) && t0 <= 1;
        if (!ok) break;
        K.log_domain = (int)L;
        K.modulus = (spfss_host::Word)modulus;
        K.t0 = t0;
        K.sCW.resize((size_t)L);
        K.tLCW.resize((size_t)L);
        K.tRCW.resize((size_t)L);
        for (int i = 0; i < L && ok; ++i) ok = detail::get_u128(f, K.sCW[i]);
        for (int i = 0; i < L && ok; ++i) ok = detail::get<uint8_t>(f, K.tLCW[i]);
        for (int i = 0; i < L && ok; ++i) ok = detail::get<uint8_t>(f, K.tRCW[i]);
        ok = ok && detail::get<uint64_t>(f, final_cw);
        if (!ok) break;
        K.finalCW = (spfss_host::Word)final_cw;
        keys.push_back(std::move(K));
    }
    // Trailing bytes mean a format mismatch.
    if (ok) {
        uint8_t extra = 0;
        if (std::fread(&extra, 1, 1, f) == 1) ok = false;
    }
    party = (int)party_byte;
    std::fclose(f);
    return ok && keys.size() == count;
}

// TEST-ONLY input record: "RLPNMETA" + u8 version + u8 party + u32 count +
// count * (u64 off, u64 beta_factor).
inline bool write_test_inputs(const std::string &path, int party,
                              const std::vector<TestInput> &inputs) {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite("RLPNMETA", 1, 8, f) == 8;
    ok = ok && detail::put<uint8_t>(f, kVersion);
    ok = ok && detail::put<uint8_t>(f, (uint8_t)party);
    ok = ok && detail::put<uint32_t>(f, (uint32_t)inputs.size());
    for (const TestInput &in : inputs) {
        ok = ok && detail::put<uint64_t>(f, in.off);
        ok = ok && detail::put<uint64_t>(f, in.beta_factor);
    }
    if (std::fclose(f) != 0) ok = false;
    return ok;
}

inline bool read_test_inputs(const std::string &path, int &party,
                             std::vector<TestInput> &inputs) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8] = {0};
    bool ok = std::fread(magic, 1, 8, f) == 8 &&
              std::memcmp(magic, "RLPNMETA", 8) == 0;
    uint8_t version = 0, party_byte = 0;
    uint32_t count = 0;
    ok = ok && detail::get<uint8_t>(f, version) && version == kVersion;
    ok = ok && detail::get<uint8_t>(f, party_byte) && party_byte <= 1;
    ok = ok && detail::get<uint32_t>(f, count);
    inputs.clear();
    for (uint32_t k = 0; ok && k < count; ++k) {
        TestInput in;
        ok = ok && detail::get<uint64_t>(f, in.off);
        ok = ok && detail::get<uint64_t>(f, in.beta_factor);
        if (ok) inputs.push_back(in);
    }
    party = (int)party_byte;
    std::fclose(f);
    return ok && inputs.size() == count;
}

}  // namespace ringlpn_keyio
