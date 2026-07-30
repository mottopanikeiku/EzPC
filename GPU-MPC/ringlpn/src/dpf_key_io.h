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

// (namespace ringlpn_keyio continues below)

// ----- grouped SPFSS key file (Ring-LPN OLE consumer) -----------------------
//
// The Figure 2 Ring-LPN expansion consumes one multi-point SPFSS key per
// (limb, direction-pair, noise group). This file carries all of a single party's
// groups in the order the engine indexes them.
//
// Format: magic "RLPNSPF1", u8 version, u8 party, i32 log_domain, u64 modulus,
//         u32 group_count, then per group: u32 key_count, then each key as
//         16B seed, log_domain * (16B sCW, 1B tLCW, 1B tRCW), u64 finalCW.
// Seeds and control bits are party-private; correction words are public.

namespace spfss_groups {

inline bool write(const std::string &path, int party, int log_domain,
                  uint64_t modulus,
                  const std::vector<std::vector<spfss_host::DPFKey>> &groups) {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite("RLPNSPF1", 1, 8, f) == 8;
    ok = ok && detail::put<uint8_t>(f, kVersion);
    ok = ok && detail::put<uint8_t>(f, (uint8_t)party);
    ok = ok && detail::put<int32_t>(f, (int32_t)log_domain);
    ok = ok && detail::put<uint64_t>(f, modulus);
    ok = ok && detail::put<uint32_t>(f, (uint32_t)groups.size());
    for (const std::vector<spfss_host::DPFKey> &g : groups) {
        ok = ok && detail::put<uint32_t>(f, (uint32_t)g.size());
        for (const spfss_host::DPFKey &K : g) {
            if (!ok) break;
            if (K.log_domain != log_domain || (uint64_t)K.modulus != modulus ||
                (int)K.sCW.size() != log_domain) {
                ok = false;
                break;
            }
            ok = ok && detail::put_u128(f, K.seed);
            for (int i = 0; i < log_domain && ok; ++i) {
                ok = detail::put_u128(f, K.sCW[(size_t)i]) &&
                     detail::put<uint8_t>(f, K.tLCW[(size_t)i]) &&
                     detail::put<uint8_t>(f, K.tRCW[(size_t)i]);
            }
            ok = ok && detail::put<uint64_t>(f, (uint64_t)K.finalCW);
        }
    }
    if (std::fclose(f) != 0) ok = false;
    return ok;
}

inline bool read(const std::string &path, int &party, int &log_domain,
                 uint64_t &modulus,
                 std::vector<std::vector<spfss_host::DPFKey>> &groups) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8] = {0};
    bool ok = std::fread(magic, 1, 8, f) == 8 &&
              std::memcmp(magic, "RLPNSPF1", 8) == 0;
    uint8_t version = 0, party_byte = 0;
    int32_t levels = 0;
    uint32_t group_count = 0;
    ok = ok && detail::get<uint8_t>(f, version) && version == kVersion;
    ok = ok && detail::get<uint8_t>(f, party_byte) && party_byte <= 1;
    ok = ok && detail::get<int32_t>(f, levels) && levels > 0 && levels <= 40;
    ok = ok && detail::get<uint64_t>(f, modulus) && modulus > 1;
    ok = ok && detail::get<uint32_t>(f, group_count);
    groups.clear();
    for (uint32_t g = 0; ok && g < group_count; ++g) {
        uint32_t key_count = 0;
        ok = detail::get<uint32_t>(f, key_count);
        std::vector<spfss_host::DPFKey> group;
        for (uint32_t k = 0; ok && k < key_count; ++k) {
            spfss_host::DPFKey K;
            K.log_domain = (int)levels;
            K.modulus = (spfss_host::Word)modulus;
            K.t0 = party_byte;
            K.sCW.resize((size_t)levels);
            K.tLCW.resize((size_t)levels);
            K.tRCW.resize((size_t)levels);
            ok = detail::get_u128(f, K.seed);
            for (int i = 0; i < levels && ok; ++i) {
                ok = detail::get_u128(f, K.sCW[(size_t)i]) &&
                     detail::get<uint8_t>(f, K.tLCW[(size_t)i]) &&
                     detail::get<uint8_t>(f, K.tRCW[(size_t)i]);
            }
            uint64_t final_cw = 0;
            ok = ok && detail::get<uint64_t>(f, final_cw);
            K.finalCW = (spfss_host::Word)final_cw;
            if (ok) group.push_back(std::move(K));
        }
        if (ok) groups.push_back(std::move(group));
    }
    if (ok) {
        uint8_t extra = 0;
        if (std::fread(&extra, 1, 1, f) == 1) ok = false;  // trailing bytes
    }
    party = (int)party_byte;
    log_domain = (int)levels;
    std::fclose(f);
    return ok && groups.size() == group_count;
}

// TEST-ONLY noise record: one party's sparse noise polynomials, so the
// dealerless keygen processes can take their own private inputs from disk.
// Format: magic "RLPNNOIS", u8 version, u8 party, i32 c, i32 t, i32 log_domain,
//         u64 modulus, u8 regular, i32 bucket, then c*t (u64 position, u64 value).
struct NoiseRecord {
    int party = 0;
    int c = 0;
    int t = 0;
    int log_domain = 0;
    uint64_t modulus = 0;
    bool regular = false;
    int bucket = 0;
    std::vector<uint64_t> positions;  // c * t, poly-major
    std::vector<uint64_t> values;     // c * t, poly-major
};

inline bool write_noise(const std::string &path, const NoiseRecord &r) {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite("RLPNNOIS", 1, 8, f) == 8;
    ok = ok && detail::put<uint8_t>(f, kVersion);
    ok = ok && detail::put<uint8_t>(f, (uint8_t)r.party);
    ok = ok && detail::put<int32_t>(f, (int32_t)r.c);
    ok = ok && detail::put<int32_t>(f, (int32_t)r.t);
    ok = ok && detail::put<int32_t>(f, (int32_t)r.log_domain);
    ok = ok && detail::put<uint64_t>(f, r.modulus);
    ok = ok && detail::put<uint8_t>(f, (uint8_t)(r.regular ? 1 : 0));
    ok = ok && detail::put<int32_t>(f, (int32_t)r.bucket);
    const size_t total = (size_t)r.c * (size_t)r.t;
    if (r.positions.size() != total || r.values.size() != total) ok = false;
    for (size_t i = 0; ok && i < total; ++i) {
        ok = detail::put<uint64_t>(f, r.positions[i]) &&
             detail::put<uint64_t>(f, r.values[i]);
    }
    if (std::fclose(f) != 0) ok = false;
    return ok;
}

inline bool read_noise(const std::string &path, NoiseRecord &r) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8] = {0};
    bool ok = std::fread(magic, 1, 8, f) == 8 &&
              std::memcmp(magic, "RLPNNOIS", 8) == 0;
    uint8_t version = 0, party_byte = 0, regular = 0;
    int32_t c = 0, t = 0, levels = 0, bucket = 0;
    ok = ok && detail::get<uint8_t>(f, version) && version == kVersion;
    ok = ok && detail::get<uint8_t>(f, party_byte) && party_byte <= 1;
    ok = ok && detail::get<int32_t>(f, c) && c > 0;
    ok = ok && detail::get<int32_t>(f, t) && t > 0;
    ok = ok && detail::get<int32_t>(f, levels) && levels > 0;
    ok = ok && detail::get<uint64_t>(f, r.modulus) && r.modulus > 1;
    ok = ok && detail::get<uint8_t>(f, regular);
    ok = ok && detail::get<int32_t>(f, bucket);
    r.party = (int)party_byte;
    r.c = (int)c;
    r.t = (int)t;
    r.log_domain = (int)levels;
    r.regular = regular != 0;
    r.bucket = (int)bucket;
    const size_t total = (size_t)r.c * (size_t)r.t;
    r.positions.assign(total, 0);
    r.values.assign(total, 0);
    for (size_t i = 0; ok && i < total; ++i) {
        ok = detail::get<uint64_t>(f, r.positions[i]) &&
             detail::get<uint64_t>(f, r.values[i]);
    }
    if (ok) {
        uint8_t extra = 0;
        if (std::fread(&extra, 1, 1, f) == 1) ok = false;
    }
    std::fclose(f);
    return ok;
}

}  // namespace spfss_groups

}  // namespace ringlpn_keyio
