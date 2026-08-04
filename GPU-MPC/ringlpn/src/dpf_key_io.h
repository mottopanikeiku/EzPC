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
#include <type_traits>
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
    static_assert(std::is_integral<T>::value, "integer serialization only");
    using U = typename std::make_unsigned<T>::type;
    U x = static_cast<U>(v);
    uint8_t buf[sizeof(T)];
    for (size_t i = 0; i < sizeof(T); ++i) {
        buf[i] = static_cast<uint8_t>(x >> (8 * i));
    }
    return std::fwrite(buf, 1, sizeof(buf), f) == sizeof(buf);
}

template <typename T>
inline bool get(std::FILE *f, T &v) {
    static_assert(std::is_integral<T>::value, "integer serialization only");
    using U = typename std::make_unsigned<T>::type;
    uint8_t buf[sizeof(T)];
    if (std::fread(buf, 1, sizeof(buf), f) != sizeof(buf)) return false;
    U x = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
        x |= U(buf[i]) << (8 * i);
    }
    v = static_cast<T>(x);
    return true;
}

inline bool put_u128(std::FILE *f, spfss_host::U128 v) {
    uint8_t buf[16];
    for (int i = 0; i < 16; ++i) {
        buf[i] = static_cast<uint8_t>(v >> (8 * i));
    }
    return std::fwrite(buf, 1, sizeof(buf), f) == sizeof(buf);
}

inline bool get_u128(std::FILE *f, spfss_host::U128 &v) {
    uint8_t buf[16];
    if (std::fread(buf, 1, sizeof(buf), f) != sizeof(buf)) return false;
    v = 0;
    for (int i = 0; i < 16; ++i) {
        v |= spfss_host::U128(buf[i]) << (8 * i);
    }
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
// Format: magic "RLPNSPF2", u8 version=2, u8 party, i32 log_domain,
//         u64 modulus, u32 group_count, u32 binding_word_count, then the exact
//         canonical u64 noise-binding words, then per group: u32 key_count,
//         and each key as 16B seed, log_domain * (16B sCW, 1B tLCW,
//         1B tRCW), u64 finalCW. The binding duplicates the party-private noise
//         record inside its own private key artifact so stale/mismatched keys
//         cannot be attributed to a different record without adding a hash
//         dependency. Seeds/control bits/binding are party-private; correction
//         words are public.

namespace spfss_groups {

constexpr uint8_t kGroupedVersion = 2;
constexpr uint32_t kMaxGroupedItems = 1U << 20;
constexpr uint32_t kMaxBindingWords = (1U << 21) + 8;

inline bool write(const std::string &path, int party, int log_domain,
                  uint64_t modulus,
                  const std::vector<uint64_t> &noise_binding,
                  const std::vector<std::vector<spfss_host::DPFKey>> &groups) {
    if ((party != 0 && party != 1) || log_domain <= 0 || log_domain > 40 ||
        modulus <= 1 || groups.empty() || groups.size() > kMaxGroupedItems ||
        noise_binding.empty() || noise_binding.size() > kMaxBindingWords) {
        return false;
    }
    uint64_t total_keys = 0;
    for (const auto &g : groups) {
        if (g.empty() || g.size() > kMaxGroupedItems ||
            total_keys > kMaxGroupedItems - g.size()) {
            return false;
        }
        total_keys += g.size();
    }
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite("RLPNSPF2", 1, 8, f) == 8;
    ok = ok && detail::put<uint8_t>(f, kGroupedVersion);
    ok = ok && detail::put<uint8_t>(f, static_cast<uint8_t>(party));
    ok = ok && detail::put<int32_t>(f, static_cast<int32_t>(log_domain));
    ok = ok && detail::put<uint64_t>(f, modulus);
    ok = ok && detail::put<uint32_t>(f, static_cast<uint32_t>(groups.size()));
    ok = ok && detail::put<uint32_t>(
                   f, static_cast<uint32_t>(noise_binding.size()));
    for (uint64_t word : noise_binding) {
        ok = ok && detail::put<uint64_t>(f, word);
    }
    for (const auto &g : groups) {
        ok = ok && detail::put<uint32_t>(f, static_cast<uint32_t>(g.size()));
        for (const spfss_host::DPFKey &K : g) {
            if (!ok) break;
            if (K.t0 != static_cast<uint8_t>(party) ||
                K.log_domain != log_domain ||
                static_cast<uint64_t>(K.modulus) != modulus ||
                K.sCW.size() != static_cast<size_t>(log_domain) ||
                K.tLCW.size() != static_cast<size_t>(log_domain) ||
                K.tRCW.size() != static_cast<size_t>(log_domain) ||
                K.finalCW >= modulus) {
                ok = false;
                break;
            }
            ok = detail::put_u128(f, K.seed);
            for (int i = 0; i < log_domain && ok; ++i) {
                if (K.tLCW[static_cast<size_t>(i)] > 1 ||
                    K.tRCW[static_cast<size_t>(i)] > 1) {
                    ok = false;
                    break;
                }
                ok = detail::put_u128(f, K.sCW[static_cast<size_t>(i)]) &&
                     detail::put<uint8_t>(f, K.tLCW[static_cast<size_t>(i)]) &&
                     detail::put<uint8_t>(f, K.tRCW[static_cast<size_t>(i)]);
            }
            ok = ok && detail::put<uint64_t>(f, K.finalCW);
        }
    }
    if (std::fclose(f) != 0) ok = false;
    return ok;
}

inline bool read(const std::string &path, int &party, int &log_domain,
                 uint64_t &modulus, std::vector<uint64_t> &noise_binding,
                 std::vector<std::vector<spfss_host::DPFKey>> &groups) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8] = {0};
    bool ok = std::fread(magic, 1, 8, f) == 8 &&
              std::memcmp(magic, "RLPNSPF2", 8) == 0;
    uint8_t version = 0, party_byte = 0;
    int32_t levels = 0;
    uint32_t group_count = 0, binding_count = 0;
    ok = ok && detail::get<uint8_t>(f, version) &&
         version == kGroupedVersion;
    ok = ok && detail::get<uint8_t>(f, party_byte) && party_byte <= 1;
    ok = ok && detail::get<int32_t>(f, levels) && levels > 0 && levels <= 40;
    ok = ok && detail::get<uint64_t>(f, modulus) && modulus > 1;
    ok = ok && detail::get<uint32_t>(f, group_count) &&
         group_count > 0 && group_count <= kMaxGroupedItems;
    ok = ok && detail::get<uint32_t>(f, binding_count) &&
         binding_count > 0 && binding_count <= kMaxBindingWords;
    noise_binding.clear();
    if (ok) noise_binding.resize(binding_count);
    for (uint32_t i = 0; ok && i < binding_count; ++i) {
        ok = detail::get<uint64_t>(f, noise_binding[i]);
    }
    groups.clear();
    uint64_t total_keys = 0;
    for (uint32_t g = 0; ok && g < group_count; ++g) {
        uint32_t key_count = 0;
        ok = detail::get<uint32_t>(f, key_count) && key_count > 0 &&
             key_count <= kMaxGroupedItems &&
             total_keys <= kMaxGroupedItems - key_count;
        total_keys += key_count;
        std::vector<spfss_host::DPFKey> group;
        if (ok) group.reserve(key_count);
        for (uint32_t k = 0; ok && k < key_count; ++k) {
            spfss_host::DPFKey K;
            K.t0 = party_byte;
            K.log_domain = static_cast<int>(levels);
            K.modulus = static_cast<spfss_host::Word>(modulus);
            K.sCW.resize(static_cast<size_t>(levels));
            K.tLCW.resize(static_cast<size_t>(levels));
            K.tRCW.resize(static_cast<size_t>(levels));
            ok = detail::get_u128(f, K.seed);
            for (int i = 0; i < levels && ok; ++i) {
                ok = detail::get_u128(f, K.sCW[static_cast<size_t>(i)]) &&
                     detail::get<uint8_t>(f, K.tLCW[static_cast<size_t>(i)]) &&
                     K.tLCW[static_cast<size_t>(i)] <= 1 &&
                     detail::get<uint8_t>(f, K.tRCW[static_cast<size_t>(i)]) &&
                     K.tRCW[static_cast<size_t>(i)] <= 1;
            }
            uint64_t final_cw = 0;
            ok = ok && detail::get<uint64_t>(f, final_cw) &&
                 final_cw < modulus;
            K.finalCW = static_cast<spfss_host::Word>(final_cw);
            if (ok) group.push_back(std::move(K));
        }
        if (ok) groups.push_back(std::move(group));
    }
    if (ok) {
        uint8_t extra = 0;
        if (std::fread(&extra, 1, 1, f) == 1) ok = false;
    }
    party = static_cast<int>(party_byte);
    log_domain = static_cast<int>(levels);
    std::fclose(f);
    return ok && groups.size() == group_count &&
           noise_binding.size() == binding_count;
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
inline std::vector<uint64_t> noise_binding(const NoiseRecord &r) {
    std::vector<uint64_t> out;
    if ((r.party != 0 && r.party != 1) || r.c <= 0 || r.t <= 0 ||
        r.positions.size() != r.values.size() ||
        r.positions.size() != static_cast<size_t>(r.c) *
                                  static_cast<size_t>(r.t)) {
        return out;
    }
    out.reserve(7 + 2 * r.positions.size());
    out.push_back(static_cast<uint64_t>(r.party));
    out.push_back(static_cast<uint64_t>(r.c));
    out.push_back(static_cast<uint64_t>(r.t));
    out.push_back(static_cast<uint64_t>(r.log_domain));
    out.push_back(r.modulus);
    out.push_back(static_cast<uint64_t>(r.regular));
    out.push_back(static_cast<uint64_t>(static_cast<int64_t>(r.bucket)));
    for (size_t i = 0; i < r.positions.size(); ++i) {
        out.push_back(r.positions[i]);
        out.push_back(r.values[i]);
    }
    return out;
}

inline bool write_noise(const std::string &path, const NoiseRecord &r) {
    if ((r.party != 0 && r.party != 1) || r.c <= 0 || r.t <= 0 ||
        r.log_domain < 2 || r.log_domain > 20 || r.modulus <= 1 ||
        (r.regular ? r.bucket <= 0 : r.bucket != 0)) {
        return false;
    }
    const size_t total = size_t(r.c) * size_t(r.t);
    constexpr size_t kMaxNoiseTerms = size_t(1) << 20;
    if (total > kMaxNoiseTerms || r.positions.size() != total ||
        r.values.size() != total) {
        return false;
    }
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
    ok = ok && detail::get<int32_t>(f, levels) && levels >= 2 && levels <= 20;
    ok = ok && detail::get<uint64_t>(f, r.modulus) && r.modulus > 1;
    ok = ok && detail::get<uint8_t>(f, regular) && regular <= 1;
    ok = ok && detail::get<int32_t>(f, bucket);
    if (!ok) {
        std::fclose(f);
        return false;
    }
    r.party = (int)party_byte;
    r.c = (int)c;
    r.t = (int)t;
    r.log_domain = (int)levels;
    r.regular = regular != 0;
    r.bucket = (int)bucket;
    ok = ok && (r.regular ? r.bucket > 0 : r.bucket == 0);
    if (!ok) {
        std::fclose(f);
        return false;
    }
    const size_t total = (size_t)r.c * (size_t)r.t;
    constexpr size_t kMaxNoiseTerms = size_t(1) << 20;
    if (total > kMaxNoiseTerms) {
        std::fclose(f);
        return false;
    }
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
