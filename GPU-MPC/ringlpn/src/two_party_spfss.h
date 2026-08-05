// Party-local preparation and distributed key generation for Ring-LPN SPFSS.
// Every input and output in this API belongs to exactly one live party.
#pragma once

#include "dpf_key_io.h"
#include "two_party_dpf_protocol.h"
#include <openssl/evp.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ringlpn_spfss {

using NoiseRecord = ringlpn_keyio::spfss_groups::NoiseRecord;
using Word = ringlpn_2pc::Word;

struct SpfssPublicParams {
    int c = 0;
    int t = 0;
    int log_domain = 0;
    uint64_t modulus = 0;
    bool regular = false;
    uint64_t sid = 0;
    // Full live correlation-scope ID. `sid` is only a compact compatibility
    // handle. A zero value is accepted solely for standalone component
    // baselines; the live FC/Conv caller requires and binds a nonzero ID.
    std::array<uint8_t, 32> correlation_id{};
};

struct SpfssWork {
    uint64_t half_domain = 0;
    uint64_t ring_size = 0;
    uint64_t terms_per_party = 0;
    uint64_t tree_count = 0;
    size_t group_count = 0;
    int groups_per_pair = 0;
    int bucket = 0;
};

struct SpfssPartyBatch {
    std::vector<uint64_t> offsets;
    std::vector<Word> beta_factors;
    std::vector<size_t> group_sizes;
};

struct DpfCounters {
    uint64_t string_ots_128 = 0;
    uint64_t bit_triples = 0;
    uint64_t scalar_oles = 0;
    uint64_t logical_opened_bits = 0;
    uint64_t meaningful_share_bits = 0;
    double phase_a_microseconds = 0.0;
    double phase_b_microseconds = 0.0;
    double phase_c_microseconds = 0.0;
    double spfss_grouping_microseconds = 0.0;
    uint64_t phase_a_dependency_rounds = 0;
    uint64_t phase_b_dependency_rounds = 0;
    uint64_t phase_c_dependency_rounds = 0;
    uint64_t gpu_kernel_launches = 0;
    uint64_t gpu_h2d_bytes = 0;
    uint64_t gpu_d2h_bytes = 0;
    uint64_t gpu_peak_bytes = 0;
    uint64_t gpu_level_synchronizations = 0;
};

using SpfssPublicManifest = std::array<uint8_t, 104>;
using SpfssDigest = std::array<uint8_t, 32>;
using GroupedHostKeys = std::vector<std::vector<spfss_host::DPFKey>>;

constexpr uint64_t kMaxNoiseTerms = 1ULL << 20;
constexpr uint64_t kMaxSpfssTrees = 1ULL << 14;
constexpr uint64_t kMaxSpfssFrontierNodes = 1ULL << 24;
constexpr uint64_t kMaxSpfssArtifactBytes = 1ULL << 28;
constexpr uint64_t kMinRingDegree = 1ULL << 13;
constexpr uint64_t kMaxRingDegree = 1ULL << 20;

inline bool is_power_of_two_u64(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

inline bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t &out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

inline bool derive_spfss_work(const SpfssPublicParams &params,
                              SpfssWork &work) {
    work = SpfssWork{};
    if (params.sid == 0 || params.c <= 0 || params.t <= 0 ||
        params.log_domain < 2 || params.log_domain > 20 ||
        (params.modulus != ringlpn_2pdpf::kPrime62 &&
         params.modulus != ringlpn_2pdpf::kPrime62Crt2)) {
        return false;
    }

    const uint64_t c = static_cast<uint64_t>(params.c);
    const uint64_t t = static_cast<uint64_t>(params.t);
    uint64_t ct = 0;
    uint64_t trees = 0;
    uint64_t c_squared = 0;
    if (!checked_mul_u64(c, t, ct) || ct == 0 || ct > kMaxNoiseTerms ||
        !checked_mul_u64(ct, ct, trees) || trees == 0 ||
        !checked_mul_u64(c, c, c_squared)) {
        return false;
    }

    const uint64_t half_domain = 1ULL << (params.log_domain - 1);
    const uint64_t domain = 1ULL << params.log_domain;
    uint64_t ring_size = half_domain;
    if (params.regular &&
        (!is_power_of_two_u64(t) ||
         !checked_mul_u64(t, half_domain, ring_size))) {
        return false;
    }
    if (!is_power_of_two_u64(ring_size) ||
        ring_size < kMinRingDegree || ring_size > kMaxRingDegree ||
        t > ring_size || trees > kMaxSpfssTrees ||
        trees > kMaxSpfssFrontierNodes / domain) {
        return false;
    }

    const uint64_t key_bytes_per_tree =
        16ULL + static_cast<uint64_t>(params.log_domain) * 18ULL +
        sizeof(uint64_t);
    if (trees > kMaxSpfssArtifactBytes / (2ULL * sizeof(uint64_t)) ||
        trees > kMaxSpfssArtifactBytes / key_bytes_per_tree ||
        trees > static_cast<uint64_t>(std::numeric_limits<int>::max()) /
                    static_cast<uint64_t>(params.log_domain - 1)) {
        return false;
    }

    const uint64_t groups_per_pair = params.regular ? 2ULL * t - 1ULL : 1ULL;
    uint64_t group_count = 0;
    if (!checked_mul_u64(c_squared, groups_per_pair, group_count) ||
        group_count == 0 ||
        group_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        groups_per_pair >
            static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        half_domain >
            static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    // The noise binding is 7 header words followed by two words per term.
    if (ct > (static_cast<uint64_t>(
                  ringlpn_keyio::spfss_groups::kMaxBindingWords) - 7ULL) /
                 2ULL) {
        return false;
    }

    work.half_domain = half_domain;
    work.ring_size = ring_size;
    work.terms_per_party = ct;
    work.tree_count = trees;
    work.group_count = static_cast<size_t>(group_count);
    work.groups_per_pair = static_cast<int>(groups_per_pair);
    work.bucket = params.regular ? static_cast<int>(half_domain) : 0;
    return true;
}

inline bool validate_party_noise(const NoiseRecord &noise, int party,
                                 const SpfssPublicParams &params) {
    SpfssWork work;
    if ((party != 0 && party != 1) || !derive_spfss_work(params, work) ||
        noise.party != party || noise.c != params.c || noise.t != params.t ||
        noise.log_domain != params.log_domain ||
        noise.modulus != params.modulus || noise.regular != params.regular ||
        noise.bucket != work.bucket ||
        noise.positions.size() != static_cast<size_t>(work.terms_per_party) ||
        noise.values.size() != static_cast<size_t>(work.terms_per_party)) {
        return false;
    }

    for (int poly = 0; poly < params.c; ++poly) {
        std::unordered_set<uint64_t> seen;
        if (!params.regular) {
            const size_t reserve_count = static_cast<size_t>(params.t) * 2 + 1;
            seen.reserve(reserve_count);
        }
        for (int k = 0; k < params.t; ++k) {
            const size_t idx = static_cast<size_t>(poly) *
                                   static_cast<size_t>(params.t) +
                               static_cast<size_t>(k);
            const uint64_t position = noise.positions[idx];
            const uint64_t value = noise.values[idx];
            const bool position_ok =
                params.regular
                    ? position >= static_cast<uint64_t>(k) * work.half_domain &&
                          position <
                              static_cast<uint64_t>(k + 1) * work.half_domain
                    : position < work.ring_size && seen.insert(position).second;
            if (!position_ok || value == 0 || value >= params.modulus) {
                return false;
            }
        }
    }
    return true;
}

inline bool sample_party_noise(const SpfssPublicParams &params, int party,
                               ringlpn_2pc::PartyRandom &rng,
                               NoiseRecord &noise) {
    SpfssWork work;
    if ((party != 0 && party != 1) || !derive_spfss_work(params, work)) {
        return false;
    }

    NoiseRecord sampled;
    sampled.party = party;
    sampled.c = params.c;
    sampled.t = params.t;
    sampled.log_domain = params.log_domain;
    sampled.modulus = params.modulus;
    sampled.regular = params.regular;
    sampled.bucket = work.bucket;
    sampled.positions.reserve(static_cast<size_t>(work.terms_per_party));
    sampled.values.reserve(static_cast<size_t>(work.terms_per_party));

    for (int poly = 0; poly < params.c; ++poly) {
        std::unordered_set<uint64_t> seen;
        if (!params.regular) {
            seen.reserve(static_cast<size_t>(params.t) * 2 + 1);
        }
        for (int k = 0; k < params.t; ++k) {
            uint64_t position = 0;
            if (params.regular) {
                position = static_cast<uint64_t>(k) * work.half_domain +
                           rng.field(static_cast<Word>(work.half_domain));
            } else {
                do {
                    position = rng.field(static_cast<Word>(work.ring_size));
                } while (!seen.insert(position).second);
            }
            sampled.positions.push_back(position);
            sampled.values.push_back(
                1 + rng.field(static_cast<Word>(params.modulus - 1)));
        }
    }
    if (!validate_party_noise(sampled, party, params)) return false;
    noise = std::move(sampled);
    return true;
}

inline bool make_party_spfss_batch(int party,
                                   const SpfssPublicParams &params,
                                   const NoiseRecord &noise,
                                   SpfssPartyBatch &batch) {
    SpfssWork work;
    if (!derive_spfss_work(params, work) ||
        !validate_party_noise(noise, party, params)) {
        return false;
    }

    SpfssPartyBatch made;
    made.offsets.reserve(static_cast<size_t>(work.tree_count));
    made.beta_factors.reserve(static_cast<size_t>(work.tree_count));
    made.group_sizes.reserve(work.group_count);

    // Exact consumer order: key_idx = (i + j*c) * groups + group.
    for (int j = 0; j < params.c; ++j) {
        for (int i = 0; i < params.c; ++i) {
            const int poly = party == 0 ? i : j;
            const size_t poly_base = static_cast<size_t>(poly) *
                                     static_cast<size_t>(params.t);
            for (int group = 0; group < work.groups_per_pair; ++group) {
                size_t points = 0;
                if (params.regular) {
                    for (int k = 0; k < params.t; ++k) {
                        const int l = group - k;
                        if (l < 0 || l >= params.t) continue;
                        const int own = party == 0 ? k : l;
                        const size_t idx = poly_base + static_cast<size_t>(own);
                        made.offsets.push_back(
                            noise.positions[idx] -
                            static_cast<uint64_t>(own) * work.half_domain);
                        made.beta_factors.push_back(noise.values[idx]);
                        ++points;
                    }
                } else {
                    for (int k = 0; k < params.t; ++k) {
                        for (int l = 0; l < params.t; ++l) {
                            const int own = party == 0 ? k : l;
                            const size_t idx =
                                poly_base + static_cast<size_t>(own);
                            made.offsets.push_back(noise.positions[idx]);
                            made.beta_factors.push_back(noise.values[idx]);
                            ++points;
                        }
                    }
                }
                if (points == 0) return false;
                made.group_sizes.push_back(points);
            }
        }
    }

    if (made.offsets.size() != static_cast<size_t>(work.tree_count) ||
        made.beta_factors.size() != made.offsets.size() ||
        made.group_sizes.size() != work.group_count) {
        return false;
    }
    batch = std::move(made);
    return true;
}

inline bool encode_spfss_public_manifest(const SpfssPublicParams &params,
                                         SpfssPublicManifest &manifest) {
    SpfssWork work;
    const bool params_valid = derive_spfss_work(params, work);
    if (!params_valid) work = SpfssWork{};
    manifest.fill(0);
    size_t cursor = 0;
    auto put32 = [&](uint32_t x) {
        for (int i = 0; i < 4; ++i) manifest[cursor++] = uint8_t(x >> (8 * i));
    };
    auto put64 = [&](uint64_t x) {
        for (int i = 0; i < 8; ++i) manifest[cursor++] = uint8_t(x >> (8 * i));
    };
    put32(3);  // manifest version: full correlation scope binding
    put64(params.sid);
    for (uint8_t byte : params.correlation_id) manifest[cursor++] = byte;
    put32(static_cast<uint32_t>(params.c));
    put32(static_cast<uint32_t>(params.t));
    put32(static_cast<uint32_t>(params.log_domain));
    put64(params.modulus);
    put32(static_cast<uint32_t>(params.regular));
    put32(static_cast<uint32_t>(work.bucket));
    put64(work.tree_count);
    put64(static_cast<uint64_t>(work.group_count));
    put64(work.terms_per_party);
    put64(0x5350465353505542ULL);  // "SPFSSPUB"
    return cursor == manifest.size();
}

inline bool agree_spfss_public_manifest(
    ringlpn_2pc::PartyChannel &channel, const SpfssPublicParams &params,
    bool local_valid = true) {
    SpfssPublicManifest mine;
    SpfssPublicManifest theirs;
    SpfssWork work;
    const bool params_valid = derive_spfss_work(params, work);
    const bool encoded = encode_spfss_public_manifest(params, mine);
    // Encoding is total for parsed fixed-width fields, so even a locally
    // unsupported work set reaches the same exchange and both peers abort.
    if (!encoded) mine.fill(0);
    channel.exchange_bytes(mine.data(), theirs.data(), mine.size());
    const uint8_t mine_valid =
        (params_valid && local_valid && encoded) ? 1 : 0;
    uint8_t theirs_valid = 0;
    channel.exchange_bytes(&mine_valid, &theirs_valid, 1);
    return std::memcmp(mine.data(), theirs.data(), mine.size()) == 0 &&
           mine_valid == 1 && theirs_valid == 1;
}

namespace detail {

class Sha256State {
  public:
    Sha256State() : ctx_(EVP_MD_CTX_new()) {
        ok_ = ctx_ != nullptr &&
              EVP_DigestInit_ex(ctx_, EVP_sha256(), nullptr) == 1;
    }
    ~Sha256State() { EVP_MD_CTX_free(ctx_); }
    Sha256State(const Sha256State &) = delete;
    Sha256State &operator=(const Sha256State &) = delete;

    bool update(const void *data, size_t size) {
        if (!ok_ || EVP_DigestUpdate(ctx_, data, size) != 1) {
            ok_ = false;
        }
        return ok_;
    }
    bool put_u32(uint32_t value) {
        uint8_t bytes[4];
        for (int i = 0; i < 4; ++i) bytes[i] = uint8_t(value >> (8 * i));
        return update(bytes, sizeof(bytes));
    }
    bool put_u64(uint64_t value) {
        uint8_t bytes[8];
        for (int i = 0; i < 8; ++i) bytes[i] = uint8_t(value >> (8 * i));
        return update(bytes, sizeof(bytes));
    }
    bool put_u128(spfss_host::U128 value) {
        uint8_t bytes[16];
        for (int i = 0; i < 16; ++i) bytes[i] = uint8_t(value >> (8 * i));
        return update(bytes, sizeof(bytes));
    }
    bool finish(SpfssDigest &digest) {
        unsigned int size = 0;
        if (!ok_ || EVP_DigestFinal_ex(ctx_, digest.data(), &size) != 1 ||
            size != digest.size()) {
            digest.fill(0);
            return false;
        }
        return true;
    }

  private:
    EVP_MD_CTX *ctx_ = nullptr;
    bool ok_ = false;
};

}  // namespace detail

inline bool digest_spfss_public_manifest(const SpfssPublicParams &params,
                                         SpfssDigest &digest) {
    digest.fill(0);
    SpfssWork work;
    SpfssPublicManifest manifest;
    if (!derive_spfss_work(params, work) ||
        !encode_spfss_public_manifest(params, manifest)) {
        return false;
    }
    unsigned int size = 0;
    if (EVP_Digest(manifest.data(), manifest.size(), digest.data(), &size,
                   EVP_sha256(), nullptr) != 1 ||
        size != digest.size()) {
        digest.fill(0);
        return false;
    }
    return true;
}

inline bool digest_party_noise(const NoiseRecord &noise, int party,
                               const SpfssPublicParams &params,
                               SpfssDigest &digest) {
    digest.fill(0);
    if (!validate_party_noise(noise, party, params)) return false;
    detail::Sha256State hash;
    constexpr char domain[] = "RINGLPN-SPFSS-LOCAL-NOISE-V1";
    if (!hash.update(domain, sizeof(domain) - 1) ||
        !hash.put_u32(static_cast<uint32_t>(noise.party)) ||
        !hash.put_u32(static_cast<uint32_t>(noise.c)) ||
        !hash.put_u32(static_cast<uint32_t>(noise.t)) ||
        !hash.put_u32(static_cast<uint32_t>(noise.log_domain)) ||
        !hash.put_u64(noise.modulus) ||
        !hash.put_u32(static_cast<uint32_t>(noise.regular)) ||
        !hash.put_u32(static_cast<uint32_t>(noise.bucket))) {
        return false;
    }
    for (size_t i = 0; i < noise.positions.size(); ++i) {
        if (!hash.put_u64(noise.positions[i]) ||
            !hash.put_u64(noise.values[i])) {
            return false;
        }
    }
    return hash.finish(digest);
}

inline bool digest_noise_content(const NoiseRecord &noise, int party,
                                 const SpfssPublicParams &params,
                                 SpfssDigest &digest) {
    digest.fill(0);
    if (!validate_party_noise(noise, party, params)) return false;
    detail::Sha256State hash;
    constexpr char domain[] = "RINGLPN-SPFSS-NOISE-CONTENT-V1";
    if (!hash.update(domain, sizeof(domain) - 1) ||
        !hash.put_u32(static_cast<uint32_t>(noise.c)) ||
        !hash.put_u32(static_cast<uint32_t>(noise.t)) ||
        !hash.put_u32(static_cast<uint32_t>(noise.log_domain)) ||
        !hash.put_u64(noise.modulus) ||
        !hash.put_u32(static_cast<uint32_t>(noise.regular)) ||
        !hash.put_u32(static_cast<uint32_t>(noise.bucket))) {
        return false;
    }
    for (size_t i = 0; i < noise.positions.size(); ++i) {
        if (!hash.put_u64(noise.positions[i]) ||
            !hash.put_u64(noise.values[i])) {
            return false;
        }
    }
    return hash.finish(digest);
}

inline bool digest_common_cw_transcript(
    int party, const SpfssPublicParams &params,
    const GroupedHostKeys &grouped_keys, SpfssDigest &digest) {
    digest.fill(0);
    SpfssWork work;
    SpfssPublicManifest manifest;
    if ((party != 0 && party != 1) || !derive_spfss_work(params, work) ||
        !encode_spfss_public_manifest(params, manifest) ||
        grouped_keys.size() != work.group_count) {
        return false;
    }

    detail::Sha256State hash;
    constexpr char domain[] = "RINGLPN-SPFSS-COMMON-CW-V1";
    if (!hash.update(domain, sizeof(domain) - 1) ||
        !hash.update(manifest.data(), manifest.size()) ||
        !hash.put_u64(static_cast<uint64_t>(grouped_keys.size()))) {
        return false;
    }
    uint64_t total_keys = 0;
    for (size_t group_index = 0; group_index < grouped_keys.size();
         ++group_index) {
        const int group =
            static_cast<int>(group_index %
                             static_cast<size_t>(work.groups_per_pair));
        const size_t expected_size =
            params.regular
                ? static_cast<size_t>(group < params.t
                                          ? group + 1
                                          : 2 * params.t - 1 - group)
                : static_cast<size_t>(params.t) *
                      static_cast<size_t>(params.t);
        const auto &keys = grouped_keys[group_index];
        if (keys.size() != expected_size ||
            keys.size() > work.tree_count - total_keys ||
            !hash.put_u64(static_cast<uint64_t>(keys.size()))) {
            return false;
        }
        total_keys += keys.size();
        for (const spfss_host::DPFKey &key : keys) {
            if (key.t0 != static_cast<uint8_t>(party) ||
                key.log_domain != params.log_domain ||
                static_cast<uint64_t>(key.modulus) != params.modulus ||
                key.sCW.size() != static_cast<size_t>(params.log_domain) ||
                key.tLCW.size() != static_cast<size_t>(params.log_domain) ||
                key.tRCW.size() != static_cast<size_t>(params.log_domain) ||
                key.finalCW >= params.modulus) {
                return false;
            }
            for (int level = 0; level < params.log_domain; ++level) {
                const size_t index = static_cast<size_t>(level);
                if (key.tLCW[index] > 1 || key.tRCW[index] > 1 ||
                    !hash.put_u128(key.sCW[index]) ||
                    !hash.update(&key.tLCW[index], 1) ||
                    !hash.update(&key.tRCW[index], 1)) {
                    return false;
                }
            }
            if (!hash.put_u64(key.finalCW)) return false;
        }
    }
    return total_keys == work.tree_count && hash.finish(digest);
}

inline bool validate_party_batch(const SpfssPublicParams &params,
                                 const SpfssPartyBatch &batch) {
    SpfssWork work;
    if (!derive_spfss_work(params, work) ||
        batch.offsets.size() != static_cast<size_t>(work.tree_count) ||
        batch.beta_factors.size() != batch.offsets.size() ||
        batch.group_sizes.size() != work.group_count) {
        return false;
    }
    size_t grouped_total = 0;
    for (size_t group_size : batch.group_sizes) {
        if (group_size == 0 || group_size > batch.offsets.size() - grouped_total) {
            return false;
        }
        grouped_total += group_size;
    }
    if (grouped_total != batch.offsets.size()) return false;
    for (size_t i = 0; i < batch.offsets.size(); ++i) {
        if (batch.offsets[i] >= work.half_domain ||
            batch.beta_factors[i] == 0 ||
            batch.beta_factors[i] >= params.modulus) {
            return false;
        }
    }
    return true;
}

inline bool record_dpf_generation_counters(
    const ringlpn_2pc::Counters &before,
    const ringlpn_2pc::Counters &after,
    const ringlpn_2pdpf::DpfStageCounters &stages, DpfCounters &counters) {
    const auto monotonic = [](uint64_t first, uint64_t last) {
        return last >= first;
    };
    if (!monotonic(before.string_ots_128, after.string_ots_128) ||
        !monotonic(before.triple_ots, after.triple_ots) ||
        !monotonic(before.ole_ots, after.ole_ots) ||
        !monotonic(before.bit_triples, after.bit_triples) ||
        !monotonic(before.scalar_oles, after.scalar_oles) ||
        !monotonic(before.base_ots, after.base_ots)) {
        return false;
    }
    uint64_t logical_opened_bits = 0;
    uint64_t meaningful_share_bits = 0;
    const ringlpn_2pc::PhaseCosts *before_phases[] = {
        &before.phase_a, &before.phase_b, &before.phase_c};
    const ringlpn_2pc::PhaseCosts *after_phases[] = {
        &after.phase_a, &after.phase_b, &after.phase_c};
    for (size_t phase = 0; phase < 3; ++phase) {
        const ringlpn_2pc::PhaseCosts &first = *before_phases[phase];
        const ringlpn_2pc::PhaseCosts &last = *after_phases[phase];
        if (!monotonic(first.logical_bits, last.logical_bits) ||
            !monotonic(first.revealed_bits_sent, last.revealed_bits_sent) ||
            !monotonic(first.revealed_bits_recv, last.revealed_bits_recv)) {
            return false;
        }
        const uint64_t logical_delta = last.logical_bits - first.logical_bits;
        const uint64_t sent_delta =
            last.revealed_bits_sent - first.revealed_bits_sent;
        const uint64_t recv_delta =
            last.revealed_bits_recv - first.revealed_bits_recv;
        if (logical_opened_bits >
                std::numeric_limits<uint64_t>::max() - logical_delta ||
            meaningful_share_bits >
                std::numeric_limits<uint64_t>::max() - sent_delta) {
            return false;
        }
        logical_opened_bits += logical_delta;
        meaningful_share_bits += sent_delta;
        if (meaningful_share_bits >
            std::numeric_limits<uint64_t>::max() - recv_delta) {
            return false;
        }
        meaningful_share_bits += recv_delta;
    }
    counters.string_ots_128 = after.string_ots_128 - before.string_ots_128;
    counters.bit_triples = after.bit_triples - before.bit_triples;
    counters.scalar_oles = after.scalar_oles - before.scalar_oles;
    counters.logical_opened_bits = logical_opened_bits;
    counters.meaningful_share_bits = meaningful_share_bits;
    counters.phase_a_microseconds = stages.phase_a_microseconds;
    counters.phase_b_microseconds = stages.phase_b_microseconds;
    counters.phase_c_microseconds = stages.phase_c_microseconds;
    counters.phase_a_dependency_rounds = stages.phase_a_dependency_rounds;
    counters.phase_b_dependency_rounds = stages.phase_b_dependency_rounds;
    counters.phase_c_dependency_rounds = stages.phase_c_dependency_rounds;
    counters.gpu_kernel_launches = stages.gpu_kernel_launches;
    counters.gpu_h2d_bytes = stages.gpu_h2d_bytes;
    counters.gpu_d2h_bytes = stages.gpu_d2h_bytes;
    counters.gpu_peak_bytes = stages.gpu_peak_bytes;
    counters.gpu_level_synchronizations = stages.level_synchronizations;
    return true;
}

inline bool group_party_dpf_keys(const SpfssPartyBatch &batch,
                                 std::vector<spfss_host::DPFKey> &flat_keys,
                                 GroupedHostKeys &grouped_keys,
                                 DpfCounters &counters) {
    const auto grouping_start = std::chrono::steady_clock::now();
    if (flat_keys.size() != batch.offsets.size()) return false;
    GroupedHostKeys grouped;
    grouped.reserve(batch.group_sizes.size());
    size_t cursor = 0;
    for (size_t group_size : batch.group_sizes) {
        if (group_size > flat_keys.size() - cursor) return false;
        auto begin = flat_keys.begin() + static_cast<std::ptrdiff_t>(cursor);
        auto end = begin + static_cast<std::ptrdiff_t>(group_size);
        grouped.emplace_back(std::make_move_iterator(begin),
                             std::make_move_iterator(end));
        cursor += group_size;
    }
    if (cursor != flat_keys.size()) return false;
    grouped_keys = std::move(grouped);
    counters.spfss_grouping_microseconds =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - grouping_start)
            .count();
    return true;
}

// Explicit host-only comparison mode. The live publication path calls the
// GPU-batched entry point in two_party_spfss_gpu.cuh instead.
inline bool generate_party_spfss_keys_cpu_baseline(
    int party, const SpfssPublicParams &params,
    const SpfssPartyBatch &batch, ringlpn_2pc::PartyChannel &channel,
    ringlpn_2pc::PartyRandom &rng, GroupedHostKeys &grouped_keys,
    DpfCounters &counters) {
    counters = DpfCounters{};
    grouped_keys.clear();
    if ((party != 0 && party != 1) || !validate_party_batch(params, batch)) {
        return false;
    }
    const ringlpn_2pc::Counters before = channel.costs;
    std::vector<spfss_host::DPFKey> flat_keys;
    ringlpn_2pdpf::DpfStageCounters stages;
    const bool generated =
        ringlpn_2pdpf::two_party_dpf_gen_batch_cpu_baseline(
            party, params.log_domain, static_cast<Word>(params.modulus),
            ringlpn_2pdpf::PrgMode::kGpuAes, batch.offsets,
            batch.beta_factors, channel, rng, flat_keys, &stages);
    const ringlpn_2pc::Counters after = channel.costs;
    const bool counters_ok =
        record_dpf_generation_counters(before, after, stages, counters);
    return generated && counters_ok &&
           group_party_dpf_keys(batch, flat_keys, grouped_keys, counters);
}

inline bool party_noise_binding(const NoiseRecord &noise, int party,
                                const SpfssPublicParams &params,
                                std::vector<uint64_t> &binding) {
    if (!validate_party_noise(noise, party, params)) return false;
    std::vector<uint64_t> made =
        ringlpn_keyio::spfss_groups::noise_binding(noise);
    if (made.empty()) return false;
    binding = std::move(made);
    return true;
}

}  // namespace ringlpn_spfss
