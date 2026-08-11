#pragma once

// Exact Conv0->TR->MaxPool->ReLU->Conv3 compatibility record.
//
// TRUST BOUNDARY: this record is produced by a TEST-ONLY trusted stock-key
// adapter that sees both parties' mask-state records.  It exists to exercise
// Orca's unchanged GPUMaxpoolKey and GPUReluExtendKey consumers while the clear
// value is fixed to zero.  It is not dealerless DCF generation, private-model
// execution, or evidence that either live party may reconstruct the peer's
// mask share.  Each live party receives only its own additive mask shares and
// stock key bytes; the common remask delta is the narrow compatibility seam to
// the independently generated Conv3 input mask.

#include "correlation_freshness.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace ringlpn_nonlinear_prefix {

using Digest = ringlpn_freshness::Digest;
using InvocationId = ringlpn_freshness::InvocationId;

constexpr std::array<uint8_t, 8> kRecordMagic = {
    'R', 'L', 'P', 'N', 'N', 'L', '0', '1'};
constexpr uint32_t kRecordVersion = 1;
constexpr uint32_t kTrustedStockDealerKnownZeroScope = 1;
constexpr size_t kRecordHeaderBytes = 320;
constexpr size_t kRecordDigestBytes = 32;
constexpr size_t kMaxRawKeyBytes = size_t(768) << 20;
constexpr size_t kMaxRecordBytes = size_t(1) << 30;
constexpr int kTruncatedBw = 22;
constexpr int kFullBw = 32;
constexpr uint64_t kTruncWords = 1ULL * 112 * 112 * 64;
constexpr uint64_t kNonlinearWords = 1ULL * 56 * 56 * 64;

struct Header {
    int party = -1;
    uint32_t scope = 0;
    int trunc_bw = 0;
    int full_bw = 0;
    uint64_t trunc_words = 0;
    uint64_t nonlinear_words = 0;
    uint64_t maxpool_key_bytes = 0;
    uint64_t relu_key_bytes = 0;
    InvocationId invocation{};
    Digest manifest_digest{};
    std::array<Digest, 4> linear_record_digests{};
    Digest bundle_id{};
};

struct Record {
    Header header;
    std::vector<uint64_t> trunc_next_mask_share;
    std::vector<uint64_t> maxpool_output_mask_share;
    std::vector<uint64_t> relu_output_mask_share;
    std::vector<uint64_t> remask_delta;
    std::vector<uint8_t> bytes;
    size_t raw_key_offset = 0;
    Digest digest{};

    const uint8_t *raw_key_data() const {
        return bytes.data() + raw_key_offset;
    }

    size_t raw_key_bytes() const {
        return static_cast<size_t>(header.maxpool_key_bytes +
                                   header.relu_key_bytes);
    }
};

inline void put_u32(std::vector<uint8_t> &out, size_t offset,
                    uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        out[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

inline void put_u64(std::vector<uint8_t> &out, size_t offset,
                    uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        out[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

inline uint32_t get_u32(const uint8_t *in, size_t offset) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(in[offset + i]) << (8 * i);
    }
    return value;
}

inline uint64_t get_u64(const uint8_t *in, size_t offset) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(in[offset + i]) << (8 * i);
    }
    return value;
}

inline bool nonzero(const uint8_t *data, size_t size) {
    return std::any_of(data, data + size,
                       [](uint8_t byte) { return byte != 0; });
}

inline bool checked_add(size_t &target, size_t value) {
    if (value > std::numeric_limits<size_t>::max() - target) return false;
    target += value;
    return true;
}

inline bool valid_header(const Header &header) {
    const uint64_t raw_bytes =
        header.maxpool_key_bytes + header.relu_key_bytes;
    return (header.party == 0 || header.party == 1) &&
           header.scope == kTrustedStockDealerKnownZeroScope &&
           header.trunc_bw == kTruncatedBw && header.full_bw == kFullBw &&
           header.trunc_words == kTruncWords &&
           header.nonlinear_words == kNonlinearWords &&
           header.maxpool_key_bytes != 0 && header.relu_key_bytes != 0 &&
           raw_bytes >= header.maxpool_key_bytes &&
           raw_bytes <= kMaxRawKeyBytes &&
           nonzero(header.invocation.data(), header.invocation.size()) &&
           nonzero(header.manifest_digest.data(),
                   header.manifest_digest.size()) &&
           std::all_of(header.linear_record_digests.begin(),
                       header.linear_record_digests.end(),
                       [](const Digest &digest) {
                           return nonzero(digest.data(), digest.size());
                       }) &&
           nonzero(header.bundle_id.data(), header.bundle_id.size());
}

inline std::vector<uint8_t> encode_header(const Header &header) {
    std::vector<uint8_t> out(kRecordHeaderBytes, 0);
    std::copy(kRecordMagic.begin(), kRecordMagic.end(), out.begin());
    put_u32(out, 8, kRecordVersion);
    put_u32(out, 12, static_cast<uint32_t>(header.party));
    put_u32(out, 16, header.scope);
    put_u32(out, 20, static_cast<uint32_t>(header.trunc_bw));
    put_u32(out, 24, static_cast<uint32_t>(header.full_bw));
    put_u64(out, 32, header.trunc_words);
    put_u64(out, 40, header.nonlinear_words);
    put_u64(out, 48, header.maxpool_key_bytes);
    put_u64(out, 56, header.relu_key_bytes);
    std::copy(header.invocation.begin(), header.invocation.end(),
              out.begin() + 64);
    std::copy(header.manifest_digest.begin(), header.manifest_digest.end(),
              out.begin() + 80);
    for (size_t i = 0; i < header.linear_record_digests.size(); ++i) {
        std::copy(header.linear_record_digests[i].begin(),
                  header.linear_record_digests[i].end(),
                  out.begin() + 112 + 32 * i);
    }
    std::copy(header.bundle_id.begin(), header.bundle_id.end(),
              out.begin() + 240);
    return out;
}

inline bool decode_header(const uint8_t *in, size_t size, Header &header) {
    if (size < kRecordHeaderBytes ||
        !std::equal(kRecordMagic.begin(), kRecordMagic.end(), in) ||
        get_u32(in, 8) != kRecordVersion || get_u32(in, 28) != 0 ||
        !std::all_of(in + 272, in + kRecordHeaderBytes,
                     [](uint8_t byte) { return byte == 0; })) {
        return false;
    }
    header.party = static_cast<int>(get_u32(in, 12));
    header.scope = get_u32(in, 16);
    header.trunc_bw = static_cast<int>(get_u32(in, 20));
    header.full_bw = static_cast<int>(get_u32(in, 24));
    header.trunc_words = get_u64(in, 32);
    header.nonlinear_words = get_u64(in, 40);
    header.maxpool_key_bytes = get_u64(in, 48);
    header.relu_key_bytes = get_u64(in, 56);
    std::copy(in + 64, in + 80, header.invocation.begin());
    std::copy(in + 80, in + 112, header.manifest_digest.begin());
    for (size_t i = 0; i < header.linear_record_digests.size(); ++i) {
        std::copy(in + 112 + 32 * i, in + 144 + 32 * i,
                  header.linear_record_digests[i].begin());
    }
    std::copy(in + 240, in + 272, header.bundle_id.begin());
    return valid_header(header);
}

inline bool expected_record_size(const Header &header, size_t &size,
                                 size_t &raw_key_offset) {
    if (!valid_header(header) ||
        header.trunc_words > std::numeric_limits<size_t>::max() ||
        header.nonlinear_words > std::numeric_limits<size_t>::max() ||
        header.maxpool_key_bytes > std::numeric_limits<size_t>::max() ||
        header.relu_key_bytes > std::numeric_limits<size_t>::max()) {
        return false;
    }
    size = kRecordHeaderBytes;
    const size_t trunc_bytes =
        static_cast<size_t>(header.trunc_words) * sizeof(uint64_t);
    const size_t one_nonlinear =
        static_cast<size_t>(header.nonlinear_words) * sizeof(uint64_t);
    if (!checked_add(size, trunc_bytes) ||
        !checked_add(size, 3 * one_nonlinear)) {
        return false;
    }
    raw_key_offset = size;
    if (!checked_add(size, static_cast<size_t>(header.maxpool_key_bytes)) ||
        !checked_add(size, static_cast<size_t>(header.relu_key_bytes)) ||
        !checked_add(size, kRecordDigestBytes)) {
        return false;
    }
    return size <= kMaxRecordBytes;
}

inline bool read_record(const std::string &path, Record &record) {
    std::error_code error;
    const uintmax_t file_size = std::filesystem::file_size(path, error);
    if (error || file_size < kRecordHeaderBytes + kRecordDigestBytes ||
        file_size > kMaxRecordBytes) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
    in.read(reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    if (!in || in.peek() != std::ifstream::traits_type::eof()) return false;

    Header header;
    if (!decode_header(bytes.data(), bytes.size(), header)) return false;
    size_t expected = 0;
    size_t raw_key_offset = 0;
    if (!expected_record_size(header, expected, raw_key_offset) ||
        expected != bytes.size()) {
        return false;
    }
    Digest calculated{};
    if (!ringlpn_freshness::digest(
            bytes.data(), bytes.size() - kRecordDigestBytes, calculated) ||
        !std::equal(calculated.begin(), calculated.end(),
                    bytes.end() - kRecordDigestBytes)) {
        return false;
    }

    Record parsed;
    parsed.header = header;
    parsed.raw_key_offset = raw_key_offset;
    parsed.digest = calculated;
    parsed.trunc_next_mask_share.resize(
        static_cast<size_t>(header.trunc_words));
    parsed.maxpool_output_mask_share.resize(
        static_cast<size_t>(header.nonlinear_words));
    parsed.relu_output_mask_share.resize(
        static_cast<size_t>(header.nonlinear_words));
    parsed.remask_delta.resize(static_cast<size_t>(header.nonlinear_words));
    std::vector<uint64_t> *arrays[] = {
        &parsed.trunc_next_mask_share,
        &parsed.maxpool_output_mask_share,
        &parsed.relu_output_mask_share,
        &parsed.remask_delta};
    size_t cursor = kRecordHeaderBytes;
    for (std::vector<uint64_t> *array : arrays) {
        for (uint64_t &value : *array) {
            value = get_u64(bytes.data(), cursor);
            cursor += sizeof(uint64_t);
        }
    }
    const uint64_t trunc_limit = uint64_t(1) << kTruncatedBw;
    const uint64_t full_limit = uint64_t(1) << kFullBw;
    if (cursor != raw_key_offset ||
        !std::all_of(parsed.trunc_next_mask_share.begin(),
                     parsed.trunc_next_mask_share.end(),
                     [&](uint64_t value) { return value < trunc_limit; }) ||
        !std::all_of(parsed.maxpool_output_mask_share.begin(),
                     parsed.maxpool_output_mask_share.end(),
                     [&](uint64_t value) { return value < trunc_limit; }) ||
        !std::all_of(parsed.relu_output_mask_share.begin(),
                     parsed.relu_output_mask_share.end(),
                     [&](uint64_t value) { return value < full_limit; }) ||
        !std::all_of(parsed.remask_delta.begin(),
                     parsed.remask_delta.end(),
                     [&](uint64_t value) { return value < full_limit; })) {
        return false;
    }
    parsed.bytes = std::move(bytes);
    record = std::move(parsed);
    return true;
}

inline bool headers_match(const Header &p0, const Header &p1) {
    return p0.party == 0 && p1.party == 1 && p0.scope == p1.scope &&
           p0.trunc_bw == p1.trunc_bw && p0.full_bw == p1.full_bw &&
           p0.trunc_words == p1.trunc_words &&
           p0.nonlinear_words == p1.nonlinear_words &&
           p0.maxpool_key_bytes == p1.maxpool_key_bytes &&
           p0.relu_key_bytes == p1.relu_key_bytes &&
           p0.invocation == p1.invocation &&
           p0.manifest_digest == p1.manifest_digest &&
           p0.linear_record_digests == p1.linear_record_digests &&
           p0.bundle_id == p1.bundle_id;
}

}  // namespace ringlpn_nonlinear_prefix
