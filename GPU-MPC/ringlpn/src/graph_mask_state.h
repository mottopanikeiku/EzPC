#pragma once

#include "correlation_freshness.h"
#include "private_file.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <iterator>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

namespace ringlpn_graph {

constexpr std::array<uint8_t, 8> kMaskStateMagic = {
    'R', 'L', 'P', 'G', 'S', 'T', '0', '1'};
constexpr uint32_t kMaskStateVersion = 1;
constexpr size_t kMaskStateHeaderBytes = 160;
constexpr size_t kMaskStateDigestBytes = 32;
constexpr size_t kMaxMaskStateBytes = size_t(1) << 30;

struct MaskStateHeader {
    int party = -1;
    uint64_t sid = 0;
    uint64_t layer_ordinal = 0;
    int input_bw = 0;
    int output_bw = 0;
    uint64_t input_words = 0;
    uint64_t output_words = 0;
    ringlpn_freshness::InvocationId invocation_id{};
    ringlpn_freshness::Digest layer_identity{};
    ringlpn_freshness::Digest linear_record_digest{};
};

template <typename Word>
struct MaskStateRecord {
    static_assert(std::is_unsigned_v<Word> && sizeof(Word) == sizeof(uint64_t));
    MaskStateHeader header;
    std::vector<Word> input_mask_share;
    std::vector<Word> output_mask_share;
    ringlpn_freshness::Digest digest{};
};

struct ArtifactBinding {
    std::filesystem::path path;
    uint64_t bytes = 0;
    ringlpn_freshness::Digest sha256{};
    bool present = false;
};

struct LinearArtifactBindings {
    ArtifactBinding p0_record;
    ArtifactBinding p1_record;
    ArtifactBinding p0_state;
    ArtifactBinding p1_state;
};

inline bool parse_artifact_u64(const std::string &text, uint64_t &out) {
    if (text.empty()) return false;
    uint64_t value = 0;
    for (char character : text) {
        if (character < '0' || character > '9') return false;
        const uint64_t digit = static_cast<uint64_t>(character - '0');
        if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
    }
    out = value;
    return true;
}

inline bool parse_artifact_digest(const std::string &text,
                                  ringlpn_freshness::Digest &out) {
    if (text.size() != 2 * out.size()) return false;
    auto nibble = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        return -1;
    };
    for (size_t i = 0; i < out.size(); ++i) {
        const int high = nibble(text[2 * i]);
        const int low = nibble(text[2 * i + 1]);
        if (high < 0 || low < 0) return false;
        out[i] = static_cast<uint8_t>((high << 4) | low);
    }
    return true;
}

inline ArtifactBinding *artifact_binding(
    LinearArtifactBindings &bindings, const std::string &label) {
    if (label == "p0_record") return &bindings.p0_record;
    if (label == "p1_record") return &bindings.p1_record;
    if (label == "p0_state") return &bindings.p0_state;
    if (label == "p1_state") return &bindings.p1_state;
    return nullptr;
}

inline const ArtifactBinding &party_record_binding(
    const LinearArtifactBindings &bindings, int party) {
    return party == 0 ? bindings.p0_record : bindings.p1_record;
}

inline const ArtifactBinding &party_state_binding(
    const LinearArtifactBindings &bindings, int party) {
    return party == 0 ? bindings.p0_state : bindings.p1_state;
}

inline bool set_artifact_binding(LinearArtifactBindings &bindings,
                                 const std::string &label,
                                 const std::string &path,
                                 const std::string &bytes,
                                 const std::string &sha256) {
    ArtifactBinding *binding = artifact_binding(bindings, label);
    if (binding == nullptr || binding->present ||
        !parse_artifact_u64(bytes, binding->bytes) || binding->bytes == 0 ||
        !parse_artifact_digest(sha256, binding->sha256)) {
        return false;
    }
    binding->path = path;
    binding->present = binding->path.is_absolute() &&
                       std::any_of(binding->sha256.begin(),
                                   binding->sha256.end(),
                                   [](uint8_t byte) { return byte != 0; });
    return binding->present;
}

inline bool artifact_bindings_complete(
    const LinearArtifactBindings &bindings) {
    return bindings.p0_record.present && bindings.p1_record.present &&
           bindings.p0_state.present && bindings.p1_state.present;
}

inline int open_regular_nofollow(const std::filesystem::path &path) {
    if (path.empty() || path.filename().empty()) return -1;
    int directory =
        ::open(path.is_absolute() ? "/" : ".",
               O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory < 0) return -1;
    const std::filesystem::path relative =
        path.is_absolute() ? path.relative_path() : path;
    size_t component_index = 0;
    const size_t component_count =
        static_cast<size_t>(std::distance(relative.begin(), relative.end()));
    for (const std::filesystem::path &part : relative) {
        ++component_index;
        const std::string component = part.string();
        if (component.empty() || component == "." || component == "..") {
            ::close(directory);
            return -1;
        }
        const bool last = component_index == component_count;
        const int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW |
                          (last ? 0 : O_DIRECTORY);
        const int next = ::openat(directory, component.c_str(), flags);
        ::close(directory);
        if (next < 0) return -1;
        directory = next;
    }
    return directory;
}

inline void put_u32(std::vector<uint8_t> &out, size_t offset, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        out[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

inline void put_u64(std::vector<uint8_t> &out, size_t offset, uint64_t value) {
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

inline bool valid_mask_state_header(const MaskStateHeader &header) {
    const bool invocation_nonzero =
        std::any_of(header.invocation_id.begin(), header.invocation_id.end(),
                    [](uint8_t byte) { return byte != 0; });
    const bool identity_nonzero =
        std::any_of(header.layer_identity.begin(), header.layer_identity.end(),
                    [](uint8_t byte) { return byte != 0; });
    const bool record_nonzero = std::any_of(
        header.linear_record_digest.begin(), header.linear_record_digest.end(),
        [](uint8_t byte) { return byte != 0; });
    return (header.party == 0 || header.party == 1) && header.sid != 0 &&
           header.layer_ordinal != 0 && header.input_bw > 2 &&
           header.input_bw <= 32 && header.output_bw > 2 &&
           header.output_bw <= 32 && header.input_words != 0 &&
           header.output_words != 0 && invocation_nonzero && identity_nonzero &&
           record_nonzero;
}

inline std::vector<uint8_t> encode_mask_state_header(
    const MaskStateHeader &header) {
    std::vector<uint8_t> out(kMaskStateHeaderBytes, 0);
    std::copy(kMaskStateMagic.begin(), kMaskStateMagic.end(), out.begin());
    put_u32(out, 8, kMaskStateVersion);
    put_u32(out, 12, static_cast<uint32_t>(header.party));
    put_u64(out, 16, header.sid);
    put_u64(out, 24, header.layer_ordinal);
    put_u32(out, 32, static_cast<uint32_t>(header.input_bw));
    put_u32(out, 36, static_cast<uint32_t>(header.output_bw));
    put_u64(out, 40, header.input_words);
    put_u64(out, 48, header.output_words);
    std::copy(header.invocation_id.begin(), header.invocation_id.end(),
              out.begin() + 56);
    std::copy(header.layer_identity.begin(), header.layer_identity.end(),
              out.begin() + 72);
    std::copy(header.linear_record_digest.begin(),
              header.linear_record_digest.end(), out.begin() + 104);
    return out;
}

inline bool decode_mask_state_header(const uint8_t *in, size_t size,
                                     MaskStateHeader &header) {
    if (size < kMaskStateHeaderBytes ||
        !std::equal(kMaskStateMagic.begin(), kMaskStateMagic.end(), in) ||
        get_u32(in, 8) != kMaskStateVersion ||
        !std::all_of(in + 136, in + kMaskStateHeaderBytes,
                     [](uint8_t byte) { return byte == 0; })) {
        return false;
    }
    header.party = static_cast<int>(get_u32(in, 12));
    header.sid = get_u64(in, 16);
    header.layer_ordinal = get_u64(in, 24);
    header.input_bw = static_cast<int>(get_u32(in, 32));
    header.output_bw = static_cast<int>(get_u32(in, 36));
    header.input_words = get_u64(in, 40);
    header.output_words = get_u64(in, 48);
    std::copy(in + 56, in + 72, header.invocation_id.begin());
    std::copy(in + 72, in + 104, header.layer_identity.begin());
    std::copy(in + 104, in + 136, header.linear_record_digest.begin());
    return valid_mask_state_header(header);
}

template <typename Word>
bool serialize_mask_state(const MaskStateHeader &header,
                          const std::vector<Word> &input_mask_share,
                          const std::vector<Word> &output_mask_share,
                          std::vector<uint8_t> &bytes,
                          ringlpn_freshness::Digest &digest) {
    static_assert(std::is_unsigned_v<Word> && sizeof(Word) == sizeof(uint64_t));
    if (!valid_mask_state_header(header) ||
        header.input_words != input_mask_share.size() ||
        header.output_words != output_mask_share.size() ||
        input_mask_share.size() >
            (kMaxMaskStateBytes - kMaskStateHeaderBytes -
             kMaskStateDigestBytes) /
                sizeof(Word) ||
        output_mask_share.size() >
            (kMaxMaskStateBytes - kMaskStateHeaderBytes -
             kMaskStateDigestBytes - input_mask_share.size() * sizeof(Word)) /
                sizeof(Word)) {
        return false;
    }
    const uint64_t input_limit = uint64_t(1) << header.input_bw;
    const uint64_t output_limit = uint64_t(1) << header.output_bw;
    if (!std::all_of(input_mask_share.begin(), input_mask_share.end(),
                     [&](Word value) {
                         return static_cast<uint64_t>(value) < input_limit;
                     }) ||
        !std::all_of(output_mask_share.begin(), output_mask_share.end(),
                     [&](Word value) {
                         return static_cast<uint64_t>(value) < output_limit;
                     })) {
        return false;
    }

    bytes = encode_mask_state_header(header);
    bytes.resize(kMaskStateHeaderBytes +
                 (input_mask_share.size() + output_mask_share.size()) *
                     sizeof(Word));
    size_t cursor = kMaskStateHeaderBytes;
    for (Word value : input_mask_share) {
        put_u64(bytes, cursor, static_cast<uint64_t>(value));
        cursor += sizeof(Word);
    }
    for (Word value : output_mask_share) {
        put_u64(bytes, cursor, static_cast<uint64_t>(value));
        cursor += sizeof(Word);
    }
    if (!ringlpn_freshness::digest(bytes.data(), bytes.size(), digest)) {
        return false;
    }
    bytes.insert(bytes.end(), digest.begin(), digest.end());
    return true;
}

inline bool write_mask_state_bytes(const std::string &path,
                                   const std::vector<uint8_t> &bytes) {
    return ringlpn_private_file::write_atomic(path, bytes);
}

inline bool read_regular_bytes_once(const std::filesystem::path &path,
                                    uint64_t expected_bytes,
                                    bool enforce_expected,
                                    std::vector<uint8_t> &bytes) {
    const int descriptor = open_regular_nofollow(path);
    if (descriptor < 0) return false;
    struct stat metadata {};
    bool ok = ::fstat(descriptor, &metadata) == 0 &&
              S_ISREG(metadata.st_mode) && metadata.st_size >= 0;
    const uint64_t file_bytes =
        ok ? static_cast<uint64_t>(metadata.st_size) : uint64_t(0);
    ok = ok && file_bytes >= kMaskStateHeaderBytes + kMaskStateDigestBytes &&
         file_bytes <= kMaxMaskStateBytes &&
         file_bytes <= std::numeric_limits<size_t>::max() &&
         (!enforce_expected || file_bytes == expected_bytes);
    bytes.assign(ok ? static_cast<size_t>(file_bytes) : size_t(0), 0);
    size_t cursor = 0;
    while (ok && cursor < bytes.size()) {
        const ssize_t count =
            ::read(descriptor, bytes.data() + cursor, bytes.size() - cursor);
        if (count > 0) {
            cursor += static_cast<size_t>(count);
        } else if (count < 0 && errno == EINTR) {
            continue;
        } else {
            ok = false;
        }
    }
    uint8_t extra = 0;
    while (ok) {
        const ssize_t count = ::read(descriptor, &extra, 1);
        if (count == 0) break;
        if (count < 0 && errno == EINTR) continue;
        ok = false;
    }
    ::close(descriptor);
    if (!ok) bytes.clear();
    return ok;
}

template <typename Word>
bool parse_mask_state_bytes(const std::vector<uint8_t> &bytes,
                            MaskStateRecord<Word> &record) {
    static_assert(std::is_unsigned_v<Word> && sizeof(Word) == sizeof(uint64_t));
    MaskStateHeader header;
    if (!decode_mask_state_header(bytes.data(), bytes.size(), header)) {
        return false;
    }
    if (header.input_words > std::numeric_limits<size_t>::max() ||
        header.output_words > std::numeric_limits<size_t>::max() ||
        header.input_words >
            (kMaxMaskStateBytes - kMaskStateHeaderBytes -
             kMaskStateDigestBytes) /
                sizeof(Word) ||
        header.output_words >
            (kMaxMaskStateBytes - kMaskStateHeaderBytes -
             kMaskStateDigestBytes -
             static_cast<size_t>(header.input_words) * sizeof(Word)) /
                sizeof(Word)) {
        return false;
    }
    const size_t input_words = static_cast<size_t>(header.input_words);
    const size_t output_words = static_cast<size_t>(header.output_words);
    const size_t payload_bytes = (input_words + output_words) * sizeof(Word);
    if (kMaskStateHeaderBytes + payload_bytes + kMaskStateDigestBytes !=
        bytes.size()) {
        return false;
    }
    ringlpn_freshness::Digest expected{};
    if (!ringlpn_freshness::digest(
            bytes.data(), kMaskStateHeaderBytes + payload_bytes, expected) ||
        !std::equal(expected.begin(), expected.end(),
                    bytes.begin() + kMaskStateHeaderBytes + payload_bytes)) {
        return false;
    }

    MaskStateRecord<Word> parsed;
    parsed.header = header;
    parsed.input_mask_share.resize(input_words);
    parsed.output_mask_share.resize(output_words);
    size_t cursor = kMaskStateHeaderBytes;
    for (Word &value : parsed.input_mask_share) {
        value = static_cast<Word>(get_u64(bytes.data(), cursor));
        cursor += sizeof(Word);
    }
    for (Word &value : parsed.output_mask_share) {
        value = static_cast<Word>(get_u64(bytes.data(), cursor));
        cursor += sizeof(Word);
    }
    const uint64_t input_limit = uint64_t(1) << header.input_bw;
    const uint64_t output_limit = uint64_t(1) << header.output_bw;
    if (!std::all_of(parsed.input_mask_share.begin(),
                     parsed.input_mask_share.end(), [&](Word value) {
                         return static_cast<uint64_t>(value) < input_limit;
                     }) ||
        !std::all_of(parsed.output_mask_share.begin(),
                     parsed.output_mask_share.end(), [&](Word value) {
                         return static_cast<uint64_t>(value) < output_limit;
                     })) {
        return false;
    }
    parsed.digest = expected;
    record = std::move(parsed);
    return true;
}

template <typename Word>
bool read_mask_state(const ArtifactBinding &binding,
                     MaskStateRecord<Word> &record) {
    if (!binding.present ||
        binding.bytes < kMaskStateHeaderBytes + kMaskStateDigestBytes ||
        binding.bytes > kMaxMaskStateBytes) {
        return false;
    }
    std::vector<uint8_t> bytes;
    ringlpn_freshness::Digest calculated{};
    return read_regular_bytes_once(binding.path, binding.bytes, true, bytes) &&
           ringlpn_freshness::digest(bytes.data(), bytes.size(), calculated) &&
           calculated == binding.sha256 &&
           parse_mask_state_bytes(bytes, record);
}

template <typename Word>
bool read_mask_state(const std::string &path, MaskStateRecord<Word> &record) {
    std::vector<uint8_t> bytes;
    return read_regular_bytes_once(path, 0, false, bytes) &&
           parse_mask_state_bytes(bytes, record);
}

inline bool mask_state_headers_match(const MaskStateHeader &p0,
                                     const MaskStateHeader &p1) {
    return p0.party == 0 && p1.party == 1 && p0.sid == p1.sid &&
           p0.layer_ordinal == p1.layer_ordinal &&
           p0.input_bw == p1.input_bw && p0.output_bw == p1.output_bw &&
           p0.input_words == p1.input_words &&
           p0.output_words == p1.output_words &&
           p0.invocation_id == p1.invocation_id &&
           p0.layer_identity == p1.layer_identity;
}

}  // namespace ringlpn_graph
