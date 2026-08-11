#pragma once

// Versioned, digest-bound compatibility record for the exact full ResNet18
// known-zero graph checkpoint.  The raw stock key material is produced by a
// TEST-ONLY trusted adapter that reads both parties' mask states.  Linear keys
// and live secure-truncation correlations remain external and are bound here by
// ordered digests and state arrays.  This format does not imply dealerless DCF
// generation or a private/trained inference claim.

#include "correlation_freshness.h"
#include "resnet18_graph_contract.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace ringlpn_full_graph_record {

using Digest = ringlpn_freshness::Digest;
using InvocationId = ringlpn_freshness::InvocationId;
namespace contract = ringlpn_resnet18;

constexpr std::array<uint8_t, 8> kMagic = {
    'R', 'L', 'P', 'N', 'F', 'G', '0', '1'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kTrustedKnownZeroScope = 1;
constexpr uint64_t kHeaderBytes = 4096;
constexpr uint64_t kTruncBindingBytes = 32;
constexpr uint64_t kStockBindingBytes = 96;
constexpr uint64_t kRemaskBindingBytes = 32;
constexpr uint64_t kDigestBytes = 32;
constexpr uint64_t kMaxRecordBytes = uint64_t(4) << 30;
constexpr uint32_t kNoLinearIndex = std::numeric_limits<uint32_t>::max();

struct Header {
    int party = -1;
    uint32_t scope = 0;
    int full_bw = 0;
    int truncated_bw = 0;
    int scale = 0;
    uint64_t file_bytes = 0;
    uint64_t trunc_table_offset = 0;
    uint64_t stock_table_offset = 0;
    uint64_t remask_table_offset = 0;
    uint64_t trunc_data_offset = 0;
    uint64_t stock_output_mask_offset = 0;
    uint64_t remask_data_offset = 0;
    uint64_t terminal_mask_offset = 0;
    uint64_t raw_key_offset = 0;
    uint64_t raw_key_bytes = 0;
    uint64_t terminal_words = 0;
    InvocationId invocation{};
    Digest manifest_digest{};
    Digest record_set_digest{};
    Digest bundle_id{};
    std::array<Digest, contract::kLinearCount> linear_record_digests{};
    std::array<Digest, contract::kLinearCount> mask_state_digests{};
};

struct TruncBinding {
    uint32_t stream_position = 0;
    int linear_index = -2;
    uint64_t words = 0;
    uint64_t data_offset = 0;
};

struct StockBinding {
    uint32_t stream_position = 0;
    contract::StockKeyKind kind = contract::StockKeyKind::MaxPool;
    int input_bw = 0;
    int output_bw = 0;
    uint64_t input_words = 0;
    uint64_t output_words = 0;
    uint64_t key_offset = 0;
    uint64_t key_bytes = 0;
    uint64_t output_mask_offset = 0;
    uint64_t output_mask_words = 0;
    Digest key_digest{};
};

struct RemaskBinding {
    uint32_t target_linear_index = 0;
    contract::MaskSource source = contract::MaskSource::Relu2;
    uint64_t words = 0;
    uint64_t data_offset = 0;
};

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

inline bool nonzero(const uint8_t *data, size_t size) {
    return std::any_of(data, data + size,
                       [](uint8_t value) { return value != 0; });
}

inline bool host_is_little_endian() {
    const uint16_t value = 1;
    return *reinterpret_cast<const uint8_t *>(&value) == 1;
}

inline bool checked_add(uint64_t &value, uint64_t amount) {
    if (amount > std::numeric_limits<uint64_t>::max() - value) return false;
    value += amount;
    return true;
}

inline bool checked_words_bytes(uint64_t words, uint64_t &bytes) {
    if (words > std::numeric_limits<uint64_t>::max() / sizeof(uint64_t)) {
        return false;
    }
    bytes = words * sizeof(uint64_t);
    return true;
}

inline bool valid_header_identity(const Header &header) {
    return (header.party == 0 || header.party == 1) &&
           header.scope == kTrustedKnownZeroScope &&
           header.full_bw == contract::kFullBw &&
           header.truncated_bw == contract::kTruncatedBw &&
           header.scale == contract::kScale &&
           header.terminal_words == contract::kClassifierOutputWords &&
           header.file_bytes >= kHeaderBytes + kDigestBytes &&
           header.file_bytes <= kMaxRecordBytes &&
           nonzero(header.invocation.data(), header.invocation.size()) &&
           nonzero(header.manifest_digest.data(),
                   header.manifest_digest.size()) &&
           nonzero(header.record_set_digest.data(),
                   header.record_set_digest.size()) &&
           nonzero(header.bundle_id.data(), header.bundle_id.size()) &&
           std::all_of(header.linear_record_digests.begin(),
                       header.linear_record_digests.end(),
                       [](const Digest &digest) {
                           return nonzero(digest.data(), digest.size());
                       }) &&
           std::all_of(header.mask_state_digests.begin(),
                       header.mask_state_digests.end(),
                       [](const Digest &digest) {
                           return nonzero(digest.data(), digest.size());
                       });
}

inline std::vector<uint8_t> encode_header(const Header &header) {
    std::vector<uint8_t> out(static_cast<size_t>(kHeaderBytes), 0);
    std::copy(kMagic.begin(), kMagic.end(), out.begin());
    put_u32(out, 8, kVersion);
    put_u32(out, 12, static_cast<uint32_t>(header.party));
    put_u32(out, 16, header.scope);
    put_u32(out, 20, static_cast<uint32_t>(header.full_bw));
    put_u32(out, 24, static_cast<uint32_t>(header.truncated_bw));
    put_u32(out, 28, static_cast<uint32_t>(header.scale));
    put_u32(out, 32, static_cast<uint32_t>(contract::kLinearCount));
    put_u32(out, 36, static_cast<uint32_t>(contract::kTruncationCount));
    put_u32(out, 40, static_cast<uint32_t>(contract::kStockKeyCount));
    put_u32(out, 44, static_cast<uint32_t>(contract::kRemaskCount));
    put_u32(out, 48, static_cast<uint32_t>(contract::kStreamItemCount));
    put_u64(out, 56, header.file_bytes);
    put_u64(out, 64, header.trunc_table_offset);
    put_u64(out, 72, header.stock_table_offset);
    put_u64(out, 80, header.remask_table_offset);
    put_u64(out, 88, header.trunc_data_offset);
    put_u64(out, 96, header.stock_output_mask_offset);
    put_u64(out, 104, header.remask_data_offset);
    put_u64(out, 112, header.terminal_mask_offset);
    put_u64(out, 120, header.raw_key_offset);
    put_u64(out, 128, header.raw_key_bytes);
    put_u64(out, 136, header.terminal_words);
    std::copy(header.invocation.begin(), header.invocation.end(),
              out.begin() + 144);
    std::copy(header.manifest_digest.begin(), header.manifest_digest.end(),
              out.begin() + 160);
    std::copy(header.record_set_digest.begin(), header.record_set_digest.end(),
              out.begin() + 192);
    std::copy(header.bundle_id.begin(), header.bundle_id.end(),
              out.begin() + 224);
    for (size_t i = 0; i < contract::kLinearCount; ++i) {
        std::copy(header.linear_record_digests[i].begin(),
                  header.linear_record_digests[i].end(),
                  out.begin() + 256 + i * 32);
        std::copy(header.mask_state_digests[i].begin(),
                  header.mask_state_digests[i].end(),
                  out.begin() + 928 + i * 32);
    }
    return out;
}

inline bool decode_header(const uint8_t *data, size_t size, Header &header) {
    if (size < kHeaderBytes ||
        !std::equal(kMagic.begin(), kMagic.end(), data) ||
        get_u32(data, 8) != kVersion || get_u32(data, 52) != 0 ||
        get_u32(data, 32) != contract::kLinearCount ||
        get_u32(data, 36) != contract::kTruncationCount ||
        get_u32(data, 40) != contract::kStockKeyCount ||
        get_u32(data, 44) != contract::kRemaskCount ||
        get_u32(data, 48) != contract::kStreamItemCount ||
        !std::all_of(data + 1600, data + kHeaderBytes,
                     [](uint8_t value) { return value == 0; })) {
        return false;
    }
    header.party = static_cast<int>(get_u32(data, 12));
    header.scope = get_u32(data, 16);
    header.full_bw = static_cast<int>(get_u32(data, 20));
    header.truncated_bw = static_cast<int>(get_u32(data, 24));
    header.scale = static_cast<int>(get_u32(data, 28));
    header.file_bytes = get_u64(data, 56);
    header.trunc_table_offset = get_u64(data, 64);
    header.stock_table_offset = get_u64(data, 72);
    header.remask_table_offset = get_u64(data, 80);
    header.trunc_data_offset = get_u64(data, 88);
    header.stock_output_mask_offset = get_u64(data, 96);
    header.remask_data_offset = get_u64(data, 104);
    header.terminal_mask_offset = get_u64(data, 112);
    header.raw_key_offset = get_u64(data, 120);
    header.raw_key_bytes = get_u64(data, 128);
    header.terminal_words = get_u64(data, 136);
    std::copy(data + 144, data + 160, header.invocation.begin());
    std::copy(data + 160, data + 192, header.manifest_digest.begin());
    std::copy(data + 192, data + 224, header.record_set_digest.begin());
    std::copy(data + 224, data + 256, header.bundle_id.begin());
    for (size_t i = 0; i < contract::kLinearCount; ++i) {
        std::copy(data + 256 + i * 32, data + 288 + i * 32,
                  header.linear_record_digests[i].begin());
        std::copy(data + 928 + i * 32, data + 960 + i * 32,
                  header.mask_state_digests[i].begin());
    }
    return valid_header_identity(header);
}

inline std::vector<uint8_t> encode_trunc_table(
    const std::array<TruncBinding, contract::kTruncationCount> &bindings) {
    std::vector<uint8_t> out(bindings.size() * kTruncBindingBytes, 0);
    for (size_t i = 0; i < bindings.size(); ++i) {
        const size_t offset = i * kTruncBindingBytes;
        put_u32(out, offset, bindings[i].stream_position);
        put_u32(out, offset + 4,
                bindings[i].linear_index < 0
                    ? kNoLinearIndex
                    : static_cast<uint32_t>(bindings[i].linear_index));
        put_u64(out, offset + 8, bindings[i].words);
        put_u64(out, offset + 16, bindings[i].data_offset);
    }
    return out;
}

inline std::vector<uint8_t> encode_stock_table(
    const std::array<StockBinding, contract::kStockKeyCount> &bindings) {
    std::vector<uint8_t> out(bindings.size() * kStockBindingBytes, 0);
    for (size_t i = 0; i < bindings.size(); ++i) {
        const size_t offset = i * kStockBindingBytes;
        put_u32(out, offset, bindings[i].stream_position);
        put_u32(out, offset + 4, static_cast<uint32_t>(bindings[i].kind));
        put_u32(out, offset + 8, static_cast<uint32_t>(bindings[i].input_bw));
        put_u32(out, offset + 12,
                static_cast<uint32_t>(bindings[i].output_bw));
        put_u64(out, offset + 16, bindings[i].input_words);
        put_u64(out, offset + 24, bindings[i].output_words);
        put_u64(out, offset + 32, bindings[i].key_offset);
        put_u64(out, offset + 40, bindings[i].key_bytes);
        put_u64(out, offset + 48, bindings[i].output_mask_offset);
        put_u64(out, offset + 56, bindings[i].output_mask_words);
        std::copy(bindings[i].key_digest.begin(),
                  bindings[i].key_digest.end(), out.begin() + offset + 64);
    }
    return out;
}

inline std::vector<uint8_t> encode_remask_table(
    const std::array<RemaskBinding, contract::kRemaskCount> &bindings) {
    std::vector<uint8_t> out(bindings.size() * kRemaskBindingBytes, 0);
    for (size_t i = 0; i < bindings.size(); ++i) {
        const size_t offset = i * kRemaskBindingBytes;
        put_u32(out, offset, bindings[i].target_linear_index);
        put_u32(out, offset + 4, static_cast<uint32_t>(bindings[i].source));
        put_u64(out, offset + 8, bindings[i].words);
        put_u64(out, offset + 16, bindings[i].data_offset);
    }
    return out;
}

class MappedRecord {
  public:
    MappedRecord() = default;
    MappedRecord(const MappedRecord &) = delete;
    MappedRecord &operator=(const MappedRecord &) = delete;
    ~MappedRecord() { close(); }

    bool open(const std::string &path) {
        close();
        if (!host_is_little_endian() || !contract::contract_valid()) {
            return false;
        }
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd_ < 0) return false;
        struct stat metadata {};
        if (::fstat(fd_, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
            metadata.st_size < static_cast<off_t>(kHeaderBytes + kDigestBytes) ||
            static_cast<uint64_t>(metadata.st_size) > kMaxRecordBytes ||
            static_cast<uint64_t>(metadata.st_size) >
                std::numeric_limits<size_t>::max()) {
            close();
            return false;
        }
        size_ = static_cast<size_t>(metadata.st_size);
        void *mapping = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapping == MAP_FAILED) {
            data_ = nullptr;
            close();
            return false;
        }
        data_ = static_cast<const uint8_t *>(mapping);
        if (!decode_header(data_, size_, header_) ||
            header_.file_bytes != size_ || !parse_and_validate_layout()) {
            close();
            return false;
        }
        Digest calculated{};
        if (!ringlpn_freshness::digest(data_, size_ - kDigestBytes,
                                       calculated) ||
            !std::equal(calculated.begin(), calculated.end(),
                        data_ + size_ - kDigestBytes)) {
            close();
            return false;
        }
        digest_ = calculated;
        return true;
    }

    void close() {
        if (data_ != nullptr) {
            ::munmap(const_cast<uint8_t *>(data_), size_);
        }
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        data_ = nullptr;
        size_ = 0;
        header_ = Header{};
        digest_.fill(0);
    }

    bool valid() const { return data_ != nullptr; }
    const Header &header() const { return header_; }
    const Digest &digest() const { return digest_; }
    const TruncBinding &trunc_binding(size_t index) const {
        return trunc_bindings_[index];
    }
    const StockBinding &stock_binding(size_t index) const {
        return stock_bindings_[index];
    }
    const RemaskBinding &remask_binding(size_t index) const {
        return remask_bindings_[index];
    }
    const uint64_t *trunc_share(size_t index) const {
        return words(trunc_bindings_[index].data_offset);
    }
    const uint64_t *stock_output_mask_share(size_t index) const {
        return words(stock_bindings_[index].output_mask_offset);
    }
    const uint64_t *remask_delta(size_t index) const {
        return words(remask_bindings_[index].data_offset);
    }
    const uint64_t *terminal_mask() const {
        return words(header_.terminal_mask_offset);
    }
    const uint8_t *raw_key(size_t index) const {
        return data_ + stock_bindings_[index].key_offset;
    }

  private:
    const uint64_t *words(uint64_t offset) const {
        return reinterpret_cast<const uint64_t *>(data_ + offset);
    }

    bool parse_and_validate_layout() {
        const uint64_t trunc_table_bytes =
            contract::kTruncationCount * kTruncBindingBytes;
        const uint64_t stock_table_bytes =
            contract::kStockKeyCount * kStockBindingBytes;
        const uint64_t remask_table_bytes =
            contract::kRemaskCount * kRemaskBindingBytes;
        if (header_.trunc_table_offset != kHeaderBytes ||
            header_.stock_table_offset !=
                header_.trunc_table_offset + trunc_table_bytes ||
            header_.remask_table_offset !=
                header_.stock_table_offset + stock_table_bytes ||
            header_.trunc_data_offset !=
                header_.remask_table_offset + remask_table_bytes ||
            (header_.trunc_data_offset % alignof(uint64_t)) != 0) {
            return false;
        }

        uint64_t trunc_cursor = header_.trunc_data_offset;
        for (size_t i = 0; i < contract::kTruncationCount; ++i) {
            const uint64_t offset =
                header_.trunc_table_offset + i * kTruncBindingBytes;
            TruncBinding binding;
            binding.stream_position = get_u32(data_, offset);
            const uint32_t encoded_linear = get_u32(data_, offset + 4);
            binding.linear_index = encoded_linear == kNoLinearIndex
                                       ? -1
                                       : static_cast<int>(encoded_linear);
            binding.words = get_u64(data_, offset + 8);
            binding.data_offset = get_u64(data_, offset + 16);
            if (get_u64(data_, offset + 24) != 0 ||
                binding.stream_position !=
                    contract::kTruncationSpecs[i].stream_position ||
                binding.linear_index !=
                    contract::kTruncationSpecs[i].linear_index ||
                binding.words != contract::kTruncationSpecs[i].words ||
                binding.data_offset != trunc_cursor) {
                return false;
            }
            uint64_t bytes = 0;
            if (!checked_words_bytes(binding.words, bytes) ||
                !checked_add(trunc_cursor, bytes)) {
                return false;
            }
            trunc_bindings_[i] = binding;
        }
        if (header_.stock_output_mask_offset != trunc_cursor) return false;

        uint64_t output_mask_cursor = header_.stock_output_mask_offset;
        uint64_t raw_key_cursor = header_.raw_key_offset;
        for (size_t i = 0; i < contract::kStockKeyCount; ++i) {
            const uint64_t offset =
                header_.stock_table_offset + i * kStockBindingBytes;
            StockBinding binding;
            binding.stream_position = get_u32(data_, offset);
            binding.kind = static_cast<contract::StockKeyKind>(
                get_u32(data_, offset + 4));
            binding.input_bw = static_cast<int>(get_u32(data_, offset + 8));
            binding.output_bw =
                static_cast<int>(get_u32(data_, offset + 12));
            binding.input_words = get_u64(data_, offset + 16);
            binding.output_words = get_u64(data_, offset + 24);
            binding.key_offset = get_u64(data_, offset + 32);
            binding.key_bytes = get_u64(data_, offset + 40);
            binding.output_mask_offset = get_u64(data_, offset + 48);
            binding.output_mask_words = get_u64(data_, offset + 56);
            std::copy(data_ + offset + 64, data_ + offset + 96,
                      binding.key_digest.begin());
            const contract::StockKeySpec &spec = contract::kStockKeySpecs[i];
            if (binding.stream_position != spec.stream_position ||
                binding.kind != spec.kind || binding.input_bw != spec.input_bw ||
                binding.output_bw != spec.output_bw ||
                binding.input_words != spec.input_words ||
                binding.output_words != spec.output_words ||
                binding.output_mask_words != spec.output_words ||
                binding.output_mask_offset != output_mask_cursor ||
                binding.key_offset != raw_key_cursor || binding.key_bytes == 0 ||
                !nonzero(binding.key_digest.data(), binding.key_digest.size())) {
                return false;
            }
            uint64_t mask_bytes = 0;
            if (!checked_words_bytes(binding.output_mask_words, mask_bytes) ||
                !checked_add(output_mask_cursor, mask_bytes) ||
                !checked_add(raw_key_cursor, binding.key_bytes)) {
                return false;
            }
            stock_bindings_[i] = binding;
        }
        if (header_.remask_data_offset != output_mask_cursor) return false;

        uint64_t remask_cursor = header_.remask_data_offset;
        for (size_t i = 0; i < contract::kRemaskCount; ++i) {
            const uint64_t offset =
                header_.remask_table_offset + i * kRemaskBindingBytes;
            RemaskBinding binding;
            binding.target_linear_index = get_u32(data_, offset);
            binding.source = static_cast<contract::MaskSource>(
                get_u32(data_, offset + 4));
            binding.words = get_u64(data_, offset + 8);
            binding.data_offset = get_u64(data_, offset + 16);
            const contract::RemaskSpec &spec = contract::kRemaskSpecs[i];
            if (get_u64(data_, offset + 24) != 0 ||
                binding.target_linear_index != spec.target_linear_index ||
                binding.source != spec.source || binding.words != spec.words ||
                binding.data_offset != remask_cursor) {
                return false;
            }
            uint64_t bytes = 0;
            if (!checked_words_bytes(binding.words, bytes) ||
                !checked_add(remask_cursor, bytes)) {
                return false;
            }
            remask_bindings_[i] = binding;
        }
        uint64_t terminal_bytes = 0;
        if (header_.terminal_mask_offset != remask_cursor ||
            !checked_words_bytes(header_.terminal_words, terminal_bytes) ||
            !checked_add(remask_cursor, terminal_bytes) ||
            header_.raw_key_offset != remask_cursor ||
            raw_key_cursor != header_.raw_key_offset + header_.raw_key_bytes ||
            raw_key_cursor + kDigestBytes != header_.file_bytes ||
            raw_key_cursor > size_ - kDigestBytes) {
            return false;
        }

        const uint64_t truncated_limit = uint64_t(1) << header_.truncated_bw;
        const uint64_t full_limit = uint64_t(1) << header_.full_bw;
        for (size_t i = 0; i < contract::kTruncationCount; ++i) {
            const uint64_t *values = trunc_share(i);
            if (!std::all_of(values, values + trunc_bindings_[i].words,
                             [&](uint64_t value) {
                                 return value < truncated_limit;
                             })) {
                return false;
            }
        }
        for (size_t i = 0; i < contract::kStockKeyCount; ++i) {
            const uint64_t limit = uint64_t(1) << stock_bindings_[i].output_bw;
            const uint64_t *values = stock_output_mask_share(i);
            if (!std::all_of(values,
                             values + stock_bindings_[i].output_mask_words,
                             [&](uint64_t value) { return value < limit; })) {
                return false;
            }
            Digest calculated{};
            if (!ringlpn_freshness::digest(raw_key(i),
                                           stock_bindings_[i].key_bytes,
                                           calculated) ||
                calculated != stock_bindings_[i].key_digest) {
                return false;
            }
        }
        for (size_t i = 0; i < contract::kRemaskCount; ++i) {
            const uint64_t *values = remask_delta(i);
            if (!std::all_of(values, values + remask_bindings_[i].words,
                             [&](uint64_t value) { return value < full_limit; })) {
                return false;
            }
        }
        const uint64_t *terminal = terminal_mask();
        return std::all_of(terminal, terminal + header_.terminal_words,
                           [&](uint64_t value) { return value < full_limit; });
    }

    int fd_ = -1;
    const uint8_t *data_ = nullptr;
    size_t size_ = 0;
    Header header_{};
    Digest digest_{};
    std::array<TruncBinding, contract::kTruncationCount> trunc_bindings_{};
    std::array<StockBinding, contract::kStockKeyCount> stock_bindings_{};
    std::array<RemaskBinding, contract::kRemaskCount> remask_bindings_{};
};

inline bool public_headers_match(const Header &p0, const Header &p1) {
    return p0.party == 0 && p1.party == 1 && p0.scope == p1.scope &&
           p0.full_bw == p1.full_bw &&
           p0.truncated_bw == p1.truncated_bw && p0.scale == p1.scale &&
           p0.invocation == p1.invocation &&
           p0.manifest_digest == p1.manifest_digest &&
           p0.record_set_digest == p1.record_set_digest &&
           p0.bundle_id == p1.bundle_id &&
           p0.trunc_table_offset == p1.trunc_table_offset &&
           p0.stock_table_offset == p1.stock_table_offset &&
           p0.remask_table_offset == p1.remask_table_offset &&
           p0.trunc_data_offset == p1.trunc_data_offset &&
           p0.stock_output_mask_offset == p1.stock_output_mask_offset &&
           p0.remask_data_offset == p1.remask_data_offset &&
           p0.terminal_mask_offset == p1.terminal_mask_offset &&
           p0.raw_key_offset == p1.raw_key_offset &&
           p0.raw_key_bytes == p1.raw_key_bytes &&
           p0.terminal_words == p1.terminal_words;
}

}  // namespace ringlpn_full_graph_record
