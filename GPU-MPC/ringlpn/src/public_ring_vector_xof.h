#pragma once

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace ringlpn_public_vector_xof {

using Word = uint64_t;
using Wide = unsigned __int128;

constexpr size_t kMaxChunkBlocks = 4096;
constexpr std::array<uint8_t, 20> kDomain = {
    'R', 'I', 'N', 'G', 'L', 'P', 'N', '_', 'P', 'U',
    'B', 'L', 'I', 'C', '_', 'A', '_', 'V', '2', 0};

// Every field is part of the public-vector random-oracle input. Keep this
// explicit rather than hashing a padded C++ struct.
struct Context {
    uint64_t n = 0;
    uint64_t c = 0;
    uint64_t t = 0;
    uint64_t log_domain = 0;
    uint64_t direction = 0;
    uint64_t limb = 0;
    uint64_t slot_batch = 0;
    uint64_t modulus = 0;
    uint64_t regular = 0;
};
template <typename Params>
inline Context context_from(const Params &params) {
    return {
        static_cast<uint64_t>(params.n),
        static_cast<uint64_t>(params.c),
        static_cast<uint64_t>(params.t),
        static_cast<uint64_t>(params.log_domain),
        static_cast<uint64_t>(params.direction),
        static_cast<uint64_t>(params.limb),
        static_cast<uint64_t>(params.slot_batch),
        static_cast<uint64_t>(params.modulus),
        params.regular ? 1ULL : 0ULL};
}


inline void put_u64_le(uint8_t *destination, uint64_t value) {
    for (size_t byte = 0; byte < sizeof(value); ++byte) {
        destination[byte] = static_cast<uint8_t>(value >> (8 * byte));
    }
}

inline Word get_u64_le(const uint8_t *source) {
    Word value = 0;
    for (size_t byte = 0; byte < sizeof(value); ++byte) {
        value |= static_cast<Word>(source[byte]) << (8 * byte);
    }
    return value;
}

inline bool squeeze_chunk(EVP_MD_CTX *digest,
                          const std::array<uint8_t, 32> &seed,
                          const std::array<uint8_t, 32> &scope_id,
                          const Context &context, uint64_t chunk,
                          uint8_t *output, size_t output_bytes) {
    if (digest == nullptr || output == nullptr || output_bytes == 0) {
        return false;
    }
    const std::array<uint64_t, 10> words = {
        context.n,          context.c,       context.t,
        context.log_domain, context.direction, context.limb,
        context.slot_batch, context.modulus, context.regular,
        chunk};
    std::array<uint8_t, 10 * sizeof(uint64_t)> encoded{};
    for (size_t index = 0; index < words.size(); ++index) {
        put_u64_le(encoded.data() + index * sizeof(uint64_t), words[index]);
    }
    return EVP_DigestInit_ex(digest, EVP_shake256(), nullptr) == 1 &&
           EVP_DigestUpdate(digest, kDomain.data(), kDomain.size()) == 1 &&
           EVP_DigestUpdate(digest, seed.data(), seed.size()) == 1 &&
           EVP_DigestUpdate(digest, scope_id.data(), scope_id.size()) == 1 &&
           EVP_DigestUpdate(digest, encoded.data(), encoded.size()) == 1 &&
           EVP_DigestFinalXOF(digest, output, output_bytes) == 1;
}

// Narrow deterministic seam shared by production and the host control. The
// limit is floor(2^64/modulus)*modulus, so accepting only candidates below it
// makes the following reduction exact-uniform.
class UniformReducer {
  public:
    explicit UniformReducer(Word modulus) : modulus_(modulus) {
        if (modulus_ >= 2) {
            limit_ = ((Wide(1) << 64) / static_cast<Wide>(modulus_)) *
                     static_cast<Wide>(modulus_);
        }
    }

    bool valid() const { return modulus_ >= 2; }
    Wide limit() const { return limit_; }

    bool reduce(Word candidate, Word &reduced) const {
        if (!valid() || static_cast<Wide>(candidate) >= limit_) return false;
        reduced = candidate % modulus_;
        return true;
    }

  private:
    Word modulus_ = 0;
    Wide limit_ = 0;
};

inline bool generate_uniform_words(EVP_MD_CTX *digest,
                                   const std::array<uint8_t, 32> &seed,
                                   const std::array<uint8_t, 32> &scope_id,
                                   const Context &context, Word *words,
                                   size_t count,
                                   std::vector<uint8_t> &output_scratch) {
    UniformReducer reducer(context.modulus);
    if (digest == nullptr || !reducer.valid() ||
        (count != 0 && words == nullptr)) {
        return false;
    }

    output_scratch.resize(kMaxChunkBlocks * 16);
    size_t produced = 0;
    uint64_t chunk = 0;
    while (produced < count) {
        const size_t remaining = count - produced;
        const size_t blocks =
            std::min(kMaxChunkBlocks, remaining / 2 + remaining % 2);
        if (blocks == 0 ||
            !squeeze_chunk(digest, seed, scope_id, context, chunk,
                           output_scratch.data(), blocks * 16)) {
            return false;
        }
        for (size_t index = 0;
             index < 2 * blocks && produced < count; ++index) {
            Word reduced = 0;
            if (reducer.reduce(get_u64_le(output_scratch.data() +
                                          index * sizeof(Word)),
                               reduced)) {
                words[produced++] = reduced;
            }
        }
        if (chunk == std::numeric_limits<uint64_t>::max()) return false;
        ++chunk;
    }
    return true;
}

}  // namespace ringlpn_public_vector_xof
