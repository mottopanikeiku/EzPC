#include "public_ring_vector_xof.h"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using ringlpn_public_vector_xof::Context;
using ringlpn_public_vector_xof::UniformReducer;
using ringlpn_public_vector_xof::Word;

using DigestPtr = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;

struct FixedVector {
    const char *name;
    Context context;
    std::array<uint8_t, 32> scope;
    uint64_t chunk;
    const char *expected_hex;
};

std::vector<uint8_t> decode_hex(const std::string &text) {
    if (text.size() % 2 != 0) return {};
    std::vector<uint8_t> bytes(text.size() / 2);
    for (size_t index = 0; index < bytes.size(); ++index) {
        const auto digit = [](char value) -> int {
            if (value >= '0' && value <= '9') return value - '0';
            if (value >= 'a' && value <= 'f') return value - 'a' + 10;
            return -1;
        };
        const int high = digit(text[2 * index]);
        const int low = digit(text[2 * index + 1]);
        if (high < 0 || low < 0) return {};
        bytes[index] = static_cast<uint8_t>((high << 4) | low);
    }
    return bytes;
}

bool reference_reduce_mod10(Word candidate, Word &reduced) {
    constexpr Word kFirstRejected = 18446744073709551610ULL;
    if (candidate >= kFirstRejected) return false;
    reduced = candidate - (candidate / 10) * 10;
    return true;
}

int fail(const std::string &message) {
    std::cerr << "public_ring_vector_xof_control=FAIL reason=" << message
              << '\n';
    return 1;
}

}  // namespace

int main() {
    std::array<uint8_t, 32> seed{};
    std::array<uint8_t, 32> scope{};
    for (size_t index = 0; index < seed.size(); ++index) {
        seed[index] = static_cast<uint8_t>(index);
        scope[index] = static_cast<uint8_t>(0xa0 + index);
    }
    auto other_scope = scope;
    other_scope.back() ^= 1;

    constexpr Word kPrime0 = 4611686018326724609ULL;
    constexpr Word kPrime1 = 4611686018309947393ULL;
    const Context base{128, 2, 8, 8, 0, 0, 7, kPrime0, 0};
    auto changed_n = base;
    changed_n.n = 256;
    auto changed_c = base;
    changed_c.c = 3;
    auto changed_t = base;
    changed_t.t = 4;
    auto changed_log_domain = base;
    changed_log_domain.log_domain = 9;
    auto changed_direction = base;
    changed_direction.direction = 1;
    auto changed_limb = base;
    changed_limb.limb = 1;
    auto changed_batch = base;
    changed_batch.slot_batch = 8;
    auto changed_modulus = base;
    changed_modulus.modulus = kPrime1;
    auto changed_regular = base;
    changed_regular.regular = 1;

    const std::vector<FixedVector> vectors = {
        {"base_limb0", base, scope, 0,
         "eb425643fd717fa793a28f40ea17790e6f9683588573cd8616a4c562f0ef6eda"},
        {"other_handle", base, other_scope, 0,
         "dccf419bb45cea1029d5def676b04aecfba1b8251adb254546663e231ab5aa0e"},
        {"n", changed_n, scope, 0,
         "72b557abd1f25aaa11733ced99ac28789eb73b0912558ba609887b50b6813ec2"},
        {"c", changed_c, scope, 0,
         "17007d5796276c82c0c7ed0d16e78e9d2cae209588676caf75d9c943ecd3d220"},
        {"t", changed_t, scope, 0,
         "d9a339ddfa3134aaea505ded0db0ff469747a9fa404e36b2e5779fa8bac027cb"},
        {"log_domain", changed_log_domain, scope, 0,
         "91cf3f65d6eb9638c0800a59885995e7ad99eb12029482e959ff7cf1ac7bc8fe"},
        {"direction", changed_direction, scope, 0,
         "1cf387967a807a2abba5c2622b0f91f09b68b470ac58da73708542aea4b20112"},
        {"limb1", changed_limb, scope, 0,
         "1fe3f6cae912e4594fe1767ecc8065d283feb6bed688470a61c2c526bb31558e"},
        {"slot_batch", changed_batch, scope, 0,
         "a84c14df5c0cd030b3d0cff1c0a069271ae6ae217b3c4bbafa830aa0c1c7533c"},
        {"modulus", changed_modulus, scope, 0,
         "3eae99a15a3a0ec5455a19bf4df87973a50778e73f72c08659b78fd245735c1b"},
        {"regular", changed_regular, scope, 0,
         "be20ad07201e7bf2258c797550ee67c84b85579fbe64c8c5044bfb457a44f180"},
        {"adjacent_chunk", base, scope, 1,
         "3fea89a673903fe197ffb649c93824fb1aa98b31ab71824e73a025cddbb23533"},
    };

    DigestPtr digest(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!digest) return fail("digest_allocation");
    std::vector<std::array<uint8_t, 32>> observed;
    for (const FixedVector &vector : vectors) {
        std::array<uint8_t, 32> output{};
        const Context encoded_context =
            ringlpn_public_vector_xof::context_from(vector.context);
        if (!ringlpn_public_vector_xof::squeeze_chunk(
                digest.get(), seed, vector.scope, encoded_context, vector.chunk,
                output.data(), output.size())) {
            return fail(std::string("squeeze_") + vector.name);
        }
        const std::vector<uint8_t> expected = decode_hex(vector.expected_hex);
        if (expected.size() != output.size() ||
            !std::equal(output.begin(), output.end(), expected.begin())) {
            return fail(std::string("fixed_vector_") + vector.name);
        }
        if (std::find(observed.begin(), observed.end(), output) !=
            observed.end()) {
            return fail(std::string("domain_collision_") + vector.name);
        }
        observed.push_back(output);
    }

    std::array<uint8_t, 32> repeated{};
    if (!ringlpn_public_vector_xof::squeeze_chunk(
            digest.get(), seed, scope,
            ringlpn_public_vector_xof::context_from(base), 0, repeated.data(),
            repeated.size()) ||
        repeated != observed.front()) {
        return fail("identical_context_not_repeatable");
    }

    constexpr std::array<Word, 8> kExpectedReduced = {
        2846118822293160681ULL, 1042891083585921683ULL,
        490174951207442029ULL,  1904723508454597651ULL,
        1623445082922503127ULL, 3702581416989693152ULL,
        3430255960190703916ULL, 1122215359755582108ULL};
    std::array<Word, kExpectedReduced.size()> reduced{};
    std::vector<uint8_t> scratch;
    if (!ringlpn_public_vector_xof::generate_uniform_words(
            digest.get(), seed, scope,
            ringlpn_public_vector_xof::context_from(base), reduced.data(),
            reduced.size(),
            scratch) ||
        reduced != kExpectedReduced) {
        return fail("reduced_fixed_vector");
    }
    std::array<Word, kExpectedReduced.size()> repeated_reduced{};
    if (!ringlpn_public_vector_xof::generate_uniform_words(
            digest.get(), seed, scope,
            ringlpn_public_vector_xof::context_from(base),
            repeated_reduced.data(),
            repeated_reduced.size(), scratch) ||
        repeated_reduced != reduced) {
        return fail("reduced_repeatability");
    }

    const UniformReducer reducer(10);
    constexpr Word kLimit = 18446744073709551610ULL;
    if (!reducer.valid() || reducer.limit() !=
                                static_cast<ringlpn_public_vector_xof::Wide>(
                                    kLimit)) {
        return fail("rejection_limit");
    }
    constexpr std::array<Word, 6> candidates = {
        kLimit - 1, kLimit, kLimit + 1, std::numeric_limits<Word>::max(), 20,
        21};
    std::vector<Word> production_accepted;
    std::vector<Word> reference_accepted;
    for (Word candidate : candidates) {
        Word production_value = 0;
        Word reference_value = 0;
        const bool production_accepts = reducer.reduce(candidate, production_value);
        const bool reference_accepts =
            reference_reduce_mod10(candidate, reference_value);
        if (production_accepts != reference_accepts ||
            (production_accepts && production_value != reference_value)) {
            return fail("reducer_reference_mismatch");
        }
        if (production_accepts) production_accepted.push_back(production_value);
        if (reference_accepts) reference_accepted.push_back(reference_value);
    }
    const std::vector<Word> expected_accepted = {9, 0, 1};
    if (production_accepted != expected_accepted ||
        reference_accepted != expected_accepted) {
        return fail("consecutive_rejection_sequence");
    }
    Word unused = 0;
    if (UniformReducer(0).valid() || UniformReducer(0).reduce(0, unused) ||
        UniformReducer(1).valid() || UniformReducer(1).reduce(0, unused)) {
        return fail("invalid_modulus_fail_closed");
    }

    std::cout << "public_ring_vector_xof_control=pass fixed_vectors="
              << vectors.size()
              << " reducer_candidates=" << candidates.size() << '\n';
    return 0;
}
