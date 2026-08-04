#include "secure_convert.h"

#include <array>
#include <chrono>
#include <climits>
#include <limits>

namespace ringlpn_2pc {
namespace {

constexpr uint64_t kP0 = 4611686018326724609ULL;
constexpr uint64_t kP1 = 4611686018309947393ULL;

U128 modulus_for_qbits(int qbits) {
    return qbits == 128 ? U128(kP0) * U128(kP1) : U128(kP0);
}

int bit_length(U128 value) {
    int bits = 0;
    while (value != 0) {
        ++bits;
        value >>= 1;
    }
    return bits;
}

U128 bit_mask(int bits) {
    return bits == 128 ? ~U128(0) : (U128(1) << bits) - 1;
}

U128 reduce(U128 value, int bits) {
    return value & bit_mask(bits);
}

U128 subtract(U128 lhs, U128 rhs, int bits) {
    return reduce(lhs - rhs, bits);
}

uint64_t reduce_u64(U128 value, int bits) {
    return uint64_t(reduce(value, bits));
}

bool multiply_fits_size(size_t lhs, size_t rhs) {
    return rhs == 0 || lhs <= std::numeric_limits<size_t>::max() / rhs;
}
void store_u32_le(uint8_t *dst, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        dst[i] = uint8_t(value >> (8 * i));
    }
}

void store_u64_le(uint8_t *dst, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = uint8_t(value >> (8 * i));
    }
}

bool agree_convert_preflight(const SecureConvertParams &params,
                             bool locally_valid, PartyChannel &channel) {
    // Canonical public context followed by one local-validity byte. Both
    // parties execute this exchange even when local validation fails, so a
    // malformed share produces one common abort instead of a peer hang.
    std::array<uint8_t, 25> mine{};
    std::array<uint8_t, 25> peer{};
    store_u64_le(mine.data(), params.sid);
    store_u32_le(mine.data() + 8, uint32_t(params.qbits));
    store_u32_le(mine.data() + 12, uint32_t(params.bw));
    store_u64_le(mine.data() + 16, uint64_t(params.count));
    mine[24] = locally_valid ? 1 : 0;
    channel.exchange_bytes(mine.data(), peer.data(), mine.size());
    for (size_t i = 0; i < 24; ++i) {
        if (mine[i] != peer[i]) {
            return false;
        }
    }
    return mine[24] == 1 && peer[24] == 1;
}


bool validate_inputs(const SecureConvertParams &params,
                     const std::vector<U128> &own_z, int &ell,
                     U128 &modulus) {
    if (params.sid == 0 ||
        (params.qbits != 64 && params.qbits != 128) || params.bw <= 2 ||
        params.bw > 32 || params.count == 0 ||
        params.count > kMaxSecureConvertBatch || own_z.size() != params.count) {
        return false;
    }

    modulus = modulus_for_qbits(params.qbits);
    ell = bit_length(2 * modulus - 1);
    const size_t triple_factor = size_t(2 * ell - 2);
    if (!multiply_fits_size(params.count, size_t(ell)) ||
        !multiply_fits_size(params.count, triple_factor) ||
        !multiply_fits_size(params.count, sizeof(U128)) ||
        params.count * size_t(ell) > size_t(INT_MAX) ||
        params.count * triple_factor > size_t(INT_MAX) ||
        params.count * sizeof(U128) > size_t(INT_MAX)) {
        return false;
    }
    for (U128 share : own_z) {
        if (share >= modulus) {
            return false;
        }
    }
    return true;
}

struct OnlineCosts {
    uint64_t triples = 0;
    uint64_t logical = 0;
    uint64_t sent = 0;
    uint64_t received = 0;
    uint64_t post = 0;
};

std::vector<uint8_t> and_batch(const std::vector<uint8_t> &x,
                               const std::vector<uint8_t> &y,
                               const std::vector<BitTriple> &triples,
                               size_t &position, PartyChannel &channel,
                               OnlineCosts &costs) {
    const size_t count = x.size();
    std::vector<uint8_t> mine(count), peer(count), output(count);
    for (size_t i = 0; i < count; ++i) {
        mine[i] = uint8_t(((x[i] ^ triples[position + i].a) & 1) |
                          (((y[i] ^ triples[position + i].b) & 1) << 1));
    }
    channel.exchange_bytes(mine.data(), peer.data(), count);
    for (size_t i = 0; i < count; ++i) {
        const uint8_t opened = mine[i] ^ peer[i];
        const uint8_t d = opened & 1;
        const uint8_t e = (opened >> 1) & 1;
        output[i] = uint8_t((triples[position + i].c ^
                             (d & triples[position + i].b) ^
                             (e & triples[position + i].a) ^
                             (channel.is_p0() ? (d & e) : 0)) &
                            1);
    }
    position += count;
    costs.triples += count;
    costs.logical += 2 * count;
    costs.sent += 2 * count;
    costs.received += 2 * count;
    costs.post += count;
    return output;
}

struct RippleResult {
    std::vector<uint8_t> carry;
    std::vector<uint8_t> sum;
};

RippleResult ripple(const std::vector<U128> &public_words,
                    const std::vector<uint8_t> &secret_bits, int carry_in,
                    int ell, bool produce_sum,
                    const std::vector<BitTriple> &triples, size_t &position,
                    PartyChannel &channel, OnlineCosts &costs) {
    const size_t count = public_words.size();
    RippleResult result;
    result.carry.assign(count, channel.is_p0() ? uint8_t(carry_in) : 0);
    if (produce_sum) {
        result.sum.assign(count * size_t(ell), 0);
    }
    for (int bit = 0; bit < ell; ++bit) {
        std::vector<uint8_t> input_bit(count), old_carry = result.carry;
        std::vector<uint8_t> next_carry;
        for (size_t i = 0; i < count; ++i) {
            input_bit[i] = secret_bits[i * size_t(ell) + size_t(bit)];
            const uint8_t public_bit =
                uint8_t((public_words[i] >> bit) & 1);
            if (produce_sum) {
                result.sum[i * size_t(ell) + size_t(bit)] = uint8_t(
                    input_bit[i] ^ old_carry[i] ^
                    (channel.is_p0() ? public_bit : 0));
            }
        }
        if (bit == 0) {
            next_carry.resize(count);
            for (size_t i = 0; i < count; ++i) {
                next_carry[i] = uint8_t(input_bit[i] & carry_in);
            }
        } else {
            next_carry = and_batch(input_bit, old_carry, triples, position,
                                   channel, costs);
        }
        for (size_t i = 0; i < count; ++i) {
            if ((public_words[i] >> bit) & 1) {
                next_carry[i] ^= uint8_t(input_bit[i] ^ old_carry[i]);
            }
        }
        result.carry.swap(next_carry);
    }
    return result;
}

bool convert_with_correlations(
    const SecureConvertParams &params, U128 modulus, int ell,
    const std::vector<U128> &own_z,
    const std::vector<secure_convert_detail::Edabit> &edabits,
    const std::vector<secure_convert_detail::Dabit> &dabits,
    const std::vector<BitTriple> &triples, PartyChannel &channel,
    std::vector<uint64_t> &own_r, OnlineCosts &costs) {
    const size_t count = params.count;
    std::vector<U128> mine(count), peer(count), opened(count);
    for (size_t i = 0; i < count; ++i) {
        mine[i] = reduce(own_z[i] + edabits[i].arithmetic, ell);
    }
    channel.exchange_bytes(reinterpret_cast<const uint8_t *>(mine.data()),
                           reinterpret_cast<uint8_t *>(peer.data()),
                           count * sizeof(U128));
    for (size_t i = 0; i < count; ++i) {
        opened[i] = reduce(mine[i] + peer[i], ell);
    }
    costs.logical += count * size_t(ell);
    costs.sent += count * size_t(ell);
    costs.received += count * size_t(ell);

    std::vector<uint8_t> negated_mask_bits(count * size_t(ell));
    for (size_t i = 0; i < count; ++i) {
        for (int bit = 0; bit < ell; ++bit) {
            negated_mask_bits[i * size_t(ell) + size_t(bit)] = uint8_t(
                edabits[i].bits[size_t(bit)] ^ (channel.is_p0() ? 1 : 0));
        }
    }

    size_t triple_position = 0;
    const RippleResult sum = ripple(opened, negated_mask_bits, 1, ell, true,
                                    triples, triple_position, channel, costs);
    const std::vector<U128> threshold(count, reduce(-modulus, ell));
    const RippleResult wrap = ripple(threshold, sum.sum, 0, ell, false,
                                     triples, triple_position, channel, costs);

    std::vector<uint8_t> masked_wrap(count), peer_masked_wrap(count);
    for (size_t i = 0; i < count; ++i) {
        masked_wrap[i] = uint8_t((wrap.carry[i] ^ dabits[i].bit) & 1);
    }
    channel.exchange_bytes(masked_wrap.data(), peer_masked_wrap.data(), count);
    costs.logical += count;
    costs.sent += count;
    costs.received += count;
    costs.post += count;

    own_r.resize(count);
    const U128 reduced_modulus = reduce(modulus, params.bw);
    for (size_t i = 0; i < count; ++i) {
        const uint8_t opened_mask =
            uint8_t((masked_wrap[i] ^ peer_masked_wrap[i]) & 1);
        const U128 wrap_arithmetic =
            opened_mask == 0
                ? dabits[i].arithmetic
                : (channel.is_p0()
                       ? subtract(1, dabits[i].arithmetic, params.bw)
                       : subtract(0, dabits[i].arithmetic, params.bw));
        own_r[i] = reduce_u64(
            subtract(own_z[i],
                     reduce(reduced_modulus * wrap_arithmetic, params.bw),
                     params.bw),
            params.bw);
    }
    return triple_position == triples.size();
}

}  // namespace
bool validate_secure_convert_inputs(const SecureConvertParams &params,
                                    const std::vector<U128> &own_z) {
    int ell = 0;
    U128 modulus = 0;
    return validate_inputs(params, own_z, ell, modulus);
}


namespace secure_convert_detail {

std::vector<Dabit> generate_dabits(PartyChannel &channel, size_t count,
                                   int arithmetic_bits, PartyRandom &random) {
    std::vector<Dabit> output(count);
    std::vector<uint8_t> bits(count);
    for (size_t i = 0; i < count; ++i) {
        bits[i] = random.bit();
    }
    if (channel.is_p0()) {
        std::vector<U128> message0(count), message1(count);
        for (size_t i = 0; i < count; ++i) {
            const U128 arithmetic = reduce(random.u128(), arithmetic_bits);
            output[i] = {bits[i], arithmetic};
            message0[i] = subtract(U128(bits[i]), arithmetic, arithmetic_bits);
            message1[i] =
                subtract(U128(1 - bits[i]), arithmetic, arithmetic_bits);
        }
        channel.ot_send_128(message0, message1);
    } else {
        const std::vector<U128> arithmetic = channel.ot_recv_128(bits);
        for (size_t i = 0; i < count; ++i) {
            output[i] = {bits[i], reduce(arithmetic[i], arithmetic_bits)};
        }
    }
    return output;
}

std::vector<Edabit> generate_edabits(PartyChannel &channel, size_t count,
                                     int arithmetic_bits,
                                     PartyRandom &random) {
    const std::vector<Dabit> dabits = generate_dabits(
        channel, count * size_t(arithmetic_bits), arithmetic_bits, random);
    std::vector<Edabit> output(count);
    for (size_t i = 0; i < count; ++i) {
        output[i].bits.resize(size_t(arithmetic_bits));
        for (int bit = 0; bit < arithmetic_bits; ++bit) {
            const Dabit &dabit =
                dabits[i * size_t(arithmetic_bits) + size_t(bit)];
            output[i].bits[size_t(bit)] = dabit.bit;
            output[i].arithmetic = reduce(
                output[i].arithmetic + (dabit.arithmetic << bit),
                arithmetic_bits);
        }
    }
    return output;
}

}  // namespace secure_convert_detail

bool secure_convert_batch(const SecureConvertParams &params,
                          const std::vector<U128> &own_z,
                          PartyChannel &channel, PartyRandom &random,
                          std::vector<uint64_t> &own_r,
                          SecureConvertCounters &counters) {
    own_r.clear();
    counters = SecureConvertCounters{};

    int ell = 0;
    U128 modulus = 0;
    const bool locally_valid =
        (channel.party() == 0 || channel.party() == 1) &&
        validate_inputs(params, own_z, ell, modulus);
    const uint64_t preflight_bytes_before = channel.bytes_sent();
    const uint64_t preflight_switches_before = channel.direction_switches();
    const bool agreed =
        agree_convert_preflight(params, locally_valid, channel);
    counters.preflight_bytes_sent =
        channel.bytes_sent() - preflight_bytes_before;
    counters.preflight_direction_switches =
        channel.direction_switches() - preflight_switches_before;
    if (!agreed) {
        return false;
    }
    channel.setup_ots();

    const uint64_t correlation_bytes_before = channel.bytes_sent();
    const uint64_t correlation_switches_before = channel.direction_switches();
    const auto correlation_start = std::chrono::steady_clock::now();
    const std::vector<secure_convert_detail::Edabit> edabits =
        secure_convert_detail::generate_edabits(channel, params.count, ell,
                                                random);
    const std::vector<secure_convert_detail::Dabit> dabits =
        secure_convert_detail::generate_dabits(channel, params.count, params.bw,
                                               random);
    const size_t triple_count = params.count * size_t(2 * ell - 2);
    std::vector<BitTriple> triples;
    generate_bit_triples(channel, int(triple_count), random, triples);
    counters.correlation_microseconds =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - correlation_start)
            .count();
    counters.correlation_bytes_sent =
        channel.bytes_sent() - correlation_bytes_before;
    counters.correlation_direction_switches =
        channel.direction_switches() - correlation_switches_before;

    const uint64_t online_bytes_before = channel.bytes_sent();
    const uint64_t online_switches_before = channel.direction_switches();
    const auto online_start = std::chrono::steady_clock::now();
    OnlineCosts online_costs;
    if (!convert_with_correlations(params, modulus, ell, own_z, edabits, dabits,
                                   triples, channel, own_r, online_costs)) {
        own_r.clear();
        counters = SecureConvertCounters{};
        return false;
    }
    counters.online_microseconds =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - online_start)
            .count();
    counters.online_bytes_sent = channel.bytes_sent() - online_bytes_before;
    counters.online_direction_switches =
        channel.direction_switches() - online_switches_before;

    counters.conversions = params.count;
    counters.edabit_bits = params.count * size_t(ell);
    counters.dabits = params.count;
    counters.triples = online_costs.triples;
    counters.logical_opened_bits = online_costs.logical;
    counters.meaningful_share_bits = online_costs.sent + online_costs.received;
    counters.post_mask_dependencies = online_costs.post;
    return true;
}

}  // namespace ringlpn_2pc
