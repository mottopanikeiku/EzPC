#include "secure_truncate.h"

#include "secure_convert.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <limits>

namespace ringlpn_2pc {
namespace {

uint64_t mask_for_bits(int bits) {
    return bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
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

bool validate_inputs(const SecureTruncateParams &params,
                     const std::vector<uint64_t> &public_masked_input,
                     const std::vector<uint64_t> &own_mask_share,
                     const std::vector<uint64_t> &own_next_mask_share) {
    const bool correlation_id_nonzero =
        std::any_of(params.correlation_id.begin(), params.correlation_id.end(),
                    [](uint8_t byte) { return byte != 0; });
    if (params.sid == 0 || !correlation_id_nonzero || params.bw <= 2 ||
        params.bw > 32 || params.shift <= 0 || params.shift >= params.bw ||
        params.count == 0 || params.count > kMaxSecureTruncateBatch ||
        public_masked_input.size() != params.count ||
        own_mask_share.size() != params.count ||
        own_next_mask_share.size() != params.count) {
        return false;
    }

    const size_t triple_factor = size_t(2 * params.shift - 1);
    if (!multiply_fits_size(params.count, size_t(params.shift)) ||
        !multiply_fits_size(params.count, triple_factor) ||
        !multiply_fits_size(params.count, size_t(2)) ||
        params.count * size_t(params.shift) > size_t(INT_MAX) ||
        params.count * triple_factor > size_t(INT_MAX) ||
        params.count * size_t(2) > size_t(INT_MAX) ||
        params.count * sizeof(uint64_t) > size_t(INT_MAX)) {
        return false;
    }

    const uint64_t input_mask = mask_for_bits(params.bw);
    const uint64_t output_mask = mask_for_bits(params.bw - params.shift);
    for (size_t i = 0; i < params.count; ++i) {
        if ((public_masked_input[i] & ~input_mask) != 0 ||
            (own_mask_share[i] & ~input_mask) != 0 ||
            (own_next_mask_share[i] & ~output_mask) != 0) {
            return false;
        }
    }
    return true;
}

bool agree_preflight(const SecureTruncateParams &params, bool locally_valid,
                     PartyChannel &channel) {
    // Canonical public context followed by one local-validity byte. Both peers
    // exchange this record even on local failure, preventing one-sided hangs.
    std::array<uint8_t, 57> mine{};
    std::array<uint8_t, 57> peer{};
    store_u64_le(mine.data(), params.sid);
    store_u32_le(mine.data() + 8, uint32_t(params.bw));
    store_u32_le(mine.data() + 12, uint32_t(params.shift));
    store_u64_le(mine.data() + 16, uint64_t(params.count));
    std::copy(params.correlation_id.begin(), params.correlation_id.end(),
              mine.begin() + 24);
    mine[56] = locally_valid ? 1 : 0;
    channel.exchange_bytes(mine.data(), peer.data(), mine.size());
    for (size_t i = 0; i < 56; ++i) {
        if (mine[i] != peer[i]) return false;
    }
    return mine[56] == 1 && peer[56] == 1;
}

struct OnlineCosts {
    uint64_t triples = 0;
    uint64_t logical = 0;
    uint64_t sent = 0;
    uint64_t received = 0;
    uint64_t rounds = 0;
    uint64_t post = 0;
};

void and_batch(const std::vector<uint8_t> &x,
               const std::vector<uint8_t> &y,
               const std::vector<BitTriple> &triples, size_t &position,
               PartyChannel &channel, std::vector<uint8_t> &mine,
               std::vector<uint8_t> &peer, std::vector<uint8_t> &output,
               OnlineCosts &costs) {
    const size_t count = x.size();
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
    ++costs.rounds;
    costs.post += count;
}

void add_private_low_words(const std::vector<uint64_t> &own_mask_share,
                           int shift,
                           const std::vector<BitTriple> &triples,
                           size_t &triple_position, PartyChannel &channel,
                           std::vector<uint8_t> &sum_bits,
                           std::vector<uint8_t> &carry, OnlineCosts &costs) {
    const size_t count = own_mask_share.size();
    sum_bits.assign(count * size_t(shift), 0);
    carry.assign(count, 0);
    std::vector<uint8_t> input(count), x(count), y(count), next(count);
    std::vector<uint8_t> mine(count), peer(count);

    // carry' = ((x_i XOR carry) AND (x_{1-i} XOR carry)) XOR carry.
    // The two AND inputs below are XOR shares of those two operands.
    for (int bit = 0; bit < shift; ++bit) {
        for (size_t i = 0; i < count; ++i) {
            input[i] = uint8_t((own_mask_share[i] >> bit) & 1);
            sum_bits[i * size_t(shift) + size_t(bit)] =
                uint8_t(input[i] ^ carry[i]);
            x[i] = channel.is_p0() ? uint8_t(input[i] ^ carry[i]) : carry[i];
            y[i] = channel.is_p0() ? carry[i] : uint8_t(input[i] ^ carry[i]);
        }
        and_batch(x, y, triples, triple_position, channel, mine, peer, next,
                  costs);
        for (size_t i = 0; i < count; ++i) next[i] ^= carry[i];
        carry.swap(next);
    }
}

void add_public_secret_carry(
    const std::vector<uint64_t> &public_values,
    const std::vector<uint8_t> &secret_bits, int bits,
    const std::vector<BitTriple> &triples, size_t &triple_position,
    PartyChannel &channel, std::vector<uint8_t> &carry, OnlineCosts &costs) {
    const size_t count = public_values.size();
    carry.assign(count, 0);
    std::vector<uint8_t> input(count), next(count), mine(count), peer(count);

    // The final carry of public+secret is [public+secret >= 2^bits].
    // `public_values` supplies public bits; `secret_bits` supplies XOR shares.
    for (int bit = 0; bit < bits; ++bit) {
        for (size_t i = 0; i < count; ++i) {
            input[i] = secret_bits[i * size_t(bits) + size_t(bit)];
        }
        if (bit == 0) {
            std::fill(next.begin(), next.end(), uint8_t(0));
        } else {
            and_batch(input, carry, triples, triple_position, channel, mine,
                      peer, next, costs);
        }
        for (size_t i = 0; i < count; ++i) {
            if ((public_values[i] >> bit) & 1) {
                next[i] ^= uint8_t(input[i] ^ carry[i]);
            }
        }
        carry.swap(next);
    }
}

void xor_bits_to_arithmetic(
    const std::vector<uint8_t> &first_bits,
    const std::vector<uint8_t> &second_bits,
    const std::vector<secure_convert_detail::Dabit> &dabits, int output_bits,
    PartyChannel &channel, std::vector<uint64_t> &first_arithmetic,
    std::vector<uint64_t> &second_arithmetic, OnlineCosts &costs) {
    const size_t count = first_bits.size();
    std::vector<uint8_t> mine(2 * count), peer(2 * count);
    for (size_t i = 0; i < count; ++i) {
        mine[2 * i] = uint8_t((first_bits[i] ^ dabits[2 * i].bit) & 1);
        mine[2 * i + 1] =
            uint8_t((second_bits[i] ^ dabits[2 * i + 1].bit) & 1);
    }
    channel.exchange_bytes(mine.data(), peer.data(), mine.size());

    const uint64_t output_mask = mask_for_bits(output_bits);
    first_arithmetic.resize(count);
    second_arithmetic.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t positions[2] = {2 * i, 2 * i + 1};
        uint64_t *outputs[2] = {&first_arithmetic[i], &second_arithmetic[i]};
        for (int which = 0; which < 2; ++which) {
            const size_t position = positions[which];
            const uint8_t opened = uint8_t((mine[position] ^ peer[position]) & 1);
            const uint64_t arithmetic =
                uint64_t(dabits[position].arithmetic) & output_mask;
            *outputs[which] =
                opened == 0
                    ? arithmetic
                    : ((channel.is_p0() ? uint64_t(1) : uint64_t(0)) -
                       arithmetic) &
                          output_mask;
        }
    }
    costs.logical += 2 * count;
    costs.sent += 2 * count;
    costs.received += 2 * count;
    ++costs.rounds;
    costs.post += 2 * count;
}

bool truncate_with_correlations(
    const SecureTruncateParams &params,
    const std::vector<uint64_t> &public_masked_input,
    const std::vector<uint64_t> &own_mask_share,
    const std::vector<secure_convert_detail::Edabit> &uniform_masks,
    const std::vector<secure_convert_detail::Dabit> &dabits,
    const std::vector<BitTriple> &triples, PartyChannel &channel,
    std::vector<uint64_t> &own_output_share, OnlineCosts &costs,
    std::vector<uint64_t> *opened_t_trace) {
    const size_t count = params.count;
    const uint64_t low_mask = mask_for_bits(params.shift);
    const int output_bits = params.bw - params.shift;
    const uint64_t output_mask = mask_for_bits(output_bits);

    size_t triple_position = 0;
    std::vector<uint8_t> mask_low_bits;
    std::vector<uint8_t> mask_low_carry;
    add_private_low_words(own_mask_share, params.shift, triples,
                          triple_position, channel, mask_low_bits,
                          mask_low_carry, costs);

    // u is a fresh edaBit: uniform modulo 2^shift, arithmetically shared, and
    // unknown to either party. Opening t therefore one-time-pads r's low bits.
    std::vector<uint64_t> t_mine(count), t_peer(count), t(count);
    for (size_t i = 0; i < count; ++i) {
        t_mine[i] = (uint64_t(uniform_masks[i].arithmetic) +
                     (own_mask_share[i] & low_mask)) &
                    low_mask;
    }
    channel.exchange_bytes(reinterpret_cast<const uint8_t *>(t_mine.data()),
                           reinterpret_cast<uint8_t *>(t_peer.data()),
                           count * sizeof(uint64_t));
    for (size_t i = 0; i < count; ++i) {
        t[i] = (t_mine[i] + t_peer[i]) & low_mask;
    }
    if (opened_t_trace != nullptr) *opened_t_trace = t;
    costs.logical += count * size_t(params.shift);
    costs.sent += count * size_t(params.shift);
    costs.received += count * size_t(params.shift);
    ++costs.rounds;
    costs.post += count;

    // Let r_low be the reconstructed low mask, u the edaBit value, and
    // t=(u+r_low) mod 2^shift. The carry of
    // (2^shift-1-t)+r_low is [t<r_low], exactly the wrap bit of u+r_low.
    std::vector<uint64_t> comparison_addend(count);
    for (size_t i = 0; i < count; ++i) {
        comparison_addend[i] = low_mask - t[i];
    }
    std::vector<uint8_t> wrapped_low;
    add_public_secret_carry(comparison_addend, mask_low_bits, params.shift,
                            triples, triple_position, channel, wrapped_low,
                            costs);

    std::vector<uint64_t> carry_arithmetic;
    std::vector<uint64_t> wrapped_arithmetic;
    xor_bits_to_arithmetic(mask_low_carry, wrapped_low, dabits, output_bits,
                           channel, carry_arithmetic, wrapped_arithmetic,
                           costs);

    // Define rho=2^shift-1-u, which is uniform on the truncation interval.
    // Since u+r_low=t+wrapped_low*2^shift,
    //
    // floor((y_low-r_low+rho)/2^shift)
    //     = [y_low>t] - wrapped_low.
    //
    // The public expression 1-[y_low<=t] is [y_low>t]; the two arithmetic
    // carry shares subtract wrapped_low without revealing it.

    own_output_share.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const uint64_t y_low = public_masked_input[i] & low_mask;
        const uint64_t public_round_correction = y_low <= t[i] ? 1 : 0;
        const uint64_t public_term =
            channel.is_p0()
                ? ((public_masked_input[i] >> params.shift) + 1 -
                   public_round_correction) &
                      output_mask
                : 0;
        own_output_share[i] =
            (public_term - (own_mask_share[i] >> params.shift) -
             carry_arithmetic[i] - wrapped_arithmetic[i]) &
            output_mask;
    }
    return triple_position == triples.size();
}

}  // namespace

bool validate_secure_truncate_inputs(
    const SecureTruncateParams &params,
    const std::vector<uint64_t> &public_masked_input,
    const std::vector<uint64_t> &own_mask_share,
    const std::vector<uint64_t> &own_next_mask_share) {
    return validate_inputs(params, public_masked_input, own_mask_share,
                           own_next_mask_share);
}

bool secure_stochastic_truncate_batch(
    const SecureTruncateParams &params,
    const std::vector<uint64_t> &public_masked_input,
    const std::vector<uint64_t> &own_mask_share,
    const std::vector<uint64_t> &own_next_mask_share, PartyChannel &channel,
    PartyRandom &random, std::vector<uint64_t> &own_output_share,
    std::vector<uint64_t> &public_next_state,
    SecureTruncateCounters &counters,
    std::vector<uint64_t> *opened_t_trace) {
    own_output_share.clear();
    public_next_state.clear();
    counters = SecureTruncateCounters{};
    if (opened_t_trace != nullptr) opened_t_trace->clear();

    const bool locally_valid =
        (channel.party() == 0 || channel.party() == 1) &&
        validate_inputs(params, public_masked_input, own_mask_share,
                        own_next_mask_share);
    const uint64_t preflight_bytes_before = channel.bytes_sent();
    const uint64_t preflight_switches_before = channel.direction_switches();
    const bool agreed = agree_preflight(params, locally_valid, channel);
    counters.preflight_bytes_sent =
        channel.bytes_sent() - preflight_bytes_before;
    counters.preflight_direction_switches =
        channel.direction_switches() - preflight_switches_before;
    if (!agreed) return false;

    channel.setup_ots();
    const uint64_t correlation_bytes_before = channel.bytes_sent();
    const uint64_t correlation_switches_before = channel.direction_switches();
    const auto correlation_start = std::chrono::steady_clock::now();
    const std::vector<secure_convert_detail::Edabit> uniform_masks =
        secure_convert_detail::generate_edabits(channel, params.count,
                                                params.shift, random);
    const std::vector<secure_convert_detail::Dabit> dabits =
        secure_convert_detail::generate_dabits(channel, 2 * params.count,
                                               params.bw - params.shift,
                                               random);
    const size_t triple_count =
        params.count * size_t(2 * params.shift - 1);
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
    OnlineCosts costs;
    if (!truncate_with_correlations(
            params, public_masked_input, own_mask_share, uniform_masks, dabits,
            triples, channel, own_output_share, costs, opened_t_trace)) {
        own_output_share.clear();
        public_next_state.clear();
        counters = SecureTruncateCounters{};
        return false;
    }
    const int output_bits = params.bw - params.shift;
    const uint64_t output_mask = mask_for_bits(output_bits);
    std::vector<uint64_t> next_mine(params.count);
    std::vector<uint64_t> next_peer(params.count);
    public_next_state.resize(params.count);
    for (size_t i = 0; i < params.count; ++i) {
        next_mine[i] =
            (own_output_share[i] + own_next_mask_share[i]) & output_mask;
    }
    channel.exchange_bytes(
        reinterpret_cast<const uint8_t *>(next_mine.data()),
        reinterpret_cast<uint8_t *>(next_peer.data()),
        params.count * sizeof(uint64_t));
    for (size_t i = 0; i < params.count; ++i) {
        public_next_state[i] = (next_mine[i] + next_peer[i]) & output_mask;
    }
    costs.logical += params.count * size_t(output_bits);
    costs.sent += params.count * size_t(output_bits);
    costs.received += params.count * size_t(output_bits);
    ++costs.rounds;
    costs.post += params.count;
    counters.online_microseconds =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - online_start)
            .count();
    counters.online_bytes_sent = channel.bytes_sent() - online_bytes_before;
    counters.online_direction_switches =
        channel.direction_switches() - online_switches_before;

    counters.truncations = params.count;
    counters.handoffs = params.count;
    counters.edabit_bits = params.count * size_t(params.shift);
    counters.dabits = 2 * params.count;
    counters.triples = costs.triples;
    counters.logical_opened_bits = costs.logical;
    counters.meaningful_share_bits = costs.sent + costs.received;
    counters.online_dependency_rounds = costs.rounds;
    counters.post_mask_dependencies = costs.post;
    return true;
}

}  // namespace ringlpn_2pc
