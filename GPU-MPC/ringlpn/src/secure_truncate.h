#pragma once

#include "two_party_ot.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ringlpn_2pc {

constexpr size_t kMaxSecureTruncateBatch = size_t(1) << 16;

struct SecureTruncateParams {
    uint64_t sid = 0;
    int bw = 0;
    int shift = 0;
    size_t count = 0;
    // Full consume-once scope for every edaBit, daBit, and Boolean triple used
    // by this invocation. The caller must derive it from the same globally
    // claimed invocation namespace as the surrounding linear-layer record.
    std::array<uint8_t, 32> correlation_id{};
};

struct SecureTruncateCounters {
    uint64_t truncations = 0;
    uint64_t handoffs = 0;
    uint64_t edabit_bits = 0;
    uint64_t dabits = 0;
    uint64_t triples = 0;
    uint64_t logical_opened_bits = 0;
    uint64_t meaningful_share_bits = 0;
    uint64_t online_dependency_rounds = 0;
    uint64_t post_mask_dependencies = 0;

    uint64_t preflight_bytes_sent = 0;
    uint64_t preflight_direction_switches = 0;
    uint64_t correlation_bytes_sent = 0;
    uint64_t correlation_direction_switches = 0;
    uint64_t online_bytes_sent = 0;
    uint64_t online_direction_switches = 0;
    double correlation_microseconds = 0;
    double online_microseconds = 0;
};

bool validate_secure_truncate_inputs(
    const SecureTruncateParams &params,
    const std::vector<uint64_t> &public_masked_input,
    const std::vector<uint64_t> &own_mask_share,
    const std::vector<uint64_t> &own_next_mask_share);

// Secure semi-honest OT-hybrid stochastic truncation over Z_{2^bw}.
//
// public_masked_input[j] is y=(x+r) mod 2^bw. Party i owns only its additive
// share r_i and a fresh additive share of the next-state mask s_i. The output
// shares reconstruct
//
//   floor((x + rho) / 2^shift) mod 2^(bw-shift),
//
// while public_next_state reconstructs that result plus s_0+s_1. The fresh
// next mask makes this the public representation expected by the next graph
// state. rho is fresh and uniform in [0,2^shift). The protocol opens only
// t=(u+(r mod 2^shift)) mod 2^shift, for a fresh secret uniform u, and the
// remasked next state. No party learns r, u, rho, the carries, the comparison
// bit, the clear truncated result, or the reconstructed next mask.
bool secure_stochastic_truncate_batch(
    const SecureTruncateParams &params,
    const std::vector<uint64_t> &public_masked_input,
    const std::vector<uint64_t> &own_mask_share,
    const std::vector<uint64_t> &own_next_mask_share, PartyChannel &channel,
    PartyRandom &random, std::vector<uint64_t> &own_output_share,
    std::vector<uint64_t> &public_next_state,
    SecureTruncateCounters &counters,
    std::vector<uint64_t> *opened_t_trace = nullptr);

}  // namespace ringlpn_2pc
