#pragma once

#include "two_party_ot.h"
#include <array>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ringlpn_2pc {
constexpr size_t kMaxSecureConvertBatch = size_t(1) << 16;

struct SecureConvertParams {
    uint64_t sid = 0;
    int qbits = 0;
    int bw = 0;
    size_t count = 0;
    // Full canonical correlation-scope ID. `sid` is retained only as a
    // compatibility/accounting handle; live callers must derive both from the
    // globally claimed invocation namespace. Standalone correctness baselines
    // may use an explicitly labelled deterministic ID.
    std::array<uint8_t, 32> correlation_id{};
};

struct SecureConvertCounters {
    uint64_t conversions = 0;
    uint64_t edabit_bits = 0;
    uint64_t dabits = 0;
    uint64_t triples = 0;
    uint64_t logical_opened_bits = 0;
    uint64_t meaningful_share_bits = 0;
    uint64_t post_mask_dependencies = 0;

    uint64_t preflight_bytes_sent = 0;
    uint64_t preflight_direction_switches = 0;
    // Per-phase transport deltas let callers account for the fixed correlation
    // and online schedules without exposing the generated correlations.
    uint64_t correlation_bytes_sent = 0;
    uint64_t correlation_direction_switches = 0;
    uint64_t online_bytes_sent = 0;
    uint64_t online_direction_switches = 0;
    double correlation_microseconds = 0;
    double online_microseconds = 0;
};

bool validate_secure_convert_inputs(const SecureConvertParams &params,
                                    const std::vector<U128> &own_z);

// Converts this party's canonical shares in Z_Q to its shares in Z_{2^bw}.
// Q is p0 for qbits=64 and p0*p1 for qbits=128. The function validates all
// public sizes and every local share before generating any OT correlation.
bool secure_convert_batch(const SecureConvertParams &params,
                          const std::vector<U128> &own_z, PartyChannel &channel,
                          PartyRandom &random, std::vector<uint64_t> &own_r,
                          SecureConvertCounters &counters);

// Test-only correlation checks need the same generators as production. These
// declarations expose individual local correlation shares but never reconstruct
// them; reconstruction remains in the test executable.
namespace secure_convert_detail {

struct Dabit {
    uint8_t bit = 0;
    U128 arithmetic = 0;
};

struct Edabit {
    U128 arithmetic = 0;
    std::vector<uint8_t> bits;
};

std::vector<Dabit> generate_dabits(PartyChannel &channel, size_t count,
                                   int arithmetic_bits, PartyRandom &random);
std::vector<Edabit> generate_edabits(PartyChannel &channel, size_t count,
                                     int arithmetic_bits,
                                     PartyRandom &random);

}  // namespace secure_convert_detail
}  // namespace ringlpn_2pc
