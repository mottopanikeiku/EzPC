// two_party_dpf_protocol.h - the batched two-party DPF key generation protocol,
// shared by the standalone keygen artifact and the Ring-LPN SPFSS keygen.
//
// The transport/correction-word driver is shared by the explicit CPU baseline
// and the party-owned GPU frontier implementation in two_party_dpf_gpu.cuh.
// Both therefore retain the standalone artifact's exact correlation and
// opening accounting while differing only in private frontier execution.
//
// Every cross-party value goes through OT, a consume-once OLE source, or an
// explicitly counted opening. Trees are processed level-synchronously: one OT
// batch and one opening dependency stage per level, plus two Phase-C
// multiplication dependencies that produce three scalar products per tree.
// The measured direction-switch count depends on tree depth and backend, not
// batch size; it is not a network-round measurement.
//
// The expansion PRG is selectable: `kSplitmix` matches the unchanged host
// evaluator `spfss_host::dpfEvalAll`; `kGpuAes` is the bit-identical twin of the
// deployed four-call AES PRG with full 128-bit child seeds and independently
// derived control bits. The latter removes D-SEED's 127-bit encoding defect;
// P-RNG/P-DIST/P-KEY and the full reduction remain open.

#pragma once

#include "gpu_aes_prg_host.h"
#include "spfss_host.h"
#include "two_party_ot.h"

#include <cstdint>
#include <chrono>
#include <cstring>
#include <limits>
#include <vector>

namespace ringlpn_2pdpf {


using ringlpn_2pc::BitTriple;
using ringlpn_2pc::PartyChannel;
using ringlpn_2pc::PartyRandom;
using ringlpn_2pc::U128;
using ringlpn_2pc::Word;
using ringlpn_2pc::mod_add;
using ringlpn_2pc::mod_mul;
using ringlpn_2pc::mod_sub;

constexpr Word kPrime62 = 4611686018326724609ULL;      // 2^62 - 6*2^24 + 1
constexpr Word kPrime62Crt2 = 4611686018309947393ULL;  // 2^62 - 7*2^24 + 1

// PRG / convert: identical to spfss_host.cpp's file-local versions. Validation
// through the unchanged evaluator guards against drift.

inline uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline U128 make_u128(uint64_t lo, uint64_t hi) {
    return (U128)lo | ((U128)hi << 64);
}

inline void prg_expand(U128 seed, U128 &sL, uint8_t &tL, U128 &sR, uint8_t &tR) {
    uint64_t s0 = (uint64_t)seed ^ 0xA24BAED4963EE407ULL;
    uint64_t s1 = (uint64_t)(seed >> 64) ^ 0x9FB21C651E98DF25ULL;
    uint64_t a = splitmix64(s0);
    uint64_t b = splitmix64(s0);
    uint64_t c = splitmix64(s1);
    uint64_t d = splitmix64(s1);
    uint64_t tag = splitmix64(s0) ^ splitmix64(s1);
    sL = make_u128(a, b);
    sR = make_u128(c, d);
    tL = (uint8_t)(tag & 1);
    tR = (uint8_t)((tag >> 1) & 1);
}

inline Word convert_zp(U128 s, Word m) {
    Word lo = (Word)((uint64_t)s % m);
    Word hi = (Word)((uint64_t)(s >> 64) % m);
    return mod_add(lo, hi, m);
}

struct Node {
    U128 s;
    uint8_t t;
};

// Expansion PRG selector. `splitmix` matches the unchanged host evaluator
// `spfss_host::dpfEvalAll`; `gpu_aes` is the bit-identical twin of the deployed
// four-call GPU device PRG (src/gpu_aes_prg_host.h), so its keys are consumable
// by the GPU evaluator. The protocol transcript and accounting are unchanged.
enum class PrgMode { kSplitmix, kGpuAes };

inline void expand_node(PrgMode mode, U128 seed, U128 &sL, uint8_t &tL,
                        U128 &sR, uint8_t &tR) {
    if (mode == PrgMode::kGpuAes) {
        ringlpn_gpu_prg::gpu_aes_prg_expand(seed, sL, tL, sR, tR);
    } else {
        prg_expand(seed, sL, tL, sR, tR);
    }
}

// ----- the batched party-local protocol -------------------------------------
//
// The protocol owns transport and public correction words. A PartyTreeBatchState
// owns exactly one party's private frontier. This split lets the publication
// path keep roots/frontiers on that party's GPU without ever materializing the
// peer's state or changing the DPFKey wire/evaluator ABI.

struct DpfStageCounters {
    double phase_a_microseconds = 0.0;
    double phase_b_microseconds = 0.0;
    double phase_c_microseconds = 0.0;
    uint64_t phase_a_dependency_rounds = 0;
    uint64_t phase_b_dependency_rounds = 0;
    uint64_t phase_c_dependency_rounds = 0;
    uint64_t gpu_kernel_launches = 0;
    uint64_t gpu_h2d_bytes = 0;
    uint64_t gpu_d2h_bytes = 0;
    uint64_t gpu_peak_bytes = 0;
    uint64_t level_synchronizations = 0;
};

// Supplies additive shares of pairwise products of the parties' local inputs.
// A source must consume one independent Z_p OLE per input, exactly once, and
// account for every masked opening it sends through `channel`.
class PhaseCOleSource {
  public:
    virtual ~PhaseCOleSource() = default;
    virtual bool multiply(PartyChannel &channel,
                          const std::vector<Word> &local_inputs, Word modulus,
                          std::vector<Word> &product_shares) = 0;
};

inline bool phase_c_multiply(
    PhaseCOleSource *source, PartyChannel &channel,
    const std::vector<Word> &local_inputs, Word modulus, PartyRandom &random,
    std::vector<Word> &product_shares) {
    if (source != nullptr) {
        return source->multiply(channel, local_inputs, modulus,
                                product_shares) &&
               product_shares.size() == local_inputs.size();
    }
    product_shares =
        ringlpn_2pc::ole_batch_p0_sender(channel, local_inputs, modulus, random);
    return product_shares.size() == local_inputs.size();
}

class PartyTreeBatchState {
  public:
    virtual ~PartyTreeBatchState() = default;

    // A state object is consume-once. roots contains only this party's roots.
    virtual bool initialize(int party, int log_domain, Word modulus, PrgMode prg,
                            const std::vector<U128> &roots) = 0;
    virtual bool consumed() const = 0;
    virtual bool expand_level(int level, std::vector<U128> &aggregate_left,
                              std::vector<U128> &aggregate_right,
                              std::vector<uint8_t> &aggregate_t_left,
                              std::vector<uint8_t> &aggregate_t_right) = 0;
    virtual bool apply_level_correction(
        int level, const std::vector<U128> &seed_cw,
        const std::vector<uint8_t> &t_left_cw,
        const std::vector<uint8_t> &t_right_cw) = 0;
    virtual bool final_sums(std::vector<Word> &seed_sum,
                            std::vector<Word> &control_sum) = 0;
    virtual void add_backend_counters(DpfStageCounters &) const {}
};

// Contiguous host implementation retained only as the explicitly named
// comparison baseline. It deliberately shares the same protocol driver as the
// GPU implementation, so transport/correction-word semantics cannot drift.
class CpuBaselinePartyTreeBatchState final : public PartyTreeBatchState {
  public:
    bool initialize(int party, int log_domain, Word modulus, PrgMode prg,
                    const std::vector<U128> &roots) override {
        if (consumed_ || (party != 0 && party != 1) || log_domain < 2 ||
            roots.empty()) {
            return false;
        }
        consumed_ = true;
        party_ = party;
        log_domain_ = log_domain;
        modulus_ = modulus;
        prg_ = prg;
        batch_ = roots.size();
        width_ = 1;
        level_ = 0;
        current_.resize(batch_);
        for (size_t tree = 0; tree < batch_; ++tree) {
            current_[tree] = Node{roots[tree], static_cast<uint8_t>(party)};
        }
        return true;
    }
    bool consumed() const override { return consumed_; }

    bool expand_level(int level, std::vector<U128> &aggregate_left,
                      std::vector<U128> &aggregate_right,
                      std::vector<uint8_t> &aggregate_t_left,
                      std::vector<uint8_t> &aggregate_t_right) override {
        if (level != level_ || level_ >= log_domain_ ||
            current_.size() != batch_ * width_) {
            return false;
        }
        aggregate_left.assign(batch_, 0);
        aggregate_right.assign(batch_, 0);
        aggregate_t_left.assign(batch_, 0);
        aggregate_t_right.assign(batch_, 0);
        next_.resize(batch_ * width_ * 2);
        for (size_t tree = 0; tree < batch_; ++tree) {
            const size_t current_base = tree * width_;
            const size_t next_base = tree * width_ * 2;
            for (size_t node = 0; node < width_; ++node) {
                U128 left, right;
                uint8_t t_left, t_right;
                expand_node(prg_, current_[current_base + node].s, left, t_left,
                            right, t_right);
                aggregate_left[tree] ^= left;
                aggregate_right[tree] ^= right;
                aggregate_t_left[tree] ^= t_left;
                aggregate_t_right[tree] ^= t_right;
                next_[next_base + 2 * node] = Node{left, t_left};
                next_[next_base + 2 * node + 1] = Node{right, t_right};
            }
        }
        return true;
    }

    bool apply_level_correction(
        int level, const std::vector<U128> &seed_cw,
        const std::vector<uint8_t> &t_left_cw,
        const std::vector<uint8_t> &t_right_cw) override {
        if (level != level_ || seed_cw.size() != batch_ ||
            t_left_cw.size() != batch_ || t_right_cw.size() != batch_ ||
            next_.size() != batch_ * width_ * 2) {
            return false;
        }
        for (size_t tree = 0; tree < batch_; ++tree) {
            const size_t current_base = tree * width_;
            const size_t next_base = tree * width_ * 2;
            for (size_t node = 0; node < width_; ++node) {
                if (current_[current_base + node].t == 0) continue;
                Node &left = next_[next_base + 2 * node];
                Node &right = next_[next_base + 2 * node + 1];
                left.s ^= seed_cw[tree];
                right.s ^= seed_cw[tree];
                left.t = static_cast<uint8_t>(left.t ^ t_left_cw[tree]);
                right.t = static_cast<uint8_t>(right.t ^ t_right_cw[tree]);
            }
        }
        current_.swap(next_);
        width_ *= 2;
        ++level_;
        return true;
    }

    bool final_sums(std::vector<Word> &seed_sum,
                    std::vector<Word> &control_sum) override {
        if (level_ != log_domain_ ||
            current_.size() != batch_ * width_) {
            return false;
        }
        seed_sum.assign(batch_, 0);
        control_sum.assign(batch_, 0);
        for (size_t tree = 0; tree < batch_; ++tree) {
            const size_t base = tree * width_;
            for (size_t node = 0; node < width_; ++node) {
                const Node &value = current_[base + node];
                const Word converted = convert_zp(value.s, modulus_);
                if (party_ == 0) {
                    seed_sum[tree] = mod_add(seed_sum[tree], converted, modulus_);
                    control_sum[tree] =
                        mod_add(control_sum[tree], value.t, modulus_);
                } else {
                    seed_sum[tree] = mod_sub(seed_sum[tree], converted, modulus_);
                    control_sum[tree] =
                        mod_sub(control_sum[tree], value.t, modulus_);
                }
            }
        }
        return true;
    }

  private:
    bool consumed_ = false;
    int party_ = -1;
    int log_domain_ = 0;
    int level_ = 0;
    Word modulus_ = 0;
    PrgMode prg_ = PrgMode::kSplitmix;
    size_t batch_ = 0;
    size_t width_ = 0;
    std::vector<Node> current_;
    std::vector<Node> next_;
};

inline double elapsed_microseconds(
    const std::chrono::steady_clock::time_point &start) {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - start)
        .count();
}

// Every cross-party value goes through OT or an explicitly counted opening.
// Trees advance level-synchronously. `state` contains only this party's roots
// and frontier and is consumed once.
inline bool two_party_dpf_gen_batch_with_state(
    int party, int log_domain, Word p, PrgMode prg,
    const std::vector<uint64_t> &offs,
    const std::vector<Word> &beta_factors, PartyChannel &ch, PartyRandom &rng,
    PartyTreeBatchState &state, std::vector<spfss_host::DPFKey> &keys,
    PhaseCOleSource *phase_c_ole_source,
    DpfStageCounters *stage_counters = nullptr) {
    keys.clear();
    if (stage_counters != nullptr) *stage_counters = DpfStageCounters{};
    const int L = log_domain;
    const size_t B = offs.size();
    const size_t max_int =
        static_cast<size_t>(std::numeric_limits<int>::max());
    if ((party != 0 && party != 1) || L < 2 || L > 20 ||
        (p != kPrime62 && p != kPrime62Crt2) || B == 0 ||
        beta_factors.size() != B || B > max_int / size_t(L - 1) ||
        state.consumed()) {
        return false;
    }
    const uint64_t half_domain = 1ULL << (L - 1);
    for (size_t tree = 0; tree < B; ++tree) {
        // Validation precedes every consume-once correlation and GPU action.
        if (offs[tree] >= half_domain || beta_factors[tree] == 0 ||
            beta_factors[tree] >= p) {
            return false;
        }
    }

    const auto phase_a_start = std::chrono::steady_clock::now();
    std::vector<BitTriple> triples;
    ringlpn_2pc::generate_bit_triples(ch, (int)(B * (size_t)(L - 1)), rng,
                                      triples);
    if (stage_counters != nullptr) {
        ++stage_counters->phase_a_dependency_rounds;
    }
    std::vector<uint8_t> abit(B * (size_t)L, 0);
    std::vector<uint8_t> carry(B, 0);
    std::vector<uint8_t> open_mine(2 * B), open_theirs(2 * B);
    for (int bit = 0; bit < L; ++bit) {
        for (size_t tree = 0; tree < B; ++tree) {
            const uint8_t input_bit =
                static_cast<uint8_t>((offs[tree] >> bit) & 1);
            abit[tree * (size_t)L + (size_t)bit] =
                static_cast<uint8_t>(input_bit ^ carry[tree]);
        }
        if (bit + 1 >= L) break;
        for (size_t tree = 0; tree < B; ++tree) {
            const uint8_t input_bit =
                static_cast<uint8_t>((offs[tree] >> bit) & 1);
            const uint8_t x_mine =
                party == 0 ? static_cast<uint8_t>(input_bit ^ carry[tree])
                           : carry[tree];
            const uint8_t y_mine =
                party == 0 ? carry[tree]
                           : static_cast<uint8_t>(input_bit ^ carry[tree]);
            const BitTriple &triple =
                triples[tree * (size_t)(L - 1) + (size_t)bit];
            open_mine[2 * tree] =
                static_cast<uint8_t>((x_mine ^ triple.a) & 1);
            open_mine[2 * tree + 1] =
                static_cast<uint8_t>((y_mine ^ triple.b) & 1);
        }
        ch.exchange_bytes(open_mine.data(), open_theirs.data(), 2 * B);
        if (stage_counters != nullptr) {
            ++stage_counters->phase_a_dependency_rounds;
        }
        ch.costs.phase_a.logical_bits += 2ULL * (uint64_t)B;
        ch.costs.phase_a.revealed_bits_sent += 2ULL * (uint64_t)B;
        ch.costs.phase_a.revealed_bits_recv += 2ULL * (uint64_t)B;
        for (size_t tree = 0; tree < B; ++tree) {
            const uint8_t d =
                static_cast<uint8_t>((open_mine[2 * tree] ^
                                      open_theirs[2 * tree]) &
                                     1);
            const uint8_t e =
                static_cast<uint8_t>((open_mine[2 * tree + 1] ^
                                      open_theirs[2 * tree + 1]) &
                                     1);
            const BitTriple &triple =
                triples[tree * (size_t)(L - 1) + (size_t)bit];
            const uint8_t and_share = static_cast<uint8_t>(
                ((party == 0 ? static_cast<uint8_t>(d & e)
                             : static_cast<uint8_t>(0)) ^
                 static_cast<uint8_t>(d & triple.b) ^
                 static_cast<uint8_t>(e & triple.a) ^ triple.c) &
                1);
            carry[tree] = static_cast<uint8_t>(and_share ^ carry[tree]);
        }
    }
    if (stage_counters != nullptr) {
        stage_counters->phase_a_microseconds =
            elapsed_microseconds(phase_a_start);
    }

    const auto phase_b_start = std::chrono::steady_clock::now();
    std::vector<spfss_host::DPFKey> made_keys(B);
    std::vector<U128> roots(B);
    for (size_t tree = 0; tree < B; ++tree) {
        spfss_host::DPFKey &key = made_keys[tree];
        key.log_domain = L;
        key.modulus = p;
        key.seed = rng.u128();
        key.t0 = static_cast<uint8_t>(party);
        key.sCW.assign((size_t)L, 0);
        key.tLCW.assign((size_t)L, 0);
        key.tRCW.assign((size_t)L, 0);
        roots[tree] = key.seed;
    }
    if (!state.initialize(party, L, p, prg, roots)) {
        if (stage_counters != nullptr) {
            stage_counters->phase_b_microseconds =
                elapsed_microseconds(phase_b_start);
        }
        return false;
    }

    std::vector<U128> aggregate_left, aggregate_right;
    std::vector<uint8_t> aggregate_t_left, aggregate_t_right;
    std::vector<U128> z(B), masks(B), ot_m0(B), ot_m1(B), ot_out(B);
    std::vector<uint8_t> choices(B);
    std::vector<uint8_t> level_mine(B * 17), level_theirs(B * 17);
    std::vector<U128> seed_cw(B);
    std::vector<uint8_t> t_left_cw(B), t_right_cw(B);
    for (int level = 0; level < L; ++level) {
        if (!state.expand_level(level, aggregate_left, aggregate_right,
                                aggregate_t_left, aggregate_t_right) ||
            aggregate_left.size() != B || aggregate_right.size() != B ||
            aggregate_t_left.size() != B ||
            aggregate_t_right.size() != B) {
            if (stage_counters != nullptr) {
                stage_counters->phase_b_microseconds =
                    elapsed_microseconds(phase_b_start);
            }
            return false;
        }
        const int alpha_bit_index = L - 1 - level;
        for (size_t tree = 0; tree < B; ++tree) {
            z[tree] = aggregate_left[tree] ^ aggregate_right[tree];
            masks[tree] = rng.u128();
            ot_m0[tree] = masks[tree];
            ot_m1[tree] = masks[tree] ^ z[tree];
            choices[tree] =
                abit[tree * (size_t)L + (size_t)alpha_bit_index];
        }
        if (party == 0) {
            ch.ot_send_128(ot_m0, ot_m1);
            ot_out = ch.ot_recv_128(choices);
        } else {
            ot_out = ch.ot_recv_128(choices);
            ch.ot_send_128(ot_m0, ot_m1);
        }
        if (stage_counters != nullptr) {
            ++stage_counters->phase_b_dependency_rounds;
        }
        for (size_t tree = 0; tree < B; ++tree) {
            const uint8_t alpha_share = choices[tree];
            const U128 seed_cw_share =
                aggregate_right[tree] ^
                (alpha_share ? z[tree] : static_cast<U128>(0)) ^ ot_out[tree] ^
                masks[tree];
            std::memcpy(&level_mine[tree * 17], &seed_cw_share, 16);
            const uint8_t left_share = static_cast<uint8_t>(
                (aggregate_t_left[tree] ^ alpha_share ^
                 (party == 0 ? 1u : 0u)) &
                1);
            const uint8_t right_share = static_cast<uint8_t>(
                (aggregate_t_right[tree] ^ alpha_share) & 1);
            level_mine[tree * 17 + 16] =
                static_cast<uint8_t>(left_share | (right_share << 1));
        }
        ch.exchange_bytes(level_mine.data(), level_theirs.data(), B * 17);
        if (stage_counters != nullptr) {
            ++stage_counters->phase_b_dependency_rounds;
        }
        ch.costs.phase_b.logical_bits += 130ULL * (uint64_t)B;
        ch.costs.phase_b.revealed_bits_sent += 130ULL * (uint64_t)B;
        ch.costs.phase_b.revealed_bits_recv += 130ULL * (uint64_t)B;
        for (size_t tree = 0; tree < B; ++tree) {
            U128 mine_cw = 0, peer_cw = 0;
            std::memcpy(&mine_cw, &level_mine[tree * 17], 16);
            std::memcpy(&peer_cw, &level_theirs[tree * 17], 16);
            seed_cw[tree] = mine_cw ^ peer_cw;
            const uint8_t flags = static_cast<uint8_t>(
                level_mine[tree * 17 + 16] ^
                level_theirs[tree * 17 + 16]);
            t_left_cw[tree] = static_cast<uint8_t>(flags & 1);
            t_right_cw[tree] = static_cast<uint8_t>((flags >> 1) & 1);
            made_keys[tree].sCW[(size_t)level] = seed_cw[tree];
            made_keys[tree].tLCW[(size_t)level] = t_left_cw[tree];
            made_keys[tree].tRCW[(size_t)level] = t_right_cw[tree];
        }
        if (!state.apply_level_correction(level, seed_cw, t_left_cw,
                                          t_right_cw)) {
            if (stage_counters != nullptr) {
                stage_counters->phase_b_microseconds =
                    elapsed_microseconds(phase_b_start);
            }
            return false;
        }
    }
    if (stage_counters != nullptr) {
        stage_counters->phase_b_microseconds =
            elapsed_microseconds(phase_b_start);
    }

    const auto phase_c_start = std::chrono::steady_clock::now();
    std::vector<Word> seed_sum, control_sum;
    if (!state.final_sums(seed_sum, control_sum) || seed_sum.size() != B ||
        control_sum.size() != B) {
        if (stage_counters != nullptr) {
            stage_counters->phase_c_microseconds =
                elapsed_microseconds(phase_c_start);
            state.add_backend_counters(*stage_counters);
        }
        return false;
    }
    std::vector<Word> gamma;
    if (!phase_c_multiply(phase_c_ole_source, ch, beta_factors, p, rng,
                          gamma)) {
        if (stage_counters != nullptr) {
            stage_counters->phase_c_microseconds =
                elapsed_microseconds(phase_c_start);
            state.add_backend_counters(*stage_counters);
        }
        return false;
    }
    if (stage_counters != nullptr) {
        ++stage_counters->phase_c_dependency_rounds;
    }
    std::vector<Word> d(B), s(B);
    for (size_t tree = 0; tree < B; ++tree) {
        d[tree] = mod_sub(gamma[tree], seed_sum[tree], p);
        s[tree] = control_sum[tree];
    }
    std::vector<Word> cross_inputs(2 * B);
    for (size_t tree = 0; tree < B; ++tree) {
        cross_inputs[tree] = party == 0 ? d[tree] : s[tree];
        cross_inputs[B + tree] = party == 0 ? s[tree] : d[tree];
    }
    std::vector<Word> cross;
    if (!phase_c_multiply(phase_c_ole_source, ch, cross_inputs, p, rng,
                          cross)) {
        if (stage_counters != nullptr) {
            stage_counters->phase_c_microseconds =
                elapsed_microseconds(phase_c_start);
            state.add_backend_counters(*stage_counters);
        }
        return false;
    }
    if (stage_counters != nullptr) {
        ++stage_counters->phase_c_dependency_rounds;
    }
    std::vector<uint64_t> final_mine(B), final_theirs(B);
    for (size_t tree = 0; tree < B; ++tree) {
        final_mine[tree] = static_cast<uint64_t>(mod_add(
            mod_add(mod_mul(d[tree], s[tree], p), cross[tree], p),
            cross[B + tree], p));
    }
    // Three OLEs expose neither d nor s. Only the standard public finalCW is
    // opened, exactly as in the stock DPFKey material.
    ch.exchange_bytes(reinterpret_cast<const uint8_t *>(final_mine.data()),
                      reinterpret_cast<uint8_t *>(final_theirs.data()),
                      B * sizeof(uint64_t));
    if (stage_counters != nullptr) {
        ++stage_counters->phase_c_dependency_rounds;
    }
    const uint64_t field_bit_count =
        static_cast<uint64_t>(ringlpn_2pc::field_bits(p));
    ch.costs.phase_c.logical_bits += field_bit_count * (uint64_t)B;
    ch.costs.phase_c.revealed_bits_sent += field_bit_count * (uint64_t)B;
    ch.costs.phase_c.revealed_bits_recv += field_bit_count * (uint64_t)B;
    for (size_t tree = 0; tree < B; ++tree) {
        made_keys[tree].finalCW =
            mod_add(static_cast<Word>(final_mine[tree]),
                    static_cast<Word>(final_theirs[tree]), p);
    }
    if (stage_counters != nullptr) {
        stage_counters->phase_c_microseconds =
            elapsed_microseconds(phase_c_start);
        state.add_backend_counters(*stage_counters);
    }
    keys = std::move(made_keys);
    return true;
}

inline bool two_party_dpf_gen_batch_cpu_baseline(
    int party, int log_domain, Word p, PrgMode prg,
    const std::vector<uint64_t> &offs,
    const std::vector<Word> &beta_factors, PartyChannel &ch, PartyRandom &rng,
    std::vector<spfss_host::DPFKey> &keys,
    PhaseCOleSource *phase_c_ole_source,
    DpfStageCounters *stage_counters = nullptr) {
    CpuBaselinePartyTreeBatchState state;
    return two_party_dpf_gen_batch_with_state(
        party, log_domain, p, prg, offs, beta_factors, ch, rng, state, keys,
        phase_c_ole_source, stage_counters);
}

}  // namespace ringlpn_2pdpf
