// two_party_ot.h - real two-party transport and correlated-randomness
// primitives for the ringlpn distributed DPF key generation (milestone M1,
// stage S4 "real OT/OLE transport").
//
// WHAT IS REAL HERE
//   * Two OS processes, two TCP sockets. No shared memory, no shared RNG seed,
//     no party reading the other party's state.
//   * Real 1-of-2 oblivious transfer: the IKNP OT extension over Naor-Pinkas
//     base OTs, used UNMODIFIED from this repository's SCI stack
//     (SCI/src/OT/split-iknp.h, SCI/src/OT/np.h, SCI/src/utils/*). Only
//     OpenSSL is linked; SEAL/GMP/libOTe are not needed.
//   * Boolean AND triples: two 1-bit OTs per triple (Gilboa cross terms).
//   * Z_p scalar OLE: Gilboa multiplication, ceil(log2(p-1)) OTs of field
//     elements per OLE.
//   * Party root seeds: OS CSPRNG (std::random_device), never a shared seed.
//   * Every wire byte and every direction switch is counted by sci::NetIO
//     (`counter`, `num_rounds`), per party and per socket.
//
// WHAT IS NOT CLAIMED
//   * IKNP is OT *extension*, not silent OT. Ferret/Silver-class silent OT is
//     future work; the measured byte counts here are therefore an upper bound
//     for the setup material, not a silent-OT figure.
//   * The DPF expansion PRG remains spfss_host's non-cryptographic splitmix64,
//     because the independent consumer (`spfss_host::dpfEvalAll`) is unchanged.
//     This artifact makes NO 128-bit DPF-security claim; see
//     results/reports/dealerless_orca_fc_security_contract_2026_07_29.md
//     obligations D-SEED, P-RNG, P-KEY.
//   * Semi-honest, authenticated point-to-point channels. Malicious security,
//     active attacks, and side channels are out of scope.

#pragma once

#include "OT/split-iknp.h"
#include "utils/net_io_channel.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ringlpn_2pc {

using Word = uint64_t;
using U128 = __uint128_t;

// ----- field helpers (same arithmetic as spfss_host / the host prototype) ----

inline Word mod_add(Word a, Word b, Word m) {
    Word s = a + b;
    if (s >= m || s < a) s -= m;
    return s;
}

inline Word mod_sub(Word a, Word b, Word m) {
    return a >= b ? a - b : m - (b - a);
}

inline Word mod_mul(Word a, Word b, Word m) {
    return (Word)(((__uint128_t)a * b) % m);
}

inline int field_bits(Word p) {
    int bits = 0;
    for (Word x = p - 1; x != 0; x >>= 1) ++bits;
    return bits;
}

inline sci::block128 to_block(U128 x) {
    uint64_t parts[2] = {(uint64_t)x, (uint64_t)(x >> 64)};
    sci::block128 b;
    std::memcpy(&b, parts, sizeof(b));
    return b;
}

inline U128 from_block(sci::block128 b) {
    uint64_t parts[2];
    std::memcpy(parts, &b, sizeof(b));
    return ((U128)parts[1] << 64) | (U128)parts[0];
}

// ----- transcript accounting -------------------------------------------------
//
// `logical_bits` counts the bits of each COMMON value the two parties
// reconstruct. `revealed_bits_sent` / `revealed_bits_recv` split the raw share
// payload by direction; their sum is the contract's `revealed_share_bits`,
// which the single-process prototype could only report as one number.

struct PhaseCosts {
    uint64_t logical_bits = 0;
    uint64_t revealed_bits_sent = 0;
    uint64_t revealed_bits_recv = 0;
};

struct Counters {
    uint64_t string_ots_128 = 0;  // 1-of-2 OT on 128-bit strings (Phase B MUX)
    uint64_t triple_ots = 0;      // 1-bit OTs consumed by AND triples
    uint64_t ole_ots = 0;         // field-element OTs consumed by Gilboa OLE
    uint64_t bit_triples = 0;     // AND triples produced
    uint64_t scalar_oles = 0;     // Z_p OLEs produced
    uint64_t base_ots = 0;        // Naor-Pinkas base OTs for the OT extension
    PhaseCosts phase_a, phase_b, phase_c;

    uint64_t logical_opened_bits() const {
        return phase_a.logical_bits + phase_b.logical_bits + phase_c.logical_bits;
    }
    uint64_t revealed_share_bits() const {
        return phase_a.revealed_bits_sent + phase_a.revealed_bits_recv +
               phase_b.revealed_bits_sent + phase_b.revealed_bits_recv +
               phase_c.revealed_bits_sent + phase_c.revealed_bits_recv;
    }
};

// ----- the party channel ----------------------------------------------------
//
// Party 0 is the TCP server and SCI's ALICE; party 1 connects and is BOB.
// Two sockets are used exactly as SCI's OTPack does: the "straight" channel
// carries OTs where party 0 is the sender, the "reversed" channel carries OTs
// where party 1 is the sender. A party therefore always sends on
// `sender_ot()` and receives on `receiver_ot()`, and the two directions must be
// scheduled in a fixed global order (party-0-sender first) to stay
// deadlock-free.

class PartyChannel {
  public:
    PartyChannel(int party, const std::string &peer_host, int base_port)
        : party_(party) {
        const char *addr = (party == 0) ? nullptr : peer_host.c_str();
        io_ = new sci::NetIO(addr, base_port, /*full_buffer=*/false, /*quiet=*/true);
        io_rev_ = new sci::NetIO(addr, base_port + 1, false, true);
        const int sci_party = (party == 0) ? sci::ALICE : sci::BOB;
        ot_straight_ = new sci::SplitIKNP<sci::NetIO>(sci_party, io_);
        ot_reversed_ = new sci::SplitIKNP<sci::NetIO>(3 - sci_party, io_rev_);
        if (party == 0) {
            ot_straight_->setup_send();
            ot_reversed_->setup_recv();
        } else {
            ot_straight_->setup_recv();
            ot_reversed_->setup_send();
        }
        io_->flush();
        io_rev_->flush();
        costs.base_ots += 2 * 128;  // one base-OT batch per direction
        setup_bytes_sent_ = io_->counter + io_rev_->counter;
        setup_rounds_ = io_->num_rounds + io_rev_->num_rounds;
    }

    ~PartyChannel() {
        delete ot_straight_;
        delete ot_reversed_;
        delete io_;
        delete io_rev_;
    }

    PartyChannel(const PartyChannel &) = delete;
    PartyChannel &operator=(const PartyChannel &) = delete;

    int party() const { return party_; }
    bool is_p0() const { return party_ == 0; }

    sci::SplitIKNP<sci::NetIO> *sender_ot() {
        return party_ == 0 ? ot_straight_ : ot_reversed_;
    }
    sci::SplitIKNP<sci::NetIO> *receiver_ot() {
        return party_ == 0 ? ot_reversed_ : ot_straight_;
    }
    sci::NetIO *sender_io() { return party_ == 0 ? io_ : io_rev_; }
    sci::NetIO *receiver_io() { return party_ == 0 ? io_rev_ : io_; }

    uint64_t bytes_sent() const { return io_->counter + io_rev_->counter; }
    uint64_t direction_switches() const {
        return io_->num_rounds + io_rev_->num_rounds;
    }
    uint64_t setup_bytes_sent() const { return setup_bytes_sent_; }
    uint64_t setup_rounds() const { return setup_rounds_; }

    void sync() {
        io_->sync();
        io_rev_->sync();
    }

    // Symmetric share exchange on the straight channel. Party 0 sends first so
    // the pair never blocks on itself.
    void exchange(const void *mine, void *theirs, int nbytes) {
        if (party_ == 0) {
            io_->send_data(mine, nbytes);
            io_->flush();
            io_->recv_data(theirs, nbytes);
        } else {
            io_->recv_data(theirs, nbytes);
            io_->send_data(mine, nbytes);
            io_->flush();
        }
    }

    // ---- openings (each records logical bits and both raw share directions) --

    uint8_t open_bits(uint8_t mine, int nbits, PhaseCosts &phase) {
        uint8_t theirs = 0;
        exchange(&mine, &theirs, 1);
        phase.logical_bits += (uint64_t)nbits;
        phase.revealed_bits_sent += (uint64_t)nbits;
        phase.revealed_bits_recv += (uint64_t)nbits;
        return (uint8_t)(mine ^ theirs);
    }

    U128 open_u128(U128 mine, PhaseCosts &phase) {
        uint8_t out[16], in[16];
        std::memcpy(out, &mine, 16);
        exchange(out, in, 16);
        U128 theirs = 0;
        std::memcpy(&theirs, in, 16);
        phase.logical_bits += 128;
        phase.revealed_bits_sent += 128;
        phase.revealed_bits_recv += 128;
        return mine ^ theirs;
    }

    Word open_field(Word mine, Word p, PhaseCosts &phase) {
        uint64_t theirs = 0;
        exchange(&mine, &theirs, 8);
        const uint64_t bits = (uint64_t)field_bits(p);
        phase.logical_bits += bits;
        phase.revealed_bits_sent += bits;
        phase.revealed_bits_recv += bits;
        return mod_add(mine, (Word)theirs, p);
    }

    // ---- 128-bit string OT (Phase B seed MUX) ------------------------------

    void ot_send_128(const std::vector<U128> &m0, const std::vector<U128> &m1) {
        const int n = (int)m0.size();
        std::vector<sci::block128> b0(n), b1(n);
        for (int i = 0; i < n; ++i) {
            b0[i] = to_block(m0[i]);
            b1[i] = to_block(m1[i]);
        }
        sender_ot()->send(b0.data(), b1.data(), n);
        sender_io()->flush();
        costs.string_ots_128 += (uint64_t)n;
    }

    std::vector<U128> ot_recv_128(const std::vector<uint8_t> &choices) {
        const int n = (int)choices.size();
        std::vector<uint8_t> raw(n);
        for (int i = 0; i < n; ++i) raw[i] = choices[i] != 0;
        std::vector<sci::block128> out(n);
        receiver_ot()->recv(out.data(), reinterpret_cast<const bool *>(raw.data()), n);
        receiver_io()->flush();
        costs.string_ots_128 += (uint64_t)n;
        std::vector<U128> res(n);
        for (int i = 0; i < n; ++i) res[i] = from_block(out[i]);
        return res;
    }

    // ---- 1-bit OT (AND-triple cross terms) ---------------------------------

    // SCI's l-bit OT wrappers expect one ROW of N=2 messages per instance
    // (`data[i][k]`, see SCI/src/OT/ot-utils.h pack_ot_messages), not two
    // parallel arrays.
    void ot_send_bits(const std::vector<uint8_t> &m0,
                      const std::vector<uint8_t> &m1) {
        const int n = (int)m0.size();
        std::vector<uint8_t> flat((size_t)2 * n);
        std::vector<uint8_t *> rows((size_t)n);
        for (int i = 0; i < n; ++i) {
            flat[(size_t)2 * i] = (uint8_t)(m0[(size_t)i] & 1);
            flat[(size_t)2 * i + 1] = (uint8_t)(m1[(size_t)i] & 1);
            rows[(size_t)i] = &flat[(size_t)2 * i];
        }
        sender_ot()->send(rows.data(), n, 1);
        sender_io()->flush();
        costs.triple_ots += (uint64_t)n;
    }

    std::vector<uint8_t> ot_recv_bits(const std::vector<uint8_t> &choices) {
        const int n = (int)choices.size();
        std::vector<uint8_t> out(n), sel(choices);
        receiver_ot()->recv(out.data(), sel.data(), n, 1);
        receiver_io()->flush();
        costs.triple_ots += (uint64_t)n;
        for (int i = 0; i < n; ++i) out[i] &= 1;
        return out;
    }

    // ---- field-element OT (Gilboa OLE) ------------------------------------

    void ot_send_field(const std::vector<Word> &m0, const std::vector<Word> &m1,
                       int l) {
        const int n = (int)m0.size();
        std::vector<uint64_t> flat((size_t)2 * n);
        std::vector<uint64_t *> rows((size_t)n);
        for (int i = 0; i < n; ++i) {
            flat[(size_t)2 * i] = (uint64_t)m0[(size_t)i];
            flat[(size_t)2 * i + 1] = (uint64_t)m1[(size_t)i];
            rows[(size_t)i] = &flat[(size_t)2 * i];
        }
        sender_ot()->send(rows.data(), n, l);
        sender_io()->flush();
        costs.ole_ots += (uint64_t)n;
    }

    std::vector<Word> ot_recv_field(const std::vector<uint8_t> &choices, int l) {
        const int n = (int)choices.size();
        std::vector<uint64_t> out(n);
        std::vector<uint8_t> sel(choices);
        receiver_ot()->recv(out.data(), sel.data(), n, l);
        receiver_io()->flush();
        costs.ole_ots += (uint64_t)n;
        return std::vector<Word>(out.begin(), out.end());
    }

    Counters costs;

  private:
    int party_;
    sci::NetIO *io_ = nullptr;
    sci::NetIO *io_rev_ = nullptr;
    sci::SplitIKNP<sci::NetIO> *ot_straight_ = nullptr;
    sci::SplitIKNP<sci::NetIO> *ot_reversed_ = nullptr;
    uint64_t setup_bytes_sent_ = 0;
    uint64_t setup_rounds_ = 0;
};

// ----- party-private randomness (OS CSPRNG, never shared) -------------------

class PartyRandom {
  public:
    PartyRandom() {
        std::random_device rd;
        std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        gen_.seed(seq);
    }
    explicit PartyRandom(uint64_t fixed_seed) : gen_(fixed_seed) {}

    uint64_t u64() { return gen_(); }
    uint8_t bit() { return (uint8_t)(gen_() & 1); }
    U128 u128() { return ((U128)gen_() << 64) | (U128)gen_(); }
    Word field(Word p) { return (Word)(((U128)gen_() << 64 | (U128)gen_()) % p); }

  private:
    std::mt19937_64 gen_;
};

// ----- boolean AND triples from two 1-bit OTs -------------------------------
//
// c = (a0^a1)&(b0^b1) = a0b0 ^ a1b1 ^ (a0&b1) ^ (a1&b0). Each party is sender
// once (input b_mine) and receiver once (choice a_mine), so the construction is
// symmetric; the schedule is party-0-sender first.

struct BitTriple {
    uint8_t a = 0, b = 0, c = 0;
};

inline void generate_bit_triples(PartyChannel &ch, int n, PartyRandom &rng,
                                 std::vector<BitTriple> &out) {
    out.assign(n, BitTriple{});
    std::vector<uint8_t> a(n), b(n), mask(n), m0(n), m1(n), choice(n);
    for (int i = 0; i < n; ++i) {
        a[i] = rng.bit();
        b[i] = rng.bit();
        mask[i] = rng.bit();
        m0[i] = mask[i];
        m1[i] = (uint8_t)(mask[i] ^ b[i]);
        choice[i] = a[i];
    }
    std::vector<uint8_t> received;
    if (ch.is_p0()) {
        ch.ot_send_bits(m0, m1);
        received = ch.ot_recv_bits(choice);
    } else {
        received = ch.ot_recv_bits(choice);
        ch.ot_send_bits(m0, m1);
    }
    for (int i = 0; i < n; ++i) {
        out[i].a = a[i];
        out[i].b = b[i];
        out[i].c = (uint8_t)(((a[i] & b[i]) ^ mask[i] ^ received[i]) & 1);
    }
    ch.costs.bit_triples += (uint64_t)n;
}

// ----- Gilboa Z_p scalar OLE ------------------------------------------------
//
// Sender holds x, receiver holds y; the pair ends with additive shares
// u + v = x*y mod p. Cost: ceil(log2(p-1)) OTs of field elements.

inline Word ole_send(PartyChannel &ch, Word x, Word p, PartyRandom &rng) {
    const int k = field_bits(p);
    std::vector<Word> m0(k), m1(k);
    Word acc = 0;
    Word shifted = x % p;
    for (int j = 0; j < k; ++j) {
        const Word r = rng.field(p);
        m0[j] = r;
        m1[j] = mod_add(r, shifted, p);
        acc = mod_add(acc, r, p);
        shifted = mod_add(shifted, shifted, p);  // 2^(j+1) * x mod p
    }
    ch.ot_send_field(m0, m1, k);
    ch.costs.scalar_oles += 1;
    return mod_sub(0, acc, p);  // u = -sum(r) mod p
}

inline Word ole_recv(PartyChannel &ch, Word y, Word p) {
    const int k = field_bits(p);
    std::vector<uint8_t> choice(k);
    for (int j = 0; j < k; ++j) choice[j] = (uint8_t)((y >> j) & 1);
    const std::vector<Word> got = ch.ot_recv_field(choice, k);
    Word v = 0;
    for (int j = 0; j < k; ++j) v = mod_add(v, got[j] % p, p);
    ch.costs.scalar_oles += 1;
    return v;
}

// Directional wrapper: party 0 is always the OLE sender in this protocol.
inline Word ole_p0_sender(PartyChannel &ch, Word my_input, Word p,
                          PartyRandom &rng) {
    return ch.is_p0() ? ole_send(ch, my_input, p, rng)
                      : ole_recv(ch, my_input, p);
}

}  // namespace ringlpn_2pc
