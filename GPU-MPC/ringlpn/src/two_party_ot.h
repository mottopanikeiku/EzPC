// two_party_ot.h - real two-party transport and correlated-randomness
// primitives for the ringlpn distributed DPF key generation (milestone M1,
// stage S4 "real OT/OLE transport").
//
// WHAT IS REAL HERE
//   * Two OS processes, two TCP sockets. No shared memory, no shared RNG seed,
//     no party reading the other party's state.
//   * Default real 1-of-2 OT: the IKNP extension over Naor-Pinkas base OTs,
//     used from this repository's SCI stack. Explicit `emp-silent` selection
//     instead loads the separately compiled C++20 opaque bridge pinned by
//     RINGLPN_EMP_SILENT_REVISION; EMP templates never enter this C++17 header.
//   * Boolean AND triples: two 1-bit OTs per triple (Gilboa cross terms).
//   * Z_p scalar OLE: Gilboa multiplication, ceil(log2(p-1)) OTs of field
//     elements per OLE.
//   * Party protocol randomness/root seeds: OpenSSL's private CSPRNG, never a
//     shared seed. Fixed-seed mode exists only for reproducible public test inputs.
//   * Every wire byte and every direction switch is counted by sci::NetIO
//     (`counter`, `num_rounds`), per party and per socket.
//
// WHAT IS NOT CLAIMED
//   * The opt-in EMP SilentFerret source path and exact 1/62/128-bit packed
//     chosen-message adapter are unreviewed and unmeasured. They are never the
//     default and support no security or bandwidth claim until focused evidence
//     and independent review pass. SCI/IKNP remains the default evidence path.
//   * DPF expansion is outside this transport wrapper. The host-reference mode
//     uses non-cryptographic splitmix64; the GPU-consumable mode uses four
//     domain-separated AES calls with full 128-bit seeds. Device parity is
//     gated, but P-RNG/P-DIST/P-KEY and the DPF reduction remain open; see
//     results/reports/dealerless_orca_fc_security_contract_2026_07_29.md.
//   * SCI NetIO remains plain TCP. Live FC deployment therefore admits only
//     loopback endpoints and relies on the authenticated SSH launcher to carry
//     both sockets between hosts. Loopback without that launcher is explicitly
//     local-only evidence. Active attacks by either endpoint, denial of service,
//     and side channels are out of scope.

#pragma once

#include "OT/split-iknp.h"
#include "emp_silent_adapter.h"
#include <openssl/rand.h>

#include "utils/net_io_channel.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>
#include <random>
#include <stdexcept>
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
// `logical_bits` counts the meaningful bits of each common value reconstructed.
// `revealed_bits_sent` / `revealed_bits_recv` count the meaningful bit width of
// each share, not its byte-aligned encoding or measured wire traffic. Their sum
// is exposed as `meaningful_share_bits()`.

struct PhaseCosts {
    uint64_t logical_bits = 0;
    uint64_t revealed_bits_sent = 0;
    uint64_t revealed_bits_recv = 0;
};

struct Counters {
    uint64_t string_ots_128 = 0;  // 1-of-2 OT on 128-bit strings
    uint64_t triple_ots = 0;      // 1-bit OTs consumed by AND triples
    uint64_t ole_ots = 0;         // field-element OTs consumed by Gilboa OLE
    uint64_t bit_triples = 0;     // AND triples produced
    uint64_t scalar_oles = 0;     // Z_p OLEs produced
    uint64_t base_ots = 0;        // Naor-Pinkas base OTs for the OT extension
    PhaseCosts phase_a, phase_b, phase_c;

    uint64_t logical_opened_bits() const {
        return phase_a.logical_bits + phase_b.logical_bits + phase_c.logical_bits;
    }
    uint64_t meaningful_share_bits() const {
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
    PartyChannel(int party, const std::string &peer_host, int base_port,
                 bool defer_ot_setup = false,
                 bool require_loopback_endpoints = false,
                 OtBackend ot_backend = OtBackend::SciIknp,
                 const EmpSilentPlan *emp_plan = nullptr)
        : party_(party), ot_backend_(ot_backend) {
        if (ot_backend_ == OtBackend::EmpSilent) {
            if (emp_plan == nullptr) {
                throw std::invalid_argument(
                    "emp-silent requires an explicit public inventory plan");
            }
            emp_plan_ = *emp_plan;
        }
        if (require_loopback_endpoints && party != 0 &&
            peer_host != "127.0.0.1") {
            throw std::runtime_error(
                "authenticated/local-only channels must target 127.0.0.1");
        }
        const char *addr = (party == 0) ? nullptr : peer_host.c_str();
        std::unique_ptr<sci::NetIO> straight(
            new sci::NetIO(addr, base_port, /*full_buffer=*/false,
                           /*quiet=*/true));
        std::unique_ptr<sci::NetIO> reversed(
            new sci::NetIO(addr, base_port + 1, false, true));
        if (require_loopback_endpoints &&
            (!socket_is_loopback(straight->consocket) ||
             !socket_is_loopback(reversed->consocket))) {
            throw std::runtime_error(
                "SCI channel rejected a non-loopback socket endpoint");
        }
        io_ = straight.release();
        io_rev_ = reversed.release();
        if (ot_backend_ == OtBackend::SciIknp) {
            const int sci_party = (party == 0) ? sci::ALICE : sci::BOB;
            ot_straight_ = new sci::SplitIKNP<sci::NetIO>(sci_party, io_);
            ot_reversed_ = new sci::SplitIKNP<sci::NetIO>(3 - sci_party, io_rev_);
        }
        if (!defer_ot_setup) setup_ots();
    }

    // Idempotent so callers may agree on public work parameters over the raw
    // channels before paying for or deriving any OT correlation setup.
    void setup_ots() {
        if (ots_ready_) return;
        const uint64_t straight_before = straight_bytes_sent();
        const uint64_t reversed_before = reversed_bytes_sent();
        const uint64_t switches_before = direction_switches();
        if (ot_backend_ == OtBackend::SciIknp) {
            if (party_ == 0) {
                ot_straight_->setup_send();
                ot_reversed_->setup_recv();
            } else {
                ot_straight_->setup_recv();
                ot_reversed_->setup_send();
            }
            io_->flush();
            io_rev_->flush();
            costs.base_ots += 2 * 128;
        } else {
            emp_api_ =
                std::make_shared<EmpSilentApi>(emp_plan_.bridge_library);
            emp_straight_.reset(new EmpSilentDirectionalOt(
                emp_api_, io_, party_, RINGLPN_EMP_STRAIGHT,
                emp_plan_.straight_count, emp_plan_.threads,
                emp_plan_.public_manifest_digest));
            emp_reversed_.reset(new EmpSilentDirectionalOt(
                emp_api_, io_rev_, party_, RINGLPN_EMP_REVERSED,
                emp_plan_.reversed_count, emp_plan_.threads,
                emp_plan_.public_manifest_digest));
            // Both parties enter directions in this public order.
            emp_straight_->begin();
            emp_reversed_->begin();
        }
        const uint64_t straight_after = straight_bytes_sent();
        const uint64_t reversed_after = reversed_bytes_sent();
        const uint64_t switches_after = direction_switches();
        if (straight_after < straight_before ||
            reversed_after < reversed_before ||
            switches_after < switches_before) {
            throw std::overflow_error("OT transport counter wrapped during setup");
        }
        setup_straight_bytes_sent_ = straight_after - straight_before;
        setup_reversed_bytes_sent_ = reversed_after - reversed_before;
        if (setup_straight_bytes_sent_ >
            std::numeric_limits<uint64_t>::max() -
                setup_reversed_bytes_sent_) {
            throw std::overflow_error("OT setup byte total overflow");
        }
        setup_bytes_sent_ =
            setup_straight_bytes_sent_ + setup_reversed_bytes_sent_;
        setup_direction_switches_ = switches_after - switches_before;
        ots_ready_ = true;
    }

    // Required for emp-silent: verifies both public inventories were consumed
    // exactly and closes the two prepaid sessions. SCI has no session bookend.
    void finish_ots() {
        if (!ots_ready_ || ots_finished_) {
            if (ot_backend_ == OtBackend::EmpSilent)
                throw std::runtime_error("EMP OT finish called out of order");
            return;
        }
        if (ot_backend_ == OtBackend::EmpSilent) {
            emp_straight_->end();
            emp_reversed_->end();
        }
        ots_finished_ = true;
    }

    EmpSilentMetrics emp_silent_metrics() const {
        if (ot_backend_ != OtBackend::EmpSilent || !emp_straight_ ||
            !emp_reversed_)
            throw std::runtime_error("EMP SilentFerret metrics unavailable");
        return EmpSilentMetrics{emp_straight_->counters(),
                                emp_reversed_->counters()};
    }

    OtBackend ot_backend() const { return ot_backend_; }

    ~PartyChannel() {
        emp_reversed_.reset();
        emp_straight_.reset();
        emp_api_.reset();
        delete ot_straight_;
        delete ot_reversed_;
        delete io_;
        delete io_rev_;
    }

    PartyChannel(const PartyChannel &) = delete;
    PartyChannel &operator=(const PartyChannel &) = delete;
    PartyChannel(PartyChannel &&) = delete;
    PartyChannel &operator=(PartyChannel &&) = delete;

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

    uint64_t bytes_sent() const {
        if (io_->counter >
            std::numeric_limits<uint64_t>::max() - io_rev_->counter) {
            throw std::overflow_error("SCI byte total overflow");
        }
        return io_->counter + io_rev_->counter;
    }
    uint64_t straight_bytes_sent() const { return io_->counter; }
    uint64_t reversed_bytes_sent() const { return io_rev_->counter; }
    uint64_t direction_switches() const {
        if (io_->num_rounds >
            std::numeric_limits<uint64_t>::max() - io_rev_->num_rounds) {
            throw std::overflow_error("SCI direction-switch total overflow");
        }
        return io_->num_rounds + io_rev_->num_rounds;
    }
    uint64_t setup_bytes_sent() const { return setup_bytes_sent_; }
    uint64_t setup_straight_bytes_sent() const {
        return setup_straight_bytes_sent_;
    }
    uint64_t setup_reversed_bytes_sent() const {
        return setup_reversed_bytes_sent_;
    }
    uint64_t setup_direction_switches() const {
        return setup_direction_switches_;
    }

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

    // Raw symmetric exchange for batched openings; the caller records the
    // logical and per-direction share bits it actually opened.
    void exchange_bytes(const uint8_t *mine, uint8_t *theirs, size_t nbytes) {
        exchange(mine, theirs, (int)nbytes);
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
        if (m0.size() != m1.size())
            throw std::invalid_argument("128-bit OT sender vector size mismatch");
        const int n = checked_ot_count(m0.size());
        if (ot_backend_ == OtBackend::EmpSilent) {
            sender_emp()->send(128, m0.data(), m1.data(), (uint64_t)n);
        } else {
            std::unique_ptr<sci::block128[]> b0(new sci::block128[(size_t)n]);
            std::unique_ptr<sci::block128[]> b1(new sci::block128[(size_t)n]);
            for (int i = 0; i < n; ++i) {
                b0[i] = to_block(m0[i]);
                b1[i] = to_block(m1[i]);
            }
            sender_ot()->send(b0.get(), b1.get(), n);
            sender_io()->flush();
        }
        costs.string_ots_128 += (uint64_t)n;
    }

    std::vector<U128> ot_recv_128(const std::vector<uint8_t> &choices) {
        const int n = checked_ot_count(choices.size());
        std::vector<U128> res((size_t)n);
        if (ot_backend_ == OtBackend::EmpSilent) {
            receiver_emp()->recv(128, choices.data(), res.data(), (uint64_t)n);
        } else {
            std::unique_ptr<bool[]> raw(new bool[(size_t)n]);
            for (int i = 0; i < n; ++i)
                raw[(size_t)i] = choices[(size_t)i] != 0;
            std::unique_ptr<sci::block128[]> out(new sci::block128[(size_t)n]);
            receiver_ot()->recv(out.get(), raw.get(), n);
            receiver_io()->flush();
            for (int i = 0; i < n; ++i) res[(size_t)i] = from_block(out[i]);
        }
        costs.string_ots_128 += (uint64_t)n;
        return res;
    }

    // ---- 1-bit OT (AND-triple cross terms) ---------------------------------

    // SCI's l-bit OT wrappers expect one ROW of N=2 messages per instance
    // (`data[i][k]`, see SCI/src/OT/ot-utils.h pack_ot_messages), not two
    // parallel arrays.
    void ot_send_bits(const std::vector<uint8_t> &m0,
                      const std::vector<uint8_t> &m1) {
        if (m0.size() != m1.size())
            throw std::invalid_argument("bit OT sender vector size mismatch");
        const int n = checked_ot_count(m0.size());
        if (ot_backend_ == OtBackend::EmpSilent) {
            sender_emp()->send(1, m0.data(), m1.data(), (uint64_t)n);
        } else {
            std::vector<uint8_t> flat((size_t)2 * n);
            std::vector<uint8_t *> rows((size_t)n);
            for (int i = 0; i < n; ++i) {
                flat[(size_t)2 * i] = (uint8_t)(m0[(size_t)i] & 1);
                flat[(size_t)2 * i + 1] = (uint8_t)(m1[(size_t)i] & 1);
                rows[(size_t)i] = &flat[(size_t)2 * i];
            }
            sender_ot()->send(rows.data(), n, 1);
            sender_io()->flush();
        }
        costs.triple_ots += (uint64_t)n;
    }

    std::vector<uint8_t> ot_recv_bits(const std::vector<uint8_t> &choices) {
        const int n = checked_ot_count(choices.size());
        std::vector<uint8_t> out((size_t)n);
        if (ot_backend_ == OtBackend::EmpSilent) {
            receiver_emp()->recv(1, choices.data(), out.data(), (uint64_t)n);
        } else {
            std::vector<uint8_t> sel(choices);
            receiver_ot()->recv(out.data(), sel.data(), n, 1);
            receiver_io()->flush();
        }
        costs.triple_ots += (uint64_t)n;
        for (int i = 0; i < n; ++i) out[(size_t)i] &= 1;
        return out;
    }

    // ---- field-element OT (Gilboa OLE) ------------------------------------

    void ot_send_field(const std::vector<Word> &m0, const std::vector<Word> &m1,
                       int l) {
        if (m0.size() != m1.size())
            throw std::invalid_argument("field OT sender vector size mismatch");
        const int n = checked_ot_count(m0.size());
        if (ot_backend_ == OtBackend::EmpSilent) {
            if (l != 62)
                throw std::invalid_argument(
                    "emp-silent field OT requires the unchanged q62 width");
            sender_emp()->send(62, m0.data(), m1.data(), (uint64_t)n);
        } else {
            std::vector<uint64_t> flat((size_t)2 * n);
            std::vector<uint64_t *> rows((size_t)n);
            for (int i = 0; i < n; ++i) {
                flat[(size_t)2 * i] = (uint64_t)m0[(size_t)i];
                flat[(size_t)2 * i + 1] = (uint64_t)m1[(size_t)i];
                rows[(size_t)i] = &flat[(size_t)2 * i];
            }
            sender_ot()->send(rows.data(), n, l);
            sender_io()->flush();
        }
        costs.ole_ots += (uint64_t)n;
    }

    std::vector<Word> ot_recv_field(const std::vector<uint8_t> &choices, int l) {
        const int n = checked_ot_count(choices.size());
        std::vector<uint64_t> out((size_t)n);
        if (ot_backend_ == OtBackend::EmpSilent) {
            if (l != 62)
                throw std::invalid_argument(
                    "emp-silent field OT requires the unchanged q62 width");
            receiver_emp()->recv(62, choices.data(), out.data(), (uint64_t)n);
        } else {
            std::vector<uint8_t> sel(choices);
            receiver_ot()->recv(out.data(), sel.data(), n, l);
            receiver_io()->flush();
        }
        costs.ole_ots += (uint64_t)n;
        return std::vector<Word>(out.begin(), out.end());
    }

    Counters costs;

  private:
    static int checked_ot_count(size_t count) {
        if (count > (size_t)std::numeric_limits<int>::max())
            throw std::overflow_error("OT batch exceeds SCI-compatible count ABI");
        return (int)count;
    }

    EmpSilentDirectionalOt *sender_emp() {
        if (!ots_ready_)
            throw std::runtime_error("EMP OT consume before setup/begin");
        return party_ == 0 ? emp_straight_.get() : emp_reversed_.get();
    }
    EmpSilentDirectionalOt *receiver_emp() {
        if (!ots_ready_)
            throw std::runtime_error("EMP OT consume before setup/begin");
        return party_ == 0 ? emp_reversed_.get() : emp_straight_.get();
    }

    static bool ipv4_is_loopback(const sockaddr_in &address) {
        return (ntohl(address.sin_addr.s_addr) >> 24) == 127;
    }

    static bool socket_is_loopback(int fd) {
        sockaddr_in local{};
        sockaddr_in peer{};
        socklen_t local_size = sizeof(local);
        socklen_t peer_size = sizeof(peer);
        return ::getsockname(fd, reinterpret_cast<sockaddr *>(&local),
                             &local_size) == 0 &&
               ::getpeername(fd, reinterpret_cast<sockaddr *>(&peer),
                             &peer_size) == 0 &&
               local.sin_family == AF_INET && peer.sin_family == AF_INET &&
               ipv4_is_loopback(local) && ipv4_is_loopback(peer);
    }

    int party_;
    sci::NetIO *io_ = nullptr;
    sci::NetIO *io_rev_ = nullptr;
    sci::SplitIKNP<sci::NetIO> *ot_straight_ = nullptr;
    sci::SplitIKNP<sci::NetIO> *ot_reversed_ = nullptr;
    OtBackend ot_backend_ = OtBackend::SciIknp;
    EmpSilentPlan emp_plan_{};
    std::shared_ptr<EmpSilentApi> emp_api_;
    std::unique_ptr<EmpSilentDirectionalOt> emp_straight_;
    std::unique_ptr<EmpSilentDirectionalOt> emp_reversed_;
    uint64_t setup_bytes_sent_ = 0;
    uint64_t setup_straight_bytes_sent_ = 0;
    uint64_t setup_reversed_bytes_sent_ = 0;
    uint64_t setup_direction_switches_ = 0;
    bool ots_ready_ = false;
    bool ots_finished_ = false;
};

// ----- party-private randomness ---------------------------------------------
//
// The default stream is OpenSSL's private DRBG, seeded from the operating
// system. `std::mt19937_64` is deliberately confined to explicit fixed-seed
// mode for reproducible public test inputs; it is never protocol randomness.

class PartyRandom {
  public:
    PartyRandom() = default;
    explicit PartyRandom(uint64_t fixed_seed)
        : deterministic_(true), deterministic_gen_(fixed_seed) {}
    PartyRandom(const PartyRandom &) = delete;
    PartyRandom &operator=(const PartyRandom &) = delete;
    PartyRandom(PartyRandom &&) = delete;
    PartyRandom &operator=(PartyRandom &&) = delete;

    uint64_t u64() {
        if (deterministic_) return deterministic_gen_();
        if (buffer_pos_ == buffer_.size()) refill();
        return buffer_[buffer_pos_++];
    }

    uint8_t bit() {
        if (bits_left_ == 0) {
            bit_pool_ = u64();
            bits_left_ = 64;
        }
        const uint8_t out = static_cast<uint8_t>(bit_pool_ & 1);
        bit_pool_ >>= 1;
        --bits_left_;
        return out;
    }

    U128 u128() {
        const uint64_t hi = u64();
        const uint64_t lo = u64();
        return (static_cast<U128>(hi) << 64) | static_cast<U128>(lo);
    }

    Word field(Word p) {
        if (p == 0) throw std::invalid_argument("PartyRandom::field modulus is zero");
        const U128 threshold = (U128(0) - U128(p)) % U128(p);
        U128 x = 0;
        do {
            x = u128();
        } while (x < threshold);
        return static_cast<Word>(x % U128(p));
    }

  private:
    void refill() {
        if (RAND_priv_bytes(reinterpret_cast<unsigned char *>(buffer_.data()),
                            sizeof(buffer_)) != 1) {
            throw std::runtime_error("OpenSSL RAND_priv_bytes failed");
        }
        buffer_pos_ = 0;
    }

    bool deterministic_ = false;
    std::mt19937_64 deterministic_gen_{};
    std::array<uint64_t, 512> buffer_{};
    size_t buffer_pos_ = buffer_.size();
    uint64_t bit_pool_ = 0;
    int bits_left_ = 0;
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

// ----- Gilboa Z_p scalar OLE (batched) --------------------------------------

// Batched Gilboa OLE. For B sender inputs x_b and B receiver inputs y_b the
// pair ends with additive shares u_b + v_b = x_b * y_b mod p. Cost:
// B * ceil(log2(p-1)) OTs of field elements in ONE OT batch, so the measured
// direction-switch count does not grow with B.

inline std::vector<Word> ole_batch_send(PartyChannel &ch,
                                        const std::vector<Word> &x, Word p,
                                        PartyRandom &rng) {
    const int k = field_bits(p);
    const size_t B = x.size();
    std::vector<Word> m0(B * (size_t)k), m1(B * (size_t)k), out(B, 0);
    for (size_t b = 0; b < B; ++b) {
        Word acc = 0;
        Word shifted = x[b] % p;
        for (int j = 0; j < k; ++j) {
            const Word r = rng.field(p);
            m0[b * (size_t)k + (size_t)j] = r;
            m1[b * (size_t)k + (size_t)j] = mod_add(r, shifted, p);
            acc = mod_add(acc, r, p);
            shifted = mod_add(shifted, shifted, p);  // 2^(j+1) * x mod p
        }
        out[b] = mod_sub(0, acc, p);  // u_b = -sum_j r_(b,j) mod p
    }
    ch.ot_send_field(m0, m1, k);
    ch.costs.scalar_oles += (uint64_t)B;
    return out;
}

inline std::vector<Word> ole_batch_recv(PartyChannel &ch,
                                        const std::vector<Word> &y, Word p) {
    const int k = field_bits(p);
    const size_t B = y.size();
    std::vector<uint8_t> choice(B * (size_t)k);
    for (size_t b = 0; b < B; ++b) {
        for (int j = 0; j < k; ++j) {
            choice[b * (size_t)k + (size_t)j] = (uint8_t)((y[b] >> j) & 1);
        }
    }
    const std::vector<Word> got = ch.ot_recv_field(choice, k);
    std::vector<Word> out(B, 0);
    for (size_t b = 0; b < B; ++b) {
        Word v = 0;
        for (int j = 0; j < k; ++j) {
            v = mod_add(v, got[b * (size_t)k + (size_t)j] % p, p);
        }
        out[b] = v;
    }
    ch.costs.scalar_oles += (uint64_t)B;
    return out;
}

// Directional wrapper: party 0 is always the OLE sender in this protocol.
inline std::vector<Word> ole_batch_p0_sender(PartyChannel &ch,
                                             const std::vector<Word> &my_inputs,
                                             Word p, PartyRandom &rng) {
    return ch.is_p0() ? ole_batch_send(ch, my_inputs, p, rng)
                      : ole_batch_recv(ch, my_inputs, p);
}

}  // namespace ringlpn_2pc
