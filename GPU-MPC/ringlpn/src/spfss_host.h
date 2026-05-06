// spfss_host: host-side DPF + SPFSS with Z_p payloads for the Figure 2 OLE
// Expand procedure. Standard Boyle-Gilboa-Ishai GGM tree with one Z_p
// correction word at the last level.
//
// Scope note: PRG is splitmix64 (deterministic, non-cryptographic). This is
// correctness infrastructure for the Figure 2 artifact; a real PRG is a
// drop-in replacement at prg_expand().

#pragma once

#include <cstdint>
#include <vector>

namespace spfss_host {

using Word = uint64_t;
using U128 = __uint128_t;

// A single DPF key for a (log_domain)-bit input domain with a Z_p payload.
// Correction words for the tree levels plus one final-level Z_p correction.
struct DPFKey {
    int log_domain;
    Word modulus;
    U128 seed;               // initial 128-bit seed
    uint8_t t0;              // initial control bit (0 for party 0, 1 for party 1)
    std::vector<U128> sCW;   // seed correction words, one per level
    std::vector<uint8_t> tLCW;
    std::vector<uint8_t> tRCW;
    Word finalCW;            // Z_p correction word applied at the target leaf
};

struct SPFSSKey {
    int log_domain;
    Word modulus;
    std::vector<DPFKey> dpf_keys;
};

// Deterministic DPF keygen. alpha in [0, 2^log_domain), beta in [0, modulus).
// rng_state is consumed+advanced so sequential calls produce independent keys.
void dpfGen(uint64_t alpha, int log_domain, Word beta, Word modulus,
            uint64_t &rng_state, DPFKey &K0, DPFKey &K1);

// Full-domain evaluation. out[x] gets party's Z_p share of beta*[x == alpha].
// out is resized to 2^log_domain.
void dpfEvalAll(int party, const DPFKey &K, std::vector<Word> &out);

// Single-point evaluation at x.
Word dpfEval(int party, const DPFKey &K, uint64_t x);

// SPFSS wrapper. m = alphas.size() = betas.size(). Each (alpha_k, beta_k) is
// one point with alpha in [0, 2^log_domain) and beta in Z_p. Output shares sum
// to the sum of point functions: Sum_k beta_k * [x == alpha_k] mod p.
void spfssGen(const std::vector<uint64_t> &alphas,
              const std::vector<Word> &betas,
              int log_domain, Word modulus, uint64_t &rng_state,
              SPFSSKey &K0, SPFSSKey &K1);

void spfssFullEval(int party, const SPFSSKey &K, std::vector<Word> &out);

} // namespace spfss_host
