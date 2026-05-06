// bench_ole_ringlpn_host: end-to-end host-side benchmark of Figure 2
// (BCG+20 / §6.4 of the draft) OLE Expand with SPFSS-backed shares.
//
// Pipeline (one call, one party each step):
//   1. Sample public a = (1, a_1, ..., a_{c-1}) uniform in R = Z_p[X]/(X^N+1).
//   2. Sample sparse noise e^i_σ with t nonzeros per (σ, i).
//   3. For each (i, j) in [0, c)^2: SPFSS.Gen over [0, 2N) with t^2 points
//      (α_{k,l} = A^i_0[k] + A^j_1[l],   β_{k,l} = b^i_0[k] * b^j_1[l] mod p).
//      Full-evaluate on each party ⇒ u_σ[i+jc] ∈ Z_p^{2N}, shares of e^i_0 * e^j_1
//      as a degree-<2N polynomial.
//   4. Fold each u_σ[i+jc] from 2N coefficients to N (negacyclic: coefficient
//      k≥N wraps to -u[k-N] mod p).
//   5. x_σ = Σ_i a_i * e^i_σ mod (X^N+1) [computed directly on sparse reps,
//      not via SPFSS — Figure 2 step "x_σ = <a, e_σ>"].
//   6. z_σ = Σ_{i,j} a_i * a_j * u_σ[i+jc] mod (X^N+1).
//   7. Validate: (z_0 + z_1) == (x_0 * x_1) mod (X^N+1), coefficient-wise.
//
// Correctness artifact for the Figure 2 claim; does not yet use GPU acceleration.

#include "spfss_host.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr uint64_t kModulus62 = 4611686018326724609ULL;

using Word = uint64_t;
using U128 = __uint128_t;

inline Word mod_add(Word a, Word b, Word m) {
    Word s = a + b;
    if (s >= m || s < a) s -= m;
    return s;
}
inline Word mod_sub(Word a, Word b, Word m) {
    return a >= b ? a - b : m - (b - a);
}
inline Word mod_mul(Word a, Word b, Word m) {
    return (Word)((U128)a * (U128)b % (U128)m);
}

struct SparsePoly {
    std::vector<int> positions; // in [0, N)
    std::vector<Word> values;   // in [0, m)
};

using DensePoly = std::vector<Word>; // size N, coefficients in [0, m)

SparsePoly sample_sparse(int N, int t, Word m, std::mt19937_64 &rng) {
    std::uniform_int_distribution<int> pos_dist(0, N - 1);
    std::uniform_int_distribution<uint64_t> val_dist(1, m - 1);
    SparsePoly out;
    std::unordered_set<int> used;
    used.reserve((size_t)t * 2 + 1);
    while ((int)used.size() < t) {
        int p = pos_dist(rng);
        if (!used.insert(p).second) continue;
        out.positions.push_back(p);
        out.values.push_back(val_dist(rng));
    }
    return out;
}

DensePoly sample_dense(int N, Word m, std::mt19937_64 &rng) {
    std::uniform_int_distribution<uint64_t> dist(0, m - 1);
    DensePoly out(N);
    for (int i = 0; i < N; ++i) out[i] = dist(rng);
    return out;
}

DensePoly densify(const SparsePoly &s, int N, Word m) {
    DensePoly out(N, 0);
    for (size_t k = 0; k < s.positions.size(); ++k) {
        out[s.positions[k]] = mod_add(out[s.positions[k]], s.values[k], m);
    }
    return out;
}

// Dense * Dense mod (X^N + 1). O(N^2) schoolbook; fine for bench correctness.
DensePoly dense_mul(const DensePoly &a, const DensePoly &b, int N, Word m) {
    DensePoly out(N, 0);
    for (int i = 0; i < N; ++i) {
        if (!a[i]) continue;
        for (int j = 0; j < N; ++j) {
            if (!b[j]) continue;
            Word prod = mod_mul(a[i], b[j], m);
            int p = i + j;
            if (p < N) out[p] = mod_add(out[p], prod, m);
            else out[p - N] = mod_sub(out[p - N], prod, m);
        }
    }
    return out;
}

DensePoly dense_sparse_mul(const DensePoly &a, const SparsePoly &s, int N, Word m) {
    DensePoly out(N, 0);
    for (size_t k = 0; k < s.positions.size(); ++k) {
        int j = s.positions[k];
        Word bj = s.values[k];
        for (int i = 0; i < N; ++i) {
            Word ai = a[i];
            if (!ai) continue;
            Word prod = mod_mul(ai, bj, m);
            int p = i + j;
            if (p < N) out[p] = mod_add(out[p], prod, m);
            else out[p - N] = mod_sub(out[p - N], prod, m);
        }
    }
    return out;
}

// Fold a length-2N polynomial u to length N using X^N = -1 negacyclic wrap.
DensePoly fold_2N_to_N(const std::vector<Word> &u_2n, int N, Word m) {
    DensePoly out(N, 0);
    for (int k = 0; k < N; ++k) out[k] = u_2n[k];
    for (int k = N; k < 2 * N; ++k) {
        out[k - N] = mod_sub(out[k - N], u_2n[k], m);
    }
    return out;
}

struct Args {
    int N = 64;
    int c = 2;
    int t = 8;
    uint64_t seed = 1;
    bool verbose = false;
};

void usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " [--n N] [--c c] [--t t] [--seed s] [--verbose]\n"
              << "  N: polynomial degree (power of 2).\n"
              << "  c: Ring-LPN compression factor.\n"
              << "  t: noise weight (nonzeros per sparse e^i_sigma).\n";
}

Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) a.N = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--c") && i + 1 < argc) a.c = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--t") && i + 1 < argc) a.t = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) a.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--verbose")) a.verbose = true;
        else { usage(argv[0]); std::exit(1); }
    }
    if (a.N < 2 || (a.N & (a.N - 1)) || a.c < 1 || a.t < 1 || a.t > a.N) {
        usage(argv[0]); std::exit(1);
    }
    return a;
}

int log2i(int x) {
    int r = 0;
    while ((1 << r) < x) ++r;
    return r;
}

} // namespace

int main(int argc, char **argv) {
    Args args = parse(argc, argv);
    const int N = args.N, c = args.c, t = args.t;
    const Word m = kModulus62;
    const int log_domain = log2i(2 * N); // SPFSS domain = [0, 2N)
    std::mt19937_64 rng(args.seed);

    // 1. Public a = (1, a_1, ..., a_{c-1})
    std::vector<DensePoly> a(c);
    a[0] = DensePoly(N, 0); a[0][0] = 1;
    for (int i = 1; i < c; ++i) a[i] = sample_dense(N, m, rng);

    // 2. Sparse noise (A^i_σ, b^i_σ) for σ ∈ {0,1}, i ∈ [0, c)
    std::vector<std::vector<SparsePoly>> e(2, std::vector<SparsePoly>(c));
    for (int sigma = 0; sigma < 2; ++sigma) {
        for (int i = 0; i < c; ++i) {
            e[sigma][i] = sample_sparse(N, t, m, rng);
        }
    }

    // 3. For each (i, j), SPFSS.Gen with t^2 points on [0, 2N):
    //      α_{k,l} = A^i_0[k] + A^j_1[l]     (sum in [0, 2N-1))
    //      β_{k,l} = b^i_0[k] * b^j_1[l] mod p
    // Each party calls SPFSS.FullEval ⇒ u^{i,j}_σ ∈ Z_p^{2N}.
    // Note: α values may collide (two (k,l) producing the same sum); SPFSS handles this
    // because each DPF is independent and shares sum linearly.
    std::vector<std::vector<Word>> u0(c * c), u1(c * c);
    uint64_t spfss_rng_state = args.seed ^ 0xC6BC279692B5C323ULL;
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < c; ++i) {
            const SparsePoly &e0 = e[0][i];
            const SparsePoly &e1 = e[1][j];
            std::vector<uint64_t> alphas; alphas.reserve((size_t)t * t);
            std::vector<Word> betas; betas.reserve((size_t)t * t);
            for (int k = 0; k < t; ++k) {
                for (int l = 0; l < t; ++l) {
                    alphas.push_back((uint64_t)e0.positions[k] + (uint64_t)e1.positions[l]);
                    betas.push_back(mod_mul(e0.values[k], e1.values[l], m));
                }
            }
            spfss_host::SPFSSKey K0, K1;
            spfss_host::spfssGen(alphas, betas, log_domain, m, spfss_rng_state, K0, K1);
            size_t idx = (size_t)i + (size_t)j * (size_t)c;
            spfss_host::spfssFullEval(0, K0, u0[idx]);
            spfss_host::spfssFullEval(1, K1, u1[idx]);
        }
    }

    // 4. Fold each u to degree < N via X^N = -1 negacyclic wrap.
    std::vector<DensePoly> u0_folded(c * c), u1_folded(c * c);
    for (int k = 0; k < c * c; ++k) {
        u0_folded[k] = fold_2N_to_N(u0[k], N, m);
        u1_folded[k] = fold_2N_to_N(u1[k], N, m);
    }

    // 5. x_σ = Σ_i a_i * e^i_σ mod (X^N + 1)
    std::vector<DensePoly> x(2, DensePoly(N, 0));
    for (int sigma = 0; sigma < 2; ++sigma) {
        for (int i = 0; i < c; ++i) {
            DensePoly term = dense_sparse_mul(a[i], e[sigma][i], N, m);
            for (int k = 0; k < N; ++k) x[sigma][k] = mod_add(x[sigma][k], term[k], m);
        }
    }

    // 6. z_σ = Σ_{i,j} a_i * a_j * u_σ[i+jc] mod (X^N + 1)
    std::vector<DensePoly> z(2, DensePoly(N, 0));
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < c; ++i) {
            DensePoly aiaj = dense_mul(a[i], a[j], N, m);
            size_t idx = (size_t)i + (size_t)j * (size_t)c;
            DensePoly term0 = dense_mul(aiaj, u0_folded[idx], N, m);
            DensePoly term1 = dense_mul(aiaj, u1_folded[idx], N, m);
            for (int k = 0; k < N; ++k) {
                z[0][k] = mod_add(z[0][k], term0[k], m);
                z[1][k] = mod_add(z[1][k], term1[k], m);
            }
        }
    }

    // 7. Validate z_0 + z_1 == x_0 * x_1 mod (X^N + 1)
    DensePoly z_sum(N, 0);
    for (int k = 0; k < N; ++k) z_sum[k] = mod_add(z[0][k], z[1][k], m);
    DensePoly x0x1 = dense_mul(x[0], x[1], N, m);

    int mismatches = 0, first_idx = -1;
    Word first_got = 0, first_expected = 0;
    for (int k = 0; k < N; ++k) {
        if (z_sum[k] != x0x1[k]) {
            if (first_idx < 0) { first_idx = k; first_got = z_sum[k]; first_expected = x0x1[k]; }
            ++mismatches;
        }
    }
    bool pass = (mismatches == 0);

    if (!pass || args.verbose) {
        std::cerr << "N=" << N << " c=" << c << " t=" << t << " seed=" << args.seed
                  << " log_domain=" << log_domain
                  << " mismatches=" << mismatches;
        if (first_idx >= 0) {
            std::cerr << " first_idx=" << first_idx
                      << " got=" << first_got
                      << " expected=" << first_expected;
        }
        std::cerr << "\n";
    }

    std::cout << "n=" << N << ",c=" << c << ",t=" << t << ",seed=" << args.seed
              << ",q=" << m << ",log_domain=" << log_domain
              << ",ole_pass=" << (pass ? 1 : 0) << "\n";
    return pass ? 0 : 1;
}
