// verify_figure2_expand: host-side correctness oracle for the Figure 2 (BCG+20)
// Ring-LPN OLE Expand procedure in Zp[X]/(X^N + 1).
//
// Checks the algebraic identity underpinning the protocol:
//   z := <a (x) a, u>    with u[i + j*c] = e^i_0 * e^j_1 mod (X^N + 1)
//   ==   x_0 * x_1       with x_sigma   = <a, e_sigma> mod (X^N + 1)
// all reductions taken mod (X^N + 1). This is protocol-independent: any correct
// SPFSS-based sharing of u must agree with the plaintext u computed here, so
// this oracle establishes the ground truth the GPU OLE path must reproduce.

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

// 62-bit NTT-friendly prime used by the cheddar/ringlpn CUDA path
// (kConfig62 in bench_ntt_cuda_cheddar.cu). Declared here independently so
// this oracle has no GPU/code dependency.
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
    return static_cast<Word>((U128)a * (U128)b % (U128)m);
}

struct SparsePoly {
    std::vector<int> positions; // in [0, N)
    std::vector<Word> values;   // in [0, m)
};

// Dense polynomial in Zp[X]/(X^N + 1), coefficients at indices [0, N).
using DensePoly = std::vector<Word>;

DensePoly densify(const SparsePoly &s, int N, Word m) {
    DensePoly out(N, 0);
    for (size_t k = 0; k < s.positions.size(); ++k) {
        int p = s.positions[k];
        out[p] = mod_add(out[p], s.values[k], m);
    }
    return out;
}

// Dense * Dense mod (X^N + 1). Schoolbook O(N^2). Use only for small N in the
// oracle; the GPU path is the fast path.
DensePoly dense_mul(const DensePoly &a, const DensePoly &b, int N, Word m) {
    DensePoly out(N, 0);
    for (int i = 0; i < N; ++i) {
        if (!a[i]) continue;
        for (int j = 0; j < N; ++j) {
            if (!b[j]) continue;
            Word prod = mod_mul(a[i], b[j], m);
            int p = i + j;
            if (p < N) {
                out[p] = mod_add(out[p], prod, m);
            } else {
                // X^N = -1, so X^{N+r} = -X^r.
                out[p - N] = mod_sub(out[p - N], prod, m);
            }
        }
    }
    return out;
}

// Dense * Sparse mod (X^N + 1). O(N * |support|).
DensePoly dense_sparse_mul(const DensePoly &a, const SparsePoly &s, int N, Word m) {
    DensePoly out(N, 0);
    for (size_t k = 0; k < s.positions.size(); ++k) {
        int j = s.positions[k];
        Word bj = s.values[k];
        if (!bj) continue;
        for (int i = 0; i < N; ++i) {
            Word ai = a[i];
            if (!ai) continue;
            Word prod = mod_mul(ai, bj, m);
            int p = i + j;
            if (p < N) {
                out[p] = mod_add(out[p], prod, m);
            } else {
                out[p - N] = mod_sub(out[p - N], prod, m);
            }
        }
    }
    return out;
}

// Sparse * Sparse mod (X^N + 1). O(|supp_a| * |supp_b|).
DensePoly sparse_sparse_mul(const SparsePoly &a, const SparsePoly &b, int N, Word m) {
    DensePoly out(N, 0);
    for (size_t i = 0; i < a.positions.size(); ++i) {
        int pa = a.positions[i];
        Word va = a.values[i];
        for (size_t j = 0; j < b.positions.size(); ++j) {
            int pb = b.positions[j];
            Word vb = b.values[j];
            Word prod = mod_mul(va, vb, m);
            int p = pa + pb;
            if (p < N) {
                out[p] = mod_add(out[p], prod, m);
            } else {
                out[p - N] = mod_sub(out[p - N], prod, m);
            }
        }
    }
    return out;
}

bool poly_equal(const DensePoly &a, const DensePoly &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

SparsePoly sample_sparse(int N, int t, Word m, std::mt19937_64 &rng) {
    std::uniform_int_distribution<int> pos_dist(0, N - 1);
    std::uniform_int_distribution<uint64_t> val_dist(1, m - 1);
    SparsePoly out;
    std::unordered_set<int> used;
    used.reserve(static_cast<size_t>(t) * 2 + 1);
    while (static_cast<int>(used.size()) < t) {
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

struct Args {
    int N = 128;
    int c = 2;
    int t = 16;
    uint64_t seed = 1;
    bool verbose = false;
};

void usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " [--n N] [--c c] [--t t] [--seed s] [--verbose]\n"
              << "  N: polynomial degree (power of 2).\n"
              << "  c: Ring-LPN compression factor (number of public a_i).\n"
              << "  t: noise weight (nonzeros per sparse e^i_sigma).\n";
}

Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) a.N = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--c") && i + 1 < argc) a.c = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--t") && i + 1 < argc) a.t = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) a.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--verbose")) a.verbose = true;
        else { usage(argv[0]); std::exit(1); }
    }
    if (a.N < 2 || (a.N & (a.N - 1)) || a.c < 1 || a.t < 1 || a.t > a.N) {
        usage(argv[0]); std::exit(1);
    }
    return a;
}

} // namespace

int main(int argc, char **argv) {
    Args args = parse(argc, argv);
    const int N = args.N, c = args.c, t = args.t;
    const Word m = kModulus62;
    std::mt19937_64 rng(args.seed);

    // Public a = (1, a_1, ..., a_{c-1}) uniform in R.
    std::vector<DensePoly> a(c);
    a[0] = DensePoly(N, 0);
    a[0][0] = 1;
    for (int i = 1; i < c; ++i) a[i] = sample_dense(N, m, rng);

    // Sparse noise (A^i_sigma, b^i_sigma) for sigma in {0,1}, i in [0,c).
    std::vector<std::vector<SparsePoly>> e(2, std::vector<SparsePoly>(c));
    for (int sigma = 0; sigma < 2; ++sigma) {
        for (int i = 0; i < c; ++i) {
            e[sigma][i] = sample_sparse(N, t, m, rng);
        }
    }

    // x_sigma = <a, e_sigma> mod F.
    std::vector<DensePoly> x(2, DensePoly(N, 0));
    for (int sigma = 0; sigma < 2; ++sigma) {
        for (int i = 0; i < c; ++i) {
            DensePoly term = dense_sparse_mul(a[i], e[sigma][i], N, m);
            for (int k = 0; k < N; ++k) x[sigma][k] = mod_add(x[sigma][k], term[k], m);
        }
    }

    // u[i + j*c] = e^i_0 * e^j_1 mod F. (Figure 2 step 4 indexing.)
    std::vector<DensePoly> u(static_cast<size_t>(c) * c);
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < c; ++i) {
            u[static_cast<size_t>(i) + static_cast<size_t>(j) * c] =
                sparse_sparse_mul(e[0][i], e[1][j], N, m);
        }
    }

    // z = <a (x) a, u> = sum_{i,j} a_i * a_j * u[i + j*c] mod F.
    DensePoly z(N, 0);
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < c; ++i) {
            DensePoly aa = dense_mul(a[i], a[j], N, m);
            DensePoly term = dense_mul(aa, u[static_cast<size_t>(i) + static_cast<size_t>(j) * c], N, m);
            for (int k = 0; k < N; ++k) z[k] = mod_add(z[k], term[k], m);
        }
    }

    // x_0 * x_1 mod F.
    DensePoly x0x1 = dense_mul(x[0], x[1], N, m);

    bool pass = poly_equal(z, x0x1);

    if (!pass || args.verbose) {
        int mismatches = 0, first_idx = -1;
        Word first_got = 0, first_expected = 0;
        for (int k = 0; k < N; ++k) {
            if (z[k] != x0x1[k]) {
                if (first_idx < 0) { first_idx = k; first_got = z[k]; first_expected = x0x1[k]; }
                ++mismatches;
            }
        }
        std::cerr << "N=" << N << " c=" << c << " t=" << t << " seed=" << args.seed
                  << " mismatches=" << mismatches;
        if (first_idx >= 0) {
            std::cerr << " first_idx=" << first_idx
                      << " got=" << first_got
                      << " expected=" << first_expected;
        }
        std::cerr << "\n";
    }

    std::cout << "n=" << N << ",c=" << c << ",t=" << t << ",seed=" << args.seed
              << ",q=" << m << ",expand_pass=" << (pass ? 1 : 0) << "\n";
    return pass ? 0 : 1;
}
