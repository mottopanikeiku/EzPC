#ifndef RINGLPN_DEVICE_LABEL
#define RINGLPN_DEVICE_LABEL "cuda_ringlpn_ole"
#endif
#include "ringlpn_ole_party.cuh"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using Word = uint64_t;
using Clock = std::chrono::high_resolution_clock;

struct OleArgs {
    int n = kMinDegree;
    int c = 2;
    int t = 64;
    int qbits = 64;
    int iters = 3;
    int warmup = 1;
    int chunk_size = 8192;
    uint64_t seed = 1;
    std::string noise = "uniform";
    // Raw suffix inserted between an artifact prefix and "_limb"; empty keeps
    // the standalone filenames unchanged (embedded callers may use "_dir0").
    std::string artifact_suffix;
    bool csv_header = false;
    bool skip_validation = false;
};

struct SparsePoly {
    std::vector<int> positions;
    std::vector<Word> values;
};

struct SummaryStats {
    double mean_us = 0.0;
    double stddev_us = 0.0;
};

__device__ __forceinline__ Word mod_add_device(Word a, Word b, Word modulus) {
    Word s = a + b;
    return (s >= modulus || s < a) ? s - modulus : s;
}

__global__ void add_vectors_kernel(const Word *a,
                                   const Word *b,
                                   Word *out,
                                   size_t count,
                                   Word modulus) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = mod_add_device(a[idx], b[idx], modulus);
    }
}

static void ole_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " --n <deg> [--qbits 64|128] [--c N] [--t N] [--seed N]"
              << " [--iters N] [--warmup N] [--chunk-size N]"
              << " [--noise uniform|regular] [--csv-header] [--skip-validation]\n";
}
static bool ole_supported_work(const OleArgs &args) {
    constexpr uint64_t kMaxAggregateWords = 1ULL << 30;  // 8 GiB at 64 bits
    constexpr uint64_t kMaxTrees = 1ULL << 20;
    constexpr uint64_t kMaxKeyBytes = 1ULL << 32;
    const uint64_t n = static_cast<uint64_t>(args.n);
    const uint64_t c = static_cast<uint64_t>(args.c);
    const uint64_t t = static_cast<uint64_t>(args.t);
    if (c == 0 || t == 0 || c > std::numeric_limits<uint64_t>::max() / c) {
        return false;
    }
    const uint64_t cc = c * c;
    if (cc > std::numeric_limits<uint64_t>::max() / n) return false;
    const uint64_t cc_coeffs = cc * n;
    const uint64_t c_coeffs = c * n;
    if (cc_coeffs > (kMaxAggregateWords - 16 * n) / 12 ||
        c_coeffs > (kMaxAggregateWords - 16 * n - 12 * cc_coeffs) / 8) {
        return false;
    }
    if (t > std::numeric_limits<uint64_t>::max() / t ||
        cc > kMaxTrees / (t * t)) {
        return false;
    }
    const uint64_t trees = cc * t * t;
    const uint64_t domain =
        args.noise == "regular" ? 2 * (n / t) : 2 * n;
    int L = 0;
    for (uint64_t x = domain; x > 1; x >>= 1) ++L;
    const uint64_t per_key = 24 + 18 * static_cast<uint64_t>(L);
    if (trees > kMaxKeyBytes / per_key) return false;
    const bool artifact_mode = std::getenv("RINGLPN_OLE_NOISE") ||
                               std::getenv("RINGLPN_OLE_EXPORT_NOISE") ||
                               std::getenv("RINGLPN_OLE_SPFSS_KEYS");
    return !artifact_mode || (L >= 2 && L <= 20);
}

static OleArgs parse_ole_args(int argc, char **argv) {
    OleArgs args;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) {
            args.n = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--qbits") && i + 1 < argc) {
            args.qbits = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--c") && i + 1 < argc) {
            args.c = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--t") && i + 1 < argc) {
            args.t = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            args.seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--noise") && i + 1 < argc) {
            args.noise = argv[++i];
        } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
            args.iters = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
            args.warmup = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--chunk-size") && i + 1 < argc) {
            args.chunk_size = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--csv-header")) {
            args.csv_header = true;
        } else if (!std::strcmp(argv[i], "--skip-validation")) {
            args.skip_validation = true;
        } else {
            ole_usage(argv[0]);
            std::exit(1);
        }
    }

    if (!is_power_of_two(args.n) || args.n < kMinDegree || args.n > kMaxDegree) {
        ole_usage(argv[0]);
        std::exit(1);
    }
    if ((args.qbits != 64 && args.qbits != 128) ||
        args.c <= 0 || args.t <= 0 || args.t > args.n ||
        args.iters <= 0 || args.warmup < 0 || args.chunk_size <= 0) {
        ole_usage(argv[0]);
        std::exit(1);
    }
    if (args.noise != "uniform" && args.noise != "regular") {
        ole_usage(argv[0]);
        std::exit(1);
    }
    if (args.noise == "regular" &&
        (args.n % args.t != 0 || !is_power_of_two(args.t))) {
        std::cerr << "Regular noise requires power-of-two t dividing n for the first paper-compatible GPU artifact\n";
        std::exit(1);
    }
    if (static_cast<uint64_t>(args.c) * static_cast<uint64_t>(args.c) >
            65535ULL ||
        !ole_supported_work(args)) {
        std::cerr << "Unsupported work size or artifact domain\n";
        std::exit(1);
    }
    return args;
}

static std::vector<ModulusConfig<Word>> ole_modulus_configs(int qbits) {
    if (qbits == 128) {
        return {kConfig62, kConfig62Crt2};
    }
    return {kConfig62};
}

static int ole_actual_qbits(const std::vector<ModulusConfig<Word>> &configs) {
    int total = 0;
    for (const auto &config : configs) {
        total += config.actual_qbits;
    }
    return total;
}

static uint64_t mix_seed(uint64_t seed, uint64_t tag) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL + (tag << 6) + (tag >> 2);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static bool use_regular_noise(const OleArgs &args) {
    return args.noise == "regular";
}

static int regular_bucket_size(const OleArgs &args) {
    return args.n / args.t;
}

static int spfss_domain_size(const OleArgs &args) {
    return use_regular_noise(args) ? 2 * regular_bucket_size(args) : 2 * args.n;
}

static int spfss_group_count(const OleArgs &args) {
    return use_regular_noise(args) ? 2 * args.t - 1 : 1;
}

static SummaryStats summarize(const std::vector<double> &samples) {
    SummaryStats s;
    if (samples.empty()) {
        return s;
    }
    s.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) /
                static_cast<double>(samples.size());
    double var = 0.0;
    for (double sample : samples) {
        double delta = sample - s.mean_us;
        var += delta * delta;
    }
    s.stddev_us = std::sqrt(var / static_cast<double>(samples.size()));
    return s;
}

static int log2i(int n) {
    int out = 0;
    while (n > 1) {
        n >>= 1;
        ++out;
    }
    return out;
}

static Word host_mod_add(Word a, Word b, Word modulus) {
    Word s = a + b;
    return (s >= modulus || s < a) ? s - modulus : s;
}

static SparsePoly sample_sparse(int n, int t, Word modulus, std::mt19937_64 &rng) {
    SparsePoly out;
    out.positions.reserve(t);
    out.values.reserve(t);
    std::uniform_int_distribution<int> pos_dist(0, n - 1);
    std::uniform_int_distribution<Word> value_dist(1, modulus - 1);
    std::unordered_set<int> used;
    used.reserve(static_cast<size_t>(t) * 2 + 1);
    while (static_cast<int>(used.size()) < t) {
        int pos = pos_dist(rng);
        if (!used.insert(pos).second) {
            continue;
        }
        out.positions.push_back(pos);
        out.values.push_back(value_dist(rng));
    }
    return out;
}

static SparsePoly sample_sparse_regular(int n, int t, Word modulus, std::mt19937_64 &rng) {
    SparsePoly out;
    out.positions.reserve(t);
    out.values.reserve(t);
    int bucket = n / t;
    std::uniform_int_distribution<Word> value_dist(1, modulus - 1);
    for (int b = 0; b < t; ++b) {
        std::uniform_int_distribution<int> offset_dist(0, bucket - 1);
        out.positions.push_back(b * bucket + offset_dist(rng));
        out.values.push_back(value_dist(rng));
    }
    return out;
}


static void add_sparse_to_dense(const SparsePoly &s, std::vector<Word> &out, int n, Word modulus) {
    for (size_t i = 0; i < s.positions.size(); ++i) {
        out[s.positions[i]] = host_mod_add(out[s.positions[i]], s.values[i], modulus);
    }
}

static std::vector<Word> flatten_dense(const std::vector<std::vector<Word>> &polys, int n) {
    std::vector<Word> out(static_cast<size_t>(polys.size()) * static_cast<size_t>(n));
    for (size_t i = 0; i < polys.size(); ++i) {
        std::copy(polys[i].begin(), polys[i].end(),
                  out.begin() + i * static_cast<size_t>(n));
    }
    return out;
}

static void copy_to_device(Word **dst, const std::vector<Word> &src, const char *label) {
    check(cudaMalloc(reinterpret_cast<void **>(dst), src.size() * sizeof(Word)), label);
    check(cudaMemcpy(*dst, src.data(), src.size() * sizeof(Word), cudaMemcpyHostToDevice), label);
}

static void alloc_device(Word **dst, size_t count, const char *label) {
    check(cudaMalloc(reinterpret_cast<void **>(dst), count * sizeof(Word)), label);
}


static void add_vectors(const Word *d_a, const Word *d_b, Word *d_out, int n, Word modulus) {
    dim3 block(256);
    dim3 grid(grid_size(static_cast<size_t>(n), block.x));
    add_vectors_kernel<<<grid, block>>>(d_a, d_b, d_out, static_cast<size_t>(n), modulus);
    check(cudaGetLastError(), "launch add_vectors_kernel");
}


struct OleState {
    OleArgs args;
    int log_degree = 0;
    int log_domain = 0;
    int limb_index = 0;
    ModulusConfig<Word> config = kConfig62;
    Word modulus = kConfig62.modulus;
    HostTables<Word> host_tables;
    DeviceTables<Word> tables;
    std::vector<Word> phi_norm;
    std::vector<Word> post_norm;
    std::vector<std::vector<Word>> a;
    std::vector<uint64_t> noise_binding[2];
    std::vector<std::vector<SparsePoly>> e;
    std::vector<ringlpn_spfss_zp::GPUDPFZpKey> keys0;
    std::vector<ringlpn_spfss_zp::GPUDPFZpKey> keys1;
    size_t spfss_pair_key_bytes = 0;

    Word *d_a = nullptr;
    Word *d_a_ntt = nullptr;   // cached forward NTT of the public a vector
    Word *d_e0 = nullptr;
    Word *d_e1 = nullptr;
    Word *d_aa_lhs = nullptr;
    std::unique_ptr<ringlpn_ole_party::RingOlePartyContext> party_context0;
    std::unique_ptr<ringlpn_ole_party::RingOlePartyContext> party_context1;
    Word *d_aa_rhs = nullptr;
    Word *d_aa = nullptr;
    Word *d_aa_ntt = nullptr;  // cached forward NTT of the a_i*a_j products
    Word *d_aw = nullptr;
    Word *d_bw = nullptr;
    Word *d_cw = nullptr;
    Word *d_terms = nullptr;
    Word *d_x0 = nullptr;
    Word *d_x1 = nullptr;
    Word *d_u0 = nullptr;
    Word *d_u1 = nullptr;
    Word *d_u2n0 = nullptr;
    Word *d_u2n1 = nullptr;
    Word *d_group0 = nullptr;
    Word *d_group1 = nullptr;
    Word *d_z0 = nullptr;
    Word *d_z1 = nullptr;
    Word *d_zsum = nullptr;
    Word *d_expected = nullptr;

    void cleanup() {
        free_tables(tables);
        cudaFree(d_a);
        cudaFree(d_a_ntt);
        cudaFree(d_aa_ntt);
        cudaFree(d_e0);
        party_context0.reset();
        party_context1.reset();
        cudaFree(d_e1);
        cudaFree(d_aa_lhs);
        cudaFree(d_aa_rhs);
        cudaFree(d_aa);
        cudaFree(d_aw);
        cudaFree(d_bw);
        cudaFree(d_cw);
        cudaFree(d_terms);
        cudaFree(d_x0);
        cudaFree(d_x1);
        cudaFree(d_u0);
        cudaFree(d_u1);
        cudaFree(d_u2n0);
        cudaFree(d_u2n1);
        cudaFree(d_group0);
        cudaFree(d_group1);
        cudaFree(d_z0);
        cudaFree(d_z1);
        cudaFree(d_zsum);
        cudaFree(d_expected);
    }
};
static ringlpn_ole_party::RingOlePublicParams party_public_params(
    const OleState &state);
static ringlpn_ole_party::NoiseRecord party_noise_record(
    const OleState &state, int party);

struct OleLimbResult {
    bool correct = false;
    bool host_validation_ran = false;
    size_t key_bytes = 0;
    double keygen_us = 0.0;
    double key_load_us = 0.0;
};

static std::string ole_artifact_path(const char *prefix,
                                     const OleState &state,
                                     int party,
                                     const char *extension) {
    return std::string(prefix) + state.args.artifact_suffix + "_limb" +
           std::to_string(state.limb_index) + "_p" + std::to_string(party) +
           extension;
}

static void load_two_party_noise(OleState &state, const char *prefix) {
    const int n = state.args.n;
    const int c = state.args.c;
    const int t = state.args.t;
    const bool regular = use_regular_noise(state.args);
    const int bucket = regular ? regular_bucket_size(state.args) : 0;
    state.e.assign(2, std::vector<SparsePoly>(c));
    for (int party = 0; party < 2; ++party) {
        const std::string path =
            ole_artifact_path(prefix, state, party, ".noise");
        ringlpn_keyio::spfss_groups::NoiseRecord rec;
        if (!ringlpn_keyio::spfss_groups::read_noise(path, rec)) {
            std::cerr << "failed to read two-party noise " << path << "\n";
            std::exit(1);
        }
        const size_t expected_terms = static_cast<size_t>(c) * t;
        if (rec.party != party || rec.c != c || rec.t != t ||
            rec.log_domain != state.log_domain ||
            rec.modulus != state.modulus || rec.regular != regular ||
            rec.bucket != bucket || rec.positions.size() != expected_terms ||
            rec.values.size() != expected_terms) {
            std::cerr << "two-party noise record " << path
                      << " does not match this limb\n";
            std::exit(1);
        }
        for (int poly = 0; poly < c; ++poly) {
            SparsePoly &dst = state.e[party][poly];
            dst.positions.reserve(t);
            dst.values.reserve(t);
            std::unordered_set<uint64_t> seen;
            if (!regular) seen.reserve(static_cast<size_t>(t) * 2 + 1);
            for (int k = 0; k < t; ++k) {
                const size_t idx = static_cast<size_t>(poly) * t + k;
                const uint64_t pos = rec.positions[idx];
                const uint64_t value = rec.values[idx];
                const bool position_ok =
                    regular
                        ? pos >= static_cast<uint64_t>(k) * bucket &&
                              pos < static_cast<uint64_t>(k + 1) * bucket
                        : pos < static_cast<uint64_t>(n) &&
                              seen.insert(pos).second;
                if (!position_ok || value == 0 || value >= state.modulus) {
                    std::cerr << "invalid term " << idx
                              << " in two-party noise record " << path << "\n";
                    std::exit(1);
                }
                dst.positions.push_back(static_cast<int>(pos));
                dst.values.push_back(static_cast<Word>(value));
            }
        }
        state.noise_binding[party] =
            ringlpn_keyio::spfss_groups::noise_binding(rec);
        if (state.noise_binding[party].empty()) {
            std::cerr << "cannot bind two-party noise record " << path << "\n";
            std::exit(1);
        }
    }
}

static void build_inputs(OleState &state, bool allocate_omniscient_device_work = true) {
    const int n = state.args.n;
    const int c = state.args.c;
    const Word modulus = state.modulus;
    std::mt19937_64 rng(state.args.seed);

    state.a =
        ringlpn_ole_party::make_public_polynomials(
            party_public_params(state), rng);

    if (const char *prefix = std::getenv("RINGLPN_OLE_NOISE")) {
        load_two_party_noise(state, prefix);
    } else {
        state.e.assign(2, std::vector<SparsePoly>(c));
        for (int party = 0; party < 2; ++party) {
            for (int i = 0; i < c; ++i) {
                state.e[party][i] =
                    use_regular_noise(state.args)
                        ? sample_sparse_regular(n, state.args.t, modulus, rng)
                        : sample_sparse(n, state.args.t, modulus, rng);
            }
        }
    }


    // Artifact hooks are off unless their environment variables are set, so
    // standalone benchmark behaviour and filenames remain unchanged:
    //   RINGLPN_OLE_NOISE=<prefix>        load independently sampled party noise;
    //   RINGLPN_OLE_EXPORT_NOISE=<prefix> export the active party noise;
    //   RINGLPN_OLE_SPFSS_KEYS=<prefix>   load matching two-party SPFSS keys.
    if (const char *prefix = std::getenv("RINGLPN_OLE_EXPORT_NOISE")) {
        for (int party = 0; party < 2; ++party) {
            ringlpn_keyio::spfss_groups::NoiseRecord rec;
            rec.party = party;
            rec.c = c;
            rec.t = state.args.t;
            rec.log_domain = state.log_domain;
            rec.modulus = modulus;
            rec.regular = use_regular_noise(state.args);
            rec.bucket = rec.regular ? regular_bucket_size(state.args) : 0;
            for (int i = 0; i < c; ++i) {
                const SparsePoly &e = state.e[party][i];
                for (int k = 0; k < state.args.t; ++k) {
                    rec.positions.push_back((uint64_t)e.positions[(size_t)k]);
                    rec.values.push_back((uint64_t)e.values[(size_t)k]);
                }
            }
            const std::string path =
                ole_artifact_path(prefix, state, party, ".noise");
            if (!ringlpn_keyio::spfss_groups::write_noise(path, rec)) {
                std::cerr << "failed to write noise record " << path << "\n";
                std::exit(1);
            }
        }
    }
    if (!allocate_omniscient_device_work) return;

    std::vector<std::vector<Word>> e0_dense(c, std::vector<Word>(n, 0));
    std::vector<std::vector<Word>> e1_dense(c, std::vector<Word>(n, 0));
    for (int i = 0; i < c; ++i) {
        add_sparse_to_dense(state.e[0][i], e0_dense[i], n, modulus);
        add_sparse_to_dense(state.e[1][i], e1_dense[i], n, modulus);
    }
    copy_to_device(&state.d_a, flatten_dense(state.a, n), "copy a");
    copy_to_device(&state.d_e0, flatten_dense(e0_dense, n), "copy e0");
    copy_to_device(&state.d_e1, flatten_dense(e1_dense, n), "copy e1");

    const int cc = c * c;
    std::vector<std::vector<Word>> aa_lhs(cc);
    std::vector<std::vector<Word>> aa_rhs(cc);
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < c; ++i) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * c;
            aa_lhs[idx] = state.a[i];
            aa_rhs[idx] = state.a[j];
        }
    }
    copy_to_device(&state.d_aa_lhs, flatten_dense(aa_lhs, n), "copy aa lhs");
    copy_to_device(&state.d_aa_rhs, flatten_dense(aa_rhs, n), "copy aa rhs");

    size_t c_coeffs = static_cast<size_t>(c) * n;
    size_t cc_coeffs = static_cast<size_t>(cc) * n;
    alloc_device(&state.d_aw, std::max(c_coeffs, cc_coeffs), "alloc work a");
    alloc_device(&state.d_bw, std::max(c_coeffs, cc_coeffs), "alloc work b");
    alloc_device(&state.d_cw, std::max(c_coeffs, cc_coeffs), "alloc work c");
    alloc_device(&state.d_terms, cc_coeffs, "alloc terms");
    alloc_device(&state.d_aa, cc_coeffs, "alloc aa");
    alloc_device(&state.d_x0, n, "alloc x0");
    alloc_device(&state.d_x1, n, "alloc x1");
    alloc_device(&state.d_u0, cc_coeffs, "alloc u0");
    alloc_device(&state.d_u1, cc_coeffs, "alloc u1");
    alloc_device(&state.d_u2n0, static_cast<size_t>(2) * n, "alloc u2n0");
    alloc_device(&state.d_u2n1, static_cast<size_t>(2) * n, "alloc u2n1");
    if (use_regular_noise(state.args)) {
        alloc_device(&state.d_group0, spfss_domain_size(state.args), "alloc regular group0");
        alloc_device(&state.d_group1, spfss_domain_size(state.args), "alloc regular group1");
    }
    alloc_device(&state.d_z0, n, "alloc z0");
    alloc_device(&state.d_z1, n, "alloc z1");
    alloc_device(&state.d_zsum, n, "alloc zsum");
    alloc_device(&state.d_expected, n, "alloc expected");

    run_full_polymul(state.d_aa_lhs,
                     state.d_aa_rhs,
                     state.d_aw,
                     state.d_bw,
                     state.d_cw,
                     state.d_aa,
                     state.tables,
                     n,
                     cc,
                     state.log_degree);
    check(cudaDeviceSynchronize(), "sync aa precompute");

    // a and a_i*a_j are fixed for the lifetime of the instance: cache their
    // forward NTTs once so every expand iteration skips half its forward NTTs.
    alloc_device(&state.d_a_ntt, c_coeffs, "alloc a ntt cache");
    alloc_device(&state.d_aa_ntt, cc_coeffs, "alloc aa ntt cache");
    run_forward_only(state.d_a, state.d_a_ntt, state.tables, n, c, state.log_degree);
    run_forward_only(state.d_aa, state.d_aa_ntt, state.tables, n, cc, state.log_degree);
    check(cudaDeviceSynchronize(), "sync ntt caches");
}

// Loads SPFSS keys generated by the two-process dealerless protocol
// (src/test_two_party_spfss_keygen.cpp) in place of the centralized GPU keygen.
// Returns the load time in microseconds; the expansion and validation paths are
// untouched, so a passing run proves the real OLE engine works on dealerless
// keys.
static double load_two_party_spfss_keys(OleState &state, const char *prefix) {
    const int c = state.args.c;
    const int groups = spfss_group_count(state.args);
    const size_t expected_groups = static_cast<size_t>(c) * c * groups;
    auto start = Clock::now();
    state.keys0.clear();
    state.keys1.clear();
    state.spfss_pair_key_bytes = 0;
    for (int party = 0; party < 2; ++party) {
        const std::string path =
            ole_artifact_path(prefix, state, party, ".spfss");
        int file_party = -1;
        int file_levels = 0;
        uint64_t file_modulus = 0;
        std::vector<uint64_t> file_binding;
        std::vector<std::vector<spfss_host::DPFKey>> grouped;
        if (!ringlpn_keyio::spfss_groups::read(
                path, file_party, file_levels, file_modulus, file_binding,
                grouped)) {
            std::cerr << "failed to read two-party SPFSS keys " << path << "\n";
            std::exit(1);
        }
        if (file_party != party || file_levels != state.log_domain ||
            file_modulus != state.modulus || grouped.size() != expected_groups ||
            file_binding != state.noise_binding[party]) {
            std::cerr << "two-party SPFSS key file " << path
                      << " does not match this limb/noise binding (party "
                      << file_party << ", levels " << file_levels << ", groups "
                      << grouped.size() << ")\n";
            std::exit(1);
        }
        const auto noise = party_noise_record(state, party);
        ringlpn_ole_party::RingOlePartyKeys packed;
        if (!ringlpn_ole_party::pack_gpu_party_keys(
                party_public_params(state), party, noise, file_binding,
                grouped, packed)) {
            std::cerr << "invalid two-party SPFSS groups in " << path << "\n";
            std::exit(1);
        }
        ringlpn_ole_party::RingOlePartyCounters counters;
        if (!ringlpn_ole_party::validate_party_keys(
                party_public_params(state), party, noise, packed, &counters) ||
            state.spfss_pair_key_bytes >
                std::numeric_limits<size_t>::max() - counters.key_bytes) {
            std::cerr << "invalid two-party SPFSS key counters in " << path
                      << "\n";
            std::exit(1);
        }
        state.spfss_pair_key_bytes += counters.key_bytes;
        if (party == 0) {
            state.keys0 = std::move(packed.grouped);
        } else {
            state.keys1 = std::move(packed.grouped);
        }
    }
    for (size_t g = 0; g < expected_groups; ++g) {
        if (state.keys0[g].count != state.keys1[g].count) {
            std::cerr << "two-party SPFSS key group " << g
                      << " has mismatched point counts\n";
            std::exit(1);
        }
        const size_t levels = static_cast<size_t>(state.log_domain);
        for (int k = 0; k < state.keys0[g].count; ++k) {
            const size_t idx = static_cast<size_t>(k);
            if (state.keys0[g].seeds[idx] == state.keys1[g].seeds[idx] ||
                state.keys0[g].final_cw[idx] != state.keys1[g].final_cw[idx]) {
                std::cerr << "two-party SPFSS key group " << g
                          << " has invalid private/public material\n";
                std::exit(1);
            }
            for (size_t l = 0; l < levels; ++l) {
                const size_t off = idx * levels + l;
                if (state.keys0[g].s_cw[off] != state.keys1[g].s_cw[off] ||
                    state.keys0[g].t_l_cw[off] !=
                        state.keys1[g].t_l_cw[off] ||
                    state.keys0[g].t_r_cw[off] !=
                        state.keys1[g].t_r_cw[off]) {
                    std::cerr << "two-party SPFSS key group " << g
                              << " has mismatched public corrections\n";
                    std::exit(1);
                }
            }
        }
    }
    auto end = Clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count());
}

static double build_spfss_keys(OleState &state, AESGlobalContext *gaes) {
    const int c = state.args.c;
    const int t = state.args.t;
    const Word modulus = state.modulus;
    const int cc = c * c;
    const int groups = spfss_group_count(state.args);
    const int bucket = use_regular_noise(state.args) ? regular_bucket_size(state.args) : 0;
    if (const char *prefix = std::getenv("RINGLPN_OLE_SPFSS_KEYS")) {
        return load_two_party_spfss_keys(state, prefix);
    }
    state.keys0.resize(static_cast<size_t>(cc) * groups);
    state.keys1.resize(static_cast<size_t>(cc) * groups);
    state.spfss_pair_key_bytes = 0;

    auto start = Clock::now();
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < c; ++i) {
            const SparsePoly &e0 = state.e[0][i];
            const SparsePoly &e1 = state.e[1][j];
            size_t matrix_idx = static_cast<size_t>(i) + static_cast<size_t>(j) * c;
            if (use_regular_noise(state.args)) {
                for (int group = 0; group < groups; ++group) {
                    std::vector<Word> alphas;
                    std::vector<Word> betas;
                    alphas.reserve(t);
                    betas.reserve(t);
                    for (int k = 0; k < t; ++k) {
                        int l = group - k;
                        if (l < 0 || l >= t) {
                            continue;
                        }
                        int off0 = e0.positions[k] - k * bucket;
                        int off1 = e1.positions[l] - l * bucket;
                        alphas.push_back(static_cast<Word>(off0 + off1));
                        betas.push_back(mod_mul_host(e0.values[k], e1.values[l], modulus));
                    }
                    size_t key_idx = matrix_idx * static_cast<size_t>(groups) +
                                     static_cast<size_t>(group);
                    ringlpn_spfss_zp::gpuKeyGenDPFZpPair(
                        alphas,
                        betas,
                        state.log_domain,
                        modulus,
                        state.args.seed ^ (0xA24BAED4963EE407ULL +
                                           key_idx * 0x9E3779B97F4A7C15ULL),
                        gaes,
                        state.keys0[key_idx],
                        state.keys1[key_idx]);
                    state.spfss_pair_key_bytes +=
                        ringlpn_spfss_zp::serializedSizeGPUDPFZpKey(state.keys0[key_idx]) +
                        ringlpn_spfss_zp::serializedSizeGPUDPFZpKey(state.keys1[key_idx]);
                }
            } else {
                std::vector<Word> alphas;
                std::vector<Word> betas;
                alphas.reserve(static_cast<size_t>(t) * t);
                betas.reserve(static_cast<size_t>(t) * t);
                for (int k = 0; k < t; ++k) {
                    for (int l = 0; l < t; ++l) {
                        alphas.push_back(static_cast<Word>(e0.positions[k] + e1.positions[l]));
                        betas.push_back(mod_mul_host(e0.values[k], e1.values[l], modulus));
                    }
                }
                size_t key_idx = matrix_idx;
                ringlpn_spfss_zp::gpuKeyGenDPFZpPair(
                    alphas,
                    betas,
                    state.log_domain,
                    modulus,
                    state.args.seed ^ (0xA24BAED4963EE407ULL +
                                       key_idx * 0x9E3779B97F4A7C15ULL),
                    gaes,
                    state.keys0[key_idx],
                    state.keys1[key_idx]);
                state.spfss_pair_key_bytes +=
                    ringlpn_spfss_zp::serializedSizeGPUDPFZpKey(state.keys0[key_idx]) +
                    ringlpn_spfss_zp::serializedSizeGPUDPFZpKey(state.keys1[key_idx]);
            }
        }
    }
    check(cudaDeviceSynchronize(), "sync SPFSS keygen");
    auto end = Clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

static ringlpn_ole_party::RingOlePublicParams party_public_params(
    const OleState &state) {
    ringlpn_ole_party::RingOlePublicParams params;
    params.n = state.args.n;
    params.c = state.args.c;
    params.t = state.args.t;
    params.log_domain = state.log_domain;
    params.direction = 0;
    params.limb = state.limb_index;
    params.slot_batch = 0;
    params.modulus = state.modulus;
    params.public_a_seed = state.args.seed;
    params.regular = use_regular_noise(state.args);
    return params;
}

static void run_x_phase(OleState &state) {
    const auto params = party_public_params(state);
    ringlpn_ole_party::run_x(params, state.tables, state.d_a_ntt, state.d_e0,
                             state.d_bw, state.d_cw, state.d_terms, state.d_x0);
    ringlpn_ole_party::run_x(params, state.tables, state.d_a_ntt, state.d_e1,
                             state.d_bw, state.d_cw, state.d_terms, state.d_x1);
}

static void run_spfss_eval_phase(OleState &state, AESGlobalContext *gaes) {
    const auto params = party_public_params(state);
    ringlpn_ole_party::run_spfss(params, state.keys0, state.d_group0,
                                 state.d_u2n0, state.d_u0, gaes);
    ringlpn_ole_party::run_spfss(params, state.keys1, state.d_group1,
                                 state.d_u2n1, state.d_u1, gaes);
}

static void run_z_phase(OleState &state) {
    const auto params = party_public_params(state);
    ringlpn_ole_party::run_z(params, state.tables, state.d_aa_ntt, state.d_u0,
                             state.d_bw, state.d_cw, state.d_terms, state.d_z0);
    ringlpn_ole_party::run_z(params, state.tables, state.d_aa_ntt, state.d_u1,
                             state.d_bw, state.d_cw, state.d_terms, state.d_z1);
}

static bool validate_outputs(OleState &state, bool &host_validation_ran) {
    host_validation_ran = false;
    if (state.args.skip_validation) {
        return true;
    }
    int n = state.args.n;
    add_vectors(state.d_z0, state.d_z1, state.d_zsum, n, state.modulus);
    run_full_polymul(state.d_x0,
                     state.d_x1,
                     state.d_aw,
                     state.d_bw,
                     state.d_cw,
                     state.d_expected,
                     state.tables,
                     n,
                     1,
                     state.log_degree);
    check(cudaDeviceSynchronize(), "sync validation");

    std::vector<Word> zsum(n);
    std::vector<Word> expected(n);
    check(cudaMemcpy(zsum.data(), state.d_zsum, sizeof(Word) * n, cudaMemcpyDeviceToHost),
          "copy zsum");
    check(cudaMemcpy(expected.data(), state.d_expected, sizeof(Word) * n, cudaMemcpyDeviceToHost),
          "copy expected");
    bool gpu_ok = compare_vectors(expected, zsum, n, "OLE z0+z1");

    if (n <= 8192) {
        std::vector<Word> x0(n);
        std::vector<Word> x1(n);
        check(cudaMemcpy(x0.data(), state.d_x0, sizeof(Word) * n, cudaMemcpyDeviceToHost),
              "copy x0");
        check(cudaMemcpy(x1.data(), state.d_x1, sizeof(Word) * n, cudaMemcpyDeviceToHost),
              "copy x1");
        std::vector<Word> host_expected =
            host_polymul_reference(x0,
                                   x1,
                                   state.phi_norm,
                                   state.post_norm,
                                   state.config,
                                   n,
                                   state.log_degree);
        host_validation_ran = true;
        gpu_ok = compare_vectors(host_expected, zsum, n, "host OLE oracle") && gpu_ok;
    }
    return gpu_ok;
}
static ringlpn_ole_party::NoiseRecord party_noise_record(const OleState &state,
                                                         int party) {
    ringlpn_ole_party::NoiseRecord noise;
    noise.party = party;
    noise.c = state.args.c;
    noise.t = state.args.t;
    noise.log_domain = state.log_domain;
    noise.modulus = state.modulus;
    noise.regular = use_regular_noise(state.args);
    noise.bucket = noise.regular ? regular_bucket_size(state.args) : 0;
    noise.positions.reserve(static_cast<size_t>(noise.c) * noise.t);
    noise.values.reserve(static_cast<size_t>(noise.c) * noise.t);
    for (int poly = 0; poly < noise.c; ++poly) {
        for (int k = 0; k < noise.t; ++k) {
            noise.positions.push_back(
                static_cast<uint64_t>(state.e[party][poly].positions[k]));
            noise.values.push_back(
                static_cast<uint64_t>(state.e[party][poly].values[k]));
        }
    }
    return noise;
}

static bool initialize_party_contexts(OleState &state) {
    const auto params = party_public_params(state);
    auto noise0 = party_noise_record(state, 0);
    auto noise1 = party_noise_record(state, 1);
    ringlpn_ole_party::RingOlePartyKeys keys0;
    keys0.party = 0;
    keys0.log_domain = state.log_domain;
    keys0.modulus = state.modulus;
    keys0.noise_binding = ringlpn_keyio::spfss_groups::noise_binding(noise0);
    keys0.grouped = std::move(state.keys0);
    ringlpn_ole_party::RingOlePartyKeys keys1;
    keys1.party = 1;
    keys1.log_domain = state.log_domain;
    keys1.modulus = state.modulus;
    keys1.noise_binding = ringlpn_keyio::spfss_groups::noise_binding(noise1);
    keys1.grouped = std::move(state.keys1);

    state.party_context0 =
        std::make_unique<ringlpn_ole_party::RingOlePartyContext>();
    state.party_context1 =
        std::make_unique<ringlpn_ole_party::RingOlePartyContext>();
    if (!state.party_context0->initialize(
            params, 0, noise0, std::move(keys0)) ||
        !state.party_context1->initialize(
            params, 1, noise1, std::move(keys1))) {
        return false;
    }
    const size_t bytes0 = state.party_context0->counters().key_bytes;
    const size_t bytes1 = state.party_context1->counters().key_bytes;
    return bytes0 <= std::numeric_limits<size_t>::max() - bytes1 &&
           bytes0 + bytes1 == state.spfss_pair_key_bytes;
}

static bool validate_party_context_outputs(OleState &state,
                                           bool &host_validation_ran) {
    host_validation_ran = false;
    if (state.args.skip_validation) return true;
    const int n = state.args.n;
    auto &party0 = *state.party_context0;
    auto &party1 = *state.party_context1;
    Word *d_aw = nullptr;
    Word *d_bw = nullptr;
    Word *d_cw = nullptr;
    Word *d_zsum = nullptr;
    Word *d_expected = nullptr;
    alloc_device(&d_aw, n, "alloc party validation work a");
    alloc_device(&d_bw, n, "alloc party validation work b");
    alloc_device(&d_cw, n, "alloc party validation work c");
    alloc_device(&d_zsum, n, "alloc party validation zsum");
    alloc_device(&d_expected, n, "alloc party validation expected");
    add_vectors(party0.device_z_coeff(), party1.device_z_coeff(), d_zsum,
                n, state.modulus);
    run_full_polymul(party0.device_x_coeff(), party1.device_x_coeff(),
                     d_aw, d_bw, d_cw, d_expected, party0.device_tables(),
                     n, 1, state.log_degree);
    check(cudaDeviceSynchronize(), "sync party API validation");

    std::vector<Word> zsum(static_cast<size_t>(n));
    std::vector<Word> expected(static_cast<size_t>(n));
    check(cudaMemcpy(zsum.data(), d_zsum, sizeof(Word) * n,
                     cudaMemcpyDeviceToHost), "copy party API zsum");
    check(cudaMemcpy(expected.data(), d_expected, sizeof(Word) * n,
                     cudaMemcpyDeviceToHost), "copy party API expected");
    bool gpu_ok = compare_vectors(expected, zsum, n, "OLE z0+z1");
    if (n <= 8192) {
        std::vector<Word> x0(static_cast<size_t>(n));
        std::vector<Word> x1(static_cast<size_t>(n));
        check(cudaMemcpy(x0.data(), party0.device_x_coeff(), sizeof(Word) * n,
                         cudaMemcpyDeviceToHost), "copy party API x0");
        check(cudaMemcpy(x1.data(), party1.device_x_coeff(), sizeof(Word) * n,
                         cudaMemcpyDeviceToHost), "copy party API x1");
        const std::vector<Word> host_expected = host_polymul_reference(
            x0, x1, party0.phi_norm(), party0.post_norm(), party0.config(),
            n, state.log_degree);
        host_validation_ran = true;
        gpu_ok = compare_vectors(host_expected, zsum, n, "host OLE oracle") &&
                 gpu_ok;
    }
    cudaFree(d_aw);
    cudaFree(d_bw);
    cudaFree(d_cw);
    cudaFree(d_zsum);
    cudaFree(d_expected);
    return gpu_ok;
}


static void init_ole_state(OleState &state,
                           const OleArgs &args,
                           const ModulusConfig<Word> &config,
                           int limb_index,
                           bool allocate_omniscient_device_work = true) {
    state.args = args;
    state.args.seed = mix_seed(args.seed, static_cast<uint64_t>(limb_index));
    state.log_degree = log2i(args.n);
    state.log_domain = log2i(spfss_domain_size(args));
    state.limb_index = limb_index;
    state.config = config;
    state.modulus = config.modulus;
    if (allocate_omniscient_device_work) {
        compute_cheddar_tables(state.host_tables, args.n, config);
        alloc_and_copy(state.tables, state.host_tables);
        compute_reference_vectors(state.phi_norm, state.post_norm, args.n, config);
    }
    build_inputs(state, allocate_omniscient_device_work);
}

static OleLimbResult run_initial_ole_limb(OleState &state, AESGlobalContext *gaes) {
    OleLimbResult result;
    const double key_time_us = build_spfss_keys(state, gaes);
    if (std::getenv("RINGLPN_OLE_SPFSS_KEYS")) {
        result.keygen_us = -1.0;
        result.key_load_us = key_time_us;
    } else {
        result.keygen_us = key_time_us;
    }
    result.key_bytes = state.spfss_pair_key_bytes;
    if (state.d_e0 != nullptr) {
        run_x_phase(state);
        run_spfss_eval_phase(state, gaes);
        run_z_phase(state);
        check(cudaDeviceSynchronize(), "sync initial omniscient OLE run");
        result.correct =
            validate_outputs(state, result.host_validation_ran);
    } else {
        if (!initialize_party_contexts(state) ||
            !state.party_context0->expand_device(gaes) ||
            !state.party_context1->expand_device(gaes)) {
            std::cerr << "failed to initialize party-local Ring-OLE expansion\n";
            std::exit(1);
        }
        check(cudaDeviceSynchronize(), "sync initial party API OLE run");
        result.correct =
            validate_party_context_outputs(state, result.host_validation_ran);
    }
    return result;
}

static int run_benchmark(const OleArgs &args) {
    initGPUMemPool();
    AESGlobalContext gaes;
    initAESContext(&gaes);

    std::vector<ModulusConfig<Word>> configs = ole_modulus_configs(args.qbits);
    std::vector<std::unique_ptr<OleState>> states;
    states.reserve(configs.size());
    bool correct = true;
    bool host_validation_ran = true;
    size_t key_bytes = 0;
    double keygen_us = 0.0;
    double key_load_us = 0.0;

    for (size_t limb = 0; limb < configs.size(); ++limb) {
        auto state = std::make_unique<OleState>();
        init_ole_state(*state, args, configs[limb], static_cast<int>(limb),
                       true);
        OleLimbResult limb_result = run_initial_ole_limb(*state, &gaes);
        correct = limb_result.correct && correct;
        host_validation_ran = limb_result.host_validation_ran && host_validation_ran;
        key_bytes += limb_result.key_bytes;
        if (limb_result.keygen_us < 0.0) {
            keygen_us = -1.0;
        } else if (keygen_us >= 0.0) {
            keygen_us += limb_result.keygen_us;
        }
        key_load_us += limb_result.key_load_us;
        states.push_back(std::move(state));
    }

    for (int iter = 0; iter < args.warmup; ++iter) {
        for (auto &state : states) {
            run_x_phase(*state);
            run_spfss_eval_phase(*state, &gaes);
            run_z_phase(*state);
        }
        check(cudaDeviceSynchronize(), "sync OLE warmup");
    }

    std::vector<double> samples;
    samples.reserve(args.iters);
    for (int iter = 0; iter < args.iters; ++iter) {
        auto start = Clock::now();
        for (auto &state : states) {
            run_x_phase(*state);
            run_spfss_eval_phase(*state, &gaes);
            run_z_phase(*state);
        }
        check(cudaDeviceSynchronize(), "sync OLE iter");
        auto end = Clock::now();
        samples.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()));
    }
    SummaryStats ole = summarize(samples);

    const char *validation =
        args.skip_validation ? "skipped" : (correct ? "pass" : "fail");
    const char *host_validation =
        args.skip_validation ? "skipped" : (host_validation_ran ? "pass" : "skipped");

    const char *noise_source =
        std::getenv("RINGLPN_OLE_NOISE") ? "party_records" : "benchmark_generated";
    const char *key_source =
        std::getenv("RINGLPN_OLE_SPFSS_KEYS") ? "two_party_files" : "centralized";
    std::cout << RINGLPN_DEVICE_LABEL << ",figure2_spfss_" << args.noise << ","
              << args.n << "," << log2i(args.n) << ","
              << log2i(spfss_domain_size(args)) << "," << args.qbits << ","
              << ole_actual_qbits(configs) << "," << args.noise << ","
              << noise_source << "," << key_source << ","
              << spfss_domain_size(args) << "," << args.c << "," << args.t
              << "," << args.chunk_size << "," << args.iters << ","
              << validation << "," << host_validation << "," << key_bytes
              << "," << keygen_us << "," << key_load_us << ","
              << ole.mean_us << "," << ole.stddev_us << ","
              << (args.skip_validation ? -1 : (correct ? 1 : 0)) << "\n";

    for (auto &state : states) {
        state->cleanup();
    }
    freeAESGlobalContext(&gaes);
    check(cudaDeviceSynchronize(), "sync cleanup");
    return (args.skip_validation || correct) ? 0 : 2;
}

}  // namespace

#ifndef RINGLPN_OLE_DISABLE_MAIN
int main(int argc, char **argv) {
    OleArgs args = parse_ole_args(argc, argv);
    if (args.csv_header) {
        std::cout << "device,input_mode,n,logn,log_domain,requested_qbits,actual_qbits,noise_mode,noise_source,key_source,spfss_domain,c,t,chunk_size,iters,validation,host_validation,spfss_pair_key_bytes,spfss_keygen_us,spfss_key_load_us,ole_expand_mean_us,ole_expand_std_us,correct\n";
    }
    return run_benchmark(args);
}
#endif
