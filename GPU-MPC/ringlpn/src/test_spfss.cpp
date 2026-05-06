// test_spfss: unit test for spfss_host. Validates
//   share0[x] + share1[x] == Sum_k beta_k * [x == alpha_k]  (mod p)
// for both single-point (DPF) and m-point sum (SPFSS).

#include "spfss_host.h"

#include <cstdint>
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

Word mod_add(Word a, Word b, Word m) {
    Word s = a + b;
    if (s >= m || s < a) s -= m;
    return s;
}

struct Args {
    int log_domain = 10;
    int m = 16;
    uint64_t seed = 1;
    int trials = 1;
    bool verbose = false;
};

void usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " [--log-domain d] [--m m] [--seed s] [--trials t] [--verbose]\n";
}

Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--log-domain") && i + 1 < argc) a.log_domain = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--m") && i + 1 < argc) a.m = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) a.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--trials") && i + 1 < argc) a.trials = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--verbose")) a.verbose = true;
        else { usage(argv[0]); std::exit(1); }
    }
    if (a.log_domain < 1 || a.log_domain > 20 || a.m < 1 || a.trials < 1) {
        usage(argv[0]); std::exit(1);
    }
    return a;
}

bool run_trial(int log_domain, int m, Word modulus, uint64_t trial_seed, bool verbose) {
    std::mt19937_64 rng(trial_seed);
    uint64_t domain = 1ULL << log_domain;
    std::uniform_int_distribution<uint64_t> alpha_dist(0, domain - 1);
    std::uniform_int_distribution<uint64_t> beta_dist(1, modulus - 1);

    std::vector<uint64_t> alphas;
    std::vector<Word> betas;
    std::unordered_set<uint64_t> used;
    alphas.reserve(m);
    betas.reserve(m);
    while ((int)alphas.size() < m) {
        uint64_t a = alpha_dist(rng);
        if (!used.insert(a).second) continue;
        alphas.push_back(a);
        betas.push_back(beta_dist(rng));
    }

    uint64_t key_rng_state = trial_seed ^ 0xD1B54A32D192ED03ULL;
    spfss_host::SPFSSKey K0, K1;
    spfss_host::spfssGen(alphas, betas, log_domain, modulus, key_rng_state, K0, K1);

    std::vector<Word> out0, out1;
    spfss_host::spfssFullEval(0, K0, out0);
    spfss_host::spfssFullEval(1, K1, out1);

    std::vector<Word> expected((size_t)domain, 0);
    for (int k = 0; k < m; ++k) {
        expected[(size_t)alphas[k]] = mod_add(expected[(size_t)alphas[k]], betas[k], modulus);
    }

    int mismatches = 0;
    uint64_t first_bad = 0;
    Word got_lo = 0, got_hi = 0, want = 0;
    for (uint64_t x = 0; x < domain; ++x) {
        Word got = mod_add(out0[(size_t)x], out1[(size_t)x], modulus);
        Word w = expected[(size_t)x];
        if (got != w) {
            if (mismatches == 0) { first_bad = x; got_lo = out0[(size_t)x]; got_hi = out1[(size_t)x]; want = w; }
            ++mismatches;
        }
    }

    bool pass = (mismatches == 0);
    if (verbose || !pass) {
        std::cerr << "log_domain=" << log_domain << " m=" << m
                  << " trial_seed=" << trial_seed
                  << " mismatches=" << mismatches;
        if (mismatches > 0) {
            std::cerr << " first_bad_idx=" << first_bad
                      << " share0=" << got_lo
                      << " share1=" << got_hi
                      << " want=" << want;
        }
        std::cerr << "\n";
    }
    return pass;
}

} // namespace

int main(int argc, char **argv) {
    Args args = parse(argc, argv);
    const Word modulus = kModulus62;
    int failed = 0;
    for (int i = 0; i < args.trials; ++i) {
        uint64_t s = args.seed + (uint64_t)i;
        bool ok = run_trial(args.log_domain, args.m, modulus, s, args.verbose);
        if (!ok) ++failed;
    }
    std::cout << "log_domain=" << args.log_domain
              << ",m=" << args.m
              << ",trials=" << args.trials
              << ",seed_base=" << args.seed
              << ",p=" << modulus
              << ",spfss_pass=" << (failed == 0 ? 1 : 0)
              << ",failed=" << failed
              << "\n";
    return failed == 0 ? 0 : 1;
}
