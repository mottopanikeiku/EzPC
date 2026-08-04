// TEST-ONLY GPU checker: does the UNMODIFIED GPU evaluator accept keys produced
// by the two-process distributed keygen?
//
// It reads the two key files written by `test_two_party_dpf_keygen --prg gpu-aes`
// (each party writes only its own file), builds the corresponding
// `ringlpn_spfss_zp::GPUDPFZpKey` for each party WITHOUT touching the GPU keygen,
// and runs `ringlpn_spfss_zp::gpuDpfZpFullEvalSum` - the same expand-side entry
// point the Ring-LPN OLE engine uses. Two checks:
//
//   1. batched SPFSS semantics: the two parties' full-domain outputs must sum to
//      sum_b beta_b * [x == alpha_b] over the whole batch;
//   2. per-tree semantics: each single-tree key pair must sum to
//      beta_b * [x == alpha_b].
//
// A corrupted-`final_cw` negative control must fail, so a vacuous pass is
// detectable. Like the host checker this program deliberately sees both parties'
// private inputs, which is why it is a separate offline binary.

#include "dpf_key_io.h"
#include "gpu_spfss_zp.cuh"
#include "spfss_host.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using ringlpn_spfss_zp::GPUDPFZpKey;
using ringlpn_spfss_zp::Word;

constexpr Word kPrime62 = 4611686018326724609ULL;
constexpr Word kPrime62Crt2 = 4611686018309947393ULL;

bool validate_key_set(int file_party0,
                      const std::vector<spfss_host::DPFKey> &keys0,
                      int file_party1,
                      const std::vector<spfss_host::DPFKey> &keys1,
                      std::string &error) {
    if (file_party0 != 0 || file_party1 != 1) {
        error = "key-file party headers are not 0/1";
        return false;
    }
    if (keys0.empty() || keys0.size() != keys1.size()) {
        error = "key files are empty or have different record counts";
        return false;
    }

    const int common_L = keys0[0].log_domain;
    const Word common_modulus = keys0[0].modulus;
    if (common_L < 2 || common_L > 20) {
        error = "unsupported log_domain " + std::to_string(common_L);
        return false;
    }
    if (common_modulus != kPrime62 && common_modulus != kPrime62Crt2) {
        error = "unsupported modulus " +
                std::to_string((unsigned long long)common_modulus);
        return false;
    }

    for (int party = 0; party <= 1; ++party) {
        const std::vector<spfss_host::DPFKey> &keys =
            party == 0 ? keys0 : keys1;
        for (size_t i = 0; i < keys.size(); ++i) {
            const spfss_host::DPFKey &K = keys[i];
            const std::string where =
                "party " + std::to_string(party) + " record " +
                std::to_string(i);
            if (K.log_domain != common_L) {
                error = where + " has heterogeneous log_domain " +
                        std::to_string(K.log_domain) + " (expected " +
                        std::to_string(common_L) + ")";
                return false;
            }
            if (K.modulus != common_modulus) {
                error = where + " has heterogeneous modulus";
                return false;
            }
            const size_t levels = (size_t)common_L;
            if (K.sCW.size() != levels || K.tLCW.size() != levels ||
                K.tRCW.size() != levels) {
                error = where + " has incorrect correction-array lengths";
                return false;
            }
            if (K.t0 != (uint8_t)party) {
                error = where + " has invalid initial control bit";
                return false;
            }
            for (uint8_t t : K.tLCW) {
                if (t > 1) {
                    error = where + " has non-Boolean left correction bit";
                    return false;
                }
            }
            for (uint8_t t : K.tRCW) {
                if (t > 1) {
                    error = where + " has non-Boolean right correction bit";
                    return false;
                }
            }
            if (K.finalCW >= common_modulus) {
                error = where + " has final correction outside the field";
                return false;
            }
        }
    }

    for (size_t i = 0; i < keys0.size(); ++i) {
        const spfss_host::DPFKey &K0 = keys0[i];
        const spfss_host::DPFKey &K1 = keys1[i];
        if (K0.sCW != K1.sCW || K0.tLCW != K1.tLCW ||
            K0.tRCW != K1.tRCW || K0.finalCW != K1.finalCW) {
            error = "paired record " + std::to_string(i) +
                    " has different public correction material";
            return false;
        }
        if (K0.seed == K1.seed) {
            error = "paired record " + std::to_string(i) +
                    " has identical private roots";
            return false;
        }
    }
    error.clear();
    return true;
}

void resize_key_levels(spfss_host::DPFKey &key, int log_domain) {
    key.log_domain = log_domain;
    key.sCW.resize((size_t)log_domain);
    key.tLCW.resize((size_t)log_domain, 0);
    key.tRCW.resize((size_t)log_domain, 0);
}

bool run_invalid_metadata_selftest(
    int p0, const std::vector<spfss_host::DPFKey> &keys0, int p1,
    const std::vector<spfss_host::DPFKey> &keys1) {
    std::string error;
    if (!validate_key_set(p0, keys0, p1, keys1, error)) {
        std::fprintf(stderr,
                     "[two-party-gpu] invalid-metadata selftest requires a "
                     "valid baseline: %s\n",
                     error.c_str());
        return false;
    }
    if (keys0.size() < 2) {
        std::fprintf(stderr,
                     "[two-party-gpu] invalid-metadata selftest requires at "
                     "least two records\n");
        return false;
    }

    const int other_L =
        keys0[0].log_domain == 20 ? keys0[0].log_domain - 1
                                  : keys0[0].log_domain + 1;
    std::vector<spfss_host::DPFKey> within0 = keys0;
    resize_key_levels(within0[1], other_L);
    const bool within_accepted =
        validate_key_set(p0, within0, p1, keys1, error);
    const bool within_rejected =
        !within_accepted &&
        error.find("party 0 record 1 has heterogeneous log_domain") == 0;
    std::fprintf(stderr,
                 "[two-party-gpu] invalid metadata control within-file "
                 "heterogeneous L: %s%s%s\n",
                 within_rejected ? "rejected as expected (" : "CONTROL FAILED (",
                 within_accepted ? "validator accepted mutation" : error.c_str(),
                 ")");

    std::vector<spfss_host::DPFKey> cross1 = keys1;
    for (spfss_host::DPFKey &K : cross1) resize_key_levels(K, other_L);
    const bool cross_accepted =
        validate_key_set(p0, keys0, p1, cross1, error);
    const bool cross_rejected =
        !cross_accepted &&
        error.find("party 1 record 0 has heterogeneous log_domain") == 0;
    std::fprintf(stderr,
                 "[two-party-gpu] invalid metadata control cross-file L "
                 "mismatch: %s%s%s\n",
                 cross_rejected ? "rejected as expected (" : "CONTROL FAILED (",
                 cross_accepted ? "validator accepted mutation" : error.c_str(),
                 ")");
    return within_rejected && cross_rejected;
}

Word mod_add_host(Word a, Word b, Word m) {
    Word s = a + b;
    return (s >= m || s < a) ? (s - m) : s;
}

Word mod_mul_host(Word a, Word b, Word m) {
    return (Word)(((unsigned __int128)a * b) % m);
}

GPUDPFZpKey build_key(int party, const std::vector<spfss_host::DPFKey> &keys,
                      size_t first, size_t count) {
    GPUDPFZpKey k;
    k.party = party;
    k.log_domain = keys[first].log_domain;
    k.count = (int)count;
    k.modulus = keys[first].modulus;
    const size_t levels = (size_t)k.log_domain;
    k.seeds.resize(count);
    k.s_cw.resize(count * levels);
    k.t_l_cw.resize(count * levels);
    k.t_r_cw.resize(count * levels);
    k.final_cw.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const spfss_host::DPFKey &K = keys[first + i];
        k.seeds[i] = (AESBlock)K.seed;
        k.final_cw[i] = K.finalCW;
        for (size_t l = 0; l < levels; ++l) {
            k.s_cw[i * levels + l] = (AESBlock)K.sCW[l];
            k.t_l_cw[i * levels + l] = K.tLCW[l];
            k.t_r_cw[i * levels + l] = K.tRCW[l];
        }
    }
    return k;
}

// Full-domain reconstruction of one key pair set on the GPU.
std::vector<Word> reconstruct(const GPUDPFZpKey &k0, const GPUDPFZpKey &k1,
                              AESGlobalContext *gaes) {
    const Word domain = Word(1) << k0.log_domain;
    Word *d_out = nullptr;
    ringlpn_spfss_zp::cuda_check(
        cudaMalloc(reinterpret_cast<void **>(&d_out),
                   (size_t)domain * sizeof(Word)),
        "alloc gpu out");
    std::vector<Word> out0((size_t)domain), out1((size_t)domain);
    ringlpn_spfss_zp::gpuDpfZpFullEvalSum(k0, d_out, gaes);
    ringlpn_spfss_zp::cuda_check(
        cudaMemcpy(out0.data(), d_out, (size_t)domain * sizeof(Word),
                   cudaMemcpyDeviceToHost),
        "copy out0");
    ringlpn_spfss_zp::gpuDpfZpFullEvalSum(k1, d_out, gaes);
    ringlpn_spfss_zp::cuda_check(
        cudaMemcpy(out1.data(), d_out, (size_t)domain * sizeof(Word),
                   cudaMemcpyDeviceToHost),
        "copy out1");
    cudaFree(d_out);
    std::vector<Word> sum((size_t)domain);
    for (size_t x = 0; x < (size_t)domain; ++x) {
        sum[x] = mod_add_host(out0[x], out1[x], k0.modulus);
    }
    return sum;
}

}  // namespace

int main(int argc, char **argv) {
    std::string prefix = "two_party_dpf_gpu";
    bool csv_header = false;
    bool selftest_invalid = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--prefix" && i + 1 < argc) {
            prefix = argv[++i];
        } else if (a == "--csv-header") {
            csv_header = true;
        } else if (a == "--selftest-invalid") {
            selftest_invalid = true;
        } else {
            std::fprintf(stderr, "unknown flag %s\n", a.c_str());
            return 2;
        }
    }

    int p0 = -1, p1 = -1;
    std::vector<spfss_host::DPFKey> K0s, K1s;
    const bool keys_loaded =
        ringlpn_keyio::read_keys(prefix + "_p0.key", p0, K0s) &&
        ringlpn_keyio::read_keys(prefix + "_p1.key", p1, K1s);
    if (!keys_loaded) {
        std::fprintf(stderr,
                     "[two-party-gpu] rejecting unreadable key set for prefix "
                     "%s before GPU setup\n",
                     prefix.c_str());
        return 2;
    }

    std::string key_error;
    if (!validate_key_set(p0, K0s, p1, K1s, key_error)) {
        std::fprintf(stderr,
                     "[two-party-gpu] rejecting invalid key set for prefix %s "
                     "before GPU setup: %s\n",
                     prefix.c_str(), key_error.c_str());
        return 2;
    }
    if (selftest_invalid) {
        return run_invalid_metadata_selftest(p0, K0s, p1, K1s) ? 0 : 1;
    }

    int m0 = -1, m1 = -1;
    std::vector<ringlpn_keyio::TestInput> in0, in1;
    const bool meta_loaded =
        ringlpn_keyio::read_test_inputs(prefix + "_p0.testmeta", m0, in0) &&
        ringlpn_keyio::read_test_inputs(prefix + "_p1.testmeta", m1, in1);
    if (!meta_loaded || m0 != 0 || m1 != 1 ||
        K0s.size() != in0.size() || in0.size() != in1.size()) {
        std::fprintf(stderr,
                     "[two-party-gpu] inconsistent test metadata for prefix "
                     "%s\n",
                     prefix.c_str());
        return 2;
    }

    const size_t n = K0s.size();
    const Word p = K0s[0].modulus;
    const int L = K0s[0].log_domain;
    const Word domain = Word(1) << L;

    // Diagnostic only. The deployed AES PRG consumes all 128 seed bits and
    // stores the DPF control bit separately. Low-bit coverage is not a gate:
    // requiring both values in a finite random batch would make this
    // correctness test probabilistic; the parity fixture checks both values.
    size_t seed_low_bit_ones = 0;
    for (size_t i = 0; i < n; ++i) {
        seed_low_bit_ones += size_t(K0s[i].seed & 1);
        seed_low_bit_ones += size_t(K1s[i].seed & 1);
    }
    const size_t public_mismatch = 0;

    AESGlobalContext gaes;
    initAESContext(&gaes);

    // Check 1: batched SPFSS semantics over the whole key file.
    const GPUDPFZpKey batch0 = build_key(0, K0s, 0, n);
    const GPUDPFZpKey batch1 = build_key(1, K1s, 0, n);
    const std::vector<Word> batch_sum = reconstruct(batch0, batch1, &gaes);
    std::vector<Word> expected((size_t)domain, 0);
    for (size_t i = 0; i < n; ++i) {
        const uint64_t alpha = in0[i].off + in1[i].off;
        const Word beta =
            mod_mul_host((Word)in0[i].beta_factor, (Word)in1[i].beta_factor, p);
        if (alpha < domain) {
            expected[(size_t)alpha] = mod_add_host(expected[(size_t)alpha], beta, p);
        }
    }
    size_t batch_mismatch = 0;
    for (size_t x = 0; x < (size_t)domain; ++x) {
        if (batch_sum[x] != expected[x]) ++batch_mismatch;
    }

    // Check 2: per-tree semantics.
    size_t tree_pass = 0, tree_fail = 0;
    for (size_t i = 0; i < n; ++i) {
        const GPUDPFZpKey k0 = build_key(0, K0s, i, 1);
        const GPUDPFZpKey k1 = build_key(1, K1s, i, 1);
        const std::vector<Word> got = reconstruct(k0, k1, &gaes);
        const uint64_t alpha = in0[i].off + in1[i].off;
        const Word beta =
            mod_mul_host((Word)in0[i].beta_factor, (Word)in1[i].beta_factor, p);
        bool ok = true;
        for (size_t x = 0; x < (size_t)domain && ok; ++x) {
            const Word want = (x == alpha) ? beta : 0;
            if (got[x] != want) ok = false;
        }
        if (ok) ++tree_pass; else ++tree_fail;
    }

    // Negative control: one corrupted public correction word must break it.
    GPUDPFZpKey bad1 = build_key(1, K1s, 0, 1);
    bad1.final_cw[0] = mod_add_host(bad1.final_cw[0], 1, p);
    const GPUDPFZpKey good0 = build_key(0, K0s, 0, 1);
    const std::vector<Word> bad_sum = reconstruct(good0, bad1, &gaes);
    const uint64_t alpha0 = in0[0].off + in1[0].off;
    const Word beta0 =
        mod_mul_host((Word)in0[0].beta_factor, (Word)in1[0].beta_factor, p);
    bool negative_control_failed = false;
    for (size_t x = 0; x < (size_t)domain; ++x) {
        const Word want = (x == alpha0) ? beta0 : 0;
        if (bad_sum[x] != want) {
            negative_control_failed = true;
            break;
        }
    }

    const bool all_ok = public_mismatch == 0 &&
                        batch_mismatch == 0 && tree_fail == 0 &&
                        tree_pass == n && negative_control_failed;
    if (csv_header) {
        std::printf("prefix,keys,log_domain,modulus,root_seed_low_bit_ones,"
                    "public_material_mismatch,batch_sum_mismatch,tree_pass,"
                    "tree_fail,negative_control,gpu_validation\n");
    }
    std::printf("%s,%zu,%d,%llu,%zu,%zu,%zu,%zu,%zu,%s,%s\n", prefix.c_str(), n,
                L, (unsigned long long)p, seed_low_bit_ones, public_mismatch,
                batch_mismatch, tree_pass, tree_fail,
                negative_control_failed ? "failed_as_expected" : "DID_NOT_FAIL",
                all_ok ? "pass" : "FAIL");
    std::fprintf(stderr,
                 "[two-party-gpu] %zu keys at L=%d: batch mismatch %zu, per-tree "
                 "%zu/%zu, root low-bit ones %zu/%zu, public mismatch %zu, "
                 "negative control %s -> %s\n",
                 n, L, batch_mismatch, tree_pass, n, seed_low_bit_ones, 2 * n,
                 public_mismatch,
                 negative_control_failed ? "failed as expected" : "DID NOT FAIL",
                 all_ok ? "pass" : "FAIL");
    return all_ok ? 0 : 1;
}
