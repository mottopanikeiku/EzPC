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
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--prefix" && i + 1 < argc) {
            prefix = argv[++i];
        } else if (a == "--csv-header") {
            csv_header = true;
        } else {
            std::fprintf(stderr, "unknown flag %s\n", a.c_str());
            return 2;
        }
    }

    int p0 = -1, p1 = -1, m0 = -1, m1 = -1;
    std::vector<spfss_host::DPFKey> K0s, K1s;
    std::vector<ringlpn_keyio::TestInput> in0, in1;
    const bool loaded =
        ringlpn_keyio::read_keys(prefix + "_p0.key", p0, K0s) &&
        ringlpn_keyio::read_keys(prefix + "_p1.key", p1, K1s) &&
        ringlpn_keyio::read_test_inputs(prefix + "_p0.testmeta", m0, in0) &&
        ringlpn_keyio::read_test_inputs(prefix + "_p1.testmeta", m1, in1);
    if (!loaded || p0 != 0 || p1 != 1 || m0 != 0 || m1 != 1 ||
        K0s.size() != K1s.size() || K0s.size() != in0.size() ||
        in0.size() != in1.size() || K0s.empty()) {
        std::fprintf(stderr,
                     "[two-party-gpu] inconsistent key/meta set for prefix %s\n",
                     prefix.c_str());
        return 2;
    }

    const size_t n = K0s.size();
    const Word p = K0s[0].modulus;
    const int L = K0s[0].log_domain;
    const Word domain = Word(1) << L;

    // Public material must agree while private roots differ. Count root low
    // bits as a diagnostic: full-width AES seeds no longer reserve that bit.
    size_t seed_low_bit_ones = 0, public_mismatch = 0;
    for (size_t i = 0; i < n; ++i) {
        seed_low_bit_ones += size_t(K0s[i].seed & 1);
        seed_low_bit_ones += size_t(K1s[i].seed & 1);
        if (K0s[i].sCW != K1s[i].sCW || K0s[i].tLCW != K1s[i].tLCW ||
            K0s[i].tRCW != K1s[i].tRCW || K0s[i].finalCW != K1s[i].finalCW ||
            K0s[i].seed == K1s[i].seed || K0s[i].t0 != 0 || K1s[i].t0 != 1) {
            ++public_mismatch;
        }
    }

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

    const bool all_ok = public_mismatch == 0 && batch_mismatch == 0 &&
                        tree_fail == 0 && tree_pass == n &&
                        negative_control_failed;
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
                 "%zu/%zu, root low-bit ones %zu, public mismatch %zu, negative "
                 "control %s -> %s\n",
                 n, L, batch_mismatch, tree_pass, n, seed_low_bit_ones,
                 public_mismatch,
                 negative_control_failed ? "failed as expected" : "DID NOT FAIL",
                 all_ok ? "pass" : "FAIL");
    return all_ok ? 0 : 1;
}
