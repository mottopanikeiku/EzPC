// TEST-ONLY offline checker for the two-process distributed DPF keygen.
//
// It runs AFTER both parties have exited, reads the two key files and the two
// test-input records, and verifies with the UNCHANGED consumer
// spfss_host::dpfEvalAll that the two key halves reconstruct
// beta * [x == alpha] over the full domain, where alpha = off_0 + off_1 and
// beta = beta_0 * beta_1 mod p.
//
// This program is not part of the protocol: it deliberately sees both parties'
// private inputs, which is why it is a separate binary reading files on disk
// rather than anything either party can call. A corrupted-key negative control
// is included so a vacuous pass is detectable.

#include "dpf_key_io.h"
#include "spfss_host.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Word = spfss_host::Word;

Word mod_add(Word a, Word b, Word m) {
    Word s = a + b;
    if (s >= m || s < a) s -= m;
    return s;
}

Word mod_mul(Word a, Word b, Word m) {
    return (Word)(((__uint128_t)a * b) % m);
}

bool check_pair(const spfss_host::DPFKey &K0, const spfss_host::DPFKey &K1,
                uint64_t alpha, Word beta, Word p) {
    std::vector<Word> out0, out1;
    spfss_host::dpfEvalAll(0, K0, out0);
    spfss_host::dpfEvalAll(1, K1, out1);
    if (out0.size() != out1.size()) return false;
    for (size_t x = 0; x < out0.size(); ++x) {
        const Word got = mod_add(out0[x], out1[x], p);
        const Word want = (x == alpha) ? beta : 0;
        if (got != want) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    std::string prefix = "two_party_dpf";
    bool csv_header = false;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--prefix" && i + 1 < argc) {
            prefix = argv[++i];
        } else if (k == "--csv-header") {
            csv_header = true;
        } else {
            std::fprintf(stderr, "unknown flag %s\n", k.c_str());
            return 2;
        }
    }

    int party0 = -1, party1 = -1, meta0 = -1, meta1 = -1;
    std::vector<spfss_host::DPFKey> K0s, K1s;
    std::vector<ringlpn_keyio::TestInput> in0, in1;
    const bool loaded =
        ringlpn_keyio::read_keys(prefix + "_p0.key", party0, K0s) &&
        ringlpn_keyio::read_keys(prefix + "_p1.key", party1, K1s) &&
        ringlpn_keyio::read_test_inputs(prefix + "_p0.testmeta", meta0, in0) &&
        ringlpn_keyio::read_test_inputs(prefix + "_p1.testmeta", meta1, in1);
    if (!loaded || party0 != 0 || party1 != 1 || meta0 != 0 || meta1 != 1 ||
        K0s.size() != K1s.size() || K0s.size() != in0.size() ||
        in0.size() != in1.size() || K0s.empty()) {
        std::fprintf(stderr,
                     "[two-party-validate] could not load a consistent key/meta "
                     "set for prefix %s\n",
                     prefix.c_str());
        return 2;
    }

    const size_t n = K0s.size();
    size_t pass = 0, fail = 0, mismatched_public = 0;
    for (size_t i = 0; i < n; ++i) {
        const spfss_host::DPFKey &K0 = K0s[i];
        const spfss_host::DPFKey &K1 = K1s[i];
        // The public key material must be identical on both sides; the seeds
        // and control bits must not be.
        bool public_ok = K0.log_domain == K1.log_domain &&
                         K0.modulus == K1.modulus &&
                         K0.finalCW == K1.finalCW && K0.sCW == K1.sCW &&
                         K0.tLCW == K1.tLCW && K0.tRCW == K1.tRCW &&
                         K0.t0 == 0 && K1.t0 == 1 && K0.seed != K1.seed;
        if (!public_ok) ++mismatched_public;

        const Word p = K0.modulus;
        const uint64_t alpha = in0[i].off + in1[i].off;
        const Word beta = mod_mul((Word)in0[i].beta_factor,
                                  (Word)in1[i].beta_factor, p);
        if (public_ok && check_pair(K0, K1, alpha, beta, p)) {
            ++pass;
        } else {
            ++fail;
        }
    }

    // Negative control: flipping one public correction word must break the
    // reconstruction, otherwise the check above is vacuous.
    bool negative_control_failed_as_expected = false;
    {
        spfss_host::DPFKey corrupted = K1s[0];
        corrupted.finalCW = mod_add(corrupted.finalCW, 1, corrupted.modulus);
        const uint64_t alpha = in0[0].off + in1[0].off;
        const Word beta = mod_mul((Word)in0[0].beta_factor,
                                  (Word)in1[0].beta_factor, corrupted.modulus);
        negative_control_failed_as_expected =
            !check_pair(K0s[0], corrupted, alpha, beta, corrupted.modulus);
    }

    const bool all_ok = fail == 0 && mismatched_public == 0 && pass == n &&
                        negative_control_failed_as_expected;
    if (csv_header) {
        std::printf("prefix,keys,pass,fail,public_material_mismatch,"
                    "negative_control,validation\n");
    }
    std::printf("%s,%zu,%zu,%zu,%zu,%s,%s\n", prefix.c_str(), n, pass, fail,
                mismatched_public,
                negative_control_failed_as_expected ? "failed_as_expected"
                                                    : "DID_NOT_FAIL",
                all_ok ? "pass" : "FAIL");
    std::fprintf(stderr,
                 "[two-party-validate] %zu keys: %zu pass, %zu fail, %zu public "
                 "mismatch, negative control %s\n",
                 n, pass, fail, mismatched_public,
                 negative_control_failed_as_expected ? "failed as expected"
                                                     : "DID NOT FAIL");
    return all_ok ? 0 : 1;
}
