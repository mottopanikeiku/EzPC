// Gate: the host GPU-AES PRG twin must agree with the device PRG on every bit.
//
// It reads device-dumped vectors (produced by src/dump_gpu_aes_prg_vectors.cu on
// a GPU) and recomputes each row with the host implementation. Any mismatch is a
// hard failure: without this parity the two-party host keygen cannot emit keys
// the unmodified GPU evaluator accepts.

#include "gpu_aes_prg_host.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using ringlpn_gpu_prg::U128;

U128 parse_hex128(const std::string &hi, const std::string &lo) {
    const uint64_t h = std::strtoull(hi.c_str(), nullptr, 16);
    const uint64_t l = std::strtoull(lo.c_str(), nullptr, 16);
    return ((U128)h << 64) | (U128)l;
}

struct Row {
    U128 seed, sl, sr;
    uint8_t tl, tr;
};

}  // namespace

int main(int argc, char **argv) {
    std::string path = "results/dpf/gpu_aes_prg_vectors_2026_07_29.csv";
    bool csv_header = false;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--vectors" && i + 1 < argc) {
            path = argv[++i];
        } else if (k == "--csv-header") {
            csv_header = true;
        } else {
            std::fprintf(stderr, "unknown flag %s\n", k.c_str());
            return 2;
        }
    }

    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "[gpu-aes-parity] cannot open %s\n", path.c_str());
        return 2;
    }

    std::vector<Row> rows;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (first) {  // header
            first = false;
            if (line.find("seed_hi") != std::string::npos) continue;
        }
        std::stringstream ss(line);
        std::string f[8];
        int n = 0;
        while (n < 8 && std::getline(ss, f[n], ',')) ++n;
        if (n != 8) {
            std::fprintf(stderr, "[gpu-aes-parity] malformed row: %s\n",
                         line.c_str());
            return 2;
        }
        Row r;
        r.seed = parse_hex128(f[0], f[1]);
        r.sl = parse_hex128(f[2], f[3]);
        r.tl = (uint8_t)std::strtoul(f[4].c_str(), nullptr, 10);
        r.sr = parse_hex128(f[5], f[6]);
        r.tr = (uint8_t)std::strtoul(f[7].c_str(), nullptr, 10);
        rows.push_back(r);
    }

    if (rows.empty()) {
        std::fprintf(stderr, "[gpu-aes-parity] no vectors in %s\n", path.c_str());
        return 2;
    }

    size_t left_mismatch = 0, right_mismatch = 0, tag_mismatch = 0;
    for (const Row &r : rows) {
        U128 sl = 0, sr = 0;
        uint8_t tl = 0, tr = 0;
        ringlpn_gpu_prg::gpu_aes_prg_expand(r.seed, sl, tl, sr, tr);
        if (sl != r.sl) ++left_mismatch;
        if (sr != r.sr) ++right_mismatch;
        if (tl != r.tl || tr != r.tr) ++tag_mismatch;
    }

    // Controls: the high-bit perturbation catches broad seed insensitivity.
    // More importantly, rows whose seeds differ only in bit 0 must produce
    // different meaningful outputs for both children.  The device vector
    // generator deterministically includes the pair 0 and 1; requiring a pair
    // here makes that precondition non-vacuous.  Comparing both the dumped
    // device outputs and fresh host outputs makes jointly clearing seed bit 0
    // fail even when ordinary host/device parity still agrees.
    size_t insensitive = 0;
    for (const Row &r : rows) {
        U128 sl0 = 0, sr0 = 0, sl1 = 0, sr1 = 0;
        uint8_t t = 0;
        ringlpn_gpu_prg::gpu_aes_prg_expand(r.seed, sl0, t, sr0, t);
        ringlpn_gpu_prg::gpu_aes_prg_expand(r.seed ^ ((U128)1 << 100), sl1, t,
                                            sr1, t);
        if (sl0 == sl1 || sr0 == sr1) ++insensitive;
    }

    size_t low_bit_pairs = 0, low_bit_insensitive = 0;
    for (const Row &r0 : rows) {
        if ((r0.seed & 1) != 0) continue;
        for (const Row &r1 : rows) {
            if (r1.seed != (r0.seed ^ (U128)1)) continue;
            ++low_bit_pairs;
            U128 host_l0 = 0, host_r0 = 0, host_l1 = 0, host_r1 = 0;
            uint8_t host_tl0 = 0, host_tr0 = 0, host_tl1 = 0, host_tr1 = 0;
            ringlpn_gpu_prg::gpu_aes_prg_expand(
                r0.seed, host_l0, host_tl0, host_r0, host_tr0);
            ringlpn_gpu_prg::gpu_aes_prg_expand(
                r1.seed, host_l1, host_tl1, host_r1, host_tr1);
            // Tag bits may legitimately collide.  Each 128-bit child seed word
            // itself must be sensitive to input seed bit 0 on both backends.
            const bool device_left_same = r0.sl == r1.sl;
            const bool device_right_same = r0.sr == r1.sr;
            const bool host_left_same = host_l0 == host_l1;
            const bool host_right_same = host_r0 == host_r1;
            if (device_left_same || device_right_same || host_left_same ||
                host_right_same) {
                ++low_bit_insensitive;
            }
            break;
        }
    }

    const bool all_ok = left_mismatch == 0 && right_mismatch == 0 &&
                        tag_mismatch == 0 && insensitive == 0 &&
                        low_bit_pairs != 0 && low_bit_insensitive == 0;
    if (csv_header) {
        std::printf("vectors,rows,left_mismatch,right_mismatch,tag_mismatch,"
                    "seed_insensitive,low_bit_pairs,low_bit_insensitive,"
                    "parity\n");
    }
    std::printf("%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s\n", path.c_str(),
                rows.size(), left_mismatch, right_mismatch, tag_mismatch,
                insensitive, low_bit_pairs, low_bit_insensitive,
                all_ok ? "pass" : "FAIL");
    std::fprintf(stderr,
                 "[gpu-aes-parity] %zu device vectors: left %zu / right %zu / "
                 "tag %zu mismatches, %zu seed-insensitive; low-bit pairs %zu, "
                 "%zu insensitive -> %s\n",
                 rows.size(), left_mismatch, right_mismatch, tag_mismatch,
                 insensitive, low_bit_pairs, low_bit_insensitive,
                 all_ok ? "pass" : "FAIL");
    return all_ok ? 0 : 1;
}
