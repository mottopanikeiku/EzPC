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

    size_t seed_mismatch = 0, left_mismatch = 0, right_mismatch = 0,
           tag_mismatch = 0;
    for (const Row &r : rows) {
        U128 sl = 0, sr = 0;
        uint8_t tl = 0, tr = 0;
        ringlpn_gpu_prg::gpu_aes_prg_expand(r.seed, sl, tl, sr, tr);
        if (r.seed != ringlpn_gpu_prg::clear_tag(r.seed)) ++seed_mismatch;
        if (sl != r.sl) ++left_mismatch;
        if (sr != r.sr) ++right_mismatch;
        if (tl != r.tl || tr != r.tr) ++tag_mismatch;
    }

    // Control: a one-bit seed change must change both children. Without this a
    // constant-output bug would pass the comparison above on cleared seeds.
    size_t insensitive = 0;
    for (const Row &r : rows) {
        U128 sl0 = 0, sr0 = 0, sl1 = 0, sr1 = 0;
        uint8_t t = 0;
        ringlpn_gpu_prg::gpu_aes_prg_expand(r.seed, sl0, t, sr0, t);
        ringlpn_gpu_prg::gpu_aes_prg_expand(r.seed ^ ((U128)1 << 100), sl1, t,
                                            sr1, t);
        if (sl0 == sl1 || sr0 == sr1) ++insensitive;
    }

    const bool all_ok = left_mismatch == 0 && right_mismatch == 0 &&
                        tag_mismatch == 0 && seed_mismatch == 0 &&
                        insensitive == 0;
    if (csv_header) {
        std::printf("vectors,rows,seed_not_cleared,left_mismatch,right_mismatch,"
                    "tag_mismatch,seed_insensitive,parity\n");
    }
    std::printf("%s,%zu,%zu,%zu,%zu,%zu,%zu,%s\n", path.c_str(), rows.size(),
                seed_mismatch, left_mismatch, right_mismatch, tag_mismatch,
                insensitive, all_ok ? "pass" : "FAIL");
    std::fprintf(stderr,
                 "[gpu-aes-parity] %zu device vectors: left %zu / right %zu / "
                 "tag %zu mismatches, %zu seed-insensitive -> %s\n",
                 rows.size(), left_mismatch, right_mismatch, tag_mismatch,
                 insensitive, all_ok ? "pass" : "FAIL");
    return all_ok ? 0 : 1;
}
