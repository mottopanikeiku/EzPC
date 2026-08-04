// Party-local Figure 2 Ring-LPN OLE expansion.
//
// One invocation reads exactly one party's noise record and one party's SPFSS
// key file. It never accepts a peer path and never reconstructs an OLE. This
// diagnostic writer is not the live FC path and is intentionally direction 0
// only; the live two-direction composition samples fresh state in memory.

#include <cuda_runtime.h>

#define RINGLPN_DEVICE_LABEL "cuda_ringlpn_ole_party"
#include "ringlpn_ole_party.cuh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {
using Word = ringlpn_ole_party::Word;

uint64_t mix_public_seed(uint64_t seed, uint64_t tag) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL + (tag << 6) + (tag >> 2);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct PartyArgs {
    int party = -1;
    int n = 8192;
    int c = 2;
    int t = 8;
    int qbits = 64;
    int limb = 0;
    int direction = 0;
    uint64_t public_seed = 1;
    std::string noise_mode = "uniform";
    std::string noise_path;
    std::string key_path;
    std::string out_path;
};

[[noreturn]] void party_usage(const char *prog) {
    std::fprintf(stderr,
                 "usage: %s --party 0|1 --n N --c C --t T --qbits 64|128 "
                 "--limb I --direction 0 --public-seed S "
                 "--noise-mode uniform|regular --noise OWN.noise "
                 "--keys OWN.spfss --out OWN.slots\n",
                 prog);
    std::exit(2);
}

uint64_t parse_party_u64(const char *s, const char *name) {
    if (s == nullptr || *s == '\0' || *s == '-') {
        std::fprintf(stderr, "invalid %s\n", name);
        std::exit(2);
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        std::fprintf(stderr, "invalid %s\n", name);
        std::exit(2);
    }
    return static_cast<uint64_t>(value);
}

int parse_party_int(const char *s, const char *name) {
    const uint64_t value = parse_party_u64(s, name);
    if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "%s is too large\n", name);
        std::exit(2);
    }
    return static_cast<int>(value);
}

bool paths_alias(const std::string &lhs, const std::string &rhs) {
    if (lhs == rhs) return true;
    std::error_code error;
    const bool equivalent = std::filesystem::equivalent(lhs, rhs, error);
    return !error && equivalent;
}

PartyArgs parse_party_args(int argc, char **argv) {
    PartyArgs a;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) party_usage(argv[0]);
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--party")) a.party = parse_party_int(next(), "party");
        else if (!std::strcmp(argv[i], "--n")) a.n = parse_party_int(next(), "n");
        else if (!std::strcmp(argv[i], "--c")) a.c = parse_party_int(next(), "c");
        else if (!std::strcmp(argv[i], "--t")) a.t = parse_party_int(next(), "t");
        else if (!std::strcmp(argv[i], "--qbits")) a.qbits = parse_party_int(next(), "qbits");
        else if (!std::strcmp(argv[i], "--limb")) a.limb = parse_party_int(next(), "limb");
        else if (!std::strcmp(argv[i], "--direction")) a.direction = parse_party_int(next(), "direction");
        else if (!std::strcmp(argv[i], "--public-seed")) a.public_seed = parse_party_u64(next(), "public-seed");
        else if (!std::strcmp(argv[i], "--noise-mode")) a.noise_mode = next();
        else if (!std::strcmp(argv[i], "--noise")) a.noise_path = next();
        else if (!std::strcmp(argv[i], "--keys")) a.key_path = next();
        else if (!std::strcmp(argv[i], "--out")) a.out_path = next();
        else party_usage(argv[0]);
    }
    const int limbs = a.qbits == 128 ? 2 : 1;
    if ((a.party != 0 && a.party != 1) || !is_power_of_two(a.n) ||
        a.n < kMinDegree || a.n > kMaxDegree || a.c <= 0 || a.t <= 0 ||
        a.t > a.n || (a.qbits != 64 && a.qbits != 128) || a.limb < 0 ||
        a.limb >= limbs || a.direction != 0 ||
        (a.noise_mode != "uniform" && a.noise_mode != "regular") ||
        a.noise_path.empty() || a.key_path.empty() || a.out_path.empty() ||
        paths_alias(a.out_path, a.noise_path) ||
        paths_alias(a.out_path, a.key_path) ||
        static_cast<uint64_t>(a.c) * static_cast<uint64_t>(a.c) > 65535ULL ||
        (a.noise_mode == "regular" &&
         (a.n % a.t != 0 || !is_power_of_two(a.t)))) {
        party_usage(argv[0]);
    }
    return a;
}

ringlpn_ole_party::RingOlePublicParams make_public_params(
    const PartyArgs &a, const ModulusConfig<Word> &config) {
    ringlpn_ole_party::RingOlePublicParams params;
    params.n = a.n;
    params.c = a.c;
    params.t = a.t;
    params.direction = a.direction;
    params.limb = a.limb;
    params.slot_batch = 0;
    params.modulus = config.modulus;
    params.public_a_seed =
        mix_public_seed(a.public_seed, static_cast<uint64_t>(a.limb));
    params.regular = a.noise_mode == "regular";
    params.log_domain = ringlpn_ole_party::log2_exact(
        ringlpn_ole_party::domain_size(params));
    return params;
}

bool load_own_noise(const ringlpn_ole_party::RingOlePublicParams &params,
                    int party,
                    const std::string &path,
                    ringlpn_ole_party::NoiseRecord &noise) {
    return ringlpn_keyio::spfss_groups::read_noise(path, noise) &&
           ringlpn_ole_party::validate_party_noise(params, party, noise);
}

bool load_own_keys(const ringlpn_ole_party::RingOlePublicParams &params,
                   int party,
                   const ringlpn_ole_party::NoiseRecord &noise,
                   const std::string &path,
                   ringlpn_ole_party::RingOlePartyKeys &out) {
    int file_party = -1;
    int file_levels = 0;
    uint64_t file_modulus = 0;
    std::vector<uint64_t> file_binding;
    std::vector<std::vector<spfss_host::DPFKey>> grouped;
    if (!ringlpn_keyio::spfss_groups::read(
            path, file_party, file_levels, file_modulus, file_binding, grouped) ||
        file_party != party || file_levels != params.log_domain ||
        file_modulus != params.modulus ||
        file_binding != ringlpn_keyio::spfss_groups::noise_binding(noise)) {
        return false;
    }
    return ringlpn_ole_party::pack_gpu_party_keys(
        params, party, noise, file_binding, grouped, out);
}

void put_le32(std::ostream &o, uint32_t x) {
    for (int i = 0; i < 4; ++i) o.put(static_cast<char>((x >> (8 * i)) & 255));
}
void put_le64(std::ostream &o, uint64_t x) {
    for (int i = 0; i < 8; ++i) o.put(static_cast<char>((x >> (8 * i)) & 255));
}

bool write_slots(const PartyArgs &a, Word modulus,
                 const std::vector<Word> &X, const std::vector<Word> &Z,
                 size_t key_bytes) {
    if (X.size() != static_cast<size_t>(a.n) || Z.size() != X.size()) return false;
    const std::string temporary = a.out_path + ".tmp";
    std::remove(temporary.c_str());
    std::ofstream o(temporary, std::ios::binary | std::ios::trunc);
    if (!o) return false;
    o.write("RLPNSLOT", 8);
    put_le32(o, 1);
    put_le32(o, static_cast<uint32_t>(a.party));
    put_le32(o, static_cast<uint32_t>(a.direction));
    put_le32(o, static_cast<uint32_t>(a.limb));
    put_le32(o, static_cast<uint32_t>(a.qbits));
    put_le32(o, static_cast<uint32_t>(a.n));
    put_le32(o, static_cast<uint32_t>(a.c));
    put_le32(o, static_cast<uint32_t>(a.t));
    put_le32(o, static_cast<uint32_t>(a.noise_mode == "regular"));
    put_le64(o, modulus);
    put_le64(o, a.public_seed);
    put_le64(o, static_cast<uint64_t>(key_bytes));
    put_le64(o, static_cast<uint64_t>(X.size()));
    for (Word x : X) put_le64(o, x);
    for (Word z : Z) put_le64(o, z);
    o.flush();
    const bool complete = o.good();
    o.close();
    if (!complete || o.fail() ||
        std::rename(temporary.c_str(), a.out_path.c_str()) != 0) {
        std::remove(temporary.c_str());
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    const PartyArgs a = parse_party_args(argc, argv);
    initGPUMemPool();
    AESGlobalContext gaes;
    initAESContext(&gaes);
    const ModulusConfig<Word> config =
        a.limb == 0 ? kConfig62 : kConfig62Crt2;
    const auto params = make_public_params(a, config);
    ringlpn_ole_party::NoiseRecord noise;
    ringlpn_ole_party::RingOlePartyKeys keys;
    if (!load_own_noise(params, a.party, a.noise_path, noise) ||
        !load_own_keys(params, a.party, noise, a.key_path, keys)) {
        std::fprintf(stderr, "party-local noise/key record mismatch\n");
        freeAESGlobalContext(&gaes);
        return 2;
    }
    ringlpn_ole_party::RingOlePartyShares shares;
    ringlpn_ole_party::RingOlePartyCounters counters;
    if (!ringlpn_ole_party::expand_ring_ole_party(
            params, a.party, noise, std::move(keys), &gaes, shares, counters)) {
        std::fprintf(stderr, "party-local Ring-OLE expansion failed\n");
        freeAESGlobalContext(&gaes);
        return 2;
    }
    const bool wrote = write_slots(a, params.modulus, shares.X_slots,
                                   shares.Z_slots, counters.key_bytes);
    freeAESGlobalContext(&gaes);
    check(cudaDeviceSynchronize(), "party cleanup sync");
    if (!wrote) {
        std::fprintf(stderr, "failed to write party slot record\n");
        return 2;
    }
    std::fprintf(stderr,
                 "[ole-party] party %d dir %d limb %d expanded %zu slots\n",
                 a.party, a.direction, a.limb, shares.X_slots.size());
    return 0;
}
