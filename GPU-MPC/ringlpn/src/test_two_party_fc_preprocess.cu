// Live two-process Ring-LPN preprocessing for one Orca forward-FC key.
//
// Each process owns only its private A/B/output-mask shares, Ring-LPN noise,
// distributed SPFSS key halves, Ring-OLE slot shares, and final A||B||C key
// payload. Public values and protocol openings travel through PartyChannel.
// No live process reads the peer's files or private state. After both processes
// exit, --check is an explicitly offline validation oracle: it reads both
// records and drives Orca's unchanged readGPUMatmulKey/gpuMatmulBeaver path.
//
// This is component/composition correctness evidence. It does not establish a
// DPF/Ring-LPN/conversion security proof or a concrete security level.

#include "secure_convert.h"
#include "two_party_spfss.h"

// OpenSSL names a global BUF_MEM type; Llama used the same spelling for an
// unrelated enum value. Keep that third-party collision local to its headers.
#define BUF_MEM LLAMA_BUF_MEM
#include "ringlpn_ole_party.cuh"
#include "orca_fc_ringlpn_keywriter.cuh"
#include "fss/gpu_matmul.h"
#undef BUF_MEM
#include "utils/gpu_mem.h"

#include <cuda_runtime.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace ringlpn_fc_live {

using Clock = std::chrono::steady_clock;
using T = u64;
using Word = uint64_t;
using U128 = ringlpn_2pc::U128;

constexpr std::array<uint8_t, 8> kRecordMagic = {'R', 'L', 'P', 'N', 'F', 'C', '2', 'P'};
constexpr uint32_t kRecordVersion = 1;
constexpr size_t kRecordHeaderBytes = 80;
constexpr size_t kDigestBytes = 32;
constexpr size_t kMaxRecordBytes = size_t(1) << 30;
constexpr size_t kPreflightBytes = 64;

struct Args {
    bool check = false;
    bool csv_header = false;
    bool force_rename_failure = false;  // TEST-ONLY publication control
    int party = -1;
    std::string host = "127.0.0.1";
    int port = 48000;
    uint64_t sid = 0;
    int qbits = 64;
    int bw = 16;
    int rows = 2;
    int inner = 2;
    int cols = 2;
    int ole_n = 8192;
    int ole_c = 2;
    int ole_t = 8;
    std::string noise = "regular";
    std::string out_prefix;
    std::string p0_record;
    std::string p1_record;
};

struct PublicWork {
    MatmulParams matmul{};
    uint64_t size_a = 0;
    uint64_t size_b = 0;
    uint64_t size_c = 0;
    uint64_t cross_terms = 0;
    uint64_t ring_batches = 0;
    int limbs = 0;
    bool regular = false;
};

struct RecordHeader {
    int party = -1;
    uint64_t sid = 0;
    int qbits = 0;
    int bw = 0;
    int rows = 0;
    int inner = 0;
    int cols = 0;
    int ole_n = 0;
    int ole_c = 0;
    int ole_t = 0;
    bool regular = false;
    uint64_t ring_batches = 0;
    uint64_t payload_words = 0;
};

struct Record {
    RecordHeader header;
    std::vector<T> payload;
};

struct Counters {
    uint64_t ring_ole_instances = 0;
    uint64_t slots_used = 0;
    uint64_t dpf_trees = 0;
    uint64_t dpf_string_ots = 0;
    uint64_t dpf_bit_triples = 0;
    uint64_t dpf_scalar_oles = 0;
    uint64_t dpf_logical_opened_bits = 0;
    uint64_t dpf_meaningful_share_bits = 0;
    uint64_t key_bytes = 0;
    uint64_t public_a_words_sent = 0;
    uint64_t derandomization_words_sent = 0;
    ringlpn_2pc::SecureConvertCounters conversion;
    uint64_t protocol_bytes_sent = 0;
    uint64_t protocol_direction_switches = 0;
    double total_us = 0.0;
};

void usage(const char *program) {
    std::fprintf(
        stderr,
        "Usage:\n"
        "  %s --party 0|1 --sid N --out-prefix P [--host H] [--port N] "
        "[--qbits 64|128] [--bw N] [--rows M] [--inner K] [--cols N] "
        "[--ole-n N] [--ole-c N] [--ole-t N] [--noise uniform|regular] "
        "[--csv-header] [--force-rename-failure]\n"
        "  %s --check (--out-prefix P | --p0-record F --p1-record F) "
        "[--csv-header]\n",
        program, program);
}

bool parse_int(const char *text, int &out) {
    try {
        size_t used = 0;
        const long long value = std::stoll(text, &used, 10);
        if (used != std::strlen(text) || value < INT_MIN || value > INT_MAX) return false;
        out = static_cast<int>(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_u64(const char *text, uint64_t &out) {
    try {
        size_t used = 0;
        const unsigned long long value = std::stoull(text, &used, 10);
        if (used != std::strlen(text)) return false;
        out = static_cast<uint64_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_args(int argc, char **argv, Args &args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) return nullptr;
            return argv[++i];
        };
        const char *value = nullptr;
        if (key == "--check") args.check = true;
        else if (key == "--csv-header") args.csv_header = true;
        else if (key == "--force-rename-failure") args.force_rename_failure = true;
        else if (key == "--party" && (value = next())) {
            if (!parse_int(value, args.party)) return false;
        } else if (key == "--host" && (value = next())) args.host = value;
        else if (key == "--port" && (value = next())) {
            if (!parse_int(value, args.port)) return false;
        } else if (key == "--sid" && (value = next())) {
            if (!parse_u64(value, args.sid)) return false;
        } else if (key == "--qbits" && (value = next())) {
            if (!parse_int(value, args.qbits)) return false;
        } else if (key == "--bw" && (value = next())) {
            if (!parse_int(value, args.bw)) return false;
        } else if (key == "--rows" && (value = next())) {
            if (!parse_int(value, args.rows)) return false;
        } else if (key == "--inner" && (value = next())) {
            if (!parse_int(value, args.inner)) return false;
        } else if (key == "--cols" && (value = next())) {
            if (!parse_int(value, args.cols)) return false;
        } else if (key == "--ole-n" && (value = next())) {
            if (!parse_int(value, args.ole_n)) return false;
        } else if (key == "--ole-c" && (value = next())) {
            if (!parse_int(value, args.ole_c)) return false;
        } else if (key == "--ole-t" && (value = next())) {
            if (!parse_int(value, args.ole_t)) return false;
        } else if (key == "--noise" && (value = next())) args.noise = value;
        else if (key == "--out-prefix" && (value = next())) args.out_prefix = value;
        else if (key == "--p0-record" && (value = next())) args.p0_record = value;
        else if (key == "--p1-record" && (value = next())) args.p1_record = value;
        else return false;
    }
    if (args.check) {
        const bool common_prefix = !args.out_prefix.empty() &&
                                   args.p0_record.empty() &&
                                   args.p1_record.empty();
        const bool explicit_records = args.out_prefix.empty() &&
                                      !args.p0_record.empty() &&
                                      !args.p1_record.empty();
        return args.party == -1 && !args.force_rename_failure &&
               (common_prefix || explicit_records);
    }
    return !args.out_prefix.empty() &&
           (args.party == 0 || args.party == 1) && args.sid != 0 &&
           args.port > 0 && args.port < 65535;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t &out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

bool derive_work(const Args &args, PublicWork &work) {
    work = PublicWork{};
    if ((args.party != 0 && args.party != 1) || args.sid == 0 ||
        (args.qbits != 64 && args.qbits != 128) || args.bw <= 2 || args.bw > 32 ||
        args.rows <= 0 || args.inner <= 0 || args.cols <= 0 ||
        args.ole_n <= 0 || args.ole_n > (1 << 20) ||
        args.ole_c <= 0 || args.ole_t <= 0 ||
        (args.noise != "uniform" && args.noise != "regular")) {
        return false;
    }
    work.regular = args.noise == "regular";
    work.limbs = args.qbits == 128 ? 2 : 1;

    uint64_t size_a = 0, size_b = 0, size_c = 0, cross = 0;
    if (!checked_mul(static_cast<uint64_t>(args.rows),
                     static_cast<uint64_t>(args.inner), size_a) ||
        !checked_mul(static_cast<uint64_t>(args.inner),
                     static_cast<uint64_t>(args.cols), size_b) ||
        !checked_mul(static_cast<uint64_t>(args.rows),
                     static_cast<uint64_t>(args.cols), size_c) ||
        !checked_mul(size_c, static_cast<uint64_t>(args.inner), cross) ||
        size_a > static_cast<uint64_t>(INT_MAX) ||
        size_b > static_cast<uint64_t>(INT_MAX) ||
        size_c > static_cast<uint64_t>(INT_MAX) || cross == 0) {
        return false;
    }
    work.size_a = size_a;
    work.size_b = size_b;
    work.size_c = size_c;
    work.cross_terms = cross;
    work.ring_batches =
        1 + (cross - 1) / static_cast<uint64_t>(args.ole_n);
    if (work.ring_batches > static_cast<uint64_t>(INT_MAX) ||
        work.size_a + work.size_b + work.size_c >
            (kMaxRecordBytes - kRecordHeaderBytes - kDigestBytes) / sizeof(T)) {
        return false;
    }
    const int log_domain =
        work.regular
            ? ringlpn_ole_party::log2_exact(2 * (args.ole_n / args.ole_t))
            : ringlpn_ole_party::log2_exact(2 * args.ole_n);
    ringlpn_spfss::SpfssPublicParams sp{
        args.ole_c, args.ole_t, log_domain, ringlpn_orca::kPrime62,
        work.regular, args.sid};
    ringlpn_spfss::SpfssWork sp_work;
    ringlpn_ole_party::RingOlePublicParams ole;
    ole.n = args.ole_n;
    ole.c = args.ole_c;
    ole.t = args.ole_t;
    ole.log_domain = log_domain;
    ole.modulus = ringlpn_orca::kPrime62;
    ole.public_a_seed = 1;
    ole.regular = work.regular;
    if (!ringlpn_spfss::derive_spfss_work(sp, sp_work) ||
        !ringlpn_ole_party::validate_public_params(ole)) {
        return false;
    }

    const U128 modulus = args.qbits == 128
                             ? static_cast<U128>(ringlpn_orca::q128Modulus())
                             : static_cast<U128>(ringlpn_orca::kPrime62);
    if ((static_cast<U128>(args.inner) << (2 * args.bw + 2)) >= modulus) return false;

    MatmulParams p;
    p.batchSz = 1;
    p.M = args.rows;
    p.K = args.inner;
    p.N = args.cols;
    stdInit(p, args.bw, 0);
    if (p.size_A != static_cast<int>(size_a) || p.size_B != static_cast<int>(size_b) ||
        p.size_C != static_cast<int>(size_c)) {
        return false;
    }
    work.matmul = p;
    return true;
}

void put_u32(std::vector<uint8_t> &out, size_t offset, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) out[offset + i] = static_cast<uint8_t>(value >> (8 * i));
}

void put_u64(std::vector<uint8_t> &out, size_t offset, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) out[offset + i] = static_cast<uint8_t>(value >> (8 * i));
}

uint32_t get_u32(const uint8_t *in, size_t offset) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) value |= static_cast<uint32_t>(in[offset + i]) << (8 * i);
    return value;
}

uint64_t get_u64(const uint8_t *in, size_t offset) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) value |= static_cast<uint64_t>(in[offset + i]) << (8 * i);
    return value;
}

std::vector<uint8_t> encode_header(const RecordHeader &h) {
    std::vector<uint8_t> out(kRecordHeaderBytes, 0);
    std::copy(kRecordMagic.begin(), kRecordMagic.end(), out.begin());
    put_u32(out, 8, kRecordVersion);
    put_u32(out, 12, static_cast<uint32_t>(h.party));
    put_u64(out, 16, h.sid);
    put_u32(out, 24, static_cast<uint32_t>(h.qbits));
    put_u32(out, 28, static_cast<uint32_t>(h.bw));
    put_u32(out, 32, static_cast<uint32_t>(h.rows));
    put_u32(out, 36, static_cast<uint32_t>(h.inner));
    put_u32(out, 40, static_cast<uint32_t>(h.cols));
    put_u32(out, 44, static_cast<uint32_t>(h.ole_n));
    put_u32(out, 48, static_cast<uint32_t>(h.ole_c));
    put_u32(out, 52, static_cast<uint32_t>(h.ole_t));
    put_u32(out, 56, h.regular ? 1U : 0U);
    put_u64(out, 60, h.ring_batches);
    put_u64(out, 68, h.payload_words);
    return out;
}

bool decode_header(const uint8_t *in, size_t size, RecordHeader &h) {
    if (size < kRecordHeaderBytes ||
        !std::equal(kRecordMagic.begin(), kRecordMagic.end(), in) ||
        get_u32(in, 8) != kRecordVersion || get_u32(in, 76) != 0) {
        return false;
    }
    h.party = static_cast<int>(get_u32(in, 12));
    h.sid = get_u64(in, 16);
    h.qbits = static_cast<int>(get_u32(in, 24));
    h.bw = static_cast<int>(get_u32(in, 28));
    h.rows = static_cast<int>(get_u32(in, 32));
    h.inner = static_cast<int>(get_u32(in, 36));
    h.cols = static_cast<int>(get_u32(in, 40));
    h.ole_n = static_cast<int>(get_u32(in, 44));
    h.ole_c = static_cast<int>(get_u32(in, 48));
    h.ole_t = static_cast<int>(get_u32(in, 52));
    const uint32_t regular = get_u32(in, 56);
    h.regular = regular == 1;
    h.ring_batches = get_u64(in, 60);
    h.payload_words = get_u64(in, 68);
    return (h.party == 0 || h.party == 1) && h.sid != 0 &&
           (h.qbits == 64 || h.qbits == 128) && h.bw > 2 && h.bw <= 32 &&
           h.rows > 0 && h.inner > 0 && h.cols > 0 && h.ole_n > 0 &&
           h.ole_c > 0 && h.ole_t > 0 && regular <= 1;
}

bool sha256(const uint8_t *data, size_t size,
            std::array<uint8_t, kDigestBytes> &digest) {
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    if (!context) return false;
    unsigned int written = 0;
    const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                    EVP_DigestUpdate(context, data, size) == 1 &&
                    EVP_DigestFinal_ex(context, digest.data(), &written) == 1 &&
                    written == digest.size();
    EVP_MD_CTX_free(context);
    return ok;
}


// The trailing SHA-256 detects accidental record corruption only. It is not
// authentication, is never printed as evidence, and is not a protocol message.
bool serialize_record(const RecordHeader &header, const std::vector<T> &payload,
                      std::vector<uint8_t> &bytes,
                      std::array<uint8_t, kDigestBytes> &digest) {
    if (header.payload_words != payload.size() ||
        payload.size() > (kMaxRecordBytes - kRecordHeaderBytes - kDigestBytes) / sizeof(T)) {
        return false;
    }
    bytes = encode_header(header);
    bytes.resize(kRecordHeaderBytes + payload.size() * sizeof(T));
    size_t cursor = kRecordHeaderBytes;
    for (T word : payload) {
        put_u64(bytes, cursor, static_cast<uint64_t>(word));
        cursor += sizeof(T);
    }
    if (!sha256(bytes.data(), bytes.size(), digest)) return false;
    bytes.insert(bytes.end(), digest.begin(), digest.end());
    return true;
}

bool write_bytes(const std::string &path, const std::vector<uint8_t> &bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.flush();
    const bool complete = out.good();
    out.close();
    return complete && !out.fail();
}

bool read_record(const std::string &path, Record &record) {
    std::error_code ec;
    const uintmax_t file_size = std::filesystem::file_size(path, ec);
    if (ec || file_size < kRecordHeaderBytes + kDigestBytes ||
        file_size > kMaxRecordBytes) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
    in.read(reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    if (!in || in.peek() != std::ifstream::traits_type::eof()) return false;

    RecordHeader header;
    if (!decode_header(bytes.data(), bytes.size(), header)) return false;
    uint64_t payload_bytes = 0;
    if (!checked_mul(header.payload_words, sizeof(T), payload_bytes) ||
        payload_bytes > kMaxRecordBytes ||
        kRecordHeaderBytes + payload_bytes + kDigestBytes != bytes.size()) {
        return false;
    }
    std::array<uint8_t, kDigestBytes> expected{};
    if (!sha256(bytes.data(), kRecordHeaderBytes + static_cast<size_t>(payload_bytes), expected) ||
        !std::equal(expected.begin(), expected.end(),
                    bytes.begin() + kRecordHeaderBytes + payload_bytes)) {
        return false;
    }

    Record parsed;
    parsed.header = header;
    parsed.payload.resize(static_cast<size_t>(header.payload_words));
    for (size_t i = 0; i < parsed.payload.size(); ++i) {
        parsed.payload[i] = static_cast<T>(get_u64(bytes.data(), kRecordHeaderBytes + i * 8));
    }
    record = std::move(parsed);
    return true;
}

std::string record_path(const std::string &prefix, int party) {
    return prefix + "_p" + std::to_string(party) + ".fc";
}

std::array<uint8_t, kPreflightBytes> encode_preflight(const Args &args) {
    std::array<uint8_t, kPreflightBytes> out{};
    std::copy(kRecordMagic.begin(), kRecordMagic.end(), out.begin());
    auto put32 = [&](size_t off, uint32_t v) {
        for (size_t i = 0; i < 4; ++i) out[off + i] = static_cast<uint8_t>(v >> (8 * i));
    };
    auto put64 = [&](size_t off, uint64_t v) {
        for (size_t i = 0; i < 8; ++i) out[off + i] = static_cast<uint8_t>(v >> (8 * i));
    };
    put32(8, kRecordVersion);
    put64(12, args.sid);
    put32(20, static_cast<uint32_t>(args.qbits));
    put32(24, static_cast<uint32_t>(args.bw));
    put32(28, static_cast<uint32_t>(args.rows));
    put32(32, static_cast<uint32_t>(args.inner));
    put32(36, static_cast<uint32_t>(args.cols));
    put32(40, static_cast<uint32_t>(args.ole_n));
    put32(44, static_cast<uint32_t>(args.ole_c));
    put32(48, static_cast<uint32_t>(args.ole_t));
    put32(52, args.noise == "regular" ? 1U : 0U);
    return out;
}

bool agree_preflight(ringlpn_2pc::PartyChannel &channel, const Args &args,
                     bool local_valid) {
    const auto manifest = encode_preflight(args);
    std::array<uint8_t, kPreflightBytes + 1> mine{};
    std::array<uint8_t, kPreflightBytes + 1> peer{};
    std::copy(manifest.begin(), manifest.end(), mine.begin());
    mine.back() = local_valid ? 1 : 0;
    channel.exchange_bytes(mine.data(), peer.data(), mine.size());
    return mine.back() == 1 && peer.back() == 1 &&
           std::equal(mine.begin(), mine.end() - 1, peer.begin());
}

uint64_t derive_sid(uint64_t base, uint64_t kind, uint64_t batch,
                    uint64_t direction, uint64_t limb) {
    uint64_t value = ringlpn_orca::mixSeed(base, kind);
    value = ringlpn_orca::mixSeed(value, batch + 1);
    value = ringlpn_orca::mixSeed(value, direction + 1);
    value = ringlpn_orca::mixSeed(value, limb + 1);
    return value == 0 ? 1 : value;
}

Word modulus_for_limb(int limb) {
    return limb == 0 ? ringlpn_orca::kPrime62 : ringlpn_orca::kPrime62Crt2;
}

U128 garner_lift_q128(Word r0, Word r1) {
    constexpr Word p0 = ringlpn_orca::kPrime62;
    constexpr Word p1 = ringlpn_orca::kPrime62Crt2;
    static const Word inv_p0_mod_p1 = mod_inv<Word>(p0 % p1, p1);
    const Word diff = mod_sub<Word>(r1 % p1, r0 % p1, p1);
    const Word factor = mod_mul_host<Word>(diff, inv_p0_mod_p1, p1);
    return static_cast<U128>(r0) + static_cast<U128>(p0) * factor;
}

size_t matrix_index(const MatmulParams &p, bool first, int row, int col) {
    if (first) {
        return p.rowMaj_A ? static_cast<size_t>(row) * p.K + col
                          : static_cast<size_t>(col) * p.M + row;
    }
    return p.rowMaj_B ? static_cast<size_t>(row) * p.N + col
                      : static_cast<size_t>(col) * p.K + row;
}

std::vector<T> sample_ring_words(size_t count, int bw,
                                 ringlpn_2pc::PartyRandom &random) {
    std::vector<T> values(count);
    const uint64_t mask = ringlpn_orca::ringMask(bw);
    for (T &value : values) value = static_cast<T>(random.u64() & mask);
    return values;
}

bool add_u64(uint64_t &target, uint64_t value) {
    if (target > std::numeric_limits<uint64_t>::max() - value) return false;
    target += value;
    return true;
}

bool generate_ring_ole(const Args &args, const PublicWork &work,
                       uint64_t ring_batch, int direction, int limb,
                       ringlpn_2pc::PartyChannel &channel,
                       ringlpn_2pc::PartyRandom &random,
                       AESGlobalContext *gaes,
                       ringlpn_ole_party::RingOlePartyShares &shares,
                       Counters &counters) {
    const Word modulus = modulus_for_limb(limb);
    const int log_domain = work.regular
                               ? ringlpn_ole_party::log2_exact(2 * (args.ole_n / args.ole_t))
                               : ringlpn_ole_party::log2_exact(2 * args.ole_n);
    const uint64_t spfss_sid = derive_sid(args.sid, 0x5350465353ULL, ring_batch,
                                          static_cast<uint64_t>(direction),
                                          static_cast<uint64_t>(limb));
    ringlpn_spfss::SpfssPublicParams spfss_params{
        args.ole_c, args.ole_t, log_domain, modulus, work.regular, spfss_sid};
    ringlpn_spfss::NoiseRecord noise;
    ringlpn_spfss::SpfssPartyBatch batch;
    bool local_valid = ringlpn_spfss::sample_party_noise(
                           spfss_params, args.party, random, noise) &&
                       ringlpn_spfss::make_party_spfss_batch(
                           args.party, spfss_params, noise, batch);
    if (!ringlpn_spfss::agree_spfss_public_manifest(channel, spfss_params,
                                                     local_valid)) {
        return false;
    }
    ringlpn_spfss::GroupedHostKeys grouped;
    ringlpn_spfss::DpfCounters dpf;
    if (!local_valid || !ringlpn_spfss::generate_party_spfss_keys(
                            args.party, spfss_params, batch, channel, random,
                            grouped, dpf)) {
        return false;
    }
    std::vector<uint64_t> binding;
    if (!ringlpn_spfss::party_noise_binding(noise, args.party, spfss_params,
                                             binding)) {
        return false;
    }

    // Jointly sample the full public Ring-LPN polynomial. Revealing a short
    // PRG seed would restrict `a` to the seed image; instead, each party sends
    // one uniform field-element share per coefficient. Their sum is exactly
    // uniform in Z_p even conditioned on either party's contribution.
    const size_t public_coefficients =
        static_cast<size_t>(args.ole_c) * static_cast<size_t>(args.ole_n);
    if (public_coefficients >
        std::numeric_limits<size_t>::max() / sizeof(Word)) {
        return false;
    }
    std::vector<Word> public_a(public_coefficients);
    std::vector<uint8_t> coin_mine(public_coefficients * sizeof(Word));
    std::vector<uint8_t> coin_peer(coin_mine.size());
    for (size_t coefficient = 0; coefficient < public_coefficients;
         ++coefficient) {
        public_a[coefficient] = random.field(modulus);
        for (size_t byte = 0; byte < sizeof(Word); ++byte) {
            coin_mine[coefficient * sizeof(Word) + byte] =
                static_cast<uint8_t>(public_a[coefficient] >> (8 * byte));
        }
    }
    channel.exchange_bytes(coin_mine.data(), coin_peer.data(), coin_mine.size());
    if (!add_u64(counters.public_a_words_sent, public_coefficients)) return false;
    for (size_t coefficient = 0; coefficient < public_coefficients;
         ++coefficient) {
        Word peer_coefficient = 0;
        for (size_t byte = 0; byte < sizeof(Word); ++byte) {
            peer_coefficient |= static_cast<Word>(
                                    coin_peer[coefficient * sizeof(Word) + byte])
                                << (8 * byte);
        }
        if (peer_coefficient >= modulus) return false;
        public_a[coefficient] =
            mod_add<Word>(public_a[coefficient], peer_coefficient, modulus);
    }

    ringlpn_ole_party::RingOlePublicParams ole_params;
    ole_params.n = args.ole_n;
    ole_params.c = args.ole_c;
    ole_params.t = args.ole_t;
    ole_params.log_domain = log_domain;
    ole_params.direction = direction;
    ole_params.limb = limb;
    ole_params.slot_batch = static_cast<int>(ring_batch);
    ole_params.modulus = modulus;
    ole_params.public_a_seed = derive_sid(
        args.sid, 0x50554241ULL, ring_batch,
        static_cast<uint64_t>(direction), static_cast<uint64_t>(limb));
    ole_params.regular = work.regular;

    ringlpn_ole_party::RingOlePartyKeys keys;
    if (!ringlpn_ole_party::pack_gpu_party_keys(
            ole_params, args.party, noise, binding, grouped, keys)) {
        return false;
    }
    ringlpn_ole_party::RingOlePartyCounters ole_counters;
    if (!ringlpn_ole_party::expand_ring_ole_party(
            ole_params, args.party, noise, std::move(keys), gaes, shares,
            ole_counters, &public_a)) {
        return false;
    }
    ++counters.ring_ole_instances;
    if (!add_u64(counters.dpf_trees, ole_counters.trees) ||
        !add_u64(counters.key_bytes, ole_counters.key_bytes) ||
        !add_u64(counters.dpf_string_ots, dpf.string_ots_128) ||
        !add_u64(counters.dpf_bit_triples, dpf.bit_triples) ||
        !add_u64(counters.dpf_scalar_oles, dpf.scalar_oles) ||
        !add_u64(counters.dpf_logical_opened_bits, dpf.logical_opened_bits) ||
        !add_u64(counters.dpf_meaningful_share_bits,
                 dpf.meaningful_share_bits)) {
        return false;
    }
    return shares.X_slots.size() == static_cast<size_t>(args.ole_n) &&
           shares.Z_slots.size() == static_cast<size_t>(args.ole_n);
}

bool exchange_openings(const Args &args, const PublicWork &work,
                       uint64_t ring_batch, int direction, int limb,
                       const std::vector<T> &a_share,
                       const std::vector<T> &b_share,
                       const ringlpn_ole_party::RingOlePartyShares &ole,
                       ringlpn_2pc::PartyChannel &channel,
                       std::vector<std::vector<Word>> &limb_acc,
                       Counters &counters) {
    const uint64_t start = ring_batch * static_cast<uint64_t>(args.ole_n);
    const uint64_t count = std::min(static_cast<uint64_t>(args.ole_n),
                                    work.cross_terms - start);
    const Word modulus = modulus_for_limb(limb);
    std::vector<uint8_t> mine(static_cast<size_t>(count) * sizeof(Word));
    std::vector<uint8_t> peer(mine.size());
    for (uint64_t local = 0; local < count; ++local) {
        const uint64_t global = start + local;
        const uint64_t output = global / static_cast<uint64_t>(args.inner);
        const int k = static_cast<int>(global % static_cast<uint64_t>(args.inner));
        const int row = static_cast<int>(output / static_cast<uint64_t>(args.cols));
        const int col = static_cast<int>(output % static_cast<uint64_t>(args.cols));
        const size_t a_idx = matrix_index(work.matmul, true, row, k);
        const size_t b_idx = matrix_index(work.matmul, false, k, col);
        const uint64_t own_operand =
            direction == 0
                ? (args.party == 0 ? static_cast<uint64_t>(a_share[a_idx])
                                   : static_cast<uint64_t>(b_share[b_idx]))
                : (args.party == 0 ? static_cast<uint64_t>(b_share[b_idx])
                                   : static_cast<uint64_t>(a_share[a_idx]));
        const Word opening = mod_sub<Word>(own_operand % modulus,
                                           ole.X_slots[static_cast<size_t>(local)],
                                           modulus);
        for (size_t byte = 0; byte < sizeof(Word); ++byte) {
            mine[static_cast<size_t>(local) * sizeof(Word) + byte] =
                static_cast<uint8_t>(opening >> (8 * byte));
        }
    }
    channel.exchange_bytes(mine.data(), peer.data(), mine.size());
    if (!add_u64(counters.derandomization_words_sent, count)) return false;

    for (uint64_t local = 0; local < count; ++local) {
        Word own_open = 0;
        Word peer_open = 0;
        for (size_t byte = 0; byte < sizeof(Word); ++byte) {
            own_open |= static_cast<Word>(mine[static_cast<size_t>(local) * 8 + byte])
                        << (8 * byte);
            peer_open |= static_cast<Word>(peer[static_cast<size_t>(local) * 8 + byte])
                         << (8 * byte);
        }
        if (own_open >= modulus || peer_open >= modulus) return false;
        const Word d = args.party == 0 ? own_open : peer_open;
        const Word e = args.party == 0 ? peer_open : own_open;
        Word cross = 0;
        if (args.party == 0) {
            cross = mod_add<Word>(
                mod_add<Word>(mod_mul_host<Word>(d, e, modulus),
                              mod_mul_host<Word>(e, ole.X_slots[static_cast<size_t>(local)],
                                                 modulus),
                              modulus),
                ole.Z_slots[static_cast<size_t>(local)], modulus);
        } else {
            cross = mod_add<Word>(
                mod_mul_host<Word>(d, ole.X_slots[static_cast<size_t>(local)], modulus),
                ole.Z_slots[static_cast<size_t>(local)], modulus);
        }
        const size_t output = static_cast<size_t>((start + local) /
                                                   static_cast<uint64_t>(args.inner));
        limb_acc[static_cast<size_t>(limb)][output] = mod_add<Word>(
            limb_acc[static_cast<size_t>(limb)][output], cross, modulus);
    }
    if (!add_u64(counters.slots_used, count)) return false;
    return true;
}

void accumulate_local_products(const Args &args, const PublicWork &work,
                               const std::vector<T> &a_share,
                               const std::vector<T> &b_share,
                               std::vector<std::vector<Word>> &limb_acc) {
    for (int row = 0; row < args.rows; ++row) {
        for (int col = 0; col < args.cols; ++col) {
            const size_t output = static_cast<size_t>(row) * args.cols + col;
            for (int k = 0; k < args.inner; ++k) {
                const uint64_t a = static_cast<uint64_t>(
                    a_share[matrix_index(work.matmul, true, row, k)]);
                const uint64_t b = static_cast<uint64_t>(
                    b_share[matrix_index(work.matmul, false, k, col)]);
                for (int limb = 0; limb < work.limbs; ++limb) {
                    const Word modulus = modulus_for_limb(limb);
                    limb_acc[static_cast<size_t>(limb)][output] = mod_add<Word>(
                        limb_acc[static_cast<size_t>(limb)][output],
                        mod_mul_host<Word>(a % modulus, b % modulus, modulus),
                        modulus);
                }
            }
        }
    }
}

bool convert_outputs(const Args &args, const PublicWork &work,
                     const std::vector<std::vector<Word>> &limb_acc,
                     ringlpn_2pc::PartyChannel &channel,
                     ringlpn_2pc::PartyRandom &random,
                     std::vector<T> &converted, Counters &counters) {
    std::vector<U128> lifted(work.size_c);
    for (size_t i = 0; i < lifted.size(); ++i) {
        lifted[i] = work.limbs == 1
                        ? static_cast<U128>(limb_acc[0][i])
                        : garner_lift_q128(limb_acc[0][i], limb_acc[1][i]);
    }
    converted.resize(work.size_c);
    size_t cursor = 0;
    uint64_t chunk_index = 0;
    while (cursor < lifted.size()) {
        const size_t count = std::min(ringlpn_2pc::kMaxSecureConvertBatch,
                                      lifted.size() - cursor);
        std::vector<U128> chunk(lifted.begin() + static_cast<ptrdiff_t>(cursor),
                                lifted.begin() + static_cast<ptrdiff_t>(cursor + count));
        ringlpn_2pc::SecureConvertParams params;
        params.sid = derive_sid(args.sid, 0x434F4E56ULL, chunk_index, 0, 0);
        params.qbits = args.qbits;
        params.bw = args.bw;
        params.count = count;
        std::vector<uint64_t> result;
        ringlpn_2pc::SecureConvertCounters one;
        if (!ringlpn_2pc::secure_convert_batch(params, chunk, channel, random,
                                                result, one) ||
            result.size() != count) {
            return false;
        }
        for (size_t i = 0; i < count; ++i) {
            converted[cursor + i] = static_cast<T>(result[i]);
        }
        auto add_counter = [](uint64_t &target, uint64_t value) {
            return add_u64(target, value);
        };
        if (!add_counter(counters.conversion.conversions, one.conversions) ||
            !add_counter(counters.conversion.edabit_bits, one.edabit_bits) ||
            !add_counter(counters.conversion.dabits, one.dabits) ||
            !add_counter(counters.conversion.triples, one.triples) ||
            !add_counter(counters.conversion.logical_opened_bits,
                         one.logical_opened_bits) ||
            !add_counter(counters.conversion.meaningful_share_bits,
                         one.meaningful_share_bits) ||
            !add_counter(counters.conversion.post_mask_dependencies,
                         one.post_mask_dependencies) ||
            !add_counter(counters.conversion.preflight_bytes_sent,
                         one.preflight_bytes_sent) ||
            !add_counter(counters.conversion.preflight_direction_switches,
                         one.preflight_direction_switches) ||
            !add_counter(counters.conversion.correlation_bytes_sent,
                         one.correlation_bytes_sent) ||
            !add_counter(counters.conversion.correlation_direction_switches,
                         one.correlation_direction_switches) ||
            !add_counter(counters.conversion.online_bytes_sent,
                         one.online_bytes_sent) ||
            !add_counter(counters.conversion.online_direction_switches,
                         one.online_direction_switches)) {
            return false;
        }
        counters.conversion.correlation_microseconds += one.correlation_microseconds;
        counters.conversion.online_microseconds += one.online_microseconds;
        cursor += count;
        ++chunk_index;
    }
    return true;
}

bool publish_record(const Args &args, const PublicWork &work,
                    const std::vector<T> &a_share,
                    const std::vector<T> &b_share,
                    const std::vector<T> &c_share,
                    ringlpn_2pc::PartyChannel &channel) {
    std::vector<T> payload;
    payload.reserve(a_share.size() + b_share.size() + c_share.size());
    payload.insert(payload.end(), a_share.begin(), a_share.end());
    payload.insert(payload.end(), b_share.begin(), b_share.end());
    payload.insert(payload.end(), c_share.begin(), c_share.end());
    RecordHeader header;
    header.party = args.party;
    header.sid = args.sid;
    header.qbits = args.qbits;
    header.bw = args.bw;
    header.rows = args.rows;
    header.inner = args.inner;
    header.cols = args.cols;
    header.ole_n = args.ole_n;
    header.ole_c = args.ole_c;
    header.ole_t = args.ole_t;
    header.regular = work.regular;
    header.ring_batches = work.ring_batches;
    header.payload_words = payload.size();

    const std::string output = record_path(args.out_prefix, args.party);
    const std::string temporary = output + ".tmp";
    std::remove(temporary.c_str());
    std::vector<uint8_t> bytes;
    std::array<uint8_t, kDigestBytes> digest{};
    bool staged = serialize_record(header, payload, bytes, digest) &&
                  write_bytes(temporary, bytes);
    std::error_code permission_error;
    if (staged) {
        std::filesystem::permissions(
            temporary,
            std::filesystem::perms::owner_read |
                std::filesystem::perms::owner_write,
            std::filesystem::perm_options::replace, permission_error);
        staged = !permission_error;
    }
    uint8_t mine = staged ? 1 : 0;
    uint8_t peer = 0;
    channel.exchange_bytes(&mine, &peer, 1);
    if (!staged || peer != 1) {
        std::remove(temporary.c_str());
        return false;
    }

    if (args.force_rename_failure && args.party == 0) {
        std::error_code ec;
        std::filesystem::create_directory(output, ec);
        if (!ec) {
            std::ofstream sentinel(output + "/sentinel", std::ios::binary);
            sentinel << "TEST_ONLY";
        }
    }
    const bool renamed = std::rename(temporary.c_str(), output.c_str()) == 0;
    mine = renamed ? 1 : 0;
    peer = 0;
    channel.exchange_bytes(&mine, &peer, 1);
    if (!renamed || peer != 1) {
        std::remove(temporary.c_str());
        if (renamed) std::remove(output.c_str());
        if (args.force_rename_failure && args.party == 0) {
            std::error_code ec;
            std::filesystem::remove_all(output, ec);
        }
        return false;
    }
    return true;
}

void print_party_header() {
    std::cout
        << "party,qbits,bw,rows,inner,cols,ole_n,ole_c,ole_t,noise,ring_batches,"
        << "ring_ole_instances,slots_used,dpf_trees,dpf_string_ots,dpf_bit_triples,"
        << "dpf_scalar_oles,dpf_logical_opened_bits,dpf_meaningful_share_bits,"
        << "spfss_key_bytes,public_a_words_sent,derandomization_words_sent,"
        << "conversions,conversion_logical_opened_bits,"
        << "conversion_meaningful_share_bits,protocol_bytes_sent,"
        << "protocol_direction_switches,total_us,status\n";
}

int run_party(const Args &args) {
    PublicWork work;
    bool local_valid = derive_work(args, work);
    const std::string output = record_path(args.out_prefix, args.party);
    const std::string temporary = output + ".tmp";
    std::error_code ec;
    if (std::filesystem::exists(output, ec) || std::filesystem::exists(temporary, ec)) {
        local_valid = false;
    }

    ringlpn_2pc::PartyChannel channel(args.party, args.host, args.port,
                                      /*defer_ot_setup=*/true);
    if (!agree_preflight(channel, args, local_valid)) {
        std::remove(temporary.c_str());
        std::fprintf(stderr,
                     "[two-party-fc] public/local preflight rejected before OT/output\n");
        return 2;
    }
    channel.setup_ots();
    const uint64_t protocol_begin_bytes = channel.bytes_sent();
    const uint64_t protocol_begin_switches = channel.direction_switches();
    const auto started = Clock::now();

    initGPUMemPool();
    AESGlobalContext gaes;
    initAESContext(&gaes);
    ringlpn_2pc::PartyRandom random;
    std::vector<T> a_share = sample_ring_words(work.size_a, args.bw, random);
    std::vector<T> b_share = sample_ring_words(work.size_b, args.bw, random);
    std::vector<T> y_share = sample_ring_words(work.size_c, args.bw, random);
    std::vector<std::vector<Word>> limb_acc(
        static_cast<size_t>(work.limbs), std::vector<Word>(work.size_c, 0));
    accumulate_local_products(args, work, a_share, b_share, limb_acc);

    Counters counters;
    bool ok = true;
    for (uint64_t ring_batch = 0; ok && ring_batch < work.ring_batches; ++ring_batch) {
        for (int direction = 0; ok && direction < 2; ++direction) {
            for (int limb = 0; ok && limb < work.limbs; ++limb) {
                ringlpn_ole_party::RingOlePartyShares shares;
                const bool generated = generate_ring_ole(
                    args, work, ring_batch, direction, limb, channel, random,
                    &gaes, shares, counters);
                uint8_t mine = generated ? 1 : 0;
                uint8_t peer = 0;
                channel.exchange_bytes(&mine, &peer, 1);
                ok = mine == 1 && peer == 1;
                if (ok) {
                    ok = exchange_openings(
                        args, work, ring_batch, direction, limb, a_share,
                        b_share, shares, channel, limb_acc, counters);
                }
            }
        }
    }

    std::vector<T> converted;
    if (ok) ok = convert_outputs(args, work, limb_acc, channel, random,
                                 converted, counters);
    {
        uint8_t mine = ok ? 1 : 0;
        uint8_t peer = 0;
        channel.exchange_bytes(&mine, &peer, 1);
        ok = mine == 1 && peer == 1;
    }
    std::vector<T> c_share(work.size_c);
    if (ok) {
        for (size_t i = 0; i < c_share.size(); ++i) {
            c_share[i] = static_cast<T>(ringlpn_orca::ringAdd(
                static_cast<uint64_t>(converted[i]),
                static_cast<uint64_t>(y_share[i]), args.bw));
        }
    }
    freeAESGlobalContext(&gaes);

    if (ok) {
        ok = publish_record(args, work, a_share, b_share, c_share, channel);
    }
    counters.protocol_bytes_sent = channel.bytes_sent() - protocol_begin_bytes;
    counters.protocol_direction_switches =
        channel.direction_switches() - protocol_begin_switches;
    counters.total_us = std::chrono::duration<double, std::micro>(
                            Clock::now() - started)
                            .count();

    if (args.csv_header) print_party_header();
    std::cout << args.party << ',' << args.qbits << ',' << args.bw << ','
              << args.rows << ',' << args.inner << ',' << args.cols << ','
              << args.ole_n << ',' << args.ole_c << ',' << args.ole_t << ','
              << args.noise << ',' << work.ring_batches << ','
              << counters.ring_ole_instances << ',' << counters.slots_used << ','
              << counters.dpf_trees << ',' << counters.dpf_string_ots << ','
              << counters.dpf_bit_triples << ',' << counters.dpf_scalar_oles << ','
              << counters.dpf_logical_opened_bits << ','
              << counters.dpf_meaningful_share_bits << ',' << counters.key_bytes
              << ',' << counters.public_a_words_sent << ','
              << counters.derandomization_words_sent << ','
              << counters.conversion.conversions << ','
              << counters.conversion.logical_opened_bits << ','
              << counters.conversion.meaningful_share_bits << ','
              << counters.protocol_bytes_sent << ','
              << counters.protocol_direction_switches << ',' << counters.total_us
              << ',' << (ok ? "pass" : "FAIL") << '\n';
    return ok ? 0 : 1;
}

bool headers_match(const RecordHeader &a, const RecordHeader &b) {
    return a.party == 0 && b.party == 1 && a.sid == b.sid &&
           a.qbits == b.qbits && a.bw == b.bw && a.rows == b.rows &&
           a.inner == b.inner && a.cols == b.cols && a.ole_n == b.ole_n &&
           a.ole_c == b.ole_c && a.ole_t == b.ole_t &&
           a.regular == b.regular && a.ring_batches == b.ring_batches &&
           a.payload_words == b.payload_words;
}

template <typename V>
void copy_to_gpu(const V &source, T **destination) {
    check(cudaMalloc(reinterpret_cast<void **>(destination),
                     source.size() * sizeof(T)), "two-party FC checker cudaMalloc");
    check(cudaMemcpy(*destination, source.data(), source.size() * sizeof(T),
                     cudaMemcpyHostToDevice), "two-party FC checker H2D");
}

int run_check(const Args &args) {
    Record p0;
    Record p1;
    const std::string p0_path = args.p0_record.empty()
                                    ? record_path(args.out_prefix, 0)
                                    : args.p0_record;
    const std::string p1_path = args.p1_record.empty()
                                    ? record_path(args.out_prefix, 1)
                                    : args.p1_record;
    bool records_ok = read_record(p0_path, p0) &&
                      read_record(p1_path, p1) &&
                      headers_match(p0.header, p1.header);
    PublicWork work;
    Args public_args;
    if (records_ok) {
        public_args.party = 0;
        public_args.sid = p0.header.sid;
        public_args.qbits = p0.header.qbits;
        public_args.bw = p0.header.bw;
        public_args.rows = p0.header.rows;
        public_args.inner = p0.header.inner;
        public_args.cols = p0.header.cols;
        public_args.ole_n = p0.header.ole_n;
        public_args.ole_c = p0.header.ole_c;
        public_args.ole_t = p0.header.ole_t;
        public_args.noise = p0.header.regular ? "regular" : "uniform";
        records_ok = derive_work(public_args, work) &&
                     work.ring_batches == p0.header.ring_batches &&
                     p0.header.payload_words ==
                         work.size_a + work.size_b + work.size_c;
    }
    if (!records_ok) {
        std::fprintf(stderr, "[two-party-fc-check] record validation failed\n");
        return 1;
    }

    const size_t a_off = 0;
    const size_t b_off = static_cast<size_t>(work.size_a);
    const size_t c_off = b_off + static_cast<size_t>(work.size_b);
    std::vector<T> mask_a(work.size_a);
    std::vector<T> mask_b(work.size_b);
    std::vector<T> c_sum(work.size_c);
    for (size_t i = 0; i < mask_a.size(); ++i) {
        mask_a[i] = static_cast<T>(ringlpn_orca::ringAdd(
            p0.payload[a_off + i], p1.payload[a_off + i], public_args.bw));
    }
    for (size_t i = 0; i < mask_b.size(); ++i) {
        mask_b[i] = static_cast<T>(ringlpn_orca::ringAdd(
            p0.payload[b_off + i], p1.payload[b_off + i], public_args.bw));
    }
    for (size_t i = 0; i < c_sum.size(); ++i) {
        c_sum[i] = static_cast<T>(ringlpn_orca::ringAdd(
            p0.payload[c_off + i], p1.payload[c_off + i], public_args.bw));
    }

    std::vector<T> output_mask(work.size_c, 0);
    for (int row = 0; row < public_args.rows; ++row) {
        for (int col = 0; col < public_args.cols; ++col) {
            U128 product = 0;
            for (int k = 0; k < public_args.inner; ++k) {
                product += static_cast<U128>(mask_a[matrix_index(work.matmul, true, row, k)]) *
                           static_cast<U128>(mask_b[matrix_index(work.matmul, false, k, col)]);
            }
            const size_t output = static_cast<size_t>(row) * public_args.cols + col;
            output_mask[output] = static_cast<T>(ringlpn_orca::ringSub(
                c_sum[output], ringlpn_orca::ringReduce(product, public_args.bw),
                public_args.bw));
        }
    }

    std::mt19937_64 rng(ringlpn_orca::mixSeed(public_args.sid, 0x434845434BULL));
    std::uniform_int_distribution<uint64_t> dist(
        0, ringlpn_orca::ringMask(public_args.bw));
    std::vector<T> input(work.size_a);
    std::vector<T> weight(work.size_b);
    std::vector<T> masked_input(work.size_a);
    std::vector<T> masked_weight(work.size_b);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<T>(dist(rng));
        masked_input[i] = static_cast<T>(ringlpn_orca::ringAdd(
            input[i], mask_a[i], public_args.bw));
    }
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<T>(dist(rng));
        masked_weight[i] = static_cast<T>(ringlpn_orca::ringAdd(
            weight[i], mask_b[i], public_args.bw));
    }
    std::vector<T> expected(work.size_c);
    for (int row = 0; row < public_args.rows; ++row) {
        for (int col = 0; col < public_args.cols; ++col) {
            U128 product = 0;
            for (int k = 0; k < public_args.inner; ++k) {
                product += static_cast<U128>(input[matrix_index(work.matmul, true, row, k)]) *
                           static_cast<U128>(weight[matrix_index(work.matmul, false, k, col)]);
            }
            const size_t output = static_cast<size_t>(row) * public_args.cols + col;
            expected[output] = static_cast<T>(ringlpn_orca::ringAdd(
                ringlpn_orca::ringReduce(product, public_args.bw),
                output_mask[output], public_args.bw));
        }
    }

    std::vector<uint8_t> key0(p0.payload.size() * sizeof(T));
    std::vector<uint8_t> key1(p1.payload.size() * sizeof(T));
    std::memcpy(key0.data(), p0.payload.data(), key0.size());
    std::memcpy(key1.data(), p1.payload.data(), key1.size());
    uint8_t *cursor0 = key0.data();
    uint8_t *cursor1 = key1.data();
    GPUMatmulKey<T> gkey0 = readGPUMatmulKey<T>(work.matmul,
                                                TruncateType::None, &cursor0);
    GPUMatmulKey<T> gkey1 = readGPUMatmulKey<T>(work.matmul,
                                                TruncateType::None, &cursor1);
    const bool key_order_ok = cursor0 == key0.data() + key0.size() &&
                              cursor1 == key1.data() + key1.size();

    initGPUMemPool();
    T *d_x = nullptr, *d_w = nullptr, *d_a0 = nullptr, *d_a1 = nullptr;
    T *d_b0 = nullptr, *d_b1 = nullptr;
    T *d_mask_a = nullptr, *d_mask_b = nullptr, *d_output_mask = nullptr;
    copy_to_gpu(masked_input, &d_x);
    copy_to_gpu(masked_weight, &d_w);
    copy_to_gpu(std::vector<T>(gkey0.A, gkey0.A + work.matmul.size_A), &d_a0);
    copy_to_gpu(std::vector<T>(gkey1.A, gkey1.A + work.matmul.size_A), &d_a1);
    copy_to_gpu(std::vector<T>(gkey0.B, gkey0.B + work.matmul.size_B), &d_b0);
    copy_to_gpu(std::vector<T>(gkey1.B, gkey1.B + work.matmul.size_B), &d_b1);
    copy_to_gpu(mask_a, &d_mask_a);
    copy_to_gpu(mask_b, &d_mask_b);
    copy_to_gpu(output_mask, &d_output_mask);

    std::vector<uint8_t> dealer_key0(key0.size(), 0);
    std::vector<uint8_t> dealer_key1(key1.size(), 0);
    uint8_t *dealer_cursor0 = dealer_key0.data();
    uint8_t *dealer_cursor1 = dealer_key1.data();
    check(cudaDeviceSynchronize(), "two-party FC checker dealer timer start");
    const auto dealer_start = Clock::now();
    initGPURandomness();
    T *dealer_return0 = gpuKeygenMatmul<T>(
        &dealer_cursor0, SERVER0, work.matmul, d_mask_a, d_mask_b, nullptr,
        TruncateType::None, nullptr, true, d_output_mask);
    destroyGPURandomness();
    initGPURandomness();
    T *dealer_return1 = gpuKeygenMatmul<T>(
        &dealer_cursor1, SERVER1, work.matmul, d_mask_a, d_mask_b, nullptr,
        TruncateType::None, nullptr, true, d_output_mask);
    destroyGPURandomness();
    check(cudaDeviceSynchronize(), "two-party FC checker dealer timer stop");
    const double dealer_keygen_us =
        std::chrono::duration<double, std::micro>(Clock::now() - dealer_start)
            .count();
    const bool dealer_keygen_ok =
        dealer_cursor0 == dealer_key0.data() + dealer_key0.size() &&
        dealer_cursor1 == dealer_key1.data() + dealer_key1.size() &&
        dealer_return0 == d_output_mask && dealer_return1 == d_output_mask;
    Stats stats0;
    Stats stats1;
    check(cudaDeviceSynchronize(), "two-party FC checker online timer start");
    const auto online_start = Clock::now();
    T *d_o0 = gpuMatmulBeaver<T>(work.matmul, gkey0, SERVER0, d_x, d_w,
                                  d_a0, d_b0, nullptr, &stats0);
    T *d_o1 = gpuMatmulBeaver<T>(work.matmul, gkey1, SERVER1, d_x, d_w,
                                  d_a1, d_b1, nullptr, &stats1);
    check(cudaDeviceSynchronize(), "two-party FC checker online timer stop");
    const double online_two_share_us =
        std::chrono::duration<double, std::micro>(Clock::now() - online_start)
            .count();
    std::vector<T> o0(work.size_c);
    std::vector<T> o1(work.size_c);
    check(cudaMemcpy(o0.data(), d_o0, o0.size() * sizeof(T), cudaMemcpyDeviceToHost),
          "two-party FC checker output 0");
    check(cudaMemcpy(o1.data(), d_o1, o1.size() * sizeof(T), cudaMemcpyDeviceToHost),
          "two-party FC checker output 1");
    bool online_ok = key_order_ok && dealer_keygen_ok;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (ringlpn_orca::ringAdd(o0[i], o1[i], public_args.bw) != expected[i]) {
            online_ok = false;
            break;
        }
    }
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_a0); cudaFree(d_a1);
    cudaFree(d_b0); cudaFree(d_b1); cudaFree(d_mask_a); cudaFree(d_mask_b);
    cudaFree(d_output_mask); gpuFree(d_o0); gpuFree(d_o1);

    if (args.csv_header) {
        std::cout
            << "qbits,bw,rows,inner,cols,ring_batches,final_payload_bytes_per_party,"
            << "matched_dealer_keygen_us,checker_two_share_online_us,"
            << "matched_dealer_keygen_contract,key_order,online_contract,status\n";
    }
    std::cout << public_args.qbits << ',' << public_args.bw << ','
              << public_args.rows << ',' << public_args.inner << ','
              << public_args.cols << ',' << work.ring_batches << ','
              << key0.size() << ',' << dealer_keygen_us << ','
              << online_two_share_us << ','
              << (dealer_keygen_ok ? "pass" : "FAIL") << ','
              << (key_order_ok ? "pass" : "FAIL") << ','
              << (online_ok ? "pass" : "FAIL") << ','
              << (online_ok ? "pass" : "FAIL") << '\n';
    return online_ok ? 0 : 1;
}

}  // namespace ringlpn_fc_live

int main(int argc, char **argv) {
    ringlpn_fc_live::Args args;
    if (!ringlpn_fc_live::parse_args(argc, argv, args)) {
        ringlpn_fc_live::usage(argv[0]);
        return 2;
    }
    return args.check ? ringlpn_fc_live::run_check(args)
                      : ringlpn_fc_live::run_party(args);
}
