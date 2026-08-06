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
#include "correlation_freshness.h"
#include "two_party_spfss.h"

// OpenSSL names a global BUF_MEM type; Llama used the same spelling for an
// unrelated enum value. Keep that third-party collision local to its headers.
#define BUF_MEM LLAMA_BUF_MEM
#include "ringlpn_ole_party.cuh"
#include "two_party_spfss_gpu.cuh"
#include "orca_fc_ringlpn_keywriter.cuh"
#include "fss/gpu_matmul.h"
#include "fss/gpu_conv2d.h"
#undef BUF_MEM
#include "utils/gpu_mem.h"

#include <cuda_runtime.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <set>
#include <chrono>
#include <climits>
#include <cmath>
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
#include <sys/resource.h>

#ifdef RINGLPN_LIVE_CONV
namespace ringlpn_conv_live {
#else
namespace ringlpn_fc_live {
#endif

using Clock = std::chrono::steady_clock;
using T = u64;
using Word = uint64_t;
using U128 = ringlpn_2pc::U128;

#ifdef RINGLPN_LIVE_CONV
constexpr std::array<uint8_t, 8> kRecordMagic = {'R', 'L', 'P', 'N', 'C', 'V', '2', 'P'};
constexpr uint32_t kRecordVersion = 3;
constexpr size_t kLayerManifestBytes = 144;
constexpr size_t kRecordHeaderBytes = 224;
constexpr size_t kDigestBytes = 32;
constexpr size_t kMaxRecordBytes = size_t(1) << 30;
constexpr size_t kPreflightBytes = 192;
constexpr int kConvN = 1;
constexpr int kConvH = 4;
constexpr int kConvW = 4;
constexpr int kConvCI = 1;
constexpr int kConvFH = 3;
constexpr int kConvFW = 3;
constexpr int kConvCO = 2;
constexpr int kConvPadding = 1;
constexpr int kConvStride = 1;
#else
constexpr std::array<uint8_t, 8> kRecordMagic = {'R', 'L', 'P', 'N', 'F', 'C', '2', 'P'};
constexpr uint32_t kRecordVersion = 3;
constexpr size_t kLayerManifestBytes = 112;
constexpr size_t kRecordHeaderBytes = 176;
constexpr size_t kDigestBytes = 32;
constexpr size_t kMaxRecordBytes = size_t(1) << 30;
constexpr size_t kPreflightBytes = 160;
#endif

struct Args {
    bool check = false;
    bool csv_header = false;
    bool force_rename_failure = false;  // TEST-ONLY publication control
    int party = -1;
    std::string host = "127.0.0.1";
    int port = 48000;
    uint64_t sid = 0;
    ringlpn_freshness::InvocationId invocation_id{};
    std::string invocation_id_text;
    std::string ledger_path;
    int qbits = 64;
    int bw = 16;
#ifdef RINGLPN_LIVE_CONV
    int rows = kConvN;
    int inner = kConvH;
    int cols = kConvW;
    int ci = kConvCI;
    int fh = kConvFH;
    int fw = kConvFW;
    int co = kConvCO;
    int padding = kConvPadding;
    int stride = kConvStride;
#else
    int rows = 2;
    int inner = 2;
    int cols = 2;
#endif
    int ole_n = 8192;
    int ole_c = 2;
    int ole_t = 8;
    std::string noise = "regular";
    std::string channel = "local-loopback";
    std::string ot_backend = "sci-iknp";
    std::string emp_silent_bridge;
    std::string out_prefix;
    std::string p0_record;
    std::string p1_record;
};
#ifdef RINGLPN_LIVE_CONV
struct ConvTerm {
    size_t output = 0;
    size_t input = 0;
    size_t filter = 0;
};
#endif

struct PublicWork {
    MatmulParams matmul{};
#ifdef RINGLPN_LIVE_CONV
    GPUConv2DKey<T> conv{};
    std::vector<uint64_t> conv_spatial_prefix;
    uint64_t conv_terms_per_image = 0;
#endif
    uint64_t size_a = 0;
    uint64_t size_b = 0;
    uint64_t size_c = 0;
    uint64_t cross_terms = 0;
    uint64_t ring_batches = 0;
    uint64_t ring_application_slots = 0;
    uint64_t ring_bootstrap_slots = 0;
    int limbs = 0;
    bool regular = false;
};

struct RecordHeader {
    int party = -1;
    uint64_t sid = 0;
    ringlpn_freshness::InvocationId invocation_id{};
    ringlpn_freshness::Digest ledger_digest{};
    ringlpn_2pc::OtBackend ot_backend = ringlpn_2pc::OtBackend::SciIknp;
    int qbits = 0;
    int bw = 0;
    int rows = 0;
    int inner = 0;
    int cols = 0;
#ifdef RINGLPN_LIVE_CONV
    int ci = 0;
    int fh = 0;
    int fw = 0;
    int co = 0;
    int padding = 0;
    int stride = 0;
#endif
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
    uint64_t ring_application_slots_discarded = 0;
    uint64_t dpf_trees = 0;
    uint64_t dpf_string_ots = 0;
    uint64_t dpf_bit_triples = 0;
    uint64_t dpf_scalar_oles = 0;
    uint64_t dpf_epoch_zero_scalar_oles = 0;
    uint64_t dpf_pcg_scalar_oles = 0;
    uint64_t dpf_pcg_oles_reserved = 0;
    uint64_t dpf_pcg_oles_discarded = 0;
    uint64_t dpf_pcg_opening_words_sent = 0;
    uint64_t dpf_logical_opened_bits = 0;
    uint64_t dpf_meaningful_share_bits = 0;
    uint64_t key_bytes = 0;
    uint64_t public_a_words_sent = 0;
    uint64_t derandomization_words_sent = 0;
    ringlpn_2pc::SecureConvertCounters conversion;
    uint64_t protocol_bytes_sent = 0;
    uint64_t protocol_direction_switches = 0;
    uint64_t protocol_dependency_rounds = 0;
    uint64_t transport_straight_bytes_sent = 0;
    uint64_t transport_straight_bytes_received = 0;
    uint64_t transport_reversed_bytes_sent = 0;
    uint64_t transport_reversed_bytes_received = 0;
    uint64_t base_ots = 0;
    uint64_t base_ot_setup_bytes_sent = 0;
    uint64_t base_ot_setup_bytes_received = 0;
    bool transport_available = false;
    bool transport_received_available = false;
    bool emp_metrics_available = false;
    ringlpn_2pc::EmpSilentMetrics emp_metrics{};
    double preflight_us = std::numeric_limits<double>::quiet_NaN();
    double ot_setup_us = std::numeric_limits<double>::quiet_NaN();
    double dpf_phase_a_us = std::numeric_limits<double>::quiet_NaN();
    double dpf_phase_b_us = std::numeric_limits<double>::quiet_NaN();
    double dpf_phase_c_us = std::numeric_limits<double>::quiet_NaN();
    double spfss_grouping_us = std::numeric_limits<double>::quiet_NaN();
    double public_polynomial_exchange_us =
        std::numeric_limits<double>::quiet_NaN();
    double gpu_ringlpn_expansion_us =
        std::numeric_limits<double>::quiet_NaN();
    double derandomization_openings_us =
        std::numeric_limits<double>::quiet_NaN();
    double conversion_us = std::numeric_limits<double>::quiet_NaN();
    double serialization_us = std::numeric_limits<double>::quiet_NaN();
    double commit_us = std::numeric_limits<double>::quiet_NaN();
    double total_us = 0.0;
};

struct GpuMemorySampler {
    static_assert(sizeof(size_t) <= sizeof(uint64_t),
                  "CUDA memory counters must fit publication CSV fields");
    bool available = true;
    bool sampled = false;
    uint64_t peak_bytes = 0;
    uint64_t min_free_bytes = std::numeric_limits<uint64_t>::max();

    void sample() {
        if (!available) return;
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
            free_bytes > total_bytes) {
            available = false;
            sampled = false;
            return;
        }
        const uint64_t free_u64 = static_cast<uint64_t>(free_bytes);
        const uint64_t used_u64 =
            static_cast<uint64_t>(total_bytes - free_bytes);
        peak_bytes = std::max(peak_bytes, used_u64);
        min_free_bytes = std::min(min_free_bytes, free_u64);
        sampled = true;
    }
};

void usage(const char *program) {
#ifdef RINGLPN_LIVE_CONV
    std::fprintf(
        stderr,
        "Usage:\n"
        "  %s --party 0|1 --sid N --invocation-id 32hex --ledger ABS "
        "--out-prefix P [--host H] [--port N] "
        "[--qbits 64|128] [--bw N] [--n N] [--h H] [--w W] [--ci CI] "
        "[--fh FH] [--fw FW] [--co CO] [--padding P] [--stride S] "
        "[--ole-n N] [--ole-c N] [--ole-t N] [--noise uniform|regular] "
        "[--channel local-loopback|external-loopback-tunnel] "
        "[--ot-backend sci-iknp|emp-silent] [--emp-silent-bridge SO] "
        "[--csv-header] [--force-rename-failure]\n"
        "  %s --check (--out-prefix P | --p0-record F --p1-record F) "
        "[--csv-header]\n",
        program, program);
#else
    std::fprintf(
        stderr,
        "Usage:\n"
        "  %s --party 0|1 --sid N --invocation-id 32hex --ledger ABS "
        "--out-prefix P [--host H] [--port N] "
        "[--qbits 64|128] [--bw N] [--rows M] [--inner K] [--cols N] "
        "[--ole-n N] [--ole-c N] [--ole-t N] [--noise uniform|regular] "
        "[--channel local-loopback|external-loopback-tunnel] "
        "[--ot-backend sci-iknp|emp-silent] [--emp-silent-bridge SO] "
        "[--csv-header] [--force-rename-failure]\n"
        "  %s --check (--out-prefix P | --p0-record F --p1-record F) "
        "[--csv-header]\n",
        program, program);
#endif
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
        } else if (key == "--invocation-id" && (value = next())) {
            args.invocation_id_text = value;
        } else if (key == "--ledger" && (value = next())) {
            args.ledger_path = value;
        } else if (key == "--qbits" && (value = next())) {
            if (!parse_int(value, args.qbits)) return false;
        } else if (key == "--bw" && (value = next())) {
            if (!parse_int(value, args.bw)) return false;
#ifdef RINGLPN_LIVE_CONV
        } else if (key == "--n" && (value = next())) {
            if (!parse_int(value, args.rows)) return false;
        } else if (key == "--h" && (value = next())) {
            if (!parse_int(value, args.inner)) return false;
        } else if (key == "--w" && (value = next())) {
            if (!parse_int(value, args.cols)) return false;
        } else if (key == "--ci" && (value = next())) {
            if (!parse_int(value, args.ci)) return false;
        } else if (key == "--fh" && (value = next())) {
            if (!parse_int(value, args.fh)) return false;
        } else if (key == "--fw" && (value = next())) {
            if (!parse_int(value, args.fw)) return false;
        } else if (key == "--co" && (value = next())) {
            if (!parse_int(value, args.co)) return false;
        } else if (key == "--padding" && (value = next())) {
            if (!parse_int(value, args.padding)) return false;
        } else if (key == "--stride" && (value = next())) {
            if (!parse_int(value, args.stride)) return false;
#endif
#ifndef RINGLPN_LIVE_CONV
        } else if (key == "--rows" && (value = next())) {
            if (!parse_int(value, args.rows)) return false;
        } else if (key == "--inner" && (value = next())) {
            if (!parse_int(value, args.inner)) return false;
        } else if (key == "--cols" && (value = next())) {
            if (!parse_int(value, args.cols)) return false;
#endif
        } else if (key == "--ole-n" && (value = next())) {
            if (!parse_int(value, args.ole_n)) return false;
        } else if (key == "--ole-c" && (value = next())) {
            if (!parse_int(value, args.ole_c)) return false;
        } else if (key == "--ole-t" && (value = next())) {
            if (!parse_int(value, args.ole_t)) return false;
        } else if (key == "--ot-backend" && (value = next())) {
            args.ot_backend = value;
        } else if (key == "--emp-silent-bridge" && (value = next())) {
            args.emp_silent_bridge = value;
        } else if (key == "--noise" && (value = next())) args.noise = value;
        else if (key == "--channel" && (value = next())) args.channel = value;
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
    ringlpn_2pc::OtBackend parsed_backend;
    return !args.out_prefix.empty() &&
           (args.party == 0 || args.party == 1) && args.sid != 0 &&
           ringlpn_freshness::parse_invocation_id(args.invocation_id_text,
                                                  args.invocation_id) &&
           !args.ledger_path.empty() && args.ledger_path.front() == '/' &&
           ringlpn_2pc::parse_ot_backend(args.ot_backend, parsed_backend) &&
           (parsed_backend == ringlpn_2pc::OtBackend::SciIknp
                ? args.emp_silent_bridge.empty()
                : !args.emp_silent_bridge.empty()) &&
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
        (args.noise != "uniform" && args.noise != "regular") ||
        (args.channel != "local-loopback" &&
         args.channel != "external-loopback-tunnel") ||
        (args.ot_backend != "sci-iknp" && args.ot_backend != "emp-silent")) {
        return false;
    }
#ifdef RINGLPN_LIVE_CONV
    if (args.ci <= 0 || args.fh <= 0 || args.fw <= 0 || args.co <= 0 ||
        args.padding < 0 || args.stride <= 0) {
        return false;
    }
#endif
    work.regular = args.noise == "regular";
    work.limbs = args.qbits == 128 ? 2 : 1;

    uint64_t size_a = 0, size_b = 0, size_c = 0, cross = 0;
#ifdef RINGLPN_LIVE_CONV
    const int64_t padded_h = static_cast<int64_t>(args.inner) - args.fh +
                             2 * static_cast<int64_t>(args.padding);
    const int64_t padded_w = static_cast<int64_t>(args.cols) - args.fw +
                             2 * static_cast<int64_t>(args.padding);
    if (padded_h < 0 || padded_w < 0) return false;
    const int64_t oh64 = padded_h / args.stride + 1;
    const int64_t ow64 = padded_w / args.stride + 1;
    if (oh64 <= 0 || ow64 <= 0 || oh64 > INT_MAX || ow64 > INT_MAX) {
        return false;
    }
    const int oh = static_cast<int>(oh64);
    const int ow = static_cast<int>(ow64);
    uint64_t tmp = 0;
    if (!checked_mul(static_cast<uint64_t>(args.rows), args.inner, tmp) ||
        !checked_mul(tmp, static_cast<uint64_t>(args.cols), tmp) ||
        !checked_mul(tmp, static_cast<uint64_t>(args.ci), size_a) ||
        !checked_mul(static_cast<uint64_t>(args.co), args.fh, tmp) ||
        !checked_mul(tmp, static_cast<uint64_t>(args.fw), tmp) ||
        !checked_mul(tmp, static_cast<uint64_t>(args.ci), size_b) ||
        !checked_mul(static_cast<uint64_t>(args.rows), oh, tmp) ||
        !checked_mul(tmp, static_cast<uint64_t>(ow), tmp) ||
        !checked_mul(tmp, static_cast<uint64_t>(args.co), size_c)) {
        return false;
    }
    const uint64_t max_payload_words =
        (kMaxRecordBytes - kRecordHeaderBytes - kDigestBytes) / sizeof(T);
    if (size_a > static_cast<uint64_t>(INT_MAX) ||
        size_b > static_cast<uint64_t>(INT_MAX) ||
        size_c > static_cast<uint64_t>(INT_MAX) ||
        size_a + size_b + size_c > max_payload_words) {
        return false;
    }
    GPUConv2DKey<T> conv{};
    conv.p = {args.bw, args.bw, args.rows, args.inner, args.cols, args.ci,
              args.fh, args.fw, args.co, args.padding, args.padding,
              args.padding, args.padding, args.stride, args.stride, oh, ow};
    conv.p.size_I = static_cast<size_t>(size_a);
    conv.p.size_F = static_cast<size_t>(size_b);
    conv.p.size_O = static_cast<size_t>(size_c);
    conv.mem_size_I = conv.p.size_I * sizeof(T);
    conv.mem_size_F = conv.p.size_F * sizeof(T);
    conv.mem_size_O = conv.p.size_O * sizeof(T);
    work.conv = conv;

    uint64_t spatial_positions = 0;
    if (!checked_mul(static_cast<uint64_t>(oh), static_cast<uint64_t>(ow),
                     spatial_positions) ||
        spatial_positions > kMaxRecordBytes / sizeof(uint64_t) - 1) {
        return false;
    }
    work.conv_spatial_prefix.reserve(static_cast<size_t>(spatial_positions) + 1);
    work.conv_spatial_prefix.push_back(0);
    uint64_t per_image = 0;
    for (int out_h = 0; out_h < oh; ++out_h) {
        const int64_t input_h0 =
            static_cast<int64_t>(out_h) * args.stride - args.padding;
        const int64_t fh_begin = std::max<int64_t>(0, -input_h0);
        const int64_t fh_end =
            std::min<int64_t>(args.fh, static_cast<int64_t>(args.inner) - input_h0);
        for (int out_w = 0; out_w < ow; ++out_w) {
            const int64_t input_w0 =
                static_cast<int64_t>(out_w) * args.stride - args.padding;
            const int64_t fw_begin = std::max<int64_t>(0, -input_w0);
            const int64_t fw_end =
                std::min<int64_t>(args.fw, static_cast<int64_t>(args.cols) - input_w0);
            const uint64_t valid_h = fh_end > fh_begin
                                         ? static_cast<uint64_t>(fh_end - fh_begin)
                                         : 0;
            const uint64_t valid_w = fw_end > fw_begin
                                         ? static_cast<uint64_t>(fw_end - fw_begin)
                                         : 0;
            uint64_t terms_here = valid_h;
            if (!checked_mul(terms_here, valid_w,
                             terms_here) ||
                !checked_mul(terms_here, static_cast<uint64_t>(args.ci),
                             terms_here) ||
                !checked_mul(terms_here, static_cast<uint64_t>(args.co),
                             terms_here) ||
                per_image > std::numeric_limits<uint64_t>::max() - terms_here) {
                return false;
            }
            per_image += terms_here;
            work.conv_spatial_prefix.push_back(per_image);
        }
    }
    work.conv_terms_per_image = per_image;
    if (per_image == 0 ||
        !checked_mul(per_image, static_cast<uint64_t>(args.rows), cross)) {
        return false;
    }
#else
    if (!checked_mul(static_cast<uint64_t>(args.rows),
                     static_cast<uint64_t>(args.inner), size_a) ||
        !checked_mul(static_cast<uint64_t>(args.inner),
                     static_cast<uint64_t>(args.cols), size_b) ||
        !checked_mul(static_cast<uint64_t>(args.rows),
                     static_cast<uint64_t>(args.cols), size_c) ||
        !checked_mul(size_c, static_cast<uint64_t>(args.inner), cross)) {
        return false;
    }
#endif
    if (size_a > static_cast<uint64_t>(INT_MAX) ||
        size_b > static_cast<uint64_t>(INT_MAX) ||
        size_c > static_cast<uint64_t>(INT_MAX) || cross == 0) {
        return false;
    }
    work.size_a = size_a;
    work.size_b = size_b;
    work.size_c = size_c;
    work.cross_terms = cross;
    uint64_t ct = 0;
    uint64_t trees = 0;
    if (!checked_mul(static_cast<uint64_t>(args.ole_c),
                     static_cast<uint64_t>(args.ole_t), ct) ||
        !checked_mul(ct, ct, trees) ||
        !checked_mul(trees, 3, work.ring_bootstrap_slots) ||
        work.ring_bootstrap_slots >= static_cast<uint64_t>(args.ole_n)) {
        return false;
    }
    work.ring_application_slots =
        static_cast<uint64_t>(args.ole_n) - work.ring_bootstrap_slots;
    work.ring_batches =
        1 + (cross - 1) / work.ring_application_slots;
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
#ifdef RINGLPN_LIVE_CONV
    const U128 max_inner =
        static_cast<U128>(args.fh) * args.fw * args.ci;
#else
    const U128 max_inner = static_cast<U128>(args.inner);
#endif
    if (max_inner >= (modulus >> (2 * args.bw + 2))) return false;

#ifndef RINGLPN_LIVE_CONV
    MatmulParams p;
    p.batchSz = 1;
    p.M = args.rows;
    p.K = args.inner;
    p.N = args.cols;
    stdInit(p, args.bw, 0);
    if (p.size_A != static_cast<int>(size_a) ||
        p.size_B != static_cast<int>(size_b) ||
        p.size_C != static_cast<int>(size_c)) {
        return false;
    }
    work.matmul = p;
#endif
    return true;
}
bool public_a_validation_gate() {
    ringlpn_ole_party::RingOlePublicParams params;
    params.n = kMinDegree;
    params.c = 2;
    params.t = 1;
    params.log_domain = ringlpn_ole_party::log2_exact(2 * params.n);
    params.modulus = ringlpn_orca::kPrime62;
    std::vector<Word> public_a(
        static_cast<size_t>(params.c) * static_cast<size_t>(params.n), 0);
    public_a[0] = 1;
    public_a[static_cast<size_t>(params.n)] = params.modulus - 1;
    if (!ringlpn_ole_party::validate_public_polynomials(params, public_a)) {
        return false;
    }
    public_a[0] = 0;
    if (ringlpn_ole_party::validate_public_polynomials(params, public_a)) {
        return false;
    }
    public_a[0] = 1;
    public_a[1] = 1;
    if (ringlpn_ole_party::validate_public_polynomials(params, public_a)) {
        return false;
    }
    public_a[1] = 0;
    public_a[static_cast<size_t>(params.n)] = params.modulus;
    if (ringlpn_ole_party::validate_public_polynomials(params, public_a)) {
        return false;
    }
    public_a[static_cast<size_t>(params.n)] = 0;
    public_a.pop_back();
    return !ringlpn_ole_party::validate_public_polynomials(params, public_a);
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
const char *backend_revision(ringlpn_2pc::OtBackend backend) {
    return backend == ringlpn_2pc::OtBackend::EmpSilent
               ? RINGLPN_EMP_SILENT_REVISION
               : "SCI-IKNP-IN-TREE";
}

#ifdef RINGLPN_LIVE_CONV
constexpr size_t kRecordBackendOffset = 176;
#else
constexpr size_t kRecordBackendOffset = 128;
#endif


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
#ifdef RINGLPN_LIVE_CONV
    put_u32(out, 76, static_cast<uint32_t>(h.ci));
    put_u32(out, 80, static_cast<uint32_t>(h.fh));
    put_u32(out, 84, static_cast<uint32_t>(h.fw));
    put_u32(out, 88, static_cast<uint32_t>(h.co));
    put_u32(out, 92, static_cast<uint32_t>(h.padding));
    put_u32(out, 96, static_cast<uint32_t>(h.stride));
    std::copy(h.invocation_id.begin(), h.invocation_id.end(), out.begin() + 128);
    std::copy(h.ledger_digest.begin(), h.ledger_digest.end(), out.begin() + 144);
#else
    std::copy(h.invocation_id.begin(), h.invocation_id.end(), out.begin() + 80);
    std::copy(h.ledger_digest.begin(), h.ledger_digest.end(), out.begin() + 96);
#endif
    put_u32(out, kRecordBackendOffset,
            h.ot_backend == ringlpn_2pc::OtBackend::EmpSilent ? 1U : 0U);
    const char *revision = backend_revision(h.ot_backend);
    std::memcpy(out.data() + kRecordBackendOffset + 4, revision,
                std::strlen(revision));
    return out;
}

bool decode_header(const uint8_t *in, size_t size, RecordHeader &h) {
    if (size < kRecordHeaderBytes ||
        !std::equal(kRecordMagic.begin(), kRecordMagic.end(), in) ||
        get_u32(in, 8) != kRecordVersion) {
        return false;
    }
#ifdef RINGLPN_LIVE_CONV
    if (!std::all_of(in + 100, in + 128,
                     [](uint8_t byte) { return byte == 0; })) {
        return false;
    }
#else
    if (get_u32(in, 76) != 0) return false;
#endif
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
#ifdef RINGLPN_LIVE_CONV
    h.ci = static_cast<int>(get_u32(in, 76));
    h.fh = static_cast<int>(get_u32(in, 80));
    h.fw = static_cast<int>(get_u32(in, 84));
    h.co = static_cast<int>(get_u32(in, 88));
    h.padding = static_cast<int>(get_u32(in, 92));
    h.stride = static_cast<int>(get_u32(in, 96));
    std::copy(in + 128, in + 144, h.invocation_id.begin());
    std::copy(in + 144, in + 176, h.ledger_digest.begin());
#else
    std::copy(in + 80, in + 96, h.invocation_id.begin());
    std::copy(in + 96, in + 128, h.ledger_digest.begin());
#endif
    const uint32_t backend = get_u32(in, kRecordBackendOffset);
    if (backend > 1) return false;
    h.ot_backend = backend == 1 ? ringlpn_2pc::OtBackend::EmpSilent
                                : ringlpn_2pc::OtBackend::SciIknp;
    const char *revision = backend_revision(h.ot_backend);
    const size_t revision_size = std::strlen(revision);
    if (!std::equal(revision, revision + revision_size,
                    in + kRecordBackendOffset + 4) ||
        !std::all_of(in + kRecordBackendOffset + 4 + revision_size,
                     in + kRecordBackendOffset + 48,
                     [](uint8_t byte) { return byte == 0; })) {
        return false;
    }
    const bool invocation_nonzero =
        std::any_of(h.invocation_id.begin(), h.invocation_id.end(),
                    [](uint8_t byte) { return byte != 0; });
    const bool ledger_nonzero =
        std::any_of(h.ledger_digest.begin(), h.ledger_digest.end(),
                    [](uint8_t byte) { return byte != 0; });
    return (h.party == 0 || h.party == 1) && h.sid != 0 &&
           invocation_nonzero && ledger_nonzero &&
           (h.qbits == 64 || h.qbits == 128) && h.bw > 2 && h.bw <= 32 &&
           h.rows > 0 && h.inner > 0 && h.cols > 0 && h.ole_n > 0 &&
           h.ole_c > 0 && h.ole_t > 0 && regular <= 1
#ifdef RINGLPN_LIVE_CONV
           && h.ci > 0 && h.fh > 0 && h.fw > 0 && h.co > 0 &&
           h.padding >= 0 && h.stride > 0
#endif
        ;
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
#ifdef RINGLPN_LIVE_CONV
    return prefix + "_p" + std::to_string(party) + ".conv";
#else
    return prefix + "_p" + std::to_string(party) + ".fc";
#endif
}

std::array<uint8_t, kPreflightBytes> encode_preflight(
    const Args &args, const ringlpn_freshness::Digest &ledger_digest) {
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
    const ringlpn_2pc::OtBackend backend =
        args.ot_backend == "emp-silent" ? ringlpn_2pc::OtBackend::EmpSilent
                                         : ringlpn_2pc::OtBackend::SciIknp;
    put32(kLayerManifestBytes,
          backend == ringlpn_2pc::OtBackend::EmpSilent ? 1U : 0U);
    const char *revision = backend_revision(backend);
    std::memcpy(out.data() + kLayerManifestBytes + 4, revision,
                std::strlen(revision));
    put32(52, args.noise == "regular" ? 1U : 0U);
#ifdef RINGLPN_LIVE_CONV
    put32(56, static_cast<uint32_t>(args.ci));
    put32(60, static_cast<uint32_t>(args.fh));
    put32(64, static_cast<uint32_t>(args.fw));
    put32(68, static_cast<uint32_t>(args.co));
    put32(72, static_cast<uint32_t>(args.padding));
    put32(76, static_cast<uint32_t>(args.stride));
    put32(80, args.channel == "external-loopback-tunnel" ? 1U : 0U);
    std::copy(args.invocation_id.begin(), args.invocation_id.end(),
              out.begin() + 96);
    std::copy(ledger_digest.begin(), ledger_digest.end(), out.begin() + 112);
#else
    put32(56, args.channel == "external-loopback-tunnel" ? 1U : 0U);
    std::copy(args.invocation_id.begin(), args.invocation_id.end(),
              out.begin() + 64);
    std::copy(ledger_digest.begin(), ledger_digest.end(), out.begin() + 80);
#endif
    return out;
}

bool agree_preflight(ringlpn_2pc::PartyChannel &channel, const Args &args,
                     const ringlpn_freshness::Claim &claim,
                     bool local_valid) {
    const auto manifest = encode_preflight(args, claim.ledger_digest);
    std::array<uint8_t, kPreflightBytes + 1> mine{};
    std::array<uint8_t, kPreflightBytes + 1> peer{};
    std::copy(manifest.begin(), manifest.end(), mine.begin());
    mine.back() = local_valid ? 1 : 0;
    channel.exchange_bytes(mine.data(), peer.data(), mine.size());
    return mine.back() == 1 && peer.back() == 1 &&
           std::equal(mine.begin(), mine.end() - 1, peer.begin());
}

bool derive_scope_id(const Args &args,
                     const ringlpn_freshness::Digest &layer_identity,
                     const ringlpn_freshness::Coordinates &coordinates,
                     ringlpn_freshness::Digest &id, uint64_t &handle) {
    if (!ringlpn_freshness::derive_correlation_id(
            args.invocation_id, layer_identity, coordinates, id)) {
        return false;
    }
    handle = ringlpn_freshness::compatibility_handle(id);
    return true;
}

struct OtInventory {
    uint64_t straight = 0;
    uint64_t reversed = 0;
};

bool compute_correlation_plan(
    const Args &args, const PublicWork &work,
    ringlpn_freshness::Digest &layer_identity,
    ringlpn_freshness::Digest &plan_digest, OtInventory &inventory) {
    inventory = OtInventory{};
    auto add = [](uint64_t &target, uint64_t value) {
        if (target > std::numeric_limits<uint64_t>::max() - value) return false;
        target += value;
        return true;
    };
    auto mul = [](uint64_t a, uint64_t b, uint64_t &out) {
        if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
        out = a * b;
        return true;
    };

    // Canonical layer identity is deployment/backend/invocation independent.
    std::vector<uint8_t> layer;
    static constexpr uint8_t layer_domain[] =
        "RINGLPN-FC-CONV-LAYER-IDENTITY-v1";
    layer.insert(layer.end(), layer_domain, layer_domain + sizeof(layer_domain) - 1);
    ringlpn_freshness::put_u32_be(layer, ringlpn_freshness::kProtocolVersion);
#ifdef RINGLPN_LIVE_CONV
    ringlpn_freshness::put_u32_be(layer, 2);
#else
    ringlpn_freshness::put_u32_be(layer, 1);
#endif
    ringlpn_freshness::put_u64_be(layer, 0);  // stable layer ordinal
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.qbits));
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.bw));
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.rows));
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.inner));
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.cols));
#ifdef RINGLPN_LIVE_CONV
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.ci));
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.fh));
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.fw));
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.co));
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.padding));
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.stride));
#endif
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.ole_n));
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.ole_c));
    ringlpn_freshness::put_u64_be(layer, static_cast<uint64_t>(args.ole_t));
    ringlpn_freshness::put_u64_be(layer, work.regular ? 1 : 0);
    if (!ringlpn_freshness::digest(layer.data(), layer.size(), layer_identity))
        return false;

    std::vector<uint8_t> plan;
    static constexpr uint8_t domain[] = "RINGLPN-FC-CONV-CORRELATION-PLAN-v3";
    plan.insert(plan.end(), domain, domain + sizeof(domain) - 1);
    ringlpn_freshness::put_u32_be(plan, ringlpn_freshness::kProtocolVersion);
    plan.insert(plan.end(), layer_identity.begin(), layer_identity.end());
    ringlpn_freshness::put_u64_be(plan, work.ring_batches);
    ringlpn_freshness::put_u64_be(plan, static_cast<uint64_t>(work.limbs));
    ringlpn_freshness::put_u64_be(plan, work.cross_terms);
    ringlpn_freshness::put_u64_be(plan, work.ring_application_slots);
    ringlpn_freshness::put_u64_be(plan, work.ring_bootstrap_slots);
    ringlpn_freshness::put_u64_be(plan, work.size_a);
    ringlpn_freshness::put_u64_be(plan, work.size_b);
    ringlpn_freshness::put_u64_be(plan, work.size_c);

    uint64_t ct = 0, trees = 0;
    if (!mul(static_cast<uint64_t>(args.ole_c),
             static_cast<uint64_t>(args.ole_t), ct) ||
        !mul(ct, ct, trees)) {
        return false;
    }
    const int log_domain = work.regular
                               ? ringlpn_ole_party::log2_exact(
                                     2 * (args.ole_n / args.ole_t))
                               : ringlpn_ole_party::log2_exact(2 * args.ole_n);
    if (log_domain < 2) return false;
    uint64_t phase_a = 0, phase_b_one_direction = 0;
    uint64_t phase_b_total = 0, phase_c_scalar = 0, phase_c_ots = 0;
    uint64_t public_shares = 0;
    if (!mul(trees, static_cast<uint64_t>(log_domain - 1), phase_a) ||
        !mul(trees, static_cast<uint64_t>(log_domain),
             phase_b_one_direction) ||
        !mul(phase_b_one_direction, 2, phase_b_total) ||
        !mul(trees, 3, phase_c_scalar) ||
        !mul(phase_c_scalar, 62, phase_c_ots) ||
        !mul(static_cast<uint64_t>(args.ole_c - 1),
             static_cast<uint64_t>(args.ole_n), public_shares)) {
        return false;
    }

    std::set<uint64_t> handles;
    for (uint64_t batch = 0; batch < work.ring_batches; ++batch) {
        const uint64_t start = batch * work.ring_application_slots;
        const uint64_t used_slots =
            std::min(work.ring_application_slots, work.cross_terms - start);
        for (int direction = 0; direction < 2; ++direction) {
            for (int limb = 0; limb < work.limbs; ++limb) {
                ringlpn_freshness::Coordinates coordinates;
                coordinates.kind = ringlpn_freshness::Kind::kRingOle;
                coordinates.direction = static_cast<uint64_t>(direction);
                coordinates.crt_limb = static_cast<uint64_t>(limb);
                coordinates.ring_batch = batch;
                coordinates.phase = ringlpn_freshness::Phase::kRingExpansion;
                coordinates.primitive_ordinal = 0;
                ringlpn_freshness::Digest id{};
                uint64_t handle = 0;
                if (!derive_scope_id(args, layer_identity, coordinates, id,
                                     handle) ||
                    !handles.insert(handle).second) {
                    return false;
                }
                coordinates.kind =
                    ringlpn_freshness::Kind::kPublicPolynomialShare;
                coordinates.phase =
                    ringlpn_freshness::Phase::kPublicPolynomial;
                if (!derive_scope_id(args, layer_identity, coordinates, id,
                                     handle) ||
                    !handles.insert(handle).second) {
                    return false;
                }
                ringlpn_freshness::put_u64_be(plan, batch);
                ringlpn_freshness::put_u64_be(
                    plan, static_cast<uint64_t>(direction));
                ringlpn_freshness::put_u64_be(
                    plan, static_cast<uint64_t>(limb));
                ringlpn_freshness::put_u64_be(plan, trees);
                ringlpn_freshness::put_u64_be(plan, phase_a);
                ringlpn_freshness::put_u64_be(plan, phase_b_total);
                ringlpn_freshness::put_u64_be(plan, phase_c_scalar);
                ringlpn_freshness::put_u64_be(plan, public_shares);
                ringlpn_freshness::put_u64_be(plan, used_slots);
                const bool epoch_zero = batch == 0 && direction == 0;
                ringlpn_freshness::put_u64_be(plan, epoch_zero ? 1 : 0);
                if (!add(inventory.straight, phase_a) ||
                    !add(inventory.straight, phase_b_one_direction) ||
                    (epoch_zero &&
                     !add(inventory.straight, phase_c_ots)) ||
                    !add(inventory.reversed, phase_a) ||
                    !add(inventory.reversed, phase_b_one_direction)) {
                    return false;
                }
            }
        }
    }

    U128 conversion_modulus =
        args.qbits == 64
            ? static_cast<U128>(ringlpn_orca::kPrime62)
            : static_cast<U128>(ringlpn_orca::kPrime62) *
                  static_cast<U128>(ringlpn_orca::kPrime62Crt2);
    U128 ell_value = 2 * conversion_modulus - 1;
    uint64_t ell = 0;
    while (ell_value != 0) {
        ++ell;
        ell_value >>= 1;
    }
    uint64_t cursor = 0;
    uint64_t chunk = 0;
    while (cursor < work.size_c) {
        const uint64_t count = std::min<uint64_t>(
            ringlpn_2pc::kMaxSecureConvertBatch, work.size_c - cursor);
        ringlpn_freshness::Coordinates coordinates;
        coordinates.kind = ringlpn_freshness::Kind::kConversionEdabit;
        coordinates.phase = ringlpn_freshness::Phase::kConvertCorrelation;
        coordinates.primitive_ordinal = 0;
        coordinates.conversion_chunk = chunk;
        ringlpn_freshness::Digest id{};
        uint64_t handle = 0;
        if (!derive_scope_id(args, layer_identity, coordinates, id, handle) ||
            !handles.insert(handle).second) {
            return false;
        }
        uint64_t edabit_components = 0, triples = 0, strings = 0;
        if (!mul(count, ell, edabit_components) ||
            !mul(count, 2 * ell - 2, triples) ||
            !add(strings, edabit_components) || !add(strings, count)) {
            return false;
        }
        ringlpn_freshness::put_u64_be(plan, chunk);
        ringlpn_freshness::put_u64_be(plan, count);
        ringlpn_freshness::put_u64_be(plan, ell);
        ringlpn_freshness::put_u64_be(plan, edabit_components);
        ringlpn_freshness::put_u64_be(plan, count);  // wrap daBits
        ringlpn_freshness::put_u64_be(plan, triples);
        ringlpn_freshness::put_u64_be(plan, count);  // output slots
        ringlpn_freshness::put_u64_be(plan, count);  // output masks
        if (!add(inventory.straight, strings) ||
            !add(inventory.straight, triples) ||
            !add(inventory.reversed, triples)) {
            return false;
        }
        cursor += count;
        ++chunk;
    }
    const ringlpn_2pc::OtBackend backend =
        args.ot_backend == "emp-silent" ? ringlpn_2pc::OtBackend::EmpSilent
                                         : ringlpn_2pc::OtBackend::SciIknp;
    ringlpn_freshness::put_u64_be(
        plan, backend == ringlpn_2pc::OtBackend::EmpSilent ? 1 : 0);
    const char *revision = backend_revision(backend);
    plan.insert(plan.end(), revision, revision + std::strlen(revision));
    ringlpn_freshness::put_u64_be(plan, inventory.straight);
    ringlpn_freshness::put_u64_be(plan, inventory.reversed);
    return inventory.straight != 0 && inventory.reversed != 0 &&
           ringlpn_freshness::digest(plan.data(), plan.size(), plan_digest);
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
#ifdef RINGLPN_LIVE_CONV
bool conv_term_at(const PublicWork &work, uint64_t global, ConvTerm &term) {
    const Conv2DParams &p = work.conv.p;
    if (global >= work.cross_terms || work.conv_terms_per_image == 0 ||
        work.conv_spatial_prefix.size() !=
            static_cast<size_t>(p.OH) * p.OW + 1) {
        return false;
    }
    const uint64_t image = global / work.conv_terms_per_image;
    const uint64_t within_image = global % work.conv_terms_per_image;
    const auto it = std::upper_bound(work.conv_spatial_prefix.begin(),
                                     work.conv_spatial_prefix.end(),
                                     within_image);
    if (it == work.conv_spatial_prefix.begin() ||
        it == work.conv_spatial_prefix.end()) {
        return false;
    }
    const size_t spatial =
        static_cast<size_t>(it - work.conv_spatial_prefix.begin() - 1);
    const uint64_t within_spatial =
        within_image - work.conv_spatial_prefix[spatial];
    const int out_h = static_cast<int>(spatial / static_cast<size_t>(p.OW));
    const int out_w = static_cast<int>(spatial % static_cast<size_t>(p.OW));
    const int64_t input_h0 =
        static_cast<int64_t>(out_h) * p.strideH - p.zPadHLeft;
    const int64_t input_w0 =
        static_cast<int64_t>(out_w) * p.strideW - p.zPadWLeft;
    const int fh_begin = static_cast<int>(std::max<int64_t>(0, -input_h0));
    const int fh_end = static_cast<int>(
        std::min<int64_t>(p.FH, static_cast<int64_t>(p.H) - input_h0));
    const int fw_begin = static_cast<int>(std::max<int64_t>(0, -input_w0));
    const int fw_end = static_cast<int>(
        std::min<int64_t>(p.FW, static_cast<int64_t>(p.W) - input_w0));
    if (fh_begin >= fh_end || fw_begin >= fw_end) return false;
    const uint64_t valid_w = static_cast<uint64_t>(fw_end - fw_begin);
    const uint64_t per_filter =
        static_cast<uint64_t>(fh_end - fh_begin) * valid_w * p.CI;
    if (per_filter == 0) return false;
    const uint64_t co = within_spatial / per_filter;
    const uint64_t filter_offset = within_spatial % per_filter;
    if (co >= static_cast<uint64_t>(p.CO)) return false;
    const uint64_t filter_position = filter_offset / p.CI;
    const int ci = static_cast<int>(filter_offset % p.CI);
    const int fh = fh_begin + static_cast<int>(filter_position / valid_w);
    const int fw = fw_begin + static_cast<int>(filter_position % valid_w);
    const int ih = static_cast<int>(input_h0 + fh);
    const int iw = static_cast<int>(input_w0 + fw);
    term.output =
        ((static_cast<size_t>(image) * p.OH + out_h) * p.OW + out_w) * p.CO +
        static_cast<size_t>(co);
    term.input =
        ((static_cast<size_t>(image) * p.H + ih) * p.W + iw) * p.CI + ci;
    term.filter =
        ((static_cast<size_t>(co) * p.FH + fh) * p.FW + fw) * p.CI + ci;
    return term.output < work.size_c && term.input < work.size_a &&
           term.filter < work.size_b;
}
#endif

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

double elapsed_us(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start)
        .count();
}

bool add_us(double &target, double value) {
    if (!std::isfinite(value) || value < 0.0) {
        target = std::numeric_limits<double>::quiet_NaN();
        return false;
    }
    if (std::isnan(target)) {
        target = value;
        return true;
    }
    if (!std::isfinite(target) ||
        value > std::numeric_limits<double>::max() - target) {
        target = std::numeric_limits<double>::quiet_NaN();
        return false;
    }
    target += value;
    return true;
}

class RingOleBootstrapPool final
    : public ringlpn_2pdpf::PhaseCOleSource {
  public:
    RingOleBootstrapPool(int party, Word modulus)
        : party_(party), modulus_(modulus) {}

    size_t available() const {
        return x_.size() >= cursor_ ? x_.size() - cursor_ : 0;
    }

    bool refill(const ringlpn_ole_party::RingOlePartyShares &shares,
                uint64_t application_slots, uint64_t bootstrap_slots) {
        if (available() != 0 || application_slots > shares.X_slots.size() ||
            bootstrap_slots >
                shares.X_slots.size() - static_cast<size_t>(application_slots) ||
            shares.X_slots.size() != shares.Z_slots.size() ||
            application_slots + bootstrap_slots != shares.X_slots.size()) {
            return false;
        }
        const size_t begin = static_cast<size_t>(application_slots);
        x_.assign(shares.X_slots.begin() + static_cast<ptrdiff_t>(begin),
                  shares.X_slots.end());
        z_.assign(shares.Z_slots.begin() + static_cast<ptrdiff_t>(begin),
                  shares.Z_slots.end());
        cursor_ = 0;
        return x_.size() == static_cast<size_t>(bootstrap_slots);
    }

    uint64_t discard_remaining() {
        const uint64_t discarded = static_cast<uint64_t>(available());
        x_.clear();
        z_.clear();
        cursor_ = 0;
        return discarded;
    }

    bool multiply(ringlpn_2pc::PartyChannel &channel,
                  const std::vector<Word> &local_inputs, Word modulus,
                  std::vector<Word> &product_shares) override {
        product_shares.clear();
        const size_t count = local_inputs.size();
        const uint64_t bit_count =
            static_cast<uint64_t>(ringlpn_2pc::field_bits(modulus));
        if ((party_ != 0 && party_ != 1) || channel.party() != party_ ||
            modulus != modulus_ || count == 0 || count > available() ||
            count > static_cast<size_t>(INT_MAX) / sizeof(Word) ||
            count > std::numeric_limits<uint64_t>::max() / (2 * bit_count)) {
            return false;
        }
        const uint64_t count64 = static_cast<uint64_t>(count);
        const uint64_t logical_bits = 2 * bit_count * count64;
        const uint64_t share_bits = bit_count * count64;
        const auto can_add = [](uint64_t target, uint64_t value) {
            return target <= std::numeric_limits<uint64_t>::max() - value;
        };
        if (!can_add(channel.costs.scalar_oles, count64) ||
            !can_add(channel.costs.phase_c.logical_bits, logical_bits) ||
            !can_add(channel.costs.phase_c.revealed_bits_sent, share_bits) ||
            !can_add(channel.costs.phase_c.revealed_bits_recv, share_bits)) {
            return false;
        }
        std::vector<Word> mine(count);
        std::vector<Word> peer(count);
        for (size_t i = 0; i < count; ++i) {
            if (local_inputs[i] >= modulus) return false;
            mine[i] =
                ringlpn_2pc::mod_sub(local_inputs[i], x_[cursor_ + i], modulus);
        }
        channel.exchange_bytes(reinterpret_cast<const uint8_t *>(mine.data()),
                               reinterpret_cast<uint8_t *>(peer.data()),
                               count * sizeof(Word));
        product_shares.resize(count);
        for (size_t i = 0; i < count; ++i) {
            if (peer[i] >= modulus) return false;
            Word share = ringlpn_2pc::mod_add(
                z_[cursor_ + i],
                mod_mul_host<Word>(peer[i], x_[cursor_ + i], modulus),
                modulus);
            if (party_ == 0) {
                share = ringlpn_2pc::mod_add(
                    share, mod_mul_host<Word>(mine[i], peer[i], modulus),
                    modulus);
            }
            product_shares[i] = share;
        }
        cursor_ += count;
        channel.costs.scalar_oles += count64;
        channel.costs.phase_c.logical_bits += logical_bits;
        channel.costs.phase_c.revealed_bits_sent += share_bits;
        channel.costs.phase_c.revealed_bits_recv += share_bits;
        return true;
    }

  private:
    int party_;
    Word modulus_;
    std::vector<Word> x_;
    std::vector<Word> z_;
    size_t cursor_ = 0;
};

bool peak_host_rss_bytes(uint64_t &bytes) {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
        return false;
    }
    const uint64_t kib = static_cast<uint64_t>(usage.ru_maxrss);
    if (kib > std::numeric_limits<uint64_t>::max() / 1024ULL) {
        return false;
    }
    bytes = kib * 1024ULL;
    return true;
}

void print_metric(double value) {
    if (std::isfinite(value) && value >= 0.0) {
        std::cout << value;
    } else {
        std::cout << "NA";
    }
}

void print_metric(bool available, uint64_t value) {
    if (available) {
        std::cout << value;
    } else {
        std::cout << "NA";
    }
}

bool generate_ring_ole(
    const Args &args, const PublicWork &work,
    const ringlpn_freshness::Digest &layer_identity, uint64_t ring_batch,
    int direction, int limb, ringlpn_2pc::PartyChannel &channel,
    ringlpn_2pc::PartyRandom &random, AESGlobalContext *gaes,
    RingOleBootstrapPool &bootstrap_pool,
    ringlpn_ole_party::RingOlePartyShares &shares, Counters &counters) {
    const Word modulus = modulus_for_limb(limb);
    const bool epoch_zero = ring_batch == 0 && direction == 0;
    if ((epoch_zero && bootstrap_pool.available() != 0) ||
        (!epoch_zero &&
         bootstrap_pool.available() != work.ring_bootstrap_slots)) {
        return false;
    }
    ringlpn_2pdpf::PhaseCOleSource *phase_c_source =
        epoch_zero ? nullptr : &bootstrap_pool;
    const int log_domain = work.regular
                               ? ringlpn_ole_party::log2_exact(2 * (args.ole_n / args.ole_t))
                               : ringlpn_ole_party::log2_exact(2 * args.ole_n);
    ringlpn_freshness::Coordinates scope_coordinates;
    scope_coordinates.kind = ringlpn_freshness::Kind::kRingOle;
    scope_coordinates.direction = static_cast<uint64_t>(direction);
    scope_coordinates.crt_limb = static_cast<uint64_t>(limb);
    scope_coordinates.ring_batch = ring_batch;
    scope_coordinates.phase = ringlpn_freshness::Phase::kRingExpansion;
    scope_coordinates.primitive_ordinal = 0;
    ringlpn_freshness::Digest scope_id{};
    uint64_t spfss_sid = 0;
    if (!derive_scope_id(args, layer_identity, scope_coordinates, scope_id,
                         spfss_sid)) {
        return false;
    }
    ringlpn_spfss::SpfssPublicParams spfss_params{
        args.ole_c, args.ole_t, log_domain, modulus, work.regular, spfss_sid};
    spfss_params.correlation_id = scope_id;
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
    if (!add_u64(counters.protocol_dependency_rounds, 1)) return false;
    ringlpn_spfss::GroupedHostKeys grouped;
    ringlpn_spfss::DpfCounters dpf;
    if (!local_valid ||
        !ringlpn_spfss::generate_party_spfss_keys_gpu_batched(
            args.party, spfss_params, batch, channel, random, gaes, grouped,
            dpf, phase_c_source) ||
        !add_us(counters.dpf_phase_a_us, dpf.phase_a_microseconds) ||
        !add_us(counters.dpf_phase_b_us, dpf.phase_b_microseconds) ||
        !add_us(counters.dpf_phase_c_us, dpf.phase_c_microseconds) ||
        !add_us(counters.spfss_grouping_us,
                dpf.spfss_grouping_microseconds) ||
        !add_u64(counters.protocol_dependency_rounds,
                 dpf.phase_a_dependency_rounds) ||
        !add_u64(counters.protocol_dependency_rounds,
                 dpf.phase_b_dependency_rounds) ||
        !add_u64(counters.protocol_dependency_rounds,
                 dpf.phase_c_dependency_rounds)) {
        return false;
    }
    std::vector<uint64_t> binding;
    if (!ringlpn_spfss::party_noise_binding(noise, args.party, spfss_params,
                                             binding)) {
        return false;
    }

    // The module-Ring-LPN public vector is a=(1,a_1,...,a_{c-1}).
    // The identity polynomial is fixed and unsent. For every coefficient of
    // i>=1, each party sends one full uniform field-element share; their sum is
    // exactly uniform in Z_p even conditioned on either contribution. A short
    // revealed PRG seed would instead restrict `a` to the seed image.
    const auto public_polynomial_start = Clock::now();
    if (args.ole_c < 1 || args.ole_n < 1 ||
        static_cast<size_t>(args.ole_c) >
            std::numeric_limits<size_t>::max() /
                static_cast<size_t>(args.ole_n) ||
        static_cast<size_t>(args.ole_c - 1) >
            std::numeric_limits<size_t>::max() /
                static_cast<size_t>(args.ole_n)) {
        return false;
    }
    const size_t public_coefficients =
        static_cast<size_t>(args.ole_c) * static_cast<size_t>(args.ole_n);
    const size_t public_random_coefficients =
        static_cast<size_t>(args.ole_c - 1) *
        static_cast<size_t>(args.ole_n);
    if (public_coefficients >
            std::numeric_limits<size_t>::max() / sizeof(Word) ||
        public_random_coefficients >
            std::numeric_limits<size_t>::max() / sizeof(Word)) {
        return false;
    }
    std::vector<Word> public_a(public_coefficients, 0);
    public_a[0] = 1;
    std::vector<uint8_t> coin_mine(public_random_coefficients * sizeof(Word));
    std::vector<uint8_t> coin_peer(coin_mine.size());
    for (size_t coefficient = 0; coefficient < public_random_coefficients;
         ++coefficient) {
        const Word share = random.field(modulus);
        public_a[static_cast<size_t>(args.ole_n) + coefficient] = share;
        for (size_t byte = 0; byte < sizeof(Word); ++byte) {
            coin_mine[coefficient * sizeof(Word) + byte] =
                static_cast<uint8_t>(share >> (8 * byte));
        }
    }
    channel.exchange_bytes(coin_mine.data(), coin_peer.data(), coin_mine.size());
    if (!add_u64(counters.public_a_words_sent, public_random_coefficients)) {
        return false;
    }
    for (size_t coefficient = 0; coefficient < public_random_coefficients;
         ++coefficient) {
        Word peer_coefficient = 0;
        for (size_t byte = 0; byte < sizeof(Word); ++byte) {
            peer_coefficient |= static_cast<Word>(
                                    coin_peer[coefficient * sizeof(Word) + byte])
                                << (8 * byte);
        }
        if (peer_coefficient >= modulus) return false;
        const size_t public_index =
            static_cast<size_t>(args.ole_n) + coefficient;
        public_a[public_index] =
            mod_add<Word>(public_a[public_index], peer_coefficient, modulus);
    }
    if (!add_us(counters.public_polynomial_exchange_us,
                elapsed_us(public_polynomial_start)) ||
        !add_u64(counters.protocol_dependency_rounds, 1)) {
        return false;
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
    ringlpn_freshness::Coordinates public_coordinates = scope_coordinates;
    public_coordinates.kind =
        ringlpn_freshness::Kind::kPublicPolynomialShare;
    public_coordinates.phase =
        ringlpn_freshness::Phase::kPublicPolynomial;
    ringlpn_freshness::Digest public_id{};
    uint64_t public_handle = 0;
    if (!derive_scope_id(args, layer_identity, public_coordinates, public_id,
                         public_handle)) {
        return false;
    }
    ole_params.public_a_seed = public_handle;
    ole_params.regular = work.regular;
    if (!ringlpn_ole_party::validate_public_polynomials(ole_params, public_a)) {
        return false;
    }

    ringlpn_ole_party::RingOlePartyKeys keys;
    if (!ringlpn_ole_party::pack_gpu_party_keys(
            ole_params, args.party, noise, binding, grouped, keys)) {
        return false;
    }
    const auto gpu_expansion_start = Clock::now();
    ringlpn_ole_party::RingOlePartyCounters ole_counters;
    const bool expanded = ringlpn_ole_party::expand_ring_ole_party(
        ole_params, args.party, noise, std::move(keys), gaes, shares,
        ole_counters, &public_a);
    if (!add_us(counters.gpu_ringlpn_expansion_us,
                elapsed_us(gpu_expansion_start)) ||
        !expanded ||
        !add_u64(counters.ring_ole_instances, 1)) {
        return false;
    }
    if (dpf.scalar_oles != work.ring_bootstrap_slots ||
        bootstrap_pool.available() != 0 ||
        shares.X_slots.size() != static_cast<size_t>(args.ole_n) ||
        shares.Z_slots.size() != static_cast<size_t>(args.ole_n) ||
        !bootstrap_pool.refill(shares, work.ring_application_slots,
                               work.ring_bootstrap_slots) ||
        !add_u64(counters.dpf_trees, ole_counters.trees) ||
        !add_u64(counters.key_bytes, ole_counters.key_bytes) ||
        !add_u64(counters.dpf_string_ots, dpf.string_ots_128) ||
        !add_u64(counters.dpf_bit_triples, dpf.bit_triples) ||
        !add_u64(counters.dpf_scalar_oles, dpf.scalar_oles) ||
        !add_u64(epoch_zero ? counters.dpf_epoch_zero_scalar_oles
                            : counters.dpf_pcg_scalar_oles,
                 dpf.scalar_oles) ||
        (!epoch_zero &&
         !add_u64(counters.dpf_pcg_opening_words_sent, dpf.scalar_oles)) ||
        !add_u64(counters.dpf_pcg_oles_reserved,
                 work.ring_bootstrap_slots) ||
        !add_u64(counters.dpf_logical_opened_bits, dpf.logical_opened_bits) ||
        !add_u64(counters.dpf_meaningful_share_bits,
                 dpf.meaningful_share_bits)) {
        return false;
    }
    return true;
}

bool exchange_openings(const Args &args, const PublicWork &work,
                       uint64_t ring_batch, int direction, int limb,
                       const std::vector<T> &a_share,
                       const std::vector<T> &b_share,
                       const ringlpn_ole_party::RingOlePartyShares &ole,
                       ringlpn_2pc::PartyChannel &channel,
                       std::vector<std::vector<Word>> &limb_acc,
                       Counters &counters) {
    const auto openings_start = Clock::now();
    const uint64_t start = ring_batch * work.ring_application_slots;
    const uint64_t count =
        std::min(work.ring_application_slots, work.cross_terms - start);
#ifdef RINGLPN_LIVE_CONV
    std::vector<ConvTerm> batch_terms(static_cast<size_t>(count));
    for (uint64_t local = 0; local < count; ++local) {
        if (!conv_term_at(work, start + local,
                          batch_terms[static_cast<size_t>(local)])) {
            return false;
        }
    }
#endif
    const Word modulus = modulus_for_limb(limb);
    std::vector<uint8_t> mine(static_cast<size_t>(count) * sizeof(Word));
    std::vector<uint8_t> peer(mine.size());
    for (uint64_t local = 0; local < count; ++local) {
#ifdef RINGLPN_LIVE_CONV
        const ConvTerm &term = batch_terms[static_cast<size_t>(local)];
        const size_t a_idx = term.input;
        const size_t b_idx = term.filter;
#else
        const uint64_t global = start + local;
        const uint64_t output = global / static_cast<uint64_t>(args.inner);
        const int k = static_cast<int>(global % static_cast<uint64_t>(args.inner));
        const int row = static_cast<int>(output / static_cast<uint64_t>(args.cols));
        const int col = static_cast<int>(output % static_cast<uint64_t>(args.cols));
        const size_t a_idx = matrix_index(work.matmul, true, row, k);
        const size_t b_idx = matrix_index(work.matmul, false, k, col);
#endif
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
#ifdef RINGLPN_LIVE_CONV
        const size_t output =
            batch_terms[static_cast<size_t>(local)].output;
#else
        const size_t output = static_cast<size_t>((start + local) /
                                                   static_cast<uint64_t>(args.inner));
#endif
        limb_acc[static_cast<size_t>(limb)][output] = mod_add<Word>(
            limb_acc[static_cast<size_t>(limb)][output], cross, modulus);
    }
    if (!add_u64(counters.slots_used, count) ||
        !add_u64(counters.ring_application_slots_discarded,
                 work.ring_application_slots - count) ||
        !add_u64(counters.protocol_dependency_rounds, 1) ||
        !add_us(counters.derandomization_openings_us,
                elapsed_us(openings_start))) {
        return false;
    }
    return true;
}

bool accumulate_local_products(const Args &args, const PublicWork &work,
                               const std::vector<T> &a_share,
                               const std::vector<T> &b_share,
                               std::vector<std::vector<Word>> &limb_acc) {
#ifdef RINGLPN_LIVE_CONV
    for (uint64_t global = 0; global < work.cross_terms; ++global) {
        ConvTerm term;
        if (!conv_term_at(work, global, term)) return false;
        const uint64_t a = static_cast<uint64_t>(a_share[term.input]);
        const uint64_t b = static_cast<uint64_t>(b_share[term.filter]);
        for (int limb = 0; limb < work.limbs; ++limb) {
            const Word modulus = modulus_for_limb(limb);
            limb_acc[static_cast<size_t>(limb)][term.output] = mod_add<Word>(
                limb_acc[static_cast<size_t>(limb)][term.output],
                mod_mul_host<Word>(a % modulus, b % modulus, modulus), modulus);
        }
    }
#else
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
#endif
    return true;
}

bool convert_outputs(
    const Args &args, const PublicWork &work,
    const ringlpn_freshness::Digest &layer_identity,
    const std::vector<std::vector<Word>> &limb_acc,
    ringlpn_2pc::PartyChannel &channel, ringlpn_2pc::PartyRandom &random,
    std::vector<T> &converted, Counters &counters) {
    const auto conversion_start = Clock::now();
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
        ringlpn_freshness::Coordinates coordinates;
        coordinates.kind = ringlpn_freshness::Kind::kConversionEdabit;
        coordinates.phase =
            ringlpn_freshness::Phase::kConvertCorrelation;
        coordinates.primitive_ordinal = 0;
        coordinates.conversion_chunk = chunk_index;
        uint64_t handle = 0;
        if (!derive_scope_id(args, layer_identity, coordinates,
                             params.correlation_id, handle)) {
            return false;
        }
        params.sid = handle;
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
        if (one.conversions == 0 ||
            one.post_mask_dependencies % one.conversions != 0) {
            return false;
        }
        uint64_t chunk_dependency_rounds = 5;
        const uint64_t post_mask_rounds =
            one.post_mask_dependencies / one.conversions;
        if (!add_us(counters.conversion.correlation_microseconds,
                    one.correlation_microseconds) ||
            !add_us(counters.conversion.online_microseconds,
                    one.online_microseconds) ||
            !add_u64(chunk_dependency_rounds, post_mask_rounds) ||
            !add_u64(counters.protocol_dependency_rounds,
                     chunk_dependency_rounds)) {
            return false;
        }
        cursor += count;
        if (!add_u64(chunk_index, 1)) return false;
    }
    return add_us(counters.conversion_us, elapsed_us(conversion_start));
}

bool publish_record(
    const Args &args, const PublicWork &work,
    const ringlpn_freshness::Claim &claim, const std::vector<T> &a_share,
    const std::vector<T> &b_share, const std::vector<T> &c_share,
    ringlpn_2pc::PartyChannel &channel, Counters &counters) {
    const auto serialization_start = Clock::now();
    std::vector<T> payload;
    payload.reserve(a_share.size() + b_share.size() + c_share.size());
    payload.insert(payload.end(), a_share.begin(), a_share.end());
    payload.insert(payload.end(), b_share.begin(), b_share.end());
    payload.insert(payload.end(), c_share.begin(), c_share.end());
    RecordHeader header;
    header.party = args.party;
    header.sid = args.sid;
    header.invocation_id = args.invocation_id;
    header.ledger_digest = claim.ledger_digest;
    header.ot_backend =
        args.ot_backend == "emp-silent" ? ringlpn_2pc::OtBackend::EmpSilent
                                         : ringlpn_2pc::OtBackend::SciIknp;
    header.qbits = args.qbits;
    header.bw = args.bw;
    header.rows = args.rows;
    header.inner = args.inner;
    header.cols = args.cols;
#ifdef RINGLPN_LIVE_CONV
    header.ci = args.ci;
    header.fh = args.fh;
    header.fw = args.fw;
    header.co = args.co;
    header.padding = args.padding;
    header.stride = args.stride;
#endif
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
    staged = staged &&
             add_us(counters.serialization_us,
                    elapsed_us(serialization_start));
    const auto commit_start = Clock::now();
    uint8_t mine = staged ? 1 : 0;
    uint8_t peer = 0;
    channel.exchange_bytes(&mine, &peer, 1);
    if (!add_u64(counters.protocol_dependency_rounds, 1)) return false;
    if (!staged || peer != 1) {
        std::remove(temporary.c_str());
        add_us(counters.commit_us, elapsed_us(commit_start));
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
    const bool commit_accounted =
        add_u64(counters.protocol_dependency_rounds, 1) &&
        add_us(counters.commit_us, elapsed_us(commit_start));
    if (!renamed || peer != 1 || !commit_accounted) {
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

bool collect_transport_metrics(ringlpn_2pc::PartyChannel &channel,
                               Counters &counters) {
    const uint64_t straight_sent = channel.straight_bytes_sent();
    const uint64_t reversed_sent = channel.reversed_bytes_sent();
    const uint64_t setup_straight_sent =
        channel.setup_straight_bytes_sent();
    const uint64_t setup_reversed_sent =
        channel.setup_reversed_bytes_sent();
    uint64_t setup_sent = 0;
    if (setup_straight_sent > straight_sent ||
        setup_reversed_sent > reversed_sent ||
        !add_u64(setup_sent, setup_straight_sent) ||
        !add_u64(setup_sent, setup_reversed_sent)) {
        return false;
    }
    // A process can read only its local NetIO send counters without adding a
    // metrics message to the protocol. CSV aggregators fill each receive field
    // from the peer's matching sent field after both processes exit.
    counters.transport_straight_bytes_sent = straight_sent;
    counters.transport_reversed_bytes_sent = reversed_sent;
    if (channel.ot_backend() == ringlpn_2pc::OtBackend::SciIknp) {
        counters.base_ot_setup_bytes_sent = setup_sent;
        counters.base_ots = channel.costs.base_ots;
    }
    counters.transport_available = true;
    return true;
}

void print_party_header() {
    std::cout
#ifdef RINGLPN_LIVE_CONV
        << "party,qbits,bw,n,h,w,ci,fh,fw,co,padding,stride,oh,ow,"
           "ole_n,ole_c,ole_t,noise,ring_batches,ring_application_slots,"
           "ring_bootstrap_slots,"
#else
        << "party,qbits,bw,rows,inner,cols,ole_n,ole_c,ole_t,noise,ring_batches,"
           "ring_application_slots,ring_bootstrap_slots,"
#endif
        << "ring_ole_instances,slots_used,dpf_trees,dpf_string_ots,dpf_bit_triples,"
        << "dpf_scalar_oles,dpf_epoch_zero_scalar_oles,dpf_pcg_scalar_oles,"
        << "dpf_pcg_oles_reserved,dpf_pcg_oles_discarded,"
        << "dpf_pcg_opening_words_sent,dpf_logical_opened_bits,"
        << "dpf_meaningful_share_bits,spfss_key_bytes,public_a_words_sent,"
        << "derandomization_words_sent,"
        << "conversions,conversion_logical_opened_bits,"
        << "conversion_meaningful_share_bits,protocol_bytes_sent,"
        << "protocol_direction_switches,total_us,status,"
        << "protocol_dependency_rounds,preflight_us,ot_setup_us,"
        << "dpf_phase_a_us,dpf_phase_b_us,dpf_phase_c_us,spfss_grouping_us,"
        << "public_polynomial_exchange_us,gpu_ringlpn_expansion_us,"
        << "derandomization_openings_us,conversion_us,serialization_us,commit_us,"
        << "peak_host_rss_bytes,peak_gpu_bytes,min_gpu_free_bytes,"
        << "transport_straight_bytes_sent,transport_straight_bytes_received,"
        << "transport_reversed_bytes_sent,transport_reversed_bytes_received,"
        << "base_ots,base_ot_setup_bytes_sent,base_ot_setup_bytes_received,"
        << "transport_bytes_include_base_ot,base_ot_setup_dependency_rounds,"
        << "invocation_id,ledger_digest,ot_backend,ot_backend_revision,"
        << "ot_correlation_straight_bytes_sent,"
        << "ot_correlation_straight_bytes_received,"
        << "ot_correlation_reversed_bytes_sent,"
        << "ot_correlation_reversed_bytes_received,"
        << "ot_adjustment_bytes_sent,ot_adjustment_bytes_received,"
        << "ot_ciphertext_bytes_sent,ot_ciphertext_bytes_received,"
        << "ot_inventory_straight_declared,ot_inventory_straight_consumed,"
        << "ot_inventory_reversed_declared,ot_inventory_reversed_consumed,"
        << "ot_backend_review_status,ring_application_slots_discarded\n";
}

int run_party(const Args &args) {
    PublicWork work;
    const bool work_valid = derive_work(args, work);
    const std::string output = record_path(args.out_prefix, args.party);
    const std::string temporary = output + ".tmp";
    std::error_code ec;
    const bool output_absent = !std::filesystem::exists(output, ec) &&
                               !std::filesystem::exists(temporary, ec);
    ringlpn_freshness::Digest layer_identity{};
    ringlpn_freshness::Digest plan_digest{};
    OtInventory ot_inventory;
    ringlpn_freshness::Claim claim;
    const auto ledger = std::filesystem::path(args.ledger_path).lexically_normal();
    const auto output_absolute =
        std::filesystem::absolute(std::filesystem::path(output), ec)
            .lexically_normal();
    const auto ledger_is_prefix = std::mismatch(
        ledger.begin(), ledger.end(), output_absolute.begin(),
        output_absolute.end());
    const auto output_is_prefix = std::mismatch(
        output_absolute.begin(), output_absolute.end(), ledger.begin(),
        ledger.end());
    const bool paths_disjoint =
        !ec && ledger_is_prefix.first != ledger.end() &&
        output_is_prefix.first != output_absolute.end();
    const bool plan_ok =
        work_valid && output_absent && paths_disjoint &&
        compute_correlation_plan(args, work, layer_identity, plan_digest,
                                 ot_inventory);
    const bool claim_ok =
        plan_ok && ringlpn_freshness::claim_namespace_once(
                       args.ledger_path, args.party, args.invocation_id,
                       layer_identity, plan_digest, claim);
    const bool local_valid = plan_ok && claim_ok;
    if (!local_valid) {
        std::fprintf(stderr,
                     "[two-party-fc] local preflight failed: work=%d "
                     "output-absent=%d paths-disjoint=%d plan=%d claim=%d\n",
                     work_valid ? 1 : 0, output_absent ? 1 : 0,
                     paths_disjoint ? 1 : 0, plan_ok ? 1 : 0,
                     claim_ok ? 1 : 0);
    }

    Counters counters;
    ringlpn_2pc::OtBackend ot_backend;
    if (!ringlpn_2pc::parse_ot_backend(args.ot_backend, ot_backend))
        return 2;
    ringlpn_2pc::EmpSilentPlan emp_plan;
    emp_plan.bridge_library = args.emp_silent_bridge;
    emp_plan.public_manifest_digest = plan_digest;
    emp_plan.straight_count = ot_inventory.straight;
    emp_plan.reversed_count = ot_inventory.reversed;
    emp_plan.threads = 1;
    ringlpn_2pc::PartyChannel channel(
        args.party, args.host, args.port, /*defer_ot_setup=*/true,
        /*require_loopback_endpoints=*/true, ot_backend,
        ot_backend == ringlpn_2pc::OtBackend::EmpSilent ? &emp_plan : nullptr);
    const auto preflight_start = Clock::now();
    const bool preflight_ok = agree_preflight(channel, args, claim, local_valid);
    if (!add_us(counters.preflight_us, elapsed_us(preflight_start)) ||
        !add_u64(counters.protocol_dependency_rounds, 1) ||
        !preflight_ok) {
        std::remove(temporary.c_str());
        std::fprintf(stderr,
                     "[two-party-fc] public/local preflight rejected before OT/output\n");
        return 2;
    }
    const auto ot_setup_start = Clock::now();
    channel.setup_ots();
    if (!add_us(counters.ot_setup_us, elapsed_us(ot_setup_start))) return 1;
    const uint64_t protocol_begin_bytes = channel.bytes_sent();
    const uint64_t protocol_begin_switches = channel.direction_switches();
    const auto started = Clock::now();

    initGPUMemPool();
    GpuMemorySampler gpu_memory;
    gpu_memory.sample();
    AESGlobalContext gaes;
    initAESContext(&gaes);
    ringlpn_2pc::PartyRandom random;
    std::vector<T> a_share = sample_ring_words(work.size_a, args.bw, random);
    std::vector<T> b_share = sample_ring_words(work.size_b, args.bw, random);
    std::vector<T> y_share = sample_ring_words(work.size_c, args.bw, random);
    std::vector<std::vector<Word>> limb_acc(
        static_cast<size_t>(work.limbs), std::vector<Word>(work.size_c, 0));
    std::vector<RingOleBootstrapPool> bootstrap_pools;
    bootstrap_pools.reserve(static_cast<size_t>(work.limbs));
    for (int limb = 0; limb < work.limbs; ++limb) {
        bootstrap_pools.emplace_back(args.party, modulus_for_limb(limb));
    }
    bool ok =
        accumulate_local_products(args, work, a_share, b_share, limb_acc);
    for (uint64_t ring_batch = 0;
         ok && ring_batch < work.ring_batches; ++ring_batch) {
        for (int direction = 0; ok && direction < 2; ++direction) {
            for (int limb = 0; ok && limb < work.limbs; ++limb) {
                ringlpn_ole_party::RingOlePartyShares shares;
                const bool generated = generate_ring_ole(
                    args, work, layer_identity, ring_batch, direction, limb,
                    channel, random, &gaes,
                    bootstrap_pools[static_cast<size_t>(limb)], shares,
                    counters);
                gpu_memory.sample();
                uint8_t mine = generated ? 1 : 0;
                uint8_t peer = 0;
                channel.exchange_bytes(&mine, &peer, 1);
                ok = add_u64(counters.protocol_dependency_rounds, 1) &&
                     mine == 1 && peer == 1;
                if (ok) {
                    ok = exchange_openings(
                        args, work, ring_batch, direction, limb, a_share,
                        b_share, shares, channel, limb_acc, counters);
                }
            }
        }
    }
    if (ok) {
        for (RingOleBootstrapPool &pool : bootstrap_pools) {
            const uint64_t discarded = pool.discard_remaining();
            if (discarded != work.ring_bootstrap_slots ||
                !add_u64(counters.dpf_pcg_oles_discarded, discarded)) {
                ok = false;
                break;
            }
        }
    }
    uint64_t expected_ring_ole_instances = 0;
    uint64_t expected_public_a_words = 0;
    uint64_t expected_scalar_oles = 0;
    uint64_t expected_epoch_zero_oles = 0;
    uint64_t expected_pcg_oles = 0;
    uint64_t expected_slots_used = 0;
    uint64_t expected_application_capacity = 0;
    uint64_t expected_application_discarded = 0;
    if (!checked_mul(work.ring_batches, static_cast<uint64_t>(2 * work.limbs),
                     expected_ring_ole_instances) ||
        !checked_mul(expected_ring_ole_instances,
                     static_cast<uint64_t>(args.ole_c - 1),
                     expected_public_a_words) ||
        !checked_mul(expected_public_a_words,
                     static_cast<uint64_t>(args.ole_n),
                     expected_public_a_words) ||
        !checked_mul(expected_ring_ole_instances, work.ring_bootstrap_slots,
                     expected_scalar_oles) ||
        !checked_mul(static_cast<uint64_t>(work.limbs),
                     work.ring_bootstrap_slots, expected_epoch_zero_oles) ||
        expected_scalar_oles < expected_epoch_zero_oles ||
        !checked_mul(work.cross_terms,
                     static_cast<uint64_t>(2 * work.limbs),
                     expected_slots_used) ||
        !checked_mul(expected_ring_ole_instances,
                     work.ring_application_slots,
                     expected_application_capacity) ||
        expected_application_capacity < expected_slots_used) {
        ok = false;
    } else {
        expected_pcg_oles = expected_scalar_oles - expected_epoch_zero_oles;
        expected_application_discarded =
            expected_application_capacity - expected_slots_used;
    }
    if (counters.ring_ole_instances != expected_ring_ole_instances ||
        counters.public_a_words_sent != expected_public_a_words ||
        counters.slots_used != expected_slots_used ||
        counters.dpf_scalar_oles != expected_scalar_oles ||
        counters.dpf_epoch_zero_scalar_oles != expected_epoch_zero_oles ||
        counters.dpf_pcg_scalar_oles != expected_pcg_oles ||
        counters.dpf_pcg_oles_reserved != expected_scalar_oles ||
        counters.dpf_pcg_oles_discarded != expected_epoch_zero_oles ||
        counters.dpf_pcg_opening_words_sent != expected_pcg_oles ||
        counters.ring_application_slots_discarded !=
            expected_application_discarded) {
        ok = false;
    }


    std::vector<T> converted;
    if (ok) {
        ok = convert_outputs(args, work, layer_identity, limb_acc, channel,
                             random, converted, counters);
    }
    {
        uint8_t mine = ok ? 1 : 0;
        uint8_t peer = 0;
        channel.exchange_bytes(&mine, &peer, 1);
        ok = add_u64(counters.protocol_dependency_rounds, 1) &&
             mine == 1 && peer == 1;
    }
    if (ok) {
        try {
            channel.finish_ots();
            if (ot_backend == ringlpn_2pc::OtBackend::EmpSilent) {
                counters.emp_metrics = channel.emp_silent_metrics();
                counters.emp_metrics_available = true;
                ok = counters.emp_metrics.straight.declared_count ==
                         ot_inventory.straight &&
                     counters.emp_metrics.straight.consumed_count ==
                         ot_inventory.straight &&
                     counters.emp_metrics.reversed.declared_count ==
                         ot_inventory.reversed &&
                     counters.emp_metrics.reversed.consumed_count ==
                         ot_inventory.reversed;
            }
        } catch (const std::exception &error) {
            std::fprintf(stderr, "[two-party-fc] OT exhaustion failed: %s\n",
                         error.what());
            ok = false;
        }
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
    gpu_memory.sample();

    if (ok) {
        ok = publish_record(args, work, claim, a_share, b_share, c_share,
                            channel, counters);
    }
    const uint64_t protocol_end_bytes = channel.bytes_sent();
    const uint64_t protocol_end_switches = channel.direction_switches();
    if (protocol_end_bytes < protocol_begin_bytes ||
        protocol_end_switches < protocol_begin_switches) {
        ok = false;
    } else {
        counters.protocol_bytes_sent =
            protocol_end_bytes - protocol_begin_bytes;
        counters.protocol_direction_switches =
            protocol_end_switches - protocol_begin_switches;
    }
    counters.total_us = elapsed_us(started);
    gpu_memory.sample();
    if (!collect_transport_metrics(channel, counters)) ok = false;
    uint64_t host_rss_bytes = 0;
    const bool host_rss_available = peak_host_rss_bytes(host_rss_bytes);

    if (args.csv_header) print_party_header();
#ifdef RINGLPN_LIVE_CONV
    std::cout << args.party << ',' << args.qbits << ',' << args.bw << ','
              << work.conv.p.N << ',' << work.conv.p.H << ',' << work.conv.p.W
              << ',' << work.conv.p.CI << ',' << work.conv.p.FH << ','
              << work.conv.p.FW << ',' << work.conv.p.CO << ','
              << work.conv.p.zPadHLeft << ',' << work.conv.p.strideH << ','
              << work.conv.p.OH << ',' << work.conv.p.OW << ',';
#else
    std::cout << args.party << ',' << args.qbits << ',' << args.bw << ','
              << args.rows << ',' << args.inner << ',' << args.cols << ',';
#endif
    std::cout << args.ole_n << ',' << args.ole_c << ',' << args.ole_t << ','
              << args.noise << ',' << work.ring_batches << ','
              << work.ring_application_slots << ','
              << work.ring_bootstrap_slots << ','
              << counters.ring_ole_instances << ',' << counters.slots_used << ','
              << counters.dpf_trees << ',' << counters.dpf_string_ots << ','
              << counters.dpf_bit_triples << ',' << counters.dpf_scalar_oles << ','
              << counters.dpf_epoch_zero_scalar_oles << ','
              << counters.dpf_pcg_scalar_oles << ','
              << counters.dpf_pcg_oles_reserved << ','
              << counters.dpf_pcg_oles_discarded << ','
              << counters.dpf_pcg_opening_words_sent << ','
              << counters.dpf_logical_opened_bits << ','
              << counters.dpf_meaningful_share_bits << ',' << counters.key_bytes
              << ',' << counters.public_a_words_sent << ','
              << counters.derandomization_words_sent << ','
              << counters.conversion.conversions << ','
              << counters.conversion.logical_opened_bits << ','
              << counters.conversion.meaningful_share_bits << ','
              << counters.protocol_bytes_sent << ','
              << counters.protocol_direction_switches << ',' << counters.total_us
              << ',' << (ok ? "pass" : "FAIL") << ','
              << counters.protocol_dependency_rounds << ',';
    print_metric(counters.preflight_us);
    std::cout << ',';
    print_metric(counters.ot_setup_us);
    std::cout << ',';
    print_metric(counters.dpf_phase_a_us);
    std::cout << ',';
    print_metric(counters.dpf_phase_b_us);
    std::cout << ',';
    print_metric(counters.dpf_phase_c_us);
    std::cout << ',';
    print_metric(counters.spfss_grouping_us);
    std::cout << ',';
    print_metric(counters.public_polynomial_exchange_us);
    std::cout << ',';
    print_metric(counters.gpu_ringlpn_expansion_us);
    std::cout << ',';
    print_metric(counters.derandomization_openings_us);
    std::cout << ',';
    print_metric(counters.conversion_us);
    std::cout << ',';
    print_metric(counters.serialization_us);
    std::cout << ',';
    print_metric(counters.commit_us);
    std::cout << ',';
    print_metric(host_rss_available, host_rss_bytes);
    std::cout << ',';
    print_metric(gpu_memory.available && gpu_memory.sampled,
                 gpu_memory.peak_bytes);
    std::cout << ',';
    print_metric(gpu_memory.available && gpu_memory.sampled,
                 gpu_memory.min_free_bytes);
    std::cout << ',';
    print_metric(counters.transport_available,
                 counters.transport_straight_bytes_sent);
    std::cout << ',';
    print_metric(counters.transport_received_available,
                 counters.transport_straight_bytes_received);
    std::cout << ',';
    print_metric(counters.transport_available,
                 counters.transport_reversed_bytes_sent);
    std::cout << ',';
    print_metric(counters.transport_received_available,
                 counters.transport_reversed_bytes_received);
    std::cout << ',';
    const bool sci_metrics =
        counters.transport_available &&
        ot_backend == ringlpn_2pc::OtBackend::SciIknp;
    print_metric(sci_metrics, counters.base_ots);
    std::cout << ',';
    print_metric(sci_metrics, counters.base_ot_setup_bytes_sent);
    std::cout << ",NA," << (sci_metrics ? "yes" : "NA")
              << ",NA," << ringlpn_freshness::hex(args.invocation_id) << ','
              << ringlpn_freshness::hex(claim.ledger_digest) << ','
              << ringlpn_2pc::ot_backend_name(ot_backend) << ','
              << backend_revision(ot_backend) << ',';
    if (counters.emp_metrics_available) {
        const auto &straight = counters.emp_metrics.straight;
        const auto &reversed = counters.emp_metrics.reversed;
        uint64_t adjustment_sent = 0;
        uint64_t ciphertext_sent = 0;
        const bool sums_ok =
            add_u64(adjustment_sent, straight.adjustment_bytes_sent) &&
            add_u64(adjustment_sent, reversed.adjustment_bytes_sent) &&
            add_u64(ciphertext_sent, straight.ciphertext_bytes_sent) &&
            add_u64(ciphertext_sent, reversed.ciphertext_bytes_sent);
        print_metric(true, straight.correlation_bytes_sent);
        std::cout << ",NA,";
        print_metric(true, reversed.correlation_bytes_sent);
        std::cout << ",NA,";
        print_metric(sums_ok, adjustment_sent);
        std::cout << ",NA,";
        print_metric(sums_ok, ciphertext_sent);
        std::cout << ",NA," << straight.declared_count << ','
                  << straight.consumed_count << ',' << reversed.declared_count
                  << ',' << reversed.consumed_count
                  << ",unreviewed-unmeasured,"
                  << counters.ring_application_slots_discarded << '\n';
    } else {
        std::cout << "NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,"
                  << (ot_backend == ringlpn_2pc::OtBackend::SciIknp
                          ? "existing-default"
                          : "NA")
                  << ',' << counters.ring_application_slots_discarded << '\n';
    }
    return ok ? 0 : 1;
}

bool headers_match(const RecordHeader &a, const RecordHeader &b) {
    return a.party == 0 && b.party == 1 && a.sid == b.sid &&
           a.invocation_id == b.invocation_id &&
           a.ledger_digest == b.ledger_digest &&
           a.ot_backend == b.ot_backend &&
           a.qbits == b.qbits && a.bw == b.bw && a.rows == b.rows &&
           a.inner == b.inner && a.cols == b.cols && a.ole_n == b.ole_n &&
           a.ole_c == b.ole_c && a.ole_t == b.ole_t &&
           a.regular == b.regular && a.ring_batches == b.ring_batches &&
           a.payload_words == b.payload_words
#ifdef RINGLPN_LIVE_CONV
           && a.ci == b.ci && a.fh == b.fh && a.fw == b.fw &&
           a.co == b.co && a.padding == b.padding && a.stride == b.stride
#endif
        ;
}

template <typename V>
void copy_to_gpu(const V &source, T **destination) {
    check(cudaMalloc(reinterpret_cast<void **>(destination),
                     source.size() * sizeof(T)), "two-party FC checker cudaMalloc");
    check(cudaMemcpy(*destination, source.data(), source.size() * sizeof(T),
                     cudaMemcpyHostToDevice), "two-party FC checker H2D");
}

int run_check(const Args &args) {
    double checker_us = 0.0;
    auto checker_segment_start = Clock::now();
    GpuMemorySampler gpu_memory;
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
        public_args.invocation_id = p0.header.invocation_id;
        public_args.sid = p0.header.sid;
        public_args.qbits = p0.header.qbits;
        public_args.bw = p0.header.bw;
        public_args.rows = p0.header.rows;
        public_args.inner = p0.header.inner;
        public_args.cols = p0.header.cols;
#ifdef RINGLPN_LIVE_CONV
        public_args.ci = p0.header.ci;
        public_args.fh = p0.header.fh;
        public_args.fw = p0.header.fw;
        public_args.co = p0.header.co;
        public_args.padding = p0.header.padding;
        public_args.stride = p0.header.stride;
#endif
        public_args.ole_n = p0.header.ole_n;
        public_args.ole_c = p0.header.ole_c;
        public_args.ole_t = p0.header.ole_t;
        public_args.noise = p0.header.regular ? "regular" : "uniform";
        public_args.ot_backend =
            ringlpn_2pc::ot_backend_name(p0.header.ot_backend);
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
#ifdef RINGLPN_LIVE_CONV
    std::vector<U128> mask_products(work.size_c, 0);
    for (uint64_t global = 0; global < work.cross_terms; ++global) {
        ConvTerm term;
        if (!conv_term_at(work, global, term)) return 1;
        mask_products[term.output] +=
            static_cast<U128>(mask_a[term.input]) * mask_b[term.filter];
    }
    for (size_t output = 0; output < output_mask.size(); ++output) {
        output_mask[output] = static_cast<T>(ringlpn_orca::ringSub(
            c_sum[output],
            ringlpn_orca::ringReduce(mask_products[output], public_args.bw),
            public_args.bw));
    }
#else
    for (int row = 0; row < public_args.rows; ++row) {
        for (int col = 0; col < public_args.cols; ++col) {
            U128 product = 0;
            for (int k = 0; k < public_args.inner; ++k) {
                product += static_cast<U128>(
                               mask_a[matrix_index(work.matmul, true, row, k)]) *
                           mask_b[matrix_index(work.matmul, false, k, col)];
            }
            const size_t output = static_cast<size_t>(row) * public_args.cols + col;
            output_mask[output] = static_cast<T>(ringlpn_orca::ringSub(
                c_sum[output], ringlpn_orca::ringReduce(product, public_args.bw),
                public_args.bw));
        }
    }
#endif

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
#ifdef RINGLPN_LIVE_CONV
    std::vector<U128> expected_products(work.size_c, 0);
    for (uint64_t global = 0; global < work.cross_terms; ++global) {
        ConvTerm term;
        if (!conv_term_at(work, global, term)) return 1;
        expected_products[term.output] +=
            static_cast<U128>(input[term.input]) * weight[term.filter];
    }
    for (size_t output = 0; output < expected.size(); ++output) {
        expected[output] = static_cast<T>(ringlpn_orca::ringAdd(
            ringlpn_orca::ringReduce(expected_products[output], public_args.bw),
            output_mask[output], public_args.bw));
    }
#else
    for (int row = 0; row < public_args.rows; ++row) {
        for (int col = 0; col < public_args.cols; ++col) {
            U128 product = 0;
            for (int k = 0; k < public_args.inner; ++k) {
                product += static_cast<U128>(
                               input[matrix_index(work.matmul, true, row, k)]) *
                           weight[matrix_index(work.matmul, false, k, col)];
            }
            const size_t output = static_cast<size_t>(row) * public_args.cols + col;
            expected[output] = static_cast<T>(ringlpn_orca::ringAdd(
                ringlpn_orca::ringReduce(product, public_args.bw),
                output_mask[output], public_args.bw));
        }
    }
#endif

    std::vector<uint8_t> key0(p0.payload.size() * sizeof(T));
    std::vector<uint8_t> key1(p1.payload.size() * sizeof(T));
    std::memcpy(key0.data(), p0.payload.data(), key0.size());
    std::memcpy(key1.data(), p1.payload.data(), key1.size());
    uint8_t *cursor0 = key0.data();
    uint8_t *cursor1 = key1.data();
#ifdef RINGLPN_LIVE_CONV
    GPUConv2DKey<T> gkey0 = work.conv;
    gkey0.I = reinterpret_cast<T *>(cursor0);
    cursor0 += gkey0.mem_size_I;
    gkey0.F = reinterpret_cast<T *>(cursor0);
    cursor0 += gkey0.mem_size_F;
    gkey0.O = reinterpret_cast<T *>(cursor0);
    cursor0 += gkey0.mem_size_O;
    GPUConv2DKey<T> gkey1 = work.conv;
    gkey1.I = reinterpret_cast<T *>(cursor1);
    cursor1 += gkey1.mem_size_I;
    gkey1.F = reinterpret_cast<T *>(cursor1);
    cursor1 += gkey1.mem_size_F;
    gkey1.O = reinterpret_cast<T *>(cursor1);
    cursor1 += gkey1.mem_size_O;
#else
    GPUMatmulKey<T> gkey0 = readGPUMatmulKey<T>(
        work.matmul, TruncateType::None, &cursor0);
    GPUMatmulKey<T> gkey1 = readGPUMatmulKey<T>(
        work.matmul, TruncateType::None, &cursor1);
#endif
    const bool key_order_ok = cursor0 == key0.data() + key0.size() &&
                              cursor1 == key1.data() + key1.size();

    initGPUMemPool();
    gpu_memory.sample();
    T *d_x = nullptr, *d_w = nullptr, *d_a0 = nullptr, *d_a1 = nullptr;
    T *d_b0 = nullptr, *d_b1 = nullptr;
    T *d_mask_a = nullptr, *d_mask_b = nullptr, *d_output_mask = nullptr;
    copy_to_gpu(masked_input, &d_x);
    copy_to_gpu(masked_weight, &d_w);
#ifdef RINGLPN_LIVE_CONV
    copy_to_gpu(std::vector<T>(gkey0.I, gkey0.I + work.conv.p.size_I), &d_a0);
    copy_to_gpu(std::vector<T>(gkey1.I, gkey1.I + work.conv.p.size_I), &d_a1);
    copy_to_gpu(std::vector<T>(gkey0.F, gkey0.F + work.conv.p.size_F), &d_b0);
    copy_to_gpu(std::vector<T>(gkey1.F, gkey1.F + work.conv.p.size_F), &d_b1);
#else
    copy_to_gpu(std::vector<T>(gkey0.A, gkey0.A + work.matmul.size_A), &d_a0);
    copy_to_gpu(std::vector<T>(gkey1.A, gkey1.A + work.matmul.size_A), &d_a1);
    copy_to_gpu(std::vector<T>(gkey0.B, gkey0.B + work.matmul.size_B), &d_b0);
    copy_to_gpu(std::vector<T>(gkey1.B, gkey1.B + work.matmul.size_B), &d_b1);
#endif
    copy_to_gpu(mask_a, &d_mask_a);
    copy_to_gpu(mask_b, &d_mask_b);
    copy_to_gpu(output_mask, &d_output_mask);

    std::vector<uint8_t> dealer_key0(key0.size(), 0);
    std::vector<uint8_t> dealer_key1(key1.size(), 0);
    uint8_t *dealer_cursor0 = dealer_key0.data();
    uint8_t *dealer_cursor1 = dealer_key1.data();
    check(cudaDeviceSynchronize(), "two-party FC checker dealer timer start");
    if (!add_us(checker_us, elapsed_us(checker_segment_start))) return 1;
    const auto dealer_start = Clock::now();
    initGPURandomness();
#ifdef RINGLPN_LIVE_CONV
    T *dealer_return0 = gpuKeygenConv2D<T>(
        &dealer_cursor0, SERVER0, work.conv, d_mask_a, mask_b.data(), true,
        d_output_mask);
#else
    T *dealer_return0 = gpuKeygenMatmul<T>(
        &dealer_cursor0, SERVER0, work.matmul, d_mask_a, d_mask_b, nullptr,
        TruncateType::None, nullptr, true, d_output_mask);
#endif
    destroyGPURandomness();
    initGPURandomness();
#ifdef RINGLPN_LIVE_CONV
    T *dealer_return1 = gpuKeygenConv2D<T>(
        &dealer_cursor1, SERVER1, work.conv, d_mask_a, mask_b.data(), true,
        d_output_mask);
#else
    T *dealer_return1 = gpuKeygenMatmul<T>(
        &dealer_cursor1, SERVER1, work.matmul, d_mask_a, d_mask_b, nullptr,
        TruncateType::None, nullptr, true, d_output_mask);
#endif
    destroyGPURandomness();
    check(cudaDeviceSynchronize(), "two-party FC checker dealer timer stop");
    const double dealer_keygen_us = elapsed_us(dealer_start);
    gpu_memory.sample();
    checker_segment_start = Clock::now();
    const bool dealer_keygen_ok =
        dealer_cursor0 == dealer_key0.data() + dealer_key0.size() &&
        dealer_cursor1 == dealer_key1.data() + dealer_key1.size() &&
        dealer_return0 == d_output_mask && dealer_return1 == d_output_mask;
    Stats stats0;
    Stats stats1;
    check(cudaDeviceSynchronize(), "two-party FC checker online timer start");
    if (!add_us(checker_us, elapsed_us(checker_segment_start))) return 1;
    const auto online_start = Clock::now();
#ifdef RINGLPN_LIVE_CONV
    T *d_o0 = gpuConv2DBeaver<T>(gkey0, SERVER0, d_x, d_w, d_a0, d_b0,
                                  nullptr, &stats0, 0);
    T *d_o1 = gpuConv2DBeaver<T>(gkey1, SERVER1, d_x, d_w, d_a1, d_b1,
                                  nullptr, &stats1, 0);
#else
    T *d_o0 = gpuMatmulBeaver<T>(work.matmul, gkey0, SERVER0, d_x, d_w,
                                  d_a0, d_b0, nullptr, &stats0);
    T *d_o1 = gpuMatmulBeaver<T>(work.matmul, gkey1, SERVER1, d_x, d_w,
                                  d_a1, d_b1, nullptr, &stats1);
#endif
    check(cudaDeviceSynchronize(), "two-party FC checker online timer stop");
    const double online_two_share_us = elapsed_us(online_start);
    gpu_memory.sample();
    checker_segment_start = Clock::now();
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
    gpu_memory.sample();
    if (!add_us(checker_us, elapsed_us(checker_segment_start))) return 1;
    uint64_t checker_host_rss_bytes = 0;
    const bool checker_host_rss_available =
        peak_host_rss_bytes(checker_host_rss_bytes);

    if (args.csv_header) {
#ifdef RINGLPN_LIVE_CONV
        std::cout
            << "qbits,bw,n,h,w,ci,fh,fw,co,padding,stride,oh,ow,ring_batches,"
            << "final_payload_bytes_per_party,matched_dealer_keygen_us,"
            << "checker_two_share_online_us,matched_dealer_keygen_contract,"
            << "key_order,unchanged_online_contract,status,checker_us,"
            << "peak_host_rss_bytes,peak_gpu_bytes,min_gpu_free_bytes,"
            << "invocation_id,ledger_digest\n";
#else
        std::cout
            << "qbits,bw,rows,inner,cols,ring_batches,final_payload_bytes_per_party,"
            << "matched_dealer_keygen_us,checker_two_share_online_us,"
            << "matched_dealer_keygen_contract,key_order,online_contract,status,"
            << "checker_us,peak_host_rss_bytes,peak_gpu_bytes,min_gpu_free_bytes,"
            << "invocation_id,ledger_digest\n";
#endif
    }
#ifdef RINGLPN_LIVE_CONV
    std::cout << public_args.qbits << ',' << public_args.bw << ','
              << work.conv.p.N << ',' << work.conv.p.H << ',' << work.conv.p.W
              << ',' << work.conv.p.CI << ',' << work.conv.p.FH << ','
              << work.conv.p.FW << ',' << work.conv.p.CO << ','
              << work.conv.p.zPadHLeft << ',' << work.conv.p.strideH << ','
              << work.conv.p.OH << ',' << work.conv.p.OW << ','
              << work.ring_batches << ',';
#else
    std::cout << public_args.qbits << ',' << public_args.bw << ','
              << public_args.rows << ',' << public_args.inner << ','
              << public_args.cols << ',' << work.ring_batches << ',';
#endif
    std::cout << key0.size() << ',' << dealer_keygen_us << ','
              << online_two_share_us << ','
              << (dealer_keygen_ok ? "pass" : "FAIL") << ','
              << (key_order_ok ? "pass" : "FAIL") << ','
              << (online_ok ? "pass" : "FAIL") << ','
              << (online_ok ? "pass" : "FAIL") << ',' << checker_us << ',';
    print_metric(checker_host_rss_available, checker_host_rss_bytes);
    std::cout << ',';
    print_metric(gpu_memory.available && gpu_memory.sampled,
                 gpu_memory.peak_bytes);
    std::cout << ',';
    print_metric(gpu_memory.available && gpu_memory.sampled,
                 gpu_memory.min_free_bytes);
    std::cout << ',' << ringlpn_freshness::hex(p0.header.invocation_id) << ','
              << ringlpn_freshness::hex(p0.header.ledger_digest) << '\n';
    return online_ok ? 0 : 1;
}

#ifdef RINGLPN_LIVE_CONV
}  // namespace ringlpn_conv_live
#else
}  // namespace ringlpn_fc_live
#endif

int main(int argc, char **argv) {
#ifdef RINGLPN_LIVE_CONV
    ringlpn_conv_live::Args args;
    if (!ringlpn_conv_live::public_a_validation_gate()) return 2;
    if (!ringlpn_conv_live::parse_args(argc, argv, args)) {
        ringlpn_conv_live::usage(argv[0]);
        return 2;
    }
    return args.check ? ringlpn_conv_live::run_check(args)
                      : ringlpn_conv_live::run_party(args);
#else
    ringlpn_fc_live::Args args;
    if (!ringlpn_fc_live::public_a_validation_gate()) return 2;
    if (!ringlpn_fc_live::parse_args(argc, argv, args)) {
        ringlpn_fc_live::usage(argv[0]);
        return 2;
    }
    return args.check ? ringlpn_fc_live::run_check(args)
                      : ringlpn_fc_live::run_party(args);
#endif
}
