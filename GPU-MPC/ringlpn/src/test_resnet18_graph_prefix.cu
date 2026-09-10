// Exact ResNet18 Conv0 -> StochasticTR -> stock MaxPool -> stock ReLU ->
// Conv3 composition control.  Conv0 and Conv3 use the live two-party
// Ring-LPN records; a TEST-ONLY trusted adapter supplies the nonlinear key
// material.  Each live process loads only its own private records, runs Orca's
// unchanged gpuMaxPool/gpuReluExtend/gpuConv2DBeaver consumers, and emits a
// digest-bound record.  This remains a known-zero functionality/choreography
// artifact, not a dealerless-nonlinear, privacy, or trained-model claim.

#define BUF_MEM LLAMA_BUF_MEM
#include "utils/gpu_comms.h"
#undef BUF_MEM
#include "utils/gpu_file_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "fss/dcf/gpu_maxpool.h"
#include "fss/dcf/gpu_relu.h"
#include "fss/gpu_conv2d.h"
#include "graph_mask_state.h"
#include "secure_truncate.h"
#include "stock_nonlinear_prefix_record.h"
#include "linear_preprocess.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace nonlinear = ringlpn_nonlinear_prefix;
using ringlpn_2pc::PartyChannel;
using ringlpn_2pc::PartyRandom;
using T = uint64_t;
using Digest = ringlpn_freshness::Digest;
using InvocationId = ringlpn_freshness::InvocationId;
using MaskState = ringlpn_graph::MaskStateRecord<T>;

constexpr std::array<uint8_t, 8> kRunMagic = {
    'R', 'L', 'P', 'G', 'R', 'P', 'H', '2'};
constexpr uint32_t kRunVersion = 2;
constexpr size_t kRunHeaderBytes = 320;
constexpr size_t kRunDigestBytes = 32;
constexpr uint64_t kExpectedMaxpoolKeyBytes = 503968064;
constexpr uint64_t kExpectedReluKeyBytes = 68289576;
constexpr size_t kRunArrayCount = 8;
constexpr size_t kStockCommOneGB = size_t(2) << 20;
constexpr uint64_t kConv0InputWords = 1ULL * 224 * 224 * 3;
constexpr uint64_t kConv0OutputWords = 1ULL * 112 * 112 * 64;
constexpr uint64_t kConv3InputWords = 1ULL * 56 * 56 * 64;
constexpr uint64_t kConv3OutputWords = kConv3InputWords;
constexpr int kBw = 32;
constexpr int kShift = 10;
constexpr int kTruncatedBw = kBw - kShift;
constexpr size_t kMaxRunBytes = size_t(1) << 30;

struct GraphArgs {
    bool check = false;
    bool csv_header = false;
    int party = -1;
    std::string host = "127.0.0.1";
    int port = 58100;
    std::string ledger;
    std::string invocation_text;
    InvocationId invocation{};
    Digest manifest_digest{};
    std::array<Digest, 4> record_digests{};
    std::array<Digest, 2> nonlinear_record_digests{};
    std::string conv0_record;
    std::string conv0_state;
    std::string conv3_record;
    std::string conv3_state;
    std::string nonlinear_record;
    std::string output;
    std::string p0_output;
    std::string p1_output;
    std::string p0_conv0_state;
    std::string p1_conv0_state;
    std::string p0_conv3_state;
    std::string p1_conv3_state;
    std::string p0_nonlinear_record;
    std::string p1_nonlinear_record;
};

struct RunHeader {
    int party = -1;
    uint64_t conv0_words = 0;
    uint64_t trunc_words = 0;
    uint64_t conv3_words = 0;
    InvocationId invocation{};
    uint64_t nonlinear_words = 0;
    Digest manifest_digest{};
    Digest nonlinear_record_digest{};
    Digest nonlinear_bundle_id{};
    Digest layer_identity{};
    Digest conv0_record_digest{};
    Digest conv3_record_digest{};
};

struct RunRecord {
    RunHeader header;
    std::vector<T> public_conv0;
    std::vector<T> public_trunc;
    std::vector<T> public_maxpool;
    std::vector<T> public_relu;
    std::vector<T> public_conv3;
    std::vector<T> input_mask_share;
    std::vector<T> conv3_input_mask_share;
    Digest digest{};
    std::vector<T> conv3_output_mask_share;
};
struct TruncationTotals {
    uint64_t truncations = 0;
    uint64_t handoffs = 0;
    uint64_t edabit_bits = 0;
    uint64_t dabits = 0;
    uint64_t triples = 0;
    uint64_t logical_opened_bits = 0;
    uint64_t meaningful_share_bits = 0;
    uint64_t online_dependency_rounds = 0;
    uint64_t post_mask_dependencies = 0;
    uint64_t bytes_sent = 0;
};

struct NonlinearTotals {
    uint64_t maxpool_us = 0;
    uint64_t relu_us = 0;
    uint64_t protocol_bytes_sent = 0;
    uint64_t protocol_bytes_received = 0;
};

struct PrefixTotals {
    uint64_t conv0_us = 0;
    uint64_t trunc_us = 0;
    uint64_t maxpool_us = 0;
    uint64_t relu_us = 0;
    uint64_t conv3_us = 0;
    uint64_t party_channel_bytes_sent = 0;
    uint64_t stock_nonlinear_bytes_sent = 0;
    uint64_t stock_nonlinear_bytes_received = 0;
    uint64_t party_channel_direction_switches = 0;
    uint64_t daBits = 0;
    uint64_t edaBits = 0;
    uint64_t logical_opened_bits = 0;
    uint64_t triples = 0;
    uint64_t maxpool_key_bytes = 0;
    uint64_t relu_key_bytes = 0;
};

bool parse_hex(const std::string &text, uint8_t *out, size_t bytes) {
    if (text.size() != 2 * bytes) return false;
    auto nibble = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        return -1;
    };
    for (size_t i = 0; i < bytes; ++i) {
        const int high = nibble(text[2 * i]);
        const int low = nibble(text[2 * i + 1]);
        if (high < 0 || low < 0) return false;
        out[i] = static_cast<uint8_t>((high << 4) | low);
    }
    return true;
}
bool parse_int(const std::string &text, int &out) {
    try {
        size_t used = 0;
        const long long value = std::stoll(text, &used, 10);
        if (used != text.size() || value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max()) {
            return false;
        }
        out = static_cast<int>(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_args(int argc, char **argv, GraphArgs &args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) return nullptr;
            return argv[++i];
        };
        const char *value = nullptr;
        if (key == "--check") args.check = true;
        else if (key == "--csv-header") args.csv_header = true;
        else if (key == "--party" && (value = next())) {
            if (!parse_int(value, args.party)) return false;
        } else if (key == "--host" && (value = next())) args.host = value;
        else if (key == "--port" && (value = next())) {
            if (!parse_int(value, args.port)) return false;
        } else if (key == "--ledger" && (value = next())) args.ledger = value;
        else if (key == "--invocation-id" && (value = next())) {
            args.invocation_text = value;
        } else if (key == "--manifest-digest" && (value = next())) {
            if (!parse_hex(value, args.manifest_digest.data(),
                           args.manifest_digest.size())) {
                return false;
            }
        } else if (key == "--conv0-p0-digest" && (value = next())) {
            if (!parse_hex(value, args.record_digests[0].data(), 32)) return false;
        } else if (key == "--conv0-p1-digest" && (value = next())) {
            if (!parse_hex(value, args.record_digests[1].data(), 32)) return false;
        } else if (key == "--conv3-p0-digest" && (value = next())) {
            if (!parse_hex(value, args.record_digests[2].data(), 32)) return false;
        } else if (key == "--conv3-p1-digest" && (value = next())) {
            if (!parse_hex(value, args.record_digests[3].data(), 32)) return false;
        } else if (key == "--nonlinear-p0-digest" && (value = next())) {
            if (!parse_hex(value, args.nonlinear_record_digests[0].data(),
                           args.nonlinear_record_digests[0].size())) {
                return false;
            }
        } else if (key == "--nonlinear-p1-digest" && (value = next())) {
            if (!parse_hex(value, args.nonlinear_record_digests[1].data(),
                           args.nonlinear_record_digests[1].size())) {
                return false;
            }
        } else if (key == "--conv0-record" && (value = next())) {
            args.conv0_record = value;
        } else if (key == "--conv0-state" && (value = next())) {
            args.conv0_state = value;
        } else if (key == "--conv3-record" && (value = next())) {
            args.conv3_record = value;
        } else if (key == "--conv3-state" && (value = next())) {
            args.conv3_state = value;
        } else if (key == "--nonlinear-record" && (value = next())) {
            args.nonlinear_record = value;
        } else if (key == "--output" && (value = next())) {
            args.output = value;
        } else if (key == "--p0-output" && (value = next())) {
            args.p0_output = value;
        } else if (key == "--p1-output" && (value = next())) {
            args.p1_output = value;
        } else if (key == "--p0-conv0-state" && (value = next())) {
            args.p0_conv0_state = value;
        } else if (key == "--p1-conv0-state" && (value = next())) {
            args.p1_conv0_state = value;
        } else if (key == "--p0-conv3-state" && (value = next())) {
            args.p0_conv3_state = value;
        } else if (key == "--p1-conv3-state" && (value = next())) {
            args.p1_conv3_state = value;
        } else if (key == "--p0-nonlinear-record" && (value = next())) {
            args.p0_nonlinear_record = value;
        } else if (key == "--p1-nonlinear-record" && (value = next())) {
            args.p1_nonlinear_record = value;
        } else {
            return false;
        }
    }

    const bool manifest_nonzero = std::any_of(
        args.manifest_digest.begin(), args.manifest_digest.end(),
        [](uint8_t byte) { return byte != 0; });
    if (!manifest_nonzero) return false;
    if (args.check) {
        return args.party == -1 && !args.p0_output.empty() &&
               !args.p1_output.empty() && !args.p0_conv0_state.empty() &&
               !args.p1_conv0_state.empty() && !args.p0_conv3_state.empty() &&
               !args.p1_conv3_state.empty() &&
               !args.p0_nonlinear_record.empty() &&
               !args.p1_nonlinear_record.empty() && args.output.empty() &&
               args.conv0_record.empty() && args.conv3_record.empty() &&
               args.nonlinear_record.empty();
    }
    const auto digest_nonzero = [](const Digest &digest) {
        return std::any_of(digest.begin(), digest.end(),
                           [](uint8_t byte) { return byte != 0; });
    };
    const bool digests_nonzero =
        std::all_of(args.record_digests.begin(), args.record_digests.end(),
                    digest_nonzero) &&
        std::all_of(args.nonlinear_record_digests.begin(),
                    args.nonlinear_record_digests.end(), digest_nonzero);
    return (args.party == 0 || args.party == 1) && args.port > 0 &&
           args.port < 65534 && !args.ledger.empty() &&
           args.ledger.front() == '/' &&
           ringlpn_freshness::parse_invocation_id(args.invocation_text,
                                                  args.invocation) &&
           digests_nonzero && !args.conv0_record.empty() &&
           !args.conv0_state.empty() && !args.conv3_record.empty() &&
           !args.conv3_state.empty() && !args.nonlinear_record.empty() &&
           !args.output.empty() && args.p0_output.empty() &&
           args.p1_output.empty();
}

void put_u32(std::vector<uint8_t> &out, size_t offset, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        out[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

void put_u64(std::vector<uint8_t> &out, size_t offset, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        out[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

uint32_t get_u32(const uint8_t *in, size_t offset) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(in[offset + i]) << (8 * i);
    }
    return value;
}

uint64_t get_u64(const uint8_t *in, size_t offset) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(in[offset + i]) << (8 * i);
    }
    return value;
}

bool add_u64(uint64_t &target, uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max() - target) return false;
    target += value;
    return true;
}

bool derive_run_identities(const GraphArgs &args, Digest &layer_identity,
                           Digest &plan_digest) {
    static constexpr uint8_t layer_domain[] =
        "RINGLPN-RESNET18-GRAPH-PREFIX-STOCK-NONLINEAR-ZERO-CONTROL-V2";
    std::vector<uint8_t> layer(layer_domain,
                               layer_domain + sizeof(layer_domain) - 1);
    layer.insert(layer.end(), args.manifest_digest.begin(),
                 args.manifest_digest.end());
    for (const Digest &digest : args.record_digests) {
        layer.insert(layer.end(), digest.begin(), digest.end());
    }
    for (const Digest &digest : args.nonlinear_record_digests) {
        layer.insert(layer.end(), digest.begin(), digest.end());
    }
    std::vector<uint8_t> dimensions(7 * sizeof(uint64_t), 0);
    put_u64(dimensions, 0, kConv0InputWords);
    put_u64(dimensions, 8, kConv0OutputWords);
    put_u64(dimensions, 16, nonlinear::kTruncWords);
    put_u64(dimensions, 24, nonlinear::kNonlinearWords);
    put_u64(dimensions, 32, kConv3InputWords);
    put_u64(dimensions, 40, kConv3OutputWords);
    put_u64(dimensions, 48, kShift);
    layer.insert(layer.end(), dimensions.begin(), dimensions.end());
    if (!ringlpn_freshness::digest(layer.data(), layer.size(),
                                   layer_identity)) {
        return false;
    }
    static constexpr uint8_t plan_domain[] =
        "RINGLPN-RESNET18-GRAPH-PREFIX-TRUNCATION-PLAN-V2";
    std::vector<uint8_t> plan(plan_domain,
                              plan_domain + sizeof(plan_domain) - 1);
    plan.insert(plan.end(), layer_identity.begin(), layer_identity.end());
    const uint64_t chunks =
        (kConv0OutputWords + ringlpn_2pc::kMaxSecureTruncateBatch - 1) /
        ringlpn_2pc::kMaxSecureTruncateBatch;
    std::vector<uint8_t> counts(16, 0);
    put_u64(counts, 0, kConv0OutputWords);
    put_u64(counts, 8, chunks);
    plan.insert(plan.end(), counts.begin(), counts.end());
    return ringlpn_freshness::digest(plan.data(), plan.size(), plan_digest);
}

bool copy_public_plan(const ringlpn_linear::OwnedRecord &record,
                      ringlpn_linear::LinearPlan &plan) {
    if (record.empty()) return false;
    plan = record.plan();
    return true;
}

bool exact_conv0(const ringlpn_linear::RecordMetadata &metadata,
                 const ringlpn_linear::LinearPlan &plan) {
    return metadata.protocol.bw == kBw && metadata.conv.n == 1 &&
           metadata.conv.h == 224 && metadata.conv.w == 224 &&
           metadata.conv.ci == 3 && metadata.conv.fh == 7 &&
           metadata.conv.fw == 7 && metadata.conv.co == 64 &&
           metadata.conv.padding == 3 && metadata.conv.stride == 2 &&
           plan.input_words == kConv0InputWords &&
           plan.output_words == kConv0OutputWords;
}

bool exact_conv3(const ringlpn_linear::RecordMetadata &metadata,
                 const ringlpn_linear::LinearPlan &plan) {
    return metadata.protocol.bw == kBw && metadata.conv.n == 1 &&
           metadata.conv.h == 56 && metadata.conv.w == 56 &&
           metadata.conv.ci == 64 && metadata.conv.fh == 3 &&
           metadata.conv.fw == 3 && metadata.conv.co == 64 &&
           metadata.conv.padding == 1 && metadata.conv.stride == 1 &&
           plan.input_words == kConv3InputWords &&
           plan.output_words == kConv3OutputWords;
}

bool validate_local_layer(const GraphArgs &args,
                          const ringlpn_linear::OwnedRecord &record,
                          const MaskState &state,
                          const ringlpn_linear::LinearPlan &plan,
                          uint64_t ordinal,
                          const Digest &expected_digest) {
    ringlpn_linear::RecordExpectation expected;
    expected.plan = plan;
    expected.party = args.party;
    expected.sid = record.metadata().sid;
    expected.invocation_id = record.metadata().invocation_id;
    expected.require_invocation = true;
    return ringlpn_linear::Conv2dPreprocessor::validate_record(record, expected) ==
               ringlpn_linear::Status::Ok &&
           record.metadata().digest == expected_digest &&
           state.header.party == args.party &&
           state.header.sid == record.metadata().sid &&
           state.header.layer_ordinal == ordinal &&
           state.header.input_bw == kBw && state.header.output_bw == kBw &&
           state.header.input_words == plan.input_words &&
           state.header.output_words == plan.output_words &&
           state.header.invocation_id == record.metadata().invocation_id &&
           state.header.linear_record_digest == record.metadata().digest &&
           state.input_mask_share.size() == plan.input_words &&
           state.output_mask_share.size() == plan.output_words &&
           std::equal(state.input_mask_share.begin(),
                      state.input_mask_share.end(), record.payload());
}

bool exchange_preflight(PartyChannel &channel, const GraphArgs &args,
                        const Digest &layer_identity, bool local_valid) {
    std::array<uint8_t, 97> mine{};
    std::array<uint8_t, 97> peer{};
    static constexpr uint8_t magic[16] = {
        'R','L','P','N','G','R','A','P','H','P','R','E','F','L','T','1'};
    std::copy(std::begin(magic), std::end(magic), mine.begin());
    std::copy(args.invocation.begin(), args.invocation.end(), mine.begin() + 16);
    std::copy(layer_identity.begin(), layer_identity.end(), mine.begin() + 32);
    std::copy(args.manifest_digest.begin(), args.manifest_digest.end(),
              mine.begin() + 64);
    mine[96] = local_valid ? 1 : 0;
    channel.exchange_bytes(mine.data(), peer.data(), mine.size());
    return std::equal(mine.begin(), mine.begin() + 96, peer.begin()) &&
           mine[96] == 1 && peer[96] == 1;
}

std::vector<T> reconstruct_additive(const std::vector<T> &mine, int bits,
                                    PartyChannel &channel) {
    std::vector<T> peer(mine.size());
    channel.exchange_bytes(reinterpret_cast<const uint8_t *>(mine.data()),
                           reinterpret_cast<uint8_t *>(peer.data()),
                           mine.size() * sizeof(T));
    std::vector<T> result(mine.size());
    const uint64_t mask = (uint64_t(1) << bits) - 1;
    for (size_t i = 0; i < mine.size(); ++i) {
        result[i] = (mine[i] + peer[i]) & mask;
    }
    return result;
}

bool run_conv(const ringlpn_linear::OwnedRecord &record,
              const ringlpn_linear::LinearPlan &plan,
              const std::vector<T> &public_input,
              const std::vector<T> &public_weight,
              std::vector<T> &own_output) {
    if (public_input.size() != plan.input_words ||
        public_weight.size() != plan.weight_words) {
        return false;
    }
    GPUConv2DKey<T> key{};
    key.p = {plan.protocol.bw, plan.protocol.bw, plan.conv.n, plan.conv.h,
             plan.conv.w, plan.conv.ci, plan.conv.fh, plan.conv.fw,
             plan.conv.co, plan.conv.padding, plan.conv.padding,
             plan.conv.padding, plan.conv.padding, plan.conv.stride,
             plan.conv.stride, plan.output_h, plan.output_w};
    key.p.size_I = static_cast<size_t>(plan.input_words);
    key.p.size_F = static_cast<size_t>(plan.weight_words);
    key.p.size_O = static_cast<size_t>(plan.output_words);
    key.mem_size_I = key.p.size_I * sizeof(T);
    key.mem_size_F = key.p.size_F * sizeof(T);
    key.mem_size_O = key.p.size_O * sizeof(T);
    key.I = const_cast<T *>(record.input_mask().data);
    key.F = const_cast<T *>(record.weight_mask().data);
    key.O = const_cast<T *>(record.output_correction().data);
    T *d_input = nullptr;
    T *d_weight = nullptr;
    T *d_a = nullptr;
    T *d_b = nullptr;
    T *d_output = nullptr;
    auto copy_to_gpu = [](const T *source, size_t words, T **destination) {
        return cudaMalloc(reinterpret_cast<void **>(destination),
                          words * sizeof(T)) == cudaSuccess &&
               cudaMemcpy(*destination, source, words * sizeof(T),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    };
    bool ok = copy_to_gpu(public_input.data(), public_input.size(), &d_input) &&
              copy_to_gpu(public_weight.data(), public_weight.size(),
                          &d_weight) &&
              copy_to_gpu(record.input_mask().data, record.input_mask().size,
                          &d_a) &&
              copy_to_gpu(record.weight_mask().data, record.weight_mask().size,
                          &d_b);
    if (ok) {
        Stats stats;
        d_output = gpuConv2DBeaver<T>(
            key, record.metadata().party, d_input, d_weight, d_a, d_b,
            nullptr, &stats, 0);
        own_output.resize(static_cast<size_t>(plan.output_words));
        ok = d_output != nullptr &&
             cudaMemcpy(own_output.data(), d_output,
                        own_output.size() * sizeof(T),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_output);
    return ok;
}

bool add_truncation_totals(
    TruncationTotals &total,
    const ringlpn_2pc::SecureTruncateCounters &one) {
    uint64_t bytes = 0;
    return add_u64(bytes, one.preflight_bytes_sent) &&
           add_u64(bytes, one.correlation_bytes_sent) &&
           add_u64(bytes, one.online_bytes_sent) &&
           add_u64(total.truncations, one.truncations) &&
           add_u64(total.handoffs, one.handoffs) &&
           add_u64(total.edabit_bits, one.edabit_bits) &&
           add_u64(total.dabits, one.dabits) &&
           add_u64(total.triples, one.triples) &&
           add_u64(total.logical_opened_bits, one.logical_opened_bits) &&
           add_u64(total.meaningful_share_bits,
                   one.meaningful_share_bits) &&
           add_u64(total.online_dependency_rounds,
                   one.online_dependency_rounds) &&
           add_u64(total.post_mask_dependencies,
                   one.post_mask_dependencies) &&
           add_u64(total.bytes_sent, bytes);
}

std::vector<uint8_t> encode_run_header(const RunHeader &header) {
    std::vector<uint8_t> out(kRunHeaderBytes, 0);
    std::copy(kRunMagic.begin(), kRunMagic.end(), out.begin());
    put_u32(out, 8, kRunVersion);
    put_u32(out, 12, static_cast<uint32_t>(header.party));
    put_u32(out, 16, kBw);
    put_u32(out, 20, kShift);
    put_u64(out, 24, header.conv0_words);
    put_u64(out, 32, header.trunc_words);
    put_u64(out, 40, header.nonlinear_words);
    put_u64(out, 48, header.conv3_words);
    std::copy(header.invocation.begin(), header.invocation.end(),
              out.begin() + 56);
    std::copy(header.manifest_digest.begin(), header.manifest_digest.end(),
              out.begin() + 72);
    std::copy(header.layer_identity.begin(), header.layer_identity.end(),
              out.begin() + 104);
    std::copy(header.conv0_record_digest.begin(),
              header.conv0_record_digest.end(), out.begin() + 136);
    std::copy(header.conv3_record_digest.begin(),
              header.conv3_record_digest.end(), out.begin() + 168);
    std::copy(header.nonlinear_record_digest.begin(),
              header.nonlinear_record_digest.end(), out.begin() + 200);
    std::copy(header.nonlinear_bundle_id.begin(),
              header.nonlinear_bundle_id.end(), out.begin() + 232);
    return out;
}

bool decode_run_header(const uint8_t *in, size_t size, RunHeader &header) {
    if (size < kRunHeaderBytes ||
        !std::equal(kRunMagic.begin(), kRunMagic.end(), in) ||
        get_u32(in, 8) != kRunVersion || get_u32(in, 16) != kBw ||
        get_u32(in, 20) != kShift ||
        !std::all_of(in + 264, in + kRunHeaderBytes,
                     [](uint8_t byte) { return byte == 0; })) {
        return false;
    }
    header.party = static_cast<int>(get_u32(in, 12));
    header.conv0_words = get_u64(in, 24);
    header.trunc_words = get_u64(in, 32);
    header.nonlinear_words = get_u64(in, 40);
    header.conv3_words = get_u64(in, 48);
    std::copy(in + 56, in + 72, header.invocation.begin());
    std::copy(in + 72, in + 104, header.manifest_digest.begin());
    std::copy(in + 104, in + 136, header.layer_identity.begin());
    std::copy(in + 136, in + 168, header.conv0_record_digest.begin());
    std::copy(in + 168, in + 200, header.conv3_record_digest.begin());
    std::copy(in + 200, in + 232, header.nonlinear_record_digest.begin());
    std::copy(in + 232, in + 264, header.nonlinear_bundle_id.begin());
    return (header.party == 0 || header.party == 1) &&
           header.conv0_words == kConv0OutputWords &&
           header.trunc_words == nonlinear::kTruncWords &&
           header.nonlinear_words == nonlinear::kNonlinearWords &&
           header.conv3_words == kConv3OutputWords;
}

bool serialize_run_record(const RunRecord &record,
                          std::vector<uint8_t> &bytes, Digest &digest) {
    if (record.public_conv0.size() != kConv0OutputWords ||
        record.public_trunc.size() != nonlinear::kTruncWords ||
        record.public_maxpool.size() != nonlinear::kNonlinearWords ||
        record.public_relu.size() != nonlinear::kNonlinearWords ||
        record.public_conv3.size() != kConv3OutputWords ||
        record.input_mask_share.size() != kConv0InputWords ||
        record.conv3_input_mask_share.size() != kConv3InputWords ||
        record.conv3_output_mask_share.size() != kConv3OutputWords) {
        return false;
    }
    const std::vector<T> *arrays[kRunArrayCount] = {
        &record.public_conv0, &record.public_trunc,
        &record.public_maxpool, &record.public_relu,
        &record.public_conv3, &record.input_mask_share,
        &record.conv3_input_mask_share, &record.conv3_output_mask_share};
    bytes = encode_run_header(record.header);
    const size_t words =
        static_cast<size_t>(kConv0OutputWords) +
        static_cast<size_t>(nonlinear::kTruncWords) +
        2 * static_cast<size_t>(nonlinear::kNonlinearWords) +
        static_cast<size_t>(kConv3OutputWords) +
        static_cast<size_t>(kConv0InputWords) +
        static_cast<size_t>(kConv3InputWords) +
        static_cast<size_t>(kConv3OutputWords);
    if (words > (kMaxRunBytes - kRunHeaderBytes - kRunDigestBytes) /
                    sizeof(T)) {
        return false;
    }
    bytes.resize(kRunHeaderBytes + words * sizeof(T));
    size_t cursor = kRunHeaderBytes;
    for (const std::vector<T> *array : arrays) {
        for (T value : *array) {
            put_u64(bytes, cursor, value);
            cursor += sizeof(T);
        }
    }
    if (!ringlpn_freshness::digest(bytes.data(), bytes.size(), digest)) {
        return false;
    }
    bytes.insert(bytes.end(), digest.begin(), digest.end());
    return true;
}

bool write_private_atomic_pair(const GraphArgs &args, PartyChannel &channel,
                               const std::vector<uint8_t> &bytes) {
    const std::string temporary = args.output + ".tmp";
    std::remove(temporary.c_str());
    bool staged = ringlpn_graph::write_mask_state_bytes(temporary, bytes);
    if (staged) {
        std::error_code error;
        std::filesystem::permissions(
            temporary,
            std::filesystem::perms::owner_read |
                std::filesystem::perms::owner_write,
            std::filesystem::perm_options::replace, error);
        staged = !error;
    }
    uint8_t mine = staged ? 1 : 0;
    uint8_t peer = 0;
    channel.exchange_bytes(&mine, &peer, 1);
    if (!staged || peer != 1) {
        std::remove(temporary.c_str());
        return false;
    }
    const bool renamed =
        std::rename(temporary.c_str(), args.output.c_str()) == 0;
    mine = renamed ? 1 : 0;
    peer = 0;
    channel.exchange_bytes(&mine, &peer, 1);
    if (!renamed || peer != 1) {
        std::remove(temporary.c_str());
        if (renamed) std::remove(args.output.c_str());
        return false;
    }
    return true;
}

bool read_run_record(const std::string &path, RunRecord &record) {
    std::error_code error;
    const uintmax_t file_size = std::filesystem::file_size(path, error);
    if (error || file_size < kRunHeaderBytes + kRunDigestBytes ||
        file_size > kMaxRunBytes) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
    in.read(reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    if (!in || in.peek() != std::ifstream::traits_type::eof()) return false;
    RunHeader header;
    if (!decode_run_header(bytes.data(), bytes.size(), header)) return false;
    const size_t words =
        static_cast<size_t>(header.conv0_words) +
        static_cast<size_t>(header.trunc_words) +
        2 * static_cast<size_t>(header.nonlinear_words) +
        static_cast<size_t>(header.conv3_words) +
        static_cast<size_t>(kConv0InputWords) +
        static_cast<size_t>(kConv3InputWords) +
        static_cast<size_t>(header.conv3_words);
    if (words > (kMaxRunBytes - kRunHeaderBytes - kRunDigestBytes) /
                    sizeof(T) ||
        kRunHeaderBytes + words * sizeof(T) + kRunDigestBytes !=
            bytes.size()) {
        return false;
    }
    Digest expected{};
    if (!ringlpn_freshness::digest(
            bytes.data(), kRunHeaderBytes + words * sizeof(T), expected) ||
        !std::equal(expected.begin(), expected.end(),
                    bytes.begin() + kRunHeaderBytes + words * sizeof(T))) {
        return false;
    }
    RunRecord parsed;
    parsed.header = header;
    std::vector<T> *arrays[kRunArrayCount] = {
        &parsed.public_conv0, &parsed.public_trunc,
        &parsed.public_maxpool, &parsed.public_relu,
        &parsed.public_conv3, &parsed.input_mask_share,
        &parsed.conv3_input_mask_share,
        &parsed.conv3_output_mask_share};
    const size_t sizes[kRunArrayCount] = {
        static_cast<size_t>(header.conv0_words),
        static_cast<size_t>(header.trunc_words),
        static_cast<size_t>(header.nonlinear_words),
        static_cast<size_t>(header.nonlinear_words),
        static_cast<size_t>(header.conv3_words),
        static_cast<size_t>(kConv0InputWords),
        static_cast<size_t>(kConv3InputWords),
        static_cast<size_t>(header.conv3_words)};
    size_t cursor = kRunHeaderBytes;
    for (size_t array = 0; array < kRunArrayCount; ++array) {
        arrays[array]->resize(sizes[array]);
        for (T &value : *arrays[array]) {
            value = get_u64(bytes.data(), cursor);
            cursor += sizeof(T);
        }
    }
    parsed.digest = expected;
    record = std::move(parsed);
    return true;
}
bool validate_nonlinear_record(const GraphArgs &args,
                               const nonlinear::Record &record,
                               const Digest &expected_digest) {
    return record.header.party == args.party &&
           record.header.scope ==
               nonlinear::kTrustedStockDealerKnownZeroScope &&
           record.header.invocation == args.invocation &&
           record.header.manifest_digest == args.manifest_digest &&
           record.header.linear_record_digests == args.record_digests &&
           record.header.trunc_bw == kTruncatedBw &&
           record.header.full_bw == kBw &&
           record.header.trunc_words == nonlinear::kTruncWords &&
           record.header.nonlinear_words == nonlinear::kNonlinearWords &&
           record.header.maxpool_key_bytes == kExpectedMaxpoolKeyBytes &&
           record.header.relu_key_bytes == kExpectedReluKeyBytes &&
           record.digest == expected_digest;
}

MaxpoolParams exact_maxpool_params() {
    MaxpoolParams params = {
        kTruncatedBw, kTruncatedBw, 0, 0, kBw,
        1, 112, 112, 64,
        3, 3,
        2, 2,
        1, 1,
        1, 1,
        0, 0, false};
    initPoolParams(params);
    return params;
}

class ScopedStdoutToStderr {
  public:
    ScopedStdoutToStderr() {
        std::fflush(stdout);
        saved_ = ::dup(STDOUT_FILENO);
        ok_ = saved_ >= 0 && ::dup2(STDERR_FILENO, STDOUT_FILENO) >= 0;
    }

    ScopedStdoutToStderr(const ScopedStdoutToStderr &) = delete;
    ScopedStdoutToStderr &operator=(const ScopedStdoutToStderr &) = delete;

    ~ScopedStdoutToStderr() {
        std::fflush(stdout);
        if (saved_ >= 0) {
            ::dup2(saved_, STDOUT_FILENO);
            ::close(saved_);
        }
    }

    bool ok() const { return ok_; }

  private:
    int saved_ = -1;
    bool ok_ = false;
};

void release_maxpool_key(dcf::GPUMaxpoolKey<T> &key) {
    for (int round = 1; round <= 8; ++round) {
        delete[] key.reluKey[round].dreluKey.dcfKey.dcfTreeKey;
    }
    delete[] key.reluKey;
    delete[] key.andKey;
}

void release_relu_key(dcf::GPUReluExtendKey<T> &key) {
    delete[] key.dReluKey.dcfKey.dcfTreeKey;
}

bool run_stock_nonlinear(int party, const std::string &host, int port,
                         const nonlinear::Record &record,
                         const std::vector<T> &public_input,
                         std::vector<T> &public_maxpool,
                         std::vector<T> &public_relu,
                         NonlinearTotals &totals) {
    if (public_input.size() != nonlinear::kTruncWords ||
        record.raw_key_bytes() !=
            kExpectedMaxpoolKeyBytes + kExpectedReluKeyBytes) {
        return false;
    }
    uint8_t *cursor =
        const_cast<uint8_t *>(record.raw_key_data());
    uint8_t *const maxpool_end = cursor + kExpectedMaxpoolKeyBytes;
    uint8_t *const key_end = maxpool_end + kExpectedReluKeyBytes;
    MaxpoolParams params = exact_maxpool_params();
    dcf::GPUMaxpoolKey<T> maxpool_key =
        dcf::readGPUMaxpoolKey<T>(params, &cursor);
    maxpool_key.andKey = new GPUAndKey[9]();
    const bool maxpool_key_ok =
        cursor == maxpool_end &&
        std::all_of(maxpool_key.reluKey + 1, maxpool_key.reluKey + 9,
                    [](const dcf::GPU2RoundReLUKey<T> &key) {
                        return key.bin == kTruncatedBw &&
                               key.bout == kTruncatedBw &&
                               key.N ==
                                   static_cast<int>(
                                       nonlinear::kNonlinearWords);
                    });
    dcf::GPUReluExtendKey<T> relu_key =
        dcf::readGPUReluExtendKey<T>(&cursor);
    const bool relu_key_ok =
        cursor == key_end && relu_key.bin == kTruncatedBw &&
        relu_key.bout == kBw &&
        relu_key.N == static_cast<int>(nonlinear::kNonlinearWords);
    if (!maxpool_key_ok || !relu_key_ok) {
        release_maxpool_key(maxpool_key);
        release_relu_key(relu_key);
        return false;
    }

    T *d_input = reinterpret_cast<T *>(moveToGPU(
        reinterpret_cast<uint8_t *>(const_cast<T *>(public_input.data())),
        public_input.size() * sizeof(T), nullptr));
    AESGlobalContext gaes;
    initAESContext(&gaes);
    Stats stats;
    const size_t original_one_gb = OneGB;
    OneGB = kStockCommOneGB;
    ScopedStdoutToStderr redirect;
    if (!redirect.ok()) {
        OneGB = original_one_gb;
        gpuFree(d_input);
        release_maxpool_key(maxpool_key);
        release_relu_key(relu_key);
        return false;
    }
    bool ok = false;
    {
        GpuPeer peer(true);
        peer.connect(party, host, port);
        const auto maxpool_started = std::chrono::steady_clock::now();
        T *d_maxpool = dcf::gpuMaxPool<T>(
            &peer, party, params, maxpool_key, d_input, nullptr, &gaes,
            &stats);
        totals.maxpool_us = static_cast<uint64_t>(
            std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - maxpool_started)
                .count());
        public_maxpool.resize(
            static_cast<size_t>(nonlinear::kNonlinearWords));
        const cudaError_t maxpool_copy = cudaMemcpy(
            public_maxpool.data(), d_maxpool,
            public_maxpool.size() * sizeof(T), cudaMemcpyDeviceToHost);

        const auto relu_started = std::chrono::steady_clock::now();
        std::pair<u32 *, T *> relu_result = dcf::gpuReluExtend<T>(
            &peer, party, relu_key, d_maxpool, &gaes, &stats);
        T *d_relu = relu_result.second;
        totals.relu_us = static_cast<uint64_t>(
            std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - relu_started)
                .count());
        public_relu.resize(
            static_cast<size_t>(nonlinear::kNonlinearWords));
        const cudaError_t relu_copy = cudaMemcpy(
            public_relu.data(), d_relu, public_relu.size() * sizeof(T),
            cudaMemcpyDeviceToHost);
        gpuFree(d_relu);
        gpuFree(d_maxpool);
        gpuFree(relu_result.first);
        gpuFree(d_input);
        const cudaError_t synchronized = cudaDeviceSynchronize();
        totals.protocol_bytes_sent = peer.bytesSent();
        totals.protocol_bytes_received = peer.bytesReceived();
        peer.close();
        peer.freeCommBufs(true);
        delete static_cast<SocketBuf *>(peer.peer->keyBuf);
        delete peer.peer;
        ok = maxpool_copy == cudaSuccess && relu_copy == cudaSuccess &&
             synchronized == cudaSuccess;
    }
    OneGB = original_one_gb;
    release_maxpool_key(maxpool_key);
    release_relu_key(relu_key);
    return ok;
}

int run_party(const GraphArgs &args) {
    const auto started = std::chrono::steady_clock::now();
    ringlpn_linear::OwnedRecord conv0;
    ringlpn_linear::OwnedRecord conv3;
    nonlinear::Record nonlinear_record;
    MaskState conv0_state;
    MaskState conv3_state;
    ringlpn_linear::LinearPlan conv0_plan;
    ringlpn_linear::LinearPlan conv3_plan;
    const size_t own_conv0_digest = args.party == 0 ? 0 : 1;
    const size_t own_conv3_digest = args.party == 0 ? 2 : 3;
    const size_t own_nonlinear_digest = static_cast<size_t>(args.party);
    std::error_code output_error;
    const bool output_absent =
        !std::filesystem::exists(args.output, output_error) &&
        !std::filesystem::exists(args.output + ".tmp", output_error);
    const bool local_valid =
        !output_error && output_absent &&
        ringlpn_linear::Conv2dPreprocessor::open_record(
            args.conv0_record, conv0) == ringlpn_linear::Status::Ok &&
        ringlpn_linear::Conv2dPreprocessor::open_record(
            args.conv3_record, conv3) == ringlpn_linear::Status::Ok &&
        nonlinear::read_record(args.nonlinear_record, nonlinear_record) &&
        ringlpn_graph::read_mask_state(args.conv0_state, conv0_state) &&
        ringlpn_graph::read_mask_state(args.conv3_state, conv3_state) &&
        copy_public_plan(conv0, conv0_plan) &&
        copy_public_plan(conv3, conv3_plan) &&
        exact_conv0(conv0.metadata(), conv0_plan) &&
        exact_conv3(conv3.metadata(), conv3_plan) &&
        validate_local_layer(args, conv0, conv0_state, conv0_plan, 1,
                             args.record_digests[own_conv0_digest]) &&
        validate_local_layer(args, conv3, conv3_state, conv3_plan, 2,
                             args.record_digests[own_conv3_digest]) &&
        validate_nonlinear_record(
            args, nonlinear_record,
            args.nonlinear_record_digests[own_nonlinear_digest]);

    Digest layer_identity{};
    Digest plan_digest{};
    ringlpn_freshness::Claim claim;
    const bool identity_ok =
        derive_run_identities(args, layer_identity, plan_digest);
    const bool claim_ok =
        local_valid && identity_ok &&
        ringlpn_freshness::claim_namespace_once(
            args.ledger, args.party, args.invocation, layer_identity,
            plan_digest, claim);
    PartyChannel channel(args.party, args.host, args.port,
                         /*defer_ot_setup=*/true,
                         /*require_loopback_endpoints=*/true);
    if (!exchange_preflight(channel, args, layer_identity,
                            local_valid && claim_ok)) {
        std::remove((args.output + ".tmp").c_str());
        std::fprintf(stderr,
                     "[graph-prefix] public/local preflight rejected\n");
        return 2;
    }

    PrefixTotals totals;
    if (ringlpn_linear::Conv2dPreprocessor::initialize_gpu() !=
        ringlpn_linear::Status::Ok) {
        std::fprintf(stderr, "[graph-prefix] GPU initialization failed\n");
        return 2;
    }
    std::vector<T> public_input0 = reconstruct_additive(
        conv0_state.input_mask_share, kBw, channel);
    const ringlpn_linear::WordView conv0_weight = conv0.weight_mask();
    std::vector<T> conv0_weight_share(conv0_weight.begin(),
                                      conv0_weight.end());
    std::vector<T> public_weight0 =
        reconstruct_additive(conv0_weight_share, kBw, channel);
    std::vector<T> conv0_output_share;
    const auto conv0_started = std::chrono::steady_clock::now();
    bool ok = run_conv(conv0, conv0_plan, public_input0, public_weight0,
                       conv0_output_share);
    std::vector<T> public_conv0;
    if (ok) {
        public_conv0 =
            reconstruct_additive(conv0_output_share, kBw, channel);
    }
    totals.conv0_us = static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - conv0_started)
            .count());

    PartyRandom random;
    const std::vector<T> &trunc_next_mask =
        nonlinear_record.trunc_next_mask_share;
    std::vector<T> trunc_output;
    std::vector<T> public_trunc;
    TruncationTotals truncation;
    size_t cursor = 0;
    uint64_t chunk = 0;
    const auto trunc_started = std::chrono::steady_clock::now();
    while (ok && cursor < public_conv0.size()) {
        const size_t count = std::min(
            ringlpn_2pc::kMaxSecureTruncateBatch,
            public_conv0.size() - cursor);
        ringlpn_2pc::SecureTruncateParams params;
        params.bw = kBw;
        params.shift = kShift;
        params.count = count;
        ringlpn_freshness::Coordinates coordinates;
        coordinates.kind = ringlpn_freshness::Kind::kConversionEdabit;
        coordinates.layer = 1;
        coordinates.phase =
            ringlpn_freshness::Phase::kConvertCorrelation;
        coordinates.primitive_ordinal = 2;
        coordinates.conversion_chunk = chunk;
        ok = ringlpn_freshness::derive_correlation_id(
            args.invocation, layer_identity, coordinates,
            params.correlation_id);
        params.sid =
            ringlpn_freshness::compatibility_handle(params.correlation_id);
        std::vector<T> masked_chunk(public_conv0.begin() + cursor,
                                    public_conv0.begin() + cursor + count);
        std::vector<T> mask_chunk(
            conv0_state.output_mask_share.begin() + cursor,
            conv0_state.output_mask_share.begin() + cursor + count);
        std::vector<T> next_chunk(trunc_next_mask.begin() + cursor,
                                  trunc_next_mask.begin() + cursor + count);
        std::vector<T> output_chunk;
        std::vector<T> public_chunk;
        ringlpn_2pc::SecureTruncateCounters one;
        if (ok) {
            ok = ringlpn_2pc::secure_stochastic_truncate_batch(
                     params, masked_chunk, mask_chunk, next_chunk, channel,
                     random, output_chunk, public_chunk, one) &&
                 add_truncation_totals(truncation, one);
        }
        if (ok) {
            trunc_output.insert(trunc_output.end(), output_chunk.begin(),
                                output_chunk.end());
            public_trunc.insert(public_trunc.end(), public_chunk.begin(),
                                public_chunk.end());
            cursor += count;
            ++chunk;
        }
    }
    totals.trunc_us = static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - trunc_started)
            .count());

    NonlinearTotals nonlinear_totals;
    std::vector<T> public_maxpool;
    std::vector<T> public_relu;
    if (ok) {
        ok = run_stock_nonlinear(
            args.party, args.host, args.port + 2, nonlinear_record,
            public_trunc, public_maxpool, public_relu, nonlinear_totals);
    }
    totals.maxpool_us = nonlinear_totals.maxpool_us;
    totals.relu_us = nonlinear_totals.relu_us;
    totals.maxpool_key_bytes = nonlinear_record.header.maxpool_key_bytes;
    totals.relu_key_bytes = nonlinear_record.header.relu_key_bytes;

    std::vector<T> public_input3;
    std::vector<T> public_weight3;
    std::vector<T> conv3_output_share;
    std::vector<T> public_conv3;
    if (ok) {
        public_input3.resize(
            static_cast<size_t>(nonlinear::kNonlinearWords));
        const uint64_t mask = std::numeric_limits<uint32_t>::max();
        for (size_t i = 0; i < public_input3.size(); ++i) {
            public_input3[i] =
                (public_relu[i] + nonlinear_record.remask_delta[i]) & mask;
        }
        const ringlpn_linear::WordView conv3_weight = conv3.weight_mask();
        std::vector<T> conv3_weight_share(conv3_weight.begin(),
                                          conv3_weight.end());
        public_weight3 =
            reconstruct_additive(conv3_weight_share, kBw, channel);
        const auto conv3_started = std::chrono::steady_clock::now();
        ok = run_conv(conv3, conv3_plan, public_input3, public_weight3,
                      conv3_output_share);
        if (ok) {
            public_conv3 = reconstruct_additive(
                conv3_output_share, kBw, channel);
        }
        totals.conv3_us = static_cast<uint64_t>(
            std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - conv3_started)
                .count());
    }
    try {
        channel.finish_ots();
        totals.party_channel_bytes_sent = channel.bytes_sent();
        totals.stock_nonlinear_bytes_sent =
            nonlinear_totals.protocol_bytes_sent;
        totals.stock_nonlinear_bytes_received =
            nonlinear_totals.protocol_bytes_received;
        totals.party_channel_direction_switches =
            channel.direction_switches();
    } catch (...) {
        ok = false;
    }
    totals.daBits = truncation.dabits;
    totals.edaBits = truncation.edabit_bits;
    totals.logical_opened_bits = truncation.logical_opened_bits;
    totals.triples = truncation.triples;
    ok = ok && truncation.truncations == kConv0OutputWords &&
         truncation.handoffs == kConv0OutputWords &&
         trunc_output.size() == nonlinear::kTruncWords &&
         public_trunc.size() == nonlinear::kTruncWords &&
         public_maxpool.size() == nonlinear::kNonlinearWords &&
         public_relu.size() == nonlinear::kNonlinearWords &&
         public_input3.size() == kConv3InputWords &&
         public_conv3.size() == kConv3OutputWords;

    RunRecord result;
    result.header.party = args.party;
    result.header.conv0_words = kConv0OutputWords;
    result.header.trunc_words = nonlinear::kTruncWords;
    result.header.nonlinear_words = nonlinear::kNonlinearWords;
    result.header.conv3_words = kConv3OutputWords;
    result.header.invocation = args.invocation;
    result.header.manifest_digest = args.manifest_digest;
    result.header.layer_identity = layer_identity;
    result.header.conv0_record_digest = conv0.digest;
    result.header.conv3_record_digest = conv3.digest;
    result.header.nonlinear_record_digest = nonlinear_record.digest;
    result.header.nonlinear_bundle_id = nonlinear_record.header.bundle_id;
    result.public_conv0 = std::move(public_conv0);
    result.public_trunc = std::move(public_trunc);
    result.public_maxpool = std::move(public_maxpool);
    result.public_relu = std::move(public_relu);
    result.public_conv3 = std::move(public_conv3);
    result.input_mask_share = conv0_state.input_mask_share;
    result.conv3_input_mask_share = conv3_state.input_mask_share;
    result.conv3_output_mask_share = conv3_state.output_mask_share;
    std::vector<uint8_t> bytes;
    Digest result_digest{};
    bool serialized = ok && serialize_run_record(result, bytes, result_digest);
    uint8_t ready = serialized ? 1 : 0;
    uint8_t peer_ready = 0;
    channel.exchange_bytes(&ready, &peer_ready, 1);
    ok = serialized && peer_ready == 1 &&
         write_private_atomic_pair(args, channel, bytes);

    const uint64_t total_us = static_cast<uint64_t>(
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - started)
            .count());
    std::printf(
        "%d,"
        "%llu,%llu,%llu,%llu,%llu,%llu,"
        "%llu,%llu,%llu,%llu,%llu,"
        "%llu,%llu,%llu,%llu,"
        "%llu,%llu,%llu,"
        "%s\n",
        args.party, static_cast<unsigned long long>(total_us),
        static_cast<unsigned long long>(totals.conv0_us),
        static_cast<unsigned long long>(totals.trunc_us),
        static_cast<unsigned long long>(totals.maxpool_us),
        static_cast<unsigned long long>(totals.relu_us),
        static_cast<unsigned long long>(totals.conv3_us),
        static_cast<unsigned long long>(
            totals.party_channel_bytes_sent +
            totals.stock_nonlinear_bytes_sent),
        static_cast<unsigned long long>(totals.party_channel_bytes_sent),
        static_cast<unsigned long long>(totals.stock_nonlinear_bytes_sent),
        static_cast<unsigned long long>(totals.stock_nonlinear_bytes_received),
        static_cast<unsigned long long>(
            totals.party_channel_direction_switches),
        static_cast<unsigned long long>(totals.daBits),
        static_cast<unsigned long long>(totals.edaBits),
        static_cast<unsigned long long>(totals.logical_opened_bits),
        static_cast<unsigned long long>(totals.triples),
        static_cast<unsigned long long>(totals.maxpool_key_bytes),
        static_cast<unsigned long long>(totals.relu_key_bytes),
        static_cast<unsigned long long>(
            totals.maxpool_key_bytes + totals.relu_key_bytes),
        ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}

bool same_run_headers(const RunHeader &p0, const RunHeader &p1,
                      const Digest &manifest_digest) {
    return p0.party == 0 && p1.party == 1 &&
           p0.conv0_words == p1.conv0_words &&
           p0.trunc_words == p1.trunc_words &&
           p0.nonlinear_words == p1.nonlinear_words &&
           p0.conv3_words == p1.conv3_words &&
           p0.invocation == p1.invocation &&
           p0.manifest_digest == manifest_digest &&
           p1.manifest_digest == manifest_digest &&
           p0.layer_identity == p1.layer_identity &&
           p0.nonlinear_bundle_id == p1.nonlinear_bundle_id;
}

int run_check(const GraphArgs &args) {
    RunRecord p0;
    RunRecord p1;
    MaskState conv0_p0;
    MaskState conv0_p1;
    MaskState conv3_p0;
    MaskState conv3_p1;
    nonlinear::Record nonlinear_p0;
    nonlinear::Record nonlinear_p1;
    const bool parsed =
        read_run_record(args.p0_output, p0) &&
        read_run_record(args.p1_output, p1) &&
        same_run_headers(p0.header, p1.header, args.manifest_digest) &&
        ringlpn_graph::read_mask_state(args.p0_conv0_state, conv0_p0) &&
        ringlpn_graph::read_mask_state(args.p1_conv0_state, conv0_p1) &&
        ringlpn_graph::read_mask_state(args.p0_conv3_state, conv3_p0) &&
        ringlpn_graph::read_mask_state(args.p1_conv3_state, conv3_p1) &&
        nonlinear::read_record(args.p0_nonlinear_record, nonlinear_p0) &&
        nonlinear::read_record(args.p1_nonlinear_record, nonlinear_p1);
    if (!parsed) {
        std::fprintf(stderr, "[graph-prefix-check] record parsing failed\n");
        return 1;
    }
    const std::array<Digest, 4> linear_digests = {
        conv0_p0.header.linear_record_digest,
        conv0_p1.header.linear_record_digest,
        conv3_p0.header.linear_record_digest,
        conv3_p1.header.linear_record_digest};
    const bool records_ok =
        ringlpn_graph::mask_state_headers_match(conv0_p0.header,
                                                conv0_p1.header) &&
        ringlpn_graph::mask_state_headers_match(conv3_p0.header,
                                                conv3_p1.header) &&
        conv0_p0.header.layer_ordinal == 1 &&
        conv3_p0.header.layer_ordinal == 2 &&
        conv0_p0.header.input_words == kConv0InputWords &&
        conv0_p0.header.output_words == kConv0OutputWords &&
        conv3_p0.header.input_words == kConv3InputWords &&
        conv3_p0.header.output_words == kConv3OutputWords &&
        p0.header.conv0_record_digest == linear_digests[0] &&
        p1.header.conv0_record_digest == linear_digests[1] &&
        p0.header.conv3_record_digest == linear_digests[2] &&
        p1.header.conv3_record_digest == linear_digests[3] &&
        nonlinear::headers_match(nonlinear_p0.header,
                                 nonlinear_p1.header) &&
        nonlinear_p0.header.scope ==
            nonlinear::kTrustedStockDealerKnownZeroScope &&
        nonlinear_p0.header.manifest_digest == args.manifest_digest &&
        nonlinear_p0.header.invocation == p0.header.invocation &&
        nonlinear_p0.header.linear_record_digests == linear_digests &&
        nonlinear_p0.header.maxpool_key_bytes ==
            kExpectedMaxpoolKeyBytes &&
        nonlinear_p0.header.relu_key_bytes == kExpectedReluKeyBytes &&
        p0.header.nonlinear_record_digest == nonlinear_p0.digest &&
        p1.header.nonlinear_record_digest == nonlinear_p1.digest &&
        p0.header.nonlinear_bundle_id == nonlinear_p0.header.bundle_id &&
        p1.header.nonlinear_bundle_id == nonlinear_p1.header.bundle_id &&
        p0.input_mask_share == conv0_p0.input_mask_share &&
        p1.input_mask_share == conv0_p1.input_mask_share &&
        p0.conv3_input_mask_share == conv3_p0.input_mask_share &&
        p1.conv3_input_mask_share == conv3_p1.input_mask_share &&
        p0.conv3_output_mask_share == conv3_p0.output_mask_share &&
        p1.conv3_output_mask_share == conv3_p1.output_mask_share;
    if (!records_ok) {
        std::fprintf(stderr,
                     "[graph-prefix-check] source/binding validation failed\n");
        return 1;
    }

    const uint64_t full_mask = std::numeric_limits<uint32_t>::max();
    const uint64_t trunc_mask = (uint64_t(1) << kTruncatedBw) - 1;
    bool conv0_ok = p0.public_conv0 == p1.public_conv0;
    bool truncation_ok = p0.public_trunc == p1.public_trunc;
    bool maxpool_ok = p0.public_maxpool == p1.public_maxpool;
    bool relu_ok = p0.public_relu == p1.public_relu;
    bool conv3_ok = p0.public_conv3 == p1.public_conv3;
    bool state_link_ok = true;

    std::vector<T> clear_truncated(
        static_cast<size_t>(nonlinear::kTruncWords));
    for (size_t i = 0; i < clear_truncated.size(); ++i) {
        const T conv0_mask =
            (conv0_p0.output_mask_share[i] +
             conv0_p1.output_mask_share[i]) &
            full_mask;
        conv0_ok = conv0_ok && p0.public_conv0[i] == conv0_mask;
        const T next_mask =
            (nonlinear_p0.trunc_next_mask_share[i] +
             nonlinear_p1.trunc_next_mask_share[i]) &
            trunc_mask;
        clear_truncated[i] =
            (p0.public_trunc[i] - next_mask) & trunc_mask;
        truncation_ok =
            truncation_ok && clear_truncated[i] == 0;
    }

    std::vector<T> maxpool_reference(
        static_cast<size_t>(nonlinear::kNonlinearWords), 0);
    for (int oh = 0; oh < 56; ++oh) {
        for (int ow = 0; ow < 56; ++ow) {
            for (int channel = 0; channel < 64; ++channel) {
                T best = 0;
                for (int fh = 0; fh < 3; ++fh) {
                    for (int fw = 0; fw < 3; ++fw) {
                        const int ih = oh * 2 + fh - 1;
                        const int iw = ow * 2 + fw - 1;
                        if (ih < 0 || ih >= 112 || iw < 0 || iw >= 112) {
                            continue;
                        }
                        const size_t index =
                            (static_cast<size_t>(ih) * 112 + iw) * 64 +
                            channel;
                        const T value = clear_truncated[index];
                        if (value > best) best = value;
                    }
                }
                maxpool_reference[
                    (static_cast<size_t>(oh) * 56 + ow) * 64 + channel] =
                    best;
            }
        }
    }
    std::vector<T> relu_reference(maxpool_reference.size(), 0);
    for (size_t i = 0; i < maxpool_reference.size(); ++i) {
        const T maxpool_mask =
            (nonlinear_p0.maxpool_output_mask_share[i] +
             nonlinear_p1.maxpool_output_mask_share[i]) &
            trunc_mask;
        const T observed =
            (p0.public_maxpool[i] - maxpool_mask) & trunc_mask;
        maxpool_ok =
            maxpool_ok && observed == maxpool_reference[i];

        const T relu_mask =
            (nonlinear_p0.relu_output_mask_share[i] +
             nonlinear_p1.relu_output_mask_share[i]) &
            full_mask;
        relu_reference[i] = maxpool_reference[i];
        const T observed_relu =
            (p0.public_relu[i] - relu_mask) & full_mask;
        relu_ok = relu_ok && observed_relu == relu_reference[i];

        const T conv3_input_mask =
            (conv3_p0.input_mask_share[i] +
             conv3_p1.input_mask_share[i]) &
            full_mask;
        const T remasked_input =
            (p0.public_relu[i] + nonlinear_p0.remask_delta[i]) &
            full_mask;
        state_link_ok =
            state_link_ok &&
            remasked_input ==
                ((relu_reference[i] + conv3_input_mask) & full_mask);
    }
    for (size_t i = 0; i < static_cast<size_t>(kConv3OutputWords); ++i) {
        const T conv3_mask =
            (conv3_p0.output_mask_share[i] +
             conv3_p1.output_mask_share[i]) &
            full_mask;
        conv3_ok = conv3_ok && p0.public_conv3[i] == conv3_mask;
    }

    const bool all_ok = conv0_ok && truncation_ok && maxpool_ok && relu_ok &&
                        state_link_ok && conv3_ok;
    if (args.csv_header) {
        std::printf(
            "conv0_linear,secure_stochastic_truncation,stock_maxpool_execution,"
            "stock_relu_execution,conv3_linear,state_link,"
            "stock_nonlinear_key_source,scope,status\n");
    }
    std::printf("%s,%s,%s,%s,%s,%s,stock_trusted_dealer_test_only,"
                "known-zero-stock-nonlinear-prefix,%s\n",
                conv0_ok ? "pass" : "FAIL",
                truncation_ok ? "pass" : "FAIL",
                maxpool_ok ? "pass" : "FAIL",
                relu_ok ? "pass" : "FAIL",
                conv3_ok ? "pass" : "FAIL",
                state_link_ok ? "pass" : "FAIL",
                all_ok ? "pass" : "FAIL");
    return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    GraphArgs args;
    if (!parse_args(argc, argv, args)) {
        std::fprintf(stderr, "invalid graph-prefix arguments\n");
        return 2;
    }
    if (args.check) return run_check(args);
    if (args.csv_header) {
        std::printf(
            "party,graph_control_us,conv0_us,secure_truncate_us,maxpool_us,"
            "relu_us,conv3_us,total_protocol_bytes_sent,"
            "party_channel_bytes_sent,stock_nonlinear_bytes_sent,"
            "stock_nonlinear_bytes_received,"
            "party_channel_direction_switches,dabits,edabits,"
            "logical_opened_bits,triples,maxpool_key_bytes,relu_key_bytes,"
            "stock_nonlinear_key_bytes,status\n");
    }
    return run_party(args);
}
