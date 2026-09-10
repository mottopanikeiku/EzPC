#pragma once

// Private implementation generator. Include once from each shape-specific CUDA
// translation unit after selecting RINGLPN_LINEAR_BACKEND_CLASS and, for
// Conv2D, RINGLPN_LIVE_CONV. Public callers include linear_preprocess.h only.

#include "linear_preprocess.h"
#include "graph_mask_state.h"
#include "two_party_linear_preprocess.cuh"

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef RINGLPN_LINEAR_BACKEND_CLASS
#error "RINGLPN_LINEAR_BACKEND_CLASS must name the facade class"
#endif
#ifdef RINGLPN_LIVE_CONV
#define RINGLPN_LINEAR_DETAIL_NAMESPACE linear_preprocess_conv_detail
#else
#define RINGLPN_LINEAR_DETAIL_NAMESPACE linear_preprocess_fc_detail
#endif


namespace ringlpn_linear {
namespace RINGLPN_LINEAR_DETAIL_NAMESPACE {

#ifdef RINGLPN_LIVE_CONV
namespace impl = ringlpn_conv_live;
constexpr LinearKind kKind = LinearKind::Conv2d;
#else
namespace impl = ringlpn_fc_live;
constexpr LinearKind kKind = LinearKind::Fc;
#endif

inline impl::Args public_args(const ProtocolParameters &protocol) {
    impl::Args args;
    args.party = 0;
    args.sid = 1;
    args.qbits = protocol.qbits;
    args.bw = protocol.bw;
    args.ole_n = protocol.ole_n;
    args.ole_c = protocol.ole_c;
    args.ole_t = protocol.ole_t;
    args.noise = protocol.regular_noise ? "regular" : "uniform";
    args.channel = protocol.channel;
    args.ot_backend = protocol.ot_backend;
    return args;
}

inline void set_shape(impl::Args &args, const LinearPlan &plan) {
#ifdef RINGLPN_LIVE_CONV
    args.rows = plan.conv.n;
    args.inner = plan.conv.h;
    args.cols = plan.conv.w;
    args.ci = plan.conv.ci;
    args.fh = plan.conv.fh;
    args.fw = plan.conv.fw;
    args.co = plan.conv.co;
    args.padding = plan.conv.padding;
    args.stride = plan.conv.stride;
#else
    args.rows = plan.fc.rows;
    args.inner = plan.fc.inner;
    args.cols = plan.fc.cols;
#endif
}

inline void set_shape(impl::Args &args,
#ifdef RINGLPN_LIVE_CONV
                      const Conv2dShape &shape
#else
                      const FcShape &shape
#endif
) {
#ifdef RINGLPN_LIVE_CONV
    args.rows = shape.n;
    args.inner = shape.h;
    args.cols = shape.w;
    args.ci = shape.ci;
    args.fh = shape.fh;
    args.fw = shape.fw;
    args.co = shape.co;
    args.padding = shape.padding;
    args.stride = shape.stride;
#else
    args.rows = shape.rows;
    args.inner = shape.inner;
    args.cols = shape.cols;
#endif
}

inline LinearPlan export_plan(const impl::Args &args,
                              const impl::PublicWork &work) {
    LinearPlan plan;
    plan.kind = kKind;
    plan.protocol.qbits = args.qbits;
    plan.protocol.bw = args.bw;
    plan.protocol.ole_n = args.ole_n;
    plan.protocol.ole_c = args.ole_c;
    plan.protocol.ole_t = args.ole_t;
    plan.protocol.regular_noise = args.noise == "regular";
    plan.protocol.channel = args.channel;
    plan.protocol.ot_backend = args.ot_backend;
#ifdef RINGLPN_LIVE_CONV
    plan.conv = {args.rows, args.inner, args.cols, args.ci, args.fh,
                 args.fw, args.co, args.padding, args.stride};
    plan.output_h = work.conv.p.OH;
    plan.output_w = work.conv.p.OW;
#else
    plan.fc = {args.rows, args.inner, args.cols};
#endif
    plan.input_words = work.size_a;
    plan.weight_words = work.size_b;
    plan.output_words = work.size_c;
    plan.cross_terms = work.cross_terms;
    plan.ring_batches = work.ring_batches;
    plan.ring_application_slots = work.ring_application_slots;
    plan.ring_bootstrap_slots = work.ring_bootstrap_slots;
    return plan;
}

inline RecordMetadata export_metadata(const impl::Record &record) {
    const impl::RecordHeader &header = record.header;
    RecordMetadata metadata;
    metadata.kind = kKind;
    metadata.party = header.party;
    metadata.sid = header.sid;
    metadata.invocation_id = header.invocation_id;
    metadata.ledger_digest = header.ledger_digest;
    metadata.emp_silent_bridge_digest =
        header.emp_silent_bridge_digest;
    metadata.protocol.qbits = header.qbits;
    metadata.protocol.bw = header.bw;
    metadata.protocol.ole_n = header.ole_n;
    metadata.protocol.ole_c = header.ole_c;
    metadata.protocol.ole_t = header.ole_t;
    metadata.protocol.regular_noise = header.regular;
    metadata.protocol.ot_backend = ringlpn_2pc::ot_backend_name(header.ot_backend);
    // The v3 record does not bind a transport channel.
    metadata.protocol.channel.clear();
#ifdef RINGLPN_LIVE_CONV
    metadata.conv = {header.rows, header.inner, header.cols, header.ci,
                     header.fh, header.fw, header.co, header.padding,
                     header.stride};
#else
    metadata.fc = {header.rows, header.inner, header.cols};
#endif
    metadata.ring_batches = header.ring_batches;
    metadata.payload_words = header.payload_words;
    metadata.digest = record.digest;
    return metadata;
}

inline bool same_protocol(const ProtocolParameters &a,
                          const ProtocolParameters &b) {
    return a.qbits == b.qbits && a.bw == b.bw && a.ole_n == b.ole_n &&
           a.ole_c == b.ole_c && a.ole_t == b.ole_t &&
           a.regular_noise == b.regular_noise && a.channel == b.channel &&
           a.ot_backend == b.ot_backend;
}

inline bool same_plan(const LinearPlan &a, const LinearPlan &b) {
    if (a.kind != b.kind || !same_protocol(a.protocol, b.protocol) ||
        a.input_words != b.input_words || a.weight_words != b.weight_words ||
        a.output_words != b.output_words || a.cross_terms != b.cross_terms ||
        a.ring_batches != b.ring_batches ||
        a.ring_application_slots != b.ring_application_slots ||
        a.ring_bootstrap_slots != b.ring_bootstrap_slots ||
        a.output_h != b.output_h || a.output_w != b.output_w) {
        return false;
    }
    if (a.kind == LinearKind::Fc) {
        return a.fc.rows == b.fc.rows && a.fc.inner == b.fc.inner &&
               a.fc.cols == b.fc.cols;
    }
    return a.conv.n == b.conv.n && a.conv.h == b.conv.h &&
           a.conv.w == b.conv.w && a.conv.ci == b.conv.ci &&
           a.conv.fh == b.conv.fh && a.conv.fw == b.conv.fw &&
           a.conv.co == b.conv.co && a.conv.padding == b.conv.padding &&
           a.conv.stride == b.conv.stride;
}
inline bool same_record_plan(const LinearPlan &actual,
                             const LinearPlan &expected) {
    LinearPlan normalized = expected;
    // The transport channel is a runtime choice and is not encoded in v3.
    normalized.protocol.channel = actual.protocol.channel;
    return same_plan(actual, normalized);
}


inline Status plan_from_record(const impl::Record &record, LinearPlan &plan) {
    ProtocolParameters protocol;
    protocol.qbits = record.header.qbits;
    protocol.bw = record.header.bw;
    protocol.ole_n = record.header.ole_n;
    protocol.ole_c = record.header.ole_c;
    protocol.ole_t = record.header.ole_t;
    protocol.regular_noise = record.header.regular;
    protocol.ot_backend =
        ringlpn_2pc::ot_backend_name(record.header.ot_backend);
    impl::Args args = public_args(protocol);
    args.sid = record.header.sid;
#ifdef RINGLPN_LIVE_CONV
    args.rows = record.header.rows;
    args.inner = record.header.inner;
    args.cols = record.header.cols;
    args.ci = record.header.ci;
    args.fh = record.header.fh;
    args.fw = record.header.fw;
    args.co = record.header.co;
    args.padding = record.header.padding;
    args.stride = record.header.stride;
#else
    args.rows = record.header.rows;
    args.inner = record.header.inner;
    args.cols = record.header.cols;
#endif
    impl::PublicWork work;
    if (!impl::derive_work(args, work) ||
        work.ring_batches != record.header.ring_batches ||
        record.header.payload_words !=
            work.size_a + work.size_b + work.size_c) {
        return Status::RecordMismatch;
    }
    plan = export_plan(args, work);
    return Status::Ok;
}
inline bool record_matches(const OwnedRecord &record,
                           const RecordExpectation &expected) {
    const RecordMetadata &metadata = record.metadata();
    const LinearPlan &plan = expected.plan;
    const bool protocol_matches =
        metadata.protocol.qbits == plan.protocol.qbits &&
        metadata.protocol.bw == plan.protocol.bw &&
        metadata.protocol.ole_n == plan.protocol.ole_n &&
        metadata.protocol.ole_c == plan.protocol.ole_c &&
        metadata.protocol.ole_t == plan.protocol.ole_t &&
        metadata.protocol.regular_noise == plan.protocol.regular_noise &&
        metadata.protocol.ot_backend == plan.protocol.ot_backend;
    if (record.empty() || metadata.kind != kKind || plan.kind != kKind ||
        !same_record_plan(record.plan(), plan) || !protocol_matches ||
        metadata.party != expected.party ||
        (expected.sid != 0 && metadata.sid != expected.sid) ||
        (expected.require_invocation &&
         metadata.invocation_id != expected.invocation_id) ||
        metadata.ring_batches != plan.ring_batches ||
        metadata.payload_words !=
            plan.input_words + plan.weight_words + plan.output_words ||
        record.payload_words() != metadata.payload_words) {
        return false;
    }
#ifdef RINGLPN_LIVE_CONV
    return metadata.conv.n == plan.conv.n &&
           metadata.conv.h == plan.conv.h &&
           metadata.conv.w == plan.conv.w &&
           metadata.conv.ci == plan.conv.ci &&
           metadata.conv.fh == plan.conv.fh &&
           metadata.conv.fw == plan.conv.fw &&
           metadata.conv.co == plan.conv.co &&
           metadata.conv.padding == plan.conv.padding &&
           metadata.conv.stride == plan.conv.stride;
#else
    return metadata.fc.rows == plan.fc.rows &&
           metadata.fc.inner == plan.fc.inner &&
           metadata.fc.cols == plan.fc.cols;
#endif
}
class ScopedFd {
  public:
    explicit ScopedFd(int value) : value_(value) {}
    ~ScopedFd() {
        if (value_ >= 0) ::close(value_);
    }
    int get() const { return value_; }

  private:
    int value_;
};

class ScopedBytes {
  public:
    explicit ScopedBytes(size_t size = 0) : value(size) {}
    ~ScopedBytes() {
        volatile uint8_t *bytes =
            reinterpret_cast<volatile uint8_t *>(value.data());
        for (size_t i = 0; i < value.size(); ++i) bytes[i] = 0;
    }
    std::vector<uint8_t> value;
};

inline Status read_private_bytes(const std::string &path, size_t minimum,
                                 size_t maximum, ScopedBytes &bytes) {
    const int raw_fd =
        ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (raw_fd < 0) return Status::IoError;
    ScopedFd fd(raw_fd);
    struct stat before {};
    if (::fstat(fd.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
        before.st_uid != ::geteuid() || before.st_nlink != 1 ||
        (before.st_mode & 07777) != 0600 || before.st_size < 0) {
        return Status::IoError;
    }
    const uint64_t file_size = static_cast<uint64_t>(before.st_size);
    if (file_size < minimum || file_size > maximum ||
        file_size > std::numeric_limits<size_t>::max()) {
        return Status::CorruptRecord;
    }
    bytes.value.assign(static_cast<size_t>(file_size), 0);
    size_t cursor = 0;
    while (cursor < bytes.value.size()) {
        const ssize_t got =
            ::read(fd.get(), bytes.value.data() + cursor,
                   bytes.value.size() - cursor);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return Status::IoError;
        cursor += static_cast<size_t>(got);
    }
    uint8_t extra = 0;
    ssize_t trailing = -1;
    do {
        trailing = ::read(fd.get(), &extra, 1);
    } while (trailing < 0 && errno == EINTR);
    struct stat after {};
    if (trailing != 0 || ::fstat(fd.get(), &after) != 0 ||
        before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
        before.st_size != after.st_size || before.st_uid != after.st_uid ||
        before.st_mode != after.st_mode || before.st_nlink != after.st_nlink ||
        before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
        before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
        before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
        before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
        return Status::IoError;
    }
    return Status::Ok;
}

inline Status read_private_record(const std::string &path,
                                  impl::Record &record) {
    if (path.empty()) return Status::InvalidConfiguration;
    ScopedBytes private_bytes;
    Status status = read_private_bytes(
        path, impl::kRecordHeaderBytes + impl::kDigestBytes,
        impl::kMaxRecordBytes, private_bytes);
    if (status != Status::Ok) return status;

#ifdef RINGLPN_LIVE_CONV
    constexpr std::array<uint8_t, 8> kOtherKindMagic = {
        'R', 'L', 'P', 'N', 'F', 'C', '2', 'P'};
#else
    constexpr std::array<uint8_t, 8> kOtherKindMagic = {
        'R', 'L', 'P', 'N', 'C', 'V', '2', 'P'};
#endif
    if (std::equal(kOtherKindMagic.begin(), kOtherKindMagic.end(),
                   private_bytes.value.begin())) {
        return Status::RecordMismatch;
    }

    impl::RecordHeader header;
    if (!impl::decode_header(private_bytes.value.data(),
                             private_bytes.value.size(), header)) {
        return Status::CorruptRecord;
    }
    uint64_t payload_bytes = 0;
    if (!impl::checked_mul(header.payload_words, sizeof(uint64_t),
                           payload_bytes) ||
        payload_bytes > impl::kMaxRecordBytes ||
        impl::kRecordHeaderBytes + payload_bytes + impl::kDigestBytes !=
            private_bytes.value.size()) {
        return Status::CorruptRecord;
    }
    ringlpn_freshness::Digest expected{};
    if (!impl::sha256(private_bytes.value.data(),
                      impl::kRecordHeaderBytes +
                          static_cast<size_t>(payload_bytes),
                      expected) ||
        !std::equal(expected.begin(), expected.end(),
                    private_bytes.value.begin() +
                        impl::kRecordHeaderBytes + payload_bytes)) {
        return Status::CorruptRecord;
    }
    impl::Record parsed;
    parsed.header = header;
    parsed.digest = expected;
    parsed.payload.resize(static_cast<size_t>(header.payload_words));
    for (size_t i = 0; i < parsed.payload.size(); ++i) {
        parsed.payload[i] = static_cast<uint64_t>(impl::get_u64(
            private_bytes.value.data(),
            impl::kRecordHeaderBytes + i * sizeof(uint64_t)));
    }
    const uint64_t limit = uint64_t{1} << header.bw;
    if (!std::all_of(parsed.payload.begin(), parsed.payload.end(),
                     [limit](uint64_t word) { return word < limit; })) {
        volatile unsigned char *bytes =
            reinterpret_cast<volatile unsigned char *>(
                parsed.payload.data());
        for (size_t i = 0;
             i < parsed.payload.size() * sizeof(uint64_t); ++i) {
            bytes[i] = 0;
        }
        return Status::CorruptRecord;
    }
    record = std::move(parsed);
    return Status::Ok;
}

inline bool valid_invocation(
    const ringlpn_freshness::InvocationId &invocation) {
    return std::any_of(invocation.begin(), invocation.end(),
                       [](uint8_t byte) { return byte != 0; });
}
class ScopedPrivateRecord {
  public:
    impl::Record record;

    ScopedPrivateRecord() = default;
    ScopedPrivateRecord(const ScopedPrivateRecord &) = delete;
    ScopedPrivateRecord &operator=(const ScopedPrivateRecord &) = delete;

    ~ScopedPrivateRecord() {
        volatile unsigned char *bytes =
            reinterpret_cast<volatile unsigned char *>(record.payload.data());
        for (size_t i = 0; i < record.payload.size() * sizeof(uint64_t); ++i) {
            bytes[i] = 0;
        }
    }
};

inline void scrub_words(std::vector<uint64_t> &words) noexcept {
    volatile unsigned char *bytes =
        reinterpret_cast<volatile unsigned char *>(words.data());
    for (size_t i = 0; i < words.size() * sizeof(uint64_t); ++i) {
        bytes[i] = 0;
    }
    words.clear();
}

class ScopedPrivateState {
  public:
    ringlpn_graph::MaskStateRecord<uint64_t> record;

    ScopedPrivateState() = default;
    ScopedPrivateState(const ScopedPrivateState &) = delete;
    ScopedPrivateState &operator=(const ScopedPrivateState &) = delete;

    ~ScopedPrivateState() {
        scrub_words(record.input_mask_share);
        scrub_words(record.output_mask_share);
    }
};

inline Status read_private_state(
    const std::string &path,
    ringlpn_graph::MaskStateRecord<uint64_t> &record) {
    if (path.empty()) return Status::IoError;
    ScopedBytes private_bytes;
    Status status = read_private_bytes(
        path,
        ringlpn_graph::kMaskStateHeaderBytes +
            ringlpn_graph::kMaskStateDigestBytes,
        ringlpn_graph::kMaxMaskStateBytes, private_bytes);
    if (status != Status::Ok) return status;
    ringlpn_graph::MaskStateHeader header;
    if (!ringlpn_graph::decode_mask_state_header(
            private_bytes.value.data(), private_bytes.value.size(), header) ||
        header.input_words > std::numeric_limits<size_t>::max() ||
        header.output_words > std::numeric_limits<size_t>::max()) {
        return Status::CorruptRecord;
    }
    const size_t input_words = static_cast<size_t>(header.input_words);
    const size_t output_words = static_cast<size_t>(header.output_words);
    if (input_words >
            (ringlpn_graph::kMaxMaskStateBytes -
             ringlpn_graph::kMaskStateHeaderBytes -
             ringlpn_graph::kMaskStateDigestBytes) /
                sizeof(uint64_t) ||
        output_words >
            (ringlpn_graph::kMaxMaskStateBytes -
             ringlpn_graph::kMaskStateHeaderBytes -
             ringlpn_graph::kMaskStateDigestBytes -
             input_words * sizeof(uint64_t)) /
                sizeof(uint64_t) ||
        ringlpn_graph::kMaskStateHeaderBytes +
                (input_words + output_words) * sizeof(uint64_t) +
                ringlpn_graph::kMaskStateDigestBytes !=
            private_bytes.value.size()) {
        return Status::CorruptRecord;
    }
    size_t cursor = ringlpn_graph::kMaskStateHeaderBytes;
    const uint64_t input_limit = uint64_t{1} << header.input_bw;
    for (size_t i = 0; i < input_words; ++i, cursor += sizeof(uint64_t)) {
        if (ringlpn_graph::get_u64(private_bytes.value.data(), cursor) >=
            input_limit) {
            return Status::CorruptRecord;
        }
    }
    const uint64_t output_limit = uint64_t{1} << header.output_bw;
    for (size_t i = 0; i < output_words; ++i, cursor += sizeof(uint64_t)) {
        if (ringlpn_graph::get_u64(private_bytes.value.data(), cursor) >=
            output_limit) {
            return Status::CorruptRecord;
        }
    }
    ringlpn_graph::MaskStateRecord<uint64_t> parsed;
    if (!ringlpn_graph::parse_mask_state_bytes(private_bytes.value, parsed)) {
        scrub_words(parsed.input_mask_share);
        scrub_words(parsed.output_mask_share);
        return Status::CorruptRecord;
    }
    record = std::move(parsed);
    scrub_words(parsed.input_mask_share);
    scrub_words(parsed.output_mask_share);
    return Status::Ok;
}

inline bool derive_layer_identity(const LinearPlan &plan,
                                  uint64_t layer_ordinal,
                                  Digest &identity) {
    std::vector<uint8_t> layer;
    static constexpr uint8_t kLayerDomain[] =
        "RINGLPN-FC-CONV-LAYER-IDENTITY-v1";
    layer.insert(layer.end(), kLayerDomain,
                 kLayerDomain + sizeof(kLayerDomain) - 1);
    ringlpn_freshness::put_u32_be(
        layer, ringlpn_freshness::kProtocolVersion);
    ringlpn_freshness::put_u32_be(
        layer, plan.kind == LinearKind::Fc ? 1U : 2U);
    ringlpn_freshness::put_u64_be(layer, layer_ordinal);
    ringlpn_freshness::put_u64_be(
        layer, static_cast<uint64_t>(plan.protocol.qbits));
    ringlpn_freshness::put_u64_be(
        layer, static_cast<uint64_t>(plan.protocol.bw));
    if (plan.kind == LinearKind::Fc) {
        ringlpn_freshness::put_u64_be(
            layer, static_cast<uint64_t>(plan.fc.rows));
        ringlpn_freshness::put_u64_be(
            layer, static_cast<uint64_t>(plan.fc.inner));
        ringlpn_freshness::put_u64_be(
            layer, static_cast<uint64_t>(plan.fc.cols));
    } else {
        ringlpn_freshness::put_u64_be(
            layer, static_cast<uint64_t>(plan.conv.n));
        ringlpn_freshness::put_u64_be(
            layer, static_cast<uint64_t>(plan.conv.h));
        ringlpn_freshness::put_u64_be(
            layer, static_cast<uint64_t>(plan.conv.w));
        ringlpn_freshness::put_u64_be(
            layer, static_cast<uint64_t>(plan.conv.ci));
        ringlpn_freshness::put_u64_be(
            layer, static_cast<uint64_t>(plan.conv.fh));
        ringlpn_freshness::put_u64_be(
            layer, static_cast<uint64_t>(plan.conv.fw));
        ringlpn_freshness::put_u64_be(
            layer, static_cast<uint64_t>(plan.conv.co));
        ringlpn_freshness::put_u64_be(
            layer, static_cast<uint64_t>(plan.conv.padding));
        ringlpn_freshness::put_u64_be(
            layer, static_cast<uint64_t>(plan.conv.stride));
    }
    ringlpn_freshness::put_u64_be(
        layer, static_cast<uint64_t>(plan.protocol.ole_n));
    ringlpn_freshness::put_u64_be(
        layer, static_cast<uint64_t>(plan.protocol.ole_c));
    ringlpn_freshness::put_u64_be(
        layer, static_cast<uint64_t>(plan.protocol.ole_t));
    ringlpn_freshness::put_u64_be(
        layer, plan.protocol.regular_noise ? 1U : 0U);
    return ringlpn_freshness::digest(layer.data(), layer.size(), identity);
}


}  // namespace RINGLPN_LINEAR_DETAIL_NAMESPACE

#ifdef RINGLPN_LIVE_CONV
Status RINGLPN_LINEAR_BACKEND_CLASS::plan(
    const Conv2dShape &shape, const ProtocolParameters &protocol,
    LinearPlan &out) noexcept {
#else
Status RINGLPN_LINEAR_BACKEND_CLASS::plan(
    const FcShape &shape, const ProtocolParameters &protocol,
    LinearPlan &out) noexcept {
#endif
    try {
        using namespace RINGLPN_LINEAR_DETAIL_NAMESPACE;
        impl::Args args = public_args(protocol);
        set_shape(args, shape);
        impl::PublicWork work;
        if (!impl::derive_work(args, work)) return Status::InvalidPlan;
        out = export_plan(args, work);
        return Status::Ok;
    } catch (...) {
        return Status::InvalidPlan;
    }
}

Status RINGLPN_LINEAR_BACKEND_CLASS::preprocess_party(
    const LinearPlan &plan, const PartyConfig &config) noexcept {
    try {
        using namespace RINGLPN_LINEAR_DETAIL_NAMESPACE;
        const bool backend_config_ok =
            (plan.protocol.ot_backend == "sci-iknp" &&
             config.emp_silent_bridge.empty()) ||
            (plan.protocol.ot_backend == "emp-silent" &&
             !config.emp_silent_bridge.empty());
        if (plan.kind != kKind || !backend_config_ok ||
            (config.party != 0 && config.party != 1) || config.sid == 0 ||
            !valid_invocation(config.invocation_id) || config.host.empty() ||
            config.ledger_path.empty() || config.ledger_path.front() != '/' ||
            config.channel_auth_file.empty() ||
            config.channel_auth_file.front() != '/' ||
            config.output_prefix.empty() ||
            (!config.state_record_path.empty() &&
             config.layer_ordinal == 0) ||
            config.port <= 0 || config.port >= 65535) {
            return Status::InvalidConfiguration;
        }
        impl::Args args = public_args(plan.protocol);
        set_shape(args, plan);
        impl::PublicWork derived;
        if (!impl::derive_work(args, derived) ||
            !same_plan(plan, export_plan(args, derived))) {
            return Status::InvalidPlan;
        }
        args.channel_auth_file = config.channel_auth_file;
        args.party = config.party;
        args.sid = config.sid;
        args.layer_ordinal = config.layer_ordinal;
        args.invocation_id = config.invocation_id;
        args.invocation_id_text = ringlpn_freshness::hex(config.invocation_id);
        args.host = config.host;
        args.port = config.port;
        args.ledger_path = config.ledger_path;
        args.out_prefix = config.output_prefix;
        args.state_record = config.state_record_path;
        args.emp_silent_bridge = config.emp_silent_bridge;
        return impl::run_party(args) == 0 ? Status::Ok
                                          : Status::ProtocolFailure;
    } catch (...) {
        return Status::ProtocolFailure;
    }
}
Status RINGLPN_LINEAR_BACKEND_CLASS::initialize_gpu() noexcept {
    try {
        RINGLPN_LINEAR_DETAIL_NAMESPACE::impl::init_ringlpn_gpu_pool();
        return Status::Ok;
    } catch (...) {
        return Status::ProtocolFailure;
    }
}


Status RINGLPN_LINEAR_BACKEND_CLASS::open_record(
    const std::string &path, OwnedRecord &out) noexcept {
    try {
        using namespace RINGLPN_LINEAR_DETAIL_NAMESPACE;
        ScopedPrivateRecord private_record;
        impl::Record &record = private_record.record;
        Status status = read_private_record(path, record);
        if (status != Status::Ok) return status;
        LinearPlan plan;
        status = plan_from_record(record, plan);
        if (status != Status::Ok) return status;
        out.adopt(export_metadata(record), std::move(plan),
                  std::move(record.payload));
        return Status::Ok;
    } catch (...) {
        return Status::IoError;
    }
}
Status RINGLPN_LINEAR_BACKEND_CLASS::validate_record(
    const OwnedRecord &record, const RecordExpectation &expected) noexcept {
    try {
        return RINGLPN_LINEAR_DETAIL_NAMESPACE::record_matches(record, expected)
                   ? Status::Ok
                   : Status::RecordMismatch;
    } catch (...) {
        return Status::RecordMismatch;
    }
}


Status RINGLPN_LINEAR_BACKEND_CLASS::open_and_validate_record(
    const std::string &path, const RecordExpectation &expected,
    OwnedRecord &out) noexcept {
    try {
        using namespace RINGLPN_LINEAR_DETAIL_NAMESPACE;
        ScopedPrivateRecord private_record;
        impl::Record &record = private_record.record;
        Status status = read_private_record(path, record);
        if (status != Status::Ok) return status;
        LinearPlan actual;
        status = plan_from_record(record, actual);
        if (status != Status::Ok ||
            !same_record_plan(actual, expected.plan) ||
            record.header.party != expected.party ||
            (expected.sid != 0 && record.header.sid != expected.sid) ||
            (expected.require_invocation &&
             record.header.invocation_id != expected.invocation_id)) {
            return Status::RecordMismatch;
        }
        out.adopt(export_metadata(record), std::move(actual),
                  std::move(record.payload));
        return Status::Ok;
    } catch (...) {
        return Status::IoError;
    }
}

Status RINGLPN_LINEAR_BACKEND_CLASS::open_and_validate_layer_material(
    const std::string &record_path, const std::string &state_path,
    const LayerExpectation &expected, OwnedLayerMaterial &out) noexcept {
    out.reset();
    try {
        using namespace RINGLPN_LINEAR_DETAIL_NAMESPACE;
        if (record_path.empty() || state_path.empty()) {
            return Status::IoError;
        }

        ScopedPrivateRecord private_record;
        impl::Record &record = private_record.record;
        Status status = read_private_record(record_path, record);
        if (status != Status::Ok) return status;

        LinearPlan actual;
        status = plan_from_record(record, actual);
        if (status != Status::Ok) return Status::CorruptRecord;
        if (expected.plan.kind != kKind ||
            !same_record_plan(actual, expected.plan) ||
            record.header.party != expected.party ||
            record.header.sid != expected.sid ||
            (expected.require_invocation &&
             record.header.invocation_id != expected.invocation_id) ||
            (expected.require_digests &&
             record.digest != expected.record_digest)) {
            return Status::RecordMismatch;
        }

        ScopedPrivateState private_state;
        auto &state = private_state.record;
        status = read_private_state(state_path, state);
        if (status != Status::Ok) return status;

        Digest derived_identity{};
        const size_t input_words = static_cast<size_t>(actual.input_words);
        const size_t output_words = static_cast<size_t>(actual.output_words);
        if (state.header.party != record.header.party ||
            state.header.sid != record.header.sid ||
            state.header.layer_ordinal != expected.layer_ordinal ||
            state.header.input_bw != actual.protocol.bw ||
            state.header.output_bw != actual.protocol.bw ||
            state.header.input_words != actual.input_words ||
            state.header.output_words != actual.output_words ||
            state.header.invocation_id != record.header.invocation_id ||
            state.header.linear_record_digest != record.digest ||
            (expected.require_digests &&
             state.digest != expected.state_digest) ||
            record.payload.size() !=
                input_words + static_cast<size_t>(actual.weight_words) +
                    output_words ||
            state.input_mask_share.size() != input_words ||
            state.output_mask_share.size() != output_words ||
            !derive_layer_identity(actual, state.header.layer_ordinal,
                                   derived_identity) ||
            state.header.layer_identity != derived_identity ||
            !std::equal(state.input_mask_share.begin(),
                        state.input_mask_share.end(),
                        record.payload.begin())) {
            return Status::RecordMismatch;
        }

        out.adopt(export_metadata(record), std::move(actual),
                  state.header.layer_ordinal, state.header.layer_identity,
                  state.digest, std::move(record.payload),
                  std::move(state.input_mask_share),
                  std::move(state.output_mask_share));
        return Status::Ok;
    } catch (...) {
        out.reset();
        return Status::IoError;
    }
}


int RINGLPN_LINEAR_BACKEND_CLASS::run_cli(int argc, char **argv) {
    try {
        using namespace RINGLPN_LINEAR_DETAIL_NAMESPACE;
        impl::Args args;
        if (!impl::public_a_validation_gate()) return 2;
        if (!impl::parse_args(argc, argv, args)) {
            impl::usage(argv[0]);
            return 2;
        }
        if (args.plan) return impl::run_plan(args);
        return args.check ? impl::run_check(args) : impl::run_party(args);
    } catch (...) {
        std::fprintf(stderr, "[two-party-linear] authenticated execution rejected\n");
        return 2;
    }
}

}  // namespace ringlpn_linear

#undef RINGLPN_LINEAR_BACKEND_CLASS
#undef RINGLPN_LINEAR_DETAIL_NAMESPACE
