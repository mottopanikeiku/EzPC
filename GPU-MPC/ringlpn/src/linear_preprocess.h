#pragma once

// Stable library facade for dealerless two-party forward-linear preprocessing.
// It excludes CLI state, CUDA/Orca implementation types, and test-only
// checkers. Public plans own no secrets. The facade retains no sampled masks,
// noise, or correlation state; the delegated backend owns those values only
// for the duration of preprocess_party. Opened records are private key material
// and are move-only; their payload storage is scrubbed when ownership ends.

#include <array>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ringlpn_linear {
using InvocationId = std::array<uint8_t, 16>;
using Digest = std::array<uint8_t, 32>;
#if defined(__GNUC__)
#define RINGLPN_LINEAR_PUBLIC __attribute__((visibility("default")))
#else
#define RINGLPN_LINEAR_PUBLIC
#endif



enum class LinearKind : uint8_t { Fc = 0, Conv2d = 1 };

enum class Status : uint8_t {
    Ok = 0,
    InvalidPlan,
    InvalidConfiguration,
    IoError,
    CorruptRecord,
    RecordMismatch,
    ProtocolFailure,
};

inline const char *status_name(Status status) noexcept {
    switch (status) {
        case Status::Ok: return "ok";
        case Status::InvalidPlan: return "invalid-plan";
        case Status::InvalidConfiguration: return "invalid-configuration";
        case Status::IoError: return "io-error";
        case Status::CorruptRecord: return "corrupt-record";
        case Status::RecordMismatch: return "record-mismatch";
        case Status::ProtocolFailure: return "protocol-failure";
    }
    return "unknown";
}

struct ProtocolParameters {
    int qbits = 64;
    int bw = 16;
    int ole_n = 8192;
    int ole_c = 2;
    int ole_t = 8;
    bool regular_noise = true;
    std::string channel = "local-loopback";
    std::string ot_backend = "sci-iknp";
};

struct FcShape {
    int rows = 0;
    int inner = 0;
    int cols = 0;
};

struct Conv2dShape {
    int n = 0;
    int h = 0;
    int w = 0;
    int ci = 0;
    int fh = 0;
    int fw = 0;
    int co = 0;
    int padding = 0;
    int stride = 1;
};

// Public, fully derived work description. It contains no masks, keys, noise,
// ledger contents, or protocol randomness.
struct LinearPlan {
    LinearKind kind = LinearKind::Fc;
    ProtocolParameters protocol;
    FcShape fc;
    Conv2dShape conv;
    uint64_t input_words = 0;
    uint64_t weight_words = 0;
    uint64_t output_words = 0;
    uint64_t cross_terms = 0;
    uint64_t ring_batches = 0;
    uint64_t ring_application_slots = 0;
    uint64_t ring_bootstrap_slots = 0;
    int output_h = 0;
    int output_w = 0;
};

// Output/state paths name private caller-owned storage and are not deleted.
// The ledger is an absolute persistent owner-only path.
struct PartyConfig {
    int party = -1;
    uint64_t sid = 0;
    uint64_t layer_ordinal = 0;
    InvocationId invocation_id{};
    std::string host = "127.0.0.1";
    int port = 48000;
    std::string ledger_path;
    // Absolute owner-private 0600 file containing exactly 32 bytes. The
    // backend consumes and unlinks it before protocol traffic.
    std::string channel_auth_file;
    std::string output_prefix;
    std::string state_record_path;
    // Locates the optional bridge only. Authorization is the immutable digest
    // compiled into the loader; this caller-supplied path conveys no trust.
    std::string emp_silent_bridge;
};

struct RecordMetadata {
    LinearKind kind = LinearKind::Fc;
    int party = -1;
    uint64_t sid = 0;
    InvocationId invocation_id{};
    Digest ledger_digest{};
    // Nonzero only for EMP-Silent and binds the exact authorized immutable
    // bridge image recorded by the producer.
    Digest emp_silent_bridge_digest{};
    // channel is empty because the v3 on-disk record does not bind transport.
    ProtocolParameters protocol;
    FcShape fc;
    Conv2dShape conv;
    uint64_t ring_batches = 0;
    uint64_t payload_words = 0;
    Digest digest{};
};

struct WordView {
    const uint64_t *data = nullptr;
    size_t size = 0;

    const uint64_t *begin() const noexcept { return data; }
    const uint64_t *end() const noexcept {
        return size == 0 ? data : data + size;
    }
};

class OwnedRecord {
  public:
    OwnedRecord() = default;
    ~OwnedRecord() { reset(); }
    OwnedRecord(const OwnedRecord &) = delete;
    OwnedRecord &operator=(const OwnedRecord &) = delete;

    OwnedRecord(OwnedRecord &&other) noexcept { move_from(std::move(other)); }
    OwnedRecord &operator=(OwnedRecord &&other) noexcept {
        if (this != &other) {
            reset();
            move_from(std::move(other));
        }
        return *this;
    }

    const RecordMetadata &metadata() const noexcept { return metadata_; }
    const LinearPlan &plan() const noexcept { return plan_; }
    const uint64_t *payload() const noexcept { return payload_.data(); }
    size_t payload_words() const noexcept { return payload_.size(); }
    bool empty() const noexcept { return payload_.empty(); }
    WordView input_mask() const noexcept {
        return {payload_.empty() ? nullptr : payload_.data(),
                static_cast<size_t>(plan_.input_words)};
    }
    WordView weight_mask() const noexcept {
        return {payload_.empty() ? nullptr
                                 : payload_.data() + plan_.input_words,
                static_cast<size_t>(plan_.weight_words)};
    }
    WordView output_correction() const noexcept {
        return {payload_.empty()
                    ? nullptr
                    : payload_.data() + plan_.input_words +
                          plan_.weight_words,
                static_cast<size_t>(plan_.output_words)};
    }

    void reset() noexcept {
        // A volatile byte walk prevents dead-store elimination of the scrub.
        volatile unsigned char *bytes =
            reinterpret_cast<volatile unsigned char *>(payload_.data());
        for (size_t i = 0; i < payload_.size() * sizeof(uint64_t); ++i) {
            bytes[i] = 0;
        }
        payload_.clear();
        metadata_ = RecordMetadata{};
        plan_ = LinearPlan{};
    }

  private:
    friend class FcPreprocessor;
    friend class Conv2dPreprocessor;

    void adopt(RecordMetadata metadata, LinearPlan plan,
               std::vector<uint64_t> payload) noexcept {
        reset();
        metadata_ = std::move(metadata);
        plan_ = std::move(plan);
        payload_ = std::move(payload);
    }

    void move_from(OwnedRecord &&other) noexcept {
        metadata_ = std::move(other.metadata_);
        plan_ = std::move(other.plan_);
        payload_ = std::move(other.payload_);
        other.metadata_ = RecordMetadata{};
        other.plan_ = LinearPlan{};
    }

    RecordMetadata metadata_{};
    LinearPlan plan_{};
    std::vector<uint64_t> payload_;
};

struct RecordExpectation {
    LinearPlan plan;
    int party = -1;
    uint64_t sid = 0;
    InvocationId invocation_id{};
    bool require_invocation = false;
};


class RINGLPN_LINEAR_PUBLIC FcPreprocessor {
  public:
    static Status plan(const FcShape &shape,
                       const ProtocolParameters &protocol,
                       LinearPlan &out) noexcept;
    // Once configuration and plan validation succeed, the invocation namespace
    // is consume-once: callers MUST treat it as spent even if this returns
    // ProtocolFailure. A retry requires a fresh invocation ID.
    static Status preprocess_party(const LinearPlan &plan,
                                   const PartyConfig &config) noexcept;
    static Status initialize_gpu() noexcept;
    static Status open_record(const std::string &path,
                              OwnedRecord &out) noexcept;
    static Status validate_record(const OwnedRecord &record,
                                  const RecordExpectation &expected) noexcept;
    static Status open_and_validate_record(const std::string &path,
                                           const RecordExpectation &expected,
                                           OwnedRecord &out) noexcept;

    // Compatibility entrypoint for the canonical executable. Reusable callers
    // should use the typed methods above.
    static int run_cli(int argc, char **argv);
};

class RINGLPN_LINEAR_PUBLIC Conv2dPreprocessor {
  public:
    static Status plan(const Conv2dShape &shape,
                       const ProtocolParameters &protocol,
                       LinearPlan &out) noexcept;
    // Once configuration and plan validation succeed, the invocation namespace
    // is consume-once: callers MUST treat it as spent even if this returns
    // ProtocolFailure. A retry requires a fresh invocation ID.
    static Status preprocess_party(const LinearPlan &plan,
                                   const PartyConfig &config) noexcept;
    static Status initialize_gpu() noexcept;
    static Status open_record(const std::string &path,
                              OwnedRecord &out) noexcept;
    static Status validate_record(const OwnedRecord &record,
                                  const RecordExpectation &expected) noexcept;
    static Status open_and_validate_record(const std::string &path,
                                           const RecordExpectation &expected,
                                           OwnedRecord &out) noexcept;
    static int run_cli(int argc, char **argv);
};

}  // namespace ringlpn_linear
#undef RINGLPN_LINEAR_PUBLIC
