#pragma once

#include "emp_silent_bridge.h"
#include "utils/net_io_channel.h"

#include <dlfcn.h>
#include <openssl/evp.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ringlpn_2pc {

enum class OtBackend { SciIknp, EmpSilent };

inline const char *ot_backend_name(OtBackend backend) {
  return backend == OtBackend::SciIknp ? "sci-iknp" : "emp-silent";
}

inline bool parse_ot_backend(const std::string &name, OtBackend &out) {
  if (name == "sci-iknp") {
    out = OtBackend::SciIknp;
    return true;
  }
  if (name == "emp-silent") {
    out = OtBackend::EmpSilent;
    return true;
  }
  return false;
}

struct EmpSilentPlan {
  std::string bridge_library;
  std::array<uint8_t, 32> public_manifest_digest{};
  uint64_t straight_count = 0;
  uint64_t reversed_count = 0;
  uint32_t threads = 1;
};

struct EmpSilentMetrics {
  ringlpn_emp_counters straight{};
  ringlpn_emp_counters reversed{};
};

class EmpSilentApi {
 public:
  explicit EmpSilentApi(const std::string &path) {
    if (path.empty())
      throw std::invalid_argument(
          "emp-silent requires an explicit bridge library path");
    library_ = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr)
      throw std::runtime_error(std::string("failed to load EMP bridge: ") +
                               ::dlerror());
    load(abi_version_, "ringlpn_emp_silent_abi_version");
    load(revision_, "ringlpn_emp_silent_revision");
    load(create_, "ringlpn_emp_silent_create");
    load(begin_, "ringlpn_emp_silent_begin");
    load(send_, "ringlpn_emp_silent_send");
    load(recv_, "ringlpn_emp_silent_recv");
    load(end_, "ringlpn_emp_silent_end");
    load(counters_, "ringlpn_emp_silent_get_counters");
    load(destroy_, "ringlpn_emp_silent_destroy");
    if (abi_version_() != RINGLPN_EMP_SILENT_ABI_VERSION)
      throw std::runtime_error("loaded EMP bridge ABI version is not pinned");
    const char *revision = revision_();
    if (revision == nullptr ||
        std::strcmp(revision, RINGLPN_EMP_SILENT_REVISION) != 0)
      throw std::runtime_error("loaded EMP bridge revision is not pinned");
  }

  ~EmpSilentApi() {
    if (library_ != nullptr) ::dlclose(library_);
  }
  EmpSilentApi(const EmpSilentApi &) = delete;
  EmpSilentApi &operator=(const EmpSilentApi &) = delete;

  ringlpn_emp_handle *create(const ringlpn_emp_config &config) const {
    std::array<char, 256> error{};
    ringlpn_emp_handle *out = create_(&config, error.data(), error.size());
    if (out == nullptr) throw std::runtime_error(error.data());
    return out;
  }
  void begin(ringlpn_emp_handle *handle) const {
    call(begin_, handle, "begin");
  }
  void send(ringlpn_emp_handle *handle, uint32_t width, const void *m0,
            const void *m1, uint64_t n) const {
    std::array<char, 256> error{};
    if (send_(handle, width, m0, m1, n, error.data(), error.size()) != 0)
      throw std::runtime_error(std::string("EMP bridge send: ") + error.data());
  }
  void recv(ringlpn_emp_handle *handle, uint32_t width, const uint8_t *choices,
            void *out, uint64_t n) const {
    std::array<char, 256> error{};
    if (recv_(handle, width, choices, out, n, error.data(), error.size()) != 0)
      throw std::runtime_error(std::string("EMP bridge recv: ") + error.data());
  }
  void end(ringlpn_emp_handle *handle) const { call(end_, handle, "end"); }
  ringlpn_emp_counters counters(ringlpn_emp_handle *handle) const {
    ringlpn_emp_counters out{};
    std::array<char, 256> error{};
    if (counters_(handle, &out, error.data(), error.size()) != 0)
      throw std::runtime_error(std::string("EMP bridge counters: ") + error.data());
    return out;
  }
  void destroy(ringlpn_emp_handle *handle) const noexcept { destroy_(handle); }

 private:
  using AbiVersion = uint32_t (*)(void);
  using Revision = const char *(*)(void);
  using Create = ringlpn_emp_handle *(*)(const ringlpn_emp_config *, char *, size_t);
  using Begin = int (*)(ringlpn_emp_handle *, char *, size_t);
  using Send = int (*)(ringlpn_emp_handle *, uint32_t, const void *, const void *,
                       uint64_t, char *, size_t);
  using Recv = int (*)(ringlpn_emp_handle *, uint32_t, const uint8_t *, void *,
                       uint64_t, char *, size_t);
  using End = int (*)(ringlpn_emp_handle *, char *, size_t);
  using Counters = int (*)(const ringlpn_emp_handle *, ringlpn_emp_counters *,
                           char *, size_t);
  using Destroy = void (*)(ringlpn_emp_handle *);

  template <typename Function>
  void load(Function &function, const char *name) {
    ::dlerror();
    void *symbol = ::dlsym(library_, name);
    const char *error = ::dlerror();
    if (error != nullptr || symbol == nullptr)
      throw std::runtime_error(std::string("EMP bridge symbol missing: ") + name);
    std::memcpy(&function, &symbol, sizeof(function));
  }

  void call(Begin function, ringlpn_emp_handle *handle, const char *operation) const {
    std::array<char, 256> error{};
    if (function(handle, error.data(), error.size()) != 0)
      throw std::runtime_error(std::string("EMP bridge ") + operation + ": " +
                               error.data());
  }

  void *library_ = nullptr;
  AbiVersion abi_version_ = nullptr;
  Revision revision_ = nullptr;
  Create create_ = nullptr;
  Begin begin_ = nullptr;
  Send send_ = nullptr;
  Recv recv_ = nullptr;
  End end_ = nullptr;
  Counters counters_ = nullptr;
  Destroy destroy_ = nullptr;
};

class EmpSilentDirectionalOt {
 public:
  EmpSilentDirectionalOt(std::shared_ptr<EmpSilentApi> api, sci::NetIO *io,
                         int local_party, int direction, uint64_t count,
                         uint32_t threads,
                         const std::array<uint8_t, 32> &manifest_digest)
      : api_(std::move(api)) {
    if (api_ == nullptr || io == nullptr)
      throw std::invalid_argument("null EMP directional OT dependency");
    ringlpn_emp_config config{};
    config.abi_version = RINGLPN_EMP_SILENT_ABI_VERSION;
    config.local_party = static_cast<uint32_t>(local_party);
    config.sender_party = direction == RINGLPN_EMP_STRAIGHT ? 0u : 1u;
    config.direction = static_cast<uint32_t>(direction);
    config.parameter = direction == RINGLPN_EMP_STRAIGHT
                           ? RINGLPN_EMP_FERRET_B13
                           : RINGLPN_EMP_FERRET_B11;
    config.threads = threads;
    config.declared_count = count;
    config.capacity_floor = direction == RINGLPN_EMP_STRAIGHT
                                ? uint64_t{13727984}
                                : uint64_t{1602752};
    derive_sid(config.sid, manifest_digest, config.direction,
               config.sender_party, count);
    context_.io = io;
    config.io.context = &context_;
    config.io.send = &send_callback;
    config.io.recv = &recv_callback;
    config.io.flush = &flush_callback;
    handle_ = api_->create(config);
  }

  ~EmpSilentDirectionalOt() {
    if (handle_ != nullptr) api_->destroy(handle_);
  }
  EmpSilentDirectionalOt(const EmpSilentDirectionalOt &) = delete;
  EmpSilentDirectionalOt &operator=(const EmpSilentDirectionalOt &) = delete;

  void begin() { api_->begin(handle_); }
  void send(uint32_t width, const void *m0, const void *m1, uint64_t n) {
    api_->send(handle_, width, m0, m1, n);
  }
  void recv(uint32_t width, const uint8_t *choices, void *out, uint64_t n) {
    api_->recv(handle_, width, choices, out, n);
  }
  void end() { api_->end(handle_); }
  ringlpn_emp_counters counters() const { return api_->counters(handle_); }

 private:
  struct Context { sci::NetIO *io = nullptr; };

  static int send_callback(void *opaque, const void *data, size_t n) noexcept {
    try {
      auto *bytes = static_cast<const uint8_t *>(data);
      auto *io = static_cast<Context *>(opaque)->io;
      while (n != 0) {
        const int chunk = n > static_cast<size_t>(std::numeric_limits<int>::max())
                              ? std::numeric_limits<int>::max()
                              : static_cast<int>(n);
        io->send_data(bytes, chunk);
        bytes += chunk;
        n -= static_cast<size_t>(chunk);
      }
      return 0;
    } catch (...) { return -1; }
  }
  static int recv_callback(void *opaque, void *data, size_t n) noexcept {
    try {
      auto *bytes = static_cast<uint8_t *>(data);
      auto *io = static_cast<Context *>(opaque)->io;
      while (n != 0) {
        const int chunk = n > static_cast<size_t>(std::numeric_limits<int>::max())
                              ? std::numeric_limits<int>::max()
                              : static_cast<int>(n);
        io->recv_data(bytes, chunk);
        bytes += chunk;
        n -= static_cast<size_t>(chunk);
      }
      return 0;
    } catch (...) { return -1; }
  }
  static int flush_callback(void *opaque) noexcept {
    try {
      static_cast<Context *>(opaque)->io->flush();
      return 0;
    } catch (...) { return -1; }
  }

  static void derive_sid(uint8_t out[32],
                         const std::array<uint8_t, 32> &manifest,
                         uint32_t direction, uint32_t sender_party,
                         uint64_t count) {
    static constexpr char domain[] = "RINGLPN-EMP-SILENT-SID-v1";
    uint8_t fields[16]{};
    auto put32 = [](uint8_t *p, uint32_t v) {
      p[0] = static_cast<uint8_t>(v >> 24);
      p[1] = static_cast<uint8_t>(v >> 16);
      p[2] = static_cast<uint8_t>(v >> 8);
      p[3] = static_cast<uint8_t>(v);
    };
    put32(fields, direction);
    put32(fields + 4, sender_party);
    for (unsigned i = 0; i < 8; ++i)
      fields[8 + i] = static_cast<uint8_t>(count >> (56 - 8 * i));
    EVP_MD_CTX *raw = EVP_MD_CTX_new();
    if (raw == nullptr) throw std::runtime_error("EMP SID digest allocation failed");
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> md(raw,
                                                               EVP_MD_CTX_free);
    unsigned int size = 0;
    if (EVP_DigestInit_ex(md.get(), EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(md.get(), domain, sizeof(domain) - 1) != 1 ||
        EVP_DigestUpdate(md.get(), RINGLPN_EMP_SILENT_REVISION,
                         sizeof(RINGLPN_EMP_SILENT_REVISION) - 1) != 1 ||
        EVP_DigestUpdate(md.get(), manifest.data(), manifest.size()) != 1 ||
        EVP_DigestUpdate(md.get(), fields, sizeof(fields)) != 1 ||
        EVP_DigestFinal_ex(md.get(), out, &size) != 1 || size != 32)
      throw std::runtime_error("EMP SID digest failed");
    uint8_t manifest_or = 0;
    for (uint8_t byte : manifest) manifest_or |= byte;
    if (manifest_or == 0)
      throw std::invalid_argument("EMP SID requires a nonzero manifest digest");
  }

  std::shared_ptr<EmpSilentApi> api_;
  Context context_{};
  ringlpn_emp_handle *handle_ = nullptr;
};

}  // namespace ringlpn_2pc
