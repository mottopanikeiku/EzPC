#pragma once

#include "emp_silent_bridge.h"
#include "emp_silent_bridge_authorization.h"
#include "utils/net_io_channel.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
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
class EmpSilentApi;

struct EmpSilentPlan {
  std::shared_ptr<EmpSilentApi> api;
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
    try {
      open_authorized(path);
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
        throw std::runtime_error("loaded EMP bridge ABI version is incompatible");
      const char *revision = revision_();
      if (revision == nullptr ||
          std::strcmp(revision, RINGLPN_EMP_SILENT_REVISION) != 0)
        throw std::runtime_error("loaded EMP bridge revision is incompatible");
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~EmpSilentApi() { cleanup(); }
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
  const std::array<uint8_t, 32> &measured_digest() const noexcept {
    return measured_digest_;
  }
  static std::array<uint8_t, 32> source_authorized_digest() {
    return parse_authorized_digest();
  }


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
  static std::array<uint8_t, 32> parse_authorized_digest() {
    constexpr char hex[] = RINGLPN_EMP_SILENT_BRIDGE_SHA256;
    static_assert(sizeof(hex) == 65,
                  "authorized EMP bridge SHA-256 must have 64 hex digits");
    std::array<uint8_t, 32> digest{};
    auto nibble = [](char c) -> uint8_t {
      if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
      if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
      throw std::runtime_error(
          "authorized EMP bridge SHA-256 is not lowercase hexadecimal");
    };
    for (size_t i = 0; i < digest.size(); ++i)
      digest[i] = static_cast<uint8_t>((nibble(hex[2 * i]) << 4) |
                                       nibble(hex[2 * i + 1]));
    return digest;
  }

  static bool same_identity(const struct stat &a, const struct stat &b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino &&
           a.st_uid == b.st_uid && a.st_nlink == b.st_nlink &&
           a.st_mode == b.st_mode && a.st_size == b.st_size &&
           a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
           a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
           a.st_ctim.tv_sec == b.st_ctim.tv_sec &&
           a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
  }

  static void close_fd(int &fd) noexcept {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }

  void cleanup() noexcept {
    if (library_ != nullptr) {
      ::dlclose(library_);
      library_ = nullptr;
    }
    close_fd(sealed_fd_);
  }

  void open_authorized(const std::string &path) {
    int source = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (source < 0)
      throw std::runtime_error(std::string("failed to open EMP bridge: ") +
                               std::strerror(errno));
    struct stat before {};
    if (::fstat(source, &before) != 0) {
      close_fd(source);
      throw std::runtime_error("failed to stat EMP bridge");
    }
    const uid_t euid = ::geteuid();
    if (!S_ISREG(before.st_mode) ||
        (before.st_uid != euid && before.st_uid != 0) ||
        before.st_nlink != 1 || (before.st_mode & 0222) != 0 ||
        before.st_size <= 0 || before.st_size > (off_t{1} << 30)) {
      close_fd(source);
      throw std::runtime_error(
          "EMP bridge must be a read-only, single-link regular file owned by "
          "the effective user or root");
    }
#if !defined(SYS_memfd_create) || !defined(F_ADD_SEALS) || \
    !defined(F_SEAL_SEAL) || !defined(F_SEAL_SHRINK) || \
    !defined(F_SEAL_GROW) || !defined(F_SEAL_WRITE)
    close_fd(source);
    throw std::runtime_error("sealed EMP bridge loading is unsupported");
#else
    constexpr unsigned kMemfdCloexec = 0x0001U;
    constexpr unsigned kMemfdAllowSealing = 0x0002U;
    sealed_fd_ = static_cast<int>(
        ::syscall(SYS_memfd_create, "ringlpn-emp-silent",
                  kMemfdCloexec | kMemfdAllowSealing));
    if (sealed_fd_ < 0) {
      close_fd(source);
      throw std::runtime_error("failed to create sealed EMP bridge image");
    }
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> md(
        EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!md || EVP_DigestInit_ex(md.get(), EVP_sha256(), nullptr) != 1) {
      close_fd(source);
      throw std::runtime_error("failed to initialize EMP bridge digest");
    }
    std::array<uint8_t, 65536> buffer{};
    off_t copied = 0;
    while (copied < before.st_size) {
      const size_t request = static_cast<size_t>(
          std::min<off_t>(before.st_size - copied, buffer.size()));
      const ssize_t got = ::pread(source, buffer.data(), request, copied);
      if (got <= 0) {
        close_fd(source);
        throw std::runtime_error("failed to read complete EMP bridge");
      }
      size_t written = 0;
      while (written < static_cast<size_t>(got)) {
        const ssize_t count =
            ::write(sealed_fd_, buffer.data() + written,
                    static_cast<size_t>(got) - written);
        if (count <= 0) {
          close_fd(source);
          throw std::runtime_error("failed to copy EMP bridge");
        }
        written += static_cast<size_t>(count);
      }
      if (EVP_DigestUpdate(md.get(), buffer.data(),
                           static_cast<size_t>(got)) != 1) {
        close_fd(source);
        throw std::runtime_error("failed to hash EMP bridge");
      }
      copied += got;
    }
    struct stat after {};
    unsigned int digest_size = 0;
    if (::fstat(source, &after) != 0 || !same_identity(before, after) ||
        EVP_DigestFinal_ex(md.get(), measured_digest_.data(), &digest_size) !=
            1 ||
        digest_size != measured_digest_.size()) {
      close_fd(source);
      throw std::runtime_error("EMP bridge changed while being measured");
    }
    close_fd(source);
    struct stat sealed {};
    if (::fstat(sealed_fd_, &sealed) != 0 ||
        !S_ISREG(sealed.st_mode) || sealed.st_size != before.st_size)
      throw std::runtime_error("sealed EMP bridge copy is incomplete");
    const auto expected = source_authorized_digest();
    const int seals =
        F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;
    if (::fcntl(sealed_fd_, F_ADD_SEALS, seals) != 0)
      throw std::runtime_error("failed to seal authorized EMP bridge bytes");
    std::array<uint8_t, 32> sealed_digest{};
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> sealed_md(
        EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!sealed_md ||
        EVP_DigestInit_ex(sealed_md.get(), EVP_sha256(), nullptr) != 1)
      throw std::runtime_error(
          "failed to initialize sealed EMP bridge digest");
    off_t sealed_offset = 0;
    while (sealed_offset < sealed.st_size) {
      const size_t request = static_cast<size_t>(
          std::min<off_t>(sealed.st_size - sealed_offset, buffer.size()));
      const ssize_t got =
          ::pread(sealed_fd_, buffer.data(), request, sealed_offset);
      if (got <= 0 ||
          EVP_DigestUpdate(sealed_md.get(), buffer.data(),
                           static_cast<size_t>(got)) != 1)
        throw std::runtime_error("failed to hash sealed EMP bridge bytes");
      sealed_offset += got;
    }
    unsigned int sealed_digest_size = 0;
    if (EVP_DigestFinal_ex(sealed_md.get(), sealed_digest.data(),
                           &sealed_digest_size) != 1 ||
        sealed_digest_size != sealed_digest.size() ||
        CRYPTO_memcmp(sealed_digest.data(), measured_digest_.data(),
                      sealed_digest.size()) != 0 ||
        CRYPTO_memcmp(sealed_digest.data(), expected.data(),
                      sealed_digest.size()) != 0)
      throw std::runtime_error("sealed EMP bridge SHA-256 is not authorized");
    measured_digest_ = sealed_digest;
    if (::lseek(sealed_fd_, 0, SEEK_SET) != 0)
      throw std::runtime_error("failed to rewind sealed EMP bridge image");
    const std::string sealed_path =
        "/proc/self/fd/" + std::to_string(sealed_fd_);
    library_ = ::dlopen(sealed_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr)
      throw std::runtime_error(std::string("failed to load EMP bridge: ") +
                               ::dlerror());
#endif
  }


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
  int sealed_fd_ = -1;
  std::array<uint8_t, 32> measured_digest_{};
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
