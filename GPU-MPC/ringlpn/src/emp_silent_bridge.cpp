#include "emp_silent_bridge.h"

#include <emp-ot/ot_extension/ferret/silent_ferret.h>
#include <emp-tool/emp-tool.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kRevision[] = RINGLPN_EMP_SILENT_REVISION;
constexpr char kControlDomain[] = "RINGLPN-EMP-SILENT-CONTROL-v1";
constexpr char kPadDomain[] = "RINGLPN-EMP-SILENT-PACKED-OT-PAD-v1";
constexpr uint64_t kB13Floor = 13727984;
constexpr uint64_t kB11Floor = 1602752;
constexpr size_t kControlBytes = 160;

void put_error(char *out, size_t size, const std::string &message) noexcept {
  if (out == nullptr || size == 0) return;
  const size_t n = std::min(size - 1, message.size());
  std::memcpy(out, message.data(), n);
  out[n] = '\0';
}

void put_u32_be(uint8_t *out, uint32_t value) {
  out[0] = static_cast<uint8_t>(value >> 24);
  out[1] = static_cast<uint8_t>(value >> 16);
  out[2] = static_cast<uint8_t>(value >> 8);
  out[3] = static_cast<uint8_t>(value);
}

void put_u64_be(uint8_t *out, uint64_t value) {
  for (unsigned i = 0; i < 8; ++i)
    out[i] = static_cast<uint8_t>(value >> (56 - 8 * i));
}

uint64_t checked_packed_bytes(uint64_t elements, uint32_t width,
                              uint64_t branches) {
  if (width == 0 || branches == 0 ||
      elements > std::numeric_limits<uint64_t>::max() / width / branches)
    throw std::overflow_error("packed OT bit count overflow");
  const uint64_t bits = elements * width * branches;
  if (bits > std::numeric_limits<uint64_t>::max() - 7)
    throw std::overflow_error("packed OT byte count overflow");
  return (bits + 7) / 8;
}

void checked_add(uint64_t &counter, uint64_t amount, const char *label) {
  if (amount > std::numeric_limits<uint64_t>::max() - counter)
    throw std::overflow_error(std::string(label) + " counter overflow");
  counter += amount;
}

class CallbackIO final : public emp::IOChannel {
 public:
  explicit CallbackIO(ringlpn_emp_io callbacks) : callbacks_(callbacks) {}

  void send_data_internal(const void *data, int64_t n) override {
    if (n < 0 || callbacks_.send(callbacks_.context, data,
                                  static_cast<size_t>(n)) != 0)
      throw std::runtime_error("transport send callback failed");
  }
  void recv_data_internal(void *data, int64_t n) override {
    if (n < 0 || callbacks_.recv(callbacks_.context, data,
                                  static_cast<size_t>(n)) != 0)
      throw std::runtime_error("transport receive callback failed");
  }
  void flush() override {
    if (callbacks_.flush(callbacks_.context) != 0)
      throw std::runtime_error("transport flush callback failed");
    if (flushes_count == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("flush counter overflow");
    ++flushes_count;
  }

 private:
  ringlpn_emp_io callbacks_;
};

struct Handle {
  explicit Handle(const ringlpn_emp_config &input)
      : config(input), io(input.io) {
    const int emp_party = config.local_party == config.sender_party
                              ? emp::ALICE
                              : emp::BOB;
    const auto parameter = config.parameter == RINGLPN_EMP_FERRET_B13
                               ? emp::tuning::ferret_b13
                               : emp::tuning::ferret_b11;
    ferret = std::make_unique<emp::SilentFerret>(
        emp_party, &io, true, parameter, nullptr,
        static_cast<int>(config.threads));

    std::array<uint8_t, 32> digest{};
    EVP_MD_CTX *raw = EVP_MD_CTX_new();
    if (raw == nullptr) throw std::runtime_error("EVP_MD_CTX_new failed");
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> md(raw,
                                                               EVP_MD_CTX_free);
    unsigned int digest_size = 0;
    if (EVP_DigestInit_ex(md.get(), EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(md.get(), kControlDomain,
                         sizeof(kControlDomain) - 1) != 1 ||
        EVP_DigestUpdate(md.get(), config.sid, sizeof(config.sid)) != 1 ||
        EVP_DigestUpdate(md.get(), kRevision, sizeof(kRevision) - 1) != 1 ||
        EVP_DigestFinal_ex(md.get(), digest.data(), &digest_size) != 1 ||
        digest_size != digest.size())
      throw std::runtime_error("session-id digest failed");
    emp::block sid_block;
    std::memcpy(&sid_block, digest.data(), sizeof(sid_block));
    ferret->set_sid(emp::SessionID(sid_block));
  }

  ringlpn_emp_config config{};
  CallbackIO io;
  std::unique_ptr<emp::SilentFerret> ferret;
  ringlpn_emp_counters counters{};
  bool active = false;
  bool failed = false;
};

std::array<uint8_t, kControlBytes> make_control(const Handle &h) {
  std::array<uint8_t, kControlBytes> out{};
  const size_t domain_size = sizeof(kControlDomain) - 1;
  static_assert(domain_size < 32);
  std::memcpy(out.data(), kControlDomain, domain_size);
  put_u32_be(out.data() + 32, RINGLPN_EMP_SILENT_ABI_VERSION);
  put_u32_be(out.data() + 36, h.config.direction);
  put_u32_be(out.data() + 40, h.config.sender_party);
  put_u32_be(out.data() + 44, h.config.parameter);
  put_u64_be(out.data() + 48, h.config.declared_count);
  put_u64_be(out.data() + 56, h.config.capacity_floor);
  std::memcpy(out.data() + 64, h.config.sid, sizeof(h.config.sid));
  std::memcpy(out.data() + 96, kRevision, sizeof(kRevision) - 1);
  return out;
}

void validate_width(uint32_t width) {
  if (width != 1 && width != 62 && width != 128)
    throw std::invalid_argument("OT message width must be exactly 1, 62, or 128");
}

uint8_t message_bit(const void *messages, uint32_t width, uint64_t index,
                    uint32_t bit) {
  if (width == 1) {
    const auto *values = static_cast<const uint8_t *>(messages);
    return static_cast<uint8_t>((values[index] >> bit) & 1u);
  }
  if (width == 62) {
    const auto *values = static_cast<const uint64_t *>(messages);
    return static_cast<uint8_t>((values[index] >> bit) & 1u);
  }
  const auto *values = static_cast<const uint8_t *>(messages) + 16 * index;
  return static_cast<uint8_t>((values[bit / 8] >> (bit % 8)) & 1u);
}

void set_message_bit(void *messages, uint32_t width, uint64_t index,
                     uint32_t bit, uint8_t value) {
  if (width == 1) {
    auto *values = static_cast<uint8_t *>(messages);
    values[index] = static_cast<uint8_t>((values[index] & ~(1u << bit)) |
                                         ((value & 1u) << bit));
    return;
  }
  if (width == 62) {
    auto *values = static_cast<uint64_t *>(messages);
    const uint64_t mask = uint64_t{1} << bit;
    values[index] = (values[index] & ~mask) |
                    (static_cast<uint64_t>(value & 1u) << bit);
    return;
  }
  auto *values = static_cast<uint8_t *>(messages) + 16 * index;
  const uint8_t mask = static_cast<uint8_t>(1u << (bit % 8));
  values[bit / 8] = static_cast<uint8_t>((values[bit / 8] & ~mask) |
                                         ((value & 1u) << (bit % 8)));
}

uint8_t packed_bit(const uint8_t *data, uint64_t bit) {
  return static_cast<uint8_t>((data[bit / 8] >> (bit % 8)) & 1u);
}

void set_packed_bit(uint8_t *data, uint64_t bit, uint8_t value) {
  data[bit / 8] = static_cast<uint8_t>(
      data[bit / 8] | static_cast<uint8_t>((value & 1u) << (bit % 8)));
}

std::array<uint8_t, 16> hash_pad(const Handle &h, const emp::block &input,
                                 uint32_t width, uint64_t index,
                                 uint8_t output_branch) {
  std::array<uint8_t, 16> out{};
  std::array<uint8_t, 12> numbers{};
  put_u32_be(numbers.data(), width);
  put_u64_be(numbers.data() + 4, index);
  EVP_MD_CTX *raw = EVP_MD_CTX_new();
  if (raw == nullptr) throw std::runtime_error("EVP_MD_CTX_new failed");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> md(raw,
                                                             EVP_MD_CTX_free);
  std::array<uint8_t, 32> digest{};
  unsigned int digest_size = 0;
  if (EVP_DigestInit_ex(md.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(md.get(), kPadDomain, sizeof(kPadDomain) - 1) != 1 ||
      EVP_DigestUpdate(md.get(), kRevision, sizeof(kRevision) - 1) != 1 ||
      EVP_DigestUpdate(md.get(), h.config.sid, sizeof(h.config.sid)) != 1 ||
      EVP_DigestUpdate(md.get(), &h.config.direction,
                       sizeof(h.config.direction)) != 1 ||
      EVP_DigestUpdate(md.get(), &h.config.sender_party,
                       sizeof(h.config.sender_party)) != 1 ||
      EVP_DigestUpdate(md.get(), &h.config.declared_count,
                       sizeof(h.config.declared_count)) != 1 ||
      EVP_DigestUpdate(md.get(), numbers.data(), numbers.size()) != 1 ||
      EVP_DigestUpdate(md.get(), &output_branch, 1) != 1 ||
      EVP_DigestUpdate(md.get(), &input, sizeof(input)) != 1 ||
      EVP_DigestFinal_ex(md.get(), digest.data(), &digest_size) != 1 ||
      digest_size != digest.size())
    throw std::runtime_error("packed OT pad hash failed");
  std::copy_n(digest.begin(), out.size(), out.begin());
  return out;
}

void require_usable(Handle &h, bool sender, uint64_t n) {
  if (h.failed || h.counters.poisoned)
    throw std::runtime_error("EMP SilentFerret handle is poisoned");
  if (!h.active || !h.counters.begun || h.counters.ended)
    throw std::runtime_error("OT consume requires a successful begin");
  const bool local_sender = h.config.local_party == h.config.sender_party;
  if (local_sender != sender)
    throw std::runtime_error("chosen OT operation conflicts with fixed role");
  if (n > h.counters.declared_count - h.counters.consumed_count)
    throw std::runtime_error("chosen OT consume exceeds declared inventory");
}

template <typename Function>
int guard(Handle *h, char *error, size_t error_size, Function &&function) noexcept {
  try {
    if (h == nullptr) throw std::invalid_argument("null EMP SilentFerret handle");
    function(*h);
    put_error(error, error_size, "");
    return 0;
  } catch (const std::exception &e) {
    if (h != nullptr) {
      h->failed = true;
      h->counters.poisoned = 1;
    }
    put_error(error, error_size, e.what());
    return -1;
  } catch (...) {
    if (h != nullptr) {
      h->failed = true;
      h->counters.poisoned = 1;
    }
    put_error(error, error_size, "unknown EMP SilentFerret failure");
    return -1;
  }
}

void validate_config(const ringlpn_emp_config &c) {
  if (c.abi_version != RINGLPN_EMP_SILENT_ABI_VERSION)
    throw std::invalid_argument("EMP SilentFerret ABI version mismatch");
  if (c.local_party > 1 || c.sender_party > 1)
    throw std::invalid_argument("EMP SilentFerret party must be 0 or 1");
  if (c.direction > RINGLPN_EMP_REVERSED)
    throw std::invalid_argument("EMP SilentFerret direction is invalid");
  if ((c.direction == RINGLPN_EMP_STRAIGHT && c.sender_party != 0) ||
      (c.direction == RINGLPN_EMP_REVERSED && c.sender_party != 1))
    throw std::invalid_argument("EMP SilentFerret role/direction mismatch");
  if ((c.direction == RINGLPN_EMP_STRAIGHT &&
       c.parameter != RINGLPN_EMP_FERRET_B13) ||
      (c.direction == RINGLPN_EMP_REVERSED &&
       c.parameter != RINGLPN_EMP_FERRET_B11))
    throw std::invalid_argument("EMP SilentFerret direction/parameter mismatch");
  const uint64_t required_floor = c.direction == RINGLPN_EMP_STRAIGHT
                                      ? kB13Floor
                                      : kB11Floor;
  if (c.capacity_floor < required_floor)
    throw std::invalid_argument("EMP SilentFerret capacity floor is too small");
  if (c.declared_count == 0 ||
      c.declared_count >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max() /
                                sizeof(emp::block)))
    throw std::invalid_argument("EMP SilentFerret declared count is invalid");
  if (c.threads == 0 || c.threads > static_cast<uint32_t>(INT32_MAX))
    throw std::invalid_argument("EMP SilentFerret thread count is invalid");
  if (c.io.send == nullptr || c.io.recv == nullptr || c.io.flush == nullptr)
    throw std::invalid_argument("EMP SilentFerret transport callback is null");
  uint8_t sid_or = 0;
  for (uint8_t byte : c.sid) sid_or |= byte;
  if (sid_or == 0) throw std::invalid_argument("EMP SilentFerret SID is zero");
}

}  // namespace

struct ringlpn_emp_handle : Handle {
  explicit ringlpn_emp_handle(const ringlpn_emp_config &config) : Handle(config) {}
};

extern "C" uint32_t ringlpn_emp_silent_abi_version(void) {
  return RINGLPN_EMP_SILENT_ABI_VERSION;
}

extern "C" const char *ringlpn_emp_silent_revision(void) { return kRevision; }

extern "C" ringlpn_emp_handle *ringlpn_emp_silent_create(
    const ringlpn_emp_config *config, char *error, size_t error_size) {
  try {
    if (config == nullptr) throw std::invalid_argument("null bridge config");
    validate_config(*config);
    auto handle = std::make_unique<ringlpn_emp_handle>(*config);
    handle->counters.declared_count = config->declared_count;
    put_error(error, error_size, "");
    return handle.release();
  } catch (const std::exception &e) {
    put_error(error, error_size, e.what());
    return nullptr;
  } catch (...) {
    put_error(error, error_size, "unknown EMP SilentFerret construction failure");
    return nullptr;
  }
}

extern "C" int ringlpn_emp_silent_begin(ringlpn_emp_handle *handle,
                                          char *error, size_t error_size) {
  return guard(handle, error, error_size, [](Handle &h) {
    if (h.active || h.counters.begun || h.counters.ended)
      throw std::runtime_error("EMP SilentFerret begin called out of order");
    const uint64_t sent_before = h.io.send_counter;
    const uint64_t recv_before = h.io.recv_counter;
    const auto mine = make_control(h);
    std::array<uint8_t, kControlBytes> peer{};
    if (h.config.local_party == h.config.sender_party) {
      h.io.send_data(mine.data(), mine.size());
      h.io.flush();
      h.io.recv_data(peer.data(), peer.size());
    } else {
      h.io.recv_data(peer.data(), peer.size());
      h.io.send_data(mine.data(), mine.size());
      h.io.flush();
    }
    const uint8_t local_match =
        CRYPTO_memcmp(mine.data(), peer.data(), mine.size()) == 0 ? 1 : 0;
    uint8_t peer_match = 0;
    if (h.config.local_party == h.config.sender_party) {
      h.io.send_data(&local_match, 1);
      h.io.flush();
      h.io.recv_data(&peer_match, 1);
    } else {
      h.io.recv_data(&peer_match, 1);
      h.io.send_data(&local_match, 1);
      h.io.flush();
    }
    if (local_match != 1 || peer_match != 1)
      throw std::runtime_error("EMP SilentFerret public control/SID mismatch");
    const uint64_t sent_control = h.io.send_counter;
    const uint64_t recv_control = h.io.recv_counter;
    h.ferret->begin(static_cast<int64_t>(h.config.declared_count));
    h.active = true;
    const int64_t capacity = h.ferret->prepared_capacity();
    if (capacity < 0 || static_cast<uint64_t>(capacity) < h.config.declared_count ||
        static_cast<uint64_t>(capacity) < h.config.capacity_floor)
      throw std::runtime_error("EMP SilentFerret prepared capacity is insufficient");
    h.counters.prepared_capacity = static_cast<uint64_t>(capacity);
    h.counters.setup_bytes_sent = h.io.send_counter - sent_before;
    h.counters.setup_bytes_received = h.io.recv_counter - recv_before;
    h.counters.correlation_bytes_sent = h.io.send_counter - sent_control;
    h.counters.correlation_bytes_received = h.io.recv_counter - recv_control;
    h.counters.begun = 1;
    h.counters.flushes = h.io.flushes_count;
  });
}

extern "C" int ringlpn_emp_silent_send(ringlpn_emp_handle *handle,
                                         uint32_t width, const void *messages0,
                                         const void *messages1, uint64_t n,
                                         char *error, size_t error_size) {
  return guard(handle, error, error_size, [&](Handle &h) {
    validate_width(width);
    require_usable(h, true, n);
    if (n != 0 && (messages0 == nullptr || messages1 == nullptr))
      throw std::invalid_argument("null chosen OT sender buffer");
    if (width == 62) {
      const auto *m0 = static_cast<const uint64_t *>(messages0);
      const auto *m1 = static_cast<const uint64_t *>(messages1);
      for (uint64_t i = 0; i < n; ++i)
        if ((m0[i] >> 62) != 0 || (m1[i] >> 62) != 0)
          throw std::invalid_argument("62-bit OT message exceeds field width");
    }
    if (n == 0) return;
    const uint64_t adjustment_bytes = checked_packed_bytes(n, 1, 1);
    const uint64_t ciphertext_bytes = checked_packed_bytes(n, width, 2);
    if (n > static_cast<uint64_t>(std::numeric_limits<size_t>::max() /
                                  sizeof(emp::block)) ||
        adjustment_bytes > std::numeric_limits<size_t>::max() ||
        ciphertext_bytes > std::numeric_limits<size_t>::max())
      throw std::overflow_error("chosen OT allocation size overflow");
    std::vector<emp::block> rcot(static_cast<size_t>(n));
    h.ferret->next_n(rcot.data(), static_cast<int64_t>(n));
    std::vector<uint8_t> adjustment(static_cast<size_t>(adjustment_bytes));
    h.io.recv_data(adjustment.data(), adjustment.size());
    std::vector<uint8_t> ciphertext(static_cast<size_t>(ciphertext_bytes), 0);
    for (uint64_t i = 0; i < n; ++i) {
      const uint8_t swap = packed_bit(adjustment.data(), i);
      for (uint8_t branch = 0; branch < 2; ++branch) {
        emp::block candidate = rcot[static_cast<size_t>(i)];
        if ((branch ^ swap) != 0) candidate = candidate ^ h.ferret->Delta;
        const auto pad = hash_pad(h, candidate, width,
                                  h.counters.consumed_count + i, branch);
        const void *source = branch == 0 ? messages0 : messages1;
        const uint64_t base = (2 * i + branch) * width;
        for (uint32_t bit = 0; bit < width; ++bit) {
          const uint8_t encrypted = static_cast<uint8_t>(
              message_bit(source, width, i, bit) ^
              ((pad[bit / 8] >> (bit % 8)) & 1u));
          set_packed_bit(ciphertext.data(), base + bit, encrypted);
        }
      }
    }
    h.io.send_data(ciphertext.data(), ciphertext.size());
    h.io.flush();
    checked_add(h.counters.adjustment_bytes_received, adjustment_bytes,
                "adjustment receive");
    checked_add(h.counters.ciphertext_bytes_sent, ciphertext_bytes,
                "ciphertext send");
    checked_add(h.counters.consumed_count, n, "consumed OT");
    h.counters.flushes = h.io.flushes_count;
  });
}

extern "C" int ringlpn_emp_silent_recv(ringlpn_emp_handle *handle,
                                         uint32_t width,
                                         const uint8_t *choices, void *messages,
                                         uint64_t n, char *error,
                                         size_t error_size) {
  return guard(handle, error, error_size, [&](Handle &h) {
    validate_width(width);
    require_usable(h, false, n);
    if (n != 0 && (choices == nullptr || messages == nullptr))
      throw std::invalid_argument("null chosen OT receiver buffer");
    for (uint64_t i = 0; i < n; ++i)
      if (choices[i] > 1)
        throw std::invalid_argument("chosen OT choice is not a bit");
    if (n == 0) return;
    const uint64_t adjustment_bytes = checked_packed_bytes(n, 1, 1);
    const uint64_t ciphertext_bytes = checked_packed_bytes(n, width, 2);
    if (n > static_cast<uint64_t>(std::numeric_limits<size_t>::max() /
                                  sizeof(emp::block)) ||
        adjustment_bytes > std::numeric_limits<size_t>::max() ||
        ciphertext_bytes > std::numeric_limits<size_t>::max())
      throw std::overflow_error("chosen OT allocation size overflow");
    std::vector<emp::block> rcot(static_cast<size_t>(n));
    h.ferret->next_n(rcot.data(), static_cast<int64_t>(n));
    std::vector<uint8_t> adjustment(static_cast<size_t>(adjustment_bytes), 0);
    for (uint64_t i = 0; i < n; ++i) {
      const uint8_t random_choice = static_cast<uint8_t>(
          emp::getLSB(rcot[static_cast<size_t>(i)]) ? 1 : 0);
      set_packed_bit(adjustment.data(), i,
                     static_cast<uint8_t>(choices[i] ^ random_choice));
    }
    h.io.send_data(adjustment.data(), adjustment.size());
    h.io.flush();
    std::vector<uint8_t> ciphertext(static_cast<size_t>(ciphertext_bytes));
    h.io.recv_data(ciphertext.data(), ciphertext.size());
    if (width == 1)
      std::memset(messages, 0, static_cast<size_t>(n));
    else if (width == 62)
      std::memset(messages, 0, static_cast<size_t>(n) * sizeof(uint64_t));
    else
      std::memset(messages, 0, static_cast<size_t>(n) * 16);
    for (uint64_t i = 0; i < n; ++i) {
      const uint8_t branch = choices[i];
      const auto pad = hash_pad(h, rcot[static_cast<size_t>(i)], width,
                                h.counters.consumed_count + i, branch);
      const uint64_t base = (2 * i + branch) * width;
      for (uint32_t bit = 0; bit < width; ++bit) {
        const uint8_t clear = static_cast<uint8_t>(
            packed_bit(ciphertext.data(), base + bit) ^
            ((pad[bit / 8] >> (bit % 8)) & 1u));
        set_message_bit(messages, width, i, bit, clear);
      }
    }
    checked_add(h.counters.adjustment_bytes_sent, adjustment_bytes,
                "adjustment send");
    checked_add(h.counters.ciphertext_bytes_received, ciphertext_bytes,
                "ciphertext receive");
    checked_add(h.counters.consumed_count, n, "consumed OT");
    h.counters.flushes = h.io.flushes_count;
  });
}

extern "C" int ringlpn_emp_silent_end(ringlpn_emp_handle *handle,
                                        char *error, size_t error_size) {
  return guard(handle, error, error_size, [](Handle &h) {
    if (h.failed || h.counters.poisoned)
      throw std::runtime_error("cannot end a poisoned EMP SilentFerret session");
    if (!h.active || !h.counters.begun || h.counters.ended)
      throw std::runtime_error("EMP SilentFerret end called out of order");
    if (h.counters.consumed_count != h.counters.declared_count)
      throw std::runtime_error("EMP SilentFerret inventory was not exactly exhausted");
    h.ferret->end();
    h.active = false;
    h.counters.ended = 1;
    h.counters.flushes = h.io.flushes_count;
  });
}

extern "C" int ringlpn_emp_silent_get_counters(
    const ringlpn_emp_handle *handle, ringlpn_emp_counters *out, char *error,
    size_t error_size) {
  try {
    if (handle == nullptr || out == nullptr)
      throw std::invalid_argument("null bridge counter argument");
    *out = handle->counters;
    put_error(error, error_size, "");
    return 0;
  } catch (const std::exception &e) {
    put_error(error, error_size, e.what());
    return -1;
  } catch (...) {
    put_error(error, error_size, "unknown bridge counter failure");
    return -1;
  }
}

extern "C" void ringlpn_emp_silent_destroy(ringlpn_emp_handle *handle) {
  if (handle == nullptr) return;
  if (handle->active) {
    try {
      handle->ferret->end();
      handle->active = false;
    } catch (...) {
      /* An active EMP object cannot be destroyed safely after end failed. */
      return;
    }
  }
  try {
    delete handle;
  } catch (...) {
    /* EMP's lifecycle tripwire may reject a transport-failed session. */
  }
}
