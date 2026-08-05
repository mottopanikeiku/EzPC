// Canonical live-correlation namespaces and crash-persistent consume-once claims.
//
// A live invocation claims its entire namespace before PartyChannel OT setup or
// PartyRandom construction.  The namespace tuple is party independent; parties
// consume matching state slices in the same public order.  The per-party claim
// filename prevents one local party from blocking the matching peer while still
// making every retry/restart of that party fail closed.  Claim files and pending
// files are immutable append-only ledger entries.  A malformed/truncated entry
// makes the ledger unusable rather than permitting rollback.
#pragma once

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iterator>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1U << 0)
#endif

namespace ringlpn_freshness {

constexpr uint32_t kProtocolVersion = 1;
constexpr size_t kInvocationBytes = 16;
constexpr size_t kDigestBytes = 32;
using InvocationId = std::array<uint8_t, kInvocationBytes>;
using Digest = std::array<uint8_t, kDigestBytes>;

constexpr uint64_t kUnused = std::numeric_limits<uint64_t>::max();

enum class Kind : uint32_t {
    kRingOle = 1,
    kDpfBitTriple = 2,
    kDpfStringOt = 3,
    kDpfScalarOle = 4,
    kPublicPolynomialShare = 5,
    kDerandomizationOpening = 6,
    kConversionDabit = 7,
    kConversionEdabit = 8,
    kConversionBooleanTriple = 9,
    kConversionMaskOpening = 10,
    kOperandMask = 11,
    kOutputMask = 12,
};

enum class Phase : uint32_t {
    kInputMask = 1,
    kDpfA = 2,
    kDpfB = 3,
    kDpfC = 4,
    kPublicPolynomial = 5,
    kRingExpansion = 6,
    kDerandomize = 7,
    kConvertCorrelation = 8,
    kConvertOnline = 9,
    kOutputMask = 10,
};

// Every field is encoded for every ID. kUnused is the canonical value when a
// coordinate does not apply. `layer` is a stable public layer ordinal, while
// `layer_identity` binds the complete party-independent public preflight.
struct Coordinates {
    Kind kind = Kind::kRingOle;
    uint64_t layer = 0;
    uint64_t direction = kUnused;
    uint64_t crt_limb = kUnused;
    uint64_t ring_batch = kUnused;
    uint64_t dpf_tree = kUnused;
    Phase phase = Phase::kRingExpansion;
    uint64_t primitive_ordinal = kUnused;
    uint64_t conversion_chunk = kUnused;
    uint64_t output_slot = kUnused;
};

inline bool digest(const uint8_t *data, size_t size, Digest &out) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (ctx == nullptr) return false;
    unsigned int written = 0;
    const bool ok = EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1 &&
                    EVP_DigestUpdate(ctx, data, size) == 1 &&
                    EVP_DigestFinal_ex(ctx, out.data(), &written) == 1 &&
                    written == out.size();
    EVP_MD_CTX_free(ctx);
    return ok;
}

inline char hex_digit(uint8_t value) {
    return value < 10 ? static_cast<char>('0' + value)
                      : static_cast<char>('a' + value - 10);
}

template <size_t N>
inline std::string hex(const std::array<uint8_t, N> &bytes) {
    std::string out(2 * N, '0');
    for (size_t i = 0; i < N; ++i) {
        out[2 * i] = hex_digit(static_cast<uint8_t>(bytes[i] >> 4));
        out[2 * i + 1] = hex_digit(static_cast<uint8_t>(bytes[i] & 15));
    }
    return out;
}

inline int unhex(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
}

inline bool parse_invocation_id(const std::string &text, InvocationId &out) {
    if (text.size() != 2 * out.size()) return false;
    InvocationId parsed{};
    uint8_t any = 0;
    for (size_t i = 0; i < parsed.size(); ++i) {
        const int high = unhex(text[2 * i]);
        const int low = unhex(text[2 * i + 1]);
        if (high < 0 || low < 0) return false;
        parsed[i] = static_cast<uint8_t>((high << 4) | low);
        any = static_cast<uint8_t>(any | parsed[i]);
    }
    if (any == 0) return false;
    out = parsed;
    return true;
}

inline void put_u32_be(std::vector<uint8_t> &out, uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8) {
        out.push_back(static_cast<uint8_t>(value >> shift));
    }
}

inline void put_u64_be(std::vector<uint8_t> &out, uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        out.push_back(static_cast<uint8_t>(value >> shift));
    }
}

// SHA-256 is used only after an injective, fixed-width canonical encoding.  Its
// collision resistance is an explicit external assumption, not an algebraic
// injectivity claim about the hash itself.
inline bool derive_correlation_id(const InvocationId &invocation,
                                  const Digest &layer_identity,
                                  const Coordinates &coordinates,
                                  Digest &out) {
    static constexpr uint8_t domain[] =
        "RINGLPN-CORRELATION-ID-v1";
    std::vector<uint8_t> encoded;
    encoded.reserve(sizeof(domain) - 1 + 4 + invocation.size() +
                    layer_identity.size() + 4 + 9 * 8 + 4);
    encoded.insert(encoded.end(), domain, domain + sizeof(domain) - 1);
    put_u32_be(encoded, kProtocolVersion);
    encoded.insert(encoded.end(), invocation.begin(), invocation.end());
    encoded.insert(encoded.end(), layer_identity.begin(), layer_identity.end());
    put_u32_be(encoded, static_cast<uint32_t>(coordinates.kind));
    put_u64_be(encoded, coordinates.layer);
    put_u64_be(encoded, coordinates.direction);
    put_u64_be(encoded, coordinates.crt_limb);
    put_u64_be(encoded, coordinates.ring_batch);
    put_u64_be(encoded, coordinates.dpf_tree);
    put_u32_be(encoded, static_cast<uint32_t>(coordinates.phase));
    put_u64_be(encoded, coordinates.primitive_ordinal);
    put_u64_be(encoded, coordinates.conversion_chunk);
    put_u64_be(encoded, coordinates.output_slot);
    return digest(encoded.data(), encoded.size(), out);
}

// Existing 64-bit protocol manifest fields are compatibility handles only.  A
// live security/correlation identity is always the full 256-bit value above.
inline uint64_t compatibility_handle(const Digest &id) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) value = (value << 8) | id[i];
    return value == 0 ? 1 : value;
}

struct Claim {
    InvocationId invocation{};
    Digest layer_identity{};
    Digest plan_digest{};
    Digest ledger_digest{};  // digest of the canonical claim body
};

inline bool write_all(int fd, const uint8_t *data, size_t size) {
    size_t cursor = 0;
    while (cursor < size) {
        const ssize_t wrote = ::write(fd, data + cursor, size - cursor);
        if (wrote < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (wrote == 0) return false;
        cursor += static_cast<size_t>(wrote);
    }
    return true;
}

inline bool has_suffix(const std::string &value, const char *suffix) {
    const size_t n = std::strlen(suffix);
    return value.size() >= n && value.compare(value.size() - n, n, suffix) == 0;
}

inline bool read_exact_claim(const std::string &path) {
    constexpr size_t kBodyBytes = 16 + 4 + kInvocationBytes + 2 * kDigestBytes;
    constexpr size_t kFileBytes = kBodyBytes + kDigestBytes;
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat info {};
    std::array<uint8_t, kFileBytes> bytes{};
    size_t cursor = 0;
    bool ok = ::fstat(fd, &info) == 0 && S_ISREG(info.st_mode) &&
              info.st_uid == ::geteuid() && (info.st_mode & 0077) == 0 &&
              info.st_nlink == 1 &&
              static_cast<uint64_t>(info.st_size) == kFileBytes;
    while (ok && cursor < bytes.size()) {
        const ssize_t got = ::read(fd, bytes.data() + cursor,
                                   bytes.size() - cursor);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) {
            ok = false;
            break;
        }
        cursor += static_cast<size_t>(got);
    }
    ::close(fd);
    static constexpr uint8_t magic[16] = {
        'R','L','P','N','F','R','E','S','H','L','E','D','G','E','R','1'};
    if (!ok || !std::equal(std::begin(magic), std::end(magic), bytes.begin()) ||
        bytes[16] != 0 || bytes[17] != 0 || bytes[18] != 0 ||
        bytes[19] != kProtocolVersion) {
        return false;
    }
    Digest expected{};
    return digest(bytes.data(), kBodyBytes, expected) &&
           std::equal(expected.begin(), expected.end(),
                      bytes.begin() + kBodyBytes);
}

inline bool validate_ledger_entries(const std::string &root) {
    DIR *directory = ::opendir(root.c_str());
    if (directory == nullptr) return false;
    bool ok = true;
    errno = 0;
    while (dirent *entry = ::readdir(directory)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        if (!has_suffix(name, ".claim") && !has_suffix(name, ".pending")) {
            ok = false;
            break;
        }
        if (!read_exact_claim(root + "/" + name)) {
            ok = false;
            break;
        }
        errno = 0;
    }
    if (errno != 0) ok = false;
    ::closedir(directory);
    return ok;
}

inline bool mkdirs_private(const std::string &absolute) {
    if (absolute.empty() || absolute.front() != '/' ||
        (absolute.size() > 1 && absolute.back() == '/')) {
        return false;
    }
    size_t position = 1;
    while (position <= absolute.size()) {
        const size_t slash = absolute.find('/', position);
        const size_t end = slash == std::string::npos ? absolute.size() : slash;
        if (end > 1) {
            const std::string cursor = absolute.substr(0, end);
            if (::mkdir(cursor.c_str(), 0700) != 0 && errno != EEXIST) {
                return false;
            }
            struct stat info {};
            if (::lstat(cursor.c_str(), &info) != 0 || !S_ISDIR(info.st_mode)) {
                return false;
            }
            if (end == absolute.size() &&
                (info.st_uid != ::geteuid() || (info.st_mode & 0077) != 0)) {
                return false;
            }
            const size_t parent_end = cursor.find_last_of('/');
            const std::string parent =
                parent_end == 0 ? "/" : cursor.substr(0, parent_end);
            const int parent_fd =
                ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                            O_NOFOLLOW);
            if (parent_fd < 0 || ::fsync(parent_fd) != 0) {
                if (parent_fd >= 0) ::close(parent_fd);
                return false;
            }
            if (::close(parent_fd) != 0) return false;
        }
        if (slash == std::string::npos) break;
        position = slash + 1;
    }
    return true;
}

inline bool rename_no_replace(const std::string &from, const std::string &to) {
#if defined(SYS_renameat2) && defined(RENAME_NOREPLACE)
    if (::syscall(SYS_renameat2, AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(),
                  RENAME_NOREPLACE) == 0) {
        return true;
    }
    if (errno != ENOSYS && errno != EINVAL) return false;
#endif
    // link()+unlink() is the no-replace atomic fallback. The pending inode was
    // already fsynced, and the directory is fsynced by the caller.
    if (::link(from.c_str(), to.c_str()) != 0) return false;
    return ::unlink(from.c_str()) == 0;
}

inline bool claim_namespace_once(const std::string &ledger_root, int party,
                                 const InvocationId &invocation,
                                 const Digest &layer_identity,
                                 const Digest &plan_digest, Claim &claim) {
    claim = Claim{};
    const auto nonzero = [](const auto &bytes) {
        return std::any_of(bytes.begin(), bytes.end(),
                           [](uint8_t byte) { return byte != 0; });
    };
    if ((party != 0 && party != 1) || !nonzero(invocation) ||
        !nonzero(layer_identity) || !nonzero(plan_digest) ||
        !mkdirs_private(ledger_root) || !validate_ledger_entries(ledger_root)) {
        return false;
    }
    static constexpr uint8_t magic[16] = {
        'R','L','P','N','F','R','E','S','H','L','E','D','G','E','R','1'};
    std::vector<uint8_t> body;
    body.insert(body.end(), std::begin(magic), std::end(magic));
    put_u32_be(body, kProtocolVersion);
    body.insert(body.end(), invocation.begin(), invocation.end());
    body.insert(body.end(), layer_identity.begin(), layer_identity.end());
    body.insert(body.end(), plan_digest.begin(), plan_digest.end());
    Digest claim_digest{};
    if (!digest(body.data(), body.size(), claim_digest)) return false;
    std::vector<uint8_t> file = body;
    file.insert(file.end(), claim_digest.begin(), claim_digest.end());

    const std::string stem = hex(invocation) + ".p" + std::to_string(party);
    const std::string pending = ledger_root + "/" + stem + ".pending";
    const std::string final = ledger_root + "/" + stem + ".claim";
    int fd = ::open(pending.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC |
                                      O_NOFOLLOW,
                    0600);
    if (fd < 0) return false;  // duplicate/retry/restart is consumed
    bool ok = write_all(fd, file.data(), file.size()) && ::fsync(fd) == 0;
    if (::close(fd) != 0) ok = false;
    if (!ok) return false;  // retain pending entry: never roll back consumption
    if (!rename_no_replace(pending, final)) return false;
    const int directory_fd = ::open(ledger_root.c_str(), O_RDONLY | O_DIRECTORY |
                                                          O_CLOEXEC | O_NOFOLLOW);
    if (directory_fd < 0) return false;
    ok = ::fsync(directory_fd) == 0;
    if (::close(directory_fd) != 0) ok = false;
    if (!ok) return false;
    claim.invocation = invocation;
    claim.layer_identity = layer_identity;
    claim.plan_digest = plan_digest;
    claim.ledger_digest = claim_digest;
    return true;
}

}  // namespace ringlpn_freshness
