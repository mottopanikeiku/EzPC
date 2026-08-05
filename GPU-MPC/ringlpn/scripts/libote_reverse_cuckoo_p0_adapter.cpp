// Adapter for osu-crypto/libOTe dmpf@edb5d32822eabf2dda9f6844d85d0ce2e402cdd5.
// libOTe is MIT licensed; this file is not a vendored copy of libOTe.
#include "coproto/Socket/LocalAsyncSock.h"
#include "cryptoTools/Common/Matrix.h"
#include "cryptoTools/Crypto/PRNG.h"
#include "libOTe/Tools/Field/Fp.h"
#include "libOTe/Triple/RingLpn/RingLpnTriple.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using namespace osuCrypto;
using Field = Fp<4611686018326724609ULL, u64, __uint128_t>;
constexpr u64 kModulus = 4611686018326724609ULL;
constexpr u64 kN = 1ULL << 20;
constexpr u64 kC = 4;
constexpr u64 kT = 16;
constexpr u64 kBlockSize = kN / kT;

struct P0CanonicalCoeffCtx : CoeffCtxFp {
    template <typename F>
    u64 bitSize() const {
        static_assert(std::is_same_v<F, Field>, "adapter only supports the pinned p0 field");
        return 62;
    }

    template <typename F>
    BitVector binaryDecomposition(F& x) const {
        static_assert(std::is_same_v<F, Field>, "adapter only supports the pinned p0 field");
        if (x.mVal >= kModulus) throw std::runtime_error("non-canonical p0 coefficient");
        return BitVector(reinterpret_cast<u8*>(&x.mVal), 62);
    }

    template <typename F>
    void fromBlock(F& out, const block& in) const {
        static_assert(std::is_same_v<F, Field>, "adapter only supports the pinned p0 field");
        const __uint128_t wide = (__uint128_t(in.get<u64>(1)) << 64) | in.get<u64>(0);
        out.mVal = static_cast<u64>(wide % kModulus);
    }

    template <typename F>
    void powerOfTwo(F& out, u64 power) const {
        static_assert(std::is_same_v<F, Field>, "adapter only supports the pinned p0 field");
        if (power >= 62) throw std::runtime_error("p0 powerOfTwo outside canonical decomposition");
        out = F(1ULL << power);
    }
};

using Protocol = RingLpnTriple<Field, P0CanonicalCoeffCtx>;
using Clock = std::chrono::steady_clock;
using Reference = std::array<std::map<u64, Field>, kC * kC>;

u64 elapsedUs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

std::string jsonEscape(const std::string& input) {
    std::ostringstream out;
    for (unsigned char ch : input) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(ch)
                    << std::dec;
            } else {
                out << ch;
            }
        }
    }
    return out.str();
}

Reference buildReference(
    const std::array<Matrix<u64>, 2>& positions,
    const std::array<std::vector<Field>, 2>& coefficients,
    u64& duplicateTerms) {
    Reference reference;
    duplicateTerms = 0;
    for (u64 i = 0; i < kC; ++i) {
        for (u64 j = 0; j < kC; ++j) {
            auto& row = reference[i * kC + j];
            for (u64 r = 0; r < kT; ++r) {
                for (u64 s = 0; s < kT; ++s) {
                    const u64 blockSum = r + s;
                    const u64 foldedBlock = blockSum & (kT - 1);
                    const u64 leaf = positions[0](i, r) + positions[1](j, s);
                    u64 rawIndex = foldedBlock * kBlockSize + leaf;
                    Field value = coefficients[0][i * kT + r] * coefficients[1][j * kT + s];
                    if (blockSum >= kT) value = -value;
                    if (rawIndex >= kN) {
                        rawIndex -= kN;
                        value = -value;
                    }
                    auto [it, inserted] = row.try_emplace(rawIndex, Field(0));
                    if (!inserted) ++duplicateTerms;
                    it->second += value;
                }
            }
        }
    }
    return reference;
}

bool validateFullDomain(
    const std::array<Protocol, 2>& protocols,
    const Reference& reference,
    u64& mismatchRow,
    u64& mismatchIndex) {
    for (u64 row = 0; row < kC * kC; ++row) {
        auto expected = reference[row].begin();
        for (u64 x = 0; x < kN; ++x) {
            Field want(0);
            if (expected != reference[row].end() && expected->first == x) {
                want = expected->second;
                ++expected;
            }
            const Field got = protocols[0].mNativeFoldedOutput(row, x)
                + protocols[1].mNativeFoldedOutput(row, x);
            if (got != want) {
                mismatchRow = row;
                mismatchIndex = x;
                return false;
            }
        }
        if (expected != reference[row].end()) {
            mismatchRow = row;
            mismatchIndex = expected->first;
            return false;
        }
    }
    return true;
}

void writeFailure(const std::string& path, const std::string& blocker) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write report: " + path);
    out << "{\n"
        << "  \"schema_version\": \"reverse-cuckoo-p0-baseline-1\",\n"
        << "  \"date\": \"2026-08-04\",\n"
        << "  \"status\": \"unsupported\",\n"
        << "  \"blocker\": \"" << jsonEscape(blocker) << "\",\n"
        << "  \"metrics_emitted\": false\n"
        << "}\n";
}

int run(const std::string& reportPath) {
    static_assert(sizeof(Field) == sizeof(u64));
    P0CanonicalCoeffCtx ctx;
    if (ctx.bitSize<Field>() != 62) throw std::runtime_error("coefficient context is not 62-bit");
    for (u64 bit = 0; bit < 62; ++bit) {
        Field power;
        ctx.powerOfTwo(power, bit);
        if (power != Field(1ULL << bit)) throw std::runtime_error("powerOfTwo self-check failed");
    }

    std::array<Protocol, 2> protocols;
    for (u64 party = 0; party < 2; ++party) {
        protocols[party].mNumPolys = kC;
        protocols[party].mPolyWeight = kT;
        protocols[party].init(
            party, kN, Protocol::Mode::Ole, Protocol::DpfType::RevCuckooDmpf,
            Protocol::TensorBaseCorType::OtBased, ctx);
        protocols[party].setNativeFoldedCaptureOnly(true);
    }

    std::array<Matrix<u64>, 2> positions;
    std::array<std::vector<Field>, 2> coefficients;
    for (u64 party = 0; party < 2; ++party) {
        positions[party].resize(kC, kT);
        coefficients[party].resize(kC * kT);
        for (u64 i = 0; i < kC; ++i) {
            for (u64 r = 0; r < kT; ++r) {
                // This deliberate collision fixture also exercises the one-block leaf overflow.
                positions[party](i, r) = kBlockSize - 1;
                coefficients[party][i * kT + r] =
                    Field(1 + party * 1000003ULL + i * 65537ULL + r * 257ULL);
            }
        }
    }

    auto sockets = coproto::LocalAsyncSocket::makePair();
    PRNG prng0(block(0x69f6c8f95f2892d1ULL, 0xa2ccad51ff26aa63ULL));
    PRNG prng1(block(0x5ea7b67c58ad319dULL, 0x97be9e62742f1134ULL));

    const auto totalBegin = Clock::now();
    const u64 setupBytesBefore = sockets[0].bytesReceived() + sockets[1].bytesReceived();
    const auto setupBegin = Clock::now();
    auto setupResult = macoro::sync_wait(macoro::when_all_ready(
        protocols[0].genBaseCors(prng0, sockets[0]),
        protocols[1].genBaseCors(prng1, sockets[1])));
    std::get<0>(setupResult).result();
    std::get<1>(setupResult).result();

    for (u64 party = 0; party < 2; ++party) {
        protocols[party].setCallerFactors(positions[party], coefficients[party]);
    }
    auto factorResult = macoro::sync_wait(macoro::when_all_ready(
        protocols[0].tensor(prng0, sockets[0]),
        protocols[1].tensor(prng1, sockets[1])));
    std::get<0>(factorResult).result();
    std::get<1>(factorResult).result();
    auto pointResult = macoro::sync_wait(macoro::when_all_ready(
        protocols[0].genDpf(prng0, sockets[0]),
        protocols[1].genDpf(prng1, sockets[1])));
    std::get<0>(pointResult).result();
    std::get<1>(pointResult).result();
    const auto setupEnd = Clock::now();
    const u64 setupBytes = sockets[0].bytesReceived() + sockets[1].bytesReceived() - setupBytesBefore;

    std::array<std::vector<Field>, 2> unusedA, unusedB;
    const u64 onlineBytesBefore = sockets[0].bytesReceived() + sockets[1].bytesReceived();
    const auto onlineBegin = Clock::now();
    auto onlineResult = macoro::sync_wait(macoro::when_all_ready(
        protocols[0].expand(unusedA[0], unusedB[0], prng0, sockets[0]),
        protocols[1].expand(unusedA[1], unusedB[1], prng1, sockets[1])));
    std::get<0>(onlineResult).result();
    std::get<1>(onlineResult).result();
    const auto onlineEnd = Clock::now();
    const u64 onlineBytes = sockets[0].bytesReceived() + sockets[1].bytesReceived() - onlineBytesBefore;

    if (protocols[0].mNativeFoldedOutput.rows() != kC * kC
        || protocols[0].mNativeFoldedOutput.cols() < kN
        || protocols[1].mNativeFoldedOutput.rows() != kC * kC
        || protocols[1].mNativeFoldedOutput.cols() < kN) {
        throw std::runtime_error("native folded capture has unexpected dimensions");
    }

    u64 duplicateTerms = 0;
    const Reference reference = buildReference(positions, coefficients, duplicateTerms);
    if (duplicateTerms == 0) throw std::runtime_error("duplicate fixture did not contain collisions");
    u64 mismatchRow = 0, mismatchIndex = 0;
    if (!validateFullDomain(protocols, reference, mismatchRow, mismatchIndex)) {
        throw std::runtime_error("full-domain p0 differential mismatch at row "
            + std::to_string(mismatchRow) + ", index " + std::to_string(mismatchIndex));
    }

    const Field original = protocols[0].mNativeFoldedOutput(0, 0);
    protocols[0].mNativeFoldedOutput(0, 0) += Field(1);
    const bool corruptionAccepted = validateFullDomain(protocols, reference, mismatchRow, mismatchIndex);
    protocols[0].mNativeFoldedOutput(0, 0) = original;
    if (corruptionAccepted) throw std::runtime_error("corruption control was not rejected");
    if (!validateFullDomain(protocols, reference, mismatchRow, mismatchIndex)) {
        throw std::runtime_error("restored output failed differential validation");
    }
    const auto totalEnd = Clock::now();

    const std::string tempPath = reportPath + ".tmp";
    std::ofstream out(tempPath, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write report: " + tempPath);
    out << "{\n"
        << "  \"schema_version\": \"reverse-cuckoo-p0-baseline-1\",\n"
        << "  \"date\": \"2026-08-04\",\n"
        << "  \"status\": \"complete\",\n"
        << "  \"artifact\": {\n"
        << "    \"libote_commit\": \"edb5d32822eabf2dda9f6844d85d0ce2e402cdd5\",\n"
        << "    \"libote_license\": \"MIT\",\n"
        << "    \"paper_repo_commit\": \"b55bcc4696d10e57bdea8c282a851fdd4fad0c2b\",\n"
        << "    \"paper_repo_code_license\": null\n"
        << "  },\n"
        << "  \"parameters\": {\"n\": 1048576, \"c\": 4, \"t\": 16, "
        << "\"p0\": \"4611686018326724609\", \"coefficient_bits\": 62},\n"
        << "  \"layout\": {\n"
        << "    \"name\": \"libote_native_16_folded_raw\",\n"
        << "    \"sets\": 256, \"points_per_set\": 16, \"point_terms\": 4096,\n"
        << "    \"comparable_to_raw_31_diagonal_timing\": false\n"
        << "  },\n"
        << "  \"controls\": {\n"
        << "    \"full_domain_positions_checked\": " << (kC * kC * kN) << ",\n"
        << "    \"collision_accumulating_reference\": true,\n"
        << "    \"duplicate_terms_accumulated\": " << duplicateTerms << ",\n"
        << "    \"corruption_rejected\": true\n"
        << "  },\n"
        << "  \"metrics\": {\n"
        << "    \"setup_us\": " << elapsedUs(setupBegin, setupEnd) << ",\n"
        << "    \"online_full_domain_us\": " << elapsedUs(onlineBegin, onlineEnd) << ",\n"
        << "    \"end_to_end_protocol_us\": " << elapsedUs(totalBegin, onlineEnd) << ",\n"
        << "    \"end_to_end_including_validation_us\": " << elapsedUs(totalBegin, totalEnd) << ",\n"
        << "    \"setup_wire_bytes\": " << setupBytes << ",\n"
        << "    \"online_wire_bytes\": " << onlineBytes << ",\n"
        << "    \"end_to_end_protocol_wire_bytes\": " << (setupBytes + onlineBytes) << "\n"
        << "  },\n"
        << "  \"claims\": {\"performance_speedup\": null, \"security_level\": null}\n"
        << "}\n";
    out.close();
    if (!out) throw std::runtime_error("failed while writing report: " + tempPath);
    if (std::rename(tempPath.c_str(), reportPath.c_str()) != 0) {
        throw std::runtime_error("cannot atomically install report: " + reportPath);
    }
    std::cout << reportPath << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    const std::string reportPath = argc > 1 ? argv[1] : "reverse_cuckoo_p0_baseline_2026_08_04.json";
    try {
        return run(reportPath);
    } catch (const std::exception& error) {
        try { writeFailure(reportPath, error.what()); } catch (...) {}
        std::cerr << "unsupported: " << error.what() << '\n';
        return 2;
    }
}
