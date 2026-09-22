// Experimental same-function baseline; never selected by the live adapter.
// Gilboa OLE computes both cross products directly. The existing conversion,
// freshness, authenticated transport, private record writer and stock checker
// are reused. Compatibility headers retain the REFERENCE Ring-LPN plan;
// these are not this baseline's cryptographic parameters or executed work.
#include "two_party_linear_preprocess.cuh"

namespace direct_ot_fc {
using namespace ringlpn_fc_live;
using ringlpn_fc_live::Args;
constexpr uint64_t kOleBatch = 4096;
constexpr char kProtocol[] = "direct-gilboa-ot-fc-v1";

bool make_plan(const Args &args, const PublicWork &work,
               ringlpn_freshness::Digest &layer,
               ringlpn_freshness::Digest &plan, uint64_t &oles) {
    if (!checked_mul(work.cross_terms, 2 * work.limbs, oles)) return false;
    const auto manifest = encode_preflight(args, {});
    std::vector<uint8_t> bytes(kProtocol, kProtocol + sizeof(kProtocol) - 1);
    bytes.insert(bytes.end(), manifest.begin(), manifest.end());
    ringlpn_freshness::put_u64_be(bytes, args.layer_ordinal);
    if (!ringlpn_freshness::digest(bytes.data(), bytes.size(), layer)) return false;
    ringlpn_freshness::put_u64_be(bytes, kOleBatch);
    ringlpn_freshness::put_u64_be(bytes, oles);
    ringlpn_freshness::put_u64_be(bytes, work.size_c);
    ringlpn_freshness::put_u64_be(bytes, ringlpn_2pc::kMaxSecureConvertBatch);
    return ringlpn_freshness::digest(bytes.data(), bytes.size(), plan);
}

bool disjoint_paths(const Args &args) {
    const auto disjoint = [](const std::filesystem::path &a,
                             const std::filesystem::path &b) {
        return std::mismatch(a.begin(), a.end(), b.begin(), b.end()).first != a.end() &&
               std::mismatch(b.begin(), b.end(), a.begin(), a.end()).first != b.end();
    };
    const auto ledger = std::filesystem::path(args.ledger_path).lexically_normal();
    const auto output = std::filesystem::absolute(record_path(args.out_prefix, args.party)).lexically_normal();
    if (!disjoint(ledger, output)) return false;
    if (args.state_record.empty()) return true;
    const auto state = std::filesystem::absolute(args.state_record).lexically_normal();
    return disjoint(ledger, state) && disjoint(output, state);
}

bool cross_products(const Args &args, const PublicWork &work,
                    const std::vector<T> &a, const std::vector<T> &b,
                    ringlpn_2pc::PartyChannel &channel,
                    ringlpn_2pc::PartyRandom &random,
                    std::vector<std::vector<Word>> &acc) {
    // The public schedule never depends on masks, support or collisions.
    std::vector<Word> operand;
    operand.reserve(kOleBatch);
    for (int direction = 0; direction < 2; ++direction) {
        for (int limb = 0; limb < work.limbs; ++limb) {
            const Word p = modulus_for_limb(limb);
            for (uint64_t start = 0; start < work.cross_terms; start += kOleBatch) {
                const size_t count = std::min(kOleBatch, work.cross_terms - start);
                operand.resize(count);
                for (size_t i = 0; i < count; ++i) {
                    const uint64_t term = start + i;
                    const uint64_t output = term / args.inner;
                    const int k = term % args.inner;
                    const int row = output / args.cols;
                    const int col = output % args.cols;
                    const bool use_a = (direction == 0) == (args.party == 0);
                    operand[i] = use_a ? a[matrix_index(work.matmul, true, row, k)] % p
                                       : b[matrix_index(work.matmul, false, k, col)] % p;
                }
                const auto product = ringlpn_2pc::ole_batch_p0_sender(channel, operand, p, random);
                if (product.size() != count) return false;
                for (size_t i = 0; i < count; ++i) {
                    const size_t output = (start + i) / args.inner;
                    acc[limb][output] = mod_add<Word>(acc[limb][output], product[i], p);
                }
            }
        }
    }
    return true;
}

int run(Args args) {
    // This baseline deliberately does not offer EMP or a new trust model.
    if (args.ot_backend != "sci-iknp") return 2;
    PublicWork work;
    ringlpn_freshness::Digest layer{}, plan{};
    uint64_t expected_oles = 0;
    if (!derive_work(args, work) || !make_plan(args, work, layer, plan, expected_oles)) return 2;
    const auto started = Clock::now();
    ringlpn_freshness::Claim claim;
    const bool local_valid =
        disjoint_paths(args) &&
        ringlpn_private_file::AtomicWriter::destination_absent(record_path(args.out_prefix, args.party)) &&
        (args.state_record.empty() || ringlpn_private_file::AtomicWriter::destination_absent(args.state_record)) &&
        ringlpn_freshness::claim_namespace_once(args.ledger_path, args.party, args.invocation_id,
                                               layer, plan, claim);
    if (!local_valid) {
        std::fprintf(stderr, "[direct-ot-fc] local admission rejected before network/OT/output\n");
        return 2;
    }
    ringlpn_2pc::ChannelAuthContext auth;
    auth.secret_file = args.channel_auth_file;
    auth.invocation_id = args.invocation_id;
    auth.claim_digest = claim.ledger_digest;
    ringlpn_2pc::PartyChannel channel(args.party, args.host, args.port, true, true,
                                    ringlpn_2pc::OtBackend::SciIknp, nullptr, &auth);
    if (!agree_preflight(channel, args, claim, true)) return 2;
    const auto setup_start = Clock::now();
    channel.setup_ots();
    const double setup_us = elapsed_us(setup_start);
    const auto protocol_start = Clock::now();
    const uint64_t protocol_begin_bytes = channel.bytes_sent();
    ringlpn_2pc::PartyRandom random;
    const auto a = sample_ring_words(work.size_a, args.bw, random);
    const auto b = sample_ring_words(work.size_b, args.bw, random);
    const auto y = sample_ring_words(work.size_c, args.bw, random);
    std::vector<std::vector<Word>> acc(work.limbs, std::vector<Word>(work.size_c));
    const auto cross_start = Clock::now();
    bool ok = accumulate_local_products(args, work, a, b, acc) &&
              cross_products(args, work, a, b, channel, random, acc);
    const double cross_us = elapsed_us(cross_start);
    const uint64_t scalar_oles = channel.costs.scalar_oles;
    const uint64_t ole_ots = channel.costs.ole_ots;
    uint64_t expected_ots = 0;
    ok = ok && checked_mul(expected_oles, 62, expected_ots) &&
         scalar_oles == expected_oles && ole_ots == expected_ots;
    Counters counters;
    std::vector<T> converted;
    if (ok) ok = convert_outputs(args, work, layer, acc, channel, random, converted, counters);
    uint8_t mine = ok ? 1 : 0, peer = 0;
    channel.exchange_bytes(&mine, &peer, 1);
    ok = ok && peer == 1;
    channel.finish_ots();
    std::vector<T> c(work.size_c);
    if (ok) {
        for (size_t i = 0; i < c.size(); ++i)
            c[i] = ringlpn_orca::ringAdd(converted[i], y[i], args.bw);
        ok = publish_record(args, work, claim, a, b, c, y, channel, counters);
    }
    ok = collect_transport_metrics(channel, counters) && ok;
    std::cout << "DIRECT_OT_FC_RESULT {\"schema_version\":1,\"protocol\":\"" << kProtocol
              << "\",\"party\":" << args.party << ",\"qbits\":" << args.qbits
              << ",\"bw\":" << args.bw << ",\"rows\":" << args.rows
              << ",\"inner\":" << args.inner << ",\"cols\":" << args.cols
              << ",\"reference_ringlpn_n\":" << args.ole_n
              << ",\"reference_ringlpn_c\":" << args.ole_c
              << ",\"reference_ringlpn_t\":" << args.ole_t
              << ",\"cross_terms\":" << work.cross_terms << ",\"scalar_oles\":" << scalar_oles
              << ",\"ole_ots\":" << ole_ots << ",\"ole_batch_limit\":" << kOleBatch
              << ",\"conversions\":" << counters.conversion.conversions
              << ",\"payload_bytes\":" << (work.size_a + work.size_b + work.size_c) * sizeof(T)
              << ",\"protocol_bytes_sent\":" << channel.bytes_sent() - protocol_begin_bytes
              << ",\"straight_bytes_sent\":" << counters.transport_straight_bytes_sent
              // SCI exposes local send counters, not independent receive counters.
              << ",\"straight_bytes_received\":null"
              << ",\"reversed_bytes_sent\":" << counters.transport_reversed_bytes_sent
              << ",\"reversed_bytes_received\":null"
              << ",\"authentication_bytes_sent\":" << channel.auth_bytes_sent()
              << ",\"base_ots\":" << counters.base_ots
              << ",\"base_ot_bytes_sent\":" << counters.base_ot_setup_bytes_sent
              << ",\"setup_us\":" << setup_us << ",\"cross_product_us\":" << cross_us
              << ",\"conversion_us\":" << counters.conversion_us
              << ",\"protocol_us\":" << elapsed_us(protocol_start)
              << ",\"total_us\":" << elapsed_us(started)
              << ",\"invocation_id\":\"" << ringlpn_freshness::hex(args.invocation_id)
              << "\",\"ledger_digest\":\"" << ringlpn_freshness::hex(claim.ledger_digest)
              << "\",\"status\":\"" << (ok ? "pass" : "FAIL") << "\"}\n";
    return ok ? 0 : 1;
}
}  // namespace direct_ot_fc

int main(int argc, char **argv) {
    try {
        ringlpn_fc_live::Args args;
        if (!ringlpn_fc_live::parse_args(argc, argv, args) || args.plan) {
            std::fprintf(stderr, "direct-ot-fc: use the existing FC party/check CLI; --plan belongs to the reference adapter\n");
            return 2;
        }
        if (args.check) return ringlpn_fc_live::run_check(args);
        return direct_ot_fc::run(args);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[direct-ot-fc] failed: %s\n", error.what());
        return 1;
    }
}
