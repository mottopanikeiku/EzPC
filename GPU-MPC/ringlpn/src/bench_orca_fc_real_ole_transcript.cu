// Real-OLE dealerless FC transcript (Step 5 of the dealerless roadmap).
//
// Replaces the ideal-OLE oracle of orca_fc_ideal_ole_transcript.cuh with the
// real Figure 2 Ring-LPN OLE engine (bench_ole_ringlpn_cuda.cu): each Beaver
// cross term is derandomized against a slot of a *random ring OLE* produced by
// SPFSS expansion over Z_p[X]/(X^n+1). Because the deployed primes fully split
// the ring, the forward negacyclic NTT is a slot isomorphism R_p -> Z_p^n, so
// ONE ring OLE per (direction, limb) yields up to n independent scalar OLEs.
// This is the dense-packing amortization the constant-polynomial artifacts
// could not claim: ring_ole_instances = 2 * limbs regardless of M*K*N (as long
// as batch*M*K*N <= n), versus 2*M*K*N ideal-OLE calls in the Step-1 artifact.
//
// Party separation (transcript discipline; single process, party-tagged state):
//   P0-local: A0, B0, Y0 sampling; d-openings from its OLE slot shares;
//             its Beaver accumulation z0 and final c0 key entries.
//   P1-local: A1, B1, Y1; e-openings; z1; c1.
//   Opened:   per cross term and limb, d = a - X0 and e = b - X1 (the standard
//             OLE derandomization messages; X_sigma is pseudorandom under
//             splittable Ring-LPN, so the openings hide the share operands).
//
// Remaining oracle boundaries (unchanged from the roadmap, stated honestly):
//   1. SPFSS key generation is centralized: build_spfss_keys() sees both
//      parties' noise vectors (distributed DPF keygen via OT is future work).
//   2. The Z_M -> Z_{2^bw} export uses the exact carry-correction oracle
//      exactZmToRingShares() which reads both shares (the secure protocol is
//      prototyped host-side in test_secure_convert.cpp, not yet wired here).
//   3. c (compression) and t (noise weight) default to correctness parameters
//      (c=2, t=8), not audited security parameters.
//
// q64 runs one q62 limb; q128 runs two q62 CRT limbs, each party lifting its
// per-limb Beaver shares to Z_M (M = p0*p1) locally via Garner recombination
// (the lift is linear, so additive shares lift to additive shares).
//
// Validation: engine-internal z0+z1 == x0*x1 (GPU + host oracle), slot-domain
// OLE identity on used slots, derandomized cross-term identity, and the full
// Orca contract: party key buffers (A_i || B_i || C_i) drive the unchanged
// gpuMatmulBeaver online path and must reconstruct clear-matmul + mask_Y.

#include <cuda_runtime.h>

#define RINGLPN_OLE_DISABLE_MAIN 1
#ifndef RINGLPN_DEVICE_LABEL
#define RINGLPN_DEVICE_LABEL "cuda_orca_fc_real_ole_transcript"
#endif
#include "ringlpn/src/bench_ole_ringlpn_cuda.cu"

#include "fss/gpu_matmul.h"
#include "ringlpn/src/orca_fc_ringlpn_keywriter.cuh"
#include "utils/gpu_mem.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using T = u64;
using ringlpn_orca::u128;

struct RealOleArgs {
    int rows = 2;
    int inner = 2;
    int cols = 2;
    int bw = 16;
    int qbits = 64;
    int ole_n = 8192;
    int ole_c = 2;
    int ole_t = 8;
    std::string noise = "uniform";
    uint64_t seed = 1;
    bool csv_header = false;
};

struct RealOleCounters {
    uint64_t ring_ole_instances = 0;  // Figure 2 engine runs (2 directions * limbs)
    uint64_t ideal_equiv_ole = 0;     // 2*M*K*N: scalar OLE calls the ideal artifact needed
    uint64_t slots_used = 0;          // cross terms packed per direction (batch*M*K*N)
    uint64_t slot_capacity = 0;       // n slots per ring OLE
    uint64_t opened_words = 0;        // derandomization openings (d,e) in Z_p words
    uint64_t conversions = 0;         // Z_M -> Z_{2^bw} conversions (batch*M*N)
    bool bound_ok = true;
    bool engine_valid = true;         // engine z0+z1 == x0*x1 (GPU + host oracle)
    bool slot_identity_ok = true;     // Z0+Z1 == X0*X1 on used slots (test-only check)
    bool cross_identity_ok = true;    // u0+u1 == a*b mod p per cross term (test-only check)
    size_t pair_key_bytes = 0;
    double keygen_us = 0.0;
    double expand_us = 0.0;
};

// One ring OLE in the slot (evaluation) domain: per slot s,
// X0[s]*X1[s] == Z0[s] + Z1[s] (mod modulus).
struct SlotOle {
    std::vector<Word> X0, X1, Z0, Z1;
    Word modulus = 0;
};

// Conservative no-wrap bound, generalized to the (possibly CRT) modulus M:
// shares are < 2^bw, so the integer Beaver sum per output is < K * 2^(2*bw+2).
static bool real_no_wrap_bound(int K, int bw, u128 modulus) {
    if (bw <= 2 || bw > 32) {
        return false;
    }
    return (u128(K) << (2 * bw + 2)) < modulus;
}

// Garner lift of per-limb residues to Z_{p0*p1}. Linear in (r0, r1), so each
// party lifts its own additive shares locally and the shares stay additive.
static u128 garner_lift_q128(Word r0, Word r1) {
    constexpr Word p0 = ringlpn_orca::kPrime62;
    constexpr Word p1 = ringlpn_orca::kPrime62Crt2;
    static const Word inv_p0_mod_p1 = mod_inv<Word>(p0 % p1, p1);
    const Word diff = mod_sub<Word>(r1 % p1, r0 % p1, p1);
    const Word t = mod_mul_host<Word>(diff, inv_p0_mod_p1, p1);
    return u128(r0) + u128(p0) * t;
}

// Runs the Figure 2 engine for one limb and transforms the random ring OLE
// (x0, z0; x1, z1) into the slot domain via the forward negacyclic NTT.
static bool build_slot_ole(const OleArgs &engine_args,
                           const ModulusConfig<Word> &config,
                           int limb_index,
                           AESGlobalContext *gaes,
                           SlotOle &out,
                           RealOleCounters &counters) {
    OleState state;
    init_ole_state(state, engine_args, config, limb_index);
    OleLimbResult limb = run_initial_ole_limb(state, gaes);
    counters.engine_valid = counters.engine_valid && limb.correct;
    counters.pair_key_bytes += limb.key_bytes;
    counters.keygen_us += limb.keygen_us;

    auto expand_start = Clock::now();
    run_x_phase(state);
    run_spfss_eval_phase(state, gaes);
    run_z_phase(state);
    check(cudaDeviceSynchronize(), "sync timed expand");
    auto expand_end = Clock::now();
    counters.expand_us += static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(expand_end - expand_start)
            .count());

    const int n = engine_args.n;
    const int log_degree = state.log_degree;
    out.modulus = config.modulus;
    out.X0.resize(n);
    out.X1.resize(n);
    out.Z0.resize(n);
    out.Z1.resize(n);
    check(cudaMemcpy(out.X0.data(), state.d_x0, sizeof(Word) * n, cudaMemcpyDeviceToHost),
          "copy x0 for slots");
    check(cudaMemcpy(out.X1.data(), state.d_x1, sizeof(Word) * n, cudaMemcpyDeviceToHost),
          "copy x1 for slots");
    check(cudaMemcpy(out.Z0.data(), state.d_z0, sizeof(Word) * n, cudaMemcpyDeviceToHost),
          "copy z0 for slots");
    check(cudaMemcpy(out.Z1.data(), state.d_z1, sizeof(Word) * n, cudaMemcpyDeviceToHost),
          "copy z1 for slots");

    // P0 transforms (x0, z0); P1 transforms (x1, z1). The transform is public
    // and local. After it, Z0[s] + Z1[s] == X0[s] * X1[s] per slot.
    host_forward_ntt(out.X0, state.phi_norm, n, log_degree, config);
    host_forward_ntt(out.Z0, state.phi_norm, n, log_degree, config);
    host_forward_ntt(out.X1, state.phi_norm, n, log_degree, config);
    host_forward_ntt(out.Z1, state.phi_norm, n, log_degree, config);

    state.cleanup();
    return limb.correct;
}

// Builds party-local A, B, C key shares for one matmul from real-OLE-backed
// Beaver cross terms. Mirrors buildIdealOleTranscript's interface so the same
// gpuMatmulBeaver harness validates both artifacts.
static bool build_real_ole_transcript(const MatmulParams &p,
                                      const RealOleArgs &args,
                                      std::vector<T> &a0,
                                      std::vector<T> &a1,
                                      std::vector<T> &b0,
                                      std::vector<T> &b1,
                                      std::vector<T> &c0,
                                      std::vector<T> &c1,
                                      std::vector<T> &mask_a,
                                      std::vector<T> &mask_b,
                                      std::vector<T> &mask_y,
                                      AESGlobalContext *gaes,
                                      RealOleCounters &counters) {
    const int bw = p.bw;
    const u128 modulus = ringlpn_orca::modulusForQbits(args.qbits);
    counters.bound_ok = real_no_wrap_bound(p.K, bw, modulus);
    if (!counters.bound_ok) {
        return false;
    }

    const uint64_t cross_terms =
        static_cast<uint64_t>(p.batchSz) * p.M * p.K * p.N;
    counters.slots_used = cross_terms;
    counters.slot_capacity = static_cast<uint64_t>(args.ole_n);
    counters.ideal_equiv_ole = 2 * cross_terms;
    if (cross_terms > static_cast<uint64_t>(args.ole_n)) {
        // One ring OLE per direction is the claim under test; larger layers
        // would batch ceil(cross_terms / n) ring OLEs per direction.
        return false;
    }

    // Each party samples its Z_{2^bw} mask shares locally.
    std::mt19937_64 rng0(ringlpn_orca::mixSeed(args.seed, 0xF0));  // P0-local
    std::mt19937_64 rng1(ringlpn_orca::mixSeed(args.seed, 0xF1));  // P1-local
    std::uniform_int_distribution<uint64_t> ring_dist(0, ringlpn_orca::ringMask(bw));
    auto sample_shares = [&](size_t count,
                             std::vector<T> &s0,
                             std::vector<T> &s1,
                             std::vector<T> &mask) {
        s0.resize(count);
        s1.resize(count);
        mask.resize(count);
        for (size_t i = 0; i < count; ++i) {
            s0[i] = static_cast<T>(ring_dist(rng0));
            s1[i] = static_cast<T>(ring_dist(rng1));
            mask[i] = static_cast<T>(ringlpn_orca::ringAdd(
                static_cast<uint64_t>(s0[i]), static_cast<uint64_t>(s1[i]), bw));
        }
    };
    std::vector<T> y0, y1;
    sample_shares(static_cast<size_t>(p.size_A), a0, a1, mask_a);
    sample_shares(static_cast<size_t>(p.size_B), b0, b1, mask_b);
    sample_shares(static_cast<size_t>(p.size_C), y0, y1, mask_y);

    // Run the Figure 2 engine: one ring OLE per (direction, limb).
    //   direction 0 computes shares of A0*B1 (P0 inputs a-values, P1 b-values);
    //   direction 1 computes shares of A1*B0 (P0 inputs b-values, P1 a-values).
    const std::vector<ModulusConfig<Word>> configs = ole_modulus_configs(args.qbits);
    const size_t limbs = configs.size();
    std::vector<std::vector<SlotOle>> slot_ole(2, std::vector<SlotOle>(limbs));
    for (int dir = 0; dir < 2; ++dir) {
        OleArgs engine_args;
        engine_args.n = args.ole_n;
        engine_args.c = args.ole_c;
        engine_args.t = args.ole_t;
        engine_args.qbits = args.qbits;
        engine_args.noise = args.noise;
        engine_args.chunk_size = args.ole_n;
        engine_args.seed = ringlpn_orca::mixSeed(args.seed, 0xA0 + dir);
        for (size_t limb = 0; limb < limbs; ++limb) {
            counters.ring_ole_instances++;
            if (!build_slot_ole(engine_args, configs[limb], static_cast<int>(limb),
                                gaes, slot_ole[dir][limb], counters)) {
                return false;
            }
        }
    }

    // Test-only check (reads both parties' slot shares): the slot OLE identity.
    for (int dir = 0; dir < 2 && counters.slot_identity_ok; ++dir) {
        for (size_t limb = 0; limb < limbs; ++limb) {
            const SlotOle &ole = slot_ole[dir][limb];
            for (uint64_t s = 0; s < cross_terms; ++s) {
                const Word expect = mod_mul_host<Word>(ole.X0[s], ole.X1[s], ole.modulus);
                const Word got = mod_add<Word>(ole.Z0[s], ole.Z1[s], ole.modulus);
                if (expect != got) {
                    counters.slot_identity_ok = false;
                    break;
                }
            }
        }
    }

    // Derandomize each cross term against its slot, per limb:
    //   P0 opens d = a - X0; P1 opens e = b - X1;
    //   P0's share u0 = d*e + e*X0 + Z0; P1's share u1 = d*X1 + Z1;
    //   u0 + u1 == a*b (mod p).
    // Accumulate per-output Beaver sums per limb, then lift to Z_M and convert.
    c0.assign(static_cast<size_t>(p.size_C), T(0));
    c1.assign(static_cast<size_t>(p.size_C), T(0));
    counters.conversions = 0;
    counters.opened_words = 0;

    auto cross_share = [&](int dir, size_t limb, uint64_t slot, Word a_val, Word b_val,
                           Word &share0, Word &share1) {
        const SlotOle &ole = slot_ole[dir][limb];
        const Word q = ole.modulus;
        const Word d = mod_sub<Word>(a_val % q, ole.X0[slot], q);  // P0 opens
        const Word e = mod_sub<Word>(b_val % q, ole.X1[slot], q);  // P1 opens
        counters.opened_words += 2;
        const Word de = mod_mul_host<Word>(d, e, q);
        const Word ex0 = mod_mul_host<Word>(e, ole.X0[slot], q);
        share0 = mod_add<Word>(mod_add<Word>(de, ex0, q), ole.Z0[slot], q);  // P0-local
        const Word dx1 = mod_mul_host<Word>(d, ole.X1[slot], q);
        share1 = mod_add<Word>(dx1, ole.Z1[slot], q);  // P1-local
        if (counters.cross_identity_ok) {
            const Word expect = mod_mul_host<Word>(a_val % q, b_val % q, q);
            if (mod_add<Word>(share0, share1, q) != expect) {
                counters.cross_identity_ok = false;
            }
        }
        return;
    };

    for (int batch = 0; batch < p.batchSz; ++batch) {
        const size_t a_base = static_cast<size_t>(batch) * p.stride_A;
        const size_t b_base = static_cast<size_t>(batch) * p.stride_B;
        const size_t c_base = static_cast<size_t>(batch) * p.stride_C;
        for (int row = 0; row < p.M; ++row) {
            for (int col = 0; col < p.N; ++col) {
                std::vector<Word> z0_limb(limbs, 0);
                std::vector<Word> z1_limb(limbs, 0);
                for (int k = 0; k < p.K; ++k) {
                    const uint64_t av0 = ringlpn_orca::matrixValue(
                        a0, a_base, p.M, p.K, p.rowMaj_A, row, k, bw);
                    const uint64_t av1 = ringlpn_orca::matrixValue(
                        a1, a_base, p.M, p.K, p.rowMaj_A, row, k, bw);
                    const uint64_t bv0 = ringlpn_orca::matrixValue(
                        b0, b_base, p.K, p.N, p.rowMaj_B, k, col, bw);
                    const uint64_t bv1 = ringlpn_orca::matrixValue(
                        b1, b_base, p.K, p.N, p.rowMaj_B, k, col, bw);
                    const uint64_t slot =
                        ((static_cast<uint64_t>(batch) * p.M + row) * p.N + col) * p.K + k;
                    for (size_t limb = 0; limb < limbs; ++limb) {
                        const Word q = configs[limb].modulus;
                        // Local terms: each party multiplies its own shares.
                        z0_limb[limb] = mod_add<Word>(
                            z0_limb[limb], mod_mul_host<Word>(av0 % q, bv0 % q, q), q);
                        z1_limb[limb] = mod_add<Word>(
                            z1_limb[limb], mod_mul_host<Word>(av1 % q, bv1 % q, q), q);
                        // Cross terms from the real ring OLE slots.
                        Word u0 = 0, u1 = 0, v0 = 0, v1 = 0;
                        cross_share(0, limb, slot, static_cast<Word>(av0),
                                    static_cast<Word>(bv1), u0, u1);
                        cross_share(1, limb, slot, static_cast<Word>(bv0),
                                    static_cast<Word>(av1), v0, v1);
                        z0_limb[limb] = mod_add<Word>(z0_limb[limb], mod_add<Word>(u0, v0, q), q);
                        z1_limb[limb] = mod_add<Word>(z1_limb[limb], mod_add<Word>(u1, v1, q), q);
                    }
                }
                // Each party lifts its per-limb shares to Z_M locally.
                u128 z0 = 0;
                u128 z1 = 0;
                if (args.qbits == 128) {
                    z0 = garner_lift_q128(z0_limb[0], z0_limb[1]);
                    z1 = garner_lift_q128(z1_limb[0], z1_limb[1]);
                } else {
                    z0 = z0_limb[0];
                    z1 = z1_limb[0];
                }
                // Conversion oracle boundary (see header): exact carry correction.
                uint64_t r0 = 0;
                uint64_t r1 = 0;
                ringlpn_orca::exactZmToRingShares(z0, z1, modulus, bw, r0, r1);
                ++counters.conversions;
                const size_t c_idx = c_base + static_cast<size_t>(row) * p.N + col;
                c0[c_idx] = static_cast<T>(ringlpn_orca::ringAdd(
                    r0, ringlpn_orca::ringValue(y0[c_idx], bw), bw));
                c1[c_idx] = static_cast<T>(ringlpn_orca::ringAdd(
                    r1, ringlpn_orca::ringValue(y1[c_idx], bw), bw));
            }
        }
    }
    return counters.slot_identity_ok && counters.cross_identity_ok;
}

struct RealCaseResult {
    bool transcript_ok = false;
    bool key_order_ok = false;
    bool masks_consistent = false;
    bool online_ok = false;
    RealOleCounters counters;
};

static MatmulParams make_real_matmul_params(const RealOleArgs &args) {
    MatmulParams p;
    p.batchSz = 1;
    p.M = args.rows;
    p.K = args.inner;
    p.N = args.cols;
    stdInit(p, args.bw, 0);
    return p;
}

static void real_copy_to_gpu(const std::vector<T> &src, T **dst) {
    check(cudaMalloc(reinterpret_cast<void **>(dst), src.size() * sizeof(T)),
          "real transcript cudaMalloc");
    check(cudaMemcpy(*dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice),
          "real transcript H2D");
}

static RealCaseResult run_real_case(const RealOleArgs &args, AESGlobalContext *gaes) {
    RealCaseResult result;
    MatmulParams p = make_real_matmul_params(args);

    std::vector<T> a0, a1, b0, b1, c0, c1, mask_a, mask_b, mask_y;
    result.transcript_ok = build_real_ole_transcript(
        p, args, a0, a1, b0, b1, c0, c1, mask_a, mask_b, mask_y, gaes, result.counters);
    if (!result.transcript_ok) {
        return result;
    }

    result.masks_consistent = true;
    for (int i = 0; i < p.size_A; ++i) {
        if (ringlpn_orca::ringAdd(static_cast<uint64_t>(a0[i]),
                                  static_cast<uint64_t>(a1[i]), args.bw) !=
            ringlpn_orca::ringValue(mask_a[i], args.bw)) {
            result.masks_consistent = false;
        }
    }

    // Assemble party-local key buffers in Orca's A || B || C byte order.
    std::vector<uint8_t> key0;
    std::vector<uint8_t> key1;
    auto append_words = [](std::vector<uint8_t> &buf, const std::vector<T> &src) {
        const size_t off = buf.size();
        buf.resize(off + src.size() * sizeof(T));
        std::memcpy(buf.data() + off, src.data(), src.size() * sizeof(T));
    };
    append_words(key0, a0);
    append_words(key0, b0);
    append_words(key0, c0);
    append_words(key1, a1);
    append_words(key1, b1);
    append_words(key1, c1);

    // Clear inputs, masked online inputs, expected masked output.
    std::mt19937_64 rng(args.seed ^ 0x0DEA11E5ULL);
    std::uniform_int_distribution<uint64_t> ring_dist(0, ringlpn_orca::ringMask(args.bw));
    std::vector<T> input(p.size_A);
    std::vector<T> weight(p.size_B);
    std::vector<T> masked_input(p.size_A);
    std::vector<T> masked_weight(p.size_B);
    for (int i = 0; i < p.size_A; ++i) {
        input[i] = static_cast<T>(ring_dist(rng));
        masked_input[i] = static_cast<T>(ringlpn_orca::ringAdd(
            static_cast<uint64_t>(input[i]), static_cast<uint64_t>(mask_a[i]), args.bw));
    }
    for (int i = 0; i < p.size_B; ++i) {
        weight[i] = static_cast<T>(ring_dist(rng));
        masked_weight[i] = static_cast<T>(ringlpn_orca::ringAdd(
            static_cast<uint64_t>(weight[i]), static_cast<uint64_t>(mask_b[i]), args.bw));
    }
    std::vector<T> expected(p.size_C);
    for (int row = 0; row < args.rows; ++row) {
        for (int col = 0; col < args.cols; ++col) {
            u128 acc = 0;
            for (int k = 0; k < args.inner; ++k) {
                acc += u128(input[static_cast<size_t>(row) * args.inner + k]) *
                       weight[static_cast<size_t>(k) * args.cols + col];
            }
            const size_t idx = static_cast<size_t>(row) * args.cols + col;
            expected[idx] = static_cast<T>(ringlpn_orca::ringAdd(
                ringlpn_orca::ringReduce(acc, args.bw),
                ringlpn_orca::ringValue(mask_y[idx], args.bw), args.bw));
        }
    }

    // Read keys back and run the unchanged Beaver matmul online path.
    uint8_t *key_ptr0 = key0.data();
    uint8_t *key_ptr1 = key1.data();
    GPUMatmulKey<T> gkey0 = readGPUMatmulKey<T>(p, TruncateType::None, &key_ptr0);
    GPUMatmulKey<T> gkey1 = readGPUMatmulKey<T>(p, TruncateType::None, &key_ptr1);
    result.key_order_ok = key_ptr0 == key0.data() + key0.size() &&
                          key_ptr1 == key1.data() + key1.size();

    T *d_x = nullptr;
    T *d_w = nullptr;
    T *d_a0 = nullptr;
    T *d_a1 = nullptr;
    T *d_b0 = nullptr;
    T *d_b1 = nullptr;
    real_copy_to_gpu(masked_input, &d_x);
    real_copy_to_gpu(masked_weight, &d_w);
    real_copy_to_gpu(std::vector<T>(gkey0.A, gkey0.A + p.size_A), &d_a0);
    real_copy_to_gpu(std::vector<T>(gkey1.A, gkey1.A + p.size_A), &d_a1);
    real_copy_to_gpu(std::vector<T>(gkey0.B, gkey0.B + p.size_B), &d_b0);
    real_copy_to_gpu(std::vector<T>(gkey1.B, gkey1.B + p.size_B), &d_b1);

    Stats stats0;
    Stats stats1;
    T *d_o0 = gpuMatmulBeaver<T>(p, gkey0, SERVER0, d_x, d_w, d_a0, d_b0, nullptr, &stats0);
    T *d_o1 = gpuMatmulBeaver<T>(p, gkey1, SERVER1, d_x, d_w, d_a1, d_b1, nullptr, &stats1);
    std::vector<T> o0(p.size_C);
    std::vector<T> o1(p.size_C);
    check(cudaMemcpy(o0.data(), d_o0, p.size_C * sizeof(T), cudaMemcpyDeviceToHost),
          "copy online out0");
    check(cudaMemcpy(o1.data(), d_o1, p.size_C * sizeof(T), cudaMemcpyDeviceToHost),
          "copy online out1");

    bool reconstructed_ok = true;
    for (int i = 0; i < p.size_C; ++i) {
        if (ringlpn_orca::ringAdd(static_cast<uint64_t>(o0[i]),
                                  static_cast<uint64_t>(o1[i]), args.bw) !=
            ringlpn_orca::ringValue(expected[i], args.bw)) {
            reconstructed_ok = false;
        }
    }

    result.online_ok = result.key_order_ok && result.masks_consistent &&
                       result.counters.engine_valid &&
                       result.counters.slot_identity_ok &&
                       result.counters.cross_identity_ok && reconstructed_ok;

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_a0);
    cudaFree(d_a1);
    cudaFree(d_b0);
    cudaFree(d_b1);
    gpuFree(d_o0);
    gpuFree(d_o1);
    return result;
}

static void real_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " [--rows M] [--inner K] [--cols N] [--bw N] [--qbits 64|128]"
              << " [--ole-n N] [--ole-c N] [--ole-t N] [--noise uniform|regular]"
              << " [--seed N] [--csv-header]\n";
}

static RealOleArgs parse_real_args(int argc, char **argv) {
    RealOleArgs args;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) {
            args.rows = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--inner") && i + 1 < argc) {
            args.inner = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--cols") && i + 1 < argc) {
            args.cols = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--bw") && i + 1 < argc) {
            args.bw = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--qbits") && i + 1 < argc) {
            args.qbits = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--ole-n") && i + 1 < argc) {
            args.ole_n = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--ole-c") && i + 1 < argc) {
            args.ole_c = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--ole-t") && i + 1 < argc) {
            args.ole_t = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--noise") && i + 1 < argc) {
            args.noise = argv[++i];
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            args.seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--csv-header")) {
            args.csv_header = true;
        } else {
            real_usage(argv[0]);
            std::exit(1);
        }
    }
    const bool qbits_ok = args.qbits == 64 || args.qbits == 128;
    const int max_bw = args.qbits == 128 ? 32 : 28;
    if (args.rows <= 0 || args.inner <= 0 || args.cols <= 0 || args.rows > 64 ||
        args.inner > 64 || args.cols > 64 || args.bw <= 2 || args.bw > max_bw ||
        !qbits_ok || !is_power_of_two(args.ole_n) ||
        (args.noise != "uniform" && args.noise != "regular")) {
        real_usage(argv[0]);
        std::exit(1);
    }
    return args;
}

}  // namespace

int main(int argc, char **argv) {
    RealOleArgs args = parse_real_args(argc, argv);
    initGPUMemPool();
    AESGlobalContext gaes;
    initAESContext(&gaes);

    RealCaseResult r = run_real_case(args, &gaes);
    freeAESGlobalContext(&gaes);

    if (args.csv_header) {
        std::cout
            << "device,mode,requested_qbits,actual_qbits,noise_mode,ole_n,ole_c,ole_t,seed,"
            << "rows,inner,cols,bw,no_wrap_bound,ring_ole_instances,ideal_equiv_ole,"
            << "slots_used,slot_capacity,opened_words,conversions,engine_validation,"
            << "slot_identity,cross_identity,spfss_pair_key_bytes,spfss_keygen_us,"
            << "ole_expand_us,transcript_built,masks_consistent,key_order,online_contract,"
            << "validation\n";
    }
    std::cout << RINGLPN_DEVICE_LABEL << ",real_ole_q" << args.qbits << "_slot_packed,"
              << args.qbits << "," << ringlpn_orca::actualQbitsForQbits(args.qbits) << ","
              << args.noise << "," << args.ole_n << "," << args.ole_c << "," << args.ole_t
              << "," << args.seed << "," << args.rows << "," << args.inner << ","
              << args.cols << "," << args.bw << "," << (r.counters.bound_ok ? 1 : 0) << ","
              << r.counters.ring_ole_instances << "," << r.counters.ideal_equiv_ole << ","
              << r.counters.slots_used << "," << r.counters.slot_capacity << ","
              << r.counters.opened_words << "," << r.counters.conversions << ","
              << (r.counters.engine_valid ? "pass" : "fail") << ","
              << (r.counters.slot_identity_ok ? 1 : 0) << ","
              << (r.counters.cross_identity_ok ? 1 : 0) << ","
              << r.counters.pair_key_bytes << "," << r.counters.keygen_us << ","
              << r.counters.expand_us << "," << (r.transcript_ok ? 1 : 0) << ","
              << (r.masks_consistent ? 1 : 0) << "," << (r.key_order_ok ? 1 : 0) << ","
              << (r.online_ok ? "pass" : "fail") << "," << (r.online_ok ? "pass" : "fail")
              << "\n";

    return r.online_ok ? 0 : 2;
}
