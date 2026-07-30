// gpu_aes_prg_host.h - host implementation of the GPU DPF expansion PRG,
// bit-identical to the device `aes_prg_expand` in src/gpu_spfss_zp.cuh.
//
// WHY: the two-party host keygen must emit keys that the UNCHANGED GPU
// evaluator (`ringlpn_spfss_zp::gpuDpfZpFullEvalSum`) accepts, so its expansion
// has to agree with the device PRG on every bit.
//
// EXACT SEMANTICS (derived from GPU-MPC/fss/gpu_aes_shm.cu
// `applyAESPRGTwoTimes` and verified against device-dumped vectors in
// results/dpf/gpu_aes_prg_vectors_2026_07_29.csv):
//   * the device reverses bytes inside each 32-bit word of the key before the
//     key schedule, which is exactly the identity on the little-endian byte
//     image of the 128-bit seed. So the AES-128 key is the seed's 16 bytes in
//     little-endian order, with the low bit cleared.
//   * left  = AES_k(0x00^16)
//   * right = AES_k(0x02 || 0x00^15)
//   * the same per-word byte reversal on the output makes the returned 128-bit
//     value the little-endian image of the ciphertext bytes.
//   * the child control bit is the LSB of that value and is cleared from the
//     child seed (Doerner--shelat-style low-bit control encoding, so the secret
//     seed state is 127 bits, not 128; see the S1 contract obligation D-SEED).
//
// This header makes no security claim: it reproduces the deployed GPU semantics
// so the two-party protocol can drive the unmodified GPU consumer.

#pragma once

#include <openssl/evp.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ringlpn_gpu_prg {

using U128 = unsigned __int128;

inline U128 clear_tag(U128 seed) { return seed & ~static_cast<U128>(1); }

// One AES-128-ECB block encryption, no padding.
inline void aes128_block(const uint8_t key[16], const uint8_t in[16],
                         uint8_t out[16]) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (ctx == nullptr) throw std::runtime_error("EVP_CIPHER_CTX_new failed");
    if (EVP_EncryptInit_ex(ctx, EVP_aes_128_ecb(), nullptr, key, nullptr) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw std::runtime_error("EVP_EncryptInit_ex failed");
    }
    EVP_CIPHER_CTX_set_padding(ctx, 0);
    int len = 0;
    if (EVP_EncryptUpdate(ctx, out, &len, in, 16) != 1 || len != 16) {
        EVP_CIPHER_CTX_free(ctx);
        throw std::runtime_error("EVP_EncryptUpdate failed");
    }
    EVP_CIPHER_CTX_free(ctx);
}

// Bit-identical host twin of the device `aes_prg_expand`.
inline void gpu_aes_prg_expand(U128 seed, U128 &s_l, uint8_t &t_l, U128 &s_r,
                               uint8_t &t_r) {
    uint8_t key[16];
    const U128 masked = clear_tag(seed);
    std::memcpy(key, &masked, 16);

    uint8_t pt_left[16];
    uint8_t pt_right[16];
    std::memset(pt_left, 0, sizeof(pt_left));
    std::memset(pt_right, 0, sizeof(pt_right));
    pt_right[0] = 0x02;  // device sets plaintext byte 3 of word 0, i.e. AES byte 0

    uint8_t left[16], right[16];
    aes128_block(key, pt_left, left);
    aes128_block(key, pt_right, right);

    U128 lv = 0, rv = 0;
    std::memcpy(&lv, left, 16);
    std::memcpy(&rv, right, 16);
    t_l = static_cast<uint8_t>(lv & 1);
    t_r = static_cast<uint8_t>(rv & 1);
    s_l = clear_tag(lv);
    s_r = clear_tag(rv);
}

}  // namespace ringlpn_gpu_prg
