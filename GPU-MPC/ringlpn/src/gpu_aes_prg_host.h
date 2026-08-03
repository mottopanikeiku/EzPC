// gpu_aes_prg_host.h - host implementation of the GPU DPF expansion PRG,
// bit-identical to the device `aes_prg_expand` in src/gpu_spfss_zp.cuh.
//
// WHY: the two-party host keygen must emit keys that the UNCHANGED GPU
// evaluator (`ringlpn_spfss_zp::gpuDpfZpFullEvalSum`) accepts, so its expansion
// has to agree with the device PRG on every bit.
//
// EXACT SEMANTICS (derived from GPU-MPC/fss/gpu_aes_shm.cu
// `applyAESPRGFourTimes` and verified against device-dumped vectors in
// results/dpf/gpu_aes_prg_vectors_2026_07_29.csv):
//   * the device reverses bytes inside each 32-bit word of the key before the
//     key schedule, which is exactly the identity on the little-endian byte
//     image of the 128-bit seed. The AES-128 key is therefore the seed's 16
//     bytes in little-endian order.
//   * left seed  = AES_k(0x00 || 0^120)
//   * left tag   = LSB(AES_k(0x01 || 0^120))
//   * right seed = AES_k(0x02 || 0^120)
//   * right tag  = LSB(AES_k(0x03 || 0^120))
//   * the same per-word byte reversal on each output makes the returned 128-bit
//     seed the little-endian image of the ciphertext bytes.
//
// Seeds and control bits use distinct PRF outputs: the seed state retains all
// 128 bits. This header reproduces the deployed GPU semantics; the surrounding
// protocol/security reduction remains a separate claim.

#pragma once

#include <openssl/evp.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ringlpn_gpu_prg {

using U128 = unsigned __int128;


// One AES-128-ECB context per thread: the key changes on every node but the
// cipher does not, so re-keying a persistent context avoids an allocation per
// expansion. Both seeds and both tag blocks are produced in one ECB update.
inline EVP_CIPHER_CTX *ecb_ctx() {
    static thread_local EVP_CIPHER_CTX *ctx = nullptr;
    if (ctx == nullptr) {
        ctx = EVP_CIPHER_CTX_new();
        if (ctx == nullptr) throw std::runtime_error("EVP_CIPHER_CTX_new failed");
        if (EVP_EncryptInit_ex(ctx, EVP_aes_128_ecb(), nullptr, nullptr,
                               nullptr) != 1) {
            throw std::runtime_error("EVP_EncryptInit_ex(cipher) failed");
        }
        EVP_CIPHER_CTX_set_padding(ctx, 0);
    }
    return ctx;
}

// Bit-identical host twin of the device `aes_prg_expand`.
inline void gpu_aes_prg_expand(U128 seed, U128 &s_l, uint8_t &t_l, U128 &s_r,
                               uint8_t &t_r) {
    uint8_t key[16];
    std::memcpy(key, &seed, 16);

    // Device plaintexts 0,1,2,3 map to four consecutive AES input blocks.
    uint8_t in[64];
    std::memset(in, 0, sizeof(in));
    in[16] = 0x01;
    in[32] = 0x02;
    in[48] = 0x03;

    EVP_CIPHER_CTX *ctx = ecb_ctx();
    if (EVP_EncryptInit_ex(ctx, nullptr, nullptr, key, nullptr) != 1) {
        throw std::runtime_error("EVP_EncryptInit_ex(key) failed");
    }
    uint8_t out[64];
    int len = 0;
    if (EVP_EncryptUpdate(ctx, out, &len, in, 64) != 1 || len != 64) {
        throw std::runtime_error("EVP_EncryptUpdate failed");
    }

    U128 left_seed = 0, left_tag = 0, right_seed = 0, right_tag = 0;
    std::memcpy(&left_seed, out, 16);
    std::memcpy(&left_tag, out + 16, 16);
    std::memcpy(&right_seed, out + 32, 16);
    std::memcpy(&right_tag, out + 48, 16);
    s_l = left_seed;
    s_r = right_seed;
    t_l = static_cast<uint8_t>(left_tag & 1);
    t_r = static_cast<uint8_t>(right_tag & 1);
}

}  // namespace ringlpn_gpu_prg
