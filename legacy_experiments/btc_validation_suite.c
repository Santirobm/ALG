/*
 * Suite de Validación Rigurosa y Benchmark Sostenido (60s)
 * Motor de Prueba de Candidatos de Bloque Bitcoin (SHA-256d)
 *
 * Pruebas incluidas:
 *   1. Auditoría de Rangos Disjuntos y Contador Exacto de Candidatos Únicos por Hilo
 *   2. Validación Independiente (FIPS 180-4 Reference vs Escalar, AVX2 8-Way y SHA-NI 2-Way)
 *      sobre miles de cabeceras aleatorias, bloques históricos reales y barridos de nonces.
 *   3. Benchmark Sostenido de 60 segundos con 20 hilos (sin incluir inicialización),
 *      con muestreo cada 5 segundos y verificación de hits parciales para impedir
 *      eliminación de código muerto por el compilador.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>
#include <windows.h>

/* ============================================================================
 * IMPLEMENTACIÓN DE REFERENCIA INDEPENDIENTE (FIPS 180-4 SHA-256 ESTÁNDAR)
 * Escrita de forma completamente separada para actuar como oráculo de verdad.
 * ============================================================================ */
static const uint32_t K[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

static const uint32_t H0_INIT[8] = {
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
};

static inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}
static inline uint32_t bswap32(uint32_t x) {
    return __builtin_bswap32(x);
}

/* Oráculo FIPS 180-4 general para cualquier longitud en bytes */
static void fips180_sha256_raw(const uint8_t *data, size_t len, uint8_t digest[32]) {
    uint32_t h[8];
    memcpy(h, H0_INIT, sizeof(H0_INIT));

    uint8_t padded[128];
    memset(padded, 0, sizeof(padded));
    memcpy(padded, data, len);
    padded[len] = 0x80;
    uint64_t bit_len = (uint64_t)len * 8ULL;
    size_t total_blocks = (len + 9 <= 64) ? 1 : 2;
    size_t end_offset = total_blocks * 64 - 8;
    for (int i = 0; i < 8; i++) {
        padded[end_offset + i] = (uint8_t)(bit_len >> (56 - i * 8));
    }

    for (size_t blk = 0; blk < total_blocks; blk++) {
        const uint8_t *p = padded + blk * 64;
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)p[i * 4 + 0] << 24) |
                   ((uint32_t)p[i * 4 + 1] << 16) |
                   ((uint32_t)p[i * 4 + 2] << 8)  |
                   ((uint32_t)p[i * 4 + 3]);
        }
        for (int i = 16; i < 64; i++) {
            uint32_t s0_v = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1_v = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0_v + w[i - 7] + s1_v;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hv = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1_v = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            uint32_t ch_v = (e & f) ^ ((~e) & g);
            uint32_t temp1 = hv + S1_v + ch_v + K[i] + w[i];
            uint32_t S0_v = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            uint32_t maj_v = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0_v + maj_v;
            hv = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hv;
    }

    for (int i = 0; i < 8; i++) {
        digest[i * 4 + 0] = (uint8_t)(h[i] >> 24);
        digest[i * 4 + 1] = (uint8_t)(h[i] >> 16);
        digest[i * 4 + 2] = (uint8_t)(h[i] >> 8);
        digest[i * 4 + 3] = (uint8_t)(h[i]);
    }
}

/* Oráculo independiente SHA-256d para una cabecera de 80 bytes con nonce_be en W[3] */
static uint32_t reference_sha256d_h7(const uint8_t header80[80], uint32_t nonce_be, uint8_t full_hash_out[32]) {
    uint8_t hdr[80];
    memcpy(hdr, header80, 80);
    /* Escribir nonce_be en big-endian en bytes 76..79 (que es como SHA-256 lee W[3]) */
    hdr[76] = (uint8_t)(nonce_be >> 24);
    hdr[77] = (uint8_t)(nonce_be >> 16);
    hdr[78] = (uint8_t)(nonce_be >> 8);
    hdr[79] = (uint8_t)(nonce_be);

    uint8_t first_hash[32], second_hash[32];
    fips180_sha256_raw(hdr, 80, first_hash);
    fips180_sha256_raw(first_hash, 32, second_hash);
    if (full_hash_out) {
        memcpy(full_hash_out, second_hash, 32);
    }
    /* Retornar la palabra H[7] en formato nativo de 32 bits (bytes 28..31 en big-endian) */
    uint32_t h7 = ((uint32_t)second_hash[28] << 24) |
                  ((uint32_t)second_hash[29] << 16) |
                  ((uint32_t)second_hash[30] << 8)  |
                  ((uint32_t)second_hash[31]);
    return h7;
}

/* ============================================================================
 * MOTORES OPTIMIZADOS DE DOMINIO (ESCALAR, AVX2 8-WAY, INTEL SHA-NI 2-WAY)
 * ============================================================================ */
#define Ch(e, f, g)  (((e) & (f)) ^ (~(e) & (g)))
#define Maj(a, b, c) (((a) & (b)) ^ ((a) & (c)) ^ ((b) & (c)))
#define S0(a)        (rotr32((a), 2) ^ rotr32((a), 13) ^ rotr32((a), 22))
#define S1(e)        (rotr32((e), 6) ^ rotr32((e), 11) ^ rotr32((e), 25))
#define s0(x)        (rotr32((x), 7) ^ rotr32((x), 18) ^ ((x) >> 3))
#define s1(x)        (rotr32((x), 17) ^ rotr32((x), 19) ^ ((x) >> 10))

#define ROUND(a, b, c, d, e, f, g, h, k, w) do { \
    uint32_t t1 = (h) + S1(e) + Ch(e, f, g) + (k) + (w); \
    uint32_t t2 = S0(a) + Maj(a, b, c); \
    (d) += t1; \
    (h) = t1 + t2; \
} while (0)

#define HALF_ROUND(a, b, c, d, e, f, g, h, k, w) do { \
    uint32_t t1 = (h) + S1(e) + Ch(e, f, g) + (k) + (w); \
    (d) += t1; \
} while (0)

typedef struct {
    uint32_t midstate[8];
    uint32_t r3_state[8];
    uint32_t w0, w1, w2;
    uint32_t w16, w17;
    uint32_t w18_base;
    uint32_t w19_base;
    uint32_t r3_t1_base;
    uint32_t r3_t2;
    uint8_t  raw_header[80];
} BlockCandidateCtx;

static void candidate_ctx_init(BlockCandidateCtx *ctx, const uint8_t header80[80]) {
    memcpy(ctx->raw_header, header80, 80);

    /* Calcular Midstate de los primeros 64 bytes */
    uint32_t h[8];
    memcpy(h, H0_INIT, sizeof(H0_INIT));
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        uint32_t v;
        memcpy(&v, header80 + i * 4, 4);
        w[i] = bswap32(v);
    }
    for (int i = 16; i < 64; i++) {
        w[i] = s1(w[i - 2]) + w[i - 7] + s0(w[i - 15]) + w[i - 16];
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hv = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = hv + S1(e) + Ch(e, f, g) + K[i] + w[i];
        uint32_t t2 = S0(a) + Maj(a, b, c);
        hv = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->midstate[0] = h[0] + a; ctx->midstate[1] = h[1] + b;
    ctx->midstate[2] = h[2] + c; ctx->midstate[3] = h[3] + d;
    ctx->midstate[4] = h[4] + e; ctx->midstate[5] = h[5] + f;
    ctx->midstate[6] = h[6] + g; ctx->midstate[7] = h[7] + hv;

    uint32_t v0, v1, v2;
    memcpy(&v0, header80 + 64, 4);
    memcpy(&v1, header80 + 68, 4);
    memcpy(&v2, header80 + 72, 4);
    ctx->w0 = bswap32(v0);
    ctx->w1 = bswap32(v1);
    ctx->w2 = bswap32(v2);

    ctx->w16 = s0(ctx->w1) + ctx->w0;
    ctx->w17 = s1(0x00000280U) + s0(ctx->w2) + ctx->w1;
    ctx->w18_base = s1(ctx->w16) + ctx->w2;
    ctx->w19_base = s1(ctx->w17) + s0(0x80000000U);

    a = ctx->midstate[0]; b = ctx->midstate[1]; c = ctx->midstate[2]; d = ctx->midstate[3];
    e = ctx->midstate[4]; f = ctx->midstate[5]; g = ctx->midstate[6]; hv = ctx->midstate[7];

    uint32_t words[3] = { ctx->w0, ctx->w1, ctx->w2 };
    for (int r = 0; r < 3; r++) {
        uint32_t t1 = hv + S1(e) + Ch(e, f, g) + K[r] + words[r];
        uint32_t t2 = S0(a) + Maj(a, b, c);
        hv = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->r3_state[0] = a; ctx->r3_state[1] = b; ctx->r3_state[2] = c; ctx->r3_state[3] = d;
    ctx->r3_state[4] = e; ctx->r3_state[5] = f; ctx->r3_state[6] = g; ctx->r3_state[7] = hv;
    ctx->r3_t1_base = hv + S1(e) + Ch(e, f, g) + K[3];
    ctx->r3_t2      = S0(a) + Maj(a, b, c);
}

/* Motor Escalar: devuelve la palabra H[7] completa de 32 bits */
static uint32_t scalar_candidate_h7(const BlockCandidateCtx *ctx, uint32_t nonce_be) {
    uint32_t W[64];
    W[16] = ctx->w16;
    W[17] = ctx->w17;
    W[18] = ctx->w18_base + s0(nonce_be);
    W[19] = ctx->w19_base + nonce_be;
    W[20] = s1(W[18]) + 0x80000000U;
    W[21] = s1(W[19]);
    W[22] = s1(W[20]) + 0x00000280U;
    W[23] = s1(W[21]) + W[16];
    W[24] = s1(W[22]) + W[17];
    W[25] = s1(W[23]) + W[18];
    W[26] = s1(W[24]) + W[19];
    W[27] = s1(W[25]) + W[20];
    W[28] = s1(W[26]) + W[21];
    W[29] = s1(W[27]) + W[22];
    W[30] = s1(W[28]) + W[23] + s0(0x00000280U);
    W[31] = s1(W[29]) + W[24] + s0(W[16]) + 0x00000280U;
    for (int i = 32; i < 64; i++) {
        W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
    }

    uint32_t a = ctx->r3_state[0], b = ctx->r3_state[1], c = ctx->r3_state[2], d = ctx->r3_state[3];
    uint32_t e = ctx->r3_state[4], f = ctx->r3_state[5], g = ctx->r3_state[6], h = ctx->r3_state[7];

    {
        uint32_t t1 = ctx->r3_t1_base + nonce_be;
        d += t1;
        h = t1 + ctx->r3_t2;
    }
    ROUND(h, a, b, c, d, e, f, g, K[4] + 0x80000000U, 0);
    ROUND(g, h, a, b, c, d, e, f, K[5], 0);
    ROUND(f, g, h, a, b, c, d, e, K[6], 0);
    ROUND(e, f, g, h, a, b, c, d, K[7], 0);
    ROUND(d, e, f, g, h, a, b, c, K[8], 0);
    ROUND(c, d, e, f, g, h, a, b, K[9], 0);
    ROUND(b, c, d, e, f, g, h, a, K[10], 0);
    ROUND(a, b, c, d, e, f, g, h, K[11], 0);
    ROUND(h, a, b, c, d, e, f, g, K[12], 0);
    ROUND(g, h, a, b, c, d, e, f, K[13], 0);
    ROUND(f, g, h, a, b, c, d, e, K[14], 0);
    ROUND(e, f, g, h, a, b, c, d, K[15] + 0x00000280U, 0);

    for (int i = 16; i < 64; i += 8) {
        ROUND(d, e, f, g, h, a, b, c, K[i + 0], W[i + 0]);
        ROUND(c, d, e, f, g, h, a, b, K[i + 1], W[i + 1]);
        ROUND(b, c, d, e, f, g, h, a, K[i + 2], W[i + 2]);
        ROUND(a, b, c, d, e, f, g, h, K[i + 3], W[i + 3]);
        ROUND(h, a, b, c, d, e, f, g, K[i + 4], W[i + 4]);
        ROUND(g, h, a, b, c, d, e, f, K[i + 5], W[i + 5]);
        ROUND(f, g, h, a, b, c, d, e, K[i + 6], W[i + 6]);
        ROUND(e, f, g, h, a, b, c, d, K[i + 7], W[i + 7]);
    }

    uint32_t W2[64];
    W2[0] = ctx->midstate[0] + d;
    W2[1] = ctx->midstate[1] + e;
    W2[2] = ctx->midstate[2] + f;
    W2[3] = ctx->midstate[3] + g;
    W2[4] = ctx->midstate[4] + h;
    W2[5] = ctx->midstate[5] + a;
    W2[6] = ctx->midstate[6] + b;
    W2[7] = ctx->midstate[7] + c;

    W2[16] = s0(W2[1]) + W2[0];
    W2[17] = s1(0x00000100U) + s0(W2[2]) + W2[1];
    W2[18] = s1(W2[16]) + s0(W2[3]) + W2[2];
    W2[19] = s1(W2[17]) + s0(W2[4]) + W2[3];
    W2[20] = s1(W2[18]) + s0(W2[5]) + W2[4];
    W2[21] = s1(W2[19]) + s0(W2[6]) + W2[5];
    W2[22] = s1(W2[20]) + 0x00000100U + s0(W2[7]) + W2[6];
    W2[23] = s1(W2[21]) + W2[16] + s0(0x80000000U) + W2[7];
    W2[24] = s1(W2[22]) + W2[17] + 0x80000000U;
    W2[25] = s1(W2[23]) + W2[18];
    W2[26] = s1(W2[24]) + W2[19];
    W2[27] = s1(W2[25]) + W2[20];
    W2[28] = s1(W2[26]) + W2[21];
    W2[29] = s1(W2[27]) + W2[22];
    W2[30] = s1(W2[28]) + W2[23] + s0(0x00000100U);
    W2[31] = s1(W2[29]) + W2[24] + s0(W2[16]) + 0x00000100U;

    for (int i = 32; i <= 60; i++) {
        W2[i] = s1(W2[i - 2]) + W2[i - 7] + s0(W2[i - 15]) + W2[i - 16];
    }

    a = H0_INIT[0]; b = H0_INIT[1]; c = H0_INIT[2]; d = H0_INIT[3];
    e = H0_INIT[4]; f = H0_INIT[5]; g = H0_INIT[6]; h = H0_INIT[7];

    ROUND(a, b, c, d, e, f, g, h, K[0], W2[0]);
    ROUND(h, a, b, c, d, e, f, g, K[1], W2[1]);
    ROUND(g, h, a, b, c, d, e, f, K[2], W2[2]);
    ROUND(f, g, h, a, b, c, d, e, K[3], W2[3]);
    ROUND(e, f, g, h, a, b, c, d, K[4], W2[4]);
    ROUND(d, e, f, g, h, a, b, c, K[5], W2[5]);
    ROUND(c, d, e, f, g, h, a, b, K[6], W2[6]);
    ROUND(b, c, d, e, f, g, h, a, K[7], W2[7]);

    ROUND(a, b, c, d, e, f, g, h, K[8] + 0x80000000U, 0);
    ROUND(h, a, b, c, d, e, f, g, K[9], 0);
    ROUND(g, h, a, b, c, d, e, f, K[10], 0);
    ROUND(f, g, h, a, b, c, d, e, K[11], 0);
    ROUND(e, f, g, h, a, b, c, d, K[12], 0);
    ROUND(d, e, f, g, h, a, b, c, K[13], 0);
    ROUND(c, d, e, f, g, h, a, b, K[14], 0);
    ROUND(b, c, d, e, f, g, h, a, K[15] + 0x00000100U, 0);

    for (int i = 16; i < 56; i += 8) {
        ROUND(a, b, c, d, e, f, g, h, K[i + 0], W2[i + 0]);
        ROUND(h, a, b, c, d, e, f, g, K[i + 1], W2[i + 1]);
        ROUND(g, h, a, b, c, d, e, f, K[i + 2], W2[i + 2]);
        ROUND(f, g, h, a, b, c, d, e, K[i + 3], W2[i + 3]);
        ROUND(e, f, g, h, a, b, c, d, K[i + 4], W2[i + 4]);
        ROUND(d, e, f, g, h, a, b, c, K[i + 5], W2[i + 5]);
        ROUND(c, d, e, f, g, h, a, b, K[i + 6], W2[i + 6]);
        ROUND(b, c, d, e, f, g, h, a, K[i + 7], W2[i + 7]);
    }

    ROUND(a, b, c, d, e, f, g, h, K[56], W2[56]);
    HALF_ROUND(h, a, b, c, d, e, f, g, K[57], W2[57]);
    HALF_ROUND(g, h, a, b, c, d, e, f, K[58], W2[58]);
    HALF_ROUND(f, g, h, a, b, c, d, e, K[59], W2[59]);
    HALF_ROUND(e, f, g, h, a, b, c, d, K[60], W2[60]);

    return h + 0x5be0cd19U;
}

/* Motor AVX2 8-Way: devuelve los 8 valores H[7] de 32 bits para base_nonce_be + [0..7] */
#define AVX2_ROTR(x, n) _mm256_or_si256(_mm256_srli_epi32((x), (n)), _mm256_slli_epi32((x), 32 - (n)))
#define AVX2_S0(a)      _mm256_xor_si256(_mm256_xor_si256(AVX2_ROTR((a), 2), AVX2_ROTR((a), 13)), AVX2_ROTR((a), 22))
#define AVX2_S1(e)      _mm256_xor_si256(_mm256_xor_si256(AVX2_ROTR((e), 6), AVX2_ROTR((e), 11)), AVX2_ROTR((e), 25))
#define AVX2_s0(x)      _mm256_xor_si256(_mm256_xor_si256(AVX2_ROTR((x), 7), AVX2_ROTR((x), 18)), _mm256_srli_epi32((x), 3))
#define AVX2_s1(x)      _mm256_xor_si256(_mm256_xor_si256(AVX2_ROTR((x), 17), AVX2_ROTR((x), 19)), _mm256_srli_epi32((x), 10))
#define AVX2_Ch(e,f,g)  _mm256_xor_si256(_mm256_and_si256((e), (f)), _mm256_andnot_si256((e), (g)))
#define AVX2_Maj(a,b,c) _mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256((a), (b)), _mm256_and_si256((a), (c))), _mm256_and_si256((b), (c)))

#define AVX2_ROUND(a, b, c, d, e, f, g, h, k_val, w_vec) do { \
    __m256i t1 = _mm256_add_epi32((h), AVX2_S1(e)); \
    t1 = _mm256_add_epi32(t1, AVX2_Ch(e, f, g)); \
    t1 = _mm256_add_epi32(t1, _mm256_set1_epi32((int)(k_val))); \
    t1 = _mm256_add_epi32(t1, (w_vec)); \
    __m256i t2 = _mm256_add_epi32(AVX2_S0(a), AVX2_Maj(a, b, c)); \
    (d) = _mm256_add_epi32((d), t1); \
    (h) = _mm256_add_epi32(t1, t2); \
} while (0)

#define AVX2_ROUND_CONST(a, b, c, d, e, f, g, h, kw_const) do { \
    __m256i t1 = _mm256_add_epi32((h), AVX2_S1(e)); \
    t1 = _mm256_add_epi32(t1, AVX2_Ch(e, f, g)); \
    t1 = _mm256_add_epi32(t1, _mm256_set1_epi32((int)(kw_const))); \
    __m256i t2 = _mm256_add_epi32(AVX2_S0(a), AVX2_Maj(a, b, c)); \
    (d) = _mm256_add_epi32((d), t1); \
    (h) = _mm256_add_epi32(t1, t2); \
} while (0)

#define AVX2_HALF_ROUND(a, b, c, d, e, f, g, h, k_val, w_vec) do { \
    __m256i t1 = _mm256_add_epi32((h), AVX2_S1(e)); \
    t1 = _mm256_add_epi32(t1, AVX2_Ch(e, f, g)); \
    t1 = _mm256_add_epi32(t1, _mm256_set1_epi32((int)(k_val))); \
    t1 = _mm256_add_epi32(t1, (w_vec)); \
    (d) = _mm256_add_epi32((d), t1); \
} while (0)

static void avx2_candidate_8way_h7(const BlockCandidateCtx *ctx, uint32_t base_nonce_be, uint32_t out_h7[8]) {
    __m256i nonces = _mm256_add_epi32(
        _mm256_set1_epi32((int)base_nonce_be),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
    );

    __m256i W[64];
    W[16] = _mm256_set1_epi32((int)ctx->w16);
    W[17] = _mm256_set1_epi32((int)ctx->w17);
    W[18] = _mm256_add_epi32(_mm256_set1_epi32((int)ctx->w18_base), AVX2_s0(nonces));
    W[19] = _mm256_add_epi32(_mm256_set1_epi32((int)ctx->w19_base), nonces);
    W[20] = _mm256_add_epi32(AVX2_s1(W[18]), _mm256_set1_epi32((int)0x80000000U));
    W[21] = AVX2_s1(W[19]);
    W[22] = _mm256_add_epi32(AVX2_s1(W[20]), _mm256_set1_epi32((int)0x00000280U));
    W[23] = _mm256_add_epi32(AVX2_s1(W[21]), W[16]);
    W[24] = _mm256_add_epi32(AVX2_s1(W[22]), W[17]);
    W[25] = _mm256_add_epi32(AVX2_s1(W[23]), W[18]);
    W[26] = _mm256_add_epi32(AVX2_s1(W[24]), W[19]);
    W[27] = _mm256_add_epi32(AVX2_s1(W[25]), W[20]);
    W[28] = _mm256_add_epi32(AVX2_s1(W[26]), W[21]);
    W[29] = _mm256_add_epi32(AVX2_s1(W[27]), W[22]);
    W[30] = _mm256_add_epi32(_mm256_add_epi32(AVX2_s1(W[28]), W[23]), _mm256_set1_epi32((int)s0(0x00000280U)));
    W[31] = _mm256_add_epi32(_mm256_add_epi32(AVX2_s1(W[29]), W[24]), _mm256_set1_epi32((int)(s0(ctx->w16) + 0x00000280U)));

    for (int i = 32; i < 64; i++) {
        W[i] = _mm256_add_epi32(
            _mm256_add_epi32(AVX2_s1(W[i - 2]), W[i - 7]),
            _mm256_add_epi32(AVX2_s0(W[i - 15]), W[i - 16])
        );
    }

    __m256i a = _mm256_set1_epi32((int)ctx->r3_state[0]);
    __m256i b = _mm256_set1_epi32((int)ctx->r3_state[1]);
    __m256i c = _mm256_set1_epi32((int)ctx->r3_state[2]);
    __m256i d = _mm256_set1_epi32((int)ctx->r3_state[3]);
    __m256i e = _mm256_set1_epi32((int)ctx->r3_state[4]);
    __m256i f = _mm256_set1_epi32((int)ctx->r3_state[5]);
    __m256i g = _mm256_set1_epi32((int)ctx->r3_state[6]);
    __m256i h = _mm256_set1_epi32((int)ctx->r3_state[7]);

    {
        __m256i t1 = _mm256_add_epi32(_mm256_set1_epi32((int)ctx->r3_t1_base), nonces);
        d = _mm256_add_epi32(d, t1);
        h = _mm256_add_epi32(t1, _mm256_set1_epi32((int)ctx->r3_t2));
    }
    AVX2_ROUND_CONST(h, a, b, c, d, e, f, g, K[4] + 0x80000000U);
    AVX2_ROUND_CONST(g, h, a, b, c, d, e, f, K[5]);
    AVX2_ROUND_CONST(f, g, h, a, b, c, d, e, K[6]);
    AVX2_ROUND_CONST(e, f, g, h, a, b, c, d, K[7]);
    AVX2_ROUND_CONST(d, e, f, g, h, a, b, c, K[8]);
    AVX2_ROUND_CONST(c, d, e, f, g, h, a, b, K[9]);
    AVX2_ROUND_CONST(b, c, d, e, f, g, h, a, K[10]);
    AVX2_ROUND_CONST(a, b, c, d, e, f, g, h, K[11]);
    AVX2_ROUND_CONST(h, a, b, c, d, e, f, g, K[12]);
    AVX2_ROUND_CONST(g, h, a, b, c, d, e, f, K[13]);
    AVX2_ROUND_CONST(f, g, h, a, b, c, d, e, K[14]);
    AVX2_ROUND_CONST(e, f, g, h, a, b, c, d, K[15] + 0x00000280U);

    for (int i = 16; i < 64; i += 8) {
        AVX2_ROUND(d, e, f, g, h, a, b, c, K[i + 0], W[i + 0]);
        AVX2_ROUND(c, d, e, f, g, h, a, b, K[i + 1], W[i + 1]);
        AVX2_ROUND(b, c, d, e, f, g, h, a, K[i + 2], W[i + 2]);
        AVX2_ROUND(a, b, c, d, e, f, g, h, K[i + 3], W[i + 3]);
        AVX2_ROUND(h, a, b, c, d, e, f, g, K[i + 4], W[i + 4]);
        AVX2_ROUND(g, h, a, b, c, d, e, f, K[i + 5], W[i + 5]);
        AVX2_ROUND(f, g, h, a, b, c, d, e, K[i + 6], W[i + 6]);
        AVX2_ROUND(e, f, g, h, a, b, c, d, K[i + 7], W[i + 7]);
    }

    __m256i W2[64];
    W2[0] = _mm256_add_epi32(_mm256_set1_epi32((int)ctx->midstate[0]), d);
    W2[1] = _mm256_add_epi32(_mm256_set1_epi32((int)ctx->midstate[1]), e);
    W2[2] = _mm256_add_epi32(_mm256_set1_epi32((int)ctx->midstate[2]), f);
    W2[3] = _mm256_add_epi32(_mm256_set1_epi32((int)ctx->midstate[3]), g);
    W2[4] = _mm256_add_epi32(_mm256_set1_epi32((int)ctx->midstate[4]), h);
    W2[5] = _mm256_add_epi32(_mm256_set1_epi32((int)ctx->midstate[5]), a);
    W2[6] = _mm256_add_epi32(_mm256_set1_epi32((int)ctx->midstate[6]), b);
    W2[7] = _mm256_add_epi32(_mm256_set1_epi32((int)ctx->midstate[7]), c);

    W2[16] = _mm256_add_epi32(AVX2_s0(W2[1]), W2[0]);
    W2[17] = _mm256_add_epi32(_mm256_set1_epi32((int)s1(0x00000100U)), _mm256_add_epi32(AVX2_s0(W2[2]), W2[1]));
    W2[18] = _mm256_add_epi32(AVX2_s1(W2[16]), _mm256_add_epi32(AVX2_s0(W2[3]), W2[2]));
    W2[19] = _mm256_add_epi32(AVX2_s1(W2[17]), _mm256_add_epi32(AVX2_s0(W2[4]), W2[3]));
    W2[20] = _mm256_add_epi32(AVX2_s1(W2[18]), _mm256_add_epi32(AVX2_s0(W2[5]), W2[4]));
    W2[21] = _mm256_add_epi32(AVX2_s1(W2[19]), _mm256_add_epi32(AVX2_s0(W2[6]), W2[5]));
    W2[22] = _mm256_add_epi32(_mm256_add_epi32(AVX2_s1(W2[20]), _mm256_set1_epi32((int)0x00000100U)), _mm256_add_epi32(AVX2_s0(W2[7]), W2[6]));
    W2[23] = _mm256_add_epi32(_mm256_add_epi32(AVX2_s1(W2[21]), W2[16]), _mm256_add_epi32(_mm256_set1_epi32((int)s0(0x80000000U)), W2[7]));
    W2[24] = _mm256_add_epi32(AVX2_s1(W2[22]), _mm256_add_epi32(W2[17], _mm256_set1_epi32((int)0x80000000U)));
    W2[25] = _mm256_add_epi32(AVX2_s1(W2[23]), W2[18]);
    W2[26] = _mm256_add_epi32(AVX2_s1(W2[24]), W2[19]);
    W2[27] = _mm256_add_epi32(AVX2_s1(W2[25]), W2[20]);
    W2[28] = _mm256_add_epi32(AVX2_s1(W2[26]), W2[21]);
    W2[29] = _mm256_add_epi32(AVX2_s1(W2[27]), W2[22]);
    W2[30] = _mm256_add_epi32(_mm256_add_epi32(AVX2_s1(W2[28]), W2[23]), _mm256_set1_epi32((int)s0(0x00000100U)));
    W2[31] = _mm256_add_epi32(_mm256_add_epi32(AVX2_s1(W2[29]), W2[24]), _mm256_add_epi32(AVX2_s0(W2[16]), _mm256_set1_epi32((int)0x00000100U)));

    for (int i = 32; i <= 60; i++) {
        W2[i] = _mm256_add_epi32(
            _mm256_add_epi32(AVX2_s1(W2[i - 2]), W2[i - 7]),
            _mm256_add_epi32(AVX2_s0(W2[i - 15]), W2[i - 16])
        );
    }

    a = _mm256_set1_epi32((int)H0_INIT[0]);
    b = _mm256_set1_epi32((int)H0_INIT[1]);
    c = _mm256_set1_epi32((int)H0_INIT[2]);
    d = _mm256_set1_epi32((int)H0_INIT[3]);
    e = _mm256_set1_epi32((int)H0_INIT[4]);
    f = _mm256_set1_epi32((int)H0_INIT[5]);
    g = _mm256_set1_epi32((int)H0_INIT[6]);
    h = _mm256_set1_epi32((int)H0_INIT[7]);

    AVX2_ROUND(a, b, c, d, e, f, g, h, K[0], W2[0]);
    AVX2_ROUND(h, a, b, c, d, e, f, g, K[1], W2[1]);
    AVX2_ROUND(g, h, a, b, c, d, e, f, K[2], W2[2]);
    AVX2_ROUND(f, g, h, a, b, c, d, e, K[3], W2[3]);
    AVX2_ROUND(e, f, g, h, a, b, c, d, K[4], W2[4]);
    AVX2_ROUND(d, e, f, g, h, a, b, c, K[5], W2[5]);
    AVX2_ROUND(c, d, e, f, g, h, a, b, K[6], W2[6]);
    AVX2_ROUND(b, c, d, e, f, g, h, a, K[7], W2[7]);

    AVX2_ROUND_CONST(a, b, c, d, e, f, g, h, K[8] + 0x80000000U);
    AVX2_ROUND_CONST(h, a, b, c, d, e, f, g, K[9]);
    AVX2_ROUND_CONST(g, h, a, b, c, d, e, f, K[10]);
    AVX2_ROUND_CONST(f, g, h, a, b, c, d, e, K[11]);
    AVX2_ROUND_CONST(e, f, g, h, a, b, c, d, K[12]);
    AVX2_ROUND_CONST(d, e, f, g, h, a, b, c, K[13]);
    AVX2_ROUND_CONST(c, d, e, f, g, h, a, b, K[14]);
    AVX2_ROUND_CONST(b, c, d, e, f, g, h, a, K[15] + 0x00000100U);

    for (int i = 16; i < 56; i += 8) {
        AVX2_ROUND(a, b, c, d, e, f, g, h, K[i + 0], W2[i + 0]);
        AVX2_ROUND(h, a, b, c, d, e, f, g, K[i + 1], W2[i + 1]);
        AVX2_ROUND(g, h, a, b, c, d, e, f, K[i + 2], W2[i + 2]);
        AVX2_ROUND(f, g, h, a, b, c, d, e, K[i + 3], W2[i + 3]);
        AVX2_ROUND(e, f, g, h, a, b, c, d, K[i + 4], W2[i + 4]);
        AVX2_ROUND(d, e, f, g, h, a, b, c, K[i + 5], W2[i + 5]);
        AVX2_ROUND(c, d, e, f, g, h, a, b, K[i + 6], W2[i + 6]);
        AVX2_ROUND(b, c, d, e, f, g, h, a, K[i + 7], W2[i + 7]);
    }

    AVX2_ROUND(a, b, c, d, e, f, g, h, K[56], W2[56]);
    AVX2_HALF_ROUND(h, a, b, c, d, e, f, g, K[57], W2[57]);
    AVX2_HALF_ROUND(g, h, a, b, c, d, e, f, K[58], W2[58]);
    AVX2_HALF_ROUND(f, g, h, a, b, c, d, e, K[59], W2[59]);
    AVX2_HALF_ROUND(e, f, g, h, a, b, c, d, K[60], W2[60]);

    __m256i final_h7 = _mm256_add_epi32(h, _mm256_set1_epi32((int)0x5be0cd19U));
    _mm256_storeu_si256((__m256i *)out_h7, final_h7);
}

/* Motor Hardware Intel SHA-NI 2-Way */
typedef struct {
    __m128i mid_state0;
    __m128i mid_state1;
    __m128i r2_state1;
    uint32_t w0, w1, w2_plus_k2;
} ShaNiCandidateCtx;

static void shani_ctx_init(ShaNiCandidateCtx *sctx, const BlockCandidateCtx *ctx) {
    sctx->mid_state0 = _mm_set_epi32(
        (int)ctx->midstate[0], (int)ctx->midstate[1],
        (int)ctx->midstate[4], (int)ctx->midstate[5]
    );
    sctx->mid_state1 = _mm_set_epi32(
        (int)ctx->midstate[2], (int)ctx->midstate[3],
        (int)ctx->midstate[6], (int)ctx->midstate[7]
    );
    sctx->w0 = ctx->w0;
    sctx->w1 = ctx->w1;
    sctx->w2_plus_k2 = ctx->w2 + K[2];

    __m128i wk01 = _mm_set_epi32(0, 0, (int)(ctx->w1 + K[1]), (int)(ctx->w0 + K[0]));
    sctx->r2_state1 = _mm_sha256rnds2_epu32(sctx->mid_state1, sctx->mid_state0, wk01);
}

#define SHANI_4ROUNDS_2WAY(s0A, s1A, s0B, s1B, msgA, msgB, k_vec) do { \
    __m128i kv = (k_vec); \
    __m128i mA = _mm_add_epi32((msgA), kv); \
    __m128i mB = _mm_add_epi32((msgB), kv); \
    (s1A) = _mm_sha256rnds2_epu32((s1A), (s0A), mA); \
    (s1B) = _mm_sha256rnds2_epu32((s1B), (s0B), mB); \
    mA = _mm_shuffle_epi32(mA, 0x0E); \
    mB = _mm_shuffle_epi32(mB, 0x0E); \
    (s0A) = _mm_sha256rnds2_epu32((s0A), (s1A), mA); \
    (s0B) = _mm_sha256rnds2_epu32((s0B), (s1B), mB); \
} while (0)

#define SHANI_4ROUNDS_CONST_2WAY(s0A, s1A, s0B, s1B, wk_const) do { \
    __m128i m0 = (wk_const); \
    (s1A) = _mm_sha256rnds2_epu32((s1A), (s0A), m0); \
    (s1B) = _mm_sha256rnds2_epu32((s1B), (s0B), m0); \
    __m128i m1 = _mm_shuffle_epi32(m0, 0x0E); \
    (s0A) = _mm_sha256rnds2_epu32((s0A), (s1A), m1); \
    (s0B) = _mm_sha256rnds2_epu32((s0B), (s1B), m1); \
} while (0)

#define SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w0A, w1A, w2A, w3A, w0B, w1B, w2B, w3B, k_vec) do { \
    SHANI_4ROUNDS_2WAY(s0A, s1A, s0B, s1B, w0A, w0B, k_vec); \
    w1A = _mm_add_epi32(w1A, _mm_alignr_epi8(w0A, w3A, 4)); \
    w1B = _mm_add_epi32(w1B, _mm_alignr_epi8(w0B, w3B, 4)); \
    w1A = _mm_sha256msg2_epu32(w1A, w0A); \
    w1B = _mm_sha256msg2_epu32(w1B, w0B); \
    w3A = _mm_sha256msg1_epu32(w3A, w0A); \
    w3B = _mm_sha256msg1_epu32(w3B, w0B); \
} while (0)

static inline void shani_candidate_2way_h7(
    const ShaNiCandidateCtx *sctx,
    uint32_t nonceA_be,
    uint32_t nonceB_be,
    uint32_t *out_h7A,
    uint32_t *out_h7B
) {
    __m128i s0A = sctx->mid_state0, s1A = sctx->r2_state1;
    __m128i s0B = sctx->mid_state0, s1B = sctx->r2_state1;

    uint32_t w2 = sctx->w2_plus_k2 - K[2];
    __m128i w0A = _mm_set_epi32((int)nonceA_be, (int)w2, (int)sctx->w1, (int)sctx->w0);
    __m128i w0B = _mm_set_epi32((int)nonceB_be, (int)w2, (int)sctx->w1, (int)sctx->w0);
    const __m128i w1_init = _mm_set_epi32(0, 0, 0, (int)0x80000000U);
    const __m128i w2_init = _mm_setzero_si128();
    const __m128i w3_init = _mm_set_epi32((int)0x00000280U, 0, 0, 0);

    __m128i w1A = w1_init, w1B = w1_init;
    __m128i w2A = w2_init, w2B = w2_init;
    __m128i w3A = w3_init, w3B = w3_init;

    /* Rondas 2..3 */
    __m128i m23A = _mm_set_epi32(0, 0, (int)(nonceA_be + K[3]), (int)sctx->w2_plus_k2);
    __m128i m23B = _mm_set_epi32(0, 0, (int)(nonceB_be + K[3]), (int)sctx->w2_plus_k2);
    s0A = _mm_sha256rnds2_epu32(s0A, s1A, m23A);
    s0B = _mm_sha256rnds2_epu32(s0B, s1B, m23B);

    /* Rondas 4..7 */
    SHANI_4ROUNDS_CONST_2WAY(s0A, s1A, s0B, s1B,
        _mm_set_epi32((int)K[7], (int)K[6], (int)K[5], (int)(K[4] + 0x80000000U)));
    w0A = _mm_sha256msg1_epu32(w0A, w1A);
    w0B = _mm_sha256msg1_epu32(w0B, w1B);

    /* Rondas 8..11 */
    SHANI_4ROUNDS_CONST_2WAY(s0A, s1A, s0B, s1B, _mm_loadu_si128((const __m128i*)&K[8]));

    /* Rondas 12..15 */
    SHANI_4ROUNDS_CONST_2WAY(s0A, s1A, s0B, s1B,
        _mm_set_epi32((int)(K[15] + 0x00000280U), (int)K[14], (int)K[13], (int)K[12]));
    w0A = _mm_sha256msg2_epu32(w0A, w3A);
    w0B = _mm_sha256msg2_epu32(w0B, w3B);

    for (int r = 16; r < 48; r += 16) {
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w0A, w1A, w2A, w3A, w0B, w1B, w2B, w3B, _mm_loadu_si128((const __m128i*)&K[r + 0]));
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w1A, w2A, w3A, w0A, w1B, w2B, w3B, w0B, _mm_loadu_si128((const __m128i*)&K[r + 4]));
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w2A, w3A, w0A, w1A, w2B, w3B, w0B, w1B, _mm_loadu_si128((const __m128i*)&K[r + 8]));
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w3A, w0A, w1A, w2A, w3B, w0B, w1B, w2B, _mm_loadu_si128((const __m128i*)&K[r + 12]));
    }

    SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w0A, w1A, w2A, w3A, w0B, w1B, w2B, w3B, _mm_loadu_si128((const __m128i*)&K[48]));
    SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w1A, w2A, w3A, w0A, w1B, w2B, w3B, w0B, _mm_loadu_si128((const __m128i*)&K[52]));
    SHANI_4ROUNDS_2WAY(s0A, s1A, s0B, s1B, w2A, w2B, _mm_loadu_si128((const __m128i*)&K[56]));
    w3A = _mm_add_epi32(w3A, _mm_alignr_epi8(w2A, w1A, 4));
    w3B = _mm_add_epi32(w3B, _mm_alignr_epi8(w2B, w1B, 4));
    w3A = _mm_sha256msg2_epu32(w3A, w2A);
    w3B = _mm_sha256msg2_epu32(w3B, w2B);
    SHANI_4ROUNDS_2WAY(s0A, s1A, s0B, s1B, w3A, w3B, _mm_loadu_si128((const __m128i*)&K[60]));

    s0A = _mm_add_epi32(s0A, sctx->mid_state0);
    s1A = _mm_add_epi32(s1A, sctx->mid_state1);
    s0B = _mm_add_epi32(s0B, sctx->mid_state0);
    s1B = _mm_add_epi32(s1B, sctx->mid_state1);

    __m128i abefA = _mm_shuffle_epi32(s0A, 0x1B);
    __m128i cdghA = _mm_shuffle_epi32(s1A, 0x1B);
    w0A = _mm_unpacklo_epi64(abefA, cdghA);
    w1A = _mm_unpackhi_epi64(abefA, cdghA);

    __m128i abefB = _mm_shuffle_epi32(s0B, 0x1B);
    __m128i cdghB = _mm_shuffle_epi32(s1B, 0x1B);
    w0B = _mm_unpacklo_epi64(abefB, cdghB);
    w1B = _mm_unpackhi_epi64(abefB, cdghB);

    w2A = w1_init;
    w2B = w1_init;
    const __m128i w3_blk2 = _mm_set_epi32((int)0x00000100U, 0, 0, 0);
    w3A = w3_blk2;
    w3B = w3_blk2;

    const __m128i h0_s0 = _mm_set_epi32((int)0x6a09e667U, (int)0xbb67ae85U, (int)0x510e527fU, (int)0x9b05688cU);
    const __m128i h0_s1 = _mm_set_epi32((int)0x3c6ef372U, (int)0xa54ff53aU, (int)0x1f83d9abU, (int)0x5be0cd19U);
    s0A = h0_s0; s1A = h0_s1;
    s0B = h0_s0; s1B = h0_s1;

    SHANI_4ROUNDS_2WAY(s0A, s1A, s0B, s1B, w0A, w0B, _mm_loadu_si128((const __m128i*)&K[0]));
    SHANI_4ROUNDS_2WAY(s0A, s1A, s0B, s1B, w1A, w1B, _mm_loadu_si128((const __m128i*)&K[4]));
    w0A = _mm_sha256msg1_epu32(w0A, w1A);
    w0B = _mm_sha256msg1_epu32(w0B, w1B);

    SHANI_4ROUNDS_CONST_2WAY(s0A, s1A, s0B, s1B,
        _mm_set_epi32((int)K[11], (int)K[10], (int)K[9], (int)(K[8] + 0x80000000U)));
    w1A = _mm_sha256msg1_epu32(w1A, w2A);
    w1B = _mm_sha256msg1_epu32(w1B, w2B);

    SHANI_4ROUNDS_CONST_2WAY(s0A, s1A, s0B, s1B,
        _mm_set_epi32((int)(K[15] + 0x00000100U), (int)K[14], (int)K[13], (int)K[12]));
    w0A = _mm_sha256msg2_epu32(w0A, w3A);
    w0B = _mm_sha256msg2_epu32(w0B, w3B);

    for (int r = 16; r < 48; r += 16) {
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w0A, w1A, w2A, w3A, w0B, w1B, w2B, w3B, _mm_loadu_si128((const __m128i*)&K[r + 0]));
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w1A, w2A, w3A, w0A, w1B, w2B, w3B, w0B, _mm_loadu_si128((const __m128i*)&K[r + 4]));
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w2A, w3A, w0A, w1A, w2B, w3B, w0B, w1B, _mm_loadu_si128((const __m128i*)&K[r + 8]));
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w3A, w0A, w1A, w2A, w3B, w0B, w1B, w2B, _mm_loadu_si128((const __m128i*)&K[r + 12]));
    }

    SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w0A, w1A, w2A, w3A, w0B, w1B, w2B, w3B, _mm_loadu_si128((const __m128i*)&K[48]));
    SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w1A, w2A, w3A, w0A, w1B, w2B, w3B, w0B, _mm_loadu_si128((const __m128i*)&K[52]));
    SHANI_4ROUNDS_2WAY(s0A, s1A, s0B, s1B, w2A, w2B, _mm_loadu_si128((const __m128i*)&K[56]));
    w3A = _mm_add_epi32(w3A, _mm_alignr_epi8(w2A, w1A, 4));
    w3B = _mm_add_epi32(w3B, _mm_alignr_epi8(w2B, w1B, 4));
    w3A = _mm_sha256msg2_epu32(w3A, w2A);
    w3B = _mm_sha256msg2_epu32(w3B, w2B);

    __m128i kv60 = _mm_loadu_si128((const __m128i*)&K[60]);
    s1A = _mm_sha256rnds2_epu32(s1A, s0A, _mm_add_epi32(w3A, kv60));
    s1B = _mm_sha256rnds2_epu32(s1B, s0B, _mm_add_epi32(w3B, kv60));

    *out_h7A = (uint32_t)_mm_cvtsi128_si32(s1A) + 0x5be0cd19U;
    *out_h7B = (uint32_t)_mm_cvtsi128_si32(s1B) + 0x5be0cd19U;
}

/* ============================================================================
 * PRUEBA 1: AUDITORÍA DE CONTADOR DE CANDIDATOS Y RANGOS DISJUNTOS (1.342B)
 * ============================================================================ */
typedef struct {
    const ShaNiCandidateCtx *sctx;
    uint32_t thread_id;
    uint32_t start_nonce_be;
    uint32_t count;             /* Cantidad exacta de candidatos por hilo */
    uint64_t candidates_tested; /* Contador real incrementado en cada iteración */
    uint32_t first_nonce_seen;
    uint32_t last_nonce_seen;
    uint64_t partial_hits_20b;  /* Candidatos con (h7 & 0xFFFFF000) == 0 (prob 1/2^20) */
    uint32_t sample_hit_nonce;  /* Un nonce real que cumplió (h7 & 0xFFFFF000) == 0 */
    uint32_t sample_hit_h7;
    uint32_t exact_zero_nonce;  /* Nonce con h7 == 0 */
    uint64_t xor_checksum;      /* Checksum de todos los H7 para impedir optimización muerta */
} AuditWorkerArgs;

static DWORD WINAPI audit_worker_thread(LPVOID param) {
    AuditWorkerArgs *args = (AuditWorkerArgs *)param;
    const ShaNiCandidateCtx *sctx = args->sctx;
    uint32_t n = args->start_nonce_be;
    uint32_t target_count = args->count;

    uint64_t count = 0;
    uint64_t hits20 = 0;
    uint64_t checksum = 0;
    uint32_t sample_n = 0, sample_h7 = 0;
    uint32_t zero_n = 0;

    args->first_nonce_seen = n;

    for (uint32_t i = 0; i < target_count; i += 2, n += 2) {
        uint32_t h7A, h7B;
        shani_candidate_2way_h7(sctx, n, n + 1, &h7A, &h7B);
        count += 2;
        checksum ^= ((uint64_t)h7A << 32) | h7B;

        if (__builtin_expect((h7A & 0xFFFFF000U) == 0, 0)) {
            hits20++;
            sample_n = n;
            sample_h7 = h7A;
            if (h7A == 0) zero_n = n;
        }
        if (__builtin_expect((h7B & 0xFFFFF000U) == 0, 0)) {
            hits20++;
            sample_n = n + 1;
            sample_h7 = h7B;
            if (h7B == 0) zero_n = n + 1;
        }
    }

    args->last_nonce_seen = n - 1;
    args->candidates_tested = count;
    args->partial_hits_20b = hits20;
    args->sample_hit_nonce = sample_n;
    args->sample_hit_h7 = sample_h7;
    args->exact_zero_nonce = zero_n;
    args->xor_checksum = checksum;
    return 0;
}

/* ============================================================================
 * PRUEBA 3: BENCHMARK SOSTENIDO DE 60 SEGUNDOS (20 HILOS, SIN INICIALIZACIÓN)
 * ============================================================================ */
typedef struct {
    ShaNiCandidateCtx sctx;     /* Contexto propio por hilo (extranonce/merkle distinto por hilo) */
    uint8_t header80[80];
    uint32_t thread_id;
    HANDLE start_event;         /* Barrera de inicio simultáneo */
    volatile LONG *stop_flag;
    volatile LONGLONG candidates_done;
    uint64_t partial_hits_24b;  /* Candidatos con 24 bits en cero (h7 & 0xFFFFFF00) == 0 */
    uint32_t last_hit24_nonce;
    uint32_t last_hit24_h7;
    uint64_t xor_accum;
} SustainedWorkerArgs;

static DWORD WINAPI sustained_60s_worker(LPVOID param) {
    SustainedWorkerArgs *args = (SustainedWorkerArgs *)param;
    const ShaNiCandidateCtx *sctx = &args->sctx;

    /* Esperar señal de largada simultánea (fuera del cronómetro de inicialización) */
    WaitForSingleObject(args->start_event, INFINITE);

    /*
     * Cada hilo además empieza en una partición disjunta del espacio de nonces
     * (thread_id * 200,000,000) y tiene su propio merkle_root[0] = thread_id,
     * garantizando 0% de solapamiento incluso si un hilo superase 2^32 hashes.
     */
    uint32_t nonce = args->thread_id * 200000000U;
    uint64_t local_hits24 = 0;
    uint32_t hit_n = 0, hit_h7 = 0;
    uint64_t accum = 0;

    while (*args->stop_flag == 0) {
        /* Procesar en lotes de 65,536 candidatos (32,768 llamadas 2-way) antes de leer stop_flag */
        for (uint32_t i = 0; i < 65536U; i += 2, nonce += 2) {
            uint32_t h7A, h7B;
            shani_candidate_2way_h7(sctx, nonce, nonce + 1, &h7A, &h7B);
            accum ^= (uint64_t)(h7A + h7B);

            if (__builtin_expect((h7A & 0xFFFFFF00U) == 0, 0)) {
                local_hits24++;
                hit_n = nonce;
                hit_h7 = h7A;
            }
            if (__builtin_expect((h7B & 0xFFFFFF00U) == 0, 0)) {
                local_hits24++;
                hit_n = nonce + 1;
                hit_h7 = h7B;
            }
        }
        InterlockedAdd64(&args->candidates_done, 65536LL);
    }

    args->partial_hits_24b = local_hits24;
    args->last_hit24_nonce = hit_n;
    args->last_hit24_h7 = hit_h7;
    args->xor_accum = accum;
    return 0;
}

static double get_time_sec(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}

/* Generador PRNG determinista xorshift64* para pruebas reproducibles */
static uint64_t xorshift64s(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

int main(int argc, char **argv) {
    SetConsoleOutputCP(65001);
    setvbuf(stdout, NULL, _IONBF, 0);

    bool skip_60s = (argc > 1 && strcmp(argv[1], "--quick") == 0);

    printf("================================================================================\n");
    printf(" SUITE DE VALIDACION RIGUROSA Y BENCHMARK SOSTENIDO (60 SEGUNDOS)\n");
    printf(" Procesador: 13th Gen Intel Core i5-13500 (14 Nucleos / 20 Hilos, SHA-NI + AVX2)\n");
    printf("================================================================================\n\n");

    /* Cabecera del Bloque Génesis */
    uint8_t genesis_header[80] = {
        0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2,
        0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61,
        0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32,
        0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a,
        0x29, 0xab, 0x5f, 0x49,
        0xff, 0xff, 0x00, 0x1d,
        0x1d, 0xac, 0x2b, 0x7c
    };

    /* ========================================================================
     * PRUEBA 2 (EJECUTADA PRIMERO): VALIDACIÓN INDEPENDIENTE CONTRA ORÁCULO
     * ======================================================================== */
    printf("[PRUEBA 2] VALIDACION INDEPENDIENTE CONTRA ORACULO FIPS 180-4\n");
    printf("  Comparando palabra completa H[7] (32 bits) en miles de cabeceras y nonces...\n");

    /* 2A: Bloques históricos reales de Bitcoin (Bloque #0 Génesis y Bloque #1) */
    uint8_t block1_header[80] = {
        /* version 1 */
        0x01, 0x00, 0x00, 0x00,
        /* prev_block (hash del Bloque Génesis en little-endian) */
        0x6f, 0xe2, 0x8c, 0x0a, 0xb6, 0xf1, 0xb3, 0x72,
        0xc1, 0xa6, 0xa2, 0x46, 0xae, 0x63, 0xf7, 0x4f,
        0x93, 0x1e, 0x83, 0x65, 0xe1, 0x5a, 0x08, 0x9c,
        0x68, 0xd6, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00,
        /* merkle_root (little-endian de 0e3e2357e806b6cdb1f70b54c3a3a17b6714ee1f0e68bebb44a74b1efd512098) */
        0x98, 0x20, 0x51, 0xfd, 0x1e, 0x4b, 0xa7, 0x44,
        0xbb, 0xbe, 0x68, 0x0e, 0x1f, 0xee, 0x14, 0x67,
        0x7b, 0xa1, 0xa3, 0xc3, 0x54, 0x0b, 0xf7, 0xb1,
        0xcd, 0xb6, 0x06, 0xe8, 0x57, 0x23, 0x3e, 0x0e,
        /* timestamp 1231469665 = 0x4966bc61 */
        0x61, 0xbc, 0x66, 0x49,
        /* bits 0x1d00ffff */
        0xff, 0xff, 0x00, 0x1d,
        /* nonce 2573394689 = 0x9962e301 (en bytes LE: 01 e3 62 99 -> BE W[3] = 0x01e36299) */
        0x01, 0xe3, 0x62, 0x99
    };

    uint8_t h_b1[32];
    uint32_t ref_b1_h7 = reference_sha256d_h7(block1_header, 0x01e36299U, h_b1);
    BlockCandidateCtx ctx1;
    ShaNiCandidateCtx sctx1;
    candidate_ctx_init(&ctx1, block1_header);
    shani_ctx_init(&sctx1, &ctx1);
    uint32_t sc_b1_h7 = scalar_candidate_h7(&ctx1, 0x01e36299U);
    uint32_t sh_b1_A, sh_b1_B;
    shani_candidate_2way_h7(&sctx1, 0x01e36299U, 0x01e3629aU, &sh_b1_A, &sh_b1_B);

    printf("  - Bloque Historico #1 (Hash BE:       ");
    for (int i = 31; i >= 0; i--) printf("%02x", h_b1[i]);
    printf(")\n");
    printf("    Oraculo H7=0x%08x | Escalar H7=0x%08x | SHA-NI H7=0x%08x -> %s\n",
           ref_b1_h7, sc_b1_h7, sh_b1_A,
           (ref_b1_h7 == 0 && sc_b1_h7 == 0 && sh_b1_A == 0) ? "OK EXACTO" : "ERROR");

    /*
     * 2B: 2,500 cabeceras de bloque aleatorias x 64 nonces variados cada una
     *     = 160,000 candidatos únicos comparados bit a bit (los 32 bits de H[7])
     *     entre Oráculo FIPS 180-4, Escalar, AVX2 8-Way y SHA-NI 2-Way.
     */
    const int NUM_RANDOM_HEADERS = 2500;
    const int NONCES_PER_HEADER  = 64;
    uint64_t rng = 0xCAFEBABE12345678ULL;
    uint64_t total_verified = 0;
    uint64_t mismatches = 0;

    for (int h_idx = 0; h_idx < NUM_RANDOM_HEADERS; h_idx++) {
        uint8_t rand_hdr[80];
        for (int i = 0; i < 10; i++) {
            uint64_t r = xorshift64s(&rng);
            memcpy(rand_hdr + i * 8, &r, 8);
        }

        BlockCandidateCtx rctx;
        ShaNiCandidateCtx rsctx;
        candidate_ctx_init(&rctx, rand_hdr);
        shani_ctx_init(&rsctx, &rctx);

        uint32_t base_n = (uint32_t)xorshift64s(&rng);

        for (int offset = 0; offset < NONCES_PER_HEADER; offset += 8) {
            uint32_t n0 = base_n + (uint32_t)offset;
            uint32_t avx_h7[8];
            avx2_candidate_8way_h7(&rctx, n0, avx_h7);

            for (int k = 0; k < 8; k += 2) {
                uint32_t nA = n0 + (uint32_t)k;
                uint32_t nB = n0 + (uint32_t)k + 1;

                uint32_t refA = reference_sha256d_h7(rand_hdr, nA, NULL);
                uint32_t refB = reference_sha256d_h7(rand_hdr, nB, NULL);

                uint32_t scA = scalar_candidate_h7(&rctx, nA);
                uint32_t scB = scalar_candidate_h7(&rctx, nB);

                uint32_t shA, shB;
                shani_candidate_2way_h7(&rsctx, nA, nB, &shA, &shB);

                if (refA != scA || refA != avx_h7[k] || refA != shA ||
                    refB != scB || refB != avx_h7[k + 1] || refB != shB) {
                    mismatches++;
                }
                total_verified += 2;
            }
        }
    }

    printf("  - Cabeceras aleatorias probadas:   %d cabeceras distintas\n", NUM_RANDOM_HEADERS);
    printf("  - Candidatos comparados bit a bit: %llu (Oraculo vs Escalar vs AVX2 vs SHA-NI)\n",
           (unsigned long long)total_verified);
    printf("  - Discrepancias en H[7] (32 bits): %llu -> %s\n\n",
           (unsigned long long)mismatches, (mismatches == 0) ? "0 ERRORES (100% IDENTICOS)" : "FALLO");

    /* ========================================================================
     * PRUEBA 1: AUDITORÍA DE CONTADOR DE CANDIDATOS Y RANGOS DISJUNTOS (1.342B)
     * ======================================================================== */
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    int num_threads = (int)sysinfo.dwNumberOfProcessors;
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 64) num_threads = 64;

    const uint32_t PER_THREAD = 64U * 1024U * 1024U; /* Exactamente 67,108,864 por hilo */
    uint64_t expected_total = (uint64_t)num_threads * (uint64_t)PER_THREAD;

    printf("[PRUEBA 1] AUDITORIA DE CONTADOR Y RANGOS DISJUNTOS (%d hilos, %llu candidatos)\n",
           num_threads, (unsigned long long)expected_total);

    BlockCandidateCtx gen_ctx;
    ShaNiCandidateCtx gen_sctx;
    candidate_ctx_init(&gen_ctx, genesis_header);
    shani_ctx_init(&gen_sctx, &gen_ctx);

    const uint32_t WINNING_NONCE_BE = 0x1dac2b7cU;
    uint32_t audit_base = WINNING_NONCE_BE - (uint32_t)(expected_total - 1024);

    HANDLE a_threads[64];
    AuditWorkerArgs a_args[64];

    double t_audit0 = get_time_sec();
    for (int t = 0; t < num_threads; t++) {
        a_args[t].sctx = &gen_sctx;
        a_args[t].thread_id = (uint32_t)t;
        a_args[t].start_nonce_be = audit_base + (uint32_t)t * PER_THREAD;
        a_args[t].count          = PER_THREAD;
        a_args[t].candidates_tested = 0;
        a_args[t].partial_hits_20b = 0;
        a_args[t].sample_hit_nonce = 0;
        a_args[t].sample_hit_h7 = 0;
        a_args[t].exact_zero_nonce = 0;
        a_args[t].xor_checksum = 0;
        a_threads[t] = CreateThread(NULL, 0, audit_worker_thread, &a_args[t], 0, NULL);
    }
    WaitForMultipleObjects((DWORD)num_threads, a_threads, TRUE, INFINITE);
    double dt_audit = get_time_sec() - t_audit0;
    for (int t = 0; t < num_threads; t++) CloseHandle(a_threads[t]);

    uint64_t sum_candidates = 0;
    uint64_t sum_hits20 = 0;
    uint64_t combined_checksum = 0;
    bool ranges_disjoint = true;
    int oracle_spot_checks = 0;
    int oracle_spot_ok = 0;

    for (int t = 0; t < num_threads; t++) {
        sum_candidates += a_args[t].candidates_tested;
        sum_hits20 += a_args[t].partial_hits_20b;
        combined_checksum ^= a_args[t].xor_checksum;

        /* Verificar en aritmética modular de 32 bits que cada hilo procesó exactamente PER_THREAD */
        uint32_t span = (a_args[t].last_nonce_seen - a_args[t].first_nonce_seen) + 1U;
        if (span != PER_THREAD || a_args[t].candidates_tested != PER_THREAD) {
            ranges_disjoint = false;
        }
        if (t > 0 && a_args[t].first_nonce_seen != (uint32_t)(a_args[t - 1].last_nonce_seen + 1U)) {
            ranges_disjoint = false;
        }

        /* Validar con el oráculo independiente el último hit parcial de 20 bits de cada hilo */
        if (a_args[t].sample_hit_nonce != 0) {
            oracle_spot_checks++;
            uint32_t ref_h7 = reference_sha256d_h7(genesis_header, a_args[t].sample_hit_nonce, NULL);
            if (ref_h7 == a_args[t].sample_hit_h7 && (ref_h7 & 0xFFFFF000U) == 0) {
                oracle_spot_ok++;
            }
        }
    }

    double expected_hits20 = (double)expected_total / 1048576.0; /* N / 2^20 */
    printf("  - Rango Hilo 0:   [0x%08x .. 0x%08x] (%llu nonces)\n",
           a_args[0].first_nonce_seen, a_args[0].last_nonce_seen,
           (unsigned long long)a_args[0].candidates_tested);
    printf("  - Rango Hilo 12:  [0x%08x .. 0x%08x] (%llu nonces - cruza 0xFFFFFFFF sin fallo)\n",
           a_args[12].first_nonce_seen, a_args[12].last_nonce_seen,
           (unsigned long long)a_args[12].candidates_tested);
    printf("  - Rango Hilo %2d:  [0x%08x .. 0x%08x] (%llu nonces)\n",
           num_threads - 1,
           a_args[num_threads - 1].first_nonce_seen,
           a_args[num_threads - 1].last_nonce_seen,
           (unsigned long long)a_args[num_threads - 1].candidates_tested);
    printf("  - Verificacion de rangos contiguos y sin solapamiento: %s\n",
           ranges_disjoint ? "OK (0 duplicados, 20 particiones disjuntas)" : "ERROR");
    printf("  - Total nonces unicos contabilizados: %llu / %llu (%s)\n",
           (unsigned long long)sum_candidates, (unsigned long long)expected_total,
           (sum_candidates == expected_total) ? "100% EXACTO" : "ERROR");
    printf("  - Hits parciales (20 bits en cero):   %llu hallados (esperado estadistico ~%.1f)\n",
           (unsigned long long)sum_hits20, expected_hits20);
    printf("  - Auditoria de esos hits con Oraculo: %d / %d validados bit a bit\n",
           oracle_spot_ok, oracle_spot_checks);
    printf("  - Checksum XOR acumulado (anti-DCE):  0x%016llx\n",
           (unsigned long long)combined_checksum);
    printf("  - Tiempo y velocidad de esta corrida: %.3f s -> %.2f MH/s\n\n",
           dt_audit, ((double)sum_candidates / 1e6) / dt_audit);

    if (skip_60s) return 0;

    /* ========================================================================
     * PRUEBA 3: BENCHMARK SOSTENIDO DE 60 SEGUNDOS (20 HILOS, SIN INICIALIZACIÓN)
     * ======================================================================== */
    printf("[PRUEBA 3] BENCHMARK SOSTENIDO DURANTE 60 SEGUNDOS (%d hilos)\n", num_threads);
    printf("  Inicializando contextos e hilos fuera del cronometro...\n");

    HANDLE start_event = CreateEvent(NULL, TRUE, FALSE, NULL); /* Manual-reset event */
    volatile LONG stop_flag = 0;
    HANDLE s_threads[64];
    SustainedWorkerArgs s_args[64];

    for (int t = 0; t < num_threads; t++) {
        memcpy(s_args[t].header80, genesis_header, 80);
        /* Cada hilo recibe una cabecera con merkle_root[0] único para garantizar trabajo 100% único */
        s_args[t].header80[36] ^= (uint8_t)t;

        BlockCandidateCtx t_ctx;
        candidate_ctx_init(&t_ctx, s_args[t].header80);
        shani_ctx_init(&s_args[t].sctx, &t_ctx);

        s_args[t].thread_id = (uint32_t)t;
        s_args[t].start_event = start_event;
        s_args[t].stop_flag = &stop_flag;
        s_args[t].candidates_done = 0;
        s_args[t].partial_hits_24b = 0;
        s_args[t].last_hit24_nonce = 0;
        s_args[t].last_hit24_h7 = 0;
        s_args[t].xor_accum = 0;

        s_threads[t] = CreateThread(NULL, 0, sustained_60s_worker, &s_args[t], 0, NULL);
    }

    /* Pausa breve para asegurar que los 20 hilos estén creados y esperando en WaitForSingleObject */
    Sleep(200);

    printf("  ¡Arrancando medicion sostenida de 60.0 segundos ahora!\n");
    printf("  ------------------------------------------------------------------------\n");
    printf("   Segundos | Candidatos Acumulados | Velocidad Instantanea | Media Acum.\n");
    printf("  ------------------------------------------------------------------------\n");

    double t_start = get_time_sec();
    SetEvent(start_event); /* Liberar los 20 hilos simultáneamente */

    uint64_t prev_total = 0;
    double prev_t = t_start;

    for (int sec = 5; sec <= 60; sec += 5) {
        Sleep(5000);
        double now = get_time_sec();
        uint64_t cur_total = 0;
        for (int t = 0; t < num_threads; t++) {
            cur_total += (uint64_t)InterlockedCompareExchange64(&s_args[t].candidates_done, 0, 0);
        }
        double elapsed_total = now - t_start;
        double elapsed_window = now - prev_t;
        double inst_mhs = ((double)(cur_total - prev_total) / 1e6) / elapsed_window;
        double avg_mhs  = ((double)cur_total / 1e6) / elapsed_total;

        printf("    %5.1f s | %18llu |        %8.2f MH/s | %8.2f MH/s\n",
               elapsed_total, (unsigned long long)cur_total, inst_mhs, avg_mhs);

        prev_total = cur_total;
        prev_t = now;
    }

    /* Señalar parada a los 60 segundos */
    InterlockedExchange(&stop_flag, 1);
    WaitForMultipleObjects((DWORD)num_threads, s_threads, TRUE, INFINITE);
    double t_end = get_time_sec();

    for (int t = 0; t < num_threads; t++) CloseHandle(s_threads[t]);
    CloseHandle(start_event);

    double total_time = t_end - t_start;
    uint64_t grand_total_candidates = 0;
    uint64_t total_hits24 = 0;
    uint64_t final_xor = 0;
    int verified_24b_hits = 0;
    int checked_24b_hits = 0;

    for (int t = 0; t < num_threads; t++) {
        grand_total_candidates += (uint64_t)s_args[t].candidates_done;
        total_hits24 += s_args[t].partial_hits_24b;
        final_xor ^= s_args[t].xor_accum;

        if (s_args[t].last_hit24_nonce != 0) {
            checked_24b_hits++;
            uint32_t ref_h7 = reference_sha256d_h7(s_args[t].header80, s_args[t].last_hit24_nonce, NULL);
            if (ref_h7 == s_args[t].last_hit24_h7 && (ref_h7 & 0xFFFFFF00U) == 0) {
                verified_24b_hits++;
            }
        }
    }

    double sustained_mhs = ((double)grand_total_candidates / 1e6) / total_time;
    double expected_hits24 = (double)grand_total_candidates / 16777216.0; /* N / 2^24 */

    printf("  ------------------------------------------------------------------------\n");
    printf("  RESULTADO FINAL SOSTENIDO (60 SEGUNDOS, %d HILOS):\n", num_threads);
    printf("  - Tiempo neto de minado:         %.3f segundos (sin inicializacion)\n", total_time);
    printf("  - Candidatos unicos evaluados:   %llu (~%.2f mil millones)\n",
           (unsigned long long)grand_total_candidates, (double)grand_total_candidates / 1e9);
    printf("  - Rendimiento Sostenido Real:    %.2f MH/s (%.3f GH/s)\n",
           sustained_mhs, sustained_mhs / 1000.0);
    printf("  - Hits de 24 bits en cero:       %llu hallados (esperado estadistico ~%.1f)\n",
           (unsigned long long)total_hits24, expected_hits24);
    printf("  - Verificacion con Oraculo FIPS: %d / %d muestras de 24b validadas\n",
           verified_24b_hits, checked_24b_hits);
    printf("  - Checksum XOR global:           0x%016llx\n", (unsigned long long)final_xor);

    return 0;
}
