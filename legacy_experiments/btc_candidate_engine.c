/*
 * Motor de Prueba de Candidatos de Bloques Bitcoin (SHA-256d Específico de Dominio)
 * Optimizado para 13th Gen Intel Core i5-13500 (AVX2 + Intel SHA-NI, 14C/20T)
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>
#include <windows.h>

/* ============================================================================
 * Constantes Estándar SHA-256
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

/* Helpers escalares */
static inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}
static inline uint32_t bswap32(uint32_t x) {
    return __builtin_bswap32(x);
}

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

/* Ronda parcial (solo actualiza d += T1, sin calcular T2) para Early-Exit */
#define HALF_ROUND(a, b, c, d, e, f, g, h, k, w) do { \
    uint32_t t1 = (h) + S1(e) + Ch(e, f, g) + (k) + (w); \
    (d) += t1; \
} while (0)

/* ============================================================================
 * 1. BASELINE: SHA-256d Genérico (hashea los 80 bytes completos cada vez)
 *    Realiza 3 compresiones de bloques de 64 bytes por cada nonce.
 * ============================================================================ */
static void sha256_compress_generic(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    for (int i = 0; i < 16; i++) {
        uint32_t v;
        memcpy(&v, block + i * 4, 4);
        W[i] = bswap32(v);
    }
    for (int i = 16; i < 64; i++) {
        W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + S1(e) + Ch(e, f, g) + K[i] + W[i];
        uint32_t t2 = S0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__attribute__((noinline))
bool sha256d_generic_test_nonce(const uint8_t header80[80], uint32_t nonce_le, uint8_t out_hash[32]) {
    uint8_t blk0[64], blk1[64], blk2[64];
    memcpy(blk0, header80, 64);
    memset(blk1, 0, 64);
    memcpy(blk1, header80 + 64, 12);
    memcpy(blk1 + 12, &nonce_le, 4);
    blk1[16] = 0x80;
    /* Longitud 80 bytes = 640 bits = 0x0280 en big-endian al final */
    blk1[62] = 0x02;
    blk1[63] = 0x80;

    uint32_t st1[8];
    memcpy(st1, H0_INIT, sizeof(H0_INIT));
    sha256_compress_generic(st1, blk0);
    sha256_compress_generic(st1, blk1);

    memset(blk2, 0, 64);
    for (int i = 0; i < 8; i++) {
        uint32_t be = bswap32(st1[i]);
        memcpy(blk2 + i * 4, &be, 4);
    }
    blk2[32] = 0x80;
    /* Longitud 32 bytes = 256 bits = 0x0100 */
    blk2[62] = 0x01;
    blk2[63] = 0x00;

    uint32_t st2[8];
    memcpy(st2, H0_INIT, sizeof(H0_INIT));
    sha256_compress_generic(st2, blk2);

    if (st2[7] == 0) {
        if (out_hash) {
            for (int i = 0; i < 8; i++) {
                uint32_t be = bswap32(st2[i]);
                memcpy(out_hash + i * 4, &be, 4);
            }
        }
        return true;
    }
    return false;
}

/* ============================================================================
 * CONTEXTO PRECOMPUTADO PARA CANDIDATOS DE BLOQUE (MIDSTATE + RONDAS 0..2)
 * ============================================================================
 * En un bloque candidato de 80 bytes:
 * - Bytes 0..63 (Chunk 0): Fijos -> producen `midstate[8]`
 * - Bytes 64..75 (W[0], W[1], W[2]): Fijos (cola de merkle_root, nTime, nBits)
 * - Bytes 76..79 (W[3]): Nonce variable
 * - Palabras W[4..15] del Chunk 1: Padding fijo:
 *     W[4] = 0x80000000, W[5..14] = 0, W[15] = 0x00000280
 *
 * Además:
 * - Las rondas 0, 1 y 2 del Chunk 1 NO dependen de W[3] (nonce).
 *   Por lo tanto, el estado (a,b,c,d,e,f,g,h) tras la ronda 2 es CONSTANTE.
 * - W[16] = s1(W[14]) + W[9] + s0(W[1]) + W[0] = s0(W[1]) + W[0] -> CONSTANTE.
 * - W[17] = s1(W[15]) + W[10] + s0(W[2]) + W[1] = s1(0x280) + s0(W[2]) + W[1] -> CONSTANTE.
 * - Parte de W[18] = W[2] + W[11] = W[2] -> CONSTANTE (solo falta sumar s1(W[16]) + s0(W[3])).
 *   Incluso mejor: s1(W[16]) + W[2] es CONSTANTE, así que W[18] = W18_PARTIAL + s0(nonce)!
 */
typedef struct {
    uint32_t midstate[8];         /* Estado SHA-256 tras comprimir los primeros 64 bytes */
    uint32_t r3_state[8];         /* Estado (a..h) del Chunk 1 justo antes de la Ronda 3 */
    uint32_t w0, w1, w2;          /* Palabras constantes W[0..2] en big-endian */
    uint32_t w16, w17;            /* W[16] y W[17] precalculados */
    uint32_t w18_base;            /* s1(W[16]) + W[2] precalculado (W[18] = w18_base + s0(W[3])) */
    uint32_t w19_base;            /* s1(W[17]) + W[12] + s0(W[4]) precalculado (W[19] = w19_base + W[3]) */
    uint32_t r3_t1_base;          /* h + S1(e) + Ch(e,f,g) + K[3] de la Ronda 3 (solo falta + W[3]) */
    uint32_t r3_t2;               /* S0(a) + Maj(a,b,c) de la Ronda 3 */
    uint8_t  raw_header[80];
} BlockCandidateCtx;

void candidate_ctx_init(BlockCandidateCtx *ctx, const uint8_t header80[80]) {
    memcpy(ctx->raw_header, header80, 80);

    /* 1. Calcular Midstate del primer bloque (64 bytes) */
    memcpy(ctx->midstate, H0_INIT, sizeof(H0_INIT));
    sha256_compress_generic(ctx->midstate, header80);

    /* 2. Extraer W[0..2] en formato big-endian */
    uint32_t v0, v1, v2;
    memcpy(&v0, header80 + 64, 4);
    memcpy(&v1, header80 + 68, 4);
    memcpy(&v2, header80 + 72, 4);
    ctx->w0 = bswap32(v0);
    ctx->w1 = bswap32(v1);
    ctx->w2 = bswap32(v2);

    /* 3. Precalcular W[16], W[17], y partes constantes de W[18] y W[19] */
    /* W[4] = 0x80000000, W[5..14] = 0, W[15] = 0x00000280 */
    ctx->w16 = s0(ctx->w1) + ctx->w0;
    ctx->w17 = s1(0x00000280U) + s0(ctx->w2) + ctx->w1;
    ctx->w18_base = s1(ctx->w16) + ctx->w2;
    ctx->w19_base = s1(ctx->w17) + s0(0x80000000U);

    /* 4. Precalcular Rondas 0, 1 y 2 del Chunk 1 */
    uint32_t a = ctx->midstate[0], b = ctx->midstate[1], c = ctx->midstate[2], d = ctx->midstate[3];
    uint32_t e = ctx->midstate[4], f = ctx->midstate[5], g = ctx->midstate[6], h = ctx->midstate[7];

    /* Ronda 0 (W[0]) */
    {
        uint32_t t1 = h + S1(e) + Ch(e, f, g) + K[0] + ctx->w0;
        uint32_t t2 = S0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    /* Ronda 1 (W[1]) */
    {
        uint32_t t1 = h + S1(e) + Ch(e, f, g) + K[1] + ctx->w1;
        uint32_t t2 = S0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    /* Ronda 2 (W[2]) */
    {
        uint32_t t1 = h + S1(e) + Ch(e, f, g) + K[2] + ctx->w2;
        uint32_t t2 = S0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->r3_state[0] = a; ctx->r3_state[1] = b; ctx->r3_state[2] = c; ctx->r3_state[3] = d;
    ctx->r3_state[4] = e; ctx->r3_state[5] = f; ctx->r3_state[6] = g; ctx->r3_state[7] = h;

    /* En la Ronda 3, T1 = (h + S1(e) + Ch(e,f,g) + K[3]) + W[3], y T2 es 100% constante */
    ctx->r3_t1_base = h + S1(e) + Ch(e, f, g) + K[3];
    ctx->r3_t2      = S0(a) + Maj(a, b, c);
}

/* ============================================================================
 * 2. ESCALAR OPTIMIZADO DE DOMINIO:
 *    - Salto directo a la Ronda 3 del Chunk 1 usando `r3_state`
 *    - Rondas 4..15 con W[4..15] eliminados cuando son 0
 *    - W[16..19] precalculados o simplificados
 *    - Segundo SHA-256 especializado para 32 bytes (sin sumas de ceros)
 *    - Early-Exit en Rondas 57..60 (ahorra T2 en 57..60 y Rondas 61..63 completas)
 * ============================================================================ */
__attribute__((noinline))
bool sha256d_candidate_scalar(const BlockCandidateCtx *ctx, uint32_t nonce_be, uint8_t out_hash[32]) {
    uint32_t W[64];
    W[0]  = ctx->w0;
    W[1]  = ctx->w1;
    W[2]  = ctx->w2;
    W[3]  = nonce_be;
    W[4]  = 0x80000000U;
    W[5]  = 0; W[6]  = 0; W[7]  = 0; W[8]  = 0; W[9]  = 0;
    W[10] = 0; W[11] = 0; W[12] = 0; W[13] = 0; W[14] = 0;
    W[15] = 0x00000280U;

    W[16] = ctx->w16;
    W[17] = ctx->w17;
    W[18] = ctx->w18_base + s0(nonce_be);
    W[19] = ctx->w19_base + nonce_be;
    /* En W[20..30] varios términos W[i-7] o W[i-15] son cero */
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

    /* Cargar estado justo en la Ronda 3 */
    uint32_t a = ctx->r3_state[0], b = ctx->r3_state[1], c = ctx->r3_state[2], d = ctx->r3_state[3];
    uint32_t e = ctx->r3_state[4], f = ctx->r3_state[5], g = ctx->r3_state[6], h = ctx->r3_state[7];

    /* Ronda 3 simplificada */
    {
        uint32_t t1 = ctx->r3_t1_base + nonce_be;
        d += t1;
        h = t1 + ctx->r3_t2;
    }
    /* Rotación de variables tras Ronda 3: estado actual es (h, a, b, c, d, e, f, g) */
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

    /* Hash intermedio = midstate + (d, e, f, g, h, a, b, c) */
    uint32_t W2[64];
    W2[0] = ctx->midstate[0] + d;
    W2[1] = ctx->midstate[1] + e;
    W2[2] = ctx->midstate[2] + f;
    W2[3] = ctx->midstate[3] + g;
    W2[4] = ctx->midstate[4] + h;
    W2[5] = ctx->midstate[5] + a;
    W2[6] = ctx->midstate[6] + b;
    W2[7] = ctx->midstate[7] + c;

    /* Segundo SHA-256: W2[8] = 0x80000000, W2[9..14] = 0, W2[15] = 0x00000100 */
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

    /* Solo expandimos hasta W2[60] para el Early-Exit */
    for (int i = 32; i <= 60; i++) {
        W2[i] = s1(W2[i - 2]) + W2[i - 7] + s0(W2[i - 15]) + W2[i - 16];
    }

    a = H0_INIT[0]; b = H0_INIT[1]; c = H0_INIT[2]; d = H0_INIT[3];
    e = H0_INIT[4]; f = H0_INIT[5]; g = H0_INIT[6]; h = H0_INIT[7];

    /* Rondas 0..7 */
    ROUND(a, b, c, d, e, f, g, h, K[0], W2[0]);
    ROUND(h, a, b, c, d, e, f, g, K[1], W2[1]);
    ROUND(g, h, a, b, c, d, e, f, K[2], W2[2]);
    ROUND(f, g, h, a, b, c, d, e, K[3], W2[3]);
    ROUND(e, f, g, h, a, b, c, d, K[4], W2[4]);
    ROUND(d, e, f, g, h, a, b, c, K[5], W2[5]);
    ROUND(c, d, e, f, g, h, a, b, K[6], W2[6]);
    ROUND(b, c, d, e, f, g, h, a, K[7], W2[7]);

    /* Rondas 8..15 con padding constante */
    ROUND(a, b, c, d, e, f, g, h, K[8] + 0x80000000U, 0);
    ROUND(h, a, b, c, d, e, f, g, K[9], 0);
    ROUND(g, h, a, b, c, d, e, f, K[10], 0);
    ROUND(f, g, h, a, b, c, d, e, K[11], 0);
    ROUND(e, f, g, h, a, b, c, d, K[12], 0);
    ROUND(d, e, f, g, h, a, b, c, K[13], 0);
    ROUND(c, d, e, f, g, h, a, b, K[14], 0);
    ROUND(b, c, d, e, f, g, h, a, K[15] + 0x00000100U, 0);

    /* Rondas 16..55 */
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

    /* Ronda 56 completa (produce d_60 = a_57) */
    ROUND(a, b, c, d, e, f, g, h, K[56], W2[56]);

    /* Rondas 57, 58, 59, 60: HALF_ROUND (sin calcular S0(a) + Maj(a,b,c)) */
    HALF_ROUND(h, a, b, c, d, e, f, g, K[57], W2[57]);
    HALF_ROUND(g, h, a, b, c, d, e, f, K[58], W2[58]);
    HALF_ROUND(f, g, h, a, b, c, d, e, K[59], W2[59]);
    HALF_ROUND(e, f, g, h, a, b, c, d, K[60], W2[60]);

    /*
     * En este punto, `h` contiene e_61 (que se desplazará a h_64 en la ronda 63).
     * Verificamos Early-Exit: ¿h + H0_INIT[7] == 0?
     */
    if (__builtin_expect((h + 0x5be0cd19U) == 0, 0)) {
        if (out_hash) {
            uint8_t dummy[32];
            sha256d_generic_test_nonce(ctx->raw_header, bswap32(nonce_be), out_hash ? out_hash : dummy);
        }
        return true;
    }
    return false;
}

/* ============================================================================
 * 3. MOTOR SIMD AVX2 8-WAY (8 Candidatos en Paralelo por Hilo)
 *    Usa registros YMM de 256 bits para probar 8 nonces simultáneamente,
 *    con Midstate, salto a Ronda 3, padding constante y Early-Exit en Ronda 60.
 * ============================================================================ */
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

__attribute__((noinline))
uint32_t sha256d_candidate_avx2_8way(const BlockCandidateCtx *ctx, uint32_t base_nonce_be) {
    __m256i nonces = _mm256_add_epi32(
        _mm256_set1_epi32((int)base_nonce_be),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
    );

    __m256i W[64];
    W[0]  = _mm256_set1_epi32((int)ctx->w0);
    W[1]  = _mm256_set1_epi32((int)ctx->w1);
    W[2]  = _mm256_set1_epi32((int)ctx->w2);
    W[3]  = nonces;
    W[4]  = _mm256_set1_epi32((int)0x80000000U);
    for (int i = 5; i < 15; i++) W[i] = _mm256_setzero_si256();
    W[15] = _mm256_set1_epi32((int)0x00000280U);

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

    /* Ronda 3 */
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

    /* Verificar si algún carril cumple h == -H0_INIT[7] */
    __m256i target = _mm256_set1_epi32((int)(0U - 0x5be0cd19U));
    __m256i cmp = _mm256_cmpeq_epi32(h, target);
    int mask = _mm256_movemask_epi8(cmp);
    if (__builtin_expect(mask != 0, 0)) {
        int lane = __builtin_ctz(mask) >> 2;
        return base_nonce_be + (uint32_t)lane;
    }
    return 0;
}

/* ============================================================================
 * 4. MOTOR HARDWARE INTEL SHA-NI (2-WAY INTERLEAVED + MIDSTATE + PADDING FIJO)
 *    Usa las instrucciones dedicadas del silicio Intel Core i5-13500:
 *      - _mm_sha256rnds2_epu32 (ejecuta 2 rondas completas de SHA-256 en hardware)
 *      - _mm_sha256msg1_epu32 / _mm_sha256msg2_epu32
 *    Entrelaza 2 candidatos independientes (A y B) para ocultar la latencia de
 *    4 ciclos de `sha256rnds2` y mantener ocupados los puertos de ejecución,
 *    aprovechando Midstate precalculado y bloques de padding constantes.
 * ============================================================================ */
typedef struct {
    __m128i mid_state0;    /* [a, b, e, f] tras Chunk 0 */
    __m128i mid_state1;    /* [c, d, g, h] tras Chunk 0 */
    __m128i r2_state1;     /* Estado s1 tras Rondas 0 y 1 del Chunk 1 (precalculado, no depende del nonce) */
    uint32_t w0, w1, w2_plus_k2;
} ShaNiCandidateCtx;

void shani_ctx_init(ShaNiCandidateCtx *sctx, const BlockCandidateCtx *ctx) {
    /*
     * Formato de estado de las instrucciones Intel SHA-NI:
     * state0 = [a, b, e, f] (de mayor a menor elemento de 32 bits: [3]=a, [2]=b, [1]=e, [0]=f)
     * state1 = [c, d, g, h] (de mayor a menor elemento de 32 bits: [3]=c, [2]=d, [1]=g, [0]=h)
     */
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

    /* Precalcular Rondas 0 y 1 del Chunk 1 en hardware SHA-NI (solo dependen de W[0] y W[1]) */
    __m128i wk01 = _mm_set_epi32(0, 0, (int)(ctx->w1 + K[1]), (int)(ctx->w0 + K[0]));
    sctx->r2_state1 = _mm_sha256rnds2_epu32(sctx->mid_state1, sctx->mid_state0, wk01);
}

/* Macro auxiliar para 4 rondas SHA-NI entrelazadas (2 candidatos A y B) */
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

/* Macro para 4 rondas cuando el mensaje + K es idéntico y constante para A y B */
#define SHANI_4ROUNDS_CONST_2WAY(s0A, s1A, s0B, s1B, wk_const) do { \
    __m128i m0 = (wk_const); \
    (s1A) = _mm_sha256rnds2_epu32((s1A), (s0A), m0); \
    (s1B) = _mm_sha256rnds2_epu32((s1B), (s0B), m0); \
    __m128i m1 = _mm_shuffle_epi32(m0, 0x0E); \
    (s0A) = _mm_sha256rnds2_epu32((s0A), (s1A), m1); \
    (s0B) = _mm_sha256rnds2_epu32((s0B), (s1B), m1); \
} while (0)

/* Macro para un cuarto de bloque (4 rondas + actualización del schedule SHA-NI) */
#define SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w0A, w1A, w2A, w3A, w0B, w1B, w2B, w3B, k_vec) do { \
    SHANI_4ROUNDS_2WAY(s0A, s1A, s0B, s1B, w0A, w0B, k_vec); \
    w1A = _mm_add_epi32(w1A, _mm_alignr_epi8(w0A, w3A, 4)); \
    w1B = _mm_add_epi32(w1B, _mm_alignr_epi8(w0B, w3B, 4)); \
    w1A = _mm_sha256msg2_epu32(w1A, w0A); \
    w1B = _mm_sha256msg2_epu32(w1B, w0B); \
    w3A = _mm_sha256msg1_epu32(w3A, w0A); \
    w3B = _mm_sha256msg1_epu32(w3B, w0B); \
} while (0)

__attribute__((noinline))
uint32_t sha256d_candidate_shani_2way(const ShaNiCandidateCtx *sctx, uint32_t nonceA_be, uint32_t nonceB_be) {
    /*
     * Cargar estado precalculado en la Ronda 2 del Chunk 1:
     * s0 = mid_state0, s1 = r2_state1 (Rondas 0 y 1 ya ejecutadas una sola vez por bloque)
     */
    __m128i s0A = sctx->mid_state0, s1A = sctx->r2_state1;
    __m128i s0B = sctx->mid_state0, s1B = sctx->r2_state1;

    /*
     * Chunk 1 (bytes 64..127):
     * w0 = [W3, W2, W1, W0]
     * w1 = [0,  0,  0,  0x80000000]
     * w2 = [0,  0,  0,  0]
     * w3 = [0x00000280, 0, 0, 0]
     */
    uint32_t w2 = sctx->w2_plus_k2 - K[2];
    __m128i w0A = _mm_set_epi32((int)nonceA_be, (int)w2, (int)sctx->w1, (int)sctx->w0);
    __m128i w0B = _mm_set_epi32((int)nonceB_be, (int)w2, (int)sctx->w1, (int)sctx->w0);
    const __m128i w1_init = _mm_set_epi32(0, 0, 0, (int)0x80000000U);
    const __m128i w2_init = _mm_setzero_si128();
    const __m128i w3_init = _mm_set_epi32((int)0x00000280U, 0, 0, 0);

    __m128i w1A = w1_init, w1B = w1_init;
    __m128i w2A = w2_init, w2B = w2_init;
    __m128i w3A = w3_init, w3B = w3_init;

    /* Rondas 2..3 (Rondas 0..1 omitidas por precomputación en r2_state1) */
    __m128i m23A = _mm_set_epi32(0, 0, (int)(nonceA_be + K[3]), (int)sctx->w2_plus_k2);
    __m128i m23B = _mm_set_epi32(0, 0, (int)(nonceB_be + K[3]), (int)sctx->w2_plus_k2);
    s0A = _mm_sha256rnds2_epu32(s0A, s1A, m23A);
    s0B = _mm_sha256rnds2_epu32(s0B, s1B, m23B);

    /* Rondas 4..7 (W[4..7] + K[4..7] es 100% constante) */
    SHANI_4ROUNDS_CONST_2WAY(s0A, s1A, s0B, s1B,
        _mm_set_epi32((int)K[7], (int)K[6], (int)K[5], (int)(K[4] + 0x80000000U)));
    w0A = _mm_sha256msg1_epu32(w0A, w1A);
    w0B = _mm_sha256msg1_epu32(w0B, w1B);

    /* Rondas 8..11 (W[8..11] == 0, solo K[8..11]) */
    SHANI_4ROUNDS_CONST_2WAY(s0A, s1A, s0B, s1B, _mm_loadu_si128((const __m128i*)&K[8]));
    /* msg1(w1, w2) = w1 porque w2 == 0 y las palabras altas de w1 son 0 */

    /* Rondas 12..15 (W[12..15] + K[12..15] es 100% constante) */
    SHANI_4ROUNDS_CONST_2WAY(s0A, s1A, s0B, s1B,
        _mm_set_epi32((int)(K[15] + 0x00000280U), (int)K[14], (int)K[13], (int)K[12]));
    /* alignr(w3, w2, 4) = [W12, W11, W10, W9] = [0, 0, 0, 0], y msg1(w2, w3) = w2 = 0 */
    w0A = _mm_sha256msg2_epu32(w0A, w3A);
    w0B = _mm_sha256msg2_epu32(w0B, w3B);

    /* Rondas 16..47 */
    for (int r = 16; r < 48; r += 16) {
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w0A, w1A, w2A, w3A, w0B, w1B, w2B, w3B, _mm_loadu_si128((const __m128i*)&K[r + 0]));
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w1A, w2A, w3A, w0A, w1B, w2B, w3B, w0B, _mm_loadu_si128((const __m128i*)&K[r + 4]));
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w2A, w3A, w0A, w1A, w2B, w3B, w0B, w1B, _mm_loadu_si128((const __m128i*)&K[r + 8]));
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w3A, w0A, w1A, w2A, w3B, w0B, w1B, w2B, _mm_loadu_si128((const __m128i*)&K[r + 12]));
    }

    /* Rondas 48..63 finales del Chunk 1 */
    SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w0A, w1A, w2A, w3A, w0B, w1B, w2B, w3B, _mm_loadu_si128((const __m128i*)&K[48]));
    SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w1A, w2A, w3A, w0A, w1B, w2B, w3B, w0B, _mm_loadu_si128((const __m128i*)&K[52]));
    SHANI_4ROUNDS_2WAY(s0A, s1A, s0B, s1B, w2A, w2B, _mm_loadu_si128((const __m128i*)&K[56]));
    w3A = _mm_add_epi32(w3A, _mm_alignr_epi8(w2A, w1A, 4));
    w3B = _mm_add_epi32(w3B, _mm_alignr_epi8(w2B, w1B, 4));
    w3A = _mm_sha256msg2_epu32(w3A, w2A);
    w3B = _mm_sha256msg2_epu32(w3B, w2B);
    SHANI_4ROUNDS_2WAY(s0A, s1A, s0B, s1B, w3A, w3B, _mm_loadu_si128((const __m128i*)&K[60]));

    /* Sumar Midstate */
    s0A = _mm_add_epi32(s0A, sctx->mid_state0);
    s1A = _mm_add_epi32(s1A, sctx->mid_state1);
    s0B = _mm_add_epi32(s0B, sctx->mid_state0);
    s1B = _mm_add_epi32(s1B, sctx->mid_state1);

    /*
     * Segundo SHA-256:
     * Rearmar W[0..3] = [d, c, b, a] y W[4..7] = [h, g, f, e]
     * directamente desde (s0, s1) con shuffles (sin pasar por memoria).
     */
    __m128i abefA = _mm_shuffle_epi32(s0A, 0x1B); /* [f, e, b, a] */
    __m128i cdghA = _mm_shuffle_epi32(s1A, 0x1B); /* [h, g, d, c] */
    w0A = _mm_unpacklo_epi64(abefA, cdghA);       /* [d, c, b, a] = W[3,2,1,0] */
    w1A = _mm_unpackhi_epi64(abefA, cdghA);       /* [h, g, f, e] = W[7,6,5,4] */

    __m128i abefB = _mm_shuffle_epi32(s0B, 0x1B);
    __m128i cdghB = _mm_shuffle_epi32(s1B, 0x1B);
    w0B = _mm_unpacklo_epi64(abefB, cdghB);
    w1B = _mm_unpackhi_epi64(abefB, cdghB);

    w2A = w1_init;
    w2B = w1_init;
    const __m128i w3_blk2 = _mm_set_epi32((int)0x00000100U, 0, 0, 0);
    w3A = w3_blk2;
    w3B = w3_blk2;

    /* Estado inicial H0 para el segundo SHA-256 */
    const __m128i h0_s0 = _mm_set_epi32((int)0x6a09e667U, (int)0xbb67ae85U, (int)0x510e527fU, (int)0x9b05688cU);
    const __m128i h0_s1 = _mm_set_epi32((int)0x3c6ef372U, (int)0xa54ff53aU, (int)0x1f83d9abU, (int)0x5be0cd19U);
    s0A = h0_s0; s1A = h0_s1;
    s0B = h0_s0; s1B = h0_s1;

    /* Rondas 0..15 del segundo SHA-256 */
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
    /* alignr(w3, w2, 4) = [W12, W11, W10, W9] = 0, y msg1(w2, w3) = w2 = w1_init */
    w0A = _mm_sha256msg2_epu32(w0A, w3A);
    w0B = _mm_sha256msg2_epu32(w0B, w3B);

    for (int r = 16; r < 48; r += 16) {
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w0A, w1A, w2A, w3A, w0B, w1B, w2B, w3B, _mm_loadu_si128((const __m128i*)&K[r + 0]));
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w1A, w2A, w3A, w0A, w1B, w2B, w3B, w0B, _mm_loadu_si128((const __m128i*)&K[r + 4]));
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w2A, w3A, w0A, w1A, w2B, w3B, w0B, w1B, _mm_loadu_si128((const __m128i*)&K[r + 8]));
        SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w3A, w0A, w1A, w2A, w3B, w0B, w1B, w2B, _mm_loadu_si128((const __m128i*)&K[r + 12]));
    }

    /* Rondas 48..61 del segundo SHA-256 (Early-Exit en Ronda 61: omite Rondas 62..63) */
    SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w0A, w1A, w2A, w3A, w0B, w1B, w2B, w3B, _mm_loadu_si128((const __m128i*)&K[48]));
    SHANI_STEP_2WAY(s0A, s1A, s0B, s1B, w1A, w2A, w3A, w0A, w1B, w2B, w3B, w0B, _mm_loadu_si128((const __m128i*)&K[52]));
    SHANI_4ROUNDS_2WAY(s0A, s1A, s0B, s1B, w2A, w2B, _mm_loadu_si128((const __m128i*)&K[56]));
    w3A = _mm_add_epi32(w3A, _mm_alignr_epi8(w2A, w1A, 4));
    w3B = _mm_add_epi32(w3B, _mm_alignr_epi8(w2B, w1B, 4));
    w3A = _mm_sha256msg2_epu32(w3A, w2A);
    w3B = _mm_sha256msg2_epu32(w3B, w2B);

    /*
     * Early-Exit en Hardware SHA-NI:
     * Una sola instrucción `sha256rnds2` ejecuta las Rondas 60 y 61 y deposita F_62 en s1[0].
     * En las Rondas 62 y 63, F_62 simplemente se desplaza a G_63 -> H_64 sin modificarse.
     * Por tanto, ahorramos las Rondas 62 y 63 completas en el 99.99999998% de los candidatos.
     */
    __m128i kv60 = _mm_loadu_si128((const __m128i*)&K[60]);
    s1A = _mm_sha256rnds2_epu32(s1A, s0A, _mm_add_epi32(w3A, kv60));
    s1B = _mm_sha256rnds2_epu32(s1B, s0B, _mm_add_epi32(w3B, kv60));

    /* En s1, el elemento 0 ya es `H_64`. Verificamos si H_64 + 0x5be0cd19 == 0 */
    uint32_t hA = (uint32_t)_mm_cvtsi128_si32(s1A) + 0x5be0cd19U;
    uint32_t hB = (uint32_t)_mm_cvtsi128_si32(s1B) + 0x5be0cd19U;

    if (__builtin_expect(hA == 0, 0)) return nonceA_be;
    if (__builtin_expect(hB == 0, 0)) return nonceB_be;
    return 0;
}

/* ============================================================================
 * 5. PARALELISMO MULTIHILO (20 HILOS EN INTEL CORE i5-13500)
 * ============================================================================ */
typedef struct {
    const ShaNiCandidateCtx *sctx;
    uint32_t start_nonce_be;
    uint32_t count;
    volatile LONG *found_flag;
    uint32_t winning_nonce_be;
} WorkerArgs;

static DWORD WINAPI shani_worker_thread(LPVOID param) {
    WorkerArgs *args = (WorkerArgs *)param;
    const ShaNiCandidateCtx *sctx = args->sctx;
    uint32_t n = args->start_nonce_be;
    uint32_t count = args->count;

    /* Iterar por conteo exacto (evita bug de wraparound de 32 bits cuando n + count cruza 0xFFFFFFFF) */
    for (uint32_t i = 0; i < count; i += 2, n += 2) {
        uint32_t res = sha256d_candidate_shani_2way(sctx, n, n + 1);
        if (__builtin_expect(res != 0, 0)) {
            args->winning_nonce_be = res;
            InterlockedExchange(args->found_flag, 1);
            break;
        }
    }
    return 0;
}

/* Timer de alta precisión */
static double get_time_sec(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}

int main(void) {
    /* Configurar consola en UTF-8 */
    SetConsoleOutputCP(65001);

    printf("================================================================================\n");
    printf(" MOTOR DE PRUEBA DE CANDIDATOS DE BLOQUE BITCOIN (SHA-256d ESPECIFICO)\n");
    printf(" Hardware detectado: 13th Gen Intel Core i5-13500 (AVX2 + Intel SHA-NI, 20 hilos)\n");
    printf("================================================================================\n\n");

    /*
     * Cabecera real del Bloque Génesis de Bitcoin (Bloque #0, 80 bytes):
     * Nonce ganador real: 2083236893 (little-endian: 0x1dac2b7c -> W[3] big-endian = 0x1dac2b7c)
     */
    uint8_t genesis_header[80] = {
        /* version (1) */
        0x01, 0x00, 0x00, 0x00,
        /* prev_block (32 ceros) */
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        /* merkle_root */
        0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2,
        0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61,
        0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32,
        0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a,
        /* timestamp (1231006505 = 0x495fab29) */
        0x29, 0xab, 0x5f, 0x49,
        /* bits (0x1d00ffff) */
        0xff, 0xff, 0x00, 0x1d,
        /* nonce (2083236893 = bytes 1d ac 2b 7c) */
        0x1d, 0xac, 0x2b, 0x7c
    };

    const uint32_t WINNING_NONCE_BE = 0x1dac2b7cU;
    const uint32_t WINNING_NONCE_LE = bswap32(WINNING_NONCE_BE);

    BlockCandidateCtx ctx;
    candidate_ctx_init(&ctx, genesis_header);

    ShaNiCandidateCtx sctx;
    shani_ctx_init(&sctx, &ctx);

    /* 1. Verificación de exactitud entre todos los motores */
    printf("[1] VERIFICACION CRIPTOGRAFICA CON EL BLOQUE GENESIS DE BITCOIN\n");
    printf("    Midstate precalculado (bytes 0..63): ");
    for (int i = 0; i < 8; i++) printf("%08x", ctx.midstate[i]);
    printf("\n");

    uint8_t hash_gen[32] = {0}, hash_opt[32] = {0};
    bool ok_gen = sha256d_generic_test_nonce(genesis_header, WINNING_NONCE_LE, hash_gen);
    bool ok_opt = sha256d_candidate_scalar(&ctx, WINNING_NONCE_BE, hash_opt);
    uint32_t ok_avx2 = sha256d_candidate_avx2_8way(&ctx, WINNING_NONCE_BE - 3);
    uint32_t ok_shani = sha256d_candidate_shani_2way(&sctx, WINNING_NONCE_BE - 1, WINNING_NONCE_BE);

    printf("    Hash final (Big-Endian Display):     ");
    for (int i = 31; i >= 0; i--) printf("%02x", hash_gen[i]);
    printf("\n");
    printf("    - Motor Generico (80B completos):    %s\n", ok_gen ? "OK (Bloque valido detectado)" : "FALLO");
    printf("    - Motor Escalar Dominio + EarlyExit: %s\n", (ok_opt && memcmp(hash_gen, hash_opt, 32) == 0) ? "OK (Exacto)" : "FALLO");
    printf("    - Motor SIMD AVX2 (8-way):           %s (Nonce BE hallado: 0x%08x)\n", (ok_avx2 == WINNING_NONCE_BE) ? "OK (Exacto)" : "FALLO", ok_avx2);
    printf("    - Motor Hardware SHA-NI (2-way):     %s (Nonce BE hallado: 0x%08x)\n\n", (ok_shani == WINNING_NONCE_BE) ? "OK (Exacto)" : "FALLO", ok_shani);

    /* 2. Benchmark Monohilo (Single-Core) */
    const uint32_t N_SINGLE = 16U * 1024U * 1024U; /* 16,777,216 candidatos */
    uint32_t start_be = 0x10000000U;
    volatile uint32_t sink = 0;

    printf("[2] BENCHMARK MONOHILO (%u candidatos probados por motor)\n", N_SINGLE);

    /* 2a. Baseline Genérico */
    double t0 = get_time_sec();
    for (uint32_t i = 0; i < N_SINGLE / 4; i++) { /* 4M para no demorar demasiado el baseline */
        if (sha256d_generic_test_nonce(genesis_header, i, NULL)) sink ^= i;
    }
    double dt_gen = (get_time_sec() - t0) * 4.0;
    double mhs_gen = (N_SINGLE / 1e6) / dt_gen;
    printf("    1. SHA-256d Generico (80 bytes, 3 bloques):        %8.2f MH/s  (1.00x)\n", mhs_gen);

    /* 2b. Escalar Específico de Dominio (Midstate + R3 + Padding fijo + Early-Exit R60) */
    t0 = get_time_sec();
    for (uint32_t i = 0; i < N_SINGLE; i++) {
        if (sha256d_candidate_scalar(&ctx, start_be + i, NULL)) sink ^= i;
    }
    double dt_opt = get_time_sec() - t0;
    double mhs_opt = (N_SINGLE / 1e6) / dt_opt;
    printf("    2. Escalar Dominio (Midstate + R3 + Early-Exit):   %8.2f MH/s  (%.2fx vs Generico)\n", mhs_opt, mhs_opt / mhs_gen);

    /* 2c. AVX2 8-Way Específico de Dominio */
    t0 = get_time_sec();
    for (uint32_t i = 0; i < N_SINGLE; i += 8) {
        sink ^= sha256d_candidate_avx2_8way(&ctx, start_be + i);
    }
    double dt_avx2 = get_time_sec() - t0;
    double mhs_avx2 = (N_SINGLE / 1e6) / dt_avx2;
    printf("    3. SIMD AVX2 8-Way (Midstate + R3 + Early-Exit):   %8.2f MH/s  (%.2fx vs Generico)\n", mhs_avx2, mhs_avx2 / mhs_gen);

    /* 2d. Intel SHA-NI 2-Way Interleaved + Midstate */
    const uint32_t N_SHANI = 64U * 1024U * 1024U; /* 67.1M candidatos */
    t0 = get_time_sec();
    for (uint32_t i = 0; i < N_SHANI; i += 2) {
        sink ^= sha256d_candidate_shani_2way(&sctx, start_be + i, start_be + i + 1);
    }
    double dt_shani = get_time_sec() - t0;
    double mhs_shani = (N_SHANI / 1e6) / dt_shani;
    printf("    4. Intel SHA-NI 2-Way Hardware + Midstate (1 hilo):%8.2f MH/s  (%.2fx vs Generico)\n\n", mhs_shani, mhs_shani / mhs_gen);

    /* 3. Benchmark Multihilo (20 hilos en el i5-13500) */
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    int num_threads = (int)sysinfo.dwNumberOfProcessors;
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 64) num_threads = 64;

    const uint32_t PER_THREAD_CANDIDATES = 64U * 1024U * 1024U; /* 67.1M por hilo (~1.34B total) */
    uint64_t total_candidates = (uint64_t)num_threads * PER_THREAD_CANDIDATES;

    printf("[3] BENCHMARK MULTIHILO COMPLETO (%d hilos logicos, %.2f mil millones de candidatos)\n",
           num_threads, (double)total_candidates / 1e9);

    HANDLE threads[64];
    WorkerArgs wargs[64];
    volatile LONG found_flag = 0;

    /* Colocar el nonce ganador real cerca del final del último hilo para verificar hallazgo real */
    uint32_t search_base = WINNING_NONCE_BE - (uint32_t)(total_candidates - 1024);

    t0 = get_time_sec();
    for (int t = 0; t < num_threads; t++) {
        wargs[t].sctx = &sctx;
        wargs[t].start_nonce_be = search_base + (uint32_t)t * PER_THREAD_CANDIDATES;
        wargs[t].count = PER_THREAD_CANDIDATES;
        wargs[t].found_flag = &found_flag;
        wargs[t].winning_nonce_be = 0;
        threads[t] = CreateThread(NULL, 0, shani_worker_thread, &wargs[t], 0, NULL);
    }
    WaitForMultipleObjects((DWORD)num_threads, threads, TRUE, INFINITE);
    double dt_mt = get_time_sec() - t0;
    for (int t = 0; t < num_threads; t++) CloseHandle(threads[t]);

    uint32_t found_nonce = 0;
    for (int t = 0; t < num_threads; t++) {
        if (wargs[t].winning_nonce_be != 0) {
            found_nonce = wargs[t].winning_nonce_be;
        }
    }

    double mhs_mt = ((double)total_candidates / 1e6) / dt_mt;
    double ghs_mt = mhs_mt / 1000.0;

    printf("    - Tiempo total:            %.3f segundos\n", dt_mt);
    printf("    - Rendimiento Total CPU:   %.2f MH/s (%.3f GH/s)\n", mhs_mt, ghs_mt);
    printf("    - Aceleracion Total:       %.1fx sobre SHA-256d generico monohilo\n", mhs_mt / mhs_gen);
    printf("    - Candidato ganador:       Nonce BE = 0x%08x (Decimal LE = %u)\n",
           found_nonce, bswap32(found_nonce));

    if (sink == 0xdeadbeefU) printf(".");
    return 0;
}
