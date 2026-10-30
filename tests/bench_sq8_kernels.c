#include "../include/sq8.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__clang__)
#include <intrin.h>
#endif

/* Escalar de referencia */
static float dist_sq8_scalar(const Sq8Index *s, const uint8_t *q, uint32_t id) {
    const uint8_t *v = s->codes + (size_t)id * s->dim;
    float sum = 0.0f;
    for (uint32_t d = 0; d < s->dim; ++d) {
        float delta = ((int)q[d] - (int)v[d]) * s->scales[d];
        sum += delta * delta;
    }
    return sum;
}

/* Kernel actual (Baseline) */
static float dist_sq8_baseline(const Sq8Index *s, const uint8_t *q, uint32_t id) {
    return l2_distance_sq8_avx2(s, q, id);
}

/* U1: Unroll 16 (2 x 8 dims, 2 accumulators) con _mm_loadl_epi64 */
static float dist_sq8_u1_16(const Sq8Index *s, const uint8_t *q, uint32_t id) {
    const uint8_t *v = s->codes + (size_t)id * s->dim;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (uint32_t d = 0; d < 64; d += 16) {
        __m128i qb0 = _mm_loadl_epi64((const __m128i*)(q + d));
        __m128i vb0 = _mm_loadl_epi64((const __m128i*)(v + d));
        __m256i qi0 = _mm256_cvtepu8_epi32(qb0);
        __m256i vi0 = _mm256_cvtepu8_epi32(vb0);
        __m256  df0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(qi0, vi0));
        __m256  sc0 = _mm256_loadu_ps(&s->scales[d]);
        __m256  w0  = _mm256_mul_ps(df0, sc0);
        acc0 = _mm256_fmadd_ps(w0, w0, acc0);

        __m128i qb1 = _mm_loadl_epi64((const __m128i*)(q + d + 8));
        __m128i vb1 = _mm_loadl_epi64((const __m128i*)(v + d + 8));
        __m256i qi1 = _mm256_cvtepu8_epi32(qb1);
        __m256i vi1 = _mm256_cvtepu8_epi32(vb1);
        __m256  df1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(qi1, vi1));
        __m256  sc1 = _mm256_loadu_ps(&s->scales[d + 8]);
        __m256  w1  = _mm256_mul_ps(df1, sc1);
        acc1 = _mm256_fmadd_ps(w1, w1, acc1);
    }
    __m256 total = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(total);
    __m128 hi = _mm256_extractf128_ps(total, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

/* U3: Full unroll 64 dims con 4 acumuladores independientes y cargas directas */
static float dist_sq8_u3_full64(const Sq8Index *s, const uint8_t *q, uint32_t id) {
    const uint8_t *v = s->codes + (size_t)id * 64;

    /* Dims 0..15 */
    __m128i q0 = _mm_loadu_si128((const __m128i*)(q));
    __m128i v0 = _mm_load_si128((const __m128i*)(v));
    __m256i diff0 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(q0), _mm256_cvtepu8_epi16(v0));
    __m256  w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff0))), _mm256_loadu_ps(&s->scales[0]));
    __m256  w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff0, 1))), _mm256_loadu_ps(&s->scales[8]));
    __m256  acc0 = _mm256_mul_ps(w0, w0);
    __m256  acc1 = _mm256_mul_ps(w1, w1);

    /* Dims 16..31 */
    __m128i q1 = _mm_loadu_si128((const __m128i*)(q + 16));
    __m128i v1 = _mm_load_si128((const __m128i*)(v + 16));
    __m256i diff1 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(q1), _mm256_cvtepu8_epi16(v1));
    __m256  w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff1))), _mm256_loadu_ps(&s->scales[16]));
    __m256  w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff1, 1))), _mm256_loadu_ps(&s->scales[24]));
    __m256  acc2 = _mm256_mul_ps(w2, w2);
    __m256  acc3 = _mm256_mul_ps(w3, w3);

    /* Dims 32..47 */
    __m128i q2 = _mm_loadu_si128((const __m128i*)(q + 32));
    __m128i v2 = _mm_load_si128((const __m128i*)(v + 32));
    __m256i diff2 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(q2), _mm256_cvtepu8_epi16(v2));
    __m256  w4 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff2))), _mm256_loadu_ps(&s->scales[32]));
    __m256  w5 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff2, 1))), _mm256_loadu_ps(&s->scales[40]));
    acc0 = _mm256_fmadd_ps(w4, w4, acc0);
    acc1 = _mm256_fmadd_ps(w5, w5, acc1);

    /* Dims 48..63 */
    __m128i q3 = _mm_loadu_si128((const __m128i*)(q + 48));
    __m128i v3 = _mm_load_si128((const __m128i*)(v + 48));
    __m256i diff3 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(q3), _mm256_cvtepu8_epi16(v3));
    __m256  w6 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff3))), _mm256_loadu_ps(&s->scales[48]));
    __m256  w7 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff3, 1))), _mm256_loadu_ps(&s->scales[56]));
    acc2 = _mm256_fmadd_ps(w6, w6, acc2);
    acc3 = _mm256_fmadd_ps(w7, w7, acc3);

    __m256 total = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(total);
    __m128 hi = _mm256_extractf128_ps(total, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

typedef float (*DistFn)(const Sq8Index *s, const uint8_t *q, uint32_t id);

static void bench_kernel(const char *name, DistFn fn, const Sq8Index *s, const uint8_t *q, uint32_t n_vectors, uint32_t iters) {
    /* Warmup */
    volatile float dummy = 0.0f;
    for (uint32_t i = 0; i < 1000; ++i) {
        dummy += fn(s, q, i % n_vectors);
    }

    LARGE_INTEGER fq, t0, t1;
    QueryPerformanceFrequency(&fq);

    uint64_t c0 = __rdtsc();
    QueryPerformanceCounter(&t0);

    for (uint32_t iter = 0; iter < iters; ++iter) {
        uint32_t id = (iter * 17U) % n_vectors;
        dummy += fn(s, q, id);
    }

    QueryPerformanceCounter(&t1);
    uint64_t c1 = __rdtsc();

    double sec = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;
    double ns_per_call = (sec * 1e9) / (double)iters;
    double cycles_per_call = (double)(c1 - c0) / (double)iters;

    printf("%-20s: %6.2f ns/call | %6.1f cycles/call | val=%.4f\n",
           name, ns_per_call, cycles_per_call, dummy);
}

int main(void) {
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    const uint32_t n = 100000, dim = 64; /* 100.000 vectores: 6.4 MiB (presiona L3 / DRAM) */
    float *vectors = (float *)malloc((size_t)n * dim * sizeof(float));
    for (uint32_t i = 0; i < n * dim; ++i)
        vectors[i] = sinf((float)i * 0.017f) * 3.7f + (float)(i % 13) * 0.11f;

    Sq8Index *sq8 = sq8_build(vectors, n, dim);
    if (!sq8) { printf("Error sq8_build\n"); return 1; }

    uint8_t q[64];
    sq8_encode_query(sq8, vectors + 37U * dim, q);

    /* Verificación de exactitud numérica */
    for (uint32_t id = 0; id < 100; ++id) {
        float ref = dist_sq8_scalar(sq8, q, id);
        float b0  = dist_sq8_baseline(sq8, q, id);
        float u1  = dist_sq8_u1_16(sq8, q, id);
        float u3  = dist_sq8_u3_full64(sq8, q, id);

        if (fabsf(ref - b0) > 1e-3f || fabsf(ref - u1) > 1e-3f || fabsf(ref - u3) > 1e-3f) {
            printf("FAIL at id %u: ref=%.6f b0=%.6f u1=%.6f u3=%.6f\n",
                   id, ref, b0, u1, u3);
            return 2;
        }
    }
    printf("Verificacion numerica: PASS (todos los kernels con error < 1e-3 frente a ref)\n\n");

    const uint32_t ITERS = 2000000;
    printf("--- Test 1: N = 1.000 (L1/L2 Cache Resident, 64 KB) ---\n");
    bench_kernel("Scalar Reference", dist_sq8_scalar, sq8, q, 1000, ITERS / 4);
    bench_kernel("Baseline AVX2",   dist_sq8_baseline, sq8, q, 1000, ITERS);
    bench_kernel("U1: Unroll-16",    dist_sq8_u1_16, sq8, q, 1000, ITERS);
    bench_kernel("U3: Full-64-EPI16",dist_sq8_u3_full64, sq8, q, 1000, ITERS);

    printf("\n--- Test 2: N = 100.000 (L3 / DRAM Resident, 6.4 MB) ---\n");
    bench_kernel("Baseline AVX2",   dist_sq8_baseline, sq8, q, 100000, ITERS);
    bench_kernel("U1: Unroll-16",    dist_sq8_u1_16, sq8, q, 100000, ITERS);
    bench_kernel("U3: Full-64-EPI16",dist_sq8_u3_full64, sq8, q, 100000, ITERS);

    sq8_free(sq8);
    free(vectors);
    return 0;
}
