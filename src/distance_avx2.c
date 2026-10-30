#include "../include/alg_hnsw.h"
#include <immintrin.h>
#include <math.h>

/*
 * Fallback escalar determinista:
 * Acumula en el mismo orden de bloques de 8 floats que AVX2 para
 * minimizar desviaciones de coma flotante entre plataformas.
 */
float l2_distance_scalar(const float *a, const float *b, uint32_t dim) {
    float total_sum = 0.0f;
    uint32_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        float block_sum = 0.0f;
        for (uint32_t j = 0; j < 8; j++) {
            float diff = a[i + j] - b[i + j];
            block_sum += diff * diff;
        }
        total_sum += block_sum;
    }

    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        total_sum += diff * diff;
    }

    return total_sum;
}

/*
 * Distancia L2 SIMD AVX2:
 * Para dim=64, procesa exactamente 8 registros ymm de 8 floats (64 elementos).
 * Utiliza FMA (_mm256_fmadd_ps) disponible en Intel Core i5-13500.
 */
float l2_distance_avx2(const float *a, const float *b, uint32_t dim) {
#if defined(__AVX2__)
    if (dim == 64U) {
        __m256 va0 = _mm256_loadu_ps(a);
        __m256 vb0 = _mm256_loadu_ps(b);
        __m256 d0  = _mm256_sub_ps(va0, vb0);
        __m256 acc = _mm256_mul_ps(d0, d0);

        __m256 va1 = _mm256_loadu_ps(a + 8);
        __m256 vb1 = _mm256_loadu_ps(b + 8);
        __m256 d1  = _mm256_sub_ps(va1, vb1);
        acc = _mm256_fmadd_ps(d1, d1, acc);

        __m256 va2 = _mm256_loadu_ps(a + 16);
        __m256 vb2 = _mm256_loadu_ps(b + 16);
        __m256 d2  = _mm256_sub_ps(va2, vb2);
        acc = _mm256_fmadd_ps(d2, d2, acc);

        __m256 va3 = _mm256_loadu_ps(a + 24);
        __m256 vb3 = _mm256_loadu_ps(b + 24);
        __m256 d3  = _mm256_sub_ps(va3, vb3);
        acc = _mm256_fmadd_ps(d3, d3, acc);

        __m256 va4 = _mm256_loadu_ps(a + 32);
        __m256 vb4 = _mm256_loadu_ps(b + 32);
        __m256 d4  = _mm256_sub_ps(va4, vb4);
        acc = _mm256_fmadd_ps(d4, d4, acc);

        __m256 va5 = _mm256_loadu_ps(a + 40);
        __m256 vb5 = _mm256_loadu_ps(b + 40);
        __m256 d5  = _mm256_sub_ps(va5, vb5);
        acc = _mm256_fmadd_ps(d5, d5, acc);

        __m256 va6 = _mm256_loadu_ps(a + 48);
        __m256 vb6 = _mm256_loadu_ps(b + 48);
        __m256 d6  = _mm256_sub_ps(va6, vb6);
        acc = _mm256_fmadd_ps(d6, d6, acc);

        __m256 va7 = _mm256_loadu_ps(a + 56);
        __m256 vb7 = _mm256_loadu_ps(b + 56);
        __m256 d7  = _mm256_sub_ps(va7, vb7);
        acc = _mm256_fmadd_ps(d7, d7, acc);

        /* Reducción horizontal del registro __m256 a float escalar */
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        return _mm_cvtss_f32(sum128);
    }

    /* Soporte genérico AVX2 si dim != 64 */
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }

    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float res = _mm_cvtss_f32(sum128);

    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        res += diff * diff;
    }
    return res;
#else
    return l2_distance_scalar(a, b, dim);
#endif
}
