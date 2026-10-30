#include "../include/sq8.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

int g_sq8_kernel_mode = 1;

static inline uint8_t quantize_one(float x, float lo, float scale) {
    if (!(scale > 0.0f) || !isfinite(x)) return 0;
    float q = (x - lo) / scale;
    if (q <= 0.0f) return 0;
    if (q >= 255.0f) return 255;
    return (uint8_t)floorf(q + 0.5f);
}

Sq8Index *sq8_build(const float *vectors, uint32_t n, uint32_t dim) {
    if (!vectors || !n || !dim) return NULL;
    Sq8Index *s = (Sq8Index *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->num_vectors = n;
    s->dim = dim;
    s->min_values = (float *)malloc((size_t)dim * sizeof(float));
    s->scales = (float *)malloc((size_t)dim * sizeof(float));
    s->codes = (uint8_t *)aligned_alloc_64((size_t)n * dim);
    if (!s->min_values || !s->scales || !s->codes) { sq8_free(s); return NULL; }

    for (uint32_t d = 0; d < dim; ++d) {
        float lo = INFINITY, hi = -INFINITY;
        for (uint32_t i = 0; i < n; ++i) {
            float x = vectors[(size_t)i * dim + d];
            if (!isfinite(x)) { sq8_free(s); return NULL; }
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }
        s->min_values[d] = lo;
        s->scales[d] = (hi > lo) ? (hi - lo) / 255.0f : 0.0f;
    }
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t d = 0; d < dim; ++d) {
            s->codes[(size_t)i * dim + d] = quantize_one(
                vectors[(size_t)i * dim + d], s->min_values[d], s->scales[d]);
        }
    }
    s->bytes = (size_t)n * dim + (size_t)dim * sizeof(float) * 2U + sizeof(*s);
    return s;
}

void sq8_free(Sq8Index *s) {
    if (!s) return;
    aligned_free_64(s->codes); free(s->min_values); free(s->scales); free(s);
}

void sq8_encode_query(const Sq8Index *s, const float *query, uint8_t *out) {
    if (!s || !query || !out) return;
    for (uint32_t d = 0; d < s->dim; ++d)
        out[d] = quantize_one(query[d], s->min_values[d], s->scales[d]);
}

float l2_distance_sq8_avx2_base(const Sq8Index *s, const uint8_t *q, uint32_t id) {
    if (!s || !q || id >= s->num_vectors) return INFINITY;
    const uint8_t *v = s->codes + (size_t)id * s->dim;
    float sum = 0.0f;
#if defined(__AVX2__)
    uint32_t d = 0;
    __m256 accum = _mm256_setzero_ps();
    for (; d + 8U <= s->dim; d += 8U) {
        uint64_t q8, v8;
        memcpy(&q8, q + d, sizeof(q8));
        memcpy(&v8, v + d, sizeof(v8));
        __m128i qbytes = _mm_cvtsi64_si128((long long)q8);
        __m128i vbytes = _mm_cvtsi64_si128((long long)v8);
        __m256i qi = _mm256_cvtepu8_epi32(qbytes);
        __m256i vi = _mm256_cvtepu8_epi32(vbytes);
        __m256i di = _mm256_sub_epi32(qi, vi);
        __m256 df = _mm256_cvtepi32_ps(di);
        __m256 scale = _mm256_loadu_ps(&s->scales[d]);
        __m256 weighted = _mm256_mul_ps(df, scale);
        accum = _mm256_fmadd_ps(weighted, weighted, accum);
    }
    __m128 lo = _mm256_castps256_ps128(accum);
    __m128 hi = _mm256_extractf128_ps(accum, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum = _mm_cvtss_f32(sum4);
    for (; d < s->dim; ++d) {
        float delta = ((int)q[d] - (int)v[d]) * s->scales[d];
        sum += delta * delta;
    }
#else
    for (uint32_t d = 0; d < s->dim; ++d) {
        float delta = ((int)q[d] - (int)v[d]) * s->scales[d];
        sum += delta * delta;
    }
#endif
    return sum;
}

float l2_distance_sq8_avx2_opt(const Sq8Index *s, const uint8_t *q, uint32_t id) {
    if (!s || !q || id >= s->num_vectors) return INFINITY;
    const uint8_t *v = s->codes + (size_t)id * s->dim;
#if defined(__AVX2__)
    if (s->dim == 64U) {
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
#endif
    return l2_distance_sq8_avx2_base(s, q, id);
}

float l2_distance_sq8_avx2(const Sq8Index *s, const uint8_t *q, uint32_t id) {
    if (g_sq8_kernel_mode == 0) {
        return l2_distance_sq8_avx2_base(s, q, id);
    }
    return l2_distance_sq8_avx2_opt(s, q, id);
}
