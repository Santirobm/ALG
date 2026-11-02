#include "../include/geometric_probe.h"
#include <windows.h>
#include <immintrin.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>

#define GPRB_MAGIC 0x47505242U /* 'GPRB' */
#define GPRB_VERSION 1U

static inline uint64_t gp_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float l2_dist_sq_avx2(const float *a, const float *b, uint32_t dim) {
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d  = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

extern "C" int geometric_probe_build(
    GeometricProbe *probe,
    const float *train_vectors,
    uint32_t num_vectors,
    uint32_t dim,
    uint32_t num_prototypes,
    uint32_t num_iterations,
    uint64_t seed
) {
    if (!probe || !train_vectors || num_vectors == 0 || dim != 64 || num_prototypes == 0) return -1;
    if (num_prototypes > GEOMETRIC_PROBE_MAX_PROTOTYPES) num_prototypes = GEOMETRIC_PROBE_MAX_PROTOTYPES;

    memset(probe, 0, sizeof(GeometricProbe));
    probe->num_prototypes = num_prototypes;
    probe->dim = dim;
    probe->train_seed = seed;
    probe->local_threshold_sq = 0.75f;
    probe->adversarial_threshold_sq = 0.95f;

    uint32_t sample_size = (num_vectors < 5000U) ? num_vectors : 5000U;
    probe->train_sample_size = sample_size;

    uint64_t rng = seed ^ 0x55AA55AA11223344ULL;
    std::vector<uint32_t> sample_indices(sample_size);
    for (uint32_t i = 0; i < sample_size; i++) {
        sample_indices[i] = (uint32_t)(gp_rng_next(&rng) % num_vectors);
    }

    /* 1. Inicializar prototipos muestreando vectores del dataset de entrenamiento */
    for (uint32_t k = 0; k < num_prototypes; k++) {
        uint32_t v_idx = sample_indices[k % sample_size];
        const float *src = train_vectors + (size_t)v_idx * dim;
        float *dst = probe->prototypes + (size_t)k * dim;
        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            dst[d] = src[d];
            norm_sq += dst[d] * dst[d];
        }
        float inv = 1.0f / sqrtf(norm_sq > 1e-8f ? norm_sq : 1.0f);
        for (uint32_t d = 0; d < dim; d++) dst[d] *= inv;
    }

    /* 2. Spherical k-means sobre la muestra */
    std::vector<float> accum(num_prototypes * dim, 0.0f);
    std::vector<uint32_t> counts(num_prototypes, 0);

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        std::fill(accum.begin(), accum.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (uint32_t s = 0; s < sample_size; s++) {
            const float *v = train_vectors + (size_t)sample_indices[s] * dim;
            float best_d = 1e9f;
            uint32_t best_k = 0;

            for (uint32_t k = 0; k < num_prototypes; k++) {
                float d = l2_dist_sq_avx2(v, probe->prototypes + (size_t)k * dim, dim);
                if (d < best_d) {
                    best_d = d;
                    best_k = k;
                }
            }

            counts[best_k]++;
            float *acc_ptr = accum.data() + (size_t)best_k * dim;
            for (uint32_t d = 0; d < dim; d++) {
                acc_ptr[d] += v[d];
            }
        }

        /* Actualizar y normalizar prototipos */
        for (uint32_t k = 0; k < num_prototypes; k++) {
            float *proto = probe->prototypes + (size_t)k * dim;
            if (counts[k] > 0) {
                const float *acc_ptr = accum.data() + (size_t)k * dim;
                float norm_sq = 0.0f;
                for (uint32_t d = 0; d < dim; d++) {
                    proto[d] = acc_ptr[d];
                    norm_sq += proto[d] * proto[d];
                }
                float inv = 1.0f / sqrtf(norm_sq > 1e-8f ? norm_sq : 1.0f);
                for (uint32_t d = 0; d < dim; d++) proto[d] *= inv;
            } else {
                /* Si un cluster queda vacío, re-inicializar con vector aleatorio */
                uint32_t r_idx = sample_indices[gp_rng_next(&rng) % sample_size];
                const float *src = train_vectors + (size_t)r_idx * dim;
                for (uint32_t d = 0; d < dim; d++) proto[d] = src[d];
            }
        }
    }

    return 0;
}

extern "C" QueryRegime geometric_probe_classify(
    const GeometricProbe *probe,
    const float *query,
    float *out_min_dist_sq
) {
    if (!probe || !query) return QUERY_UNKNOWN;
    const uint32_t num_prototypes = probe->num_prototypes;
    const uint32_t dim = probe->dim;

    float min_d = 1e9f;
    for (uint32_t k = 0; k < num_prototypes; k++) {
        float d = l2_dist_sq_avx2(query, probe->prototypes + (size_t)k * dim, dim);
        if (d < min_d) {
            min_d = d;
            if (min_d < 0.35f) break; /* Early exit para consultas hiperlocales */
        }
    }

    if (out_min_dist_sq) *out_min_dist_sq = min_d;

    if (min_d < probe->local_threshold_sq) {
        return QUERY_LOCAL;
    } else if (min_d > probe->adversarial_threshold_sq) {
        return QUERY_ADVERSARIAL;
    } else {
        return QUERY_UNKNOWN; /* Franja de incertidumbre */
    }
}

extern "C" void confusion_matrix_compute(
    const GeometricProbe *probe,
    const float *queries_R,
    uint32_t nq_R,
    const float *queries_D,
    uint32_t nq_D,
    uint32_t dim,
    ConfusionMatrix *cm
) {
    if (!cm) return;
    memset(cm, 0, sizeof(ConfusionMatrix));
    cm->total_r = nq_R;
    cm->total_d = nq_D;

    LARGE_INTEGER fq, t0, t1;
    QueryPerformanceFrequency(&fq);
    QueryPerformanceCounter(&t0);

    /* Evaluar consultas R */
    for (uint32_t q = 0; q < nq_R; q++) {
        const float *qv = queries_R + (size_t)q * dim;
        float dist = 0.0f;
        QueryRegime reg = geometric_probe_classify(probe, qv, &dist);
        if (reg == QUERY_LOCAL) cm->tp++;
        else if (reg == QUERY_ADVERSARIAL) cm->fn++;
        else cm->unknown_r++;
    }

    /* Evaluar consultas D */
    for (uint32_t q = 0; q < nq_D; q++) {
        const float *qv = queries_D + (size_t)q * dim;
        float dist = 0.0f;
        QueryRegime reg = geometric_probe_classify(probe, qv, &dist);
        if (reg == QUERY_LOCAL) cm->fp++;
        else if (reg == QUERY_ADVERSARIAL) cm->tn++;
        else cm->unknown_d++;
    }

    QueryPerformanceCounter(&t1);
    double elapsed_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;
    uint32_t total_queries = nq_R + nq_D;
    cm->avg_latency_ns = (total_queries > 0) ? (elapsed_sec * 1e9 / total_queries) : 0.0;

    cm->precision = (cm->tp + cm->fp > 0) ? ((double)cm->tp / (cm->tp + cm->fp)) : 0.0;
    cm->recall = (nq_R > 0) ? ((double)cm->tp / nq_R) : 0.0;
    cm->specificity = (nq_D > 0) ? ((double)cm->tn / nq_D) : 0.0;
    cm->fpr = (nq_D > 0) ? ((double)cm->fp / nq_D) : 0.0;
    cm->fnr = (nq_R > 0) ? ((double)cm->fn / nq_R) : 0.0;
    cm->balanced_accuracy = 0.5 * (cm->recall + cm->specificity);
}

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t num_prototypes;
    uint32_t dim;
    float local_threshold_sq;
    float adversarial_threshold_sq;
    uint64_t train_seed;
    uint32_t train_sample_size;
} ProbeFileHeader;
#pragma pack(pop)

extern "C" int geometric_probe_save(const GeometricProbe *probe, const char *filepath) {
    if (!probe || !filepath) return -1;
    FILE *f = fopen(filepath, "wb");
    if (!f) return -2;

    ProbeFileHeader hdr;
    hdr.magic = GPRB_MAGIC;
    hdr.version = GPRB_VERSION;
    hdr.num_prototypes = probe->num_prototypes;
    hdr.dim = probe->dim;
    hdr.local_threshold_sq = probe->local_threshold_sq;
    hdr.adversarial_threshold_sq = probe->adversarial_threshold_sq;
    hdr.train_seed = probe->train_seed;
    hdr.train_sample_size = probe->train_sample_size;

    fwrite(&hdr, sizeof(hdr), 1, f);
    fwrite(probe->prototypes, sizeof(float), (size_t)probe->num_prototypes * probe->dim, f);
    fclose(f);
    return 0;
}

extern "C" int geometric_probe_load(GeometricProbe *probe, const char *filepath) {
    if (!probe || !filepath) return -1;
    FILE *f = fopen(filepath, "rb");
    if (!f) return -2;

    ProbeFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.magic != GPRB_MAGIC || hdr.version != GPRB_VERSION) {
        fclose(f);
        return -3;
    }

    memset(probe, 0, sizeof(GeometricProbe));
    probe->num_prototypes = hdr.num_prototypes;
    probe->dim = hdr.dim;
    probe->local_threshold_sq = hdr.local_threshold_sq;
    probe->adversarial_threshold_sq = hdr.adversarial_threshold_sq;
    probe->train_seed = hdr.train_seed;
    probe->train_sample_size = hdr.train_sample_size;

    fread(probe->prototypes, sizeof(float), (size_t)probe->num_prototypes * probe->dim, f);
    fclose(f);
    return 0;
}
