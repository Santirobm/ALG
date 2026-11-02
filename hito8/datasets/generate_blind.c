#include "../../hito7/common/dataset_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

#define VECTOR_DIM 64U

static inline uint64_t h8_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float h8_uniform(uint64_t *state) {
    return (float)(h8_rng_next(state) & 0xFFFFFFULL) / (float)0x1000000;
}

/* Box-Muller transform for independent Gaussian noise */
static inline float h8_gaussian(uint64_t *state, float mean, float std) {
    float u1 = h8_uniform(state);
    float u2 = h8_uniform(state);
    if (u1 <= 1e-7f) u1 = 1e-7f;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.1415926535f * u2);
    return mean + z0 * std;
}

static inline float l2_dist_avx2(const float *a, const float *b, uint32_t dim) {
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

typedef struct {
    float dist;
    uint32_t id;
} GtPair;

static int cmp_gt_pair(const void *a, const void *b) {
    float da = ((const GtPair *)a)->dist;
    float db = ((const GtPair *)b)->dist;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    SetConsoleOutputCP(65001);
    uint32_t seed_idx = 6;
    uint32_t nv = 100000;
    uint32_t nq = 1000;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed-idx") && i + 1 < argc) seed_idx = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--num-vectors") && i + 1 < argc) nv = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--num-queries") && i + 1 < argc) nq = (uint32_t)atoi(argv[++i]);
    }

    uint64_t seed_val = (seed_idx == 6) ? 0x1122334455667788ULL : 0x99AABBCCDDEEFF00ULL;

    printf("========================================================================\n");
    printf("Generador de Dataset Ciego Hito 8 (Out-of-Sample): Semilla %u (0x%llX)\n", seed_idx, seed_val);
    printf("N=%u | Dim=64 | Consultas=%u | Selectividades: 10.0%% - 25.0%%\n", nv, nq);
    printf("========================================================================\n");

    const uint32_t selectivities[NUM_SELECTIVITIES] = {100, 125, 150, 175, 200, 225, 250};
    DatasetHeader header;
    memset(&header, 0, sizeof(header));
    header.magic = DATASET_MAGIC;
    header.version = DATASET_VERSION;
    header.seed_index = seed_idx;
    header.seed_value = seed_val;
    header.num_vectors = nv;
    header.dim = VECTOR_DIM;
    header.num_queries = nq;
    header.num_selectivities = NUM_SELECTIVITIES;

    for (uint32_t i = 0; i < NUM_SELECTIVITIES; i++) {
        header.selectivities[i] = selectivities[i];
        header.target_categories[i] = selectivities[i] * 10U; /* 1000..2500 */
    }

    /* 1. Generar 64 centroides completamente nuevos e independientes para esta semilla */
    const uint32_t n_centroids = 64U;
    float *centroids = (float *)malloc((size_t)n_centroids * VECTOR_DIM * sizeof(float));
    uint64_t centroid_rng = seed_val ^ 0x9988776655443322ULL;
    for (uint32_t c = 0; c < n_centroids; c++) {
        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < VECTOR_DIM; d++) {
            float v = h8_uniform(&centroid_rng) * 2.0f - 1.0f;
            centroids[(size_t)c * VECTOR_DIM + d] = v;
            norm_sq += v * v;
        }
        float inv = 1.0f / sqrtf(norm_sq ? norm_sq : 1.0f);
        for (uint32_t d = 0; d < VECTOR_DIM; d++) centroids[(size_t)c * VECTOR_DIM + d] *= inv;
    }

    /* 2. Generar vectores del dataset con ruido gaussiano independiente */
    float *vectors = (float *)_aligned_malloc((size_t)nv * VECTOR_DIM * sizeof(float), 64);
    uint32_t *categories = (uint32_t *)malloc((size_t)nv * sizeof(uint32_t));
    uint64_t point_rng = seed_val ^ 0x3344556677889900ULL;

    for (uint32_t i = 0; i < nv; i++) {
        uint32_t c = (uint32_t)(h8_rng_next(&point_rng) % n_centroids);
        const float *center = &centroids[(size_t)c * VECTOR_DIM];
        float *v = &vectors[(size_t)i * VECTOR_DIM];
        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < VECTOR_DIM; d++) {
            float x = center[d] + h8_gaussian(&point_rng, 0.0f, 0.09f);
            v[d] = x; norm_sq += x * x;
        }
        float inv = 1.0f / sqrtf(norm_sq ? norm_sq : 1.0f);
        for (uint32_t d = 0; d < VECTOR_DIM; d++) v[d] *= inv;

        /* Generar categorias con hash uniforme entre 0 y 9999 */
        uint64_t z = seed_val + (uint64_t)i * 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        categories[i] = (uint32_t)(z % 10000ULL);
    }
    free(centroids);

    /* 3. Generar consultas R (Local, 85% concentradas en 8 cuencas) y D (Adversario) */
    float *queries_R = (float *)_aligned_malloc((size_t)nq * VECTOR_DIM * sizeof(float), 64);
    float *queries_D = (float *)_aligned_malloc((size_t)nq * VECTOR_DIM * sizeof(float), 64);
    uint64_t query_rng = seed_val ^ 0xFEEDFACECAFEBEEFULL;

    const uint32_t NUM_BASINS = 8U;
    uint32_t basin_nodes[8];
    for (uint32_t b = 0; b < NUM_BASINS; b++) {
        basin_nodes[b] = (uint32_t)(h8_rng_next(&query_rng) % nv);
    }

    /* Consultas R */
    for (uint32_t q = 0; q < nq; q++) {
        float *qv = &queries_R[(size_t)q * VECTOR_DIM];
        uint64_t r = h8_rng_next(&query_rng);
        if ((r % 100U) < 85U) {
            uint32_t b_idx = basin_nodes[(r >> 8) % NUM_BASINS];
            const float *base_v = &vectors[(size_t)b_idx * VECTOR_DIM];
            float norm_sq = 0.0f;
            for (uint32_t d = 0; d < VECTOR_DIM; d++) {
                float val = base_v[d] + h8_gaussian(&query_rng, 0.0f, 0.05f);
                qv[d] = val; norm_sq += val * val;
            }
            float inv = 1.0f / sqrtf(norm_sq ? norm_sq : 1.0f);
            for (uint32_t d = 0; d < VECTOR_DIM; d++) qv[d] *= inv;
        } else {
            float norm_sq = 0.0f;
            for (uint32_t d = 0; d < VECTOR_DIM; d++) {
                float val = h8_uniform(&query_rng) * 2.0f - 1.0f;
                qv[d] = val; norm_sq += val * val;
            }
            float inv = 1.0f / sqrtf(norm_sq ? norm_sq : 1.0f);
            for (uint32_t d = 0; d < VECTOR_DIM; d++) qv[d] *= inv;
        }
    }

    /* Consultas D */
    for (uint32_t q = 0; q < nq; q++) {
        float *qv = &queries_D[(size_t)q * VECTOR_DIM];
        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < VECTOR_DIM; d++) {
            float val = h8_uniform(&query_rng) * 2.0f - 1.0f;
            qv[d] = val; norm_sq += val * val;
        }
        float inv = 1.0f / sqrtf(norm_sq ? norm_sq : 1.0f);
        for (uint32_t d = 0; d < VECTOR_DIM; d++) qv[d] *= inv;
    }

    CreateDirectoryA("hito8", NULL);
    CreateDirectoryA("hito8/data", NULL);
    char out_path[256];
    snprintf(out_path, sizeof(out_path), "hito8/data/dataset_seed%u.bin", seed_idx);
    FILE *f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "Error abriendo archivo %s\n", out_path); return 1; }

    fwrite(&header, sizeof(header), 1, f);
    fwrite(vectors, sizeof(float), (size_t)nv * VECTOR_DIM, f);
    fwrite(categories, sizeof(uint32_t), (size_t)nv, f);
    fwrite(queries_R, sizeof(float), (size_t)nq * VECTOR_DIM, f);
    fwrite(queries_D, sizeof(float), (size_t)nq * VECTOR_DIM, f);

    printf("Calculando Ground Truth exacto (14 casos: R/D x 7 selectividades)...\n");
    GtPair *candidates = (GtPair *)malloc((size_t)nv * sizeof(GtPair));

    for (int sc = 0; sc < 2; sc++) {
        const float *q_ptr = (sc == 0) ? queries_R : queries_D;
        const char *sc_name = (sc == 0) ? "R_Local" : "D_Adversarial";

        for (uint32_t seli = 0; seli < NUM_SELECTIVITIES; seli++) {
            uint32_t target_cat = header.target_categories[seli];
            printf("  -> GT %s | Selectividad %u.%u%% (cat < %u)...\n",
                   sc_name, selectivities[seli] / 10U, selectivities[seli] % 10U, target_cat);

            for (uint32_t q = 0; q < nq; q++) {
                const float *query = q_ptr + (size_t)q * VECTOR_DIM;
                uint32_t match_count = 0;

                for (uint32_t i = 0; i < nv; i++) {
                    if (categories[i] < target_cat) {
                        candidates[match_count].id = i;
                        candidates[match_count].dist = l2_dist_avx2(vectors + (size_t)i * VECTOR_DIM, query, VECTOR_DIM);
                        match_count++;
                    }
                }

                qsort(candidates, match_count, sizeof(GtPair), cmp_gt_pair);
                uint32_t ret = (match_count < TOP_K_GT) ? match_count : TOP_K_GT;
                fwrite(&ret, sizeof(uint32_t), 1, f);

                uint32_t ids[TOP_K_GT];
                float dists[TOP_K_GT];
                memset(ids, 0, sizeof(ids));
                memset(dists, 0, sizeof(dists));
                for (uint32_t k = 0; k < ret; k++) {
                    ids[k] = candidates[k].id;
                    dists[k] = candidates[k].dist;
                }
                fwrite(ids, sizeof(uint32_t), TOP_K_GT, f);
                fwrite(dists, sizeof(float), TOP_K_GT, f);
            }
        }
    }

    fclose(f);
    printf("Dataset ciego guardado exitosamente en %s\n", out_path);

    free(candidates);
    _aligned_free(vectors);
    free(categories);
    _aligned_free(queries_R);
    _aligned_free(queries_D);
    return 0;
}
