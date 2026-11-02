#include "dataset_format.h"
#include "../../include/alg_hnsw.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>


static inline uint64_t h7_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float h7_uniform(uint64_t *state) {
    return (float)(h7_rng_next(state) & 0xFFFFFFULL) / (float)0x1000000;
}

/* Same 64 centroid coordinates for every seed; assignments and noise vary by seed. */
static float *vectors_generate_hito7(uint32_t nv, uint32_t dim, uint64_t data_seed) {
    const uint32_t n_centroids = 64U;
    float *vectors = (float *)aligned_alloc_64((size_t)nv * dim * sizeof(float));
    float *centroids = (float *)malloc((size_t)n_centroids * dim * sizeof(float));
    if (!vectors || !centroids) {
        if (vectors) aligned_free_64(vectors);
        free(centroids);
        return NULL;
    }

    uint64_t centroid_rng = 0x4849544F375F434FULL; /* fixed Hito 7 centroid seed */
    uint64_t point_rng = data_seed ^ 0x517CC1B727220A95ULL;
    for (uint32_t c = 0; c < n_centroids; c++) {
        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            float v = h7_uniform(&centroid_rng) * 2.0f - 1.0f;
            centroids[(size_t)c * dim + d] = v;
            norm_sq += v * v;
        }
        float inv = 1.0f / sqrtf(norm_sq ? norm_sq : 1.0f);
        for (uint32_t d = 0; d < dim; d++) centroids[(size_t)c * dim + d] *= inv;
    }

    for (uint32_t i = 0; i < nv; i++) {
        uint32_t c = (uint32_t)(h7_rng_next(&point_rng) % n_centroids);
        const float *center = &centroids[(size_t)c * dim];
        float *v = &vectors[(size_t)i * dim];
        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            float x = center[d] + (h7_uniform(&point_rng) * 2.0f - 1.0f) * 0.15f;
            v[d] = x; norm_sq += x * x;
        }
        float inv = 1.0f / sqrtf(norm_sq ? norm_sq : 1.0f);
        for (uint32_t d = 0; d < dim; d++) v[d] *= inv;
    }
    free(centroids);
    return vectors;
}

int main(int argc, char **argv) {
    SetConsoleOutputCP(65001);
    uint32_t seed_idx = 1;
    uint32_t nv = 100000;
    uint32_t nq = 1000;
    
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed-idx") && i + 1 < argc) seed_idx = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--num-vectors") && i + 1 < argc) nv = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--num-queries") && i + 1 < argc) nq = (uint32_t)atoi(argv[++i]);
    }

    const uint64_t seeds[5] = {42ULL, 1337ULL, 987654321ULL, 0xCAFEBABEULL, 0xDEADBEEFULL};
    if (seed_idx < 1 || seed_idx > 5) seed_idx = 1;
    uint64_t seed_val = seeds[seed_idx - 1];

    printf("========================================================================\n");
    printf("Exportando Dataset Comun Hito 7: Semilla %u (0x%llX)\n", seed_idx, seed_val);
    printf("N=%u | Dim=64 | Consultas=%u | Selectividades: 10.0%%, 12.5%%, 15.0%%, 17.5%%, 20.0%%, 22.5%%, 25.0%%\n", nv, nq);
    printf("========================================================================\n");

    const uint32_t selectivities[NUM_SELECTIVITIES] = {100, 125, 150, 175, 200, 225, 250}; /* 10.0..25.0% */
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

    FilterPredicate preds[NUM_SELECTIVITIES];
    for (uint32_t i = 0; i < NUM_SELECTIVITIES; i++) {
        header.selectivities[i] = selectivities[i];
        memset(&preds[i], 0, sizeof(preds[i]));
        preds[i].selectivity_pct = 0U; /* fuerza la rama category < target */
        preds[i].target_category = selectivities[i] * 10U; /* category range [0,9999] */
        preds[i].min_region = 0U; preds[i].max_region = 15U; preds[i].max_price = 1000.0f;
        header.target_categories[i] = preds[i].target_category;
    }

    printf("Generando vectores de base sintéticos...\n");
    float *vectors = vectors_generate_hito7(nv, VECTOR_DIM, seed_val);
    AttributeSet *attrs = attributes_generate(nv, seed_val);
    if (attrs) {
        /* Expand category precision from 100 to 10,000 buckets for half-percent selectivity steps. */
        for (uint32_t i = 0; i < nv; i++) {
            uint64_t z = seed_val + (uint64_t)i * 0x9E3779B97F4A7C15ULL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z ^= z >> 31;
            attrs->category[i] = (uint32_t)(z % 10000ULL);
        }
    }
    if (!vectors || !attrs) { fprintf(stderr, "Error generando vectores/atributos.\n"); return 1; }

    printf("Generando consultas escenario R (local, 85%% reuso)...\n");
    float *queries_R = queries_generate_scenario(nq, VECTOR_DIM, vectors, nv, true, seed_val ^ 0xABCDULL);
    printf("Generando consultas escenario D (adversario, 0%% reuso)...\n");
    float *queries_D = queries_generate_scenario(nq, VECTOR_DIM, vectors, nv, false, seed_val ^ 0xABCDULL);
    if (!queries_R || !queries_D) { fprintf(stderr, "Error generando consultas.\n"); return 2; }

    char out_path[256];
    CreateDirectoryA("hito7", NULL);
    CreateDirectoryA("hito7/data", NULL);
    snprintf(out_path, sizeof(out_path), "hito7/data/dataset_seed%u.bin", seed_idx);
    FILE *f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "Error abriendo %s para escritura.\n", out_path); return 3; }

    /* Escribir encabezado */
    fwrite(&header, sizeof(header), 1, f);

    /* Escribir vectores */
    fwrite(vectors, sizeof(float), (size_t)nv * VECTOR_DIM, f);

    /* Escribir categorias */
    fwrite(attrs->category, sizeof(uint32_t), (size_t)nv, f);

    /* Escribir consultas R y D */
    fwrite(queries_R, sizeof(float), (size_t)nq * VECTOR_DIM, f);
    fwrite(queries_D, sizeof(float), (size_t)nq * VECTOR_DIM, f);

    /* Computar y escribir Ground Truth para cada escenario y selectividad */
    printf("Calculando Ground Truth exacto (fuerza bruta) para los 14 casos...\n");
    for (int sc = 0; sc < 2; sc++) {
        const float *q_ptr = (sc == 0) ? queries_R : queries_D;
        const char *sc_name = (sc == 0) ? "R_Local" : "D_Adversarial";

        for (uint32_t seli = 0; seli < NUM_SELECTIVITIES; seli++) {
            uint32_t sel = selectivities[seli];
            printf("  -> GT %s | Selectividad %u.%u%%...\n", sc_name, sel / 10U, sel % 10U);

            for (uint32_t q = 0; q < nq; q++) {
                SearchResult r = ground_truth_linear_scan(
                    vectors, nv, VECTOR_DIM, attrs, &preds[seli], &q_ptr[(size_t)q * VECTOR_DIM], TOP_K_GT);

                /* Escribir returned_count */
                uint32_t ret = r.returned_count;
                fwrite(&ret, sizeof(uint32_t), 1, f);

                /* Escribir IDs y dists (hasta TOP_K_GT) */
                fwrite(r.ids, sizeof(uint32_t), TOP_K_GT, f);
                fwrite(r.dists, sizeof(float), TOP_K_GT, f);
            }
        }
    }

    fclose(f);
    printf("Exportacion completada exitosamente en %s (formato v2; selectividades en decimas de porcentaje).\n", out_path);

    aligned_free_64(vectors);
    attributes_free(attrs);
    aligned_free_64(queries_R);
    aligned_free_64(queries_D);
    return 0;
}
