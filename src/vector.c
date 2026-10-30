#include "../include/alg_hnsw.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline uint64_t xorshift64s_vec(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float uniform_float(uint64_t *state) {
    uint64_t r = xorshift64s_vec(state);
    return (float)(r & 0xFFFFFF) / (float)0x1000000;
}

/*
 * Genera N vectores en dimensión 64 con estructura de agrupamiento (clusters).
 * Los clusters permiten que el grafo HNSW desarrolle estructura jerárquica
 * representativa de embeddings de lenguaje o visión.
 */
float *vectors_generate_synthetic(uint32_t num_vectors, uint32_t dim, uint64_t seed) {
    size_t total_floats = (size_t)num_vectors * (size_t)dim;
    float *vecs = (float *)aligned_alloc_64(total_floats * sizeof(float));
    if (!vecs) return NULL;

    uint64_t rng = seed ^ 0x517CC1B727220A95ULL;

    /* Generar 64 centroides base */
    const uint32_t NUM_CENTROIDS = 64U;
    float *centroids = (float *)malloc(NUM_CENTROIDS * dim * sizeof(float));
    for (uint32_t c = 0; c < NUM_CENTROIDS; c++) {
        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            float v = uniform_float(&rng) * 2.0f - 1.0f;
            centroids[c * dim + d] = v;
            norm_sq += v * v;
        }
        float inv_norm = 1.0f / sqrtf(norm_sq ? norm_sq : 1.0f);
        for (uint32_t d = 0; d < dim; d++) {
            centroids[c * dim + d] *= inv_norm;
        }
    }

    /* Generar vectores con desplazamiento gaussiano/uniforme respecto a su centroide */
    for (uint32_t i = 0; i < num_vectors; i++) {
        uint32_t c = (uint32_t)(xorshift64s_vec(&rng) % NUM_CENTROIDS);
        const float *cent = &centroids[c * dim];
        float *v = &vecs[i * dim];

        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            float noise = (uniform_float(&rng) * 2.0f - 1.0f) * 0.15f;
            float val = cent[d] + noise;
            v[d] = val;
            norm_sq += val * val;
        }
        /* Normalizar a hiperesfera unidad */
        float inv_norm = 1.0f / sqrtf(norm_sq ? norm_sq : 1.0f);
        for (uint32_t d = 0; d < dim; d++) {
            v[d] *= inv_norm;
        }
    }

    free(centroids);
    return vecs;
}

float *queries_generate_scenario(
    uint32_t num_queries,
    uint32_t dim,
    const float *dataset_vectors,
    uint32_t num_vectors,
    bool is_reuse_scenario,
    uint64_t seed
) {
    float *queries = (float *)aligned_alloc_64((size_t)num_queries * (size_t)dim * sizeof(float));
    if (!queries) return NULL;

    uint64_t rng = seed ^ 0x9876543210ABCDEFULL;

    /* Elegir 8 cuencas de atracción calientes si es escenario de reutilización */
    const uint32_t NUM_BASINS = 8U;
    uint32_t basin_nodes[8];
    for (uint32_t b = 0; b < NUM_BASINS; b++) {
        basin_nodes[b] = (uint32_t)(xorshift64s_vec(&rng) % num_vectors);
    }

    for (uint32_t q = 0; q < num_queries; q++) {
        float *qv = &queries[q * dim];
        uint64_t r = xorshift64s_vec(&rng);

        if (is_reuse_scenario && ((r % 100U) < 85U)) {
            /* 85% de consultas concentradas en las 8 cuencas de atracción calientes */
            uint32_t chosen_basin = basin_nodes[(r >> 8) % NUM_BASINS];
            const float *base_v = &dataset_vectors[chosen_basin * dim];

            float norm_sq = 0.0f;
            for (uint32_t d = 0; d < dim; d++) {
                float noise = (uniform_float(&rng) * 2.0f - 1.0f) * 0.08f;
                float val = base_v[d] + noise;
                qv[d] = val;
                norm_sq += val * val;
            }
            float inv = 1.0f / sqrtf(norm_sq ? norm_sq : 1.0f);
            for (uint32_t d = 0; d < dim; d++) qv[d] *= inv;
        } else {
            /* 100% en Escenario D (o 15% de ruido en Escenario R): vectores aleatorios en hiperesfera */
            float norm_sq = 0.0f;
            for (uint32_t d = 0; d < dim; d++) {
                float val = uniform_float(&rng) * 2.0f - 1.0f;
                qv[d] = val;
                norm_sq += val * val;
            }
            float inv = 1.0f / sqrtf(norm_sq ? norm_sq : 1.0f);
            for (uint32_t d = 0; d < dim; d++) qv[d] *= inv;
        }
    }

    return queries;
}
