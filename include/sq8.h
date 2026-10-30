#ifndef ALG_HNSW_SQ8_H
#define ALG_HNSW_SQ8_H

#include "alg_hnsw.h"

typedef struct Sq8Index {
    uint32_t num_vectors;
    uint32_t dim;
    uint8_t *codes;       /* num_vectors * dim, row-major */
    float *min_values;    /* per-dimension affine offset */
    float *scales;        /* (max-min)/255 per dimension */
    size_t bytes;
} Sq8Index;

/* Modos de kernel:
 * 0: Baseline AVX2 (bucle de 8 con memcpy y acumulador único)
 * 1: Optimizado AVX2 U1 (unroll 16, cargas _mm_loadl directas, acumuladores duales independientes)
 * 2: Optimizado AVX2 U1 + Prefetch de software del siguiente vecino en HNSW
 */
extern int g_sq8_kernel_mode;

Sq8Index *sq8_build(const float *vectors, uint32_t num_vectors, uint32_t dim);
void sq8_free(Sq8Index *sq8);
void sq8_encode_query(const Sq8Index *sq8, const float *query, uint8_t *out_codes);
float l2_distance_sq8_avx2(const Sq8Index *sq8, const uint8_t *query_codes,
                           uint32_t vector_id);
float l2_distance_sq8_avx2_base(const Sq8Index *sq8, const uint8_t *query_codes,
                                uint32_t vector_id);
float l2_distance_sq8_avx2_opt(const Sq8Index *sq8, const uint8_t *query_codes,
                               uint32_t vector_id);

#endif
