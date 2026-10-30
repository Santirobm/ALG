#include "../include/alg_hnsw.h"
#include <stdlib.h>
#include <string.h>

static int cmp_dist_pair_asc(const void *a, const void *b) {
    float da = ((const DistPair *)a)->dist;
    float db = ((const DistPair *)b)->dist;
    return (da > db) - (da < db);
}

/*
 * Búsqueda Lineal Exhaustiva (Fuerza Bruta):
 * Referencia autoritativa de exactitud.
 * Evalúa únicamente los vectores que cumplen estrictamente con el predicado.
 */
SearchResult ground_truth_linear_scan(
    const float *vectors,
    uint32_t num_vectors,
    uint32_t dim,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    const float *query,
    uint32_t top_k
) {
    SearchResult res;
    memset(&res, 0, sizeof(SearchResult));
    res.k = top_k;

    if (top_k > 128U) top_k = 128U;

    DistPair *candidates = (DistPair *)malloc(num_vectors * sizeof(DistPair));
    uint32_t match_count = 0;

    for (uint32_t i = 0; i < num_vectors; i++) {
        if (filter_matches(attrs, i, pred)) {
            const float *v = &vectors[i * dim];
            float d = l2_distance_avx2(v, query, dim);
            candidates[match_count].id = i;
            candidates[match_count].dist = d;
            match_count++;
        }
    }

    if (match_count > 0) {
        qsort(candidates, match_count, sizeof(DistPair), cmp_dist_pair_asc);
        uint32_t ret = (match_count < top_k) ? match_count : top_k;
        res.returned_count = ret;
        for (uint32_t i = 0; i < ret; i++) {
            res.ids[i]   = candidates[i].id;
            res.dists[i] = candidates[i].dist;
        }
    } else {
        res.returned_count = 0;
    }

    res.distance_comps = match_count;
    res.visited_nodes  = num_vectors;

    free(candidates);
    return res;
}
