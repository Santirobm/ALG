#include "../include/alg_hnsw.h"
#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double calculate_recall_k(const SearchResult *pred, const SearchResult *gt, uint32_t k) {
    uint32_t gt_k = (gt->returned_count < k) ? gt->returned_count : k;
    if (gt_k == 0) return 1.0;

    uint32_t pred_k = (pred->returned_count < k) ? pred->returned_count : k;
    uint32_t matches = 0;

    for (uint32_t i = 0; i < pred_k; i++) {
        uint32_t p_id = pred->ids[i];
        for (uint32_t j = 0; j < gt_k; j++) {
            if (p_id == gt->ids[j]) {
                matches++;
                break;
            }
        }
    }

    return (double)matches / (double)gt_k;
}

void benchmark_evaluate_variant_ex(
    SearchVariant variant,
    HnswIndex *index,
    const HnswIndex *prefiltered_index,
    const uint32_t *h1_id_mapping,
    HnswTileEngine *tile_engine,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    const float *query_vectors,
    const SearchResult *ground_truth,
    uint32_t num_queries,
    uint32_t ef_search,
    BenchmarkMetrics *out
) {
    memset(out, 0, sizeof(BenchmarkMetrics));

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    double ns_per_tick = 1e9 / (double)freq.QuadPart;

    double *latencies = (double *)malloc(num_queries * sizeof(double));
    double recall10_sum = 0.0;
    double recall100_sum = 0.0;
    uint64_t total_dist_comps = 0;
    uint64_t total_visited = 0;
    uint64_t total_hot_hits = 0;

    if (tile_engine) {
        hnsw_tile_engine_reset(tile_engine, index);
    }

    /* Asegurar que ef_search permita devolver hasta 100 resultados para Recall@100 */
    uint32_t actual_ef = (ef_search < 100U) ? 100U : ef_search;

    LARGE_INTEGER total_t0, total_t1;
    QueryPerformanceCounter(&total_t0);

    for (uint32_t q = 0; q < num_queries; q++) {
        const float *query = &query_vectors[q * index->dim];

        LARGE_INTEGER q0, q1;
        QueryPerformanceCounter(&q0);

        SearchResult res;
        switch (variant) {
            case VARIANT_H0_POSTFILTER:
                res = hnsw_search_h0_postfilter(index, query, attrs, pred, 100U, actual_ef);
                break;
            case VARIANT_H1_PREFILTER:
                res = hnsw_search_h1_prefilter(prefiltered_index, h1_id_mapping, query, 100U, actual_ef);
                break;
            case VARIANT_H2_FILTERAWARE:
                res = hnsw_search_h2_filteraware(index, query, attrs, pred, 100U, actual_ef);
                break;
            case VARIANT_H3_TILES_64:
            case VARIANT_H3_TILES_128:
            case VARIANT_H3_TILES_256:
                res = hnsw_search_h3_swizzled(index, tile_engine, query, attrs, pred, 100U, actual_ef, true);
                break;
            default:
                memset(&res, 0, sizeof(SearchResult));
                break;
        }

        QueryPerformanceCounter(&q1);
        double lat_us = (double)(q1.QuadPart - q0.QuadPart) * ns_per_tick / 1000.0;
        latencies[q] = lat_us;

        recall10_sum  += calculate_recall_k(&res, &ground_truth[q], 10U);
        recall100_sum += calculate_recall_k(&res, &ground_truth[q], 100U);
        total_dist_comps += res.distance_comps;
        total_visited    += res.visited_nodes;
        total_hot_hits   += res.tile_hot_hits;
    }

    QueryPerformanceCounter(&total_t1);
    double total_sec = (double)(total_t1.QuadPart - total_t0.QuadPart) / (double)freq.QuadPart;

    qsort(latencies, num_queries, sizeof(double), cmp_double_asc);

    out->recall_at_10       = recall10_sum / (double)num_queries;
    out->recall_at_100      = recall100_sum / (double)num_queries;
    out->latency_p50_us     = latencies[(size_t)(num_queries * 0.50)];
    out->latency_p95_us     = latencies[(size_t)(num_queries * 0.95)];
    out->latency_p99_us     = latencies[(size_t)(num_queries * 0.99)];
    out->throughput_qps     = (total_sec > 0.0) ? ((double)num_queries / total_sec) : 0.0;
    out->avg_distance_comps = (double)total_dist_comps / (double)num_queries;
    out->avg_visited_nodes  = (double)total_visited / (double)num_queries;

    if (tile_engine) {
        out->tile_hit_rate_pct = total_visited ? (100.0 * (double)total_hot_hits / (double)total_visited) : 0.0;
        out->total_query_ms    = tile_engine->query_ns_sum / 1e6;
        out->total_maint_ms    = tile_engine->maint_ns_sum / 1e6;
        out->promotions        = tile_engine->promotions;
    }

    memory_get_stats(&out->mem_stats);

    free(latencies);
}
