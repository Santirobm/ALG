#include "../include/alg_hnsw.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_banner(void) {
    printf("========================================================================================================================\n");
    printf(" EXPERIMENTO H3: ABLACION DE GRANULARIDAD DE MICRO-TILES Y POINTER SWIZZLING EN HNSW FILTRADO\n");
    printf(" Comparativa Causal: H2 vs H3-64 vs H3-128 vs H3-256 | Intel Core i5-13500 P-Core 0\n");
    printf("========================================================================================================================\n\n");
}

int main(int argc, char **argv) {
    SetConsoleOutputCP(65001);
    setvbuf(stdout, NULL, _IONBF, 0);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);

    uint32_t num_vectors = 100000U;
    uint32_t num_queries = 1000U;
    uint32_t ef_search   = HNSW_EF_SEARCH;
    uint32_t selectivity = 5U; /* 5% representativo */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--num-vectors") == 0 && i + 1 < argc) num_vectors = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--num-queries") == 0 && i + 1 < argc) num_queries = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--selectivity") == 0 && i + 1 < argc) selectivity = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--ef-search") == 0 && i + 1 < argc) ef_search = (uint32_t)atoi(argv[++i]);
    }

    print_banner();
    printf(" [Configuración del Experimento]\n");
    printf("   - Cantidad de Vectores: %u (64-D float32 | Métrica: L2)\n", num_vectors);
    printf("   - Selectividad del Filtro: %u%% | efSearch: %u\n", selectivity, ef_search);
    printf("   - Consultas por Escenario: %u | Número de Semillas: 5\n\n", num_queries);

    const uint64_t SEEDS[5] = {
        42ULL,
        1337ULL,
        987654321ULL,
        0xCAFEBABEULL,
        0xDEADBEEFULL
    };

    CreateDirectoryA("results", NULL);
    FILE *csv = fopen("results/h3_ablation.csv", "w");
    if (csv) {
        fprintf(csv, "Seed,Scenario,Variant,Recall@10,Recall@100,Lat_p50_us,Lat_p95_us,Throughput_QPS,DistComps,TileHitRate,QueryTime_ms,MaintTime_ms,TotalTime_ms,WorkingSet_MB\n");
    }

    FilterPredicate pred = filter_create_predicate(selectivity);

    /* Estructuras de Micro-Tiles a comparar */
    const struct { SearchVariant v; uint32_t cap; const char *name; } VARIANTS[4] = {
        { VARIANT_H2_FILTERAWARE, 0U,   "H2 Baseline (Sin Tiles)   " },
        { VARIANT_H3_TILES_64,    64U,  "H3-64 (Micro-Tiles 64)    " },
        { VARIANT_H3_TILES_128,   128U, "H3-128 (Micro-Tiles 128)  " },
        { VARIANT_H3_TILES_256,   256U, "H3-256 (Micro-Tiles 256)  " }
    };

    /* Bucle principal sobre las 5 semillas */
    for (uint32_t s = 0; s < 5; s++) {
        uint64_t seed = SEEDS[s];
        printf("========================================================================================================================\n");
        printf(" SEMILLA %u / 5 [Seed: 0x%llX]\n", s + 1, seed);
        printf("========================================================================================================================\n");

        printf(" 1. Generando 100K vectores y construyendo índice HNSW base...\n");
        float *vectors = vectors_generate_synthetic(num_vectors, VECTOR_DIM, seed);
        AttributeSet *attrs = attributes_generate(num_vectors, seed);
        HnswIndex *index = hnsw_create(num_vectors, VECTOR_DIM, HNSW_M_DEFAULT, HNSW_EF_CONSTRUCT);

        LARGE_INTEGER tb0, tb1, freq;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&tb0);
        for (uint32_t i = 0; i < num_vectors; i++) {
            hnsw_insert(index, i, &vectors[i * VECTOR_DIM]);
        }
        QueryPerformanceCounter(&tb1);
        double bld_sec = (double)(tb1.QuadPart - tb0.QuadPart) / (double)freq.QuadPart;
        printf("    * Índice base construido en %.2f s\n", bld_sec);

        /* Crear motores de tiles para las tres capacidades */
        HnswTileEngine *eng_64  = hnsw_tile_engine_create(num_vectors, 64U, 128U);
        HnswTileEngine *eng_128 = hnsw_tile_engine_create(num_vectors, 128U, 128U);
        HnswTileEngine *eng_256 = hnsw_tile_engine_create(num_vectors, 256U, 128U);

        /* Evaluar Escenario R (Localidad/Reutilización) y Escenario D (Adversarial) */
        for (int sc = 0; sc < 2; sc++) {
            bool is_reuse = (sc == 0);
            const char *sc_name = is_reuse ? "Escenario R (Reutilización 85% Localidad)"
                                           : "Escenario D (Adversarial 0% Localidad)";
            printf("\n [%s]\n", sc_name);
            printf(" Variante                    | Recall@10 | Recall@100 | Lat p50 (us) | Lat p95 (us) | QPS     | HitRate | Maint(ms) | Ahorro p50\n");
            printf("-----------------------------+-----------+------------+--------------+--------------+---------+---------+-----------+-----------\n");

            float *queries = queries_generate_scenario(num_queries, VECTOR_DIM, vectors, num_vectors, is_reuse, seed ^ 0xABCDULL);
            SearchResult *ground_truth = (SearchResult *)malloc(num_queries * sizeof(SearchResult));
            for (uint32_t q = 0; q < num_queries; q++) {
                ground_truth[q] = ground_truth_linear_scan(vectors, num_vectors, VECTOR_DIM,
                                                            attrs, &pred, &queries[q * VECTOR_DIM], 100U);
            }

            double base_p50 = 0.0;

            for (int v_idx = 0; v_idx < 4; v_idx++) {
                SearchVariant v = VARIANTS[v_idx].v;
                HnswTileEngine *eng = (v == VARIANT_H3_TILES_64)  ? eng_64 :
                                      (v == VARIANT_H3_TILES_128) ? eng_128 :
                                      (v == VARIANT_H3_TILES_256) ? eng_256 : NULL;

                BenchmarkMetrics m;
                benchmark_evaluate_variant_ex(v, index, NULL, NULL, eng,
                                             attrs, &pred, queries, ground_truth,
                                             num_queries, ef_search, &m);

                if (v_idx == 0) base_p50 = m.latency_p50_us;
                double p50_gain = (base_p50 > 0.0) ? (100.0 * (base_p50 - m.latency_p50_us) / base_p50) : 0.0;

                printf(" %s |  %7.4f  |   %7.4f  |   %8.1f   |   %8.1f   | %7.1f |  %5.1f%% |  %7.2f  | %+5.1f%%\n",
                       VARIANTS[v_idx].name, m.recall_at_10, m.recall_at_100,
                       m.latency_p50_us, m.latency_p95_us, m.throughput_qps,
                       m.tile_hit_rate_pct, m.total_maint_ms, p50_gain);

                if (csv) {
                    double tot_ms = m.total_query_ms + m.total_maint_ms;
                    fprintf(csv, "%u,%s,%s,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                            s + 1, is_reuse ? "R_Local" : "D_Adversarial",
                            (v_idx == 0) ? "H2" : (v_idx == 1) ? "H3-64" : (v_idx == 2) ? "H3-128" : "H3-256",
                            m.recall_at_10, m.recall_at_100,
                            m.latency_p50_us, m.latency_p95_us, m.throughput_qps,
                            m.avg_distance_comps, m.tile_hit_rate_pct,
                            m.total_query_ms, m.total_maint_ms, tot_ms,
                            (double)m.mem_stats.working_set_bytes / (1024.0 * 1024.0));
                }
            }

            aligned_free_64(queries);
            free(ground_truth);
        }

        hnsw_tile_engine_free(eng_64);
        hnsw_tile_engine_free(eng_128);
        hnsw_tile_engine_free(eng_256);
        hnsw_free(index);
        attributes_free(attrs);
        aligned_free_64(vectors);
    }

    if (csv) {
        fclose(csv);
        printf("\n * Resultados completos exportados a 'results/h3_ablation.csv'\n");
    }

    return 0;
}
