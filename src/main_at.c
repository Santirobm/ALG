#include "../include/alg_hnsw.h"
#include "../include/heap.h"
#include "../include/sq8.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_double(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static double recall_k(const SearchResult *p, const SearchResult *g, uint32_t k) {
    uint32_t n = g->returned_count < k ? g->returned_count : k;
    uint32_t m = 0;
    uint32_t pn = p->returned_count < k ? p->returned_count : k;
    if (!n) return 1.0;
    for (uint32_t i = 0; i < pn; i++) {
        for (uint32_t j = 0; j < n; j++) {
            if (p->ids[i] == g->ids[j]) { m++; break; }
        }
    }
    return (double)m / (double)n;
}

static void run_at_case(
    uint32_t seed, uint32_t sel, const char *scenario, const char *variant,
    int mode, uint32_t ef_result, uint32_t ef_explore,
    HnswIndex *index, const Sq8Index *sq8,
    const AttributeSet *attrs, const FilterPredicate *pred,
    const float *queries, const SearchResult *gt,
    uint32_t nq, FILE *csv
) {
    double *lat = (double *)malloc((size_t)nq * sizeof(double));
    double r10 = 0, r100 = 0;
    uint64_t nav_dist = 0, rerank = 0, visited = 0, expanded = 0, transit = 0;
    uint32_t leaks = 0;

    LARGE_INTEGER fq, t0, t1;
    QueryPerformanceFrequency(&fq);
    QueryPerformanceCounter(&t0);

    for (uint32_t q = 0; q < nq; q++) {
        LARGE_INTEGER a, b;
        QueryPerformanceCounter(&a);
        SearchResult r;
        if (mode == 0) {
            /* B0: H2 actual baseline */
            r = hnsw_search_h2_sq8_filteraware(
                index, sq8, &queries[(size_t)q * index->dim], attrs, pred, 100, ef_result);
        } else {
            /* T1: Adaptive Transit con presupuesto separado */
            r = hnsw_search_h2_at_sq8(
                index, sq8, &queries[(size_t)q * index->dim], attrs, pred, 100, ef_result, ef_explore);
        }
        QueryPerformanceCounter(&b);

        lat[q] = (double)(b.QuadPart - a.QuadPart) * 1e6 / (double)fq.QuadPart;
        r10 += recall_k(&r, &gt[q], 10);
        r100 += recall_k(&r, &gt[q], 100);
        nav_dist += r.distance_comps;
        rerank += r.rerank_comps;
        visited += r.visited_nodes;
        expanded += r.expanded_nodes;
        transit += r.transit_nodes;

        /* Verificacion de cero filtracion de no elegibles */
        for (uint32_t i = 0; i < r.returned_count; i++) {
            if (!filter_matches(attrs, r.ids[i], pred)) {
                leaks++;
            }
        }
    }
    QueryPerformanceCounter(&t1);
    qsort(lat, nq, sizeof(double), cmp_double);

    double sec = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;
    double p50 = lat[(size_t)(nq * 0.50)];
    double p95 = lat[(size_t)(nq * 0.95)];
    double p99 = lat[(size_t)(nq * 0.99)];
    double qps = sec > 0 ? (double)nq / sec : 0;
    double avg_exp = (double)expanded / nq;
    double avg_trans = (double)transit / nq;
    double pct_transit = avg_exp > 0 ? (avg_trans / avg_exp) * 100.0 : 0.0;

    printf("%-10s [sel=%2u%%] R10: %6.4f | R100: %6.4f | p50: %6.1f us | p95: %6.1f us | QPS: %5.1f | exp: %5.0f | trans%%: %4.1f%% | leaks: %u\n",
           variant, sel, r10 / nq, r100 / nq, p50, p95, qps, avg_exp, pct_transit, leaks);

    if (csv) {
        fprintf(csv, "%u,%u,%s,%s,%u,%u,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%u\n",
                seed, sel, scenario, variant, ef_result, ef_explore,
                r10 / nq, r100 / nq, p50, p95, p99, qps,
                (double)nav_dist / nq, (double)rerank / nq, (double)visited / nq,
                avg_exp, pct_transit, leaks);
        fflush(csv);
    }
    free(lat);
}

int main(int argc, char **argv) {
    SetConsoleOutputCP(65001);
    setvbuf(stdout, NULL, _IONBF, 0);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);

    uint32_t nv = 100000, nq = 1000, num_seeds = 5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--num-vectors") && i + 1 < argc) nv = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--num-queries") && i + 1 < argc) nq = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--num-seeds") && i + 1 < argc) num_seeds = (uint32_t)atoi(argv[++i]);
    }
    if (num_seeds > 5) num_seeds = 5;

    const uint64_t seeds[5] = {42ULL, 1337ULL, 987654321ULL, 0xCAFEBABEULL, 0xDEADBEEFULL};
    CreateDirectoryA("results", NULL);

    FILE *csv = fopen("results/at_transit_ablation.csv", "w");
    if (csv) {
        fprintf(csv, "Seed,Selectivity,Scenario,Variant,ef_result,ef_explore,Recall10,Recall100,p50_us,p95_us,p99_us,QPS,NavDistComps,RerankComps,VisitedNodes,ExpandedNodes,PctTransit,Leaks\n");
    }

    g_hnsw_use_heaps = 1;
    g_sq8_kernel_mode = 1; /* U1 optimizado */

    printf("========================================================================================\n");
    printf("ALG-HNSW Hito 5: Rediseño de Frontera de Transito (H2-AT)\n");
    printf("N=%u | Consultas=%u | Semillas=%u | P-Core 0 | Selectividades: 1%%, 5%%, 10%%, 25%%, 50%%\n",
           nv, nq, num_seeds);
    printf("========================================================================================\n");

    const uint32_t selectivities[5] = {1, 5, 10, 25, 50};

    for (uint32_t si = 0; si < num_seeds; si++) {
        printf("\n============================= SEED %u/%u =============================\n", si + 1, num_seeds);
        float *vectors = vectors_generate_synthetic(nv, VECTOR_DIM, seeds[si]);
        AttributeSet *attrs = attributes_generate(nv, seeds[si]);
        HnswIndex *index = hnsw_create(nv, VECTOR_DIM, HNSW_M_DEFAULT, HNSW_EF_CONSTRUCT);

        if (!vectors || !attrs || !index) {
            fprintf(stderr, "Error asignando memoria de grafo.\n");
            return 3;
        }

        LARGE_INTEGER fq, a, b;
        QueryPerformanceFrequency(&fq);
        QueryPerformanceCounter(&a);
        for (uint32_t i = 0; i < nv; i++) {
            hnsw_insert(index, i, &vectors[(size_t)i * VECTOR_DIM]);
        }
        QueryPerformanceCounter(&b);
        printf("HNSW construido en %.2f s\n", (double)(b.QuadPart - a.QuadPart) / (double)fq.QuadPart);

        QueryPerformanceCounter(&a);
        Sq8Index *sq8 = sq8_build(vectors, nv, VECTOR_DIM);
        QueryPerformanceCounter(&b);
        printf("SQ8 construido en %.3f s | huella: %.2f MiB\n",
               (double)(b.QuadPart - a.QuadPart) / (double)fq.QuadPart,
               (double)sq8->bytes / 1048576.0);

        for (int sc = 0; sc < 2; sc++) {
            bool reuse = (sc == 0);
            const char *sn = reuse ? "R_Local" : "D_Adversarial";
            printf("\n-------------------- Escenario %s --------------------\n", sn);

            float *queries = queries_generate_scenario(nq, VECTOR_DIM, vectors, nv, reuse, seeds[si] ^ 0xABCDULL);
            SearchResult *gt = (SearchResult *)malloc((size_t)nq * sizeof(SearchResult));
            if (!queries || !gt) {
                fprintf(stderr, "Fallo memoria consultas.\n");
                return 4;
            }

            for (int seli = 0; seli < 5; seli++) {
                uint32_t sel = selectivities[seli];
                FilterPredicate pred = filter_create_predicate(sel);

                /* Ground truth exacto brute-force para esta selectividad */
                for (uint32_t q = 0; q < nq; q++) {
                    gt[q] = ground_truth_linear_scan(vectors, nv, VECTOR_DIM, attrs, &pred, &queries[(size_t)q * VECTOR_DIM], 100);
                }

                printf("\n--- Selectividad %2u%% (Disponibles: ~%u vectores elegibles) ---\n", sel, (nv * sel) / 100);
                run_at_case(si + 1, sel, sn, "B0_H2_Base", 0, 100, 100,  index, sq8, attrs, &pred, queries, gt, nq, csv);
                run_at_case(si + 1, sel, sn, "T1_A_512",   1, 100, 512,  index, sq8, attrs, &pred, queries, gt, nq, csv);
                run_at_case(si + 1, sel, sn, "T1_B_1024",  1, 100, 1024, index, sq8, attrs, &pred, queries, gt, nq, csv);
                run_at_case(si + 1, sel, sn, "T1_C_2048",  1, 100, 2048, index, sq8, attrs, &pred, queries, gt, nq, csv);
                run_at_case(si + 1, sel, sn, "T1_D_4096",  1, 100, 4096, index, sq8, attrs, &pred, queries, gt, nq, csv);
            }

            aligned_free_64(queries);
            free(gt);
        }

        sq8_free(sq8);
        hnsw_free(index);
        attributes_free(attrs);
        aligned_free_64(vectors);
    }

    if (csv) fclose(csv);
    printf("\n========================================================================================\n");
    printf("Hito 5 completado exitosamente. Resultados en results/at_transit_ablation.csv\n");
    return 0;
}
