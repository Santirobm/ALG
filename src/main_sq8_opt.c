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

static void run_sq8_case(uint32_t seed, const char *scenario, const char *name, int kernel_mode,
                         HnswIndex *index, const Sq8Index *sq8,
                         const AttributeSet *attrs, const FilterPredicate *pred,
                         const float *queries, const SearchResult *gt,
                         uint32_t nq, uint32_t ef, FILE *csv) {
    g_sq8_kernel_mode = kernel_mode;
    double *lat = (double *)malloc((size_t)nq * sizeof(double));
    double r10 = 0, r100 = 0;
    uint64_t nav = 0, rerank = 0, visited = 0;
    LARGE_INTEGER fq, t0, t1;
    QueryPerformanceFrequency(&fq);
    QueryPerformanceCounter(&t0);

    for (uint32_t q = 0; q < nq; q++) {
        LARGE_INTEGER a, b;
        QueryPerformanceCounter(&a);
        SearchResult r = hnsw_search_h2_sq8_filteraware(
            index, sq8, &queries[(size_t)q * index->dim], attrs, pred, 100, ef);
        QueryPerformanceCounter(&b);

        lat[q] = (double)(b.QuadPart - a.QuadPart) * 1e6 / (double)fq.QuadPart;
        r10 += recall_k(&r, &gt[q], 10);
        r100 += recall_k(&r, &gt[q], 100);
        nav += r.distance_comps;
        rerank += r.rerank_comps;
        visited += r.visited_nodes;
    }
    QueryPerformanceCounter(&t1);
    qsort(lat, nq, sizeof(double), cmp_double);

    double sec = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;
    double p50 = lat[(size_t)(nq * 0.50)];
    double p95 = lat[(size_t)(nq * 0.95)];
    double p99 = lat[(size_t)(nq * 0.99)];
    double qps = sec > 0 ? (double)nq / sec : 0;

    printf("%-18s [ef=%3u] R10 %.4f R100 %.4f p50 %6.2f us p95 %6.2f us p99 %6.2f us QPS %6.1f nav %5.1f rerank %4.1f\n",
           name, ef, r10 / nq, r100 / nq, p50, p95, p99, qps,
           (double)nav / nq, (double)rerank / nq);

    if (csv) {
        fprintf(csv, "%u,%s,%s,%u,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                seed, scenario, name, ef, r10 / nq, r100 / nq, p50, p95, p99, qps,
                (double)nav / nq, (double)rerank / nq, (double)visited / nq);
        fflush(csv);
    }
    free(lat);
}

int main(int argc, char **argv) {
    SetConsoleOutputCP(65001);
    setvbuf(stdout, NULL, _IONBF, 0);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);

    uint32_t nv = 100000, nq = 1000, sel = 5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--num-vectors") && i + 1 < argc) nv = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--num-queries") && i + 1 < argc) nq = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--selectivity") && i + 1 < argc) sel = (uint32_t)atoi(argv[++i]);
    }

    const uint64_t seeds[5] = {42ULL, 1337ULL, 987654321ULL, 0xCAFEBABEULL, 0xDEADBEEFULL};
    CreateDirectoryA("results", NULL);

    FILE *csv_ablation = fopen("results/sq8_kernel_ablation.csv", "w");
    if (csv_ablation) {
        fprintf(csv_ablation, "Seed,Scenario,Variant,efSearch,Recall10,Recall100,p50_us,p95_us,p99_us,QPS,NavDistanceComps,RerankComps,Visited\n");
    }

    FILE *csv_curve = fopen("results/sq8_efsearch_curve.csv", "w");
    if (csv_curve) {
        fprintf(csv_curve, "Seed,Scenario,Variant,efSearch,Recall10,Recall100,p50_us,p95_us,p99_us,QPS,NavDistanceComps,RerankComps,Visited\n");
    }

    FilterPredicate pred = filter_create_predicate(sel);
    g_hnsw_use_heaps = 1;

    printf("========================================================================\n");
    printf("ALG-HNSW Hito 4: Optimizacion de Kernel SQ8 & Curva Recall/Latencia\n");
    printf("N=%u | Consultas=%u | Selectividad=%u%% | 5 semillas | P-Core 0\n", nv, nq, sel);
    printf("========================================================================\n");

    const uint32_t ef_values[5] = {100, 150, 200, 300, 500};

    for (uint32_t si = 0; si < 5; si++) {
        printf("\n==================== SEED %u/5 ====================\n", si + 1);
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
        if (!sq8) {
            fprintf(stderr, "Fallo construyendo SQ8.\n");
            return 4;
        }
        printf("SQ8 construido en %.3f s | huella: %.2f MiB\n",
               (double)(b.QuadPart - a.QuadPart) / (double)fq.QuadPart,
               (double)sq8->bytes / 1048576.0);

        for (int sc = 0; sc < 2; sc++) {
            bool reuse = (sc == 0);
            const char *sn = reuse ? "R_Local" : "D_Adversarial";
            printf("\n--- Escenario %s ---\n", sn);

            float *queries = queries_generate_scenario(nq, VECTOR_DIM, vectors, nv, reuse, seeds[si] ^ 0xABCDULL);
            SearchResult *gt = (SearchResult *)malloc((size_t)nq * sizeof(SearchResult));
            if (!queries || !gt) {
                fprintf(stderr, "Fallo reservando memoria de consultas.\n");
                return 5;
            }

            for (uint32_t q = 0; q < nq; q++) {
                gt[q] = ground_truth_linear_scan(vectors, nv, VECTOR_DIM, attrs, &pred, &queries[(size_t)q * VECTOR_DIM], 100);
            }

            /* 1. Ablacion de Kernels SQ8 (efSearch = 100) */
            printf("\n[1. Ablacion de Kernels a efSearch=100]\n");
            run_sq8_case(si + 1, sn, "H2_SQ8_Base",     0, index, sq8, attrs, &pred, queries, gt, nq, 100, csv_ablation);
            run_sq8_case(si + 1, sn, "H2_SQ8_Opt",      1, index, sq8, attrs, &pred, queries, gt, nq, 100, csv_ablation);
            run_sq8_case(si + 1, sn, "H2_SQ8_Prefetch", 2, index, sq8, attrs, &pred, queries, gt, nq, 100, csv_ablation);

            /* 2. Barrido de Curva Recall/Latencia con kernel optimizado */
            printf("\n[2. Curva Recall/Latencia (efSearch: 100 -> 500)]\n");
            for (int efi = 0; efi < 5; efi++) {
                uint32_t ef = ef_values[efi];
                run_sq8_case(si + 1, sn, "H2_SQ8_Opt", 1, index, sq8, attrs, &pred, queries, gt, nq, ef, csv_curve);
            }

            aligned_free_64(queries);
            free(gt);
        }

        sq8_free(sq8);
        hnsw_free(index);
        attributes_free(attrs);
        aligned_free_64(vectors);
    }

    if (csv_ablation) fclose(csv_ablation);
    if (csv_curve) fclose(csv_curve);
    printf("\n========================================================================\n");
    printf("Experimento completado exitosamente.\n");
    return 0;
}
