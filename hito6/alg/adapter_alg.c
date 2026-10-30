#include "../common/dataset_format.h"
#include "../../include/alg_hnsw.h"
#include "../../include/sq8.h"
#include "../../include/heap.h"
#include <windows.h>
#include <psapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static MemoryStats get_memory_stats(void) {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    memset(&pmc, 0, sizeof(pmc));
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc))) {
        MemoryStats m = { pmc.WorkingSetSize, pmc.PrivateUsage };
        return m;
    }
    MemoryStats m = { 0, 0 };
    return m;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

typedef struct {
    uint32_t count;
    uint32_t ids[TOP_K_GT];
    float dists[TOP_K_GT];
} QueryGT;

static double recall_at_k(const SearchResult *pred, const QueryGT *gt, uint32_t k) {
    uint32_t n_gt = gt->count < k ? gt->count : k;
    if (n_gt == 0) return 1.0;
    uint32_t n_pred = pred->returned_count < k ? pred->returned_count : k;
    uint32_t match = 0;
    for (uint32_t i = 0; i < n_pred; i++) {
        for (uint32_t j = 0; j < n_gt; j++) {
            if (pred->ids[i] == gt->ids[j]) {
                match++;
                break;
            }
        }
    }
    return (double)match / (double)n_gt;
}

int main(int argc, char **argv) {
    SetConsoleOutputCP(65001);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);

    const char *data_file = "hito6/data/dataset_seed1.bin";
    if (argc > 1) data_file = argv[1];

    printf("========================================================================\n");
    printf("Hito 6: Adaptador ALG-HNSW (Congelado) sobre Dataset Comun\n");
    printf("Archivo: %s\n", data_file);
    printf("========================================================================\n");

    FILE *f = fopen(data_file, "rb");
    if (!f) {
        fprintf(stderr, "Error abriendo archivo de dataset: %s\n", data_file);
        return 1;
    }

    DatasetHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1 || header.magic != DATASET_MAGIC) {
        fprintf(stderr, "Formato invalido en header.\n");
        fclose(f);
        return 2;
    }

    uint32_t nv = header.num_vectors;
    uint32_t dim = header.dim;
    uint32_t nq = header.num_queries;

    printf("Cargando datos: %u vectores (dim=%u), %u consultas...\n", nv, dim, nq);
    float *vectors = (float *)malloc((size_t)nv * dim * sizeof(float));
    AttributeSet attrs;
    attrs.num_elements = nv;
    attrs.category = (uint32_t *)malloc(nv * sizeof(uint32_t));
    attrs.region = NULL;
    attrs.price = NULL;

    float *queries_R = (float *)malloc((size_t)nq * dim * sizeof(float));
    float *queries_D = (float *)malloc((size_t)nq * dim * sizeof(float));

    fread(vectors, sizeof(float), (size_t)nv * dim, f);
    fread(attrs.category, sizeof(uint32_t), nv, f);
    fread(queries_R, sizeof(float), (size_t)nq * dim, f);
    fread(queries_D, sizeof(float), (size_t)nq * dim, f);

    QueryGT *ground_truth[2][NUM_SELECTIVITIES];
    for (int sc = 0; sc < 2; sc++) {
        for (uint32_t seli = 0; seli < NUM_SELECTIVITIES; seli++) {
            ground_truth[sc][seli] = (QueryGT *)malloc(nq * sizeof(QueryGT));
            for (uint32_t q = 0; q < nq; q++) {
                QueryGT *gt = &ground_truth[sc][seli][q];
                fread(&gt->count, sizeof(uint32_t), 1, f);
                fread(gt->ids, sizeof(uint32_t), TOP_K_GT, f);
                fread(gt->dists, sizeof(float), TOP_K_GT, f);
            }
        }
    }
    fclose(f);
    printf("Datos y Ground Truth cargados correctamente.\n");

    /* Construir índice HNSW autoritativo de ALG */
    MemoryStats mem_before = get_memory_stats();
    LARGE_INTEGER fq, t0, t1;
    QueryPerformanceFrequency(&fq);

    printf("\nConstruyendo indice HNSW de ALG (M=16, efConstruction=64)...\n");
    QueryPerformanceCounter(&t0);
    HnswIndex *index = hnsw_create(nv, dim, HNSW_M_DEFAULT, HNSW_EF_CONSTRUCT);
    for (uint32_t i = 0; i < nv; i++) {
        hnsw_insert(index, i, &vectors[(size_t)i * dim]);
    }
    QueryPerformanceCounter(&t1);
    double build_hnsw_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;

    QueryPerformanceCounter(&t0);
    Sq8Index *sq8 = sq8_build(vectors, nv, dim);
    QueryPerformanceCounter(&t1);
    double build_sq8_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;
    MemoryStats mem_after = get_memory_stats();

    double total_build_sec = build_hnsw_sec + build_sq8_sec;
    double index_mem_mb = (double)(mem_after.working_set_bytes - mem_before.working_set_bytes) / 1048576.0;
    printf("ALG construido en %.2f s (HNSW: %.2f s, SQ8: %.3f s) | Memoria residente: %.2f MiB (SQ8 payload: %.2f MiB)\n",
           total_build_sec, build_hnsw_sec, build_sq8_sec, index_mem_mb, (double)sq8->bytes / 1048576.0);

    CreateDirectoryA("hito6/results", NULL);
    FILE *csv_summary = fopen("hito6/results/alg_summary.csv", "w");
    fprintf(csv_summary, "Seed,Scenario,Selectivity,Variant,ef_result,ef_explore,Recall10,Recall100,p50_us,p95_us,p99_us,QPS,BuildSec,IndexMemMiB,ExpandedNodes,PctTransit,Leaks\n");

    FILE *csv_per_query = fopen("hito6/results/alg_per_query.csv", "w");
    fprintf(csv_per_query, "Seed,Scenario,Selectivity,Variant,ef_result,ef_explore,QueryID,Latency_us,Recall10,Recall100,ReturnedCount,ExpandedNodes,TransitNodes,Leaks\n");

    g_hnsw_use_heaps = 1;
    g_sq8_kernel_mode = 1; /* U1 optimizado */

    /* Evaluación de ALG: H2-AT T1-A (512) y H2-Base */
    const char *variants[2] = {"ALG_H2_AT_T1A", "ALG_H2_Base"};
    uint32_t explore_budgets[2] = {512, 100};

    for (int sc = 0; sc < 2; sc++) {
        const char *scenario_name = (sc == 0) ? "R_Local" : "D_Adversarial";
        const float *q_base = (sc == 0) ? queries_R : queries_D;
        printf("\n==================== Escenario %s ====================\n", scenario_name);

        for (uint32_t seli = 0; seli < NUM_SELECTIVITIES; seli++) {
            uint32_t sel = header.selectivities[seli];
            FilterPredicate pred = filter_create_predicate(sel);

            printf("\n-- Selectividad %2u%% (target_category=%u) --\n", sel, pred.target_category);

            for (int vi = 0; vi < 2; vi++) {
                const char *vname = variants[vi];
                uint32_t ef_exp = explore_budgets[vi];
                int is_at = (vi == 0);

                double *lat = (double *)malloc(nq * sizeof(double));
                double sum_r10 = 0.0, sum_r100 = 0.0;
                uint64_t sum_exp = 0, sum_trans = 0;
                uint32_t total_leaks = 0;

                QueryPerformanceCounter(&t0);
                for (uint32_t q = 0; q < nq; q++) {
                    const float *query = q_base + (size_t)q * dim;

                    LARGE_INTEGER qa, qb;
                    QueryPerformanceCounter(&qa);
                    SearchResult r;
                    if (is_at) {
                        r = hnsw_search_h2_at_sq8(index, sq8, query, &attrs, &pred, 100, 100, ef_exp);
                    } else {
                        r = hnsw_search_h2_sq8_filteraware(index, sq8, query, &attrs, &pred, 100, 100);
                    }
                    QueryPerformanceCounter(&qb);

                    double us = (double)(qb.QuadPart - qa.QuadPart) * 1e6 / (double)fq.QuadPart;
                    lat[q] = us;

                    uint32_t query_leaks = 0;
                    for (uint32_t i = 0; i < r.returned_count; i++) {
                        if (!filter_matches(&attrs, r.ids[i], &pred)) {
                            query_leaks++;
                            total_leaks++;
                        }
                    }

                    double r10 = recall_at_k(&r, &ground_truth[sc][seli][q], 10);
                    double r100 = recall_at_k(&r, &ground_truth[sc][seli][q], 100);
                    sum_r10 += r10;
                    sum_r100 += r100;
                    sum_exp += r.expanded_nodes;
                    sum_trans += r.transit_nodes;

                    fprintf(csv_per_query, "%u,%s,%u,%s,100,%u,%u,%.2f,%.6f,%.6f,%u,%llu,%llu,%u\n",
                            header.seed_index, scenario_name, sel, vname, ef_exp, q, us,
                            r10, r100, r.returned_count, r.expanded_nodes, r.transit_nodes, query_leaks);
                }
                QueryPerformanceCounter(&t1);

                double total_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;
                double qps = total_sec > 0 ? (double)nq / total_sec : 0.0;

                qsort(lat, nq, sizeof(double), cmp_double);
                double p50 = lat[(size_t)(nq * 0.50)];
                double p95 = lat[(size_t)(nq * 0.95)];
                double p99 = lat[(size_t)(nq * 0.99)];
                double avg_r10 = sum_r10 / (double)nq;
                double avg_r100 = sum_r100 / (double)nq;
                double avg_exp = (double)sum_exp / (double)nq;
                double avg_trans = (double)sum_trans / (double)nq;
                double pct_trans = avg_exp > 0 ? (avg_trans / avg_exp) * 100.0 : 0.0;

                printf("%-15s R10: %6.4f | R100: %6.4f | p50: %6.1f us | p95: %6.1f us | QPS: %5.1f | exp: %5.0f | trans%%: %4.1f%% | Leaks: %u\n",
                       vname, avg_r10, avg_r100, p50, p95, qps, avg_exp, pct_trans, total_leaks);

                fprintf(csv_summary, "%u,%s,%u,%s,100,%u,%.6f,%.6f,%.3f,%.3f,%.3f,%.2f,%.3f,%.2f,%.1f,%.1f,%u\n",
                        header.seed_index, scenario_name, sel, vname, ef_exp,
                        avg_r10, avg_r100, p50, p95, p99, qps,
                        total_build_sec, index_mem_mb, avg_exp, pct_trans, total_leaks);
                fflush(csv_summary);
                free(lat);
            }
        }
    }

    fclose(csv_summary);
    fclose(csv_per_query);
    sq8_free(sq8);
    hnsw_free(index);
    free(attrs.category);
    free(vectors);
    free(queries_R);
    free(queries_D);
    for (int sc = 0; sc < 2; sc++) {
        for (uint32_t seli = 0; seli < NUM_SELECTIVITIES; seli++) {
            free(ground_truth[sc][seli]);
        }
    }

    printf("\n========================================================================\n");
    printf("Evaluacion ALG completada exitosamente.\n");
    printf("Resultados guardados en hito6/results/alg_summary.csv y alg_per_query.csv\n");
    return 0;
}
