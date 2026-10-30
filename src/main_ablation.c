#include "../include/alg_hnsw.h"
#include "../include/heap.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    SearchVariant variant;
    int use_heap;
} AblationCase;

static const AblationCase CASES[4] = {
    { "A_H2_qsort", VARIANT_H2_FILTERAWARE, 0 },
    { "B_H2_heap",  VARIANT_H2_FILTERAWARE, 1 },
    { "C_H3_128_qsort", VARIANT_H3_TILES_128, 0 },
    { "D_H3_128_heap",  VARIANT_H3_TILES_128, 1 }
};

static void banner(void) {
    puts("==================================================================================================");
    puts(" ALG-HNSW | ABLACION 2x2: ORDENAMIENTO qsort vs HEAPS x REPRESENTACION H2 vs H3-128");
    puts("==================================================================================================");
}


static int same_search_result(const SearchResult *a, const SearchResult *b) {
    if (a->returned_count != b->returned_count) return 0;
    for (uint32_t i=0;i<a->returned_count;i++) {
        if (a->ids[i] != b->ids[i]) return 0;
        if (memcmp(&a->dists[i],&b->dists[i],sizeof(float)) != 0) return 0;
    }
    return 1;
}

/* Valida la semántica de las colas sobre una muestra determinista de consultas. */
static int validate_exact_queue_parity(
    HnswIndex *index, HnswTileEngine *eng, const AttributeSet *attrs,
    const FilterPredicate *pred, const float *queries, uint32_t num_queries,
    uint32_t ef_search
) {
    const uint32_t n = num_queries < 100U ? num_queries : 100U;
    const uint32_t ef = ef_search < 100U ? 100U : ef_search;
    SearchResult ref, got;

    g_hnsw_use_heaps=0;
    for(uint32_t q=0;q<n;q++) {
        ref=hnsw_search_h2_filteraware(index,&queries[q*VECTOR_DIM],attrs,pred,100U,ef);
        g_hnsw_use_heaps=1;
        got=hnsw_search_h2_filteraware(index,&queries[q*VECTOR_DIM],attrs,pred,100U,ef);
        if(!same_search_result(&ref,&got)) { g_hnsw_use_heaps=0; return 0; }
        g_hnsw_use_heaps=0;
    }

    hnsw_tile_engine_reset(eng,index);
    for(uint32_t q=0;q<n;q++) {
        ref=hnsw_search_h3_swizzled(index,eng,&queries[q*VECTOR_DIM],attrs,pred,100U,ef,false);
        g_hnsw_use_heaps=1;
        got=hnsw_search_h3_swizzled(index,eng,&queries[q*VECTOR_DIM],attrs,pred,100U,ef,false);
        if(!same_search_result(&ref,&got)) { g_hnsw_use_heaps=0; return 0; }
        g_hnsw_use_heaps=0;
    }
    hnsw_tile_engine_reset(eng,index);
    return 1;
}

int main(int argc, char **argv) {
    SetConsoleOutputCP(65001);
    setvbuf(stdout, NULL, _IONBF, 0);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);

    uint32_t num_vectors=100000U, num_queries=1000U, ef_search=100U, selectivity=5U;
    for (int i=1;i<argc;i++) {
        if (!strcmp(argv[i],"--num-vectors") && i+1<argc) num_vectors=(uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i],"--num-queries") && i+1<argc) num_queries=(uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i],"--selectivity") && i+1<argc) selectivity=(uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i],"--ef-search") && i+1<argc) ef_search=(uint32_t)atoi(argv[++i]);
    }
    if (num_vectors==0 || num_queries==0 || ef_search==0 || ef_search>512U || selectivity>100U) {
        fprintf(stderr,"Argumentos fuera de rango (efSearch debe estar entre 1 y 512).\n"); return 2;
    }

    banner();
    printf("Vectores=%u | consultas/escenario=%u | selectividad=%u%% | efSearch=%u | semillas=5\n",
           num_vectors,num_queries,selectivity,ef_search);

    const uint64_t SEEDS[5]={42ULL,1337ULL,987654321ULL,0xCAFEBABEULL,0xDEADBEEFULL};
    CreateDirectoryA("results",NULL);
    FILE *csv=fopen("results/main_ablation.csv","w");
    if (csv) fprintf(csv,"Seed,Scenario,Variant,Queue,Representation,Recall@10,Recall@100,Lat_p50_us,Lat_p95_us,Lat_p99_us,QPS,DistComps,Visited,TileHitRate,QueryTime_ms,MaintTime_ms,TotalTime_ms,WorkingSet_MB,Private_MB\n");

    FilterPredicate pred=filter_create_predicate(selectivity);
    for (uint32_t si=0;si<5;si++) {
        const uint64_t seed=SEEDS[si];
        printf("\n--- Seed %u/5: 0x%llX ---\n",si+1,(unsigned long long)seed);
        float *vectors=vectors_generate_synthetic(num_vectors,VECTOR_DIM,seed);
        AttributeSet *attrs=attributes_generate(num_vectors,seed);
        HnswIndex *index=hnsw_create(num_vectors,VECTOR_DIM,HNSW_M_DEFAULT,HNSW_EF_CONSTRUCT);
        if (!vectors || !attrs || !index) { fprintf(stderr,"Fallo de asignacion.\n"); return 3; }

        LARGE_INTEGER t0,t1,freq;
        QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&t0);
        for(uint32_t i=0;i<num_vectors;i++) hnsw_insert(index,i,&vectors[i*VECTOR_DIM]);
        QueryPerformanceCounter(&t1);
        printf("Indice construido: %.2f s\n",(double)(t1.QuadPart-t0.QuadPart)/(double)freq.QuadPart);

        HnswTileEngine *eng=hnsw_tile_engine_create(num_vectors,128U,128U);
        if(!eng){fprintf(stderr,"No se pudo crear H3-128.\n");return 4;}

        for(int sc=0;sc<2;sc++) {
            const bool reuse=(sc==0);
            const char *sc_name=reuse?"R_Local":"D_Adversarial";
            float *queries=queries_generate_scenario(num_queries,VECTOR_DIM,vectors,num_vectors,reuse,seed^0xABCDULL);
            SearchResult *gt=(SearchResult*)malloc(num_queries*sizeof(SearchResult));
            if(!queries||!gt){fprintf(stderr,"Fallo de memoria en consultas.\n");return 5;}
            for(uint32_t q=0;q<num_queries;q++)
                gt[q]=ground_truth_linear_scan(vectors,num_vectors,VECTOR_DIM,attrs,&pred,&queries[q*VECTOR_DIM],100U);

            if (!validate_exact_queue_parity(index,eng,attrs,&pred,queries,num_queries,ef_search)) {
                fprintf(stderr,"ERROR: qsort y heap no producen resultados bitwise identicos en %s, seed %u.\n",sc_name,si+1);
                return 6;
            }
            printf("Validacion exacta qsort/heap: PASS (%u consultas H2 + %u H3; IDs y distancias bitwise).\\n",
                   num_queries<100U?num_queries:100U,num_queries<100U?num_queries:100U);

            /* Rotacion de orden para repartir deriva termica y de frecuencia entre variantes. */
            int order[4];
            for(int j=0;j<4;j++) order[j]=(j+(int)((si+(uint32_t)sc)%4U))%4;
            printf("\nEscenario %s | orden rotado:",sc_name);
            for(int j=0;j<4;j++) printf(" %s",CASES[order[j]].name);
            puts("");

            for(int oi=0;oi<4;oi++) {
                const AblationCase *c=&CASES[order[oi]];
                g_hnsw_use_heaps=c->use_heap;
                BenchmarkMetrics m;
                benchmark_evaluate_variant_ex(c->variant,index,NULL,NULL,
                    c->variant==VARIANT_H3_TILES_128?eng:NULL,
                    attrs,&pred,queries,gt,num_queries,ef_search,&m);

                printf("%-16s | R10 %.4f R100 %.4f | p50 %.1f us p95 %.1f us p99 %.1f us | %.1f QPS | comps %.1f | hit %.1f%% | maint %.2f ms\n",
                    c->name,m.recall_at_10,m.recall_at_100,m.latency_p50_us,m.latency_p95_us,
                    m.latency_p99_us,m.throughput_qps,m.avg_distance_comps,m.tile_hit_rate_pct,m.total_maint_ms);

                if(csv) {
                    double total=m.total_query_ms+m.total_maint_ms;
                    fprintf(csv,"%u,%s,%s,%s,%s,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                        si+1,sc_name,c->name,c->use_heap?"heap":"qsort",
                        c->variant==VARIANT_H2_FILTERAWARE?"H2":"H3-128",
                        m.recall_at_10,m.recall_at_100,m.latency_p50_us,m.latency_p95_us,m.latency_p99_us,
                        m.throughput_qps,m.avg_distance_comps,m.avg_visited_nodes,m.tile_hit_rate_pct,
                        m.total_query_ms,m.total_maint_ms,total,
                        (double)m.mem_stats.working_set_bytes/1048576.0,
                        (double)m.mem_stats.private_bytes/1048576.0);
                }
            }
            aligned_free_64(queries); free(gt);
        }
        hnsw_tile_engine_free(eng); hnsw_free(index); attributes_free(attrs); aligned_free_64(vectors);
    }
    g_hnsw_use_heaps=0;
    if(csv){fclose(csv);puts("\nCSV: results/main_ablation.csv");}
    return 0;
}
