#include "dataset_format.h"
#include "../../include/alg_hnsw.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

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
    printf("Exportando Dataset Comun Hito 6: Semilla %u (0x%llX)\n", seed_idx, seed_val);
    printf("N=%u | Dim=64 | Consultas=%u | Selectividades: 1%%, 5%%, 10%%, 25%%, 50%%\n", nv, nq);
    printf("========================================================================\n");

    const uint32_t selectivities[NUM_SELECTIVITIES] = {1, 5, 10, 25, 50};
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
        preds[i] = filter_create_predicate(selectivities[i]);
        header.target_categories[i] = preds[i].target_category;
    }

    printf("Generando vectores de base sintéticos...\n");
    float *vectors = vectors_generate_synthetic(nv, VECTOR_DIM, seed_val);
    AttributeSet *attrs = attributes_generate(nv, seed_val);
    if (!vectors || !attrs) { fprintf(stderr, "Error generando vectores/atributos.\n"); return 1; }

    printf("Generando consultas escenario R (local, 85%% reuso)...\n");
    float *queries_R = queries_generate_scenario(nq, VECTOR_DIM, vectors, nv, true, seed_val ^ 0xABCDULL);
    printf("Generando consultas escenario D (adversario, 0%% reuso)...\n");
    float *queries_D = queries_generate_scenario(nq, VECTOR_DIM, vectors, nv, false, seed_val ^ 0xABCDULL);
    if (!queries_R || !queries_D) { fprintf(stderr, "Error generando consultas.\n"); return 2; }

    char out_path[256];
    snprintf(out_path, sizeof(out_path), "hito6/data/dataset_seed%u.bin", seed_idx);
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
    printf("Calculando Ground Truth exacto (fuerza bruta) para los 10 casos...\n");
    for (int sc = 0; sc < 2; sc++) {
        const float *q_ptr = (sc == 0) ? queries_R : queries_D;
        const char *sc_name = (sc == 0) ? "R_Local" : "D_Adversarial";

        for (uint32_t seli = 0; seli < NUM_SELECTIVITIES; seli++) {
            uint32_t sel = selectivities[seli];
            printf("  -> GT %s | Selectividad %2u%%...\n", sc_name, sel);

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
    printf("Exportacion completada exitosamente en %s (tamano aproximado: 34.5 MiB)\n", out_path);

    aligned_free_64(vectors);
    attributes_free(attrs);
    aligned_free_64(queries_R);
    aligned_free_64(queries_D);
    return 0;
}
