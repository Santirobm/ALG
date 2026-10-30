#ifndef ALG_HNSW_HEAP_H
#define ALG_HNSW_HEAP_H

#include "../include/alg_hnsw.h"

/* 0 = referencia qsort; 1 = colas binarias. Se cambia entre benchmarks, no durante una query. */
extern int g_hnsw_use_heaps;

int distpair_compare_asc(const DistPair *a, const DistPair *b);
void hnsw_candidates_push(DistPair *data, uint32_t *size, uint32_t capacity, DistPair value);
DistPair hnsw_candidates_pop(DistPair *data, uint32_t *size);
void hnsw_results_insert(DistPair *data, uint32_t *size, uint32_t capacity, DistPair value);
DistPair hnsw_results_worst(DistPair *data, uint32_t size);
void hnsw_results_sort(DistPair *data, uint32_t size);

#endif
