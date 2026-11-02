#ifndef DATASET_FORMAT_H
#define DATASET_FORMAT_H

#include <stdint.h>

#define DATASET_MAGIC 0x414C4736U /* 'ALG6' */
#define DATASET_VERSION 2U /* Hito 7: selectivities stored in deci-percent */
#define NUM_SELECTIVITIES 7U
#define TOP_K_GT 100U

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t seed_index;
    uint64_t seed_value;
    uint32_t num_vectors;
    uint32_t dim;
    uint32_t num_queries;
    uint32_t num_selectivities;
    uint32_t selectivities[NUM_SELECTIVITIES]; /* deci-percent: 100 = 10.0% */
    uint32_t target_categories[NUM_SELECTIVITIES];
} DatasetHeader;
#pragma pack(pop)

#endif
