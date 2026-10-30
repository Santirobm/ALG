#include "../include/alg_hnsw.h"
#include <stdlib.h>
#include <string.h>

static inline uint64_t xorshift64s_flt(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

AttributeSet *attributes_generate(uint32_t num_elements, uint64_t seed) {
    AttributeSet *attrs = (AttributeSet *)malloc(sizeof(AttributeSet));
    if (!attrs) return NULL;

    attrs->category = (uint32_t *)malloc(num_elements * sizeof(uint32_t));
    attrs->region   = (uint32_t *)malloc(num_elements * sizeof(uint32_t));
    attrs->price    = (float *)malloc(num_elements * sizeof(float));
    attrs->num_elements = num_elements;

    uint64_t rng = seed ^ 0xFEEDFACECAFEF00DULL;

    for (uint32_t i = 0; i < num_elements; i++) {
        uint64_t r = xorshift64s_flt(&rng);
        attrs->category[i] = (uint32_t)(r % 100U);          /* [0..99] */
        attrs->region[i]   = (uint32_t)((r >> 16) % 16U);   /* [0..15] */
        attrs->price[i]    = (float)((r >> 32) & 0xFFFF) / 65.535f; /* [0.0 .. 1000.0] */
    }

    return attrs;
}

void attributes_free(AttributeSet *attrs) {
    if (!attrs) return;
    if (attrs->category) free(attrs->category);
    if (attrs->region) free(attrs->region);
    if (attrs->price) free(attrs->price);
    free(attrs);
}

/*
 * Genera un predicado calibrado para una selectividad teórica exacta:
 * 1%, 5%, 10%, 25%, 50%, 100%
 */
FilterPredicate filter_create_predicate(uint32_t selectivity_pct) {
    FilterPredicate p;
    memset(&p, 0, sizeof(FilterPredicate));
    p.selectivity_pct = selectivity_pct;

    switch (selectivity_pct) {
        case 1:
            /* 1 categoría exacta de 100 -> ~1% */
            p.target_category = 42U;
            p.min_region = 0U;
            p.max_region = 15U;
            p.max_price = 1000.0f;
            break;
        case 5:
            /* 5 categorías -> ~5% (0..4) */
            p.target_category = 5U; /* < 5 */
            p.min_region = 0U;
            p.max_region = 15U;
            p.max_price = 1000.0f;
            break;
        case 10:
            /* 10 categorías -> ~10% (0..9) */
            p.target_category = 10U; /* < 10 */
            p.min_region = 0U;
            p.max_region = 15U;
            p.max_price = 1000.0f;
            break;
        case 25:
            /* 25 categorías -> ~25% (0..24) */
            p.target_category = 25U; /* < 25 */
            p.min_region = 0U;
            p.max_region = 15U;
            p.max_price = 1000.0f;
            break;
        case 50:
            /* 50 categorías -> ~50% (0..49) */
            p.target_category = 50U; /* < 50 */
            p.min_region = 0U;
            p.max_region = 15U;
            p.max_price = 1000.0f;
            break;
        case 100:
        default:
            /* 100% elegible */
            p.target_category = 100U;
            p.min_region = 0U;
            p.max_region = 15U;
            p.max_price = 1000.0f;
            break;
    }

    return p;
}

bool filter_matches(const AttributeSet *attrs, uint32_t id, const FilterPredicate *pred) {
    if (!attrs || !pred) return true;
    if (pred->selectivity_pct == 100U) return true;
    if (id >= attrs->num_elements) return false;

    if (pred->selectivity_pct == 1U) {
        return (attrs->category[id] == pred->target_category);
    }

    return (attrs->category[id] < pred->target_category);
}
