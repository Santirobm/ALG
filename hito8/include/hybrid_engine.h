#ifndef HITO8_HYBRID_ENGINE_H
#define HITO8_HYBRID_ENGINE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    POLICY_EXACT = 0,
    POLICY_BALANCED = 1,
    POLICY_APPROXIMATE = 2
} DispatcherPolicy;

typedef enum {
    ENGINE_EXACT_SCAN = 0,
    ENGINE_HNSWLIB = 1,
    ENGINE_ALG_H2_AT = 2,
    ENGINE_FAISS = 3
} HybridEngineId;

typedef enum {
    QUERY_LOCAL = 0,
    QUERY_ADVERSARIAL = 1,
    QUERY_UNKNOWN = 2
} QueryRegime;

typedef struct {
    DispatcherPolicy policy;
    double target_recall;          /* e.g. 0.95 */
    double max_latency_budget_us;  /* <= 0 means no budget */
    uint32_t k;
} QualityRequirements;

typedef struct {
    uint32_t eligible_count;       /* Must be supplied cheaply by caller/index */
    uint32_t dataset_size;
    uint32_t dimension;
    QueryRegime regime;
    double selectivity_pct;        /* Optional if eligible_count is known */
} QueryFeatures;

typedef struct {
    uint32_t id;
    float distance;
} HybridNeighbor;

typedef struct {
    HybridNeighbor *neighbors;     /* Caller-owned output buffer */
    uint32_t capacity;
    uint32_t count;
} HybridSearchOutput;

typedef struct {
    HybridEngineId chosen_engine;
    double estimated_cost_us;
    double estimated_recall;
    double recall_lower_confidence_bound;
    double actual_latency_us;
    double regret_us;
    bool fallback_triggered;
    bool quality_supported;
} HybridQueryResult;

/* Engine callback. Implementations must return only predicate-eligible IDs. */
typedef int (*HybridSearchFn)(void *context,
                              const float *query,
                              const void *predicate,
                              uint32_t k,
                              uint32_t search_budget,
                              HybridSearchOutput *output);

typedef struct {
    void *context;
    HybridSearchFn search;
    uint32_t default_budget;
    bool available;
} HybridEngineBinding;

typedef struct {
    HybridEngineBinding exact_scan;
    HybridEngineBinding hnswlib;
    HybridEngineBinding alg_h2_at;
} HybridEngineBindings;

/* Selects an engine and executes it. Timing is measured by the caller/benchmark
   layer so dispatcher overhead can be measured separately and reproducibly. */
int hybrid_search(const HybridEngineBindings *bindings,
                  const QueryFeatures *features,
                  const QualityRequirements *requirements,
                  const float *query,
                  const void *predicate,
                  HybridSearchOutput *output,
                  HybridQueryResult *metadata);

#ifdef __cplusplus
}
#endif
#endif
