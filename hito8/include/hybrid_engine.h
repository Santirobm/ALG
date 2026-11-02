#ifndef HITO8_HYBRID_ENGINE_H
#define HITO8_HYBRID_ENGINE_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { POLICY_EXACT=0, POLICY_BALANCED=1, POLICY_APPROXIMATE=2 } DispatcherPolicy;
typedef enum { ENGINE_EXACT_SCAN=0, ENGINE_HNSWLIB=1, ENGINE_ALG_H2_AT=2, ENGINE_FAISS=3 } HybridEngineId;
typedef enum { QUERY_LOCAL=0, QUERY_ADVERSARIAL=1, QUERY_UNKNOWN=2 } QueryRegime;
typedef struct { DispatcherPolicy policy; double target_recall; double max_latency_budget_us; uint32_t k; } QualityRequirements;
typedef struct { uint32_t eligible_count; uint32_t dataset_size; uint32_t dimension; QueryRegime regime; double selectivity_pct; } QueryFeatures;
typedef struct { uint32_t id; float distance; } HybridNeighbor;
typedef struct { HybridNeighbor *neighbors; uint32_t capacity; uint32_t count; } HybridSearchOutput;
typedef struct { HybridEngineId chosen_engine; double estimated_cost_us, estimated_recall, recall_lower_confidence_bound, actual_latency_us, regret_us; bool fallback_triggered, quality_supported; } HybridQueryResult;
typedef int (*HybridSearchFn)(void*, const float*, const void*, uint32_t, uint32_t, HybridSearchOutput*);
typedef struct { void *context; HybridSearchFn search; uint32_t default_budget; bool available; } HybridEngineBinding;
typedef struct { HybridEngineBinding exact_scan, hnswlib, alg_h2_at; } HybridEngineBindings;
struct ExactCostModel;
struct HnswCostModel;
int hybrid_search(const HybridEngineBindings*, const QueryFeatures*, const QualityRequirements*, const struct ExactCostModel*, const struct HnswCostModel*, const float*, const void*, HybridSearchOutput*, HybridQueryResult*);
#ifdef __cplusplus
}
#endif
#endif
