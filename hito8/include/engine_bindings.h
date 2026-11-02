#ifndef HITO8_ENGINE_BINDINGS_H
#define HITO8_ENGINE_BINDINGS_H

#include "hybrid_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Estructura para pasar parámetros de filtrado a los motores */
typedef struct {
    uint32_t target_category;
    const uint32_t *eligible_ids; /* Lista opcional preindexada de IDs elegibles */
    uint32_t eligible_count;      /* Cantidad de IDs elegibles */
} FilterQueryContext;

/* Contexto para ExactFilteredScan */
typedef struct {
    const float *vectors;
    const uint32_t *categories;
    uint32_t num_vectors;
    uint32_t dim;
} ExactEngineContext;

/* Contexto para hnswlib */
typedef struct {
    void *hnsw_index; /* Puntero opaco a hnswlib::HierarchicalNSW<float> */
    const uint32_t *categories;
    uint32_t num_vectors;
    uint32_t dim;
} HnswlibEngineContext;

/* Contexto para sonda de régimen de consulta (Feature Extractor) */
typedef struct {
    const float *probe_centroids;
    uint32_t num_probes;
    uint32_t dim;
    float distance_threshold_sq;
} QueryFeatureProbe;

/* Funciones de búsqueda desacopladas compatibles con HybridSearchFn */
int exact_scan_binding_search(void *ctx, const float *q, const void *pred, uint32_t k, uint32_t budget, HybridSearchOutput *out);
int hnswlib_binding_search(void *ctx, const float *q, const void *pred, uint32_t k, uint32_t budget, HybridSearchOutput *out);

/* Extractor observable de régimen de consulta (sin etiquetas ground truth) */
QueryRegime extract_query_regime(const QueryFeatureProbe *probe, const float *query);

#ifdef __cplusplus
}
#endif

#endif /* HITO8_ENGINE_BINDINGS_H */
