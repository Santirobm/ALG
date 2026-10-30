#ifndef ALG_HNSW_H
#define ALG_HNSW_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define VECTOR_DIM          64U
#define HNSW_M_DEFAULT      16U
#define HNSW_M0_DEFAULT     32U
#define HNSW_EF_CONSTRUCT   64U
#define HNSW_EF_SEARCH      64U
#define MAX_LAYERS          16U
#define NO_TILE             0xFFFFU
#define EXTERNAL_SLOT       0xFFFFU

/* Tipos de variantes de búsqueda */
typedef enum {
    VARIANT_H0_POSTFILTER = 0,
    VARIANT_H1_PREFILTER  = 1,
    VARIANT_H2_FILTERAWARE = 2,
    VARIANT_H3_TILES_64   = 3,
    VARIANT_H3_TILES_128  = 4,
    VARIANT_H3_TILES_256  = 5,
    NUM_VARIANTS          = 6
} SearchVariant;

/* Estructura para registrar memoria en Windows */
typedef struct {
    size_t working_set_bytes; /* Páginas físicas residentes */
    size_t pagefile_bytes;    /* Memoria comprometida total */
    size_t private_bytes;     /* Memoria privada comprometida */
} MemoryStats;

/* Elemento para colas de prioridad (min-heap / max-heap) */
typedef struct {
    float    dist;
    uint32_t id;
} DistPair;

/* Vista de lectura directa para Pointer Swizzling */
typedef struct {
    const uint32_t *neighbors;
    const float    *vector;
    uint16_t        deg;
    bool            is_swizzled;
} HnswReadView;

/* Nodo del grafo HNSW autoritativo */
typedef struct {
    uint32_t id;
    int32_t  level;
    uint32_t *friends[MAX_LAYERS];
    uint16_t friend_count[MAX_LAYERS];
    uint16_t friend_cap[MAX_LAYERS];
    HnswReadView read_view;
} HnswNode;

/* Índice HNSW principal */
typedef struct {
    uint32_t  max_elements;
    uint32_t  cur_elements;
    uint32_t  dim;
    uint32_t  M;
    uint32_t  M0;
    uint32_t  ef_construction;
    uint32_t  ef_search;
    double    mult_l;
    int32_t   enter_node;
    int32_t   max_level;
    float    *vectors;        /* Buffer contiguo alineado a 64 bytes */
    HnswNode *nodes;          /* Array de nodos */
} HnswIndex;

/* Conjunto de atributos categóricos/numéricos para filtrado */
typedef struct {
    uint32_t *category;
    uint32_t *region;
    float    *price;
    uint32_t num_elements;
} AttributeSet;

/* Predicado de filtro */
typedef struct {
    uint32_t target_category;
    uint32_t min_region;
    uint32_t max_region;
    float    max_price;
    uint32_t selectivity_pct;
} FilterPredicate;

/* Resultado de una consulta */
typedef struct {
    uint32_t k;
    uint32_t returned_count;
    uint32_t ids[128];
    float    dists[128];
    double   latency_us;
    uint64_t distance_comps;
    uint64_t rerank_comps;
    uint64_t visited_nodes;
    uint64_t expanded_nodes;
    uint64_t transit_nodes;
    uint64_t tile_hot_hits;
} SearchResult;

/* Métricas de rendimiento consolidadas */
typedef struct {
    double recall_at_10;
    double recall_at_100;
    double latency_p50_us;
    double latency_p95_us;
    double latency_p99_us;
    double throughput_qps;
    double avg_distance_comps;
    double avg_visited_nodes;
    double tile_hit_rate_pct;
    double total_query_ms;
    double total_maint_ms;
    uint32_t promotions;
    MemoryStats mem_stats;
} BenchmarkMetrics;

/* Funciones de distancia */
float l2_distance_avx2(const float *a, const float *b, uint32_t dim);
float l2_distance_scalar(const float *a, const float *b, uint32_t dim);

/* Gestión de memoria */
void memory_get_stats(MemoryStats *out);
void *aligned_alloc_64(size_t bytes);
void aligned_free_64(void *ptr);

/* Vectores y atributos */
float *vectors_generate_synthetic(uint32_t num_vectors, uint32_t dim, uint64_t seed);
AttributeSet *attributes_generate(uint32_t num_elements, uint64_t seed);
void attributes_free(AttributeSet *attrs);
bool filter_matches(const AttributeSet *attrs, uint32_t id, const FilterPredicate *pred);
FilterPredicate filter_create_predicate(uint32_t selectivity_pct);

/* Ground Truth exacto (Fuerza Bruta) */
SearchResult ground_truth_linear_scan(
    const float *vectors,
    uint32_t num_vectors,
    uint32_t dim,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    const float *query,
    uint32_t top_k
);

/* Motor HNSW */
HnswIndex *hnsw_create(uint32_t max_elements, uint32_t dim, uint32_t M, uint32_t ef_construction);
void hnsw_free(HnswIndex *index);
void hnsw_insert(HnswIndex *index, uint32_t id, const float *vec);

/* Búsquedas filtradas */
SearchResult hnsw_search_h0_postfilter(
    const HnswIndex *index,
    const float *query,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    uint32_t top_k,
    uint32_t ef_search
);

SearchResult hnsw_search_h1_prefilter(
    const HnswIndex *filtered_index,
    const uint32_t *id_mapping,
    const float *query,
    uint32_t top_k,
    uint32_t ef_search
);

SearchResult hnsw_search_h2_filteraware(
    const HnswIndex *index,
    const float *query,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    uint32_t top_k,
    uint32_t ef_search
);

struct Sq8Index;
SearchResult hnsw_search_h2_sq8_filteraware(
    const HnswIndex *index,
    const struct Sq8Index *sq8,
    const float *query,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    uint32_t top_k,
    uint32_t ef_search
);

SearchResult hnsw_search_h2_at_sq8(
    const HnswIndex *index,
    const struct Sq8Index *sq8,
    const float *query,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    uint32_t top_k,
    uint32_t ef_result,
    uint32_t ef_explore
);

/* ============================================================================
 * ESTRUCTURAS DE MICRO-TILES VECTORIALES Y POINTER SWIZZLING (H3)
 * ============================================================================ */
#define MAX_HASH_CAP_HNSW  1024U
#define MAX_HNSW_TILES     256U

typedef struct {
    uint32_t key_plus_one[MAX_HASH_CAP_HNSW];
    uint16_t slot[MAX_HASH_CAP_HNSW];
    uint32_t mask;
    uint32_t shift;
} CompactLocalIndexHnsw;

typedef struct {
    uint32_t logical_id;
    uint16_t deg;
    uint32_t neighbors[HNSW_M0_DEFAULT];
    float    vector[VECTOR_DIM] __attribute__((aligned(32)));
} SwizzledNodeView;

typedef struct {
    uint16_t tile_id;
    uint16_t capacity;       /* 64, 128 o 256 */
    uint16_t count;
    bool     valid;
    uint32_t version;
    SwizzledNodeView *nodes; /* Buffer continuo de nodos y vectores contiguos */
    CompactLocalIndexHnsw local_idx;
} HnswMicroTile;

typedef struct {
    HnswMicroTile tiles[MAX_HNSW_TILES];
    uint16_t *assigned_tile; /* NUM_VERTICES */
    uint16_t *assigned_slot; /* NUM_VERTICES */
    uint8_t  *node_heat;     /* NUM_VERTICES */
    uint32_t  tile_capacity; /* 64, 128, 256 */
    uint32_t  max_tiles;
    uint32_t  active_tiles;
    uint32_t  promotions;
    uint32_t  num_elements;

    /* Telemetría */
    uint64_t total_evals;
    uint64_t hot_tile_hits;
    double   maint_ns_sum;
    double   query_ns_sum;
} HnswTileEngine;

HnswTileEngine *hnsw_tile_engine_create(uint32_t num_elements, uint32_t tile_capacity, uint32_t max_tiles);
void hnsw_tile_engine_free(HnswTileEngine *eng);
void hnsw_tile_engine_reset(HnswTileEngine *eng, HnswIndex *index);

SearchResult hnsw_search_h3_swizzled(
    HnswIndex *index,
    HnswTileEngine *eng,
    const float *query,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    uint32_t top_k,
    uint32_t ef_search,
    bool update_heat_and_promote
);

void benchmark_evaluate_variant_ex(
    SearchVariant variant,
    HnswIndex *index,
    const HnswIndex *prefiltered_index,
    const uint32_t *h1_id_mapping,
    HnswTileEngine *tile_engine,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    const float *query_vectors,
    const SearchResult *ground_truth,
    uint32_t num_queries,
    uint32_t ef_search,
    BenchmarkMetrics *out
);

float *queries_generate_scenario(
    uint32_t num_queries,
    uint32_t dim,
    const float *dataset_vectors,
    uint32_t num_vectors,
    bool is_reuse_scenario,
    uint64_t seed
);

#endif /* ALG_HNSW_H */
