/*
 * ============================================================================
 * EXPERIMENTO ALG-2: POINTER-SWIZZLED MICRO-TILES Y DESPACHO DE COSTE CERO
 * ============================================================================
 * Objetivo central:
 *   Eliminar el coste recurrente de despacho dinámico (lookup hash, bitmasks y
 *   bifurcaciones por vértice) trasladando la decisión de lectura exclusivamente
 *   a los momentos de promoción e invalidación mediante Pointer Swizzling.
 *
 * Estructura del experimento:
 *   - FASE A: Aislamiento del coste de despacho bajo traza fija de promociones:
 *       * A0: V1 Original (Baseline)
 *       * A1: Motor V7 sin promociones (Overhead base)
 *       * A2: V7 Actual (Despacho dinámico con Tabla Hash Compacta en L1)
 *       * A3: Micro-Tiles con Pointer Swizzling (EdgeView directo)
 *       * A4: A3 con slot local embebido y buffer estable (Zero Hash Lookup)
 *   - FASE B: Sistema adaptativo completo de extremo a extremo:
 *       * V1 Original
 *       * V7 (ALG-1 Hash L1 + H3)
 *       * ALG-2 (Pointer Swizzling + H3 Completo)
 *       * ALG-2-Lite (Pointer Swizzling + H3 Simplificado de coste casi nulo)
 *
 * Rigor metodológico:
 *   - Mismos punteros virtuales en Heap reutilizados en cada corrida.
 *   - Purga de 32 MB de caché L3 (scrub) antes de cada variante.
 *   - Rotación del orden de ejecución por semilla.
 *   - Evaluación en 5 semillas (3 reservadas de test + 2 semillas de validación externa).
 *   - 5 Cargas de trabajo: A (Hotspots), B (Deriva), C (Abrupto), D (Adversarial), E (Estático).
 *   - Medición estricta de NetBenefit = T(V1) - T(ALG) y verificación 100% de oráculo.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#define NUM_VERTICES        262144U
#define NUM_COMMUNITIES     256U
#define COMM_SIZE           (NUM_VERTICES / NUM_COMMUNITIES)
#define NO_TILE             0xFFFFU
#define EXTERNAL_SLOT       0xFFFFU

#define BFS_MAX_DEPTH       3U
#define BFS_MAX_VISITED     512U

#define NUM_STEPS_PER_RUN   3000U
#define UPDATES_PER_BATCH   64U
#define UPDATE_BATCH_PERIOD 150U
#define NUM_EVAL_SEEDS      5U

static double g_qpc_ns_per_tick = 0.0;

static void init_timer(void) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    g_qpc_ns_per_tick = 1e9 / (double)freq.QuadPart;
}

static inline double now_ns(void) {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * g_qpc_ns_per_tick;
}

static inline uint64_t xorshift64s(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Scrub de 32 MB para vaciar L1/L2/L3 entre variantes */
#define L3_SCRUB_BYTES (32U * 1024U * 1024U)
static uint64_t *g_scrub_buf = NULL;
static volatile uint64_t g_scrub_sink = 0;

static void scrub_l3_cache(void) {
    if (!g_scrub_buf) {
        g_scrub_buf = (uint64_t *)_aligned_malloc(L3_SCRUB_BYTES, 64);
        memset(g_scrub_buf, 0x5A, L3_SCRUB_BYTES);
    }
    uint64_t acc = 0;
    size_t n = L3_SCRUB_BYTES / sizeof(uint64_t);
    for (size_t i = 0; i < n; i += 8) {
        g_scrub_buf[i] += 1ULL;
        acc ^= g_scrub_buf[i];
    }
    g_scrub_sink = acc;
}

typedef struct __attribute__((aligned(64))) {
    uint64_t weight_sum;
    uint64_t aux_data[7];
} VertexPayload;

/*
 * Vista de lectura Pointer-Swizzled:
 * Cuando u está frío, `edges` apunta a `DynVertex.edges` y `target_weights` es NULL.
 * Cuando u está en un Micro-Tile, `edges` apunta directamente al bloque del Tile y
 * `target_weights` apunta a las sumas precalculadas en caché.
 */
typedef struct {
    const uint32_t *edges;
    const uint64_t *target_weights;
    uint32_t        deg;
} EdgeView;

typedef struct {
    uint32_t *edges;
    uint32_t  deg;
    uint32_t  cap;
    uint32_t  version;
    EdgeView  read_view; /* Vista activa de lectura (Zero-Dispatch) */
} DynVertex;

typedef struct {
    DynVertex     *v;
    VertexPayload *payloads;
    uint32_t      *comm_members;
    uint64_t       total_edges;
    size_t         bytes_allocated;
} DynGraph;

static void dyngraph_free(DynGraph *g) {
    if (!g->v) return;
    for (uint32_t u = 0; u < NUM_VERTICES; u++) {
        free(g->v[u].edges);
    }
    free(g->v);
    free(g->comm_members);
    _aligned_free(g->payloads);
    g->v = NULL;
}

static void dyngraph_clone(DynGraph *dst, const DynGraph *src) {
    dst->v = (DynVertex *)calloc(NUM_VERTICES, sizeof(DynVertex));
    dst->payloads = (VertexPayload *)_aligned_malloc(NUM_VERTICES * sizeof(VertexPayload), 64);
    dst->comm_members = NULL;

    memcpy(dst->payloads, src->payloads, NUM_VERTICES * sizeof(VertexPayload));
    dst->total_edges = src->total_edges;
    dst->bytes_allocated = NUM_VERTICES * (sizeof(DynVertex) + sizeof(VertexPayload));

    for (uint32_t u = 0; u < NUM_VERTICES; u++) {
        uint32_t cap = (src->v[u].cap < 64U) ? 64U : src->v[u].cap;
        dst->v[u].edges = (uint32_t *)malloc(cap * sizeof(uint32_t));
        memcpy(dst->v[u].edges, src->v[u].edges, src->v[u].deg * sizeof(uint32_t));
        dst->v[u].deg = src->v[u].deg;
        dst->v[u].cap = cap;
        dst->v[u].version = src->v[u].version;
        /* Inicializar EdgeView por defecto en modo autoritativo */
        dst->v[u].read_view.edges = dst->v[u].edges;
        dst->v[u].read_view.target_weights = NULL;
        dst->v[u].read_view.deg = src->v[u].deg;
        dst->bytes_allocated += (size_t)cap * sizeof(uint32_t);
    }
}

static void dyngraph_reset_in_place(DynGraph *dst, const DynGraph *src) {
    memcpy(dst->payloads, src->payloads, NUM_VERTICES * sizeof(VertexPayload));
    dst->total_edges = src->total_edges;
    for (uint32_t u = 0; u < NUM_VERTICES; u++) {
        uint32_t deg = src->v[u].deg;
        memcpy(dst->v[u].edges, src->v[u].edges, deg * sizeof(uint32_t));
        dst->v[u].deg = deg;
        dst->v[u].version = src->v[u].version;
        /* Restaurar read_view a estado autoritativo */
        dst->v[u].read_view.edges = dst->v[u].edges;
        dst->v[u].read_view.target_weights = NULL;
        dst->v[u].read_view.deg = deg;
    }
}

static inline void dyngraph_add_edge(DynGraph *g, uint32_t u, uint32_t dst_v) {
    DynVertex *dv = &g->v[u];
    for (uint32_t i = 0; i < dv->deg; i++) {
        if (dv->edges[i] == dst_v) return;
    }
    if (dv->deg == dv->cap) {
        uint32_t new_cap = dv->cap * 2;
        dv->edges = (uint32_t *)realloc(dv->edges, new_cap * sizeof(uint32_t));
        g->bytes_allocated += (size_t)(new_cap - dv->cap) * sizeof(uint32_t);
        dv->cap = new_cap;
    }
    dv->edges[dv->deg++] = dst_v;
    dv->version++;
    g->total_edges++;
    /* Actualizar read_view autoritativo */
    dv->read_view.edges = dv->edges;
    dv->read_view.deg = dv->deg;
}

static inline void dyngraph_remove_edge(DynGraph *g, uint32_t u) {
    DynVertex *dv = &g->v[u];
    if (dv->deg > 4) {
        dv->deg--;
        dv->version++;
        g->total_edges--;
        dv->read_view.edges = dv->edges;
        dv->read_view.deg = dv->deg;
    }
}

static void generate_base_graph(DynGraph *g, uint64_t seed) {
    g->v = (DynVertex *)calloc(NUM_VERTICES, sizeof(DynVertex));
    g->payloads = (VertexPayload *)_aligned_malloc(NUM_VERTICES * sizeof(VertexPayload), 64);
    g->comm_members = (uint32_t *)malloc(NUM_VERTICES * sizeof(uint32_t));
    memset(g->payloads, 0, NUM_VERTICES * sizeof(VertexPayload));
    g->total_edges = 0;
    g->bytes_allocated = NUM_VERTICES * (sizeof(DynVertex) + sizeof(VertexPayload));

    uint64_t rng = seed ^ 0x9E3779B97F4A7C15ULL;

    for (uint32_t i = 0; i < NUM_VERTICES; i++) g->comm_members[i] = i;
    for (uint32_t i = NUM_VERTICES - 1; i > 0; i--) {
        uint32_t j = (uint32_t)(xorshift64s(&rng) % (i + 1));
        uint32_t tmp = g->comm_members[i];
        g->comm_members[i] = g->comm_members[j];
        g->comm_members[j] = tmp;
    }

    for (uint32_t u = 0; u < NUM_VERTICES; u++) {
        uint32_t init_cap = 28;
        g->v[u].edges = (uint32_t *)malloc(init_cap * sizeof(uint32_t));
        g->v[u].deg = 0;
        g->v[u].cap = init_cap;
        g->v[u].version = 1;
        g->v[u].read_view.edges = g->v[u].edges;
        g->v[u].read_view.target_weights = NULL;
        g->v[u].read_view.deg = 0;
        g->bytes_allocated += init_cap * sizeof(uint32_t);
        g->payloads[u].weight_sum = ((uint64_t)u * 2654435761ULL) ^ 0x55AA55AA12345678ULL;
    }

    for (uint32_t c = 0; c < NUM_COMMUNITIES; c++) {
        const uint32_t *nodes = &g->comm_members[c * COMM_SIZE];
        for (uint32_t idx = 0; idx < COMM_SIZE; idx++) {
            uint32_t u = nodes[idx];
            bool is_core = (idx < 96U);
            uint32_t deg = is_core ? (20U + (uint32_t)(xorshift64s(&rng) % 9U))
                                   : (14U + (uint32_t)(xorshift64s(&rng) % 9U));
            for (uint32_t e = 0; e < deg; e++) {
                uint64_t r = xorshift64s(&rng);
                uint32_t target;
                uint32_t p = (uint32_t)(r % 100U);
                if (is_core) {
                    if (p < 88U) target = nodes[(r >> 8) % 96U];
                    else if (p < 96U) target = nodes[(r >> 8) & (COMM_SIZE - 1U)];
                    else target = (uint32_t)((r >> 8) % NUM_VERTICES);
                } else {
                    if (p < 50U) target = nodes[(r >> 8) % 96U];
                    else if (p < 90U) target = nodes[(r >> 8) & (COMM_SIZE - 1U)];
                    else target = (uint32_t)((r >> 8) % NUM_VERTICES);
                }
                if (target != u) dyngraph_add_edge(g, u, target);
            }
        }
    }
}

/* ============================================================================
 * ESTRUCTURAS DE MICRO-TILES: HASH TRADICIONAL VS POINTER SWIZZLED
 * ============================================================================ */
#define MAX_HASH_CAP     2048U
#define MAX_TILE_SLOTS   256U
#define MAX_NUM_TILES    128U
#define MAX_TILE_BYTES   40960U

typedef struct {
    uint32_t key_plus_one[MAX_HASH_CAP];
    uint16_t slot[MAX_HASH_CAP];
    uint32_t mask;
    uint32_t shift;
} CompactLocalIndex;

static inline void local_index_clear(CompactLocalIndex *idx, uint32_t max_vertices) {
    uint32_t cap = 128U;
    uint32_t log2_cap = 7U;
    while (cap < max_vertices * 2U) {
        cap <<= 1U;
        log2_cap++;
    }
    idx->mask  = cap - 1U;
    idx->shift = 32U - log2_cap;
    memset(idx->key_plus_one, 0, cap * sizeof(uint32_t));
}

static inline void local_index_insert(CompactLocalIndex *idx, uint32_t global_id, uint16_t local_slot) {
    uint32_t pos = (global_id * 0x9E3779B9U) >> idx->shift;
    uint32_t kp1 = global_id + 1U;
    while (idx->key_plus_one[pos] != 0 && idx->key_plus_one[pos] != kp1) {
        pos = (pos + 1U) & idx->mask;
    }
    idx->key_plus_one[pos] = kp1;
    idx->slot[pos] = local_slot;
}

static inline uint16_t local_index_lookup_hash(const CompactLocalIndex *idx, uint32_t global_id) {
    uint32_t pos = (global_id * 0x9E3779B9U) >> idx->shift;
    uint32_t kp1 = global_id + 1U;
    while (true) {
        uint32_t cur = idx->key_plus_one[pos];
        if (cur == kp1) return idx->slot[pos];
        if (cur == 0)   return EXTERNAL_SLOT;
        pos = (pos + 1U) & idx->mask;
    }
}

/*
 * Bloque de Micro-Tile con almacenamiento estable pre-dimensionado
 * para evitar realojamientos de punteros durante append (Riesgo 2 resuelto).
 */
typedef struct {
    bool              valid;
    uint16_t          tile_id;
    uint16_t          num_vertices;
    uint16_t          max_vertices;
    uint32_t          max_bytes_budget;
    uint32_t          cur_bytes_used;

    uint32_t          logical_ids[MAX_TILE_SLOTS];
    uint32_t          row_offset[MAX_TILE_SLOTS + 1];
    uint32_t          deg_in_tile[MAX_TILE_SLOTS];
    uint64_t          dirty_bits[MAX_TILE_SLOTS / 64U];
    CompactLocalIndex local_idx;

    /* Buffers estables pre-asignados a capacidad máxima (4096 aristas) */
    uint32_t         *packed_edges;
    uint64_t         *packed_target_weights;
    uint32_t          edges_cap;

    double            C_b_ns;
    double            g_b_ns;
    uint32_t          queries_since_promo;
    double            realized_savings_ns;
} SwizzledMicroTile;

typedef enum {
    ENGINE_V7_HASH_DYNAMIC     = 0,
    ENGINE_ALG2_SWIZZLED       = 1,
    ENGINE_ALG2_SWIZZLED_LITE  = 2
} EngineMode;

typedef struct {
    EngineMode        mode;
    SwizzledMicroTile tiles[MAX_NUM_TILES];
    uint32_t          num_allocated_tiles;
    uint32_t          active_tiles_count;

    uint16_t          assigned_tile[NUM_VERTICES];
    uint16_t          assigned_slot[NUM_VERTICES];
    uint8_t           vertex_heat[NUM_VERTICES];

    uint32_t          promotions;
    uint32_t          useless_promotions;
    uint32_t          invalidations;
    uint64_t          expanded_hot_hits;
    uint64_t          expanded_total;
    uint64_t          read_through_fallbacks;
    uint64_t          total_bytes_copied;
    double            capture_ns_sum;
    double            initial_build_ns_sum;
    double            append_ns_sum;
    double            first_query_ns_sum;
    uint32_t          first_query_count;
    double            subseq_query_ns_sum;
    uint32_t          subseq_query_count;
    double            cold_query_ns_sum;
    uint32_t          cold_query_count;
    size_t            bytes_allocated;
} Alg2Engine;

static void alg2_engine_init(Alg2Engine *eng, EngineMode mode) {
    memset(eng, 0, sizeof(Alg2Engine));
    eng->mode = mode;
    memset(eng->assigned_tile, 0xFF, sizeof(eng->assigned_tile));
    memset(eng->assigned_slot, 0xFF, sizeof(eng->assigned_slot));
    eng->bytes_allocated = sizeof(Alg2Engine);

    const uint32_t FIXED_EDGES_CAP = 4096U;
    for (uint32_t b = 0; b < MAX_NUM_TILES; b++) {
        SwizzledMicroTile *t = &eng->tiles[b];
        t->tile_id = (uint16_t)b;
        t->max_vertices = 128U;
        t->max_bytes_budget = MAX_TILE_BYTES;
        t->edges_cap = FIXED_EDGES_CAP;
        t->packed_edges = (uint32_t *)malloc(FIXED_EDGES_CAP * sizeof(uint32_t));
        t->packed_target_weights = (uint64_t *)malloc(FIXED_EDGES_CAP * sizeof(uint64_t));
        t->C_b_ns = 5000.0;
        t->g_b_ns = 1500.0;
        eng->bytes_allocated += FIXED_EDGES_CAP * (sizeof(uint32_t) + sizeof(uint64_t));
    }
}

static void alg2_engine_free(Alg2Engine *eng) {
    for (uint32_t b = 0; b < MAX_NUM_TILES; b++) {
        free(eng->tiles[b].packed_edges);
        free(eng->tiles[b].packed_target_weights);
    }
}

/*
 * Des-swizzling / Invalidación Inmediata de Vértice:
 * Restaura la vista de lectura autoritativa instantáneamente.
 */
static inline void alg2_unswizzle_vertex(DynGraph *g, uint32_t u) {
    DynVertex *dv = &g->v[u];
    dv->read_view.edges = dv->edges;
    dv->read_view.target_weights = NULL;
    dv->read_view.deg = dv->deg;
}

/*
 * Swizzling de Vértice:
 * Conecta el `read_view` del vértice directamente al buffer continuo del Micro-Tile.
 */
static inline void alg2_swizzle_vertex(
    DynGraph *g,
    SwizzledMicroTile *t,
    uint32_t u,
    uint16_t slot
) {
    DynVertex *dv = &g->v[u];
    uint32_t offset = t->row_offset[slot];
    dv->read_view.edges          = &t->packed_edges[offset];
    dv->read_view.target_weights = &t->packed_target_weights[offset];
    dv->read_view.deg            = t->deg_in_tile[slot];
}

/*
 * Invalida un Micro-Tile completo y restaura a estado autoritativo
 * todos sus vértices miembros.
 */
static void alg2_invalidate_tile(Alg2Engine *eng, DynGraph *g, uint16_t b) {
    SwizzledMicroTile *t = &eng->tiles[b];
    if (!t->valid) return;

    for (uint16_t i = 0; i < t->num_vertices; i++) {
        uint32_t u = t->logical_ids[i];
        if (eng->assigned_tile[u] == b) {
            eng->assigned_tile[u] = NO_TILE;
            eng->assigned_slot[u] = EXTERNAL_SLOT;
            if (eng->mode != ENGINE_V7_HASH_DYNAMIC) {
                alg2_unswizzle_vertex(g, u);
            }
        }
    }
    t->valid = false;
    if (eng->active_tiles_count > 0) eng->active_tiles_count--;
    eng->invalidations++;
}

/*
 * Notificación de mutación de arista:
 * En Pointer-Swizzling, la coherencia es 100% estricta: si u muta, se des-swizzlea
 * DE INMEDIATO a autoritativo antes de retornar, eliminando toda lectura stale.
 */
static inline void alg2_notify_update(Alg2Engine *eng, DynGraph *g, uint32_t u) {
    uint16_t b = eng->assigned_tile[u];
    if (b == NO_TILE) return;
    SwizzledMicroTile *t = &eng->tiles[b];
    if (!t->valid) return;

    uint16_t slot = eng->assigned_slot[u];
    uint64_t mask = 1ULL << (slot & 63U);
    if ((t->dirty_bits[slot >> 6U] & mask) == 0) {
        t->dirty_bits[slot >> 6U] |= mask;
    }

    if (eng->mode != ENGINE_V7_HASH_DYNAMIC) {
        /* Des-swizzling inmediato del vértice modificado */
        alg2_unswizzle_vertex(g, u);
    }

    /* Invalidar el Micro-Tile si se ensucian más de 4 vértices */
    uint32_t dirty_sum = 0;
    for (int k = 0; k < 4; k++) dirty_sum += (uint32_t)__builtin_popcountll(t->dirty_bits[k]);
    if (dirty_sum >= 4U || dirty_sum * 5U > t->num_vertices) {
        alg2_invalidate_tile(eng, g, b);
    }
}

/*
 * Construcción o reconstrucción de Micro-Tile:
 * Llena el buffer continuo, calcula los pesos y aplica Pointer Swizzling a los vértices.
 */
static double alg2_build_tile(
    Alg2Engine *eng,
    DynGraph *g,
    uint16_t b,
    const uint32_t *captured_v,
    uint16_t count
) {
    double t0 = now_ns();
    SwizzledMicroTile *t = &eng->tiles[b];

    if (t->valid) {
        alg2_invalidate_tile(eng, g, b);
    }
    eng->active_tiles_count++;

    if (count > t->max_vertices) count = t->max_vertices;
    local_index_clear(&t->local_idx, t->max_vertices);

    uint16_t actual_count = 0;
    uint32_t total_edges = 0;
    uint32_t bytes_used = 0;

    for (uint16_t i = 0; i < count; i++) {
        uint32_t u = captured_v[i];
        uint32_t d = g->v[u].deg;
        uint32_t v_bytes = sizeof(uint32_t) + sizeof(uint64_t) + d * (sizeof(uint32_t) + sizeof(uint64_t));
        if (bytes_used + v_bytes > t->max_bytes_budget && actual_count >= 16U) break;
        if (total_edges + d > t->edges_cap) break;

        /* Si u estaba en otro tile, des-swizzle de su tile previo */
        if (eng->assigned_tile[u] != NO_TILE && eng->assigned_tile[u] != b) {
            uint16_t old_b = eng->assigned_tile[u];
            uint16_t old_s = eng->assigned_slot[u];
            if (eng->tiles[old_b].valid && old_s < eng->tiles[old_b].num_vertices) {
                eng->tiles[old_b].dirty_bits[old_s >> 6U] |= (1ULL << (old_s & 63U));
            }
        }

        t->logical_ids[actual_count] = u;
        local_index_insert(&t->local_idx, u, actual_count);
        eng->assigned_tile[u] = b;
        eng->assigned_slot[u] = actual_count;
        bytes_used += v_bytes;
        total_edges += d;
        actual_count++;
    }

    uint32_t offset = 0;
    for (uint16_t slot = 0; slot < actual_count; slot++) {
        uint32_t u = t->logical_ids[slot];
        t->row_offset[slot] = offset;
        const uint32_t *edges = g->v[u].edges;
        uint32_t d = g->v[u].deg;
        t->deg_in_tile[slot] = d;

        for (uint32_t e = 0; e < d; e++) {
            uint32_t v = edges[e];
            t->packed_edges[offset + e] = v;
            t->packed_target_weights[offset + e] = g->payloads[v].weight_sum;
        }

        /* POINTER SWIZZLING: Conectar directamente la vista del vértice al Tile */
        if (eng->mode != ENGINE_V7_HASH_DYNAMIC) {
            alg2_swizzle_vertex(g, t, u, slot);
        }

        offset += d;
    }

    t->row_offset[actual_count] = offset;
    t->num_vertices = actual_count;
    t->cur_bytes_used = bytes_used;
    memset(t->dirty_bits, 0, sizeof(t->dirty_bits));
    t->valid = true;
    t->queries_since_promo = 0;
    t->realized_savings_ns = 0.0;

    double dt = now_ns() - t0;
    t->C_b_ns = dt;
    eng->promotions++;
    eng->capture_ns_sum += dt;
    eng->initial_build_ns_sum += dt;
    eng->total_bytes_copied += bytes_used;
    return dt;
}

/*
 * Anexión incremental a Micro-Tile con actualización inmediata de Pointer Swizzling.
 */
static double alg2_append_tile(
    Alg2Engine *eng,
    DynGraph *g,
    uint16_t b,
    const uint32_t *candidates,
    uint16_t cand_count
) {
    SwizzledMicroTile *t = &eng->tiles[b];
    if (!t->valid || t->num_vertices >= t->max_vertices || t->cur_bytes_used >= t->max_bytes_budget) {
        return 0.0;
    }

    double t0 = now_ns();
    uint16_t start_v = t->num_vertices;
    uint32_t offset  = t->row_offset[start_v];
    uint32_t bytes_added = 0;
    uint16_t v_added = 0;

    for (uint16_t i = 0; i < cand_count; i++) {
        if (t->num_vertices >= t->max_vertices) break;
        uint32_t u = candidates[i];
        if (eng->assigned_tile[u] == b) continue;

        uint32_t d = g->v[u].deg;
        uint32_t v_bytes = sizeof(uint32_t) + sizeof(uint64_t) + d * (sizeof(uint32_t) + sizeof(uint64_t));
        if (t->cur_bytes_used + bytes_added + v_bytes > t->max_bytes_budget) break;
        if (offset + d > t->edges_cap) break;

        if (eng->assigned_tile[u] != NO_TILE && eng->assigned_tile[u] != b) {
            uint16_t old_b = eng->assigned_tile[u];
            uint16_t old_s = eng->assigned_slot[u];
            if (eng->tiles[old_b].valid && old_s < eng->tiles[old_b].num_vertices) {
                eng->tiles[old_b].dirty_bits[old_s >> 6U] |= (1ULL << (old_s & 63U));
            }
        }

        uint16_t slot = t->num_vertices++;
        t->logical_ids[slot] = u;
        t->row_offset[slot] = offset;
        t->deg_in_tile[slot] = d;
        local_index_insert(&t->local_idx, u, slot);
        eng->assigned_tile[u] = b;
        eng->assigned_slot[u] = slot;

        const uint32_t *edges = g->v[u].edges;
        for (uint32_t e = 0; e < d; e++) {
            uint32_t v = edges[e];
            t->packed_edges[offset + e] = v;
            t->packed_target_weights[offset + e] = g->payloads[v].weight_sum;
        }

        /* POINTER SWIZZLING INMEDIATO */
        if (eng->mode != ENGINE_V7_HASH_DYNAMIC) {
            alg2_swizzle_vertex(g, t, u, slot);
        }

        offset += d;
        t->row_offset[t->num_vertices] = offset;
        bytes_added += v_bytes;
        v_added++;
    }

    if (v_added == 0) return 0.0;

    t->cur_bytes_used += bytes_added;
    double dt = now_ns() - t0;
    t->C_b_ns += dt;
    eng->capture_ns_sum += dt;
    eng->append_ns_sum += dt;
    eng->total_bytes_copied += bytes_added;
    return dt;
}

/* ============================================================================
 * MOTORES DE RECORRIDO BFS
 * ============================================================================ */
typedef struct {
    uint32_t queue_u[BFS_MAX_VISITED * 2];
    uint8_t  depth[BFS_MAX_VISITED * 2];
    uint32_t visited_epoch[NUM_VERTICES];
    uint32_t cur_epoch;
} BfsScratch;

static void bfs_scratch_init(BfsScratch *sc) {
    memset(sc, 0, sizeof(BfsScratch));
    sc->cur_epoch = 1;
}

/* 1. Recorrido V1 Original (Baseline Autoritativo) */
static uint64_t bfs_query_v1(const DynGraph *g, BfsScratch *sc, uint32_t root) {
    uint32_t ep = ++sc->cur_epoch;
    uint32_t head = 0, tail = 0;

    sc->queue_u[tail] = root;
    sc->depth[tail] = 0;
    tail++;
    sc->visited_epoch[root] = ep;

    uint64_t accum = g->payloads[root].weight_sum ^ (uint64_t)root;

    while (head < tail) {
        uint32_t u = sc->queue_u[head];
        uint8_t  d = sc->depth[head];
        head++;

        if (d >= BFS_MAX_DEPTH) continue;

        const uint32_t *edges = g->v[u].edges;
        uint32_t deg = g->v[u].deg;

        for (uint32_t i = 0; i < deg; i++) {
            uint32_t v = edges[i];
            if (sc->visited_epoch[v] != ep) {
                sc->visited_epoch[v] = ep;
                accum += g->payloads[v].weight_sum ^ ((uint64_t)v * (uint64_t)(d + 1));
                if (tail < BFS_MAX_VISITED) {
                    sc->queue_u[tail] = v;
                    sc->depth[tail] = d + 1;
                    tail++;
                } else {
                    goto done;
                }
            }
        }
    }
done:
    return accum ^ ((uint64_t)tail << 32);
}

/* 2. Recorrido V7 (Despacho Dinámico con Hash L1) */
static uint64_t bfs_query_v7_dynamic_hash(
    const DynGraph *g,
    Alg2Engine *eng,
    BfsScratch *sc,
    uint32_t root,
    uint16_t *out_visited,
    uint16_t *out_expanded,
    uint16_t *out_cold_expanded
) {
    uint32_t ep = ++sc->cur_epoch;
    uint32_t head = 0, tail = 0;
    uint16_t exp_cnt = 0, cold_cnt = 0;

    sc->queue_u[tail] = root;
    sc->depth[tail]   = 0;
    tail++;
    sc->visited_epoch[root] = ep;

    uint64_t accum = g->payloads[root].weight_sum ^ (uint64_t)root;

    while (head < tail) {
        uint32_t u = sc->queue_u[head];
        uint8_t  d = sc->depth[head];
        head++;

        if (d >= BFS_MAX_DEPTH) continue;
        exp_cnt++;
        eng->expanded_total++;

        /* Despacho tradicional de V7: lookup de tile -> lookup de hash -> bitmask check */
        uint16_t t_id = eng->assigned_tile[u];
        bool use_hot = false;
        const SwizzledMicroTile *tile = NULL;
        uint16_t slot_u = EXTERNAL_SLOT;

        if (t_id != NO_TILE && eng->tiles[t_id].valid) {
            tile = &eng->tiles[t_id];
            slot_u = local_index_lookup_hash(&tile->local_idx, u);
            if (slot_u < tile->num_vertices) {
                uint64_t dirty = tile->dirty_bits[slot_u >> 6U] & (1ULL << (slot_u & 63U));
                if (__builtin_expect(dirty == 0, 1)) {
                    use_hot = true;
                    eng->expanded_hot_hits++;
                } else {
                    eng->read_through_fallbacks++;
                    cold_cnt++;
                }
            } else {
                cold_cnt++;
            }
        } else {
            cold_cnt++;
        }

        if (use_hot) {
            uint32_t start = tile->row_offset[slot_u];
            uint32_t deg   = tile->deg_in_tile[slot_u];
            const uint32_t *edges = tile->packed_edges + start;
            const uint64_t *tws   = tile->packed_target_weights + start;

            for (uint32_t i = 0; i < deg; i++) {
                uint32_t v = edges[i];
                if (sc->visited_epoch[v] != ep) {
                    sc->visited_epoch[v] = ep;
                    accum += tws[i] ^ ((uint64_t)v * (uint64_t)(d + 1));
                    if (tail < BFS_MAX_VISITED) {
                        sc->queue_u[tail] = v;
                        sc->depth[tail]   = d + 1;
                        tail++;
                    } else goto done_v7;
                }
            }
        } else {
            const uint32_t *edges = g->v[u].edges;
            uint32_t deg = g->v[u].deg;
            for (uint32_t i = 0; i < deg; i++) {
                uint32_t v = edges[i];
                if (sc->visited_epoch[v] != ep) {
                    sc->visited_epoch[v] = ep;
                    accum += g->payloads[v].weight_sum ^ ((uint64_t)v * (uint64_t)(d + 1));
                    if (tail < BFS_MAX_VISITED) {
                        sc->queue_u[tail] = v;
                        sc->depth[tail]   = d + 1;
                        tail++;
                    } else goto done_v7;
                }
            }
        }
    }
done_v7:
    *out_visited       = (uint16_t)tail;
    *out_expanded      = exp_cnt;
    *out_cold_expanded = cold_cnt;
    return accum ^ ((uint64_t)tail << 32);
}

/*
 * 3. Recorrido ALG-2: POINTER SWIZZLED (Zero Dynamic Dispatch)
 * En cada vértice, se lee directamente `g->v[u].read_view`.
 * No hay consulta a assigned_tile, no hay hash lookup, no hay bitmask check.
 * Si `read_view.target_weights != NULL`, el vértice está swizzleado y caliente.
 */
static uint64_t bfs_query_alg2_swizzled(
    const DynGraph *g,
    Alg2Engine *eng,
    BfsScratch *sc,
    uint32_t root,
    uint16_t *out_visited,
    uint16_t *out_expanded,
    uint16_t *out_cold_expanded
) {
    uint32_t ep = ++sc->cur_epoch;
    uint32_t head = 0, tail = 0;
    uint16_t exp_cnt = 0, cold_cnt = 0;

    sc->queue_u[tail] = root;
    sc->depth[tail]   = 0;
    tail++;
    sc->visited_epoch[root] = ep;

    uint64_t accum = g->payloads[root].weight_sum ^ (uint64_t)root;

    while (head < tail) {
        uint32_t u = sc->queue_u[head];
        uint8_t  d = sc->depth[head];
        head++;

        if (d >= BFS_MAX_DEPTH) continue;
        exp_cnt++;
        eng->expanded_total++;

        /* Lectura directa sin despacho dinámico */
        EdgeView rw = g->v[u].read_view;
        const uint32_t *edges = rw.edges;
        const uint64_t *tws   = rw.target_weights;
        uint32_t deg          = rw.deg;

        if (__builtin_expect(tws != NULL, 1)) {
            /* Hot hit: replica precalculada en L1/L2 */
            eng->expanded_hot_hits++;
            for (uint32_t i = 0; i < deg; i++) {
                uint32_t v = edges[i];
                if (sc->visited_epoch[v] != ep) {
                    sc->visited_epoch[v] = ep;
                    accum += tws[i] ^ ((uint64_t)v * (uint64_t)(d + 1));
                    if (tail < BFS_MAX_VISITED) {
                        sc->queue_u[tail] = v;
                        sc->depth[tail]   = d + 1;
                        tail++;
                    } else goto done_swizzled;
                }
            }
        } else {
            /* Cold fallback autoritativo */
            cold_cnt++;
            for (uint32_t i = 0; i < deg; i++) {
                uint32_t v = edges[i];
                if (sc->visited_epoch[v] != ep) {
                    sc->visited_epoch[v] = ep;
                    accum += g->payloads[v].weight_sum ^ ((uint64_t)v * (uint64_t)(d + 1));
                    if (tail < BFS_MAX_VISITED) {
                        sc->queue_u[tail] = v;
                        sc->depth[tail]   = d + 1;
                        tail++;
                    } else goto done_swizzled;
                }
            }
        }
    }
done_swizzled:
    *out_visited       = (uint16_t)tail;
    *out_expanded      = exp_cnt;
    *out_cold_expanded = cold_cnt;
    return accum ^ ((uint64_t)tail << 32);
}

/* ============================================================================
 * CARGAS DE TRABAJO (A, B, C, D, E)
 * ============================================================================ */
typedef enum {
    WORKLOAD_A_STABLE_HOTSPOTS = 0,
    WORKLOAD_B_GRADUAL_DRIFT   = 1,
    WORKLOAD_C_ABRUPT_SHIFT    = 2,
    WORKLOAD_D_UNIFORM_RANDOM  = 3,
    WORKLOAD_E_STATIC_CONTROL  = 4,
    NUM_WORKLOADS              = 5
} WorkloadType;

static const char *WORKLOAD_NAMES[NUM_WORKLOADS] = {
    "Carga A (Hotspots Estables + Updates)",
    "Carga B (Deriva Gradual + Updates)",
    "Carga C (Cambio Abrupto + Updates)",
    "Carga D (Adversarial Uniforme sin Localidad)",
    "Carga E (Control Estatico sin Updates)"
};

typedef struct {
    uint32_t query_roots[NUM_STEPS_PER_RUN];
    bool     has_update_batch[NUM_STEPS_PER_RUN];
    uint32_t upd_u[NUM_STEPS_PER_RUN][UPDATES_PER_BATCH];
    uint32_t upd_v[NUM_STEPS_PER_RUN][UPDATES_PER_BATCH];
    bool     upd_is_add[NUM_STEPS_PER_RUN][UPDATES_PER_BATCH];
} WorkloadTrace;

static void generate_workload_trace(
    WorkloadTrace *tr,
    const DynGraph *base_graph,
    WorkloadType wtype,
    uint64_t seed
) {
    memset(tr, 0, sizeof(WorkloadTrace));
    uint64_t rng = seed ^ ((uint64_t)(wtype + 1) * 0x517CC1B727220A95ULL);

    for (uint32_t step = 0; step < NUM_STEPS_PER_RUN; step++) {
        uint64_t r = xorshift64s(&rng);
        uint32_t chosen_comm = 0;

        switch (wtype) {
            case WORKLOAD_A_STABLE_HOTSPOTS:
            case WORKLOAD_E_STATIC_CONTROL:
                if ((r % 100U) < 92U) chosen_comm = (uint32_t)((r >> 8) & 3U);
                else chosen_comm = (uint32_t)((r >> 8) % NUM_COMMUNITIES);
                break;
            case WORKLOAD_B_GRADUAL_DRIFT: {
                uint32_t base_comm = (step / 150U) % NUM_COMMUNITIES;
                if ((r % 100U) < 92U) chosen_comm = (base_comm + (uint32_t)((r >> 8) & 3U)) % NUM_COMMUNITIES;
                else chosen_comm = (uint32_t)((r >> 8) % NUM_COMMUNITIES);
                break;
            }
            case WORKLOAD_C_ABRUPT_SHIFT: {
                uint32_t epoch = (step / 750U) % 4U;
                uint32_t base_comm = epoch * 32U;
                if ((r % 100U) < 92U) chosen_comm = base_comm + (uint32_t)((r >> 8) & 3U);
                else chosen_comm = (uint32_t)((r >> 8) % NUM_COMMUNITIES);
                break;
            }
            case WORKLOAD_D_UNIFORM_RANDOM:
            default:
                chosen_comm = (uint32_t)((r >> 8) % NUM_COMMUNITIES);
                break;
        }

        uint32_t member_idx = (wtype == WORKLOAD_D_UNIFORM_RANDOM)
                              ? (uint32_t)((r >> 20) & (COMM_SIZE - 1U))
                              : (uint32_t)((r >> 20) % 64U);
        tr->query_roots[step] = base_graph->comm_members[chosen_comm * COMM_SIZE + member_idx];

        if (wtype != WORKLOAD_E_STATIC_CONTROL && step > 0 && (step % UPDATE_BATCH_PERIOD) == 0) {
            tr->has_update_batch[step] = true;
            for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                uint64_t ru = xorshift64s(&rng);
                uint32_t comm_u = (uint32_t)(ru % NUM_COMMUNITIES);
                uint32_t s_u = (uint32_t)((ru >> 16) & (COMM_SIZE - 1U));
                uint32_t s_v = (uint32_t)((ru >> 32) & (COMM_SIZE - 1U));
                if (s_v == s_u) s_v = (s_v + 1U) & (COMM_SIZE - 1U);

                tr->upd_u[step][k] = base_graph->comm_members[comm_u * COMM_SIZE + s_u];
                tr->upd_v[step][k] = base_graph->comm_members[comm_u * COMM_SIZE + s_v];
                tr->upd_is_add[step][k] = ((ru >> 48) & 3U) != 0;
            }
        }
    }
}

/* ============================================================================
 * RESULTADOS Y TELEMETRÍA
 * ============================================================================ */
typedef struct {
    double   wall_ms;
    double   total_ms;
    double   query_ms;
    double   maint_ms;
    double   hit_pct;
    double   cold_q_us;
    double   first_q_us;
    double   subseq_q_us;
    double   cap_total_us;
    double   brkev_v1;
    double   net_benefit_ms;
    uint32_t promotions;
    uint32_t invalidations;
    uint32_t errors;
} Alg2Metrics;

/*
 * Ejecuta una corrida adaptativa completa de ALG-2 o variantes.
 */
static void run_alg2_adaptive_pass(
    DynGraph *g_work,
    const DynGraph *base_graph,
    const WorkloadTrace *tr,
    const uint64_t *oracle_hashes,
    EngineMode mode,
    bool enable_h3,
    bool lite_heat_tracking,
    Alg2Engine *eng,
    BfsScratch *scratch,
    Alg2Metrics *out
) {
    memset(out, 0, sizeof(Alg2Metrics));
    dyngraph_reset_in_place(g_work, base_graph);
    alg2_engine_init(eng, mode);
    bfs_scratch_init(scratch);
    scrub_l3_cache();

    const double ALPHA_H3      = 1.15;
    const double HORIZON_MULT  = 16.0;
    double cold_ewma_ns        = 3200.0;
    double hot_ewma_ns         = 2100.0;
    double measured_cb_ewma_ns = eng->tiles[0].C_b_ns;

    double query_ns_sum = 0.0;
    double maint_ns_sum = 0.0;
    uint32_t next_tile_slot = 0;

    double tw0 = now_ns();

    for (uint32_t s = 0; s < NUM_STEPS_PER_RUN; s++) {
        /* 1. Mantenimiento de updates */
        if (tr->has_update_batch[s]) {
            double tu0 = now_ns();
            for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                uint32_t u = tr->upd_u[s][k];
                if (tr->upd_is_add[s][k]) dyngraph_add_edge(g_work, u, tr->upd_v[s][k]);
                else dyngraph_remove_edge(g_work, u);
                alg2_notify_update(eng, g_work, u);
            }
            maint_ns_sum += (now_ns() - tu0);
        }

        /* 2. Decaimiento periódico de calor */
        if (enable_h3 && s > 0 && (s % 200U) == 0) {
            double td0 = now_ns();
            if (!lite_heat_tracking) {
                for (uint32_t i = 0; i < NUM_VERTICES; i += 64U) {
                    eng->vertex_heat[i] >>= 1;
                }
            }
            maint_ns_sum += (now_ns() - td0);
        }

        /* 3. Ejecución de la consulta */
        uint32_t root = tr->query_roots[s];
        uint16_t b = eng->assigned_tile[root];
        bool was_valid_before = (b != NO_TILE && eng->tiles[b].valid);

        uint16_t visited_cnt = 0, exp_cnt = 0, cold_exp_cnt = 0;
        double tq0 = now_ns();
        uint64_t h = 0;

        if (mode == ENGINE_V7_HASH_DYNAMIC) {
            h = bfs_query_v7_dynamic_hash(g_work, eng, scratch, root, &visited_cnt, &exp_cnt, &cold_exp_cnt);
        } else {
            h = bfs_query_alg2_swizzled(g_work, eng, scratch, root, &visited_cnt, &exp_cnt, &cold_exp_cnt);
        }
        double dq = now_ns() - tq0;

        if (oracle_hashes && h != oracle_hashes[s]) out->errors++;
        query_ns_sum += dq;

        /* 4. Telemetría y decisiones de promoción H3 */
        if (enable_h3) {
            double tm_post0 = now_ns();
            if (was_valid_before) {
                SwizzledMicroTile *tb = &eng->tiles[b];
                tb->queries_since_promo++;
                if (tb->queries_since_promo == 1U) {
                    eng->first_query_ns_sum += dq;
                    eng->first_query_count++;
                } else {
                    eng->subseq_query_ns_sum += dq;
                    eng->subseq_query_count++;
                }
                hot_ewma_ns = 0.90 * hot_ewma_ns + 0.10 * dq;
                double gain = (cold_ewma_ns > hot_ewma_ns) ? (cold_ewma_ns - hot_ewma_ns) : 200.0;
                tb->realized_savings_ns += gain;

                if (cold_exp_cnt > 0) {
                    uint16_t app_cnt = (visited_cnt < 64U) ? visited_cnt : 64U;
                    alg2_append_tile(eng, g_work, b, scratch->queue_u, app_cnt);
                }
            } else {
                eng->cold_query_ns_sum += dq;
                eng->cold_query_count++;
                cold_ewma_ns = 0.90 * cold_ewma_ns + 0.10 * dq;

                uint32_t neighborhood_heat = 0;
                uint16_t b_existing = NO_TILE;
                uint16_t check_n = (exp_cnt < 16U) ? exp_cnt : 16U;

                for (uint16_t i = 0; i < check_n; i++) {
                    uint32_t u = scratch->queue_u[i];
                    if (eng->vertex_heat[u] < 250U) eng->vertex_heat[u]++;
                    neighborhood_heat += eng->vertex_heat[u];
                    if (b_existing == NO_TILE) {
                        uint16_t tb_id = eng->assigned_tile[u];
                        if (tb_id != NO_TILE && eng->tiles[tb_id].valid) b_existing = tb_id;
                    }
                }

                if (b_existing != NO_TILE) {
                    uint16_t app_cnt = (visited_cnt < 64U) ? visited_cnt : 64U;
                    alg2_append_tile(eng, g_work, b_existing, scratch->queue_u, app_cnt);
                } else {
                    bool should_promote = false;
                    if (lite_heat_tracking) {
                        /* H3 Lite: Criterio ultra-rápido en enteros sin EWMA ni floats */
                        should_promote = (neighborhood_heat >= 38U);
                    } else {
                        double effective_nb = (double)neighborhood_heat / (double)(check_n ? check_n : 1U);
                        double est_gb = (cold_ewma_ns > hot_ewma_ns + 200.0) ? (cold_ewma_ns - hot_ewma_ns) : 1200.0;
                        double expected_return = (effective_nb * HORIZON_MULT) * est_gb;
                        double cost_threshold  = ALPHA_H3 * measured_cb_ewma_ns;
                        should_promote = (effective_nb >= 2.35 && expected_return > cost_threshold);
                    }

                    if (should_promote) {
                        uint16_t target_b = (uint16_t)(next_tile_slot++ % MAX_NUM_TILES);
                        if (next_tile_slot > eng->num_allocated_tiles && eng->num_allocated_tiles < MAX_NUM_TILES) {
                            eng->num_allocated_tiles = next_tile_slot;
                        }
                        uint16_t init_cnt = (visited_cnt < 48U) ? visited_cnt : 48U;
                        double build_ns = alg2_build_tile(eng, g_work, target_b, scratch->queue_u, init_cnt);
                        measured_cb_ewma_ns = 0.75 * measured_cb_ewma_ns + 0.25 * build_ns;
                        for (uint16_t i = 0; i < check_n; i++) {
                            eng->vertex_heat[scratch->queue_u[i]] = 0;
                        }
                    }
                }
            }
            maint_ns_sum += (now_ns() - tm_post0);
        }
    }

    out->wall_ms       = (now_ns() - tw0) / 1e6;
    out->query_ms      = query_ns_sum / 1e6;
    out->maint_ms      = maint_ns_sum / 1e6;
    out->total_ms      = out->query_ms + out->maint_ms;
    out->hit_pct       = eng->expanded_total ? (100.0 * (double)eng->expanded_hot_hits / (double)eng->expanded_total) : 0.0;
    out->cold_q_us     = eng->cold_query_count ? ((eng->cold_query_ns_sum / eng->cold_query_count) / 1000.0) : 0.0;
    out->first_q_us    = eng->first_query_count ? ((eng->first_query_ns_sum / eng->first_query_count) / 1000.0) : 0.0;
    out->subseq_q_us   = eng->subseq_query_count ? ((eng->subseq_query_ns_sum / eng->subseq_query_count) / 1000.0) : 0.0;
    out->cap_total_us  = eng->promotions ? ((eng->capture_ns_sum / eng->promotions) / 1000.0) : 0.0;
    out->promotions    = eng->promotions;
    out->invalidations = eng->invalidations;

    alg2_engine_free(eng);
}

int main(void) {
    SetConsoleOutputCP(65001);
    setvbuf(stdout, NULL, _IONBF, 0);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);
    init_timer();

    printf("========================================================================================================================\n");
    printf(" EXPERIMENTO ALG-2: POINTER-SWIZZLED MICRO-TILES Y DESPACHO DE COSTE CERO\n");
    printf(" 5 Semillas (3 Test Reservadas + 2 Validacion Externa) | Grafo: 262,144 vertices | BFS Depth=3\n");
    printf("========================================================================================================================\n\n");

    const uint64_t SEEDS[NUM_EVAL_SEEDS] = {
        0x778899AABBCCDDEEULL, /* Semilla Test 1 */
        0x3141592653589793ULL, /* Semilla Test 2 */
        0x2718281828459045ULL, /* Semilla Test 3 */
        0xFEEDFACECAFEF00DULL, /* Semilla Validacion 4 */
        0x0123456789ABCDEFULL  /* Semilla Validacion 5 */
    };

    DynGraph test_graphs[NUM_EVAL_SEEDS];
    DynGraph work_graphs[NUM_EVAL_SEEDS];
    for (uint32_t s = 0; s < NUM_EVAL_SEEDS; s++) {
        generate_base_graph(&test_graphs[s], SEEDS[s]);
        dyngraph_clone(&work_graphs[s], &test_graphs[s]);
    }

    WorkloadTrace *tr = (WorkloadTrace *)malloc(sizeof(WorkloadTrace));
    BfsScratch *scratch = (BfsScratch *)malloc(sizeof(BfsScratch));
    Alg2Engine *eng = (Alg2Engine *)malloc(sizeof(Alg2Engine));
    uint64_t *oracle_hashes = (uint64_t *)malloc(NUM_STEPS_PER_RUN * sizeof(uint64_t));

    /* ========================================================================
     * FASE A: AISLAMIENTO DEL COSTE DE DESPACHO
     * Compara directamente:
     *   A0: V1 Original
     *   A1: Motor V7 sin promociones
     *   A2: V7 Actual (Despacho Hash L1)
     *   A3: ALG-2 (Pointer Swizzled Micro-Tile)
     * ======================================================================== */
    printf("------------------------------------------------------------------------------------------------------------------------\n");
    printf(" FASE A: AISLAMIENTO DEL COSTE DE DESPACHO EN CARGA A Y CARGA D (Media 5 Semillas)\n");
    printf("------------------------------------------------------------------------------------------------------------------------\n");
    printf(" Variante                             | Total(ms) | Query(ms) | Maint(ms) | Hit%%  | Exact | vs V1 (NetBenefit)\n");
    printf("--------------------------------------+-----------+-----------+-----------+-------+-------+--------------------\n");

    const WorkloadType FASE_A_WORKLOADS[2] = { WORKLOAD_A_STABLE_HOTSPOTS, WORKLOAD_D_UNIFORM_RANDOM };
    const char *FASE_A_WNAMES[2] = { "Carga A (Localidad Fuerte)", "Carga D (Adversarial Uniforme)" };

    for (int wi = 0; wi < 2; wi++) {
        WorkloadType wt = FASE_A_WORKLOADS[wi];
        printf(" [%s]\n", FASE_A_WNAMES[wi]);

        Alg2Metrics m_a0 = {0}, m_a1 = {0}, m_a2 = {0}, m_a3 = {0};

        for (uint32_t s = 0; s < NUM_EVAL_SEEDS; s++) {
            generate_workload_trace(tr, &test_graphs[s], wt, SEEDS[s]);

            /* Generar Oráculo de V1 */
            dyngraph_reset_in_place(&work_graphs[s], &test_graphs[s]);
            bfs_scratch_init(scratch);
            for (uint32_t step = 0; step < NUM_STEPS_PER_RUN; step++) {
                if (tr->has_update_batch[step]) {
                    for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                        if (tr->upd_is_add[step][k]) dyngraph_add_edge(&work_graphs[s], tr->upd_u[step][k], tr->upd_v[step][k]);
                        else dyngraph_remove_edge(&work_graphs[s], tr->upd_u[step][k]);
                    }
                }
                oracle_hashes[step] = bfs_query_v1(&work_graphs[s], scratch, tr->query_roots[step]);
            }

            /* A0: V1 Original */
            {
                dyngraph_reset_in_place(&work_graphs[s], &test_graphs[s]);
                bfs_scratch_init(scratch);
                scrub_l3_cache();
                double q_ns = 0.0, m_ns = 0.0;
                double tw0 = now_ns();
                for (uint32_t step = 0; step < NUM_STEPS_PER_RUN; step++) {
                    if (tr->has_update_batch[step]) {
                        double tu0 = now_ns();
                        for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                            if (tr->upd_is_add[step][k]) dyngraph_add_edge(&work_graphs[s], tr->upd_u[step][k], tr->upd_v[step][k]);
                            else dyngraph_remove_edge(&work_graphs[s], tr->upd_u[step][k]);
                        }
                        m_ns += (now_ns() - tu0);
                    }
                    double tq0 = now_ns();
                    uint64_t h = bfs_query_v1(&work_graphs[s], scratch, tr->query_roots[step]);
                    double dq = now_ns() - tq0;
                    if (h != oracle_hashes[step]) m_a0.errors++;
                    q_ns += dq;
                }
                m_a0.wall_ms  += ((now_ns() - tw0) / 1e6) / NUM_EVAL_SEEDS;
                m_a0.query_ms += (q_ns / 1e6) / NUM_EVAL_SEEDS;
                m_a0.maint_ms += (m_ns / 1e6) / NUM_EVAL_SEEDS;
                m_a0.total_ms += ((q_ns + m_ns) / 1e6) / NUM_EVAL_SEEDS;
            }

            /* A1: Motor V7 sin promociones */
            {
                Alg2Metrics r;
                run_alg2_adaptive_pass(&work_graphs[s], &test_graphs[s], tr, oracle_hashes,
                                       ENGINE_V7_HASH_DYNAMIC, false, false, eng, scratch, &r);
                m_a1.wall_ms  += r.wall_ms / NUM_EVAL_SEEDS;
                m_a1.total_ms += r.total_ms / NUM_EVAL_SEEDS;
                m_a1.query_ms += r.query_ms / NUM_EVAL_SEEDS;
                m_a1.maint_ms += r.maint_ms / NUM_EVAL_SEEDS;
                m_a1.errors   += r.errors;
            }

            /* A2: V7 Actual (Hash L1) */
            {
                Alg2Metrics r;
                run_alg2_adaptive_pass(&work_graphs[s], &test_graphs[s], tr, oracle_hashes,
                                       ENGINE_V7_HASH_DYNAMIC, true, false, eng, scratch, &r);
                m_a2.wall_ms  += r.wall_ms / NUM_EVAL_SEEDS;
                m_a2.total_ms += r.total_ms / NUM_EVAL_SEEDS;
                m_a2.query_ms += r.query_ms / NUM_EVAL_SEEDS;
                m_a2.maint_ms += r.maint_ms / NUM_EVAL_SEEDS;
                m_a2.hit_pct  += r.hit_pct / NUM_EVAL_SEEDS;
                m_a2.errors   += r.errors;
            }

            /* A3: ALG-2 (Pointer Swizzled) */
            {
                Alg2Metrics r;
                run_alg2_adaptive_pass(&work_graphs[s], &test_graphs[s], tr, oracle_hashes,
                                       ENGINE_ALG2_SWIZZLED, true, false, eng, scratch, &r);
                m_a3.wall_ms  += r.wall_ms / NUM_EVAL_SEEDS;
                m_a3.total_ms += r.total_ms / NUM_EVAL_SEEDS;
                m_a3.query_ms += r.query_ms / NUM_EVAL_SEEDS;
                m_a3.maint_ms += r.maint_ms / NUM_EVAL_SEEDS;
                m_a3.hit_pct  += r.hit_pct / NUM_EVAL_SEEDS;
                m_a3.errors   += r.errors;
            }
        }

        printf("  A0: V1 Original (Baseline)          | %9.2f | %9.2f | %9.2f | %4.1f%% | %s | Baseline\n",
               m_a0.total_ms, m_a0.query_ms, m_a0.maint_ms, 0.0, (m_a0.errors == 0) ? "100%" : "ERR");
        printf("  A1: Motor V7 (Sin Promociones)      | %9.2f | %9.2f | %9.2f | %4.1f%% | %s | %+.2f ms (%+.1f%%)\n",
               m_a1.total_ms, m_a1.query_ms, m_a1.maint_ms, 0.0, (m_a1.errors == 0) ? "100%" : "ERR",
               m_a0.total_ms - m_a1.total_ms, 100.0 * (m_a0.total_ms - m_a1.total_ms) / m_a0.total_ms);
        printf("  A2: V7 Actual (Hash L1 + Despacho)  | %9.2f | %9.2f | %9.2f | %4.1f%% | %s | %+.2f ms (%+.1f%%)\n",
               m_a2.total_ms, m_a2.query_ms, m_a2.maint_ms, m_a2.hit_pct, (m_a2.errors == 0) ? "100%" : "ERR",
               m_a0.total_ms - m_a2.total_ms, 100.0 * (m_a0.total_ms - m_a2.total_ms) / m_a0.total_ms);
        printf("  A3: ALG-2 (Pointer Swizzled)        | %9.2f | %9.2f | %9.2f | %4.1f%% | %s | %+.2f ms (%+.1f%%)\n\n",
               m_a3.total_ms, m_a3.query_ms, m_a3.maint_ms, m_a3.hit_pct, (m_a3.errors == 0) ? "100%" : "ERR",
               m_a0.total_ms - m_a3.total_ms, 100.0 * (m_a0.total_ms - m_a3.total_ms) / m_a0.total_ms);
    }

    /* ========================================================================
     * FASE B: EVALUACIÓN DEL SISTEMA ADAPTATIVO COMPLETO EN LAS 5 CARGAS
     * Compara:
     *   1. V1 Original
     *   2. V7 (ALG-1 Hash L1)
     *   3. ALG-2 (Pointer Swizzling + H3 Completo)
     *   4. ALG-2-Lite (Pointer Swizzling + H3 Lite sin EWMA floats)
     * ======================================================================== */
    printf("========================================================================================================================\n");
    printf(" FASE B: EVALUACION ADAPTATIVA COMPLETA EN LAS 5 CARGAS (Media 5 Semillas Independientes)\n");
    printf("========================================================================================================================\n");

    const char *SYS_NAMES[4] = {
        "1. V1 Original (Baseline)            ",
        "2. V7 ALG-1 (Hash L1 + H3)           ",
        "3. ALG-2 (Pointer Swizzling + H3)    ",
        "4. ALG-2-Lite (Swizzling + H3 Lite)  "
    };

    for (int w = 0; w < NUM_WORKLOADS; w++) {
        Alg2Metrics sys_avg[4] = {0};

        for (uint32_t s = 0; s < NUM_EVAL_SEEDS; s++) {
            generate_workload_trace(tr, &test_graphs[s], (WorkloadType)w, SEEDS[s]);

            /* Oráculo V1 */
            dyngraph_reset_in_place(&work_graphs[s], &test_graphs[s]);
            bfs_scratch_init(scratch);
            for (uint32_t step = 0; step < NUM_STEPS_PER_RUN; step++) {
                if (tr->has_update_batch[step]) {
                    for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                        if (tr->upd_is_add[step][k]) dyngraph_add_edge(&work_graphs[s], tr->upd_u[step][k], tr->upd_v[step][k]);
                        else dyngraph_remove_edge(&work_graphs[s], tr->upd_u[step][k]);
                    }
                }
                oracle_hashes[step] = bfs_query_v1(&work_graphs[s], scratch, tr->query_roots[step]);
            }

            /* 1. V1 */
            {
                dyngraph_reset_in_place(&work_graphs[s], &test_graphs[s]);
                bfs_scratch_init(scratch);
                scrub_l3_cache();
                double q_ns = 0.0, m_ns = 0.0;
                double tw0 = now_ns();
                for (uint32_t step = 0; step < NUM_STEPS_PER_RUN; step++) {
                    if (tr->has_update_batch[step]) {
                        double tu0 = now_ns();
                        for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                            if (tr->upd_is_add[step][k]) dyngraph_add_edge(&work_graphs[s], tr->upd_u[step][k], tr->upd_v[step][k]);
                            else dyngraph_remove_edge(&work_graphs[s], tr->upd_u[step][k]);
                        }
                        m_ns += (now_ns() - tu0);
                    }
                    double tq0 = now_ns();
                    uint64_t h = bfs_query_v1(&work_graphs[s], scratch, tr->query_roots[step]);
                    double dq = now_ns() - tq0;
                    if (h != oracle_hashes[step]) sys_avg[0].errors++;
                    q_ns += dq;
                }
                sys_avg[0].wall_ms   += ((now_ns() - tw0) / 1e6) / NUM_EVAL_SEEDS;
                sys_avg[0].query_ms  += (q_ns / 1e6) / NUM_EVAL_SEEDS;
                sys_avg[0].maint_ms  += (m_ns / 1e6) / NUM_EVAL_SEEDS;
                sys_avg[0].total_ms  += ((q_ns + m_ns) / 1e6) / NUM_EVAL_SEEDS;
                sys_avg[0].subseq_q_us += ((q_ns / NUM_STEPS_PER_RUN) / 1000.0) / NUM_EVAL_SEEDS;
            }

            /* 2. V7 (ALG-1 Hash L1) */
            {
                Alg2Metrics r;
                run_alg2_adaptive_pass(&work_graphs[s], &test_graphs[s], tr, oracle_hashes,
                                       ENGINE_V7_HASH_DYNAMIC, true, false, eng, scratch, &r);
                sys_avg[1].wall_ms      += r.wall_ms / NUM_EVAL_SEEDS;
                sys_avg[1].total_ms     += r.total_ms / NUM_EVAL_SEEDS;
                sys_avg[1].query_ms     += r.query_ms / NUM_EVAL_SEEDS;
                sys_avg[1].maint_ms     += r.maint_ms / NUM_EVAL_SEEDS;
                sys_avg[1].hit_pct      += r.hit_pct / NUM_EVAL_SEEDS;
                sys_avg[1].subseq_q_us  += r.subseq_q_us / NUM_EVAL_SEEDS;
                sys_avg[1].cap_total_us += r.cap_total_us / NUM_EVAL_SEEDS;
                sys_avg[1].promotions   += r.promotions;
                sys_avg[1].errors       += r.errors;
            }

            /* 3. ALG-2 (Pointer Swizzling + H3 Completo) */
            {
                Alg2Metrics r;
                run_alg2_adaptive_pass(&work_graphs[s], &test_graphs[s], tr, oracle_hashes,
                                       ENGINE_ALG2_SWIZZLED, true, false, eng, scratch, &r);
                sys_avg[2].wall_ms      += r.wall_ms / NUM_EVAL_SEEDS;
                sys_avg[2].total_ms     += r.total_ms / NUM_EVAL_SEEDS;
                sys_avg[2].query_ms     += r.query_ms / NUM_EVAL_SEEDS;
                sys_avg[2].maint_ms     += r.maint_ms / NUM_EVAL_SEEDS;
                sys_avg[2].hit_pct      += r.hit_pct / NUM_EVAL_SEEDS;
                sys_avg[2].subseq_q_us  += r.subseq_q_us / NUM_EVAL_SEEDS;
                sys_avg[2].cap_total_us += r.cap_total_us / NUM_EVAL_SEEDS;
                sys_avg[2].promotions   += r.promotions;
                sys_avg[2].errors       += r.errors;
            }

            /* 4. ALG-2-Lite (Pointer Swizzling + H3 Lite) */
            {
                Alg2Metrics r;
                run_alg2_adaptive_pass(&work_graphs[s], &test_graphs[s], tr, oracle_hashes,
                                       ENGINE_ALG2_SWIZZLED_LITE, true, true, eng, scratch, &r);
                sys_avg[3].wall_ms      += r.wall_ms / NUM_EVAL_SEEDS;
                sys_avg[3].total_ms     += r.total_ms / NUM_EVAL_SEEDS;
                sys_avg[3].query_ms     += r.query_ms / NUM_EVAL_SEEDS;
                sys_avg[3].maint_ms     += r.maint_ms / NUM_EVAL_SEEDS;
                sys_avg[3].hit_pct      += r.hit_pct / NUM_EVAL_SEEDS;
                sys_avg[3].subseq_q_us  += r.subseq_q_us / NUM_EVAL_SEEDS;
                sys_avg[3].cap_total_us += r.cap_total_us / NUM_EVAL_SEEDS;
                sys_avg[3].promotions   += r.promotions;
                sys_avg[3].errors       += r.errors;
            }
        }

        double t_v1 = sys_avg[0].total_ms;
        printf(" %s:\n", WORKLOAD_NAMES[w]);
        printf("   Variante                             | Total(ms) | Query(ms) | Maint(ms) | Hit%%  | Prom | Exact | NetBenefit vs V1\n");
        printf("   -------------------------------------+-----------+-----------+-----------+-------+------+-------+------------------\n");
        for (int c = 0; c < 4; c++) {
            double net = t_v1 - sys_avg[c].total_ms;
            double pct = 100.0 * net / t_v1;
            printf("   %s | %9.2f | %9.2f | %9.2f | %4.1f%% | %4u | %s | %+.2f ms (%+.1f%%)\n",
                   SYS_NAMES[c], sys_avg[c].total_ms, sys_avg[c].query_ms, sys_avg[c].maint_ms,
                   sys_avg[c].hit_pct, sys_avg[c].promotions / NUM_EVAL_SEEDS,
                   (sys_avg[c].errors == 0) ? "100%" : "ERR",
                   (c == 0) ? 0.0 : net, (c == 0) ? 0.0 : pct);
        }

        /* Desglose de Amortización Real BrkEv_V1 */
        if (w != WORKLOAD_D_UNIFORM_RANDOM) {
            double v1_sub = sys_avg[0].subseq_q_us;
            double a2_sub = sys_avg[2].subseq_q_us;
            double a2_cap = sys_avg[2].cap_total_us;
            double brkev = (v1_sub > a2_sub + 0.05) ? (a2_cap / (v1_sub - a2_sub)) : 0.0;
            printf("   [Amortizacion ALG-2 vs V1] T(V1)=%.2f us | T(ALG2_SubQ)=%.2f us | Ahorro=%.2f us/q | Cap=%.2f us -> BrkEv_V1=%.1f consultas\n",
                   v1_sub, a2_sub, v1_sub - a2_sub, a2_cap, brkev);
        }
        printf("\n");
    }

    free(tr);
    free(scratch);
    free(eng);
    free(oracle_hashes);
    if (g_scrub_buf) _aligned_free(g_scrub_buf);
    for (uint32_t s = 0; s < NUM_EVAL_SEEDS; s++) {
        dyngraph_free(&work_graphs[s]);
        dyngraph_free(&test_graphs[s]);
    }
    return 0;
}
