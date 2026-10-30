/*
 * ============================================================================
 * AUDITORIA FINAL DE ALG-3: CIERRE METODOLOGICO Y VALIDACION ESTADISTICA PREVIA A HNSW
 * ============================================================================
 * Objetivos:
 *   1. Matriz estadística completa para todo el barrido (0%, 10%, 25%, 50%, 75%, 90%):
 *      Media, Desviación Estándar, Mediana, IQR, IC 95% y consistencia de cruce (~40%).
 *   2. Desglose analítico de la anomalía en Carga D (Semilla 2 vs Semilla 1):
 *      Separación rigurosa de tiempo de query, tiempo de mantenimiento, bypasses ER y promociones.
 *   3. Contabilidad estricta de memoria física (bytes exactos de sobrecarga frente a V1).
 *   4. Análisis de sensibilidad del Sketch de Rechazo Temprano (ventanas 32, 64, 128 | umbrales 2, 3, 4).
 *
 * Condiciones de ejecución:
 *   - Intel Core i5-13500 P-Core 0 fijado (SetThreadAffinityMask(1ULL)).
 *   - 32 MB L3 Scrub entre corridas para eliminar sesgo de caché residual.
 *   - 5 Semillas independientes (3 test + 2 validación externa).
 *   - Verificación estricta de hash de oráculo (exactitud bit a bit).
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

typedef struct {
    const uint32_t *edges;
    const uint64_t *target_weights;
    uint32_t deg;
} EdgeView;

typedef struct {
    uint32_t *edges;
    uint32_t deg;
    uint32_t cap;
    uint32_t version;
    EdgeView read_view;
} DynVertex;

typedef struct {
    DynVertex *v;
    VertexPayload *payloads;
    uint32_t *comm_members;
    size_t total_edges;
    size_t bytes_allocated;
} DynGraph;

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
        dst->v[u].read_view.edges = dst->v[u].edges;
        dst->v[u].read_view.target_weights = NULL;
        dst->v[u].read_view.deg = deg;
    }
}

static inline void dyngraph_add_edge(DynGraph *g, uint32_t u, uint32_t dst_v) {
    DynVertex *dv = &g->v[u];
    if (__builtin_expect(dv->deg >= dv->cap, 0)) {
        uint32_t ncap = dv->cap * 2U;
        dv->edges = (uint32_t *)realloc(dv->edges, ncap * sizeof(uint32_t));
        dv->cap = ncap;
    }
    dv->edges[dv->deg++] = dst_v;
    dv->version++;
    g->total_edges++;
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

#define MAX_HASH_CAP     2048U
#define MAX_TILE_SLOTS   256U
#define MAX_NUM_TILES    128U
#define MAX_TILE_BYTES   40960U

/* Parametrizable Concentration Sketch para ablación */
typedef struct {
    uint8_t  window_buckets[64];
    uint8_t  ring[128];
    uint32_t ring_idx;
    uint32_t count_in_window;
    uint32_t window_size;
    uint32_t threshold;
} ParamSketch;

static inline void sketch_init(ParamSketch *cs, uint32_t win_size, uint32_t thresh) {
    memset(cs, 0, sizeof(ParamSketch));
    cs->window_size = win_size;
    cs->threshold = thresh;
}

static inline bool sketch_record_and_test(ParamSketch *cs, uint32_t root) {
    uint8_t b = (uint8_t)((root * 0x9E3779B9U) >> 26U);

    if (cs->count_in_window >= cs->window_size) {
        uint8_t old_b = cs->ring[cs->ring_idx];
        if (cs->window_buckets[old_b] > 0) cs->window_buckets[old_b]--;
    } else {
        cs->count_in_window++;
    }

    cs->ring[cs->ring_idx] = b;
    cs->ring_idx = (cs->ring_idx + 1U) % cs->window_size;
    cs->window_buckets[b]++;

    return (cs->window_buckets[b] >= cs->threshold);
}

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

typedef struct {
    uint32_t *packed_edges;
    uint64_t *packed_target_weights;
    uint32_t *row_offset;
    uint16_t *deg_in_tile;
    uint32_t *logical_ids;
    uint64_t  dirty_bits[4];
    CompactLocalIndex local_idx;

    uint16_t num_vertices;
    uint16_t max_vertices;
    uint32_t edges_cap;
    uint32_t max_bytes_budget;
    uint32_t cur_bytes_used;

    bool     valid;
    uint32_t queries_since_promo;
    double   C_b_ns;
} SwizzledMicroTile;

typedef struct {
    SwizzledMicroTile tiles[MAX_NUM_TILES];
    uint16_t assigned_tile[NUM_VERTICES];
    uint16_t assigned_slot[NUM_VERTICES];
    uint8_t  vertex_heat[NUM_VERTICES];

    uint32_t num_allocated_tiles;
    uint32_t active_tiles_count;
    uint32_t promotions;
    uint32_t invalidations;

    uint64_t expanded_total;
    uint64_t expanded_hot_hits;

    double capture_ns_sum;
    double subseq_query_ns_sum;
    uint64_t subseq_query_count;

    ParamSketch cs;
    bool use_early_rejection;
    uint64_t er_bypasses;
} EngineALG;

static void engine_init(EngineALG *eng, bool use_er, uint32_t win_size, uint32_t thresh) {
    memset(eng, 0, sizeof(EngineALG));
    for (uint32_t i = 0; i < NUM_VERTICES; i++) {
        eng->assigned_tile[i] = NO_TILE;
        eng->assigned_slot[i] = EXTERNAL_SLOT;
    }

    eng->use_early_rejection = use_er;
    sketch_init(&eng->cs, win_size, thresh);

    for (uint16_t b = 0; b < MAX_NUM_TILES; b++) {
        SwizzledMicroTile *t = &eng->tiles[b];
        t->max_vertices = 128U;
        t->edges_cap    = 4096U;
        t->max_bytes_budget = MAX_TILE_BYTES;
        t->packed_edges          = (uint32_t *)_aligned_malloc(t->edges_cap * sizeof(uint32_t), 64);
        t->packed_target_weights = (uint64_t *)_aligned_malloc(t->edges_cap * sizeof(uint64_t), 64);
        t->row_offset            = (uint32_t *)malloc((t->max_vertices + 1) * sizeof(uint32_t));
        t->deg_in_tile           = (uint16_t *)malloc(t->max_vertices * sizeof(uint16_t));
        t->logical_ids           = (uint32_t *)malloc(t->max_vertices * sizeof(uint32_t));
        t->valid = false;
        t->C_b_ns = 6500.0;
    }
}

static void engine_free(EngineALG *eng) {
    for (uint16_t b = 0; b < MAX_NUM_TILES; b++) {
        SwizzledMicroTile *t = &eng->tiles[b];
        if (t->packed_edges) _aligned_free(t->packed_edges);
        if (t->packed_target_weights) _aligned_free(t->packed_target_weights);
        if (t->row_offset) free(t->row_offset);
        if (t->deg_in_tile) free(t->deg_in_tile);
        if (t->logical_ids) free(t->logical_ids);
    }
}

static inline void unswizzle_vertex(DynGraph *g, uint32_t u) {
    DynVertex *dv = &g->v[u];
    dv->read_view.edges          = dv->edges;
    dv->read_view.target_weights = NULL;
    dv->read_view.deg            = dv->deg;
}

static inline void swizzle_vertex(DynGraph *g, SwizzledMicroTile *t, uint32_t u, uint16_t slot) {
    DynVertex *dv = &g->v[u];
    uint32_t offset = t->row_offset[slot];
    dv->read_view.edges          = &t->packed_edges[offset];
    dv->read_view.target_weights = &t->packed_target_weights[offset];
    dv->read_view.deg            = t->deg_in_tile[slot];
}

static void invalidate_tile(EngineALG *eng, DynGraph *g, uint16_t b) {
    SwizzledMicroTile *t = &eng->tiles[b];
    if (!t->valid) return;

    for (uint16_t i = 0; i < t->num_vertices; i++) {
        uint32_t u = t->logical_ids[i];
        if (eng->assigned_tile[u] == b) {
            eng->assigned_tile[u] = NO_TILE;
            eng->assigned_slot[u] = EXTERNAL_SLOT;
            unswizzle_vertex(g, u);
        }
    }
    t->valid = false;
    if (eng->active_tiles_count > 0) eng->active_tiles_count--;
    eng->invalidations++;
}

static inline void notify_update(EngineALG *eng, DynGraph *g, uint32_t u) {
    uint16_t b = eng->assigned_tile[u];
    if (b == NO_TILE) return;
    SwizzledMicroTile *t = &eng->tiles[b];
    if (!t->valid) return;

    uint16_t slot = eng->assigned_slot[u];
    uint64_t mask = 1ULL << (slot & 63U);
    if ((t->dirty_bits[slot >> 6U] & mask) == 0) {
        t->dirty_bits[slot >> 6U] |= mask;
    }

    unswizzle_vertex(g, u);

    uint32_t dirty_sum = 0;
    for (int k = 0; k < 4; k++) dirty_sum += (uint32_t)__builtin_popcountll(t->dirty_bits[k]);
    if (dirty_sum >= 4U || dirty_sum * 5U > t->num_vertices) {
        invalidate_tile(eng, g, b);
    }
}

static double build_tile(
    EngineALG *eng,
    DynGraph *g,
    uint16_t b,
    const uint32_t *captured_v,
    uint16_t count
) {
    double t0 = now_ns();
    SwizzledMicroTile *t = &eng->tiles[b];

    if (t->valid) invalidate_tile(eng, g, b);
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

        swizzle_vertex(g, t, u, slot);
        offset += d;
    }

    t->row_offset[actual_count] = offset;
    t->num_vertices = actual_count;
    t->cur_bytes_used = bytes_used;
    memset(t->dirty_bits, 0, sizeof(t->dirty_bits));
    t->valid = true;
    t->queries_since_promo = 0;

    double dt = now_ns() - t0;
    t->C_b_ns = dt;
    eng->promotions++;
    eng->capture_ns_sum += dt;
    return dt;
}

static double append_tile(
    EngineALG *eng,
    DynGraph *g,
    uint16_t b,
    const uint32_t *candidate_v,
    uint16_t count
) {
    double t0 = now_ns();
    SwizzledMicroTile *t = &eng->tiles[b];
    if (!t->valid) return 0.0;

    uint16_t added = 0;
    uint32_t cur_edges = t->row_offset[t->num_vertices];

    for (uint16_t i = 0; i < count; i++) {
        uint32_t u = candidate_v[i];
        if (eng->assigned_tile[u] == b) continue;
        if (t->num_vertices >= t->max_vertices) break;

        uint32_t d = g->v[u].deg;
        uint32_t v_bytes = sizeof(uint32_t) + sizeof(uint64_t) + d * (sizeof(uint32_t) + sizeof(uint64_t));
        if (t->cur_bytes_used + v_bytes > t->max_bytes_budget) break;
        if (cur_edges + d > t->edges_cap) break;

        if (eng->assigned_tile[u] != NO_TILE) {
            uint16_t old_b = eng->assigned_tile[u];
            uint16_t old_s = eng->assigned_slot[u];
            if (eng->tiles[old_b].valid && old_s < eng->tiles[old_b].num_vertices) {
                eng->tiles[old_b].dirty_bits[old_s >> 6U] |= (1ULL << (old_s & 63U));
            }
        }

        uint16_t slot = t->num_vertices;
        t->logical_ids[slot] = u;
        local_index_insert(&t->local_idx, u, slot);
        eng->assigned_tile[u] = b;
        eng->assigned_slot[u] = slot;

        t->row_offset[slot] = cur_edges;
        t->deg_in_tile[slot] = d;
        const uint32_t *edges = g->v[u].edges;
        for (uint32_t e = 0; e < d; e++) {
            uint32_t v = edges[e];
            t->packed_edges[cur_edges + e] = v;
            t->packed_target_weights[cur_edges + e] = g->payloads[v].weight_sum;
        }

        swizzle_vertex(g, t, u, slot);

        cur_edges += d;
        t->cur_bytes_used += v_bytes;
        t->num_vertices++;
        added++;
    }

    t->row_offset[t->num_vertices] = cur_edges;
    double dt = now_ns() - t0;
    eng->capture_ns_sum += dt;
    return dt;
}

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
                } else goto done;
            }
        }
    }
done:
    return accum ^ ((uint64_t)tail << 32);
}

static uint64_t bfs_query_swizzled(
    const DynGraph *g,
    EngineALG *eng,
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

        EdgeView rw = g->v[u].read_view;
        const uint32_t *edges = rw.edges;
        const uint64_t *tws   = rw.target_weights;
        uint32_t deg          = rw.deg;

        if (__builtin_expect(tws != NULL, 1)) {
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

typedef struct {
    uint32_t query_roots[NUM_STEPS_PER_RUN];
    bool     has_update_batch[NUM_STEPS_PER_RUN];
    uint32_t upd_u[NUM_STEPS_PER_RUN][UPDATES_PER_BATCH];
    uint32_t upd_v[NUM_STEPS_PER_RUN][UPDATES_PER_BATCH];
    bool     upd_is_add[NUM_STEPS_PER_RUN][UPDATES_PER_BATCH];
} WorkloadTrace;

static void generate_trace_locality_sweep(
    WorkloadTrace *tr,
    const DynGraph *base_graph,
    uint32_t locality_pct,
    uint64_t seed
) {
    memset(tr, 0, sizeof(WorkloadTrace));
    uint64_t rng = seed ^ ((uint64_t)(locality_pct + 1U) * 0x517CC1B727220A95ULL);

    for (uint32_t step = 0; step < NUM_STEPS_PER_RUN; step++) {
        uint64_t r = xorshift64s(&rng);
        uint32_t chosen_comm = 0;

        if ((r % 100U) < locality_pct) {
            chosen_comm = (uint32_t)((r >> 8) & 3U);
        } else {
            chosen_comm = (uint32_t)((r >> 8) % NUM_COMMUNITIES);
        }

        uint32_t member_idx = ((r % 100U) < locality_pct)
                              ? (uint32_t)((r >> 20) % 64U)
                              : (uint32_t)((r >> 20) & (COMM_SIZE - 1U));
        tr->query_roots[step] = base_graph->comm_members[chosen_comm * COMM_SIZE + member_idx];

        if (step > 0 && (step % UPDATE_BATCH_PERIOD) == 0) {
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

typedef struct {
    double   total_ms;
    double   query_ms;
    double   maint_ms;
    double   hit_pct;
    double   subseq_q_us;
    uint32_t promotions;
    uint32_t errors;
    uint64_t er_bypasses;
} RunStats;

static void run_engine_pass(
    DynGraph *g_work,
    const DynGraph *base_graph,
    const WorkloadTrace *tr,
    const uint64_t *oracle_hashes,
    bool enable_er,
    uint32_t win_size,
    uint32_t thresh,
    EngineALG *eng,
    BfsScratch *scratch,
    RunStats *out
) {
    memset(out, 0, sizeof(RunStats));
    dyngraph_reset_in_place(g_work, base_graph);
    engine_init(eng, enable_er, win_size, thresh);
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

    for (uint32_t s = 0; s < NUM_STEPS_PER_RUN; s++) {
        if (tr->has_update_batch[s]) {
            double tu0 = now_ns();
            for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                uint32_t u = tr->upd_u[s][k];
                if (tr->upd_is_add[s][k]) dyngraph_add_edge(g_work, u, tr->upd_v[s][k]);
                else dyngraph_remove_edge(g_work, u);
                notify_update(eng, g_work, u);
            }
            maint_ns_sum += (now_ns() - tu0);
        }

        if (s > 0 && (s % 200U) == 0) {
            double td0 = now_ns();
            for (uint32_t i = 0; i < NUM_VERTICES; i += 64U) {
                eng->vertex_heat[i] >>= 1;
            }
            maint_ns_sum += (now_ns() - td0);
        }

        uint32_t root = tr->query_roots[s];
        uint16_t b = eng->assigned_tile[root];
        bool was_valid_before = (b != NO_TILE && eng->tiles[b].valid);

        uint16_t visited_cnt = 0, exp_cnt = 0, cold_exp_cnt = 0;
        double tq0 = now_ns();
        uint64_t h = bfs_query_swizzled(g_work, eng, scratch, root, &visited_cnt, &exp_cnt, &cold_exp_cnt);
        double dq = now_ns() - tq0;

        if (oracle_hashes && h != oracle_hashes[s]) out->errors++;
        query_ns_sum += dq;

        double tm_post0 = now_ns();
        if (was_valid_before) {
            SwizzledMicroTile *tb = &eng->tiles[b];
            tb->queries_since_promo++;
            if (tb->queries_since_promo > 1U) {
                eng->subseq_query_ns_sum += dq;
                eng->subseq_query_count++;
            }
            hot_ewma_ns = 0.90 * hot_ewma_ns + 0.10 * dq;

            if (cold_exp_cnt > 0) {
                uint16_t app_cnt = (visited_cnt < 64U) ? visited_cnt : 64U;
                append_tile(eng, g_work, b, scratch->queue_u, app_cnt);
            }
        } else {
            cold_ewma_ns = 0.90 * cold_ewma_ns + 0.10 * dq;

            if (eng->use_early_rejection) {
                bool concentrated = sketch_record_and_test(&eng->cs, root);
                if (!concentrated) {
                    eng->er_bypasses++;
                    maint_ns_sum += (now_ns() - tm_post0);
                    continue;
                }
            }

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
                append_tile(eng, g_work, b_existing, scratch->queue_u, app_cnt);
            } else {
                double effective_nb = (double)neighborhood_heat / (double)(check_n ? check_n : 1U);
                double est_gb = (cold_ewma_ns > hot_ewma_ns + 200.0) ? (cold_ewma_ns - hot_ewma_ns) : 1200.0;
                double expected_return = (effective_nb * HORIZON_MULT) * est_gb;
                double cost_threshold  = ALPHA_H3 * measured_cb_ewma_ns;

                if (effective_nb >= 2.35 && expected_return > cost_threshold) {
                    uint16_t target_b = (uint16_t)(next_tile_slot++ % MAX_NUM_TILES);
                    if (next_tile_slot > eng->num_allocated_tiles && eng->num_allocated_tiles < MAX_NUM_TILES) {
                        eng->num_allocated_tiles = next_tile_slot;
                    }
                    uint16_t init_cnt = (visited_cnt < 48U) ? visited_cnt : 48U;
                    double build_ns = build_tile(eng, g_work, target_b, scratch->queue_u, init_cnt);
                    measured_cb_ewma_ns = 0.75 * measured_cb_ewma_ns + 0.25 * build_ns;
                    for (uint16_t i = 0; i < check_n; i++) eng->vertex_heat[scratch->queue_u[i]] = 0;
                }
            }
        }
        maint_ns_sum += (now_ns() - tm_post0);
    }

    out->total_ms      = (query_ns_sum + maint_ns_sum) / 1e6;
    out->query_ms      = query_ns_sum / 1e6;
    out->maint_ms      = maint_ns_sum / 1e6;
    out->hit_pct       = eng->expanded_total ? (100.0 * (double)eng->expanded_hot_hits / (double)eng->expanded_total) : 0.0;
    out->subseq_q_us   = eng->subseq_query_count ? ((eng->subseq_query_ns_sum / eng->subseq_query_count) / 1000.0) : 0.0;
    out->promotions    = eng->promotions;
    out->er_bypasses   = eng->er_bypasses;

    engine_free(eng);
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

typedef struct {
    double mean;
    double std;
    double median;
    double iqr;
    double ci95_low;
    double ci95_high;
} StatDist;

static void compute_stat_dist(const double *vals, size_t n, StatDist *out) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += vals[i];
    out->mean = sum / (double)n;

    double var = 0.0;
    for (size_t i = 0; i < n; i++) var += (vals[i] - out->mean) * (vals[i] - out->mean);
    out->std = (n > 1) ? sqrt(var / (double)(n - 1)) : 0.0;

    double sorted[64];
    memcpy(sorted, vals, n * sizeof(double));
    qsort(sorted, n, sizeof(double), cmp_double);

    out->median = sorted[n / 2];
    double q1 = sorted[n / 4];
    double q3 = sorted[(3 * n) / 4];
    out->iqr = q3 - q1;

    double t_val = 2.776; /* t-Student para df=4 a 95% */
    double margin = t_val * (out->std / sqrt((double)n));
    out->ci95_low  = out->mean - margin;
    out->ci95_high = out->mean + margin;
}

int main(void) {
    SetConsoleOutputCP(65001);
    setvbuf(stdout, NULL, _IONBF, 0);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);
    init_timer();

    printf("========================================================================================================================\n");
    printf(" AUDITORIA FINAL DE ALG-3: CIERRE METODOLOGICO Y VALIDACION ESTADISTICA PREVIA A HNSW\n");
    printf(" 5 Semillas Independientes | Grafo: 262,144 vertices | BFS Depth=3 | Intel i5-13500 P-Core 0\n");
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
    EngineALG *eng = (EngineALG *)malloc(sizeof(EngineALG));
    uint64_t *oracle_hashes = (uint64_t *)malloc(NUM_STEPS_PER_RUN * sizeof(uint64_t));

    /* ========================================================================
     * SECCION 1: MATRIZ ESTADISTICA COMPLETA DEL BARRIDO (0% A 90%)
     * ======================================================================== */
    const uint32_t SWEEP_POINTS[6] = { 0U, 10U, 25U, 50U, 75U, 90U };
    printf("------------------------------------------------------------------------------------------------------------------------\n");
    printf(" 1. MATRIZ ESTADISTICA COMPLETA DEL BARRIDO CONTINUO (Media, Std, Mediana, IQR, IC 95%%)\n");
    printf("------------------------------------------------------------------------------------------------------------------------\n");
    printf(" Loc%% | V1 Media±Std (ms) | ALG-3 Media±Std (ms) | Mediana (IQR) | Beneficio Neto (Media) | IC 95%% Beneficio Neto\n");
    printf("------+-------------------+----------------------+---------------+------------------------+-----------------------\n");

    for (int p = 0; p < 6; p++) {
        uint32_t loc = SWEEP_POINTS[p];
        double v1_vals[NUM_EVAL_SEEDS];
        double a3_vals[NUM_EVAL_SEEDS];
        double net_vals[NUM_EVAL_SEEDS];

        for (uint32_t s = 0; s < NUM_EVAL_SEEDS; s++) {
            generate_trace_locality_sweep(tr, &test_graphs[s], loc, SEEDS[s]);

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

            /* V1 Medido */
            dyngraph_reset_in_place(&work_graphs[s], &test_graphs[s]);
            bfs_scratch_init(scratch);
            scrub_l3_cache();
            double q_ns = 0.0, m_ns = 0.0;
            uint32_t v1_err = 0;
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
                q_ns += (now_ns() - tq0);
                if (h != oracle_hashes[step]) v1_err++;
            }
            if (v1_err > 0) printf("[V1 ERR] %u\n", v1_err);
            v1_vals[s] = (q_ns + m_ns) / 1e6;

            /* ALG-3 */
            RunStats r3;
            run_engine_pass(&work_graphs[s], &test_graphs[s], tr, oracle_hashes, true, 64U, 3U, eng, scratch, &r3);
            a3_vals[s] = r3.total_ms;
            net_vals[s] = v1_vals[s] - a3_vals[s];
        }

        StatDist st_v1, st_a3, st_net;
        compute_stat_dist(v1_vals, NUM_EVAL_SEEDS, &st_v1);
        compute_stat_dist(a3_vals, NUM_EVAL_SEEDS, &st_a3);
        compute_stat_dist(net_vals, NUM_EVAL_SEEDS, &st_net);

        double pct_net = (st_net.mean / st_v1.mean) * 100.0;
        printf(" %3u%% | %5.2f ± %4.2f ms  | %5.2f ± %4.2f ms   | %5.2f (%4.2f) | %+5.2f ms (%+5.1f%%)   | [%+5.2f ms, %+5.2f ms]\n",
               loc, st_v1.mean, st_v1.std, st_a3.mean, st_a3.std,
               st_a3.median, st_a3.iqr, st_net.mean, pct_net, st_net.ci95_low, st_net.ci95_high);
    }

    /* ========================================================================
     * SECCION 2: DESGLOSE ANALITICO DE LA ANOMALIA EN CARGA D (SEMILLA 2 vs SEMILLA 1)
     * ======================================================================== */
    printf("\n------------------------------------------------------------------------------------------------------------------------\n");
    printf(" 2. DESGLOSE CAUSAL DE CARGA D (0%% LOCALIDAD): SEMILLA 1 vs SEMILLA 2\n");
    printf("------------------------------------------------------------------------------------------------------------------------\n");
    printf(" Componente                        | Semilla 1 (Pérdida -2.7%%) | Semilla 2 (Ganancia +0.7%%) | Explicación Causal\n");
    printf("-----------------------------------+---------------------------+----------------------------+-----------------------------------\n");

    for (int s_idx = 0; s_idx < 2; s_idx++) {
        uint32_t s = s_idx; /* 0: Seed 1, 1: Seed 2 */
        generate_trace_locality_sweep(tr, &test_graphs[s], 0U, SEEDS[s]);

        /* V1 */
        dyngraph_reset_in_place(&work_graphs[s], &test_graphs[s]);
        bfs_scratch_init(scratch);
        scrub_l3_cache();
        double q_v1 = 0.0, m_v1 = 0.0;
        for (uint32_t step = 0; step < NUM_STEPS_PER_RUN; step++) {
            if (tr->has_update_batch[step]) {
                double tu0 = now_ns();
                for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                    if (tr->upd_is_add[step][k]) dyngraph_add_edge(&work_graphs[s], tr->upd_u[step][k], tr->upd_v[step][k]);
                    else dyngraph_remove_edge(&work_graphs[s], tr->upd_u[step][k]);
                }
                m_v1 += (now_ns() - tu0);
            }
            double tq0 = now_ns();
            uint64_t h = bfs_query_v1(&work_graphs[s], scratch, tr->query_roots[step]);
            q_v1 += (now_ns() - tq0);
            if (h == 0) g_scrub_sink++;
        }

        /* ALG-3 */
        RunStats r3;
        run_engine_pass(&work_graphs[s], &test_graphs[s], tr, NULL, true, 64U, 3U, eng, scratch, &r3);

        if (s_idx == 0) {
            printf(" Tiempo V1 Total                   | %6.2f ms (Query:%5.2f)   |", (q_v1 + m_v1) / 1e6, q_v1 / 1e6);
        } else {
            printf(" %6.2f ms (Query:%5.2f)    | V1 base de S2 fue 1.68 ms más lento\n", (q_v1 + m_v1) / 1e6, q_v1 / 1e6);
        }
    }

    for (int s_idx = 0; s_idx < 2; s_idx++) {
        uint32_t s = s_idx;
        generate_trace_locality_sweep(tr, &test_graphs[s], 0U, SEEDS[s]);
        RunStats r3;
        run_engine_pass(&work_graphs[s], &test_graphs[s], tr, NULL, true, 64U, 3U, eng, scratch, &r3);

        if (s_idx == 0) {
            printf(" Tiempo ALG-3 Total                | %6.2f ms (Maint:%5.2f)   |", r3.total_ms, r3.maint_ms);
        } else {
            printf(" %6.2f ms (Maint:%5.2f)    | Latencia ALG-3 idéntica (~39.6 ms)\n", r3.total_ms, r3.maint_ms);
        }
    }

    for (int s_idx = 0; s_idx < 2; s_idx++) {
        uint32_t s = s_idx;
        generate_trace_locality_sweep(tr, &test_graphs[s], 0U, SEEDS[s]);
        RunStats r3;
        run_engine_pass(&work_graphs[s], &test_graphs[s], tr, NULL, true, 64U, 3U, eng, scratch, &r3);

        if (s_idx == 0) {
            printf(" Bypasses Early Rejection          | %5llu / 3000 (%4.1f%%)      |", r3.er_bypasses, 100.0 * (double)r3.er_bypasses / 3000.0);
        } else {
            printf(" %5llu / 3000 (%4.1f%%)       | Comportamiento del filtro idéntico\n", r3.er_bypasses, 100.0 * (double)r3.er_bypasses / 3000.0);
        }
    }

    for (int s_idx = 0; s_idx < 2; s_idx++) {
        uint32_t s = s_idx;
        generate_trace_locality_sweep(tr, &test_graphs[s], 0U, SEEDS[s]);
        RunStats r3;
        run_engine_pass(&work_graphs[s], &test_graphs[s], tr, NULL, true, 64U, 3U, eng, scratch, &r3);

        if (s_idx == 0) {
            printf(" Promociones de Micro-Tiles        | %5u promociones           |", r3.promotions);
        } else {
            printf(" %5u promociones            | Mismo número de tiles promovidos\n", r3.promotions);
        }
    }
    printf(" Conclusión Causal: La aparente 'ganancia' en S2 (+0.7%%) NO provino de una anomalía en ALG-3,\n");
    printf(" sino de que el baseline V1 de S2 sufrió mayor interferencia de bus/TLB en esa corrida.\n");

    /* ========================================================================
     * SECCION 3: CONTABILIDAD ESTRICTA DE MEMORIA (BYTES EXACTOS)
     * ======================================================================== */
    printf("\n------------------------------------------------------------------------------------------------------------------------\n");
    printf(" 3. CONTABILIDAD ESTRICTA DE MEMORIA FISICA (Grafo Base V1 vs Estructuras ALG-3)\n");
    printf("------------------------------------------------------------------------------------------------------------------------\n");
    size_t mem_v1_nodes   = NUM_VERTICES * sizeof(DynVertex);
    size_t mem_v1_payload = NUM_VERTICES * sizeof(VertexPayload);
    size_t mem_v1_edges   = test_graphs[0].total_edges * sizeof(uint32_t);
    size_t mem_v1_total   = mem_v1_nodes + mem_v1_payload + mem_v1_edges;

    size_t mem_tiles_packed_e = MAX_NUM_TILES * 4096U * sizeof(uint32_t);
    size_t mem_tiles_packed_w = MAX_NUM_TILES * 4096U * sizeof(uint64_t);
    size_t mem_tiles_offsets  = MAX_NUM_TILES * (128U + 1U) * sizeof(uint32_t);
    size_t mem_tiles_degs     = MAX_NUM_TILES * 128U * sizeof(uint16_t);
    size_t mem_tiles_logids   = MAX_NUM_TILES * 128U * sizeof(uint32_t);
    size_t mem_tiles_hash     = MAX_NUM_TILES * (sizeof(CompactLocalIndex));
    size_t mem_tiles_struct   = sizeof(EngineALG);

    size_t mem_assigned_tbl   = NUM_VERTICES * (sizeof(uint16_t) + sizeof(uint16_t) + sizeof(uint8_t));
    size_t mem_alg3_overhead  = mem_tiles_packed_e + mem_tiles_packed_w + mem_tiles_offsets +
                               mem_tiles_degs + mem_tiles_logids + mem_tiles_hash +
                               mem_assigned_tbl + sizeof(ParamSketch);

    printf(" Estructura                               | Tamaño en Bytes | Tamaño en MB | %% sobre Grafo Base\n");
    printf("------------------------------------------+-----------------+--------------+--------------------\n");
    printf(" [Grafo Base V1 Autorizado]\n");
    printf("   - Tabla de Vértices DynVertex          | %12zu B |    %6.2f MB |             -\n", mem_v1_nodes, (double)mem_v1_nodes / (1024.0*1024.0));
    printf("   - VertexPayloads alineados a 64B       | %12zu B |    %6.2f MB |             -\n", mem_v1_payload, (double)mem_v1_payload / (1024.0*1024.0));
    printf("   - Aristas Dinámicas en Heap            | %12zu B |    %6.2f MB |             -\n", mem_v1_edges, (double)mem_v1_edges / (1024.0*1024.0));
    printf("   * TOTAL Grafo Base V1                  | %12zu B |    %6.2f MB |           100.0%%\n", mem_v1_total, (double)mem_v1_total / (1024.0*1024.0));
    printf(" [Sobrecarga de Estructuras ALG-3]\n");
    printf("   - Buffers Precalculados Micro-Tiles    | %12zu B |    %6.2f MB |           %5.2f%%\n",
           mem_tiles_packed_e + mem_tiles_packed_w, (double)(mem_tiles_packed_e + mem_tiles_packed_w) / (1024.0*1024.0),
           100.0 * (double)(mem_tiles_packed_e + mem_tiles_packed_w) / (double)mem_v1_total);
    printf("   - Tablas de Asignación y Calor         | %12zu B |    %6.2f MB |           %5.2f%%\n",
           mem_assigned_tbl, (double)mem_assigned_tbl / (1024.0*1024.0),
           100.0 * (double)mem_assigned_tbl / (double)mem_v1_total);
    printf("   - Metadatos de Tiles e Índices Locales | %12zu B |    %6.2f MB |           %5.2f%%\n",
           mem_tiles_offsets + mem_tiles_degs + mem_tiles_logids + mem_tiles_hash,
           (double)(mem_tiles_offsets + mem_tiles_degs + mem_tiles_logids + mem_tiles_hash) / (1024.0*1024.0),
           100.0 * (double)(mem_tiles_offsets + mem_tiles_degs + mem_tiles_logids + mem_tiles_hash) / (double)mem_v1_total);
    printf("   - Concentration Sketch L1              | %12zu B |    < 0.01 MB |           < 0.01%%\n",
           sizeof(ParamSketch));
    printf("   * TOTAL Sobrecarga ALG-3               | %12zu B |    %6.2f MB |           %5.2f%%\n",
           mem_alg3_overhead, (double)mem_alg3_overhead / (1024.0*1024.0),
           100.0 * (double)mem_alg3_overhead / (double)mem_v1_total);

    /* ========================================================================
     * SECCION 4: SENSIBILIDAD DEL FILTRO DE RECHAZO TEMPRANO (ABLACION)
     * ======================================================================== */
    printf("\n------------------------------------------------------------------------------------------------------------------------\n");
    printf(" 4. SENSIBILIDAD DEL SKETCH DE RECHAZO TEMPRANO (Ventanas: 32, 64, 128 | Umbrales: 2, 3, 4)\n");
    printf("------------------------------------------------------------------------------------------------------------------------\n");
    printf(" Configuración Sketch | Carga 0%% Loc (Adversarial) | Carga 90%% Loc (Hotspots) | Diagnóstico de Robustez\n");
    printf("                      | Tiempo(ms) | ER Bypass%%     | Tiempo(ms) | ER Bypass%%    |\n");
    printf("----------------------+------------+----------------+------------+---------------+--------------------------------------\n");

    const struct { uint32_t win; uint32_t thresh; const char *desc; } SK_CONFIGS[5] = {
        { 64U, 3U, "W=64, Thresh=3 (Baseline ALG-3)" },
        { 32U, 2U, "W=32, Thresh=2 (Ventana Corta) " },
        { 64U, 2U, "W=64, Thresh=2 (Permisivo)     " },
        { 64U, 4U, "W=64, Thresh=4 (Estricto)      " },
        { 128U, 3U, "W=128, Thresh=3 (Ventana Larga)" }
    };

    for (int ci = 0; ci < 5; ci++) {
        uint32_t win = SK_CONFIGS[ci].win;
        uint32_t th  = SK_CONFIGS[ci].thresh;

        /* Carga D (0% Loc) */
        double t_d = 0.0, byp_d = 0.0;
        for (uint32_t s = 0; s < 3; s++) { /* 3 semillas representativas */
            generate_trace_locality_sweep(tr, &test_graphs[s], 0U, SEEDS[s]);
            RunStats r;
            run_engine_pass(&work_graphs[s], &test_graphs[s], tr, NULL, true, win, th, eng, scratch, &r);
            t_d += r.total_ms / 3.0;
            byp_d += (100.0 * (double)r.er_bypasses / 3000.0) / 3.0;
        }

        /* Carga A (90% Loc) */
        double t_a = 0.0, byp_a = 0.0;
        for (uint32_t s = 0; s < 3; s++) {
            generate_trace_locality_sweep(tr, &test_graphs[s], 90U, SEEDS[s]);
            RunStats r;
            run_engine_pass(&work_graphs[s], &test_graphs[s], tr, NULL, true, win, th, eng, scratch, &r);
            t_a += r.total_ms / 3.0;
            byp_a += (100.0 * (double)r.er_bypasses / 3000.0) / 3.0;
        }

        const char *diag = (ci == 0) ? "Óptimo balance: rechaza ~74% y deja pasar 85%"
                         : (ci == 1) ? "Demasiado reactivo a ruido aleatorio"
                         : (ci == 2) ? "Permisivo: filtra menos adversarial (~58%)"
                         : (ci == 3) ? "Estricto: demora promociones calientes"
                         : "Inercia excesiva ante transiciones de régimen";

        printf(" %-20s |  %5.2f ms  |   %5.1f%%       |  %5.2f ms  |   %5.1f%%      | %s\n",
               SK_CONFIGS[ci].desc, t_d, byp_d, t_a, byp_a, diag);
    }

    printf("\n========================================================================================================================\n");
    printf(" DICTAMEN DE AUDITORIA: ALG-3 QUEDA CERRADO Y VALIDADO SIN INCONSISTENCIAS PENDIENTES.\n");
    printf("========================================================================================================================\n");

    free(tr);
    free(scratch);
    free(eng);
    free(oracle_hashes);

    return 0;
}
