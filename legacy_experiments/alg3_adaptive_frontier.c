/*
 * ============================================================================
 * EXPERIMENTO ALG-3 (DEFINITIVO): EARLY REJECTION & CURVA CONTINUA DE LOCALIDAD
 * ============================================================================
 * Incorpora:
 *   1. El motor ALG-2 Pointer-Swizzled exacto y verificado.
 *   2. Decaimiento periódico de calor cada 200 pasos (`vertex_heat[i] >>= 1`).
 *   3. Detector de Early Rejection en 2 Niveles:
 *      - Nivel 1: Sketch de concentración en ventana corta. Si el subespacio
 *        no muestra colisiones en 64 consultas, se omite el recorrido de vecindad
 *        y la evaluación H3.
 *      - Nivel 2: Evaluación H3 con la regla matemática exacta de ALG-2.
 *   4. Generación continua de localidad (0%, 10%, 25%, 50%, 75%, 90%):
 *      Hotspots concentrados en 4 comunidades con probabilidad Loc%,
 *      ruido uniforme en 256 comunidades con probabilidad (100 - Loc)%.
 *   5. Análisis de dispersión estadística completa en 5 semillas:
 *      Media, Desviación Estándar, Mediana, IQR, IC 95% y NetBenefit real.
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
    uint32_t        deg;
} EdgeView;

typedef struct {
    uint32_t *edges;
    uint32_t  deg;
    uint32_t  cap;
    uint32_t  version;
    EdgeView  read_view;
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
    for (uint32_t u = 0; u < NUM_VERTICES; u++) free(g->v[u].edges);
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

/*
 * Sketch circular de concentración (64 bins en L1):
 * Agrupa raíces en 64 buckets. Si una región concentra 3 o más consultas
 * en la ventana de 64 pasos, se autoriza la inspección de vecindad H3.
 */
typedef struct {
    uint8_t  window_buckets[64];
    uint8_t  ring[64];
    uint32_t ring_idx;
    uint32_t count_in_window;
} ConcentrationSketch;

static inline void conc_sketch_init(ConcentrationSketch *cs) {
    memset(cs, 0, sizeof(ConcentrationSketch));
}

static inline bool conc_sketch_record_and_test(ConcentrationSketch *cs, uint32_t root) {
    uint8_t b = (uint8_t)((root * 0x9E3779B9U) >> 26U);

    if (cs->count_in_window >= 64U) {
        uint8_t old_b = cs->ring[cs->ring_idx];
        if (cs->window_buckets[old_b] > 0) cs->window_buckets[old_b]--;
    } else {
        cs->count_in_window++;
    }

    cs->ring[cs->ring_idx] = b;
    cs->ring_idx = (cs->ring_idx + 1U) & 63U;
    cs->window_buckets[b]++;

    return (cs->window_buckets[b] >= 3U);
}

typedef struct {
    uint32_t key_plus_one[MAX_HASH_CAP];
    uint16_t slot[MAX_HASH_CAP];
    uint32_t mask;
    uint32_t shift;
} CompactLocalIndex;

static inline void local_index_clear(CompactLocalIndex *idx, uint32_t max_vertices) {
    uint32_t cap = 128U, log2_cap = 7U;
    while (cap < max_vertices * 2U) { cap <<= 1U; log2_cap++; }
    idx->mask  = cap - 1U;
    idx->shift = 32U - log2_cap;
    memset(idx->key_plus_one, 0, cap * sizeof(uint32_t));
}

static inline void local_index_insert(CompactLocalIndex *idx, uint32_t global_id, uint16_t local_slot) {
    uint32_t pos = (global_id * 0x9E3779B9U) >> idx->shift;
    uint32_t kp1 = global_id + 1U;
    while (idx->key_plus_one[pos] != 0 && idx->key_plus_one[pos] != kp1) pos = (pos + 1U) & idx->mask;
    idx->key_plus_one[pos] = kp1;
    idx->slot[pos] = local_slot;
}

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

    uint32_t         *packed_edges;
    uint64_t         *packed_target_weights;
    uint32_t          edges_cap;

    double            C_b_ns;
    double            g_b_ns;
    uint32_t          queries_since_promo;
    double            realized_savings_ns;
} SwizzledMicroTile;

typedef struct {
    SwizzledMicroTile   tiles[MAX_NUM_TILES];
    uint32_t            num_allocated_tiles;
    uint32_t            active_tiles_count;
    uint16_t            assigned_tile[NUM_VERTICES];
    uint16_t            assigned_slot[NUM_VERTICES];
    uint8_t             vertex_heat[NUM_VERTICES];

    ConcentrationSketch cs;
    bool                use_early_rejection;

    uint64_t            er_bypasses;
    uint32_t            promotions;
    uint32_t            invalidations;
    uint64_t            expanded_hot_hits;
    uint64_t            expanded_total;
    uint64_t            total_bytes_copied;
    double              capture_ns_sum;
    double              subseq_query_ns_sum;
    uint32_t            subseq_query_count;
    size_t              bytes_allocated;
} EngineALG;

static void engine_init(EngineALG *eng, bool use_er) {
    memset(eng, 0, sizeof(EngineALG));
    eng->use_early_rejection = use_er;
    conc_sketch_init(&eng->cs);
    memset(eng->assigned_tile, 0xFF, sizeof(eng->assigned_tile));
    memset(eng->assigned_slot, 0xFF, sizeof(eng->assigned_slot));
    eng->bytes_allocated = sizeof(EngineALG);

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

static void engine_free(EngineALG *eng) {
    for (uint32_t b = 0; b < MAX_NUM_TILES; b++) {
        free(eng->tiles[b].packed_edges);
        free(eng->tiles[b].packed_target_weights);
    }
}

static inline void unswizzle_vertex(DynGraph *g, uint32_t u) {
    DynVertex *dv = &g->v[u];
    dv->read_view.edges = dv->edges;
    dv->read_view.target_weights = NULL;
    dv->read_view.deg = dv->deg;
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
    eng->total_bytes_copied += bytes_used;
    return dt;
}

static double append_tile(
    EngineALG *eng,
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

        swizzle_vertex(g, t, u, slot);
        offset += d;
        t->row_offset[t->num_vertices] = offset;
        bytes_added += v_bytes;
        v_added++;
    }

    if (v_added == 0) return 0.0;

    t->cur_bytes_used += bytes_added;
    double dt = now_ns() - t0;
    eng->capture_ns_sum += dt;
    eng->total_bytes_copied += bytes_added;
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
    uint64_t rng = seed ^ ((uint64_t)(locality_pct + 1) * 0x517CC1B727220A95ULL);

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
    EngineALG *eng,
    BfsScratch *scratch,
    RunStats *out
) {
    memset(out, 0, sizeof(RunStats));
    dyngraph_reset_in_place(g_work, base_graph);
    engine_init(eng, enable_er);
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

        /* Decaimiento periódico de calor cada 200 pasos */
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

            /* Early Rejection Nivel 1: Test rápido de concentración en sketch circular */
            if (eng->use_early_rejection) {
                bool concentrated = conc_sketch_record_and_test(&eng->cs, root);
                if (!concentrated) {
                    eng->er_bypasses++;
                    maint_ns_sum += (now_ns() - tm_post0);
                    continue; /* Salto de coste cero */
                }
            }

            /* Nivel 2: H3 Adaptativo Completo */
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

int main(void) {
    SetConsoleOutputCP(65001);
    setvbuf(stdout, NULL, _IONBF, 0);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);
    init_timer();

    printf("========================================================================================================================\n");
    printf(" EXPERIMENTO ALG-3: EARLY REJECTION TOPOLOGICO Y BARRIDO CONTINUO DE LOCALIDAD\n");
    printf(" Evaluacion en 5 Semillas Independientes | Grafo: 262,144 vertices | BFS Depth=3 | Intel i5-13500 P-Core 0\n");
    printf("========================================================================================================================\n\n");

    const uint64_t SEEDS[NUM_EVAL_SEEDS] = {
        0x778899AABBCCDDEEULL,
        0x3141592653589793ULL,
        0x2718281828459045ULL,
        0xFEEDFACECAFEF00DULL,
        0x0123456789ABCDEFULL
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
     * PARTE 1: BARRIDO CONTINUO DE INTENSIDAD DE LOCALIDAD (0% A 90%)
     * ======================================================================== */
    const uint32_t SWEEP_POINTS[6] = { 0U, 10U, 25U, 50U, 75U, 90U };
    printf("------------------------------------------------------------------------------------------------------------------------\n");
    printf(" PARTE 1: BARRIDO CONTINUO DE LOCALIDAD (Curva de Rentabilidad: V1 Base vs ALG-2 vs ALG-3 Early Rejection)\n");
    printf("------------------------------------------------------------------------------------------------------------------------\n");
    printf(" Loc%% | V1 Base (ms) | ALG-2 (ms) | ALG-3 ER (ms) | Hit%% (A3) | ER Byp%% | NetBenefit (ALG-2) | NetBenefit (ALG-3)\n");
    printf("------+--------------+------------+---------------+-----------+---------+--------------------+--------------------\n");

    for (int p = 0; p < 6; p++) {
        uint32_t loc = SWEEP_POINTS[p];
        double v1_sum = 0.0, a2_sum = 0.0, a3_sum = 0.0;
        double a3_hit_sum = 0.0;
        uint64_t er_byp_sum = 0;

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

            /* V1 */
            {
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
                v1_sum += (q_ns + m_ns) / 1e6;
            }

            /* ALG-2 (Sin Early Rejection) */
            {
                RunStats r;
                run_engine_pass(&work_graphs[s], &test_graphs[s], tr, oracle_hashes,
                                false, eng, scratch, &r);
                a2_sum += r.total_ms;
            }

            /* ALG-3 (Con Early Rejection Topológico) */
            {
                RunStats r;
                run_engine_pass(&work_graphs[s], &test_graphs[s], tr, oracle_hashes,
                                true, eng, scratch, &r);
                a3_sum += r.total_ms;
                a3_hit_sum += r.hit_pct;
                er_byp_sum += r.er_bypasses;
            }
        }

        double v1_m  = v1_sum / NUM_EVAL_SEEDS;
        double a2_m  = a2_sum / NUM_EVAL_SEEDS;
        double a3_m  = a3_sum / NUM_EVAL_SEEDS;
        double a3_h  = a3_hit_sum / NUM_EVAL_SEEDS;
        double byp_pct = 100.0 * (double)(er_byp_sum / NUM_EVAL_SEEDS) / (double)NUM_STEPS_PER_RUN;

        double net_a2 = v1_m - a2_m;
        double net_a3 = v1_m - a3_m;

        printf("  %2u%% | %12.2f | %10.2f | %13.2f | %8.1f%% | %6.1f%% | %+.2f ms (%+.1f%%) | %+.2f ms (%+.1f%%)\n",
               loc, v1_m, a2_m, a3_m, a3_h, byp_pct,
               net_a2, 100.0 * net_a2 / v1_m,
               net_a3, 100.0 * net_a3 / v1_m);
    }

    /* ========================================================================
     * PARTE 2: RIGOR ESTADÍSTICO DETALLADO EN CARGAS A Y D (5 SEMILLAS)
     * ======================================================================== */
    printf("\n========================================================================================================================\n");
    printf(" PARTE 2: DISPERSION Y ESTADISTICA DETALLADA (MEDIA, STD, MEDIANA, IQR, IC 95%%)\n");
    printf("========================================================================================================================\n");

    const uint32_t FOCUS_LOC[2] = { 92U, 0U };
    const char *FOCUS_NAMES[2] = { "Carga A (Hotspots Estables - 92% Loc)", "Carga D (Adversarial Uniforme - 0% Loc)" };

    for (int wi = 0; wi < 2; wi++) {
        uint32_t loc_pct = FOCUS_LOC[wi];
        printf(" [%s - Detalle por Semilla]\n", FOCUS_NAMES[wi]);
        printf(" Semilla | V1 Base (ms) | ALG-2 (ms) | ALG-3 ER (ms) | NetBenefit A2 | NetBenefit A3 | ER Bypass%%\n");
        printf("---------+--------------+------------+---------------+---------------+---------------+------------\n");

        double v1_vals[NUM_EVAL_SEEDS], a2_vals[NUM_EVAL_SEEDS], a3_vals[NUM_EVAL_SEEDS];
        double net_a2_vals[NUM_EVAL_SEEDS], net_a3_vals[NUM_EVAL_SEEDS];

        for (uint32_t s = 0; s < NUM_EVAL_SEEDS; s++) {
            generate_trace_locality_sweep(tr, &test_graphs[s], loc_pct, SEEDS[s]);

            /* Oráculo */
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

            /* V1 */
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

            /* ALG-2 */
            RunStats r2;
            run_engine_pass(&work_graphs[s], &test_graphs[s], tr, oracle_hashes, false, eng, scratch, &r2);
            a2_vals[s] = r2.total_ms;
            net_a2_vals[s] = v1_vals[s] - a2_vals[s];

            /* ALG-3 */
            RunStats r3;
            run_engine_pass(&work_graphs[s], &test_graphs[s], tr, oracle_hashes, true, eng, scratch, &r3);
            a3_vals[s] = r3.total_ms;
            net_a3_vals[s] = v1_vals[s] - a3_vals[s];

            printf("  Seed %u | %12.2f | %10.2f | %13.2f | %+6.2f (%+4.1f%%) | %+6.2f (%+4.1f%%) | %4.1f%%\n",
                   s + 1, v1_vals[s], a2_vals[s], a3_vals[s],
                   net_a2_vals[s], 100.0 * net_a2_vals[s] / v1_vals[s],
                   net_a3_vals[s], 100.0 * net_a3_vals[s] / v1_vals[s],
                   100.0 * (double)r3.er_bypasses / (double)NUM_STEPS_PER_RUN);
        }

        double sum_a3 = 0.0, sum_sq_a3 = 0.0, sum_net3 = 0.0;
        for (int i = 0; i < NUM_EVAL_SEEDS; i++) {
            sum_a3 += a3_vals[i];
            sum_net3 += net_a3_vals[i];
        }
        double mean_a3 = sum_a3 / NUM_EVAL_SEEDS;
        double mean_net3 = sum_net3 / NUM_EVAL_SEEDS;

        for (int i = 0; i < NUM_EVAL_SEEDS; i++) {
            double diff = a3_vals[i] - mean_a3;
            sum_sq_a3 += diff * diff;
        }
        double std_a3 = sqrt(sum_sq_a3 / (NUM_EVAL_SEEDS - 1));

        double sorted_a3[NUM_EVAL_SEEDS];
        memcpy(sorted_a3, a3_vals, sizeof(sorted_a3));
        qsort(sorted_a3, NUM_EVAL_SEEDS, sizeof(double), cmp_double);
        double median_a3 = sorted_a3[2];
        double iqr_a3 = sorted_a3[3] - sorted_a3[1];
        double sem = std_a3 / sqrt(NUM_EVAL_SEEDS);
        double ci95 = 2.776 * sem;

        printf("  --------------------------------------------------------------------------------------------\n");
        printf("  * ALG-3 Resumen Estadistico: Media=%.2f ms (Std=%.2f ms) | Mediana=%.2f ms (IQR=%.2f ms)\n",
               mean_a3, std_a3, median_a3, iqr_a3);
        printf("  * Beneficio Neto ALG-3: %+.2f ms | IC 95%%: [%.2f ms, %.2f ms]\n\n",
               mean_net3, mean_a3 - ci95, mean_a3 + ci95);
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
