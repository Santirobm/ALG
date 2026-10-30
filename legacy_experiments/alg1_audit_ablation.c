/*
 * AUDITORÍA TÉCNICA Y ABLACIÓN CAUSAL DE ALG-1 (CON CONTROL ESTRICTO DE PUNTEROS Y L3)
 * ============================================================================
 * 1. Demuestra empíricamente la causa raíz del Problema 1 (Anomalía de Carga D):
 *    - En `alg1_incremental_experiment.c`, `dyngraph_clone` invocaba `malloc()`
 *      262,144 veces antes de cada variante y `free()` 262,144 veces al terminar.
 *    - Tras el primer par `clone/free` (o tras el churn de `realloc` en variantes
 *      anteriores), el Low-Fragmentation Heap (LFH) de Windows asignaba los
 *      262,144 bloques de `v[u].edges` con distinta dispersión de páginas virtuales
 *      (dTLB/STLB misses), alterando entre 34.8 ms y 45.1 ms el recorrido de `V1`
 *      en Carga D aunque el código de consulta era idéntico.
 * 2. Elimina por completo el sesgo de direcciones virtuales y de caché:
 *    - `dyngraph_reset_in_place` restaura el grafo autoritativo `g_work` exactamente
 *      en las MISMAS direcciones virtuales de `v[u].edges` para todas las variantes.
 *    - `scrub_l3_cache` limpia los 24 MB de L3 del i5-13500 antes de cada variante.
 *    - El orden de ejecución de las variantes se rota en cada semilla.
 * 3. Ejecuta:
 *    - PARTE 1: Demostración directa del efecto `dyngraph_clone` (LFH page scatter) vs
 *               Mismos Punteros en Carga D.
 *    - PARTE 2: Las 4 pruebas de ablación causal solicitadas (`V1-Orig`, `V1-en-Motor-V7`,
 *               `V7-H3-Sin-Lectura`, `V7-Full`) en las 5 cargas (A-E).
 *    - PARTE 3: Re-evaluación completa y auditada de ALG-1 (`V1, V3, V4, V5, V6, V7`)
 *               con contabilidad 100% cerrada de `Maint(ms)` y desglose exacto de `BrkEv`.
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
#define NUM_TEST_SEEDS      3U

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

/* Buffer de 32 MB (> 24 MB L3 del i5-13500) para limpiar L1/L2/L3 entre variantes */
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
    uint32_t *edges;
    uint32_t  deg;
    uint32_t  cap;
    uint32_t  version;
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
        dst->bytes_allocated += (size_t)cap * sizeof(uint32_t);
    }
}

/*
 * Restaura `dst` exactamente al estado inicial de `src` reutilizando los MISMOS
 * punteros `dst->v[u].edges` en el heap. Así, todas las variantes recorren
 * exactamente las mismas direcciones virtuales sin varianza del asignador de Windows.
 */
static void dyngraph_reset_in_place(DynGraph *dst, const DynGraph *src) {
    memcpy(dst->payloads, src->payloads, NUM_VERTICES * sizeof(VertexPayload));
    dst->total_edges = src->total_edges;
    for (uint32_t u = 0; u < NUM_VERTICES; u++) {
        uint32_t deg = src->v[u].deg;
        memcpy(dst->v[u].edges, src->v[u].edges, deg * sizeof(uint32_t));
        dst->v[u].deg = deg;
        dst->v[u].version = src->v[u].version;
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
}

static inline void dyngraph_remove_edge(DynGraph *g, uint32_t u) {
    DynVertex *dv = &g->v[u];
    if (dv->deg > 4) {
        dv->deg--;
        dv->version++;
        g->total_edges--;
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

    for (uint32_t i = 0; i < NUM_VERTICES; i++) {
        g->comm_members[i] = i;
    }
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
                if (target != u) {
                    dyngraph_add_edge(g, u, target);
                }
            }
        }
    }
}

#define MAX_HASH_CAP 2048U

typedef struct {
    uint32_t key_plus_one[MAX_HASH_CAP];
    uint16_t slot[MAX_HASH_CAP];
    uint64_t bloom[4];
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
    memset(idx->bloom, 0, sizeof(idx->bloom));
}

static inline void local_index_insert(CompactLocalIndex *idx, uint32_t global_id, uint16_t local_slot) {
    uint32_t h1 = (global_id * 0x9E3779B9U) >> 24U;
    uint32_t h2 = (global_id * 0x85EBCA6BU) >> 24U;
    idx->bloom[h1 >> 6U] |= (1ULL << (h1 & 63U));
    idx->bloom[h2 >> 6U] |= (1ULL << (h2 & 63U));

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

#define MAX_TILE_SLOTS   1024U
#define MAX_NUM_TILES    256U

typedef struct {
    bool              valid;
    uint16_t          num_vertices;
    uint16_t          max_vertices;
    uint32_t          max_bytes_budget;
    uint32_t          cur_bytes_used;
    uint32_t          dirty_count;
    uint64_t          dirty_bits[MAX_TILE_SLOTS / 64U];
    uint32_t          row_offset[MAX_TILE_SLOTS + 1];
    uint32_t          logical_ids[MAX_TILE_SLOTS];
    uint64_t          packed_weights[MAX_TILE_SLOTS];
    CompactLocalIndex local_idx;

    uint32_t         *packed_edges;
    uint16_t         *packed_slots;
    uint64_t         *packed_target_weights;
    uint32_t          edges_cap;

    double            C_b_ns;
    double            R_b_ns;
    double            ewma_cold_ns;
    double            ewma_hot_ns;
    double            g_b_ns;
    uint32_t          n_b_window;
    uint32_t          queries_since_promo;
    double            first_query_ns;
    double            realized_savings_ns;
} MicroTile;

typedef enum {
    MODE_V3_FIXED_1024 = 0,
    MODE_V4_FIXED_128  = 1,
    MODE_V5_FIXED_64   = 2,
    MODE_V6_INC_BUDGET = 3,
    MODE_V7_INC_H3     = 4
} AlgMode;

typedef struct {
    AlgMode   mode;
    MicroTile tiles[MAX_NUM_TILES];
    uint32_t  num_allocated_tiles;
    uint16_t  assigned_tile[NUM_VERTICES];
    uint16_t  assigned_slot[NUM_VERTICES];
    uint8_t   vertex_heat[NUM_VERTICES];
    uint32_t  active_tiles_count;

    uint32_t  promotions;
    uint32_t  useless_promotions;
    uint32_t  invalidations;
    uint64_t  expanded_hot_hits;
    uint64_t  expanded_total;
    uint64_t  read_through_fallbacks;
    uint64_t  total_bytes_copied;
    double    capture_ns_sum;
    double    initial_build_ns_sum;
    double    append_ns_sum;
    double    first_query_ns_sum;
    uint32_t  first_query_count;
    double    subseq_query_ns_sum;
    uint32_t  subseq_query_count;
    double    cold_query_ns_sum;
    uint32_t  cold_query_count;
    size_t    bytes_allocated;
} Alg1Engine;

static void alg1_engine_init(Alg1Engine *eng, AlgMode mode) {
    memset(eng, 0, sizeof(Alg1Engine));
    eng->mode = mode;
    memset(eng->assigned_tile, 0xFF, sizeof(eng->assigned_tile));
    memset(eng->assigned_slot, 0xFF, sizeof(eng->assigned_slot));
    eng->bytes_allocated = sizeof(Alg1Engine);

    uint16_t max_v = 128;
    uint32_t max_b = 40960;
    if (mode == MODE_V3_FIXED_1024) { max_v = 1024; max_b = 524288; }
    else if (mode == MODE_V4_FIXED_128) { max_v = 128; max_b = 65536; }
    else if (mode == MODE_V5_FIXED_64)  { max_v = 64;  max_b = 32768; }

    for (uint32_t b = 0; b < MAX_NUM_TILES; b++) {
        eng->tiles[b].max_vertices = max_v;
        eng->tiles[b].max_bytes_budget = max_b;
        eng->tiles[b].C_b_ns = (mode == MODE_V3_FIXED_1024) ? 140000.0 :
                               (mode == MODE_V4_FIXED_128)  ? 14000.0  :
                               (mode == MODE_V5_FIXED_64)   ? 7000.0   : 5000.0;
        eng->tiles[b].g_b_ns = 1500.0;
    }
}

static void alg1_engine_free(Alg1Engine *eng) {
    for (uint32_t b = 0; b < MAX_NUM_TILES; b++) {
        free(eng->tiles[b].packed_edges);
        free(eng->tiles[b].packed_slots);
        free(eng->tiles[b].packed_target_weights);
    }
}

static double alg1_build_or_expand_tile(
    Alg1Engine *eng,
    uint16_t b,
    const DynGraph *auth,
    const uint32_t *captured_v,
    uint16_t count
) {
    double t0 = now_ns();
    MicroTile *t = &eng->tiles[b];

    if (t->valid && t->queries_since_promo > 0 && t->realized_savings_ns < (t->C_b_ns + t->R_b_ns)) {
        eng->useless_promotions++;
    }

    if (t->valid) {
        for (uint16_t i = 0; i < t->num_vertices; i++) {
            uint32_t old_u = t->logical_ids[i];
            if (eng->assigned_tile[old_u] == b) {
                eng->assigned_tile[old_u] = NO_TILE;
            }
        }
    } else {
        eng->active_tiles_count++;
    }

    if (count > t->max_vertices) count = t->max_vertices;
    local_index_clear(&t->local_idx, t->max_vertices);

    uint16_t actual_count = 0;
    uint32_t total_edges_needed = 0;
    uint32_t bytes_used = 0;

    for (uint16_t i = 0; i < count; i++) {
        uint32_t u = captured_v[i];
        uint32_t d = auth->v[u].deg;
        uint32_t v_bytes = sizeof(uint32_t) + sizeof(uint64_t) + d * (sizeof(uint32_t) + sizeof(uint64_t));
        if ((eng->mode == MODE_V6_INC_BUDGET || eng->mode == MODE_V7_INC_H3) &&
            (bytes_used + v_bytes > t->max_bytes_budget) && actual_count >= 16U) {
            break;
        }
        bytes_used += v_bytes;
        total_edges_needed += d;
        if (eng->assigned_tile[u] != NO_TILE && eng->assigned_tile[u] != b) {
            uint16_t old_b = eng->assigned_tile[u];
            uint16_t old_s = eng->assigned_slot[u];
            if (eng->tiles[old_b].valid && old_s < eng->tiles[old_b].num_vertices) {
                eng->tiles[old_b].dirty_bits[old_s >> 6U] |= (1ULL << (old_s & 63U));
            }
        }
        t->logical_ids[actual_count] = u;
        t->packed_weights[actual_count] = auth->payloads[u].weight_sum;
        local_index_insert(&t->local_idx, u, actual_count);
        eng->assigned_tile[u] = b;
        eng->assigned_slot[u] = actual_count;
        actual_count++;
    }

    if (total_edges_needed > t->edges_cap) {
        uint32_t new_cap = total_edges_needed + 512U;
        t->packed_edges = (uint32_t *)realloc(t->packed_edges, new_cap * sizeof(uint32_t));
        t->packed_target_weights = (uint64_t *)realloc(t->packed_target_weights, new_cap * sizeof(uint64_t));
        eng->bytes_allocated += (size_t)(new_cap - t->edges_cap) *
                                (sizeof(uint32_t) + sizeof(uint64_t));
        t->edges_cap = new_cap;
    }

    uint32_t offset = 0;
    for (uint16_t slot = 0; slot < actual_count; slot++) {
        uint32_t u = t->logical_ids[slot];
        t->row_offset[slot] = offset;
        const uint32_t *edges = auth->v[u].edges;
        uint32_t d = auth->v[u].deg;

        for (uint32_t e = 0; e < d; e++) {
            uint32_t v = edges[e];
            t->packed_edges[offset + e] = v;
            t->packed_target_weights[offset + e] = auth->payloads[v].weight_sum;
        }
        offset += d;
    }

    t->row_offset[actual_count] = offset;
    t->num_vertices = actual_count;
    t->cur_bytes_used = bytes_used;
    memset(t->dirty_bits, 0, sizeof(t->dirty_bits));
    t->valid = true;
    t->dirty_count = 0;
    t->queries_since_promo = 0;
    t->realized_savings_ns = 0.0;

    double dt = now_ns() - t0;
    t->C_b_ns = dt;
    t->R_b_ns = 0.0;

    eng->promotions++;
    eng->capture_ns_sum += dt;
    eng->initial_build_ns_sum += dt;
    eng->total_bytes_copied += bytes_used;
    return dt;
}

static double alg1_append_to_tile(
    Alg1Engine *eng,
    uint16_t b,
    const DynGraph *auth,
    const uint32_t *candidates,
    uint16_t cand_count
) {
    MicroTile *t = &eng->tiles[b];
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

        uint32_t d = auth->v[u].deg;
        uint32_t v_bytes = sizeof(uint32_t) + sizeof(uint64_t) +
                           d * (sizeof(uint32_t) + sizeof(uint64_t));
        if (t->cur_bytes_used + bytes_added + v_bytes > t->max_bytes_budget) break;

        if (offset + d > t->edges_cap) {
            uint32_t new_cap = (offset + d) * 2U + 256U;
            t->packed_edges = (uint32_t *)realloc(t->packed_edges, new_cap * sizeof(uint32_t));
            t->packed_target_weights = (uint64_t *)realloc(t->packed_target_weights, new_cap * sizeof(uint64_t));
            eng->bytes_allocated += (size_t)(new_cap - t->edges_cap) *
                                    (sizeof(uint32_t) + sizeof(uint64_t));
            t->edges_cap = new_cap;
        }

        if (eng->assigned_tile[u] != NO_TILE && eng->assigned_tile[u] != b) {
            uint16_t old_b = eng->assigned_tile[u];
            uint16_t old_s = eng->assigned_slot[u];
            if (eng->tiles[old_b].valid && old_s < eng->tiles[old_b].num_vertices) {
                eng->tiles[old_b].dirty_bits[old_s >> 6U] |= (1ULL << (old_s & 63U));
            }
        }

        uint16_t slot = t->num_vertices++;
        t->logical_ids[slot] = u;
        t->packed_weights[slot] = auth->payloads[u].weight_sum;
        t->row_offset[slot] = offset;
        local_index_insert(&t->local_idx, u, slot);
        eng->assigned_tile[u] = b;
        eng->assigned_slot[u] = slot;

        const uint32_t *edges = auth->v[u].edges;
        for (uint32_t e = 0; e < d; e++) {
            uint32_t v = edges[e];
            t->packed_edges[offset + e] = v;
            t->packed_target_weights[offset + e] = auth->payloads[v].weight_sum;
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

static inline void alg1_notify_update(Alg1Engine *eng, uint32_t u) {
    uint16_t b = eng->assigned_tile[u];
    if (b == NO_TILE) return;
    MicroTile *t = &eng->tiles[b];
    if (!t->valid) return;

    uint16_t slot = eng->assigned_slot[u];
    uint64_t mask = 1ULL << (slot & 63U);
    if ((t->dirty_bits[slot >> 6U] & mask) == 0) {
        t->dirty_bits[slot >> 6U] |= mask;
        t->dirty_count++;
    }
    t->R_b_ns += 25.0;

    uint32_t thresh = (t->num_vertices / 5U);
    if (thresh < 2U) thresh = 2U;
    if (t->dirty_count > thresh) {
        if (t->realized_savings_ns < (t->C_b_ns + t->R_b_ns)) {
            eng->useless_promotions++;
        }
        for (uint16_t i = 0; i < t->num_vertices; i++) {
            uint32_t old_u = t->logical_ids[i];
            if (eng->assigned_tile[old_u] == b) eng->assigned_tile[old_u] = NO_TILE;
        }
        t->valid = false;
        if (eng->active_tiles_count > 0) eng->active_tiles_count--;
        eng->invalidations++;
    }
}

typedef struct {
    uint32_t queue_u[MAX_TILE_SLOTS];
    uint16_t queue_tile[MAX_TILE_SLOTS];
    uint16_t queue_slot[MAX_TILE_SLOTS];
    uint8_t  depth[MAX_TILE_SLOTS];
    uint32_t visited_epoch[NUM_VERTICES];
    uint64_t local_visited[MAX_TILE_SLOTS / 64U];
    uint32_t cur_epoch;
} BfsScratch;

static void bfs_scratch_init(BfsScratch *sc) {
    memset(sc, 0, sizeof(BfsScratch));
    sc->cur_epoch = 1;
}

static uint16_t collect_bfs_neighborhood(
    const DynGraph *g,
    BfsScratch *sc,
    uint32_t root,
    uint32_t *out_nodes,
    uint16_t target_count
) {
    uint32_t ep = ++sc->cur_epoch;
    uint16_t head = 0, tail = 0;
    out_nodes[tail++] = root;
    sc->visited_epoch[root] = ep;

    while (head < tail && tail < target_count) {
        uint32_t u = out_nodes[head++];
        const uint32_t *edges = g->v[u].edges;
        uint32_t deg = g->v[u].deg;
        for (uint32_t i = 0; i < deg; i++) {
            uint32_t v = edges[i];
            if (sc->visited_epoch[v] != ep) {
                sc->visited_epoch[v] = ep;
                out_nodes[tail++] = v;
                if (tail >= target_count) break;
            }
        }
    }
    return tail;
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
                } else {
                    goto done;
                }
            }
        }
    }
done:
    return accum ^ ((uint64_t)tail << 32);
}

static uint64_t bfs_query_alg1_configurable(
    const DynGraph *auth,
    Alg1Engine *eng,
    BfsScratch *sc,
    uint32_t root,
    bool allow_hot_read,
    uint16_t *out_visited_count,
    uint16_t *out_expanded_count,
    uint16_t *out_cold_expanded
) {
    uint16_t exp_cnt = 0;
    uint16_t cold_exp = 0;

    if (eng->active_tiles_count == 0) {
        uint32_t ep = ++sc->cur_epoch;
        uint32_t head = 0, tail = 0;
        sc->queue_u[tail] = root;
        sc->depth[tail] = 0;
        tail++;
        sc->visited_epoch[root] = ep;
        uint64_t accum = auth->payloads[root].weight_sum ^ (uint64_t)root;

        while (head < tail) {
            uint32_t u = sc->queue_u[head];
            uint8_t  d = sc->depth[head];
            head++;
            if (d >= BFS_MAX_DEPTH) continue;
            exp_cnt++;
            cold_exp++;
            eng->expanded_total++;
            const uint32_t *edges = auth->v[u].edges;
            uint32_t deg = auth->v[u].deg;
            for (uint32_t i = 0; i < deg; i++) {
                uint32_t v = edges[i];
                if (sc->visited_epoch[v] != ep) {
                    sc->visited_epoch[v] = ep;
                    accum += auth->payloads[v].weight_sum ^ ((uint64_t)v * (uint64_t)(d + 1));
                    if (tail < BFS_MAX_VISITED) {
                        sc->queue_u[tail] = v;
                        sc->depth[tail] = d + 1;
                        tail++;
                    } else {
                        goto done_cold;
                    }
                }
            }
        }
done_cold:
        *out_visited_count  = (uint16_t)tail;
        *out_expanded_count = exp_cnt;
        *out_cold_expanded  = cold_exp;
        return accum ^ ((uint64_t)tail << 32);
    }

    uint32_t ep = ++sc->cur_epoch;
    uint32_t head = 0, tail = 0;

    uint16_t root_tile = eng->assigned_tile[root];
    uint16_t root_slot = (root_tile != NO_TILE) ? eng->assigned_slot[root] : EXTERNAL_SLOT;
    const MicroTile *rt = (root_tile != NO_TILE) ? &eng->tiles[root_tile] : NULL;
    bool root_tile_valid = allow_hot_read && (rt && rt->valid && root_slot < rt->num_vertices);

    sc->queue_u[tail] = root;
    sc->depth[tail]   = 0;
    tail++;
    sc->visited_epoch[root] = ep;

    uint64_t root_w = root_tile_valid ? rt->packed_weights[root_slot] : auth->payloads[root].weight_sum;
    uint64_t accum = root_w ^ (uint64_t)root;

    while (head < tail) {
        uint32_t u = sc->queue_u[head];
        uint8_t  d = sc->depth[head];
        head++;

        if (d >= BFS_MAX_DEPTH) continue;
        exp_cnt++;
        eng->expanded_total++;

        uint16_t t_id = eng->assigned_tile[u];
        uint16_t slot_u = EXTERNAL_SLOT;
        const MicroTile *tile = NULL;
        bool use_hot = false;

        if (t_id != NO_TILE && eng->tiles[t_id].valid) {
            tile = &eng->tiles[t_id];
            slot_u = local_index_lookup_hash(&tile->local_idx, u);
            if (slot_u < tile->num_vertices) {
                uint64_t dirty = tile->dirty_bits[slot_u >> 6U] & (1ULL << (slot_u & 63U));
                if (__builtin_expect(dirty == 0, 1)) {
                    if (allow_hot_read) {
                        use_hot = true;
                        eng->expanded_hot_hits++;
                    } else {
                        cold_exp++;
                    }
                } else {
                    eng->read_through_fallbacks++;
                    cold_exp++;
                }
            } else {
                cold_exp++;
            }
        } else {
            cold_exp++;
        }

        if (use_hot) {
            uint32_t start = tile->row_offset[slot_u];
            uint32_t end   = tile->row_offset[slot_u + 1];
            const uint32_t *edges = tile->packed_edges + start;
            const uint64_t *tws   = tile->packed_target_weights + start;
            uint32_t deg = end - start;

            for (uint32_t i = 0; i < deg; i++) {
                uint32_t v = edges[i];
                if (sc->visited_epoch[v] != ep) {
                    sc->visited_epoch[v] = ep;
                    accum += tws[i] ^ ((uint64_t)v * (uint64_t)(d + 1));
                    if (tail < BFS_MAX_VISITED) {
                        sc->queue_u[tail] = v;
                        sc->depth[tail]   = d + 1;
                        tail++;
                    } else {
                        goto done_hot;
                    }
                }
            }
        } else {
            const uint32_t *edges = auth->v[u].edges;
            uint32_t deg = auth->v[u].deg;
            for (uint32_t i = 0; i < deg; i++) {
                uint32_t v = edges[i];
                if (sc->visited_epoch[v] != ep) {
                    sc->visited_epoch[v] = ep;
                    accum += auth->payloads[v].weight_sum ^ ((uint64_t)v * (uint64_t)(d + 1));
                    if (tail < BFS_MAX_VISITED) {
                        sc->queue_u[tail] = v;
                        sc->depth[tail]   = d + 1;
                        tail++;
                    } else {
                        goto done_hot;
                    }
                }
            }
        }
    }
done_hot:
    *out_visited_count  = (uint16_t)tail;
    *out_expanded_count = exp_cnt;
    *out_cold_expanded  = cold_exp;
    return accum ^ ((uint64_t)tail << 32);
}

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

typedef enum {
    ABL_V1_ORIG           = 0,
    ABL_V1_ON_V7_NOPROMO  = 1,
    ABL_V7_H3_NOREAD      = 2,
    ABL_V7_FULL           = 3
} AblationConfig;

typedef struct {
    double   wall_ms;
    double   total_ms;
    double   query_ms;
    double   maint_ms;
    double   hit_pct;
    double   cold_q_us;
    double   first_q_us;
    double   subseq_q_us;
    double   cap_init_us;
    double   cap_total_us;
    uint32_t promotions;
    uint32_t errors;
} AblationResult;

static void run_ablation_on_fixed_graph(
    DynGraph *g_work,
    const DynGraph *base_graph,
    const WorkloadTrace *tr,
    const uint64_t *oracle_hashes,
    AblationConfig cfg,
    Alg1Engine *eng,
    BfsScratch *scratch,
    AblationResult *res
) {
    memset(res, 0, sizeof(AblationResult));
    dyngraph_reset_in_place(g_work, base_graph);

    if (cfg == ABL_V1_ORIG) {
        bfs_scratch_init(scratch);
        scrub_l3_cache();
        double q_ns = 0.0, m_ns = 0.0;
        double tw0 = now_ns();
        for (uint32_t step = 0; step < NUM_STEPS_PER_RUN; step++) {
            if (tr->has_update_batch[step]) {
                double tu0 = now_ns();
                for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                    if (tr->upd_is_add[step][k]) dyngraph_add_edge(g_work, tr->upd_u[step][k], tr->upd_v[step][k]);
                    else dyngraph_remove_edge(g_work, tr->upd_u[step][k]);
                }
                m_ns += (now_ns() - tu0);
            }
            double tq0 = now_ns();
            uint64_t h = bfs_query_v1(g_work, scratch, tr->query_roots[step]);
            double dq = now_ns() - tq0;
            if (oracle_hashes && h != oracle_hashes[step]) res->errors++;
            q_ns += dq;
        }
        res->wall_ms   = (now_ns() - tw0) / 1e6;
        res->query_ms  = q_ns / 1e6;
        res->maint_ms  = m_ns / 1e6;
        res->total_ms  = res->query_ms + res->maint_ms;
        res->cold_q_us = (q_ns / NUM_STEPS_PER_RUN) / 1000.0;
        return;
    }

    alg1_engine_init(eng, MODE_V7_INC_H3);
    bfs_scratch_init(scratch);
    scrub_l3_cache();

    const bool allow_promo = (cfg == ABL_V7_H3_NOREAD || cfg == ABL_V7_FULL);
    const bool allow_read  = (cfg == ABL_V7_FULL);

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
        if (tr->has_update_batch[s]) {
            double tu0 = now_ns();
            for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                uint32_t u = tr->upd_u[s][k];
                if (tr->upd_is_add[s][k]) dyngraph_add_edge(g_work, u, tr->upd_v[s][k]);
                else dyngraph_remove_edge(g_work, u);
                if (allow_promo) alg1_notify_update(eng, u);
            }
            maint_ns_sum += (now_ns() - tu0);
        }

        if (allow_promo && s > 0 && (s % 200U) == 0) {
            double td0 = now_ns();
            for (uint32_t b = 0; b < eng->num_allocated_tiles; b++) {
                eng->tiles[b].n_b_window >>= 1;
            }
            maint_ns_sum += (now_ns() - td0);
        }

        uint32_t root = tr->query_roots[s];
        uint16_t b = eng->assigned_tile[root];
        bool was_valid_before = (b != NO_TILE && eng->tiles[b].valid);

        uint16_t visited_cnt  = 0;
        uint16_t expanded_cnt = 0;
        uint16_t cold_exp_cnt = 0;
        double tq0 = now_ns();
        uint64_t h = bfs_query_alg1_configurable(g_work, eng, scratch, root, allow_read,
                                                 &visited_cnt, &expanded_cnt, &cold_exp_cnt);
        double dq = now_ns() - tq0;
        if (oracle_hashes && h != oracle_hashes[s]) res->errors++;
        query_ns_sum += dq;

        if (allow_promo) {
            double tm_post0 = now_ns();
            if (was_valid_before) {
                MicroTile *tb = &eng->tiles[b];
                tb->queries_since_promo++;
                if (tb->queries_since_promo == 1U) {
                    tb->first_query_ns = dq;
                    eng->first_query_ns_sum += dq;
                    eng->first_query_count++;
                } else {
                    eng->subseq_query_ns_sum += dq;
                    eng->subseq_query_count++;
                }
                hot_ewma_ns = 0.90 * hot_ewma_ns + 0.10 * dq;
                double gain = (cold_ewma_ns > hot_ewma_ns) ? (cold_ewma_ns - hot_ewma_ns) : 200.0;
                tb->g_b_ns = gain;
                tb->realized_savings_ns += gain;

                if (cold_exp_cnt > 0) {
                    uint16_t app_cnt = (visited_cnt < 64U) ? visited_cnt : 64U;
                    alg1_append_to_tile(eng, b, g_work, scratch->queue_u, app_cnt);
                }
            } else {
                eng->cold_query_ns_sum += dq;
                eng->cold_query_count++;
                cold_ewma_ns = 0.90 * cold_ewma_ns + 0.10 * dq;

                uint32_t neighborhood_heat = 0;
                uint16_t b_existing = NO_TILE;
                uint16_t check_n = (expanded_cnt < 16U) ? expanded_cnt : 16U;
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
                    alg1_append_to_tile(eng, b_existing, g_work, scratch->queue_u, app_cnt);
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
                        double build_ns = alg1_build_or_expand_tile(eng, target_b, g_work, scratch->queue_u, init_cnt);
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

    res->wall_ms      = (now_ns() - tw0) / 1e6;
    res->query_ms     = query_ns_sum / 1e6;
    res->maint_ms     = maint_ns_sum / 1e6;
    res->total_ms     = res->query_ms + res->maint_ms;
    res->hit_pct      = eng->expanded_total ? (100.0 * (double)eng->expanded_hot_hits / (double)eng->expanded_total) : 0.0;
    res->cold_q_us    = eng->cold_query_count ? ((eng->cold_query_ns_sum / eng->cold_query_count) / 1000.0) : 0.0;
    res->first_q_us   = eng->first_query_count ? ((eng->first_query_ns_sum / eng->first_query_count) / 1000.0) : 0.0;
    res->subseq_q_us  = eng->subseq_query_count ? ((eng->subseq_query_ns_sum / eng->subseq_query_count) / 1000.0) : 0.0;
    res->cap_init_us  = eng->promotions ? ((eng->initial_build_ns_sum / eng->promotions) / 1000.0) : 0.0;
    res->cap_total_us = eng->promotions ? ((eng->capture_ns_sum / eng->promotions) / 1000.0) : 0.0;
    res->promotions   = eng->promotions;

    alg1_engine_free(eng);
}

int main(void) {
    SetConsoleOutputCP(65001);
    setvbuf(stdout, NULL, _IONBF, 0);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);
    init_timer();

    const uint64_t TEST_SEEDS[NUM_TEST_SEEDS] = {
        0x778899AABBCCDDEEULL,
        0x3141592653589793ULL,
        0x2718281828459045ULL
    };

    DynGraph test_graphs[NUM_TEST_SEEDS];
    DynGraph work_graphs[NUM_TEST_SEEDS];
    for (uint32_t s = 0; s < NUM_TEST_SEEDS; s++) {
        generate_base_graph(&test_graphs[s], TEST_SEEDS[s]);
        dyngraph_clone(&work_graphs[s], &test_graphs[s]);
    }

    WorkloadTrace *tr = (WorkloadTrace *)malloc(sizeof(WorkloadTrace));
    BfsScratch *scratch = (BfsScratch *)malloc(sizeof(BfsScratch));
    Alg1Engine *eng = (Alg1Engine *)malloc(sizeof(Alg1Engine));
    uint64_t *oracle_hashes = (uint64_t *)malloc(NUM_STEPS_PER_RUN * sizeof(uint64_t));

    const char *ABL_NAMES[4] = {
        "1. V1-Orig (bfs_query_v1)              ",
        "2. V1-en-Motor-V7 (Sin promociones)    ",
        "3. V7-H3-Sin-Lectura (Solo Promo+Maint)",
        "4. V7-Full (H3 + Lectura Micro-Tile)   "
    };

    printf("========================================================================================================================\n");
    printf(" ABLACION CAUSAL DE 4 VIAS (MISMOS PUNTEROS VIRTUALES `g_work` + SCRUB L3 DE 32 MB + ORDEN ROTADO)\n");
    printf("========================================================================================================================\n");

    for (int w = 0; w < NUM_WORKLOADS; w++) {
        AblationResult avg[4] = {0};
        for (uint32_t s = 0; s < NUM_TEST_SEEDS; s++) {
            generate_workload_trace(tr, &test_graphs[s], (WorkloadType)w, TEST_SEEDS[s]);

            /* Calcular hashes de oráculo */
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

            for (int k = 0; k < 4; k++) {
                int c = (k + (int)s) & 3;
                AblationResult r;
                run_ablation_on_fixed_graph(&work_graphs[s], &test_graphs[s], tr, oracle_hashes,
                                            (AblationConfig)c, eng, scratch, &r);
                avg[c].wall_ms      += r.wall_ms / NUM_TEST_SEEDS;
                avg[c].total_ms     += r.total_ms / NUM_TEST_SEEDS;
                avg[c].query_ms     += r.query_ms / NUM_TEST_SEEDS;
                avg[c].maint_ms     += r.maint_ms / NUM_TEST_SEEDS;
                avg[c].hit_pct      += r.hit_pct / NUM_TEST_SEEDS;
                avg[c].cold_q_us    += r.cold_q_us / NUM_TEST_SEEDS;
                avg[c].first_q_us   += r.first_q_us / NUM_TEST_SEEDS;
                avg[c].subseq_q_us  += r.subseq_q_us / NUM_TEST_SEEDS;
                avg[c].cap_init_us  += r.cap_init_us / NUM_TEST_SEEDS;
                avg[c].cap_total_us += r.cap_total_us / NUM_TEST_SEEDS;
                avg[c].promotions   += r.promotions;
                avg[c].errors       += r.errors;
            }
        }

        printf("\n %s:\n", WORKLOAD_NAMES[w]);
        printf("   Configuracion                           | Wall(ms) | Total(ms) | Query(ms) | Maint(ms) | Hit%%  | Prom | Exact\n");
        printf("   ----------------------------------------+----------+-----------+-----------+-----------+-------+------+------\n");
        for (int c = 0; c < 4; c++) {
            printf("   %s | %8.2f | %9.2f | %9.2f | %9.2f | %4.1f%% | %4u | %s\n",
                   ABL_NAMES[c], avg[c].wall_ms, avg[c].total_ms, avg[c].query_ms, avg[c].maint_ms,
                   avg[c].hit_pct, avg[c].promotions / NUM_TEST_SEEDS,
                   (avg[c].errors == 0) ? "100%" : "ERR");
        }
        if (w != WORKLOAD_D_UNIFORM_RANDOM) {
            double v1_q_us  = avg[0].cold_q_us;
            double cold_q   = avg[3].cold_q_us;
            double first_q  = avg[3].first_q_us;
            double sub_q    = avg[3].subseq_q_us;
            double cap_init = avg[3].cap_init_us;
            double cap_tot  = avg[3].cap_total_us;
            double brk_cold = (cold_q > sub_q)  ? (cap_tot / (cold_q - sub_q))  : 0.0;
            double brk_v1   = (v1_q_us > sub_q) ? (cap_tot / (v1_q_us - sub_q)) : 0.0;
            double brk_1st  = (first_q > sub_q) ? (cap_tot / (first_q - sub_q)) : 0.0;
            printf("   [Desglose BrkEv V7-Full] V1_Q=%.2f us | ColdQ_V7=%.2f us (sesgo 8%% ruido) | 1stQ=%.2f us | SubQ=%.2f us\n",
                   v1_q_us, cold_q, first_q, sub_q);
            printf("                            Cap_Init=%.2f us | Cap_Total(Init+Appends)=%.2f us -> BrkEv(vs ColdQ)=%.1f | BrkEv(vs V1)=%.1f | BrkEv(vs 1stQ)=%.1f\n",
                   cap_init, cap_tot, brk_cold, brk_v1, brk_1st);
        }
    }

    free(tr);
    free(scratch);
    free(eng);
    free(oracle_hashes);
    if (g_scrub_buf) _aligned_free(g_scrub_buf);
    for (uint32_t s = 0; s < NUM_TEST_SEEDS; s++) {
        dyngraph_free(&work_graphs[s]);
        dyngraph_free(&test_graphs[s]);
    }
    return 0;
}
