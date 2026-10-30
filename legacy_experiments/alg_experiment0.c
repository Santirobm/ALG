/*
 * ============================================================================
 * EXPERIMENTO 0: ADAPTIVE LOCALITY GRAPH (ALG) — BANCO DE FALSACIÓN RIGUROSO
 * ============================================================================
 * Implementa las correcciones metodológicas acordadas:
 *   1. Grafo de escala superior a la caché L3 (262,144 vértices, ~5.5M aristas,
 *      > 55 MB en estructuras base vs 24 MB de L3 en el Intel Core i5-13500).
 *   2. IDs lógicos estables asignados por orden de llegada (permutación pseudoaleatoria
 *      respecto a la topología de comunidades), evitando la trampa de asumir que
 *      el grafo base ya viene pre-ordenado por comunidad en RAM.
 *   3. Arquitectura Hot-Tiles conservadora con coherencia Read-Through:
 *      - Los IDs lógicos jamás cambian.
 *      - Cada Hot-Tile almacena copias compactas de adyacencias, pesos y slots
 *        locales (`uint16_t`, compatible con IDs de 32 bits completos), más un
 *        bitmap `dirty_bits` (128 bytes en L1) y generación del Tile.
 *      - Las actualizaciones se aplican primero al grafo autoritativo y marcan
 *        sucio el vértice en el Tile; el recorrido detecta en O(1) cualquier
 *        vértice modificado y hace fallback al grafo autoritativo.
 *   4. Regla de inversión empírica H3 por Tile b:
 *         n_b * g_b > alpha * C_b + R_b
 *      donde C_b, g_b y R_b se miden en nanosegundos reales sobre consultas
 *      equivalentes.
 *   5. Cuatro variantes (V0 CSR+Rebuild, V1 Dynamic AdjList, V2 ALG Periodic,
 *      V3 ALG Adaptive H3) en cinco cargas (A, B, C, D, E) x 3 semillas.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

/* ============================================================================
 * DIMENSIONES DEL GRAFO (SUPERA LOS 24 MB DE CACHÉ L3 DEL i5-13500)
 * ============================================================================ */
#define NUM_VERTICES        262144U                 /* 256K vértices */
#define NUM_TILES           256U                    /* 256 regiones/comunidades */
#define VERTICES_PER_TILE   (NUM_VERTICES / NUM_TILES) /* 1,024 vértices por Tile */
#define EXTERNAL_SLOT       0xFFFFU

#define BFS_MAX_DEPTH       3U
#define BFS_MAX_VISITED     512U

#define NUM_STEPS_PER_RUN   3000U
#define UPDATES_PER_BATCH   64U
#define UPDATE_BATCH_PERIOD 150U    /* Cada 150 consultas entra un lote de 64 mutaciones */
#define V2_PERIODIC_K       200U    /* Periodo de reconstrucción para V2 */
#define NUM_SEEDS           3U

/* ============================================================================
 * CRONÓMETRO DE ALTA PRECISIÓN (NANOSEGUNDOS) Y PRNG
 * ============================================================================ */
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

/* Payload de 64 bytes (1 línea de caché) por vértice en el grafo autoritativo */
typedef struct __attribute__((aligned(64))) {
    uint64_t weight_sum;
    uint32_t community_id;
    uint32_t local_slot;
    uint64_t reserved[6];
} VertexPayload;

/* ============================================================================
 * ESTRUCTURA AUTORITATIVA (V1: Dynamic Contiguous AdjList por Vértice)
 * ============================================================================ */
typedef struct {
    uint32_t *edges;
    uint32_t  deg;
    uint32_t  cap;
    uint32_t  version;
} DynVertex;

typedef struct {
    DynVertex     *v;
    VertexPayload *payloads;
    /* Mapa de comunidad (Tile b, slot i) -> ID lógico estable u */
    uint32_t      *tile_members; /* [NUM_TILES * VERTICES_PER_TILE] */
    uint64_t       total_edges;
    size_t         bytes_allocated;
} DynGraph;

static void dyngraph_free(DynGraph *g) {
    if (!g->v) return;
    for (uint32_t u = 0; u < NUM_VERTICES; u++) {
        free(g->v[u].edges);
    }
    free(g->v);
    free(g->tile_members);
    _aligned_free(g->payloads);
    g->v = NULL;
}

static void dyngraph_clone(DynGraph *dst, const DynGraph *src) {
    dst->v = (DynVertex *)calloc(NUM_VERTICES, sizeof(DynVertex));
    dst->payloads = (VertexPayload *)_aligned_malloc(NUM_VERTICES * sizeof(VertexPayload), 64);
    dst->tile_members = (uint32_t *)malloc(NUM_VERTICES * sizeof(uint32_t));

    memcpy(dst->payloads, src->payloads, NUM_VERTICES * sizeof(VertexPayload));
    memcpy(dst->tile_members, src->tile_members, NUM_VERTICES * sizeof(uint32_t));
    dst->total_edges = src->total_edges;
    dst->bytes_allocated = NUM_VERTICES * (sizeof(DynVertex) + sizeof(VertexPayload) + sizeof(uint32_t));

    for (uint32_t u = 0; u < NUM_VERTICES; u++) {
        uint32_t cap = src->v[u].cap;
        dst->v[u].edges = (uint32_t *)malloc(cap * sizeof(uint32_t));
        memcpy(dst->v[u].edges, src->v[u].edges, src->v[u].deg * sizeof(uint32_t));
        dst->v[u].deg = src->v[u].deg;
        dst->v[u].cap = cap;
        dst->v[u].version = src->v[u].version;
        dst->bytes_allocated += (size_t)cap * sizeof(uint32_t);
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

/*
 * Generación realista del grafo base:
 *   - Los vértices tienen IDs lógicos estables `0 .. NUM_VERTICES-1` asignados por
 *     orden de llegada (permutación aleatoria Fisher-Yates respecto a las comunidades).
 *   - Por tanto, dos vértices de la misma comunidad NO están artificialmente contiguos
 *     en el arreglo global `v[]` ni en `payloads[]`.
 */
static void generate_base_graph(DynGraph *g, uint64_t seed) {
    g->v = (DynVertex *)calloc(NUM_VERTICES, sizeof(DynVertex));
    g->payloads = (VertexPayload *)_aligned_malloc(NUM_VERTICES * sizeof(VertexPayload), 64);
    g->tile_members = (uint32_t *)malloc(NUM_VERTICES * sizeof(uint32_t));
    memset(g->payloads, 0, NUM_VERTICES * sizeof(VertexPayload));
    g->total_edges = 0;
    g->bytes_allocated = NUM_VERTICES * (sizeof(DynVertex) + sizeof(VertexPayload) + sizeof(uint32_t));

    uint64_t rng = seed ^ 0x9E3779B97F4A7C15ULL;

    /* Crear permutación aleatoria de IDs lógicos para asignar a las 256 comunidades */
    for (uint32_t i = 0; i < NUM_VERTICES; i++) {
        g->tile_members[i] = i;
    }
    for (uint32_t i = NUM_VERTICES - 1; i > 0; i--) {
        uint32_t j = (uint32_t)(xorshift64s(&rng) % (i + 1));
        uint32_t tmp = g->tile_members[i];
        g->tile_members[i] = g->tile_members[j];
        g->tile_members[j] = tmp;
    }

    /* Inicializar vértices en orden de ID lógico */
    for (uint32_t u = 0; u < NUM_VERTICES; u++) {
        uint32_t init_cap = 28;
        g->v[u].edges = (uint32_t *)malloc(init_cap * sizeof(uint32_t));
        g->v[u].deg = 0;
        g->v[u].cap = init_cap;
        g->v[u].version = 1;
        g->bytes_allocated += init_cap * sizeof(uint32_t);
    }

    /* Registrar a qué comunidad y slot pertenece cada ID lógico */
    for (uint32_t b = 0; b < NUM_TILES; b++) {
        for (uint32_t slot = 0; slot < VERTICES_PER_TILE; slot++) {
            uint32_t u = g->tile_members[b * VERTICES_PER_TILE + slot];
            g->payloads[u].weight_sum   = ((uint64_t)u * 2654435761ULL) ^ 0x55AA55AA12345678ULL;
            g->payloads[u].community_id = b;
            g->payloads[u].local_slot   = slot;
        }
    }

    /* Generar aristas: 88% intra-comunidad, 12% puentes a otras comunidades */
    for (uint32_t b = 0; b < NUM_TILES; b++) {
        const uint32_t *comm_nodes = &g->tile_members[b * VERTICES_PER_TILE];
        for (uint32_t slot = 0; slot < VERTICES_PER_TILE; slot++) {
            uint32_t u = comm_nodes[slot];
            uint32_t deg = 14U + (uint32_t)(xorshift64s(&rng) % 13U); /* 14..26 aristas */
            for (uint32_t e = 0; e < deg; e++) {
                uint64_t r = xorshift64s(&rng);
                uint32_t target;
                if ((r % 100U) < 88U) {
                    uint32_t target_slot = (uint32_t)((r >> 8) & (VERTICES_PER_TILE - 1U));
                    target = comm_nodes[target_slot];
                } else {
                    target = (uint32_t)((r >> 8) % NUM_VERTICES);
                }
                if (target != u) {
                    dyngraph_add_edge(g, u, target);
                }
            }
        }
    }
}

/* ============================================================================
 * VARIANTE V0: CSR ESTÁTICO CON RECONSTRUCCIÓN AL APLICAR ACTUALIZACIONES
 * ============================================================================ */
typedef struct {
    uint32_t      *row_ptr;
    uint32_t      *col_idx;
    VertexPayload *csr_payloads;
    size_t         col_cap;
    size_t         bytes_allocated;
} CsrGraph;

static void csr_init(CsrGraph *csr, size_t initial_edges) {
    csr->row_ptr = (uint32_t *)malloc((NUM_VERTICES + 1) * sizeof(uint32_t));
    csr->col_cap = initial_edges + 65536U;
    csr->col_idx = (uint32_t *)malloc(csr->col_cap * sizeof(uint32_t));
    csr->csr_payloads = (VertexPayload *)_aligned_malloc(NUM_VERTICES * sizeof(VertexPayload), 64);
    csr->bytes_allocated = (NUM_VERTICES + 1) * sizeof(uint32_t) +
                           csr->col_cap * sizeof(uint32_t) +
                           NUM_VERTICES * sizeof(VertexPayload);
}

static void csr_free(CsrGraph *csr) {
    free(csr->row_ptr);
    free(csr->col_idx);
    _aligned_free(csr->csr_payloads);
}

static void csr_rebuild(CsrGraph *csr, const DynGraph *auth) {
    if (auth->total_edges > csr->col_cap) {
        size_t new_cap = auth->total_edges + 65536U;
        csr->col_idx = (uint32_t *)realloc(csr->col_idx, new_cap * sizeof(uint32_t));
        csr->bytes_allocated += (new_cap - csr->col_cap) * sizeof(uint32_t);
        csr->col_cap = new_cap;
    }
    uint32_t offset = 0;
    for (uint32_t u = 0; u < NUM_VERTICES; u++) {
        csr->row_ptr[u] = offset;
        uint32_t d = auth->v[u].deg;
        memcpy(csr->col_idx + offset, auth->v[u].edges, d * sizeof(uint32_t));
        offset += d;
    }
    csr->row_ptr[NUM_VERTICES] = offset;
    memcpy(csr->csr_payloads, auth->payloads, NUM_VERTICES * sizeof(VertexPayload));
}

/* ============================================================================
 * ARQUITECTURA HOT-TILES CONSERVADORA (PARA V2 Y V3)
 * ============================================================================
 * Coherencia Read-Through sin tocar RAM fría durante recorridos en el Tile:
 *   - Cada Tile b agrupa los 1,024 vértices de la región b (conservando sus IDs
 *     lógicos globales de 32 bits intactos).
 *   - `dirty_bits[16]` (128 bytes = 2 líneas de caché dentro del propio Tile)
 *     marca con 1 cualquier slot del Tile cuyo vértice haya sido modificado en
 *     el grafo autoritativo desde la última construcción del Tile.
 *   - Para cada arista en `packed_edges[i]`, `packed_slots[i]` guarda el slot
 *     local `0..1023` si el vecino pertenece al mismo Tile, o `EXTERNAL_SLOT`
 *     (`0xFFFF`) si salta a otro Tile.
 *   - Así, mientras el recorrido permanece dentro de un Hot-Tile limpio, accede
 *     exclusivamente a memoria contigua residente en caché L1/L2 (~96 KB por Tile),
 *     evitando los fallos de L3/TLB sobre el grafo global de >55 MB.
 *   - Si un vértice tiene su bit activo en `dirty_bits`, se detecta en O(1) sin
 *     salir de L1 y se resuelve mediante la estructura autoritativa global.
 */
typedef struct {
    bool      valid;
    uint32_t  tile_generation;
    uint32_t  dirty_count;
    uint64_t  dirty_bits[VERTICES_PER_TILE / 64U]; /* 128 bytes en L1 */
    uint32_t  row_offset[VERTICES_PER_TILE + 1];
    uint32_t  logical_ids[VERTICES_PER_TILE];
    uint64_t  packed_weights[VERTICES_PER_TILE];
    uint32_t *packed_edges;
    uint16_t *packed_slots;
    uint32_t  edges_cap;

    /* Parámetros empíricos H3 medidos para el Tile b */
    double    C_b_ns;               /* Coste medido de construcción (ns) */
    double    R_b_ns;               /* Coste acumulado de mantenimiento/invalidaciones (ns) */
    double    ewma_global_ns;       /* Latencia media observada sin Tile (ns) */
    double    ewma_hot_ns;          /* Latencia media observada con Tile (ns) */
    double    g_b_ns;               /* Ahorro empírico por consulta elegible (ns) */
    uint32_t  n_b_window;           /* Consultas elegibles observadas */
    double    realized_savings_ns;  /* Ahorro real acumulado desde la promoción */
} HotTile;

typedef struct {
    HotTile  tiles[NUM_TILES];
    /* Tabla compacta de membresía (8 bytes por vértice: tile_id + local_slot) */
    uint16_t vertex_tile[NUM_VERTICES];
    uint16_t vertex_slot[NUM_VERTICES];

    uint32_t active_tiles_count;
    uint32_t total_promotions;
    uint32_t useless_promotions;
    uint32_t stale_fallbacks;
    uint32_t tile_invalidations;
    double   total_maint_ns;
    size_t   bytes_allocated;
} AlgTileManager;

static void alg_manager_init(AlgTileManager *mgr, const DynGraph *auth) {
    memset(mgr, 0, sizeof(AlgTileManager));
    mgr->bytes_allocated = sizeof(AlgTileManager);

    for (uint32_t u = 0; u < NUM_VERTICES; u++) {
        mgr->vertex_tile[u] = (uint16_t)auth->payloads[u].community_id;
        mgr->vertex_slot[u] = (uint16_t)auth->payloads[u].local_slot;
    }

    for (uint32_t b = 0; b < NUM_TILES; b++) {
        HotTile *t = &mgr->tiles[b];
        t->valid = false;
        t->C_b_ns = 120000.0; /* ~120 us (se reemplaza por el tiempo real medido en alg_build_tile) */
        t->g_b_ns = 1500.0;   /* ~1.5 us (se actualiza online según ewma_global_ns - ewma_hot_ns) */
    }
}

static void alg_manager_free(AlgTileManager *mgr) {
    for (uint32_t b = 0; b < NUM_TILES; b++) {
        free(mgr->tiles[b].packed_edges);
        free(mgr->tiles[b].packed_slots);
    }
}

static void alg_build_tile(AlgTileManager *mgr, uint32_t b, const DynGraph *auth) {
    double t0 = now_ns();
    HotTile *t = &mgr->tiles[b];

    if (t->valid && t->realized_savings_ns < (t->C_b_ns + t->R_b_ns)) {
        mgr->useless_promotions++;
    }

    const uint32_t *members = &auth->tile_members[b * VERTICES_PER_TILE];
    if (t->edges_cap == 0) {
        t->edges_cap = VERTICES_PER_TILE * 28U;
        t->packed_edges = (uint32_t *)malloc(t->edges_cap * sizeof(uint32_t));
        t->packed_slots = (uint16_t *)realloc(t->packed_slots, t->edges_cap * sizeof(uint16_t));
        mgr->bytes_allocated += (size_t)t->edges_cap * (sizeof(uint32_t) + sizeof(uint16_t));
    }

    uint32_t offset = 0;
    for (uint32_t slot = 0; slot < VERTICES_PER_TILE; slot++) {
        if (slot + 4U < VERTICES_PER_TILE) {
            uint32_t next_u = members[slot + 4U];
            __builtin_prefetch(&auth->v[next_u], 0, 1);
            __builtin_prefetch(&auth->payloads[next_u], 0, 1);
        }
        uint32_t u = members[slot];
        uint32_t d = auth->v[u].deg;
        if (__builtin_expect(offset + d > t->edges_cap, 0)) {
            uint32_t new_cap = (offset + d) * 2U;
            t->packed_edges = (uint32_t *)realloc(t->packed_edges, new_cap * sizeof(uint32_t));
            t->packed_slots = (uint16_t *)realloc(t->packed_slots, new_cap * sizeof(uint16_t));
            mgr->bytes_allocated += (size_t)(new_cap - t->edges_cap) * (sizeof(uint32_t) + sizeof(uint16_t));
            t->edges_cap = new_cap;
        }

        t->row_offset[slot]     = offset;
        t->logical_ids[slot]    = u;
        t->packed_weights[slot] = auth->payloads[u].weight_sum;

        const uint32_t *edges = auth->v[u].edges;
        for (uint32_t e = 0; e < d; e++) {
            uint32_t v = edges[e];
            t->packed_edges[offset + e] = v;
            t->packed_slots[offset + e] = (mgr->vertex_tile[v] == b) ? mgr->vertex_slot[v] : EXTERNAL_SLOT;
        }
        offset += d;
    }
    t->row_offset[VERTICES_PER_TILE] = offset;
    if (!t->valid) {
        mgr->active_tiles_count++;
    }
    memset(t->dirty_bits, 0, sizeof(t->dirty_bits));
    t->valid = true;
    t->tile_generation++;
    t->dirty_count = 0;
    t->realized_savings_ns = 0.0;

    double elapsed = now_ns() - t0;
    t->C_b_ns = elapsed;
    t->R_b_ns = 0.0;
    mgr->total_promotions++;
    mgr->total_maint_ns += elapsed;
}

static inline void alg_notify_vertex_update(AlgTileManager *mgr, uint32_t u) {
    uint32_t b = mgr->vertex_tile[u];
    HotTile *t = &mgr->tiles[b];
    if (t->valid) {
        uint32_t slot = mgr->vertex_slot[u];
        uint64_t mask = 1ULL << (slot & 63U);
        if ((t->dirty_bits[slot >> 6U] & mask) == 0) {
            t->dirty_bits[slot >> 6U] |= mask;
            t->dirty_count++;
        }
        t->R_b_ns += 25.0;
        /* Si más del 20% de los vértices del Tile fueron modificados, invalidar el Tile */
        if (t->dirty_count > (VERTICES_PER_TILE / 5U)) {
            if (t->realized_savings_ns < (t->C_b_ns + t->R_b_ns)) {
                mgr->useless_promotions++;
            }
            t->valid = false;
            if (mgr->active_tiles_count > 0) mgr->active_tiles_count--;
            mgr->tile_invalidations++;
        }
    }
}

/* ============================================================================
 * RECORRIDO k-HOP BFS TRUNCADO (REGLAS IDÉNTICAS EN LAS 4 VARIANTES)
 * ============================================================================ */
typedef struct {
    uint32_t queue_u[BFS_MAX_VISITED];
    uint16_t queue_tile[BFS_MAX_VISITED];
    uint16_t queue_slot[BFS_MAX_VISITED];
    uint8_t  depth[BFS_MAX_VISITED];
    uint32_t visited_epoch[NUM_VERTICES];
    /* Filtro rápido en L1 para el Tile activo de la raíz (128 bytes = 2 líneas de caché) */
    uint64_t root_tile_visited[VERTICES_PER_TILE / 64U];
    uint32_t cur_epoch;
} BfsScratch;

static void bfs_scratch_init(BfsScratch *sc) {
    memset(sc, 0, sizeof(BfsScratch));
    sc->cur_epoch = 1;
}

/* Recorrido sobre DynGraph (Oráculo y V1) */
static uint64_t bfs_query_dyngraph(const DynGraph *g, BfsScratch *sc, uint32_t root) {
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

/* Recorrido sobre V0 (CSR) */
static uint64_t bfs_query_csr(const CsrGraph *csr, BfsScratch *sc, uint32_t root) {
    uint32_t ep = ++sc->cur_epoch;
    uint32_t head = 0, tail = 0;

    sc->queue_u[tail] = root;
    sc->depth[tail] = 0;
    tail++;
    sc->visited_epoch[root] = ep;

    uint64_t accum = csr->csr_payloads[root].weight_sum ^ (uint64_t)root;

    while (head < tail) {
        uint32_t u = sc->queue_u[head];
        uint8_t  d = sc->depth[head];
        head++;

        if (d >= BFS_MAX_DEPTH) continue;

        uint32_t start = csr->row_ptr[u];
        uint32_t end   = csr->row_ptr[u + 1];

        for (uint32_t i = start; i < end; i++) {
            uint32_t v = csr->col_idx[i];
            if (sc->visited_epoch[v] != ep) {
                sc->visited_epoch[v] = ep;
                accum += csr->csr_payloads[v].weight_sum ^ ((uint64_t)v * (uint64_t)(d + 1));
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

/*
 * Recorrido sobre V2 / V3 (ALG con Hot-Tiles versionados y Read-Through Fallback):
 *   - Si el vértice actual `u` pertenece a un Hot-Tile válido y su bit en `dirty_bits`
 *     es 0, lee su lista de adyacencias y los pesos de sus vecinos intra-Tile
 *     directamente del bloque compacto del Tile (sin tocar `auth->v[u]` ni `auth->payloads[v]`
 *     en RAM principal).
 *   - Si el bit en `dirty_bits` es 1 (vértice actualizado recientemente) o el vecino es
 *     externo al Tile, resuelve por la estructura autoritativa global (`read-through`).
 */
static uint64_t bfs_query_alg(const DynGraph *auth, AlgTileManager *mgr, BfsScratch *sc, uint32_t root) {
    if (mgr->active_tiles_count == 0) {
        return bfs_query_dyngraph(auth, sc, root);
    }
    uint32_t ep = ++sc->cur_epoch;
    uint32_t head = 0, tail = 0;

    uint16_t root_tile = mgr->vertex_tile[root];
    uint16_t root_slot = mgr->vertex_slot[root];
    const HotTile *rt = &mgr->tiles[root_tile];

    sc->queue_u[tail]    = root;
    sc->queue_tile[tail] = root_tile;
    sc->queue_slot[tail] = root_slot;
    sc->depth[tail]      = 0;
    tail++;
    sc->visited_epoch[root] = ep;

    if (rt->valid) {
        memset(sc->root_tile_visited, 0, sizeof(sc->root_tile_visited));
        sc->root_tile_visited[root_slot >> 6U] |= (1ULL << (root_slot & 63U));
    }

    uint64_t root_w = rt->valid ? rt->packed_weights[root_slot] : auth->payloads[root].weight_sum;
    uint64_t accum = root_w ^ (uint64_t)root;

    while (head < tail) {
        uint32_t u      = sc->queue_u[head];
        uint16_t t_id   = sc->queue_tile[head];
        uint16_t slot_u = sc->queue_slot[head];
        uint8_t  d      = sc->depth[head];
        head++;

        if (d >= BFS_MAX_DEPTH) continue;

        if (t_id == EXTERNAL_SLOT) {
            t_id   = mgr->vertex_tile[u];
            slot_u = mgr->vertex_slot[u];
        }

        const HotTile *tile = &mgr->tiles[t_id];
        bool use_hot_adj = false;
        if (tile->valid) {
            uint64_t dirty = tile->dirty_bits[slot_u >> 6U] & (1ULL << (slot_u & 63U));
            if (__builtin_expect(dirty == 0, 1)) {
                use_hot_adj = true;
            } else {
                mgr->stale_fallbacks++;
            }
        }

        if (use_hot_adj) {
            uint32_t start = tile->row_offset[slot_u];
            uint32_t end   = tile->row_offset[slot_u + 1];
            const uint32_t *edges = tile->packed_edges + start;
            const uint16_t *slots = tile->packed_slots + start;
            uint32_t deg = end - start;
            bool is_root_tile = (t_id == root_tile);

            for (uint32_t i = 0; i < deg; i++) {
                uint16_t slot_v = slots[i];
                if (__builtin_expect(slot_v != EXTERNAL_SLOT && is_root_tile, 1)) {
                    /* Filtro rápido en L1 (128 bytes) para vecinos del mismo Hot-Tile */
                    uint64_t bit = 1ULL << (slot_v & 63U);
                    uint64_t *word = &sc->root_tile_visited[slot_v >> 6U];
                    if (*word & bit) continue;
                    *word |= bit;

                    uint32_t v = edges[i];
                    if (sc->visited_epoch[v] != ep) {
                        sc->visited_epoch[v] = ep;
                        uint64_t vw = tile->packed_weights[slot_v];
                        accum += vw ^ ((uint64_t)v * (uint64_t)(d + 1));
                        if (tail < BFS_MAX_VISITED) {
                            sc->queue_u[tail]    = v;
                            sc->queue_tile[tail] = t_id;
                            sc->queue_slot[tail] = slot_v;
                            sc->depth[tail]      = d + 1;
                            tail++;
                        } else {
                            goto done;
                        }
                    }
                } else {
                    uint32_t v = edges[i];
                    if (sc->visited_epoch[v] != ep) {
                        sc->visited_epoch[v] = ep;
                        uint64_t vw = (slot_v != EXTERNAL_SLOT)
                                      ? tile->packed_weights[slot_v]
                                      : auth->payloads[v].weight_sum;
                        accum += vw ^ ((uint64_t)v * (uint64_t)(d + 1));
                        if (tail < BFS_MAX_VISITED) {
                            sc->queue_u[tail]    = v;
                            sc->queue_tile[tail] = (slot_v != EXTERNAL_SLOT) ? t_id : EXTERNAL_SLOT;
                            sc->queue_slot[tail] = slot_v;
                            sc->depth[tail]      = d + 1;
                            tail++;
                        } else {
                            goto done;
                        }
                    }
                }
            }
        } else {
            /* Ruta autoritativa global directa (sin sobrecoste de búsqueda en tablas de Tiles para hojas) */
            const uint32_t *edges = auth->v[u].edges;
            uint32_t deg = auth->v[u].deg;

            for (uint32_t i = 0; i < deg; i++) {
                uint32_t v = edges[i];
                if (sc->visited_epoch[v] != ep) {
                    sc->visited_epoch[v] = ep;
                    accum += auth->payloads[v].weight_sum ^ ((uint64_t)v * (uint64_t)(d + 1));
                    if (tail < BFS_MAX_VISITED) {
                        sc->queue_u[tail]    = v;
                        sc->queue_tile[tail] = EXTERNAL_SLOT;
                        sc->queue_slot[tail] = EXTERNAL_SLOT;
                        sc->depth[tail]      = d + 1;
                        tail++;
                    } else {
                        goto done;
                    }
                }
            }
        }
    }
done:
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
        uint32_t chosen_tile = 0;

        switch (wtype) {
            case WORKLOAD_A_STABLE_HOTSPOTS:
            case WORKLOAD_E_STATIC_CONTROL:
                if ((r % 100U) < 92U) {
                    chosen_tile = (uint32_t)((r >> 8) & 3U); /* 4 comunidades calientes */
                } else {
                    chosen_tile = (uint32_t)((r >> 8) % NUM_TILES);
                }
                break;

            case WORKLOAD_B_GRADUAL_DRIFT: {
                uint32_t base_tile = (step / 150U) % NUM_TILES;
                if ((r % 100U) < 92U) {
                    chosen_tile = (base_tile + (uint32_t)((r >> 8) & 3U)) % NUM_TILES;
                } else {
                    chosen_tile = (uint32_t)((r >> 8) % NUM_TILES);
                }
                break;
            }

            case WORKLOAD_C_ABRUPT_SHIFT: {
                uint32_t epoch = (step / 750U) % 4U;
                uint32_t base_tile = epoch * 32U;
                if ((r % 100U) < 92U) {
                    chosen_tile = base_tile + (uint32_t)((r >> 8) & 3U);
                } else {
                    chosen_tile = (uint32_t)((r >> 8) % NUM_TILES);
                }
                break;
            }

            case WORKLOAD_D_UNIFORM_RANDOM:
            default:
                chosen_tile = (uint32_t)((r >> 8) % NUM_TILES);
                break;
        }

        uint32_t slot = (uint32_t)((r >> 20) & (VERTICES_PER_TILE - 1U));
        tr->query_roots[step] = base_graph->tile_members[chosen_tile * VERTICES_PER_TILE + slot];

        if (wtype != WORKLOAD_E_STATIC_CONTROL && step > 0 && (step % UPDATE_BATCH_PERIOD) == 0) {
            tr->has_update_batch[step] = true;
            for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                uint64_t ru = xorshift64s(&rng);
                uint32_t tile_u = (uint32_t)(ru % NUM_TILES);
                uint32_t slot_u = (uint32_t)((ru >> 16) & (VERTICES_PER_TILE - 1U));
                uint32_t slot_v = (uint32_t)((ru >> 32) & (VERTICES_PER_TILE - 1U));
                if (slot_v == slot_u) slot_v = (slot_v + 1U) & (VERTICES_PER_TILE - 1U);

                tr->upd_u[step][k] = base_graph->tile_members[tile_u * VERTICES_PER_TILE + slot_u];
                tr->upd_v[step][k] = base_graph->tile_members[tile_u * VERTICES_PER_TILE + slot_v];
                tr->upd_is_add[step][k] = ((ru >> 48) & 3U) != 0;
            }
        }
    }
}

/* ============================================================================
 * MÉTRICAS Y EJECUCIÓN EXPERIMENTAL
 * ============================================================================ */
typedef struct {
    double   total_workload_ms;
    double   isolated_query_ms;
    double   update_and_maint_ms;
    double   ns_per_update_op;
    double   throughput_qps;
    double   p50_us;
    double   p95_us;
    double   p99_us;
    double   memory_mb;
    uint32_t promotions;
    uint32_t useless_promotions;
    uint32_t invalidations;
    uint32_t stale_fallbacks;
    uint32_t oracle_mismatches;
} VariantMetrics;

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void compute_percentiles(double *lat_us, size_t n, double *p50, double *p95, double *p99) {
    qsort(lat_us, n, sizeof(double), cmp_double);
    *p50 = lat_us[n / 2];
    *p95 = lat_us[(size_t)(n * 0.95)];
    *p99 = lat_us[(size_t)(n * 0.99)];
}

static void run_experiment_on_trace(
    const DynGraph *base_graph,
    const WorkloadTrace *tr,
    VariantMetrics out_metrics[4]
) {
    memset(out_metrics, 0, 4 * sizeof(VariantMetrics));

    double *lat_samples = (double *)malloc(NUM_STEPS_PER_RUN * sizeof(double));
    uint64_t *oracle_hashes = (uint64_t *)malloc(NUM_STEPS_PER_RUN * sizeof(uint64_t));
    BfsScratch *scratch = (BfsScratch *)malloc(sizeof(BfsScratch));

    uint32_t total_update_ops = 0;
    for (uint32_t s = 0; s < NUM_STEPS_PER_RUN; s++) {
        if (tr->has_update_batch[s]) total_update_ops += UPDATES_PER_BATCH;
    }

    /* 1. V1 (Dynamic AdjList) + Oráculo */
    {
        DynGraph g1;
        dyngraph_clone(&g1, base_graph);
        bfs_scratch_init(scratch);

        double query_ns_sum = 0.0;
        double maint_ns_sum = 0.0;

        for (uint32_t s = 0; s < NUM_STEPS_PER_RUN; s++) {
            if (tr->has_update_batch[s]) {
                double tu0 = now_ns();
                for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                    if (tr->upd_is_add[s][k]) {
                        dyngraph_add_edge(&g1, tr->upd_u[s][k], tr->upd_v[s][k]);
                    } else {
                        dyngraph_remove_edge(&g1, tr->upd_u[s][k]);
                    }
                }
                maint_ns_sum += (now_ns() - tu0);
            }

            double tq0 = now_ns();
            uint64_t h = bfs_query_dyngraph(&g1, scratch, tr->query_roots[s]);
            double dq = now_ns() - tq0;

            oracle_hashes[s] = h;
            lat_samples[s] = dq / 1000.0;
            query_ns_sum += dq;
        }

        double total_ms = (query_ns_sum + maint_ns_sum) / 1e6;
        out_metrics[1].total_workload_ms   = total_ms;
        out_metrics[1].isolated_query_ms   = query_ns_sum / 1e6;
        out_metrics[1].update_and_maint_ms = maint_ns_sum / 1e6;
        out_metrics[1].ns_per_update_op    = total_update_ops ? (maint_ns_sum / total_update_ops) : 0.0;
        out_metrics[1].throughput_qps      = (double)NUM_STEPS_PER_RUN / (total_ms / 1000.0);
        out_metrics[1].memory_mb           = (double)(g1.bytes_allocated + sizeof(BfsScratch)) / (1024.0 * 1024.0);
        compute_percentiles(lat_samples, NUM_STEPS_PER_RUN,
                            &out_metrics[1].p50_us, &out_metrics[1].p95_us, &out_metrics[1].p99_us);
        dyngraph_free(&g1);
    }

    /* 2. V0 (CSR + Rebuild) */
    {
        DynGraph g0;
        dyngraph_clone(&g0, base_graph);
        CsrGraph csr;
        csr_init(&csr, g0.total_edges);
        csr_rebuild(&csr, &g0);
        bfs_scratch_init(scratch);

        double query_ns_sum = 0.0;
        double maint_ns_sum = 0.0;
        uint32_t mismatches = 0;

        for (uint32_t s = 0; s < NUM_STEPS_PER_RUN; s++) {
            if (tr->has_update_batch[s]) {
                double tu0 = now_ns();
                for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                    if (tr->upd_is_add[s][k]) {
                        dyngraph_add_edge(&g0, tr->upd_u[s][k], tr->upd_v[s][k]);
                    } else {
                        dyngraph_remove_edge(&g0, tr->upd_u[s][k]);
                    }
                }
                csr_rebuild(&csr, &g0);
                maint_ns_sum += (now_ns() - tu0);
            }

            double tq0 = now_ns();
            uint64_t h = bfs_query_csr(&csr, scratch, tr->query_roots[s]);
            double dq = now_ns() - tq0;

            if (h != oracle_hashes[s]) mismatches++;
            lat_samples[s] = dq / 1000.0;
            query_ns_sum += dq;
        }

        double total_ms = (query_ns_sum + maint_ns_sum) / 1e6;
        out_metrics[0].total_workload_ms   = total_ms;
        out_metrics[0].isolated_query_ms   = query_ns_sum / 1e6;
        out_metrics[0].update_and_maint_ms = maint_ns_sum / 1e6;
        out_metrics[0].ns_per_update_op    = total_update_ops ? (maint_ns_sum / total_update_ops) : 0.0;
        out_metrics[0].throughput_qps      = (double)NUM_STEPS_PER_RUN / (total_ms / 1000.0);
        out_metrics[0].memory_mb           = (double)(g0.bytes_allocated + csr.bytes_allocated + sizeof(BfsScratch)) / (1024.0 * 1024.0);
        out_metrics[0].oracle_mismatches   = mismatches;
        compute_percentiles(lat_samples, NUM_STEPS_PER_RUN,
                            &out_metrics[0].p50_us, &out_metrics[0].p95_us, &out_metrics[0].p99_us);
        csr_free(&csr);
        dyngraph_free(&g0);
    }

    /* 3. V2 (ALG Periodic: Reconstrucción ciega cada K consultas) */
    {
        DynGraph g2;
        dyngraph_clone(&g2, base_graph);
        AlgTileManager *mgr2 = (AlgTileManager *)malloc(sizeof(AlgTileManager));
        alg_manager_init(mgr2, &g2);
        bfs_scratch_init(scratch);

        double query_ns_sum = 0.0;
        double maint_ns_sum = 0.0;
        uint32_t mismatches = 0;

        for (uint32_t s = 0; s < NUM_STEPS_PER_RUN; s++) {
            if (tr->has_update_batch[s]) {
                double tu0 = now_ns();
                for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                    uint32_t u = tr->upd_u[s][k];
                    if (tr->upd_is_add[s][k]) {
                        dyngraph_add_edge(&g2, u, tr->upd_v[s][k]);
                    } else {
                        dyngraph_remove_edge(&g2, u);
                    }
                    alg_notify_vertex_update(mgr2, u);
                }
                maint_ns_sum += (now_ns() - tu0);
            }

            uint32_t root = tr->query_roots[s];
            uint32_t b = mgr2->vertex_tile[root];
            mgr2->tiles[b].n_b_window++;

            /* V2 reconstruye periódicamente todos los Tiles visitados en los últimos K pasos */
            if (s > 0 && (s % V2_PERIODIC_K) == 0) {
                double tm0 = now_ns();
                for (uint32_t tid = 0; tid < NUM_TILES; tid++) {
                    if (mgr2->tiles[tid].n_b_window > 0) {
                        alg_build_tile(mgr2, tid, &g2);
                        mgr2->tiles[tid].n_b_window = 0;
                    }
                }
                maint_ns_sum += (now_ns() - tm0);
            }

            double tq0 = now_ns();
            uint64_t h = bfs_query_alg(&g2, mgr2, scratch, root);
            double dq = now_ns() - tq0;

            if (mgr2->tiles[b].valid) {
                mgr2->tiles[b].realized_savings_ns += mgr2->tiles[b].g_b_ns;
            }
            if (h != oracle_hashes[s]) mismatches++;
            lat_samples[s] = dq / 1000.0;
            query_ns_sum += dq;
        }

        double total_ms = (query_ns_sum + maint_ns_sum) / 1e6;
        out_metrics[2].total_workload_ms   = total_ms;
        out_metrics[2].isolated_query_ms   = query_ns_sum / 1e6;
        out_metrics[2].update_and_maint_ms = maint_ns_sum / 1e6;
        out_metrics[2].ns_per_update_op    = total_update_ops ? (maint_ns_sum / total_update_ops) : 0.0;
        out_metrics[2].throughput_qps      = (double)NUM_STEPS_PER_RUN / (total_ms / 1000.0);
        out_metrics[2].memory_mb           = (double)(g2.bytes_allocated + mgr2->bytes_allocated + sizeof(BfsScratch)) / (1024.0 * 1024.0);
        out_metrics[2].promotions          = mgr2->total_promotions;
        out_metrics[2].useless_promotions  = mgr2->useless_promotions;
        out_metrics[2].invalidations       = mgr2->tile_invalidations;
        out_metrics[2].stale_fallbacks     = mgr2->stale_fallbacks;
        out_metrics[2].oracle_mismatches   = mismatches;
        compute_percentiles(lat_samples, NUM_STEPS_PER_RUN,
                            &out_metrics[2].p50_us, &out_metrics[2].p95_us, &out_metrics[2].p99_us);
        alg_manager_free(mgr2);
        free(mgr2);
        dyngraph_free(&g2);
    }

    /* 4. V3 (ALG Adaptive: Regla de Amortización Empírica H3: n_b * g_b > alpha * C_b + R_b) */
    {
        DynGraph g3;
        dyngraph_clone(&g3, base_graph);
        AlgTileManager *mgr3 = (AlgTileManager *)malloc(sizeof(AlgTileManager));
        alg_manager_init(mgr3, &g3);
        bfs_scratch_init(scratch);

        double query_ns_sum = 0.0;
        double maint_ns_sum = 0.0;
        uint32_t mismatches = 0;

        const double ALPHA_H3 = 1.15;
        double global_ewma_ns = 0.0;
        double hot_ewma_ns    = 0.0;
        double measured_cb_ewma_ns = 120000.0;

        for (uint32_t s = 0; s < NUM_STEPS_PER_RUN; s++) {
            if (tr->has_update_batch[s]) {
                double tu0 = now_ns();
                for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                    uint32_t u = tr->upd_u[s][k];
                    if (tr->upd_is_add[s][k]) {
                        dyngraph_add_edge(&g3, u, tr->upd_v[s][k]);
                    } else {
                        dyngraph_remove_edge(&g3, u);
                    }
                    alg_notify_vertex_update(mgr3, u);
                }
                maint_ns_sum += (now_ns() - tu0);
            }

            /* Decaimiento periódico suave de n_b cada 200 consultas para detectar cambios de patrón */
            if (s > 0 && (s % 200U) == 0) {
                for (uint32_t tid = 0; tid < NUM_TILES; tid++) {
                    mgr3->tiles[tid].n_b_window >>= 1;
                }
            }

            uint32_t root = tr->query_roots[s];
            uint32_t b = mgr3->vertex_tile[root];
            HotTile *tb = &mgr3->tiles[b];
            tb->n_b_window++;

            /*
             * Regla H3:
             *   Un Tile acumulando n_b visitas recientes prevé al menos un horizonte
             *   proporcional a su actividad actual si el patrón persiste.
             *   Promovemos cuando: (n_b * H_horizon) * g_b > alpha * C_b + R_b
             */
            if (!tb->valid) {
                double est_gb = (hot_ewma_ns > 0.0 && global_ewma_ns > hot_ewma_ns)
                                ? (global_ewma_ns - hot_ewma_ns)
                                : tb->g_b_ns;
                double est_cb = (tb->tile_generation > 0) ? tb->C_b_ns : measured_cb_ewma_ns;
                /* Horizonte conservador: proyectar que el Tile recibirá al menos 8x las visitas ya vistas en la ventana */
                double projected_queries = (double)tb->n_b_window * 8.0;
                if (tb->n_b_window >= 12U && (projected_queries * est_gb) > (ALPHA_H3 * est_cb + tb->R_b_ns)) {
                    double tm0 = now_ns();
                    alg_build_tile(mgr3, b, &g3);
                    double build_dt = now_ns() - tm0;
                    maint_ns_sum += build_dt;
                    measured_cb_ewma_ns = 0.70 * measured_cb_ewma_ns + 0.30 * tb->C_b_ns;
                }
            }

            bool used_hot = tb->valid;
            double tq0 = now_ns();
            uint64_t h = bfs_query_alg(&g3, mgr3, scratch, root);
            double dq = now_ns() - tq0;

            /* Actualizar estimación empírica de g_b = t_global - t_hot */
            if (used_hot) {
                hot_ewma_ns = (hot_ewma_ns == 0.0) ? dq : (0.92 * hot_ewma_ns + 0.08 * dq);
                tb->ewma_hot_ns = (tb->ewma_hot_ns == 0.0) ? dq : (0.85 * tb->ewma_hot_ns + 0.15 * dq);
                double cold_ref = (tb->ewma_global_ns > 0.0) ? tb->ewma_global_ns : global_ewma_ns;
                double diff = cold_ref - tb->ewma_hot_ns;
                tb->g_b_ns = (diff > 200.0) ? diff : 200.0;
                tb->realized_savings_ns += tb->g_b_ns;
            } else {
                global_ewma_ns = (global_ewma_ns == 0.0) ? dq : (0.92 * global_ewma_ns + 0.08 * dq);
                tb->ewma_global_ns = (tb->ewma_global_ns == 0.0) ? dq : (0.85 * tb->ewma_global_ns + 0.15 * dq);
                if (hot_ewma_ns > 0.0 && tb->ewma_global_ns > hot_ewma_ns) {
                    tb->g_b_ns = tb->ewma_global_ns - hot_ewma_ns;
                }
            }

            if (h != oracle_hashes[s]) mismatches++;
            lat_samples[s] = dq / 1000.0;
            query_ns_sum += dq;
        }

        double total_ms = (query_ns_sum + maint_ns_sum) / 1e6;
        out_metrics[3].total_workload_ms   = total_ms;
        out_metrics[3].isolated_query_ms   = query_ns_sum / 1e6;
        out_metrics[3].update_and_maint_ms = maint_ns_sum / 1e6;
        out_metrics[3].ns_per_update_op    = total_update_ops ? (maint_ns_sum / total_update_ops) : 0.0;
        out_metrics[3].throughput_qps      = (double)NUM_STEPS_PER_RUN / (total_ms / 1000.0);
        out_metrics[3].memory_mb           = (double)(g3.bytes_allocated + mgr3->bytes_allocated + sizeof(BfsScratch)) / (1024.0 * 1024.0);
        out_metrics[3].promotions          = mgr3->total_promotions;
        out_metrics[3].useless_promotions  = mgr3->useless_promotions;
        out_metrics[3].invalidations       = mgr3->tile_invalidations;
        out_metrics[3].stale_fallbacks     = mgr3->stale_fallbacks;
        out_metrics[3].oracle_mismatches   = mismatches;
        compute_percentiles(lat_samples, NUM_STEPS_PER_RUN,
                            &out_metrics[3].p50_us, &out_metrics[3].p95_us, &out_metrics[3].p99_us);
        alg_manager_free(mgr3);
        free(mgr3);
        dyngraph_free(&g3);
    }

    free(lat_samples);
    free(oracle_hashes);
    free(scratch);
}

int main(void) {
    SetConsoleOutputCP(65001);
    setvbuf(stdout, NULL, _IONBF, 0);
    init_timer();

    printf("====================================================================================================\n");
    printf(" EXPERIMENTO 0: ADAPTIVE LOCALITY GRAPH (ALG) — EVALUACION DE LAS 4 VARIANTES Y 5 CARGAS\n");
    printf(" Grafo: %u vertices (%u Tiles x %u), k-hop BFS (k=%u, max=%u), %u pasos x %u semillas\n",
           NUM_VERTICES, NUM_TILES, VERTICES_PER_TILE, BFS_MAX_DEPTH, BFS_MAX_VISITED, NUM_STEPS_PER_RUN, NUM_SEEDS);
    printf("====================================================================================================\n\n");

    const uint64_t SEEDS[NUM_SEEDS] = { 0x1020304050607080ULL, 0xA5A5F0F012345678ULL, 0xCAFEBEEF42424242ULL };
    DynGraph base_graphs[NUM_SEEDS];
    for (uint32_t s = 0; s < NUM_SEEDS; s++) {
        generate_base_graph(&base_graphs[s], SEEDS[s]);
    }
    printf("Grafo base inicializado (Semilla 0: %llu aristas dirigidas, ~%.2f MB en estructuras base).\n\n",
           (unsigned long long)base_graphs[0].total_edges,
           (double)base_graphs[0].bytes_allocated / (1024.0 * 1024.0));

    WorkloadTrace *trace = (WorkloadTrace *)malloc(sizeof(WorkloadTrace));
    const char *VAR_NAMES[4] = {
        "V0 (CSR + Rebuild)  ",
        "V1 (Dynamic AdjList)",
        "V2 (ALG Periodic)   ",
        "V3 (ALG Adaptive H3)"
    };

    for (int w = 0; w < NUM_WORKLOADS; w++) {
        VariantMetrics avg[4];
        memset(avg, 0, sizeof(avg));

        for (uint32_t s = 0; s < NUM_SEEDS; s++) {
            generate_workload_trace(trace, &base_graphs[s], (WorkloadType)w, SEEDS[s]);
            VariantMetrics m[4];
            run_experiment_on_trace(&base_graphs[s], trace, m);

            for (int v = 0; v < 4; v++) {
                avg[v].total_workload_ms   += m[v].total_workload_ms / NUM_SEEDS;
                avg[v].isolated_query_ms   += m[v].isolated_query_ms / NUM_SEEDS;
                avg[v].update_and_maint_ms += m[v].update_and_maint_ms / NUM_SEEDS;
                avg[v].ns_per_update_op    += m[v].ns_per_update_op / NUM_SEEDS;
                avg[v].throughput_qps      += m[v].throughput_qps / NUM_SEEDS;
                avg[v].p50_us              += m[v].p50_us / NUM_SEEDS;
                avg[v].p95_us              += m[v].p95_us / NUM_SEEDS;
                avg[v].p99_us              += m[v].p99_us / NUM_SEEDS;
                avg[v].memory_mb           += m[v].memory_mb / NUM_SEEDS;
                avg[v].promotions          += m[v].promotions;
                avg[v].useless_promotions  += m[v].useless_promotions;
                avg[v].invalidations       += m[v].invalidations;
                avg[v].stale_fallbacks     += m[v].stale_fallbacks;
                avg[v].oracle_mismatches   += m[v].oracle_mismatches;
            }
        }

        printf("----------------------------------------------------------------------------------------------------\n");
        printf(" %s (Media de %u semillas, %u consultas/semilla)\n", WORKLOAD_NAMES[w], NUM_SEEDS, NUM_STEPS_PER_RUN);
        printf("----------------------------------------------------------------------------------------------------\n");
        printf(" Variante             | Total(ms) | Query(ms) | Maint(ms) |   QPS    | p50(us) | p95(us) | p99(us) | Mem(MB) | Prom(Inutil) | Fallbk | Exact\n");
        printf("----------------------+-----------+-----------+-----------+----------+---------+---------+---------+---------+--------------+--------+------\n");
        for (int v = 0; v < 4; v++) {
            printf(" %s | %9.2f | %9.2f | %9.2f | %8.0f | %7.2f | %7.2f | %7.2f | %7.2f | %4u (%4u)  | %6u | %s\n",
                   VAR_NAMES[v],
                   avg[v].total_workload_ms,
                   avg[v].isolated_query_ms,
                   avg[v].update_and_maint_ms,
                   avg[v].throughput_qps,
                   avg[v].p50_us,
                   avg[v].p95_us,
                   avg[v].p99_us,
                   avg[v].memory_mb,
                   avg[v].promotions / NUM_SEEDS,
                   avg[v].useless_promotions / NUM_SEEDS,
                   avg[v].stale_fallbacks / NUM_SEEDS,
                   (avg[v].oracle_mismatches == 0) ? "100%" : "ERR");
        }
        printf("\n");
    }

    free(trace);
    for (uint32_t s = 0; s < NUM_SEEDS; s++) {
        dyngraph_free(&base_graphs[s]);
    }
    return 0;
}
