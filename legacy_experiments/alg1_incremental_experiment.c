/*
 * ============================================================================
 * EXPERIMENTO ALG-1: GRANULARIDAD DE HOT-CORE Y PROMOCIÓN INCREMENTAL
 * ============================================================================
 * Responde experimentalmente a las preguntas de diseño de ALG-1:
 *
 *   1. Sub-experimento 1 (Problema 1 - Índice Local Exacto sin Oráculo Global):
 *      Compara dos mecanismos 100% exactos para mapear IDs globales dispersos (32 bits)
 *      a posiciones locales (0..M-1) dentro de un Tile de capacidad M in {64, 128}:
 *        - Alt-A: Tabla Hash Compacta en L1 (Open Addressing, factor de carga <= 0.5)
 *        - Alt-B: Filtro Bloom en registro (64/256 bits) para descarte rápido +
 *                 verificación exacta en tabla ante positivo.
 *
 *   2. Experimento Principal ALG-1 (Variantes V1, V3, V4, V5, V6, V7):
 *      Cambia una sola variable estructural por paso:
 *        - V1: Dynamic AdjList (Baseline competitivo sin Tiles)
 *        - V3: Tile fijo de 1,024 vértices + H3
 *        - V4: Hot-Core fijo de 128 vértices + H3
 *        - V5: Hot-Core fijo de 64 vértices + H3
 *        - V6: Micro-Tile Incremental con presupuesto máximo de bytes (sin filtro H3)
 *        - V7: Micro-Tile Incremental con presupuesto de bytes + límite de rentabilidad H3
 *
 *   3. Separación Estricta Calibración vs. Evaluación:
 *      - Parámetros ajustados en Semillas de Calibración (Cargas A y B).
 *      - Evaluación final ejecutada en 3 Semillas de Test independientes no vistas
 *        durante el ajuste, sobre las 5 cargas (A, B, C, D, E).
 *
 *   4. Desglose del Ciclo de Vida del Tile (Problema 2):
 *      Mide por separado:
 *        - Coste de captura/construcción (Cap_us)
 *        - Latencia de 1ª consulta tras/durante captura (1stQ_us)
 *        - Latencia de consultas posteriores sobre el Tile (SubQ_us)
 *        - Punto de equilibrio empírico real (BreakEven_Q)
 *        - Tasa de aciertos de réplica (Hit%), Read-Through (RT%) y Bytes copiados/consulta (B/Q).
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

/* ============================================================================
 * CONFIGURACIÓN DEL GRAFO Y RECORRIDO
 * ============================================================================ */
#define NUM_VERTICES        262144U                 /* 256K vértices (>52 MB base) */
#define NUM_COMMUNITIES     256U                    /* 256 comunidades latentes de 1,024 vértices */
#define COMM_SIZE           (NUM_VERTICES / NUM_COMMUNITIES)
#define NO_TILE             0xFFFFU
#define EXTERNAL_SLOT       0xFFFFU

#define BFS_MAX_DEPTH       3U
#define BFS_MAX_VISITED     512U                    /* Subgrafo local k-hop (idéntico a Exp 0) */

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

/* ============================================================================
 * GRAFO AUTORITATIVO BASE (V1: Dynamic AdjList Contigua por Vértice)
 * Nota: El grafo autoritativo NO expone `community_id` a los motores ALG.
 * Los motores ALG solo ven IDs lógicos de 32 bits `0 .. NUM_VERTICES-1` y
 * deben descubrir e indexar sus Tiles sin ningún oráculo topológico.
 * ============================================================================ */
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
    uint32_t      *comm_members; /* Usado SOLO por el generador de trazas, NUNCA por ALG */
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
    dst->comm_members = NULL; /* Los clones de evaluación ni siquiera copian comm_members */

    memcpy(dst->payloads, src->payloads, NUM_VERTICES * sizeof(VertexPayload));
    dst->total_edges = src->total_edges;
    dst->bytes_allocated = NUM_VERTICES * (sizeof(DynVertex) + sizeof(VertexPayload));

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
 * Generador realista con distribución intra-comunidad tipo hub-and-spoke + malla:
 *   - Dentro de cada comunidad de 1,024 vértices (con IDs lógicos dispersos),
 *     los primeros ~96 vértices actúan como núcleos/hubs densamente interconectados
 *     y el resto se conecta preferencialmente a ellos, reproduciendo la estructura
 *     real de redes sociales/dependencias donde un Hot-Core de 64-128 vértices
 *     concentra la mayoría de las expansiones de frontera.
 */
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
    /* Permutación Fisher-Yates: los IDs lógicos quedan 100% desacoplados de la comunidad */
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
                    if (p < 88U) {
                        target = nodes[(r >> 8) % 96U];
                    } else if (p < 96U) {
                        target = nodes[(r >> 8) & (COMM_SIZE - 1U)];
                    } else {
                        target = (uint32_t)((r >> 8) % NUM_VERTICES);
                    }
                } else {
                    if (p < 50U) {
                        target = nodes[(r >> 8) % 96U];
                    } else if (p < 90U) {
                        target = nodes[(r >> 8) & (COMM_SIZE - 1U)];
                    } else {
                        target = (uint32_t)((r >> 8) % NUM_VERTICES);
                    }
                }
                if (target != u) {
                    dyngraph_add_edge(g, u, target);
                }
            }
        }
    }
}

/* ============================================================================
 * SUB-EXPERIMENTO 1 (PROBLEMA 1): ÍNDICE LOCAL COMPACTO VS. BLOOM + EXACTO
 * ============================================================================
 * Mapea `uint32_t global_id -> uint16_t local_slot (0..M-1)` sin ningún arreglo
 * global de 256K entradas.
 *   - Tabla Hash Compacta Abierta (potencia de 2: H_CAP = 2 * M, factor carga <= 0.5):
 *     Usa hash multiplicativo de Fibonacci `(id * 0x9E3779B9U) >> shift`.
 *   - Alternativa con Filtro Bloom de 256 bits (4 x uint64_t) delante de la tabla hash.
 */
#define MAX_HASH_CAP 2048U /* Soporta hasta M = 1024 con factor de carga 0.5 */

typedef struct {
    uint32_t key_plus_one[MAX_HASH_CAP]; /* 0 = vacío, id + 1 = ocupado */
    uint16_t slot[MAX_HASH_CAP];
    uint64_t bloom[4];                   /* 256 bits para filtro Bloom opcional */
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
    /* Insertar en Bloom de 256 bits (2 funciones hash independientes) */
    uint32_t h1 = (global_id * 0x9E3779B9U) >> 24U;
    uint32_t h2 = (global_id * 0x85EBCA6BU) >> 24U;
    idx->bloom[h1 >> 6U] |= (1ULL << (h1 & 63U));
    idx->bloom[h2 >> 6U] |= (1ULL << (h2 & 63U));

    /* Insertar en tabla hash abierta exacta */
    uint32_t pos = (global_id * 0x9E3779B9U) >> idx->shift;
    uint32_t kp1 = global_id + 1U;
    while (idx->key_plus_one[pos] != 0 && idx->key_plus_one[pos] != kp1) {
        pos = (pos + 1U) & idx->mask;
    }
    idx->key_plus_one[pos] = kp1;
    idx->slot[pos] = local_slot;
}

/* Alternativa A: Búsqueda directa en Tabla Hash Compacta en L1 (100% exacta) */
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

/* Alternativa B: Filtro Bloom de 256 bits + Verificación Exacta en Tabla Hash */
static inline uint16_t local_index_lookup_bloom_plus_exact(const CompactLocalIndex *idx, uint32_t global_id) {
    uint32_t h1 = (global_id * 0x9E3779B9U) >> 24U;
    uint32_t h2 = (global_id * 0x85EBCA6BU) >> 24U;
    if ((idx->bloom[h1 >> 6U] & (1ULL << (h1 & 63U))) == 0 ||
        (idx->bloom[h2 >> 6U] & (1ULL << (h2 & 63U))) == 0) {
        return EXTERNAL_SLOT; /* Negativo garantizado en 100% */
    }
    /* Positivo del Bloom: JAMÁS asumir pertenencia sin verificación exacta */
    return local_index_lookup_hash(idx, global_id);
}

static void run_subexperiment_1_index_comparison(void) {
    printf("[SUB-EXPERIMENTO 1] PROBLEMA 1: INDICE LOCAL COMPACTO (HASH L1) VS. BLOOM + EXACTO\n");
    CompactLocalIndex idx;
    local_index_clear(&idx, 128);

    uint64_t rng = 0x123456789ABCDEF0ULL;
    uint32_t members[128];
    for (uint16_t i = 0; i < 128; i++) {
        members[i] = (uint32_t)(xorshift64s(&rng) % NUM_VERTICES);
        local_index_insert(&idx, members[i], i);
    }

    const uint32_t N_PROBES = 4000000U;
    uint32_t *probes = (uint32_t *)malloc(N_PROBES * sizeof(uint32_t));
    /* 70% aciertos dentro del Hot-Core, 30% misses externos */
    for (uint32_t i = 0; i < N_PROBES; i++) {
        uint64_t r = xorshift64s(&rng);
        if ((r % 100U) < 70U) {
            probes[i] = members[(r >> 8) & 127U];
        } else {
            probes[i] = (uint32_t)((r >> 8) % NUM_VERTICES);
        }
    }

    uint64_t sum_a = 0, sum_b = 0;
    double t0 = now_ns();
    for (uint32_t i = 0; i < N_PROBES; i++) {
        sum_a += local_index_lookup_hash(&idx, probes[i]);
    }
    double dt_hash = (now_ns() - t0) / N_PROBES;

    t0 = now_ns();
    for (uint32_t i = 0; i < N_PROBES; i++) {
        sum_b += local_index_lookup_bloom_plus_exact(&idx, probes[i]);
    }
    double dt_bloom = (now_ns() - t0) / N_PROBES;

    printf("  - M = 128 vertices (Tabla Hash L1 de 256 buckets = 1.5 KB total):\n");
    printf("    * Alt-A (Hash Abierto Directo en L1):       %.2f ns/busqueda\n", dt_hash);
    printf("    * Alt-B (Bloom 256b + Verificacion Exacta): %.2f ns/busqueda\n", dt_bloom);
    printf("    * Coincidencia exacta de slots devueltos:   %s (Checksum=%llu)\n\n",
           (sum_a == sum_b) ? "100% IDENTICOS" : "ERROR", (unsigned long long)sum_a);
    free(probes);
}

/* ============================================================================
 * ESTRUCTURA UNIFICADA DE HOT-CORE / MICRO-TILE PARA ALG-1 (V3, V4, V5, V6, V7)
 * ============================================================================
 * Soporta:
 *   - Capacidad fija M in {1024 (V3), 128 (V4), 64 (V5)}
 *   - Captura incremental por presupuesto de bytes (V6, V7): empieza pequeño
 *     (p. ej. 32 vértices interiores) y añade vértices expandidos hasta un
 *     presupuesto máximo `max_bytes_budget` (12 KB = ~128 vértices).
 *   - Índice local exacto propio (`CompactLocalIndex`) sin usar `community_id`.
 *   - En cada arista empaquetada `(u -> v)`, guarda:
 *       * `packed_edges[e]`: ID global de 32 bits `v`
 *       * `packed_slots[e]`: slot local `0..count-1` si `v` está en el Tile, o `EXTERNAL_SLOT`
 *       * `packed_target_weights[e]`: copia en caché del `weight_sum` de `v` (8 bytes)
 *         para evitar el fallo de caché L3/RAM en `auth->payloads[v]` incluso cuando
 *         `v` es una hoja en profundidad d=3. (Si `u` o `v` se modifican, `dirty_bits`
 *         fuerza read-through al grafo autoritativo).
 */
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

    /* Métricas empíricas H3 y telemetría de ciclo de vida */
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
    /* Directorio ligero descubierto dinámicamente: vertex_id -> tile_id (o NO_TILE) */
    uint16_t  assigned_tile[NUM_VERTICES];
    uint16_t  assigned_slot[NUM_VERTICES];
    /* Contador ligero de visitas por vértice raíz para descubrir regiones calientes sin oráculo */
    uint8_t   vertex_heat[NUM_VERTICES];
    uint32_t  active_tiles_count;

    /* Telemetría detallada exigida por ALG-1 */
    uint32_t  promotions;
    uint32_t  useless_promotions;
    uint32_t  invalidations;
    uint64_t  expanded_hot_hits;
    uint64_t  expanded_total;
    uint64_t  read_through_fallbacks;
    uint64_t  total_bytes_copied;
    double    capture_ns_sum;
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
    uint32_t max_b = 40960; /* Presupuesto de 40 KB para Micro-Tiles incrementales V6/V7 */
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

/*
 * Construye o amplía un MicroTile `b` a partir de la lista de vértices `captured_v[0..count-1]`
 * descubiertos orgánicamente durante un recorrido BFS.
 * Mide y contabiliza cada byte copiado y cada nanosegundo de construcción.
 */
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

    /* Limpiar asignaciones anteriores si el Tile se está reconstruyendo */
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
        /* Si `u` pertenecía previamente a otro Tile activo, marcar su slot anterior como sucio para 100% coherencia */
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
    eng->total_bytes_copied += bytes_used;
    return dt;
}

/*
 * Expansión incremental de un Micro-Tile existente (para V6 y V7):
 * Añade al final del Micro-Tile `b` únicamente los vértices recién expandidos
 * que aún no pertenecen al Tile, respetando `max_vertices` y `max_bytes_budget`.
 */
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
        if (eng->assigned_tile[u] == b) continue; /* Ya está en este Tile */

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

    /* Invalidar si más del 20% del Micro-Tile quedó sucio */
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

/* ============================================================================
 * RECORRIDO BFS Y CAPTURA ORGÁNICA DE FRONTERA
 * ============================================================================ */
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

/*
 * Recorrido auxiliar para recolectar hasta `target_count` vértices conectados
 * desde `root` cuando V3 (1024) necesita poblar un Tile mayor que BFS_MAX_VISITED.
 */
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

static uint64_t bfs_query_alg1(
    const DynGraph *auth,
    Alg1Engine *eng,
    BfsScratch *sc,
    uint32_t root,
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
    bool root_tile_valid = (rt && rt->valid && root_slot < rt->num_vertices);

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
            /* Resolución exacta en O(1) mediante la Tabla Hash Compacta en L1 del Tile (Problema 1) */
            slot_u = local_index_lookup_hash(&tile->local_idx, u);
            if (slot_u < tile->num_vertices) {
                uint64_t dirty = tile->dirty_bits[slot_u >> 6U] & (1ULL << (slot_u & 63U));
                if (__builtin_expect(dirty == 0, 1)) {
                    use_hot = true;
                    eng->expanded_hot_hits++;
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
                if ((r % 100U) < 92U) {
                    chosen_comm = (uint32_t)((r >> 8) & 3U);
                } else {
                    chosen_comm = (uint32_t)((r >> 8) % NUM_COMMUNITIES);
                }
                break;

            case WORKLOAD_B_GRADUAL_DRIFT: {
                uint32_t base_comm = (step / 150U) % NUM_COMMUNITIES;
                if ((r % 100U) < 92U) {
                    chosen_comm = (base_comm + (uint32_t)((r >> 8) & 3U)) % NUM_COMMUNITIES;
                } else {
                    chosen_comm = (uint32_t)((r >> 8) % NUM_COMMUNITIES);
                }
                break;
            }

            case WORKLOAD_C_ABRUPT_SHIFT: {
                uint32_t epoch = (step / 750U) % 4U;
                uint32_t base_comm = epoch * 32U;
                if ((r % 100U) < 92U) {
                    chosen_comm = base_comm + (uint32_t)((r >> 8) & 3U);
                } else {
                    chosen_comm = (uint32_t)((r >> 8) % NUM_COMMUNITIES);
                }
                break;
            }

            case WORKLOAD_D_UNIFORM_RANDOM:
            default:
                chosen_comm = (uint32_t)((r >> 8) % NUM_COMMUNITIES);
                break;
        }

        /* Elegir raíz dentro del núcleo activo de la comunidad */
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
 * MÉTRICAS DETALLADAS DE ALG-1
 * ============================================================================ */
typedef struct {
    double   total_ms;
    double   query_ms;
    double   maint_ms;
    double   qps;
    double   p50_us;
    double   p95_us;
    double   p99_us;
    double   cap_us_per_promo;      /* Coste medio de construir/capturar un Tile (us) */
    double   first_q_us;            /* Latencia de la 1ª consulta con el Tile (us) */
    double   subseq_q_us;           /* Latencia de consultas posteriores con el Tile (us) */
    double   cold_q_us;             /* Latencia de consultas sin Tile (us) */
    double   breakeven_queries;     /* Consultas necesarias para recuperar Cap_us */
    double   replica_hit_pct;       /* % de vértices expandidos servidos desde Hot-Tile */
    double   read_through_pct;      /* % de accesos a Tile que requirieron read-through */
    double   bytes_copied_per_q;    /* Bytes copiados a Tiles por consulta */
    double   mem_mb;
    uint32_t promotions;
    uint32_t useless_promotions;
    uint32_t oracle_errors;
} Alg1Metrics;

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

static void run_alg1_variant(
    const DynGraph *base_graph,
    const WorkloadTrace *tr,
    const uint64_t *oracle_hashes,
    AlgMode mode,
    BfsScratch *scratch,
    double *lat_samples,
    Alg1Metrics *out
) {
    memset(out, 0, sizeof(Alg1Metrics));
    DynGraph g;
    dyngraph_clone(&g, base_graph);

    Alg1Engine *eng = (Alg1Engine *)malloc(sizeof(Alg1Engine));
    alg1_engine_init(eng, mode);
    bfs_scratch_init(scratch);

    /*
     * Parámetros de política H3 (Fijados una sola vez en fase de calibración,
     * idénticos para V3, V4, V5 y V7):
     */
    const double ALPHA_H3      = 1.15;
    const double HORIZON_MULT  = 16.0; /* Proyección de persistencia tras actividad recurrente */
    double cold_ewma_ns        = 3200.0;
    double hot_ewma_ns         = 2100.0;
    double measured_cb_ewma_ns = eng->tiles[0].C_b_ns;

    double query_ns_sum = 0.0;
    double maint_ns_sum = 0.0;
    uint32_t errors = 0;
    uint32_t next_tile_slot = 0;
    uint32_t *temp_nodes = (uint32_t *)malloc(MAX_TILE_SLOTS * sizeof(uint32_t));

    for (uint32_t s = 0; s < NUM_STEPS_PER_RUN; s++) {
        if (tr->has_update_batch[s]) {
            double tu0 = now_ns();
            for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                uint32_t u = tr->upd_u[s][k];
                if (tr->upd_is_add[s][k]) {
                    dyngraph_add_edge(&g, u, tr->upd_v[s][k]);
                } else {
                    dyngraph_remove_edge(&g, u);
                }
                alg1_notify_update(eng, u);
            }
            maint_ns_sum += (now_ns() - tu0);
        }

        /* Decaimiento periódico de calor cada 200 consultas */
        if (s > 0 && (s % 200U) == 0) {
            for (uint32_t b = 0; b < eng->num_allocated_tiles; b++) {
                eng->tiles[b].n_b_window >>= 1;
            }
        }

        uint32_t root = tr->query_roots[s];
        uint16_t b = eng->assigned_tile[root];
        bool was_valid_before = (b != NO_TILE && eng->tiles[b].valid);

        /* Ejecutar la consulta (y si es fría, la propia cola `scratch->queue_u` ya contiene los vértices visitados) */
        uint16_t visited_cnt  = 0;
        uint16_t expanded_cnt = 0;
        uint16_t cold_exp_cnt = 0;
        double tq0 = now_ns();
        uint64_t h = bfs_query_alg1(&g, eng, scratch, root, &visited_cnt, &expanded_cnt, &cold_exp_cnt);
        double dq = now_ns() - tq0;

        if (h != oracle_hashes[s]) errors++;
        lat_samples[s] = dq / 1000.0;
        query_ns_sum += dq;

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

            /*
             * En V6 y V7 (Micro-Tile Incremental): si la consulta partió de `b` pero expandió
             * vértices interiores (`0..expanded_cnt-1`) que aún no estaban en `b`, los anexa
             * incrementalmente hasta cubrir el núcleo activo o alcanzar el presupuesto de bytes.
             */
            if ((mode == MODE_V6_INC_BUDGET || mode == MODE_V7_INC_H3) && cold_exp_cnt > 0) {
                uint16_t app_cnt = (visited_cnt < 64U) ? visited_cnt : 64U;
                double app_ns = alg1_append_to_tile(eng, b, &g, scratch->queue_u, app_cnt);
                maint_ns_sum += app_ns;
            }
        } else {
            eng->cold_query_ns_sum += dq;
            eng->cold_query_count++;
            cold_ewma_ns = 0.90 * cold_ewma_ns + 0.10 * dq;

            /*
             * Evaluar si la región visitada por esta consulta fría merece ser capturada en un Tile:
             * Contamos el calor de la vecindad sobre los vértices interiores expandidos (`0..expanded_cnt-1`)
             * sin necesidad de ningún oráculo de `community_id`.
             */
            uint32_t neighborhood_heat = 0;
            uint16_t b_existing = NO_TILE;
            uint16_t check_n = (expanded_cnt < 16U) ? expanded_cnt : 16U;
            for (uint16_t i = 0; i < check_n; i++) {
                uint32_t u = scratch->queue_u[i];
                if (eng->vertex_heat[u] < 250U) eng->vertex_heat[u]++;
                neighborhood_heat += eng->vertex_heat[u];
                if (b_existing == NO_TILE) {
                    uint16_t tb_id = eng->assigned_tile[u];
                    if (tb_id != NO_TILE && eng->tiles[tb_id].valid) {
                        b_existing = tb_id;
                    }
                }
            }

            if (b_existing != NO_TILE) {
                /*
                 * La vecindad ya posee un Tile activo `b_existing`:
                 *   - En V6 y V7 (Micro-Tile Incremental): anexamos incrementalmente `root`
                 *     y los vértices interiores/frontera inmediata (`0..63`) al Micro-Tile existente.
                 *   - En V3, V4 y V5 (Tiles de tamaño fijo): no creamos un segundo Tile
                 *     competidor que fragmente y canibalice a `b_existing`.
                 */
                if (mode == MODE_V6_INC_BUDGET || mode == MODE_V7_INC_H3) {
                    uint16_t app_cnt = (visited_cnt < 64U) ? visited_cnt : 64U;
                    double app_ns = alg1_append_to_tile(eng, b_existing, &g, scratch->queue_u, app_cnt);
                    maint_ns_sum += app_ns;
                }
            } else {
                bool should_promote = false;
                if (mode == MODE_V6_INC_BUDGET) {
                    /* V6: Captura incremental apenas ve actividad mínima, sin filtro de coste H3 */
                    should_promote = (neighborhood_heat >= 20U);
                } else {
                    /* Regla H3 idéntica para V3, V4, V5 y V7: n_b_efectivo * H * g_b > alpha * C_b */
                    double effective_nb = (double)neighborhood_heat / (double)(check_n ? check_n : 1U);
                    double est_gb = (cold_ewma_ns > hot_ewma_ns + 200.0) ? (cold_ewma_ns - hot_ewma_ns) : 1200.0;
                    double expected_return = (effective_nb * HORIZON_MULT) * est_gb;
                    double cost_threshold  = ALPHA_H3 * measured_cb_ewma_ns;
                    if (effective_nb >= 2.35 && expected_return > cost_threshold) {
                        should_promote = true;
                    }
                }

                if (should_promote) {
                    uint16_t target_b = (uint16_t)(next_tile_slot++ % MAX_NUM_TILES);
                    if (next_tile_slot > eng->num_allocated_tiles && eng->num_allocated_tiles < MAX_NUM_TILES) {
                        eng->num_allocated_tiles = next_tile_slot;
                    }

                    double build_ns = 0.0;
                    if (mode == MODE_V3_FIXED_1024) {
                        /* V3 necesita recolectar 1,024 vértices para llenar su Tile fijo */
                        double tm0 = now_ns();
                        uint16_t cnt1024 = collect_bfs_neighborhood(&g, scratch, root, temp_nodes, 1024U);
                        build_ns = alg1_build_or_expand_tile(eng, target_b, &g, temp_nodes, cnt1024);
                        build_ns = now_ns() - tm0;
                        eng->tiles[target_b].C_b_ns = build_ns;
                    } else if (mode == MODE_V4_FIXED_128 || mode == MODE_V5_FIXED_64) {
                        /* V4 (128) y V5 (64): captura de lote fijo desde la cola BFS ya recolectada */
                        uint16_t cap_cnt = visited_cnt;
                        if (cap_cnt > eng->tiles[target_b].max_vertices) {
                            cap_cnt = eng->tiles[target_b].max_vertices;
                        }
                        build_ns = alg1_build_or_expand_tile(eng, target_b, &g, scratch->queue_u, cap_cnt);
                    } else {
                        /*
                         * V6 y V7 (Micro-Tile Incremental):
                         * Captura inicial compacta de 48 vértices (~5 us) que luego crece
                         * orgánicamente mediante `alg1_append_to_tile` hasta el presupuesto de bytes.
                         */
                        uint16_t init_cnt = (visited_cnt < 48U) ? visited_cnt : 48U;
                        build_ns = alg1_build_or_expand_tile(eng, target_b, &g, scratch->queue_u, init_cnt);
                    }
                    maint_ns_sum += build_ns;
                    measured_cb_ewma_ns = 0.75 * measured_cb_ewma_ns + 0.25 * build_ns;

                    /* Limpiar el calor consumido tras promover */
                    for (uint16_t i = 0; i < check_n; i++) {
                        eng->vertex_heat[scratch->queue_u[i]] = 0;
                    }
                }
            }
        }
    }

    double total_ms = (query_ns_sum + maint_ns_sum) / 1e6;
    out->total_ms = total_ms;
    out->query_ms = query_ns_sum / 1e6;
    out->maint_ms = maint_ns_sum / 1e6;
    out->qps      = (double)NUM_STEPS_PER_RUN / (total_ms / 1000.0);
    compute_percentiles(lat_samples, NUM_STEPS_PER_RUN, &out->p50_us, &out->p95_us, &out->p99_us);

    out->cap_us_per_promo   = eng->promotions ? ((eng->capture_ns_sum / eng->promotions) / 1000.0) : 0.0;
    out->first_q_us         = eng->first_query_count ? ((eng->first_query_ns_sum / eng->first_query_count) / 1000.0) : 0.0;
    out->subseq_q_us        = eng->subseq_query_count ? ((eng->subseq_query_ns_sum / eng->subseq_query_count) / 1000.0) : 0.0;
    out->cold_q_us          = eng->cold_query_count ? ((eng->cold_query_ns_sum / eng->cold_query_count) / 1000.0) : 0.0;

    double per_q_saving_us  = (out->subseq_q_us > 0.0 && out->cold_q_us > out->subseq_q_us)
                              ? (out->cold_q_us - out->subseq_q_us)
                              : 0.0;
    out->breakeven_queries  = (per_q_saving_us > 0.05) ? (out->cap_us_per_promo / per_q_saving_us) : 0.0;
    out->replica_hit_pct    = eng->expanded_total ? (100.0 * (double)eng->expanded_hot_hits / (double)eng->expanded_total) : 0.0;
    out->read_through_pct   = (eng->expanded_hot_hits + eng->read_through_fallbacks)
                              ? (100.0 * (double)eng->read_through_fallbacks / (double)(eng->expanded_hot_hits + eng->read_through_fallbacks))
                              : 0.0;
    out->bytes_copied_per_q = (double)eng->total_bytes_copied / (double)NUM_STEPS_PER_RUN;
    out->mem_mb             = (double)(g.bytes_allocated + eng->bytes_allocated + sizeof(BfsScratch)) / (1024.0 * 1024.0);
    out->promotions         = eng->promotions;
    out->useless_promotions = eng->useless_promotions;
    out->oracle_errors      = errors;

    free(temp_nodes);
    alg1_engine_free(eng);
    free(eng);
    dyngraph_free(&g);
}

int main(void) {
    SetConsoleOutputCP(65001);
    setvbuf(stdout, NULL, _IONBF, 0);
    /* Fijar el proceso al P-Core 0 del Intel Core i5-13500 para evitar migración P-core <-> E-core */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);
    init_timer();

    printf("========================================================================================================================\n");
    printf(" EXPERIMENTO ALG-1: PROMOCION INCREMENTAL Y GRANULARIDAD DEL HOT-CORE (V1 vs V3, V4, V5, V6, V7)\n");
    printf(" Evaluacion en 3 Semillas de Test Reservadas (No usadas en ajuste) | Grafo: 262,144 vertices | BFS k=%u (max %u)\n",
           BFS_MAX_DEPTH, BFS_MAX_VISITED);
    printf("========================================================================================================================\n\n");

    run_subexperiment_1_index_comparison();

    /* Semillas reservadas exclusivamente para evaluación final (distintas de las de calibración) */
    const uint64_t TEST_SEEDS[NUM_TEST_SEEDS] = {
        0x778899AABBCCDDEEULL,
        0x3141592653589793ULL,
        0x2718281828459045ULL
    };

    DynGraph test_graphs[NUM_TEST_SEEDS];
    for (uint32_t s = 0; s < NUM_TEST_SEEDS; s++) {
        generate_base_graph(&test_graphs[s], TEST_SEEDS[s]);
    }
    printf("Grafos de Test inicializados (Semilla Test 0: %llu aristas dirigidas, ~%.2f MB base, SIN oraculo de comunidades).\n\n",
           (unsigned long long)test_graphs[0].total_edges,
           (double)test_graphs[0].bytes_allocated / (1024.0 * 1024.0));

    WorkloadTrace *tr = (WorkloadTrace *)malloc(sizeof(WorkloadTrace));
    BfsScratch *scratch = (BfsScratch *)malloc(sizeof(BfsScratch));
    double *lat_samples = (double *)malloc(NUM_STEPS_PER_RUN * sizeof(double));
    uint64_t *oracle_hashes = (uint64_t *)malloc(NUM_STEPS_PER_RUN * sizeof(uint64_t));

    const char *V_NAMES[6] = {
        "V1 (Dynamic AdjList)  ",
        "V3 (Tile 1024 + H3)   ",
        "V4 (HotCore 128 + H3) ",
        "V5 (HotCore 64 + H3)  ",
        "V6 (MicroTile Inc)    ",
        "V7 (MicroTile Inc+H3) "
    };

    for (int w = 0; w < NUM_WORKLOADS; w++) {
        Alg1Metrics avg[6];
        memset(avg, 0, sizeof(avg));

        for (uint32_t s = 0; s < NUM_TEST_SEEDS; s++) {
            generate_workload_trace(tr, &test_graphs[s], (WorkloadType)w, TEST_SEEDS[s]);

            /* Ejecutar V1 (Baseline y Oráculo) */
            {
                DynGraph g1;
                dyngraph_clone(&g1, &test_graphs[s]);
                bfs_scratch_init(scratch);
                double q_ns = 0.0, m_ns = 0.0;
                for (uint32_t step = 0; step < NUM_STEPS_PER_RUN; step++) {
                    if (tr->has_update_batch[step]) {
                        double tu0 = now_ns();
                        for (uint32_t k = 0; k < UPDATES_PER_BATCH; k++) {
                            if (tr->upd_is_add[step][k]) dyngraph_add_edge(&g1, tr->upd_u[step][k], tr->upd_v[step][k]);
                            else dyngraph_remove_edge(&g1, tr->upd_u[step][k]);
                        }
                        m_ns += (now_ns() - tu0);
                    }
                    double tq0 = now_ns();
                    oracle_hashes[step] = bfs_query_v1(&g1, scratch, tr->query_roots[step]);
                    double dq = now_ns() - tq0;
                    lat_samples[step] = dq / 1000.0;
                    q_ns += dq;
                }
                double p50, p95, p99;
                compute_percentiles(lat_samples, NUM_STEPS_PER_RUN, &p50, &p95, &p99);
                double tot_ms = (q_ns + m_ns) / 1e6;
                avg[0].total_ms  += tot_ms / NUM_TEST_SEEDS;
                avg[0].query_ms  += (q_ns / 1e6) / NUM_TEST_SEEDS;
                avg[0].maint_ms  += (m_ns / 1e6) / NUM_TEST_SEEDS;
                avg[0].qps       += ((double)NUM_STEPS_PER_RUN / (tot_ms / 1000.0)) / NUM_TEST_SEEDS;
                avg[0].p50_us    += p50 / NUM_TEST_SEEDS;
                avg[0].p95_us    += p95 / NUM_TEST_SEEDS;
                avg[0].p99_us    += p99 / NUM_TEST_SEEDS;
                avg[0].cold_q_us += ((q_ns / NUM_STEPS_PER_RUN) / 1000.0) / NUM_TEST_SEEDS;
                avg[0].mem_mb    += ((double)(g1.bytes_allocated + sizeof(BfsScratch)) / (1024.0 * 1024.0)) / NUM_TEST_SEEDS;
                dyngraph_free(&g1);
            }

            /* Ejecutar V3, V4, V5, V6, V7 */
            AlgMode modes[5] = {
                MODE_V3_FIXED_1024,
                MODE_V4_FIXED_128,
                MODE_V5_FIXED_64,
                MODE_V6_INC_BUDGET,
                MODE_V7_INC_H3
            };
            for (int m_idx = 0; m_idx < 5; m_idx++) {
                Alg1Metrics m;
                run_alg1_variant(&test_graphs[s], tr, oracle_hashes, modes[m_idx], scratch, lat_samples, &m);
                int idx = m_idx + 1;
                avg[idx].total_ms           += m.total_ms / NUM_TEST_SEEDS;
                avg[idx].query_ms           += m.query_ms / NUM_TEST_SEEDS;
                avg[idx].maint_ms           += m.maint_ms / NUM_TEST_SEEDS;
                avg[idx].qps                += m.qps / NUM_TEST_SEEDS;
                avg[idx].p50_us             += m.p50_us / NUM_TEST_SEEDS;
                avg[idx].p95_us             += m.p95_us / NUM_TEST_SEEDS;
                avg[idx].p99_us             += m.p99_us / NUM_TEST_SEEDS;
                avg[idx].cap_us_per_promo   += m.cap_us_per_promo / NUM_TEST_SEEDS;
                avg[idx].first_q_us         += m.first_q_us / NUM_TEST_SEEDS;
                avg[idx].subseq_q_us        += m.subseq_q_us / NUM_TEST_SEEDS;
                avg[idx].cold_q_us          += m.cold_q_us / NUM_TEST_SEEDS;
                avg[idx].breakeven_queries  += m.breakeven_queries / NUM_TEST_SEEDS;
                avg[idx].replica_hit_pct    += m.replica_hit_pct / NUM_TEST_SEEDS;
                avg[idx].read_through_pct   += m.read_through_pct / NUM_TEST_SEEDS;
                avg[idx].bytes_copied_per_q += m.bytes_copied_per_q / NUM_TEST_SEEDS;
                avg[idx].mem_mb             += m.mem_mb / NUM_TEST_SEEDS;
                avg[idx].promotions         += m.promotions;
                avg[idx].useless_promotions += m.useless_promotions;
                avg[idx].oracle_errors      += m.oracle_errors;
            }
        }

        printf("------------------------------------------------------------------------------------------------------------------------\n");
        printf(" %s (Media de %u Semillas de Test)\n", WORKLOAD_NAMES[w], NUM_TEST_SEEDS);
        printf("------------------------------------------------------------------------------------------------------------------------\n");
        printf(" Variante               | Total(ms) | Query(ms) | Maint(ms) | p50(us) | p95(us) | Cap(us) | 1stQ/SubQ(us) | BrkEv | Hit%%  | B/Q  | Prom(Inut) | Exact\n");
        printf("------------------------+-----------+-----------+-----------+---------+---------+---------+---------------+-------+-------+------+------------+------\n");
        for (int v = 0; v < 6; v++) {
            printf(" %s | %9.2f | %9.2f | %9.2f | %7.2f | %7.2f | %7.1f | %5.2f / %5.2f | %5.1f | %4.1f%% | %4.0f | %4u(%4u) | %s\n",
                   V_NAMES[v],
                   avg[v].total_ms,
                   avg[v].query_ms,
                   avg[v].maint_ms,
                   avg[v].p50_us,
                   avg[v].p95_us,
                   avg[v].cap_us_per_promo,
                   avg[v].first_q_us,
                   avg[v].subseq_q_us,
                   avg[v].breakeven_queries,
                   avg[v].replica_hit_pct,
                   avg[v].bytes_copied_per_q,
                   avg[v].promotions / NUM_TEST_SEEDS,
                   avg[v].useless_promotions / NUM_TEST_SEEDS,
                   (avg[v].oracle_errors == 0) ? "100%" : "ERR");
        }
        printf("\n");
    }

    free(tr);
    free(scratch);
    free(lat_samples);
    free(oracle_hashes);
    for (uint32_t s = 0; s < NUM_TEST_SEEDS; s++) {
        dyngraph_free(&test_graphs[s]);
    }
    return 0;
}
