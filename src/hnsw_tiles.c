#include "../include/alg_hnsw.h"
#include "../include/heap.h"
#include <windows.h>
#include <stdlib.h>
#include <string.h>

static inline double now_ns_tiles(void) {
    static double ns_per_tick = 0.0;
    if (ns_per_tick == 0.0) {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        ns_per_tick = 1e9 / (double)freq.QuadPart;
    }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * ns_per_tick;
}

static inline void local_index_clear(CompactLocalIndexHnsw *idx, uint32_t max_vertices) {
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

static inline void local_index_insert(CompactLocalIndexHnsw *idx, uint32_t global_id, uint16_t local_slot) {
    uint32_t pos = (global_id * 0x9E3779B9U) >> idx->shift;
    uint32_t kp1 = global_id + 1U;
    while (idx->key_plus_one[pos] != 0 && idx->key_plus_one[pos] != kp1) {
        pos = (pos + 1U) & idx->mask;
    }
    idx->key_plus_one[pos] = kp1;
    idx->slot[pos] = local_slot;
}

HnswTileEngine *hnsw_tile_engine_create(uint32_t num_elements, uint32_t tile_capacity, uint32_t max_tiles) {
    HnswTileEngine *eng = (HnswTileEngine *)malloc(sizeof(HnswTileEngine));
    if (!eng) return NULL;
    memset(eng, 0, sizeof(HnswTileEngine));

    eng->num_elements  = num_elements;
    eng->tile_capacity = tile_capacity;
    eng->max_tiles     = (max_tiles < MAX_HNSW_TILES) ? max_tiles : MAX_HNSW_TILES;

    eng->assigned_tile = (uint16_t *)malloc(num_elements * sizeof(uint16_t));
    eng->assigned_slot = (uint16_t *)malloc(num_elements * sizeof(uint16_t));
    eng->node_heat     = (uint8_t *)calloc(num_elements, sizeof(uint8_t));

    for (uint32_t i = 0; i < num_elements; i++) {
        eng->assigned_tile[i] = NO_TILE;
        eng->assigned_slot[i] = EXTERNAL_SLOT;
    }

    for (uint32_t b = 0; b < eng->max_tiles; b++) {
        HnswMicroTile *t = &eng->tiles[b];
        t->tile_id  = (uint16_t)b;
        t->capacity = (uint16_t)tile_capacity;
        t->count    = 0;
        t->valid    = false;
        t->version  = 1;
        t->nodes    = (SwizzledNodeView *)_aligned_malloc(tile_capacity * sizeof(SwizzledNodeView), 64);
        local_index_clear(&t->local_idx, tile_capacity);
    }

    return eng;
}

static inline void unswizzle_node(HnswIndex *index, uint32_t u) {
    HnswNode *n = &index->nodes[u];
    n->read_view.neighbors   = n->friends[0];
    n->read_view.vector      = &index->vectors[u * index->dim];
    n->read_view.deg         = n->friend_count[0];
    n->read_view.is_swizzled = false;
}

static inline void swizzle_node(HnswIndex *index, HnswMicroTile *t, uint32_t u, uint16_t slot) {
    HnswNode *n = &index->nodes[u];
    SwizzledNodeView *sn = &t->nodes[slot];
    n->read_view.neighbors   = sn->neighbors;
    n->read_view.vector      = sn->vector;
    n->read_view.deg         = sn->deg;
    n->read_view.is_swizzled = true;
}

static void invalidate_tile(HnswTileEngine *eng, HnswIndex *index, uint16_t b) {
    HnswMicroTile *t = &eng->tiles[b];
    if (!t->valid) return;

    for (uint16_t i = 0; i < t->count; i++) {
        uint32_t u = t->nodes[i].logical_id;
        if (eng->assigned_tile[u] == b) {
            eng->assigned_tile[u] = NO_TILE;
            eng->assigned_slot[u] = EXTERNAL_SLOT;
            unswizzle_node(index, u);
        }
    }
    t->valid = false;
    t->count = 0;
    if (eng->active_tiles > 0) eng->active_tiles--;
}

void hnsw_tile_engine_reset(HnswTileEngine *eng, HnswIndex *index) {
    if (!eng || !index) return;
    for (uint16_t b = 0; b < eng->max_tiles; b++) {
        eng->tiles[b].valid = false;
        eng->tiles[b].count = 0;
    }
    for (uint32_t i = 0; i < eng->num_elements; i++) {
        eng->assigned_tile[i] = NO_TILE;
        eng->assigned_slot[i] = EXTERNAL_SLOT;
        eng->node_heat[i]     = 0;

        HnswNode *n = &index->nodes[i];
        n->read_view.neighbors   = n->friends[0];
        n->read_view.vector      = &index->vectors[i * index->dim];
        n->read_view.deg         = n->friend_count[0];
        n->read_view.is_swizzled = false;
    }
    eng->active_tiles  = 0;
    eng->promotions    = 0;
    eng->total_evals   = 0;
    eng->hot_tile_hits = 0;
    eng->maint_ns_sum  = 0.0;
    eng->query_ns_sum  = 0.0;
}

void hnsw_tile_engine_free(HnswTileEngine *eng) {
    if (!eng) return;
    for (uint32_t b = 0; b < eng->max_tiles; b++) {
        if (eng->tiles[b].nodes) _aligned_free(eng->tiles[b].nodes);
    }
    if (eng->assigned_tile) free(eng->assigned_tile);
    if (eng->assigned_slot) free(eng->assigned_slot);
    if (eng->node_heat) free(eng->node_heat);
    free(eng);
}

/*
 * Construcción e instalación de Micro-Tile con Pointer Swizzling inmediato
 */
static void build_micro_tile(
    HnswTileEngine *eng,
    HnswIndex *index,
    uint16_t b,
    const uint32_t *captured_nodes,
    uint16_t count
) {
    double t0 = now_ns_tiles();
    HnswMicroTile *t = &eng->tiles[b];

    if (t->valid) invalidate_tile(eng, index, b);
    eng->active_tiles++;

    if (count > t->capacity) count = t->capacity;
    local_index_clear(&t->local_idx, t->capacity);

    uint16_t actual = 0;
    for (uint16_t i = 0; i < count; i++) {
        uint32_t u = captured_nodes[i];
        if (u >= index->cur_elements) continue;

        /* Si u estaba en otro tile, des-swizzle previo */
        if (eng->assigned_tile[u] != NO_TILE && eng->assigned_tile[u] != b) {
            uint16_t old_b = eng->assigned_tile[u];
            uint16_t old_s = eng->assigned_slot[u];
            if (eng->tiles[old_b].valid && old_s < eng->tiles[old_b].count) {
                unswizzle_node(index, u);
            }
        }

        SwizzledNodeView *sn = &t->nodes[actual];
        sn->logical_id = u;
        HnswNode *orig = &index->nodes[u];
        sn->deg = orig->friend_count[0];
        if (sn->deg > HNSW_M0_DEFAULT) sn->deg = HNSW_M0_DEFAULT;

        for (uint16_t k = 0; k < sn->deg; k++) {
            sn->neighbors[k] = orig->friends[0][k];
        }
        memcpy(sn->vector, &index->vectors[u * index->dim], index->dim * sizeof(float));

        local_index_insert(&t->local_idx, u, actual);
        eng->assigned_tile[u] = b;
        eng->assigned_slot[u] = actual;

        /* Pointer Swizzling */
        swizzle_node(index, t, u, actual);
        actual++;
    }

    t->count = actual;
    t->valid = true;
    t->version++;
    eng->promotions++;
    eng->maint_ns_sum += (now_ns_tiles() - t0);
}

/* Visited Set con epoch */
typedef struct {
    uint32_t *epochs;
    uint32_t cur_epoch;
    uint32_t size;
} VisitedSet;

static VisitedSet *visited_create_t(uint32_t size) {
    VisitedSet *v = (VisitedSet *)malloc(sizeof(VisitedSet));
    v->epochs = (uint32_t *)calloc(size, sizeof(uint32_t));
    v->cur_epoch = 1;
    v->size = size;
    return v;
}

static inline bool visited_test_and_set_t(VisitedSet *v, uint32_t id) {
    if (v->epochs[id] == v->cur_epoch) return true;
    v->epochs[id] = v->cur_epoch;
    return false;
}

/*
 * ============================================================================
 * BUSQUEDA H3: FILTER-AWARE CON POINTER-SWIZZLED MICRO-TILES
 * ============================================================================
 */
SearchResult hnsw_search_h3_swizzled(
    HnswIndex *index,
    HnswTileEngine *eng,
    const float *query,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    uint32_t top_k,
    uint32_t ef_search,
    bool update_heat_and_promote
) {
    double tq0 = now_ns_tiles();
    SearchResult res;
    memset(&res, 0, sizeof(SearchResult));
    res.k = top_k;

    if (index->enter_node == -1) return res;

    uint64_t dist_comps = 0;
    uint64_t visited_cnt = 0;
    uint64_t hot_hits = 0;

    int32_t curr_obj = index->enter_node;
    float curr_dist = l2_distance_avx2(query, &index->vectors[curr_obj * index->dim], index->dim);
    dist_comps++;

    /* Descenso normal por capas superiores */
    for (int32_t l = index->max_level; l > 0; l--) {
        bool changed = true;
        while (changed) {
            changed = false;
            const HnswNode *n = &index->nodes[curr_obj];
            for (uint16_t i = 0; i < n->friend_count[l]; i++) {
                uint32_t neighbor = n->friends[l][i];
                float d = l2_distance_avx2(query, &index->vectors[neighbor * index->dim], index->dim);
                dist_comps++;
                if (d < curr_dist) {
                    curr_dist = d;
                    curr_obj = (int32_t)neighbor;
                    changed = true;
                }
            }
        }
    }

    /* Búsqueda en Capa L0 con Pointer Swizzling */
    static VisitedSet *vset = NULL;
    if (!vset || vset->size < index->max_elements) {
        if (vset) { free(vset->epochs); free(vset); }
        vset = visited_create_t(index->max_elements);
    }
    vset->cur_epoch++;
    if (vset->cur_epoch == 0) {
        memset(vset->epochs, 0, vset->size * sizeof(uint32_t));
        vset->cur_epoch = 1;
    }

    visited_test_and_set_t(vset, (uint32_t)curr_obj);
    visited_cnt++;

    DistPair candidates[512];
    DistPair W[512];
    uint32_t cand_cnt = 0, w_cnt = 0;

    DistPair initial = { curr_dist, (uint32_t)curr_obj };
    hnsw_candidates_push(candidates, &cand_cnt, 512U, initial);

    if (filter_matches(attrs, (uint32_t)curr_obj, pred)) {
        hnsw_results_insert(W, &w_cnt, ef_search, initial);
    }

    uint32_t visited_transit_list[512];
    uint32_t transit_count = 0;

    while (cand_cnt > 0) {
        DistPair c = hnsw_candidates_pop(candidates, &cand_cnt);

        if (w_cnt >= ef_search) {
            DistPair furthest_res = hnsw_results_worst(W, w_cnt);
            if (distpair_compare_asc(&c, &furthest_res) > 0) break;
        }

        if (transit_count < 512) {
            visited_transit_list[transit_count++] = c.id;
        }

        /* LECTURA CON POINTER SWIZZLING */
        HnswReadView rw = index->nodes[c.id].read_view;
        if (rw.is_swizzled) hot_hits++;

        const uint32_t *neighbors = rw.neighbors;
        uint32_t deg = rw.deg;

        for (uint16_t i = 0; i < deg; i++) {
            uint32_t neighbor = neighbors[i];
            if (!visited_test_and_set_t(vset, neighbor)) {
                visited_cnt++;

                /* Si el vecino está swizzleado, lee su vector contiguo en L1/L2 */
                const float *v_vec = index->nodes[neighbor].read_view.vector;
                if (__builtin_expect(!v_vec, 0)) v_vec = &index->vectors[neighbor * index->dim];
                float d = l2_distance_avx2(query, v_vec, index->dim);
                dist_comps++;

                bool eligible = filter_matches(attrs, neighbor, pred);

                bool should_expand = (w_cnt < ef_search);
                if (!should_expand) {
                    DistPair worst = hnsw_results_worst(W, w_cnt);
                    DistPair candidate = { d, neighbor };
                    should_expand = (distpair_compare_asc(&candidate, &worst) < 0);
                }

                if (should_expand && cand_cnt < 510) {
                    DistPair candidate = { d, neighbor };
                    hnsw_candidates_push(candidates, &cand_cnt, 512U, candidate);
                }

                if (eligible) {
                    DistPair eligible_pair = { d, neighbor };
                    hnsw_results_insert(W, &w_cnt, ef_search, eligible_pair);
                }
            }
        }
    }

    hnsw_results_sort(W, w_cnt);
    uint32_t ret = (w_cnt < top_k) ? w_cnt : top_k;
    res.returned_count = ret;
    for (uint32_t i = 0; i < ret; i++) {
        res.ids[i]   = W[i].id;
        res.dists[i] = W[i].dist;
    }
    res.distance_comps = dist_comps;
    res.visited_nodes  = visited_cnt;
    res.tile_hot_hits  = hot_hits;

    double dt = now_ns_tiles() - tq0;
    res.latency_us = dt / 1000.0;
    eng->query_ns_sum += dt;
    eng->total_evals  += visited_cnt;
    eng->hot_tile_hits += hot_hits;

    /* Gestión adaptativa de calor y promoción a Micro-Tiles */
    if (update_heat_and_promote && transit_count > 8) {
        uint32_t heat_acc = 0;
        uint16_t existing_tile = NO_TILE;

        uint32_t check_n = (transit_count < 32U) ? transit_count : 32U;
        for (uint32_t i = 0; i < check_n; i++) {
            uint32_t u = visited_transit_list[i];
            if (eng->node_heat[u] < 250U) eng->node_heat[u]++;
            heat_acc += eng->node_heat[u];
            if (existing_tile == NO_TILE && eng->assigned_tile[u] != NO_TILE) {
                existing_tile = eng->assigned_tile[u];
            }
        }

        /* Umbral de calor acumulado para promocionar cuenca de tránsito recurrente */
        if (heat_acc >= 28U) {
            static uint32_t next_slot = 0;
            uint16_t target_tile = (existing_tile != NO_TILE) ? existing_tile : (uint16_t)(next_slot++ % eng->max_tiles);
            build_micro_tile(eng, index, target_tile, visited_transit_list, (uint16_t)transit_count);

            for (uint32_t i = 0; i < check_n; i++) {
                eng->node_heat[visited_transit_list[i]] = 0;
            }
        }
    }

    return res;
}
