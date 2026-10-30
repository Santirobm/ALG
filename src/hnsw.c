#include "../include/alg_hnsw.h"
#include "../include/heap.h"
#include "../include/sq8.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline uint64_t xorshift64s_hnsw(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float uniform_float_hnsw(uint64_t *state) {
    uint64_t r = xorshift64s_hnsw(state);
    return (float)((r & 0xFFFFFF) + 1) / (float)0x1000001;
}

static int32_t random_level(double mult_l, uint64_t *rng) {
    float r = uniform_float_hnsw(rng);
    int32_t lvl = (int32_t)(-logf(r) * (float)mult_l);
    if (lvl >= (int32_t)MAX_LAYERS) lvl = (int32_t)MAX_LAYERS - 1;
    return lvl;
}

HnswIndex *hnsw_create(uint32_t max_elements, uint32_t dim, uint32_t M, uint32_t ef_construction) {
    HnswIndex *idx = (HnswIndex *)malloc(sizeof(HnswIndex));
    if (!idx) return NULL;

    idx->max_elements    = max_elements;
    idx->cur_elements    = 0;
    idx->dim             = dim;
    idx->M               = M;
    idx->M0              = M * 2U;
    idx->ef_construction = ef_construction;
    idx->ef_search       = HNSW_EF_SEARCH;
    idx->mult_l          = 1.0 / log((double)M);
    idx->enter_node      = -1;
    idx->max_level       = -1;

    idx->vectors = (float *)aligned_alloc_64((size_t)max_elements * (size_t)dim * sizeof(float));
    idx->nodes   = (HnswNode *)calloc(max_elements, sizeof(HnswNode));

    for (uint32_t i = 0; i < max_elements; i++) {
        idx->nodes[i].id = i;
        idx->nodes[i].level = -1;
    }

    return idx;
}

void hnsw_free(HnswIndex *idx) {
    if (!idx) return;
    if (idx->vectors) aligned_free_64(idx->vectors);
    if (idx->nodes) {
        for (uint32_t i = 0; i < idx->cur_elements; i++) {
            for (int32_t l = 0; l <= idx->nodes[i].level; l++) {
                if (idx->nodes[i].friends[l]) free(idx->nodes[i].friends[l]);
            }
        }
        free(idx->nodes);
    }
    free(idx);
}

/* Priority Queue helpers */
static int cmp_pair_asc(const void *a, const void *b) {
    return distpair_compare_asc((const DistPair *)a, (const DistPair *)b);
}


/* Visited Set con epoch para O(1) clear */
typedef struct {
    uint32_t *epochs;
    uint32_t cur_epoch;
    uint32_t size;
} VisitedSet;

static VisitedSet *visited_create(uint32_t size) {
    VisitedSet *v = (VisitedSet *)malloc(sizeof(VisitedSet));
    v->epochs = (uint32_t *)calloc(size, sizeof(uint32_t));
    v->cur_epoch = 1;
    v->size = size;
    return v;
}

static void visited_free(VisitedSet *v) {
    if (v) {
        if (v->epochs) free(v->epochs);
        free(v);
    }
}

static inline bool visited_test_and_set(VisitedSet *v, uint32_t id) {
    if (v->epochs[id] == v->cur_epoch) return true;
    v->epochs[id] = v->cur_epoch;
    return false;
}

static inline void visited_clear(VisitedSet *v) {
    v->cur_epoch++;
    if (v->cur_epoch == 0) {
        memset(v->epochs, 0, v->size * sizeof(uint32_t));
        v->cur_epoch = 1;
    }
}

/*
 * Inserción de un nuevo vector en HNSW
 */
void hnsw_insert(HnswIndex *index, uint32_t id, const float *vec) {
    if (id >= index->max_elements) return;

    memcpy(&index->vectors[id * index->dim], vec, index->dim * sizeof(float));

    static uint64_t rng = 0x123456789ABCDEF0ULL;
    int32_t insert_level = random_level(index->mult_l, &rng);
    index->nodes[id].level = insert_level;

    for (int32_t l = 0; l <= insert_level; l++) {
        uint32_t cap = (l == 0) ? index->M0 : index->M;
        index->nodes[id].friend_cap[l]   = (uint16_t)cap;
        index->nodes[id].friend_count[l] = 0;
        index->nodes[id].friends[l]      = (uint32_t *)malloc(cap * sizeof(uint32_t));
    }

    if (index->enter_node == -1) {
        index->enter_node = (int32_t)id;
        index->max_level  = insert_level;
        index->cur_elements++;
        return;
    }

    int32_t curr_obj = index->enter_node;
    float curr_dist = l2_distance_avx2(vec, &index->vectors[curr_obj * index->dim], index->dim);

    /* 1. Descenso voraz por niveles superiores a insert_level */
    for (int32_t l = index->max_level; l > insert_level; l--) {
        bool changed = true;
        while (changed) {
            changed = false;
            HnswNode *n = &index->nodes[curr_obj];
            for (uint16_t i = 0; i < n->friend_count[l]; i++) {
                uint32_t friend_id = n->friends[l][i];
                float d = l2_distance_avx2(vec, &index->vectors[friend_id * index->dim], index->dim);
                if (d < curr_dist) {
                    curr_dist = d;
                    curr_obj = (int32_t)friend_id;
                    changed = true;
                }
            }
        }
    }

    /* 2. Conexión en capas insert_level hasta nivel 0 */
    VisitedSet *vset = visited_create(index->max_elements);
    DistPair candidates[512];
    DistPair W[512];

    for (int32_t l = (insert_level < index->max_level ? insert_level : index->max_level); l >= 0; l--) {
        visited_clear(vset);
        visited_test_and_set(vset, (uint32_t)curr_obj);

        uint32_t cand_cnt = 0;
        uint32_t w_cnt = 0;

        candidates[cand_cnt].dist = curr_dist;
        candidates[cand_cnt].id   = (uint32_t)curr_obj;
        cand_cnt++;

        W[w_cnt].dist = curr_dist;
        W[w_cnt].id   = (uint32_t)curr_obj;
        w_cnt++;

        while (cand_cnt > 0) {
            /* Extraer el candidato más cercano */
            qsort(candidates, cand_cnt, sizeof(DistPair), cmp_pair_asc);
            DistPair c = candidates[0];
            memmove(&candidates[0], &candidates[1], (cand_cnt - 1) * sizeof(DistPair));
            cand_cnt--;

            /* Extraer el peor elemento de W */
            qsort(W, w_cnt, sizeof(DistPair), cmp_pair_asc);
            DistPair f = W[w_cnt - 1];

            if (c.dist > f.dist && w_cnt >= index->ef_construction) break;

            HnswNode *n = &index->nodes[c.id];
            for (uint16_t i = 0; i < n->friend_count[l]; i++) {
                uint32_t neighbor = n->friends[l][i];
                if (!visited_test_and_set(vset, neighbor)) {
                    float d = l2_distance_avx2(vec, &index->vectors[neighbor * index->dim], index->dim);
                    qsort(W, w_cnt, sizeof(DistPair), cmp_pair_asc);
                    if (d < W[w_cnt - 1].dist || w_cnt < index->ef_construction) {
                        if (cand_cnt < 510) {
                            candidates[cand_cnt].dist = d;
                            candidates[cand_cnt].id   = neighbor;
                            cand_cnt++;
                        }
                        if (w_cnt < index->ef_construction) {
                            W[w_cnt].dist = d;
                            W[w_cnt].id   = neighbor;
                            w_cnt++;
                        } else {
                            W[w_cnt - 1].dist = d;
                            W[w_cnt - 1].id   = neighbor;
                        }
                    }
                }
            }
        }

        /* Seleccionar vecinos para la arista bidireccional */
        qsort(W, w_cnt, sizeof(DistPair), cmp_pair_asc);
        uint32_t M_target = (l == 0) ? index->M0 : index->M;
        uint32_t connect_count = (w_cnt < M_target) ? w_cnt : M_target;

        for (uint32_t i = 0; i < connect_count; i++) {
            uint32_t neighbor = W[i].id;
            index->nodes[id].friends[l][index->nodes[id].friend_count[l]++] = neighbor;

            /* Añadir enlace reverso */
            HnswNode *nn = &index->nodes[neighbor];
            if (nn->friend_count[l] < nn->friend_cap[l]) {
                nn->friends[l][nn->friend_count[l]++] = id;
            } else {
                /* Podar aristas si excede capacidad: mantener las más cercanas */
                DistPair n_dists[64];
                uint32_t cur_cnt = nn->friend_count[l];
                for (uint32_t k = 0; k < cur_cnt; k++) {
                    n_dists[k].id = nn->friends[l][k];
                    n_dists[k].dist = l2_distance_avx2(&index->vectors[neighbor * index->dim],
                                                       &index->vectors[n_dists[k].id * index->dim],
                                                       index->dim);
                }
                n_dists[cur_cnt].id = id;
                n_dists[cur_cnt].dist = l2_distance_avx2(&index->vectors[neighbor * index->dim], vec, index->dim);
                qsort(n_dists, cur_cnt + 1, sizeof(DistPair), cmp_pair_asc);
                for (uint32_t k = 0; k < cur_cnt; k++) {
                    nn->friends[l][k] = n_dists[k].id;
                }
            }
        }
        curr_obj = (int32_t)W[0].id;
        curr_dist = W[0].dist;
    }

    if (insert_level > index->max_level) {
        index->max_level  = insert_level;
        index->enter_node = (int32_t)id;
    }

    visited_free(vset);
    index->cur_elements++;

    /* Inicializar read_view autoritativa en L0 */
    index->nodes[id].read_view.neighbors = index->nodes[id].friends[0];
    index->nodes[id].read_view.vector    = &index->vectors[id * index->dim];
    index->nodes[id].read_view.deg       = index->nodes[id].friend_count[0];
    index->nodes[id].read_view.is_swizzled = false;
}

/*
 * ============================================================================
 * VARIANTES DE BUSQUEDA FILTRADA
 * ============================================================================
 */

/*
 * H0: Postfiltrado
 * Recorre el grafo HNSW normalmente hasta obtener ef_search candidatos en L0.
 * Descarta los que no cumplen el filtro y retorna top_k.
 */
SearchResult hnsw_search_h0_postfilter(
    const HnswIndex *index,
    const float *query,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    uint32_t top_k,
    uint32_t ef_search
) {
    SearchResult res;
    memset(&res, 0, sizeof(SearchResult));
    res.k = top_k;

    if (index->enter_node == -1) return res;

    uint64_t dist_comps = 0;
    uint64_t visited_cnt = 0;

    int32_t curr_obj = index->enter_node;
    float curr_dist = l2_distance_avx2(query, &index->vectors[curr_obj * index->dim], index->dim);
    dist_comps++;

    /* Descenso por capas superiores */
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

    /* Búsqueda en capa L0 */
    VisitedSet *vset = visited_create(index->max_elements);
    visited_test_and_set(vset, (uint32_t)curr_obj);
    visited_cnt++;

    DistPair candidates[512];
    DistPair W[512];
    uint32_t cand_cnt = 0, w_cnt = 0;

    candidates[0].dist = curr_dist;
    candidates[0].id   = (uint32_t)curr_obj;
    cand_cnt = 1;

    W[0].dist = curr_dist;
    W[0].id   = (uint32_t)curr_obj;
    w_cnt = 1;

    while (cand_cnt > 0) {
        qsort(candidates, cand_cnt, sizeof(DistPair), cmp_pair_asc);
        DistPair c = candidates[0];
        memmove(&candidates[0], &candidates[1], (cand_cnt - 1) * sizeof(DistPair));
        cand_cnt--;

        qsort(W, w_cnt, sizeof(DistPair), cmp_pair_asc);
        DistPair f = W[w_cnt - 1];

        if (c.dist > f.dist && w_cnt >= ef_search) break;

        const HnswNode *n = &index->nodes[c.id];
        for (uint16_t i = 0; i < n->friend_count[0]; i++) {
            uint32_t neighbor = n->friends[0][i];
            if (!visited_test_and_set(vset, neighbor)) {
                visited_cnt++;
                float d = l2_distance_avx2(query, &index->vectors[neighbor * index->dim], index->dim);
                dist_comps++;

                qsort(W, w_cnt, sizeof(DistPair), cmp_pair_asc);
                if (d < W[w_cnt - 1].dist || w_cnt < ef_search) {
                    if (cand_cnt < 510) {
                        candidates[cand_cnt].dist = d;
                        candidates[cand_cnt].id   = neighbor;
                        cand_cnt++;
                    }
                    if (w_cnt < ef_search) {
                        W[w_cnt].dist = d;
                        W[w_cnt].id   = neighbor;
                        w_cnt++;
                    } else {
                        W[w_cnt - 1].dist = d;
                        W[w_cnt - 1].id   = neighbor;
                    }
                }
            }
        }
    }

    /* Postfiltrado: conservar únicamente los candidatos de W que cumplen el filtro */
    qsort(W, w_cnt, sizeof(DistPair), cmp_pair_asc);
    uint32_t matched = 0;
    for (uint32_t i = 0; i < w_cnt; i++) {
        if (filter_matches(attrs, W[i].id, pred)) {
            res.ids[matched]   = W[i].id;
            res.dists[matched] = W[i].dist;
            matched++;
            if (matched >= top_k) break;
        }
    }
    res.returned_count = matched;
    res.distance_comps = dist_comps;
    res.visited_nodes  = visited_cnt;

    visited_free(vset);
    return res;
}

/*
 * H1: Prefiltrado Estricto
 * Ejecuta la búsqueda estándar sobre el índice prefiltrado exclusivo de nodos elegibles.
 */
SearchResult hnsw_search_h1_prefilter(
    const HnswIndex *filtered_index,
    const uint32_t *id_mapping,
    const float *query,
    uint32_t top_k,
    uint32_t ef_search
) {
    SearchResult res;
    memset(&res, 0, sizeof(SearchResult));
    res.k = top_k;

    if (filtered_index->enter_node == -1 || filtered_index->cur_elements == 0) return res;

    uint64_t dist_comps = 0;
    uint64_t visited_cnt = 0;

    int32_t curr_obj = filtered_index->enter_node;
    float curr_dist = l2_distance_avx2(query, &filtered_index->vectors[curr_obj * filtered_index->dim], filtered_index->dim);
    dist_comps++;

    for (int32_t l = filtered_index->max_level; l > 0; l--) {
        bool changed = true;
        while (changed) {
            changed = false;
            const HnswNode *n = &filtered_index->nodes[curr_obj];
            for (uint16_t i = 0; i < n->friend_count[l]; i++) {
                uint32_t neighbor = n->friends[l][i];
                float d = l2_distance_avx2(query, &filtered_index->vectors[neighbor * filtered_index->dim], filtered_index->dim);
                dist_comps++;
                if (d < curr_dist) {
                    curr_dist = d;
                    curr_obj = (int32_t)neighbor;
                    changed = true;
                }
            }
        }
    }

    VisitedSet *vset = visited_create(filtered_index->max_elements);
    visited_test_and_set(vset, (uint32_t)curr_obj);
    visited_cnt++;

    DistPair candidates[512];
    DistPair W[512];
    uint32_t cand_cnt = 0, w_cnt = 0;

    candidates[0].dist = curr_dist;
    candidates[0].id   = (uint32_t)curr_obj;
    cand_cnt = 1;

    W[0].dist = curr_dist;
    W[0].id   = (uint32_t)curr_obj;
    w_cnt = 1;

    while (cand_cnt > 0) {
        qsort(candidates, cand_cnt, sizeof(DistPair), cmp_pair_asc);
        DistPair c = candidates[0];
        memmove(&candidates[0], &candidates[1], (cand_cnt - 1) * sizeof(DistPair));
        cand_cnt--;

        qsort(W, w_cnt, sizeof(DistPair), cmp_pair_asc);
        DistPair f = W[w_cnt - 1];

        if (c.dist > f.dist && w_cnt >= ef_search) break;

        const HnswNode *n = &filtered_index->nodes[c.id];
        for (uint16_t i = 0; i < n->friend_count[0]; i++) {
            uint32_t neighbor = n->friends[0][i];
            if (!visited_test_and_set(vset, neighbor)) {
                visited_cnt++;
                float d = l2_distance_avx2(query, &filtered_index->vectors[neighbor * filtered_index->dim], filtered_index->dim);
                dist_comps++;

                qsort(W, w_cnt, sizeof(DistPair), cmp_pair_asc);
                if (d < W[w_cnt - 1].dist || w_cnt < ef_search) {
                    if (cand_cnt < 510) {
                        candidates[cand_cnt].dist = d;
                        candidates[cand_cnt].id   = neighbor;
                        cand_cnt++;
                    }
                    if (w_cnt < ef_search) {
                        W[w_cnt].dist = d;
                        W[w_cnt].id   = neighbor;
                        w_cnt++;
                    } else {
                        W[w_cnt - 1].dist = d;
                        W[w_cnt - 1].id   = neighbor;
                    }
                }
            }
        }
    }

    qsort(W, w_cnt, sizeof(DistPair), cmp_pair_asc);
    uint32_t ret = (w_cnt < top_k) ? w_cnt : top_k;
    res.returned_count = ret;
    for (uint32_t i = 0; i < ret; i++) {
        res.ids[i]   = id_mapping ? id_mapping[W[i].id] : W[i].id;
        res.dists[i] = W[i].dist;
    }
    res.distance_comps = dist_comps;
    res.visited_nodes  = visited_cnt;

    visited_free(vset);
    return res;
}

/*
 * H2: Filter-Aware con Nodos de Tránsito
 * - Todos los nodos pueden ser expandidos para mantener conectividad del grafo.
 * - SOLO los nodos que cumplen con el filtro ingresan al conjunto de resultados W.
 */
SearchResult hnsw_search_h2_filteraware(
    const HnswIndex *index,
    const float *query,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    uint32_t top_k,
    uint32_t ef_search
) {
    SearchResult res;
    memset(&res, 0, sizeof(SearchResult));
    res.k = top_k;

    if (index->enter_node == -1) return res;

    uint64_t dist_comps = 0;
    uint64_t visited_cnt = 0;

    int32_t curr_obj = index->enter_node;
    float curr_dist = l2_distance_avx2(query, &index->vectors[curr_obj * index->dim], index->dim);
    dist_comps++;

    /* Descenso normal por capas superiores para entrar en la cuenca de atracción */
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

    VisitedSet *vset = visited_create(index->max_elements);
    visited_test_and_set(vset, (uint32_t)curr_obj);
    visited_cnt++;

    DistPair candidates[512];
    DistPair W[512];
    uint32_t cand_cnt = 0, w_cnt = 0;

    DistPair initial = { curr_dist, (uint32_t)curr_obj };
    hnsw_candidates_push(candidates, &cand_cnt, 512U, initial);

    /* Solo se inserta en W si cumple el predicado */
    if (filter_matches(attrs, (uint32_t)curr_obj, pred)) {
        hnsw_results_insert(W, &w_cnt, ef_search, initial);
    }

    while (cand_cnt > 0) {
        DistPair c = hnsw_candidates_pop(candidates, &cand_cnt);

        if (w_cnt >= ef_search) {
            DistPair furthest_res = hnsw_results_worst(W, w_cnt);
            if (distpair_compare_asc(&c, &furthest_res) > 0) break;
        }

        const HnswNode *n = &index->nodes[c.id];
        for (uint16_t i = 0; i < n->friend_count[0]; i++) {
            uint32_t neighbor = n->friends[0][i];
            if (!visited_test_and_set(vset, neighbor)) {
                visited_cnt++;
                float d = l2_distance_avx2(query, &index->vectors[neighbor * index->dim], index->dim);
                dist_comps++;

                bool eligible = filter_matches(attrs, neighbor, pred);

                /* Expansión a candidates (tránsito) */
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

                /* Inserción en resultados elegibles W */
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

    visited_free(vset);
    return res;
}


/* Variante experimental: navegación SQ8 y reranking float32 de W. */
SearchResult hnsw_search_h2_sq8_filteraware(
    const HnswIndex *index,
    const Sq8Index *sq8,
    const float *query,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    uint32_t top_k,
    uint32_t ef_search
) {
    SearchResult res;
    memset(&res, 0, sizeof(SearchResult));
    res.k = top_k;

    if (index->enter_node == -1 || !sq8 || sq8->num_vectors < index->cur_elements || sq8->dim != index->dim) return res;

    uint64_t dist_comps = 0;
    uint64_t visited_cnt = 0;
    uint8_t query_codes[VECTOR_DIM];
    if (index->dim > VECTOR_DIM) return res;
    sq8_encode_query(sq8, query, query_codes);

    int32_t curr_obj = index->enter_node;
    float curr_dist = l2_distance_sq8_avx2(sq8, query_codes, (uint32_t)curr_obj);
    dist_comps++;

    /* Descenso normal por capas superiores para entrar en la cuenca de atracción */
    for (int32_t l = index->max_level; l > 0; l--) {
        bool changed = true;
        while (changed) {
            changed = false;
            const HnswNode *n = &index->nodes[curr_obj];
            for (uint16_t i = 0; i < n->friend_count[l]; i++) {
                uint32_t neighbor = n->friends[l][i];
                float d = l2_distance_sq8_avx2(sq8, query_codes, neighbor);
                dist_comps++;
                if (d < curr_dist) {
                    curr_dist = d;
                    curr_obj = (int32_t)neighbor;
                    changed = true;
                }
            }
        }
    }

    VisitedSet *vset = visited_create(index->max_elements);
    visited_test_and_set(vset, (uint32_t)curr_obj);
    visited_cnt++;

    DistPair candidates[512];
    DistPair W[512];
    uint32_t cand_cnt = 0, w_cnt = 0;

    DistPair initial = { curr_dist, (uint32_t)curr_obj };
    hnsw_candidates_push(candidates, &cand_cnt, 512U, initial);

    /* Solo se inserta en W si cumple el predicado */
    if (filter_matches(attrs, (uint32_t)curr_obj, pred)) {
        hnsw_results_insert(W, &w_cnt, ef_search, initial);
    }

    while (cand_cnt > 0) {
        DistPair c = hnsw_candidates_pop(candidates, &cand_cnt);

        if (w_cnt >= ef_search) {
            DistPair furthest_res = hnsw_results_worst(W, w_cnt);
            if (distpair_compare_asc(&c, &furthest_res) > 0) break;
        }

        const HnswNode *n = &index->nodes[c.id];
        for (uint16_t i = 0; i < n->friend_count[0]; i++) {
            uint32_t neighbor = n->friends[0][i];
            if (!visited_test_and_set(vset, neighbor)) {
                visited_cnt++;
                float d = l2_distance_sq8_avx2(sq8, query_codes, neighbor);
                dist_comps++;

                bool eligible = filter_matches(attrs, neighbor, pred);

                /* Expansión a candidates (tránsito) */
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

                /* Inserción en resultados elegibles W */
                if (eligible) {
                    DistPair eligible_pair = { d, neighbor };
                    hnsw_results_insert(W, &w_cnt, ef_search, eligible_pair);
                }
            }
        }
    }

    /* Reranking exacto float32 de todos los candidatos elegibles retenidos en W.
       No recupera candidatos descartados durante navegación SQ8; solo corrige su orden. */
    for (uint32_t i = 0; i < w_cnt; ++i) {
        W[i].dist = l2_distance_avx2(query, &index->vectors[(size_t)W[i].id * index->dim], index->dim);
    }
    res.rerank_comps = w_cnt;
    hnsw_results_sort(W, w_cnt);
    uint32_t ret = (w_cnt < top_k) ? w_cnt : top_k;
    res.returned_count = ret;
    for (uint32_t i = 0; i < ret; i++) {
        res.ids[i]   = W[i].id;
        res.dists[i] = W[i].dist;
    }
    res.distance_comps = dist_comps;
    res.visited_nodes  = visited_cnt;

    visited_free(vset);
    return res;
}

/* ============================================================================
 * HITO 5: H2-AT (Adaptive Transit)
 * Separa explícitamente el presupuesto de exploración (ef_explore)
 * del presupuesto de resultados elegibles retenidos (ef_result).
 * ============================================================================ */
#if defined(_MSC_VER) && !defined(__clang__)
static __declspec(thread) DistPair s_at_candidates[16384];
#else
static _Thread_local DistPair s_at_candidates[16384];
#endif

SearchResult hnsw_search_h2_at_sq8(
    const HnswIndex *index,
    const struct Sq8Index *sq8,
    const float *query,
    const AttributeSet *attrs,
    const FilterPredicate *pred,
    uint32_t top_k,
    uint32_t ef_result,
    uint32_t ef_explore
) {
    SearchResult res;
    memset(&res, 0, sizeof(SearchResult));
    res.k = top_k;

    if (index->enter_node == -1 || !sq8 || sq8->num_vectors < index->cur_elements || sq8->dim != index->dim) return res;
    if (ef_result > 1024U) ef_result = 1024U;
    if (ef_explore > 16384U) ef_explore = 16384U;

    uint64_t dist_comps = 0;
    uint64_t visited_cnt = 0;
    uint64_t expanded_cnt = 0;
    uint64_t transit_cnt = 0;

    uint8_t query_codes[VECTOR_DIM];
    if (index->dim > VECTOR_DIM) return res;
    sq8_encode_query(sq8, query, query_codes);

    int32_t curr_obj = index->enter_node;
    float curr_dist = l2_distance_sq8_avx2(sq8, query_codes, (uint32_t)curr_obj);
    dist_comps++;

    /* Descenso por capas superiores */
    for (int32_t l = index->max_level; l > 0; l--) {
        bool changed = true;
        while (changed) {
            changed = false;
            const HnswNode *n = &index->nodes[curr_obj];
            for (uint16_t i = 0; i < n->friend_count[l]; i++) {
                uint32_t neighbor = n->friends[l][i];
                float d = l2_distance_sq8_avx2(sq8, query_codes, neighbor);
                dist_comps++;
                if (d < curr_dist) {
                    curr_dist = d;
                    curr_obj = (int32_t)neighbor;
                    changed = true;
                }
            }
        }
    }

    VisitedSet *vset = visited_create(index->max_elements);
    visited_test_and_set(vset, (uint32_t)curr_obj);
    visited_cnt++;

    DistPair *candidates = s_at_candidates;
    DistPair W[1024];
    uint32_t cand_cnt = 0, w_cnt = 0;

    DistPair initial = { curr_dist, (uint32_t)curr_obj };
    hnsw_candidates_push(candidates, &cand_cnt, 16384U, initial);

    /* Solo ingresa en W si cumple el predicado */
    if (filter_matches(attrs, (uint32_t)curr_obj, pred)) {
        hnsw_results_insert(W, &w_cnt, ef_result, initial);
    }

    while (cand_cnt > 0) {
        if (expanded_cnt >= ef_explore) break;

        /* Si W ya está lleno, comprobamos si el mejor candidato en la frontera
           es peor que el peor resultado elegible retenido. Si lo es, no puede mejorar W. */
        if (w_cnt >= ef_result) {
            DistPair worst_res = hnsw_results_worst(W, w_cnt);
            if (distpair_compare_asc(&candidates[0], &worst_res) > 0) break;
        }

        DistPair c = hnsw_candidates_pop(candidates, &cand_cnt);
        expanded_cnt++;

        bool c_eligible = filter_matches(attrs, c.id, pred);
        if (!c_eligible) transit_cnt++;

        const HnswNode *n = &index->nodes[c.id];
        for (uint16_t i = 0; i < n->friend_count[0]; i++) {
            uint32_t neighbor = n->friends[0][i];
            if (!visited_test_and_set(vset, neighbor)) {
                visited_cnt++;
                float d = l2_distance_sq8_avx2(sq8, query_codes, neighbor);
                dist_comps++;

                bool eligible = filter_matches(attrs, neighbor, pred);

                /* Expansión a frontera de tránsito */
                bool should_expand = true;
                if (w_cnt >= ef_result) {
                    DistPair worst = hnsw_results_worst(W, w_cnt);
                    DistPair candidate = { d, neighbor };
                    should_expand = (distpair_compare_asc(&candidate, &worst) < 0);
                }

                if (should_expand && cand_cnt < 16380U) {
                    DistPair candidate = { d, neighbor };
                    hnsw_candidates_push(candidates, &cand_cnt, 16384U, candidate);
                }

                /* Retención en resultados elegibles */
                if (eligible) {
                    DistPair eligible_pair = { d, neighbor };
                    hnsw_results_insert(W, &w_cnt, ef_result, eligible_pair);
                }
            }
        }
    }

    /* Reranking exacto float32 de todos los candidatos elegibles retenidos en W */
    for (uint32_t i = 0; i < w_cnt; ++i) {
        W[i].dist = l2_distance_avx2(query, &index->vectors[(size_t)W[i].id * index->dim], index->dim);
    }
    res.rerank_comps = w_cnt;
    hnsw_results_sort(W, w_cnt);
    uint32_t ret = (w_cnt < top_k) ? w_cnt : top_k;
    res.returned_count = ret;
    for (uint32_t i = 0; i < ret; i++) {
        res.ids[i]   = W[i].id;
        res.dists[i] = W[i].dist;
    }
    res.distance_comps = dist_comps;
    res.visited_nodes  = visited_cnt;
    res.expanded_nodes = expanded_cnt;
    res.transit_nodes  = transit_cnt;

    visited_free(vset);
    return res;
}
