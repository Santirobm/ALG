#include "../include/engine_bindings.h"
#include "../../hito7/third_party/hnswlib/hnswlib/hnswlib.h"
#include <immintrin.h>
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

struct DistPair {
    float dist;
    uint32_t id;
    bool operator<(const DistPair &other) const {
        return dist < other.dist;
    }
};

static inline float l2_distance_avx2(const float *a, const float *b, uint32_t dim) {
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d  = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

class CategoryFilter : public hnswlib::BaseFilterFunctor {
    const uint32_t *categories_;
    uint32_t target_;
public:
    CategoryFilter(const uint32_t *categories, uint32_t target)
        : categories_(categories), target_(target) {}
    bool operator()(hnswlib::labeltype label_id) override {
        return categories_[label_id] < target_;
    }
};

extern "C" int exact_scan_binding_search(void *ctx, const float *q, const void *pred, uint32_t k, uint32_t budget, HybridSearchOutput *out) {
    (void)budget;
    if (!ctx || !q || !pred || !out || k == 0 || out->capacity < k) return -1;
    const auto *ectx = static_cast<const ExactEngineContext*>(ctx);
    const auto *fctx = static_cast<const FilterQueryContext*>(pred);

    std::priority_queue<DistPair> pq;
    const uint32_t dim = ectx->dim;

    if (fctx->eligible_ids && fctx->eligible_count > 0) {
        for (uint32_t i = 0; i < fctx->eligible_count; i++) {
            uint32_t id = fctx->eligible_ids[i];
            float d = l2_distance_avx2(q, ectx->vectors + (size_t)id * dim, dim);
            if (pq.size() < k) {
                pq.push({d, id});
            } else if (d < pq.top().dist) {
                pq.pop();
                pq.push({d, id});
            }
        }
    } else {
        for (uint32_t id = 0; id < ectx->num_vectors; id++) {
            if (ectx->categories[id] < fctx->target_category) {
                float d = l2_distance_avx2(q, ectx->vectors + (size_t)id * dim, dim);
                if (pq.size() < k) {
                    pq.push({d, id});
                } else if (d < pq.top().dist) {
                    pq.pop();
                    pq.push({d, id});
                }
            }
        }
    }

    uint32_t count = (uint32_t)pq.size();
    out->count = count;
    for (int i = (int)count - 1; i >= 0; i--) {
        out->neighbors[i] = {pq.top().id, pq.top().dist};
        pq.pop();
    }
    return 0;
}

extern "C" int hnswlib_binding_search(void *ctx, const float *q, const void *pred, uint32_t k, uint32_t budget, HybridSearchOutput *out) {
    if (!ctx || !q || !pred || !out || k == 0 || out->capacity < k) return -1;
    const auto *hctx = static_cast<const HnswlibEngineContext*>(ctx);
    const auto *fctx = static_cast<const FilterQueryContext*>(pred);
    auto *index = reinterpret_cast<hnswlib::HierarchicalNSW<float>*>(hctx->hnsw_index);
    if (!index) return -2;

    if (budget > 0) {
        index->setEf(budget);
    }

    CategoryFilter filter(hctx->categories, fctx->target_category);
    auto res = index->searchKnn(q, k, &filter);

    uint32_t count = (uint32_t)res.size();
    out->count = count;
    for (int i = (int)count - 1; i >= 0; i--) {
        out->neighbors[i] = {(uint32_t)res.top().second, res.top().first};
        res.pop();
    }
    return 0;
}

extern "C" QueryRegime extract_query_regime(const QueryFeatureProbe *probe, const float *query) {
    if (!probe || !probe->probe_centroids || !probe->num_probes || !query) return QUERY_UNKNOWN;
    float min_d = std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < probe->num_probes; i++) {
        float d = l2_distance_avx2(query, probe->probe_centroids + (size_t)i * probe->dim, probe->dim);
        if (d < min_d) {
            min_d = d;
            if (min_d < probe->distance_threshold_sq) {
                return QUERY_LOCAL;
            }
        }
    }
    return (min_d < probe->distance_threshold_sq) ? QUERY_LOCAL : QUERY_ADVERSARIAL;
}
