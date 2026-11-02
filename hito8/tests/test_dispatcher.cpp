#include "../include/hybrid_engine.h"
#include "../include/cost_model.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdio>

static int mock_exact_search(void *ctx, const float *q, const void *pred, uint32_t k, uint32_t budget, HybridSearchOutput *out) {
    (void)ctx; (void)q; (void)pred; (void)budget;
    out->count = (k <= out->capacity) ? k : out->capacity;
    for (uint32_t i = 0; i < out->count; i++) {
        out->neighbors[i] = {i, (float)i * 0.1f};
    }
    return 0;
}

static int mock_hnsw_search(void *ctx, const float *q, const void *pred, uint32_t k, uint32_t budget, HybridSearchOutput *out) {
    (void)ctx; (void)q; (void)pred; (void)budget;
    out->count = (k <= out->capacity) ? k : out->capacity;
    for (uint32_t i = 0; i < out->count; i++) {
        out->neighbors[i] = {i + 100, (float)i * 0.1f};
    }
    return 0;
}

int main() {
    printf("=== Test: Hybrid Query Dispatcher ===\n");

    HybridEngineBindings bindings{};
    bindings.exact_scan = {nullptr, mock_exact_search, 0, true};
    bindings.hnswlib    = {nullptr, mock_hnsw_search, 50, true};
    bindings.alg_h2_at  = {nullptr, nullptr, 512, false};

    ExactCostModel em{};
    cost_model_default_exact(&em);

    HnswCalibrationPoint pts[] = {
        {QUERY_LOCAL, 10.0, 50, 160.0, 0.970, 0.90, 0.965, 0.0, 5, 5000},
        {QUERY_LOCAL, 15.0, 50, 122.0, 0.959, 0.88, 0.952, 0.0, 5, 5000},
        {QUERY_LOCAL, 25.0, 50,  90.0, 0.942, 0.85, 0.935, 0.0, 5, 5000},
        {QUERY_ADVERSARIAL, 10.0, 50, 360.0, 0.83, 0.70, 0.81, 0.0, 5, 5000},
        {QUERY_ADVERSARIAL, 25.0, 50, 177.0, 0.65, 0.55, 0.63, 0.0, 5, 5000}
    };
    HnswCostModel hm{pts, 5};

    float dummy_q[64] = {0};
    HybridNeighbor nbuf[100];
    HybridSearchOutput out{nbuf, 100, 0};
    HybridQueryResult meta{};

    // Case 1: EXACT policy always chooses ExactScan
    {
        QueryFeatures f{20000, 100000, 64, QUERY_LOCAL, 20.0};
        QualityRequirements r{POLICY_EXACT, 0.95, 0.0, 10};
        int rc = hybrid_search(&bindings, &f, &r, &em, &hm, dummy_q, nullptr, &out, &meta);
        assert(rc == 0);
        assert(meta.chosen_engine == ENGINE_EXACT_SCAN);
        assert(!meta.fallback_triggered);
        assert(meta.estimated_recall == 1.0);
    }

    // Case 2: BALANCED policy with target_recall=0.95 at 12.5% in QUERY_LOCAL
    // Exact cost = 15.93 + 0.00895 * 12500 = 127.8 us
    // Hnsw cost  = (160 + 122)/2 = 141.0 us
    // Since HNSW cost (141.0) > Exact cost (127.8), it must choose ExactScan!
    {
        QueryFeatures f{12500, 100000, 64, QUERY_LOCAL, 12.5};
        QualityRequirements r{POLICY_BALANCED, 0.95, 0.0, 10};
        int rc = hybrid_search(&bindings, &f, &r, &em, &hm, dummy_q, nullptr, &out, &meta);
        assert(rc == 0);
        assert(meta.chosen_engine == ENGINE_EXACT_SCAN);
        assert(meta.estimated_cost_us < 130.0);
    }

    // Case 3: BALANCED policy with target_recall=0.95 at 20.0% in QUERY_LOCAL
    // Exact cost = 15.93 + 0.00895 * 20000 = 194.9 us
    // Hnsw cost  = 122 - 0.5*(122 - 90) = 106.0 us < 194.9 us
    // Recall LCB = 0.952 - 0.5*(0.952 - 0.935) = 0.9435 < 0.95
    // Since LCB (0.9435) < target (0.95), BALANCED must fallback to ExactScan!
    {
        QueryFeatures f{20000, 100000, 64, QUERY_LOCAL, 20.0};
        QualityRequirements r{POLICY_BALANCED, 0.95, 0.0, 10};
        int rc = hybrid_search(&bindings, &f, &r, &em, &hm, dummy_q, nullptr, &out, &meta);
        assert(rc == 0);
        assert(meta.chosen_engine == ENGINE_EXACT_SCAN);
    }

    // Case 4: APPROXIMATE policy at 20.0% in QUERY_LOCAL
    // Hnsw cost (106 us) < Exact cost (194.9 us) -> chooses HNSWLIB
    {
        QueryFeatures f{20000, 100000, 64, QUERY_LOCAL, 20.0};
        QualityRequirements r{POLICY_APPROXIMATE, 0.0, 0.0, 10};
        int rc = hybrid_search(&bindings, &f, &r, &em, &hm, dummy_q, nullptr, &out, &meta);
        assert(rc == 0);
        assert(meta.chosen_engine == ENGINE_HNSWLIB);
        assert(meta.estimated_cost_us < 115.0);
    }

    // Case 5: QUERY_ADVERSARIAL at 15.0%
    // Exact cost = 15.93 + 0.00895 * 15000 = 150.2 us
    // HNSW in D at 15% costs ~258 us -> ExactScan must be chosen!
    {
        QueryFeatures f{15000, 100000, 64, QUERY_ADVERSARIAL, 15.0};
        QualityRequirements r{POLICY_APPROXIMATE, 0.0, 0.0, 10};
        int rc = hybrid_search(&bindings, &f, &r, &em, &hm, dummy_q, nullptr, &out, &meta);
        assert(rc == 0);
        assert(meta.chosen_engine == ENGINE_EXACT_SCAN);
    }

    // Microbenchmark of dispatcher overhead
    {
        QueryFeatures f{20000, 100000, 64, QUERY_LOCAL, 20.0};
        QualityRequirements r{POLICY_APPROXIMATE, 0.0, 0.0, 10};
        const int iters = 100000;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            hybrid_search(&bindings, &f, &r, &em, &hm, dummy_q, nullptr, &out, &meta);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double per_call_ns = total_ns / iters;
        printf("Dispatcher overhead: %.2f ns per query (%.4f us)\n", per_call_ns, per_call_ns / 1000.0);
        assert(per_call_ns < 1000.0); // Strict requirement: < 1.0 us
    }

    printf("Todos los tests de hybrid_engine pasaron exitosamente.\n");
    return 0;
}
