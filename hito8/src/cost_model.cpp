#include "../include/cost_model.h"
#include <cmath>
#include <limits>

extern "C" void cost_model_default_exact(ExactCostModel *m) {
    if (!m) return;
    /* Hito 7 owner-reported fit; calibrated for the current i5-13500 benchmark only. */
    m->exact_intercept_us = 15.93;
    m->exact_per_eligible_us = 0.00895;
    m->min_eligible_count = 9000;
    m->max_eligible_count = 26000;
    m->calibrated = true;
}

static const HnswCalibrationPoint s_default_hnsw_points[] = {
    /* R_Local multi-seed (1..5) calibrated points on i5-13500 */
    {QUERY_LOCAL, 10.0, 50, 160.34, 0.9707, 0.9532, 0.9653, 0.9503, 5, 5000},
    {QUERY_LOCAL, 12.5, 50, 140.08, 0.9654, 0.9479, 0.9620, 0.9456, 5, 5000},
    {QUERY_LOCAL, 15.0, 50, 122.42, 0.9592, 0.9431, 0.9561, 0.9395, 5, 5000},
    {QUERY_LOCAL, 17.5, 50, 114.02, 0.9536, 0.9388, 0.9509, 0.9355, 5, 5000},
    {QUERY_LOCAL, 20.0, 50, 100.90, 0.9493, 0.9349, 0.9453, 0.9305, 5, 5000},
    {QUERY_LOCAL, 22.5, 50,  95.68, 0.9458, 0.9311, 0.9411, 0.9262, 5, 5000},
    {QUERY_LOCAL, 25.0, 50,  90.34, 0.9421, 0.9279, 0.9366, 0.9221, 5, 5000},
    /* D_Adversarial multi-seed (1..5) calibrated points on i5-13500 */
    {QUERY_ADVERSARIAL, 10.0, 50, 361.20, 0.8333, 0.7218, 0.8300, 0.7189, 5, 5000},
    {QUERY_ADVERSARIAL, 12.5, 50, 295.20, 0.7928, 0.6868, 0.7877, 0.6798, 5, 5000},
    {QUERY_ADVERSARIAL, 15.0, 50, 257.90, 0.7556, 0.6576, 0.7488, 0.6504, 5, 5000},
    {QUERY_ADVERSARIAL, 17.5, 50, 227.20, 0.7239, 0.6322, 0.7191, 0.6237, 5, 5000},
    {QUERY_ADVERSARIAL, 20.0, 50, 214.30, 0.6947, 0.6100, 0.6885, 0.6001, 5, 5000},
    {QUERY_ADVERSARIAL, 22.5, 50, 189.90, 0.6720, 0.5903, 0.6653, 0.5826, 5, 5000},
    {QUERY_ADVERSARIAL, 25.0, 50, 177.30, 0.6508, 0.5724, 0.6434, 0.5654, 5, 5000}
};

extern "C" void cost_model_default_hnsw(HnswCostModel *m) {
    if (!m) return;
    m->points = s_default_hnsw_points;
    m->point_count = sizeof(s_default_hnsw_points) / sizeof(s_default_hnsw_points[0]);
}

extern "C" double cost_model_exact_predict(const ExactCostModel *m, uint32_t n) {
    if (!m || !m->calibrated || n < m->min_eligible_count || n > m->max_eligible_count)
        return std::numeric_limits<double>::infinity();
    return m->exact_intercept_us + m->exact_per_eligible_us * n;
}

extern "C" EngineEstimate cost_model_hnsw_predict(const HnswCostModel *m, QueryRegime regime, double sel, uint32_t k) {
    EngineEstimate o{};
    o.cost_us = std::numeric_limits<double>::infinity();
    if (!m || !m->points || !m->point_count || regime == QUERY_UNKNOWN || !std::isfinite(sel))
        return o;

    const HnswCalibrationPoint *a = nullptr, *b = nullptr;
    for (size_t i = 0; i < m->point_count; i++) {
        const auto &p = m->points[i];
        if (p.regime != regime || !std::isfinite(p.p50_us)) continue;
        if (p.selectivity_pct <= sel && (!a || p.selectivity_pct > a->selectivity_pct)) a = &p;
        if (p.selectivity_pct >= sel && (!b || p.selectivity_pct < b->selectivity_pct)) b = &p;
    }

    if (!a || !b) return o; /* No extrapolation outside calibration domain. */

    double t = (a == b) ? 0.0 : (sel - a->selectivity_pct) / (b->selectivity_pct - a->selectivity_pct);
    o.cost_us = a->p50_us + t * (b->p50_us - a->p50_us);
    o.recall10 = a->recall10 + t * (b->recall10 - a->recall10);
    o.recall100 = a->recall100 + t * (b->recall100 - a->recall100);
    o.recall_lcb = a->recall10_lcb + t * (b->recall10_lcb - a->recall10_lcb);
    o.budget = (t < 0.5) ? a->ef_search : b->ef_search;
    o.supported = true;
    (void)k;
    return o;
}
