#ifndef HITO8_COST_MODEL_H
#define HITO8_COST_MODEL_H

#include "hybrid_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ExactCostModel {
    double exact_intercept_us;
    double exact_per_eligible_us;
    double min_eligible_count;
    double max_eligible_count;
    bool calibrated;
} ExactCostModel;

typedef struct HnswCalibrationPoint {
    QueryRegime regime;
    double selectivity_pct;
    uint32_t ef_search;
    double p50_us;
    double recall10;
    double recall100;
    double recall10_lcb;
    double recall100_lcb;
    uint32_t seed_count;
    uint64_t query_count;
} HnswCalibrationPoint;

typedef struct HnswCostModel {
    const HnswCalibrationPoint *points;
    size_t point_count;
} HnswCostModel;

typedef struct {
    double cost_us;
    double recall10;
    double recall100;
    double recall_lcb;
    uint32_t budget;
    bool supported;
} EngineEstimate;

void cost_model_default_exact(ExactCostModel *model);
void cost_model_default_hnsw(HnswCostModel *model);
double cost_model_exact_predict(const ExactCostModel *model, uint32_t eligible_count);
EngineEstimate cost_model_hnsw_predict(const HnswCostModel *model,
                                       QueryRegime regime,
                                       double selectivity_pct,
                                       uint32_t k);

#ifdef __cplusplus
}
#endif
#endif
