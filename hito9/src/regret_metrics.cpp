#include "../include/regret_metrics.h"
#include <algorithm>

RegretEvaluation evaluate_regret(DispatcherPolicy policy,
                                 double actual_latency_us,
                                 double exact_latency_us,
                                 double hnsw_latency_us,
                                 bool hnsw_quality_admissible,
                                 HybridEngineId chosen_engine) {
    RegretEvaluation res;

    /* 1. Oráculo de diagnóstico no restringido (mínimo de latencia puro) */
    res.unconstrained_oracle_latency_us = std::min(exact_latency_us, hnsw_latency_us);
    res.diagnostic_signed_regret_us = actual_latency_us - res.unconstrained_oracle_latency_us;
    res.diagnostic_positive_regret_us = (res.diagnostic_signed_regret_us > 0.0) ? res.diagnostic_signed_regret_us : 0.0;

    /* 2. Oráculo admisible por política */
    res.is_candidate_admissible = hnsw_quality_admissible;

    switch (policy) {
        case POLICY_EXACT:
            /* Solo ExactScan es admisible para satisfacer 100% determinista */
            res.admissible_oracle_latency_us = exact_latency_us;
            if (chosen_engine == ENGINE_EXACT_SCAN) {
                /* Por definición teórica: el motor exacto alcanza el oráculo exacto;
                   el ruido de medición entre réplicas no debe computar como regret algorítmico */
                res.policy_signed_regret_us = 0.0;
                res.policy_positive_regret_us = 0.0;
            } else {
                res.policy_signed_regret_us = actual_latency_us - exact_latency_us;
                res.policy_positive_regret_us = (res.policy_signed_regret_us > 0.0) ? res.policy_signed_regret_us : 0.0;
            }
            break;

        case POLICY_BALANCED:
            /* HNSW es admisible únicamente si el arnés de calidad certifica LCB >= target_recall */
            if (hnsw_quality_admissible) {
                res.admissible_oracle_latency_us = std::min(exact_latency_us, hnsw_latency_us);
            } else {
                res.admissible_oracle_latency_us = exact_latency_us;
            }
            res.policy_signed_regret_us = actual_latency_us - res.admissible_oracle_latency_us;
            res.policy_positive_regret_us = (res.policy_signed_regret_us > 0.0) ? res.policy_signed_regret_us : 0.0;
            break;

        case POLICY_APPROXIMATE:
            /* Sin restricción de calidad: ambos motores son admisibles */
            res.admissible_oracle_latency_us = std::min(exact_latency_us, hnsw_latency_us);
            res.policy_signed_regret_us = actual_latency_us - res.admissible_oracle_latency_us;
            res.policy_positive_regret_us = (res.policy_signed_regret_us > 0.0) ? res.policy_signed_regret_us : 0.0;
            break;

        default:
            res.admissible_oracle_latency_us = exact_latency_us;
            res.policy_signed_regret_us = 0.0;
            res.policy_positive_regret_us = 0.0;
            break;
    }

    return res;
}
