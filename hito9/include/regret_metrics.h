#ifndef HITO9_REGRET_METRICS_H
#define HITO9_REGRET_METRICS_H

#include "../../hito8/include/hybrid_engine.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double diagnostic_signed_regret_us;   /* Regret frente al oráculo de latencia sin restricciones */
    double diagnostic_positive_regret_us; /* max(0, diagnostic_signed_regret) */
    double policy_signed_regret_us;       /* Regret frente al oráculo admisible por política de calidad */
    double policy_positive_regret_us;     /* max(0, policy_signed_regret) */
    double admissible_oracle_latency_us;  /* Latencia del oráculo admisible */
    double unconstrained_oracle_latency_us; /* Latencia del oráculo no restringido */
    bool is_candidate_admissible;         /* Indica si el motor alternativo (HNSW) era admisible ex-ante */
} RegretEvaluation;

/* Evalúa el regret separando la perspectiva de política (admisible) del diagnóstico no restringido.
   Para POLICY_EXACT, ExactScan es el único motor admisible; su regret de política es 0.0 por definición. */
RegretEvaluation evaluate_regret(DispatcherPolicy policy,
                                 double actual_latency_us,
                                 double exact_latency_us,
                                 double hnsw_latency_us,
                                 bool hnsw_quality_admissible,
                                 HybridEngineId chosen_engine);

#ifdef __cplusplus
}
#endif

#endif /* HITO9_REGRET_METRICS_H */
