#ifndef HITO9_ONLINE_CALIBRATOR_H
#define HITO9_ONLINE_CALIBRATOR_H

#include "../../hito8/include/hybrid_engine.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Estados del autómata de calibración por estrato */
typedef enum {
    STRATUM_STATE_CALIBRATED = 0,    /* Nominal: estimaciones confiables, baja dispersión de residuales */
    STRATUM_STATE_SUSPECT = 1,       /* Sospechoso: residual anómalo detectado (> tripwire sigma) */
    STRATUM_STATE_FALLBACK_EXACT = 2,/* Fallback seguro: desvío persistente, enrutar forzosamente a ExactScan */
    STRATUM_STATE_RECOVERY_PROBE = 3 /* Sonda de recuperación: cooldown expirado, testeando probes controladas */
} StratumState;

/* Causas de transición o disparo */
typedef enum {
    DRIFT_NONE = 0,
    DRIFT_LATENCY_SPIKE,      /* Residual individual > tripwire_sigma */
    DRIFT_LATENCY_PERSISTENT, /* Residuales repetidos o corrimiento persistente */
    DRIFT_QUALITY_DEGRADATION,/* Recall auditado por debajo del objetivo */
    RECOVERY_SUCCESS,         /* Probes en recuperación dentro de límites normales */
    RECOVERY_FAILED           /* Probes en recuperación continúan degradadas */
} DriftReason;

/* Configuración del calibrador de latencia */
typedef struct {
    size_t window_size;              /* Tamaño de la ventana deslizante (default: 50) */
    size_t min_samples;              /* Mínimo de muestras para activar detección (default: 10) */
    double tripwire_sigma;           /* Múltiplo de escala robusta (MAD * 1.4826) para alerta (default: 2.0) */
    uint32_t suspect_patience;       /* Cantidad de anomalías consecutivas antes de pasar a FALLBACK_EXACT (default: 3) */
    uint32_t fallback_cooldown;      /* Consultas en FALLBACK_EXACT antes de pasar a RECOVERY_PROBE (default: 20) */
    uint32_t recovery_probe_count;   /* Muestras de prueba requeridas para validar recuperación (default: 5) */
    double recovery_tolerance_sigma; /* Tolerancia durante recuperación (default: 1.5) */
} LatencyCalibratorConfig;

/* Estadísticas calculadas sobre la ventana */
typedef struct {
    double mean;                     /* Media muestral */
    double stddev;                   /* Desviación estándar muestral */
    double median;                   /* Mediana de residuales (e_tilde) */
    double mad;                      /* Median Absolute Deviation (MAD) */
    double robust_sigma;             /* 1.4826 * MAD */
    size_t sample_count;             /* Cantidad de muestras actuales en ventana */
} StratumResidualStats;

/* Estado y métricas de un estrato */
typedef struct {
    StratumState state;
    QueryRegime regime;
    uint32_t selectivity_bp;        /* Selectividad en basis points (ej: 1250 = 12.5%) */
    HybridEngineId engine;
    
    StratumResidualStats stats;
    
    uint32_t consecutive_anomalies;
    uint32_t fallback_remaining;
    uint32_t recovery_probes_done;
    uint32_t recovery_probes_passed;
    
    uint64_t total_queries;
    uint64_t false_alarms;
    uint64_t fallback_events;
    uint64_t total_fallback_queries;
} StratumCalibrator;

/* Calibrador de latencia multirégimen/multiestrato */
typedef struct LatencyCalibrator LatencyCalibrator;

LatencyCalibrator* latency_calibrator_create(const LatencyCalibratorConfig *config);
void latency_calibrator_free(LatencyCalibrator *cal);
void latency_calibrator_reset(LatencyCalibrator *cal);
void latency_calibrator_default_config(LatencyCalibratorConfig *config);

/* Registrar una observación de latencia:
   - actual_latency_us: latencia real observada
   - estimated_cost_us: latencia predicha por el modelo de costo
   Retorna el estado del estrato luego de procesar la muestra.
*/
StratumState latency_calibrator_record(LatencyCalibrator *cal,
                                       QueryRegime regime,
                                       uint32_t selectivity_bp,
                                       HybridEngineId engine,
                                       double actual_latency_us,
                                       double estimated_cost_us,
                                       DriftReason *out_reason);

/* Notificar que una consulta en este estrato fue derivada a ExactScan debido al estado de fallback */
void latency_calibrator_notify_fallback_served(LatencyCalibrator *cal,
                                               QueryRegime regime,
                                               uint32_t selectivity_bp,
                                               HybridEngineId engine);

/* Consultar si para un estrato dado se debe forzar fallback seguro a ExactScan */
bool latency_calibrator_should_fallback(const LatencyCalibrator *cal,
                                        QueryRegime regime,
                                        uint32_t selectivity_bp,
                                        HybridEngineId engine);

/* Obtener estadísticas y estado de un estrato */
bool latency_calibrator_get_stratum(const LatencyCalibrator *cal,
                                    QueryRegime regime,
                                    uint32_t selectivity_bp,
                                    HybridEngineId engine,
                                    StratumCalibrator *out_stratum);

/* --- Módulo B: Auditoría de Calidad (Recall Audit Harness) --- */

typedef struct {
    double target_recall;            /* Umbral de calidad requerido (ej: 0.95) */
    double audit_rate;               /* Fracción de consultas auditadas en shadow (ej: 0.05 = 5%) */
    size_t audit_window_size;        /* Tamaño de ventana para auditar recall (default: 50) */
    size_t min_audit_samples;        /* Mínimo de auditorías para evaluación (default: 10) */
    double confidence_z;             /* Factor Z para LCB (ej: 1.645 para 95% unilateral) */
} QualityAuditConfig;

typedef struct QualityAuditHarness QualityAuditHarness;

QualityAuditHarness* quality_audit_create(const QualityAuditConfig *config);
void quality_audit_free(QualityAuditHarness *audit);
void quality_audit_reset(QualityAuditHarness *audit);
void quality_audit_default_config(QualityAuditConfig *config);

/* Determina determinísticamente si una consulta debe ser auditada en shadow mode */
bool quality_audit_should_sample(QualityAuditHarness *audit, uint64_t query_index);

/* Registra el resultado auditado: recall medido contra Ground Truth o ExactScan */
void quality_audit_record(QualityAuditHarness *audit,
                          QueryRegime regime,
                          uint32_t selectivity_bp,
                          double observed_recall,
                          double shadow_overhead_us);

/* Consulta si la calidad en un estrato ha degradado por debajo de target_recall */
bool quality_audit_is_degraded(const QualityAuditHarness *audit,
                               QueryRegime regime,
                               uint32_t selectivity_bp,
                               double *out_empirical_lcb);

/* Métricas globales de la auditoría */
typedef struct {
    uint64_t total_queries;
    uint64_t audited_queries;
    double total_shadow_overhead_us;
    uint32_t quality_fallback_triggers;
} QualityAuditMetrics;

void quality_audit_get_metrics(const QualityAuditHarness *audit, QualityAuditMetrics *out_metrics);

#ifdef __cplusplus
}
#endif

#endif /* HITO9_ONLINE_CALIBRATOR_H */
