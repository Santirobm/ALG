#include "../include/online_calibrator.h"
#include <vector>
#include <deque>
#include <map>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <cstring>

/* Clave para identificar el estrato */
typedef std::tuple<uint32_t, uint32_t, uint32_t> StratumKey;

static inline StratumKey make_key(QueryRegime regime, uint32_t selectivity_bp, HybridEngineId engine) {
    return std::make_tuple((uint32_t)regime, selectivity_bp, (uint32_t)engine);
}

typedef std::tuple<uint32_t, uint32_t> QualityStratumKey;
static inline QualityStratumKey make_quality_key(QueryRegime regime, uint32_t selectivity_bp) {
    return std::make_tuple((uint32_t)regime, selectivity_bp);
}

struct StratumStateData {
    StratumCalibrator info;
    std::deque<double> residuals;
};

struct LatencyCalibrator {
    LatencyCalibratorConfig config;
    std::map<StratumKey, StratumStateData> strata;
};

void latency_calibrator_default_config(LatencyCalibratorConfig *cfg) {
    if (!cfg) return;
    cfg->window_size = 50;
    cfg->min_samples = 10;
    cfg->tripwire_sigma = 2.0;
    cfg->suspect_patience = 3;
    cfg->fallback_cooldown = 20;
    cfg->recovery_probe_count = 5;
    cfg->recovery_tolerance_sigma = 1.5;
    cfg->conservative_unseen = true; /* Principio de precaución: desconocidos van a ExactScan */
}

LatencyCalibrator* latency_calibrator_create(const LatencyCalibratorConfig *config) {
    LatencyCalibrator *cal = new LatencyCalibrator();
    if (config) {
        cal->config = *config;
    } else {
        latency_calibrator_default_config(&cal->config);
    }
    return cal;
}

void latency_calibrator_free(LatencyCalibrator *cal) {
    if (cal) {
        delete cal;
    }
}

void latency_calibrator_reset(LatencyCalibrator *cal) {
    if (!cal) return;
    cal->strata.clear();
}

static void update_residual_stats(const std::deque<double> &residuals, StratumResidualStats &stats) {
    size_t n = residuals.size();
    stats.sample_count = n;
    if (n == 0) {
        stats.mean = 0.0;
        stats.stddev = 0.0;
        stats.median = 0.0;
        stats.mad = 0.0;
        stats.robust_sigma = 1.0;
        return;
    }

    double sum = 0.0;
    for (double r : residuals) sum += r;
    double mean = sum / n;
    stats.mean = mean;

    if (n > 1) {
        double sq_sum = 0.0;
        for (double r : residuals) {
            double d = r - mean;
            sq_sum += d * d;
        }
        stats.stddev = std::sqrt(sq_sum / (n - 1));
    } else {
        stats.stddev = 0.0;
    }

    std::vector<double> sorted(residuals.begin(), residuals.end());
    std::sort(sorted.begin(), sorted.end());

    double median = 0.0;
    if (n % 2 == 1) {
        median = sorted[n / 2];
    } else {
        median = 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    }
    stats.median = median;

    std::vector<double> dev(n);
    for (size_t i = 0; i < n; ++i) {
        dev[i] = std::fabs(sorted[i] - median);
    }
    std::sort(dev.begin(), dev.end());

    double mad = 0.0;
    if (n % 2 == 1) {
        mad = dev[n / 2];
    } else {
        mad = 0.5 * (dev[n / 2 - 1] + dev[n / 2]);
    }
    stats.mad = mad;

    /* Factor de consistencia normal 1.4826 para escala robusta */
    double robust_sigma = 1.4826 * mad;
    if (robust_sigma < 1.0) {
        robust_sigma = (stats.stddev >= 1.0) ? stats.stddev : 1.0;
    }
    stats.robust_sigma = robust_sigma;
}

StratumState latency_calibrator_record(LatencyCalibrator *cal,
                                       QueryRegime regime,
                                       uint32_t selectivity_bp,
                                       HybridEngineId engine,
                                       double actual_latency_us,
                                       double estimated_cost_us,
                                       DriftReason *out_reason) {
    if (!cal) return STRATUM_STATE_CALIBRATED;
    if (out_reason) *out_reason = DRIFT_NONE;

    StratumKey key = make_key(regime, selectivity_bp, engine);
    StratumStateData &data = cal->strata[key];
    StratumCalibrator &info = data.info;

    if (info.total_queries == 0) {
        info.state = STRATUM_STATE_CALIBRATED;
        info.regime = regime;
        info.selectivity_bp = selectivity_bp;
        info.engine = engine;
    }
    info.total_queries++;

    double residual = actual_latency_us - estimated_cost_us;

    /* En RECOVERY_PROBE se evalúa la probe antes de actualizar ventana nominal */
    if (info.state == STRATUM_STATE_RECOVERY_PROBE) {
        info.recovery_probes_done++;
        double tol = cal->config.recovery_tolerance_sigma * info.stats.robust_sigma;
        if (residual <= info.stats.median + tol) {
            info.recovery_probes_passed++;
        }

        if (info.recovery_probes_done >= cal->config.recovery_probe_count) {
            if (info.recovery_probes_passed == info.recovery_probes_done) {
                info.state = STRATUM_STATE_CALIBRATED;
                info.consecutive_anomalies = 0;
                if (out_reason) *out_reason = RECOVERY_SUCCESS;
            } else {
                info.state = STRATUM_STATE_FALLBACK_EXACT;
                info.fallback_remaining = cal->config.fallback_cooldown * 2; /* backoff exponencial */
                info.fallback_events++;
                if (out_reason) *out_reason = RECOVERY_FAILED;
            }
        }
        return info.state;
    }

    /* Mantenimiento de ventana deslizante */
    data.residuals.push_back(residual);
    if (data.residuals.size() > cal->config.window_size) {
        data.residuals.pop_front();
    }
    update_residual_stats(data.residuals, info.stats);

    /* Si no hay suficientes muestras, permanecer sin disparar falsas alarmas */
    if (info.stats.sample_count < cal->config.min_samples) {
        return info.state;
    }

    double tripwire_threshold = info.stats.median + (cal->config.tripwire_sigma * info.stats.robust_sigma);
    bool is_anomalous = (residual > tripwire_threshold);

    switch (info.state) {
        case STRATUM_STATE_CALIBRATED:
            if (is_anomalous) {
                info.state = STRATUM_STATE_SUSPECT;
                info.consecutive_anomalies = 1;
                if (out_reason) *out_reason = DRIFT_LATENCY_SPIKE;
            }
            break;

        case STRATUM_STATE_SUSPECT:
            if (is_anomalous) {
                info.consecutive_anomalies++;
                if (info.consecutive_anomalies >= cal->config.suspect_patience) {
                    info.state = STRATUM_STATE_FALLBACK_EXACT;
                    info.fallback_remaining = cal->config.fallback_cooldown;
                    info.fallback_events++;
                    if (out_reason) *out_reason = DRIFT_LATENCY_PERSISTENT;
                }
            } else {
                /* Outlier aislado disipado: retorno seguro a CALIBRATED */
                info.state = STRATUM_STATE_CALIBRATED;
                info.consecutive_anomalies = 0;
                info.false_alarms++;
                if (out_reason) *out_reason = RECOVERY_SUCCESS;
            }
            break;

        case STRATUM_STATE_FALLBACK_EXACT:
            if (info.fallback_remaining > 0) {
                info.fallback_remaining--;
            }
            if (info.fallback_remaining == 0) {
                info.state = STRATUM_STATE_RECOVERY_PROBE;
                info.recovery_probes_done = 0;
                info.recovery_probes_passed = 0;
            }
            break;

        default:
            break;
    }

    return info.state;
}

void latency_calibrator_notify_fallback_served(LatencyCalibrator *cal,
                                               QueryRegime regime,
                                               uint32_t selectivity_bp,
                                               HybridEngineId engine) {
    if (!cal) return;
    StratumKey key = make_key(regime, selectivity_bp, engine);
    auto it = cal->strata.find(key);
    if (it == cal->strata.end()) return;

    StratumCalibrator &info = it->second.info;
    info.total_fallback_queries++;

    if (info.state == STRATUM_STATE_FALLBACK_EXACT) {
        if (info.fallback_remaining > 0) {
            info.fallback_remaining--;
        }
        if (info.fallback_remaining == 0) {
            info.state = STRATUM_STATE_RECOVERY_PROBE;
            info.recovery_probes_done = 0;
            info.recovery_probes_passed = 0;
        }
    }
}

bool latency_calibrator_is_calibrated(const LatencyCalibrator *cal,
                                      QueryRegime regime,
                                      uint32_t selectivity_bp,
                                      HybridEngineId engine) {
    if (!cal) return false;
    StratumKey key = make_key(regime, selectivity_bp, engine);
    auto it = cal->strata.find(key);
    if (it == cal->strata.end()) return false;
    return (it->second.info.stats.sample_count >= cal->config.min_samples);
}

bool latency_calibrator_should_fallback(const LatencyCalibrator *cal,
                                        QueryRegime regime,
                                        uint32_t selectivity_bp,
                                        HybridEngineId engine) {
    if (!cal) return true;
    /* Consultas fuera de distribución o desconocidas van forzosamente a ExactScan */
    if (regime == QUERY_UNKNOWN) return true;

    StratumKey key = make_key(regime, selectivity_bp, engine);
    auto it = cal->strata.find(key);
    if (it == cal->strata.end()) {
        /* Estrato nunca antes visto: bajo modo conservador, exige fallback */
        return cal->config.conservative_unseen;
    }

    if (cal->config.conservative_unseen && it->second.info.stats.sample_count < cal->config.min_samples) {
        return true;
    }

    return (it->second.info.state == STRATUM_STATE_FALLBACK_EXACT);
}

bool latency_calibrator_get_stratum(const LatencyCalibrator *cal,
                                    QueryRegime regime,
                                    uint32_t selectivity_bp,
                                    HybridEngineId engine,
                                    StratumCalibrator *out_stratum) {
    if (!cal || !out_stratum) return false;
    StratumKey key = make_key(regime, selectivity_bp, engine);
    auto it = cal->strata.find(key);
    if (it == cal->strata.end()) return false;
    *out_stratum = it->second.info;
    return true;
}

/* ========================================================================= */
/* --- Módulo B: Auditoría de Calidad Dual (Recall@10 y Recall@100) ---      */
/* ========================================================================= */

struct QualityStratumData {
    std::deque<double> recall10_window;
    std::deque<double> recall100_window;
    uint64_t audited_count;
    uint32_t fallback_triggers;
};

struct QualityAuditHarness {
    QualityAuditConfig config;
    std::map<QualityStratumKey, QualityStratumData> strata;
    QualityAuditMetrics metrics;
};

void quality_audit_default_config(QualityAuditConfig *cfg) {
    if (!cfg) return;
    cfg->target_recall10 = 0.95;
    cfg->target_recall100 = 0.90;
    cfg->audit_rate = 0.05;         /* 5% de consultas en shadow audit */
    cfg->audit_window_size = 50;
    cfg->min_audit_samples = 10;
    cfg->confidence_z = 1.645;      /* 95% unilateral */
}

QualityAuditHarness* quality_audit_create(const QualityAuditConfig *config) {
    QualityAuditHarness *q = new QualityAuditHarness();
    if (config) {
        q->config = *config;
    } else {
        quality_audit_default_config(&q->config);
    }
    std::memset(&q->metrics, 0, sizeof(q->metrics));
    return q;
}

void quality_audit_free(QualityAuditHarness *audit) {
    if (audit) delete audit;
}

void quality_audit_reset(QualityAuditHarness *audit) {
    if (!audit) return;
    audit->strata.clear();
    std::memset(&audit->metrics, 0, sizeof(audit->metrics));
}

bool quality_audit_should_sample(QualityAuditHarness *audit, uint64_t query_index) {
    if (!audit || audit->config.audit_rate <= 0.0) return false;
    audit->metrics.total_queries++;

    uint64_t stride = (uint64_t)std::floor(1.0 / audit->config.audit_rate);
    if (stride == 0) stride = 1;
    return (query_index % stride) == 0;
}

void quality_audit_record(QualityAuditHarness *audit,
                          QueryRegime regime,
                          uint32_t selectivity_bp,
                          double observed_recall10,
                          double observed_recall100,
                          double shadow_overhead_us) {
    if (!audit) return;
    audit->metrics.audited_queries++;
    audit->metrics.total_shadow_overhead_us += shadow_overhead_us;

    QualityStratumKey key = make_quality_key(regime, selectivity_bp);
    QualityStratumData &data = audit->strata[key];
    data.audited_count++;

    data.recall10_window.push_back(observed_recall10);
    if (data.recall10_window.size() > audit->config.audit_window_size) {
        data.recall10_window.pop_front();
    }

    data.recall100_window.push_back(observed_recall100);
    if (data.recall100_window.size() > audit->config.audit_window_size) {
        data.recall100_window.pop_front();
    }
}

static double compute_lcb_from_window(const std::deque<double> &w, double confidence_z) {
    size_t n = w.size();
    if (n == 0) return 0.0;
    double sum = 0.0;
    for (double r : w) sum += r;
    double mean = sum / n;

    double var = 0.0;
    if (n > 1) {
        double sq = 0.0;
        for (double r : w) {
            double d = r - mean;
            sq += d * d;
        }
        var = sq / (n - 1);
    }
    double se = std::sqrt(var / n);
    double lcb = mean - (confidence_z * se);
    if (lcb < 0.0) lcb = 0.0;
    if (lcb > 1.0) lcb = 1.0;
    return lcb;
}

QualityStratumState quality_audit_get_state(const QualityAuditHarness *audit,
                                            QueryRegime regime,
                                            uint32_t selectivity_bp,
                                            double *out_lcb10,
                                            double *out_lcb100) {
    if (!audit) {
        if (out_lcb10) *out_lcb10 = 0.0;
        if (out_lcb100) *out_lcb100 = 0.0;
        return QUALITY_STATE_UNCALIBRATED;
    }

    QualityStratumKey key = make_quality_key(regime, selectivity_bp);
    auto it = audit->strata.find(key);
    if (it == audit->strata.end()) {
        /* Estrato no visto: UNCALIBRATED (0.0) */
        if (out_lcb10) *out_lcb10 = 0.0;
        if (out_lcb100) *out_lcb100 = 0.0;
        return QUALITY_STATE_UNCALIBRATED;
    }

    const auto &w10 = it->second.recall10_window;
    const auto &w100 = it->second.recall100_window;
    size_t n10 = w10.size();
    size_t n100 = w100.size();

    if (n10 < audit->config.min_audit_samples || n100 < audit->config.min_audit_samples) {
        /* Muestras insuficientes en alguna de las metricas: UNCALIBRATED */
        if (out_lcb10) *out_lcb10 = 0.0;
        if (out_lcb100) *out_lcb100 = 0.0;
        return QUALITY_STATE_UNCALIBRATED;
    }

    double lcb10 = compute_lcb_from_window(w10, audit->config.confidence_z);
    double lcb100 = compute_lcb_from_window(w100, audit->config.confidence_z);

    if (out_lcb10) *out_lcb10 = lcb10;
    if (out_lcb100) *out_lcb100 = lcb100;

    /* Certificacion dual: AMBAS metricas deben superar su objetivo */
    if (lcb10 < audit->config.target_recall10 || lcb100 < audit->config.target_recall100) {
        return QUALITY_STATE_DEGRADED;
    }
    return QUALITY_STATE_HEALTHY;
}

bool quality_audit_should_fallback(const QualityAuditHarness *audit,
                                   QueryRegime regime,
                                   uint32_t selectivity_bp,
                                   double *out_lcb10,
                                   double *out_lcb100) {
    QualityStratumState state = quality_audit_get_state(audit, regime, selectivity_bp, out_lcb10, out_lcb100);
    if (state == QUALITY_STATE_UNCALIBRATED || state == QUALITY_STATE_DEGRADED) {
        if (audit) {
            const_cast<QualityAuditHarness*>(audit)->metrics.quality_fallback_triggers++;
        }
        return true;
    }
    return false;
}

bool quality_audit_is_calibrated(const QualityAuditHarness *audit,
                                 QueryRegime regime,
                                 uint32_t selectivity_bp) {
    double l10 = 0.0, l100 = 0.0;
    return (quality_audit_get_state(audit, regime, selectivity_bp, &l10, &l100) != QUALITY_STATE_UNCALIBRATED);
}

bool quality_audit_is_degraded(const QualityAuditHarness *audit,
                               QueryRegime regime,
                               uint32_t selectivity_bp,
                               double *out_lcb10,
                               double *out_lcb100) {
    return (quality_audit_get_state(audit, regime, selectivity_bp, out_lcb10, out_lcb100) == QUALITY_STATE_DEGRADED);
}

void quality_audit_get_metrics(const QualityAuditHarness *audit, QualityAuditMetrics *out_metrics) {
    if (!audit || !out_metrics) return;
    *out_metrics = audit->metrics;
}
