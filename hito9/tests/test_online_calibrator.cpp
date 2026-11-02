#include "../include/online_calibrator.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <iomanip>
#include <windows.h>

static void test_insufficient_samples_and_reset() {
    std::cout << "[Test 1] Ventana insuficiente y reinicio de calibrador...\n";
    LatencyCalibratorConfig cfg;
    latency_calibrator_default_config(&cfg);
    cfg.min_samples = 10;
    LatencyCalibrator *cal = latency_calibrator_create(&cfg);

    DriftReason reason = DRIFT_NONE;
    /* Enviar 8 muestras nominales (residual 0) */
    for (int i = 0; i < 8; ++i) {
        StratumState s = latency_calibrator_record(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, 100.0, 100.0, &reason);
        assert(s == STRATUM_STATE_CALIBRATED);
        assert(reason == DRIFT_NONE);
    }

    /* Muestra anómala antes de alcanzar min_samples: NO debe disparar SUSPECT */
    StratumState s = latency_calibrator_record(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, 500.0, 100.0, &reason);
    assert(s == STRATUM_STATE_CALIBRATED);
    assert(reason == DRIFT_NONE);

    /* Reset y verificación */
    latency_calibrator_reset(cal);
    StratumCalibrator info;
    bool found = latency_calibrator_get_stratum(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, &info);
    assert(!found);

    latency_calibrator_free(cal);
    std::cout << "  -> OK: No se disparan alarmas prematuras con ventana insuficiente; reset limpia estado.\n";
}

static void test_isolated_outlier_no_false_alarm() {
    std::cout << "[Test 2] Outlier aislado y absorción de falsas alarmas (SUSPECT -> CALIBRATED)...\n";
    LatencyCalibratorConfig cfg;
    latency_calibrator_default_config(&cfg);
    cfg.min_samples = 10;
    cfg.suspect_patience = 3;
    cfg.tripwire_sigma = 2.0;
    LatencyCalibrator *cal = latency_calibrator_create(&cfg);

    DriftReason reason = DRIFT_NONE;
    /* 15 muestras normales con leve ruido en torno a 100 us */
    for (int i = 0; i < 15; ++i) {
        double lat = 100.0 + (i % 3) * 2.0; /* 100, 102, 104 */
        latency_calibrator_record(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, lat, 102.0, &reason);
    }
    assert(!latency_calibrator_should_fallback(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB));

    /* 1 outlier aislado severo (latencia 180 us vs est 102 us) */
    StratumState s1 = latency_calibrator_record(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, 180.0, 102.0, &reason);
    assert(s1 == STRATUM_STATE_SUSPECT);
    assert(reason == DRIFT_LATENCY_SPIKE);
    /* En SUSPECT todavía no se activa FALLBACK_EXACT si suspect_patience > 1 */
    assert(!latency_calibrator_should_fallback(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB));

    /* Siguiente muestra es nominal (102 us) */
    StratumState s2 = latency_calibrator_record(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, 102.0, 102.0, &reason);
    assert(s2 == STRATUM_STATE_CALIBRATED);
    assert(reason == RECOVERY_SUCCESS);

    StratumCalibrator info;
    latency_calibrator_get_stratum(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, &info);
    assert(info.false_alarms == 1);
    assert(info.fallback_events == 0);

    latency_calibrator_free(cal);
    std::cout << "  -> OK: Outlier aislado transiciona a SUSPECT y retorna a CALIBRATED sin entrar a fallback.\n";
}

static void test_sudden_latency_spike_to_fallback() {
    std::cout << "[Test 3] Deriva abrupta de latencia y transición a FALLBACK_EXACT...\n";
    LatencyCalibratorConfig cfg;
    latency_calibrator_default_config(&cfg);
    cfg.min_samples = 10;
    cfg.suspect_patience = 3;
    cfg.fallback_cooldown = 10;
    LatencyCalibrator *cal = latency_calibrator_create(&cfg);

    DriftReason reason = DRIFT_NONE;
    /* Entrenar ventana con 20 muestras nominales */
    for (int i = 0; i < 20; ++i) {
        latency_calibrator_record(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, 100.0, 100.0, &reason);
    }

    /* Ráfaga persistente de alta latencia (300 us vs est 100 us) */
    StratumState s1 = latency_calibrator_record(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, 300.0, 100.0, &reason);
    assert(s1 == STRATUM_STATE_SUSPECT);
    assert(reason == DRIFT_LATENCY_SPIKE);

    StratumState s2 = latency_calibrator_record(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, 310.0, 100.0, &reason);
    assert(s2 == STRATUM_STATE_SUSPECT);

    StratumState s3 = latency_calibrator_record(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, 305.0, 100.0, &reason);
    assert(s3 == STRATUM_STATE_FALLBACK_EXACT);
    assert(reason == DRIFT_LATENCY_PERSISTENT);
    assert(latency_calibrator_should_fallback(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB));

    StratumCalibrator info;
    latency_calibrator_get_stratum(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, &info);
    assert(info.fallback_events == 1);
    assert(info.fallback_remaining == 10);

    latency_calibrator_free(cal);
    std::cout << "  -> OK: 3 anomalías consecutivas disparan exitosamente FALLBACK_EXACT determinista.\n";
}

static void test_gradual_drift_and_recovery_cycle() {
    std::cout << "[Test 4] Deriva gradual, cooldown e histéresis en RECOVERY_PROBE...\n";
    LatencyCalibratorConfig cfg;
    latency_calibrator_default_config(&cfg);
    cfg.min_samples = 15;
    cfg.suspect_patience = 2;
    cfg.fallback_cooldown = 5;
    cfg.recovery_probe_count = 3;
    cfg.tripwire_sigma = 2.0;
    LatencyCalibrator *cal = latency_calibrator_create(&cfg);

    DriftReason reason = DRIFT_NONE;
    /* 20 consultas calibradas */
    for (int i = 0; i < 20; ++i) {
        latency_calibrator_record(cal, QUERY_LOCAL, 1750, ENGINE_HNSWLIB, 150.0, 150.0, &reason);
    }

    /* Deriva gradual en ráfaga */
    latency_calibrator_record(cal, QUERY_LOCAL, 1750, ENGINE_HNSWLIB, 220.0, 150.0, &reason);
    assert(reason == DRIFT_LATENCY_SPIKE);
    latency_calibrator_record(cal, QUERY_LOCAL, 1750, ENGINE_HNSWLIB, 230.0, 150.0, &reason);
    assert(reason == DRIFT_LATENCY_PERSISTENT);
    assert(latency_calibrator_should_fallback(cal, QUERY_LOCAL, 1750, ENGINE_HNSWLIB));

    /* Se sirven 5 consultas en fallback ExactScan */
    for (int i = 0; i < 5; ++i) {
        latency_calibrator_notify_fallback_served(cal, QUERY_LOCAL, 1750, ENGINE_HNSWLIB);
    }

    /* Cooldown expiró -> debe entrar a RECOVERY_PROBE */
    StratumCalibrator info;
    latency_calibrator_get_stratum(cal, QUERY_LOCAL, 1750, ENGINE_HNSWLIB, &info);
    assert(info.state == STRATUM_STATE_RECOVERY_PROBE);

    /* Enviar 3 probes recuperadas (150 us nominal) */
    latency_calibrator_record(cal, QUERY_LOCAL, 1750, ENGINE_HNSWLIB, 150.0, 150.0, &reason);
    latency_calibrator_record(cal, QUERY_LOCAL, 1750, ENGINE_HNSWLIB, 151.0, 150.0, &reason);
    StratumState s_rec = latency_calibrator_record(cal, QUERY_LOCAL, 1750, ENGINE_HNSWLIB, 150.0, 150.0, &reason);

    assert(s_rec == STRATUM_STATE_CALIBRATED);
    assert(reason == RECOVERY_SUCCESS);
    assert(!latency_calibrator_should_fallback(cal, QUERY_LOCAL, 1750, ENGINE_HNSWLIB));

    latency_calibrator_free(cal);
    std::cout << "  -> OK: Ciclo completo FALLBACK_EXACT -> RECOVERY_PROBE -> CALIBRATED validado con éxito.\n";
}

static void test_quality_drift_with_flat_latency() {
    std::cout << "[Test 5] Deriva de Calidad (Recall) con latencia plana e invariable...\n";
    QualityAuditConfig qcfg;
    quality_audit_default_config(&qcfg);
    qcfg.target_recall = 0.95;
    qcfg.min_audit_samples = 10;
    qcfg.confidence_z = 1.645;
    qcfg.audit_rate = 1.0; /* 100% en test para control exacto */

    LatencyCalibratorConfig lcfg;
    latency_calibrator_default_config(&lcfg);
    LatencyCalibrator *lcal = latency_calibrator_create(&lcfg);
    QualityAuditHarness *qaudit = quality_audit_create(&qcfg);

    DriftReason reason = DRIFT_NONE;
    /* Latencia perfectamente constante a 100 us */
    /* Pero recall cae de 0.98 a 0.85 (muy por debajo de 0.95) */
    for (int i = 0; i < 20; ++i) {
        StratumState ls = latency_calibrator_record(lcal, QUERY_LOCAL, 2000, ENGINE_HNSWLIB, 100.0, 100.0, &reason);
        assert(ls == STRATUM_STATE_CALIBRATED); /* La latencia NO detecta el fallo */

        /* El recall cae */
        double recall = (i < 10) ? 0.98 : 0.82;
        quality_audit_record(qaudit, QUERY_LOCAL, 2000, recall, 15.0);
    }

    /* Comprobación de la tesis fundamental del Track 3:
       El monitor de latencia NO vio desvío, pero el arnés de auditoría de calidad sí */
    assert(!latency_calibrator_should_fallback(lcal, QUERY_LOCAL, 2000, ENGINE_HNSWLIB));

    double lcb = 0.0;
    bool quality_degraded = quality_audit_is_degraded(qaudit, QUERY_LOCAL, 2000, &lcb);
    assert(quality_degraded);
    assert(lcb < 0.95);

    QualityAuditMetrics m;
    quality_audit_get_metrics(qaudit, &m);
    assert(m.audited_queries == 20);
    assert(m.quality_fallback_triggers > 0);
    assert(m.total_shadow_overhead_us > 0.0);

    latency_calibrator_free(lcal);
    quality_audit_free(qaudit);
    std::cout << "  -> OK: Se demuestra que la degradación de calidad es detectada por QualityAudit (LCB="
              << std::fixed << std::setprecision(3) << lcb << " < 0.95) mientras el monitor de latencia permanece nominal.\n";
}

static void test_ood_safety_default() {
    std::cout << "[Test 6] Enrutamiento conservador para régimen desconocido (QUERY_UNKNOWN)...\n";
    LatencyCalibrator *cal = latency_calibrator_create(nullptr);
    /* Cualquier consulta QUERY_UNKNOWN debe forzar fallback seguro a ExactScan */
    assert(latency_calibrator_should_fallback(cal, QUERY_UNKNOWN, 1500, ENGINE_HNSWLIB) == true);
    assert(latency_calibrator_should_fallback(cal, QUERY_UNKNOWN, 2500, ENGINE_ALG_H2_AT) == true);
    latency_calibrator_free(cal);
    std::cout << "  -> OK: QUERY_UNKNOWN siempre deriva determinísticamente a ExactScan.\n";
}

int main() {
    SetConsoleOutputCP(65001);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetProcessAffinityMask(GetCurrentProcess(), 1ULL);

    std::cout << "========================================================================\n";
    std::cout << "Hito 9: Track 3 — Test de Calibración Online, Fallback y Calidad Shadow\n";
    std::cout << "========================================================================\n";

    test_insufficient_samples_and_reset();
    test_isolated_outlier_no_false_alarm();
    test_sudden_latency_spike_to_fallback();
    test_gradual_drift_and_recovery_cycle();
    test_quality_drift_with_flat_latency();
    test_ood_safety_default();

    std::cout << "========================================================================\n";
    std::cout << "Todos los tests de Track 3 (Calibrador Online & Auditoría de Calidad) PASARON.\n";
    std::cout << "========================================================================\n";
    return 0;
}
