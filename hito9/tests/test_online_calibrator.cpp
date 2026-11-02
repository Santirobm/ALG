#include "../include/online_calibrator.h"
#include "../include/regret_metrics.h"
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
    cfg.conservative_unseen = false; /* Solo para test 1 de ventana */
    LatencyCalibrator *cal = latency_calibrator_create(&cfg);

    DriftReason reason = DRIFT_NONE;
    /* Enviar 8 muestras nominales (residual 0) */
    for (int i = 0; i < 8; ++i) {
        StratumState s = latency_calibrator_record(cal, QUERY_LOCAL, 1500, ENGINE_HNSWLIB, 100.0, 100.0, &reason);
        assert(s == STRATUM_STATE_CALIBRATED);
        assert(reason == DRIFT_NONE);
    }

    /* Muestra anómala antes de alcanzar min_samples: NO debe disparar SUSPECT prematuro */
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

static void test_conservative_quality_uncalibrated_and_degraded() {
    std::cout << "[Test 5] Calidad conservadora dual (Recall@10 y Recall@100 independizados)...\n";
    QualityAuditConfig qcfg;
    quality_audit_default_config(&qcfg);
    qcfg.target_recall10 = 0.95;
    qcfg.target_recall100 = 0.90;
    qcfg.min_audit_samples = 10;
    QualityAuditHarness *qaudit = quality_audit_create(&qcfg);

    double lcb10 = -1.0, lcb100 = -1.0;
    /* Caso A: Estrato nunca visto */
    QualityStratumState s0 = quality_audit_get_state(qaudit, QUERY_LOCAL, 1500, &lcb10, &lcb100);
    assert(s0 == QUALITY_STATE_UNCALIBRATED);
    assert(lcb10 == 0.0 && lcb100 == 0.0); /* NUNCA 1.0 artificial */
    assert(quality_audit_should_fallback(qaudit, QUERY_LOCAL, 1500, &lcb10, &lcb100) == true);
    assert(quality_audit_is_calibrated(qaudit, QUERY_LOCAL, 1500) == false);

    /* Caso B: Muestras insuficientes (5 de 10) */
    for (int i = 0; i < 5; ++i) {
        quality_audit_record(qaudit, QUERY_LOCAL, 1500, 1.0, 1.0, 10.0);
    }
    QualityStratumState s1 = quality_audit_get_state(qaudit, QUERY_LOCAL, 1500, &lcb10, &lcb100);
    assert(s1 == QUALITY_STATE_UNCALIBRATED);
    assert(lcb10 == 0.0 && lcb100 == 0.0);
    assert(quality_audit_should_fallback(qaudit, QUERY_LOCAL, 1500, &lcb10, &lcb100) == true);

    /* Caso C: Se completan 10 muestras de calidad excelente (R@10=1.0, R@100=1.0) -> HEALTHY */
    for (int i = 0; i < 5; ++i) {
        quality_audit_record(qaudit, QUERY_LOCAL, 1500, 1.0, 1.0, 10.0);
    }
    QualityStratumState s2 = quality_audit_get_state(qaudit, QUERY_LOCAL, 1500, &lcb10, &lcb100);
    assert(s2 == QUALITY_STATE_HEALTHY);
    assert(lcb10 >= 0.95 && lcb100 >= 0.90);
    assert(quality_audit_should_fallback(qaudit, QUERY_LOCAL, 1500, &lcb10, &lcb100) == false);
    assert(quality_audit_is_calibrated(qaudit, QUERY_LOCAL, 1500) == true);

    /* Caso D: R@10 cumple (0.98 >= 0.95), pero R@100 degrada (0.80 < 0.90) -> DEGRADED */
    for (int i = 0; i < 15; ++i) {
        quality_audit_record(qaudit, QUERY_LOCAL, 1500, 0.98, 0.80, 10.0);
    }
    QualityStratumState s3 = quality_audit_get_state(qaudit, QUERY_LOCAL, 1500, &lcb10, &lcb100);
    assert(s3 == QUALITY_STATE_DEGRADED);
    assert(lcb10 >= 0.95);
    assert(lcb100 < 0.90);
    assert(quality_audit_should_fallback(qaudit, QUERY_LOCAL, 1500, &lcb10, &lcb100) == true);

    quality_audit_free(qaudit);
    std::cout << "  -> OK: Auditoria dual certifica R@10 y R@100 independientemente; fallback se activa si cualquiera falla.\n";
}

static void test_conservative_unseen_latency_stratum() {
    std::cout << "[Test 6] Latencia conservadora: estratos desconocidos o sin datos...\n";
    LatencyCalibratorConfig cfg;
    latency_calibrator_default_config(&cfg);
    cfg.conservative_unseen = true;
    LatencyCalibrator *cal = latency_calibrator_create(&cfg);

    /* Estrato nunca visto: debe forzar fallback seguro a ExactScan */
    assert(latency_calibrator_should_fallback(cal, QUERY_LOCAL, 1250, ENGINE_HNSWLIB) == true);
    assert(latency_calibrator_is_calibrated(cal, QUERY_LOCAL, 1250, ENGINE_HNSWLIB) == false);

    /* Régimen UNKNOWN: siempre deriva forzosamente a ExactScan */
    assert(latency_calibrator_should_fallback(cal, QUERY_UNKNOWN, 1500, ENGINE_HNSWLIB) == true);

    latency_calibrator_free(cal);
    std::cout << "  -> OK: Modo conservador enruta estratos no calibrados a ExactScan hasta acumular perfil.\n";
}

static void test_policy_vs_diagnostic_regret() {
    std::cout << "[Test 7] Evaluación de Regret: Separación entre Política y Diagnóstico...\n";
    double exact_lat = 100.0;
    double hnsw_lat = 50.0;

    /* Escenario 1: POLICY_EXACT con ExactScan ejecutado */
    /* Ruido de medición: actual_lat = 104.0 us (levemente superior a exact_lat de referencia) */
    RegretEvaluation r1 = evaluate_regret(POLICY_EXACT, 104.0, exact_lat, hnsw_lat, false, ENGINE_EXACT_SCAN);
    /* Por definición: ExactScan frente a POLICY_EXACT tiene regret CERO */
    assert(r1.policy_signed_regret_us == 0.0);
    assert(r1.policy_positive_regret_us == 0.0);
    /* El oráculo admisible es exactamente ExactScan */
    assert(r1.admissible_oracle_latency_us == exact_lat);
    /* Diagnóstico no restringido registra la diferencia contra el motor más rápido (HNSW a 50 us) */
    assert(r1.diagnostic_signed_regret_us == 54.0);
    assert(r1.diagnostic_positive_regret_us == 54.0);

    /* Escenario 2: POLICY_BALANCED con HNSW no admisible (calidad no certificada) */
    RegretEvaluation r2 = evaluate_regret(POLICY_BALANCED, exact_lat, exact_lat, hnsw_lat, false, ENGINE_EXACT_SCAN);
    assert(r2.admissible_oracle_latency_us == exact_lat);
    assert(r2.policy_positive_regret_us == 0.0);

    /* Escenario 3: POLICY_BALANCED con HNSW admisible (calidad certificada LCB >= 0.95) */
    RegretEvaluation r3 = evaluate_regret(POLICY_BALANCED, exact_lat, exact_lat, hnsw_lat, true, ENGINE_EXACT_SCAN);
    assert(r3.admissible_oracle_latency_us == hnsw_lat);
    assert(r3.policy_signed_regret_us == exact_lat - hnsw_lat);
    assert(r3.policy_positive_regret_us == 50.0);

    /* Escenario 4: POLICY_APPROXIMATE (mínimo de latencia puro) */
    RegretEvaluation r4 = evaluate_regret(POLICY_APPROXIMATE, hnsw_lat, exact_lat, hnsw_lat, true, ENGINE_HNSWLIB);
    assert(r4.admissible_oracle_latency_us == hnsw_lat);
    assert(r4.policy_signed_regret_us == 0.0);
    assert(r4.policy_positive_regret_us == 0.0);

    std::cout << "  -> OK: ExactScan en POLICY_EXACT tiene regret cero por definición; oráculos admisibles y diagnósticos separados.\n";
}

int main() {
    SetConsoleOutputCP(65001);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetProcessAffinityMask(GetCurrentProcess(), 1ULL);

    std::cout << "========================================================================\n";
    std::cout << "Hito 9: Auditoría Metodológica — Test de Calibración, Calidad y Regret\n";
    std::cout << "========================================================================\n";

    test_insufficient_samples_and_reset();
    test_isolated_outlier_no_false_alarm();
    test_sudden_latency_spike_to_fallback();
    test_gradual_drift_and_recovery_cycle();
    test_conservative_quality_uncalibrated_and_degraded();
    test_conservative_unseen_latency_stratum();
    test_policy_vs_diagnostic_regret();

    std::cout << "========================================================================\n";
    std::cout << "Todos los tests de Auditoría Metodológica PASARON con éxito (7/7).\n";
    std::cout << "========================================================================\n";
    return 0;
}
