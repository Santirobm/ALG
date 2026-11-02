#include "../include/online_calibrator.h"
#include "../include/regret_metrics.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <windows.h>

/*
 * Test de Integración de Lazo Cerrado (Closed-Loop Safety):
 * Valida determinísticamente:
 * 1. Fallback y recuperación ante derivas de latencia.
 * 2. Fallback y recuperación ante degradación real de calidad (Recall@10 y Recall@100)
 *    con latencia plana e invariable.
 * 3. Prohibición de rehabilitar HNSW antes de acumular evidencia suficiente en shadow.
 */

struct MockEngineSystem {
    double hnsw_latency_us;
    double hnsw_recall10;
    double hnsw_recall100;
    double exact_latency_us;
};

static void test_latency_drift_closed_loop() {
    std::cout << "\n--- Subtest A: Lazo Cerrado ante Deriva de Latencia ---\n";
    LatencyCalibratorConfig lcfg;
    latency_calibrator_default_config(&lcfg);
    lcfg.min_samples = 5;
    lcfg.suspect_patience = 2;
    lcfg.fallback_cooldown = 10;
    lcfg.recovery_probe_count = 3;
    lcfg.conservative_unseen = false;

    QualityAuditConfig qcfg;
    quality_audit_default_config(&qcfg);
    qcfg.min_audit_samples = 5;
    qcfg.target_recall10 = 0.95;
    qcfg.target_recall100 = 0.90;
    qcfg.audit_rate = 1.0;

    LatencyCalibrator *lat_cal = latency_calibrator_create(&lcfg);
    QualityAuditHarness *qual_audit = quality_audit_create(&qcfg);

    MockEngineSystem sys{80.0, 0.98, 0.95, 150.0};
    QueryRegime regime = QUERY_LOCAL;
    uint32_t sel_bp = 1500;

    uint32_t exact_dispatched = 0, hnsw_dispatched = 0;

    /* Fase 1: Calibración nominal */
    for (uint32_t q = 0; q < 15; q++) {
        bool lat_fb = latency_calibrator_should_fallback(lat_cal, regime, sel_bp, ENGINE_HNSWLIB);
        double l10 = 0.0, l100 = 0.0;
        bool qual_fb = (q >= 5) && quality_audit_should_fallback(qual_audit, regime, sel_bp, &l10, &l100);

        HybridEngineId chosen = (lat_fb || qual_fb) ? ENGINE_EXACT_SCAN : ENGINE_HNSWLIB;
        if (chosen == ENGINE_EXACT_SCAN) exact_dispatched++;
        else hnsw_dispatched++;

        DriftReason reason = DRIFT_NONE;
        latency_calibrator_record(lat_cal, regime, sel_bp, ENGINE_HNSWLIB, sys.hnsw_latency_us, 80.0, &reason);
        quality_audit_record(qual_audit, regime, sel_bp, sys.hnsw_recall10, sys.hnsw_recall100, 10.0);
    }
    assert(hnsw_dispatched >= 10);
    std::cout << "  [1.1] Regimen nominal: HNSW despachado normalmente (" << hnsw_dispatched << " consultas).\n";

    /* Fase 2: Salto abrupto de latencia a 350 us */
    sys.hnsw_latency_us = 350.0;
    uint32_t pre_exact = exact_dispatched;
    for (uint32_t q = 15; q < 30; q++) {
        bool lat_fb = latency_calibrator_should_fallback(lat_cal, regime, sel_bp, ENGINE_HNSWLIB);
        HybridEngineId chosen = lat_fb ? ENGINE_EXACT_SCAN : ENGINE_HNSWLIB;

        if (chosen == ENGINE_EXACT_SCAN) {
            exact_dispatched++;
            latency_calibrator_notify_fallback_served(lat_cal, regime, sel_bp, ENGINE_HNSWLIB);
        } else {
            hnsw_dispatched++;
            DriftReason reason = DRIFT_NONE;
            latency_calibrator_record(lat_cal, regime, sel_bp, ENGINE_HNSWLIB, sys.hnsw_latency_us, 80.0, &reason);
        }
    }
    assert(exact_dispatched > pre_exact);
    std::cout << "  [1.2] Anomalía de latencia: Fallback a ExactScan activado (" << (exact_dispatched - pre_exact) << " consultas protegidas).\n";

    /* Fase 3: Recuperación en probes */
    sys.hnsw_latency_us = 80.0;
    StratumCalibrator sinfo{};
    latency_calibrator_get_stratum(lat_cal, regime, sel_bp, ENGINE_HNSWLIB, &sinfo);
    while (sinfo.state == STRATUM_STATE_FALLBACK_EXACT) {
        latency_calibrator_notify_fallback_served(lat_cal, regime, sel_bp, ENGINE_HNSWLIB);
        latency_calibrator_get_stratum(lat_cal, regime, sel_bp, ENGINE_HNSWLIB, &sinfo);
    }
    assert(sinfo.state == STRATUM_STATE_RECOVERY_PROBE);

    DriftReason rec_reason = DRIFT_NONE;
    for (uint32_t p = 0; p < lcfg.recovery_probe_count; p++) {
        latency_calibrator_record(lat_cal, regime, sel_bp, ENGINE_HNSWLIB, sys.hnsw_latency_us, 80.0, &rec_reason);
    }
    assert(rec_reason == RECOVERY_SUCCESS);
    assert(!latency_calibrator_should_fallback(lat_cal, regime, sel_bp, ENGINE_HNSWLIB));
    std::cout << "  [1.3] Recuperacion: Probes validadas y despacho a HNSW restaurado.\n";

    latency_calibrator_free(lat_cal);
    quality_audit_free(qual_audit);
}

static void test_quality_drift_closed_loop() {
    std::cout << "\n--- Subtest B: Lazo Cerrado ante Degradacion Real de Calidad (Recall Drop) ---\n";
    LatencyCalibratorConfig lcfg;
    latency_calibrator_default_config(&lcfg);
    lcfg.min_samples = 5;
    lcfg.conservative_unseen = false;

    QualityAuditConfig qcfg;
    quality_audit_default_config(&qcfg);
    qcfg.min_audit_samples = 5;
    qcfg.target_recall10 = 0.95;
    qcfg.target_recall100 = 0.90;
    qcfg.audit_rate = 1.0;

    LatencyCalibrator *lat_cal = latency_calibrator_create(&lcfg);
    QualityAuditHarness *qual_audit = quality_audit_create(&qcfg);

    MockEngineSystem sys{80.0, 0.98, 0.95, 150.0};
    QueryRegime regime = QUERY_LOCAL;
    uint32_t sel_bp = 1500;

    uint32_t exact_dispatched = 0, hnsw_dispatched = 0;

    /* Fase 1: Calidad nominal certificada */
    for (uint32_t q = 0; q < 15; q++) {
        double l10 = 0.0, l100 = 0.0;
        bool qual_fb = (q >= 5) && quality_audit_should_fallback(qual_audit, regime, sel_bp, &l10, &l100);
        HybridEngineId chosen = qual_fb ? ENGINE_EXACT_SCAN : ENGINE_HNSWLIB;

        if (chosen == ENGINE_EXACT_SCAN) exact_dispatched++;
        else hnsw_dispatched++;

        DriftReason r = DRIFT_NONE;
        latency_calibrator_record(lat_cal, regime, sel_bp, ENGINE_HNSWLIB, sys.hnsw_latency_us, 80.0, &r);
        quality_audit_record(qual_audit, regime, sel_bp, sys.hnsw_recall10, sys.hnsw_recall100, 5.0);
    }
    assert(hnsw_dispatched >= 10);
    std::cout << "  [2.1] Fase Nominal: R@10=98%, R@100=95% -> HNSW despachado en produccion.\n";

    /* Fase 2: Latencia sigue PERFECTAMENTE RAPIDA (80 us), pero Recall cae (R@10=0.82 < 0.95, R@100=0.74 < 0.90) */
    sys.hnsw_recall10 = 0.82;
    sys.hnsw_recall100 = 0.74;

    uint32_t exact_in_quality_fallback = 0;
    for (uint32_t q = 15; q < 35; q++) {
        double l10 = 0.0, l100 = 0.0;
        bool qual_fb = quality_audit_should_fallback(qual_audit, regime, sel_bp, &l10, &l100);
        bool lat_fb  = latency_calibrator_should_fallback(lat_cal, regime, sel_bp, ENGINE_HNSWLIB);

        /* Comprobación clave: Latencia NO ve anomalía, pero Calidad SÍ */
        assert(lat_fb == false);

        HybridEngineId chosen = (qual_fb || lat_fb) ? ENGINE_EXACT_SCAN : ENGINE_HNSWLIB;
        if (chosen == ENGINE_EXACT_SCAN) {
            exact_in_quality_fallback++;
            /* Consulta del usuario servida por ExactScan con recall 100% garantizado */
            /* En background / shadow se audita HNSW */
            quality_audit_record(qual_audit, regime, sel_bp, sys.hnsw_recall10, sys.hnsw_recall100, 5.0);
        } else {
            hnsw_dispatched++;
            quality_audit_record(qual_audit, regime, sel_bp, sys.hnsw_recall10, sys.hnsw_recall100, 5.0);
        }
    }

    assert(exact_in_quality_fallback >= 15);
    std::cout << "  [2.2] Degradacion de Recall: Detectada con exito (LCB cayo bajo umbral). "
              << exact_in_quality_fallback << " consultas derivadas a ExactScan mientras latencia permanecio plana.\n";

    /* Fase 3: Recuperacion condicionada a evidencia suficiente en shadow */
    sys.hnsw_recall10 = 0.99;
    sys.hnsw_recall100 = 0.98;

    /* Enviar 2 muestras shadow recuperadas (insuficiente para elevar LCB con ventana previa de 0.82) */
    quality_audit_record(qual_audit, regime, sel_bp, sys.hnsw_recall10, sys.hnsw_recall100, 5.0);
    quality_audit_record(qual_audit, regime, sel_bp, sys.hnsw_recall10, sys.hnsw_recall100, 5.0);

    double l10_chk = 0.0, l100_chk = 0.0;
    /* Comprobar que NO habilita HNSW prematuramente */
    bool still_fb = quality_audit_should_fallback(qual_audit, regime, sel_bp, &l10_chk, &l100_chk);
    assert(still_fb == true);
    std::cout << "  [2.3] Seguridad: 2 muestras de recuperacion NO alcanzan para habilitar HNSW (LCB aun bajo control de ventana previa).\n";

    /* Acumular muestras shadow suficientes hasta que la ventana desplace las mediciones degradadas */
    for (int s = 0; s < 45; s++) {
        quality_audit_record(qual_audit, regime, sel_bp, sys.hnsw_recall10, sys.hnsw_recall100, 5.0);
    }

    bool recovered = !quality_audit_should_fallback(qual_audit, regime, sel_bp, &l10_chk, &l100_chk);
    assert(recovered == true);
    assert(l10_chk >= 0.95 && l100_chk >= 0.90);
    std::cout << "  [2.4] Recuperacion Certificada: Tras evidencia estadistica completa, LCB10="
              << std::fixed << std::setprecision(3) << l10_chk << ", LCB100=" << l100_chk << " -> HNSW rehabilitado.\n";

    latency_calibrator_free(lat_cal);
    quality_audit_free(qual_audit);
}

int main() {
    SetConsoleOutputCP(65001);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetProcessAffinityMask(GetCurrentProcess(), 1ULL);

    std::cout << "========================================================================\n";
    std::cout << "Hito 9: Test de Integracion de Lazo Cerrado (Closed-Loop Safety)\n";
    std::cout << "========================================================================\n";

    test_latency_drift_closed_loop();
    test_quality_drift_closed_loop();

    std::cout << "\n========================================================================\n";
    std::cout << "Todos los Tests de Integracion de Lazo Cerrado PASARON EXITOSAMENTE.\n";
    std::cout << "========================================================================\n";
    return 0;
}
