#include "../include/online_calibrator.h"
#include "../include/regret_metrics.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <windows.h>

/*
 * Test de Integración de Lazo Cerrado:
 * Demuestra que la degradación detectada en tiempo real modifica efectivamente
 * el camino de despacho de las consultas subsiguientes, y que la recuperación
 * mediante probes controladas restablece el enrutamiento normal.
 */

struct MockEngineSystem {
    bool inject_latency_spike;
    bool inject_quality_drop;
    double hnsw_latency_us;
    double hnsw_recall;
    double exact_latency_us;
};

int main() {
    SetConsoleOutputCP(65001);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetProcessAffinityMask(GetCurrentProcess(), 1ULL);

    std::cout << "========================================================================\n";
    std::cout << "Hito 9: Test de Integracion de Lazo Cerrado (Closed-Loop Safety)\n";
    std::cout << "========================================================================\n";

    LatencyCalibratorConfig lcfg;
    latency_calibrator_default_config(&lcfg);
    lcfg.min_samples = 5;
    lcfg.suspect_patience = 2;
    lcfg.fallback_cooldown = 10;
    lcfg.recovery_probe_count = 3;
    lcfg.conservative_unseen = false; /* Inicialmente pre-calibrado para el test */

    QualityAuditConfig qcfg;
    quality_audit_default_config(&qcfg);
    qcfg.min_audit_samples = 5;
    qcfg.target_recall = 0.95;
    qcfg.audit_rate = 1.0; /* 100% shadow audit para seguimiento exacto */

    LatencyCalibrator *lat_cal = latency_calibrator_create(&lcfg);
    QualityAuditHarness *qual_audit = quality_audit_create(&qcfg);

    MockEngineSystem sys{false, false, 80.0, 0.98, 150.0};
    QueryRegime regime = QUERY_LOCAL;
    uint32_t sel_bp = 1500; /* 15.0% */

    uint32_t exact_dispatched = 0;
    uint32_t hnsw_dispatched = 0;
    uint32_t fallback_events = 0;

    std::cout << "Fase 1: 15 consultas nominales (HNSW rapido a 80 us, Recall 98%)...\n";
    for (uint32_t q = 0; q < 15; q++) {
        /* Despacho en lazo cerrado */
        bool lat_fb = latency_calibrator_should_fallback(lat_cal, regime, sel_bp, ENGINE_HNSWLIB);
        double lcb = 0.0;
        bool qual_fb = (q >= 5) && quality_audit_should_fallback(qual_audit, regime, sel_bp, &lcb);

        HybridEngineId chosen = (lat_fb || qual_fb) ? ENGINE_EXACT_SCAN : ENGINE_HNSWLIB;
        if (chosen == ENGINE_EXACT_SCAN) exact_dispatched++;
        else hnsw_dispatched++;

        /* Ejecución y retroalimentación */
        DriftReason reason = DRIFT_NONE;
        latency_calibrator_record(lat_cal, regime, sel_bp, ENGINE_HNSWLIB, sys.hnsw_latency_us, 80.0, &reason);
        quality_audit_record(qual_audit, regime, sel_bp, sys.hnsw_recall, 10.0);
    }

    assert(hnsw_dispatched >= 10);
    std::cout << "  -> Fase 1 exitosa: HNSW despachado en regimen nominal (" << hnsw_dispatched << " consultas).\n";

    std::cout << "\nFase 2: Inyeccion de anomalia persistente (Latencia salta a 350 us > 2*sigma)...\n";
    sys.inject_latency_spike = true;
    sys.hnsw_latency_us = 350.0;

    uint32_t pre_fb_exact = exact_dispatched;
    for (uint32_t q = 15; q < 30; q++) {
        bool lat_fb = latency_calibrator_should_fallback(lat_cal, regime, sel_bp, ENGINE_HNSWLIB);
        HybridEngineId chosen = lat_fb ? ENGINE_EXACT_SCAN : ENGINE_HNSWLIB;

        if (chosen == ENGINE_EXACT_SCAN) {
            exact_dispatched++;
            latency_calibrator_notify_fallback_served(lat_cal, regime, sel_bp, ENGINE_HNSWLIB);
        } else {
            hnsw_dispatched++;
            DriftReason reason = DRIFT_NONE;
            StratumState s = latency_calibrator_record(lat_cal, regime, sel_bp, ENGINE_HNSWLIB, sys.hnsw_latency_us, 80.0, &reason);
            if (s == STRATUM_STATE_FALLBACK_EXACT) {
                fallback_events++;
            }
        }
    }

    /* Verificar que el fallback se activo y desvio consultas posteriores a ExactScan */
    assert(fallback_events >= 1);
    assert(exact_dispatched > pre_fb_exact);
    std::cout << "  -> Fase 2 exitosa: La anomalia forzo el estado FALLBACK_EXACT. "
              << (exact_dispatched - pre_fb_exact) << " consultas fueron derivadas a ExactScan para proteger SLA.\n";

    std::cout << "\nFase 3: Normalizacion del motor y recuperacion en RECOVERY_PROBE...\n";
    sys.inject_latency_spike = false;
    sys.hnsw_latency_us = 80.0;

    /* Servir consultas en fallback hasta agotar cooldown y entrar a RECOVERY_PROBE */
    StratumCalibrator sinfo{};
    latency_calibrator_get_stratum(lat_cal, regime, sel_bp, ENGINE_HNSWLIB, &sinfo);
    while (sinfo.state == STRATUM_STATE_FALLBACK_EXACT) {
        latency_calibrator_notify_fallback_served(lat_cal, regime, sel_bp, ENGINE_HNSWLIB);
        latency_calibrator_get_stratum(lat_cal, regime, sel_bp, ENGINE_HNSWLIB, &sinfo);
    }
    assert(sinfo.state == STRATUM_STATE_RECOVERY_PROBE);
    std::cout << "  -> Cooldown cumplido: Estado transiciono a RECOVERY_PROBE.\n";

    /* Probes de recuperacion */
    DriftReason rec_reason = DRIFT_NONE;
    for (uint32_t p = 0; p < lcfg.recovery_probe_count; p++) {
        latency_calibrator_record(lat_cal, regime, sel_bp, ENGINE_HNSWLIB, sys.hnsw_latency_us, 80.0, &rec_reason);
    }
    assert(rec_reason == RECOVERY_SUCCESS);
    latency_calibrator_get_stratum(lat_cal, regime, sel_bp, ENGINE_HNSWLIB, &sinfo);
    assert(sinfo.state == STRATUM_STATE_CALIBRATED);
    assert(!latency_calibrator_should_fallback(lat_cal, regime, sel_bp, ENGINE_HNSWLIB));
    std::cout << "  -> Fase 3 exitosa: Probes nominales validaron la recuperacion a CALIBRATED.\n";

    latency_calibrator_free(lat_cal);
    quality_audit_free(qual_audit);

    std::cout << "\n========================================================================\n";
    std::cout << "Test de Integracion de Lazo Cerrado PASO EXITOSAMENTE.\n";
    std::cout << "========================================================================\n";
    return 0;
}
