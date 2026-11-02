#include "../include/geometric_probe.h"
#include "../../hito7/common/dataset_format.h"
#include <windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <string>
#include <iomanip>

struct DatasetData {
    DatasetHeader header;
    std::vector<float> vectors;
    std::vector<uint32_t> categories;
    std::vector<float> queries_R;
    std::vector<float> queries_D;
};

static bool load_dataset(const std::string &path, DatasetData &data) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;

    if (fread(&data.header, sizeof(data.header), 1, f) != 1) { fclose(f); return false; }
    uint32_t nv = data.header.num_vectors;
    uint32_t dim = data.header.dim;
    uint32_t nq = data.header.num_queries;

    data.vectors.resize((size_t)nv * dim);
    data.categories.resize(nv);
    data.queries_R.resize((size_t)nq * dim);
    data.queries_D.resize((size_t)nq * dim);

    fread(data.vectors.data(), sizeof(float), (size_t)nv * dim, f);
    fread(data.categories.data(), sizeof(uint32_t), nv, f);
    fread(data.queries_R.data(), sizeof(float), (size_t)nq * dim, f);
    fread(data.queries_D.data(), sizeof(float), (size_t)nq * dim, f);

    fclose(f);
    return true;
}

int main() {
    SetConsoleOutputCP(65001);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetProcessAffinityMask(GetCurrentProcess(), 1ULL);
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);

    std::cout << "========================================================================\n";
    std::cout << "Hito 9: Track 1 — Test de Sonda Geométrica y Matriz de Confusión\n";
    std::cout << "========================================================================\n";

    /* Cargar Semilla 1 de calibración para persistencia y microbenchmark */
    std::string calib_path = "hito7/data/dataset_seed1.bin";
    DatasetData seed1_data;
    if (!load_dataset(calib_path, seed1_data)) {
        std::cerr << "Error cargando " << calib_path << "\n";
        return 1;
    }

    std::cout << "1. Entrenando sonda base (K=64 prototipos, 10 iteraciones, seed=42)...\n";
    GeometricProbe probe_seed1{};
    int rc = geometric_probe_build(&probe_seed1, seed1_data.vectors.data(),
                                   seed1_data.header.num_vectors, seed1_data.header.dim,
                                   64, 10, 42ULL);
    assert(rc == 0);

    /* Test de persistencia */
    std::string probe_save_path = "hito9/config/trained_probe_k64.bin";
    rc = geometric_probe_save(&probe_seed1, probe_save_path.c_str());
    assert(rc == 0);

    GeometricProbe loaded_probe{};
    rc = geometric_probe_load(&loaded_probe, probe_save_path.c_str());
    assert(rc == 0);
    assert(loaded_probe.num_prototypes == 64);
    assert(loaded_probe.dim == 64);
    assert(memcmp(probe_seed1.prototypes, loaded_probe.prototypes, 64 * 64 * sizeof(float)) == 0);
    std::cout << "   Persistencia verificada con éxito (" << probe_save_path << ").\n";

    /* Microbenchmark de latencia */
    LARGE_INTEGER fq, t0, t1;
    QueryPerformanceFrequency(&fq);
    QueryPerformanceCounter(&t0);
    const uint32_t BENCH_ROUNDS = 10;
    float dummy_d = 0.0f;
    for (uint32_t r = 0; r < BENCH_ROUNDS; r++) {
        for (uint32_t q = 0; q < seed1_data.header.num_queries; q++) {
            geometric_probe_classify(&loaded_probe, seed1_data.queries_R.data() + (size_t)q * 64, &dummy_d);
        }
    }
    QueryPerformanceCounter(&t1);
    double bench_sec = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;
    double ns_per_query = bench_sec * 1e9 / (BENCH_ROUNDS * seed1_data.header.num_queries);
    std::cout << "   Latencia media de clasificación AVX2: " << std::fixed << std::setprecision(2)
              << ns_per_query << " ns por consulta.\n";

    std::string csv_out_path = "hito9/results/confusion_matrix_summary.csv";
    std::ofstream csv(csv_out_path);
    csv << "ProbeMode,Seed,DatasetRole,TotalR,TotalD,TP_R,FN_D,Unknown_R,FP_R,TN_D,Unknown_D,Precision,Recall_R,Specificity_D,FPR_D,FNR_R,BalancedAcc,Latency_ns\n";

    /* ========================================================================
     * MODALIDAD A: Sonda entrenada en Build-Time del Índice (por dataset)
     * ======================================================================== */
    std::cout << "\n========================================================================\n";
    std::cout << "MODALIDAD A: Sonda Geométrica en Build-Time (64 prototipos sobre vectores base)\n";
    std::cout << "========================================================================\n";
    std::cout << std::left << std::setw(6)  << "Seed"
              << std::setw(12) << "Rol"
              << std::setw(10) << "TP (R->R)"
              << std::setw(10) << "FN (R->D)"
              << std::setw(10) << "Unk (R)"
              << std::setw(10) << "FP (D->R)"
              << std::setw(10) << "TN (D->D)"
              << std::setw(10) << "Unk (D)"
              << std::setw(12) << "Prec (%)"
              << std::setw(12) << "Rec R (%)"
              << std::setw(12) << "FPR D (%)"
              << std::setw(12) << "BalAcc (%)" << "\n";
    std::cout << std::string(116, '-') << "\n";

    for (uint32_t s = 1; s <= 7; s++) {
        std::string dpath = (s <= 5) ? ("hito7/data/dataset_seed" + std::to_string(s) + ".bin")
                                     : ("hito8/data/dataset_seed" + std::to_string(s) + ".bin");
        std::string role  = (s <= 5) ? "Calibracion" : "Blind_Test";

        DatasetData d;
        if (!load_dataset(dpath, d)) continue;

        /* Entrenar sonda específica en build-time con muestra de este dataset */
        GeometricProbe build_probe{};
        geometric_probe_build(&build_probe, d.vectors.data(), d.header.num_vectors, d.header.dim, 64, 10, 42ULL);

        ConfusionMatrix cm{};
        confusion_matrix_compute(&build_probe, d.queries_R.data(), d.header.num_queries,
                                 d.queries_D.data(), d.header.num_queries, d.header.dim, &cm);

        csv << "BuildTime," << s << "," << role << "," << cm.total_r << "," << cm.total_d << ","
            << cm.tp << "," << cm.fn << "," << cm.unknown_r << ","
            << cm.fp << "," << cm.tn << "," << cm.unknown_d << ","
            << cm.precision * 100.0 << "," << cm.recall * 100.0 << ","
            << cm.specificity * 100.0 << "," << cm.fpr * 100.0 << ","
            << cm.fnr * 100.0 << "," << cm.balanced_accuracy * 100.0 << ","
            << cm.avg_latency_ns << "\n";

        std::cout << std::left << std::setw(6)  << s
                  << std::setw(12) << role
                  << std::setw(10) << cm.tp
                  << std::setw(10) << cm.fn
                  << std::setw(10) << cm.unknown_r
                  << std::setw(10) << cm.fp
                  << std::setw(10) << cm.tn
                  << std::setw(10) << cm.unknown_d
                  << std::setw(12) << std::fixed << std::setprecision(2) << cm.precision * 100.0
                  << std::setw(12) << cm.recall * 100.0
                  << std::setw(12) << cm.fpr * 100.0
                  << std::setw(12) << cm.balanced_accuracy * 100.0 << "\n";
    }

    /* ========================================================================
     * MODALIDAD B: Sonda estática congelada (Transferencia Cross-Dataset / OOD)
     * ======================================================================== */
    std::cout << "\n========================================================================\n";
    std::cout << "MODALIDAD B: Sonda Estática Congelada en Semilla 1 (Evaluación OOD)\n";
    std::cout << "========================================================================\n";
    std::cout << std::left << std::setw(6)  << "Seed"
              << std::setw(12) << "Rol"
              << std::setw(10) << "TP (R->R)"
              << std::setw(10) << "FN (R->D)"
              << std::setw(10) << "Unk (R)"
              << std::setw(10) << "FP (D->R)"
              << std::setw(10) << "TN (D->D)"
              << std::setw(10) << "Unk (D)"
              << std::setw(12) << "Prec (%)"
              << std::setw(12) << "Rec R (%)"
              << std::setw(12) << "FPR D (%)"
              << std::setw(12) << "BalAcc (%)" << "\n";
    std::cout << std::string(116, '-') << "\n";

    for (uint32_t s = 1; s <= 7; s++) {
        std::string dpath = (s <= 5) ? ("hito7/data/dataset_seed" + std::to_string(s) + ".bin")
                                     : ("hito8/data/dataset_seed" + std::to_string(s) + ".bin");
        std::string role  = (s <= 5) ? "Calibracion" : "Blind_Test";

        DatasetData d;
        if (!load_dataset(dpath, d)) continue;

        ConfusionMatrix cm{};
        confusion_matrix_compute(&loaded_probe, d.queries_R.data(), d.header.num_queries,
                                 d.queries_D.data(), d.header.num_queries, d.header.dim, &cm);

        csv << "FrozenSeed1," << s << "," << role << "," << cm.total_r << "," << cm.total_d << ","
            << cm.tp << "," << cm.fn << "," << cm.unknown_r << ","
            << cm.fp << "," << cm.tn << "," << cm.unknown_d << ","
            << cm.precision * 100.0 << "," << cm.recall * 100.0 << ","
            << cm.specificity * 100.0 << "," << cm.fpr * 100.0 << ","
            << cm.fnr * 100.0 << "," << cm.balanced_accuracy * 100.0 << ","
            << cm.avg_latency_ns << "\n";

        std::cout << std::left << std::setw(6)  << s
                  << std::setw(12) << role
                  << std::setw(10) << cm.tp
                  << std::setw(10) << cm.fn
                  << std::setw(10) << cm.unknown_r
                  << std::setw(10) << cm.fp
                  << std::setw(10) << cm.tn
                  << std::setw(10) << cm.unknown_d
                  << std::setw(12) << std::fixed << std::setprecision(2) << cm.precision * 100.0
                  << std::setw(12) << cm.recall * 100.0
                  << std::setw(12) << cm.fpr * 100.0
                  << std::setw(12) << cm.balanced_accuracy * 100.0 << "\n";
    }

    std::cout << std::string(116, '-') << "\n";
    std::cout << "Resultados completos guardados en " << csv_out_path << "\n";
    return 0;
}
