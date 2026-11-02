# Hito 8 — Selector Híbrido Adaptativo (Adaptive Vector Dispatcher)

El Hito 8 implementa y valida experimentalmente un selector de consultas híbrido y desacoplado para búsqueda vectorial con filtrado categórico. El despachador selecciona dinámicamente entre **ExactFilteredScan** (escaneo lineal AVX2 exacto sobre lista de elegibles) y **hnswlib** (búsqueda aproximada basada en grafo con `BaseFilterFunctor`), optimizando la latencia bajo restricciones de calidad estadística sin modificar el núcleo congelado de los hitos anteriores.

---

## 1. Arquitectura y Componentes (`hito8/`)

- `include/hybrid_engine.h`: Definición de tipos públicos, contratos de políticas (`POLICY_EXACT`, `POLICY_BALANCED`, `POLICY_APPROXIMATE`), regímenes de consulta (`QUERY_LOCAL`, `QUERY_ADVERSARIAL`, `QUERY_UNKNOWN`) y firma del despachador `hybrid_search`.
- `include/cost_model.h` / `src/cost_model.cpp`: Modelos paramétricos de costo y calidad:
  - $T_{\text{exact}}(N_{el}) = 15.93\,\mu\text{s} + 0.00895\,\mu\text{s} \times N_{el}$ ($R^2 = 0.961$).
  - Modelo HNSW calibrado multi-semilla (Semillas 1–5 en Core i5-13500) con límite inferior de confianza (LCB) al 95%.
- `include/engine_bindings.h` / `src/engine_bindings.cpp`: Capa de bindings desacoplada para `exact_scan_binding_search` y `hnswlib_binding_search`, junto con la sonda de régimen de consulta observable `extract_query_regime` (32 centroides muestra, < 25 ns con AVX2).
- `datasets/generate_blind.c`: Generador reproducible de datasets ciegos out-of-sample (Semillas 6 y 7) con 64 centroides completamente independientes, perturbaciones gaussianas y Ground Truth exacto de 14 bloques (R/D x 7 selectividades).
- `benchmarks/run_hito8.cpp`: Harness de evaluación end-to-end con fijación estricta a P-Core 0 (`SetProcessAffinityMask`), medición de latencia QPC, recall@10/@100 y cálculo de regret admisible:
  $$R_{\pi}(q) = T_{\text{selector}}(q) - \min_{m \in \mathcal{M}_{\text{admissible}}(\pi, q)} T_m(q)$$
- `benchmarks/analyze_hito8.py`: Agregador estadístico y generador de matrices comparativas de síntesis multi-semilla.

---

## 2. Protocolo de Evaluación Ciega (Out-of-Sample Seeds 6 y 7)

Las semillas 6 y 7 **no participaron en la calibración de modelos**. Cada semilla cuenta con:
- $N = 100.000$ vectores de dimensión 64.
- 64 centroides sintéticos independientes generados con RNG exclusivo.
- 1.000 consultas por escenario:
  - **Escenario R (Local)**: 85% de consultas concentradas en cuencas calientes.
  - **Escenario D (Adversario)**: 100% de consultas dispersas en la hiperesfera.
- 7 selectividades de grilla fina: 10.0%, 12.5%, 15.0%, 17.5%, 20.0%, 22.5%, 25.0%.
- Ground truth exhaustivo de 14.000 consultas por semilla.

Estrategias evaluadas:
1. `Fixed_ExactScan`: Control de exactitud absoluta (100% recall).
2. `Fixed_hnswlib_ef50`: Baseline aproximado fijo.
3. `Static_Threshold_15pct`: Regla heurística estática (corta a 15% independientemente del escenario).
4. `Dispatcher_EXACT`: Política de fidelidad determinista.
5. `Dispatcher_BALANCED`: Enrutamiento con garantía de calidad empírica (target Recall@10 $\ge 95\%$).
6. `Dispatcher_APPROXIMATE`: Minimización irrestricta de latencia.

---

## 3. Resultados Experimentales Agregados (Semillas 6 y 7)

### Matriz de Resumen Global (Promedio en todo el espectro):

| Estrategia | Latencia Med ($\mu$s) | Recall@10 (%) | Recall@100 (%) | % Exact | % HNSW | Regret Med ($\mu$s) | Regret p95 ($\mu$s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dispatcher_EXACT** | 192.3 | **100.00%** | **100.00%** | 100.0% | 0.0% | 8.61 | 41.46 |
| **Fixed_ExactScan** | 192.9 | **100.00%** | **100.00%** | 100.0% | 0.0% | 7.25 | 43.86 |
| **Dispatcher_BALANCED** | 194.4 | **100.00%** | **100.00%** | 97.2% | 2.8% | 28.86 | 91.27 |
| **Dispatcher_APPROXIMATE** | 166.3 | 97.65% | 95.96% | 67.2% | 32.8% | 10.75 | 44.57 |
| **Static_Threshold_15pct** | 161.8 | 95.82% | 92.38% | 42.9% | 57.1% | 9.01 | 42.45 |
| **Fixed_hnswlib_ef50** | 221.7 | 94.02% | 87.98% | 0.0% | 100.0% | 59.82 | 172.98 |

---

## 4. Hallazgos Metodológicos Clave

1. **Invalidez de umbrales estáticos universales:**
   - En el Escenario D (Adversario), el crossover se desplaza hacia $\sim 22.5\%\text{--}25.0\%$ porque el recorrido de grafos en espacios dispersos sufre un costo severo ($250\text{--}400\,\mu\text{s}$ frente a $110\text{--}180\,\mu\text{s}$ de ExactScan).
   - El selector estático (`Static_Threshold_15pct`) enruta erróneamente a HNSW a partir del 17.5% en Escenario D, degradando el Recall@10 al **84.88%** (y Recall@100 al **74.69%**), mientras que ExactScan era más rápido y 100% exacto.
2. **Protección de calidad de `Dispatcher_BALANCED`:**
   - En el Escenario D, los modelos calibrados detectan correctamente que el Recall LCB de HNSW está por debajo del 95% ($\approx 83\%\text{--}65\%$). En consecuencia, `Dispatcher_BALANCED` **enruta el 100% de las consultas en Escenario D a ExactFilteredScan**, preservando el 100.00% de Recall en todo el estrato adverso.
3. **Overhead sub-microsegundo verificado:**
   - El overhead del despachador (evaluación de sonda métrica + cálculo analítico de costos + decisión) es de **14.57 nanosegundos** en microbenchmark y $< 1\,\mu\text{s}$ end-to-end, amortizándose completamente en cualquier consulta real.

---

## 5. Instrucciones de Compilación y Ejecución

Compilar con Zig C++ sobre Windows x64:
```powershell
# 1. Compilar tests unitarios
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" c++ -O3 -mavx2 -mfma -std=c++17 hito8/tests/test_dispatcher.cpp hito8/src/cost_model.cpp hito8/src/hybrid_engine.cpp -o hito8/tests/test_dispatcher.exe
.\hito8\tests\test_dispatcher.exe

# 2. Compilar generador ciego y benchmark
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" cc -O3 -mavx2 -mfma hito8/datasets/generate_blind.c -o hito8/datasets/generate_blind.exe
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" c++ -O3 -mavx2 -mfma -std=c++17 hito8/benchmarks/run_hito8.cpp hito8/src/cost_model.cpp hito8/src/hybrid_engine.cpp hito8/src/engine_bindings.cpp -o hito8/benchmarks/run_hito8.exe

# 3. Generar datasets ciegos 6 y 7
.\hito8\datasets\generate_blind.exe --seed-idx 6
.\hito8\datasets\generate_blind.exe --seed-idx 7

# 4. Ejecutar benchmark sobre P-Core 0
.\hito8\benchmarks\run_hito8.exe hito8/data/dataset_seed6.bin
.\hito8\benchmarks\run_hito8.exe hito8/data/dataset_seed7.bin

# 5. Generar informe sintético
python hito8/benchmarks/analyze_hito8.py
```
