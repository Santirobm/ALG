# ALG-HNSW: Plataforma Experimental de Búsqueda Vectorial Filtrada

Plataforma de evaluación empírica y benchmarking riguroso para búsqueda de vecinos más cercanos con filtrado categórico y continuo (*Filtered Approximate Nearest Neighbor Search*), evaluando motores aproximados basados en grafos (HNSW), escaneo exhaustivo contiguo vectorizado (ExactScan AVX2) y selectores híbridos adaptativos en lazo cerrado (Hito 8 e Hito 9).

---

## 📌 Estado del Proyecto: Conclusión Científica Formal (Hito 9)

La línea de investigación experimental de ALG ha concluido formalmente en el **Hito 9** tras completar la batería multi-semilla factorial (**Semillas 8, 9 y 10**) en 6 escenarios OOD (*Out-Of-Distribution*) y 252 estratos experimentales con Bootstrap Jerárquico Bi-Nivel.

### Veredicto y Aplicación de la Barrera de Parada (*Stop Rule*)
Bajo las reglas metodológicas fijadas ex-ante:
> *Si en Seeds 8, 9 y 10 con lazo cerrado de seguridad activo, `Dispatcher_BALANCED` permanece más lento que `Fixed_ExactScan` y `Dispatcher_APPROXIMATE` no entrega una ventaja sustancial, estadísticamente significativa y reproducible (>15%), la línea de investigación se da por finalizada en Hito 9 sin avanzar a Hito 10.*

**Resultado empírico:**
1. **`Dispatcher_BALANCED` es más lento que `Fixed_ExactScan`:** Promedia **$195.04\,\mu\text{s}$** frente a **$190.00\,\mu\text{s}$** de ExactScan (+2.65% de sobrecosto).
2. **`Dispatcher_APPROXIMATE` no supera a `Fixed_ExactScan`:** Promedia **$194.68\,\mu\text{s}$** (+2.46% más lento que ExactScan), sin presentar ventaja reproducible.
3. **Causa raíz:** En 100.000 vectores de 64 dimensiones con selectividades operativas (10% a 25%), un escaneo contiguo vectorizado con AVX2 resuelve las consultas en **$\sim 184\,\mu\text{s}$ (p50)**. El overhead de extracción de características geométricas (~4 $\mu$s), predicción de costos y verificación de deriva en lazo cerrado no se amortiza, dado que HNSW colapsa en regímenes OOD difusos (llegando a $422\,\mu\text{s}$).

Por consiguiente, **el proyecto concluye que un motor ExactScan AVX2 contiguo es técnica y operativamente superior en este régimen frente a cualquier arquitectura híbrida adaptativa.**

---

## 📊 Síntesis Estadística Multi-Semilla (Seeds 8, 9 y 10)

Resultados agregados mediante **Bootstrap Jerárquico Bi-Nivel** ($B = 2.000$ réplicas; Nivel 1: Semillas como clusters primarios; Nivel 2: Consultas por estrato):

| Estrategia | Latencia Media ($\mu$s) | Latencia p50 ($\mu$s) | Latencia p95 ($\mu$s) | Recall@10 (%) | Recall@100 (%) | % a ExactScan | % Fallback | Policy Regret Medio ($\mu$s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Fixed_ExactScan`** | **190.00** | **183.81** | **231.70** | **100.00** | **100.00** | **100.0%** | **0.0%** | **0.00** |
| **`Dispatcher_EXACT`** | 193.11 | 183.73 | 251.88 | 100.00 | 100.00 | 100.0% | 0.0% | 0.00 |
| **`Dispatcher_BALANCED`** | 195.04 | 186.77 | 251.36 | 100.00 | 100.00 | 94.68% | 75.78% | 20.94 |
| **`Dispatcher_APPROXIMATE`** | 194.68 | 186.86 | 278.22 | 99.00 | 99.00 | 80.69% | 47.79% | 35.57 |
| **`Static_Threshold_15pct`**| 177.84 | 168.50 | 249.80 | 97.00 | 93.00 | 42.86% | 0.0% | 18.57 |
| **`Fixed_hnswlib_ef50`** | 254.29 | 236.98 | 398.62 | 95.00 | 89.00 | 0.0% | 0.0% | 87.20 |

### Hallazgos Clave de la Arquitectura de Seguridad
- **Tasa de Falsos Positivos (FPR) = 0.00%:** La sonda geométrica de 64 prototipos evaluó más de 4.500 consultas adversarias $D$ a través de las 3 semillas sin cometer un solo falso positivo ($TN = 100\%$, $FP = 0$).
- **Cumplimiento Determinista de Calidad:** `Dispatcher_BALANCED` mantuvo el **100.00% de Recall@10 y Recall@100** gracias a la auditoría shadow dual, mientras que `Fixed_hnswlib_ef50` cayó a 89% en Recall@100 y el umbral estático del 15% cayó a 93%.
- **Protección ante Desastres Geométricos:** En distribuciones con dispersión difusa ($\sigma = 0.25$), donde HNSW colapsó a $422.3\,\mu\text{s}$, el lazo cerrado activó el fallback al 100%, sirviendo el tráfico en $191.5\,\mu\text{s}$ vía ExactScan.

---

## 🗺️ Estructura del Repositorio y Evolución de Hitos

```
ALG/
├── include/              # Encabezados base C (HNSW, Heaps, SQ8, filtrado)
├── src/                  # Motores congelados C (hnsw.c, heap.c, sq8.c, etc.)
├── hito6/                # Hito 6: Ablación de micro-estructuras y representación
├── hito7/                # Hito 7: Adaptadores ExactScan, hnswlib, FAISS y topología
├── hito8/                # Hito 8: Selector Híbrido Adaptativo y Modelo de Costos
│   ├── include/          # Cost model y hybrid engine API
│   ├── src/              # Implementación del dispatcher y bindings
│   └── benchmarks/       # Evaluación multi-semilla (Seeds 6 y 7)
└── hito9/                # Hito 9: Robustez, OOD, Lazo Cerrado y Cierre Científico
    ├── include/          # Sonda geométrica, calibrador online, métricas de regret
    ├── src/              # Implementaciones en C++17
    ├── datasets/         # Generador sintético factorial OOD (generate_ood.c)
    ├── benchmarks/       # Runner en lazo cerrado, multi-seed matrix y bootstrap
    └── results/          # CSVs por consulta, resúmenes y síntesis bootstrap
```

### Resumen de Fases
- **Fase Inicial / Ablación:** Implementación en C puro de estructuras de colas (qsort vs Min/Bounded Max-Heaps) y cuantización escalar UINT8 (SQ8).
- **Hito 7 (Benchmarking Riguroso):** Comparación estandarizada de ExactScan (AVX2), hnswlib y FAISS bajo 5 semillas fijas e instrumentación de degradación topológica.
- **Hito 8 (Dispatcher Adaptativo):** Primera arquitectura de selección por consulta basada en modelo de costos de regresión cuadrática y calibración por estratos de selectividad.
- **Hito 9 (Robustez y Cierre):** Generalización fuera de distribución (*OOD*), sonda geométrica de 64 centroides, detector robusto de anomalías basado en MAD, auditoría shadow dual de calidad, oráculo con intercalado aleatorio en 3 rondas y bootstrap jerárquico.

---

## 🛠️ Instrucciones de Reproducción

### Requisitos
- Windows 10/11 x64.
- Compilador Clang/LLVM vía **Zig C/C++ 0.17.0+** (o MinGW GCC con flags `-O3 -mavx2 -mfma -std=c++17`).
- Python 3.10+ con `pandas` y `numpy`.

### Compilación y Ejecución de la Matriz Final (Hito 9)

1. **Compilar el generador OOD y el runner:**
   ```powershell
   # Generador de distribuciones OOD
   & "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" cc -O3 -mavx2 -mfma hito9/datasets/generate_ood.c -o hito9/datasets/generate_ood.exe

   # Runner en lazo cerrado con afinidad a P-Core 0
   & "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" c++ -O3 -std=c++17 -mavx2 -mfma -I hito9/include -I hito8/include -I hito7/third_party/hnswlib -I hito7/common `
     hito9/src/geometric_probe.cpp hito9/src/online_calibrator.cpp hito9/src/regret_metrics.cpp `
     hito8/src/cost_model.cpp hito8/src/hybrid_engine.cpp hito8/src/engine_bindings.cpp `
     hito9/benchmarks/run_hito9.cpp -o hito9/benchmarks/run_hito9.exe
   ```

2. **Ejecutar tests unitarios e integración de lazo cerrado:**
   ```powershell
   & "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" c++ -O3 -std=c++17 -mavx2 -I hito9/include hito9/src/online_calibrator.cpp hito9/tests/test_online_calibrator.cpp -o hito9/tests/test_online_calibrator.exe
   .\hito9\tests\test_online_calibrator.exe

   & "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" c++ -O3 -std=c++17 -mavx2 -I hito9/include hito9/src/online_calibrator.cpp hito9/tests/test_closed_loop_integration.cpp -o hito9/tests/test_closed_loop_integration.exe
   .\hito9\tests\test_closed_loop_integration.exe
   ```

3. **Ejecutar la batería multi-semilla (Seeds 8, 9 y 10):**
   ```powershell
   python hito9/benchmarks/run_multi_seed_matrix.py
   ```

4. **Regenerar síntesis y bootstrap jerárquico bi-nivel:**
   ```powershell
   python hito9/benchmarks/bootstrap_regret.py
   python hito9/benchmarks/analyze_ood.py
   ```

---

## 📜 Licencia y Cita
Código desarrollado con fines de experimentación científica y reproducibilidad abierta. Ver ramas individuales para la trazabilidad completa de commits y auditorías metodológicas.
