# ALG-HNSW: Hito 5 — Adaptive Transit (H2-AT) y Preparación Hito 6

Este paquete contiene la implementación completa y congelada de **ALG-HNSW H2-AT (Adaptive Transit)** junto con los datos experimentales, ground truth exacto, kernels AVX2 y resultados del Hito 5.

---

## 1. Arquitectura de H2-AT (Adaptive Transit)

### Componentes Congelados
1. **Representación Vectorial SQ8 (AVX2 U1):**
   - Vectores comprimidos a 8 bits por dimensión ($6,10\text{ MiB}$ vs $24,41\text{ MiB}$ en float32 para 100k vectores).
   - Kernel SIMD AVX2 U1 (Unroll-16, cargas directas `_mm_loadl_epi64`, acumuladores duales independientes): $14,6\text{ ciclos/distancia}$ bajo presión de memoria.
2. **Colas de Prioridad Binarias:**
   - Min-Heap de candidatos con capacidad de 16.384 elementos (`_Thread_local`).
   - Bounded Max-Heap de resultados retenidos ($W$) de capacidad `ef_result` (100).
3. **Desacoplamiento de Presupuestos (H2-AT):**
   - `ef_result`: Capacidad máxima de resultados elegibles retenidos (100).
   - `ef_explore`: Presupuesto de nodos expandidos en la frontera de exploración (512 en `T1-A`).
   - **Regla de tránsito:** Los nodos no elegibles pueden ser atravesados para conectar regiones aisladas del grafo, pero tienen **estrictamente prohibido entrar en $W$** (0 filtraciones verificadas bit a bit).
   - **Auto-corte en alta selectividad:** En cuanto $W$ se llena con 100 elegibles, cualquier candidato cuya distancia supere la cota de $W$ aborta la exploración inmediatamente.
4. **Reranking Float32 Exacto:**
   - Recálculo de distancias L2 exactas en float32 sobre los candidatos retenidos en $W$.

---

## 2. Estructura del Repositorio

- `include/`:
  - `alg_hnsw.h`: API principal, estructuras del grafo, predicados de filtrado y SearchResult.
  - `sq8.h`: API de cuantización SQ8 y kernels AVX2.
  - `heap.h`: Implementación de Min-Heap y Bounded Max-Heap binarios.
- `src/`:
  - `hnsw.c`: Motor HNSW principal con las variantes H0, H1, H2, H2-SQ8 y `hnsw_search_h2_at_sq8`.
  - `sq8.c`: Construcción de índice SQ8 y kernels de distancia AVX2 (Baseline y U1).
  - `heap.c`: Primitivas de sift-up y sift-down para heaps binarios.
  - `filter.c`: Generador determinista de atributos y predicados de filtro multicriterio.
  - `ground_truth.c`: Escaneo lineal exacto (fuerza bruta) sobre el subconjunto elegible.
  - `vector.c`: Generador de vectores sintéticos gaussianos normalizados y consultas (R y D).
  - `distance_avx2.c`: Distancia euclídea float32 AVX2 con FMA.
  - `memory.c`: Asignador alineado a 64 bytes (`aligned_alloc_64`).
  - `main_at.c`: Arnés experimental del Hito 5 (barrido de selectividades 1%, 5%, 10%, 25%, 50% en 5 semillas).
  - `main_sq8_opt.c`: Benchmark de ablación de kernels SQ8 y barrido efSearch.
- `tests/`:
  - `bench_sq8_kernels.c`: Microbenchmark aislado de ciclos y nanosegundos por distancia SQ8.
  - `test_sq8_kernel.c`: Test unitario de precisión numérica del kernel SQ8.
- `results/`:
  - `at_transit_ablation.csv`: Resultados completos del Hito 5 (5 semillas, 5 selectividades, 5 variantes, R y D).
  - `sq8_kernel_ablation.csv`: Resultados de la ablación de kernels del Hito 4.
  - `sq8_efsearch_curve.csv`: Resultados del barrido efSearch del Hito 4.
  - `sq8_ablation.csv`: Resultados de H2-F32 vs H2-SQ8 (Hito 3).
  - `main_ablation.csv`: Resultados de la ablación 2x2 qsort vs Heap (Hito 2).
  - `baseline.csv`: Resultados del baseline H0, H1, H2 (Hito 1).

---

## 3. Compilación y Ejecución

Compilador utilizado: **Zig cc** (Clang/LLVM) en Windows x64:

```powershell
# Compilar arnés de Hito 5 (H2-AT):
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" cc -O3 -march=native -mavx2 -mfma -I include src/main_at.c src/sq8.c src/heap.c src/distance_avx2.c src/vector.c src/filter.c src/ground_truth.c src/hnsw.c src/memory.c -o alg_hnsw_at.exe -lpsapi

# Ejecutar benchmark H2-AT (100k vectores, 1.000 consultas, 5 semillas):
.\alg_hnsw_at.exe --num-vectors 100000 --num-queries 1000 --num-seeds 5
```
