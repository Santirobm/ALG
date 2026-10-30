# ALG-HNSW: Plataforma Experimental de Búsqueda Vectorial Filtrada

Plataforma modular para evaluar HNSW bajo predicados de filtrado categóricos y continuos.

## Nueva ablación 2x2: qsort vs heaps / H2 vs H3-128

Se incorporó `src/heap.c` + `include/heap.h` y un ejecutable nuevo `src/main_ablation.c`.

| Caso | Cola de candidatos | Conjunto W | Representación |
|---|---|---|---|
| A | qsort + extracción por desplazamiento | qsort | H2 |
| B | Min-Heap binario | Bounded Max-Heap | H2 |
| C | qsort + extracción por desplazamiento | qsort | H3-128 |
| D | Min-Heap binario | Bounded Max-Heap | H3-128 |

La selección se controla con `g_hnsw_use_heaps` (0=qsort, 1=heaps). El arnés cambia el modo entre ejecuciones, rota el orden de las cuatro variantes por semilla/escenario y escribe `results/main_ablation.csv`.

Antes de medir, el arnés compara los resultados de qsort y heap para hasta 100 consultas por escenario y valida igualdad bitwise de IDs y distancias para H2 y H3. También imprime Recall@10/100, p50/p95/p99, QPS, evaluaciones de distancia, hit-rate y mantenimiento.

### Comparador determinista

El orden total común es:
1. Distancia ascendente.
2. En empate, ID ascendente.
3. NaN se ordena después de las distancias numéricas; entre NaN, desempate por ID.

Esto reemplaza el comparador anterior que solo miraba distancia. Por lo tanto, los casos de distancias exactamente iguales pueden cambiar respecto de los binarios previos, pero qsort y heaps usan ahora la misma regla.

### Compilación en Windows (zig cc)

Desde la raíz del proyecto:

```powershell
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" cc -O3 -march=native -mavx2 -mfma -I include `
  src/main_ablation.c src/heap.c src/hnsw_tiles.c src/distance_avx2.c `
  src/vector.c src/filter.c src/ground_truth.c src/hnsw.c `
  src/benchmark.c src/memory.c -o alg_hnsw_ablation.exe -lpsapi
```

Los ejecutables `.exe` y `.pdb` incluidos en el ZIP original son binarios de la versión anterior a esta ablación. Hay que recompilar para generar `alg_hnsw_ablation.exe`.

### Ejecución

```powershell
.\alg_hnsw_ablation.exe --num-vectors 100000 --num-queries 1000 --selectivity 5 --ef-search 100
```

`ef-search` debe estar entre 1 y 512 porque los buffers de candidatos/resultados son arrays fijos de 512 elementos. Para Recall@100, el benchmark eleva el valor efectivo a un mínimo de 100.

## Estructura

- `include/alg_hnsw.h`: tipos y API del motor.
- `include/heap.h`: API de colas y selector qsort/heaps.
- `src/hnsw.c`: motor HNSW y búsquedas H0/H1/H2.
- `src/hnsw_tiles.c`: búsqueda H3 con Micro-Tiles y Pointer Swizzling.
- `src/heap.c`: Min-Heap y Bounded Max-Heap sin asignaciones dinámicas.
- `src/main_ablation.c`: matriz 2x2 y validación exacta.
- `src/main.c`, `src/main_h3.c`: arneses anteriores.
- `results/baseline.csv`, `results/h3_ablation.csv`: resultados originales, sin modificación.

## Criterio de interpretación

La comparación B contra A cuantifica el efecto de sustituir el ordenamiento de colas en H2. D contra C mide el mismo cambio en H3-128. D contra B aísla el efecto de la representación H3-128 una vez que ambos motores usan heaps. No inferir una mejora general a partir de una única semilla o escenario: usar los cinco seeds y ambos escenarios, y reportar dispersión además de medianas.

## Experimento SQ8 (H2-Heap)

Se agregó una primera implementación experimental de cuantización escalar UINT8 por dimensión y navegación H2 con distancia AVX2 ponderada, seguida de reranking float32 exacto de todos los candidatos elegibles retenidos en W.

- Implementación y límites: `README_SQ8.md`
- API: `include/sq8.h`
- Cuantizador/kernel: `src/sq8.c`
- Búsqueda integrada: `hnsw_search_h2_sq8_filteraware` en `src/hnsw.c`
- Benchmark F32 vs SQ8: `src/main_sq8.c`
- Test del kernel: `tests/test_sq8_kernel.c`

La variante SQ8 mantiene los vectores float32 originales para el reranking. Por tanto, esta primera fase **no reduce la RAM total**; agrega un sidecar de códigos. La calidad y el rendimiento deben medirse en el i5-13500 con el benchmark de 100K y cinco semillas antes de decidir si se conserva.
