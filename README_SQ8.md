# H2-SQ8: experimento de cuantización escalar AVX2

## Qué se agregó

- `include/sq8.h`, `src/sq8.c`: cuantización affine UINT8 por dimensión, con `min[d]` y `scale[d]` calculados sobre el conjunto de vectores.
- `l2_distance_sq8_avx2`: distancia L2 aproximada sobre códigos SQ8. Expande 8 códigos a int32, calcula diferencias y aplica escalas por dimensión con AVX2/FMA.
- `hnsw_search_h2_sq8_filteraware`: usa SQ8 durante el descenso greedy y la navegación H2. Al finalizar, recalcula en float32 la distancia exacta de **todos los candidatos elegibles retenidos en W** y devuelve el top-K reordenado.
- `src/main_sq8.c`: compara H2-F32-Heap contra H2-SQ8-Heap, con cinco semillas y escenarios R/D. Produce `results/sq8_ablation.csv`.

## Importante: no confundir reranking con recuperación de candidatos

El reranking exacto corrige el orden de los candidatos que llegaron a `W`; no puede recuperar un vecino que la navegación SQ8 nunca visitó o descartó. Por eso hay que evaluar Recall@10/100 contra ground truth, no asumir que rerank preserva Recall.

Se rerankean hasta `efSearch` candidatos elegibles retenidos en `W`, no solamente K. Esto reduce el riesgo de perder precisión en el orden final.

## Memoria

Esta primera implementación mantiene el arreglo float32 original porque:

1. HNSW se construye con vectores float32.
2. El reranking exacto requiere los vectores originales.
3. El ground truth también usa float32.

Por lo tanto, **SQ8 agrega un sidecar de aproximadamente 6.1 MiB para 100.000 x 64 códigos, más metadatos**, pero no reduce por sí solo la memoria residente de 86 MB a 30 MB. La reducción real requiere decidir si los float32 originales se guardan fuera de RAM, se eliminan (perdiendo reranking exacto) o se comprimen con otra representación. No declarar ahorro de RAM con esta variante.

## Kernel AVX2

No se usa `_mm256_maddubs_epi16` directamente: la diferencia entre dos códigos UINT8 puede estar en [-255,255], que no cabe en int8, y las escalas por dimensión hacen que la suma de cuadrados requiera ponderación. El kernel evita ese error, expande códigos a int32 y aplica `scale[d]` antes de acumular la suma ponderada.

## Compilación Windows con zig cc

Desde la raíz del proyecto:

```powershell
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" cc -O3 -march=native -mavx2 -mfma -I include `
  src/main_sq8.c src/sq8.c src/heap.c src/distance_avx2.c `
  src/vector.c src/filter.c src/ground_truth.c src/hnsw.c src/memory.c `
  -o alg_hnsw_sq8.exe -lpsapi
```

## Ejecución

```powershell
.\alg_hnsw_sq8.exe --num-vectors 100000 --num-queries 1000 --selectivity 5 --ef-search 100
```

Genera `results/sq8_ablation.csv` con Seed, Scenario, Variant, Recall@10/100, p50/p95/p99, QPS, evaluaciones de distancia de navegación, rerank y nodos visitados.

## Interpretación del experimento

- La comparación principal es SQ8 vs F32 dentro de la misma semilla y escenario.
- Exigir equivalencia de calidad mediante Recall, no igualdad bitwise de IDs (SQ8 cambia deliberadamente las distancias de navegación).
- Informar deltas de Recall por semilla y escenario; no ocultar pérdidas promediando todo.
- El umbral de pérdida de Recall@10 menor a 1 punto porcentual es una hipótesis/criterio de aceptación, no una garantía del cuantizador.
- Ejecutar en el i5-13500 con afinidad, prioridad, compilador y flags iguales a la ablación previa. El rendimiento medido en otro host no es comparable.

## Limitaciones de esta primera versión

- Calibración min/max global por dimensión: sensible a outliers y a consultas fuera del rango de entrenamiento (se saturan a 0/255).
- No hay actualización incremental del cuantizador; si cambian los vectores o rangos, reconstruir el sidecar.
- SQ8 está limitado a dimensiones `<= VECTOR_DIM` en el buscador integrado (actualmente 64).
- El benchmark compara una sola configuración efSearch por ejecución; repetir con varios valores para trazar la curva Recall/latencia.
