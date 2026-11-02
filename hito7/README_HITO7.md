# ALG-HNSW Hito 7 — Validación del crossover (formato dataset v2)

## Diseño

Hito 7 conserva los motores del Hito 6 y crea datasets independientes en `hito7/data/`. No sobrescribe los binarios ni CSV del Hito 6.

- 100.000 vectores, dimensión 64; 1.000 consultas por escenario R y D.
- Selectividades: 10.0%, 12.5%, 15.0%, 17.5%, 20.0%, 22.5%, 25.0%.
- Cinco semillas: 42, 1337, 987654321, 0xCAFEBABE, 0xDEADBEEF.
- Ground truth exacto K=100 para las 14 combinaciones escenario/selectividad por semilla.
- Categorías con resolución 10.000 buckets; `selectivities[]` usa décimas de porcentaje (p.ej. 125 = 12.5%) y `target_categories[]` usa el umbral correspondiente (1250).
- Formato binario `DATASET_VERSION=2`; no compatible con lectores Hito 6.

## Orden de ejecución (desde la raíz `alg_hnsw/`)

1. Compilar `hito6/common/dataset_export.c` como `dataset_export.exe` enlazando `src/vector.c`, `src/filter.c`, `src/ground_truth.c` y el resto de dependencias del proyecto según el build local.
2. Ejecutar `dataset_export.exe --seed-idx 1` hasta `--seed-idx 5`.
3. Para cada `hito7/data/dataset_seedN.bin`, ejecutar los cuatro adaptadores Hito 7 (ALG, hnswlib, ExactScan y FAISS). Los adaptadores aceptan el path del dataset como primer argumento; FAISS lo acepta como primer argumento posicional.
4. Cada adaptador escribe archivos separados por semilla y motor dentro de `hito7/results/`; al terminar, consolidar esos CSV para el análisis estadístico.

## Importante: limitaciones que deben quedar explícitas

- Este parche amplía la malla y el formato; no ejecuta los benchmarks. La compilación y las mediciones deben realizarse en Windows sobre el i5-13500 objetivo.
- El generador crea categorías uniformes sobre 10.000 buckets con PRNG determinista por `(seed, vector_id)`. Por tanto, los conteos reales pueden diferir ligeramente de los porcentajes nominales; cada adaptador debe reportar `EligibleCount` medido.
- Los 64 centroides se generan una sola vez con una semilla fija compartida por las cinco semillas; la asignación de puntos y el ruido varían por semilla. Las consultas R/D se generan por semilla sobre cada dataset.
- `ExactFilteredScan` recorre IDs elegibles y accede a `vectors[id]`: es exacto, pero no es necesariamente un recorrido contiguo de vectores. No describirlo como memoria secuencial contigua sin una reorganización física o medición de locality.
- El barrido `efSearch` es 50, 100, 150, 200, 300, 500 para FAISS/hnswlib. ALG conserva H2-AT T1-A y H2-Base.
- Los contadores de componentes conexas y nodos de tránsito elegibles/no elegibles aún no están implementados en este parche.
