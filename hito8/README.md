# Hito 8 — Adaptive Hybrid Dispatcher

Hito 8 agrega un despachador experimental sin modificar los motores ni los artefactos de Hito 1–7.

## Estado inicial

- Rama: `hito8-adaptive-dispatcher`
- Motores de referencia: ExactFilteredScan, hnswlib y ALG H2-AT.
- FAISS queda como comparador experimental; no se despacha en la primera implementación hasta normalizar su interfaz y semántica de filtro.
- Las implementaciones Hito 7 son actualmente ejecutables independientes. No se deben incluir sus archivos `adapter_*.cpp/.c` directamente porque contienen `main()` y lógica de benchmark. Hito 8 debe extraer o envolver funciones de motor en una capa reutilizable.

## Principios

1. `EXACT` siempre ejecuta ExactFilteredScan.
2. `BALANCED` usa HNSW solamente si el límite inferior de confianza de Recall calibrado alcanza el objetivo para el estrato consultado; si no hay soporte suficiente, ejecuta ExactScan.
3. `APPROXIMATE` elige la menor latencia estimada, sin restricción de Recall.
4. La cardinalidad de elegibles debe llegar como dato precalculado o estimación barata. El dispatcher no debe recorrer los vectores para calcularla.
5. Regret se calcula contra el mejor motor admisible por la política y por la calidad, no contra el más rápido sin restricciones en BALANCED.
6. Los seeds ciegos 6 y 7 no participan en el ajuste de modelos.

## Próximos pasos

- Extraer APIs reutilizables de los adaptadores Hito 7.
- Ajustar modelos de costo y calidad desde CSV de semillas 1–5.
- Generar datasets ciegos 6–7 con centroides nuevos y manifiestos reproducibles.
- Evaluar p50/p95/p99, Recall@10/@100, regret, fallback y overhead del selector.

## Compilación y ejecución

Se documentarán después de integrar las APIs de los motores. No se declara todavía una compilación end-to-end.