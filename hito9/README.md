# Hito 9 — Robustez, Generalización y Calibración Online

**Rama:** hito9-robustness-generalization  
**Base:** hito8-adaptive-dispatcher  
**Estado:** Implementación y auditoría completadas para Tracks 1, 2, 3 y 4. No se modifican los motores ni se incorporan nuevos motores en este hito.

## Objetivo

Determinar si el selector híbrido de Hito 8 puede reconocer consultas fuera de distribución, estimar costos y calidad sin depender de una sonda frágil, y degradar de forma segura cuando sus modelos dejan de ser confiables.

La prioridad es la **validez del selector**, no reducir latencia a cualquier costo. POLICY_EXACT continúa siendo la única política con garantía determinista de Recall 100%.

## Reglas metodológicas

1. Semillas de calibración y semillas de evaluación deben permanecer separadas. Ningún dato de test puede intervenir en el ajuste de centroides, umbrales, modelos ni ventanas.
2. Todo resultado debe publicarse por semilla y estrato antes de presentar agregados.
3. No confundir una anomalía de latencia con una pérdida de calidad: son señales distintas y requieren detectores distintos.
4. El fallback conservador ante incertidumbre es ExactScan.
5. Registrar configuración, versión del compilador, CPU/topología, afinidad, parámetros, semillas y hashes de datasets/binarios.

---

## Track 1 — Sonda geométrica y clasificación de régimen

### Implementación

- Sustituir los 32 vectores equidistantes por una sonda geométrica entrenada en build-time.
- Primera variante: k-means ligero con **64 prototipos** sobre una muestra reproducible de vectores del conjunto de construcción. Evaluar k-medoids como alternativa si el costo o la estabilidad lo justifican.
- La sonda y sus umbrales se entrenan únicamente con datos permitidos de calibración/build. Las semillas ciegas 6 y 7 no participan del entrenamiento ni de la selección de hiperparámetros.
- Guardar centroides, parámetros, semilla del entrenamiento y metadatos en un artefacto versionado.
- Medir costo de construcción, memoria, latencia de clasificación y sensibilidad al cambio de distribución.

### Evaluación obligatoria

Para cada semilla, escenario, selectividad y variante:

- Matriz de confusión completa: TP, FP, TN, FN.
- Precision, recall por clase, FPR, FNR y balanced accuracy.
- Matriz normalizada y conteo absoluto.
- Latencia p50/p95/p99 de la sonda.
- Porcentaje de consultas clasificadas como UNKNOWN o fuera del dominio.

En regímenes mixtos, no reducir la evaluación a una etiqueta binaria artificial: reportar la composición real de la consulta y la decisión del clasificador. La matriz binaria R/D se conserva para comparabilidad con Hito 8.

### Criterio de aceptación

La nueva sonda debe mejorar sustancialmente la cobertura de consultas locales sin introducir falsos positivos adversarios que hagan seleccionar HNSW cuando el modelo no respalda esa decisión. Los límites numéricos se fijarán antes de ejecutar la evaluación ciega.

---

## Track 2 — Generalización y distribuciones OOD

### Matriz experimental

Mantener como control el dataset base y generar escenarios OOD independientes:

| Factor | Valores |
|---|---|
| Mezcla de consultas | R, D, M50/50 |
| Dispersión intracluster | sigma = 0.03, 0.09 (base), 0.25 |
| Número de cuencas | K = 16, 64 (base), 128 |
| Selectividad | Grilla Hito 8: 10–25% |
| Semillas | Calibración separada de validación y test |

Evitar un factorial innecesariamente explosivo en la primera ejecución: usar una matriz principal predefinida y, si el costo lo exige, una matriz secundaria de ablación. No elegir escenarios después de observar qué favorece al selector.

### Métricas

- Calidad: Recall@10 y Recall@100, por consulta y por estrato.
- Rendimiento: p50/p95/p99 y throughput.
- Clasificación: métricas completas del Track 1.
- Enrutamiento: proporción ExactScan/HNSW/UNKNOWN y fallback.
- Robustez: degradación frente a la distribución base y tasa de decisiones inseguras.
- Reportar resultados por semilla, no solo promedios globales.

### Criterio de aceptación

Ante un régimen no representado o con evidencia insuficiente, el selector debe reconocer la incertidumbre y preferir ExactScan. No se considerará generalización demostrada por un promedio global favorable si existen estratos OOD con degradación severa.

---

## Track 3 — Calibración online y fallback seguro

### Separar dos tipos de deriva

**A. Deriva de costo (latencia):**

- Mantener una ventana deslizante de residuales entre latencia observada y estimada para cada combinación de régimen, selectividad y motor.
- Usar una escala robusta además de media/desviación estándar (por ejemplo MAD), para que outliers aislados no descalibren el modelo.
- Si el residual supera el umbral predefinido (la propuesta inicial es >2 sigma), marcar temporalmente el estrato como degradado y enrutar a ExactScan.
- Incorporar mínimo de muestras, cooldown, recuperación con histéresis y límites de memoria/estado.
- Medir falsos disparos, tiempo de detección, duración del fallback y costo acumulado.

**B. Deriva de calidad (Recall):**

La latencia por sí sola **no permite detectar** pérdida de Recall. Se necesita una señal de calidad independiente:

- En benchmark, usar Ground Truth exacto por consulta.
- En un entorno operativo, evaluar auditorías shadow de una muestra de consultas contra ExactScan, o un conjunto de consultas de validación etiquetadas. Documentar el costo de esa auditoría.
- Si no existe señal de calidad verificable, no afirmar que el sistema detecta degradación de Recall online. En ese caso, aplicar límites conservadores de dominio y fallback ante OOD.

### Estados propuestos por estrato

CALIBRATED → SUSPECT → FALLBACK_EXACT → RECOVERY_PROBE → CALIBRATED

Las transiciones deben ser deterministas, testeables y registradas. La política EXACT no depende de este mecanismo.

### Pruebas

- Deriva abrupta y gradual de latencia.
- Deriva de calidad con latencia estable.
- Outliers aislados y ráfagas.
- Ventana insuficiente, datos faltantes y reinicio del proceso.
- Recuperación después de una falsa alarma.
- Confirmar que ninguna consulta marcada OOD se enruta a HNSW por defecto.

---

## Track 4 — Regret auditado e incertidumbre estadística

### Definiciones

Para cada consulta q:

R_signed(q) = T_selector(q) - T*_admisible(q)

R_+(q) = max(0, R_signed(q))

El conjunto admisible debe definirse **ex ante**, usando la misma regla de calidad y la misma información disponible para la política. No usar Recall individual ex-post para construir el oráculo de una política que decide mediante LCB por estrato.

Documentar por separado:
- Oráculo de latencia sin restricción (solo diagnóstico).
- Oráculo admisible por política y objetivo de calidad (regret principal).
- Consultas para las que no existe motor aproximado admisible (ExactScan como referencia).

Los valores negativos de Signed Regret se conservan; Positive Regret nunca puede ser negativo. No truncar la serie signed.

### Estadística

Por estrategia, escenario, selectividad y semilla:

- Signed Regret y Positive Regret: media, p50, p95 y p99.
- Intervalos de confianza del 95%.
- Latencias y Recall con intervalos, además de percentiles.
- Resultados individuales por semilla y síntesis agregada.

Usar bootstrap jerárquico/agrupado: remuestrear semillas como unidades independientes y luego consultas dentro de cada semilla. No tratar miles de consultas de una misma semilla como miles de réplicas independientes del proceso generador.

### Criterio de aceptación

El cálculo debe ser reproducible desde los CSV por consulta y producir el mismo resultado al regenerar el informe. Publicar también el método, número de réplicas, semilla del bootstrap y unidad de remuestreo.

---

## Estructura de archivos propuesta

    hito9/
      README.md
      config/
        experiment_matrix.json
        acceptance_thresholds.json
      include/
        geometric_probe.h
        online_calibrator.h
        regret_metrics.h
      src/
        geometric_probe.cpp
        online_calibrator.cpp
        regret_metrics.cpp
      datasets/
        generate_ood.c
        manifests/
      benchmarks/
        run_hito9.cpp
        analyze_hito9.py
      tests/
        test_geometric_probe.cpp
        test_online_calibrator.cpp
        test_regret_metrics.cpp
      results/
        raw/
        per_seed/
        synthesis/

Esta estructura es una propuesta; los nombres y APIs definitivos se fijarán al inspeccionar los contratos reales de Hito 8 antes de implementar.

## Orden de ejecución

1. Congelar y documentar el baseline Hito 8.
2. Implementar y testear la sonda geométrica; validar matriz de confusión.
3. Generar y validar manifiestos OOD.
4. Ejecutar la matriz OOD sin recalibrar con los datos de test.
5. Corregir regret y bootstrap; reproducir informes desde datos crudos.
6. Implementar calibración online de costo y, por separado, auditoría de calidad.
7. Ejecutar pruebas de deriva, fallback y recuperación.
8. Publicar una síntesis con límites, fallos conocidos y criterios de uso.

## Fuera de alcance

- Incorporar FAISS u otros motores al dispatcher.
- Modificar hnswlib o el núcleo ALG de hitos anteriores.
- Prometer Recall determinista en BALANCED.
- Presentar una alarma de latencia como detector de pérdida de calidad.
- Ajustar modelos o umbrales mirando los resultados finales de test.


## Track 2 — Implementación inicial del generador OOD

El generador inicial se encuentra en:

- `hito9/datasets/ood_dataset_format.h`
- `hito9/datasets/generate_ood.c`
- `hito9/config/experiment_matrix.json`

### Formato y etiquetas

El generador escribe un formato binario propio versionado (`AOD9`), no el formato Hito 7/8. Esto evita que un lector antiguo interprete silenciosamente consultas mixtas como si fueran dos bloques homogéneos R/D.

Cada registro de consulta lleva una etiqueta explícita de clase y componente:
- Clase: `OOD_QUERY_LOCAL` o `OOD_QUERY_ADVERSARIAL`.
- Componente: `OOD_COMPONENT_LOCAL_BASIN`, `OOD_COMPONENT_LOCAL_UNIFORM` o `OOD_COMPONENT_ADVERSARIAL_UNIFORM`.

En `M50/50`, el archivo contiene ambas clases en una mezcla 50/50 y conserva la etiqueta de cada consulta. La distribución R mantiene 85% de consultas concentradas alrededor de representantes de cuencas distintas y 15% uniformes. El generador escribe además un CSV lateral con la semilla, parámetros y conteos reales por componente.

### Compilación y uso

Compilar como C11 con la biblioteca matemática. Ejemplo:

```sh
gcc -O3 -std=c11 hito9/datasets/generate_ood.c -lm -o generate_ood
./generate_ood --seed 8001 --basins 64 --sigma 0.25 --mix M50/50 --out hito9/datasets/manifests/ood_s8001_k64_s025_mixed.bin
```

Parámetros permitidos:
- `--basins`: 16, 64 o 128.
- `--sigma`: 0.03, 0.09 o 0.25.
- `--mix`: R, D o M50/50.
- `--seed`, `--num-vectors`, `--num-queries`, `--out`.


El Ground Truth no se incrusta en el generador. El runner (`run_hito9.cpp`) lo calcula exactamente a partir de los vectores y categorías guardados (fuerza bruta exacta top-100 por query y selectividad), desacoplando completamente la generación de datos de la evaluación del motor.

---

## Síntesis Experimental de Hito 9

### 1. Track 1 — Sonda Geométrica Build-Time
- K=64 prototipos esféricos entrenados sobre el subconjunto de indexación.
- **0.00% False Positive Rate en consultas adversarias (D)** a través de todas las semillas evaluadas (1 a 8).
- Latencia de clasificación AVX2: **265 ns** por consulta.

### 2. Track 2 — Evaluación en Matriz OOD Primaria (Semilla 8)
- 6 configuraciones factoriales (mezcla M50/50, dispersiones sigma in {0.03, 0.09, 0.25}, cuencas K in {16, 64, 128}).
- 252.000 evaluaciones de consultas a lo largo de 7 selectividades (10.0% a 25.0%).
- Con alta dispersión (sigma = 0.25), el espacio difuso colapsa la velocidad del grafo HNSW (465.4 µs p50 frente a 183.8 µs de ExactScan). La sonda detecta la lejanía a los prototipos y enruta el 99.8% a ExactScan, protegiendo tanto la latencia como el recall.

### 3. Track 3 — Calibrador Online y Arnés de Calidad Shadow
- Implementado en `hito9/include/online_calibrator.h` y `hito9/src/online_calibrator.cpp`.
- Verificado exhaustivamente en `hito9/tests/test_online_calibrator.cpp`:
  - Outliers aislados transicionan a `SUSPECT` y se recuperan a `CALIBRATED` sin falsos disparos de fallback.
  - Derivas persistentes activan determinísticamente `FALLBACK_EXACT` con enfriamiento e histéresis en `RECOVERY_PROBE`.
  - Demostración empírica de separación: la pérdida de recall con latencia plana no altera el monitor de costo pero es detectada por el arnés de auditoría shadow (`quality_audit_is_degraded`), activando el fallback de calidad.
  - Toda consulta fuera de distribución o no reconocida (`QUERY_UNKNOWN`) deriva forzosamente a ExactScan.

### 4. Track 4 — Síntesis Estadística con Bootstrap Jerárquico (B=2000)

Resumen global auditado sobre los 252 estratos evaluados:

| Estrategia | Latencia Media (µs) [IC 95%] | Recall@10 Media [IC 95%] | Recall@100 Media [IC 95%] | Positive Regret (µs) [IC 95%] | Regret p95 (µs) | Enrutamiento ExactScan (%) |
|---|---|---|---|---|---|---|
| **Fixed_ExactScan** | 206.0 [205.2, 206.9] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] | 11.89 [11.39, 12.39] | 68.2 | 100.0% |
| **Dispatcher_EXACT** | 207.9 [207.0, 208.8] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] | 12.81 [12.28, 13.36] | 69.3 | 100.0% |
| **Dispatcher_BALANCED** | 215.3 [214.3, 216.3] | **99.99% [99.99%, 100.00%]** | **99.97% [99.97%, 99.97%]** | 19.68 [18.93, 20.38] | 120.7 | 88.9% |
| **Dispatcher_APPROXIMATE** | 197.8 [196.7, 198.8] | 98.16% [98.08%, 98.23%] | 96.50% [96.41%, 96.60%] | 25.28 [24.48, 26.08] | 141.1 | 56.9% |
| **Static_Threshold_15pct** | 192.0 [191.1, 192.9] | 96.66% [96.57%, 96.76%] | 93.48% [93.36%, 93.60%] | 19.16 [18.64, 19.70] | 121.6 | 42.9% |
| **Fixed_hnswlib_ef50** | 281.8 [280.3, 283.5] | 95.06% [94.95%, 95.17%] | 89.67% [89.55%, 89.80%] | 95.32 [93.78, 96.80] | 405.8 | 0.0% |

**Conclusión Clave:**
Un umbral estático de selectividad (15%) degrada Recall@10 a 96.66% y Recall@100 a 93.48% bajo distribuciones adversarias y difusas. Por el contrario, el selector adaptativo con sonda geométrica y calibrador mantiene **99.99% de Recall@10** en `BALANCED` aislando de forma segura las consultas OOD y adversarias hacia ExactScan. Para garantías estrictas y deterministas de Recall 100%, `Dispatcher_EXACT` añade solo +1.9 µs de sobrecarga respecto al escaneo bruto.
