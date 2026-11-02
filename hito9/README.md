# Hito 9 — Robustez, Generalización y Calibración Online

**Rama:** hito9-robustness-generalization  
**Base:** hito8-adaptive-dispatcher  
**Estado:** planificación metodológica. No se modifican los motores ni se incorporan nuevos motores en este hito.

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

Cada registro de consulta lleva una etiqueta explícita:
- `OOD_QUERY_LOCAL`
- `OOD_QUERY_ADVERSARIAL`

En `M50/50`, el archivo contiene ambas clases en una mezcla 50/50 (la consulta conserva su etiqueta). La distribución R mantiene 85% de consultas concentradas alrededor de nodos de cuenca y 15% uniformes, como en el generador Hito 8.

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

El Ground Truth no se incrusta en el generador. El runner deberá calcularlo exactamente a partir de los vectores y categorías guardados, para evitar mezclar generación de datos con evaluación del motor. El formato incluye la grilla de selectividades de Hito 8.

**Limitación actual:** esta entrega implementa generación reproducible de corpus y consultas etiquetadas; todavía no implementa el lector/runner ni el cálculo de Ground Truth y no se presenta como benchmark OOD ejecutado.
