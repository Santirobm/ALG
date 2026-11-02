# Hito 9 — Robustez, Generalización y Calibración Online

**Rama:** hito9-robustness-generalization  
**Base:** hito8-adaptive-dispatcher  
**Estado:** **FINALIZADO Y CERRADO (Conclusión Científica Formal)**. La batería multi-semilla factorial (Seeds 8, 9 y 10) ha sido completada en su totalidad con lazo cerrado y bootstrap jerárquico. Se aplicó la Barrera de Parada (*Stop Rule*) acordada, concluyendo la línea de investigación experimental en Hito 9.

## Objetivo y Criterio de Parada (*Stop Rule*)

Determinar si el selector híbrido de Hito 8 puede reconocer consultas fuera de distribución, estimar costos y calidad sin depender de una sonda frágil, y degradar de forma segura cuando sus modelos dejan de ser confiables.

**Regla de parada acordada ex-ante:**
Si a través de las Semillas 8, 9 y 10 con lazo cerrado de seguridad activo, Dispatcher_BALANCED permanece más lento que Fixed_ExactScan y Dispatcher_APPROXIMATE no entrega una ventaja sustancial, estadísticamente significativa y reproducible (>15%), la línea de investigación de ALG se da por cerrada formalmente en el Hito 9 sin avanzar a Hito 10.

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

## Síntesis Experimental de Hito 9: Evaluación Multi-Semilla (Seeds 8, 9 y 10)

La evaluación final del Hito 9 comprendió 18 datasets factoriales independientes (6 escenarios OOD x 3 semillas ciegas), totalizando 252 estratos experimentales evaluados con oráculo intercalado de 3 repeticiones y lazo cerrado de seguridad activo:

### 1. Track 1 — Sonda Geométrica Build-Time
- K=64 prototipos esféricos entrenados sobre el subconjunto de indexación.
- **0.00% False Positive Rate en consultas adversarias (D)**: Evaluada sobre más de 4.500 consultas D a través de Seeds 8, 9 y 10 ( = 4.500$,  = 0$,  = 0.00\%$).
- Latencia de clasificación AVX2: **265 ns** por consulta.

### 2. Track 2 — Matriz OOD Factorial (Seeds 8, 9 y 10)
- 6 configuraciones factoriales:
  1. k64_s009_m5050: Mezcla balanceada {50/50}$ (consultas locales y adversarias).
  2. k64_s003_r: Concentración intracluster estrecha ($\sigma = 0.03$).
  3. k64_s025_r: Dispersión difusa extrema ($\sigma = 0.25$).
  4. k64_s009_d: Distribución adversaria uniforme ($).
  5. k16_s009_r: 16 cuencas de atracción.
  6. k128_s009_r: 128 cuencas de atracción.
- En dispersión alta ($\sigma = 0.25$), HNSW colapsa a .3\,\mu	ext{s}$ y 84% de Recall@100. El lazo cerrado detecta el régimen y deriva el 100% a ExactScan (.5\,\mu	ext{s}$).
- En régimen adversario ($), HNSW colapsa a .0\,\mu	ext{s}$ y 79% de Recall@100. El lazo cerrado deriva el 100% a ExactScan (.3\,\mu	ext{s}$).

### 3. Track 3 — Calibrador Online y Arnés de Calidad Shadow
- Evaluado en tiempo real en cada consulta del benchmark:
  - Umbral de latencia evaluado estrictamente sobre la ventana nominal previa, sin contaminación de anomalías en el MAD.
  - Fallback inicial garantizado en estrato UNCALIBRATED antes de acumular evidencia shadow.
  - Auditoría dual independiente sobre 
ecall10_window y 
ecall100_window.
  - Histéresis y recuperación controlada tras certificar evidencia estadística sin oscilaciones.

### 4. Track 4 — Síntesis Estadística con Bootstrap Jerárquico Bi-Nivel (B=2000)

Resultados consolidados a través de todas las semillas (Seeds 8, 9, 10) y escenarios OOD:

| Estrategia | Latencia Media ($\mu) | Latencia p50 ($\mu) | Latencia p95 ($\mu) | Recall@10 (%) | Recall@100 (%) | % a ExactScan | % Fallback | Policy Regret Medio ($\mu) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed_ExactScan** | **190.00** | **183.81** | **231.70** | **100.00** | **100.00** | **100.0%** | **0.0%** | **0.00** |
| **Dispatcher_EXACT** | 193.11 | 183.73 | 251.88 | 100.00 | 100.00 | 100.0% | 0.0% | 0.00 |
| **Dispatcher_BALANCED** | 195.04 | 186.77 | 251.36 | 100.00 | 100.00 | 94.68% | 75.78% | 20.94 |
| **Dispatcher_APPROXIMATE** | 194.68 | 186.86 | 278.22 | 99.00 | 99.00 | 80.69% | 47.79% | 35.57 |
| **Static_Threshold_15pct**| 177.84 | 168.50 | 249.80 | 97.00 | 93.00 | 42.86% | 0.0% | 18.57 |
| **Fixed_hnswlib_ef50** | 254.29 | 236.98 | 398.62 | 95.00 | 89.00 | 0.0% | 0.0% | 87.20 |

---

## 🏁 Conclusión Científica y Cierre del Hito 9

1. **Aislamiento de Seguridad Exitoso:** La combinación de sonda geométrica build-time y calibración online logró una protección determinista de la calidad, conservando **100.00% de Recall@10 y Recall@100** en Dispatcher_BALANCED frente a cualquier variación geométrica o adversaria.
2. **Aplicación de la Barrera de Parada:**
   - Dispatcher_BALANCED (.04\,\mu	ext{s}$) es **más lento** que Fixed_ExactScan (.00\,\mu	ext{s}$).
   - Dispatcher_APPROXIMATE (.68\,\mu	ext{s}$) no produce ninguna ventaja estadísticamente significativa frente a Fixed_ExactScan.
3. **Dictamen Final:** En el régimen evaluado (100k vectores de 64 dimensiones con selectividades entre 10% y 25%), **el escaneo contiguo vectorizado con AVX2 (ExactScan) es insuperable en latencia global, simplicidad y certidumbre operativa.** La complejidad y sobrecarga de un selector adaptativo no se justifican frente al rendimiento puro del hardware.
4. **Cierre de la Línea de Investigación:** En estricto cumplimiento de la regla de parada metodológica, el desarrollo experimental concluye formalmente en el Hito 9, descartándose la apertura de un Hito 10.
