# Registro del estudio: qué es evidencia y qué sigue abierto

Actualizado el 7 de octubre de 2026. Este índice conserva el hilo científico; los recibos de ejecución, hashes y bitácoras de cada fase son la fuente de verdad operativa. Los resultados de una fase ya observada no se vuelven a presentar como un test independiente para una receta elegida después.

| Fase | Estado actual | Pregunta y alcance | Evidencia / siguiente acción |
|---|---|---|---|
| v1, modelos 1.7B | Terminada | Recompensa binaria, condicionada y control fijo cuando existe; prompting/filtro | `artifacts/runs/`, `artifacts/predictions/`; mantener puntuación original |
| Extensión Qwen3-14B inicial | Terminada | Viabilidad local y primer condicionado, semilla 17 | `experiments/14b-20260930/` |
| Controles de 14B | Terminada | Tres semillas de binario, condicionado y fijo; desarrollo y filtro | `experiments/14b-controls-20260930/` |
| Auditoría humana de 150 respuestas histórica | Recibida y analizada | Un anotador usó IA de apoyo y revisó personalmente cada etiqueta; muestra estratificada | `artifacts/reviews/2026-09-30-human-audit/`; no tratar desacuerdo bruto como tasa poblacional |
| Validación de juez semántico | Falló criterios | Casos difíciles, indeterminación y referencias defectuosas | `artifacts/reviews/2026-09-30-hard-evaluator/`; no usarlo como recompensa |
| Confirmación de 28 variantes existentes | Terminada y revisada | Preguntas nuevas de TriviaQA/NQ, sin reentrenar ni elegir checkpoints | `experiments/final-eval-20261001/`; `docs/resultados-para-paper.md` (dossier local) |
| Diagnóstico TRAIN de crédito RL | Terminado y revisado | G4/G8/G16 sobre las mismas 6,144 salidas; no actualiza pesos | `experiments/rl-improvements-20261001/`; no es evidencia de una política mejor |
| **Tres mejoras v2** | **Terminada y revisada** | 12 entrenamientos RL, tres selectores; 2,000 preguntas nuevas de TriviaQA y 500 de NQ | [Resultados](../experiments/v2-20261002/artifacts/analysis/results.md), [revisión independiente](../experiments/v2-20261002/artifacts/reviews/2026-10-04-completion/scientific.json), [protocolo](tres-mejoras-v2.md) |
| Auditoría v2 y sensibilidad parcial | Recibida, importada y analizada; agregados incluidos | 150 unidades, preanotación ChatGPT y revisión personal posterior; tres casos indeterminados conservados | [Método y límites](auditoria-v2.md), [seis agregados anónimos](../results/v2-audit-20261006/); no sustituye los endpoints originales |

## Decisiones de v2 y su motivo

1. Reducir el abanico del diagnóstico a tres mecanismos comprobables. Se posponen barridos de LR/KL, modelos de 30B, nuevos jueces y remuestreo selectivo: no forman parte de esta receta.
2. Mantener Qwen3-14B, original sin SFT, y tres semillas. Ya cabe en memoria; la hipótesis es mejorar el aprendizaje o la selección, no demostrar causalmente un efecto del tamaño.
3. Comparar G8 con dos grupos de 4 a **ocho generaciones por prompt en ambos**. Usar LOO en ambos evita confundir el cambio de tamaño con su factor de reducción de la ventaja centrada.
4. Repetir cada pregunta tres veces en todos los brazos de RL. Solo cambia si esas exposiciones contienen uno o tres umbrales. El control fijo recibe los mismos prompts que el condicionado.
5. Entrenar con 0.60/0.75/0.90, cuyo promedio es 0.75; reservar 0.65/0.85 para el contraste principal. Esto cambia el protocolo frente a v1 y obliga a interpretar v1 como antecedente.
6. Separar generador y selector para estudiar calibración/discriminación sin destruir capacidad de respuesta. El selector aprende exact match sobre candidatas reales; no se inventan probabilidades objetivo a partir del piloto.
7. Reservar 2,000 preguntas de TriviaQA y 500 de NQ adicionales. Se revalidaron 8,309 de TriviaQA y 610 de NQ elegibles antes de reservar; quedan 6,309 y 110 respectivamente. Se excluyeron también los ejemplos fijos de los prompts. No hubo inferencia de test al seleccionar los IDs.
8. Ejecutar pilotos técnicos separados y elegir entre 1,008/504 preguntas RL únicamente por tiempo proyectado. El resto de decisiones se congela antes de ampliar entrenamientos y abrir el test. Todo intento cuenta, incluso fallos.

## Cierre de v2: 4 de octubre

Las 39 tareas terminaron sin errores el 3 de octubre a las 23:50 de Ciudad de México. El tiempo registrado de la fase es **35.43755 h**; el acumulado, **75.93842 h**. La reducción uniforme a 504 preguntas RL se decidió por tiempo medido antes del test: 252 actualizaciones y 12,096 generaciones por corrida. Las tres semillas y todos los controles se conservaron.

La revisión técnica comprobó fuentes, adaptadores, grid de evaluación y costos sin doble conteo. La revisión científica reconstruyó etiquetas, métricas y los diez contrastes con sus intervalos. Las cinco figuras PNG fueron inspeccionadas; queda una nota menor de presentación sobre la leyenda del gráfico de calibración. Los recibos están en `experiments/v2-20261002/artifacts/reviews/2026-10-04-completion/`.

**Resultado:** el selector calibrado mejora la utilidad de TriviaQA de 0.02418 a 0.04248 frente al filtro original calibrado (Δ=0.01830; IC99 [0.01041, 0.02593]), con mayor cobertura y menor riesgo. Las comparaciones B−A, C−B y C−D siguen inconclusas. En NQ hay mejora relativa, pero utilidad negativa (−0.01197). No se afirma equivalencia de brazos RL ni control interno aprendido del umbral.

## Cierre editorial y auditoría nueva

El usuario autorizó el 4 de octubre actualizar y condensar el manuscrito, preparar la validación humana y organizar el repositorio **sin incluir el manuscrito**. La congelación editorial documentada durante la ejecución terminó con esa autorización; fuentes científicas, configuración, locks y resultados registrados permanecen intactos. El lock conserva la huella histórica del manuscrito: no se reescribe para aparentar que el texto editorial nuevo existía antes del test. La versión anterior se archivó localmente en un área excluida de Git.

El manuscrito local incorpora v2 como comparación principal, antecedentes en apéndice, todas las semillas, costo medido y límites de transferencia. El compilador integrado verificó la fuente guardada. Su resumen de exact match no se presenta como validación humana.

La auditoría nueva de **150 unidades sobre 144 preguntas** se devolvió e importó el 6 de octubre: 100 candidatas compartidas por los filtros y 50 respuestas RL, con 75 unidades por dataset. ChatGPT preanotó las 150 unidades y una persona declaró revisar personalmente cada etiqueta. Se conservaron 104 `correct`, 19 `error`, 24 `abstain` y tres `unclear`. Se describe como anotación asistida por IA con revisión humana, no como acuerdo entre dos anotadores independientes. El formulario original vacío y la devolución se conservan por separado; no se imputan ni se mezclan etiquetas de otras fases.

La auditoría acepta como correctas 40 salidas calificadas como error por EM: 21 de TriviaQA y 19 de NQ. El conteo procede de una muestra estratificada y no estima una tasa poblacional. La sensibilidad propaga las etiquetas solo a unidades auditadas idénticas, mantiene EM en las demás y enumera ocho asignaciones coherentes de correcto/error para los tres casos indeterminados. No modifica decisiones, probabilidades, calibradores ni endpoints. Tampoco extrapola mediante pesos de inclusión.

En esa sensibilidad parcial, la diferencia selector calibrado − P(True) calibrado en TriviaQA es +0.01797, IC99 [0.01027, 0.02559]. Los tres contrastes RL siguen inconclusos. En NQ el contraste entre filtros tiene puntos entre +0.01478 y +0.01578; sus intervalos descriptivos al 95% permanecen positivos, pero no todos al 99%. La utilidad absoluta del selector es −0.00397, IC95 [−0.01522, 0.00643]: no se demuestra superioridad frente a abstenerse siempre. Los intervalos no incluyen incertidumbre de selección de la auditoría ni de anotación. [Método completo y tablas](auditoria-v2.md).

La [guía de reproducción y exportación](reproducibilidad.md) explica el índice del repositorio, las exclusiones y la conservación de los locks. El paquete ligero no contiene preguntas, predicciones individuales, pesos, anotaciones ni documentos editoriales. No se iniciarán mejoras de NQ en esta fase: quedan para una decisión posterior del usuario, con datos de desarrollo separados y nueva evaluación si se aprueban.

## Publicación del código y agregados: 7 de octubre

El repositorio [Can-LLMs-learn-to-abstain](https://github.com/jrterven/Can-LLMs-learn-to-abstain) reúne el código, la documentación, los resultados registrados y las copias de [seis agregados anónimos de auditoría](../results/v2-audit-20261006/), autorizados para publicación el 7 de octubre. El manuscrito y las anotaciones individuales permanecen fuera de GitHub. La lista cerrada de archivos y sus hashes están en `EXPORT-MANIFEST.json`; `UNBUNDLED-INPUTS.json` identifica entradas históricas omitidas.

Los paquetes locales `dist/abstention-v2-public.tar.gz` y `dist/abstention-v2-public-20261007.tar.gz` se conservan como entregas anteriores; no son enlaces de descarga del repositorio. La publicación no cambia los endpoints registrados ni convierte los agregados en una reproducción completa: recalcularlos requiere predicciones y etiquetas no distribuidas.

## Cómo retomar sin perder contexto

Las instrucciones de esta sección son para el proyecto local completo. `PROGRESO.md`, los eventos, logs, recibos individuales y varios directorios históricos no se distribuyen en el checkout público. En el proyecto local, leer este índice, después `PROGRESO.md` y el último evento de la bitácora. Una tarea con estado `running` requiere comprobar el proceso; una con `failed` conserva el intento y no debe relanzarse borrando archivos. Los sellos de fuentes, presupuesto y evaluación determinan qué cambió y cuándo. Las versiones del diagnóstico anteriores a v2 se guardan en `experiments/v2-20261002/history/`.

Comandos históricos de lectura desde la raíz del proyecto local completo (no verificaciones del checkout público):

```bash
.venv/bin/python experiments/v2-20261002/run.py status
docker logs --tail 40 threshold-v2-three-improvements
tail -n 15 experiments/v2-20261002/artifacts/events.jsonl
```

Los logs detallados de cada tarea viven en `artifacts/jobs/<id>/output.log` dentro de v2. Durante la ejecución, el supervisor actualizaba automáticamente los archivos de progreso. Las corridas v2 descritas aquí ya terminaron; publicar el repositorio no las reinicia.
