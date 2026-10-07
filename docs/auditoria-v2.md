# Auditoría humana de 150 respuestas v2

Esta auditoría comprueba si el evaluador automático acepta respuestas falsas o rechaza respuestas correctas por cómo están escritas. Se enfoca en el selector aprendido, que decide sobre las mismas candidatas que el filtro original, e incluye los cuatro brazos de RL.

La devolución de las **150 unidades, sobre 144 preguntas**, se recibió e importó el **6 de octubre de 2026**. ChatGPT produjo una preanotación de las 150 unidades y una persona declaró haber revisado personalmente cada etiqueta. Se describe como **anotación asistida por IA con revisión humana**; no son dos anotadores independientes ni permite estimar acuerdo entre anotadores.

Se conservaron 104 etiquetas `correct`, 19 `error`, 24 `abstain` y tres `unclear`. Los tres casos indeterminados no se adjudican automáticamente. El formulario original vacío permanece en `artifacts/audit-v2-20261004/auditoria-v2-150-respuestas.xlsx`; la devolución y la importación se archivan separadamente en el área privada. No se reutilizaron etiquetas de la auditoría histórica.

## Procedimiento de anotación conservado

1. Abre **Auditoria** y completa **human_outcome**: `correct`, `error`, `abstain` o `unclear`.
2. Juzga el **significado**. Acepta paráfrasis y nombres equivalentes. Los aliases son una guía que puede estar incompleta o equivocada. Usa `abstain` para una abstención en lenguaje natural; no hace falta que diga exactamente `IDK`.
3. Si falta contexto, hay ambigüedad o no puedes resolver el caso, elige `unclear` y explica la duda en **notes**. No adivines. Si consultas fuentes, añade sus enlaces en **evidence**.
4. **format_note es opcional**. `truncated` viene del registro de generación: no convierte automáticamente tu etiqueta semántica en `error`. Evalúa el texto visible y registra el formato por separado si ayuda. Una respuesta vacía es `error`.
5. Completa **Procedencia**, incluyendo si usaste IA y cómo. Revisa personalmente cada etiqueta y declara con precisión lo que hiciste.
6. Guarda una **copia XLSX** y devuélvela. Conserva IDs y columnas A–E; puedes ordenar las filas. No necesitas abrir claves ni archivos JSON.

El formulario omitía modelos, condiciones y etiquetas automáticas. La instrucción era mantener `private-key.json`, `sampling.json` y los resultados por condición cerrados hasta terminar. Este cegamiento de la presentación no convierte la preanotación de ChatGPT y su revisión en evaluaciones independientes.

## Selección y límites

La regla se definió **después de conocer los resultados agregados v2 y antes de obtener estas etiquetas**. Es una auditoría diagnóstica posterior al experimento, no una nueva prueba confirmatoria predefinida.

Hay **75 unidades de TriviaQA y 75 de NQ**. Cada dataset aporta 50 candidatas compartidas y 25 respuestas de RL. De las 50 candidatas, 35 presentan alguna discrepancia entre selector calibrado y filtro original calibrado en las seis combinaciones de semilla y umbral principal; 15 presentan concordancia. Se cubren ambas direcciones de discrepancia y se estratifica por etiqueta automática, sin elegir solamente casos favorables al selector. Los ejemplos RL se estratifican por brazo y etiqueta. Sus unidades excluyen las del banco completo de candidatas originales, para mantener disjuntos ambos marcos.

La unidad es **pregunta + texto exacto + truncamiento**. Son 150 unidades distintas, aunque alguna pregunta puede repetirse con otra respuesta. Un representante RL se fija por hash antes de asignar el brazo de cada unidad. Dentro de cada bloque se distribuyen cupos mediante recorrido circular por estratos ordenados por hash; luego se toman los menores hashes de unidad, con semilla `20261004`. Si un bloque no tiene capacidad, la exportación se detiene en vez de retocar el criterio.

La clave privada conserva todas las ocurrencias pertinentes, tamaños `N_h` y `n_h`, fracción nominal de inclusión `n_h/N_h` y peso `N_h/n_h`. Corresponden a estos marcos de unidades. **El desacuerdo bruto no estima una tasa poblacional de error, falsos negativos ni utilidad corregida.** Los pesos permiten una sensibilidad posterior con tratamiento explícito de `unclear`, duplicados y selección; no autorizan sustituir silenciosamente los resultados exact-match.

Se mantienen separadas las diferencias de formato y de significado. El importador respeta las etiquetas humanas; no las adjudica con IA ni modifica predicciones, recompensas, calibradores, pesos o endpoints sellados. Una anotación individual asistida por IA debe describirse así; no equivale al acuerdo de dos anotadores independientes.

## Resultado y sensibilidad parcial: 6 de octubre

Se encontraron **40 discrepancias de error automático a respuesta aceptada como correcta por la anotación**: 21 en TriviaQA y 19 en NQ. Los tres `unclear` corresponden a NQ. Estos son conteos de una muestra estratificada deliberada, **no una tasa poblacional de falsos negativos** ni una estimación de exactitud semántica de todo el test.

El análisis posterior propaga una etiqueta solamente a las ocurrencias de la misma **pregunta, texto exacto y truncamiento**. Cada unidad no auditada conserva su etiqueta automática. Las decisiones de responder/abstenerse, las probabilidades, los calibradores y las predicciones permanecen fijos; no se reentrena ni se relaja el contrato de salida. En esta devolución no hubo conflictos entre la anotación semántica y las reglas estrictas de formato.

La columna `partial_em_unclear` conserva EM en los tres casos indeterminados. También se enumeran las ocho asignaciones posibles de correcto/error para esas tres unidades: cada asignación se aplica de manera coherente a todas las apariciones de una unidad. Son escenarios hipotéticos, no adjudicaciones. Los rangos solo describen esos tres casos, no acotan los errores semánticos de las respuestas no auditadas. **No se utiliza un estimador ponderado por la probabilidad de inclusión.**

Las comparaciones promedian los umbrales 0.65 y 0.85 y conservan todas las semillas:

| Comparación | Exact match original | Sensibilidad parcial |
|---|---:|---:|
| Selector calibrado − P(True) calibrado, TriviaQA | +0.01830 | +0.01797; IC99 [0.01027, 0.02559] |
| Selector calibrado − P(True) calibrado, NQ | +0.01878 | Puntos entre +0.01478 y +0.01578, según los casos indeterminados |
| Utilidad absoluta del selector calibrado, NQ | −0.01197 | −0.00397; IC95 [−0.01522, 0.00643] |

Los intervalos de los tres contrastes RL siguen incluyendo cero. En NQ, los escenarios de la comparación entre filtros conservan intervalos descriptivos al 95% positivos, pero **no al 99%**: la envolvente de extremos al 99% es [−0.00113, 0.03100]. Esa envolvente no es un nuevo intervalo de confianza calibrado. Tampoco se demuestra que el selector tenga utilidad positiva frente a la política de siempre abstenerse, cuya utilidad es cero.

Se utiliza el mismo bootstrap pareado por pregunta y semilla del análisis original, con ambos umbrales juntos, 10,000 repeticiones y semilla 20261002. El 99% mantiene el nivel nominal de Bonferroni para los cinco contrastes originales; no convierte este análisis posterior en una nueva prueba confirmatoria. Los intervalos están **condicionados a las etiquetas y a la selección de auditoría**, además del banco TRAIN y los calibradores ajustados: no incorporan la incertidumbre de muestreo de la auditoría ni del juicio humano.

Las seis tablas/resumen anónimos están incluidos en [results/v2-audit-20261006](../results/v2-audit-20261006/), y en el repositorio público, con autorización expresa del **7 de octubre**. Son copias de los agregados del análisis del día 6, vinculadas a sus fuentes mediante hashes; la entrega local archivada correspondiente es `dist/abstention-v2-public-20261007.tar.gz`, no incluida como descarga de GitHub. Los resultados exact-match originales permanecen como endpoints registrados; estos archivos son una sensibilidad separada, no una sustitución silenciosa de esos resultados.

## Herramientas reproducibles, solo CPU

Las pruebas sintéticas funcionan en el checkout público, sin archivos privados. Prepare el entorno como indica la [guía de reproducibilidad](reproducibilidad.md) y ejecute:

```bash
.venv/bin/python -m pytest tests/test_audit_v2.py tests/test_audit_v2_sensitivity.py -q
```

Los comandos siguientes documentan el procedimiento usado en el **proyecto local completo**. No funcionan solo con el checkout público: necesitan predicciones, recibos y claves omitidos.

```bash
.venv/bin/python analysis/audit_v2.py export
```

La exportación exige los 13 recibos directos, los tres recibos del selector, el análisis final y su revisión independiente. Comprueba hashes, IDs, etiquetas automáticas y candidatas compartidas. No sobrescribe una auditoría existente. El XLSX conserva literalmente textos que empiecen con `=`; el CSV de respaldo añade un apóstrofo protector ante caracteres que Excel puede interpretar como fórmulas.

Para validar y después importar una devolución, conservando los originales:

```bash
.venv/bin/python analysis/audit_v2.py validate --workbook /ruta/auditoria-respondida.xlsx
.venv/bin/python analysis/audit_v2.py import --workbook /ruta/auditoria-respondida.xlsx
```

Se exigen 150 IDs únicos conocidos, textos y referencias intactos, etiquetas válidas y procedencia explícita. `unclear` requiere explicación. Se rechazan fórmulas ejecutables, hojas adicionales y contenido oculto. La importación se archiva por hash en `artifacts/audit-v2-20261004/submissions/`, sin sobrescribir ni publicar.

Recalcular el análisis real con `analysis/audit_v2_sensitivity.py` requiere la importación y las claves privadas conservadas localmente. El comando no se vuelve a ejecutar sobre el directorio final para sobrescribirlo.

El código, esta guía y las copias seleccionadas de agregados forman parte de la publicación autorizada del 7 de octubre. **Libros, claves, etiquetas individuales, preguntas, respuestas, notas, evidencias e identificación del anotador permanecen privados.** El directorio privado no se exporta ni se publica en GitHub.
