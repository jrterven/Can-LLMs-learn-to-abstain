# V2: resultados reservados completos

Resultados descriptivos de todos los brazos y semillas registrados. Este archivo no declara automáticamente éxito ni modifica el manuscrito.

| Dataset | Contraste de utilidad | Efecto | IC95 descriptivo | IC99 Bonferroni |
|---|---|---:|---|---|
| test_trivia | b_g8_single_tau − a_g4_single_tau | 0.008271 | [-0.017789, 0.029325] | [-0.023301, 0.034275] |
| test_trivia | c_g8_paired_tau − b_g8_single_tau | 0.010271 | [-0.018721, 0.047566] | [-0.024450, 0.057188] |
| test_trivia | c_g8_paired_tau − d_g8_paired_fixed075 | 0.035775 | [-0.015663, 0.076484] | [-0.022400, 0.084238] |
| test_trivia | critic_calibrated − original_calibrated | 0.018304 | [0.012350, 0.024213] | [0.010412, 0.025925] |
| test_trivia | critic_calibrated − train_logistic_recalibrated | 0.018342 | [0.012379, 0.024271] | [0.010412, 0.026017] |
| test_nq | b_g8_single_tau − a_g4_single_tau | 0.002533 | [-0.029801, 0.040801] | [-0.037750, 0.052951] |
| test_nq | c_g8_paired_tau − b_g8_single_tau | 0.013067 | [-0.030100, 0.066751] | [-0.039250, 0.083901] |
| test_nq | c_g8_paired_tau − d_g8_paired_fixed075 | 0.054367 | [-0.021652, 0.117754] | [-0.032751, 0.131250] |
| test_nq | critic_calibrated − original_calibrated | 0.018783 | [0.006050, 0.030967] | [0.001817, 0.034883] |
| test_nq | critic_calibrated − train_logistic_recalibrated | 0.018783 | [0.006050, 0.030967] | [0.001817, 0.034883] |

La familia principal son los cinco contrastes en TriviaQA; NQ es transferencia descriptiva. Se promedian τ=0.65/0.85 por pregunta antes de remuestrear preguntas y semillas emparejadas. Los filtros deterministas son un único baseline compartido. Los IC99 tienen cobertura familiar nominal 95% por Bonferroni, sujeta a la aproximación bootstrap; no se calcularon p-valores. Tres semillas limitan la inferencia entre entrenamientos; no se remuestreó la muestra de calibración.

Las tablas metrics.csv y joint-primary.csv distinguen preguntas únicas, semillas y filas de predicción. Riesgo es errores/respuestas, indefinido sin cobertura. Menor cobertura no acredita por sí sola mejor selección. forced.csv y threshold-response.csv registran retención, IDK persistente y cambios ante el umbral; los seis puntos diagnósticos usan las mismas 200 preguntas por dataset.

calibration.csv separa todas las candidatas de las sustantivas válidas. Una menor ECE/Brier no garantiza mejor ordenación ni utilidad. risk-coverage.csv ordena candidatas compartidas y conserva empates completos: no describe las políticas de prompting. La calibración de TriviaQA se aplica a NQ sin ajustes. Exact match sigue siendo un proxy, no verdad semántica auditada.

Costo cerrado al iniciar este análisis: 35.423518 horas de tareas v2. Incluye pilotos/fallos/carga registrados por el supervisor; excluye el análisis CPU en curso y tareas posteriores. No se suman otra vez los ledgers anidados. La cifra final se obtiene del ledger autoritativo cuando cierre el job de análisis.

La incertidumbre del selector está condicionada a un único banco de candidatas TRAIN compartido y a la partición fija; no representa regenerar ese banco. El control logístico iguala etiquetas, no capacidad paramétrica ni cómputo. C−B cambia la exposición y la mezcla de umbrales dentro del batch; no identifica por sí solo comprensión del umbral. No existe un brazo G4 con exposición emparejada para estimar esa interacción factorial. Los diagnósticos de actividad de recompensa no equivalen a medir señal/ruido de gradientes ni establecen mediación causal.

Las figuras requieren inspección visual y las conclusiones requieren revisión científica. No se han añadido etiquetas humanas ni actualizado el manuscrito.
