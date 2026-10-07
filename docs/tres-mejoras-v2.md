# Tres mejoras, con comparaciones que permitan aprender del resultado

Fase `v2-20261002`, iniciada el 2 de octubre de 2026 por instrucción del usuario. Este es el protocolo de implementación; el estado real, los tiempos y las incidencias se registran en `experiments/v2-20261002/artifacts/`. La ejecución es continua, de día y de noche. No se incorporan resultados nuevos al manuscrito todavía.

## Qué queremos mejorar

El estudio anterior mostró que penalizar errores supera al entrenamiento binario, pero no acreditó que el entrenamiento condicionado supere al costo fijo o al filtro calibrado. Además, la señal que distingue responder de abstenerse aparece en pocos grupos. La calibración logística mejora la escala de confianza, pero no mejora por sí misma la ordenación de candidatas cuando es una transformación creciente.

Elegimos **tres intervenciones** con mecanismos distintos y controles explícitos. Son hipótesis con fundamento, no una garantía de resultados positivos. Un estudio publicable exige también explicar un resultado negativo y contabilizar su costo.

| Mejora | Cambio concreto | Comparación que la identifica | Resultado que buscamos |
|---|---|---|---|
| 1. Crédito de RL con grupos mayores | Ocho generaciones por prompt: una referencia leave-one-out de ocho frente a dos referencias de cuatro | B−A, mismo número de preguntas, generaciones y actualizaciones | Mayor utilidad, sin lograrla únicamente dejando de responder |
| 2. Aprender el efecto del umbral | Cada pregunta aparece con 0.60, 0.75 y 0.90 dentro de una misma actualización | C−B: tres umbrales frente a tres repeticiones de un solo umbral; C−D: recompensa condicionada frente a costo fijo | Uso útil de umbrales nuevos y ventaja frente al control fijo |
| 3. Aprender a seleccionar candidatas | LoRA que predice si una respuesta del generador original es correcta; pérdida binaria sobre True/False | Selector frente a P(True) calibrado y una alternativa logística con las mismas etiquetas de entrenamiento | Mejor discriminación y utilidad; calibración que transfiera a NQ a costo justificable |

Leave-one-out tiene antecedentes en [REINFORCE Leave-One-Out](https://arxiv.org/abs/2402.14740), y la estimación P(True) en [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221). No reclamaremos esas ideas como algoritmos nuevos. Nuestra posible contribución es la comparación controlada de mecanismos, umbrales no vistos, transferencia y costo en una sola computadora.

## Brazos de RL y presupuesto idéntico

Todos parten de **Qwen3-14B original**, revisión `40c069824f4251a91eefaf281ebe4c544efd3e18`, sin SFT ni adaptadores anteriores. Usan semillas 17, 29 y 43 emparejadas: doce entrenamientos.

| ID estable | Agrupación de las ocho salidas | Tres exposiciones por pregunta | Recompensa |
|---|---|---|---|
| `a_g4_single_tau` | Dos grupos de cuatro | Mismo τ asignado, repetido tres veces | Condicionada al τ solicitado |
| `b_g8_single_tau` | Un grupo de ocho | Mismo τ asignado, repetido tres veces | Condicionada al τ solicitado |
| `c_g8_paired_tau` | Un grupo de ocho | Una vez cada τ=0.60/0.75/0.90 | Condicionada al τ solicitado |
| `d_g8_paired_fixed075` | Un grupo de ocho | Una vez cada τ=0.60/0.75/0.90 | Siempre costo 0.75 |

Seleccionamos 1,008 preguntas de las 2,000 de TRAIN por hash, sin utilizar resultados. En A/B, 336 preguntas reciben cada umbral; la asignación y el orden son iguales entre los brazos emparejados. C/D usan las mismas preguntas y orden. Cada actualización contiene dos preguntas, con sus tres exposiciones y ocho generaciones por exposición: **48 generaciones**. Cada corrida realiza **504 actualizaciones y 24,192 generaciones**. Los tres slots conservan identidades distintas aunque su texto se repita; no se mezclan las 24 salidas de una pregunta en un solo grupo.

La recompensa es R=c−τa: c indica acierto exact-match y a indica respuesta emitida. La abstención exacta `IDK` vale cero; vacías o truncadas son errores. D siempre sustituye τ por 0.75 en la recompensa, pero recibe los mismos prompts que C. La media del nuevo conjunto de umbrales es 0.75; evita confundir variar el costo con cambiar su promedio.

Igualar ese promedio no iguala el costo realmente pagado: este depende de cuáles preguntas se responden y de sus resultados. La referencia analítica **siempre-IDK tiene utilidad cero y cobertura cero**, con riesgo selectivo indefinido; se mostrará junto a las comparaciones entre métodos.

Todos usan ventajas **LOO** `R_i − promedio(R_j, j≠i)`, sin dividir por desviación estándar. La reducción de pérdida se conserva como `dr_grpo` de TRL 0.24.0: mismo denominador por token y acumulación en ambos tamaños de grupo. No se mezclan grupos con preguntas, umbrales o exposiciones diferentes. Cada generación conserva recompensa, ventaja, IDs y grupo para poder recalcular el crédito.

Receta común: BF16, LoRA r=16/alpha=32 en capas lineales, dropout=0, LR=1e−5 constante, KL=0.04, temperatura=0.8, prompts≤512 tokens, respuestas≤32, microbatch=4, acumulación=12, una sola pasada por el panel de exposiciones. No se descartan ni remuestrean los grupos con recompensa constante. Cambiarán realmente las políticas durante RL: generaciones emparejadas significa mismo presupuesto y semillas, no textos idénticos después de entrenar.

Estas comparaciones identifican diferencias **dentro de v2**. Frente a v1 cambian simultáneamente LOO, exposición, umbrales y presupuesto; v1 es antecedente descriptivo, no una ablación causal de una sola variable.

C−B identifica el calendario completo de exposición pregunta×umbral: también cambia el balance de umbrales dentro de cada actualización y su ruido. Por sí solo no aísla una supuesta comprensión del umbral. No incluimos un quinto brazo G4 con exposición pareada, por lo que tampoco estimamos toda la interacción factorial entre ambas mejoras. El presupuesto igualado es el número de generaciones y actualizaciones; los tokens y segundos realmente consumidos pueden diferir y se medirán.

## Selector: aprender corrección, después calibrar

El generador permanece congelado. Para cada una de las 2,000 preguntas TRAIN originales produce una candidata greedy y tres muestreadas a temperatura 0.8: **8,000 candidatas**. Todas se conservan, incluso errores, abstenciones y formatos inválidos. No se equilibran clases ni se seleccionan preguntas por sus resultados. Cada pregunta tiene igual peso total. Las 16 muestras del piloto anterior no se convierten en probabilidades verdaderas ni en etiquetas de confianza.

El selector recibe exclusivamente pregunta y candidata, mediante el prompt P(True) ya existente. No recibe aliases, etiqueta, umbral ni condición experimental. Aprende con BCE sobre `logit(True)−logit(False)` y etiqueta de coincidencia exacta. Los dos tokens son individuales en el tokenizer fijado. El modelo base y su cabeza permanecen congelados; solo cambia LoRA. Tres semillas, LR=1e−5, una época, microbatch=4 y acumulación=4; 500 actualizaciones por corrida. No se elige época ni semilla por resultados.

Cada selector se calibra con una regresión logística C=1 sobre las **500 preguntas de calibración**, distintas de TRAIN y DEV. La candidata greedy de cada pregunta se genera una vez y se comparte entre todos los filtros. También se comparan P(True) original sin calibrar, P(True) original calibrado y una regresión sobre el P(True) original entrenada con las mismas 8,000 etiquetas y recalibrada con las 500. Esta última tiene capacidad limitada: dos transformaciones afines de un único logit siguen siendo afines; es un control con las mismas etiquetas, no una alternativa expresiva equivalente al LoRA.

Ese control iguala las etiquetas disponibles; **no** iguala capacidad del modelo ni tiempo de optimización. Las tres semillas del selector comparten un único banco de candidatas TRAIN: la incertidumbre entre semillas es condicional a ese banco y a la partición de entrenamiento, además de a los calibradores ajustados. No incluye regenerar otro banco de candidatas.

La decisión es responder si p>τ; si no, `IDK`. Una candidata vacía, truncada o ya abstendida se rechaza en todos los filtros. Todas se puntúan para analizar calibración, sin excluir las finalmente rechazadas. Informaremos por separado las candidatas sustantivas válidas. Exact match es un proxy: esto aprende corrección según ese verificador, no una garantía de verdad semántica. El juez semántico fallido anterior no se utiliza como recompensa ni etiqueta.

## Evaluación reservada y criterios

Se reservan por hash **2,000 preguntas nuevas de TriviaQA y 500 de NQ-Open**; se excluyen IDs y textos normalizados de todos los conjuntos y paneles anteriores. Las 500 de NQ reflejan el conjunto todavía disponible; no se inventará una muestra independiente de mayor tamaño. Los manifiestos contienen fuentes, exclusiones, IDs y hashes. Ser nuevas para este estudio no implica ausencia de contaminación del preentrenamiento.

Los umbrales principales son **0.65 y 0.85**, ausentes del entrenamiento. Ambos se evalúan sobre todos los ejemplos. En 200 preguntas prefijadas por dataset se diagnostican 0.60/0.75/0.90 y se explora 0.95. Las pruebas de guardado/recarga y los pilotos solo usan TRAIN. El presupuesto y las implementaciones se congelan antes de generar resultados de test; no se seleccionan recetas, checkpoints ni semillas mirando estos resultados.

Medidas: utilidad media conjunta de los dos umbrales, cobertura, riesgo selectivo, aciertos/errores totales, frecuencia de IDK, vacías/truncadas y exactitud bajo respuesta obligatoria. Riesgo con cobertura cero es indefinido. Para filtros: Brier, log loss, ROC AUC, ECE con diez intervalos fijos, curvas de riesgo-cobertura con empates juntos y costo completo. La calibración en TriviaQA se aplica a NQ sin reajuste.

Contrastes predefinidos en TriviaQA: **B−A**, **C−B**, **C−D**, selector calibrado−filtro original calibrado y selector calibrado−control logístico con etiquetas TRAIN. Se repetirán descriptivamente en NQ. IC95 mediante 10,000 bootstraps cruzados por pregunta y semillas emparejadas, manteniendo ambos umbrales juntos; el baseline único se comparte entre semillas. Se reportarán todas las semillas, efectos e intervalos. Además se mostrarán IC99 por contraste como ajuste Bonferroni para la familia de cinco comparaciones (cobertura familiar nominal 95%, sujeta a la aproximación bootstrap). Los IC95 ordinarios son descriptivos y condicionados a los calibradores ajustados; no contienen incertidumbre de haber elegido otra muestra de calibración. Solo tres semillas limitan la precisión de la incertidumbre entre entrenamientos.

Mayor utilidad con menor cobertura no demuestra por sí misma mejor selección. Para atribuir control aprendido necesitamos la comparación con D, respuesta al umbral en las mismas preguntas y ausencia de colapso. Para justificar el selector necesitamos discriminar mejor o producir mayor utilidad frente a alternativas baratas; reducir ECE solamente no basta. Ninguna mejora se declarará por elegir el mejor seed, dataset o umbral. Un efecto pequeño/incierto o una transferencia fallida se reportará como tal.

Un intervalo que contiene cero no demuestra equivalencia ni descarta el mecanismo: informaremos el rango de efectos compatible con estos datos y la precisión limitada, especialmente en NQ y con tres semillas.

## Pilotos, cómputo y reglas de parada

Antes de las corridas completas: 32 actualizaciones reales de RL y 16 del selector, gradientes finitos, reserva mínima de 16 GiB, guardar/recargar adaptadores y comprobar reproducción. Son pilotos técnicos separados: se reinician las corridas definitivas desde el modelo original. Su costo y los fallos se cuentan.

Hay aproximadamente **40.501 horas registradas** antes de esta fase. La fase v2 tiene proyección objetivo≤80 h y límite duro de 110 h, dentro del límite global original de 160 h. La proyección incorpora carga, generación de candidatas, los doce RL, los tres selectores, evaluación y margen de 25%. Si no cabe, antes del test se reduce uniformemente RL a **504 preguntas, 252 actualizaciones y 12,096 generaciones por corrida**; se mantienen controles, semillas y selector. Si aun así no cabe, se detiene antes de ampliar entrenamientos y se registra el bloqueo presupuestal. No se recorta por resultados.

El supervisor ejecuta una tarea GPU a la vez, registra inicio/fin/costo incluso ante errores, aplica plazos y no reintenta silenciosamente. Un fallo técnico conserva sus archivos y requiere recuperación explícita y documentada; no se borra para aparentar una corrida limpia. El estado legible y el registro de eventos se actualizan automáticamente, sin necesitar sondeo continuo desde el chat.

## Hilo experimental y archivos

Los datos, código, adaptadores y predicciones anteriores permanecen como evidencia histórica. Todo lo nuevo vive en `experiments/v2-20261002/`; `src/abstention/` y `paper/manuscript.tex` permanecen congelados.

| Archivo | Propósito |
|---|---|
| `config.yaml` | Receta y límites explícitos |
| `data/prepared/manifest.json` | Selección, exclusiones y hashes de datos |
| `contract.json` | Rutas/modelo/datos acordados entre módulos |
| `artifacts/source-lock.json` | Código/configuración y fuentes congeladas |
| `artifacts/budget-lock.json` | Mediciones piloto y decisión uniforme de tamaño |
| `artifacts/evaluation-lock.json` | Adaptadores finales y apertura de test |
| `artifacts/events.jsonl` | Bitácora cronológica, sin sobrescribir eventos |
| `artifacts/status.json` y `PROGRESO.md` | Estado actual y cómo consultarlo |
| `artifacts/jobs/` | Logs y costos por intento |
| `artifacts/runs/` | Adaptadores, rollouts, gradientes y resultados de RL |
| `artifacts/selector/` | Candidatas, selectores y calibradores |
| `artifacts/analysis/` | Tablas, contrastes y figuras de la nueva fase |

El diagnóstico detallado anterior queda en [rl-mejoras-y-truthrl.md](rl-mejoras-y-truthrl.md); los resultados concluidos en [resultados-para-paper.md](resultados-para-paper.md). Este documento reduce las opciones a un programa ejecutable y conserva la separación entre hechos, hipótesis y resultados pendientes.
