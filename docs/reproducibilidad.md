# Reproducibilidad y paquete compartible

El repositorio público es [Can-LLMs-learn-to-abstain](https://github.com/jrterven/Can-LLMs-learn-to-abstain). La entrega principal es `experiments/v2-20261002`: Qwen3-14B, cuatro brazos de RL por tres semillas y tres selectores LoRA, todos desde la misma base original. Se completaron 39 tareas. La decisión de presupuesto, tomada por tiempo medido antes de abrir el test, redujo cada RL a 504 preguntas, 252 actualizaciones y 12,096 generaciones; el selector mantuvo 2,000 preguntas/8,000 candidatas y 500 actualizaciones. Las rutas históricas se conservan en el proyecto local; su mención no implica que todas se distribuyan en GitHub.

## Tres niveles de reproducción

1. **Revisar código y agregados.** El paquete público incluye fuentes y tablas/figuras seleccionadas, con hashes. Su verificación no necesita GPU.
2. **Repetir pruebas sintéticas.** Los tests de auditoría, sensibilidad y empaquetado usan datos artificiales, sin GPU ni respuestas privadas. Se ejecutan desde el checkout público con las dependencias normales, pytest y `openpyxl==3.1.5`. La batería científica adicional necesita el entorno compatible indicado abajo.
3. **Volver a ejecutar el estudio.** Requiere descargar modelos y datasets oficiales, reconstruir las particiones por los IDs publicados y crear una nueva ejecución con registro y directorio propios. El paquete no distribuye adaptadores, predicciones o anotaciones. La verificación completa del historial de exclusiones requiere además las preguntas y respuestas de las fases anteriores, que permanecen locales.

El paquete liviano no contiene todos los insumos para recalcular resultados individuales. Sus agregados son evidencia del experimento terminado; una nueva corrida debe producir y conservar sus propios artefactos. El código histórico conserva esas exigencias, en vez de desactivar comprobaciones para simular una reproducción completa.

## Comprobaciones del checkout público

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]' openpyxl==3.1.5
.venv/bin/python -m pytest -q tests/test_public_bundle.py tests/test_audit_v2.py tests/test_audit_v2_sensitivity.py
```

Los archivos `EXPORT-MANIFEST.json` y `UNBUNDLED-INPUTS.json` identifican el contenido publicado y las entradas históricas omitidas. Para comprobar los hashes del checkout y las fuentes congeladas incluidas, sin instalar dependencias:

```bash
python3 - <<'PY_CHECK'
import json
from pathlib import Path
import runpy
bundle = runpy.run_path("scripts/public_bundle.py")
paths = set(bundle["PUBLIC_FILES"]) | bundle["GENERATED"]
payload = {path: bundle["regular_bytes"](Path.cwd(), path) for path in paths}
print(json.dumps(bundle["validate_payload"](payload), indent=2))
PY_CHECK
```

La comprobación falla si se modifica cualquier archivo publicado. No verifica entradas privadas ni reconstruye predicciones ausentes.

## Entorno fijado

- Plataforma verificada: NVIDIA GB10, ARM64, Ubuntu 24.04 y aproximadamente 128 GB de memoria unificada.
- Base: `nvcr.io/nvidia/pytorch:25.11-py3`, digest OCI `sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21`, en `Dockerfile`.
- Imagen local usada por v2: `sha256:106bd8033f516b97e47ee39eb17c7477e516ae5329977665d25cac96ce90c10f`. Es un identificador de imagen construida localmente, no una dirección pública para descargarla.
- Versiones principales: PyTorch NVIDIA `2.10.0a0+b558c986e8.nv25.11`, Transformers 4.57.1, TRL 0.24.0, PEFT 0.17.1 y datasets 4.2.0. Otras dependencias están fijadas en `pyproject.toml` y en las verificaciones de entorno del selector.

```bash
docker build -t threshold-abstention:rebuild .
```

Una reconstrucción puede tener otro ID aunque use la misma receta. No se debe presentar ese ID como el original ni modificar sus locks. Registre las diferencias de entorno en la nueva ejecución. La igualdad de paquetes tampoco promete resultados bit a bit entre plataformas o versiones de CUDA.

Pruebas científicas en una imagen compatible, sin GPU:

```bash
docker run --rm --network none --cpus 2 \
  --user "$(id -u):$(id -g)" -v "$PWD:/workspace" -w /workspace \
  -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  --entrypoint python threshold-abstention:rebuild -m pytest -q \
  experiments/v2-20261002/tests_prepare_data.py \
  experiments/v2-20261002/tests_training_v2.py \
  experiments/v2-20261002/tests_selector_v2.py \
  experiments/v2-20261002/tests_evaluate_v2.py \
  experiments/v2-20261002/tests_analysis_v2.py \
  experiments/v2-20261002/tests_run.py
```

Los guards de versiones o de código interno pueden rechazar otra reconstrucción. Ese fallo informa una diferencia que debe investigarse; no autoriza a reemplazar hashes históricos. La auditoría XLSX tiene una dependencia CPU adicional, `openpyxl==3.1.5`; véase [su guía](auditoria-v2.md).

## Modelos y particiones

La base v2 es `Qwen/Qwen3-14B`, revisión `40c069824f4251a91eefaf281ebe4c544efd3e18`. Al responder no recibe documentos, aliases ni búsquedas. Los pesos deben obtenerse del proveedor respetando su licencia.

TriviaQA usa `mandarjoshi/trivia_qa`, configuración `unfiltered.nocontext`, revisión `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`. NQ-Open usa el Original Dev oficial, commit `fb26a3073b1fe636c97302890a27b491d6530130`. Los manifiestos conservan IDs en orden, tamaños, hashes, subconjuntos diagnósticos y procedencia. TRAIN, DEV y calibración provienen de las particiones iniciales separadas; el test v2 agrega 2,000 TriviaQA y 500 NQ sin solapar los conjuntos previos documentados.

`train.jsonl` registra el panel de 1,008 preguntas. La corrida usó su subconjunto anidado `train_reduced.jsonl`, de 504, según `artifacts/budget-lock.json`. No debe confundirse con `train_full.jsonl`, las 2,000 preguntas del selector. No se distribuye ningún archivo de datos JSONL.

## Locks históricos y manuscrito local

`source-lock.json` se conserva byte por byte. Al crearlo se incluyó una huella de `paper/manuscript.tex`, aunque ese archivo editorial no es una entrada del entrenamiento. El manuscrito sigue local; Git, Docker y el exportador lo excluyen. Una revisión editorial posterior puede causar que el antiguo `run.py check` rechace ese archivo.

```bash
python3 scripts/public_bundle.py check-local
```

**Este comando está destinado al proyecto local completo, no al checkout público.** Necesita los JSONL y otras entradas omitidas; su ausencia causa un error esperado. Comprueba todas las demás entradas de `source-lock.json` y muestra la diferencia editorial por separado, con su hash histórico y actual si existe. No declara que el historial entero sea idéntico ni modifica el lock. Un cambio en código, configuración o datos científicos sí provoca error. Los resultados y sus recibos conservan la procedencia original.

En el paquete público algunas entradas científicas están deliberadamente ausentes, especialmente datos. `UNBUNDLED-INPUTS.json` las enumera, diferenciándolas del archivo editorial. `verify` comprueba la integridad del paquete y de las fuentes congeladas incluidas, no la disponibilidad de esas entradas omitidas.

## Exportar, inspeccionar y compartir

GitHub contiene la selección de código, documentación y agregados descrita aquí, incluidos seis agregados anónimos del análisis de auditoría del **6 de octubre**. El archivo `dist/abstention-v2-public-20261007.tar.gz` es una entrega local archivada anterior a la publicación; no está incluido en GitHub ni se presenta como un release descargable. Si dispone de ese archivo, puede inspeccionarlo así:

```bash
python3 scripts/public_bundle.py verify dist/abstention-v2-public-20261007.tar.gz
tar -tzf dist/abstention-v2-public-20261007.tar.gz
```

El archivo local del 4 de octubre, `dist/abstention-v2-public.tar.gz`, no incluye esta auditoría y se conserva intacto. Para generar otra copia desde el **proyecto local completo**, utilice un nombre nuevo: el exportador exige los insumos de `check-local` y rechaza sobrescribir un archivo existente. Estos comandos no prometen reconstruir la entrega desde un checkout que omite esos insumos.

```bash
python3 scripts/public_bundle.py verify dist/abstention-v2-public.tar.gz
python3 scripts/public_bundle.py export --output dist/abstention-v2-public-copia.tar.gz
```

La selección está escrita explícitamente en el exportador. Incluye código/configuración/documentación, manifiestos sin respuestas o etiquetas individuales, locks de procedencia y métricas/figuras agregadas v2. Los archivos se copian sin editar su contenido; las copias no reemplazan los originales. Los miembros del tar se ordenan y sus metadatos se normalizan para generar el mismo SHA256 si los insumos no cambian.

La actualización incluye `analysis/audit_v2_sensitivity.py`, sus pruebas y los [seis agregados seleccionados](../results/v2-audit-20261006/). Las copias en `results/v2-audit-20261006/` son visibles a Git y su `provenance.json` enlaza hashes de origen y código, sin identidad del anotador. Los originales permanecen en `artifacts/audit-v2-20261004/analysis-20261006/public/`, dentro del área privada excluida de Git. No se altera ningún resultado exact-match, calibrador ni lock histórico. El verificador admite tanto el inventario original como el ampliado con la auditoría.

La auditoría se recibió como preanotación de ChatGPT revisada personalmente por una persona. Sus agregados permiten inspeccionar el diagnóstico y la sensibilidad, pero **recalcularlos requiere las etiquetas y claves privadas**. El código incluido permite repetir sus pruebas sintéticas en CPU; no distribuye esas entradas ni sustituye la falta de etiquetas con respuestas inventadas.

Se rechazan enlaces simbólicos, rutas fuera del proyecto, miembros inesperados y nombres asociados a manuscritos, fuentes LaTeX, pesos, caches, respuestas o anotaciones. No se exportan notebooks con salidas, hojas XLSX/CSV de auditoría, claves privadas, JSONL ni `artifacts/audit-v2-20261004/`. El exportador solo crea archivos locales: no sube contenido ni configura remotos. La publicación en GitHub se realiza por separado sobre la misma selección explícita de archivos.

Antes de compartir, ejecute `verify` y conserve `EXPORT-MANIFEST.json`, `UNBUNDLED-INPUTS.json` y el hash mostrado. Una exportación posterior con código o documentación cambiados tendrá otro hash; no debe sustituir la identificación de una versión anterior.

## Archivo histórico

Los directorios citados en esta sección son antecedentes locales, no entregables públicos completos. El protocolo congelado `docs/tres-mejoras-v2.md` conserva enlaces a `docs/rl-mejoras-y-truthrl.md` y `docs/resultados-para-paper.md`, dos antecedentes locales omitidos; esos enlaces no se resolverán en GitHub. Se preserva el protocolo byte por byte para no alterar su sello histórico. El estudio inicial de Qwen3-1.7B y SmolLM2-1.7B, la extensión inicial a 14B y la evaluación de 28 variantes permanecen en sus directorios originales. [El registro](registro-experimentos.md) explica la secuencia. No se incorporan automáticamente al paquete v2 sus datos, predicciones, anotaciones ni todos sus análisis auxiliares. La preparación anterior de datos sí se conserva como dependencia de código de v2.
