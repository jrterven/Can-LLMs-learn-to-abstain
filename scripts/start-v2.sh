#!/usr/bin/env bash
# Persistent local queue; existing TRAIN pilot must finish before new GPU work.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image_id=sha256:106bd8033f516b97e47ee39eb17c7477e516ae5329977665d25cac96ce90c10f
container_name=threshold-v2-three-improvements
cd "$project_dir"
if docker container inspect "$container_name" >/dev/null 2>&1; then
  echo 'La cola v2 ya existe; revisar su estado antes de una recuperación explícita.' >&2
  exit 1
fi
"$project_dir/.venv/bin/python" experiments/v2-20261002/run.py check
if docker container inspect threshold-v2-technical-pilot >/dev/null 2>&1; then
  pilot_code="$(docker wait threshold-v2-technical-pilot)"
  if [[ "$pilot_code" != 0 ]]; then
    echo 'El piloto falló; se conserva la evidencia y no se amplía automáticamente.' >&2
    exit 1
  fi
fi
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$project_dir:/workspace" -w /workspace \
  --entrypoint python "$image_id" experiments/v2-20261002/run.py check
exec docker run -d --name "$container_name" --init --gpus all --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -v "$project_dir:/workspace" -w /workspace \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e MPLCONFIGDIR=/workspace/.cache/matplotlib \
  -e ABSTENTION_IMAGE_ID="$image_id" \
  --entrypoint python "$image_id" experiments/v2-20261002/run.py queue
