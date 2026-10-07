FROM nvcr.io/nvidia/pytorch:25.11-py3@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21
WORKDIR /workspace
ENV HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src
COPY pyproject.toml /opt/abstention/pyproject.toml
COPY src /opt/abstention/src
RUN python -m pip install --no-build-isolation '/opt/abstention[train,test]' && \
    python -m pip freeze > /opt/abstention-environment.txt
ENTRYPOINT ["python", "-m", "abstention.cli"]
