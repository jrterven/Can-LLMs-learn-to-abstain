from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

from .config import fingerprint, matrix
from .data import verify_data
from .io import digest, file_digest, read_json, read_jsonl, utcnow, write_json, append_jsonl


def code_fingerprint():
    root = Path(__file__).parent
    return digest({p.name: file_digest(p) for p in sorted(root.glob("*.py"))})


def environment():
    packages = {}
    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "numpy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}
    if packages["torch"]:
        import torch
        info.update(cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            info.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    return info


def adapter_fingerprint(directory):
    directory = Path(directory)
    return {name: file_digest(directory / name) for name in ("adapter_model.safetensors", "adapter_config.json")}


def record_cost(workdir, operation, seconds, **metadata):
    append_jsonl(Path(workdir) / "artifacts/costs.jsonl",
                 {"time": utcnow(), "operation": operation, "seconds": seconds, **metadata})


def spent_hours(workdir):
    path = Path(workdir) / "artifacts/costs.jsonl"
    return sum(r["seconds"] for r in read_jsonl(path)) / 3600 if path.exists() else 0.0


def require_budget(config, workdir):
    if spent_hours(workdir) >= config["budget"]["total_gpu_hours"]:
        raise RuntimeError("The registered 160-hour compute budget is exhausted.")


def project_hours(config, pilots, size):
    t = config["training"]
    per_update = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] // t["num_generations"]
    seconds = 0.0
    # 2 primary thresholds on full tests; 4 diagnostic thresholds on fixed subsets.
    test_n = config["data"]["test_trivia"] + config["data"]["test_nq"]
    tau_outputs = len(config["thresholds"]["primary"]) * test_n + len(config["thresholds"]["diagnostic"]) * 2 * config["data"]["diagnostic"]
    for model in config["models"]:
        p = pilots[model]
        runs = sum(r["model"] == model for r in matrix(config))
        seconds += runs * (size / per_update) * p["seconds_per_update"]
        # Trained runs + original prompted; warmup checkpoint if activated.
        seconds += (runs + 1 + int(bool(p.get("initializer")))) * tau_outputs * p["seconds_per_eval_question"]
        # Forced candidates reused for all external filter thresholds and forced baseline.
        candidates = test_n + config["data"]["calibration"]
        seconds += candidates * (p["seconds_per_eval_question"] + p["seconds_per_confidence_question"])
        seconds += p.get("elapsed_seconds", 0)
    return seconds / 3600


def freeze(config, workdir):
    workdir = Path(workdir)
    path = workdir / "artifacts/protocol.lock.json"
    if path.exists():
        return require_lock(config, workdir)
    if (workdir / "artifacts/test-opened.json").exists():
        raise RuntimeError("Cannot register a new protocol after opening the test set.")
    manifest = verify_data(config, workdir)
    if manifest["provenance"]["kind"] != "official":
        raise ValueError("Research runs require the official data; fixture data is only for tests.")
    pilots = {}
    for model in config["models"]:
        p = read_json(workdir / "artifacts/pilots" / model / "result.json")
        if p["config_sha256"] != fingerprint(config) or p["code_sha256"] != code_fingerprint():
            raise ValueError("Pilot code/configuration is stale; rerun the pilot before registration.")
        if not p["technical_pass"] or not p["exploration_pass"]:
            raise RuntimeError(f"Pilot failed for {model}; the confirmatory study cannot start.")
        pilots[model] = p
    size = config["data"]["train"]
    hours = project_hours(config, pilots, size)
    if hours > config["budget"]["planned_gpu_hours"]:
        size = config["data"]["reduced_train"]
        hours = project_hours(config, pilots, size)
    if hours > config["budget"]["planned_gpu_hours"]:
        raise RuntimeError(f"Even the reduced design projects {hours:.1f}h; budget gate failed.")
    lock = {"created_at": utcnow(), "config": config, "config_sha256": fingerprint(config),
            "code_sha256": code_fingerprint(), "data_manifest_sha256": file_digest(workdir / "data/prepared/manifest.json"),
            "models": read_json(workdir / "artifacts/models.lock.json"),
            "pilots": pilots, "train_size": size, "projected_hours": hours,
            "matrix": list(matrix(config)), "environment": environment()}
    write_json(path, lock)
    return lock


def require_lock(config, workdir, open_test=False):
    workdir = Path(workdir)
    lock = read_json(workdir / "artifacts/protocol.lock.json")
    if lock["config_sha256"] != fingerprint(config) or lock["code_sha256"] != code_fingerprint():
        raise RuntimeError("Code or configuration changed after protocol registration.")
    if lock["data_manifest_sha256"] != file_digest(workdir / "data/prepared/manifest.json"):
        raise RuntimeError("Data manifest changed after protocol registration.")
    verify_data(config, workdir)
    if lock["models"] != read_json(workdir / "artifacts/models.lock.json"):
        raise RuntimeError("Model revisions changed after protocol registration.")
    for pilot in lock["pilots"].values():
        if pilot["initializer"] and adapter_fingerprint(pilot["initializer"]) != pilot["initializer_sha256"]:
            raise RuntimeError("Shared initialization adapter changed after registration.")
    if open_test and not (workdir / "artifacts/test-opened.json").exists():
        write_json(workdir / "artifacts/test-opened.json", {"time": utcnow(), "protocol_sha256": digest(lock)})
    return lock
