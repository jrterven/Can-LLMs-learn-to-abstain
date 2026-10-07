from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import load_split
from .io import append_jsonl, digest, read_json, read_jsonl, write_json, write_jsonl
from .models import Engine
from .prompts import messages
from .protocol import adapter_fingerprint, record_cost, require_budget, require_lock
from .scoring import grade


def fit_calibrator(rows):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Calibration contains duplicate questions.")
    y = np.array([r["outcome"] == "correct" for r in rows], dtype=int)
    if len(set(y)) != 2:
        raise ValueError("Calibration requires both correct and incorrect candidate answers.")
    x = np.array([[r["confidence_logit"]] for r in rows])
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0).fit(x, y)
    return {"coefficient": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0]),
            "n": len(rows), "ids": [r["id"] for r in rows], "label": "exact_match_correctness",
            "feature": "log P(True) - log P(False)", "C": 1.0}


def calibrated_probability(calibrator, logit):
    z = calibrator["coefficient"] * logit + calibrator["intercept"]
    return 1 / (1 + math.exp(-max(-700, min(700, z))))


def filter_candidate(row, tau, calibrator):
    p = calibrated_probability(calibrator, row["confidence_logit"])
    output = {**row, "tau": tau, "arm": "filter", "p_calibrated": p}
    if row["reason"] not in {"empty", "truncated"} and p <= tau:
        output.update(text="IDK", outcome="abstain", reason="calibrated_filter", truncated=False)
    return output


def collect(config, workdir, engine, rows, path, arm, seed, tau=None, forced=False, confidence=False):
    path = Path(path)
    existing = read_jsonl(path) if path.exists() else []
    done = {r["id"] for r in existing}
    expected = {r["id"] for r in rows}
    if len(done) != len(existing) or not done <= expected:
        raise ValueError(f"Duplicate/unexpected cached predictions: {path}")
    for r in existing:
        if (r["model"], r["arm"], r["seed"], r["tau"]) != (engine.key, arm, seed, tau):
            raise ValueError("Prediction cache belongs to a different condition.")
    pending = [r for r in rows if r["id"] not in done]
    batch = config["evaluation"]["batch_size"]
    for offset in range(0, len(pending), batch):
        require_budget(config, workdir)
        deadline = os.environ.get("ABSTENTION_STOP_AT")
        if deadline and time.time() >= float(deadline):
            from .runner import RunPaused
            raise RunPaused("Evaluation saved at the night-window boundary.")
        chunk = pending[offset:offset+batch]
        started = time.monotonic()
        outputs = engine.generate([messages(r["question"], tau, forced=forced) for r in chunk])
        for row, out in zip(chunk, outputs):
            result = {**row, **out, **grade(out["text"], row["aliases"], out["truncated"]).asdict(),
                      "model": engine.key, "arm": arm, "seed": seed, "tau": tau}
            if confidence:
                result.update(engine.confidence(row["question"], out["text"]))
            append_jsonl(path, result)
            existing.append(result)
        record_cost(workdir, "evaluation", time.monotonic() - started, model=engine.key, arm=arm,
                    tau=tau, questions=len(chunk), confidence=confidence)
        print(f"{path.name}: {len(existing)}/{len(rows)}", flush=True)
    return existing


def evaluate(config, workdir, key, arm="original", seed=17):
    workdir = Path(workdir)
    lock = require_lock(config, workdir, open_test=True)
    initializer = lock["pilots"][key]["initializer"]
    adapter = None
    if arm in {"binary", "conditioned", "fixed"}:
        run_dir = workdir / "artifacts/runs" / f"{key}-{arm}-s{seed}"
        if not (run_dir / "result.json").exists():
            raise RuntimeError("Cannot evaluate an unfinished training run.")
        adapter = run_dir / "adapter"
        if adapter_fingerprint(adapter) != read_json(run_dir / "result.json")["adapter_sha256"]:
            raise RuntimeError("Trained adapter changed since the run finished.")
    elif arm == "warmup":
        if not initializer:
            raise ValueError("This model did not require warmup.")
        # Match the merged shared initialization used by every GRPO arm.
        adapter = None
        seed = 0
    elif arm == "original":
        initializer = None
        seed = 0
    else:
        raise ValueError("Unknown evaluation arm.")
    directory = workdir / "artifacts/predictions" / f"{key}-{arm}-s{seed}"
    metadata = {"protocol_sha256": digest(lock), "model": key, "arm": arm, "seed": seed}
    if (directory / "metadata.json").exists() and read_json(directory / "metadata.json") != metadata:
        raise ValueError("Prediction metadata changed; refusing cache reuse.")
    write_json(directory / "metadata.json", metadata)
    if (directory / "complete.json").exists():
        return read_json(directory / "complete.json")
    engine = Engine(config, workdir, key, adapter=adapter, initializer=initializer)
    manifest = read_json(workdir / "data/prepared/manifest.json")
    try:
        if arm == "original":
            calibration = collect(config, workdir, engine, load_split(workdir, "calibration"),
                                  directory / "calibration-candidates.jsonl", "candidate", seed,
                                  forced=True, confidence=True)
            calibrator = fit_calibrator(calibration)
            write_json(directory / "calibrator.json", calibrator)
        for split in ("test_trivia", "test_nq"):
            all_rows = load_split(workdir, split)
            diagnostic_ids = set(manifest["diagnostic_ids"][split])
            if arm == "original":
                candidates = collect(config, workdir, engine, all_rows, directory / f"{split}-candidates.jsonl",
                                     "candidate", seed, forced=True, confidence=True)
            for tau in config["thresholds"]["primary"] + config["thresholds"]["diagnostic"]:
                rows = all_rows if tau in config["thresholds"]["primary"] else [r for r in all_rows if r["id"] in diagnostic_ids]
                final_arm = "prompt" if arm == "original" else arm
                collect(config, workdir, engine, rows, directory / f"{split}-{final_arm}-t{tau:.2f}.jsonl",
                        final_arm, seed, tau=tau)
                if arm == "original":
                    ids = {r["id"] for r in rows}
                    filtered = [filter_candidate(r, tau, calibrator) for r in candidates if r["id"] in ids]
                    forced = [{**r, "tau": tau, "arm": "forced"} for r in candidates if r["id"] in ids]
                    write_jsonl(directory / f"{split}-filter-t{tau:.2f}.jsonl", filtered)
                    write_jsonl(directory / f"{split}-forced-t{tau:.2f}.jsonl", forced)
        result = {**metadata, "status": "complete"}
        write_json(directory / "complete.json", result)
        return result
    finally:
        engine.close()
