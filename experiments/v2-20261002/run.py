"""Durable, serial v2 coordinator; all scientific choices precede test access."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from abstention.config import load_config
from abstention.io import append_jsonl, file_digest, read_json, read_jsonl, utcnow, write_json
from abstention.protocol import adapter_fingerprint

IMAGE = "sha256:106bd8033f516b97e47ee39eb17c7477e516ae5329977665d25cac96ce90c10f"
ARMS = ("a_g4_single_tau", "b_g8_single_tau", "c_g8_paired_tau", "d_g8_paired_fixed075")
SEEDS = (17, 29, 43)
MANUSCRIPT = "2d9e2fc0010f944982693fa43526a1867a5ca1715ef63aceb517edef85a22c5b"


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def cfg():
    return load_config(HERE / "config.yaml")


def event(kind, **fields):
    append_jsonl(ART / "events.jsonl", {"at": utcnow(), "event": kind, **fields})


def prepare():
    from prepare_data import verify
    verify()
    contract_path = HERE / "contract.json"
    if contract_path.exists():
        return read_json(contract_path)
    data = {}
    for key, filename in (("train", "train_full"), ("dev", "dev"), ("calibration", "calibration")):
        p = HERE / f"data/prepared/{filename}.jsonl"
        data[key] = {"path": relative(p), "sha256": file_digest(p)}
    c = {"study_id": cfg()["study_id"], "created_at": utcnow(),
         "model_workdir": "experiments/final-eval-20261001", "model_key": "qwen14b",
         "model_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18", "pinned_image": IMAGE,
         "config_path": relative(HERE / "config.yaml"),
         "data_manifest": relative(HERE / "data/prepared/manifest.json"),
         "output_root": relative(ART / "selector"), "data": data,
         "evaluation": {"primary_thresholds": [.65, .85], "diagnostic_thresholds": [.6, .75, .9, .95],
                        "dev_thresholds": [.6, .75, .9], "diagnostic_count": 200},
         "selector": cfg()["selector"], "seeds": list(SEEDS),
         "min_system_available_gib": 16}
    write_json(contract_path, c)
    event("contract_created", contract_sha256=file_digest(contract_path))
    return c


def seal():
    """Only call after meaningful tests; refuse to overwrite the registered seal."""
    from prepare_data import verify
    verify()
    prepare()
    needed = [HERE / f for f in ("training_v2.py", "selector_v2.py", "evaluate_v2.py", "analysis_v2.py",
              "tests_training_v2.py", "tests_selector_v2.py", "tests_prepare_data.py",
              "tests_evaluate_v2.py", "tests_analysis_v2.py", "tests_run.py")]
    for p in needed:
        if not p.is_file():
            raise RuntimeError(f"Implementation incomplete: {p}")
    test_receipt = read_json(ART / "tests.json")
    if test_receipt.get("status") != "passed":
        raise RuntimeError("A passing test receipt is required")
    for p, sha in test_receipt["files"].items():
        if file_digest(ROOT / p) != sha:
            raise RuntimeError(f"Tests predate the current code: {p}")
    protected = sorted((ROOT / "src/abstention").glob("*.py")) + [ROOT / "paper/manuscript.tex"]
    if file_digest(ROOT / "paper/manuscript.tex") != MANUSCRIPT:
        raise RuntimeError("Manuscript changed; inspect instead of overwriting history")
    paths = [*HERE.glob("*.py"), HERE / "config.yaml", HERE / "contract.json",
             ROOT / "docs/tres-mejoras-v2.md", ROOT / "scripts/start-v2.sh",
             *sorted((HERE / "data/prepared").glob("*.json*")),
             ROOT / "experiments/final-eval-20261001/artifacts/models.lock.json", *protected]
    target = ART / "source-lock.json"
    if target.exists():
        return check()
    value = {"created_at": utcnow(), "status": "frozen_before_main_training_and_test",
             "image": IMAGE, "tests_sha256": file_digest(ART / "tests.json"),
             "files": {relative(p): file_digest(p) for p in sorted(set(paths))}}
    write_json(target, value)
    event("sources_frozen", sha256=file_digest(target))
    return value


def check():
    lock = read_json(ART / "source-lock.json")
    if lock["image"] != IMAGE:
        raise RuntimeError("Image mismatch")
    for p, expected in lock["files"].items():
        if file_digest(ROOT / p) != expected:
            raise RuntimeError(f"Frozen source/data changed: {p}")
    return lock


def arm_spec(arm, seed):
    from training_v2 import V2Spec
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("Unregistered arm or seed")
    return V2Spec(group_size=4 if arm == ARMS[0] else 8,
                  arm="fixed" if arm == ARMS[3] else "conditioned", seed=seed,
                  exposure_mode="single_tau" if arm in ARMS[:2] else "paired_tau")


def roundtrip(adapter, questions):
    from abstention.models import Engine
    from abstention.prompts import messages
    outputs = []
    for _ in range(2):
        engine = Engine(cfg(), ROOT / prepare()["model_workdir"], "qwen14b", adapter=adapter)
        try:
            outputs.append(engine.generate([messages(r["question"], .75) for r in questions]))
        finally:
            engine.close()
    stable = lambda rows: [(r["text"], r["truncated"], r["generated_tokens"]) for r in rows]
    if stable(outputs[0]) != stable(outputs[1]):
        raise RuntimeError("Independent adapter reloads did not reproduce deterministic TRAIN responses")
    return {"passed": True, "ids": [r["id"] for r in questions], "outputs": outputs,
            "adapter_sha256": adapter_fingerprint(adapter)}


def worker_rl(arm, seed, max_seconds, pilot=False):
    from prepare_data import training_rows
    from training_v2 import train_v2
    if not pilot:
        check()
        count = read_json(ART / "budget-lock.json")["rl_questions"]
    else:
        for path, expected in read_json(ART / "pilot-source-lock.json")["files"].items():
            if file_digest(ROOT / path) != expected:
                raise RuntimeError(f"Pilot implementation changed: {path}")
        count = 1008
    schedule = training_rows(arm, seed, n_questions=count)
    out = ART / ("pilots/rl" if pilot else f"runs/{arm}-s{seed}")
    if pilot:
        if arm != ARMS[2] or seed != 17:
            raise ValueError("The technical pilot is fixed to C, seed17,32steps")
        schedule = schedule[:32 * 2 * 3]
    result = train_v2(cfg(), ROOT / prepare()["model_workdir"], schedule, out,
                      arm_spec(arm, seed), max_runtime_seconds=max_seconds)
    if pilot:
        reload_check = roundtrip(out / "adapter", schedule[::3][:8])
        write_json(out / "reload-check.json", reload_check)
    return result


def job_records():
    return [read_json(p) for p in sorted((ART / "jobs").glob("*/job.json"))]


def spent_seconds():
    rows = job_records()
    unknown = [r["id"] for r in rows if r["status"] == "running"]
    if unknown:
        raise RuntimeError(f"An unresolved attempt requires explicit recovery: {unknown}")
    return sum(r["seconds"] for r in rows)


def live_counter(job_id):
    """Small append-only logs or atomic shard counts; never read test outcomes."""
    if job_id == "rl-pilot":
        path, total = ART / "pilots/rl/steps.jsonl", 32
    elif job_id and any(job_id.startswith(a + "-s") for a in ARMS):
        path = ART / f"runs/{job_id}/steps.jsonl"
        total = read_json(ART / "budget-lock.json")["rl_steps"]
    elif job_id == "selector-pilot":
        path, total = ART / "selector/pilots/s17/steps.jsonl", 16
    elif job_id in {f"selector-s{s}" for s in SEEDS}:
        path, total = ART / f"selector/runs/{job_id.split('-')[-1]}/steps.jsonl", 500
    elif job_id in {"selector-candidates", "selector-test-candidates"}:
        splits = ("train", "dev", "calibration") if job_id == "selector-candidates" else ("test_trivia", "test_nq")
        done = sum(sum(1 for _ in (ART / "selector/candidate-shards" / s).glob("*.json")) for s in splits)
        return {"unit": "preguntas con candidatas guardadas", "completed": done,
                "total": 3000 if job_id == "selector-candidates" else 2500}
    else:
        return None
    # The RL writer appends a complete line; ignore its possibly incomplete tail.
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return {"unit": "actualizaciones", "completed": len(rows), "total": total}


def progress(status, current=None, **extra):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    counter = live_counter(current)
    if counter is not None:
        extra["progress"] = counter
    state = {"at": utcnow(), "status": status, "current_job": current,
             "jobs_complete": sum(r["status"] == "complete" for r in job_records()), **extra}
    write_json(ART / "status.json", state)
    text = ("# Progreso de las tres mejoras v2\n\n"
            f"Actualizado: {datetime.now(ZoneInfo('America/Mexico_City')).strftime('%Y-%m-%d %H:%M:%S')} (Ciudad de México).\n\nEstado: **{status}**. "
            f"Tarea: `{current or 'ninguna'}`. Tareas completadas: {state['jobs_complete']}.\n\n"
            "[Protocolo y comparaciones](../../docs/tres-mejoras-v2.md) · "
            "[Estado JSON](artifacts/status.json) · [Bitácora](artifacts/events.jsonl) · "
            "[Logs por tarea](artifacts/jobs/)\n\n"
            "La cola usa una sola GPU continuamente. Si falla una tarea, conserva sus archivos y se detiene; "
            "no elige mejores semillas ni reintenta silenciosamente. Las métricas de test se analizan al completar todos los brazos.\n")
    if counter:
        text += f"\n**Avance actual: {counter['completed']} / {counter['total']} {counter['unit']}.**\n"
    if extra:
        text += "\n```json\n" + json.dumps(extra, indent=2, ensure_ascii=False) + "\n```\n"
    tmp = HERE / ".PROGRESO.tmp"
    tmp.write_text(text)
    os.replace(tmp, HERE / "PROGRESO.md")


def job(job_id, argv, cap_seconds):
    """OS timeout covers model loading, training, saving and cleanup, including hangs."""
    path = ART / "jobs" / job_id
    existing = path / "job.json"
    if existing.exists():
        r = read_json(existing)
        if r["status"] != "complete" or r["argv"] != argv:
            raise RuntimeError(f"No automatic retry or identity change: {job_id}")
        if file_digest(ROOT / r["source_lock_path"]) != r["source_lock_sha256"]:
            raise RuntimeError(f"Completed job's source registration changed: {job_id}")
        return r
    available = cfg()["budget"]["max_phase_hours"] * 3600 - spent_seconds()
    cap_seconds = min(cap_seconds, available)
    if cap_seconds <= 0:
        raise RuntimeError("Phase hard budget exhausted")
    path.mkdir(parents=True, exist_ok=False)
    seal_path = ART / ("source-lock.json" if (ART / "source-lock.json").exists() else "pilot-source-lock.json")
    record = {"id": job_id, "argv": argv, "started_at": utcnow(), "status": "running",
              "cap_seconds": cap_seconds, "source_lock_path": relative(seal_path),
              "source_lock_sha256": file_digest(seal_path)}
    write_json(existing, record)
    event("job_started", id=job_id, cap_seconds=cap_seconds)
    start = time.monotonic()
    code = None
    proc = None
    try:
        with (path / "output.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, "-u", *argv], cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                progress("running", job_id, elapsed_job_seconds=elapsed,
                         phase_hours_including_current=(sum(r.get("seconds", 0) for r in job_records()) + elapsed) / 3600,
                         log=relative(path / "output.log"), cap_seconds=cap_seconds)
                if elapsed >= cap_seconds:
                    raise TimeoutError(f"Job exceeded registered cap: {job_id}")
                try:
                    proc.wait(timeout=min(30, max(.01, cap_seconds - elapsed)))
                except subprocess.TimeoutExpired:
                    pass
            code = proc.returncode
            if code:
                raise RuntimeError(f"{job_id} exited {code}; inspect its preserved output.log")
        record["status"] = "complete"
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        raise
    finally:
        record.update(seconds=time.monotonic() - start, finished_at=utcnow(), returncode=code)
        write_json(existing, record)
        event("job_finished", **{k: record[k] for k in ("id", "status", "seconds")})
    return record


def technical_pilot():
    """Technical TRAIN-only pilot can overlap CPU work on the final evaluators."""
    from prepare_data import verify
    verify()
    prepare()
    target = ART / "pilot-source-lock.json"
    if target.exists():
        raise RuntimeError("Pilot already registered; no automatic relaunch")
    paths = [HERE / f for f in ("training_v2.py", "tests_training_v2.py", "prepare_data.py",
                               "tests_prepare_data.py", "config.yaml", "contract.json")]
    paths += sorted((ROOT / "src/abstention").glob("*.py"))
    paths += sorted((HERE / "data/prepared").glob("*.json*"))
    write_json(target, {"at": utcnow(), "scope": "TRAIN-only disposable32step pilot",
                       "files": {relative(p): file_digest(p) for p in paths}})
    job("rl-pilot", [relative(HERE / "run.py"), "worker-rl", "--arm", ARMS[2], "--seed", "17",
                     "--pilot", "--max-seconds", "10800"], 11400)
    progress("technical_rl_pilot_complete", total_phase_hours=spent_seconds() / 3600)


def budget_projection(rl_step_seconds, selector_step_seconds, already_hours, planned_hours=80, margin=1.25):
    import math
    if not all(math.isfinite(v) and v > 0 for v in (rl_step_seconds, selector_step_seconds, planned_hours, margin)):
        raise ValueError("Projection requires positive finite measurements and limits")
    if not math.isfinite(already_hours) or already_hours < 0:
        raise ValueError("Invalid accumulated cost")
    overhead_hours = 15 + 1 + 15 / 3
    selector_hours = 3 * 500 * selector_step_seconds / 3600
    estimates = {n: already_hours + margin * (12 * (n // 2) * rl_step_seconds / 3600 + selector_hours + overhead_hours)
                 for n in (1008, 504)}
    feasible = [n for n in (1008, 504) if estimates[n] <= planned_hours]
    return {"selected_questions": feasible[0] if feasible else None,
            "projected_phase_hours": estimates, "fixed_overhead_hours": overhead_hours}


def freeze_budget():
    target = ART / "budget-lock.json"
    if target.exists():
        return read_json(target)
    import numpy as np
    from prepare_data import training_rows
    from training_v2 import prepared_records, verify_rollouts
    rl = read_json(ART / "pilots/rl/result.json")
    steps = read_jsonl(ART / "pilots/rl/steps.jsonl")
    reload_check = read_json(ART / "pilots/rl/reload-check.json")
    spec = arm_spec(ARMS[2], 17)
    records = prepared_records(training_rows(ARMS[2], 17, n_questions=1008)[:32 * 2 * 3], spec)
    rollout_path = ART / "pilots/rl/rollouts.jsonl"
    verify_rollouts(read_jsonl(rollout_path), records, spec, 32)
    if (rl["optimizer_steps"] != 32 or rl["gradient_steps"] != 32
            or rl["questions"] != 64 or rl["completions"] != 1536 or rl["spec"] != asdict(spec)
            or [r["step"] for r in steps] != list(range(1, 33))
            or rl["rollouts_sha256"] != file_digest(rollout_path)
            or not reload_check["passed"]
            or reload_check["adapter_sha256"] != rl["adapter_sha256"]
            or rl["adapter_sha256"] != adapter_fingerprint(ART / "pilots/rl/adapter")):
        raise RuntimeError("Real32step RL pilot and reload verification required")
    critic = read_json(ART / "selector/pilots/s17/complete.json")
    # Selector pilot exposes measured optimizer-step time, including backward/optimizer.
    selector_steps = read_jsonl(ART / "selector/pilots/s17/steps.jsonl")
    from selector_v2 import context, tree_hash
    ctx = context(HERE / "contract.json")
    if (len(selector_steps) != 16 or critic["status"] != "complete" or critic["steps"] != 16
            or not critic["technical_pilot_disposable"] or critic["examples_seen"] != 256
            or critic["identity_sha256"] != ctx["sha"]
            or tree_hash(ctx["out"] / critic["checkpoint"]) != critic["checkpoint_sha256"]
            or rl["minimum_available_gib"] < 16):
        raise RuntimeError("Real16step selector pilot required")
    if not all(np.isfinite(r["seconds"]) and r["seconds"] > 0 for r in steps + selector_steps):
        raise RuntimeError("Invalid pilot timings")
    rl_sec = float(np.mean([r["seconds"] for r in steps[8:]]))
    selector_sec = float(np.mean([r["seconds"] for r in selector_steps[4:]]))
    already = spent_seconds() / 3600
    # Fixed conservative allowances:15h final inference/analysis,1h dev selector scoring,
    # plus20minutes/model-run load+save across15full trainings; margin also covers these.
    margin = cfg()["budget"]["projection_margin"]
    projected = budget_projection(rl_sec, selector_sec, already, cfg()["budget"]["planned_phase_hours"], margin)
    count = projected["selected_questions"]
    estimates = projected["projected_phase_hours"]
    if count is None:
        write_json(ART / "budget-blocked.json", {"at": utcnow(), "projected_phase_hours": estimates,
                    "reason": "Even the uniform reduced recipe exceeds80 planned hours; no test opened"})
        raise RuntimeError("Measured pilot budget exceeds planned80hours even after uniform reduction")
    value = {"created_at": utcnow(), "rl_questions": count, "rl_steps": count // 2,
             "rl_completions": count * 24, "selector_steps": 500,
             "rl_step_seconds": rl_sec, "selector_step_seconds": selector_sec,
             "spent_pilot_and_candidate_hours": already, "projection_margin": margin,
             "fixed_overhead_hours": projected["fixed_overhead_hours"], "projected_phase_hours": estimates,
             "decision_uses": "Runtime only, before test; identical RL size across all arms/seeds",
             "pilot_receipts": {"rl": file_digest(ART / "pilots/rl/result.json"),
                                "rl_steps": file_digest(ART / "pilots/rl/steps.jsonl"),
                                "rl_reload": file_digest(ART / "pilots/rl/reload-check.json"),
                                "selector": file_digest(ART / "selector/pilots/s17/complete.json"),
                                "selector_steps": file_digest(ART / "selector/pilots/s17/steps.jsonl")}}
    write_json(target, value)
    event("budget_frozen", **value)
    return value


def open_evaluation():
    check()
    target = ART / "evaluation-lock.json"
    if target.exists():
        return read_json(target)
    lock = {"status": "ready", "created_at": utcnow(), "contract_path": relative(HERE / "contract.json"),
            "contract_sha256": file_digest(HERE / "contract.json"), "data": {},
            "source_lock_sha256": file_digest(ART / "source-lock.json"),
            "budget_lock_sha256": file_digest(ART / "budget-lock.json"), "rl_completed": {},
            "selector_completion_sha256": {}, "selector_evaluation_sha256": {}}
    budget = read_json(ART / "budget-lock.json")
    for seed in SEEDS:
        for arm in ARMS:
            variant = f"{arm}-s{seed}"
            path = ART / f"runs/{variant}/result.json"
            result = read_json(path)
            adapter = path.parent / "adapter"
            if (result["status"] != "complete" or result["optimizer_steps"] != budget["rl_steps"]
                    or result["gradient_steps"] != budget["rl_steps"]
                    or result["questions"] != budget["rl_questions"]
                    or result["completions"] != budget["rl_completions"]
                    or result["adapter_sha256"] != adapter_fingerprint(adapter)
                    or result["spec"] != asdict(arm_spec(arm, seed))
                    or not result["save_roundtrip"] or not result["sampler_traversal_verified"]):
                raise RuntimeError(f"Incomplete/changed adapter: {variant}")
            lock["rl_completed"][variant] = {"adapter": relative(adapter),
                "adapter_sha256": result["adapter_sha256"], "result_path": relative(path),
                "result_sha256": file_digest(path)}
        p = ART / f"selector/runs/s{seed}/complete.json"
        from selector_v2 import context, verify_run
        completion = verify_run(context(HERE / "contract.json"), seed)
        if completion["status"] != "complete" or completion["technical_pilot_disposable"]:
            raise RuntimeError("Selector incomplete")
        lock["selector_completion_sha256"][str(seed)] = file_digest(p)
        ep = ART / f"selector/evaluation/s{seed}/complete.json"
        lock["selector_evaluation_sha256"][str(seed)] = file_digest(ep)
    for split in ("test_trivia", "test_nq"):
        p = HERE / f"data/prepared/{split}.jsonl"
        lock["data"][split] = {"path": relative(p), "sha256": file_digest(p)}
    write_json(target, lock)
    event("test_opened", lock_sha256=file_digest(target))
    return lock


def queue():
    check()
    if os.environ.get("ABSTENTION_IMAGE_ID") != IMAGE:
        raise RuntimeError("Run inside the pinned image")
    ART.mkdir(exist_ok=True)
    with (ART / "queue.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            runner = relative(HERE / "run.py")
            selector = relative(HERE / "selector_v2.py")
            contract = relative(HERE / "contract.json")
            job("rl-pilot", [runner, "worker-rl", "--arm", ARMS[2], "--seed", "17", "--pilot",
                             "--max-seconds", "10800"], 11400)
            job("selector-candidates", [selector, "prepare-candidates", "--contract", contract,
                                         "--max-seconds", "14400"], 14700)
            job("selector-pilot", [selector, "pilot", "--contract", contract, "--seed", "17",
                                    "--max-seconds", "7200"], 7500)
            budget = freeze_budget()
            rl_cap = min(12 * 3600, max(3600, budget["rl_steps"] * budget["rl_step_seconds"] * 1.6 + 1200))
            selector_cap = min(8 * 3600, max(3600, 500 * budget["selector_step_seconds"] * 1.6 + 1200))
            # Complete each seed's four controls before the next, with a deterministic
            # cyclic arm order to avoid a fixed arm always occupying the first slot.
            for index, seed in enumerate(SEEDS):
                for arm in ARMS[index:] + ARMS[:index]:
                    job(f"{arm}-s{seed}", [runner, "worker-rl", "--arm", arm, "--seed", str(seed),
                        "--max-seconds", str(rl_cap)], rl_cap + 120)
                job(f"selector-s{seed}", [selector, "train", "--contract", contract, "--seed", str(seed),
                                         "--max-seconds", str(selector_cap)], selector_cap + 120)
                job(f"selector-dev-s{seed}", [selector, "evaluate", "--contract", contract, "--seed", str(seed),
                                              "--max-seconds", "10800"], 11100)
            open_evaluation()
            elock = relative(ART / "evaluation-lock.json")
            job("selector-test-candidates", [selector, "prepare-test-candidates", "--contract", contract,
                 "--evaluation-lock", elock, "--max-seconds", "14400"], 14700)
            for variant in ["original", *read_json(ART / "evaluation-lock.json")["rl_completed"]]:
                job(f"test-{variant}", [relative(HERE / "evaluate_v2.py"), "--variant", variant,
                     "--evaluation-lock", elock, "--max-seconds", "10800"], 11100)
            for seed in SEEDS:
                job(f"selector-test-s{seed}", [selector, "evaluate-test", "--contract", contract,
                     "--seed", str(seed), "--evaluation-lock", elock, "--max-seconds", "10800"], 11100)
            job("analysis", [relative(HERE / "analysis_v2.py")], 7200)
            check()
            progress("artifacts_ready_for_review", total_phase_hours=spent_seconds() / 3600,
                     review_pending="Inspect statistical outputs and rendered figures before manuscript claims")
            event("queue_complete", total_phase_hours=spent_seconds() / 3600, scientific_review_pending=True)
        except BaseException as exc:
            progress("stopped_after_error", error=repr(exc))
            event("queue_stopped", error=repr(exc), traceback=traceback.format_exc())
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "seal", "check", "queue", "worker-rl", "status", "technical-pilot"])
    p.add_argument("--arm", choices=ARMS, default=ARMS[2])
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--max-seconds", type=float)
    a = p.parse_args()
    if a.command == "worker-rl":
        if a.max_seconds is None:
            p.error("--max-seconds is required")
        result = worker_rl(a.arm, a.seed, a.max_seconds, a.pilot)
    elif a.command == "status":
        result = read_json(ART / "status.json")
    elif a.command == "technical-pilot":
        result = technical_pilot()
    else:
        result = globals()[a.command]()
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
