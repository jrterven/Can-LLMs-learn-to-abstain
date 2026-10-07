"""Locked deterministic v2 evaluation; no training, calibration, or test tuning.

Original forced answers reuse the selector's exact frozen-generator candidates.
All threshold policies and all adapted forced answers are generated independently.
Partial outputs require explicit recovery; this module never resumes/overwrites.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from abstention.config import load_config
from abstention.io import digest, file_digest, read_json, read_jsonl, utcnow, write_json, write_jsonl
from abstention.prompts import messages
from abstention.protocol import adapter_fingerprint
from abstention.scoring import grade, normalize_question

SEEDS = (17, 29, 43)
ARMS = ("a_g4_single_tau", "b_g8_single_tau", "c_g8_paired_tau", "d_g8_paired_fixed075")
VARIANTS = {f"{arm}-s{seed}" for arm in ARMS for seed in SEEDS}
PRIMARY = (.65, .85)
DIAGNOSTIC = (.6, .75, .9, .95)
COUNTS = {"test_trivia": 2000, "test_nq": 500}
N_DIAGNOSTIC = 200
IMAGE = "sha256:106bd8033f516b97e47ee39eb17c7477e516ae5329977665d25cac96ce90c10f"


def resolve(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Contract paths must stay inside the repository")
    return path


def selector_module():
    spec = importlib.util.spec_from_file_location("v2_evaluation_selector", WORK / "selector_v2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_disjoint(splits):
    seen_ids, seen_questions = set(), set()
    for name, rows in splits.items():
        ids = [r["id"] for r in rows]
        questions = [normalize_question(r["question"]) for r in rows]
        if len(set(ids)) != len(rows) or len(set(questions)) != len(rows):
            raise ValueError(f"Duplicate ID or normalized question in {name}")
        if set(ids) & seen_ids or set(questions) & seen_questions:
            raise ValueError("Learning/calibration/evaluation questions overlap")
        if any(not q for q in questions) or any(not r.get("aliases") for r in rows):
            raise ValueError("Invalid question/reference record")
        seen_ids.update(ids); seen_questions.update(questions)


def check_lock(lock_path, variant, selector=None):
    """Validate all training completions before opening held-out records."""
    if variant != "original" and variant not in VARIANTS:
        raise ValueError("Unknown evaluation variant")
    lock_path = Path(lock_path).resolve()
    if not lock_path.is_relative_to(ROOT.resolve()):
        raise ValueError("Evaluation lock must stay inside repository")
    lock = read_json(lock_path)
    if lock.get("status") != "ready" or set(lock.get("rl_completed", {})) != VARIANTS:
        raise ValueError("Evaluation requires a ready lock and all twelve RL variants")
    if any(set(lock.get(key, {})) != {str(s) for s in SEEDS}
           for key in ("selector_completion_sha256", "selector_evaluation_sha256")):
        raise ValueError("Evaluation requires all three trained and calibrated selector seeds")
    contract_path = resolve(lock["contract_path"])
    if file_digest(contract_path) != lock["contract_sha256"]:
        raise ValueError("Evaluation belongs to another contract")
    contract = read_json(contract_path)
    hashes = {str(lock_path.relative_to(ROOT)): file_digest(lock_path),
              str(contract_path.relative_to(ROOT)): file_digest(contract_path)}
    def attest(path, expected):
        if file_digest(path) != expected:
            raise ValueError(f"Changed attested input: {path}")
        relative = str(path.relative_to(ROOT))
        if relative in hashes and hashes[relative] != expected:
            raise ValueError("Conflicting source attestations")
        hashes[relative] = expected
    for name, key in (("source-lock.json", "source_lock_sha256"), ("budget-lock.json", "budget_lock_sha256")):
        attest(WORK / "artifacts" / name, lock[key])
    source_lock = read_json(WORK / "artifacts/source-lock.json")
    if source_lock["image"] != IMAGE:
        raise ValueError("Unregistered runtime image")
    for relative, expected in source_lock["files"].items():
        attest(resolve(relative), expected)
    budget = read_json(WORK / "artifacts/budget-lock.json")
    if (budget["rl_questions"], budget["rl_steps"], budget["rl_completions"]) not in {(1008, 504, 24192), (504, 252, 12096)}:
        raise ValueError("RL budget is not one of the two registered complete budgets")
    for name, spec in lock["rl_completed"].items():
        result_path = resolve(spec["result_path"]); attest(result_path, spec["result_sha256"])
        result = read_json(result_path)
        if (result.get("status") != "complete" or result.get("optimizer_steps") != budget["rl_steps"]
                or result.get("gradient_steps") != budget["rl_steps"] or result.get("questions") != budget["rl_questions"]
                or result.get("completions") != budget["rl_completions"] or result.get("save_roundtrip") is not True
                or result.get("sampler_traversal_verified") is not True):
            raise ValueError(f"Incomplete RL training: {name}")
        arm, seed = name.rsplit("-s", 1)
        group_size, exposure, reward = {
            ARMS[0]: (4, "single_tau", "conditioned"), ARMS[1]: (8, "single_tau", "conditioned"),
            ARMS[2]: (8, "paired_tau", "conditioned"), ARMS[3]: (8, "paired_tau", "fixed")}[arm]
        identity = {"group_size": group_size, "exposure_mode": exposure, "arm": reward, "seed": int(seed),
                    "questions_per_update": 2, "micro_batch_size": 4}
        if result.get("spec") != identity:
            raise ValueError(f"RL receipt belongs to another arm or seed: {name}")
        adapter = resolve(spec["adapter"])
        if result["adapter_sha256"] != spec["adapter_sha256"] or adapter_fingerprint(adapter) != spec["adapter_sha256"]:
            raise ValueError("Completed RL adapter changed")
        for filename, checksum in spec["adapter_sha256"].items():
            attest(adapter / filename, checksum)
    selector = selector_module() if selector is None else selector
    selector_ctx = selector.context(contract_path, root=ROOT)
    for seed in SEEDS:
        complete = selector_ctx["out"] / "runs" / f"s{seed}" / "complete.json"
        attest(complete, lock["selector_completion_sha256"][str(seed)])
        result = selector.verify_run(selector_ctx, seed)
        if (result.get("status") != "complete" or result.get("seed") != seed or result.get("steps") != 500
                or result.get("examples_seen") != 8000 or result.get("technical_pilot_disposable") is not False):
            raise ValueError("Selector completion is a pilot or incomplete training")
        evaluation = selector_ctx["out"] / "evaluation" / f"s{seed}" / "complete.json"
        attest(evaluation, lock["selector_evaluation_sha256"][str(seed)])
        done = read_json(evaluation)
        if done.get("status") != "complete" or done.get("identity_sha256") != selector_ctx["sha"]:
            raise ValueError("Incomplete or unrelated selector calibration")
        for name, checksum in done["files"].items():
            attest(evaluation.parent / name, checksum)
    # All fifteen training runs are attested. Held-out references may now be read.
    data, diagnostic = {}, {}
    manifest_path = resolve(contract["data_manifest"])
    manifest = read_json(manifest_path)
    if str(manifest_path.relative_to(ROOT)) not in source_lock["files"]:
        raise ValueError("Source lock does not attest the data manifest")
    for split, count in COUNTS.items():
        spec = lock["data"][split]; path = resolve(spec["path"]); attest(path, spec["sha256"])
        rows = read_jsonl(path)
        if len(rows) != count or manifest["splits"][split]["sha256"] != spec["sha256"] or manifest["splits"][split]["ids"] != [r["id"] for r in rows]:
            raise ValueError("Wrong held-out population or provenance")
        ids = manifest["diagnostic_ids"][split]
        if len(ids) != N_DIAGNOSTIC or len(set(ids)) != N_DIAGNOSTIC or not set(ids) <= {r["id"] for r in rows}:
            raise ValueError("Invalid fixed diagnostic panel")
        data[split] = rows; diagnostic[split] = set(ids)
    assert_disjoint({**selector_ctx["splits"], **data})
    config_path = resolve(contract["config_path"])
    if str(config_path.relative_to(ROOT)) not in source_lock["files"]:
        raise ValueError("Configuration is absent from source lock")
    config = load_config(config_path)
    if (config["training"]["max_prompt_length"], config["training"]["max_completion_length"], config["evaluation"]["batch_size"]) != (512, 32, 16):
        raise ValueError("Generation limits changed")
    if contract["evaluation"]["primary_thresholds"] != list(PRIMARY) or contract["evaluation"]["diagnostic_thresholds"] != list(DIAGNOSTIC):
        raise ValueError("Evaluation thresholds changed")
    return {"lock": lock, "lock_path": lock_path, "contract": contract, "config": config, "data": data,
            "diagnostic": diagnostic, "hashes": hashes, "selector": selector, "selector_ctx": selector_ctx}


def prediction(output, reference, variant, split, tau, reused=False):
    if not isinstance(output.get("text"), str) or type(output.get("truncated")) is not bool:
        raise ValueError("Invalid generated text/truncation")
    if type(output.get("generated_tokens")) is not int or not 0 <= output["generated_tokens"] <= 32:
        raise ValueError("Invalid generated token count")
    if type(output.get("seconds")) not in (int, float) or not math.isfinite(output["seconds"]) or output["seconds"] < 0:
        raise ValueError("Invalid generation timing")
    outcome = grade(output["text"], reference["aliases"], output["truncated"]).asdict()
    result = {**reference, **{k: output[k] for k in ("text", "truncated", "generated_tokens", "seconds")}, **outcome,
              "variant": variant, "variant_id": variant, "model": "qwen14b",
              "arm": "original" if variant == "original" else variant.rsplit("-s", 1)[0],
              "seed": 0 if variant == "original" else int(variant.rsplit("-s", 1)[1]),
              "evaluation_split": split, "mode": "forced" if tau is None else "threshold", "tau": tau,
              "reused_original_candidate": reused}
    return {**result, "row_sha256": digest(result)}


def shared_original_candidates(ctx):
    selector = ctx["selector"]
    enhanced = selector.locked_test_context(ctx["selector_ctx"], ctx["lock_path"])
    result = {}
    for split, references in ctx["data"].items():
        path = enhanced["out"] / "candidates" / f"{split}.json"
        before = file_digest(path)
        rows = selector.candidates(enhanced, split)
        if file_digest(path) != before:
            raise ValueError("Shared candidates changed while being read")
        ctx["hashes"][str(path.relative_to(ROOT))] = before
        if len(rows) != len(references):
            raise ValueError("Incomplete shared forced-candidate population")
        checked = []
        for item, reference in zip(rows, references):
            if item["id"] != reference["id"] or item["question"] != reference["question"] or item["split"] != split or item["candidate_index"] != 0 or item["generator"] != "original_frozen":
                raise ValueError("Shared original candidate identity/order changed")
            row = prediction(item, reference, "original", split, None, reused=True)
            if any(item[key] != row[key] for key in ("outcome", "reason")) or item["y"] != int(row["outcome"] == "correct"):
                raise ValueError("Shared candidate grade disagrees with original references")
            checked.append(row)
        result[split] = checked
    return result


def resources():
    import torch
    available = next(int(line.split()[1]) / 1024**2 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    return {"available_gib": available, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}


class Deadline:
    def __init__(self, seconds, probe, clock=time.monotonic, alarms=True):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("A finite positive maximum runtime is required")
        self.seconds, self.probe, self.clock, self.alarms = seconds, probe, clock, alarms
        self.minimum_available_gib = math.inf
        self.latest_resources = {}

    def check(self):
        if self.clock() - self.started >= self.seconds:
            raise TimeoutError("Evaluation runtime budget exhausted")
        state = self.probe()
        available = state["available_gib"]
        if not math.isfinite(available) or available < 16:
            raise MemoryError("System memory reserve fell below 16 GiB")
        self.minimum_available_gib = min(self.minimum_available_gib, available)
        self.latest_resources = state

    def __enter__(self):
        self.started = self.clock()
        if self.alarms:
            self.previous = signal.getsignal(signal.SIGALRM)
            def expired(*_):
                raise TimeoutError("Evaluation runtime alarm expired")
            signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *_):
        if self.alarms:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous)


def evaluate(evaluation_lock, variant, max_seconds, *, engine_factory=None, resource_probe=None, selector=None):
    # Injected engines are for CPU fixtures only; the CLI has no bypass switch.
    real = engine_factory is None
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("A finite positive maximum runtime is required")
    started = time.monotonic()
    with Deadline(max_seconds, resource_probe or resources, alarms=real) as budget:
        ctx = check_lock(evaluation_lock, variant, selector)
        output_dir = WORK / "artifacts/predictions" / variant
        if output_dir.exists():
            raise FileExistsError("Evaluation output exists; no overwrite or automatic partial resume")
        if real:
            ctx["selector"].env_check()
            from abstention.models import Engine
            engine_factory = Engine
        budget.check()
        shared = shared_original_candidates(ctx) if variant == "original" else {}
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json(output_dir / "attempt.json", {"status": "running", "at": utcnow(), "variant": variant,
                   "evaluation_lock_sha256": file_digest(ctx["lock_path"]), "max_seconds": max_seconds})
        engine, steps, fresh_rows, reused_rows = None, 0, 0, 0
        outputs, populations = {}, {}
        try:
            adapter = None if variant == "original" else resolve(ctx["lock"]["rl_completed"][variant]["adapter"])
            model_started = time.monotonic()
            engine = engine_factory(ctx["config"], resolve(ctx["contract"]["model_workdir"]), "qwen14b", adapter=adapter)
            model_seconds = time.monotonic() - model_started
            budget.check()
            for split, source in ctx["data"].items():
                for tau in (None, *PRIMARY, *DIAGNOSTIC):
                    chosen = source if tau is None or tau in PRIMARY else [r for r in source if r["id"] in ctx["diagnostic"][split]]
                    filename = f"{split}-forced.jsonl" if tau is None else f"{split}-t{tau:.2f}.jsonl"
                    if variant == "original" and tau is None:
                        rows = shared[split]; reused_rows += len(rows)
                    else:
                        rows = []
                        for offset in range(0, len(chosen), 16):
                            budget.check(); chunk = chosen[offset:offset+16]
                            generated = engine.generate([messages(r["question"], tau=tau, forced=tau is None) for r in chunk], sample=False)
                            budget.check()
                            if len(generated) != len(chunk):
                                raise ValueError("Incomplete generation batch")
                            rows.extend(prediction(item, ref, variant, split, tau) for item, ref in zip(generated, chunk))
                            fresh_rows += len(chunk); steps += 1
                    if len(rows) != len(chosen) or [r["id"] for r in rows] != [r["id"] for r in chosen]:
                        raise ValueError("Prediction population/order mismatch")
                    write_jsonl(output_dir / filename, rows)
                    outputs[filename] = file_digest(output_dir / filename); populations[filename] = len(rows)
                    write_json(output_dir / "progress.json", {"at": utcnow(), "variant": variant, "stage": filename,
                               "completed_files": len(outputs), "generation_batches": steps, "fresh_generated_rows": fresh_rows,
                               "reused_candidate_rows": reused_rows, "elapsed_seconds": time.monotonic()-started})
            engine.close(); engine = None
            budget.check()
            for relative, expected in ctx["hashes"].items():
                if file_digest(resolve(relative)) != expected:
                    raise ValueError(f"Source changed during evaluation: {relative}")
            total_expected = 3 * sum(COUNTS.values()) + len(DIAGNOSTIC) * N_DIAGNOSTIC * len(COUNTS)
            if fresh_rows + reused_rows != total_expected or len(outputs) != len(COUNTS) * (1 + len(PRIMARY) + len(DIAGNOSTIC)):
                raise ValueError("Incomplete registered evaluation matrix")
            runtime = {"status": "complete", "at": utcnow(), "variant": variant, "elapsed_seconds_including_load_and_close": time.monotonic()-started,
                       "model_load_seconds": model_seconds, "generation_batches": steps, "fresh_generated_rows": fresh_rows,
                       "reused_candidate_rows": reused_rows, "prediction_rows": total_expected,
                       "minimum_available_gib": budget.minimum_available_gib, **budget.latest_resources,
                       "cost_note": "Shared original forced-candidate seconds are provenance, not new generation costs; the root job ledger counts actual elapsed attempts."}
            write_json(output_dir / "runtime.json", runtime); outputs["runtime.json"] = file_digest(output_dir / "runtime.json")
            completion = {**runtime, "outputs": outputs, "populations": populations,
                          "evaluation_lock_sha256": file_digest(ctx["lock_path"]), "contract_sha256": ctx["lock"]["contract_sha256"],
                          "adapter_sha256": None if variant == "original" else ctx["lock"]["rl_completed"][variant]["adapter_sha256"],
                          "script_sha256": file_digest(Path(__file__)), "source_sha256": ctx["hashes"],
                          "primary_thresholds": list(PRIMARY), "diagnostic_thresholds": list(DIAGNOSTIC),
                          "shared_original_forced_candidates": variant == "original", "weights_updated": False}
            write_json(output_dir / "complete.json", completion)
            return completion
        except BaseException as exc:
            write_json(output_dir / "failure.json", {"status": "failed", "at": utcnow(), "error": repr(exc),
                       "variant": variant, "elapsed_seconds": time.monotonic()-started, "generation_batches": steps,
                       "fresh_generated_rows": fresh_rows, "reused_candidate_rows": reused_rows})
            raise
        finally:
            if engine is not None:
                engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["original", *sorted(VARIANTS)])
    parser.add_argument("--evaluation-lock", required=True, type=Path)
    parser.add_argument("--max-seconds", required=True, type=float)
    args = parser.parse_args()
    result = evaluate(args.evaluation_lock, args.variant, args.max_seconds)
    print({"variant": args.variant, "status": result["status"], "prediction_rows": result["prediction_rows"]})
