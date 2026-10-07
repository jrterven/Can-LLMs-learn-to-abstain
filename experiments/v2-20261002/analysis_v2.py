"""CPU-only registered v2 analysis, after every direct and selector test completes.

No fitting, checkpoint choice, prediction generation, or manuscript modification.
Five paired contrasts have ordinary 95% and Bonferroni 99% percentile intervals.
The same question draw and training-seed draw are used for both thresholds and
both methods of a contrast. Deterministic filters are shared, not extra seeds.
"""
from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
ART = WORK / "artifacts"
sys.path.insert(0, str(ROOT / "src"))
from abstention.io import digest, file_digest, read_json, read_jsonl, utcnow, write_json
from abstention.scoring import grade, reward, summarize

SEEDS = (17, 29, 43)
ARMS = ("a_g4_single_tau", "b_g8_single_tau", "c_g8_paired_tau", "d_g8_paired_fixed075")
VARIANTS = ("original", *(f"{arm}-s{seed}" for arm in ARMS for seed in SEEDS))
PRIMARY = (.65, .85)
DIAGNOSTIC = (.60, .75, .90, .95)
COUNTS = {"test_trivia": 2000, "test_nq": 500}
FILTERS = ("original_raw", "original_calibrated", "train_logistic_raw",
           "train_logistic_recalibrated", "critic_raw", "critic_calibrated")
DETERMINISTIC = FILTERS[:4]
CONTRASTS = ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[1]), (ARMS[2], ARMS[3]),
             ("critic_calibrated", "original_calibrated"),
             ("critic_calibrated", "train_logistic_recalibrated"))
LABELS = {"original": "Original prompt", ARMS[0]: "A: 2×G4, single τ",
          ARMS[1]: "B: G8, single τ", ARMS[2]: "C: G8, paired τ",
          ARMS[3]: "D: G8, fixed cost", "original_raw": "Original P(True)",
          "original_calibrated": "Original + calibration",
          "train_logistic_raw": "TRAIN logistic", "train_logistic_recalibrated": "TRAIN logistic + calibration",
          "critic_raw": "Learned selector", "critic_calibrated": "Learned selector + calibration"}
BOOTSTRAPS, BOOTSTRAP_SEED = 10000, 20261002


def import_local(filename):
    spec = importlib.util.spec_from_file_location("v2_analysis_" + filename[:-3], WORK / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def attest(path, expected, hashes):
    actual = file_digest(path)
    key = relative(path)
    if actual != expected or (key in hashes and hashes[key] != actual):
        raise ValueError(f"Changed/conflicting input: {key}")
    hashes[key] = actual


def direct_names():
    return {f"{split}-forced.jsonl" for split in COUNTS} | {
        f"{split}-t{tau:.2f}.jsonl" for split in COUNTS for tau in (*PRIMARY, *DIAGNOSTIC)}


def receipt_barrier(art=ART):
    """Check all 16 completion receipts before reading any prediction rows."""
    hashes, receipts = {}, {}
    lock_path = art / "evaluation-lock.json"
    lock_sha = file_digest(lock_path)
    attest(lock_path, lock_sha, hashes)
    lock = read_json(lock_path)
    if lock.get("status") != "ready":
        raise ValueError("No ready evaluation lock")
    paths = [art / "predictions" / name / "complete.json" for name in VARIANTS]
    paths += [art / "selector/test-evaluation" / f"s{s}" / "complete.json" for s in SEEDS]
    # Missing later runs must fail before parsing earlier scientific outputs.
    if any(not p.is_file() for p in paths):
        raise ValueError("All thirteen direct and three selector test receipts are required")
    for path in paths:
        done = read_json(path)
        if done.get("status") != "complete" or done.get("evaluation_lock_sha256") != lock_sha:
            raise ValueError("Incomplete or unrelated test receipt")
        attest(path, file_digest(path), hashes)
        if path.parent.parent.name == "predictions":
            name = path.parent.name
            if done.get("variant") != name or done.get("contract_sha256") != lock["contract_sha256"]:
                raise ValueError("Wrong direct variant/contract")
            expected = direct_names() | {"runtime.json"}
            mapping = done["outputs"]
            if (set(mapping) != expected or set(done["populations"]) != direct_names()
                    or done.get("primary_thresholds") != list(PRIMARY)
                    or done.get("diagnostic_thresholds") != list(DIAGNOSTIC)):
                raise ValueError("Incomplete direct output grid")
            for split, count in COUNTS.items():
                for tau in (None, *PRIMARY, *DIAGNOSTIC):
                    filename = f"{split}-forced.jsonl" if tau is None else f"{split}-t{tau:.2f}.jsonl"
                    if done["populations"][filename] != (count if tau is None or tau in PRIMARY else 200):
                        raise ValueError("Incomplete direct population")
            if done.get("prediction_rows") != 9100 or done.get("weights_updated") is not False:
                raise ValueError("Unexpected evaluation size or weight update")
            receipts[name] = done
        else:
            mapping = done["files"]
            if set(mapping) != {"test_trivia-scores.json", "test_nq-scores.json", "results.json"}:
                raise ValueError("Incomplete selector output grid")
            receipts[path.parent.name] = done
        for filename, checksum in mapping.items():
            if Path(filename).name != filename:
                raise ValueError("Receipt contains a nonlocal output path")
            attest(path.parent / filename, checksum, hashes)
    return lock, hashes, receipts


def validated_context():
    lock, hashes, receipts = receipt_barrier()
    evaluator = import_local("evaluate_v2.py")
    ctx = evaluator.check_lock(ART / "evaluation-lock.json", "original")
    for path, checksum in ctx["hashes"].items():
        attest(ROOT / path, checksum, hashes)
    if ctx["config"]["evaluation"]["bootstrap_samples"] != BOOTSTRAPS or ctx["config"]["evaluation"]["bootstrap_seed"] != BOOTSTRAP_SEED:
        raise ValueError("Bootstrap settings changed")
    for name, done in receipts.items():
        if name in VARIANTS:
            if done["script_sha256"] != file_digest(WORK / "evaluate_v2.py"):
                raise ValueError("Evaluator source mismatch")
            expected = None if name == "original" else lock["rl_completed"][name]["adapter_sha256"]
            if done["adapter_sha256"] != expected:
                raise ValueError("Evaluation uses another adapter")
            for path, checksum in done["source_sha256"].items():
                attest(evaluator.resolve(path), checksum, hashes)
        elif done["identity_sha256"] != ctx["selector_ctx"]["sha"]:
            raise ValueError("Selector identity mismatch")
    ctx.update(hashes=hashes, receipts=receipts)
    return ctx


def check_rows(rows, references, variant, split, tau):
    if [r["id"] for r in rows] != [r["id"] for r in references]:
        raise ValueError("Wrong or repeated question population/order")
    arm, seed = ("original", 0) if variant == "original" else (variant.rsplit("-s", 1)[0], int(variant.rsplit("-s", 1)[1]))
    for row, ref in zip(rows, references):
        body = {k: v for k, v in row.items() if k != "row_sha256"}
        if digest(body) != row["row_sha256"] or any(row.get(k) != v for k, v in ref.items()):
            raise ValueError("Changed row/reference")
        expected = {"variant": variant, "variant_id": variant, "seed": seed, "arm": arm,
                    "model": "qwen14b", "evaluation_split": split, "tau": tau,
                    "mode": "forced" if tau is None else "threshold"}
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("Wrong prediction condition")
        if not isinstance(row["text"], str) or type(row["truncated"]) is not bool:
            raise ValueError("Invalid text/truncation")
        if type(row["generated_tokens"]) is not int or not 0 <= row["generated_tokens"] <= 32:
            raise ValueError("Invalid token count")
        graded = grade(row["text"], ref["aliases"], row["truncated"])
        if (row["outcome"], row["reason"]) != (graded.outcome, graded.reason):
            raise ValueError("Strict exact-match regrade disagrees")
    return rows


def sigmoid(z):
    if not math.isfinite(z):
        raise ValueError("Nonfinite logit")
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def eligible(row):
    return bool(row["text"].strip()) and not row["truncated"] and row["text"].strip() != "IDK"


def filtered(row, method, tau):
    emitted = eligible(row) and sigmoid(row["logits"][method]) > tau
    return {"id": row["id"], "tau": tau, "outcome": row["outcome"] if emitted else "abstain",
            "reason": row["reason"] if emitted else "confidence_gate", "candidate_invalid": row["reason"] in {"empty", "truncated"}}


def check_selector_rows(rows, references, original, split):
    if [r["id"] for r in rows] != [r["id"] for r in references]:
        raise ValueError("Wrong selector population/order")
    for row, ref, candidate in zip(rows, references, original):
        if (row["question"] != ref["question"] or row["split"] != split or row["candidate_index"] != 0
                or row["generator"] != "original_frozen" or type(row["truncated"]) is not bool):
            raise ValueError("Wrong selector candidate identity")
        if any(row[k] != candidate[k] for k in ("text", "truncated", "outcome", "reason", "generated_tokens")):
            raise ValueError("Filters and original do not share the same candidate")
        if row["y"] != int(candidate["outcome"] == "correct") or set(row["logits"]) != set(FILTERS):
            raise ValueError("Wrong selector labels/method grid")
        if row["original_logit"] != row["logits"]["original_raw"]:
            raise ValueError("Original score mismatch")
        for logit in row["logits"].values():
            sigmoid(logit)
    return rows


def deterministic_signature(rows):
    # Exclude only stochastic critic scores; candidate/provenance must be identical.
    return digest([{**{k: v for k, v in row.items() if k != "logits"},
                    "logits": {m: row["logits"][m] for m in DETERMINISTIC}} for row in rows])


def load_predictions(ctx):
    policies, forced, scores = {}, {}, {}
    for variant in VARIANTS:
        method, seed = ("original", 0) if variant == "original" else (variant.rsplit("-s", 1)[0], int(variant.rsplit("-s", 1)[1]))
        for split, refs in ctx["data"].items():
            for tau in (None, *PRIMARY, *DIAGNOSTIC):
                refs_now = refs if tau is None or tau in PRIMARY else [r for r in refs if r["id"] in ctx["diagnostic"][split]]
                filename = f"{split}-forced.jsonl" if tau is None else f"{split}-t{tau:.2f}.jsonl"
                rows = check_rows(read_jsonl(ART / "predictions" / variant / filename), refs_now, variant, split, tau)
                if tau is None:
                    forced[split, method, seed] = rows
                else:
                    if tau in PRIMARY:
                        policies[split, method, seed, "primary", tau] = rows
                    policies[split, method, seed, "diagnostic", tau] = [r for r in rows if r["id"] in ctx["diagnostic"][split]]
    for split, refs in ctx["data"].items():
        original = forced[split, "original", 0]
        expected_signature = None
        for seed in SEEDS:
            path = ART / "selector/test-evaluation" / f"s{seed}" / f"{split}-scores.json"
            rows = ctx["selector"].read_envelope(path, ctx["selector_ctx"], split=split, seed=seed)
            check_selector_rows(rows, refs, original, split)
            signature = deterministic_signature(rows)
            if expected_signature is not None and signature != expected_signature:
                raise ValueError("Deterministic baselines/candidates differ between selector seeds")
            expected_signature = signature
            for method in FILTERS:
                score_seed = 0 if method in DETERMINISTIC else seed
                if (split, method, score_seed) in scores:
                    continue
                scores[split, method, score_seed] = rows
                for tau in (*PRIMARY, *DIAGNOSTIC):
                    full = [filtered(r, method, tau) for r in rows]
                    if tau in PRIMARY:
                        policies[split, method, score_seed, "primary", tau] = full
                    policies[split, method, score_seed, "diagnostic", tau] = [r for r in full if r["id"] in ctx["diagnostic"][split]]
    return policies, forced, scores


def metric_record(rows, **identity):
    result = summarize(rows)
    return {**identity, **result, "explicit_idk": sum(r.get("reason") == "explicit_idk" for r in rows),
            "empty": sum(r.get("reason") == "empty" for r in rows),
            "truncated": sum(r.get("reason") == "truncated" for r in rows),
            "candidate_invalid": sum(r.get("candidate_invalid", False) for r in rows)}


def method_seeds(method):
    return (0,) if method == "original" or method in DETERMINISTIC else SEEDS


def policy_metrics(policies):
    cells = [metric_record(rows, split=split, method=method, seed=seed, scope=scope, tau=tau)
             for (split, method, seed, scope, tau), rows in sorted(policies.items())]
    aggregate = []
    for split, method, scope, tau in sorted({(s, m, sc, t) for s, m, _, sc, t in policies}):
        seeds = method_seeds(method)
        groups = [policies[split, method, seed, scope, tau] for seed in seeds]
        aggregate.append(metric_record([row for rows in groups for row in rows], split=split, method=method,
                         scope=scope, tau=tau, n_questions=len(groups[0]), n_seeds=len(seeds), seed_ids=list(seeds)))
    joint = []
    for split, method in sorted({(s, m) for s, m, _, _, _ in policies}):
        seeds = method_seeds(method)
        rows = [r for seed in seeds for tau in PRIMARY for r in policies[split, method, seed, "primary", tau]]
        joint.append(metric_record(rows, split=split, method=method, scope="primary_two_thresholds",
                                  n_questions=len(policies[split, method, seeds[0], "primary", PRIMARY[0]]),
                                  n_seeds=len(seeds), thresholds=list(PRIMARY), seed_ids=list(seeds)))
    return cells, aggregate, joint


def paired_differences(policies, split, left, right):
    arrays = []
    for method in (left, right):
        by_seed = []
        expected_ids = None
        for seed in SEEDS:
            actual_seed = 0 if method in DETERMINISTIC else seed
            threshold_values = []
            for tau in PRIMARY:
                rows = policies[split, method, actual_seed, "primary", tau]
                ids = [r["id"] for r in rows]
                if expected_ids is not None and ids != expected_ids:
                    raise ValueError("Unpaired thresholds/seeds/questions")
                expected_ids = ids
                threshold_values.append([reward(r["outcome"], tau) for r in rows])
            by_seed.append(np.mean(threshold_values, axis=0))
        arrays.append((expected_ids, np.asarray(by_seed)))
    if arrays[0][0] != arrays[1][0]:
        raise ValueError("Unpaired methods/questions")
    return arrays[0][1] - arrays[1][1]


def crossed_bootstrap(differences, n_boot=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] < 1 or not np.isfinite(values).all() or n_boot < 1:
        raise ValueError("Expected three paired seeds × finite nonempty questions")
    rng = np.random.default_rng(seed)
    n_seed, n_question = values.shape
    estimates = np.empty(n_boot)
    # Draw each replicate separately to make output invariant to memory batching.
    for i in range(n_boot):
        selected_seeds = rng.integers(0, n_seed, n_seed)
        selected_questions = rng.integers(0, n_question, n_question)
        estimates[i] = values[selected_seeds][:, selected_questions].mean()
    return {"effect": float(values.mean()), "ci95_low": float(np.quantile(estimates, .025)),
            "ci95_high": float(np.quantile(estimates, .975)), "ci99_low": float(np.quantile(estimates, .005)),
            "ci99_high": float(np.quantile(estimates, .995)),
            "per_seed": {str(s): float(v.mean()) for s, v in zip(SEEDS, values)},
            "n_questions": n_question, "paired_seeds": list(SEEDS), "bootstrap_replicates": n_boot,
            "bootstrap_seed": seed, "family_size": 5, "thresholds": list(PRIMARY)}


def contrast_metrics(policies):
    result = []
    for split in COUNTS:
        for left, right in CONTRASTS:
            result.append({"split": split, "left": left, "right": right,
                           "role": "primary_family" if split == "test_trivia" else "descriptive_transfer",
                           "left_n_seeds": len(method_seeds(left)), "right_n_seeds": len(method_seeds(right)),
                           **crossed_bootstrap(paired_differences(policies, split, left, right))})
    return result


def calibration_metrics(rows, method, scope):
    selected = rows if scope == "all" else [r for r in rows if eligible(r)]
    if scope not in {"all", "substantive"}:
        raise ValueError("Unknown calibration scope")
    if not selected:
        return {"n": 0, "positives": 0, "brier": None, "log_loss": None, "auc": None, "ece10": None, "bins": []}
    from sklearn.metrics import roc_auc_score
    z = np.array([r["logits"][method] for r in selected])
    p = np.array([sigmoid(v) for v in z])
    y = np.array([r["y"] for r in selected])
    bins = []
    # [0,.1),...,[.9,1], including p=1 in the last bin.
    indices = np.minimum((p * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        here = indices == index
        n = int(here.sum())
        confidence = float(p[here].mean()) if n else None
        accuracy = float(y[here].mean()) if n else None
        if n:
            ece += n / len(y) * abs(confidence - accuracy)
        bins.append({"index": index, "low": index / 10, "high": (index + 1) / 10,
                     "n": n, "confidence": confidence, "accuracy": accuracy})
    return {"n": len(y), "positives": int(y.sum()), "brier": float(np.mean((p-y)**2)),
            "log_loss": float(np.mean(np.logaddexp(0, z) - y*z)),
            "auc": float(roc_auc_score(y, z)) if len(set(y)) == 2 else None, "ece10": ece, "bins": bins}


def risk_coverage(rows, method):
    """Rank the shared forced candidates; inseparable ties enter together.

    Invalid/IDK candidates never enter. Coverage denominator is ALL questions.
    These are ranking curves, not curves over the model's prompted thresholds.
    """
    selected = sorted((r for r in rows if eligible(r)), key=lambda r: r["logits"][method], reverse=True)
    curve = [{"emitted": 0, "errors": 0, "coverage": 0., "risk": None, "score": None}]
    total = errors = 0
    index = 0
    while index < len(selected):
        score = selected[index]["logits"][method]
        stop = index
        while stop < len(selected) and selected[stop]["logits"][method] == score:
            errors += int(selected[stop]["outcome"] == "error")
            total += 1; stop += 1
        curve.append({"emitted": total, "errors": errors, "coverage": total / len(rows),
                      "risk": errors / total, "score": score})
        index = stop
    return curve


def forced_and_response_diagnostics(forced, policies):
    forced_metrics, changes = [], []
    for (split, method, seed), rows in sorted(forced.items()):
        result = metric_record([{**r, "tau": .75} for r in rows], split=split, method=method, seed=seed)
        result.pop("utility")  # Forced accuracy has no requested decision threshold.
        forced_metrics.append(result)
        low = policies[split, method, seed, "primary", PRIMARY[0]]
        high = policies[split, method, seed, "primary", PRIMARY[1]]
        ml, mh = summarize(low), summarize(high)
        changes.append({"split": split, "method": method, "seed": seed, "n_questions": len(low),
                        "coverage_high_minus_low": mh["coverage"]-ml["coverage"],
                        "risk_high_minus_low": None if ml["selective_risk"] is None or mh["selective_risk"] is None else mh["selective_risk"]-ml["selective_risk"],
                        "answered_low_abstain_high": sum(a["outcome"] != "abstain" and b["outcome"] == "abstain" for a, b in zip(low, high)),
                        "abstain_low_answered_high": sum(a["outcome"] == "abstain" and b["outcome"] != "abstain" for a, b in zip(low, high)),
                        "idk_in_forced_and_both_thresholds": sum(a["outcome"] == b["outcome"] == c["outcome"] == "abstain" for a, b, c in zip(rows, low, high))})
    # Every diagnostic point is measured on the same fixed 200 questions.
    for (split, method, seed, scope, tau), rows in policies.items():
        if scope == "diagnostic" and len(rows) != 200:
            raise ValueError("Diagnostic panel must contain exactly 200 fixed questions")
    return forced_metrics, changes


def rollout_diagnostics(rows, group_size, total_steps):
    """Credit activity is descriptive, not gradient SNR or evidence of causality."""
    if group_size not in (4, 8) or not rows or len(rows) % 8:
        raise ValueError("Expected complete eight-sample exposure blocks")
    groups, prompt_mixed, active_updates = [], 0, set()
    for start in range(0, len(rows), 8):
        block = rows[start:start+8]
        if (len({(r["step"], r["id"], r["exposure_id"], r["tau"], r["slot"]) for r in block}) != 1
                or [r["sample_index"] for r in block] != list(range(8))):
            raise ValueError("Exposure identity or eight-draw ordering changed")
        outcomes = {r["outcome"] for r in block}
        prompt_mixed += int("abstain" in outcomes and len(outcomes) > 1)
    for start in range(0, len(rows), group_size):
        block = rows[start:start+group_size]
        if len({r["baseline_group_id"] for r in block}) != 1 or len({r["exposure_id"] for r in block}) != 1:
            raise ValueError("Baseline crosses an exposure or credit group")
        for r in block:
            scored = grade(r["text"], r["aliases"], r["truncated"])
            expected_reward = reward(scored.outcome, r["tau"], r["arm"])
            if (scored.outcome, scored.reason) != (r["outcome"], r["reason"]) or not math.isclose(r["reward"], expected_reward, abs_tol=1e-9):
                raise ValueError("Rollout strict reward/grade mismatch")
            expected_group = digest([r["step"], r["exposure_id"], r["sample_index"] // group_size])
            if (r["group_size"] != group_size or r["baseline_group_id"] != expected_group
                    or r["baseline_group_within_prompt"] != r["sample_index"] // group_size):
                raise ValueError("Wrong registered credit group")
            expected_credit = r["reward"] - (sum(x["reward"] for x in block) - r["reward"]) / (group_size-1)
            if not math.isclose(r["loo_advantage"], expected_credit, abs_tol=1e-6, rel_tol=1e-6):
                raise ValueError("Recorded LOO advantage mismatch")
        outcomes = {r["outcome"] for r in block}
        active = max(r["reward"] for r in block) - min(r["reward"] for r in block) > 1e-9
        if active:
            active_updates.add(block[0]["step"])
        groups.append({"active": active, "mixed": "abstain" in outcomes and len(outcomes) > 1,
                       "composition": "".join(c for name, c in (("correct", "C"), ("error", "E"), ("abstain", "A")) if name in outcomes)})
    n = len(rows)
    result = {"completions": n, "credit_group_size": group_size, "credit_groups": len(groups),
              "constant_credit_groups": sum(not g["active"] for g in groups),
              "mixed_answer_idk_credit_groups": sum(g["mixed"] for g in groups),
              "eight_draw_exposures": n//8, "mixed_answer_idk_eight_draw_exposures": prompt_mixed,
              "updates_in_scope": len({r["step"] for r in rows}), "active_reward_updates": len(active_updates),
              "registered_total_updates": total_steps,
              "loo_advantage_rms": math.sqrt(sum(r["loo_advantage"]**2 for r in rows)/n),
              "generated_tokens": sum(len(r["completion_ids"]) for r in rows)}
    for name in ("correct", "error", "abstain"):
        result[name] = sum(r["outcome"] == name for r in rows)
        result[f"{name}_fraction"] = result[name]/n
    for composition in ("C", "E", "A", "CE", "CA", "EA", "CEA"):
        result[f"groups_{composition}"] = sum(g["composition"] == composition for g in groups)
    result.update(empty=sum(r["reason"] == "empty" for r in rows), truncated=sum(r["reason"] == "truncated" for r in rows))
    return result


def training_diagnostics(ctx):
    trainer, data = import_local("training_v2.py"), import_local("prepare_data.py")
    runs, panels, updates = [], [], []
    for variant, locked in sorted(ctx["lock"]["rl_completed"].items()):
        directory = (ROOT / locked["result_path"]).parent
        result = read_json(directory / "result.json")
        spec = trainer.V2Spec(**result["spec"])
        arm = variant.rsplit("-s", 1)[0]
        schedule = data.training_rows(arm, spec.seed, n_questions=result["questions"])
        records = trainer.prepared_records(schedule, spec)
        path = directory / "rollouts.jsonl"
        attest(path, result["rollouts_sha256"], ctx["hashes"])
        rows = read_jsonl(path)
        trainer.verify_rollouts(rows, records, spec, result["optimizer_steps"])
        expected = {r["exposure_id"]: r for r in records}
        for row in rows:
            source = expected[row["exposure_id"]]
            if any(row[k] != source[k] for k in ("id", "question", "aliases", "tau", "slot", "question_index", "update_index", "v2_prompt_id")):
                raise ValueError("Rollout reference/exposure differs from registered TRAIN")
            if row["arm"] != spec.arm or row["step"] != row["update_index"]:
                raise ValueError("Rollout arm/optimizer update mismatch")
        for filename in ("run.json", "steps.jsonl", "logs.jsonl"):
            path = directory / filename
            attest(path, file_digest(path), ctx["hashes"])
        provenance = read_json(directory / "run.json")
        if provenance["schedule_sha256"] != digest(schedule) or provenance["spec"] != result["spec"]:
            raise ValueError("Training schedule provenance changed")
        steps = read_jsonl(directory / "steps.jsonl")
        if [r["step"] for r in steps] != list(range(1, result["optimizer_steps"]+1)):
            raise ValueError("Missing/repeated training step timing")
        logs = {}
        for log in read_jsonl(directory / "logs.jsonl"):
            if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in log.values()):
                raise ValueError("Nonfinite recorded training log")
            logs.setdefault(log["step"], {}).update(log)
        identity = {"variant": variant, "arm": arm, "seed": spec.seed}
        runs.append({**identity, **rollout_diagnostics(rows, spec.group_size, result["optimizer_steps"])})
        for scope, choices in (("tau", (.6, .75, .9)), ("quartile", range(1, 5))):
            for value in choices:
                chosen = [r for r in rows if (r["tau"] == value if scope == "tau" else min(3, 4*r["step"]//result["optimizer_steps"])+1 == value)]
                panels.append({**identity, "scope": scope, "value": value,
                               **rollout_diagnostics(chosen, spec.group_size, result["optimizer_steps"])})
        for step in steps:
            if any(not math.isfinite(step[k]) or step[k] < 0 for k in ("seconds", "available_gib", "cuda_allocated_gib", "cuda_reserved_gib")):
                raise ValueError("Invalid step timing/memory")
            selected = rows[(step["step"]-1)*48:step["step"]*48]
            credit = rollout_diagnostics(selected, spec.group_size, result["optimizer_steps"])
            updates.append({**identity, **step, **credit,
                            **{k: v for k, v in logs.get(step["step"], {}).items() if k in {"kl", "loss", "grad_norm", "learning_rate", "v2/loo_advantage_rms"}}})
    return {"runs": runs, "panels": panels, "updates": updates,
            "limits": "Descriptive on-policy trajectories, not gradient signal-to-noise or causal mediation. Active means nonconstant raw reward in at least one credit group; KL can still generate gradients in reward-constant updates. G4 has twice as many credit groups; the eight-draw exposure denominator is matched. Quartiles are optimizer-update quartiles within the fixed one-pass schedule."}


def cost_snapshot():
    rows, hashes, excluded = [], {}, []
    for path in sorted((ART / "jobs").glob("*/job.json")):
        job = read_json(path)
        if job["status"] == "running":
            if job["id"] != "analysis":
                raise ValueError("Another job is running during analysis")
            excluded.append(job["id"])
            continue
        if job["status"] not in {"complete", "failed"} or not math.isfinite(job["seconds"]) or job["seconds"] < 0:
            raise ValueError("Invalid authoritative job ledger")
        attest(path, file_digest(path), hashes)
        rows.append({"job": job["id"], "status": job["status"], "seconds": job["seconds"], "hours": job["seconds"] / 3600,
                     "source": relative(path), "source_sha256": hashes[relative(path)]})
    return {"closed_jobs": rows, "closed_attempt_seconds": sum(r["seconds"] for r in rows),
            "excluded_running_jobs": excluded,
            "scope": "Closed root job attempts at analysis entry, including pilots/failures/loading. The current CPU analysis and later tasks are not yet final costs. Nested selector ledgers and model timings are NOT added."}, hashes


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refuse an empty table")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def figures(output, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ("original", *ARMS, "original_calibrated", "train_logistic_recalibrated", "critic_calibrated")
    captions = {}
    for metric, ylabel in (("utility", "Mean utility"), ("coverage", "Coverage"), ("selective_risk", "Selective risk")):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, split in zip(axes, COUNTS):
            values = {r["method"]: r[metric] for r in stats["joint_primary"] if r["split"] == split}
            for x, method in enumerate(order):
                val = values[method]
                if val is not None:
                    ax.plot(x, val, "o", color="black", markersize=6)
                for i, seed in enumerate(method_seeds(method)):
                    cells = [r for r in stats["metrics"] if r["split"] == split and r["method"] == method and r["seed"] == seed and r["scope"] == "primary"]
                    if metric == "selective_risk":
                        answered = sum(r["correct"] + r["error"] for r in cells)
                        value = sum(r["error"] for r in cells) / answered if answered else None
                    else:
                        value = sum(r[metric] for r in cells) / len(cells)
                    if value is not None:
                        ax.plot(x + (i-1)*.08 if seed else x, value, "x", markersize=5, color=f"C{i}")
            ax.set_xticks(range(len(order)), [LABELS[m] for m in order], rotation=55, ha="right")
            ax.set_title(f"{split}: {COUNTS[split]} questions")
            ax.grid(axis="y", alpha=.2)
            if metric == "utility":
                ax.axhline(0, color="grey", linewidth=.7)
            else:
                ax.set_ylim(-.02, 1.02)
        axes[0].set_ylabel(ylabel)
        fig.suptitle("Primary τ = 0.65 and 0.85; pooled outcomes; black = aggregate, × = individual seed")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{metric}.{ext}", dpi=180)
        plt.close(fig)
        captions[metric] = "Black dots pool all registered seeds and both primary thresholds. Colored crosses are seeds 17/29/43 (blue/orange/green); deterministic methods have one score, seed 0. Selective risk is a ratio of pooled errors to emitted responses; undefined values are omitted. These are point estimates, not confidence intervals; the contrast table gives the registered intervals."
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, split in zip(axes, COUNTS):
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=.7)
        for method, color in (("original_raw", "C0"), ("original_calibrated", "C1"), ("critic_calibrated", "C2")):
            for seed in method_seeds(method):
                rec = next(r for r in stats["calibration"] if r["split"] == split and r["method"] == method and r["seed"] == seed and r["scope"] == "substantive")
                bins = [b for b in rec["bins"] if b["n"]]
                ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o", markersize=3,
                        color=color, linestyle={0:"-", 17:"-", 29:"--", 43:":"}[seed], label=f"{LABELS[method]} (s{seed})")
        ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025), title=split, xlabel="Mean predicted probability")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Exact-match fraction")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"calibration.{ext}", dpi=180)
    plt.close(fig)
    captions["calibration"] = "Ten fixed equal-width bins; valid substantive shared candidates only. Empty bins omitted; counts in calibration-bins.csv. Lines join bin means, without confidence bands; calibration was fitted on separate TriviaQA calibration data and transferred unchanged to NQ."
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, split in zip(axes, COUNTS):
        for method, color in (("original_calibrated", "C0"), ("train_logistic_recalibrated", "C1"), ("critic_calibrated", "C2")):
            for seed in method_seeds(method):
                rec = next(r for r in stats["risk_coverage"] if r["split"] == split and r["method"] == method and r["seed"] == seed)
                points = [p for p in rec["points"] if p["risk"] is not None]
                ax.plot([p["coverage"] for p in points], [p["risk"] for p in points], color=color,
                        linestyle={0:"-", 17:"-", 29:"--", 43:":"}[seed], label=f"{LABELS[method]} (s{seed})")
        ax.set(xlim=(-.015, 1.015), ylim=(-.015, 1.015), title=split, xlabel="Coverage over all questions")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Selective risk")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"risk-coverage.{ext}", dpi=180)
    plt.close(fig)
    captions["risk-coverage"] = "Ranked shared forced candidates, not prompted threshold policies. Ties enter together; invalid/IDK candidates never enter. Coverage denominator is all questions and zero-coverage risk is undefined. Lines only connect observed attainable tie-group endpoints. Every selector seed is shown; deterministic filters are shown once."
    write_json(output / "captions.json", captions)


def markdown(stats):
    lines = ["# V2: resultados reservados completos", "", "Resultados descriptivos de todos los brazos y semillas registrados. Este archivo no declara automáticamente éxito ni modifica el manuscrito.", "",
             "| Dataset | Contraste de utilidad | Efecto | IC95 descriptivo | IC99 Bonferroni |", "|---|---|---:|---|---|"]
    for r in stats["contrasts"]:
        lines.append(f"| {r['split']} | {r['left']} − {r['right']} | {r['effect']:.6f} | [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}] | [{r['ci99_low']:.6f}, {r['ci99_high']:.6f}] |")
    lines += ["", "La familia principal son los cinco contrastes en TriviaQA; NQ es transferencia descriptiva. Se promedian τ=0.65/0.85 por pregunta antes de remuestrear preguntas y semillas emparejadas. Los filtros deterministas son un único baseline compartido. Los IC99 tienen cobertura familiar nominal 95% por Bonferroni, sujeta a la aproximación bootstrap; no se calcularon p-valores. Tres semillas limitan la inferencia entre entrenamientos; no se remuestreó la muestra de calibración.", "",
              "Las tablas metrics.csv y joint-primary.csv distinguen preguntas únicas, semillas y filas de predicción. Riesgo es errores/respuestas, indefinido sin cobertura. Menor cobertura no acredita por sí sola mejor selección. forced.csv y threshold-response.csv registran retención, IDK persistente y cambios ante el umbral; los seis puntos diagnósticos usan las mismas 200 preguntas por dataset.", "",
              "calibration.csv separa todas las candidatas de las sustantivas válidas. Una menor ECE/Brier no garantiza mejor ordenación ni utilidad. risk-coverage.csv ordena candidatas compartidas y conserva empates completos: no describe las políticas de prompting. La calibración de TriviaQA se aplica a NQ sin ajustes. Exact match sigue siendo un proxy, no verdad semántica auditada.", "",
              f"Costo cerrado al iniciar este análisis: {stats['cost']['closed_attempt_seconds']/3600:.6f} horas de tareas v2. Incluye pilotos/fallos/carga registrados por el supervisor; excluye el análisis CPU en curso y tareas posteriores. No se suman otra vez los ledgers anidados. La cifra final se obtiene del ledger autoritativo cuando cierre el job de análisis.", "",
              "La incertidumbre del selector está condicionada a un único banco de candidatas TRAIN compartido y a la partición fija; no representa regenerar ese banco. El control logístico iguala etiquetas, no capacidad paramétrica ni cómputo. C−B cambia la exposición y la mezcla de umbrales dentro del batch; no identifica por sí solo comprensión del umbral. No existe un brazo G4 con exposición emparejada para estimar esa interacción factorial. Los diagnósticos de actividad de recompensa no equivalen a medir señal/ruido de gradientes ni establecen mediación causal.", "",
              "Las figuras requieren inspección visual y las conclusiones requieren revisión científica. No se han añadido etiquetas humanas ni actualizado el manuscrito.", ""]
    return "\n".join(lines)


def main():
    output = ART / "analysis"
    if output.exists():
        raise FileExistsError("Analysis output exists; no overwrite or silent partial recovery")
    ctx = validated_context()
    policies, forced, scores = load_predictions(ctx)
    cells, aggregate, joint = policy_metrics(policies)
    force_metrics, changes = forced_and_response_diagnostics(forced, policies)
    calibration, curves = [], []
    for (split, method, seed), rows in sorted(scores.items()):
        for scope in ("all", "substantive"):
            calibration.append({"split": split, "method": method, "seed": seed, "scope": scope,
                                **calibration_metrics(rows, method, scope)})
        curves.append({"split": split, "method": method, "seed": seed, "n_questions": len(rows),
                       "points": risk_coverage(rows, method)})
    costs, cost_hashes = cost_snapshot()
    ctx["hashes"].update(cost_hashes)
    training = training_diagnostics(ctx)
    stats = {"created_at": utcnow(), "status": "complete", "methods": LABELS, "seeds": list(SEEDS),
             "primary_thresholds": list(PRIMARY), "diagnostic_thresholds": list(DIAGNOSTIC),
             "metrics": cells, "aggregate_metrics": aggregate, "joint_primary": joint,
             "contrasts": contrast_metrics(policies), "forced": force_metrics,
             "threshold_response": changes, "calibration": calibration, "risk_coverage": curves, "cost": costs,
             "training_diagnostics": training,
             "inference": {"bootstrap": "paired crossed question × training seed; thresholds averaged within question first",
                           "calibration_uncertainty": "conditional on the fitted calibrators; calibration rows not resampled",
                           "selector_uncertainty": "conditional on one shared TRAIN candidate bank and fixed training/calibration partitions; candidate-regeneration uncertainty not included",
                           "selector_control": "TRAIN logistic matches label budget, not parameter capacity or compute",
                           "exposure_interpretation": "C−B changes within-update threshold exposure/balance; no G4-paired arm to estimate a full factorial interaction",
                           "probability_scoring": "Brier/ECE use sigmoid probabilities; AUC uses logits to preserve ordering; log loss uses stable logaddexp without probability clipping",
                           "family": "five prespecified TriviaQA contrasts; ordinary95 descriptive and nominal Bonferroni99",
                           "transfer": "NQ is descriptive", "test_used_for_selection": False},
             "source_sha256": ctx["hashes"]}
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output / "statistics.json", stats)
        tables = {"metrics": cells, "aggregate-metrics": aggregate, "joint-primary": joint,
                  "contrasts": stats["contrasts"], "forced": force_metrics, "threshold-response": changes,
                  "calibration": [{k: v for k, v in r.items() if k != "bins"} for r in calibration],
                  "calibration-bins": [{**{k: r[k] for k in ("split", "method", "seed", "scope")}, **b} for r in calibration for b in r["bins"]],
                  "risk-coverage": [{**{k: r[k] for k in ("split", "method", "seed", "n_questions")}, **p} for r in curves for p in r["points"]],
                  "job-costs-snapshot": costs["closed_jobs"]}
        tables.update({"training-runs": training["runs"], "training-panels": training["panels"], "training-updates": training["updates"]})
        for name, table in tables.items():
            write_csv(output / f"{name}.csv", table)
        (output / "results.md").write_text(markdown(stats))
        figures(output, stats)
        for path, checksum in ctx["hashes"].items():
            if file_digest(ROOT / path) != checksum:
                raise ValueError(f"Input changed during analysis: {path}")
        complete = {"status": "artifacts_ready_for_review", "created_at": utcnow(), "goal_complete": False,
                    "direct_variants": 13, "selector_seeds": 3, "direct_prediction_rows": sum(len(r) for r in forced.values()) + sum(len(r) for (s,m,se,scope,t),r in policies.items() if m not in FILTERS and (scope == "primary" or t in DIAGNOSTIC)),
                    "source_sha256": ctx["hashes"], "script_sha256": file_digest(Path(__file__)),
                    "manuscript_modified": False, "visual_inspection_required": True,
                    "outputs": {p.name: file_digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
        write_json(output / "complete.json", complete)
        return complete
    except BaseException as exc:
        write_json(output / "failure.json", {"status": "failed", "error": repr(exc), "at": utcnow()})
        raise


if __name__ == "__main__":
    result = main()
    print({"status": result["status"], "outputs": len(result["outputs"]), "goal_complete": False})
