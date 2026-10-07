"""CPU fixtures: paired inference, strict outcomes, immutable receipts and plots."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("tested_analysis_v2", Path(__file__).with_name("analysis_v2.py"))
a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = a
SPEC.loader.exec_module(a)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def candidate(text="Paris", truncated=False, logit=0., qid="q", aliases=None):
    aliases = ["Paris"] if aliases is None else aliases
    score = a.grade(text, aliases, truncated)
    return {"id": qid, "question": "Capital?", "aliases": aliases, "text": text, "truncated": truncated,
            "outcome": score.outcome, "reason": score.reason, "y": int(score.outcome == "correct"),
            "logits": {m: logit for m in a.FILTERS}, "original_logit": logit,
            "generated_tokens": 3, "generator": "original_frozen", "candidate_index": 0, "split": "test_trivia"}


def test_filter_strict_boundary_invalid_idk_and_punctuated_idk():
    assert a.filtered(candidate(logit=0), "critic_raw", .5)["outcome"] == "abstain"
    for text, trunc in (("", False), ("IDK", False), ("Paris", True), ("IDK", True)):
        row = a.filtered(candidate(text, trunc, 100), "critic_raw", .6)
        assert row["outcome"] == "abstain"
        assert row["reason"] == "confidence_gate"
    assert a.filtered(candidate(".IDK", logit=100), "critic_raw", .6)["outcome"] == "error"
    assert a.filtered(candidate("Paris", logit=100), "critic_raw", .6)["outcome"] == "correct"
    assert a.sigmoid(-1000) == 0 and a.sigmoid(1000) == 1
    with pytest.raises(ValueError):
        a.sigmoid(float("nan"))


def test_empty_coverage_undefined_and_filtered_invalid_not_emitted_error():
    row = a.filtered(candidate("", logit=100), "critic_raw", .65)
    metrics = a.metric_record([row])
    assert metrics["coverage"] == 0 and metrics["selective_risk"] is None
    assert metrics["utility"] == 0 and metrics["invalid"] == 0 and metrics["candidate_invalid"] == 1


def synthetic_policies(n=4, splits=None):
    policies = {}
    for split in splits or a.COUNTS:
        for method_index, method in enumerate(("original", *a.ARMS, *a.FILTERS)):
            for seed in a.method_seeds(method):
                for scope in ("primary", "diagnostic"):
                    for tau in (a.PRIMARY if scope == "primary" else (*a.PRIMARY, *a.DIAGNOSTIC)):
                        policies[split, method, seed, scope, tau] = [
                            {"id": str(i), "tau": tau,
                             "outcome": ("correct", "error", "abstain")[(i + method_index + seed) % 3],
                             "reason": "fixture"} for i in range(n)]
    return policies


def test_pooled_risk_is_ratio_of_counts_and_deterministic_n_seeds_one():
    policies = synthetic_policies(4)
    method = a.ARMS[0]
    for seed in a.SEEDS:
        for tau in a.PRIMARY:
            rows = policies["test_trivia", method, seed, "primary", tau]
            for r in rows:
                r["outcome"] = "abstain"
            if seed == 17:
                rows[0]["outcome"] = "error"
            if seed == 29:
                for r in rows:
                    r["outcome"] = "correct"
    _, _, joint = a.policy_metrics(policies)
    value = next(r for r in joint if r["split"] == "test_trivia" and r["method"] == method)
    assert value["selective_risk"] == pytest.approx(1/5)
    assert value["n"] == 24 and value["n_questions"] == 4 and value["n_seeds"] == 3
    fixed = next(r for r in joint if r["split"] == "test_trivia" and r["method"] == "original_calibrated")
    assert fixed["n"] == 8 and fixed["n_seeds"] == 1


def test_differences_pair_both_thresholds_and_broadcast_baseline():
    p = synthetic_policies()
    values = a.paired_differences(p, "test_trivia", "critic_calibrated", "original_calibrated")
    for s, seed in enumerate(a.SEEDS):
        for i in range(4):
            expected = sum(a.reward(p["test_trivia", "critic_calibrated", seed, "primary", t][i]["outcome"], t)
                           - a.reward(p["test_trivia", "original_calibrated", 0, "primary", t][i]["outcome"], t)
                           for t in a.PRIMARY)/2
            assert values[s, i] == pytest.approx(expected)
    p["test_trivia", "critic_calibrated", 29, "primary", .85].reverse()
    with pytest.raises(ValueError, match="Unpaired"):
        a.paired_differences(p, "test_trivia", "critic_calibrated", "original_calibrated")


def test_crossed_bootstrap_matches_manual_draws_and_intervals():
    differences = np.array([[0., .2, .4], [.1, .3, .7], [-.1, .1, .6]])
    result = a.crossed_bootstrap(differences, n_boot=101, seed=42)
    rng = np.random.default_rng(42)
    values = []
    for _ in range(101):
        seeds = rng.integers(0, 3, 3); questions = rng.integers(0, 3, 3)
        values.append(sum(differences[s, q] for s in seeds for q in questions)/9)
    assert result["effect"] == pytest.approx(differences.mean())
    for key, q in (("ci95_low", .025), ("ci95_high", .975), ("ci99_low", .005), ("ci99_high", .995)):
        assert result[key] == pytest.approx(np.quantile(values, q))
    assert result["family_size"] == 5 and result["bootstrap_replicates"] == 101
    assert a.crossed_bootstrap(np.full((3, 2), .25), 20)["ci99_low"] == .25
    with pytest.raises(ValueError):
        a.crossed_bootstrap(np.zeros((1, 2)))


def test_five_contrasts_each_dataset_and_no_unregistered_pvalues(monkeypatch):
    actual = a.crossed_bootstrap
    monkeypatch.setattr(a, "crossed_bootstrap", lambda d: actual(d, 31))
    contrasts = a.contrast_metrics(synthetic_policies())
    assert len(contrasts) == 10
    assert sum(r["role"] == "primary_family" for r in contrasts) == 5
    selector = next(r for r in contrasts if r["right"] == "original_calibrated")
    assert selector["right_n_seeds"] == 1 and selector["left_n_seeds"] == 3
    assert all(not any("p_value" in k for k in r) for r in contrasts)


def test_calibration_denominators_fixed_bins_and_saturated_auc():
    rows = [candidate("wrong", logit=40), candidate("Paris", logit=50), candidate("IDK", logit=0)]
    substantive = a.calibration_metrics(rows, "critic_raw", "substantive")
    all_rows = a.calibration_metrics(rows, "critic_raw", "all")
    assert a.sigmoid(40) == a.sigmoid(50) == 1
    assert substantive["auc"] == 1  # Logits retain the ordering lost to float sigmoid.
    assert substantive["n"] == 2 and all_rows["n"] == 3
    assert substantive["brier"] == .5 and substantive["bins"][-1]["n"] == 2
    assert substantive["log_loss"] == pytest.approx(20)  # No 1e-15 clipping of an extreme wrong answer.
    assert substantive["ece10"] == .5
    assert sum(b["n"] for b in all_rows["bins"]) == 3
    assert a.calibration_metrics([candidate("IDK")], "critic_raw", "substantive")["n"] == 0
    assert a.calibration_metrics([candidate()], "critic_raw", "all")["auc"] is None


def test_risk_coverage_groups_ties_and_excludes_invalid_and_idk():
    rows = [candidate("wrong", logit=5), candidate("Paris", logit=5), candidate("Paris", logit=1),
            candidate("IDK", logit=10), candidate("", logit=20), candidate("Paris", True, 30)]
    points = a.risk_coverage(rows, "critic_raw")
    assert [p["emitted"] for p in points] == [0, 2, 3]
    assert points[0]["risk"] is None
    assert points[1]["coverage"] == 2/6 and points[1]["risk"] == .5
    assert points[-1]["coverage"] == .5 and points[-1]["risk"] == pytest.approx(1/3)


def direct_row(ref, variant="original", tau=.65, split="test_trivia", text="The_Paris!", truncated=False):
    row = {**ref, "variant": variant, "variant_id": variant, "arm": "original", "seed": 0,
           "model": "qwen14b", "evaluation_split": split, "mode": "forced" if tau is None else "threshold", "tau": tau,
           "text": text, "truncated": truncated, "generated_tokens": 3, **a.grade(text, ref["aliases"], truncated).asdict()}
    return {**row, "row_sha256": a.digest(row)}


def test_direct_regrade_hash_reference_and_condition_failures():
    ref = {"id": "a", "question": "Capital?", "aliases": ["Paris"]}
    row = direct_row(ref)
    assert a.check_rows([row], [ref], "original", "test_trivia", .65)[0]["outcome"] == "correct"
    for mutation in ({"text": "Rome"}, {"tau": .85}, {"outcome": "error"}, {"aliases": ["Rome"]}):
        changed = {**row, **mutation}
        changed["row_sha256"] = a.digest({k:v for k,v in changed.items() if k != "row_sha256"})
        with pytest.raises(ValueError):
            a.check_rows([changed], [ref], "original", "test_trivia", .65)
    empty = direct_row(ref, text="")
    truncated = direct_row(ref, text="IDK", truncated=True)
    assert a.check_rows([empty], [ref], "original", "test_trivia", .65)[0]["reason"] == "empty"
    assert a.check_rows([truncated], [ref], "original", "test_trivia", .65)[0]["reason"] == "truncated"


def test_deterministic_filters_allow_only_critic_seed_variation():
    one = [candidate()]; two = copy.deepcopy(one)
    two[0]["logits"]["critic_raw"] += 2
    two[0]["logits"]["critic_calibrated"] -= 1
    assert a.deterministic_signature(one) == a.deterministic_signature(two)
    two[0]["logits"]["original_calibrated"] += .01
    assert a.deterministic_signature(one) != a.deterministic_signature(two)


def receipts_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "ROOT", tmp_path)
    art = tmp_path / "artifacts"
    lock = {"status": "ready", "contract_sha256": "contract"}
    put(art / "evaluation-lock.json", lock)
    lock_sha = a.file_digest(art / "evaluation-lock.json")
    for variant in a.VARIANTS:
        folder = art / "predictions" / variant
        names = a.direct_names() | {"runtime.json"}
        for name in names:
            put(folder / name, {})
        populations = {}
        for split, count in a.COUNTS.items():
            for tau in (None, *a.PRIMARY, *a.DIAGNOSTIC):
                name = f"{split}-forced.jsonl" if tau is None else f"{split}-t{tau:.2f}.jsonl"
                populations[name] = count if tau is None or tau in a.PRIMARY else 200
        put(folder / "complete.json", {"status": "complete", "evaluation_lock_sha256": lock_sha,
            "variant": variant, "contract_sha256": "contract", "outputs": {n: a.file_digest(folder/n) for n in names},
            "populations": populations, "prediction_rows": 9100, "weights_updated": False,
            "primary_thresholds": list(a.PRIMARY), "diagnostic_thresholds": list(a.DIAGNOSTIC)})
    for seed in a.SEEDS:
        folder = art / "selector/test-evaluation" / f"s{seed}"
        names = ("test_trivia-scores.json", "test_nq-scores.json", "results.json")
        for name in names:
            put(folder / name, {})
        put(folder / "complete.json", {"status": "complete", "evaluation_lock_sha256": lock_sha,
                                       "files": {n: a.file_digest(folder/n) for n in names}})
    return art


def test_barrier_requires_all_sixteen_before_prediction_loading(tmp_path, monkeypatch):
    art = receipts_fixture(tmp_path, monkeypatch)
    _, hashes, receipts = a.receipt_barrier(art)
    assert len(receipts) == 16 and hashes
    (art / "selector/test-evaluation/s43/complete.json").unlink()
    # Earlier malformed scientific data must not be parsed before all receipts.
    (art / "predictions/original/test_trivia-t0.65.jsonl").write_text("not json")
    with pytest.raises(ValueError, match="All thirteen"):
        a.receipt_barrier(art)


@pytest.mark.parametrize("what", ["output", "grid", "lock", "population"])
def test_barrier_rejects_mutated_output_grid_and_provenance(tmp_path, monkeypatch, what):
    art = receipts_fixture(tmp_path, monkeypatch)
    path = art / "predictions/original/complete.json"
    done = a.read_json(path)
    if what == "output":
        (path.parent / "test_trivia-t0.65.jsonl").write_text("changed")
    elif what == "grid":
        del done["outputs"]["test_nq-forced.jsonl"]
    elif what == "lock":
        done["evaluation_lock_sha256"] = "wrong"
    else:
        done["populations"]["test_nq-forced.jsonl"] = 499
    if what != "output":
        put(path, done)
    with pytest.raises(ValueError):
        a.receipt_barrier(art)


def rollout_fixture(group_size=4):
    rows = []
    texts = ["Paris"]*4 + ["IDK"]*4
    for index, text in enumerate(texts):
        score = a.grade(text, ["Paris"])
        value = a.reward(score.outcome, .6)
        rows.append({"step": 0, "id": "q", "exposure_id": "exposure", "tau": .6, "slot": 0,
            "sample_index": index, "text": text, "aliases": ["Paris"], "truncated": False,
            **score.asdict(), "reward": value, "arm": "conditioned", "group_size": group_size,
            "baseline_group_within_prompt": index//group_size,
            "baseline_group_id": a.digest([0, "exposure", index//group_size]), "completion_ids": [1, 2]})
    for start in range(0, 8, group_size):
        block = rows[start:start+group_size]
        for row in block:
            row["loo_advantage"] = row["reward"] - (sum(r["reward"] for r in block)-row["reward"])/(group_size-1)
    return rows


def test_rollout_diagnostics_matched_exposure_but_distinct_credit_and_bad_reward():
    four = a.rollout_diagnostics(rollout_fixture(4), 4, 1)
    eight = a.rollout_diagnostics(rollout_fixture(8), 8, 1)
    assert four["completions"] == eight["completions"] == 8
    assert four["mixed_answer_idk_eight_draw_exposures"] == eight["mixed_answer_idk_eight_draw_exposures"] == 1
    assert four["constant_credit_groups"] == 2 and four["active_reward_updates"] == 0
    assert eight["mixed_answer_idk_credit_groups"] == 1 and eight["active_reward_updates"] == 1
    assert four["abstain_fraction"] == eight["abstain_fraction"] == .5
    rows = rollout_fixture(8); rows[0]["reward"] += .1
    with pytest.raises(ValueError, match="reward/grade"):
        a.rollout_diagnostics(rows, 8, 1)


def test_rollout_bad_loo_identity_and_truncated_grade_are_detected():
    for field, value in (("loo_advantage", 99), ("truncated", True), ("sample_index", 7), ("baseline_group_id", "other")):
        rows = rollout_fixture(8); rows[0][field] = value
        with pytest.raises(ValueError):
            a.rollout_diagnostics(rows, 8, 1)


def test_cost_snapshot_counts_closed_outer_jobs_only(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "ROOT", tmp_path); monkeypatch.setattr(a, "ART", tmp_path)
    for name, status, seconds in (("pilot", "complete", 10), ("failed", "failed", 3), ("analysis", "running", None)):
        put(tmp_path / "jobs" / name / "job.json", {"id": name, "status": status, "seconds": seconds})
    put(tmp_path / "selector/costs/duplicate.json", {"seconds": 10000})
    costs, hashes = a.cost_snapshot()
    assert costs["closed_attempt_seconds"] == 13 and costs["excluded_running_jobs"] == ["analysis"]
    assert len(hashes) == 2
    put(tmp_path / "jobs/other/job.json", {"id": "other", "status": "running"})
    with pytest.raises(ValueError, match="Another job"):
        a.cost_snapshot()


def test_output_exists_aborts_before_any_new_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "ART", tmp_path)
    (tmp_path / "analysis").mkdir(); sentinel = tmp_path / "analysis/do-not-change"
    sentinel.write_text("old")
    monkeypatch.setattr(a, "validated_context", lambda: pytest.fail("should not read"))
    with pytest.raises(FileExistsError):
        a.main()
    assert sentinel.read_text() == "old"


def test_end_to_end_cpu_artifacts_and_figures_without_test_access(tmp_path, monkeypatch):
    # All inputs here are generated fixtures; no real TEST files are opened.
    monkeypatch.setattr(a, "ART", tmp_path)
    monkeypatch.setattr(a, "COUNTS", {"test_trivia": 200, "test_nq": 200})
    policies = synthetic_policies(200)
    forced, scores = {}, {}
    for split in a.COUNTS:
        for method in ("original", *a.ARMS):
            for seed in a.method_seeds(method):
                forced[split, method, seed] = [{**candidate(qid=str(i)), "tau": None} for i in range(200)]
        for method in a.FILTERS:
            for seed in a.method_seeds(method):
                scores[split, method, seed] = [candidate("Paris" if i%2 else "wrong", logit=(i%10-5)/3, qid=str(i)) for i in range(200)]
    monkeypatch.setattr(a, "validated_context", lambda: {"hashes": {}})
    monkeypatch.setattr(a, "load_predictions", lambda _: (policies, forced, scores))
    original_bootstrap = a.crossed_bootstrap
    monkeypatch.setattr(a, "crossed_bootstrap", lambda d: original_bootstrap(d, 51))
    monkeypatch.setattr(a, "cost_snapshot", lambda: ({"closed_jobs": [{"job":"fixture","seconds":10}], "closed_attempt_seconds":10, "excluded_running_jobs":[]}, {}))
    monkeypatch.setattr(a, "training_diagnostics", lambda _: {"runs":[{"fixture":1}], "panels":[{"fixture":1}], "updates":[{"fixture":1}]})
    done = a.main()
    assert done["status"] == "artifacts_ready_for_review" and done["goal_complete"] is False
    assert done["manuscript_modified"] is False and done["visual_inspection_required"] is True
    assert done["direct_prediction_rows"] == 36400
    assert sum(name.endswith(".png") for name in done["outputs"]) == 5
    for name, expected in done["outputs"].items():
        assert a.file_digest(tmp_path / "analysis" / name) == expected
    stats = a.read_json(tmp_path / "analysis/statistics.json")
    assert len(stats["contrasts"]) == 10
    assert len(stats["calibration"]) == 40  # 2 datasets × (4 fixed+2×3 critic) × 2 scopes.
    assert "No se han añadido etiquetas humanas" in (tmp_path / "analysis/results.md").read_text()
    with pytest.raises(FileExistsError):
        a.main()
