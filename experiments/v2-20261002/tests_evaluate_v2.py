"""CPU-only fixtures for the complete v2 gate and deterministic evaluation matrix."""
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tested_v2_evaluator", HERE / "evaluate_v2.py")
ev = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ev
spec.loader.exec_module(ev)


def record(split, i):
    return {"id": f"{split}-{i}", "dataset": split, "question": f"Which entity belongs to {split} case {i}?", "aliases": ["Paris"]}


def output(text="Paris", **overrides):
    return {"text": text, "truncated": False, "generated_tokens": 2, "seconds": .01, **overrides}


class FakeSelector:
    def __init__(self, root, work, splits):
        self.root, self.work, self.splits = root, work, splits
        self.out = work / "artifacts/selector"
        self.checked = []

    def context(self, contract_path, root):
        assert root == self.root
        return {"out": self.out, "sha": "selector-identity", "splits": self.splits}

    def verify_run(self, ctx, seed):
        self.checked.append(seed)
        return ev.read_json(self.out / "runs" / f"s{seed}" / "complete.json")

    def locked_test_context(self, ctx, lock_path):
        assert self.checked == list(ev.SEEDS)
        return ctx

    def candidates(self, ctx, split):
        envelope = ev.read_json(self.out / "candidates" / f"{split}.json")
        assert ev.digest(envelope["rows"]) == envelope["sha256"]
        return envelope["rows"]


class FakeEngine:
    calls = []
    instances = []

    def __init__(self, config, workdir, model, adapter=None):
        assert model == "qwen14b"
        self.closed = False
        self.adapter = adapter
        self.instances.append(self)

    def generate(self, conversations, sample=False):
        assert sample is False
        self.calls.extend(conversations)
        return [output() for _ in conversations]

    def close(self):
        assert not self.closed
        self.closed = True


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path
    work = root / "experiments/v2"
    work.mkdir(parents=True)
    monkeypatch.setattr(ev, "ROOT", root)
    monkeypatch.setattr(ev, "WORK", work)
    FakeEngine.calls = []
    FakeEngine.instances = []
    rel = lambda path: str(path.relative_to(root))
    def save(path, value):
        ev.write_json(path, value)
        return ev.file_digest(path)
    config = {
        "training": {"max_prompt_length": 512, "max_completion_length": 32, "scale_rewards": "none",
                     "loss_type": "dr_grpo", "per_device_train_batch_size": 4,
                     "gradient_accumulation_steps": 12, "num_generations": 8, "steps_per_generation": 12},
        "thresholds": {"train": [.6, .75, .9], "primary": [.65, .85], "diagnostic": [.6, .75, .9, .95]},
        "data": {"train": 1008, "reduced_train": 504}, "evaluation": {"batch_size": 16}}
    config_path = work / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    splits = {split: [record(split, i) for i in range(n)] for split, n in ev.COUNTS.items()}
    data, manifest = {}, {"splits": {}, "diagnostic_ids": {}}
    for split, rows in splits.items():
        path = work / "data" / f"{split}.jsonl"
        ev.write_jsonl(path, rows)
        data[split] = {"path": rel(path), "sha256": ev.file_digest(path)}
        manifest["splits"][split] = {"sha256": ev.file_digest(path), "ids": [r["id"] for r in rows]}
        manifest["diagnostic_ids"][split] = [r["id"] for r in rows[-200:]]
    manifest_path = work / "data/manifest.json"
    save(manifest_path, manifest)
    contract_path = work / "contract.json"
    save(contract_path, {"config_path": rel(config_path), "data_manifest": rel(manifest_path),
                        "model_workdir": "existing-models", "evaluation": {
                            "primary_thresholds": list(ev.PRIMARY), "diagnostic_thresholds": list(ev.DIAGNOSTIC)}})
    source_path = work / "artifacts/source-lock.json"
    save(source_path, {"image": ev.IMAGE, "files": {rel(p): ev.file_digest(p) for p in (config_path, manifest_path, contract_path)}})
    budget_path = work / "artifacts/budget-lock.json"
    save(budget_path, {"rl_steps": 504, "rl_questions": 1008, "rl_completions": 24192})
    lock = {"status": "ready", "contract_path": rel(contract_path), "contract_sha256": ev.file_digest(contract_path),
            "source_lock_sha256": ev.file_digest(source_path), "budget_lock_sha256": ev.file_digest(budget_path),
            "data": data, "rl_completed": {}, "selector_completion_sha256": {}, "selector_evaluation_sha256": {}}
    for variant in sorted(ev.VARIANTS):
        folder = work / "artifacts/runs" / variant
        adapter = folder / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(variant.encode())
        (adapter / "adapter_config.json").write_text('{}\n')
        hashes = ev.adapter_fingerprint(adapter)
        arm, seed = variant.rsplit("-s", 1)
        group, exposure, reward = {ev.ARMS[0]: (4, "single_tau", "conditioned"), ev.ARMS[1]: (8, "single_tau", "conditioned"),
                                  ev.ARMS[2]: (8, "paired_tau", "conditioned"), ev.ARMS[3]: (8, "paired_tau", "fixed")}[arm]
        result_path = folder / "result.json"
        save(result_path, {"status": "complete", "optimizer_steps": 504, "gradient_steps": 504, "questions": 1008,
                           "completions": 24192, "save_roundtrip": True, "sampler_traversal_verified": True,
                           "adapter_sha256": hashes, "spec": {"group_size": group, "exposure_mode": exposure,
                           "arm": reward, "seed": int(seed), "questions_per_update": 2, "micro_batch_size": 4}})
        lock["rl_completed"][variant] = {"adapter": rel(adapter), "adapter_sha256": hashes,
                                         "result_path": rel(result_path), "result_sha256": ev.file_digest(result_path)}
    selector = FakeSelector(root, work, {s: [record(s, i) for i in range(3)] for s in ("train", "dev", "calibration")})
    for seed in ev.SEEDS:
        path = selector.out / "runs" / f"s{seed}" / "complete.json"
        lock["selector_completion_sha256"][str(seed)] = save(path, {"status": "complete", "seed": seed,
                     "steps": 500, "examples_seen": 8000, "technical_pilot_disposable": False})
        directory = selector.out / "evaluation" / f"s{seed}"
        files = {}
        for name in ("calibrators.json", "dev-scores.json", "results.json"):
            files[name] = save(directory / name, {"synthetic": True})
        lock["selector_evaluation_sha256"][str(seed)] = save(directory / "complete.json", {
            "status": "complete", "identity_sha256": "selector-identity", "files": files})
    for split, refs in splits.items():
        rows = [{**output(), **ev.grade("Paris", r["aliases"]).asdict(), "id": r["id"], "question": r["question"],
                 "y": 1, "split": split, "candidate_index": 0, "generator": "original_frozen"} for r in refs]
        save(selector.out / "candidates" / f"{split}.json", {"rows": rows, "sha256": ev.digest(rows)})
    lock_path = work / "artifacts/evaluation-lock.json"
    save(lock_path, lock)
    return {"root": root, "work": work, "lock": lock, "lock_path": lock_path, "selector": selector,
            "splits": splits, "manifest": manifest, "manifest_path": manifest_path,
            "source_path": source_path, "save": save, "rel": rel}


def memory():
    return {"available_gib": 64., "peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0}


def run(study, variant, engine=FakeEngine):
    return ev.evaluate(study["lock_path"], variant, 60., engine_factory=engine,
                       resource_probe=memory, selector=study["selector"])


@pytest.mark.parametrize("damage", ["missing_rl", "missing_selector", "missing_calibration", "unready"])
def test_gate_rejects_before_test_read_or_model(study, monkeypatch, damage):
    lock = study["lock"]
    if damage == "missing_rl": lock["rl_completed"].pop(sorted(ev.VARIANTS)[0])
    elif damage == "missing_selector": lock["selector_completion_sha256"].pop("29")
    elif damage == "missing_calibration": lock["selector_evaluation_sha256"].pop("43")
    else: lock["status"] = "draft"
    study["save"](study["lock_path"], lock)
    def forbidden(*args): raise AssertionError("Test references were read before the gate")
    monkeypatch.setattr(ev, "read_jsonl", forbidden)
    with pytest.raises(ValueError): run(study, "original")
    assert FakeEngine.instances == []


@pytest.mark.parametrize("damage", ["adapter", "source", "calibrator", "swapped_identity", "pilot_budget"])
def test_attested_inputs_and_real_training_required(study, damage):
    name = sorted(ev.VARIANTS)[0]
    entry = study["lock"]["rl_completed"][name]
    if damage == "adapter":
        (study["root"] / entry["adapter"] / "adapter_model.safetensors").write_text("changed")
    elif damage == "source":
        (study["work"] / "config.yaml").write_text("changed")
    elif damage == "calibrator":
        (study["selector"].out / "evaluation/s17/calibrators.json").write_text("changed")
    elif damage == "swapped_identity":
        path = study["root"] / entry["result_path"]
        result = ev.read_json(path); result["spec"]["seed"] = 43
        entry["result_sha256"] = study["save"](path, result)
        study["save"](study["lock_path"], study["lock"])
    else:
        path = study["work"] / "artifacts/budget-lock.json"
        study["lock"]["budget_lock_sha256"] = study["save"](path, {"rl_steps": 32, "rl_questions": 64, "rl_completions": 1536})
        study["save"](study["lock_path"], study["lock"])
    with pytest.raises(ValueError): run(study, "original")
    assert FakeEngine.instances == []


def test_full_matrix_original_reuses_exact_forced_answers(study):
    complete = run(study, "original")
    assert complete["status"] == "complete"
    assert complete["prediction_rows"] == 9100
    assert complete["fresh_generated_rows"] == len(FakeEngine.calls) == 6600
    assert complete["reused_candidate_rows"] == 2500
    assert len(complete["outputs"]) == 15 and len(complete["populations"]) == 14
    assert FakeEngine.instances[0].closed
    assert all("Do not abstain" not in c[0]["content"] for c in FakeEngine.calls)
    directory = study["work"] / "artifacts/predictions/original"
    for name, checksum in complete["outputs"].items(): assert ev.file_digest(directory / name) == checksum
    for split, count in ev.COUNTS.items():
        forced = ev.read_jsonl(directory / f"{split}-forced.jsonl")
        assert len(forced) == count and all(r["reused_original_candidate"] for r in forced)
        for row in forced: assert row["row_sha256"] == ev.digest({k: v for k, v in row.items() if k != "row_sha256"})
        for tau in (*ev.PRIMARY, *ev.DIAGNOSTIC):
            rows = ev.read_jsonl(directory / f"{split}-t{tau:.2f}.jsonl")
            expected = count if tau in ev.PRIMARY else 200
            assert len(rows) == expected
            assert all(r["tau"] == tau and r["mode"] == "threshold" for r in rows)
            if tau in ev.DIAGNOSTIC:
                assert [r["id"] for r in rows] == study["manifest"]["diagnostic_ids"][split]
    with pytest.raises(FileExistsError): run(study, "original")


def test_full_matrix_adapter_generates_forced_and_all_thresholds(study):
    variant = "c_g8_paired_tau-s29"
    complete = run(study, variant)
    assert complete["fresh_generated_rows"] == 9100 and complete["reused_candidate_rows"] == 0
    assert sum("Do not abstain" in c[0]["content"] for c in FakeEngine.calls) == 2500
    assert FakeEngine.instances[0].adapter == study["root"] / study["lock"]["rl_completed"][variant]["adapter"]
    assert complete["adapter_sha256"] == study["lock"]["rl_completed"][variant]["adapter_sha256"]
    first = ev.read_jsonl(study["work"] / "artifacts/predictions" / variant / "test_trivia-t0.65.jsonl")[0]
    assert (first["arm"], first["seed"], first["variant_id"]) == ("c_g8_paired_tau", 29, variant)


@pytest.mark.parametrize("damage", ["order", "grade", "count"])
def test_shared_candidates_need_exact_identity_grading_and_population(study, damage):
    path = study["selector"].out / "candidates/test_nq.json"
    data = ev.read_json(path)
    if damage == "order": data["rows"][0], data["rows"][1] = data["rows"][1], data["rows"][0]
    elif damage == "grade": data["rows"][0]["outcome"] = "incorrect"
    else: data["rows"].pop()
    data["sha256"] = ev.digest(data["rows"]); study["save"](path, data)
    with pytest.raises(ValueError): run(study, "original")
    assert FakeEngine.instances == []


def test_duplicates_overlap_and_invalid_diagnostic_are_rejected(study):
    ref = record("train", 1)
    with pytest.raises(ValueError): ev.assert_disjoint({"train": [ref, ref]})
    with pytest.raises(ValueError): ev.assert_disjoint({"train": [ref], "test": [{**ref, "id": "other", "question": ref["question"].upper()}]})
    study["manifest"]["diagnostic_ids"]["test_nq"][0] = "absent"
    study["save"](study["manifest_path"], study["manifest"])
    source = ev.read_json(study["source_path"])
    source["files"][study["rel"](study["manifest_path"])] = ev.file_digest(study["manifest_path"])
    study["lock"]["source_lock_sha256"] = study["save"](study["source_path"], source)
    study["save"](study["lock_path"], study["lock"])
    with pytest.raises(ValueError, match="diagnostic"): run(study, "original")


def test_prediction_grades_empty_truncated_idk_and_rejects_invalid_metadata():
    ref = record("test", 1)
    for text, truncated, expected in [("Paris", False, "correct"), ("Paris", True, "error"),
                                      ("", False, "error"), ("IDK", False, "abstain")]:
        row = ev.prediction(output(text, truncated=truncated), ref, "original", "test_trivia", .65)
        assert row["outcome"] == expected
    for item in (output(seconds=math.nan), output(seconds=True), output(generated_tokens=True),
                 output(generated_tokens=33), output(truncated=1), output(text=None)):
        with pytest.raises(ValueError): ev.prediction(item, ref, "original", "test_trivia", .65)


def test_runtime_and_memory_bounds_are_enforced():
    for seconds in (0, -1, math.nan, math.inf):
        with pytest.raises(ValueError): ev.Deadline(seconds, memory, alarms=False)
    now = [0.]
    with ev.Deadline(10, memory, clock=lambda: now[0], alarms=False) as budget:
        budget.check(); now[0] = 10
        with pytest.raises(TimeoutError): budget.check()
    with ev.Deadline(10, lambda: {"available_gib": 15.99}, alarms=False) as budget:
        with pytest.raises(MemoryError): budget.check()


def test_partial_generation_is_closed_and_never_automatically_resumed(study):
    class BrokenEngine(FakeEngine):
        def generate(self, conversations, sample=False): return []
    variant = "a_g4_single_tau-s17"
    with pytest.raises(ValueError, match="batch"): run(study, variant, BrokenEngine)
    directory = study["work"] / "artifacts/predictions" / variant
    assert (directory / "failure.json").exists() and not (directory / "complete.json").exists()
    assert FakeEngine.instances[0].closed
    with pytest.raises(FileExistsError): run(study, variant)


def test_mutation_during_generation_cannot_produce_completion(study):
    class MutatingEngine(FakeEngine):
        def generate(self, conversations, sample=False):
            (study["work"] / "config.yaml").write_text("changed after validation")
            return super().generate(conversations, sample)
    variant = "d_g8_paired_fixed075-s43"
    with pytest.raises(ValueError, match="Source changed"): run(study, variant, MutatingEngine)
    directory = study["work"] / "artifacts/predictions" / variant
    assert not (directory / "complete.json").exists()
    assert (directory / "failure.json").exists()
