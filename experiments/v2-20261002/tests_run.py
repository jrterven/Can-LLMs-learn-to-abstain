"""Fail-closed orchestration and budget tests; no GPU/model downloads."""
import importlib.util
from pathlib import Path
import sys
import pytest

spec = importlib.util.spec_from_file_location("v2_coordinator_test", Path(__file__).with_name("run.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    here = tmp_path / "experiments/v2"
    art = here / "artifacts"
    art.mkdir(parents=True)
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "HERE", here)
    monkeypatch.setattr(r, "ART", art)
    monkeypatch.setattr(r, "cfg", lambda: {"budget": {"max_phase_hours": 1,
             "planned_phase_hours": 80, "projection_margin": 1.25}})
    r.write_json(art / "source-lock.json", {"image": r.IMAGE, "files": {}})
    return art


def test_mutated_frozen_source_blocks_work(sandbox):
    source = r.ROOT / "source.py"
    source.write_text("old")
    r.write_json(sandbox / "source-lock.json", {"image": r.IMAGE,
                 "files": {"source.py": r.file_digest(source)}})
    assert r.check()["image"] == r.IMAGE
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        r.check()


def test_success_records_all_elapsed_and_never_repeats(sandbox):
    argv = ["-c", "print('complete')"]
    row = r.job("one", argv, 5)
    assert row["status"] == "complete" and row["seconds"] > 0
    assert r.job("one", argv, 5) == row
    assert r.spent_seconds() == row["seconds"]
    assert len(r.read_jsonl(sandbox / "events.jsonl")) == 2
    with pytest.raises(RuntimeError, match="identity change"):
        r.job("one", ["-c", "print('different')"], 5)


def test_failed_and_unknown_attempts_block_silent_retries(sandbox):
    with pytest.raises(RuntimeError, match="exited"):
        r.job("failed", ["-c", "raise RuntimeError('test failure')"], 5)
    row = r.read_json(sandbox / "jobs/failed/job.json")
    assert row["status"] == "failed" and row["seconds"] > 0
    with pytest.raises(RuntimeError, match="automatic retry"):
        r.job("failed", ["-c", "raise RuntimeError('test failure')"], 5)
    r.write_json(sandbox / "jobs/crashed/job.json", {"id": "crashed", "status": "running"})
    with pytest.raises(RuntimeError, match="unresolved"):
        r.spent_seconds()


def test_external_timeout_counts_failure_and_stops_child(sandbox):
    with pytest.raises(TimeoutError):
        r.job("deadline", ["-c", "import time; time.sleep(30)"], .1)
    record = r.read_json(sandbox / "jobs/deadline/job.json")
    assert record["status"] == "failed"
    assert .1 <= record["seconds"] < 10


def test_evaluation_lock_requires_all_final_adapters(sandbox):
    r.write_json(r.HERE / "contract.json", {})
    r.write_json(sandbox / "budget-lock.json", {"rl_steps": 504})
    with pytest.raises(FileNotFoundError):
        r.open_evaluation()
    assert not (sandbox / "evaluation-lock.json").exists()


def test_configuration_preserves_new_thresholds_and_matched_budget():
    import yaml
    c = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    assert set(c["thresholds"]["train"]).isdisjoint(c["thresholds"]["primary"])
    assert sum(c["thresholds"]["train"]) / 3 == .75
    for arm in r.ARMS:
        s = r.arm_spec(arm, 17)
        assert s.completions_per_update == 48
        assert s.accumulation_steps == 12
    b = c["budget"]
    assert b["historical_recorded_hours"] + b["max_phase_hours"] <= b["global_limit_hours"]


def test_budget_uses_time_only_and_reduces_all_rl_equally():
    assert r.budget_projection(10, 1, 2)["selected_questions"] == 1008
    assert r.budget_projection(30, 3, 2)["selected_questions"] == 504
    assert r.budget_projection(100, 3, 2)["selected_questions"] is None
    assert r.budget_projection(10, 1, 79)["selected_questions"] is None
    with pytest.raises(ValueError):
        r.budget_projection(float("nan"), 1, 0)
