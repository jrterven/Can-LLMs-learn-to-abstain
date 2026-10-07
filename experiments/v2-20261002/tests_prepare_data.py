"""CPU checks of v2 data isolation, nested budgets and equal-exposure schedules."""
import importlib.util
from collections import Counter
from pathlib import Path
import shutil

import pytest

spec = importlib.util.spec_from_file_location("v2_data", Path(__file__).with_name("prepare_data.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def row(ident, question=None, answer="reference"):
    return {"id": ident, "dataset": ident.split(":")[0], "question": question or "Question " + ident, "aliases": [answer]}


def training():
    return [row(f"trivia:train-{i}") for i in range(2000)]


def test_identity_projection_includes_probes_but_never_response_or_outcome():
    first = {"question_id": "nq:old", "question": "Who IS this?", "query": "Alternate probe?",
             "response": "trivia:response-only", "outcome": "correct",
             "nested": [{"original_question": "Synthetic—Case"}]}
    second = {**first, "response": "changed", "outcome": "error"}
    assert d.identities(first) == d.identities(second)
    assert d.identities(first) == ({"nq:old"}, {"who is this", "alternate probe", "synthetic case"})


def test_inventory_includes_all_prior_test_probe_and_synthetic_roots(tmp_path):
    destination = tmp_path / "experiments/v2"
    inputs = {
        "data/prepared/train.jsonl": [row("trivia:train")],
        "experiments/final-eval-20261001/data/prepared/test_nq.jsonl": [row("nq:fresh-previous")],
        "artifacts/judge/cases.jsonl": [{"question": "Prior synthetic judge question"}],
        "tests/fixtures/probe.json": {"question_text": "Prior probe question"},
        "analysis/audit.ipynb": {"outputs": [{"question": "Notebook question"}]},
        "data/raw/NQ-open.dev.jsonl": [row("nq:raw-not-exposed")],
        "experiments/v2/data/prepared/test_nq.jsonl": [row("nq:own-reserve")],
    }
    for relative, value in inputs.items():
        path = tmp_path / relative
        (d.write_jsonl if path.suffix == ".jsonl" else d.write_json)(path, value)
    ids, questions, inventory = d.prior_inventory(tmp_path, destination)
    assert ids == {"trivia:train", "nq:fresh-previous"}
    assert {"prior synthetic judge question", "prior probe question", "notebook question"} <= questions
    assert len(inventory) == 5
    assert all("data/raw" not in item["path"] and "experiments/v2/" not in item["path"] for item in inventory)


def test_fresh_unequal_populations_and_all_nq_pool_reserved():
    nq = [row("nq:old", "already seen"), row("nq:q-old", "Prior QUESTION!"),
          row("nq:1", "shared source"), row("nq:2", "nq two"), row("nq:3", "nq three")]
    trivia = [row("trivia:shared", "SHARED source!"), *[row(f"trivia:{i}") for i in range(5)]]
    counts = {"test_nq": 2, "test_trivia": 4}
    splits, diagnostic, stats = d.select_fresh(trivia, nq, {"nq:old"}, {"prior question"}, counts, diagnostic=2)
    d.validate_fresh(splits, {"nq:old"}, {"prior question"}, diagnostic, counts, diagnostic=2)
    assert {k: len(v) for k, v in splits.items()} == counts
    assert "trivia:shared" not in {r["id"] for r in splits["test_trivia"]}
    assert stats["test_nq"]["available_after_exclusion"] == 3
    assert stats["test_trivia"]["within_or_cross_duplicates"] == 1


def test_test_selection_does_not_depend_on_answers_or_source_order():
    nq = [row(f"nq:{i}") for i in range(9)]
    trivia = [row(f"trivia:{i}") for i in range(12)]
    counts = {"test_nq": 3, "test_trivia": 5}
    a, da, _ = d.select_fresh(trivia, nq, set(), set(), counts, diagnostic=2)
    b, db, _ = d.select_fresh([{**r, "aliases": ["changed"]} for r in reversed(trivia)], list(reversed(nq)), set(), set(), counts, diagnostic=2)
    assert {k: [r["id"] for r in v] for k, v in a.items()} == {k: [r["id"] for r in v] for k, v in b.items()}
    assert da == db


def test_insufficient_nq_aborts_without_any_output(tmp_path, monkeypatch):
    prompt_source = tmp_path / "src/abstention/prompts.py"
    prompt_source.parent.mkdir(parents=True)
    prompt_source.write_text("# synthetic pinned prompt source\n")
    monkeypatch.setattr(d, "load_sources", lambda root: ([row(f"trivia:{i}") for i in range(2000)],
                                                       [row(f"nq:{i}") for i in range(499)], {}, {}, {}))
    monkeypatch.setattr(d, "prior_inventory", lambda *args: (set(), set(), []))
    with pytest.raises(ValueError, match="499 fresh questions in test_nq; need 500"):
        d.prepare(tmp_path / "v2", tmp_path)
    assert not (tmp_path / "v2/data/prepared").exists()


def test_validation_rejects_question_leakage_and_bad_diagnostics():
    splits = {"test_nq": [row("nq:1", "Seen question!")], "test_trivia": [row("trivia:1")]}
    ids = {k: [r["id"] for r in v] for k, v in splits.items()}
    counts = {k: 1 for k in splits}
    with pytest.raises(ValueError, match="overlap"):
        d.validate_fresh(splits, set(), {"seen question"}, ids, counts, 1)
    with pytest.raises(ValueError, match="diagnostic"):
        d.validate_fresh(splits, set(), set(), {**ids, "test_nq": ["nq:missing"]}, counts, 1)


def test_train_subsets_are_nested_and_selection_ignores_references():
    full, reduced = d.training_subsets(training())
    reverse, _ = d.training_subsets([{**r, "aliases": ["different"]} for r in reversed(training())])
    assert len(full) == 1008 and reduced == full[:504]
    assert [r["id"] for r in full] == [r["id"] for r in reverse]


def test_fixed_prompt_examples_are_excluded_without_the_final_question():
    assert d.fixed_prompt_questions() == {"what is the capital of france", "what is the chemical symbol for gold",
                                         "what exact integer did an unspecified person privately choose yesterday"}
    assert not any("sentinel" in text for text in d.fixed_prompt_questions())


@pytest.mark.parametrize("size", [1008, 504])
@pytest.mark.parametrize("seed", d.SEEDS)
def test_schedules_equal_questions_generations_balance_and_update_boundaries(size, seed):
    selected = d.training_subsets(training())[0][:size]
    single = d.training_schedule(selected, seed, "single_tau")
    paired = d.training_schedule(selected, seed, "paired_tau")
    assert len(single) == len(paired) == 3 * size
    assert [r["id"] for r in single] == [r["id"] for r in paired]
    assert [r["exposure_id"] for r in single] == [r["exposure_id"] for r in paired]
    assert len({r["exposure_id"] for r in single}) == 3 * size
    assert Counter(r["tau"] for r in single) == Counter(r["tau"] for r in paired) == {t: size for t in d.TAUS}
    assert Counter(r["update_index"] for r in single) == {i: 6 for i in range(size // 2)}
    for index in range(size):
        a, b = single[3*index:3*index+3], paired[3*index:3*index+3]
        assert len({r["id"] for r in a}) == 1 and len({r["tau"] for r in a}) == 1
        assert [r["tau"] for r in b] == list(d.TAUS)
        assert [r["slot"] for r in a] == [0, 1, 2]
        assert all(r["question_index"] == index and r["update_index"] == index // 2 for r in a)
    assert Counter(r["tau"] for r in single[::3]) == {t: size // 3 for t in d.TAUS}
    assert len(single) * 8 == (504 if size == 1008 else 252) * 48


def test_schedule_rejects_unsupported_seeds_counts_modes_and_duplicate_ids():
    for rows, seed, mode in [([],17,"single_tau"), (training()[:5],17,"single_tau"),
                             (training()[:6],99,"single_tau"), (training()[:6],17,"unknown"),
                             ([row("trivia:duplicate")] * 6,17,"paired_tau")]:
        with pytest.raises(ValueError):
            d.training_schedule(rows, seed, mode)


def test_partial_snapshot_never_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "data/prepared/test_nq.jsonl"
    d.write_jsonl(path, [row("nq:precious")])
    original = path.read_bytes()
    monkeypatch.setattr(d, "load_sources", lambda *args: pytest.fail("Must abort before source access"))
    with pytest.raises(FileExistsError, match="Partial"):
        d.prepare(tmp_path)
    assert path.read_bytes() == original


def test_full_snapshot_roundtrip_preserves_originals_and_detects_mutation(tmp_path, monkeypatch):
    root, destination = tmp_path, tmp_path / "experiments/v2"
    parent = root / "experiments/final-eval-20261001"
    parent.mkdir(parents=True)
    shutil.copyfile(d.PARENT / "prepare_data.py", parent / "prepare_data.py")
    monkeypatch.setattr(d, "PARENT", parent)
    for name in ("data", "scoring", "io", "prompts"):
        path = root / f"src/abstention/{name}.py";path.parent.mkdir(parents=True, exist_ok=True);path.write_text("# synthetic source fixture\n")
    reused = {}
    for old, new, rows in (("train", "train_full", training()),
                           ("dev", "dev", [row(f"trivia:dev-{i}") for i in range(500)]),
                           ("calibration", "calibration", [row(f"trivia:cal-{i}") for i in range(500)])):
        path = root / f"data/prepared/{old}.jsonl";d.write_jsonl(path, rows);reused[new] = path
    protected = {str(p.relative_to(root)): d.file_digest(p) for p in reused.values()}
    trivia = [row(f"trivia:new-{i}") for i in range(2005)]
    nq = [row(f"nq:new-{i}") for i in range(505)]
    monkeypatch.setattr(d, "load_sources", lambda root: (trivia, nq, reused, protected, {"kind": "synthetic_fixture"}))
    before = {k: p.read_bytes() for k,p in reused.items()}
    manifest = d.prepare(destination, root)
    assert {k:v["count"] for k,v in manifest["splits"].items()} == d.COUNTS
    assert d.verify(destination, root) == manifest
    assert d.prepare(destination, root) == manifest
    assert all(p.read_bytes() == before[k] for k,p in reused.items())
    assert d.training_rows("b_g8_single_tau", 17, destination=destination) == d.training_rows("a_g4_single_tau", 17, destination=destination)
    assert d.training_rows("c_g8_paired_tau", 17, 504, destination) == d.training_rows("d_g8_paired_fixed075", 17, 504, destination)
    target = destination / "data/prepared/test_nq.jsonl"
    target.write_text(target.read_text().replace("reference", "tampered", 1))
    with pytest.raises(ValueError, match="Changed prepared split"):
        d.verify(destination, root)
