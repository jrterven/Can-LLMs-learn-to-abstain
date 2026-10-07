"""Prepare v2 TRAIN schedules and a fresh held-out reserve using local pinned data.

CPU only; never downloads, loads a model, or uses correctness to select questions.
Run without arguments to prepare, or --verify to check the existing snapshot.
Preparing test records is not authorization to run inference before the v2 lock.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/final-eval-20261001"
PARENT_SCRIPT_SHA256 = "8848f97e162753e83286457b3481eca7243bf86ca0e27a2e5375f3e481d5172e"
sys.path.insert(0, str(ROOT / "src"))
from abstention.data import clean_rows
from abstention.io import digest, file_digest, ordered, read_json, read_jsonl, utcnow, write_json, write_jsonl
from abstention.scoring import normalize_question
from abstention.prompts import messages

SEED = 20261002
SEEDS = (17, 29, 43)
TAUS = (.60, .75, .90)
COUNTS = {"train_full": 2000, "train": 1008, "train_reduced": 504,
          "dev": 500, "calibration": 500, "test_trivia": 2000, "test_nq": 500}
TEST_COUNTS = {"test_nq": 500, "test_trivia": 2000}
DIAGNOSTIC = 200
ARMS = {"a_g4_single_tau": "single_tau", "b_g8_single_tau": "single_tau",
        "c_g8_paired_tau": "paired_tau", "d_g8_paired_fixed075": "paired_tau"}


def fixed_prompt_questions():
    return {normalize_question(item["content"]) for forced in (False, True)
            for item in messages("__FINAL_QUESTION_SENTINEL__", tau=.6, forced=forced)[:-1]
            if item["role"] == "user"}


def parent_module():
    path = PARENT / "prepare_data.py"
    if file_digest(path) != PARENT_SCRIPT_SHA256:
        raise ValueError("The pinned parent data preparer changed")
    spec = importlib.util.spec_from_file_location("v2_parent_data", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def identities(value):
    """Identity projection also covers alternate question fields in probe fixtures."""
    parent = parent_module()
    ids, questions = parent.identities(value)
    def visit(obj):
        if isinstance(obj, dict):
            for key, child in obj.items():
                key = str(key).strip().lower()
                if key in {"question_text", "original_question", "pregunta", "query"} and isinstance(child, str) and child.strip():
                    questions.add(normalize_question(child))
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(obj, list):
            for child in obj:
                visit(child)
    visit(value)
    return ids, questions


def prior_inventory(root=ROOT, destination=HERE):
    """All structured prior study records, including final tests, probes and audits.

    Raw official pools are deliberately not exposures. Python source literals are
    not parsed as observations; executed synthetic/judge panels are retained in
    their structured case/response artifacts. No row selection uses an outcome.
    """
    parent = parent_module()
    roots = [root / name for name in ("data", "artifacts", "experiments", "analysis", "tests")]
    files = sorted({p for directory in roots for p in directory.rglob("*")
                    if p.is_file() and p.suffix in parent.SUFFIXES | {".ipynb"}
                    and not p.is_relative_to(destination)
                    and not p.is_relative_to(root / "data/raw")
                    and not any(part.startswith("checkpoint-") or part in {".cache", "__pycache__"} for part in p.parts)})
    ids, questions, inventory = set(), set(), []
    for path in files:
        before = file_digest(path)
        local_ids, local_questions = set(), set()
        records = [read_json(path)] if path.suffix == ".ipynb" else parent.records(path)
        for record in records:
            # Parent parser is reused without repeatedly importing it per row.
            ii, qq = parent.identities(record)
            def extra(obj):
                if isinstance(obj, dict):
                    for key, child in obj.items():
                        if str(key).strip().lower() in {"question_text", "original_question", "pregunta", "query"} and isinstance(child, str) and child.strip():
                            qq.add(normalize_question(child))
                        if isinstance(child, (list, dict)):
                            extra(child)
                elif isinstance(obj, list):
                    for child in obj:
                        extra(child)
            extra(record)
            local_ids.update(ii); local_questions.update(qq)
        if file_digest(path) != before:
            raise ValueError(f"Prior source changed during inventory: {path}")
        ids.update(local_ids); questions.update(local_questions)
        inventory.append({"path": str(path.relative_to(root)), "sha256": before,
                          "ids": len(local_ids), "normalized_questions": len(local_questions)})
    return ids, questions, inventory


def select_fresh(trivia, nq, prior_ids, prior_questions, counts=None, diagnostic=DIAGNOSTIC, seed=SEED):
    counts = TEST_COUNTS if counts is None else counts
    if set(counts) != {"test_trivia", "test_nq"} or not 0 < diagnostic <= min(counts.values()):
        raise ValueError("Invalid fixed evaluation populations")
    seen_ids, seen_questions = set(prior_ids), set(prior_questions)
    splits, statistics = {}, {}
    for name, source in (("test_nq", nq), ("test_trivia", trivia)):
        available, by_id, by_question, duplicates = [], 0, 0, 0
        for row in ordered(source, seed, "v2-reserve-" + name):
            question = normalize_question(row["question"])
            if row["id"] in prior_ids or question in prior_questions:
                by_id += int(row["id"] in prior_ids); by_question += int(question in prior_questions)
                continue
            if row["id"] in seen_ids or question in seen_questions:
                duplicates += 1
                continue
            seen_ids.add(row["id"]); seen_questions.add(question)
            available.append(row)
        if len(available) < counts[name]:
            raise ValueError(f"Only {len(available)} fresh questions in {name}; need {counts[name]}; do not silently reduce the test")
        splits[name] = available[:counts[name]]
        statistics[name] = {"clean_source_rows": len(source), "available_after_exclusion": len(available),
                            "matched_prior_id": by_id, "matched_prior_normalized_question": by_question,
                            "within_or_cross_duplicates": duplicates, "selected": counts[name],
                            "remaining_after_selection": len(available) - counts[name]}
    diagnostics = {name: [r["id"] for r in ordered(rows, seed, "v2-diagnostic-" + name)[:diagnostic]]
                   for name, rows in splits.items()}
    return splits, diagnostics, statistics


def validate_fresh(splits, ids, questions, diagnostics, counts=None, diagnostic=DIAGNOSTIC):
    counts = TEST_COUNTS if counts is None else counts
    seen_ids, seen_questions = set(ids), set(questions)
    if set(splits) != set(counts) or set(diagnostics) != set(counts):
        raise ValueError("Missing evaluation split")
    for name, rows in splits.items():
        if len(rows) != counts[name]:
            raise ValueError(f"Wrong test population: {name}")
        for row in rows:
            q = normalize_question(row["question"])
            if row["id"] in seen_ids or q in seen_questions:
                raise ValueError(f"Evaluation overlap: {row['id']}")
            if not q or not row.get("aliases"):
                raise ValueError("Missing reference/question in prepared source")
            seen_ids.add(row["id"]); seen_questions.add(q)
        chosen = diagnostics[name]
        if len(chosen) != diagnostic or len(set(chosen)) != diagnostic or not set(chosen) <= {r["id"] for r in rows}:
            raise ValueError(f"Invalid diagnostic IDs: {name}")


def training_subsets(rows):
    if len(rows) != 2000 or len({r["id"] for r in rows}) != 2000:
        raise ValueError("Expected the original 2,000 unique TRAIN questions")
    selected = ordered(rows, SEED, "v2-rl-train")
    return selected[:1008], selected[:504]


def training_schedule(rows, seed, exposure_mode):
    """Contiguous Q0 slots0..2, Q1 slots0..2; each two questions share an update.

    A slot is eight generations. The trainer may partition those eight into G4
    groups but must never mix slots, or turn two G4 groups into two updates.
    """
    if exposure_mode not in {"single_tau", "paired_tau"} or seed not in SEEDS:
        raise ValueError("Unknown exposure mode or unregistered training seed")
    if not rows or len(rows) % 6 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Training questions must be unique and divisible by six")
    assigned = {r["id"]: TAUS[i % 3] for i, r in enumerate(ordered(rows, seed, "v2-single-tau-assignment"))}
    result = []
    for qi, row in enumerate(ordered(rows, seed, "v2-question-order")):
        for slot, paired_tau in enumerate(TAUS):
            result.append({**row, "tau": assigned[row["id"]] if exposure_mode == "single_tau" else paired_tau,
                           "slot": slot, "question_index": qi, "update_index": qi // 2,
                           "exposure_id": digest([row["id"], slot])})
    return result


def training_rows(arm, seed, n_questions=1008, destination=HERE):
    if arm not in ARMS or n_questions not in {1008, 504}:
        raise ValueError("Unknown canonical arm or training budget")
    name = "train" if n_questions == 1008 else "train_reduced"
    path = destination / f"data/prepared/{name}.jsonl"
    info = read_json(destination / "data/prepared/manifest.json")["splits"][name]
    rows = read_jsonl(path)
    if file_digest(path) != info["sha256"] or len(rows) != n_questions or [r["id"] for r in rows] != info["ids"]:
        raise ValueError("The selected training split changed")
    return training_schedule(rows, seed, ARMS[arm])


def load_sources(root=ROOT):
    """The same pinned official caches used by the parent evaluation; no network."""
    original_path = root / "data/prepared/manifest.json"
    final_path = root / "experiments/final-eval-20261001/data/prepared/manifest.json"
    original, final = read_json(original_path), read_json(final_path)
    p = original["provenance"]
    if p["kind"] != "official" or final["provenance"]["trivia_revision"] != p["trivia_revision"]:
        raise ValueError("Expected matching pinned official sources")
    parquet = root / final["provenance"]["trivia_source_file"]
    nq_path = root / "data/raw/NQ-open.dev.jsonl"
    protected = {str(original_path.relative_to(root)): file_digest(original_path),
                 str(final_path.relative_to(root)): file_digest(final_path),
                 str(parquet.relative_to(root)): final["provenance"]["trivia_source_sha256"],
                 str(nq_path.relative_to(root)): p["nq_sha256"]}
    reused = {}
    for old_name, new_name in (("train", "train_full"), ("dev", "dev"), ("calibration", "calibration")):
        path = root / f"data/prepared/{old_name}.jsonl"
        protected[str(path.relative_to(root))] = original["splits"][old_name]["sha256"]
        reused[new_name] = path
    for relative, checksum in protected.items():
        if not (root / relative).is_file() or file_digest(root / relative) != checksum:
            raise ValueError(f"Missing or changed pinned source: {relative}; downloads are forbidden")
    import pyarrow.parquet as pq
    raw_trivia, raw_nq = pq.read_table(parquet).to_pylist(), read_jsonl(nq_path)
    if len(raw_nq) != 3610:
        raise ValueError("Expected 3,610 rows of NQ-Open Original Dev")
    return clean_rows(raw_trivia, "trivia"), clean_rows(raw_nq, "nq"), reused, protected, final["provenance"]


def verify(destination=HERE, root=ROOT):
    directory = destination / "data/prepared"
    manifest = read_json(directory / "manifest.json")
    if manifest["script_sha256"] != file_digest(Path(__file__)):
        raise ValueError("Data preparation implementation changed after snapshot")
    for relative, checksum in manifest["protected_inputs"].items():
        if file_digest(root / relative) != checksum:
            raise ValueError(f"Changed protected input: {relative}")
    data = {}
    if set(manifest["splits"]) != set(COUNTS):
        raise ValueError("Prepared split set differs from the v2 contract")
    for name, info in manifest["splits"].items():
        path = directory / f"{name}.jsonl"; rows = read_jsonl(path)
        if file_digest(path) != info["sha256"] or len(rows) != COUNTS[name] or [r["id"] for r in rows] != info["ids"]:
            raise ValueError(f"Changed prepared split: {name}")
        data[name] = rows
    for new, old in (("train_full", "train"), ("dev", "dev"), ("calibration", "calibration")):
        if file_digest(directory / f"{new}.jsonl") != file_digest(root / f"data/prepared/{old}.jsonl"):
            raise ValueError("Original TRAIN/dev/calibration partitions must remain byte-identical")
    full, reduced = training_subsets(data["train_full"])
    if data["train"] != full or data["train_reduced"] != reduced:
        raise ValueError("TRAIN subset/nesting differs from its registered hash selection")
    seen_ids, seen_q = set(), set()
    for name in ("train_full", "dev", "calibration"):
        for row in data[name]:
            q = normalize_question(row["question"])
            if row["id"] in seen_ids or q in seen_q:
                raise ValueError("Original learning/calibration partitions overlap")
            seen_ids.add(row["id"]); seen_q.add(q)
    exclusions_path = directory / "exclusions.json"
    if file_digest(exclusions_path) != manifest["exclusions_sha256"]:
        raise ValueError("Exclusion snapshot changed")
    excluded = read_json(exclusions_path)
    if not fixed_prompt_questions() <= set(excluded["normalized_questions"]):
        raise ValueError("Exclusion snapshot omitted fixed prompt examples")
    if not seen_ids <= set(excluded["ids"]) or not seen_q <= set(excluded["normalized_questions"]):
        raise ValueError("Exclusion inventory omitted learning/calibration data")
    validate_fresh({k: data[k] for k in TEST_COUNTS}, set(excluded["ids"]), set(excluded["normalized_questions"]), manifest["diagnostic_ids"])
    for name in ("train", "train_reduced"):
        for seed in SEEDS:
            for mode in ("single_tau", "paired_tau"):
                schedule = training_schedule(data[name], seed, mode)
                if digest(schedule) != manifest["schedule_sha256"][f"{name}:{seed}:{mode}"]:
                    raise ValueError("Training exposure schedule changed")
    return manifest


def prepare(destination=HERE, root=ROOT):
    directory = destination / "data/prepared"
    if (directory / "manifest.json").exists():
        return verify(destination, root)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Partial v2 data snapshot exists; explicit recovery required, never overwrite")
    trivia, nq, reused, protected, provenance = load_sources(root)
    prior_ids, prior_questions, inventory = prior_inventory(root, destination)
    prompt_path = root / "src/abstention/prompts.py"
    prior_questions.update(fixed_prompt_questions())
    inventory.append({"path": str(prompt_path.relative_to(root)), "sha256": file_digest(prompt_path),
                      "ids": 0, "normalized_questions": len(fixed_prompt_questions()),
                      "role": "Fixed user-question examples shared by forced and threshold prompts"})
    splits, diagnostics, statistics = select_fresh(trivia, nq, prior_ids, prior_questions)
    validate_fresh(splits, prior_ids, prior_questions, diagnostics)
    train_full = read_jsonl(reused["train_full"])
    splits["train"], splits["train_reduced"] = training_subsets(train_full)
    splits.update({name: read_jsonl(path) for name, path in reused.items()})
    for name, rows in splits.items():
        if len(rows) != COUNTS[name]:
            raise ValueError(f"Incorrect original partition size: {name}")
    # Only identity-bearing inventory files need remain immutable. Metadata-only
    # files are snapshotted for inspection but cannot affect question exclusion.
    protected.update({r["path"]: r["sha256"] for r in inventory if r["ids"] or r["normalized_questions"]})
    for path in (PARENT / "prepare_data.py", root / "src/abstention/data.py", root / "src/abstention/scoring.py", root / "src/abstention/io.py"):
        protected[str(path.relative_to(root))] = file_digest(path)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".v2-prepare-") as temporary:
        staged = Path(temporary)
        write_json(staged / "exclusions.json", {"ids": sorted(prior_ids), "normalized_questions": sorted(prior_questions), "inventory": inventory})
        for name, rows in splits.items():
            if name in reused:
                shutil.copyfile(reused[name], staged / f"{name}.jsonl")
            else:
                write_jsonl(staged / f"{name}.jsonl", rows)
        manifest = {"schema_version": 2, "created_at": utcnow(), "selection_seed": SEED,
                    "script_sha256": file_digest(Path(__file__)), "provenance": provenance,
                    "parent_evaluation": "experiments/final-eval-20261001", "protected_inputs": protected,
                    "selection_rule": "Identity hash order only; exclude all prior IDs and normalized questions; all eligible NQ questions reserved against TriviaQA; no correctness-based selection",
                    "freshness_limit": "Disjoint within this study under the registered ID/question normalization; neither semantic equivalence nor pretraining contamination is ruled out",
                    "test_access": "Reserved, not inferred; model inference requires the separate v2 protocol/implementation lock",
                    "previous_pool_estimate_before_revalidation": {"test_trivia": 8309, "test_nq": 610},
                    "selection_statistics": statistics, "diagnostic_ids": diagnostics,
                    "exclusions_sha256": file_digest(staged / "exclusions.json"),
                    "excluded_unique_ids": len(prior_ids), "excluded_unique_normalized_questions": len(prior_questions),
                    "fixed_prompt_questions_excluded": sorted(fixed_prompt_questions()),
                    "exclusion_inventory_files": len(inventory),
                    "training": {"thresholds": list(TAUS), "mean_threshold": .75, "seeds": list(SEEDS), "slots_per_question": 3,
                                 "generations_per_slot": 8, "questions_per_update": 2, "generations_per_update": 48,
                                 "updates_full": 504, "updates_reduced": 252, "arms": ARMS,
                                 "question_order": "Identical per seed across arms; contiguous three slots per question; no outcome-dependent sampling"},
                    "schedule_sha256": {f"{name}:{seed}:{mode}": digest(training_schedule(splits[name], seed, mode))
                                        for name in ("train", "train_reduced") for seed in SEEDS for mode in ("single_tau", "paired_tau")},
                    "splits": {name: {"count": len(rows), "ids": [r["id"] for r in rows], "sha256": file_digest(staged / f"{name}.jsonl")}
                               for name, rows in splits.items()}}
        write_json(staged / "manifest.json", manifest)
        for relative, checksum in protected.items():
            if file_digest(root / relative) != checksum:
                raise ValueError(f"Source changed before publication: {relative}")
        directory.mkdir(exist_ok=True)
        for path in sorted(staged.iterdir(), key=lambda p: p.name == "manifest.json"):
            os.link(path, directory / path.name)  # fails atomically on collision; never overwrites annotations/data
    return verify(destination, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify() if args.verify else prepare()
    print(json.dumps({"counts": {k: v["count"] for k, v in result["splits"].items()},
                      "selection_statistics": result["selection_statistics"],
                      "manifest_sha256": file_digest(HERE / "data/prepared/manifest.json")}, indent=2))
