"""Freeze fresh, prediction-independent evaluation questions from official local caches.

CPU only. No network access, Hugging Face discovery, model loading, or training.
The input inventory extracts identities/questions only: outcomes and predictions
never affect selection. Existing evaluation questions are considered exposed even
when they appeared in a pilot, an audit, or an incomplete historical run.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from abstention.data import clean_rows
from abstention.io import digest, file_digest, ordered, read_json, read_jsonl, utcnow, write_json, write_jsonl
from abstention.scoring import normalize_question

SEED = 20261001
N_PER_DATASET = 2000
N_DIAGNOSTIC = 200
ID_RE = re.compile(r"^(?:trivia|nq):[^\s]+$")
SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".xlsx"}


def identities(value):
    """Project a record/tree onto question identities, never outcome-dependent."""
    ids, questions = set(), set()

    def id_values(obj):
        if isinstance(obj, str) and ID_RE.fullmatch(obj):
            ids.add(obj)
        elif isinstance(obj, list):
            for child in obj:
                id_values(child)
        elif isinstance(obj, dict):
            for child in obj.values():
                id_values(child)

    def visit(obj):
        if isinstance(obj, dict):
            for key, child in obj.items():
                key = str(key).strip().lower()
                if key in {"id", "question_id", "source_id", "ids", "diagnostic_ids", "question_ids"}:
                    id_values(child)
                if key == "question" and isinstance(child, str) and child.strip():
                    questions.add(normalize_question(child))
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(obj, list):
            for child in obj:
                visit(child)

    visit(value)
    return ids, questions


def spreadsheet_records(path):
    """Read the existing audit workbooks without introducing an openpyxl dependency."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(x.itertext()) for x in root.findall("s:si", ns)]
        for name in sorted(archive.namelist()):
            if not re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name):
                continue
            header = None
            for row in ET.fromstring(archive.read(name)).findall(".//s:row", ns):
                cells = {}
                for cell in row.findall("s:c", ns):
                    column = re.sub(r"\d+", "", cell.attrib["r"])
                    node = cell.find("s:v", ns)
                    text = node.text if node is not None else ""
                    if cell.get("t") == "s":
                        text = strings[int(text)]
                    elif cell.get("t") == "inlineStr":
                        inline = cell.find("s:is", ns)
                        text = "".join(inline.itertext()) if inline is not None else ""
                    cells[column] = text or ""
                if header is None or any(v.strip().lower() in {"question", "question_id"} for v in cells.values()):
                    header = cells
                else:
                    yield {header[c]: value for c, value in cells.items() if c in header}


def records(path):
    if path.suffix == ".json":
        yield read_json(path)
    elif path.suffix == ".jsonl":
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix in {".csv", ".tsv"}:
        with path.open(newline="") as stream:
            yield from csv.DictReader(stream, delimiter="\t" if path.suffix == ".tsv" else ",")
    else:
        yield from spreadsheet_records(path)


def prior_inventory(root=ROOT, destination=HERE):
    """Inventory all previous prepared data and structured experimental artifacts."""
    ids, questions, inventory = set(), set(), []
    roots = [root / "data/prepared", root / "artifacts", root / "experiments"]
    files = sorted({p for directory in roots for p in directory.rglob("*")
                    if p.is_file() and p.suffix in SUFFIXES
                    and not p.is_relative_to(destination)
                    and not any(part.startswith("checkpoint-") for part in p.parts)})
    for path in files:
        local_ids, local_questions = set(), set()
        for row in records(path):
            row_ids, row_questions = identities(row)
            local_ids.update(row_ids)
            local_questions.update(row_questions)
        ids.update(local_ids)
        questions.update(local_questions)
        inventory.append({"path": str(path.relative_to(root)), "sha256": file_digest(path),
                          "ids": len(local_ids), "normalized_questions": len(local_questions)})
    return ids, questions, inventory


def select_fresh(trivia, nq, prior_ids, prior_questions, count=N_PER_DATASET, diagnostic=N_DIAGNOSTIC, seed=SEED):
    """NQ-first deduplication, matching v1's source priority; no answer-based sampling."""
    if not 0 < diagnostic <= count:
        raise ValueError("Diagnostic count must be positive and no larger than the test count")
    seen_ids, seen_questions = set(prior_ids), set(prior_questions)
    splits, statistics = {}, {}
    for name, source in (("test_nq", nq), ("test_trivia", trivia)):
        available, prior_id, prior_question, within_or_cross_duplicate = [], 0, 0, 0
        for row in ordered(source, seed, "fresh-" + name):
            question = normalize_question(row["question"])
            if row["id"] in prior_ids or question in prior_questions:
                prior_id += int(row["id"] in prior_ids)
                prior_question += int(question in prior_questions)
                continue
            if row["id"] in seen_ids or question in seen_questions:
                within_or_cross_duplicate += 1
                continue
            seen_ids.add(row["id"])
            seen_questions.add(question)
            available.append(row)
        if len(available) < count:
            raise ValueError(f"Only {len(available)} fresh questions in {name}; need {count}")
        splits[name] = available[:count]
        statistics[name] = {"clean_source_rows": len(source), "available_after_exclusion": len(available),
                            "matched_prior_id": prior_id, "matched_prior_normalized_question": prior_question,
                            "duplicate_within_or_across_official_sources": within_or_cross_duplicate,
                            "selected": count,
                            "note": "ID and question exclusion counts may overlap; all eligible NQ questions are reserved against TriviaQA."}
    diagnostics = {name: [r["id"] for r in ordered(rows, seed, "fresh-diagnostic-" + name)[:diagnostic]]
                   for name, rows in splits.items()}
    return splits, diagnostics, statistics


def validate_splits(splits, prior_ids, prior_questions, diagnostic_ids, count=N_PER_DATASET, diagnostic=N_DIAGNOSTIC):
    seen_ids, seen_questions = set(), set()
    for name, rows in splits.items():
        if len(rows) != count:
            raise ValueError(f"Unexpected count for {name}")
        for row in rows:
            question = normalize_question(row["question"])
            if row["id"] in seen_ids | prior_ids or question in seen_questions | prior_questions:
                raise ValueError(f"Evaluation overlap: {row['id']}")
            if not question or not row.get("aliases"):
                raise ValueError(f"Missing question/aliases: {row['id']}")
            seen_ids.add(row["id"])
            seen_questions.add(question)
        diagnostics = diagnostic_ids[name]
        if len(diagnostics) != diagnostic or len(set(diagnostics)) != diagnostic or not set(diagnostics) <= {r["id"] for r in rows}:
            raise ValueError(f"Invalid diagnostic subset: {name}")


def verify(destination=HERE, root=ROOT):
    directory = destination / "data/prepared"
    manifest = read_json(directory / "manifest.json")
    for path, expected in manifest["protected_inputs"].items():
        if file_digest(root / path) != expected:
            raise ValueError(f"Changed protected input: {path}")
    for name, info in manifest["splits"].items():
        path = directory / f"{name}.jsonl"
        rows = read_jsonl(path)
        if file_digest(path) != info["sha256"] or [r["id"] for r in rows] != info["ids"] or len(rows) != info["count"]:
            raise ValueError(f"Changed prepared split: {name}")
    exclusion = read_json(directory / "exclusions.json")
    if file_digest(directory / "exclusions.json") != manifest["exclusions_sha256"]:
        raise ValueError("Changed exclusion snapshot")
    splits = {name: read_jsonl(directory / f"{name}.jsonl") for name in ("test_nq", "test_trivia")}
    validate_splits(splits, set(exclusion["ids"]), set(exclusion["normalized_questions"]), manifest["diagnostic_ids"])
    if file_digest(directory / "calibration.jsonl") != file_digest(root / "data/prepared/calibration.jsonl"):
        raise ValueError("Calibration must be the unchanged v1 partition")
    return manifest


def prepare(destination=HERE, root=ROOT):
    directory = destination / "data/prepared"
    if (directory / "manifest.json").exists():
        return verify(destination, root)
    old_manifest_path = root / "data/prepared/manifest.json"
    old = read_json(old_manifest_path)
    provenance = old["provenance"]
    if provenance["kind"] != "official":
        raise ValueError("Expected official pinned v1 sources")
    revision = provenance["trivia_revision"]
    parquet = root / ".cache/huggingface/hub/datasets--mandarjoshi--trivia_qa/snapshots" / revision / provenance["trivia_config"] / "validation-00000-of-00001.parquet"
    nq_path = root / "data/raw/NQ-open.dev.jsonl"
    if not parquet.is_file() or not nq_path.is_file():
        raise FileNotFoundError("Official sources are not cached; this script never downloads data")
    if file_digest(nq_path) != provenance["nq_sha256"]:
        raise ValueError("NQ cache differs from the frozen official source")
    for name, spec in old["splits"].items():
        if file_digest(root / "data/prepared" / f"{name}.jsonl") != spec["sha256"]:
            raise ValueError(f"Changed original data split: {name}")
    import pyarrow.parquet as pq
    raw_trivia, raw_nq = pq.read_table(parquet).to_pylist(), read_jsonl(nq_path)
    if len(raw_nq) != 3610:
        raise ValueError("Expected NQ-Open Original Dev with 3,610 rows")
    prior_ids, prior_questions, inventory = prior_inventory(root, destination)
    splits, diagnostics, stats = select_fresh(clean_rows(raw_trivia, "trivia"), clean_rows(raw_nq, "nq"), prior_ids, prior_questions)
    validate_splits(splits, prior_ids, prior_questions, diagnostics)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "exclusions.json", {"ids": sorted(prior_ids), "normalized_questions": sorted(prior_questions), "inventory": inventory})
    for name, rows in splits.items():
        write_jsonl(directory / f"{name}.jsonl", rows)
    shutil.copyfile(root / "data/prepared/calibration.jsonl", directory / "calibration.jsonl")
    splits["calibration"] = read_jsonl(directory / "calibration.jsonl")
    protected = {str(path.relative_to(root)): file_digest(path) for path in (
        old_manifest_path, parquet, nq_path, Path(__file__).resolve(), root / "src/abstention/data.py",
        root / "src/abstention/scoring.py", root / "src/abstention/io.py")}
    protected.update({row["path"]: row["sha256"] for row in inventory})
    manifest = {
        "schema_version": 1, "created_at": utcnow(), "selection_seed": SEED,
        "selection_rule": "SHA256 ordered question IDs; prior IDs and normalized questions excluded; NQ-first deduplication; answers/predictions/correctness unused",
        "freshness_scope": "Not used in earlier local study artifacts; does not establish absence from model pretraining",
        "provenance": {**provenance, "trivia_source_file": str(parquet.relative_to(root)), "trivia_source_sha256": file_digest(parquet), "trivia_raw_rows": len(raw_trivia)},
        "protected_inputs": protected,
        "exclusions_sha256": file_digest(directory / "exclusions.json"),
        "excluded_unique_ids": len(prior_ids), "excluded_unique_normalized_questions": len(prior_questions),
        "exclusion_inventory_files": len(inventory), "selection_statistics": stats,
        "diagnostic_ids": diagnostics,
        "splits": {name: {"count": len(rows), "sha256": file_digest(directory / f"{name}.jsonl"), "ids": [r["id"] for r in rows],
                          "role": "reused calibration only; never evaluation" if name == "calibration" else "new held-out evaluation"}
                   for name, rows in splits.items()},
    }
    write_json(directory / "manifest.json", manifest)
    return verify(destination, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify() if args.verify else prepare()
    print(json.dumps({"counts": {k: v["count"] for k, v in result["splits"].items()}, "statistics": result["selection_statistics"],
                      "manifest_sha256": file_digest(HERE / "data/prepared/manifest.json")}, indent=2))
