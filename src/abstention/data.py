from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

from .config import fingerprint
from .io import digest, file_digest, ordered, read_json, read_jsonl, utcnow, write_json, write_jsonl
from .scoring import normalize_question

NQ_REVISION = "fb26a3073b1fe636c97302890a27b491d6530130"
NQ_URL = f"https://raw.githubusercontent.com/google-research-datasets/natural-questions/{NQ_REVISION}/nq_open/NQ-open.dev.jsonl"


def clean_rows(rows, dataset):
    cleaned = []
    for r in rows:
        question = str(r["question"]).strip()
        answer = r.get("answer", r.get("answers"))
        if isinstance(answer, dict):
            aliases = list(answer.get("aliases", [])) + list(answer.get("normalized_aliases", []))
            aliases += [answer["value"]] if answer.get("value") else []
        else:
            aliases = answer if isinstance(answer, list) else [answer]
        aliases = sorted({str(a).strip() for a in aliases if a is not None and str(a).strip()})
        if not question or not aliases:
            continue
        source_id = str(r.get("question_id", r.get("id", digest(normalize_question(question))[:24])))
        cleaned.append({"id": f"{dataset}:{source_id}", "dataset": dataset,
                        "question": question, "aliases": aliases})
    return cleaned


def partition(config, trivia_train, trivia_validation, nq):
    """Test-first deduplication; selection never uses a model prediction or correctness."""
    seen_questions, seen_ids = set(), set()
    removed = {}

    def unique(rows, name):
        result = []
        for r in ordered(rows, config["split_seed"], name):
            key = normalize_question(r["question"])
            if key in seen_questions or r["id"] in seen_ids:
                removed[name] = removed.get(name, 0) + 1
                continue
            seen_questions.add(key)
            seen_ids.add(r["id"])
            result.append(r)
        return result

    # Reserve every official evaluation question against training overlap, not only selected tests.
    nq = unique(nq, "nq")
    trivia_validation = unique(trivia_validation, "trivia_validation")
    trivia_train = unique(trivia_train, "trivia_train")
    n = config["data"]
    if len(nq) < n["test_nq"] or len(trivia_validation) < n["test_trivia"]:
        raise ValueError("Insufficient disjoint test questions.")
    splits = {"test_nq": nq[:n["test_nq"]], "test_trivia": trivia_validation[:n["test_trivia"]]}
    offset = 0
    for name in ("pilot", "dev", "calibration", "train"):
        splits[name] = trivia_train[offset:offset + n[name]]
        offset += n[name]
        if len(splits[name]) != n[name]:
            raise ValueError(f"Insufficient disjoint questions for {name}.")
    return splits, removed


def prepare(config, workdir, local_source=None):
    workdir = Path(workdir)
    directory = workdir / "data/prepared"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = verify_data(config, workdir)
        return manifest
    if local_source:
        source = Path(local_source)
        raw_train = read_jsonl(source / "trivia_train.jsonl")
        raw_val = read_jsonl(source / "trivia_validation.jsonl")
        raw_nq = read_jsonl(source / "nq_dev.jsonl")
        provenance = {"kind": "local", "sources": {p.name: file_digest(p) for p in source.glob("*.jsonl")}}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        repo = "mandarjoshi/trivia_qa"
        revision = HfApi().dataset_info(repo).sha
        ds = load_dataset(repo, "unfiltered.nocontext", revision=revision,
                          cache_dir=str(workdir / ".cache/datasets"))
        raw_train, raw_val = ds["train"], ds["validation"]
        rawdir = workdir / "data/raw"
        rawdir.mkdir(parents=True, exist_ok=True)
        nq_path = rawdir / "NQ-open.dev.jsonl"
        if not nq_path.exists():
            with urllib.request.urlopen(NQ_URL, timeout=120) as response:
                content = response.read()
            # The original source is plain JSONL; tolerate a gzip transfer artifact explicitly.
            if content[:2] == b"\x1f\x8b":
                content = gzip.decompress(content)
            nq_path.write_bytes(content)
        raw_nq = read_jsonl(nq_path)
        if len(raw_nq) != 3610:
            raise ValueError("Expected NQ-Open Original Dev (3,610 rows); source changed.")
        provenance = {"kind": "official", "trivia_repo": repo, "trivia_revision": revision,
                      "trivia_config": "unfiltered.nocontext", "nq_url": NQ_URL,
                      "nq_sha256": file_digest(nq_path), "nq_rows": len(raw_nq)}
    splits, removed = partition(config, clean_rows(raw_train, "trivia"),
                                clean_rows(raw_val, "trivia"), clean_rows(raw_nq, "nq"))
    manifest = {"created_at": utcnow(), "config_sha256": fingerprint(config),
                "provenance": provenance, "duplicates_removed": removed, "splits": {}}
    for name, rows in splits.items():
        path = directory / f"{name}.jsonl"
        write_jsonl(path, rows)
        manifest["splits"][name] = {"count": len(rows), "sha256": file_digest(path), "ids": [r["id"] for r in rows]}
    manifest["diagnostic_ids"] = {
        name: [r["id"] for r in ordered(splits[name], config["split_seed"], "diagnostic")[:config["data"]["diagnostic"]]]
        for name in ("test_trivia", "test_nq")}
    write_json(directory / "manifest.json", manifest)
    return manifest


def verify_data(config, workdir):
    directory = Path(workdir) / "data/prepared"
    manifest = read_json(directory / "manifest.json")
    if manifest["config_sha256"] != fingerprint(config):
        raise ValueError("Prepared data belongs to a different configuration.")
    for name, info in manifest["splits"].items():
        if file_digest(directory / f"{name}.jsonl") != info["sha256"]:
            raise ValueError(f"Data changed after preparation: {name}")
    return manifest


def load_split(workdir, name):
    return read_jsonl(Path(workdir) / "data/prepared" / f"{name}.jsonl")


def training_rows(rows, thresholds, seed):
    return [{**r, "tau": thresholds[i % len(thresholds)]} for i, r in enumerate(ordered(rows, seed, "tau-assignment"))]
