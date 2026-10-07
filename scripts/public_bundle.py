#!/usr/bin/env python3
"""Create a closed, deterministic code/aggregate bundle. Never publish or extract it.

Historical source locks remain byte-exact. The manuscript is deliberately absent;
its later editorial changes are reported separately from scientific modifications.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
V2 = "experiments/v2-20261002"
SOURCE_LOCK = f"{V2}/artifacts/source-lock.json"
SOURCE_LOCK_SHA256 = "b2db40510e951172fb5d32386dfec3be5d7d88268c734a1d337ce82c8b255b30"
EDITORIAL_INPUT = "paper/manuscript.tex"
GENERATED = frozenset({"EXPORT-MANIFEST.json", "UNBUNDLED-INPUTS.json"})
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024

# Deliberately no globbing: new files never enter a release automatically.
LEGACY_PUBLIC_FILES = tuple(sorted([
    "README.md", ".gitignore", ".dockerignore", "Dockerfile", "pyproject.toml",
    "configs/study.yaml", "docs/reproducibilidad.md", "docs/tres-mejoras-v2.md",
    "docs/registro-experimentos.md", "docs/auditoria-v2.md",
    "analysis/audit_v2.py", "tests/test_audit_v2.py", "tests/test_public_bundle.py",
    "scripts/public_bundle.py", "scripts/start-v2.sh",
    "experiments/final-eval-20261001/prepare_data.py",
    "experiments/final-eval-20261001/test_prepare_data.py",
    "experiments/final-eval-20261001/artifacts/models.lock.json",
    "data/prepared/manifest.json",
    "experiments/final-eval-20261001/data/prepared/manifest.json",
    *[f"src/abstention/{name}.py" for name in (
        "__init__", "cli", "config", "data", "evaluation", "io", "models", "prompts",
        "protocol", "reporting", "runner", "scoring", "statistics", "training")],
    *[f"{V2}/{name}.py" for name in (
        "prepare_data", "training_v2", "selector_v2", "evaluate_v2", "analysis_v2", "run",
        "tests_prepare_data", "tests_training_v2", "tests_selector_v2",
        "tests_evaluate_v2", "tests_analysis_v2", "tests_run")],
    f"{V2}/config.yaml", f"{V2}/contract.json", f"{V2}/data/prepared/manifest.json",
    SOURCE_LOCK, f"{V2}/artifacts/evaluation-lock.json", f"{V2}/artifacts/budget-lock.json",
    f"{V2}/artifacts/status.json", f"{V2}/artifacts/analysis/complete.json",
    *[f"{V2}/artifacts/reviews/2026-10-04-completion/{name}.json"
      for name in ("technical", "scientific", "visual")],
    *[f"{V2}/artifacts/analysis/{name}.csv" for name in (
        "contrasts", "joint-primary", "aggregate-metrics", "calibration", "forced",
        "threshold-response", "training-runs", "job-costs-snapshot")],
    f"{V2}/artifacts/analysis/results.md", f"{V2}/artifacts/analysis/captions.json",
    *[f"{V2}/artifacts/analysis/{name}.{ext}" for name in (
        "utility", "coverage", "selective_risk", "risk-coverage", "calibration")
      for ext in ("png", "pdf")],
]))
AUDIT_PUBLIC = "results/v2-audit-20261006"
AUDIT_AGGREGATES = (
    "audit-diagnostics.csv", "method-sensitivity.csv", "contrast-sensitivity.csv",
    "absolute-utility-sensitivity.csv", "per-seed-contrasts.csv", "summary.json",
)
PUBLIC_FILES = tuple(sorted([
    *LEGACY_PUBLIC_FILES,
    "analysis/audit_v2_sensitivity.py", "tests/test_audit_v2_sensitivity.py",
    *[f"{AUDIT_PUBLIC}/{name}" for name in (*AUDIT_AGGREGATES, "provenance.json")],
]))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    lower = name.lower()
    if (not name or path.is_absolute() or "\\" in name or
            any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"Unsafe path: {name!r}")
    forbidden_parts = {"paper", "manuscript", ".git", ".cache", ".venv", "submissions"}
    forbidden_suffixes = {".tex", ".bib", ".cls", ".sty", ".jsonl", ".xlsx", ".xls",
                          ".ipynb", ".safetensors", ".pt", ".pth", ".bin", ".parquet",
                          ".arrow", ".gz", ".zip"}
    if (any(part.lower() in forbidden_parts for part in path.parts) or
            "manuscript" in lower or "private-key" in lower or "audit-v2-20261004" in lower or
            path.suffix.lower() in forbidden_suffixes):
        raise ValueError(f"Private/editorial/unsupported file: {name}")


def regular_bytes(root: Path, relative: str) -> bytes:
    """Reject symlinks in every component, including links inside the workspace."""
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink forbidden: {relative}")
    if not path.is_file() or path.resolve().is_relative_to(root.resolve()) is False:
        raise ValueError(f"Missing/nonregular source: {relative}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Oversized source: {relative}")
    return path.read_bytes()


def checked_source_lock(root: Path) -> tuple[dict, bytes]:
    raw = regular_bytes(root, SOURCE_LOCK)
    if sha(raw) != SOURCE_LOCK_SHA256:
        raise ValueError("Historical source-lock hash changed; do not rewrite the lock")
    lock = json.loads(raw)
    if not isinstance(lock.get("files"), dict) or EDITORIAL_INPUT not in lock["files"]:
        raise ValueError("Invalid historical source lock")
    return lock, raw


def check_local(root: Path = ROOT) -> dict:
    lock, raw = checked_source_lock(root)
    changes, checked = [], 0
    for name, expected in lock["files"].items():
        if name == EDITORIAL_INPUT:
            continue
        try:
            actual = sha(regular_bytes(root, name))
        except ValueError as exc:
            actual = None
            changes.append({"path": name, "expected": expected, "error": str(exc)})
        else:
            if actual != expected:
                changes.append({"path": name, "expected": expected, "actual": actual})
        checked += 1
    editorial_path = root / EDITORIAL_INPUT
    editorial_actual = sha(regular_bytes(root, EDITORIAL_INPUT)) if editorial_path.exists() else None
    return {
        "status": "passed_scientific_scope" if not changes else "failed_scientific_scope",
        "source_lock_sha256": sha(raw), "scientific_entries_checked": checked,
        "scientific_changes": changes,
        "editorial_exception": {"path": EDITORIAL_INPUT,
            "historical_sha256": lock["files"][EDITORIAL_INPUT], "current_sha256": editorial_actual,
            "status": "absent" if editorial_actual is None else
                "unchanged" if editorial_actual == lock["files"][EDITORIAL_INPUT] else "changed",
            "distributed": False, "reason": "Local editorial document; excluded from the public code bundle"},
        "lock_rewritten": False,
    }


def omissions(lock: dict, selected: set[str]) -> dict:
    return {"schema_version": 1, "source_lock_sha256": SOURCE_LOCK_SHA256,
        "scope": "Inputs of the historical v2 source lock absent from this bundle; not a reconstruction of all runtime artifacts",
        "inputs": {name: {"sha256": expected,
            "reason": "editorial_excluded" if name == EDITORIAL_INPUT else "local_input_not_distributed"}
            for name, expected in sorted(lock["files"].items()) if name not in selected}}


def validate_payload(payload: dict[str, bytes]) -> dict:
    selected = set(payload) - GENERATED
    if selected not in (set(PUBLIC_FILES), set(LEGACY_PUBLIC_FILES)) or not GENERATED.issubset(payload):
        raise ValueError("Archive members differ from the closed public allowlist")
    for name, data in payload.items():
        safe_name(name)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"Oversized member: {name}")
    manifest = json.loads(payload["EXPORT-MANIFEST.json"])
    if manifest.get("schema_version") != 1 or set(manifest["files"]) != selected | {"UNBUNDLED-INPUTS.json"}:
        raise ValueError("Invalid export manifest")
    for name, item in manifest["files"].items():
        if item != {"sha256": sha(payload[name]), "bytes": len(payload[name])}:
            raise ValueError(f"Content hash/size mismatch: {name}")
    lock_raw = payload[SOURCE_LOCK]
    if sha(lock_raw) != SOURCE_LOCK_SHA256:
        raise ValueError("Frozen source-lock mismatch in bundle")
    lock = json.loads(lock_raw)
    for name, expected in lock["files"].items():
        if name in payload and sha(payload[name]) != expected:
            raise ValueError(f"Frozen scientific source mismatch: {name}")
    receipt_name = f"{V2}/artifacts/analysis/complete.json"
    if receipt_name in payload:
        receipt = json.loads(payload[receipt_name])
        if (receipt.get("status") != "artifacts_ready_for_review" or
                receipt["source_sha256"].get(SOURCE_LOCK) != SOURCE_LOCK_SHA256 or
                receipt["script_sha256"] != sha(payload[f"{V2}/analysis_v2.py"])):
            raise ValueError("Analysis receipt provenance mismatch")
        prefix = f"{V2}/artifacts/analysis/"
        for name, raw in payload.items():
            if name.startswith(prefix) and name != receipt_name:
                if receipt["outputs"].get(name[len(prefix):]) != sha(raw):
                    raise ValueError(f"Aggregate differs from analysis receipt: {name}")
    public_receipt = f"{AUDIT_PUBLIC}/provenance.json"
    if public_receipt in payload:
        provenance = json.loads(payload[public_receipt])
        if provenance.get("status") != "aggregate_copies_verified" or set(provenance["files"]) != set(AUDIT_AGGREGATES):
            raise ValueError("Invalid public audit provenance")
        for name, expected in provenance["files"].items():
            if sha(payload[f"{AUDIT_PUBLIC}/{name}"]) != expected:
                raise ValueError(f"Public audit aggregate differs from its source copy: {name}")
        audit_sources = {"analysis/audit_v2_sensitivity.py", "tests/test_audit_v2_sensitivity.py"}
        if set(provenance["analysis_source_sha256"]) != audit_sources:
            raise ValueError("Incomplete audit analysis provenance")
        for name, expected in provenance["analysis_source_sha256"].items():
            if sha(payload[name]) != expected:
                raise ValueError(f"Audit analysis source mismatch: {name}")
    if json.loads(payload["UNBUNDLED-INPUTS.json"]) != omissions(lock, selected):
        raise ValueError("Omitted historical inputs incorrectly documented")
    return {"status": "verified", "members": len(payload),
            "total_uncompressed_bytes": sum(map(len, payload.values())),
            "editorial_files": 0, "private_data_files": 0,
            "source_lock_sha256": SOURCE_LOCK_SHA256,
            "scope": "Package integrity and included frozen sources; omitted inputs unavailable"}


def verify_archive(path: Path) -> dict:
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Archive exceeds the size limit")
    payload = {}
    total = 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            safe_name(member.name)
            if (member.name not in set(PUBLIC_FILES) | set(LEGACY_PUBLIC_FILES) | GENERATED or member.name in payload or
                    not member.isfile() or member.size > MAX_FILE_BYTES):
                raise ValueError(f"Unexpected, duplicate or nonregular member: {member.name}")
            total += member.size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("Uncompressed package exceeds the size limit")
            payload[member.name] = archive.extractfile(member).read()
    result = validate_payload(payload)
    return {**result, "archive": str(path), "archive_sha256": sha(path.read_bytes()),
            "compressed_bytes": path.stat().st_size}


def export_bundle(root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing release: {output}")
    report = check_local(root)
    if report["scientific_changes"]:
        raise ValueError("Scientific source drift: " + json.dumps(report["scientific_changes"]))
    payload = {}
    for name in PUBLIC_FILES:
        safe_name(name)
        payload[name] = regular_bytes(root, name)
    lock = json.loads(payload[SOURCE_LOCK])
    payload["UNBUNDLED-INPUTS.json"] = json_bytes(omissions(lock, set(PUBLIC_FILES)))
    payload["EXPORT-MANIFEST.json"] = json_bytes({"schema_version": 1,
        "selection": "Closed allowlist in scripts/public_bundle.py; no recursive directory exports",
        "source_lock_sha256": SOURCE_LOCK_SHA256,
        "files": {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in sorted(payload.items())}})
    validate_payload(payload)
    # Abort when an author changes any selected file while the snapshot is read.
    for name in PUBLIC_FILES:
        if regular_bytes(root, name) != payload[name]:
            raise ValueError(f"Source changed while exporting: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for name, raw in sorted(payload.items()):
                    member = tarfile.TarInfo(name)
                    member.size, member.mode, member.mtime = len(raw), 0o644, 0
                    archive.addfile(member, io.BytesIO(raw))
    return verify_archive(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-local", help="Verify scientific lock entries; report editorial exception separately")
    export = sub.add_parser("export", help="Create a local closed bundle; never upload it")
    export.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="Inspect archive in memory without extracting files")
    verify.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        result = (check_local() if args.command == "check-local" else
                  export_bundle(ROOT, args.output) if args.command == "export" else verify_archive(args.archive))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return int(result["status"] == "failed_scientific_scope")
    except (ValueError, OSError, KeyError, json.JSONDecodeError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
