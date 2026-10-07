"""Privacy and integrity checks for the release boundary, using synthetic inputs."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

spec = importlib.util.spec_from_file_location("public_bundle", Path(__file__).parents[1] / "scripts/public_bundle.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    files = {"source.py": b"print('frozen')\n", "data/private.jsonl": b'{"answer":"private"}\n',
             b.EDITORIAL_INPUT: b"Original editorial document"}
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    lock = b.json_bytes({"files": {name: b.sha(raw) for name, raw in files.items()}})
    (root / b.SOURCE_LOCK).parent.mkdir(parents=True, exist_ok=True)
    (root / b.SOURCE_LOCK).write_bytes(lock)
    monkeypatch.setattr(b, "SOURCE_LOCK_SHA256", b.sha(lock))
    monkeypatch.setattr(b, "PUBLIC_FILES", ("source.py", b.SOURCE_LOCK))
    return root


def rewrite_archive(path, mutate):
    with tarfile.open(path, "r:gz") as archive:
        payload = {member.name: archive.extractfile(member).read() for member in archive}
    mutate(payload)
    with tarfile.open(path, "w:gz") as archive:
        for name, raw in payload.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))


def test_deterministic_bundle_and_explicit_omissions(project, tmp_path):
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    one = b.export_bundle(project, first)
    two = b.export_bundle(project, second)
    assert one["archive_sha256"] == two["archive_sha256"]
    assert one["editorial_files"] == one["private_data_files"] == 0
    with tarfile.open(first) as archive:
        names = archive.getnames()
        assert not any("paper/" in name or name.endswith(".jsonl") for name in names)
        omitted = json.load(archive.extractfile("UNBUNDLED-INPUTS.json"))["inputs"]
        assert omitted[b.EDITORIAL_INPUT]["reason"] == "editorial_excluded"
        assert omitted["data/private.jsonl"]["reason"] == "local_input_not_distributed"
        assert archive.extractfile(b.SOURCE_LOCK).read() == (project / b.SOURCE_LOCK).read_bytes()


def test_editorial_change_is_reported_but_scientific_change_blocks(project, tmp_path):
    (project / b.EDITORIAL_INPUT).write_text("Authorized edited manuscript")
    report = b.check_local(project)
    assert report["status"] == "passed_scientific_scope"
    assert report["editorial_exception"]["status"] == "changed"
    assert report["scientific_changes"] == []
    (project / "data/private.jsonl").write_text("scientific change")
    assert b.check_local(project)["status"] == "failed_scientific_scope"
    with pytest.raises(ValueError, match="Scientific source drift"):
        b.export_bundle(project, tmp_path / "blocked.tar.gz")


@pytest.mark.parametrize("path", ["paper/manuscript.tex", "copy.tex.gz", "secret.xlsx",
                                    "artifacts/audit-v2-20261004/key.json", "../source.py"])
def test_private_or_traversal_member_rejected_even_with_new_manifest(project, tmp_path, path):
    target = tmp_path / "release.tar.gz"
    b.export_bundle(project, target)
    def mutate(payload):
        payload[path] = b"private"
        manifest = json.loads(payload["EXPORT-MANIFEST.json"])
        manifest["files"][path] = {"sha256": b.sha(b"private"), "bytes": 7}
        payload["EXPORT-MANIFEST.json"] = b.json_bytes(manifest)
    rewrite_archive(target, mutate)
    with pytest.raises(ValueError):
        b.verify_archive(target)


def test_frozen_content_rewrite_cannot_be_hidden_by_export_manifest(project, tmp_path):
    target = tmp_path / "release.tar.gz"
    b.export_bundle(project, target)
    def mutate(payload):
        payload["source.py"] = b"changed"
        manifest = json.loads(payload["EXPORT-MANIFEST.json"])
        manifest["files"]["source.py"] = {"sha256": b.sha(b"changed"), "bytes": 7}
        payload["EXPORT-MANIFEST.json"] = b.json_bytes(manifest)
    rewrite_archive(target, mutate)
    with pytest.raises(ValueError, match="Frozen scientific source mismatch"):
        b.verify_archive(target)


def test_corrupted_archive_hash_rejected(project, tmp_path):
    target = tmp_path / "release.tar.gz"
    b.export_bundle(project, target)
    rewrite_archive(target, lambda payload: payload.update({"source.py": b"changed"}))
    with pytest.raises(ValueError, match="hash/size mismatch"):
        b.verify_archive(target)


def test_symlink_sources_and_overwriting_release_are_forbidden(project, tmp_path):
    target = tmp_path / "release.tar.gz"
    b.export_bundle(project, target)
    before = target.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        b.export_bundle(project, target)
    assert target.read_bytes() == before
    source = project / "source.py"
    source.rename(project / "real-source.py")
    source.symlink_to("real-source.py")
    with pytest.raises(ValueError, match="Scientific source drift"):
        b.export_bundle(project, tmp_path / "symlink.tar.gz")


def test_changed_lock_and_missing_required_member_fail(project, tmp_path):
    lock_path = project / b.SOURCE_LOCK
    original = lock_path.read_bytes()
    lock_path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="source-lock hash changed"):
        b.check_local(project)
    lock_path.write_bytes(original)
    (project / "source.py").unlink()
    with pytest.raises(ValueError, match="Scientific source drift"):
        b.export_bundle(project, tmp_path / "missing.tar.gz")


def test_new_exporter_can_verify_unchanged_previous_release(project, tmp_path, monkeypatch):
    target = tmp_path / "old.tar.gz"
    original = b.export_bundle(project, target)
    monkeypatch.setattr(b, "LEGACY_PUBLIC_FILES", b.PUBLIC_FILES)
    monkeypatch.setattr(b, "PUBLIC_FILES", (*b.PUBLIC_FILES, "new-code.py"))
    assert b.verify_archive(target)["archive_sha256"] == original["archive_sha256"]


def test_audit_aggregate_and_code_are_bound_to_public_provenance(project, tmp_path, monkeypatch):
    files = {f"{b.AUDIT_PUBLIC}/{name}": b"aggregate-only\n" for name in b.AUDIT_AGGREGATES}
    code = {"analysis/audit_v2_sensitivity.py": b"# synthetic source\n",
            "tests/test_audit_v2_sensitivity.py": b"# synthetic tests\n"}
    files.update(code)
    files[f"{b.AUDIT_PUBLIC}/provenance.json"] = b.json_bytes({
        "status": "aggregate_copies_verified",
        "files": {name: b.sha(files[f"{b.AUDIT_PUBLIC}/{name}"]) for name in b.AUDIT_AGGREGATES},
        "analysis_source_sha256": {name: b.sha(raw) for name, raw in code.items()},
    })
    for name, raw in files.items():
        path = project / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    monkeypatch.setattr(b, "PUBLIC_FILES", (*b.PUBLIC_FILES, *files))
    target = tmp_path / "audit.tar.gz"
    b.export_bundle(project, target)
    def mutate(payload):
        name = f"{b.AUDIT_PUBLIC}/audit-diagnostics.csv"
        payload[name] = b"changed aggregate\n"
        manifest = json.loads(payload["EXPORT-MANIFEST.json"])
        manifest["files"][name] = {"sha256": b.sha(payload[name]), "bytes": len(payload[name])}
        payload["EXPORT-MANIFEST.json"] = b.json_bytes(manifest)
    rewrite_archive(target, mutate)
    with pytest.raises(ValueError, match="Public audit aggregate differs"):
        b.verify_archive(target)
