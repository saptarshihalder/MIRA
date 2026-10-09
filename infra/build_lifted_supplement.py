"""Build/verify a deterministic, code-only anonymous supplement draft (stdlib only)."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import zipfile


SOURCE_REVISION = "e1b1d9d09e98667dc4590e6f54a6c6641c3e24a9"
PANEL_REVISION = "799f473459c99e8decbb39ff9924c86a90cc8dc7"
# Explicit paths, never recursive copying or extension-based discovery.
SOURCE_FILES = (
    "experiments/anchored_cavity.py",
    "experiments/cavity_site.py",
    "experiments/lifted_cavity/airq.py",
    "experiments/lifted_cavity/anchors.py",
    "experiments/lifted_cavity/bop.py",
    "experiments/lifted_cavity/bop_run.py",
    "experiments/lifted_cavity/colab_v4.py",
    "experiments/lifted_cavity/collect.py",
    "experiments/lifted_cavity/collect_airq.py",
    "experiments/lifted_cavity/confirm2.py",
    "experiments/lifted_cavity/confirm3.py",
    "experiments/lifted_cavity/confirm4.py",
    "experiments/lifted_cavity/data.py",
    "experiments/lifted_cavity/devutil.py",
    "experiments/lifted_cavity/eval_real.py",
    "experiments/lifted_cavity/evaluate.py",
    "experiments/lifted_cavity/evaluate2.py",
    "experiments/lifted_cavity/export_panels.py",
    "experiments/lifted_cavity/extra_refs.py",
    "experiments/lifted_cavity/fa.py",
    "experiments/lifted_cavity/family.py",
    "experiments/lifted_cavity/finetune_airq.py",
    "experiments/lifted_cavity/finetune_real.py",
    "experiments/lifted_cavity/gauge.py",
    "experiments/lifted_cavity/gauge_source_gate.py",
    "experiments/lifted_cavity/lct.py",
    "experiments/lifted_cavity/lct_train.py",
    "experiments/lifted_cavity/models.py",
    "experiments/lifted_cavity/nlfa.py",
    "experiments/lifted_cavity/pfn.py",
    "experiments/lifted_cavity/pfn_train.py",
    "experiments/lifted_cavity/realdata.py",
    "experiments/lifted_cavity/resume.py",
    "experiments/lifted_cavity/score_validation.py",
    "experiments/lifted_cavity/tabpfn_colab.py",
    "experiments/lifted_cavity/train.py",
    "tests/test_lifted_cavity.py",
    "tests/test_lifted_gauge.py",
    "docs/LIFTED_CAVITY_PROTOCOL_V2.md",
    "docs/LIFTED_CAVITY_PROTOCOL_V3.md",
    "docs/LIFTED_CAVITY_PROTOCOL_V4.md",
    "docs/LIFTED_GAUGE_V2_PROTOCOL.md",
    "configs/lifted_gauge_v2_source.json",
)
LOCAL_FILES = {
    "docs/LIFTED_SUPPLEMENT.md": "README.md",
    "docs/LIFTED_V4_TIME_AMENDMENT.md": "docs/LIFTED_V4_TIME_AMENDMENT.md",
    "infra/build_lifted_supplement.py": "infra/build_lifted_supplement.py",
}
PANEL_FILES = {
    "README.md": "provenance/PANELS_README.md",
    "SHA256SUMS": "provenance/PANELS_SHA256SUMS",
}
REQUIREMENTS = b"""# Direct v4 runtime pins; not a complete transitive environment lock.
torch==2.8.0
numpy==2.3.5
scipy==1.17.0
pandas==2.3.3
tabpfn==9.1.0
scikit-learn==1.8.0
psutil==7.0.0
# Test tool only; not a scientific runtime pin.
pytest>=8,<9
"""
EXPECTED = set(SOURCE_FILES) | set(LOCAL_FILES.values()) | set(PANEL_FILES.values()) | {"requirements-v4.txt"}
MANIFEST = "MANIFEST.json"
MAX_FILE = 1_000_000
MAX_TOTAL = 5_000_000
STAMP = (1980, 1, 1, 0, 0, 0)
GUARDS = {
    "personal home path": re.compile(r"(?i)(?:[a-z]:[\\/]Users[\\/]|/(?:home|Users)/)[a-z0-9_.-]+"),
    "email address": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
    "private key": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    "credential token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{20,})\b"),
    "URL credentials": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
    "Git transport identity": re.compile(r"\bgit@[^\s:]+:"),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(name: str, data: bytes) -> None:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or str(p) != name or name not in EXPECTED:
        raise ValueError(f"Non-allowlisted path: {name}")
    if len(data) > MAX_FILE:
        raise ValueError(f"Oversized file: {name}")
    text = data.decode("utf-8")
    if "\x00" in text:
        raise ValueError(f"Binary content: {name}")
    for label, pattern in GUARDS.items():
        if pattern.search(text):
            # Never print the matched identity or secret.
            raise ValueError(f"Anonymity/secret audit failed ({label}): {name}")
    if name.endswith(".py"):
        ast.parse(text, filename=name)


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Required frozen Git objects are unavailable; no network fetch is attempted")
    return result.stdout


def frozen_files(repo: Path, revision: str, paths: dict[str, str]) -> dict[str, bytes]:
    entries = git(repo, "ls-tree", "-z", revision, "--", *paths)
    blobs = {}
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        name = raw_path.decode("utf-8")
        if name not in paths or mode not in ("100644", "100755") or kind != "blob":
            raise ValueError("Frozen input is not a regular allowlisted source file")
        if int(git(repo, "cat-file", "-s", oid)) > MAX_FILE:
            raise ValueError("Frozen input exceeds code-only size limit")
        blobs[paths[name]] = git(repo, "cat-file", "blob", oid)
    if set(blobs) != set(paths.values()):
        raise ValueError("A required frozen source file is missing")
    return blobs


def reject_links(path: Path) -> None:
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/junction/reparse-point paths are not accepted")


def payload(repo: Path) -> dict[str, bytes]:
    reject_links(repo)
    files = frozen_files(repo, SOURCE_REVISION, {p: p for p in SOURCE_FILES})
    files.update(frozen_files(repo, PANEL_REVISION, PANEL_FILES))
    for source, destination in LOCAL_FILES.items():
        path = repo / source
        reject_links(path)
        if not path.is_file() or path.stat().st_size > MAX_FILE:
            raise ValueError("Local packaging input is not a small regular file")
        files[destination] = path.read_bytes()
    files["requirements-v4.txt"] = REQUIREMENTS
    return files


def records(files: dict[str, bytes]) -> dict:
    if set(files) != EXPECTED or sum(map(len, files.values())) > MAX_TOTAL:
        raise ValueError("Archive membership/size differs from code-only allowlist")
    for name, data in files.items():
        audit(name, data)
    return {name: {"bytes": len(data), "sha256": digest(data)} for name, data in sorted(files.items())}


def write_archive(path: Path, files: dict[str, bytes]) -> None:
    manifest = {"schema": 1, "status": "incomplete-code-only-draft", "source_revision": SOURCE_REVISION,
                "panel_revision": PANEL_REVISION, "files": records(files)}
    contents = dict(files, **{MANIFEST: (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()})
    # Stored entries avoid compressor-version differences; all metadata is fixed.
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, STAMP)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, data)


def verify(path: Path) -> dict:
    reject_links(path.absolute())
    if path.stat().st_size > MAX_TOTAL + 100_000:
        raise ValueError("Archive exceeds code-only size limit")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or set(names) != EXPECTED | {MANIFEST}:
            raise ValueError("Duplicate, missing or non-allowlisted archive member")
        for entry in entries:
            if (entry.file_size > MAX_FILE or entry.compress_type != zipfile.ZIP_STORED or entry.flag_bits & 1
                    or entry.date_time != STAMP or entry.create_system != 3
                    or entry.external_attr != (stat.S_IFREG | 0o644) << 16 or entry.extra or entry.comment):
                raise ValueError("Archive has unsafe/noncanonical metadata")
        if archive.comment:
            raise ValueError("Archive comments are not allowed")
        files = {name: archive.read(name) for name in EXPECTED}
        manifest = json.loads(archive.read(MANIFEST))
    expected = {"schema": 1, "status": "incomplete-code-only-draft", "source_revision": SOURCE_REVISION,
                "panel_revision": PANEL_REVISION, "files": records(files)}
    if manifest != expected:
        raise ValueError("Manifest or content hash mismatch")
    return {"files": len(files), "bytes": sum(map(len, files.values())), "sha256": digest(path.read_bytes())}


def self_test() -> None:
    # Artificial text only: no scientific training, artifact loading or network access.
    fixture = {name: b"# fixture\n" for name in EXPECTED}
    with tempfile.TemporaryDirectory(prefix="lifted-supplement-") as directory:
        a, b = (Path(directory) / name for name in ("a.zip", "b.zip"))
        write_archive(a, fixture)
        write_archive(b, fixture)
        assert a.read_bytes() == b.read_bytes()
        verify(a)
        # Valid CRCs but stale content hashes must still be rejected.
        c = Path(directory) / "changed.zip"
        with zipfile.ZipFile(a) as source, zipfile.ZipFile(c, "x") as changed:
            for entry in source.infolist():
                changed.writestr(entry, b"changed\n" if entry.filename == "README.md" else source.read(entry))
        for bad_name, bad_data in (("../private.txt", b"x"), ("README.md", b"/home/" + b"researcher/file"),
                                   ("README.md", b"name" + b"@example.org"), ("README.md", b"\x00"),
                                   ("README.md", b"ghp_" + b"A" * 24), ("README.md", b"x" * (MAX_FILE + 1))):
            try:
                audit(bad_name, bad_data)
            except ValueError:
                pass
            else:
                raise AssertionError("Unsafe input accepted")
        try:
            verify(c)
        except (ValueError, zipfile.BadZipFile):
            pass
        else:
            raise AssertionError("Modified content accepted")
    print("PASS: deterministic bytes, integrity verification, traversal, identity and binary guards")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path, help="new draft ZIP; existing files are never overwritten")
    action.add_argument("--verify", type=Path, help="verify without extracting or executing archive contents")
    action.add_argument("--self-test", action="store_true", help="run local packaging safety checks only")
    parser.add_argument("--repo", type=Path, default=Path(__file__).absolute().parents[1])
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.verify:
        print(json.dumps(verify(args.verify), sort_keys=True))
    else:
        out = args.out.absolute()
        reject_links(out.parent)
        if out.exists():
            raise ValueError("Output already exists")
        write_archive(out, payload(args.repo.absolute()))
        print(json.dumps(verify(out), sort_keys=True))


if __name__ == "__main__":
    main()
