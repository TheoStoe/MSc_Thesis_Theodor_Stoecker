"""List, verify, and restore the optional local full-artifact archives.

The numerical notebooks run without these archives. The archives contain
dataset-derived records and are deliberately excluded from Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def selected(manifest, arguments):
    for entry in manifest["archives"]:
        metadata = entry.get("metadata", {})
        if arguments.category and entry["category"] != arguments.category:
            continue
        if arguments.dataset and metadata.get("dataset") != arguments.dataset:
            continue
        if arguments.seed is not None and metadata.get("seed") != arguments.seed:
            continue
        if arguments.id and entry["id"] not in arguments.id:
            continue
        yield entry


def verify_archive(entry):
    archive = ROOT / entry["archive"]
    if not archive.is_file():
        raise FileNotFoundError(f"Local archive is not present: {entry['archive']}")
    if archive.stat().st_size != entry["bytes"] or sha256(archive) != entry["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {entry['archive']}")
    return archive


def restore(entry):
    archive = verify_archive(entry)
    expected = {item["path"]: item for item in entry["files"]}
    restored = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                raise ValueError("Unsafe archive member path")
            if not member.isfile() or member.name not in expected:
                raise ValueError(f"Unexpected archive member: {member.name}")
            destination = ROOT.joinpath(*name.parts)
            if not destination.resolve().is_relative_to(ROOT):
                raise ValueError("Archive member escaped the repository")
            record = expected.pop(member.name)
            if destination.exists():
                if not destination.is_file() or sha256(destination) != record["sha256"]:
                    raise FileExistsError(f"Refusing to replace a different file: {member.name}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            if temporary.exists():
                raise FileExistsError(f"Unfinished restore already exists: {temporary.relative_to(ROOT)}")
            with tar.extractfile(member) as source, temporary.open("xb") as target:
                shutil.copyfileobj(source, target, length=2**20)
            if temporary.stat().st_size != record["bytes"] or sha256(temporary) != record["sha256"]:
                temporary.unlink()
                raise ValueError(f"Restored file failed its checksum: {member.name}")
            temporary.replace(destination)
            restored += 1
    if expected:
        raise ValueError(f"Archive is missing {len(expected)} recorded files")
    return restored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "verify", "restore"))
    parser.add_argument("--category", choices=("runs", "inputs", "analysis"))
    parser.add_argument("--dataset", choices=("movielens", "goodreads"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--id", action="append", help="Archive ID shown by list; repeat to select several")
    arguments = parser.parse_args(argv)
    manifest = json.loads((ROOT / "artifacts/manifest.json").read_text(encoding="utf8"))
    entries = list(selected(manifest, arguments))
    if not entries:
        parser.error("No archives match those filters")
    total = 0
    for entry in entries:
        if arguments.action == "list":
            print(f"{entry['id']}  {entry['category']:8}  {entry['bytes']/2**20:8.1f} MiB  {entry['source_path']}")
        elif arguments.action == "verify":
            verify_archive(entry)
        else:
            total += restore(entry)
    if arguments.action == "verify":
        print(f"Verified {len(entries)} archives.")
    elif arguments.action == "restore":
        print(f"Restored {total} files from {len(entries)} archives; identical existing files were kept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
