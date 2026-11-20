"""Offline evidence export and protected workbook creation for RQ III review."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from .profile_review import build_profile_review
from .comparisons import RUN_KEYS


WORKBOOK_NAME = "profile_review.xlsx"
OUTPUT_DIRECTORY = "reported_runs/analysis/outputs/profile_review_20261102"


def _records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def _signature(results, repo):
    """Invalidate saved evidence if a selected source/checkpoint/code changes."""
    identities = results["qualitative_review"][RUN_KEYS].drop_duplicates()
    selected = results["sources"].merge(identities, on=RUN_KEYS, how="inner", validate="one_to_one")
    if len(selected) != len(identities):
        raise ValueError("Each sampled profile-review condition needs its exact recorded source")
    stamps = []
    for row in selected.to_dict("records"):
        root = (repo / row["source_directory"]).resolve()
        paths = [root / name for name in ("_COMPLETE.json", "manifest.json", "events.jsonl")]
        paths += sorted((root / "checkpoints").glob("iteration_*/checkpoint.pt"))
        for path in paths:
            stat = path.stat()
            stamps.append([str(path), stat.st_size, stat.st_mtime_ns])
    code = [Path(__file__), Path(__file__).with_name("profile_review.py"), Path(__file__).with_name("comparisons.py")]
    value = {"snapshot": results.metadata["snapshot_id"], "stamps": stamps,
        "selection": _records(results["qualitative_review"]),
        "code": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare_profile_review(results, repo_root, directory=None):
    """Return evidence frames and save reproducible flat JSON/CSV inputs.

    The editable workbook is outside the snapshot and is never rewritten here.
    Cached evidence remains bound to the pinned sample, code, and source files.
    """
    repo = Path(repo_root).resolve()
    directory = Path(directory or repo / OUTPUT_DIRECTORY).resolve()
    if not directory.is_relative_to(repo):
        raise ValueError("Profile review output must remain inside the repository")
    directory.mkdir(parents=True, exist_ok=True)
    payload_path = directory / "evidence.json"
    signature = _signature(results, repo)
    payload = json.loads(payload_path.read_text(encoding="utf-8")) if payload_path.is_file() else None
    if payload is None or payload.get("input_signature") != signature:
        if (directory / WORKBOOK_NAME).is_file():
            raise ValueError("The existing workbook belongs to different saved evidence; preserve it and use its original snapshot or a new review directory")
        evidence = build_profile_review(results["profile_changes"], results["profiles"], results["sources"], repo,
                                        expected_sample=results["qualitative_review"], n=10)
        if len(evidence["cases"]) != 30 or evidence["cases"].groupby("seed").size().to_dict() != {42: 10, 43: 10, 44: 10}:
            raise ValueError("This review workbook requires the agreed ten cases for each of seeds42,43,44")
        payload = {"schema_version": "rq3-profile-review-v1", "snapshot_id": results.metadata["snapshot_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "input_signature": signature,
            **{name: _records(evidence[name]) for name in ("cases", "changes", "history", "selection")},
            "method": evidence["method"]}
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    frames = {name: pd.DataFrame(payload[name]) for name in ("cases", "changes", "history", "selection")}
    for name, frame in frames.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    return {**frames, "method": payload["method"], "payload_path": payload_path,
            "directory": directory, "workbook_path": directory / WORKBOOK_NAME}


def ensure_profile_workbook(evidence):
    """Open a preserved private review workbook without replacing human coding."""
    output = Path(evidence["workbook_path"])
    if not output.is_file():
        raise FileNotFoundError("The protected source-bearing review workbook is not part of the public replay. Restore it from the private analysis archive to inspect original evidence; notebook08 replays the public saved categories without it.")
    binding_path = output.parent / "workbook_binding.json"
    if not binding_path.is_file():
        raise ValueError("Existing review workbook has no saved evidence binding; it will not be overwritten")
    payload = json.loads(Path(evidence["payload_path"]).read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("input_signature") != payload["input_signature"]:
        raise ValueError("Existing review workbook and saved evidence have different signatures")
    return output

