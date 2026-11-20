"""Offline counts of recorded agent modules and reasoning-limit fallbacks.

``module_trace[].module`` identifies a completed module. A planning step's
``module_invoked`` field requests its next module; it is not that step's own
identity. These counts describe saved completed module traces, not provider
requests, retries, or failed calls omitted from a terminal failure record.
Only caller-selected, completed agent sources are read; presentation arms stay
separate. No simulator, model, or provider is imported.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import pandas as pd

VERSION = "logged-agent-usage-v1"
IDENTIFIERS = ("source_id", "study", "dataset", "model", "stage", "seed", "population_seed",
               "condition_id", "logical_condition_id", "user_update_policy", "intervention", "run_id",
               "source_directory")
COUNT_COLUMNS = (
    "decisions", "valid_decisions", "nonvalid_decisions", "memory_calls_observed",
    "decisions_with_memory_observed", "memory_route_requests_observed",
    "trace_present", "trace_missing", "trace_malformed", "forced_actions_observed",
    "forced_flag_present", "forced_flag_known", "forced_flag_missing", "forced_flag_malformed",
)
NULLABLE_COUNTS = ("memory_calls", "decisions_with_memory", "forced_actions")
ALL_COLUMNS = (*IDENTIFIERS, "arm", "first_iteration", "last_iteration", *COUNT_COLUMNS,
               *NULLABLE_COUNTS, "events_sha256_verified", "events_sha256", "complete_sha256")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _path(repo, value):
    path = Path(value)
    path = (path if path.is_absolute() else repo / path).resolve()
    if not path.is_relative_to(repo):
        raise ValueError(f"Agent source must be inside the repository: {path}")
    return path


def _signature(source, repo):
    root = _path(repo, source["source_directory"])
    marker_path = root / "_COMPLETE.json"
    marker = _json(marker_path)
    complete_hash = _hash(marker_path)
    if (root / "_FAILED.json").exists() or root.name.endswith(".incomplete") or marker.get("status") != "complete":
        raise ValueError(f"Agent usage requires a published completed source: {root}")
    if marker.get("run_id") != source["run_id"]:
        raise ValueError("Completion marker and source run identity differ")
    if source.get("complete_sha256") and complete_hash != source["complete_sha256"]:
        raise ValueError("Completion marker changed since the selected source inventory")
    expected = marker.get("file_sha256", {}).get("events.jsonl")
    if not expected or source.get("events_sha256", expected) != expected:
        raise ValueError("Event checksum is absent or differs from source inventory")
    if source.get("iterations") is not None and marker.get("completed_iteration") != source["iterations"]:
        raise ValueError("Source horizon differs from its completion marker")
    event_path = root / "events.jsonl"
    info = event_path.stat()
    signature = {"version": VERSION, "parser_sha256": _hash(Path(__file__)),
                 "events_sha256": expected, "complete_sha256": complete_hash,
                 "event_bytes": info.st_size, "event_mtime_ns": info.st_mtime_ns,
                 "source_directory": str(root)}
    return root, marker, signature


def _scan(root, marker, signature):
    counters, first, last = {}, {}, {}
    digest, row_count, seen = hashlib.sha256(), 0, set()
    horizon = int(marker["completed_iteration"])
    with (root / "events.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.endswith(b"\n"):
                raise ValueError("Truncated event record in completed agent source")
            event = json.loads(line)
            row_count += 1
            if event.get("run_id") != marker["run_id"]:
                raise ValueError("Event run identity differs from selected source")
            iteration = int(event["iteration"])
            if not 1 <= iteration <= horizon:
                raise ValueError("Agent event lies outside the completed horizon")
            arm = event.get("presentation_metadata", {}).get("probe_arm", "canonical")
            key = (iteration, int(event["user_id"]), arm)
            if key in seen:
                raise ValueError("Duplicate user/iteration/presentation-arm record")
            seen.add(key)
            count = counters.setdefault(arm, Counter())
            first[arm] = min(first.get(arm, iteration), iteration)
            last[arm] = max(last.get(arm, iteration), iteration)
            count["decisions"] += 1
            action = event.get("action", {})
            valid = action.get("status") == "valid"
            count["valid_decisions" if valid else "nonvalid_decisions"] += 1
            diagnostics = action.get("diagnostics", {})
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            if "forced_action" not in diagnostics:
                count["forced_flag_missing"] += 1
            else:
                count["forced_flag_present"] += 1
                flag = diagnostics["forced_action"]
                if isinstance(flag, bool):
                    count["forced_flag_known"] += 1
                    count["forced_actions_observed"] += flag
                else:
                    count["forced_flag_malformed"] += 1
            if "module_trace" not in diagnostics:
                count["trace_missing"] += 1
                continue
            count["trace_present"] += 1
            trace = diagnostics["module_trace"]
            if not isinstance(trace, list) or any(not isinstance(step, dict) or not isinstance(step.get("module"), str) for step in trace):
                count["trace_malformed"] += 1
                continue
            memory = sum(step["module"] == "memory" for step in trace)
            count["memory_calls_observed"] += memory
            count["decisions_with_memory_observed"] += memory > 0
            count["memory_route_requests_observed"] += sum(step.get("module_invoked") == "memory" for step in trace)
    if row_count != marker.get("row_counts", {}).get("events"):
        raise ValueError("Agent event row count differs from completion marker")
    if digest.hexdigest() != signature["events_sha256"]:
        raise ValueError("Agent event bytes differ from recorded completion checksum")
    rows = []
    for arm, counts in sorted(counters.items()):
        row = {"arm": arm, "first_iteration": first[arm], "last_iteration": last[arm],
               **{name: int(counts[name]) for name in COUNT_COLUMNS},
               "events_sha256_verified": True, "events_sha256": signature["events_sha256"],
               "complete_sha256": signature["complete_sha256"]}
        trace_known = not (row["trace_missing"] or row["trace_malformed"])
        flags_known = not (row["forced_flag_missing"] or row["forced_flag_malformed"])
        row["memory_calls"] = row["memory_calls_observed"] if trace_known else None
        row["decisions_with_memory"] = row["decisions_with_memory_observed"] if trace_known else None
        row["forced_actions"] = row["forced_actions_observed"] if flags_known else None
        rows.append(row)
    return rows


def _valid_cache(payload, signature):
    if payload.get("signature") != signature or not isinstance(payload.get("rows"), list):
        return False
    rows = payload["rows"]
    if payload.get("rows_sha256") != _payload_hash(rows):
        return False
    for row in rows:
        if any(not isinstance(row.get(name), int) or isinstance(row.get(name), bool) or row[name] < 0 for name in COUNT_COLUMNS):
            return False
        if row["forced_flag_present"] + row["forced_flag_missing"] != row["decisions"]:
            return False
        if row["forced_flag_known"] + row["forced_flag_malformed"] != row["forced_flag_present"]:
            return False
    return True


def load_agent_usage(sources, repo_root, cache_dir):
    """Return per-source/per-arm recorded usage, verifying fresh event streams once.

    Pass an explicit study/configuration selection. Repeated copies of an exact
    membership are collapsed; competing logical memberships for the same
    physical source must first be narrowed by the caller. Missing or malformed
    trace/forced-action fields make the corresponding aggregate unknown, while
    observed counts and field coverage remain available separately.
    """
    from .packaged import source_frame
    return source_frame('agent_usage', sources)
    repo = Path(repo_root).resolve()
    selected = pd.DataFrame(sources).copy()
    if selected.empty:
        return pd.DataFrame(columns=ALL_COLUMNS)
    selected = selected.loc[selected.model.eq("agentic_recommender")]
    if "status" in selected:
        selected = selected.loc[selected.status.eq("complete")]
    membership = [name for name in IDENTIFIERS if name in selected]
    selected = selected.drop_duplicates(membership)
    if selected.source_id.duplicated().any():
        raise ValueError("Select one logical study membership per physical agent source before pooling")
    directory = _path(repo, cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = []
    for source in selected.to_dict("records"):
        source_id = str(source["source_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
            raise ValueError("Unsafe source ID for agent-usage cache")
        root, marker, signature = _signature(source, repo)
        cached = directory / f"{source_id}.json"
        payload = None
        if cached.is_file():
            try:
                payload = _json(cached)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if not isinstance(payload, dict) or not _valid_cache(payload, signature):
            rows = _scan(root, marker, signature)
            payload = {"signature": signature, "rows": rows, "rows_sha256": _payload_hash(rows),
                       "interpretation": "Saved completed module traces, not provider attempts or unrecorded terminal-failure steps."}
            temporary = cached.with_name(cached.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, cached)
        tags = {name: source[name] for name in IDENTIFIERS if name in source}
        output.extend({**row, **tags} for row in payload["rows"])
    frame = pd.DataFrame(output, columns=ALL_COLUMNS)
    for name in NULLABLE_COUNTS:
        frame[name] = frame[name].astype("Int64")
    return frame
