"""Small, provider-free readers for saved simulation artifacts.

These readers never construct a simulator or load a model for inference. JSONL
is consumed one opportunity at a time, including every diagnostic presentation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value, repo_root):
    candidate = Path(str(value).replace("\\", "/"))
    return candidate if candidate.is_absolute() else Path(repo_root) / candidate


def jsonl_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}") from exc


def action(event):
    response = event["action"]
    status, kind, item = response["status"], response["action"], response.get("clicked_item_id")
    if status != "valid":
        if item is not None or kind != "unavailable":
            raise ValueError("A non-valid response cannot be counted as a click or no-click")
        return False, False, None
    if kind == "click" and item in event["presented_slate"]:
        return True, True, int(item)
    if kind == "no_click" and item is None:
        return True, False, None
    raise ValueError("A valid response must select a displayed item or record no-click")


def opportunities(path, *, expected_arms, run_id, horizon, slate_size, catalogue,
                  expected_hash=None, expected_rows=None):
    """Yield contiguous canonical/probe sets and verify the recorded event bytes."""
    digest, row_count, previous, records = hashlib.sha256(), 0, None, {}
    seen = set()

    def checked():
        if set(records) != set(expected_arms):
            raise ValueError(f"Missing or unexpected presentation arms for {previous}: {set(records)}")
        canonical = records["canonical"]
        for arm, event in records.items():
            if event["recommender_slate"] != canonical["recommender_slate"]:
                raise ValueError("Paired probes must share the original recommender slate")
            if set(event["presented_slate"]) != set(canonical["presented_slate"]):
                raise ValueError("Paired probes must show the same items")
            if arm == "replica" and event["presented_slate"] != canonical["presented_slate"]:
                raise ValueError("Replica must retain the canonical order")
            if arm != "canonical" and event.get("user_update") is not None:
                raise ValueError("A diagnostic probe cannot carry a persistent update")
        return records

    with Path(path).open("rb") as handle:
        for line in handle:
            digest.update(line)
            if not line.endswith(b"\n"):
                raise ValueError(f"Truncated event record in {path}")
            row_count += 1
            event = json.loads(line)
            if event.get("run_id") != run_id:
                raise ValueError("Event run identity differs from manifest")
            key = int(event["iteration"]), int(event["user_id"])
            if key[0] < 1 or key[0] > horizon:
                raise ValueError("Event iteration outside configured horizon")
            arm = event.get("presentation_metadata", {}).get("probe_arm", "canonical")
            for name in ("presented_slate", "recommender_slate"):
                slate = event[name]
                if len(slate) != slate_size or len(set(slate)) != len(slate) or not set(slate) <= catalogue:
                    raise ValueError("Invalid slate width, duplicate, or unknown catalogue item")
            action(event)
            if key != previous:
                if records:
                    yield checked()
                if key in seen:
                    raise ValueError("Non-contiguous or duplicate opportunity")
                seen.add(key)
                previous, records = key, {}
            if arm in records:
                raise ValueError("Duplicate presentation arm")
            records[arm] = event
    if records:
        yield checked()
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError("Event row count differs from completion marker")
    if expected_hash is not None and digest.hexdigest() != expected_hash:
        raise ValueError("Event bytes differ from completion checksum")


def recorded_probabilities(event):
    """Return saved forced-choice probabilities keyed by item, never raw logits."""
    diagnostics = event["action"].get("diagnostics", {})
    values = diagnostics.get("conditioned_choice_probabilities", diagnostics.get("probabilities"))
    if values is None:
        return None
    slate = event["presented_slate"]
    if len(values) != len(slate):
        raise ValueError("Saved choice probabilities do not match the displayed list")
    import math
    values = [float(value) for value in values]
    if not all(math.isfinite(value) and value >= 0 for value in values) or not math.isclose(sum(values), 1, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Saved item-choice probabilities are not a probability distribution")
    return dict(zip(map(int, slate), values))
