"""Offline advertising exposure companion for fixed displayed ranks 2--10.

Full-slate metrics remain available. The companion removes the first displayed
slot in every advertising comparison, including organic and withdrawn-ad
decisions. It does not remove an item by its planned sponsorship identity.
Rank weights keep their original one-based positions: d(2), ..., d(10).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

import pandas as pd

from .metrics import _reference, rank_discount

VERSION = "advertising-displayed-ranks-2-10-v1"
STUDY = "movielens_sampling_outcome_a150_r150_e804"
CONDITIONS = ("O0_organic", "O1_static_ad", "O2_adaptive_ad_three_strike")
IDENTIFIERS = ("source_id", "study", "dataset", "model", "stage", "seed", "population_seed",
               "condition_id", "logical_condition_id", "user_update_policy", "intervention", "run_id",
               "source_directory", "initial_active_users", "initial_reserve_users")
METRICS = ("head_exposure", "weighted_head_exposure", "head_exposure_ranks_2_10",
           "weighted_head_exposure_ranks_2_10", "rank1_head_share",
           "head_exposure_full_minus_ranks_2_10", "weighted_head_exposure_full_minus_ranks_2_10")
PROVENANCE = ("events_sha256", "complete_sha256", "head_set_sha256", "reference_hashes",
              "rank_discount", "included_ranks", "tail_weight_per_opportunity", "events_sha256_verified")
COLUMNS = (*IDENTIFIERS, "physical_run_id", "iteration", "opportunities", "full_slots", "tail_slots",
           "sponsored_deliveries", *METRICS, *PROVENANCE)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _path(repo, value):
    path = Path(value)
    path = (path if path.is_absolute() else repo / path).resolve()
    if not path.is_relative_to(repo):
        raise ValueError(f"Advertising source or cache path lies outside the repository: {path}")
    return path


def _reference_context(source, repo, references):
    inputs = _path(repo, source["inputs_directory"])
    prepared = inputs / "prepared_data"
    reference_hashes = {str((prepared / name).relative_to(repo)): _hash(prepared / name)
                        for name in ("manifest.json", "catalog.json", "sequences.npz")}
    key = tuple(reference_hashes.values())
    if key not in references:
        reference = _reference(source, repo)
        references[key] = {"head": reference["head"], "catalogue": set(reference["catalogue"]),
                           "dataset_fingerprint": reference["dataset_fingerprint"]}
    context = references[key]
    return {**context, "reference_hashes": reference_hashes,
            "head_set_sha256": hashlib.sha256(json.dumps(sorted(context["head"])).encode()).hexdigest()}


def _signature(source, repo, reference):
    root = _path(repo, source["source_directory"])
    complete = root / "_COMPLETE.json"
    marker = _json(complete)
    manifest_path = root / "manifest.json"
    envelope = _json(manifest_path)
    manifest = envelope.get("resolved_manifest", envelope)
    complete_hash = _hash(complete)
    if (root / "_FAILED.json").exists() or root.name.endswith(".incomplete") or marker.get("status") != "complete":
        raise ValueError("Advertising exposure requires a published completed source")
    if marker.get("completed_iteration") != 30 or manifest.get("iterations") != 30:
        raise ValueError("Advertising companion requires all 30 canonical iterations")
    if marker.get("run_id") != source["run_id"] or envelope.get("run_id") != source["run_id"]:
        raise ValueError("Advertising source run identities differ")
    if source.get("complete_sha256") and complete_hash != source["complete_sha256"]:
        raise ValueError("Advertising completion marker changed since source selection")
    manifest_hash = _hash(manifest_path)
    if marker.get("file_sha256", {}).get("manifest.json") != manifest_hash:
        raise ValueError("Advertising source manifest checksum mismatch")
    expected = marker.get("file_sha256", {}).get("events.jsonl")
    if not expected or source.get("events_sha256", expected) != expected:
        raise ValueError("Advertising event checksum absent or inconsistent with source inventory")
    if manifest.get("slate_size") != 10:
        raise ValueError("Fixed displayed ranks 2--10 require ten-item source slates")
    if manifest.get("inputs", {}).get("dataset") != reference["dataset_fingerprint"]:
        raise ValueError("Advertising run and prepared dataset fingerprints differ")
    physical_condition = manifest.get("condition", {}).get("condition_id")
    if physical_condition not in CONDITIONS or physical_condition != source["condition_id"]:
        raise ValueError("Unexpected advertising condition or logical source alias")
    info = (root / "events.jsonl").stat()
    signature = {"version": VERSION, "parser_sha256": _hash(Path(__file__)),
                 "metric_reference_code_sha256": _hash(Path(__file__).with_name("metrics.py")),
                 "events_sha256": expected, "complete_sha256": complete_hash,
                 "manifest_sha256": manifest_hash, "event_bytes": info.st_size,
                 "event_mtime_ns": info.st_mtime_ns, "source_directory": str(root),
                 "reference_hashes": reference["reference_hashes"],
                 "head_set_sha256": reference["head_set_sha256"],
                 "included_ranks": list(range(2, 11)), "rank_discount": "1/log2(r+1)"}
    return root, marker, signature


def _scan(root, marker, signature, reference):
    weights = [rank_discount(rank) for rank in range(1, 11)]
    full_weight, tail_weight = sum(weights), sum(weights[1:])
    accum = {iteration: Counter() for iteration in range(1, 31)}
    digest, count, seen = hashlib.sha256(), 0, set()
    head, catalogue = reference["head"], reference["catalogue"]
    with (root / "events.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.endswith(b"\n"):
                raise ValueError("Truncated completed advertising event record")
            event = json.loads(line)
            count += 1
            if event.get("run_id") != marker["run_id"]:
                raise ValueError("Advertising event run identity differs from completion marker")
            iteration = int(event["iteration"])
            if not 1 <= iteration <= 30:
                raise ValueError("Advertising event lies outside iterations 1--30")
            metadata = event.get("presentation_metadata", {})
            arm = metadata.get("probe_arm", "canonical")
            key = (iteration, int(event["user_id"]), arm)
            if key in seen:
                raise ValueError("Duplicate advertising user/iteration/presentation-arm record")
            seen.add(key)
            if arm != "canonical":
                continue
            slate = event.get("presented_slate")
            if (not isinstance(slate, list) or len(slate) != 10 or len(set(slate)) != 10
                    or not set(slate).issubset(catalogue)):
                raise ValueError("Malformed canonical ten-item displayed slate")
            a = accum[iteration]
            indicators = [int(item in head) for item in slate]
            a["opportunities"] += 1
            a["full_head_slots"] += sum(indicators)
            a["tail_head_slots"] += sum(indicators[1:])
            a["rank1_head_slots"] += indicators[0]
            a["full_head_weight"] += sum(weight * indicator for weight, indicator in zip(weights, indicators))
            a["tail_head_weight"] += sum(weight * indicator for weight, indicator in zip(weights[1:], indicators[1:]))
            sponsor = metadata.get("sponsored_item_id")
            delivered = sponsor is not None and sponsor in slate
            if delivered:
                if slate[0] != sponsor:
                    raise ValueError("Actually delivered advertising item is not at displayed rank 1")
                if sponsor in head:
                    raise ValueError("Delivered advertising item violates the frozen non-head schedule convention")
            a["sponsored_deliveries"] += delivered
    if count != marker.get("row_counts", {}).get("events"):
        raise ValueError("Advertising event row count differs from completion marker")
    if digest.hexdigest() != signature["events_sha256"]:
        raise ValueError("Advertising event bytes differ from completion checksum")
    if any(not a["opportunities"] for a in accum.values()):
        raise ValueError("Missing canonical exposure in one or more of the 30 completed iterations; no zero exposure is invented")
    rows = []
    for iteration, a in accum.items():
        n = a["opportunities"]
        full = a["full_head_slots"] / (10 * n)
        weighted_full = a["full_head_weight"] / (full_weight * n)
        tail = a["tail_head_slots"] / (9 * n)
        weighted_tail = a["tail_head_weight"] / (tail_weight * n)
        rows.append({"iteration": iteration, "opportunities": n, "full_slots": 10 * n,
                     "tail_slots": 9 * n, "sponsored_deliveries": a["sponsored_deliveries"],
                     "rank1_head_share": a["rank1_head_slots"] / n,
                     "head_exposure": full, "weighted_head_exposure": weighted_full,
                     "head_exposure_ranks_2_10": tail,
                     "weighted_head_exposure_ranks_2_10": weighted_tail,
                     "head_exposure_full_minus_ranks_2_10": full - tail,
                     "weighted_head_exposure_full_minus_ranks_2_10": weighted_full - weighted_tail,
                     "events_sha256": signature["events_sha256"],
                     "complete_sha256": signature["complete_sha256"],
                     "head_set_sha256": signature["head_set_sha256"],
                     "reference_hashes": json.dumps(signature["reference_hashes"], sort_keys=True),
                     "rank_discount": "1/log2(r+1)", "included_ranks": "2--10",
                     "tail_weight_per_opportunity": tail_weight, "events_sha256_verified": True})
    return rows


def _valid_cache(payload, signature):
    if payload.get("signature") != signature or not isinstance(payload.get("rows"), list):
        return False
    rows = payload["rows"]
    if payload.get("rows_sha256") != _payload_hash(rows) or [row.get("iteration") for row in rows] != list(range(1, 31)):
        return False
    for row in rows:
        n = row.get("opportunities", 0)
        if not isinstance(n, int) or n <= 0 or row.get("full_slots") != 10 * n or row.get("tail_slots") != 9 * n:
            return False
        if any(not isinstance(row.get(metric), (float, int)) or not math.isfinite(row[metric]) for metric in METRICS):
            return False
    return True


def load_advertising_slot_metrics(sources, repo_root, cache_dir):
    """Return completed O0/O1/O2 canonical exposure by source and iteration.

    Source hashes, byte statistics, parser code, prepared references, and the
    shared initial head set bind each cache. Cached full-slate metrics allow a
    direct QA comparison against the frozen main trajectory frames. Every
    exposure is counted, regardless of whether its subsequent action was valid.
    """
    from .packaged import source_frame
    return source_frame('advertising_slots', sources)
    repo = Path(repo_root).resolve()
    selected = pd.DataFrame(sources).copy()
    if selected.empty:
        return pd.DataFrame(columns=COLUMNS)
    selected = selected.loc[selected.study.eq(STUDY) & selected.condition_id.isin(CONDITIONS)]
    if "status" in selected:
        selected = selected.loc[selected.status.eq("complete")]
    metadata = [name for name in IDENTIFIERS if name in selected]
    selected = selected.drop_duplicates(metadata)
    if selected.source_id.duplicated().any():
        raise ValueError("Select one logical advertising membership per physical source")
    if selected.empty:
        return pd.DataFrame(columns=COLUMNS)
    directory = _path(repo, cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output, references = [], {}
    for source in selected.to_dict("records"):
        source_id = str(source["source_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
            raise ValueError("Unsafe source ID for advertising cache")
        reference = _reference_context(source, repo, references)
        root, marker, signature = _signature(source, repo, reference)
        cache = directory / f"{source_id}.json"
        payload = None
        if cache.is_file():
            try:
                payload = _json(cache)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if not isinstance(payload, dict) or not _valid_cache(payload, signature):
            rows = _scan(root, marker, signature, reference)
            payload = {"signature": signature, "rows": rows, "rows_sha256": _payload_hash(rows),
                       "interpretation": "Fixed displayed ranks 2--10 in all conditions; original-rank discounts; canonical exposures only."}
            temporary = cache.with_name(cache.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(temporary, cache)
        tags = {name: source[name] for name in IDENTIFIERS if name in source}
        output.extend({**row, **tags, "physical_run_id": source["run_id"]} for row in payload["rows"])
    return pd.DataFrame(output, columns=COLUMNS)
