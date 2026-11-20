"""Saved post-processing responses before and after the immediate reordering.

Canonical and original-order probes come from the same saved user/iteration.
Each arm retains its own valid-click denominator. Original-order probes are
measurements in the post-processing trajectory, not independent control runs.
This loader never constructs a simulator, trains a model, or calls a provider.
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

from .advertising_slots import _hash, _json, _path, _payload_hash, _reference_context
from .metrics import rank_discount
from .readers import action, opportunities

VERSION = "postprocessing-canonical-original-order-components-v1"
STUDY = "movielens_mitigation_a300_e804"
ARMS = ("canonical", "replica", "shuffled", "original_order_probe")
IDENTIFIERS = ("source_id", "study", "dataset", "model", "stage", "seed", "population_seed",
               "condition_id", "logical_condition_id", "user_update_policy", "intervention", "run_id",
               "source_directory", "initial_active_users", "initial_reserve_users")
METRICS = ("head_click_share", "original_order_head_click_share", "head_exposure", "weighted_head_exposure",
           "recommender_head_exposure", "recommender_weighted_head_exposure")
COUNTS = ("opportunities", "canonical_opportunities", "original_order_opportunities", "both_arms_valid_clicks",
          "canonical_valid_actions", "canonical_valid_clicks", "canonical_head_clicks", "canonical_valid_no_clicks",
          "canonical_invalid_responses", "original_order_valid_actions", "original_order_valid_clicks",
          "original_order_head_clicks", "original_order_valid_no_clicks", "original_order_invalid_responses")
PROOF = ("slate_membership_preserved", "unweighted_exposure_identity", "rank_weight_total_per_opportunity",
         "rank_discount", "presentation_arm_rows", "events_sha256_verified", "events_sha256", "complete_sha256",
         "head_set_sha256", "reference_hashes")
COLUMNS = (*IDENTIFIERS, "physical_run_id", "iteration", *COUNTS, *METRICS, *PROOF)


def _signature(source, repo, reference):
    root = _path(repo, source["source_directory"])
    complete = root / "_COMPLETE.json"
    marker = _json(complete)
    complete_hash = _hash(complete)
    envelope = _json(root / "manifest.json")
    manifest = envelope.get("resolved_manifest", envelope)
    if root.name.endswith(".incomplete") or (root / "_FAILED.json").exists() or marker.get("status") != "complete":
        raise ValueError("Post-processing decomposition requires a published completed source")
    if marker.get("completed_iteration") != 30 or manifest.get("iterations") != 30:
        raise ValueError("Post-processing decomposition requires all 30 iterations")
    if envelope.get("run_id") != source["run_id"] or marker.get("run_id") != source["run_id"]:
        raise ValueError("Post-processing source identities differ")
    if source.get("complete_sha256") and complete_hash != source["complete_sha256"]:
        raise ValueError("Post-processing completion marker changed since source selection")
    condition = manifest.get("condition", {})
    if condition.get("intervention") != "post_processing" or condition.get("user_model") != source["model"]:
        raise ValueError("Manifest is not the selected post-processing model")
    if manifest.get("slate_size") != 10:
        raise ValueError("Selected post-processing sources require ten-item slates")
    if manifest.get("inputs", {}).get("dataset") != reference["dataset_fingerprint"]:
        raise ValueError("Run and prepared dataset reference fingerprints differ")
    frozen_head = manifest.get("mitigation", {}).get("head_item_ids")
    if frozen_head is not None and set(frozen_head) != reference["head"]:
        raise ValueError("Post-processing head set differs from the shared initial reference")
    arms = tuple(manifest.get("presentation_arms", ARMS))
    if len(arms) != 4 or set(arms) != set(ARMS):
        raise ValueError("Post-processing source must retain all four presentation arms")
    expected = marker.get("file_sha256", {}).get("events.jsonl")
    if not expected or source.get("events_sha256", expected) != expected:
        raise ValueError("Post-processing event checksum absent or differs from source selection")
    manifest_hash = _hash(root / "manifest.json")
    if marker.get("file_sha256", {}).get("manifest.json") != manifest_hash:
        raise ValueError("Post-processing manifest checksum mismatch")
    info = (root / "events.jsonl").stat()
    here = Path(__file__).parent
    signature = {"version": VERSION, "parser_sha256": _hash(Path(__file__)),
                 "shared_code_sha256": {name: _hash(here / name)
                                        for name in ("metrics.py", "readers.py", "advertising_slots.py")},
                 "complete_sha256": complete_hash, "manifest_sha256": manifest_hash,
                 "events_sha256": expected, "event_bytes": info.st_size, "event_mtime_ns": info.st_mtime_ns,
                 "source_directory": str(root), "reference_hashes": reference["reference_hashes"],
                 "head_set_sha256": reference["head_set_sha256"], "rank_discount": "1/log2(r+1)"}
    return root, marker, manifest, signature


def _scan(root, marker, manifest, signature, reference):
    head, catalogue = reference["head"], reference["catalogue"]
    weights = [rank_discount(rank) for rank in range(1, 11)]
    total_weight = sum(weights)
    accum = {iteration: Counter() for iteration in range(1, 31)}
    seen_users = {iteration: set() for iteration in range(1, 31)}
    arm_rows = Counter()
    for records in opportunities(root / "events.jsonl", expected_arms=ARMS, run_id=manifest["run_id"] if "run_id" in manifest else marker["run_id"],
                                 horizon=30, slate_size=10, catalogue=catalogue,
                                 expected_hash=signature["events_sha256"], expected_rows=marker.get("row_counts", {}).get("events")):
        canonical, original = records["canonical"], records["original_order_probe"]
        iteration, uid = int(canonical["iteration"]), int(canonical["user_id"])
        if uid in seen_users[iteration]:
            raise ValueError("Duplicate canonical post-processing user/iteration")
        seen_users[iteration].add(uid)
        if original["presented_slate"] != canonical["recommender_slate"]:
            raise ValueError("Original-order probe does not exactly reproduce the canonical recommender slate")
        if set(canonical["presented_slate"]) != set(canonical["recommender_slate"]):
            raise ValueError("Post-processing changed slate membership")
        # Saved fingerprints add a direct check on shared response-state inputs
        # where the model records them. Prompt hashes may change with ordering.
        for field in ("profile_fingerprint", "checkpoint_fingerprint", "embedding_fingerprint"):
            left = canonical["action"].get("diagnostics", {}).get(field)
            right = original["action"].get("diagnostics", {}).get(field)
            if left is not None and right is not None and left != right:
                raise ValueError(f"Canonical/original-order response-state fingerprint differs: {field}")
        arm_rows.update(records.keys())
        a = accum[iteration]
        a["opportunities"] += 1
        cv, cc, ci = action(canonical)
        ov, oc, oi = action(original)
        a["both_arms_valid_clicks"] += cv and cc and ov and oc
        for prefix, valid, click, item in (("canonical", cv, cc, ci), ("original_order", ov, oc, oi)):
            a[f"{prefix}_valid_actions"] += valid
            a[f"{prefix}_valid_clicks"] += click
            a[f"{prefix}_head_clicks"] += click and item in head
            a[f"{prefix}_valid_no_clicks"] += valid and not click
            a[f"{prefix}_invalid_responses"] += not valid
        canonical_flags = [int(item in head) for item in canonical["presented_slate"]]
        original_flags = [int(item in head) for item in original["presented_slate"]]
        if sum(canonical_flags) != sum(original_flags):
            raise ValueError("Unweighted head exposure changed despite a pure reordering")
        a["head_slots"] += sum(canonical_flags)
        a["recommender_head_slots"] += sum(original_flags)
        a["head_weight"] += sum(weight * flag for weight, flag in zip(weights, canonical_flags))
        a["recommender_head_weight"] += sum(weight * flag for weight, flag in zip(weights, original_flags))
    n_active = int(manifest["active_users"])
    if any(len(users) != n_active for users in seen_users.values()):
        raise ValueError("Missing canonical users in one or more of the 30 completed post-processing iterations")
    if dict(arm_rows) != {arm: 30 * n_active for arm in ARMS}:
        raise ValueError("Post-processing arm counts differ from the configured completed fixed-population horizon")
    rows = []
    for iteration, a in accum.items():
        n = a["opportunities"]
        cv, ov = a["canonical_valid_clicks"], a["original_order_valid_clicks"]
        exposure = a["head_slots"] / (10 * n)
        original_exposure = a["recommender_head_slots"] / (10 * n)
        if exposure != original_exposure:
            raise ValueError("Unweighted post-processing exposure identity failed")
        rows.append({"iteration": iteration, **{name: int(a[name]) for name in COUNTS},
                     "canonical_opportunities": n, "original_order_opportunities": n,
                     "head_click_share": a["canonical_head_clicks"] / cv if cv else None,
                     "original_order_head_click_share": a["original_order_head_clicks"] / ov if ov else None,
                     "head_exposure": exposure, "weighted_head_exposure": a["head_weight"] / (n * total_weight),
                     "recommender_head_exposure": original_exposure,
                     "recommender_weighted_head_exposure": a["recommender_head_weight"] / (n * total_weight),
                     "slate_membership_preserved": True, "unweighted_exposure_identity": True,
                     "rank_weight_total_per_opportunity": total_weight, "rank_discount": "1/log2(r+1)",
                     "presentation_arm_rows": json.dumps(dict(arm_rows), sort_keys=True),
                     "events_sha256_verified": True, "events_sha256": signature["events_sha256"],
                     "complete_sha256": signature["complete_sha256"], "head_set_sha256": signature["head_set_sha256"],
                     "reference_hashes": json.dumps(signature["reference_hashes"], sort_keys=True)})
    return rows


def _valid_cache(payload, signature):
    if payload.get("signature") != signature or not isinstance(payload.get("rows"), list):
        return False
    rows = payload["rows"]
    if payload.get("rows_sha256") != _payload_hash(rows) or [row.get("iteration") for row in rows] != list(range(1, 31)):
        return False
    for row in rows:
        if any(not isinstance(row.get(name), int) or row[name] < 0 for name in COUNTS):
            return False
        if not row["opportunities"] or row["opportunities"] != row["canonical_opportunities"] or row["opportunities"] != row["original_order_opportunities"]:
            return False
        for metric in METRICS:
            if row.get(metric) is None and metric in ("head_click_share", "original_order_head_click_share"):
                continue
            if not isinstance(row.get(metric), (float, int)) or not math.isfinite(row[metric]):
                return False
    return True


def load_postprocessing_components(sources, repo_root, cache_dir):
    """Read completed MovieLens post-processing sources once; return 30 rows each.

    Counts and shares retain canonical and original-order denominators separately.
    Undefined click shares remain NaN. Cache keys bind parser/shared definitions,
    prepared references and head set, completion/event hashes, and source stats.
    """
    from .packaged import source_frame
    return source_frame('postprocessing_components', sources)
    repo = Path(repo_root).resolve()
    selected = pd.DataFrame(sources).copy()
    if selected.empty:
        return pd.DataFrame(columns=COLUMNS)
    selected = selected.loc[selected.study.eq(STUDY) & selected.dataset.eq("movielens")
                            & selected.intervention.eq("post_processing")]
    if "status" in selected:
        selected = selected.loc[selected.status.eq("complete")]
    metadata = [name for name in IDENTIFIERS if name in selected]
    selected = selected.drop_duplicates(metadata)
    if selected.source_id.duplicated().any():
        raise ValueError("Select one logical mitigation membership per physical source")
    if selected.empty:
        return pd.DataFrame(columns=COLUMNS)
    directory = _path(repo, cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output, references = [], {}
    for source in selected.to_dict("records"):
        source_id = str(source["source_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
            raise ValueError("Unsafe source ID for post-processing component cache")
        reference = _reference_context(source, repo, references)
        root, marker, manifest, signature = _signature(source, repo, reference)
        cache = directory / f"{source_id}.json"
        payload = None
        if cache.is_file():
            try:
                payload = _json(cache)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if not isinstance(payload, dict) or not _valid_cache(payload, signature):
            rows = _scan(root, marker, manifest, signature, reference)
            payload = {"signature": signature, "rows": rows, "rows_sha256": _payload_hash(rows),
                       "interpretation": "Canonical/original-order post-processing responses; each arm retains its own valid-click denominator."}
            temporary = cache.with_name(cache.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(temporary, cache)
        tags = {name: source[name] for name in IDENTIFIERS if name in source}
        output.extend({**row, **tags, "physical_run_id": source["run_id"]} for row in payload["rows"])
    result = pd.DataFrame(output, columns=COLUMNS)
    for metric in METRICS:
        result[metric] = pd.to_numeric(result[metric])
    return result
