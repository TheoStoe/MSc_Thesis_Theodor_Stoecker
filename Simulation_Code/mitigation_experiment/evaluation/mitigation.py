"""Descriptive completed-run mitigation and within-opportunity probe analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "mitigation-evaluation-v1"
BASE_ROLES = ("canonical", "replica", "shuffled")
FOUR_ROLES = (*BASE_ROLES, "original_order_probe")
EFFECT_METRICS = (
    "raw_head_slot_share", "presented_head_slot_share", "head_click_share_per_valid_click",
    "head_clicks_per_opportunity", "clicks_per_opportunity", "no_clicks_per_valid_action",
    "changed_slate_rate", "top_three_change_rate", "top_three_head_share",
    "discounted_head_exposure", "mean_absolute_rank_displacement",
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _action(event: dict[str, Any]) -> tuple[bool, bool]:
    action = event["action"]
    status, kind, item = action["status"], action["action"], action.get("clicked_item_id")
    if status not in {"valid", "invalid", "technical_failure", "fallback"}:
        raise ValueError("unknown response status")
    if status != "valid":
        if kind != "unavailable" or item is not None:
            raise ValueError("non-valid response masquerades as behavior")
        return False, False
    if not ((kind == "click" and item in event["presented_slate"]) or (kind == "no_click" and item is None)):
        raise ValueError("invalid behavioral action in completed run")
    return True, kind == "click"


def _accumulate(records, roles, head, sums, probes, reasons):
    if tuple(records) != roles:
        raise ValueError("opportunity has missing, duplicate, or out-of-order probe roles")
    canonical = records["canonical"]
    raw, shown = canonical["recommender_slate"], canonical["presented_slate"]
    for arm, event in records.items():
        displayed = event["presented_slate"]
        if event["recommender_slate"] != raw or len(displayed) != len(raw) or set(displayed) != set(raw):
            raise ValueError("probe does not preserve the raw item set")
        for field, items in (("recommender_rank_by_item", raw), ("displayed_rank_by_item", displayed)):
            if event[field] != {str(item): rank for rank, item in enumerate(items, 1)}:
                raise ValueError("saved ranks disagree with item order")
        if arm == "replica" and displayed != shown:
            raise ValueError("replica differs from canonical presentation")
        if arm == "original_order_probe" and displayed != raw:
            raise ValueError("original-order probe differs from raw SASRec")
        if arm != "canonical" and event.get("user_update") is not None:
            raise ValueError("diagnostic probe contains a persistent user update")
        _action(event)
    valid, clicked = _action(canonical)
    action = canonical["action"]
    head_clicked = clicked and action["clicked_item_id"] in head
    sums.update(opportunities=1, valid_actions=int(valid), valid_clicks=int(clicked), valid_no_clicks=int(valid and not clicked), head_clicks=int(head_clicked))
    if not valid:
        sums[action["status"] + "_actions"] += 1
    moved = sum(a != b for a, b in zip(raw, shown))
    ranks = {item: index for index, item in enumerate(raw)}
    discounts = [1 / math.log2(rank + 1) for rank in range(1, len(raw) + 1)]
    attention = lambda items: sum(weight for item, weight in zip(items, discounts) if item in head) / sum(discounts)
    top = min(3, len(raw))
    sums.update(raw_head_slots=sum(item in head for item in raw), presented_head_slots=sum(item in head for item in shown), slots=len(raw),
                changed_slates=int(bool(moved)), top_three_changes=int(raw[:top] != shown[:top]),
                top_three_head_slots=sum(item in head for item in shown[:top]), top_three_slots=top,
                discounted_head_exposure=attention(shown), raw_discounted_head_exposure=attention(raw),
                mean_absolute_rank_displacement=sum(abs(ranks[item] - index) for index, item in enumerate(shown)) / len(raw))
    if not moved:
        count = sum(item in head for item in raw)
        reason = "all_head" if count == len(raw) else "all_tail" if count == 0 else "tail_already_before_head" if raw == sorted(raw, key=lambda item: item in head) else "zero_penalty_or_ties"
        reasons[reason] += 1
    for arm in roles[1:]:
        probe = records[arm]
        pvalid, pclicked = _action(probe)
        count = probes[arm]
        count["opportunities"] += 1
        if not valid or not pvalid:
            count["excluded_non_valid_pairs"] += 1
            continue
        count["valid_pairs"] += 1
        count["action_agreements"] += action["action"] == probe["action"]["action"]
        same_item = action.get("clicked_item_id") == probe["action"].get("clicked_item_id")
        count["action_and_item_agreements"] += action["action"] == probe["action"]["action"] and same_item
        count["both_clicked"] += clicked and pclicked
        count["same_item_both_clicked"] += clicked and pclicked and same_item
        count["probe_minus_canonical_head_clicks"] += int(pclicked and probe["action"]["clicked_item_id"] in head) - int(head_clicked)


def _iteration_summary(iteration, sums, probes, reasons):
    row = {"iteration": iteration, **{key: sums[key] for key in ("opportunities", "valid_actions", "valid_clicks", "valid_no_clicks", "head_clicks", "invalid_actions", "technical_failure_actions", "fallback_actions", "raw_head_slots", "presented_head_slots", "slots")}}
    row["non_valid_actions"] = sums["opportunities"] - sums["valid_actions"]
    for output, numerator, denominator in (
        ("raw_head_slot_share", "raw_head_slots", "slots"), ("presented_head_slot_share", "presented_head_slots", "slots"),
        ("head_click_share_per_valid_click", "head_clicks", "valid_clicks"), ("head_clicks_per_opportunity", "head_clicks", "opportunities"),
        ("clicks_per_opportunity", "valid_clicks", "opportunities"), ("clicks_per_valid_action", "valid_clicks", "valid_actions"),
        ("no_clicks_per_valid_action", "valid_no_clicks", "valid_actions"), ("changed_slate_rate", "changed_slates", "opportunities"),
        ("top_three_change_rate", "top_three_changes", "opportunities"), ("top_three_head_share", "top_three_head_slots", "top_three_slots"),
        ("discounted_head_exposure", "discounted_head_exposure", "opportunities"), ("raw_discounted_head_exposure", "raw_discounted_head_exposure", "opportunities"),
        ("mean_absolute_rank_displacement", "mean_absolute_rank_displacement", "opportunities"),
    ):
        row[output] = _ratio(sums[numerator], sums[denominator])
    row["unchanged_reasons"] = dict(reasons)
    row["probes"] = {}
    for arm, count in probes.items():
        row["probes"][arm] = {
            **{key: count[key] for key in ("opportunities", "valid_pairs", "excluded_non_valid_pairs", "both_clicked", "action_agreements", "action_and_item_agreements", "same_item_both_clicked")},
            "action_agreement": _ratio(count["action_agreements"], count["valid_pairs"]),
            "action_and_item_agreement": _ratio(count["action_and_item_agreements"], count["valid_pairs"]),
            "same_item_given_both_clicked": _ratio(count["same_item_both_clicked"], count["both_clicked"]),
            "probe_minus_canonical_head_clicks_per_valid_pair": _ratio(count["probe_minus_canonical_head_clicks"], count["valid_pairs"]),
        }
    return row


def summarize_run(run_root: Path) -> dict[str, Any]:
    """Stream and validate one completed run; diagnostic receipts are excluded."""
    root = Path(run_root)
    if root.name.endswith(".incomplete") or (root / "_FAILED.json").exists() or not (root / "_COMPLETE.json").is_file():
        raise ValueError(f"analysis requires a completed, non-failed run: {root}")
    complete = json.loads((root / "_COMPLETE.json").read_text(encoding="utf-8"))
    envelope = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    run_id, manifest = envelope["run_id"], envelope["resolved_manifest"]
    if complete.get("status") != "complete" or complete.get("run_id") != run_id:
        raise ValueError("completion marker disagrees with run manifest")
    for name in ("manifest.json", "diagnostics.jsonl"):
        if complete.get("file_sha256", {}).get(name) != _hash(root / name):
            raise ValueError(f"completed artifact hash mismatch: {name}")
    roles = tuple(manifest["presentation_arms"])
    if roles not in (BASE_ROLES, FOUR_ROLES):
        raise ValueError("unsupported presentation arms")
    head = set(manifest["mitigation"]["head_item_ids"])
    aggregates, users = {}, defaultdict(set)
    digest, event_count, pending, key = hashlib.sha256(), 0, {}, None
    health = Counter()

    def consume():
        iteration, user_id = key
        if user_id in users[iteration]:
            raise ValueError("duplicated user opportunity")
        users[iteration].add(user_id)
        sums, probes, reasons = aggregates.setdefault(iteration, (Counter(), defaultdict(Counter), Counter()))
        _accumulate(pending, roles, head, sums, probes, reasons)

    with (root / "events.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            event_count += 1
            if not line.endswith(b"\n"):
                raise ValueError("incomplete event line")
            event = json.loads(line)
            health[event["action"]["status"]] += 1
            diagnostics = event["action"].get("diagnostics", {})
            update = (event.get("user_update") or {}).get("diagnostics", {})
            health["provider_calls_recorded"] += diagnostics.get("provider_attempts", 0) + update.get("reflection_attempts", 0)
            if manifest["condition"]["user_model"] in {"direct_llm", "agentic_recommender"}:
                health["incomplete_provider_accounting"] += int("provider_attempts" not in diagnostics) + int(update.get("reflection_status") == "failed_and_skipped")
            current = (event["iteration"], event["user_id"])
            arm = event["presentation_metadata"]["probe_arm"]
            if event["run_id"] != run_id or event["condition_id"] != manifest["condition"]["condition_id"] or event["seed"] != manifest["simulation_seed"]:
                raise ValueError("event identity disagrees with manifest")
            if event["interaction_id"] != f"{run_id}__t{current[0]:04d}__u{current[1]}__{arm}":
                raise ValueError("event interaction identity mismatch")
            raw = event["recommender_slate"]
            if len(raw) != manifest["slate_size"] or len(set(raw)) != len(raw):
                raise ValueError("invalid raw slate size or duplicate items")
            if current != key:
                if pending:
                    consume()
                pending, key = {}, current
            if arm in pending:
                raise ValueError("duplicated probe role")
            pending[arm] = event
    if pending:
        consume()
    if digest.hexdigest() != complete.get("file_sha256", {}).get("events.jsonl") or event_count != complete.get("row_counts", {}).get("events"):
        raise ValueError("completed event hash or count mismatch")
    iterations = manifest["iterations"]
    if complete.get("completed_iteration") != iterations or set(users) != set(range(1, iterations + 1)):
        raise ValueError("completed run does not cover its declared horizon")
    if len(users[1]) != manifest["active_users"] or any(cohort != users[1] for cohort in users.values()):
        raise ValueError("completed run does not contain the fixed active cohort")
    trajectory = [_iteration_summary(i, *aggregates[i]) for i in range(1, iterations + 1)]
    return {"run_id": run_id, "run_root": str(root.resolve()), "manifest": manifest, "events": event_count,
            "health": {"scope": "all_presentation_events", **{key: health[key] for key in ("valid", "invalid", "technical_failure", "fallback", "provider_calls_recorded", "incomplete_provider_accounting")}},
            "trajectory": trajectory, "endpoint": trajectory[-1]}


def _comparison_key(run):
    manifest = run["manifest"]
    return json.dumps({key: manifest.get(key) for key in (
        "simulation_seed", "population_seed", "training_scope", "training_users", "iterations", "slate_size",
        "inputs", "provider", "user_model_settings", "recommender_settings",
    )} | {"user_model": manifest["condition"]["user_model"], "user_update_policy": manifest["condition"]["user_update_policy"],
          "head_definition_fingerprint": manifest["mitigation"]["head_definition_fingerprint"],
          "head_item_ids": sorted(manifest["mitigation"]["head_item_ids"])}, sort_keys=True)


def analyze_runs(run_roots: Sequence[Path]) -> dict[str, Any]:
    runs = [summarize_run(path) for path in run_roots]
    if len({run["run_root"] for run in runs}) != len(runs):
        raise ValueError("duplicate run directory")
    controls, comparisons, unmatched = {}, [], []
    for run in runs:
        if run["manifest"]["condition"]["intervention"] == "control":
            key = _comparison_key(run)
            if key in controls:
                raise ValueError("ambiguous matched control")
            controls[key] = run
    for run in runs:
        if run["manifest"]["condition"]["intervention"] == "control":
            continue
        control = controls.get(_comparison_key(run))
        if control is None:
            unmatched.append({"run_id": run["run_id"], "reason": "no control with matching model, seed and input/training provenance"})
            continue
        effects = [{"iteration": treatment["iteration"], **{
            metric: treatment[metric] - baseline[metric] if treatment[metric] is not None and baseline[metric] is not None else None
            for metric in EFFECT_METRICS
        }} for treatment, baseline in zip(run["trajectory"], control["trajectory"], strict=True)]
        comparisons.append({"run_id": run["run_id"], "control_run_id": control["run_id"],
                            "user_model": run["manifest"]["condition"]["user_model"],
                            "intervention": run["manifest"]["condition"]["intervention"],
                            "trajectory_effects": effects, "endpoint_effects": effects[-1]})
    return {"schema_version": SCHEMA, "interpretation": "Descriptive run-level effects; users and iterations are not independent experimental replicates.",
            "runs": runs, "comparisons": comparisons, "unmatched_runs": unmatched}
