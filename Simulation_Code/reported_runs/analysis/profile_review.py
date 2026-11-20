"""Verified evidence for a seeded manual review of applied profile changes.

This module reuses the agreed sample; it never generates profiles or judgements.
Only committed canonical interactions supply behavioural evidence. Intermediate
texts come from trusted saved checkpoints and must match logged semantic profile
fingerprints. Missing texts remain unavailable rather than being reconstructed
from an LLM's explanation or a later profile.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import pandas as pd

from feedback_loop_experiment.utility.fingerprints import fingerprint_semantic_state

from .comparisons import RUN_KEYS, qualitative_profile_sample
from .metrics import _reference
from .readers import action, file_hash, read_json


REVIEW_FIELDS = (
    "extent_of_change", "support_in_available_evidence", "retained_interests",
    "supported_additions", "unsupported_changes", "evidence_notes", "manual_assessment",
)
EXTENT_CATEGORIES = ("minor adjustment", "substantial change", "unclear")
SUPPORT_CATEGORIES = ("supported", "mixed", "unsupported", "insufficient evidence")
MAIN_STUDY = "movielens_feedback_a300_e804"
SELECTION_METHOD = ("Reuse qualitative_profile_sample: eligible user IDs sorted by (str(type(id)), str(id)); "
    "numpy.random.default_rng(simulation_seed).choice without replacement; retain sampled order.")


def _sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if isinstance(text, str) else None


def _path(value, repo_root):
    candidate = Path(str(value).replace("\\", "/"))
    candidate = (candidate if candidate.is_absolute() else repo_root/candidate).resolve()
    if not candidate.is_relative_to(repo_root):
        raise ValueError(f"Profile-review evidence lies outside the repository: {candidate}")
    return candidate


def _profile_fingerprint(text, source_fingerprint):
    if not isinstance(text, str):
        return None
    return fingerprint_semantic_state({"source_profile_fingerprint": source_fingerprint,
        "current_profile_text": text}, "agentic-current-profile-v1")


def _checkpoint(path, source, physical_condition, selected_users, *, expected_iteration, expected_sha=None):
    """Read trusted local state on CPU and validate its semantic payload."""
    if not path.is_file():
        return None
    checksum = file_hash(path)
    if expected_sha is not None and checksum != expected_sha:
        raise ValueError("Final profile checkpoint bytes differ from the completion marker")
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("file_schema") != "feedback-loop-checkpoint-file-v1":
        raise ValueError("Unsupported trusted profile checkpoint schema")
    state = payload.get("checkpoint_state", {})
    for key, expected in (("run_id", source["run_id"]), ("condition_id", physical_condition),
                          ("seed", int(source["seed"])), ("completed_iteration", expected_iteration)):
        if state.get(key) != expected:
            raise ValueError(f"Profile checkpoint {key} differs from the completed source")
    semantic = fingerprint_semantic_state(state, state.get("schema_version", ""))
    if semantic != payload.get("checkpoint_fingerprint"):
        raise ValueError("Profile checkpoint semantic fingerprint mismatch")
    users = state.get("component_states", {}).get("user_model", {}).get("users", {})
    selected = {}
    for uid in selected_users:
        user = users.get(uid, users.get(str(uid)))
        if user is None:
            raise ValueError(f"Selected profile-review user {uid} is absent from checkpoint {path}")
        selected[uid] = {field: user.get(field) for field in
            ("initial_profile_text", "current_profile_text", "source_profile_fingerprint")}
        if not all(isinstance(value, str) for value in selected[uid].values()):
            raise ValueError("A saved selected-user profile text or source fingerprint is missing")
    result = {"users": selected, "path": str(path), "sha256": checksum, "semantic_fingerprint": semantic}
    del payload, state, users
    gc.collect()
    return result


def _source_events(source, root, marker, selected_users, catalogue):
    """Stream all event bytes for checksum verification; retain selected canonical history."""
    expected = marker.get("file_sha256", {}).get("events.jsonl")
    if not expected or (pd.notna(source.get("events_sha256")) and source.get("events_sha256") != expected):
        raise ValueError("Profile-review event checksum is absent or differs from the source inventory")
    digest, count, rows, seen = hashlib.sha256(), 0, [], set()
    with (root/"events.jsonl").open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            count += 1
            if not line.endswith(b"\n"):
                raise ValueError("Truncated profile-review event record")
            event = json.loads(line)
            if event.get("run_id") != source["run_id"] or int(event.get("seed", -1)) != int(source["seed"]):
                raise ValueError("Canonical event source identity differs from the selected completed run")
            uid = event.get("user_id")
            arm = event.get("presentation_metadata", {}).get("probe_arm", "canonical")
            if arm != "canonical":
                continue
            if uid not in selected_users:
                continue
            iteration = int(event["iteration"])
            if not 1 <= iteration <= 30 or (uid, iteration) in seen:
                raise ValueError("Duplicate or out-of-horizon selected canonical interaction")
            seen.add((uid, iteration))
            slate = event["presented_slate"]
            if len(set(slate)) != len(slate) or not set(slate).issubset(catalogue):
                raise ValueError("Selected canonical slate contains duplicate or unknown items")
            valid, clicked, item = action(event)
            update = event.get("user_update") or {}
            diag = update.get("diagnostics", {})
            decision_diag = event["action"].get("diagnostics", {})
            changed = update.get("status") == "applied" and "profile" in update.get("changed_domains", [])
            rows.append({"user_id": uid, "iteration": iteration, "event_line": line_number,
                "interaction_id": event.get("interaction_id"), "action": "click" if clicked else "no_click" if valid else "invalid",
                "action_valid": valid, "clicked_item_id": item, "clicked_title": catalogue[item].get("title", str(item)) if clicked else "",
                "displayed_item_ids": json.dumps(slate),
                "displayed_titles": "\n".join(f"{rank}. {catalogue[shown].get('title', str(shown))}" for rank, shown in enumerate(slate, 1)),
                "action_reason": event["action"].get("reason") or decision_diag.get("reason") or "",
                "planning_note": decision_diag.get("plan", ""), "reflection_note": diag.get("memory_note", ""),
                "reflection_status": diag.get("reflection_status", "not_requested"), "satisfaction": diag.get("satisfaction", ""),
                "profile_change_applied": changed, "profile_fingerprint_before": diag.get("profile_fingerprint_before") or decision_diag.get("profile_fingerprint"),
                "profile_fingerprint_after": diag.get("profile_fingerprint"),
                "decision_profile_fingerprint": decision_diag.get("profile_fingerprint"),
                "reflection_response_id": diag.get("reflection_response_id"), "update_id": update.get("update_id")})
    if digest.hexdigest() != expected or count != marker.get("row_counts", {}).get("events"):
        raise ValueError("Profile-review event bytes/count differ from the completion marker")
    for uid in selected_users:
        if sorted(row["iteration"] for row in rows if row["user_id"] == uid) != list(range(1, 31)):
            raise ValueError(f"The selected user {uid} does not have all 30 canonical interactions")
    return pd.DataFrame(rows), expected


def _verify_expected_sample(sample, expected):
    if expected is None:
        return
    keys = RUN_KEYS+["user_id", "review_order", "eligible_users", "requested_sample", "applied_change_count"]
    if set(keys)-set(expected):
        raise ValueError("The frozen qualitative sample is missing selection metadata")
    left = sample[keys].sort_values(RUN_KEYS+["review_order"]).reset_index(drop=True)
    right = expected[keys].sort_values(RUN_KEYS+["review_order"]).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise ValueError("Profile-review selection differs from the existing agreed sample") from exc


def build_profile_review(profile_changes, profiles, sources, repo_root, *, expected_sample=None, n=10,
                         expected_seeds=(42, 43, 44), progress=None):
    """Return cases, every applied change, full canonical history, and method metadata.

    Files are only read. Checkpoints are deserialised on CPU without constructing
    a response model, and their semantic fingerprints are checked. Missing
    intermediate checkpoints leave the corresponding text explicitly missing.
    The final checkpoint, completion marker, event bytes, cached selected-change
    identities, and available cached initial/final texts must agree.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("The requested number of review users must be a positive integer")
    repo_root = Path(repo_root).resolve()
    sample = qualitative_profile_sample(profile_changes, n=n)
    sample = sample.loc[sample.study.eq(MAIN_STUDY)].copy()
    if set(map(int, sample.seed.unique())) != set(expected_seeds):
        raise ValueError("The main full-reflection sample does not cover the expected simulation seeds")
    _verify_expected_sample(sample, expected_sample)
    sample["case_id"] = [f"s{int(seed)}_u{uid}" for seed, uid in zip(sample.seed, sample.user_id)]
    if sample.duplicated(["seed", "user_id"]).any():
        raise ValueError("Each seed's qualitative sample must contain distinct users")
    cases, changes, histories, selections, proof = [], [], [], [], []
    metadata_keys = RUN_KEYS+["source_id", "run_id", "physical_run_id", "source_directory", "physical_condition_id", "events_sha256", "complete_sha256"]
    for key, selected in sample.groupby(RUN_KEYS, sort=True, dropna=False):
        identity = dict(zip(RUN_KEYS, key))
        matching = sources
        for column, value in identity.items():
            matching = matching.loc[matching[column].eq(value)]
        if len(matching) != 1:
            raise ValueError("Exactly one main physical source must match each selected profile-review population")
        source = matching.iloc[0].to_dict()
        if source.get("status", "complete") != "complete" or source.get("model") != "agentic_recommender" or source.get("user_update_policy") != "full_reflection":
            raise ValueError("Profile review requires the completed main full-reflection agent condition")
        root = _path(source["source_directory"], repo_root)
        complete_path = root/"_COMPLETE.json"
        complete_sha = file_hash(complete_path)
        if source.get("complete_sha256") and complete_sha != source["complete_sha256"]:
            raise ValueError("Profile-review completion marker differs from the selected source")
        marker = read_json(complete_path)
        if marker.get("status") != "complete" or marker.get("completed_iteration") != 30 or marker.get("run_id") != source["run_id"] or (root/"_FAILED.json").exists():
            raise ValueError("Profile review requires a clean completed 30-iteration source")
        if file_hash(root/"manifest.json") != marker.get("file_sha256", {}).get("manifest.json"):
            raise ValueError("Profile-review source manifest checksum mismatch")
        envelope = read_json(root/"manifest.json")
        manifest = envelope.get("resolved_manifest", envelope)
        physical_condition = manifest["condition"]["condition_id"]
        reference = _reference(source, repo_root)
        if manifest["inputs"]["dataset"] != reference["dataset_fingerprint"]:
            raise ValueError("Profile-review catalogue does not match the source dataset")
        users = selected.user_id.tolist()
        if progress:
            progress(f"Profile review: seed {int(source['seed'])}, {len(users)} selected users")
        final_path = _path(root/str(marker["final_checkpoint_path"]), repo_root)
        final = _checkpoint(final_path, source, physical_condition, users, expected_iteration=30,
                            expected_sha=marker["final_checkpoint_sha256"])
        if final is None:
            raise ValueError("The completed source's final profile checkpoint is unavailable")
        states = {30: final}
        missing_checkpoints = []
        for iteration in range(1, 30):
            path = root/"checkpoints"/f"iteration_{iteration:04d}"/"checkpoint.pt"
            state = _checkpoint(path, source, physical_condition, users, expected_iteration=iteration)
            states[iteration] = state
            if state is None:
                missing_checkpoints.append(iteration)
        history, events_sha = _source_events(source, root, marker, users, reference["catalogue"])
        metadata = {**identity, "source_id": source["source_id"], "run_id": source["run_id"],
            "physical_run_id": source["run_id"], "source_directory": str(root), "physical_condition_id": physical_condition,
            "events_sha256": events_sha, "complete_sha256": complete_sha}
        selections.append({**metadata, "eligible_users": int(selected.eligible_users.iloc[0]),
            "requested_sample": n, "selected_users": len(selected), "simulation_seed_for_selection": int(source["seed"]),
            "selection_method": SELECTION_METHOD, "missing_intermediate_checkpoints": ", ".join(map(str, missing_checkpoints))})
        for entry in selected.to_dict("records"):
            uid = entry["user_id"]
            user_final = final["users"][uid]
            initial_text, final_text = user_final["initial_profile_text"], user_final["current_profile_text"]
            source_profile_fingerprint = user_final["source_profile_fingerprint"]
            cached = profiles
            for column, value in {**identity, "user_id": uid}.items():
                cached = cached.loc[cached[column].eq(value)]
            if len(cached) > 1:
                raise ValueError("Duplicate cached initial/final profiles in the selected main condition")
            if len(cached):
                for column, expected in (("initial_profile", initial_text), ("final_profile", final_text)):
                    if pd.notna(cached.iloc[0].get(column)) and cached.iloc[0][column] != expected:
                        raise ValueError(f"Cached {column} differs from verified saved state")
            case_meta = {**metadata, "case_id": entry["case_id"], "user_id": uid, "review_order": entry["review_order"]}
            user_history = history.loc[history.user_id.eq(uid)].sort_values("iteration")
            expected_changes = profile_changes
            for column, value in {**identity, "user_id": uid}.items():
                expected_changes = expected_changes.loc[expected_changes[column].eq(value)]
            actual_change_iterations = user_history.loc[user_history.profile_change_applied, "iteration"].tolist()
            if sorted(expected_changes.iteration.tolist()) != actual_change_iterations or len(actual_change_iterations) != int(entry["applied_change_count"]):
                raise ValueError("Applied canonical changes differ from the frozen selection evidence")
            case_changes = []
            for event in user_history.to_dict("records"):
                iteration = event["iteration"]
                before_state = states.get(iteration-1)
                after_state = states[iteration]
                before = initial_text if iteration == 1 else before_state["users"][uid]["current_profile_text"] if before_state else None
                after = after_state["users"][uid]["current_profile_text"] if after_state else None
                for saved in (before_state, after_state):
                    if saved and (saved["users"][uid]["initial_profile_text"] != initial_text or saved["users"][uid]["source_profile_fingerprint"] != source_profile_fingerprint):
                        raise ValueError("Initial profile/source identity changes across saved checkpoints")
                before_fingerprint = _profile_fingerprint(before, source_profile_fingerprint)
                after_fingerprint = _profile_fingerprint(after, source_profile_fingerprint)
                for actual, logged, name in ((before_fingerprint, event["profile_fingerprint_before"], "before"),
                    (before_fingerprint, event["decision_profile_fingerprint"], "decision"), (after_fingerprint, event["profile_fingerprint_after"], "after")):
                    if actual is not None and logged is not None and actual != logged:
                        raise ValueError(f"Checkpoint profile {name} fingerprint differs from the canonical event")
                if before is not None and after is not None and (before != after) != bool(event["profile_change_applied"]):
                    raise ValueError("Checkpoint text changes disagree with the committed profile-update flag")
                evidence = {**case_meta, **event, "profile_before": before, "profile_after": after,
                    "profile_before_available": before is not None, "profile_after_available": after is not None,
                    "profile_before_text_sha256": _sha_text(before), "profile_after_text_sha256": _sha_text(after),
                    "profile_text_evidence": "verified iteration checkpoint" if before is not None and after is not None else "intermediate checkpoint unavailable",
                    "before_checkpoint_path": final["path"] if iteration == 1 else before_state["path"] if before_state else None,
                    "before_checkpoint_sha256": final["sha256"] if iteration == 1 else before_state["sha256"] if before_state else None,
                    "before_text_origin": "initial_profile_text in verified final checkpoint" if iteration == 1 else "current_profile_text in preceding checkpoint" if before_state else "unavailable",
                    "after_checkpoint_path": after_state["path"] if after_state else None,
                    "after_checkpoint_sha256": after_state["sha256"] if after_state else None,
                    "after_checkpoint_semantic_fingerprint": after_state["semantic_fingerprint"] if after_state else None}
                histories.append(evidence)
                if event["profile_change_applied"]:
                    evidence = {**evidence, "change_order": len(case_changes)+1, **{field: "" for field in REVIEW_FIELDS}}
                    changes.append(evidence)
                    case_changes.append(evidence)
            cases.append({**entry, **case_meta, "initial_profile": initial_text, "final_profile": final_text,
                "initial_profile_text_sha256": _sha_text(initial_text), "final_profile_text_sha256": _sha_text(final_text),
                "source_profile_fingerprint": source_profile_fingerprint, "final_checkpoint_path": final["path"],
                "final_checkpoint_sha256": final["sha256"], "final_checkpoint_semantic_fingerprint": final["semantic_fingerprint"],
                "canonical_history_count": len(user_history), "change_evidence_count": len(case_changes),
                "changes_with_both_texts": sum(row["profile_before_available"] and row["profile_after_available"] for row in case_changes),
                "review_status": "Not yet reviewed", **{field: "" for field in REVIEW_FIELDS}})
        proof.append({"seed": int(source["seed"]), "source_id": source["source_id"], "events_sha256": events_sha,
            "complete_sha256": complete_sha, "available_checkpoints": 30-len(missing_checkpoints),
            "missing_checkpoint_iterations": missing_checkpoints})
    case_frame = pd.DataFrame(cases).sort_values(["seed", "review_order"]).reset_index(drop=True)
    change_frame = pd.DataFrame(changes).sort_values(["seed", "review_order", "iteration"]).reset_index(drop=True)
    history_frame = pd.DataFrame(histories).sort_values(["seed", "review_order", "iteration"]).reset_index(drop=True)
    method = {"condition": "Main MovieLens full-reflection L3, 300 active users", "selection": SELECTION_METHOD,
        "requested_per_seed": n, "selected_per_seed": {str(int(seed)): int(count) for seed, count in case_frame.groupby("seed").size().items()},
        "extent_categories": list(EXTENT_CATEGORIES), "support_categories": list(SUPPORT_CATEGORIES),
        "review_fields": list(REVIEW_FIELDS), "source_proof": proof,
        "sample_sha256": hashlib.sha256(case_frame[RUN_KEYS+["user_id", "review_order"]].to_json(orient="records").encode()).hexdigest(),
        "evidence_scope": "All 30 canonical interactions and every applied profile change for each selected user; judge against the original profile and full available history.",
        "interpretation": "Unfilled descriptive human review; no automatic assessments, psychometric score, or validation of human preference change.",
        "text_provenance": "Saved checkpoint text checked against project semantic profile fingerprints; plain text SHA256 is provided separately for export integrity."}
    return {"cases": case_frame, "changes": change_frame, "history": history_frame,
            "selection": pd.DataFrame(selections), "method": method}
