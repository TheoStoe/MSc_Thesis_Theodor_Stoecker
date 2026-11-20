"""Explicit, offline final-analysis selection; original runs and registry stay intact.

Logical study memberships are separate from physical trajectories. In particular,
reused controls and L0/L1 in-processing aliases never become new replications.
Only published, successfully completed 30-iteration trajectories enter analysis.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = Path(__file__).resolve().parent / "sources"
MAIN_STUDIES = (
    "movielens_feedback_a300_e804",
    "movielens_sampling_outcome_a150_r150_e804",
    "movielens_mitigation_a300_e804",
)
GOODREADS_LLM_ROOTS = {
    42: "new_dataset_experiment/outputs/llm_representative_seed42_descriptions200_v1",
    43: "new_dataset_experiment/outputs/llm_representative_seed43_population11615_descriptions200_v1",
    44: "new_dataset_experiment/outputs/llm_representative_seed44_population5245_descriptions200_v1",
}
GOODREADS_FROZEN_AGENT_L0_ROOT = "new_dataset_experiment/outputs/llm_frozen_agent_l0_seed42_43_44_descriptions200_v1/runs"
GOODREADS_FROZEN_AGENT_L0_PLAN = str(Path(GOODREADS_FROZEN_AGENT_L0_ROOT).parent / "plan.json")
GOODREADS_FROZEN_AGENT_L0_CONDITION = "L0_control__agentic_recommender__sasrec__order_probe"
GOODREADS_ROOTS = (
    "new_dataset_experiment/outputs/computational_representative_seed42_v3",
    "new_dataset_experiment/outputs/computational_representative_seed42_l2_parallel3_restart1",
    "new_dataset_experiment/outputs/computational_representative_seed43_population11615_v1",
    "new_dataset_experiment/outputs/computational_representative_seed44_population5245_v1",
    *GOODREADS_LLM_ROOTS.values(),
    GOODREADS_FROZEN_AGENT_L0_ROOT,
)
GOODREADS_LLM_REPLICATIONS = {
    GOODREADS_LLM_ROOTS[43]: (43, 11615),
    GOODREADS_LLM_ROOTS[44]: (44, 5245),
}
GOODREADS_LLM_PLAN = "new_dataset_experiment/outputs/llm_seed43_44_parallel3_descriptions200_v1/plan.json"
GOODREADS_LLM_CONDITIONS = {
    "L0_control__direct_llm__sasrec__order_probe": ("direct_llm", "L0_control", "frozen"),
    "L2_ml_model__direct_llm__sasrec__order_probe": ("direct_llm", "L2_ml_model", "frozen"),
    "L2_ml_model__agentic_recommender__sasrec__order_probe": ("agentic_recommender", "L2_ml_model", "frozen"),
    "L3_individual__agentic_recommender__sasrec__full_reflection__order_probe": (
        "agentic_recommender", "L3_individual", "full_reflection"),
}
ML44_ROOT = "mitigation_experiment/outputs/sasrec_budget_p18844_seed44_a300_r0_e804_b128_treatments_only_v1"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(repo, path):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(repo):
        raise ValueError(f"Source outside repository: {candidate}")
    return candidate


def _input_paths(repo, dataset, seed, population, active):
    if dataset == "movielens":
        prefix = "new_sampling" if active == 300 else "l4_l5"
        reserve = 300 - active
        bundle = repo / f"feedback_loop_experiment/inputs/{prefix}_p{population}_seed{seed}_a{active}_r{reserve}_e804_v1"
        return {"inputs_directory": str(bundle), "prepared_directory": str(bundle / "prepared_data"),
                "population_path": str(bundle / "population.json"), "cohort_path": "", "labels_path": ""}
    suffix = "" if seed == 42 else f"_population_{population}"
    bundle = repo / f"new_dataset_experiment/inputs/goodreads_representative_a300_v2{suffix}"
    labels = (bundle / "item_labels.json" if seed == 42 else
              repo / f"new_dataset_experiment/inputs/gender_labels_expanded_population_{population}_v1/item_labels.json")
    return {"inputs_directory": str(bundle), "prepared_directory": str(bundle / "prepared"),
            "population_path": str(bundle / "population.json"),
            "cohort_path": str(repo / f"new_dataset_experiment/inputs/representative_a300_v2{suffix}/cohort.json"),
            "labels_path": str(labels)}


def _published(root):
    return [p.parent for p in sorted(root.rglob("_COMPLETE.json"))
            if not any(part.endswith(".incomplete") for part in p.relative_to(root).parts)
            and (p.parent / "events.jsonl").is_file()]


def _validate_goodreads_llm_replication(repo, root, resolved, run_id):
    """Bind the two added cohorts to their saved descriptions200 protocol."""
    relative = root.parent.relative_to(repo).as_posix()
    if relative not in GOODREADS_LLM_REPLICATIONS:
        return
    seed, population = GOODREADS_LLM_REPLICATIONS[relative]
    plan = _read(repo / GOODREADS_LLM_PLAN)
    reference = _read(repo / "new_dataset_experiment/outputs/llm_seed42_parallel4_descriptions200_v1/plan.json")
    assets = plan["assets"][str(seed)]
    condition = resolved["condition"]
    name = condition["condition_id"]
    expected_id = f"goodreads_a300_e804_v1__{name}__seed_{seed}"
    planned = [job for job in plan["jobs"] if job["kind"] == "campaign" and job["seed"] == seed]
    selected = [job for job in planned if job["condition"] == name]
    if (plan.get("schema") != "goodreads-llm-replications-v1"
            or assets["population"] != population
            or _inside(repo, assets["output"]) != root.parent
            or len(planned) != 4 or {job["condition"] for job in planned} != set(GOODREADS_LLM_CONDITIONS)
            or len(selected) != 1 or selected[0]["run_id"] != expected_id
            or _inside(repo, selected[0]["marker"]) != root / "_COMPLETE.json"
            or run_id != expected_id):
        raise ValueError(f"Goodreads LLM source differs from its saved replication plan: {root}")
    observed_condition = tuple(condition[key] for key in ("user_model", "feedback_loop", "user_update_policy"))
    if (observed_condition != GOODREADS_LLM_CONDITIONS[name]
            or condition.get("presentation_policy") != "canonical_replica_and_shuffled_probe"
            or condition.get("population_policy") != "fixed_active_users"
            or resolved.get("input_format") != "books"
            or (resolved["simulation_seed"], resolved["population_seed"]) != (seed, population)
            or tuple(resolved[key] for key in ("active_users", "reserve_users", "training_users", "slate_size")) != (300, 0, 300, 10)
            or resolved.get("recommender_settings") != {"batch_size": 128, "device": "cuda", "max_sequence_length": 200, "num_epochs": 804}
            or plan["provider"] != reference["provider"]
            or any(resolved["provider"].get(key) != value for key, value in reference["provider"].items())):
        raise ValueError(f"Goodreads LLM source has a different response or training protocol: {root}")
    overlay = _inside(repo, assets["overlay"])
    text = _read(overlay / "manifest.json")
    model_settings = resolved.get("user_model_settings", {})
    if (text["policy"].get("description_max_chars") != 200
            or text["policy_fingerprint"] != reference["item_summaries_identity"]["policy_fingerprint"]
            or text["source_dataset_fingerprint"] != resolved["inputs"]["dataset"]
            or text["item_text_fingerprint"] != assets["item_text_fingerprint"]
            or model_settings.get("llm_item_text_fingerprint") != assets["item_text_fingerprint"]
            or _inside(repo, model_settings.get("llm_item_text_artifact_path", "")) != overlay):
        raise ValueError(f"Goodreads LLM source has a different descriptions200 overlay: {root}")


def _validate_goodreads_frozen_agent_l0(repo, root, resolved, run_id):
    """Bind the added L0 runs to their saved plan and matched frozen L2 inputs."""
    if root.parent != _inside(repo, GOODREADS_FROZEN_AGENT_L0_ROOT):
        return
    plan = _read(repo / GOODREADS_FROZEN_AGENT_L0_PLAN)
    populations = {42: 16040, 43: 11615, 44: 5245}
    seed = resolved.get("simulation_seed")
    jobs = [job for job in plan["jobs"] if job["kind"] == "campaign"]
    name = GOODREADS_FROZEN_AGENT_L0_CONDITION
    expected_id = f"goodreads_a300_e804_v1__{name}__seed_{seed}"
    selected = [job for job in jobs if job["seed"] == seed]
    if (plan.get("schema") != "goodreads-frozen-agent-l0-v1"
            or seed not in populations or len(jobs) != 3
            or {job["seed"] for job in jobs} != set(populations)
            or any(job["condition"] != name for job in jobs)
            or len(selected) != 1 or selected[0]["run_id"] != expected_id
            or _inside(repo, selected[0]["marker"]) != root / "_COMPLETE.json"
            or run_id != expected_id):
        raise ValueError(f"Goodreads frozen L0 source differs from its saved plan: {root}")
    assets = plan["assets"][str(seed)]
    condition = resolved["condition"]
    if (assets["population"] != populations[seed]
            or resolved.get("population_seed") != populations[seed]
            or condition.get("condition_id") != name
            or tuple(condition.get(key) for key in ("user_model", "feedback_loop", "user_update_policy"))
               != ("agentic_recommender", "L0_control", "frozen")
            or condition.get("realized_loop_types") != ["candidate_eligibility_feedback"]
            or condition.get("presentation_policy") != "canonical_replica_and_shuffled_probe"
            or condition.get("population_policy") != "fixed_active_users"
            or resolved.get("input_format") != "books"
            or tuple(resolved.get(key) for key in ("active_users", "reserve_users", "training_users", "slate_size", "iterations"))
               != (300, 0, 300, 10, 30)):
        raise ValueError(f"Goodreads frozen L0 source has a different scientific protocol: {root}")
    l2_name = "L2_ml_model__agentic_recommender__sasrec__order_probe"
    l2_id = f"goodreads_a300_e804_v1__{l2_name}__seed_{seed}"
    reference = _inside(repo, GOODREADS_LLM_ROOTS[seed]) / l2_id
    envelope = _read(reference / "manifest.json")
    l2 = envelope["resolved_manifest"]
    marker = _read(reference / "_COMPLETE.json")
    if (marker.get("status") != "complete" or marker.get("completed_iteration") != 30
            or marker.get("run_id") != l2_id or envelope.get("run_id") != l2_id
            or (reference / "_FAILED.json").exists()
            or marker.get("file_sha256", {}).get("manifest.json") != _sha(reference / "manifest.json")
            or l2["condition"].get("condition_id") != l2_name
            or tuple(l2["condition"].get(key) for key in ("user_model", "feedback_loop", "user_update_policy"))
               != ("agentic_recommender", "L2_ml_model", "frozen")):
        raise ValueError(f"Goodreads frozen L0 lacks its verified frozen L2 reference: {root}")
    shared = ("simulation_seed", "population_seed", "input_format", "inputs", "provider",
              "user_model_settings", "recommender_settings", "response_artifact_fingerprints",
              "training_scope", "active_users", "reserve_users", "training_users", "slate_size", "iterations",
              "policy_settings", "initial_recommender_restored")
    stage_fields = {"condition_id", "feedback_loop", "analysis_bundle", "realized_loop_types"}
    if (any(resolved.get(key) != l2.get(key) for key in shared)
            or {k: v for k, v in condition.items() if k not in stage_fields}
               != {k: v for k, v in l2["condition"].items() if k not in stage_fields}
            or resolved["inputs"] != assets["matched_l2_inputs"]
            or any(resolved["provider"].get(key) != value for key, value in plan["provider"].items())):
        raise ValueError(f"Goodreads frozen L0 inputs/provider differ from matched L2: {root}")
    overlay = _inside(repo, assets["overlay"])
    text = _read(overlay / "manifest.json")
    settings = resolved["user_model_settings"]
    if (text["policy"].get("description_max_chars") != 200
            or text["source_dataset_fingerprint"] != resolved["inputs"]["dataset"]
            or text["item_text_fingerprint"] != settings.get("llm_item_text_fingerprint")
            or _inside(repo, settings.get("llm_item_text_artifact_path", "")) != overlay):
        raise ValueError(f"Goodreads frozen L0 source has a different descriptions200 overlay: {root}")


def _physical_record(repo, path, *, verify_bytes):
    root = _inside(repo, path)
    envelope = _read(root / "manifest.json")
    resolved = envelope.get("resolved_manifest", envelope)
    condition = resolved["condition"]
    marker_path = root / "_COMPLETE.json"
    complete = marker_path.is_file() and not root.name.endswith(".incomplete")
    marker = _read(marker_path) if complete else {}
    run_id = envelope["run_id"]
    _validate_goodreads_llm_replication(repo, root, resolved, run_id)
    _validate_goodreads_frozen_agent_l0(repo, root, resolved, run_id)
    status = "complete" if complete else "pending"
    errors = []
    if (root / "_FAILED.json").exists():
        status = "failed"
    if complete:
        if marker.get("status") != "complete" or marker.get("completed_iteration") != 30 or marker.get("run_id") != run_id:
            raise ValueError(f"Invalid completion marker: {root}")
        if root.name != run_id:
            raise ValueError(f"Published run directory has a different identity: {root}")
        for name in ("manifest.json", "events.jsonl", "diagnostics.jsonl"):
            expected = marker.get("file_sha256", {}).get(name)
            if not expected or not (root / name).is_file():
                raise ValueError(f"Missing completion checksum or source log: {root / name}")
            if name == "manifest.json" or verify_bytes:
                if _sha(root / name) != expected:
                    raise ValueError(f"Source log checksum mismatch: {root / name}")
        final = _inside(root, marker["final_checkpoint_path"].replace("\\", "/"))
        if not final.is_file():
            raise ValueError(f"Missing final checkpoint: {root}")
    stage = condition["feedback_loop"]
    mitigation = resolved.get("policy_settings", {}).get("gender_mitigation", {})
    intervention = condition.get("intervention") or mitigation.get("strategy") or "control"
    seed = int(resolved["simulation_seed"])
    population = int(resolved["population_seed"])
    dataset = "goodreads" if resolved.get("input_format") == "books" or "goodreads" in run_id else "movielens"
    population_map = {"movielens": {42: 1054, 43: 19277, 44: 18844},
                      "goodreads": {42: 16040, 43: 11615, 44: 5245}}
    if population_map[dataset].get(seed) != population:
        raise ValueError(f"Unexpected dataset/seed/population combination: {dataset}/{seed}/{population}")
    if resolved.get("iterations") != 30 or resolved.get("recommender_settings", {}).get("num_epochs") != 804:
        raise ValueError(f"Unexpected analysis horizon or SASRec epoch budget: {root}")
    relative = root.relative_to(repo).as_posix()
    complete_hash = _sha(marker_path) if complete else "unpublished"
    source_id = hashlib.sha256((relative + "\n" + complete_hash).encode()).hexdigest()[:20]
    active = int(resolved["active_users"])
    record = {"source_id": source_id, "source_directory": relative, "run_id": run_id,
              "dataset": dataset, "status": status, "completion": complete,
              "simulation_seed": seed, "seed": seed, "population_seed": population,
              "source_condition_id": condition["condition_id"], "model": condition["user_model"],
              "feedback_stage": stage, "stage": stage, "intervention": intervention,
              "user_update_policy": condition["user_update_policy"],
              "initial_active_users": active, "initial_reserve_users": int(resolved["reserve_users"]),
              "training_users": int(resolved["training_users"]), "iterations": int(resolved["iterations"]),
              "sasrec_epochs": resolved.get("recommender_settings", {}).get("num_epochs"),
              "slate_size": int(resolved["slate_size"]),
              "complete_sha256": complete_hash, "manifest_sha256": _sha(root / "manifest.json"),
              "events_sha256": marker.get("file_sha256", {}).get("events.jsonl", ""),
              "diagnostics_sha256": marker.get("file_sha256", {}).get("diagnostics.jsonl", ""),
              "source_bytes_verified": bool(complete and verify_bytes),
              "event_rows": marker.get("row_counts", {}).get("events"),
              "required_arms": "canonical|replica|shuffled",
              "additional_original_order_probe": "post_processing" == intervention and dataset == "movielens",
              "error": "; ".join(errors),
              **_input_paths(repo, dataset, seed, population, active)}
    return record


def discover_sources(repo_root=REPO, *, verify_bytes=False):
    """Verify the exact final-study source selection after local archive restore.

    The pinned inventory preserves logical aliases and original settings. This
    command does not rediscover campaigns, change membership, or run simulations.
    Source catalogues, histories, and other licensed inputs must be obtained or
    restored separately; they are not included in the public replay data.
    """
    repo = Path(repo_root).resolve()
    records = _read(Path(__file__).with_name("sources") / "pinned_inventory.json")
    if not records:
        raise ValueError("The pinned final-study inventory is empty")
    checked = set()
    path_fields = ("inputs_directory", "prepared_directory", "population_path", "cohort_path", "labels_path")
    for row in records:
        root = _inside(repo, row["source_directory"])
        marker_path, manifest_path = root / "_COMPLETE.json", root / "manifest.json"
        if not marker_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("Raw re-extraction requires the private run archives restored at their original relative paths; missing: " + str(root))
        if row["source_id"] not in checked:
            if _sha(marker_path) != row["complete_sha256"] or _sha(manifest_path) != row["manifest_sha256"]:
                raise ValueError("Restored completion marker or manifest differs from the pinned source: " + str(root))
            marker = _read(marker_path)
            if marker.get("status") != "complete" or marker.get("completed_iteration") != 30:
                raise ValueError("Pinned source is not a completed30-iteration run: " + str(root))
            for filename, key in (("events.jsonl", "events_sha256"), ("diagnostics.jsonl", "diagnostics_sha256")):
                expected = row.get(key)
                if expected and (not (root / filename).is_file() or marker.get("file_sha256", {}).get(filename) != expected):
                    raise ValueError("Restored source log or its recorded hash is missing: " + str(root / filename))
                if expected and verify_bytes and _sha(root / filename) != expected:
                    raise ValueError("Restored source log bytes differ: " + str(root / filename))
            checked.add(row["source_id"])
        row["source_directory"] = root.relative_to(repo).as_posix()
        for field in path_fields:
            if row.get(field):
                row[field] = str(_inside(repo, row[field]))
        row["source_bytes_verified"] = bool(verify_bytes)
    return pd.DataFrame(records).sort_values(["study", "simulation_seed", "logical_condition_id"]).reset_index(drop=True)




def snapshot_sources(repo_root=REPO, *, output_dir=None, verify_bytes=False):
    """Refresh only analysis inventories; never copy or alter a source result."""
    repo = Path(repo_root).resolve()
    directory = Path(output_dir) if output_dir is not None else repo / "reported_runs/analysis/sources"
    frame = discover_sources(repo, verify_bytes=verify_bytes)
    completed = frame.loc[frame.status.eq("complete")]
    health = {"schema_version": "thesis-analysis-source-status-v1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "completed_logical_cells": len(completed),
              "unique_completed_trajectories": completed.source_id.nunique(),
              "pending_logical_cells": int(frame.status.ne("complete").sum()),
              "completed_by_study": completed.groupby("study").size().to_dict(),
              "unique_by_study": completed.groupby("study").source_id.nunique().to_dict(),
              "pending": frame.loc[frame.status.ne("complete"), ["study", "model", "intervention", "simulation_seed", "source_directory"]].to_dict("records"),
              "source_log_hashes_verified": bool(verify_bytes),
              "source_selection": "Reviewed existing MovieLens main studies plus explicit later completed campaigns and immutable extension plans.",
              "notes": ["Pending trajectories are listed for visibility and excluded from metric calculations.",
                        "Logical aliases share a source_id and are not independent replications.",
                        "Goodreads Direct LLM and agent comparisons cover seeds 42, 43, and 44 under the descriptions200 protocol; these replications also vary the frozen population.",
                        "Psychological modelling remains optional and is excluded from the seven main RQ inventories.",
                        "No experiments, providers, training, or original result copies are invoked."]}
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "inventory.csv", index=False)
    (directory / "inventory.json").write_text(json.dumps(frame.to_dict("records"), indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "status.json").write_text(json.dumps(health, indent=2, ensure_ascii=False), encoding="utf-8")
    return frame, health


def load_sources(repo_root=REPO, *, refresh=False, verify_bytes=False, include_pending=False):
    """Notebook-friendly source DataFrame, completed cells only by default."""
    path = Path(repo_root).resolve() / "reported_runs/analysis/sources/inventory.json"
    if refresh or not path.is_file():
        frame, _ = snapshot_sources(repo_root, verify_bytes=verify_bytes)
    else:
        frame = pd.DataFrame(_read(path))
    return frame if include_pending else frame.loc[frame.status.eq("complete")].reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify source event and diagnostics bytes against saved completion hashes.")
    args = parser.parse_args()
    _, report = snapshot_sources(verify_bytes=args.verify)
    print(json.dumps(report, indent=2))
