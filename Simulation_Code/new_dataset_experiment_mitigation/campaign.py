"""Prepare matched Goodreads mitigation; execute only with explicit run flags."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
SOURCE = REPOSITORY / "new_dataset_experiment"
EXPERIMENT_ID = "goodreads_gender_mitigation_e804_v1"
PLAN_SCHEMA = "goodreads-gender-mitigation-campaign-v1"
MODELS = ("position_biased", "ncm_inspired_gru_uniform", "ncm_inspired_gru_mansoury")
STAGES = ("L0_control", "L1_feature", "L2_ml_model")
INTERVENTIONS = ("control", "in_processing", "post_processing")
SEEDS = (42, 43, 44)
DEFAULT_PLAN = ROOT / "configs" / "gender_mitigation_v1_all_seeds.json"
STRENGTHS = {"women_positive_weight": 1.25, "women_rank_promotion": 5}


@dataclass(frozen=True)
class Binding:
    seed: int
    population_seed: int
    inputs_root: Path
    labels_path: Path
    l0_source_root: Path
    l2_source_root: Path


def bindings() -> dict[int, Binding]:
    result = {}
    for seed, population, suffix, label_suffix, l0, l2 in (
        (42, 16040, "", "a300", "computational_representative_seed42_v3",
         "computational_representative_seed42_l2_parallel3_restart1"),
        (43, 11615, "_population_11615", "population_11615",
         "computational_representative_seed43_population11615_v1",
         "computational_representative_seed43_population11615_v1"),
        (44, 5245, "_population_5245", "population_5245",
         "computational_representative_seed44_population5245_v1",
         "computational_representative_seed44_population5245_v1"),
    ):
        result[seed] = Binding(seed, population,
            SOURCE / "inputs" / f"goodreads_representative_a300_v2{suffix}",
            SOURCE / "inputs" / f"gender_labels_expanded_{label_suffix}_v1" / "item_labels.json",
            SOURCE / "outputs" / l0, SOURCE / "outputs" / l2)
    return result


@dataclass(frozen=True)
class Cell:
    user_model: str
    stage: str
    intervention: str

    def __post_init__(self) -> None:
        if self.user_model not in MODELS or self.stage not in STAGES or self.intervention not in INTERVENTIONS:
            raise ValueError("unknown computational model, feedback stage, or intervention")

    @property
    def condition_id(self) -> str:
        return f"{self.stage}__{self.user_model}__sasrec__gender_{self.intervention}__order_probe"

    @property
    def alias_of(self) -> str | None:
        if self.intervention == "in_processing" and self.stage != "L2_ml_model":
            return Cell(self.user_model, self.stage, "control").condition_id
        return None


def cells() -> tuple[Cell, ...]:
    return tuple(Cell(model, stage, intervention)
        for model in MODELS for stage in STAGES for intervention in INTERVENTIONS)


def selected_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(seeds))
    if not result or any(isinstance(seed, bool) or seed not in SEEDS for seed in result):
        raise ValueError("select seeds 42, 43, and/or 44")
    return result


def describe(seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    seeds = selected_seeds(seeds)
    count = len(seeds)
    return {
        "schema": PLAN_SCHEMA, "status": "described_not_launched",
        "experiment_id": EXPERIMENT_ID, "simulation_seeds": list(seeds),
        "population_seeds": {str(seed): bindings()[seed].population_seed for seed in seeds},
        "protocol": {"active_users": 300, "reserve_users": 0, "training_users": 300,
            "iterations": 30, "slate_size": 10, "sasrec_epochs": 804,
            "batch_size": 128, "max_sequence_length": 200,
            "training_scope": "active_reserve", "canonical_only_feedback": True,
            "presentation_arms": ["canonical", "replica", "shuffled"],
            "shared_initial_model": "earlier unweighted initial SASRec checkpoint",
            "in_processing_start": "subsequent L2 refits only", "paid_api_calls": 0},
        "intervention_strengths": dict(STRENGTHS),
        "logical_cells": 27 * count, "in_processing_aliases": 6 * count,
        "distinct_trajectories": 21 * count, "existing_controls_to_validate": 6 * count,
        "new_trajectories_if_controls_validate": 15 * count,
        "conditions": [{**asdict(cell), "condition_id": cell.condition_id,
                        "alias_of": cell.alias_of} for cell in cells()],
        "analysis": {"rank_discount": "1/log2(r+1)", "ranks_one_based": True,
            "unknown_and_ambiguous_retained": True, "recommendation_quality_metrics": [],
            "psychological_modelling": "optional"},
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confined(path: str | Path) -> Path:
    """All writable destinations resolve inside this extension folder."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path == ROOT or not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"destination must be inside {ROOT}: {path}")
    return path


def _inside(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes source run")
    return result


def output_root(seed: int) -> Path:
    selected_seeds((seed,))
    return confined(ROOT / "outputs" / EXPERIMENT_ID / f"seed_{seed}")


def output_directory(seed: int, cell: Cell) -> Path:
    return output_root(seed) / f"{EXPERIMENT_ID}__{cell.condition_id}__seed_{seed}"


def original_control(binding: Binding, cell: Cell) -> Path:
    if cell.stage not in ("L0_control", "L2_ml_model") or cell.intervention != "control":
        raise ValueError("only original L0/L2 controls can be reused")
    root = binding.l0_source_root if cell.stage == "L0_control" else binding.l2_source_root
    condition = f"{cell.stage}__{cell.user_model}__sasrec__order_probe"
    return root / f"goodreads_a300_e804_v1__{condition}__seed_{binding.seed}"


def validate_control(source: Path, binding: Binding, cell: Cell,
                     audit: dict[str, Any]) -> dict[str, Any]:
    """Verify source evidence without relabelling or copying previous results."""
    source = source.resolve()
    complete = json.loads((source / "_COMPLETE.json").read_text(encoding="utf-8"))
    envelope = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest = envelope["resolved_manifest"]
    if complete.get("status") != "complete" or complete.get("completed_iteration") != 30:
        raise ValueError("control is not complete through iteration 30")
    if (source / "_FAILED.json").exists() or complete.get("run_id") != source.name or envelope.get("run_id") != source.name:
        raise ValueError("control identity or failure marker is inconsistent")
    expected = {"active_users": 300, "reserve_users": 0, "training_users": 300,
        "iterations": 30, "slate_size": 10, "simulation_seed": binding.seed,
        "population_seed": binding.population_seed, "training_scope": "active_reserve",
        "recommender_fit_interval": 1, "initial_recommender_restored": True,
        "input_format": "books", "input_fingerprint_namespace": "goodreads-books"}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"control protocol mismatch: {key}")
    if manifest.get("inputs") != audit["shared_start"]:
        raise ValueError("control does not share the frozen starting state")
    for key, value in {"condition_id": f"{cell.stage}__{cell.user_model}__sasrec__order_probe",
        "recommender": "sasrec", "feedback_loop": cell.stage, "user_model": cell.user_model,
        "user_update_policy": "frozen", "candidate_policy": "exclude_seen",
        "presentation_policy": "canonical_replica_and_shuffled_probe",
        "population_policy": "fixed_active_users", "campaign_policy": "no_campaign"}.items():
        if manifest["condition"].get(key) != value:
            raise ValueError(f"control condition mismatch: {key}")
    if manifest.get("policy_settings") or manifest.get("provider"):
        raise ValueError("control has an unexpected intervention or provider")
    for key, value in {"num_epochs": 804, "batch_size": 128, "max_sequence_length": 200}.items():
        if manifest["recommender_settings"].get(key) != value:
            raise ValueError(f"control recommender setting mismatch: {key}")
    if cell.user_model == "position_biased":
        if manifest.get("user_model_settings", {}).get("alpha") != -1.0:
            raise ValueError("control position parameter differs")
    else:
        for key, value in audit["response_artifacts"][cell.user_model].items():
            if manifest.get("response_artifact_fingerprints", {}).get(key) != value:
                raise ValueError(f"control response checkpoint mismatch: {key}")
    hashes = complete.get("file_sha256", {})
    if not {"manifest.json", "events.jsonl", "diagnostics.jsonl"}.issubset(hashes):
        raise ValueError("control completion lacks required file hashes")
    for name, expected_hash in hashes.items():
        if file_sha256(_inside(source, name)) != expected_hash:
            raise ValueError(f"control file hash mismatch: {name}")
    if file_sha256(_inside(source, complete["final_checkpoint_path"])) != complete["final_checkpoint_sha256"]:
        raise ValueError("control final checkpoint hash mismatch")
    if complete.get("row_counts", {}).get("events") != 27000:
        raise ValueError("control does not contain the expected three response arms")
    return {"source_run_directory": str(source), "source_run_id": source.name, "source_condition_id": manifest["condition"]["condition_id"],
        "completion_sha256": file_sha256(source / "_COMPLETE.json"),
        "manifest_sha256": hashes["manifest.json"], "verified": True}


def prepare_plan(seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    from .artifact_checks import audit_binding
    plan = describe(seeds)
    plan.update(status="validated_not_launched", bindings={}, entries=[], new_jobs=[])
    for seed in plan["simulation_seeds"]:
        binding = bindings()[seed]
        audit = audit_binding(binding.inputs_root, binding.labels_path, binding.population_seed, seed)
        plan["bindings"][str(seed)] = {"seed": seed, "population_seed": binding.population_seed,
            "inputs_root": str(binding.inputs_root), "labels_path": str(binding.labels_path), **audit}
        for cell in cells():
            entry = {**asdict(cell), "condition_id": cell.condition_id, "seed": seed,
                     "population_seed": binding.population_seed}
            if cell.alias_of:
                entry.update(action="alias", alias_of=cell.alias_of,
                             reason="shared initial model; no refitting at this stage")
            elif cell.intervention == "control" and cell.stage != "L1_feature":
                entry.update(action="reuse", **validate_control(original_control(binding, cell), binding, cell, audit))
            else:
                entry.update(action="run", output_directory=str(output_directory(seed, cell)),
                             output_root=str(output_root(seed)))
                plan["new_jobs"].append({"seed": seed, "condition_id": cell.condition_id,
                                         "output_directory": entry["output_directory"]})
            plan["entries"].append(entry)
    plan["validated_counts"] = dict(Counter(entry["action"] for entry in plan["entries"]))
    plan["default_max_parallel"] = 3
    plan["code_sha256"] = {str(path.relative_to(ROOT)): file_sha256(path)
        for path in sorted(ROOT.rglob("*.py")) if "tests" not in path.parts and "__pycache__" not in path.parts}
    return plan


def write_plan(destination: Path, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    destination = confined(destination)
    plan = prepare_plan(seeds)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return plan


def validate_saved_plan(path: Path, seeds: Sequence[int] | None = None) -> dict[str, Any]:
    plan = json.loads(confined(path).read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "validated_not_launched":
        raise ValueError("only this extension's validated plans are executable")
    expected = describe(plan["simulation_seeds"])
    for key in ("protocol", "intervention_strengths", "conditions", "experiment_id"):
        if plan.get(key) != expected[key]:
            raise ValueError(f"saved campaign settings changed: {key}")
    selected = selected_seeds(plan["simulation_seeds"] if seeds is None else seeds)
    if not set(selected).issubset(plan["simulation_seeds"]):
        raise ValueError("seed not present in saved plan")
    for name, digest in plan["code_sha256"].items():
        if file_sha256(confined(name)) != digest:
            raise ValueError(f"implementation changed since plan validation: {name}")
    expected_jobs = {(seed, cell.condition_id) for seed in plan["simulation_seeds"] for cell in cells()
        if not cell.alias_of and not (cell.intervention == "control" and cell.stage != "L1_feature")}
    jobs = plan["new_jobs"]
    if len(jobs) != len(expected_jobs) or {(job["seed"], job["condition_id"]) for job in jobs} != expected_jobs:
        raise ValueError("saved plan job matrix differs from the selected design")
    for seed in selected:
        for name, digest in plan["bindings"][str(seed)]["input_file_sha256"].items():
            if file_sha256(Path(name)) != digest:
                raise ValueError(f"frozen input changed since validation: {name}")
        for entry in plan["entries"]:
            if entry["seed"] == seed and entry["action"] == "reuse":
                if file_sha256(Path(entry["source_run_directory"]) / "_COMPLETE.json") != entry["completion_sha256"]:
                    raise ValueError("reused control completion changed since validation")
    return plan


def launch_one(seed: int, condition_id: str, *, device: str = "cuda", plan_path: Path | None = None) -> Any:
    selected_seeds((seed,))
    selected = {cell.condition_id: cell for cell in cells()}
    if condition_id not in selected:
        raise ValueError("condition is not part of the selected gender-mitigation matrix")
    cell = selected[condition_id]
    if cell.alias_of:
        raise ValueError(f"logical cell aliases {cell.alias_of}; do not run it twice")
    if cell.intervention == "control" and cell.stage != "L1_feature":
        raise ValueError("this control is reused from the earlier completed Goodreads experiment")
    from .artifact_checks import audit_binding
    binding = bindings()[seed]
    if plan_path is not None:
        audit = validate_saved_plan(plan_path, (seed,))["bindings"][str(seed)]
    else:
        audit = audit_binding(binding.inputs_root, binding.labels_path, binding.population_seed, seed)
    destination = output_directory(seed, cell)
    if destination.exists():
        raise FileExistsError(f"preserve the existing attempt; destination already exists: {destination}")
    from .gender_mitigation import load_gender_labels
    from .runtime_adapter import build_book_config, load_book_inputs
    from .engine import experiment_methodology as methodology
    from .engine.run_simulator import run_experiment
    inputs = load_book_inputs(binding.inputs_root / "prepared", binding.inputs_root / "population.json",
        None, binding.inputs_root / "initial_sasrec" / f"seed_{seed}")
    labels = load_gender_labels(binding.labels_path, catalogue_item_ids=inputs.dataset.catalog,
        cohort_fingerprint=audit["cohort_sha256"], histories=inputs.dataset.training_histories,
        expected_labels_sha256=audit["labels_sha256"])
    model = {"position_biased": methodology.POSITION_BIASED,
        "ncm_inspired_gru_uniform": methodology.NCM_GRU_UNIFORM,
        "ncm_inspired_gru_mansoury": methodology.NCM_GRU_MANSOURY}[cell.user_model]
    condition = methodology.build_condition(model, cell.stage,
        condition_suffix=f"gender_{cell.intervention}__order_probe",
        presentation_policy="canonical_replica_and_shuffled_probe")
    settings = {"alpha": -1.0} if cell.user_model == "position_biased" else {
        "checkpoint_directory": str(binding.inputs_root / "ncm" / f"seed_{seed}" /
            ("P3" if cell.user_model.endswith("uniform") else "P4") / "checkpoint"),
        "embedding_directory": str(binding.inputs_root / "embeddings"), "device": "cpu"}
    config = build_book_config(binding.inputs_root, output_root(seed), seed=seed,
        experiment_id=EXPERIMENT_ID, population_seed=binding.population_seed, user_model=cell.user_model,
        provider_settings={}, user_model_settings=settings,
        recommender_settings={"device": device, "num_epochs": 804, "batch_size": 128, "max_sequence_length": 200},
        mitigation={"strategy": cell.intervention, "women_item_ids": sorted(labels.women_item_ids),
            "women_weight": STRENGTHS["women_positive_weight"],
            "rank_promotion": STRENGTHS["women_rank_promotion"], "label_fingerprint": labels.labels_sha256})
    return run_experiment(config, condition, seed)


def run_campaign(plan_path: Path, *, max_parallel: int = 3, device: str = "cuda") -> dict[str, Any]:
    if isinstance(max_parallel, bool) or max_parallel not in (1, 2, 3, 4):
        raise ValueError("max_parallel must be between one and four")
    plan_path = confined(plan_path)
    plan = validate_saved_plan(plan_path)
    for job in plan["new_jobs"]:
        cell = next(cell for cell in cells() if cell.condition_id == job["condition_id"])
        if output_directory(job["seed"], cell).exists():
            raise FileExistsError("a planned output already exists; inspect it before creating a new launch plan")
    logs = confined(ROOT / "logs" / EXPERIMENT_ID)
    logs.mkdir(parents=True, exist_ok=True)
    results = []
    def execute(job: dict[str, Any]) -> dict[str, Any]:
        name = f"{job['condition_id']}__seed_{job['seed']}"
        log = logs / f"{name}.log"
        command = [sys.executable, "-B", "-u", "-X", "utf8", "-m",
            "new_dataset_experiment_mitigation.campaign", "--run-one", "--seed", str(job["seed"]),
            "--condition", job["condition_id"], "--plan", str(plan_path), "--device", device]
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=REPOSITORY, stdout=handle, stderr=subprocess.STDOUT, check=False)
        return {**job, "returncode": completed.returncode, "log": str(log)}
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        for future in as_completed([executor.submit(execute, job) for job in plan["new_jobs"]]):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    summary = {"status": "complete" if all(r["returncode"] == 0 for r in results) else "failures",
        "jobs": results, "max_parallel": max_parallel}
    with (logs / "launch_results.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true")
    action.add_argument("--write-plan", type=Path)
    action.add_argument("--run-one", action="store_true")
    action.add_argument("--run-campaign", action="store_true")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--condition")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    seeds = selected_seeds(SEEDS if args.seed is None else args.seed)
    if args.run_one:
        if len(seeds) != 1 or not args.condition:
            parser.error("--run-one needs one --seed and --condition")
        print(str(launch_one(seeds[0], args.condition, device=args.device, plan_path=args.plan)))
    elif args.run_campaign:
        if args.plan is None:
            parser.error("--run-campaign needs --plan")
        result = run_campaign(args.plan, max_parallel=args.max_parallel, device=args.device)
        return 0 if result["status"] == "complete" else 1
    elif args.write_plan:
        result = write_plan(args.write_plan, seeds)
        print(json.dumps({"plan": str(confined(args.write_plan)), "validated_counts": result["validated_counts"]}, indent=2))
    elif args.check:
        result = prepare_plan(seeds)
        print(json.dumps({"status": result["status"], "validated_counts": result["validated_counts"],
            "bindings": {seed: {key: value for key, value in row.items()
                if key in ("population_seed", "shared_start", "label_coverage")}
                for seed, row in result["bindings"].items()}}, indent=2))
    else:
        print(json.dumps(describe(seeds), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
