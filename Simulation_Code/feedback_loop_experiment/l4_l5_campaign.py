"""Prepare a matched canonical-only 150-active / 150-reserve L3--L5 block.

Preparation copies and validates existing artifacts; it never constructs a
provider, trains a model, or starts a simulation. The separate ``--run`` action
requires one seed and one named condition. Existing experiments remain intact.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .prepare_new_sampling import copy_verified, file_sha256


ROOT = Path(__file__).resolve().parents[1]
POPULATION_SEEDS = {42: 1054, 43: 19277, 44: 18844}
SCHEMA = "feedback-l4-l5-canonical-150-150-inputs-v1"
SETTINGS = {"device": "cuda", "num_epochs": 804, "batch_size": 128}
PROVIDER = {"model": "gpt-5.4-nano", "reasoning_effort": "none", "max_output_tokens": 2056}
L3 = "L3_individual__agentic_recommender__sasrec__full_reflection"
L4 = "L4_sampling__agentic_recommender__sasrec__full_reflection"
OUTCOME_CONDITIONS = ("O0_organic", "O1_static_ad", "O2_adaptive_ad_three_strike")


def condition_ids() -> tuple[str, ...]:
    return (L3, L4 + "__exit_only_ablation", L4 + "__uniform_refill_control",
            L4, *OUTCOME_CONDITIONS)


def _seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in POPULATION_SEEDS:
        raise ValueError("approved simulation seeds are 42, 43, and 44")
    return POPULATION_SEEDS[seed]


def campaign_name(seed: int) -> str:
    return f"l4_l5_p{_seed(seed)}_seed{seed}_a150_r150_e804_v1"


def default_input_root(seed: int) -> Path:
    return ROOT / "feedback_loop_experiment/inputs" / campaign_name(seed)


def default_output_root(seed: int) -> Path:
    return ROOT / "feedback_loop_experiment/outputs" / campaign_name(seed)


def describe_campaign(seed: int = 42) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "simulation_seed": seed, "population_seed": _seed(seed),
        "active_users": 150, "reserve_users": 150, "training_users": 300,
        "iterations": 30, "slate_size": 10, "training_scope": "active_reserve",
        "presentation_arms": ["canonical"], "conditions": list(condition_ids()),
        "condition_count": 7, "recommender_settings": dict(SETTINGS),
        "provider_settings": dict(PROVIDER), "status": "prepared_only_not_started",
        "matching": "one immutable seed-specific partition, profiles, clusters and initial model for all seven conditions",
        "outcome_matching": "O0/O1/O2 share a fresh schedule; ordinary L4 does not reserve sponsor items",
        "candidate_schedule_semantics": "exclude_seen_with_sponsor_schedule_v2: current/future sponsor items excluded from organic candidates; past unclicked sponsors become organically eligible",
        "input_root": str(default_input_root(seed)),
        "output_root": str(default_output_root(seed)),
    }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def make_sponsor_schedule(dataset: Any, population: Any, *, count: int = 30) -> dict[str, Any]:
    """Freeze popular initially unseen items, with seed-only randomized ordering.

    Popularity comes from historical training data; no simulated outcomes enter
    selection. All active and reserve histories are excluded before selection.
    """
    import numpy as np
    from .policies.sponsor_schedule import SPONSOR_SCHEDULE_SCHEMA

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("sponsor count must be a positive integer")
    users = tuple(member.user_id for member in population.members
                  if member.role.value in {"active", "reserve"})
    seen = {item for user in users for item in dataset.training_histories[user]}
    eligible = sorted(set(dataset.catalog).difference(seen),
                      key=lambda item: (-dataset.initial_item_popularity[item], item))
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} items are initially unseen by the entire selected cohort; need {count}")
    ordered = tuple(int(item) for item in np.random.default_rng(
        population.population_seed).permutation(eligible[:count]))
    return {
        "schema_version": SPONSOR_SCHEDULE_SCHEMA,
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "selection_rule": "top_30_training_popularity_after_active_reserve_history_exclusion",
        "selection_population": {"active_users": 150, "reserve_users": 150},
        "ordering_rule": "numpy_default_rng_population_seed_permutation",
        "schedules": {str(population.population_seed): list(ordered)},
    }


def _load_inputs(root: Path) -> Any:
    """Validate source namespace before assigning the independently pinned roles."""
    from .mitigation_compat import load_mitigation_inputs
    from .utility.user_handling import (
        load_population_artifact, _validate_population_against_dataset, _validate_profiles,
    )
    from .l4_l5_population import partition_population, training_history_fingerprint

    source = load_mitigation_inputs(root / "prepared_data", root / "source_population.json",
                                    root / "profiles", root / "initial_recommender")
    population = load_population_artifact(root / "population.json")
    expected, _ = partition_population(source.population, source.dataset)
    if population != expected:
        raise ValueError("population differs from the reproducible frozen 150/150 partition")
    _validate_population_against_dataset(population, source.dataset)
    _validate_profiles(source.profiles_by_user, population, source.dataset)
    if training_history_fingerprint(population, source.dataset) != training_history_fingerprint(
            source.population, source.dataset):
        raise ValueError("partition changed the original 300 training histories")
    return replace(source, population=population)


def _fingerprints(inputs: Any) -> dict[str, str]:
    from .utility.fingerprints import fingerprint_semantic_state
    from .l4_l5_population import training_history_fingerprint
    return {
        "dataset": inputs.dataset.dataset_fingerprint,
        "population": inputs.population.population_fingerprint,
        "initial_recommender": inputs.recommender_checkpoint_fingerprint,
        "profiles": fingerprint_semantic_state({user: profile.profile_fingerprint
            for user, profile in sorted(inputs.profiles_by_user.items())},
            "mitigation-frozen-profile-cohort-v1"),
        "training_histories": training_history_fingerprint(inputs.population, inputs.dataset),
    }


def prepare_campaign(seed: int = 42, *, source_manifest: Path | None = None,
                     destination: Path | None = None) -> Path:
    """Copy inputs into a fresh folder and freeze one shared partition/schedule."""
    from .new_sampling_campaign import read_inputs_manifest, input_identity
    from .mitigation_compat import load_mitigation_inputs
    from .prepare_new_sampling import validate_initial
    from .run_artifacts import input_fingerprints
    from .l4_l5_population import partition_population, demographics_from_profile_inputs
    from .utility.saving import save_population_artifact
    from mitigation_experiment.data_preparation.clustering import load_cluster_artifact

    population_seed = _seed(seed)
    source_manifest = Path(source_manifest or ROOT / "feedback_loop_experiment/inputs" /
        f"new_sampling_p{population_seed}_seed{seed}_a300_r0_e804_v1/inputs_manifest.json").resolve()
    source = read_inputs_manifest(source_manifest)
    if input_identity(source) != (seed, population_seed):
        raise ValueError("source simulation/population seed pair differs")
    destination = Path(destination or default_input_root(seed)).resolve()
    staging = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or staging.exists():
        raise FileExistsError(f"preserve existing preparations: {destination}")
    source_paths = {key: Path(source["paths"][key]) for key in (
        "prepared_data", "profiles", "initial_recommender")}
    source_paths["source_population.json"] = Path(source["paths"]["population"])
    source_paths["clusters"] = Path(source["file_inventory"]["clusters"]["destination"])
    native = load_mitigation_inputs(source_paths["prepared_data"],
        source_paths["source_population.json"], source_paths["profiles"], source_paths["initial_recommender"])
    if input_fingerprints(native, input_format="mitigation") != source["fingerprints"]:
        raise ValueError("native source artifacts differ from their recorded input fingerprints")
    validate_initial(native, seed)
    clusters = load_cluster_artifact(source_paths["clusters"])
    if (clusters.artifact_fingerprint != native.population.cluster_artifact_fingerprint
            or clusters.dataset_fingerprint != native.dataset.dataset_fingerprint
            or any(clusters.cluster_by_user.get(member.user_id) != member.cluster_id
                   for member in native.population.members)):
        raise ValueError("frozen cluster artifact differs from original population")
    population, report = partition_population(native.population, native.dataset,
        demographics_by_source=demographics_from_profile_inputs(source_paths["profiles"]))
    schedule = make_sponsor_schedule(native.dataset, population)
    staging.mkdir(parents=True)
    inventory = {}
    for name, path in source_paths.items():
        inventory[name] = {"source": str(path.resolve()),
            "file_sha256": copy_verified(path, staging / name)}
    copy_verified(source_manifest, staging / "source_inputs_manifest.json")
    save_population_artifact(staging / "population.json", population)
    _write(staging / "partition_report.json", report)
    _write(staging / "sponsor_schedule.json", schedule)
    copied = _load_inputs(staging)
    if _fingerprints(copied) != _fingerprints(replace(native, population=population)):
        raise ValueError("copied inputs changed scientific identities")
    manifest = {**describe_campaign(seed), "input_root": str(destination),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "fingerprints": _fingerprints(copied),
        "source_population_fingerprint": native.population.population_fingerprint,
        "source_artifact_namespaces": {"dataset": "mitigation_experiment", "profiles": "mitigation_experiment",
            "initial_recommender": "mitigation_experiment", "population": "feedback_loop_experiment"},
        "profile_reuse": "original bytes and individual fingerprints unchanged; only active/reserve roles reassigned",
        "file_inventory": inventory,
        "file_sha256": {path.relative_to(staging).as_posix(): file_sha256(path)
            for path in sorted(staging.rglob("*")) if path.is_file()},
    }
    _write(staging / "inputs_manifest.json", manifest)
    staging.rename(destination)
    return destination / "inputs_manifest.json"


def validate_inputs(manifest_path: Path) -> dict[str, Any]:
    """Verify immutable files and model compatibility, without providers or fits."""
    from .prepare_new_sampling import validate_initial
    from .policies.sponsor_schedule import load_sponsor_schedule

    manifest_path = Path(manifest_path).resolve()
    manifest = _read(manifest_path)
    seed = manifest["simulation_seed"]
    expected = describe_campaign(seed)
    for key in ("schema", "population_seed", "active_users", "reserve_users", "training_users",
                "iterations", "slate_size", "training_scope", "presentation_arms", "conditions",
                "condition_count", "recommender_settings", "provider_settings"):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"campaign {key} differs from the approved design")
    root = manifest_path.parent
    if Path(manifest["input_root"]).resolve() != root:
        raise ValueError("manifest input root differs from its actual location")
    hashes = manifest.get("file_sha256", {})
    required = {"population.json", "partition_report.json", "source_population.json",
                "source_inputs_manifest.json", "sponsor_schedule.json"}
    if not required.issubset(hashes):
        raise ValueError("manifest does not pin all prepared artifacts")
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*")
                    if path.is_file() and path.name != "inputs_manifest.json"}
    if set(hashes) != actual_files:
        raise ValueError("prepared artifact inventory changed")
    for relative, digest in hashes.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"prepared artifact checksum differs: {relative}")
    inputs = _load_inputs(root)
    if inputs.population.population_seed != manifest["population_seed"]:
        raise ValueError("prepared population seed differs")
    if _fingerprints(inputs) != manifest["fingerprints"]:
        raise ValueError("prepared semantic fingerprints differ")
    validate_initial(inputs, seed)
    schedule = load_sponsor_schedule(root / "sponsor_schedule.json",
        population_seed=manifest["population_seed"],
        expected_dataset_fingerprint=inputs.dataset.dataset_fingerprint,
        catalog_item_ids=tuple(sorted(inputs.dataset.catalog)))
    expected_schedule = make_sponsor_schedule(inputs.dataset, inputs.population)
    if _read(root / "sponsor_schedule.json") != expected_schedule or len(schedule.item_ids) != 30:
        raise ValueError("sponsor schedule differs from the frozen thirty-item selection")
    return manifest


def build_config(manifest_path: Path, output_root: Path, condition_id: str) -> Any:
    from .config import SimulatorConfig
    if condition_id not in condition_ids():
        raise ValueError("condition is outside the seven canonical-only conditions")
    manifest = validate_inputs(manifest_path)
    root = Path(manifest_path).resolve().parent
    policy_settings = {
        # Provenance-only settings are preserved in each resolved run manifest.
        # The runtime policies read only their own named behavioural settings.
        "campaign_inputs_manifest_path": str(Path(manifest_path).resolve()),
        "campaign_inputs_manifest_sha256": file_sha256(Path(manifest_path)),
        "frozen_profile_cohort_fingerprint": manifest["fingerprints"]["profiles"],
        "common_partition_fingerprint": manifest["fingerprints"]["population"],
    }
    if condition_id in OUTCOME_CONDITIONS:
        policy_settings["sponsor_schedule_path"] = str(root / "sponsor_schedule.json")
    return SimulatorConfig(
        experiment_id=campaign_name(manifest["simulation_seed"]),
        population_seed=manifest["population_seed"], num_iterations=30, slate_size=10,
        active_user_count=150, reserve_user_count=150,
        prepared_data_path=root / "prepared_data", population_artifact_path=root / "population.json",
        profile_artifact_path=root / "profiles", initial_recommender_path=root / "initial_recommender",
        output_root=Path(output_root), training_scope="active_reserve", input_format="feedback_loop",
        recommender_settings=dict(SETTINGS), provider_id="openai", provider_settings=dict(PROVIDER),
        user_model_settings={}, policy_settings=policy_settings,
    )


def run_condition(input_manifest: Path, output_root: Path, condition_id: str) -> Any:
    """Explicit paid launch for exactly one condition; preparation never calls it."""
    from .experiment_methodology import CONDITIONS_BY_ID
    from .run_simulator import run_experiment

    config = build_config(input_manifest, output_root, condition_id)
    manifest = _read(Path(input_manifest))
    return run_experiment(config, CONDITIONS_BY_ID[condition_id], manifest["simulation_seed"],
                          prepared_inputs=_load_inputs(Path(input_manifest).resolve().parent))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", choices=condition_ids())
    parser.add_argument("--seed", type=int, choices=tuple(POPULATION_SEEDS), action="append")
    parser.add_argument("--inputs-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args(argv)
    seeds = arguments.seed or list(POPULATION_SEEDS)
    if arguments.run:
        if len(seeds) != 1 or not arguments.seed:
            parser.error("--run requires exactly one explicit --seed")
        manifest_path = arguments.inputs_manifest or default_input_root(seeds[0]) / "inputs_manifest.json"
        if _read(manifest_path)["simulation_seed"] != seeds[0]:
            parser.error("manifest seed differs from explicit --seed")
        run_condition(manifest_path, arguments.output_root or default_output_root(seeds[0]), arguments.run)
    elif arguments.inputs_manifest is not None and not arguments.check:
        parser.error("--inputs-manifest is supported only with --check or --run")
    elif arguments.check:
        paths = ([arguments.inputs_manifest] if arguments.inputs_manifest else
                 [default_input_root(seed) / "inputs_manifest.json" for seed in seeds])
        print(json.dumps([validate_inputs(path) for path in paths], indent=2))
    elif arguments.prepare:
        print(json.dumps([str(prepare_campaign(seed)) for seed in seeds], indent=2))
    else:
        print(json.dumps([describe_campaign(seed) for seed in seeds], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
