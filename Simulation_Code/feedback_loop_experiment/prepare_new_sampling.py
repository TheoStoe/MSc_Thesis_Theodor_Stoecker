"""Prepare immutable frozen-cohort inputs without simulation or API calls.

The copied files retain their original bytes and scientific fingerprints.
``mitigation_compat`` validates them natively before adapting runtime contracts.
An explicit flag permits the missing seed-44 initial SASRec fit. This command
never starts a simulation or overwrites an earlier preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "mitigation_experiment"
DEFAULT_DESTINATION = (
    ROOT / "feedback_loop_experiment/inputs/new_sampling_p1054_seed42_a300_r0_e804_v1"
)
INITIAL_KEY = "330c57014778d76f7644e9e3dc77be48620eaf912f75637bf28d4f03b4eae5a2"
SOURCE_QUEUE = (
    SOURCE / "outputs/sasrec_budget_p1054_seed42_a300_r0_e804_b128_llm_split4_v1"
    / "queue_1_agent_control/agentic_recommender__control"
)
SOURCE_RUN = SOURCE_QUEUE / (
    "mitigation_v1_primary_p1054_a300_r0__agentic_recommender__control__seed_42"
)
POPULATION_SEEDS = {42: 1054, 43: 19277, 44: 18844}
TRAINING_SETTINGS = {"device": "cuda", "num_epochs": 804, "batch_size": 128}
HEAD_DEFINITION = "top_ceil_20pct_catalog_by_full_training_count_id_ties_v1"


def default_destination(seed: int) -> Path:
    if seed not in POPULATION_SEEDS:
        raise ValueError("The approved simulation seeds are 42, 43, and 44")
    return ROOT / ("feedback_loop_experiment/inputs/"
                   f"new_sampling_p{POPULATION_SEEDS[seed]}_seed{seed}_a300_r0_e804_v1")


def source_control(seed: int) -> tuple[Path, Path] | None:
    """Only completed 804-epoch source controls may supply reusable lineage."""
    if seed == 42:
        return SOURCE_QUEUE, SOURCE_RUN
    if seed == 43:
        queue = (SOURCE / "outputs/sasrec_budget_p19277_seed43_a300_r0_e804_b128_split4_v1"
                 / "queue_1_agent_control/agentic_recommender__control")
        return queue, queue / "mitigation_v1_primary_p19277_a300_r0__agentic_recommender__control__seed_43"
    if seed == 44:
        return None
    raise ValueError("The approved simulation seeds are 42, 43, and 44")


def head_reference(dataset: Any) -> dict[str, Any]:
    from mitigation_experiment.utility.fingerprints import fingerprint_semantic_state

    catalog = tuple(sorted(dataset.catalog))
    head = tuple(sorted(sorted(catalog, key=lambda item: (
        -dataset.initial_item_popularity[item], item,
    ))[:math.ceil(0.2 * len(catalog))]))
    return {
        "head_definition": HEAD_DEFINITION,
        "head_definition_fingerprint": fingerprint_semantic_state(
            {"dataset": dataset.dataset_fingerprint, "head_item_ids": head,
             "definition": HEAD_DEFINITION}, "mitigation-prior-head-v1",
        ),
        "head_count": len(head), "catalog_item_count": len(catalog),
    }


def validate_initial(inputs: Any, simulation_seed: int) -> None:
    """Verify every model setting and the exact 300 historical training logs."""
    from mitigation_experiment.recommenders.sasrec import SASRecAdapter
    from mitigation_experiment.runtime_setup import _restore_initial_recommender
    from mitigation_experiment.utility.checkpoints import load_checkpoint

    checkpoint = load_checkpoint(inputs.recommender_checkpoint_path)
    if checkpoint.seed != simulation_seed:
        raise ValueError("Initial checkpoint simulation seed differs from the requested seed")
    users = tuple(member.user_id for member in inputs.population.members
                  if member.role.value in {"active", "reserve"})
    histories = {user: tuple(inputs.dataset.training_histories[user]) for user in users}
    if len(histories) != 300:
        raise ValueError("Initial checkpoint requires exactly 300 frozen-cohort training users")
    # Device is operational, so a CPU restore verifies the immutable state
    # without starting a fit or contending with the one authorized GPU fit.
    recommender = SASRecAdapter(
        **{**TRAINING_SETTINGS, "device": "cpu"},
        catalog_item_ids=tuple(sorted(inputs.dataset.catalog)), random_seed=simulation_seed,
    )
    _restore_initial_recommender(
        recommender=recommender, checkpoint_path=inputs.recommender_checkpoint_path,
        dataset_fingerprint=inputs.dataset.dataset_fingerprint, training_histories=histories,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path) -> dict[str, str]:
    """Copy a file/tree exclusively, checking every file against its source."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Preserve the existing destination: {destination}")
    if source.is_symlink():
        raise ValueError(f"Symbolic links are not input artifacts: {source}")
    sources = [source] if source.is_file() else sorted(
        path for path in source.rglob("*") if path.is_file()
    )
    if not sources:
        raise ValueError(f"Missing or empty input artifact: {source}")
    expected = {}
    for path in sources:
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not input artifacts: {path}")
        relative = path.name if source.is_file() else path.relative_to(source).as_posix()
        expected[relative] = file_sha256(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        shutil.copy2(source, destination)
    else:
        shutil.copytree(source, destination)
    for relative, digest in expected.items():
        copied = destination if source.is_file() else destination / relative
        original = source if source.is_file() else source / relative
        if file_sha256(copied) != digest or file_sha256(original) != digest:
            raise ValueError(f"Input changed or copy verification failed: {original}")
    return expected


def _save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def prepare_inputs(
    simulation_seed: int = 42, destination: Path | None = None, *,
    prepare_missing_initial: bool = False,
) -> Path:
    """Copy frozen inputs, optionally fitting only the missing seed-44 initial.

    Source controls are required for seeds 42/43. Seed 44 intentionally has no
    completed 804-epoch source campaign; its profile and model artifacts are
    validated directly, with no provider construction or population sampling.
    """
    from .mitigation_compat import load_mitigation_inputs, _validate_frozen_selection
    from .run_artifacts import input_fingerprints
    from mitigation_experiment.data_preparation.clustering import load_cluster_artifact
    from mitigation_experiment.prepare_ncm_artifacts import ncm_settings
    from mitigation_experiment.utility.user_handling import load_prepared_inputs, PopulationRole

    destination = Path(destination or default_destination(simulation_seed)).resolve()
    staging = destination.with_name(destination.name + ".incomplete")
    for path in (destination, staging):
        if path.exists():
            raise FileExistsError(f"Preserve prior preparation and choose a fresh path: {path}")
    base = SOURCE / "inputs/mitigation_v1"
    population_seed = POPULATION_SEEDS[simulation_seed]
    lineage = source_control(simulation_seed)
    campaign = resolved = None
    if lineage is not None:
        queue, source_run = lineage
        campaign = json.loads((queue / "campaign_manifest.json").read_text(encoding="utf-8"))
        resolved = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))["resolved_manifest"]
        if campaign["seed"] != simulation_seed or campaign["population_seed"] != population_seed:
            raise ValueError("Source campaign seeds do not match the requested frozen cohort")
        if campaign["recommender_settings"] != TRAINING_SETTINGS:
            raise ValueError("Source campaign does not have the agreed 804-epoch training settings")
        if not (source_run / "_COMPLETE.json").is_file():
            raise ValueError("Reusable source control has not completed")
    source_paths = {
        "prepared_data": base / "ml1m",
        "clusters": base / "ml1m_clusters",
        "population": base / f"population__seed_{population_seed}__active_300__reserve_0.json",
        "profiles": base / f"profiles__population_{population_seed}__selected_300",
        "ncm_p3": SOURCE / f"data/ncm_checkpoints/seed_{simulation_seed}/P3/checkpoint",
        "ncm_p4": SOURCE / f"data/ncm_checkpoints/seed_{simulation_seed}/P4/checkpoint",
        "ncm_embeddings": SOURCE / "data/text_embedding_3_small_movielens_v1",
    }
    uninitialized = load_prepared_inputs(
        source_paths["prepared_data"], source_paths["population"], source_paths["profiles"], None,
    )
    _validate_frozen_selection(uninitialized.population)
    if len(uninitialized.profiles_by_user) != 300:
        raise ValueError("Frozen profile bundle must cover exactly the 300 selected users")
    clusters = load_cluster_artifact(source_paths["clusters"])
    if (clusters.dataset_fingerprint != uninitialized.dataset.dataset_fingerprint
            or clusters.artifact_fingerprint != uninitialized.population.cluster_artifact_fingerprint
            or any(clusters.cluster_by_user.get(member.user_id) != member.cluster_id
                   for member in uninitialized.population.members)):
        raise ValueError("Frozen clusters do not match the population and dataset")
    validated_ncm = {label: ncm_settings(
        uninitialized.dataset, model_folder=label,
        checkpoint_root=SOURCE / "data/ncm_checkpoints",
        embedding_directory=source_paths["ncm_embeddings"], seed=simulation_seed,
    ) for label in ("P3", "P4")}
    if campaign is not None:
        if validated_ncm != campaign["ncm_settings"]:
            raise ValueError("Source campaign NCM settings differ from validated frozen artifacts")
        source_paths["initial_recommender"] = Path(campaign["initial_recommender"])
    else:
        if not prepare_missing_initial:
            raise ValueError("Seed 44 needs --prepare-missing-initial for its one local 804-epoch SASRec fit")
        from mitigation_experiment.initial_recommender import ensure_initial_recommender
        training_users = tuple(member.user_id for member in uninitialized.population.members
                               if member.role in {PopulationRole.ACTIVE, PopulationRole.RESERVE})
        source_paths["initial_recommender"] = ensure_initial_recommender(
            dataset=uninitialized.dataset, training_user_ids=training_users,
            seed=simulation_seed, settings=dict(TRAINING_SETTINGS),
            checkpoint_root=base / "initial_sasrec",
        )
    names = {key: key for key in source_paths}
    names["population"] = "population.json"
    paths = {key: destination / name for key, name in names.items()}
    source_inputs = load_mitigation_inputs(
        source_paths["prepared_data"], source_paths["population"],
        source_paths["profiles"], source_paths["initial_recommender"],
    )
    validate_initial(source_inputs, simulation_seed)
    fingerprints = input_fingerprints(source_inputs, input_format="mitigation")
    if resolved is not None and resolved["inputs"] != fingerprints:
        raise ValueError("The source run and reusable inputs do not share the same fingerprints")
    reference = head_reference(source_inputs.dataset)
    if (resolved is not None and resolved["mitigation"]["head_definition_fingerprint"]
            != reference["head_definition_fingerprint"]):
        raise ValueError("Source run head definition differs from the full historical training reference")

    staging.mkdir(parents=True)
    inventory = {}
    for key, source in source_paths.items():
        inventory[key] = {
            "source": str(source.resolve()), "destination": str(paths[key]),
            "file_sha256": copy_verified(source, staging / names[key]),
        }
    copied = load_mitigation_inputs(
        staging / names["prepared_data"], staging / names["population"],
        staging / names["profiles"], staging / names["initial_recommender"],
    )
    if input_fingerprints(copied, input_format="mitigation") != fingerprints:
        raise ValueError("Copied inputs changed their scientific fingerprints")
    model_settings = {}
    for family, label, copied_key in (
        ("ncm_inspired_gru_uniform", "P3", "ncm_p3"),
        ("ncm_inspired_gru_mansoury", "P4", "ncm_p4"),
    ):
        settings = dict(validated_ncm[label])
        settings["checkpoint_directory"] = str(paths[copied_key])
        settings["embedding_directory"] = str(paths["ncm_embeddings"])
        model_settings[family] = settings
    model_settings.update({family: {} for family in (
        "position_biased", "direct_llm", "agentic_recommender",
    )})
    file_hashes = {}
    for key, entry in inventory.items():
        if source_paths[key].is_file():
            file_hashes[names[key]] = next(iter(entry["file_sha256"].values()))
        else:
            file_hashes.update({
                f"{names[key]}/{relative}": digest
                for relative, digest in entry["file_sha256"].items()
            })
    manifest = {
        "schema": "feedback-new-sampling-inputs-v1",
        "simulation_seed": simulation_seed, "population_seed": population_seed,
        "active_users": 300, "reserve_users": 0, "iterations": 30,
        "training_scope": "active_reserve", "training_users": 300,
        "input_format": "mitigation",
        "paths": {key: str(paths[key]) for key in (
            "prepared_data", "population", "profiles", "initial_recommender",
        )},
        "fingerprints": fingerprints,
        "user_model_settings": model_settings,
        "provider_settings": {
            "model": "gpt-5.4-nano", "reasoning_effort": "none", "max_output_tokens": 2056,
        },
        "recommender_settings": dict(TRAINING_SETTINGS),
        "reference": reference,
        "source_campaign_manifest": None if lineage is None else str((lineage[0] / "campaign_manifest.json").resolve()),
        "source_campaign_manifest_sha256": None if lineage is None else file_sha256(lineage[0] / "campaign_manifest.json"),
        "preparation_validation": {
            "frozen_membership": "exact_ordered_300_union",
            "profile_count": len(copied.profiles_by_user),
            "ncm": "validated_seed_specific_full_population_frozen_checkpoints",
            "initial": "unweighted_iteration_zero_804_epochs_batch128",
            "completed_804_source_control": lineage is not None,
        },
        "file_inventory": inventory,
        "file_sha256": file_hashes,
        "status": "prepared_only_no_campaign_started",
    }
    _save_json(staging / "inputs_manifest.json", manifest)
    staging.rename(destination)
    return destination / "inputs_manifest.json"


def prepare_seed42_inputs(destination: Path = DEFAULT_DESTINATION) -> Path:
    """Backward-compatible first-seed entry point; never performs a fit."""
    return prepare_inputs(42, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=tuple(POPULATION_SEEDS), default=42)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--prepare-missing-initial", action="store_true",
                        help="Allow the one missing seed-44 local SASRec fit; never uses an API")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = prepare_inputs(args.seed, args.destination, prepare_missing_initial=args.prepare_missing_initial)
    print(f"Prepared and verified inputs only; no simulation started: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
