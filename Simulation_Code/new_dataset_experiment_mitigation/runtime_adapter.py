"""Read frozen Goodreads inputs and configure isolated mitigation runs.

The earlier experiment owns preparation, embeddings, response checkpoints, and
the initial SASRec model. They are read-only inputs here. This module deliberately
provides no preparation, embedding, profile-generation, or initial-fit command.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .engine.config import SimulatorConfig
from .engine.data_preparation.books import DATASET_SCHEMA
from .engine.utility.user_handling import PopulationRole, PreparedInputs, load_prepared_inputs


ROOT = Path(__file__).resolve().parent
NCM_FOLDERS = {"pointwise_logistic": "P1", "ncm_inspired_gru_uniform": "P3", "ncm_inspired_gru_mansoury": "P4"}
SASREC_SETTINGS = {"num_epochs": 804, "batch_size": 128, "max_sequence_length": 200}
PROVIDER_SETTINGS = {"model": "gpt-5.4-nano", "reasoning_effort": "none", "max_output_tokens": 2056}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_root(path: str | Path, *, read_only_inputs: tuple[Path, ...] = ()) -> Path:
    """Resolve before writing and prohibit outputs through prior-input paths."""
    output = Path(path).resolve()
    if output == ROOT:
        raise ValueError("choose an output directory distinct from the source package")
    for source in read_only_inputs:
        source = Path(source).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("mitigation output overlaps a read-only input artifact")
    return output


def load_book_inputs(prepared_data_path: str | Path, population_artifact_path: str | Path,
                     profile_artifact_path: str | Path | None = None,
                     initial_recommender_path: str | Path | None = None) -> PreparedInputs:
    """Read and checksum the exact prepared cohort and existing initial model."""
    if initial_recommender_path is not None:
        checkpoint_root = Path(initial_recommender_path)
        complete = json.loads((checkpoint_root / "_COMPLETE.json").read_text(encoding="utf-8"))
        if complete.get("schema_version") != "goodreads-initial-sasrec-files-v1" or complete.get("status") != "complete":
            raise ValueError("shared initial recommender is not a complete Goodreads checkpoint")
        required = {"checkpoint.pt", "baseline_slates.json"}
        hashes = complete.get("files_sha256", {})
        if not required.issubset(hashes):
            raise ValueError("shared initial recommender lacks required file checksums")
        for filename, expected in hashes.items():
            if Path(filename).name != filename or file_sha256(checkpoint_root / filename) != expected:
                raise ValueError("shared initial recommender file hash mismatch")
    inputs = load_prepared_inputs(
        Path(prepared_data_path), Path(population_artifact_path),
        None if profile_artifact_path is None else Path(profile_artifact_path),
        None if initial_recommender_path is None else Path(initial_recommender_path),
    )
    if inputs.dataset.schema_version != DATASET_SCHEMA:
        raise ValueError("mitigation requires the existing Goodreads prepared inputs")
    if any(member.role is not PopulationRole.ACTIVE for member in inputs.population.members):
        raise ValueError("this experiment has no reserve or background readers")
    if len(inputs.population.members) != len(inputs.dataset.training_histories):
        raise ValueError("all and only cohort readers must be active")
    if initial_recommender_path is not None:
        baseline = json.loads((checkpoint_root / "baseline_slates.json").read_text(encoding="utf-8"))
        if complete.get("dataset_fingerprint") != inputs.dataset.dataset_fingerprint or baseline.get("dataset_fingerprint") != inputs.dataset.dataset_fingerprint:
            raise ValueError("shared initial checkpoint and prepared dataset differ")
        if baseline.get("population_fingerprint") != inputs.population.population_fingerprint or baseline.get("schema_version") != "books-initial-baseline-v1":
            raise ValueError("shared initial baseline and frozen population differ")
        slates = {int(row["user_id"]): row["item_ids"] for row in baseline["slates"]}
        if len(slates) != len(baseline["slates"]) or set(slates) != set(inputs.dataset.training_histories):
            raise ValueError("shared initial baseline does not cover every reader exactly once")
        for uid, slate in slates.items():
            if len(slate) != baseline["slate_size"] or len(slate) != len(set(slate)) or not set(slate).issubset(inputs.dataset.catalog):
                raise ValueError("shared initial baseline contains invalid candidate slates")
            if set(slate).intersection(inputs.dataset.training_histories[uid]):
                raise ValueError("shared initial baseline contains consumed books")
    return inputs


def build_book_config(inputs_root: str | Path, output_root: str | Path, *,
                      experiment_id: str = "goodreads_gender_mitigation_e804_v1",
                      population_seed: int = 1054, active_user_count: int = 300,
                      seed: int = 42, user_model: str = "position_biased",
                      mitigation: Mapping[str, Any] | None = None,
                      **overrides: Any) -> SimulatorConfig:
    """Build a job that always restores the earlier, unweighted initial model.

    ``mitigation`` accepts strategy, women_item_ids, women_weight,
    rank_promotion, and label_fingerprint. Configuration performs no training.
    """
    root = Path(inputs_root).resolve()
    user_settings: dict[str, Any] = {}
    if user_model in NCM_FOLDERS:
        user_settings.update(
            checkpoint_directory=str(root / "ncm" / f"seed_{seed}" / NCM_FOLDERS[user_model] / "checkpoint"),
            embedding_directory=str(root / "embeddings"), device="cpu",
        )
    defaults = dict(
        experiment_id=experiment_id, population_seed=population_seed,
        num_iterations=30, slate_size=10, active_user_count=active_user_count,
        reserve_user_count=0, prepared_data_path=root / "prepared",
        population_artifact_path=root / "population.json",
        profile_artifact_path=root / "profiles" if user_model == "agentic_recommender" else None,
        initial_recommender_path=root / "initial_sasrec" / f"seed_{seed}",
        output_root=Path(output_root), recommender_settings=dict(SASREC_SETTINGS),
        user_model_settings=user_settings, provider_settings=dict(PROVIDER_SETTINGS),
        training_scope="active_reserve", input_format="books",
    )
    defaults.update(overrides)
    policy = dict(defaults.get("policy_settings", {}))
    if "gender_mitigation" in policy and mitigation is not None:
        raise ValueError("configure gender mitigation once, through mitigation=")
    if mitigation is not None:
        policy["gender_mitigation"] = dict(mitigation)
    if "gender_mitigation" not in policy:
        raise ValueError("a gender mitigation specification is required, including for controls")
    defaults["policy_settings"] = policy
    config = SimulatorConfig(**defaults)
    if config.initial_recommender_path is None:
        raise ValueError("the earlier shared initial SASRec checkpoint is mandatory")
    validate_output_root(config.output_root, read_only_inputs=(root, config.prepared_data_path,
                         config.population_artifact_path, config.initial_recommender_path))
    return config


def make_runtime(config: SimulatorConfig, condition: Any, seed: int,
                 inputs: PreparedInputs, **kwargs: Any):
    """Construct a runtime without fitting; the runner owns subsequent L2 refits."""
    from .engine.runtime_setup import build_runtime
    return build_runtime(config, condition, seed, inputs, **kwargs)


__all__ = ["SASREC_SETTINGS", "PROVIDER_SETTINGS", "NCM_FOLDERS", "file_sha256",
           "load_book_inputs", "build_book_config", "make_runtime", "validate_output_root"]
