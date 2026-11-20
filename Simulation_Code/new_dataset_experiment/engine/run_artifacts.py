"""Small run-level helpers for fitting, manifests, and checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .config import SimulatorConfig
from .contracts import SimulationCheckpoint
from .experiment_methodology import ExperimentCondition
from .recommenders.base import FitSpec
from .runtime_setup import SimulationRuntime
from .utility.checkpoints import save_checkpoint
from .utility.fingerprints import fingerprint_semantic_state
from .utility.user_handling import PreparedInputs


RUN_SCHEMA = "feedback-loop-simulation-run-v1"


def input_fingerprints(
    inputs: PreparedInputs, *, input_format: str = "books",
) -> dict[str, str]:
    """Return immutable input identities used by this simulation job."""

    values = {
        "dataset": inputs.dataset.dataset_fingerprint,
        "population": inputs.population.population_fingerprint,
    }
    if inputs.recommender_checkpoint_fingerprint is not None:
        values["initial_recommender"] = (
            inputs.recommender_checkpoint_fingerprint
        )
    if inputs.profiles_by_user:
        values["profiles"] = fingerprint_semantic_state(
            {user: profile.profile_fingerprint for user, profile in sorted(inputs.profiles_by_user.items())},
            "books-frozen-profile-cohort-v1",
        )
    return values


def resolved_manifest(
    config: SimulatorConfig,
    condition: ExperimentCondition,
    seed: int,
    inputs: PreparedInputs,
    state: SimulationRuntime,
) -> dict[str, Any]:
    """Describe one resolved job once, rather than repeating it per event."""

    condition_payload = asdict(condition)
    if condition.memory_treatment is None:
        # Preserve the pre-memory manifest contract for every existing arm.
        # The treatment-only fields must not make an inactive feature look
        # like a semantic change to otherwise unchanged runs.
        for field in (
            "memory_treatment",
            "memory_decay",
            "memory_retrieval_limit",
        ):
            condition_payload.pop(field)
    if condition.fatigue_treatment is None:
        # As above, do not alter any pre-fatigue condition manifest.
        condition_payload.pop("fatigue_treatment")

    policies = {
        "candidate": state.candidate_policy.variant_id,
        "presentation": state.slate_policy.variant_id,
        "population": state.population_policy.variant_id,
        "campaign": state.campaign_policy.variant_id,
    }
    if state.order_probe_policy is not None:
        policies["order_probe"] = state.order_probe_policy.variant_id
    manifest = {
        "run_schema": RUN_SCHEMA,
        "experiment_id": config.experiment_id,
        "condition": condition_payload,
        "simulation_seed": seed,
        "population_seed": config.population_seed,
        "iterations": config.num_iterations,
        "slate_size": config.slate_size,
        "active_users": config.active_user_count,
        "reserve_users": config.reserve_user_count,
        "training_users": len(state.training_user_ids),
        "training_scope": config.training_scope,
        "recommender_fit_interval": config.recommender_fit_interval,
        "checkpoint_interval": config.checkpoint_interval,
        "initial_recommender_restored": state.initial_recommender_restored,
        "inputs": input_fingerprints(
            inputs, input_format=getattr(config, "input_format", "feedback_loop"),
        ),
        "implementations": {
            "recommender": getattr(
                state.recommender, "VARIANT_ID", type(state.recommender).__name__
            ),
            "user_model": getattr(
                state.user_model, "VARIANT_ID", type(state.user_model).__name__
            ),
            **policies,
        },
        "provider": dict(getattr(state.user_model, "provider_settings", {})),
        "response_artifact_fingerprints": {
            name: value
            for name in (
                "checkpoint_fingerprint",
                "embedding_fingerprint",
                "source_embedding_fingerprint",
            )
            if isinstance(value := getattr(state.user_model, name, None), str)
            and value.strip()
        },
        "recommender_settings": dict(config.recommender_settings),
        "user_model_settings": dict(config.user_model_settings),
        "policy_settings": dict(config.policy_settings),
    }
    manifest["input_format"] = "books"
    manifest["input_fingerprint_namespace"] = "goodreads-books"
    return manifest


def fit_recommender(state: SimulationRuntime, iteration: int) -> Mapping[str, Any]:
    """Fit on the current training histories and return its diagnostics."""

    training_fingerprint = fingerprint_semantic_state(
        {
            user_id: tuple(history)
            for user_id, history in sorted(
                state.recommender_training_histories.items()
            )
        },
        "recommender-training-histories-v1",
    )
    report = state.recommender.fit(
        state.recommender_training_histories,
        FitSpec(
            iteration=iteration,
            training_fingerprint=training_fingerprint,
            settings={
                "fit_role": (
                    "initial" if iteration == 0 else "feedback_loop_retraining"
                )
            },
        ),
    )
    return dict(report.diagnostics)


def checkpoint_iteration(
    output_dir: str | Path,
    *,
    schema_version: str,
    run_id: str,
    condition_id: str,
    seed: int,
    completed_iteration: int,
    input_fingerprints: Mapping[str, str],
    component_states: Mapping[str, Any],
    policy_states: Mapping[str, Any],
    randomness_state: Mapping[str, Any],
) -> Path:
    checkpoint = SimulationCheckpoint(
        schema_version=schema_version,
        run_id=run_id,
        condition_id=condition_id,
        seed=seed,
        completed_iteration=completed_iteration,
        input_fingerprints=input_fingerprints,
        component_states=component_states,
        policy_states=policy_states,
        randomness_state=randomness_state,
    )
    return save_checkpoint(
        Path(output_dir) / "checkpoints" / f"iteration_{completed_iteration:04d}",
        checkpoint,
    )


def save_runtime_checkpoint(
    output_dir: str | Path,
    state: SimulationRuntime,
    *,
    run_id: str,
    condition_id: str,
    seed: int,
    iteration: int,
    inputs: Mapping[str, str],
) -> Path:
    """Persist the state required to continue after a complete iteration."""

    return checkpoint_iteration(
        output_dir,
        schema_version=RUN_SCHEMA,
        run_id=run_id,
        condition_id=condition_id,
        seed=seed,
        completed_iteration=iteration,
        input_fingerprints=inputs,
        component_states={
            "recommender": state.recommender.export_state(),
            "user_model": state.user_model.export_state(),
            "observed_histories": {
                user_id: tuple(history)
                for user_id, history in sorted(state.observed_histories.items())
            },
            "recommender_training_histories": {
                user_id: tuple(history)
                for user_id, history in sorted(
                    state.recommender_training_histories.items()
                )
            },
            "training_user_ids": state.training_user_ids,
        },
        policy_states={
            "candidate": state.candidate_policy.export_state(),
            "presentation": state.slate_policy.export_state(),
            "population": state.population_policy.export_state(),
            "campaign": state.campaign_policy.export_state(),
            "order_probe": (
                None
                if state.order_probe_policy is None
                else state.order_probe_policy.export_state()
            ),
        },
        randomness_state={"simulation_seed": seed},
    )


__all__ = [
    "RUN_SCHEMA",
    "checkpoint_iteration",
    "fit_recommender",
    "input_fingerprints",
    "resolved_manifest",
    "save_runtime_checkpoint",
]
