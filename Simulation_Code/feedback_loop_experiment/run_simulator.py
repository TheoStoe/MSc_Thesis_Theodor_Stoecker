"""Run one named L0--L5 feedback-loop condition for one seed."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import SimulatorConfig
from .contracts import RunArtifact
from .experiment_methodology import CONDITIONS_BY_ID, ExperimentCondition
from .feedback_loops import FEEDBACK_LOOP_PRESETS
from .policies import SponsoredOutcome
from .run_artifacts import (
    fit_recommender,
    input_fingerprints,
    resolved_manifest,
    save_runtime_checkpoint,
)
from .run_turn import run_user_opportunity
from .run_resume import ResumeSource
from .runtime_setup import SimulationRuntime, build_runtime
from .utility.saving import RunOutputWriter
from .utility.user_handling import PreparedInputs, load_prepared_inputs


logger = logging.getLogger(__name__)


def run_experiment(
    config: SimulatorConfig,
    condition: ExperimentCondition,
    seed: int,
    *,
    prepared_inputs: PreparedInputs | None = None,
    runtime: SimulationRuntime | None = None,
) -> RunArtifact:
    """Run one complete condition and save its events and checkpoints."""

    run_id = f"{config.experiment_id}__{condition.condition_id}__seed_{seed}"
    writer = None
    iteration = 0
    stage = "setup"
    try:
        _validate_job(condition, seed)
        source = None
        if config.resume_checkpoint_path is not None:
            stage = "restart_validation"
            logger.info("Validating restart checkpoint %s", config.resume_checkpoint_path)
            source = ResumeSource.load(
                config.resume_checkpoint_path, config.output_root,
                run_id=run_id, condition_id=condition.condition_id, seed=seed,
            )
        logger.info("Loading run inputs run=%s", run_id)
        if prepared_inputs is not None:
            inputs = prepared_inputs
        else:
            input_loader = load_prepared_inputs
            if config.input_format == "mitigation":
                from .mitigation_compat import load_mitigation_inputs

                input_loader = load_mitigation_inputs
            inputs = input_loader(
                config.prepared_data_path,
                config.population_artifact_path,
                config.profile_artifact_path,
                config.initial_recommender_path,
            )
        if inputs.population.population_seed != config.population_seed:
            raise ValueError("population artifact seed disagrees with configuration")
        logger.info("Building runtime run=%s", run_id)
        if runtime is not None:
            state = runtime
        elif source is None:
            state = build_runtime(config, condition, seed, inputs)
        else:
            state = build_runtime(
                config, condition, seed, inputs,
                resumed_user_state=source.checkpoint.component_states["user_model"],
            )
        fingerprints = input_fingerprints(inputs, input_format=config.input_format)
        manifest = resolved_manifest(config, condition, seed, inputs, state)
        if source is not None:
            source.restore(state, manifest)
            iteration = source.checkpoint.completed_iteration
            manifest["resume"] = source.provenance()
        writer = RunOutputWriter.create(
            config.output_root / run_id,
            run_id=run_id,
            manifest=manifest,
        )
        final_checkpoint: Path | None = None
        stage = "initial_recommender"
        if source is not None:
            stage = "restart_copy"
            final_checkpoint = source.copy_prefix(writer)
            logger.info("Restored checkpoint after iteration %s; continuing in %s", iteration, writer.root)
        elif state.initial_recommender_restored:
            logger.info("Restored shared initial recommender checkpoint")
            writer.write_diagnostic(
                {
                    "event_name": "initial_recommender_restored",
                    "run_id": run_id,
                    "condition_id": condition.condition_id,
                    "seed": seed,
                    "iteration": 0,
                    "source_checkpoint_fingerprint": (
                        inputs.recommender_checkpoint_fingerprint
                    ),
                    "recommender_fingerprint": state.recommender.fingerprint(),
                }
            )
        else:
            _fit_and_log(state, writer, run_id, condition, seed, iteration=0)
        for iteration in range(iteration + 1, config.num_iterations + 1):
            stage = "iteration"
            logger.info("Starting iteration %s/%s", iteration, config.num_iterations)
            summary = _run_iteration(
                config, condition, seed, run_id, state, writer, iteration
            )
            if (
                iteration % config.checkpoint_interval == 0
                or iteration == config.num_iterations
            ):
                stage = "checkpoint"
                logger.info("Saving checkpoint after iteration %s", iteration)
                final_checkpoint = save_runtime_checkpoint(
                    writer.root,
                    state,
                    run_id=run_id,
                    condition_id=condition.condition_id,
                    seed=seed,
                    iteration=iteration,
                    inputs=fingerprints,
                )
                logger.info("Saved checkpoint after iteration %s", iteration)
            writer.write_diagnostic({
                "event_name": "iteration_complete",
                "iteration": iteration,
                **summary,
            })
        stage = "finalization"
        logger.info("Finalizing run %s", run_id)
        return writer.finalize(final_checkpoint_path=final_checkpoint)
    except BaseException as exc:
        logger.exception("Run failed: %s stage=%s iteration=%s", run_id, stage, iteration)
        if writer is not None:
            writer.record_failure(exc, iteration=iteration, stage=stage)
        raise


def _run_iteration(
    config: SimulatorConfig,
    condition: ExperimentCondition,
    seed: int,
    run_id: str,
    state: SimulationRuntime,
    writer: RunOutputWriter,
    iteration: int,
) -> dict[str, object]:
    preset = FEEDBACK_LOOP_PRESETS[condition.feedback_loop]
    population = state.population_policy.apply_scheduled_transitions(iteration)
    if population.exited_user_ids or population.activated_user_ids:
        logger.info(
            "Population changed: exited=%s activated=%s",
            population.exited_user_ids,
            population.activated_user_ids,
        )

    active_users = tuple(population.active_user_ids)
    eligible = {
        user_id: tuple(state.candidate_policy.eligible_items(user_id, iteration))
        for user_id in active_users
    }
    if any(len(items) < config.slate_size for items in eligible.values()):
        raise ValueError("at least one active user has too few eligible items")
    logger.info("Recommendation generation started iteration=%s users=%s", iteration, len(active_users))
    recommendations = state.recommender.recommend(
        active_users,
        config.slate_size,
        state.observed_histories,
        eligible,
    )
    if set(recommendations.items_by_user) != set(active_users):
        raise ValueError("recommender output does not cover the active users")
    logger.info("Recommendation generation completed iteration=%s", iteration)

    sponsored: list[SponsoredOutcome] = []
    for user_id in active_users:
        outcome = run_user_opportunity( # not just advertising but entire user turn
            state,
            writer,
            condition,
            run_id=run_id,
            seed=seed,
            iteration=iteration,
            user_id=user_id,
            recommender_items=recommendations.items_by_user[user_id],
        )
        if outcome is not None:
            sponsored.append(outcome)
    state.campaign_policy.observe_iteration(iteration, sponsored)

    if (
        preset.retrain_recommender
        and iteration % config.recommender_fit_interval == 0
    ):
        _fit_and_log(state, writer, run_id, condition, seed, iteration)
    return {
        "active_user_ids": active_users,
        "exited_user_ids": population.exited_user_ids,
        "activated_user_ids": population.activated_user_ids,
        "sponsored_outcomes": len(sponsored),
    }


def _fit_and_log(
    state: SimulationRuntime,
    writer: RunOutputWriter,
    run_id: str,
    condition: ExperimentCondition,
    seed: int,
    iteration: int,
) -> None:
    logger.info("Recommender fit started at iteration %s", iteration)
    diagnostics = fit_recommender(state, iteration)
    logger.info("Recommender fit completed at iteration %s", iteration)
    writer.write_diagnostic(
        {
            "event_name": "recommender_fit_completed",
            "run_id": run_id,
            "condition_id": condition.condition_id,
            "seed": seed,
            "iteration": iteration,
            "diagnostics": diagnostics,
        }
    )


def _validate_job(condition: ExperimentCondition, seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if CONDITIONS_BY_ID.get(condition.condition_id) != condition:
        raise ValueError("condition is not part of the methodology table")


def main(argv: list[str] | None = None) -> int:
    from .run_cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
