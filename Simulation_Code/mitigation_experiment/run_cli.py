"""Command-line parsing for the simulation runner."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import SimulatorConfig
from .experiment_methodology import CONDITIONS_BY_ID
from .utility.run_logging import log_run_attempt


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-conditions", action="store_true")
    parser.add_argument("--experiment", default="mitigation_experiment")
    parser.add_argument("--condition", choices=tuple(sorted(CONDITIONS_BY_ID)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-seed", type=int, default=1054)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--slate-size", type=int, default=10)
    parser.add_argument("--active-users", type=int, default=300)
    parser.add_argument("--reserve-users", type=int, default=0)
    parser.add_argument("--prepared-data", type=Path)
    parser.add_argument("--population", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=Path("mitigation_experiment/data/ml-1m"),
    )
    parser.add_argument(
        "--auto-input-root",
        type=Path,
        default=Path("mitigation_experiment/inputs/auto"),
    )
    parser.add_argument("--profile-max-interactions", type=int, default=100)
    parser.add_argument("--profile-batch-size", type=int, default=10)
    parser.add_argument("--initial-recommender", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--resume-from", type=Path,
        help="Saved iteration checkpoint directory; requires a new output root and the original inputs/settings",
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--provider-settings", default="{}")
    parser.add_argument("--recommender-settings", default="{}")
    parser.add_argument("--device")
    parser.add_argument("--recommender-epochs", type=int)
    parser.add_argument("--recommender-batch-size", type=int)
    parser.add_argument("--user-model-settings")
    parser.add_argument("--policy-settings")
    parser.add_argument("--recommender-fit-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--training-scope",
        choices=("all", "active-reserve"),
        default="all",
    )
    parser.add_argument("--pilot-training-users", type=int)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    arguments = parser.parse_args(argv)
    if arguments.list_conditions:
        for condition_id in sorted(CONDITIONS_BY_ID):
            print(condition_id)
        return 0
    missing = [
        flag
        for flag, value in (
            ("--condition", arguments.condition),
            ("--output-root", arguments.output_root),
        )
        if value is None
    ]
    if missing:
        parser.error("required for a run: " + ", ".join(missing))
    if arguments.resume_from is not None:
        from .run_resume import validate_resume_destination

        validate_resume_destination(arguments.resume_from, arguments.output_root)
        if arguments.prepared_data is None or arguments.population is None:
            parser.error("--resume-from requires explicit --prepared-data and --population")
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run_id = f"{arguments.experiment}__{arguments.condition}__seed_{arguments.seed}"
    with log_run_attempt(arguments.output_root, run_id):
        return _run(arguments, parser)


def _run(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Prepare inputs and execute within the attempt's logging boundary."""
    explicit_inputs = (
        arguments.prepared_data is not None or arguments.population is not None
    )
    if explicit_inputs and (
        arguments.prepared_data is None or arguments.population is None
    ):
        parser.error("provide both --prepared-data and --population, or neither")

    population_seed = (
        arguments.seed
        if arguments.population_seed is None
        else arguments.population_seed
    )
    profile_path = arguments.profiles
    if not explicit_inputs:
        from .automatic_inputs import ensure_automatic_inputs
        from .utility.providers import ProviderConfig, build_provider

        condition = CONDITIONS_BY_ID[arguments.condition]
        require_profiles = condition.user_model == "agentic_recommender"
        provider = None
        if require_profiles and profile_path is None:
            provider_values = _json_object(arguments.provider_settings)
            if "provider" in provider_values:
                parser.error("provider is controlled by --provider")
            provider = build_provider(
                ProviderConfig(provider=arguments.provider, **provider_values)
            )
        generated = ensure_automatic_inputs(
            raw_data_directory=arguments.raw_data,
            input_root=arguments.auto_input_root,
            population_seed=population_seed,
            active_user_count=arguments.active_users,
            reserve_user_count=arguments.reserve_users,
            require_profiles=require_profiles and profile_path is None,
            provider=provider,
            profile_max_interactions=arguments.profile_max_interactions,
            profile_batch_size=arguments.profile_batch_size,
        )
        arguments.prepared_data = generated.prepared_data
        arguments.population = generated.population
        if profile_path is None:
            profile_path = generated.profiles

    recommender_settings = _json_object(arguments.recommender_settings)
    _set_optional(recommender_settings, "device", arguments.device)
    _set_optional(recommender_settings, "num_epochs", arguments.recommender_epochs)
    _set_optional(
        recommender_settings,
        "batch_size",
        arguments.recommender_batch_size,
    )
    config = SimulatorConfig(
        experiment_id=arguments.experiment,
        population_seed=population_seed,
        num_iterations=arguments.iterations,
        slate_size=arguments.slate_size,
        active_user_count=arguments.active_users,
        reserve_user_count=arguments.reserve_users,
        prepared_data_path=arguments.prepared_data,
        population_artifact_path=arguments.population,
        profile_artifact_path=profile_path,
        initial_recommender_path=arguments.initial_recommender,
        output_root=arguments.output_root,
        raw_data_path=arguments.raw_data,
        resume_checkpoint_path=arguments.resume_from,
        provider_id=arguments.provider,
        provider_settings=_json_object(arguments.provider_settings),
        recommender_settings=recommender_settings,
        user_model_settings=_json_object(
            arguments.user_model_settings
            if arguments.user_model_settings is not None
            else os.getenv("MITIGATION_USER_MODEL_SETTINGS_JSON", "{}")
        ),
        policy_settings=_json_object(
            arguments.policy_settings
            if arguments.policy_settings is not None
            else os.getenv("MITIGATION_POLICY_SETTINGS_JSON", "{}")
        ),
        recommender_fit_interval=arguments.recommender_fit_interval,
        checkpoint_interval=arguments.checkpoint_interval,
        training_scope=arguments.training_scope.replace("-", "_"),
        pilot_training_user_count=arguments.pilot_training_users,
    )
    from .run_simulator import run_experiment

    artifact = run_experiment(
        config, CONDITIONS_BY_ID[arguments.condition], arguments.seed
    )
    print(artifact.output_dir)
    return 0


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("settings must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("settings must contain a JSON object")
    return value


def _set_optional(settings: dict[str, Any], name: str, value: Any) -> None:
    if value is None:
        return
    if name in settings:
        raise ValueError(f"{name} was supplied twice")
    settings[name] = value


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            return 0
        with log_run_attempt(Path(__file__).resolve().parent / "launcher_logs", "single_condition_failure"):
            raise


__all__ = ["main"]
