"""Small immutable configuration for one simulation job."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


DEFAULT_EXPERIMENT_SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class SimulatorConfig:
    """Shared inputs and operational settings for one named condition.

    Scientific mutation rights are deliberately absent: L0--L5 own those.
    The settings mappings configure concrete implementations only.
    """

    experiment_id: str
    population_seed: int
    num_iterations: int
    slate_size: int
    active_user_count: int
    reserve_user_count: int
    prepared_data_path: Path
    population_artifact_path: Path
    profile_artifact_path: Path | None
    initial_recommender_path: Path | None
    output_root: Path
    provider_id: str = "openai"
    provider_settings: Mapping[str, Any] = field(default_factory=dict)
    recommender_settings: Mapping[str, Any] = field(default_factory=dict)
    user_model_settings: Mapping[str, Any] = field(default_factory=dict)
    policy_settings: Mapping[str, Any] = field(default_factory=dict)
    recommender_fit_interval: int = 1
    checkpoint_interval: int = 1
    training_scope: str = "all"
    pilot_training_user_count: int | None = None
    raw_data_path: Path | None = None
    resume_checkpoint_path: Path | None = None
    input_format: str = "feedback_loop"

    def __post_init__(self) -> None:
        for name in ("experiment_id", "provider_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        for name in (
            "population_seed",
            "num_iterations",
            "slate_size",
            "active_user_count",
            "reserve_user_count",
            "recommender_fit_interval",
            "checkpoint_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            minimum = 0 if name in {"population_seed", "reserve_user_count"} else 1
            if value < minimum:
                relation = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be {relation}")
        if self.pilot_training_user_count is not None:
            value = self.pilot_training_user_count
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("pilot_training_user_count must be an integer or None")
            if value <= 0:
                raise ValueError("pilot_training_user_count must be positive")
        if self.training_scope not in {"all", "active_reserve"}:
            raise ValueError("training_scope must be 'all' or 'active_reserve'")
        if self.input_format not in {"feedback_loop", "mitigation"}:
            raise ValueError("input_format must be 'feedback_loop' or 'mitigation'")
        if (
            self.training_scope != "all"
            and self.pilot_training_user_count is not None
        ):
            raise ValueError(
                "training_scope and pilot_training_user_count cannot both be set"
            )
        for name in (
            "provider_settings",
            "recommender_settings",
            "user_model_settings",
            "policy_settings",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        for name in (
            "prepared_data_path",
            "population_artifact_path",
            "output_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in (
            "profile_artifact_path",
            "initial_recommender_path",
            "raw_data_path",
            "resume_checkpoint_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


__all__ = ["DEFAULT_EXPERIMENT_SEEDS", "SimulatorConfig"]
