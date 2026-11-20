"""Prepared input and frozen-profile loading definitions."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..contracts import ItemId, UserId
from ..persona_extraction.base import ProfileArtifact
from .fingerprints import fingerprint_semantic_state


class PopulationRole(str, Enum):
    ACTIVE = "active"
    RESERVE = "reserve"
    BACKGROUND = "background"


@dataclass(frozen=True)
class PopulationMember:
    user_id: UserId
    source_user_id: str
    role: PopulationRole
    draw_position: int
    reserve_order: int | None
    cluster_id: int
    profile_fingerprint: str | None


@dataclass(frozen=True)
class PopulationArtifact:
    schema_version: str
    population_seed: int
    source_population_fingerprint: str
    cluster_artifact_fingerprint: str
    population_fingerprint: str
    members: tuple[PopulationMember, ...]


@dataclass(frozen=True)
class PreparedTrainingInteraction:
    """One ordered human training event retained for profile extraction."""

    item_id: ItemId
    source_item_id: str
    rating: int
    timestamp: int
    source_row: int

    def __post_init__(self) -> None:
        for name in ("item_id", "rating", "timestamp", "source_row"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.item_id <= 0 or self.timestamp <= 0:
            raise ValueError("item_id and timestamp must be positive")
        if not 1 <= self.rating <= 5:
            raise ValueError("rating must lie between 1 and 5")
        if self.source_row < 0:
            raise ValueError("source_row cannot be negative")
        if (
            not isinstance(self.source_item_id, str)
            or not self.source_item_id.isdecimal()
            or str(int(self.source_item_id)) != self.source_item_id
        ):
            raise ValueError("source_item_id must be a canonical decimal string")


@dataclass(frozen=True)
class PreparedDatasetArtifact:
    """Loadable, split-safe dataset bundle produced before simulation."""

    schema_version: str
    preparation_protocol: str
    preparation_settings: Mapping[str, Any]
    preparation_statistics: Mapping[str, Any]
    dataset_fingerprint: str
    id_mapping_fingerprint: str
    catalog_fingerprint: str
    item_text_fingerprint: str
    source_file_fingerprints: Mapping[str, str]
    catalog: Mapping[ItemId, Mapping[str, Any]]
    training_histories: Mapping[UserId, tuple[ItemId, ...]]
    training_interactions: Mapping[
        UserId, tuple[PreparedTrainingInteraction, ...]
    ]
    training_history_fingerprints: Mapping[UserId, str]
    validation_targets: Mapping[UserId, ItemId]
    test_targets: Mapping[UserId, ItemId]
    runtime_to_source_user_id: Mapping[UserId, str]
    runtime_to_source_item_id: Mapping[ItemId, str]
    initial_item_popularity: Mapping[ItemId, int]


@dataclass(frozen=True)
class PreparedInputs:
    dataset: PreparedDatasetArtifact
    population: PopulationArtifact
    recommender_checkpoint_fingerprint: str | None
    recommender_checkpoint_path: Path | None
    profiles_by_user: Mapping[UserId, ProfileArtifact]


def load_prepared_inputs(
    prepared_data_path: Path,
    population_artifact_path: Path,
    profile_artifact_path: Path | None,
    initial_recommender_path: Path | None,
) -> PreparedInputs:
    """Load and cross-check all inputs required by one job.
    """

    from ..data_preparation.artifacts import load_prepared_dataset

    dataset = load_prepared_dataset(prepared_data_path)
    population = load_population_artifact(population_artifact_path)
    _validate_population_against_dataset(population, dataset)
    profiles = (
        {}
        if profile_artifact_path is None
        else load_profile_artifacts(profile_artifact_path)
    )
    if profile_artifact_path is not None:
        _validate_profiles(profiles, population, dataset)
    recommender_path = (
        None if initial_recommender_path is None else Path(initial_recommender_path)
    )
    recommender_fingerprint = (
        None if recommender_path is None else _path_fingerprint(recommender_path)
    )
    return PreparedInputs(
        dataset=dataset,
        population=population,
        recommender_checkpoint_fingerprint=recommender_fingerprint,
        recommender_checkpoint_path=recommender_path,
        profiles_by_user=profiles,
    )


def load_population_artifact(path: str | Path) -> PopulationArtifact:
    payload = _read_json(Path(path))
    if payload.get("file_schema") != "feedback-loop-population-file-v1":
        raise ValueError("unsupported population file schema")
    raw = payload.get("population")
    if not isinstance(raw, Mapping):
        raise ValueError("population file does not contain a population object")
    members = tuple(
        PopulationMember(
            user_id=int(member["user_id"]),
            source_user_id=str(member["source_user_id"]),
            role=PopulationRole(str(member["role"])),
            draw_position=int(member["draw_position"]),
            reserve_order=(
                None
                if member.get("reserve_order") is None
                else int(member["reserve_order"])
            ),
            cluster_id=int(member["cluster_id"]),
            profile_fingerprint=member.get("profile_fingerprint"),
        )
        for member in raw["members"]
    )
    artifact = PopulationArtifact(
        schema_version=str(raw["schema_version"]),
        population_seed=int(raw["population_seed"]),
        source_population_fingerprint=str(raw["source_population_fingerprint"]),
        cluster_artifact_fingerprint=str(raw["cluster_artifact_fingerprint"]),
        population_fingerprint=str(raw["population_fingerprint"]),
        members=members,
    )
    fingerprint_payload = {
        field.name: getattr(artifact, field.name)
        for field in fields(artifact)
        if field.name != "population_fingerprint"
    }
    if artifact.population_fingerprint != fingerprint_semantic_state(
        fingerprint_payload, artifact.schema_version
    ):
        raise ValueError("population artifact fingerprint mismatch")
    return artifact


def load_profile_artifacts(path: str | Path) -> Mapping[UserId, ProfileArtifact]:
    root = Path(path)
    profiles_path = root / "profiles.jsonl"
    complete_path = root / "_COMPLETE.json"
    if not profiles_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError("profile bundle is missing profiles.jsonl or _COMPLETE")
    complete = _read_json(complete_path)
    if complete.get("status") != "complete":
        raise ValueError("profile bundle is not marked complete")
    expected_hash = complete.get("file_sha256", {}).get("profiles.jsonl")
    if expected_hash and _file_sha256(profiles_path) != expected_hash:
        raise ValueError("profile bundle profiles.jsonl hash mismatch")
    profiles: dict[UserId, ProfileArtifact] = {}
    for row_number, line in enumerate(
        profiles_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"profiles.jsonl row {row_number} is blank")
        raw = json.loads(line)
        if raw.get("schema_version") != "goodreads-profile-artifact-v1":
            raise ValueError("book runs require fresh book-domain profile artifacts")
        profile = ProfileArtifact(
            schema_version=str(raw["schema_version"]),
            user_id=int(raw["user_id"]),
            source_user_id=str(raw["source_user_id"]),
            extractor_id=str(raw["extractor_id"]),
            extraction_seed=int(raw["extraction_seed"]),
            profile_text=str(raw["profile_text"]),
            intermediate_profiles=tuple(raw["intermediate_profiles"]),
            update_summaries=tuple(raw["update_summaries"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            training_history_fingerprint=str(raw["training_history_fingerprint"]),
            user_text_builder_version=raw.get("user_text_builder_version"),
            user_text_fingerprint=raw.get("user_text_fingerprint"),
            input_fingerprint=str(raw["input_fingerprint"]),
            prompt_fingerprint=str(raw["prompt_fingerprint"]),
            method_settings=dict(raw["method_settings"]),
            provider_settings=dict(raw["provider_settings"]),
            profile_fingerprint=str(raw["profile_fingerprint"]),
        )
        if profile.user_id in profiles:
            raise ValueError(f"profile bundle contains duplicate user {profile.user_id}")
        profile_payload = {
            field.name: getattr(profile, field.name)
            for field in fields(profile)
            if field.name != "profile_fingerprint"
        }
        if profile.profile_fingerprint != fingerprint_semantic_state(
            profile_payload, profile.schema_version
        ):
            raise ValueError(f"profile fingerprint mismatch for user {profile.user_id}")
        profiles[profile.user_id] = profile
    return profiles


def load_profile_user_texts(
    path: str | Path,
    user_ids: set[UserId],
) -> Mapping[UserId, str]:
    """Load frozen training-only reader preference text for selected profiles."""

    root = Path(path)
    inputs_path = root / "inputs.jsonl"
    complete_path = root / "_COMPLETE.json"
    if not inputs_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError("profile bundle is missing inputs.jsonl or _COMPLETE")
    complete = _read_json(complete_path)
    expected_hash = complete.get("file_sha256", {}).get("inputs.jsonl")
    if expected_hash and _file_sha256(inputs_path) != expected_hash:
        raise ValueError("profile bundle inputs.jsonl hash mismatch")

    selected: dict[UserId, str] = {}
    for line in inputs_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        user_id = int(row["user_id"])
        if user_id not in user_ids:
            continue
        text = row.get("user_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"profile input has no user text for user {user_id}")
        selected[user_id] = text.strip()
    missing = sorted(user_ids.difference(selected))
    if missing:
        raise ValueError(f"profile inputs are missing users: {missing}")
    return selected


def _validate_population_against_dataset(
    population: PopulationArtifact,
    dataset: PreparedDatasetArtifact,
) -> None:
    if population.source_population_fingerprint != dataset.dataset_fingerprint:
        raise ValueError("population and prepared dataset fingerprints differ")
    if not population.cluster_artifact_fingerprint.strip():
        raise ValueError("population cluster artifact fingerprint cannot be blank")
    user_ids = [member.user_id for member in population.members]
    if len(user_ids) != len(set(user_ids)):
        raise ValueError("population contains duplicate runtime users")
    if set(user_ids) != set(dataset.training_histories):
        raise ValueError("population does not partition every prepared user")
    if sorted(member.draw_position for member in population.members) != list(
        range(len(population.members))
    ):
        raise ValueError("population draw positions must be contiguous from zero")
    for member in population.members:
        if dataset.runtime_to_source_user_id[member.user_id] != member.source_user_id:
            raise ValueError(f"population source user mismatch for {member.user_id}")
        if isinstance(member.cluster_id, bool) or not isinstance(member.cluster_id, int):
            raise TypeError("population cluster IDs must be integers")
        if member.cluster_id < 0:
            raise ValueError("population cluster IDs cannot be negative")
    if not any(
        member.role is PopulationRole.ACTIVE for member in population.members
    ):
        raise ValueError("population must contain at least one active user")
    reserve_members = tuple(
        member
        for member in population.members
        if member.role is PopulationRole.RESERVE
    )
    if any(member.reserve_order is None for member in reserve_members):
        raise ValueError("every reserve member requires a reserve order")
    reserve_orders = sorted(int(member.reserve_order) for member in reserve_members)
    if reserve_orders != list(range(len(reserve_orders))):
        raise ValueError("reserve orders must be contiguous from zero")
    if any(
        member.reserve_order is not None
        for member in population.members
        if member.role is not PopulationRole.RESERVE
    ):
        raise ValueError("only reserve members may have a reserve order")


def _validate_profiles(
    profiles: Mapping[UserId, ProfileArtifact],
    population: PopulationArtifact,
    dataset: PreparedDatasetArtifact,
) -> None:
    required = {
        member.user_id
        for member in population.members
        if member.role in {PopulationRole.ACTIVE, PopulationRole.RESERVE}
    }
    missing = sorted(required.difference(profiles))
    if missing:
        raise ValueError(f"profile bundle is missing active/reserve users: {missing}")
    unexpected = sorted(set(profiles).difference(required))
    if unexpected:
        raise ValueError(
            "profile bundle contains users outside the active/reserve cohort: "
            f"{unexpected}"
        )
    members = {member.user_id: member for member in population.members}
    for user_id, profile in profiles.items():
        member = members[user_id]
        if profile.source_user_id != member.source_user_id:
            raise ValueError(f"profile source user mismatch for user {user_id}")
        if profile.dataset_fingerprint != dataset.dataset_fingerprint:
            raise ValueError(f"profile dataset fingerprint mismatch for user {user_id}")
        if (
            profile.training_history_fingerprint
            != dataset.training_history_fingerprints[user_id]
        ):
            raise ValueError(
                f"profile training-history fingerprint mismatch for user {user_id}"
            )
        if (
            member.profile_fingerprint is not None
            and member.profile_fingerprint != profile.profile_fingerprint
        ):
            raise ValueError(f"population/profile fingerprint mismatch for {user_id}")


def _path_fingerprint(path: Path) -> str:
    if path.is_file():
        return _file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(f"recommender checkpoint does not exist: {path}")
    files = sorted(value for value in path.rglob("*") if value.is_file())
    if not files:
        raise ValueError(f"recommender checkpoint directory is empty: {path}")
    return fingerprint_semantic_state(
        {value.relative_to(path).as_posix(): _file_sha256(value) for value in files},
        "checkpoint-directory-v1",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
