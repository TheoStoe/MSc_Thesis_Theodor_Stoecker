"""Explicit, read-only import of validated mitigation experiment inputs.

Prepared interaction fingerprints include their Python contract namespace.
Consequently, copying a mitigation bundle and opening it with the ordinary
feedback loader must fail validation. This boundary validates the original
namespace first, then translates only runtime contract classes. Scientific
fingerprints continue to identify the immutable, validated source artifacts;
they are not recomputed from the translated in-memory objects.

This importer never samples users, extracts profiles, fits a model, writes an
artifact, or invokes a provider. Legacy feedback inputs retain their existing
loader and validation rules.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from .persona_extraction.base import ProfileArtifact
from .utility.user_handling import (
    PopulationArtifact,
    PopulationMember,
    PopulationRole,
    PreparedDatasetArtifact,
    PreparedInputs,
    PreparedTrainingInteraction,
    _validate_population_against_dataset,
    _validate_profiles,
)


INPUT_FORMAT = "mitigation"
SOURCE_FINGERPRINT_NAMESPACE = "mitigation_experiment"


def _contract_fields(source: Any, target: type) -> dict[str, Any]:
    """Fail closed if the independently maintained contracts have diverged."""

    names = {field.name for field in fields(target)}
    if {field.name for field in fields(source)} != names:
        raise ValueError(f"incompatible mitigation contract for {target.__name__}")
    return {name: getattr(source, name) for name in names}


def _validate_frozen_selection(population: Any) -> None:
    from mitigation_experiment.frozen_population import load_frozen_membership
    from mitigation_experiment.utility.user_handling import (
        PopulationRole as SourceRole,
    )

    ordered = sorted(population.members, key=lambda member: member.draw_position)
    active = [member for member in ordered if member.role is SourceRole.ACTIVE]
    reserve = [member for member in ordered if member.role is SourceRole.RESERVE]
    if len(active) != 300 or reserve:
        raise ValueError("mitigation input import requires 300 active users and no reserves")
    frozen = load_frozen_membership(population.population_seed)
    membership = frozen["membership"]
    expected = tuple(str(user) for user in membership["active"] + membership["reserve"])
    if tuple(member.source_user_id for member in active) != expected:
        raise ValueError("mitigation population differs from the approved frozen selection")
    if tuple(member.draw_position for member in active) != tuple(range(300)):
        raise ValueError("mitigation active-user order differs from the frozen selection")


def load_mitigation_inputs(
    prepared_data_path: Path,
    population_artifact_path: Path,
    profile_artifact_path: Path | None,
    initial_recommender_path: Path | None,
) -> PreparedInputs:
    """Read a copied 300-active frozen cohort without changing its identities.

    The source loaders verify file hashes, semantic fingerprints and dataset /
    population / profile joins in their native namespace. The approved frozen
    membership is checked independently of a population file's own hash. An
    initial checkpoint must be the unweighted iteration-zero initialization;
    the normal runtime restore subsequently verifies exact settings, catalog,
    training histories and model weights before any provider is constructed.
    """

    from mitigation_experiment.utility.checkpoints import load_checkpoint
    from mitigation_experiment.utility.user_handling import (
        load_prepared_inputs as load_source_inputs,
    )

    if initial_recommender_path is None:
        raise ValueError("mitigation input import requires a shared initial recommender")
    source = load_source_inputs(
        prepared_data_path,
        population_artifact_path,
        profile_artifact_path,
        initial_recommender_path,
    )
    _validate_frozen_selection(source.population)

    checkpoint = load_checkpoint(initial_recommender_path)
    state = checkpoint.component_states.get("recommender", {})
    if (
        checkpoint.schema_version != "mitigation-initial-sasrec-v1"
        or checkpoint.condition_id != "unweighted_initial"
        or checkpoint.completed_iteration != 0
        or checkpoint.input_fingerprints.get("dataset")
        != source.dataset.dataset_fingerprint
        or state.get("variant_id") != "sasrec_pmixer_pytorch_v1"
        or state.get("model_state") is None
        or state.get("mitigation_weight_config") is not None
        or state.get("settings", {}).get("random_seed") != checkpoint.seed
    ):
        raise ValueError("mitigation input checkpoint is not the matching unweighted initialization")

    dataset_values = _contract_fields(source.dataset, PreparedDatasetArtifact)
    dataset_values["training_interactions"] = {
        user: tuple(
            PreparedTrainingInteraction(**_contract_fields(event, PreparedTrainingInteraction))
            for event in events
        )
        for user, events in source.dataset.training_interactions.items()
    }
    dataset = PreparedDatasetArtifact(**dataset_values)
    population_values = _contract_fields(source.population, PopulationArtifact)
    population_values["members"] = tuple(
        PopulationMember(
            **{
                **_contract_fields(member, PopulationMember),
                "role": PopulationRole(member.role.value),
            }
        )
        for member in source.population.members
    )
    population = PopulationArtifact(**population_values)
    profiles = {
        user: ProfileArtifact(**_contract_fields(profile, ProfileArtifact))
        for user, profile in source.profiles_by_user.items()
    }
    # Recheck runtime joins after translation; no source identity is rewritten.
    _validate_population_against_dataset(population, dataset)
    if profile_artifact_path is not None:
        _validate_profiles(profiles, population, dataset)
    return PreparedInputs(
        dataset=dataset,
        population=population,
        profiles_by_user=profiles,
        recommender_checkpoint_path=source.recommender_checkpoint_path,
        recommender_checkpoint_fingerprint=source.recommender_checkpoint_fingerprint,
    )


__all__ = ["INPUT_FORMAT", "SOURCE_FINGERPRINT_NAMESPACE", "load_mitigation_inputs"]
