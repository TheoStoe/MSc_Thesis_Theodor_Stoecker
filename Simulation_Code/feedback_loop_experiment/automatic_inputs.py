"""Create or reuse the concrete inputs needed by one simulator command."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from .data_preparation import (
    ClusterPipelineConfig,
    fit_user_clusters,
    load_cluster_artifact,
    load_prepared_dataset,
    prepare_movielens_1m,
    sample_population,
    save_cluster_artifact,
    save_prepared_dataset,
)
from .persona_extraction import (
    LusiferProfileExtractor,
    ProfileArtifact,
    ProfileExtractionFailure,
    ProfileExtractionInput,
    ProfileInteraction,
    match_movielens_1m_user_texts,
    parse_movielens_1m_users,
)
from .utility.providers import JSONProvider
from .utility.saving import save_population_artifact, save_profile_bundle
from .utility.user_handling import (
    PopulationRole,
    load_population_artifact,
    load_prepared_inputs,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutomaticInputPaths:
    prepared_data: Path
    clusters: Path
    population: Path
    profiles: Path | None


def ensure_automatic_inputs(
    *,
    raw_data_directory: Path,
    input_root: Path,
    population_seed: int,
    active_user_count: int,
    reserve_user_count: int,
    require_profiles: bool,
    provider: JSONProvider | None = None,
    profile_max_interactions: int = 100,
    profile_batch_size: int = 10,
) -> AutomaticInputPaths:
    """Create missing inputs once and validate everything reused later."""

    raw_data_directory = Path(raw_data_directory)
    input_root = Path(input_root)
    cohort_name = (
        f"seed_{population_seed}__active_{active_user_count}"
        f"__reserve_{reserve_user_count}"
    )
    prepared_path = input_root / "ml1m"
    cluster_path = input_root / "ml1m_clusters"
    population_path = input_root / f"population__{cohort_name}.json"
    profile_path = input_root / f"profiles__{cohort_name}"

    if prepared_path.exists():
        dataset = load_prepared_dataset(prepared_path)
    else:
        _reject_incomplete(prepared_path)
        logger.info("Preparing MovieLens data at %s", prepared_path)
        dataset = prepare_movielens_1m(raw_data_directory)
        save_prepared_dataset(prepared_path, dataset)

    if cluster_path.exists():
        clusters = load_cluster_artifact(cluster_path)
    else:
        _reject_incomplete(cluster_path)
        logger.info("Fitting user clusters at %s", cluster_path)
        clusters = fit_user_clusters(dataset, config=ClusterPipelineConfig())
        save_cluster_artifact(cluster_path, clusters)
    if clusters.dataset_fingerprint != dataset.dataset_fingerprint:
        raise ValueError("automatic cluster and prepared-data artifacts disagree")

    if population_path.exists():
        population = load_population_artifact(population_path)
    else:
        _reject_incomplete(population_path)
        logger.info("Sampling population at %s", population_path)
        population = sample_population(
            dataset,
            clusters,
            population_seed=population_seed,
            active_user_count=active_user_count,
            reserve_user_count=reserve_user_count,
        )
        save_population_artifact(population_path, population)

    _validate_population_request(
        population,
        population_seed=population_seed,
        active_user_count=active_user_count,
        reserve_user_count=reserve_user_count,
    )
    # Reuse the normal loader as the final dataset/population consistency gate.
    load_prepared_inputs(prepared_path, population_path, None, None)

    resolved_profiles: Path | None = None
    if require_profiles:
        if profile_path.exists():
            load_prepared_inputs(prepared_path, population_path, profile_path, None)
        else:
            _reject_incomplete(profile_path)
            if provider is None:
                raise ValueError("a provider is required to create agent profiles")
            logger.info("Extracting agent profiles at %s", profile_path)
            profile_inputs = _profile_inputs(
                raw_data_directory=raw_data_directory,
                dataset=dataset,
                population=population,
                extraction_seed=population_seed,
            )
            extractor = LusiferProfileExtractor(
                provider=provider,
                max_interactions=profile_max_interactions,
                batch_size=profile_batch_size,
            )
            results = tuple(extractor.extract_many(profile_inputs))
            profiles = tuple(
                result for result in results if isinstance(result, ProfileArtifact)
            )
            failures = tuple(
                result
                for result in results
                if isinstance(result, ProfileExtractionFailure)
            )
            save_profile_bundle(
                profile_path,
                inputs=profile_inputs,
                profiles=profiles,
                failures=failures,
                metadata={
                    "purpose": "automatic_feedback_loop_inputs",
                    "population_fingerprint": population.population_fingerprint,
                },
            )
            if failures:
                raise RuntimeError(
                    f"profile extraction failed for {len(failures)} user(s); "
                    f"inspect {profile_path}"
                )
            load_prepared_inputs(prepared_path, population_path, profile_path, None)
        resolved_profiles = profile_path

    return AutomaticInputPaths(
        prepared_data=prepared_path,
        clusters=cluster_path,
        population=population_path,
        profiles=resolved_profiles,
    )


def _profile_inputs(
    *,
    raw_data_directory: Path,
    dataset,
    population,
    extraction_seed: int,
) -> tuple[ProfileExtractionInput, ...]:
    selected = tuple(
        member
        for member in population.members
        if member.role in {PopulationRole.ACTIVE, PopulationRole.RESERVE}
    )
    runtime_to_source = {
        member.user_id: member.source_user_id for member in selected
    }
    user_texts = match_movielens_1m_user_texts(
        runtime_to_source,
        parse_movielens_1m_users(Path(raw_data_directory) / "users.dat"),
    )
    records: list[ProfileExtractionInput] = []
    for member in sorted(selected, key=lambda value: value.user_id):
        matched = user_texts[member.user_id]
        interactions = tuple(
            ProfileInteraction(
                event_id=f"rating-row-{event.source_row}",
                source_order=event.source_row,
                source_split="train",
                item_id=event.item_id,
                source_item_id=event.source_item_id,
                timestamp=event.timestamp,
                rating=float(event.rating),
                item_text=str(dataset.catalog[event.item_id]["item_text"]),
            )
            for event in dataset.training_interactions[member.user_id]
        )
        records.append(
            ProfileExtractionInput(
                user_id=member.user_id,
                source_user_id=member.source_user_id,
                user_text=matched.user_text,
                user_text_builder_version=matched.user_text_builder_version,
                user_text_fingerprint=matched.user_text_fingerprint,
                interactions=interactions,
                training_history_fingerprint=(
                    dataset.training_history_fingerprints[member.user_id]
                ),
                dataset_fingerprint=dataset.dataset_fingerprint,
                extraction_seed=extraction_seed,
            )
        )
    return tuple(records)


def _validate_population_request(
    population,
    *,
    population_seed: int,
    active_user_count: int,
    reserve_user_count: int,
) -> None:
    if population.population_seed != population_seed:
        raise ValueError("automatic population seed disagrees with the request")
    active = sum(
        member.role is PopulationRole.ACTIVE for member in population.members
    )
    reserve = sum(
        member.role is PopulationRole.RESERVE for member in population.members
    )
    if (active, reserve) != (active_user_count, reserve_user_count):
        raise ValueError("automatic population size disagrees with the request")


def _reject_incomplete(path: Path) -> None:
    incomplete = path.with_name(path.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(f"incomplete automatic input exists: {incomplete}")


__all__ = ["AutomaticInputPaths", "ensure_automatic_inputs"]
