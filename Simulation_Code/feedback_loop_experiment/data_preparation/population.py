"""Deterministic per-seed active/reserve/background population draws."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..contracts import UserId
from ..utility.fingerprints import fingerprint_semantic_state
from ..utility.user_handling import (
    PopulationArtifact,
    PopulationMember,
    PopulationRole,
    PreparedDatasetArtifact,
)
from .clustering import FrozenClusterArtifact


POPULATION_SCHEMA = "feedback-loop-population-v1"


def sample_population(
    dataset: PreparedDatasetArtifact,
    clusters: FrozenClusterArtifact,
    *,
    population_seed: int,
    active_user_count: int,
    reserve_user_count: int,
    profile_fingerprints: Mapping[UserId, str] | None = None,
) -> PopulationArtifact:
    """Partition every retained user after one deterministic permutation."""

    if not isinstance(dataset, PreparedDatasetArtifact):
        raise TypeError("dataset must be PreparedDatasetArtifact")
    if not isinstance(clusters, FrozenClusterArtifact):
        raise TypeError("clusters must be FrozenClusterArtifact")
    for name, value in (
        ("population_seed", population_seed),
        ("active_user_count", active_user_count),
        ("reserve_user_count", reserve_user_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if population_seed < 0:
        raise ValueError("population_seed cannot be negative")
    if active_user_count < 1 or reserve_user_count < 0:
        raise ValueError("active count must be positive and reserve count non-negative")
    if clusters.dataset_fingerprint != dataset.dataset_fingerprint:
        raise ValueError("cluster and prepared dataset fingerprints differ")
    user_ids = tuple(sorted(dataset.training_histories))
    if set(clusters.cluster_by_user) != set(user_ids):
        raise ValueError("cluster assignments do not cover the prepared population")
    if active_user_count + reserve_user_count > len(user_ids):
        raise ValueError("active plus reserve counts exceed the population")
    supplied_profiles = dict(profile_fingerprints or {})
    unknown_profiles = sorted(set(supplied_profiles).difference(user_ids))
    if unknown_profiles:
        raise ValueError(f"profile fingerprints contain unknown users: {unknown_profiles}")

    generator = np.random.default_rng(population_seed)
    draw = tuple(int(value) for value in generator.permutation(user_ids))
    members: list[PopulationMember] = []
    reserve_start = active_user_count
    reserve_stop = active_user_count + reserve_user_count
    for draw_position, user_id in enumerate(draw):
        if draw_position < active_user_count:
            role = PopulationRole.ACTIVE
            reserve_order = None
        elif draw_position < reserve_stop:
            role = PopulationRole.RESERVE
            reserve_order = draw_position - reserve_start
        else:
            role = PopulationRole.BACKGROUND
            reserve_order = None
        members.append(
            PopulationMember(
                user_id=user_id,
                source_user_id=dataset.runtime_to_source_user_id[user_id],
                role=role,
                draw_position=draw_position,
                reserve_order=reserve_order,
                cluster_id=clusters.cluster_by_user[user_id],
                profile_fingerprint=supplied_profiles.get(user_id),
            )
        )

    payload = {
        "schema_version": POPULATION_SCHEMA,
        "population_seed": population_seed,
        "source_population_fingerprint": dataset.dataset_fingerprint,
        "cluster_artifact_fingerprint": clusters.artifact_fingerprint,
        "members": tuple(members),
    }
    return PopulationArtifact(
        **payload,
        population_fingerprint=fingerprint_semantic_state(
            payload, POPULATION_SCHEMA
        ),
    )


__all__ = ["POPULATION_SCHEMA", "sample_population"]
