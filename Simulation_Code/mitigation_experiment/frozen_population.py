"""Materialize the notebook's frozen cohorts without sampling again."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data_preparation.clustering import FrozenClusterArtifact
from .data_preparation.population import POPULATION_SCHEMA
from .utility.fingerprints import fingerprint_semantic_state
from .utility.user_handling import (
    PopulationArtifact, PopulationMember, PopulationRole, PreparedDatasetArtifact,
)

MEMBERSHIP_DIRECTORY = Path(__file__).resolve().parent / "data" / "populations"
MEMBERSHIP_HASHES = {
    1054: "e0d5f22cdc9150f1fb490823a3cf872eb3392dc90ec619f8e1ce5ac2599086da",
    19277: "6c7a066b29dc922e7e86a71800d6c008894844bf40c75d6cf116ddedcceadea2",
    18844: "3ada19d8a532269873e65efd7575ae30ef3862b2b6d3dc3ca82836202dd1ceb4",
}


def load_frozen_membership(population_seed: int) -> dict:
    """Read and validate immutable source IDs and their original role order."""
    if isinstance(population_seed, bool) or not isinstance(population_seed, int):
        raise TypeError("population_seed must be an integer")
    if population_seed not in MEMBERSHIP_HASHES:
        raise ValueError(f"no approved frozen population for seed {population_seed}")
    path = MEMBERSHIP_DIRECTORY / f"population_{population_seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "mitigation-frozen-membership-v1":
        raise ValueError("unsupported frozen membership schema")
    if payload.get("population_seed") != population_seed:
        raise ValueError("frozen membership seed differs from requested seed")
    membership = payload["membership"]
    if set(membership) != {"active", "reserve"}:
        raise ValueError("frozen membership must declare active and reserve lists")
    active, reserve = membership["active"], membership["reserve"]
    if not isinstance(active, list) or not isinstance(reserve, list):
        raise ValueError("frozen membership role values must be ordered lists")
    if (len(active), len(reserve)) != (100, 200):
        raise ValueError("frozen membership must contain 100 plus 200 users")
    selected = active + reserve
    if any(type(value) is not int or not 1 <= value <= 6040 for value in selected):
        raise ValueError("frozen membership contains invalid source user IDs")
    if len(set(selected)) != 300:
        raise ValueError("frozen membership contains duplicate users")
    digest = hashlib.sha256(json.dumps(
        membership, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if digest != payload.get("membership_sha256") or digest != MEMBERSHIP_HASHES[population_seed]:
        raise ValueError("frozen membership hash differs from the executed notebook")
    return payload


def _validate_notebook_inputs(dataset, clusters, provenance) -> tuple[int, ...]:
    users = tuple(sorted(dataset.training_histories))
    if len(users) != 6040:
        raise ValueError("the frozen MovieLens design requires all 6040 retained users")
    if set(dataset.runtime_to_source_user_id) != set(users):
        raise ValueError("runtime/source user mapping does not cover training users")
    source_ids = [int(dataset.runtime_to_source_user_id[u]) for u in users]
    if source_ids != list(range(1, 6041)):
        raise ValueError("runtime/source user mapping differs from the notebook")
    if dict(dataset.source_file_fingerprints) != provenance["raw_files"]:
        raise ValueError("raw dataset hashes differ from the notebook")
    if clusters.dataset_fingerprint != dataset.dataset_fingerprint:
        raise ValueError("cluster and prepared dataset fingerprints differ")
    if set(clusters.cluster_by_user) != set(users):
        raise ValueError("cluster assignments do not cover the prepared population")
    for name, expected in provenance["cluster_config"].items():
        if getattr(clusters.config, name) != expected:
            raise ValueError(f"cluster setting {name} differs from the notebook")
    labels = [clusters.cluster_by_user[u] for u in users]
    digest = hashlib.sha256(np.column_stack([source_ids, labels]).astype("<i8").tobytes()).hexdigest()
    if digest != provenance["cluster_assignments_sha256"]:
        raise ValueError("cluster assignments differ from the frozen notebook design")
    # Match the notebook's binary CSR evidence, including catalog column order.
    columns = {item: column for column, item in enumerate(sorted(dataset.catalog))}
    offsets, indices = [0], []
    for user in users:
        try:
            indices.extend(sorted({columns[item] for item in dataset.training_histories[user]}))
        except KeyError as exc:
            raise ValueError("training history contains an unknown catalog item") from exc
        offsets.append(len(indices))
    training_digest = hashlib.sha256(
        np.asarray(offsets, dtype="<i8").tobytes()
        + np.asarray(indices, dtype="<i8").tobytes()
    ).hexdigest()
    if training_digest != provenance["binary_training_matrix_sha256"]:
        raise ValueError("training histories differ from the frozen notebook design")
    return users


def build_frozen_population(
    dataset: PreparedDatasetArtifact,
    clusters: FrozenClusterArtifact,
    *,
    population_seed: int,
    active_user_count: int,
    reserve_user_count: int,
) -> PopulationArtifact:
    """Build 100+200, the same ordered union as 300+0, or all 6040+0.

    Every retained user remains represented in the artifact. The selected
    cohort precedes sorted background users; source IDs are joined explicitly.
    Full-population mode uses sorted IDs and does not draw a new sample.
    """
    if not isinstance(dataset, PreparedDatasetArtifact):
        raise TypeError("dataset must be PreparedDatasetArtifact")
    if not isinstance(clusters, FrozenClusterArtifact):
        raise TypeError("clusters must be FrozenClusterArtifact")
    for name, value in (("population_seed", population_seed),
                        ("active_user_count", active_user_count),
                        ("reserve_user_count", reserve_user_count)):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
    if population_seed < 0:
        raise ValueError("population_seed cannot be negative")
    counts = (active_user_count, reserve_user_count)
    if counts not in {(100, 200), (300, 0), (6040, 0)}:
        raise ValueError("supported populations are 100+200, 300+0, or 6040+0")
    frozen = load_frozen_membership(1054 if counts == (6040, 0) else population_seed)
    users = _validate_notebook_inputs(dataset, clusters, frozen["provenance"])
    if counts == (6040, 0):
        draw = users
    else:
        membership = frozen["membership"]
        reverse = {source: user for user, source in dataset.runtime_to_source_user_id.items()}
        selected = tuple(reverse[str(source)] for source in membership["active"] + membership["reserve"])
        selected_set = set(selected)
        draw = selected + tuple(user for user in users if user not in selected_set)
    members = []
    for position, user in enumerate(draw):
        if position < active_user_count:
            role, reserve_order = PopulationRole.ACTIVE, None
        elif position < active_user_count + reserve_user_count:
            role, reserve_order = PopulationRole.RESERVE, position - active_user_count
        else:
            role, reserve_order = PopulationRole.BACKGROUND, None
        members.append(PopulationMember(
            user_id=user, source_user_id=dataset.runtime_to_source_user_id[user],
            role=role, draw_position=position, reserve_order=reserve_order,
            cluster_id=clusters.cluster_by_user[user], profile_fingerprint=None,
        ))
    payload = dict(
        schema_version=POPULATION_SCHEMA, population_seed=population_seed,
        source_population_fingerprint=dataset.dataset_fingerprint,
        cluster_artifact_fingerprint=clusters.artifact_fingerprint,
        members=tuple(members),
    )
    return PopulationArtifact(**payload, population_fingerprint=fingerprint_semantic_state(payload, POPULATION_SCHEMA))


__all__ = ["build_frozen_population", "load_frozen_membership", "MEMBERSHIP_HASHES"]
