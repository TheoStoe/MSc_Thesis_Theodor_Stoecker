"""Freeze 150 active / 150 reserve roles within an approved 300-user cohort.

This is a role assignment, never a fresh draw from MovieLens. Only the original
300 selected users may be active or reserve, and all 300 historical logs remain
in the recommender's ``active_reserve`` training scope. Role assignment uses
frozen cluster labels and the population seed; simulation outcomes and simulation
seeds are not inputs. Existing populations, profiles, and checkpoints are read
only. No provider or model is constructed here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

from .utility.fingerprints import fingerprint_semantic_state
from .utility.saving import save_json_artifact, save_population_artifact
from .utility.user_handling import (
    PopulationArtifact, PopulationMember, PopulationRole,
    _validate_population_against_dataset,
)


PARTITION_SCHEMA = "feedback-l4-l5-frozen-cohort-150-150-v1"
PARTITION_METHOD = "cluster_largest_remainder_half_seeded_sha256_priority_v1"
ACTIVE_COUNT = 150
RESERVE_COUNT = 150


def _priority(population_seed: int, kind: str, *identifiers: Any) -> str:
    value = [PARTITION_SCHEMA, population_seed, kind, *identifiers]
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _member(source: Any, *, role: PopulationRole | None = None,
            reserve_order: int | None = None) -> PopulationMember:
    return PopulationMember(
        user_id=source.user_id, source_user_id=source.source_user_id,
        role=PopulationRole(source.role.value) if role is None else role,
        draw_position=source.draw_position, reserve_order=reserve_order,
        cluster_id=source.cluster_id, profile_fingerprint=source.profile_fingerprint,
    )


def training_history_fingerprint(population: Any, dataset: Any) -> str:
    """Namespace/order-independent identity of the complete selected histories.

    Histories themselves retain chronological item order. Roles and the order of
    population members cannot change this digest. Source history fingerprints
    additionally identify ratings, timestamps, and source rows in the report.
    """
    selected = sorted((member for member in population.members
                       if member.role.value in {"active", "reserve"}),
                      key=lambda member: member.user_id)
    payload = tuple((member.user_id, member.source_user_id,
                     tuple(dataset.training_histories[member.user_id]))
                    for member in selected)
    return fingerprint_semantic_state(payload, "l4-l5-selected-full-histories-v1")


def _counts(members: tuple[Any, ...], metadata: Mapping[str, Mapping[str, Any]],
            field: str) -> dict[str, int]:
    result = Counter(str(metadata.get(member.source_user_id, {}).get(field) or "unknown")
                     for member in members)
    return dict(sorted(result.items()))


def _describe_group(members: tuple[Any, ...], dataset: Any,
                    metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    lengths = [len(dataset.training_histories[member.user_id]) for member in members]
    return {
        "users": len(members),
        "cluster_counts": dict(sorted(Counter(str(member.cluster_id)
                                              for member in members).items())),
        "demographic_counts": {field: _counts(members, metadata, field)
                               for field in ("gender", "age", "occupation")},
        "training_history_lengths": {
            "total_events": sum(lengths), "mean": statistics.mean(lengths),
            "median": statistics.median(lengths), "minimum": min(lengths),
            "maximum": max(lengths),
        },
    }


def partition_population(
    population: Any, dataset: Any, *, population_seed: int | None = None,
    demographics_by_source: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[PopulationArtifact, dict[str, Any]]:
    """Assign matched roles and return a native feedback population plus audit.

    ``population`` and ``dataset`` may be validated mitigation contracts or
    validated feedback contracts. File-based use must validate the original
    immutable bundle first; :func:`prepare_population` provides that boundary.
    Cluster quotas use exact integer largest-remainder allocation of half each
    cluster. Seeded hash priorities break remainder ties and choose members.
    Demographics and history lengths are reported only, never selected on.
    """
    seed = population.population_seed if population_seed is None else population_seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("population_seed must be a non-negative integer")
    if seed != population.population_seed:
        raise ValueError("role seed must equal the frozen population seed")
    original = tuple(sorted(population.members, key=lambda member: member.draw_position))
    selected = tuple(member for member in original if member.role.value == "active")
    if (len(selected) != ACTIVE_COUNT + RESERVE_COUNT
            or any(member.role.value == "reserve" for member in original)):
        raise ValueError("source must have exactly 300 active users and no reserves")
    if len({member.user_id for member in original}) != len(original):
        raise ValueError("source population contains duplicate user IDs")
    if population.source_population_fingerprint != dataset.dataset_fingerprint:
        raise ValueError("source population and dataset fingerprints differ")
    if set(member.user_id for member in original) != set(dataset.training_histories):
        raise ValueError("source population must retain every dataset user")
    if tuple(member.draw_position for member in original) != tuple(range(len(original))):
        raise ValueError("source population draw positions are not contiguous")
    for member in original:
        if dataset.runtime_to_source_user_id[member.user_id] != member.source_user_id:
            raise ValueError("source population user identity does not match dataset")
        if isinstance(member.cluster_id, bool) or not isinstance(member.cluster_id, int) or member.cluster_id < 0:
            raise ValueError("frozen cluster IDs must be non-negative integers")

    groups: dict[int, list[Any]] = defaultdict(list)
    for member in selected:
        groups[member.cluster_id].append(member)
    quotas = {cluster: len(members) // 2 for cluster, members in groups.items()}
    extra = ACTIVE_COUNT - sum(quotas.values())
    odd_clusters = sorted((cluster for cluster, members in groups.items()
                           if len(members) % 2),
                          key=lambda cluster: (_priority(seed, "cluster_tie", cluster), cluster))
    for cluster in odd_clusters[:extra]:
        quotas[cluster] += 1
    active = set()
    for cluster, members in groups.items():
        ordered = sorted(members, key=lambda member: (
            _priority(seed, "user", cluster, member.source_user_id), member.source_user_id,
        ))
        active.update(member.user_id for member in ordered[:quotas[cluster]])
    reserves = tuple(member for member in selected if member.user_id not in active)
    if len(active) != ACTIVE_COUNT or len(reserves) != RESERVE_COUNT:
        raise AssertionError("cluster allocation did not produce exactly 150 / 150 roles")
    reserve_order = {member.user_id: position for position, member in enumerate(reserves)}
    # Preserve original member/draw order. Only the two role fields change.
    members = tuple(_member(member,
                            role=(PopulationRole.ACTIVE if member.user_id in active
                                  else PopulationRole.RESERVE),
                            reserve_order=reserve_order.get(member.user_id))
                    if member.role.value == "active" else _member(member)
                    for member in original)
    payload = {
        "schema_version": population.schema_version, "population_seed": seed,
        "source_population_fingerprint": population.source_population_fingerprint,
        "cluster_artifact_fingerprint": population.cluster_artifact_fingerprint,
        "members": members,
    }
    result = PopulationArtifact(**payload, population_fingerprint=fingerprint_semantic_state(
        payload, population.schema_version))
    _validate_population_against_dataset(result, dataset)
    initial_training = tuple(member.user_id for member in selected)
    after_training = tuple(member.user_id for member in members
                           if member.role in {PopulationRole.ACTIVE, PopulationRole.RESERVE})
    if initial_training != after_training:
        raise AssertionError("role assignment changed the ordered training cohort")
    before_digest = training_history_fingerprint(population, dataset)
    if training_history_fingerprint(result, dataset) != before_digest:
        raise AssertionError("role assignment changed selected training histories")

    metadata = dict(demographics_by_source or {})
    cohorts = {
        "cohort_300": selected,
        "active_150": tuple(member for member in members if member.role is PopulationRole.ACTIVE),
        "reserve_150": tuple(member for member in members if member.role is PopulationRole.RESERVE),
    }
    summaries = {name: _describe_group(group, dataset, metadata)
                 for name, group in cohorts.items()}
    differences = {}
    for name in ("active_150", "reserve_150"):
        differences[name] = {
            field: {label: 100 * (summaries[name]["demographic_counts"][field].get(label, 0) / 150
                                  - count / 300)
                    for label, count in summaries["cohort_300"]["demographic_counts"][field].items()}
            for field in ("gender", "age", "occupation")
        }
    report = {
        "schema": PARTITION_SCHEMA, "method": PARTITION_METHOD,
        "population_seed": seed, "simulation_seed_used_for_role_assignment": False,
        "source_population_fingerprint": population.population_fingerprint,
        "population_fingerprint": result.population_fingerprint,
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "cluster_artifact_fingerprint": population.cluster_artifact_fingerprint,
        "active_users": ACTIVE_COUNT, "reserve_users": RESERVE_COUNT,
        "background_users": len(original) - 300,
        "training_scope": "active_reserve", "training_users": 300,
        "ordered_training_user_ids": initial_training,
        "ordered_training_source_user_ids": tuple(member.source_user_id for member in selected),
        "training_history_fingerprint": before_digest,
        "training_user_history_fingerprints": {str(member.user_id):
            dataset.training_history_fingerprints[member.user_id] for member in selected},
        "initial_active_source_user_ids": tuple(member.source_user_id
                                                 for member in cohorts["active_150"]),
        "initial_reserve_source_user_ids": tuple(member.source_user_id
                                                  for member in cohorts["reserve_150"]),
        "cluster_allocation": [
            {"cluster_id": cluster, "cohort": len(groups[cluster]),
             "active": quotas[cluster], "reserve": len(groups[cluster]) - quotas[cluster],
             "ideal_active_quota": len(groups[cluster]) / 2}
            for cluster in sorted(groups)
        ],
        "descriptive_comparison": summaries,
        "demographic_share_difference_from_cohort_pp": differences,
        "demographics_available": bool(metadata),
        "interpretation": (
            "Cluster-stratified role halves of the approved 300, with no new users. "
            "Cluster quotas differ from exact halves by at most 0.5 user. "
            "Demographic and activity summaries are descriptive; exact balance or "
            "population representativeness is not asserted. No outcome-based selection."
        ),
    }
    return result, report


def demographics_from_profile_inputs(profile_directory: str | Path) -> dict[str, dict[str, str]]:
    """Read frozen demographic fields without inspecting generated profile prose."""
    root = Path(profile_directory)
    path = root / "inputs.jsonl"
    complete = json.loads((root / "_COMPLETE.json").read_text(encoding="utf-8"))
    expected = complete.get("file_sha256", {}).get("inputs.jsonl")
    if complete.get("status") != "complete" or not expected:
        raise ValueError("profile inputs require a complete bundle and checksum")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError("frozen profile input checksum mismatch")
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_id = str(row["source_user_id"])
            if source_id in result:
                raise ValueError("profile inputs contain duplicate users")
            values = {}
            for part in row.get("user_text", "").splitlines():
                key, separator, value = part.partition(":")
                if separator and key.strip().lower() in {"gender", "age", "occupation"}:
                    values[key.strip().lower()] = value.strip() or "unknown"
            result[source_id] = values
    return result


def prepare_population(
    source_population_path: str | Path, prepared_data_path: str | Path,
    destination: str | Path, *, input_format: str = "mitigation",
    profiles_path: str | Path | None = None,
) -> Path:
    """Validate original files and save a new immutable role-only population bundle."""
    destination = Path(destination)
    staging = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or staging.exists():
        raise FileExistsError("preserve previous population preparation; choose a fresh destination")
    if input_format == "mitigation":
        from mitigation_experiment.utility.user_handling import load_prepared_inputs
        from .mitigation_compat import _validate_frozen_selection
    elif input_format == "feedback":
        from .utility.user_handling import load_prepared_inputs
        _validate_frozen_selection = None
    else:
        raise ValueError("input_format must be mitigation or feedback")
    loaded = load_prepared_inputs(Path(prepared_data_path), Path(source_population_path),
                                  None if profiles_path is None else Path(profiles_path), None)
    if _validate_frozen_selection is not None:
        _validate_frozen_selection(loaded.population)
    metadata = {} if profiles_path is None else demographics_from_profile_inputs(profiles_path)
    population, report = partition_population(loaded.population, loaded.dataset,
                                              demographics_by_source=metadata)
    report["source_paths"] = {
        "population": str(Path(source_population_path).resolve()),
        "prepared_data": str(Path(prepared_data_path).resolve()),
        "profiles": None if profiles_path is None else str(Path(profiles_path).resolve()),
    }
    staging.mkdir(parents=True)
    save_population_artifact(staging / "population.json", population)
    save_json_artifact(staging / "partition_report.json", report)
    staging.rename(destination)
    return destination / "partition_report.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-population", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--input-format", choices=("mitigation", "feedback"), default="mitigation")
    args = parser.parse_args(argv)
    print(prepare_population(args.source_population, args.prepared_data, args.destination,
                             input_format=args.input_format, profiles_path=args.profiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
