"""Deterministic reserve-selection mechanics for population policies."""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import UserId
from ..utility.randomness import RandomKey, random_generator


_REFILL_STREAM = 1337  # Independent RNG namespace for population refill.


def select_uniform_reserve(
    *,
    reserve: list[UserId],
    seed: int,
    iteration: int,
    slot: int,
) -> tuple[UserId | None, Mapping[str, Any]]:
    if not reserve:
        return None, {
            "slot": slot,
            "status": "reserve_exhausted",
            "selection_probability": 0.0,
        }
    generator = random_generator(
        seed,
        RandomKey(stream_id=_REFILL_STREAM, iteration=iteration, index=slot),
    )
    index = int(generator.integers(0,len(reserve)))
    selected = reserve.pop(index)
    return selected, {
        "slot": slot,
        "status": "activated",
        "selected_user_id": selected,
        "selection_probability": 1.0 / (len(reserve) + 1),
    }


def select_survivor_similar_reserve(
    *,
    reserve: list[UserId],
    seed: int,
    iteration: int,
    slot: int,
    survivors: tuple[UserId, ...],
    clusters: Mapping[UserId, int],
    selection_rule_id: str,
) -> tuple[UserId | None, Mapping[str, Any]]:
    if not reserve:
        return None, {
            "slot": slot,
            "status": "reserve_exhausted",
            "selection_probability": 0.0,
        }
    if not survivors:
        return None, {
            "slot": slot,
            "status": "no_survivor_reference",
            "selection_probability": 0.0,
        }

    counts: dict[int, int] = {}
    for user_id in survivors:
        cluster = clusters[user_id]
        counts[cluster] = counts.get(cluster, 0) + 1
    cluster_ids = tuple(sorted(counts))
    probabilities = tuple(counts[value] / len(survivors) for value in cluster_ids)
    generator = random_generator(
        seed,
        RandomKey(stream_id=_REFILL_STREAM, iteration=iteration, index=slot),
    )
    target_index = int(generator.choice(len(cluster_ids), p=probabilities))
    target_cluster = cluster_ids[target_index]
    candidates = [
        user_id for user_id in reserve if clusters[user_id] == target_cluster
    ]
    fallback_used = False
    if candidates:
        selection_mode = "target_cluster_uniform"
    elif selection_rule_id.endswith("leave_unfilled_v1"):
        return None, {
            "slot": slot,
            "status": "target_cluster_exhausted",
            "target_cluster": target_cluster,
            "target_cluster_probability": probabilities[target_index],
            "selection_probability": 0.0,
        }
    else:
        # The preregistered global fallback preserves cohort size when the
        # selected survivor cluster has no remaining reserve member.
        candidates = list(reserve)
        fallback_used = True
        selection_mode = "global_uniform_fallback"

    candidate_index = int(generator.integers(0, len(candidates)))
    selected = candidates[candidate_index]
    within_pool_probability = 1.0 / len(candidates)
    reserve.remove(selected)
    return selected, {
        "slot": slot,
        "status": "activated",
        "selected_user_id": selected,
        "target_cluster": target_cluster,
        "target_cluster_probability": probabilities[target_index],
        "candidate_pool_size": len(candidates),
        "within_pool_probability": within_pool_probability,
        "selection_probability": (
            probabilities[target_index] * within_pool_probability
        ),
        "selection_mode": selection_mode,
        "fallback_used": fallback_used,
    }
