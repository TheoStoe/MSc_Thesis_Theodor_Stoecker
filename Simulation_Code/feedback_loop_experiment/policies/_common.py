"""Small validation and fingerprint helpers used across policy modules."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..contracts import ItemId
from ..utility.fingerprints import fingerprint_semantic_state


def positive_id(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def iteration(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("iteration must be a positive integer")
    return value


def unique_ids(
    values: Sequence[int],
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    result = tuple(positive_id(value, name) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} cannot be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique IDs")
    return result


def policy_fingerprint(variant_id: str, state: Mapping[str, Any]) -> str:
    return fingerprint_semantic_state(
        {"variant_id": variant_id, "state": state},
        "feedback-loop-policy-state-v1",
    )


def rank_mapping(items: Sequence[ItemId]) -> Mapping[ItemId, int]:
    return MappingProxyType(
        {item_id: rank for rank, item_id in enumerate(items, start=1)}
    )
