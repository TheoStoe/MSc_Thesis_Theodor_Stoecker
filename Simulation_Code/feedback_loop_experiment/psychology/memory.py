"""Deterministic ACT-R accessibility derived from an authoritative ledger.

The ledger remains the complete source of truth.  This module only derives a
bounded view for a simulated user's choice prompt; forgetting never deletes or
rewrites recorded interactions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Any

from ..contracts import ItemId


MEMORY_VIEW_SCHEMA = "actr_memory_view_v1"


@dataclass(frozen=True)
class MemoryEntry:
    """One retrieved item chunk and the evidence behind its accessibility."""

    item_id: ItemId
    occurrence_timesteps: tuple[int, ...]
    occurrence_ages: tuple[int, ...]
    activation: float
    accessibility: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit record."""

        return {
            "item_id": self.item_id,
            "occurrence_timesteps": list(self.occurrence_timesteps),
            "occurrence_ages": list(self.occurrence_ages),
            "activation": self.activation,
            "accessibility": self.accessibility,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class MemoryRetrievalView:
    """Top-K accessible chunks for one canonical user opportunity."""

    current_timestep: int
    decay: float
    retrieval_limit: int
    candidate_count: int
    entries: tuple[MemoryEntry, ...]

    @property
    def retrieved_accessibility_mass(self) -> float:
        """Accessibility retained by the bounded view.

        Accessibility is normalized over every item in the authoritative
        ledger, not re-normalized after top-K selection.  The resulting mass
        therefore makes truncation explicit.
        """

        return sum(entry.accessibility for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view suitable for event diagnostics."""

        return {
            "schema_version": MEMORY_VIEW_SCHEMA,
            "current_timestep": self.current_timestep,
            "decay": self.decay,
            "retrieval_limit": self.retrieval_limit,
            "candidate_count": self.candidate_count,
            "retrieved_accessibility_mass": (
                self.retrieved_accessibility_mass
            ),
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class _Candidate:
    item_id: ItemId
    occurrence_timesteps: tuple[int, ...]
    occurrence_ages: tuple[int, ...]
    activation: float
    strength: float


def seed_historical_memories(
    history: Sequence[ItemId],
    *,
    limit: int,
) -> Mapping[ItemId, tuple[int, ...]]:
    """Convert recent historical events into pre-simulation memory chunks.

    ``history`` is ordered oldest to newest.  Only its last ``limit`` events
    are retained.  The newest retained event is assigned timestep zero and
    each preceding event one smaller timestep, so at simulation iteration one
    their ages run from ``N`` down to one.  Repeated items remain repeated
    occurrences within one chunk.
    """

    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError("history must be an item sequence")
    _validate_positive_integer(limit, name="limit")
    retained = tuple(history[-limit:])
    occurrences: dict[ItemId, list[int]] = {}
    first_timestep = 1 - len(retained)
    for offset, item_id in enumerate(retained):
        _validate_positive_integer(item_id, name="historical item_id")
        occurrences.setdefault(item_id, []).append(first_timestep + offset)
    return MappingProxyType(
        {
            item_id: tuple(occurrences[item_id])
            for item_id in sorted(occurrences)
        }
    )


def compose_memory_candidates(
    historical_occurrences: Mapping[ItemId, Sequence[int]],
    online_occurrences: Mapping[ItemId, Sequence[int]],
    *,
    capacity: int,
) -> Mapping[ItemId, tuple[int, ...]]:
    """Compose online chunks with the most recent non-overlapping history.

    Every online chunk is retained, even when the number of online chunks is
    already at or above ``capacity``.  Otherwise, the most recently occurring
    historical chunks fill the remaining slots.  A historical chunk for an
    item already represented online is excluded rather than merged.
    """

    _validate_positive_integer(capacity, name="capacity")
    historical = _copy_occurrence_mapping(
        historical_occurrences,
        name="historical_occurrences",
    )
    online = _copy_occurrence_mapping(
        online_occurrences,
        name="online_occurrences",
    )

    selected_historical: dict[ItemId, tuple[int, ...]] = {}
    remaining_slots = capacity - len(online)
    if remaining_slots > 0:
        eligible = (
            (item_id, timesteps)
            for item_id, timesteps in historical.items()
            if item_id not in online
        )
        most_recent_first = sorted(
            eligible,
            key=lambda item: (-max(item[1]), item[0]),
        )
        selected_historical = dict(most_recent_first[:remaining_slots])

    combined = {**selected_historical, **online}
    return MappingProxyType(
        {item_id: combined[item_id] for item_id in sorted(combined)}
    )


def retrieve_memories(
    occurrences_by_item: Mapping[ItemId, Sequence[int]],
    *,
    current_timestep: int,
    decay: float,
    retrieval_limit: int,
) -> MemoryRetrievalView:
    """Derive a deterministic ACT-R top-K view without mutating ``ledger``.

    For item ``i`` with occurrence ages ``j``, base-level activation is
    ``ln(sum(age_j ** -decay))``.  One subsequent user opportunity advances an
    occurrence to age one; the minimum age is also one when validating state
    immediately after an interaction. Accessibility is ``softmax(activation)``
    across all candidate chunks, equivalently each unnormalized power-law
    strength divided by the total strength.
    """

    if not isinstance(occurrences_by_item, Mapping):
        raise TypeError("occurrences_by_item must be a mapping")
    _validate_timestep(current_timestep, name="current_timestep")
    decay_value = _validate_decay(decay)
    _validate_positive_integer(retrieval_limit, name="retrieval_limit")

    candidates: list[_Candidate] = []
    for item_id, occurrence_timesteps in occurrences_by_item.items():
        _validate_positive_integer(item_id, name="item_id")
        normalized_timesteps = _validate_occurrences(
            occurrence_timesteps,
            current_timestep=current_timestep,
            item_id=item_id,
        )
        occurrence_ages = tuple(
            max(1, current_timestep - timestep)
            for timestep in normalized_timesteps
        )
        strength = math.fsum(
            age ** (-decay_value) for age in occurrence_ages
        )
        activation = math.log(strength)
        candidates.append(
            _Candidate(
                item_id=item_id,
                occurrence_timesteps=normalized_timesteps,
                occurrence_ages=occurrence_ages,
                activation=activation,
                strength=strength,
            )
        )

    candidates.sort(
        key=lambda candidate: (-candidate.activation, candidate.item_id)
    )
    total_strength = math.fsum(
        candidate.strength for candidate in candidates
    )
    selected = candidates[:retrieval_limit]
    entries = tuple(
        MemoryEntry(
            item_id=candidate.item_id,
            occurrence_timesteps=candidate.occurrence_timesteps,
            occurrence_ages=candidate.occurrence_ages,
            activation=candidate.activation,
            accessibility=candidate.strength / total_strength,
            rank=rank,
        )
        for rank, candidate in enumerate(selected, start=1)
    )
    return MemoryRetrievalView(
        current_timestep=current_timestep,
        decay=decay_value,
        retrieval_limit=retrieval_limit,
        candidate_count=len(candidates),
        entries=entries,
    )


def _validate_occurrences(
    occurrence_timesteps: Sequence[int],
    *,
    current_timestep: int,
    item_id: ItemId,
) -> tuple[int, ...]:
    if isinstance(occurrence_timesteps, (str, bytes)) or not isinstance(
        occurrence_timesteps,
        Sequence,
    ):
        raise TypeError(
            f"occurrences for item {item_id} must be an integer sequence"
        )
    if not occurrence_timesteps:
        raise ValueError(f"occurrences for item {item_id} cannot be empty")

    normalized: list[int] = []
    for timestep in occurrence_timesteps:
        _validate_timestep(
            timestep,
            name=f"occurrence timestep for item {item_id}",
            allow_negative=True,
        )
        if timestep > current_timestep:
            raise ValueError(
                f"occurrence timestep for item {item_id} cannot be in the future"
            )
        normalized.append(timestep)
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"occurrence timesteps for item {item_id} must be unique"
        )
    return tuple(sorted(normalized))


def _copy_occurrence_mapping(
    occurrences_by_item: Mapping[ItemId, Sequence[int]],
    *,
    name: str,
) -> dict[ItemId, tuple[int, ...]]:
    if not isinstance(occurrences_by_item, Mapping):
        raise TypeError(f"{name} must be a mapping")
    copied: dict[ItemId, tuple[int, ...]] = {}
    for item_id, occurrence_timesteps in occurrences_by_item.items():
        _validate_positive_integer(item_id, name=f"{name} item_id")
        if isinstance(occurrence_timesteps, (str, bytes)) or not isinstance(
            occurrence_timesteps,
            Sequence,
        ):
            raise TypeError(f"{name} values must be integer sequences")
        if not occurrence_timesteps:
            raise ValueError(f"{name} occurrence sequences cannot be empty")
        normalized: list[int] = []
        for timestep in occurrence_timesteps:
            _validate_timestep(
                timestep,
                name=f"{name} occurrence timestep",
                allow_negative=True,
            )
            normalized.append(timestep)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} occurrence timesteps must be unique")
        copied[item_id] = tuple(sorted(normalized))
    return copied


def _validate_timestep(
    value: Any,
    *,
    name: str,
    allow_negative: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 and not allow_negative:
        raise ValueError(f"{name} cannot be negative")


def _validate_positive_integer(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_decay(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("decay must be numeric")
    decay = float(value)
    if not math.isfinite(decay):
        raise ValueError("decay must be finite")
    if decay < 0:
        raise ValueError("decay cannot be negative")
    return decay
