"""Load one frozen, seed-specific sponsored-item schedule."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import ItemId
from ..utility.fingerprints import fingerprint_semantic_state
from ._common import positive_id, unique_ids


SPONSOR_SCHEDULE_SCHEMA = "feedback-loop-sponsor-schedules-v1"


@dataclass(frozen=True)
class SponsorSchedule:
    population_seed: int
    item_ids: tuple[ItemId, ...]
    artifact_fingerprint: str
    source_path: Path


def load_sponsor_schedule(
    path: str | Path,
    *,
    population_seed: int,
    expected_dataset_fingerprint: str,
    catalog_item_ids: Sequence[ItemId],
) -> SponsorSchedule:
    """Load and validate the schedule selected for one population seed."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"sponsor schedule does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("sponsor schedule is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("sponsor schedule must contain a JSON object")
    if payload.get("schema_version") != SPONSOR_SCHEDULE_SCHEMA:
        raise ValueError("unsupported sponsor schedule schema")
    if payload.get("dataset_fingerprint") != expected_dataset_fingerprint:
        raise ValueError("sponsor schedule dataset fingerprint mismatch")
    schedules = payload.get("schedules")
    if not isinstance(schedules, Mapping):
        raise ValueError("sponsor schedule is missing schedules")
    raw_items = schedules.get(str(positive_id(population_seed, "population_seed")))
    if not isinstance(raw_items, list):
        raise ValueError(
            f"sponsor schedule has no list for population seed {population_seed}"
        )
    item_ids = unique_ids(raw_items, "sponsored item IDs")
    catalog = frozenset(unique_ids(catalog_item_ids, "catalog_item_ids"))
    unknown = tuple(item_id for item_id in item_ids if item_id not in catalog)
    if unknown:
        raise ValueError(f"sponsor schedule contains unknown items: {unknown}")
    artifact_fingerprint = fingerprint_semantic_state(
        _semantic_payload(payload),
        SPONSOR_SCHEDULE_SCHEMA,
    )
    return SponsorSchedule(
        population_seed=population_seed,
        item_ids=item_ids,
        artifact_fingerprint=artifact_fingerprint,
        source_path=source,
    )


def _semantic_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "artifact_fingerprint"
    }


__all__ = [
    "SPONSOR_SCHEDULE_SCHEMA",
    "SponsorSchedule",
    "load_sponsor_schedule",
]
