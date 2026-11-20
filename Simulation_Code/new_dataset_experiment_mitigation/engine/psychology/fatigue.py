"""Deterministic cumulative fatigue for the rank-step prompt ablation.

The bucket width of four is inspired by Agent4Rec's default four-item page,
but this is deliberately not a reproduction of Agent4Rec's bundled activity,
satisfaction, pagination, and exit mechanism. It is a local, auditable
operationalization in which lower-ranked clicks add more fatigue, a top-four
click allows one unit of recovery, and a no-click adds the largest increment
available for the displayed slate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..contracts import ItemId


RANK_STEP_FATIGUE_IMPLEMENTATION_VERSION = "rank_step_cumulative_prompt_v1"
RANK_STEP_FATIGUE_BUCKET_SIZE = 4
RANK_STEP_FATIGUE_TOP_CLICK_DELTA = -1
RANK_STEP_FATIGUE_FLOOR = 0


@dataclass(frozen=True)
class RankStepFatigueTransition:
    """One deterministic cumulative-fatigue update."""

    value_before: int
    delta: int
    value_after: int
    clicked_rank: int | None
    slate_size: int

    def to_dict(self) -> dict[str, int | None]:
        return {
            "value_before": self.value_before,
            "delta": self.delta,
            "value_after": self.value_after,
            "clicked_rank": self.clicked_rank,
            "slate_size": self.slate_size,
        }


def update_rank_step_fatigue(
    current_value: int,
    *,
    displayed_rank_by_item: Mapping[ItemId, int],
    clicked_item_id: ItemId | None,
) -> RankStepFatigueTransition:
    """Update cumulative fatigue from one click/no-click outcome.

    Clicks at ranks 1--4 recover one unit, ranks 5--8 add one, ranks 9--12
    add two, and so on. A no-click adds one more than the deepest click bucket
    represented by the slate. The cumulative value is floored at zero and is
    otherwise deliberately uncapped.
    """

    if isinstance(current_value, bool) or not isinstance(current_value, int):
        raise TypeError("current fatigue value must be an integer")
    if current_value < RANK_STEP_FATIGUE_FLOOR:
        raise ValueError("current fatigue value cannot be negative")
    ranks = dict(displayed_rank_by_item)
    if not ranks:
        raise ValueError("displayed_rank_by_item cannot be empty")
    if sorted(ranks.values()) != list(range(1, len(ranks) + 1)):
        raise ValueError("displayed ranks must be exactly 1..slate_size")

    if clicked_item_id is None:
        clicked_rank = None
        delta = (len(ranks) + RANK_STEP_FATIGUE_BUCKET_SIZE - 1) // (
            RANK_STEP_FATIGUE_BUCKET_SIZE
        )
    else:
        try:
            clicked_rank = ranks[clicked_item_id]
        except KeyError as exc:
            raise ValueError("clicked item has no displayed rank") from exc
        if clicked_rank <= RANK_STEP_FATIGUE_BUCKET_SIZE:
            delta = RANK_STEP_FATIGUE_TOP_CLICK_DELTA
        else:
            delta = (clicked_rank - 1) // RANK_STEP_FATIGUE_BUCKET_SIZE

    value_after = max(RANK_STEP_FATIGUE_FLOOR, current_value + delta)
    return RankStepFatigueTransition(
        value_before=current_value,
        delta=delta,
        value_after=value_after,
        clicked_rank=clicked_rank,
        slate_size=len(ranks),
    )


__all__ = [
    "RANK_STEP_FATIGUE_BUCKET_SIZE",
    "RANK_STEP_FATIGUE_FLOOR",
    "RANK_STEP_FATIGUE_IMPLEMENTATION_VERSION",
    "RANK_STEP_FATIGUE_TOP_CLICK_DELTA",
    "RankStepFatigueTransition",
    "update_rank_step_fatigue",
]
