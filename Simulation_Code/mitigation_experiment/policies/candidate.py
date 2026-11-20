"""Candidate eligibility policies."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts import ActionDecision, ActionStatus, ItemId, UserAction, UserId
from ._common import (
    iteration as validate_iteration,
    policy_fingerprint,
    positive_id,
    unique_ids,
)
from .contracts import CandidatePolicy


class _ExcludeSeenCandidatePolicy:
    variant_id = "exclude_seen_v1"

    def __init__(
        self,
        catalog_item_ids: Sequence[ItemId],
        initial_histories: Mapping[UserId, Sequence[ItemId]],
        sponsored_item_ids: Sequence[ItemId] = (),
    ) -> None:
        self._catalog = unique_ids(catalog_item_ids, "catalog_item_ids")
        self._catalog_set = frozenset(self._catalog)
        if not initial_histories:
            raise ValueError("initial_histories cannot be empty")
        self._excluded: dict[UserId, set[ItemId]] = {}
        for user_id, history in initial_histories.items():
            user_id = positive_id(user_id, "user_id")
            if user_id in self._excluded:
                raise ValueError(f"duplicate user history: {user_id}")
            values = tuple(positive_id(item, "history item") for item in history)
            unknown = sorted(set(values).difference(self._catalog_set))
            if unknown:
                raise ValueError(
                    f"history for user {user_id} contains unknown items: {unknown}"
                )
            self._excluded[user_id] = set(values)
        self._baseline_excluded = {
            user_id: frozenset(items) for user_id, items in self._excluded.items()
        }
        self._sponsored_item_ids = unique_ids(
            sponsored_item_ids,
            "sponsored_item_ids",
            allow_empty=True,
        )
        unknown_sponsors = tuple(
            item_id
            for item_id in self._sponsored_item_ids
            if item_id not in self._catalog_set
        )
        if unknown_sponsors:
            raise ValueError(
                f"sponsored items are outside the catalog: {unknown_sponsors}"
            )
        self.variant_id = (
            "exclude_seen_with_sponsor_schedule_v2"
            if self._sponsored_item_ids
            else "exclude_seen_v1"
        )

    def eligible_items(self, user_id: UserId, iteration: int) -> tuple[ItemId, ...]:
        user_id = positive_id(user_id, "user_id")
        checked_iteration = validate_iteration(iteration)
        if user_id not in self._excluded:
            raise KeyError(f"unknown user: {user_id}")
        excluded = self._excluded[user_id]
        future_sponsors = frozenset(
            self._sponsored_item_ids[checked_iteration - 1 :]
        )
        return tuple(
            item
            for item in self._catalog
            if item not in excluded and item not in future_sponsors
        )

    def observe_valid_click(self, user_id: UserId, action: ActionDecision) -> None:
        user_id = positive_id(user_id, "user_id")
        if user_id not in self._excluded:
            raise KeyError(f"unknown user: {user_id}")
        if not isinstance(action, ActionDecision):
            raise TypeError("action must be an ActionDecision")
        if action.status is not ActionStatus.VALID or action.action is not UserAction.CLICK:
            return
        if action.clicked_item_id is None:
            raise ValueError("a valid click must name a clicked item")
        item_id = positive_id(action.clicked_item_id, "clicked_item_id")
        if item_id not in self._catalog_set:
            raise ValueError(f"clicked item is outside the catalog: {item_id}")
        self._excluded[user_id].add(item_id)

    def export_state(self) -> Mapping[str, Any]:
        return {
            "variant_id": self.variant_id,
            "catalog_item_ids": self._catalog,
            "sponsored_item_ids": self._sponsored_item_ids,
            "excluded_by_user": tuple(
                (user_id, tuple(sorted(items)))
                for user_id, items in sorted(self._excluded.items())
            ),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("candidate-policy state must be a mapping")
        if state.get("variant_id") != self.variant_id:
            raise ValueError("candidate-policy variant mismatch")
        if tuple(state.get("catalog_item_ids", ())) != self._catalog:
            raise ValueError("candidate-policy catalog mismatch")
        if tuple(state.get("sponsored_item_ids", ())) != self._sponsored_item_ids:
            raise ValueError("candidate-policy sponsor schedule mismatch")
        raw = state.get("excluded_by_user")
        if not isinstance(raw, (tuple, list)):
            raise ValueError("candidate-policy exclusions are missing")
        restored: dict[UserId, set[ItemId]] = {}
        for record in raw:
            if not isinstance(record, (tuple, list)) or len(record) != 2:
                raise ValueError("invalid candidate-policy exclusion record")
            user_id = positive_id(record[0], "user_id")
            items = set(unique_ids(record[1], "excluded items", allow_empty=True))
            if not items.issubset(self._catalog_set) or user_id in restored:
                raise ValueError("invalid candidate-policy exclusions")
            restored[user_id] = items
        if set(restored) != set(self._excluded):
            raise ValueError("candidate-policy user set mismatch")
        for user_id, baseline in self._baseline_excluded.items():
            if not baseline.issubset(restored[user_id]):
                raise ValueError("restored exclusions omit initial human history")
        self._excluded = restored

    def fingerprint(self) -> str:
        return policy_fingerprint(self.variant_id, self.export_state())


def build_exclude_seen_candidate_policy(
    *,
    catalog_item_ids: Sequence[ItemId],
    initial_histories: Mapping[UserId, Sequence[ItemId]],
    sponsored_item_ids: Sequence[ItemId] = (),
) -> CandidatePolicy:
    """Exclude history/clicks and reserve current or future campaign items."""

    return _ExcludeSeenCandidatePolicy(
        catalog_item_ids,
        initial_histories,
        sponsored_item_ids,
    )


__all__ = ["build_exclude_seen_candidate_policy"]
