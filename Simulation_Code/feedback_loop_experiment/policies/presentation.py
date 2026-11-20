"""Organic and sponsored slate-presentation policies."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..contracts import ItemId, UserId
from ._common import (
    iteration as validate_iteration,
    policy_fingerprint,
    positive_id,
    rank_mapping,
    unique_ids,
)
from .contracts import SlatePolicy, SlateResult


class _UnchangedOrderSlatePolicy:
    variant_id = "unchanged_recommender_order_v1"

    def __init__(self, slate_size: int) -> None:
        self._slate_size = positive_id(slate_size, "slate_size")

    def present(
        self,
        user_id: UserId,
        iteration: int,
        recommender_items: Sequence[ItemId],
        *,
        sponsored_item_id: ItemId | None = None,
    ) -> SlateResult:
        positive_id(user_id, "user_id")
        validate_iteration(iteration)
        if sponsored_item_id is not None:
            raise ValueError("organic presentation cannot receive a sponsored item")
        ranking = unique_ids(recommender_items, "recommender_items")
        if len(ranking) < self._slate_size:
            raise ValueError("recommender returned fewer items than the slate size")
        slate = ranking[: self._slate_size]
        ranks = rank_mapping(slate)
        return SlateResult(
            recommender_items=slate,
            presented_items=slate,
            recommender_rank_by_item=ranks,
            displayed_rank_by_item=ranks,
            metadata=MappingProxyType(
                {"policy_variant": self.variant_id, "sponsored_item_id": None}
            ),
        )

    def export_state(self) -> Mapping[str, Any]:
        return {"variant_id": self.variant_id, "slate_size": self._slate_size}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if dict(state) != self.export_state():
            raise ValueError("unchanged-order policy state/configuration mismatch")

    def fingerprint(self) -> str:
        return policy_fingerprint(self.variant_id, self.export_state())


class _SponsoredRankOneSlatePolicy:
    variant_id = "sponsored_rank_1_plus_recommender_top_9_v1"
    supported_rule = "remove_sponsor_from_organic_top_k_v1"

    def __init__(self, slate_size: int, deduplication_rule_id: str) -> None:
        self._slate_size = positive_id(slate_size, "slate_size")
        if self._slate_size < 2:
            raise ValueError("a sponsored slate must contain at least two items")
        if deduplication_rule_id != self.supported_rule:
            raise ValueError(
                f"deduplication_rule_id must be {self.supported_rule!r}"
            )
        self._deduplication_rule_id = deduplication_rule_id
        self._organic_policy = _UnchangedOrderSlatePolicy(self._slate_size)

    def present(
        self,
        user_id: UserId,
        iteration: int,
        recommender_items: Sequence[ItemId],
        *,
        sponsored_item_id: ItemId | None = None,
    ) -> SlateResult:
        positive_id(user_id, "user_id")
        validate_iteration(iteration)
        if sponsored_item_id is None:
            organic = self._organic_policy.present(
                user_id, iteration, recommender_items
            )
            return SlateResult(
                recommender_items=organic.recommender_items,
                presented_items=organic.presented_items,
                recommender_rank_by_item=organic.recommender_rank_by_item,
                displayed_rank_by_item=organic.displayed_rank_by_item,
                metadata=MappingProxyType(
                    {
                        "policy_variant": self.variant_id,
                        "campaign_active": False,
                        "sponsored_item_id": None,
                        "displaced_item_id": None,
                        "deduplication_rule_id": self._deduplication_rule_id,
                    }
                ),
            )
        sponsor = positive_id(sponsored_item_id, "sponsored_item_id")
        ranking = unique_ids(recommender_items, "recommender_items")
        if len(ranking) < self._slate_size:
            raise ValueError("recommender returned fewer items than the slate size")

        organic_slate = ranking[: self._slate_size]
        organic_without_sponsor = tuple(
            item for item in organic_slate if item != sponsor
        )
        required_organic = self._slate_size - 1
        if len(organic_without_sponsor) < required_organic:
            raise ValueError("not enough unique organic items after sponsor de-duplication")
        presented = (sponsor,) + organic_without_sponsor[:required_organic]
        source_ranks = rank_mapping(organic_slate)
        sponsor_source_rank = source_ranks.get(sponsor)
        sponsor_full_source_rank = (
            ranking.index(sponsor) + 1 if sponsor in ranking else None
        )
        displaced = organic_slate[-1] if sponsor_source_rank is None else None
        return SlateResult(
            recommender_items=organic_slate,
            presented_items=presented,
            recommender_rank_by_item=source_ranks,
            displayed_rank_by_item=rank_mapping(presented),
            metadata=MappingProxyType(
                {
                    "policy_variant": self.variant_id,
                    "campaign_active": True,
                    "sponsored_item_id": sponsor,
                    "sponsored_displayed_rank": 1,
                    "sponsored_recommender_rank": sponsor_source_rank,
                    "sponsored_full_recommender_rank": sponsor_full_source_rank,
                    "sponsor_was_in_organic_top_k": sponsor_source_rank is not None,
                    "displaced_item_id": displaced,
                    "displaced_recommender_rank": (
                        self._slate_size if displaced is not None else None
                    ),
                    "deduplication_rule_id": self._deduplication_rule_id,
                }
            ),
        )

    def export_state(self) -> Mapping[str, Any]:
        return {
            "variant_id": self.variant_id,
            "slate_size": self._slate_size,
            "deduplication_rule_id": self._deduplication_rule_id,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if dict(state) != self.export_state():
            raise ValueError("sponsored-slate policy state/configuration mismatch")

    def fingerprint(self) -> str:
        return policy_fingerprint(self.variant_id, self.export_state())


def build_unchanged_recommender_order_policy(
    *, slate_size: int = 10
) -> SlatePolicy:
    """Build the thin top-k adapter that records identity rank mappings."""

    return _UnchangedOrderSlatePolicy(slate_size)


def build_sponsored_rank_one_slate_policy(
    *, slate_size: int = 10, deduplication_rule_id: str
) -> SlatePolicy:
    """Build rank-1 sponsor plus the remaining organic top-k items."""

    return _SponsoredRankOneSlatePolicy(slate_size, deduplication_rule_id)


__all__ = [
    "build_sponsored_rank_one_slate_policy",
    "build_unchanged_recommender_order_policy",
]
