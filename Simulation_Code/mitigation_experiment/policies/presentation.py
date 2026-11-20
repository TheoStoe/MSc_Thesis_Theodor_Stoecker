"""Organic and sponsored slate-presentation policies."""

from __future__ import annotations

import math
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


def rank_exposure_diagnostics(
    raw_items: Sequence[ItemId],
    displayed_items: Sequence[ItemId],
    head_item_ids: Sequence[ItemId],
) -> dict[str, Any]:
    """Describe a same-item ordering; head identity is fixed before simulation."""
    raw = tuple(raw_items)
    shown = tuple(displayed_items)
    if not raw or len(set(raw)) != len(raw) or set(raw) != set(shown) or len(raw) != len(shown):
        raise ValueError("order diagnostics require equal unique nonempty item sets")
    head = frozenset(head_item_ids)
    ranks = {item: index for index, item in enumerate(raw)}
    moved = sum(item != raw[index] for index, item in enumerate(shown))
    head_count = sum(item in head for item in raw)
    unchanged_reason = None
    if moved == 0:
        if head_count == len(raw):
            unchanged_reason = "all_head"
        elif head_count == 0:
            unchanged_reason = "all_tail"
        elif raw == tuple(sorted(raw, key=lambda item: item in head)):
            unchanged_reason = "tail_already_before_head"
        else:
            unchanged_reason = "zero_penalty_or_ties"
    top_k = min(3, len(raw))
    discounts = tuple(1.0 / math.log2(rank + 1) for rank in range(1, len(raw) + 1))
    total_discount = sum(discounts)
    attention = lambda items: sum(weight for item, weight in zip(items, discounts) if item in head) / total_discount
    return {
        "changed": bool(moved),
        "moved_item_count": moved,
        "mean_absolute_rank_displacement": sum(abs(ranks[item] - index) for index, item in enumerate(shown)) / len(raw),
        "head_count": head_count,
        "head_share": head_count / len(raw),
        "first_item_changed": raw[0] != shown[0],
        "top_three_changed": raw[:top_k] != shown[:top_k],
        "raw_top_three_head_share": sum(item in head for item in raw[:top_k]) / top_k,
        "displayed_top_three_head_share": sum(item in head for item in shown[:top_k]) / top_k,
        "raw_discounted_head_exposure": attention(raw),
        "displayed_discounted_head_exposure": attention(shown),
        "unchanged_reason": unchanged_reason,
    }


class _PriorHeadRankPenaltyPolicy:
    variant_id = "mitigation_prior_head_rank_penalty_v1"

    def __init__(
        self, head_item_ids: Sequence[ItemId], penalty: int, slate_size: int,
        head_definition_fingerprint: str | None,
    ) -> None:
        self._organic = _UnchangedOrderSlatePolicy(slate_size)
        if isinstance(penalty, bool) or not isinstance(penalty, int) or penalty < 0:
            raise ValueError("penalty must be a non-negative integer")
        self._head_ids = tuple(sorted(unique_ids(head_item_ids, "head_item_ids")))
        self._head = frozenset(self._head_ids)
        self._penalty = penalty
        if head_definition_fingerprint is not None and (
            not isinstance(head_definition_fingerprint, str) or not head_definition_fingerprint.strip()
        ):
            raise ValueError("head_definition_fingerprint must be nonblank or None")
        self._head_fingerprint = head_definition_fingerprint or policy_fingerprint(
            "mitigation_prior_head_items_v1", {"head_item_ids": self._head_ids}
        )

    def present(
        self, user_id: UserId, iteration: int, recommender_items: Sequence[ItemId],
        *, sponsored_item_id: ItemId | None = None,
    ) -> SlateResult:
        raw = self._organic.present(user_id, iteration, recommender_items, sponsored_item_id=sponsored_item_id)
        ranks = raw.recommender_rank_by_item
        canonical = tuple(sorted(raw.recommender_items, key=lambda item: (ranks[item] + self._penalty * (item in self._head), ranks[item])))
        diagnostics = rank_exposure_diagnostics(raw.recommender_items, canonical, self._head_ids)
        return SlateResult(
            recommender_items=raw.recommender_items,
            presented_items=canonical,
            recommender_rank_by_item=ranks,
            displayed_rank_by_item=rank_mapping(canonical),
            metadata=MappingProxyType({
                "policy_variant": self.variant_id,
                "sponsored_item_id": None,
                "head_rank_penalty": self._penalty,
                "head_definition_fingerprint": self._head_fingerprint,
                "head_item_ids_in_slate": tuple(item for item in raw.recommender_items if item in self._head),
                "raw_order": raw.recommender_items,
                "canonical_order": canonical,
                "canonical_order_diagnostics": diagnostics,
                "displayed_order_diagnostics": diagnostics,
            }),
        )

    def export_state(self) -> Mapping[str, Any]:
        return {
            "variant_id": self.variant_id,
            "slate_size": self._organic.export_state()["slate_size"],
            "head_item_ids": self._head_ids,
            "penalty": self._penalty,
            "head_definition_fingerprint": self._head_fingerprint,
            "tie_rule": "original_rank_stable_v1",
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if dict(state) != self.export_state():
            raise ValueError("head-rank policy state/configuration mismatch")

    def fingerprint(self) -> str:
        return policy_fingerprint(self.variant_id, self.export_state())


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


def build_prior_head_rank_penalty_policy(
    *, head_item_ids: Sequence[ItemId], penalty: int = 5, slate_size: int = 10,
    head_definition_fingerprint: str | None = None,
) -> SlatePolicy:
    """Demote head items within the existing slate; zero is an exact control."""
    return _PriorHeadRankPenaltyPolicy(head_item_ids, penalty, slate_size, head_definition_fingerprint)


def build_sponsored_rank_one_slate_policy(
    *, slate_size: int = 10, deduplication_rule_id: str
) -> SlatePolicy:
    """Build rank-1 sponsor plus the remaining organic top-k items."""

    return _SponsoredRankOneSlatePolicy(slate_size, deduplication_rule_id)


__all__ = [
    "build_prior_head_rank_penalty_policy",
    "build_sponsored_rank_one_slate_policy",
    "build_unchanged_recommender_order_policy",
]
