"""Canonical, replica, and shuffled order-probe policy."""

from __future__ import annotations

from itertools import permutations
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..contracts import ItemId, UserId
from ..utility.randomness import RandomKey, random_generator
from ._common import (
    iteration as validate_iteration,
    policy_fingerprint,
    positive_id,
    rank_mapping,
)
from .contracts import OrderProbePolicy, OrderProbeSlateResult, SlateResult
from .presentation import _UnchangedOrderSlatePolicy


class _CanonicalReplicaShuffledProbePolicy:
    variant_id = "canonical_replica_and_shuffled_probe_v1"
    _SHUFFLE_STREAM = 2345
    _arms = ("canonical", "replica", "shuffled")
    _call_orders = tuple(permutations(_arms))

    def __init__(self, slate_size: int, seed: int) -> None:
        self._organic = _UnchangedOrderSlatePolicy(slate_size)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self._seed = seed

    @staticmethod
    def _with_arm(result: SlateResult, arm: str) -> SlateResult:
        # Preserve the exact slate while recording which non-persistent probe
        # produced the response in the normalized event metadata.
        return SlateResult(
            recommender_items=result.recommender_items,
            presented_items=result.presented_items,
            recommender_rank_by_item=result.recommender_rank_by_item,
            displayed_rank_by_item=result.displayed_rank_by_item,
            metadata=MappingProxyType({**dict(result.metadata), "probe_arm": arm}),
        )

    def present_variants(
        self,
        user_id: UserId,
        iteration: int,
        recommender_items: Sequence[ItemId],
    ) -> OrderProbeSlateResult:
        user_id = positive_id(user_id, "user_id")
        iteration = validate_iteration(iteration)
        base = self._organic.present(user_id, iteration, recommender_items)
        canonical = self._with_arm(base, "canonical")
        replica = self._with_arm(base, "replica")

        generator = random_generator(
            self._seed,
            RandomKey(
                stream_id=self._SHUFFLE_STREAM,
                user_id=user_id,
                iteration=iteration,
            ),
        )
        indices = generator.permutation(len(base.presented_items))
        shuffled_items = tuple(base.presented_items[int(index)] for index in indices)
        if len(shuffled_items) > 1 and shuffled_items == base.presented_items:
            shuffled_items = shuffled_items[1:] + shuffled_items[:1]
        shuffled = SlateResult(
            recommender_items=base.recommender_items,
            presented_items=shuffled_items,
            recommender_rank_by_item=base.recommender_rank_by_item,
            displayed_rank_by_item=rank_mapping(shuffled_items),
            metadata=MappingProxyType(
                {
                    **dict(base.metadata),
                    "probe_arm": "shuffled",
                    "source_rank_by_displayed_position": tuple(
                        base.recommender_rank_by_item[item_id]
                        for item_id in shuffled_items
                    ),
                }
            ),
        )

        order_index = (iteration - 1) % len(self._call_orders)
        return OrderProbeSlateResult(
            canonical=canonical,
            replica=replica,
            shuffled=shuffled,
            provider_call_order=self._call_orders[order_index],
        )

    def export_state(self) -> Mapping[str, Any]:
        return {
            "variant_id": self.variant_id,
            "slate_size": self._organic.export_state()["slate_size"],
            "seed": self._seed,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if dict(state) != self.export_state():
            raise ValueError("order-probe policy state/configuration mismatch")

    def fingerprint(self) -> str:
        return policy_fingerprint(self.variant_id, self.export_state())


def build_canonical_replica_and_shuffled_probe_policy(
    *, slate_size: int = 10, seed: int
) -> OrderProbePolicy:
    """Build equal-state canonical, replica, and shuffled presentations."""

    return _CanonicalReplicaShuffledProbePolicy(slate_size, seed)


__all__ = ["build_canonical_replica_and_shuffled_probe_policy"]
