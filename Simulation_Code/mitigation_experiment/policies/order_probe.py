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
from .presentation import _UnchangedOrderSlatePolicy, rank_exposure_diagnostics


class _CanonicalReplicaShuffledProbePolicy:
    variant_id = "mitigation_composed_order_probes_v1"
    _SHUFFLE_STREAM = 2345

    def __init__(self, slate_size: int, seed: int, include_original_order_probe: bool) -> None:
        self._organic = _UnchangedOrderSlatePolicy(slate_size)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self._seed = seed
        if not isinstance(include_original_order_probe, bool):
            raise TypeError("include_original_order_probe must be boolean")
        self._arms = ("canonical", "replica", "shuffled") + (("original_order_probe",) if include_original_order_probe else ())
        # Reuse the previous experiment's permutation cycle for all roles.
        self._call_orders = tuple(permutations(self._arms))
        self._call_schedule_version = (
            "four_arm_all_permutations_v1" if include_original_order_probe
            else "legacy_three_arm_six_permutations_v1"
        )

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
        *,
        canonical_slate: SlateResult | None = None,
    ) -> OrderProbeSlateResult:
        user_id = positive_id(user_id, "user_id")
        iteration = validate_iteration(iteration)
        raw = self._organic.present(user_id, iteration, recommender_items)
        base = canonical_slate or raw
        if (
            base.recommender_items != raw.recommender_items
            or dict(base.recommender_rank_by_item) != dict(raw.recommender_rank_by_item)
            or len(base.presented_items) != len(raw.presented_items)
            or set(base.presented_items) != set(raw.presented_items)
            or dict(base.displayed_rank_by_item) != dict(rank_mapping(base.presented_items))
        ):
            raise ValueError("canonical presentation must retain the exact raw item set and ranks")
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
                    "canonical_order": base.presented_items,
                    "source_canonical_rank_by_displayed_position": tuple(
                        base.displayed_rank_by_item[item_id] for item_id in shuffled_items
                    ),
                    "source_rank_by_displayed_position": tuple(
                        base.recommender_rank_by_item[item_id]
                        for item_id in shuffled_items
                    ),
                }
            ),
        )
        slates = {"canonical": canonical, "replica": replica, "shuffled": shuffled}
        if "original_order_probe" in self._arms:
            slates["original_order_probe"] = SlateResult(
                recommender_items=raw.recommender_items,
                presented_items=raw.presented_items,
                recommender_rank_by_item=raw.recommender_rank_by_item,
                displayed_rank_by_item=raw.displayed_rank_by_item,
                metadata=MappingProxyType({
                    **dict(base.metadata), "probe_arm": "original_order_probe",
                    "canonical_order": base.presented_items,
                    "probe_interpretation": "raw_order_given_current_treated_state",
                }),
            )
        # Distinguish canonical intervention effect from the displayed probe.
        for arm, slate in tuple(slates.items()):
            metadata = {**dict(slate.metadata), "canonical_order": base.presented_items,
                        "raw_order": raw.presented_items,
                        "call_schedule_version": self._call_schedule_version}
            if "head_item_ids_in_slate" in base.metadata:
                metadata["displayed_order_diagnostics"] = rank_exposure_diagnostics(
                    raw.presented_items, slate.presented_items,
                    base.metadata["head_item_ids_in_slate"],
                )
            slates[arm] = SlateResult(
                recommender_items=slate.recommender_items, presented_items=slate.presented_items,
                recommender_rank_by_item=slate.recommender_rank_by_item,
                displayed_rank_by_item=slate.displayed_rank_by_item,
                metadata=MappingProxyType(metadata),
            )
        order_index = (iteration - 1) % len(self._call_orders)
        return OrderProbeSlateResult(
            slates=MappingProxyType(slates),
            provider_call_order=self._call_orders[order_index],
        )

    def export_state(self) -> Mapping[str, Any]:
        return {
            "variant_id": self.variant_id,
            "slate_size": self._organic.export_state()["slate_size"],
            "seed": self._seed,
            "arms": self._arms,
            "call_schedule_version": self._call_schedule_version,
            "call_orders": self._call_orders,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if dict(state) != self.export_state():
            raise ValueError("order-probe policy state/configuration mismatch")

    def fingerprint(self) -> str:
        return policy_fingerprint(self.variant_id, self.export_state())


def build_canonical_replica_and_shuffled_probe_policy(
    *, slate_size: int = 10, seed: int, include_original_order_probe: bool = False
) -> OrderProbePolicy:
    """Build equal-state canonical, replica, and shuffled presentations."""

    return _CanonicalReplicaShuffledProbePolicy(slate_size, seed, include_original_order_probe)


__all__ = ["build_canonical_replica_and_shuffled_probe_policy"]
