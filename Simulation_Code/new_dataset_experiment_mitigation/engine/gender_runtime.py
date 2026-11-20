"""Local runtime wiring for explicit women-targeted interventions."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from ..gender_mitigation import GenderMitigationSpec, stable_promote_slate
from .policies._common import policy_fingerprint, rank_mapping
from .policies.contracts import SlateResult
from .policies.order_probe import _CanonicalReplicaShuffledProbePolicy
from .policies.presentation import _UnchangedOrderSlatePolicy


def resolved_gender_spec(settings: Mapping[str, Any], catalog_ids: tuple[int, ...]):
    raw = settings.get("gender_mitigation")
    if not isinstance(raw, Mapping):
        raise ValueError("every mitigation job requires an explicit gender_mitigation specification")
    allowed = {"strategy", "women_item_ids", "women_weight", "rank_promotion", "label_fingerprint"}
    if set(raw).difference(allowed):
        raise ValueError("unrecognised gender mitigation setting")
    spec = GenderMitigationSpec(
        intervention=raw.get("strategy", "control"),
        women_positive_weight=raw.get("women_weight", 1.25),
        women_rank_promotion=raw.get("rank_promotion", 5),
    )
    women = tuple(raw.get("women_item_ids", ()))
    if (any(isinstance(i, bool) or not isinstance(i, int) or i <= 0 for i in women)
        or len(women) != len(set(women)) or not set(women).issubset(catalog_ids)):
        raise ValueError("women_item_ids must be unique positive IDs in the frozen catalogue")
    fingerprint = raw.get("label_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ValueError("gender mitigation requires the fixed label fingerprint")
    return spec, frozenset(women), fingerprint


class GenderPromotionSlatePolicy:
    """Reorder the selected organic top-K; do not change slate membership."""
    variant_id = "women_stable_top_k_rank_promotion_v1"

    def __init__(self, slate_size: int, women_item_ids: frozenset[int], *,
                 promotion: int, label_fingerprint: str) -> None:
        self._organic = _UnchangedOrderSlatePolicy(slate_size)
        self._women = women_item_ids
        self._promotion = promotion
        self._label_fingerprint = label_fingerprint

    def present(self, user_id, iteration, recommender_items, *, sponsored_item_id=None):
        base = self._organic.present(user_id, iteration, recommender_items,
                                     sponsored_item_id=sponsored_item_id)
        displayed = stable_promote_slate(base.presented_items, self._women,
                                        promotion=self._promotion)
        return SlateResult(
            recommender_items=base.recommender_items, presented_items=displayed,
            recommender_rank_by_item=base.recommender_rank_by_item,
            displayed_rank_by_item=rank_mapping(displayed),
            metadata=MappingProxyType({
                **dict(base.metadata), "policy_variant": self.variant_id,
                "mitigation_strategy": "post_processing",
                "women_rank_promotion": self._promotion,
                "label_fingerprint": self._label_fingerprint,
                "slate_membership_preserved": True,
            }),
        )

    def export_state(self):
        return {"variant_id": self.variant_id,
                "slate_size": self._organic.export_state()["slate_size"],
                "women_item_ids": tuple(sorted(self._women)),
                "promotion": self._promotion, "label_fingerprint": self._label_fingerprint}

    def restore_state(self, state):
        if dict(state) != self.export_state():
            raise ValueError("gender promotion state/configuration mismatch")

    def fingerprint(self):
        return policy_fingerprint(self.variant_id, self.export_state())


class GenderPromotionProbePolicy(_CanonicalReplicaShuffledProbePolicy):
    """Preserve existing RNG streams and probe order around a promoted slate."""
    variant_id = "women_promotion_canonical_replica_shuffled_probe_v1"

    def __init__(self, slate_policy: GenderPromotionSlatePolicy, seed: int):
        super().__init__(slate_policy.export_state()["slate_size"], seed)
        self._organic = slate_policy

    def export_state(self):
        return {**super().export_state(), "gender_slate_policy": self._organic.export_state()}


__all__ = ["resolved_gender_spec", "GenderPromotionSlatePolicy", "GenderPromotionProbePolicy"]
