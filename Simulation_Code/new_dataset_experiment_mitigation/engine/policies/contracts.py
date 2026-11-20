"""Public data and protocol contracts shared by policy implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ..contracts import ActionDecision, ActionStatus, ItemId, UserAction, UserId


@dataclass(frozen=True)
class SlateResult:
    """Presented items plus the mapping needed to audit order interventions."""

    recommender_items: tuple[ItemId, ...]
    presented_items: tuple[ItemId, ...]
    recommender_rank_by_item: Mapping[ItemId, int]
    displayed_rank_by_item: Mapping[ItemId, int]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PopulationTransitionResult:
    """Transitions applied at the start of one simulation iteration."""

    iteration: int
    exited_user_ids: tuple[UserId, ...]
    activated_user_ids: tuple[UserId, ...]
    active_user_ids: tuple[UserId, ...]
    risk_set_fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SponsoredOutcome:
    """One sponsored impression outcome used by an adaptive campaign."""

    user_id: UserId
    iteration: int
    sponsored_item_id: ItemId
    action_status: ActionStatus
    action: UserAction
    clicked_item_id: ItemId | None


class CandidatePolicy(Protocol):
    """Owns item eligibility and exclusion state, not recommender scoring."""

    variant_id: str

    def eligible_items(self, user_id: UserId, iteration: int) -> Sequence[ItemId]: ...

    def observe_valid_click(self, user_id: UserId, action: ActionDecision) -> None: ...

    def export_state(self) -> Mapping[str, Any]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...

    def fingerprint(self) -> str: ...


class SlatePolicy(Protocol):
    """Transforms the recommender ranking while retaining its original ranks."""

    variant_id: str

    def present(
        self,
        user_id: UserId,
        iteration: int,
        recommender_items: Sequence[ItemId],
        *,
        sponsored_item_id: ItemId | None = None,
    ) -> SlateResult: ...

    def export_state(self) -> Mapping[str, Any]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...

    def fingerprint(self) -> str: ...


class PopulationPolicy(Protocol):
    """Owns active membership, delayed exits, and optional reserve entry."""

    variant_id: str

    def apply_scheduled_transitions(
        self, iteration: int
    ) -> PopulationTransitionResult: ...

    def schedule_exit(self, user_id: UserId, effective_iteration: int) -> None: ...

    def export_state(self) -> Mapping[str, Any]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...

    def fingerprint(self) -> str: ...


class CampaignPolicy(Protocol):
    """Owns sponsored eligibility and outcome-dependent campaign state."""

    variant_id: str

    def sponsored_item_for(
        self, user_id: UserId, iteration: int
    ) -> ItemId | None: ...

    def observe_iteration(
        self, iteration: int, outcomes: Sequence[SponsoredOutcome]
    ) -> None: ...

    def export_state(self) -> Mapping[str, Any]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...

    def fingerprint(self) -> str: ...


@dataclass(frozen=True)
class OrderProbeSlateResult:
    """Canonical, exact-replica, and shuffled slates for reliability analysis."""

    canonical: SlateResult
    replica: SlateResult
    shuffled: SlateResult
    provider_call_order: tuple[str, str, str]


class OrderProbePolicy(Protocol):
    """Builds paired order-test slates without calling a response model."""

    variant_id: str

    def present_variants(
        self,
        user_id: UserId,
        iteration: int,
        recommender_items: Sequence[ItemId],
    ) -> OrderProbeSlateResult: ...

    def export_state(self) -> Mapping[str, Any]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...

    def fingerprint(self) -> str: ...


__all__ = [
    "CampaignPolicy",
    "CandidatePolicy",
    "OrderProbePolicy",
    "OrderProbeSlateResult",
    "PopulationPolicy",
    "PopulationTransitionResult",
    "SlatePolicy",
    "SlateResult",
    "SponsoredOutcome",
]
