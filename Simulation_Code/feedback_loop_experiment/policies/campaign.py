"""O0, O1, and outcome-adaptive O2 advertisement campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..contracts import ActionStatus, ItemId, UserAction, UserId
from ..utility.fingerprints import fingerprint_semantic_state
from ._common import iteration as validate_iteration
from ._common import positive_id, unique_ids
from .contracts import CampaignPolicy, SponsoredOutcome


def _fingerprint(policy: CampaignPolicy) -> str:
    return fingerprint_semantic_state(
        {"variant_id": policy.variant_id, "state": policy.export_state()},
        "campaign-policy-state-v1",
    )


def _scheduled_item(
    sponsored_item_ids: tuple[ItemId, ...], iteration: int
) -> ItemId:
    checked_iteration = validate_iteration(iteration)
    if checked_iteration > len(sponsored_item_ids):
        raise ValueError("iteration exceeds the sponsored-item schedule")
    return sponsored_item_ids[checked_iteration - 1]


def _validate_outcomes(
    iteration: int,
    outcomes: Sequence[SponsoredOutcome],
    sponsored_item_id: ItemId,
) -> tuple[SponsoredOutcome, ...]:
    checked_iteration = validate_iteration(iteration)
    checked: list[SponsoredOutcome] = []
    seen_users: set[UserId] = set()
    for outcome in outcomes:
        if not isinstance(outcome, SponsoredOutcome):
            raise TypeError("outcomes must contain SponsoredOutcome values")
        user_id = positive_id(outcome.user_id, "outcome user_id")
        if user_id in seen_users:
            raise ValueError(
                "a user may have at most one sponsored outcome per iteration"
            )
        seen_users.add(user_id)
        if outcome.iteration != checked_iteration:
            raise ValueError("sponsored outcome iteration does not match the batch")
        if outcome.sponsored_item_id != sponsored_item_id:
            raise ValueError("sponsored outcome names a different scheduled item")
        if not isinstance(outcome.action_status, ActionStatus):
            raise TypeError(
                "sponsored outcome action_status must be an ActionStatus"
            )
        if not isinstance(outcome.action, UserAction):
            raise TypeError("sponsored outcome action must be a UserAction")

        if outcome.action_status is ActionStatus.VALID:
            if outcome.action is UserAction.CLICK:
                positive_id(outcome.clicked_item_id, "clicked_item_id")
            elif outcome.action is UserAction.NO_CLICK:
                if outcome.clicked_item_id is not None:
                    raise ValueError("a valid no-click cannot name a clicked item")
            else:
                raise ValueError(
                    "a valid sponsored outcome must be click or no-click"
                )
        elif (
            outcome.action is not UserAction.UNAVAILABLE
            or outcome.clicked_item_id is not None
        ):
            raise ValueError(
                "a non-valid sponsored outcome must use unavailable with no item"
            )
        checked.append(outcome)
    return tuple(checked)


class _NoAdCampaignPolicy:
    variant_id = "no_campaign_v1"

    def sponsored_item_for(self, user_id: UserId, iteration: int) -> None:
        positive_id(user_id, "user_id")
        validate_iteration(iteration)
        return None

    def observe_iteration(
        self, iteration: int, outcomes: Sequence[SponsoredOutcome]
    ) -> None:
        validate_iteration(iteration)
        if outcomes:
            raise ValueError("a no-campaign condition cannot have sponsored outcomes")

    def export_state(self) -> Mapping[str, Any]:
        return {}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state:
            raise ValueError("no-campaign state must be an empty mapping")

    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True)
class _StaticSponsorCampaignPolicy:
    sponsored_item_ids: tuple[ItemId, ...]
    sponsor_artifact_fingerprint: str
    variant_id: str = "static_sponsor_schedule_v2"

    def sponsored_item_for(self, user_id: UserId, iteration: int) -> ItemId:
        positive_id(user_id, "user_id")
        return _scheduled_item(self.sponsored_item_ids, iteration)

    def observe_iteration(
        self, iteration: int, outcomes: Sequence[SponsoredOutcome]
    ) -> None:
        _validate_outcomes(
            iteration,
            outcomes,
            _scheduled_item(self.sponsored_item_ids, iteration),
        )

    def export_state(self) -> Mapping[str, Any]:
        return {
            "sponsored_item_ids": self.sponsored_item_ids,
            "sponsor_artifact_fingerprint": self.sponsor_artifact_fingerprint,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or dict(state) != self.export_state():
            raise ValueError("static campaign state does not match its configuration")

    def fingerprint(self) -> str:
        return _fingerprint(self)


class _PerUserThreeStrikeCampaignPolicy:
    variant_id = "per_user_three_strike_sponsor_schedule_v2"

    def __init__(
        self,
        *,
        sponsored_item_ids: tuple[ItemId, ...],
        sponsor_artifact_fingerprint: str,
        threshold: int = 3,
    ) -> None:
        self.sponsored_item_ids = sponsored_item_ids
        self.sponsor_artifact_fingerprint = sponsor_artifact_fingerprint
        self.threshold = threshold
        self._non_click_counts: dict[UserId, int] = {}
        self._withdrawal_effective_iteration: dict[UserId, int] = {}

    def sponsored_item_for(
        self, user_id: UserId, iteration: int
    ) -> ItemId | None:
        checked_user_id = positive_id(user_id, "user_id")
        checked_iteration = validate_iteration(iteration)
        scheduled_item = _scheduled_item(self.sponsored_item_ids, checked_iteration)
        effective_iteration = self._withdrawal_effective_iteration.get(
            checked_user_id
        )
        if (
            effective_iteration is not None
            and checked_iteration >= effective_iteration
        ):
            return None
        return scheduled_item

    def observe_iteration(
        self, iteration: int, outcomes: Sequence[SponsoredOutcome]
    ) -> None:
        checked_iteration = validate_iteration(iteration)
        scheduled_item = _scheduled_item(self.sponsored_item_ids, checked_iteration)
        checked_outcomes = _validate_outcomes(
            checked_iteration,
            outcomes,
            scheduled_item,
        )
        next_counts = dict(self._non_click_counts)
        next_withdrawals = dict(self._withdrawal_effective_iteration)
        for outcome in checked_outcomes:
            if self.sponsored_item_for(outcome.user_id, checked_iteration) is None:
                raise ValueError("a withdrawn user cannot have a sponsored outcome")
            if outcome.action_status is not ActionStatus.VALID:
                continue
            if (
                outcome.action is UserAction.CLICK
                and outcome.clicked_item_id == scheduled_item
            ):
                next_counts.pop(outcome.user_id, None)
                continue

            count = next_counts.get(outcome.user_id, 0) + 1
            next_counts[outcome.user_id] = min(count, self.threshold)
            if count >= self.threshold:
                next_withdrawals[outcome.user_id] = checked_iteration + 1

        self._non_click_counts = next_counts
        self._withdrawal_effective_iteration = next_withdrawals

    def export_state(self) -> Mapping[str, Any]:
        return {
            "sponsored_item_ids": self.sponsored_item_ids,
            "sponsor_artifact_fingerprint": self.sponsor_artifact_fingerprint,
            "threshold": self.threshold,
            "consecutive_sponsored_non_clicks": dict(
                sorted(self._non_click_counts.items())
            ),
            "withdrawal_effective_iteration": dict(
                sorted(self._withdrawal_effective_iteration.items())
            ),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("adaptive campaign state must be a mapping")
        if tuple(state.get("sponsored_item_ids", ())) != self.sponsored_item_ids:
            raise ValueError("adaptive campaign sponsored schedule does not match")
        if (
            state.get("sponsor_artifact_fingerprint")
            != self.sponsor_artifact_fingerprint
        ):
            raise ValueError("adaptive campaign sponsor artifact does not match")
        if state.get("threshold") != self.threshold:
            raise ValueError("adaptive campaign threshold does not match")
        counts = state.get("consecutive_sponsored_non_clicks")
        withdrawals = state.get("withdrawal_effective_iteration")
        if not isinstance(counts, Mapping) or not isinstance(withdrawals, Mapping):
            raise ValueError("adaptive campaign counters must be mappings")

        restored_counts: dict[UserId, int] = {}
        for user_id, count in counts.items():
            checked_user_id = positive_id(user_id, "counter user_id")
            checked_count = positive_id(count, "counter value")
            if checked_count > self.threshold:
                raise ValueError("campaign counter exceeds its threshold")
            restored_counts[checked_user_id] = checked_count

        restored_withdrawals: dict[UserId, int] = {}
        for user_id, effective_iteration in withdrawals.items():
            checked_user_id = positive_id(user_id, "withdrawal user_id")
            restored_withdrawals[checked_user_id] = validate_iteration(
                effective_iteration
            )
        threshold_users = {
            user_id
            for user_id, count in restored_counts.items()
            if count == self.threshold
        }
        if set(restored_withdrawals) != threshold_users:
            raise ValueError(
                "withdrawn users must be exactly those at the campaign threshold"
            )
        self._non_click_counts = restored_counts
        self._withdrawal_effective_iteration = restored_withdrawals

    def fingerprint(self) -> str:
        return _fingerprint(self)


def _validate_configuration(
    sponsored_item_ids: Sequence[ItemId],
    sponsor_artifact_fingerprint: str,
) -> tuple[tuple[ItemId, ...], str]:
    checked_items = unique_ids(sponsored_item_ids, "sponsored_item_ids")
    if (
        not isinstance(sponsor_artifact_fingerprint, str)
        or not sponsor_artifact_fingerprint.strip()
    ):
        raise ValueError("sponsor_artifact_fingerprint must be a non-blank string")
    return checked_items, sponsor_artifact_fingerprint


def build_no_ad_campaign_policy() -> CampaignPolicy:
    """Build the O0 no-ad condition with no outcome state."""

    return _NoAdCampaignPolicy()


def build_static_sponsor_campaign_policy(
    *, sponsored_item_ids: Sequence[ItemId], sponsor_artifact_fingerprint: str
) -> CampaignPolicy:
    """Build O1: scheduled sponsor delivery that ignores outcomes."""

    item_ids, artifact_fingerprint = _validate_configuration(
        sponsored_item_ids, sponsor_artifact_fingerprint
    )
    return _StaticSponsorCampaignPolicy(item_ids, artifact_fingerprint)


def build_per_user_three_strike_campaign_policy(
    *,
    sponsored_item_ids: Sequence[ItemId],
    sponsor_artifact_fingerprint: str,
    threshold: int = 3,
) -> CampaignPolicy:
    """Build O2: withdraw scheduled delivery after valid sponsored non-clicks."""

    item_ids, artifact_fingerprint = _validate_configuration(
        sponsored_item_ids, sponsor_artifact_fingerprint
    )
    checked_threshold = positive_id(threshold, "threshold")
    return _PerUserThreeStrikeCampaignPolicy(
        sponsored_item_ids=item_ids,
        sponsor_artifact_fingerprint=artifact_fingerprint,
        threshold=checked_threshold,
    )


__all__ = [
    "build_no_ad_campaign_policy",
    "build_per_user_three_strike_campaign_policy",
    "build_static_sponsor_campaign_policy",
]
