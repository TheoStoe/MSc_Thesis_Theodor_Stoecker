"""One user opportunity inside the feedback-loop simulation."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .contracts import (
    ActionDecision,
    ActionStatus,
    Continuation,
    ContinuationDecision,
    ContinuationStatus,
    TurnEvent,
    TurnInput,
    UpdateStatus,
    UserAction,
    UserUpdateResult,
)
from .experiment_methodology import ExperimentCondition
from .feedback_loops import FEEDBACK_LOOP_PRESETS
from .policies import OrderProbeSlateResult, SlateResult, SponsoredOutcome
from .runtime_setup import SimulationRuntime
from .utility.saving import RunOutputWriter


logger = logging.getLogger(__name__)


def run_user_opportunity(
    state: SimulationRuntime,
    writer: RunOutputWriter,
    condition: ExperimentCondition,
    *,
    run_id: str,
    seed: int,
    iteration: int,
    user_id: int,
    recommender_items: Sequence[int],
) -> SponsoredOutcome | None:
    """Show one slate, collect responses, apply canonical mutations, and save."""

    logger.info("User opportunity started iteration=%s user=%s", iteration, user_id)
    sponsor = state.campaign_policy.sponsored_item_for(user_id, iteration)
    slates, call_order = _presentations(
        state, user_id, iteration, recommender_items, sponsor
    )
    turns = {
        arm: _turn(
            run_id,
            condition,
            seed,
            iteration,
            user_id,
            arm,
            slate,
            state.observed_histories[user_id],
            call_order,
        )
        for arm, slate in slates.items()
    }
    actions: dict[str, ActionDecision] = {}
    for arm in call_order: #canonical replica shuffling in random order
        action = state.user_model.respond(turns[arm])
        _validate_action(action, turns[arm])
        actions[arm] = action
        logger.info(
            "Response validated interaction=%s status=%s action=%s clicked_item_id=%s",
            turns[arm].interaction_id, action.status.value, action.action.value,
            action.clicked_item_id,
        )
        if action.status is not ActionStatus.VALID:
            logger.warning(
                "Non-valid response for %s: %s",
                turns[arm].interaction_id,
                action.status.value,
            )

    turn = turns["canonical"]
    action = actions["canonical"]
    preset = FEEDBACK_LOOP_PRESETS[condition.feedback_loop]
    update: UserUpdateResult | None = None
    exit_at: int | None = None

    valid_click = (
        action.status is ActionStatus.VALID
        and action.action is UserAction.CLICK
    )
    if valid_click:
        clicked = action.clicked_item_id
        assert clicked is not None
        if preset.exclude_clicked_items:
            state.candidate_policy.observe_valid_click(user_id, action)
        if preset.update_user_history:
            state.observed_histories[user_id].append(clicked)
            if user_id in state.recommender_training_histories:
                state.recommender_training_histories[user_id].append(clicked)
            else:
                logger.warning("User %s is outside the recommender fit subset", user_id)

    if (
        action.status is ActionStatus.VALID
        and condition.user_update_policy != "frozen"
    ):
        logger.info("User update started interaction=%s", turn.interaction_id)
        update = state.user_model.update(action, {"turn": turn})
        if not isinstance(update, UserUpdateResult):
            raise TypeError("user model update() must return UserUpdateResult")

    continuation = _not_requested(turn.interaction_id)
    update_failed = update is not None and update.status in {
        UpdateStatus.INVALID,
        UpdateStatus.TECHNICAL_FAILURE,
    }
    if preset.allow_user_exit and action.status is ActionStatus.VALID:
        if update_failed:
            logger.warning(
                "Skipping continuation after failed update for %s",
                turn.interaction_id,
            )
        else:
            continuation = state.user_model.decide_continuation(
                {"turn": turn, "interaction_id": turn.interaction_id}
            )
            _validate_continuation(continuation)
            if (
                continuation.status is ContinuationStatus.VALID
                and continuation.continuation is Continuation.EXIT
            ):
                exit_at = iteration + 1
                state.population_policy.schedule_exit(user_id, exit_at)
            elif continuation.status is not ContinuationStatus.VALID:
                logger.warning(
                    "Non-valid continuation defaults to stay for %s",
                    turn.interaction_id,
                )

    for arm, arm_turn in turns.items():
        canonical = arm == "canonical"
        logger.info("Saving turn event interaction=%s", arm_turn.interaction_id)
        writer.write_event(
            TurnEvent(
                run_id=run_id,
                condition_id=condition.condition_id,
                seed=seed,
                interaction_id=arm_turn.interaction_id,
                user_id=user_id,
                iteration=iteration,
                recommender_slate=arm_turn.recommender_slate,
                presented_slate=arm_turn.presented_slate,
                recommender_rank_by_item=arm_turn.recommender_rank_by_item,
                displayed_rank_by_item=arm_turn.displayed_rank_by_item,
                action=actions[arm],
                user_update=update if canonical else None,
                continuation=(
                    continuation
                    if canonical
                    else _not_requested(arm_turn.interaction_id)
                ),
                scheduled_exit_effective_iteration=exit_at if canonical else None,
                presentation_metadata={
                    **dict(slates[arm].metadata),
                    "probe_arm": arm,
                    "provider_call_order": call_order,
                },
            )
        )
        logger.info("Turn event saved interaction=%s", arm_turn.interaction_id)

    delivered_sponsor = slates["canonical"].metadata.get("sponsored_item_id")
    if delivered_sponsor is None:
        return None
    return SponsoredOutcome(
        user_id=user_id,
        iteration=iteration,
        sponsored_item_id=int(delivered_sponsor),
        action_status=action.status,
        action=action.action,
        clicked_item_id=action.clicked_item_id,
    )


def _presentations(
    state: SimulationRuntime,
    user_id: int,
    iteration: int,
    items: Sequence[int],
    sponsor: int | None,
) -> tuple[dict[str, SlateResult], tuple[str, ...]]:
    if state.order_probe_policy is None:
        slate = state.slate_policy.present(
            user_id, iteration, items, sponsored_item_id=sponsor
        )
        return {"canonical": slate}, ("canonical",)
    if sponsor is not None:
        raise ValueError("order probes cannot combine with sponsorship")
    probes: OrderProbeSlateResult = state.order_probe_policy.present_variants(
        user_id, iteration, items
    )
    return {
        "canonical": probes.canonical,
        "replica": probes.replica,
        "shuffled": probes.shuffled,
    }, tuple(probes.provider_call_order)


def _turn(
    run_id: str,
    condition: ExperimentCondition,
    seed: int,
    iteration: int,
    user_id: int,
    arm: str,
    slate: SlateResult,
    history: Sequence[int],
    call_order: Sequence[str],
) -> TurnInput:
    interaction_id = f"{run_id}__t{iteration:04d}__u{user_id}__{arm}"
    return TurnInput(
        run_id=run_id,
        condition_id=condition.condition_id,
        seed=seed,
        interaction_id=interaction_id,
        user_id=user_id,
        iteration=iteration,
        recommender_slate=tuple(slate.recommender_items),
        presented_slate=tuple(slate.presented_items),
        recommender_rank_by_item=dict(slate.recommender_rank_by_item),
        displayed_rank_by_item=dict(slate.displayed_rank_by_item),
        observed_history=tuple(history),
        context={"probe_arm": arm, "provider_call_order": tuple(call_order)},
    )


def _validate_action(action: ActionDecision, turn: TurnInput) -> None:
    if not isinstance(action, ActionDecision):
        raise TypeError("respond() must return ActionDecision")
    if action.status is ActionStatus.VALID:
        if (
            action.action is UserAction.CLICK
            and action.clicked_item_id not in turn.presented_slate
        ):
            raise ValueError("valid click must name one displayed item")
        if (
            action.action is UserAction.NO_CLICK
            and action.clicked_item_id is not None
        ):
            raise ValueError("valid no-click cannot name an item")
        if action.action is UserAction.UNAVAILABLE:
            raise ValueError("valid action cannot be unavailable")
    elif (
        action.action is not UserAction.UNAVAILABLE
        or action.clicked_item_id is not None
    ):
        raise ValueError("non-valid action must be unavailable with no item")


def _validate_continuation(decision: ContinuationDecision) -> None:
    if not isinstance(decision, ContinuationDecision):
        raise TypeError("decide_continuation() must return ContinuationDecision")
    if decision.status is ContinuationStatus.VALID:
        if decision.continuation not in {Continuation.STAY, Continuation.EXIT}:
            raise ValueError("valid continuation must be stay or exit")
    elif decision.continuation is not Continuation.UNAVAILABLE:
        raise ValueError("non-valid continuation must be unavailable")


def _not_requested(interaction_id: str) -> ContinuationDecision:
    return ContinuationDecision(
        decision_id=f"{interaction_id}__continuation_not_requested",
        status=ContinuationStatus.NOT_REQUESTED,
        continuation=Continuation.UNAVAILABLE,
    )


__all__ = ["run_user_opportunity"]
