"""Call-order-independent Mansoury rank-biased forced-click baseline.
source https://dl.acm.org/doi/10.1145/3340531.3412152
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from ..contracts import (
    ActionDecision,
    ActionStatus,
    Continuation,
    ContinuationDecision,
    ContinuationStatus,
    TurnInput,
    UpdateStatus,
    UserAction,
    UserUpdateResult,
)
from ..utility.fingerprints import fingerprint_semantic_state
from ..utility.randomness import RandomKey, random_generator
from .base import UserModelCapabilities


class PositionBiasedResponseModel:
    """Select exactly one displayed item with weights ``exp(alpha * rank)``."""

    VARIANT_ID = "position_biased_mansoury_v1"
    capabilities = UserModelCapabilities(
        supports_no_click=False,
        supports_state_update=False,
        supports_continuation=False,
    )

    def __init__(
        self,
        *,
        alpha: float = -1.0,
        seed: int = 42,
    ) -> None:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a real number")
        self.alpha = float(alpha)
        if not math.isfinite(self.alpha) or self.alpha >= 0:
            raise ValueError("alpha must be finite and negative")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed cannot be negative")
        self.seed = seed

    def choice_probabilities(self, slate_size: int) -> tuple[float, ...]:
        if isinstance(slate_size, bool) or not isinstance(slate_size, int):
            raise TypeError("slate_size must be an integer")
        if slate_size <= 0:
            raise ValueError("slate_size must be positive")
        ranks = np.arange(1, slate_size + 1, dtype=np.float64)
        log_weights = self.alpha * ranks
        weights = np.exp(log_weights - log_weights.max())
        return tuple(float(value) for value in weights / weights.sum())

    def respond(self, turn: TurnInput) -> ActionDecision:
        slate = turn.presented_slate
        if not slate or len(slate) != len(set(slate)):
            raise ValueError("presented_slate must contain unique items")
        probabilities = self.choice_probabilities(len(slate))
        rng = random_generator(
            self.seed,
            RandomKey(
                stream_id=1,
                user_id=turn.user_id,
                iteration=turn.iteration,
            ),
        )
        draw = float(rng.random())
        selected_index = min(
            int(np.searchsorted(np.cumsum(probabilities), draw, side="right")),
            len(slate) - 1,
        )
        return ActionDecision(
            decision_id=f"{turn.interaction_id}__position_biased",
            status=ActionStatus.VALID,
            action=UserAction.CLICK,
            clicked_item_id=slate[selected_index],
            diagnostics={
                "alpha": self.alpha,
                "probabilities": probabilities,
                "random_draw": draw,
                "selected_displayed_rank": selected_index + 1,
            },
        )

    def update(
        self, action: ActionDecision, context: Mapping[str, Any]
    ) -> UserUpdateResult:
        del action, context
        return UserUpdateResult(
            update_id="position_biased__no_state_update",
            status=UpdateStatus.SKIPPED,
        )

    def decide_continuation(
        self, post_update_context: Mapping[str, Any]
    ) -> ContinuationDecision:
        del post_update_context
        return ContinuationDecision(
            decision_id="position_biased__continuation_not_requested",
            status=ContinuationStatus.NOT_REQUESTED,
            continuation=Continuation.UNAVAILABLE,
        )

    def export_state(self) -> Mapping[str, Any]:
        return {
            "variant_id": self.VARIANT_ID,
            "alpha": self.alpha,
            "seed": self.seed,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("variant_id") != self.VARIANT_ID:
            raise ValueError("position-biased state variant mismatch")
        replacement = type(self)(
            alpha=state["alpha"],
            seed=state["seed"],
        )
        self.__dict__.update(replacement.__dict__)

    def fingerprint(self) -> str:
        return fingerprint_semantic_state(self.export_state(), self.VARIANT_ID)
