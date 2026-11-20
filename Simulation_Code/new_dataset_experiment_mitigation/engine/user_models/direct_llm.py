"""Single-call, history-conditioned LLM user-response baseline.

This is the small non-agentic counterpart to ``agentic_recommender``.  It is
adapted from the first experiment's direct LLM condition, but uses only the
new simulator contracts and shared :class:`~utility.providers.JSONProvider`.
The model has no private user memory: the runner supplies the observable
history in every :class:`~new_dataset_experiment_mitigation.engine.contracts.TurnInput`.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..contracts import (
    ActionDecision,
    ActionStatus,
    Continuation,
    ContinuationDecision,
    ContinuationStatus,
    ItemId,
    TurnInput,
    UpdateStatus,
    UserAction,
    UserUpdateResult,
)
from ..utility.fingerprints import fingerprint_semantic_state
from ..utility.providers import (
    InvalidProviderResponse,
    JSONProvider,
    ProviderError,
)
from .base import UserModelCapabilities


logger = logging.getLogger(__name__)

PROMPT_VERSION = "direct_book_choice_or_no_click_v1"

SYSTEM_PROMPT = """
You are simulating a user choosing from book recommendation slate.
Use only the supplied read history and displayed book information.
Click exactly one displayed item when one is appealing enough, otherwise
choose no-click. Do not invent ratings, book facts, or hidden evidence.
Never invent reader age, gender or occupation. Never infer author gender.
Return only the requested structured result.
""".strip()

CHOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["click", "no_click"]},
        "selected_item_id": {
            "anyOf": [{"type": "integer"}, {"type": "null"}]
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["action", "selected_item_id", "reason"],
    "additionalProperties": False,
}


class DirectLLMResponseModel:
    """Make one bounded LLM choice without agentic or latent user state."""

    VARIANT_ID = "direct_llm_single_call_v1"
    capabilities = UserModelCapabilities(
        supports_no_click=True,
        supports_state_update=False,
        supports_continuation=False,
    )

    def __init__(
        self,
        *,
        provider: JSONProvider,
        item_text_by_id: Mapping[ItemId, str],
        history_limit: int = 20,
    ) -> None:
        if isinstance(history_limit, bool) or not isinstance(history_limit, int):
            raise TypeError("history_limit must be an integer")
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        if not item_text_by_id:
            raise ValueError("item_text_by_id cannot be empty")

        item_texts: dict[ItemId, str] = {}
        for item_id, text in item_text_by_id.items():
            if isinstance(item_id, bool) or not isinstance(item_id, int):
                raise TypeError("item-text keys must be integers")
            if item_id <= 0:
                raise ValueError("item IDs must be positive")
            item_texts[item_id] = _clean_text(text, f"item_text[{item_id}]")

        self.provider = provider
        self.provider_settings = dict(provider.semantic_settings())
        self.history_limit = history_limit
        self.item_text_by_id = MappingProxyType(item_texts)
        self.item_text_fingerprint = fingerprint_semantic_state(
            item_texts,
            "direct-llm-item-text-v1",
        )
        self.prompt_fingerprint = fingerprint_semantic_state(
            {
                "prompt_version": PROMPT_VERSION,
                "system_prompt": SYSTEM_PROMPT,
                "choice_schema": CHOICE_SCHEMA,
                "history_order": "chronological_recent_tail",
                "slate_order": "displayed_rank",
            },
            "direct-llm-prompt-contract-v1",
        )
        logger.info(
            "Direct LLM initialized variant=%s items=%s history_limit=%s "
            "prompt_fingerprint=%s item_text_fingerprint=%s",
            self.VARIANT_ID,
            len(self.item_text_by_id),
            self.history_limit,
            self.prompt_fingerprint,
            self.item_text_fingerprint,
        )

    def respond(self, turn: TurnInput) -> ActionDecision:
        if not isinstance(turn, TurnInput):
            raise TypeError("turn must be a TurnInput")
        slate = turn.presented_slate
        if not slate or len(slate) != len(set(slate)):
            raise ValueError("presented_slate must contain unique items")

        rendered_history = turn.observed_history[-self.history_limit :]
        required_items = (*rendered_history, *slate)
        missing = [
            item_id for item_id in required_items if item_id not in self.item_text_by_id
        ]
        if missing:
            raise KeyError(f"missing item text for item {missing[0]}")

        user_prompt = self._user_prompt(rendered_history, slate)
        logger.info(
            "Direct LLM request started user=%s interaction=%s iteration=%s "
            "history_items=%s slate_items=%s",
            turn.user_id,
            turn.interaction_id,
            turn.iteration,
            len(rendered_history),
            len(slate),
        )
        try:
            result = self.provider.call_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=_choice_schema(slate),
                validator=lambda payload: _validate_choice(payload, slate),
                schema_name="direct_llm_choice",
            )
        except InvalidProviderResponse as exc:
            logger.error(
                "Direct LLM response invalid user=%s interaction=%s "
                "iteration=%s error=%s",
                turn.user_id,
                turn.interaction_id,
                turn.iteration,
                exc,
            )
            return self._failed_action(turn, ActionStatus.INVALID, str(exc))
        except ProviderError as exc:
            logger.error(
                "Direct LLM provider failure user=%s interaction=%s "
                "iteration=%s error=%s",
                turn.user_id,
                turn.interaction_id,
                turn.iteration,
                exc,
            )
            return self._failed_action(
                turn,
                ActionStatus.TECHNICAL_FAILURE,
                str(exc),
            )

        payload = result.payload
        action = (
            UserAction.CLICK
            if payload["action"] == "click"
            else UserAction.NO_CLICK
        )
        trace_ids = (result.response_id,) if result.response_id else ()
        logger.info(
            "Direct LLM decision completed user=%s interaction=%s "
            "iteration=%s action=%s selected_item_id=%s response_id=%s "
            "attempts=%s",
            turn.user_id,
            turn.interaction_id,
            turn.iteration,
            action.value,
            payload["selected_item_id"],
            result.response_id,
            result.attempts,
        )
        return ActionDecision(
            decision_id=f"{turn.interaction_id}__direct_llm",
            status=ActionStatus.VALID,
            action=action,
            clicked_item_id=payload["selected_item_id"],
            reason=payload["reason"],
            trace_ids=trace_ids,
            diagnostics={
                "user_id": turn.user_id,
                "interaction_id": turn.interaction_id,
                "iteration": turn.iteration,
                "presented_slate": turn.presented_slate,
                "full_history_size": len(turn.observed_history),
                "rendered_history_size": len(rendered_history),
                "provider_attempts": result.attempts,
                "provider_usage": dict(result.usage),
                "prompt_version": PROMPT_VERSION,
                "prompt_fingerprint": self.prompt_fingerprint,
                "item_text_fingerprint": self.item_text_fingerprint,
            },
        )

    def _user_prompt(
        self,
        rendered_history: Sequence[ItemId],
        slate: Sequence[ItemId],
    ) -> str:
        history_text = _render_items(
            rendered_history,
            self.item_text_by_id,
            ranked=False,
        )
        slate_text = _render_items(
            slate,
            self.item_text_by_id,
            ranked=True,
        )
        return (
            "Read books in chronological order (oldest to newest):\n"
            f"{history_text}\n\n"
            "Displayed books in order:\n"
            f"{slate_text}\n\n"
            "Choose action=click with exactly one displayed selected_item_id, "
            "or action=no_click with selected_item_id=null when none is "
            "appealing enough. Give one short observable reason."
        )

    def _failed_action(
        self,
        turn: TurnInput,
        status: ActionStatus,
        reason: str,
    ) -> ActionDecision:
        return ActionDecision(
            decision_id=f"{turn.interaction_id}__direct_llm",
            status=status,
            action=UserAction.UNAVAILABLE,
            clicked_item_id=None,
            reason=reason,
            diagnostics={
                "user_id": turn.user_id,
                "interaction_id": turn.interaction_id,
                "iteration": turn.iteration,
                "prompt_version": PROMPT_VERSION,
                "prompt_fingerprint": self.prompt_fingerprint,
                "item_text_fingerprint": self.item_text_fingerprint,
            },
        )

    def update(
        self,
        action: ActionDecision,
        context: Mapping[str, Any],
    ) -> UserUpdateResult:
        del action, context
        return UserUpdateResult(
            update_id="direct_llm__no_state_update",
            status=UpdateStatus.SKIPPED,
        )

    def decide_continuation(
        self,
        post_update_context: Mapping[str, Any],
    ) -> ContinuationDecision:
        del post_update_context
        return ContinuationDecision(
            decision_id="direct_llm__continuation_not_requested",
            status=ContinuationStatus.NOT_REQUESTED,
            continuation=Continuation.UNAVAILABLE,
        )

    def export_state(self) -> Mapping[str, Any]:
        return {
            "variant_id": self.VARIANT_ID,
            "history_limit": self.history_limit,
            "provider_settings": self.provider_settings,
            "prompt_fingerprint": self.prompt_fingerprint,
            "item_text_fingerprint": self.item_text_fingerprint,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("variant_id") != self.VARIANT_ID:
            raise ValueError("direct-LLM state variant mismatch")
        for field, expected in (
            ("history_limit", self.history_limit),
            ("provider_settings", self.provider_settings),
            ("prompt_fingerprint", self.prompt_fingerprint),
            ("item_text_fingerprint", self.item_text_fingerprint),
        ):
            if state.get(field) != expected:
                raise ValueError(f"direct-LLM {field} mismatch")
        logger.info(
            "Direct LLM state restored variant=%s state_fingerprint=%s",
            self.VARIANT_ID,
            self.fingerprint(),
        )

    def fingerprint(self) -> str:
        return fingerprint_semantic_state(self.export_state(), self.VARIANT_ID)


def _choice_schema(slate: Sequence[ItemId]) -> Mapping[str, Any]:
    schema = {
        **CHOICE_SCHEMA,
        "properties": dict(CHOICE_SCHEMA["properties"]),
    }
    schema["properties"]["selected_item_id"] = {
        "anyOf": [
            {"type": "integer", "enum": list(slate)},
            {"type": "null"},
        ]
    }
    return schema


def _validate_choice(
    payload: Mapping[str, Any],
    slate: Sequence[ItemId],
) -> Mapping[str, Any]:
    if set(payload) != {"action", "selected_item_id", "reason"}:
        raise ValueError("choice must contain exactly the required fields")
    action = payload["action"]
    selected_item_id = payload["selected_item_id"]
    if action == "click":
        if isinstance(selected_item_id, bool) or not isinstance(
            selected_item_id,
            int,
        ):
            raise ValueError("click requires an integer selected_item_id")
        if selected_item_id not in slate:
            raise ValueError("selected_item_id must occur in the displayed slate")
    elif action == "no_click":
        if selected_item_id is not None:
            raise ValueError("no_click requires selected_item_id=null")
    else:
        raise ValueError("action must be click or no_click")
    return {
        "action": action,
        "selected_item_id": selected_item_id,
        "reason": _clean_text(payload["reason"], "reason"),
    }


def _render_items(
    item_ids: Sequence[ItemId],
    item_text_by_id: Mapping[ItemId, str],
    *,
    ranked: bool,
) -> str:
    if not item_ids:
        return "[none]"
    return "\n".join(
        (
            f"{index}. item_id={item_id}: {item_text_by_id[item_id]}"
            if ranked
            else f"- item_id={item_id}: {item_text_by_id[item_id]}"
        )
        for index, item_id in enumerate(item_ids, start=1)
    )


def _clean_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-blank string")
    return " ".join(value.split())


__all__ = ["DirectLLMResponseModel"]
