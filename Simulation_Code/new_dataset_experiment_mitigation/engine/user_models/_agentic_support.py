"""Private stateless mechanics for the modular agentic response model.

Prompts, structured-output schemas, lifecycle control, state mutation, and
logging intentionally remain in :mod:`agentic_recommender`.  This module only
contains deterministic validation, rendering, and bookkeeping helpers so the
behavioral specification stays readable without creating another framework.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts import (
    ActionDecision,
    Continuation,
    ContinuationDecision,
    ContinuationStatus,
    ItemId,
    TurnInput,
    UserAction,
    UserId,
)
from ..utility.providers import ProviderResult


def clean_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-blank string")
    return " ".join(value.split())


def profile_text(value: Any, field: str) -> str:
    """Validate profile text without destroying Lusifer paragraph layout."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-blank string")
    return value.strip()


def choice_schema(
    base_schema: Mapping[str, Any],
    *,
    item_field: str,
    slate: Sequence[ItemId],
) -> Mapping[str, Any]:
    """Specialize one schema with the exact displayed-item IDs.

    The copy is deliberate: condition-specific item enums must never mutate
    the module-level schema constants that define the prompt contract.
    """

    schema = {
        **base_schema,
        "properties": dict(base_schema["properties"]),
    }
    schema["properties"][item_field] = {
        "anyOf": [
            {"type": "integer", "enum": list(slate)},
            {"type": "null"},
        ]
    }
    return schema


def _validate_choice(
    *,
    action: Any,
    selected_item_id: Any,
    slate: Sequence[ItemId],
    action_field: str,
    item_field: str,
) -> tuple[str, ItemId | None]:
    if action == "click":
        if isinstance(selected_item_id, bool) or not isinstance(
            selected_item_id,
            int,
        ):
            raise ValueError(f"click requires an integer {item_field}")
        if selected_item_id not in slate:
            raise ValueError(f"{item_field} must occur in the displayed slate")
    elif action == "no_click":
        if selected_item_id is not None:
            raise ValueError(f"no_click requires {item_field}=null")
    else:
        raise ValueError(f"{action_field} must be click or no_click")
    return action, selected_item_id


def validate_planning(
    payload: Mapping[str, Any],
    slate: Sequence[ItemId],
) -> Mapping[str, Any]:
    expected = {
        "module_invoked",
        "suggested_action",
        "suggested_selected_item_id",
        "reason",
    }
    if set(payload) != expected:
        raise ValueError("planning must contain exactly the required fields")
    module_invoked = payload["module_invoked"]
    if module_invoked not in {"action", "memory"}:
        raise ValueError("module_invoked must be action or memory")
    action, selected = _validate_choice(
        action=payload["suggested_action"],
        selected_item_id=payload["suggested_selected_item_id"],
        slate=slate,
        action_field="suggested_action",
        item_field="suggested_selected_item_id",
    )
    return {
        "module_invoked": module_invoked,
        "suggested_action": action,
        "suggested_selected_item_id": selected,
        "reason": clean_text(payload["reason"], "reason"),
    }


def validate_memory(
    payload: Mapping[str, Any],
    slate: Sequence[ItemId],
) -> Mapping[str, Any]:
    expected = {
        "suggested_action",
        "suggested_selected_item_id",
        "reason",
    }
    if set(payload) != expected:
        raise ValueError("memory must contain exactly the required fields")
    action, selected = _validate_choice(
        action=payload["suggested_action"],
        selected_item_id=payload["suggested_selected_item_id"],
        slate=slate,
        action_field="suggested_action",
        item_field="suggested_selected_item_id",
    )
    return {
        "suggested_action": action,
        "suggested_selected_item_id": selected,
        "reason": clean_text(payload["reason"], "reason"),
    }


def validate_action(
    payload: Mapping[str, Any],
    slate: Sequence[ItemId],
) -> Mapping[str, Any]:
    if set(payload) != {"action", "selected_item_id", "reason"}:
        raise ValueError("action must contain exactly the required fields")
    action, selected = _validate_choice(
        action=payload["action"],
        selected_item_id=payload["selected_item_id"],
        slate=slate,
        action_field="action",
        item_field="selected_item_id",
    )
    return {
        "action": action,
        "selected_item_id": selected,
        "reason": clean_text(payload["reason"], "reason"),
    }


def validate_reflection(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {"updated_profile", "satisfaction", "memory_note"}
    if set(payload) != expected:
        raise ValueError("reflection must contain exactly the required fields")
    satisfaction = payload["satisfaction"]
    if satisfaction not in {"satisfied", "neutral", "unsatisfied"}:
        raise ValueError("invalid satisfaction")
    return {
        "updated_profile": profile_text(
            payload["updated_profile"],
            "updated_profile",
        ),
        "satisfaction": satisfaction,
        "memory_note": clean_text(payload["memory_note"], "memory_note"),
    }


def validate_continuation(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(payload) != {"continuation", "reason"}:
        raise ValueError("continuation must contain exactly the required fields")
    continuation = payload["continuation"]
    if continuation not in {"stay", "exit"}:
        raise ValueError("continuation must be stay or exit")
    return {
        "continuation": continuation,
        "reason": clean_text(payload["reason"], "reason"),
    }


def module_trace_text(trace: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for step, payload in enumerate(trace, start=1):
        module = payload["module"]
        action = payload.get("suggested_action") or payload.get("action")
        item_id = payload.get(
            "suggested_selected_item_id",
            payload.get("selected_item_id"),
        )
        lines.append(
            f"step={step}; module={module}; suggested_action={action}; "
            f"suggested_selected_item_id={item_id}; reason={payload['reason']}"
        )
    return "\n".join(lines)


def module_trace_summary(trace: Sequence[Mapping[str, Any]]) -> str:
    if not trace:
        return "Proceed directly to the action module."
    return " | ".join(
        f"{payload['module']}: {clean_text(payload['reason'], 'reason')}"
        for payload in trace
    )


def latest_suggestion(
    trace: Sequence[Mapping[str, Any]],
) -> tuple[str, ItemId | None] | None:
    for payload in reversed(trace):
        if "suggested_action" in payload:
            return (
                str(payload["suggested_action"]),
                payload.get("suggested_selected_item_id"),
            )
    return None


def provider_step(module: str, result: ProviderResult) -> dict[str, Any]:
    return {
        "module": module,
        "response_id": result.response_id,
        "attempts": result.attempts,
        "usage": dict(result.usage),
    }


def aggregate_provider_usage(
    provider_steps: Sequence[Mapping[str, Any]],
) -> Mapping[str, int | float]:
    totals: dict[str, int | float] = {}
    for step in provider_steps:
        usage = step.get("usage", {})
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[str(key)] = totals.get(str(key), 0) + value
    return totals


def validated_update_context(
    action: ActionDecision,
    context: Mapping[str, Any],
    *,
    expected_prompt_fingerprint: str,
) -> TurnInput:
    turn = context.get("turn")
    if not isinstance(turn, TurnInput):
        raise ValueError("agentic update context requires the canonical TurnInput")
    diagnostics = action.diagnostics
    expected = {
        "user_id": turn.user_id,
        "interaction_id": turn.interaction_id,
        "iteration": turn.iteration,
        "presented_slate": turn.presented_slate,
    }
    for field, value in expected.items():
        if diagnostics.get(field) != value:
            raise ValueError(f"action does not match canonical turn field {field}")
    if diagnostics.get("prompt_fingerprint") != expected_prompt_fingerprint:
        raise ValueError("action prompt fingerprint mismatch")
    if action.action is UserAction.CLICK:
        if action.clicked_item_id not in turn.presented_slate:
            raise ValueError("valid click must name a canonical displayed item")
    elif action.action is UserAction.NO_CLICK:
        if action.clicked_item_id is not None:
            raise ValueError("valid no-click cannot name an item")
    else:
        raise ValueError("valid action must be click or no-click")
    return turn


def context_user_id(context: Mapping[str, Any]) -> UserId:
    value = context.get("user_id")
    if value is None and isinstance(context.get("turn"), TurnInput):
        value = context["turn"].user_id
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("post-update context requires an integer user_id")
    return value


def failed_continuation(
    interaction_id: str,
    status: ContinuationStatus,
    reason: str,
) -> ContinuationDecision:
    return ContinuationDecision(
        decision_id=f"{interaction_id}__continuation",
        status=status,
        continuation=Continuation.UNAVAILABLE,
        reason=reason,
    )


def numbered_items(
    item_ids: Sequence[ItemId],
    item_text_by_id: Mapping[ItemId, str],
) -> str:
    if not item_ids:
        return "[none]"
    return "\n".join(
        f"{rank}. item_id={item_id}: {item_text_by_id[item_id]}"
        for rank, item_id in enumerate(item_ids, start=1)
    )


__all__ = [
    "aggregate_provider_usage",
    "choice_schema",
    "clean_text",
    "context_user_id",
    "failed_continuation",
    "latest_suggestion",
    "module_trace_summary",
    "module_trace_text",
    "numbered_items",
    "profile_text",
    "provider_step",
    "validate_action",
    "validate_continuation",
    "validate_memory",
    "validate_planning",
    "validate_reflection",
    "validated_update_context",
]
