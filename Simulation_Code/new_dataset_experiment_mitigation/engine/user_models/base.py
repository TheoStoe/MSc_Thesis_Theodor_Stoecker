"""User-response interface required by the generic simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ..contracts import (
    ActionDecision,
    ContinuationDecision,
    TurnInput,
    UserUpdateResult,
)


@dataclass(frozen=True)
class UserModelCapabilities:
    supports_no_click: bool
    supports_state_update: bool
    supports_continuation: bool


class UserResponseModel(Protocol):
    """Every registered model is checkpointable; that is not optional."""

    capabilities: UserModelCapabilities

    def respond(self, turn: TurnInput) -> ActionDecision: ...

    def update(
        self, action: ActionDecision, context: Mapping[str, Any]
    ) -> UserUpdateResult: ...

    def decide_continuation(
        self, post_update_context: Mapping[str, Any]
    ) -> ContinuationDecision: ...

    def export_state(self) -> Mapping[str, Any]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...

    def fingerprint(self) -> str: ...
