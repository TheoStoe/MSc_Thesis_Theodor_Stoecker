"""Stable, model-neutral data contracts for simulation events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

UserId = int
ItemId = int


class ActionStatus(str, Enum):
    """Separate observed behavior from recovery and technical outcomes."""

    VALID = "valid"
    FALLBACK = "fallback"
    INVALID = "invalid"
    TECHNICAL_FAILURE = "technical_failure"


class ContinuationStatus(str, Enum):
    VALID = "valid"
    FALLBACK = "fallback"
    INVALID = "invalid"
    TECHNICAL_FAILURE = "technical_failure"
    NOT_REQUESTED = "not_requested"


class UpdateStatus(str, Enum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    INVALID = "invalid"
    TECHNICAL_FAILURE = "technical_failure"


class UserAction(str, Enum):
    CLICK = "click"
    NO_CLICK = "no_click"
    UNAVAILABLE = "unavailable"


class Continuation(str, Enum):
    STAY = "stay"
    EXIT = "exit"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class TurnInput:
    """Everything a response model may observe for one simulation iteration."""

    run_id: str
    condition_id: str
    seed: int
    interaction_id: str
    user_id: UserId
    iteration: int
    recommender_slate: tuple[ItemId, ...]
    presented_slate: tuple[ItemId, ...]
    recommender_rank_by_item: Mapping[ItemId, int]
    displayed_rank_by_item: Mapping[ItemId, int]
    observed_history: tuple[ItemId, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionDecision:
    """Typed click/no-click result; failures never masquerade as behavior.

    Implementation validation must enforce: a valid click names exactly one
    displayed item; a valid no-click names none; every non-valid result uses
    ``UNAVAILABLE`` and no item.
    """

    decision_id: str
    status: ActionStatus
    action: UserAction
    clicked_item_id: ItemId | None = None
    reason: str | None = None
    trace_ids: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UserUpdateResult:
    """Audit result of a post-action user-state transition."""

    update_id: str
    status: UpdateStatus
    changed_domains: tuple[str, ...] = ()
    trace_ids: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContinuationDecision:
    """A participation decision requested only after the user-state update.

    ``NOT_REQUESTED + UNAVAILABLE`` is the expected combination outside L4.
    Non-valid requested outcomes default to stay operationally but remain
    non-behavioral in the scientific event.
    """

    decision_id: str
    status: ContinuationStatus
    continuation: Continuation
    reason: str | None = None
    trace_ids: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnEvent:
    """Normalized per-user simulation-iteration record used by analysis.

    ``presentation_metadata`` carries policy-produced audit fields that do not
    belong to either ranking itself.  Examples include a sponsored item,
    displaced organic item, de-duplication rule, or order-probe role.  Keeping
    those fields on the persisted turn avoids making L5 analysis depend on a
    separate diagnostic log.
    """

    run_id: str
    condition_id: str
    seed: int
    interaction_id: str
    user_id: UserId
    iteration: int
    recommender_slate: tuple[ItemId, ...]
    presented_slate: tuple[ItemId, ...]
    recommender_rank_by_item: Mapping[ItemId, int]
    displayed_rank_by_item: Mapping[ItemId, int]
    action: ActionDecision
    user_update: UserUpdateResult | None
    continuation: ContinuationDecision
    scheduled_exit_effective_iteration: int | None
    presentation_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationCheckpoint:
    """One container for everything required to validate and resume a job.

    Run identity and progress are stored beside opaque component/policy states
    so a checkpoint cannot be loaded into the wrong condition or seed.
    """

    schema_version: str
    run_id: str
    condition_id: str
    seed: int
    completed_iteration: int
    input_fingerprints: Mapping[str, str]
    component_states: Mapping[str, Any]
    policy_states: Mapping[str, Any]
    randomness_state: Mapping[str, Any]


@dataclass(frozen=True)
class RunArtifact:
    """Paths and identity of one completed immutable condition/seed job."""

    run_id: str
    output_dir: Path
    manifest_path: Path
    events_path: Path
    diagnostics_path: Path
    final_checkpoint_path: Path | None
