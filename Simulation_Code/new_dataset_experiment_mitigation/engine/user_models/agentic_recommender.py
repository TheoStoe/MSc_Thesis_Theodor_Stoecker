"""A small agentic recommender-user simulator built around Lusifer profiles.

Agent4Rec is a conceptual reference for separating profile, planning, action,
reflection, and memory.
This local design instead consumes normalized :class:`ProfileArtifact` values,
supports an explicit behavioral no-click, and keeps participation separate from
the item-selection action.

Conceptual reference:
https://github.com/LehengTHU/Agent4Rec/
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any, Literal, Mapping, Sequence

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
    UserId,
    UserUpdateResult,
)
from ..persona_extraction.base import ProfileArtifact
from ..psychology.fatigue import (
    RANK_STEP_FATIGUE_BUCKET_SIZE,
    RANK_STEP_FATIGUE_FLOOR,
    RANK_STEP_FATIGUE_IMPLEMENTATION_VERSION,
    RANK_STEP_FATIGUE_TOP_CLICK_DELTA,
    update_rank_step_fatigue,
)
from ..psychology.memory import (
    MemoryRetrievalView,
    compose_memory_candidates,
    retrieve_memories,
    seed_historical_memories,
)
from ..psychology.tipi import TipiScores
from ..utility.fingerprints import fingerprint_semantic_state
from ..utility.providers import (
    InvalidProviderResponse,
    JSONProvider,
    ProviderError,
)
from . import _agentic_support as support
from .base import UserModelCapabilities


UpdateMode = Literal[
    "frozen",
    "full_reflection",
    "memory_only_reflection",
    "no_reflection_control",
]
MemoryTreatment = Literal["forced_memory", "power_law_decay"]
FatigueTreatment = Literal["rank_step_cumulative_prompt"]
UPDATE_MODES = frozenset(
    {
        "frozen",
        "full_reflection",
        "memory_only_reflection",
        "no_reflection_control",
    }
)
MEMORY_TREATMENTS = frozenset({"forced_memory", "power_law_decay"})
FATIGUE_TREATMENTS = frozenset({"rank_step_cumulative_prompt"})

DECISION_PROMPT_VERSION = "agentic_recommender_modules_v2"
REFLECTION_PROMPT_VERSION = "agentic_recommender_reflection_v1"
CONTINUATION_PROMPT_VERSION = "agentic_recommender_continuation_v1"
TREATED_MEMORY_PROMPT_VERSION = "agentic_recommender_actr_memory_v2"
RANK_STEP_FATIGUE_PROMPT_VERSION = "rank_step_fatigue_continuation_v1"
MEMORY_TIE_RULE = "descending_activation_then_ascending_numeric_item_id"
HISTORICAL_MEMORY_SEED_RULE = "last_k_log_events_oldest_to_newest_v1"

PERSONALITY_PROMPT_TEMPLATE = """
Frozen Big Five personality (on a 1-7 scale).
Interpret scores for traits as 1 = very low, 4 = midpoint, and 7 = very high :

- Openness: {openness:.1f}
- Conscientiousness: {conscientiousness:.1f}
- Extraversion: {extraversion:.1f}
- Agreeableness: {agreeableness:.1f}
- Neuroticism: {neuroticism:.1f}

Use this stable tendency when making decisions. Do not revise it or copy it
into the evolving book-preference profile.
""".strip()

DECISION_SYSTEM_PROMPT = """
You are a user on a book-platform. Use only the supplied factual profile,
observable interaction memory, and displayed books. 
The goal is to either click exactly one displayed item or choose no-click.
Do not invent ratings, demographics, exposures, book facts, or hidden reasons.
Return only the requested structured result.
""".strip()

REFLECTION_SYSTEM_PROMPT = """
You just made a decision about recommended books. Reflect on the choice made. 
Update the preference profile only when the current interaction supports a change. 
Preserve the supported reading preferences. Never invent reader age, gender, occupation, ratings, exposures, or book facts. Never infer author gender.
Return only the requested structured result.
""".strip()

CONTINUATION_SYSTEM_PROMPT = """
You can also choose to leave the platform. 
Use only the supplied current profile and observable memory. 
Return stay or exit with a short observable reason.
Return only the requested structured result.
""".strip()

MAX_NUMBER_AGENTIC_LOOP = 3
MODULES_TO_INVOKE = ("planning", "memory", "action")

PLAN_PROMPT = f"""
You are now planning which modules to invoke if further reasoning over histories are required.
When the decision is finalised, invoke the action module and end the planning loop.
Always give a working suggested selected item id or null if you suggest no_click for the given slate.
Memory has access to books you read in the past with descriptions.
Action is for when enough information to make a decision is available.
The reason will be passed on to the invoked module, to steer them.
After {MAX_NUMBER_AGENTIC_LOOP} planning or memory steps, the action module is
invoked automatically.
""".strip()

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "module_invoked": {"type": "string", "enum": ["action", "memory"]},
        "suggested_action": {"type": "string", "enum": ["click", "no_click"]},
        "suggested_selected_item_id": {
            "anyOf": [{"type": "integer"}, {"type": "null"}]
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": [
        "module_invoked",
        "suggested_action",
        "suggested_selected_item_id",
        "reason",
    ],
    "additionalProperties": False,
}

ACTION_DECISION_SCHEMA: dict[str, Any] = {
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

MEMORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "suggested_action": {"type": "string", "enum": ["click", "no_click"]},
        "suggested_selected_item_id": {
            "anyOf": [{"type": "integer"}, {"type": "null"}]
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": [
        "suggested_action",
        "suggested_selected_item_id",
        "reason",
    ],
    "additionalProperties": False,
}

REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "updated_profile": {"type": "string", "minLength": 1},
        "satisfaction": {
            "type": "string",
            "enum": ["satisfied", "neutral", "unsatisfied"],
        },
        "memory_note": {"type": "string", "minLength": 1},
    },
    "required": [
        "updated_profile",
        "satisfaction",
        "memory_note",
    ],
    "additionalProperties": False,
}

CONTINUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "continuation": {"type": "string", "enum": ["stay", "exit"]},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["continuation", "reason"],
    "additionalProperties": False,
}

RANK_STEP_FATIGUE_PROMPT = (
    "Additional fatigue signal: your current cumulative fatigue value is "
    "{fatigue_value}. A value of 0 means no accumulated fatigue; higher values "
    "mean more accumulated fatigue. Use this signal only when deciding whether "
    "to stay or exit."
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InteractionMemory:
    interaction_id: str
    iteration: int
    presented_slate: tuple[ItemId, ...]
    plan: str
    action: str
    clicked_item_id: ItemId | None
    reason: str


@dataclass(frozen=True)
class ReflectionMemory:
    interaction_id: str
    satisfaction: str
    note: str


@dataclass
class _UserState:
    source_profile_fingerprint: str
    initial_profile_text: str
    current_profile_text: str
    interactions: list[InteractionMemory]
    reflections: list[ReflectionMemory]
    historical_memory_occurrences_by_item: dict[ItemId, tuple[int, ...]]
    memory_occurrences_by_item: dict[ItemId, list[int]]
    tipi_scores: TipiScores | None = None
    fatigue_value: int = 0


@dataclass(frozen=True)
class _ComposedMemoryView:
    retrieval: MemoryRetrievalView
    historical_fill_item_ids: tuple[ItemId, ...]
    online_item_ids: tuple[ItemId, ...]


class AgenticRecommenderResponseModel:
    """Profile-conditioned agent with explicit, independently gated state."""

    VARIANT_ID = "agentic_recommender_lusifer_v2"
    capabilities = UserModelCapabilities(
        supports_no_click=True,
        supports_state_update=True,
        supports_continuation=True,
    )

    def __init__(
        self,
        *,
        provider: JSONProvider,
        profiles_by_user: Mapping[UserId, ProfileArtifact],
        item_text_by_id: Mapping[ItemId, str],
        tipi_treatment: str | None = None,
        tipi_scores_by_user: Mapping[UserId, TipiScores] | None = None,
        update_mode: UpdateMode = "full_reflection",
        memory_limit: int = 20,
        history_limit: int = 100,
        memory_treatment: MemoryTreatment | None = None,
        memory_decay: float | None = None,
        memory_retrieval_limit: int | None = None,
        initial_histories_by_user: (
            Mapping[UserId, Sequence[ItemId]] | None
        ) = None,
        fatigue_treatment: FatigueTreatment | None = None,
    ) -> None:
        if update_mode not in UPDATE_MODES:
            raise ValueError(f"unsupported update_mode: {update_mode}")
        for value, name in (
            (memory_limit, "memory_limit"),
            (history_limit, "history_limit"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not profiles_by_user:
            raise ValueError("profiles_by_user cannot be empty")
        if tipi_treatment is not None and (
            not isinstance(tipi_treatment, str) or not tipi_treatment.strip()
        ):
            raise ValueError("tipi_treatment must be a non-blank string or None")
        if fatigue_treatment is not None:
            if fatigue_treatment not in FATIGUE_TREATMENTS:
                raise ValueError(
                    f"unsupported fatigue_treatment: {fatigue_treatment}"
                )
            if update_mode != "no_reflection_control":
                raise ValueError(
                    "rank-step fatigue treatments require "
                    "no_reflection_control"
                )
            if tipi_treatment is not None:
                raise ValueError(
                    "rank-step fatigue and TIPI treatments cannot be combined"
                )
            if memory_treatment is not None:
                raise ValueError(
                    "rank-step fatigue and ACT-R memory treatments cannot be "
                    "combined"
                )
        if memory_treatment is None:
            if (
                memory_decay is not None
                or memory_retrieval_limit is not None
                or initial_histories_by_user is not None
            ):
                raise ValueError(
                    "memory decay, retrieval limit, and initial histories "
                    "require a memory treatment"
                )
            normalized_memory_decay = None
            initial_histories: dict[UserId, Sequence[ItemId]] = {}
        else:
            if memory_treatment not in MEMORY_TREATMENTS:
                raise ValueError(
                    f"unsupported memory_treatment: {memory_treatment}"
                )
            if memory_decay is None or memory_retrieval_limit is None:
                raise ValueError(
                    "memory treatments require decay and retrieval limit"
                )
            if initial_histories_by_user is None:
                raise ValueError("memory treatments require initial histories")
            if not isinstance(initial_histories_by_user, Mapping):
                raise TypeError("initial_histories_by_user must be a mapping")
            initial_histories = dict(initial_histories_by_user)
            if update_mode != "no_reflection_control":
                raise ValueError(
                    "memory treatments require no_reflection_control so the "
                    "profile and reflection state cannot leak forgotten items"
                )
            empty_view = retrieve_memories(
                {},
                current_timestep=0,
                decay=memory_decay,
                retrieval_limit=memory_retrieval_limit,
            )
            normalized_memory_decay = empty_view.decay
            if (
                memory_treatment == "forced_memory"
                and normalized_memory_decay != 0
            ):
                raise ValueError("forced_memory requires memory_decay=0")
            if (
                memory_treatment == "power_law_decay"
                and normalized_memory_decay <= 0
            ):
                raise ValueError("power_law_decay requires memory_decay>0")
        scores_by_user = dict(tipi_scores_by_user or {})
        unknown_scores = sorted(set(scores_by_user).difference(profiles_by_user))
        if unknown_scores:
            raise ValueError(
                f"TIPI scores contain users without profiles: {unknown_scores}"
            )
        if any(
            not isinstance(scores, TipiScores)
            for scores in scores_by_user.values()
        ):
            raise TypeError("tipi_scores_by_user must contain TipiScores values")
        if tipi_treatment is None and scores_by_user:
            raise ValueError("hidden personality conditions cannot supply TIPI scores")
        if tipi_treatment is not None and not scores_by_user:
            raise ValueError("visible personality conditions require TIPI scores")
        if memory_treatment is not None:
            invalid_history_users = [
                user_id
                for user_id in initial_histories
                if isinstance(user_id, bool) or not isinstance(user_id, int)
            ]
            if invalid_history_users:
                raise TypeError("initial-history user IDs must be integers")
            if set(initial_histories) != set(profiles_by_user):
                raise ValueError(
                    "initial histories must exactly match the profile population"
                )

        states: dict[UserId, _UserState] = {}
        for user_id, artifact in profiles_by_user.items():
            if isinstance(user_id, bool) or not isinstance(user_id, int):
                raise TypeError("profile-map user IDs must be integers")
            if not isinstance(artifact, ProfileArtifact):
                raise TypeError("profiles_by_user must contain ProfileArtifact values")
            if artifact.user_id != user_id:
                raise ValueError("profile map key does not match artifact user_id")
            profile_text = support.profile_text(
                artifact.profile_text,
                "profile_text",
            )
            states[user_id] = _UserState(
                source_profile_fingerprint=artifact.profile_fingerprint,
                initial_profile_text=profile_text,
                current_profile_text=profile_text,
                interactions=[],
                reflections=[],
                historical_memory_occurrences_by_item={},
                memory_occurrences_by_item={},
                tipi_scores=scores_by_user.get(user_id),
            )

        item_texts: dict[ItemId, str] = {}
        for item_id, text in item_text_by_id.items():
            if isinstance(item_id, bool) or not isinstance(item_id, int):
                raise TypeError("item-text keys must be integers")
            if item_id <= 0:
                raise ValueError("item IDs must be positive")
            item_texts[item_id] = support.clean_text(
                text,
                f"item_text[{item_id}]",
            )
        if not item_texts:
            raise ValueError("item_text_by_id cannot be empty")
        if memory_treatment is not None:
            assert memory_retrieval_limit is not None
            for user_id, state in states.items():
                history = initial_histories[user_id]
                if isinstance(history, (str, bytes)) or not isinstance(
                    history,
                    Sequence,
                ):
                    raise TypeError(
                        f"initial history for user {user_id} must be an item sequence"
                    )
                retained_history = tuple(history[-memory_retrieval_limit:])
                historical_seed = seed_historical_memories(
                    retained_history,
                    limit=memory_retrieval_limit,
                )
                missing_seed_text = [
                    item_id
                    for item_id in historical_seed
                    if item_id not in item_texts
                ]
                if missing_seed_text:
                    raise KeyError(
                        "missing item text for historical seed item "
                        f"{missing_seed_text[0]}"
                    )
                state.historical_memory_occurrences_by_item = dict(
                    historical_seed
                )

        self.provider = provider
        self.provider_settings = dict(provider.semantic_settings())
        self.tipi_treatment = tipi_treatment
        self.update_mode = update_mode
        self.memory_limit = memory_limit
        self.history_limit = history_limit
        self.memory_treatment = memory_treatment
        self.memory_decay = normalized_memory_decay
        self.memory_retrieval_limit = memory_retrieval_limit
        self.fatigue_treatment = fatigue_treatment
        self.item_text_by_id = item_texts
        self.item_text_fingerprint = fingerprint_semantic_state(
            item_texts,
            "agentic-recommender-item-text-v1",
        )
        self._states = states
        prompt_contract: dict[str, Any] = {
            "decision_prompt_version": DECISION_PROMPT_VERSION,
            "reflection_prompt_version": REFLECTION_PROMPT_VERSION,
            "continuation_prompt_version": CONTINUATION_PROMPT_VERSION,
            "decision_system": DECISION_SYSTEM_PROMPT,
            "reflection_system": REFLECTION_SYSTEM_PROMPT,
            "continuation_system": CONTINUATION_SYSTEM_PROMPT,
            "planning_prompt": PLAN_PROMPT,
            "planning_schema": PLAN_SCHEMA,
            "memory_schema": MEMORY_SCHEMA,
            "action_schema": ACTION_DECISION_SCHEMA,
            "reflection_schema": REFLECTION_SCHEMA,
            "continuation_schema": CONTINUATION_SCHEMA,
            "module_routing": {
                "modules": MODULES_TO_INVOKE,
                "max_reasoning_steps": MAX_NUMBER_AGENTIC_LOOP,
                "final_action_is_mandatory": True,
            },
        }
        if tipi_treatment is not None:
            prompt_contract.update(
                {
                    "personality_prompt_template": PERSONALITY_PROMPT_TEMPLATE,
                }
            )
        if memory_treatment is not None:
            prompt_contract.update(
                {
                    "treated_memory_prompt_version": TREATED_MEMORY_PROMPT_VERSION,
                    "memory_treatment": memory_treatment,
                    "memory_decay": normalized_memory_decay,
                    "memory_retrieval_limit": memory_retrieval_limit,
                    "memory_tie_rule": MEMORY_TIE_RULE,
                    "treated_module_routing": ("memory", "action"),
                    "memory_sources": (
                        "historical_log",
                        "simulated_click",
                    ),
                    "historical_seed_rule": HISTORICAL_MEMORY_SEED_RULE,
                    "memory_composition_rule": (
                        "all_online_chunks_plus_newest_historical_fill_v1"
                    ),
                }
            )
        if fatigue_treatment is not None:
            prompt_contract.update(
                {
                    "rank_step_fatigue": {
                        "implementation_version": (
                            RANK_STEP_FATIGUE_IMPLEMENTATION_VERSION
                        ),
                        "prompt_version": RANK_STEP_FATIGUE_PROMPT_VERSION,
                        "treatment": fatigue_treatment,
                        "bucket_size": RANK_STEP_FATIGUE_BUCKET_SIZE,
                        "top_click_delta": RANK_STEP_FATIGUE_TOP_CLICK_DELTA,
                        "floor": RANK_STEP_FATIGUE_FLOOR,
                        "no_click_delta_rule": "ceil(slate_size / bucket_size)",
                        "fatigue_prompt": RANK_STEP_FATIGUE_PROMPT,
                        "continuation_schema": CONTINUATION_SCHEMA,
                    }
                }
            )
        self.prompt_fingerprint = fingerprint_semantic_state(
            prompt_contract,
            "agentic-recommender-prompt-contract-v2",
        )
        logger.info(
            "Agentic model initialized variant=%s update_mode=%s "
            "tipi_treatment=%s users=%s "
            "items=%s prompt_fingerprint=%s item_text_fingerprint=%s",
            self.VARIANT_ID,
            self.update_mode,
            self.tipi_treatment,
            len(self._states),
            len(self.item_text_by_id),
            self.prompt_fingerprint,
            self.item_text_fingerprint,
        )

    # Profile module -----------------------------------------------------

    def current_profile(self, user_id: UserId) -> str:
        return self._state_for(user_id).current_profile_text

    def _state_for(self, user_id: UserId) -> _UserState:
        try:
            return self._states[user_id]
        except KeyError as exc:
            raise KeyError(f"no profile state for user {user_id}") from exc

    def _profile_fingerprint(self, user_id: UserId) -> str:
        state = self._state_for(user_id)
        return fingerprint_semantic_state(
            {
                "source_profile_fingerprint": state.source_profile_fingerprint,
                "current_profile_text": state.current_profile_text,
            },
            "agentic-current-profile-v1",
        )

    @staticmethod
    def _personality_block(state: _UserState) -> str:
        if state.tipi_scores is None:
            return ""
        return PERSONALITY_PROMPT_TEMPLATE.format(
            **state.tipi_scores.as_ocean()
        ) + "\n\n"

    def _add_personality_diagnostics(
        self,
        diagnostics: dict[str, Any],
        state: _UserState,
    ) -> None:
        if self.tipi_treatment is not None:
            diagnostics["tipi_treatment"] = self.tipi_treatment
        if state.tipi_scores is not None:
            diagnostics["tipi_scores"] = dict(state.tipi_scores.as_ocean())

    def _tipi_checkpoint_state(self) -> Mapping[str, Any] | None:
        if self.tipi_treatment is None:
            return None
        return {
            "treatment": self.tipi_treatment,
            "scores_by_user": {
                user_id: dict(state.tipi_scores.as_ocean())
                for user_id, state in self._states.items()
                if state.tipi_scores is not None
            },
        }

    def _fatigue_checkpoint_state(self) -> Mapping[str, Any] | None:
        if self.fatigue_treatment is None:
            return None
        return {
            "treatment": self.fatigue_treatment,
            "implementation_version": RANK_STEP_FATIGUE_IMPLEMENTATION_VERSION,
            "prompt_version": RANK_STEP_FATIGUE_PROMPT_VERSION,
            "bucket_size": RANK_STEP_FATIGUE_BUCKET_SIZE,
            "top_click_delta": RANK_STEP_FATIGUE_TOP_CLICK_DELTA,
            "floor": RANK_STEP_FATIGUE_FLOOR,
            "no_click_delta_rule": "ceil(slate_size / bucket_size)",
        }

    # Planning and action modules ---------------------------------------

    def respond(self, turn: TurnInput) -> ActionDecision:
        if not isinstance(turn, TurnInput):
            raise TypeError("turn must be a TurnInput")
        state = self._state_for(turn.user_id)
        if self.tipi_treatment is not None and state.tipi_scores is None:
            raise ValueError(f"TIPI scores are missing for user {turn.user_id}")
        if not turn.presented_slate or len(turn.presented_slate) != len(
            set(turn.presented_slate)
        ):
            raise ValueError("presented_slate must contain unique items")
        observable_items = turn.presented_slate
        if self.memory_treatment is None:
            observable_items = (*observable_items, *turn.observed_history)
        missing = [
            item_id
            for item_id in observable_items
            if item_id not in self.item_text_by_id
        ]
        if missing:
            raise KeyError(f"missing item text for item {missing[0]}")

        # Planning and memory may exchange concise structured suggestions. The
        # action module is invoked exactly once afterwards and is the only
        # module allowed to produce observable behavior.
        module_trace: list[dict[str, Any]] = []
        provider_steps: list[dict[str, Any]] = []
        response_ids: list[str] = []
        forced_action = False
        memory_diagnostics: dict[str, Any] | None = None

        if self.memory_treatment is not None:
            composed_memory = self._treated_memory_view(turn, state)
            memory_diagnostics = self._treated_memory_diagnostics(
                state,
                composed_memory,
            )
            logger.info(
                "Agentic module started user=%s interaction=%s iteration=%s "
                "step=1 module=memory treatment=%s",
                turn.user_id,
                turn.interaction_id,
                turn.iteration,
                self.memory_treatment,
            )
            try:
                result = self.provider.call_json(
                    system_prompt=DECISION_SYSTEM_PROMPT,
                    user_prompt=self._treated_memory_prompt(
                        turn,
                        state,
                        composed_memory,
                    ),
                    response_schema=_memory_schema(turn.presented_slate),
                    validator=lambda payload: support.validate_memory(
                        payload,
                        turn.presented_slate,
                    ),
                    schema_name="agentic_recommender_memory",
                )
            except InvalidProviderResponse as exc:
                logger.error(
                    "Agentic module invalid user=%s interaction=%s iteration=%s "
                    "step=1 module=memory error=%s",
                    turn.user_id,
                    turn.interaction_id,
                    turn.iteration,
                    exc,
                )
                return self._failed_action(
                    turn,
                    ActionStatus.INVALID,
                    str(exc),
                    memory_diagnostics=memory_diagnostics,
                )
            except ProviderError as exc:
                logger.error(
                    "Agentic module failed user=%s interaction=%s iteration=%s "
                    "step=1 module=memory error=%s",
                    turn.user_id,
                    turn.interaction_id,
                    turn.iteration,
                    exc,
                )
                return self._failed_action(
                    turn,
                    ActionStatus.TECHNICAL_FAILURE,
                    str(exc),
                    memory_diagnostics=memory_diagnostics,
                )
            payload = dict(result.payload)
            module_trace.append({"module": "memory", **payload})
            provider_steps.append(support.provider_step("memory", result))
            if result.response_id:
                response_ids.append(result.response_id)
            logger.info(
                "Agentic module completed user=%s interaction=%s iteration=%s "
                "step=1 module=memory next_module=action response_id=%s attempts=%s",
                turn.user_id,
                turn.interaction_id,
                turn.iteration,
                result.response_id,
                result.attempts,
            )
        else:
            next_module = "planning"
            for reasoning_step in range(1, MAX_NUMBER_AGENTIC_LOOP + 1):
                if next_module not in {"planning", "memory"}:
                    logger.error(
                        "Agentic module routing invalid user=%s interaction=%s "
                        "iteration=%s step=%s module=%s; falling back to planning",
                        turn.user_id,
                        turn.interaction_id,
                        turn.iteration,
                        reasoning_step,
                        next_module,
                    )
                    next_module = "planning"

                trace_text = support.module_trace_text(module_trace)
                if next_module == "planning":
                    prompt = self._planning_prompt(turn, state, trace_text)
                    schema = _planning_schema(turn.presented_slate)
                    validator = lambda payload: support.validate_planning(
                        payload,
                        turn.presented_slate,
                    )
                else:
                    prompt = self._memory_prompt(turn, state, trace_text)
                    schema = _memory_schema(turn.presented_slate)
                    validator = lambda payload: support.validate_memory(
                        payload,
                        turn.presented_slate,
                    )

                logger.info(
                    "Agentic module started user=%s interaction=%s iteration=%s "
                    "step=%s module=%s",
                    turn.user_id,
                    turn.interaction_id,
                    turn.iteration,
                    reasoning_step,
                    next_module,
                )
                try:
                    result = self.provider.call_json(
                        system_prompt=DECISION_SYSTEM_PROMPT,
                        user_prompt=prompt,
                        response_schema=schema,
                        validator=validator,
                        schema_name=f"agentic_recommender_{next_module}",
                    )
                except InvalidProviderResponse as exc:
                    logger.error(
                        "Agentic module invalid user=%s interaction=%s iteration=%s "
                        "step=%s module=%s error=%s",
                        turn.user_id,
                        turn.interaction_id,
                        turn.iteration,
                        reasoning_step,
                        next_module,
                        exc,
                    )
                    return self._failed_action(
                        turn,
                        ActionStatus.INVALID,
                        str(exc),
                    )
                except ProviderError as exc:
                    logger.error(
                        "Agentic module failed user=%s interaction=%s iteration=%s "
                        "step=%s module=%s error=%s",
                        turn.user_id,
                        turn.interaction_id,
                        turn.iteration,
                        reasoning_step,
                        next_module,
                        exc,
                    )
                    return self._failed_action(
                        turn,
                        ActionStatus.TECHNICAL_FAILURE,
                        str(exc),
                    )

                payload = dict(result.payload)
                module_trace.append({"module": next_module, **payload})
                provider_steps.append(support.provider_step(next_module, result))
                if result.response_id:
                    response_ids.append(result.response_id)

                if next_module == "planning":
                    proposed_next_module = payload["module_invoked"]
                else:
                    proposed_next_module = "planning"
                logger.info(
                    "Agentic module completed user=%s interaction=%s iteration=%s "
                    "step=%s module=%s next_module=%s response_id=%s attempts=%s",
                    turn.user_id,
                    turn.interaction_id,
                    turn.iteration,
                    reasoning_step,
                    next_module,
                    proposed_next_module,
                    result.response_id,
                    result.attempts,
                )
                if proposed_next_module == "action":
                    break
                next_module = proposed_next_module
            else:
                forced_action = True
                logger.warning(
                    "Agentic reasoning limit reached; forcing action user=%s "
                    "interaction=%s iteration=%s max_steps=%s",
                    turn.user_id,
                    turn.interaction_id,
                    turn.iteration,
                    MAX_NUMBER_AGENTIC_LOOP,
                )

        action_step = len(module_trace) + 1
        logger.info(
            "Agentic module started user=%s interaction=%s iteration=%s "
            "step=%s module=action",
            turn.user_id,
            turn.interaction_id,
            turn.iteration,
            action_step,
        )
        try:
            result = self.provider.call_json(
                system_prompt=DECISION_SYSTEM_PROMPT,
                user_prompt=self._action_prompt(
                    turn,
                    state,
                    support.module_trace_text(module_trace),
                ),
                response_schema=_action_schema(turn.presented_slate),
                validator=lambda payload: support.validate_action(
                    payload,
                    turn.presented_slate,
                ),
                schema_name="agentic_recommender_action",
            )
        except InvalidProviderResponse as exc:
            logger.error(
                "Agentic module invalid user=%s interaction=%s iteration=%s "
                "step=%s module=action error=%s",
                turn.user_id,
                turn.interaction_id,
                turn.iteration,
                action_step,
                exc,
            )
            return self._failed_action(
                turn,
                ActionStatus.INVALID,
                str(exc),
                memory_diagnostics=memory_diagnostics,
            )
        except ProviderError as exc:
            logger.error(
                "Agentic module failed user=%s interaction=%s iteration=%s "
                "step=%s module=action error=%s",
                turn.user_id,
                turn.interaction_id,
                turn.iteration,
                action_step,
                exc,
            )
            return self._failed_action(
                turn,
                ActionStatus.TECHNICAL_FAILURE,
                str(exc),
                memory_diagnostics=memory_diagnostics,
            )

        payload = dict(result.payload)
        module_trace.append({"module": "action", **payload})
        provider_steps.append(support.provider_step("action", result))
        if result.response_id:
            response_ids.append(result.response_id)
        suggestion = support.latest_suggestion(module_trace[:-1])
        suggestion_agreement = (
            suggestion is not None
            and suggestion[0] == payload["action"]
            and suggestion[1] == payload["selected_item_id"]
        )
        logger.info(
            "Agentic module completed user=%s interaction=%s iteration=%s "
            "step=%s module=action action=%s selected_item_id=%s "
            "suggestion_agreement=%s response_id=%s attempts=%s",
            turn.user_id,
            turn.interaction_id,
            turn.iteration,
            action_step,
            payload["action"],
            payload["selected_item_id"],
            suggestion_agreement,
            result.response_id,
            result.attempts,
        )
        action = (
            UserAction.CLICK
            if payload["action"] == "click"
            else UserAction.NO_CLICK
        )
        plan_summary = support.module_trace_summary(module_trace[:-1])
        logger.info(
            "Agentic decision completed user=%s interaction=%s iteration=%s "
            "action=%s selected_item_id=%s reasoning_steps=%s forced_action=%s",
            turn.user_id,
            turn.interaction_id,
            turn.iteration,
            payload["action"],
            payload["selected_item_id"],
            len(module_trace) - 1,
            forced_action,
        )
        diagnostics = {
            "user_id": turn.user_id,
            "interaction_id": turn.interaction_id,
            "iteration": turn.iteration,
            "presented_slate": turn.presented_slate,
            "plan": plan_summary,
            "reason": payload["reason"],
            "provider_attempts": sum(
                int(step["attempts"]) for step in provider_steps
            ),
            "provider_usage": support.aggregate_provider_usage(provider_steps),
            "provider_steps": tuple(provider_steps),
            "profile_fingerprint": self._profile_fingerprint(turn.user_id),
            "prompt_fingerprint": self.prompt_fingerprint,
            "prompt_version": DECISION_PROMPT_VERSION,
            "item_text_fingerprint": self.item_text_fingerprint,
            "module_trace": tuple(module_trace),
            "forced_action": forced_action,
            "suggested_action": suggestion[0] if suggestion else None,
            "suggested_selected_item_id": suggestion[1] if suggestion else None,
            "suggestion_agreement": suggestion_agreement,
        }
        if memory_diagnostics is not None:
            diagnostics["memory_view"] = memory_diagnostics
        self._add_personality_diagnostics(diagnostics, state)
        return ActionDecision(
            decision_id=f"{turn.interaction_id}__agentic_recommender",
            status=ActionStatus.VALID,
            action=action,
            clicked_item_id=payload["selected_item_id"],
            reason=payload["reason"],
            trace_ids=tuple(response_ids),
            diagnostics=diagnostics,
        )

    def _failed_action(
        self,
        turn: TurnInput,
        status: ActionStatus,
        reason: str,
        *,
        memory_diagnostics: Mapping[str, Any] | None = None,
    ) -> ActionDecision:
        diagnostics = {
            "user_id": turn.user_id,
            "interaction_id": turn.interaction_id,
            "iteration": turn.iteration,
            "prompt_fingerprint": self.prompt_fingerprint,
        }
        if memory_diagnostics is not None:
            diagnostics["memory_view"] = dict(memory_diagnostics)
        return ActionDecision(
            decision_id=f"{turn.interaction_id}__agentic_recommender",
            status=status,
            action=UserAction.UNAVAILABLE,
            clicked_item_id=None,
            reason=reason,
            diagnostics=diagnostics,
        )

    def _planning_prompt(
        self,
        turn: TurnInput,
        state: _UserState,
        module_trace_text: str = "",
    ) -> str:
        slate_text = support.numbered_items(
            turn.presented_slate,
            self.item_text_by_id,
        )
        prior_steps = (
            f"\n\nPrior module outputs:\n{module_trace_text}"
            if module_trace_text
            else ""
        )
        return (
            f"{PLAN_PROMPT}\n\n"
            f"Current profile:\n{state.current_profile_text}\n\n"
            f"{self._personality_block(state)}"
            "Recent reflection memory:\n"
            f"{_reflection_memory_text(state.reflections, self.memory_limit)}\n\n"
            f"Displayed books in exact rank order:\n{slate_text}\n\n"
            "Suggest click with one displayed item ID, or no_click with "
            "suggested_selected_item_id=null when none is appealing enough."
            f"{prior_steps}"
        )

    def _memory_prompt(
        self,
        turn: TurnInput,
        state: _UserState,
        module_trace_text: str = "",
    ) -> str:
        history = turn.observed_history[-self.history_limit :]
        history_text = support.numbered_items(history, self.item_text_by_id)
        slate_text = support.numbered_items(
            turn.presented_slate,
            self.item_text_by_id,
        )
        prior_steps = (
            f"\n\nPrior module outputs:\n{module_trace_text}"
            if module_trace_text
            else ""
        )
        return (
            "You now have access to memory of prior books.\n\n"
            f"Current profile:\n{state.current_profile_text}\n\n"
            f"{self._personality_block(state)}"
            f"Recent read/clicked history:\n{history_text}\n\n"
            "Recent observable interaction memory:\n"
            f"{_interaction_memory_text(state.interactions, self.memory_limit)}\n\n"
            "Recent reflection memory:\n"
            f"{_reflection_memory_text(state.reflections, self.memory_limit)}\n\n"
            f"Displayed books in exact rank order:\n{slate_text}\n\n"
            "Suggest click with one displayed item ID, or no_click with "
            "suggested_selected_item_id=null when none is appealing enough."
            f"{prior_steps}"
        )

    def _treated_memory_view(
        self,
        turn: TurnInput,
        state: _UserState,
    ) -> _ComposedMemoryView:
        if (
            self.memory_treatment is None
            or self.memory_decay is None
            or self.memory_retrieval_limit is None
        ):
            raise RuntimeError("treated memory view requested without treatment")
        candidates = compose_memory_candidates(
            state.historical_memory_occurrences_by_item,
            state.memory_occurrences_by_item,
            capacity=self.memory_retrieval_limit,
        )
        online_item_ids = tuple(sorted(state.memory_occurrences_by_item))
        online_item_set = set(online_item_ids)
        historical_fill_item_ids = tuple(
            sorted(
                (
                    item_id
                    for item_id in candidates
                    if item_id not in online_item_set
                ),
                key=lambda item_id: (
                    -max(state.historical_memory_occurrences_by_item[item_id]),
                    item_id,
                ),
            )
        )
        retrieval = retrieve_memories(
            candidates,
            current_timestep=turn.iteration,
            decay=self.memory_decay,
            retrieval_limit=self.memory_retrieval_limit,
        )
        return _ComposedMemoryView(
            retrieval=retrieval,
            historical_fill_item_ids=historical_fill_item_ids,
            online_item_ids=online_item_ids,
        )

    def _treated_memory_diagnostics(
        self,
        state: _UserState,
        composed: _ComposedMemoryView,
    ) -> dict[str, Any]:
        if self.memory_treatment is None:
            raise RuntimeError("treated memory diagnostics require a treatment")
        view = composed.retrieval
        view_payload = view.to_dict()
        ledger_payload = {
            item_id: tuple(timesteps)
            for item_id, timesteps in sorted(
                state.memory_occurrences_by_item.items()
            )
        }
        retrieved_item_ids = [entry.item_id for entry in view.entries]
        retrieved_item_set = set(retrieved_item_ids)
        online_item_set = set(composed.online_item_ids)
        entry_payloads = [
            {
                **entry.to_dict(),
                "source": (
                    "simulated_click"
                    if entry.item_id in online_item_set
                    else "historical_log"
                ),
            }
            for entry in view.entries
        ]
        historical_seed_payload = {
            item_id: tuple(timesteps)
            for item_id, timesteps in sorted(
                state.historical_memory_occurrences_by_item.items()
            )
        }
        fingerprint_payload = {
            "treatment": self.memory_treatment,
            "tie_rule": MEMORY_TIE_RULE,
            "view": {**view_payload, "entries": entry_payloads},
        }
        return {
            "schema_version": view_payload["schema_version"],
            "treatment": self.memory_treatment,
            "current_timestep": view.current_timestep,
            "d": view.decay,
            "K": view.retrieval_limit,
            "forced_invocation": True,
            "source": "historical_log_plus_simulated_click",
            "historical_seed_rule": HISTORICAL_MEMORY_SEED_RULE,
            "historical_seed_event_count": sum(
                len(timesteps)
                for timesteps in (
                    state.historical_memory_occurrences_by_item.values()
                )
            ),
            "historical_seed_fingerprint": fingerprint_semantic_state(
                historical_seed_payload,
                "agentic-historical-memory-seed-v1",
            ),
            "historical_fill_item_ids": list(
                composed.historical_fill_item_ids
            ),
            "historical_fill_count": len(composed.historical_fill_item_ids),
            "online_item_ids": list(composed.online_item_ids),
            "online_item_count": len(composed.online_item_ids),
            "ledger_item_count": len(state.memory_occurrences_by_item),
            "ledger_occurrence_count": sum(
                len(timesteps)
                for timesteps in state.memory_occurrences_by_item.values()
            ),
            "ledger_fingerprint": fingerprint_semantic_state(
                ledger_payload,
                "agentic-online-click-memory-ledger-v1",
            ),
            "candidate_count": view.candidate_count,
            "retrieved_item_ids": retrieved_item_ids,
            "retrieved_sources_by_item": {
                entry.item_id: (
                    "simulated_click"
                    if entry.item_id in online_item_set
                    else "historical_log"
                )
                for entry in view.entries
            },
            "not_retrieved_item_ids": [
                item_id
                for item_id in ledger_payload
                if item_id not in retrieved_item_set
            ],
            "retrieved_ages_by_item": {
                entry.item_id: list(entry.occurrence_ages)
                for entry in view.entries
            },
            "activation_by_item": {
                entry.item_id: entry.activation for entry in view.entries
            },
            "accessibility_by_item": {
                entry.item_id: entry.accessibility for entry in view.entries
            },
            "retrieved_accessibility_mass": (
                view.retrieved_accessibility_mass
            ),
            "tie_rule": MEMORY_TIE_RULE,
            "entries": entry_payloads,
            "fingerprint": fingerprint_semantic_state(
                fingerprint_payload,
                "agentic-treated-memory-view-v1",
            ),
            "prompt_version": TREATED_MEMORY_PROMPT_VERSION,
        }

    def _treated_memory_prompt(
        self,
        turn: TurnInput,
        state: _UserState,
        composed: _ComposedMemoryView,
    ) -> str:
        slate_text = support.numbered_items(
            turn.presented_slate,
            self.item_text_by_id,
        )
        memory_text = _treated_memory_text(composed, self.item_text_by_id)
        return (
            "The memory available for this decision is limited to the "
            "retrieved chunks below. Each chunk identifies whether it came "
            "from the bounded historical log seed or a simulated click. Do "
            "not infer older log events, unlisted online interactions, or "
            "reflection notes. The fixed background profile remains available "
            "and does not change during this experiment.\n\n"
            f"Current fixed profile:\n{state.current_profile_text}\n\n"
            f"{self._personality_block(state)}"
            f"Accessible memory chunks:\n{memory_text}\n\n"
            f"Displayed books in exact rank order:\n{slate_text}\n\n"
            "Suggest click with one displayed item ID, or no_click with "
            "suggested_selected_item_id=null when none is appealing enough."
        )

    def _action_prompt(
        self,
        turn: TurnInput,
        state: _UserState,
        module_trace_text: str = "",
    ) -> str:
        slate_text = support.numbered_items(
            turn.presented_slate,
            self.item_text_by_id,
        )
        prior_steps = (
            f"\n\nPrior module outputs:\n{module_trace_text}"
            if module_trace_text
            else ""
        )
        return (
            "You are now taking the final observable action.\n\n"
            f"Current profile:\n{state.current_profile_text}\n\n"
            f"{self._personality_block(state)}"
            f"Displayed books in exact rank order:\n{slate_text}\n\n"
            "Choose click with one displayed item ID, or no_click with "
            "selected_item_id=null when none is appealing enough. Give one "
            "short observable reason for the final action."
            f"{prior_steps}"
        )

    # Memory and reflection modules -------------------------------------

    def update(
        self,
        action: ActionDecision,
        context: Mapping[str, Any],
    ) -> UserUpdateResult:
        if self.update_mode == "frozen":
            return UserUpdateResult(
                update_id=f"{action.decision_id}__update_skipped",
                status=UpdateStatus.SKIPPED,
                diagnostics={"reason": "user_state_frozen"},
            )
        if action.status is not ActionStatus.VALID:
            return UserUpdateResult(
                update_id=f"{action.decision_id}__update_skipped",
                status=UpdateStatus.SKIPPED,
                diagnostics={"reason": "action_not_valid"},
            )
        turn = support.validated_update_context(
            action,
            context,
            expected_prompt_fingerprint=self.prompt_fingerprint,
        )
        memory = _interaction_from_action(action)
        user_id = turn.user_id
        state = self._state_for(user_id)
        if any(
            existing.interaction_id == memory.interaction_id
            for existing in state.interactions
        ):
            raise ValueError("the same interaction cannot update user state twice")
        fatigue_transition = None
        if self.fatigue_treatment is not None:
            fatigue_transition = update_rank_step_fatigue(
                state.fatigue_value,
                displayed_rank_by_item=turn.displayed_rank_by_item,
                clicked_item_id=action.clicked_item_id,
            )
        occurrence_timesteps: list[int] | None = None
        if (
            self.memory_treatment is not None
            and action.action is UserAction.CLICK
        ):
            if action.clicked_item_id is None:
                raise ValueError("valid click must name one clicked item")
            occurrence_timesteps = state.memory_occurrences_by_item.setdefault(
                action.clicked_item_id,
                [],
            )
            if memory.iteration in occurrence_timesteps:
                raise ValueError(
                    "one user cannot record the same clicked item twice in one "
                    "canonical timestep"
                )

        # Factual interaction memory is deterministic and commits for every
        # valid behavioral click or no-click. Reflection is a separate step.
        state.interactions.append(memory)
        changed_domains = ["interaction_memory"]
        diagnostics: dict[str, Any] = {
            "update_mode": self.update_mode,
            "reflection_status": "not_requested",
        }
        if fatigue_transition is not None:
            state.fatigue_value = fatigue_transition.value_after
            diagnostics["rank_step_fatigue"] = {
                **fatigue_transition.to_dict(),
                "bucket_size": RANK_STEP_FATIGUE_BUCKET_SIZE,
                "treatment": self.fatigue_treatment,
            }
            if fatigue_transition.value_after != fatigue_transition.value_before:
                changed_domains.append("fatigue_value")
        if occurrence_timesteps is not None:
            occurrence_timesteps.append(memory.iteration)
            occurrence_timesteps.sort()
            changed_domains.append("memory_occurrence_ledger")
            diagnostics["memory_occurrence_recorded"] = True
            diagnostics["memory_occurrence_item_id"] = action.clicked_item_id
            diagnostics["memory_occurrence_timestep"] = memory.iteration
        elif self.memory_treatment is not None:
            diagnostics["memory_occurrence_recorded"] = False
        if self.memory_treatment is not None:
            diagnostics["memory_ledger_occurrence_count"] = sum(
                len(timesteps)
                for timesteps in state.memory_occurrences_by_item.values()
            )
        update_trace_ids: tuple[str, ...] = ()

        if self.update_mode != "no_reflection_control":
            reflection = self._reflect(user_id, memory)
            diagnostics.update(reflection["diagnostics"])
            update_trace_ids = tuple(reflection["trace_ids"])
            if reflection["memory"] is not None:
                state.reflections.append(reflection["memory"])
                changed_domains.append("reflection_memory")
                if self.update_mode == "full_reflection":
                    if reflection["updated_profile"] != state.current_profile_text:
                        state.current_profile_text = reflection["updated_profile"]
                        changed_domains.append("profile")
                else:
                    diagnostics["profile_update_suppressed"] = True

        diagnostics["profile_fingerprint"] = self._profile_fingerprint(user_id)
        logger.info(
            "Agentic update applied user=%s interaction=%s iteration=%s "
            "update_mode=%s changed_domains=%s reflection_status=%s",
            user_id,
            memory.interaction_id,
            memory.iteration,
            self.update_mode,
            ",".join(changed_domains),
            diagnostics["reflection_status"],
        )
        return UserUpdateResult(
            update_id=f"{action.decision_id}__user_update",
            status=UpdateStatus.APPLIED,
            changed_domains=tuple(changed_domains),
            trace_ids=update_trace_ids,
            diagnostics=diagnostics,
        )

    def _reflect(
        self,
        user_id: UserId,
        memory: InteractionMemory,
    ) -> Mapping[str, Any]:
        state = self._state_for(user_id)
        prompt = (
            f"Current profile:\n{state.current_profile_text}\n\n"
            f"{self._personality_block(state)}"
            "Recent interaction memory, including the latest decision once:\n"
            f"{_interaction_memory_text(state.interactions, self.memory_limit)}\n\n"
            "Return the complete profile to use next time, plus satisfaction, "
            "and one concise memory note. The profile may remain "
            "unchanged when this interaction provides no new evidence."
        )
        logger.info(
            "Agentic reflection started user=%s interaction=%s",
            user_id, memory.interaction_id,
        )
        try:
            result = self.provider.call_json(
                system_prompt=REFLECTION_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_schema=REFLECTION_SCHEMA,
                validator=support.validate_reflection,
                schema_name="agentic_recommender_reflection",
            )
        except (InvalidProviderResponse, ProviderError) as exc:
            logger.error(
                "Agentic reflection failure was logged and skipped user=%s "
                "interaction=%s error=%s",
                user_id,
                memory.interaction_id,
                exc,
            )
            return {
                "updated_profile": state.current_profile_text,
                "memory": None,
                "trace_ids": (),
                "diagnostics": {
                    "reflection_status": "failed_and_skipped",
                    "reflection_error": str(exc),
                },
            }

        payload = result.payload
        updated_profile = payload["updated_profile"]
        before_profile_fingerprint = self._profile_fingerprint(user_id)
        after_profile_fingerprint = fingerprint_semantic_state(
            {
                "source_profile_fingerprint": state.source_profile_fingerprint,
                "current_profile_text": updated_profile,
            },
            "agentic-current-profile-v1",
        )
        reflection_memory = ReflectionMemory(
            interaction_id=memory.interaction_id,
            satisfaction=payload["satisfaction"],
            note=payload["memory_note"],
        )
        profile_changed = updated_profile != state.current_profile_text
        logger.info(
            "Agentic reflection completed user=%s interaction=%s "
            "satisfaction=%s profile_changed=%s response_id=%s attempts=%s",
            user_id,
            memory.interaction_id,
            payload["satisfaction"],
            profile_changed,
            result.response_id,
            result.attempts,
        )
        return {
            "updated_profile": updated_profile,
            "memory": reflection_memory,
            "trace_ids": (result.response_id,) if result.response_id else (),
            "diagnostics": {
                "reflection_status": "applied",
                "satisfaction": payload["satisfaction"],
                "memory_note": payload["memory_note"],
                "profile_changed": profile_changed,
                "profile_fingerprint_before": before_profile_fingerprint,
                "profile_fingerprint_after_proposed": after_profile_fingerprint,
                "reflection_response_id": result.response_id,
                "reflection_attempts": result.attempts,
                "reflection_usage": dict(result.usage),
                "reflection_prompt_version": REFLECTION_PROMPT_VERSION,
            },
        }

    # Continuation is the action module in participation mode ------------

    def decide_continuation(
        self,
        post_update_context: Mapping[str, Any],
    ) -> ContinuationDecision:
        user_id = support.context_user_id(post_update_context)
        interaction_id = str(
            post_update_context.get("interaction_id")
            or self._latest_interaction_id(user_id)
        )
        if interaction_id != self._latest_interaction_id(user_id):
            raise ValueError("continuation must follow the latest user update")
        state = self._state_for(user_id)
        prompt = (
            f"Current profile:\n{state.current_profile_text}\n\n"
            f"{self._personality_block(state)}"
            "Recent observable interactions:\n"
            f"{_interaction_memory_text(state.interactions, self.memory_limit)}\n\n"
            "Recent reflections:\n"
            f"{_reflection_memory_text(state.reflections, self.memory_limit)}\n\n"
            "Decide whether you stay for a future recommendation or exit "
            "the platform after this opportunity."
        )
        if self.fatigue_treatment is not None:
            prompt += "\n\n" + RANK_STEP_FATIGUE_PROMPT.format(
                fatigue_value=state.fatigue_value
            )
        logger.info(
            "Agentic continuation started user=%s interaction=%s",
            user_id, interaction_id,
        )
        try:
            result = self.provider.call_json(
                system_prompt=CONTINUATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_schema=CONTINUATION_SCHEMA,
                validator=support.validate_continuation,
                schema_name="agentic_recommender_continuation",
            )
        except InvalidProviderResponse as exc:
            logger.error(
                "Agentic continuation invalid user=%s interaction=%s error=%s",
                user_id,
                interaction_id,
                exc,
            )
            return support.failed_continuation(
                interaction_id,
                ContinuationStatus.INVALID,
                str(exc),
            )
        except ProviderError as exc:
            logger.error(
                "Agentic continuation failed user=%s interaction=%s error=%s",
                user_id,
                interaction_id,
                exc,
            )
            return support.failed_continuation(
                interaction_id,
                ContinuationStatus.TECHNICAL_FAILURE,
                str(exc),
            )

        continuation = (
            Continuation.STAY
            if result.payload["continuation"] == "stay"
            else Continuation.EXIT
        )
        logger.info(
            "Agentic continuation completed user=%s interaction=%s "
            "continuation=%s response_id=%s attempts=%s",
            user_id,
            interaction_id,
            continuation.value,
            result.response_id,
            result.attempts,
        )
        diagnostics = {
            "provider_attempts": result.attempts,
            "provider_usage": dict(result.usage),
            "profile_fingerprint": self._profile_fingerprint(user_id),
            "prompt_version": (
                RANK_STEP_FATIGUE_PROMPT_VERSION
                if self.fatigue_treatment is not None
                else CONTINUATION_PROMPT_VERSION
            ),
        }
        if self.fatigue_treatment is not None:
            diagnostics["rank_step_fatigue"] = {
                "value": state.fatigue_value,
                "treatment": self.fatigue_treatment,
                "implementation_version": (
                    RANK_STEP_FATIGUE_IMPLEMENTATION_VERSION
                ),
            }
        self._add_personality_diagnostics(diagnostics, state)
        return ContinuationDecision(
            decision_id=f"{interaction_id}__continuation",
            status=ContinuationStatus.VALID,
            continuation=continuation,
            reason=result.payload["reason"],
            trace_ids=(result.response_id,) if result.response_id else (),
            diagnostics=diagnostics,
        )

    def _latest_interaction_id(self, user_id: UserId) -> str:
        interactions = self._state_for(user_id).interactions
        if not interactions:
            raise ValueError("continuation requires at least one valid interaction")
        return interactions[-1].interaction_id

    # Checkpoint state ---------------------------------------------------

    def export_state(self) -> Mapping[str, Any]:
        exported: dict[str, Any] = {
            "variant_id": self.VARIANT_ID,
            "update_mode": self.update_mode,
            "memory_limit": self.memory_limit,
            "history_limit": self.history_limit,
            "provider_settings": self.provider_settings,
            "prompt_fingerprint": self.prompt_fingerprint,
            "item_text_fingerprint": self.item_text_fingerprint,
            "users": {
                user_id: {
                    "source_profile_fingerprint": (
                        state.source_profile_fingerprint
                    ),
                    "initial_profile_text": state.initial_profile_text,
                    "current_profile_text": state.current_profile_text,
                    "interactions": [
                        asdict(value) for value in state.interactions
                    ],
                    "reflections": [
                        asdict(value) for value in state.reflections
                    ],
                    **(
                        {
                            "historical_memory_occurrences_by_item": {
                                item_id: list(timesteps)
                                for item_id, timesteps in sorted(
                                    state.historical_memory_occurrences_by_item.items()
                                )
                            },
                            "memory_occurrences_by_item": {
                                item_id: list(timesteps)
                                for item_id, timesteps in sorted(
                                    state.memory_occurrences_by_item.items()
                                )
                            }
                        }
                        if self.memory_treatment is not None
                        else {}
                    ),
                    **(
                        {"fatigue_value": state.fatigue_value}
                        if self.fatigue_treatment is not None
                        else {}
                    ),
                }
                for user_id, state in self._states.items()
            },
        }
        if self.tipi_treatment is not None:
            exported["tipi"] = self._tipi_checkpoint_state()
        if self.memory_treatment is not None:
            exported.update(
                {
                    "memory_treatment": self.memory_treatment,
                    "memory_decay": self.memory_decay,
                    "memory_retrieval_limit": self.memory_retrieval_limit,
                    "memory_tie_rule": MEMORY_TIE_RULE,
                    "historical_memory_seed_rule": (
                        HISTORICAL_MEMORY_SEED_RULE
                    ),
                }
            )
        if self.fatigue_treatment is not None:
            exported["fatigue"] = self._fatigue_checkpoint_state()
        return exported

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("variant_id") != self.VARIANT_ID:
            raise ValueError("agentic-recommender state variant mismatch")
        if state.get("tipi") != self._tipi_checkpoint_state():
            raise ValueError("agentic-recommender TIPI treatment mismatch")
        if state.get("fatigue") != self._fatigue_checkpoint_state():
            raise ValueError("agentic-recommender fatigue treatment mismatch")
        expected_memory_settings = {
            "memory_treatment": self.memory_treatment,
            "memory_decay": self.memory_decay,
            "memory_retrieval_limit": self.memory_retrieval_limit,
            "memory_tie_rule": (
                MEMORY_TIE_RULE if self.memory_treatment is not None else None
            ),
            "historical_memory_seed_rule": (
                HISTORICAL_MEMORY_SEED_RULE
                if self.memory_treatment is not None
                else None
            ),
        }
        for field, expected in expected_memory_settings.items():
            if state.get(field) != expected:
                raise ValueError(f"agentic-recommender {field} mismatch")
        for field, expected in (
            ("update_mode", self.update_mode),
            ("memory_limit", self.memory_limit),
            ("history_limit", self.history_limit),
            ("provider_settings", self.provider_settings),
            ("prompt_fingerprint", self.prompt_fingerprint),
            ("item_text_fingerprint", self.item_text_fingerprint),
        ):
            if state.get(field) != expected:
                raise ValueError(f"agentic-recommender {field} mismatch")
        users = state.get("users")
        if not isinstance(users, Mapping) or set(users) != set(self._states):
            raise ValueError("agentic-recommender user population mismatch")

        restored: dict[UserId, _UserState] = {}
        for user_id, current in self._states.items():
            payload = users[user_id]
            if not isinstance(payload, Mapping):
                raise TypeError("agentic-recommender user state must be a mapping")
            if (
                payload.get("source_profile_fingerprint")
                != current.source_profile_fingerprint
                or payload.get("initial_profile_text") != current.initial_profile_text
            ):
                raise ValueError("agentic-recommender source profile mismatch")
            interactions = [
                InteractionMemory(**value) for value in payload["interactions"]
            ]
            if self.fatigue_treatment is None:
                if "fatigue_value" in payload:
                    raise ValueError(
                        "untreated agentic state cannot contain fatigue state"
                    )
                fatigue_value = RANK_STEP_FATIGUE_FLOOR
            else:
                fatigue_value = payload.get("fatigue_value")
                if (
                    isinstance(fatigue_value, bool)
                    or not isinstance(fatigue_value, int)
                ):
                    raise TypeError("fatigue_value must be an integer")
                if fatigue_value < RANK_STEP_FATIGUE_FLOOR:
                    raise ValueError("fatigue_value cannot be negative")
            if self.memory_treatment is None:
                if (
                    "memory_occurrences_by_item" in payload
                    or "historical_memory_occurrences_by_item" in payload
                ):
                    raise ValueError(
                        "untreated agentic state cannot contain memory ledgers"
                    )
                memory_occurrences_by_item: dict[ItemId, list[int]] = {}
                historical_memory_occurrences_by_item: dict[
                    ItemId,
                    tuple[int, ...],
                ] = {}
            else:
                historical_memory_occurrences_by_item = (
                    _validated_historical_memory_seed(
                        payload.get("historical_memory_occurrences_by_item"),
                        expected=current.historical_memory_occurrences_by_item,
                        item_text_by_id=self.item_text_by_id,
                        decay=self.memory_decay,
                        retrieval_limit=self.memory_retrieval_limit,
                    )
                )
                memory_occurrences_by_item = _validated_memory_ledger(
                    payload.get("memory_occurrences_by_item"),
                    item_text_by_id=self.item_text_by_id,
                    interactions=interactions,
                    decay=self.memory_decay,
                    retrieval_limit=self.memory_retrieval_limit,
                )
            restored_profile_text = support.profile_text(
                payload.get("current_profile_text"),
                "current_profile_text",
            )
            if (
                self.memory_treatment is not None
                and restored_profile_text != current.initial_profile_text
            ):
                raise ValueError(
                    "memory treatment checkpoint cannot rewrite the fixed profile"
                )
            restored[user_id] = _UserState(
                source_profile_fingerprint=current.source_profile_fingerprint,
                initial_profile_text=current.initial_profile_text,
                current_profile_text=restored_profile_text,
                interactions=interactions,
                reflections=[
                    ReflectionMemory(**value) for value in payload["reflections"]
                ],
                historical_memory_occurrences_by_item=(
                    historical_memory_occurrences_by_item
                ),
                memory_occurrences_by_item=memory_occurrences_by_item,
                tipi_scores=current.tipi_scores,
                fatigue_value=fatigue_value,
            )
        self._states = restored
        logger.info(
            "Agentic state restored variant=%s users=%s state_fingerprint=%s",
            self.VARIANT_ID,
            len(self._states),
            self.fingerprint(),
        )

    def fingerprint(self) -> str:
        return fingerprint_semantic_state(self.export_state(), self.VARIANT_ID)


def _planning_schema(slate: Sequence[ItemId]) -> Mapping[str, Any]:
    return support.choice_schema(
        PLAN_SCHEMA,
        item_field="suggested_selected_item_id",
        slate=slate,
    )


def _memory_schema(slate: Sequence[ItemId]) -> Mapping[str, Any]:
    return support.choice_schema(
        MEMORY_SCHEMA,
        item_field="suggested_selected_item_id",
        slate=slate,
    )


def _action_schema(slate: Sequence[ItemId]) -> Mapping[str, Any]:
    return support.choice_schema(
        ACTION_DECISION_SCHEMA,
        item_field="selected_item_id",
        slate=slate,
    )


def _interaction_from_action(action: ActionDecision) -> InteractionMemory:
    diagnostics = action.diagnostics
    iteration = diagnostics.get("iteration")
    slate = diagnostics.get("presented_slate")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise ValueError("valid agentic action is missing its iteration")
    if not isinstance(slate, (tuple, list)):
        raise ValueError("valid agentic action is missing its presented slate")
    interaction_id = support.clean_text(
        diagnostics.get("interaction_id"),
        "interaction_id",
    )
    return InteractionMemory(
        interaction_id=interaction_id,
        iteration=iteration,
        presented_slate=tuple(slate),
        plan=support.clean_text(diagnostics.get("plan"), "plan"),
        action=action.action.value,
        clicked_item_id=action.clicked_item_id,
        reason=support.clean_text(action.reason, "reason"),
    )


def _one_interaction_text(memory: InteractionMemory) -> str:
    return (
        f"interaction={memory.interaction_id}; action={memory.action}; "
        f"clicked_item_id={memory.clicked_item_id}; plan={memory.plan}; "
        f"reason={memory.reason}"
    )


def _interaction_memory_text(
    memories: Sequence[InteractionMemory],
    limit: int,
) -> str:
    values = memories[-limit:]
    if not values:
        return "[none]"
    return "\n".join(_one_interaction_text(value) for value in values)


def _reflection_memory_text(
    memories: Sequence[ReflectionMemory],
    limit: int,
) -> str:
    values = memories[-limit:]
    if not values:
        return "[none]"
    return "\n".join(
        f"interaction={value.interaction_id}; satisfaction={value.satisfaction}; "
        f"note={value.note}"
        for value in values
    )


def _treated_memory_text(
    composed: _ComposedMemoryView,
    item_text_by_id: Mapping[ItemId, str],
) -> str:
    view = composed.retrieval
    if not view.entries:
        return "[none]"
    online_item_ids = set(composed.online_item_ids)
    return "\n".join(
        (
            f"memory_rank={entry.rank}; item_id={entry.item_id}; "
            "source="
            f"{'simulated_click' if entry.item_id in online_item_ids else 'historical_log'}; "
            f"occurrence_ages={list(entry.occurrence_ages)}; "
            f"activation={entry.activation:.12g}; "
            f"accessibility={entry.accessibility:.12g}; "
            f"book={item_text_by_id[entry.item_id]}"
        )
        for entry in view.entries
    )


def _validated_historical_memory_seed(
    payload: Any,
    *,
    expected: Mapping[ItemId, Sequence[int]],
    item_text_by_id: Mapping[ItemId, str],
    decay: float | None,
    retrieval_limit: int | None,
) -> dict[ItemId, tuple[int, ...]]:
    if not isinstance(payload, Mapping):
        raise TypeError("historical memory seed must be a mapping")
    if decay is None or retrieval_limit is None:
        raise RuntimeError("historical seed validation requires memory settings")
    normalized: dict[ItemId, tuple[int, ...]] = {}
    for item_id, raw_timesteps in payload.items():
        if isinstance(item_id, bool) or not isinstance(item_id, int):
            raise TypeError("historical memory seed item IDs must be integers")
        if item_id not in item_text_by_id:
            raise ValueError("historical memory seed contains an unknown item")
        if isinstance(raw_timesteps, (str, bytes)) or not isinstance(
            raw_timesteps,
            Sequence,
        ):
            raise TypeError("historical memory seed timesteps must be a sequence")
        normalized[item_id] = tuple(sorted(raw_timesteps))
    retrieve_memories(
        normalized,
        current_timestep=1,
        decay=decay,
        retrieval_limit=retrieval_limit,
    )
    expected_normalized = {
        item_id: tuple(timesteps)
        for item_id, timesteps in expected.items()
    }
    if normalized != expected_normalized:
        raise ValueError(
            "checkpointed historical memory seed does not match initial history"
        )
    return normalized


def _validated_memory_ledger(
    payload: Any,
    *,
    item_text_by_id: Mapping[ItemId, str],
    interactions: Sequence[InteractionMemory],
    decay: float | None,
    retrieval_limit: int | None,
) -> dict[ItemId, list[int]]:
    if not isinstance(payload, Mapping):
        raise TypeError("agentic memory occurrence ledger must be a mapping")
    if decay is None or retrieval_limit is None:
        raise RuntimeError("memory ledger validation requires treatment settings")
    normalized: dict[ItemId, list[int]] = {}
    for item_id, raw_timesteps in payload.items():
        if isinstance(item_id, bool) or not isinstance(item_id, int):
            raise TypeError("memory occurrence ledger item IDs must be integers")
        if item_id not in item_text_by_id:
            raise ValueError("memory occurrence ledger contains an unknown item")
        if isinstance(raw_timesteps, (str, bytes)) or not isinstance(
            raw_timesteps,
            Sequence,
        ):
            raise TypeError("memory occurrence timesteps must be a sequence")
        normalized[item_id] = list(raw_timesteps)

    latest_timestep = max(
        (
            timestep
            for timesteps in normalized.values()
            for timestep in timesteps
            if isinstance(timestep, int) and not isinstance(timestep, bool)
        ),
        default=0,
    )
    retrieve_memories(
        normalized,
        current_timestep=latest_timestep,
        decay=decay,
        retrieval_limit=retrieval_limit,
    )
    normalized = {
        item_id: sorted(timesteps)
        for item_id, timesteps in normalized.items()
    }
    expected: dict[ItemId, list[int]] = {}
    for interaction in interactions:
        if interaction.action != UserAction.CLICK.value:
            continue
        if interaction.clicked_item_id is None:
            raise ValueError("checkpointed click lacks a clicked item")
        expected.setdefault(interaction.clicked_item_id, []).append(
            interaction.iteration
        )
    expected = {
        item_id: sorted(timesteps)
        for item_id, timesteps in expected.items()
    }
    if normalized != expected:
        raise ValueError(
            "memory occurrence ledger does not match canonical clicked interactions"
        )
    return normalized


__all__ = [
    "AgenticRecommenderResponseModel",
    "FatigueTreatment",
    "InteractionMemory",
    "MemoryTreatment",
    "ReflectionMemory",
]
