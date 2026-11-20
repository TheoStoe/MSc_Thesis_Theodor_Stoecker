"""Feedback-loop experiment methodology and deliberate condition matrix.

Model identity, user-state adaptation, response-model fitting, and population
replacement are separate choices.  This prevents names such as ``adaptive``
from implying that a state update occurred in a stage where updates are frozen.

Each condition/seed is an isolated job with its own checkpoint, log, and output
folder.  Campaigns are only lists of those complete named jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .feedback_loops import (
    FEEDBACK_LOOP_PRESETS,
    L0_CONTROL,
    L1_FEATURE,
    L2_ML_MODEL,
    L3_INDIVIDUAL,
    L4_SAMPLING,
    L5_OUTCOME,
)


L0 = L0_CONTROL.name
L1 = L1_FEATURE.name
L2 = L2_ML_MODEL.name
L3 = L3_INDIVIDUAL.name
L4 = L4_SAMPLING.name
L5 = L5_OUTCOME.name
CORE_STAGES = (L0, L1, L2, L3, L4, L5)
FROZEN_USER_STAGES = (L0, L1, L2)


@dataclass(frozen=True)
class ResponseModelSpec:
    """Capabilities required by the planned response-model condition.

    This declares the methodology. The runtime registry must independently
    confirm that the current adapter actually implements each capability before
    a job can run.
    """

    name: str
    response_protocol: str
    role: str
    supports_user_state_update: bool = False
    supports_continuation: bool = False


@dataclass(frozen=True)
class ExperimentCondition:
    """One complete, named, independently runnable scientific condition.

    ``analysis_role`` is an analysis-only label explaining why the condition
    exists, such as primary stage, ablation, control, or non-persistent probe.
    It never changes runtime behavior.
    """

    condition_id: str
    feedback_loop: str
    recommender: str
    user_model: str
    user_update_policy: str
    candidate_policy: str
    presentation_policy: str
    campaign_policy: str
    population_policy: str
    response_protocol: str
    realized_loop_types: tuple[str, ...]
    analysis_bundle: str
    analysis_role: str
    purpose: str
    tipi_treatment: str | None = None
    memory_treatment: str | None = None
    memory_decay: float | None = None
    memory_retrieval_limit: int | None = None
    fatigue_treatment: str | None = None
    intervention: str = "control"


AGENTIC_RECOMMENDER = ResponseModelSpec(
    name="agentic_recommender",
    response_protocol="click_or_no_click",
    role="Lusifer-profile-conditioned reflective LLM-agent response model.",
    supports_user_state_update=True,
    supports_continuation=True,
)

POSITION_BIASED = ResponseModelSpec(
    name="position_biased",
    response_protocol="forced_single_click_rank_baseline",
    role="Mechanistic rank-sensitivity baseline.",
)

DIRECT_LLM = ResponseModelSpec(
    name="direct_llm",
    response_protocol="click_or_no_click",
    role="Non-agentic LLM response baseline.",
)

POINTWISE_LOGISTIC = ResponseModelSpec(
    name="pointwise_logistic",
    response_protocol="forced_single_click_calibration",
    role="NCM-paper pointwise logistic baseline under the single-choice protocol.",
)

NCM_GRU_UNIFORM = ResponseModelSpec(
    name="ncm_inspired_gru_uniform",
    response_protocol="forced_single_click_calibration",
    role="NCM-inspired GRU with uniform target placement.",
)

NCM_GRU_MANSOURY = ResponseModelSpec(
    name="ncm_inspired_gru_mansoury",
    response_protocol="forced_single_click_calibration",
    role="NCM-inspired GRU with Mansoury target placement.",
)

NON_AGENTIC_MODELS = (POSITION_BIASED, DIRECT_LLM)
LEARNED_FORCED_CLICK_MODELS = (
    POINTWISE_LOGISTIC,
    NCM_GRU_UNIFORM,
    NCM_GRU_MANSOURY,
)


def population_policy_for_stage(stage: str) -> str:
    """Only L4 and L5 changes membership; its policy also owns reserve activation."""

    return (
        "survivor_similar_reserve_refill"
        if FEEDBACK_LOOP_PRESETS[stage].allow_user_exit
        else "fixed_active_users"
    )


def analysis_bundle_for_stage(stage: str, response_protocol: str) -> str:
    prefix = (
        "forced_click_metrics"
        if response_protocol.startswith("forced_single_click")
        else "click_no_click_metrics"
    )
    return f"{prefix}_through_{stage}"


def realized_loops(
    stage: str,
    *,
    user_update_policy: str,
    population_policy: str,
    campaign_policy: str,
) -> tuple[str, ...]:
    """Describe mechanisms actually realized, not merely the comparison row."""

    preset = FEEDBACK_LOOP_PRESETS[stage]
    loops: list[str] = []
    if preset.exclude_clicked_items:
        loops.append("candidate_eligibility_feedback")
    if preset.update_user_history:
        loops.append("feature_feedback")
    if preset.retrain_recommender:
        loops.append("recommender_ml_model_feedback")
    if user_update_policy in {
        "full_reflection",
        "memory_only_reflection",
        "no_reflection_control",
    }:
        loops.append("individual_feedback")
    if population_policy != "fixed_active_users":
        loops.append("sampling_feedback")
    if campaign_policy == "per_user_stop_after_3_consecutive_sponsored_non_clicks":
        loops.append("outcome_feedback")
    return tuple(loops)


def build_condition(
    model: ResponseModelSpec,
    stage: str,
    *,
    user_update_policy: str = "frozen",
    condition_suffix: str = "",
    presentation_policy: str = "unchanged_recommender_order",
    campaign_policy: str = "no_campaign",
    population_policy: str | None = None,
    recommender: str = "sasrec",
    analysis_role: str = "primary_stage_cell",
    purpose_suffix: str = "",
) -> ExperimentCondition:
    """Construct a complete condition after checking required capabilities."""

    if stage not in FEEDBACK_LOOP_PRESETS:
        raise ValueError(f"unknown feedback stage: {stage}")
    preset = FEEDBACK_LOOP_PRESETS[stage]
    selected_population_policy = population_policy or population_policy_for_stage(stage)

    if user_update_policy != "frozen" and not model.supports_user_state_update:
        raise ValueError(f"{model.name} cannot apply {user_update_policy}")
    if user_update_policy != "frozen" and not preset.allow_latent_user_state_update:
        raise ValueError(f"{stage} does not permit user-state updates")
    if preset.allow_user_exit and not model.supports_continuation:
        raise ValueError(f"{model.name} has no continuation/exit mechanism")
    has_population_dynamics = selected_population_policy != "fixed_active_users"
    if has_population_dynamics != preset.allow_user_exit:
        raise ValueError("population policy disagrees with feedback stage")
    updates_from_outcomes = (
        campaign_policy
        == "per_user_stop_after_3_consecutive_sponsored_non_clicks"
    )
    if updates_from_outcomes != preset.update_campaign_from_outcomes:
        raise ValueError("campaign policy disagrees with feedback stage")

    return ExperimentCondition(
        condition_id="__".join(
            part
            for part in (
                stage,
                model.name,
                recommender,
                condition_suffix.strip("_"),
            )
            if part
        ),
        feedback_loop=stage,
        recommender=recommender,
        user_model=model.name,
        user_update_policy=user_update_policy,
        candidate_policy="exclude_seen",
        presentation_policy=presentation_policy,
        campaign_policy=campaign_policy,
        population_policy=selected_population_policy,
        response_protocol=model.response_protocol,
        realized_loop_types=realized_loops(
            stage,
            user_update_policy=user_update_policy,
            population_policy=selected_population_policy,
            campaign_policy=campaign_policy,
        ),
        analysis_bundle=analysis_bundle_for_stage(stage, model.response_protocol),
        analysis_role=analysis_role,
        purpose=f"{model.role} {purpose_suffix}".strip(),
    )


# The frozen agent is an L0-L2 response baseline: it may observe the stage's
# public history but never commits private interaction/reflection/profile
# state. Persistent agent-state variants begin at L3. Full reflection is the
# primary mechanism. The no-reflection control still accumulates factual
# interaction memory, so it realizes a restricted individual-feedback path;
# it specifically removes reflection and profile rewriting.
AGENTIC_RECOMMENDER_FROZEN_CELLS = tuple(
    build_condition(
        AGENTIC_RECOMMENDER,
        stage,
        user_update_policy="frozen",
        purpose_suffix="Keeps profile, interaction memory, and reflection state frozen.",
    )
    for stage in FROZEN_USER_STAGES
)

_agentic_recommender_adaptive_cells: list[ExperimentCondition] = []
for stage in (L3, L4):
    for user_update_policy, condition_suffix, analysis_role, purpose_suffix in (
        (
            "full_reflection",
            "__full_reflection",
            "primary_stage_cell",
            "Updates profile and emotional/accessibility memory.",
        ),
        (
            "no_reflection_control",
            "__no_reflection_control",
            "reflection_disabled_control",
            "Keeps factual interaction memory but disables reflection and profile rewriting.",
        ),
        (
            "memory_only_reflection",
            "__fixed_persona_memory_only",
            "restricted_update_ablation",
            "Keeps persona text fixed but updates emotional/accessibility memory.",
        ),
    ):
        _agentic_recommender_adaptive_cells.append(
            build_condition(
                AGENTIC_RECOMMENDER,
                stage,
                user_update_policy=user_update_policy,
                condition_suffix=condition_suffix,
                analysis_role=analysis_role,
                purpose_suffix=purpose_suffix,
            )
        )
AGENTIC_RECOMMENDER_ADAPTIVE_CELLS = tuple(
    _agentic_recommender_adaptive_cells
)


_l3_full_reflection = next(
    condition
    for condition in AGENTIC_RECOMMENDER_ADAPTIVE_CELLS
    if condition.feedback_loop == L3
    and condition.user_update_policy == "full_reflection"
)

# Change this one table to add or adjust a TIPI treatment. ``None`` means that
# no personality block is shown; an empty mapping uses each user's natural
# questionnaire scores.
TIPI_TREATMENTS: dict[str, dict[str, float] | None] = {
    "hidden_control": None,
    "natural": {},
    "openness_low": {"openness": 2.0},
    "openness_high": {"openness": 6.0},
    "agreeableness_low": {"agreeableness": 2.0},
    "agreeableness_high": {"agreeableness": 6.0},
}

TIPI_PERSONALITY_CONDITIONS = tuple(
    replace(
        _l3_full_reflection,
        condition_id=f"{_l3_full_reflection.condition_id}__tipi_{treatment}",
        tipi_treatment=treatment,
        analysis_role="tipi_personality_treatment",
        purpose=(
            "L3 full-reflection agent with TIPI treatment "
            f"{treatment.replace('_', ' ')}."
        ),
    )
    for treatment in TIPI_TREATMENTS
)


_l3_no_reflection_control = next(
    condition
    for condition in AGENTIC_RECOMMENDER_ADAPTIVE_CELLS
    if condition.feedback_loop == L3
    and condition.user_update_policy == "no_reflection_control"
)

# Both memory conditions hold the L3 agent, factual-memory update path, and
# retrieval budget fixed.  The only treatment difference is whether factual
# interaction memories remain fully accessible or decay with age.
MEMORY_FORGETTING_CONDITIONS = (
    replace(
        _l3_no_reflection_control,
        condition_id=f"{_l3_no_reflection_control.condition_id}__memory_forced",
        memory_treatment="forced_memory",
        memory_decay=0.0,
        memory_retrieval_limit=20,
        analysis_bundle="memory_forgetting_metrics",
        analysis_role="memory_forgetting_control",
        purpose=(
            "L3 no-reflection agent with forced no-decay memory retrieval "
            "and a fixed 20-memory budget."
        ),
    ),
    replace(
        _l3_no_reflection_control,
        condition_id=(
            f"{_l3_no_reflection_control.condition_id}__memory_power_law"
        ),
        memory_treatment="power_law_decay",
        memory_decay=0.5,
        memory_retrieval_limit=20,
        analysis_bundle="memory_forgetting_metrics",
        analysis_role="memory_forgetting_treatment",
        purpose=(
            "L3 no-reflection agent with power-law memory decay d=0.5 "
            "and a fixed 20-memory budget."
        ),
    ),
)


_l4_no_reflection_control = next(
    condition
    for condition in AGENTIC_RECOMMENDER_ADAPTIVE_CELLS
    if condition.feedback_loop == L4
    and condition.user_update_policy == "no_reflection_control"
)

# The existing L4 no-reflection condition is the unchanged control.  The one
# new condition differs only by maintaining a deterministic cumulative fatigue
# value and appending that value to the ordinary continuation prompt.
RANK_STEP_FATIGUE_TREATMENT_CONDITIONS = (
    replace(
        _l4_no_reflection_control,
        condition_id=(
            f"{_l4_no_reflection_control.condition_id}"
            "__rank_step_cumulative_fatigue"
        ),
        fatigue_treatment="rank_step_cumulative_prompt",
        analysis_bundle="rank_step_fatigue_metrics",
        analysis_role="rank_step_fatigue_treatment",
        purpose=(
            "L4 no-reflection factual-memory agent with a cumulative "
            "displayed-rank fatigue value added only to the continuation "
            "prompt."
        ),
    ),
)
RANK_STEP_FATIGUE_CONDITIONS = (
    _l4_no_reflection_control,
    *RANK_STEP_FATIGUE_TREATMENT_CONDITIONS,
)

_non_agentic_cells: list[ExperimentCondition] = []
for model in NON_AGENTIC_MODELS:
    for stage in FROZEN_USER_STAGES:
        _non_agentic_cells.append(build_condition(model, stage))
NON_AGENTIC_CELLS = tuple(_non_agentic_cells)

_learned_frozen_cells: list[ExperimentCondition] = []
for model in LEARNED_FORCED_CLICK_MODELS:
    for stage in FROZEN_USER_STAGES:
        _learned_frozen_cells.append(build_condition(model, stage))
LEARNED_FROZEN_CELLS = tuple(_learned_frozen_cells)


PRIMARY_STAGE_BY_RESPONSE_MODEL = (
    AGENTIC_RECOMMENDER_FROZEN_CELLS
    + AGENTIC_RECOMMENDER_ADAPTIVE_CELLS
    + NON_AGENTIC_CELLS
    + LEARNED_FROZEN_CELLS
)


# Population replacement is orthogonal to user-state adaptation.  The primary
# L4 cells use survivor-similar refill.  Exit-only and uniform refill are
# controls, and the population policy (not the response model) activates the
# replacement users.
_l4_population_policy_controls: list[ExperimentCondition] = []
for user_update_policy, treatment_suffix in (
        ("full_reflection", "__full_reflection"),
        ("no_reflection_control", "__no_reflection_control"),
        ("memory_only_reflection", "__fixed_persona_memory_only"),
    ):
    for population_policy, policy_suffix, purpose_suffix in (
        (
            "exit_without_refill",
            "__exit_only_ablation",
            "Exit-only ablation; active places remain empty.",
        ),
        (
            "uniform_reserve_refill",
            "__uniform_refill_control",
            "Uniform-refill control for survivor-similar reserve selection.",
        ),
    ):
        _l4_population_policy_controls.append(
            build_condition(
                AGENTIC_RECOMMENDER,
                L4,
                user_update_policy=user_update_policy,
                population_policy=population_policy,
                condition_suffix=f"{treatment_suffix}{policy_suffix}",
                analysis_role="population_policy_control",
                purpose_suffix=purpose_suffix,
            )
        )
L4_POPULATION_POLICY_CONTROLS = tuple(_l4_population_policy_controls)


# Optional cumulative L5 outcome-feedback mechanism test.  O0 and O1 retain
# the same L4 agentic/reflection/population mechanisms; O1 adds a static ad
# but ignores outcomes.  Only O2 is L5 because it changes future delivery from
# observed outcomes. O0, O1, and O2 use the same seed-specific item schedule;
# only O1/O2 insert the item assigned to the current iteration.
# Changes first affect the next iteration and never remove an item from organic
# eligibility.
OUTCOME_FEEDBACK_ADVERTISEMENT_CONDITIONS = (
    replace(
        build_condition(
            AGENTIC_RECOMMENDER,
            L4,
            user_update_policy="full_reflection",
        ),
        condition_id="O0_organic",
        campaign_policy="no_campaign",
        analysis_role="outcome_feedback_reference",
        analysis_bundle="outcome_feedback_metrics",
        purpose="Organic top-10 reference with no sponsored insertion.",
    ),
    replace(
        build_condition(
            AGENTIC_RECOMMENDER,
            L4,
            user_update_policy="full_reflection",
            presentation_policy="sponsored_rank_1_plus_recommender_top_9",
            campaign_policy="static_sponsor_all_iterations",
        ),
        condition_id="O1_static_ad",
        analysis_role="static_ad_control",
        analysis_bundle="outcome_feedback_metrics",
        purpose=(
            "Static sponsored rank-1 insertion independent of observed outcomes; "
            "controls insertion, position, and displacement."
        ),
    ),
    replace(
        build_condition(
            AGENTIC_RECOMMENDER,
            L5,
            user_update_policy="full_reflection",
            presentation_policy="sponsored_rank_1_plus_recommender_top_9",
            campaign_policy="per_user_stop_after_3_consecutive_sponsored_non_clicks",
        ),
        condition_id="O2_adaptive_ad_three_strike",
        analysis_role="adaptive_outcome_feedback",
        analysis_bundle="outcome_feedback_metrics",
        purpose=(
            "Per-user sponsored insertion stops from t+1 after three consecutive "
            "valid sponsored non-click outcomes; sponsored clicks reset the count."
        ),
    ),
)


# Replica/shuffled calls observe a clone of the canonical pre-response state.
# Only the canonical decision can enter any history or fit buffer.
ORDER_AND_RELIABILITY_PROBES = tuple(
    replace(
        condition,
        condition_id=f"{condition.condition_id}__order_probe",
        presentation_policy="canonical_replica_and_shuffled_probe",
        analysis_role="non_persistent_order_probe",
        purpose=(
            f"{condition.purpose} Adds identical-state replica and shuffled-order "
            "measurements; probe responses never mutate state."
        ),
    )
    for condition in PRIMARY_STAGE_BY_RESPONSE_MODEL
)


FEEDBACK_LOOP_EXPERIMENT_CONDITIONS = (
    PRIMARY_STAGE_BY_RESPONSE_MODEL
    + L4_POPULATION_POLICY_CONTROLS
    + ORDER_AND_RELIABILITY_PROBES
    + OUTCOME_FEEDBACK_ADVERTISEMENT_CONDITIONS
    + TIPI_PERSONALITY_CONDITIONS
    + MEMORY_FORGETTING_CONDITIONS
    + RANK_STEP_FATIGUE_TREATMENT_CONDITIONS
)

CONDITIONS_BY_ID = {
    condition.condition_id: condition for condition in FEEDBACK_LOOP_EXPERIMENT_CONDITIONS
}

if len(CONDITIONS_BY_ID) != len(FEEDBACK_LOOP_EXPERIMENT_CONDITIONS):
    raise RuntimeError("Feedback loop experiment condition IDs must be unique")

CAMPAIGNS: dict[str, tuple[str, ...]] = {
    "primary_stage_by_response_model": tuple(
        condition.condition_id for condition in PRIMARY_STAGE_BY_RESPONSE_MODEL
    ),
    "l4_population_policy_controls": tuple(
        condition.condition_id for condition in L4_POPULATION_POLICY_CONTROLS
    ),
    "order_and_reliability_probes": tuple(
        condition.condition_id for condition in ORDER_AND_RELIABILITY_PROBES
    ),
    "outcome_feedback_advertisement": tuple(
        condition.condition_id
        for condition in OUTCOME_FEEDBACK_ADVERTISEMENT_CONDITIONS
    ),
    "tipi_personality": tuple(
        condition.condition_id for condition in TIPI_PERSONALITY_CONDITIONS
    ),
    "memory_forgetting": tuple(
        condition.condition_id for condition in MEMORY_FORGETTING_CONDITIONS
    ),
    "rank_step_fatigue": tuple(
        condition.condition_id for condition in RANK_STEP_FATIGUE_CONDITIONS
    ),
}

# This local runner exposes only the mitigation matrix. The copied builders
# above document adapter capabilities; older experiment entry points are not
# part of this campaign and remain in their original, unchanged package.
MITIGATION_RESPONSE_MODELS = (
    POSITION_BIASED, NCM_GRU_UNIFORM, NCM_GRU_MANSOURY,
    DIRECT_LLM, AGENTIC_RECOMMENDER,
)
MITIGATION_INTERVENTIONS = ("control", "post_processing", "in_processing")
MITIGATION_CONDITIONS = tuple(
    replace(
        build_condition(
            model,
            L3 if model is AGENTIC_RECOMMENDER else L2,
            user_update_policy=("full_reflection" if model is AGENTIC_RECOMMENDER else "frozen"),
            presentation_policy="mitigation_order_probes",
            analysis_role="mitigation_comparison",
        ),
        condition_id=f"{model.name}__{intervention}",
        intervention=intervention,
        analysis_bundle="mitigation_v1",
        purpose=f"{intervention} with {model.role} Canonical-only fixed-population feedback.",
    )
    for model in MITIGATION_RESPONSE_MODELS
    for intervention in MITIGATION_INTERVENTIONS
)
CONDITIONS_BY_ID = {condition.condition_id: condition for condition in MITIGATION_CONDITIONS}
CAMPAIGNS = {
    "full-no-llm": tuple(c.condition_id for c in MITIGATION_CONDITIONS
                         if c.user_model not in {"direct_llm", "agentic_recommender"}),
    "primary": tuple(c.condition_id for c in MITIGATION_CONDITIONS),
}


# Public release: expose only response-model and feedback conditions reported
# in the thesis. Optional psychological and pointwise pilot experiments are
# not part of this release's runnable condition catalogue.
_THESIS_RESPONSE_MODELS = {
    "position_biased", "ncm_inspired_gru_uniform", "ncm_inspired_gru_mansoury",
    "direct_llm", "agentic_recommender",
}
def _in_thesis_scope(condition):
    return (
        condition.user_model in _THESIS_RESPONSE_MODELS
        and condition.recommender == "sasrec"
        and (condition.feedback_loop not in {"L4_sampling", "L5_outcome"}
             or condition.user_update_policy == "full_reflection")
        and condition.tipi_treatment is None
        and condition.memory_treatment is None
        and condition.fatigue_treatment is None
    )
FEEDBACK_LOOP_EXPERIMENT_CONDITIONS = tuple(
    condition for condition in FEEDBACK_LOOP_EXPERIMENT_CONDITIONS
    if _in_thesis_scope(condition)
)
CONDITIONS_BY_ID = {
    key: condition for key, condition in CONDITIONS_BY_ID.items()
    if _in_thesis_scope(condition)
}
CAMPAIGNS = {
    name: tuple(key for key in keys if key in CONDITIONS_BY_ID)
    for name, keys in CAMPAIGNS.items()
    if any(key in CONDITIONS_BY_ID for key in keys)
}
