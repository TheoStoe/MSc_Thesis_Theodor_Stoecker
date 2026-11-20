"""Visible component registries; no decorator or import-time discovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComponentRegistration:
    factory: Callable[..., Any]
    variant_id: str
    capabilities: frozenset[str]


def _position_biased_factory(**settings: Any) -> Any:
    from .user_models.position_biased import PositionBiasedResponseModel

    return PositionBiasedResponseModel(**settings)


def _ncm_factory(**settings: Any) -> Any:
    from .user_models.ncm import NCMResponseModel

    if "checkpoint_directory" in settings or "embedding_directory" in settings:
        return NCMResponseModel.from_artifacts(**settings)
    return NCMResponseModel(**settings)


def _agentic_recommender_factory(**settings: Any) -> Any:
    from .user_models.agentic_recommender import AgenticRecommenderResponseModel

    return AgenticRecommenderResponseModel(**settings)


def _direct_llm_factory(**settings: Any) -> Any:
    from .user_models.direct_llm import DirectLLMResponseModel

    return DirectLLMResponseModel(**settings)


def _sasrec_factory(**settings: Any) -> Any:
    from .recommenders.sasrec import SASRecAdapter

    return SASRecAdapter(**settings)


def _lusifer_profile_factory(**settings: Any) -> Any:
    from .persona_extraction.lusifer_adapter import LusiferProfileExtractor

    return LusiferProfileExtractor(**settings)


USER_MODELS: dict[str, ComponentRegistration] = {
    "position_biased": ComponentRegistration(
        factory=_position_biased_factory,
        variant_id="position_biased_mansoury_v1",
        capabilities=frozenset({"forced_single_click", "checkpoint"}),
    ),
    "pointwise_logistic": ComponentRegistration(
        factory=_ncm_factory,
        variant_id="ncm_response_v1",
        capabilities=frozenset(
            {"forced_single_click", "observed_history", "checkpoint"}
        ),
    ),
    **{
        model_name: ComponentRegistration(
            factory=_ncm_factory,
            variant_id="ncm_response_v1",
            capabilities=frozenset(
                {
                    "forced_single_click",
                    "observed_history",
                    "checkpoint",
                }
            ),
        )
        for model_name in ( #TODO: why remove pointwise logistic here?
            "ncm_inspired_gru_uniform",
            "ncm_inspired_gru_mansoury",
        )
    },
    "agentic_recommender": ComponentRegistration(
        factory=_agentic_recommender_factory,
        variant_id="agentic_recommender_lusifer_v2",
        capabilities=frozenset(
            {
                "click_or_no_click",
                "state_update",
                "continuation",
                "checkpoint",
            }
        ),
    ),
    "direct_llm": ComponentRegistration(
        factory=_direct_llm_factory,
        variant_id="direct_llm_single_call_v1",
        capabilities=frozenset(
            {
                "click_or_no_click",
                "observed_history",
                "checkpoint",
            }
        ),
    ),
}
RECOMMENDERS: dict[str, ComponentRegistration] = {
    "sasrec": ComponentRegistration(
        factory=_sasrec_factory,
        variant_id="sasrec_pmixer_pytorch_v1",
        capabilities=frozenset(
            {"fit", "explicit_inference_history", "candidate_scoring", "checkpoint"}
        ),
    ),
}
PROFILE_EXTRACTORS: dict[str, ComponentRegistration] = {
    "lusifer": ComponentRegistration(
        factory=_lusifer_profile_factory,
        variant_id="lusifer_upstream_prompts_v1",
        capabilities=frozenset(
            {
                "offline",
                "explicit_user_join",
                "training_only",
                "fingerprinted_artifact",
            }
        ),
    ),
}
CANDIDATE_POLICIES: dict[str, ComponentRegistration] = {}
SLATE_POLICIES: dict[str, ComponentRegistration] = {}
POPULATION_POLICIES: dict[str, ComponentRegistration] = {}
CAMPAIGN_POLICIES: dict[str, ComponentRegistration] = {}
ANALYSIS_BUNDLES: dict[str, ComponentRegistration] = {}

# Registrations are deliberately explicit. NCM entries share one
# adapter factory; their separately fingerprinted predictor/checkpoint selects
# logistic versus GRU and uniform versus Mansoury training provenance.
