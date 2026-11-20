"""Named cumulative feedback-loop stages using the thesis terminology."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackLoopPreset:
    """Which scientific state changes a stage permits after a valid response.

    A complete condition still selects the supported mechanism that actually
    runs. For example, L3 permits either a persistent user-state update or a
    shared response-model refit; it does not imply that every L3 model does
    both.
    """

    name: str
    exclude_clicked_items: bool
    update_user_history: bool
    retrain_recommender: bool
    allow_latent_user_state_update: bool
    allow_response_model_refit: bool
    allow_user_exit: bool
    update_campaign_from_outcomes: bool


L0_CONTROL = FeedbackLoopPreset(
    name="L0_control",
    exclude_clicked_items=True,
    update_user_history=False,
    retrain_recommender=False,
    allow_latent_user_state_update=False,
    allow_response_model_refit=False,
    allow_user_exit=False,
    update_campaign_from_outcomes=False,
)

L1_FEATURE = FeedbackLoopPreset(
    name="L1_feature",
    exclude_clicked_items=True,
    update_user_history=True,
    retrain_recommender=False,
    allow_latent_user_state_update=False,
    allow_response_model_refit=False,
    allow_user_exit=False,
    update_campaign_from_outcomes=False,
)

L2_ML_MODEL = FeedbackLoopPreset(
    name="L2_ml_model",
    exclude_clicked_items=True,
    update_user_history=True,
    retrain_recommender=True,
    allow_latent_user_state_update=False,
    allow_response_model_refit=False,
    allow_user_exit=False,
    update_campaign_from_outcomes=False,
)

L3_INDIVIDUAL = FeedbackLoopPreset(
    name="L3_individual",
    exclude_clicked_items=True,
    update_user_history=True,
    retrain_recommender=True,
    allow_latent_user_state_update=True,
    allow_response_model_refit=True,
    allow_user_exit=False,
    update_campaign_from_outcomes=False,
)

L4_SAMPLING = FeedbackLoopPreset(
    name="L4_sampling",
    exclude_clicked_items=True,
    update_user_history=True,
    retrain_recommender=True,
    allow_latent_user_state_update=True,
    # L4 permits an online-fit response model to retain its L3 mechanism, but
    # a condition with a frozen/unsupported model still performs no refit.
    allow_response_model_refit=True,
    allow_user_exit=True,
    update_campaign_from_outcomes=False,
)

# Cumulative outcome-feedback stage.  It retains L4 participation dynamics and
# additionally permits a campaign to change future presentation from observed
# outcomes.  A static advertisement arm remains an L4 control because it never
# consumes outcomes.
L5_OUTCOME = FeedbackLoopPreset(
    name="L5_outcome",
    exclude_clicked_items=True,
    update_user_history=True,
    retrain_recommender=True,
    allow_latent_user_state_update=True,
    # L5 retains this permission from L4. The complete condition and model
    # capability determine whether response fitting actually occurs.
    allow_response_model_refit=True,
    allow_user_exit=True,
    update_campaign_from_outcomes=True,
)


FEEDBACK_LOOP_PRESETS: dict[str, FeedbackLoopPreset] = {
    preset.name: preset
    for preset in (
        L0_CONTROL,
        L1_FEATURE,
        L2_ML_MODEL,
        L3_INDIVIDUAL,
        L4_SAMPLING,
        L5_OUTCOME,
    )
}

# Candidate depletion and model-visible history are deliberately independent.
# Every stage removes a valid canonical click from that user's later candidate
# sets. L0 otherwise freezes history and model state; L1 additionally exposes
# the click as a feature, and L2 additionally refits the recommender.
#
# L4 enables exit; the selected population policy determines whether the active
# place remains empty or is filled from the frozen 300-user reserve.
