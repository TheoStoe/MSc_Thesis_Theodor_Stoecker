"""Concrete component construction for :mod:`run_simulator`.

The iteration lifecycle stays in one file.  This module contains only the
string-to-constructor wiring needed by the explicit methodology table.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

from .config import SimulatorConfig
from .contracts import ItemId, UserId
from .experiment_methodology import ExperimentCondition, TIPI_TREATMENTS
from .policies import (
    CampaignPolicy,
    CandidatePolicy,
    OrderProbePolicy,
    PopulationPolicy,
    SlatePolicy,
    build_canonical_replica_and_shuffled_probe_policy,
    build_exclude_seen_candidate_policy,
    build_exit_without_refill_population_policy,
    build_fixed_active_population_policy,
    build_no_ad_campaign_policy,
    build_per_user_three_strike_campaign_policy,
    build_sponsored_rank_one_slate_policy,
    build_static_sponsor_campaign_policy,
    build_survivor_similar_reserve_refill_population_policy,
    build_unchanged_recommender_order_policy,
    build_uniform_reserve_refill_population_policy,
)
from .policies.sponsor_schedule import SponsorSchedule, load_sponsor_schedule
from .psychology.tipi import TIPI_CACHE_FILENAME, TipiScores, load_or_create_tipi_scores
from .recommenders.base import Recommender
from .registry import RECOMMENDERS, USER_MODELS
from .user_models.base import UserResponseModel
from .utility.providers import ProviderConfig, build_provider
from .utility.randomness import RandomKey, random_generator
from .utility.user_handling import (
    PopulationRole,
    PreparedInputs,
    load_profile_user_texts,
)


@dataclass
class SimulationRuntime:
    """Isolated mutable state for exactly one condition and simulation seed."""

    recommender: Recommender
    user_model: UserResponseModel
    candidate_policy: CandidatePolicy
    slate_policy: SlatePolicy
    population_policy: PopulationPolicy
    campaign_policy: CampaignPolicy
    order_probe_policy: OrderProbePolicy | None
    observed_histories: dict[UserId, list[ItemId]]
    recommender_training_histories: dict[UserId, list[ItemId]]
    training_user_ids: tuple[UserId, ...]
    initial_recommender_restored: bool = False
    restored_initial_recommender_fingerprint: str | None = None


def build_runtime(
    config: SimulatorConfig,
    condition: ExperimentCondition,
    seed: int,
    inputs: PreparedInputs,
    *,
    resumed_user_state: Mapping[str, Any] | None = None,
) -> SimulationRuntime:
    """Construct one isolated runtime from explicit condition names."""

    from ..runtime_adapter import validate_output_root
    validate_output_root(config.output_root, read_only_inputs=(config.prepared_data_path,
                         config.population_artifact_path, *((config.initial_recommender_path,)
                         if config.initial_recommender_path is not None else ())))
    if config.initial_recommender_path is None or inputs.recommender_checkpoint_path is None:
        raise ValueError("the earlier shared initial SASRec checkpoint is mandatory; initial fitting is disabled")
    if condition.feedback_loop not in {"L0_control", "L1_feature", "L2_ml_model"}:
        raise ValueError("this mitigation runtime is restricted to L0, L1, and L2")

    members = tuple(inputs.population.members)
    active = tuple(
        member.user_id for member in members if member.role is PopulationRole.ACTIVE
    )
    reserve = tuple(
        member.user_id for member in members if member.role is PopulationRole.RESERVE
    )
    if len(active) != config.active_user_count:
        raise ValueError("population active-user count disagrees with configuration")
    if len(reserve) != config.reserve_user_count:
        raise ValueError("population reserve-user count disagrees with configuration")

    all_histories = {
        user_id: list(history)
        for user_id, history in inputs.dataset.training_histories.items()
    }
    response_users = (*active, *reserve)
    missing_histories = sorted(set(response_users).difference(all_histories))
    if missing_histories:
        raise ValueError(f"population users lack histories: {missing_histories}")
    response_histories = {
        user_id: tuple(all_histories[user_id]) for user_id in response_users
    }
    training_user_ids = _training_users(
        all_user_ids=tuple(sorted(all_histories)),
        required_user_ids=active,
        preferred_user_ids=reserve,
        requested_count=(
            len(response_users)
            if config.training_scope == "active_reserve"
            else config.pilot_training_user_count
        ),
        seed=seed,
    )
    training_histories = {
        user_id: list(all_histories[user_id]) for user_id in training_user_ids
    }

    catalog_ids = tuple(sorted(inputs.dataset.catalog))
    from .gender_runtime import resolved_gender_spec
    mitigation, women_item_ids, label_fingerprint = resolved_gender_spec(config.policy_settings, catalog_ids)
    recommender_settings = dict(config.recommender_settings)
    for controlled_name in ("catalog_item_ids", "random_seed"):
        if controlled_name in recommender_settings:
            raise ValueError(
                f"{controlled_name} is controlled by the resolved job, not settings"
            )
    recommender_settings["catalog_item_ids"] = catalog_ids
    recommender_settings["random_seed"] = seed
    recommender = RECOMMENDERS[condition.recommender].factory(
        **recommender_settings
    )
    configured_checkpoint = config.initial_recommender_path
    loaded_checkpoint = inputs.recommender_checkpoint_path
    if (configured_checkpoint is None) != (loaded_checkpoint is None):
        raise ValueError("configured and loaded initial recommender paths disagree")
    if (
        configured_checkpoint is not None
        and loaded_checkpoint is not None
        and configured_checkpoint.resolve() != loaded_checkpoint.resolve()
    ):
        raise ValueError("configured and loaded initial recommender paths differ")
    initial_recommender_restored = False
    restored_initial_recommender_fingerprint = None
    if loaded_checkpoint is not None:
        _restore_initial_recommender(
            recommender=recommender,
            checkpoint_path=loaded_checkpoint,
            dataset_fingerprint=inputs.dataset.dataset_fingerprint,
            training_histories=training_histories,
        )
        initial_recommender_restored = True
        restored_initial_recommender_fingerprint = recommender.fingerprint()
        if mitigation.intervention == "in_processing":
            recommender.configure_gender_weights(
                women_item_ids, women_weight=mitigation.women_positive_weight,
                label_fingerprint=label_fingerprint,
                initial_history_lengths={u: len(h) for u, h in training_histories.items()},
            )

    item_text_by_id = _item_text_by_id(inputs.dataset.catalog)
    user_model = _build_user_model(
        config=config,
        condition=condition,
        seed=seed,
        inputs=inputs,
        item_text_by_id=item_text_by_id,
        resumed_user_state=resumed_user_state,
    )
    sponsor_schedule = _load_sponsor_schedule(
        config=config,
        condition=condition,
        inputs=inputs,
        response_histories=response_histories,
        catalog_item_ids=catalog_ids,
    )
    candidate_policy = build_exclude_seen_candidate_policy(
        catalog_item_ids=catalog_ids,
        initial_histories=response_histories,
        sponsored_item_ids=(
            () if sponsor_schedule is None else sponsor_schedule.item_ids
        ),
    )
    slate_policy, order_probe_policy = _build_presentation_policies(
        condition=condition,
        slate_size=config.slate_size,
        seed=seed,
        settings=config.policy_settings,
    )
    if mitigation.intervention == "post_processing":
        from .gender_runtime import GenderPromotionSlatePolicy, GenderPromotionProbePolicy
        slate_policy = GenderPromotionSlatePolicy(
            config.slate_size, women_item_ids,
            promotion=mitigation.women_rank_promotion, label_fingerprint=label_fingerprint,
        )
        if order_probe_policy is not None:
            order_probe_policy = GenderPromotionProbePolicy(slate_policy, seed)
    population_policy = _build_population_policy(
        condition=condition,
        inputs=inputs,
        active=active,
        reserve=reserve,
        seed=seed,
        settings=config.policy_settings,
    )
    campaign_policy = _build_campaign_policy(
        condition=condition,
        settings=config.policy_settings,
        sponsor_schedule=sponsor_schedule,
    )
    return SimulationRuntime(
        recommender=recommender,
        user_model=user_model,
        candidate_policy=candidate_policy,
        slate_policy=slate_policy,
        population_policy=population_policy,
        campaign_policy=campaign_policy,
        order_probe_policy=order_probe_policy,
        observed_histories=all_histories,
        recommender_training_histories=training_histories,
        training_user_ids=training_user_ids,
        initial_recommender_restored=initial_recommender_restored,
        restored_initial_recommender_fingerprint=restored_initial_recommender_fingerprint,
    )


def _restore_initial_recommender(
    *,
    recommender: Recommender,
    checkpoint_path: Path,
    dataset_fingerprint: str,
    training_histories: Mapping[UserId, list[ItemId]],
) -> None:
    """Restore only the recommender from one trusted simulation checkpoint."""

    from .utility.checkpoints import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.input_fingerprints.get("dataset") != dataset_fingerprint:
        raise ValueError("initial recommender and prepared dataset differ")
    raw_state = checkpoint.component_states.get("recommender")
    if not isinstance(raw_state, Mapping):
        raise ValueError("initial checkpoint has no recommender state")
    if raw_state.get("model_state") is None:
        raise ValueError("initial checkpoint recommender is not fitted")
    if raw_state.get("gender_weight_config") is not None or raw_state.get("mitigation_weight_config") is not None:
        raise ValueError("the shared initial checkpoint must be the earlier unweighted model")

    configured_state = recommender.export_state()
    configured_variant = configured_state.get("variant_id")
    if raw_state.get("variant_id") != configured_variant:
        raise ValueError("initial recommender variant differs from configured model")

    configured_settings = dict(configured_state.get("settings", {}))
    restored_settings = dict(raw_state.get("settings", {}))
    if "device" in configured_settings:
        restored_settings["device"] = configured_settings["device"]
    if restored_settings != configured_settings:
        raise ValueError("initial recommender settings differ from resolved settings")

    expected_catalog = tuple(configured_settings.get("catalog_item_ids", ()))
    if tuple(raw_state.get("catalog_item_ids", ())) != expected_catalog:
        raise ValueError("initial recommender catalog differs from prepared catalog")
    expected_histories = {
        user_id: tuple(history)
        for user_id, history in sorted(training_histories.items())
    }
    restored_histories = {
        int(user_id): tuple(history)
        for user_id, history in dict(raw_state.get("histories", {})).items()
    }
    if restored_histories != expected_histories:
        raise ValueError("initial recommender training histories differ")

    state = dict(raw_state)
    state["settings"] = restored_settings
    recommender.restore_state(state)


def _build_user_model(
    *,
    config: SimulatorConfig,
    condition: ExperimentCondition,
    seed: int,
    inputs: PreparedInputs,
    item_text_by_id: Mapping[ItemId, str],
    resumed_user_state: Mapping[str, Any] | None = None,
) -> UserResponseModel:
    settings = dict(config.user_model_settings)
    text_overlay = settings.pop("llm_item_text_artifact_path", None)
    expected_text_fingerprint = settings.pop("llm_item_text_fingerprint", None)
    if (condition.user_model == "agentic_recommender" and text_overlay is None
            and config.profile_artifact_path is not None):
        manifest_path = config.profile_artifact_path / "manifest.json"
        if manifest_path.is_file():
            metadata = json.loads(manifest_path.read_text(encoding="utf-8")).get("metadata", {})
            if isinstance(metadata, dict) and "llm_item_text" in metadata:
                raise ValueError("summary-based agent profiles require their matching item-summary overlay")
    if text_overlay is not None or expected_text_fingerprint is not None:
        if condition.user_model not in {"direct_llm", "agentic_recommender"}:
            raise ValueError("item-summary overlays apply only to LLM response models")
        if (text_overlay is None or not isinstance(expected_text_fingerprint, str)
                or not expected_text_fingerprint.strip()):
            raise ValueError("LLM item summaries require both an artifact path and expected text fingerprint")
        from new_dataset_experiment_mitigation.item_summaries import load_item_summaries
        overlay = load_item_summaries(text_overlay, prepared_path=config.prepared_data_path)
        if overlay["manifest"]["source_dataset_fingerprint"] != inputs.dataset.dataset_fingerprint:
            raise ValueError("LLM item summaries belong to a different prepared dataset")
        if overlay["manifest"]["item_text_fingerprint"] != expected_text_fingerprint:
            raise ValueError("LLM item-summary fingerprint differs from the configured policy")
        if set(overlay["item_text_by_id"]) != set(inputs.dataset.catalog):
            raise ValueError("LLM item summaries must cover the exact prepared catalogue")
        item_text_by_id = overlay["item_text_by_id"]
        if condition.user_model == "agentic_recommender":
            if config.profile_artifact_path is None or not inputs.profiles_by_user:
                raise ValueError("enriched agent conditions require completed summary-based profiles")
            from new_dataset_experiment_mitigation.prepare_profiles import validate_profile_item_summaries
            validate_profile_item_summaries(
                config.prepared_data_path.parent, config.profile_artifact_path, text_overlay,
                expected_item_text_fingerprint=expected_text_fingerprint,
            )
    if condition.user_model in {
        "position_biased",
        "pointwise_logistic",
        "ncm_inspired_gru_uniform",
        "ncm_inspired_gru_mansoury",
    }:
        if "seed" in settings:
            raise ValueError("user-model seed is controlled by the resolved job")
        settings["seed"] = seed
    if condition.user_model in {
        "pointwise_logistic",
        "ncm_inspired_gru_uniform",
        "ncm_inspired_gru_mansoury",
    }:
        if "model_name" in settings:
            raise ValueError("NCM model_name is controlled by the condition")
        settings["model_name"] = condition.user_model
        if "checkpoint_directory" in settings or "embedding_directory" in settings:
            configured_fingerprint = settings.get("expected_prepared_fingerprint")
            if configured_fingerprint is None:
                # Self-contained NCM artifacts use the simulator's prepared-data
                # fingerprint. Legacy deployed NCMs instead havea a calibration-artifact fingerprint supplied by the run.
                settings["expected_prepared_fingerprint"] = (
                    inputs.dataset.dataset_fingerprint
                )
    if condition.user_model in {"direct_llm", "agentic_recommender"}:
        provider_values = dict(config.provider_settings)
        if "provider" in provider_values:
            raise ValueError("provider is controlled by --provider, not settings")
        provider = build_provider(
            ProviderConfig(provider=config.provider_id, **provider_values)
        )
        for controlled_name in ("provider", "item_text_by_id"):
            if controlled_name in settings:
                raise ValueError(
                    f"{controlled_name} is controlled by prepared inputs/config"
                )
        settings["provider"] = provider
        settings["item_text_by_id"] = item_text_by_id
    if condition.user_model == "agentic_recommender":
        if not inputs.profiles_by_user:
            raise ValueError("agentic conditions require completed profile artifacts")
        for controlled_name in (
            "profiles_by_user",
            "update_mode",
            "tipi_scores_by_user",
            "tipi_treatment",
            "memory_treatment",
            "memory_decay",
            "memory_retrieval_limit",
            "initial_histories_by_user",
            "fatigue_treatment",
        ):
            if controlled_name in settings:
                raise ValueError(
                    f"{controlled_name} is controlled by the resolved condition"
                )
        settings["profiles_by_user"] = inputs.profiles_by_user
        settings["update_mode"] = condition.user_update_policy
        if condition.memory_treatment is not None:
            if (
                condition.memory_decay is None
                or condition.memory_retrieval_limit is None
            ):
                raise ValueError(
                    "memory treatments require decay and retrieval limit"
                )
            settings["memory_treatment"] = condition.memory_treatment
            settings["memory_decay"] = condition.memory_decay
            settings["memory_retrieval_limit"] = (
                condition.memory_retrieval_limit
            )
            initial_histories = inputs.dataset.training_histories
            missing_initial_histories = sorted(
                set(inputs.profiles_by_user).difference(initial_histories)
            )
            if missing_initial_histories:
                raise ValueError(
                    "profile users lack initial training histories: "
                    f"{missing_initial_histories}"
                )
            settings["initial_histories_by_user"] = {
                user_id: tuple(initial_histories[user_id])
                for user_id in sorted(inputs.profiles_by_user)
            }
        fatigue_treatment = condition.fatigue_treatment
        if fatigue_treatment is not None:
            settings["fatigue_treatment"] = fatigue_treatment
        treatment = condition.tipi_treatment
        if treatment not in {None, "hidden_control"}:
            if config.profile_artifact_path is None:
                raise ValueError("TIPI conditions require a profile bundle path")
            settings["tipi_treatment"] = treatment
            overrides = TIPI_TREATMENTS[treatment]
            active_profiles = {
                member.user_id: inputs.profiles_by_user[member.user_id]
                for member in inputs.population.members
                if member.role is PopulationRole.ACTIVE
            }
            if resumed_user_state is not None:
                # The checkpoint already contains the assigned treatment scores.
                # Never rerun questionnaires or create a fresh cache on restart.
                tipi = resumed_user_state["tipi"]
                if tipi["treatment"] != treatment:
                    raise ValueError("checkpoint TIPI treatment differs")
                scores_by_user = tipi["scores_by_user"]
                if set(scores_by_user) != set(active_profiles):
                    raise ValueError("checkpoint TIPI population differs")
                settings["tipi_scores_by_user"] = {
                    user_id: TipiScores(
                        **scores, emotional_stability=8.0 - scores["neuroticism"]
                    )
                    for user_id, scores in scores_by_user.items()
                }
            else:
                user_characteristics = load_profile_user_texts(
                    config.profile_artifact_path,
                    set(active_profiles),
                )
                natural_scores = load_or_create_tipi_scores(
                    config.output_root / TIPI_CACHE_FILENAME,
                    profiles_by_user=active_profiles,
                    user_characteristics_by_user=user_characteristics,
                    provider=provider,
                )
                settings["tipi_scores_by_user"] = {
                    user_id: replace(scores, **overrides)
                    for user_id, scores in natural_scores.items()
                }
    model = USER_MODELS[condition.user_model].factory(**settings)
    if condition.user_model in {
        "pointwise_logistic",
        "ncm_inspired_gru_uniform",
        "ncm_inspired_gru_mansoury",
    }:
        embedded_items = set(getattr(model, "item_embeddings", {}))
        catalog_items = set(item_text_by_id)
        if embedded_items != catalog_items:
            raise ValueError(
                "NCM embedding catalog does not exactly match prepared items"
            )
    return model


def _build_presentation_policies(
    *,
    condition: ExperimentCondition,
    slate_size: int,
    seed: int,
    settings: Mapping[str, Any],
) -> tuple[SlatePolicy, OrderProbePolicy | None]:
    if condition.presentation_policy == "unchanged_recommender_order":
        return build_unchanged_recommender_order_policy(slate_size=slate_size), None
    if condition.presentation_policy == "sponsored_rank_1_plus_recommender_top_9":
        policy = build_sponsored_rank_one_slate_policy(
            slate_size=slate_size,
            deduplication_rule_id=str(
                settings.get(
                    "sponsor_deduplication_rule_id",
                    "remove_sponsor_from_organic_top_k_v1",
                )
            ),
        )
        return policy, None
    if condition.presentation_policy == "canonical_replica_and_shuffled_probe":
        return (
            build_unchanged_recommender_order_policy(slate_size=slate_size),
            build_canonical_replica_and_shuffled_probe_policy(
                slate_size=slate_size,
                seed=seed,
            ),
        )
    raise KeyError(f"unknown presentation policy: {condition.presentation_policy}")


def _build_population_policy(
    *,
    condition: ExperimentCondition,
    inputs: PreparedInputs,
    active: tuple[UserId, ...],
    reserve: tuple[UserId, ...],
    seed: int,
    settings: Mapping[str, Any],
) -> PopulationPolicy:
    name = condition.population_policy
    if name == "fixed_active_users":
        return build_fixed_active_population_policy(initial_active_user_ids=active)
    if name == "exit_without_refill":
        return build_exit_without_refill_population_policy(
            initial_active_user_ids=active
        )
    if name == "uniform_reserve_refill":
        return build_uniform_reserve_refill_population_policy(
            initial_active_user_ids=active,
            reserve_user_ids=reserve,
            seed=seed,
        )
    if name == "survivor_similar_reserve_refill":
        response_users = set(active).union(reserve)
        clusters = {
            member.user_id: member.cluster_id
            for member in inputs.population.members
            if member.user_id in response_users
        }
        return build_survivor_similar_reserve_refill_population_policy(
            initial_active_user_ids=active,
            reserve_user_ids=reserve,
            cluster_by_user=clusters,
            similarity_artifact_fingerprint=(
                inputs.population.cluster_artifact_fingerprint
            ),
            selection_rule_id=str(
                settings.get(
                    "survivor_selection_rule_id",
                    "survivor_cluster_weighted_uniform_leave_unfilled_v1",
                )
            ),
            seed=seed,
        )
    raise KeyError(f"unknown population policy: {name}")


def _build_campaign_policy(
    *,
    condition: ExperimentCondition,
    settings: Mapping[str, Any],
    sponsor_schedule: SponsorSchedule | None,
) -> CampaignPolicy:
    name = condition.campaign_policy
    if name == "no_campaign":
        return build_no_ad_campaign_policy()
    if sponsor_schedule is None:
        raise ValueError(
            "advertisement conditions require sponsor_schedule_path"
        )
    if name == "static_sponsor_all_iterations":
        return build_static_sponsor_campaign_policy(
            sponsored_item_ids=sponsor_schedule.item_ids,
            sponsor_artifact_fingerprint=(
                sponsor_schedule.artifact_fingerprint
            ),
        )
    if name == "per_user_stop_after_3_consecutive_sponsored_non_clicks":
        return build_per_user_three_strike_campaign_policy(
            sponsored_item_ids=sponsor_schedule.item_ids,
            sponsor_artifact_fingerprint=(
                sponsor_schedule.artifact_fingerprint
            ),
            threshold=int(settings.get("sponsored_non_click_threshold", 3)),
        )
    raise KeyError(f"unknown campaign policy: {name}")


def _load_sponsor_schedule(
    *,
    config: SimulatorConfig,
    condition: ExperimentCondition,
    inputs: PreparedInputs,
    response_histories: Mapping[UserId, tuple[ItemId, ...]],
    catalog_item_ids: tuple[ItemId, ...],
) -> SponsorSchedule | None:
    path = config.policy_settings.get("sponsor_schedule_path")
    if path is None:
        if condition.analysis_bundle == "outcome_feedback_metrics":
            raise ValueError(
                "O0/O1/O2 conditions require sponsor_schedule_path"
            )
        return None
    if "sponsored_item_id" in config.policy_settings:
        raise ValueError(
            "use sponsor_schedule_path instead of a single sponsored_item_id"
        )
    schedule = load_sponsor_schedule(
        str(path),
        population_seed=config.population_seed,
        expected_dataset_fingerprint=inputs.dataset.dataset_fingerprint,
        catalog_item_ids=catalog_item_ids,
    )
    if len(schedule.item_ids) < config.num_iterations:
        raise ValueError(
            "sponsor schedule is shorter than the requested simulation"
        )
    scheduled_items = frozenset(schedule.item_ids)
    users_with_seen_sponsors = tuple(
        user_id
        for user_id, history in sorted(response_histories.items())
        if scheduled_items.intersection(history)
    )
    if users_with_seen_sponsors:
        raise ValueError(
            "sponsor schedule is not initially unseen for response users: "
            f"{users_with_seen_sponsors}"
        )
    return schedule


def _item_text_by_id(
    catalog: Mapping[ItemId, Mapping[str, Any]],
) -> dict[ItemId, str]:
    texts: dict[ItemId, str] = {}
    for item_id in sorted(catalog):
        metadata = catalog[item_id]
        explicit = metadata.get("item_text", metadata.get("text"))
        if isinstance(explicit, str) and explicit.strip():
            texts[item_id] = explicit.strip()
            continue
        title = metadata.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"catalog item {item_id} has no usable title/text")
        genres = metadata.get("genres", ())
        genre_text = ", ".join(str(value) for value in genres) if genres else "unknown"
        texts[item_id] = f"{title.strip()} | Genres: {genre_text}"
    return texts


def _training_users(
    *,
    all_user_ids: tuple[UserId, ...],
    required_user_ids: tuple[UserId, ...],
    preferred_user_ids: tuple[UserId, ...] = (),
    requested_count: int | None,
    seed: int,
) -> tuple[UserId, ...]:
    if requested_count is None or requested_count >= len(all_user_ids):
        return all_user_ids
    all_users = set(all_user_ids)
    required_set = set(required_user_ids)
    if not required_set.issubset(all_users):
        raise ValueError("required pilot users are absent from prepared histories")
    required = tuple(sorted(required_set))
    if requested_count < len(required):
        raise ValueError(
            "pilot_training_user_count cannot be smaller than the active cohort"
        )
    needed = requested_count - len(required)
    preferred = tuple(
        dict.fromkeys(
            user_id
            for user_id in preferred_user_ids
            if user_id not in required_set
        )
    )
    if not set(preferred).issubset(all_users):
        raise ValueError("preferred pilot users are absent from prepared histories")
    preferred_selected = preferred[:needed]
    needed -= len(preferred_selected)
    remaining = tuple(
        sorted(
            all_users
            .difference(required_set)
            .difference(preferred_selected)
        )
    )
    generator = random_generator(seed, RandomKey(stream_id=8001))
    random_selected = (
        tuple(int(value) for value in generator.choice(remaining, needed, replace=False))
        if needed
        else ()
    )
    return tuple(sorted((*required, *preferred_selected, *random_selected)))


__all__ = ["SimulationRuntime", "build_runtime"]
