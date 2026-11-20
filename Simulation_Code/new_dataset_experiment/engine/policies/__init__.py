"""Policy package for eligibility, presentation, participation, and campaigns.

The package facade preserves the original ``new_dataset_experiment.engine.policies``
import path while keeping each policy domain in a focused module.
"""

from .campaign import (
    build_no_ad_campaign_policy,
    build_per_user_three_strike_campaign_policy,
    build_static_sponsor_campaign_policy,
)
from .candidate import build_exclude_seen_candidate_policy
from .contracts import (
    CampaignPolicy,
    CandidatePolicy,
    OrderProbePolicy,
    OrderProbeSlateResult,
    PopulationPolicy,
    PopulationTransitionResult,
    SlatePolicy,
    SlateResult,
    SponsoredOutcome,
)
from .order_probe import build_canonical_replica_and_shuffled_probe_policy
from .population import (
    build_exit_without_refill_population_policy,
    build_fixed_active_population_policy,
    build_survivor_similar_reserve_refill_population_policy,
    build_uniform_reserve_refill_population_policy,
)
from .presentation import (
    build_sponsored_rank_one_slate_policy,
    build_unchanged_recommender_order_policy,
)

__all__ = [
    "CampaignPolicy",
    "CandidatePolicy",
    "OrderProbePolicy",
    "OrderProbeSlateResult",
    "PopulationPolicy",
    "PopulationTransitionResult",
    "SlatePolicy",
    "SlateResult",
    "SponsoredOutcome",
    "build_canonical_replica_and_shuffled_probe_policy",
    "build_exclude_seen_candidate_policy",
    "build_exit_without_refill_population_policy",
    "build_fixed_active_population_policy",
    "build_no_ad_campaign_policy",
    "build_per_user_three_strike_campaign_policy",
    "build_sponsored_rank_one_slate_policy",
    "build_static_sponsor_campaign_policy",
    "build_survivor_similar_reserve_refill_population_policy",
    "build_unchanged_recommender_order_policy",
    "build_uniform_reserve_refill_population_policy",
]
