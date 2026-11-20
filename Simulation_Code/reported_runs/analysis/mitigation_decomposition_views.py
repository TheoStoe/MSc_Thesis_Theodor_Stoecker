"""Descriptive post-processing components from saved matched decisions.

The original-order probe and recommender exposures describe the post-processing
run's current state before reranking. Their difference from its own control is
the remaining trajectory difference, including state, candidate, refit, and run
variability. Calling this a feedback component does not identify a pure causal
feedback effect. No simulator, provider, or run-file access occurs here.
For LLMs, the immediate click component also includes variability between
separate canonical and original-order response calls.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .comparisons import RUN_KEYS, RUN_METADATA, seed_consistency, summarise_iterations


DECOMPOSITION_METRICS = ("head_click_share", "head_exposure", "weighted_head_exposure")
ORIGINAL_ORDER_METRICS = {
    "head_click_share": "original_order_head_click_share",
    "head_exposure": "recommender_head_exposure",
    "weighted_head_exposure": "recommender_weighted_head_exposure",
}
COMPONENT_KEYS = ("total", "immediate", "feedback")
COMPONENT_LABELS = {"total": "Total", "immediate": "Immediate reranking", "feedback": "Feedback component"}
COMPONENT_ORDER = tuple(COMPONENT_LABELS[key] for key in COMPONENT_KEYS)
TABLE_LABELS = {"total": "Total (pp)", "immediate": "Immediate (pp)", "feedback": "Feedback (pp)"}
COUNT_COLUMNS = (
    "opportunities", "canonical_opportunities", "original_order_opportunities",
    "canonical_valid_clicks", "canonical_head_clicks", "original_order_valid_clicks",
    "original_order_head_clicks", "canonical_valid_actions", "canonical_valid_no_clicks", "canonical_no_clicks",
    "canonical_invalid_responses", "original_order_valid_actions", "original_order_valid_no_clicks", "original_order_no_clicks",
    "original_order_invalid_responses", "both_arms_valid_clicks", "control_opportunities",
    "control_valid_clicks", "control_valid_actions", "control_valid_no_clicks", "control_invalid_responses",
)
PAIR_COLUMNS = (
    "dataset", "study", "seed", "population_seed", "condition_id", "comparison", "kind", "rqs", "model", "stage",
    "intervention", "treatment_condition", "control_condition", "treatment_physical_run_id",
    "control_physical_run_id", "pair_status", "source_id", "run_id", "physical_run_id", "user_update_policy",
)


def _validate_frame(frame, required, name):
    missing = set(required)-set(frame)
    if missing:
        raise ValueError(f"Missing {name} columns: {sorted(missing)}")
    if frame[RUN_KEYS].isna().any().any():
        raise ValueError(f"{name} run identities must be defined")
    if frame.duplicated(RUN_KEYS+["iteration"]).any():
        raise ValueError(f"Duplicate {name} run/iteration rows")
    iterations = pd.to_numeric(frame.iteration, errors="raise")
    if not (iterations.between(1, 30) & iterations.eq(np.floor(iterations))).all():
        raise ValueError(f"{name} iterations must be integers from 1 to 30")


def _one_metadata(block, columns, name):
    for column in columns:
        if column in block and block[column].nunique(dropna=False) != 1:
            raise ValueError(f"{name} metadata changes within one run: {column}")
    return block.iloc[0]


def _validate_shares(block, columns):
    for column in columns:
        values = pd.to_numeric(block[column], errors="raise").to_numpy(dtype=float)
        defined = values[~np.isnan(values)]
        if not np.isfinite(defined).all() or (defined < -1e-12).any() or (defined > 1+1e-12).any():
            raise ValueError(f"{column} must contain raw shares in 0--1, not percentage points")


def build_postprocessing_decomposition(post_frame, control_trajectory, comparison_plan):
    """Pair exact own-controls and calculate all components per iteration.

    ``total = post canonical - control canonical``;
    ``immediate = post canonical - post original order``;
    ``feedback = post original order - control canonical``.
    Original-order click share uses its own valid-click denominator, while
    exposure uses the saved unreordered recommender slate. The raw loader must
    validate that reranking preserves slate membership. The extra unweighted
    exposure check here requires equal shares and sets its immediate component
    to exactly zero wherever both exposures are defined.

    Values are converted from raw shares to percentage points once. Primary
    means require 30 defined component values; the total remains available if
    only the original-order probe is undefined. Identity claims are restricted
    to common-defined iterations/windows, whose counts are exported explicitly.
    """
    post_required = RUN_METADATA+["iteration"]+list(DECOMPOSITION_METRICS)+list(ORIGINAL_ORDER_METRICS.values())
    control_required = RUN_KEYS+["iteration", "model", "stage", "intervention"]+list(DECOMPOSITION_METRICS)
    _validate_frame(post_frame, post_required, "post-processing")
    _validate_frame(control_trajectory, control_required, "control")
    plan_required = {"dataset", "study", "seed", "population_seed", "model", "stage", "intervention",
                     "kind", "comparison", "treatment_condition", "control_condition", "eligible"}
    if plan_required-set(comparison_plan):
        raise ValueError(f"Missing comparison-plan columns: {sorted(plan_required-set(comparison_plan))}")
    if not post_frame.intervention.eq("post_processing").all():
        raise ValueError("Post-processing decomposition accepts post_processing treatments only")
    for proof in ("slate_membership_preserved", "unweighted_exposure_identity"):
        if proof in post_frame and not post_frame[proof].eq(True).all():
            raise ValueError(f"The saved post-processing supplement failed {proof}")
    plans = comparison_plan.loc[comparison_plan.kind.eq("mitigation") & comparison_plan.intervention.eq("post_processing")]
    rows, count_blocks, eligibility = [], [], []
    for run_key, treatment in post_frame.groupby(RUN_KEYS, dropna=False, sort=False):
        identity = dict(zip(RUN_KEYS, run_key))
        post_meta = _one_metadata(treatment, RUN_METADATA+['source_id'], "Post-processing")
        selected = plans
        for column, value in identity.items():
            selected = selected.loc[selected["treatment_condition" if column == "condition_id" else column].eq(value)]
        if len(selected) != 1 or not bool(selected.iloc[0].eligible):
            raise ValueError(f"Exactly one eligible own-control plan is required for {identity}")
        pair = selected.iloc[0].to_dict()
        if pair["model"] != post_meta.model or pair["stage"] != post_meta.stage:
            raise ValueError("The treatment model/stage differs from its own-control comparison plan")
        control = control_trajectory
        for column, value in identity.items():
            control = control.loc[control[column].eq(pair["control_condition"] if column == "condition_id" else value)]
        if control.empty:
            raise ValueError(f"Exact own-control trajectory is unavailable for {identity}")
        control_meta = _one_metadata(control, RUN_METADATA, "Control")
        if not control.intervention.eq("control").all():
            raise ValueError("The matched own-control condition must have no mitigation")
        for column in ("model", "stage", "user_update_policy"):
            if column in control and control_meta[column] != post_meta[column]:
                raise ValueError(f"Treatment and control {column} differ; they cannot be paired")
        for column, meta in (("treatment_physical_run_id", post_meta), ("control_physical_run_id", control_meta)):
            if column in pair and pd.notna(pair[column]) and "physical_run_id" in meta and pair[column] != meta.physical_run_id:
                raise ValueError(f"The {column} differs from the explicit comparison plan")
        _validate_shares(treatment, list(DECOMPOSITION_METRICS)+list(ORIGINAL_ORDER_METRICS.values()))
        _validate_shares(control, DECOMPOSITION_METRICS)
        common = treatment.head_exposure.notna() & treatment.recommender_head_exposure.notna()
        if not np.allclose(treatment.loc[common, "head_exposure"], treatment.loc[common, "recommender_head_exposure"], rtol=0, atol=1e-12):
            raise ValueError("Unweighted post-processing exposure differs from its original recommender slate")

        metadata = {column: pair[column] for column in PAIR_COLUMNS if column in pair}
        metadata.update({column: post_meta[column] for column in ("condition_id", "source_id", "run_id", "physical_run_id", "user_update_policy") if column in post_meta})
        post_columns = ["iteration"]+list(DECOMPOSITION_METRICS)+list(ORIGINAL_ORDER_METRICS.values())+[c for c in COUNT_COLUMNS if c in treatment]
        post_columns = list(dict.fromkeys(post_columns))
        control_counts = {c: f"control_{c}" for c in ("opportunities", "valid_clicks", "valid_actions", "valid_no_clicks", "invalid_responses") if c in control}
        control_columns = ["iteration"]+list(DECOMPOSITION_METRICS)+list(control_counts)
        paired = treatment[post_columns].merge(control[control_columns].rename(columns={
            **{metric: f"control_{metric}" for metric in DECOMPOSITION_METRICS}, **control_counts}),
            on="iteration", how="outer", validate="one_to_one").sort_values("iteration")
        counts = [c for c in COUNT_COLUMNS if c in paired]
        if counts:
            count_blocks.append(paired[["iteration"]+counts].assign(**metadata))
        horizon = paired.iteration.tolist() == list(range(1, 31))
        for metric in DECOMPOSITION_METRICS:
            post = paired[metric].to_numpy(dtype=float)
            original = paired[ORIGINAL_ORDER_METRICS[metric]].to_numpy(dtype=float)
            baseline = paired[f"control_{metric}"].to_numpy(dtype=float)
            # Membership-preserving reranking has no immediate unweighted
            # exposure effect. Avoid treating floating-point summation noise as
            # a nonzero intervention component after the equality check above.
            immediate = np.where(np.isfinite(post) & np.isfinite(original), 0., np.nan) if metric == "head_exposure" else post-original
            components = {"total": post-baseline, "immediate": immediate, "feedback": original-baseline}
            all_defined = np.isfinite(post) & np.isfinite(original) & np.isfinite(baseline)
            n_common = int(all_defined.sum())
            endpoint_common = bool(all_defined[paired.iteration.eq(30).to_numpy()][0]) if paired.iteration.eq(30).any() else False
            for window in ("mean_1_30", "iteration_30"):
                eligibility.append({**metadata, "metric": metric, "window": window,
                    "n_post_defined": int(np.isfinite(post).sum()), "n_original_order_defined": int(np.isfinite(original).sum()),
                    "n_control_defined": int(np.isfinite(baseline).sum()), "n_common_defined": n_common,
                    "complete_horizon": horizon, "identity_defined": bool(horizon and n_common == 30) if window == "mean_1_30" else endpoint_common})
            for component, values in components.items():
                for index, saved in enumerate(paired.to_dict("records")):
                    rows.append({**metadata, "iteration": saved["iteration"], "metric": metric,
                        "component": component, "component_label": COMPONENT_LABELS[component],
                        "value": values[index]*100, "effect_unit": "percentage points",
                        "post_value": post[index], "original_order_value": original[index], "control_value": baseline[index],
                        "all_components_defined": bool(all_defined[index]),
                        **{column: saved[column] for column in counts}})
    long = pd.DataFrame(rows)
    eligibility = pd.DataFrame(eligibility)
    if long.empty:
        return {"per_iteration": long, "summaries": pd.DataFrame(), "eligibility": eligibility,
                "count_summaries": pd.DataFrame(), "seed_consistency": pd.DataFrame(),
                "tables": {metric: pd.DataFrame() for metric in DECOMPOSITION_METRICS}}
    ids = [column for column in PAIR_COLUMNS if column in long]+["source_metric", "component", "component_label", "effect_unit"]
    summary = summarise_iterations(long.rename(columns={"metric": "source_metric", "value": "component_effect"}),
        ["component_effect"], id_columns=ids).drop(columns="metric").rename(columns={"source_metric": "metric"})
    eligibility_keys = [column for column in PAIR_COLUMNS if column in eligibility]+["metric", "window"]
    summary = summary.merge(eligibility.drop(columns="complete_horizon"), on=eligibility_keys, how="left", validate="many_to_one")
    counts = pd.concat(count_blocks, ignore_index=True) if count_blocks else pd.DataFrame()
    count_ids = [column for column in PAIR_COLUMNS if column in counts]
    count_summary = summarise_iterations(counts, [column for column in COUNT_COLUMNS if column in counts], id_columns=count_ids) if not counts.empty else pd.DataFrame()
    return {"per_iteration": long, "summaries": summary, "eligibility": eligibility,
            "seed_consistency": component_seed_consistency(summary),
            "count_summaries": count_summary,
            "tables": {metric: postprocessing_component_table(summary, metric) for metric in DECOMPOSITION_METRICS}}


def component_seed_consistency(summaries):
    """Summarise paired seed effects without pooling the three components."""
    groups = [column for column in ["dataset", "study", "model", "stage", "intervention",
              "metric", "window", "component", "component_label", "effect_unit"] if column in summaries]
    return seed_consistency(summaries, group_columns=groups)


def postprocessing_component_table(summaries, metric=None):
    """Readable component values and defined-iteration counts, without pooling.

    All values are already percentage points and are copied without a second
    conversion. ``identity_defined`` states whether all three components share
    a defined full mean or endpoint; it is not a significance indicator.
    """
    required = {"metric", "component", "window", "value", "effect_unit", "n_defined", "n_common_defined", "identity_defined"}
    if required-set(summaries):
        raise ValueError(f"Missing component-summary columns: {sorted(required-set(summaries))}")
    selected = summaries.loc[summaries.metric.eq(metric)].copy() if metric is not None else summaries.copy()
    if not selected.effect_unit.eq("percentage points").all():
        raise ValueError("Component tables require values already expressed in percentage points")
    ids = [column for column in PAIR_COLUMNS if column in selected]+["metric", "window"]
    if selected.duplicated(ids+["component"]).any():
        raise ValueError("Duplicate component summaries cannot be pooled")
    audit = ["n_common_defined", "identity_defined", "n_iterations", "complete_horizon"]
    audit = [column for column in audit if column in selected]
    output = selected[ids+audit].drop_duplicates()
    if output.duplicated(ids).any():
        raise ValueError("Component summaries disagree about their paired eligibility")
    for component in COMPONENT_KEYS:
        block = selected.loc[selected.component.eq(component), ids+["value", "n_defined"]].rename(columns={
            "value": TABLE_LABELS[component], "n_defined": f"{COMPONENT_LABELS[component]} defined iterations"})
        output = output.merge(block, on=ids, how="left", validate="one_to_one")
    return output.reset_index(drop=True)


def decomposition_scales(base, summaries):
    """Copy global pp bounds and expand only for an otherwise clipped component.

    Use this same registry for total mitigation and component plots within RQ IV.
    Absolute share/count scales and unrelated metrics remain unchanged. Bounds
    are never narrowed; an extension is symmetric with 10% headroom around the
    largest finite component. No values, labels, or source scales are mutated.
    """
    if {"metric", "value"}-set(summaries):
        raise ValueError("Decomposition scale checks require long metric/value summaries")
    if "effect_unit" in summaries and not summaries.effect_unit.eq("percentage points").all():
        raise ValueError("Decomposition summaries must already be in percentage points")
    output = deepcopy(base)
    for metric in DECOMPOSITION_METRICS:
        if metric not in base or "difference_limits" not in base[metric]:
            raise ValueError(f"A frozen difference scale is required for {metric}")
        entry = output[metric]
        bounds = np.asarray(entry["difference_limits"], dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid difference bounds for {metric}")
        if entry.get("difference_unit") != "percentage points":
            bounds *= 100
        values = pd.to_numeric(summaries.loc[summaries.metric.eq(metric), "value"], errors="raise").dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"An infinite decomposition component cannot be plotted: {metric}")
        if values.size and ((values < bounds[0]).any() or (values > bounds[1]).any()):
            radius = max(float(np.abs(bounds).max()), float(np.abs(values).max())*1.1)
            bounds = np.array([-radius, radius])
        entry["difference_limits"] = bounds.tolist()
        entry["difference_unit"] = "percentage points"
    return output
