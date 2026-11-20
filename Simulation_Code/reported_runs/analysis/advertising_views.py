"""Matched views of advertising exposure in fixed displayed ranks 2--10.

The input is an offline supplement calculated from canonical saved slates.
This module reads no run files and changes no metric definitions. Removing
displayed rank 1 in all three conditions leaves a comparable slot scope; it
does not isolate learned feedback from other presentation or displacement
effects. Full-slate measurements remain available alongside the supplement.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .comparisons import (
    RUN_KEYS, RUN_METADATA, SHARE_METRICS, paired_effects, seed_consistency,
    summarise_effects, summarise_iterations,
)


FULL_METRICS = ("head_exposure", "weighted_head_exposure")
SLOT_METRICS = ("head_exposure_ranks_2_10", "weighted_head_exposure_ranks_2_10")
ACCOUNTING_METRICS = (
    "head_exposure_full_minus_ranks_2_10",
    "weighted_head_exposure_full_minus_ranks_2_10",
)
ACCOUNTING_LABELS = {
    ACCOUNTING_METRICS[0]: "Including rank 1 changes Head exposure by (pp)",
    ACCOUNTING_METRICS[1]: "Including rank 1 changes Weighted head exposure by (pp)",
}
ADVERTISING_CONDITIONS = ("O0_organic", "O1_static_ad", "O2_adaptive_ad_three_strike")


def build_slot_views(frame, comparison_plan):
    """Copy slot measurements and reuse the strict matched-summary pipeline.

    Absolute summaries retain raw fractions. Both new exposure contrasts and
    the optional full-minus-tail accounting contrasts are treatment minus
    matched control in percentage points. The latter describe a change in
    same-slate accounting, rather than a causal feedback decomposition.
    Only advertising plan rows in the supplied dataset/study/seed/population
    blocks are used. Missing observations are never replaced by zero, and
    the primary mean still requires all 30 defined per-iteration values.
    """
    required = set(RUN_KEYS + ["iteration"]) | set(FULL_METRICS + SLOT_METRICS + ACCOUNTING_METRICS)
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing advertising slot columns: {sorted(missing)}")
    if not set(SLOT_METRICS + ACCOUNTING_METRICS) <= SHARE_METRICS:
        raise ValueError("Advertising share metrics must be registered for percentage-point effects")
    if not frame.condition_id.isin(ADVERTISING_CONDITIONS).all():
        raise ValueError("Advertising slot views require organic, static, or adaptive advertising conditions")
    for column in ("arm", "presentation_arm", "probe_arm"):
        if column in frame and not frame[column].eq("canonical").all():
            raise ValueError("Advertising slot trajectories must contain canonical observations only")
    if "kind" not in comparison_plan:
        raise ValueError("The comparison plan must identify advertising comparisons by kind")

    trajectory = frame.copy(deep=True)
    contexts = ["dataset", "study", "seed", "population_seed"]
    blocks = trajectory[contexts].drop_duplicates()
    plan = comparison_plan.loc[comparison_plan.kind.eq("advertising")].merge(
        blocks, on=contexts, how="inner", validate="many_to_one")
    summaries = summarise_iterations(trajectory, list(FULL_METRICS + SLOT_METRICS + ACCOUNTING_METRICS))
    effects = paired_effects(trajectory, plan, list(SLOT_METRICS))
    effect_summaries = summarise_effects(effects)
    accounting_effects = paired_effects(trajectory, plan, list(ACCOUNTING_METRICS))
    return {
        "trajectory": trajectory,
        "summaries": summaries,
        "effects": effects,
        "effect_summaries": effect_summaries,
        "seed_consistency": seed_consistency(effect_summaries),
        "accounting_effect_summaries": summarise_effects(accounting_effects),
    }


def slot_scales(base, views=None):
    """Copy global scales, matching tail-effect axes to full-slate axes.

    The two tail shares always use 0--1 absolute bounds. Their effect bounds
    exactly match the corresponding full-slate bounds, expressed explicitly
    in percentage points to prevent a second conversion during plotting.
    Optional views (or an effect-summary DataFrame) are checked for clipping;
    they never tighten or silently expand the frozen global bounds.
    """
    output = deepcopy(base)
    effects = views.get("effect_summaries") if isinstance(views, dict) else views
    for full, tail in zip(FULL_METRICS, SLOT_METRICS):
        if full not in base or "difference_limits" not in base[full]:
            raise ValueError(f"A frozen full-slate difference scale is required for {full}")
        entry = deepcopy(base[full])
        bounds = np.asarray(entry["difference_limits"], dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid full-slate difference bounds for {full}")
        if entry.get("difference_unit") != "percentage points":
            bounds = 100 * bounds
        entry.update(limits=[0, 1], difference_limits=bounds.tolist(), difference_unit="percentage points")
        output[tail] = entry
        if isinstance(effects, pd.DataFrame) and not effects.empty:
            selected = effects.loc[effects.metric.eq(tail)]
            if "effect_unit" in selected and not selected.effect_unit.eq("percentage points").all():
                raise ValueError("Slot-effect scale checks require percentage-point summaries")
            values = pd.to_numeric(selected.value, errors="raise").dropna().to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < bounds[0]).any() or (values > bounds[1]).any():
                raise ValueError(f"Advertising slot effects exceed the shared full-slate bounds for {tail}")
    return output


def slot_accounting_table(summaries):
    """Wide same-slate rank-1 inclusion differences, converted to pp once.

    Input must be raw long absolute summaries returned by ``build_slot_views``.
    Each value is full-slate exposure minus exposure in displayed ranks 2--10
    for the same condition, seed/population, and time window. These are signed
    descriptive accounting differences, not treatment effects or evidence that
    rank 1 caused a feedback effect. Undefined summaries remain missing.
    The wide output cannot accidentally be passed through this helper twice.
    """
    required = {"metric", "window", "value"}
    if required - set(summaries):
        raise ValueError("Accounting tables require raw long summaries with metric, window, and value")
    if "effect_unit" in summaries:
        raise ValueError("Accounting tables require raw absolute summaries, not paired effects")
    selected = summaries.loc[summaries.metric.isin(ACCOUNTING_METRICS)].copy(deep=True)
    identities = [column for column in RUN_METADATA if column in selected] + ["window"]
    if selected.duplicated(identities + ["metric"]).any():
        raise ValueError("Duplicate condition/seed/window accounting summaries cannot be combined")
    if selected.empty:
        return pd.DataFrame(columns=identities + list(ACCOUNTING_LABELS.values()))
    selected["value"] = pd.to_numeric(selected.value, errors="raise")
    if np.isinf(selected.value.to_numpy(dtype=float)).any():
        raise ValueError("Accounting differences cannot be infinite")
    selected["value"] *= 100
    output = selected[identities].drop_duplicates()
    for metric, label in ACCOUNTING_LABELS.items():
        metric_rows = selected.loc[selected.metric.eq(metric), identities + ["value"]].rename(columns={"value": label})
        output = output.merge(metric_rows, on=identities, how="left", validate="one_to_one")
    return output.reset_index(drop=True)
