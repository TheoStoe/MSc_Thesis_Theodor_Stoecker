"""Descriptive RQ III presentation views of already calculated paired effects.

The first contrast is displayed as a decision-input noise check following the
implementation audit. This module changes neither its source identities nor its
values. Its three observed seed/population effects provide a case-specific
reference, not a calibrated noise floor, confidence interval, or test threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .comparisons import seed_consistency


ORIGINAL_FIRST_COMPARISON = "Private memory: no reflection minus frozen agent"
ABLATION_COMPARISON_ORDER = (
    "Decision-input noise check: no reflection minus frozen agent",
    "Reflection notes: fixed profile minus no reflection",
    "Profile updates: full reflection minus fixed profile",
)


def label_ablation_comparisons(frame, *, include_order=True):
    """Copy a results view, changing only the exact reviewed comparison label.

    Run IDs, condition IDs, numeric measurements, and unrelated comparisons
    remain unchanged. ``ablation_order`` is optional presentation metadata.
    Applying this helper twice is safe.
    """
    if "comparison" not in frame:
        raise ValueError("An ablation view requires the comparison column")
    output = frame.copy(deep=True)
    output["comparison"] = output["comparison"].replace(
        {ORIGINAL_FIRST_COMPARISON: ABLATION_COMPARISON_ORDER[0]})
    if include_order:
        order = {comparison: index for index, comparison in enumerate(ABLATION_COMPARISON_ORDER)}
        output["ablation_order"] = output["comparison"].map(order).astype("Int64")
    return output


def ablation_effect_ranges(effect_summaries, metrics, *, expected_seeds=(42, 43, 44)):
    """Observed seed ranges with the first contrast's matching range alongside.

    The first comparison remains in the returned table and is marked by
    ``is_reference``. Each mean, minimum, and maximum uses defined seed/population
    effects only; ``n_seed_blocks`` is that defined count. Available-but-undefined
    blocks are reported separately. Missing values and missing reference cells
    remain missing. Entirely absent comparison/metric/window cells are not
    invented. Every available dataset, study, model, stage, mitigation, kind,
    metric, time window, and effect unit is retained when grouping and matching.
    """
    metrics = list(metrics)
    required = {"comparison", "metric", "window", "value", "seed", "population_seed"}
    if required - set(effect_summaries):
        raise ValueError(f"Missing ablation summary columns: {sorted(required-set(effect_summaries))}")
    if len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("Expected simulation seeds must be distinct")
    expected_seeds = tuple(expected_seeds)
    selected = label_ablation_comparisons(effect_summaries)
    selected = selected.loc[selected.comparison.isin(ABLATION_COMPARISON_ORDER) & selected.metric.isin(metrics)].copy()
    groups = [column for column in ("dataset", "study", "model", "stage", "intervention", "kind",
              "metric", "window", "effect_unit", "comparison", "ablation_order") if column in selected]
    added = ["n_seed_blocks", "n_available_seed_blocks", "n_undefined_seed_blocks", "expected_seed_count",
             "expected_seeds", "mean_effect", "min_effect", "max_effect", "seed_blocks", "missing_seeds",
             "all_expected_seeds", "reference_min", "reference_max", "reference_n_seed_blocks",
             "reference_available", "is_reference"]
    if selected.empty:
        return pd.DataFrame(columns=groups + added)
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    if np.isinf(selected.value.to_numpy(dtype=float)).any():
        raise ValueError("A paired effect cannot be infinite")
    summary = seed_consistency(selected, expected_seeds=expected_seeds, group_columns=groups)
    available = selected.groupby(groups, dropna=False, sort=False).size().rename("n_available_seed_blocks").reset_index()
    summary = summary.merge(available, on=groups, validate="one_to_one")
    summary["n_undefined_seed_blocks"] = summary.n_available_seed_blocks-summary.n_seed_blocks
    summary["expected_seed_count"] = len(expected_seeds)
    summary["expected_seeds"] = ", ".join(map(str, expected_seeds))
    # The same metric and time summary, in the same scientific context, supply
    # the descriptive reference. Neither a different metric nor an endpoint is
    # allowed to stand in for a missing mean reference.
    reference_keys = [column for column in groups if column not in {"comparison", "ablation_order"}]
    reference = summary.loc[summary.comparison.eq(ABLATION_COMPARISON_ORDER[0]),
        reference_keys + ["min_effect", "max_effect", "n_seed_blocks"]].rename(columns={
            "min_effect": "reference_min", "max_effect": "reference_max", "n_seed_blocks": "reference_n_seed_blocks"})
    summary = summary.merge(reference, on=reference_keys, how="left", validate="many_to_one")
    summary["reference_available"] = summary.reference_n_seed_blocks.fillna(0).gt(0)
    summary["is_reference"] = summary.comparison.eq(ABLATION_COMPARISON_ORDER[0])
    summary["reference_n_seed_blocks"] = summary.reference_n_seed_blocks.astype("Int64")
    ordered = [column for column in groups if column not in {"comparison", "ablation_order"}] + ["ablation_order"]
    return summary.sort_values(ordered, kind="stable").reset_index(drop=True)
