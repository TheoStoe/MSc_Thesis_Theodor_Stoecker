"""One visual convention for all seven research-question notebooks.

Functions consume checked, already aggregated dataframes. They never read run
artifacts, fit a model, pool probe arms, or infer independent user replications.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

MODEL_ORDER = ["position_biased", "ncm_inspired_gru_uniform", "ncm_inspired_gru_mansoury", "direct_llm", "agentic_recommender"]
MODEL_COLOURS = dict(zip(MODEL_ORDER, ["#CC79A7", "#009E73", "#006B50", "#E69F00", "#0072B2"]))
MODEL_LABELS = dict(zip(MODEL_ORDER, ["Position-biased", "NCM P3 (uniform)", "NCM P4 (Mansoury)", "Direct LLM", "Agent"]))
NEUTRAL = "#6B7280"
SEED_MARKERS = {42: "o", 43: "s", 44: "^"}
MODEL_MARKERS = dict(zip(MODEL_ORDER, ["o", "s", "^", "D", "X"]))
SERIES_MARKERS = ["o", "s", "^", "D", "X", "P", "v", "<", ">", "h"]
LINE_STYLES = ["-", "--", ":", "-.", (0, (5, 2, 1, 2)), (0, (2, 2))]
PROBE_STYLES = {"canonical": ("-", "o"), "replica": ("--", "s"), "shuffled": (":", "^"), "original_order_probe": ("-.", "D")}

LABELS = {
    "head_click_share": "Head-click share", "head_exposure": "Head exposure",
    "weighted_head_exposure": "Weighted head exposure",
    "head_exposure_ranks_2_10": "Head exposure (displayed ranks 2–10)",
    "weighted_head_exposure_ranks_2_10": "Weighted head exposure (displayed ranks 2–10)",
    "click_exposure_gap": "Click-exposure gap (unweighted exposure)",
    "weighted_click_exposure_gap": "Click-exposure gap (rank-weighted exposure)",
    "click_rate": "Click rate", "no_click_rate": "No-click rate", "invalid_response_rate": "Invalid-response rate",
    "valid_clicks": "Valid click counts", "valid_click_count": "Valid click counts",
    "canonical_valid_clicks": "Post-processing canonical valid clicks",
    "original_order_valid_clicks": "Original-order probe valid clicks",
    "control_valid_clicks": "Control canonical valid clicks",
    "both_arms_valid_clicks": "Valid clicks in both post-processing arms",
    "exposure_coverage": "Catalogue coverage (exposure)", "click_coverage": "Catalogue coverage (clicks)",
    "cumulative_exposure_coverage": "Cumulative catalogue coverage (exposure)",
    "cumulative_click_coverage": "Cumulative catalogue coverage (clicks)",
    "exposure_gini": "Gini (exposure)", "click_gini": "Gini (clicks)",
    "mean_clicked_rank": "Mean clicked rank", "first_position_click_share": "First-position click share",
    "rank_one_click_share": "First-position click share", "click_share": "Share of valid clicks",
    "female_click_share": "Women-authored share among valid clicks",
    "female_exposure": "Women-authored exposure share",
    "weighted_female_exposure": "Rank-weighted women-authored exposure share",
    "female_exposure_share": "Women-authored exposure share",
    "female_weighted_exposure_share": "Rank-weighted women-authored exposure share",
    "female_known_click_share": "Women-authored click share (women + men)",
    "female_known_exposure": "Women-authored exposure share (women + men)",
    "weighted_female_known_exposure": "Rank-weighted women-authored exposure (women + men)",
    "female_candidate_share": "Women-authored share among eligible candidates",
    "female_exposure_candidate_gap": "Women-authored exposure minus candidate share",
    "female_click_exposure_gap": "Gender click-exposure gap (all items)",
    "weighted_female_click_exposure_gap": "Gender click-exposure gap (rank-weighted, all items)",
    "unknown_exposure": "Unknown exposure share", "ambiguous_exposure": "Ambiguous exposure share",
    "unknown_click_share": "Unknown share among valid clicks", "ambiguous_click_share": "Ambiguous share among valid clicks",
    "replica_agreement": "Same-item replica agreement", "shuffled_agreement": "Same-item shuffled agreement",
    "replica_minus_shuffled": "Replica agreement minus shuffled agreement",
    "order_sensitivity": "Order sensitivity", "computational_tvd": "Item-choice probability TVD",
    "active_users": "Active-user counts", "departures": "Departures", "replacements": "Replacements",
    "sponsored_exposures": "Sponsored exposures", "sponsored_clicks": "Sponsored clicks",
    "advertising_eligible_users": "Users eligible for advertising",
    "profile_changes": "Applied profile changes", "reflection_failure_rate": "Reflection failure rate",
    "invalid_rate": "Invalid-response rate", "shuffle_tvd": "Item-choice probability TVD",
    "replica_tvd": "Identical-order probability TVD", "reflection_error_rate": "Reflection failure rate",
    "female_among_binary_clicks": "Women-authored click share (women + men)",
    "female_among_binary_exposure": "Women-authored exposure share (women + men)",
    "weighted_female_among_binary_exposure": "Rank-weighted women-authored exposure (women + men)",
    "binary_female_click_exposure_gap": "Gender click-exposure gap (women + men)",
    "weighted_binary_female_click_exposure_gap": "Gender click-exposure gap (rank-weighted, women + men)",
    "eligible_female_share": "Women-authored share among eligible candidates",
    "female_exposure_minus_candidate_share": "Women-authored exposure minus candidate share",
    "weighted_female_exposure_minus_candidate_share": "Rank-weighted women-authored exposure minus candidate share",
    "weighted_unknown_exposure": "Rank-weighted unknown exposure share",
    "weighted_ambiguous_exposure": "Rank-weighted ambiguous exposure share",
    "agreement_sensitivity": "Replica agreement minus shuffled agreement",
    "replica_pair_agreement": "Same-item replica agreement (both click)",
    "replica_action_agreement": "Click/no-click replica agreement",
    "shuffled_action_agreement": "Click/no-click shuffled agreement",
    "replica_action_disagreement": "Click/no-click replica disagreement",
    "profile_change_rate": "Applied profile-change rate",
    "cumulative_profile_changed_users": "Users with at least one applied profile change",
    "reflection_requested": "Requested reflection updates", "reflection_applied": "Successful reflection updates",
    "reflection_failed": "Failed reflection updates", "invocations": "Module-invocation counts",
    "invocation_rate": "Proportion of decisions invoking the module",
}
COUNT_WORDS = ("count", "users", "opportunities", "decisions", "invocations", "departures", "replacements", "clicks", "exposures", "updates", "profile_changes")


def label(column: str) -> str:
    return LABELS.get(column, column.replace("_", " ").capitalize())


def is_share(metric: str) -> bool:
    if metric in {"female_among_binary_clicks", "female_among_binary_exposure", "weighted_female_among_binary_exposure", "shuffle_tvd", "replica_tvd", "agreement_sensitivity"}:
        return True
    if metric in {"valid_clicks", "sponsored_clicks", "sponsored_exposures", "profile_changes"}:
        return False
    return any(word in metric for word in ("share", "rate", "coverage", "gini", "agreement", "tvd", "exposure", "gap")) and not any(metric.endswith(word) for word in ("count", "counts", "exposures"))


def is_signed(metric: str) -> bool:
    return "gap" in metric or "minus_candidate" in metric or metric in {"replica_minus_shuffled", "order_sensitivity", "agreement_sensitivity"}


def _finite(values):
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def make_scale_registry(frames: dict[str, pd.DataFrame]) -> dict:
    """Freeze each metric's axes once from the whole snapshot, not each panel."""
    collected = {}
    for frame in frames.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        for column in frame.select_dtypes(include="number"):
            collected.setdefault(column, []).extend(_finite(frame[column]).tolist())
    registry = {}
    for metric, values in collected.items():
        array = _finite(values)
        maximum = float(np.abs(array).max()) if array.size else 1
        if metric == "mean_clicked_rank":
            limits = [1, 10]
        elif is_signed(metric):
            limits = [-1, 1]
        elif is_share(metric):
            limits = [0, 1]
        else:
            limits = [0, max(1, maximum * 1.08)]
        registry[metric] = {"limits": limits, "difference_limits": [-max(.01, maximum * 1.08), max(.01, maximum * 1.08)]}
    return registry


def _axis(axis, metric, scales, *, difference=False, reference=None, effect_unit=None):
    entry = scales.get(metric, {})
    if difference:
        limits = entry.get("difference_limits", [-1, 1])
        if effect_unit == "percentage points" and entry.get("difference_unit") != "percentage points":
            limits = [100 * v for v in limits]
        axis.axhline(0, color=NEUTRAL, linewidth=.8, zorder=0)
    else:
        limits = entry.get("limits", [-1, 1] if is_signed(metric) else [0, 1])
        if is_signed(metric):
            axis.axhline(0, color=NEUTRAL, linewidth=.8, zorder=0)
    axis.set_ylim(*limits)
    count_axis = not is_share(metric) and any(word in metric for word in COUNT_WORDS)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=count_axis))
    ratio_units = is_share(metric) and not any(token in metric for token in ("gini", "tvd"))
    if ratio_units:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    if ((difference and ratio_units) or (not difference and is_signed(metric))) and effect_unit != "percentage points":
        axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{100*x:g}"))
    elif difference:
        axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    suffix = " difference (pp)" if difference and (ratio_units or effect_unit == "percentage points") else (" difference" if difference else " (pp)" if is_signed(metric) else "")
    axis.set_ylabel(fill(label(metric) + suffix, 37), fontsize=9)
    axis.grid(axis="both", color="#D8D8D8", linewidth=.5, alpha=.7)
    axis.set_axisbelow(True)
    sns.despine(ax=axis)


def _ordered(values, preferred=()):
    existing = list(dict.fromkeys(values))
    return [v for v in preferred if v in existing] + sorted([v for v in existing if v not in preferred], key=str)


def _series_label(value, series):
    if series == "cluster":
        return f"Cluster {_cluster_number(value)}"
    return MODEL_LABELS.get(value, str(value))


def _legend(figure, handles, title):
    figure.suptitle(title, fontsize=13, y=.995)
    lines = math.ceil(len(handles) / 3) if handles else 0
    bottom = min(.36, (.35 + .24 * lines) / figure.get_figheight())
    figure.tight_layout(rect=(0, bottom, 1, .965))
    if handles:
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .008), ncol=min(3, len(handles)), frameon=False, fontsize=8)
    return figure


def _protocol_scales(frame, metrics, scales, overrides):
    """Apply explicit protocol count bounds without changing the shared registry."""
    resolved = {metric: dict(settings) for metric, settings in scales.items()}
    for metric, limits in (overrides or {}).items():
        limits = tuple(float(value) for value in limits)
        if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError("Scale overrides need two increasing finite bounds")
        if is_share(metric) and not is_signed(metric) and limits != (0, 1):
            raise ValueError("Bounded shares retain the complete 0--1 axis")
        if not is_signed(metric) and not is_share(metric) and limits[0] != 0:
            raise ValueError("Protocol count axes start at zero")
        if metric in metrics and metric in frame:
            values = _finite(frame[metric])
            if values.size and (values.min() < limits[0] - 1e-10 or values.max() > limits[1] + 1e-10):
                raise ValueError(f"Scale override would hide observed {metric} values")
        resolved.setdefault(metric, {})["limits"] = limits
    return resolved


def trajectory_grid(frame, metrics, *, title, scales, facet="seed", series="model", scale_overrides=None):
    """Rows are metrics; columns are explicit facets; no seed-averaged curves."""
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    scales = _protocol_scales(frame, metrics, scales, scale_overrides)
    facets = _ordered(frame[facet].dropna().tolist(), [42, 43, 44] if facet == "seed" else MODEL_ORDER)
    if not facets:
        return empty_figure(title, "No completed observations in this selection.")
    figure, axes = plt.subplots(len(metrics), len(facets), squeeze=False,
                                figsize=(max(6, 4.1*len(facets)), max(3.8, 3*len(metrics))))
    figure._analysis_spec = {"kind": "trajectory", "facet": facet, "series": series, "metrics": metrics}
    series_values = _ordered(frame[series].dropna().tolist(), MODEL_ORDER)
    styles = {value: LINE_STYLES[i % len(LINE_STYLES)] for i, value in enumerate(series_values)}
    markers = {value: MODEL_MARKERS.get(value, SERIES_MARKERS[i % len(SERIES_MARKERS)])
               if series == "model" else SERIES_MARKERS[i % len(SERIES_MARKERS)]
               for i, value in enumerate(series_values)}
    handles = {}
    for row, metric in enumerate(metrics):
        for col, value in enumerate(facets):
            axis = axes[row, col]
            part = frame.loc[frame[facet].eq(value)]
            if metric not in part:
                axis.text(.5, .5, "Metric unavailable", ha="center", transform=axis.transAxes)
                _axis(axis, metric, scales)
                continue
            # Each physical trajectory stays separate, even when it shares a label.
            group_columns = [series] + [c for c in ("seed", "physical_run_id", "condition_id") if c in part and c != series and c != facet]
            for group_key, group in part.groupby(group_columns, dropna=False, sort=False):
                group = group.sort_values("iteration")
                if group["iteration"].duplicated().any():
                    raise ValueError("Filter duplicate logical aliases before plotting a physical trajectory")
                key = group[series].iloc[0]
                model = group["model"].iloc[0] if "model" in group else key
                colour = MODEL_COLOURS.get(model, NEUTRAL)
                style = styles[key] if series != "model" else "-"
                seed = int(group["seed"].iloc[0]) if "seed" in group else 42
                # Seed is already identified by its panel. Shapes then identify
                # the compared series and stay the same across seed panels.
                marker = markers[key] if facet == "seed" else SEED_MARKERS.get(seed, "o")
                axis.plot(group.iteration, group[metric], color=colour, linestyle=style,
                          marker=marker, markevery=5, markersize=3.5, linewidth=1.4)
                text = _series_label(key, series)
                handles[text] = Line2D([], [], color=colour if series == "model" else NEUTRAL,
                                       linestyle=style, marker=marker if facet == "seed" else "None",
                                       label=fill(text, 43), markersize=4)
            _axis(axis, metric, scales)
            axis.set_xlim(1, 30); axis.set_xticks([1, 5, 10, 15, 20, 25, 30]); axis.margins(x=0)
            axis.set_xlabel("Iteration")
            axis.set_title(_facet_title(value, facet), fontsize=10)
    if facet != "seed" and "seed" in frame and frame.seed.nunique() > 1:
        for seed in sorted(frame.seed.unique()):
            handles[f"seed{seed}"] = Line2D([], [], color=NEUTRAL, marker=SEED_MARKERS.get(int(seed), "o"), linestyle="none", label=f"Seed {seed}")
    if series != "model" and facet != "model" and "model" in frame:
        for model in _ordered(frame.model.dropna().tolist(), MODEL_ORDER):
            handles[f"model_{model}"] = Line2D([], [], color=MODEL_COLOURS.get(model, NEUTRAL), linestyle="-", label=MODEL_LABELS.get(model, model))
    if series == "model":
        ordered = [MODEL_LABELS.get(model, model) for model in MODEL_ORDER]
    else:
        ordered = [_series_label(value, series) for value in series_values]
    ordered += [key for key in handles if key not in ordered]
    return _legend(figure, [handles[key] for key in ordered if key in handles], title)


def _effect_category_label(value, *, facet):
    text = str(value)
    if facet == "stage" and text.lower().endswith("minus control"):
        if "post-processing" in text.lower():
            return "Post-processing"
        if "in-processing" in text.lower():
            return "In-processing"
    if text.lower().endswith(" minus control"):
        return text[:-len(" minus control")] + "\nminus control"
    return fill(text, 20 if facet == "stage" else 26)


def _facet_title(value, facet):
    if facet == "stage":
        return {"L0_control": "L0: Control", "L1_feature": "L1: Feature feedback",
                "L2_ml_model": "L2: Model feedback", "L3_individual": "L3: Individual feedback",
                "L4_sampling": "L4: Sampling feedback", "L5_outcome": "L5: Outcome feedback"}.get(str(value), str(value))
    if facet == "seed":
        return f"Seed {value}"
    if facet == "dataset":
        return {"movielens": "MovieLens", "goodreads": "Goodreads"}.get(str(value), str(value))
    if facet == "model":
        return MODEL_LABELS.get(value, str(value))
    return str(value).replace("mean_1_30", "Mean over iterations 1–30").replace("iteration_30", "Iteration 30")


def effect_grid(frame, metrics, *, title, scales, category="comparison", value="effect", facet="summary", category_order=()):
    """Show each paired seed effect, retaining model colours and seed markers."""
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    if frame.empty:
        return empty_figure(title, "No complete matched pairs are available.")
    facets = _ordered(frame[facet].dropna().tolist(), ["mean_1_30", "iteration_30", "mean", "endpoint"])
    # A stage facet contains only that stage's interventions. Categories from
    # another facet must not create empty positions or misleading labels here.
    facet_categories = {key: _ordered(frame.loc[frame[facet].eq(key), category].dropna().tolist(), category_order)
                        for key in facets}
    figure, axes = plt.subplots(len(metrics), len(facets), squeeze=False,
                                figsize=(max(6, (4.8 if facet == "stage" else 6)*len(facets)), max(4.2, 3.8*len(metrics))))
    figure._analysis_spec = {"kind": "effect", "facet": facet, "category": category, "metrics": metrics}
    handles = {}
    models = _ordered(frame.model.dropna().tolist(), MODEL_ORDER) if "model" in frame else [None]
    seeds = _ordered(frame.seed.dropna().tolist(), [42, 43, 44]) if "seed" in frame else [42]
    width = .66 / max(1, len(models))
    for row, metric in enumerate(metrics):
        for col, facet_value in enumerate(facets):
            axis = axes[row, col]
            part = frame.loc[frame[facet].eq(facet_value) & frame.metric.eq(metric)]
            categories = facet_categories[facet_value]
            for _, item in part.iterrows():
                model = item.get("model", None)
                seed = int(item.get("seed", 42))
                x = categories.index(item[category]) + (models.index(model) - (len(models)-1)/2)*width
                x += (seeds.index(seed) - (len(seeds)-1)/2)*width/4
                colour = MODEL_COLOURS.get(model, NEUTRAL)
                axis.scatter(x, item[value], color=colour, marker=SEED_MARKERS.get(seed, "o"), s=34, edgecolors="white", linewidths=.35, zorder=3)
                handles[model] = Line2D([], [], color=colour, marker="o", linestyle="none", label=MODEL_LABELS.get(model, str(model)), markersize=5)
            effect_unit = part.effect_unit.iloc[0] if "effect_unit" in part and len(part) else None
            _axis(axis, metric, scales, difference=True, effect_unit=effect_unit)
            axis.set_xticks(range(len(categories)), [_effect_category_label(x, facet=facet) for x in categories], fontsize=8)
            axis.set_xlim(-.5, max(.5, len(categories)-.5))
            axis.set_title(_facet_title(facet_value, facet), fontsize=10)
    for seed in seeds:
        handles[f"seed_{seed}"] = Line2D([], [], color=NEUTRAL, marker=SEED_MARKERS.get(int(seed), "o"), linestyle="none", label=f"Seed {seed}", markersize=5)
    ordered = [m for m in MODEL_ORDER if m in handles] + [key for key in handles if key not in MODEL_ORDER]
    return _legend(figure, [handles[key] for key in ordered], title)


CLUSTER_COLOURS = {0: "#440154", 1: "#3B528B", 2: "#21918C", 3: "#5EC962", 4: "#FDE725"}
CLUSTER_HATCHES = ("", "//", "\\\\", "xx", "..")


def _cluster_number(value):
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number.is_integer() else value
    except (TypeError, ValueError):
        return value


def _cluster_hatch(cluster, fallback):
    index = cluster if isinstance(cluster, int) and cluster >= 0 else fallback
    return CLUSTER_HATCHES[index % len(CLUSTER_HATCHES)]


def composition_grid(frame, *, title, snapshots=(1, 10, 20, 30), facet="seed",
                     row="condition_label", group="cluster", value="active_share", x="iteration"):
    """100% stacked bars for saved cluster shares, without resampling or pooling.

    Conditions form rows, seeds form columns, and each bar is one selected saved
    iteration. Undefined composition after all users depart stays explicitly
    undefined rather than being normalised into an invented population.
    Cluster colours and hatches have meanings local to this labelled companion;
    response-model colours elsewhere are unchanged.
    """
    if frame.empty:
        return empty_figure(title, "No saved cluster-composition observations are available.")
    required = {facet, group, value, x}
    if required - set(frame):
        raise ValueError(f"Missing cluster-composition columns: {sorted(required-set(frame))}")
    data = frame.copy()
    if row not in data:
        row = "condition_id" if "condition_id" in data else "_composition_row"
        if row not in data:
            data[row] = "Cluster composition"
    data[group] = data[group].map(_cluster_number)
    snapshots = tuple(snapshots)
    if not snapshots or len(snapshots) != len(set(snapshots)):
        raise ValueError("Choose distinct saved iteration snapshots")
    data = data.loc[data[x].isin(snapshots)]
    if data.empty:
        return empty_figure(title, "None of the requested iteration snapshots was saved.")
    if data.duplicated([row, facet, x, group]).any():
        raise ValueError("Cluster composition needs one source observation per condition, seed, iteration, and cluster")
    facets = _ordered(data[facet].dropna().tolist(), [42, 43, 44] if facet == "seed" else ())
    rows = _ordered(data[row].dropna().tolist())
    clusters = _ordered(data[group].dropna().tolist(), range(5))
    figure, axes = plt.subplots(len(rows), len(facets), squeeze=False,
        figsize=(max(6, 4.1*len(facets)), max(3.8, 2.8*len(rows))))
    figure._analysis_spec = {"kind": "composition", "facet": facet, "snapshots": list(snapshots), "group": group}
    handles = [Patch(facecolor=CLUSTER_COLOURS.get(cluster, NEUTRAL), edgecolor="black", linewidth=.4,
                     hatch=_cluster_hatch(cluster, index), label=f"Cluster {cluster}")
               for index, cluster in enumerate(clusters)]
    positions = np.arange(len(snapshots))
    for row_index, row_value in enumerate(rows):
        for column_index, facet_value in enumerate(facets):
            axis = axes[row_index, column_index]
            panel = data.loc[data[row].eq(row_value) & data[facet].eq(facet_value)]
            for position, snapshot in zip(positions, snapshots):
                part = panel.loc[panel[x].eq(snapshot)].set_index(group)
                if part.empty:
                    axis.text(position, .5, "No saved\nobservation", ha="center", va="center", fontsize=7)
                    continue
                mass = pd.to_numeric(part[value], errors="raise")
                if mass.isna().all():
                    no_users = "active_users" in part and part.active_users.sum() == 0
                    text = "No active\nusers" if no_users else "Undefined\ncomposition"
                    axis.text(position, .5, text, ha="center", va="center", fontsize=7)
                    continue
                if mass.isna().any() or not np.isfinite(mass).all() or (mass < 0).any():
                    raise ValueError("Cluster shares must be finite and nonnegative, or all undefined")
                if not math.isclose(float(mass.sum()), 1, rel_tol=1e-7, abs_tol=1e-8):
                    raise ValueError("Defined cluster shares must sum to one; no implicit renormalisation is applied")
                bottom = 0.0
                for index, cluster in enumerate(clusters):
                    height = float(mass.get(cluster, 0.0))
                    axis.bar(position, height, bottom=bottom, width=.68,
                        color=CLUSTER_COLOURS.get(cluster, NEUTRAL), edgecolor="black", linewidth=.35,
                        hatch=_cluster_hatch(cluster, index))
                    bottom += height
            axis.set_ylim(0, 1)
            axis.set_yticks([0, .25, .5, .75, 1])
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            axis.set_xticks(positions, [str(snapshot) for snapshot in snapshots])
            axis.set_xlim(-.55, len(snapshots)-.45)
            axis.set_xlabel("Iteration")
            axis.set_ylabel("Active-user share")
            axis.set_title(fill(str(row_value), 42)+"\n"+_facet_title(facet_value, facet), fontsize=9)
            axis.grid(axis="y", color="#D8D8D8", linewidth=.5, alpha=.7)
            axis.set_axisbelow(True)
            sns.despine(ax=axis)
    return _legend(figure, handles, title)


def rank_grid(frame, *, title, scales=None):
    """Clicked-rank composition for canonical/replica/shuffled arms."""
    if frame.empty:
        return empty_figure(title, "No rank diagnostics available.")
    models = _ordered(frame.model.dropna().tolist(), MODEL_ORDER)
    figure, axes = plt.subplots(1, len(models), squeeze=False, figsize=(max(6, 3.5*len(models)), 4.3))
    figure._analysis_spec = {"kind": "rank", "facet": "model", "pooled_iterations": [1, 30]}
    handles = {}
    for col, model in enumerate(models):
        axis = axes[0, col]
        part = frame.loc[frame.model.eq(model)]
        arm_column = "presentation_arm" if "presentation_arm" in part else "arm"
        measure = "click_share" if "click_share" in part else "clicks"
        # A plotting call must receive rank summaries already calculated per seed.
        for (seed, arm), group in part.groupby(["seed", arm_column], sort=False):
            if group["rank"].duplicated().any():
                raise ValueError("Aggregate rank counts within each run/seed before plotting")
            style, marker = PROBE_STYLES.get(arm, ("-", "o"))
            axis.plot(group["rank"], group[measure], color=MODEL_COLOURS.get(model, NEUTRAL), linestyle=style,
                      marker=SEED_MARKERS.get(int(seed), marker), markersize=3, linewidth=1.2)
            arm_label = {"canonical": "Canonical", "replica": "Same-order replica", "shuffled": "Shuffled", "original_order_probe": "Original-order probe"}.get(arm, str(arm))
            handles[f"{arm}_{seed}"] = Line2D([], [], color=NEUTRAL, linestyle=style, marker=SEED_MARKERS.get(int(seed), marker), label=f"{arm_label}; seed {seed}", markersize=4)
        axis.set_xlim(1, 10); axis.set_xticks([1, 3, 5, 7, 10]); axis.margins(x=0)
        axis.set_ylim(0, 1); axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_title(MODEL_LABELS.get(model, model), fontsize=10)
        axis.set_xlabel("Displayed rank"); axis.set_ylabel("Share of valid clicks")
        axis.grid(alpha=.25); sns.despine(ax=axis)
    return _legend(figure, list(handles.values()), title)


def association_grid(frame, *, title, x, y, x_label, y_label):
    """Within-seed exploratory associations, with every model point visible."""
    if frame.empty:
        return empty_figure(title, "No complete model-level pairs available.")
    seeds = sorted(frame.seed.unique())
    figure, axes = plt.subplots(1, len(seeds), squeeze=False, figsize=(max(6, 4.4*len(seeds)), 4.5))
    figure._analysis_spec = {"kind": "association", "facet": "seed", "x": x, "y": y}
    xv, yv = _finite(frame[x]), _finite(frame[y])
    xmax = max(.01, float(np.abs(xv).max())*1.18) if xv.size else 1
    ymax = 100 if y == "head_click_reduction_pp" else max(.01, float(np.abs(yv).max())*1.18) if yv.size else 1
    computational = "predictor_family" in frame and frame.predictor_family.astype(str).str.startswith("Computational").all()
    x_limits = (0, 1) if computational else (-1, 1) if "predictor_family" in frame else (-xmax, xmax)
    handles = {}
    for col, seed in enumerate(seeds):
        axis = axes[0, col]
        for _, row in frame.loc[frame.seed.eq(seed)].iterrows():
            colour = MODEL_COLOURS.get(row.model, NEUTRAL)
            marker = MODEL_MARKERS.get(row.model, "o")
            axis.scatter(row[x], row[y], color=colour, s=40, marker=marker)
            handles[row.model] = Line2D([], [], color=colour, marker=marker, linestyle="none", label=MODEL_LABELS.get(row.model, row.model), markersize=5)
        axis.set_xlim(*x_limits); axis.set_ylim(-ymax, ymax)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        axis.axhline(0, color=NEUTRAL, linewidth=.7); axis.axvline(0, color=NEUTRAL, linewidth=.7)
        axis.set_xlabel(fill(x_label, 39)); axis.set_ylabel(fill(y_label, 39)); axis.set_title(f"Seed {seed}")
        axis.grid(alpha=.25); sns.despine(ax=axis)
    return _legend(figure, [handles[key] for key in MODEL_ORDER if key in handles], title)


def empty_figure(title, reason):
    figure, axis = plt.subplots(figsize=(8, 2.6))
    axis.axis("off"); axis.text(.5, .5, fill(reason, 85), ha="center", va="center", transform=axis.transAxes)
    figure.suptitle(title)
    figure._analysis_spec = {"kind": "empty"}
    return figure


def annotate_figure_context(figure, context):
    """Keep stage, seed, and iteration scope visible in standalone exports.

    Extra header space is added instead of shrinking the original plot panels.
    Calling this again replaces the context and does not grow the figure twice.
    """
    if hasattr(figure, "_analysis_context_text"):
        figure._analysis_context_text.remove()
    if not hasattr(figure, "_analysis_base_size"):
        figure._analysis_base_size = tuple(figure.get_size_inches())
    width, base_height = figure._analysis_base_size
    lines = []
    for line in context["text"].splitlines():
        lines.extend(fill(line, max(65, int(width * 15))).splitlines())
    header = .48 + .17 * len(lines)
    height = base_height + max(0, header - .35)
    figure.set_size_inches(width, height, forward=True)
    if figure._suptitle is not None:
        figure._suptitle.set_position((.5, 1 - .06 / height))
        figure._suptitle.set_verticalalignment("top")
    figure._analysis_context_text = figure.text(.5, 1 - .34 / height, "\n".join(lines),
        ha="center", va="top", fontsize=8, linespacing=1.25)
    legend_rows = max((math.ceil(len(legend.get_texts()) / max(1, legend._ncols))
                       for legend in figure.legends), default=0)
    bottom = min(.36, (.35 + .24 * legend_rows) / height)
    # The reserved header already includes the title; avoid reserving it twice.
    title_in_layout = figure._suptitle.get_in_layout() if figure._suptitle is not None else None
    if figure._suptitle is not None:
        figure._suptitle.set_in_layout(False)
    figure.tight_layout(rect=(0, bottom, 1, 1 - header / height))
    if figure._suptitle is not None:
        figure._suptitle.set_in_layout(title_in_layout)
    figure._analysis_context = context
    return figure


def save_figure(figure, *, analysis_name, output_dir, caption, provenance, recommended=False, rationale=""):
    """Export the plot and provenance without generated caption text."""
    from .tables import table_label
    table_label(analysis_name)  # one shared snake_case validation
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    pdf, png = [directory / f"{analysis_name}.{extension}" for extension in ("pdf", "png")]
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    figure.savefig(png, bbox_inches="tight", dpi=300, facecolor="white")
    figure_label = f"chap:results:fig:{analysis_name}"
    tex = "\n".join([r"\begin{figure}[htbp!]", r"\centering", rf"\includegraphics[width=\linewidth]{{\detokenize{{{pdf.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()}}}}}",
                      r"\refstepcounter{figure}", rf"\label{{{figure_label}}}", r"\end{figure}", ""])
    (directory / f"{analysis_name}.tex").write_text(tex, encoding="utf-8")
    (directory / f"{analysis_name}.json").write_text(json.dumps({"analysis_name": analysis_name, "label": figure_label,
        "recommended_for_main_text": recommended, "rationale": rationale,
        "context": getattr(figure, "_analysis_context", None),
        "provenance": provenance, "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (pdf, png)}}, indent=2), encoding="utf-8")
    return {"pdf": pdf, "png": png, "label": figure_label}
