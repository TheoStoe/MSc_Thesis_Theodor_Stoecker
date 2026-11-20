"""Print-width figures and compact tables for the Results chapter.

These are main-text companions to the RQ notebooks 01--07. They only select,
arrange, and draw values that the snapshot or the shared helpers already
provide; no metric or comparison is defined here. Every figure

* is sized for the thesis text width (14.7 cm) with 7.5--8.5 pt text,
  so it can be included at ``width=\\linewidth`` without shrinking the text,
* shows one time summary (by default the mean over iterations 1--30); the
  iteration-30 endpoint stays in the two-window figures of the RQ notebooks,
* has no in-figure title or context header, because the LaTeX caption takes
  over that role, and
* keeps the model colours of the RQ notebooks. The two NCM greens are close
  for colour-blind readers, so models are always also separated by position
  (effect plots) or by marker (line plots).

Nothing in this module reads run logs, fits a model, or starts a simulation.
"""
from __future__ import annotations

from pathlib import Path
import shutil
from textwrap import fill

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from .figure_context import describe_figure
from .plotting import MODEL_COLOURS, MODEL_LABELS, MODEL_MARKERS, MODEL_ORDER, NEUTRAL, SEED_MARKERS, save_figure

TEXT_WIDTH = 5.8  # inches: the thesis text width (418 pt, 14.7 cm)
GRID = "#DADADA"
RULE = "#B5B5B5"
BAND = "#E6E6E6"
TICK_STEPS = [1, 2, 2.5, 5, 10]
PRINT_STYLE = {
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": .6, "xtick.major.width": .6,
    "ytick.major.width": .6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.dpi": 300,
}
SHORT_LABELS = {
    "head_click_share": "Head-click share", "head_exposure": "Head exposure",
    "weighted_head_exposure": "Weighted head exposure",
    "head_exposure_ranks_2_10": "Head exposure", "weighted_head_exposure_ranks_2_10": "Weighted head exposure",
    "click_exposure_gap": "Click-exposure gap", "weighted_click_exposure_gap": "Weighted click-exposure gap",
    "female_click_share": "Women-authored click share", "female_exposure": "Women-authored exposure",
    "weighted_female_exposure": "Weighted women-authored exposure",
    "sponsored_exposures": "Sponsored items shown", "sponsored_clicks": "Sponsored clicks",
}
CONFIGURATIONS = {
    "L2 -- frozen agent": "L2 frozen agent", "L3 -- no reflection": "L3 no reflection",
    "L3 -- fixed profile": "L3 fixed profile", "L3 -- full reflection": "L3 full reflection",
}
# Table labels stay ASCII apart from en dashes, so the exported .tex compiles with pdfLaTeX.
SAMPLING_CONTRASTS = {
    "L4 -- no refill minus matched L3": "L4 departures, no refill minus L3",
    "L4 -- uniform refill minus matched L3": "L4 uniform refill minus L3",
    "L4 -- cluster refill minus matched L3": "L4 cluster refill minus L3",
}
ADVERTISING_CONTRASTS = {
    "Static advertising minus matched organic": "Static advertising minus organic control",
    "L5 adaptive advertising minus static advertising": "L5 adaptive minus static advertising",
}


def use_print_style():
    """Apply the print style to every figure created afterwards in this kernel."""
    matplotlib.rcParams.update(PRINT_STYLE)


def short_label(metric):
    return SHORT_LABELS.get(metric, str(metric).replace("_", " ").capitalize())


def nice_limits(values, *, step=5, include_zero=True, symmetric=False, pad=.04):
    """Axis limits on multiples of ``step`` that contain every finite value."""
    array = pd.to_numeric(pd.Series(np.ravel(values)), errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return (-float(step), float(step))
    low, high = float(array.min()), float(array.max())
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    span = max(high - low, float(step))
    low, high = low - pad * span, high + pad * span
    if symmetric:
        bound = max(abs(low), abs(high))
        low, high = -bound, bound
    low, high = step * np.floor(low / step), step * np.ceil(high / step)
    return (float(low), float(high))


def _minus(value, _=None):
    """Tick label with a typographic minus sign (and no negative zero)."""
    return f"{0.0 if abs(value) < 1e-12 else value:g}".replace("-", "−")


def _ylabel(metric, unit=" (pp)", width=22):
    return fill(short_label(metric) + unit, width, break_on_hyphens=False)


def _style(axis, *, grid="y"):
    axis.grid(axis=grid, color=GRID, linewidth=.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def _ordered_models(values):
    present = list(dict.fromkeys(values))
    return [m for m in MODEL_ORDER if m in present] + sorted(m for m in present if m not in MODEL_ORDER)


def _model_handles(models):
    return [Patch(facecolor=MODEL_COLOURS.get(m, NEUTRAL), edgecolor="none", label=MODEL_LABELS.get(m, str(m)))
            for m in models]


def _seed_handles(seeds):
    return [Line2D([], [], color=NEUTRAL, marker=SEED_MARKERS.get(int(s), "o"), linestyle="none",
                   markersize=4.5, label=f"Seed {int(s)}") for s in seeds]


def _finish(figure, groups):
    """One legend row per handle group below the panels; their space is reserved before layout."""
    groups = [group for group in groups if group]
    height = figure.get_figheight()
    line = .21
    bottom = (.08 + line * len(groups)) / height if groups else 0
    figure.tight_layout(rect=(0, bottom, 1, 1), h_pad=.7, w_pad=.9)
    for index, handles in enumerate(groups):
        y = (.02 + line * (len(groups) - 1 - index)) / height
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=len(handles), frameon=False,
                      handlelength=1.3, columnspacing=1.1, handletextpad=.4, borderaxespad=0)
    return figure


def _limits_for(limits, metric, fitted):
    if limits is None:
        return fitted
    if isinstance(limits, dict):
        return tuple(limits.get(metric, fitted))
    return tuple(limits)


def _slots(data, category, categories, models, gap=.7):
    """One x slot per category/model pair that has data; categories are separated by a gap."""
    slots, ticks, rules, x = {}, [], {}, 0.0
    for index, value in enumerate(categories):
        present = [m for m in models if (data[category].eq(value) & data.model.eq(m)).any()]
        if index:
            rules[value] = x - (1 + gap) / 2
        start = x
        for model in present:
            slots[(value, model)] = x
            x += 1
        ticks.append((start + x - 1) / 2)
        x += gap
    return slots, ticks, rules, (-.6, x - gap - 1 + .6)


def effect_panels(frame, metrics, *, categories, category="comparison", category_labels=None, facet=None,
                  facet_labels=None, layout="rows", limits=None, step=5, window="mean_1_30", separators=(),
                  reference_category=None, reference_label="Range of the noise check", panel_height=None,
                  show_means=True):
    """Seed-level paired effects for one time summary, as dots per model.

    ``frame`` is a long effect summary with ``metric``, ``window``, ``value``,
    ``model``, ``seed`` and the ``category`` column. Each dot is one paired
    seed/population block (marker = seed); a short horizontal line marks the
    descriptive mean of the available seeds. Every category/model pair has its
    own x slot, in the fixed model order. ``limits`` is one (low, high) pair for
    all panels, a {metric: (low, high)} mapping, or None for limits fitted to
    the plotted values and shared across the figure. Categories in
    ``separators`` get a thin vertical rule on their left. ``reference_category``
    draws the range of that category's seed effects as a light band in each
    panel. With ``facet`` (e.g. ``"dataset"``), metrics form rows and facet
    values columns.
    """
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    required = {"metric", "window", "value", "model", "seed", category} | ({facet} if facet else set())
    if required - set(frame):
        raise ValueError(f"Missing effect columns: {sorted(required - set(frame))}")
    data = frame.loc[frame.window.eq(window) & frame.metric.isin(metrics)].copy()
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    unknown = sorted(set(data[category].dropna()) - set(categories), key=str)
    if unknown:
        raise ValueError(f"Categories without a position in the figure: {unknown}")
    keys = [category, "model", "seed", "metric"] + ([facet] if facet else [])
    if data.duplicated(keys).any():
        raise ValueError("One effect per category, model, seed, metric, and facet is required")
    categories = [c for c in categories if c in set(data[category])]
    labels = {c: (category_labels or {}).get(c, fill(str(c), 22)) for c in categories}
    models = _ordered_models(data.model.dropna())
    seeds = sorted(int(s) for s in data.seed.dropna().unique())
    facets = ([v for v in (facet_labels or {}) if v in set(data[facet])] +
              sorted(v for v in data[facet].unique() if v not in (facet_labels or {}))) if facet else [None]
    stacked = bool(facet) or layout == "rows"
    if stacked:
        rows, cols = len(metrics), len(facets)
        height = panel_height or 1.4
    elif layout == "columns":
        rows, cols = 1, len(metrics)
        height = panel_height or 1.95
    else:
        raise ValueError("layout must be 'rows' or 'columns'")
    figure, axes = plt.subplots(rows, cols, squeeze=False, sharex=stacked,
                                figsize=(TEXT_WIDTH, height * rows + .3 + .21 * 2))
    figure._analysis_spec = {"kind": "effect", "facet": facet or "window", "category": category,
                             "metrics": metrics, "main_text": True}
    reference = {}
    if reference_category is not None:
        groups = ["metric", facet] if facet else "metric"
        reference = {key: (block.value.min(), block.value.max())
                     for key, block in data.loc[data[category].eq(reference_category)].groupby(groups)}
    slots, ticks, rules, x_limits = _slots(data, category, categories, models)
    fitted = nice_limits(data.value, step=step)
    for index, metric in enumerate(metrics):
        for column, facet_value in enumerate(facets):
            axis = axes[index, column] if stacked else axes[0, index]
            part = data.loc[data.metric.eq(metric)]
            if facet:
                part = part.loc[part[facet].eq(facet_value)]
            band = reference.get((metric, facet_value) if facet else metric)
            if band is not None:
                axis.axhspan(*band, color=BAND, linewidth=0, zorder=0)
            axis.axhline(0, color=NEUTRAL, linewidth=.7, zorder=1)
            for name in separators:
                if name in rules:
                    axis.axvline(rules[name], color=RULE, linewidth=.6, zorder=1)
            for (value, model), block in part.groupby([category, "model"], sort=False):
                centre = slots[(value, model)]
                colour = MODEL_COLOURS.get(model, NEUTRAL)
                for _, row in block.iterrows():
                    offset = (seeds.index(int(row.seed)) - (len(seeds) - 1) / 2) * .22
                    axis.scatter(centre + offset, row.value, s=16, color=colour, zorder=3,
                                 marker=SEED_MARKERS.get(int(row.seed), "o"), edgecolors="white", linewidths=.4)
                if show_means and block.value.notna().sum() > 1:
                    axis.hlines(block.value.mean(), centre - .36, centre + .36, colors=colour, linewidth=1.3, zorder=2)
            axis.set_ylim(*_limits_for(limits, metric, fitted))
            axis.set_xlim(*x_limits)
            axis.set_xticks(ticks, [labels[c] for c in categories])
            axis.tick_params(axis="x", length=0)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=TICK_STEPS))
            axis.yaxis.set_major_formatter(FuncFormatter(_minus))
            _style(axis)
            if stacked:
                if column == 0:
                    axis.set_ylabel(_ylabel(metric))
                if index == 0 and facet:
                    axis.set_title((facet_labels or {}).get(facet_value, str(facet_value)))
            else:
                axis.set_title(short_label(metric))
                if index == 0 or isinstance(limits, dict):
                    axis.set_ylabel("Difference (pp)")
    if stacked:
        figure.align_ylabels(axes[:, 0])
    extras = _seed_handles(seeds)
    if show_means:
        extras.append(Line2D([], [], color=NEUTRAL, linewidth=1.3, label="Mean of seeds"))
    if reference:
        extras.append(Patch(facecolor=BAND, edgecolor="none", label=reference_label))
    figure.plotted = data
    return _finish(figure, [_model_handles(models), extras])


def effect_values(plotted, *, category="comparison", category_labels=None, facet=None):
    """Exact values behind an effect figure: one row per point group, seeds as columns."""
    frame = plotted.copy()
    labels = category_labels or {}
    order = {value: index for index, value in enumerate(dict.fromkeys([*labels, *frame[category]]))}
    frame = frame.assign(_category=frame[category].map(order), _model=frame.model.map(
        {model: index for index, model in enumerate(_ordered_models(frame.model))}),
        _metric=frame.metric.map({metric: index for index, metric in enumerate(dict.fromkeys(frame.metric))}))
    if facet:
        facets = {"movielens": 0, "goodreads": 1} if facet == "dataset" else {}
        frame["_facet"] = frame[facet].map(lambda value: facets.get(value, len(facets)))
    frame = frame.sort_values((["_facet", facet] if facet else []) + ["_category", "_model", "_metric", "seed"])
    frame["Contrast"] = frame[category].map(lambda value: " ".join(str(labels.get(value, value)).split()))
    frame["Contrast"] = frame["Contrast"].str.replace("−", "minus", regex=False)
    frame["Model"] = frame.model.map(MODEL_LABELS).fillna(frame.model)
    frame["Metric"] = frame.metric.map(short_label)
    index = ([facet] if facet else []) + ["Contrast", "Model", "Metric"]
    table = frame.pivot_table(index=index, columns="seed", values="value", aggfunc="first", sort=False)
    table.columns = [f"Seed {int(seed)}" for seed in table.columns]
    seeds = list(table.columns)
    table["Mean of seeds"] = table[seeds].mean(axis=1)
    table["Seeds"] = table[seeds].notna().sum(axis=1)
    table = table.reset_index()
    if facet:
        table = table.rename(columns={facet: facet.capitalize()})
    return table


def _iteration_axis(axis, *, bottom):
    axis.set_xlim(1, 30)
    axis.set_xticks([1, 10, 20, 30])
    if bottom:
        axis.set_xlabel("Iteration")


def condition_trajectories(frame, metrics, *, conditions, colours, styles, limits=None, panel_height=1.3):
    """Per-iteration values of several conditions of one model; seeds as columns."""
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    data = frame.loc[frame.condition_id.isin(conditions)].copy()
    seeds = sorted(int(s) for s in data.seed.unique())
    figure, axes = plt.subplots(len(metrics), len(seeds), squeeze=False, sharex=True, sharey="row",
                                figsize=(TEXT_WIDTH, panel_height * len(metrics) + .5))
    figure._analysis_spec = {"kind": "trajectory", "facet": "seed", "series": "condition_label",
                             "metrics": metrics, "main_text": True}
    for row, metric in enumerate(metrics):
        upper = nice_limits(data[metric], step=10, pad=.04)[1]
        for column, seed in enumerate(seeds):
            axis = axes[row, column]
            for condition, label in conditions.items():
                block = data.loc[data.condition_id.eq(condition) & data.seed.eq(seed)].sort_values("iteration")
                if block.iteration.duplicated().any():
                    raise ValueError("One observation per condition, seed, and iteration is required")
                axis.plot(block.iteration, block[metric], color=colours[condition], linestyle=styles[condition],
                          linewidth=1.2, label=label)
            axis.set_ylim(*(limits or {}).get(metric, (0, upper)))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=TICK_STEPS, integer=True))
            _iteration_axis(axis, bottom=row == len(metrics) - 1)
            _style(axis)
            if row == 0:
                axis.set_title(f"Seed {seed}")
            if column == 0:
                axis.set_ylabel(fill(short_label(metric), 16, break_on_hyphens=False))
    figure.align_ylabels(axes[:, 0])
    handles = [Line2D([], [], color=colours[c], linestyle=styles[c], linewidth=1.2, label=label)
               for c, label in conditions.items()]
    figure.plotted = data
    return _finish(figure, [handles])


def advertising_delivery(frame):
    """Sponsored items shown and clicked per iteration: static versus L5 adaptive."""
    conditions = {"O1_static_ad": "Static advertising", "O2_adaptive_ad_three_strike": "L5 adaptive advertising"}
    colours = {"O1_static_ad": "#7A7A7A", "O2_adaptive_ad_three_strike": MODEL_COLOURS["agentic_recommender"]}
    styles = {"O1_static_ad": "--", "O2_adaptive_ad_three_strike": "-"}
    return condition_trajectories(frame, ["sponsored_exposures", "sponsored_clicks"], conditions=conditions,
                                  colours=colours, styles=styles)


def model_trajectories(frame, metrics, *, limits=None, percentage_points=(), panel_height=1.35):
    """Per-iteration values of all response models; seeds as columns, models by colour and marker."""
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    data = frame.copy()
    seeds = sorted(int(s) for s in data.seed.unique())
    models = _ordered_models(data.model)
    figure, axes = plt.subplots(len(metrics), len(seeds), squeeze=False, sharex=True, sharey="row",
                                figsize=(TEXT_WIDTH, panel_height * len(metrics) + .5))
    figure._analysis_spec = {"kind": "trajectory", "facet": "seed", "series": "model", "metrics": metrics,
                             "main_text": True}
    for row, metric in enumerate(metrics):
        scale = 100 if metric in percentage_points else 1
        fitted = nice_limits(data[metric] * scale, step=5 if scale == 100 else .1,
                             include_zero=metric in percentage_points)
        for column, seed in enumerate(seeds):
            axis = axes[row, column]
            if metric in percentage_points:
                axis.axhline(0, color=NEUTRAL, linewidth=.7, zorder=1)
            for model in models:
                block = data.loc[data.model.eq(model) & data.seed.eq(seed)].sort_values("iteration")
                if block.empty:
                    continue
                if block.iteration.duplicated().any():
                    raise ValueError("One observation per model, seed, and iteration is required")
                axis.plot(block.iteration, block[metric] * scale, color=MODEL_COLOURS.get(model, NEUTRAL),
                          marker=MODEL_MARKERS.get(model, "o"), markevery=[0, 4, 9, 14, 19, 24, 29],
                          markersize=3.2, markeredgecolor="white", markeredgewidth=.3, linewidth=1.1)
            axis.set_ylim(*(limits or {}).get(metric, fitted))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=TICK_STEPS))
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0) if scale == 1 else FuncFormatter(_minus))
            _iteration_axis(axis, bottom=row == len(metrics) - 1)
            _style(axis)
            if row == 0:
                axis.set_title(f"Seed {seed}")
            if column == 0:
                axis.set_ylabel(_ylabel(metric, " (pp)" if scale == 100 else "", width=18))
    figure.align_ylabels(axes[:, 0])
    handles = [Line2D([], [], color=MODEL_COLOURS.get(m, NEUTRAL), marker=MODEL_MARKERS.get(m, "o"), markersize=4,
                      linewidth=1.1, label=MODEL_LABELS.get(m, str(m))) for m in models]
    figure.plotted = data
    return _finish(figure, [handles])


def rank_panels(rank_counts, *, arms=("canonical", "shuffled")):
    """Share of valid clicks per displayed rank, one panel per model (pooled over iterations 1--30)."""
    arm = "presentation_arm" if "presentation_arm" in rank_counts else "arm"
    data = rank_counts.loc[rank_counts[arm].isin(arms)].copy()
    if data.duplicated(["model", "seed", arm, "rank"]).any():
        raise ValueError("Aggregate rank counts per model, seed, and arm before plotting")
    models = _ordered_models(data.model)
    seeds = sorted(int(s) for s in data.seed.unique())
    names = {"canonical": "Canonical list", "shuffled": "Shuffled list", "replica": "Same-order replica"}
    look = {arms[0]: {"linestyle": "-", "alpha": 1.0, "markers": True}}
    look.update({other: {"linestyle": "--", "alpha": .75, "markers": False} for other in arms[1:]})
    columns = 3
    rows = int(np.ceil((len(models) + 1) / columns))
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(TEXT_WIDTH, 1.7 * rows + .1), sharey=True)
    figure._analysis_spec = {"kind": "rank", "facet": "model", "pooled_iterations": [1, 30], "main_text": True}
    for position, model in enumerate(models):
        axis = axes.flat[position]
        for presentation in reversed(arms):
            for seed in seeds:
                block = data.loc[data.model.eq(model) & data.seed.eq(seed) & data[arm].eq(presentation)].sort_values("rank")
                style = look[presentation]
                axis.plot(block["rank"], block.click_share, color=MODEL_COLOURS.get(model, NEUTRAL),
                          linestyle=style["linestyle"], alpha=style["alpha"], linewidth=1.0,
                          marker=SEED_MARKERS.get(seed, "o") if style["markers"] else None, markersize=2.8,
                          markeredgecolor="white", markeredgewidth=.3)
        axis.set_title(MODEL_LABELS.get(model, model))
        axis.set_xlim(.6, 10.4)
        axis.set_xticks(range(1, 11))
        axis.tick_params(axis="x", length=0)
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=TICK_STEPS))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.set_xlabel("Displayed rank")
        if position % columns == 0:
            axis.set_ylabel("Share of valid clicks")
        _style(axis)
    legend_axis = axes.flat[len(models)]
    legend_axis.axis("off")
    arm_handles = [Line2D([], [], color=NEUTRAL, linewidth=1.0, label=names.get(a, a), linestyle=look[a]["linestyle"],
                          marker="o" if look[a]["markers"] else None, markersize=2.8) for a in arms]
    first = legend_axis.legend(handles=arm_handles, title="Presentation", loc="upper left", bbox_to_anchor=(.05, 1),
                               frameon=False, alignment="left")
    legend_axis.add_artist(first)
    legend_axis.legend(handles=_seed_handles(seeds), title="Seed (canonical list)", loc="upper left",
                       bbox_to_anchor=(.05, .52), frameon=False, alignment="left")
    for axis in list(axes.flat)[len(models) + 1:]:
        axis.axis("off")
    figure.plotted = data
    figure.tight_layout(h_pad=1.0, w_pad=.9)
    return figure


def order_sensitivity_panels(pairs, *, window="mean_1_30"):
    """Control order sensitivity against the total post-processing reduction, by predictor family."""
    data = pairs.loc[pairs.available.astype(bool) & pairs.window.eq(window)].copy()
    families = [("Computational", "Computational models", "Item-choice probability TVD\n(canonical vs shuffled list)", 1, .05),
                ("LLM", "LLM-based models", "Replica minus shuffled\nagreement (pp)", 100, 5)]
    figure, axes = plt.subplots(1, 2, squeeze=False, sharey=True, figsize=(TEXT_WIDTH, 2.35 + .5))
    figure._analysis_spec = {"kind": "association", "facet": "predictor_family", "x": "baseline_order_sensitivity",
                             "y": "head_click_reduction_pp", "main_text": True}
    upper = nice_limits(data.head_click_reduction_pp, step=10, pad=.04)[1]
    for axis, (prefix, title, x_label, scale, step) in zip(axes[0], families):
        part = data.loc[data.predictor_family.astype(str).str.startswith(prefix)]
        x = part.baseline_order_sensitivity * scale
        for _, row in part.iterrows():
            axis.scatter(row.baseline_order_sensitivity * scale, row.head_click_reduction_pp, s=24, zorder=3,
                         color=MODEL_COLOURS.get(row.model, NEUTRAL), marker=SEED_MARKERS.get(int(row.seed), "o"),
                         edgecolors="white", linewidths=.4)
        axis.set_xlim(step * np.floor((x.min() - step / 2) / step), step * np.ceil((x.max() + step / 2) / step))
        axis.set_ylim(0, upper)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=TICK_STEPS))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=TICK_STEPS))
        _style(axis, grid="both")
    axes[0, 0].set_ylabel("Reduction in head-click\nshare (pp)")
    figure.plotted = data
    return _finish(figure, [_model_handles(_ordered_models(data.model)), _seed_handles(sorted(data.seed.unique()))])


def l4_l5_effect_table(sampling_effects, slot_effects, *, window="mean_1_30"):
    """Seed effects and their mean for the L4 sampling and advertising/L5 contrasts (pp).

    Sampling contrasts use all displayed ranks 1--10 (no sponsored item exists in
    these conditions). Advertising contrasts use displayed ranks 2--10 in every
    condition, so the sponsored first position never enters the measurement.
    """
    parts = [
        (sampling_effects, SAMPLING_CONTRASTS, {"head_exposure": "Head exposure",
                                               "weighted_head_exposure": "Weighted head exposure"}, "Ranks 1–10"),
        (slot_effects, ADVERTISING_CONTRASTS, {"head_exposure_ranks_2_10": "Head exposure",
                                              "weighted_head_exposure_ranks_2_10": "Weighted head exposure"}, "Ranks 2–10"),
    ]
    rows = []
    for frame, contrasts, metrics, scope in parts:
        selected = frame.loc[frame.window.eq(window) & frame.comparison.isin(contrasts) & frame.metric.isin(metrics)]
        if selected.duplicated(["comparison", "metric", "seed"]).any():
            raise ValueError("One effect per contrast, metric, and seed is required")
        if selected.effect_unit.ne("percentage points").any():
            raise ValueError("The L4/L5 table expects percentage-point effects")
        for contrast, contrast_label in contrasts.items():
            for metric, metric_label in metrics.items():
                block = selected.loc[selected.comparison.eq(contrast) & selected.metric.eq(metric)]
                values = {int(seed): value for seed, value in zip(block.seed, block.value) if pd.notna(value)}
                row = {"Contrast": contrast_label, "Exposure scope": scope, "Metric": metric_label}
                row.update({f"Seed {seed}": values.get(seed, np.nan) for seed in (42, 43, 44)})
                row["Mean of seeds"] = float(np.mean(list(values.values()))) if values else np.nan
                row["Seeds"] = len(values)
                rows.append(row)
    return pd.DataFrame(rows)


def agent_module_table(usage, agent_diagnostics):
    """Module use, reflection, and applied profile changes per RQ III configuration and seed.

    ``usage`` is the output of ``load_agent_usage`` for the 300-active-user
    L2/L3 agent runs (only the canonical arm is used); ``agent_diagnostics`` the
    matching labelled rows of ``results['agent']``. Counts are sums over
    iterations 1--30; changed users are distinct users up to iteration 30.
    """
    from .presentation import label_conditions
    from .sections_rq1_3 import horizon_counts
    canonical = label_conditions(usage.loc[usage.arm.eq("canonical")])
    if canonical.source_id.duplicated().any():
        raise ValueError("Usage must contain one canonical row per physical run")
    counts = canonical.groupby(["condition_label", "seed"], as_index=False)[
        ["decisions", "memory_calls", "forced_actions"]].sum(min_count=1)
    reflection = horizon_counts(agent_diagnostics, ["reflection_requested", "reflection_failed", "profile_changes"])
    final = agent_diagnostics.loc[agent_diagnostics.iteration.eq(30),
                                  ["condition_label", "seed", "cumulative_profile_changed_users"]]
    if final.duplicated(["condition_label", "seed"]).any():
        raise ValueError("One iteration-30 agent row per configuration and seed is required")
    keys = {"condition_label": "Condition", "seed": "Seed"}
    table = (counts.rename(columns=keys)
             .merge(reflection.drop(columns=["Model"]), on=["Condition", "Seed"], how="outer", validate="one_to_one")
             .merge(final.rename(columns=keys), on=["Condition", "Seed"], how="outer", validate="one_to_one"))
    order = {name: index for index, name in enumerate(CONFIGURATIONS)}
    table = table.assign(_order=table.Condition.map(order)).sort_values(["_order", "Seed"]).drop(columns="_order")
    table["Condition"] = table.Condition.map(CONFIGURATIONS).fillna(table.Condition)
    return table.rename(columns={
        "Condition": "Configuration", "decisions": "Decisions", "memory_calls": "Memory-module calls",
        "forced_actions": "Forced actions", "reflection_requested": "Reflection requests",
        "reflection_failed": "Failed reflections", "profile_changes": "Applied profile changes",
        "cumulative_profile_changed_users": "Users with a change by iteration 30"}).reset_index(drop=True)


def export_main_figure(report, figure, name, caption):
    """Save PDF, PNG, TeX, and JSON like ``NotebookReport.figure``, without the context header."""
    plotted = getattr(figure, "plotted", pd.DataFrame())
    figure._analysis_context = describe_figure(plotted, getattr(figure, "_analysis_spec", {}))
    files = save_figure(figure, analysis_name=name, output_dir=report.exports / "figures", caption=caption,
                        provenance=report.provenance(plotted), recommended=True,
                        rationale="Main-text version: one time summary, print width, no in-figure title.")
    display(figure)
    plt.close(figure)
    display(Markdown(f"[PDF]({files['pdf'].as_posix()}) · label `{files['label']}`"))
    return name


def copy_to_thesis(report, names, repo, *, target="writing/latex/chapters/results_figures"):
    """Copy exported main-text PDFs next to the LaTeX chapters (adds or replaces files in ``target`` only)."""
    repo = Path(repo).resolve()
    destination = (repo / target).resolve()
    if not destination.is_relative_to(repo):
        raise ValueError("The LaTeX figure folder must lie inside the repository")
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in names:
        source = report.exports / "figures" / f"{name}.pdf"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / source.name)
        copied.append(destination / source.name)
    display(Markdown(f"Copied {len(copied)} PDFs to `{destination}`."))
    return copied
