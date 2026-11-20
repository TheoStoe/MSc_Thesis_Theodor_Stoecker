"""Offline appendix analysis of SASRec's 201/804-epoch sensitivity study.

The checked cache contains canonical head-click share and mean prior popularity
only. It deliberately excludes every legacy rank-discounted exposure column.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from .plotting import MODEL_COLOURS, MODEL_LABELS, NEUTRAL, save_figure
from .tables import export_latex_table

ROOT = Path(__file__).resolve().parent
MODELS = ("position_biased", "ncm_inspired_gru_uniform", "ncm_inspired_gru_mansoury")
SETTINGS = ("sample_201", "sample_804", "full_201")
SETTING_LABELS = {
    "sample_201": "300 users / 201 epochs",
    "sample_804": "300 users / 804 epochs",
    "full_201": "6,040 users / 201 epochs\n(context only)",
}
MITIGATIONS = ("control", "post_processing", "in_processing")
MITIGATION_LABELS = {
    "control": "Control", "post_processing": "Post-processing", "in_processing": "In-processing",
}
STYLES = {"control": ("-", "o"), "post_processing": ("--", "s"), "in_processing": (":", "^")}
METRICS = ("head_click_share", "mean_clicked_prior_popularity")


def load_epoch_budget(directory: str | Path | None = None) -> pd.DataFrame:
    """Read and validate the small, content-hashed numeric cache."""
    directory = ROOT / "derived_inputs" if directory is None else Path(directory)
    path = directory / "epoch_budget_sensitivity.csv"
    proof = json.loads((directory / "epoch_budget_provenance.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != proof["csv_sha256"]:
        raise ValueError("epoch-budget aggregate cache has changed")
    if proof["legacy_rank_weighted_columns_retained"] is not False:
        raise ValueError("legacy exposure measures must not enter this analysis")
    frame = pd.read_csv(path)
    required = {"setting", "model", "intervention", "seed", "iteration", *METRICS}
    if not required.issubset(frame.columns) or len(frame) != 810:
        raise ValueError("incomplete epoch-budget cache")
    if set(frame.seed) != {42}:
        raise ValueError("this sensitivity study contains only seed 42")
    keys = ["setting", "model", "intervention", "iteration"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicated epoch-budget observations")
    for (setting, model, mitigation), group in frame.groupby(keys[:-1]):
        if setting not in SETTINGS or model not in MODELS or mitigation not in MITIGATIONS:
            raise ValueError("unexpected epoch-budget condition")
        if len(group) != 30 or set(group.iteration) != set(range(1, 31)):
            raise ValueError("incomplete iteration horizon")
        if not np.allclose(group.head_click_share, group.head_clicks / group.valid_clicks):
            raise ValueError("head-click denominator mismatch")
    if len(frame.groupby(keys[:-1])) != 27:
        raise ValueError("missing epoch-budget condition")
    return frame.sort_values(keys).reset_index(drop=True)


def epoch_budget_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise each evolving system; never pool users as replications."""
    rows = []
    for (setting, model, mitigation), group in frame.groupby(["setting", "model", "intervention"]):
        group = group.sort_values("iteration")
        for window in ("mean_1_30", "iteration_30"):
            values = {
                metric: (group[metric].mean() if len(group) == 30 and group[metric].notna().all()
                    else np.nan) if window == "mean_1_30" else group.loc[group.iteration.eq(30), metric].iloc[0]
                for metric in METRICS
            }
            rows.append({"setting": SETTING_LABELS[setting].replace("\n", " "),
                "model": MODEL_LABELS[model], "mitigation": MITIGATION_LABELS[mitigation],
                "time_summary": window,
                "head_click_share_percent": values["head_click_share"] * 100,
                "mean_clicked_prior_popularity": values["mean_clicked_prior_popularity"]})
    return pd.DataFrame(rows)


def epoch_budget_plot(frame: pd.DataFrame, metric: str = "head_click_share"):
    """Use one scale for all nine comparable panels and fixed family colours."""
    if metric not in METRICS:
        raise ValueError("only head-click share and mean prior popularity are supported")
    sns.set_theme(style="whitegrid", context="notebook", font_scale=.95)
    rate = metric == "head_click_share"
    upper = 1 if rate else max(250, math.ceil(float(frame[metric].max()) / 250) * 250)
    figure, axes = plt.subplots(3, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    for row, setting in enumerate(SETTINGS):
        for column, model in enumerate(MODELS):
            axis = axes[row, column]
            for mitigation in MITIGATIONS:
                selected = frame.loc[frame.setting.eq(setting) & frame.model.eq(model) &
                    frame.intervention.eq(mitigation)].sort_values("iteration")
                line, marker = STYLES[mitigation]
                axis.plot(selected.iteration, selected[metric], color=MODEL_COLOURS[model],
                    linestyle=line, marker=marker, markevery=4, markersize=3.5, linewidth=1.5)
            axis.set(xlim=(1, 30), ylim=(0, upper), xticks=[1, 5, 10, 15, 20, 25, 30])
            if rate:
                axis.yaxis.set_major_formatter(PercentFormatter(1))
            if row == 0:
                axis.set_title(MODEL_LABELS[model])
            if column == 0:
                axis.set_ylabel(SETTING_LABELS[setting] + "\n" +
                    ("Head-click share" if rate else "Mean initial training count\nper clicked item"))
            if row == 2:
                axis.set_xlabel("Iteration")
    handles = [Line2D([], [], color=NEUTRAL, linestyle=STYLES[item][0],
        marker=STYLES[item][1], label=MITIGATION_LABELS[item]) for item in MITIGATIONS]
    figure.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(.5, .005))
    figure.tight_layout(rect=(0, .055, 1, 1))
    return figure


def export_epoch_budget(frame: pd.DataFrame, destination: str | Path | None = None) -> dict:
    """Create PDF/PNG/TeX figures and tables from the exact displayed values."""
    destination = ROOT / "exports" / "appendix_epoch_budget" if destination is None else Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((ROOT / "derived_inputs/epoch_budget_provenance.json").read_text(encoding="utf-8"))
    figures = {}
    for metric, stem in [
        ("head_click_share", "appendix_epoch_budget_head_click_share"),
        ("mean_clicked_prior_popularity", "appendix_epoch_budget_clicked_popularity"),
    ]:
        figure = epoch_budget_plot(frame, metric)
        figures[metric] = save_figure(figure, analysis_name=stem, output_dir=destination,
            caption="", provenance={"cache_sha256": provenance["csv_sha256"],
                "seed": 42, "canonical_only": True,
                "full_population_source": "existing audited aggregate CSV"},
            recommended=False, rationale="")
        # The public export must not embed a machine-specific absolute path.
        wrapper = destination / f"{stem}.tex"
        text = wrapper.read_text(encoding="utf-8")
        pdf = figures[metric]["pdf"]
        text = text.replace(pdf.resolve().as_posix(), pdf.name)
        wrapper.write_text(text, encoding="utf-8")
        plt.close(figure)
    summary = epoch_budget_summary(frame)
    summary.to_csv(destination / "appendix_epoch_budget_all_summaries.csv", index=False)
    tables = {}
    for window, name in (
        ("mean_1_30", "appendix_epoch_budget_summary"),
        ("iteration_30", "appendix_epoch_budget_endpoint"),
    ):
        selected = summary.loc[summary.time_summary.eq(window)].drop(columns="time_summary").reset_index(drop=True)
        selected["setting"] = selected["setting"].replace({
            "300 users / 201 epochs": "300 / 201",
            "300 users / 804 epochs": "300 / 804",
            "6,040 users / 201 epochs (context only)": "6,040 / 201",
        })
        tables[window] = export_latex_table(selected, analysis_name=name,
            caption="",
            output_dir=destination, column_labels={"setting": "Users / epochs", "model": "Model",
                "mitigation": "Mitigation", "head_click_share_percent": "Head-click share (%)",
                "mean_clicked_prior_popularity": "Prior popularity"},
            formatters={"head_click_share_percent": ".2f", "mean_clicked_prior_popularity": ".2f"},
            provenance={"cache_sha256": provenance["csv_sha256"], "seed": 42})
    return {"figures": figures, "tables": tables}
