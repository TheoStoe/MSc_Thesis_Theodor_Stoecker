"""Small notebook-facing helpers; calculations live in metrics/comparisons."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
import matplotlib.pyplot as plt

from .tables import export_latex_table
from .plotting import save_figure, label, annotate_figure_context
from .figure_context import describe_figure


@dataclass
class Results:
    snapshot_dir: Path
    metadata: dict
    frames: dict[str, pd.DataFrame]

    def __getitem__(self, key):
        return self.frames[key]


def load_results(snapshot: str | Path | None = None) -> Results:
    """Load numeric cached frames, never run a simulator or call a provider."""
    analysis = Path(__file__).resolve().parent
    if snapshot is None:
        pointer = json.loads((analysis / "latest.json").read_text(encoding="utf-8"))
        snapshot = analysis / pointer["snapshot_directory"]
    directory = Path(snapshot)
    metadata = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
    frames = {}
    for name, info in metadata["dataframes"].items():
        path = directory / info["path"]
        if path.name.endswith((".pkl", ".pkl.gz")):
            # Only the local cache written by this offline pipeline is read.
            frames[name] = pd.read_pickle(path)
        else:
            try:
                frames[name] = pd.read_csv(path, low_memory=False)
            except pd.errors.EmptyDataError:
                frames[name] = pd.DataFrame()
    return Results(directory, metadata, frames)


def unique_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated logical memberships without pooling independent runs."""
    if "physical_run_id" not in frame:
        return frame.copy()
    columns = [c for c in ["physical_run_id", "iteration", "arm", "rank", "module", "cluster"] if c in frame]
    return frame.drop_duplicates(columns).copy() if columns else frame.copy()


def condition_label(row):
    stage = str(row.get("stage", "")).split("_")[0]
    condition = str(row.get("condition_id", ""))
    for token, text in [("O0_organic", "Organic advertising control"), ("O1_static", "Static advertising"),
                        ("O2_adaptive", "L5 adaptive advertising"), ("exit_only", "L4: departures, no refill"),
                        ("uniform_refill", "L4: uniform refill"), ("cluster_refill", "L4: cluster refill")]:
        if token in condition:
            return text
    if condition.startswith("L4_"):
        return "L4: cluster refill"
    policy = str(row.get("user_update_policy", "frozen"))
    suffix = {"no_reflection_control": "no reflection", "memory_only_reflection": "fixed profile",
              "fixed_persona_memory_only": "fixed profile", "full_reflection": "full reflection"}.get(policy, "frozen agent")
    if row.get("model") == "agentic_recommender":
        stage += " -- " + suffix
    intervention = str(row.get("intervention", "control"))
    if intervention not in ("control", "", "None", "nan"):
        stage += ": " + intervention.replace("_", "-")
    return stage


def label_conditions(frame):
    output = frame.copy()
    if len(output):
        output["condition_label"] = output.apply(condition_label, axis=1)
    return output


def temporal_table(summaries, metrics, *, group_columns=None):
    """Readable wide view of already calculated mean/endpoint summaries."""
    if summaries.empty:
        return pd.DataFrame(columns=["Model", "Condition", "Seed", "Time summary", *map(label, metrics)])
    frame = summaries.loc[summaries.metric.isin(metrics)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["Model", "Condition", "Seed", "Time summary", *map(label, metrics)])
    columns = group_columns or [c for c in ["dataset", "study", "model", "condition_id", "seed", "window"] if c in frame]
    if frame.duplicated(columns + ["metric"]).any():
        raise ValueError("Summary table grouping would merge different trajectories")
    table = frame.pivot(index=columns, columns="metric", values="value").reset_index()
    table.columns.name = None
    gap_metrics = [m for m in metrics if "gap" in m or "minus_candidate" in m]
    for metric in gap_metrics:
        if metric in table:
            table[metric] *= 100
    return table.rename(columns={m: label(m) + (" (pp)" if m in gap_metrics else "") for m in metrics})


def effect_table(summary, metric, *, average_seeds=False):
    """Keep seed effects visible; optional descriptive mean records block counts."""
    if summary.empty:
        return pd.DataFrame(columns=["Comparison", "Model", "Seed", "Time summary", "Effect", "Unit"])
    frame = summary.loc[summary.metric.eq(metric)].copy()
    if average_seeds:
        groups = [c for c in ["dataset", "study", "comparison", "model", "window", "effect_unit"] if c in frame]
        frame = frame.groupby(groups, dropna=False, as_index=False).agg(
            mean_effect=("value", "mean"), min_effect=("value", "min"), max_effect=("value", "max"), n_seeds=("value", "count"))
        return frame
    columns = [c for c in ["dataset", "comparison", "model", "seed", "population_seed", "window", "value", "effect_unit"] if c in frame]
    return frame[columns].reset_index(drop=True)


def rank_summary(frame):
    """Aggregate counts before plotting; divide by each arm's valid-click count."""
    if frame.empty:
        return frame.copy()
    arm = "presentation_arm" if "presentation_arm" in frame else "arm"
    identifiers = [c for c in ["dataset", "study", "condition_id", "model", "stage", "seed", "population_seed", arm, "rank"] if c in frame]
    count = next((c for c in ["clicks", "click_count", "valid_clicks"] if c in frame), None)
    if count is None:
        raise ValueError("Rank summary requires raw click counts, not averaged rank shares")
    output = frame.groupby(identifiers, dropna=False, as_index=False)[count].sum().rename(columns={count: "clicks"})
    denominator_keys = [c for c in identifiers if c != "rank"]
    denominator = output.groupby(denominator_keys, dropna=False).clicks.transform("sum")
    output["click_share"] = output.clicks.div(denominator.where(denominator.ne(0)))
    return output


def cross_seed_table(results, rq, metrics):
    """Show descriptive available-seed means and signs of the exact RQ effects."""
    from .comparisons import seed_consistency
    frame = results["effect_summaries"]
    if rq == "rq1":
        selected = frame.loc[frame.dataset.eq("movielens") & frame.kind.isin(["feedback", "individual", "sampling", "advertising"])]
    elif rq == "rq2":
        selected = frame.loc[frame.study.eq("movielens_feedback_a300_e804") & frame.kind.eq("feedback")]
    elif rq == "rq3":
        selected = frame.loc[frame.study.eq("movielens_feedback_a300_e804") & frame.kind.eq("ablation")]
    elif rq == "rq4":
        selected = frame.loc[frame.study.eq("movielens_mitigation_a300_e804") & frame.kind.isin(["mitigation", "mitigation_method"])]
    elif rq in {"rq6", "rq7"}:
        from .sections_rq6_7 import transfer_effects
        selected = transfer_effects(frame)
        if rq == "rq7":
            selected = pd.concat([selected.loc[selected.dataset.eq("goodreads")],
                frame.loc[frame.study.eq("goodreads_gender_mitigation_a300_e804") & frame.kind.isin(["mitigation", "mitigation_method"])]], ignore_index=True)
    else:
        raise ValueError("This RQ has no common cross-seed effect summary")
    selected = selected.loc[selected.metric.isin(metrics)]
    table = seed_consistency(selected)
    columns = [c for c in ["dataset", "comparison", "model", "metric", "window", "n_seed_blocks", "mean_effect", "min_effect", "max_effect", "positive", "negative", "zero", "missing_seeds", "effect_unit"] if c in table]
    return table[columns].reset_index(drop=True)


def table_view(frame):
    """Use mitigation terminology in visible tables and their matching exports.

    The frozen source schema retains ``intervention`` for backwards-compatible
    matching. A value of ``control`` there means no mitigation, independently of
    whether the run is the treatment or control in a feedback-stage comparison.
    """
    output = frame.copy()
    for original, renamed in (("intervention", "mitigation"), ("Intervention", "Mitigation")):
        if original in output:
            if renamed in output:
                raise ValueError("Ambiguous table: both intervention and mitigation columns are present")
            output = output.rename(columns={original: renamed})
            output[renamed] = output[renamed].replace({
                "control": "none", "in_processing": "in-processing", "post_processing": "post-processing"})
    return output


class NotebookReport:
    def __init__(self, results: Results, rq: str):
        self.results = results
        self.rq = rq
        self.exports = results.snapshot_dir / "exports"

    @property
    def scales(self):
        return self.results.metadata["plot_scales"]

    def provenance(self, frame=None):
        columns = [c for c in ["physical_run_id", "run_id", "condition_id", "seed", "population_seed"] if frame is not None and c in frame]
        source_rows = frame[columns].drop_duplicates().to_dict("records") if columns else []
        # NumPy scalar conversion via JSON roundtrip keeps provenance serialisable.
        source_rows = json.loads(json.dumps(source_rows, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
        return {"snapshot": self.results.metadata["snapshot_id"], "snapshot_sha256": self.results.metadata.get("source_inventory_sha256"),
                "metric_version": self.results.metadata["metric_version"], "draft_sha256": self.results.metadata["draft_sha256"],
                "rq": self.rq, "source_blocks": source_rows,
                "scope": "Completed trajectories only; seed/population blocks remain distinct; canonical decisions determine longitudinal outcomes."}

    def table(self, frame, name, caption, *, columns=None, source_frame=None, column_format=None):
        if columns is not None:
            frame = frame[[c for c in columns if c in frame]].copy()
        frame = table_view(frame).reset_index(drop=True)
        if not len(frame.columns):
            frame = pd.DataFrame(columns=["Availability"])
        output = export_latex_table(frame, analysis_name=name, caption=caption,
            output_dir=self.exports / "tables", provenance=self.provenance(frame if source_frame is None else source_frame),
            column_format=column_format)
        display(output.display_frame)
        display(Markdown(f"LaTeX label: `{output.label}` · [table]({output.path.as_posix()}) · [numeric CSV]({output.data_path.as_posix()})"))
        # Notebook display calls return None so an extra technical object is
        # never printed by IPython's final-expression display hook.

    def figure(self, figure, frame, name, caption, *, recommended=False, why=""):
        context = describe_figure(frame, getattr(figure, "_analysis_spec", {}))
        annotate_figure_context(figure, context)
        files = save_figure(figure, analysis_name=name, output_dir=self.exports / "figures", caption=caption,
                            provenance=self.provenance(frame), recommended=recommended, rationale=why)
        display(figure)
        plt.close(figure)
        display(Markdown(f"[PDF]({files['pdf'].as_posix()}) · [PNG]({files['png'].as_posix()}) · label `{files['label']}`"))

    def availability(self, sources):
        columns = [c for c in ["dataset", "study", "seed", "model", "stage", "intervention", "completion_status", "status", "completed", "reason"] if c in sources]
        return self.table(sources[columns], f"{self.rq}_source_availability", "")


def selection(frame, **equalities):
    """Optional convenience; ordinary dataframe filters work identically."""
    out = frame
    for column, values in equalities.items():
        out = out.loc[out[column].isin(values) if isinstance(values, (list, tuple, set)) else out[column].eq(values)]
    return out.copy()
