"""Describe the plotted source selection without calculating any metrics."""
from __future__ import annotations

import pandas as pd

MODEL_ORDER = ["position_biased", "ncm_inspired_gru_uniform", "ncm_inspired_gru_mansoury", "direct_llm", "agentic_recommender"]
SHORT_MODELS = dict(zip(MODEL_ORDER, ["Position-biased", "NCM P3", "NCM P4", "Direct LLM", "Agent"]))
DATASETS = {"movielens": "MovieLens", "goodreads": "Goodreads"}
POLICIES = {"full_reflection": "full reflection", "no_reflection_control": "no reflection",
            "memory_only_reflection": "fixed profile", "fixed_persona_memory_only": "fixed profile"}


def _values(frame, column):
    return sorted(frame[column].dropna().unique().tolist(), key=str) if column in frame else []


def _models(names):
    names = set(names)
    if names == set(MODEL_ORDER[:3]):
        return "Computational models"
    return "/".join(SHORT_MODELS.get(model, str(model)) for model in MODEL_ORDER if model in names)


def _by_model(frame, field, *, stages=False):
    """Group equal stage/seed scopes, retaining unequal model coverage explicitly."""
    groups = {}
    if "model" not in frame or field not in frame:
        return "", {}
    recorded = {}
    for model, part in frame.groupby("model", sort=False):
        values = _values(part, field)
        if stages:
            values = sorted(set(str(value).split("_")[0] for value in values))
            policies = _values(part.loc[part[field].astype(str).str.startswith("L3")], "user_update_policy")
            if len(policies) == 1 and policies[0] in POLICIES and "L3" in values:
                values[values.index("L3")] += " (" + POLICIES[policies[0]] + ")"
        else:
            values = sorted(int(value) for value in values)
        recorded[str(model)] = values
        groups.setdefault(tuple(values), []).append(model)
    if len(groups) == 1:
        scope = next(iter(groups))
        return ", ".join(map(str, scope)), recorded
    return "; ".join(f"{_models(models)}: {', '.join(map(str, values))}"
                     for values, models in groups.items()), recorded


def _condition_seeds(frame):
    """Do not imply that a completed control also supplies its missing treatment."""
    if not {"model", "seed"}.issubset(frame):
        return {}, {}
    condition = next((column for column in ["intervention", "condition_label", "condition_id"]
                      if column in frame and frame[column].nunique() > 1), None)
    if condition is None:
        return {}, {}
    scopes, exceptions = {}, {}
    names = {"control": "no mitigation", "in_processing": "in-processing", "post_processing": "post-processing"}
    for model, part in frame.groupby("model", sort=False):
        coverage = {names.get(str(key), str(key)): sorted(int(v) for v in _values(group, "seed"))
                    for key, group in part.groupby(condition, sort=False)}
        scopes[str(model)] = coverage
        if len({tuple(values) for values in coverage.values()}) > 1:
            exceptions[str(model)] = coverage
    return scopes, exceptions


def describe_figure(frame, spec):
    """Return both readable scope and auditable structured plot metadata."""
    datasets = [DATASETS.get(str(value), str(value)) for value in _values(frame, "dataset")]
    stage_scope, stages = _by_model(frame, "stage", stages=True)
    seed_scope, seeds = _by_model(frame, "seed")
    if not seed_scope:
        seed_scope = ", ".join(map(str, _values(frame, "seed")))
    condition_seeds, seed_exceptions = _condition_seeds(frame)
    if seed_exceptions:
        remaining = frame.loc[~frame.model.isin(seed_exceptions)]
        ordinary, _ = _by_model(remaining, "seed")
        scopes = [f"{_models(remaining.model.unique())}: {ordinary}"] if ordinary else []
        for model, coverage in seed_exceptions.items():
            scopes.append(SHORT_MODELS.get(model, model) + " [" + "; ".join(
                f"{condition}: {', '.join(map(str, values))}" for condition, values in coverage.items()) + "]")
        seed_scope = "; ".join(scopes)
    windows = _values(frame, "window")
    kind = spec.get("kind", "unspecified")
    if kind == "rank":
        time = "Click counts pooled over iterations 1–30"
    elif kind == "composition":
        time = "Iteration snapshots: " + ", ".join(map(str, spec.get("snapshots", [])))
    elif kind == "association":
        outcome = "iteration 30" if windows == ["iteration_30"] else "mean over iterations 1–30" if windows == ["mean_1_30"] else "/".join(windows)
        time = f"Predictor: mean over iterations 1–30; outcome: {outcome}"
    elif windows == ["mean_1_30"]:
        time = "Mean over iterations 1–30"
    elif windows == ["iteration_30"]:
        time = "Iteration 30"
    elif set(windows) == {"mean_1_30", "iteration_30"}:
        time = "Mean over iterations 1–30 and iteration-30 endpoint (separate panels)"
    elif "iteration" in frame and len(frame):
        time = f"Per-iteration values, iterations {int(frame.iteration.min())}–{int(frame.iteration.max())}"
    else:
        time = "No completed time scope" if frame.empty else "Time scope unavailable"

    studies = _values(frame, "study")
    protocol = "Initially 150 active + 150 reserve users" if studies and all("a150_r150" in str(s) for s in studies) else "300 active users" if studies and all("a300" in str(s) for s in studies) else ""
    comparisons = _values(frame, "comparison")
    contrast = ""
    if kind == "effect":
        if _values(frame, "kind") == ["mitigation"]:
            contrast = "Each mitigation minus its matched unmitigated control"
        elif spec.get("category") not in {"comparison", "dataset_comparison"}:
            contrast = "Contrasts: " + "; ".join(map(str, comparisons)) if comparisons else ""
        if spec.get("contrast_description"):
            contrast = str(spec["contrast_description"])

    facet, series = spec.get("facet"), spec.get("series")
    if kind == "trajectory":
        if facet == "seed":
            encoding = "Panels: simulation seeds; " + ("colours/markers: models" if series == "model" else "line styles/markers: clusters" if series == "cluster" else "line styles/markers: conditions")
        else:
            encoding = "Panels: models; line styles: conditions; markers: simulation seeds"
    elif kind == "effect":
        encoding = "Colours: models; markers: simulation seeds"
    elif kind == "rank":
        encoding = "Panels: models; line styles: presentation arms; markers: simulation seeds"
    elif kind == "association":
        encoding = "Panels: simulation seeds; colours/markers: models"
    elif kind == "composition":
        encoding = "Rows: conditions; columns: simulation seeds; colours/hatches: clusters"
    else:
        encoding = ""
    stage_prefix = "Treatment stages: " if kind == "effect" else "Stages: "
    scope = " | ".join(value for value in [" + ".join(datasets), protocol, stage_prefix + stage_scope if stage_scope else ""] if value)
    time_seed = " | ".join(value for value in [time, "Simulation seeds: " + seed_scope if seed_scope else ""] if value)
    # Legends already label colour, marker, and line encodings. Keep the visible
    # scope strip short; retain the full encoding contract in the JSON metadata.
    lines = [line for line in [scope, time_seed, contrast] if line]
    return {"datasets": datasets, "stages_by_model": stages, "seeds_by_model": seeds,
            "seeds_by_model_condition": condition_seeds,
            "time_summary": time, "windows": windows, "protocol": protocol,
            "comparisons": comparisons, "encoding": encoding, "text": "\n".join(lines),
            "plot_spec": spec}
