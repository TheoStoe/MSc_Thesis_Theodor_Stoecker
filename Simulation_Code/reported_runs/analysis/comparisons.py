"""Matched, descriptive RQ comparisons of saved per-iteration measurements.

The independent campaign block is a simulation seed together with its frozen
population. Iterations, users, and reused-control aliases are never treated as
additional replications. No significance tests or simulation calls occur here.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd


RUN_KEYS = ["dataset", "study", "seed", "population_seed", "condition_id"]
RUN_METADATA = RUN_KEYS + ["model", "stage", "intervention", "user_update_policy",
                           "run_id", "physical_run_id"]
SHARE_METRICS = {
    "head_click_share", "head_exposure", "weighted_head_exposure",
    "head_exposure_ranks_2_10", "weighted_head_exposure_ranks_2_10",
    "head_exposure_full_minus_ranks_2_10", "weighted_head_exposure_full_minus_ranks_2_10",
    "click_exposure_gap", "weighted_click_exposure_gap", "click_rate",
    "no_click_rate", "invalid_rate", "exposure_coverage", "click_coverage",
    "cumulative_exposure_coverage", "cumulative_click_coverage",
    "female_click_share", "female_exposure", "weighted_female_exposure",
    "female_among_binary_clicks", "female_among_binary_exposure",
    "weighted_female_among_binary_exposure", "female_click_exposure_gap",
    "weighted_female_click_exposure_gap", "female_binary_click_exposure_gap",
    "weighted_female_binary_click_exposure_gap", "female_candidate_share",
    "female_binary_candidate_share", "female_exposure_candidate_gap",
    "binary_female_click_exposure_gap", "weighted_binary_female_click_exposure_gap",
    "eligible_female_share", "eligible_male_share", "eligible_head_share",
    "eligible_unknown_share", "eligible_ambiguous_share",
    "female_exposure_minus_candidate_share", "weighted_female_exposure_minus_candidate_share",
    "male_exposure", "weighted_male_exposure", "male_click_share",
    "unknown_exposure", "weighted_unknown_exposure", "unknown_click_share",
    "ambiguous_exposure", "weighted_ambiguous_exposure", "ambiguous_click_share",
    "replica_agreement", "shuffled_agreement", "agreement_sensitivity",
    "replica_action_agreement", "shuffled_action_agreement",
    "replica_pair_agreement", "shuffled_pair_agreement",
    "shuffle_tvd", "replica_tvd", "first_position_click_share",
}
CORE_METRICS = ["head_click_share", "head_exposure", "weighted_head_exposure",
                "click_exposure_gap", "weighted_click_exposure_gap", "click_rate",
                "no_click_rate", "invalid_rate", "valid_clicks",
                "exposure_coverage", "click_coverage", "exposure_gini", "click_gini"]


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept the central inventory names as well as the metric-frame names."""
    renames = {"simulation_seed": "seed", "logical_condition_id": "condition_id",
               "feedback_stage": "stage"}
    out = frame.rename(columns={a: b for a, b in renames.items() if b not in frame}).copy()
    if "dataset" not in out and "study" in out:
        out["dataset"] = out["study"].str.split("_").str[0]
    if "intervention" not in out:
        out["intervention"] = "control"
    out["intervention"] = out["intervention"].fillna("control").replace("", "control")
    if "user_update_policy" not in out:
        out["user_update_policy"] = "frozen"
    return out


def _identity_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in RUN_METADATA if column in frame]


def summarise_iterations(frame: pd.DataFrame, metrics: list[str] | None = None,
                         *, id_columns: list[str] | None = None) -> pd.DataFrame:
    """Strict mean of iterations 1--30 and iteration-30 value for every run.

    A partial trajectory or any undefined metric value makes its primary mean
    undefined. ``available_mean`` is retained, explicitly labelled, for auditing.
    Metrics remain in their original units; ``paired_effects`` converts shares
    to percentage-point effects. Input frames are not changed.
    """
    if frame.empty:
        return pd.DataFrame(columns=(id_columns or RUN_KEYS) +
                            ["metric", "window", "value", "n_iterations", "n_defined", "complete_horizon"])
    frame = _normalise(frame)
    ids = id_columns or _identity_columns(frame)
    metrics = metrics or [m for m in CORE_METRICS if m in frame]
    missing = set(ids + ["iteration"] + metrics) - set(frame)
    if missing:
        raise ValueError(f"Missing summary columns: {sorted(missing)}")
    if frame.duplicated(ids + ["iteration"]).any():
        raise ValueError("A run has duplicate per-iteration measurements.")
    rows = []
    for key, block in frame.groupby(ids, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(ids, key))
        block = block.loc[block.iteration.between(1, 30)].sort_values("iteration")
        horizon = block.iteration.tolist() == list(range(1, 31))
        for metric in metrics:
            values = pd.to_numeric(block[metric], errors="raise")
            if np.isinf(values.to_numpy(dtype=float)).any():
                raise ValueError(f"Infinite metric value: {metric}")
            n_defined = int(values.notna().sum())
            available_mean = values.mean() if n_defined else np.nan
            endpoint = block.loc[block.iteration.eq(30), metric]
            for window, value in [
                ("mean_1_30", available_mean if horizon and n_defined == 30 else np.nan),
                ("iteration_30", endpoint.iloc[0] if len(endpoint) else np.nan),
            ]:
                rows.append({**identity, "metric": metric, "window": window,
                             "value": value, "available_mean": available_mean,
                             "n_iterations": len(block), "n_defined": n_defined,
                             "complete_horizon": horizon})
    return pd.DataFrame(rows)


def build_comparison_plan(sources: pd.DataFrame) -> pd.DataFrame:
    """Resolve exact within-study controls without crossing active/reserve designs.

    Includes incomplete source pairs with ``eligible=False`` so omissions remain
    visible. A missing control is recorded rather than substituted from another
    population, stage, or study. Physical aliases are logical conditions, not
    independent replicates.
    """
    sources = _normalise(sources)
    if sources.empty:
        return pd.DataFrame()
    required = set(RUN_KEYS + ["model", "stage"])
    if required - set(sources):
        raise ValueError(f"Missing source metadata: {sorted(required - set(sources))}")
    if sources.duplicated(RUN_KEYS).any():
        raise ValueError("Duplicate logical source identities; resolve the registry first.")
    result = []
    block_columns = ["dataset", "study", "seed", "population_seed"]
    for block_key, block in sources.groupby(block_columns, dropna=False, sort=False):
        identity = dict(zip(block_columns, block_key))
        study = str(identity["study"])

        def choose(**fields):
            mask = pd.Series(True, index=block.index)
            for field, value in fields.items():
                if field == "stage":
                    mask &= block.stage.astype(str).str.split("_").str[0].eq(value)
                else:
                    mask &= block[field].eq(value)
            found = block.loc[mask]
            if len(found) > 1:
                raise ValueError(f"Ambiguous control in {identity}: {fields}")
            return found.iloc[0] if len(found) else None

        def add(treatment, control, comparison, kind, rqs):
            if treatment is None:
                return
            t_complete = str(treatment.get("status", "complete")) == "complete"
            c_complete = control is not None and str(control.get("status", "complete")) == "complete"
            result.append({**identity, "comparison": comparison, "kind": kind, "rqs": rqs,
                           "model": treatment["model"], "stage": treatment["stage"],
                           "intervention": treatment["intervention"],
                           "treatment_condition": treatment["condition_id"],
                           "control_condition": control["condition_id"] if control is not None else None,
                           "treatment_physical_run_id": treatment.get("physical_run_id", treatment.get("run_id")),
                           "control_physical_run_id": control.get("physical_run_id", control.get("run_id")) if control is not None else None,
                           "eligible": bool(t_complete and c_complete),
                           "pair_status": "complete" if t_complete and c_complete else
                           "missing_control" if control is None else "incomplete_pair"})

        if "sampling_outcome" in study:
            reference = choose(stage="L3")
            for _, treatment in block.loc[block.condition_id.str.startswith("L4_")].iterrows():
                suffix = str(treatment.condition_id)
                variant = "no refill" if "exit_only" in suffix else "uniform refill" if "uniform_refill" in suffix else "cluster refill"
                add(treatment, reference, f"L4 -- {variant} minus matched L3", "sampling", "I")
            organic = choose(condition_id="O0_organic")
            static = choose(condition_id="O1_static_ad")
            adaptive = choose(condition_id="O2_adaptive_ad_three_strike")
            add(static, organic, "Static advertising minus matched organic", "advertising", "I")
            add(adaptive, static, "L5 adaptive advertising minus static advertising", "advertising", "I")
        elif "mitigation" in study:
            for (model, stage), group in block.groupby(["model", "stage"], dropna=False):
                controls = group.loc[group.intervention.eq("control")]
                if len(controls) > 1:
                    raise ValueError(f"Ambiguous mitigation control: {identity}, {model}, {stage}")
                control = controls.iloc[0] if len(controls) else None
                for intervention in ["in_processing", "post_processing"]:
                    treatments = group.loc[group.intervention.eq(intervention)]
                    if len(treatments) > 1:
                        raise ValueError("Duplicate mitigation treatment in one model/stage.")
                    if len(treatments):
                        add(treatments.iloc[0], control,
                            f"{stage.split('_')[0]} {intervention.replace('_', '-')} minus control",
                            "mitigation", "IV,V" if identity["dataset"] == "movielens" else "VII")
                ins = group.loc[group.intervention.eq("in_processing")]
                posts = group.loc[group.intervention.eq("post_processing")]
                if len(ins):
                    add(ins.iloc[0], posts.iloc[0] if len(posts) else None,
                        f"{stage.split('_')[0]} in-processing minus post-processing", "mitigation_method",
                        "IV" if identity["dataset"] == "movielens" else "VII")
        elif "feedback" in study or study.startswith("goodreads_gender_"):
            for model in block.model.drop_duplicates():
                frozen = {s: choose(model=model, stage=s, user_update_policy="frozen")
                          for s in ["L0", "L1", "L2"]}
                for t, c in [("L1", "L0"), ("L2", "L1")]:
                    if frozen[t] is not None and (frozen[c] is not None or identity["dataset"] == "movielens"):
                        add(frozen[t], frozen[c], f"{t} minus {c}", "feedback", "I,II" if identity["dataset"] == "movielens" else "VI,VII")
                if frozen["L2"] is not None and (frozen["L0"] is not None or identity["dataset"] == "movielens"):
                    add(frozen["L2"], frozen["L0"], "L2 minus L0", "transfer", "VI,VII")
                if model == "agentic_recommender":
                    none = choose(model=model, stage="L3", user_update_policy="no_reflection_control")
                    fixed = choose(model=model, stage="L3", user_update_policy="memory_only_reflection")
                    full = choose(model=model, stage="L3", user_update_policy="full_reflection")
                    add(full, frozen["L2"], "L3 full reflection minus L2 frozen agent", "individual", "I,III,VI,VII")
                    if identity["dataset"] == "movielens":
                        add(none, frozen["L2"], "Private memory: no reflection minus frozen agent", "ablation", "III")
                        add(fixed, none, "Reflection notes: fixed profile minus no reflection", "ablation", "III")
                        add(full, fixed, "Profile updates: full reflection minus fixed profile", "ablation", "III")
    return pd.DataFrame(result)


def paired_effects(trajectory: pd.DataFrame, plan: pd.DataFrame,
                   metrics: list[str] | None = None) -> pd.DataFrame:
    """Signed treatment-minus-control effects, pairing exact iterations first."""
    trajectory = _normalise(trajectory)
    metrics = metrics or [m for m in CORE_METRICS if m in trajectory]
    if not set(metrics).issubset(trajectory):
        raise ValueError("Requested comparison metrics are absent from the trajectory frame.")
    if trajectory.duplicated(RUN_KEYS + ["iteration"]).any():
        raise ValueError("Duplicate run/iteration rows would multiply matched comparisons.")
    rows = []
    for pair in plan.to_dict("records"):
        if not pair["eligible"]:
            continue
        mask = pd.Series(True, index=trajectory.index)
        for column in ["dataset", "study", "seed", "population_seed"]:
            mask &= trajectory[column].eq(pair[column])
        group = trajectory.loc[mask]
        treatment = group.loc[group.condition_id.eq(pair["treatment_condition"]), ["iteration"] + metrics]
        control = group.loc[group.condition_id.eq(pair["control_condition"]), ["iteration"] + metrics]
        merged = treatment.merge(control, on="iteration", suffixes=("_t", "_c"), validate="one_to_one", how="outer")
        for metric in metrics:
            scale = 100.0 if metric in SHARE_METRICS else 1.0
            for row in merged.to_dict("records"):
                rows.append({**pair, "iteration": row["iteration"], "metric": metric,
                             "treatment_value": row[f"{metric}_t"], "control_value": row[f"{metric}_c"],
                             "value": (row[f"{metric}_t"] - row[f"{metric}_c"]) * scale,
                             "effect_unit": "percentage points" if scale == 100 else "metric units"})
    return pd.DataFrame(rows)


def summarise_effects(effects: pd.DataFrame) -> pd.DataFrame:
    """Summarise paired trajectories rather than subtract unmatched time windows."""
    if effects.empty:
        return pd.DataFrame()
    wide = effects.rename(columns={"value": "effect", "metric": "source_metric"})
    ids = [c for c in wide if c not in {"iteration", "treatment_value", "control_value", "effect"}]
    result = summarise_iterations(wide, ["effect"], id_columns=ids)
    return result.drop(columns="metric").rename(columns={"source_metric": "metric"})


def seed_consistency(summary: pd.DataFrame, *, expected_seeds=(42, 43, 44),
                     group_columns: list[str] | None = None) -> pd.DataFrame:
    """Descriptive seed/population consistency, retaining n and missing blocks."""
    if summary.empty:
        return pd.DataFrame()
    columns = group_columns or [c for c in ["dataset", "study", "comparison", "kind", "model", "stage", "intervention", "metric", "window", "effect_unit"] if c in summary]
    rows = []
    for key, group in summary.groupby(columns, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        if group.duplicated(["seed", "population_seed"]).any():
            raise ValueError("More than one result per seed/population block; specify finer groups.")
        defined = group.loc[group.value.notna()]
        values = defined.value.to_numpy(dtype=float)
        seen = sorted(int(s) for s in defined.seed)
        rows.append({**dict(zip(columns, key)), "n_seed_blocks": len(defined),
                     "seed_blocks": "; ".join(f"{int(r.seed)}/{int(r.population_seed)}" for r in defined.itertuples()),
                     "missing_seeds": ", ".join(str(s) for s in expected_seeds if s not in seen),
                     "all_expected_seeds": set(seen) == set(expected_seeds),
                     "mean_effect": float(values.mean()) if len(values) else np.nan,
                     "min_effect": float(values.min()) if len(values) else np.nan,
                     "max_effect": float(values.max()) if len(values) else np.nan,
                     "positive": int((values > 0).sum()), "negative": int((values < 0).sum()),
                     "zero": int((values == 0).sum()),
                     "same_nonzero_sign": bool(len(values) and ((values > 0).all() or (values < 0).all()))})
    return pd.DataFrame(rows)


def model_differences(trajectory: pd.DataFrame, metric="head_click_share") -> pd.DataFrame:
    """RQ II/IV model differences: absolute means and mean absolute trajectories.

    Each pair remains within a dataset, study, seed/population, feedback stage,
    and intervention. Reflective-agent L3 is not silently compared with L2 here.
    """
    frame = _normalise(trajectory)
    group_cols = ["dataset", "study", "seed", "population_seed", "stage", "intervention"]
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=False):
        for a, b in combinations(sorted(group.model.unique()), 2):
            left = group.loc[group.model.eq(a), ["iteration", metric]]
            right = group.loc[group.model.eq(b), ["iteration", metric]]
            if left.duplicated("iteration").any() or right.duplicated("iteration").any():
                continue  # L3 ablations are compared by explicit configuration instead.
            paired = left.merge(right, on="iteration", suffixes=("_a", "_b"), validate="one_to_one")
            if sorted(paired.iteration) != list(range(1, 31)) or paired.isna().any().any():
                continue
            difference = paired[f"{metric}_a"] - paired[f"{metric}_b"]
            scale = 100 if metric in SHARE_METRICS else 1
            rows.append({**dict(zip(group_cols, key)), "model_a": a, "model_b": b, "metric": metric,
                         "signed_difference_of_means": float(difference.mean() * scale),
                         "absolute_difference_of_means": float(abs(difference.mean()) * scale),
                         "mean_absolute_iteration_difference": float(difference.abs().mean() * scale),
                         "effect_unit": "percentage points" if scale == 100 else "metric units"})
    return pd.DataFrame(rows)


def mitigation_magnitude_comparison(trajectory: pd.DataFrame,
                                     metric="head_click_share") -> pd.DataFrame:
    """RQ IV pairwise model-configuration versus own-mitigation magnitudes.

    The agent's L3 is deliberately retained, with an explicit protocol warning.
    This comparison is broader than ``model_differences``, which compares only
    equal stages. Each row retains its model pair, focal model, and intervention.
    """
    frame = _normalise(trajectory)
    frame = frame.loc[frame.study.str.contains("mitigation") & frame.dataset.eq("movielens")]
    rows = []
    blocks = ["dataset", "study", "seed", "population_seed"]
    for key, group in frame.groupby(blocks, dropna=False, sort=False):
        controls = group.loc[group.intervention.eq("control")]
        models = sorted(controls.model.unique())
        for a, b in combinations(models, 2):
            pair = {}
            for model in [a, b]:
                run = controls.loc[controls.model.eq(model)]
                if run.condition_id.nunique() != 1 or run.duplicated("iteration").any():
                    raise ValueError("Magnitude comparison needs one control per model configuration.")
                pair[model] = run[["iteration", metric, "stage"]]
            paired = pair[a].merge(pair[b], on="iteration", suffixes=("_a", "_b"), validate="one_to_one")
            complete = sorted(paired.iteration) == list(range(1, 31)) and not paired[[f"{metric}_a", f"{metric}_b"]].isna().any().any()
            if not complete:
                continue
            model_difference = paired[f"{metric}_a"] - paired[f"{metric}_b"]
            scale = 100 if metric in SHARE_METRICS else 1
            for focal in [a, b]:
                for intervention in ["in_processing", "post_processing"]:
                    treatment = group.loc[group.model.eq(focal) & group.intervention.eq(intervention)]
                    if treatment.empty:
                        continue
                    if treatment.condition_id.nunique() != 1 or treatment.duplicated("iteration").any():
                        raise ValueError("Magnitude comparison has ambiguous mitigation treatment.")
                    paired_t = treatment[["iteration", metric]].merge(pair[focal][["iteration", metric]], on="iteration", suffixes=("_t", "_c"), validate="one_to_one")
                    complete_t = sorted(paired_t.iteration) == list(range(1, 31)) and not paired_t.isna().any().any()
                    if not complete_t:
                        continue
                    mitigation_difference = paired_t[f"{metric}_t"] - paired_t[f"{metric}_c"]
                    stages = controls.loc[controls.model.isin([a, b]), "stage"].unique()
                    rows.append({**dict(zip(blocks, key)), "model_a": a, "model_b": b,
                                 "focal_model": focal, "intervention": intervention, "metric": metric,
                                 "model_absolute_difference_of_means": abs(model_difference.mean()) * scale,
                                 "model_mean_absolute_iteration_difference": model_difference.abs().mean() * scale,
                                 "mitigation_signed_difference_of_means": mitigation_difference.mean() * scale,
                                 "mitigation_absolute_difference_of_means": abs(mitigation_difference.mean()) * scale,
                                 "mitigation_mean_absolute_iteration_difference": mitigation_difference.abs().mean() * scale,
                                 "model_difference_larger": abs(model_difference.mean()) > abs(mitigation_difference.mean()),
                                 "same_feedback_stage": len(stages) == 1,
                                 "comparison_scope": "Same-stage response models" if len(stages) == 1 else "Model configurations: reflective-agent L3 versus other-model L2",
                                 "effect_unit": "percentage points" if scale == 100 else "metric units"})
    return pd.DataFrame(rows)


def order_sensitivity_association(trajectory: pd.DataFrame, probes: pd.DataFrame,
                                 sources: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RQ V pairs and within-seed descriptive Spearman associations.

    LLM predictor: replica minus shuffled same-item agreement on the common
    all-valid-click triple subset. Computational predictor: exact saved
    probability TVD. These are separate predictor families, not a pooled scale.
    Returns input pairs and correlations (no p-values). No probabilities are
    inferred from observed choices when the saved TVD is unavailable.
    """
    trajectory, probes, sources = map(_normalise, [trajectory, probes, sources])
    source_mask = sources.study.str.contains("movielens_mitigation") & sources.intervention.eq("control")
    controls = sources.loc[source_mask]
    source_ids = controls[RUN_KEYS].drop_duplicates()
    baseline = probes.merge(source_ids, on=RUN_KEYS, how="inner", validate="many_to_one")
    available = [c for c in ["agreement_sensitivity", "shuffle_tvd"] if c in baseline]
    predictor = summarise_iterations(baseline, available) if available else pd.DataFrame()
    if predictor.empty:
        return pd.DataFrame(), pd.DataFrame()
    predictor = predictor.loc[predictor.window.eq("mean_1_30")].copy()
    plan = build_comparison_plan(sources)
    plan = plan.loc[plan.kind.eq("mitigation") & plan.intervention.eq("post_processing") & plan.dataset.eq("movielens")]
    outcomes = summarise_effects(paired_effects(trajectory, plan, ["head_click_share"]))
    if outcomes.empty:
        return pd.DataFrame(), pd.DataFrame()
    pairs = []
    for row in outcomes.to_dict("records"):
        is_llm = row["model"] in {"direct_llm", "agentic_recommender"}
        metric = "agreement_sensitivity" if is_llm else "shuffle_tvd"
        matching = predictor.loc[predictor.metric.eq(metric) & predictor.seed.eq(row["seed"]) &
                                 predictor.population_seed.eq(row["population_seed"]) &
                                 predictor.study.eq(row["study"]) &
                                 predictor.condition_id.eq(row["control_condition"])]
        value = matching.iloc[0].value if len(matching) == 1 else np.nan
        pairs.append({"dataset": row["dataset"], "study": row["study"], "seed": row["seed"],
                      "population_seed": row["population_seed"], "model": row["model"], "stage": row["stage"],
                      "window": row["window"], "predictor_metric": metric,
                      "predictor_family": "LLM common-valid agreement gap" if is_llm else "Computational probability TVD",
                      "baseline_order_sensitivity": value,
                      "head_click_reduction_pp": -row["value"],
                      "available": bool(pd.notna(value) and pd.notna(row["value"]))})
    pair_frame = pd.DataFrame(pairs)
    associations = []
    grouping = ["dataset", "study", "seed", "population_seed", "window", "predictor_family"]
    for key, group in pair_frame.groupby(grouping, dropna=False, sort=False):
        valid = group.loc[group.available]
        # Pearson correlation of ranks is Spearman's coefficient; no asymptotic test.
        rho = valid.baseline_order_sensitivity.rank().corr(valid.head_click_reduction_pp.rank()) if len(valid) >= 3 and valid.baseline_order_sensitivity.nunique() > 1 and valid.head_click_reduction_pp.nunique() > 1 else np.nan
        associations.append({**dict(zip(grouping, key)), "n_models": len(valid), "spearman_rho": rho,
                             "models": "; ".join(valid.model), "interpretation": "Descriptive only; separate predictor families, no p-value."})
    return pair_frame, pd.DataFrame(associations)


def qualitative_profile_sample(profile_changes: pd.DataFrame, *, n=10) -> pd.DataFrame:
    """Select deterministic, unassessed RQ III cases among applied profile changes.

    Only the main MovieLens full-reflection L3 is eligible; proposed or suppressed
    changes must be excluded by the metric engine. Never invent manual judgements.
    """
    frame = _normalise(profile_changes)
    if frame.empty:
        return pd.DataFrame(columns=RUN_KEYS + ["user_id", "review_order", "manual_assessment"])
    keep = frame.dataset.eq("movielens") & frame.study.str.contains("feedback_a300") & frame.stage.astype(str).str.startswith("L3") & frame.user_update_policy.eq("full_reflection")
    if "applied" in frame:
        keep &= frame.applied.eq(True)
    frame = frame.loc[keep]
    selected = []
    for key, block in frame.groupby(RUN_KEYS, sort=True, dropna=False):
        user_ids = sorted(block.user_id.dropna().unique(), key=lambda x: (str(type(x)), str(x)))
        count = min(n, len(user_ids))
        positions = np.random.default_rng(int(dict(zip(RUN_KEYS, key))["seed"])).choice(len(user_ids), size=count, replace=False)
        for order, position in enumerate(positions, 1):
            user = user_ids[int(position)]
            events = block.loc[block.user_id.eq(user)]
            selected.append({**dict(zip(RUN_KEYS, key)), "user_id": user, "review_order": order,
                             "eligible_users": len(user_ids), "requested_sample": n,
                             "applied_change_count": len(events),
                             "first_change_iteration": events.iteration.min(),
                             "last_change_iteration": events.iteration.max(),
                             "manual_assessment": "Not yet reviewed",
                             "extent_of_change": "", "retained_interests": "",
                             "supported_additions": "", "unsupported_changes": ""})
    return pd.DataFrame(selected)


def load_rq_plan(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or Path(__file__).with_name("analysis_config.json")).read_text(encoding="utf-8"))
