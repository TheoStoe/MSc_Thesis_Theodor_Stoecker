"""Pooled identical-order replica agreement for complete physical LLM runs.

This is repeat-response agreement at the same saved user state and slate. It
does not measure complete-trajectory run-to-run differences or isolate a batch
effect. Agreement uses both-valid-click canonical/replica pairs, independently
of the shuffled response. Reused logical memberships are not extra runs.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


LLM_MODELS = ("direct_llm", "agentic_recommender")
AGREEMENT_COLUMN = "pooled_pair_replica_agreement"
REPLICA_COUNTS = (
    "replica_pairs", "replica_valid_pairs", "replica_excluded_non_valid_pairs",
    "replica_both_clicks", "replica_pair_same_item",
)
PHYSICAL_CONFIGURATION = (
    "dataset", "model", "stage", "intervention", "user_update_policy", "seed", "population_seed",
    "initial_active_users", "initial_reserve_users", "training_users", "iterations", "slate_size",
)
PHYSICAL_IDENTIFIERS = ("source_directory", "run_id", "physical_run_id", "source_condition_id")


def _constant(block, column, context):
    values = block[column].dropna().drop_duplicates()
    if len(values) > 1:
        raise ValueError(f"Conflicting physical {context} aliases for {column}")
    return values.iloc[0] if len(values) else np.nan


def _validate_counts(frame):
    for column in REPLICA_COUNTS:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not (np.isfinite(values) & (values >= 0) & (values == np.floor(values))).all():
            raise ValueError(f"Replica counts must be finite nonnegative integers: {column}")
        frame[column] = values
    if not frame.replica_valid_pairs.add(frame.replica_excluded_non_valid_pairs).eq(frame.replica_pairs).all():
        raise ValueError("Replica valid and excluded pairs do not sum to paired opportunities")
    if not (frame.replica_pair_same_item.le(frame.replica_both_clicks)
            & frame.replica_both_clicks.le(frame.replica_valid_pairs)
            & frame.replica_valid_pairs.le(frame.replica_pairs)).all():
        raise ValueError("Replica same-item/eligible/valid counts have inconsistent denominators")


def build_replica_agreement(probes, sources):
    """One row per complete physical Direct LLM or agent source, including N/A.

    Counts are pooled over iterations 1--30, then same-item count is divided by
    both-valid-click count. This is not an unweighted mean of per-iteration
    ratios and never uses the triple-click-conditioned ``replica_agreement``.
    Duplicate logical memberships are removed only after their count values
    and physical scientific configuration agree. Zero observed replica pairs
    in all 30 metric rows yield ``not_recorded``; recorded pairs without valid
    clicks yield ``no_eligible_clicks``. Missing metric rows instead yield
    ``incomplete_horizon``, and replicas present in only part of the horizon
    yield ``incomplete_replica_coverage``. Agreement is missing for either
    incomplete case; partial counts remain available only for auditing.
    """
    required_sources = {"source_id", "model", "dataset", "stage", "intervention", "user_update_policy", "seed", "population_seed"}
    if required_sources-set(sources):
        raise ValueError(f"Missing source columns: {sorted(required_sources-set(sources))}")
    selected = sources.loc[sources.model.isin(LLM_MODELS)].copy(deep=True)
    if "status" in selected:
        selected = selected.loc[selected.status.eq("complete")]
    if selected.source_id.isna().any():
        raise ValueError("Physical source IDs must be defined")
    relevant = probes.loc[probes.source_id.isin(selected.source_id)].copy(deep=True) if "source_id" in probes else pd.DataFrame()
    if not relevant.empty:
        required_probes = {"source_id", "iteration", *REPLICA_COUNTS}
        if required_probes-set(relevant):
            raise ValueError(f"Missing replica-count columns: {sorted(required_probes-set(relevant))}")
        iterations = pd.to_numeric(relevant.iteration, errors="raise")
        if not (np.isfinite(iterations) & iterations.eq(np.floor(iterations)) & iterations.between(1, 30)).all():
            raise ValueError("Replica iterations must be integers from 1 to 30")
        _validate_counts(relevant)
    rows = []
    for source_id, aliases in selected.groupby("source_id", sort=False):
        metadata = {column: _constant(aliases, column, "source") for column in
                    (*PHYSICAL_CONFIGURATION, *PHYSICAL_IDENTIFIERS) if column in aliases}
        if "physical_run_id" not in metadata and "run_id" in metadata:
            metadata["physical_run_id"] = metadata["run_id"]
        block = relevant.loc[relevant.source_id.eq(source_id)] if not relevant.empty else pd.DataFrame()
        for column in (*PHYSICAL_CONFIGURATION, "run_id", "physical_run_id"):
            if column in block:
                observed = _constant(block, column, "probe")
                if column in metadata and pd.notna(metadata[column]) and pd.notna(observed) and metadata[column] != observed:
                    raise ValueError(f"Probe/source physical configuration differs for {column}")
                if column not in metadata:
                    metadata[column] = observed
        if not block.empty:
            for iteration, duplicates in block.groupby("iteration", sort=False):
                if len(duplicates[list(REPLICA_COUNTS)].drop_duplicates()) != 1:
                    raise ValueError(f"Conflicting replica counts across aliases for source {source_id}, iteration {iteration}")
            block = block.drop_duplicates(["source_id", "iteration"]).sort_values("iteration")
        membership_columns = [c for c in ("study", "condition_id", "logical_condition_id", "relationship") if c in aliases]
        # Identical registry rows do not create additional logical memberships.
        membership_keys = [c for c in ("study", "condition_id") if c in aliases]
        memberships = aliases[membership_columns].drop_duplicates(membership_keys or None)
        studies = sorted(str(study) for study in aliases.study.dropna().unique()) if "study" in aliases else []
        totals = {column: int(block[column].sum()) if not block.empty else np.nan for column in REPLICA_COUNTS}
        recorded = bool(not block.empty and totals["replica_pairs"] > 0)
        eligible = bool(recorded and totals["replica_both_clicks"] > 0)
        n_metric = len(block)
        n_replica = int(block.replica_pairs.gt(0).sum()) if not block.empty else 0
        n_eligible = int(block.replica_both_clicks.gt(0).sum()) if not block.empty else 0
        expected = int(metadata.get("iterations", 30)) if pd.notna(metadata.get("iterations", 30)) else 30
        if expected != 30:
            raise ValueError("Replica agreement overview requires a completed 30-iteration protocol")
        status = ("incomplete_horizon" if n_metric != expected else
                  "incomplete_replica_coverage" if 0 < n_replica < expected else
                  "recorded" if eligible else "no_eligible_clicks" if recorded else "not_recorded")
        rows.append({"source_id": source_id, **metadata,
            "logical_study_memberships": "; ".join(studies),
            "logical_memberships": json.dumps(memberships.where(pd.notna(memberships), None).to_dict("records"), ensure_ascii=False),
            "logical_membership_count": len(memberships), "reused_membership_count": max(0, len(memberships)-1),
            **totals, "n_expected_iterations": expected, "n_metric_iterations": n_metric,
            "n_replica_iterations": n_replica, "n_click_eligible_iterations": n_eligible,
            "complete_horizon": bool(n_metric == expected), "replica_recorded": recorded,
            "replica_status": status,
            AGREEMENT_COLUMN: totals["replica_pair_same_item"]/totals["replica_both_clicks"] if status == "recorded" else np.nan,
            "pooling_window": "iterations_1_30", "agreement_denominator": "both valid clicks"})
    return pd.DataFrame(rows)
