"""Compact, seed-visible presentation of pooled LLM replica agreement."""
from __future__ import annotations

import pandas as pd


TABLE_NAME = "analysis_llm_replica_agreement_all_cases"
COUNT_EXPORT_NAME = "analysis_llm_replica_agreement_runs.csv"
TABLE_COLUMNS = ["Dataset", "Model", "Condition", "Mitigation", "Seed 42", "Seed 43", "Seed 44"]
TABLE_COLUMN_FORMAT = (
    "p{0.11\\linewidth}p{0.10\\linewidth}p{0.20\\linewidth}"
    "p{0.13\\linewidth}p{0.115\\linewidth}p{0.115\\linewidth}p{0.115\\linewidth}"
)
TABLE_CAPTION = (
    "Same-item replica agreement for every LLM condition with recorded identical-order replicas. "
    "Each cell reports agreement in per cent (number of pairs where both responses are valid clicks), "
    "pooling matching-item counts and eligible pairs over iterations 1--30 before division. "
    "The shuffled response does not affect eligibility. All listed runs have 300 active users. "
    "An em dash means that this seed was not run for that condition. "
    "The 150/150 sampling and advertising conditions have no recorded replicas and are excluded. "
    "Reused mitigation controls appear once, alongside their original feedback condition. "
    "This measures repeat-response variability at the same state and slate, rather than variability between complete simulation runs."
)


def _condition(row):
    stage = str(row["stage"]).split("_")[0]
    if row["model"] == "direct_llm":
        return stage
    policy = row["user_update_policy"]
    suffixes = {
        "frozen": "frozen agent",
        "no_reflection_control": "no reflection",
        "memory_only_reflection": "fixed profile",
        "fixed_persona_memory_only": "fixed profile",
        "full_reflection": "full reflection",
    }
    if policy not in suffixes:
        raise ValueError(f"Unknown agent policy for replica table: {policy}")
    return f"{stage} -- {suffixes[policy]}"


def compact_replica_table(runs: pd.DataFrame) -> pd.DataFrame:
    """One condition row, with each seed's agreement and actual denominator.

    The full-precision agreement, raw counts, missing-probe status, physical
    source IDs, and alias memberships remain in the accompanying run CSV.
    """
    if runs.replica_status.isin(["incomplete_horizon", "incomplete_replica_coverage"]).any():
        raise ValueError("Incomplete replica coverage cannot be presented as a complete iterations 1--30 table")
    selected = runs.loc[runs.replica_status.isin(["recorded", "no_eligible_clicks"])].copy()
    if selected.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    if not selected.initial_active_users.eq(300).all() or not selected.initial_reserve_users.eq(0).all():
        raise ValueError("Replica table caption requires 300 active users and no reserve users")
    if not selected.seed.isin([42, 43, 44]).all():
        raise ValueError("Replica table needs an explicit column for every observed seed")
    selected["Dataset"] = selected.dataset.map({"movielens": "MovieLens", "goodreads": "Goodreads"})
    selected["Model"] = selected.model.map({"direct_llm": "Direct LLM", "agentic_recommender": "Agent"})
    selected["Condition"] = selected.apply(_condition, axis=1)
    selected["Mitigation"] = selected.intervention.map({
        "control": "None", "in_processing": "In-processing", "post_processing": "Post-processing"})
    keys = TABLE_COLUMNS[:4]
    if selected[keys].isna().any().any():
        raise ValueError("Unknown dataset, model, or mitigation in replica table")
    if selected.duplicated(keys + ["seed"]).any():
        raise ValueError("Replica display would merge distinct physical runs")
    selected["cell"] = selected.apply(
        lambda row: ("Undefined (0)" if row.replica_status == "no_eligible_clicks" else
            f"{100 * row.pooled_pair_replica_agreement:.1f}% ({int(row.replica_both_clicks):,})"), axis=1)
    table = selected.pivot(index=keys, columns="seed", values="cell").reindex(columns=[42, 43, 44])
    table.columns = [f"Seed {seed}" for seed in table.columns]
    table = table.reset_index()
    # Scientific ordering: dataset, response model, stage/agent variant, and
    # mitigation. No ordering depends on agreement magnitude.
    table["dataset_order"] = table.Dataset.map({"MovieLens": 0, "Goodreads": 1})
    table["model_order"] = table.Model.map({"Direct LLM": 0, "Agent": 1})
    condition_order = ["L0", "L1", "L2", "L0 -- frozen agent", "L1 -- frozen agent", "L2 -- frozen agent",
                       "L3 -- no reflection", "L3 -- fixed profile", "L3 -- full reflection"]
    table["condition_order"] = table.Condition.map(dict(zip(condition_order, range(len(condition_order)))))
    if table.condition_order.isna().any():
        raise ValueError("New replica condition needs an explicit scientific table order")
    table["mitigation_order"] = table.Mitigation.map({"None": 0, "In-processing": 1, "Post-processing": 2})
    table = table.sort_values(["dataset_order", "model_order", "condition_order", "mitigation_order"])
    return table[TABLE_COLUMNS].reset_index(drop=True)
