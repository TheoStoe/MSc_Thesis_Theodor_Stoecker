"""Short, seed-visible definition audit for the Goodreads popularity head."""
from __future__ import annotations

import pandas as pd


TABLE_NAME = "rq6_goodreads_head_cutoff_ties"
SUMMARY_EXPORT_NAME = "rq6_goodreads_head_cutoff_ties_numeric.csv"
ITEMS_EXPORT_NAME = "rq6_goodreads_strict_head_items.csv"
TABLE_COLUMN_FORMAT = (
    "p{0.07\\linewidth}p{0.14\\linewidth}p{0.10\\linewidth}"
    "p{0.16\\linewidth}p{0.21\\linewidth}p{0.21\\linewidth}"
)
TABLE_CAPTION = (
    "Cut-off ties and strict-head membership in Goodreads. "
    "The original head uses the top 20% of catalogue items by initial training count, rounded up, with item-ID tie-breaking. "
    "Tied items in head is shown as count (percentage of the original head). "
    "Strict head includes only items with counts above the cut-off, shown as count (percentage of the catalogue); "
    "it is not refilled to 20%. "
    "Seed denotes the simulation seed. Trajectory metrics retain the original head."
)


def head_tie_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Format counts and denominator-specific percentages without recalculation."""
    selected = summary.sort_values(["seed", "population_seed"]).copy()
    if not selected.dataset.eq("goodreads").all():
        raise ValueError("This table caption describes Goodreads only")
    if selected.duplicated(["seed", "population_seed"]).any():
        raise ValueError("Head reference table needs one row per seed/population")
    if not selected.head_count.sub(selected.selected_tie_count).eq(selected.strict_head_count).all():
        raise ValueError("Strict head must exclude the entire cut-off group from the original head")
    return pd.DataFrame({
        "Seed": selected.seed.astype(int),
        "Population seed": selected.population_seed.astype(int),
        "Cut-off count": selected.cutoff_count.astype(int),
        "Original head (items)": selected.head_count.map(lambda count: f"{int(count):,}"),
        "Tied items in head": selected.apply(
            lambda row: f"{int(row.selected_tie_count):,} ({100 * row.selected_tie_head_share:.1f}%)", axis=1),
        "Strict head": selected.apply(
            lambda row: f"{int(row.strict_head_count):,} ({100 * row.strict_head_catalogue_share:.1f}%)", axis=1),
    }).reset_index(drop=True)
