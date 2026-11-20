"""Compact thesis tables from typed manual-review frequency data."""
import pandas as pd


def frequency_table(frequencies, dimension):
    selected = frequencies.loc[frequencies.dimension.eq(dimension)].copy()
    if dimension not in {"extent_of_change", "support_in_available_evidence", "review_status"}:
        raise ValueError("Unknown qualitative profile-review dimension")
    table = pd.DataFrame({
        "Seed": ["All seeds" if row.scope == "overall" else str(int(row.seed)) for row in selected.itertuples()],
        "Status" if dimension == "review_status" else "Category": selected.category.to_numpy(),
        "Profiles": selected["count"].to_numpy(),
    })
    if dimension != "review_status":
        table["Reviewed"] = selected.reviewed_profiles.to_numpy()
        table["Unfinished"] = selected.unassessed_profiles.to_numpy()
        table["Share reviewed"] = [f"{value:.1%}" if pd.notna(value) else None for value in selected.share_of_reviewed]
    return table.reset_index(drop=True)


CATEGORY_COLUMNS = r"p{0.12\linewidth}p{0.26\linewidth}rrrr"
STATUS_COLUMNS = r"p{0.18\linewidth}p{0.42\linewidth}r"
