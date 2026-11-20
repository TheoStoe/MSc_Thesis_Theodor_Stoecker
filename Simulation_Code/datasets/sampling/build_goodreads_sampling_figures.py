"""Replay the two Goodreads sampling figures from saved aggregate CSVs.

No feature cache, individual cohort records, sampling, fitting, notebook execution,
simulation, or provider access is required. Inputs are read beside this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
COHORTS = ((42, 16040), (43, 11615), (44, 5245))
REFERENCE_N = 750205
SAMPLE_N = 300
INPUT_NAMES = {
    "means": "goodreads_sampling_300_feature_means.csv",
    "composition": "goodreads_sampling_300_composition.csv",
    "criteria": "goodreads_sampling_300_balance_criteria.csv",
}
LABELS = {
    "training_count": "Training-history length",
    "log_training_count": "log(1 + history length)",
    "mean_rating": "Mean training rating",
    "mean_log_item_popularity": "Mean log(1 + work popularity)",
    "head_share": "Historical head-item share",
    **{f"rating_share_{rating}": f"Proportion of rating {rating}" for rating in range(1, 6)},
    "training_span_days": "Training-history span (days)",
}
FEATURES = tuple(LABELS)
BEHAVIOUR_FEATURES = FEATURES[:4]
OTHER_FEATURES = ("head_share", "training_span_days")
TOLERANCES = {
    "cluster_tvd": .012, "activity_decile_tvd": .10,
    "behavior_max_abs_smd": .13, "rating_histogram_tvd": .05,
    "other_mean_abs_smd": .13,
}
GROUPS = {
    "preference_cluster": tuple(range(5)),
    "training_activity_decile": tuple(range(1, 11)),
    "within_reader_rating_proportions": tuple(range(1, 6)),
}
SAMPLE_COLOUR = "#009E73"
REFERENCE_COLOUR = "#6B7280"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_keys(frame, columns, expected, name):
    if frame.duplicated(columns).any() or set(frame[columns].itertuples(index=False, name=None)) != expected:
        raise ValueError(f"{name} must contain exactly one row per expected cohort/category")


def validate_aggregates(means, composition, criteria):
    """Check completeness and agreement among the three saved tables."""
    import numpy as np

    schemas = (
        (means, {"simulation_seed", "population_seed", "feature", "label", "reference_n",
                 "reference_mean", "reference_population_sd", "sample_n", "sample_mean",
                 "sample_minus_reference", "signed_smd", "absolute_smd"}, "Feature means"),
        (composition, {"simulation_seed", "population_seed", "feature_group", "category",
                       "reference_n", "sample_n", "reference_count", "sample_count",
                       "reference_share", "sample_share", "weighting"}, "Composition"),
        (criteria, {"simulation_seed", "population_seed", "criterion", "tolerance",
                    "observed_discrepancy", "discrepancy_divided_by_tolerance"}, "Balance criteria"),
    )
    for frame, required, name in schemas:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    _check_keys(means, ["simulation_seed", "population_seed", "feature"],
                {(simulation, population, feature) for simulation, population in COHORTS
                 for feature in FEATURES}, "Feature means")
    _check_keys(composition, ["simulation_seed", "population_seed", "feature_group", "category"],
                {(simulation, population, group, category) for simulation, population in COHORTS
                 for group, categories in GROUPS.items() for category in categories}, "Composition")
    _check_keys(criteria, ["simulation_seed", "population_seed", "criterion"],
                {(simulation, population, criterion) for simulation, population in COHORTS
                 for criterion in TOLERANCES}, "Balance criteria")
    for frame, _, name in schemas[:2]:
        if not frame.reference_n.eq(REFERENCE_N).all() or not frame.sample_n.eq(SAMPLE_N).all():
            raise ValueError(f"{name} must compare 300 readers with 750,205 eligible readers")
    if not means.label.eq(means.feature.map(LABELS)).all():
        raise ValueError("Feature labels differ from the saved figure labels")
    numeric = means[["reference_mean", "reference_population_sd", "sample_mean",
                     "sample_minus_reference", "signed_smd", "absolute_smd"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not means.reference_population_sd.gt(0).all():
        raise ValueError("Feature means and population standard deviations must be finite")
    for column, expected in (
        ("sample_minus_reference", means.sample_mean - means.reference_mean),
        ("signed_smd", (means.sample_mean - means.reference_mean) / means.reference_population_sd),
        ("absolute_smd", abs(means.signed_smd)),
    ):
        if not np.allclose(means[column], expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{column} disagrees with the saved means and population SD")
    for column in ("reference_mean", "reference_population_sd"):
        reference = means.pivot(index="feature", columns="population_seed", values=column)
        if not reference.eq(reference.iloc[:, 0], axis=0).all().all():
            raise ValueError("The cohorts must share the same eligible-population reference")
    shares = composition[["reference_share", "sample_share"]].to_numpy(dtype=float)
    if not np.isfinite(shares).all() or (shares < 0).any() or (shares > 1).any():
        raise ValueError("Composition shares must be finite values from zero to one")
    if not np.allclose(composition.groupby(["population_seed", "feature_group"])[
            ["reference_share", "sample_share"]].sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Composition shares must sum to one within each panel")
    counted = composition.loc[composition.feature_group.ne("within_reader_rating_proportions")]
    counts = counted[["reference_count", "sample_count"]].to_numpy(dtype=float)
    if (not np.isfinite(counts).all() or (counts < 0).any()
            or not np.equal(counts, np.floor(counts)).all()):
        raise ValueError("Reader-category counts must be nonnegative integers")
    for prefix in ("reference", "sample"):
        if not np.allclose(counted[f"{prefix}_share"],
                           counted[f"{prefix}_count"] / counted[f"{prefix}_n"], rtol=0, atol=1e-12):
            raise ValueError("Reader shares disagree with the saved category counts")
    for population in (pair[1] for pair in COHORTS):
        feature_rows = means.loc[means.population_seed.eq(population)].set_index("feature")
        rating = composition.loc[composition.population_seed.eq(population)
                                 & composition.feature_group.eq("within_reader_rating_proportions")]
        rating = rating.set_index("category").loc[list(GROUPS["within_reader_rating_proportions"])]
        for prefix in ("reference", "sample"):
            expected = [feature_rows.loc[f"rating_share_{value}", f"{prefix}_mean"] for value in range(1, 6)]
            if not np.allclose(rating[f"{prefix}_share"], expected, rtol=0, atol=1e-12):
                raise ValueError("Rating shares disagree with mean within-reader rating proportions")
        observed = {}
        for group, criterion in (("preference_cluster", "cluster_tvd"),
                                  ("training_activity_decile", "activity_decile_tvd"),
                                  ("within_reader_rating_proportions", "rating_histogram_tvd")):
            rows = composition.loc[composition.population_seed.eq(population)
                                   & composition.feature_group.eq(group)]
            observed[criterion] = float(abs(rows.sample_share - rows.reference_share).sum() / 2)
        observed["behavior_max_abs_smd"] = float(feature_rows.loc[list(BEHAVIOUR_FEATURES), "absolute_smd"].max())
        observed["other_mean_abs_smd"] = float(feature_rows.loc[list(OTHER_FEATURES), "absolute_smd"].mean())
        rows = criteria.loc[criteria.population_seed.eq(population)].set_index("criterion")
        for criterion, tolerance in TOLERANCES.items():
            row = rows.loc[criterion]
            if (not np.isclose(row.tolerance, tolerance, rtol=0, atol=1e-15)
                    or not np.isclose(row.observed_discrepancy, observed[criterion], rtol=1e-12, atol=1e-12)
                    or not np.isclose(row.discrepancy_divided_by_tolerance,
                                      row.observed_discrepancy / tolerance, rtol=1e-12, atol=1e-12)):
                raise ValueError(f"Inconsistent {criterion} aggregate for population {population}")
    return means, composition, criteria


def load_aggregates(directory: Path = HERE):
    import pandas as pd

    frames = [pd.read_csv(directory / filename, float_precision="round_trip")
              for filename in INPUT_NAMES.values()]
    return validate_aggregates(*frames)


def plot_aggregates(means, composition, output_dir: Path) -> list[Path]:
    """Preserve the two original figures' order, axes, labels, and styles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save_figure(fig, name):
        for suffix in ("pdf", "svg", "png"):
            path = output_dir / f"{name}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    style = {"font.family": "DejaVu Sans", "font.size": 11.5,
             "axes.titlesize": 12, "axes.labelsize": 11.5,
             "figure.facecolor": "white", "savefig.facecolor": "white",
             "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        activity_labels = ["1 (3–10)", "2 (11–20)", "3 (21–30)", "4 (31–43)",
                           "5 (44–61)", "6 (62–86)", "7 (87–123)", "8 (124–186)",
                           "9 (187–324)", "10 (325+)"]
        panels = (
            ("preference_cluster", [f"Cluster {value}" for value in range(5)], "Preference cluster"),
            ("training_activity_decile", activity_labels, "Activity decile (training count)"),
            ("within_reader_rating_proportions", [str(value) for value in range(1, 6)], "Rating value"),
        )
        fig, axes = plt.subplots(3, 3, figsize=(8.5, 10.4), sharex=True, sharey="row",
                                 gridspec_kw={"height_ratios": [1, 1.75, 1]})
        fig.subplots_adjust(left=.25, right=.985, bottom=.16, top=.90, hspace=.23, wspace=.21)
        for column, (simulation, population) in enumerate(COHORTS):
            for row, (group_name, labels, axis_label) in enumerate(panels):
                ax = axes[row, column]
                data = composition.loc[composition.population_seed.eq(population)
                                       & composition.feature_group.eq(group_name)]
                data = data.set_index("category").loc[list(GROUPS[group_name])]
                y = np.arange(len(labels))
                ax.scatter(data.reference_share, y, color=REFERENCE_COLOUR, marker="x",
                           s=58, linewidths=1.6, zorder=4)
                ax.scatter(data.sample_share, y, color=SAMPLE_COLOUR, marker="^", s=45, zorder=5)
                ax.set_yticks(y, labels)
                ax.set_ylim(len(labels) - .55, -.55)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, .5, 1])
                ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                ax.grid(axis="both", color="#DDDDDD", linewidth=.7, zorder=0)
                ax.tick_params(axis="both", labelsize=10.5)
                if row == 0:
                    ax.set_title(f"Population {population}\nSimulation seed {simulation}", pad=31)
                if column == 0:
                    ax.set_ylabel(axis_label)
                if row == 2:
                    ax.set_xlabel("Mean within-reader\nrating share")
                    ax.get_xticklabels()[0].set_ha("left")
                    ax.get_xticklabels()[-1].set_ha("right")
                elif row == 1:
                    ax.text(.98, 1.01, "Reader share", ha="right", va="bottom",
                            transform=ax.transAxes, fontsize=9.5)
                if row == 0:
                    ax.text(.98, 1.01, "Reader share", ha="right", va="bottom",
                            transform=ax.transAxes, fontsize=9.5)
        fig.legend(handles=[
            Line2D([], [], color=REFERENCE_COLOUR, marker="x", linestyle="None", markersize=8,
                   label="All 750,205 eligible readers"),
            Line2D([], [], color=SAMPLE_COLOUR, marker="^", linestyle="None", markersize=8,
                   label="Frozen sample of 300 readers"),
        ], loc="lower center", bbox_to_anchor=(.57, .012), ncol=1, frameon=False)
        save_figure(fig, "goodreads_composition_300_vs_eligible")

        fig, axes = plt.subplots(1, 3, figsize=(8.5, 6.0), sharex=True, sharey=True)
        fig.subplots_adjust(left=.355, right=.98, bottom=.16, top=.86, wspace=.15)
        maximum = float(np.abs(means.signed_smd).max())
        limit = max(.05, np.ceil(maximum * 1.17 * 100) / 100)
        for ax, (simulation, population) in zip(axes, COHORTS):
            data = means.loc[means.population_seed.eq(population)].set_index("feature").loc[list(FEATURES)]
            y = np.arange(len(FEATURES))
            ax.axvline(0, color=REFERENCE_COLOUR, linewidth=1.4, zorder=2)
            ax.scatter(data.signed_smd, y, color=SAMPLE_COLOUR, marker="^", s=50, zorder=4)
            ax.set_title(f"Population {population}\nSimulation seed {simulation}", pad=12)
            ax.set_yticks(y, [LABELS[name] for name in FEATURES])
            ax.set_ylim(len(FEATURES) - .45, -.65)
            ax.set_xlim(-limit, limit)
            ax.set_xticks([-.10, 0, .10], ["−0.10", "0", "0.10"])
            ax.grid(axis="both", color="#DDDDDD", linewidth=.7, zorder=0)
            ax.tick_params(axis="both", labelsize=10.5)
        fig.supxlabel("Signed standardised mean difference", x=.65, y=.055, fontsize=11)
        fig.text(.65, .025, "Zero = eligible-population mean", ha="center", fontsize=10.5)
        save_figure(fig, "goodreads_all_features_300_vs_eligible")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures",
                        help="Destination for the two PDF/PNG/SVG figure bundles and replay receipt")
    args = parser.parse_args(argv)
    try:
        means, composition, criteria = load_aggregates()
        paths = plot_aggregates(means, composition, args.output_dir)
        receipt = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "saved_aggregate_replay", "reference_readers": REFERENCE_N,
            "cohorts": [{"simulation_seed": simulation, "population_seed": population,
                         "readers": SAMPLE_N} for simulation, population in COHORTS],
            "input_rows": {"feature_means": len(means), "composition": len(composition),
                           "balance_criteria": len(criteria)},
            "resampling_performed": False, "clustering_fitted": False,
            "seed_search_run": False, "notebook_cells_executed": False,
            "training_or_paid_calls": False,
            "inputs_sha256": {name: sha256(HERE / name) for name in INPUT_NAMES.values()},
            "outputs_sha256": {path.name: sha256(path) for path in paths},
        }
        (args.output_dir / "goodreads_replay_provenance.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Goodreads sampling figures: {error}\n")
    print(f"Saved two PDF/PNG/SVG figure bundles and replay receipt to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
