"""Replay the MovieLens sampling figure from saved aggregate counts.

By default, only the sibling CSV is read. ``--recompute`` instead counts age and
occupation in locally supplied users.dat and the three existing frozen cohorts.
Neither mode samples users, fits clusters, executes notebooks, or runs models.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
STEM = "movielens_age_occupation_300_vs_full"
COHORTS = ((42, 1054), (43, 19277), (44, 18844))
AGE_LABELS = {
    1: "Under 18", 18: "18–24", 25: "25–34", 35: "35–44",
    45: "45–49", 50: "50–55", 56: "56+",
}
OCCUPATION_LABELS = {
    0: "Other / unspecified", 1: "Academic / educator", 2: "Artist",
    3: "Clerical / admin", 4: "College / grad student", 5: "Customer service",
    6: "Doctor / health care", 7: "Executive / managerial", 8: "Farmer",
    9: "Homemaker", 10: "K–12 student", 11: "Lawyer", 12: "Programmer",
    13: "Retired", 14: "Sales / marketing", 15: "Scientist",
    16: "Self-employed", 17: "Technician / engineer", 18: "Trades / crafts",
    19: "Unemployed", 20: "Writer",
}
VARIABLES = (("age_code", AGE_LABELS, "Age"),
             ("occupation_code", OCCUPATION_LABELS, "Occupation"))
COLUMNS = (
    "simulation_seed", "population_seed", "variable", "category_code", "category",
    "sample_n", "sample_total", "reference_n", "reference_total", "sample_share",
    "reference_share", "difference_percentage_points",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulation_root() -> Path:
    """Find the enclosing code root after copying to datasets/sampling/."""
    for candidate in HERE.parents:
        if (candidate / "mitigation_experiment").is_dir():
            return candidate
    raise ValueError("--recompute requires this folder inside Simulation_Code, "
                     "which contains mitigation_experiment/.")


def validate_aggregates(frame):
    """Reject incomplete or internally inconsistent saved figure inputs."""
    import numpy as np

    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Aggregate CSV is missing columns: {sorted(missing)}")
    expected = {(simulation, population, variable, category)
                for simulation, population in COHORTS
                for variable, labels, _ in VARIABLES for category in labels}
    keys = ["simulation_seed", "population_seed", "variable", "category_code"]
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Expected one row per category for each of the three frozen cohorts")
    integer_columns = ["simulation_seed", "population_seed", "category_code",
                       "sample_n", "sample_total", "reference_n", "reference_total"]
    values = frame[integer_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Seeds, category codes, and counts must be finite integers")
    if (frame[["sample_n", "reference_n"]] < 0).any().any():
        raise ValueError("Category counts cannot be negative")
    if not frame.sample_total.eq(300).all() or not frame.reference_total.eq(6040).all():
        raise ValueError("The figure compares 300 selected users with 6,040 prepared users")
    for variable, labels, _ in VARIABLES:
        subset = frame.loc[frame.variable.eq(variable)]
        if not subset.category.eq(subset.category_code.map(labels)).all():
            raise ValueError(f"Unexpected category labels for {variable}")
        for _, group in subset.groupby("population_seed", sort=False):
            if group.sample_n.sum() != 300 or group.reference_n.sum() != 6040:
                raise ValueError("Category counts do not sum to the declared population sizes")
        reference = subset.pivot(index="category_code", columns="population_seed", values="reference_n")
        if not reference.eq(reference.iloc[:, 0], axis=0).all().all():
            raise ValueError("The three panels must share one full-population reference")
    for column, expected_values in (
        ("sample_share", frame.sample_n / frame.sample_total),
        ("reference_share", frame.reference_n / frame.reference_total),
        ("difference_percentage_points", 100 * (frame.sample_share - frame.reference_share)),
    ):
        if not np.allclose(frame[column], expected_values, rtol=0, atol=1e-12):
            raise ValueError(f"{column} disagrees with the saved counts/shares")
    return frame.loc[:, list(COLUMNS)]


def load_aggregates(path: Path):
    import pandas as pd

    return validate_aggregates(pd.read_csv(path, float_precision="round_trip"))


def _validate_prepared(root: Path, simulation: int, selected: list[int], all_users: set[int],
                       *, required: bool, sources: dict) -> None:
    directory = root / f"ml_seed{simulation}"
    mapping_path = directory / "prepared" / "mappings.json"
    population_path = directory / "population.json"
    for path in (mapping_path, population_path):
        if required and not path.is_file():
            raise FileNotFoundError(f"Explicit --prepared-root is missing {path}")
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        actual = [int(value) for value in mapping["runtime_to_source_user_id"].values()]
        if len(actual) != 6040 or set(actual) != all_users:
            raise ValueError(f"Prepared user mapping differs from users.dat: {mapping_path}")
        sources[str(mapping_path)] = sha256(mapping_path)
    if population_path.is_file():
        payload = json.loads(population_path.read_text(encoding="utf-8"))
        population = payload.get("population", payload)
        active = [int(member["source_user_id"]) for member in population["members"]
                  if member["role"] == "active"]
        if len(active) != 300 or set(active) != set(selected):
            raise ValueError(f"Active-300 population differs from the frozen union: {population_path}")
        sources[str(population_path)] = sha256(population_path)


def recompute_aggregates(root: Path, raw_data: Path, prepared_root: Path,
                         *, require_prepared: bool = False):
    """Count fixed memberships; never draw users or rebuild prepared inputs."""
    import pandas as pd

    users_path = raw_data / "users.dat"
    users = pd.read_csv(users_path, sep="::", engine="python", header=None,
                        names=["source_user_id", "gender", "age_code", "occupation_code", "zip_code"],
                        dtype={"source_user_id": int, "age_code": int,
                               "occupation_code": int, "zip_code": str})
    if len(users) != 6040 or not users.source_user_id.is_unique:
        raise ValueError("users.dat must contain all 6,040 unique MovieLens 1M users")
    all_users = set(users.source_user_id)
    if all_users != set(range(1, 6041)):
        raise ValueError("Expected the original MovieLens 1M source user IDs 1–6040")
    for variable, labels, _ in VARIABLES:
        if not set(users[variable]).issubset(labels):
            raise ValueError(f"Unknown {variable} categories in users.dat")
    users = users.set_index("source_user_id")
    sources = {str(users_path): sha256(users_path)}
    rows = []
    for simulation, population in COHORTS:
        path = root / "mitigation_experiment" / "data" / "populations" / f"population_{population}.json"
        frozen = json.loads(path.read_text(encoding="utf-8"))
        membership = frozen["membership"]
        active, reserve = membership["active"], membership["reserve"]
        selected = active + reserve
        digest = hashlib.sha256(json.dumps(membership, sort_keys=True,
                                          separators=(",", ":")).encode("utf-8")).hexdigest()
        if (frozen.get("population_seed") != population or len(active) != 100
                or len(reserve) != 200 or len(set(selected)) != 300
                or not set(selected).issubset(all_users)
                or digest != frozen.get("membership_sha256")):
            raise ValueError(f"Invalid frozen 100-active/200-reserve membership: {path}")
        sources[str(path)] = sha256(path)
        _validate_prepared(prepared_root, simulation, selected, all_users,
                           required=require_prepared, sources=sources)
        cohort = users.loc[selected]
        for variable, labels, _ in VARIABLES:
            reference_counts = users[variable].value_counts()
            sample_counts = cohort[variable].value_counts()
            for category, label in labels.items():
                full_n = int(reference_counts.get(category, 0))
                sample_n = int(sample_counts.get(category, 0))
                rows.append({"simulation_seed": simulation, "population_seed": population,
                             "variable": variable, "category_code": category, "category": label,
                             "sample_n": sample_n, "sample_total": 300,
                             "reference_n": full_n, "reference_total": 6040,
                             "sample_share": sample_n / 300, "reference_share": full_n / 6040,
                             "difference_percentage_points": 100 * (sample_n / 300 - full_n / 6040)})
    return validate_aggregates(pd.DataFrame(rows)), sources


def plot_aggregates(frame, output_dir: Path) -> list[Path]:
    """Use the original figure's labels, limits, layout, and marker styles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 13,
             "axes.titlesize": 15, "axes.labelsize": 13,
             "savefig.facecolor": "white", "figure.facecolor": "white",
             "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(12, 10), sharex=True, sharey="row",
                                 gridspec_kw={"height_ratios": [1, 2.45]}, layout="constrained")
        for col, (_, population) in enumerate(COHORTS):
            for row, (variable, labels, heading) in enumerate(VARIABLES):
                axis = axes[row, col]
                group = frame.loc[frame.population_seed.eq(population) & frame.variable.eq(variable)]
                group = group.set_index("category_code").loc[list(labels)]
                y = np.arange(len(labels))
                axis.hlines(y, group.reference_share, group.sample_share,
                            color="#B5B5B5", linewidth=1, zorder=1)
                axis.scatter(group.sample_share, y, marker="^", s=60, color="#009E73",
                             edgecolors="black", linewidths=.4, zorder=3)
                axis.scatter(group.reference_share, y, marker="x", s=47,
                             color="#6B7280", linewidths=1.5, zorder=4)
                axis.set(xlim=(0, 1), ylim=(len(labels) - .65, -.65),
                         title=f"{heading}\nPopulation seed {population}", xlabel="User share")
                axis.set_xticks([0, .2, .4, .6, .8, 1])
                axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                axis.set_yticks(y, list(labels.values()), fontsize=12)
                axis.grid(axis="x", color="#DDDDDD", linewidth=.7, zorder=0)
                axis.grid(axis="y", color="#EAEAEA", linewidth=.6, zorder=0)
                axis.spines[["top", "right"]].set_visible(False)
                axis.spines[["left", "bottom"]].set_color("#000000")
                axis.tick_params(axis="both", length=0, pad=5, labelbottom=True)
                axis.set_axisbelow(True)
        handles = [Line2D([], [], linestyle="none", marker="^", markersize=8,
                          markerfacecolor="#009E73", markeredgecolor="black", markeredgewidth=.5,
                          label="Selected population (300 users)"),
                   Line2D([], [], linestyle="none", marker="x", markersize=8, color="#6B7280",
                          label="Full prepared population (6,040 users)")]
        fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=13)
        paths = [output_dir / f"{STEM}.{suffix}" for suffix in ("png", "pdf", "svg")]
        for path in paths:
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=HERE / f"{STEM}.csv",
                        help="Saved aggregate input (default: sibling CSV; unused with --recompute)")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures",
                        help="Destination for PNG/PDF/SVG, counts, and replay receipt")
    parser.add_argument("--recompute", action="store_true",
                        help="Recount fixed cohorts from local users.dat; never sample users")
    parser.add_argument("--raw-data", type=Path,
                        help="MovieLens folder (default: Simulation_Code/datasets/raw/ml-1m)")
    parser.add_argument("--prepared-root", type=Path,
                        help="Validate all ml_seed42/43/44 mappings and populations here; "
                             "default datasets/processed validates files only when present")
    args = parser.parse_args(argv)
    if not args.recompute and (args.raw_data is not None or args.prepared_root is not None):
        parser.error("--raw-data and --prepared-root require --recompute")
    try:
        if args.recompute:
            root = simulation_root()
            frame, sources = recompute_aggregates(
                root, args.raw_data or root / "datasets/raw/ml-1m",
                args.prepared_root or root / "datasets/processed",
                require_prepared=args.prepared_root is not None)
        else:
            frame = load_aggregates(args.csv)
            sources = {str(args.csv): sha256(args.csv)}
        paths = plot_aggregates(frame, args.output_dir)
        csv_path = args.output_dir / f"{STEM}.csv"
        if args.recompute:
            frame.to_csv(csv_path, index=False)
        elif args.csv.resolve() != csv_path.resolve():
            shutil.copyfile(args.csv, csv_path)
        paths.append(csv_path)
        receipt = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "fixed_membership_recount" if args.recompute else "saved_aggregate_replay",
            "resampling_performed": False, "clustering_fitted": False,
            "seed_search_run": False, "notebook_cells_executed": False,
            "training_or_paid_calls": False, "user_weighted": True,
            "cohorts": [{"simulation_seed": simulation, "population_seed": population,
                         "selected_users": 300, "reference_users": 6040}
                        for simulation, population in COHORTS],
            "sources_sha256": sources,
            "outputs_sha256": {path.name: sha256(path) for path in paths},
        }
        (args.output_dir / "replay_provenance.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Sampling figure: {error}\n")
    print(f"Saved PNG, PDF, SVG, aggregate CSV, and replay receipt to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
