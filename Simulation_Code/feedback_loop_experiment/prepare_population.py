"""Create one seed-specific population from frozen data and clusters."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .data_preparation import (
    load_cluster_artifact,
    load_prepared_dataset,
    sample_population,
)
from .utility.saving import save_population_artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population-seed", type=int, required=True)
    parser.add_argument("--active-users", type=int, default=150)
    parser.add_argument("--reserve-users", type=int, default=300)
    arguments = parser.parse_args(argv)

    dataset = load_prepared_dataset(arguments.prepared_data)
    clusters = load_cluster_artifact(arguments.clusters)
    population = sample_population(
        dataset,
        clusters,
        population_seed=arguments.population_seed,
        active_user_count=arguments.active_users,
        reserve_user_count=arguments.reserve_users,
    )
    save_population_artifact(arguments.output, population)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
