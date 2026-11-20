"""Create prepared MovieLens, clustering, and population artifacts once."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .data_preparation import (
    ClusterPipelineConfig,
    fit_user_clusters,
    prepare_movielens_1m,
    sample_population,
    save_cluster_artifact,
    save_prepared_dataset,
)
from .utility.saving import save_population_artifact


def prepare_inputs(
    *,
    raw_data_directory: Path,
    prepared_output: Path,
    cluster_output: Path,
    population_output: Path,
    population_seed: int,
    active_user_count: int,
    reserve_user_count: int,
    cluster_config: ClusterPipelineConfig,
) -> None:
    """Build and save a complete non-profile input set without overwriting."""

    _preflight_destination(prepared_output, directory=True)
    _preflight_destination(cluster_output, directory=True)
    _preflight_destination(population_output, directory=False)

    dataset = prepare_movielens_1m(raw_data_directory)
    clusters = fit_user_clusters(dataset, config=cluster_config)
    population = sample_population(
        dataset,
        clusters,
        population_seed=population_seed,
        active_user_count=active_user_count,
        reserve_user_count=reserve_user_count,
    )

    save_prepared_dataset(prepared_output, dataset)
    save_cluster_artifact(cluster_output, clusters)
    save_population_artifact(population_output, population)


def _preflight_destination(path: Path, *, directory: bool) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    incomplete = path.with_name(path.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(f"incomplete output already exists: {incomplete}")
    if not directory and path.suffix.lower() != ".json":
        raise ValueError("population output must be a .json file")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--prepared-output", type=Path, required=True)
    parser.add_argument("--cluster-output", type=Path, required=True)
    parser.add_argument("--population-output", type=Path, required=True)
    parser.add_argument("--population-seed", type=int, default=42)
    parser.add_argument("--active-users", type=int, default=150)
    parser.add_argument("--reserve-users", type=int, default=300)
    parser.add_argument("--svd-components", type=int, default=32)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--cluster-n-init", type=int, default=20)
    parser.add_argument("--svd-seed", type=int, default=42)
    parser.add_argument("--kmeans-seed", type=int, default=42)
    arguments = parser.parse_args(argv)
    prepare_inputs(
        raw_data_directory=arguments.raw_data,
        prepared_output=arguments.prepared_output,
        cluster_output=arguments.cluster_output,
        population_output=arguments.population_output,
        population_seed=arguments.population_seed,
        active_user_count=arguments.active_users,
        reserve_user_count=arguments.reserve_users,
        cluster_config=ClusterPipelineConfig(
            n_components=arguments.svd_components,
            n_clusters=arguments.clusters,
            n_init=arguments.cluster_n_init,
            svd_seed=arguments.svd_seed,
            kmeans_seed=arguments.kmeans_seed,
        ),
    )
    print(arguments.prepared_output)
    print(arguments.cluster_output)
    print(arguments.population_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["prepare_inputs"]
