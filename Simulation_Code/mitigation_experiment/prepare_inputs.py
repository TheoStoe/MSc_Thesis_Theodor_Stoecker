"""Create prepared MovieLens, clustering, and population artifacts once."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_preparation import ClusterPipelineConfig

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


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

    from .data_preparation import (
        ClusterPipelineConfig, fit_user_clusters, prepare_movielens_1m,
        save_cluster_artifact, save_prepared_dataset,
    )
    from .frozen_population import build_frozen_population, load_frozen_membership
    from .utility.saving import save_population_artifact

    if cluster_config != ClusterPipelineConfig():
        raise ValueError("mitigation preparation requires the notebook's frozen cluster configuration")
    if (active_user_count, reserve_user_count) != (6040, 0):
        if (active_user_count, reserve_user_count) not in {(100, 200), (300, 0)}:
            raise ValueError("supported populations are 100+200, 300+0, or 6040+0")
        load_frozen_membership(population_seed)

    _preflight_destination(prepared_output, directory=True)
    _preflight_destination(cluster_output, directory=True)
    _preflight_destination(population_output, directory=False)

    logger.info("Preparing mitigation MovieLens data source=%s", raw_data_directory)
    dataset = prepare_movielens_1m(raw_data_directory)
    logger.info("Fitting frozen mitigation clusters users=%s", len(dataset.training_histories))
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        clusters = fit_user_clusters(dataset, config=cluster_config)
    logger.info("Validating frozen membership seed=%s active=%s reserve=%s", population_seed, active_user_count, reserve_user_count)
    population = build_frozen_population(
        dataset,
        clusters,
        population_seed=population_seed,
        active_user_count=active_user_count,
        reserve_user_count=reserve_user_count,
    )

    logger.info("Saving mitigation preparation artifacts")
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
    from .utility.run_logging import log_run_attempt
    logging.basicConfig(level=logging.INFO)
    with log_run_attempt(ROOT / "preparation_logs", "prepare_inputs"):
        try:
            return _main(argv)
        except SystemExit as exc:
            if exc.code == 0:
                return 0
            raise


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, default=ROOT / "data/ml-1m")
    parser.add_argument("--prepared-output", type=Path, default=ROOT / "inputs/ml1m")
    parser.add_argument("--cluster-output", type=Path, default=ROOT / "inputs/ml1m_clusters")
    parser.add_argument("--population-output", type=Path)
    parser.add_argument("--population-seed", type=int, default=1054)
    parser.add_argument("--active-users", type=int, default=300)
    parser.add_argument("--reserve-users", type=int, default=0)
    parser.add_argument("--svd-components", type=int, default=32)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--cluster-n-init", type=int, default=20)
    parser.add_argument("--svd-seed", type=int, default=42)
    parser.add_argument("--kmeans-seed", type=int, default=42)
    arguments = parser.parse_args(argv)
    from .data_preparation import ClusterPipelineConfig
    if arguments.population_output is None:
        arguments.population_output = ROOT / "inputs" / (
            f"population__seed_{arguments.population_seed}__active_{arguments.active_users}"
            f"__reserve_{arguments.reserve_users}.json"
        )
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
