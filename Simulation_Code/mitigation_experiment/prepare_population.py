"""Create one seed-specific population from frozen data and clusters."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    from .utility.run_logging import log_run_attempt
    logging.basicConfig(level=logging.INFO)
    with log_run_attempt(ROOT / "preparation_logs", "prepare_population"):
        try:
            return _main(argv)
        except SystemExit as exc:
            if exc.code == 0:
                return 0
            raise


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, default=ROOT / "inputs/ml1m")
    parser.add_argument("--clusters", type=Path, default=ROOT / "inputs/ml1m_clusters")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--population-seed", type=int, default=1054)
    parser.add_argument("--active-users", type=int, default=300)
    parser.add_argument("--reserve-users", type=int, default=0)
    arguments = parser.parse_args(argv)
    from .data_preparation import load_cluster_artifact, load_prepared_dataset
    from .frozen_population import build_frozen_population
    from .prepare_inputs import _preflight_destination
    from .utility.saving import save_population_artifact
    if arguments.output is None:
        arguments.output = ROOT / "inputs" / (
            f"population__seed_{arguments.population_seed}__active_{arguments.active_users}"
            f"__reserve_{arguments.reserve_users}.json"
        )
    _preflight_destination(arguments.output, directory=False)

    logger.info("Loading mitigation prepared data=%s clusters=%s", arguments.prepared_data, arguments.clusters)
    dataset = load_prepared_dataset(arguments.prepared_data)
    clusters = load_cluster_artifact(arguments.clusters)
    logger.info("Materializing frozen membership seed=%s active=%s reserve=%s", arguments.population_seed, arguments.active_users, arguments.reserve_users)
    population = build_frozen_population(
        dataset,
        clusters,
        population_seed=arguments.population_seed,
        active_user_count=arguments.active_users,
        reserve_user_count=arguments.reserve_users,
    )
    save_population_artifact(arguments.output, population)
    logger.info("Saved frozen population at %s", arguments.output)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
