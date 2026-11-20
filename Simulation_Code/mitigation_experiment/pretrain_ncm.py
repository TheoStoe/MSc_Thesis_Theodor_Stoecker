"""Explicit optional NCM pretraining for mitigation, with legacy slate semantics."""

from __future__ import annotations

import argparse
import csv
import gc
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from .data_preparation import load_prepared_dataset
from .prepare_ncm_artifacts import validate_embedding_catalog
from .utility.run_logging import log_run_attempt
from .user_models.ncm import NCMClickModelConfig
from .user_models.ncm_artifacts import (
    load_ncm_checkpoint_artifact,
    load_ncm_item_embeddings,
    save_ncm_checkpoint_artifact,
)
from .user_models.ncm_training import (
    NCMTrainingConfig,
    load_ncm_training_examples,
    train_ncm_model,
)
from .user_models.ncm_slates import ensure_ncm_training_slates


LOGGER = logging.getLogger(__name__)
POLICIES = {
    "uniform": {
        "model_folder": "P3",
        "condition": "T2_slatewise_gru_uniform",
    },
    "mansoury": {
        "model_folder": "P4",
        "condition": "T3_slatewise_gru_mansoury",
    },
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--policy", choices=tuple(POLICIES), action="append")
    parser.add_argument(
        "--embedding-directory",
        type=Path,
        default=Path(
            "mitigation_experiment/data/text_embedding_3_small_movielens_v1"
        ),
    )
    parser.add_argument(
        "--placement-root",
        type=Path,
    )
    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=Path("mitigation_experiment/inputs/main_seed42_a100_r200/ml1m"),
    )
    parser.add_argument("--slate-seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "mitigation_experiment/outputs/pre_trained_NCM_models/self_contained"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--minimum-improvement", type=float, default=1e-5)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    seeds = tuple(dict.fromkeys(arguments.seed))
    if any(seed < 0 for seed in seeds):
        parser.error("--seed values must be non-negative")
    policies = tuple(dict.fromkeys(arguments.policy or ("uniform", "mansoury")))

    # Logs are unique attempts outside immutable checkpoint directories. This
    # boundary also captures data/embedding loading and checkpoint-save errors.
    with log_run_attempt(arguments.output_root, "pretrain_ncm"):
        return _run_pretraining(arguments, seeds, policies)


def _run_pretraining(arguments, seeds: tuple[int, ...], policies: tuple[str, ...]) -> int:

    dataset = load_prepared_dataset(arguments.prepared_data)
    catalog_verification = validate_embedding_catalog(dataset, arguments.embedding_directory)
    LOGGER.info("Verified NCM item/text catalog: %s", catalog_verification)
    placement_root = arguments.placement_root or Path(
        "mitigation_experiment/data/ncm_training/generated"
    ) / f"slate_seed_{arguments.slate_seed}"
    ensure_ncm_training_slates(
        dataset,
        placement_root,
        slate_seed=arguments.slate_seed,
    )

    embedding_manifest_path = arguments.embedding_directory / "manifest.json"
    if not embedding_manifest_path.is_file():
        raise FileNotFoundError(
            f"missing NCM embedding manifest: {embedding_manifest_path}"
        )
    embeddings, embedding_fingerprint = load_ncm_item_embeddings(
        arguments.embedding_directory
    )
    if set(embeddings) != set(dataset.catalog):
        raise ValueError("NCM embeddings and prepared-data catalog differ")
    embedding_dimension = len(next(iter(embeddings.values())))

    for policy in policies:
        settings = POLICIES[policy]
        placement_directory = placement_root / policy
        LOGGER.info("loading fixed %s NCM training artifacts", policy)
        training, validation, metadata = load_ncm_training_examples(
            placement_directory,
            embeddings,
            prepared_fingerprint=dataset.dataset_fingerprint,
        )
        if metadata.policy != policy:
            raise ValueError(
                f"expected {policy} placement artifact, got {metadata.policy}"
            )
        LOGGER.info(
            "loaded %d training and %d validation examples",
            len(training),
            len(validation),
        )
        model_config = NCMClickModelConfig(
            architecture="slatewise_gru",
            embedding_dimension=embedding_dimension,
            slate_size=10,
            hidden_size=64,
        )
        for seed in seeds:
            run_directory = (
                arguments.output_root
                / str(settings["model_folder"])
                / f"seed_{seed}"
            )
            checkpoint_directory = run_directory / "checkpoint"
            if checkpoint_directory.exists():
                _validate_existing_checkpoint(
                    checkpoint_directory,
                    seed=seed,
                    condition=str(settings["condition"]),
                    embedding_fingerprint=embedding_fingerprint,
                    metadata=metadata,
                )
                LOGGER.info("valid checkpoint already exists; skipping %s", run_directory)
                continue
            if run_directory.exists():
                raise FileExistsError(
                    f"partial NCM run directory will not be overwritten: {run_directory}"
                )
            training_config = NCMTrainingConfig(
                seed=seed,
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                learning_rate=arguments.learning_rate,
                weight_decay=arguments.weight_decay,
                patience=arguments.patience,
                minimum_improvement=arguments.minimum_improvement,
                device=arguments.device,
            )
            LOGGER.info("training %s NCM with seed %d", policy, seed)
            result = train_ncm_model(
                model_config,
                training_config,
                training,
                validation,
            )
            checkpoint_fingerprint = save_ncm_checkpoint_artifact(
                checkpoint_directory,
                model=result.model,
                condition=str(settings["condition"]),
                model_config=asdict(model_config),
                training_config=training_config.to_dict(),
                prepared_fingerprint=metadata.prepared_fingerprint,
                parent_fingerprint=metadata.parent_fingerprint,
                placement_fingerprint=metadata.placement_fingerprint,
                embedding_fingerprint=embedding_fingerprint,
                best_epoch=result.best_epoch,
            )
            _write_training_curve(run_directory / "training_curve.csv", result.records)
            _validate_existing_checkpoint(
                checkpoint_directory,
                seed=seed,
                condition=str(settings["condition"]),
                embedding_fingerprint=embedding_fingerprint,
                metadata=metadata,
            )
            LOGGER.info(
                "saved %s seed %d checkpoint %s",
                policy,
                seed,
                checkpoint_fingerprint,
            )
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del training, validation
        gc.collect()
    return 0


def _validate_existing_checkpoint(
    checkpoint_directory: Path,
    *,
    seed: int,
    condition: str,
    embedding_fingerprint: str,
    metadata,
) -> None:
    manifest, _, _ = load_ncm_checkpoint_artifact(
        checkpoint_directory,
        expected_embedding_fingerprint=embedding_fingerprint,
        expected_prepared_fingerprint=metadata.prepared_fingerprint,
        expected_parent_fingerprint=metadata.parent_fingerprint,
        expected_placement_fingerprint=metadata.placement_fingerprint,
        device="cpu",
    )
    if manifest.get("condition") != condition:
        raise ValueError("existing NCM checkpoint condition mismatch")
    if manifest.get("training_config", {}).get("seed") != seed:
        raise ValueError("existing NCM checkpoint training seed mismatch")


def _write_training_curve(path: Path, records) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "train_binary_cross_entropy",
                "validation_binary_cross_entropy",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


if __name__ == "__main__":
    raise SystemExit(main())
