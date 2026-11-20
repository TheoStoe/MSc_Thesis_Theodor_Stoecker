"""Run one gender-mitigation condition from explicitly supplied Goodreads inputs.

The default only describes the requested job. ``--check`` validates existing
inputs, and ``--run`` explicitly enables the simulation. No historical queue,
launch receipt, or machine-specific campaign plan is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .campaign import Cell, MODELS, STAGES, INTERVENTIONS, cells


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--list-conditions", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--condition")
    parser.add_argument("--model", choices=MODELS, default="position_biased")
    parser.add_argument("--stage", choices=STAGES, default="L2_ml_model")
    parser.add_argument("--intervention", choices=INTERVENTIONS, default="control")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--recommender-epochs", type=int, default=804)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.list_conditions:
        for cell in cells():
            print(cell.condition_id)
        return 0
    if args.condition:
        matches = [cell for cell in cells() if cell.condition_id == args.condition]
        if not matches:
            parser.error("unknown gender-mitigation condition")
        cell = matches[0]
    else:
        cell = Cell(args.model, args.stage, args.intervention)
    description = {
        "condition": cell.condition_id,
        "seed": args.seed,
        "population_seed": args.population_seed,
        "inputs": str(args.inputs) if args.inputs else None,
        "labels": str(args.labels) if args.labels else None,
        "output": str(args.output) if args.output else None,
        "iterations": args.iterations,
        "recommender_epochs": args.recommender_epochs,
        "positive_women_weight": 1.25,
        "rank_promotion": 5,
        "presentation_arms": ["canonical", "replica", "shuffled"],
        "shared_initial_model": True,
        "alias_of": cell.alias_of,
    }
    if not (args.check or args.run):
        print(json.dumps({"status": "described_not_started", **description}, indent=2))
        return 0
    if args.inputs is None or args.labels is None or args.output is None:
        parser.error("--check and --run require --inputs, --labels, and --output")

    # Heavy numerical libraries and the simulation runner are imported only
    # after an explicit check or execution request.
    from .runtime_adapter import build_book_config, load_book_inputs
    from .gender_mitigation import load_gender_labels
    from .engine import experiment_methodology as methodology

    root = args.inputs.resolve()
    prepared = load_book_inputs(
        root / "prepared", root / "population.json", None,
        root / "initial_sasrec" / f"seed_{args.seed}",
    )
    population_seed = prepared.population.population_seed
    if args.population_seed is not None and args.population_seed != population_seed:
        parser.error("requested population seed differs from the prepared cohort")
    labels = load_gender_labels(
        args.labels,
        catalogue_item_ids=prepared.dataset.catalog,
        cohort_fingerprint=prepared.dataset.source_file_fingerprints["cohort_json_sha256"],
        histories=prepared.dataset.training_histories,
    )
    model = {
        "position_biased": methodology.POSITION_BIASED,
        "ncm_inspired_gru_uniform": methodology.NCM_GRU_UNIFORM,
        "ncm_inspired_gru_mansoury": methodology.NCM_GRU_MANSOURY,
    }[cell.user_model]
    condition = methodology.build_condition(
        model, cell.stage,
        condition_suffix=f"gender_{cell.intervention}__order_probe",
        presentation_policy="canonical_replica_and_shuffled_probe",
    )
    config = build_book_config(
        root, args.output, seed=args.seed, population_seed=population_seed,
        active_user_count=len(prepared.population.members),
        user_model=cell.user_model, num_iterations=args.iterations,
        provider_settings={},
        user_model_settings={"alpha": -1.0} if cell.user_model == "position_biased" else {
            "checkpoint_directory": str(root / "ncm" / f"seed_{args.seed}" /
                ("P3" if cell.user_model.endswith("uniform") else "P4") / "checkpoint"),
            "embedding_directory": str(root / "embeddings"), "device": "cpu",
        },
        recommender_settings={
            "device": args.device, "num_epochs": args.recommender_epochs,
            "batch_size": args.batch_size, "max_sequence_length": 200,
        },
        mitigation={
            "strategy": cell.intervention, "women_item_ids": sorted(labels.women_item_ids),
            "women_weight": 1.25, "rank_promotion": 5,
            "label_fingerprint": labels.labels_sha256,
        },
    )
    if args.check:
        if cell.user_model != "position_biased":
            for name in ("checkpoint_directory", "embedding_directory"):
                if not Path(config.user_model_settings[name]).is_dir():
                    raise FileNotFoundError(config.user_model_settings[name])
        print(json.dumps({"status": "ready_not_started", **description,
            "population_seed": population_seed,
            "active_users": len(prepared.population.members),
            "labels_sha256": labels.labels_sha256}, indent=2))
        return 0
    from .engine.run_simulator import run_experiment
    artifact = run_experiment(config, condition, args.seed)
    print(artifact.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
