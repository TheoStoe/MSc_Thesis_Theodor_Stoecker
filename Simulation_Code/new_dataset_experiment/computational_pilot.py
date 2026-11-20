"""Prepare the first Goodreads computational comparison without starting it.

The six cells reuse the primary campaign implementation and label-readiness
checks. Run outputs have their own root, so exploratory analysis cannot overwrite
the main study. A run requires one explicit seed and condition; no bulk launch is
provided here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from . import campaign


SIMULATION_SEED = 42
POPULATION_SEED = 16040
DEFAULT_COHORT = campaign.ROOT / "inputs" / "representative_a300_v2" / "cohort.json"
DEFAULT_INPUTS = campaign.ROOT / "inputs" / "goodreads_representative_a300_v2"
DEFAULT_OUTPUTS = campaign.ROOT / "outputs" / "computational_representative_seed42_v3"
PILOT_MODELS = ("position_biased", "ncm_inspired_gru_uniform", "ncm_inspired_gru_mansoury")


def cells() -> tuple[campaign.BookCell, ...]:
    """Select the three agreed response families while preserving their cells."""
    return tuple(cell for cell in campaign.cells() if cell.user_model in PILOT_MODELS)


def describe(*, seed: int = SIMULATION_SEED,
             population_seed: int = POPULATION_SEED) -> dict[str, Any]:
    """Describe only; no dataset, provider or scientific runtime is imported."""
    if isinstance(seed, bool) or seed != SIMULATION_SEED:
        raise ValueError("the initial computational pilot uses simulation seed 42; extend seeds in the primary campaign")
    design = campaign.describe(seeds=(seed,), population_seed=population_seed)
    design.pop("provider")
    selected = cells()
    design.update({
        "schema_version": "goodreads-computational-pilot-v2",
        "purpose": "Test and inspect gender-exposure and consumption trajectories before the full response-model comparison.",
        "conditions": [{
            "condition_id": cell.condition_id, "user_model": cell.user_model,
            "stage": cell.stage, "update_policy": cell.update_policy, "probes": cell.probes,
        } for cell in selected],
        "comparisons": [row for row in campaign.comparisons() if row["user_model"] in PILOT_MODELS],
        "cells_per_seed": len(selected), "trajectories": len(selected),
        "response_api_calls": False,
        "profile_generation_required": False,
        "initial_model_preparation": {
            "shared_population_seed": population_seed,
            "fresh_sasrec_seeds": [seed],
            "fresh_response_variants": ["P3", "P4"],
            "embeddings": "Fresh book-text embeddings are required for P3/P4 and are shared by both models. Their one-off preparation may use a paid API; the six simulations use no LLM calls.",
            "training": "Local SASRec/NCM preparation and L2 SASRec refits still require compute time.",
        },
        "readiness": {
            "status": "not_checked_by_describe",
            "dataset_files": "Book metadata, author metadata and dated reader interactions/reviews; optional genre metadata.",
            "frozen_cohort": "Exactly 300 readers sampled independently of gender outcomes; 30 iterations, no reserve or background readers.",
            "gender_labels": "A complete cohort-scoped authority audit, the versioned non-unknown-consensus annotation policy, explicit unresolved categories and validated coverage are required; automated name links remain provisional.",
            "prerequisites": ["frozen cohort and prepared inputs", "validated versioned author-label artifact", "fresh seed-42 initial SASRec and slate audit", "book embeddings", "fresh seed-42 P3/P4 checkpoints"],
            "no_placeholder_launch": "Missing gender labels or dated reader histories must not be substituted with synthetic or inferred labels for a real campaign.",
        },
        "analysis": {
            "primary": "Canonical gender exposure and consumption trajectories; unknown items remain in denominators.",
            "secondary": ["genre breakdowns", "author-label coverage", "head-item exposure and consumption", "replica/shuffled response diagnostics"],
            "contrasts": "Matched L2-L0 per response family; a single seed is descriptive and is not an independent-seed uncertainty estimate.",
            "before_adapting": "Inspect cohort structure, annotation provenance and label coverage, not whether a chosen cohort creates a stronger gender effect. Version any scientific change before new runs.",
        },
        "reuse_in_full_campaign": {
            "potentially_reusable": True,
            "rule": "Reuse completed cell artifacts only when frozen cohort, prepared inputs, seed, initial checkpoint, response checkpoint, conditions, probes and implementation are unchanged; verify manifests before copying into the primary analysis.",
            "avoid_duplicate_evidence": "An unchanged pilot cell and its copied production artifact are the same trajectory, not two replications.",
            "changed_design": "If any scientific setting changes, use a fresh versioned input/output root and do not merge the old cell into the final matched comparison.",
        },
    })
    design["interpretation"] = {
        key: value for key, value in design["interpretation"].items()
        if key in {"fixed_population", "L2_vs_L0"}
    }
    design["interpretation"]["L0"] = "Public history and recommender are frozen; consumed items remain excluded."
    return design


def preflight_all(inputs_root: Path, *, labels_path: Path | None = None,
                  population_seed: int | None = None) -> dict[str, Any]:
    """Check the real campaign's prerequisites, without preparing or launching."""
    checks = [campaign.preflight(inputs_root, SIMULATION_SEED, cell, labels_path=labels_path,
                                 population_seed=population_seed) for cell in cells()]
    return {"schema_version": "goodreads-computational-readiness-v1", "simulation_seed": SIMULATION_SEED,
            "ready": all(check["ready"] for check in checks), "checks": checks}


def write_plan(destination: Path, *, inputs_root: Path = DEFAULT_INPUTS,
               output_root: Path = DEFAULT_OUTPUTS, labels_path: Path | None = None,
               population_seed: int = POPULATION_SEED,
               cohort_path: Path = DEFAULT_COHORT) -> dict[str, Any]:
    """Write a new plan exclusively; commands are recorded and never executed."""
    payload = describe(population_seed=population_seed)
    payload.update({"inputs_root": str(inputs_root.resolve()), "output_root": str(output_root.resolve()),
                    "labels_path": str((labels_path or inputs_root / "item_labels.json").resolve())})
    labels_arguments = ["--labels", str(labels_path.resolve())] if labels_path is not None else []
    payload["commands"] = [
        ["python", "-m", "new_dataset_experiment.computational_pilot", "--run", "--seed", str(SIMULATION_SEED),
         "--population-seed", str(population_seed),
         "--condition", cell.condition_id, "--inputs", str(inputs_root.resolve()),
         "--output", str(output_root.resolve()), *labels_arguments]
        for cell in cells()
    ]
    payload["preparation_commands"] = [
        {"stage": "inputs", "effect": "Local artifact preparation from the frozen cohort; never resample based on labels.",
         "command": ["python", "-m", "new_dataset_experiment.prepare_models", "--stage", "inputs", "--inputs", str(inputs_root.resolve()), "--cohort", str(cohort_path.resolve()), "--population-seed", str(population_seed), "--run"]},
        {"stage": "sasrec", "effect": "Local training; preserve the shared initial checkpoint for all six cells.",
         "command": ["python", "-m", "new_dataset_experiment.prepare_models", "--stage", "sasrec", "--inputs", str(inputs_root.resolve()), "--population-seed", str(population_seed), "--seed", str(SIMULATION_SEED), "--device", "cuda", "--run"]},
        {"stage": "embeddings", "effect": "One-off paid embedding API calls shared by P3/P4; requires explicit --run authorization.",
         "command": ["python", "-m", "new_dataset_experiment.prepare_models", "--stage", "embeddings", "--inputs", str(inputs_root.resolve()), "--population-seed", str(population_seed), "--run"]},
        {"stage": "ncm", "effect": "Local response-model training; no LLM calls.",
         "command": ["python", "-m", "new_dataset_experiment.prepare_models", "--stage", "ncm", "--inputs", str(inputs_root.resolve()), "--population-seed", str(population_seed), "--seed", str(SIMULATION_SEED), "--variant", "P3", "--variant", "P4", "--device", "cuda", "--run"]},
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--population-seed", type=int,
                        help="Selected cohort seed; check/run otherwise derive it from prepared inputs.")
    parser.add_argument("--condition")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT,
                        help="Frozen cohort recorded in an offline preparation plan.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--write-plan", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        design = describe(seed=SIMULATION_SEED if args.seed is None else args.seed,
                          population_seed=POPULATION_SEED if args.population_seed is None else args.population_seed)
        selected = tuple(cell for cell in cells() if not args.condition or cell.condition_id == args.condition)
        if not selected:
            raise ValueError("condition is not in the six-cell computational pilot")
        if args.run:
            if args.seed is None or args.condition is None or len(selected) != 1:
                raise ValueError("--run requires one explicit --seed 42 and --condition")
            if args.check or args.write_plan:
                raise ValueError("--run cannot be combined with --check or --write-plan")
            extra = {} if args.population_seed is None else {"population_seed": args.population_seed}
            campaign.launch(args.inputs, args.output, SIMULATION_SEED, selected[0],
                            device=args.device, labels_path=args.labels, **extra)
            return 0
        if args.write_plan:
            write_plan(args.write_plan, inputs_root=args.inputs, output_root=args.output,
                       labels_path=args.labels, population_seed=design["population_seed"], cohort_path=args.cohort)
        if args.check:
            report = preflight_all(args.inputs, labels_path=args.labels, population_seed=args.population_seed)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if report["ready"] else 2
        print(json.dumps(design, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
