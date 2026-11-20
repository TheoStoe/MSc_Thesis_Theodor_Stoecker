"""Describe and explicitly launch the frozen Goodreads comparison.

The default command only describes the design. No data acquisition, model
training, profile generation or experiment launch occurs on import or describe.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = ROOT / "inputs" / "goodreads_a300_v1"
DEFAULT_OUTPUTS = ROOT / "outputs"
SIMULATION_SEEDS = (42, 43, 44)
POPULATION_SEED = 1054
NCM_FOLDERS = {
    "pointwise_logistic": "P1",
    "ncm_inspired_gru_uniform": "P3",
    "ncm_inspired_gru_mansoury": "P4",
}
COMPUTATIONAL_MODELS = (
    "position_biased", "ncm_inspired_gru_uniform", "ncm_inspired_gru_mansoury",
)
LLM_MODELS = ("direct_llm", "agentic_recommender")
PROVIDER_SETTINGS = {
    "model": "gpt-5.4-nano", "reasoning_effort": "none",
    "max_output_tokens": 2056,
}


@dataclass(frozen=True)
class BookCell:
    user_model: str
    stage: str
    update_policy: str = "frozen"
    probes: bool = True

    @property
    def condition_id(self) -> str:
        parts = [self.stage, self.user_model, "sasrec"]
        if self.update_policy == "full_reflection":
            parts.append("full_reflection")
        if self.probes:
            parts.append("order_probe")
        return "__".join(parts)


def cells(*, probes: bool = True) -> tuple[BookCell, ...]:
    """Match ordinary L2 arms to L0 and reflective-agent L3 to frozen-agent L2."""
    result = [
        BookCell(model, stage, probes=probes)
        for model in (*COMPUTATIONAL_MODELS, "direct_llm")
        for stage in ("L0_control", "L2_ml_model")
    ]
    result.extend((
        BookCell("agentic_recommender", "L0_control", probes=probes),
        BookCell("agentic_recommender", "L2_ml_model", probes=probes),
        BookCell("agentic_recommender", "L3_individual", "full_reflection", probes),
    ))
    return tuple(result)


def comparator_stage(user_model: str) -> str:
    """Return the prespecified within-family comparator without heavy imports."""
    if user_model == "agentic_recommender":
        return "L2_ml_model"
    if user_model in (*COMPUTATIONAL_MODELS, "direct_llm"):
        return "L0_control"
    raise ValueError(f"user model is not in the selected book campaign: {user_model}")


def comparisons() -> list[dict[str, str]]:
    """Describe distinct estimands rather than treating all contrasts as L2-L0."""
    common = [
        {
            "user_model": model,
            "treatment_stage": "L2_ml_model",
            "comparator_stage": "L0_control",
            "estimand": "Combined public-history update and recommender-refitting effect beyond L0 candidate depletion; agent state is frozen at both stages.",
        }
        for model in (*COMPUTATIONAL_MODELS, "direct_llm", "agentic_recommender")
    ]
    return common + [{
        "user_model": "agentic_recommender",
        "treatment_stage": "L3_individual",
        "comparator_stage": "L2_ml_model",
        "estimand": "Joint persistent private memory, reflection, and profile adaptation beyond frozen-agent L2.",
    }]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_population_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("population seed must be a non-negative integer")
    return value


def prepared_population_seed(inputs: Any, *, expected: int | None = None) -> int:
    """Check selection provenance rather than assigning a seed to a saved cohort."""
    actual = _validated_population_seed(inputs.population.population_seed)
    sampling = inputs.dataset.preparation_settings.get("sampling", {})
    recorded = [sampling[key] for key in ("population_seed", "seed") if key in sampling]
    if not recorded:
        raise ValueError("prepared dataset is missing its cohort selection seed")
    if any(_validated_population_seed(value) != actual for value in recorded):
        raise ValueError("prepared dataset cohort selection seed differs from the frozen population seed")
    if expected is not None and _validated_population_seed(expected) != actual:
        raise ValueError("requested population seed differs from the frozen cohort")
    return actual


def describe(*, seeds: Sequence[int] = (42,), probes: bool = True,
             population_seed: int = POPULATION_SEED) -> dict[str, Any]:
    seeds = tuple(dict.fromkeys(seeds))
    if not seeds or any(isinstance(seed, bool) or seed not in SIMULATION_SEEDS for seed in seeds):
        raise ValueError("select simulation seeds 42, 43 and/or 44")
    selected = cells(probes=probes)
    return {
        "schema_version": "goodreads-campaign-v1",
        "status": "configured_not_started",
        "dataset": "UCSD Goodreads with VIAF authority metadata for evaluation",
        "population_seed": _validated_population_seed(population_seed),
        "simulation_seeds": list(seeds),
        "active_users": 300, "reserve_users": 0, "background_training_users": 0,
        "iterations": 30, "slate_size": 10,
        "recommender": {"name": "sasrec", "num_epochs": 804,
                        "batch_size": 128, "max_sequence_length": 200},
        "provider": {"provider": "openai", **PROVIDER_SETTINGS},
        "presentation_arms": ["canonical", "replica", "shuffled"] if probes else ["canonical"],
        "canonical_only_feedback": True,
        "recommendation_quality_metrics": [],
        "cells_per_seed": len(selected), "trajectories": len(selected) * len(seeds),
        "conditions": [{**asdict(cell), "condition_id": cell.condition_id} for cell in selected],
        "comparisons": comparisons(),
        "interpretation": {
            "fixed_population": "All simulation seeds share one frozen 300-reader cohort.",
            "L0": "Public history and recommender are frozen; consumed items remain excluded. No agent L0 is selected.",
            "L2_vs_L0": "Combined history-update and recommender-refitting effect beyond candidate depletion.",
            "agent_L2": "Public interaction history updates and SASRec refits; private interaction memory, reflection state and persona remain frozen.",
            "agent_L3_vs_L2": "Joint persistent private memory, reflection and profile-adaptation effect; reflection alone is not isolated.",
            "between_families": "All response families have an L2 condition with public-history updates and recommender refitting. Agent L3 adds the complete private adaptation mechanism.",
            "different_contrasts": "L2-L0 effects for other families and L3-L2 for the agent answer different questions and must not be ranked as the same effect.",
        },
    }


def write_plan(destination: Path, *, seeds: Sequence[int], inputs_root: Path,
               output_root: Path, probes: bool = True,
               population_seed: int = POPULATION_SEED,
               item_summaries_path: Path | None = None,
               profiles_path: Path | None = None) -> dict[str, Any]:
    payload = describe(seeds=seeds, probes=probes, population_seed=population_seed)
    payload.update({"inputs_root": str(inputs_root.resolve()),
                    "output_root": str(output_root.resolve())})
    if item_summaries_path is not None:
        payload["llm_item_summaries_path"] = str(item_summaries_path.resolve())
    if profiles_path is not None:
        payload["llm_profiles_path"] = str(profiles_path.resolve())
    payload["commands"] = [
        ["python", "-m", "new_dataset_experiment.campaign", "--run",
         "--seed", str(seed), "--population-seed", str(population_seed), "--condition", cell.condition_id,
         "--inputs", str(inputs_root.resolve()), "--output", str(output_root.resolve()),
         *(["--item-summaries", str(item_summaries_path.resolve())]
           if item_summaries_path is not None and cell.user_model in LLM_MODELS else []),
         *(["--profiles", str(profiles_path.resolve())]
           if profiles_path is not None and cell.user_model == "agentic_recommender" else []),
         *([] if probes else ["--without-probes"])]
        for seed in payload["simulation_seeds"] for cell in cells(probes=probes)
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return payload


def validate_reviewed_label_coverage(labels: dict[str, Any], training_histories: Any) -> dict[str, float]:
    """Recompute the readiness gate from reviewed items and actual train events.

    Summary fields in an annotation artifact are audit outputs, not substitutes
    for checking the item statuses and denominators used by this campaign.
    """
    if labels.get("authority_scan_complete") is not True:
        raise ValueError("a complete authority scan is required before the gender campaign is launched")
    known_genders = {"female", "male", "nonbinary", "other"}
    item_rows = labels.get("labels", {})
    if not item_rows:
        raise ValueError("author labels must contain the complete nonempty catalogue")
    reviewed_known = set()
    for key, row in item_rows.items():
        status, gender = row.get("status"), row.get("gender")
        if gender == "mixed":
            raise ValueError("mixed teams belong in secondary metadata, not the primary-author gender")
        if status not in {"reviewed", "provisional", "unknown"}:
            raise ValueError("every item label requires an explicit reviewed/provisional/unknown status")
        if gender not in {"unknown", "ambiguous"} and status != "reviewed":
            raise ValueError("non-unknown item gender labels require reviewed status")
        if status == "reviewed" and gender in known_genders:
            reviewed_known.add(int(key))
    training_count, known_training = 0, 0
    item_ids = {int(key) for key in item_rows}
    for history in training_histories.values():
        for item_id in history:
            if item_id not in item_ids:
                raise ValueError("training interaction references an item missing from author labels")
            training_count += 1
            known_training += item_id in reviewed_known
    if not training_count:
        raise ValueError("author-label coverage requires nonempty training histories")
    actual = {
        "known_book_share": len(reviewed_known) / len(item_rows),
        "known_training_interaction_share": known_training / training_count,
    }
    declared = labels.get("coverage", {})
    for metric, share in actual.items():
        value = declared.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"author-label {metric} must be a finite share between zero and one")
        if not math.isclose(value, share, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"declared {metric} disagrees with reviewed item labels and training interactions")
    if min(actual.values()) < 0.8:
        raise ValueError("provisional 80% book/interaction coverage criterion is not met; review and version the design before launch")
    return actual


def validate_label_coverage(labels: dict[str, Any], training_histories: Any) -> dict[str, float]:
    """Dispatch explicitly versioned policies; legacy manual-review rules persist."""
    if "annotation_policy" in labels:
        from .gender_labels import validate_policy_labels
        return validate_policy_labels(labels, training_histories)
    return validate_reviewed_label_coverage(labels, training_histories)


def preflight(inputs_root: Path, seed: int, cell: BookCell, *, labels_path: Path | None = None,
              validate: bool = True, population_seed: int | None = None,
              item_summaries_path: Path | None = None,
              profiles_path: Path | None = None) -> dict[str, Any]:
    """Report missing artifacts without constructing providers or training models."""
    describe(seeds=(seed,), probes=cell.probes,
             population_seed=POPULATION_SEED if population_seed is None else population_seed)
    actual_population_seed = None
    item_text_fingerprint = None
    profile_root = inputs_root / "profiles" if profiles_path is None else profiles_path
    option_problems = []
    if item_summaries_path is not None and cell.user_model not in LLM_MODELS:
        option_problems.append("Item-summary overlays apply only to LLM response models")
    if profiles_path is not None and cell.user_model != "agentic_recommender":
        option_problems.append("Extracted profiles apply only to agent conditions")
    required = {
        "prepared dataset": inputs_root / "prepared" / "_COMPLETE.json",
        "frozen population": inputs_root / "population.json",
        "fresh initial SASRec": inputs_root / "initial_sasrec" / f"seed_{seed}",
        "initial slate audit": inputs_root / "initial_sasrec" / f"seed_{seed}" / "baseline_slates.json",
    }
    if cell.user_model == "agentic_recommender":
        required["book-domain profiles"] = profile_root / "_COMPLETE.json"
    if item_summaries_path is not None:
        required["LLM item summaries"] = item_summaries_path / "_COMPLETE.json"
    if cell.user_model in NCM_FOLDERS:
        required["book-domain embeddings"] = inputs_root / "embeddings" / "manifest.json"
        required["fresh book-domain response model"] = (
            inputs_root / "ncm" / f"seed_{seed}" / NCM_FOLDERS[cell.user_model] / "checkpoint"
        )
    if labels_path is None:
        labels_path = inputs_root / "item_labels.json"
    required["versioned author labels"] = labels_path
    problems = option_problems + [f"Missing {name}: {path}" for name, path in required.items() if not path.exists()]
    if not problems and validate:
        try:
            from .runtime_adapter import load_book_inputs
            inputs = load_book_inputs(
                inputs_root / "prepared", inputs_root / "population.json",
                profile_root if cell.user_model == "agentic_recommender" else None,
                inputs_root / "initial_sasrec" / f"seed_{seed}",
            )
            if cell.user_model == "agentic_recommender" and item_summaries_path is None:
                profile_manifest = profile_root / "manifest.json"
                if profile_manifest.is_file():
                    metadata = json.loads(profile_manifest.read_text(encoding="utf-8")).get("metadata", {})
                    if isinstance(metadata, dict) and "llm_item_text" in metadata:
                        raise ValueError("summary-based agent profiles require their matching item-summary overlay")
            if item_summaries_path is not None:
                from .item_summaries import load_item_summaries
                overlay = load_item_summaries(item_summaries_path, prepared_path=inputs_root / "prepared")
                if (overlay["manifest"]["source_dataset_fingerprint"] != inputs.dataset.dataset_fingerprint
                        or set(overlay["item_text_by_id"]) != set(inputs.dataset.catalog)):
                    raise ValueError("LLM item summaries differ from the prepared catalogue")
                item_text_fingerprint = overlay["manifest"]["item_text_fingerprint"]
                if cell.user_model == "agentic_recommender":
                    from .prepare_profiles import validate_profile_item_summaries
                    validate_profile_item_summaries(
                        inputs_root, profile_root, item_summaries_path,
                        expected_item_text_fingerprint=item_text_fingerprint,
                    )
            active = [member for member in inputs.population.members if member.role.value == "active"]
            if len(active) != 300 or len(inputs.population.members) != 300:
                raise ValueError("primary campaign requires exactly 300 active users and no other users")
            actual_population_seed = prepared_population_seed(inputs, expected=population_seed)
            if len(inputs.dataset.training_histories) != 300:
                raise ValueError("SASRec training cohort must be exactly the same 300 readers")
            for history in inputs.dataset.training_histories.values():
                if len(inputs.dataset.catalog) - len(set(history)) < 40:
                    raise ValueError("candidate pool cannot support 30 unique clicks and a top-10 slate")
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            if labels.get("schema_version") != "goodreads-item-labels-v1":
                raise ValueError("unsupported author-label artifact schema")
            if labels.get("label_semantics_version") != "goodreads-primary-literary-author-recorded-gender-v1":
                raise ValueError("the primary campaign requires recorded primary-literary-author labels")
            if "annotation_policy" not in labels and not labels.get("linkage_reviewed", False):
                raise ValueError("author linkage must be reviewed before the gender campaign is launched")
            label_ids = {int(key) for key in labels.get("labels", {})}
            if label_ids != set(inputs.dataset.catalog):
                raise ValueError("label artifact must cover the complete catalogue, including unknown items")
            if labels.get("dataset_fingerprint") not in (None, inputs.dataset.dataset_fingerprint):
                raise ValueError("author-label artifact belongs to a different prepared dataset")
            source_hash = inputs.dataset.source_file_fingerprints.get("cohort_json_sha256")
            if not source_hash or labels.get("cohort_fingerprint") != source_hash:
                raise ValueError("author-label artifact does not match the frozen cohort file")
            for label in labels["labels"].values():
                if label.get("gender") not in {"female", "male", "nonbinary", "unknown", "ambiguous", "mixed", "other"}:
                    raise ValueError("unrecognised item gender label")
            validate_label_coverage(labels, inputs.dataset.training_histories)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            problems.append(f"Artifact validation failed: {exc}")
    return {"condition_id": cell.condition_id, "simulation_seed": seed,
            "population_seed": actual_population_seed,
            "ready": not problems, "problems": problems,
            "llm_item_text_fingerprint": item_text_fingerprint,
            "required": {name: str(path) for name, path in required.items()}}


def launch(inputs_root: Path, output_root: Path, seed: int, cell: BookCell,
           *, device: str = "cuda", labels_path: Path | None = None,
           population_seed: int | None = None,
           item_summaries_path: Path | None = None,
           profiles_path: Path | None = None) -> Any:
    readiness = preflight(inputs_root, seed, cell, labels_path=labels_path,
                          population_seed=population_seed, item_summaries_path=item_summaries_path,
                          profiles_path=profiles_path)
    if not readiness["ready"]:
        raise ValueError("Run is not ready:\n" + "\n".join(readiness["problems"]))
    # All heavy imports and provider construction stay behind explicit launch.
    from .engine.experiment_methodology import CONDITIONS_BY_ID
    from .engine.run_simulator import run_experiment
    from .runtime_adapter import build_book_config

    condition = CONDITIONS_BY_ID[cell.condition_id]
    settings: dict[str, Any] = {}
    if cell.user_model == "position_biased":
        settings["alpha"] = -1.0
    elif cell.user_model in NCM_FOLDERS:
        settings.update({
            "checkpoint_directory": str(inputs_root / "ncm" / f"seed_{seed}" /
                                        NCM_FOLDERS[cell.user_model] / "checkpoint"),
            "embedding_directory": str(inputs_root / "embeddings"),
        })
    if item_summaries_path is not None:
        settings.update({"llm_item_text_artifact_path": str(item_summaries_path.resolve()),
                         "llm_item_text_fingerprint": readiness["llm_item_text_fingerprint"]})
    profile_override = ({"profile_artifact_path": profiles_path}
                        if profiles_path is not None else {})
    config = build_book_config(
        inputs_root, output_root, seed=seed, user_model=cell.user_model,
        experiment_id="goodreads_a300_e804_v1", population_seed=readiness["population_seed"],
        active_user_count=300, num_iterations=30, slate_size=10,
        provider_settings=PROVIDER_SETTINGS,
        user_model_settings=settings,
        recommender_settings={"device": device, "num_epochs": 804, "batch_size": 128,
                              "max_sequence_length": 200},
        **profile_override,
    )
    return run_experiment(config, condition, seed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--population-seed", type=int,
                        help="Selected cohort seed; check/run otherwise derive it from prepared inputs.")
    parser.add_argument("--condition")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--item-summaries", type=Path,
                        help="Verified source-description text for LLM conditions only.")
    parser.add_argument("--profiles", type=Path,
                        help="Shared agent profile bundle; defaults to INPUTS/profiles.")
    parser.add_argument("--without-probes", action="store_true")
    parser.add_argument("--write-plan", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    seeds = tuple(dict.fromkeys(args.seed or (42,)))
    selected = cells(probes=not args.without_probes)
    try:
        design = describe(seeds=seeds, probes=not args.without_probes,
                          population_seed=POPULATION_SEED if args.population_seed is None else args.population_seed)
        if args.condition:
            selected = tuple(cell for cell in selected if cell.condition_id == args.condition)
            if not selected:
                raise ValueError("condition is not in the selected thesis book campaign")
        if args.run:
            if len(seeds) != 1 or len(selected) != 1 or not args.condition:
                raise ValueError("--run requires exactly one explicit --seed and --condition")
            if args.write_plan or args.check:
                raise ValueError("--run cannot be combined with --check or --write-plan")
            launch(args.inputs, args.output, seeds[0], selected[0],
                   device=args.device, labels_path=args.labels, population_seed=args.population_seed,
                   item_summaries_path=args.item_summaries, profiles_path=args.profiles)
            return 0
        if args.write_plan:
            write_plan(args.write_plan, seeds=seeds, inputs_root=args.inputs,
                       output_root=args.output, probes=not args.without_probes,
                       population_seed=design["population_seed"],
                       item_summaries_path=args.item_summaries, profiles_path=args.profiles)
        if args.check:
            checks = [preflight(args.inputs, seed, cell, labels_path=args.labels,
                                population_seed=args.population_seed,
                                item_summaries_path=args.item_summaries, profiles_path=args.profiles)
                      for seed in seeds for cell in selected]
            print(json.dumps({"checks": checks, "ready": all(row["ready"] for row in checks)}, indent=2))
            return 0 if all(row["ready"] for row in checks) else 2
        print(json.dumps(design, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
