"""Prepare a frozen-population L0--L3 campaign without starting it.

The default command only describes the design. ``--prepare-only`` validates and
copies existing control trajectories; only the explicit ``--run-queue`` action
can invoke the simulator. Copied trajectories retain their original identity
and are stage counterparts, never additional independent replications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_NAME = "new_sampling_p1054_seed42_a300_r0_e804_v1"
DEFAULT_INPUT_MANIFEST = (
    REPOSITORY_ROOT / "feedback_loop_experiment" / "inputs" / CAMPAIGN_NAME
    / "inputs_manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "feedback_loop_experiment" / "outputs" / CAMPAIGN_NAME
)
EXPERIMENT_ID = "feedback_new_sampling_p1054_a300_r0_e804"
SEED = 42
POPULATION_SEED = 1054
POPULATION_SEEDS = {42: 1054, 43: 19277, 44: 18844}
COMPUTATIONAL_MODELS = (
    "position_biased", "ncm_inspired_gru_uniform", "ncm_inspired_gru_mansoury",
)
INPUT_NAMES = ("prepared_data", "population", "profiles", "initial_recommender")
FINGERPRINT_NAMES = ("dataset", "population", "profiles", "initial_recommender")


def _condition(stage: str, model: str, suffix: str = "") -> str:
    return f"{stage}__{model}__sasrec{suffix}__order_probe"


def campaign_name(seed: int = SEED) -> str:
    return f"new_sampling_p{POPULATION_SEEDS[seed]}_seed{seed}_a300_r0_e804_v1"


def input_identity(inputs: Mapping[str, Any]) -> tuple[int, int]:
    """Retain the original seed-42 API while checking every explicit pair."""
    seed = inputs.get("simulation_seed", SEED)
    population = inputs.get("population_seed", POPULATION_SEEDS.get(seed))
    if seed not in POPULATION_SEEDS or population != POPULATION_SEEDS[seed]:
        raise ValueError("unsupported simulation/population seed pair")
    return seed, population


def queue_conditions(seed: int = SEED) -> dict[int, tuple[str, ...]]:
    """The thirteen uncompleted cells, retaining the original three probe arms."""
    agent = "agentic_recommender"
    if seed not in POPULATION_SEEDS:
        raise ValueError(f"unsupported simulation seed: {seed}")
    if seed != SEED:
        # Separate long jobs so a combined plan can start them together. The
        # seed-42 layout below remains unchanged for archived launch records.
        paid = [
            _condition(stage, agent)
            for stage in ("L0_control", "L1_feature", "L2_ml_model")
        ] + [
            _condition("L3_individual", agent, suffix)
            for suffix in ("__no_reflection_control", "__fixed_persona_memory_only")
        ]
        if seed == 44:
            paid.append(_condition("L3_individual", agent, "__full_reflection"))
        paid.extend(_condition(stage, "direct_llm") for stage in (
            ("L0_control", "L1_feature", "L2_ml_model") if seed == 44
            else ("L0_control", "L1_feature")
        ))
        queues = {number: (condition,) for number, condition in enumerate(paid, 1)}
        queues[len(queues) + 1] = tuple(
            _condition(stage, model)
            for model in COMPUTATIONAL_MODELS
            for stage in (("L0_control", "L1_feature", "L2_ml_model")
                          if seed == 44 else ("L0_control", "L1_feature"))
        )
        return queues
    return {
        1: (
            _condition("L0_control", agent),
            _condition("L3_individual", agent, "__no_reflection_control"),
        ),
        2: (
            _condition("L1_feature", agent),
            _condition("L3_individual", agent, "__fixed_persona_memory_only"),
        ),
        3: (
            _condition("L2_ml_model", agent),
            _condition("L0_control", "direct_llm"),
            _condition("L1_feature", "direct_llm"),
        ),
        4: tuple(
            _condition(stage, model)
            for model in COMPUTATIONAL_MODELS
            for stage in ("L0_control", "L1_feature")
        ),
    }


def reused_conditions() -> dict[str, str]:
    """Original mitigation model names mapped to their feedback-stage counterparts."""
    return {
        **{model: _condition("L2_ml_model", model) for model in COMPUTATIONAL_MODELS},
        "direct_llm": _condition("L2_ml_model", "direct_llm"),
        "agentic_recommender": _condition(
            "L3_individual", "agentic_recommender", "__full_reflection"
        ),
    }


def default_source_runs(seed: int = SEED) -> dict[str, Path]:
    if seed not in POPULATION_SEEDS:
        raise ValueError(f"unsupported simulation seed: {seed}")
    if seed == 44:
        return {}
    base = REPOSITORY_ROOT / "mitigation_experiment" / "outputs"
    if seed == 43:
        base = base / "sasrec_budget_p19277_seed43_a300_r0_e804_b128_split4_v1"
        queues = {
            "position_biased": "queue_4_other",
            "ncm_inspired_gru_uniform": "queue_4_other",
            "ncm_inspired_gru_mansoury": "queue_4_other",
            "direct_llm": "queue_4_other",
            "agentic_recommender": "queue_1_agent_control",
        }
        return {
            model: base / queue / f"{model}__control" /
            f"mitigation_v1_primary_p19277_a300_r0__{model}__control__seed_43"
            for model, queue in queues.items()
        }
    computational = base / "sasrec_budget_p1054_seed42_a300_r0_e804_b128_split3_v1"
    llm = base / "sasrec_budget_p1054_seed42_a300_r0_e804_b128_llm_split4_v1"
    result: dict[str, Path] = {}
    for number, model in enumerate(COMPUTATIONAL_MODELS, 1):
        result[model] = (
            computational / f"queue_{number}" / f"{model}__control"
            / f"mitigation_v1_primary_p1054_a300_r0__{model}__control__seed_42"
        )
    for model, queue in (("agentic_recommender", "queue_1_agent_control"),
                         ("direct_llm", "queue_4_direct_llm")):
        result[model] = (
            llm / queue / f"{model}__control"
            / f"mitigation_v1_primary_p1054_a300_r0__{model}__control__seed_42"
        )
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _relative_file(root: Path, relative: str) -> Path:
    # Completion manifests use Windows separators even on another test platform.
    path = root.joinpath(*relative.replace("\\", "/").split("/")).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"manifest file is missing or outside its run: {relative}")
    return path


def _require(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, found {actual!r}")


def read_inputs_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    paths = manifest.get("paths", {})
    for name in INPUT_NAMES:
        configured = paths.get(name)
        if not configured or not Path(configured).exists():
            raise ValueError(f"input manifest lacks an existing {name} path")
    fingerprints = manifest.get("fingerprints", {})
    for name in FINGERPRINT_NAMES:
        value = fingerprints.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"input manifest lacks {name} fingerprint")
    _require(manifest.get("input_format", "mitigation"), "mitigation", "input format")
    settings = manifest.get("provider_settings", {})
    _require(settings.get("model"), "gpt-5.4-nano", "provider model")
    _require(settings.get("reasoning_effort"), "none", "provider reasoning")
    _require(settings.get("max_output_tokens"), 2056, "provider output cap")
    input_identity(manifest)
    for name, expected in {
        "active_users": 300, "reserve_users": 0, "training_users": 300,
        "iterations": 30, "training_scope": "active_reserve",
    }.items():
        if name in manifest:
            _require(manifest[name], expected, name)
    if "recommender_settings" in manifest:
        _require(manifest["recommender_settings"],
                 {"device": "cuda", "num_epochs": 804, "batch_size": 128},
                 "recommender settings")
    for relative, expected in manifest.get("file_sha256", {}).items():
        _require(file_sha256(_relative_file(path.parent, relative)), expected,
                 f"copied input checksum {relative}")
    return manifest


def validate_completed_source(
    source: Path, model: str, inputs: Mapping[str, Any],
    *, reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify completion, content hashes and the scientific control contract."""
    manifest = _read_json(source / "manifest.json")
    complete = _read_json(source / "_COMPLETE.json")
    _require(complete.get("status"), "complete", "completion status")
    _require(complete.get("completed_iteration"), 30, "completed iterations")
    _require(complete.get("run_id"), manifest.get("run_id"), "original run identity")
    hashes = complete.get("file_sha256", {})
    for required in ("manifest.json", "events.jsonl", "diagnostics.jsonl"):
        if required not in hashes:
            raise ValueError(f"completion manifest lacks checksum for {required}")
    for relative, expected in hashes.items():
        _require(file_sha256(_relative_file(source, relative)), expected,
                 f"source checksum {relative}")
    checkpoint = _relative_file(source, complete["final_checkpoint_path"])
    _require(file_sha256(checkpoint), complete.get("final_checkpoint_sha256"),
             "final checkpoint checksum")
    resolved = manifest["resolved_manifest"]
    seed, population_seed = input_identity(inputs)
    expected_values = {
        "simulation_seed": seed, "population_seed": population_seed,
        "active_users": 300, "reserve_users": 0, "training_users": 300,
        "iterations": 30, "slate_size": 10, "training_scope": "active_reserve",
        "recommender_fit_interval": 1, "checkpoint_interval": 1,
        "initial_recommender_restored": True,
        "presentation_arms": ["canonical", "replica", "shuffled"],
    }
    for name, expected in expected_values.items():
        _require(resolved.get(name), expected, name)
    for name, expected in {"num_epochs": 804, "batch_size": 128}.items():
        _require(resolved.get("recommender_settings", {}).get(name), expected, name)
    fingerprints = inputs["fingerprints"]
    for name in ("dataset", "population", "initial_recommender"):
        _require(resolved["inputs"].get(name), fingerprints[name], f"input {name}")
    if model == "agentic_recommender":
        _require(resolved["inputs"].get("profiles"), fingerprints["profiles"],
                 "input profiles")
    condition = resolved["condition"]
    for name, expected in {
        "user_model": model, "recommender": "sasrec", "intervention": "control",
        "candidate_policy": "exclude_seen", "population_policy": "fixed_active_users",
        "campaign_policy": "no_campaign", "presentation_policy": "mitigation_order_probes",
        "feedback_loop": "L3_individual" if model == "agentic_recommender" else "L2_ml_model",
        "user_update_policy": "full_reflection" if model == "agentic_recommender" else "frozen",
    }.items():
        _require(condition.get(name), expected, f"control {name}")
    if model in {"direct_llm", "agentic_recommender"}:
        for name, expected in inputs["provider_settings"].items():
            _require(resolved.get("provider", {}).get(name), expected, f"provider {name}")
    expected_settings = inputs.get("user_model_settings", {}).get(model, {})
    for name, expected in expected_settings.items():
        if not name.endswith("_directory") and not name.endswith("_path"):
            _require(resolved.get("user_model_settings", {}).get(name), expected,
                     f"user model {name}")
    mitigation = resolved.get("mitigation", {})
    _require(mitigation.get("intervention"), "control", "mitigation intervention")
    if reference:
        for name in ("head_definition", "head_definition_fingerprint", "head_item_ids",
                     "head_count", "initial_history_lengths"):
            if name in reference:
                _require(mitigation.get(name), reference[name], f"reference {name}")
    input_reference = inputs.get("reference", {})
    if "head_definition_fingerprint" in input_reference:
        _require(mitigation.get("head_definition_fingerprint"),
                 input_reference["head_definition_fingerprint"], "head definition")
    # Dataset semantic identity above includes the catalog; avoid inventing a
    # second catalog definition from the head/tail partition alone.
    return {"manifest": manifest, "complete": complete, "reference": mitigation}


def _tree_inventory(directory: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is not permitted in an archived run: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = file_sha256(path)
    return files


def _write_json_exclusive(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def describe_campaign(seed: int = SEED) -> dict[str, Any]:
    reuse = reused_conditions() if seed in (42, 43) else {}
    queues = queue_conditions(seed)
    return {
        "campaign": campaign_name(seed), "simulation_seed": seed,
        "population_seed": POPULATION_SEEDS[seed], "active_users": 300, "reserve_users": 0,
        "iterations": 30, "sasrec_epochs": 804, "sasrec_batch_size": 128,
        "presentation_arms": ["canonical", "replica", "shuffled"],
        "retained_cells": 18, "reused_completed_cells": len(reuse),
        "new_cells": sum(map(len, queues.values())),
        "excluded_response_models": ["pointwise_logistic"],
        "queues": {str(number): list(cells) for number, cells in queues.items()},
        "reused_stage_counterparts": reuse,
        "note": "Reuse preserves original run identity and adds no independent replicate.",
    }


def prepare_campaign(
    input_manifest: Path, output_root: Path,
    *, source_runs: Mapping[str, Path] | None = None,
) -> Path:
    """Validate then copy completed controls into a fresh, non-overwriting folder."""
    if output_root.exists():
        raise FileExistsError(f"campaign destination already exists: {output_root}")
    inputs = read_inputs_manifest(input_manifest)
    seed, _ = input_identity(inputs)
    design = describe_campaign(seed)
    sources = dict(default_source_runs(seed) if source_runs is None else source_runs)
    _require(set(sources), set(design["reused_stage_counterparts"]), "reused model selection")
    for source in sources.values():
        if output_root.resolve().is_relative_to(source.resolve()):
            raise ValueError("campaign destination must not be inside a source run")
    validated: dict[str, dict[str, Any]] = {}
    reference = None
    for model, source in sources.items():
        result = validate_completed_source(source, model, inputs, reference=reference)
        reference = result["reference"] if reference is None else reference
        validated[model] = result
    # Resolve every condition before any mutation, without importing a provider.
    from .experiment_methodology import CONDITIONS_BY_ID

    conditions = [cell for cells in queue_conditions(seed).values() for cell in cells]
    conditions.extend(design["reused_stage_counterparts"].values())
    _require(len(conditions), 18, "retained cell count")
    _require(len(set(conditions)), 18, "unique retained cell count")
    for condition in conditions:
        if condition not in CONDITIONS_BY_ID:
            raise ValueError(f"unregistered campaign condition: {condition}")
    output_root.mkdir(parents=True, exist_ok=False)
    aliases = []
    for model, source in sources.items():
        original_id = validated[model]["manifest"]["run_id"]
        destination = output_root / "reused_controls" / original_id
        source_inventory = _tree_inventory(source)
        shutil.copytree(source, destination)
        _require(_tree_inventory(destination), source_inventory, "copied run content")
        aliases.append({
            "target_condition_id": reused_conditions()[model],
            "source_run_id": original_id, "copied_run_id": original_id,
            "source_directory": str(source.resolve()),
            "copied_directory": str(destination.resolve()),
            "relationship": "existing_control_stage_counterpart",
            "independent_replicate": False, "file_sha256": source_inventory,
        })
    _write_json_exclusive(output_root / "reused_control_aliases.json", {
        "schema": "feedback-stage-counterparts-v1", "aliases": aliases,
        "interpretation": "Existing trajectories copied byte-for-byte; count each original system once.",
    })
    for number in queue_conditions(seed):
        (output_root / f"queue_{number}").mkdir()
    _write_json_exclusive(output_root / "campaign_manifest.json", {
        **design, "schema": "feedback-new-sampling-campaign-v1",
        "status": "prepared_not_started", "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_manifest.resolve()),
        "input_manifest_sha256": file_sha256(input_manifest),
        "output_root": str(output_root.resolve()),
        "source_input_fingerprints": inputs["fingerprints"],
        "reference": {name: (reference or inputs.get("reference", {}))[name] for name in (
            "head_definition", "head_definition_fingerprint", "head_count"
        ) if name in (reference or inputs.get("reference", {}))},
    })
    return output_root


def queue_command(
    condition_id: str, queue_root: Path, input_manifest: Mapping[str, Any],
) -> list[str]:
    """Build an explicit-input command; never trigger automatic population sampling."""
    model = condition_id.split("__")[1]
    seed, population_seed = input_identity(input_manifest)
    experiment = (EXPERIMENT_ID if seed == SEED else
                  f"feedback_new_sampling_p{population_seed}_a300_r0_e804")
    paths = input_manifest["paths"]
    command = [
        sys.executable, "-m", "feedback_loop_experiment.run_simulator",
        "--experiment", experiment, "--condition", condition_id,
        "--seed", str(seed), "--population-seed", str(population_seed),
        "--iterations", "30", "--slate-size", "10", "--active-users", "300",
        "--reserve-users", "0", "--input-format", "mitigation",
        "--prepared-data", str(paths["prepared_data"]),
        "--population", str(paths["population"]),
        "--initial-recommender", str(paths["initial_recommender"]),
        "--output-root", str(queue_root), "--training-scope", "active-reserve",
        "--recommender-settings", json.dumps({
            "num_epochs": 804, "batch_size": 128, "device": "cuda"
        }),
        "--user-model-settings", json.dumps(
            input_manifest.get("user_model_settings", {}).get(model, {})
        ),
        "--policy-settings", "{}", "--provider", "openai",
        "--provider-settings", json.dumps(input_manifest["provider_settings"]),
        "--recommender-fit-interval", "1", "--checkpoint-interval", "1",
    ]
    if model == "agentic_recommender":
        command.extend(["--profiles", str(paths["profiles"])])
    return command


def run_queue(output_root: Path, number: int) -> None:
    """Explicit future launch entry point; preparation never calls this function."""
    campaign = _read_json(output_root / "campaign_manifest.json")
    seed = campaign["simulation_seed"]
    _require(campaign.get("schema"), "feedback-new-sampling-campaign-v1", "campaign schema")
    _require(campaign.get("population_seed"), POPULATION_SEEDS[seed], "campaign population seed")
    _require(campaign.get("queues"), describe_campaign(seed)["queues"], "queued scientific design")
    input_path = Path(campaign["input_manifest"])
    _require(file_sha256(input_path), campaign["input_manifest_sha256"], "input manifest checksum")
    inputs = read_inputs_manifest(input_path)
    _require(input_identity(inputs), (seed, POPULATION_SEEDS[seed]), "queue seed identity")
    if number not in queue_conditions(seed):
        raise ValueError(f"unknown queue {number} for seed {seed}")
    queue_root = output_root / f"queue_{number}"
    _write_json_exclusive(queue_root / "_QUEUE_STARTED.json", {
        "queue": number, "started_at": datetime.now(timezone.utc).isoformat(),
        "explicit_action": "run-queue", "conditions": list(queue_conditions(seed)[number]),
    })
    for condition in queue_conditions(seed)[number]:
        subprocess.run(queue_command(condition, queue_root, inputs),
                       cwd=REPOSITORY_ROOT, check=True)
    _write_json_exclusive(queue_root / "_QUEUE_COMPLETE.json", {
        "queue": number, "completed_at": datetime.now(timezone.utc).isoformat(),
        "conditions": list(queue_conditions(seed)[number]),
    })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--describe", action="store_true")
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run-queue", type=int)
    parser.add_argument("--seed", type=int, choices=tuple(POPULATION_SEEDS), default=SEED)
    parser.add_argument("--inputs-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args(argv)
    input_path = arguments.inputs_manifest or (
        REPOSITORY_ROOT / "feedback_loop_experiment/inputs" /
        campaign_name(arguments.seed) / "inputs_manifest.json"
    )
    output_path = arguments.output_root or (
        REPOSITORY_ROOT / "feedback_loop_experiment/outputs" / campaign_name(arguments.seed)
    )
    if arguments.prepare_only:
        _require(input_identity(read_inputs_manifest(input_path))[0], arguments.seed, "requested seed")
        print(prepare_campaign(input_path, output_path))
    elif arguments.run_queue is not None:
        _require(_read_json(output_path / "campaign_manifest.json")["simulation_seed"],
                 arguments.seed, "requested seed")
        run_queue(output_path, arguments.run_queue)
    else:
        print(json.dumps(describe_campaign(arguments.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
