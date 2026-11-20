"""Create deterministic uniform and Mansoury slates for NCM training."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from ..utility.filesystem import rename_directory_with_retry
from ..utility.user_handling import PreparedDatasetArtifact


LOGGER = logging.getLogger(__name__)
SLATE_SIZE = 10
POLICIES = ("uniform", "mansoury")
PLACEMENT_SCHEMA = "feedback-loop-ncm-placement-v2"
GENERATOR_VERSION = "shared-negatives-policy-placement-v1"


def ensure_ncm_training_slates(
    dataset: PreparedDatasetArtifact,
    directory: str | Path,
    *,
    slate_seed: int = 42,
    mansoury_alpha: float = -1.0,
) -> Path:
    """Create the two training artifacts, or validate an existing cache."""

    if isinstance(slate_seed, bool) or not isinstance(slate_seed, int):
        raise TypeError("slate_seed must be an integer")
    if slate_seed < 0:
        raise ValueError("slate_seed cannot be negative")
    if not math.isfinite(mansoury_alpha) or mansoury_alpha >= 0:
        raise ValueError("mansoury_alpha must be finite and negative")

    root = Path(directory)
    if root.exists():
        _validate_cache(root, dataset.dataset_fingerprint, slate_seed, mansoury_alpha)
        LOGGER.info("using validated NCM slate cache at %s", root)
        return root

    staging = root.with_name(root.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(f"incomplete NCM slate artifact exists: {staging}")
    staging.mkdir(parents=True)

    probabilities = {
        "uniform": tuple(1.0 / SLATE_SIZE for _ in range(SLATE_SIZE)),
        "mansoury": _mansoury_probabilities(mansoury_alpha),
    }
    parent_fingerprint = _fingerprint(
        {
            "dataset": dataset.dataset_fingerprint,
            "slate_seed": slate_seed,
            "slate_size": SLATE_SIZE,
            "negative_policy": "exclude_all_observed_items",
            "generator": GENERATOR_VERSION,
        }
    )

    LOGGER.info(
        "creating uniform and Mansoury NCM slates for %d users",
        len(dataset.training_histories),
    )
    with ExitStack() as stack:
        example_files = _open_policy_files(stack, staging, "examples.jsonl")
        history_files = _open_policy_files(stack, staging, "histories.jsonl")
        example_count, user_count = _write_rows(
            dataset,
            example_files,
            history_files,
            probabilities,
            slate_seed,
        )

    for policy in POLICIES:
        _write_manifest(
            staging / policy,
            policy=policy,
            probabilities=probabilities[policy],
            mansoury_alpha=mansoury_alpha,
            dataset_fingerprint=dataset.dataset_fingerprint,
            parent_fingerprint=parent_fingerprint,
            slate_seed=slate_seed,
            example_count=example_count,
            user_count=user_count,
        )

    rename_directory_with_retry(staging, root, logger=LOGGER)
    LOGGER.info("saved NCM slate cache at %s", root)
    return root


def _write_rows(
    dataset: PreparedDatasetArtifact,
    example_files: dict[str, TextIO],
    history_files: dict[str, TextIO],
    probabilities: dict[str, tuple[float, ...]],
    slate_seed: int,
) -> tuple[int, int]:
    catalog = tuple(sorted(dataset.catalog))
    example_count = 0
    user_ids = tuple(sorted(dataset.training_histories))

    for user_number, user_id in enumerate(user_ids, start=1):
        training = tuple(dataset.training_histories[user_id])
        chronology = (
            *training,
            int(dataset.validation_targets[user_id]),
            int(dataset.test_targets[user_id]),
        )
        history_line = _json_line(
            {"user_id": user_id, "history_item_ids": chronology[:-1]}
        )
        for file in history_files.values():
            file.write(history_line)

        observed = set(chronology)
        eligible_items = np.asarray(
            [item_id for item_id in catalog if item_id not in observed],
            dtype=np.int64,
        )
        if len(eligible_items) < SLATE_SIZE - 1:
            raise ValueError(f"user {user_id} has too few eligible negatives")

        for event_index, target_item in enumerate(chronology[1:], start=1):
            negative_rng = _random_generator(
                slate_seed,
                "negatives",
                dataset.dataset_fingerprint,
                user_id,
                event_index,
            )
            negatives = negative_rng.choice(
                eligible_items,
                size=SLATE_SIZE - 1,
                replace=False,
            )
            for policy in POLICIES:
                generator = _random_generator(
                    slate_seed,
                    "placement",
                    policy,
                    user_id,
                    event_index,
                )
                clicked_index = int(
                    generator.choice(SLATE_SIZE, p=probabilities[policy])
                )
                slate = [int(item_id) for item_id in generator.permutation(negatives)]
                slate.insert(clicked_index, int(target_item))
                example_files[policy].write(
                    _json_line(
                        {
                            "split": _split(event_index, len(chronology)),
                            "user_id": user_id,
                            "event_index": event_index,
                            "history_length": event_index,
                            "target_item_id": int(target_item),
                            "ordered_item_ids": slate,
                            "clicked_index": clicked_index,
                        }
                    )
                )
            example_count += 1

        if user_number % 500 == 0:
            LOGGER.info("created slates for %d/%d users", user_number, len(user_ids))

    return example_count, len(user_ids)


def _open_policy_files(
    stack: ExitStack,
    root: Path,
    filename: str,
) -> dict[str, TextIO]:
    files = {}
    for policy in POLICIES:
        policy_root = root / policy
        policy_root.mkdir(exist_ok=True)
        files[policy] = stack.enter_context(
            (policy_root / filename).open("x", encoding="utf-8", newline="\n")
        )
    return files


def _write_manifest(
    directory: Path,
    *,
    policy: str,
    probabilities: tuple[float, ...],
    mansoury_alpha: float,
    dataset_fingerprint: str,
    parent_fingerprint: str,
    slate_seed: int,
    example_count: int,
    user_count: int,
) -> None:
    examples = directory / "examples.jsonl"
    histories = directory / "histories.jsonl"
    metadata = {
        "schema_version": PLACEMENT_SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "slate_size": SLATE_SIZE,
        "prepared_fingerprint": dataset_fingerprint,
        "parent_fingerprint": parent_fingerprint,
        "seed": slate_seed,
        "policy": policy,
        "rank_probabilities": probabilities,
        "mansoury_alpha": mansoury_alpha if policy == "mansoury" else None,
        "example_count": example_count,
        "history_sequence_count": user_count,
        "examples_file": examples.name,
        "examples_file_sha256": _file_sha256(examples),
        "histories_file": histories.name,
        "histories_file_sha256": _file_sha256(histories),
    }
    manifest = {
        "artifact_type": "first_click_placement_slates",
        **metadata,
        "content_fingerprint": _fingerprint(metadata),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_cache(
    root: Path,
    dataset_fingerprint: str,
    slate_seed: int,
    mansoury_alpha: float,
) -> None:
    for policy in POLICIES:
        manifest_path = root / policy / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing NCM slate manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_alpha = mansoury_alpha if policy == "mansoury" else None
        if (
            manifest.get("schema_version") != PLACEMENT_SCHEMA
            or manifest.get("generator_version") != GENERATOR_VERSION
            or manifest.get("prepared_fingerprint") != dataset_fingerprint
            or manifest.get("seed") != slate_seed
            or manifest.get("policy") != policy
            or manifest.get("mansoury_alpha") != expected_alpha
        ):
            raise ValueError(f"cached NCM slate configuration differs: {policy}")
        for filename, expected_hash in (
            (manifest["examples_file"], manifest["examples_file_sha256"]),
            (manifest["histories_file"], manifest["histories_file_sha256"]),
        ):
            path = root / policy / filename
            if _file_sha256(path) != expected_hash:
                raise ValueError(f"cached NCM slate hash differs: {path}")


def _mansoury_probabilities(alpha: float) -> tuple[float, ...]:
    ranks = np.arange(1, SLATE_SIZE + 1, dtype=np.float64)
    weights = np.exp(alpha * ranks)
    return tuple(float(value) for value in weights / weights.sum())


def _split(event_index: int, number_of_events: int) -> str:
    if event_index == number_of_events - 2:
        return "validation"
    if event_index == number_of_events - 1:
        return "test"
    return "train"


def _random_generator(seed: int, *parts: Any) -> np.random.Generator:
    derived_seed = int.from_bytes(
        hashlib.sha256(_json_bytes((seed, *parts))).digest()[:8],
        "big",
    )
    return np.random.default_rng(derived_seed)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _json_line(value: Any) -> str:
    return _json_bytes(value).decode("utf-8") + "\n"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["PLACEMENT_SCHEMA", "ensure_ncm_training_slates"]
