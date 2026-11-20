"""Portable, non-overwriting storage for prepared MovieLens inputs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from ..utility.fingerprints import fingerprint_semantic_state
from ..utility.filesystem import rename_directory_with_retry
from ..utility.user_handling import (
    PreparedDatasetArtifact,
    PreparedTrainingInteraction,
)
from .movielens import DATASET_SCHEMA, dataset_semantic_payload


PREPARED_FILE_SCHEMA = "feedback-loop-prepared-files-v1"
logger = logging.getLogger(__name__)


def save_prepared_dataset(
    directory: str | Path,
    artifact: PreparedDatasetArtifact,
) -> Path:
    """Write one immutable prepared-data directory through a staging sibling."""

    if not isinstance(artifact, PreparedDatasetArtifact):
        raise TypeError("artifact must be PreparedDatasetArtifact")
    _validate_artifact(artifact)
    root = Path(directory)
    staging = root.with_name(root.name + ".incomplete")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise FileExistsError(f"prepared-data artifact already exists: {root}")
    if staging.exists():
        raise FileExistsError(f"incomplete prepared-data artifact exists: {staging}")
    staging.mkdir()

    catalog_path = staging / "catalog.json"
    mappings_path = staging / "mappings.json"
    sequences_path = staging / "sequences.npz"
    manifest_path = staging / "manifest.json"

    _write_json(
        catalog_path,
        {str(item_id): artifact.catalog[item_id] for item_id in sorted(artifact.catalog)},
    )
    _write_json(
        mappings_path,
        {
            "runtime_to_source_user_id": artifact.runtime_to_source_user_id,
            "runtime_to_source_item_id": artifact.runtime_to_source_item_id,
        },
    )
    _write_sequences(sequences_path, artifact)
    _write_json(
        manifest_path,
        {
            "file_schema": PREPARED_FILE_SCHEMA,
            "schema_version": artifact.schema_version,
            "preparation_protocol": artifact.preparation_protocol,
            "preparation_settings": artifact.preparation_settings,
            "preparation_statistics": artifact.preparation_statistics,
            "dataset_fingerprint": artifact.dataset_fingerprint,
            "id_mapping_fingerprint": artifact.id_mapping_fingerprint,
            "catalog_fingerprint": artifact.catalog_fingerprint,
            "item_text_fingerprint": artifact.item_text_fingerprint,
            "source_file_fingerprints": artifact.source_file_fingerprints,
            "statistics": {
                "users": len(artifact.training_histories),
                "items": len(artifact.catalog),
                "training_interactions": sum(
                    len(history) for history in artifact.training_histories.values()
                ),
            },
        },
    )
    _write_json(
        staging / "_COMPLETE.json",
        {
            "file_schema": PREPARED_FILE_SCHEMA,
            "status": "complete",
            "dataset_fingerprint": artifact.dataset_fingerprint,
            "file_sha256": {
                path.name: _file_sha256(path)
                for path in (catalog_path, mappings_path, sequences_path, manifest_path)
            },
        },
    )
    _rename_prepared_directory(staging, root)
    return root


def _rename_prepared_directory(source: Path, destination: Path) -> None:
    """Commit prepared data while tolerating short Windows/OneDrive locks."""

    rename_directory_with_retry(
        source,
        destination,
        rename=os.rename,
        logger=logger,
        sleep=time.sleep,
    )


def load_prepared_dataset(directory: str | Path) -> PreparedDatasetArtifact:
    root = Path(directory)
    required = (
        "catalog.json",
        "mappings.json",
        "sequences.npz",
        "manifest.json",
        "_COMPLETE.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"prepared-data artifact is missing files: {missing}")
    manifest = _read_json(root / "manifest.json")
    if manifest.get("file_schema") != PREPARED_FILE_SCHEMA:
        raise ValueError("unsupported prepared-data file schema")
    complete = _read_json(root / "_COMPLETE.json")
    if complete.get("status") != "complete":
        raise ValueError("prepared-data artifact is not marked complete")
    for filename, expected in complete.get("file_sha256", {}).items():
        if not _file_hash_matches(root / filename, expected):
            raise ValueError(f"prepared-data file hash mismatch: {filename}")

    raw_catalog = _read_json(root / "catalog.json")
    catalog = {
        int(item_id): {
            **values,
            "item_id": int(values["item_id"]),
            "source_item_id": str(values["source_item_id"]),
            "genres": tuple(values["genres"]),
        }
        for item_id, values in raw_catalog.items()
    }
    mappings = _read_json(root / "mappings.json")
    runtime_to_source_user_id = {
        int(user_id): str(source_id)
        for user_id, source_id in mappings["runtime_to_source_user_id"].items()
    }
    runtime_to_source_item_id = {
        int(item_id): str(source_id)
        for item_id, source_id in mappings["runtime_to_source_item_id"].items()
    }
    (
        histories,
        training_interactions,
        validation,
        test,
        popularity,
    ) = _read_sequences(
        root / "sequences.npz",
        runtime_to_source_item_id=runtime_to_source_item_id,
    )
    preparation_statistics = dict(manifest["preparation_statistics"])
    preparation_statistics["filter_rounds"] = tuple(
        dict(round_values)
        for round_values in preparation_statistics.get("filter_rounds", ())
    )
    artifact = PreparedDatasetArtifact(
        schema_version=str(manifest["schema_version"]),
        preparation_protocol=str(manifest["preparation_protocol"]),
        preparation_settings=dict(manifest["preparation_settings"]),
        preparation_statistics=preparation_statistics,
        dataset_fingerprint=str(manifest["dataset_fingerprint"]),
        id_mapping_fingerprint=str(manifest["id_mapping_fingerprint"]),
        catalog_fingerprint=str(manifest["catalog_fingerprint"]),
        item_text_fingerprint=str(manifest["item_text_fingerprint"]),
        source_file_fingerprints={
            str(name): str(value)
            for name, value in manifest["source_file_fingerprints"].items()
        },
        catalog=catalog,
        training_histories=histories,
        training_interactions=training_interactions,
        training_history_fingerprints={
            user_id: fingerprint_semantic_state(
                {
                    "user_id": user_id,
                    "source_user_id": runtime_to_source_user_id[user_id],
                    "interactions": training_interactions[user_id],
                },
                "ml1m-user-training-history-v1",
            )
            for user_id in sorted(training_interactions)
        },
        validation_targets=validation,
        test_targets=test,
        runtime_to_source_user_id=runtime_to_source_user_id,
        runtime_to_source_item_id=runtime_to_source_item_id,
        initial_item_popularity=popularity,
    )
    _validate_artifact(artifact)
    if complete.get("dataset_fingerprint") != artifact.dataset_fingerprint:
        raise ValueError("completion and manifest dataset fingerprints differ")
    return artifact


def _validate_artifact(artifact: PreparedDatasetArtifact) -> None:
    if artifact.schema_version != DATASET_SCHEMA:
        raise ValueError(f"unsupported dataset schema: {artifact.schema_version}")
    users = set(artifact.training_histories)
    if not users or users != set(artifact.validation_targets) or users != set(
        artifact.test_targets
    ):
        raise ValueError("train, validation, and test user sets must match")
    if users != set(artifact.runtime_to_source_user_id):
        raise ValueError("runtime user mapping does not cover every user")
    if sorted(users) != list(range(1, len(users) + 1)):
        raise ValueError("runtime user IDs must be contiguous and one-based")
    if len(set(artifact.runtime_to_source_user_id.values())) != len(users):
        raise ValueError("source user IDs must be unique")
    if users != set(artifact.training_interactions):
        raise ValueError("training interactions do not cover every user")
    if users != set(artifact.training_history_fingerprints):
        raise ValueError("training-history fingerprints do not cover every user")
    items = set(artifact.catalog)
    if not items or items != set(artifact.runtime_to_source_item_id):
        raise ValueError("runtime item mapping does not cover the catalog")
    if sorted(items) != list(range(1, len(items) + 1)):
        raise ValueError("runtime item IDs must be contiguous and one-based")
    if len(set(artifact.runtime_to_source_item_id.values())) != len(items):
        raise ValueError("source item IDs must be unique")
    if items != set(artifact.initial_item_popularity):
        raise ValueError("item popularity does not cover the catalog")
    for item_id, metadata in artifact.catalog.items():
        if metadata.get("item_id") != item_id:
            raise ValueError(f"catalog runtime item mismatch for {item_id}")
        if metadata.get("source_item_id") != artifact.runtime_to_source_item_id[
            item_id
        ]:
            raise ValueError(f"catalog source item mismatch for {item_id}")
    observed = {
        item
        for history in artifact.training_histories.values()
        for item in history
    } | set(artifact.validation_targets.values()) | set(artifact.test_targets.values())
    if not observed.issubset(items):
        raise ValueError("prepared sequences contain unknown items")
    expected_statistics = {
        "retained_users": len(users),
        "retained_items": len(items),
        "training_events": sum(
            len(history) for history in artifact.training_histories.values()
        ),
    }
    expected_statistics["retained_events"] = (
        expected_statistics["training_events"] + 2 * len(users)
    )
    for name, expected in expected_statistics.items():
        if artifact.preparation_statistics.get(name) != expected:
            raise ValueError(f"preparation statistic mismatch: {name}")
    source_events = artifact.preparation_statistics.get("source_events")
    if (
        isinstance(source_events, bool)
        or not isinstance(source_events, int)
        or source_events < expected_statistics["retained_events"]
    ):
        raise ValueError("preparation source-event statistic is invalid")
    filter_rounds = artifact.preparation_statistics.get("filter_rounds")
    if not isinstance(filter_rounds, (tuple, list)) or not filter_rounds:
        raise ValueError("preparation filter rounds cannot be empty")
    if filter_rounds[-1].get("output_events") != expected_statistics[
        "retained_events"
    ]:
        raise ValueError("final filter-round count does not match retained events")
    for user_id in users:
        interactions = artifact.training_interactions[user_id]
        if tuple(value.item_id for value in interactions) != artifact.training_histories[
            user_id
        ]:
            raise ValueError(f"training interaction/history mismatch for {user_id}")
        for interaction in interactions:
            if (
                artifact.runtime_to_source_item_id[interaction.item_id]
                != interaction.source_item_id
            ):
                raise ValueError(
                    f"training interaction source item mismatch for {user_id}"
                )
        expected_history_fingerprint = fingerprint_semantic_state(
            {
                "user_id": user_id,
                "source_user_id": artifact.runtime_to_source_user_id[user_id],
                "interactions": interactions,
            },
            "ml1m-user-training-history-v1",
        )
        if (
            artifact.training_history_fingerprints[user_id]
            != expected_history_fingerprint
        ):
            raise ValueError(f"training-history fingerprint mismatch for {user_id}")

    id_mapping_payload = {
        "runtime_to_source_user_id": artifact.runtime_to_source_user_id,
        "runtime_to_source_item_id": artifact.runtime_to_source_item_id,
    }
    if artifact.id_mapping_fingerprint != fingerprint_semantic_state(
        id_mapping_payload, "ml1m-id-map-v1"
    ):
        raise ValueError("prepared-data ID mapping fingerprint mismatch")
    if artifact.catalog_fingerprint != fingerprint_semantic_state(
        artifact.catalog, "ml1m-catalog-v1"
    ):
        raise ValueError("prepared-data catalog fingerprint mismatch")
    expected_item_text = fingerprint_semantic_state(
        {
            "builder_version": "ml1m-title-genres-v1",
            "item_text_by_id": {
                item_id: artifact.catalog[item_id]["item_text"]
                for item_id in sorted(artifact.catalog)
            },
        },
        "ml1m-title-genres-v1",
    )
    if artifact.item_text_fingerprint != expected_item_text:
        raise ValueError("prepared-data item-text fingerprint mismatch")
    if artifact.dataset_fingerprint != fingerprint_semantic_state(
        dataset_semantic_payload(artifact), artifact.schema_version
    ):
        raise ValueError("prepared-data dataset fingerprint mismatch")


def _write_sequences(path: Path, artifact: PreparedDatasetArtifact) -> None:
    user_ids = np.asarray(sorted(artifact.training_histories), dtype=np.int64)
    items: list[int] = []
    ratings: list[int] = []
    timestamps: list[int] = []
    source_rows: list[int] = []
    offsets = [0]
    for user_id in user_ids.tolist():
        items.extend(artifact.training_histories[user_id])
        interactions = artifact.training_interactions[user_id]
        ratings.extend(value.rating for value in interactions)
        timestamps.extend(value.timestamp for value in interactions)
        source_rows.extend(value.source_row for value in interactions)
        offsets.append(len(items))
    np.savez_compressed(
        path,
        user_ids=user_ids,
        training_items=np.asarray(items, dtype=np.int64),
        training_offsets=np.asarray(offsets, dtype=np.int64),
        training_ratings=np.asarray(ratings, dtype=np.int8),
        training_timestamps=np.asarray(timestamps, dtype=np.int64),
        training_source_rows=np.asarray(source_rows, dtype=np.int64),
        validation_items=np.asarray(
            [artifact.validation_targets[user_id] for user_id in user_ids],
            dtype=np.int64,
        ),
        test_items=np.asarray(
            [artifact.test_targets[user_id] for user_id in user_ids], dtype=np.int64
        ),
        popularity_item_ids=np.asarray(
            sorted(artifact.initial_item_popularity), dtype=np.int64
        ),
        popularity_counts=np.asarray(
            [
                artifact.initial_item_popularity[item_id]
                for item_id in sorted(artifact.initial_item_popularity)
            ],
            dtype=np.int64,
        ),
    )


def _read_sequences(
    path: Path,
    *,
    runtime_to_source_item_id: Mapping[int, str],
) -> tuple[
    dict[int, tuple[int, ...]],
    dict[int, tuple[PreparedTrainingInteraction, ...]],
    dict[int, int],
    dict[int, int],
    dict[int, int],
]:
    with np.load(path, allow_pickle=False) as arrays:
        user_ids = arrays["user_ids"].astype(np.int64, copy=False)
        items = arrays["training_items"].astype(np.int64, copy=False)
        offsets = arrays["training_offsets"].astype(np.int64, copy=False)
        ratings = arrays["training_ratings"].astype(np.int64, copy=False)
        timestamps = arrays["training_timestamps"].astype(np.int64, copy=False)
        source_rows = arrays["training_source_rows"].astype(np.int64, copy=False)
        validation_items = arrays["validation_items"].astype(np.int64, copy=False)
        test_items = arrays["test_items"].astype(np.int64, copy=False)
        popularity_items = arrays["popularity_item_ids"].astype(np.int64, copy=False)
        popularity_counts = arrays["popularity_counts"].astype(np.int64, copy=False)
    if len(offsets) != len(user_ids) + 1 or offsets[0] != 0 or offsets[-1] != len(
        items
    ):
        raise ValueError("invalid training sequence offsets")
    if len(validation_items) != len(user_ids) or len(test_items) != len(user_ids):
        raise ValueError("held-out target arrays do not match user count")
    if not (len(items) == len(ratings) == len(timestamps) == len(source_rows)):
        raise ValueError("training event arrays have inconsistent lengths")
    histories = {
        int(user_id): tuple(
            int(item_id) for item_id in items[offsets[index] : offsets[index + 1]]
        )
        for index, user_id in enumerate(user_ids)
    }
    training_interactions = {
        int(user_id): tuple(
            PreparedTrainingInteraction(
                item_id=int(items[position]),
                source_item_id=runtime_to_source_item_id[int(items[position])],
                rating=int(ratings[position]),
                timestamp=int(timestamps[position]),
                source_row=int(source_rows[position]),
            )
            for position in range(int(offsets[index]), int(offsets[index + 1]))
        )
        for index, user_id in enumerate(user_ids)
    }
    validation = {
        int(user_id): int(validation_items[index])
        for index, user_id in enumerate(user_ids)
    }
    test = {
        int(user_id): int(test_items[index])
        for index, user_id in enumerate(user_ids)
    }
    popularity = {
        int(item_id): int(popularity_counts[index])
        for index, item_id in enumerate(popularity_items)
    }
    return histories, training_interactions, validation, test, popularity


def _write_json(path: Path, value: Any) -> None:
    # Keep artifact bytes stable across Windows, macOS, and Linux.  Older
    # artifacts were written with the platform newline and are handled by the
    # compatibility check in ``_file_hash_matches``.
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hash_matches(path: Path, expected: Any) -> bool:
    if not isinstance(expected, str):
        return False
    data = path.read_bytes()
    variants = (data,)
    if path.suffix.lower() == ".json":
        normalized = data.replace(b"\r\n", b"\n")
        variants = (data, normalized, normalized.replace(b"\n", b"\r\n"))
    return any(hashlib.sha256(value).hexdigest() == expected for value in variants)


__all__ = [
    "PREPARED_FILE_SCHEMA",
    "load_prepared_dataset",
    "save_prepared_dataset",
]
