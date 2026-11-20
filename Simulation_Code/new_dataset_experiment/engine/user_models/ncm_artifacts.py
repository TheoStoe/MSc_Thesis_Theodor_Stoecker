"""Validated loaders for immutable NCM embedding and checkpoint artifacts.

The schema literals retain their historical first-click names because the
completed pilot artifacts already persist them. New Python APIs use NCM naming.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


NCM_CHECKPOINT_SCHEMA_VERSION = "goodreads-ncm-checkpoint-v1"
NCM_EMBEDDING_SCHEMA_VERSION = "goodreads-ncm-item-embeddings-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_fingerprint(value: Any) -> str:
    """Reproduce the canonical hash used by the existing artifacts."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_fingerprint(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": canonical.shape},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def load_ncm_item_embeddings(
    directory: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> tuple[dict[int, np.ndarray], str]:
    """Load and validate the frozen full-catalog NCM embedding table."""

    root = Path(directory)
    manifest_path = root / "manifest.json"
    arrays_path = root / "embeddings.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(
            "NCM embedding artifact needs manifest.json and embeddings.npz"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != NCM_EMBEDDING_SCHEMA_VERSION:
        raise ValueError("unsupported NCM embedding artifact schema")
    if manifest.get("arrays_file_sha256") != _file_sha256(arrays_path):
        raise ValueError("NCM embedding array hash mismatch")
    fingerprint = manifest.get("content_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("NCM embedding manifest has no content fingerprint")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError("NCM embedding artifact fingerprint mismatch")

    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"item_ids", "vectors"}:
            raise ValueError("NCM embeddings must contain item_ids and vectors")
        item_ids = np.asarray(arrays["item_ids"])
        vectors = np.array(arrays["vectors"], copy=True)
    if item_ids.ndim != 1 or vectors.ndim != 2:
        raise ValueError("NCM embedding arrays have invalid dimensions")
    if vectors.shape[0] != item_ids.shape[0]:
        raise ValueError("NCM embedding item IDs and vectors do not align")
    if vectors.shape[1] != manifest.get("dimension"):
        raise ValueError("NCM embedding dimension disagrees with manifest")
    if not np.issubdtype(item_ids.dtype, np.integer):
        raise TypeError("NCM embedding item IDs must use an integer dtype")
    if not np.issubdtype(vectors.dtype, np.floating):
        raise TypeError("NCM embedding vectors must use a floating dtype")
    if vectors.dtype.name != manifest.get("dtype"):
        raise ValueError("NCM embedding vector dtype disagrees with manifest")
    if not bool(np.isfinite(vectors).all()):
        raise ValueError("NCM embedding vectors must be finite")

    normalized_item_ids = tuple(int(value) for value in item_ids.tolist())
    if tuple(sorted(normalized_item_ids)) != normalized_item_ids:
        raise ValueError("NCM embedding item IDs must be in ascending row order")
    catalog_fingerprint = _artifact_fingerprint(
        {"item_ids": normalized_item_ids}
    )
    vector_fingerprint = _array_fingerprint(vectors)
    if catalog_fingerprint != manifest.get("catalog_fingerprint"):
        raise ValueError("NCM embedding catalog fingerprint mismatch")
    if vector_fingerprint != manifest.get("vector_content_fingerprint"):
        raise ValueError("NCM embedding vector fingerprint mismatch")
    semantic_payload = {
        "schema_version": manifest["schema_version"],
        "provider": manifest["provider"],
        "model": manifest["model"],
        "request_parameters": manifest["request_parameters"],
        "text_builder_version": manifest["text_builder_version"],
        "included_metadata_fields": manifest["included_metadata_fields"],
        "preprocessing_fingerprint": manifest["preprocessing_fingerprint"],
        "catalog_fingerprint": manifest["catalog_fingerprint"],
        "item_texts_fingerprint": manifest["item_texts_fingerprint"],
        "item_ids": normalized_item_ids,
        "dimension": int(vectors.shape[1]),
        "dtype": vectors.dtype.name,
        "normalization": manifest["normalization"],
        "vector_content_fingerprint": vector_fingerprint,
    }
    if _artifact_fingerprint(semantic_payload) != fingerprint:
        raise ValueError("NCM embedding content fingerprint mismatch")

    table: dict[int, np.ndarray] = {}
    for row, raw_item_id in enumerate(item_ids.tolist()):
        if isinstance(raw_item_id, bool) or not isinstance(raw_item_id, int):
            raise TypeError("NCM embedding item IDs must be integers")
        if raw_item_id <= 0 or raw_item_id in table:
            raise ValueError("NCM embedding item IDs must be unique and positive")
        value = vectors[row].copy()
        value.setflags(write=False)
        table[raw_item_id] = value
    return table, fingerprint


def load_ncm_checkpoint_artifact(
    directory: str | Path,
    *,
    expected_embedding_fingerprint: str,
    expected_prepared_fingerprint: str | None = None,
    expected_parent_fingerprint: str | None = None,
    expected_placement_fingerprint: str | None = None,
    device: str = "cpu",
) -> tuple[Mapping[str, Any], Mapping[str, torch.Tensor], str]:
    """Validate an NCM checkpoint and return its manifest and raw state dict."""

    root = Path(directory)
    manifest_path = root / "manifest.json"
    weights_path = root / "model.pt"
    if not manifest_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("NCM checkpoint needs manifest.json and model.pt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != NCM_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported NCM checkpoint schema")
    if manifest.get("weights_sha256") != _file_sha256(weights_path):
        raise ValueError("NCM checkpoint weight hash mismatch")
    expectations = {
        "embedding_fingerprint": expected_embedding_fingerprint,
        "prepared_fingerprint": expected_prepared_fingerprint,
        "parent_fingerprint": expected_parent_fingerprint,
        "placement_fingerprint": expected_placement_fingerprint,
    }
    for field, expected in expectations.items():
        if expected is not None and manifest.get(field) != expected:
            raise ValueError(f"NCM checkpoint {field} mismatch")

    checkpoint_fingerprint = manifest.get("checkpoint_fingerprint")
    if not isinstance(checkpoint_fingerprint, str) or not checkpoint_fingerprint:
        raise ValueError("NCM checkpoint manifest has no checkpoint fingerprint")
    metadata_payload = {
        field: manifest[field]
        for field in (
            "condition",
            "model_config",
            "training_config",
            "prepared_fingerprint",
            "parent_fingerprint",
            "placement_fingerprint",
            "embedding_fingerprint",
            "best_epoch",
            "schema_version",
        )
    }
    if _artifact_fingerprint(metadata_payload) != manifest.get(
        "metadata_fingerprint"
    ):
        raise ValueError("NCM checkpoint metadata fingerprint mismatch")
    fingerprint_payload = {
        key: value
        for key, value in manifest.items()
        if key != "checkpoint_fingerprint"
    }
    if _artifact_fingerprint(fingerprint_payload) != checkpoint_fingerprint:
        raise ValueError("NCM checkpoint fingerprint mismatch")

    state = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("NCM checkpoint model.pt must contain a state mapping")
    return manifest, state, checkpoint_fingerprint


def save_ncm_checkpoint_artifact(
    directory: str | Path,
    *,
    model: torch.nn.Module,
    condition: str,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    prepared_fingerprint: str,
    parent_fingerprint: str,
    placement_fingerprint: str,
    embedding_fingerprint: str,
    best_epoch: int,
) -> str:
    """Save one non-overwriting checkpoint compatible with the active loader."""

    for value, name in (
        (condition, "condition"),
        (prepared_fingerprint, "prepared_fingerprint"),
        (parent_fingerprint, "parent_fingerprint"),
        (placement_fingerprint, "placement_fingerprint"),
        (embedding_fingerprint, "embedding_fingerprint"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-blank string")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
        raise TypeError("best_epoch must be an integer")
    if best_epoch <= 0:
        raise ValueError("best_epoch must be positive")

    output = Path(directory)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"checkpoint destination already exists: {output}"
        ) from exc
    weights_path = output / "model.pt"
    torch.save(model.state_dict(), weights_path)
    metadata = {
        "condition": condition,
        "model_config": dict(model_config),
        "training_config": dict(training_config),
        "prepared_fingerprint": prepared_fingerprint,
        "parent_fingerprint": parent_fingerprint,
        "placement_fingerprint": placement_fingerprint,
        "embedding_fingerprint": embedding_fingerprint,
        "best_epoch": best_epoch,
        "schema_version": NCM_CHECKPOINT_SCHEMA_VERSION,
    }
    manifest = {
        **metadata,
        "metadata_fingerprint": _artifact_fingerprint(metadata),
        "weights_sha256": _file_sha256(weights_path),
    }
    checkpoint_fingerprint = _artifact_fingerprint(manifest)
    manifest["checkpoint_fingerprint"] = checkpoint_fingerprint
    (output / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint_fingerprint


__all__ = [
    "NCM_CHECKPOINT_SCHEMA_VERSION",
    "NCM_EMBEDDING_SCHEMA_VERSION",
    "load_ncm_checkpoint_artifact",
    "load_ncm_item_embeddings",
    "save_ncm_checkpoint_artifact",
]
