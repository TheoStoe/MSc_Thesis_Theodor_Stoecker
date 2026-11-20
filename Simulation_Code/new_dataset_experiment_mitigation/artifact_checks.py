"""Read-only checks that gender-mitigation jobs share the earlier Goodreads start.

Only JSON, NPZ arrays, and file digests are inspected. Model checkpoint files
are hashed as opaque bytes; this module never deserialises model tensors.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .gender_mitigation import load_gender_labels
from .runtime_adapter import file_sha256, load_book_inputs


NCM_MODELS = {
    "ncm_inspired_gru_uniform": ("P3", "T2_slatewise_gru_uniform"),
    "ncm_inspired_gru_mansoury": ("P4", "T3_slatewise_gru_mansoury"),
}


def _read(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return result


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps({"dtype": contiguous.dtype.str, "shape": contiguous.shape},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    digest = hashlib.sha256(header + b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _pin(path: Path, pins: dict[str, str], expected: str | None = None) -> str:
    resolved = path.resolve(strict=True)
    actual = file_sha256(resolved)
    if expected is not None and actual != expected:
        raise ValueError(f"artifact file SHA256 mismatch: {resolved}")
    pins[str(resolved)] = actual
    return actual


def _declared_hash(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("artifact requires a complete lowercase SHA256 digest")
    return value


def _check_complete(root: Path, key: str, required: set[str], pins: dict[str, str]) -> dict[str, Any]:
    complete = _read(root / "_COMPLETE.json")
    _pin(root / "_COMPLETE.json", pins)
    if complete.get("status") != "complete":
        raise ValueError(f"artifact bundle is incomplete: {root}")
    hashes = complete.get(key, {})
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise ValueError(f"artifact completion record lacks required hashes: {root}")
    for filename, expected in hashes.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("completion record must contain plain filenames")
        _pin(root / filename, pins, _declared_hash(expected))
    return complete


def _check_embeddings(root: Path, dataset: Any, pins: dict[str, str]) -> tuple[str, int]:
    directory = root / "embeddings"
    manifest = _read(directory / "manifest.json")
    _pin(directory / "manifest.json", pins)
    _pin(directory / "embeddings.npz", pins, _declared_hash(manifest.get("arrays_file_sha256")))
    if manifest.get("schema_version") != "goodreads-ncm-item-embeddings-v1":
        raise ValueError("unsupported Goodreads embedding schema")
    if manifest.get("preprocessing_fingerprint") != dataset.dataset_fingerprint:
        raise ValueError("embedding prepared-dataset fingerprint mismatch")
    if manifest.get("item_texts_fingerprint") != dataset.item_text_fingerprint:
        raise ValueError("embedding item-text fingerprint mismatch")
    with np.load(directory / "embeddings.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"item_ids", "vectors"}:
            raise ValueError("embedding NPZ must contain item_ids and vectors")
        ids, vectors = arrays["item_ids"], arrays["vectors"]
    if ids.ndim != 1 or vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ValueError("embedding arrays have invalid or inconsistent dimensions")
    if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(vectors.dtype, np.floating):
        raise ValueError("embedding arrays require integer IDs and floating vectors")
    actual_ids = tuple(int(item) for item in ids)
    if actual_ids != tuple(sorted(dataset.catalog)) or manifest.get("item_ids") != list(actual_ids):
        raise ValueError("embedding IDs do not match the complete frozen catalogue")
    if vectors.shape[1] != manifest.get("dimension") or vectors.dtype.name != manifest.get("dtype"):
        raise ValueError("embedding vector shape or dtype differs from its manifest")
    if not all(bool(np.isfinite(vectors[start:start + 8192]).all()) for start in range(0, len(vectors), 8192)):
        raise ValueError("embedding vectors must be finite")
    catalog_hash = _canonical_hash({"item_ids": actual_ids})
    if manifest.get("catalog_fingerprint") != catalog_hash or manifest.get("vector_content_fingerprint") != _array_hash(vectors):
        raise ValueError("embedding catalogue or vector content digest mismatch")
    semantic_fields = ("schema_version", "provider", "model", "request_parameters", "text_builder_version",
        "included_metadata_fields", "preprocessing_fingerprint", "catalog_fingerprint", "item_texts_fingerprint",
        "item_ids", "dimension", "dtype", "normalization", "vector_content_fingerprint")
    if _canonical_hash({key: manifest[key] for key in semantic_fields}) != manifest.get("content_fingerprint"):
        raise ValueError("embedding semantic content fingerprint mismatch")
    identity = _read(directory / "request_identity.json")
    _pin(directory / "request_identity.json", pins)
    for key in ("preprocessing_fingerprint", "item_texts_fingerprint", "provider", "model", "request_parameters", "text_builder_version"):
        if identity.get(key) != manifest.get(key):
            raise ValueError(f"embedding request identity mismatch: {key}")
    return manifest["content_fingerprint"], int(vectors.shape[1])


def _check_ncm(root: Path, model: str, dataset: Any, seed: int,
               embedding_fingerprint: str, dimension: int, pins: dict[str, str]) -> dict[str, str]:
    folder, condition = NCM_MODELS[model]
    checkpoint = root / "ncm" / f"seed_{seed}" / folder / "checkpoint"
    manifest = _read(checkpoint / "manifest.json")
    _pin(checkpoint / "manifest.json", pins)
    _pin(checkpoint / "model.pt", pins, _declared_hash(manifest.get("weights_sha256")))
    if manifest.get("schema_version") != "goodreads-ncm-checkpoint-v1" or manifest.get("condition") != condition:
        raise ValueError("NCM checkpoint schema or placement condition mismatch")
    if manifest.get("prepared_fingerprint") != dataset.dataset_fingerprint or manifest.get("embedding_fingerprint") != embedding_fingerprint:
        raise ValueError("NCM checkpoint prepared-input or embedding binding mismatch")
    if manifest.get("training_config", {}).get("seed") != seed:
        raise ValueError("NCM checkpoint training seed mismatch")
    config = manifest.get("model_config", {})
    if config.get("architecture") != "slatewise_gru" or config.get("slate_size") != 10 or config.get("embedding_dimension") != dimension:
        raise ValueError("NCM checkpoint architecture disagrees with selected computational model")
    metadata_fields = ("condition", "model_config", "training_config", "prepared_fingerprint", "parent_fingerprint",
        "placement_fingerprint", "embedding_fingerprint", "best_epoch", "schema_version")
    if manifest.get("metadata_fingerprint") != _canonical_hash({key: manifest[key] for key in metadata_fields}):
        raise ValueError("NCM checkpoint metadata digest mismatch")
    if manifest.get("checkpoint_fingerprint") != _canonical_hash({key: value for key, value in manifest.items() if key != "checkpoint_fingerprint"}):
        raise ValueError("NCM checkpoint fingerprint mismatch")
    return {"checkpoint_fingerprint": manifest["checkpoint_fingerprint"],
            "source_embedding_fingerprint": embedding_fingerprint}


def audit_binding(inputs_root: Path, labels_path: Path, population_seed: int, seed: int) -> dict[str, Any]:
    """Check a seed's frozen start and return compact, JSON-serialisable pins.

    This inspects CPU arrays and opaque checkpoint hashes only. Completion-file
    digests and every consumed model/embedding artifact are included so a plan
    can detect source changes before execution. No output files are written.
    """
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (population_seed, seed)):
        raise ValueError("population and simulation seeds must be non-negative integers")
    root, labels_path = Path(inputs_root).resolve(), Path(labels_path).resolve()
    prepared, initial = root / "prepared", root / "initial_sasrec" / f"seed_{seed}"
    inputs = load_book_inputs(prepared, root / "population.json", None, initial)
    dataset, population = inputs.dataset, inputs.population
    if len(population.members) != 300 or len(dataset.training_histories) != 300:
        raise ValueError("gender mitigation requires exactly 300 active readers")
    if any(getattr(member.role, "value", member.role) != "active" for member in population.members):
        raise ValueError("gender mitigation cannot contain reserve or background readers")
    if population.population_seed != population_seed or dataset.preparation_settings.get("sampling", {}).get("seed") != population_seed:
        raise ValueError("prepared sampling and population seeds must match the declared population")
    unseen = {user: len(set(dataset.catalog) - set(history)) for user, history in dataset.training_histories.items()}
    if min(unseen.values()) < 40:
        raise ValueError("each reader requires at least 40 initially unseen catalogue items")
    pins: dict[str, str] = {}
    prepared_complete = _check_complete(prepared, "file_sha256",
        {"catalog.json", "mappings.json", "sequences.npz", "manifest.json"}, pins)
    if prepared_complete.get("dataset_fingerprint") != dataset.dataset_fingerprint:
        raise ValueError("prepared completion record dataset mismatch")
    _pin(root / "population.json", pins)
    _pin(root / "source_ids.json", pins)
    source_ids = _read(root / "source_ids.json")
    if source_ids.get("dataset_fingerprint") != dataset.dataset_fingerprint:
        raise ValueError("source-ID mapping dataset mismatch")
    initial_complete = _check_complete(initial, "files_sha256",
        {"checkpoint.pt", "baseline_slates.json", "fit_diagnostics.json"}, pins)
    baseline, diagnostics = _read(initial / "baseline_slates.json"), _read(initial / "fit_diagnostics.json")
    if initial_complete.get("seed") != seed or baseline.get("simulation_seed") != seed:
        raise ValueError("shared initial checkpoint seed mismatch")
    if diagnostics.get("fit_iteration") != 0 or diagnostics.get("n_training_users") != 300 or diagnostics.get("n_catalog_items") != len(dataset.catalog):
        raise ValueError("initial fit diagnostics disagree with the frozen cohort")
    label_artifact = _read(labels_path)
    cohort_hash = dataset.source_file_fingerprints.get("cohort_json_sha256")
    labels = load_gender_labels(labels_path, catalogue_item_ids=dataset.catalog,
        cohort_fingerprint=cohort_hash, histories=dataset.training_histories)
    cohort_path = Path(label_artifact["audit_provenance"]["original_sources"]["cohort"]["path"])
    _pin(cohort_path, pins, cohort_hash)
    _pin(labels_path, pins, labels.labels_sha256)
    embedding_fp, dimension = _check_embeddings(root, dataset, pins)
    response_artifacts: dict[str, dict[str, str]] = {"position_biased": {}}
    for model in NCM_MODELS:
        response_artifacts[model] = _check_ncm(root, model, dataset, seed, embedding_fp, dimension, pins)
    return {
        "shared_start": {"dataset": dataset.dataset_fingerprint,
            "population": population.population_fingerprint,
            "initial_recommender": inputs.recommender_checkpoint_fingerprint},
        "cohort_sha256": cohort_hash, "labels_sha256": labels.labels_sha256,
        "label_coverage": dict(labels.coverage), "input_file_sha256": pins,
        "response_artifacts": response_artifacts,
        "population_seed": population_seed, "simulation_seed": seed,
        "active_user_count": 300, "minimum_initial_unseen_items": min(unseen.values()),
        "women_item_count": len(labels.women_item_ids),
        "initial_recommender_network_fingerprint": baseline.get("recommender_fingerprint"),
    }
