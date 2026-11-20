"""Small frozen sparse-history clustering pipeline for L4 population analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import warnings

import numpy as np

from ..contracts import UserId
from ..utility.fingerprints import fingerprint_semantic_state
from ..utility.filesystem import rename_directory_with_retry
from ..utility.user_handling import PreparedDatasetArtifact


CLUSTER_SCHEMA = "feedback-loop-user-clusters-v1"
CLUSTER_FILE_SCHEMA = "feedback-loop-user-cluster-files-v1"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterPipelineConfig:
    """All stochastic and dimensional choices covered by the artifact hash."""

    n_components: int = 32
    n_clusters: int = 5
    n_init: int = 20
    svd_seed: int = 42
    kmeans_seed: int = 42
    svd_n_iter: int = 5

    def __post_init__(self) -> None:
        for name in (
            "n_components",
            "n_clusters",
            "n_init",
            "svd_seed",
            "kmeans_seed",
            "svd_n_iter",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        for name in ("n_components", "n_clusters", "n_init", "svd_n_iter"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("svd_seed", "kmeans_seed"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    def to_dict(self) -> dict[str, int | str]:
        return {
            **asdict(self),
            "input_normalization": "l2_rows",
            "reduction": "sklearn_TruncatedSVD_randomized",
            "reduced_normalization": "l2_rows",
            "clustering": "sklearn_KMeans_lloyd",
            "cluster_label_rule": "lexicographic_centers_v1",
        }


@dataclass(frozen=True)
class FrozenClusterArtifact:
    schema_version: str
    dataset_fingerprint: str
    training_history_fingerprint: str
    config: ClusterPipelineConfig
    user_ids: tuple[UserId, ...]
    item_ids: tuple[int, ...]
    cluster_by_user: Mapping[UserId, int]
    svd_components: np.ndarray
    svd_explained_variance: np.ndarray
    svd_explained_variance_ratio: np.ndarray
    svd_singular_values: np.ndarray
    reduced_user_vectors: np.ndarray
    cluster_centers: np.ndarray
    artifact_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cluster_by_user",
            MappingProxyType(dict(self.cluster_by_user)),
        )
        for name in (
            "svd_components",
            "svd_explained_variance",
            "svd_explained_variance_ratio",
            "svd_singular_values",
            "reduced_user_vectors",
            "cluster_centers",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def fit_user_clusters(
    dataset: PreparedDatasetArtifact,
    *,
    config: ClusterPipelineConfig | None = None,
) -> FrozenClusterArtifact:
    """Fit binary training histories only; held-out targets are never inputs."""

    if not isinstance(dataset, PreparedDatasetArtifact):
        raise TypeError("dataset must be PreparedDatasetArtifact")
    effective = config or ClusterPipelineConfig()
    user_ids = tuple(sorted(dataset.training_histories))
    item_ids = tuple(sorted(dataset.catalog))
    if not user_ids or not item_ids:
        raise ValueError("dataset users and catalog cannot be empty")
    if effective.n_clusters > len(user_ids):
        raise ValueError("n_clusters cannot exceed the number of users")
    if effective.n_components >= min(len(user_ids), len(item_ids)):
        raise ValueError(
            "n_components must be smaller than both user and item counts"
        )

    # Imported only for clustering so the rest of data preparation remains
    # usable in lightweight environments that have not installed sklearn.
    try:
        from scipy.sparse import csr_matrix
        from sklearn.cluster import KMeans
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
    except ImportError as exc:  # pragma: no cover - dependency metadata protects runs
        raise RuntimeError(
            "user clustering requires the project scikit-learn dependency"
        ) from exc

    item_to_column = {item_id: index for index, item_id in enumerate(item_ids)}
    rows: list[int] = []
    columns: list[int] = []
    distinct_binary_histories: set[tuple[int, ...]] = set()
    for row, user_id in enumerate(user_ids):
        history = dataset.training_histories[user_id]
        if not history:
            raise ValueError(f"training history is empty for user {user_id}")
        unknown = sorted(set(history).difference(item_to_column))
        if unknown:
            raise ValueError(f"training history contains unknown items: {unknown}")
        # Binary, not count-valued: repeated history items still contribute 1.
        binary_history = tuple(sorted(set(history)))
        distinct_binary_histories.add(binary_history)
        for item_id in binary_history:
            rows.append(row)
            columns.append(item_to_column[item_id])
    if len(distinct_binary_histories) < effective.n_clusters:
        raise ValueError(
            "binary histories contain fewer distinct groups than requested clusters"
        )
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float64,
    )
    normalized_matrix = normalize(matrix, norm="l2", axis=1, copy=True)
    svd = TruncatedSVD(
        n_components=effective.n_components,
        algorithm="randomized",
        n_iter=effective.svd_n_iter,
        random_state=effective.svd_seed,
    )
    reduced = np.asarray(svd.fit_transform(normalized_matrix), dtype=np.float64)
    components = np.asarray(svd.components_, dtype=np.float64).copy()
    _canonicalize_svd_signs(components, reduced)
    normalized_reduced = np.asarray(
        normalize(reduced, norm="l2", axis=1, copy=True), dtype=np.float64
    )
    kmeans = KMeans(
        n_clusters=effective.n_clusters,
        n_init=effective.n_init,
        random_state=effective.kmeans_seed,
        algorithm="lloyd",
    )
    try:
        with warnings.catch_warnings():
            # Fewer distinct groups than requested makes the population strata
            # scientifically different from the frozen config, so preparation
            # fails rather than continuing after sklearn's warning.
            warnings.simplefilter("error", ConvergenceWarning)
            raw_labels = np.asarray(
                kmeans.fit_predict(normalized_reduced), dtype=np.int64
            )
    except ConvergenceWarning as exc:
        raise ValueError(
            "KMeans produced fewer distinct clusters than requested"
        ) from exc
    raw_centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    labels, centers = _canonicalize_cluster_labels(raw_labels, raw_centers)
    cluster_by_user = {
        user_id: int(labels[index]) for index, user_id in enumerate(user_ids)
    }
    training_history_fingerprint = fingerprint_semantic_state(
        {
            "catalog_item_ids": item_ids,
            "binary_training_items_by_user": {
                user_id: tuple(sorted(set(dataset.training_histories[user_id])))
                for user_id in user_ids
            },
        },
        "binary-training-user-item-v1",
    )
    payload = {
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "training_history_fingerprint": training_history_fingerprint,
        "config": effective.to_dict(),
        "user_ids": user_ids,
        "item_ids": item_ids,
        "cluster_by_user": cluster_by_user,
        "svd_components": components,
        "svd_explained_variance": np.asarray(
            svd.explained_variance_, dtype=np.float64
        ),
        "svd_explained_variance_ratio": np.asarray(
            svd.explained_variance_ratio_, dtype=np.float64
        ),
        "svd_singular_values": np.asarray(svd.singular_values_, dtype=np.float64),
        "reduced_user_vectors": normalized_reduced,
        "cluster_centers": centers,
    }
    artifact = FrozenClusterArtifact(
        schema_version=CLUSTER_SCHEMA,
        dataset_fingerprint=dataset.dataset_fingerprint,
        training_history_fingerprint=training_history_fingerprint,
        config=effective,
        user_ids=user_ids,
        item_ids=item_ids,
        cluster_by_user=cluster_by_user,
        svd_components=components,
        svd_explained_variance=np.asarray(
            svd.explained_variance_, dtype=np.float64
        ),
        svd_explained_variance_ratio=np.asarray(
            svd.explained_variance_ratio_, dtype=np.float64
        ),
        svd_singular_values=np.asarray(svd.singular_values_, dtype=np.float64),
        reduced_user_vectors=normalized_reduced,
        cluster_centers=centers,
        artifact_fingerprint=fingerprint_semantic_state(payload, CLUSTER_SCHEMA),
    )
    _validate_cluster_artifact(artifact)
    return artifact


def save_cluster_artifact(
    directory: str | Path,
    artifact: FrozenClusterArtifact,
) -> Path:
    if not isinstance(artifact, FrozenClusterArtifact):
        raise TypeError("artifact must be FrozenClusterArtifact")
    _validate_cluster_artifact(artifact)
    root = Path(directory)
    staging = root.with_name(root.name + ".incomplete")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise FileExistsError(f"cluster artifact already exists: {root}")
    if staging.exists():
        raise FileExistsError(f"incomplete cluster artifact exists: {staging}")
    staging.mkdir()
    assignments_path = staging / "assignments.json"
    arrays_path = staging / "model_arrays.npz"
    manifest_path = staging / "manifest.json"
    _write_json(
        assignments_path,
        {str(user): artifact.cluster_by_user[user] for user in artifact.user_ids},
    )
    np.savez_compressed(
        arrays_path,
        user_ids=np.asarray(artifact.user_ids, dtype=np.int64),
        item_ids=np.asarray(artifact.item_ids, dtype=np.int64),
        svd_components=artifact.svd_components,
        svd_explained_variance=artifact.svd_explained_variance,
        svd_explained_variance_ratio=artifact.svd_explained_variance_ratio,
        svd_singular_values=artifact.svd_singular_values,
        reduced_user_vectors=artifact.reduced_user_vectors,
        cluster_centers=artifact.cluster_centers,
    )
    _write_json(
        manifest_path,
        {
            "file_schema": CLUSTER_FILE_SCHEMA,
            "schema_version": artifact.schema_version,
            "dataset_fingerprint": artifact.dataset_fingerprint,
            "training_history_fingerprint": artifact.training_history_fingerprint,
            "config": artifact.config.to_dict(),
            "artifact_fingerprint": artifact.artifact_fingerprint,
        },
    )
    _write_json(
        staging / "_COMPLETE.json",
        {
            "file_schema": CLUSTER_FILE_SCHEMA,
            "status": "complete",
            "artifact_fingerprint": artifact.artifact_fingerprint,
            "file_sha256": {
                path.name: _file_sha256(path)
                for path in (assignments_path, arrays_path, manifest_path)
            },
        },
    )
    rename_directory_with_retry(staging, root, logger=logger)
    return root


def load_cluster_artifact(directory: str | Path) -> FrozenClusterArtifact:
    root = Path(directory)
    required = (
        "assignments.json",
        "model_arrays.npz",
        "manifest.json",
        "_COMPLETE.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cluster artifact is missing files: {missing}")
    manifest = _read_json(root / "manifest.json")
    complete = _read_json(root / "_COMPLETE.json")
    if manifest.get("file_schema") != CLUSTER_FILE_SCHEMA:
        raise ValueError("unsupported cluster file schema")
    if complete.get("status") != "complete":
        raise ValueError("cluster artifact is not marked complete")
    for filename, expected in complete.get("file_sha256", {}).items():
        if _file_sha256(root / filename) != expected:
            raise ValueError(f"cluster artifact file hash mismatch: {filename}")
    assignments = {
        int(user_id): int(cluster_id)
        for user_id, cluster_id in _read_json(root / "assignments.json").items()
    }
    with np.load(root / "model_arrays.npz", allow_pickle=False) as arrays:
        values = {name: arrays[name].copy() for name in arrays.files}
    raw_config = dict(manifest["config"])
    config = ClusterPipelineConfig(
        **{
            name: int(raw_config[name])
            for name in (
                "n_components",
                "n_clusters",
                "n_init",
                "svd_seed",
                "kmeans_seed",
                "svd_n_iter",
            )
        }
    )
    artifact = FrozenClusterArtifact(
        schema_version=str(manifest["schema_version"]),
        dataset_fingerprint=str(manifest["dataset_fingerprint"]),
        training_history_fingerprint=str(manifest["training_history_fingerprint"]),
        config=config,
        user_ids=tuple(int(value) for value in values.pop("user_ids")),
        item_ids=tuple(int(value) for value in values.pop("item_ids")),
        cluster_by_user=assignments,
        artifact_fingerprint=str(manifest["artifact_fingerprint"]),
        **values,
    )
    _validate_cluster_artifact(artifact)
    if complete.get("artifact_fingerprint") != artifact.artifact_fingerprint:
        raise ValueError("cluster completion and manifest fingerprints differ")
    return artifact


def _canonicalize_svd_signs(components: np.ndarray, reduced: np.ndarray) -> None:
    for component_index, component in enumerate(components):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            components[component_index] *= -1.0
            reduced[:, component_index] *= -1.0


def _canonicalize_cluster_labels(
    labels: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(
        range(len(centers)),
        key=lambda index: tuple(float(value) for value in centers[index]),
    )
    old_to_new = {old: new for new, old in enumerate(order)}
    canonical_labels = np.asarray(
        [old_to_new[int(label)] for label in labels], dtype=np.int64
    )
    return canonical_labels, centers[np.asarray(order, dtype=np.int64)].copy()


def _cluster_payload(artifact: FrozenClusterArtifact) -> Mapping[str, Any]:
    return {
        "dataset_fingerprint": artifact.dataset_fingerprint,
        "training_history_fingerprint": artifact.training_history_fingerprint,
        "config": artifact.config.to_dict(),
        "user_ids": artifact.user_ids,
        "item_ids": artifact.item_ids,
        "cluster_by_user": artifact.cluster_by_user,
        "svd_components": artifact.svd_components,
        "svd_explained_variance": artifact.svd_explained_variance,
        "svd_explained_variance_ratio": artifact.svd_explained_variance_ratio,
        "svd_singular_values": artifact.svd_singular_values,
        "reduced_user_vectors": artifact.reduced_user_vectors,
        "cluster_centers": artifact.cluster_centers,
    }


def _validate_cluster_artifact(artifact: FrozenClusterArtifact) -> None:
    if artifact.schema_version != CLUSTER_SCHEMA:
        raise ValueError(f"unsupported cluster schema: {artifact.schema_version}")
    if set(artifact.user_ids) != set(artifact.cluster_by_user):
        raise ValueError("cluster assignments do not cover every user")
    if artifact.reduced_user_vectors.shape != (
        len(artifact.user_ids),
        artifact.config.n_components,
    ):
        raise ValueError("reduced user-vector shape mismatch")
    if artifact.svd_components.shape != (
        artifact.config.n_components,
        len(artifact.item_ids),
    ):
        raise ValueError("SVD component shape mismatch")
    if artifact.cluster_centers.shape != (
        artifact.config.n_clusters,
        artifact.config.n_components,
    ):
        raise ValueError("cluster-center shape mismatch")
    labels = set(artifact.cluster_by_user.values())
    if labels != set(range(artifact.config.n_clusters)):
        raise ValueError("cluster labels must be canonical contiguous integers")
    expected = fingerprint_semantic_state(_cluster_payload(artifact), CLUSTER_SCHEMA)
    if artifact.artifact_fingerprint != expected:
        raise ValueError("cluster artifact fingerprint mismatch")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
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


__all__ = [
    "CLUSTER_FILE_SCHEMA",
    "CLUSTER_SCHEMA",
    "ClusterPipelineConfig",
    "FrozenClusterArtifact",
    "fit_user_clusters",
    "load_cluster_artifact",
    "save_cluster_artifact",
]
