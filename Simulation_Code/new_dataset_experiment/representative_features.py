"""Train-only baseline features for a frozen representative Goodreads cohort.

The source SQLite database is read-only.  One ordered interaction scan removes
each eligible reader's two chronological holdouts before any feature, item
popularity or clustering input is formed.  Sparse binary histories use the same
normalization/SVD/KMeans procedure as the earlier MovieLens sampling design.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from operator import itemgetter
from pathlib import Path
import time
import warnings

import numpy as np

from .data_preparation.goodreads import INDEX_SCHEMA, connect_index, settings

SCHEMA = "goodreads-representative-features-v1"
FEATURE_NAMES = (
    "training_count", "log_training_count", "mean_rating",
    "mean_log_item_popularity", "head_share",
    "rating_share_1", "rating_share_2", "rating_share_3",
    "rating_share_4", "rating_share_5", "training_span_days",
)
DEFAULT_CLUSTERING = {
    "n_components": 32, "n_clusters": 5, "n_init": 20,
    "svd_seed": 42, "kmeans_seed": 42, "svd_n_iter": 5,
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as target:
        json.dump(value, target, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        target.write("\n")


def _progress(path, phase, **details):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
              **details}
    with path.open("a", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                allow_nan=False) + "\n")


def _sha256(path, progress_path=None, phase="fingerprint"):
    digest = hashlib.sha256()
    size, checkpoint = 0, 0
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            if progress_path and size - checkpoint >= 512 * 1024 * 1024:
                _progress(progress_path, phase, bytes_read=size)
                checkpoint = size
    return digest.hexdigest()


def _index_stat(path):
    stat = Path(path).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _finalize_cache(staging, destination):
    """Retry transient Windows/OneDrive locks without replacing an artifact."""
    staging, destination = Path(staging).resolve(), Path(destination).resolve()
    if (staging.parent != destination.parent or
            staging.name != destination.name + ".incomplete"):
        raise ValueError("Cache completion must rename its checked sibling staging directory")
    for attempt in range(8):
        if destination.exists():
            raise FileExistsError(f"Feature cache appeared during preparation: {destination}")
        try:
            staging.rename(destination)
            return
        except PermissionError:
            # An existing result always takes precedence over retrying a lock.
            if destination.exists():
                raise FileExistsError(f"Feature cache appeared during preparation: {destination}")
            if attempt == 7:
                raise
            time.sleep(min(.25 * (attempt + 1), .5))


def _source_summary(connection, min_history):
    row = connection.execute("""SELECT count(*),sum(history_length),
        sum(CASE WHEN history_length>=? THEN 1 ELSE 0 END),
        sum(CASE WHEN history_length>=? THEN history_length ELSE 0 END)
        FROM user_summary""", (min_history, min_history)).fetchone()
    total_users, total_interactions, eligible_users, eligible_interactions = map(
        lambda value: int(value or 0), row)
    return {
        "indexed_users": total_users,
        "indexed_interactions": total_interactions,
        "eligible_users": eligible_users,
        "eligible_observed_interactions": eligible_interactions,
        "eligible_training_interactions": eligible_interactions - 2 * eligible_users,
        "excluded_users_below_minimum": total_users - eligible_users,
        "excluded_interactions_below_minimum": total_interactions - eligible_interactions,
        "heldout_interactions_excluded_from_features": 2 * eligible_users,
        "eligibility": "At least min_history distinct valid dated rated works, before leave-last-two splitting",
    }


def _training_matrix(connection, min_history, source_population, progress_path):
    """Read ordered source records in bounded reader groups, never whole JSON."""
    from scipy.sparse import csr_matrix

    users = connection.execute("""SELECT h.user_key,u.source_user_id,h.history_length
        FROM user_summary h JOIN users u USING(user_key)
        WHERE h.history_length>=? ORDER BY h.user_key""", (min_history,)).fetchall()
    if not users:
        raise ValueError("No readers meet the declared history minimum")
    user_keys = np.asarray([row[0] for row in users], dtype=np.int64)
    source_user_ids = np.asarray([row[1] for row in users], dtype=np.str_)
    training_counts = np.asarray([row[2] - 2 for row in users], dtype=np.int64)
    del users
    expected_nnz = int(training_counts.sum())
    if expected_nnz != source_population["eligible_training_interactions"]:
        raise ValueError("Source summary and eligible histories disagree")
    largest_user = int(connection.execute("SELECT max(user_key) FROM users").fetchone()[0])
    largest_item = int(connection.execute("SELECT max(item_key) FROM items").fetchone()[0])
    # Both IDs are integer primary keys assigned by the source importer.  Their
    # dense lookup arrays are tiny relative to the sparse training matrix.
    user_row = np.full(largest_user + 1, -1, dtype=np.int64)
    user_row[user_keys] = np.arange(len(user_keys), dtype=np.int64)
    indices_dtype = np.int32 if largest_item < np.iinfo(np.int32).max else np.int64
    indices = np.empty(expected_nnz, dtype=indices_dtype)
    indptr = np.empty(len(user_keys) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(training_counts, out=indptr[1:])
    values = np.zeros((len(user_keys), len(FEATURE_NAMES)), dtype=np.float64)
    values[:, 0] = training_counts
    values[:, 1] = np.log1p(training_counts)

    # Disabling Row construction avoids an extra object for every one of the
    # 104m source records. itertools.groupby holds at most one reader's rows.
    cursor = connection.cursor()
    cursor.row_factory = None
    cursor.execute("""SELECT user_key,item_key,rating,timestamp,source_row
        FROM interactions ORDER BY user_key,item_key""")
    scanned, completed = 0, 0
    _progress(progress_path, "training_history_scan", read_rows=0,
              eligible_readers=len(user_keys), expected_training_records=expected_nnz)
    for key, group in itertools.groupby(cursor, itemgetter(0)):
        records = list(group)
        scanned += len(records)
        row = int(user_row[key])
        if row < 0:
            continue
        records = np.asarray(records, dtype=np.int64)
        if len(records) != training_counts[row] + 2:
            raise ValueError("Indexed history length differs from user_summary")
        # Match cohort export exactly: (timestamp, source_row, item_key).
        order = np.lexsort((records[:, 1], records[:, 4], records[:, 3]))
        keep = np.ones(len(records), dtype=np.bool_)
        keep[order[-2:]] = False
        training = records[keep]
        begin, end = int(indptr[row]), int(indptr[row + 1])
        indices[begin:end] = training[:, 1]
        ratings = training[:, 2]
        if np.any((ratings < 1) | (ratings > 5)):
            raise ValueError("Training ratings must remain integers 1..5")
        values[row, 2] = float(np.mean(ratings))
        values[row, 5:10] = np.bincount(ratings, minlength=6)[1:6] / len(ratings)
        values[row, 10] = float(training[:, 3].max() - training[:, 3].min()) / 86400
        completed += 1
        if completed % 5000 == 0:
            _progress(progress_path, "training_history_scan", read_rows=scanned,
                      eligible_readers_completed=completed,
                      eligible_readers=len(user_keys))
    cursor.close()
    if completed != len(user_keys) or scanned != source_population["indexed_interactions"]:
        raise ValueError("Ordered source scan did not cover the complete source population")
    _progress(progress_path, "training_history_scan_complete", read_rows=scanned,
              eligible_readers_completed=completed, training_records=expected_nnz)

    _progress(progress_path, "training_item_popularity")
    source_counts = np.bincount(indices, minlength=largest_item + 1)
    item_keys = np.flatnonzero(source_counts).astype(np.int64, copy=False)
    item_popularity = source_counts[item_keys].astype(np.int64, copy=False)
    mapping = np.full(largest_item + 1, -1, dtype=indices_dtype)
    mapping[item_keys] = np.arange(len(item_keys), dtype=indices_dtype)
    # Compact columns include only eligible readers' training books; held-out
    # titles never alter the SVD dimension, popularity denominator or head set.
    indices[:] = mapping[indices]
    del mapping, source_counts, user_row
    matrix = csr_matrix((np.ones(expected_nnz, dtype=np.float64), indices, indptr),
                        shape=(len(user_keys), len(item_keys)))
    if not matrix.has_sorted_indices or matrix.nnz != expected_nnz:
        raise ValueError("Source histories are not distinct sorted reader/work records")

    head_count = max(1, math.ceil(.2 * len(item_keys)))
    popularity_order = np.lexsort((item_keys, -item_popularity))
    is_head = np.zeros(len(item_keys), dtype=np.bool_)
    is_head[popularity_order[:head_count]] = True
    log_popularity = np.log1p(item_popularity)
    for start in range(0, len(user_keys), 10000):
        stop = min(start + 10000, len(user_keys))
        first, last = int(matrix.indptr[start]), int(matrix.indptr[stop])
        offsets = matrix.indptr[start:stop] - first
        columns = matrix.indices[first:last]
        values[start:stop, 3] = np.add.reduceat(log_popularity[columns], offsets) / training_counts[start:stop]
        values[start:stop, 4] = np.add.reduceat(is_head[columns].astype(np.int64), offsets) / training_counts[start:stop]
        _progress(progress_path, "training_item_features", readers_completed=stop,
                  eligible_readers=len(user_keys))
    if not np.isfinite(values).all():
        raise ValueError("Baseline feature extraction produced non-finite values")
    return user_keys, source_user_ids, values, matrix, item_keys, item_popularity, head_count


def _cluster(matrix, config, progress_path):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.preprocessing import normalize

    components, clusters = config["n_components"], config["n_clusters"]
    if components >= min(matrix.shape):
        raise ValueError("SVD components must be smaller than both eligible readers and training items; use explicit smaller fixture settings only for tests")
    if clusters > matrix.shape[0]:
        raise ValueError("Clusters cannot exceed eligible readers")
    _progress(progress_path, "binary_history_normalization", shape=list(matrix.shape), nnz=matrix.nnz)
    normalized = normalize(matrix, norm="l2", axis=1, copy=True)
    svd = TruncatedSVD(n_components=components, n_iter=config["svd_n_iter"],
                       algorithm="randomized", random_state=config["svd_seed"])
    _progress(progress_path, "svd_fit", **config)
    reduced = np.asarray(svd.fit_transform(normalized), dtype=np.float64)
    del normalized
    # Canonical signs and label ordering match the original frozen pipeline.
    for index, component in enumerate(svd.components_):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            svd.components_[index] *= -1
            reduced[:, index] *= -1
    reduced = normalize(reduced, norm="l2", axis=1, copy=True)
    _progress(progress_path, "kmeans_fit", eligible_readers=matrix.shape[0], **config)
    kmeans = KMeans(n_clusters=clusters, n_init=config["n_init"],
                    random_state=config["kmeans_seed"], algorithm="lloyd")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            labels = kmeans.fit_predict(reduced)
        except ConvergenceWarning as error:
            raise ValueError("Training histories do not support the declared number of distinct clusters") from error
    order = sorted(range(clusters), key=lambda index: tuple(float(value) for value in kmeans.cluster_centers_[index]))
    remap = np.empty(clusters, dtype=np.int64)
    remap[np.asarray(order)] = np.arange(clusters)
    labels = remap[labels]
    model_arrays = {
        "svd_components": np.asarray(svd.components_, dtype=np.float64),
        "svd_explained_variance_ratio": svd.explained_variance_ratio_,
        "svd_explained_variance": svd.explained_variance_,
        "svd_singular_values": svd.singular_values_,
        "reduced_user_vectors": reduced,
        "cluster_centers": kmeans.cluster_centers_[np.asarray(order)].copy(),
    }
    _progress(progress_path, "clustering_complete", cluster_counts=np.bincount(labels, minlength=clusters).tolist())
    return labels.astype(np.int64), model_arrays


def prepare_features(index, destination, *, progress_path=None,
                     min_history=5, clustering_settings=None):
    """Create an exclusive immutable offline feature cache and return metadata.

    The default clustering settings reproduce the MovieLens design. Explicit
    reduced dimensions are available for small synthetic fixtures; real runs
    should retain the defaults, which are recorded in the completion manifest.
    No simulation, model fitting for recommendations or provider call occurs.
    """
    if isinstance(min_history, bool) or not isinstance(min_history, int) or min_history < 5:
        raise ValueError("History minimum must be at least five distinct works")
    config = dict(DEFAULT_CLUSTERING)
    if clustering_settings is not None:
        unknown = set(clustering_settings) - set(config)
        if unknown:
            raise ValueError(f"Unknown clustering settings: {sorted(unknown)}")
        config.update(clustering_settings)
    for name, value in config.items():
        lower = 0 if name.endswith("_seed") else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"Invalid clustering setting: {name}")
    index = Path(index).resolve()
    root = Path(destination).resolve()
    staging = root.with_name(root.name + ".incomplete")
    if root.exists() or staging.exists():
        raise FileExistsError(f"Feature cache or incomplete preparation already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    _progress(progress_path, "feature_preparation_started", schema_version=SCHEMA)
    initial_stat = _index_stat(index)
    _progress(progress_path, "index_fingerprint", total_bytes=initial_stat["size_bytes"])
    index_sha256 = _sha256(index, progress_path, "index_fingerprint")
    connection = connect_index(index, readonly=True)
    try:
        source_settings = settings(connection)
        if source_settings.get("schema_version") != INDEX_SCHEMA or not source_settings.get("complete"):
            raise ValueError("Cannot extract features from an incompatible or incomplete source index")
        source_population = _source_summary(connection, min_history)
        _progress(progress_path, "eligible_population", **source_population)
        arrays = _training_matrix(connection, min_history, source_population, progress_path)
    finally:
        connection.close()
    user_keys, source_user_ids, feature_values, matrix, item_keys, popularity, head_count = arrays
    labels, model_arrays = _cluster(matrix, config, progress_path)
    decile_edges = np.quantile(feature_values[:, 0], np.arange(.1, 1, .1), method="linear")
    activity_decile = (np.searchsorted(decile_edges, feature_values[:, 0], side="right") + 1).astype(np.int64)
    _progress(progress_path, "cache_write", eligible_readers=len(user_keys), training_records=matrix.nnz)
    from scipy.sparse import save_npz
    save_npz(staging / "training_matrix.npz", matrix, compressed=False)
    np.savez(staging / "features.npz", user_keys=user_keys,
             source_user_ids=source_user_ids, feature_values=feature_values,
             cluster_labels=labels, activity_decile=activity_decile)
    np.savez(staging / "item_training_popularity.npz", item_keys=item_keys,
             training_popularity=popularity)
    np.savez(staging / "clustering_arrays.npz", **model_arrays)
    if _index_stat(index) != initial_stat:
        raise ValueError("The source index changed during feature preparation")
    import scipy
    import sklearn
    feature_means = feature_values.mean(axis=0)
    feature_stds = feature_values.std(axis=0, ddof=0)
    files = {}
    for path in sorted(staging.glob("*.npz")):
        _progress(progress_path, "cache_fingerprint", artifact=path.name)
        files[path.name] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    metadata = {
        "schema_version": SCHEMA,
        "status": "complete",
        "source_index": {"path": str(index), **initial_stat, "sha256": index_sha256},
        "index_schema": source_settings["schema_version"],
        "source_settings_sha256": hashlib.sha256(_canonical(source_settings)).hexdigest(),
        "source_manifest": source_settings["source_manifest"],
        "source_population": source_population,
        "min_history": min_history,
        "eligible_users": len(user_keys),
        "feature_names": list(FEATURE_NAMES),
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "feature_statistics": {name: {"mean": float(feature_means[i]),
                                       "std": float(feature_stds[i]),
                                       "min": float(feature_values[:, i].min()),
                                       "max": float(feature_values[:, i].max())}
                               for i, name in enumerate(FEATURE_NAMES)},
        "feature_population_weighting": "One equal-weight observation per eligible reader; feature_stds use population ddof=0",
        "history_scope": "Only eligible readers' training histories after chronological leave-last-two; two latest records sorted by timestamp, source_row, item_key excluded before all features, popularity and clustering",
        "training_matrix": {"shape": list(matrix.shape), "nnz": int(matrix.nnz),
                            "values": "Binary reader/work presence; float64",
                            "columns": "Sorted source item keys observed in eligible training histories only"},
        "item_popularity": {"population": "All eligible readers' training records only",
                            "count": "Distinct eligible training readers per work",
                            "log_transform": "natural log(1 + count)",
                            "head_fraction": .2, "head_item_count": head_count,
                            "head_rule": "Top ceil(20% of training-observed works) by training reader count; ties by ascending source item key; fixed before seed search"},
        "activity_deciles": {"edges": decile_edges.tolist(), "labels": "1..10",
                             "rule": "Linear population quantiles of training count; right insertion; equal counts remain in the same bin"},
        "clustering": {**config, "input_normalization": "l2_rows",
                       "reduction": "sklearn_TruncatedSVD_randomized",
                       "reduced_normalization": "l2_rows",
                       "clustering": "sklearn_KMeans_lloyd",
                       "cluster_label_rule": "lexicographic_centers_v1",
                       "cluster_counts": np.bincount(labels, minlength=config["n_clusters"]).tolist()},
        "available_metadata": {"genres_available": bool(source_settings.get("genres_available")),
                               "genres_used_for_selection": False,
                               "genre_note": "Optional genre source is absent in the current index; no genre labels are inferred from shelf names or descriptions" if not source_settings.get("genres_available") else "Genres are source-supplied but not part of this version's fixed feature set",
                               "reader_demographics_available": False,
                               "author_gender_used_for_selection": False},
        "gender_or_outcome_selection": False,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__},
        "files": files,
    }
    metadata["cache_fingerprint"] = hashlib.sha256(_canonical(metadata)).hexdigest()
    _write_json(staging / "metadata.json", metadata)
    _write_json(staging / "_COMPLETE.json", {
        "schema_version": SCHEMA, "cache_fingerprint": metadata["cache_fingerprint"],
        "metadata_sha256": _sha256(staging / "metadata.json"),
    })
    _finalize_cache(staging, root)
    _progress(progress_path, "feature_preparation_complete", eligible_users=len(user_keys),
              training_records=int(matrix.nnz), cache_fingerprint=metadata["cache_fingerprint"])
    return root / "metadata.json"


def load_features(cache):
    """Verify the immutable cache and return compact baseline search arrays."""
    cache = Path(cache).resolve()
    root = cache.parent if cache.name == "metadata.json" else cache
    metadata_path, complete_path = root / "metadata.json", root / "_COMPLETE.json"
    if not metadata_path.is_file() or not complete_path.is_file():
        raise ValueError("Feature cache is incomplete: metadata or completion marker is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA or complete.get("schema_version") != SCHEMA:
        raise ValueError("Incompatible feature cache schema")
    expected = metadata.get("cache_fingerprint")
    actual = hashlib.sha256(_canonical({key: value for key, value in metadata.items()
                                       if key != "cache_fingerprint"})).hexdigest()
    if expected != actual or complete.get("cache_fingerprint") != expected or complete.get("metadata_sha256") != _sha256(metadata_path):
        raise ValueError("Feature cache metadata fingerprint mismatch")
    required_files = {"features.npz", "item_training_popularity.npz", "training_matrix.npz", "clustering_arrays.npz"}
    if set(metadata.get("files", {})) != required_files:
        raise ValueError("Feature cache artifact inventory differs from the declared schema")
    for filename, record in metadata["files"].items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"Feature cache artifact fingerprint mismatch: {filename}")
    with np.load(root / "features.npz", allow_pickle=False) as source:
        arrays = {name: source[name] for name in ("user_keys", "source_user_ids", "feature_values", "cluster_labels", "activity_decile")}
    n_users = metadata["eligible_users"]
    if arrays["feature_values"].shape != (n_users, len(FEATURE_NAMES)) or metadata["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Feature cache dimensions or names disagree")
    if any(arrays[name].shape != (n_users,) for name in ("user_keys", "source_user_ids", "cluster_labels", "activity_decile")):
        raise ValueError("Feature cache user axes disagree")
    if not np.isfinite(arrays["feature_values"]).all() or not np.all(np.diff(arrays["user_keys"]) > 0):
        raise ValueError("Feature cache contains invalid feature values or duplicate reader keys")
    if (len(np.unique(arrays["source_user_ids"])) != n_users or
            np.any(arrays["cluster_labels"] < 0) or
            np.any(arrays["cluster_labels"] >= metadata["clustering"]["n_clusters"]) or
            np.any(arrays["activity_decile"] < 1) or np.any(arrays["activity_decile"] > 10)):
        raise ValueError("Feature cache contains invalid reader IDs, cluster labels or deciles")
    return {"metadata": metadata, "feature_names": list(FEATURE_NAMES), **arrays}


def main(argv=None):
    """Expose the existing local feature-preparation function as a portable CLI."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args(argv)
    print(prepare_features(args.index, args.output, progress_path=args.progress))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
