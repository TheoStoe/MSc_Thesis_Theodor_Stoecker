"""Freeze one feature-balanced Goodreads cohort before any simulated outcomes.

This adapts the MovieLens five-cluster proportional-quota, 20,000-seed minimax
search. The frozen source feature cache, not a simulated outcome or author label,
defines every balance target. Structural readiness is assessed after selection
and never changes the winning population or removes a selected reader.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from . import sampling
from .data_preparation.goodreads import INDEX_SCHEMA, connect_index, settings

PROTOCOL = "goodreads-cluster-proportional-minimax-v1"
SEARCH_SCHEMA = "goodreads-representative-search-v1"
DEFAULT_SEED_COUNT = 20_000
CLUSTER_COUNT = 5
BEHAVIOR_FEATURES = ("training_count", "log_training_count", "mean_rating",
                     "mean_log_item_popularity")
OTHER_FEATURES = ("head_share", "training_span_days")
RATING_FEATURES = tuple(f"rating_share_{rating}" for rating in range(1, 6))
# These are fixed before any real candidate is evaluated. The first three
# thresholds reproduce the MovieLens combined-300-reader balance criteria.
TOLERANCES = {
    "cluster_tvd": .012,
    "activity_decile_tvd": .10,
    "behavior_max_abs_smd": .13,
    "rating_histogram_tvd": .05,
    "other_mean_abs_smd": .13,
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _progress(path, phase, **details):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                                 "phase": phase, **details}, sort_keys=True) + "\n")


def _load_features(cache):
    from .representative_features import load_features
    values = load_features(cache)
    n = len(values["user_keys"])
    if n < sampling.COHORT_SIZE:
        raise ValueError("At least 300 eligible readers are required")
    if len(values["source_user_ids"]) != n or len(set(values["source_user_ids"].tolist())) != n:
        raise ValueError("Feature cache source reader IDs are not unique")
    if len(set(values["user_keys"].tolist())) != n:
        raise ValueError("Feature cache user keys are not unique")
    if np.asarray(values["feature_values"]).shape != (n, len(values["feature_names"])):
        raise ValueError("Feature cache dimensions disagree")
    if not np.isfinite(values["feature_values"]).all():
        raise ValueError("Feature cache contains nonfinite values")
    if len(values["cluster_labels"]) != n or set(values["cluster_labels"].tolist()) != set(range(CLUSTER_COUNT)):
        raise ValueError("Exactly five nonempty frozen clusters are required")
    if len(values["activity_decile"]) != n:
        raise ValueError("Activity deciles do not match eligible readers")
    missing = set((*BEHAVIOR_FEATURES, *OTHER_FEATURES, *RATING_FEATURES)) - set(values["feature_names"])
    if missing:
        raise ValueError(f"Missing declared training-only features: {sorted(missing)}")
    unexpected = set(values["feature_names"]) - set((*BEHAVIOR_FEATURES, *OTHER_FEATURES, *RATING_FEATURES))
    if unexpected:
        raise ValueError(f"Undeclared selection features are not permitted: {sorted(unexpected)}")
    if len(set(values["feature_names"])) != len(values["feature_names"]):
        raise ValueError("Feature names must be unique")
    return values


def _feature_digest(features):
    digest = hashlib.sha256(_canonical({"metadata": features["metadata"],
                                       "feature_names": features["feature_names"]}))
    for key in ("user_keys", "source_user_ids", "cluster_labels", "activity_decile", "feature_values"):
        values = np.ascontiguousarray(features[key])
        if values.dtype.hasobject:
            digest.update(_canonical(values.tolist()))
        else:
            digest.update(_canonical({"array": key, "dtype": values.dtype.str,
                                       "shape": list(values.shape)}))
            digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def proportional_quota(n, sizes):
    """Hamilton largest-remainder allocation, with stable cluster-ID ties."""
    sizes = np.asarray(sizes, dtype=np.int64)
    if n < 0 or n > int(sizes.sum()) or (sizes < 0).any() or not sizes.sum():
        raise ValueError("Invalid population size or proportional allocation")
    expected = n * sizes.astype(np.float64) / sizes.sum()
    result = np.floor(expected).astype(np.int64)
    order = np.lexsort((np.arange(len(sizes)), -(expected - result)))
    result[order[:n - int(result.sum())]] += 1
    if (result > sizes).any():
        raise ValueError("Quota exceeds the available cluster population")
    return result


def _design(features):
    labels = np.asarray(features["cluster_labels"], dtype=np.int64)
    positions = [np.flatnonzero(labels == number) for number in range(CLUSTER_COUNT)]
    sizes = np.array([len(part) for part in positions], dtype=np.int64)
    matrix = np.asarray(features["feature_values"], dtype=np.float64)
    names = features["feature_names"]
    mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=0)
    # A constant source feature has zero discrepancy in every possible cohort.
    scale = np.where(sd > 0, sd, 1.)
    levels, deciles = np.unique(features["activity_decile"], return_inverse=True)
    decile_target = np.bincount(deciles, minlength=len(levels)) / len(labels)
    return {"positions": positions, "cluster_sizes": sizes,
            "cluster_labels": labels,
            "quota": proportional_quota(sampling.COHORT_SIZE, sizes),
            "cluster_target": sizes / sizes.sum(), "activity_codes": deciles,
            "activity_levels": levels, "activity_target": decile_target,
            "matrix": matrix, "mean": mean, "sd": sd, "scale": scale,
            "behavior": [names.index(name) for name in BEHAVIOR_FEATURES],
            "ratings": [names.index(name) for name in RATING_FEATURES],
            "other": [names.index(name) for name in OTHER_FEATURES]}


def draw_population(seed, design):
    """Draw only the quota from each stratum; never shuffle the full source."""
    rng = np.random.default_rng(int(seed))
    selected = np.concatenate([rng.choice(positions, size=int(quota), replace=False)
                               for positions, quota in zip(design["positions"], design["quota"])])
    return rng.permutation(selected)


def draw_indices(panel, seed, cohort_size=sampling.COHORT_SIZE):
    """Public reproduction helper for the notebook's selected-reader plots."""
    if cohort_size != sampling.COHORT_SIZE:
        raise ValueError("The frozen protocol requires exactly 300 active readers")
    return draw_population(seed, _design(panel))


def balance_rows(panel, indices):
    """Detailed feature and categorical balance without fitting any model."""
    design = _design(panel)
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) != sampling.COHORT_SIZE or len(np.unique(indices)) != len(indices):
        raise ValueError("Balance requires 300 distinct selected-reader positions")
    if (indices < 0).any() or (indices >= len(panel["user_keys"])).any():
        raise ValueError("Selected-reader positions are outside the feature cache")
    selected_mean = design["matrix"][indices].mean(axis=0)
    continuous = [{"feature": name, "eligible_mean": float(design["mean"][j]),
                   "selected_mean": float(selected_mean[j]),
                   "selected_minus_eligible": float(selected_mean[j] - design["mean"][j]),
                   "absolute_smd": float(abs(selected_mean[j] - design["mean"][j]) / design["scale"][j])}
                  for j, name in enumerate(panel["feature_names"])]
    categorical = []
    for label, values in (("cluster", np.asarray(panel["cluster_labels"])),
                          ("activity_decile", np.asarray(panel["activity_decile"]))):
        for level in np.unique(values):
            population_share = float(np.mean(values == level))
            sample_share = float(np.mean(values[indices] == level))
            categorical.append({"feature": label, "category": int(level),
                                "eligible_share": population_share,
                                "selected_share": sample_share,
                                "selected_minus_eligible": sample_share - population_share})
    return {"continuous": continuous, "categorical": categorical,
            "metrics": _metrics(indices, design)}


def _metrics(selected, design):
    mean = design["matrix"][selected].mean(axis=0)
    smd = np.abs(mean - design["mean"]) / design["scale"]
    cluster = np.bincount(design["cluster_labels"][selected], minlength=CLUSTER_COUNT) / len(selected)
    activity = np.bincount(design["activity_codes"][selected],
                           minlength=len(design["activity_levels"])) / len(selected)
    return {
        "cluster_tvd": float(np.abs(cluster - design["cluster_target"]).sum() / 2),
        "activity_decile_tvd": float(np.abs(activity - design["activity_target"]).sum() / 2),
        "behavior_max_abs_smd": float(smd[design["behavior"]].max()),
        "rating_histogram_tvd": float(np.abs(mean[design["ratings"]] - design["mean"][design["ratings"]]).sum() / 2),
        "other_mean_abs_smd": float(smd[design["other"]].mean()),
    }


def _record(seed, design):
    metrics = _metrics(draw_population(seed, design), design)
    ratios = [metrics[name] / limit for name, limit in TOLERANCES.items()]
    return {"seed": int(seed), "worst_ratio": max(ratios),
            "mean_ratio": float(np.mean(ratios)), "passes": max(ratios) <= 1 + 1e-12,
            **metrics}


def search_populations(cache, destination, *, seed_count=DEFAULT_SEED_COUNT, progress_path=None):
    """Save the full deterministic baseline-only search; never run a model."""
    if isinstance(seed_count, bool) or not isinstance(seed_count, int) or seed_count < 1:
        raise ValueError("seed_count must be a positive integer")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Search destination already exists: {destination}")
    features = _load_features(cache)
    design = _design(features)
    digest = _feature_digest(features)
    protocol = {"schema_version": SEARCH_SCHEMA, "protocol": PROTOCOL,
                "feature_cache": str(Path(cache).resolve()), "feature_cache_sha256": digest,
                "feature_cache_metadata": features["metadata"],
                "feature_names": features["feature_names"],
                "target_users": sampling.COHORT_SIZE, "eligible_users": len(features["user_keys"]),
                "cluster_count": CLUSTER_COUNT, "cluster_counts": design["cluster_sizes"].tolist(),
                "cluster_quotas": design["quota"].tolist(),
                "seed_range": [0, seed_count - 1], "seed_count": seed_count,
                "tolerances": dict(TOLERANCES),
                "ranking": ["worst_ratio", "mean_ratio", "seed"],
                "sampling_algorithm": "numpy.default_rng.PCG64.choice_without_replacement_within_frozen_clusters_then_permute",
                "behavior_features": list(BEHAVIOR_FEATURES),
                "rating_histogram_features": list(RATING_FEATURES),
                "other_mean_abs_smd_features": list(OTHER_FEATURES),
                "target_weighting": "Each eligible reader has equal weight; rating shares are user-level shares",
                "selection_inputs": "Training histories only, after chronological leave-last-two exclusion",
                "author_gender_or_simulated_outcome_selection": False,
                "structural_selection": "None; assess the one best passing feature-balanced cohort after selection",
                "genre_selection": "Genres are outside this fixed cache feature set; no genres are inferred",
                "versions": {"python": platform.python_version(), "numpy": np.__version__}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    # The immutable protocol is written before candidate metrics exist.
    _json(destination / "protocol.json", protocol)
    records = []
    _progress(progress_path, "population_seed_search", completed=0, total=seed_count)
    for seed in range(seed_count):
        records.append(_record(seed, design))
        if (seed + 1) % 1000 == 0 or seed + 1 == seed_count:
            _progress(progress_path, "population_seed_search", completed=seed + 1, total=seed_count)
    records.sort(key=lambda row: (row["worst_ratio"], row["mean_ratio"], row["seed"]))
    table = destination / "all_candidates.csv"
    with table.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rank", *records[0]])
        writer.writeheader()
        writer.writerows({"rank": rank, **row} for rank, row in enumerate(records, 1))
    passing = [row for row in records if row["passes"]]
    best = passing[0] if passing else None
    summary = {"schema_version": SEARCH_SCHEMA, "protocol": PROTOCOL,
               "protocol_file": "protocol.json", "protocol_sha256": _sha(destination / "protocol.json"),
               "candidate_table": table.name, "candidate_table_sha256": _sha(table),
               "feature_cache_sha256": digest,
               "candidate_count": seed_count, "passing_count": len(passing),
               "selected_seed": best["seed"] if best else None,
               "selected_record": best, "top_three_passing": passing[:3],
               "selected_source_user_ids": [str(features["source_user_ids"][i]) for i in draw_population(best["seed"], design)] if best else [],
               "eligible_feature_means": dict(zip(features["feature_names"], design["mean"].tolist())),
               "eligible_feature_standard_deviations": dict(zip(features["feature_names"], design["sd"].tolist())),
               "interpretation": "Best among the searched candidates under declared baseline balance criteria; not an unfiltered random sample or proof of exact representation",
               "shared_population": "One selected cohort is shared across all conditions and simulation seeds"}
    summary["summary_sha256"] = hashlib.sha256(_canonical(summary)).hexdigest()
    _json(destination / "summary.json", summary)
    _progress(progress_path, "population_search_complete", selected_seed=summary["selected_seed"],
              passing_candidates=len(passing), candidate_count=seed_count)
    return destination / "summary.json"


def load_search(search_root):
    """Verify immutable search reports, retaining every candidate's rank."""
    root = Path(search_root)
    if root.is_file():
        root = root.parent
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    declared = summary.pop("summary_sha256", None)
    if declared != hashlib.sha256(_canonical(summary)).hexdigest():
        raise ValueError("Search summary fingerprint does not match")
    summary["summary_sha256"] = declared
    if summary.get("schema_version") != SEARCH_SCHEMA or summary.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported representative search protocol")
    for label, key in (("protocol", "protocol_file"), ("candidate_table", "candidate_table")):
        name = summary[key]
        if Path(name).name != name or _sha(root / name) != summary[f"{label}_sha256"]:
            raise ValueError(f"Search {label} fingerprint does not match")
    protocol = json.loads((root / summary["protocol_file"]).read_text(encoding="utf-8"))
    with (root / summary["candidate_table"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != summary["candidate_count"]:
        raise ValueError("Search candidate count disagrees with its table")
    for row in rows:
        for key, value in row.items():
            row[key] = int(value) if key in {"rank", "seed"} else value == "True" if key == "passes" else float(value)
    return {"summary": summary, "protocol": protocol, "rows": rows}


def _selected_diagnostics(users, items):
    events = [event for user in users for event in [*user["train"], user["validation"], user["test"]]]
    ratings = Counter(str(event["rating"]) for event in events)
    histories = np.asarray([len(user["train"]) + 2 for user in users])
    return {"users": len(users), "unique_user_work_interactions": len(events),
            "distinct_items": len(items), "history_length": {
                "min": int(histories.min()), "median": float(np.median(histories)),
                "p90": float(np.quantile(histories, .9)), "max": int(histories.max()),
                "mean": float(histories.mean())},
            "mean_rating": float(np.mean([event["rating"] for event in events])),
            "rating_counts": dict(ratings),
            "rating_shares": {key: value / len(events) for key, value in ratings.items()},
            "scope": "Complete observed histories; descriptive only and never used for selection"}


def freeze_selected_population(index, cache, search_root, output_root, *, candidate_rank=1):
    """Export a prespecified passing rank from the original immutable search.

    The default remains the original winner with identical sampling metadata.
    Additional populations retain their original rank and all 300 selected users;
    neither annotation coverage nor simulation outcomes can reorder candidates.
    """
    if isinstance(candidate_rank, bool) or not isinstance(candidate_rank, int) or candidate_rank < 1:
        raise ValueError("candidate_rank must be a positive integer")
    output_root = Path(output_root).resolve()
    output = output_root / "cohort.json"
    if output_root.exists():
        raise FileExistsError(f"Frozen population destination already exists: {output_root}")
    loaded = load_search(search_root)
    summary, protocol = loaded["summary"], loaded["protocol"]
    features = _load_features(cache)
    if _feature_digest(features) != summary["feature_cache_sha256"]:
        raise ValueError("Feature cache changed since the population search")
    cached_index_hash = features["metadata"].get("source_index", {}).get("sha256")
    if cached_index_hash and _sha(index) != cached_index_hash:
        raise ValueError("Source index content differs from the frozen feature cache")
    if protocol["tolerances"] != TOLERANCES:
        raise ValueError("Search tolerances disagree with the current declared protocol")
    if summary["selected_seed"] is None:
        raise ValueError("No candidate passed all declared feature-balance tolerances")
    passing = [row for row in loaded["rows"] if row["passes"]]
    if candidate_rank > len(passing):
        raise ValueError("candidate_rank exceeds the passing candidates in the pinned search")
    ranked = passing[candidate_rank - 1]
    seed = ranked["seed"]
    expected_record = {key: value for key, value in ranked.items() if key != "rank"}
    design = _design(features)
    record = _record(seed, design)
    if record != expected_record or not record["passes"]:
        raise ValueError("Selected population metrics do not reproduce the pinned search")
    selected = draw_population(seed, design)
    source_ids = [str(features["source_user_ids"][i]) for i in selected]
    if candidate_rank == 1 and (seed != summary["selected_seed"] or source_ids != summary["selected_source_user_ids"]):
        raise ValueError("Selected source user IDs disagree with the search")
    connection = connect_index(index, readonly=True)
    try:
        config = settings(connection)
        if config.get("schema_version") != INDEX_SCHEMA or not config.get("complete"):
            raise ValueError("Cannot export from an incomplete or incompatible source index")
        expected_source = features["metadata"].get("source_manifest")
        if expected_source is None or config["source_manifest"] != expected_source:
            raise ValueError("Source index manifest differs from the feature cache")
        min_history = int(features["metadata"].get("min_history", 5))
        if min_history < 5:
            raise ValueError("Feature cache cannot weaken the history minimum")
        chosen = [(int(features["user_keys"][i]), str(features["source_user_ids"][i]), int(i)) for i in selected]
        connection.execute("CREATE TEMP TABLE selected_users (user_key INTEGER PRIMARY KEY)")
        connection.executemany("INSERT INTO selected_users VALUES (?)", [(key,) for key, _, _ in chosen])
        for key, source_id, _ in chosen:
            actual = connection.execute("SELECT u.source_user_id,h.history_length FROM users u JOIN user_summary h USING(user_key) WHERE user_key=?", (key,)).fetchone()
            if actual is None or actual["source_user_id"] != source_id or actual["history_length"] < min_history:
                raise ValueError("Selected source user identity or eligibility differs from the cache")
        item_rows = connection.execute("""SELECT DISTINCT i.item_key,i.source_item_id
            FROM selected_users s CROSS JOIN interactions r ON r.user_key=s.user_key
            JOIN items i ON r.item_key=i.item_key ORDER BY i.source_item_id""").fetchall()
        mapping = {row["item_key"]: number for number, row in enumerate(item_rows, 1)}
        items = [sampling._export_item(connection, row["item_key"], mapping[row["item_key"]]) for row in item_rows]
        users = []
        for number, (key, source_id, position) in enumerate(chosen, 1):
            history = [{"item_id": mapping[row["item_key"]], "rating": row["rating"],
                        "timestamp": row["timestamp"], "source_row": row["source_row"],
                        "source_book_id": row["source_book_id"]}
                       for row in connection.execute("SELECT * FROM interactions WHERE user_key=? ORDER BY timestamp,source_row,item_key", (key,))]
            if len(history) - 2 != features["feature_values"][position, features["feature_names"].index("training_count")]:
                raise ValueError("Selected source training history changed since feature extraction")
            users.append({"user_id": number, "source_user_id": source_id,
                          "train": history[:-2], "validation": history[-2], "test": history[-1]})
        source_count = connection.execute("SELECT count(*),sum(history_length) FROM user_summary").fetchone()
        full = {"users": source_count[0], "unique_user_work_interactions": source_count[1],
                "scope": "All indexed distinct valid dated rated user-work histories; descriptive eligibility denominator"}
        graph = sampling._graph_diagnostics(users, items)
        eligible_means = design["mean"]
        selected_means = design["matrix"][selected].mean(axis=0)
        feature_balance = [{"feature": name, "eligible_mean": float(eligible_means[j]),
                            "selected_mean": float(selected_means[j]),
                            "selected_minus_eligible": float(selected_means[j] - eligible_means[j]),
                            "absolute_smd": float(abs(selected_means[j] - eligible_means[j]) / design["scale"][j])}
                           for j, name in enumerate(features["feature_names"])]
        coverage = {"catalogue_items": len(items),
                    "items_without_book_metadata": sum(not item["metadata_present"] for item in items),
                    "items_without_work_id": sum(not item["work_id"] for item in items),
                    "items_without_primary_author_id": sum(not item["primary_author_ids"] for item in items),
                    "items_without_literary_author_names": sum(not item["authors"] for item in items),
                    "unresolved_contributor_author_ids": sorted({p["author_id"] for item in items for p in item["contributors"] if not p["name"]}),
                    "items_without_genres": sum(not item["genres"] for item in items)}
        cohort = {"schema_version": sampling.COHORT_SCHEMA, "dataset": "UCSD Goodreads Book Graph",
                  "users": users, "items": items, "source_manifest": config["source_manifest"],
                  "sampling": {"seed": seed, "method": PROTOCOL,
                               "eligible_population": "All indexed users with at least min_history distinct valid dated rated works; no gender/topic/outcome restrictions",
                               "min_history": min_history, "target_users": sampling.COHORT_SIZE,
                               "selected_users": len(users), "eligible_users": len(features["user_keys"]),
                               "reserve_users": 0, "background_training_users": 0,
                               "split": "chronological leave-last-two: train, validation, test; ties by original source row",
                               "no_postselection_pruning": True, "date_policy": config["date_policy"],
                               "duplicate_policy": config["duplicate_policy"],
                               "representativeness": f"Baseline covariate-balanced sample selected by the declared {summary['candidate_count']:,}-seed search; exact representation of every full-dataset property is not guaranteed",
                               "gender_or_outcome_selection": False,
                               "shared_across_conditions_and_simulation_seeds": True,
                               "search_candidate_count": summary["candidate_count"],
                               "search_summary_sha256": summary["summary_sha256"],
                               "feature_cache_sha256": summary["feature_cache_sha256"],
                               "cluster_counts": design["cluster_sizes"].tolist(),
                               "cluster_quotas": design["quota"].tolist(),
                               "selected_user_cluster_labels": [int(features["cluster_labels"][i]) for i in selected]},
                  "diagnostics": {"full_valid_dated_rated_source": full,
                                  "eligible_population": {"users": len(features["user_keys"]),
                                      "training_interactions": int(design["matrix"][:, features["feature_names"].index("training_count")].sum()),
                                      "feature_means": dict(zip(features["feature_names"], eligible_means.tolist()))},
                                  "selected_cohort_all_observed_histories": _selected_diagnostics(users, items),
                                  "training_feature_balance": feature_balance, "balance_metrics": record,
                                  "balance_tolerances": dict(TOLERANCES),
                                  "genre_semantics": "Only supplied source genres are retained; no genres or genders are inferred",
                                  "metadata_coverage": coverage, "training_graph": graph,
                                  "source_feature_cache_metadata": features["metadata"],
                                  "ingest_audit": config.get("ingest_audit", {})},
                  "readiness": sampling._readiness(users, items, graph, len(features["user_keys"]))}
        if candidate_rank != 1:
            cohort["sampling"].update({
                "selected_passing_rank": candidate_rank,
                "original_search_rank": ranked["rank"],
                "original_search_winner_seed": summary["selected_seed"],
                "shared_across_conditions_and_simulation_seeds": False,
                "replication_scope": "This population is shared across matched conditions for its assigned simulation seed; other replication populations differ.",
                "selection_rule": "Successive passing ranks of the original baseline-only search; no reranking by gender labels or simulated outcomes.",
            })
        cohort["cohort_sha256"] = sampling.cohort_digest(cohort)
        sampling._validate_cohort(cohort)
        raw = (json.dumps(cohort, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        manifest = {"schema_version": "goodreads-cohort-manifest-v1", "cohort_file": output.name,
                    "sha256": hashlib.sha256(raw).hexdigest(), "cohort_sha256": cohort["cohort_sha256"],
                    "source_manifest": cohort["source_manifest"], "sampling": cohort["sampling"]}
        output_root.parent.mkdir(parents=True, exist_ok=True)
        output_root.mkdir()
        with output.open("xb") as stream:
            stream.write(raw)
        _json(output.with_suffix(".manifest.json"), manifest)
        # Verify exactly the format consumed by all existing local preparation.
        sampling.load_cohort(output)
        return output
    finally:
        connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--search", required=True, type=Path)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--candidate-rank", type=int, default=1,
                        help="Prespecified passing rank of the existing frozen search; default 1 preserves the original winner.")
    args = parser.parse_args(argv)
    try:
        if args.freeze is not None:
            if args.index is None:
                parser.error("--freeze requires --index")
            result = freeze_selected_population(args.index, args.cache, args.search, args.freeze,
                                                candidate_rank=args.candidate_rank)
        else:
            result = search_populations(args.cache, args.search, seed_count=args.seed_count,
                                        progress_path=args.progress)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Representative sampling failed: {error}\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
