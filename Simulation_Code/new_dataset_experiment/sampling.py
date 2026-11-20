"""Freeze one uniform, outcome-independent 300-user Goodreads cohort."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path

from .data_preparation.goodreads import INDEX_SCHEMA, LITERARY_ROLES, connect_index, settings

COHORT_SCHEMA = "goodreads-cohort-v1"
POPULATION_SEED = 1054
COHORT_SIZE = 300
MIN_CANDIDATE_BUDGET = 40  # top 10 plus 30 possible unique consumptions
READINESS_POLICY = "goodreads-simulation-readiness-v3"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def cohort_digest(cohort):
    return hashlib.sha256(_canonical({k: v for k, v in cohort.items() if k != "cohort_sha256"})).hexdigest()


def _priority(seed, user_id):
    return int.from_bytes(hashlib.sha256(_canonical([seed, user_id])).digest(), "big")


def _choose_users(connection, seed):
    """Only 300 hash priorities reside in memory, irrespective of source size."""
    heap = []
    for row in connection.execute("SELECT e.user_key,u.source_user_id FROM eligible_users e JOIN users u USING(user_key)"):
        priority = _priority(seed, row["source_user_id"])
        entry = (-priority, row["source_user_id"], row["user_key"])
        if len(heap) < COHORT_SIZE:
            heapq.heappush(heap, entry)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    return sorted([(user_key, user_id) for _, user_id, user_key in heap], key=lambda x: x[1])


def _quantiles(connection, query, n):
    if not n:
        return {"min": None, "median": None, "p90": None, "max": None}
    values = {}
    for label, offset in (("min", 0), ("median", (n - 1) // 2), ("p90", int((n - 1) * .9)), ("max", n - 1)):
        values[label] = connection.execute(f"SELECT history_length FROM ({query}) ORDER BY history_length LIMIT 1 OFFSET ?", (offset,)).fetchone()[0]
    if n % 2 == 0:
        upper = connection.execute(f"SELECT history_length FROM ({query}) ORDER BY history_length LIMIT 1 OFFSET ?", (n // 2,)).fetchone()[0]
        values["median"] = (values["median"] + upper) / 2
    return values


def _population_summary(connection, user_table=None):
    # Table names are fixed internal constants, never source/user input.
    restriction = f"JOIN {user_table} p ON p.user_key=r.user_key" if user_table else ""
    # A fixed outer join order makes the selected cohort use 300 indexed history
    # ranges, rather than scanning every source interaction to test membership.
    interaction_source = ("selected_users p CROSS JOIN interactions r ON p.user_key=r.user_key"
                          if user_table == "selected_users" else f"interactions r {restriction}")
    history_restriction = f"JOIN {user_table} p ON p.user_key=h.user_key" if user_table else ""
    histories = f"SELECT h.user_key,h.history_length FROM user_summary h {history_restriction}"
    total = connection.execute(f"SELECT count(*),count(DISTINCT r.user_key),count(DISTINCT r.item_key),avg(r.rating) FROM {interaction_source}").fetchone()
    interactions, users, items, mean_rating = total
    activity = dict(connection.execute(f"""SELECT CASE WHEN history_length<5 THEN '1-4'
        WHEN history_length<10 THEN '5-9' WHEN history_length<20 THEN '10-19'
        WHEN history_length<50 THEN '20-49' WHEN history_length<100 THEN '50-99' ELSE '100+' END,
        count(*) FROM ({histories}) GROUP BY 1""").fetchall())
    user_ratings = dict(connection.execute(f"""SELECT CASE WHEN h.mean_rating<2 THEN '[1,2)'
        WHEN h.mean_rating<3 THEN '[2,3)' WHEN h.mean_rating<4 THEN '[3,4)' ELSE '[4,5]' END,
        count(*) FROM user_summary h {history_restriction} GROUP BY 1""").fetchall())
    user_mean_rating = connection.execute(f"SELECT avg(h.mean_rating) FROM user_summary h {history_restriction}").fetchone()[0]
    ratings = dict(connection.execute(f"SELECT CAST(r.rating AS TEXT),count(*) FROM {interaction_source} GROUP BY r.rating").fetchall())
    genres = dict(connection.execute(f"SELECT g.genre,count(*) FROM {interaction_source} JOIN genres g ON g.item_key=r.item_key GROUP BY g.genre").fetchall())
    missing_genres = connection.execute(f"SELECT count(*) FROM {interaction_source} WHERE NOT EXISTS (SELECT 1 FROM genres g WHERE g.item_key=r.item_key)").fetchone()[0]
    denom = interactions or 1
    return {"users": users, "unique_user_work_interactions": interactions, "distinct_items": items,
            "history_length": {**_quantiles(connection, histories, users),
                               "mean": interactions / users if users else None},
            "user_activity_counts": activity,
            "user_activity_shares": {k: v / users for k, v in activity.items()} if users else {},
            "mean_rating": mean_rating, "rating_counts": ratings,
            "rating_weighting": "mean_rating/rating_shares weight interactions; user_mean_rating weights users equally",
            "user_mean_rating": user_mean_rating,
            "user_mean_rating_bin_counts": user_ratings,
            "user_mean_rating_bin_shares": {k: v / users for k, v in user_ratings.items()} if users else {},
            "rating_shares": {k: v / denom for k, v in ratings.items()},
            "genre_interaction_presence_counts": genres,
            "genre_interaction_presence_shares": {k: v / denom for k, v in genres.items()},
            "interactions_without_genre": missing_genres,
            "genre_missing_share": missing_genres / denom if interactions else None}


def _export_item(connection, key, runtime_id):
    item = dict(connection.execute("SELECT * FROM items WHERE item_key=?", (key,)).fetchone())
    people = []
    for person in connection.execute("""SELECT c.author_id,a.name,c.role,c.primary_literary_author,c.book_id,c.ordinal
            FROM contributors c LEFT JOIN authors a ON a.author_id=c.author_id
            WHERE c.item_key=? ORDER BY c.book_id,c.ordinal""", (key,)):
        person = dict(person)
        person["name"] = person["name"] or ""
        person["primary_literary_author"] = bool(person["primary_literary_author"])
        people.append(person)
    literary = [p for p in people if p["role"].casefold() in LITERARY_ROLES]
    return {"item_id": runtime_id, "source_item_id": item["source_item_id"], "work_id": item["work_id"],
            "book_ids": [row[0] for row in connection.execute("SELECT book_id FROM book_map WHERE item_key=? ORDER BY book_id", (key,))],
            "title": item["title"], "description": item["description"],
            "authors": sorted({p["name"] for p in literary if p["name"]}),
            "author_ids": sorted({p["author_id"] for p in literary}),
            "primary_author_ids": sorted({p["author_id"] for p in people if p["primary_literary_author"]}),
            "contributors": people,
            "genres": [row[0] for row in connection.execute("SELECT genre FROM genres WHERE item_key=? ORDER BY genre", (key,))],
            "metadata_present": bool(item["metadata_present"])}


def _graph_diagnostics(users, items):
    """Support in the selected training graph, including retained unsupported items."""
    n = len(users)
    parent = list(range(n))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first, second):
        parent[find(first)] = find(second)

    owners = {}
    support = Counter()
    for number, user in enumerate(users):
        for event in user["train"]:
            item_id = event["item_id"]
            support[item_id] += 1
            if item_id in owners:
                union(number, owners[item_id])
            else:
                owners[item_id] = number
    components = Counter(find(i) for i in range(n))
    shared_counts, candidate_counts, conservative_counts = [], [], []
    heldout = {"validation": {"total": 0, "without_training_support": 0},
               "test": {"total": 0, "without_training_support": 0}}
    for user in users:
        shared_counts.append(sum(support[e["item_id"]] >= 2 for e in user["train"]))
        candidate_counts.append(len(items) - len(user["train"]))
        conservative_counts.append(len(items) - len(user["train"]) - 2)
        for split in heldout:
            heldout[split]["total"] += 1
            heldout[split]["without_training_support"] += int(support[user[split]["item_id"]] == 0)
    unsupported = [item["item_id"] for item in items if support[item["item_id"]] == 0]
    return {"training_interactions": sum(support.values()), "catalogue_items": len(items),
            "training_items": len(support), "catalogue_items_without_training_support": unsupported,
            "catalogue_items_without_training_support_count": len(unsupported),
            "training_item_user_support_histogram": {str(k): v for k, v in sorted(Counter(support.values()).items())},
            "training_user_components": len(components),
            "component_user_sizes_descending": sorted(components.values(), reverse=True),
            "largest_component_user_share": max(components.values(), default=0) / n if n else 0,
            "users_without_shared_training_items": sum(value == 0 for value in shared_counts),
            "per_user_shared_training_item_counts": shared_counts,
            "heldout_training_support": heldout,
            "candidate_budget": {"policy": "catalogue minus initial training history",
                                 "minimum": min(candidate_counts, default=0),
                                 "required_minimum": MIN_CANDIDATE_BUDGET,
                                 "users_below_required": sum(value < MIN_CANDIDATE_BUDGET for value in candidate_counts),
                                 "minimum_excluding_all_observed_history": min(conservative_counts, default=0)}}


def _readiness(users, items, graph, eligible_count):
    failures, warnings = [], []
    if len(users) != COHORT_SIZE:
        failures.append(f"Exactly {COHORT_SIZE} users required; eligible source has {eligible_count}")
    if graph["candidate_budget"]["users_below_required"]:
        failures.append("Some users have fewer than 40 unseen training-policy candidates (top 10 plus 30 consumptions)")
    if graph["users_without_shared_training_items"]:
        warnings.append(
            f"{graph['users_without_shared_training_items']} selected readers have no shared training item; "
            "retained as a descriptive training-graph diagnostic"
        )
    if graph["largest_component_user_share"] < .8:
        failures.append("Largest training component contains fewer than the predeclared 80% of selected users")
    # Held-out support remains in graph diagnostics. Ranking-quality evaluation
    # is outside this experiment's scope, so it does not determine readiness.
    if graph["catalogue_items_without_training_support_count"]:
        warnings.append("Catalogue retains items absent from training; see support diagnostics")
    if any(not i["metadata_present"] for i in items):
        failures.append("Some selected catalogue items lack matching book metadata")
    if any(not i["title"] for i in items):
        failures.append("Some selected catalogue items lack a source title for book-domain prompts")
    return {"policy_version": READINESS_POLICY,
            "ready": not failures, "failures": failures, "warnings": warnings,
            "scope": "Structural data readiness only; authority identity/gender coverage requires a separate audit",
            "descriptive_only": ["heldout_training_support", "users_without_shared_training_items"],
            "thresholds": {"exact_users": COHORT_SIZE, "minimum_candidate_budget": MIN_CANDIDATE_BUDGET,
                           "minimum_largest_component_share": .8}}


def assess_cohort_readiness(cohort):
    """Apply current checks to a verified frozen cohort without changing it.

    Call load_cohort first to verify file and content fingerprints. Saved
    readiness records describe the policy in effect when the cohort was frozen.
    """
    users, items = cohort["users"], cohort["items"]
    return _readiness(users, items, _graph_diagnostics(users, items),
                      cohort["sampling"]["eligible_users"])


def sample_cohort(index, output, seed=POPULATION_SEED, min_history=5):
    """Uniform users; no outcome, gender, genre or readiness-driven resampling."""
    if min_history < 5:
        raise ValueError("History minimum must be at least 5 (three training ratings plus two holdouts)")
    output = Path(output).resolve()
    sidecar = output.with_suffix(".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Frozen cohort or manifest already exists: {output}")
    connection = connect_index(index, readonly=True)
    try:
        config = settings(connection)
        if config.get("schema_version") != INDEX_SCHEMA or not config.get("complete"):
            raise ValueError("Cannot sample an incompatible or incomplete source index")
        connection.execute("CREATE TEMP TABLE eligible_users AS SELECT user_key,history_length FROM user_summary WHERE history_length>=?", (min_history,))
        connection.execute("CREATE UNIQUE INDEX eligible_key ON eligible_users(user_key)")
        eligible_count = connection.execute("SELECT count(*) FROM eligible_users").fetchone()[0]
        if eligible_count < COHORT_SIZE:
            raise ValueError(f"Exactly 300 users required; only {eligible_count} users meet the declared history minimum. No smaller cohort was frozen.")
        chosen = _choose_users(connection, seed)
        connection.execute("CREATE TEMP TABLE selected_users (user_key INTEGER PRIMARY KEY)")
        connection.executemany("INSERT INTO selected_users VALUES (?)", [(key,) for key, _ in chosen])
        item_rows = connection.execute("""SELECT DISTINCT i.item_key,i.source_item_id FROM selected_users s
            CROSS JOIN interactions r ON r.user_key=s.user_key JOIN items i ON r.item_key=i.item_key
            ORDER BY i.source_item_id""").fetchall()
        mapping = {row["item_key"]: i + 1 for i, row in enumerate(item_rows)}
        items = [_export_item(connection, row["item_key"], mapping[row["item_key"]]) for row in item_rows]
        users = []
        for number, (key, user_id) in enumerate(chosen, 1):
            history = [{"item_id": mapping[row["item_key"]], "rating": row["rating"], "timestamp": row["timestamp"],
                        "source_row": row["source_row"], "source_book_id": row["source_book_id"]}
                       for row in connection.execute("SELECT * FROM interactions WHERE user_key=? ORDER BY timestamp,source_row,item_key", (key,))]
            users.append({"user_id": number, "source_user_id": user_id,
                          "train": history[:-2], "validation": history[-2], "test": history[-1]})
        full = _population_summary(connection)
        eligible = _population_summary(connection, "eligible_users")
        selected = _population_summary(connection, "selected_users")
        graph = _graph_diagnostics(users, items)
        genre_keys = sorted(set(eligible["genre_interaction_presence_shares"]) | set(selected["genre_interaction_presence_shares"]))
        deviations = {"rating_share_selected_minus_eligible": {
            str(i): selected["rating_shares"].get(str(i), 0) - eligible["rating_shares"].get(str(i), 0) for i in range(1, 6)},
            "genre_interaction_presence_share_selected_minus_eligible": {
                key: selected["genre_interaction_presence_shares"].get(key, 0) - eligible["genre_interaction_presence_shares"].get(key, 0) for key in genre_keys},
            "user_activity_share_selected_minus_eligible": {
                key: selected["user_activity_shares"].get(key, 0) - eligible["user_activity_shares"].get(key, 0)
                for key in sorted(set(eligible["user_activity_shares"]) | set(selected["user_activity_shares"]))},
            "user_mean_rating_selected_minus_eligible": selected["user_mean_rating"] - eligible["user_mean_rating"],
            "mean_history_length_selected_minus_eligible": selected["history_length"]["mean"] - eligible["history_length"]["mean"]}
        coverage = {"catalogue_items": len(items),
                    "items_without_book_metadata": sum(not i["metadata_present"] for i in items),
                    "items_without_work_id": sum(not i["work_id"] for i in items),
                    "items_without_primary_author_id": sum(not i["primary_author_ids"] for i in items),
                    "items_without_literary_author_names": sum(not i["authors"] for i in items),
                    "unresolved_contributor_author_ids": sorted({p["author_id"] for i in items for p in i["contributors"] if not p["name"]}),
                    "items_without_genres": sum(not i["genres"] for i in items)}
        cohort = {"schema_version": COHORT_SCHEMA, "dataset": "UCSD Goodreads Book Graph",
                  "users": users, "items": items, "source_manifest": config["source_manifest"],
                  "sampling": {"seed": seed, "method": "uniform without replacement by SHA-256 priority over source user IDs",
                               "eligible_population": "All indexed users with at least min_history distinct valid dated rated works; no gender/topic/outcome restrictions",
                               "min_history": min_history, "target_users": COHORT_SIZE, "selected_users": len(users),
                               "eligible_users": eligible_count, "reserve_users": 0, "background_training_users": 0,
                               "split": "chronological leave-last-two: train, validation, test; ties by original source row",
                               "no_postselection_pruning": True, "date_policy": config["date_policy"],
                               "duplicate_policy": config["duplicate_policy"],
                               "representativeness": "Probability sample of the declared eligible population; descriptive deviations are reported, exact population balance is not guaranteed",
                               "gender_or_outcome_selection": False},
                  "diagnostics": {"full_valid_dated_rated_source": full, "eligible_population": eligible,
                                  "selected_cohort_all_observed_histories": selected, "selected_minus_eligible": deviations,
                                  "genre_semantics": "Source-supplied, multi-label descriptive genre presence; shares may sum above one; missing genres remain unknown",
                                  "metadata_coverage": coverage, "training_graph": graph,
                                  "ingest_audit": config["ingest_audit"]},
                  "readiness": _readiness(users, items, graph, eligible_count)}
        cohort["cohort_sha256"] = cohort_digest(cohort)
        data = (json.dumps(cohort, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        manifest = {"schema_version": "goodreads-cohort-manifest-v1", "cohort_file": output.name,
                    "sha256": hashlib.sha256(data).hexdigest(), "cohort_sha256": cohort["cohort_sha256"],
                    "source_manifest": cohort["source_manifest"], "sampling": cohort["sampling"]}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as target:
            target.write(data)
        with sidecar.open("x", encoding="utf-8", newline="\n") as target:
            json.dump(manifest, target, ensure_ascii=False, sort_keys=True, indent=2)
            target.write("\n")
        return cohort
    finally:
        connection.close()


def load_cohort(path):
    """Read and verify a frozen cohort's content fingerprint and reversible IDs."""
    path = Path(path)
    raw = path.read_bytes()
    cohort = json.loads(raw)
    if cohort.get("schema_version") != COHORT_SCHEMA:
        raise ValueError("Unsupported Goodreads cohort schema")
    if cohort.get("cohort_sha256") != cohort_digest(cohort):
        raise ValueError("Cohort content fingerprint does not match")
    sidecar = path.with_suffix(".manifest.json")
    if not sidecar.is_file():
        raise ValueError("Frozen cohort manifest is missing")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("Cohort file fingerprint does not match manifest")
    if manifest.get("cohort_sha256") != cohort["cohort_sha256"]:
        raise ValueError("Cohort and manifest content fingerprints disagree")
    _validate_cohort(cohort)
    return cohort


def _validate_cohort(cohort):
    users, items = cohort["users"], cohort["items"]
    if len(users) != COHORT_SIZE:
        raise ValueError("Frozen cohort must contain exactly 300 users")
    if [u["user_id"] for u in users] != list(range(1, len(users) + 1)):
        raise ValueError("User IDs must be consecutive one-based integers")
    if [i["item_id"] for i in items] != list(range(1, len(items) + 1)):
        raise ValueError("Item IDs must be consecutive one-based integers")
    if len({u["source_user_id"] for u in users}) != len(users) or len({i["source_item_id"] for i in items}) != len(items):
        raise ValueError("Source ID mappings must be reversible")
    item_ids = {i["item_id"] for i in items}
    min_history = cohort["sampling"]["min_history"]
    for user in users:
        events = [*user["train"], user["validation"], user["test"]]
        if len(events) < min_history or len({e["item_id"] for e in events}) != len(events):
            raise ValueError("Invalid or duplicate user history")
        if events != sorted(events, key=lambda e: (e["timestamp"], e["source_row"])):
            raise ValueError("Chronological split is invalid")
        if any(e["item_id"] not in item_ids or isinstance(e["rating"], bool) or e["rating"] not in range(1, 6) for e in events):
            raise ValueError("Invalid rating or item mapping")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New immutable cohort JSON plus .manifest.json")
    parser.add_argument("--seed", type=int, default=POPULATION_SEED)
    parser.add_argument("--min-history", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        cohort = sample_cohort(args.index, args.output, args.seed, args.min_history)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Sampling failed: {error}\n")
    print(json.dumps({"cohort": str(args.output), "cohort_sha256": cohort["cohort_sha256"],
                      "sampling": cohort["sampling"], "readiness": cohort["readiness"]}, indent=2, sort_keys=True))
    return 0 if cohort["readiness"]["ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
