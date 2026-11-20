"""Goodreads cohort -> isolated simulator inputs, without provider calls.

Book/work identifiers remain reversible in catalog metadata and source_ids.json.
The copied engine's numeric item-ID contract is deliberately local; it is not an
assertion that Goodreads book IDs or work IDs are numeric simulator IDs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .engine.config import SimulatorConfig
from .engine.data_preparation.artifacts import load_prepared_dataset, save_prepared_dataset
from .engine.data_preparation.books import DATASET_SCHEMA, dataset_semantic_payload
from .engine.utility.fingerprints import fingerprint_semantic_state as fingerprint
from .engine.utility.saving import save_json_artifact, save_population_artifact
from .engine.utility.user_handling import (
    PopulationArtifact, PopulationMember, PopulationRole, PreparedDatasetArtifact,
    PreparedInputs, PreparedTrainingInteraction, load_prepared_inputs,
)

ITEM_TEXT_VERSION = "goodreads-title-authors-genres-v1"
NCM_FOLDERS = {"pointwise_logistic": "P1", "ncm_inspired_gru_uniform": "P3", "ncm_inspired_gru_mansoury": "P4"}
SASREC_SETTINGS = {"num_epochs": 804, "batch_size": 128, "max_sequence_length": 200}
PROVIDER_SETTINGS = {"model": "gpt-5.4-nano", "reasoning_effort": "none", "max_output_tokens": 2056}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def book_item_text(item: Mapping[str, Any]) -> str:
    """Explicit allowlist: no annotations, reader demographics or contributor gender."""
    title = str(item.get("title", "")).strip()
    if not title:
        raise ValueError("every book requires a title")
    authors = ", ".join(str(value).strip() for value in item.get("authors", ()) if str(value).strip())
    genres = ", ".join(str(value).strip() for value in item.get("genres", ()) if str(value).strip())
    parts = [f"Book: {title}"]
    if authors:
        parts.append(f"Author(s): {authors}")
    if genres:
        parts.append(f"Genres: {genres}")
    # Descriptions are retained for provenance but not put into prompts: public
    # descriptions can contain demographic annotations outside our input policy.
    return "; ".join(parts)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def build_book_dataset(cohort: Mapping[str, Any], *, source_sha256: str) -> PreparedDatasetArtifact:
    if cohort.get("schema_version") not in {"goodreads-cohort-v1", "book-cohort-v1"}:
        raise ValueError("expected a Goodreads/book cohort v1")
    raw_users, raw_items = cohort.get("users", ()), cohort.get("items", ())
    catalog = {}
    for row in raw_items:
        item_id = _integer(row["item_id"], "item_id", minimum=1)
        if item_id in catalog:
            raise ValueError("duplicate catalog item ID")
        original = str(row["source_item_id"])
        if not original.startswith(("work:", "book:")):
            raise ValueError("source book identifiers must be explicit work: or book: IDs")
        catalog[item_id] = {
            "item_id": item_id, "source_item_id": str(item_id),
            "original_source_item_id": original, "work_id": row.get("work_id"),
            "book_ids": [str(value) for value in row.get("book_ids", ())],
            "title": str(row["title"]), "authors": [str(value) for value in row.get("authors", ())],
            "author_ids": [str(value) for value in row.get("author_ids", ())],
            "contributors": [{key: entry[key] for key in ("author_id", "name", "role", "primary_literary_author") if key in entry} for entry in row.get("contributors", ())],
            "genres": tuple(str(value) for value in row.get("genres", ())),
            "description": str(row.get("description") or ""), "item_text": book_item_text(row),
        }
    if sorted(catalog) != list(range(1, len(catalog) + 1)) or not catalog:
        raise ValueError("book item IDs must be contiguous and one-based")
    if len({row["original_source_item_id"] for row in catalog.values()}) != len(catalog):
        raise ValueError("original book/work identifiers must be unique")
    histories, events, histories_fp, validation, test, user_map = {}, {}, {}, {}, {}, {}
    for user in raw_users:
        uid = _integer(user["user_id"], "user_id", minimum=1)
        if uid in user_map:
            raise ValueError("duplicate user ID")
        user_map[uid] = str(user["source_user_id"])
        if not user_map[uid].strip():
            raise ValueError("source user IDs cannot be blank")
        ordered = [*user["train"], user["validation"], user["test"]]
        if len(ordered) < 3:
            raise ValueError("each user requires training, validation and test events")
        keys = []
        converted = []
        for event in ordered:
            iid = _integer(event["item_id"], "event item_id", minimum=1)
            if iid not in catalog:
                raise ValueError("event references an unknown book")
            rating = _integer(event["rating"], "rating", minimum=1)
            timestamp = _integer(event["timestamp"], "timestamp", minimum=1)
            source_row = _integer(event["source_row"], "source_row")
            converted.append(PreparedTrainingInteraction(iid, str(iid), rating, timestamp, source_row))
            keys.append((timestamp, source_row))
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("cohort events must have a strict chronological (timestamp, source_row) order")
        events[uid] = tuple(converted[:-2])
        histories[uid] = tuple(event.item_id for event in events[uid])
        validation[uid], test[uid] = converted[-2].item_id, converted[-1].item_id
        histories_fp[uid] = fingerprint({"user_id": uid, "source_user_id": user_map[uid], "interactions": events[uid]}, "goodreads-user-training-history-v1")
    if sorted(user_map) != list(range(1, len(user_map) + 1)) or not user_map:
        raise ValueError("user IDs must be contiguous and one-based")
    if len(set(user_map.values())) != len(user_map):
        raise ValueError("source reader IDs must be unique")
    item_map = {iid: str(iid) for iid in catalog}
    counts = Counter(iid for history in histories.values() for iid in history)
    total = sum(map(len, histories.values())) + 2 * len(user_map)
    sampling = dict(cohort.get("sampling", {}))
    # JSON-shaped sampling settings stay lists on both save and load.
    settings = {"dataset": "ucsd-goodreads", "source_schema": cohort["schema_version"], "sampling": sampling,
                "split": "chronological-last-two", "item_text_builder_version": ITEM_TEXT_VERSION,
                "profile_history_limit": 40, "training_user_scope": "selected-active-cohort-only"}
    statistics = {"retained_users": len(user_map), "retained_items": len(catalog), "training_events": total - 2 * len(user_map),
                  "retained_events": total, "source_events": total,
                  "filter_rounds": ({"stage": "already-sampled-cohort-adaptation", "output_events": total},)}
    artifact = PreparedDatasetArtifact(
        schema_version=DATASET_SCHEMA, preparation_protocol="goodreads-frozen-cohort-adapter-v1",
        preparation_settings=settings, preparation_statistics=statistics, dataset_fingerprint="pending",
        id_mapping_fingerprint=fingerprint({"runtime_to_source_user_id": user_map, "runtime_to_source_item_id": item_map}, "goodreads-id-map-v1"),
        catalog_fingerprint=fingerprint(catalog, "goodreads-catalog-v1"),
        item_text_fingerprint=fingerprint({"builder_version": ITEM_TEXT_VERSION, "item_text_by_id": {iid: catalog[iid]["item_text"] for iid in sorted(catalog)}}, ITEM_TEXT_VERSION),
        source_file_fingerprints={"cohort_json_sha256": source_sha256,
                                  "source_manifest": fingerprint(cohort.get("source_manifest", {}), "goodreads-source-manifest-v1")},
        catalog=catalog, training_histories=histories, training_interactions=events, training_history_fingerprints=histories_fp,
        validation_targets=validation, test_targets=test, runtime_to_source_user_id=user_map, runtime_to_source_item_id=item_map,
        initial_item_popularity={iid: counts[iid] for iid in catalog},
    )
    return replace(artifact, dataset_fingerprint=fingerprint(dataset_semantic_payload(artifact), DATASET_SCHEMA))


def build_book_population(dataset: PreparedDatasetArtifact, *, population_seed: int = 1054) -> PopulationArtifact:
    _integer(population_seed, "population_seed")
    members = tuple(PopulationMember(uid, dataset.runtime_to_source_user_id[uid], PopulationRole.ACTIVE, index, None, 1, None)
                    for index, uid in enumerate(sorted(dataset.training_histories)))
    payload = {"schema_version": "goodreads-fixed-active-population-v1", "population_seed": population_seed,
               "source_population_fingerprint": dataset.dataset_fingerprint,
               "cluster_artifact_fingerprint": fingerprint({"rule": "one-placeholder-cluster-no-clustering-fit", "cluster_id": 1}, "goodreads-no-cluster-v1"), "members": members}
    return PopulationArtifact(**payload, population_fingerprint=fingerprint(payload, payload["schema_version"]))


def prepare_book_inputs(cohort_path: str | Path, inputs_root: str | Path, *, population_seed: int = 1054, expected_users: int | None = 300) -> PreparedInputs:
    cohort_path, root = Path(cohort_path), Path(inputs_root)
    if expected_users == 300:
        from .sampling import assess_cohort_readiness, load_cohort
        cohort = load_cohort(cohort_path)
        readiness = assess_cohort_readiness(cohort)
        if not readiness["ready"]:
            raise ValueError("frozen cohort failed structural readiness; retain it and inspect diagnostics before preparing models: "
                             + "; ".join(readiness["failures"]))
        if cohort.get("sampling", {}).get("population_seed", cohort.get("sampling", {}).get("seed")) != population_seed:
            raise ValueError("cohort selection seed differs from the frozen population seed")
    else:
        # Explicit small-fixture opt-in for offline integration tests. Primary
        # preparation always verifies the sampler's immutable file/sidecar.
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    dataset = build_book_dataset(cohort, source_sha256=file_sha256(cohort_path))
    if expected_users is not None and len(dataset.training_histories) != expected_users:
        raise ValueError(f"expected {expected_users} active readers, found {len(dataset.training_histories)}")
    population = build_book_population(dataset, population_seed=population_seed)
    # Validate all content before any mutation, and reject existing destinations.
    for target in (root / "prepared", root / "population.json", root / "source_ids.json"):
        if target.exists():
            raise FileExistsError(target)
    save_prepared_dataset(root / "prepared", dataset)
    save_population_artifact(root / "population.json", population)
    save_json_artifact(root / "source_ids.json", {"schema_version": "goodreads-reversible-source-map-v1", "dataset_fingerprint": dataset.dataset_fingerprint,
        "users": {str(uid): source for uid, source in dataset.runtime_to_source_user_id.items()},
        "items": {str(iid): {"source_item_id": row["original_source_item_id"], "work_id": row["work_id"], "book_ids": row["book_ids"]} for iid, row in dataset.catalog.items()}})
    return load_book_inputs(root / "prepared", root / "population.json")


def load_book_inputs(prepared_data_path: str | Path, population_artifact_path: str | Path, profile_artifact_path: str | Path | None = None, initial_recommender_path: str | Path | None = None) -> PreparedInputs:
    if initial_recommender_path is not None:
        checkpoint_root = Path(initial_recommender_path)
        complete = json.loads((checkpoint_root / "_COMPLETE.json").read_text(encoding="utf-8"))
        if complete.get("schema_version") != "goodreads-initial-sasrec-files-v1" or complete.get("status") != "complete":
            raise ValueError("initial recommender is not a complete fresh book-domain checkpoint")
        for filename, expected in complete["files_sha256"].items():
            if Path(filename).name != filename or file_sha256(checkpoint_root / filename) != expected:
                raise ValueError("initial book recommender file hash mismatch")
    inputs = load_prepared_inputs(Path(prepared_data_path), Path(population_artifact_path), None if profile_artifact_path is None else Path(profile_artifact_path), None if initial_recommender_path is None else Path(initial_recommender_path))
    if inputs.dataset.schema_version != DATASET_SCHEMA:
        raise ValueError("book runs require fresh Goodreads prepared inputs")
    if any(member.role is not PopulationRole.ACTIVE for member in inputs.population.members):
        raise ValueError("this experiment has no reserve or background readers")
    if len(inputs.population.members) != len(inputs.dataset.training_histories):
        raise ValueError("all and only cohort readers must be active")
    if initial_recommender_path is not None:
        baseline = json.loads((checkpoint_root / "baseline_slates.json").read_text(encoding="utf-8"))
        if complete.get("dataset_fingerprint") != inputs.dataset.dataset_fingerprint or baseline.get("dataset_fingerprint") != inputs.dataset.dataset_fingerprint:
            raise ValueError("initial book checkpoint/baseline and prepared dataset differ")
        if baseline.get("population_fingerprint") != inputs.population.population_fingerprint or baseline.get("schema_version") != "books-initial-baseline-v1":
            raise ValueError("initial book baseline and frozen population differ")
        slates = {int(row["user_id"]): row["item_ids"] for row in baseline["slates"]}
        if len(slates) != len(baseline["slates"]) or set(slates) != set(inputs.dataset.training_histories):
            raise ValueError("initial book baseline does not cover every cohort reader exactly once")
        for uid, slate in slates.items():
            if len(slate) != baseline["slate_size"] or len(slate) != len(set(slate)) or not set(slate).issubset(inputs.dataset.catalog):
                raise ValueError("initial book baseline contains invalid candidate slates")
            if set(slate).intersection(inputs.dataset.training_histories[uid]):
                raise ValueError("initial book baseline contains books already in the training history")
    return inputs


def build_book_config(inputs_root: str | Path, output_root: str | Path, *, experiment_id: str = "goodreads_books", population_seed: int = 1054, active_user_count: int = 300, seed: int = 42, user_model: str = "position_biased", **overrides: Any) -> SimulatorConfig:
    root = Path(inputs_root)
    user_settings: dict[str, Any] = {}
    if user_model in NCM_FOLDERS:
        user_settings.update(checkpoint_directory=str(root / "ncm" / f"seed_{seed}" / NCM_FOLDERS[user_model] / "checkpoint"), embedding_directory=str(root / "embeddings"), device="cpu")
    defaults = dict(experiment_id=experiment_id, population_seed=population_seed, num_iterations=30, slate_size=10,
                    active_user_count=active_user_count, reserve_user_count=0, prepared_data_path=root / "prepared", population_artifact_path=root / "population.json",
                    profile_artifact_path=root / "profiles" if user_model == "agentic_recommender" else None,
                    initial_recommender_path=root / "initial_sasrec" / f"seed_{seed}", output_root=Path(output_root),
                    recommender_settings=dict(SASREC_SETTINGS), user_model_settings=user_settings, provider_settings=dict(PROVIDER_SETTINGS),
                    training_scope="active_reserve", input_format="books")
    defaults.update(overrides)
    return SimulatorConfig(**defaults)


__all__ = ["SASREC_SETTINGS", "PROVIDER_SETTINGS", "NCM_FOLDERS", "build_book_dataset", "build_book_population", "prepare_book_inputs", "load_book_inputs", "build_book_config", "book_item_text"]
