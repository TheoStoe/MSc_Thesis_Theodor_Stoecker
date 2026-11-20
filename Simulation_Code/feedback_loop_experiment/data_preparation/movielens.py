"""Deterministic MovieLens-1M preparation without legacy runtime imports.

The protocol retains every rating as a positive interaction, applies a
simultaneous iterative user/item k-core, maps retained source IDs to compact
one-based runtime IDs, and performs a chronological leave-last-two split.
Only the training part is exposed as simulator/recommender history.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

from ..utility.fingerprints import fingerprint_semantic_state
from ..utility.user_handling import (
    PreparedDatasetArtifact,
    PreparedTrainingInteraction,
)


DATASET_SCHEMA = "feedback-loop-ml1m-prepared-v1"
ITEM_TEXT_BUILDER_VERSION = "ml1m-title-genres-v1"


@dataclass(frozen=True)
class MovieLensPreparationConfig:
    """Frozen choices copied from the validated first-experiment protocol."""

    protocol_version: str = "ml1m_sasrec_v1"
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    implicit_feedback: str = "all_ratings"
    split_protocol: str = "leave_last_two"

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_version, str) or not self.protocol_version:
            raise ValueError("protocol_version must be a non-blank string")
        for name in ("min_user_interactions", "min_item_interactions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.implicit_feedback != "all_ratings":
            raise ValueError("only all_ratings implicit feedback is supported")
        if self.split_protocol != "leave_last_two":
            raise ValueError("only leave_last_two splitting is supported")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, order=True)
class RatingEvent:
    source_user_id: int
    source_item_id: int
    rating: int
    timestamp: int
    source_row: int


@dataclass(frozen=True)
class SourceMovie:
    source_item_id: int
    title: str
    genres: tuple[str, ...]


@dataclass(frozen=True)
class CoreFilterRound:
    pass_index: int
    input_events: int
    output_events: int
    removed_events: int
    output_users: int
    output_items: int


def prepare_movielens_1m(
    dataset_directory: str | Path,
    *,
    config: MovieLensPreparationConfig | None = None,
) -> PreparedDatasetArtifact:
    """Prepare an extracted MovieLens-1M directory for all later conditions."""

    root = Path(dataset_directory)
    required = {
        name: root / name for name in ("ratings.dat", "movies.dat", "users.dat")
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"MovieLens 1M directory is missing required files: {missing}"
        )
    effective_config = config or MovieLensPreparationConfig()
    ratings = parse_ratings(required["ratings.dat"])
    movies = parse_movies(required["movies.dat"])
    source_user_ids = _parse_user_ids(required["users.dat"])
    artifact, _ = prepare_events(
        ratings,
        movies=movies,
        config=effective_config,
        source_file_fingerprints={
            name: _file_sha256(path) for name, path in sorted(required.items())
        },
    )
    missing_users = sorted(
        set(artifact.runtime_to_source_user_id.values()).difference(source_user_ids),
        key=int,
    )
    if missing_users:
        raise ValueError(
            "users.dat is missing users retained from ratings.dat: "
            f"{missing_users}"
        )
    return artifact


def parse_ratings(path: str | Path) -> tuple[RatingEvent, ...]:
    ratings_path = Path(path)
    if not ratings_path.is_file():
        raise FileNotFoundError(f"MovieLens ratings file not found: {ratings_path}")
    events: list[RatingEvent] = []
    with ratings_path.open("r", encoding="latin-1", newline="") as stream:
        for source_row, raw_line in enumerate(stream):
            prefix = f"{ratings_path}: row {source_row + 1}: "
            line = raw_line.rstrip("\r\n")
            if not line:
                raise ValueError(prefix + "row cannot be empty")
            fields = line.split("::")
            if len(fields) != 4:
                raise ValueError(prefix + "expected 4 fields separated by ::")
            try:
                user_id, item_id, rating, timestamp = map(int, fields)
            except ValueError as exc:
                raise ValueError(prefix + "fields must be integers") from exc
            if user_id <= 0 or item_id <= 0 or timestamp <= 0:
                raise ValueError(prefix + "IDs and timestamp must be positive")
            if not 1 <= rating <= 5:
                raise ValueError(prefix + "rating must lie between 1 and 5")
            events.append(
                RatingEvent(user_id, item_id, rating, timestamp, source_row)
            )
    if not events:
        raise ValueError(f"{ratings_path}: file contains no rating events")
    return tuple(events)


def parse_movies(path: str | Path) -> Mapping[int, SourceMovie]:
    movies_path = Path(path)
    if not movies_path.is_file():
        raise FileNotFoundError(f"MovieLens movies file not found: {movies_path}")
    movies: dict[int, SourceMovie] = {}
    with movies_path.open("r", encoding="latin-1", newline="") as stream:
        for row_number, raw_line in enumerate(stream, start=1):
            prefix = f"{movies_path}: row {row_number}: "
            line = raw_line.rstrip("\r\n")
            if not line:
                raise ValueError(prefix + "row cannot be empty")
            fields = line.split("::")
            if len(fields) != 3:
                raise ValueError(prefix + "expected 3 fields separated by ::")
            try:
                source_item_id = int(fields[0])
            except ValueError as exc:
                raise ValueError(prefix + "movie ID must be an integer") from exc
            title = fields[1].strip()
            genres = tuple(value.strip() for value in fields[2].split("|") if value)
            if source_item_id <= 0 or not title or not genres:
                raise ValueError(prefix + "movie ID, title, and genres are required")
            if source_item_id in movies:
                raise ValueError(prefix + f"duplicate movie ID {source_item_id}")
            movies[source_item_id] = SourceMovie(source_item_id, title, genres)
    if not movies:
        raise ValueError(f"{movies_path}: file contains no movies")
    return movies


def prepare_events(
    events: Sequence[RatingEvent] | Iterable[RatingEvent],
    *,
    movies: Mapping[int, SourceMovie],
    config: MovieLensPreparationConfig | None = None,
    source_file_fingerprints: Mapping[str, str] | None = None,
) -> tuple[PreparedDatasetArtifact, tuple[CoreFilterRound, ...]]:
    """Prepare in-memory rows; exposed primarily for small deterministic tests."""

    effective_config = config or MovieLensPreparationConfig()
    source_events = tuple(events)
    if not source_events:
        raise ValueError("at least one rating event is required")
    if not movies:
        raise ValueError("movies cannot be empty")
    source_rows = [event.source_row for event in source_events]
    if len(source_rows) != len(set(source_rows)) or min(source_rows) < 0:
        raise ValueError("source_row values must be unique and non-negative")
    pairs = [(event.source_user_id, event.source_item_id) for event in source_events]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate (user, item) rating pair")

    filtered, rounds = _filter_k_core(source_events, effective_config)
    if not filtered:
        raise ValueError("core filtering removed every rating event")
    source_user_ids = sorted({event.source_user_id for event in filtered})
    source_item_ids = sorted({event.source_item_id for event in filtered})
    missing_movies = sorted(set(source_item_ids).difference(movies))
    if missing_movies:
        raise ValueError(f"movie metadata is missing retained items: {missing_movies}")

    user_to_runtime = {
        source_id: runtime_id
        for runtime_id, source_id in enumerate(source_user_ids, start=1)
    }
    item_to_runtime = {
        source_id: runtime_id
        for runtime_id, source_id in enumerate(source_item_ids, start=1)
    }
    by_user: dict[int, list[RatingEvent]] = defaultdict(list)
    for event in filtered:
        by_user[user_to_runtime[event.source_user_id]].append(event)
    for user_events in by_user.values():
        user_events.sort(key=lambda event: (event.timestamp, event.source_row))
    short_users = {
        user_id: len(user_events)
        for user_id, user_events in by_user.items()
        if len(user_events) < 3
    }
    if short_users:
        raise ValueError(
            "leave_last_two requires at least three retained events per user: "
            f"{short_users}"
        )

    histories: dict[int, tuple[int, ...]] = {}
    training_interactions: dict[int, tuple[PreparedTrainingInteraction, ...]] = {}
    validation: dict[int, int] = {}
    test: dict[int, int] = {}
    for user_id in sorted(by_user):
        sequence = by_user[user_id]
        mapped_items = tuple(item_to_runtime[event.source_item_id] for event in sequence)
        histories[user_id] = mapped_items[:-2]
        training_interactions[user_id] = tuple(
            PreparedTrainingInteraction(
                item_id=item_to_runtime[event.source_item_id],
                source_item_id=str(event.source_item_id),
                rating=event.rating,
                timestamp=event.timestamp,
                source_row=event.source_row,
            )
            for event in sequence[:-2]
        )
        validation[user_id] = mapped_items[-2]
        test[user_id] = mapped_items[-1]

    popularity = Counter(item_id for history in histories.values() for item_id in history)
    catalog: dict[int, Mapping[str, Any]] = {}
    for source_item_id in source_item_ids:
        item_id = item_to_runtime[source_item_id]
        movie = movies[source_item_id]
        item_text = _item_text(movie)
        catalog[item_id] = {
            "item_id": item_id,
            "source_item_id": str(source_item_id),
            "title": movie.title,
            "genres": movie.genres,
            "item_text": item_text,
            "item_text_builder_version": ITEM_TEXT_BUILDER_VERSION,
        }

    runtime_to_source_user = {
        runtime_id: str(source_id) for source_id, runtime_id in user_to_runtime.items()
    }
    runtime_to_source_item = {
        runtime_id: str(source_id) for source_id, runtime_id in item_to_runtime.items()
    }
    training_history_fingerprints = {
        user_id: fingerprint_semantic_state(
            {
                "user_id": user_id,
                "source_user_id": runtime_to_source_user[user_id],
                "interactions": training_interactions[user_id],
            },
            "ml1m-user-training-history-v1",
        )
        for user_id in sorted(training_interactions)
    }
    id_mapping_payload = {
        "runtime_to_source_user_id": runtime_to_source_user,
        "runtime_to_source_item_id": runtime_to_source_item,
    }
    catalog_fingerprint = fingerprint_semantic_state(catalog, "ml1m-catalog-v1")
    item_text_fingerprint = fingerprint_semantic_state(
        {
            "builder_version": ITEM_TEXT_BUILDER_VERSION,
            "item_text_by_id": {
                item_id: catalog[item_id]["item_text"] for item_id in sorted(catalog)
            },
        },
        ITEM_TEXT_BUILDER_VERSION,
    )
    preparation_statistics = {
        "source_events": len(source_events),
        "retained_events": len(filtered),
        "retained_users": len(histories),
        "retained_items": len(catalog),
        "training_events": sum(len(history) for history in histories.values()),
        "filter_rounds": tuple(asdict(round_) for round_ in rounds),
    }
    semantic_payload = {
        "preparation_protocol": effective_config.to_dict(),
        "preparation_statistics": preparation_statistics,
        "source_file_fingerprints": dict(source_file_fingerprints or {}),
        "id_mapping": id_mapping_payload,
        "catalog": catalog,
        "training_histories": histories,
        "training_interactions": training_interactions,
        "training_history_fingerprints": training_history_fingerprints,
        "validation_targets": validation,
        "test_targets": test,
        "initial_item_popularity": {
            item_id: popularity[item_id] for item_id in sorted(catalog)
        },
    }
    artifact = PreparedDatasetArtifact(
        schema_version=DATASET_SCHEMA,
        preparation_protocol=effective_config.protocol_version,
        preparation_settings=effective_config.to_dict(),
        preparation_statistics=preparation_statistics,
        dataset_fingerprint=fingerprint_semantic_state(
            semantic_payload, DATASET_SCHEMA
        ),
        id_mapping_fingerprint=fingerprint_semantic_state(
            id_mapping_payload, "ml1m-id-map-v1"
        ),
        catalog_fingerprint=catalog_fingerprint,
        item_text_fingerprint=item_text_fingerprint,
        source_file_fingerprints=dict(source_file_fingerprints or {}),
        catalog=catalog,
        training_histories=histories,
        training_interactions=training_interactions,
        training_history_fingerprints=training_history_fingerprints,
        validation_targets=validation,
        test_targets=test,
        runtime_to_source_user_id=runtime_to_source_user,
        runtime_to_source_item_id=runtime_to_source_item,
        initial_item_popularity={
            item_id: popularity[item_id] for item_id in sorted(catalog)
        },
    )
    return artifact, rounds


def dataset_semantic_payload(artifact: PreparedDatasetArtifact) -> Mapping[str, Any]:
    """Return exactly the data covered by ``dataset_fingerprint``."""

    return {
        "preparation_protocol": artifact.preparation_settings,
        "preparation_statistics": artifact.preparation_statistics,
        "source_file_fingerprints": artifact.source_file_fingerprints,
        "id_mapping": {
            "runtime_to_source_user_id": artifact.runtime_to_source_user_id,
            "runtime_to_source_item_id": artifact.runtime_to_source_item_id,
        },
        "catalog": artifact.catalog,
        "training_histories": artifact.training_histories,
        "training_interactions": artifact.training_interactions,
        "training_history_fingerprints": artifact.training_history_fingerprints,
        "validation_targets": artifact.validation_targets,
        "test_targets": artifact.test_targets,
        "initial_item_popularity": artifact.initial_item_popularity,
    }


def _filter_k_core(
    events: Sequence[RatingEvent],
    config: MovieLensPreparationConfig,
) -> tuple[tuple[RatingEvent, ...], tuple[CoreFilterRound, ...]]:
    current = tuple(sorted(events, key=lambda event: event.source_row))
    rounds: list[CoreFilterRound] = []
    while True:
        user_counts = Counter(event.source_user_id for event in current)
        item_counts = Counter(event.source_item_id for event in current)
        valid_users = {
            user_id
            for user_id, count in user_counts.items()
            if count >= config.min_user_interactions
        }
        valid_items = {
            item_id
            for item_id, count in item_counts.items()
            if count >= config.min_item_interactions
        }
        filtered = tuple(
            event
            for event in current
            if event.source_user_id in valid_users
            and event.source_item_id in valid_items
        )
        rounds.append(
            CoreFilterRound(
                pass_index=len(rounds) + 1,
                input_events=len(current),
                output_events=len(filtered),
                removed_events=len(current) - len(filtered),
                output_users=len({event.source_user_id for event in filtered}),
                output_items=len({event.source_item_id for event in filtered}),
            )
        )
        if len(filtered) == len(current):
            return filtered, tuple(rounds)
        current = filtered


def _item_text(movie: SourceMovie) -> str:
    return f"Title: {movie.title}\nGenres: {', '.join(movie.genres)}"


def _parse_user_ids(path: Path) -> frozenset[str]:
    user_ids: set[str] = set()
    with path.open("r", encoding="latin-1", newline="") as stream:
        for row_number, raw_line in enumerate(stream, start=1):
            prefix = f"{path}: row {row_number}: "
            fields = raw_line.rstrip("\r\n").split("::")
            if len(fields) != 5:
                raise ValueError(prefix + "expected 5 fields separated by ::")
            source_user_id = fields[0]
            if (
                not source_user_id.isdecimal()
                or str(int(source_user_id)) != source_user_id
                or int(source_user_id) <= 0
            ):
                raise ValueError(prefix + "user ID must be canonical and positive")
            if source_user_id in user_ids:
                raise ValueError(prefix + f"duplicate user ID {source_user_id}")
            user_ids.add(source_user_id)
    if not user_ids:
        raise ValueError(f"{path}: file contains no users")
    return frozenset(user_ids)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CoreFilterRound",
    "DATASET_SCHEMA",
    "ITEM_TEXT_BUILDER_VERSION",
    "MovieLensPreparationConfig",
    "RatingEvent",
    "SourceMovie",
    "dataset_semantic_payload",
    "parse_movies",
    "parse_ratings",
    "prepare_events",
    "prepare_movielens_1m",
]
