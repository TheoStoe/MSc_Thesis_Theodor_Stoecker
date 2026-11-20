"""Local Goodreads readers and disk-backed source index; no identity inference."""
from __future__ import annotations

from collections import Counter, OrderedDict
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re
import sqlite3

from .streaming import SourceError, json_rows, open_source, source_format

INDEX_SCHEMA = "goodreads-source-index-v1"
LITERARY_ROLES = {"", "author", "coauthor", "co-author", "writer"}
MONTHS = {name: number for number, name in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}


def source_id(value):
    if isinstance(value, bool) or value is None or isinstance(value, (list, dict)):
        raise ValueError("Missing or invalid source identifier")
    value = str(value)
    if not value.strip():
        raise ValueError("Empty source identifier")
    return value


def date_timestamp(value):
    """Parse Goodreads English dates independently of the machine's locale."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing date_updated availability proxy")
    value = value.strip()
    match = re.fullmatch(
        r"[A-Za-z]{3} ([A-Za-z]{3}) +(\d{1,2}) (\d{2}):(\d{2}):(\d{2}) ([+-])(\d{2})(\d{2}) (\d{4})", value)
    if match:
        month, day, hour, minute, second, sign, off_hour, off_minute, year = match.groups()
        offset_hours, offset_minutes = int(off_hour), int(off_minute)
        if offset_hours > 23 or offset_minutes > 59:
            raise ValueError("Invalid source timezone")
        offset = (offset_hours * 60 + offset_minutes) * (1 if sign == "+" else -1)
        parsed = datetime(int(year), MONTHS[month], int(day), int(hour), int(minute), int(second),
                          tzinfo=timezone(timedelta(minutes=offset)))
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("A source timezone is required")
    return int(parsed.timestamp())


def connect_index(path, readonly=False):
    path = Path(path).resolve()
    if readonly:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def initialize(connection):
    connection.executescript("""
        CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE authors (author_id TEXT PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE items (item_key INTEGER PRIMARY KEY, source_item_id TEXT UNIQUE NOT NULL,
            work_id TEXT, title TEXT NOT NULL DEFAULT '', description TEXT NOT NULL DEFAULT '',
            metadata_present INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE book_map (book_id TEXT PRIMARY KEY, item_key INTEGER NOT NULL REFERENCES items);
        CREATE INDEX book_map_item ON book_map(item_key);
        CREATE TABLE contributors (item_key INTEGER NOT NULL REFERENCES items,
            book_id TEXT NOT NULL, ordinal INTEGER NOT NULL, author_id TEXT NOT NULL,
            role TEXT NOT NULL, primary_literary_author INTEGER NOT NULL,
            PRIMARY KEY(book_id, ordinal));
        CREATE INDEX contributors_item ON contributors(item_key);
        CREATE TABLE genres (item_key INTEGER NOT NULL REFERENCES items, genre TEXT NOT NULL,
            source_count INTEGER NOT NULL, PRIMARY KEY(item_key, genre)) WITHOUT ROWID;
        CREATE TABLE users (user_key INTEGER PRIMARY KEY, source_user_id TEXT UNIQUE NOT NULL);
        CREATE TABLE interactions (user_key INTEGER NOT NULL REFERENCES users,
            item_key INTEGER NOT NULL REFERENCES items, rating INTEGER NOT NULL,
            timestamp INTEGER NOT NULL, source_row INTEGER NOT NULL, source_book_id TEXT NOT NULL,
            PRIMARY KEY(user_key, item_key)) WITHOUT ROWID;
    """)
    set_setting(connection, "schema_version", INDEX_SCHEMA)
    set_setting(connection, "complete", False)


def set_setting(connection, key, value):
    connection.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, json.dumps(value, sort_keys=True)))


def settings(connection):
    return {row["key"]: json.loads(row["value"]) for row in connection.execute("SELECT * FROM settings")}


class BoundedCache:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.values = OrderedDict()

    def get(self, key):
        value = self.values.get(key)
        if value is not None:
            self.values.move_to_end(key)
        return value

    def put(self, key, value):
        self.values[key] = value
        self.values.move_to_end(key)
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


def _rows(path, manifest, counts, strict):
    if source_format(path) != "jsonl":
        raise SourceError("Goodreads requires dated JSON-lines, optionally gzip; undated ratings CSV is unsupported")
    with open_source(path, manifest) as stream:
        yield from json_rows(stream, counts, strict)


def _invalid(counts, strict, kind, row, error):
    counts[kind] += 1
    if strict:
        raise SourceError(f"Invalid source row {row}: {error}") from error


def ingest_authors(connection, path, manifest, strict=False, batch_size=50000):
    counts = Counter()
    for number, row in _rows(path, manifest, counts, strict):
        try:
            author_id = source_id(row.get("author_id"))
            name = row.get("name", "")
            if not isinstance(name, str):
                raise ValueError("Author name must be a string")
            cursor = connection.execute("INSERT OR IGNORE INTO authors VALUES (?,?)", (author_id, name.strip()))
            counts["accepted_rows" if cursor.rowcount else "duplicate_author_ids_first_retained"] += 1
        except (ValueError, TypeError, KeyError) as error:
            _invalid(counts, strict, "invalid_rows", number, error)
        if number % batch_size == 0:
            connection.commit()
    connection.commit()
    return dict(counts)


def _text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Metadata text must be a string")
    return value.strip()


def _contributors(value):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Book authors must be a list")
    result, primary_chosen = [], False
    for number, person in enumerate(value):
        if not isinstance(person, dict):
            raise ValueError("Book author requires author_id and role")
        author_id = source_id(person.get("author_id"))
        role = _text(person.get("role", ""))
        literary = role.casefold() in LITERARY_ROLES
        primary = literary and not primary_chosen
        primary_chosen = primary_chosen or primary
        result.append((number, author_id, role, int(primary)))
    return result


def ingest_books(connection, path, manifest, strict=False, batch_size=50000):
    counts = Counter()
    for number, row in _rows(path, manifest, counts, strict):
        try:
            book_id = source_id(row.get("book_id"))
            raw_work = row.get("work_id")
            work_id = source_id(raw_work) if raw_work not in (None, "", "0", 0) else None
            source_item_id = f"work:{work_id}" if work_id else f"book:{book_id}"
            title = _text(row.get("title"))
            description = _text(row.get("description"))
            people = _contributors(row.get("authors"))
            if connection.execute("SELECT 1 FROM book_map WHERE book_id=?", (book_id,)).fetchone():
                counts["duplicate_book_ids_first_retained"] += 1
                continue
            connection.execute("INSERT OR IGNORE INTO items(source_item_id,work_id,title,description,metadata_present) VALUES (?,?,?,?,1)",
                               (source_item_id, work_id, title, description))
            item_key = connection.execute("SELECT item_key FROM items WHERE source_item_id=?", (source_item_id,)).fetchone()[0]
            # A work keeps the first nonempty source title/description, never review text.
            connection.execute("UPDATE items SET title=CASE WHEN title='' THEN ? ELSE title END, description=CASE WHEN description='' THEN ? ELSE description END WHERE item_key=?",
                               (title, description, item_key))
            connection.execute("INSERT INTO book_map VALUES (?,?)", (book_id, item_key))
            connection.executemany("INSERT INTO contributors VALUES (?,?,?,?,?,?)",
                                   [(item_key, book_id, *person) for person in people])
            counts["accepted_rows"] += 1
            counts["rows_without_work_id"] += int(work_id is None)
            counts["rows_without_primary_literary_author"] += int(not any(p[-1] for p in people))
        except (ValueError, TypeError, KeyError) as error:
            _invalid(counts, strict, "invalid_rows", number, error)
        if number % batch_size == 0:
            connection.commit()
    connection.commit()
    return dict(counts)


def ingest_genres(connection, path, manifest, strict=False, batch_size=50000):
    counts = Counter()
    for number, row in _rows(path, manifest, counts, strict):
        try:
            book_id = source_id(row.get("book_id"))
            genres = row.get("genres")
            if not isinstance(genres, dict):
                raise ValueError("Expected source genre-count dictionary")
            parsed = []
            for label, amount in genres.items():
                if not isinstance(label, str) or not label.strip() or isinstance(amount, bool):
                    raise ValueError("Invalid genre label/count")
                count = int(amount)
                if count != float(amount) or count < 0:
                    raise ValueError("Genre counts must be nonnegative integers")
                if count:
                    parsed.append((label.strip(), count))
            item = connection.execute("SELECT item_key FROM book_map WHERE book_id=?", (book_id,)).fetchone()
            if item is None:
                counts["unmatched_book_rows"] += 1
                continue
            # Presence is the diagnostic; max avoids counting edition duplicates as votes.
            connection.executemany("INSERT INTO genres VALUES (?,?,?) ON CONFLICT(item_key,genre) DO UPDATE SET source_count=max(source_count,excluded.source_count)",
                                   [(item[0], label, count) for label, count in parsed])
            counts["accepted_rows"] += 1
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            _invalid(counts, strict, "invalid_rows", number, error)
        if number % batch_size == 0:
            connection.commit()
    connection.commit()
    return dict(counts)


def ingest_interactions(connection, path, manifest, strict=False, batch_size=50000):
    counts, user_cache, book_cache = Counter(), BoundedCache(), BoundedCache()
    for number, row in _rows(path, manifest, counts, strict):
        try:
            user_id, book_id = source_id(row.get("user_id")), source_id(row.get("book_id"))
            rating_value = row.get("rating")
            if isinstance(rating_value, bool):
                raise ValueError("Boolean rating")
            rating = int(rating_value)
            if rating != float(rating_value) or not 0 <= rating <= 5:
                raise ValueError("Rating must be an integer 0..5")
            if rating == 0:
                counts["unrated_rows_excluded"] += 1
                continue
            # The final observed rating is only available at its update date.
            timestamp = date_timestamp(row.get("date_updated"))
            user_key = user_cache.get(user_id)
            if user_key is None:
                connection.execute("INSERT OR IGNORE INTO users(source_user_id) VALUES (?)", (user_id,))
                user_key = user_cache.put(user_id, connection.execute("SELECT user_key FROM users WHERE source_user_id=?", (user_id,)).fetchone()[0])
            item_key = book_cache.get(book_id)
            if item_key is None:
                item = connection.execute("SELECT item_key FROM book_map WHERE book_id=?", (book_id,)).fetchone()
                if item is None:
                    connection.execute("INSERT OR IGNORE INTO items(source_item_id) VALUES (?)", (f"book:{book_id}",))
                    item_key = connection.execute("SELECT item_key FROM items WHERE source_item_id=?", (f"book:{book_id}",)).fetchone()[0]
                    connection.execute("INSERT INTO book_map VALUES (?,?)", (book_id, item_key))
                    counts["distinct_books_without_metadata"] += 1
                else:
                    item_key = item[0]
                book_cache.put(book_id, item_key)
            connection.execute("""INSERT INTO interactions VALUES (?,?,?,?,?,?)
                ON CONFLICT(user_key,item_key) DO UPDATE SET rating=excluded.rating,
                timestamp=excluded.timestamp, source_row=excluded.source_row, source_book_id=excluded.source_book_id
                WHERE (excluded.timestamp,excluded.source_row)<(interactions.timestamp,interactions.source_row)""",
                               (user_key, item_key, rating, timestamp, number, book_id))
            counts["valid_rated_rows"] += 1
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            _invalid(counts, strict, "invalid_or_missing_date_rows" if row.get("date_updated") in (None, "") else "invalid_rows", number, error)
        if number % batch_size == 0:
            connection.commit()
    connection.commit()
    unique_count = connection.execute("SELECT count(*) FROM interactions").fetchone()[0]
    counts["unique_user_work_records"] = unique_count
    counts["duplicate_user_work_rows"] = counts["valid_rated_rows"] - unique_count
    return dict(counts)


def build_index(interactions, books, authors, output, genres=None, strict=False, batch_size=50000):
    """Build an exclusive new SQLite index; incomplete files cannot be sampled."""
    if batch_size < 1:
        raise ValueError("Commit batch size must be positive")
    paths = {"interactions": interactions, "books": books, "authors": authors}
    if genres:
        paths["genres"] = genres
    for path in paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        if source_format(path) != "jsonl":
            raise SourceError("Goodreads requires dated JSON-lines, optionally gzip; undated ratings CSV is unsupported")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of a frozen source index.
    with output.open("xb"):
        pass
    connection = connect_index(output)
    manifest, audit = {}, {}
    try:
        initialize(connection)
        set_setting(connection, "dataset", "UCSD Goodreads Book Graph")
        set_setting(connection, "date_policy", {
            "field": "date_updated", "meaning": "availability proxy for the final observed rating",
            "missing": "exclude and count (or fail with --strict)",
            "limitation": "Not the original reading/rating event time; no retrospective rating history is available.",
        })
        set_setting(connection, "duplicate_policy", "earliest date_updated per user/work, source-row tie-break; keep that row's rating")
        connection.commit()
        for key, operation in (("authors", ingest_authors), ("books", ingest_books), ("genres", ingest_genres), ("interactions", ingest_interactions)):
            if key not in paths:
                continue
            manifest[key] = {}
            audit[key] = operation(connection, paths[key], manifest[key], strict, batch_size)
        connection.execute("""CREATE TABLE user_summary AS
            SELECT user_key,count(*) AS history_length,avg(rating) AS mean_rating
            FROM interactions GROUP BY user_key""")
        connection.execute("CREATE UNIQUE INDEX summary_user ON user_summary(user_key)")
        connection.execute("CREATE INDEX summary_history ON user_summary(history_length)")
        set_setting(connection, "source_manifest", manifest)
        set_setting(connection, "ingest_audit", audit)
        set_setting(connection, "genres_available", genres is not None)
        set_setting(connection, "complete", True)
        connection.commit()
        return {"index": str(output), "schema_version": INDEX_SCHEMA, "source_manifest": manifest, "audit": audit}
    finally:
        connection.close()
