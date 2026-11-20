"""Read-only Goodreads metadata compatibility check; never opens authority data.

The author file is scanned fully. The book scan is an explicitly bounded prefix
by default, not a claim about all works or a representative sample of books.
An on-disk author-ID lookup keeps memory independent of total source size.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from .data_preparation.goodreads import (
    _contributors, _text, connect_index, ingest_authors, ingest_books,
    initialize, source_id,
)


def _stat(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def _records(path, maximum, summary, counters):
    before = _stat(path)
    digest = hashlib.sha256()
    scanned_bytes = 0
    complete = False
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as handle:
        for number, raw in enumerate(handle, 1):
            digest.update(raw)
            scanned_bytes += len(raw)
            counters["source_lines"] += 1
            if raw.strip():
                try:
                    record = json.loads(raw.decode("utf-8-sig"))
                    if not isinstance(record, dict):
                        raise ValueError("Expected JSON object")
                except (ValueError, UnicodeError):
                    counters["malformed_rows"] += 1
                else:
                    counters["json_object_rows"] += 1
                    yield number, record
            else:
                counters["blank_lines"] += 1
            if maximum is not None and number >= maximum:
                break
        else:
            complete = True
    after = _stat(path)
    if before != after:
        raise ValueError(f"Source changed during audit: {path}")
    summary.update(before=before, after=after, complete=complete,
                   scan_scope="full_file" if complete else "bounded_prefix",
                   maximum_source_lines=maximum,
                   scanned_uncompressed_bytes=scanned_bytes,
                   sha256_scanned_uncompressed_lines=digest.hexdigest(),
                   fingerprint_scope="scanned uncompressed source lines only; full file only if complete")


def audit_sources(authors_path, books_path, output_dir, max_books=100000,
                  acceptance_rows=1000):
    """Write a new report after full authors + bounded books validation.

    ``max_books=None`` deliberately opts into scanning all book records. No
    interaction index, sample, label mapping, API call or model is prepared.
    """
    authors_path, books_path, output_dir = map(Path, (authors_path, books_path, output_dir))
    if max_books is not None and max_books < 1:
        raise ValueError("max_books must be positive or None")
    if acceptance_rows < 1:
        raise ValueError("acceptance_rows must be positive")
    for path in (authors_path, books_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    authors_count, books_count = Counter(), Counter()
    author_fields, book_fields, roles = Counter(), Counter(), Counter()
    sources = {"authors": {}, "books": {}}
    with tempfile.TemporaryDirectory(prefix="metadata-audit-", dir=output_dir) as working:
        working = Path(working)
        lookup = sqlite3.connect(working / "author_ids.sqlite")
        lookup.execute("PRAGMA cache_size=-8192")
        lookup.execute("CREATE TABLE ids (author_id TEXT PRIMARY KEY) WITHOUT ROWID")
        authors_prefix, books_prefix = working / "authors.jsonl", working / "books.jsonl"
        try:
            with authors_prefix.open("w", encoding="utf-8") as excerpt:
                for number, row in _records(authors_path, None, sources["authors"], authors_count):
                    author_fields.update(row.keys())
                    authors_count["rows_with_gender_field"] += int("gender" in row)
                    try:
                        identifier = source_id(row.get("author_id"))
                        name = row.get("name", "")
                        if not isinstance(name, str):
                            raise ValueError("Author name is not a string")
                    except (ValueError, TypeError):
                        authors_count["invalid_schema_rows"] += 1
                        continue
                    authors_count["valid_schema_rows"] += 1
                    authors_count["empty_name_rows"] += int(not name.strip())
                    added = lookup.execute("INSERT OR IGNORE INTO ids VALUES (?)", (identifier,)).rowcount
                    authors_count["unique_author_ids" if added else "duplicate_author_ids"] += 1
                    if authors_count["valid_schema_rows"] <= acceptance_rows:
                        excerpt.write(json.dumps({"author_id": identifier, "name": name}, ensure_ascii=False) + "\n")
                    if number % 10000 == 0:
                        lookup.commit()
            lookup.commit()
            with books_prefix.open("w", encoding="utf-8") as excerpt:
                for _, row in _records(books_path, max_books, sources["books"], books_count):
                    book_fields.update(row.keys())
                    books_count["rows_with_user_id"] += int("user_id" in row)
                    books_count["rows_with_date_updated"] += int("date_updated" in row)
                    books_count["rows_with_rating"] += int("rating" in row)
                    try:
                        book_id = source_id(row.get("book_id"))
                        raw_work = row.get("work_id")
                        work_id = source_id(raw_work) if raw_work not in (None, "", "0", 0) else None
                        title, description = _text(row.get("title")), _text(row.get("description"))
                        contributors = _contributors(row.get("authors"))
                    except (ValueError, TypeError):
                        books_count["invalid_schema_rows"] += 1
                        continue
                    books_count["valid_schema_rows"] += 1
                    books_count["empty_title_rows"] += int(not title)
                    books_count["empty_description_rows"] += int(not description)
                    books_count["rows_without_work_id"] += int(work_id is None)
                    books_count["rows_without_contributors"] += int(not contributors)
                    books_count["rows_without_primary_literary_author"] += int(not any(p[-1] for p in contributors))
                    all_matched = bool(contributors)
                    for _, author_id, role, primary in contributors:
                        roles[role] += 1
                        books_count["contributor_references"] += 1
                        matched = lookup.execute("SELECT 1 FROM ids WHERE author_id=?", (author_id,)).fetchone() is not None
                        books_count["matched_contributor_references" if matched else "unmatched_contributor_references"] += 1
                        books_count["primary_literary_author_references"] += int(primary)
                        books_count["matched_primary_literary_author_references"] += int(primary and matched)
                        all_matched = all_matched and matched
                    books_count["rows_with_all_contributors_matched"] += int(all_matched)
                    if books_count["valid_schema_rows"] <= acceptance_rows:
                        # Acceptance fixture excludes long descriptions and every gender field.
                        excerpt.write(json.dumps({"book_id": book_id, "work_id": work_id,
                            "title": title, "description": "", "authors": row.get("authors", [])},
                            ensure_ascii=False) + "\n")
            connection = connect_index(working / "production_acceptance.sqlite")
            try:
                initialize(connection)
                author_manifest, book_manifest = {}, {}
                accepted_authors = ingest_authors(connection, authors_prefix, author_manifest, strict=True)
                accepted_books = ingest_books(connection, books_prefix, book_manifest, strict=True)
                if accepted_books.get("accepted_rows", 0) < 1:
                    raise ValueError("No real book rows passed production ingestion")
                production_acceptance = {"status": "passed", "authors": accepted_authors,
                    "books": accepted_books,
                    "scope": "bounded real-source title, work-ID and contributor rows; long descriptions omitted",
                    "items": connection.execute("SELECT count(*) FROM items").fetchone()[0],
                    "contributors": connection.execute("SELECT count(*) FROM contributors").fetchone()[0]}
            finally:
                connection.close()
        finally:
            lookup.close()
    reference_count = books_count["contributor_references"]
    primary_count = books_count["primary_literary_author_references"]
    report = {"schema_version": "goodreads-metadata-audit-v1", "started_at": started,
              "completed_at": datetime.now(timezone.utc).isoformat(), "sources": sources,
              "authors": dict(authors_count), "books": dict(books_count),
              "author_fields": dict(sorted(author_fields.items())),
              "book_fields": dict(sorted(book_fields.items())),
              "contributor_roles": dict(roles.most_common()),
              "contributor_reference_join_share": books_count["matched_contributor_references"] / reference_count if reference_count else None,
              "primary_author_reference_join_share": books_count["matched_primary_literary_author_references"] / primary_count if primary_count else None,
              "production_acceptance": production_acceptance,
              "limitations": ["Book prefix is not a random or representative sample.",
                  "Metadata does not replace dated reader interactions/reviews.",
                  "No VIAF input, identity link or author-gender annotation was inspected.",
                  "Author-ID joins establish source consistency, not authority identity or gender coverage."]}
    with (output_dir / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    raw = Path(__file__).parent / "data/raw"
    parser.add_argument("--authors", type=Path, default=raw / "goodreads_book_authors.json")
    parser.add_argument("--books", type=Path, default=raw / "goodreads_books.json")
    parser.add_argument("--output", required=True, type=Path, help="New report directory; existing output is refused")
    parser.add_argument("--max-books", type=int, default=100000, help="Maximum source lines; default is an explicitly bounded prefix")
    parser.add_argument("--all-books", action="store_true", help="Explicitly opt into a full book scan")
    args = parser.parse_args(argv)
    try:
        report = audit_sources(args.authors, args.books, args.output,
                               None if args.all_books else args.max_books)
    except (OSError, ValueError, EOFError, sqlite3.Error) as error:
        parser.exit(2, f"Metadata audit failed: {error}\n")
    print(json.dumps({"report": str(args.output / "report.json"),
                      "authors": report["authors"], "books": report["books"],
                      "book_scan_scope": report["sources"]["books"]["scan_scope"],
                      "contributor_reference_join_share": report["contributor_reference_join_share"],
                      "production_acceptance": report["production_acceptance"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
