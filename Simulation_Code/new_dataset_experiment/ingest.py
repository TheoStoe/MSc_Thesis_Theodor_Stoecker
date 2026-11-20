"""Prepare a local Goodreads source index, without downloading or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_preparation.goodreads import build_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", required=True, type=Path, help="Dated Goodreads interactions or reviews JSONL[.gz]; undated CSV is insufficient")
    parser.add_argument("--books", required=True, type=Path, help="goodreads_books.json[.gz]")
    parser.add_argument("--authors", type=Path, default=Path(__file__).parent / "data/raw/goodreads_book_authors.json", help="Local Goodreads author metadata")
    parser.add_argument("--genres", type=Path, help="Optional goodreads_book_genres_initial.json[.gz]")
    parser.add_argument("--output", required=True, type=Path, help="New SQLite index; existing files are never overwritten")
    parser.add_argument("--strict", action="store_true", help="Fail on malformed rows instead of counting exclusions")
    args = parser.parse_args(argv)
    try:
        report = build_index(args.interactions, args.books, args.authors, args.output, args.genres, args.strict)
    except (OSError, ValueError, EOFError) as error:
        parser.exit(2, f"Preparation failed: {error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
