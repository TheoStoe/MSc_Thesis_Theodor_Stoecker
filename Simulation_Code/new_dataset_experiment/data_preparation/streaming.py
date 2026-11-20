"""Bounded-memory readers. Source fingerprints cover the original file bytes."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from contextlib import contextmanager
from pathlib import Path


class SourceError(ValueError):
    pass


class _DigestReader(io.RawIOBase):
    def __init__(self, raw):
        self.raw = raw
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        size = self.raw.readinto(buffer)
        if size:
            self.digest.update(memoryview(buffer)[:size])
        return size


@contextmanager
def open_source(path, manifest):
    """Read gzip or plain UTF-8, hashing compressed bytes without a second pass."""
    path = Path(path).resolve()
    before = path.stat()
    with path.open("rb") as raw:
        hashing = _DigestReader(raw)
        buffered = io.BufferedReader(hashing)
        binary = gzip.GzipFile(fileobj=buffered) if path.suffix.lower() == ".gz" else buffered
        with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
            yield text
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise SourceError(f"Source changed while being read: {path}")
        manifest.update(path=str(path), size_bytes=after.st_size,
                        sha256=hashing.digest.hexdigest(), fingerprint="sha256-original-file-bytes")


def source_format(path):
    name = str(path).lower().removesuffix(".gz")
    if name.endswith(".csv"):
        return "csv"
    if name.endswith((".json", ".jsonl", ".ndjson")):
        return "jsonl"
    raise SourceError("Expected local .jsonl/.json/.ndjson or .csv, optionally .gz")


def json_rows(text, counters, strict=False):
    for row_number, line in enumerate(text, 1):
        if not line.strip():
            counters["blank_rows"] += 1
            continue
        counters["rows_seen"] += 1
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected an object")
        except (ValueError, TypeError) as error:
            counters["malformed_rows"] += 1
            if strict:
                raise SourceError(f"Malformed JSON at source row {row_number}") from error
            continue
        yield row_number, row


REVIEW_HEADERS = {
    "asin": "asin", "item": "asin", "item_id": "asin", "product_id": "asin",
    "reviewerid": "reviewerID", "user": "reviewerID", "user_id": "reviewerID",
    "rating": "overall", "overall": "overall",
    "timestamp": "unixReviewTime", "unixreviewtime": "unixReviewTime",
}


def csv_reviews(text, counters, strict=False):
    """Official headerless order is item, user, rating, timestamp."""
    reader = csv.reader(text)
    columns = None
    for row_number, values in enumerate(reader, 1):
        if not values or not any(v.strip() for v in values):
            counters["blank_rows"] += 1
            continue
        if columns is None:
            mapped = [REVIEW_HEADERS.get(v.strip().lower()) for v in values]
            if {"asin", "reviewerID", "overall", "unixReviewTime"}.issubset(set(mapped)):
                columns = mapped
                counters["header_rows"] += 1
                continue
            columns = ["asin", "reviewerID", "overall", "unixReviewTime"]
        counters["rows_seen"] += 1
        if len(values) != len(columns):
            counters["malformed_rows"] += 1
            if strict:
                raise SourceError(f"Wrong CSV field count at source row {row_number}")
            continue
        yield row_number, {key: value for key, value in zip(columns, values) if key}
