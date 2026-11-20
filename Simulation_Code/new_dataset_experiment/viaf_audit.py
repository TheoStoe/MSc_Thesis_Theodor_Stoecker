"""Offline, streaming audit of provisional Goodreads-author/VIAF name joins.

This reads existing authority assertions; it never predicts gender from a name.
An exact normalized full-name match is only a candidate identity link.  This is
an audit of local files, not a ready-to-use verified author-gender annotation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from typing import Any, BinaryIO, Iterator
import unicodedata
import xml.etree.ElementTree as ET


SCHEMA = "goodreads-viaf-name-coverage-audit-v1"
LABEL_SEMANTICS = "goodreads-primary-literary-author-recorded-gender-v1"
MAX_AUTHOR_RECORD_CHARS = 8 * 1024 * 1024
UNKNOWN_VALUES = frozenset({"unknown", "not known", "undetermined", "unspecified", "not applicable", "n/a", "u", "0", "9"})
PERSONAL_NAME_TAGS = frozenset({"100", "400", "700"})


class AuditInputError(ValueError):
    """The local input cannot be safely interpreted by this audit."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _open_binary(path: Path) -> BinaryIO:
    # Check the bytes rather than trusting a filename extension.
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rb") if compressed else path.open("rb")


def _source_info(path: Path, *, hash_input: bool) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": None,
        "hash_scope": "not_computed",
    }
    if hash_input:
        result.update(sha256=file_sha256(path), hash_scope="entire_file_as_stored")
    return result


def normalized_name_keys(name: str) -> set[str]:
    """Full-name keys only; preserve diacritics and do not expand initials.

    A single comma additionally permits the documented family/given-name order
    alternative.  Keys never use a first name alone or fuzzy string similarity.
    Collisions and multiple matching clusters remain visible in the results.
    """
    name = unicodedata.normalize("NFKC", name).casefold().strip()

    def clean(value: str) -> str:
        return " ".join("".join(c if c.isalnum() else " " for c in value).split())

    keys = {clean(name)}
    if name.count(",") == 1:
        family, given = name.split(",")
        if family.strip() and given.strip():
            keys.add(clean(f"{given} {family}"))
    return keys - {""}


def _json_array(stream: Any) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer, position, eof = "", 0, False

    def refill() -> None:
        nonlocal buffer, position, eof
        buffer = buffer[position:]
        position = 0
        chunk = stream.read(65536)
        eof = not chunk
        buffer += chunk
        if len(buffer) > MAX_AUTHOR_RECORD_CHARS:
            raise AuditInputError("A Goodreads author JSON record exceeds the 8 MiB safety limit")

    def token() -> str:
        nonlocal position
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                return buffer[position]
            if eof:
                return ""
            refill()

    if token() != "[":
        raise AuditInputError("Expected a Goodreads author JSON array")
    position += 1
    if token() == "]":
        position += 1
    else:
        while True:
            token()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise AuditInputError(f"Invalid or truncated author JSON array: {exc}") from exc
                    refill()
            if not isinstance(value, dict):
                raise AuditInputError("Every Goodreads author array entry must be an object")
            yield value
            separator = token()
            if separator == "]":
                position += 1
                break
            if separator != ",":
                raise AuditInputError("Invalid or truncated author JSON array: expected comma or ]")
            position += 1
            if token() == "]":
                raise AuditInputError("Trailing comma in Goodreads author JSON array")
    if token():
        raise AuditInputError("Unexpected data after Goodreads author JSON array")


def iter_authors(path: Path) -> Iterator[dict[str, Any]]:
    """Stream Goodreads JSON objects, in JSON Lines or a JSON array."""
    import io

    with _open_binary(path) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as stream:
        first = ""
        while not first:
            character = stream.read(1)
            if not character:
                raise AuditInputError("The Goodreads author file is empty")
            if not character.isspace():
                first = character
        stream.seek(0)
        if first == "[":
            yield from _json_array(stream)
            return
        if first != "{":
            raise AuditInputError("Goodreads authors must be JSON Lines or a JSON array")
        line_number = 0
        while line := stream.readline(MAX_AUTHOR_RECORD_CHARS + 1):
            line_number += 1
            if len(line) > MAX_AUTHOR_RECORD_CHARS:
                raise AuditInputError(f"Author JSON line {line_number} exceeds the 8 MiB safety limit")
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditInputError(f"Invalid author JSON line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AuditInputError(f"Author JSON line {line_number} must be an object")
            yield value


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def detect_viaf_format(stream: BinaryIO) -> str:
    """Inspect a small prefix, without reading a whole collection into memory."""
    sample = stream.read(65536)
    stream.seek(0)
    sample = sample.lstrip(b"\xef\xbb\xbf \r\n\t")
    if not sample:
        raise AuditInputError("The VIAF file is empty")
    if re.match(rb"\d+\t\s*<", sample):
        return "numeric_id_tab_marcxml_records"
    if not sample.startswith(b"<"):
        raise AuditInputError(
            "Expected MARCXML, but the VIAF file is not XML. It may be ISO2709 "
            "binary MARC (.iso/.mrc). Use the clusters-marc21.xml.gz export."
        )
    root_prefix = re.sub(rb"^<\?xml[^?]*\?>\s*", b"", sample)
    root_prefix = re.sub(rb"^(?:<!--.*?-->\s*)+", b"", root_prefix, flags=re.DOTALL)
    root_match = re.match(rb"<([\w:.-]+)(?:\s|/|>)", root_prefix)
    root_name = root_match.group(1).split(b":")[-1] if root_match else b""
    if root_name == b"collection":
        return "marcxml_collection"
    if root_name == b"record":
        # A compact one-record-per-line file may have repeated XML declarations.
        first_line = root_prefix.splitlines()[0]
        if re.search(rb"</(?:[\w.-]+:)?record\s*>", first_line) or b"\n" not in root_prefix:
            return "marcxml_record_lines"
        return "marcxml_document"
    raise AuditInputError(
        "Expected MARCXML <collection> or <record>, not a raw VIAF cluster/RDF "
        "export or another XML format. Use clusters-marc21.xml.gz."
    )


def iter_marc_records(stream: BinaryIO, framing: str) -> Iterator[tuple[ET.Element, str | None]]:
    if framing in {"numeric_id_tab_marcxml_records", "marcxml_record_lines"}:
        for line_number, line in enumerate(stream, 1):
            line = line.lstrip(b"\xef\xbb\xbf \r\n\t")
            if not line:
                continue
            if re.fullmatch(rb"<\?xml[^?]*\?>\s*", line):
                continue
            prefix_id = None
            if re.match(rb"\d+\t", line):
                raw_id, line = line.split(b"\t", 1)
                prefix_id = raw_id.decode("ascii")
            elif framing == "numeric_id_tab_marcxml_records":
                raise AuditInputError(f"Missing numeric VIAF ID/tab prefix on XML line {line_number}")
            try:
                record = ET.fromstring(line)
            except ET.ParseError as exc:
                raise AuditInputError(f"Invalid or truncated MARCXML record line {line_number}: {exc}") from exc
            if _local(record.tag) != "record":
                raise AuditInputError(f"XML line {line_number} is not a MARCXML record")
            yield record, prefix_id
            record.clear()
        return

    stack: list[ET.Element] = []
    try:
        for event, element in ET.iterparse(stream, events=("start", "end")):
            if event == "start":
                stack.append(element)
                continue
            if _local(element.tag) == "record":
                yield element, None
                # Clearing alone leaves millions of empty record children in the
                # collection. Remove each completed record from its parent too.
                if len(stack) >= 2:
                    stack[-2].remove(element)
                element.clear()
            stack.pop()
    except ET.ParseError as exc:
        raise AuditInputError(f"Invalid or truncated MARCXML document: {exc}") from exc


def extract_record(record: ET.Element, prefix_id: str | None = None) -> dict[str, Any]:
    headings: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    control_fields: dict[str, list[str]] = {}
    leader = None
    for field in record:
        kind = _local(field.tag)
        if kind == "leader":
            leader = field.text or ""
        elif kind == "controlfield":
            control_fields.setdefault(field.get("tag", ""), []).append(field.text or "")
        elif kind == "datafield":
            tag = field.get("tag", "")
            subfields = [
                {"code": sub.get("code", ""), "value": sub.text or ""}
                for sub in field if _local(sub.tag) == "subfield"
            ]
            if tag in PERSONAL_NAME_TAGS:
                for sub in subfields:
                    if sub["code"] == "a" and sub["value"].strip():
                        headings.append({"tag": tag, "name": sub["value"], "subfields": subfields})
            if tag == "375":
                def values(code: str) -> list[str]:
                    return [sub["value"] for sub in subfields if sub["code"] == code]

                claims.append({
                    "raw_values": values("a"), "start_periods": values("s"),
                    "end_periods": values("t"), "uris": values("u"),
                    "source_notes": values("v"), "term_sources": values("2"),
                    "authority_record_ids": values("0"), "real_world_uris": values("1"),
                    "data_provenance": values("7"), "subfields": subfields,
                })
    ids = control_fields.get("001", [])
    record_id = ids[0].strip() if ids and ids[0].strip() else prefix_id
    authority_record = leader is None or (len(leader) > 6 and leader[6] == "z")
    return {
        "viaf_id": record_id, "framing_id": prefix_id,
        "record_control_fields": control_fields, "leader": leader,
        "is_personal_authority": bool(headings) and authority_record,
        "personal_name_headings": headings, "gender_375_claims": claims,
    }


def _evidence_status(evidence: list[dict[str, Any]]) -> str:
    claims = [claim for record in evidence for claim in record["gender_375_claims"]]
    if not claims:
        return "no_375"
    raw = {value.strip() for claim in claims for value in claim["raw_values"] if value.strip()}
    if not raw:
        return "missing_375_value"
    if all(value.casefold() in UNKNOWN_VALUES for value in raw):
        return "unknown_only"
    # Multiple values may be different dated assertions, vocabulary variants,
    # or actual contradictions; do not collapse or call all of them conflicts.
    return "single_raw_value" if len(raw) == 1 else "multiple_raw_values_unresolved"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    # OneDrive/Windows indexing can briefly hold a just-written status file.
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * 2 ** attempt)


def _load_requested_ids(path: Path, connection: sqlite3.Connection) -> int:
    connection.execute("CREATE TABLE requested (author_id TEXT PRIMARY KEY)")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.reader(stream)):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != 1:
                raise AuditInputError("--author-ids accepts one author_id column, with an optional header")
            value = row[0].strip()
            if index == 0 and value == "author_id":
                continue
            connection.execute("INSERT OR IGNORE INTO requested VALUES (?)", (value,))
    count = connection.execute("SELECT COUNT(*) FROM requested").fetchone()[0]
    if not count:
        raise AuditInputError("The author ID selection is empty")
    connection.execute("INSERT INTO authors(author_id, name) SELECT author_id, NULL FROM requested")
    return count


def _load_targets(path: Path, ids_path: Path | None, connection: sqlite3.Connection,
                  selected_ids: set[str] | None = None) -> dict[str, Any]:
    connection.executescript("""
        CREATE TABLE authors (author_id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE names (name_key TEXT, author_id TEXT, PRIMARY KEY(name_key, author_id));
        CREATE TABLE records (record_key INTEGER PRIMARY KEY, cluster_key TEXT, viaf_id TEXT, evidence TEXT);
        CREATE TABLE matches (author_id TEXT, cluster_key TEXT, record_key INTEGER,
                              PRIMARY KEY(author_id, cluster_key, record_key));
        CREATE INDEX match_authors ON matches(author_id, cluster_key);
    """)
    filtered = ids_path is not None or selected_ids is not None
    requested = _load_requested_ids(ids_path, connection) if ids_path else None
    if ids_path is None and selected_ids is not None:
        connection.execute("CREATE TABLE requested (author_id TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO requested VALUES (?)", ((value,) for value in sorted(selected_ids)))
        connection.execute("INSERT INTO authors(author_id, name) SELECT author_id, NULL FROM requested")
        requested = len(selected_ids)
    records_seen, duplicates, selected_records = 0, 0, 0
    for author in iter_authors(path):
        records_seen += 1
        raw_id = author.get("author_id")
        if raw_id is None or isinstance(raw_id, bool) or not str(raw_id).strip():
            raise AuditInputError(f"Author record {records_seen} has no valid author_id")
        author_id = str(raw_id).strip()
        if filtered and not connection.execute("SELECT 1 FROM requested WHERE author_id=?", (author_id,)).fetchone():
            continue
        selected_records += 1
        name = author.get("name")
        if name is None:
            name = ""
        if not isinstance(name, str):
            raise AuditInputError(f"Author {author_id} has a non-string name")
        existing = connection.execute("SELECT name FROM authors WHERE author_id=?", (author_id,)).fetchone()
        if existing and existing[0] is not None:
            if existing[0] != name:
                raise AuditInputError(f"Conflicting names for duplicate Goodreads author_id {author_id}")
            duplicates += 1
            continue
        connection.execute("INSERT OR REPLACE INTO authors VALUES (?, ?)", (author_id, name))
        connection.executemany("INSERT OR IGNORE INTO names VALUES (?, ?)", ((key, author_id) for key in normalized_name_keys(name)))
        if records_seen % 10000 == 0:
            connection.commit()
    connection.commit()
    target_count = connection.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
    if not target_count and not filtered:
        raise AuditInputError("No target Goodreads authors were found")
    return {
        "author_records_seen": records_seen, "selected_author_records": selected_records,
        "duplicate_author_records": duplicates, "requested_author_ids": requested,
        "target_authors": target_count,
        "target_author_ids_unique": target_count,
        "target_selection": "explicit_author_ids" if ids_path else "cohort_literary_author_ids" if selected_ids is not None else "all_author_records",
        "missing_requested_author_records": connection.execute("SELECT COUNT(*) FROM authors WHERE name IS NULL").fetchone()[0],
        "missing_author_names": connection.execute("SELECT COUNT(*) FROM authors WHERE name='' ").fetchone()[0],
        "normalized_name_collision_keys": connection.execute("SELECT COUNT(*) FROM (SELECT name_key FROM names GROUP BY name_key HAVING COUNT(*) > 1)").fetchone()[0],
        "authors_in_normalized_name_collisions": connection.execute("SELECT COUNT(DISTINCT author_id) FROM names WHERE name_key IN (SELECT name_key FROM names GROUP BY name_key HAVING COUNT(*) > 1)").fetchone()[0],
    }


def _write_matches(connection: sqlite3.Connection, path: Path) -> tuple[Counter[str], Counter[str]]:
    match_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for author_id, name in connection.execute("SELECT author_id, name FROM authors ORDER BY author_id"):
            candidates: list[dict[str, Any]] = []
            for cluster_key, in connection.execute("SELECT DISTINCT cluster_key FROM matches WHERE author_id=? ORDER BY cluster_key", (author_id,)):
                evidence = [json.loads(row[0]) for row in connection.execute(
                    "SELECT r.evidence FROM records r JOIN matches m USING(record_key) "
                    "WHERE m.author_id=? AND m.cluster_key=? ORDER BY r.record_key", (author_id, cluster_key)
                )]
                candidates.append({
                    "viaf_id": evidence[0]["viaf_id"], "record_evidence": evidence,
                    "gender_evidence_status": _evidence_status(evidence),
                })
            if name is None:
                status = "missing_author_record"
            elif not normalized_name_keys(name):
                status = "missing_author_name"
            elif not candidates:
                status = "unmatched"
            elif len(candidates) > 1:
                status = "ambiguous_multiple_viaf_clusters"
            elif candidates[0]["viaf_id"] is None:
                status = "matched_record_missing_viaf_id"
            else:
                status = "unique_provisional_name_match"
            match_counts[status] += 1
            if status == "unique_provisional_name_match":
                evidence_counts[candidates[0]["gender_evidence_status"]] += 1
            payload = {
                "author_id": author_id, "author_name": name, "match_status": status,
                "candidate_cluster_count": len(candidates), "candidates": candidates,
                "identity_verified": False,
            }
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)
    return match_counts, evidence_counts


LITERARY_ROLES = frozenset({"", "author", "coauthor", "co-author", "writer", "primary_literary_author"})
KNOWN_ITEM_GENDERS = frozenset({"female", "male", "nonbinary", "mixed", "other"})
KNOWN_PRIMARY_GENDERS = KNOWN_ITEM_GENDERS - {"mixed"}


def _item_authors(item: dict[str, Any]) -> list[str]:
    """Use only the pipeline's declared literary author IDs, preserving teams.

    Goodreads blank contributor roles are treated as author roles by an explicit
    dataset convention. Contributor roles and competing primary authors remain
    in the cohort; this audit does not substitute an editor or translator.
    """
    if "author_ids" in item:
        return sorted({str(value) for value in item["author_ids"] if str(value).strip()})
    return sorted({
        str(row["author_id"]) for row in item.get("contributors", [])
        if row.get("author_id") is not None and str(row.get("role", "")).casefold().strip() in LITERARY_ROLES
    })


def _primary_authors(item: dict[str, Any], literary_ids: list[str]) -> tuple[list[str], str]:
    """Use recorded per-edition primary flags without guessing across editions."""
    eligible = set(literary_ids)
    flagged = {
        str(row["author_id"]) for row in item.get("contributors", [])
        if row.get("primary_literary_author") is True
        and row.get("author_id") is not None
        and str(row.get("role", "")).casefold().strip() in LITERARY_ROLES
    }
    if not flagged:
        return [], "missing_recorded_primary_author"
    if not flagged.issubset(eligible):
        return sorted(flagged), "inconsistent_primary_and_literary_author_ids"
    declared = {str(value) for value in item.get("primary_author_ids", flagged)}
    if declared != flagged:
        return sorted(flagged | declared), "inconsistent_recorded_primary_author_ids"
    if len(flagged) > 1:
        return sorted(flagged), "competing_primary_authors_across_source_editions"
    return sorted(flagged), "single_recorded_primary_author"


def _classify_reviewed_authors(author_ids: list[str], reviews: dict[str, dict[str, Any]],
                               candidates: dict[str, Any]) -> dict[str, str]:
    if not author_ids:
        return {"gender": "unknown", "status": "unknown", "reason": "no_literary_author_ids"}
    if not all(author_id in reviews for author_id in author_ids):
        return {"gender": "unknown", "status": "provisional" if candidates else "unknown", "reason": "author_linkage_or_gender_not_reviewed"}
    genders = {reviews[author_id]["gender"] for author_id in author_ids}
    if "ambiguous" in genders:
        gender, reason = "ambiguous", "reviewed_author_evidence_ambiguous"
    elif "unknown" in genders:
        gender, reason = "unknown", "reviewed_author_gender_unknown"
    elif len(genders) > 1:
        gender, reason = "mixed", "reviewed_mixed_author_team"
    else:
        gender, reason = next(iter(genders)), "all_selected_literary_authors_reviewed"
    return {"gender": gender, "status": "reviewed", "reason": reason}


def _read_reviews(path: Path | None, matches: dict[str, dict[str, Any]], target_author_ids: set[str]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    required = {"author_id", "viaf_id", "gender", "reviewed_by", "reviewed_at", "source_url", "source_record_id", "review_note"}
    reviews: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise AuditInputError("Reviewed mapping CSV requires columns: " + ",".join(sorted(required)))
        for number, row in enumerate(reader, 2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            row = {key: str(row.get(key) or "").strip() for key in required}
            author_id = row["author_id"]
            if author_id not in target_author_ids:
                # A reviewed author mapping can be reused for later cohorts.
                continue
            if not all(row[key] for key in ("author_id", "viaf_id", "gender", "reviewed_by", "reviewed_at", "review_note")):
                raise AuditInputError(f"Reviewed mapping row {number} lacks reviewer, date, note, author, VIAF ID or gender")
            if row["gender"] not in KNOWN_ITEM_GENDERS | {"unknown", "ambiguous"} or row["gender"] == "mixed":
                raise AuditInputError(f"Invalid reviewed author gender on row {number}; mixed is an item-team category")
            if not (row["source_url"] or row["source_record_id"]):
                raise AuditInputError(f"Reviewed mapping row {number} requires a source URL or source record ID")
            if row["source_url"] and not re.match(r"https?://[^\s]+$", row["source_url"]):
                raise AuditInputError(f"Invalid source URL on reviewed mapping row {number}")
            try:
                datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
            except ValueError as exc:
                raise AuditInputError(f"Reviewed mapping row {number} needs an ISO date/date-time") from exc
            candidates = matches.get(author_id, {}).get("candidates", [])
            candidate = next((value for value in candidates if value["viaf_id"] == row["viaf_id"]), None)
            if candidate is None:
                raise AuditInputError(f"Reviewed author {author_id}/VIAF {row['viaf_id']} is not an observed name-match candidate")
            if author_id in reviews:
                raise AuditInputError(f"Duplicate/conflicting reviewed mapping for author {author_id}")
            reviews[author_id] = {**row, "selected_candidate_evidence": candidate}
    return reviews


def _read_label_cohort(cohort_path: Path) -> tuple[bytes, dict[str, Any]]:
    cohort_bytes = cohort_path.read_bytes()
    cohort = json.loads(cohort_bytes)
    if cohort.get("schema_version", cohort.get("schema")) != "goodreads-cohort-v1":
        raise AuditInputError("Expected goodreads-cohort-v1 for cohort label export")
    if "sampling" in cohort or "cohort_sha256" in cohort or len(cohort.get("users", [])) == 300:
        # Production cohorts must pass the same immutable content/sidecar/ID
        # checks as preparation. Small isolated fixtures have no sampling block.
        from .sampling import load_cohort
        if load_cohort(cohort_path) != cohort:
            raise AuditInputError("Cohort changed during frozen-manifest validation")
    return cohort_bytes, cohort


def export_cohort_labels(
    cohort_path: str | Path, output_path: str | Path, *,
    matches_path: str | Path | None = None,
    reviewed_mapping_path: str | Path | None = None,
    audit_complete: bool = False,
    audit_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export evaluation-only labels; no reviewed CSV means all labels unknown.

    This also works before VIAF is available: omit matches/reviews to preserve
    every cohort item with an unknown label. Labels never alter the cohort.
    Existing output files are rejected. A review CSV selects an observed VIAF
    candidate by ID and explicitly states the reviewed gender and provenance;
    merely having one normalized name candidate does not constitute review.
    """
    cohort_path, output_path = Path(cohort_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    cohort_bytes, cohort = _read_label_cohort(cohort_path)
    items = cohort.get("items", [])
    by_id: dict[str, dict[str, Any]] = {}
    needed_authors: set[str] = set()
    for item in items:
        item_id = str(item["item_id"])
        if item_id in by_id:
            raise AuditInputError(f"Duplicate cohort item ID {item_id}")
        by_id[item_id] = item
        needed_authors.update(_item_authors(item))
    matches: dict[str, dict[str, Any]] = {}
    if matches_path:
        with Path(matches_path).open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["author_id"] in needed_authors:
                    if row["author_id"] in matches:
                        raise AuditInputError("Duplicate author in name-match audit")
                    matches[row["author_id"]] = row
    reviews = _read_reviews(Path(reviewed_mapping_path) if reviewed_mapping_path else None, matches, needed_authors)
    labels: dict[str, dict[str, Any]] = {}
    provisional_books = 0
    multi_author_books = 0
    missing_author_books = 0
    primary_missing_books = 0
    primary_ambiguous_books = 0
    for item_id, item in by_id.items():
        author_ids = _item_authors(item)
        multi_author_books += len(author_ids) > 1
        missing_author_books += not author_ids
        candidate_evidence = {author_id: matches[author_id] for author_id in author_ids if author_id in matches}
        all_single_raw = bool(author_ids) and all(
            matches.get(author_id, {}).get("match_status") == "unique_provisional_name_match"
            and matches[author_id]["candidates"][0]["gender_evidence_status"] == "single_raw_value"
            and not any(claim["start_periods"] or claim["end_periods"] for record in matches[author_id]["candidates"][0]["record_evidence"] for claim in record["gender_375_claims"])
            for author_id in author_ids
        )
        provisional_books += all_single_raw
        team = _classify_reviewed_authors(author_ids, reviews, candidate_evidence)
        primary_ids, primary_resolution = _primary_authors(item, author_ids)
        primary_candidates = {author_id: candidate_evidence[author_id] for author_id in primary_ids if author_id in candidate_evidence}
        if primary_resolution == "single_recorded_primary_author":
            primary = _classify_reviewed_authors(primary_ids, reviews, primary_candidates)
        elif primary_resolution == "missing_recorded_primary_author":
            primary_missing_books += 1
            primary = {"gender": "unknown", "status": "unknown", "reason": primary_resolution}
        else:
            primary_ambiguous_books += 1
            primary = {"gender": "ambiguous", "status": "provisional", "reason": primary_resolution}
        labels[item_id] = {
            **primary, "classification_scope": "primary_literary_author",
            "primary_author_ids": primary_ids, "primary_author_resolution": primary_resolution,
            "author_ids": author_ids,
            "secondary_all_literary_team": {**team, "author_ids": author_ids,
                                             "classification_scope": "all_literary_authors"},
            "evidence": {"provisional_name_matches": candidate_evidence,
                         "reviewed_mapping": [reviews[author_id] for author_id in author_ids if author_id in reviews]},
        }
    train_counts: Counter[str] = Counter()
    for user in cohort.get("users", []):
        for event in user.get("train", []):
            item_id = str(event["item_id"])
            if item_id not in labels:
                raise AuditInputError(f"Training interaction references missing cohort item {item_id}")
            train_counts[item_id] += 1
    known_books = sum(row["gender"] in KNOWN_PRIMARY_GENDERS and row["status"] == "reviewed" for row in labels.values())
    known_interactions = sum(count for item_id, count in train_counts.items() if labels[item_id]["gender"] in KNOWN_PRIMARY_GENDERS and labels[item_id]["status"] == "reviewed")
    total_interactions = sum(train_counts.values())
    genre_rows: dict[str, dict[str, float]] = {}
    for item_id, item in by_id.items():
        genres = sorted(set(item.get("genres", []))) or ["[unclassified]"]
        known = labels[item_id]["gender"] in KNOWN_PRIMARY_GENDERS and labels[item_id]["status"] == "reviewed"
        for genre in genres:
            row = genre_rows.setdefault(genre, {"book_weight": 0.0, "known_book_weight": 0.0, "training_interaction_weight": 0.0, "known_training_interaction_weight": 0.0})
            weight = 1 / len(genres)
            row["book_weight"] += weight
            row["known_book_weight"] += weight * known
            row["training_interaction_weight"] += weight * train_counts[item_id]
            row["known_training_interaction_weight"] += weight * train_counts[item_id] * known
    artifact = {
        "schema": "goodreads-item-labels-v1", "schema_version": "goodreads-item-labels-v1", "created_at_utc": _utc_now(),
        "label_semantics_version": LABEL_SEMANTICS,
        "cohort_fingerprint": hashlib.sha256(cohort_bytes).hexdigest(),
        "cohort_fingerprint_scope": "cohort_file_sha256",
        "linkage_reviewed": bool(reviews) and audit_complete,
        "authority_scan_complete": audit_complete,
        "reviewed_mapping_source": _source_info(Path(reviewed_mapping_path), hash_input=True) if reviewed_mapping_path else None,
        "audit_provenance": audit_provenance,
        "labels": labels,
        "coverage": {
            "book_count": len(labels), "known_book_count": known_books,
            "known_book_share": known_books / len(labels) if labels else None,
            "training_interaction_count": total_interactions, "known_training_interaction_count": known_interactions,
            "known_training_interaction_share": known_interactions / total_interactions if total_interactions else None,
            "gender_counts": dict(Counter(row["gender"] for row in labels.values())),
            "gender_training_interaction_counts": dict(Counter({gender: sum(count for item_id, count in train_counts.items() if labels[item_id]["gender"] == gender) for gender in sorted({row['gender'] for row in labels.values()})})),
            "books_with_all_authors_unique_non_temporal_raw_375_candidates": provisional_books,
            "multi_literary_author_books": multi_author_books, "missing_literary_author_books": missing_author_books,
            "missing_recorded_primary_author_books": primary_missing_books,
            "ambiguous_recorded_primary_author_books": primary_ambiguous_books,
            "secondary_all_literary_team_gender_counts": dict(Counter(row["secondary_all_literary_team"]["gender"] for row in labels.values())),
            "unknown_books_retained": sum(row["gender"] == "unknown" for row in labels.values()),
            "genre_attribution": "fractional: each book/interaction contributes 1 / number_of_genres to each associated genre",
            "by_genre": dict(sorted(genre_rows.items())),
            "known_definition": sorted(KNOWN_PRIMARY_GENDERS),
        },
        "method": {
            "evaluation_only": True, "native_goodreads_gender_field": False,
            "identity_inference_from_names": False,
            "name_linkage": "provisional exact normalized full-name candidates; reviewed ID selections required for labels",
            "primary_metric": "recorded primary literary author from per-edition contributor flags; competing primary IDs across editions stay ambiguous regardless of gender agreement",
            "secondary_metric": "all literary author team, reported separately; it never substitutes for the primary metric",
            "author_role_convention": "cohort author_ids are literary contributors; empty Goodreads role is treated as author; primary identity requires recorded primary_literary_author=True flags",
            "multiple_or_temporal_375": "unresolved until explicit reviewer label and supporting source are supplied",
            "unknown_retention": "all original cohort books and training interactions retained",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(artifact, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return artifact


def audit(
    authors_path: str | Path, viaf_path: str | Path, output_dir: str | Path, *,
    author_ids_path: str | Path | None = None, max_records: int | None = None,
    hash_inputs: bool = False, cohort_path: str | Path | None = None,
    reviewed_mapping_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit local files. Existing output directories are always rejected.

    ``max_records`` bounds the VIAF prefix only, not the target-author file. If
    the bound is reached the output is explicitly partial, even if the prefix
    happens to equal the number of records in the file. Such a scan cannot
    establish a unique match in the unscanned remainder or full label coverage.
    Hashing, when requested, reads each entire compressed/raw source separately.
    """
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    if reviewed_mapping_path and not cohort_path:
        raise ValueError("A cohort is required to export reviewed item labels")
    authors_path, viaf_path, output = Path(authors_path), Path(viaf_path), Path(output_dir)
    ids_path = Path(author_ids_path) if author_ids_path else None
    for path in (authors_path, viaf_path, ids_path, Path(cohort_path) if cohort_path else None, Path(reviewed_mapping_path) if reviewed_mapping_path else None):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    started = _utc_now()
    _atomic_json(output / "status.json", {"schema": SCHEMA, "state": "incomplete", "started_at_utc": started})
    try:
        sources = {"authors": _source_info(authors_path, hash_input=hash_inputs), "viaf": _source_info(viaf_path, hash_input=hash_inputs)}
        if ids_path:
            sources["author_ids"] = _source_info(ids_path, hash_input=hash_inputs)
        if cohort_path:
            sources["cohort"] = _source_info(Path(cohort_path), hash_input=True)
            sidecar = Path(cohort_path).with_suffix(".manifest.json")
            if sidecar.exists():
                sources["cohort_manifest"] = _source_info(sidecar, hash_input=True)
        if reviewed_mapping_path:
            sources["reviewed_mapping"] = _source_info(Path(reviewed_mapping_path), hash_input=True)
        with tempfile.TemporaryDirectory(prefix=".audit_work_", dir=output) as temporary:
            connection = sqlite3.connect(Path(temporary) / "audit.sqlite")
            try:
                connection.execute("PRAGMA cache_size=-8192")
                selected_ids = None
                if cohort_path and ids_path is None:
                    _, selected_cohort = _read_label_cohort(Path(cohort_path))
                    selected_ids = {author_id for item in selected_cohort.get("items", []) for author_id in _item_authors(item)}
                targets = _load_targets(authors_path, ids_path, connection, selected_ids)
                counts: Counter[str] = Counter()
                raw_values: Counter[str] = Counter()
                personal_values: Counter[str] = Counter()
                prefix_reached = False
                with _open_binary(viaf_path) as stream:
                    framing = detect_viaf_format(stream)
                    for ordinal, (record, prefix_id) in enumerate(iter_marc_records(stream, framing), 1):
                        counts["records_seen"] += 1
                        evidence = extract_record(record, prefix_id)
                        claims = evidence["gender_375_claims"]
                        if claims:
                            counts["records_with_375"] += 1
                        for claim in claims:
                            raw_values.update(claim["raw_values"])
                        if evidence["is_personal_authority"]:
                            counts["personal_records"] += 1
                            if claims:
                                counts["personal_records_with_375"] += 1
                            for claim in claims:
                                personal_values.update(claim["raw_values"])
                            if evidence["viaf_id"] is None:
                                counts["personal_records_missing_id"] += 1
                            if prefix_id and evidence["viaf_id"] != prefix_id:
                                counts["framing_and_001_id_disagreements"] += 1
                            author_ids: set[str] = set()
                            for heading in evidence["personal_name_headings"]:
                                for key in normalized_name_keys(heading["name"]):
                                    author_ids.update(row[0] for row in connection.execute("SELECT author_id FROM names WHERE name_key=?", (key,)))
                            if author_ids:
                                counts["personal_records_matching_target_names"] += 1
                                cluster_key = ("id:" + evidence["viaf_id"]) if evidence["viaf_id"] else f"missing:{ordinal}"
                                connection.execute("INSERT INTO records VALUES (?, ?, ?, ?)", (ordinal, cluster_key, evidence["viaf_id"], json.dumps(evidence, ensure_ascii=False)))
                                connection.executemany("INSERT INTO matches VALUES (?, ?, ?)", ((author_id, cluster_key, ordinal) for author_id in author_ids))
                        if ordinal % 10000 == 0:
                            connection.commit()
                        if max_records is not None and ordinal >= max_records:
                            prefix_reached = True
                            break
                connection.commit()
                if not counts["records_seen"]:
                    raise AuditInputError("No MARCXML records were found")
                match_counts, evidence_counts = _write_matches(connection, output / "matches.jsonl")
                for key in ("unique_provisional_name_match", "unmatched", "ambiguous_multiple_viaf_clusters", "matched_record_missing_viaf_id", "missing_author_record", "missing_author_name"):
                    match_counts[key] += 0
                named_375 = evidence_counts["single_raw_value"] + evidence_counts["multiple_raw_values_unresolved"]
                unique_count = match_counts["unique_provisional_name_match"]
                # Always expose zero categories so absence is not confused with
                # an unimplemented/missing metric in downstream reports.
                for key in ("records_seen", "personal_records", "records_with_375", "personal_records_with_375", "personal_records_missing_id", "personal_records_matching_target_names", "framing_and_001_id_disagreements"):
                    counts[key] += 0
                summary: dict[str, Any] = {
                    "schema": SCHEMA, "state": "partial" if prefix_reached else "complete",
                    "started_at_utc": started, "finished_at_utc": _utc_now(),
                    "sources": sources, "viaf_format": framing,
                    "scan": {"viaf_eof_reached": not prefix_reached, "max_records": max_records, "counts": dict(counts)},
                    "targets": targets,
                    "provisional_match_counts": dict(match_counts),
                    "unique_match_375_evidence_counts": dict(evidence_counts),
                    "raw_375_value_occurrences_all_records": dict(sorted(raw_values.items())),
                    "raw_375_value_occurrences_personal_records": dict(sorted(personal_values.items())),
                    "coverage": {
                        "denominator_target_authors": targets["target_authors"],
                        "unique_provisional_matches_with_non_unknown_375_values": named_375,
                        "unique_provisional_match_rate": unique_count / targets["target_authors"] if targets["target_authors"] else None,
                        "provisional_non_unknown_375_value_rate": named_375 / targets["target_authors"] if targets["target_authors"] else None,
                        "full_scan_coverage_established": not prefix_reached,
                        "verified_identity_gender_coverage": None,
                    },
                    "interpretation": [
                        "Exact normalized full-name matches are provisional candidate identity links, not verified identities.",
                        "Multiple matching VIAF clusters remain ambiguous even when their raw 375 values agree.",
                        "Raw 375 assertions, dates and source qualifiers are preserved; multiple values are not automatically identity conflicts.",
                        "No gender is inferred from names; no claims are made about discrimination or causal bias.",
                        "A partial scan describes only the scanned prefix; apparent unique matches may become ambiguous later.",
                        "A non-unknown raw 375 value is recorded authority evidence, not validated current author identity.",
                    ],
                    "unknown_value_tokens_for_counts_only": sorted(UNKNOWN_VALUES),
                }
                if cohort_path:
                    artifact = export_cohort_labels(
                        cohort_path, output / "item_labels.json", matches_path=output / "matches.jsonl",
                        reviewed_mapping_path=reviewed_mapping_path, audit_complete=not prefix_reached,
                    )
                    summary["cohort"] = {
                        "cohort_fingerprint": artifact["cohort_fingerprint"],
                        "linkage_reviewed": artifact["linkage_reviewed"],
                        "coverage": artifact["coverage"],
                        "item_labels_file": "item_labels.json",
                    }
            finally:
                connection.close()
        # Detect input replacement/size changes during the scan. A default
        # fingerprint is metadata only, never misleadingly called a full hash.
        for source in sources.values():
            after = Path(source["path"]).stat()
            if (after.st_size, after.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
                raise AuditInputError(f"Input changed during audit: {source['path']}")
        summary["file_sha256"] = {"matches.jsonl": file_sha256(output / "matches.jsonl")}
        if (output / "item_labels.json").exists():
            summary["file_sha256"]["item_labels.json"] = file_sha256(output / "item_labels.json")
        _atomic_json(output / "summary.json", summary)
        _atomic_json(output / "status.json", {"schema": SCHEMA, "state": summary["state"], "started_at_utc": started, "finished_at_utc": summary["finished_at_utc"], "summary_sha256": file_sha256(output / "summary.json")})
        return summary
    except Exception as exc:
        _atomic_json(output / "status.json", {
            "schema": SCHEMA, "state": "failed", "started_at_utc": started,
            "finished_at_utc": _utc_now(), "error_type": type(exc).__name__, "error": str(exc),
            "coverage_established": False,
        })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authors", required=True, type=Path, help="Local Goodreads authors JSON Lines/array, optionally gzip compressed")
    parser.add_argument("--viaf", required=True, type=Path, help="Local MARCXML collection or record lines, optionally gzip compressed; ISO2709 is not supported")
    parser.add_argument("--output", required=True, type=Path, help="Fresh audit directory; existing directories are never overwritten")
    parser.add_argument("--author-ids", type=Path, help="Optional one-column author_id CSV/text selection; missing IDs remain reported")
    parser.add_argument("--max-records", type=int, help="Stop at this VIAF prefix length; reaching the bound always marks the audit partial")
    parser.add_argument("--hash-inputs", action="store_true", help="Compute full SHA256 of stored source files with a separate complete read, including for a partial audit")
    parser.add_argument("--cohort", type=Path, help="Optional frozen goodreads-cohort-v1; export all books with unknown or explicitly reviewed evaluation labels")
    parser.add_argument("--export-labels", type=Path, metavar="REVIEWED_CSV", help="Explicit reviewed author_id/VIAF/gender mapping with provenance; requires --cohort. Name matches alone never become reviewed labels")
    args = parser.parse_args(argv)
    try:
        summary = audit(args.authors, args.viaf, args.output, author_ids_path=args.author_ids, max_records=args.max_records, hash_inputs=args.hash_inputs, cohort_path=args.cohort, reviewed_mapping_path=args.export_labels)
    except (AuditInputError, OSError, EOFError, ValueError, ET.ParseError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")
    print(json.dumps({"state": summary["state"], "output": str(args.output.resolve()), "records_seen": summary["scan"]["counts"]["records_seen"], "target_authors": summary["targets"]["target_authors"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
