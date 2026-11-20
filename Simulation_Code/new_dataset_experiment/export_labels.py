"""Export reviewed primary-author labels from a completed offline VIAF audit.

Only the small audit artifacts, frozen cohort and reviewed mapping are read.
The original Goodreads-author and VIAF source files are never rescanned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from .viaf_audit import AuditInputError, SCHEMA, export_cohort_labels, file_sha256


def _checksum(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def export_from_audit(
    audit_directory: str | Path, cohort_path: str | Path,
    reviewed_path: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    """Verify completed audit evidence, then create one fresh label artifact."""
    root = Path(audit_directory).resolve()
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    summary_path, status_path = root / "summary.json", root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if status.get("schema") != SCHEMA or summary.get("schema") != SCHEMA:
        raise AuditInputError("Unsupported VIAF name-coverage audit schema")
    if status.get("state") != "complete" or summary.get("state") != "complete":
        raise AuditInputError("Reviewed label export requires a complete audit, not a partial/failed scan")
    if summary.get("scan", {}).get("viaf_eof_reached") is not True or summary.get("coverage", {}).get("full_scan_coverage_established") is not True:
        raise AuditInputError("Audit does not establish a complete VIAF scan")
    expected_summary = status.get("summary_sha256")
    if not _checksum(expected_summary) or file_sha256(summary_path) != expected_summary:
        raise AuditInputError("Audit summary checksum is missing or does not match status.json")
    if any(status.get(key) != summary.get(key) for key in ("started_at_utc", "finished_at_utc")):
        raise AuditInputError("Audit status and summary describe different scan lifetimes")
    checksums = summary.get("file_sha256", {})
    if "matches.jsonl" not in checksums:
        raise AuditInputError("Audit has no checksummed matches.jsonl evidence; regenerate the audit with the current tool")
    for relative, expected in checksums.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not _checksum(expected) or file_sha256(path) != expected:
            raise AuditInputError(f"Audit artifact checksum mismatch or invalid path: {relative}")
    sources = summary.get("sources", {})
    for source_name in ("authors", "viaf"):
        source = sources.get(source_name, {})
        if not isinstance(source.get("path"), str) or not source["path"] or isinstance(source.get("size_bytes"), bool) or not isinstance(source.get("size_bytes"), int) or source["size_bytes"] < 0:
            raise AuditInputError(f"Audit lacks valid {source_name} snapshot provenance")
        scope = source.get("hash_scope")
        if scope == "entire_file_as_stored":
            if not _checksum(source.get("sha256")):
                raise AuditInputError(f"Audit has invalid {source_name} source fingerprint provenance")
        elif scope != "not_computed" or source.get("sha256") is not None:
            raise AuditInputError(f"Audit has unsupported {source_name} source fingerprint provenance")
    provenance = {
        "audit_directory": str(root), "audit_schema": SCHEMA,
        "summary_sha256": expected_summary, "matches_sha256": checksums["matches.jsonl"],
        "audit_finished_at_utc": summary["finished_at_utc"], "source_snapshots": sources,
        "verification_scope": "Audit summary and derivative artifact checksums verified; original large source files are not reread",
    }
    return export_cohort_labels(
        cohort_path, output_path, matches_path=root / "matches.jsonl",
        reviewed_mapping_path=reviewed_path, audit_complete=True,
        audit_provenance=provenance,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path, help="Existing complete audit directory containing summary/status and checksummed matches")
    parser.add_argument("--cohort", required=True, type=Path, help="Frozen Goodreads cohort JSON")
    parser.add_argument("--reviewed", required=True, type=Path, help="Explicit author_id/VIAF/gender review CSV with reviewer/date/source/note")
    parser.add_argument("--output", required=True, type=Path, help="Fresh item-label JSON path; never overwrite an existing file")
    args = parser.parse_args(argv)
    try:
        labels = export_from_audit(args.audit, args.cohort, args.reviewed, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Label export failed: {exc}\n")
    print(json.dumps({"output": str(args.output.resolve()), "label_semantics_version": labels["label_semantics_version"], "linkage_reviewed": labels["linkage_reviewed"], "coverage": labels["coverage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
