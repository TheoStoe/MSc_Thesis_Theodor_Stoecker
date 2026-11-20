"""Versioned, evaluation-only labels from a complete cached VIAF name audit.

This module neither identifies people conclusively nor infers gender from names.
Unknown assertions are neutral under the selected expanded rule; distinct known
assertions conflict. The original audit and frozen reader histories are immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .viaf_audit import (
    UNKNOWN_VALUES, _item_authors, _primary_authors, _read_label_cohort,
    normalized_name_keys,
)

POLICY = "viaf-full-name-known-assertions-consensus-v1"
STRICT_POLICY = "viaf-full-name-every-candidate-known-consensus-v1"
KNOWN = frozenset({"female", "male", "nonbinary", "other"})
SEMANTICS = "goodreads-primary-literary-author-recorded-gender-v1"
FROZEN_COHORT = "97e8ed3dfc09f323877c1625fab8a2d542c17d14a2021e7f3490d2eca684c90b"
FROZEN_COUNTS = {
    POLICY: (26749, 19555, 42279, 31992),
    STRICT_POLICY: (26749, 9545, 42279, 14718),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(rows: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _viaf_id(value: Any) -> str:
    return str(value).removeprefix("viaf")


def resolve_author(row: dict[str, Any]) -> dict[str, Any]:
    """Keep exact normalized matching name headings without name-title subfield t.

    Raw 375 claims and heading qualifiers are preserved. Date-qualified claims,
    unsupported terms and contradictory known values remain unresolved.
    """
    keys = normalized_name_keys(row.get("author_name") or "")
    candidates: dict[str, dict[str, Any]] = {}
    removed = 0
    for candidate in row.get("candidates", []):
        matching = [heading for record in candidate.get("record_evidence", [])
                    for heading in record.get("personal_name_headings", [])
                    if keys & normalized_name_keys(heading.get("name", ""))
                    and not any(field["code"] == "t" for field in heading.get("subfields", []))]
        if not matching:
            removed += 1
            continue
        cid = _viaf_id(candidate["viaf_id"])
        target = candidates.setdefault(cid, {"viaf_id": cid, "known_values": set(),
            "unknown_assertion_count": 0, "has_temporal_375_qualifiers": False,
            "matching_name_headings": [], "gender_375_claims": [], "record_ids": []})
        target["matching_name_headings"].extend(matching)
        for record in candidate["record_evidence"]:
            if _viaf_id(record["viaf_id"]) != _viaf_id(record["framing_id"]):
                raise ValueError("cached candidate framing and record IDs disagree")
            if _viaf_id(record["viaf_id"]) != cid:
                raise ValueError("cached candidate and evidence IDs disagree")
            target["record_ids"].append({"viaf_id": record["viaf_id"], "framing_id": record["framing_id"]})
            for claim in record.get("gender_375_claims", []):
                target["gender_375_claims"].append(claim)
                target["has_temporal_375_qualifiers"] |= bool(claim.get("start_periods") or claim.get("end_periods"))
                for raw in claim.get("raw_values", []):
                    value = raw.casefold().strip()
                    if not value or value in UNKNOWN_VALUES:
                        target["unknown_assertion_count"] += 1
                    else:
                        target["known_values"].add(value)
    values = set().union(*(value["known_values"] for value in candidates.values())) if candidates else set()
    temporal = any(value["has_temporal_375_qualifiers"] for value in candidates.values())
    unsupported = bool(values - KNOWN)
    if len(values) > 1:
        gender, reason = "ambiguous", "conflicting_known_375_assertions"
    elif not values:
        gender = "unknown"
        reason = "missing_author_name" if not keys else "no_name_candidate" if not candidates else "unknown_375_only"
    elif temporal or unsupported:
        gender, reason = "unknown", "temporal_375_unresolved" if temporal else "unsupported_recorded_375_term"
    else:
        gender = next(iter(values))
        reason = "one_known_gender_among_retained_candidates"
    every_candidate_known = bool(candidates) and all(len(value["known_values"]) == 1 for value in candidates.values())
    strict = gender if gender in KNOWN and every_candidate_known else "ambiguous" if gender == "ambiguous" else "unknown"
    for value in candidates.values():
        value["known_values"] = sorted(value["known_values"])
    return {"author_id": str(row["author_id"]), "author_name": row.get("author_name"),
            "gender": gender, "strict_gender": strict, "reason": reason,
            "candidate_ids": sorted(candidates), "candidate_count": len(candidates),
            "unknown_only_candidate_count": sum(not value["known_values"] for value in candidates.values()),
            "has_temporal_375_qualifiers": temporal, "identity_verified": False,
            "name_title_candidates_removed": removed, "candidates": list(candidates.values())}


def label_items(cohort: dict[str, Any], authors: dict[str, dict[str, Any]], *, strict: bool = False) -> dict[str, Any]:
    labels = {}
    for item in cohort["items"]:
        item_id = str(item["item_id"])
        if item_id in labels:
            raise ValueError("duplicate catalogue item")
        literary = _item_authors(item)
        primary, resolution = _primary_authors(item, literary)
        author = authors.get(primary[0]) if resolution == "single_recorded_primary_author" else None
        if resolution == "single_recorded_primary_author" and author is None:
            raise ValueError("cached audit does not cover a primary author")
        gender = author["strict_gender" if strict else "gender"] if author else "ambiguous" if primary else "unknown"
        reason = author["reason"] if author else resolution
        if strict and author and author["gender"] in KNOWN and gender == "unknown":
            reason = "some_retained_candidates_lack_known_gender"
        labels[item_id] = {
            "gender": gender, "status": "provisional" if gender != "unknown" else "unknown",
            "reason": reason, "classification_scope": "primary_literary_author",
            "primary_author_ids": primary, "primary_author_resolution": resolution,
            "author_ids": literary, "identity_verified": False,
            "candidate_ids": author["candidate_ids"] if author else [],
            "candidate_count": author["candidate_count"] if author else 0,
            "unknown_only_candidate_count": author["unknown_only_candidate_count"] if author else 0,
        }
    return labels


def coverage(labels: dict[str, Any], histories: dict[Any, Any]) -> dict[str, Any]:
    known_ids = {int(key) for key, row in labels.items() if row["gender"] in KNOWN}
    counts = Counter(int(item) for history in histories.values() for item in history)
    if not labels or not counts:
        raise ValueError("coverage requires nonempty catalogue and training histories")
    if not set(counts).issubset({int(key) for key in labels}):
        raise ValueError("training interaction references an item missing from author labels")
    total, known = sum(counts.values()), sum(count for item, count in counts.items() if item in known_ids)
    return {"book_count": len(labels), "known_book_count": len(known_ids),
            "training_interaction_count": total, "known_training_interaction_count": known,
            "known_book_share": len(known_ids) / len(labels), "known_training_interaction_share": known / total,
            "gender_counts": dict(Counter(row["gender"] for row in labels.values())),
            "gender_training_interaction_counts": dict(Counter({gender: sum(count for item, count in counts.items()
                if labels[str(item)]["gender"] == gender) for gender in {row["gender"] for row in labels.values()}}))}


def validate_policy_labels(artifact: dict[str, Any], histories: dict[Any, Any]) -> dict[str, float]:
    """Accept explicit automatic-consensus policies without pretending manual review.

    Frozen source/dataset identifiers, item payload digest, recomputed coverage
    and exact expected cohort counts replace the legacy manual-review 80% gate.
    """
    policy = artifact.get("annotation_policy", {})
    policy_id = policy.get("policy_id")
    if policy_id not in {POLICY, STRICT_POLICY}:
        raise ValueError("unsupported automatic annotation policy")
    if artifact.get("authority_scan_complete") is not True:
        raise ValueError("a complete authority scan is required")
    if artifact.get("linkage_reviewed") is not False or policy.get("identity_links_verified") is not False:
        raise ValueError("automatic annotation policy must retain unreviewed identity status")
    if policy.get("evaluation_only") is not True or policy.get("unknown_books_retained") is not True:
        raise ValueError("automatic annotations require evaluation-only labels and unknown retention")
    if policy.get("name_title_filter") != "matched_full_name_heading_without_subfield_t":
        raise ValueError("unsupported candidate-heading filter")
    for source_key in ("matches_sha256", "audit_summary_sha256", "author_evidence_sha256"):
        value = policy.get(source_key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("automatic policy requires complete source hashes")
    rows = artifact.get("labels", {})
    if policy.get("item_labels_payload_sha256") != payload_sha256(rows):
        raise ValueError("automatic item-label payload digest mismatch")
    for row in rows.values():
        gender = row.get("gender")
        if gender not in KNOWN | {"unknown", "ambiguous"}:
            raise ValueError("invalid recorded primary-author gender")
        if row.get("identity_verified") is not False or row.get("status") != ("unknown" if gender == "unknown" else "provisional"):
            raise ValueError("automatic item labels must preserve explicit provisional status")
        if gender in KNOWN and (row.get("primary_author_resolution") != "single_recorded_primary_author"
                               or len(row.get("primary_author_ids", [])) != 1 or not row.get("candidate_ids")):
            raise ValueError("known automatic gender requires one recorded primary author and candidate provenance")
    actual = coverage(rows, histories)
    for key, value in actual.items():
        if key.endswith("_counts"):
            continue
        declared = artifact.get("coverage", {}).get(key)
        if isinstance(declared, bool) or not isinstance(declared, (int, float)) or not math.isfinite(declared) or not math.isclose(declared, value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"declared {key} disagrees with automatic labels and actual training histories")
    count_keys = ("book_count", "known_book_count", "training_interaction_count", "known_training_interaction_count")
    expected = policy.get("frozen_coverage_counts", {})
    if any(expected.get(key) != actual[key] for key in count_keys):
        raise ValueError("automatic labels disagree with frozen coverage contract")
    if artifact.get("cohort_fingerprint") == FROZEN_COHORT and tuple(actual[key] for key in count_keys) != FROZEN_COUNTS[policy_id]:
        raise ValueError("automatic labels disagree with accepted population-16040 counts")
    return {key: actual[key] for key in ("known_book_share", "known_training_interaction_share")}


def export_cached_labels(cohort_path: Path, matches_path: Path, summary_path: Path, output_root: Path) -> dict[str, Any]:
    """Stream cached matches once, writing compact author evidence separately."""
    if output_root.exists():
        raise FileExistsError(output_root)
    cohort_bytes, cohort = _read_label_cohort(cohort_path)
    cohort_hash = hashlib.sha256(cohort_bytes).hexdigest()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("state") != "complete" or summary.get("scan", {}).get("viaf_eof_reached") is not True:
        raise ValueError("cached source audit must have reached VIAF EOF")
    if summary.get("sources", {}).get("cohort", {}).get("sha256") != cohort_hash:
        raise ValueError("source audit belongs to a different frozen cohort")
    target_ids = {aid for item in cohort["items"] for aid in _item_authors(item)}
    authors = {}
    digest = hashlib.sha256()
    with matches_path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            raw = json.loads(line)
            aid = str(raw["author_id"])
            if aid in authors:
                raise ValueError("duplicate author in cached matches")
            authors[aid] = resolve_author(raw)
    if digest.hexdigest() != summary.get("file_sha256", {}).get("matches.jsonl"):
        raise ValueError("cached matches checksum mismatch")
    if set(authors) != target_ids:
        raise ValueError("cached audit target authors differ from cohort literary authors")
    histories = {user["user_id"]: [event["item_id"] for event in user["train"]] for user in cohort["users"]}
    output_root.mkdir(parents=True)
    evidence_path = output_root / "authors.jsonl"
    with evidence_path.open("x", encoding="utf-8", newline="\n") as stream:
        for aid in sorted(authors):
            stream.write(json.dumps(authors[aid], ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    evidence_hash = sha256(evidence_path)
    outputs = {}
    for strict, name, policy_id in ((False, "item_labels.json", POLICY), (True, "item_labels_strict.json", STRICT_POLICY)):
        rows = label_items(cohort, authors, strict=strict)
        cov = coverage(rows, histories)
        artifact = {
            "schema_version": "goodreads-item-labels-v1", "schema": "goodreads-item-labels-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "label_semantics_version": SEMANTICS,
            "cohort_fingerprint": cohort_hash, "cohort_fingerprint_scope": "cohort_file_sha256",
            "linkage_reviewed": False, "authority_scan_complete": True,
            "labels": rows, "coverage": cov,
            "annotation_policy": {"policy_id": policy_id, "evaluation_only": True,
                "identity_links_verified": False, "unknown_books_retained": True,
                "name_title_filter": "matched_full_name_heading_without_subfield_t",
                "unknown_assertions": "neutral" if not strict else "each_candidate_requires_known_value",
                "conflicting_known_assertions": "ambiguous", "temporal_claims": "unresolved",
                "matches_sha256": digest.hexdigest(), "audit_summary_sha256": sha256(summary_path),
                "author_evidence_sha256": evidence_hash, "author_evidence_file": "authors.jsonl",
                "item_labels_payload_sha256": payload_sha256(rows),
                "frozen_coverage_counts": {key: cov[key] for key in ("book_count", "known_book_count", "training_interaction_count", "known_training_interaction_count")}},
            "audit_provenance": {"matches_path": str(matches_path.resolve()), "summary_path": str(summary_path.resolve()),
                "original_sources": summary["sources"], "records_seen": summary["scan"].get("counts", {}).get("records_seen")},
            "method": {"evaluation_only": True, "primary_metric": "single recorded primary literary author; competing primary IDs remain ambiguous",
                "identity_inference_from_names": False, "gender_inference_from_names": False,
                "identity_assumption": "exact normalized full-name candidates can be namesakes; agreement is attribute evidence, not verified identity",
                "unknown_retention": "all original cohort books and training interactions retained"},
        }
        validate_policy_labels(artifact, histories)
        path = output_root / name
        path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        outputs[name] = {"sha256": sha256(path), "coverage": cov}
    if sha256(cohort_path) != cohort_hash:
        raise ValueError("frozen cohort changed during export")
    manifest = {"schema_version": "goodreads-automatic-gender-label-export-v1", "status": "complete",
                "cohort_sha256": cohort_hash, "author_count": len(authors),
                "author_evidence_sha256": evidence_hash, "files": outputs,
                "cohort_or_histories_modified": False, "raw_viaf_rescanned": False}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_cached_labels(args.cohort, args.matches, args.summary, args.output), indent=2))


if __name__ == "__main__":
    main()
