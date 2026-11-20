"""Offline, women-targeted mitigation of recorded primary-author representation.

This extension retains the earlier provisional VIAF annotations verbatim. It
does not identify authors, infer gender, change catalogue membership, or set a
desired gender proportion. Only an explicitly selected intervention uses labels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


EXPANDED_POLICY = "viaf-full-name-known-assertions-consensus-v1"
LABEL_SEMANTICS = "goodreads-primary-literary-author-recorded-gender-v1"
KNOWN_GENDERS = frozenset({"female", "male", "nonbinary", "other"})
GENDERS = KNOWN_GENDERS | {"unknown", "ambiguous"}
INTERVENTIONS = frozenset({"control", "in_processing", "post_processing"})


@dataclass(frozen=True)
class GenderMitigationSpec:
    """Weights affect positive targets at later L2 refits, never initial fitting.

    Promotion is an offset to original slate ranks, followed by stable sorting;
    it is not a promise that each eligible item moves exactly that many slots.
    """

    intervention: str = "control"
    women_positive_weight: float = 1.25
    women_rank_promotion: int = 5

    def __post_init__(self) -> None:
        if self.intervention not in INTERVENTIONS:
            raise ValueError("unsupported gender mitigation intervention")
        weight = self.women_positive_weight
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("women_positive_weight must be finite and positive")
        promotion = self.women_rank_promotion
        if isinstance(promotion, bool) or not isinstance(promotion, int) or promotion < 0:
            raise ValueError("women_rank_promotion must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenderLabels:
    """Validated immutable labels bound to the original frozen catalogue."""

    labels_by_item: Mapping[int, str]
    women_item_ids: frozenset[int]
    labels_sha256: str
    cohort_fingerprint: str
    policy_id: str
    payload_sha256: str
    coverage: Mapping[str, Any]


def _item_id(value: Any, *, padding: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if padding else 1):
        raise ValueError("item IDs must be positive integers (zero is allowed only for training padding)")
    return int(value)


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def _payload_sha256(rows: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def women_item_ids(labels: GenderLabels | Mapping[int, str]) -> frozenset[int]:
    """Select only the explicit female label; unresolved categories stay distinct."""
    if isinstance(labels, GenderLabels):
        return labels.women_item_ids
    for item, gender in labels.items():
        _item_id(item)
        if gender not in GENDERS:
            raise ValueError("unsupported recorded primary-author gender")
    return frozenset(item for item, gender in labels.items() if gender == "female")


def positive_target_weights(
    item_ids: Iterable[int], women_ids: Iterable[int], spec: GenderMitigationSpec,
) -> tuple[float, ...]:
    """Return per-positive-target weights; negatives always retain their own loss.

    Padding receives weight one and must still be masked by the trainer. Control
    and post-processing conditions receive unit weights for every target.
    """
    women = frozenset(_item_id(item) for item in women_ids)
    multiplier = float(spec.women_positive_weight) if spec.intervention == "in_processing" else 1.0
    return tuple(multiplier if _item_id(item, padding=True) in women else 1.0 for item in item_ids)


def stable_promote_slate(
    slate: Sequence[int], women_ids: Iterable[int], *, promotion: int = 5,
) -> tuple[int, ...]:
    """Reorder an already selected slate, retaining every item exactly once.

    Original zero-based rank minus ``promotion`` is the sorting key for women-
    labelled items; all other keys are original rank. Original rank resolves
    equal adjusted ranks, preserving order within both groups. Apply after
    top-K selection: this function must not promote new catalogue items into it.
    """
    if isinstance(promotion, bool) or not isinstance(promotion, int) or promotion < 0:
        raise ValueError("promotion must be a non-negative integer")
    original = tuple(_item_id(item) for item in slate)
    if len(set(original)) != len(original):
        raise ValueError("slate must contain unique items")
    women = frozenset(_item_id(item) for item in women_ids)
    return tuple(item for rank, item in sorted(enumerate(original),
        key=lambda pair: (pair[0] - promotion * (pair[1] in women), pair[0])))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_gender_labels(
    labels_path: Path | str, *, catalogue_item_ids: Iterable[int],
    cohort_fingerprint: str, histories: Mapping[Any, Sequence[int]] | None = None,
    expected_labels_sha256: str | None = None,
) -> GenderLabels:
    """Validate the earlier expanded-consensus artifact without altering it.

    ``cohort_fingerprint`` must be the exact original cohort file SHA256, not
    a prepared-input semantic fingerprint. The complete item-ID set must match.
    When initial histories are supplied, recompute their frozen coverage too.
    Source digest declarations are validated; the original authority scan is
    not repeated. A caller may pin the whole label file with its expected hash.
    The source artifact's evaluation-only declaration is preserved as provenance;
    this new experiment separately authorises the two narrow mitigation uses.
    """
    expected_cohort = _sha(cohort_fingerprint, "cohort_fingerprint")
    catalogue = tuple(_item_id(item) for item in catalogue_item_ids)
    if not catalogue or len(set(catalogue)) != len(catalogue):
        raise ValueError("catalogue IDs must be nonempty and unique")
    raw = Path(labels_path).read_bytes()
    file_hash = hashlib.sha256(raw).hexdigest()
    if expected_labels_sha256 is not None and file_hash != _sha(expected_labels_sha256, "expected_labels_sha256"):
        raise ValueError("label file SHA256 mismatch")
    artifact = json.loads(raw, object_pairs_hook=_unique_json_object)
    if artifact.get("schema_version") != "goodreads-item-labels-v1" or artifact.get("label_semantics_version") != LABEL_SEMANTICS:
        raise ValueError("unsupported Goodreads primary-author label schema")
    if artifact.get("cohort_fingerprint_scope") != "cohort_file_sha256" or artifact.get("cohort_fingerprint") != expected_cohort:
        raise ValueError("label cohort binding mismatch")
    if artifact.get("authority_scan_complete") is not True or artifact.get("linkage_reviewed") is not False:
        raise ValueError("labels require a complete audit and explicit unreviewed identity status")
    policy = artifact.get("annotation_policy", {})
    if policy.get("policy_id") != EXPANDED_POLICY:
        raise ValueError("gender mitigation requires the agreed expanded-consensus policy")
    expected_policy = {
        "evaluation_only": True, "identity_links_verified": False, "unknown_books_retained": True,
        "name_title_filter": "matched_full_name_heading_without_subfield_t", "unknown_assertions": "neutral",
        "conflicting_known_assertions": "ambiguous", "temporal_claims": "unresolved",
    }
    if any(policy.get(key) != value for key, value in expected_policy.items()):
        raise ValueError("expanded-consensus annotation assumptions differ")
    for key in ("matches_sha256", "audit_summary_sha256", "author_evidence_sha256"):
        _sha(policy.get(key), key)
    source_cohort = artifact.get("audit_provenance", {}).get("original_sources", {}).get("cohort", {}).get("sha256")
    if source_cohort != expected_cohort:
        raise ValueError("label audit provenance cohort binding mismatch")
    rows = artifact.get("labels")
    if not isinstance(rows, dict) or policy.get("item_labels_payload_sha256") != _payload_sha256(rows):
        raise ValueError("item-label payload digest mismatch")
    labels: dict[int, str] = {}
    for key, row in rows.items():
        try:
            item = _item_id(int(key))
        except (TypeError, ValueError) as error:
            raise ValueError("invalid catalogue item ID in label artifact") from error
        if str(item) != key or item in labels:
            raise ValueError("label item IDs must use unique canonical integer strings")
        gender = row.get("gender")
        if gender not in GENDERS or row.get("classification_scope") != "primary_literary_author":
            raise ValueError("invalid recorded primary-author gender or classification scope")
        if row.get("identity_verified") is not False or row.get("status") != ("unknown" if gender == "unknown" else "provisional"):
            raise ValueError("labels must preserve provisional and unresolved identity status")
        if gender in KNOWN_GENDERS and (row.get("primary_author_resolution") != "single_recorded_primary_author"
                or len(row.get("primary_author_ids", [])) != 1 or not row.get("candidate_ids")):
            raise ValueError("known gender requires one primary author and candidate provenance")
        labels[item] = gender
    if set(labels) != set(catalogue):
        raise ValueError("label catalogue binding mismatch")
    coverage = artifact.get("coverage", {})
    counts = Counter(labels.values())
    actual: dict[str, Any] = {
        "book_count": len(labels), "known_book_count": sum(gender in KNOWN_GENDERS for gender in labels.values()),
        "gender_counts": dict(counts),
    }
    actual["known_book_share"] = actual["known_book_count"] / len(labels)
    if histories is not None:
        training = Counter(_item_id(item) for history in histories.values() for item in history)
        if not training or not set(training).issubset(labels):
            raise ValueError("initial training histories are empty or reference an unlabelled item")
        actual["training_interaction_count"] = sum(training.values())
        actual["known_training_interaction_count"] = sum(n for item, n in training.items() if labels[item] in KNOWN_GENDERS)
        actual["known_training_interaction_share"] = actual["known_training_interaction_count"] / actual["training_interaction_count"]
        actual["gender_training_interaction_counts"] = {gender: sum(n for item, n in training.items() if labels[item] == gender) for gender in counts}
    for key, value in actual.items():
        declared = coverage.get(key)
        if isinstance(value, dict):
            if declared != value:
                raise ValueError(f"label coverage mismatch: {key}")
        elif isinstance(declared, bool) or not isinstance(declared, (int, float)) or not math.isfinite(declared) or not math.isclose(declared, value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"label coverage mismatch: {key}")
        if key.endswith("_count") and policy.get("frozen_coverage_counts", {}).get(key) != value:
            raise ValueError(f"frozen label coverage mismatch: {key}")
    return GenderLabels(MappingProxyType(labels), women_item_ids(labels), file_hash,
        expected_cohort, EXPANDED_POLICY, policy["item_labels_payload_sha256"], MappingProxyType(coverage))
