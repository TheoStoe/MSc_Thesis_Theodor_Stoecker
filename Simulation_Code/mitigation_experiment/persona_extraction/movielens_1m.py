"""Strict MovieLens 1M demographic input for profile extraction.

Population sampling owns the choice of users.  This module only joins those
already-selected runtime users to the authoritative source IDs in
``users.dat`` and renders the age, gender, and occupation text used by the
upstream Lusifer MovieLens example.  ZIP codes are parsed only as the fifth
source field and are never retained, fingerprinted, or prompted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ..contracts import UserId
from ..utility.fingerprints import fingerprint_semantic_state


MOVIELENS_USER_METADATA_SCHEMA = "movielens_1m_user_metadata_v1"
MOVIELENS_USER_TEXT_BUILDER_VERSION = "movielens_1m_user_text_v1"

GENDER_LABELS: Mapping[str, str] = MappingProxyType(
    {"F": "Female", "M": "Male"}
)
AGE_LABELS: Mapping[int, str] = MappingProxyType(
    {
        1: "Under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+",
    }
)
OCCUPATION_LABELS: Mapping[int, str] = MappingProxyType(
    {
        0: "other or not specified",
        1: "academic/educator",
        2: "artist",
        3: "clerical/admin",
        4: "college/grad student",
        5: "customer service",
        6: "doctor/health care",
        7: "executive/managerial",
        8: "farmer",
        9: "homemaker",
        10: "K-12 student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales/marketing",
        15: "scientist",
        16: "self-employed",
        17: "technician/engineer",
        18: "tradesman/craftsman",
        19: "unemployed",
        20: "writer",
    }
)


@dataclass(frozen=True)
class MovieLensUserAttributes:
    """Normalized non-ZIP attributes for one original MovieLens user."""

    source_user_id: str
    gender_code: str
    gender: str
    age_code: int
    age_range: str
    occupation_code: int
    occupation: str
    metadata_fingerprint: str


@dataclass(frozen=True)
class MatchedMovieLensUserText:
    """One explicit runtime/source-ID join ready for profile extraction."""

    user_id: UserId
    source_user_id: str
    attributes: MovieLensUserAttributes
    user_text_builder_version: str
    user_text: str
    user_text_fingerprint: str


def parse_movielens_1m_users(
    path: str | Path,
) -> Mapping[str, MovieLensUserAttributes]:
    """Parse ``UserID::Gender::Age::Occupation::Zip-code`` strictly.

    The ZIP value must exist so malformed rows are caught, but its value is
    deliberately discarded before any returned object or fingerprint exists.
    """

    users_path = Path(path)
    if not users_path.is_file():
        raise FileNotFoundError(f"MovieLens users file not found: {users_path}")

    parsed: dict[str, MovieLensUserAttributes] = {}
    with users_path.open("r", encoding="latin-1", newline="") as stream:
        for row_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            prefix = f"{users_path}: row {row_number}: "
            if not line:
                raise ValueError(prefix + "row cannot be empty")
            fields = line.split("::")
            if len(fields) != 5:
                raise ValueError(prefix + "expected 5 fields separated by ::")

            raw_user_id, gender_code, raw_age, raw_occupation, raw_zip = fields
            source_user_id = _canonical_source_user_id(
                raw_user_id,
                field_name=prefix + "user ID",
            )
            if source_user_id in parsed:
                raise ValueError(prefix + f"duplicate user ID {source_user_id}")
            if not raw_zip.strip():
                raise ValueError(prefix + "ZIP code cannot be blank")
            try:
                age_code = int(raw_age)
                occupation_code = int(raw_occupation)
            except ValueError as exc:
                raise ValueError(
                    prefix + "age and occupation codes must be integers"
                ) from exc
            try:
                gender = GENDER_LABELS[gender_code]
            except KeyError as exc:
                raise ValueError(
                    prefix + f"unknown gender code {gender_code!r}"
                ) from exc
            try:
                age_range = AGE_LABELS[age_code]
            except KeyError as exc:
                raise ValueError(prefix + f"unknown age code {age_code}") from exc
            try:
                occupation = OCCUPATION_LABELS[occupation_code]
            except KeyError as exc:
                raise ValueError(
                    prefix + f"unknown occupation code {occupation_code}"
                ) from exc

            semantic_attributes = {
                "source_user_id": source_user_id,
                "gender_code": gender_code,
                "gender": gender,
                "age_code": age_code,
                "age_range": age_range,
                "occupation_code": occupation_code,
                "occupation": occupation,
            }
            parsed[source_user_id] = MovieLensUserAttributes(
                **semantic_attributes,
                metadata_fingerprint=fingerprint_semantic_state(
                    semantic_attributes,
                    MOVIELENS_USER_METADATA_SCHEMA,
                ),
            )

    if not parsed:
        raise ValueError(f"{users_path}: file contains no users")
    return MappingProxyType(
        {source_id: parsed[source_id] for source_id in sorted(parsed, key=int)}
    )


def render_movielens_1m_user_text(
    attributes: MovieLensUserAttributes,
) -> str:
    """Render the same three demographic fields used by Lusifer upstream."""

    if not isinstance(attributes, MovieLensUserAttributes):
        raise TypeError("attributes must be MovieLensUserAttributes")
    return (
        f"Age: {attributes.age_range}\n"
        f"Gender: {attributes.gender}\n"
        f"Occupation: {attributes.occupation}"
    )


def match_movielens_1m_user_texts(
    runtime_to_source_user_id: Mapping[UserId, str | int],
    attributes_by_source_user_id: Mapping[str, MovieLensUserAttributes],
) -> Mapping[UserId, MatchedMovieLensUserText]:
    """Join a pre-sampled population to demographics by explicit source ID.

    This function never samples and never substitutes a missing user.  Passing
    only active and reserve users therefore keeps background users outside the
    profile-extraction cost boundary.
    """

    if not isinstance(runtime_to_source_user_id, Mapping):
        raise TypeError("runtime_to_source_user_id must be a mapping")
    if not isinstance(attributes_by_source_user_id, Mapping):
        raise TypeError("attributes_by_source_user_id must be a mapping")

    normalized: dict[UserId, str] = {}
    for user_id, source_value in runtime_to_source_user_id.items():
        if isinstance(user_id, bool) or not isinstance(user_id, int):
            raise TypeError("runtime user IDs must be integers")
        if user_id <= 0:
            raise ValueError("runtime user IDs must be positive")
        normalized[user_id] = _canonical_source_user_id(
            source_value,
            field_name=f"source user ID for runtime user {user_id}",
        )
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("sampled population contains duplicate source user IDs")

    missing = sorted(
        set(normalized.values()).difference(attributes_by_source_user_id),
        key=int,
    )
    if missing:
        raise ValueError(
            "sampled source user IDs are missing from users.dat: "
            + ", ".join(missing)
        )

    matched: dict[UserId, MatchedMovieLensUserText] = {}
    for user_id in sorted(normalized):
        source_user_id = normalized[user_id]
        attributes = attributes_by_source_user_id[source_user_id]
        if not isinstance(attributes, MovieLensUserAttributes):
            raise TypeError(
                "attributes_by_source_user_id must contain "
                "MovieLensUserAttributes"
            )
        if attributes.source_user_id != source_user_id:
            raise ValueError(
                "attribute mapping key does not match its source user ID: "
                f"{source_user_id} != {attributes.source_user_id}"
            )
        user_text = render_movielens_1m_user_text(attributes)
        text_fingerprint = fingerprint_semantic_state(
            {
                "builder_version": MOVIELENS_USER_TEXT_BUILDER_VERSION,
                "metadata_fingerprint": attributes.metadata_fingerprint,
                "user_text": user_text,
            },
            MOVIELENS_USER_TEXT_BUILDER_VERSION,
        )
        matched[user_id] = MatchedMovieLensUserText(
            user_id=user_id,
            source_user_id=source_user_id,
            attributes=attributes,
            user_text_builder_version=MOVIELENS_USER_TEXT_BUILDER_VERSION,
            user_text=user_text,
            user_text_fingerprint=text_fingerprint,
        )
    return MappingProxyType(matched)


def _canonical_source_user_id(value: str | int, *, field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{field_name} must be an integer or decimal string")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{field_name} must be positive")
        return str(value)
    if not value or value != value.strip() or not value.isdecimal():
        raise ValueError(f"{field_name} must be a canonical decimal string")
    parsed = int(value)
    if parsed <= 0 or str(parsed) != value:
        raise ValueError(f"{field_name} must be a canonical positive ID")
    return value
