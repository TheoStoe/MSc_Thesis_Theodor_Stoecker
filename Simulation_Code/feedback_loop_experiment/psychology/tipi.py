"""TIPI-anchored scoring and ranking for frozen MovieLens personas.

The prompt follows the three-step construction used by Zhang, Kang, and Guo:
construct a factual role, let the LLM enact that role while answering TIPI, and
add a Big-Five score to the role.  The local version starts
from an already frozen Lusifer ``ProfileArtifact`` and adapts the appendix
prompt to MovieLens demographics and movie preferences.  The LLM never scores
itself; reverse-keying and OCEAN conversion are pure functions in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..persona_extraction.base import ProfileArtifact
from ..utility.fingerprints import fingerprint_semantic_state
from ..utility.providers import JSONProvider
from ..utility.saving import save_json_artifact


SEARCH_PERSONALIZATION_PAPER_DOI = "10.1145/3701716.3717530"
SEARCH_PERSONALIZATION_PAPER_URL = (
    "https://doi.org/10.1145/3701716.3717530"
)
TIPI_INSTRUMENT_ID = "TIPI_en_original_2003"
TIPI_INSTRUMENT_SOURCE = (
    "https://gosling.psy.utexas.edu/scales-weve-developed/"
    "ten-item-personality-measure-tipi/"
)
TIPI_PROMPT_VERSION = "search_paper_movielens_tipi_prompt_v1"
TIPI_SCORING_VERSION = "tipi_official_reverse_key_mean_v1"
TIPI_NEUROTICISM_CONVERSION_VERSION = (
    "tipi_emotional_stability_to_neuroticism_8_minus_v1"
)
TIPI_ARTIFACT_SCHEMA = "tipi_profile_artifact_v1"
TIPI_RESPONSE_VECTOR_SCHEMA = "tipi_response_vector_v1"
TIPI_CACHE_FILENAME = "tipi_profiles.json"

TIPI_ITEM_TEXTS: tuple[str, ...] = (
    "Extraverted, enthusiastic.",
    "Critical, quarrelsome.",
    "Dependable, self-disciplined.",
    "Anxious, easily upset.",
    "Open to new experiences, complex.",
    "Reserved, quiet.",
    "Sympathetic, warm.",
    "Disorganized, careless.",
    "Calm, emotionally stable.",
    "Conventional, uncreative.",
)
TIPI_REVERSE_KEYED_ITEMS = (2, 4, 6, 8, 10)
TIPI_RESPONSE_SCALE: tuple[tuple[int, str], ...] = (
    (1, "Disagree strongly"),
    (2, "Disagree moderately"),
    (3, "Disagree a little"),
    (4, "Neither agree nor disagree"),
    (5, "Agree a little"),
    (6, "Agree moderately"),
    (7, "Agree strongly"),
)
TIPI_ORDERED_ITEM_TEXT_HASH = fingerprint_semantic_state(
    TIPI_ITEM_TEXTS,
    TIPI_INSTRUMENT_ID,
)

TIPI_SYSTEM_PROMPT = """
You are the user of a movie rating and recommendation platform.
Answer about yourself. Use only the supplied profile. Keep the profile's age range,
occupation, demographic information, and movie preferences unchanged. Do not
calculate personality traits, discuss scoring rules, or mention an experimental
hypothesis. Return only the requested structured response.
""".strip()

TIPI_USER_PROMPT_TEMPLATE = """
You are the user of a movie rating and recommendation platform.

<frozen_user_characteristics>
{user_characteristics}
</frozen_user_characteristics>

<frozen_movie_preference_profile>
{profile_text}
</frozen_movie_preference_profile>

Act as this exact user and answer the Ten-Item Personality Inventory (TIPI).
For every statement, choose the single integer that best represents how you as the user see yourself. 
Use the complete response scale below.

{response_scale}

Statements:
{items}

For each item, return its item number, integer score, and one concise sentence
explaining the role-play choice. Ground the explanation in the frozen profile
when evidence is available. When evidence is absent, make the smallest
role-consistent judgment needed to answer; do not invent a new age, occupation,
demographic fact, rating, or movie preference. Do not compute or name Big-Five
scores.
""".strip()

TIPI_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "responses": {
            "type": "array",
            "minItems": 10,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 7,
                    },
                    "rationale": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 320,
                    },
                },
                "required": ["item_id", "score", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["responses"],
    "additionalProperties": False,
}

TIPI_PROMPT_FINGERPRINT = fingerprint_semantic_state(
    {
        "paper_doi": SEARCH_PERSONALIZATION_PAPER_DOI,
        "system_prompt": TIPI_SYSTEM_PROMPT,
        "user_prompt_template": TIPI_USER_PROMPT_TEMPLATE,
        "response_scale": TIPI_RESPONSE_SCALE,
        "ordered_items": TIPI_ITEM_TEXTS,
        "response_schema": TIPI_RESPONSE_SCHEMA,
    },
    TIPI_PROMPT_VERSION,
)


@dataclass(frozen=True)
class TipiResponse:
    """One validated questionnaire response retained for inspection."""

    item_id: int
    score: int
    rationale: str

    def __post_init__(self) -> None:
        _validate_integer(self.item_id, name="item_id", minimum=1, maximum=10)
        _validate_integer(self.score, name="score", minimum=1, maximum=7)
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("rationale must be a non-blank string")
        object.__setattr__(self, "rationale", self.rationale.strip())


@dataclass(frozen=True)
class TipiScores:
    """Externally scored TIPI domains plus the explicit OCEAN conversion."""

    extraversion: float
    agreeableness: float
    conscientiousness: float
    emotional_stability: float
    openness: float
    neuroticism: float

    def __post_init__(self) -> None:
        for name in (
            "extraversion",
            "agreeableness",
            "conscientiousness",
            "emotional_stability",
            "openness",
            "neuroticism",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or not 1 <= float(value) <= 7:
                raise ValueError(f"{name} must lie between 1 and 7")
        if not math.isclose(
            float(self.neuroticism),
            8.0 - float(self.emotional_stability),
        ):
            raise ValueError(
                "neuroticism must equal 8 minus emotional_stability"
            )

    def as_ocean(self) -> Mapping[str, float]:
        """Return the stable numeric vector rendered to personality-aware users."""

        return MappingProxyType(
            {
                "openness": float(self.openness),
                "conscientiousness": float(self.conscientiousness),
                "extraversion": float(self.extraversion),
                "agreeableness": float(self.agreeableness),
                "neuroticism": float(self.neuroticism),
            }
        )


@dataclass(frozen=True)
class TipiProfileArtifact:
    """Frozen questionnaire result layered on one factual profile artifact."""

    schema_version: str
    user_id: int
    source_user_id: str
    factual_profile_fingerprint: str
    questionnaire_input_fingerprint: str
    instrument_id: str
    instrument_source: str
    ordered_item_text_hash: str
    response_scale: tuple[int, int]
    raw_responses: tuple[TipiResponse, ...]
    response_vector_fingerprint: str
    scoring_version: str
    neuroticism_conversion_version: str
    scores: TipiScores
    prompt_version: str
    prompt_fingerprint: str
    provider_settings: Mapping[str, Any]
    response_id: str | None
    usage: Mapping[str, Any]
    artifact_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_responses", tuple(self.raw_responses))
        object.__setattr__(
            self,
            "provider_settings",
            MappingProxyType(dict(self.provider_settings)),
        )
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))
        _validate_integer(self.user_id, name="user_id", minimum=1)
        if tuple(response.item_id for response in self.raw_responses) != tuple(
            range(1, 11)
        ):
            raise ValueError("raw_responses must contain ordered items 1 through 10")
        if self.response_scale != (1, 7):
            raise ValueError("response_scale must be (1, 7)")


@dataclass(frozen=True)
class RankedTipiProfile:
    """One natural-score rank; equal scores retain the same scientific rank."""

    position: int
    score_rank: int
    user_id: int
    source_user_id: str
    trait: str
    score: float
    artifact_fingerprint: str


def build_tipi_profile_prompt(
    profile_text: str,
    *,
    user_characteristics: str | None = None,
) -> str:
    """Render the paper-inspired prompt around one frozen MovieLens profile."""

    if not isinstance(profile_text, str) or not profile_text.strip():
        raise ValueError("profile_text must be a non-blank string")
    if user_characteristics is not None and (
        not isinstance(user_characteristics, str)
        or not user_characteristics.strip()
    ):
        raise ValueError(
            "user_characteristics must be a non-blank string or None"
        )
    response_scale = "\n".join(
        f"{score} = {label}" for score, label in TIPI_RESPONSE_SCALE
    )
    items = "\n".join(
        f"{index}. {text}" for index, text in enumerate(TIPI_ITEM_TEXTS, start=1)
    )
    return TIPI_USER_PROMPT_TEMPLATE.format(
        user_characteristics=(
            user_characteristics.strip()
            if user_characteristics is not None
            else "No additional demographic characteristics are available."
        ),
        profile_text=profile_text.strip(),
        response_scale=response_scale,
        items=items,
    )


def score_tipi(
    responses: Mapping[int, int] | Sequence[int],
) -> TipiScores:
    """Apply the official reverse-key-and-mean scoring sheet exactly."""

    values = _response_values(responses)

    def item(number: int) -> float:
        return float(values[number - 1])

    def reverse(number: int) -> float:
        return 8.0 - item(number)

    extraversion = _mean(item(1), reverse(6))
    agreeableness = _mean(reverse(2), item(7))
    conscientiousness = _mean(item(3), reverse(8))
    emotional_stability = _mean(reverse(4), item(9))
    openness = _mean(item(5), reverse(10))
    return TipiScores(
        extraversion=extraversion,
        agreeableness=agreeableness,
        conscientiousness=conscientiousness,
        emotional_stability=emotional_stability,
        openness=openness,
        neuroticism=8.0 - emotional_stability,
    )


class TipiProfileRanker:
    """Ask a frozen persona to answer TIPI, score it, and rank artifacts."""

    def __init__(self, *, provider: JSONProvider) -> None:
        self.provider = provider
        self.provider_settings = MappingProxyType(
            dict(provider.semantic_settings())
        )

    def evaluate_profile(
        self,
        profile: ProfileArtifact,
        *,
        user_characteristics: str | None = None,
    ) -> TipiProfileArtifact:
        if not isinstance(profile, ProfileArtifact):
            raise TypeError("profile must be a ProfileArtifact")
        user_prompt = build_tipi_profile_prompt(
            profile.profile_text,
            user_characteristics=user_characteristics,
        )
        outcome = self.provider.call_json(
            system_prompt=TIPI_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=TIPI_RESPONSE_SCHEMA,
            validator=_validate_tipi_payload,
            schema_name="tipi_persona_responses",
        )
        raw_responses = tuple(
            TipiResponse(
                item_id=int(row["item_id"]),
                score=int(row["score"]),
                rationale=str(row["rationale"]),
            )
            for row in outcome.payload["responses"]
        )
        response_vector = tuple(response.score for response in raw_responses)
        scores = score_tipi(response_vector)
        response_vector_fingerprint = fingerprint_semantic_state(
            response_vector,
            TIPI_RESPONSE_VECTOR_SCHEMA,
        )
        questionnaire_input_fingerprint = fingerprint_semantic_state(
            {
                "factual_profile_fingerprint": profile.profile_fingerprint,
                "user_text_builder_version": profile.user_text_builder_version,
                "user_text_fingerprint": profile.user_text_fingerprint,
                "user_characteristics": user_characteristics,
                "rendered_user_prompt": user_prompt,
            },
            TIPI_PROMPT_VERSION,
        )
        semantic_payload = {
            "schema_version": TIPI_ARTIFACT_SCHEMA,
            "user_id": profile.user_id,
            "source_user_id": profile.source_user_id,
            "factual_profile_fingerprint": profile.profile_fingerprint,
            "questionnaire_input_fingerprint": questionnaire_input_fingerprint,
            "instrument_id": TIPI_INSTRUMENT_ID,
            "instrument_source": TIPI_INSTRUMENT_SOURCE,
            "ordered_item_text_hash": TIPI_ORDERED_ITEM_TEXT_HASH,
            "response_scale": (1, 7),
            "raw_responses": raw_responses,
            "response_vector_fingerprint": response_vector_fingerprint,
            "scoring_version": TIPI_SCORING_VERSION,
            "neuroticism_conversion_version": (
                TIPI_NEUROTICISM_CONVERSION_VERSION
            ),
            "scores": scores,
            "prompt_version": TIPI_PROMPT_VERSION,
            "prompt_fingerprint": TIPI_PROMPT_FINGERPRINT,
            "provider_settings": self.provider_settings,
        }
        return TipiProfileArtifact(
            **semantic_payload,
            response_id=outcome.response_id,
            usage=outcome.usage,
            artifact_fingerprint=fingerprint_semantic_state(
                semantic_payload,
                TIPI_ARTIFACT_SCHEMA,
            ),
        )

    def evaluate_many(
        self,
        profiles: Sequence[ProfileArtifact],
        *,
        user_characteristics_by_user: Mapping[int, str] | None = None,
    ) -> tuple[TipiProfileArtifact, ...]:
        records = tuple(profiles)
        if any(not isinstance(profile, ProfileArtifact) for profile in records):
            raise TypeError("profiles must contain ProfileArtifact values")
        user_ids = [profile.user_id for profile in records]
        if len(user_ids) != len(set(user_ids)):
            raise ValueError("profiles contain duplicate user IDs")
        characteristics = dict(user_characteristics_by_user or {})
        unknown = sorted(set(characteristics).difference(user_ids))
        if unknown:
            raise ValueError(
                "user_characteristics_by_user contains unknown users: "
                f"{unknown}"
            )
        if user_characteristics_by_user is not None:
            missing = sorted(set(user_ids).difference(characteristics))
            if missing:
                raise ValueError(
                    "user_characteristics_by_user is missing users: "
                    f"{missing}"
                )
        return tuple(
            self.evaluate_profile(
                profile,
                user_characteristics=characteristics.get(profile.user_id),
            )
            for profile in sorted(records, key=lambda value: value.user_id)
        )

    @staticmethod
    def rank(
        artifacts: Sequence[TipiProfileArtifact],
        *,
        trait: str,
        descending: bool = True,
    ) -> tuple[RankedTipiProfile, ...]:
        return rank_tipi_profiles(
            artifacts,
            trait=trait,
            descending=descending,
        )


def rank_tipi_profiles(
    artifacts: Sequence[TipiProfileArtifact],
    *,
    trait: str,
    descending: bool = True,
) -> tuple[RankedTipiProfile, ...]:
    """Rank natural persona scores with deterministic user-ID tie breaking."""

    records = tuple(artifacts)
    if any(not isinstance(record, TipiProfileArtifact) for record in records):
        raise TypeError("artifacts must contain TipiProfileArtifact values")
    user_ids = [record.user_id for record in records]
    if len(user_ids) != len(set(user_ids)):
        raise ValueError("artifacts contain duplicate user IDs")
    normalized_trait = _normalized_trait_name(trait)

    def value(record: TipiProfileArtifact) -> float:
        return float(getattr(record.scores, normalized_trait))

    ordered = sorted(
        records,
        key=(
            (lambda record: (-value(record), record.user_id))
            if descending
            else (lambda record: (value(record), record.user_id))
        ),
    )
    ranked: list[RankedTipiProfile] = []
    prior_score: float | None = None
    score_rank = 0
    for position, record in enumerate(ordered, start=1):
        record_score = value(record)
        if prior_score is None or not math.isclose(record_score, prior_score):
            score_rank = position
            prior_score = record_score
        ranked.append(
            RankedTipiProfile(
                position=position,
                score_rank=score_rank,
                user_id=record.user_id,
                source_user_id=record.source_user_id,
                trait=normalized_trait,
                score=record_score,
                artifact_fingerprint=record.artifact_fingerprint,
            )
        )
    return tuple(ranked)


def load_or_create_tipi_scores(
    path: str | Path,
    *,
    profiles_by_user: Mapping[int, ProfileArtifact],
    user_characteristics_by_user: Mapping[int, str],
    provider: JSONProvider,
) -> Mapping[int, TipiScores]:
    """Load natural TIPI scores, creating one readable JSON file if needed."""

    cache_path = Path(path)
    profiles = dict(sorted(profiles_by_user.items()))
    if not profiles:
        raise ValueError("profiles_by_user cannot be empty")
    characteristics = dict(user_characteristics_by_user)
    if set(characteristics) != set(profiles):
        raise ValueError("user characteristics must match the TIPI users")
    provider_settings = dict(provider.semantic_settings())
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        artifacts = TipiProfileRanker(provider=provider).evaluate_many(
            tuple(profiles.values()),
            user_characteristics_by_user=characteristics,
        )
        payload = {
            "prompt_version": TIPI_PROMPT_VERSION,
            "provider_settings": provider_settings,
            "users": [
                {
                    "user_id": artifact.user_id,
                    "profile_fingerprint": artifact.factual_profile_fingerprint,
                    "responses": [asdict(value) for value in artifact.raw_responses],
                    "scores": asdict(artifact.scores),
                    "response_id": artifact.response_id,
                    "usage": dict(artifact.usage),
                }
                for artifact in artifacts
            ]
        }
        save_json_artifact(cache_path, payload)

    if payload.get("prompt_version") != TIPI_PROMPT_VERSION:
        raise ValueError("TIPI cache prompt version differs from this run")
    if payload.get("provider_settings") != provider_settings:
        raise ValueError("TIPI cache provider settings differ from this run")
    scores: dict[int, TipiScores] = {}
    for row in payload["users"]:
        user_id = int(row["user_id"])
        if user_id in scores:
            raise ValueError(f"TIPI cache repeats user {user_id}")
        profile = profiles.get(user_id)
        if (
            profile is None
            or row["profile_fingerprint"] != profile.profile_fingerprint
        ):
            raise ValueError(f"TIPI cache does not match profile for user {user_id}")
        computed = score_tipi(
            tuple(int(response["score"]) for response in row["responses"])
        )
        if row["scores"] != asdict(computed):
            raise ValueError(f"TIPI cache scores are inconsistent for user {user_id}")
        scores[user_id] = computed
    if set(scores) != set(profiles):
        raise ValueError("TIPI cache does not match the active user population")
    return MappingProxyType(scores)


def _validate_tipi_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(payload) != {"responses"}:
        raise ValueError("TIPI response must contain only responses")
    rows = payload["responses"]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("responses must be an array")
    if len(rows) != 10:
        raise ValueError("responses must contain exactly ten items")
    validated: list[dict[str, Any]] = []
    for expected_item_id, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise TypeError("each TIPI response must be an object")
        if set(row) != {"item_id", "score", "rationale"}:
            raise ValueError(
                "each TIPI response must contain item_id, score, and rationale"
            )
        item_id = row["item_id"]
        score = row["score"]
        rationale = row["rationale"]
        _validate_integer(item_id, name="item_id", minimum=1, maximum=10)
        if item_id != expected_item_id:
            raise ValueError("TIPI items must appear exactly once in order 1..10")
        _validate_integer(score, name="score", minimum=1, maximum=7)
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("TIPI rationale must be a non-blank string")
        validated.append(
            {
                "item_id": item_id,
                "score": score,
                "rationale": rationale.strip(),
            }
        )
    return {"responses": validated}


def _response_values(
    responses: Mapping[int, int] | Sequence[int],
) -> tuple[int, ...]:
    if isinstance(responses, Mapping):
        if any(
            isinstance(key, bool) or not isinstance(key, int)
            for key in responses
        ):
            raise TypeError("response mapping keys must be integers")
        if set(responses) != set(range(1, 11)):
            raise ValueError("response mapping must contain integer items 1..10")
        values = tuple(responses[index] for index in range(1, 11))
    else:
        if isinstance(responses, (str, bytes)) or not isinstance(
            responses, Sequence
        ):
            raise TypeError("responses must be a mapping or sequence")
        values = tuple(responses)
        if len(values) != 10:
            raise ValueError("response sequence must contain exactly ten values")
    for index, value in enumerate(values, start=1):
        _validate_integer(
            value,
            name=f"response item {index}",
            minimum=1,
            maximum=7,
        )
    return values


def _normalized_trait_name(trait: str) -> str:
    if not isinstance(trait, str) or not trait.strip():
        raise ValueError("trait must be a non-blank string")
    aliases = {
        "o": "openness",
        "openness": "openness",
        "c": "conscientiousness",
        "conscientiousness": "conscientiousness",
        "e": "extraversion",
        "extraversion": "extraversion",
        "a": "agreeableness",
        "agreeableness": "agreeableness",
        "emotional_stability": "emotional_stability",
        "emotional stability": "emotional_stability",
        "n": "neuroticism",
        "neuroticism": "neuroticism",
    }
    try:
        return aliases[trait.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported TIPI trait: {trait!r}") from exc


def _validate_integer(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise ValueError(f"{name} must be at least {minimum}")
        raise ValueError(f"{name} must lie between {minimum} and {maximum}")


def _mean(first: float, second: float) -> float:
    return (first + second) / 2.0


__all__ = [
    "SEARCH_PERSONALIZATION_PAPER_DOI",
    "SEARCH_PERSONALIZATION_PAPER_URL",
    "TIPI_ARTIFACT_SCHEMA",
    "TIPI_CACHE_FILENAME",
    "TIPI_INSTRUMENT_ID",
    "TIPI_INSTRUMENT_SOURCE",
    "TIPI_ITEM_TEXTS",
    "TIPI_NEUROTICISM_CONVERSION_VERSION",
    "TIPI_ORDERED_ITEM_TEXT_HASH",
    "TIPI_PROMPT_FINGERPRINT",
    "TIPI_PROMPT_VERSION",
    "TIPI_RESPONSE_SCHEMA",
    "TIPI_RESPONSE_SCALE",
    "TIPI_REVERSE_KEYED_ITEMS",
    "TIPI_SCORING_VERSION",
    "TIPI_SYSTEM_PROMPT",
    "RankedTipiProfile",
    "TipiProfileArtifact",
    "TipiProfileRanker",
    "TipiResponse",
    "TipiScores",
    "build_tipi_profile_prompt",
    "load_or_create_tipi_scores",
    "rank_tipi_profiles",
    "score_tipi",
]
