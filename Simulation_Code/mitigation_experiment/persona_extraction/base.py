"""Dataset-neutral profile-extraction contracts."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Protocol

from ..contracts import ItemId, UserId


def _immutable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("mapping keys must be strings")
        return MappingProxyType(
            {key: _immutable_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(item) for item in value)
    return copy.deepcopy(value)


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _immutable_value(value)
    if not isinstance(frozen, Mapping):
        raise TypeError("value must be a mapping")
    return frozen


@dataclass(frozen=True)
class ProfileInteraction:
    """One training-split interaction available to an extractor."""

    event_id: str
    source_order: int
    source_split: str
    item_id: ItemId
    source_item_id: str
    timestamp: int
    rating: float | None
    item_text: str

    def __post_init__(self) -> None:
        for name in ("event_id", "source_split", "source_item_id", "item_text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        if isinstance(self.source_order, bool) or not isinstance(
            self.source_order, int
        ):
            raise TypeError("source_order must be an integer")
        if self.source_order < 0:
            raise ValueError("source_order cannot be negative")
        if isinstance(self.item_id, bool) or not isinstance(self.item_id, int):
            raise TypeError("item_id must be an integer")
        if self.item_id <= 0:
            raise ValueError("item_id must be positive")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int):
            raise TypeError("timestamp must be an integer")
        if self.rating is not None:
            if isinstance(self.rating, bool) or not isinstance(
                self.rating, (int, float)
            ):
                raise TypeError("rating must be numeric or None")
            if not math.isfinite(float(self.rating)):
                raise ValueError("rating must be finite")


@dataclass(frozen=True)
class ProfileExtractionInput:
    """Explicit user join plus training-only evidence."""

    user_id: UserId
    source_user_id: str
    user_text: str | None
    user_text_builder_version: str | None
    user_text_fingerprint: str | None
    interactions: tuple[ProfileInteraction, ...]
    training_history_fingerprint: str
    dataset_fingerprint: str
    extraction_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "interactions", tuple(self.interactions))
        if isinstance(self.user_id, bool) or not isinstance(self.user_id, int):
            raise TypeError("user_id must be an integer")
        if self.user_id <= 0:
            raise ValueError("user_id must be positive")
        if not isinstance(self.source_user_id, str) or not self.source_user_id.strip():
            raise ValueError("source_user_id must be a non-blank string")
        if self.user_text is None:
            if self.user_text_builder_version is not None:
                raise ValueError(
                    "user_text_builder_version must be None when user_text is None"
                )
            if self.user_text_fingerprint is not None:
                raise ValueError(
                    "user_text_fingerprint must be None when user_text is None"
                )
        else:
            if not isinstance(self.user_text, str):
                raise TypeError("user_text must be a string or None")
            if not self.user_text.strip():
                raise ValueError("user_text must be non-blank when supplied")
            for name in (
                "user_text_builder_version",
                "user_text_fingerprint",
            ):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{name} is required when user_text is supplied"
                    )
        if any(
            not isinstance(interaction, ProfileInteraction)
            for interaction in self.interactions
        ):
            raise TypeError("interactions must contain ProfileInteraction values")
        for name in ("training_history_fingerprint", "dataset_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        if isinstance(self.extraction_seed, bool) or not isinstance(
            self.extraction_seed, int
        ):
            raise TypeError("extraction_seed must be an integer")
        if self.extraction_seed < 0:
            raise ValueError("extraction_seed cannot be negative")


@dataclass(frozen=True)
class ProfileArtifact:
    """One user's completed, normalized, cached factual profile.

    This is the explicit replacement for relying on the old positional persona
    rows.  The text is what the simulator consumes; the remaining fields prove
    which source user, training history, extractor, prompt, and settings
    produced it, and let every condition reuse the same profile without another
    provider call.
    """

    schema_version: str
    user_id: UserId
    source_user_id: str
    extractor_id: str
    extraction_seed: int
    profile_text: str
    intermediate_profiles: tuple[str, ...]
    update_summaries: tuple[str, ...]
    dataset_fingerprint: str
    training_history_fingerprint: str
    user_text_builder_version: str | None
    user_text_fingerprint: str | None
    input_fingerprint: str
    prompt_fingerprint: str
    method_settings: Mapping[str, Any]
    provider_settings: Mapping[str, Any]
    profile_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "intermediate_profiles", tuple(self.intermediate_profiles)
        )
        object.__setattr__(self, "update_summaries", tuple(self.update_summaries))
        object.__setattr__(
            self, "method_settings", _immutable_mapping(self.method_settings)
        )
        object.__setattr__(
            self, "provider_settings", _immutable_mapping(self.provider_settings)
        )


class ProfileFailureStatus(str, Enum):
    INVALID_INPUT = "invalid_input"
    PROVIDER_FAILURE = "provider_failure"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True)
class ProfileExtractionFailure:
    """Preparation failure retained for audit instead of replacing the user."""

    schema_version: str
    user_id: UserId
    source_user_id: str
    extractor_id: str
    extraction_seed: int
    status: ProfileFailureStatus
    input_fingerprint: str
    reason: str
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", _immutable_mapping(self.diagnostics))


ProfileExtractionResult = ProfileArtifact | ProfileExtractionFailure


class ProfileExtractor(Protocol):
    extractor_id: str

    def extract(self, input_record: ProfileExtractionInput) -> ProfileExtractionResult: ...

    def extract_many(
        self, input_records: Sequence[ProfileExtractionInput]
    ) -> Sequence[ProfileExtractionResult]: ...

# An extractor may make a predeclared number of retries (for example, three)
# for the same user and input. It must then emit a failure record. Silently
# substituting another user would change the sampled population. If replacement
# is allowed, population preparation must apply a separately declared,
# deterministic rule and write a new population/profile fingerprint.
# Validation, test, or synthetic records remain detectable through source_split
# and stable event provenance.
