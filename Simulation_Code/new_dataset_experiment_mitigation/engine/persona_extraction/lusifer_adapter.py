"""Provider-neutral factual-profile extraction using Lusifer's prompts.

The initial and update workflow derives from ``danialebrat/Lusifer`` at the
pinned revision recorded below. This copy changes the domain to books,
authors and writing styles, forbids invented reader demographics or inferred
author gender, and limits evidence to the latest 40 training events. JSON output
uses the local structured-output schemas. The implementation uses explicit IDs
and training-only inputs and emits immutable fingerprinted artifacts. It never
imports the vendored upstream runtime, pandas, an SDK, or credentials.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from typing import Any

from ..utility.fingerprints import fingerprint_semantic_state
from ..utility.providers import (
    InvalidProviderResponse,
    JSONProvider,
    ProviderError,
)
from .base import (
    ProfileArtifact,
    ProfileExtractionFailure,
    ProfileExtractionInput,
    ProfileExtractionResult,
    ProfileFailureStatus,
    ProfileInteraction,
)


LUSIFER_REVISION = "83a40a9c37ad8edc504321393964419f8505e0a1"
LUSIFER_SOURCE_URL = (
    "https://github.com/danialebrat/Lusifer/blob/"
    f"{LUSIFER_REVISION}/Lusifer.py"
)

PROFILE_ARTIFACT_SCHEMA = "goodreads-profile-artifact-v1"
PROFILE_INPUT_SCHEMA = "goodreads-profile-input-v1"
PROMPT_TEMPLATE_VERSION = "lusifer_books_training_only_v1"

SYSTEM_INSTRUCTIONS = """
You are an advanced language model assistant designed to perform specific tasks based on instructions.
    
**General Guidelines**:
- Follow the prompt instructions precisely.
- Provide clear, concise, and accurate information.
- Ensure all outputs are properly formatted as per the instructions.
""".strip()

INITIAL_PROFILE_INSTRUCTIONS = """
You are an expert data analyst specializing in user behavior and preferences in books.

**Task**: Analyze the user's characteristics based on their previous ratings and provide an in-depth profile.

**Instructions**:
- Never invent reader age, gender, occupation, or other demographics.
- Author names are permitted; author gender is not evidence and must not be inferred.
- Identify genres, themes, authors, writing styles, and other book attributes the user enjoys or dislikes.
- Describe the qualities or characteristics of books that the user enjoys (based on high ratings like 4 or 5) and those they dislike (based on low ratings, like 1 or 2) without mentioning book titles or ratings.
- Highlight any patterns or preferences evident from the ratings.
- The profile should be detailed enough to predict the user's potential ratings for unseen books.
- Ensure the profile is comprehensive and presents a cohesive analysis of the user's preferences.
- Avoid mentioning the process or using phrases like "the user seems to".
""".strip()

UPDATE_PROFILE_INSTRUCTIONS = """
You are updating an existing user profile based on new rating data.

**Task**: Integrate the new ratings into the user's profile to refine and enhance the analysis.

**Instructions**:
- Never invent reader demographics or infer author gender.
- Incorporate insights from the new ratings with the existing profile by mentioning shifts in tastes or new patterns revealed by the latest ratings.
- Update any changes in user profiles or new patterns and preferences that emerge or rise to make the user-profile richer.
- Ensure the updated profile is comprehensive and presents a cohesive analysis of the user's preferences.
- Do not reference previous summaries or indicate that this profile has been updated.
- Describe the qualities or characteristics of books that the user enjoys (based on high ratings like 4 or 5) and those they dislike (based on low ratings, like 1 or 2).
""".strip()

# Note: outputformat deliberatly differs from original lusifer design
INITIAL_OUTPUT_INSTRUCTIONS = """
Provide `user_profile` as cohesive and coherent structured text describing the
user's profile. Return exactly one JSON object containing only `user_profile`.
Do not include any text outside the JSON object.
""".strip()

UPDATE_OUTPUT_INSTRUCTIONS = """
Provide `user_profile` as cohesive and coherent structured text describing the
complete user profile in two paragraphs. Provide `update` as a brief, one-sentence
highlight of changes compared with the previous profile. Return exactly one JSON
object containing only `user_profile` and `update`. Do not include any text
outside the JSON object.
""".strip()

PROMPT_PREAMBLE = "You are provided with the following information about a user:"
USER_PROFILE_HEADING = "**User Profile**:"
RECENT_RATINGS_HEADING = "**Recent Ratings**::"
BOOK_LABEL = "- **Book**:"
RATING_LABEL = "**Rating**:"
PROFILE_PROMPT_TEMPLATE = (
    "{preamble}\n\n"
    "{user_profile_heading}\n{user_profile_input}\n\n"
    "{recent_ratings_heading}\n{evidence}\n\n"
    "{profile_instructions}\n\n"
    "{output_instructions}"
)
RATED_INTERACTION_TEMPLATE = (
    "{book_label} {item_text}\n {rating_label} {rating}"
)

INITIAL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"user_profile": {"type": "string", "minLength": 1}},
    "required": ["user_profile"],
    "additionalProperties": False,
}

UPDATE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "user_profile": {"type": "string", "minLength": 1},
        "update": {"type": "string", "minLength": 1},
    },
    "required": ["user_profile", "update"],
    "additionalProperties": False,
}

logger = logging.getLogger(__name__)


class LusiferProfileExtractor:
    """Extract frozen profiles with pinned upstream Lusifer prompt semantics."""

    extractor_id = "lusifer_books_training_only_v1"

    def __init__(
        self,
        *,
        provider: JSONProvider,
        max_interactions: int = 40,
        batch_size: int = 10,
    ) -> None:
        if isinstance(max_interactions, bool) or not isinstance(max_interactions, int):
            raise TypeError("max_interactions must be an integer")
        if max_interactions <= 0:
            raise ValueError("max_interactions must be positive")
        if max_interactions != 40:
            raise ValueError("book profiles use exactly the latest 40 available training events")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.provider = provider
        self.max_interactions = max_interactions
        self.batch_size = batch_size
        self.provider_settings = dict(provider.semantic_settings())
        self.method_settings: dict[str, Any] = {
            "max_interactions": max_interactions,
            "batch_size": batch_size,
            "selection": "latest",
            "selection_order": "timestamp_then_source_order",
            "processing_order": "oldest_to_newest",
            "source_revision": LUSIFER_REVISION,
        }
        self.prompt_fingerprint = fingerprint_semantic_state(
            {
                "system": SYSTEM_INSTRUCTIONS,
                "initial": INITIAL_PROFILE_INSTRUCTIONS,
                "update": UPDATE_PROFILE_INSTRUCTIONS,
                "initial_output": INITIAL_OUTPUT_INSTRUCTIONS,
                "update_output": UPDATE_OUTPUT_INSTRUCTIONS,
                "preamble": PROMPT_PREAMBLE,
                "user_profile_heading": USER_PROFILE_HEADING,
                "recent_ratings_heading": RECENT_RATINGS_HEADING,
                "book_label": BOOK_LABEL,
                "rating_label": RATING_LABEL,
                "profile_prompt_template": PROFILE_PROMPT_TEMPLATE,
                "rated_interaction_template": RATED_INTERACTION_TEMPLATE,
                "initial_schema": INITIAL_RESPONSE_SCHEMA,
                "update_schema": UPDATE_RESPONSE_SCHEMA,
            },
            PROMPT_TEMPLATE_VERSION,
        )

    def extract(self, input_record: ProfileExtractionInput) -> ProfileExtractionResult:
        if not isinstance(input_record, ProfileExtractionInput):
            raise TypeError("input_record must be a ProfileExtractionInput")

        ordered_interactions = tuple(
            sorted(
                input_record.interactions,
                key=lambda event: (event.timestamp, event.source_order),
            )
        )
        input_fingerprint = self._input_fingerprint(
            input_record, ordered_interactions
        )
        input_error = self._semantic_input_error(ordered_interactions)
        if input_error is not None:
            return self._failure(
                input_record=input_record,
                input_fingerprint=input_fingerprint,
                status=ProfileFailureStatus.INVALID_INPUT,
                reason=input_error,
                diagnostics={"provider_calls": 0},
            )

        selected = ordered_interactions[-self.max_interactions :]
        chunks = tuple(
            selected[index : index + self.batch_size]
            for index in range(0, len(selected), self.batch_size)
        )
        intermediate_profiles: list[str] = []
        update_summaries: list[str] = []

        for chunk_index, chunk in enumerate(chunks, start=1):
            is_update = bool(intermediate_profiles)
            current_profile = (
                intermediate_profiles[-1] if intermediate_profiles else None
            )
            user_prompt = self._profile_prompt(
                input_record=input_record,
                interactions=chunk,
                current_profile=current_profile,
            )
            validator = (
                self._validate_update_payload
                if is_update
                else self._validate_initial_payload
            )
            schema = UPDATE_RESPONSE_SCHEMA if is_update else INITIAL_RESPONSE_SCHEMA

            try:
                outcome = self.provider.call_json(
                    system_prompt=SYSTEM_INSTRUCTIONS,
                    user_prompt=user_prompt,
                    response_schema=schema,
                    validator=validator,
                    schema_name=(
                        "lusifer_updated_profile"
                        if is_update
                        else "lusifer_initial_profile"
                    ),
                )
                payload = outcome.payload
            except InvalidProviderResponse as exc:
                diagnostics = self._error_diagnostics(
                    exc=exc,
                    chunk_index=chunk_index,
                    completed_chunks=len(intermediate_profiles),
                )
                logger.error(
                    "Lusifer returned an invalid profile response for user %s "
                    "at chunk %d/%d: %s",
                    input_record.user_id,
                    chunk_index,
                    len(chunks),
                    diagnostics["error_message"],
                )
                return self._failure(
                    input_record=input_record,
                    input_fingerprint=input_fingerprint,
                    status=ProfileFailureStatus.INVALID_RESPONSE,
                    reason="profile provider returned an invalid response",
                    diagnostics=diagnostics,
                )
            except ProviderError as exc:
                diagnostics = self._error_diagnostics(
                    exc=exc,
                    chunk_index=chunk_index,
                    completed_chunks=len(intermediate_profiles),
                )
                logger.error(
                    "Lusifer profile provider failed for user %s at chunk "
                    "%d/%d: %s",
                    input_record.user_id,
                    chunk_index,
                    len(chunks),
                    diagnostics["error_message"],
                )
                return self._failure(
                    input_record=input_record,
                    input_fingerprint=input_fingerprint,
                    status=ProfileFailureStatus.PROVIDER_FAILURE,
                    reason="profile provider call failed",
                    diagnostics=diagnostics,
                )

            intermediate_profiles.append(str(payload["user_profile"]))
            if is_update:
                update_summaries.append(str(payload["update"]))

        profile_text = intermediate_profiles[-1]
        profile_payload = {
            "schema_version": PROFILE_ARTIFACT_SCHEMA,
            "user_id": input_record.user_id,
            "source_user_id": input_record.source_user_id,
            "extractor_id": self.extractor_id,
            "extraction_seed": input_record.extraction_seed,
            "profile_text": profile_text,
            "intermediate_profiles": tuple(intermediate_profiles),
            "update_summaries": tuple(update_summaries),
            "dataset_fingerprint": input_record.dataset_fingerprint,
            "training_history_fingerprint": (
                input_record.training_history_fingerprint
            ),
            "user_text_builder_version": (
                input_record.user_text_builder_version
            ),
            "user_text_fingerprint": input_record.user_text_fingerprint,
            "input_fingerprint": input_fingerprint,
            "prompt_fingerprint": self.prompt_fingerprint,
            "method_settings": self.method_settings,
            "provider_settings": self.provider_settings,
        }
        profile_fingerprint = fingerprint_semantic_state(
            profile_payload, PROFILE_ARTIFACT_SCHEMA
        )
        return ProfileArtifact(
            **profile_payload,
            profile_fingerprint=profile_fingerprint,
        )

    def extract_many(
        self, input_records: Sequence[ProfileExtractionInput]
    ) -> Sequence[ProfileExtractionResult]:
        records = tuple(input_records)
        user_ids = [record.user_id for record in records]
        source_user_ids = [record.source_user_id for record in records]
        if len(set(user_ids)) != len(user_ids):
            raise ValueError("input_records contain duplicate runtime user IDs")
        if len(set(source_user_ids)) != len(source_user_ids):
            raise ValueError("input_records contain duplicate source user IDs")
        return tuple(self.extract(record) for record in records)

    def _profile_prompt(
        self,
        *,
        input_record: ProfileExtractionInput,
        interactions: Sequence[ProfileInteraction],
        current_profile: str | None,
    ) -> str:
        evidence = self._format_interactions(interactions)
        if current_profile is None:
            user_profile_input = (
                input_record.user_text.strip()
                if input_record.user_text and input_record.user_text.strip()
                else "No additional user attributes are available."
            )
            profile_instructions = INITIAL_PROFILE_INSTRUCTIONS
            output_instructions = INITIAL_OUTPUT_INSTRUCTIONS
        else:
            # Upstream replaces the original user attributes with the complete
            # current profile when constructing an update prompt.
            user_profile_input = current_profile
            profile_instructions = UPDATE_PROFILE_INSTRUCTIONS
            output_instructions = UPDATE_OUTPUT_INSTRUCTIONS

        return PROFILE_PROMPT_TEMPLATE.format(
            preamble=PROMPT_PREAMBLE,
            user_profile_heading=USER_PROFILE_HEADING,
            user_profile_input=user_profile_input,
            recent_ratings_heading=RECENT_RATINGS_HEADING,
            evidence=evidence,
            profile_instructions=profile_instructions,
            output_instructions=output_instructions,
        )

    @staticmethod
    def _format_interactions(
        interactions: Sequence[ProfileInteraction],
    ) -> str:
        lines: list[str] = []
        for interaction in interactions:
            if interaction.rating is None:
                raise ValueError(
                    "the upstream Lusifer prompt requires explicit ratings"
                )
            line = RATED_INTERACTION_TEMPLATE.format(
                book_label=BOOK_LABEL,
                item_text=interaction.item_text,
                rating_label=RATING_LABEL,
                rating=f"{float(interaction.rating):g}",
            )
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _validate_initial_payload(
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if set(payload) != {"user_profile"}:
            raise ValueError("initial response must contain only user_profile")
        profile = payload["user_profile"]
        if not isinstance(profile, str) or not profile.strip():
            raise ValueError("user_profile must be a non-blank string")
        return {"user_profile": profile.strip()}

    @staticmethod
    def _validate_update_payload(
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if set(payload) != {"user_profile", "update"}:
            raise ValueError(
                "update response must contain only user_profile and update"
            )
        profile = payload["user_profile"]
        update = payload["update"]
        if not isinstance(profile, str) or not profile.strip():
            raise ValueError("user_profile must be a non-blank string")
        if not isinstance(update, str) or not update.strip():
            raise ValueError("update must be a non-blank string")
        return {"user_profile": profile.strip(), "update": update.strip()}

    @staticmethod
    def _semantic_input_error(
        interactions: Sequence[ProfileInteraction],
    ) -> str | None:
        if not interactions:
            return "profile extraction requires at least one interaction"
        if any(interaction.source_split != "train" for interaction in interactions):
            return "all profile evidence must declare source_split='train'"
        if any(interaction.rating is None for interaction in interactions):
            return "the upstream Lusifer prompt requires explicit ratings"
        event_ids = [interaction.event_id for interaction in interactions]
        if len(set(event_ids)) != len(event_ids):
            return "profile evidence contains duplicate event IDs"
        source_orders = [interaction.source_order for interaction in interactions]
        if len(set(source_orders)) != len(source_orders):
            return "profile evidence contains duplicate source-order values"
        return None

    @staticmethod
    def _input_fingerprint(
        input_record: ProfileExtractionInput,
        ordered_interactions: tuple[ProfileInteraction, ...],
    ) -> str:
        return fingerprint_semantic_state(
            {
                "user_id": input_record.user_id,
                "source_user_id": input_record.source_user_id,
                "user_text": input_record.user_text,
                "user_text_builder_version": (
                    input_record.user_text_builder_version
                ),
                "user_text_fingerprint": input_record.user_text_fingerprint,
                "interactions": ordered_interactions,
                "training_history_fingerprint": (
                    input_record.training_history_fingerprint
                ),
                "dataset_fingerprint": input_record.dataset_fingerprint,
                "extraction_seed": input_record.extraction_seed,
            },
            PROFILE_INPUT_SCHEMA,
        )

    def _failure(
        self,
        *,
        input_record: ProfileExtractionInput,
        input_fingerprint: str,
        status: ProfileFailureStatus,
        reason: str,
        diagnostics: Mapping[str, Any],
    ) -> ProfileExtractionFailure:
        return ProfileExtractionFailure(
            schema_version=PROFILE_ARTIFACT_SCHEMA,
            user_id=input_record.user_id,
            source_user_id=input_record.source_user_id,
            extractor_id=self.extractor_id,
            extraction_seed=input_record.extraction_seed,
            status=status,
            input_fingerprint=input_fingerprint,
            reason=reason,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _error_diagnostics(
        *,
        exc: ProviderError,
        chunk_index: int,
        completed_chunks: int,
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "error_type": type(exc).__name__,
            "error_message": LusiferProfileExtractor._safe_error_message(exc),
            "failed_chunk": chunk_index,
            "completed_chunks": completed_chunks,
        }
        return diagnostics

    @staticmethod
    def _safe_error_message(exc: Exception) -> str:
        message = " ".join(str(exc).split())
        if not message:
            return "No error message was provided."
        return message[:500]


__all__ = [
    "INITIAL_PROFILE_INSTRUCTIONS",
    "INITIAL_OUTPUT_INSTRUCTIONS",
    "INITIAL_RESPONSE_SCHEMA",
    "LUSIFER_REVISION",
    "LUSIFER_SOURCE_URL",
    "LusiferProfileExtractor",
    "BOOK_LABEL",
    "PROFILE_PROMPT_TEMPLATE",
    "PROMPT_PREAMBLE",
    "PROMPT_TEMPLATE_VERSION",
    "RATING_LABEL",
    "RATED_INTERACTION_TEMPLATE",
    "RECENT_RATINGS_HEADING",
    "SYSTEM_INSTRUCTIONS",
    "UPDATE_PROFILE_INSTRUCTIONS",
    "UPDATE_OUTPUT_INSTRUCTIONS",
    "UPDATE_RESPONSE_SCHEMA",
    "USER_PROFILE_HEADING",
]
