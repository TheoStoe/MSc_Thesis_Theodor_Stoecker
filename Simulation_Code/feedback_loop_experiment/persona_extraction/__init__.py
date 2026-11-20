"""Offline, replaceable factual-profile extraction interfaces."""

from .base import (
    ProfileArtifact,
    ProfileExtractionFailure,
    ProfileExtractionInput,
    ProfileExtractionResult,
    ProfileExtractor,
    ProfileFailureStatus,
    ProfileInteraction,
)
from .lusifer_adapter import LusiferProfileExtractor
from .movielens_1m import (
    MatchedMovieLensUserText,
    MovieLensUserAttributes,
    match_movielens_1m_user_texts,
    parse_movielens_1m_users,
    render_movielens_1m_user_text,
)

__all__ = [
    "LusiferProfileExtractor",
    "MatchedMovieLensUserText",
    "MovieLensUserAttributes",
    "ProfileArtifact",
    "ProfileExtractionFailure",
    "ProfileExtractionInput",
    "ProfileExtractionResult",
    "ProfileExtractor",
    "ProfileFailureStatus",
    "ProfileInteraction",
    "match_movielens_1m_user_texts",
    "parse_movielens_1m_users",
    "render_movielens_1m_user_text",
]
