"""Recommender interface required by the generic simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from ..contracts import ItemId, UserId


@dataclass(frozen=True)
class FitSpec:
    iteration: int
    training_fingerprint: str
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class FitReport:
    checkpoint_fingerprint: str
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class RecommendationBatch:
    items_by_user: Mapping[UserId, tuple[ItemId, ...]]
    recommender_fingerprint: str


class Recommender(Protocol):
    def fit(
        self, histories: Mapping[UserId, Sequence[ItemId]], spec: FitSpec
    ) -> FitReport: ...

    def recommend(
        self,
        user_ids: Sequence[UserId],
        k: int,
        inference_histories: Mapping[UserId, Sequence[ItemId]],
        eligible_items: Mapping[UserId, Sequence[ItemId]],
    ) -> RecommendationBatch: ...

    def score_candidates(
        self,
        user_id: UserId,
        inference_history: Sequence[ItemId],
        candidate_ids: Sequence[ItemId],
    ) -> Sequence[float]: ...

    def export_state(self) -> Mapping[str, Any]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...

    def clone(self) -> "Recommender": ...

    def fingerprint(self) -> str: ...
