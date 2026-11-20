"""Self-contained NCM response models for one-choice recommendation slates.

The simple pointwise model and the ordered-slate GRU were independently
implemented for this thesis from the architectures described in *Neural Click
Models for Recommender Systems*. The paper repository is cited for
methodological provenance, but its notebook code is not copied:

https://github.com/arabel1a/Neural-Click-Models-for-Recommender-Systems

Both models emit one independent binary click logit per displayed position.
``NCMResponseModel`` converts those logits to the first-click distribution
required by this experiment's single-choice protocol. The raw no-click mass
remains diagnostic because the reconstructed training data has no all-no-click
examples with which to calibrate it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import (
    ActionDecision,
    ActionStatus,
    Continuation,
    ContinuationDecision,
    ContinuationStatus,
    TurnInput,
    UpdateStatus,
    UserAction,
    UserUpdateResult,
)
from ..utility.fingerprints import fingerprint_semantic_state
from ..utility.randomness import RandomKey, random_generator
from .base import UserModelCapabilities


NCMArchitecture = Literal["pointwise_logistic", "slatewise_gru"]
# This literal is retained because the completed pilot checkpoints persist it.
NCM_MODEL_CONFIG_SCHEMA_VERSION = "learned-first-click-model-config-v2"
NCM_ARTIFACT_CONDITION_BY_MODEL = {
    "pointwise_logistic": "T1_pointwise_logistic_uniform",
    "ncm_inspired_gru_uniform": "T2_slatewise_gru_uniform",
    "ncm_inspired_gru_mansoury": "T3_slatewise_gru_mansoury",
}


@dataclass(frozen=True)
class NCMClickModelConfig:
    """Architecture settings stored in the compatible NCM checkpoints."""

    architecture: NCMArchitecture
    embedding_dimension: int
    slate_size: int = 10
    hidden_size: int = 64
    schema_version: str = NCM_MODEL_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.architecture not in {"pointwise_logistic", "slatewise_gru"}:
            raise ValueError(f"unsupported NCM architecture: {self.architecture}")
        for value, name in (
            (self.embedding_dimension, "embedding_dimension"),
            (self.slate_size, "slate_size"),
            (self.hidden_size, "hidden_size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.slate_size != 10:
            raise ValueError("the NCM baseline requires ten-item slates")
        if self.schema_version != NCM_MODEL_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported NCM model-config schema")


def _validate_inputs(
    item_embeddings: torch.Tensor,
    user_embeddings: torch.Tensor,
    *,
    embedding_dimension: int,
    slate_size: int,
) -> None:
    if item_embeddings.ndim != 3:
        raise ValueError("item_embeddings must have shape [batch, slate, dimension]")
    if user_embeddings.ndim != 2:
        raise ValueError("user_embeddings must have shape [batch, dimension]")
    batch, items, dimension = item_embeddings.shape
    if user_embeddings.shape != (batch, dimension):
        raise ValueError("user embeddings must align with the item batch")
    if items != slate_size:
        raise ValueError(f"model requires exactly {slate_size} displayed items")
    if dimension != embedding_dimension:
        raise ValueError("embedding dimension does not match model configuration")
    if not torch.is_floating_point(item_embeddings):
        raise TypeError("item_embeddings must be floating point")
    if not torch.is_floating_point(user_embeddings):
        raise TypeError("user_embeddings must be floating point")


class PointwiseLogisticRegression(nn.Module):
    """Paper-style linear click score over concatenated user/item embeddings.

    Rank, popularity, dot-product, and elementwise-interaction features are
    deliberately absent. The model is intentionally the simple baseline.
    """

    def __init__(self, config: NCMClickModelConfig) -> None:
        super().__init__()
        if config.architecture != "pointwise_logistic":
            raise ValueError(
                "PointwiseLogisticRegression needs pointwise_logistic config"
            )
        self.config = config
        self.click_logit = nn.Linear(2 * config.embedding_dimension, 1)

    def forward(
        self,
        item_embeddings: torch.Tensor,
        user_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(
            item_embeddings,
            user_embeddings,
            embedding_dimension=self.config.embedding_dimension,
            slate_size=self.config.slate_size,
        )
        repeated_users = user_embeddings.unsqueeze(1).expand(
            -1,
            self.config.slate_size,
            -1,
        )
        features = torch.cat((repeated_users, item_embeddings), dim=2)
        return self.click_logit(features).squeeze(-1)


class NCMSlatewiseGRU(nn.Module):
    """Ordered-slate NCM with one click logit per displayed position."""

    def __init__(self, config: NCMClickModelConfig) -> None:
        super().__init__()
        if config.architecture != "slatewise_gru":
            raise ValueError("NCMSlatewiseGRU needs slatewise_gru config")
        self.config = config
        self.user_to_hidden = nn.Linear(
            config.embedding_dimension,
            config.hidden_size,
        )
        self.slate_gru = nn.GRU(
            input_size=config.embedding_dimension,
            hidden_size=config.hidden_size,
            batch_first=True,
        )
        self.position_readout = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        item_embeddings: torch.Tensor,
        user_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(
            item_embeddings,
            user_embeddings,
            embedding_dimension=self.config.embedding_dimension,
            slate_size=self.config.slate_size,
        )
        initial_hidden = torch.tanh(self.user_to_hidden(user_embeddings)).unsqueeze(0)
        hidden_states, _ = self.slate_gru(item_embeddings, initial_hidden)
        return self.position_readout(hidden_states).squeeze(-1)


def build_ncm_model(config: NCMClickModelConfig) -> nn.Module:
    """Construct exactly the locally declared NCM architecture."""

    if config.architecture == "pointwise_logistic":
        return PointwiseLogisticRegression(config)
    if config.architecture == "slatewise_gru":
        return NCMSlatewiseGRU(config)
    raise ValueError(f"unsupported architecture: {config.architecture}")


@dataclass(frozen=True)
class NCMChoiceProbabilities:
    """NCM click tendencies and their derived single-choice distribution."""

    independent_click_probabilities: torch.Tensor
    ordered_choice_masses: torch.Tensor
    no_click_masses: torch.Tensor
    conditioned_choice_probabilities: torch.Tensor


def ncm_choice_probabilities(
    logits: torch.Tensor,
) -> NCMChoiceProbabilities:
    """Convert NCM binary logits into a stable first-click distribution."""

    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [batch, non-empty slate]")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must be floating point")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")

    log_click = F.logsigmoid(logits)
    log_no_click = F.logsigmoid(-logits)
    log_survival_before = torch.cat(
        (
            torch.zeros_like(logits[:, :1]),
            torch.cumsum(log_no_click[:, :-1], dim=1),
        ),
        dim=1,
    )
    log_ordered_choice = log_survival_before + log_click
    return NCMChoiceProbabilities(
        independent_click_probabilities=torch.sigmoid(logits),
        ordered_choice_masses=torch.exp(log_ordered_choice),
        no_click_masses=torch.exp(log_no_click.sum(dim=1, keepdim=True)),
        conditioned_choice_probabilities=torch.softmax(log_ordered_choice, dim=1),
    )


class NCMResponseModel:
    """Use observed history with a frozen NCM checkpoint."""

    VARIANT_ID = "ncm_response_v1"
    capabilities = UserModelCapabilities(
        supports_no_click=False,
        supports_state_update=False,
        supports_continuation=False,
    )

    def __init__(
        self,
        *,
        predictor: nn.Module,
        item_embeddings: Mapping[int, Sequence[float]],
        model_name: str,
        checkpoint_fingerprint: str,
        source_embedding_fingerprint: str | None = None,
        seed: int = 42,
    ) -> None:
        if not isinstance(predictor, nn.Module):
            raise TypeError("predictor must be a torch module")
        config = getattr(predictor, "config", None)
        if not isinstance(config, NCMClickModelConfig):
            raise TypeError("predictor must expose a local NCMClickModelConfig")
        for value, name in (
            (model_name, "model_name"),
            (checkpoint_fingerprint, "checkpoint_fingerprint"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed cannot be negative")

        vectors: dict[int, np.ndarray] = {}
        for item_id, vector in item_embeddings.items():
            if isinstance(item_id, bool) or not isinstance(item_id, int):
                raise TypeError("item embedding keys must be integers")
            if item_id <= 0:
                raise ValueError("item IDs must be positive")
            value = np.asarray(vector, dtype=np.float32).copy()
            value.setflags(write=False)
            vectors[item_id] = value
        if not vectors:
            raise ValueError("item_embeddings cannot be empty")
        dimensions = {vector.shape for vector in vectors.values()}
        if len(dimensions) != 1 or len(next(iter(dimensions))) != 1:
            raise ValueError("all item embeddings must be equal-length vectors")
        dimension = next(iter(dimensions))[0]
        if dimension != config.embedding_dimension:
            raise ValueError("embedding table and predictor dimensions disagree")
        if any(not bool(np.isfinite(vector).all()) for vector in vectors.values()):
            raise ValueError("item embeddings must be finite")

        self.predictor = predictor.eval()
        self.item_embeddings = MappingProxyType(vectors)
        self.model_name = model_name
        self.checkpoint_fingerprint = checkpoint_fingerprint
        self.seed = seed
        self.embedding_fingerprint = fingerprint_semantic_state(
            vectors,
            "ncm-item-embedding-table-v1",
        )
        self.source_embedding_fingerprint = (
            source_embedding_fingerprint or self.embedding_fingerprint
        )
        if not isinstance(self.source_embedding_fingerprint, str) or not (
            self.source_embedding_fingerprint.strip()
        ):
            raise ValueError("source_embedding_fingerprint must be non-blank")

    @classmethod
    def from_artifacts(
        cls,
        *,
        checkpoint_directory: str | Path,
        embedding_directory: str | Path,
        model_name: str,
        expected_prepared_fingerprint: str | None = None,
        expected_parent_fingerprint: str | None = None,
        expected_placement_fingerprint: str | None = None,
        seed: int = 42,
        device: str = "cpu",
    ) -> "NCMResponseModel":
        """Construct from existing artifacts with no legacy Python import."""

        from .ncm_artifacts import (
            load_ncm_checkpoint_artifact,
            load_ncm_item_embeddings,
        )

        for value, name in (
            (expected_prepared_fingerprint, "expected_prepared_fingerprint"),
            (expected_parent_fingerprint, "expected_parent_fingerprint"),
            (expected_placement_fingerprint, "expected_placement_fingerprint"),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be non-blank when provided")

        embeddings, source_embedding_fingerprint = load_ncm_item_embeddings(
            embedding_directory
        )
        manifest, predictor_state, checkpoint_fingerprint = (
            load_ncm_checkpoint_artifact(
                checkpoint_directory,
                expected_embedding_fingerprint=source_embedding_fingerprint,
                expected_prepared_fingerprint=expected_prepared_fingerprint,
                expected_parent_fingerprint=expected_parent_fingerprint,
                expected_placement_fingerprint=expected_placement_fingerprint,
                device=device,
            )
        )
        config = NCMClickModelConfig(**manifest["model_config"])
        predictor = build_ncm_model(config)
        predictor.load_state_dict(predictor_state, strict=True)
        predictor.to(torch.device(device))
        predictor.eval()

        expected_condition = NCM_ARTIFACT_CONDITION_BY_MODEL.get(model_name)
        if expected_condition is None:
            raise ValueError(f"unsupported NCM response model: {model_name}")
        if manifest.get("condition") != expected_condition:
            raise ValueError("NCM model name does not match checkpoint condition")
        model = cls(
            predictor=predictor,
            item_embeddings=embeddings,
            model_name=model_name,
            checkpoint_fingerprint=checkpoint_fingerprint,
            source_embedding_fingerprint=source_embedding_fingerprint,
            seed=seed,
        )
        model.checkpoint_manifest = dict(manifest)
        return model

    def _diagnostics(self, turn: TurnInput) -> Mapping[str, Any]:
        if len(turn.presented_slate) != 10 or len(set(turn.presented_slate)) != 10:
            raise ValueError("NCM response models require ten unique items")
        if not turn.observed_history:
            raise ValueError("NCM response models require non-empty history")
        try:
            item_values = np.stack(
                [self.item_embeddings[item_id] for item_id in turn.presented_slate]
            )
            history_values = np.stack(
                [self.item_embeddings[item_id] for item_id in turn.observed_history]
            )
        except KeyError as error:
            raise KeyError(f"missing embedding for item {error.args[0]}") from error
        user_value = history_values.mean(axis=0, dtype=np.float64).astype(np.float32)
        try:
            device = next(self.predictor.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        with torch.inference_mode():
            logits = self.predictor(
                torch.from_numpy(item_values).unsqueeze(0).to(device),
                torch.from_numpy(user_value).unsqueeze(0).to(device),
            )
            if logits.shape != (1, 10) or not bool(torch.isfinite(logits).all()):
                raise ValueError("predictor must emit ten finite logits")
            cascade = ncm_choice_probabilities(logits)

        conditioned = tuple(
            float(value)
            for value in cascade.conditioned_choice_probabilities.cpu().numpy()[0]
        )
        if not math.isclose(sum(conditioned), 1.0, abs_tol=1e-6):
            raise ValueError("conditioned click probabilities do not sum to one")
        return {
            "logits": tuple(float(value) for value in logits.cpu().numpy()[0]),
            "independent_click_probabilities": tuple(
                float(value)
                for value in cascade.independent_click_probabilities.cpu().numpy()[0]
            ),
            "ordered_choice_masses": tuple(
                float(value) for value in cascade.ordered_choice_masses.cpu().numpy()[0]
            ),
            "conditioned_choice_probabilities": conditioned,
            "implied_no_click_mass": float(cascade.no_click_masses.cpu().numpy()[0, 0]),
        }

    def respond(self, turn: TurnInput) -> ActionDecision:
        diagnostics = dict(self._diagnostics(turn))
        probabilities = diagnostics["conditioned_choice_probabilities"]
        rng = random_generator(
            self.seed,
            RandomKey(
                stream_id=2,
                user_id=turn.user_id,
                iteration=turn.iteration,
            ),
        )
        draw = float(rng.random())
        selected_index = min(
            int(np.searchsorted(np.cumsum(probabilities), draw, side="right")),
            len(turn.presented_slate) - 1,
        )
        diagnostics.update(
            random_draw=draw,
            selected_displayed_rank=selected_index + 1,
            checkpoint_fingerprint=self.checkpoint_fingerprint,
            embedding_fingerprint=self.embedding_fingerprint,
            source_embedding_fingerprint=self.source_embedding_fingerprint,
        )
        return ActionDecision(
            decision_id=f"{turn.interaction_id}__{self.model_name}",
            status=ActionStatus.VALID,
            action=UserAction.CLICK,
            clicked_item_id=turn.presented_slate[selected_index],
            diagnostics=diagnostics,
        )

    def update(
        self,
        action: ActionDecision,
        context: Mapping[str, Any],
    ) -> UserUpdateResult:
        del action, context
        return UserUpdateResult(
            update_id=f"{self.model_name}__no_latent_state_update",
            status=UpdateStatus.SKIPPED,
        )

    def decide_continuation(
        self,
        post_update_context: Mapping[str, Any],
    ) -> ContinuationDecision:
        del post_update_context
        return ContinuationDecision(
            decision_id=f"{self.model_name}__continuation_not_requested",
            status=ContinuationStatus.NOT_REQUESTED,
            continuation=Continuation.UNAVAILABLE,
        )

    def export_state(self) -> Mapping[str, Any]:
        return {
            "variant_id": self.VARIANT_ID,
            "model_config": asdict(self.predictor.config),
            "predictor_state": {
                key: value.detach().cpu().clone()
                for key, value in self.predictor.state_dict().items()
            },
            "model_name": self.model_name,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "embedding_fingerprint": self.embedding_fingerprint,
            "source_embedding_fingerprint": self.source_embedding_fingerprint,
            "seed": self.seed,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("variant_id") != self.VARIANT_ID:
            raise ValueError("NCM state variant mismatch")
        if state.get("model_config") != asdict(self.predictor.config):
            raise ValueError("NCM predictor architecture mismatch")
        if state.get("embedding_fingerprint") != self.embedding_fingerprint:
            raise ValueError("NCM embedding fingerprint mismatch")
        if (
            state.get("source_embedding_fingerprint")
            != self.source_embedding_fingerprint
        ):
            raise ValueError("NCM source embedding fingerprint mismatch")
        self.predictor.load_state_dict(state["predictor_state"], strict=True)
        self.predictor.eval()
        self.model_name = str(state["model_name"])
        self.checkpoint_fingerprint = str(state["checkpoint_fingerprint"])
        self.seed = int(state["seed"])

    def fingerprint(self) -> str:
        return fingerprint_semantic_state(self.export_state(), self.VARIANT_ID)


__all__ = [
    "NCMChoiceProbabilities",
    "NCMClickModelConfig",
    "NCMSlatewiseGRU",
    "NCMResponseModel",
    "PointwiseLogisticRegression",
    "build_ncm_model",
    "ncm_choice_probabilities",
]
