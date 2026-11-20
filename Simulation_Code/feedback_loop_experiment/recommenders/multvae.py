"""Self-contained MultVAE recommender for the feedback-loop simulator.

The network, objective, and initialization follow the MIT-licensed Agent4Rec
MultVAE implementation. Input histories use the binary user-item vectors from
the MultVAE paper. This module owns its fixed catalogue mapping, training,
inference, and checkpoint state; it has no runner-specific dependencies.

See ``LICENSE-MULTVAE-MIT`` in this directory for the applicable third-party
licence. SASRec's separate Apache-2.0 licence does not apply to this module.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Collection, Generator, Mapping, Sequence
from contextlib import contextmanager
import logging
import math
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..contracts import ItemId, UserId
from ..utility.fingerprints import fingerprint_semantic_state
from .base import FitReport, FitSpec, RecommendationBatch


_LOGGER = logging.getLogger(__name__)


def _multvae_loss(
    reconstructed_logits: Tensor,
    interactions: Tensor,
    mean: Tensor,
    log_variance: Tensor,
    *,
    anneal: float,
) -> Tensor:
    """Return multinomial negative log-likelihood plus annealed KL loss."""

    negative_log_likelihood = -torch.mean(
        torch.sum(
            F.log_softmax(reconstructed_logits, dim=1) * interactions,
            dim=1,
        )
    )
    kl_divergence = -0.5 * torch.mean(
        torch.sum(
            1 + log_variance - mean.pow(2) - log_variance.exp(),
            dim=1,
        )
    )
    return negative_log_likelihood + anneal * kl_divergence


class _MultVAEModel(nn.Module):
    """Agent4Rec-compatible MultVAE network."""

    def __init__(
        self,
        *,
        num_items: int,
        latent_dimension: int,
        hidden_dimension: int,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.p_dims = [latent_dimension, hidden_dimension, num_items]
        self.q_dims = list(reversed(self.p_dims))
        encoder_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList(
            nn.Linear(input_size, output_size)
            for input_size, output_size in zip(
                encoder_dims[:-1], encoder_dims[1:], strict=True
            )
        )
        self.p_layers = nn.ModuleList(
            nn.Linear(input_size, output_size)
            for input_size, output_size in zip(
                self.p_dims[:-1], self.p_dims[1:], strict=True
            )
        )
        self.dropout = nn.Dropout(dropout_rate)
        self._initialize_weights()

    def forward(self, interactions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_variance = self.encode(interactions)
        latent = self.reparameterize(mean, log_variance)
        return self.decode(latent), mean, log_variance

    def encode(self, interactions: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.dropout(F.normalize(interactions))
        for index, layer in enumerate(self.q_layers):
            hidden = layer(hidden)
            if index != len(self.q_layers) - 1:
                hidden = torch.tanh(hidden)
        latent_dimension = self.q_dims[-1]
        return hidden[:, :latent_dimension], hidden[:, latent_dimension:]

    def reparameterize(self, mean: Tensor, log_variance: Tensor) -> Tensor:
        if not self.training:
            return mean
        standard_deviation = torch.exp(0.5 * log_variance)
        noise = torch.randn_like(standard_deviation)
        return noise.mul(standard_deviation).add_(mean)

    def decode(self, latent: Tensor) -> Tensor:
        hidden = latent
        for index, layer in enumerate(self.p_layers):
            hidden = layer(hidden)
            if index != len(self.p_layers) - 1:
                hidden = torch.tanh(hidden)
        return hidden

    def _initialize_weights(self) -> None:
        with torch.no_grad():
            for layer in [*self.q_layers, *self.p_layers]:
                fan_out, fan_in = layer.weight.shape
                standard_deviation = math.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.normal_(0.0, standard_deviation)
                layer.bias.normal_(0.0, 0.001)


class MultVAEAdapter:
    """Train MultVAE and rank explicit eligible-item sets.
    """

    VARIANT_ID = "multvae_binary_v1"

    def __init__(
        self,
        *,
        catalog_item_ids: Collection[int],
        latent_dimension: int = 200,
        hidden_dimension: int = 600,
        dropout_rate: float = 0.5,
        learning_rate: float = 0.0005,
        batch_size: int = 2048,
        num_epochs: int = 2000,
        total_anneal_steps: int = 200_000,
        anneal_cap: float = 0.2,
        device: str = "auto",
        random_seed: int = 42,
    ) -> None:
        self._validate_hyperparameters(
            latent_dimension=latent_dimension,
            hidden_dimension=hidden_dimension,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            total_anneal_steps=total_anneal_steps,
            anneal_cap=anneal_cap,
            random_seed=random_seed,
        )
        configured_catalog = self._validate_catalog(catalog_item_ids)
        self._settings: dict[str, Any] = {
            "catalog_item_ids": configured_catalog,
            "latent_dimension": latent_dimension,
            "hidden_dimension": hidden_dimension,
            "dropout_rate": float(dropout_rate),
            "learning_rate": float(learning_rate),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "total_anneal_steps": total_anneal_steps,
            "anneal_cap": float(anneal_cap),
            "device": device,
            "random_seed": random_seed,
        }
        self.latent_dimension = latent_dimension
        self.hidden_dimension = hidden_dimension
        self.dropout_rate = float(dropout_rate)
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = float(anneal_cap)
        self.random_seed = random_seed
        self.device = self._resolve_device(device)
        self._configured_catalog = configured_catalog
        self._catalog_item_ids: tuple[int, ...] = ()
        self._item_to_column: dict[int, int] = {}
        self._histories: dict[int, tuple[int, ...]] = {}
        self._fallback_ranking: tuple[int, ...] = ()
        self._model: _MultVAEModel | None = None
        self._training_losses: tuple[float, ...] = ()
        self._training_updates = 0

    def fit(
        self,
        histories: Mapping[UserId, Sequence[ItemId]],
        spec: FitSpec,
    ) -> FitReport:
        started = time.perf_counter()
        validated_histories = self._validate_histories(histories)
        catalog = self._configured_catalog

        unknown_items = sorted(
            {
                item_id
                for history in validated_histories.values()
                for item_id in history
                if item_id not in set(catalog)
            }
        )
        if unknown_items:
            raise ValueError(
                "histories contain items outside catalog_item_ids: "
                f"{unknown_items}"
            )

        self._catalog_item_ids = catalog
        self._item_to_column = {
            item_id: column for column, item_id in enumerate(catalog)
        }
        self._histories = validated_histories
        self._fallback_ranking = self._rank_by_popularity(
            histories=validated_histories,
            catalog=catalog,
        )
        training_vectors = [
            self._history_vector(validated_histories[user_id])
            for user_id in sorted(validated_histories)
            if validated_histories[user_id]
        ]
        training_matrix = (
            torch.stack(training_vectors)
            if training_vectors
            else torch.empty((0, len(catalog)), dtype=torch.float32)
        )

        with self._seeded_torch_operations():
            self._model = _MultVAEModel(
                num_items=len(catalog),
                latent_dimension=self.latent_dimension,
                hidden_dimension=self.hidden_dimension,
                dropout_rate=self.dropout_rate,
            ).to(self.device)
            self._train_model(training_matrix)
        self._model.eval()

        return FitReport(
            checkpoint_fingerprint=self.fingerprint(),
            diagnostics={
                "model_name": "MultVAE",
                "variant_id": self.VARIANT_ID,
                "device": str(self.device),
                "n_catalog_items": len(self._catalog_item_ids),
                "n_training_users": sum(
                    bool(history) for history in self._histories.values()
                ),
                "training_updates": self._training_updates,
                "training_losses": self._training_losses,
                "fit_duration_seconds": time.perf_counter() - started,
                "fit_iteration": spec.iteration,
                "training_fingerprint": spec.training_fingerprint,
            },
        )

    def score_candidates(
        self,
        user_id: UserId,
        inference_history: Sequence[ItemId],
        candidate_ids: Sequence[ItemId],
    ) -> Sequence[float]:
        model = self._require_fitted_model()
        candidates = self._validate_candidates(candidate_ids)
        if not candidates:
            return ()
        history = self._validate_inference_history(inference_history)
        if not history:
            _LOGGER.info(
                "recommender_popularity_fallback_used",
                extra={
                    "event_name": "recommender_popularity_fallback_used",
                    "recommender_variant": self.VARIANT_ID,
                    "user_id": user_id,
                    "candidate_count": len(candidates),
                    "reason": "empty_inference_history",
                },
            )
            fallback_rank = {
                item_id: rank
                for rank, item_id in enumerate(self._fallback_ranking)
            }
            return tuple(float(-fallback_rank[item_id]) for item_id in candidates)

        model.eval()
        vector = self._history_vector(history).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            all_scores, _, _ = model(vector)
        scores = all_scores[0].cpu().numpy()
        return tuple(
            float(scores[self._item_to_column[item_id]])
            for item_id in candidates
        )

    def recommend(
        self,
        user_ids: Sequence[UserId],
        k: int,
        inference_histories: Mapping[UserId, Sequence[ItemId]],
        eligible_items: Mapping[UserId, Sequence[ItemId]],
    ) -> RecommendationBatch:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if len(user_ids) != len(set(user_ids)):
            raise ValueError("user_ids must be unique")

        recommendations: dict[int, tuple[int, ...]] = {}
        for user_id in user_ids:
            if user_id not in inference_histories or user_id not in eligible_items:
                raise KeyError(f"missing inference inputs for user {user_id}")
            candidates = self._validate_candidates(eligible_items[user_id])
            scores = np.asarray(
                self.score_candidates(
                    user_id,
                    inference_histories[user_id],
                    candidates,
                ),
                dtype=np.float64,
            )
            item_values = np.asarray(candidates, dtype=np.int64)
            order = np.lexsort((item_values, -scores))
            recommendations[int(user_id)] = tuple(
                int(item_values[index]) for index in order[:k]
            )

        return RecommendationBatch(
            items_by_user=recommendations,
            recommender_fingerprint=self.fingerprint(),
        )

    def export_state(self) -> Mapping[str, Any]:
        model_state = None
        if self._model is not None:
            model_state = {
                key: value.detach().cpu().clone()
                for key, value in self._model.state_dict().items()
            }
        return {
            "variant_id": self.VARIANT_ID,
            "settings": copy.deepcopy(self._settings),
            "catalog_item_ids": self._catalog_item_ids,
            "histories": copy.deepcopy(self._histories),
            "fallback_ranking": self._fallback_ranking,
            "training_losses": self._training_losses,
            "training_updates": self._training_updates,
            "model_state": model_state,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("variant_id") != self.VARIANT_ID:
            raise ValueError("MultVAE state variant mismatch")
        settings = copy.deepcopy(dict(state["settings"]))
        self.__init__(**settings)
        self._catalog_item_ids = tuple(state["catalog_item_ids"])
        self._item_to_column = {
            item_id: column
            for column, item_id in enumerate(self._catalog_item_ids)
        }
        self._histories = {
            int(user_id): tuple(history)
            for user_id, history in state["histories"].items()
        }
        self._fallback_ranking = tuple(state["fallback_ranking"])
        self._training_losses = tuple(float(value) for value in state["training_losses"])
        self._training_updates = int(state["training_updates"])

        model_state = state.get("model_state")
        if model_state is None:
            return
        if not self._catalog_item_ids:
            raise ValueError("MultVAE model state requires a fitted catalog")
        model = _MultVAEModel(
            num_items=len(self._catalog_item_ids),
            latent_dimension=self.latent_dimension,
            hidden_dimension=self.hidden_dimension,
            dropout_rate=self.dropout_rate,
        ).to(self.device)
        model.load_state_dict(model_state, strict=True)
        model.eval()
        self._model = model

    def clone(self) -> "MultVAEAdapter":
        cloned = type(self)(**self._settings)
        cloned.restore_state(self.export_state())
        return cloned

    def fingerprint(self) -> str:
        return fingerprint_semantic_state(self.export_state(), self.VARIANT_ID)

    def _train_model(self, training_matrix: Tensor) -> None:
        model = self._require_fitted_model()
        self._training_updates = 0
        losses: list[float] = []
        if not len(training_matrix):
            self._training_losses = ()
            return

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.random_seed)
        model.train()
        for _ in range(self.num_epochs):
            epoch_loss = 0.0
            batch_count = 0
            indices = rng.permutation(len(training_matrix))
            for start in range(0, len(training_matrix), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                batch = training_matrix[batch_indices].to(self.device)
                if self.total_anneal_steps > 0:
                    anneal = min(
                        self.anneal_cap,
                        1.*self._training_updates / self.total_anneal_steps,
                    )
                else:
                    anneal = self.anneal_cap
                reconstructed, mean, log_variance = model(batch)
                loss = _multvae_loss(
                    reconstructed,
                    batch,
                    mean,
                    log_variance,
                    anneal=anneal,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                batch_count += 1
                self._training_updates += 1
            losses.append(epoch_loss / batch_count)
        self._training_losses = tuple(losses)

    def _history_vector(self, history: Sequence[int]) -> Tensor:
        vector = torch.zeros(len(self._catalog_item_ids), dtype=torch.float32)
        for item_id in set(history):
            vector[self._item_to_column[item_id]] = 1.0 # can be extended to counts (see method https://arxiv.org/pdf/1802.05814)
        return vector

    def _validate_candidates(
        self,
        candidate_ids: Sequence[ItemId],
    ) -> tuple[int, ...]:
        candidates = tuple(candidate_ids)
        if any(
            isinstance(item_id, bool) or not isinstance(item_id, int)
            for item_id in candidates
        ):
            raise TypeError("candidate IDs must be integers")
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate IDs must be unique")
        unknown = sorted(set(candidates).difference(self._catalog_item_ids))
        if unknown:
            raise ValueError(f"candidate IDs lie outside fitted catalog: {unknown}")
        return candidates

    def _validate_inference_history(
        self,
        history: Sequence[ItemId],
    ) -> tuple[int, ...]:
        values = tuple(history)
        if any(
            isinstance(item_id, bool) or not isinstance(item_id, int)
            for item_id in values
        ):
            raise TypeError("inference-history item IDs must be integers")
        unknown = sorted(set(values).difference(self._catalog_item_ids))
        if unknown:
            raise ValueError(f"history contains items outside fitted catalog: {unknown}")
        return values

    def _require_fitted_model(self) -> _MultVAEModel:
        if self._model is None:
            raise RuntimeError("MultVAEAdapter must be fitted first")
        return self._model
    
    @contextmanager
    def _seeded_torch_operations(self) -> Generator[None, None, None]:
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices.append(
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.random_seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(self.random_seed)
            yield

    @staticmethod
    def _rank_by_popularity(
        *,
        histories: Mapping[int, tuple[int, ...]],
        catalog: tuple[int, ...],
    ) -> tuple[int, ...]:
        counts = Counter(
            item_id for history in histories.values() for item_id in history
        )
        return tuple(
            sorted(catalog, key=lambda item_id: (-counts[item_id], item_id))
        )

    @staticmethod
    def _validate_catalog(catalog_item_ids: Collection[int]) -> tuple[int, ...]:
        values = tuple(catalog_item_ids)
        if not values:
            raise ValueError("catalog_item_ids cannot be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise TypeError("catalog item IDs must be integers")
        if any(value <= 0 for value in values):
            raise ValueError("catalog item IDs must be positive")
        if len(values) != len(set(values)):
            raise ValueError("catalog_item_ids cannot contain duplicates")
        return tuple(sorted(values))

    @staticmethod
    def _validate_histories(
        histories: Mapping[UserId, Sequence[ItemId]],
    ) -> dict[int, tuple[int, ...]]:
        if not isinstance(histories, Mapping):
            raise TypeError("histories must be a mapping")
        result: dict[int, tuple[int, ...]] = {}
        for user_id, history in histories.items():
            if isinstance(user_id, bool) or not isinstance(user_id, int):
                raise TypeError("user IDs must be integers")
            values = tuple(history)
            if any(
                isinstance(item_id, bool) or not isinstance(item_id, int)
                for item_id in values
            ):
                raise TypeError("history item IDs must be integers")
            if any(item_id <= 0 for item_id in values):
                raise ValueError("history item IDs must be positive")
            result[user_id] = values
        if not result:
            raise ValueError("histories cannot be empty")
        return result

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if not isinstance(device, str):
            raise TypeError("device must be a string")
        normalized = device.lower()
        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if normalized == "cuda" or normalized.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            return torch.device(normalized)
        if normalized == "cpu":
            return torch.device("cpu")
        raise ValueError(f"unsupported device: {device}")

    @staticmethod
    def _validate_hyperparameters(**values: object) -> None:
        for name in (
            "latent_dimension",
            "hidden_dimension",
            "batch_size",
            "num_epochs",
        ):
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        total_steps = values["total_anneal_steps"]
        if (
            isinstance(total_steps, bool)
            or not isinstance(total_steps, int)
            or total_steps < 0
        ):
            raise ValueError("total_anneal_steps must be a non-negative integer")
        random_seed = values["random_seed"]
        if (
            isinstance(random_seed, bool)
            or not isinstance(random_seed, int)
            or random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")
        for name in ("dropout_rate", "learning_rate", "anneal_cap"):
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= float(values["dropout_rate"]) < 1:
            raise ValueError("dropout_rate must lie in [0, 1)")
        if float(values["learning_rate"]) <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= float(values["anneal_cap"]) <= 1:
            raise ValueError("anneal_cap must lie in [0, 1]")
