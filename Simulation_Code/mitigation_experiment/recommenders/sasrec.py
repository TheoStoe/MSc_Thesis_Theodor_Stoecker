"""SASRec adapted from ``pmixer/SASRec.pytorch`` to simulator contracts.

The model classes below follow the Apache-2.0-licensed PyTorch implementation
at the pinned source recorded in :data:`PMIXER_MODEL_URL`. 
The surrounding adapter is local: it replaces the upstream dataset files and
multiprocessing sampler with explicit histories, catalog-aware deterministic
sampling, checkpointing, and the simulator's candidate-ranking interface.

See ``SASREC_UPSTREAM.md`` and ``LICENSE-SASREC-APACHE-2.0`` in this directory
for provenance, deliberate adaptations, and licensing.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
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


PMIXER_REVISION = "fde8a9c08febc2c898dc269b1d98ab762c50d2be"
PMIXER_MODEL_URL = (
    "https://github.com/pmixer/SASRec.pytorch/blob/"
    f"{PMIXER_REVISION}/python/model.py"
)
PMIXER_UTILS_URL = (
    "https://github.com/pmixer/SASRec.pytorch/blob/"
    f"{PMIXER_REVISION}/python/utils.py"
)

_LOGGER = logging.getLogger(__name__)


def _sasrec_loss(
    positive_logits: Tensor,
    negative_logits: Tensor,
    positive_items: Tensor,
    positive_weights: Tensor | None = None,
) -> Tensor:
    """Original BCE, optionally weighting only observed positive targets.

    Weights are raw multipliers, not normalized. Padding and sampled-negative
    BCE remain unchanged. Unit weights take the exact original reduction path.
    """

    target_mask = positive_items.ne(0)
    positive_labels = torch.ones_like(positive_logits)
    negative_labels = torch.zeros_like(negative_logits)
    bce_criterion = nn.BCEWithLogitsLoss()

    if positive_weights is not None:
        if positive_weights.shape != positive_items.shape:
            raise ValueError("positive weights must match positive target shape")
        selected_weights = positive_weights[target_mask]
        if not bool(torch.isfinite(selected_weights).all()) or bool(
            (selected_weights <= 0).any()
        ):
            raise ValueError("positive target weights must be finite and positive")
    else:
        selected_weights = None
    if selected_weights is None or bool((selected_weights == 1).all()):
        loss = bce_criterion(
            positive_logits[target_mask], positive_labels[target_mask]
        )
    else:
        positive_loss = F.binary_cross_entropy_with_logits(
            positive_logits[target_mask], positive_labels[target_mask], reduction="none"
        )
        loss = (positive_loss * selected_weights).mean()
    loss += bce_criterion(
        negative_logits[target_mask], negative_labels[target_mask]
    )
    return loss


class _PointWiseFeedForward(nn.Module):
    """The two kernel-size-one convolutions from pmixer ``model.py``."""

    def __init__(self, hidden_size: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs: Tensor) -> Tensor:
        # (B, T, H) -> (B, H, T), as required by Conv1d.
        outputs = self.conv1(inputs.transpose(-1, -2))
        # All point-wise convolution stages retain (B, H, T).
        outputs = self.dropout1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        # Restore the SASRec sequence layout: (B, T, H).
        return outputs.transpose(-1, -2)


class _SASRecModel(nn.Module):
    """Reviewable adaptation of ``pmixer/SASRec.pytorch/python/model.py``."""

    def __init__(
        self,
        *,
        num_items: int,
        max_sequence_length: int,
        hidden_size: int,
        num_blocks: int,
        num_heads: int,
        dropout_rate: float,
        norm_first: bool,
    ) -> None:
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.norm_first = norm_first
        self.item_emb = nn.Embedding(
            num_items + 1,
            hidden_size,
            padding_idx=0,
        )
        self.pos_emb = nn.Embedding(
            max_sequence_length + 1,
            hidden_size,
            padding_idx=0,
        )
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList() #Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_size, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(
                nn.LayerNorm(hidden_size, eps=1e-8)
            )
            self.attention_layers.append(
                nn.MultiheadAttention(
                    hidden_size,
                    num_heads,
                    dropout_rate,
                )
            )
            self.forward_layernorms.append(
                nn.LayerNorm(hidden_size, eps=1e-8)
            )
            self.forward_layers.append(
                _PointWiseFeedForward(hidden_size, dropout_rate)
            )
        self._initialize_like_upstream()

    def _initialize_like_upstream(self) -> None:
        """Apply the initialization performed by pmixer ``python/main.py`` line59 following."""

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)
        with torch.no_grad():
            self.pos_emb.weight[0].zero_()
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequences: Tensor) -> Tensor:
        """Encode left-padded item sequences, matching pmixer ``log2feats``."""

        if sequences.ndim != 2:
            raise ValueError("SASRec sequences must have shape (batch, sequence)")
        if sequences.shape[1] != self.max_sequence_length:
            raise ValueError(
                "SASRec sequence width must equal max_sequence_length"
            )

        # B = batch, T = max sequence length, H = hidden size.
        # sequences: (B, T) -> values: (B, T, H)
        values = self.item_emb(sequences)
        values = values * self.item_emb.embedding_dim**0.5
        # (1, T) -> (B, T), with padding positions reset to 0.
        positions = torch.arange(
            1,
            self.max_sequence_length + 1,
            device=sequences.device,
        ).unsqueeze(0).expand(sequences.shape[0], -1)
        positions = positions * sequences.ne(0)
        # position embeddings: (B, T, H); values stays (B, T, H).
        values = values + self.pos_emb(positions)
        values = self.emb_dropout(values)

        # Causal attention mask: (T, T).
        attention_mask = torch.triu(
            torch.ones(
                (self.max_sequence_length, self.max_sequence_length),
                dtype=torch.bool,
                device=sequences.device,
            ),
            diagonal=1,
        )
        for index, attention in enumerate(self.attention_layers):
            # PyTorch MultiheadAttention expects (T, B, H).
            transposed = values.transpose(0, 1)
            if self.norm_first:
                normalized = self.attention_layernorms[index](transposed)
                # attended: (T, B, H).
                attended, _ = attention(
                    normalized,
                    normalized,
                    normalized,
                    attn_mask=attention_mask,
                    need_weights=False,
                )
                # Residual and feed-forward paths return to (B, T, H).
                values = (transposed + attended).transpose(0, 1)
                values = values + self.forward_layers[index](
                    self.forward_layernorms[index](values)
                )
            else:
                # attended: (T, B, H).
                attended, _ = attention(
                    transposed,
                    transposed,
                    transposed,
                    attn_mask=attention_mask,
                    need_weights=False,
                )
                # Attention normalization restores (B, T, H).
                values = self.attention_layernorms[index](
                    transposed + attended
                ).transpose(0, 1)
                # Feed-forward residual and normalization stay (B, T, H).
                values = self.forward_layernorms[index](
                    values + self.forward_layers[index](values)
                )
        # Encoded sequence features: (B, T, H).
        return self.last_layernorm(values)

    def forward(
        self,
        sequences: Tensor,
        positive_items: Tensor,
        negative_items: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # features and target embeddings: (B, T, H).
        features = self.log2feats(sequences)
        positive_embeddings = self.item_emb(positive_items)
        negative_embeddings = self.item_emb(negative_items)
        # One positive and negative logit per timestep: (B, T).
        return (
            (features * positive_embeddings).sum(dim=-1),
            (features * negative_embeddings).sum(dim=-1),
        )

    def predict(
        self,
        sequences: Tensor,
        candidate_items: Tensor,
    ) -> Tensor:
        # final_features: (B, H); candidate_embeddings: (I, H).
        final_features = self.log2feats(sequences)[:, -1, :]
        candidate_embeddings = self.item_emb(candidate_items)
        # Candidate logits for every user: (B, I).
        return (candidate_embeddings @ final_features.unsqueeze(-1)).squeeze(-1)


@dataclass(frozen=True)
class _TrainingExample:
    sequence: tuple[int, ...]
    positive_items: tuple[int, ...]
    negative_candidates: tuple[int, ...]
    positive_weights: tuple[float, ...] = ()
    simulated_targets: tuple[bool, ...] = ()


class SASRecAdapter:
    """Train SASRec and rank explicit eligible items for explicit histories.

    Item ID ``0`` is reserved for padding. The model's shared weights are
    learned by :meth:`fit`, while the history supplied to :meth:`recommend`
    is used only for that inference call. Consequently, a valid history can
    be scored for a user who was not part of the fitted population.
    """

    VARIANT_ID = "sasrec_pmixer_pytorch_v1"

    def __init__(
        self,
        *,
        catalog_item_ids: Sequence[int] | None = None,
        max_sequence_length: int = 200,
        hidden_size: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.2,
        norm_first: bool = False,
        learning_rate: float = 0.001,
        embedding_l2: float = 0.0,
        batch_size: int = 128,
        num_epochs: int = 201,
        device: str = "auto",
        random_seed: int = 42,
    ) -> None:
        self._validate_hyperparameters(
            max_sequence_length=max_sequence_length,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            norm_first=norm_first,
            learning_rate=learning_rate,
            embedding_l2=embedding_l2,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer")

        configured_catalog = (
            self._validate_catalog(catalog_item_ids)
            if catalog_item_ids is not None
            else None
        )
        self._settings: dict[str, Any] = {
            "catalog_item_ids": configured_catalog,
            "max_sequence_length": max_sequence_length,
            "hidden_size": hidden_size,
            "num_blocks": num_blocks,
            "num_heads": num_heads,
            "dropout_rate": dropout_rate,
            "norm_first": norm_first,
            "learning_rate": learning_rate,
            "embedding_l2": embedding_l2,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "device": device,
            "random_seed": random_seed,
        }
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.norm_first = norm_first
        self.learning_rate = learning_rate
        self.embedding_l2 = embedding_l2
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_seed = random_seed
        self.device = self._resolve_device(device)

        self._configured_catalog = configured_catalog
        self._catalog_item_ids: tuple[int, ...] = ()
        self._item_to_index: dict[int, int] = {}
        self._histories: dict[int, tuple[int, ...]] = {}
        self._fallback_ranking: tuple[int, ...] = ()
        self._model: _SASRecModel | None = None
        self._training_losses: tuple[float, ...] = ()
        self._training_updates = 0
        self._mitigation_weight_config: dict[str, Any] | None = None
        self._mitigation_last_fit: dict[str, Any] = {}
        self._fit_context: dict[str, Any] = {}

    @property
    def catalog_item_ids(self) -> tuple[int, ...]:
        """Return the fitted catalog in deterministic item-ID order."""

        return self._catalog_item_ids

    @property
    def training_losses(self) -> tuple[float, ...]:
        """Return the mean loss for every epoch of the latest fit."""

        return self._training_losses

    @property
    def mitigation_weight_config(self) -> Mapping[str, Any] | None:
        """Return the fixed positive-target intervention, or None if disabled."""

        return copy.deepcopy(self._mitigation_weight_config)

    def configure_mitigation_weights(
        self,
        head_item_ids: Collection[ItemId],
        non_head_weight: float = 1.25,
        scope: str = "all",
        initial_history_lengths: Mapping[UserId, int] | None = None,
    ) -> None:
        """Enable weighting for future refits after loading the common model.

        This does not train or alter the loaded model's parameters. Under
        ``simulated_only``, only appended target positions beyond each user's
        initial history length receive the non-head multiplier. Both scopes
        use the same fixed head set and leave sampled negatives unchanged.
        """

        self._require_fitted_model()
        heads = tuple(head_item_ids)
        self._validate_item_ids(heads, context="mitigation head item IDs")
        if len(set(heads)) != len(heads):
            raise ValueError("mitigation head item IDs must be unique")
        if not set(heads).issubset(self._catalog_item_ids):
            raise ValueError("mitigation head items must belong to the fitted catalog")
        if isinstance(non_head_weight, bool) or not isinstance(non_head_weight, (int, float)):
            raise TypeError("non_head_weight must be numeric")
        if not math.isfinite(non_head_weight) or non_head_weight < 1:
            raise ValueError("non_head_weight must be finite and at least one")
        if scope not in {"all", "simulated_only"}:
            raise ValueError("mitigation weight scope must be all or simulated_only")
        lengths = (
            {user_id: len(history) for user_id, history in self._histories.items()}
            if initial_history_lengths is None else dict(initial_history_lengths)
        )
        if set(lengths) != set(self._histories):
            raise ValueError("initial history lengths must cover the fixed training cohort")
        for user_id, length in lengths.items():
            self._validate_user_id(user_id)
            if isinstance(length, bool) or not isinstance(length, int):
                raise TypeError("initial history lengths must be integers")
            if length < 0 or length > len(self._histories[user_id]):
                raise ValueError("initial history lengths are outside the fitted histories")
        config = {
            "schema_version": "sasrec-positive-head-weight-v1",
            "head_item_ids": tuple(sorted(heads)),
            "head_weight": 1.0,
            "non_head_weight": float(non_head_weight),
            "scope": scope,
            "normalization": "none",
            "initial_history_lengths": dict(sorted(lengths.items())),
        }
        if self._mitigation_weight_config is not None and self._mitigation_weight_config != config:
            raise ValueError("mitigation weighting is fixed; restore the shared model before changing it")
        self._mitigation_weight_config = config

    def fit(
        self,
        histories: Mapping[UserId, Sequence[ItemId]],
        spec: FitSpec,
    ) -> FitReport:
        started = time.perf_counter()
        validated_histories = self._validate_histories(histories)
        self._fit_context = {
            "fit_iteration": spec.iteration,
            "training_fingerprint": spec.training_fingerprint,
            "random_seed": self.random_seed,
        }
        if self._mitigation_weight_config is not None:
            if spec.iteration <= 0:
                raise ValueError("initial SASRec fit must remain unweighted; load it before configuring mitigation")
            initial_lengths = self._mitigation_weight_config["initial_history_lengths"]
            if set(initial_lengths) != set(validated_histories):
                raise ValueError("mitigation refits require the fixed training cohort")
            if any(len(validated_histories[u]) < n for u, n in initial_lengths.items()):
                raise ValueError("mitigation histories cannot shrink below their initial lengths")
        catalog = self._configured_catalog
        if catalog is None:
            catalog = tuple(
                sorted(
                    {
                        item_id
                        for history in validated_histories.values()
                        for item_id in history
                    }
                )
            )
            if not catalog:
                raise ValueError("Cannot infer a catalog from empty user histories")

        catalog_set = set(catalog)
        unknown_items = sorted(
            {
                item_id
                for history in validated_histories.values()
                for item_id in history
                if item_id not in catalog_set
            }
        )
        if unknown_items:
            raise ValueError(
                "Histories contain items outside catalog_item_ids: "
                f"{unknown_items}"
            )

        self._catalog_item_ids = catalog
        self._item_to_index = {
            item_id: index
            for index, item_id in enumerate(catalog, start=1)
        }
        self._histories = validated_histories
        self._fallback_ranking = self._rank_by_popularity(
            histories=validated_histories,
            catalog=catalog,
        )
        indexed_histories = {
            user_id: tuple(self._item_to_index[item_id] for item_id in history)
            for user_id, history in validated_histories.items()
        }
        indexed_catalog = tuple(range(1, len(catalog) + 1))
        training_examples = self._build_training_examples(
            histories=indexed_histories,
            catalog=indexed_catalog,
        )
        try:
            with self._seeded_torch_operations():
                self._model = _SASRecModel(
                    num_items=len(catalog),
                    max_sequence_length=self.max_sequence_length,
                    hidden_size=self.hidden_size,
                    num_blocks=self.num_blocks,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    norm_first=self.norm_first,
                ).to(self.device)
                self._train_model(training_examples)
        except Exception:
            _LOGGER.exception("sasrec_fit_failed", extra={"event_name": "sasrec_fit_failed", **self._fit_context})
            raise

        duration = time.perf_counter() - started
        return FitReport(
            checkpoint_fingerprint=self.fingerprint(),
            diagnostics={
                "model_name": "SASRec",
                "implementation": "pmixer/SASRec.pytorch",
                "upstream_revision": PMIXER_REVISION,
                "norm_first": self.norm_first,
                "device": str(self.device),
                "n_catalog_items": len(self._catalog_item_ids),
                "n_training_users": sum(
                    len(history) >= 2 for history in self._histories.values()
                ),
                "training_updates": self._training_updates,
                "training_losses": list(self._training_losses),
                "fit_duration_seconds": duration,
                "fit_iteration": spec.iteration,
                "training_fingerprint": spec.training_fingerprint,
                "mitigation_weighting": copy.deepcopy(self._mitigation_weight_config),
                "mitigation_target_diagnostics": copy.deepcopy(self._mitigation_last_fit),
            },
        )

    def score_candidates(
        self,
        user_id: UserId,
        inference_history: Sequence[ItemId],
        candidate_ids: Sequence[ItemId],
    ) -> Sequence[float]:
        self._validate_user_id(user_id)
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
        indexed_history = tuple(self._item_to_index[item_id] for item_id in history)
        indexed_candidates = tuple(
            self._item_to_index[item_id] for item_id in candidates
        )
        sequence = torch.as_tensor(
            [self._inference_sequence(indexed_history)],
            dtype=torch.long,
            device=self.device,
        )
        candidate_tensor = torch.as_tensor(
            indexed_candidates,
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            scores = model.predict(sequence, candidate_tensor)[0]
        return tuple(float(value) for value in scores.cpu().numpy())

    def recommend(
        self,
        user_ids: Sequence[UserId],
        k: int,
        inference_histories: Mapping[UserId, Sequence[ItemId]],
        eligible_items: Mapping[UserId, Sequence[ItemId]],
    ) -> RecommendationBatch:
        self._require_fitted_model()
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be positive")

        result: dict[int, tuple[int, ...]] = {}
        for user_id in user_ids:
            self._validate_user_id(user_id)
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
            result[user_id] = tuple(
                int(item_values[index]) for index in order[:k]
            )

        return RecommendationBatch(
            items_by_user=result,
            recommender_fingerprint=self.fingerprint(),
        )

    def export_state(self) -> Mapping[str, Any]:
        model_state = None
        if self._model is not None:
            model_state = {
                key: value.detach().cpu().clone()
                for key, value in self._model.state_dict().items()
            }
        state = {
            "variant_id": self.VARIANT_ID,
            "settings": copy.deepcopy(self._settings),
            "catalog_item_ids": tuple(self._catalog_item_ids),
            "histories": copy.deepcopy(self._histories),
            "fallback_ranking": tuple(self._fallback_ranking),
            "training_losses": tuple(self._training_losses),
            "training_updates": self._training_updates,
            "model_state": model_state,
        }
        # Omit absent intervention fields to preserve old unweighted states and
        # fingerprints, including the shared initial notebook checkpoints.
        if self._mitigation_weight_config is not None:
            state["mitigation_weight_config"] = copy.deepcopy(self._mitigation_weight_config)
            state["mitigation_last_fit"] = copy.deepcopy(self._mitigation_last_fit)
        return state

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("variant_id") != self.VARIANT_ID:
            raise ValueError("SASRec state variant mismatch")

        settings = copy.deepcopy(dict(state["settings"]))
        self.__init__(**settings)
        raw_catalog = tuple(state["catalog_item_ids"])
        self._catalog_item_ids = (
            self._validate_catalog(raw_catalog) if raw_catalog else ()
        )
        self._item_to_index = {
            item_id: index
            for index, item_id in enumerate(self._catalog_item_ids, start=1)
        }
        raw_histories = dict(state["histories"])
        self._histories = (
            self._validate_histories(raw_histories) if raw_histories else {}
        )
        self._fallback_ranking = self._validate_restored_fallback(
            state["fallback_ranking"],
            self._catalog_item_ids,
        )
        self._training_losses = tuple(float(value) for value in state["training_losses"])
        self._training_updates = int(state["training_updates"])

        model_state = state.get("model_state")
        if model_state is not None:
            if not self._catalog_item_ids:
                raise ValueError("SASRec model state requires a fitted catalog")
            model = _SASRecModel(
                num_items=len(self._catalog_item_ids),
                max_sequence_length=self.max_sequence_length,
                hidden_size=self.hidden_size,
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                norm_first=self.norm_first,
            ).to(self.device)
            model.load_state_dict(model_state, strict=True)
            model.eval()
            self._model = model
            self._check_finite_parameters(model, stage="restore")
        mitigation = state.get("mitigation_weight_config")
        if mitigation is not None:
            if (
                mitigation.get("schema_version") != "sasrec-positive-head-weight-v1"
                or mitigation.get("head_weight") != 1.0
                or mitigation.get("normalization") != "none"
            ):
                raise ValueError("unsupported SASRec mitigation weight configuration")
            self.configure_mitigation_weights(
                mitigation["head_item_ids"],
                non_head_weight=mitigation["non_head_weight"],
                scope=mitigation["scope"],
                initial_history_lengths=mitigation["initial_history_lengths"],
            )
            self._mitigation_last_fit = copy.deepcopy(dict(state.get("mitigation_last_fit", {})))

    def clone(self) -> "SASRecAdapter":
        cloned = type(self)(**copy.deepcopy(self._settings))
        cloned.restore_state(self.export_state())
        return cloned

    def fingerprint(self) -> str:
        return fingerprint_semantic_state(self.export_state(), self.VARIANT_ID)

    def _train_model(self, examples: Sequence[_TrainingExample]) -> None:
        model = self._require_fitted_model()
        self._training_updates = 0
        self._training_losses = ()
        self._mitigation_last_fit = self._weight_diagnostics(examples)
        if not examples:
            _LOGGER.warning("sasrec_no_training_examples", extra={"event_name": "sasrec_no_training_examples", **self._fit_context})
            return
        rng = np.random.default_rng(self.random_seed)

        sequences = np.asarray(
            [example.sequence for example in examples],
            dtype=np.int64,
        )
        positive_items = np.asarray(
            [example.positive_items for example in examples],
            dtype=np.int64,
        )
        positive_weights = (
            np.asarray([example.positive_weights for example in examples], dtype=np.float32)
            if self._mitigation_weight_config is not None else None
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
        )
        losses: list[float] = []
        num_batches = (
            len(examples) + self.batch_size - 1
        ) // self.batch_size

        model.train()
        self._check_finite_parameters(model, stage="initialization")
        for epoch in range(1, self.num_epochs + 1):
            epoch_started = time.monotonic()
            self._fit_context.update(epoch=epoch, batch=None)
            negative_items = self._sample_negative_items(
                examples=examples,
                rng=rng,
            )

            shuffled_indices = rng.permutation(len(examples))
            epoch_loss = 0.0

            for step in range(num_batches):
                self._fit_context["batch"] = step + 1
                batch_start = step * self.batch_size
                batch_indices = shuffled_indices[
                    batch_start : batch_start + self.batch_size
                ]
                sequence_batch = torch.as_tensor(
                    sequences[batch_indices],
                    dtype=torch.long,
                    device=self.device,
                )
                positive_batch = torch.as_tensor(
                    positive_items[batch_indices],
                    dtype=torch.long,
                    device=self.device,
                )
                negative_batch = torch.as_tensor(
                    negative_items[batch_indices],
                    dtype=torch.long,
                    device=self.device,
                )

                # pmixer: ``pos_logits, neg_logits = model(...)``.
                positive_logits, negative_logits = model(
                    sequence_batch,
                    positive_batch,
                    negative_batch,
                )
                loss = _sasrec_loss(
                    positive_logits,
                    negative_logits,
                    positive_batch,
                    None if positive_weights is None else torch.as_tensor(
                        positive_weights[batch_indices],
                        dtype=positive_logits.dtype,
                        device=self.device,
                    ),
                )
                if self.embedding_l2:
                    # Deliberate existing local behavior: squared L2. 
                    loss = loss + self.embedding_l2 * torch.sum(
                        model.item_emb.weight**2
                    )

                if not bool(torch.isfinite(loss)):
                    self._numerical_failure("loss", "non-finite minibatch loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._check_finite_parameters(model, stage="gradient", gradients=True)
                optimizer.step()
                self._check_finite_parameters(model, stage="optimizer_step")
                epoch_loss += float(loss.detach().cpu())
                self._training_updates += 1
            losses.append(epoch_loss / num_batches)
            self._training_losses = tuple(losses)
            _LOGGER.info(
                "sasrec_epoch_completed",
                extra={
                    "event_name": "sasrec_epoch_completed",
                    **self._fit_context,
                    "training_loss": losses[-1],
                    "duration_seconds": time.monotonic() - epoch_started,
                    "weighted_positive_targets": self._mitigation_last_fit.get("weighted_positive_targets", 0),
                    "weight_scope": None if self._mitigation_weight_config is None else self._mitigation_weight_config["scope"],
                },
            )
        self._training_losses = tuple(losses)

    def _numerical_failure(self, stage: str, detail: str) -> None:
        _LOGGER.error(
            "sasrec_numerical_failure",
            extra={"event_name": "sasrec_numerical_failure", "stage": stage, "detail": detail, **self._fit_context},
        )
        raise FloatingPointError(f"SASRec {stage}: {detail}; fit context={self._fit_context}")

    def _check_finite_parameters(self, model: nn.Module, *, stage: str, gradients: bool = False) -> None:
        values = [
            (name, parameter.grad if gradients else parameter)
            for name, parameter in model.named_parameters()
            if not gradients or parameter.grad is not None
        ]
        flags = [torch.isfinite(value).all() for _, value in values]
        # Synchronize CUDA once on the normal path, not once per parameter.
        if flags and not bool(torch.stack(flags).all()):
            name = next(name for (name, _), finite in zip(values, flags) if not bool(finite))
            self._numerical_failure(stage, f"non-finite {'gradient' if gradients else 'parameter'} {name}")

    def _weight_diagnostics(self, examples: Sequence[_TrainingExample]) -> dict[str, Any]:
        targets = [
            (item, weight, simulated)
            for example in examples
            for item, weight, simulated in zip(
                example.positive_items,
                example.positive_weights or (1.0,) * len(example.positive_items),
                example.simulated_targets or (False,) * len(example.positive_items),
            ) if item != 0
        ]
        weights = np.asarray([entry[1] for entry in targets], dtype=float)
        simulated = sum(entry[2] for entry in targets)
        return {
            "positive_targets": len(targets),
            "weighted_positive_targets": int(np.count_nonzero(weights != 1)),
            "simulated_positive_targets": simulated if self._mitigation_weight_config is not None else None,
            "simulated_target_share": simulated / len(targets) if targets and self._mitigation_weight_config is not None else None,
            "positive_weight_min": float(weights.min()) if len(weights) else None,
            "positive_weight_max": float(weights.max()) if len(weights) else None,
            "positive_weight_mean": float(weights.mean()) if len(weights) else None,
            "positive_weight_effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()) if len(weights) else 0.0,
            "negative_weight": 1.0,
            "normalization": "none",
        }

    def _build_training_examples(
        self,
        *,
        histories: Mapping[int, tuple[int, ...]],
        catalog: tuple[int, ...],
    ) -> list[_TrainingExample]:
        catalog_set = set(catalog)
        examples: list[_TrainingExample] = []
        config = self._mitigation_weight_config
        indexed_heads = set() if config is None else {
            self._item_to_index[item] for item in config["head_item_ids"]
        }
        for user_id, history in histories.items():
            if len(history) < 2:
                continue
            negative_candidates = tuple(sorted(catalog_set.difference(history)))
            if not negative_candidates:
                continue

            sequence = [0] * self.max_sequence_length
            positive_items = [0] * self.max_sequence_length
            positive_weights = [1.0] * self.max_sequence_length
            simulated_targets = [False] * self.max_sequence_length
            next_item = history[-1]
            index = self.max_sequence_length - 1
            target_position = len(history) - 1
            for item_id in reversed(history[:-1]):
                sequence[index] = item_id
                positive_items[index] = next_item
                if config is not None:
                    simulated = target_position >= config["initial_history_lengths"][user_id]
                    simulated_targets[index] = simulated
                    if next_item not in indexed_heads and (config["scope"] == "all" or simulated):
                        positive_weights[index] = config["non_head_weight"]
                next_item = item_id
                index -= 1
                target_position -= 1
                if index < 0:
                    break
            examples.append(
                _TrainingExample(
                    sequence=tuple(sequence),
                    positive_items=tuple(positive_items),
                    negative_candidates=negative_candidates,
                    positive_weights=tuple(positive_weights),
                    simulated_targets=tuple(simulated_targets),
                )
            )
        return examples

    def _sample_negative_items(
        self,
        *,
        examples: Sequence[_TrainingExample],
        rng: np.random.Generator,
    ) -> np.ndarray:
        negative_items = np.zeros(
            (len(examples), self.max_sequence_length),
            dtype=np.int64,
        )
        for row, example in enumerate(examples):
            target_count = sum(
                item_id != 0 for item_id in example.positive_items
            )
            negative_items[row, -target_count:] = rng.choice(
                example.negative_candidates,
                size=target_count,
                replace=True,
            )
        return negative_items

    def _inference_sequence(self, history: Sequence[int]) -> list[int]:
        recent_items = list(history[-self.max_sequence_length :])
        return [
            *([0] * (self.max_sequence_length - len(recent_items))),
            *recent_items,
        ]

    def _require_fitted_model(self) -> _SASRecModel:
        if self._model is None:
            raise RuntimeError("SASRecAdapter must be fitted first")
        return self._model

    def _seeded_torch_operations(self):
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices.append(
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
        context = (
            torch.random.fork_rng(devices=cuda_devices)
            if hasattr(torch.random, "fork_rng")
            else nullcontext()
        )

        class _SeededContext:
            def __enter__(inner_self):
                context.__enter__()
                torch.manual_seed(self.random_seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(self.random_seed)

            def __exit__(inner_self, exc_type, exc_value, traceback):
                return context.__exit__(exc_type, exc_value, traceback)

        return _SeededContext()

    def _validate_candidates(
        self,
        candidate_ids: Sequence[ItemId],
    ) -> tuple[int, ...]:
        values = tuple(candidate_ids)
        self._validate_item_ids(values, context="candidate IDs")
        if len(values) != len(set(values)):
            raise ValueError("candidate IDs must be unique")
        unknown = sorted(set(values).difference(self._catalog_item_ids))
        if unknown:
            raise ValueError(f"candidate IDs lie outside fitted catalog: {unknown}")
        return values

    def _validate_inference_history(
        self,
        history: Sequence[ItemId],
    ) -> tuple[int, ...]:
        values = tuple(history)
        self._validate_item_ids(values, context="history item IDs")
        unknown = sorted(set(values).difference(self._catalog_item_ids))
        if unknown:
            raise ValueError(f"history contains items outside fitted catalog: {unknown}")
        return values

    @staticmethod
    def _validate_user_id(user_id: UserId) -> None:
        if isinstance(user_id, bool) or not isinstance(user_id, int):
            raise TypeError("User IDs must be integers")
        if user_id <= 0:
            raise ValueError("User IDs must be positive")

    @staticmethod
    def _validate_item_ids(values: Sequence[int], *, context: str) -> None:
        if any(
            isinstance(item_id, bool) or not isinstance(item_id, int)
            for item_id in values
        ):
            raise TypeError(f"{context} must be integers")
        if any(item_id <= 0 for item_id in values):
            raise ValueError(f"{context} must be positive; 0 is reserved for padding")

    @staticmethod
    def _rank_by_popularity(
        *,
        histories: Mapping[int, tuple[int, ...]],
        catalog: tuple[int, ...],
    ) -> tuple[int, ...]:
        counts = dict.fromkeys(catalog, 0)
        for history in histories.values():
            for item_id in history:
                counts[item_id] += 1
        return tuple(
            sorted(catalog, key=lambda item_id: (-counts[item_id], item_id))
        )

    @classmethod
    def _validate_catalog(
        cls,
        catalog_item_ids: Collection[int],
    ) -> tuple[int, ...]:
        catalog = tuple(catalog_item_ids)
        if not catalog:
            raise ValueError("catalog_item_ids cannot be empty")
        cls._validate_item_ids(catalog, context="Catalog item IDs")
        if len(catalog) != len(set(catalog)):
            raise ValueError("catalog_item_ids cannot contain duplicates")
        return tuple(sorted(catalog))

    @classmethod
    def _validate_histories(
        cls,
        histories: Mapping[UserId, Sequence[ItemId]],
    ) -> dict[int, tuple[int, ...]]:
        if not histories:
            raise ValueError("histories cannot be empty")

        validated: dict[int, tuple[int, ...]] = {}
        for user_id, history in histories.items():
            cls._validate_user_id(user_id)
            items = tuple(history)
            cls._validate_item_ids(items, context="History item IDs")
            validated[user_id] = items
        return validated

    @staticmethod
    def _validate_restored_fallback(
        fallback_ranking: Sequence[int],
        catalog: tuple[int, ...],
    ) -> tuple[int, ...]:
        fallback = tuple(fallback_ranking)
        if set(fallback) != set(catalog) or len(fallback) != len(catalog):
            raise ValueError("SASRec fallback ranking must be a catalog permutation")
        return fallback

    @staticmethod
    def _validate_hyperparameters(
        *,
        max_sequence_length: int,
        hidden_size: int,
        num_blocks: int,
        num_heads: int,
        dropout_rate: float,
        norm_first: bool,
        learning_rate: float,
        embedding_l2: float,
        batch_size: int,
        num_epochs: int,
    ) -> None:
        positive_integer_values = {
            "max_sequence_length": max_sequence_length,
            "hidden_size": hidden_size,
            "num_blocks": num_blocks,
            "num_heads": num_heads,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        }
        for name, value in positive_integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must lie in [0, 1)")
        if not isinstance(norm_first, bool):
            raise TypeError("norm_first must be a boolean")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if embedding_l2 < 0.0:
            raise ValueError("embedding_l2 cannot be negative")

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if not isinstance(requested, str):
            raise TypeError("device must be a string")
        normalized = requested.lower()
        if normalized == "auto":
            normalized = "cuda" if torch.cuda.is_available() else "cpu"
        if normalized == "cpu":
            return torch.device("cpu")
        if normalized == "cuda" or normalized.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            device = torch.device(normalized)
            if (
                device.index is not None
                and device.index >= torch.cuda.device_count()
            ):
                raise ValueError(f"CUDA device index is unavailable: {requested}")
            return device
        raise ValueError(f"Unsupported device: {requested}")
