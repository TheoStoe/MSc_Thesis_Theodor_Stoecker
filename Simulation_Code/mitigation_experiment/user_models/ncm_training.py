"""Deterministic, self-contained training for the two NCM GRU variants."""

from __future__ import annotations

from collections.abc import Iterator
import copy
import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .ncm import NCMClickModelConfig, build_ncm_model


LOGGER = logging.getLogger(__name__)
NCM_TRAINING_CONFIG_SCHEMA_VERSION = "learned-first-click-training-config-v2"
GRADIENT_CLIP_MAX_NORM = 1.0


@dataclass(frozen=True)
class NCMTrainingConfig:
    seed: int
    epochs: int = 5_000
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 7
    minimum_improvement: float = 1e-5
    device: str = "cpu"
    deterministic_algorithms: bool = True
    schema_version: str = NCM_TRAINING_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, name, allow_zero in (
            (self.seed, "seed", True),
            (self.epochs, "epochs", False),
            (self.batch_size, "batch_size", False),
            (self.patience, "patience", False),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0 or (not allow_zero and value == 0):
                adjective = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be {adjective}")
        for value, name, allow_zero in (
            (self.learning_rate, "learning_rate", False),
            (self.weight_decay, "weight_decay", True),
            (self.minimum_improvement, "minimum_improvement", True),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            if numeric < 0 or (not allow_zero and numeric == 0):
                adjective = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be {adjective}")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-blank string")
        if self.deterministic_algorithms is not True:
            raise ValueError("deterministic_algorithms is fixed to True")
        if self.schema_version != NCM_TRAINING_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported NCM training-config schema")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IndexedNCMExamples:
    """Compact slates plus one dense history embedding per example."""

    item_row_indices: np.ndarray
    user_embeddings: np.ndarray
    clicked_indices: np.ndarray
    embedding_vectors: np.ndarray

    def __post_init__(self) -> None:
        examples = self.item_row_indices.shape[0]
        if self.item_row_indices.ndim != 2:
            raise ValueError("item row indices must have two dimensions")
        if self.user_embeddings.shape != (
            examples,
            self.embedding_vectors.shape[1],
        ):
            raise ValueError("user embeddings do not align with examples")
        if self.clicked_indices.shape != (examples,):
            raise ValueError("clicked indices do not align with examples")

    def __len__(self) -> int:
        return int(self.item_row_indices.shape[0])

    @property
    def slate_size(self) -> int:
        return int(self.item_row_indices.shape[1])

    def batch(
        self,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = indices.numpy()
        item_values = np.asarray(
            self.embedding_vectors[self.item_row_indices[rows]],
            dtype=np.float32,
        )
        user_values = np.asarray(self.user_embeddings[rows], dtype=np.float32)
        targets = np.asarray(self.clicked_indices[rows], dtype=np.int64)
        return (
            torch.from_numpy(item_values),
            torch.from_numpy(user_values),
            torch.from_numpy(targets),
        )


@dataclass(frozen=True)
class NCMPlacementMetadata:
    prepared_fingerprint: str
    parent_fingerprint: str
    placement_fingerprint: str
    policy: str


@dataclass(frozen=True)
class NCMEpochRecord:
    epoch: int
    train_binary_cross_entropy: float
    validation_binary_cross_entropy: float


@dataclass(frozen=True)
class NCMTrainingResult:
    model: nn.Module
    best_epoch: int
    records: tuple[NCMEpochRecord, ...]


def load_ncm_training_examples(
    placement_directory: str | Path,
    item_embeddings: dict[int, np.ndarray],
    *,
    prepared_fingerprint: str | None = None,
) -> tuple[IndexedNCMExamples, IndexedNCMExamples, NCMPlacementMetadata]:
    """Load train/validation arrays without importing the earlier experiment."""

    root = Path(placement_directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing NCM placement manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "first_click_placement_slates":
        raise ValueError("unsupported NCM placement artifact type")
    if manifest.get("schema_version") not in {
        "first-click-placement-slates-v2",
        "feedback-loop-ncm-placement-v1",
        "feedback-loop-ncm-placement-v2",
    }:
        raise ValueError("unsupported NCM placement artifact schema")
    if int(manifest.get("slate_size", 0)) != 10:
        raise ValueError("NCM placement artifacts must contain ten-item slates")

    examples_path = root / str(manifest["examples_file"])
    histories_path = root / str(manifest["histories_file"])
    _verify_file(examples_path, manifest["examples_file_sha256"])
    _verify_file(histories_path, manifest["histories_file_sha256"])
    histories = _load_histories(histories_path)
    if len(histories) != int(manifest["history_sequence_count"]):
        raise ValueError("NCM history count disagrees with placement manifest")

    split_counts = _count_splits(examples_path)
    if sum(split_counts.values()) != int(manifest["example_count"]):
        raise ValueError("NCM example count disagrees with placement manifest")
    for required in ("train", "validation"):
        if split_counts.get(required, 0) == 0:
            raise ValueError(f"NCM placement artifact has no {required} rows")

    item_ids = tuple(sorted(item_embeddings))
    item_to_row = {item_id: row for row, item_id in enumerate(item_ids)}
    embedding_vectors = np.stack([item_embeddings[item_id] for item_id in item_ids])
    embedding_vectors = np.asarray(embedding_vectors, dtype=np.float32)
    arrays = {
        split: _empty_examples(
            split_counts[split],
            slate_size=10,
            embedding_vectors=embedding_vectors,
        )
        for split in ("train", "validation")
    }
    offsets = {"train": 0, "validation": 0}
    cumulative_by_user: dict[int, np.ndarray] = {}
    previous_user = 0
    with examples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = _json_object(line, examples_path, line_number)
            split = str(row.get("split"))
            if split not in arrays:
                continue
            user_id = int(row["user_id"])
            if user_id < previous_user:
                raise ValueError("NCM examples must be ordered by user")
            previous_user = user_id
            if user_id not in histories:
                raise ValueError(f"missing history for NCM user {user_id}")
            if user_id not in cumulative_by_user:
                try:
                    history_rows = [item_to_row[item] for item in histories[user_id]]
                except KeyError as exc:
                    raise KeyError(
                        f"missing embedding for history item {exc.args[0]}"
                    ) from exc
                cumulative_by_user.clear()
                cumulative_by_user[user_id] = np.cumsum(
                    embedding_vectors[history_rows],
                    axis=0,
                    dtype=np.float64,
                )
            history_length = int(row["history_length"])
            event_index = int(row["event_index"])
            cumulative = cumulative_by_user[user_id]
            if history_length <= 0 or history_length > cumulative.shape[0]:
                raise ValueError("NCM example history length is out of range")
            if event_index != history_length:
                raise ValueError("NCM event index and history length disagree")
            clicked_index = int(row["clicked_index"])
            if not 0 <= clicked_index < 10:
                raise ValueError("NCM clicked index is out of range")
            ordered_items = tuple(int(value) for value in row["ordered_item_ids"])
            if len(ordered_items) != 10 or len(set(ordered_items)) != 10:
                raise ValueError("NCM examples require ten unique displayed items")
            if ordered_items[clicked_index] != int(row["target_item_id"]):
                raise ValueError("NCM clicked index does not identify the target item")

            output = arrays[split]
            index = offsets[split]
            try:
                output.item_row_indices[index] = [
                    item_to_row[item_id] for item_id in ordered_items
                ]
            except KeyError as exc:
                raise KeyError(
                    f"missing embedding for displayed item {exc.args[0]}"
                ) from exc
            output.user_embeddings[index] = (
                cumulative[history_length - 1] / float(history_length)
            ).astype(np.float32)
            output.clicked_indices[index] = clicked_index
            offsets[split] += 1
    if any(offsets[split] != split_counts[split] for split in offsets):
        raise RuntimeError("NCM example arrays were not filled completely")

    stored_prepared = manifest.get("prepared_fingerprint", prepared_fingerprint)
    if not isinstance(stored_prepared, str) or not stored_prepared:
        raise ValueError("NCM placement manifest has no prepared fingerprint")
    if (
        prepared_fingerprint is not None
        and stored_prepared != prepared_fingerprint
    ):
        raise ValueError("NCM placement prepared fingerprint mismatch")
    metadata = NCMPlacementMetadata(
        prepared_fingerprint=stored_prepared,
        parent_fingerprint=str(manifest["parent_fingerprint"]),
        placement_fingerprint=str(manifest["content_fingerprint"]),
        policy=str(manifest["policy"]),
    )
    return arrays["train"], arrays["validation"], metadata


def train_ncm_model(
    model_config: NCMClickModelConfig,
    training_config: NCMTrainingConfig,
    training_examples: IndexedNCMExamples,
    validation_examples: IndexedNCMExamples,
) -> NCMTrainingResult:
    """Fit one GRU using validation-only early stopping."""

    if not len(training_examples) or not len(validation_examples):
        raise ValueError("NCM fitting requires nonempty training and validation examples")
    device = torch.device(training_config.device)
    configure_deterministic_execution(device)
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    model = build_ncm_model(model_config).to(device)
    context = {"seed": training_config.seed, "epoch": 0, "batch": 0}
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(training_config.seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    records: list[NCMEpochRecord] = []
    for epoch in range(1, training_config.epochs + 1):
        context.update(epoch=epoch, batch=0)
        model.train()
        summed_train_loss = 0.0
        for batch_number, indices in enumerate(_batch_indices(
            len(training_examples),
            batch_size=training_config.batch_size,
            generator=generator,
        ), start=1):
            context["batch"] = batch_number
            items, users, targets = training_examples.batch(indices)
            optimizer.zero_grad(set_to_none=True)
            logits = model(items.to(device), users.to(device))
            labels = F.one_hot(
                targets.to(device),
                num_classes=training_examples.slate_size,
            ).to(dtype=logits.dtype)
            loss = loss_function(logits, labels)
            _require_finite(loss, stage="training_loss", context=context)
            loss.backward()
            try:
                nn.utils.clip_grad_norm_(
                    model.parameters(), GRADIENT_CLIP_MAX_NORM,
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                _numerical_failure("gradient_norm", str(exc), context)
            optimizer.step()
            summed_train_loss += float(loss.detach().cpu()) * targets.numel()
        train_loss = summed_train_loss / len(training_examples)
        context["batch"] = 0
        validation_loss = _mean_loss(
            model,
            validation_examples,
            batch_size=training_config.batch_size,
            device=device,
            context=context,
        )
        # Invalid validation must fail rather than silently consume patience.
        if not math.isfinite(validation_loss):
            _numerical_failure("validation_loss", "non-finite validation mean", context)
        records.append(NCMEpochRecord(epoch, train_loss, validation_loss))
        LOGGER.info(
            "NCM seed %d epoch %d: train_bce=%.6f validation_bce=%.6f",
            training_config.seed,
            epoch,
            train_loss,
            validation_loss,
        )
        if validation_loss < best_loss - training_config.minimum_improvement:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break
    if best_state is None:
        raise RuntimeError("NCM training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return NCMTrainingResult(model, best_epoch, tuple(records))


def _numerical_failure(stage: str, detail: str, context: dict[str, Any]) -> None:
    message = f"NCM {stage}: {detail}; context={context}"
    LOGGER.error(message, extra={"event_name": "ncm_numerical_failure", "stage": stage, **context})
    raise FloatingPointError(message)


def _require_finite(value: torch.Tensor, *, stage: str, context: dict[str, Any]) -> None:
    if not bool(torch.isfinite(value).all()):
        _numerical_failure(stage, "non-finite value", context)


def configure_deterministic_execution(device: torch.device) -> None:
    if device.type == "cuda":
        configured = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if configured not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "deermtinistic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                "or :16:8 before CUDA initialization"
            )
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _empty_examples(
    count: int,
    *,
    slate_size: int,
    embedding_vectors: np.ndarray,
) -> IndexedNCMExamples:
    return IndexedNCMExamples(
        item_row_indices=np.empty((count, slate_size), dtype=np.int32),
        user_embeddings=np.empty(
            (count, embedding_vectors.shape[1]),
            dtype=np.float32,
        ),
        clicked_indices=np.empty(count, dtype=np.int64),
        embedding_vectors=embedding_vectors,
    )


def _mean_loss(
    model: nn.Module,
    examples: IndexedNCMExamples,
    *,
    batch_size: int,
    device: torch.device,
    context: dict[str, Any] | None = None,
) -> float:
    if not len(examples):
        raise ValueError("NCM validation examples cannot be empty")
    model.eval()
    total = 0.0
    validation_context = dict(context or {})
    with torch.inference_mode():
        for batch_number, indices in enumerate(_batch_indices(len(examples), batch_size=batch_size), start=1):
            validation_context["batch"] = batch_number
            items, users, targets = examples.batch(indices)
            logits = model(items.to(device), users.to(device))
            labels = F.one_hot(
                targets.to(device),
                num_classes=examples.slate_size,
            ).to(dtype=logits.dtype)
            batch_loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="sum",
            )
            _require_finite(batch_loss, stage="validation_loss", context=validation_context)
            total += float(batch_loss.cpu())
    return total / (len(examples) * examples.slate_size)


def _batch_indices(
    count: int,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    order = (
        torch.arange(count, dtype=torch.int64)
        if generator is None
        else torch.randperm(count, generator=generator)
    )
    for start in range(0, count, batch_size):
        yield order[start : start + batch_size]


def _load_histories(path: Path) -> dict[int, tuple[int, ...]]:
    histories: dict[int, tuple[int, ...]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = _json_object(line, path, line_number)
            user_id = int(row["user_id"])
            if user_id in histories:
                raise ValueError(f"duplicate NCM history for user {user_id}")
            values = tuple(int(value) for value in row["history_item_ids"])
            if not values:
                raise ValueError("NCM histories cannot be empty")
            histories[user_id] = values
    return histories


def _count_splits(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = _json_object(line, path, line_number)
            split = str(row.get("split"))
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"unsupported NCM split: {split}")
            counts[split] = counts.get(split, 0) + 1
    return counts


def _json_object(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}:{line_number}")
    return value


def _verify_file(path: Path, expected_hash: object) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing NCM training artifact file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"NCM training artifact hash mismatch: {path.name}")


__all__ = [
    "IndexedNCMExamples",
    "NCMEpochRecord",
    "NCMPlacementMetadata",
    "NCMTrainingConfig",
    "NCMTrainingResult",
    "configure_deterministic_execution",
    "load_ncm_training_examples",
    "train_ncm_model",
]
