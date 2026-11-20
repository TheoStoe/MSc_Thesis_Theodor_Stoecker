"""Small non-overwriting checkpoint boundary used by the simulator runner."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from ..contracts import SimulationCheckpoint
from .fingerprints import fingerprint_semantic_state


CHECKPOINT_SCHEMA = "feedback-loop-checkpoint-file-v1"


def save_checkpoint(
    directory: str | Path,
    checkpoint: SimulationCheckpoint,
) -> Path:
    """Atomically write one new checkpoint directory and refuse overwrites.

    The checkpoint object is the complete resume state plus job identity; it
    does not add another model abstraction or alter component data.
    """

    if not isinstance(checkpoint, SimulationCheckpoint):
        raise TypeError("checkpoint must be a SimulationCheckpoint")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    destination = root / "checkpoint.pt"
    temporary = root / "checkpoint.pt.incomplete"
    state = asdict(checkpoint)
    payload: Mapping[str, Any] = {
        "file_schema": CHECKPOINT_SCHEMA,
        "checkpoint_state": state,
        "checkpoint_fingerprint": fingerprint_semantic_state(
            state, checkpoint.schema_version
        ),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except BaseException:
        # Keep the uniquely created directory as an obvious incomplete marker;
        # never delete or silently reuse a partially written scientific path.
        raise
    return destination


def load_checkpoint(
    directory: str | Path,
    *,
    expected_run_id: str | None = None,
    expected_condition_id: str | None = None,
    expected_seed: int | None = None,
) -> SimulationCheckpoint:
    """Load and validate a trusted, locally generated checkpoint."""

    path = Path(directory) / "checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {path}")
    # Loading onto CPU is deliberate and portable. Each component adapter then
    # restores weights onto its configured device, including CUDA when chosen.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("file_schema") != CHECKPOINT_SCHEMA
    ):
        raise ValueError("unsupported checkpoint file schema")
    state = payload.get("checkpoint_state")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint state is missing")
    checkpoint = SimulationCheckpoint(**dict(state))
    actual = fingerprint_semantic_state(state, checkpoint.schema_version)
    if payload.get("checkpoint_fingerprint") != actual:
        raise ValueError("checkpoint state fingerprint mismatch")
    expected = {
        "run_id": expected_run_id,
        "condition_id": expected_condition_id,
        "seed": expected_seed,
    }
    for field, value in expected.items():
        if value is not None and getattr(checkpoint, field) != value:
            raise ValueError(f"checkpoint {field} mismatch")
    return checkpoint
