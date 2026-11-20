"""Feedback-loop experiment metric entry point -- definition only."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def analyze_feedback_loop_experiment(run_root: Path) -> Mapping[str, Any]:
    """Eventually derive all preregistered M0-M4 metrics from artifacts."""

    del run_root
    raise NotImplementedError("Feedback-loop experiment evaluation is not implemented yet.")
