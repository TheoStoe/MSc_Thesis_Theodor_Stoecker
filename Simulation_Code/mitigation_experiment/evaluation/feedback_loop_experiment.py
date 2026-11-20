"""Compatibility entry point for the local completed-run mitigation analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def analyze_feedback_loop_experiment(run_root: Path) -> Mapping[str, Any]:
    """Return descriptive canonical/probe metrics for one completed run."""
    from .mitigation import summarize_run

    return summarize_run(run_root)
