"""Technical health summaries kept separate from behavioral outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunHealthSummary:
    total_events: int
    valid_actions: int
    fallbacks: int
    invalid_actions: int
    technical_failures: int
    provider_calls: int


def summarize_run_health(run_root: Path) -> RunHealthSummary:
    del run_root
    raise NotImplementedError("Run-health evaluation is not implemented yet.")

