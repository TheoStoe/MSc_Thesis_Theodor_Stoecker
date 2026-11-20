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
    # Recorded attempts are a lower bound if a failed response did not retain
    # its attempt count. This flag prevents presenting missing counts as zero.
    provider_call_accounting_complete: bool = True


def summarize_run_health(run_root: Path) -> RunHealthSummary:
    from .mitigation import summarize_run

    summary = summarize_run(run_root)
    health = summary["health"]
    return RunHealthSummary(
        total_events=summary["events"], valid_actions=health["valid"],
        fallbacks=health["fallback"], invalid_actions=health["invalid"],
        technical_failures=health["technical_failure"],
        provider_calls=health["provider_calls_recorded"],
        provider_call_accounting_complete=health["incomplete_provider_accounting"] == 0,
    )
