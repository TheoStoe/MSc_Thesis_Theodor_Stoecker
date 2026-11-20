"""Portable offline replay of bundled, derived research measurements.

This module does not read private event logs, checkpoints, source catalogues,
credentials, or provider clients. Source hashes describe the original extraction.
"""
from pathlib import Path
import json

import pandas as pd


ANALYSIS = Path(__file__).resolve().parent


def derived_frame(name):
    """Load one trusted dataframe supplied with this repository."""
    if not name.replace("_", "").isalnum():
        raise ValueError("Use a bundled derived-input name")
    path = ANALYSIS / "derived_inputs" / f"{name}.pkl.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Bundled derived input is missing: {path}")
    return pd.read_pickle(path)


def source_frame(name, sources):
    """Select original logical memberships without accessing their raw sources."""
    frame = derived_frame(name)
    selected = pd.DataFrame(sources).copy()
    if "status" in selected:
        selected = selected.loc[selected.status.eq("complete")]
    keys = [key for key in ("source_id", "study", "condition_id") if key in frame and key in selected]
    if "source_id" not in keys:
        raise ValueError("Bundled companion requires explicit source identities")
    selected = selected[keys].drop_duplicates()
    missing = set(selected.source_id) - set(frame.source_id)
    if missing:
        raise ValueError(f"Requested sources are outside the bundled {name} extraction: {sorted(missing)}")
    return frame.merge(selected, on=keys, how="inner", validate="many_to_one")


def head_ties(sources):
    selected = sources.loc[sources.dataset.eq("goodreads") & sources.status.eq("complete"), ["seed", "population_seed"]].drop_duplicates()
    return tuple(derived_frame(name).merge(selected, on=["seed", "population_seed"], how="inner", validate="many_to_one")
                 for name in ("goodreads_head_ties", "goodreads_strict_head_items"))


def profile_frequencies(assessments):
    """Recount saved human categories, without redistributing their evidence text.

    Completion status was recorded from the original six source-workbook fields.
    The public package retains this status, two categories, and presence flags;
    withheld text is never interpreted as a new human judgement.
    """
    from .profile_review_assessments import profile_review_frequencies
    frame = assessments.copy()
    for column in ("retained_interests", "supported_additions", "unsupported_changes", "evidence_notes"):
        flag = "has_" + column
        if flag not in frame:
            raise ValueError(f"Missing original source-field presence flag: {flag}")
        frame[column] = frame[flag].map({True: "[source field present; text withheld]", False: ""})
    return profile_review_frequencies(frame)


def bundle_provenance():
    return json.loads((ANALYSIS / "derived_inputs/provenance.json").read_text(encoding="utf-8"))
