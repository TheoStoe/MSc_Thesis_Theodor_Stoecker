"""Frozen Goodreads head definitions with an ID-independent boundary sensitivity.

The original head retains exactly ``ceil(.2 * catalogue size)`` items, sorted
by decreasing initial training count and then increasing item ID. The strict
companion retains only counts above that head's cut-off. It neither refills
the removed boundary items nor changes any original metric or run artifact.
"""
from __future__ import annotations

import json
import math
from numbers import Integral
from pathlib import Path

import pandas as pd

from .metrics import _reference, ratio
from .readers import file_hash, resolve_path


COUNT_COLUMNS = (
    "catalogue_size", "head_count", "cutoff_count", "cutoff_tie_pool_count",
    "selected_tie_count", "selected_tie_head_share", "strict_head_count",
    "strict_head_catalogue_share", "initial_training_head_share",
    "initial_training_strict_head_share", "tie_break_required",
)
SUMMARY_COLUMNS = (
    "dataset", "seed", "population_seed", *COUNT_COLUMNS,
    "dataset_fingerprint", "physical_source_count", "logical_membership_count",
    "prepared_input_count", "source_ids", "prepared_inputs",
)
ITEM_COLUMNS = ("dataset", "seed", "population_seed", "item_id",
                "initial_training_count", "cutoff_count")
REFERENCE_INPUTS = ("prepared_directory", "cohort_path", "labels_path")


def head_tie_summary(popularity, head=None):
    """Summarise the original head and its strictly-above-cut-off companion.

    ``selected_tie_head_share`` is the original head's share at its boundary,
    even when the complete boundary group fits. ``tie_break_required`` marks
    the narrower case where only part of that equal-count group is selected.
    Training masses use all initial training events, retaining zero-count
    catalogue items and undefined masses when every training count is zero.
    """
    if not hasattr(popularity, "items") or not popularity:
        raise ValueError("Popularity must be a nonempty item-to-count mapping")
    if any(isinstance(item, bool) or not isinstance(item, Integral)
           for item in popularity):
        raise ValueError("Popularity item IDs must be integers")
    if any(isinstance(count, bool) or not isinstance(count, Integral) or count < 0
           for count in popularity.values()):
        raise ValueError("Initial training counts must be nonnegative integers")
    counts = {int(item): int(count) for item, count in popularity.items()}
    ranked = sorted(counts, key=lambda item: (-counts[item], item))
    original = set(ranked[:math.ceil(.2 * len(ranked))])
    if head is not None:
        supplied = tuple(head)
        if (any(isinstance(item, bool) or not isinstance(item, Integral) for item in supplied)
                or len(set(supplied)) != len(supplied) or set(supplied) != original):
            raise ValueError("Supplied head differs from the fixed count/item-ID head")
    cutoff = min(counts[item] for item in original)
    ties = {item for item, count in counts.items() if count == cutoff}
    strict = tuple(sorted(item for item, count in counts.items() if count > cutoff))
    selected = len(original & ties)
    training_events = sum(counts.values())
    return {
        "catalogue_size": len(counts), "head_count": len(original),
        "cutoff_count": cutoff, "cutoff_tie_pool_count": len(ties),
        "selected_tie_count": selected,
        "selected_tie_head_share": ratio(selected, len(original)),
        "strict_head_count": len(strict),
        "strict_head_catalogue_share": ratio(len(strict), len(counts)),
        "initial_training_head_share": ratio(sum(counts[item] for item in original), training_events),
        "initial_training_strict_head_share": ratio(sum(counts[item] for item in strict), training_events),
        "tie_break_required": bool(selected < len(ties)),
        "strict_head_item_ids": strict,
    }


def _input_metadata(row, reference, repo_root):
    prepared = Path(reference["prepared"])
    cohort = resolve_path(row["cohort_path"], repo_root)
    labels = resolve_path(row["labels_path"], repo_root)
    return {
        "prepared_directory": str(prepared.resolve()),
        "cohort_path": str(cohort.resolve()), "labels_path": str(labels.resolve()),
        "catalogue_sha256": file_hash(prepared / "catalog.json"),
        "sequences_sha256": file_hash(prepared / "sequences.npz"),
        "prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
        "cohort_sha256": file_hash(cohort), "labels_sha256": file_hash(labels),
        "dataset_fingerprint": reference["dataset_fingerprint"],
    }


def load_goodreads_head_ties(sources, repo_root):
    """Return one head-definition row per frozen Goodreads seed/population.

    Physical runs and logical aliases only identify the frozen references;
    they are never pooled as independent replications. Every distinct set of
    prepared, cohort, and label inputs is checked through the existing reader,
    then must agree in popularity, original head, and dataset fingerprint.
    The second frame lists the exact strict-head item IDs for reproducibility.
    """
    from .packaged import head_ties
    return head_ties(sources)
    required = {"dataset", "status", "seed", "population_seed", "source_id", *REFERENCE_INPUTS}
    if required - set(sources):
        raise ValueError(f"Missing head-reference source columns: {sorted(required-set(sources))}")
    selected = sources.loc[sources.dataset.eq("goodreads") & sources.status.eq("complete")].copy()
    if selected.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS), pd.DataFrame(columns=ITEM_COLUMNS)
    if selected[["seed", "population_seed", "source_id", *REFERENCE_INPUTS]].isna().any().any():
        raise ValueError("Complete Goodreads source identities and reference paths must be defined")
    # A physical source cannot silently acquire a different cohort through a
    # reused logical membership. Other complete sources may use verified copies.
    for source_id, aliases in selected.groupby("source_id", sort=False):
        if len(aliases[["seed", "population_seed", *REFERENCE_INPUTS]].drop_duplicates()) != 1:
            raise ValueError(f"Conflicting frozen reference aliases for source {source_id}")
    rows, items = [], []
    root = Path(repo_root)
    for (seed, population), block in selected.groupby(["seed", "population_seed"], sort=True):
        baseline, inputs = None, []
        for _, source in block.drop_duplicates(list(REFERENCE_INPUTS)).iterrows():
            reference = _reference(source, root)
            summary = head_tie_summary(reference["popularity"], reference["head"])
            if baseline is None:
                baseline = reference
            elif (reference["dataset_fingerprint"] != baseline["dataset_fingerprint"]
                  or reference["popularity"] != baseline["popularity"]
                  or set(reference["head"]) != set(baseline["head"])):
                raise ValueError(f"Conflicting frozen Goodreads references for seed {seed}, population {population}")
            inputs.append(_input_metadata(source, reference, root))
        tags = {"dataset": "goodreads", "seed": int(seed), "population_seed": int(population)}
        summary = head_tie_summary(baseline["popularity"], baseline["head"])
        strict_items = summary.pop("strict_head_item_ids")
        membership_columns = [name for name in ("source_id", "study", "condition_id", "logical_condition_id") if name in block]
        rows.append({**tags, **summary, "dataset_fingerprint": baseline["dataset_fingerprint"],
            "physical_source_count": int(block.source_id.nunique()),
            "logical_membership_count": len(block[membership_columns].drop_duplicates()),
            "prepared_input_count": len(inputs),
            "source_ids": json.dumps(sorted(map(str, block.source_id.unique()))),
            "prepared_inputs": json.dumps(inputs, ensure_ascii=False, sort_keys=True)})
        items.extend({**tags, "item_id": item, "initial_training_count": baseline["popularity"][item],
                      "cutoff_count": summary["cutoff_count"]} for item in strict_items)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS), pd.DataFrame(items, columns=ITEM_COLUMNS)
