"""Build a recorded, offline results snapshot from completed saved runs only.

No simulator, provider client, training entry point, or background-run control is
imported. Original run artifacts and thesis sources are read-only inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .sources import load_sources
from .metrics import VERSION, build_analysis_frames, with_memberships
from .comparisons import (CORE_METRICS, SHARE_METRICS, build_comparison_plan,
    paired_effects, summarise_effects, summarise_iterations, seed_consistency,
    model_differences, mitigation_magnitude_comparison, order_sensitivity_association,
    qualitative_profile_sample)
from .plotting import make_scale_registry

ANALYSIS = Path(__file__).resolve().parent
REPOSITORY = ANALYSIS.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prepare_partition(sources, cache, logfile, force):
    with Path(logfile).open("w", encoding="utf-8") as output, redirect_stdout(output):
        build_analysis_frames(sources, repo_root=REPOSITORY, cache_dir=cache, force=force)
    return len(sources)


def build_snapshot(*, snapshot_name=None, refresh=True, force=False, workers=3):
    started = datetime.now(timezone.utc)
    name = snapshot_name or started.strftime("local_results_%Y%m%dT%H%M%SZ")
    if not name.startswith("local_"):
        raise ValueError("Raw extraction contains source-bearing profile evidence. Use a local_ snapshot name; these snapshots must remain local/private.")
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple snapshot directory name")
    destination = ANALYSIS / "snapshots" / name
    if destination.exists():
        raise FileExistsError("Keep earlier snapshots; choose a new name")
    sources = load_sources(REPOSITORY, refresh=refresh, include_pending=True)
    completed = sources.loc[sources.status.eq("complete")].copy()
    if workers > 1:
        unique = completed.drop_duplicates("source_id").reset_index(drop=True)
        logs = ANALYSIS / "cache" / "build_logs" / name
        logs.mkdir(parents=True)
        # No GPU work: these processes stream and verify JSONL and write numeric
        # caches. Thread-limited BLAS prevents incidental oversubscription.
        for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[env_name] = "1"
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(_prepare_partition, unique.iloc[i::workers], ANALYSIS / "cache", logs / f"worker_{i+1}.log", force)
                    for i in range(workers)]
            for job in as_completed(jobs):
                print(f"Verified CPU partition: {job.result()} physical sources", flush=True)
    physical = build_analysis_frames(completed, repo_root=REPOSITORY, cache_dir=ANALYSIS / "cache", force=force)
    if set(physical["trajectory"].source_id) != set(completed.source_id):
        raise ValueError("Trajectory source IDs do not match the completed source inventory.")
    frames = {name: with_memberships(frame, completed) for name, frame in physical.items()}
    for column, fallback in [("seed", "simulation_seed"), ("condition_id", "logical_condition_id"), ("stage", "feedback_stage")]:
        if column not in sources and fallback in sources:
            sources[column] = sources[fallback]
    frames["sources"] = sources
    frames["pending_sources"] = sources.loc[sources.status.ne("complete")].copy()
    frames["comparison_plan"] = build_comparison_plan(sources)
    trajectory = frames["trajectory"]
    numeric = [c for c in trajectory.select_dtypes(include="number") if c not in {
        "seed", "simulation_seed", "population_seed", "iteration", "source_row", "user_id"}]
    frames["summaries"] = summarise_iterations(trajectory, numeric)
    effect_metrics = [m for m in sorted(set(CORE_METRICS) | SHARE_METRICS) if m in trajectory]
    frames["effects"] = paired_effects(trajectory, frames["comparison_plan"], effect_metrics)
    frames["effect_summaries"] = summarise_effects(frames["effects"])
    frames["seed_consistency"] = seed_consistency(frames["effect_summaries"])
    frames["model_differences"] = model_differences(trajectory)
    frames["mitigation_magnitudes"] = mitigation_magnitude_comparison(trajectory)
    frames["order_pairs"], frames["order_correlations"] = order_sensitivity_association(trajectory, frames["probes"], sources)
    frames["qualitative_review"] = qualitative_profile_sample(frames["profile_changes"])
    # Keep initial/final profile text available for manual assessment, not an
    # automated claim about supported or unsupported preference changes.
    if not frames["qualitative_review"].empty and not frames["profiles"].empty:
        profiles = frames["profiles"]
        keys = [c for c in ["dataset", "study", "seed", "population_seed", "condition_id", "user_id"] if c in profiles]
        text_columns = [c for c in ["initial_profile", "final_profile"] if c in profiles]
        frames["qualitative_review"] = frames["qualitative_review"].merge(
            profiles[keys + text_columns].drop_duplicates(keys), on=keys, how="left", validate="one_to_one")
    scales = make_scale_registry(frames)
    if not frames["effect_summaries"].empty:
        for metric, group in frames["effect_summaries"].groupby("metric"):
            finite = pd.to_numeric(group.value).dropna().to_numpy()
            extent = max(.01, float(np.abs(finite).max())*1.1) if len(finite) else 1
            scales.setdefault(metric, {"limits": [0, 1]})["difference_limits"] = [-extent, extent]
            scales[metric]["difference_unit"] = group.effect_unit.iloc[0]
    destination.mkdir(parents=True)
    data_directory = destination / "data"
    data_directory.mkdir()
    dataframes = {}
    for frame_name, frame in frames.items():
        path = data_directory / f"{frame_name}.csv"
        frame.to_csv(path, index=False)
        # Preserve dtypes and empty schemas for notebooks. CSV is the portable
        # audit/export companion; pickle is a cache of our own trusted frames.
        pickle_path = data_directory / f"{frame_name}.pkl"
        frame.to_pickle(pickle_path)
        dataframes[frame_name] = {"path": f"data/{frame_name}.pkl", "csv": f"data/{frame_name}.csv",
                                  "rows": len(frame), "sha256": sha(path), "pickle_sha256": sha(pickle_path)}
    draft = REPOSITORY / "writing/latex/chapters/experiments.tex"
    metadata = {"schema_version": "thesis-rq-results-snapshot-v1", "snapshot_id": name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "started_at_utc": started.isoformat(),
        "metric_version": VERSION, "rank_discount": "1/log2(r+1), one-based r=1..K",
        "distribution_scope": "local-only raw extraction; review source-data rights before any redistribution",
        "draft_path": str(draft.relative_to(REPOSITORY)) if draft.is_file() else None, "draft_sha256": sha(draft) if draft.is_file() else None,
        "analysis_config_sha256": sha(ANALYSIS / "analysis_config.json"),
        "source_inventory_sha256": sha(ANALYSIS / "sources/inventory.json"),
        "code_sha256": {p.name: sha(p) for p in ANALYSIS.glob("*.py")},
        "completed_logical_cells": len(completed), "completed_physical_runs": completed.source_id.nunique(),
        "pending_logical_cells": len(sources)-len(completed), "dataframes": dataframes, "plot_scales": scales,
        "notes": ["Only complete saved runs enter metrics; pending cells are explicitly retained in the availability table.",
                  "Logical aliases are calculated once physically and restored as explicit study memberships.",
                  "Mean1–30 is undefined if any of its 30 iteration values is undefined; endpoint is iteration30.",
                  "Paired share differences use percentage points; no user-level or iteration-level significance tests.",
                  "LLM and computational order predictors remain separate estimands.",
                  "No experiments or paid calls are started; rerun the build for a new snapshot after pending runs finish."]}
    (destination / "snapshot.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (ANALYSIS / "latest.json").write_text(json.dumps({"snapshot_directory": str(destination.relative_to(ANALYSIS)).replace("\\", "/"),
                                                    "created_at_utc": metadata["created_at_utc"]}, indent=2), encoding="utf-8")
    print(json.dumps({"snapshot": str(destination), "physical_runs": metadata["completed_physical_runs"],
                      "logical_cells": len(completed), "pending": metadata["pending_logical_cells"]}, indent=2), flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name")
    parser.add_argument("--force", action="store_true", help="Recalculate derived caches, never rerun simulations")
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    build_snapshot(snapshot_name=args.name, force=args.force, workers=args.workers)
