"""Offline validation on fixed initial histories, without fitting or providers.

Declare evaluation checkpoints before inspecting outcomes (initial, 1, 10, 30
in the primary plan). The final test targets are never used by this tool.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..policies import build_prior_head_rank_penalty_policy
from ..recommenders.sasrec import SASRecAdapter
from ..utility.checkpoints import load_checkpoint


def evaluate_checkpoint(dataset, checkpoint_directory: Path, *, device="cpu", penalty=0):
    checkpoint = load_checkpoint(checkpoint_directory)
    if checkpoint.input_fingerprints.get("dataset") != dataset.dataset_fingerprint:
        raise ValueError("checkpoint and validation dataset differ")
    raw = checkpoint.component_states["recommender"]
    settings = {**raw["settings"], "device": device}
    recommender = SASRecAdapter(**settings)
    recommender.restore_state({**raw, "settings": settings})
    users = tuple(sorted(int(user) for user in raw["histories"]))
    catalog = tuple(sorted(dataset.catalog))
    heads = sorted(catalog, key=lambda item: (-dataset.initial_item_popularity[item], item))[:math.ceil(.2 * len(catalog))]
    policy = build_prior_head_rank_penalty_policy(head_item_ids=heads, penalty=penalty)
    histories = {user: dataset.training_histories[user] for user in users}
    seen = {user: set(histories[user]) for user in users}
    eligible = {user: tuple(item for item in catalog if item not in seen[user]) for user in users}
    if any(len(items) < 10 for items in eligible.values()):
        raise ValueError("validation requires at least ten initially unseen candidates per user")
    rankings = recommender.recommend(users, 10, histories, eligible).items_by_user
    hits = ndcg_raw = ndcg_presented = 0.0
    for user, ranking in rankings.items():
        target = dataset.validation_targets[user]
        if target in ranking:
            hits += 1
            ndcg_raw += 1 / math.log2(ranking.index(target) + 2)
            presented = policy.present(user, max(1, checkpoint.completed_iteration), ranking).presented_items
            ndcg_presented += 1 / math.log2(presented.index(target) + 2)
    return {"schema": "mitigation-fixed-history-validation-v1",
            "checkpoint": str(checkpoint_directory.resolve()), "iteration": checkpoint.completed_iteration,
            "dataset_fingerprint": dataset.dataset_fingerprint, "users": len(users),
            "cohort": "all initial SASRec training users", "history": "fixed initial training histories",
            "candidates": "full catalog excluding only initial training history",
            "targets": "penultimate held-out validation item", "penalty": penalty,
            "validation_hits": int(hits), "hit_at_10": hits / len(users),
            "raw_ndcg_at_10": ndcg_raw / len(users), "presented_ndcg_at_10": ndcg_presented / len(users)}
