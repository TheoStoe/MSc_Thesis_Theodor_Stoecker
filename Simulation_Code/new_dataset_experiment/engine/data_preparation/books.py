"""Semantic identity for Goodreads prepared datasets."""
from __future__ import annotations
from typing import Any, Mapping
from ..utility.user_handling import PreparedDatasetArtifact
DATASET_SCHEMA = "goodreads-prepared-v1"

def dataset_semantic_payload(artifact: PreparedDatasetArtifact) -> Mapping[str, Any]:
    """Return exactly the data covered by ``dataset_fingerprint``."""

    return {
        "preparation_protocol": artifact.preparation_settings,
        "preparation_statistics": artifact.preparation_statistics,
        "source_file_fingerprints": artifact.source_file_fingerprints,
        "id_mapping": {
            "runtime_to_source_user_id": artifact.runtime_to_source_user_id,
            "runtime_to_source_item_id": artifact.runtime_to_source_item_id,
        },
        "catalog": artifact.catalog,
        "training_histories": artifact.training_histories,
        "training_interactions": artifact.training_interactions,
        "training_history_fingerprints": artifact.training_history_fingerprints,
        "validation_targets": artifact.validation_targets,
        "test_targets": artifact.test_targets,
        "initial_item_popularity": artifact.initial_item_popularity,
    }
