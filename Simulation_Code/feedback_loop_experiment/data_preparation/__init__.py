"""Self-contained, leakage-safe inputs for the feedback-loop experiment."""

from .artifacts import load_prepared_dataset, save_prepared_dataset
from .clustering import (
    ClusterPipelineConfig,
    FrozenClusterArtifact,
    fit_user_clusters,
    load_cluster_artifact,
    save_cluster_artifact,
)
from .movielens import MovieLensPreparationConfig, prepare_movielens_1m
from .population import sample_population

__all__ = [
    "ClusterPipelineConfig",
    "FrozenClusterArtifact",
    "MovieLensPreparationConfig",
    "fit_user_clusters",
    "load_cluster_artifact",
    "load_prepared_dataset",
    "prepare_movielens_1m",
    "sample_population",
    "save_cluster_artifact",
    "save_prepared_dataset",
]
