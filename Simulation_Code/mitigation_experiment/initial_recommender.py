"""One immutable unweighted SASRec initialization shared by a campaign."""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile

from .contracts import SimulationCheckpoint
from .recommenders.base import FitSpec
from .recommenders.sasrec import SASRecAdapter
from .utility.checkpoints import load_checkpoint, save_checkpoint
from .utility.fingerprints import fingerprint_semantic_state
from .utility.saving import save_json_artifact

logger = logging.getLogger(__name__)


def ensure_initial_recommender(*, dataset, training_user_ids, seed: int,
                               settings: dict, checkpoint_root: Path) -> Path:
    histories = {user: tuple(dataset.training_histories[user]) for user in sorted(training_user_ids)}
    resolved = {**settings, "catalog_item_ids": tuple(sorted(dataset.catalog)), "random_seed": seed}
    history_id = fingerprint_semantic_state(histories, "recommender-training-histories-v1")
    identity = {"dataset": dataset.dataset_fingerprint, "training_histories": history_id,
                "settings": resolved, "schema": "mitigation-initial-sasrec-v1"}
    key = fingerprint_semantic_state(identity, identity["schema"])
    directory = checkpoint_root / key
    recommender = SASRecAdapter(**resolved)
    if directory.exists():
        checkpoint = load_checkpoint(directory)
        if checkpoint.input_fingerprints != {"dataset": dataset.dataset_fingerprint, "initial_identity": key}:
            raise ValueError("shared initial checkpoint identity mismatch")
        from .runtime_setup import _restore_initial_recommender
        _restore_initial_recommender(recommender=recommender, checkpoint_path=directory,
                                     dataset_fingerprint=dataset.dataset_fingerprint,
                                     training_histories=histories)
        logger.info("Reusing verified unweighted initial SASRec: %s", directory)
        return directory
    logger.info("Fitting shared unweighted initial SASRec users=%s", len(histories))
    # Preserve a failed fit's context before any numerical work starts.
    attempts = checkpoint_root / (key + ".fit_attempts")
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=attempts))
    save_json_artifact(attempt / "request.json", identity)
    try:
        report = recommender.fit(histories, FitSpec(0, history_id, {"fit_role": "shared_initial"}))
        save_json_artifact(attempt / "fit_report.json", dict(report.diagnostics))
        save_checkpoint(directory, SimulationCheckpoint(
            schema_version=identity["schema"], run_id=f"shared_initial__{key}",
            condition_id="unweighted_initial", seed=seed, completed_iteration=0,
            input_fingerprints={"dataset": dataset.dataset_fingerprint, "initial_identity": key},
            component_states={"recommender": recommender.export_state()},
            policy_states={}, randomness_state={"simulation_seed": seed},
        ))
        save_json_artifact(attempt / "_COMPLETE.json", {"checkpoint": str(directory), "identity": key})
    except BaseException as exc:
        logger.exception("Shared initial fit failed; progress retained in %s", attempt)
        try:
            save_json_artifact(attempt / "_FAILED.json", {"error": str(exc), "exception_type": type(exc).__name__})
        except Exception:
            logger.exception("Could not save shared-fit failure marker at %s", attempt)
        raise
    return directory
