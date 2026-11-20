"""Create or reuse the concrete inputs needed by one simulator command."""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Sequence
from uuid import uuid4

from .data_preparation import (
    ClusterPipelineConfig,
    fit_user_clusters,
    load_cluster_artifact,
    load_prepared_dataset,
    prepare_movielens_1m,
    save_cluster_artifact,
    save_prepared_dataset,
)
from .frozen_population import build_frozen_population, load_frozen_membership
from .persona_extraction import (
    LusiferProfileExtractor,
    ProfileArtifact,
    ProfileExtractionFailure,
    ProfileExtractionInput,
    ProfileInteraction,
    match_movielens_1m_user_texts,
    parse_movielens_1m_users,
)
from .utility.providers import JSONProvider
from .utility.fingerprints import fingerprint_semantic_state
from .utility.filesystem import rename_directory_with_retry
from .utility.saving import (
    _json_value, save_json_artifact, save_population_artifact, save_profile_bundle,
)
from .utility.user_handling import (
    PopulationRole,
    load_population_artifact,
    load_profile_artifacts,
    load_prepared_inputs,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutomaticInputPaths:
    prepared_data: Path
    clusters: Path
    population: Path
    profiles: Path | None


def ensure_automatic_inputs(
    *,
    raw_data_directory: Path,
    input_root: Path,
    population_seed: int,
    active_user_count: int,
    reserve_user_count: int,
    require_profiles: bool,
    provider: JSONProvider | None = None,
    profile_max_interactions: int = 100,
    profile_batch_size: int = 10,
) -> AutomaticInputPaths:
    """Create missing inputs once and validate everything reused later."""

    raw_data_directory = Path(raw_data_directory)
    input_root = Path(input_root)
    if any(type(value) is not int for value in (population_seed, active_user_count, reserve_user_count)):
        raise TypeError("population seed and sizes must be integers")
    if population_seed < 0:
        raise ValueError("population seed cannot be negative")
    if (active_user_count, reserve_user_count) != (6040, 0):
        if (active_user_count, reserve_user_count) not in {(100, 200), (300, 0)}:
            raise ValueError("supported populations are 100+200, 300+0, or 6040+0")
        load_frozen_membership(population_seed)
    cohort_name = (
        f"seed_{population_seed}__active_{active_user_count}"
        f"__reserve_{reserve_user_count}"
    )
    prepared_path = input_root / "ml1m"
    cluster_path = input_root / "ml1m_clusters"
    population_path = input_root / f"population__{cohort_name}.json"
    # Changing 100+200 to 300+0 must reuse the same frozen user profiles.
    profile_cohort = (
        "full_6040" if (active_user_count, reserve_user_count) == (6040, 0)
        else f"population_{population_seed}__selected_300"
    )
    profile_path = input_root / f"profiles__{profile_cohort}"

    if prepared_path.exists():
        dataset = load_prepared_dataset(prepared_path)
    else:
        _reject_incomplete(prepared_path)
        logger.info("Preparing MovieLens data at %s", prepared_path)
        dataset = prepare_movielens_1m(raw_data_directory)
        save_prepared_dataset(prepared_path, dataset)

    if cluster_path.exists():
        clusters = load_cluster_artifact(cluster_path)
    else:
        _reject_incomplete(cluster_path)
        logger.info("Fitting user clusters at %s", cluster_path)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1): #used to match notebook output exactly
            clusters = fit_user_clusters(dataset, config=ClusterPipelineConfig())
        save_cluster_artifact(cluster_path, clusters)
    if clusters.dataset_fingerprint != dataset.dataset_fingerprint:
        raise ValueError("automatic cluster and prepared-data artifacts disagree")

    if population_path.exists():
        population = load_population_artifact(population_path)
    else:
        _reject_incomplete(population_path)
        logger.info("Materializing frozen population at %s", population_path)
        population = build_frozen_population(
            dataset,
            clusters,
            population_seed=population_seed,
            active_user_count=active_user_count,
            reserve_user_count=reserve_user_count,
        )
        save_population_artifact(population_path, population)

    _validate_population_request(
        population,
        population_seed=population_seed,
        active_user_count=active_user_count,
        reserve_user_count=reserve_user_count,
    )
    expected_population = build_frozen_population(
        dataset, clusters, population_seed=population_seed,
        active_user_count=active_user_count, reserve_user_count=reserve_user_count,
    )
    if population.population_fingerprint != expected_population.population_fingerprint:
        raise ValueError("existing population differs from the approved frozen membership")
    # Reuse the normal loader as the final dataset/population consistency gate.
    load_prepared_inputs(prepared_path, population_path, None, None)

    resolved_profiles: Path | None = None
    if require_profiles:
        profile_inputs = _profile_inputs(
            raw_data_directory=raw_data_directory,
            dataset=dataset,
            population=population,
            extraction_seed=population_seed,
        )
        if provider is None:
            if not profile_path.exists():
                raise ValueError("a provider is required to create agent profiles")
            _validate_completed_profile_bundle(profile_path, profile_inputs)
            metadata = _read_json(profile_path / "manifest.json")["metadata"]
            settings = metadata["preparation_request"]["method_settings"]
            if (settings["max_interactions"], settings["batch_size"]) != (
                profile_max_interactions, profile_batch_size,
            ):
                raise ValueError("existing profiles use different extraction settings")
        else:
            extractor = LusiferProfileExtractor(
                provider=provider,
                max_interactions=profile_max_interactions,
                batch_size=profile_batch_size,
            )
            ensure_profile_bundle(profile_path, profile_inputs=profile_inputs, extractor=extractor)
        load_prepared_inputs(prepared_path, population_path, profile_path, None)
        resolved_profiles = profile_path

    return AutomaticInputPaths(
        prepared_data=prepared_path,
        clusters=cluster_path,
        population=population_path,
        profiles=resolved_profiles,
    )


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _profile_request(profile_inputs, extractor) -> dict:
    return _json_value({
        "schema_version": "mitigation-profile-progress-v1",
        "input_fingerprint": fingerprint_semantic_state(
            tuple(profile_inputs), "mitigation-profile-input-cohort-v1",
        ),
        "user_ids": [record.user_id for record in profile_inputs],
        "extractor_id": extractor.extractor_id,
        "prompt_fingerprint": extractor.prompt_fingerprint,
        "method_settings": extractor.method_settings,
        "provider_settings": extractor.provider_settings,
    })


class _ProfileProgressProvider:
    """Persist each chunk's call metadata/usage without changing its prompts."""

    def __init__(self, provider, attempt):
        self.provider, self.attempt, self.call_index = provider, attempt, 0

    def semantic_settings(self):
        return self.provider.semantic_settings()

    def call_json(self, **kwargs):
        self.call_index += 1
        prefix = self.attempt / f"call_{self.call_index:03d}"
        save_json_artifact(prefix.with_suffix(".request.json"), {
            "schema_name": kwargs.get("schema_name"),
            "system_prompt_sha256": hashlib.sha256(kwargs["system_prompt"].encode("utf-8")).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(kwargs["user_prompt"].encode("utf-8")).hexdigest(),
        })
        started = time.monotonic()
        try:
            result = self.provider.call_json(**kwargs)
        except BaseException as exc:
            save_json_artifact(prefix.with_suffix(".failure.json"), {
                "error_type": type(exc).__name__, "error": str(exc),
                "elapsed_seconds": time.monotonic() - started,
            })
            raise
        save_json_artifact(prefix.with_suffix(".response.json"), {
            "response_id": result.response_id, "usage": result.usage,
            "attempts": result.attempts, "payload": result.payload,
            "elapsed_seconds": time.monotonic() - started,
        })
        return result


def _validate_completed_profile_bundle(path, profile_inputs, request=None):
    complete = _read_json(path / "_COMPLETE.json")
    if complete.get("failure_count") != 0 or complete.get("profile_count") != len(profile_inputs):
        raise ValueError("profile bundle is not completely successful for the requested cohort")
    hashes = complete.get("file_sha256", {})
    required_files = {"manifest.json", "inputs.jsonl", "profiles.jsonl", "failures.jsonl", "diagnostics.jsonl"}
    if set(hashes) != required_files:
        raise ValueError("profile bundle is missing required integrity hashes")
    for name, expected in hashes.items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"profile bundle integrity mismatch: {name}")
    manifest = _read_json(path / "manifest.json")
    recorded = manifest.get("metadata", {}).get("preparation_request")
    if not isinstance(recorded, dict):
        raise ValueError("profile bundle has no mitigation preparation provenance")
    if recorded.get("input_fingerprint") != fingerprint_semantic_state(
        tuple(profile_inputs), "mitigation-profile-input-cohort-v1",
    ):
        raise ValueError("profile inputs differ from the completed preparation")
    if request is not None and recorded != request:
        raise ValueError("profile preparation settings differ from the completed preparation")
    profiles = load_profile_artifacts(path)
    if set(profiles) != {record.user_id for record in profile_inputs}:
        raise ValueError("completed profiles do not exactly cover the requested users")
    return profiles


@contextmanager
def _profile_progress_lock(progress):
    """An OS-released lock prevents concurrent calls and survives process death."""
    lock_path = progress / ".writer.lock"
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"another profile preparation is writing {progress}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def ensure_profile_bundle(
    profile_path: Path,
    *,
    profile_inputs: Sequence[ProfileExtractionInput],
    extractor: LusiferProfileExtractor,
) -> Path:
    """Persist each completed user before continuing, then assemble one bundle.

    Successful users are reused only for the exact input/prompt/provider
    request. A failed or interrupted user is retried on the next explicit
    invocation, with a new attempt directory. All previous attempt requests,
    failures, and incomplete files remain available for diagnosis. A partly
    completed user's chunks may repeat; whole completed users never repeat.
    """
    path = Path(profile_path)
    inputs = tuple(sorted(profile_inputs, key=lambda record: record.user_id))
    if not inputs or any(not isinstance(record, ProfileExtractionInput) for record in inputs):
        raise ValueError("profile_inputs must contain ProfileExtractionInput values")
    if len({record.user_id for record in inputs}) != len(inputs):
        raise ValueError("profile inputs contain duplicate users")
    if len({record.source_user_id for record in inputs}) != len(inputs):
        raise ValueError("profile inputs contain duplicate source IDs")
    if any(not record.interactions or any(event.source_split != "train" for event in record.interactions)
           for record in inputs):
        raise ValueError("profile preparation requires nonempty training-only evidence")
    request = _profile_request(inputs, extractor)
    if path.exists():
        _validate_completed_profile_bundle(path, inputs, request)
        return path
    progress = path.with_name(path.name + ".progress")
    progress.mkdir(parents=True, exist_ok=True)
    with _profile_progress_lock(progress):
        if path.exists():
            _validate_completed_profile_bundle(path, inputs, request)
            return path
        request_path = progress / "request.json"
        if request_path.exists():
            if _read_json(request_path) != request:
                raise ValueError("profile progress input/prompt/provider request differs; use a new path")
        else:
            save_json_artifact(request_path, request)
        successful = []
        diagnostics = []
        failures = []
        for record in inputs:
            user_root = progress / f"user_{record.user_id}"
            user_root.mkdir(exist_ok=True)
            attempts = sorted(user_root.glob("attempt_*"))
            completed = [attempt / "result" for attempt in attempts
                         if (attempt / "result" / "_COMPLETE.json").exists()]
            if len(completed) > 1:
                raise ValueError(f"multiple completed profiles exist for user {record.user_id}")
            if completed:
                user_request = _profile_request((record,), extractor)
                cached = _validate_completed_profile_bundle(completed[0], (record,), user_request)
                successful.append(cached[record.user_id])
                diagnostics.append({"user_id": record.user_id, "status": "reused",
                                    "attempt": str(completed[0].relative_to(progress))})
                logger.info("Reused completed mitigation profile user=%s", record.user_id)
                continue
            attempt = user_root / f"attempt_{len(attempts) + 1:06d}"
            attempt.mkdir()
            save_json_artifact(attempt / "request.json", {
                "user_id": record.user_id, "input": record,
                "preparation_request": _profile_request((record,), extractor),
            })
            logger.info("Mitigation profile extraction started user=%s attempt=%s", record.user_id, attempt.name)
            try:
                attempt_extractor = copy(extractor)
                attempt_extractor.provider = _ProfileProgressProvider(extractor.provider, attempt)
                result = attempt_extractor.extract(record)
                if isinstance(result, ProfileExtractionFailure):
                    save_json_artifact(attempt / "failure.json", result)
                    failures.append(record.user_id)
                    diagnostics.append({"user_id": record.user_id, "status": "failed",
                                        "attempt": str(attempt.relative_to(progress))})
                    logger.error("Mitigation profile extraction failed user=%s reason=%s", record.user_id, result.reason)
                    continue
                if not isinstance(result, ProfileArtifact):
                    raise TypeError("extractor returned neither ProfileArtifact nor ProfileExtractionFailure")
                for field in ("extractor_id", "prompt_fingerprint", "method_settings", "provider_settings"):
                    if getattr(result, field) != getattr(extractor, field):
                        raise ValueError(f"profile result has incompatible {field}")
                save_profile_bundle(
                    attempt / "result", inputs=(record,), profiles=(result,),
                    metadata={"preparation_request": _profile_request((record,), extractor)},
                )
            except BaseException as exc:
                logger.exception("Mitigation profile extraction interrupted user=%s attempt=%s", record.user_id, attempt)
                save_json_artifact(attempt / "exception.json", {
                    "user_id": record.user_id, "error_type": type(exc).__name__, "error": str(exc),
                })
                raise
            successful.append(result)
            diagnostics.append({"user_id": record.user_id, "status": "generated",
                                "attempt": str(attempt.relative_to(progress))})
            logger.info("Mitigation profile persisted user=%s completed=%s/%s", record.user_id, len(successful), len(inputs))
        if failures:
            raise RuntimeError(f"profile extraction failed for users {failures}; progress preserved at {progress}")
        # New assembly directories preserve a failed final write and allow a
        # restart to assemble already-completed users without any provider call.
        assembly = progress / "assemblies" / uuid4().hex / "bundle"
        save_profile_bundle(
            assembly, inputs=inputs, profiles=successful, diagnostics=diagnostics,
            metadata={"purpose": "mitigation_frozen_profiles",
                      "preparation_request": request,
                      "progress_directory": progress.name},
        )
        rename_directory_with_retry(assembly, path, logger=logger)
        _validate_completed_profile_bundle(path, inputs, request)
    return path


def _profile_inputs(
    *,
    raw_data_directory: Path,
    dataset,
    population,
    extraction_seed: int,
) -> tuple[ProfileExtractionInput, ...]:
    selected = tuple(
        member
        for member in population.members
        if member.role in {PopulationRole.ACTIVE, PopulationRole.RESERVE}
    )
    runtime_to_source = {
        member.user_id: member.source_user_id for member in selected
    }
    user_texts = match_movielens_1m_user_texts(
        runtime_to_source,
        parse_movielens_1m_users(Path(raw_data_directory) / "users.dat"),
    )
    records: list[ProfileExtractionInput] = []
    for member in sorted(selected, key=lambda value: value.user_id):
        matched = user_texts[member.user_id]
        interactions = tuple(
            ProfileInteraction(
                event_id=f"rating-row-{event.source_row}",
                source_order=event.source_row,
                source_split="train",
                item_id=event.item_id,
                source_item_id=event.source_item_id,
                timestamp=event.timestamp,
                rating=float(event.rating),
                item_text=str(dataset.catalog[event.item_id]["item_text"]),
            )
            for event in dataset.training_interactions[member.user_id]
        )
        records.append(
            ProfileExtractionInput(
                user_id=member.user_id,
                source_user_id=member.source_user_id,
                user_text=matched.user_text,
                user_text_builder_version=matched.user_text_builder_version,
                user_text_fingerprint=matched.user_text_fingerprint,
                interactions=interactions,
                training_history_fingerprint=(
                    dataset.training_history_fingerprints[member.user_id]
                ),
                dataset_fingerprint=dataset.dataset_fingerprint,
                extraction_seed=extraction_seed,
            )
        )
    return tuple(records)


def _validate_population_request(
    population,
    *,
    population_seed: int,
    active_user_count: int,
    reserve_user_count: int,
) -> None:
    if population.population_seed != population_seed:
        raise ValueError("automatic population seed disagrees with the request")
    active = sum(
        member.role is PopulationRole.ACTIVE for member in population.members
    )
    reserve = sum(
        member.role is PopulationRole.RESERVE for member in population.members
    )
    if (active, reserve) != (active_user_count, reserve_user_count):
        raise ValueError("automatic population size disagrees with the request")


def _reject_incomplete(path: Path) -> None:
    incomplete = path.with_name(path.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(f"incomplete automatic input exists: {incomplete}")


__all__ = ["AutomaticInputPaths", "ensure_automatic_inputs", "ensure_profile_bundle"]
