"""Small, concrete, non-overwriting persistence helpers.

This module owns filesystem writes only.  It does not resolve conditions,
construct models, or evaluate results.  Every bundle is created at a new path;
partial work remains visibly incomplete and is never silently reused.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from ..contracts import RunArtifact, TurnEvent
from ..persona_extraction.base import (
    ProfileArtifact,
    ProfileExtractionFailure,
    ProfileExtractionInput,
)
from .checkpoints import load_checkpoint
from .fingerprints import fingerprint_semantic_state
from .filesystem import rename_directory_with_retry
from .user_handling import PopulationArtifact


POPULATION_FILE_SCHEMA = "feedback-loop-population-file-v1"
PROFILE_BUNDLE_SCHEMA = "feedback-loop-profile-bundle-v1"
RUN_OUTPUT_SCHEMA = "feedback-loop-run-output-v1"
logger = logging.getLogger(__name__)


def _json_value(value: Any) -> Any:
    """Convert contract values to deterministic, strict JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise TypeError("JSON artifact mapping keys must be strings or integers")
            normalized = str(key)
            if normalized in converted:
                raise ValueError(f"JSON artifact has duplicate key {normalized!r}")
            converted[normalized] = _json_value(item)
        return converted
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON artifact numbers must be finite")
        return value
    # NumPy scalar values occasionally occur in model diagnostics.  Convert
    # only scalars; arrays belong in model checkpoints rather than JSON logs.
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        scalar = value.item()
        if scalar is not value:
            return _json_value(scalar)
    raise TypeError(f"unsupported JSON artifact value: {type(value).__name__}")


def _json_text(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write_text(destination: Path, text: str) -> Path:
    """Write one new UTF-8 file through a visible incomplete sibling."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"artifact already exists: {destination}")
    temporary = destination.with_name(destination.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"incomplete artifact already exists: {temporary}")

    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    # On Windows rename refuses an existing destination, preserving the
    # project's no-overwrite contract.  The explicit check above also gives a
    # clearer error before any work in ordinary single-writer use.
    os.rename(temporary, destination)
    return destination


def save_json_artifact(destination: str | Path, value: Any) -> Path:
    """Atomically save one new deterministic JSON artifact."""

    return _atomic_write_text(Path(destination), _json_text(value) + "\n")


def _save_jsonl(destination: Path, records: Sequence[Any]) -> Path:
    text = "".join(_json_text(record) + "\n" for record in records)
    return _atomic_write_text(destination, text)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rename_run_directory(source: Path, destination: Path) -> None:
    """Atomically commit a run, tolerating short Windows filesystem locks."""

    rename_directory_with_retry(
        source,
        destination,
        rename=os.rename,
        logger=logger,
        sleep=time.sleep,
    )


def _line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def save_population_artifact(
    destination: str | Path,
    population: PopulationArtifact,
) -> Path:
    """Save one already-constructed population artifact."""

    if not isinstance(population, PopulationArtifact):
        raise TypeError("population must be a PopulationArtifact")
    population_payload = {
        field.name: getattr(population, field.name)
        for field in fields(population)
        if field.name != "population_fingerprint"
    }
    expected_fingerprint = fingerprint_semantic_state(
        population_payload,
        population.schema_version,
    )
    if population.population_fingerprint != expected_fingerprint:
        raise ValueError("population artifact fingerprint mismatch")
    return save_json_artifact(
        destination,
        {
            "file_schema": POPULATION_FILE_SCHEMA,
            "population": population,
        },
    )


@dataclass(frozen=True)
class ProfileBundlePaths:
    root: Path
    manifest_path: Path
    inputs_path: Path
    profiles_path: Path
    failures_path: Path
    diagnostics_path: Path
    complete_path: Path


def save_profile_bundle(
    directory: str | Path,
    *,
    inputs: Sequence[ProfileExtractionInput],
    profiles: Sequence[ProfileArtifact],
    failures: Sequence[ProfileExtractionFailure] = (),
    diagnostics: Sequence[Mapping[str, Any]] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ProfileBundlePaths:
    """Save one completed profile-extraction bundle at a new directory."""

    input_values = tuple(inputs)
    profile_values = tuple(profiles)
    failure_values = tuple(failures)
    diagnostic_values = tuple(diagnostics)
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    if not input_values:
        raise ValueError("profile inputs cannot be empty")
    if any(not isinstance(value, ProfileExtractionInput) for value in input_values):
        raise TypeError("inputs must contain ProfileExtractionInput values")
    if any(not isinstance(value, ProfileArtifact) for value in profile_values):
        raise TypeError("profiles must contain ProfileArtifact values")
    if any(
        not isinstance(value, ProfileExtractionFailure) for value in failure_values
    ):
        raise TypeError("failures must contain ProfileExtractionFailure values")
    if any(not isinstance(value, Mapping) for value in diagnostic_values):
        raise TypeError("diagnostics must contain mappings")

    input_users = [value.user_id for value in input_values]
    profile_users = [value.user_id for value in profile_values]
    failure_users = [value.user_id for value in failure_values]
    if len(input_users) != len(set(input_users)):
        raise ValueError("inputs contain duplicate users")
    if len(profile_users) != len(set(profile_users)):
        raise ValueError("profiles contain duplicate users")
    if len(failure_users) != len(set(failure_users)):
        raise ValueError("failures contain duplicate users")
    overlap = sorted(set(profile_users).intersection(failure_users))
    if overlap:
        raise ValueError(f"users cannot be both successful and failed: {overlap}")
    if set(input_users) != set(profile_users).union(failure_users):
        raise ValueError("every profile input must have exactly one result")
    inputs_by_user = {value.user_id: value for value in input_values}
    for result in (*profile_values, *failure_values):
        input_record = inputs_by_user[result.user_id]
        if result.source_user_id != input_record.source_user_id:
            raise ValueError(
                f"profile result source user mismatch for user {result.user_id}"
            )
        if result.extraction_seed != input_record.extraction_seed:
            raise ValueError(
                f"profile result extraction seed mismatch for user {result.user_id}"
            )
    source_user_ids = [
        value.source_user_id for value in (*profile_values, *failure_values)
    ]
    if len(source_user_ids) != len(set(source_user_ids)):
        raise ValueError("profile results contain duplicate source users")
    for profile in profile_values:
        input_record = inputs_by_user[profile.user_id]
        if profile.dataset_fingerprint != input_record.dataset_fingerprint:
            raise ValueError(
                f"profile dataset fingerprint mismatch for user {profile.user_id}"
            )
        if (
            profile.training_history_fingerprint
            != input_record.training_history_fingerprint
        ):
            raise ValueError(
                f"profile training-history fingerprint mismatch for user {profile.user_id}"
            )
        profile_payload = {
            field.name: getattr(profile, field.name)
            for field in fields(profile)
            if field.name != "profile_fingerprint"
        }
        expected_fingerprint = fingerprint_semantic_state(
            profile_payload,
            profile.schema_version,
        )
        if profile.profile_fingerprint != expected_fingerprint:
            raise ValueError(
                f"profile artifact fingerprint mismatch for user {profile.user_id}"
            )

    root = Path(directory)
    staging = root.with_name(root.name + ".incomplete")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise FileExistsError(f"profile bundle already exists: {root}")
    if staging.exists():
        raise FileExistsError(f"incomplete profile bundle already exists: {staging}")
    staging.mkdir()
    manifest_path = staging / "manifest.json"
    inputs_path = staging / "inputs.jsonl"
    profiles_path = staging / "profiles.jsonl"
    failures_path = staging / "failures.jsonl"
    diagnostics_path = staging / "diagnostics.jsonl"
    complete_path = staging / "_COMPLETE.json"

    ordered_inputs = tuple(sorted(input_values, key=lambda value: value.user_id))
    ordered_profiles = tuple(sorted(profile_values, key=lambda value: value.user_id))
    ordered_failures = tuple(sorted(failure_values, key=lambda value: value.user_id))
    manifest = {
        "file_schema": PROFILE_BUNDLE_SCHEMA,
        "input_count": len(ordered_inputs),
        "profile_count": len(ordered_profiles),
        "failure_count": len(ordered_failures),
        "profile_fingerprints": {
            value.user_id: value.profile_fingerprint for value in ordered_profiles
        },
        "metadata": dict(metadata or {}),
    }
    save_json_artifact(manifest_path, manifest)
    _save_jsonl(inputs_path, ordered_inputs)
    _save_jsonl(profiles_path, ordered_profiles)
    _save_jsonl(failures_path, ordered_failures)
    _save_jsonl(diagnostics_path, diagnostic_values)
    save_json_artifact(
        complete_path,
        {
            "file_schema": PROFILE_BUNDLE_SCHEMA,
            "status": "complete",
            "input_count": len(ordered_inputs),
            "profile_count": len(ordered_profiles),
            "failure_count": len(ordered_failures),
            "diagnostic_count": len(diagnostic_values),
            "file_sha256": {
                "manifest.json": _file_sha256(manifest_path),
                "inputs.jsonl": _file_sha256(inputs_path),
                "profiles.jsonl": _file_sha256(profiles_path),
                "failures.jsonl": _file_sha256(failures_path),
                "diagnostics.jsonl": _file_sha256(diagnostics_path),
            },
        },
    )
    rename_directory_with_retry(staging, root, logger=logger)
    return ProfileBundlePaths(
        root=root,
        manifest_path=root / manifest_path.name,
        inputs_path=root / inputs_path.name,
        profiles_path=root / profiles_path.name,
        failures_path=root / failures_path.name,
        diagnostics_path=root / diagnostics_path.name,
        complete_path=root / complete_path.name,
    )


class RunOutputWriter:
    """Append normalized run records to one newly created output directory."""

    def __init__(
        self,
        *,
        root: Path,
        final_root: Path,
        run_id: str,
        manifest_path: Path,
        events_path: Path,
        diagnostics_path: Path,
    ) -> None:
        self.root = root
        self.final_root = final_root
        self.run_id = run_id
        self.manifest_path = manifest_path
        self.events_path = events_path
        self.diagnostics_path = diagnostics_path
        self._complete = False

    @classmethod
    def create(
        cls,
        directory: str | Path,
        *,
        run_id: str,
        manifest: Mapping[str, Any],
    ) -> "RunOutputWriter":
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-blank string")
        if not isinstance(manifest, Mapping):
            raise TypeError("manifest must be a mapping")

        final_root = Path(directory)
        root = final_root.with_name(final_root.name + ".incomplete")
        final_root.parent.mkdir(parents=True, exist_ok=True)
        if final_root.exists():
            raise FileExistsError(f"run output already exists: {final_root}")
        if root.exists():
            raise FileExistsError(f"incomplete run output already exists: {root}")
        root.mkdir()
        (root / "checkpoints").mkdir()
        manifest_path = root / "manifest.json"
        events_path = root / "events.jsonl"
        diagnostics_path = root / "diagnostics.jsonl"

        save_json_artifact(
            manifest_path,
            {
                "file_schema": RUN_OUTPUT_SCHEMA,
                "run_id": run_id,
                "resolved_manifest": dict(manifest),
            },
        )
        for path in (events_path, diagnostics_path):
            _atomic_write_text(path, "")
        save_json_artifact(
            root / "_INCOMPLETE.json",
            {
                "file_schema": RUN_OUTPUT_SCHEMA,
                "run_id": run_id,
                "status": "incomplete",
            },
        )
        return cls(
            root=root,
            final_root=final_root,
            run_id=run_id,
            manifest_path=manifest_path,
            events_path=events_path,
            diagnostics_path=diagnostics_path,
        )

    def _append(self, path: Path, value: Any) -> None:
        if self._complete or (self.root / "_COMPLETE.json").exists():
            raise RuntimeError("cannot write to a finalized run")
        line = _json_text(value) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def write_event(self, event: TurnEvent) -> None:
        if not isinstance(event, TurnEvent):
            raise TypeError("event must be a TurnEvent")
        if event.run_id != self.run_id:
            raise ValueError("event run_id does not match output run_id")
        self._append(self.events_path, event)

    def write_diagnostic(self, diagnostic: Mapping[str, Any]) -> None:
        if not isinstance(diagnostic, Mapping):
            raise TypeError("diagnostic must be a mapping")
        self._append(self.diagnostics_path, diagnostic)

    def record_failure(self, error: BaseException, *, iteration: int, stage: str) -> None:
        """Keep a failure marker even if finalization already sealed diagnostics.

        Each destination is attempted independently. Logging failures must not
        replace the exception that stopped the simulation.
        """
        diagnostic = {
            "event_name": "run_failed", "run_id": self.run_id,
            "iteration": iteration, "stage": stage,
            "exception_type": type(error).__name__, "error": str(error),
        }
        try:
            save_json_artifact(self.root / "_FAILED.json", diagnostic)
        except Exception:
            logger.exception("Could not save the run failure marker")
        try:
            if not (self.root / "_COMPLETE.json").exists():
                self.write_diagnostic(diagnostic)
        except Exception:
            logger.exception("Could not save the failure diagnostic")

    def finalize(
        self,
        *,
        final_checkpoint_path: str | Path | None,
        note: str = "complete",
    ) -> RunArtifact:
        if self._complete:
            raise RuntimeError("run is already finalized")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("note must be a non-blank string")
        checkpoint = (
            Path(final_checkpoint_path) if final_checkpoint_path is not None else None
        )
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(f"final checkpoint does not exist: {checkpoint}")
        final_checkpoint: Path | None = None
        checkpoint_sha256: str | None = None
        checkpoint_iteration: int | None = None
        if checkpoint is not None:
            try:
                relative_checkpoint = checkpoint.resolve().relative_to(
                    (self.root / "checkpoints").resolve()
                )
            except ValueError as exc:
                raise ValueError(
                    "final checkpoint must be inside the run checkpoints directory"
                ) from exc
            if checkpoint.name != "checkpoint.pt":
                raise ValueError("final checkpoint must name checkpoint.pt")
            loaded_checkpoint = load_checkpoint(
                checkpoint.parent,
                expected_run_id=self.run_id,
            )
            relative_checkpoint = Path("checkpoints") / relative_checkpoint
            final_checkpoint = self.final_root / relative_checkpoint
            checkpoint_sha256 = _file_sha256(checkpoint)
            checkpoint_iteration = loaded_checkpoint.completed_iteration

        save_json_artifact(
            self.root / "_COMPLETE.json",
            {
                "file_schema": RUN_OUTPUT_SCHEMA,
                "run_id": self.run_id,
                "status": "complete",
                "final_checkpoint_path": (
                    str(final_checkpoint.relative_to(self.final_root))
                    if final_checkpoint is not None
                    else None
                ),
                "final_checkpoint_sha256": checkpoint_sha256,
                "completed_iteration": checkpoint_iteration,
                "note": note,
                "row_counts": {
                    "events": _line_count(self.events_path),
                    "diagnostics": _line_count(self.diagnostics_path),
                },
                "file_sha256": {
                    "manifest.json": _file_sha256(self.manifest_path),
                    "events.jsonl": _file_sha256(self.events_path),
                    "diagnostics.jsonl": _file_sha256(self.diagnostics_path),
                },
            },
        )
        incomplete = self.root / "_INCOMPLETE.json"
        incomplete.unlink()
        _rename_run_directory(self.root, self.final_root)
        self._complete = True
        return RunArtifact(
            run_id=self.run_id,
            output_dir=self.final_root,
            manifest_path=self.final_root / self.manifest_path.name,
            events_path=self.final_root / self.events_path.name,
            diagnostics_path=self.final_root / self.diagnostics_path.name,
            final_checkpoint_path=final_checkpoint,
        )


__all__ = [
    "POPULATION_FILE_SCHEMA",
    "PROFILE_BUNDLE_SCHEMA",
    "RUN_OUTPUT_SCHEMA",
    "ProfileBundlePaths",
    "RunOutputWriter",
    "save_json_artifact",
    "save_population_artifact",
    "save_profile_bundle",
]
