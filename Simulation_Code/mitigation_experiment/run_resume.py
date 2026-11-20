"""Restore a completed iteration into a new run, leaving its source untouched."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .contracts import SimulationCheckpoint
from .run_artifacts import RUN_SCHEMA
from .runtime_setup import SimulationRuntime
from .utility.checkpoints import load_checkpoint
from .utility.saving import RUN_OUTPUT_SCHEMA, RunOutputWriter, _file_sha256, _json_value


def validate_resume_destination(checkpoint_dir: Path, output_root: Path) -> Path:
    """Check paths before even opening an attempt log in the destination."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if checkpoint_dir.parent.name != "checkpoints":
        raise ValueError("--resume-from must name an iteration directory inside run/checkpoints")
    source = checkpoint_dir.parent.parent
    output_root = Path(output_root).resolve()
    if output_root == source.parent or output_root.is_relative_to(source):
        raise ValueError("restart requires a new output root outside the source run")
    return source


@dataclass(frozen=True)
class ResumeSource:
    root: Path
    checkpoint_dir: Path
    checkpoint: SimulationCheckpoint
    manifest: Mapping[str, Any]
    checkpoint_sha256: str
    # Byte lengths and hashes cover only completed records, excluding the tail.
    prefixes: Mapping[str, tuple[int, str]]

    @classmethod
    def load(
        cls, checkpoint_dir: Path, output_root: Path, *,
        run_id: str, condition_id: str, seed: int,
    ) -> "ResumeSource":
        root = validate_resume_destination(checkpoint_dir, output_root)
        checkpoint_dir = Path(checkpoint_dir).resolve()
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        digest = _file_sha256(checkpoint_path)
        checkpoint = load_checkpoint(
            checkpoint_dir, expected_run_id=run_id,
            expected_condition_id=condition_id, expected_seed=seed,
        )
        if _file_sha256(checkpoint_path) != digest:
            raise ValueError("source checkpoint changed while loading")
        iteration = checkpoint.completed_iteration
        if (
            checkpoint.schema_version != RUN_SCHEMA
            or isinstance(iteration, bool) or not isinstance(iteration, int)
            or iteration < 1
            or checkpoint_dir.name != f"iteration_{iteration:04d}"
        ):
            raise ValueError("unsupported run checkpoint or completed iteration")
        payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if payload.get("file_schema") != RUN_OUTPUT_SCHEMA or payload.get("run_id") != run_id:
            raise ValueError("source manifest does not match the checkpoint run")
        manifest = payload["resolved_manifest"]
        # Diagnostics establish which users and probe arms must be present.
        diagnostics, active_users = _diagnostic_prefix(root / "diagnostics.jsonl", iteration)
        arms = _presentation_arms(manifest, checkpoint)
        expected = {
            (completed, user_id, arm)
            for completed, users in active_users.items()
            for user_id in users for arm in arms
        }
        events = _event_prefix(root / "events.jsonl", checkpoint, expected)
        return cls(
            root, checkpoint_dir, checkpoint, manifest, digest,
            {"events.jsonl": events, "diagnostics.jsonl": diagnostics},
        )

    def restore(self, state: SimulationRuntime, manifest: Mapping[str, Any]) -> None:
        previous = dict(self.manifest)
        previous.pop("resume", None)
        current = _json_value(manifest)
        # Source metadata is informational, including checksums in older runs.
        for key in ("git", "implementation_code_fingerprint", "source_code_fingerprint"):
            previous.pop(key, None)
            current.pop(key, None)
        mismatches = sorted(key for key in set(previous) | set(current) if previous.get(key) != current.get(key))
        if mismatches:
            raise ValueError("restart settings differ from source: " + ", ".join(mismatches))
        checkpoint = self.checkpoint
        if checkpoint.completed_iteration > current["iterations"]:
            raise ValueError("checkpoint is beyond the configured final iteration")
        if dict(checkpoint.input_fingerprints) != current["inputs"]:
            raise ValueError("restart input fingerprints differ from checkpoint")
        if dict(checkpoint.randomness_state) != {"simulation_seed": checkpoint.seed}:
            raise ValueError("unsupported checkpoint randomness state")
        components = checkpoint.component_states
        if tuple(components["training_user_ids"]) != state.training_user_ids:
            raise ValueError("restart training users differ from checkpoint")
        for name in ("observed_histories", "recommender_training_histories"):
            histories = components[name]
            if set(histories) != set(getattr(state, name)):
                raise ValueError(f"restart {name} users differ from checkpoint")
            setattr(state, name, {user: list(items) for user, items in histories.items()})
        state.recommender.restore_state(components["recommender"])
        state.user_model.restore_state(components["user_model"])
        policies = checkpoint.policy_states
        for name, policy in (
            ("candidate", state.candidate_policy), ("presentation", state.slate_policy),
            ("population", state.population_policy), ("campaign", state.campaign_policy),
            ("order_probe", state.order_probe_policy),
        ):
            saved = policies[name]
            if (policy is None) != (saved is None):
                raise ValueError(f"restart {name} policy differs from checkpoint")
            if policy is not None:
                policy.restore_state(saved)

    def provenance(self) -> dict[str, Any]:
        return {
            "source_run": str(self.root),
            "source_checkpoint": str(self.checkpoint_dir / "checkpoint.pt"),
            "source_checkpoint_sha256": self.checkpoint_sha256,
            "completed_iteration": self.checkpoint.completed_iteration,
            "copied_prefixes": {
                name: {"bytes": length, "sha256": digest}
                for name, (length, digest) in self.prefixes.items()
            },
        }

    def copy_prefix(self, writer: RunOutputWriter) -> Path:
        """Copy checked bytes to new files; never open any source for writing."""
        for name, (length, digest) in self.prefixes.items():
            _copy_bytes(self.root / name, writer.root / name, length, digest, mode="ab")
        destination = writer.root / "checkpoints" / self.checkpoint_dir.name
        destination.mkdir()
        source = self.checkpoint_dir / "checkpoint.pt"
        _copy_bytes(
            source, destination / source.name, source.stat().st_size,
            self.checkpoint_sha256, mode="xb",
        )
        writer.write_diagnostic({
            "event_name": "run_resumed", "run_id": writer.run_id,
            "iteration": self.checkpoint.completed_iteration, **self.provenance(),
        })
        return destination / source.name


def _presentation_arms(manifest: Mapping[str, Any], checkpoint: SimulationCheckpoint) -> tuple[str, ...]:
    """Validate declared roles against saved policy state, never infer by name."""
    declared = manifest.get("presentation_arms")
    if not isinstance(declared, (tuple, list)):
        raise ValueError("restart manifest must explicitly declare presentation_arms")
    arms = tuple(declared)
    if any(not isinstance(arm, str) for arm in arms):
        raise ValueError("restart presentation arms must be strings")
    allowed = {
        ("canonical",),
        ("canonical", "replica", "shuffled"),
        ("canonical", "replica", "shuffled", "original_order_probe"),
    }
    if arms not in allowed:
        raise ValueError("restart manifest has unsupported or duplicated presentation arms")
    saved_probe = checkpoint.policy_states.get("order_probe")
    if saved_probe is not None and not isinstance(saved_probe, Mapping):
        raise ValueError("restart checkpoint has invalid probe policy state")
    saved_arms = ("canonical",) if saved_probe is None else tuple(saved_probe.get("arms", ()))
    if saved_arms != arms:
        raise ValueError("restart presentation arms differ from checkpoint policy")
    return arms


def _diagnostic_prefix(path: Path, iteration: int) -> tuple[tuple[int, str], dict[int, list[int]]]:
    digest = hashlib.sha256()
    active_users = {}
    with path.open("rb") as stream:
        for line in stream:
            record = json.loads(line)
            if not line.endswith(b"\n") or record.get("event_name") == "run_failed":
                raise ValueError("source diagnostics are incomplete before the checkpoint boundary")
            digest.update(line)
            if record.get("event_name") == "iteration_complete":
                completed = record["iteration"]
                if completed != len(active_users) + 1:
                    raise ValueError("source iteration-completion records are missing or duplicated")
                users = record["active_user_ids"]
                if len(set(users)) != len(users):
                    raise ValueError("source iteration has duplicate active users")
                active_users[completed] = users
                if completed == iteration:
                    return (stream.tell(), digest.hexdigest()), active_users
    raise ValueError("checkpoint lacks its iteration_complete diagnostic; choose an earlier checkpoint")


def _event_prefix(
    path: Path, checkpoint: SimulationCheckpoint, expected: set[tuple[int, int, str]],
) -> tuple[int, str]:
    remaining = set(expected)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while remaining:
            line = stream.readline()
            if not line.endswith(b"\n"):
                raise ValueError("source events are incomplete before the checkpoint boundary")
            record = json.loads(line)
            key = (record["iteration"], record["user_id"], record["presentation_metadata"]["probe_arm"])
            expected_id = f"{checkpoint.run_id}__t{key[0]:04d}__u{key[1]}__{key[2]}"
            if (
                key not in remaining or record["interaction_id"] != expected_id
                or record["run_id"] != checkpoint.run_id
                or record["condition_id"] != checkpoint.condition_id
                or record["seed"] != checkpoint.seed
            ):
                raise ValueError("source event prefix has missing, duplicated, or mismatched turns")
            remaining.remove(key)
            digest.update(line)
        length = stream.tell()
        # A torn next-iteration tail is harmless once every committed turn is present.
        tail = stream.readline()
        if tail:
            try:
                extra = json.loads(tail)
            except (ValueError, UnicodeError):
                extra = None
            if isinstance(extra, dict) and extra.get("iteration", checkpoint.completed_iteration + 1) <= checkpoint.completed_iteration:
                raise ValueError("source contains extra events at or before the checkpoint")
    return length, digest.hexdigest()


def _copy_bytes(source: Path, destination: Path, length: int, expected_hash: str, *, mode: str) -> None:
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open(mode) as writer:
        remaining = length
        while remaining:
            data = reader.read(min(remaining, 1024 * 1024))
            if not data:
                raise ValueError("source was truncated during restart")
            writer.write(data)
            digest.update(data)
            remaining -= len(data)
        writer.flush()
        os.fsync(writer.fileno())
    if digest.hexdigest() != expected_hash:
        raise ValueError("source changed during restart")
