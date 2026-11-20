"""Read-only validation of existing shared Goodreads profile evidence."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .runtime_adapter import load_book_inputs


def build_profile_inputs(inputs_root: str | Path, *, extraction_seed: int = 1054,
                         item_summaries_path: str | Path | None = None):
    from .engine.persona_extraction.base import ProfileExtractionInput, ProfileInteraction

    root = Path(inputs_root)
    dataset = load_book_inputs(root / "prepared", root / "population.json").dataset
    texts = None
    if item_summaries_path is not None:
        from .item_summaries import load_item_summaries
        overlay = load_item_summaries(item_summaries_path, prepared_path=root / "prepared")
        if overlay["manifest"]["source_dataset_fingerprint"] != dataset.dataset_fingerprint:
            raise ValueError("profile item summaries belong to a different dataset")
        texts = overlay["item_text_by_id"]
        if set(texts) != set(dataset.catalog):
            raise ValueError("profile item summaries must cover the exact catalogue")
    records = []
    for uid in sorted(dataset.training_interactions):
        selected = dataset.training_interactions[uid][-40:]
        interactions = tuple(ProfileInteraction(event_id=f"goodreads:{uid}:{event.source_row}", source_order=event.source_row, source_split="train", item_id=event.item_id,
            source_item_id=dataset.catalog[event.item_id]["original_source_item_id"], timestamp=event.timestamp, rating=event.rating,
            item_text=(dataset.catalog[event.item_id]["item_text"] if texts is None else texts[event.item_id])) for event in selected)
        records.append(ProfileExtractionInput(user_id=uid, source_user_id=dataset.runtime_to_source_user_id[uid], user_text=None,
            user_text_builder_version=None, user_text_fingerprint=None, interactions=interactions,
            training_history_fingerprint=dataset.training_history_fingerprints[uid], dataset_fingerprint=dataset.dataset_fingerprint, extraction_seed=extraction_seed))
    return tuple(records)


def _summary_metadata(item_summaries_path: str | Path, manifest: dict[str, Any], *, extraction_seed: int) -> dict[str, Any]:
    return {"artifact_path": str(Path(item_summaries_path).resolve()),
            "item_text_fingerprint": manifest["item_text_fingerprint"],
            "policy_fingerprint": manifest["policy_fingerprint"],
            "source_dataset_fingerprint": manifest["source_dataset_fingerprint"],
            "description_max_chars": manifest["policy"]["description_max_chars"],
            "extraction_seed": extraction_seed}


def validate_profile_item_summaries(inputs_root: str | Path, profile_path: str | Path,
                                    item_summaries_path: str | Path, *,
                                    expected_item_text_fingerprint: str | None = None) -> None:
    """Check the complete extraction evidence before using enriched agent profiles.

    The original prepared dataset/history identities stay unchanged. The text
    overlay is bound separately to the saved extraction inputs and profiles.
    """
    from .item_summaries import file_sha256, load_item_summaries
    from .engine.persona_extraction.lusifer_adapter import LusiferProfileExtractor

    root, output = Path(inputs_root), Path(profile_path)
    overlay = load_item_summaries(item_summaries_path, prepared_path=root / "prepared")
    item_text_fingerprint = overlay["manifest"]["item_text_fingerprint"]
    if expected_item_text_fingerprint is not None and item_text_fingerprint != expected_item_text_fingerprint:
        raise ValueError("profile item-summary fingerprint differs from the configured policy")
    complete = json.loads((output / "_COMPLETE.json").read_text(encoding="utf-8"))
    required = {"manifest.json", "inputs.jsonl", "profiles.jsonl", "failures.jsonl", "diagnostics.jsonl"}
    hashes = complete.get("file_sha256", {})
    if complete.get("status") != "complete" or set(hashes) != required:
        raise ValueError("profile summary policy requires a complete, checksummed profile bundle")
    for filename, expected in hashes.items():
        if file_sha256(output / filename) != expected:
            raise ValueError(f"profile summary-policy evidence checksum mismatch: {filename}")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    actual = manifest.get("metadata", {}).get("llm_item_text")
    if not isinstance(actual, dict):
        raise ValueError("agent profiles were not extracted with the configured item-summary policy")
    extraction_seed = actual.get("extraction_seed")
    if isinstance(extraction_seed, bool) or not isinstance(extraction_seed, int) or extraction_seed < 0:
        raise ValueError("profile summary policy has no valid extraction seed")
    if actual != _summary_metadata(item_summaries_path, overlay["manifest"], extraction_seed=extraction_seed):
        raise ValueError("agent profile item-summary policy does not match the configured overlay")
    expected_inputs = build_profile_inputs(root, extraction_seed=extraction_seed,
                                          item_summaries_path=item_summaries_path)
    # JSON roundtrip normalises dataclass tuples exactly as the bundle writer.
    expected_rows = json.loads(json.dumps([asdict(record) for record in expected_inputs]))
    rows = [json.loads(line) for line in (output / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    if rows != expected_rows:
        raise ValueError("profile extraction inputs do not match the configured item summaries")
    loaded = load_book_inputs(root / "prepared", root / "population.json", output)
    count = len(expected_inputs)
    if (complete.get("input_count") != count or complete.get("profile_count") != count
            or complete.get("failure_count") != 0 or manifest.get("failure_count") != 0):
        raise ValueError("item-summary profiles must cover every reader without extraction failures")
    for record in expected_inputs:
        ordered = tuple(sorted(record.interactions, key=lambda event: (event.timestamp, event.source_order)))
        expected_fp = LusiferProfileExtractor._input_fingerprint(record, ordered)
        if loaded.profiles_by_user[record.user_id].input_fingerprint != expected_fp:
            raise ValueError("saved agent profile was extracted from different item-summary inputs")


def prepare_profiles(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("Profile generation is disabled in the mitigation extension; reuse frozen Goodreads profiles.")


def main(argv: Any = None) -> int:
    raise SystemExit("This is a read-only validation helper; use the mitigation campaign CLI.")


if __name__ == "__main__":
    main()
