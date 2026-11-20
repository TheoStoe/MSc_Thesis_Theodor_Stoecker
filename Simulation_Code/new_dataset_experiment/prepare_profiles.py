"""Describe or explicitly extract shared book profiles from the latest 40 training events."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

from .runtime_adapter import PROVIDER_SETTINGS, load_book_inputs


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


def prepare_profiles(inputs_root: str | Path, *, extraction_seed: int = 1054, provider: Any = None, provider_id: str = "openai", provider_settings: dict[str, Any] | None = None,
                     item_summaries_path: str | Path | None = None,
                     output_path: str | Path | None = None) -> Path:
    from .engine.persona_extraction.base import ProfileArtifact
    from .engine.persona_extraction.lusifer_adapter import LusiferProfileExtractor
    from .engine.utility.providers import ProviderConfig, build_provider
    from .engine.utility.saving import save_profile_bundle

    root = Path(inputs_root)
    output = root / "profiles" if output_path is None else Path(output_path)
    prepared = (root / "prepared").resolve()
    if output.resolve() == prepared or prepared in output.resolve().parents:
        raise ValueError("profile output must be outside the original prepared dataset")
    if output.exists() or output.with_name(output.name + ".incomplete").exists():
        raise FileExistsError(output)
    records = build_profile_inputs(root, extraction_seed=extraction_seed,
                                   item_summaries_path=item_summaries_path)
    metadata = {"dataset": "ucsd-goodreads", "history_limit": 40,
        "input_split": "training-only", "reader_demographics": "not-supplied-or-inferred", "author_gender_annotations": "excluded", "shared_across_simulation_seeds": [42, 43, 44]}
    if item_summaries_path is not None:
        from .item_summaries import load_item_summaries
        overlay = load_item_summaries(item_summaries_path, prepared_path=root / "prepared")
        metadata["llm_item_text"] = _summary_metadata(item_summaries_path, overlay["manifest"], extraction_seed=extraction_seed)
    if provider is None:
        provider = build_provider(ProviderConfig(provider=provider_id, **{**PROVIDER_SETTINGS, **(provider_settings or {})}))
    extractor = LusiferProfileExtractor(provider=provider, max_interactions=40, batch_size=10)
    results = extractor.extract_many(records)
    profiles = tuple(result for result in results if isinstance(result, ProfileArtifact))
    failures = tuple(result for result in results if not isinstance(result, ProfileArtifact))
    save_profile_bundle(output, inputs=records, profiles=profiles, failures=failures, metadata=metadata)
    if failures:
        raise RuntimeError(f"profile extraction incomplete for {len(failures)} readers; inspect failures.jsonl")
    load_book_inputs(root / "prepared", root / "population.json", output)
    if item_summaries_path is not None:
        validate_profile_item_summaries(root, output, item_summaries_path)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--item-summaries", type=Path, help="Optional verified LLM-only source-description overlay.")
    parser.add_argument("--output", type=Path, help="New profile bundle; defaults to INPUTS/profiles.")
    parser.add_argument("--extraction-seed", type=int, default=1054)
    parser.add_argument("--provider", choices=("openai", "ollama"), default="openai")
    parser.add_argument("--model", default=PROVIDER_SETTINGS["model"])
    parser.add_argument("--reasoning-effort", default=PROVIDER_SETTINGS["reasoning_effort"])
    parser.add_argument("--max-output-tokens", type=int, default=PROVIDER_SETTINGS["max_output_tokens"])
    parser.add_argument("--run", action="store_true", help="Explicitly authorize profile-provider requests.")
    args = parser.parse_args(argv)
    required = [args.inputs / "prepared/_COMPLETE.json", args.inputs / "population.json"]
    missing = [str(path) for path in required if not path.is_file()]
    print(json.dumps({"mode": "run" if args.run else "describe", "provider": args.provider, "model": args.model,
        "latest_training_events": 40, "batch_size": 10, "extraction_seed": args.extraction_seed,
        "output": str(args.output or args.inputs / "profiles"),
        "item_summaries": None if args.item_summaries is None else str(args.item_summaries), "missing_inputs": missing}, indent=2))
    if missing:
        return 2
    if not args.run:
        build_profile_inputs(args.inputs, extraction_seed=args.extraction_seed,
                             item_summaries_path=args.item_summaries)
        return 0
    prepare_profiles(args.inputs, extraction_seed=args.extraction_seed, provider_id=args.provider,
        item_summaries_path=args.item_summaries, output_path=args.output,
        provider_settings={"model": args.model, "reasoning_effort": args.reasoning_effort, "max_output_tokens": args.max_output_tokens})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
