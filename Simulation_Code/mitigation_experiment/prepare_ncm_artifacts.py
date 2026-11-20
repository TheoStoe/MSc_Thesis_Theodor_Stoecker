"""Validate and copy existing frozen NCMs into this independent experiment.

This is an explicit local import, never a training or provider operation.
Runtime code subsequently reads only the imported bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil

from .data_preparation import prepare_movielens_1m
from .user_models.ncm_artifacts import (
    _artifact_fingerprint, load_ncm_checkpoint_artifact, load_ncm_item_embeddings,
)
from .utility.saving import save_json_artifact

ROOT = Path(__file__).resolve().parent
PROTOCOL = "legacy_full_observed_exclusion"
MODELS = {"P3": "ncm_inspired_gru_uniform", "P4": "ncm_inspired_gru_mansoury"}


def validate_embedding_catalog(dataset, directory: Path, *, metadata_path: Path | None = None) -> dict:
    """Check exact ordered movie text, not merely the numeric item-ID set."""
    embeddings, fingerprint = load_ncm_item_embeddings(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if set(embeddings) != set(dataset.catalog):
        raise ValueError("NCM embedding IDs differ from the current catalog")
    builder = "movielens-title-genres-description-v1"
    metadata_path = metadata_path or directory / "item_metadata.json"
    frozen_metadata = None
    if metadata_path.is_file():
        frozen_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["items"]
        if {int(item) for item in frozen_metadata} != set(dataset.catalog):
            raise ValueError("frozen embedding metadata does not cover this catalog")
    rows = []
    for item, metadata in sorted(dataset.catalog.items()):
        if frozen_metadata is not None:
            original = frozen_metadata[str(item)]
            if (int(original["item_id"]) != item
                or str(original["original_item_id"]) != dataset.runtime_to_source_item_id[item]
                or original["title"] != metadata["title"]
                or tuple(original["genres"]) != tuple(metadata["genres"])):
                raise ValueError(f"frozen embedding source movie mapping differs for item {item}")
            metadata = original
        # Reproduce the frozen embedding text-builder contract locally.
        description = metadata.get("description")
        text = (f"Title: {metadata['title'].strip()}\n"
                f"Genres: {' | '.join(g.strip() for g in metadata['genres'])}\n"
                f"Description: {description.strip() if description is not None else '[not provided]'}")
        rows.append({"item_id": item, "text": text})
    text_fingerprint = _artifact_fingerprint({"builder_version": builder, "items": rows})
    if manifest.get("text_builder_version") != builder or manifest.get("item_texts_fingerprint") != text_fingerprint:
        raise ValueError("NCM embedding movie/text mapping differs; supply the exact frozen embedding metadata, including its original descriptions")
    return {"embedding_fingerprint": fingerprint, "item_texts_fingerprint": text_fingerprint,
            "current_id_mapping_fingerprint": dataset.id_mapping_fingerprint,
            "current_dataset_fingerprint": dataset.dataset_fingerprint}


def ncm_settings(dataset, *, model_folder: str, checkpoint_root: Path,
                 embedding_directory: Path, seed: int, device: str = "cpu") -> dict:
    """Validate a complete local import and return strict adapter settings."""
    verification = validate_embedding_catalog(dataset, embedding_directory)
    root = checkpoint_root / f"seed_{seed}"
    receipt = json.loads((root / "import_manifest.json").read_text(encoding="utf-8"))
    if receipt.get("protocol") != PROTOCOL:
        raise ValueError("NCM protocol is not the declared legacy candidate protocol")
    if receipt.get("catalog_verification") != verification:
        raise ValueError("local NCM import is incompatible with prepared data/embeddings")
    directory = root / model_folder / "checkpoint"
    expected = receipt["models"][model_folder]
    manifest, _, fingerprint = load_ncm_checkpoint_artifact(
        directory, expected_embedding_fingerprint=verification["embedding_fingerprint"],
        expected_prepared_fingerprint=expected["prepared_fingerprint"],
        expected_parent_fingerprint=expected["parent_fingerprint"],
        expected_placement_fingerprint=expected["placement_fingerprint"],
    )
    if fingerprint != expected["checkpoint_fingerprint"]:
        raise ValueError("local NCM checkpoint differs from its import receipt")
    expected_condition = "T2_slatewise_gru_uniform" if model_folder == "P3" else "T3_slatewise_gru_mansoury"
    if manifest["condition"] != expected_condition:
        raise ValueError("P3/P4 checkpoint placement mismatch")
    return {"checkpoint_directory": str(directory.resolve()),
            "embedding_directory": str(embedding_directory.resolve()),
            "device": device,
            **{f"expected_{name}": expected[name] for name in
               ("prepared_fingerprint", "parent_fingerprint", "placement_fingerprint")}}


def import_checkpoints(*, source_root: Path, destination_root: Path,
                       dataset, embedding_directory: Path, seed: int,
                       embedding_metadata: Path | None = None) -> Path:
    verification = validate_embedding_catalog(dataset, embedding_directory, metadata_path=embedding_metadata)
    destination = destination_root / f"seed_{seed}"
    # Validate both sources BEFORE creating the destination. Failures preserve
    # any partial destination, and never open sources for writing.
    records = {}
    for folder in MODELS:
        source = source_root / folder / f"seed_{seed}" / "checkpoint"
        manifest, _, _ = load_ncm_checkpoint_artifact(
            source, expected_embedding_fingerprint=verification["embedding_fingerprint"])
        records[folder] = {**manifest, "source": str(source.resolve())}
    if embedding_metadata is not None:
        metadata_target = embedding_directory / "item_metadata.json"
        if metadata_target.exists():
            if metadata_target.read_bytes() != embedding_metadata.read_bytes():
                raise FileExistsError("different frozen embedding metadata already exists locally")
        else:
            with metadata_target.open("xb") as stream:
                stream.write(embedding_metadata.read_bytes())
    destination.mkdir(parents=True, exist_ok=False)
    for folder, record in records.items():
        source = Path(record["source"])
        shutil.copytree(source, destination / folder / "checkpoint")
    save_json_artifact(destination / "import_manifest.json", {
        "schema": "mitigation-local-ncm-import-v1", "protocol": PROTOCOL,
        "simulation_seed": seed, "catalog_verification": verification,
        "models": records,
        "interpretation": "Frozen full-population train-target NCM; legacy future-positive avoidance in synthetic candidates. No population-specific NCM retraining.",
    })
    for folder in MODELS:
        ncm_settings(dataset, model_folder=folder, checkpoint_root=destination_root,
                     embedding_directory=embedding_directory, seed=seed)
    return destination


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, default=ROOT / "data/ncm_checkpoints")
    parser.add_argument("--raw-data", type=Path, default=ROOT / "data/ml-1m")
    parser.add_argument("--embedding-directory", type=Path, default=ROOT / "data/text_embedding_3_small_movielens_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-metadata", type=Path, help="Exact item_metadata.json used by the frozen embedding cache")
    args = parser.parse_args(argv)
    from .utility.run_logging import log_run_attempt
    logging.basicConfig(level=logging.INFO)
    with log_run_attempt(ROOT / "preparation_logs", "import_ncm"):
        dataset = prepare_movielens_1m(args.raw_data)
        print(import_checkpoints(source_root=args.source_root, destination_root=args.destination_root,
                                 dataset=dataset, embedding_directory=args.embedding_directory, seed=args.seed,
                                 embedding_metadata=args.embedding_metadata))
    return 0


def main(argv=None) -> int:
    try:
        return _main(argv)
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            return 0
        from .utility.run_logging import log_run_attempt
        with log_run_attempt(ROOT / "preparation_logs", "ncm_import_failure"):
            raise


if __name__ == "__main__":
    raise SystemExit(main())
