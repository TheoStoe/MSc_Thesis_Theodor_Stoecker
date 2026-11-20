"""Prepare a shared MovieLens SASRec initialisation or import frozen embeddings.

Without --run, validate the supplied local inputs and print the request only.
The embeddings stage imports a validated bundle, or generates one only with
both --generate and --run. No command downloads movie descriptions. The text
builder and fingerprint schema reproduce the original local NCM embedding
contract; the optional provider implementation uses the installed OpenAI SDK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Sequence


ROOT = Path(__file__).resolve().parent


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _embedding_metadata(dataset, metadata_path: Path | None, *, title_genres_only: bool, source: str | None) -> tuple[dict, list[dict]]:
    """Validate source IDs and reproduce movielens-title-genres-description-v1."""
    if metadata_path is None and not title_genres_only:
        raise ValueError("generation needs --metadata, or explicit --title-genres-only for a changed scientific setting")
    if metadata_path is not None and title_genres_only:
        raise ValueError("choose --metadata or --title-genres-only, not both")
    supplied = json.loads(metadata_path.read_text(encoding="utf-8"))["items"] if metadata_path else None
    if supplied is not None and {int(item) for item in supplied} != set(dataset.catalog):
        raise ValueError("metadata must cover exactly the prepared runtime item IDs")
    items, rows = {}, []
    for item, original in sorted(dataset.catalog.items()):
        metadata = supplied[str(item)] if supplied is not None else {
            "item_id": item, "original_item_id": dataset.runtime_to_source_item_id[item],
            "title": original["title"], "genres": list(original["genres"]), "description": None,
        }
        if (metadata["item_id"] != item
            or str(metadata["original_item_id"]) != dataset.runtime_to_source_item_id[item]
            or metadata["title"] != original["title"]
            or tuple(metadata["genres"]) != tuple(original["genres"])):
            raise ValueError(f"metadata source mapping/title/genres differ for item {item}")
        description = metadata["description"]
        if description is not None and (not isinstance(description, str) or not description.strip()):
            raise ValueError(f"description must be a non-empty string or explicit null for item {item}")
        text = (f"Title: {metadata['title'].strip()}\n"
                f"Genres: {' | '.join(genre.strip() for genre in metadata['genres'])}\n"
                f"Description: {description.strip() if description is not None else '[not provided]'}")
        if len(text.encode("utf-8")) > 8191:
            raise ValueError(f"item {item} exceeds the conservative 8191-byte per-item request bound")
        items[str(item)] = metadata
        rows.append({"item_id": item, "text": text})
    return {
        "schema": "portable-movielens-embedding-metadata-v1", "items": items,
        "source": source or (str(metadata_path.resolve()) if metadata_path else "MovieLens titles and genres only"),
        "scientific_setting": "user_supplied_descriptions" if supplied is not None else "new_title_genres_only_setting",
    }, rows


def generate_embeddings(dataset, *, output: Path, metadata: dict, rows: list[dict], batch_size: int, client=None) -> Path:
    """Freeze native 1536-dimensional vectors, caching each successful response.

    A failed/interrupted request without a saved response may still be billed.
    Retrying this command reuses validated completed batches in .incomplete.
    The injectable SDK boundary permits an entirely offline format check.
    """
    import numpy as np
    from feedback_loop_experiment.user_models.ncm_artifacts import (
        NCM_EMBEDDING_SCHEMA_VERSION, _array_fingerprint, _artifact_fingerprint,
    )
    from mitigation_experiment.prepare_ncm_artifacts import validate_embedding_catalog

    if output.exists():
        raise FileExistsError(output)
    if not 1 <= batch_size <= 2048:
        raise ValueError("embedding batch size must be between 1 and 2048")
    if any(sum(len(row["text"].encode("utf-8")) for row in rows[start:start + batch_size]) > 299000
           for start in range(0, len(rows), batch_size)):
        raise ValueError("embedding batch exceeds the conservative 299000-byte request bound; reduce --embedding-batch-size")
    builder = "movielens-title-genres-description-v1"
    request = {
        "schema": "portable-movielens-embedding-generation-v1", "provider": "openai",
        "model": "text-embedding-3-small", "dimension": 1536, "normalization": "none",
        "request_parameters": {"encoding_format": "float"}, "batch_size": batch_size,
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "item_texts_fingerprint": _artifact_fingerprint({"builder_version": builder, "items": rows}),
        "metadata_fingerprint": _artifact_fingerprint(metadata),
    }
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        if json.loads((staging / "request.json").read_text(encoding="utf-8")) != request:
            raise ValueError("incomplete generation belongs to different texts/settings")
    else:
        staging.mkdir(parents=True, exist_ok=False)
        _write_json(staging / "request.json", request)
    vectors, receipts = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        ids = [row["item_id"] for row in batch]
        cache = staging / f"batch_{start:06d}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                matrix = saved["vectors"]
                receipt = json.loads(str(saved["receipt"].item()))
                if saved["item_ids"].tolist() != ids or receipt["vectors_fingerprint"] != _array_fingerprint(matrix):
                    raise ValueError("cached embedding batch IDs/vectors differ")
        else:
            if client is None:
                from openai import OpenAI
                client = OpenAI()  # Environment credentials stay in memory.
            response = client.embeddings.create(model=request["model"], input=[row["text"] for row in batch], encoding_format="float")
            ordered = sorted(response.data, key=lambda row: row.index)
            if [row.index for row in ordered] != list(range(len(batch))):
                raise ValueError("provider response indices do not cover the embedding batch")
            matrix = np.asarray([row.embedding for row in ordered], dtype=np.float32)
            usage = getattr(response, "usage", None)
            receipt = {
                "vectors_fingerprint": _array_fingerprint(matrix),
                "model": getattr(response, "model", request["model"]),
                "request_id": getattr(response, "_request_id", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            if matrix.shape != (len(batch), 1536) or not np.isfinite(matrix).all():
                raise ValueError("provider must return finite native 1536-dimensional vectors")
            temporary = cache.with_name(cache.name + ".incomplete")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, item_ids=np.asarray(ids, dtype=np.int64), vectors=matrix,
                                    receipt=np.asarray(json.dumps(receipt, sort_keys=True)))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.rename(cache)
        if matrix.shape != (len(batch), 1536) or not np.isfinite(matrix).all():
            raise ValueError("cached embedding dimensions/values are invalid")
        vectors.append(matrix)
        receipts.append(receipt)
    matrix = np.concatenate(vectors, axis=0)
    item_ids = tuple(row["item_id"] for row in rows)
    arrays = staging / "embeddings.npz"
    if arrays.exists():
        with np.load(arrays, allow_pickle=False) as saved:
            if tuple(saved["item_ids"].tolist()) != item_ids or not np.array_equal(saved["vectors"], matrix):
                raise ValueError("incomplete embedding arrays differ from cached batches")
    else:
        temporary = arrays.with_name(arrays.name + ".incomplete")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, item_ids=np.asarray(item_ids, dtype=np.int64), vectors=matrix)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(arrays)
    payload = {
        "schema_version": NCM_EMBEDDING_SCHEMA_VERSION, "provider": request["provider"],
        "model": request["model"], "request_parameters": request["request_parameters"],
        "text_builder_version": builder, "included_metadata_fields": ["title", "genres", "description"],
        "preprocessing_fingerprint": dataset.dataset_fingerprint,
        "catalog_fingerprint": _artifact_fingerprint({"item_ids": item_ids}),
        "item_texts_fingerprint": request["item_texts_fingerprint"], "item_ids": item_ids,
        "dimension": 1536, "dtype": "float32", "normalization": "none",
        "vector_content_fingerprint": _array_fingerprint(matrix),
    }
    documents = {
        "manifest.json": {**payload, "arrays_file_sha256": _hash(arrays), "content_fingerprint": _artifact_fingerprint(payload)},
        "item_metadata.json": metadata,
        "generation_receipt.json": {"request": request, "batches": receipts,
                                   "accounting_scope": "successfully cached responses; uncached failures/interruption may be absent"},
    }
    for filename, document in documents.items():
        destination = staging / filename
        if destination.exists():
            if json.loads(destination.read_text(encoding="utf-8")) != json.loads(json.dumps(document)):
                raise ValueError(f"incomplete {filename} differs from completed batches")
        else:
            _write_json(destination, document)
    validate_embedding_catalog(dataset, staging)
    staging.rename(output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("sasrec", "embeddings"))
    parser.add_argument("--prepared-data", required=True, type=Path)
    parser.add_argument("--population", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42, help="Simulation/model seed, not the population draw seed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--training-scope", choices=("active-reserve", "all"), default="active-reserve")
    parser.add_argument("--epochs", type=int, help="Default: recorded 804-epoch SASRec budget")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--dropout-rate", type=float)
    parser.add_argument("--embedding-source", type=Path, help="Existing manifest.json + embeddings.npz bundle; required for embeddings")
    parser.add_argument("--metadata", type=Path, help="Exact frozen item_metadata.json used to construct the embedding texts, including descriptions")
    parser.add_argument("--generate", action="store_true", help="Generate text-embedding-3-small vectors; requires --run and may incur API charges")
    parser.add_argument("--title-genres-only", action="store_true", help="Explicitly use missing descriptions: a new scientific setting, not the original frozen cache")
    parser.add_argument("--metadata-source", help="Source/provenance of user-supplied descriptions; saved with the generated bundle")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--run", action="store_true", help="Execute the local fit/import after validation")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    from feedback_loop_experiment.data_preparation import load_prepared_dataset
    from feedback_loop_experiment.utility.user_handling import load_population_artifact

    dataset = load_prepared_dataset(args.prepared_data)
    population = load_population_artifact(args.population)
    if population.source_population_fingerprint != dataset.dataset_fingerprint:
        raise ValueError("population and prepared dataset fingerprints differ")
    if set(member.user_id for member in population.members) != set(dataset.training_histories):
        raise ValueError("population membership does not cover the prepared users")
    request = {
        "stage": args.stage,
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "population_fingerprint": population.population_fingerprint,
        "population_seed": population.population_seed,
        "simulation_seed": args.seed,
    }

    if args.stage == "sasrec":
        if args.embedding_source is not None or args.metadata is not None or args.generate or args.title_genres_only:
            parser.error("--embedding-source and --metadata apply only to embeddings")
        settings = json.loads((ROOT / "feedback_loop_experiment/configs/thesis_settings.json").read_text(encoding="utf-8"))["sasrec"]
        for option, setting in (
            ("epochs", "num_epochs"), ("batch_size", "batch_size"),
            ("max_sequence_length", "max_sequence_length"), ("hidden_size", "hidden_size"),
            ("num_blocks", "num_blocks"), ("num_heads", "num_heads"),
            ("dropout_rate", "dropout_rate"),
        ):
            value = getattr(args, option)
            if value is not None:
                settings[setting] = value
        settings["device"] = args.device
        from feedback_loop_experiment.recommenders.sasrec import SASRecAdapter
        SASRecAdapter(catalog_item_ids=tuple(sorted(dataset.catalog)), random_seed=args.seed, **settings)
        training_users = tuple(sorted(
            member.user_id for member in population.members
            if args.training_scope == "all" or member.role.value in {"active", "reserve"}
        ))
        if not training_users:
            raise ValueError("selected training scope contains no users")
        request.update(training_scope=args.training_scope, training_users=list(training_users), settings=settings)
        receipt = args.output / f"sasrec_seed{args.seed}.json"
        if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")).get("request") != request:
            raise FileExistsError("this output already records different inputs/settings for this seed; use another --output")
        if not args.run:
            print(json.dumps({"status": "validated_not_run", "request": request}, indent=2))
            return 0

        from mitigation_experiment.initial_recommender import ensure_initial_recommender

        checkpoint = ensure_initial_recommender(
            dataset=dataset, training_user_ids=training_users, seed=args.seed,
            settings=settings, checkpoint_root=args.output,
        )
        payload = {
            "schema": "portable-movielens-shared-initial-v1", "request": request,
            "checkpoint_subdirectory": checkpoint.name,
            "checkpoint_sha256": _hash(checkpoint / "checkpoint.pt"),
        }
        if receipt.exists():
            if json.loads(receipt.read_text(encoding="utf-8")) != payload:
                raise ValueError("initial checkpoint differs from its preparation receipt")
        else:
            _write_json(receipt, payload)
        print(json.dumps({"status": "complete", "checkpoint_directory": str(checkpoint.resolve()), "receipt": str(receipt.resolve())}, indent=2))
        return 0

    if args.generate:
        if args.embedding_source is not None:
            parser.error("choose --generate or --embedding-source, not both")
        if not 1 <= args.embedding_batch_size <= 2048:
            parser.error("--embedding-batch-size must be between 1 and 2048")
        metadata, rows = _embedding_metadata(dataset, args.metadata, title_genres_only=args.title_genres_only, source=args.metadata_source)
        request.update(embedding_model="text-embedding-3-small", native_dimension=1536,
                       metadata_mode=metadata["scientific_setting"], item_count=len(rows),
                       embedding_batch_size=args.embedding_batch_size)
        if args.output.exists():
            raise FileExistsError("embedding generation needs a new output directory")
        if not args.run:
            print(json.dumps({"status": "validated_not_run", "request": request}, indent=2))
            return 0
        generated = generate_embeddings(dataset, output=args.output, metadata=metadata, rows=rows, batch_size=args.embedding_batch_size)
        print(json.dumps({"status": "complete", "embedding_directory": str(generated.resolve())}, indent=2))
        return 0
    if args.embedding_source is None:
        parser.error("embeddings requires --embedding-source, or explicit --generate with --metadata/--title-genres-only")
    if args.output.exists():
        raise FileExistsError("embedding import needs a new output directory")
    from mitigation_experiment.prepare_ncm_artifacts import validate_embedding_catalog

    source = args.embedding_source
    metadata = args.metadata or source / "item_metadata.json"
    verification = validate_embedding_catalog(dataset, source, metadata_path=metadata)
    request.update(catalogue_verification=verification)
    if not args.run:
        print(json.dumps({"status": "validated_not_run", "request": request}, indent=2))
        return 0

    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.json", "embeddings.npz"):
        shutil.copyfile(source / name, args.output / name)
    if metadata.is_file():
        shutil.copyfile(metadata, args.output / "item_metadata.json")
    imported = validate_embedding_catalog(dataset, args.output)
    if imported != verification:
        raise ValueError("embedding import differs from its validated source")
    files = {path.name: _hash(path) for path in args.output.iterdir() if path.is_file()}
    _write_json(args.output / "import_receipt.json", {
        "schema": "portable-movielens-embedding-import-v1", "request": request,
        "source": str(source.resolve()), "file_sha256": files,
    })
    print(json.dumps({"status": "complete", "embedding_directory": str(args.output.resolve()), "verification": verification}, indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
