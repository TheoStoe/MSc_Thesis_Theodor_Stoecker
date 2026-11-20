"""Describe or explicitly prepare fresh Goodreads models and embeddings.

No model fitting or provider request occurs without --run. Inputs are all local;
this module never downloads a dataset or pretrained model weights.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .runtime_adapter import (ITEM_TEXT_VERSION, NCM_FOLDERS, SASREC_SETTINGS, file_sha256,
                              load_book_inputs, prepare_book_inputs)


def prepare_sasrec(inputs_root: str | Path, *, seed: int = 42, settings: dict[str, Any] | None = None, slate_size: int = 10) -> Path:
    """Fit once on the frozen cohort's training histories and save t0 slates.

This explicit Python API performs a fit; the CLI guards it with --run. t0 is a
separate baseline artifact, never an invented iteration-zero observation.
"""
    from .engine.contracts import SimulationCheckpoint
    from .engine.recommenders.base import FitSpec
    from .engine.recommenders.sasrec import SASRecAdapter
    from .engine.utility.checkpoints import save_checkpoint
    from .engine.utility.filesystem import rename_directory_with_retry
    from .engine.utility.fingerprints import fingerprint_semantic_state as fingerprint
    from .engine.utility.saving import save_json_artifact

    root = Path(inputs_root)
    inputs = load_book_inputs(root / "prepared", root / "population.json")
    output = root / "initial_sasrec" / f"seed_{seed}"
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(f"initial model destination already exists: {output}")
    dataset = inputs.dataset
    histories = dataset.training_histories
    resolved_settings = {**SASREC_SETTINGS, **(settings or {})}
    if {"catalog_item_ids", "random_seed"}.intersection(resolved_settings):
        raise ValueError("catalog and random seed are controlled by prepared inputs")
    catalog = tuple(sorted(dataset.catalog))
    eligible = {uid: tuple(iid for iid in catalog if iid not in set(history)) for uid, history in histories.items()}
    if any(len(items) < slate_size for items in eligible.values()):
        raise ValueError("not enough unseen books for the initial baseline slates")
    model = SASRecAdapter(catalog_item_ids=catalog, random_seed=seed, **resolved_settings)
    training_identity = fingerprint({uid: tuple(history) for uid, history in sorted(histories.items())}, "recommender-training-histories-v1")
    report = model.fit(histories, FitSpec(0, training_identity, {"fit_role": "initial"}))
    slates = model.recommend(tuple(sorted(histories)), slate_size, histories, eligible)
    checkpoint = SimulationCheckpoint(schema_version="goodreads-initial-sasrec-v1", run_id=f"goodreads_initial_sasrec__seed_{seed}", condition_id="shared_initial_sasrec",
        seed=seed, completed_iteration=0, input_fingerprints={"dataset": dataset.dataset_fingerprint, "population": inputs.population.population_fingerprint},
        component_states={"recommender": model.export_state()}, policy_states={}, randomness_state={"simulation_seed": seed})
    save_checkpoint(staging, checkpoint)
    baseline = {"schema_version": "books-initial-baseline-v1", "dataset_fingerprint": dataset.dataset_fingerprint,
                "population_fingerprint": inputs.population.population_fingerprint, "simulation_seed": seed,
                "slate_size": slate_size, "exclusion": "initial_training_history_only", "timepoint": "before_first_simulation_opportunity",
                "recommender_fingerprint": model.fingerprint(),
                "slates": [{"user_id": uid, "item_ids": list(slates.items_by_user[uid])} for uid in sorted(histories)]}
    save_json_artifact(staging / "baseline_slates.json", baseline)
    save_json_artifact(staging / "fit_diagnostics.json", dict(report.diagnostics))
    save_json_artifact(staging / "_COMPLETE.json", {"schema_version": "goodreads-initial-sasrec-files-v1", "status": "complete",
        "dataset_fingerprint": dataset.dataset_fingerprint, "seed": seed,
        "files_sha256": {path.name: file_sha256(path) for path in staging.iterdir() if path.is_file()}})
    rename_directory_with_retry(staging, output)
    return output


def prepare_embeddings(inputs_root: str | Path, *, model: str = "text-embedding-3-small", batch_size: int = 128, client: Any = None) -> Path:
    """Create book embeddings, reusing only this request's completed batch cache.

    A returned batch is atomically cached with its provider usage before the
    next request. An interrupted request without a saved response cannot be
    recovered, so billing records cover successfully persisted responses only.
    """
    import numpy as np
    from .engine.user_models.ncm_artifacts import (NCM_EMBEDDING_SCHEMA_VERSION, _array_fingerprint,
        _artifact_fingerprint, load_ncm_item_embeddings)
    from .engine.utility.filesystem import rename_directory_with_retry
    from .engine.utility.saving import save_json_artifact

    if not 1 <= batch_size <= 2048:
        raise ValueError("embedding batch size must be between 1 and 2048")
    root = Path(inputs_root)
    inputs = load_book_inputs(root / "prepared", root / "population.json")
    dataset = inputs.dataset
    output = root / "embeddings"
    staging = output.with_name(output.name + ".incomplete")
    if output.exists():
        raise FileExistsError(output)
    item_ids = tuple(sorted(dataset.catalog))
    texts = [dataset.catalog[iid]["item_text"] for iid in item_ids]
    if not item_ids:
        raise ValueError("cannot embed an empty book catalog")
    # UTF-8 byte counts upper-bound byte-level tokenizer input lengths. This
    # deliberately conservative check needs no additional tokenizer dependency.
    byte_lengths = [len(text.encode("utf-8")) for text in texts]
    if any(not text.strip() or length > 8191 for text, length in zip(texts, byte_lengths)):
        raise ValueError("embedding text must be non-empty and at most 8191 UTF-8 bytes")
    if any(sum(byte_lengths[start:start + batch_size]) > 299000 for start in range(0, len(item_ids), batch_size)):
        raise ValueError("embedding request exceeds the conservative 299000-byte token bound; reduce batch size")
    identity = {"schema_version": "goodreads-embedding-batch-cache-v1", "provider": "openai", "model": model,
        "request_parameters": {"encoding_format": "float"}, "batch_size": batch_size,
        "text_builder_version": ITEM_TEXT_VERSION, "preprocessing_fingerprint": dataset.dataset_fingerprint,
        "item_texts_fingerprint": dataset.item_text_fingerprint,
        "request_content_fingerprint": _artifact_fingerprint({"item_ids": item_ids, "texts": texts})}
    identity_path = staging / "request_identity.json"
    if staging.exists():
        if not identity_path.is_file() or json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("incomplete embedding cache does not match the requested dataset, texts, model or batches")
    else:
        staging.mkdir(parents=True)
        save_json_artifact(identity_path, identity)
    cache = staging / "batches"
    cache.mkdir(exist_ok=True)
    expected_caches = {f"batch_{start // batch_size:05d}.npz" for start in range(0, len(item_ids), batch_size)}
    if any(path.name not in expected_caches and not path.name.endswith(".incomplete") for path in cache.iterdir()):
        raise ValueError("embedding cache contains an unexpected batch file")
    vectors, receipts = [], []
    for start in range(0, len(item_ids), batch_size):
        batch = item_ids[start:start + batch_size]
        batch_texts = texts[start:start + batch_size]
        batch_fingerprint = _artifact_fingerprint({"identity": identity, "item_ids": batch, "texts": batch_texts})
        destination = cache / f"batch_{start // batch_size:05d}.npz"
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                if set(saved.files) != {"item_ids", "vectors", "receipt"}:
                    raise ValueError("invalid embedding batch cache fields")
                receipt = json.loads(str(saved["receipt"].item()))
                batch_vectors = np.array(saved["vectors"], dtype=np.float32, copy=True)
                if tuple(saved["item_ids"].tolist()) != batch or receipt.get("request_fingerprint") != batch_fingerprint:
                    raise ValueError("embedding batch cache identity mismatch")
                if receipt.get("vector_content_fingerprint") != _array_fingerprint(batch_vectors):
                    raise ValueError("embedding batch cache vector hash mismatch")
        else:
            if client is None:
                from openai import OpenAI
                # Credentials are consumed by the SDK and never persisted.
                client = OpenAI()
            response = client.embeddings.create(model=model, input=batch_texts, encoding_format="float")
            rows = sorted(response.data, key=lambda row: row.index)
            if [row.index for row in rows] != list(range(len(batch))):
                raise ValueError("embedding response rows do not exactly cover the requested batch")
            batch_vectors = np.asarray([row.embedding for row in rows], dtype=np.float32)
            if batch_vectors.ndim != 2 or batch_vectors.shape[0] != len(batch) or not batch_vectors.shape[1] or not np.isfinite(batch_vectors).all():
                raise ValueError("provider returned invalid embedding vectors")
            usage = getattr(response, "usage", None)
            def token_count(field):
                value = usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                    raise ValueError("provider returned an invalid embedding token count")
                return value
            receipt = {"request_fingerprint": batch_fingerprint, "vector_content_fingerprint": _array_fingerprint(batch_vectors),
                "model": getattr(response, "model", model), "request_id": getattr(response, "_request_id", None),
                "prompt_tokens": token_count("prompt_tokens"), "total_tokens": token_count("total_tokens")}
            temporary = destination.with_name(destination.name + ".incomplete")
            # An interrupted local write can be replaced; a complete batch is
            # never overwritten. The one file contains both vectors and usage.
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, item_ids=np.asarray(batch, dtype=np.int64), vectors=batch_vectors,
                                    receipt=np.asarray(json.dumps(receipt, sort_keys=True)))
                stream.flush()
                os.fsync(stream.fileno())
            os.rename(temporary, destination)
        if batch_vectors.ndim != 2 or batch_vectors.shape[0] != len(batch) or not batch_vectors.shape[1] or not np.isfinite(batch_vectors).all():
            raise ValueError("invalid cached embedding dimensions or values")
        vectors.append(batch_vectors)
        receipts.append(receipt)
    matrix = np.concatenate(vectors, axis=0)
    if matrix.shape[0] != len(item_ids) or not np.isfinite(matrix).all():
        raise ValueError("provider returned invalid embedding vectors")
    arrays_path = staging / "embeddings.npz"
    if arrays_path.exists():
        with np.load(arrays_path, allow_pickle=False) as saved:
            if set(saved.files) != {"item_ids", "vectors"} or tuple(saved["item_ids"].tolist()) != item_ids or not np.array_equal(saved["vectors"], matrix):
                raise ValueError("incomplete final embedding arrays disagree with cached batches")
    else:
        temporary = arrays_path.with_name(arrays_path.name + ".incomplete")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, item_ids=np.asarray(item_ids, dtype=np.int64), vectors=matrix)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, arrays_path)
    payload = {"schema_version": NCM_EMBEDDING_SCHEMA_VERSION, "provider": "openai", "model": model,
        "request_parameters": {"encoding_format": "float"}, "text_builder_version": ITEM_TEXT_VERSION,
        "included_metadata_fields": ["title", "authors", "genres"], "preprocessing_fingerprint": dataset.dataset_fingerprint,
        "catalog_fingerprint": _artifact_fingerprint({"item_ids": item_ids}), "item_texts_fingerprint": dataset.item_text_fingerprint,
        "item_ids": item_ids, "dimension": matrix.shape[1], "dtype": matrix.dtype.name, "normalization": "none",
        "vector_content_fingerprint": _array_fingerprint(matrix)}
    manifest = {**payload, "arrays_file_sha256": file_sha256(arrays_path), "content_fingerprint": _artifact_fingerprint(payload)}
    prompt_tokens = sum(row["prompt_tokens"] for row in receipts if row["prompt_tokens"] is not None)
    usage_complete = all(row["prompt_tokens"] is not None for row in receipts)
    rate = 0.02 if model == "text-embedding-3-small" else None
    usage_report = {"schema_version": "goodreads-embedding-usage-v1", "provider": "openai", "model": model,
        "completed_requests": len(receipts), "embedded_items": len(item_ids), "prompt_tokens": prompt_tokens,
        "provider_usage_complete": usage_complete, "pricing": {"usd_per_million_input_tokens": rate,
            "pricing_checked_date": "2026-10-25", "source": "https://platform.openai.com/docs/pricing", "mode": "standard_api"},
        "estimated_cost_usd_from_recorded_usage": prompt_tokens * rate / 1_000_000 if rate is not None else None,
        "accounting_scope": "successfully cached provider responses; failed or interrupted uncached requests may be absent",
        "batches": receipts}
    for filename, document in (("manifest.json", manifest), ("usage.json", usage_report)):
        path = staging / filename
        if path.exists():
            # Canonical JSON comparison also normalizes tuple/list item IDs.
            if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(document)):
                raise ValueError(f"incomplete embedding {filename} differs from cached batches")
        else:
            save_json_artifact(path, document)
    load_ncm_item_embeddings(staging)
    rename_directory_with_retry(staging, output)
    return output


def prepare_ncm(inputs_root: str | Path, *, seed: int = 42, variants: Sequence[str] = ("P3", "P4"), epochs: int = 5000, batch_size: int = 128, patience: int = 7, device: str = "cpu", slate_seed: int = 42) -> list[Path]:
    """Fit all requested frozen response models using training-only book slates."""
    from .engine.user_models.ncm import NCMClickModelConfig, NCM_ARTIFACT_CONDITION_BY_MODEL
    from .engine.user_models.ncm_artifacts import load_ncm_item_embeddings, save_ncm_checkpoint_artifact, load_ncm_checkpoint_artifact
    from .engine.user_models.ncm_slates import ensure_ncm_training_slates
    from .engine.user_models.ncm_training import NCMTrainingConfig, load_ncm_training_examples, train_ncm_model
    from .engine.utility.saving import save_json_artifact

    root = Path(inputs_root)
    inputs = load_book_inputs(root / "prepared", root / "population.json")
    dataset = inputs.dataset
    manifest = json.loads((root / "embeddings/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("preprocessing_fingerprint") != dataset.dataset_fingerprint or manifest.get("item_texts_fingerprint") != dataset.item_text_fingerprint:
        raise ValueError("embeddings are not from these prepared book inputs")
    embeddings, embedding_fingerprint = load_ncm_item_embeddings(root / "embeddings")
    if set(embeddings) != set(dataset.catalog):
        raise ValueError("embeddings and book catalog differ")
    if not variants or any(variant not in {"P1", "P3", "P4"} for variant in variants):
        raise ValueError("NCM variants must be P1, P3 or P4")
    for variant in variants:
        if (root / "ncm" / f"seed_{seed}" / variant).exists():
            raise FileExistsError(f"NCM destination already exists: {variant}, seed {seed}")
    placements = ensure_ncm_training_slates(dataset, root / "ncm_training_slates" / f"slate_seed_{slate_seed}", slate_seed=slate_seed)
    training_config = NCMTrainingConfig(seed=seed, epochs=epochs, batch_size=batch_size, patience=patience, device=device)
    names = {folder: name for name, folder in NCM_FOLDERS.items()}
    outputs = []
    for variant in dict.fromkeys(variants):
        policy = "mansoury" if variant == "P4" else "uniform"
        training, validation, metadata = load_ncm_training_examples(placements / policy, embeddings, prepared_fingerprint=dataset.dataset_fingerprint)
        model_config = NCMClickModelConfig(architecture="pointwise_logistic" if variant == "P1" else "slatewise_gru", embedding_dimension=len(next(iter(embeddings.values()))), slate_size=10, hidden_size=64)
        result = train_ncm_model(model_config, training_config, training, validation)
        output = root / "ncm" / f"seed_{seed}" / variant
        checkpoint = output / "checkpoint"
        save_ncm_checkpoint_artifact(checkpoint, model=result.model, condition=NCM_ARTIFACT_CONDITION_BY_MODEL[names[variant]],
            model_config=asdict(model_config), training_config=training_config.to_dict(), prepared_fingerprint=metadata.prepared_fingerprint,
            parent_fingerprint=metadata.parent_fingerprint, placement_fingerprint=metadata.placement_fingerprint,
            embedding_fingerprint=embedding_fingerprint, best_epoch=result.best_epoch)
        save_json_artifact(output / "training_curve.json", [asdict(record) for record in result.records])
        load_ncm_checkpoint_artifact(checkpoint, expected_embedding_fingerprint=embedding_fingerprint, expected_prepared_fingerprint=dataset.dataset_fingerprint)
        outputs.append(checkpoint)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--stage", choices=("inputs", "sasrec", "embeddings", "ncm", "all"), default="all")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--population-seed", type=int, default=1054)
    parser.add_argument("--expected-users", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sasrec-epochs", type=int, default=804)
    parser.add_argument("--ncm-epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-sequence-length", type=int, default=200)
    parser.add_argument("--ncm-patience", type=int, default=7)
    parser.add_argument("--variant", choices=("P1", "P3", "P4"), action="append")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--run", action="store_true", help="Explicitly authorize the selected fits and/or embedding API requests.")
    arguments = parser.parse_args(argv)
    seeds = tuple(dict.fromkeys(arguments.seed or (42,)))
    if any(seed < 0 for seed in seeds):
        parser.error("seeds must be non-negative")
    stages = ("inputs", "sasrec", "embeddings", "ncm") if arguments.stage == "all" else (arguments.stage,)
    required = []
    if "inputs" in stages and not (arguments.inputs / "prepared").exists():
        required += [arguments.cohort or Path("MISSING_COHORT_ARGUMENT")]
    if "inputs" in stages and (arguments.inputs / "prepared").exists():
        required += [arguments.inputs / "prepared/_COMPLETE.json", arguments.inputs / "population.json"]
    if "inputs" not in stages:
        required += [arguments.inputs / "prepared/_COMPLETE.json", arguments.inputs / "population.json"]
    if "ncm" in stages and "embeddings" not in stages:
        required += [arguments.inputs / "embeddings/manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    description = {"mode": "run" if arguments.run else "describe", "stages": stages, "inputs": str(arguments.inputs), "simulation_seeds": seeds,
                   "population_seed": arguments.population_seed, "sasrec": {"num_epochs": arguments.sasrec_epochs, "batch_size": arguments.batch_size, "max_sequence_length": arguments.max_sequence_length},
                   "ncm_variants": arguments.variant or ["P3", "P4"], "api_stage": "embeddings" in stages, "missing_inputs": missing}
    print(json.dumps(description, indent=2))
    if missing:
        return 2
    if not arguments.run:
        return 0
    outputs = []
    if "sasrec" in stages:
        outputs.extend(arguments.inputs / "initial_sasrec" / f"seed_{seed}" for seed in seeds)
    if "embeddings" in stages:
        outputs.append(arguments.inputs / "embeddings")
    if "ncm" in stages:
        outputs.extend(arguments.inputs / "ncm" / f"seed_{seed}" / variant for seed in seeds for variant in (arguments.variant or ("P3", "P4")))
    for output in outputs:
        is_embedding_stage = output == arguments.inputs / "embeddings"
        if output.exists() or (output.with_name(output.name + ".incomplete").exists() and not is_embedding_stage):
            raise FileExistsError(f"Existing output will not be overwritten; select the next preparation stage or a fresh input root: {output}")
    if "inputs" in stages:
        if not (arguments.inputs / "prepared").exists():
            prepare_book_inputs(arguments.cohort, arguments.inputs, population_seed=arguments.population_seed, expected_users=arguments.expected_users)
        else:
            load_book_inputs(arguments.inputs / "prepared", arguments.inputs / "population.json")
    if "sasrec" in stages:
        for seed in seeds:
            prepare_sasrec(arguments.inputs, seed=seed, settings={"num_epochs": arguments.sasrec_epochs, "batch_size": arguments.batch_size, "max_sequence_length": arguments.max_sequence_length, "device": arguments.device})
    if "embeddings" in stages:
        prepare_embeddings(arguments.inputs, model=arguments.embedding_model, batch_size=arguments.batch_size)
    if "ncm" in stages:
        for seed in seeds:
            prepare_ncm(arguments.inputs, seed=seed, variants=arguments.variant or ("P3", "P4"), epochs=arguments.ncm_epochs, batch_size=arguments.batch_size, patience=arguments.ncm_patience, device=arguments.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
