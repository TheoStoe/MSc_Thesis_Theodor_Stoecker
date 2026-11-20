"""Generate MovieLens profiles separately from a simulation, with explicit consent.

Without ``--run``, validate the selected cohort and describe the pending work.
The original profile extraction algorithm and provider-neutral adapter are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-interactions", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    from .utility.user_handling import load_prepared_inputs
    inputs = load_prepared_inputs(args.prepared_data, args.population, None, None)
    if not args.run:
        print(json.dumps({"status": "described_not_started", "model": args.model,
            "users": len(inputs.population.members), "output": str(args.output),
            "max_interactions": args.max_interactions, "batch_size": args.batch_size}, indent=2))
        return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    from .automatic_inputs import _profile_inputs
    from .persona_extraction import LusiferProfileExtractor, ProfileArtifact, ProfileExtractionFailure
    from .utility.providers import ProviderConfig, build_provider
    from .utility.saving import save_profile_bundle
    records = _profile_inputs(
        raw_data_directory=args.raw_data, dataset=inputs.dataset,
        population=inputs.population, extraction_seed=inputs.population.population_seed,
    )
    provider = build_provider(ProviderConfig(provider="openai", model=args.model,
        reasoning_effort="none", max_output_tokens=2056))
    extractor = LusiferProfileExtractor(provider=provider,
        max_interactions=args.max_interactions, batch_size=args.batch_size)
    results = tuple(extractor.extract_many(records))
    profiles = tuple(item for item in results if isinstance(item, ProfileArtifact))
    failures = tuple(item for item in results if isinstance(item, ProfileExtractionFailure))
    save_profile_bundle(args.output, inputs=records, profiles=profiles, failures=failures,
        metadata={"purpose": "automatic_feedback_loop_inputs",
            "population_fingerprint": inputs.population.population_fingerprint})
    if failures:
        raise RuntimeError(f"{len(failures)} profiles failed; inspect {args.output}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
