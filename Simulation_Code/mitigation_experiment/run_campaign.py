"""Run the fixed-population mitigation matrix, or inspect it with --dry-run.

One initial unweighted SASRec checkpoint is shared by every selected condition.
No provider is constructed in the full-no-llm suite or during --prepare-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys

from .experiment_methodology import CAMPAIGNS, CONDITIONS_BY_ID

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
POPULATION_SEEDS = {42: 1054, 43: 19277, 44: 18844}


@dataclass(frozen=True)
class CampaignRequest:
    suite: str
    seed: int = 42
    population_seed: int = 1054
    active_users: int = 300
    reserve_users: int = 0
    iterations: int = 30
    rank_penalty: int = 5
    non_head_weight: float = 1.25
    weight_scope: str = "all"

    def __post_init__(self):
        if self.suite not in CAMPAIGNS:
            raise ValueError("unknown mitigation suite")
        if self.seed not in POPULATION_SEEDS:
            raise ValueError("the prepared seed panels are 42, 43, and 44")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.suite == "full-no-llm":
            if (self.active_users, self.reserve_users) != (6040, 0):
                raise ValueError("full-no-llm requires 6040 active and 0 reserve")
        elif (self.active_users, self.reserve_users) not in {(100, 200), (300, 0)}:
            raise ValueError("primary requires 100 active + 200 reserve or 300 active + 0 reserve")
        if self.suite == "primary" and self.population_seed != POPULATION_SEEDS[self.seed]:
            raise ValueError("simulation seed must match its frozen selected population")
        if self.rank_penalty != 5 or self.non_head_weight != 1.25:
            raise ValueError("the initial matrix fixes penalty=5 and non-head weight=1.25")
        if self.weight_scope not in {"all", "simulated_only"}:
            raise ValueError("unknown positive-target weighting scope")

    @property
    def experiment_id(self):
        return f"mitigation_v1_{self.suite.replace('-', '_')}_p{self.population_seed}_a{self.active_users}_r{self.reserve_users}"

    def describe(self):
        conditions = [CONDITIONS_BY_ID[name] for name in CAMPAIGNS[self.suite]]
        opportunities = self.active_users * self.iterations
        response_calls = sum(opportunities * (4 if c.intervention == "post_processing" else 3) for c in conditions)
        llm_calls = sum(opportunities * (4 if c.intervention == "post_processing" else 3)
                        for c in conditions if c.user_model in {"direct_llm", "agentic_recommender"})
        return {**asdict(self), "experiment_id": self.experiment_id,
                "conditions": [asdict(c) for c in conditions],
                "system_trajectories": len(conditions), "response_opportunities_per_trajectory": opportunities,
                "response_model_calls": response_calls,
                "llm_response_model_calls": llm_calls,
                "maximum_canonical_reflections": 3 * opportunities if self.suite == "primary" else 0,
                "profile_users": 300 if self.suite == "primary" else 0,
                "budget_note": "Agent responses may call planning, action, and optional memory. Reflections and profile extraction are additional. Retries can add calls; response-model calls are not API-call counts.",
                "ncm_protocol": "legacy_full_observed_exclusion",
                "training_users": self.active_users + self.reserve_users}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(CAMPAIGNS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-seed", type=int)
    parser.add_argument("--active-users", type=int)
    parser.add_argument("--reserve-users", type=int)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--weight-scope", choices=("all", "simulated_only"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ncm-device", default="cpu")
    parser.add_argument("--recommender-epochs", type=int, default=201)
    parser.add_argument("--recommender-batch-size", type=int, default=128)
    parser.add_argument("--raw-data", type=Path, default=ROOT / "data/ml-1m")
    parser.add_argument("--input-root", type=Path, default=ROOT / "inputs/mitigation_v1")
    parser.add_argument("--ncm-root", type=Path, default=ROOT / "data/ncm_checkpoints")
    parser.add_argument("--embedding-directory", type=Path, default=ROOT / "data/text_embedding_3_small_movielens_v1")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--provider", choices=("openai", "ollama"), default="openai")
    parser.add_argument("--provider-settings-file", type=Path,
                        help="Optional JSON provider settings; API credentials remain in the environment.")
    parser.add_argument("--profile-max-interactions", type=int, default=100)
    parser.add_argument("--profile-batch-size", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Print matrix and opportunity counts; no files or provider calls.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare/validate data, NCM and shared SASRec only; no provider calls or profiles.")
    parser.add_argument("--only-condition", choices=tuple(CONDITIONS_BY_ID))
    parser.add_argument("--resume-from", type=Path, help="Completed iteration directory; requires --only-condition and a new --output-root.")
    return parser


def _resolve(args):
    full = args.suite == "full-no-llm"
    request = CampaignRequest(
        suite=args.suite, seed=args.seed,
        population_seed=args.population_seed if args.population_seed is not None else (args.seed if full else POPULATION_SEEDS.get(args.seed, -1)),
        active_users=args.active_users if args.active_users is not None else (6040 if full else 300),
        reserve_users=args.reserve_users if args.reserve_users is not None else 0,
        iterations=args.iterations, weight_scope=args.weight_scope,
    )
    if args.recommender_epochs < 1 or args.recommender_batch_size < 1:
        raise ValueError("SASRec epochs and batch size must be positive")
    if args.profile_max_interactions < 1 or args.profile_batch_size < 1:
        raise ValueError("profile history limit and batch size must be positive")
    if args.only_condition and args.only_condition not in CAMPAIGNS[request.suite]:
        raise ValueError("selected condition is outside this suite")
    if args.resume_from and (not args.only_condition or not args.output_root):
        raise ValueError("restart requires --only-condition and an explicit new --output-root")
    return request


def main(argv=None) -> int:
    from .utility.run_logging import log_run_attempt
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        args = _parser().parse_args(argv)
        request = _resolve(args)
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            return 0
        # A fixed launcher directory keeps invalid paths and restart sources
        # untouched, while preserving argument/setup failures durably.
        with log_run_attempt(ROOT / "launcher_logs", "invalid_campaign_request"):
            raise
    if args.dry_run:
        description = request.describe()
        description["selected_condition"] = args.only_condition
        print(json.dumps(description, indent=2))
        return 0
    output_root = args.output_root or ROOT / "outputs" / f"{request.experiment_id}_seed{request.seed}_i{request.iterations}"
    with log_run_attempt(ROOT / "launcher_logs", request.experiment_id):
        execute_campaign(request, args, output_root=output_root)
    return 0


def execute_campaign(request, args, *, output_root: Path):
    from .automatic_inputs import ensure_automatic_inputs
    from .config import SimulatorConfig
    from .data_preparation import load_prepared_dataset
    from .initial_recommender import ensure_initial_recommender
    from .prepare_ncm_artifacts import ncm_settings
    from .run_resume import ResumeSource, validate_resume_destination
    from .run_simulator import run_experiment
    from .utility.saving import save_json_artifact
    from .utility.user_handling import PopulationRole, load_population_artifact

    selected = (args.only_condition,) if args.only_condition else CAMPAIGNS[request.suite]
    if args.resume_from:
        validate_resume_destination(args.resume_from, output_root)
        ResumeSource.load(args.resume_from, output_root,
                          run_id=f"{request.experiment_id}__{args.only_condition}__seed_{request.seed}",
                          condition_id=args.only_condition, seed=request.seed)
    # Detect output collisions before expensive preparation or paid calls.
    for name in selected:
        path = output_root / f"{request.experiment_id}__{name}__seed_{request.seed}"
        if path.exists() or path.with_name(path.name + ".incomplete").exists():
            raise FileExistsError(f"Run already exists; preserve it and select a fresh output root: {path}")
    provider_settings = {} if args.provider_settings_file is None else json.loads(args.provider_settings_file.read_text(encoding="utf-8"))
    if not isinstance(provider_settings, dict) or "provider" in provider_settings:
        raise ValueError("provider settings must be a JSON object without the separately selected provider")
    from .utility.providers import ProviderConfig
    provider_config = ProviderConfig(provider=args.provider, **provider_settings)
    common_inputs = dict(raw_data_directory=args.raw_data, input_root=args.input_root,
                         population_seed=request.population_seed, active_user_count=request.active_users,
                         reserve_user_count=request.reserve_users, profile_max_interactions=args.profile_max_interactions,
                         profile_batch_size=args.profile_batch_size)
    paths = ensure_automatic_inputs(**common_inputs, require_profiles=False)
    dataset = load_prepared_dataset(paths.prepared_data)
    population = load_population_artifact(paths.population)
    ncm = {folder: ncm_settings(dataset, model_folder=folder, checkpoint_root=args.ncm_root,
                              embedding_directory=args.embedding_directory, seed=request.seed, device=args.ncm_device)
           for folder in ("P3", "P4")}
    rec_settings = {"device": args.device, "num_epochs": args.recommender_epochs,
                    "batch_size": args.recommender_batch_size}
    initial = ensure_initial_recommender(dataset=dataset,
                training_user_ids=tuple(member.user_id for member in population.members
                                        if member.role in {PopulationRole.ACTIVE, PopulationRole.RESERVE}),
                seed=request.seed, settings=rec_settings, checkpoint_root=args.input_root / "initial_sasrec")
    if args.prepare_only:
        print(f"Provider-free preparation complete. Shared initial SASRec: {initial}")
        return
    # Save the actual invocation and model settings before
    # starting any trajectory. No API secret is copied from the environment.
    campaign = {**request.describe(), "selected_conditions": list(selected),
                "initial_recommender": str(initial.resolve()), "recommender_settings": rec_settings,
                "input_root": str(args.input_root.resolve()), "provider_id": args.provider,
                "provider_settings": provider_settings, "ncm_settings": ncm}
    output_root.mkdir(parents=True, exist_ok=True)
    save_json_artifact(output_root / "campaign_manifest.json", campaign)
    profile_path = None
    if args.resume_from and CONDITIONS_BY_ID[args.only_condition].user_model == "agentic_recommender":
        # Recovery needs the exact original frozen input; never regenerate it
        # through paid preparation before checkpoint compatibility is checked.
        profile_path = ensure_automatic_inputs(**common_inputs, require_profiles=True, provider=None).profiles
    for name in selected:
        condition = CONDITIONS_BY_ID[name]
        settings = ncm["P3"] if condition.user_model == "ncm_inspired_gru_uniform" else (
            ncm["P4"] if condition.user_model == "ncm_inspired_gru_mansoury" else {})
        if condition.user_model == "agentic_recommender" and profile_path is None:
            from .utility.providers import build_provider
            provider = build_provider(provider_config)
            profile_path = ensure_automatic_inputs(**common_inputs, require_profiles=True, provider=provider).profiles
        config = SimulatorConfig(
            experiment_id=request.experiment_id, population_seed=request.population_seed,
            num_iterations=request.iterations, slate_size=10,
            active_user_count=request.active_users, reserve_user_count=request.reserve_users,
            prepared_data_path=paths.prepared_data, population_artifact_path=paths.population,
            profile_artifact_path=profile_path if condition.user_model == "agentic_recommender" else None,
            initial_recommender_path=initial, output_root=output_root,
            provider_id=args.provider, provider_settings=provider_settings,
            recommender_settings=rec_settings, user_model_settings=settings,
            policy_settings={"rank_penalty": 5, "non_head_weight": 1.25, "weight_scope": request.weight_scope},
            recommender_fit_interval=1, checkpoint_interval=1, training_scope="active_reserve",
            raw_data_path=args.raw_data, resume_checkpoint_path=args.resume_from,
        )
        logger.info("Starting campaign condition %s", name)
        artifact = run_experiment(config, condition, request.seed)
        logger.info("Completed campaign condition %s at %s", name, artifact.output_dir)
    save_json_artifact(output_root / "_CAMPAIGN_COMPLETE.json", {"conditions": list(selected), "status": "complete"})


if __name__ == "__main__":
    raise SystemExit(main())
