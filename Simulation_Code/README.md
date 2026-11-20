# User modelling and bias amplification in recommender-system simulations

## Contents

| Path                                  | Purpose                                                                                  |
| ------------------------------------- | ---------------------------------------------------------------------------------------- |
| `feedback_loop_experiment/`           | MovieLens feedback stages L0–L5, user response models, agent ablations, and order probes |
| `mitigation_experiment/`              | MovieLens in-processing and post-processing mitigation                                   |
| `new_dataset_experiment/`             | Goodreads preparation, response models, and popularity/gender analyses                   |
| `new_dataset_experiment_mitigation/`  | Computational Goodreads gender-mitigation experiments                                    |
| `reported_runs/analysis/`             | Shared metrics, executed notebooks, plots, and LaTeX tables                              |
| `datasets/README.md`                  | Dataset acquisition, preparation, and adding a dataset                                   |
| `tests/test_end_to_end.py`            | One small, provider-free end-to-end test                                                 |
| `artifacts/manifest.json`             | Checksums and inventory of the optional full local artifact bundle                       |
| `LICENSES/`, `THIRD_PARTY_NOTICES.md` | Third-party licences, attribution, and dataset terms                                     |

## Installation

Use Python 3.14 and [uv](https://docs.astral.sh/uv/). The lock file retains the thesis
environment's PyTorch 2.13.0 CUDA 12.6 build on Windows/Linux. That build can also
execute the small test on the CPU; the macOS lock uses the platform build.

```powershell
cd Simulation_Code
uv sync --locked --group dev
uv run python -m pytest --basetemp=.build/e2e_tmp
```

The test creates synthetic MovieLens-shaped data, prepares a population, fits and
restores a small shared SASRec checkpoint, executes two L2 iterations and all three
probe arms, and checks the resulting checkpoint, completion hashes, and head metrics.
It does not use an API or the research datasets.

Commands below use PowerShell. `uv run python` can be replaced by `python` after
activating an environment containing the locked dependencies. Large thesis runs use
`--device cuda`; use `--device cpu` for CPU execution.

## Recreate the reported tables and figures

The default analysis uses the bundled November 16 results snapshot. It covers 183
completed physical trajectories and 243 logical condition memberships, with no pending
cells. The epoch-budget appendix is a separate, seed-42 analysis, including 18 earlier
reference trajectories. Multiple logical controls can refer to the same physical run.

```powershell
uv run python -m reported_runs.analysis.execute --report .build/notebook_execution.json
uv run jupyter lab reported_runs/analysis/notebooks
```

The notebooks contain short dataframe selections and calls to shared metric,
plotting, and table helpers. Executing them produces PDF/PNG plots and `.tex` tables;
table labels follow `chap:results:tab:analysis_name`. Executed notebooks and HTML
versions are included. Notebooks retain headings, calculation code, plots, numerical
tables, and export links; interpretation is provided in the thesis.

The default replay needs no dataset download, private archive, model training, or
API key. It uses saved measurements; it is not a new simulation. Source-bearing
profile evidence and click-reason text are held in the private bundle. The public
profile-review and advertising-spike notebooks retain numerical results and recorded
review categories without redistributing those source records.

Final exposure analyses use `d(r) = 1 / log2(r + 1)` at one-based displayed ranks.
Head-click share excludes no-clicks and invalid responses, and is undefined if there
are no valid clicks. Advertising comparisons additionally report exposure over
displayed ranks 2–10. Post-processing results distinguish total, immediate-reranking,
and remaining trajectory effects using the non-persistent original-order probe.

## Prepare inputs

Follow [the dataset guide](datasets/README.md) before running new experiments. Keep
source data and generated inputs in the ignored `datasets/raw/` and
`datasets/processed/` folders. No data is downloaded automatically.

For each MovieLens seed, prepare one dataset/population and one **unweighted initial
SASRec checkpoint**, then reuse them across the compared conditions. The same applies
to profiles and NCM artifacts. Do not regenerate these independently for treatment
and control.

The main runs use 300 active users and no reserve; sampling and advertising use 150
active users and 150 reserve users. Both settings train on the selected 300 users.
All main runs use 30 iterations, lists of ten items, 804 SASRec epochs per fit, batch
size 128, and maximum sequence length 200. These are thesis settings, not quick-start
settings. The test uses deliberately smaller settings.

| Simulation seed | MovieLens population seed | Goodreads population seed |
| --------------- | ------------------------- | ------------------------- |
| 42              | 1054                      | 16040                     |
| 43              | 19277                     | 11615                     |
| 44              | 18844                     | 5245                      |

## MovieLens feedback and user response models

After preparation, set the input locations. `initial_checkpoint` must identify the
actual hash-named checkpoint directory recorded by `prepare_movielens_models.py`.

```powershell
$data = 'datasets/processed/ml_seed42'
$checkpointName = (Get-Content "$data/models/sasrec_seed42.json" -Raw | ConvertFrom-Json).checkpoint_subdirectory
$initial = Join-Path "$data/models" $checkpointName
uv run python -m feedback_loop_experiment.run_simulator --list-conditions
uv run python -m feedback_loop_experiment.run_simulator --experiment thesis_ml --condition L2_ml_model__position_biased__sasrec__order_probe --seed 42 --population-seed 1054 --iterations 30 --slate-size 10 --active-users 300 --reserve-users 0 --prepared-data "$data/prepared" --population "$data/population.json" --initial-recommender $initial --output-root feedback_loop_experiment/outputs/new_ml_seed42_position_l2 --training-scope active-reserve --recommender-epochs 804 --recommender-batch-size 128 --device cuda
```

Change `--condition` and use a **new output directory** for each run. The condition
catalogue is authoritative; representative identifiers are:

| Comparison                                    | Condition identifiers                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------ |
| Frozen-history baseline                       | `L0_control__position_biased__sasrec__order_probe`                                   |
| History updates                               | `L1_feature__position_biased__sasrec__order_probe`                                   |
| Recommender retraining                        | `L2_ml_model__position_biased__sasrec__order_probe`                                  |
| Frozen agent                                  | `L2_ml_model__agentic_recommender__sasrec__order_probe`                              |
| Agent, interaction records without reflection | `L3_individual__agentic_recommender__sasrec__no_reflection_control__order_probe`     |
| Agent, reflection with fixed profile          | `L3_individual__agentic_recommender__sasrec__fixed_persona_memory_only__order_probe` |
| Agent, full reflection                        | `L3_individual__agentic_recommender__sasrec__full_reflection__order_probe`           |
| Sampling, proportional cluster refill         | `L4_sampling__agentic_recommender__sasrec__full_reflection`                          |
| Sampling, no refill                           | Add `__exit_only_ablation` to the preceding L4 identifier                            |
| Sampling, uniform refill                      | Add `__uniform_refill_control` to the preceding L4 identifier                        |
| Advertising controls and treatment            | `O0_organic`, `O1_static_ad`, `O2_adaptive_ad_three_strike`                          |

For L0/L1/L2, replace `position_biased` by `ncm_inspired_gru_uniform`,
`ncm_inspired_gru_mansoury`, `direct_llm`, or `agentic_recommender` where listed.
NCM runs additionally need `--user-model-settings` containing their validated
`checkpoint_directory` and `embedding_directory`; see the dataset guide. Agent
runs need `--profiles "$data/profiles"`.

For L4 and advertising, partition the existing 300-user cohort into a 150/150
population using the dataset guide, preserving its users and profiles. Prepare or
reuse the matching shared initial checkpoint. Use `--active-users 150 --reserve-users 150`
with those artifacts. The organic, static, and adaptive advertising runs all use
full-reflection agent state updates. Only the specified mechanisms differ.

Conditions ending in `__order_probe` record canonical, identical-order replica, and
shuffled responses at the same opportunity. Only the canonical response affects
later state.

## LLM conditions

LLM response and profile-generation commands make external API requests when run.
Set your own key in the shell; credentials are never part of the experiment settings.

```powershell
$env:OPENAI_API_KEY = 'your-key'
$provider = '{"model":"gpt-5.4-nano","reasoning_effort":"none","max_output_tokens":2056}'
```

Pass `--provider openai --provider-settings $provider` to the MovieLens runner. The
recorded simulation calls use this model, no provider seed, and the provider's
default temperature. Fresh responses can differ from the saved runs. Profile
generation is a separate preparation step; reusing profiles avoids charging for
them again.

## MovieLens mitigation

Use the same prepared inputs, population, profiles/NCM artifacts, and unweighted
initial checkpoint as the matched control. Mitigation identifiers are
`<model>__control`, `<model>__in_processing`, and `<model>__post_processing`.
The agent uses full-reflection L3; the other response models use L2.

```powershell
uv run python -m mitigation_experiment.run_simulator --list-conditions
uv run python -m mitigation_experiment.run_simulator --experiment thesis_ml_mitigation --condition position_biased__post_processing --seed 42 --population-seed 1054 --iterations 30 --slate-size 10 --active-users 300 --reserve-users 0 --prepared-data "$data/prepared" --population "$data/population.json" --initial-recommender $initial --output-root mitigation_experiment/outputs/new_ml_seed42_position_post --training-scope active-reserve --recommender-epochs 804 --recommender-batch-size 128 --policy-settings '{"non_head_weight":1.25,"rank_penalty":5,"weight_scope":"all"}' --device cuda
```

In-processing weights non-head training items by 1.25 during feedback refits.
Post-processing applies a rank penalty of 5 to head items within the top-ten list.
The post-processing runner includes the non-persistent original-order probe used
to separate the immediate ordering effect from the remaining trajectory difference.
The reranking step preserves unweighted exposure within the same opportunity.

## Goodreads comparisons

The dataset guide supplies commands for ingestion, cohort selection, initial models,
description excerpts, profiles, and optional gender labels. A prepared Goodreads
input root contains its `prepared/`, `population.json`, shared checkpoint, and the
response artifacts required by the selected model.

```powershell
uv run python -m new_dataset_experiment.campaign --inputs datasets/processed/goodreads42 --population-seed 16040 --seed 42 --output new_dataset_experiment/outputs/new_goodreads42 --condition L2_ml_model__position_biased__sasrec__order_probe --check
uv run python -m new_dataset_experiment.campaign --inputs datasets/processed/goodreads42 --population-seed 16040 --seed 42 --output new_dataset_experiment/outputs/new_goodreads42 --condition L2_ml_model__position_biased__sasrec__order_probe --run --device cuda
```

Without `--run`, these commands describe or validate the job. The comparison
catalogue contains L0 and L2 for all five response models, plus full-reflection agent
L3. Computational L1 runs are also represented in the separate gender-mitigation
design below. L2 minus L0 and the agent's L3 minus L2 answer different questions.

LLM conditions additionally take `--item-summaries <excerpt-artifact>` and agent
conditions take `--profiles <profile-bundle>`. Use the same verified excerpt policy
and profiles throughout each matched seed. Goodreads uses source-description
excerpts of at most 200 characters; MovieLens response prompts omit plot summaries.

## Goodreads gender mitigation

This extension is computational only: position-biased, NCM P3, and NCM P4. It exposes
L0/L1/L2 crossed with control, in-processing, and post-processing. Supply author
labels prepared from the documented external metadata; do not infer labels from
names.

```powershell
uv run python -m new_dataset_experiment_mitigation.run_cli --list-conditions
uv run python -m new_dataset_experiment_mitigation.run_cli --inputs datasets/processed/goodreads42 --labels datasets/processed/goodreads42/labels/item_labels.json --output new_dataset_experiment_mitigation/outputs/new_gender42 --seed 42 --population-seed 16040 --model position_biased --stage L2_ml_model --intervention post_processing --check
uv run python -m new_dataset_experiment_mitigation.run_cli --inputs datasets/processed/goodreads42 --labels datasets/processed/goodreads42/labels/item_labels.json --output new_dataset_experiment_mitigation/outputs/new_gender42 --seed 42 --population-seed 16040 --model position_biased --stage L2_ml_model --intervention post_processing --run --device cuda
```

The initial model remains unweighted. In-processing affects L2 feedback refits;
at L0/L1 it is the same scientific condition as the control. The implementation
records this relationship rather than interpreting it as an independent treatment.

## Full logs, checkpoints, and source-bearing evidence

Compressed full-run and input archives are excluded from Git, due to file size and licensing. They contain source histories,
ratings, metadata, and profiles, including records embedded in `.pt` checkpoints.
MovieLens and Goodreads prohibit redistribution of their datasets without the
necessary permission.

## Licensing

No licence has been assigned to the original project code. Third-party code retains
its own terms and attribution; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
