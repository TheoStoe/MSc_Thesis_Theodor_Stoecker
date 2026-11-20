# Datasets and new data

Run all commands from the repository root. Download data yourself from the original
providers and follow their terms. Raw data, prepared inputs, generated profiles,
embeddings, and checkpoints belong in the ignored `datasets/raw/`,
`datasets/processed/`, and `datasets/cache/` folders.

- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/): obtain `ratings.dat`,
  `movies.dat`, and `users.dat`. The [dataset README](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)
  includes its usage conditions and citation.
- [UCSD Goodreads](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html): obtain
  book metadata, author metadata, and **dated** review/interaction JSONL data. The
  undated interaction CSV alone cannot establish the chronological splits used here.
- Author-gender labelling additionally requires the documented VIAF authority data
  and the local matching/audit procedure. Do not substitute guessed labels.

The providers restrict redistribution. Do not commit the downloaded data or detailed
artifacts containing their records. Dataset terms and third-party attribution are
summarised in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## MovieLens preparation

Put the three original files in `datasets/raw/ml-1m/`. Prepare one population for
each simulation seed. To reproduce the recorded populations, first restore the
local input archives (`python artifact_bundle.py restore --category inputs`). These
include the frozen membership definitions. Then use the exact-cohort preparation
command below for the main 300-active, zero-reserve population:

```powershell
uv run python -m mitigation_experiment.prepare_inputs --raw-data datasets/raw/ml-1m --prepared-output datasets/processed/ml_seed42/prepared --cluster-output datasets/processed/ml_seed42/clusters --population-output datasets/processed/ml_seed42/population.json --population-seed 1054 --active-users 300 --reserve-users 0
uv run python prepare_movielens_models.py --stage sasrec --prepared-data datasets/processed/ml_seed42/prepared --population datasets/processed/ml_seed42/population.json --output datasets/processed/ml_seed42/models --seed 42 --device cuda --run
```

Preparation retains ratings as implicit interactions, applies the iterative five-core,
maps IDs consistently, sorts events chronologically, and holds out each user's final
two interactions for validation/test. Clustering uses training histories. The SASRec
preparation command fits an unweighted model once on the selected active and reserve
users, with 804 epochs, batch size 128, and sequence length 200 by default.
`models/sasrec_seed42.json` records the exact checkpoint directory and settings.
Without `--run`, model preparation only describes the requested operation.

The original 300-user cohort is the union of a frozen 100-active/200-reserve draw;
its users were subsequently all made active. Drawing 300 users directly with the
same seed need not select the same users. If you do not have the local frozen
membership bundle, use `feedback_loop_experiment.prepare_inputs` with the same flags
to create a **new cohort**, and report that choice. The generic loader remains usable
for independently prepared data; it does not promise the original membership.

Use population seeds 19277 and 18844 with simulation seeds 43 and 44. For the
sampling/advertising experiments, partition the **same selected 300 users** into
150 active and 150 reserve users instead of sampling again:

```powershell
uv run python -m feedback_loop_experiment.l4_l5_population --source-population datasets/processed/ml_seed42/population.json --prepared-data datasets/processed/ml_seed42/prepared --destination datasets/processed/ml_seed42_150150 --profiles datasets/processed/ml_seed42/profiles --input-format mitigation
```

Create the profiles below before using the optional `--profiles` argument, or omit
that argument for computational preparation. This argument provides demographic
information for the partition; the command does not generate new profiles. Reuse
the original prepared dataset and profile bundle with the resulting 150/150 population. Use
`prepare_movielens_models.py` with the resulting population to prepare/reuse its
shared initial checkpoint. All 300 histories remain in `active-reserve` training.
Source IDs, preparation fingerprints, and the population seed remain paired.
For a newly sampled generic cohort, use `--input-format feedback` for the partition
instead of the exact-membership `mitigation` validation.

### Profiles for the MovieLens agent

The following command sends the selected training histories, demographic metadata,
and item text to the configured provider. Set your own `OPENAI_API_KEY` first.
Generate one profile bundle per population and reuse it.

```powershell
uv run python -m feedback_loop_experiment.prepare_profiles --prepared-data datasets/processed/ml_seed42/prepared --population datasets/processed/ml_seed42/population.json --raw-data datasets/raw/ml-1m --output datasets/processed/ml_seed42/profiles --max-interactions 100 --model gpt-5.4-nano --run
```

### NCM artifacts

NCM training needs item embeddings covering the exact prepared catalogue. Raw
MovieLens files do not contain the movie descriptions used for the original frozen
embedding texts. The local full-artifact bundle includes the original
`ncm_embeddings`, `ncm_p3`, and `ncm_p4` directories for each published input set;
these can be restored for local reproduction.

For independently prepared embeddings, retain their manifest, ordered IDs, text
builder, and any title/genre/description metadata. Use the embedding preparation
command's `--help` for validated import or generation options:

```powershell
uv run python prepare_movielens_models.py --stage embeddings --help
uv run python prepare_movielens_models.py --stage embeddings --prepared-data datasets/processed/ml_seed42/prepared --population datasets/processed/ml_seed42/population.json --output datasets/processed/ml_seed42/embeddings --embedding-source path/to/frozen_embedding_bundle --metadata path/to/item_metadata.json --run
uv run python -m feedback_loop_experiment.pretrain_ncm --prepared-data datasets/processed/ml_seed42/prepared --embedding-directory datasets/processed/ml_seed42/embeddings --placement-root datasets/processed/ml_seed42/training_lists --output-root datasets/processed/ml_seed42/ncm --seed 42 --policy uniform --policy mansoury --device cuda
```

The two GRUs use uniform and position-biased target placement respectively. Training
constructs ten-item lists; the GRU processes the list order, while prior clicks
initialise its hidden state through their mean embedding. The saved checkpoint
records its training-data and embedding identities. At simulation time pass a JSON
object to `--user-model-settings`, for example:

```powershell
$ncm = '{"checkpoint_directory":"datasets/processed/ml_seed42/ncm/P3/seed_42/checkpoint","embedding_directory":"datasets/processed/ml_seed42/embeddings"}'
```

Use that object with the `ncm_inspired_gru_uniform` condition. The position-biased
variant uses its `P4/seed_42/checkpoint` directory and
`ncm_inspired_gru_mansoury`. For original imported NCM artifacts, retain the recorded
`expected_prepared_fingerprint` when it refers to their earlier calibration artifact;
do not replace fingerprints to bypass validation.

If you do not have a compatible frozen bundle, the same helper can generate
`text-embedding-3-small` embeddings from your supplied metadata. This makes API
requests only with both `--generate` and `--run`:

```powershell
uv run python prepare_movielens_models.py --stage embeddings --prepared-data datasets/processed/ml_seed42/prepared --population datasets/processed/ml_seed42/population.json --output datasets/processed/ml_seed42/embeddings --metadata path/to/item_metadata.json --metadata-source 'source of the supplied descriptions' --generate --run
```

`item_metadata.json` has an `items` object keyed by runtime item ID. Each entry
contains `item_id`, `original_item_id`, `title`, `genres`, and `description`; use
explicit `null` for genuinely absent descriptions. The helper validates the source
ID/title/genre mapping and records text and embedding fingerprints. It does not
download descriptions. Alternatively, `--title-genres-only` explicitly creates a
different text setting without summaries; this is not exact reproduction of the
thesis's frozen NCM inputs. Fresh provider vectors or newly trained GRUs can differ
from the saved artifacts.

## Goodreads preparation

The public ingestion code accepts JSONL, optionally gzip-compressed. These commands
only read local source data; they do not download the datasets.

```powershell
uv run python -m new_dataset_experiment.ingest --interactions datasets/raw/goodreads/goodreads_reviews_dedup.json.gz --books datasets/raw/goodreads/goodreads_books.json.gz --authors datasets/raw/goodreads/goodreads_book_authors.json.gz --output datasets/cache/goodreads.sqlite
uv run python -m new_dataset_experiment.representative_features --index datasets/cache/goodreads.sqlite --output datasets/cache/goodreads_features
uv run python -m new_dataset_experiment.representative_sampling --cache datasets/cache/goodreads_features --search datasets/cache/goodreads_population_search
uv run python -m new_dataset_experiment.representative_sampling --cache datasets/cache/goodreads_features --search datasets/cache/goodreads_population_search --index datasets/cache/goodreads.sqlite --freeze datasets/processed/goodreads_cohort42 --candidate-rank 1
uv run python -m new_dataset_experiment.prepare_models --inputs datasets/processed/goodreads42 --stage inputs --cohort datasets/processed/goodreads_cohort42/cohort.json --population-seed 16040 --run
uv run python -m new_dataset_experiment.prepare_models --inputs datasets/processed/goodreads42 --stage sasrec --seed 42 --device cuda --run
```

Cohort selection uses the frozen feature cache before any simulated outcomes.
`--candidate-rank 1`, `2`, and `3` select the three retained cohorts of that search.
Check the selected cohort's actual population seed: the thesis populations are
16040, 11615, and 5245 paired with simulation seeds 42, 43, and 44. Do not relabel an
arbitrary newly selected cohort with those numbers. A different source release or
preparation can produce a different cohort.

Goodreads preparation uses `date_updated` as the chronological proxy; it does not
recover the date on which a book was read. Prepared catalogues retain source IDs,
work/primary-author mappings, and metadata needed by subsequent label audits.

For NCMs, the embedding stage makes embedding API calls, and the NCM stage trains
the two GRUs. For the LLM conditions, the excerpt stage writes local bounded text;
profile generation makes LLM API calls.

```powershell
uv run python -m new_dataset_experiment.prepare_models --inputs datasets/processed/goodreads42 --stage embeddings --run
uv run python -m new_dataset_experiment.prepare_models --inputs datasets/processed/goodreads42 --stage ncm --seed 42 --variant P3 --variant P4 --device cuda --run
uv run python -m new_dataset_experiment.item_summaries --prepared datasets/processed/goodreads42/prepared --output datasets/processed/goodreads42/item_summaries --max-description-chars 200 --prepare
uv run python -m new_dataset_experiment.prepare_profiles --inputs datasets/processed/goodreads42 --item-summaries datasets/processed/goodreads42/item_summaries --output datasets/processed/goodreads42/profiles --run
```

Profiles use the latest 40 training events. Description excerpts are restricted to
the LLM input overlay and are distinct from the NCM embedding representation. Reuse
the same excerpt artifact and profile bundle for matched conditions.

### Author labels

The author-gender extension uses external metadata rather than inferring gender
from an author's name. Prepare the VIAF audit, inspect its ambiguous/unknown matches,
and export the established labelling policy.

```powershell
uv run python -m new_dataset_experiment.viaf_audit --authors datasets/raw/goodreads/goodreads_book_authors.json.gz --viaf datasets/raw/viaf/authority.xml.gz --cohort datasets/processed/goodreads_cohort42/cohort.json --output datasets/processed/goodreads_author_audit42
uv run python -m new_dataset_experiment.gender_labels --cohort datasets/processed/goodreads_cohort42/cohort.json --matches datasets/processed/goodreads_author_audit42/matches.jsonl --summary datasets/processed/goodreads_author_audit42/summary.json --output datasets/processed/goodreads42/labels
```

The label export needs a new output directory. Preserve its unknown and ambiguous
labels and its coverage statistics. Pass the resulting
`datasets/processed/goodreads42/labels/item_labels.json` to the portable gender-mitigation
runner. Gender labels do not affect the baseline recommender; the mitigation
extension uses the explicitly selected policy.

## Add a dataset

There is no universal raw-file loader. Add a small adapter following
`feedback_loop_experiment/data_preparation/movielens.py` or the Goodreads
`data_preparation/` modules. Keep domain-specific ingestion separate from the
simulator and metric definitions.

1. **Define IDs and events.** Map source user/item IDs to stable positive integer
   runtime IDs; reserve zero for padding. Keep the reverse maps. Resolve duplicate
   events and define which interactions count before splitting.
2. **Prepare chronological training histories.** Emit ordered user-item histories,
   training counts, and separate validation/test targets. Future held-out events
   must not enter profiles, initial popularity, clusters, embeddings that use user
   histories, or the initial training state.
3. **Prepare a catalogue.** Provide metadata and the text representation required
   by each response model. Verify that every history/target ID exists in it. State
   missing fields explicitly; do not quietly substitute another dataset's text.
4. **Save the shared artifact contract.** Return a `PreparedDatasetArtifact` and
   use the existing artifact saver. It checks schemas, source hashes, ID maps,
   catalogue identities, split targets, and history fingerprints. The simulator
   consumes these prepared contracts, not an arbitrary CSV.
5. **Select users and initialise once.** Freeze clustering and population selection,
   record the population/simulation seeds, then prepare the shared initial model,
   profiles, and NCM artifacts for that population. Reuse them across comparisons.
6. **Define optional groups and evaluation.** Freeze the head definition from
   training counts; document cutoff ties. Add external group labels only when
   appropriate, retaining unknown/ambiguous cases and denominators. Reuse the shared
   rank discount and metric definitions.
7. **Run a small local check first.** Validate catalogue coverage, a complete run,
   canonical versus non-persistent probe arms, and analysis denominators before
   starting long or paid experiments.

When analysing new runs, create an explicit source inventory and comparison mapping.
The bundled thesis inventory deliberately pins the final runs; it does not silently
discover new output folders. Shared metric functions can analyse a newly declared
source, while the published notebooks remain reproducible against their frozen
snapshot.
