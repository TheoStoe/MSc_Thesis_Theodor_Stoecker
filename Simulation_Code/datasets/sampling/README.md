# Population selection and sampling figure

Place this folder at `Simulation_Code/datasets/sampling/`. Run the commands below
from `Simulation_Code/`, using the repository's existing environment.

| File | Purpose |
| --- | --- |
| `movielens_population_selection.ipynb` | Portable copy of the historical MovieLens population-selection notebook, with aggregate saved results |
| `movielens_population_selection.html` | Readable version of the same saved notebook |
| `build_sampling_figure.py` | Recreates the age/occupation figure from saved aggregate counts, or recounts the fixed populations using locally supplied data |
| `build_goodreads_sampling_figures.py` | Recreates the two Goodreads sampling figures from saved aggregate values |
| `movielens_age_occupation_300_vs_full.csv` | Exact age/occupation counts and shares for the three selected 300-user populations and the 6,040-user reference |
| `goodreads_sampling_300_*.csv` | Recorded feature means, composition, and balance criteria for the three Goodreads populations |
| `goodreads_sampling_search.csv` | The original 20,000-candidate Goodreads seed search, containing seed-level scores only |
| `sampling_tolerances_exact.csv` | Numerical MovieLens and Goodreads sampling tolerances |
| `figures/` | Original saved PDF, PNG, and SVG of the MovieLens and Goodreads sampling figures |
| `provenance.json` | Historical source hashes and hashes of the distributed files |
| `goodreads_sampling_300_provenance.json` | Historical Goodreads extraction metadata and aggregate search summary |

## Recreate the sampling figure

```powershell
uv run python datasets/sampling/build_sampling_figure.py
```

This reads the bundled aggregate CSV and writes PDF, PNG, SVG, and the counts to
`datasets/sampling/figures/`. It requires no raw dataset, population search,
clustering, model training, or API calls. Use `--output-dir` to write elsewhere.

To recount the same fixed populations after obtaining MovieLens 1M:

```powershell
uv run python datasets/sampling/build_sampling_figure.py --recompute --output-dir datasets/processed/population_selection/figure
```

This uses `datasets/raw/ml-1m/users.dat` and the existing frozen membership files
under `mitigation_experiment/data/populations/`. It combines each population's
original 100 active and 200 reserve users. It does not draw a new sample.
Use `--raw-data` for another raw-data location. When prepared inputs exist under
`datasets/processed/ml_seed42/`, `ml_seed43/`, and `ml_seed44/`, their ID mappings
and active-300 populations are checked as well. An explicit `--prepared-root`
requires all three mappings and population files to be present.

## MovieLens population selection

```powershell
uv run jupyter lab datasets/sampling/movielens_population_selection.ipynb
```

The saved notebook can be inspected without running it. Running all cells requires
the original `ratings.dat`, `users.dat`, and `movies.dat` in
`datasets/raw/ml-1m/`; `MOVIELENS_RAW_DIR` can select another local directory.
It reconstructs preparation and the five preference clusters, searches seeds
0--19,999 for both sampling methods, and retains the original tolerance scoring
and tie-breaking rules. The recorded selected population seeds are 1054, 19277,
and 18844, paired with simulation seeds 42, 43, and 44.

The historical selection assessed 100 active users, 200 reserve users, and their
combined 300 users. The main experiments later activated all 300; the 150/150
sampling and advertising partition is a separate step in the dataset guide.

The public notebook retains aggregate tables and plots, with no individual user
rows or ordered-ID outputs. It keeps selected memberships in memory for its
checks. `EXPORT_LOCAL_RESULTS` is `False` by default. Set it to `True` to save
memberships, aggregate search results, and provenance under the Git-ignored
`datasets/processed/population_selection/` directory. The notebook does not train
SASRec, generate profiles, or make provider calls.

The adaptation changes the notebook's file hash. The historical notebook hash
recorded in the existing population files is preserved as provenance and must not
be replaced with the public copy's hash. `provenance.json` and the public notebook
metadata identify both the original source and the adaptation.

## Goodreads population selection and figures

The selection implementation is already included in
`new_dataset_experiment/representative_features.py` and
`new_dataset_experiment/representative_sampling.py`. Follow the Goodreads section
of `datasets/README.md` to prepare the source index, build the feature cache, run a
new search, or export a selected cohort from locally obtained data.

The historical search is supplied as `goodreads_sampling_search.csv`: 20,000
candidate seeds, with 17,762 passing the declared thresholds. Sorting passing
candidates by `worst_ratio`, `mean_ratio`, and `seed` gives population seeds 16040,
11615, and 5245. These are paired with simulation seeds 42, 43, and 44. Each
population has 300 readers; the eligible reference population has 750,205 readers.
This saved table contains seed-level scores, not individual reader records.

```powershell
uv run python datasets/sampling/build_goodreads_sampling_figures.py
```

This command replays both Goodreads figures from the three aggregate CSVs:
33 feature-mean rows, 60 composition rows, and 15 balance-criterion rows. It writes
PDF, PNG, and SVG into `datasets/sampling/figures/`. Use `--output-dir` to choose
another destination. No raw data, feature cache, individual cohort files, seed
search, training, or API calls are required for figure replay.

The provenance retains historical source hashes and aggregate search information.
Original reader-ID lists and machine-specific source paths are not included in the
public search summary. The raw feature cache and source-bearing cohort records
must be prepared or restored separately when rerunning the actual selection.

## Tolerances

`sampling_tolerances_exact.csv` contains 27 MovieLens thresholds: nine feature
groups evaluated for the original 100, 200, and combined 300 users. It also
contains the five Goodreads thresholds for 300 readers. These values are the
original saved thresholds, not thresholds selected for this add-on. TVD and SMD
thresholds are recorded on their original scales, not multiplied by 100.

The Goodreads thresholds were extracted from the saved representative-sampling
protocol. The Goodreads selection implementation remains in
`new_dataset_experiment/representative_sampling.py`; this add-on does not replace it.
