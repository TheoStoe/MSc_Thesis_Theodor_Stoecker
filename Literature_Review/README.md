# Literature-review materials

This directory contains the final classification, OpenAI second-screening results and prompts, and forward-search scripts with the results saved for the thesis.

- `final_classification.csv`: the final human-curated table of 35 studies across eleven dimensions. Multiple labels in a cell are separated by `; `.
- `second_screening_openAI/results.jsonl`: one consolidated file containing all 790 saved OpenAI calls, including excluded papers. Each line records the batch, stage, paper, model metadata, and the complete structured model output in `result`.
- `second_screening_openAI/prompts.py`: the original prompt templates and builders, response schemas, and recorded model settings. Importing it does not make API calls.
- `snowballing/`: the OpenAlex/Semantic Scholar query scripts, their seed lists, and the historical result CSVs.

Contains data from [Semantic Scholar](https://www.semanticscholar.org/?utm_source=api) and [OpenAlex](https://openalex.org/).

Abstracts and citation-context passages are omitted from the bundled public CSVs; the complete originals are retained privately.

## Reproduce the historical search counts

Run these commands from this directory, using Python 3.10 or later:

```powershell
python -m pip install -r requirements.txt
python snowballing/query.py --summary
```

The two saved files combine the main and additional searches. The `search_batch` column retains their origin. They contain 186 OpenAlex and 604 Semantic Scholar citation relations, and 502 screening records: 495 main and seven additional.

| Saved file in `snowballing/results/` | Meaning                                                                                                                    |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `citation_relations.csv`             | All 790 citation records, including repeated discoveries across seeds or databases: 604 Semantic Scholar and 186 OpenAlex. |
| `screening_papers.csv`               | The 502 saved screening records after deduplication.                                                                       |

## Query the databases again

The seed CSVs are retained so the searches can be rerun. The following commands describe a search without contacting the databases:

```powershell
python snowballing/query.py --input snowballing/seeds_main.csv
python snowballing/query.py --input snowballing/seeds_additional.csv
```

To issue new queries, explicitly add `--run` and choose a new output directory:

```powershell
python snowballing/query.py --input snowballing/seeds_main.csv --output new_search/main --run
python snowballing/query.py --input snowballing/seeds_additional.csv --output new_search/additional --run
```

Use `--help` for database selection. `openalex_query.py` retains the separate OpenAlex-only collector. Credentials, where needed, are read from environment variables. New queries can return different results; the included CSVs preserve the results used for the thesis. Historical raw API JSON and retrieval timestamps were not retained. The available collectors do not apply a publication-date cutoff; screening rules are separate.

## Count the final taxonomy labels

The final table contains readable labels and preserves None, Not applicable, Not reported, and Unclear as distinct values. This standard-library example reproduces category frequencies from the table; multiple labels can be assigned to one study.

```python
import csv
from collections import Counter

with open("final_classification.csv", encoding="utf-8-sig", newline="") as stream:
    studies = list(csv.DictReader(stream))
for dimension in list(studies[0])[1:]:
    counts = Counter(label for study in studies for label in study[dimension].split("; "))
    print(dimension, dict(counts))
```

Original code is supplied without an assigned licence. Dependencies retain their own licences.
