# The Effect of User Modelling on Bias Amplification through Feedback Loops in Recommender System Simulations with LLM Agents

Simulation code, executed analysis notebooks, numerical results, and the literature
review for the thesis.

- [Simulation code and analysis guide](Simulation_Code/README.md)
- [Literature review](Literature_Review/README.md)
- [Dataset acquisition and preparation](Simulation_Code/datasets/README.md)

Use Python 3.14 and [uv](https://docs.astral.sh/uv/). From this repository's root:

```powershell
cd Simulation_Code
uv sync --locked --group dev
uv run python -m reported_runs.analysis.execute --report .build/notebook_execution.json
```

The analysis replays the bundled numerical snapshots and exports the reported
figures and tables. It requires no research datasets, private input bundle,
model training, or API key.

Source datasets, detailed run logs, and checkpoints are not distributed here.
See the dataset guide above for obtaining inputs and preparing new experiments.

ChatGPT (OpenAI) supported software development, data analysis, and the preparation of plots. Claude (Anthropic) supported code reviews.
