"""One offline pipeline check, using invented data and a real CPU recommender.

Run from the repository root with::

    python -B -m pytest --basetemp=.build/e2e_tmp -p no:cacheprovider

This deliberately uses a tiny training budget; it checks that the pipeline and
its analysis agree, rather than testing recommendation quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
CONDITION = "L2_ml_model__position_biased__sasrec__order_probe"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _cli(module: str, *arguments: object) -> None:
    environment = os.environ.copy()
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(REPO),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENAI_API_KEY="",
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", "-m", module, *map(str, arguments)],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_synthetic_movielens_to_simulation_and_head_metrics(tmp_path: Path) -> None:
    """Prepare, cluster, sample, train, simulate, checkpoint, and analyse offline."""
    assert tmp_path.resolve().is_relative_to(REPO / ".build"), (
        "Keep test artefacts inside this copy: use --basetemp=.build/e2e_tmp"
    )
    raw = tmp_path / "invented_ml_1m"
    raw.mkdir()
    (raw / "users.dat").write_text(
        "".join(f"{user}::{'F' if user % 2 else 'M'}::25::1::00000\n" for user in range(1, 13)),
        encoding="latin-1",
    )
    (raw / "movies.dat").write_text(
        "".join(f"{item}::Synthetic item {item:03d} (2000)::Action|Comedy\n" for item in range(1, 51)),
        encoding="latin-1",
    )
    (raw / "ratings.dat").write_text(
        "".join(
            f"{user}::{((user - 1) * 4 + position) % 50 + 1}::5::{1_000_000 + user * 1000 + position}\n"
            for user in range(1, 13)
            for position in range(30)
        ),
        encoding="latin-1",
    )
    inputs = tmp_path / "inputs"
    prepared = inputs / "prepared_data"
    clusters = inputs / "clusters"
    population = inputs / "population.json"
    _cli(
        "feedback_loop_experiment.prepare_inputs",
        "--raw-data", raw,
        "--prepared-output", prepared,
        "--cluster-output", clusters,
        "--population-output", population,
        "--population-seed", 42,
        "--active-users", 4,
        "--reserve-users", 2,
        "--svd-components", 2,
        "--clusters", 2,
        "--cluster-n-init", 2,
    )

    from feedback_loop_experiment.data_preparation.artifacts import load_prepared_dataset
    from feedback_loop_experiment.data_preparation.clustering import load_cluster_artifact
    from feedback_loop_experiment.utility.checkpoints import load_checkpoint
    from reported_runs.analysis.metrics import analyse_source

    dataset = load_prepared_dataset(prepared)
    cluster_artifact = load_cluster_artifact(clusters)
    assert len(dataset.training_histories) == 12
    assert len(dataset.catalog) == 50
    assert set(cluster_artifact.cluster_by_user) == set(dataset.training_histories)
    assert len(set(cluster_artifact.cluster_by_user.values())) == 2
    assert all(len(history) == 28 for history in dataset.training_histories.values())

    initial_root = tmp_path / "shared_initial"
    _cli(
        "prepare_movielens_models", "--stage", "sasrec",
        "--prepared-data", prepared, "--population", population,
        "--output", initial_root, "--seed", 42, "--device", "cpu",
        "--epochs", 1, "--batch-size", 4, "--max-sequence-length", 8,
        "--hidden-size", 8, "--num-blocks", 1, "--num-heads", 1,
        "--dropout-rate", 0, "--run",
    )
    receipt = json.loads((initial_root / "sasrec_seed42.json").read_text(encoding="utf-8"))
    initial = initial_root / receipt["checkpoint_subdirectory"]
    assert len(receipt["request"]["training_users"]) == 6
    initial_hash = hashlib.sha256((initial / "checkpoint.pt").read_bytes()).hexdigest()
    assert initial_hash == receipt["checkpoint_sha256"]

    output = tmp_path / "runs"
    _cli(
        "feedback_loop_experiment.run_simulator",
        "--experiment", "offline_e2e",
        "--condition", CONDITION,
        "--seed", 42,
        "--iterations", 2,
        "--active-users", 4,
        "--reserve-users", 2,
        "--prepared-data", prepared,
        "--population", population,
        "--initial-recommender", initial,
        "--training-scope", "active-reserve",
        "--raw-data", raw,
        "--output-root", output,
        "--recommender-settings", json.dumps(receipt["request"]["settings"]),
    )
    run = output / f"offline_e2e__{CONDITION}__seed_42"
    marker = json.loads((run / "_COMPLETE.json").read_text(encoding="utf-8"))
    assert marker["status"] == "complete"
    assert marker["completed_iteration"] == 2
    assert not (run / "_INCOMPLETE.json").exists()
    assert not (run / "_FAILED.json").exists()
    for filename, expected in marker["file_sha256"].items():
        assert hashlib.sha256((run / filename).read_bytes()).hexdigest() == expected

    events = _jsonl(run / "events.jsonl")
    assert len(events) == marker["row_counts"]["events"] == 24
    opportunities: dict[tuple[int, int], dict[str, dict]] = {}
    for event in events:
        key = event["iteration"], event["user_id"]
        arm = event["presentation_metadata"]["probe_arm"]
        assert arm not in opportunities.setdefault(key, {})
        opportunities[key][arm] = event
        assert event["action"]["status"] == "valid"
        assert event["action"]["action"] == "click"
        assert event["action"]["clicked_item_id"] in event["presented_slate"]
        assert len(set(event["presented_slate"])) == 10
    assert len(opportunities) == 8
    for arms in opportunities.values():
        assert set(arms) == {"canonical", "replica", "shuffled"}
        canonical, replica, shuffled = (arms[arm] for arm in ("canonical", "replica", "shuffled"))
        assert canonical["presented_slate"] == replica["presented_slate"]
        assert set(canonical["presented_slate"]) == set(shuffled["presented_slate"])
        assert canonical["action"]["clicked_item_id"] == replica["action"]["clicked_item_id"]

    checkpoint = load_checkpoint(
        run / "checkpoints" / "iteration_0002",
        expected_run_id=run.name,
        expected_condition_id=CONDITION,
        expected_seed=42,
    )
    assert checkpoint.completed_iteration == 2
    active = {user_id for _, user_id in opportunities}
    for user_id, initial_history in dataset.training_histories.items():
        expected_history = list(initial_history)
        if user_id in active:
            for iteration in (1, 2):
                event = opportunities[iteration, user_id]["canonical"]
                assert not set(event["presented_slate"]) & set(expected_history)
                expected_history.append(event["action"]["clicked_item_id"])
        assert list(checkpoint.component_states["observed_histories"][user_id]) == expected_history
        if user_id in receipt["request"]["training_users"]:
            assert list(checkpoint.component_states["recommender_training_histories"][user_id]) == expected_history
    assert set(checkpoint.component_states["recommender_training_histories"]) == set(receipt["request"]["training_users"])
    assert hashlib.sha256((initial / "checkpoint.pt").read_bytes()).hexdigest() == initial_hash
    diagnostics = _jsonl(run / "diagnostics.jsonl")
    assert len([row for row in diagnostics if row.get("event_name") == "initial_recommender_restored"]) == 1
    fits = [row for row in diagnostics if row.get("event_name") == "recommender_fit_completed"]
    assert [row["iteration"] for row in fits] == [1, 2]
    assert all(row["diagnostics"]["training_updates"] > 0 for row in fits)
    assert len({row["diagnostics"]["training_fingerprint"] for row in fits}) == 2

    # Exercise the same definitions used by the result notebooks. Probe clicks
    # must not inflate either the click or exposure denominator.
    frames = analyse_source({
        "dataset": "movielens", "inputs_directory": str(inputs),
        "source_directory": str(run), "study": "offline_e2e",
    }, repo_root=REPO)
    trajectory = frames["trajectory"].sort_values("iteration")
    popularity = dataset.initial_item_popularity
    head = set(sorted(popularity, key=lambda item: (-popularity[item], item))[:math.ceil(.2 * len(popularity))])
    for row in trajectory.to_dict("records"):
        canonical = [arms["canonical"] for (iteration, _), arms in opportunities.items() if iteration == row["iteration"]]
        clicks = sum(event["action"]["clicked_item_id"] in head for event in canonical)
        exposure = sum(item in head for event in canonical for item in event["presented_slate"])
        weights = [1 / math.log2(rank + 1) for rank in range(1, 11)]
        weighted = sum(weight for event in canonical for item, weight in zip(event["presented_slate"], weights) if item in head)
        assert row["valid_clicks"] == 4
        assert row["head_click_share"] == pytest.approx(clicks / 4)
        assert row["head_exposure"] == pytest.approx(exposure / 40)
        assert row["weighted_head_exposure"] == pytest.approx(weighted / (4 * sum(weights)))
    trajectory.to_csv(tmp_path / "head_metrics.csv", index=False)
    assert len(trajectory) == 2
    assert frames["health"].iloc[0]["events_sha256_verified"]
