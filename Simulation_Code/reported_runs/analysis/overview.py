"""Render the offline analysis plan; no run execution or scientific aggregation."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


def load_specification(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(map(_cell, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(_cell, row)) + " |" for row in rows)
    return "\n".join(lines)


def _bullets(values) -> str:
    return "\n".join(f"- {value}" for value in values)


def decisions(spec: dict) -> str:
    c = spec["common_settings"]
    lines = ["## Common analysis decisions", "",
             f"**Scope:** seven main RQs. {spec['scope']['psychological_modelling']}", "",
             f"**Exposure discount:** `{c['rank_discount']}`. The first three weights are **1, 0.63093, and 0.5**.", "",
             c["rank_discount_normalisation"], "",
             "**Primary outcomes use canonical decisions and displayed lists.** " + c["state_updates"], "",
             "**Time summaries**", "", _bullets(c["time_summaries"]), "",
             "**Undefined values:** " + c["missing_values"], "",
             "**Matched effects:** " + c["pairing"], "",
             c["effect_sign"], "", "**Seed/population evidence:** " + c["replication"], "",
             c["seed_summaries"], "", "**Exposure source:** " + c["raw_and_displayed"], "",
             "**Candidate exclusions:** " + c["candidate_policy"], "",
             "**Recommendation quality:** " + spec["scope"]["recommendation_quality"], "",
             "**Thesis and notebook presentation:** " + c["writing_scope"]]
    return "\n".join(lines)


def study_plan(spec: dict) -> str:
    rows = []
    for s in spec["studies"]:
        pairs = "; ".join(f"{a}/{b}" for a, b in s["seed_population_pairs"])
        rows.append([s["name"], pairs, f"{s['active']}/{s['reserve']}",
                     s["expected_logical_cells"], s["scope"]])
    return ("## Planned study selection\n\n"
            "Counts below are configured logical cells, not a claim of current completion. "
            "Seed pairs are simulation/population seeds. Reused controls remain one physical trajectory.\n\n"
            + _table(["Study", "Seed pairs", "Initial active/reserve", "Configured cells", "Scope"], rows))


def exported_snapshot(report_root: str | Path, spec: dict) -> str:
    path = Path(report_root) / "manifest.json"
    if not path.is_file():
        return "## Existing export snapshot\n\nNo exported registry is present. The design remains readable; final analysis requires an explicitly selected, validated snapshot."
    m = json.loads(path.read_text(encoding="utf-8"))
    targets = {s["id"]: s for s in spec["studies"]}
    rows = []
    for s in m.get("studies", []):
        target = targets.get(s["study"])
        rows.append([s["study"], s.get("completed_logical_runs", "—"),
                     s.get("pending_runs", "—"),
                     target["expected_logical_cells"] if target else "optional supporting study"])
    return ("## Existing export snapshot\n\n"
            f"**Recorded export creation time:** `{m.get('created_at_utc', 'unknown')}`. "
            "These are saved registry observations, **not live run status**. "
            "The current Goodreads notebook selection has expanded since this export; the central selection needs updating before final results are built.\n\n"
            + _table(["Study", "Recorded complete logical cells", "Recorded pending", "Planned core cells"], rows)
            + "\n\nRecorded unique completed physical trajectories: **"
            + str(m.get("unique_completed_runs", "unknown")) + "**. "
            "Logical memberships and aliased controls must not be counted as extra replications.")


def metric_dictionary(spec: dict) -> str:
    groups = defaultdict(list)
    for metric in spec["metrics"]:
        groups[metric["group"]].append(metric)
    lines = ["## Shared metric definitions", "",
             "The final notebooks will call one implementation of these definitions. "
             "Within an iteration, exposure and click composition pool the declared slots/clicks. "
             "With ten-item slates, all-item weighted exposure equals the mean of per-user weighted exposures. "
             "The women-plus-men conditional ratio pools qualifying mass and is not the mean of user-specific conditional ratios.", ""]
    for group, metrics in groups.items():
        lines.extend([f"### {group}", "", _table(
            ["Metric", "Definition and denominator", "Units", "Zero denominator"],
            [[m["name"], m["definition"], m["unit"], m["zero_denominator"]] for m in metrics]), ""])
    return "\n".join(lines)


def rq_overview(spec: dict) -> str:
    metrics = {m["id"]: m["name"] for m in spec["metrics"]}
    rows = [[f"RQ {q['id']}", q["question"],
             "; ".join(metrics[k] for k in q["primary_metrics"]),
             q["notebook"]] for q in spec["research_questions"]]
    return "## Seven research questions\n\n" + _table(["RQ", "Question", "Primary measurements", "Planned notebook"], rows)


def rq_plan(spec: dict, rq_id: str) -> str:
    q = next(q for q in spec["research_questions"] if q["id"] == rq_id)
    metrics = {m["id"]: m["name"] for m in spec["metrics"]}
    lines = [f"## RQ {q['id']}: {q['title']}", "", f"**{q['question']}**", "",
             f"Planned results notebook: `{q['notebook']}`.", "",
             "### Selected comparisons", "", _bullets(q["comparisons"]), "",
             "**Primary measurements:** " + "; ".join(metrics[k] for k in q["primary_metrics"]) + ".", "",
             "**Supporting measurements:** " + "; ".join(metrics[k] for k in q["supporting_metrics"]) + ".", "",
             "### Notebook sections", "", _bullets(q["sections"]), "",
             "### Planned figures and numerical evidence", "", _bullets(q["views"]), "",
             "### Interpretation prompts", "", _bullets(q["interpretation_prompts"]), "",
             "### Scope of the comparison", "", _bullets(q["boundaries"])]
    return "\n".join(lines)


def qualitative_plan(spec: dict) -> str:
    q = spec["qualitative_review"]
    lines = ["## Qualitative profile review", "", q["status"], "",
             "**Condition:** " + q["condition"], "",
             "**Eligibility:** " + q["eligibility"], "",
             "**Selection:** " + q["selection"], "",
             "**Evidence:** " + q["evidence"], "",
             "**Review categories**", "", _bullets(q["rubric"]), "", q["interpretation"]]
    return "\n".join(lines)


def draft_corrections(spec: dict) -> str:
    return ("## Corrections needed in the experiment draft\n\n"
            "The questions can retain their current wording. These changes concern definitions and valid comparisons; the thesis source has not been edited.\n\n"
            + _table(["Location", "Required change"],
                     [[x["location"], x["change"]] for x in spec["required_draft_fixes"]]))


def implementation_plan(spec: dict) -> str:
    lines = ["## Results folder and implementation sequence", "",
             "The existing `reported_runs/results/` archive remains the source of consolidated saved runs. "
             "This analysis area adds shared calculations and a notebook for each RQ. "
             "Final metric building has not been run by this planning notebook.", "",
             _table(["Path", "Purpose", "State"], [[x["path"], x["purpose"], x["status"]] for x in spec["planned_files"]]), "",
             "### Build order", ""]
    lines.extend(f"{i}. {step}" for i, step in enumerate(spec["implementation_order"], 1))
    lines.extend(["", "### Derived tables", "", _bullets(spec["derived_tables"]), "",
                  "### One pattern for every metric section", "",
                  "1. State the subquestion and the outcome being examined.",
                  "2. Define the metric, denominator, valid cases, and comparison.",
                  "3. Show the plot and a numerical effect table from the same snapshot.",
                  "4. Add two or three sentences interpreting the pattern, consistency across seeds, and relevant limitations.",
                  "5. Link detailed diagnostics and source provenance within the notebook package.", "",
                  "No metric formulas or run-discovery logic should be copied into individual RQ notebooks.", "",
                  "### Consistent figures", ""])
    lines.append(_table(["Family", "Colour"], [[k, f"`{v}`"] for k, v in spec["plotting"]["family_colours"].items()]))
    lines.extend(["", _bullets(spec["plotting"][k] for k in ("scales", "domains", "seeds", "meaning", "exports")), "",
                  "### Checks before final results", "", _bullets(spec["verification_examples"]), "",
                  "### Current implementation references", "", _bullets(f"`{p}`" for p in spec["evidence_paths"])])
    return "\n".join(lines)


def latex_table_plan(spec: dict) -> str:
    """Describe exports without treating a presentation helper as validation."""
    plan = spec["latex_table_exports"]
    rows = [[f"RQ {row['rq']}", "; ".join(f"`{name}`" for name in row["analysis_names"])]
            for row in plan["planned_tables"]]
    return ("## LaTeX tables from notebook values\n\n"
            "The numerical table displayed in a final RQ notebook will also be exported "
            "as a copyable LaTeX table. Display and export use the same checked table "
            "values, explicit column names, units, precision, and missing-value rules; "
            "the export does not recalculate metrics.\n\n"
            f"**Label convention:** `{plan['label_prefix']}<analysis_name>`, using a "
            "meaningful, explicit snake_case name. For example, "
            "`chap:results:tab:rq1_feedback_mean_effects`. "
            "The same analysis name is used for the exported filenames.\n\n"
            f"**Files:** `{plan['output_directory']}/<analysis_name>.tex`, plus a "
            "CSV of the underlying values and a provenance JSON record containing "
            "the formatted table values. The final RQ "
            "notebooks will place these inside their frozen analysis snapshot.\n\n"
            + _table(["Notebook", "Planned analysis names"], rows)
            + "\n\nUndefined values remain undefined and display as an em dash. "
            "Captions identify the comparison, time summary, units, and seed/population "
            "scope; provenance records the source snapshot and selected runs when "
            "results exist. Generating LaTeX prevents transcription differences, "
            "but does not replace checking the source selection, denominators, "
            "calculations, or final thesis layout.\n\n"
            "### Working example: configured study design\n\n"
            "The example below exports the existing configured study counts from "
            "this planning specification. These are design values, **not completion "
            "counts or numerical findings**. No run artifacts are changed.")


def study_design_frame(spec: dict):
    """Return a small real design table for the offline export demonstration."""
    import pandas as pd

    return pd.DataFrame([
        {
            "Study": study["name"],
            "Simulation/population seeds": "; ".join(
                f"{seed}/{population}" for seed, population in study["seed_population_pairs"]),
            "Active/reserve users": f"{study['active']}/{study['reserve']}",
            "Configured logical cells": study["expected_logical_cells"],
        }
        for study in spec["studies"]
    ])
