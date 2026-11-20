"""Readable notebook sections for the two Goodreads research questions.

Selection and presentation helpers only: the scientific metric definitions live
in metrics.py, and matched effects live in comparisons.py. Source trajectories
and original experiment artifacts are never changed by this module.
"""
from __future__ import annotations

import pandas as pd

from .comparisons import build_comparison_plan, paired_effects, summarise_effects


MAIN_ML = "movielens_feedback_a300_e804"
MAIN_GR = "goodreads_gender_a300_e804"
GENDER_MITIGATION = "goodreads_gender_mitigation_a300_e804"
POPULARITY = ["head_click_share", "head_exposure", "weighted_head_exposure"]
HEAD_GAPS = ["click_exposure_gap", "weighted_click_exposure_gap"]
GENDER_ALL = ["female_click_share", "female_exposure", "weighted_female_exposure"]
GENDER_BINARY = ["female_among_binary_clicks", "female_among_binary_exposure", "weighted_female_among_binary_exposure"]
GENDER_GAPS = ["female_click_exposure_gap", "weighted_female_click_exposure_gap",
               "binary_female_click_exposure_gap", "weighted_binary_female_click_exposure_gap"]
COVERAGE = ["unknown_exposure", "weighted_unknown_exposure", "unknown_click_share",
            "ambiguous_exposure", "weighted_ambiguous_exposure", "ambiguous_click_share"]


def transfer_effects(frame):
    """The specific cross-dataset contrasts stated for RQ VI/VII, not every stage."""
    selected = frame.study.isin([MAIN_ML, MAIN_GR])
    added_feedback = frame.comparison.eq("L2 minus L0")
    added_individual = frame.comparison.eq("L3 full reflection minus L2 frozen agent")
    return frame.loc[selected & (added_feedback | added_individual)].copy()


def transfer_plot_labels(frame):
    """Keep the dataset and exact treatment-minus-control contrast on the axis."""
    return frame.assign(dataset_comparison=lambda selected:
        selected.dataset.map({"movielens": "MovieLens", "goodreads": "Goodreads"})
        + "\n" + selected.comparison)


def final_goodreads_conditions(frame):
    """One unmitigated displayed trajectory per response model and seed."""
    selected = frame.dataset.eq("goodreads") & frame.study.eq(MAIN_GR)
    computational_or_direct = frame.stage.eq("L2_ml_model") & frame.model.ne("agentic_recommender")
    reflective = frame.stage.eq("L3_individual") & frame.user_update_policy.eq("full_reflection")
    return frame.loc[selected & (computational_or_direct | reflective)].copy()


def reference_table(frame):
    columns = [column for column in ["dataset", "seed", "population_seed", "catalogue_size", "head_count",
        "head_catalogue_share", "initial_training_head_share", "catalogue_female_share", "initial_training_female_share",
        "catalogue_unknown_share", "catalogue_ambiguous_share", "label_policy", "labels_sha256"] if column in frame]
    return frame[columns].drop_duplicates(["dataset", "seed", "population_seed"]).sort_values(["dataset", "seed"]).reset_index(drop=True)


def computational_feedback_decomposition(trajectory, sources, metrics=None):
    """Use the extension's completed unmitigated L0/L1/L2 controls as matched pairs.

    Re-labelling only this derived selection gives the common comparison planner
    its feedback-study branch. No simulation row, source ID, or condition changes.
    The extension's logical aliases remain one physical trajectory in inventory.
    """
    study = "goodreads_gender_computational_decomposition"
    values = trajectory.loc[trajectory.study.eq(GENDER_MITIGATION) & trajectory.intervention.eq("control")].copy()
    selected = sources.loc[sources.study.eq(GENDER_MITIGATION) & sources.intervention.eq("control")].copy()
    values["study"], selected["study"] = study, study
    plan = build_comparison_plan(selected)
    if plan.empty:
        return pd.DataFrame()
    plan = plan.loc[plan.kind.eq("feedback")]
    return summarise_effects(paired_effects(values, plan, metrics or POPULARITY))


def gender_mitigation_effects(frame, *, include_method=False):
    kinds = ["mitigation", "mitigation_method"] if include_method else ["mitigation"]
    return frame.loc[frame.study.eq(GENDER_MITIGATION) & frame.kind.isin(kinds)].copy()


def notebook_sections():
    return {
        "06_goodreads_popularity.ipynb": [
            ("markdown", "# RQ VI — Popularity findings in Goodreads\n\n"
             "**Question:** How well do findings on popularity bias generalise to the Goodreads dataset for selected conditions?\n\n"
             "Compare matched *changes*, alongside the underlying trajectories. L2 minus L0 covers all five response models, including the frozen agent, for seeds 42, 43, and 44. The separate full-reflection L3 minus frozen-agent L2 contrast retains the individual-feedback comparison. Each contrast is paired within its own dataset and population; the same simulation-seed number can identify different reader populations across datasets.\n\n"
             "All trajectories use canonical outcomes. Exposure is also reported with the agreed one-based rank discount. Means require all 30 iteration values to be defined; the endpoint is iteration 30. No significance test treats users or iterations as independent runs."),
            ("code", "from reported_runs.analysis.sections_rq6_7 import (POPULARITY, HEAD_GAPS, transfer_effects, transfer_plot_labels, final_goodreads_conditions, reference_table, computational_feedback_decomposition)\n"
             "rq_sources = sources.loc[sources.study.isin(['movielens_feedback_a300_e804', 'goodreads_gender_a300_e804'])]\n"
             "report.availability(rq_sources)"),
            ("markdown", "## Frozen popularity references\n\n"
             "The head group is the fixed top 20% of catalogue items by initial training count, with the recorded ceiling and item-ID tie-break. Goodreads references concern the selected reader cohort; MovieLens references concern the full prepared historical population. The baseline head mass is descriptive, not a common target across datasets."),
            ("code", "references = results['references'].loc[results['references'].study.isin(rq_sources.study.unique())]\n"
             "context = reference_table(references)\n"
             "report.table(context, 'rq6_popularity_references', 'Frozen dataset-specific popularity references. Catalogue and initial training-history denominators are retained separately for each seed and population.', columns=['dataset', 'seed', 'population_seed', 'catalogue_size', 'head_count', 'head_catalogue_share', 'initial_training_head_share'])"),
            ("markdown", "## Head cut-off ties\n\n"
             "Equal initial training counts can span the top-20% boundary, where item IDs decide which tied items enter the original head. "
             "The strict alternative includes every item strictly above the original cut-off and excludes every item at it. "
             "It therefore has no ID-dependent boundary membership, but its catalogue share and minimum training count differ between populations. "
             "This small reference table can support a results or discussion limitation; it audits group membership rather than recomputing the trajectory effects below. "
             "Matched comparisons within a frozen population remain well-defined."),
            ("code", "from reported_runs.analysis.head_ties import load_goodreads_head_ties\n"
             "from reported_runs.analysis.head_tie_views import head_tie_table, TABLE_NAME, TABLE_CAPTION, TABLE_COLUMN_FORMAT, SUMMARY_EXPORT_NAME, ITEMS_EXPORT_NAME\n\n"
             "head_summary, strict_head_items = load_goodreads_head_ties(rq_sources, repo)\n"
             "head_numeric_csv = report.exports / 'data' / SUMMARY_EXPORT_NAME\n"
             "head_items_csv = report.exports / 'data' / ITEMS_EXPORT_NAME\n"
             "head_numeric_csv.parent.mkdir(parents=True, exist_ok=True)\n"
             "head_summary.to_csv(head_numeric_csv, index=False)\n"
             "strict_head_items.to_csv(head_items_csv, index=False)\n"
             "report.table(head_tie_table(head_summary), TABLE_NAME, TABLE_CAPTION, source_frame=rq_sources[rq_sources.dataset.eq('goodreads')], column_format=TABLE_COLUMN_FORMAT)\n"
             "display(Markdown(f'Full-precision counts and reference hashes: [numeric CSV]({head_numeric_csv.as_posix()}). Strict-head item IDs and their initial training counts: [item CSV]({head_items_csv.as_posix()}).'))"),
            ("markdown", "## Goodreads trajectories\n\n"
             "This view selects L2 for the computational models and Direct LLM, and full-reflection L3 for the agent. It provides one clearly identified trajectory per model and seed. The matched effect analysis below uses the appropriate control for each configuration."),
            ("code", "goodreads = final_goodreads_conditions(trajectory)\n"
             "figure = trajectory_grid(goodreads, POPULARITY, title='Goodreads popularity trajectories', scales=scales)\n"
             "report.figure(figure, goodreads, 'rq6_goodreads_popularity_trajectories', 'Canonical Goodreads head-click share, Head exposure, and Weighted head exposure. Computational models and Direct LLM use L2; the reflective agent uses L3. Separate columns identify simulation seeds 42, 43, and 44.')"),
            ("code", "gr_summaries = summaries.loc[summaries.study.eq('goodreads_gender_a300_e804')]\n"
             "table = temporal_table(gr_summaries, POPULARITY)\n"
             "report.table(table, 'rq6_goodreads_popularity_summaries', 'Goodreads popularity composition for every selected unmitigated condition. Values are shares. Time summaries are the strict mean over iterations 1--30 and the iteration-30 endpoint; each model, condition, and seed remains visible.')"),
            ("markdown", "## Matched changes across datasets\n\n"
             "L2 minus L0 compares all five response models, with the agent frozen at both stages. The separate agent contrast is L3 full reflection minus L2 frozen agent. Inspect the sign, magnitude, and consistency across seed/population blocks. Changes that differ between datasets show a limit to transfer within this particular setup; they do not isolate a dataset property or prompt-text effect."),
            ("code", "transfer = transfer_effects(effect_summaries)\n"
             "mean_transfer = transfer.loc[transfer.window.eq('mean_1_30')]\n"
             "figure = effect_grid(mean_transfer, POPULARITY, title='Mean matched popularity changes across datasets', scales=scales, facet='dataset', value='value')\n"
             "report.figure(figure, mean_transfer, 'rq6_cross_dataset_popularity_effects', 'Matched mean popularity effects in MovieLens and Goodreads, in percentage points. All five models use L2 minus L0, with the agent frozen at both stages; the separate agent contrast is L3 full reflection minus L2 frozen agent. Symbols identify simulation seeds, while dataset-specific population IDs remain in the accompanying table.', recommended=True, why='It answers the transfer question directly through comparable matched changes, while retaining model colours and separate seed observations.')"),
            ("code", "table = effect_table(transfer, 'head_click_share')\n"
             "report.table(table, 'rq6_cross_dataset_head_click_effects', 'Signed matched Head-click share differences, in percentage points, for both time summaries. Each row retains its dataset, model, simulation seed, population seed, and comparison.')"),
            ("code", "exposure_transfer = transfer_plot_labels(transfer)\n"
             "figure = effect_grid(exposure_transfer, ['head_exposure', 'weighted_head_exposure'], title='Cross-dataset exposure effects: mean and endpoint', scales=scales, category='dataset_comparison', facet='window', value='value')\n"
             "report.figure(figure, exposure_transfer, 'rq6_cross_dataset_exposure_mean_endpoint', 'Matched Head exposure and Weighted head exposure differences, in percentage points. Each axis category identifies both its dataset and its contrast: L2 minus L0 for all five models, with the agent frozen at both stages, or the separate L3 full reflection minus L2 frozen-agent contrast. Mean and endpoint panels share scales; colours identify response models and markers identify seeds.')"),
            ("markdown", "## Selection relative to exposure\n\n"
             "A change in Head-click share may accompany a change in what was displayed. The two descriptive click-exposure gaps distinguish click composition from unweighted and rank-weighted exposure composition. They are differences in shares, not click-through rates."),
            ("code", "figure = trajectory_grid(goodreads, HEAD_GAPS, title='Goodreads click-exposure gaps', scales=scales)\n"
             "report.figure(figure, goodreads, 'rq6_goodreads_click_exposure_gaps', 'Canonical Head-click share minus Head exposure, and minus Weighted head exposure. Signed share differences retain a zero reference and the same comparison scales across seeds.')"),
            ("code", "figure = effect_grid(mean_transfer, HEAD_GAPS, title='Mean matched gap changes across datasets', scales=scales, facet='dataset', value='value')\n"
             "report.figure(figure, mean_transfer, 'rq6_cross_dataset_gap_effects', 'Matched changes in both click-exposure gaps, in percentage points, averaged over iterations 1--30 within each run before comparing datasets.')"),
            ("markdown", "## Clicking, coverage, and concentration\n\n"
             "Read click composition alongside the valid-click count, click rate, and no-click rate. Catalogue coverage and Gini use the complete local catalogue, including items with zero exposure or zero clicks; they are descriptive within their dataset-specific catalogues."),
            ("code", "figure = trajectory_grid(goodreads, ['click_rate', 'no_click_rate', 'valid_clicks'], title='Goodreads clicking behaviour', scales=scales)\n"
             "report.figure(figure, goodreads, 'rq6_goodreads_click_behaviour', 'Valid-response click and no-click rates, and valid click counts, for the selected Goodreads configurations. Non-valid responses are excluded from behavioural-rate denominators and remain in the source summaries.')"),
            ("code", "figure = trajectory_grid(goodreads, ['exposure_coverage', 'click_coverage', 'exposure_gini', 'click_gini'], title='Goodreads coverage and concentration', scales=scales)\n"
             "report.figure(figure, goodreads, 'rq6_goodreads_coverage_concentration', 'Catalogue coverage and normalised Gini, reported separately for canonical exposure and valid clicks. All catalogue items, including zero-count items, enter the Gini calculation.')"),
            ("code", "table = temporal_table(gr_summaries, ['click_rate', 'no_click_rate', 'valid_clicks'])\n"
             "report.table(table, 'rq6_goodreads_click_behaviour_summaries', 'Goodreads valid-response click and no-click rates, and valid click counts. Strict mean and endpoint summaries are retained per unmitigated model, condition, and seed.')"),
            ("code", "table = temporal_table(gr_summaries, ['exposure_coverage', 'click_coverage', 'exposure_gini', 'click_gini'])\n"
             "report.table(table, 'rq6_goodreads_coverage_concentration_summaries', 'Goodreads catalogue coverage and Gini for exposure and clicks, with each local catalogue retained as its own denominator.')"),
            ("markdown", "## Supplementary L1 decomposition\n\n"
             "The later computational extension adds unmitigated L1 controls from the same frozen starting states. These permit separate L1-minus-L0 and L2-minus-L1 comparisons for all three computational models and seeds. They supplement the selected L2-minus-L0 transfer contrast rather than add independent replications."),
            ("code", "decomposition = computational_feedback_decomposition(trajectory, sources)\n"
             "figure = effect_grid(decomposition, POPULARITY, title='Goodreads computational effects: L1 minus L0 and L2 minus L1', scales=scales, facet='window', value='value')\n"
             "report.figure(figure, decomposition, 'rq6_computational_l1_decomposition', 'Matched computational L1-minus-L0 and L2-minus-L1 popularity effects in Goodreads, from the extension controls. Effects are percentage-point differences within the same seed, population, model, and iteration.')"),
            ("code", "table = effect_table(decomposition, 'head_click_share')\n"
             "report.table(table, 'rq6_computational_l1_decomposition_table', 'Supplementary computational feedback decomposition for Head-click share, in percentage points. Reused control trajectories are retained as explicit memberships and are not additional replications.')"),
            ("markdown", "## Writing notes\n\n"
             "Start with whether the matched effects retain their direction across datasets, then discuss magnitude and consistency across the three seed/population blocks for each response model. Use the raw trajectories to explain when changes occur. Avoid attributing differences to one dataset or prompt feature without a controlled comparison."),
        ],
        "07_goodreads_author_gender.ipynb": [
            ("markdown", "# RQ VII — Author-gender representation and mitigation\n\n"
             "**Question:** To what extent do selected findings about user-response models, feedback mechanisms, and mitigation extend from popularity bias in MovieLens to author-gender representation in Goodreads?\n\n"
             "The labels concern each work's recorded primary literary author, under the agreed expanded VIAF rule. They are provisional identity-linked metadata, not reader demographics. Unknown and ambiguous items stay in the catalogue and histories. Representation changes alone do not establish discrimination or a fairness target.\n\n"
             "The unmitigated comparisons cover computational models, Direct LLM, and the reflective agent for seeds 42, 43, and 44. Gender-mitigation comparisons cover those three seeds for the computational models only."),
            ("code", "from reported_runs.analysis.sections_rq6_7 import (GENDER_ALL, GENDER_BINARY, GENDER_GAPS, COVERAGE, final_goodreads_conditions, transfer_effects, reference_table, gender_mitigation_effects)\n"
             "rq_sources = sources.loc[sources.dataset.eq('goodreads')]\n"
             "report.availability(rq_sources)\n"
             "goodreads = final_goodreads_conditions(trajectory)"),
            ("markdown", "## Labelled catalogue and historical context\n\n"
             "The local catalogue and initial training histories have different gender compositions. Retain their own denominators for every population. The following reference table documents composition; it does not prescribe equal representation or complete identification."),
            ("code", "references = results['references'].loc[results['references'].dataset.eq('goodreads')]\n"
             "context = reference_table(references)\n"
             "report.table(context, 'rq7_gender_reference_composition', 'Goodreads primary-author gender-label composition in each frozen catalogue and its initial training histories. Unknown and ambiguous annotations are retained. Values are shares.', columns=['seed', 'population_seed', 'catalogue_size', 'catalogue_female_share', 'initial_training_female_share', 'catalogue_unknown_share', 'catalogue_ambiguous_share'])"),
            ("markdown", "## Women-authored representation among all items\n\n"
             "These denominators contain every displayed slot or valid click, including unknown, ambiguous, and any other retained labels. The figure selects L2 for computational models and Direct LLM, and full-reflection L3 for the agent; matched effects below use each configuration's stated control."),
            ("code", "figure = trajectory_grid(goodreads, GENDER_ALL, title='Women-authored representation among all items', scales=scales)\n"
             "report.figure(figure, goodreads, 'rq7_gender_all_item_trajectories', 'Women-authored shares among canonical valid clicks, unweighted displayed slots, and rank-weighted displayed exposure. Denominators retain all label categories; seed panels remain separate.', recommended=True, why='The three linked outcomes show how representation in displayed items and selected items evolves without conditioning away unresolved labels.')"),
            ("code", "gr_summaries = summaries.loc[summaries.study.eq('goodreads_gender_a300_e804')]\n"
             "table = temporal_table(gr_summaries, GENDER_ALL)\n"
             "report.table(table, 'rq7_all_item_gender_summaries', 'Women-authored shares among all valid clicks and all displayed items, with unweighted and rank-weighted exposure. Mean and endpoint summaries retain each unmitigated condition and seed.')"),
            ("markdown", "## Women-plus-men denominators\n\n"
             "This companion conditions on women- and men-labelled items. It answers a different question from the all-item shares. Exposure pools qualifying displayed mass within each iteration; weighted exposure uses the same two-label scope in numerator and denominator. A missing qualifying denominator remains undefined."),
            ("code", "figure = trajectory_grid(goodreads, GENDER_BINARY, title='Women-authored representation within women- and men-labelled items', scales=scales)\n"
             "report.figure(figure, goodreads, 'rq7_gender_women_men_trajectories', 'Women-authored shares within women-plus-men-labelled valid clicks and displayed exposure. Unweighted and rank-weighted exposure retain matching label scopes. Unknown and ambiguous items are excluded only from this explicitly conditional companion.')"),
            ("code", "table = temporal_table(gr_summaries, GENDER_BINARY)\n"
             "report.table(table, 'rq7_women_men_gender_summaries', 'Women-authored shares conditional on women- plus men-labelled items, separately for valid clicks, unweighted exposure, and rank-weighted exposure. Zero qualifying denominators remain undefined.')"),
            ("markdown", "## Selection relative to exposure\n\n"
             "Subtract the corresponding women-authored exposure share from click share, with matching all-item or women-plus-men denominators. Only the exposure term receives rank weights. A positive gap means that the women-authored share of valid clicks exceeds the corresponding exposure share: for example, 40% of valid clicks minus 35% of exposure gives +5 percentage points. A negative gap means the click share is lower than the exposure share. These descriptive gaps are neither conditional click probabilities nor causal estimates of bias."),
            ("code", "figure = trajectory_grid(goodreads, GENDER_GAPS, title='Gender click-exposure gaps', scales=scales)\n"
             "report.figure(figure, goodreads, 'rq7_gender_click_exposure_gaps', 'Gender click-exposure gaps for all items and the women-plus-men subset, with unweighted and rank-weighted exposure. Signed differences use a common zero reference and consistent share scales.')"),
            ("code", "table = temporal_table(gr_summaries, GENDER_GAPS)\n"
             "report.table(table, 'rq7_gender_click_exposure_gap_summaries', 'Women-authored click share minus its corresponding exposure share, in percentage points. Each label scope and rank-weighting choice remains separate.')"),
            ("markdown", "## Matched feedback and response-model comparisons\n\n"
             "L2 minus L0 compares all five response models, with the agent frozen at both stages. The separate agent comparison is full-reflection L3 minus frozen-agent L2. Read model differences alongside the same configuration's exposure, candidate availability, and annotation coverage."),
            ("code", "gender_effects = transfer_effects(effect_summaries)\n"
             "gender_effects = gender_effects.loc[gender_effects.dataset.eq('goodreads')]\n"
             "figure = effect_grid(gender_effects, GENDER_ALL, title='Matched gender-representation changes', scales=scales, facet='window', value='value')\n"
             "report.figure(figure, gender_effects, 'rq7_matched_gender_feedback_effects', 'Matched Goodreads changes in women-authored click share and exposure, in percentage points. L2 minus L0 compares all five models, with the agent frozen at both stages; the separate agent contrast is L3 full reflection minus L2 frozen agent. Mean and endpoint effects retain individual seed observations.')"),
            ("code", "table = effect_table(gender_effects, 'female_click_share')\n"
             "report.table(table, 'rq7_matched_women_authored_click_effects', 'Signed matched changes in women-authored share among all valid clicks, in percentage points. Comparisons and both time summaries remain explicit.')"),
            ("markdown", "## Candidate availability before each canonical decision\n\n"
             "Eligibility starts from the frozen catalogue minus the reader's historical and earlier canonical-consumed works. Candidate shares are measured before the current response, then averaged across opportunities. Displayed women-authored exposure minus this availability share provides a descriptive baseline; it is not a target-parity claim."),
            ("code", "candidate_metrics = ['eligible_female_share', 'female_exposure_minus_candidate_share', 'weighted_female_exposure_minus_candidate_share']\n"
             "figure = trajectory_grid(goodreads, candidate_metrics, title='Women-authored exposure relative to candidate availability', scales=scales)\n"
             "report.figure(figure, goodreads, 'rq7_gender_candidate_availability', 'Women-authored candidate availability before canonical decisions, with unweighted and rank-weighted women-authored exposure minus that availability share. Exclusion histories follow canonical clicks only.')"),
            ("code", "table = temporal_table(gr_summaries, candidate_metrics)\n"
             "report.table(table, 'rq7_candidate_availability_summaries', 'Women-authored candidate shares and signed exposure-minus-candidate-share differences. Candidate shares are fractions; the labelled differences are percentage points. Availability is evaluated before each canonical decision, retaining population-specific catalogues.')"),
            ("markdown", "## Unknown and ambiguous label coverage\n\n"
             "A reduction in unknown exposure can change women-authored shares even without a corresponding shift among identified women- and men-labelled works. Ambiguous labels must also remain separate from unknown labels. These panels accompany interpretation of the primary gender plots."),
            ("code", "figure = trajectory_grid(goodreads, COVERAGE[:3], title='Unknown-label exposure and click composition', scales=scales)\n"
             "report.figure(figure, goodreads, 'rq7_unknown_label_coverage', 'Unknown-label shares among unweighted exposure, rank-weighted exposure, and valid clicks. All-item denominators are retained consistently across models and seeds.')"),
            ("code", "figure = trajectory_grid(goodreads, COVERAGE[3:], title='Ambiguous-label exposure and click composition', scales=scales)\n"
             "report.figure(figure, goodreads, 'rq7_ambiguous_label_coverage', 'Ambiguous-label shares among unweighted exposure, rank-weighted exposure, and valid clicks. Ambiguous and unknown annotations remain distinct.')"),
            ("code", "table = temporal_table(gr_summaries, COVERAGE)\n"
             "report.table(table, 'rq7_unresolved_label_coverage_summaries', 'Unknown and ambiguous label coverage in unweighted exposure, rank-weighted exposure, and valid clicks. These coverage measures accompany both all-item and women-plus-men interpretations.')"),
            ("markdown", "## Clicking behaviour\n\n"
             "Composition can become unstable when few users click. Valid click counts, click rates, and no-click rates provide that denominator context. A configuration without valid clicks has undefined click composition and gaps, rather than a zero share."),
            ("code", "figure = trajectory_grid(goodreads, ['click_rate', 'no_click_rate', 'valid_clicks'], title='Goodreads clicking context for gender representation', scales=scales)\n"
             "report.figure(figure, goodreads, 'rq7_click_behaviour', 'Valid-response click and no-click rates, and canonical valid-click counts. Undefined composition is retained when no valid clicks occur.')"),
            ("code", "table = temporal_table(gr_summaries, ['click_rate', 'no_click_rate', 'valid_clicks'])\n"
             "report.table(table, 'rq7_click_behaviour_summaries', 'Goodreads clicking context for author-gender representation: valid-response click and no-click rates, and valid-click counts, per unmitigated condition and seed.')"),
            ("markdown", "## Computational gender mitigation\n\n"
             "The extension retains the earlier cohort, item mapping, initial histories, response-model artifacts, and initial SASRec checkpoint. In-processing weights women-authored positive targets by 1.25 during subsequent refits; post-processing promotes women-authored items by five ranks inside the selected slate. Unknown and ambiguous labels retain their original treatment.\n\n"
             "Each treatment is compared with its own unmitigated control at L0, L1, or L2, for the same model, seed, population, and iteration. In-processing aliases at stages without refits are intentionally identical controls, not extra replications or newly trained models."),
            ("code", "mitigation = gender_mitigation_effects(effect_summaries)\n"
             "mitigation_l2 = mitigation.loc[mitigation.stage.eq('L2_ml_model')]\n"
             "figure = effect_grid(mitigation_l2, GENDER_ALL, title='L2 gender-mitigation effects', scales=scales, facet='window', value='value')\n"
             "report.figure(figure, mitigation_l2, 'rq7_gender_mitigation_l2_effects', 'Computational L2 gender-mitigation-minus-control changes in women-authored click share and exposure, in percentage points. Each method uses the unmitigated control for the same model, population, and seed.', recommended=True, why='This directly shows the effects of the tested interventions on both exposure and selection, with all three computational seeds visible.')"),
            ("code", "mitigation_mean = mitigation.loc[mitigation.window.eq('mean_1_30')].assign(\n"
             "    method=lambda frame: frame.intervention.map({'in_processing': 'In-processing', 'post_processing': 'Post-processing'}))\n"
             "figure = effect_grid(mitigation_mean, ['female_click_share', 'weighted_female_exposure'], title='Mean gender-mitigation effects across L0, L1, and L2', scales=scales, category='method', facet='stage', value='value')\n"
             "report.figure(figure, mitigation_mean, 'rq7_gender_mitigation_all_stages', 'Mean computational gender-mitigation effects, in percentage points. Each panel identifies its feedback stage; the two categories identify the intervention. Each intervention is compared with its own unmitigated control at the same stage, model, population, and seed. In-processing at L0 and L1, where no refits occur, is an unchanged-control alias. Stage panels use matching scales and retain individual seeds.')"),
            ("markdown", "## How mitigation changes click-exposure gaps\n\n"
             "The following effect is the mitigation gap minus the matched control gap. A positive effect therefore means that mitigation increased the gap; it does not establish that the absolute gap is positive. For example, moving from a control gap of -10 to a mitigation gap of -5 percentage points gives a +5-percentage-point effect while both absolute gaps remain negative. Read these effects alongside the absolute gap trajectories above."),
            ("code", "figure = effect_grid(mitigation_l2, ['female_click_exposure_gap', 'weighted_female_click_exposure_gap'], title='L2 gender-mitigation effects on click-exposure gaps', scales=scales, facet='window', value='value')\n"
             "report.figure(figure, mitigation_l2, 'rq7_gender_mitigation_gap_effects', 'Matched L2 mitigation-minus-control changes in gender click-exposure gaps, in percentage points, for unweighted and rank-weighted exposure. Positive values mean an increased gap, which may still be negative in absolute terms. Only the exposure component receives rank weights.')"),
            ("code", "table = effect_table(mitigation, 'female_click_share')\n"
             "report.table(table, 'rq7_gender_mitigation_click_effects', 'Women-authored click-share changes for every computational gender-mitigation stage, method, model, and seed, in percentage points. Zero in-processing effects at stages without refits are recorded logical aliases, not new simulation evidence.')"),
            ("code", "method_effects = gender_mitigation_effects(effect_summaries, include_method=True)\n"
             "method_effects = method_effects.loc[method_effects.kind.eq('mitigation_method')]\n"
             "method_effects_l2 = method_effects.loc[method_effects.stage.eq('L2_ml_model')]\n"
             "figure = effect_grid(method_effects_l2, GENDER_ALL, title='L2 in-processing minus post-processing', scales=scales, facet='window', value='value')\n"
             "report.figure(figure, method_effects_l2, 'rq7_gender_in_versus_post_processing', 'Computational L2 in-processing-minus-post-processing differences, in percentage points, paired within each model, population, seed, and iteration. Positive values indicate greater women-authored representation under in-processing for the stated outcome.')"),
            ("code", "table = effect_table(method_effects, 'female_click_share')\n"
             "report.table(table, 'rq7_gender_in_versus_post_processing_table', 'Matched in-processing-minus-post-processing women-authored click-share differences, in percentage points, for all computational stages and seeds.')"),
            ("markdown", "## Coverage and clicking checks for mitigation\n\n"
             "The intervention may change exposure to unresolved labels or change the number of clicks. These checks retain all computational L2 configurations. Line styles identify the intervention; the fixed model hues retain their meaning. Interpret the gender effects alongside these denominator changes."),
            ("code", "mitigation_trajectories = trajectory.loc[trajectory.study.eq('goodreads_gender_mitigation_a300_e804') & trajectory.stage.eq('L2_ml_model')]\n"
             "figure = trajectory_grid(mitigation_trajectories, ['weighted_unknown_exposure', 'weighted_ambiguous_exposure', 'click_rate'], title='Gender-mitigation annotation and clicking context', scales=scales, series='condition_label')\n"
             "report.figure(figure, mitigation_trajectories, 'rq7_gender_mitigation_coverage_click_checks', 'Computational L2 gender-mitigation conditions, showing rank-weighted unknown and ambiguous exposure shares and valid-response click rate. Model colours and intervention line styles retain distinct meanings.')"),
            ("markdown", "## Writing notes\n\n"
             "Report the direction and magnitude of representation changes, then check whether candidate availability and unknown or ambiguous coverage change the interpretation. Discuss the women-plus-men companion as conditional evidence. Compare mitigation methods within the same computational configuration, and keep those comparisons separate from the unmitigated LLM results. Manual gender identity verification and a validated fairness target are outside what these simulations establish."),
        ],
    }
