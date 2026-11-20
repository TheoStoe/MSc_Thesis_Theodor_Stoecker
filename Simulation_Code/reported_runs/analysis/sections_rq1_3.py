"""Readable notebook cells for RQ I--III; shared aggregation stays outside notebooks."""
from __future__ import annotations

from html import escape

import pandas as pd
from IPython.display import HTML, Markdown, display

from .comparisons import RUN_METADATA, summarise_iterations
from .plotting import MODEL_LABELS, trajectory_grid
from .metrics import ratio
from .presentation import label_conditions, temporal_table


def diagnostic_table(frame, metrics, *, extra_ids=()):
    """Mean/endpoint diagnostics with the same strict 30-iteration rule."""
    if frame.empty:
        return pd.DataFrame(columns=["Condition", "Seed", "Time summary", *metrics])
    metrics = [metric for metric in metrics if metric in frame]
    ids = [column for column in RUN_METADATA if column in frame] + list(extra_ids)
    summary = summarise_iterations(frame, metrics, id_columns=ids)
    groups = [column for column in ["model", "stage", "user_update_policy", "condition_id", "seed", *extra_ids, "window"] if column in summary]
    table = temporal_table(summary, metrics, group_columns=groups)
    table = label_conditions(table)
    table["model"] = table.model.map(MODEL_LABELS).fillna(table.model)
    return table.drop(columns=[c for c in ["stage", "user_update_policy", "condition_id"] if c in table]).rename(
        columns={"model": "Model", "condition_label": "Condition", "seed": "Seed", "window": "Time summary"})


def horizon_counts(frame, metrics, *, extra_ids=()):
    """Sum actual event counts over complete 1--30 horizons, never proportions."""
    if frame.empty:
        return pd.DataFrame(columns=["Condition", "Seed", *metrics])
    ids = [column for column in ["model", "stage", "user_update_policy", "condition_id", "seed", *extra_ids] if column in frame]
    metrics = [metric for metric in metrics if metric in frame]
    rows = []
    for key, block in frame.groupby(ids, sort=False, dropna=False):
        if sorted(block.iteration.tolist()) != list(range(1, 31)):
            raise ValueError("Horizon counts require exactly one row for each iteration 1--30.")
        rows.append({**dict(zip(ids, key)), **block[metrics].sum(min_count=30).to_dict()})
    table = label_conditions(pd.DataFrame(rows))
    table["model"] = table.model.map(MODEL_LABELS).fillna(table.model)
    return table.drop(columns=[c for c in ["stage", "user_update_policy", "condition_id"] if c in table]).rename(
        columns={"model": "Model", "condition_label": "Condition", "seed": "Seed"})


def diagnostic_tables(report, frame, groups, *, prefix, context, extra_ids=()):
    """Small, consistently labelled tables instead of an unreadable wide table."""
    for suffix, metrics in groups.items():
        table = diagnostic_table(frame, metrics, extra_ids=extra_ids)
        report.table(table, f"{prefix}_{suffix}", "")


def reflection_totals(frame):
    """Full-horizon request/outcome counts and failed/requested update proportion."""
    table = horizon_counts(frame, ["reflection_requested", "reflection_applied", "reflection_failed"])
    table["Failed / requested updates"] = [ratio(failed, requested) for failed, requested in
                                            zip(table.reflection_failed, table.reflection_requested)]
    return table


def agent_usage_summary(frame, *, by_condition=False):
    """Pool recorded counts by protocol/arm, retaining unknown field coverage."""
    scopes = {
        "movielens_feedback_a300_e804": "RQ III: 300 active users (L2/L3)",
        "movielens_sampling_outcome_a150_r150_e804": "150 active / 150 reserve protocol",
    }
    keys = ["study", "arm"]
    if by_condition:
        keys += ["seed", "model", "stage", "user_update_policy", "condition_id"]
    counts = ["decisions", "memory_calls", "decisions_with_memory", "forced_actions",
              "forced_flag_missing", "forced_flag_malformed", "trace_missing", "trace_malformed"]
    rows = []
    for values, block in frame.groupby(keys, sort=False, dropna=False):
        if block.source_id.duplicated().any():
            raise ValueError("Usage totals must contain only one row per physical run and arm")
        row = dict(zip(keys, values))
        row["Runs"] = block.source_id.nunique()
        row.update({field: block[field].sum(min_count=len(block)) for field in counts})
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["Protocol", "Presentation arm", "Decisions", "Memory calls", "Forced actions"])
    table = pd.DataFrame(rows)
    if by_condition:
        table = label_conditions(table).drop(columns=["model", "stage", "user_update_policy", "condition_id"])
    table["study"] = table.study.map(scopes).fillna(table.study)
    return table.rename(columns={"study": "Protocol", "arm": "Presentation arm", "seed": "Seed",
        "condition_label": "Condition", "decisions": "Decisions", "memory_calls": "Memory calls",
        "decisions_with_memory": "Decisions using memory", "forced_actions": "Forced actions",
        "forced_flag_missing": "Missing forced-action flags", "forced_flag_malformed": "Malformed forced-action flags",
        "trace_missing": "Missing module traces", "trace_malformed": "Malformed module traces"})


def arm_rank_statistics(frame):
    """One per-iteration rank-statistic row per arm, without repeating K rows.

    The metric engine already provides each arm's mean clicked rank. Its rank-1
    click share is the first-position share. Select these saved values rather
    than reimplementing a rank formula in the notebook.
    """
    if frame.empty:
        return frame.copy()
    output = frame.loc[frame["rank"].eq(1)].copy()
    output["first_position_click_share"] = output["click_share"]
    return output


def advertising_display(frame):
    """Mark campaign eligibility in the no-campaign arm as not applicable.

    Saved metrics are preserved. The organic arm still has genuine zero
    sponsored exposures/clicks; only its campaign-eligibility view is missing,
    because it has no campaign whose delivery can be continued or withdrawn.
    """
    output = frame.copy()
    organic = output.condition_id.eq("O0_organic")
    fields = [column for column in ["advertising_eligible_users", "advertising_eligibility_share"] if column in output]
    output.loc[organic, fields] = float("nan")
    return output


def model_trajectories(report, frame, metrics, *, prefix, context):
    """Separate response families; preserve model colours and seed facets."""
    for model in frame.model.drop_duplicates():
        selected = frame.loc[frame.model.eq(model)]
        figure = trajectory_grid(selected, metrics, title=f"{context}: {MODEL_LABELS.get(model, model)}",
                                 scales=report.scales, series="condition_label")
        report.figure(figure, selected, f"{prefix}_{model}", "")


def profile_review_details(report, cases, changes):
    """Report the pinned numeric sample; source-bearing evidence stays private."""
    from .packaged import derived_frame
    assessed = derived_frame("profile_assessments")
    report.table(assessed, "rq3_profile_review_saved_categories", "", columns=["case_id", "seed", "review_order", "applied_change_count", "review_status", "extent_of_change", "support_in_available_evidence"])



def _md(text):
    return ("markdown", text.strip())


def _code(text):
    return ("code", text.strip())


def notebook_sections():
    return {
        "01_feedback_mechanisms.ipynb": [
            _md("""## Matched feedback comparisons

**RQ I:** How do popularity bias trajectories change as additional feedback mechanisms are introduced?

Each effect adds one mechanism relative to its stated control. The main comparisons are L1 minus L0, L2 minus L1, and full-reflection L3 minus frozen-agent L2. Sampling uses the separate 150-active/150-reserve protocol; its reference is its own matched L3. Advertising compares static sponsorship with its candidate-restricted organic control, then adaptive L5 delivery with static delivery.

Positive exposure effects mean a larger initial-head share. These are descriptive seed/population contrasts, without treating users or iterations as additional independent replications."""),
            _code("""from reported_runs.analysis.sections_rq1_3 import diagnostic_tables, horizon_counts, model_trajectories, advertising_display
feedback = trajectory.loc[trajectory.study.eq('movielens_feedback_a300_e804') & (~trajectory.stage.str.startswith('L3') | trajectory.user_update_policy.eq('full_reflection'))].copy()
feedback_effects = effect_summaries.loc[effect_summaries.study.eq('movielens_feedback_a300_e804') & effect_summaries.kind.isin(['feedback', 'individual'])].copy()
report.table(results['comparison_plan'].loc[results['comparison_plan'].study.str.startswith('movielens_') & results['comparison_plan'].kind.isin(['feedback', 'individual', 'sampling', 'advertising'])], 'rq1_matched_comparisons', 'Exact matched comparisons and source availability; only eligible complete pairs contribute effects.')"""),
            _md("""## Primary exposure effects

The two panels use unweighted and rank-weighted displayed head exposure. Each point retains its simulation/population seed; the mean uses equal weight for all 30 iteration values. The endpoint remains separate because an average effect and a final effect may differ."""),
            _code("""figure = effect_grid(feedback_effects, ['head_exposure', 'weighted_head_exposure'], title='Added feedback mechanisms: matched exposure effects', scales=scales, value='value', facet='window')
report.figure(figure, feedback_effects, 'rq1_feedback_exposure_effects', 'Later-stage minus matched-control head exposure: L1 minus L0, L2 minus L1, and full-reflection L3 minus frozen-agent L2. Positive percentage-point differences mean higher head exposure. Mean-over-iterations and iteration-30 effects are calculated within each seed/population block.', recommended=True, why='This directly isolates each added mechanism, shows every seed, and keeps the two exposure definitions separate.')
report.table(effect_table(feedback_effects, 'head_exposure'), 'rq1_feedback_head_exposure', 'Matched head-exposure effects by seed and time summary, in percentage points.')
report.table(effect_table(feedback_effects, 'weighted_head_exposure'), 'rq1_feedback_weighted_head_exposure', 'Matched weighted-head-exposure effects by seed and time summary, in percentage points.')"""),
            _code("""model_trajectories(report, feedback, ['head_exposure', 'weighted_head_exposure'], prefix='rq1_feedback_trajectory', context='Feedback-stage exposure trajectories')"""),
            _md("""## Selection relative to exposure, click behaviour, and coverage

Click-exposure gaps subtract displayed head exposure from head-click share. They describe the composition of clicks relative to exposure; they are not a head-item click-through rate. Click rate uses valid click/no-click decisions, while invalid responses are retained separately. Coverage and Gini concern the full frozen catalogue and are reported separately for shown and clicked items."""),
            _code("""diagnostic_tables(report, feedback, {
    'click_exposure': ['head_click_share', 'click_exposure_gap', 'weighted_click_exposure_gap'],
    'click_behaviour': ['click_rate', 'no_click_rate', 'invalid_rate', 'valid_clicks'],
    'coverage': ['exposure_coverage', 'click_coverage'],
    'concentration': ['exposure_gini', 'click_gini']}, prefix='rq1_feedback', context='MovieLens 300-active-user feedback conditions.')"""),
            _md("""## Sampling feedback: matched 150/150 comparisons

All three departure/refill variants use the matched 150-active/150-reserve L3 control. The 300-active L3 is a different protocol. Active counts, departures, replacements, and fixed preference-cluster composition help interpret changes caused by the evolving user population.

The active-user panels use this protocol's maximum of 150 active users. Departures and replacements use a separate 0--20 scale, shared across seeds, so their smaller counts remain visible. Each panel shows one seed: line style and marker shape identify the condition within that panel. Markers are not extra observations; every trajectory contains all 30 iterations. Cluster shares use the current active population as denominator, and sum to one within each iteration. A falling active count therefore does not reduce the height of a 100% composition bar."""),
            _code("""sampling = trajectory.loc[trajectory.study.eq('movielens_sampling_outcome_a150_r150_e804') & ~trajectory.condition_id.str.startswith('O')].copy()
sampling_effects = effect_summaries.loc[effect_summaries.kind.eq('sampling')].copy()
figure = effect_grid(sampling_effects, ['head_exposure', 'weighted_head_exposure'], title='Sampling feedback versus matched L3', scales=scales, value='value', facet='window')
report.figure(figure, sampling_effects, 'rq1_sampling_exposure_effects', 'Each L4 departure/refill condition minus the matched 150-active/150-reserve L3 reference. Positive percentage-point differences mean higher head exposure. The two time summaries describe the mean over iterations 1--30 and the iteration-30 endpoint.')
model_trajectories(report, sampling, ['head_exposure', 'weighted_head_exposure'], prefix='rq1_sampling_trajectory', context='Matched 150/150 sampling trajectories')"""),
            _code("""population = label_conditions(results['population'].loc[results['population'].study.eq('movielens_sampling_outcome_a150_r150_e804') & ~results['population'].condition_id.str.startswith('O')])
figure = trajectory_grid(population, ['active_users', 'departures', 'replacements'], title='Sampling feedback: participation and turnover', scales=scales, series='condition_label', scale_overrides={'active_users': (0, 150), 'departures': (0, 20), 'replacements': (0, 20)})
report.figure(figure, population, 'rq1_sampling_population', 'Active-user counts use a 0--150 scale; per-iteration departures and replacements use 0--20. Each metric retains the same scale across seed panels. Historical training logs remain those of all 300 users. Line styles and markers identify conditions within each seed panel.')
diagnostic_tables(report, population, {'population': ['active_users', 'departures', 'replacements']}, prefix='rq1_sampling', context='Participation and turnover under sampling feedback.')
report.table(horizon_counts(population, ['departures', 'replacements']), 'rq1_sampling_turnover_totals', 'Total departures and replacements over all 30 iterations; these are event counts, not counts of independent replicate users.')"""),
            _code("""clusters = label_conditions(results['clusters'].loc[results['clusters'].study.eq('movielens_sampling_outcome_a150_r150_e804') & ~results['clusters'].condition_id.str.startswith('O')])
for condition, selected in clusters.groupby('condition_label', sort=False):
    figure = trajectory_grid(selected, 'active_share', title=f'Preference-cluster composition: {condition}', scales=scales, series='cluster')
    report.figure(figure, selected, 'rq1_clusters_' + selected.condition_id.iloc[0].lower().replace('__', '_'), f'Fixed preference-cluster shares among currently active users under {condition}. These are separate cluster-share trajectories, not differences from another condition. The shares sum to one at every iteration; line styles and markers identify clusters.')
diagnostic_tables(report, clusters, {'cluster_composition': ['active_users', 'active_share']}, prefix='rq1_sampling', context='Active preference-cluster composition.', extra_ids=['cluster'])"""),
            _code("""from reported_runs.analysis.plotting import composition_grid
figure = composition_grid(clusters, title='Active preference-cluster composition: 100% stacked bars')
report.figure(figure, clusters, 'rq1_sampling_cluster_composition', 'Cluster shares among currently active users in the matched L3 reference and three L4 departure/refill conditions. Bars show iterations 1, 10, 20, and 30, not a pooled time average. Each bar totals 100%; rows identify conditions and columns identify seeds. Cluster colours/hatches have the same meaning across panels.')"""),
            _code("""diagnostic_tables(report, sampling, {
    'click_exposure': ['head_click_share', 'click_exposure_gap', 'weighted_click_exposure_gap'],
    'click_behaviour': ['click_rate', 'no_click_rate', 'valid_clicks'],
    'coverage': ['exposure_coverage', 'click_coverage'],
    'concentration': ['exposure_gini', 'click_gini']}, prefix='rq1_sampling', context='Matched 150/150 sampling conditions.')"""),
            _md("""## Advertising and outcome feedback

Static advertising is compared with its matched organic control, which shares candidate restrictions. Adaptive L5 delivery is compared with static advertising. Sponsored exposure/click counts and current-step advertising delivery indicate whether the intervention remains active, alongside the same head-exposure and selection outcomes.

**The advertising exposure companion uses displayed ranks 2–10 only.** It excludes rank 1 in every condition, including the organic control and decisions after advertising withdrawal. Unweighted exposure divides head items in these positions by their nine displayed slots. Weighted exposure retains the original discounts `d(r)=1/log2(r+1)` for ranks 2–10 and normalises over those nine discounts; the remaining positions are not renumbered.

The original exposure metrics still describe all displayed ranks 1–10, including any sponsored item. Reporting both scopes distinguishes the effect of counting rank 1 from changes among the other displayed positions. The companion for ranks 2–10 is a sensitivity analysis, not an isolated feedback effect: inserting an advertisement also shifts and truncates the organic recommendations. The separate table comparing the two exposure scopes describes rank-1 inclusion within the same saved slate, not a causal advertising-effect estimate.

The saved `advertising_eligible_users` field counts canonical slates that actually receive a sponsored item; it is not the platform's active-user count. Under the implemented static policy, every active user receives the scheduled sponsor, so the two counts coincide. For example, seed 42 has 139 active users and 139 sponsored deliveries at iteration 30. Under adaptive delivery, the active population can remain at 150 while advertising is withdrawn from many of those users.

The organic reference has **no advertising campaign**, so its campaign-eligibility count/share is displayed as not applicable rather than interpreted as users becoming ineligible. Its zero sponsored exposures and zero sponsored clicks are genuine and remain zero. The saved numeric records are unchanged. Advertising eligibility share divides current sponsored deliveries by active decision opportunities; sponsored exposure share divides sponsored slots by all displayed slots (one sponsored item in ten gives 0.1); sponsored-click share divides sponsored clicks by all valid clicks. These ratios have different denominators and should not be read as interchangeable rates."""),
            _code("""from reported_runs.analysis.advertising_slots import load_advertising_slot_metrics
from reported_runs.analysis.advertising_views import SLOT_METRICS, ACCOUNTING_LABELS, build_slot_views, slot_scales, slot_accounting_table
from reported_runs.analysis.sections_rq1_3 import diagnostic_table
from reported_runs.analysis.plotting import label
slot_sources = sources.loc[sources.study.eq('movielens_sampling_outcome_a150_r150_e804') & sources.condition_id.str.startswith('O')]
slot_metrics = load_advertising_slot_metrics(slot_sources, repo, repo / 'reported_runs/analysis/cache/advertising_slots')
slot_views = build_slot_views(slot_metrics, results['comparison_plan'])
slot_trajectory = label_conditions(slot_views['trajectory'])
slot_effects = slot_views['effect_summaries']
advertising_scales = slot_scales(scales, slot_views)
figure = effect_grid(slot_effects, SLOT_METRICS, title='Advertising exposure effects: displayed ranks 2–10 only', scales=advertising_scales, value='value', facet='window')
figure._analysis_spec['exposure_scope'] = 'Displayed ranks 2–10 only, in every condition; original rank discounts retained'
report.figure(figure, slot_effects, 'rq1_advertising_ranks_2_10_effects', 'Static advertising minus matched organic control, and adaptive L5 minus static advertising, using only displayed ranks 2--10 in every condition. Rank 1 is excluded even when no advertisement is delivered. Head exposure uses nine slots per decision; weighted exposure uses the original d(r)=1/log2(r+1) for ranks 2--10, normalised over those included ranks. Effects are paired within seed/population and iteration, in percentage points, with the mean over iterations 1--30 and iteration-30 endpoint separate. Organic rank displacement and feedback remain in these contrasts.', recommended=True, why='This companion excludes the sponsored position from every condition and shows the remaining displayed positions, while the full-slate result is retained separately.')
for metric in SLOT_METRICS:
    table = effect_table(slot_effects, metric).rename(columns={'value': label(metric) + ' difference (pp)'})
    report.table(table, 'rq1_advertising_' + metric + '_effects', label(metric) + ': matched treatment-minus-control effects in percentage points. Rank 1 is excluded from every condition, including organic and withdrawn-advertising decisions. Original displayed-rank discounts are retained in the weighted measure.', source_frame=slot_metrics)
report.table(diagnostic_table(slot_trajectory, SLOT_METRICS), 'rq1_advertising_ranks_2_10_levels', 'Head exposure and weighted head exposure using only displayed ranks 2--10 in every advertising condition. Values are shares, with the mean over iterations 1--30 and iteration-30 endpoint separate. Weighted exposure retains original rank discounts d(2) through d(10), normalised over those nine weights.', source_frame=slot_metrics)"""),
            _code("""figure = trajectory_grid(slot_trajectory, SLOT_METRICS, title='Advertising exposure trajectories: displayed ranks 2–10 only', scales=advertising_scales, series='condition_label')
figure._analysis_spec['exposure_scope'] = 'Displayed ranks 2–10 only, in every condition; original rank discounts retained'
report.figure(figure, slot_trajectory, 'rq1_advertising_ranks_2_10_trajectories', 'Canonical per-iteration head exposure restricted to displayed ranks 2--10 in organic, static-advertising, and adaptive-advertising conditions. Each decision contributes nine slots; rank 1 is excluded whether sponsored or organic. Weighted exposure preserves the original ranks 2--10 and normalises over their discounts. Panels identify seeds, while line styles and markers identify conditions. These are absolute exposure shares, not control-subtracted effects.')
slot_consistency = slot_views['seed_consistency'].copy()
slot_consistency['metric'] = slot_consistency.metric.map(label)
report.table(slot_consistency, 'rq1_advertising_ranks_2_10_seed_consistency', 'Descriptive consistency of matched advertising effects for head exposure restricted to displayed ranks 2--10 in every condition. Effects and their observed seed ranges are percentage points; the weighted version retains original rank discounts. Seed/population blocks remain separate replications.', columns=['comparison', 'metric', 'window', 'n_seed_blocks', 'mean_effect', 'min_effect', 'max_effect', 'same_nonzero_sign'], source_frame=slot_metrics)
slot_accounting = label_conditions(slot_accounting_table(slot_views['summaries']))
report.table(slot_accounting, 'rq1_advertising_rank1_inclusion_accounting', 'Within each saved canonical slate, full exposure over displayed ranks 1--10 minus exposure normalised over ranks 2--10, in percentage points. Rank 1 may contain a sponsored item or an organic item. Weighted values use original displayed-rank discounts in both scopes. This is descriptive same-slate rank-1 inclusion accounting, not a causal estimate of advertising or feedback.', columns=['condition_label', 'seed', 'population_seed', 'window', *ACCOUNTING_LABELS.values()], source_frame=slot_metrics)"""),
            _code("""advertising_effects = effect_summaries.loc[effect_summaries.kind.eq('advertising')].copy()
figure = effect_grid(advertising_effects, ['head_exposure', 'weighted_head_exposure'], title='Advertising exposure effects: all displayed ranks 1–10', scales=scales, value='value', facet='window')
figure._analysis_spec['exposure_scope'] = 'All displayed ranks 1–10, including any sponsored rank-1 item'
report.figure(figure, advertising_effects, 'rq1_advertising_exposure_effects', 'Original full-slate exposure metrics over all displayed ranks 1--10, including any rank-1 sponsored item: static advertising at L4 minus matched organic control, and adaptive L5 minus static advertising. Positive percentage-point differences mean higher full-slate head exposure. Both contrasts use the same 150-active/150-reserve protocol; the mean over iterations 1--30 and iteration-30 endpoint remain separate. This full-slate result is retained alongside the companion for ranks 2--10.')
for metric in ['head_exposure', 'weighted_head_exposure']:
    table = effect_table(advertising_effects, metric).rename(columns={'value': label(metric) + ' difference (displayed ranks 1–10; pp)'})
    report.table(table, 'rq1_advertising_' + metric + '_ranks_1_10_effects', label(metric) + ': original full-slate matched effects over displayed ranks 1--10, including any sponsored item at rank 1, in percentage points. The companion for ranks 2--10 is exported separately.')
advertising_raw = label_conditions(results['advertising'].loc[results['advertising'].study.eq('movielens_sampling_outcome_a150_r150_e804') & results['advertising'].condition_id.str.startswith('O')])
advertising = advertising_display(advertising_raw)"""),
            _code("""figure = trajectory_grid(advertising, ['advertising_eligible_users', 'sponsored_exposures', 'sponsored_clicks'], title='Advertising delivery and sponsored responses', scales=scales, series='condition_label', scale_overrides={'advertising_eligible_users': (0, 150), 'sponsored_exposures': (0, 150), 'sponsored_clicks': (0, 70)})
report.figure(figure, advertising, 'rq1_advertising_delivery', 'Per-iteration sponsored-delivery users, exposures, and clicks in organic L4, static-advertising L4, and adaptive L5 conditions. These are counts, not differences from control. Organic campaign eligibility is not applicable and is omitted from that panel; its actual sponsored exposures/clicks remain zero. Static delivery follows the current active-user count; adaptive delivery can stop while a user remains active.')
diagnostic_tables(report, advertising, {'eligibility': ['advertising_eligible_users', 'advertising_eligibility_share'], 'sponsored_counts': ['sponsored_exposures', 'sponsored_clicks'], 'sponsored_shares': ['sponsored_exposure_share', 'weighted_sponsored_exposure', 'sponsored_click_share']}, prefix='rq1_advertising', context='Advertising mechanism diagnostics. Organic campaign eligibility is not applicable; its saved zeros are masked only in this display view, while genuine zero sponsored exposures/clicks remain zero.')
report.table(horizon_counts(advertising, ['sponsored_exposures', 'sponsored_clicks']), 'rq1_advertising_totals', 'Sponsored delivery and click counts summed over 30 iterations, retaining each seed and delivery condition.')"""),
            _code("""advertising_outcomes = trajectory.loc[trajectory.study.eq('movielens_sampling_outcome_a150_r150_e804') & trajectory.condition_id.str.startswith('O')]
diagnostic_tables(report, advertising_outcomes, {
    'click_exposure': ['head_click_share', 'click_exposure_gap', 'weighted_click_exposure_gap'],
    'click_behaviour': ['click_rate', 'no_click_rate', 'valid_clicks'],
    'coverage': ['exposure_coverage', 'click_coverage'],
    'concentration': ['exposure_gini', 'click_gini']}, prefix='rq1_advertising', context='Matched advertising conditions. Click-exposure gaps here use the original full exposure over displayed ranks 1--10, including any sponsored rank-1 item; they do not use the separate ranks-2--10 companion.')"""),
            _md("""## Interpretation guide

Use the paired effects to identify direction and magnitude, then inspect the trajectories for transient versus sustained changes. Check whether seed directions agree. For L4, relate changes to turnover and cluster composition; for L5, relate them to the actual decline or persistence of sponsored delivery. Coverage and click-exposure gaps provide supporting evidence rather than additional independent replications."""),
        ],
        "02_user_response_models.ipynb": [
            _md("""## User-response models under common feedback conditions

**RQ II:** How do bias trajectories differ between user-response models under comparable feedback conditions?

The main comparison uses the five response families at the common L0--L2 stages. L3 configurations belong to the separate ablation analysis. Longitudinal slates and user states can diverge across models; identical-state comparisons apply to canonical/replica/shuffled probes within one opportunity, not to entire model trajectories."""),
            _code("""from reported_runs.analysis.sections_rq1_3 import diagnostic_tables, arm_rank_statistics
models = trajectory.loc[trajectory.study.eq('movielens_feedback_a300_e804') & trajectory.stage.str.startswith(('L0', 'L1', 'L2'))].copy()
model_effects = effect_summaries.loc[effect_summaries.study.eq('movielens_feedback_a300_e804') & effect_summaries.kind.eq('feedback')].copy()"""),
            _md("""## Click composition and selection relative to exposure

Head-click share describes the composition of valid clicks. The two click-exposure gaps subtract unweighted or rank-weighted head exposure; only the exposure component receives rank weighting. Reading both metrics distinguishes a difference in the displayed items from a difference in the response to those opportunities."""),
            _code("""for stage, selected in models.groupby('stage', sort=True):
    figure = trajectory_grid(selected, ['head_click_share', 'click_exposure_gap', 'weighted_click_exposure_gap'], title=f'User-response models: {stage.split("_")[0]}', scales=scales)
    report.figure(figure, selected, 'rq2_model_trajectories_' + stage.split('_')[0].lower(), f'Canonical response-model trajectories at {stage.split("_")[0]}. Head-click share uses all valid clicks; the two gaps subtract unweighted or rank-weighted head exposure and are expressed in percentage points. Each point describes that iteration, not a model-minus-control effect. Model colours remain fixed.', recommended=stage.startswith('L2'), why='L2 shows the model differences after repeated history updates and refitting, while the gaps distinguish click composition from exposure.')
diagnostic_tables(report, models, {'head_clicks': ['head_click_share'], 'exposure': ['head_exposure', 'weighted_head_exposure'], 'selection': ['click_exposure_gap', 'weighted_click_exposure_gap']}, prefix='rq2_models', context='Response models at comparable L0--L2 stages.')"""),
            _code("""figure = effect_grid(model_effects, ['head_click_share', 'click_exposure_gap', 'weighted_click_exposure_gap'], title='Model-specific changes after adding feedback', scales=scales, value='value', facet='window')
report.figure(figure, model_effects, 'rq2_model_feedback_effects', 'Later-stage minus prior-stage effects: L1 minus L0 and L2 minus L1 within each response model. Positive values mean an increase in the labelled metric, in percentage points. Mean effects use the 30 matched iteration differences; endpoint effects use iteration 30. Each seed/population block remains separate.')
report.table(effect_table(model_effects, 'head_click_share'), 'rq2_model_feedback_head_clicks', 'Model-specific feedback effects on head-click share, in percentage points.')"""),
            _md("""## Click behaviour, catalogue coverage, and concentration

Click/no-click rates use valid decisions. Invalid-response rate uses all attempted opportunities and remains separate. Computational forced-choice models cannot emit no-click. Catalogue coverage counts distinct shown or clicked catalogue items; Gini describes how unevenly those counts are distributed, including zero-count items."""),
            _code("""diagnostic_tables(report, models, {'click_behaviour': ['click_rate', 'no_click_rate', 'invalid_rate', 'valid_clicks'], 'coverage': ['exposure_coverage', 'click_coverage'], 'concentration': ['exposure_gini', 'click_gini']}, prefix='rq2_models', context='Click behaviour, coverage, and concentration at common feedback stages.')"""),
            _md("""## Identical-order repeatability and order sensitivity

For LLM repeatability, same-item replica agreement uses the respective both-click subset; click/no-click agreement uses valid paired actions. For the replica-adjusted shuffling diagnostic, canonical, replica, and shuffled responses must all be valid clicks, so both agreement terms use the same cases. Computational order sensitivity instead uses TVD between saved item-choice probability distributions aligned by item identity. Missing saved probabilities are reported as unavailable, never inferred from realised clicks."""),
            _code("""probes = label_conditions(results['probes'].loc[results['probes'].study.eq('movielens_feedback_a300_e804') & results['probes'].stage.str.startswith(('L0', 'L1', 'L2'))])
llm_probes = probes.loc[probes.model.isin(['direct_llm', 'agentic_recommender'])]
computational_probes = probes.loc[~probes.model.isin(['direct_llm', 'agentic_recommender'])]
diagnostic_tables(report, llm_probes, {'repeatability': ['replica_pair_agreement', 'replica_both_no_click_share'], 'action_agreement': ['replica_action_agreement', 'replica_action_disagreement', 'shuffled_action_agreement', 'shuffled_action_disagreement'], 'common_subset_agreement': ['replica_agreement', 'shuffled_agreement', 'agreement_sensitivity'], 'eligibility': ['triple_valid_clicks', 'replica_both_clicks', 'shuffled_both_clicks'], 'valid_pairs': ['replica_valid_pairs', 'shuffled_valid_pairs'], 'excluded_pairs': ['replica_excluded_non_valid_pairs', 'shuffled_excluded_non_valid_pairs']}, prefix='rq2_llm', context='Paired LLM response diagnostics.')
diagnostic_tables(report, computational_probes, {'order_sensitivity': ['shuffle_tvd', 'replica_tvd'], 'saved_probability_coverage': ['tvd_opportunities', 'triple_valid_clicks']}, prefix='rq2_computational', context='Exact saved computational item-choice probability diagnostics.')"""),
            _code("""for stage, selected in llm_probes.groupby('stage', sort=True):
    figure = trajectory_grid(selected, ['replica_agreement', 'shuffled_agreement', 'agreement_sensitivity'], title=f'LLM order diagnostics: {stage.split("_")[0]}', scales=scales)
    report.figure(figure, selected, 'rq2_llm_order_' + stage.split('_')[0].lower(), f'Paired LLM probe responses at {stage.split("_")[0]}. Top: canonical-versus-replica same-item agreement. Middle: canonical-versus-shuffled same-item agreement. Bottom: replica agreement minus shuffled agreement, in percentage points. All three use the same all-valid-click triple subset within each iteration; positive differences mean lower agreement after shuffling.')
for stage, selected in computational_probes.groupby('stage', sort=True):
    figure = trajectory_grid(selected, 'shuffle_tvd', title=f'Computational probability TVD: {stage.split("_")[0]}', scales=scales)
    report.figure(figure, selected, 'rq2_computational_tvd_' + stage.split('_')[0].lower(), f'Computational probe responses at {stage.split("_")[0]}. Each iteration averages the within-opportunity TVD between canonical and shuffled saved item-choice probabilities, aligned by item ID. Zero means identical distributions and one means disjoint probability mass. This is a probability-distribution diagnostic, not realised click disagreement or a feedback-stage effect.')"""),
            _md("""## Clicked positions

The rank distributions pool valid click counts over the 30 iterations before normalising within each model, stage, seed, and presentation arm. This count-based distribution differs from an equal-weight mean of iteration shares. Mean clicked rank and first-position share are also reported through the common mean/endpoint summaries. Canonical concentration at early ranks alone cannot separate position sensitivity from relevant items appearing first; the shuffled comparison provides the paired diagnostic."""),
            _code("""ranks = results['ranks'].loc[results['ranks'].study.eq('movielens_feedback_a300_e804') & results['ranks'].stage.str.startswith(('L0', 'L1', 'L2'))]
for stage, selected in ranks.groupby('stage', sort=True):
    rank_counts = rank_summary(selected)
    figure = rank_grid(rank_counts, title=f'Clicked displayed positions: {stage.split("_")[0]}')
    report.figure(figure, rank_counts, 'rq2_clicked_ranks_' + stage.split('_')[0].lower(), f'Displayed-rank click composition at {stage.split("_")[0]}. Each curve pools valid click counts over iterations 1--30 before normalising within its model, presentation arm, and seed; it is not a time trajectory. The x-axis is displayed rank. Line style identifies canonical, replica, or shuffled presentation; marker identifies seed; colour identifies model.')
    report.table(rank_counts, 'rq2_rank_counts_' + stage.split('_')[0].lower(), 'Clicks per displayed rank and their within-arm share over all 30 iterations; no-click and invalid actions are outside this denominator.')
rank_statistics = arm_rank_statistics(ranks)
diagnostic_tables(report, rank_statistics, {'rank_statistics': ['mean_clicked_rank', 'first_position_click_share']}, prefix='rq2_models', context='Canonical, replica, and shuffled clicked-rank statistics, calculated separately for each presentation arm.', extra_ids=['arm'])"""),
            _md("""## Interpretation guide

Compare model families within the same stage, and check whether their paired feedback effects agree in direction. Use click-exposure gaps, rates, and invalid counts to explain differences in head-click share. Treat LLM repeated responses and computational probability TVD as different diagnostics, and retain their eligibility counts. The three sampled population/seed blocks describe consistency; they do not establish how accurately a simulator reproduces human choices."""),
        ],
        "03_agent_ablations.ipynb": [
            _md("""## Sequential agent ablations

**RQ III:** How do the tested memory and reflection configurations affect simulated user behaviour and bias trajectories?

The sequential controls are: no-reflection L3 versus frozen L2, fixed-profile L3 versus no-reflection L3 (reflection notes), and full-reflection L3 versus fixed-profile L3 (applied profile updates). All use 300 active users and the same frozen population within each seed.

**The first contrast is a decision-input noise check for the realised canonical trajectories, rather than an observed memory-retrieval effect.** No-reflection L3 stores private interaction records, but its canonical decisions never retrieve them. Its fixed profile, empty reflection notes, and response-information construction match frozen L2. Their individual planning outputs, choices, and subsequent recommendation slates can nevertheless diverge. This is an A/A-style run-variability check, not a universal noise floor or a pure estimate of LLM response randomness.

Reflection notes and profile updates enter the decision prompts directly, so zero memory-module calls do not imply that these other mechanisms were unused."""),
            _code("""from reported_runs.analysis.sections_rq1_3 import diagnostic_tables, horizon_counts, reflection_totals, arm_rank_statistics, profile_review_details
from reported_runs.analysis.agent_ablation_views import ABLATION_COMPARISON_ORDER, label_ablation_comparisons, ablation_effect_ranges
agents = trajectory.loc[trajectory.study.eq('movielens_feedback_a300_e804') & trajectory.model.eq('agentic_recommender') & trajectory.stage.str.startswith(('L2', 'L3'))].copy()
ablations = label_ablation_comparisons(effect_summaries.loc[effect_summaries.study.eq('movielens_feedback_a300_e804') & effect_summaries.kind.eq('ablation')])
report.table(label_ablation_comparisons(results['comparison_plan'].loc[results['comparison_plan'].kind.eq('ablation')]), 'rq3_ablation_controls', 'Agent-configuration comparisons. No-reflection L3 minus frozen L2 is an A/A-style variability check for the realised canonical trajectories; the later contrasts add reflection notes and applied profile updates.')"""),
            _md("""## Recorded memory use and forced actions

Counts below refer to executed `module_trace[].module` entries, rather than the next module requested by a planning response. Forced actions are action-module fallbacks after the reasoning-step limit; they do not mean that a particular item was imposed. These saved traces are not a complete count of provider attempts or failed calls omitted from terminal failure records.

Keep canonical decisions and nonpersistent diagnostic probes separate. The main RQ III conditions have no canonical memory retrieval, while two diagnostic probes do retrieve memory. The separate 150/150 protocol is included as supporting context, not as additional replications of the 300-user ablations."""),
            _code("""from reported_runs.analysis.agent_usage import load_agent_usage
from reported_runs.analysis.sections_rq1_3 import agent_usage_summary
usage_sources = sources.loc[sources.model.eq('agentic_recommender') & ((sources.study.eq('movielens_feedback_a300_e804') & sources.stage.str.startswith(('L2', 'L3'))) | sources.study.eq('movielens_sampling_outcome_a150_r150_e804'))]
usage = load_agent_usage(usage_sources, repo, repo / 'reported_runs/analysis/cache/agent_usage')
canonical_usage = usage.loc[usage.arm.eq('canonical')]
usage_columns = ['Protocol', 'Presentation arm', 'Runs', 'Decisions', 'Memory calls', 'Forced actions']
report.table(agent_usage_summary(canonical_usage), 'rq3_memory_forced_action_summary', 'Recorded memory-module calls and forced-action fallbacks over iterations 1--30. Canonical RQ III decisions and the separate 150/150 protocol are counted independently; diagnostic probes are excluded.', columns=usage_columns, source_frame=canonical_usage)
report.table(agent_usage_summary(canonical_usage, by_condition=True), 'rq3_memory_forced_actions_by_condition', 'Canonical memory-module calls and forced-action fallbacks by condition and seed. Counts describe completed saved module traces, not provider attempts.', columns=['Protocol', 'Seed', 'Condition', 'Decisions', 'Memory calls', 'Forced actions'], source_frame=canonical_usage)
probe_usage = usage.loc[usage.arm.ne('canonical')]
report.table(agent_usage_summary(probe_usage), 'rq3_probe_memory_forced_actions', 'Recorded module use in nonpersistent diagnostic responses, separately by presentation arm. Probe retrieval does not change the canonical user state.', columns=usage_columns, source_frame=probe_usage)"""),
            _md("""## Bias trajectories and successive configuration effects

Signed paired effects retain the direction of a change. The trajectory view checks whether a mean effect hides a transient change or an opposite final result. Click-exposure gaps again describe selection composition relative to displayed exposure. Compare the reflection-note and profile-update effects with the observed noise-check range for the **same metric and time summary**. These three-pair ranges are descriptive references, not confidence intervals or thresholds for declaring an effect."""),
            _code("""figure = effect_grid(ablations, ['head_click_share', 'head_exposure', 'weighted_head_exposure'], title='Agent ablations and decision-input noise check', scales=scales, value='value', facet='window', category_order=ABLATION_COMPARISON_ORDER)
report.figure(figure, ablations, 'rq3_ablation_effects', 'First: the A/A-style decision-input noise check, no-reflection L3 minus frozen L2, with no canonical memory retrieval. Next: fixed-profile L3 minus no-reflection L3 (reflection notes), and full-reflection L3 minus fixed-profile L3 (profile updates). Positive percentage-point effects mean an increase in the labelled outcome. Means use iterations 1--30; endpoints use iteration 30. The observed noise-check spread is a descriptive reference across three seed/population pairs, not a general noise floor.', recommended=True, why='The explicit order separates the realised variability check from the reflection-note and profile-update contrasts, while retaining every seed.')
effect_ranges = ablation_effect_ranges(ablations, ['head_click_share', 'head_exposure', 'weighted_head_exposure'])
report.table(effect_ranges, 'rq3_effect_ranges_and_noise_reference', 'Observed effect ranges across the available seed/population pairs, with the first contrast as a matched metric/time-summary reference. All effects are percentage points. Ranges are descriptive, not confidence intervals or significance thresholds.', columns=['comparison', 'metric', 'window', 'n_seed_blocks', 'min_effect', 'max_effect', 'reference_min', 'reference_max'], source_frame=ablations)
figure = trajectory_grid(agents, ['head_click_share', 'head_exposure', 'weighted_head_exposure'], title='Agent configurations: canonical trajectories', scales=scales, series='condition_label')
report.figure(figure, agents, 'rq3_agent_trajectories', 'Canonical head-click share and exposure under frozen-agent L2 and no-reflection, fixed-profile, and full-reflection L3. Each point is the within-iteration outcome, not an ablation difference. The agent colour remains fixed; line styles and markers distinguish configurations within each seed panel.')
diagnostic_tables(report, agents, {'head_outcomes': ['head_click_share', 'head_exposure', 'weighted_head_exposure'], 'selection': ['click_exposure_gap', 'weighted_click_exposure_gap']}, prefix='rq3_agents', context='Main 300-active-user agent configurations.')"""),
            _md("""## Click behaviour, coverage, and repeated responses

Click/no-click behaviour and invalid responses can change alongside click composition. Coverage and Gini are separate for exposure and clicks. Both item-agreement terms in the ablation order diagnostic use the common canonical/replica/shuffled all-valid-click subset; valid click/no-click agreement is reported separately."""),
            _code("""diagnostic_tables(report, agents, {'click_behaviour': ['click_rate', 'no_click_rate', 'invalid_rate', 'valid_clicks'], 'coverage': ['exposure_coverage', 'click_coverage'], 'concentration': ['exposure_gini', 'click_gini'], 'rank_statistics': ['mean_clicked_rank', 'first_position_click_share']}, prefix='rq3_agents', context='Agent click behaviour, coverage, and concentration.')
agent_probes = label_conditions(results['probes'].loc[results['probes'].study.eq('movielens_feedback_a300_e804') & results['probes'].model.eq('agentic_recommender') & results['probes'].stage.str.startswith(('L2', 'L3'))])
diagnostic_tables(report, agent_probes, {'agreement': ['replica_agreement', 'shuffled_agreement', 'agreement_sensitivity'], 'actions': ['replica_action_agreement', 'replica_action_disagreement', 'shuffled_action_agreement', 'shuffled_action_disagreement'], 'eligibility': ['triple_valid_clicks', 'replica_valid_pairs', 'shuffled_valid_pairs'], 'excluded_pairs': ['replica_excluded_non_valid_pairs', 'shuffled_excluded_non_valid_pairs']}, prefix='rq3_agents', context='Paired response diagnostics by agent configuration.')"""),
            _code("""agent_ranks = results['ranks'].loc[results['ranks'].study.eq('movielens_feedback_a300_e804') & results['ranks'].model.eq('agentic_recommender') & results['ranks'].stage.str.startswith(('L2', 'L3'))]
for condition, selected in agent_ranks.groupby('condition_id', sort=False):
    rank_counts = rank_summary(selected)
    figure = rank_grid(rank_counts, title='Agent clicked ranks: ' + label_conditions(selected).condition_label.iloc[0])
    report.figure(figure, rank_counts, 'rq3_ranks_' + condition.lower().replace('__', '_'), f'Displayed-rank click composition for {label_conditions(selected).condition_label.iloc[0]}. Each curve pools valid clicks over iterations 1--30 before normalising within presentation arm and seed; the x-axis is rank rather than iteration. Line style identifies canonical, replica, or shuffled presentation and marker identifies seed.')
    report.table(rank_counts, 'rq3_rank_counts_' + condition.lower().replace('__', '_'), 'Clicks per displayed rank and within-arm click shares for this agent configuration over the complete horizon.')
agent_rank_statistics = arm_rank_statistics(agent_ranks)
diagnostic_tables(report, agent_rank_statistics, {'rank_statistics_by_arm': ['mean_clicked_rank', 'first_position_click_share']}, prefix='rq3_agents', context='Canonical, replica, and shuffled clicked-rank statistics by agent configuration and presentation arm.', extra_ids=['arm'])"""),
            _md("""## Actual module use and reflection outcomes

Module invocation counts describe what was used; available modules alone do not establish use. Reflection failures are divided by requested reflection updates, so a configuration that never requests reflection has an undefined failure rate. Requested, successful, and failed updates remain distinct. Horizon totals sum event counts; the companion tables retain mean iteration counts and iteration-30 counts/rates."""),
            _code("""agent_diagnostics = label_conditions(results['agent'].loc[results['agent'].study.eq('movielens_feedback_a300_e804') & results['agent'].model.eq('agentic_recommender') & results['agent'].stage.str.startswith(('L2', 'L3'))])
modules = label_conditions(results['modules'].loc[results['modules'].study.eq('movielens_feedback_a300_e804') & results['modules'].model.eq('agentic_recommender') & results['modules'].stage.str.startswith(('L2', 'L3'))])
report.table(horizon_counts(modules, ['invocations', 'decisions_invoking'], extra_ids=['module']), 'rq3_module_totals', 'Module invocations and canonical decisions invoking each module over all 30 iterations. Counts are descriptive and are not additional replications.')
diagnostic_tables(report, modules, {'module_rates': ['invocations', 'invocation_rate']}, prefix='rq3_agents', context='Module counts and proportions of canonical decisions invoking each module.', extra_ids=['module'])"""),
            _code("""report.table(reflection_totals(agent_diagnostics), 'rq3_reflection_totals', 'Requested, successfully applied, and failed reflection updates summed over all 30 iterations, with failed updates divided by all requested updates. Absence of a request is not a failed update; no requests gives an undefined proportion.')
diagnostic_tables(report, agent_diagnostics, {'reflection': ['reflection_requested', 'reflection_applied', 'reflection_failed'], 'reflection_failure': ['reflection_error_rate']}, prefix='rq3_agents', context='Reflection request and outcome diagnostics.')
figure = trajectory_grid(agent_diagnostics, ['reflection_requested', 'reflection_error_rate'], title='Agent reflection use and failed-update rates', scales=scales, series='condition_label')
report.figure(figure, agent_diagnostics, 'rq3_reflection_diagnostics', 'Agent configurations from frozen L2 through the three L3 ablations. The upper row counts requested reflection updates in that iteration; the lower row divides failed updates by requested updates in the same iteration. No request gives an undefined rate, not zero. These are mechanism-use trajectories rather than control differences.')"""),
            _md("""## Applied profile changes

Only changes applied to persistent profile state count. Proposals suppressed by the fixed-profile condition are excluded. Per-iteration changed-user counts are not summed as distinct users; the cumulative changed-user count at iteration 30 gives the number of distinct users with at least one applied change over the whole simulation."""),
            _code("""report.table(horizon_counts(agent_diagnostics, ['profile_changes']), 'rq3_profile_change_totals', 'Total applied profile-update events over iterations 1--30. Multiple changes by one user are multiple events.')
diagnostic_tables(report, agent_diagnostics, {'profile_changes': ['profile_changes', 'profile_change_rate', 'profile_changed_users'], 'distinct_changed_users': ['cumulative_profile_changed_users']}, prefix='rq3_agents', context='Applied profile-change events, per-iteration changed users, and cumulative distinct changed users.')
figure = trajectory_grid(agent_diagnostics, ['profile_changes', 'cumulative_profile_changed_users'], title='Applied profile updates and distinct changed users', scales=scales, series='condition_label')
report.figure(figure, agent_diagnostics, 'rq3_profile_updates', 'Applied updates across frozen L2 and the three L3 ablations. The upper row counts new profile-update events in each iteration. The lower row counts distinct users with at least one applied profile change from iteration 1 through the displayed iteration; repeated updates of one user do not increase this cumulative distinct-user count. Suppressed proposals are excluded.')"""),
            _md("""## Seeded qualitative profile review

The following cases are selected deterministically from users with at least one applied profile change in the main full-reflection L3 condition: up to ten distinct users per seed, using that simulation seed. Initial and final profiles, plus applied-change evidence, support a manual assessment of retained interests, supported additions, and unsupported preference changes. No manual classifications have been invented. These descriptive examples do not validate human preference change."""),
            _code("""review = results['qualitative_review']
report.table(review, 'rq3_profile_review_selection', 'Deterministic sample of up to ten users with applied profile changes per seed; all manual judgements remain unassessed.', columns=['seed', 'population_seed', 'user_id', 'review_order', 'eligible_users', 'applied_change_count', 'first_change_iteration', 'last_change_iteration', 'manual_assessment'])
profile_review_details(report, review, results['profile_changes'])"""),
            _md("""## Interpretation guide

Read each sequential effect alongside actual mechanism use and update counts. The first contrast shows realised run-to-run variation when private memory is not retrieved; it does not identify an observed memory-retrieval effect. Compare later effects with that observed spread within the same metric and time summary, without treating it as a universal threshold. A profile-update contrast differs from a general claim that reflection changes preferences. Check consistency across the three campaign blocks, retain undefined rates where modules were not invoked, and reserve qualitative judgements for the evidence-linked manual review. Psychological-modelling extensions remain optional and outside these core comparisons."""),
        ],
    }
