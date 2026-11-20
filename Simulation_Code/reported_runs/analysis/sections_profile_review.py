"""A separate notebook for the thirty human profile-review cases."""


def notebook_sections():
    return [
        ("markdown", "# RQ III: qualitative profile review\n\n"
         "Review ten distinct users per simulation seed (42, 43, and 44), using the main MovieLens full-reflection L3 runs with 300 active users. "
         "Eligibility requires at least one applied canonical profile change. The existing sampled users and their sampled order are preserved. "
         "This is a descriptive human review of thirty changed profiles, with all applied changes and the full canonical interaction histories supplied as evidence."),
        ("markdown", "## Prepare the same seeded sample\n\n"
         "The shared sampler orders eligible IDs by `(str(type(id)), str(id))`, then samples without replacement using `numpy.random.default_rng(simulation_seed)`. "
         "For these integer IDs, that ordering is lexicographic. We reuse the exact selection already recorded in the analysis snapshot. "
         "Evidence comes from verified canonical events and saved checkpoints. Replica and shuffled responses are excluded."),
        ("code", "from reported_runs.analysis.profile_review_workflow import prepare_profile_review, ensure_profile_workbook\n"
         "from reported_runs.analysis.profile_review_assessments import read_profile_assessments, profile_review_frequencies\n\n"
         "from reported_runs.analysis.profile_review_views import frequency_table, CATEGORY_COLUMNS, STATUS_COLUMNS\n\n"
         "evidence = prepare_profile_review(results, repo)\n"
         "cases = evidence['cases']\n"
         "changes = evidence['changes']\n"
         "history = evidence['history']\n"
         "selection = evidence['selection']\n"
         "report.table(selection, 'rq3_profile_review_sampling', 'Seeded sample of users with applied canonical profile changes in the main full-reflection L3 condition. The run seed also seeds sampling; the recorded selection is reused without resampling.', columns=['seed', 'population_seed', 'eligible_users', 'requested_sample', 'selected_users'])"),
        ("code", "report.table(cases, 'rq3_profile_review_case_index', 'Thirty unassessed profile-review cases, ten per simulation seed. Applied change counts and first/last iterations refer to canonical persistent profile updates; no manual assessment is inferred.', columns=['case_id', 'seed', 'population_seed', 'user_id', 'review_order', 'applied_change_count', 'first_change_iteration', 'last_change_iteration'])\n"
         "display(Markdown(f\"**Evidence:** {len(cases)} profiles, {len(changes)} applied changes, and {len(history)} canonical interactions. All initial/final profiles and intermediate change texts are supplied in the workbook and flat CSVs.\"))"),
        ("markdown", "## Review in Excel\n\n"
         "The workbook has one sheet per sampled user, an overview, and a flat **Review data** sheet. "
         "Each user sheet has the same six blue fields: extent of change, evidence support, retained initial interests, supported additions, unsupported changes, and evidence notes. "
         "All six fields are required for a complete case; use ‘none identified’ where appropriate. "
         "The overview explains the categories and updates frequencies as you complete reviews. "
         "Compare final with initial profiles, then examine intermediate changes and the full available history. "
         "For no-clicks, consider the displayed slate and logged reason. Private memory notes are logged context and do not establish that memory was retrieved.\n\n"
         "Extent categories: **minor adjustment**, **substantial change**, and **unclear**. "
         "Support categories: **supported**, **mixed**, **unsupported**, and **insufficient evidence**. "
         "No field is automatically classified. Existing workbooks are preserved when this notebook is rerun."),
        ("code", "workbook_path = ensure_profile_workbook(evidence)\n"
         "display(Markdown(f'[Open the review workbook]({workbook_path.as_posix()}). Save your completed reviews, then rerun the cells below. [Flat evidence CSVs]({evidence[\"directory\"].as_posix()}) remain available for later analyses.'))"),
        ("markdown", "## Import saved assessments and export the analysis\n\n"
         "Only the six editable cells on each user sheet supply human assessments. The importer verifies the case IDs, seeds, population seeds, users, and update counts. "
         "Frequency tables include only fully completed cases. Not-started and partial reviews are reported separately; undefined proportions stay missing. "
         "The denominator is reviewed profiles, not update events or the entire simulated population. "
         "Saving Excel and rerunning these cells refreshes typed CSVs and LaTeX tables without replacing the workbook."),
        ("code", "assessments = read_profile_assessments(workbook_path, cases)\n"
         "frequencies = profile_review_frequencies(assessments)\n"
         "assessments.to_csv(evidence['directory'] / 'assessments.csv', index=False)\n"
         "frequencies.to_csv(evidence['directory'] / 'assessment_frequencies.csv', index=False)\n"
         "completion = frequencies[frequencies.dimension.eq('review_status')]\n"
         "report.table(frequency_table(frequencies, 'review_status'), 'rq3_profile_review_completion', 'Human profile-review completion by seed and overall. Not-started and partial cases are separate from completed assessments; no category is assigned to an unfinished review.', column_format=STATUS_COLUMNS, source_frame=assessments)"),
        ("code", "extent = frequencies[frequencies.dimension.eq('extent_of_change')]\n"
         "support = frequencies[frequencies.dimension.eq('support_in_available_evidence')]\n"
         "report.table(frequency_table(frequencies, 'extent_of_change'), 'rq3_profile_review_extent', 'Extent-of-change classifications among completed human profile reviews. Counts and proportions use completed profiles, with unfinished case counts retained. No completed reviews means undefined proportions.', column_format=CATEGORY_COLUMNS, source_frame=assessments)\n"
         "report.table(frequency_table(frequencies, 'support_in_available_evidence'), 'rq3_profile_review_support', 'Evidence-support classifications among completed human profile reviews. Assessments use the initial profile and full available canonical history. Counts refer to profiles rather than applied change events.', column_format=CATEGORY_COLUMNS, source_frame=assessments)"),
        ("markdown", "## Later plots and results writing\n\n"
         "`assessments` has one row per selected user, `changes` one row per applied update, `history` one row per canonical interaction, and `frequencies` one row per category/seed scope. "
         "These frames and their CSVs are ready for ordinary dataframe indexing, plots, or further LaTeX exports. "
         "Report this as one researcher's descriptive review of changed profiles. It does not validate real human preference change."),
    ]
