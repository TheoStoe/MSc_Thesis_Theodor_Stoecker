"""Exact scientific instructions and prompt builders from the saved screener.

Only portability annotations/imports were changed; no paper text is embedded.
"""
from __future__ import annotations
import json
from typing import Any, Dict, Optional, Mapping

SCREENING_CRITERIA = '\nSCREENING TASK:\nScreen papers for a literature review on LLM-based user agents in recommender-system simulations.\n\nUse ONLY the title, abstract, and keywords. Do not use outside knowledge.\n\nINCLUDE if the paper appears to meet ALL of the following:\n1. The paper uses, proposes, evaluates, or discusses a Large Language Model (LLM) as a simulated user agent/user response model.\n2. The LLM-based user agent is used in the context of recommender system simulation.\n3. The recommender setting involves item recommendation, such as top-k or top-1 recommendation, or user click prediction.\n\nEXCLUDE if the paper clearly meets ANY of the following:\n1. The paper is about conversational recommender systems where the LLM is mainly used in a dialogue setting.\n2. The paper uses an LLM to generate recommendations, rank items, explain recommendations, or improve the recommender itself, but does not use the LLM as a simulated user or user response model.\n3. The paper is about general LLM agents, user simulation, or human behaviour simulation but not in a recommender-system context.\n4. The paper is about recommender systems but does not involve LLM-based user simulation, user agents, or user response modelling.\n5. The paper is only about search, question answering, conversational AI, advertising, or social media simulation without a recommender system or recommendation context.\n\nUNSURE if:\n1. The abstract suggests possible LLM-based user simulation in a recommender-system context but does not provide enough detail.\n2. The paper mentions recommender systems and LLM-based users/agents, but it is unclear whether the LLM acts as the user or as the recommender.\n\nImportant:\n- If the title/abstract/keywords are ambiguous but plausibly relevant, choose "unsure" rather than "exclude".\n- Give a brief reason tied to the inclusion/exclusion criteria.\n'

CODING_SCHEMA = '\nYou are coding papers according to a taxonomy for LLM user agents in recommender-system simulations.\n\nGeneral coding rules:\n\n* Code only information that is supported by the paper.\n* Do not infer categories from general background knowledge.\n* If a category is not discussed, use "not_reported".\n* If a category does not apply to the paper, use "not_applicable".\n* If the evidence is ambiguous, use "unclear" and explain why.\n* Use multiple labels only for dimensions marked MULTI-SELECT.\n* For every selected label, provide short evidence from the paper.\n* Evidence should refer to the paper\'s content, not your own reasoning.\n* Prefer conservative coding over speculative coding.\n* If the paper claims a capability, still check whether the described system actually fulfils the definition below.\n\nTaxonomy dimensions:\n\n1. Domain\n   Selection type: SINGLE-SELECT\n   Allowed labels:\n\n* general_purpose_recommender\n* movie_video_recommender\n* news_recommender\n* music_recommender\n* e_commerce\n* other\n* unclear\n\nCoding rule:\nClassify the recommender-system domain studied in the paper.\nIf a specific system or dataset domain is studied, use the corresponding domain.\nIf no specific domain is provided, or if datasets span multiple domains, code as "general_purpose_recommender".\nUse "other" only when the paper clearly studies a domain outside the listed categories.\n\n2. Bias addressed\n   Selection type: MULTI-SELECT\n   Allowed labels:\n\n* representation_bias\n* measurement_bias\n* historical_bias\n* evaluation_bias\n* none\n* not_reported\n* unclear\n\nCoding rule:\nCode the types of bias explicitly discussed or addressed by the paper.\nThis dimension is not mutually exclusive.\nUse "none" only if the paper explicitly states that bias is not addressed or not relevant.\nUse "not_reported" if the paper does not mention bias.\nAlways mention the actual mentioned biases in justification, where applicable.\n\n3. Modelling techniques\n   Selection type: MULTI-SELECT\n   Allowed labels:\n\n* predefined_rules\n* symbolic_equations\n* stochastic_modelling\n* machine_learning_model\n* not_reported\n* unclear\n\nCoding rule:\nClassify the modelling techniques used to simulate user behaviour or user agents.\nUse "predefined_rules" when user behaviour is determined by fixed rules, scripts, transition rules, heuristics, or if-then logic.\nUse "symbolic_equations" when user behaviour is modelled using deterministic mathematical equations.\nUse "stochastic_modelling" when user behaviour is modelled using probability distributions or random sampling processes.\nUse "machine_learning_model" when an ML model, RL model, neural model, or LLM is used as part of user modelling or user-agent behaviour.\nFor LLM-based user agents, select "machine_learning_model"; additionally select other labels if rules, equations, or stochastic components are also used.\n\n4. Basis for user modelling\n   Selection type: MULTI-SELECT\n   Allowed labels:\n\n* not_applicable\n* random_sampling\n* system_log_based\n* llm_generated\n* not_reported\n* unclear\n\nCoding rule:\nClassify how user profiles, user agents, or comparable user representations are constructed.\nUse "not_applicable" if the paper does not construct or simulate user models or user agents.\nUse "random_sampling" when agents or user attributes are sampled from a probability distribution.\nUse "system_log_based" when historical recommender-system interaction data, user logs, ratings, clicks, or behavioural traces are used to construct agents.\nUse "llm_generated" when an LLM is prompted to generate user profiles, personas, preferences, histories, or similar user representations.\nMultiple labels may apply, for example when agents are sampled from distributions estimated from system logs.\n\n5. Psychological basis\n   Selection type: MULTI-SELECT\n   Allowed labels:\n\n* cognition_inspired\n* personality_aware\n* affect_aware\n* none\n* not_reported\n* unclear\n\nCoding rule:\nCode whether the paper makes psychological assumptions about user behaviour.\nUse "cognition_inspired" when user modelling includes decision-making, memory, attention, or similar cognitive processes.\nUse "personality_aware" when user modelling includes personality traits, personas grounded in personality theory such as Big Five / five-factor model, or comparable personality constructs.\nUse "affect_aware" when user modelling includes emotion or mood as drivers of behaviour.\nUse "none" only if the paper explicitly uses user modelling but contains no psychological basis.\nUse "not_reported" if the paper gives insufficient detail.\n\n6. Agent capabilities\n   Selection type: MULTI-SELECT\n   Allowed labels:\n\n* autonomy\n* reactivity\n* social_ability\n* proactiveness\n* none\n* not_reported\n* unclear\n\nCoding rule:\nCode the agentic capabilities actually fulfilled by the simulated user agent.\nUse "autonomy" when the agent independently produces a user response. A parser or validation step to catch hallucinations does not remove autonomy.\nUse "reactivity" when the agent changes its behaviour, state, preferences, or responses depending on the environment or interaction history.\nUse "social_ability" only when there is simulated agent-to-agent interaction or social interaction between users/agents.\nUse "proactiveness" only when the agent decides how or when to interact with the system, rather than only responding to given recommendations. This includes deciding to leave the platform.\nUse "none" if the system is not meaningfully agentic.\n\n7. Agentic modules\n   Selection type: MULTI-SELECT\n   Allowed labels:\n\n* profile\n* action\n* memory\n* planning\n* self_reflection\n* other\n* none\n* not_reported\n* unclear\n\nCoding rule:\nClassify the modules used to implement the user agent.\nUse "profile" when the system includes an explicit user profile, persona, preference description, demographic profile, or comparable representation.\nUse "action" when the agent can choose or generate user actions such as clicking, rating, buying, watching, skipping, or leaving.\nUse "memory" when the agent stores or uses interaction history, long-term preferences, past experiences, or previous recommendations.\nUse "planning" when the agent formulates multi-step goals, strategies, or future-oriented interaction plans.\nUse "self_reflection" when the agent explicitly reflects on its own preferences, behaviour, satisfaction, or prior interactions and updates itself accordingly.\nUse "other" when a module fulfills a different purpose.\n\n8. Feedback-loop implementation\n   Selection type: MULTI-SELECT\n   Allowed labels:\n\n* none\n* self_reflection\n* agent_to_agent_interaction\n* retraining\n* not_reported\n* unclear\n\nCoding rule:\nCode how feedback loops are implemented in the simulation.\nUse "none" if no feedback loop is implemented.\nUse "self_reflection" when the user agent updates preferences, state, or behaviour through reflection over prior interactions.\nUse "agent_to_agent_interaction" when feedback emerges through interaction between simulated agents.\nUse "retraining" when the recommender system, user model, or agent model is retrained or updated using simulated interaction data.\n\n9. Feedback-loop type\n   Selection type: MULTI-SELECT\n   Allowed labels:\n\n* none\n* individual_feedback_loop\n* ml_model_feedback_loop\n* sampling_feedback_loop\n* outcome_feedback_loop\n* feature_feedback_loop\n* not_reported\n* unclear\n\nCoding rule:\nClassify the type of feedback loop represented.\nUse "none" if no feedback loop is modelled.\nUse "individual_feedback_loop" when a user\'s preferences, state, or behaviour are updated over time based on interactions.\nUse "ml_model_feedback_loop" when recommender outputs or simulated interactions are reinserted into the training data or used to update an ML model.\nUse "sampling_feedback_loop" when the composition of the observed user population changes, for example because agents enter, leave, or are selectively observed.\nUse "outcome_feedback_loop" when system outputs affect real or simulated outcomes that later feed back into the system.\nUse "feature_feedback_loop" when observable user features or interaction features are updated by the system\'s prior recommendations or user interactions.\n\n10. Recommender evaluation\n    Selection type: MULTI-SELECT\n    Allowed labels:\n\n* performance_metrics\n* beyond_accuracy_metrics\n* scalability\n* none\n* not_reported\n* unclear\n\nCoding rule:\nCode how the recommender system is evaluated.\nUse "performance_metrics" for standard performance related recommender metrics.\nUse "beyond_accuracy_metrics" for metrics considering diversity, novelty, fairness, or related non-accuracy goals.\nUse "scalability" when computational cost, runtime, throughput, scale, number of users/items, or deployment feasibility is evaluated.\nUse "none" if the recommender is not evaluated.\n\n11. Simulator evaluation\n    Selection type: MULTI-SELECT\n    Allowed labels:\n\n* performance_metrics\n* scalability\n* real_world_data_offline_logs\n* real_world_data_user_testing\n* none\n* not_reported\n* unclear\n\nCoding rule:\nCode how the user simulator or user-agent simulation is evaluated.\nUse "performance_metrics" when the simulator is evaluated using predictive accuracy or similar quantitative/qualitative performance criteria.\nUse "scalability" when the simulator is evaluated for computational cost, runtime, throughput, number of simulated users, number of interactions, or deployment feasibility.\nUse "real_world_data_offline_logs" when simulated behaviour is compared with historical user logs, ratings, clicks, purchases, or other offline behavioural data.\nUse "real_world_data_user_testing" when simulated behaviour is compared with live users, user studies, online experiments, or human evaluation involving real users.\nUse "none" if the simulator is not evaluated.\n\nRequired output behaviour:\nFor each of the 11 taxonomy dimensions, return one category object.\nThe category field must use the exact dimension name.\nThe value field must contain the selected label or labels.\nThe evidence field must briefly justify the coding using information from the paper.\nThe confidence field must be between 0 and 1.\nThe reviewer_notes field should mention ambiguities, missing information, or why a conservative coding choice was made.\nReturn the structured response.\n'

SYSTEM_PROMPT_COMMON = '\nYou are assisting with a literature review as a second screener.\n\nImportant rules:\n- Be conservative.\n- Do not invent information.\n- If information is missing, say "not reported" or "unclear".\n- Return only the requested structured data.\n- Use brief, auditable reasoning, not hidden chain-of-thought.\n'

MAX_EXTRACTED_TEXT_CHARS_STEP2 = 450_000

def safe_json_dumps(obj, max_chars=None):
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return text if max_chars is None or len(text) <= max_chars else text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"

def build_step1_prompt(row: Mapping[str, Any]) -> str:
    return f"""
TASK:
Screen this paper using ONLY title, abstract, and keywords. Do not use outside knowledge.

SCREENING CRITERIA:
{SCREENING_CRITERIA}

PAPER:
title: {row.get("title", "")}
keywords: {row.get("keywords", "")}
abstract:
{row.get("abstract", "")}

Return the structured result.
"""


def build_step2_prompt(row: Mapping[str, Any], step1_result: Optional[Dict[str, Any]] = None,
                       extracted_text: Optional[str] = None) -> str:
    text_block = ""
    if extracted_text:
        text_block = f"""

EXTRACTED FULL TEXT FALLBACK:
The following text was extracted locally from the PDF. Page markers are included.

{extracted_text[:MAX_EXTRACTED_TEXT_CHARS_STEP2]}
"""

    return f"""
TASK:
Conduct full-text screening and coding for this paper.

First:
- Repeat an inclusion/exclusion/unsure decision using the full paper.
- Apply the screening criteria again.
- If the paper should be excluded at full-text stage, still code what can be coded, but mark missing fields as "not applicable" or "not reported".

Second:
- Apply the coding schema.
- Return one object with paper-level decision fields and a categories array.
- Each category will later become one CSV row.

SCREENING CRITERIA:
{SCREENING_CRITERIA}

CODING SCHEMA:
{CODING_SCHEMA}

PAPER METADATA:
title: {row.get("title", "")}
keywords: {row.get("keywords", "")}
abstract: {row.get("abstract", "")}

STEP 1 ABSTRACT-ONLY RESULT:
{safe_json_dumps(step1_result or {}, max_chars=6000)}
{text_block}

Return the structured result.
"""

# Settings from the recorded run. Importing this file makes no API calls.
# The full-text fallback uses the stage2 schema as well.
RECORDED_SETTINGS = {'provider': 'openai',
 'model_step1': 'gpt-5.5',
 'model_step2': 'gpt-5.5',
 'temperature': None,
 'max_output_tokens_step1': 1800,
 'max_output_tokens_step2': 7000,
 'rate_limit_seconds': 2.0,
 'max_retries': 3,
 'retry_backoff_seconds': 8,
 'use_pdf_file_upload_step2': True,
 'fallback_to_extracted_text_if_upload_fails': True,
 'max_pdf_mb_for_upload': 32,
 'max_extracted_text_chars_step2': 450000,
 'prompt_version_step1': 'step1_v001_criteria',
 'prompt_version_step2': 'step2_v001_schema',
 'actual_recorded_model': 'gpt-5.5-2026-04-23'}

RESPONSE_SCHEMAS = {'stage1_abstract_screening': {'type': 'object',
                               'additionalProperties': False,
                               'properties': {'paper_id': {'type': 'string'},
                                              'title': {'type': 'string'},
                                              'decision': {'type': 'string',
                                                           'enum': ['include',
                                                                    'exclude',
                                                                    'unsure']},
                                              'confidence': {'type': 'number',
                                                             'minimum': 0,
                                                             'maximum': 1},
                                              'inclusion_criteria_met': {'type': 'array',
                                                                         'items': {'type': 'string'}},
                                              'exclusion_criteria_met': {'type': 'array',
                                                                         'items': {'type': 'string'}},
                                              'reasoning_brief': {'type': 'string'},
                                              'needs_human_review': {'type': 'boolean'},
                                              'human_review_notes': {'type': 'string'}},
                               'required': ['paper_id',
                                            'title',
                                            'decision',
                                            'confidence',
                                            'inclusion_criteria_met',
                                            'exclusion_criteria_met',
                                            'reasoning_brief',
                                            'needs_human_review',
                                            'human_review_notes']},
 'stage2_fulltext_coding': {'type': 'object',
                            'additionalProperties': False,
                            'properties': {'paper_id': {'type': 'string'},
                                           'title': {'type': 'string'},
                                           'full_text_decision': {'type': 'string',
                                                                  'enum': ['include',
                                                                           'exclude',
                                                                           'unsure']},
                                           'full_text_confidence': {'type': 'number',
                                                                    'minimum': 0,
                                                                    'maximum': 1},
                                           'full_text_reasoning_brief': {'type': 'string'},
                                           'exclusion_criteria_triggered': {'type': 'array',
                                                                            'items': {'type': 'string'}},
                                           'inclusion_criteria_triggered': {'type': 'array',
                                                                            'items': {'type': 'string'}},
                                           'categories': {'type': 'array',
                                                          'items': {'type': 'object',
                                                                    'additionalProperties': False,
                                                                    'properties': {'category': {'type': 'string'},
                                                                                   'selection_type': {'type': 'string',
                                                                                                      'enum': ['single_select',
                                                                                                               'multi_select']},
                                                                                   'selected_labels': {'type': 'array',
                                                                                                       'items': {'type': 'string'}},
                                                                                   'value': {'type': 'string'},
                                                                                   'evidence': {'type': 'string'},
                                                                                   'confidence': {'type': 'number',
                                                                                                  'minimum': 0,
                                                                                                  'maximum': 1},
                                                                                   'reviewer_notes': {'type': 'string'}},
                                                                    'required': ['category',
                                                                                 'selection_type',
                                                                                 'selected_labels',
                                                                                 'value',
                                                                                 'evidence',
                                                                                 'confidence',
                                                                                 'reviewer_notes']}},
                                           'paper_level_notes': {'type': 'string'}},
                            'required': ['paper_id',
                                         'title',
                                         'full_text_decision',
                                         'full_text_confidence',
                                         'full_text_reasoning_brief',
                                         'exclusion_criteria_triggered',
                                         'inclusion_criteria_triggered',
                                         'categories',
                                         'paper_level_notes']}}
