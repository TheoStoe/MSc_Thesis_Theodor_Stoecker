"""One offline scientific definition for all seven thesis result notebooks.

Canonical displayed lists determine trajectories. Replica, shuffled, and
original-order records are diagnostic only. The final rank discount is
``1 / log2(r + 1)`` for one-based ranks; legacy saved exposure summaries are
deliberately ignored. Missing denominators remain undefined.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .readers import action, file_hash, jsonl_rows, opportunities, read_json, recorded_probabilities, resolve_path

VERSION = "final-rq-metrics-v1-dcg-one-based"
REPO = Path(__file__).resolve().parents[2]
GENDERS = ("female", "male", "nonbinary", "other", "unknown", "ambiguous")
TABLES = ("trajectory", "probes", "ranks", "agent", "modules", "profile_changes", "profiles",
          "population", "clusters", "advertising", "references", "health", "top_items")
IDENTITY = ("source_id", "dataset", "study", "seed", "population_seed", "condition_id", "model", "stage",
            "intervention", "user_update_policy", "run_id", "physical_run_id")


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def rank_discount(rank):
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 1:
        raise ValueError("rank is a positive one-based integer")
    return 1 / math.log2(rank + 1)


def normalized_gini(counts, catalogue):
    """Finite-catalogue-normalised Gini, including every unobserved item."""
    if not set(counts) <= set(catalogue):
        raise ValueError("Counts contain an item outside the reference catalogue")
    values = np.sort(np.array([counts.get(item, 0) for item in catalogue], dtype=float))
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Gini counts must be finite and nonnegative")
    if values.sum() == 0:
        return float("nan")
    if len(values) == 1:
        return 0.0
    return float(np.dot(2 * np.arange(1, len(values) + 1) - len(values) - 1, values)
                 / ((len(values) - 1) * values.sum()))


def choice_tvd(left, right):
    """Distance between saved item-choice masses after aligning the same IDs."""
    p, q = recorded_probabilities(left), recorded_probabilities(right)
    if p is None or q is None:
        return float("nan")
    if set(p) != set(q):
        raise ValueError("TVD requires the same offered items")
    return .5 * sum(abs(p[item] - q[item]) for item in p)


def _defined(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and str(value) != ""


def _field(row, *names, default=None):
    return next((row[name] for name in names if name in row and _defined(row[name])), default)


def _reference(row, repo_root):
    """Read numeric prepared data directly; never construct simulator objects."""
    dataset = str(row["dataset"])
    if dataset == "goodreads":
        prepared = resolve_path(row["prepared_directory"], repo_root)
        labels_path = resolve_path(row["labels_path"], repo_root)
        labels_file = read_json(labels_path)
        labels = {int(item): value.get("gender", "unknown") for item, value in labels_file["labels"].items()}
        policy = labels_file.get("annotation_policy", {}).get("policy_id", "legacy_reviewed")
        inputs = prepared.parent
    else:
        inputs = resolve_path(row["inputs_directory"], repo_root)
        prepared = inputs / "prepared_data"
        labels, policy = {}, "not_applicable"
    manifest = read_json(prepared / "manifest.json")
    bundle_manifest = inputs / "inputs_manifest.json"
    if bundle_manifest.is_file():
        expected = read_json(bundle_manifest).get("file_sha256", {})
        for relative, checksum in expected.items():
            path = inputs / str(relative).replace("\\", "/")
            if path.name in {"manifest.json", "sequences.npz", "catalog.json", "population.json"} and file_hash(path) != checksum:
                raise ValueError(f"Frozen input bytes changed: {path}")
    if dataset == "goodreads" and _defined(row.get("cohort_path")):
        cohort_hash = file_hash(resolve_path(row["cohort_path"], repo_root))
        if labels_file.get("cohort_fingerprint") != cohort_hash:
            raise ValueError("Gender annotation belongs to a different frozen reader cohort")
        if manifest.get("source_file_fingerprints", {}).get("cohort_json_sha256") != cohort_hash:
            raise ValueError("Prepared dataset belongs to a different frozen reader cohort")
    catalogue = {int(item): value for item, value in read_json(prepared / "catalog.json").items()}
    with np.load(prepared / "sequences.npz", allow_pickle=False) as arrays:
        popularity = dict(zip(map(int, arrays["popularity_item_ids"]), map(int, arrays["popularity_counts"])))
        users, offsets, items = arrays["user_ids"], arrays["training_offsets"], arrays["training_items"]
        histories = {int(uid): set(map(int, items[offsets[index]:offsets[index+1]])) for index, uid in enumerate(users)}
    if set(popularity) != set(catalogue):
        raise ValueError("Popularity reference and catalogue differ")
    if labels and set(labels) != set(catalogue):
        raise ValueError("Gender annotation IDs and cohort catalogue differ")
    if any(label not in GENDERS for label in labels.values()):
        raise ValueError("Unknown gender-label category")
    head = set(sorted(popularity, key=lambda item: (-popularity[item], item))[:math.ceil(.2 * len(catalogue))])
    clusters = {}
    population_path = inputs / "population.json"
    if population_path.is_file():
        population = read_json(population_path)
        population = population.get("population", population)
        clusters = {int(member["user_id"]): member.get("cluster_id") for member in population.get("members", [])}
    return {"catalogue": catalogue, "popularity": popularity, "histories": histories, "head": head,
            "labels": labels, "label_policy": policy, "clusters": clusters,
            "dataset_fingerprint": manifest["dataset_fingerprint"], "prepared": prepared,
            "labels_hash": file_hash(labels_path) if labels else None}


def _counter():
    return {"counts": Counter(), "shown": Counter(), "clicks": Counter(), "raw": Counter(),
            "weighted_labels": Counter(), "labels": Counter(), "click_labels": Counter(),
            "ranks": defaultdict(Counter), "arm_counts": defaultdict(Counter), "users": set(),
            "changed_users": set(), "modules": Counter(), "module_users": Counter(),
            "probe": Counter(), "tvd": [], "replica_tvd": [], "original_tvd": [],
            "cluster": Counter(), "eligible": Counter()}


def _probe_metrics(records, accum):
    canonical = records["canonical"]
    cv, cc, ci = action(canonical)
    if {"replica", "shuffled"} <= set(records):
        rv, rc, ri = action(records["replica"])
        sv, sc, si = action(records["shuffled"])
        triple = cv and cc and rv and rc and sv and sc
        p = accum["probe"]
        p["triple_valid_clicks"] += triple
        if triple:
            p["replica_same_item"] += ci == ri
            p["shuffled_same_item"] += ci == si
        for arm, valid, click, item in (("replica", rv, rc, ri), ("shuffled", sv, sc, si)):
            p[f"{arm}_pairs"] += 1
            p[f"{arm}_valid_pairs"] += cv and valid
            p[f"{arm}_excluded_non_valid_pairs"] += not (cv and valid)
            p[f"{arm}_both_clicks"] += cv and cc and valid and click
            p[f"{arm}_pair_same_item"] += cv and cc and valid and click and ci == item
            if cv and valid:
                p[f"{arm}_action_agreements"] += cc == click
                p[f"{arm}_action_item_agreements"] += cc == click and ci == item
                p[f"{arm}_both_no_clicks"] += not cc and not click
            if records[arm]["presented_slate"] != canonical["presented_slate"]:
                p[f"{arm}_changed_order"] += 1
        for arm, destination in (("shuffled", "tvd"), ("replica", "replica_tvd")):
            value = choice_tvd(canonical, records[arm])
            if math.isfinite(value):
                accum[destination].append(value)
    for arm in ("original_order_probe", "original_order"):
        if arm in records:
            value = choice_tvd(canonical, records[arm])
            if math.isfinite(value):
                accum["original_tvd"].append(value)
            valid, click, item = action(records[arm])
            accum["probe"]["original_valid_pairs"] += cv and valid
            accum["probe"]["original_same_item"] += cv and valid and cc and click and ci == item


def analyse_source(row, *, repo_root=REPO, reference=None):
    """Extract one completed physical trajectory with strict arm separation."""
    root = resolve_path(row["source_directory"], repo_root)
    marker = read_json(root / "_COMPLETE.json")
    envelope = read_json(root / "manifest.json")
    manifest = envelope.get("resolved_manifest", envelope)
    horizon, width = int(manifest["iterations"]), int(manifest["slate_size"])
    if marker.get("status") != "complete" or marker.get("completed_iteration") != horizon or (root / "_FAILED.json").exists():
        raise ValueError(f"Analysis requires a completed trajectory: {root}")
    for name in ("manifest.json", "diagnostics.jsonl"):
        if marker.get("file_sha256", {}).get(name) != file_hash(root / name):
            raise ValueError(f"Source checksum mismatch: {root / name}")
    ref = reference or _reference(row, repo_root)
    if manifest["inputs"]["dataset"] != ref["dataset_fingerprint"]:
        raise ValueError("Frozen dataset reference differs from run input fingerprint")
    catalogue, head, labels = ref["catalogue"], ref["head"], ref["labels"]
    if manifest.get("mitigation", {}).get("head_item_ids") is not None and set(manifest["mitigation"]["head_item_ids"]) != head:
        raise ValueError("Mitigation head definition differs from initial full training popularity")
    source_id = _field(row, "source_id", default=hashlib.sha256((str(root) + "\n" + file_hash(root / "_COMPLETE.json")).encode()).hexdigest()[:20])
    condition = manifest["condition"]
    tags = {"source_id": source_id, "dataset": row["dataset"], "study": _field(row, "study", default=""),
            "seed": int(manifest["simulation_seed"]), "population_seed": int(manifest["population_seed"]),
            "condition_id": condition["condition_id"], "model": condition["user_model"], "stage": condition["feedback_loop"],
            "intervention": condition.get("intervention", "control"), "user_update_policy": condition.get("user_update_policy", "frozen"),
            "run_id": envelope["run_id"], "physical_run_id": envelope["run_id"]}
    diagnostics = {int(value["iteration"]): value for value in jsonl_rows(root / "diagnostics.jsonl")
                   if value.get("event_name") == "iteration_complete"}
    if set(diagnostics) != set(range(1, horizon + 1)):
        raise ValueError("Missing iteration-complete diagnostics")
    arms = manifest.get("presentation_arms")
    if not arms:
        arms = ("canonical", "replica", "shuffled") if "probe" in condition.get("presentation_policy", "") else ("canonical",)
        if row["dataset"] == "movielens" and condition.get("intervention") == "post_processing":
            arms = (*arms,"original_order_probe")
    arms = tuple(arms)
    accum = {iteration: _counter() for iteration in range(1, horizon + 1)}
    result = {name: [] for name in TABLES}
    excluded = {uid: set(items) for uid, items in ref["histories"].items()}
    excluded_labels = {uid: Counter(labels.get(item, "unknown") for item in items) for uid, items in excluded.items()} if labels else {}
    catalogue_labels = Counter(labels.values())
    weights = np.array([rank_discount(rank) for rank in range(1, width + 1)])
    sponsor_schedule = []
    schedule_path = manifest.get("policy_settings", {}).get("sponsor_schedule_path")
    if schedule_path:
        schedule = read_json(resolve_path(schedule_path, repo_root))
        sponsor_schedule = schedule.get("item_ids", schedule.get("sponsored_item_ids", []))
        if not sponsor_schedule:
            sponsor_schedule = schedule.get("schedules", {}).get(str(manifest["population_seed"]), [])
        if not sponsor_schedule:
            raise ValueError("Advertising schedule is declared but its item IDs are absent")
    event_rows = marker.get("row_counts", {}).get("events")
    n_opportunities = 0
    for records in opportunities(root / "events.jsonl", expected_arms=arms, run_id=envelope["run_id"],
            horizon=horizon, slate_size=width, catalogue=set(catalogue),
            expected_hash=marker["file_sha256"]["events.jsonl"], expected_rows=event_rows):
        event = records["canonical"]
        iteration, uid = int(event["iteration"]), int(event["user_id"])
        a, c = accum[iteration], accum[iteration]["counts"]
        if uid in a["users"] or uid not in excluded:
            raise ValueError("Duplicate canonical user or user absent from frozen training reference")
        a["users"].add(uid)
        a["cluster"][ref["clusters"].get(uid, -1)] += 1
        shown, raw = event["presented_slate"], event["recommender_slate"]
        valid, click, clicked = action(event)
        c.update(opportunities=1, valid_actions=int(valid), valid_clicks=int(click), valid_no_clicks=int(valid and not click),
                 head_clicks=int(click and clicked in head), slots=len(shown), head_slots=sum(item in head for item in shown))
        c["head_weight"] += sum(weight for item, weight in zip(shown, weights) if item in head)
        c["total_weight"] += float(weights.sum())
        a["shown"].update(shown)
        a["raw"].update(raw)
        if click:
            a["clicks"][clicked] += 1
        # Candidate availability is measured BEFORE this canonical response.
        reserved = set(sponsor_schedule[iteration-1:]) if sponsor_schedule else set()
        candidate_excluded = excluded[uid] | reserved
        n_candidates = len(catalogue) - len(candidate_excluded)
        a["eligible"]["count"] += n_candidates
        a["eligible"]["head"] += ratio(len(head - candidate_excluded), n_candidates)
        if labels:
            removed = excluded_labels[uid].copy()
            removed.update(labels[item] for item in reserved - excluded[uid])
            for gender in GENDERS:
                a["eligible"][gender] += ratio(catalogue_labels[gender] - removed[gender], n_candidates)
            for item, weight in zip(shown, weights):
                a["labels"][labels[item]] += 1
                a["weighted_labels"][labels[item]] += weight
            if click:
                a["click_labels"][labels[clicked]] += 1
        for arm, record in records.items():
            av, ac, item = action(record)
            counter = a["arm_counts"][arm]
            counter.update(opportunities=1, valid_actions=int(av), clicks=int(ac), no_clicks=int(av and not ac))
            if ac:
                a["ranks"][arm][record["presented_slate"].index(item)+1] += 1
        _probe_metrics(records, a)
        trace = event["action"].get("diagnostics", {}).get("module_trace", [])
        invoked = [step.get("module") for step in trace if step.get("module")]
        a["modules"].update(invoked)
        a["module_users"].update(set(invoked))
        update = event.get("user_update") or {}
        update_diag = update.get("diagnostics", {})
        status = update_diag.get("reflection_status", "not_requested")
        c["reflection_requested"] += status != "not_requested"
        c["reflection_applied"] += status == "applied"
        c["reflection_failed"] += status == "failed_and_skipped"
        applied_change = update.get("status") == "applied" and "profile" in update.get("changed_domains", [])
        c["profile_changes"] += applied_change
        if applied_change:
            a["changed_users"].add(uid)
            result["profile_changes"].append({**tags, "iteration": iteration, "user_id": uid,
                "profile_fingerprint_before": update_diag.get("profile_fingerprint_before"),
                "profile_fingerprint_after": update_diag.get("profile_fingerprint"),
                "clicked_item_id": clicked, "memory_note": update_diag.get("memory_note", ""),
                "displayed_titles": " | ".join(catalogue[item].get("title", str(item)) for item in shown)})
        c["scheduled_departures"] += event.get("scheduled_exit_effective_iteration") is not None
        sponsor = event.get("presentation_metadata", {}).get("sponsored_item_id")
        delivered = sponsor is not None and sponsor in shown
        c["sponsored_exposures"] += delivered
        c["sponsored_clicks"] += delivered and click and clicked == sponsor
        c["sponsored_weight"] += weights[shown.index(sponsor)] if delivered else 0
        if click and clicked not in excluded[uid]:
            excluded[uid].add(clicked)
            if labels:
                excluded_labels[uid][labels[clicked]] += 1
        n_opportunities += 1
    cumulative_shown, cumulative_clicks, all_users, all_changed = Counter(), Counter(), set(), set()
    for iteration, a in accum.items():
        at = {**tags, "iteration": iteration}
        c, p = a["counts"], a["probe"]
        diag = diagnostics[iteration]
        actual = set(map(int, diag.get("active_user_ids", a["users"])))
        if actual != a["users"]:
            raise ValueError("Canonical opportunities differ from recorded active population")
        cumulative_shown.update(a["shown"])
        cumulative_clicks.update(a["clicks"])
        all_users.update(a["users"])
        all_changed.update(a["changed_users"])
        n, clicks, slots = c["opportunities"], c["valid_clicks"], c["slots"]
        metric = {**at, **{key: c[key] for key in ("opportunities", "valid_actions", "valid_clicks", "valid_no_clicks", "slots")},
            "invalid_responses": n-c["valid_actions"], "click_rate": ratio(clicks, c["valid_actions"]),
            "no_click_rate": ratio(c["valid_no_clicks"], c["valid_actions"]), "invalid_rate": ratio(n-c["valid_actions"], n),
            "clicks_per_opportunity": ratio(clicks, n), "head_click_share": ratio(c["head_clicks"], clicks),
            "head_exposure": ratio(c["head_slots"], slots), "weighted_head_exposure": ratio(c["head_weight"], c["total_weight"]),
            "exposure_coverage": ratio(len(a["shown"]), len(catalogue)), "click_coverage": ratio(len(a["clicks"]), len(catalogue)),
            "exposure_gini": normalized_gini(a["shown"], catalogue), "click_gini": normalized_gini(a["clicks"], catalogue),
            "cumulative_exposure_coverage": ratio(len(cumulative_shown), len(catalogue)),
            "cumulative_click_coverage": ratio(len(cumulative_clicks), len(catalogue)),
            "mean_clicked_rank": ratio(sum(rank*count for rank, count in a["ranks"]["canonical"].items()), clicks),
            "first_position_click_share": ratio(a["ranks"]["canonical"][1], clicks),
            "eligible_head_share": ratio(a["eligible"]["head"], n),
            "eligible_candidate_count_mean": ratio(a["eligible"]["count"], n)}
        metric["click_exposure_gap"] = metric["head_click_share"]-metric["head_exposure"]
        metric["weighted_click_exposure_gap"] = metric["head_click_share"]-metric["weighted_head_exposure"]
        if labels:
            for gender in GENDERS:
                metric[f"{gender}_exposure"] = ratio(a["labels"][gender], slots)
                metric[f"weighted_{gender}_exposure"] = ratio(a["weighted_labels"][gender], c["total_weight"])
                metric[f"{gender}_click_share"] = ratio(a["click_labels"][gender], clicks)
                metric[f"eligible_{gender}_share"] = ratio(a["eligible"][gender], n)
            metric["female_among_binary_clicks"] = ratio(a["click_labels"]["female"], a["click_labels"]["female"]+a["click_labels"]["male"])
            metric["female_among_binary_exposure"] = ratio(a["labels"]["female"], a["labels"]["female"]+a["labels"]["male"])
            metric["weighted_female_among_binary_exposure"] = ratio(a["weighted_labels"]["female"], a["weighted_labels"]["female"]+a["weighted_labels"]["male"])
            metric["female_click_exposure_gap"] = metric["female_click_share"]-metric["female_exposure"]
            metric["weighted_female_click_exposure_gap"] = metric["female_click_share"]-metric["weighted_female_exposure"]
            metric["binary_female_click_exposure_gap"] = metric["female_among_binary_clicks"]-metric["female_among_binary_exposure"]
            metric["weighted_binary_female_click_exposure_gap"] = metric["female_among_binary_clicks"]-metric["weighted_female_among_binary_exposure"]
            metric["female_exposure_minus_candidate_share"] = metric["female_exposure"]-metric["eligible_female_share"]
            metric["weighted_female_exposure_minus_candidate_share"] = metric["weighted_female_exposure"]-metric["eligible_female_share"]
        result["trajectory"].append(metric)
        for arm in arms:
            counter = a["arm_counts"][arm]
            for rank in range(1, width+1):
                result["ranks"].append({**at, "arm": arm, "rank": rank, "clicks": a["ranks"][arm][rank],
                    "click_share": ratio(a["ranks"][arm][rank], counter["clicks"]),
                    "opportunities": counter["opportunities"], "valid_clicks": counter["clicks"],
                    "valid_no_clicks": counter["no_clicks"], "valid_actions":counter["valid_actions"],
                    "click_rate":ratio(counter["clicks"],counter["valid_actions"]),
                    "no_click_rate":ratio(counter["no_clicks"],counter["valid_actions"]),
                    "invalid_rate":ratio(counter["opportunities"]-counter["valid_actions"],counter["opportunities"]),
                    "mean_clicked_rank": ratio(sum(rank*count for rank,count in a["ranks"][arm].items()),counter["clicks"])})
        probe = {**at, **dict(p), "triple_valid_clicks": p["triple_valid_clicks"],
            "replica_agreement": ratio(p["replica_same_item"], p["triple_valid_clicks"]),
            "shuffled_agreement": ratio(p["shuffled_same_item"], p["triple_valid_clicks"]),
            "shuffle_tvd": float(np.mean(a["tvd"])) if a["tvd"] else float("nan"),
            "replica_tvd": float(np.mean(a["replica_tvd"])) if a["replica_tvd"] else float("nan"),
            "original_order_tvd": float(np.mean(a["original_tvd"])) if a["original_tvd"] else float("nan"),
            "tvd_opportunities": len(a["tvd"])}
        probe["agreement_sensitivity"] = probe["replica_agreement"]-probe["shuffled_agreement"]
        for arm in ("replica", "shuffled"):
            for key in ("pairs","valid_pairs","excluded_non_valid_pairs","both_clicks","pair_same_item","action_agreements","action_item_agreements","both_no_clicks"):
                probe[f"{arm}_{key}"] = p[f"{arm}_{key}"]
            probe[f"{arm}_pair_agreement"] = ratio(p[f"{arm}_pair_same_item"],p[f"{arm}_both_clicks"])
            probe[f"{arm}_action_agreement"] = ratio(p[f"{arm}_action_agreements"], p[f"{arm}_valid_pairs"])
            probe[f"{arm}_action_disagreement"] = 1-probe[f"{arm}_action_agreement"]
            probe[f"{arm}_action_item_agreement"] = ratio(p[f"{arm}_action_item_agreements"], p[f"{arm}_valid_pairs"])
            probe[f"{arm}_both_no_click_share"] = ratio(p[f"{arm}_both_no_clicks"], p[f"{arm}_valid_pairs"])
        result["probes"].append(probe)
        result["agent"].append({**at, "opportunities": n, **{key:c[key] for key in
            ("reflection_requested", "reflection_applied", "reflection_failed", "profile_changes")},
            "reflection_error_rate": ratio(c["reflection_failed"], c["reflection_requested"]),
            "profile_change_rate": ratio(c["profile_changes"], n), "profile_changed_users": len(a["changed_users"]),
            "cumulative_profile_changed_users": len(all_changed), "memory_invocations": a["modules"]["memory"],
            "memory_invocation_rate": ratio(a["module_users"]["memory"],n)})
        for module in ("planning", "memory", "action", "reflection"):
            result["modules"].append({**at,"module":module,"invocations":a["modules"][module] if module!="reflection" else c["reflection_requested"],
                "decisions_invoking":a["module_users"][module] if module!="reflection" else c["reflection_requested"],
                "invocation_rate":ratio(a["module_users"][module] if module!="reflection" else c["reflection_requested"],n)})
        result["population"].append({**at, "active_users": n, "departures":len(diag.get("exited_user_ids",[])),
            "replacements":len(diag.get("activated_user_ids",[])), "scheduled_departures": c["scheduled_departures"],
            "cumulative_users": len(all_users)})
        for cluster in sorted(set(ref["clusters"].values()) or {-1}, key=str):
            result["clusters"].append({**at, "cluster":cluster,"active_users":a["cluster"][cluster],
                "active_share":ratio(a["cluster"][cluster],n)})
        result["advertising"].append({**at,"opportunities":n,"advertising_eligible_users":c["sponsored_exposures"],
            "advertising_eligibility_share":ratio(c["sponsored_exposures"],n),
            "sponsored_exposures":c["sponsored_exposures"],"sponsored_clicks":c["sponsored_clicks"],
            "sponsored_exposure_share":ratio(c["sponsored_exposures"],slots),
            "weighted_sponsored_exposure":ratio(c["sponsored_weight"],c["total_weight"]),
            "sponsored_click_share":ratio(c["sponsored_clicks"],clicks),
            "sponsored_click_through_rate":ratio(c["sponsored_clicks"],c["sponsored_exposures"])})
    reference_row = {**tags,"catalogue_size":len(catalogue),"head_count":len(head),
        "head_catalogue_share":ratio(len(head),len(catalogue)),
        "initial_training_head_share":ratio(sum(ref["popularity"][item] for item in head),sum(ref["popularity"].values())),
        "dataset_fingerprint":ref["dataset_fingerprint"], "label_policy":ref["label_policy"],"labels_sha256":ref["labels_hash"],
        "rank_discount":"1/log2(r+1)","rank_weight_normalisation":"sum over all displayed slots"}
    if labels:
        for gender in GENDERS:
            reference_row[f"catalogue_{gender}_share"] = ratio(catalogue_labels[gender],len(catalogue))
            reference_row[f"initial_training_{gender}_share"] = ratio(sum(ref["popularity"][item] for item in catalogue if labels[item]==gender),sum(ref["popularity"].values()))
    result["references"].append(reference_row)
    result["health"].append({**tags,"events_sha256_verified":True,"diagnostics_sha256_verified":True,
        "canonical_opportunities":n_opportunities,"iterations":horizon,"presentation_arms":";".join(arms),
        "complete_sha256":file_hash(root/"_COMPLETE.json"),"source_directory":str(root)})
    for rank, item in enumerate(sorted(cumulative_shown,key=lambda item:(-cumulative_shown[item],item))[:30],1):
        result["top_items"].append({**tags,"rank":rank,"item_id":item,"title":catalogue[item].get("title",str(item)),
            "presentations":cumulative_shown[item],"clicks":cumulative_clicks[item],"is_initial_head":item in head})
    if all_changed:
        result["profiles"] = _profiles(root, marker, tags, all_changed)
    return {name:pd.DataFrame(rows) for name,rows in result.items()}


def _profiles(root, marker, tags, changed_users):
    """Read trusted project checkpoint dictionaries for descriptive manual review."""
    checkpoint = root / str(marker["final_checkpoint_path"]).replace("\\", "/")
    if file_hash(checkpoint) != marker["final_checkpoint_sha256"]:
        raise ValueError("Final profile checkpoint checksum differs from completion marker")
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["checkpoint_state"]["component_states"]["user_model"]
    users = state.get("users",{})
    rows = []
    for uid in sorted(changed_users):
        user = users.get(uid,users.get(str(uid),{}))
        rows.append({**tags,"user_id":uid,"initial_profile":user.get("initial_profile_text"),
                     "final_profile":user.get("current_profile_text"),"review_classification":None,"review_notes":None})
    del payload
    gc.collect()
    return rows


def _cache_signature(row, repo_root):
    root = resolve_path(row["source_directory"],repo_root)
    paths = [root/"_COMPLETE.json",root/"manifest.json"]
    manifest=read_json(root/"manifest.json")
    manifest=manifest.get("resolved_manifest",manifest)
    schedule_path=manifest.get("policy_settings",{}).get("sponsor_schedule_path")
    if schedule_path:
        paths.append(resolve_path(schedule_path,repo_root))
    for field in ("inputs_directory","prepared_directory"):
        if _defined(row.get(field)):
            folder = resolve_path(row[field],repo_root)
            if (folder/"population.json").is_file():
                paths.append(folder/"population.json")
            if (folder/"inputs_manifest.json").is_file():
                paths.append(folder/"inputs_manifest.json")
            if (folder/"prepared_data").is_dir():
                folder /= "prepared_data"
            elif (folder/"prepared").is_dir():
                folder /= "prepared"
            paths.extend([folder/"manifest.json",folder/"sequences.npz",folder/"catalog.json"])
    if _defined(row.get("labels_path")):
        paths.append(resolve_path(row["labels_path"],repo_root))
    if _defined(row.get("cohort_path")):
        paths.append(resolve_path(row["cohort_path"],repo_root))
    digest = hashlib.sha256((VERSION+file_hash(__file__)+file_hash(Path(__file__).with_name("readers.py"))).encode())
    for path in paths:
        digest.update(str(path.resolve()).encode())
        digest.update(file_hash(path).encode())
    for name in ("events.jsonl","diagnostics.jsonl"):
        info=(root/name).stat()
        digest.update(f"{name}:{info.st_size}:{info.st_mtime_ns}".encode())
    return digest.hexdigest()


def build_analysis_frames(sources, *, repo_root=REPO, cache_dir=None, force=False, progress=True):
    """Aggregate each completed physical source once; logical aliases stay in inventory.

    Derived per-source CSVs are safe to replace and contain no model artifacts.
    All seven notebooks can subsequently read the shared assembled CSV snapshot.
    """
    source_frame = pd.DataFrame(sources).copy()
    if "status" in source_frame:
        source_frame=source_frame.loc[source_frame.status.eq("complete")]
    if "source_id" not in source_frame:
        raise ValueError("Source inventory must provide stable physical source_id values")
    selected=source_frame.drop_duplicates("source_id")
    tables={name:[] for name in TABLES}
    references={}
    cache_dir=Path(cache_dir) if cache_dir else None
    for index, row in enumerate(selected.to_dict("records"),1):
        key=row["source_id"]
        destination=cache_dir/"physical_sources"/str(key) if cache_dir else None
        signature=_cache_signature(row,repo_root) if cache_dir else None
        cached=False
        if destination and not force and (destination/"cache.json").is_file():
            meta=read_json(destination/"cache.json")
            if meta.get("signature")==signature:
                frames={}
                for name in TABLES:
                    try:
                        frames[name]=pd.read_csv(destination/f"{name}.csv", dtype={"source_id": str})
                    except pd.errors.EmptyDataError:
                        frames[name]=pd.DataFrame()
                cached=True
        if not cached:
            reference_key=(_field(row,"inputs_directory"),_field(row,"prepared_directory"),_field(row,"labels_path"))
            if reference_key not in references:
                references[reference_key]=_reference(row,repo_root)
            frames=analyse_source(row,repo_root=repo_root,reference=references[reference_key])
            if destination:
                destination.mkdir(parents=True,exist_ok=True)
                for name,frame in frames.items():
                    frame.to_csv(destination/f"{name}.csv",index=False)
                (destination/"cache.json").write_text(json.dumps({"version":VERSION,"signature":signature,
                    "source_id":key,"definition":"canonical-only trajectories; 1/log2(r+1); valid-response click rates; all-catalogue normalized Gini"},indent=2),encoding="utf-8")
        if progress:
            print(f"[{index}/{len(selected)}] {'cached' if cached else 'verified'} {row['dataset']} seed {row.get('seed',row.get('simulation_seed'))}: {row.get('logical_condition_id',row.get('condition_id'))}",flush=True)
        for name,frame in frames.items():
            if not frame.empty:
                tables[name].append(frame)
    return {name:pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=IDENTITY)
            for name,parts in tables.items()}


def with_memberships(frame, sources, *, study=None):
    """Attach explicit logical memberships without pretending aliases replicate data."""
    selected=pd.DataFrame(sources)
    if study is not None:
        selected=selected.loc[selected.study.eq(study)]
    fields=[column for column in ("source_id","study","logical_condition_id","condition_id","intervention","stage","feedback_stage","model","user_update_policy","seed","population_seed","relationship","alias_of") if column in selected]
    right=selected[fields].copy()
    for column in ("condition_id","stage"):
        fallback="logical_condition_id" if column=="condition_id" else "feedback_stage"
        if fallback in right:
            right[column]=right[fallback]
    left=frame.drop(columns=[column for column in right if column!="source_id" and column in frame])
    return left.merge(right,on="source_id",how="inner",validate="many_to_many")
