#!/usr/bin/env python3
"""
Find papers that cite each seed paper in a CSV such as seeds_main.csv using OpenAlex.

Input:
  - CSV with a DOI column.

Outputs:
  1) openalex_citing_papers_long.csv
     One row per (seed paper, citing paper) relation.
     This PRESERVES duplicates: if the same citing paper cites multiple seed
     papers, it appears multiple times, once for each seed paper.

  2) openalex_citing_papers_deduplicated_for_screening.csv
     One row per unique citing paper, with a list of all seed papers through
     which it was found.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from .query import run_collection, redact, record_response
except ImportError:
    from query import run_collection, redact, record_response



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CSV = Path(__file__).resolve().parent / "seeds_main.csv"

OUTPUT_LONG_CSV = Path("openalex_citing_papers_long.csv")
OUTPUT_DEDUP_CSV = Path("openalex_citing_papers_deduplicated_for_screening.csv")

OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()

# OpenAlex currently pages up to 100 works per page.
PER_PAGE = 100

# Gentle delay between requests. You can increase this if you hit rate limits.
REQUEST_SLEEP_SECONDS = 0.25

# Use these to mark seed papers that you already checked manually.
# Safer: put the exact DOI into MANUALLY_CHECKED_SEED_DOIS.
# Title patterns are regexes, case-insensitive.
MANUALLY_CHECKED_SEED_DOIS = {
    "10.1145/3626772.3657844"
}



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_doi(value: Any) -> str:
    """Return a lower-case DOI without URL or doi: prefix."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.I)
    s = re.sub(r"^doi:\s*", "", s, flags=re.I)
    return s.strip().lower()


def doi_as_openalex_filter_value(doi: str) -> str:
    """OpenAlex DOI filters work well with DOI URLs."""
    doi = normalise_doi(doi)
    return f"https://doi.org/{doi}"


def short_openalex_id(openalex_id_or_url: str) -> str:
    """Convert https://openalex.org/W123 to W123."""
    if not openalex_id_or_url:
        return ""
    return str(openalex_id_or_url).rstrip("/").split("/")[-1]


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """
    OpenAlex stores abstracts as an inverted index:
      {"word": [position1, position2, ...], ...}
    Convert that back to readable text.
    """
    if not inverted_index:
        return ""
    positioned_words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            positioned_words.append((pos, word))
    return " ".join(word for pos, word in sorted(positioned_words))


def normalise_title_for_key(title: Any) -> str:
    if pd.isna(title):
        return ""
    s = str(title).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()


def citing_duplicate_key(row: pd.Series) -> str:
    """
    Prefer DOI for de-duplication; fall back to OpenAlex ID; fall back to title.
    """
    doi = normalise_doi(row.get("citing_doi", ""))
    if doi:
        return f"doi:{doi}"

    openalex_id = short_openalex_id(row.get("citing_openalex_id", ""))
    if openalex_id:
        return f"openalex:{openalex_id}"

    title_key = normalise_title_for_key(row.get("citing_title", ""))
    return f"title:{title_key}"


def get_first_existing_column(df: pd.DataFrame, wanted_names: list[str]) -> str:
    """Find a column by case-insensitive name."""
    lookup = {c.lower().strip(): c for c in df.columns}
    for name in wanted_names:
        key = name.lower().strip()
        if key in lookup:
            return lookup[key]
    raise ValueError(
        f"Could not find any of these columns: {wanted_names}. "
        f"Available columns: {list(df.columns)}"
    )


def join_unique(values: pd.Series) -> str:
    cleaned = [str(v).strip() for v in values if not pd.isna(v) and str(v).strip()]
    return " | ".join(sorted(set(cleaned)))


def matches_manual_seed(title: str, doi: str) -> bool:
    doi_norm = normalise_doi(doi)
    if doi_norm in {normalise_doi(d) for d in MANUALLY_CHECKED_SEED_DOIS}:
        return True
    return False


# ---------------------------------------------------------------------------
# OpenAlex client
# ---------------------------------------------------------------------------

class OpenAlexClient:
    def __init__(self, api_key: str):
        self.base_url = "https://api.openalex.org"
        self.session = requests.Session()

        retry = Retry(
            total=6,
            connect=6,
            read=6,
            status=6,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.api_key = api_key

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params, timeout=60)

        if response.status_code == 403 and not self.api_key:
            raise RuntimeError(
                "OpenAlex returned 403 and OPENALEX_API_KEY is not set. "
                "Create a free OpenAlex API key, then run:\n"
                '  export OPENALEX_API_KEY="your_key"'
            )

        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenAlex request failed: HTTP {response.status_code}\n"
                f"URL: {redact(str(response.url))}\n"
                f"Response: {redact(response.text[:1000])}"
            )

        return record_response(url, params, response.json())

    def resolve_work_by_doi(self, doi: str) -> dict[str, Any] | None:
        data = self.get(
            "/works",
            params={
                "filter": f"doi:{doi_as_openalex_filter_value(doi)}",
                "per_page": 1,
                "select": "id,doi,title,display_name,cited_by_count,cited_by_api_url",
            },
        )
        results = data.get("results", [])
        return results[0] if results else None

    def iter_citing_works(self, seed_work_id: str):
        cursor = "*"

        while cursor:
            data = self.get(
                "/works",
                params={
                    "filter": f"cites:{short_openalex_id(seed_work_id)}",
                    "sort": "publication_date:desc",
                    "per_page": PER_PAGE,
                    "cursor": cursor,
                    "select": ",".join(
                        [
                            "id",
                            "doi",
                            "title",
                            "display_name",
                            "publication_year",
                            "publication_date",
                            "type",
                            "authorships",
                            "primary_location",
                            "best_oa_location",
                            "open_access",
                            "cited_by_count",
                            "abstract_inverted_index",
                        ]
                    ),
                },
            )

            results = data.get("results", [])
            if not results:
                break

            for work in results:
                yield work

            cursor = data.get("meta", {}).get("next_cursor")
            time.sleep(REQUEST_SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_source_name(work: dict[str, Any]) -> str:
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    return source.get("display_name") or ""


def extract_landing_page_url(work: dict[str, Any]) -> str:
    primary = work.get("primary_location") or {}
    return primary.get("landing_page_url") or ""


def extract_pdf_url(work: dict[str, Any]) -> str:
    best_oa = work.get("best_oa_location") or {}
    return best_oa.get("pdf_url") or ""


def extract_authors(work: dict[str, Any]) -> str:
    authors = []
    for authorship in work.get("authorships", []) or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append(name)
    return "; ".join(authors)


def extract_oa_status(work: dict[str, Any]) -> str:
    oa = work.get("open_access") or {}
    return oa.get("oa_status") or ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _collect() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV.resolve()}")

    if not OPENALEX_API_KEY:
        print(
            "WARNING: OPENALEX_API_KEY is not set. "
            "If OpenAlex returns 403, create a free API key and set the environment variable.",
            file=sys.stderr,
        )

    # sep=None with engine="python" auto-detects your semicolon-separated file.
    seeds = pd.read_csv(INPUT_CSV, sep=None, engine="python", encoding="utf-8-sig")

    doi_col = get_first_existing_column(seeds, ["doi", "DOI"])
    title_col = get_first_existing_column(seeds, ["title", "paper_title", "name"])
    included_col = None
    try:
        included_col = get_first_existing_column(
            seeds,
            ["Included through", "included_through", "source", "found_through"],
        )
    except ValueError:
        pass

    client = OpenAlexClient(api_key=OPENALEX_API_KEY)

    rows: list[dict[str, Any]] = []
    unresolved_seeds: list[dict[str, Any]] = []

    for seed_idx, seed in seeds.iterrows():
        seed_title = str(seed.get(title_col, "")).strip()
        seed_doi = normalise_doi(seed.get(doi_col, ""))
        seed_included_through = str(seed.get(included_col, "")).strip() if included_col else ""
        seed_index = int(seed_idx) + 1

        if not seed_doi:
            unresolved_seeds.append(
                {
                    "seed_index": seed_index,
                    "seed_title": seed_title,
                    "seed_doi": "",
                    "reason": "Missing DOI",
                }
            )
            continue

        print(f"[{seed_index}/{len(seeds)}] Resolving seed DOI: {seed_doi} — {seed_title}")

        seed_work = client.resolve_work_by_doi(seed_doi)
        if not seed_work:
            unresolved_seeds.append(
                {
                    "seed_index": seed_index,
                    "seed_title": seed_title,
                    "seed_doi": seed_doi,
                    "reason": "DOI not found in OpenAlex",
                }
            )
            continue

        seed_openalex_id = seed_work.get("id", "")
        seed_openalex_short_id = short_openalex_id(seed_openalex_id)
        seed_openalex_title = seed_work.get("title") or seed_work.get("display_name") or ""
        seed_manual = matches_manual_seed(seed_title or seed_openalex_title, seed_doi)

        print(
            f"    OpenAlex ID: {seed_openalex_short_id}; "
            f"OpenAlex cited_by_count: {seed_work.get('cited_by_count', '')}"
        )

        count_for_seed = 0
        for citing in client.iter_citing_works(seed_openalex_short_id):
            count_for_seed += 1

            rows.append(
                {
                    # The seed/origin paper. This is the important "found through" trace.
                    "found_through_seed_index": seed_index,
                    "found_through_seed_title": seed_title,
                    "found_through_seed_doi": seed_doi,
                    "found_through_seed_included_through": seed_included_through,
                    "found_through_seed_openalex_id": seed_openalex_short_id,
                    "found_through_seed_openalex_title": seed_openalex_title,
                    "found_through_manually_checked_seed": seed_manual,

                    # The citing paper.
                    "citing_openalex_id": short_openalex_id(citing.get("id", "")),
                    "citing_title": citing.get("title") or citing.get("display_name") or "",
                    "citing_doi": normalise_doi(citing.get("doi", "")),
                    "citing_publication_year": citing.get("publication_year", ""),
                    "citing_publication_date": citing.get("publication_date", ""),
                    "citing_type": citing.get("type", ""),
                    "citing_authors": extract_authors(citing),
                    "citing_source": extract_source_name(citing),
                    "citing_landing_page_url": extract_landing_page_url(citing),
                    "citing_pdf_url": extract_pdf_url(citing),
                    "citing_open_access_status": extract_oa_status(citing),
                    "citing_cited_by_count": citing.get("cited_by_count", ""),
                    "citing_abstract": reconstruct_abstract(citing.get("abstract_inverted_index")),
                }
            )

        print(f"    Found {count_for_seed} citing works via OpenAlex.")
        time.sleep(REQUEST_SLEEP_SECONDS)

    if not rows:
        print("No citing papers found. Writing unresolved seed report only.")
        pd.DataFrame(unresolved_seeds).to_csv(OUTPUT_UNRESOLVED_CSV, index=False)
        return

    long_df = pd.DataFrame(rows)
    long_df["citing_duplicate_key"] = long_df.apply(citing_duplicate_key, axis=1)

    group_info = (
        long_df.groupby("citing_duplicate_key", dropna=False)
        .agg(
            duplicate_group_row_count=("citing_duplicate_key", "size"),
            duplicate_group_seed_count=("found_through_seed_doi", lambda x: len(set(x))),
            all_found_through_seed_indices=("found_through_seed_index", join_unique),
            all_found_through_seed_titles=("found_through_seed_title", join_unique),
            all_found_through_seed_dois=("found_through_seed_doi", join_unique),
            found_also_through_manually_checked_seed=("found_through_manually_checked_seed", "max"),
        )
        .reset_index()
    )

    long_df = long_df.merge(group_info, on="citing_duplicate_key", how="left")
    long_df["is_duplicate_across_seed_papers"] = long_df["duplicate_group_seed_count"] > 1

    def manual_overlap_status(row: pd.Series) -> str:
        if bool(row["found_through_manually_checked_seed"]):
            return "found directly through manually checked seed"
        if bool(row["found_also_through_manually_checked_seed"]):
            return "duplicate of manually checked seed result"
        return ""

    long_df["manual_checked_overlap_status"] = long_df.apply(manual_overlap_status, axis=1)

    # Useful sort order: things already seen through a manually checked seed first,
    # then duplicates, then newest papers.
    long_df = long_df.sort_values(
        by=[
            "found_also_through_manually_checked_seed",
            "is_duplicate_across_seed_papers",
            "citing_publication_year",
            "citing_title",
            "found_through_seed_index",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    )

    long_df.to_csv(OUTPUT_LONG_CSV, index=False, encoding="utf-8-sig")

    # Deduplicated screening sheet: one row per unique citing paper.
    first_cols = [
        "citing_duplicate_key",
        "citing_openalex_id",
        "citing_title",
        "citing_doi",
        "citing_publication_year",
        "citing_publication_date",
        "citing_type",
        "citing_authors",
        "citing_source",
        "citing_landing_page_url",
        "citing_pdf_url",
        "citing_open_access_status",
        "citing_cited_by_count",
        "citing_abstract",
    ]

    dedup_df = (
        long_df.sort_values(
            by=["citing_publication_year", "citing_cited_by_count"],
            ascending=[False, False],
            kind="stable",
        )
        .groupby("citing_duplicate_key", as_index=False)
        .first()[first_cols]
    )

    dedup_group_info = group_info.copy()
    dedup_df = dedup_df.merge(dedup_group_info, on="citing_duplicate_key", how="left")
    dedup_df["is_duplicate_across_seed_papers"] = dedup_df["duplicate_group_seed_count"] > 1
    dedup_df.to_csv(OUTPUT_DEDUP_CSV, index=False, encoding="utf-8-sig")

    if unresolved_seeds:
        pd.DataFrame(unresolved_seeds).to_csv(OUTPUT_UNRESOLVED_CSV, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Long duplicate-preserving output: {OUTPUT_LONG_CSV.resolve()}")
    print(f"Deduplicated screening output:    {OUTPUT_DEDUP_CSV.resolve()}")
    if unresolved_seeds:
        print("Some seed DOIs could not be resolved; see openalex_unresolved_seeds.csv")


def main(argv=None):
    return run_collection(globals(), argv, combined=False)


if __name__ == "__main__":
    raise SystemExit(main())
