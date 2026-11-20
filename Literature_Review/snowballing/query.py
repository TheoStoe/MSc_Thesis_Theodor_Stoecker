#!/usr/bin/env python3
"""
Forward snowballing from a CSV of seed papers using BOTH OpenAlex and Semantic Scholar.

Input:
  - seeds_main.csv or seeds_additional.csv with a DOI column. The script auto-detects separator, so
    semicolon-separated CSVs work as-is.

Outputs:
  1) combined_citing_papers_long.csv
     One row per (database, seed paper, citing paper) relation.
     This PRESERVES duplicates. If the same citing paper is found via multiple
     seed papers and/or multiple databases, it appears multiple times.

  2) combined_citing_papers_deduplicated_for_screening.csv
     One row per unique citing paper, with all databases and seed papers through
     which it was found.

  3) combined_unresolved_seeds.csv
     Seed DOIs that could not be resolved in OpenAlex and/or Semantic Scholar.

Optional API keys:
  export OPENALEX_API_KEY="your_openalex_key"
  export SEMANTIC_SCHOLAR_API_KEY="your_semantic_scholar_key"

Notes:
  - API access requirements may change; optional keys are read from the environment.
  - Do not put keys directly in this file if you plan to share the script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


HERE = Path(__file__).resolve().parent


def redact(value: str) -> str:
    text = str(value)
    for name in ("OPENALEX_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"):
        secret = os.environ.get(name, "").strip()
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return re.sub(r"(?i)(api[_-]?key=)[^&\s]+", r"\1[REDACTED]", text)


def record_response(url, params, payload):
    return payload


def read_saved_csv(path):
    first = path.read_text(encoding="utf-8-sig").splitlines()[0]
    return pd.read_csv(path, sep=";" if first.count(";") > first.count(",") else ",", encoding="utf-8-sig")


def historical_summary():
    """Summarise the saved searches without querying either database."""
    results = HERE / "results"
    relations = read_saved_csv(results / "citation_relations.csv")
    papers = read_saved_csv(results / "screening_papers.csv")
    return {
        "relations_by_database": {key: int(value) for key, value in relations["found_via_database"].value_counts().items()},
        "main_relations": int(relations["search_batch"].eq("main").sum()),
        "additional_relations": int(relations["search_batch"].eq("additional").sum()),
        "total_relations": len(relations),
        "main_papers": int(papers["search_batch"].eq("main").sum()),
        "additional_papers": int(papers["search_batch"].eq("additional").sum()),
        "pooled_papers": int(papers["citing_duplicate_key"].nunique()),
    }


def run_collection(namespace, argv=None, *, combined=True):
    parser = argparse.ArgumentParser(description="Describe a citation search, summarise saved results, or run a new query with --run.")
    parser.add_argument("--input", type=Path, default=HERE / "seeds_main.csv", help="CSV of seed papers with title and DOI columns")
    parser.add_argument("--output", type=Path, help="New destination directory, required with --run")
    parser.add_argument("--summary", action="store_true", help="Summarise the saved historical results without API requests")
    if combined:
        parser.add_argument("--databases", choices=("both", "openalex", "semantic-scholar"), default="both")
    parser.add_argument("--run", action="store_true", help="Explicitly query the current APIs")
    args = parser.parse_args(argv)
    if args.summary:
        if args.run:
            parser.error("Choose either --summary or --run")
        print(json.dumps(historical_summary(), indent=2, ensure_ascii=False))
        return 0
    path = args.input.resolve()
    if not path.is_file():
        parser.error(f"Seed input does not exist: {args.input}")
    seeds = read_saved_csv(path)
    doi_col = namespace["get_first_existing_column"](seeds, ["doi"])
    title_col = namespace["get_first_existing_column"](seeds, ["title", "paper_title", "name"])
    seed_rows = [{"title": str(row[title_col]), "doi": namespace["normalise_doi"](row[doi_col])} for _, row in seeds.iterrows()]
    if any(not row["doi"] for row in seed_rows):
        parser.error("Every seed must have a DOI")
    mode = args.databases if combined else "openalex"
    if not args.run:
        print(json.dumps({"mode": "dry-run", "databases": mode, "seed_count": len(seed_rows), "seeds": seed_rows,
                          "note": "No API requests. Supply --run and a new --output directory for a current search."}, indent=2, ensure_ascii=False))
        return 0
    if args.output is None:
        parser.error("--run requires --output pointing to a new directory")
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    if output == HERE / "results" or output.is_relative_to(HERE / "results"):
        parser.error("Keep new searches outside the historical results directory")
    output.mkdir(parents=True)
    prefix = "combined" if combined else "openalex"
    namespace.update(INPUT_CSV=path,
                     OUTPUT_LONG_CSV=output / f"{prefix}_citing_papers_long.csv",
                     OUTPUT_DEDUP_CSV=output / f"{prefix}_citing_papers_deduplicated_for_screening.csv",
                     OUTPUT_UNRESOLVED_CSV=output / f"{prefix}_unresolved_seeds.csv")
    if combined:
        namespace["USE_OPENALEX"] = mode in {"both", "openalex"}
        namespace["USE_SEMANTIC_SCHOLAR"] = mode in {"both", "semantic-scholar"}
    try:
        namespace["_collect"]()
    except Exception as exc:
        raise RuntimeError(redact(str(exc))) from None
    return 0




# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CSV = HERE / "seeds_main.csv"

OUTPUT_LONG_CSV = Path("new_combined_citing_papers_long.csv")
OUTPUT_DEDUP_CSV = Path("new_combined_citing_papers_deduplicated_for_screening.csv")
OUTPUT_UNRESOLVED_CSV = Path("new_combined_unresolved_seeds.csv")

USE_OPENALEX = True
USE_SEMANTIC_SCHOLAR = True

OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()

# OpenAlex currently pages up to 100 works per page.
OPENALEX_PER_PAGE = 100

# Semantic Scholar citations endpoint supports limit/offset pagination.
# 100 is a safe page size.
S2_PAGE_SIZE = 100

S2_MAX_CITATIONS_PER_SEED = 9999

# Gentle delays between requests. Increase if you hit 429 rate-limit errors.
OPENALEX_SLEEP_SECONDS = 0.15
S2_SLEEP_SECONDS_WITHOUT_KEY = 1.1
S2_SLEEP_SECONDS_WITH_KEY = 1.1  # keep conservative; many S2 keys are still about 1 req/s

MANUALLY_CHECKED_SEED_DOIS = {
}


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def normalise_doi(value: Any) -> str:
    """Return a lower-case DOI without URL or doi: prefix."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.I)
    s = re.sub(r"^doi:\s*", "", s, flags=re.I)
    return s.strip().lower()


MANUALLY_CHECKED_SEED_DOIS_NORMALISED = {
    normalise_doi(d) for d in MANUALLY_CHECKED_SEED_DOIS if normalise_doi(d)
}


def matches_manual_seed(seed_doi: str) -> bool:
    """Mark manually checked seed papers by DOI only."""
    doi_norm = normalise_doi(seed_doi)
    return bool(doi_norm and doi_norm in MANUALLY_CHECKED_SEED_DOIS_NORMALISED)


def doi_as_url(doi: str) -> str:
    doi = normalise_doi(doi)
    return f"https://doi.org/{doi}" if doi else ""


def short_openalex_id(openalex_id_or_url: Any) -> str:
    """Convert https://openalex.org/W123 to W123."""
    if not openalex_id_or_url or pd.isna(openalex_id_or_url):
        return ""
    return str(openalex_id_or_url).rstrip("/").split("/")[-1]


def normalise_title_for_key(title: Any) -> str:
    if pd.isna(title):
        return ""
    s = str(title).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()


def reconstruct_openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str:
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


def join_unique_preserve_order(values: pd.Series) -> str:
    seen = set()
    out = []
    for value in values:
        if pd.isna(value):
            continue
        s = str(value).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return " | ".join(out)


def first_non_empty(values: pd.Series) -> str:
    for value in values:
        if not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return ""


def citing_duplicate_key(row: pd.Series) -> str:
    """
    Prefer DOI for de-duplication across databases.
    Fall back to Semantic Scholar ID, then OpenAlex ID, then normalized title.

    Title fallback is imperfect, but useful because many fresh citing papers may
    not have a DOI in one or both databases.
    """
    doi = normalise_doi(row.get("citing_doi", ""))
    if doi:
        return f"doi:{doi}"

    s2_id = str(row.get("citing_semantic_scholar_id", "") or "").strip()
    if s2_id:
        return f"s2:{s2_id}"

    openalex_id = short_openalex_id(row.get("citing_openalex_id", ""))
    if openalex_id:
        return f"openalex:{openalex_id}"

    title_key = normalise_title_for_key(row.get("citing_title", ""))
    return f"title:{title_key}" if title_key else "unknown"


# ---------------------------------------------------------------------------
# HTTP session helper
# ---------------------------------------------------------------------------

def make_retrying_session() -> requests.Session:
    session = requests.Session()
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
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# OpenAlex client and extraction
# ---------------------------------------------------------------------------

class OpenAlexClient:
    def __init__(self, api_key: str):
        self.base_url = "https://api.openalex.org"
        self.session = make_retrying_session()
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
        doi = normalise_doi(doi)
        if not doi:
            return None
        data = self.get(
            "/works",
            params={
                "filter": f"doi:{doi_as_url(doi)}",
                "per_page": 1,
                "select": "id,doi,title,display_name,cited_by_count,cited_by_api_url",
            },
        )
        results = data.get("results", [])
        return results[0] if results else None

    def iter_citing_works(self, seed_work_id: str) -> Iterator[dict[str, Any]]:
        cursor = "*"

        while cursor:
            data = self.get(
                "/works",
                params={
                    "filter": f"cites:{short_openalex_id(seed_work_id)}",
                    "sort": "publication_date:desc",
                    "per_page": OPENALEX_PER_PAGE,
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
            time.sleep(OPENALEX_SLEEP_SECONDS)


def oa_extract_source_name(work: dict[str, Any]) -> str:
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    return source.get("display_name") or ""


def oa_extract_landing_page_url(work: dict[str, Any]) -> str:
    primary = work.get("primary_location") or {}
    return primary.get("landing_page_url") or ""


def oa_extract_pdf_url(work: dict[str, Any]) -> str:
    best_oa = work.get("best_oa_location") or {}
    return best_oa.get("pdf_url") or ""


def oa_extract_authors(work: dict[str, Any]) -> str:
    authors = []
    for authorship in work.get("authorships", []) or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append(name)
    return "; ".join(authors)


def oa_extract_oa_status(work: dict[str, Any]) -> str:
    oa = work.get("open_access") or {}
    return oa.get("oa_status") or ""


# ---------------------------------------------------------------------------
# Semantic Scholar client and extraction
# ---------------------------------------------------------------------------

class SemanticScholarClient:
    def __init__(self, api_key: str):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.session = requests.Session()
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key
        self.sleep_seconds = S2_SLEEP_SECONDS_WITH_KEY if api_key else S2_SLEEP_SECONDS_WITHOUT_KEY

    def get(
            self,
            endpoint: str,
            params: dict[str, Any] | None = None,
            allow_404: bool = False
        ) -> dict[str, Any] | None:
        url = f"{self.base_url}{endpoint}"
        last_error = None

        for attempt in range(8):
            try:
                print(
                    f"    Semantic Scholar request attempt {attempt + 1}/8: {url}",
                    flush=True,
                )

                response = self.session.get(
                    url,
                    params=params or {},
                    headers=self.headers,
                    timeout=(10, 25),  # connect timeout, read timeout
                )

                print(
                    f"    Semantic Scholar response: HTTP {response.status_code}",
                    flush=True,
                )
                time.sleep(self.sleep_seconds)
            except requests.exceptions.Timeout as exc:
                last_error = exc
                sleep_for = min(60, 5 * (attempt + 1))
                print(
                    f"    Semantic Scholar timeout; sleeping {sleep_for}s...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(sleep_for)
                continue

            except requests.exceptions.RequestException as exc:
                last_error = exc
                sleep_for = min(60, 5 * (attempt + 1))
                print(
                    f"    Semantic Scholar request error: {redact(str(exc))}; sleeping {sleep_for}s...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(sleep_for)
                continue

            if response.status_code == 404 and allow_404:
                return None

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_for = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(90, 15 * (attempt + 1))
                )
                print(
                    f"    Semantic Scholar rate limited; sleeping {sleep_for:.0f}s...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(sleep_for)
                continue

            if response.status_code in {500, 502, 503, 504}:
                sleep_for = min(90, 10 * (attempt + 1))
                print(
                    f"    Semantic Scholar server error HTTP {response.status_code}; sleeping {sleep_for}s...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(sleep_for)
                continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Semantic Scholar request failed: HTTP {response.status_code}\n"
                    f"URL: {redact(str(response.url))}\n"
                    f"Response: {redact(response.text[:1000])}"
                )

            return record_response(url, params, response.json())

        raise RuntimeError(
            f"Semantic Scholar request failed repeatedly: {url}\n"
            f"Last error: {redact(str(last_error))}"
        )
    
    @staticmethod
    def doi_to_path_id(doi: str) -> str:
        # S2 accepts external IDs in the path, e.g. DOI:10.xxxx/yyyy.
        # Encode ':' and '/' because this is a path segment.
        return quote(f"DOI:{normalise_doi(doi)}", safe="")

    def resolve_paper_by_doi(self, doi: str) -> dict[str, Any] | None:
        doi = normalise_doi(doi)
        if not doi:
            return None
        path_id = self.doi_to_path_id(doi)
        return self.get(
            f"/paper/{path_id}",
            params={
                "fields": ",".join(
                    [
                        "paperId",
                        "externalIds",
                        "url",
                        "title",
                        "year",
                        "publicationDate",
                        "venue",
                        "citationCount",
                    ]
                )
            },
            allow_404=True,
        )

    def iter_citing_papers(self, seed_s2_paper_id: str) -> Iterator[dict[str, Any]]:
        offset = 0

        fields = ",".join(
            [
                "contexts",
                "intents",
                "isInfluential",
                "citingPaper.paperId",
                "citingPaper.externalIds",
                "citingPaper.url",
                "citingPaper.title",
                "citingPaper.abstract",
                "citingPaper.year",
                "citingPaper.publicationDate",
                "citingPaper.venue",
                "citingPaper.authors",
                "citingPaper.citationCount",
            ]
        )

        while offset < S2_MAX_CITATIONS_PER_SEED:
            data = self.get(
                f"/paper/{seed_s2_paper_id}/citations",
                params={
                    "fields": fields,
                    "limit": S2_PAGE_SIZE,
                    "offset": offset,
                },
            )
            if data is None:
                break

            items = data.get("data", [])
            if not items:
                break

            for item in items:
                yield item

            next_offset = data.get("next")
            if next_offset is None:
                break
            offset = int(next_offset)
            time.sleep(self.sleep_seconds)


def s2_extract_doi(paper: dict[str, Any]) -> str:
    external_ids = paper.get("externalIds") or {}
    return normalise_doi(external_ids.get("DOI", ""))


def s2_extract_authors(paper: dict[str, Any]) -> str:
    authors = []
    for author in paper.get("authors", []) or []:
        name = author.get("name")
        if name:
            authors.append(name)
    return "; ".join(authors)


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def make_seed_base(
    *,
    seed_index: int,
    seed_title: str,
    seed_doi: str,
    seed_included_through: str,
    seed_manual: bool,
) -> dict[str, Any]:
    return {
        "found_through_seed_index": seed_index,
        "found_through_seed_title": seed_title,
        "found_through_seed_doi": seed_doi,
        "found_through_seed_included_through": seed_included_through,
        "found_through_manually_checked_seed": seed_manual,
    }


def row_from_openalex_citing_work(seed_base: dict[str, Any], seed_work: dict[str, Any], citing: dict[str, Any]) -> dict[str, Any]:
    return {
        **seed_base,
        "found_via_database": "OpenAlex",
        "found_through_seed_openalex_id": short_openalex_id(seed_work.get("id", "")),
        "found_through_seed_semantic_scholar_id": "",
        "found_through_seed_database_title": seed_work.get("title") or seed_work.get("display_name") or "",
        "seed_database_reported_citation_count": seed_work.get("cited_by_count", ""),

        "citing_openalex_id": short_openalex_id(citing.get("id", "")),
        "citing_semantic_scholar_id": "",
        "citing_title": citing.get("title") or citing.get("display_name") or "",
        "citing_doi": normalise_doi(citing.get("doi", "")),
        "citing_publication_year": citing.get("publication_year", ""),
        "citing_publication_date": citing.get("publication_date", ""),
        "citing_type": citing.get("type", ""),
        "citing_authors": oa_extract_authors(citing),
        "citing_source": oa_extract_source_name(citing),
        "citing_landing_page_url": oa_extract_landing_page_url(citing),
        "citing_pdf_url": oa_extract_pdf_url(citing),
        "citing_open_access_status": oa_extract_oa_status(citing),
        "citing_cited_by_count": citing.get("cited_by_count", ""),
        "citing_abstract": reconstruct_openalex_abstract(citing.get("abstract_inverted_index")),
        "semantic_scholar_intents": "",
        "semantic_scholar_is_influential": "",
        "semantic_scholar_contexts": "",
    }


def row_from_s2_citation(seed_base: dict[str, Any], seed_paper: dict[str, Any], citation_item: dict[str, Any]) -> dict[str, Any]:
    citing = citation_item.get("citingPaper") or {}
    return {
        **seed_base,
        "found_via_database": "Semantic Scholar",
        "found_through_seed_openalex_id": "",
        "found_through_seed_semantic_scholar_id": seed_paper.get("paperId", ""),
        "found_through_seed_database_title": seed_paper.get("title", ""),
        "seed_database_reported_citation_count": seed_paper.get("citationCount", ""),

        "citing_openalex_id": "",
        "citing_semantic_scholar_id": citing.get("paperId", ""),
        "citing_title": citing.get("title", ""),
        "citing_doi": s2_extract_doi(citing),
        "citing_publication_year": citing.get("year", ""),
        "citing_publication_date": citing.get("publicationDate", ""),
        "citing_type": "",
        "citing_authors": s2_extract_authors(citing),
        "citing_source": citing.get("venue", ""),
        "citing_landing_page_url": citing.get("url", ""),
        "citing_pdf_url": "",
        "citing_open_access_status": "",
        "citing_cited_by_count": citing.get("citationCount", ""),
        "citing_abstract": citing.get("abstract", ""),
        "semantic_scholar_intents": "; ".join(citation_item.get("intents") or []),
        "semantic_scholar_is_influential": citation_item.get("isInfluential", ""),
        "semantic_scholar_contexts": " | ".join(citation_item.get("contexts") or []),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _collect() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV.resolve()}")

    if USE_OPENALEX and not OPENALEX_API_KEY:
        print(
            "WARNING: OPENALEX_API_KEY is not set. OpenAlex may still work, "
            "but setting a key is recommended.",
            file=sys.stderr,
        )

    if USE_SEMANTIC_SCHOLAR and not SEMANTIC_SCHOLAR_API_KEY:
        print(
            "INFO: SEMANTIC_SCHOLAR_API_KEY is not set. Semantic Scholar allows "
            "many public endpoints without a key, but this script will use a low request rate.",
            file=sys.stderr,
        )

    # sep=None with engine="python" auto-detects semicolon-separated files.
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

    oa_client = OpenAlexClient(api_key=OPENALEX_API_KEY) if USE_OPENALEX else None
    s2_client = SemanticScholarClient(api_key=SEMANTIC_SCHOLAR_API_KEY) if USE_SEMANTIC_SCHOLAR else None

    rows: list[dict[str, Any]] = []
    unresolved_seeds: list[dict[str, Any]] = []

    for seed_idx, seed in seeds.iterrows():
        seed_index = int(seed_idx) + 1
        seed_title = str(seed.get(title_col, "")).strip()
        seed_doi = normalise_doi(seed.get(doi_col, ""))
        seed_included_through = str(seed.get(included_col, "")).strip() if included_col else ""
        seed_manual = matches_manual_seed(seed_doi)

        print(f"\n[{seed_index}/{len(seeds)}] Seed DOI: {seed_doi} — {seed_title}")

        if not seed_doi:
            unresolved_seeds.append(
                {
                    "seed_index": seed_index,
                    "seed_title": seed_title,
                    "seed_doi": "",
                    "database": "all",
                    "reason": "Missing DOI",
                }
            )
            continue

        seed_base = make_seed_base(
            seed_index=seed_index,
            seed_title=seed_title,
            seed_doi=seed_doi,
            seed_included_through=seed_included_through,
            seed_manual=seed_manual,
        )

        # -----------------------------
        # OpenAlex
        # -----------------------------
        if oa_client is not None:
            print("  Resolving in OpenAlex...")
            try:
                oa_seed = oa_client.resolve_work_by_doi(seed_doi)
                if not oa_seed:
                    unresolved_seeds.append(
                        {
                            "seed_index": seed_index,
                            "seed_title": seed_title,
                            "seed_doi": seed_doi,
                            "database": "OpenAlex",
                            "reason": "DOI not found in OpenAlex",
                        }
                    )
                else:
                    oa_seed_id = short_openalex_id(oa_seed.get("id", ""))
                    print(
                        f"    OpenAlex ID: {oa_seed_id}; "
                        f"reported cited_by_count: {oa_seed.get('cited_by_count', '')}"
                    )
                    count = 0
                    for citing in oa_client.iter_citing_works(oa_seed_id):
                        rows.append(row_from_openalex_citing_work(seed_base, oa_seed, citing))
                        count += 1
                    print(f"    Found {count} citing works via OpenAlex.")
                    time.sleep(OPENALEX_SLEEP_SECONDS)
            except Exception as exc:
                unresolved_seeds.append(
                    {
                        "seed_index": seed_index,
                        "seed_title": seed_title,
                        "seed_doi": seed_doi,
                        "database": "OpenAlex",
                        "reason": f"Error: {redact(str(exc))}",
                    }
                )
                print(f"    OpenAlex error: {redact(str(exc))}", file=sys.stderr)

        # -----------------------------
        # Semantic Scholar
        # -----------------------------
        if s2_client is not None:
            print("  Resolving in Semantic Scholar...")
            try:
                s2_seed = s2_client.resolve_paper_by_doi(seed_doi)
                if not s2_seed:
                    unresolved_seeds.append(
                        {
                            "seed_index": seed_index,
                            "seed_title": seed_title,
                            "seed_doi": seed_doi,
                            "database": "Semantic Scholar",
                            "reason": "DOI not found in Semantic Scholar",
                        }
                    )
                else:
                    s2_seed_id = s2_seed.get("paperId", "")
                    print(
                        f"    Semantic Scholar ID: {s2_seed_id}; "
                        f"reported citationCount: {s2_seed.get('citationCount', '')}"
                    )
                    count = 0
                    for citation_item in s2_client.iter_citing_papers(s2_seed_id):
                        rows.append(row_from_s2_citation(seed_base, s2_seed, citation_item))
                        count += 1
                    print(f"    Found {count} citing works via Semantic Scholar.")
                    time.sleep(s2_client.sleep_seconds)
            except Exception as exc:
                unresolved_seeds.append(
                    {
                        "seed_index": seed_index,
                        "seed_title": seed_title,
                        "seed_doi": seed_doi,
                        "database": "Semantic Scholar",
                        "reason": f"Error: {redact(str(exc))}",
                    }
                )
                print(f"    Semantic Scholar error: {redact(str(exc))}", file=sys.stderr)

    if unresolved_seeds:
        pd.DataFrame(unresolved_seeds).to_csv(OUTPUT_UNRESOLVED_CSV, index=False, encoding="utf-8-sig")

    if not rows:
        print("\nNo citing papers found.")
        if unresolved_seeds:
            print(f"Unresolved seed report: {OUTPUT_UNRESOLVED_CSV.resolve()}")
        return

    long_df = pd.DataFrame(rows)
    long_df["citing_duplicate_key"] = long_df.apply(citing_duplicate_key, axis=1)

    group_info = (
        long_df.groupby("citing_duplicate_key", dropna=False)
        .agg(
            duplicate_group_row_count=("citing_duplicate_key", "size"),
            duplicate_group_seed_count=("found_through_seed_doi", lambda x: len(set(v for v in x if str(v).strip()))),
            duplicate_group_database_count=("found_via_database", lambda x: len(set(v for v in x if str(v).strip()))),
            all_found_via_databases=("found_via_database", join_unique_preserve_order),
            all_found_through_seed_indices=("found_through_seed_index", join_unique_preserve_order),
            all_found_through_seed_titles=("found_through_seed_title", join_unique_preserve_order),
            all_found_through_seed_dois=("found_through_seed_doi", join_unique_preserve_order),
            found_also_through_manually_checked_seed=("found_through_manually_checked_seed", "max"),
        )
        .reset_index()
    )

    long_df = long_df.merge(group_info, on="citing_duplicate_key", how="left")
    long_df["is_duplicate_across_seed_papers"] = long_df["duplicate_group_seed_count"] > 1
    long_df["is_duplicate_across_databases"] = long_df["duplicate_group_database_count"] > 1

    def manual_overlap_status(row: pd.Series) -> str:
        if bool(row["found_through_manually_checked_seed"]):
            return "found directly through manually checked seed"
        if bool(row["found_also_through_manually_checked_seed"]):
            return "duplicate of manually checked seed result"
        return ""

    long_df["manual_checked_overlap_status"] = long_df.apply(manual_overlap_status, axis=1)

    # Useful sort order: manual-overlap and duplicate rows first, then newest.
    long_df = long_df.sort_values(
        by=[
            "found_also_through_manually_checked_seed",
            "is_duplicate_across_seed_papers",
            "is_duplicate_across_databases",
            "citing_publication_year",
            "citing_title",
            "found_via_database",
            "found_through_seed_index",
        ],
        ascending=[False, False, False, False, True, True, True],
        kind="stable",
    )

    long_df.to_csv(OUTPUT_LONG_CSV, index=False, encoding="utf-8-sig")

    # Deduplicated screening sheet: one row per unique citing paper.
    # For each duplicate group, pick the row with the richest metadata.
    long_df["metadata_richness_score"] = (
        long_df["citing_doi"].fillna("").astype(str).str.len().gt(0).astype(int) * 5
        + long_df["citing_abstract"].fillna("").astype(str).str.len().gt(0).astype(int) * 3
        + long_df["citing_authors"].fillna("").astype(str).str.len().gt(0).astype(int)
        + long_df["citing_landing_page_url"].fillna("").astype(str).str.len().gt(0).astype(int)
    )

    best_rows = (
        long_df.sort_values(
            by=["metadata_richness_score", "citing_publication_year", "citing_cited_by_count"],
            ascending=[False, False, False],
            kind="stable",
        )
        .groupby("citing_duplicate_key", as_index=False)
        .first()
    )

    dedup_columns = [
        "citing_duplicate_key",
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
        "citing_openalex_id",
        "citing_semantic_scholar_id",
        "duplicate_group_row_count",
        "duplicate_group_seed_count",
        "duplicate_group_database_count",
        "is_duplicate_across_seed_papers",
        "is_duplicate_across_databases",
        "all_found_via_databases",
        "all_found_through_seed_indices",
        "all_found_through_seed_titles",
        "all_found_through_seed_dois",
        "found_also_through_manually_checked_seed",
        "manual_checked_overlap_status",
    ]

    dedup_df = best_rows[dedup_columns].sort_values(
        by=[
            "found_also_through_manually_checked_seed",
            "is_duplicate_across_seed_papers",
            "is_duplicate_across_databases",
            "citing_publication_year",
            "citing_title",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
    )

    dedup_df.to_csv(OUTPUT_DEDUP_CSV, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Long duplicate-preserving output: {OUTPUT_LONG_CSV.resolve()}")
    print(f"Deduplicated screening output:    {OUTPUT_DEDUP_CSV.resolve()}")
    if unresolved_seeds:
        print(f"Unresolved seed report:          {OUTPUT_UNRESOLVED_CSV.resolve()}")


def main(argv=None):
    return run_collection(globals(), argv, combined=True)


if __name__ == "__main__":
    raise SystemExit(main())
