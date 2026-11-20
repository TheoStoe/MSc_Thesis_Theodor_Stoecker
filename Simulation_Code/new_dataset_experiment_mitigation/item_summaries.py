"""Read-only validation of existing Goodreads item-text overlays.

These are deterministic excerpts from supplied book metadata, not generated
summaries. Preparation never calls a provider or changes prepared model inputs.
"""
from __future__ import annotations

import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Mapping



SCHEMA = "goodreads-source-description-excerpts-v1"
MAX_DESCRIPTION_CHARS = 400
ITEMS_FILE = "items.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _DescriptionHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in {"script", "style"}:
            self.hidden_depth += 1
        if tag in {"p", "div", "br", "li", "section", "h1", "h2", "h3", "hr"}:
            self.parts.append(" ")

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        if tag in {"br", "hr"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.hidden_depth = max(0, self.hidden_depth - 1)
        if tag in {"p", "div", "li", "section", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


def clean_description(value: Any) -> str:
    """Remove markup, decode entities and collapse whitespace without inference."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("source description must be text or null")
    parser = _DescriptionHTML()
    parser.feed(value)
    parser.close()
    return " ".join(html.unescape("".join(parser.parts)).split())


def bounded_excerpt(text: str, max_chars: int = MAX_DESCRIPTION_CHARS) -> str:
    """Prefer a sentence ending in the latter half of the bound, then a word."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if len(text) <= max_chars:
        return text
    prefix = text[:max_chars]
    sentence_ends = [match.end() for match in re.finditer(r"[.!?](?=\s|$)", prefix)
                     if match.end() >= max_chars / 2]
    if sentence_ends:
        return prefix[:sentence_ends[-1]].rstrip()
    if text[max_chars].isspace():
        return prefix.rstrip()
    boundary = prefix.rfind(" ")
    return prefix[:boundary].rstrip() if boundary > 0 else prefix.rstrip()


def text_policy(max_chars: int = MAX_DESCRIPTION_CHARS) -> dict[str, Any]:
    bounded_excerpt("", max_chars)
    return {
        "schema_version": SCHEMA,
        "description_max_chars": max_chars,
        "source": "supplied Goodreads catalogue description; no review text",
        "cleaning": "HTMLParser visible text, HTML entity decoding, compact whitespace",
        "excerpt_rule": "prefix; sentence ending in latter half of bound, otherwise whole word; hard bound for one overlong word",
        "missing_rule": "retain supplied title/authors/genres; mark source description unavailable",
        "generation": "none; source excerpts are not generated book summaries",
        "field_allowlist": ["item_id", "original_source_item_id", "title", "authors", "genres", "description"],
        "annotations": "no VIAF files, recorded gender labels, name-based gender inference, or biography inference",
    }


def _base_text(row: Mapping[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    if not title:
        raise ValueError("every item requires a nonempty source title")
    parts = [f"Book: {title}"]
    for key, label in (("authors", "Author(s)"), ("genres", "Genres")):
        values = row.get(key) or ()
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{key} must be a sequence")
        joined = ", ".join(str(value).strip() for value in values if str(value).strip())
        if joined:
            parts.append(f"{label}: {joined}")
    return "; ".join(parts)


def build_records(catalog: Mapping[str, Any], *, max_chars: int = MAX_DESCRIPTION_CHARS) -> dict[str, dict[str, Any]]:
    policy = text_policy(max_chars)
    if not isinstance(catalog, Mapping) or not catalog:
        raise ValueError("source catalogue must be a nonempty mapping")
    rows: dict[str, dict[str, Any]] = {}
    for key, item in catalog.items():
        if not isinstance(item, Mapping):
            raise ValueError("catalogue item must be a mapping")
        iid = item.get("item_id")
        if isinstance(iid, bool) or not isinstance(iid, int) or iid < 1 or str(iid) != key:
            raise ValueError("catalogue keys must match positive integer item IDs")
        source_id = str(item.get("original_source_item_id") or "")
        if not source_id.startswith(("work:", "book:")):
            raise ValueError("every item requires its original book/work identifier")
        raw = item.get("description")
        cleaned = clean_description(raw)
        excerpt = bounded_excerpt(cleaned, max_chars)
        base = _base_text(item)
        text = (f"{base}\nSource description excerpt: {excerpt}" if excerpt
                else f"{base}\nSource description unavailable.")
        rows[key] = {
            "item_id": iid, "original_source_item_id": source_id,
            "status": "source_excerpt" if excerpt else "missing_description_fallback",
            "source_description_chars": len(raw or ""),
            "cleaned_description_chars": len(cleaned),
            "excerpt_chars": len(excerpt), "excerpt_truncated": len(excerpt) < len(cleaned),
            "source_excerpt": excerpt, "item_text": text,
        }
    if sorted(map(int, rows)) != list(range(1, len(rows) + 1)):
        raise ValueError("catalogue item IDs must be contiguous and one-based")
    if len({row["original_source_item_id"] for row in rows.values()}) != len(rows):
        raise ValueError("original source item IDs must be unique")
    assert policy["description_max_chars"] == max_chars
    return rows


def load_item_summaries(output_path: str | Path, *, prepared_path: str | Path | None = None) -> dict[str, Any]:
    """Verify bytes, source mapping and deterministic content before LLM use."""
    output = Path(output_path)
    complete = json.loads((output / "_COMPLETE.json").read_text(encoding="utf-8"))
    if complete.get("schema_version") != SCHEMA or complete.get("status") != "complete":
        raise ValueError("source excerpt artifact is not marked complete")
    hashes = complete.get("file_sha256", {})
    if set(hashes) != {ITEMS_FILE, "manifest.json"}:
        raise ValueError("source excerpt completion hashes are incomplete")
    for filename, expected in hashes.items():
        if file_sha256(output / filename) != expected:
            raise ValueError(f"source excerpt checksum mismatch: {filename}")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported source excerpt schema")
    prepared = Path(prepared_path).resolve() if prepared_path is not None else Path(manifest["source_catalog_path"]).parent.resolve()
    catalog_path, source_manifest_path = prepared / "catalog.json", prepared / "manifest.json"
    if Path(manifest["source_catalog_path"]).resolve() != catalog_path or Path(manifest["source_prepared_manifest_path"]).resolve() != source_manifest_path:
        raise ValueError("source excerpt artifact belongs to different prepared input paths")
    if (file_sha256(catalog_path) != manifest.get("source_catalog_sha256")
            or file_sha256(source_manifest_path) != manifest.get("source_prepared_manifest_sha256")):
        raise ValueError("source prepared input checksum mismatch")
    policy = manifest.get("policy", {})
    if policy != text_policy(policy.get("description_max_chars")) or _fingerprint(policy) != manifest.get("policy_fingerprint"):
        raise ValueError("source excerpt text policy fingerprint mismatch")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    records = json.loads((output / ITEMS_FILE).read_text(encoding="utf-8"))
    expected = build_records(catalog, max_chars=policy["description_max_chars"])
    if records != expected or _fingerprint(records) != manifest.get("records_fingerprint"):
        raise ValueError("source excerpt records differ from the declared catalogue/policy")
    texts = {key: row["item_text"] for key, row in records.items()}
    if _fingerprint(texts) != manifest.get("item_text_fingerprint") or not all(texts.values()):
        raise ValueError("source excerpt item-text fingerprint mismatch")
    if complete.get("item_count") != len(records) or manifest.get("coverage", {}).get("catalogue_items") != len(records):
        raise ValueError("source excerpt item count mismatch")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("dataset_fingerprint") != manifest.get("source_dataset_fingerprint"):
        raise ValueError("source excerpt dataset fingerprint mismatch")
    return {"manifest": manifest, "records": records,
            "item_text_by_id": {int(key): value for key, value in texts.items()}}


def prepare_item_summaries(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("Item-text generation is disabled in the mitigation extension; reuse frozen Goodreads overlays.")


def main(argv: Any = None) -> int:
    raise SystemExit("This is a read-only validation helper; use the mitigation campaign CLI.")


if __name__ == "__main__":
    main()
