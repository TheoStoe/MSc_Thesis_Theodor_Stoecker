"""Read human-entered profile reviews from the evidence workbook.

Only the six editable case-sheet cells supply assessments. Summary formulas,
profile text, and generated evidence never constitute manual judgements.
The original selected cases remain the authoritative population and identity.
"""
from __future__ import annotations

import math
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pandas as pd


EXTENT_CATEGORIES = ("minor adjustment", "substantial change", "unclear")
SUPPORT_CATEGORIES = ("supported", "mixed", "unsupported", "insufficient evidence")
MANUAL_FIELDS = (
    "extent_of_change", "support_in_available_evidence", "retained_interests",
    "supported_additions", "unsupported_changes", "evidence_notes",
)
REVIEW_STATUSES = ("Not started", "In progress", "Complete")
IDENTITY_FIELDS = ("case_id", "seed", "population_seed", "user_id", "review_order", "applied_change_count")
FREQUENCY_COLUMNS = ("scope", "seed", "dimension", "category", "count", "reviewed_profiles",
                     "selected_profiles", "unassessed_profiles", "share_of_reviewed")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PACKAGE_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _sheets(archive):
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationships = {}
    for relation in relations.findall(f"{PACKAGE_REL_NS}Relationship"):
        identity = relation.get("Id")
        if identity in relationships:
            raise ValueError("Duplicate workbook relationship identity")
        relationships[identity] = relation.attrib
    sheets = {}
    for sheet in workbook.findall(f"{NS}sheets/{NS}sheet"):
        name = sheet.get("name")
        if not name or name in sheets:
            raise ValueError("Missing or duplicate workbook sheet name")
        relation = relationships.get(sheet.get(f"{REL_NS}id"), {})
        if relation.get("TargetMode") == "External" or not relation.get("Type", "").endswith("/worksheet"):
            raise ValueError(f"Case workbook sheet has no local worksheet: {name}")
        target = relation.get("Target", "")
        path = posixpath.normpath(target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target))
        if not path.startswith("xl/") or path not in archive.namelist():
            raise ValueError(f"Workbook worksheet part is missing or outside xl: {name}")
        sheets[name] = path
    return sheets


def _strings(archive):
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in item.iter(f"{NS}t")) for item in root.findall(f"{NS}si")]


def _cell_text(cell, shared, context):
    if cell is None:
        return ""
    if cell.find(f"{NS}f") is not None:
        raise ValueError(f"Formula is not permitted in profile-review identity or manual cells: {context}")
    kind = cell.get("t", "n")
    if kind == "e":
        raise ValueError(f"Excel error in profile-review identity or manual cell: {context}")
    if kind == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{NS}t")).strip()
    value = cell.find(f"{NS}v")
    text = value.text if value is not None and value.text is not None else ""
    if kind == "s":
        try:
            index = int(text)
            if index < 0:
                raise ValueError
            return shared[index].strip()
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid shared-string index in profile-review cell: {context}") from exc
    if kind == "b":
        if text not in ("0", "1"):
            raise ValueError(f"Invalid Boolean profile-review cell: {context}")
        return "TRUE" if text == "1" else "FALSE"
    if kind not in ("n", "str"):
        raise ValueError(f"Unsupported profile-review cell type {kind}: {context}")
    return text.strip()


def _integer(value, field):
    if isinstance(value, bool) or pd.isna(value):
        raise ValueError(f"Profile-review identity must be an integer: {field}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Profile-review identity must be an integer: {field}") from exc
    if not math.isfinite(parsed) or parsed != math.floor(parsed):
        raise ValueError(f"Profile-review identity must be an integer: {field}")
    return int(parsed)


def _cases(cases):
    frame = cases.copy(deep=True) if isinstance(cases, pd.DataFrame) else pd.DataFrame(cases)
    if set(IDENTITY_FIELDS) - set(frame):
        raise ValueError(f"Missing selected profile-review identity fields: {sorted(set(IDENTITY_FIELDS)-set(frame))}")
    if frame.empty or frame[list(IDENTITY_FIELDS)].isna().any().any():
        raise ValueError("Selected profile-review cases must have complete identities")
    if not frame.case_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Selected profile-review case IDs must be nonempty strings")
    for column in IDENTITY_FIELDS[1:]:
        frame[column] = frame[column].map(lambda value: _integer(value, column))
    if frame.case_id.duplicated().any() or frame.duplicated(["seed", "user_id"]).any():
        raise ValueError("Duplicate selected profile-review case identity")
    return frame


def read_profile_assessments(path, cases):
    """Join the original cases to six verified human-entered assessment cells.

    Metadata values occupy A4:F4, and editable fields occupy A7:F7 in the
    fixed field orders above. Workbook sheet order and cell storage order do
    not matter. Empty fields are strings, never fabricated review categories.
    All six fields are required for Complete; partial entries are In progress.
    Category values, if present, must match the agreed choices exactly after
    surrounding whitespace is removed. Formulas and Excel errors are rejected.
    """
    frame = _cases(cases)
    sheet_names = [f"S{row.seed}_U{row.user_id}" for row in frame.itertuples()]
    expected_sheets = {"Overview", "Review data", *sheet_names}
    rows = []
    with ZipFile(Path(path)) as archive:
        sheets, shared = _sheets(archive), _strings(archive)
        if set(sheets) != expected_sheets:
            missing, extra = sorted(expected_sheets-set(sheets)), sorted(set(sheets)-expected_sheets)
            raise ValueError(f"Profile-review workbook sheets differ from selected cases; missing={missing}, extra={extra}")
        for original, name in zip(frame.to_dict("records"), sheet_names):
            root = ET.fromstring(archive.read(sheets[name]))
            cells = {}
            for cell in root.findall(f"{NS}sheetData/{NS}row/{NS}c"):
                address = cell.get("r")
                if not address or address in cells:
                    raise ValueError(f"Missing or duplicate worksheet cell address: {name}/{address}")
                cells[address] = cell
            for position, field in enumerate(IDENTITY_FIELDS):
                address = f"{chr(65+position)}4"
                actual = _cell_text(cells.get(address), shared, f"{name}!{address}")
                if field != "case_id":
                    actual = _integer(actual, f"{name}!{address}")
                if actual != original[field]:
                    raise ValueError(f"Profile-review metadata identity differs for {name}: {field}")
            review = {field: _cell_text(cells.get(f"{chr(65+position)}7"), shared, f"{name}!{chr(65+position)}7")
                      for position, field in enumerate(MANUAL_FIELDS)}
            for field, allowed in (("extent_of_change", EXTENT_CATEGORIES), ("support_in_available_evidence", SUPPORT_CATEGORIES)):
                if review[field] and review[field] not in allowed:
                    raise ValueError(f"Invalid profile-review category for {name}/{field}: {review[field]!r}")
            count = sum(bool(value) for value in review.values())
            review["review_status"] = "Not started" if count == 0 else "Complete" if count == len(MANUAL_FIELDS) else "In progress"
            rows.append(review)
    output = frame.reset_index(drop=True)
    for field in (*MANUAL_FIELDS, "review_status"):
        output[field] = [row[field] for row in rows]
    return output


def profile_review_frequencies(frame):
    """Count only Complete reviews within seeds and overall, retaining blanks.

    Category rows always include every agreed category. Their denominator is
    the number of completed profiles in the same scope. Completion-status rows
    count all selected cases but have no ``share_of_reviewed``: pending cases
    are not evidence-support or extent assessments. No category is inferred
    from missing notes, profile text, or an unreviewed case.
    """
    required = {"seed", "review_status", *MANUAL_FIELDS}
    if required-set(frame):
        raise ValueError(f"Missing profile-review assessment columns: {sorted(required-set(frame))}")
    values = frame.copy(deep=True)
    if values.seed.isna().any() or not values.review_status.isin(REVIEW_STATUSES).all():
        raise ValueError("Profile-review seed and review status must be defined")
    values["seed"] = values.seed.map(lambda value: _integer(value, "seed"))
    if "case_id" in values and values.case_id.duplicated().any():
        raise ValueError("Duplicate profile-review assessments cannot be counted")
    for field in MANUAL_FIELDS:
        values[field] = values[field].map(lambda value: "" if pd.isna(value) else str(value).strip())
    for row in values.to_dict("records"):
        for field, allowed in (("extent_of_change", EXTENT_CATEGORIES), ("support_in_available_evidence", SUPPORT_CATEGORIES)):
            if row[field] and row[field] not in allowed:
                raise ValueError(f"Invalid profile-review category: {field}")
        present = sum(bool(row[field]) for field in MANUAL_FIELDS)
        expected = "Not started" if present == 0 else "Complete" if present == len(MANUAL_FIELDS) else "In progress"
        if row["review_status"] != expected:
            raise ValueError("Profile-review status differs from completed manual fields")
    groups = [("overall", pd.NA, values)]
    groups.extend(("seed", int(seed), group) for seed, group in values.groupby("seed", sort=True))
    rows = []
    for scope, seed, group in groups:
        reviewed = group.loc[group.review_status.eq("Complete")]
        base = {"scope": scope, "seed": seed, "reviewed_profiles": len(reviewed),
                "selected_profiles": len(group), "unassessed_profiles": len(group)-len(reviewed)}
        for dimension, categories in (("extent_of_change", EXTENT_CATEGORIES), ("support_in_available_evidence", SUPPORT_CATEGORIES)):
            for category in categories:
                count = int(reviewed[dimension].eq(category).sum())
                rows.append({**base, "dimension": dimension, "category": category, "count": count,
                             "share_of_reviewed": count/len(reviewed) if len(reviewed) else float("nan")})
        for category in REVIEW_STATUSES:
            rows.append({**base, "dimension": "review_status", "category": category,
                         "count": int(group.review_status.eq(category).sum()), "share_of_reviewed": float("nan")})
    return pd.DataFrame(rows, columns=FREQUENCY_COLUMNS)
