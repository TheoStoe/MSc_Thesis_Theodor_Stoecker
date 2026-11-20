"""Export checked notebook tables without recalculating scientific metrics.

Use the returned display_frame in the notebook: it contains exactly the visible
values rendered into the LaTeX. The CSV retains the original numeric precision.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import pandas as pd
from pandas.api.types import is_scalar

LABEL_PREFIX = "chap:results:tab:"
DEFAULT_FLOAT_FORMAT = ".6g"
MISSING_DISPLAY = "—"
_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "—": r"\textemdash{}"}


@dataclass(frozen=True)
class LatexTableExport:
    label: str
    path: Path
    text: str
    data_path: Path
    provenance_path: Path
    display_frame: pd.DataFrame


def table_label(analysis_name: str) -> str:
    """Require an explicit stable snake_case name; never silently rename it."""
    if not isinstance(analysis_name, str) or not _NAME.fullmatch(analysis_name):
        raise ValueError("analysis_name must be lower-case snake_case, starting with a letter")
    return LABEL_PREFIX + analysis_name


def escape_latex(value: Any) -> str:
    # Character-wise escaping does not escape the commands it just introduced.
    text = " ".join(str(value).splitlines())
    return "".join(_ESCAPES.get(character, character) for character in text)


def _display_value(value: Any, formatter: str | Callable | None) -> str:
    if not is_scalar(value):
        raise TypeError("table cells must contain scalar values")
    if pd.isna(value):
        return MISSING_DISPLAY
    if isinstance(value, Real) and not math.isfinite(value):
        raise ValueError("infinite values cannot be exported as scientific results")
    if formatter is not None:
        return str(formatter(value) if callable(formatter) else format(value, formatter))
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(value)
    if isinstance(value, Real):
        # Six significant digits retain tiny nonzero values rather than showing 0.
        return format(value, DEFAULT_FLOAT_FORMAT)
    return str(value)


def export_latex_table(
    frame: pd.DataFrame, *, analysis_name: str, caption: str,
    output_dir: str | Path, column_labels: Mapping[str, str] | None = None,
    formatters: Mapping[str, str | Callable] | None = None,
    provenance: Mapping[str, Any] | None = None,
    column_format: str | None = None, placement: str = "htbp!",
    caption_is_latex: bool = False,
) -> LatexTableExport:
    """Export a complete table, raw CSV, and metadata from one checked DataFrame.

    Metric denominators, means, and differences are not calculated here.
    Apply them in the shared metric code before calling this. Unit formatting
    such as percentages is an explicit presentation choice in formatters.
    Formatters use Python format specs (e.g. '.2f' or '.1%') or callables.
    Undefined values remain missing and display as an em dash. A nonempty caption
    is optional; empty captions retain a numbered label without generated prose.

    An explicit non-default index must be reset first so meaningful group labels
    cannot silently disappear. Re-running a notebook replaces its named exports.
    """
    label = table_label(analysis_name)
    if not isinstance(frame, pd.DataFrame) or not len(frame.columns):
        raise ValueError("provide a DataFrame with at least one column")
    if isinstance(frame.columns, pd.MultiIndex) or not frame.columns.is_unique:
        raise ValueError("table columns must be unique, flat names")
    if any(not isinstance(column, str) or not column.strip() for column in frame.columns):
        raise ValueError("table column names must be nonempty strings")
    if frame.index.name is not None or not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError("reset_index() before exporting to preserve the table's grouping labels")
    if not isinstance(caption, str):
        raise TypeError("caption must be a string")
    if not isinstance(caption_is_latex, bool):
        raise TypeError("caption_is_latex must be boolean")
    if not isinstance(placement, str) or not re.fullmatch(r"[htbp!]+", placement):
        raise ValueError("unsupported table placement")
    labels, formats = dict(column_labels or {}), dict(formatters or {})
    unknown = (set(labels) | set(formats)) - set(frame.columns)
    if unknown:
        raise ValueError(f"column labels or formatters refer to absent columns: {sorted(unknown)}")
    headings = [labels.get(column, column) for column in frame.columns]
    if any(not isinstance(value, str) or not value.strip() for value in headings) or len(set(headings)) != len(headings):
        raise ValueError("displayed column headings must be distinct, nonempty strings")
    if any(not isinstance(fmt, str) and not callable(fmt) for fmt in formats.values()):
        raise TypeError("formatters must be Python format specifications or callables")
    # Validate metadata before creating directories or writing any export.
    source_metadata = dict(provenance or {})
    json.dumps(source_metadata, ensure_ascii=False, allow_nan=False)
    rows = [[_display_value(value, formats.get(column)) for column, value in zip(frame.columns, row)]
            for row in frame.itertuples(index=False, name=None)]
    display_frame = pd.DataFrame(rows, columns=headings)
    if column_format is None:
        column_format = "".join("r" if pd.api.types.is_numeric_dtype(frame[column]) else "l"
                                for column in frame.columns)
    if not isinstance(column_format, str) or not column_format.strip() or "\n" in column_format:
        raise ValueError("column_format must be a nonempty, single-line LaTeX column specification")
    caption_text = caption if caption_is_latex else escape_latex(caption)
    lines = [rf"\begin{{table}}[{placement}]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.1}",
        rf"\begin{{tabular}}{{{column_format}}}", r"\hline",
        " & ".join(map(escape_latex, headings)) + r" \\", r"\hline"]
    lines.extend(" & ".join(map(escape_latex, row)) + r" \\" for row in rows)
    caption_line = rf"\caption{{{caption_text}}}" if caption.strip() else r"\refstepcounter{table}"
    lines.extend([r"\hline", r"\end{tabular}", caption_line,
                  rf"\label{{{label}}}", r"\end{table}", ""])
    latex = "\n".join(lines)
    csv = frame.to_csv(index=False, lineterminator="\n")
    metadata = {"schema_version": "thesis-latex-table-export-v1", "analysis_name": analysis_name,
        "label": label, **({"caption": caption} if caption.strip() else {}), "rows": len(frame),
        "source_columns": list(frame.columns), "source_dtypes": {c: str(frame[c].dtype) for c in frame.columns},
        "display_columns": headings, "display_rows": rows,
        "formatting": {"default_float": DEFAULT_FLOAT_FORMAT, "missing_display": MISSING_DISPLAY,
            "column_format": column_format, "placement": placement, "caption_is_latex": caption_is_latex,
            "column_formatters": {c: fmt if isinstance(fmt, str) else getattr(fmt, "__qualname__", type(fmt).__name__)
                                  for c, fmt in formats.items()}},
        "provenance": source_metadata,
        "files": {f"{analysis_name}.tex": hashlib.sha256(latex.encode()).hexdigest(),
                  f"{analysis_name}.csv": hashlib.sha256(csv.encode()).hexdigest()}}
    json_text = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path, data_path, metadata_path = [directory / f"{analysis_name}.{extension}" for extension in ("tex", "csv", "json")]
    path.write_text(latex, encoding="utf-8", newline="\n")
    data_path.write_text(csv, encoding="utf-8", newline="\n")
    metadata_path.write_text(json_text, encoding="utf-8", newline="\n")
    return LatexTableExport(label, path, latex, data_path, metadata_path, display_frame)
