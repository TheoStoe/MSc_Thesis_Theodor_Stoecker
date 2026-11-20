"""Execute final-analysis notebooks sequentially and verify their saved exports.

This runner is for trusted, offline analysis notebooks. Its source checks catch
accidental simulation/provider entry points; they are not a security sandbox.
Only notebook outputs, HTML companions, and analysis execution reports are saved.
Jupyter requires local communication sockets. In a restricted Windows sandbox,
run this offline command through host execution if socketpair creation blocks.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import AsyncKernelManager

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FORBIDDEN_IMPORTS = (
    "openai", "anthropic", "requests", "httpx", "urllib.request", "socket", "subprocess",
    "feedback_loop_experiment.run", "feedback_loop_experiment.engine",
    "mitigation_experiment.run", "mitigation_experiment.engine",
    "new_dataset_experiment.engine", "new_dataset_experiment_mitigation.runner",
    "new_dataset_experiment_mitigation.movielens_seed44_queue",
)
FORBIDDEN_CALLS = {
    "run_simulator", "run_simulation", "run_campaign", "launch_campaign",
    "run_experiment", "resume_run", "fit", "train", "training_step",
    "system", "popen", "spawn", "Popen", "exec", "eval", "__import__",
}
REMOVED_CREDENTIALS = (
    "OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "COHERE_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY", "HF_TOKEN",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash(notebook):
    contents = [{"cell_type": cell.cell_type, "source": cell.source} for cell in notebook.cells]
    return hashlib.sha256(json.dumps(contents, ensure_ascii=False).encode("utf-8")).hexdigest()


def _inside(repo, value):
    path = Path(value)
    path = (path if path.is_absolute() else repo / path).resolve()
    if not path.is_relative_to(repo):
        raise ValueError(f"Analysis path must remain inside the repository: {path}")
    return path


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def validate_offline_cells(notebook):
    """Reject obvious provider, training, shell, and experiment launch calls."""
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        lines = []
        for line in cell.source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("%matplotlib ") and stripped.split(maxsplit=1)[1] in {"inline", "agg"}:
                lines.append("# Allowed plotting backend magic")
            elif stripped.startswith("%") or stripped.startswith("!"):
                raise ValueError(f"Cell {index + 1}: shell commands and non-plotting magics are disabled")
            else:
                lines.append(line)
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError as exc:
            raise ValueError(f"Cell {index + 1}: expected ordinary Python analysis code") from exc
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [entry.name for entry in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            for imported in imports:
                if any(imported == name or imported.startswith(name + ".") for name in FORBIDDEN_IMPORTS):
                    raise ValueError(f"Cell {index + 1}: non-analysis import is disabled: {imported}")
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name.rsplit(".", 1)[-1] in FORBIDDEN_CALLS:
                    raise ValueError(f"Cell {index + 1}: simulation, training, shell, or dynamic call is disabled: {name}")
                if name.endswith((".responses.create", ".completions.create", ".embeddings.create")):
                    raise ValueError(f"Cell {index + 1}: provider call is disabled: {name}")
    return True


def _environment(repo):
    environment = dict(os.environ)
    for key in REMOVED_CREDENTIALS:
        environment.pop(key, None)
    environment.update({
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2",
        "PYTHONHASHSEED": "0", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    cache = repo / ".build/notebook_runtime"
    cache.mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(cache / "matplotlib")
    environment["IPYTHONDIR"] = str(cache / "ipython")
    environment["JUPYTER_RUNTIME_DIR"] = str(cache / "jupyter")
    return environment


def _python(repo):
    candidates = [repo / ".venv/Scripts/python.exe", repo / ".venv/bin/python"]
    return str(next((candidate for candidate in candidates if candidate.is_file()), Path(sys.executable)))


def _exports(directory):
    if not directory.is_dir():
        return {}
    return {str(path.resolve()): {"bytes": path.stat().st_size, "sha256": _hash(path)}
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _atomic_text(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def _html(notebook, *, path, success):
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input = False
    exporter.exclude_input_prompt = False
    contents, _ = exporter.from_notebook_node(notebook, resources={"metadata": {"name": path.stem}})
    # nbconvert already depends on BeautifulSoup. Inputs are visible initially,
    # and the examiner can collapse each block without hiding analysis outputs.
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(contents, "html.parser")
    for wrapper in list(soup.select(".jp-CodeCell .jp-Cell-inputWrapper")):
        details = soup.new_tag("details", attrs={"class": "analysis-code", "open": ""})
        summary = soup.new_tag("summary")
        summary.string = "Code (click to collapse)"
        wrapper.wrap(details)
        details.insert(0, summary)
    style = soup.new_tag("style")
    style.string = (".analysis-code {margin: .6em 0; border: 1px solid #ddd; padding: .35em;} "
                    ".analysis-code > summary {cursor: pointer; font-size: .9em;} "
                    ".analysis-code .jp-InputArea {overflow-x: auto;} "
                    ".analysis-execution-status {padding: .6em 1em; border: 1px solid #bbb; margin: 1em 0;}")
    soup.head.append(style)
    banner = soup.new_tag("div", attrs={"class": "analysis-execution-status"})
    banner.string = ("Executed offline; every code cell completed successfully." if success else
                     "Execution did not complete successfully. See the notebook error and execution report.")
    main = soup.find("main") or soup.body
    main.insert(0, banner)
    return str(soup)


def _check_rq_exports(notebook, repo, directory):
    """Check each RQ's TeX/plot bundles against its declared notebook metadata."""
    metadata = notebook.metadata.get("analysis", {})
    rq = metadata.get("rq")
    checks, errors = [], []
    if not rq:
        return checks, errors
    for kind, singular, suffixes in (("tables", "table", (".tex", ".csv", ".json")),
                                      ("figures", "figure", (".tex", ".pdf", ".png", ".json"))):
        selected = sorted((directory / kind).glob(f"{rq}_*.tex"))
        if metadata.get(f"require_{singular}") and not selected:
            errors.append({"type": "MissingExport", "message": f"No {rq}_* {singular} bundle found under {directory / kind}"})
        for tex in selected:
            item = {"kind": singular, "analysis_name": tex.stem, "files": []}
            for suffix in suffixes:
                path = tex.with_suffix(suffix)
                exists = path.is_file() and path.stat().st_size > 0
                item["files"].append({"path": str(path), "exists": exists,
                                      "sha256": _hash(path) if exists else None})
                if not exists:
                    errors.append({"type": "MissingExport", "message": f"Absent or empty export: {path}"})
            category = "tab" if singular == "table" else "fig"
            expected_label = f"chap:results:{category}:{tex.stem}"
            item["label"] = expected_label
            item["tex_label_present"] = rf"\label{{{expected_label}}}" in tex.read_text(encoding="utf-8")
            if not item["tex_label_present"]:
                errors.append({"type": "ExportLabelMismatch", "message": f"Expected {expected_label} in {tex}"})
            checks.append(item)
    return checks, errors




def execute_notebooks(notebooks=None, *, repo_root=REPO, notebook_dir=None, only=None,
                      timeout=180, report_path=None):
    """Execute selected notebooks one at a time, persisting an auditable report.

    ``notebooks`` accepts paths; otherwise all numbered .ipynb files in the
    analysis notebook directory are selected. ``only`` accepts exact basenames
    or stems (a string or sequence), making individual retries straightforward.
    Expected exports can be declared in notebook metadata as
    ``analysis.expected_exports`` using repository-relative paths.
    """
    repo = Path(repo_root).resolve()
    directory = _inside(repo, notebook_dir or "reported_runs/analysis/notebooks")
    report_path = _inside(repo, report_path or ".build/analysis_execution_report.json")
    paths = ([_inside(repo, path) for path in notebooks] if notebooks is not None else
             sorted(directory.glob("[0-9][0-9]_*.ipynb")))
    if isinstance(only, str):
        only = [only]
    if only:
        choices = set(only)
        paths = [path for path in paths if path.name in choices or path.stem in choices]
        found = {choice for choice in choices if any(choice in {path.name, path.stem} for path in paths)}
        if found != choices:
            raise ValueError(f"No exact notebook match for: {sorted(choices - found)}")
    if not paths:
        raise ValueError("No analysis notebooks selected")
    if len(set(paths)) != len(paths):
        raise ValueError("A notebook may be selected only once")
    if timeout <= 0:
        raise ValueError("Cell timeout must be positive")
    environment = _environment(repo)
    interpreter = _python(repo)
    report = {"schema_version": "thesis-analysis-notebook-execution-v1", "started_at_utc": _now(),
              "status": "running", "repository_root": str(repo), "python": interpreter,
              "kernel_name": "python3", "execution_order": "sequential", "cell_timeout_seconds": timeout,
              "provider_credentials_removed": True, "cpu_thread_limit": 2, "notebooks": []}
    report["analysis_code_sha256"] = {path.name: _hash(path)
                                      for path in sorted((repo / "reported_runs/analysis").glob("*.py"))}
    _atomic_text(report_path, json.dumps(report, indent=2))
    for position, path in enumerate(paths, 1):
        print(f"[{position}/{len(paths)}] Executing {path.name}", flush=True)
        started = time.monotonic()
        item = {"path": str(path), "started_at_utc": _now(), "status": "failed", "errors": []}
        export_directory = repo / "reported_runs/analysis/exports"
        before_exports = _exports(export_directory)
        notebook = None
        original_hash = None
        try:
            notebook = nbformat.read(path, as_version=4)
            original_hash = _source_hash(notebook)
            export_directory = _inside(repo, notebook.metadata.get("analysis", {}).get("export_directory", export_directory))
            before_exports = _exports(export_directory)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
            validate_offline_cells(notebook)
            manager = AsyncKernelManager(kernel_name="python3")
            manager.kernel_spec.argv = [interpreter, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            client = NotebookClient(notebook, km=manager, timeout=timeout, startup_timeout=90,
                                    allow_errors=False, kernel_name="python3", record_timing=True,
                                    resources={"metadata": {"path": str(repo)}})
            client.execute(cwd=str(repo), env=environment, cleanup_kc=True)
            item["status"] = "complete"
        except Exception as exc:
            item["errors"].append({"type": type(exc).__name__, "message": str(exc)[-12000:],
                                   "traceback": traceback.format_exc()[-12000:]})
        if notebook is not None:
            code = [cell for cell in notebook.cells if cell.cell_type == "code"]
            errors = [{"cell": index + 1, "ename": output.get("ename"), "evalue": output.get("evalue")}
                      for index, cell in enumerate(notebook.cells) if cell.cell_type == "code"
                      for output in cell.get("outputs", []) if output.get("output_type") == "error"]
            item.update(total_cells=len(notebook.cells), code_cells=len(code),
                        executable_code_cells=sum(bool(cell.source.strip()) for cell in code),
                        executed_code_cells=sum(cell.execution_count is not None for cell in code),
                        error_outputs=errors, source_sha256=original_hash,
                        display_output_count=sum(output.get("output_type") in {"display_data", "execute_result"}
                                                 for cell in code for output in cell.get("outputs", [])))
            if errors or any(cell.source.strip() and cell.execution_count is None for cell in code):
                item["status"] = "failed"
            try:
                current_hash = _source_hash(nbformat.read(path, as_version=4))
                if current_hash != original_hash:
                    raise ValueError("Notebook source changed during execution; refusing to overwrite it")
                _atomic_text(path, nbformat.writes(notebook))
                item.update(notebook_sha256=_hash(path))
            except Exception as exc:
                item["status"] = "failed"
                item["errors"].append({"type": type(exc).__name__, "message": str(exc)})
            expected = notebook.metadata.get("analysis", {}).get("expected_exports", [])
            item["expected_exports"] = []
            for value in expected:
                export = _inside(repo, value)
                exists = export.is_file() and export.stat().st_size > 0
                item["expected_exports"].append({"path": str(export), "exists": exists,
                                                "sha256": _hash(export) if exists else None})
            if any(not value["exists"] for value in item["expected_exports"]):
                item["status"] = "failed"
                item["errors"].append({"type": "MissingExport", "message": "One or more declared exports is absent or empty"})
            item["export_bundles"], export_errors = _check_rq_exports(notebook, repo, export_directory)
            if export_errors:
                item["status"] = "failed"
                item["errors"].extend(export_errors)
            try:
                companion = path.with_suffix(".html")
                _atomic_text(companion, _html(notebook, path=path, success=item["status"] == "complete"))
                item.update(html_path=str(companion), html_exists=companion.is_file(), html_sha256=_hash(companion))
            except Exception as exc:
                item["status"] = "failed"
                item["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        after_exports = _exports(export_directory)
        item["created_or_changed_exports"] = [{"path": value, "exists": True, **details}
            for value, details in after_exports.items() if before_exports.get(value) != details]
        item["exports_present"] = len(after_exports)
        item["export_directory"] = str(export_directory)
        item["completed_at_utc"] = _now()
        item["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["notebooks"].append(item)
        _atomic_text(report_path, json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  {item['status']}: {item.get('executed_code_cells', 0)}/{item.get('code_cells', 0)} code cells; "
              f"{len(item['errors'])} errors", flush=True)
    report.update(completed_at_utc=_now(),
                  status="complete" if all(item["status"] == "complete" for item in report["notebooks"]) else "failed")
    report["completed_notebooks"] = sum(item["status"] == "complete" for item in report["notebooks"])
    report["selected_notebooks"] = len(paths)
    _atomic_text(report_path, json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook-dir", type=Path)
    parser.add_argument("--only", action="append", help="Exact notebook filename or stem; repeat to select several")
    parser.add_argument("--timeout", type=int, default=180, help="Maximum seconds per code cell")
    parser.add_argument("--report", type=Path, help="Repository-relative or absolute execution report path")
    args = parser.parse_args(argv)
    report = execute_notebooks(notebook_dir=args.notebook_dir, only=args.only,
                               timeout=args.timeout, report_path=args.report)
    print(f"{report['completed_notebooks']}/{report['selected_notebooks']} notebooks completed", flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
