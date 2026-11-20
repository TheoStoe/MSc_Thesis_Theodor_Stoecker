"""Restore portable offline notebook templates; no experiments or raw extraction."""
from pathlib import Path
import argparse
import shutil
import json
import nbformat

ANALYSIS = Path(__file__).resolve().parent

def create_notebooks(snapshot=None, only=None):
    if snapshot is not None:
        raise ValueError("These templates replay the pinned bundled snapshot; change latest.json deliberately for another compatible snapshot.")
    known = {p.name: p for p in (ANALYSIS / "templates").glob("*.ipynb")}
    names = list(known) if only is None else ([only] if isinstance(only, str) else list(only))
    names = [name if name.endswith(".ipynb") else name + ".ipynb" for name in names]
    if set(names) - set(known):
        raise ValueError("Select an existing bundled notebook stem or filename")
    paths = []
    for name in names:
        target = ANALYSIS / "notebooks" / name
        notebook = nbformat.read(known[name], as_version=4)
        pointer = json.loads((ANALYSIS / "latest.json").read_text(encoding="utf-8"))
        relative = "reported_runs/analysis/" + pointer["snapshot_directory"]
        notebook.metadata.analysis["snapshot_directory"] = relative
        notebook.metadata.analysis["export_directory"] = relative + "/exports"
        nbformat.write(notebook, target)
        paths.append(target)
    return paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    for path in create_notebooks(only=args.only):
        print(path)
