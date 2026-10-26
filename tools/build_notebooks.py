"""Build practice (template) + solution notebooks from jupytext sources.

Source of truth: <part>/_src/*.py in jupytext "percent" format
(https://jupytext.readthedocs.io/en/latest/formats-scripts.html).

Conventions (modelled on nbgrader's solution markers,
https://nbgrader.readthedocs.io/en/stable/user_guide/creating_and_grading_assignments.html):

  ### BEGIN SOLUTION ... ### END SOLUTION
      Inside a code cell. In the template the block is replaced by any
      "# TEMPLATE: <code>" lines it contains (uncommented), otherwise by
      `raise NotImplementedError  # YOUR CODE HERE` at the same indentation.

  # %% tags=["solution"]
      Whole cell is yours to write: the template gets an empty cell with only
      the leading comment lines (the "spec") of the solution cell kept.

  # %% tags=["solution_only"]   dropped from the template (e.g. reference plots)
  # %% tags=["template_only"]   dropped from the solution

Outputs:
  <part>/<stem>.ipynb             practice notebook (never executed, no outputs)
  <part>/solutions/<stem>.ipynb   reference solution (executed by tools/verify.py)

Every source must start with a markdown cell containing a "Sources" heading:
citations are part of the contract, not decoration.

Usage:  python tools/build_notebooks.py [paths or part dirs ...]   (default: all)
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import jupytext
import nbformat

REPO = Path(__file__).resolve().parent.parent
PARTS = ["P_pytorch", "A_coding", "G1_systems"]
BEGIN, END = "### BEGIN SOLUTION", "### END SOLUTION"
BUILD_TAGS = {"solution", "solution_only", "template_only"}
TEMPLATE_RE = re.compile(r"^(\s*)# TEMPLATE: ?(.*)$")
KERNEL = {"display_name": "Python 3 (interview-prep)", "language": "python", "name": "python3"}


def strip_solution_blocks(src: str, where: str) -> str:
    out, in_block, block = [], False, []
    for line in src.splitlines():
        s = line.strip()
        if s == BEGIN:
            if in_block:
                raise ValueError(f"{where}: nested {BEGIN}")
            in_block, block = True, []
            indent = line[: len(line) - len(line.lstrip())]
            continue
        if s == END:
            if not in_block:
                raise ValueError(f"{where}: {END} without {BEGIN}")
            tmpl = [TEMPLATE_RE.match(b) for b in block]
            tmpl = [m.group(1) + m.group(2) for m in tmpl if m]
            out.extend(tmpl or [indent + "raise NotImplementedError  # YOUR CODE HERE"])
            in_block = False
            continue
        (block if in_block else out).append(line)
    if in_block:
        raise ValueError(f"{where}: unterminated {BEGIN}")
    return "\n".join(out)


def drop_template_lines(src: str) -> str:
    """Solution view: remove BEGIN/END markers and TEMPLATE comments."""
    keep = []
    for line in src.splitlines():
        s = line.strip()
        if s in (BEGIN, END) or TEMPLATE_RE.match(line):
            continue
        keep.append(line)
    return "\n".join(keep)


def leading_comments(src: str) -> str:
    lines = []
    for line in src.splitlines():
        if line.strip().startswith("#") or not line.strip():
            lines.append(line)
        else:
            break
    body = "\n".join(lines).rstrip()
    return (body + "\n" if body else "") + "# YOUR CODE HERE\n"


def build(src_path: Path) -> tuple[Path, Path]:
    nb = jupytext.read(src_path, fmt="py:percent")
    first = nb.cells[0] if nb.cells else None
    if not (first and first.cell_type == "markdown" and re.search(r"(?im)^#+\s*sources", first.source)):
        raise ValueError(f"{src_path}: first cell must be markdown with a '## Sources' section")

    tmpl, sol = copy.deepcopy(nb), copy.deepcopy(nb)
    tmpl.cells, sol.cells = [], []
    for i, cell in enumerate(nb.cells):
        tags = set(cell.metadata.get("tags", []))
        where = f"{src_path.name} cell {i}"
        c_t, c_s = copy.deepcopy(cell), copy.deepcopy(cell)
        other_tags = sorted(tags - BUILD_TAGS)
        for c in (c_t, c_s):
            c.metadata.pop("tags", None)
            if other_tags:
                c.metadata["tags"] = other_tags
        if cell.cell_type == "code":
            if "solution" in tags:
                c_t.source = leading_comments(drop_template_lines(cell.source))
            else:
                c_t.source = strip_solution_blocks(cell.source, where)
            c_s.source = drop_template_lines(cell.source)
            for c in (c_t, c_s):
                c.outputs, c.execution_count = [], None
        if "solution_only" not in tags:
            tmpl.cells.append(c_t)
        if "template_only" not in tags:
            sol.cells.append(c_s)

    for n in (tmpl, sol):
        n.metadata = {"kernelspec": KERNEL, "language_info": {"name": "python"}}
        nbformat.validate(n)

    part = src_path.parent.parent
    t_path = part / f"{src_path.stem}.ipynb"
    s_path = part / "solutions" / f"{src_path.stem}.ipynb"
    s_path.parent.mkdir(exist_ok=True)
    nbformat.write(tmpl, t_path)
    # keep executed outputs of an existing solution if the source has not changed
    if s_path.exists():
        old = nbformat.read(s_path, as_version=4)
        if [c.source for c in old.cells] == [c.source for c in sol.cells]:
            return t_path, s_path
    nbformat.write(sol, s_path)
    return t_path, s_path


def collect(args: list[str]) -> list[Path]:
    targets = [REPO / p for p in PARTS] if not args else [Path(a).resolve() for a in args]
    files = []
    for t in targets:
        if t.is_dir():
            src = t / "_src" if (t / "_src").is_dir() else t
            files += sorted(src.glob("*.py"))
        else:
            files.append(t)
    return files


if __name__ == "__main__":
    ok = True
    for f in collect(sys.argv[1:]):
        try:
            t, s = build(f)
            print(f"built {t.relative_to(REPO)}  +  {s.relative_to(REPO)}")
        except Exception as e:  # report all broken sources, not just the first
            ok = False
            print(f"FAIL {f}: {e}")
    sys.exit(0 if ok else 1)
