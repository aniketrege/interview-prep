"""Regenerate the notebook catalog in README.md from the notebook headers.

    python tools/catalog.py

Rewrites everything between <!-- catalog:start --> and <!-- catalog:end --> in README.md.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drills import REPO, all_drills  # noqa: E402

WEEK_RE = re.compile(r"\*\*Plan week:\*\*\s*([^·\n]+)")
PART_NAMES = {"P_pytorch": "Part P · PyTorch first principles", "A_coding": "Part A · Coding fundamentals",
              "G1_systems": "Part G1 · Training & inference systems"}


def main() -> None:
    state_path = REPO / "tools" / "verification.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    out, part = [], None
    for d in all_drills():
        p = d.path.parent.name
        if p != part:
            part = p
            out += ["", f"#### {PART_NAMES.get(p, p)}", "",
                    "| ID | Drill | Target | Plan week | Practice | Solution | Verified |",
                    "| --- | --- | --- | --- | --- | --- | --- |"]
        head = nbformat.read(d.path, as_version=4).cells[0].source
        wk = WEEK_RE.search(head)
        sol = d.path.parent / "solutions" / d.path.name
        v = state.get(str(sol.relative_to(REPO)), {})
        ok = f"{v['status']} {v['date']}" if v else "-"
        title = d.title.split("·", 1)[-1].strip()
        out.append(f"| {d.id} | {title} | {d.target_min:g} min | {wk.group(1).strip() if wk else '-'} | "
                   f"[practice]({d.path.relative_to(REPO)}) | [solution]({sol.relative_to(REPO)}) | {ok} |")
    readme = REPO / "README.md"
    text = readme.read_text()
    new = re.sub(r"(<!-- catalog:start -->).*(<!-- catalog:end -->)",
                 lambda m: m.group(1) + "\n" + "\n".join(out).strip() + "\n" + m.group(2), text, flags=re.S)
    readme.write_text(new)
    print(f"catalog: {len(all_drills())} notebooks")


if __name__ == "__main__":
    main()
