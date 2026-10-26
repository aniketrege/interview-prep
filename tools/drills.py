"""Drill registry, read from the practice notebooks themselves.

Each practice notebook's first markdown cell carries a line like
    **ID:** P2.1 · **Target:** 25 min · **Plan week:** 1
so there is no separate index file to keep in sync.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parent.parent
PARTS = ["P_pytorch", "A_coding", "G1_systems"]
ID_RE = re.compile(r"\*\*ID:\*\*\s*([A-Z][\w.]*)")
TARGET_RE = re.compile(r"\*\*Target:\*\*\s*([\d.]+)\s*min")
TITLE_RE = re.compile(r"^#\s+(.+)$", re.M)


@dataclass
class Drill:
    id: str
    title: str
    target_min: float | None
    path: Path


def all_drills() -> list[Drill]:
    out = []
    for part in PARTS:
        for p in sorted((REPO / part).glob("*.ipynb")):
            nb = nbformat.read(p, as_version=4)
            head = nb.cells[0].source if nb.cells else ""
            m = ID_RE.search(head)
            if not m:
                continue
            t = TARGET_RE.search(head)
            title = TITLE_RE.search(head)
            out.append(Drill(m.group(1), title.group(1).strip() if title else p.stem,
                             float(t.group(1)) if t else None, p))
    return out


def find(drill_id: str) -> Drill:
    want = drill_id.upper()
    for d in all_drills():
        if d.id.upper() == want or d.path.stem.upper().startswith(want):
            return d
    raise SystemExit(f"no drill with id {drill_id!r}; run `python tools/progress.py --all` to list ids")
