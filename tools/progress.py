"""Progress + spaced-repetition view over practice/log.csv.

    python tools/progress.py          # what is due today, then per-drill status
    python tools/progress.py --all    # include drills never attempted
    python tools/progress.py --part P # filter by ID prefix

Scheduling is a simple Leitner rule: an attempt is "clean" if score >= 4,
no lookups, and time <= target. Each consecutive clean attempt pushes the
next review further out (1, 2, 4, 7, 14, 30, 60 days); any non-clean attempt
resets the drill to review tomorrow.
"""
from __future__ import annotations

import csv
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drills import REPO, Drill, all_drills  # noqa: E402

LOG = REPO / "practice" / "log.csv"
INTERVALS = [1, 2, 4, 7, 14, 30, 60]


def read_log() -> dict[str, list[dict]]:
    rows = defaultdict(list)
    if LOG.exists():
        with LOG.open() as f:
            for r in csv.DictReader(f):
                rows[r["id"].upper()].append(r)
    return rows


def clean(r: dict) -> bool:
    try:
        ok_score = int(r.get("score") or 0) >= 4
        ok_time = float(r["minutes"]) <= float(r.get("target_min") or 1e9)
        return ok_score and ok_time and int(r.get("lookups") or 0) == 0
    except ValueError:
        return False


def status(drill: Drill, rows: list[dict]):
    if not rows:
        return None
    streak = 0
    for r in reversed(rows):
        if not clean(r):
            break
        streak += 1
    last = dt.date.fromisoformat(rows[-1]["date"])
    gap = INTERVALS[min(streak, len(INTERVALS) - 1)] if streak else 1
    best = min(float(r["minutes"]) for r in rows)
    return dict(n=len(rows), last=last, due=last + dt.timedelta(days=gap), streak=streak, best=best,
                last_min=float(rows[-1]["minutes"]), score=rows[-1].get("score") or "-")


def due_today(prefix: str = "") -> list[Drill]:
    log, today = read_log(), dt.date.today()
    out = []
    for d in all_drills():
        s = status(d, log.get(d.id.upper(), []))
        if s and s["due"] <= today and d.id.upper().startswith(prefix.upper()):
            out.append(d)
    return out


def main(argv: list[str]) -> None:
    show_all = "--all" in argv
    prefix = argv[argv.index("--part") + 1] if "--part" in argv else ""
    log, today = read_log(), dt.date.today()
    drills = [d for d in all_drills() if d.id.upper().startswith(prefix.upper())]

    due = due_today(prefix)
    print(f"Due today ({today}): " + (", ".join(d.id for d in due) if due else "nothing - pick a new drill"))
    print()
    print(f"{'ID':<8}{'title':<46}{'target':>7}{'tries':>6}{'best':>6}{'last':>6}{'score':>6}{'streak':>7}  due")
    for d in drills:
        s = status(d, log.get(d.id.upper(), []))
        if s is None and not show_all:
            continue
        tgt = f"{d.target_min:g}" if d.target_min else "-"
        if s is None:
            print(f"{d.id:<8}{d.title[:45]:<46}{tgt:>7}{'0':>6}")
            continue
        mark = "  <- due" if s["due"] <= today else ""
        print(f"{d.id:<8}{d.title[:45]:<46}{tgt:>7}{s['n']:>6}{s['best']:>6.0f}{s['last_min']:>6.0f}"
              f"{s['score']:>6}{s['streak']:>7}  {s['due']}{mark}")
    if not show_all:
        print("\n(--all lists drills you have not attempted yet)")


if __name__ == "__main__":
    main(sys.argv[1:])
