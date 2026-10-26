"""Copy a fresh practice notebook into practice/<today>/ so templates stay blank.

    python tools/new_session.py P2.1          # by ID
    python tools/new_session.py P2.04         # or by filename prefix
    python tools/new_session.py --due         # open everything progress.py says is due
"""
from __future__ import annotations

import datetime as dt
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drills import REPO, find  # noqa: E402


def new_session(drill_id: str) -> Path:
    d = find(drill_id)
    day = REPO / "practice" / dt.date.today().isoformat()
    day.mkdir(parents=True, exist_ok=True)
    dst, k = day / d.path.name, 2
    while dst.exists():
        dst = day / f"{d.path.stem}_{k}.ipynb"
        k += 1
    shutil.copy(d.path, dst)
    print(f"{d.id}: {d.title}  (target {d.target_min} min)\n  -> {dst.relative_to(REPO)}")
    return dst


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    if args == ["--due"]:
        from progress import due_today

        args = [d.id for d in due_today()]
    for a in args:
        new_session(a)
