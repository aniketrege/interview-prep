"""The daily entry point: what to do today, with fresh practice copies ready to open.

    python tools/today.py              # today's agenda (nothing is created)
    python tools/today.py --open       # also copy today's notebooks into practice/<date>/ and create the week's notes file
    python tools/today.py --week       # the whole current week
    python tools/today.py --date 2026-11-03 [--week]   # preview another day

Reads schedule.yaml (the guide's 16 weeks, day by day) and practice/log.csv (your sessions).
A notebook ID at the *start* of a slot ("A3.10 cold ...") is opened with --open; IDs later in the text are
references and are only linked. "speed" picks the most overdue Part P drill.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drills import REPO, all_drills  # noqa: E402
from new_session import new_session  # noqa: E402
from progress import due_today, read_log, status  # noqa: E402

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
SLOTS = [("drill", "Drill (Part P)"), ("build", "Build"), ("dsa", "DSA (25 min)"), ("project", "Project"),
         ("lab", "G1 lab"), ("mock", "Mock"), ("note", "Also")]
NOTES_TEMPLATE = """# Week {week} notes · {theme}

Theory to revise: {theory}

## Half-page notes (one per build: what you had to look up, the key equation, the gotcha)

## Flashcards (5 a day: question :: answer)

## Weak spots for next week
"""


def load_schedule():
    sched = yaml.safe_load((REPO / "schedule.yaml").read_text())
    known = {k for k, _ in SLOTS}
    for w in sched["weeks"]:
        for day, slots in w["days"].items():
            bad = [k for k, v in slots.items() if k not in known or not isinstance(v, str)]
            if day not in DAYS or bad:
                raise SystemExit(f"schedule.yaml week {w['week']} {day}: bad slot(s) {bad}. "
                                 f"Quote values that contain commas or colons.")
    return sched, dt.date.fromisoformat(str(sched["start"]))


def locate(date, start, weeks):
    """(week dict or None, day key, week number). Week numbers outside the plan are still returned."""
    n = (date - start).days // 7 + 1
    week = next((w for w in weeks if w["week"] == n), None)
    return week, DAYS[date.weekday()], n


def speed_pick():
    """Most overdue Part P drill; if nothing is due, the least recently practised one (never-done first)."""
    log = read_log()
    p = [d for d in all_drills() if d.id.startswith("P2")]  # P0 is the 45-min full run, not a speed drill
    due = [d for d in due_today("P2")]
    if due:
        return min(due, key=lambda d: status(d, log[d.id.upper()])["due"])
    return min(p, key=lambda d: (bool(log.get(d.id.upper())),
                                 log[d.id.upper()][-1]["date"] if log.get(d.id.upper()) else ""))


def resolve(text, ids):
    """Return (leading drill or None, [linked drills]) for a slot's text."""
    text = str(text)
    found = [ids[m] for m in re.findall(r"(?<![\w.])([PAG]\d?\.?\d*[ab]?|P0)(?![\w.]\d|\w)", text) if m in ids]
    lead = re.match(r"\s*([PAG][\w.]*?)(?=[\s:(,;]|$)", text)
    return (ids.get(lead.group(1)) if lead else None), found


def show_day(date, start, weeks, ids):
    week, day, n = locate(date, start, weeks)
    end = start + dt.timedelta(weeks=len(weeks)) - dt.timedelta(days=1)
    if week is None:
        if date < start:
            print(f"The plan starts on {start} ({(start - date).days} days from now). Preview: --date {start}")
        else:
            print(f"The 16-week plan ended on {end}. Keep going with `python tools/progress.py` (spaced review).")
        return []
    print(f"\n=== {date:%a %b %d, %Y} · week {n} of {len(weeks)} · {week['theme']}")
    print(f"    theory this week: {week['theory']}")
    slots = week["days"].get(day, {})
    to_open = []
    for key, label in SLOTS:
        if key not in slots:
            continue
        text = str(slots[key])
        if key == "drill" and text.strip().startswith("speed"):
            d = speed_pick()
            text = f"{d.id} (speed round, 20 min: {d.title.split('·', 1)[-1].strip()})"
        lead, linked = resolve(text, ids)
        print(f"  {label:<15} {text}")
        for d in linked:
            tgt = f", target {d.target_min:g} min" if d.target_min else ""
            print(f"  {'':<15}   -> {d.path.relative_to(REPO)}{tgt}")
        if lead is not None and lead not in to_open:
            to_open.append(lead)
    return to_open


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", type=dt.date.fromisoformat, default=dt.date.today())
    ap.add_argument("--week", action="store_true", help="show the whole week")
    ap.add_argument("--open", action="store_true", help="create practice copies + the week's notes file")
    a = ap.parse_args(argv)

    sched, start = load_schedule()
    weeks = sched["weeks"]
    ids = {d.id: d for d in all_drills()}
    if a.week:
        monday = a.date - dt.timedelta(days=a.date.weekday())
        for i in range(7):
            show_day(monday + dt.timedelta(days=i), start, weeks, ids)
        return

    to_open = show_day(a.date, start, weeks, ids)
    due = [d for d in due_today() if d not in to_open]
    print("\n  Spaced review   " + (", ".join(d.id for d in due) + "  (10 min: redo one from memory)" if due
                                  else "nothing due (log sessions with the last cell of each notebook)"))
    print("  Every day       20-30 min theory note + 5 flashcards; log every notebook with its last cell")

    week, _, n = locate(a.date, start, weeks)
    notes = REPO / "practice" / "notes" / f"week-{n:02d}.md"
    if not a.open:
        if to_open:
            print(f"\nRun with --open to copy {', '.join(d.id for d in to_open)} into practice/{a.date}/")
        return
    print()
    if a.date != dt.date.today():
        print("(--open always creates copies under today's date)")
    for d in to_open:
        new_session(d.id)
    if week is not None and not notes.exists():
        notes.parent.mkdir(parents=True, exist_ok=True)
        notes.write_text(NOTES_TEMPLATE.format(week=n, theme=week["theme"], theory=week["theory"]))
        print(f"created {notes.relative_to(REPO)}")
    elif week is not None:
        print(f"notes: {notes.relative_to(REPO)}")


if __name__ == "__main__":
    main(sys.argv[1:])
