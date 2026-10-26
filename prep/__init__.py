"""Small helpers shared by every practice notebook.

Kept deliberately tiny: nothing here should do work you are expected to do
yourself in an interview (no training loops, no losses, no metrics).

    from prep import Session, check_close, seed_everything, DATA

`Session` times a drill and appends one row to practice/log.csv so that
`python tools/progress.py` can tell you what is due for review.
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
import random
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("PREP_DATA", REPO / "data"))
LOG = REPO / "practice" / "log.csv"
LOG_FIELDS = ["date", "id", "minutes", "target_min", "lookups", "score", "notes"]


def seed_everything(seed: int = 0) -> None:
    """Seed python, numpy and torch (CPU + all GPUs)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)  # also seeds every CUDA device
    except ImportError:
        pass


def device():
    """cuda if available, else mps, else cpu."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Session:
    """Wall-clock timer for one drill.

    s = Session("P2.1", target_min=25)   # first cell
    ...
    s.lap("data ready")                  # optional checkpoints
    ...
    s.finish(lookups=0, score=4, notes="forgot model.eval()")  # last cell

    score: your own 1-5 rating (5 = interview-ready, no hesitation).
    lookups: how many times you opened docs / Google / an LLM.
    """

    def __init__(self, drill_id: str, target_min: float):
        self.id, self.target = drill_id, target_min
        self.t0 = time.time()
        self.laps: list[tuple[str, float]] = []
        print(f"[{drill_id}] started {_dt.datetime.now():%H:%M}; target {target_min} min")

    def elapsed_min(self) -> float:
        return (time.time() - self.t0) / 60

    def lap(self, label: str) -> None:
        m = self.elapsed_min()
        self.laps.append((label, m))
        print(f"[{self.id}] {label}: {m:.1f} min")

    def finish(self, lookups: int = 0, score: int | None = None, notes: str = "", log: bool = True) -> float:
        m = self.elapsed_min()
        flag = "within" if m <= self.target else "OVER"
        print(f"[{self.id}] done in {m:.1f} min ({flag} target {self.target} min)")
        if log and not os.environ.get("PREP_NO_LOG"):
            LOG.parent.mkdir(parents=True, exist_ok=True)
            new = not LOG.exists()
            with LOG.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(dict(date=_dt.date.today().isoformat(), id=self.id, minutes=round(m, 1),
                                target_min=self.target, lookups=lookups, score=score, notes=notes))
            print(f"logged to {LOG.relative_to(REPO)}")
        return m


def check_close(name: str, got, expected, atol: float = 1e-5, rtol: float = 1e-4) -> None:
    """assert torch/numpy values match, with a readable message."""
    import numpy as np

    def to_np(x):
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return x.detach().cpu().double().numpy()
        except ImportError:
            pass
        return np.asarray(x, dtype=np.float64)

    g, e = to_np(got), to_np(expected)
    assert g.shape == e.shape, f"{name}: shape {g.shape} != expected {e.shape}"
    if not np.allclose(g, e, atol=atol, rtol=rtol):
        err = np.max(np.abs(g - e))
        raise AssertionError(f"{name}: max abs err {err:.3g} (atol={atol}, rtol={rtol})")
    print(f"ok  {name}")


def check(name: str, cond: bool, msg: str = "") -> None:
    assert cond, f"{name}: {msg}"
    print(f"ok  {name}")
