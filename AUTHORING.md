# Authoring conventions

How every notebook in this repo is written, built and verified. Follow this when you add a drill
(or ask an agent to add one).

## Pipeline

```
<part>/_src/<ID>_<slug>.py        source of truth (jupytext "percent" format)
        │  python tools/build_notebooks.py <part or file>
        ├─▶ <part>/<ID>_<slug>.ipynb             practice template: spec + TODOs + checks, no answers
        └─▶ <part>/solutions/<ID>_<slug>.ipynb   reference solution
                 │  python tools/verify.py <part or file>
                 └─▶ executed in place (outputs saved) + row in VERIFICATION.md
```

Never edit the generated `.ipynb` files by hand; edit `_src/*.py` and rebuild.
Daily practice happens on copies: `python tools/new_session.py P2.1` → `practice/<date>/...`.

## Markers

| Marker | Effect in the practice template | Effect in the solution |
| --- | --- | --- |
| `### BEGIN SOLUTION` … `### END SOLUTION` inside a code cell | replaced by `raise NotImplementedError  # YOUR CODE HERE`, or by the block's `# TEMPLATE: <code>` lines (uncommented) if any | markers removed |
| `# %% tags=["solution"]` | whole cell replaced by its leading comment lines (the spec) + `# YOUR CODE HERE` | kept |
| `# %% tags=["solution_only"]` (code or markdown) | dropped | kept (use for model answers, reference plots) |
| `# %% tags=["template_only"]` | kept | dropped |

Two drill styles:
- **Blank drill** (end-to-end tasks like P2.1): numbered `tags=["solution"]` cells whose leading comments are a
  precise spec (names, shapes, hyperparameters). The template looks like a blank notebook with headings.
- **Fill-in drill** (by-hand implementations like P2.7): function signatures + docstrings kept, bodies inside
  `BEGIN/END SOLUTION`, and a test cell right after each function.

## Required structure of every notebook

1. **Header markdown cell** (first cell), containing, in this order:
   - `# <ID> · <Title>`
   - `**ID:** <ID> · **Target:** <N> min · **Plan week:** <w> · **Part:** <P|A|G1>`. `tools/drills.py` parses `**ID:**` and `**Target:**`.
     Labs that are not timed drills (most of G1) still carry an ID, and the target is the expected study time.
   - What the drill is, what it must include (quote the guide's "Must include" where one exists), the rules.
   - `## Sources`: every external resource used or adapted, as a bullet with authors/title/venue/year, an
     **arXiv ID or DOI** for papers, and a URL. For code adapted from a repo: repo URL, **commit SHA**, file path,
     and license (e.g. "adapted from karpathy/micrograd@7bc720e `micrograd/engine.py`, MIT").
2. **Setup cell**: imports, `from prep import Session, ...`, `s = Session("<ID>", target_min=N)`, seeding.
3. **Body**: markdown cells that teach (equations in LaTeX, short) and code cells to complete.
4. **Checks**: `prep.check_close` / `prep.check` against a trusted reference (PyTorch built-in, sklearn,
   torchvision, an analytic value, or finite differences). Checks must pass on the solution and must be
   strict enough to fail on the typical wrong answers (wrong normalization, wrong `dim`, off-by-one).
   Avoid flaky thresholds: seed everything and leave headroom.
5. **"Say it out loud"** markdown cell with interview questions, followed by a `solution_only` markdown cell
   with concise **model answers** (inside `<details>`). Claims in answers must be correct and, if non-obvious,
   cited.
6. **Last cell**: `s.finish(lookups=0, score=None, notes="")`.

## Quality bar

- Runs top to bottom on this machine (`.venv`, torch 2.14, RTX 3090, sm_86) via `tools/verify.py`.
  Single-GPU notebooks finish in < 5 min unless the header says otherwise. CPU fallback where reasonable.
- Datasets come from `prep.DATA` (pre-downloaded: MNIST, FashionMNIST, CIFAR-10, California housing
  (`data_home=DATA/"sklearn"`), AG News parquet in `DATA/"ag_news"`). Do not download into other places.
- Every numeric claim about hardware (bandwidth, TFLOPs) is either measured in the notebook or cited with source.
  RTX 3090 facts: Ampere GA102, compute capability 8.6, 24 GB GDDR6X, no FP8 tensor cores, no NVLink on this box
  (GPUs talk over PCIe; `nvidia-smi topo -m` shows `NODE`).
- No verbatim copying of long prose from sources; code adapted from permissively licensed repos is fine with
  attribution (SHA + path + license) in the header and a short comment at the adapted function.
- Match the existing code style: short, readable, comments only where they teach something.

## Licensing

The repo is MIT ([LICENSE](LICENSE)). If a notebook **copies, transcribes or adapts** code or text from another project
(not just cites it), check that project's license, keep its attribution in the notebook header (repo, commit, file,
license), and add a row plus the license text to [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Code you only read
for reference needs a citation, not a notice. Prefer permissively licensed sources (MIT, BSD, Apache-2.0, CC-BY), and
never copy from sources without a license or with a non-commercial / share-alike license.
