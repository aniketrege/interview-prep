# %% [markdown]
# # A3.12 · Beam search with EOS handling and length normalization
#
# **ID:** A3.12 · **Target:** 20 min · **Plan week:** 5 · **Part:** A
#
# **The drill.** Implement greedy decoding and beam search over a black-box language model
# `lm(prefix) -> (V,) log-probs`, with correct end-of-sequence handling and two length normalizations: the simple
# $\log P / |Y|^\alpha$ (HF's `length_penalty`) and GNMT's $\log P / \mathrm{lp}(Y)$ with
# $\mathrm{lp}(Y) = (5+|Y|)^\alpha / (5+1)^\alpha$ (Wu et al. 2016, §7, Eq. 14). The toy LMs are tiny (4 real tokens
# plus EOS, at most 5 steps) so an exhaustive search gives the exact best sequence to test against. Guide §A3 ("beam
# search") and §C4.
#
# **Rules of the beam search you write (the tests assume them):**
# 1. Hypotheses are tuples of token ids (no BOS). A hypothesis is **finished** when its last token is `EOS`, or when it
#    reaches `max_len` tokens (length-capped). $|Y|$ counts every generated token, EOS included.
# 2. Each step, call `lm` **once per live hypothesis**, form all `B·V` one-token extensions, and rank them by raw
#    cumulative log-prob (normalization is applied only to finished hypotheses, as HF does). Walk down the ranking:
#    extensions ending in EOS go to the finished pool, the others become the next live beam, until `beam_width` live
#    hypotheses are collected. Finished hypotheses are never extended.
# 3. Stop when no live hypothesis is left or after `max_len` steps (then the live ones are finished, length-capped).
#    Return the finished hypothesis with the best **normalized** score.
#
# **Deeper study:** Raschka, *LLMs from scratch* §5.3 (decoding strategies, the `generate` loop that beam search
# replaces): `../../LLMs-from-scratch/ch05/01_main-chapter-code/ch05.ipynb`. HF `generate()` beam search (link in
# Sources) for the batched, KV-cached version: top-`2·num_beams` candidates per step so that enough non-EOS ones remain,
# cache reordering by beam index, and an early-stopping heuristic.
#
# ## Sources
# - Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi et al., *Google's Neural Machine Translation System: Bridging the
#   Gap between Human and Machine Translation*, 2016, arXiv:1609.08144 — https://arxiv.org/abs/1609.08144
#   (§7 "Decoder": length normalization $|Y|^\alpha$ first, then Eq. 14 $\mathrm{lp}(Y)=(5+|Y|)^\alpha/(5+1)^\alpha$,
#   $\alpha\in[0.6, 0.7]$ best on dev; coverage penalty $\mathrm{cp}$ not implemented here).
# - F. Stahlberg, B. Byrne, *On NMT Search Errors and Model Errors: Cat Got Your Tongue?*, EMNLP-IJCNLP 2019,
#   arXiv:1908.10090 — https://arxiv.org/abs/1908.10090 (exact search: the global best under a Transformer NMT model is
#   the empty translation for more than 50% of WMT15 En-De sentences).
# - C. Meister, T. Vieira, R. Cotterell, *If beam search is the answer, what was the question?*, EMNLP 2020,
#   arXiv:2010.02650 — https://arxiv.org/abs/2010.02650 (exact MAP decoding often gives bad text; beam search's search
#   errors act as a useful bias).
# - A. K. Vijayakumar et al., *Diverse Beam Search*, 2016, arXiv:1610.02424 — https://arxiv.org/abs/1610.02424
# - Hugging Face `transformers` `src/transformers/generation/utils.py` at commit
#   `8735a4a12bdf3e737978b30520bb22a8a710fd67` (Apache-2.0): beam search divides finished scores by
#   `(generated_len) ** length_penalty` (line 3537, generated length includes the EOS token), early-stopping heuristic
#   `_check_early_stop_heuristic` (lines 3361-3405), default `length_penalty=1.0` in `configuration_utils.py` —
#   https://github.com/huggingface/transformers/blob/8735a4a12bdf3e737978b30520bb22a8a710fd67/src/transformers/generation/utils.py
# - S. Raschka, *Build a Large Language Model (From Scratch)*, ch. 5 code, commit f9313ea (Apache-2.0) —
#   https://github.com/rasbt/LLMs-from-scratch

# %%
import itertools
import math

import torch

from prep import Session, check, check_close

s = Session("A3.12", target_min=20)
torch.manual_seed(0)
EOS = 0          # token 0 is end-of-sequence; tokens 1..V-1 are "real"


class TableLM:
    """First-order Markov LM: next-token log-probs depend only on the last token (row V = start state).
    `calls` counts forward passes so we can check you never extend finished hypotheses."""

    def __init__(self, V=5, seed=0, sharpness=1.5, eos_bias=0.0):
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(V + 1, V, generator=g, dtype=torch.float64) * sharpness
        z[:, EOS] += eos_bias
        self.V, self.logp, self.calls = V, z.log_softmax(-1), 0

    def __call__(self, prefix):
        self.calls += 1
        return self.logp[prefix[-1] if prefix else self.V]


class TreeLM:
    """Hand-written LM: next-token probabilities for a few prefixes, `default` for every other prefix."""

    def __init__(self, table, default):
        self.table = {k: torch.tensor(v, dtype=torch.float64).log() for k, v in table.items()}
        self.default, self.V, self.calls = torch.tensor(default, dtype=torch.float64).log(), len(default), 0

    def __call__(self, prefix):
        self.calls += 1
        return self.table.get(tuple(prefix), self.default)


#               EOS    a     b     c
tree = TreeLM({(): [0.02, 0.40, 0.33, 0.25],
               (1,): [0.20, 0.30, 0.26, 0.24],
               (2,): [0.60, 0.20, 0.10, 0.10],
               (3,): [0.97, 0.01, 0.01, 0.01]},
              default=[0.50, 0.20, 0.15, 0.15])
A, B_, C = 1, 2, 3


def exact_search(lm, max_len, kind="none", alpha=0.0):
    """Reference: enumerate every finished sequence (EOS-terminated or length-capped) and return the best
    (seq, normalized score). Independent of your code; it has its own copy of the length penalties."""
    lp = {"none": lambda n: 1.0, "avg": lambda n: n ** alpha, "gnmt": lambda n: ((5 + n) / 6) ** alpha}[kind]
    best = (None, -math.inf)

    def visit(prefix, logp):
        nonlocal best
        row = lm(prefix).tolist()
        for t in range(lm.V):
            seq, sc = prefix + (t,), logp + row[t]
            if t == EOS or len(seq) == max_len:
                if sc / lp(len(seq)) > best[1]:
                    best = (seq, sc / lp(len(seq)))
            else:
                visit(seq, sc)

    visit((), 0.0)
    return best


def seq_logp(lm, seq):
    return sum(lm(seq[:i])[t].item() for i, t in enumerate(seq))

# %% [markdown]
# ## 1. Greedy decoding
# Take the argmax at every step until EOS or `max_len`. Return the sequence and its total log-prob.

# %%
def greedy_decode(lm, max_len):
    """-> (seq: tuple[int, ...], logp: float)"""
    ### BEGIN SOLUTION
    seq, logp = (), 0.0
    for _ in range(max_len):
        row = lm(seq)
        t = int(row.argmax())
        seq, logp = seq + (t,), logp + row[t].item()
        if t == EOS:
            break
    return seq, logp
    ### END SOLUTION


g_seq, g_lp = greedy_decode(tree, max_len=3)
check("greedy on the tree LM is (a, a, EOS)", g_seq == (A, A, EOS), f"got {g_seq}")
check_close("greedy log-prob = log(.40 * .30 * .50)", g_lp, math.log(0.4 * 0.3 * 0.5))
lm_noeos = TableLM(seed=3, eos_bias=-30.0)
check("greedy stops at max_len when EOS never wins", len(greedy_decode(lm_noeos, 4)[0]) == 4)

# %% [markdown]
# ## 2. Length penalties
# Raw log-probs only decrease as a sequence grows, so maximizing them favors short outputs (in the extreme, the
# empty translation; see Stahlberg & Byrne). The fixes divide the finished score by a length term:
# - `"none"`: $1$
# - `"avg"`: $|Y|^\alpha$ ($\alpha=1$ is the mean per-token log-prob; HF's `length_penalty` is this $\alpha$)
# - `"gnmt"`: $\big((5+|Y|)/6\big)^\alpha$ (Wu et al. Eq. 14, $\alpha = 0$ recovers plain beam search)

# %%
def length_penalty(n, alpha, kind):
    """Divisor for a finished hypothesis of n tokens (EOS included)."""
    ### BEGIN SOLUTION
    if kind == "none":
        return 1.0
    if kind == "avg":
        return n ** alpha
    if kind == "gnmt":
        return ((5 + n) / 6) ** alpha
    raise ValueError(kind)
    ### END SOLUTION


check_close("lp none", length_penalty(4, 0.6, "none"), 1.0)
check_close("lp avg alpha=1", length_penalty(4, 1.0, "avg"), 4.0)
check_close("lp avg alpha=0.6", length_penalty(4, 0.6, "avg"), 4 ** 0.6)
check_close("lp gnmt alpha=0.6, |Y|=4", length_penalty(4, 0.6, "gnmt"), 1.5 ** 0.6)
check_close("lp gnmt |Y|=1 is exactly 1", length_penalty(1, 0.7, "gnmt"), 1.0)
check_close("lp gnmt alpha=0 is 1", length_penalty(9, 0.0, "gnmt"), 1.0)

# %% [markdown]
# ## 3. Beam search
# Follow rules 1-3 in the header. Watch for: never extending a hypothesis that ended in EOS, ranking *all* `B·V`
# extensions globally (not top-B per beam), using raw scores for pruning but normalized scores for the final choice,
# and finishing the live beams at `max_len`. (All candidates at one step have the same length, so ranking them by raw
# or by normalized score is the same; normalization only matters when finished hypotheses of different lengths are
# compared.)

# %%
def beam_search(lm, beam_width, max_len, alpha=0.0, kind="none"):
    """-> (best_seq: tuple[int, ...], best_normalized_score: float)"""
    ### BEGIN SOLUTION
    live, finished = [((), 0.0)], []
    for step in range(max_len):
        cands = []
        for seq, sc in live:
            row = lm(seq).tolist()                       # one forward pass per live hypothesis
            cands += [(seq + (t,), sc + row[t]) for t in range(len(row))]
        cands.sort(key=lambda c: c[1], reverse=True)     # rank by raw log-prob
        live = []
        for seq, sc in cands:
            if seq[-1] == EOS:
                finished.append((seq, sc))
            else:
                live.append((seq, sc))
                if len(live) == beam_width:
                    break
        if step == max_len - 1:
            finished += live                             # length-capped hypotheses
        if not live:
            break
    return max(((seq, sc / length_penalty(len(seq), alpha, kind)) for seq, sc in finished), key=lambda c: c[1])
    ### END SOLUTION


# hand-traceable example (trace it on paper: step 1 keeps {a, b}; step 2 ranks bE .198 > aa .12 > ab .104 > ...)
b_seq, b_sc = beam_search(tree, beam_width=2, max_len=3)
e_seq, e_sc = exact_search(tree, max_len=3)
print(f"greedy {g_seq} p={math.exp(g_lp):.4f} | beam(2) {b_seq} p={math.exp(b_sc):.4f} | exact {e_seq} p={math.exp(e_sc):.4f}")
check("beam(2) on the tree LM is (b, EOS) with p = .33 * .60", b_seq == (B_, EOS) and abs(math.exp(b_sc) - 0.198) < 1e-9,
      f"got {b_seq}, p={math.exp(b_sc):.4f}")
check("exact best on the tree LM is (c, EOS): beam(2) pruned c at step 1", e_seq == (C, EOS))
check("greedy != beam != exact", len({g_seq, b_seq, e_seq}) == 3)

tree.calls = 0
beam_search(tree, beam_width=2, max_len=3)
check("one lm call per live hypothesis per step (<= 1 + 2 + 2)", tree.calls <= 5, f"{tree.calls} calls")

# %% [markdown]
# ### Against exhaustive search and greedy, on random LMs
# With `beam_width >= 4**max_len` nothing is ever pruned, so beam search must equal exact search under every
# normalization. With `beam_width = 1` and no normalization it must equal greedy (a finished EOS hypothesis outranks
# everything extended after it, because log-probs only decrease).

# %%
MAXLEN = 5
for seed in range(20):
    lm = TableLM(seed=seed)
    for kind, alpha in [("none", 0.0), ("avg", 1.0), ("gnmt", 0.6)]:
        bs, bsc = beam_search(lm, beam_width=4 ** MAXLEN, max_len=MAXLEN, alpha=alpha, kind=kind)
        es, esc = exact_search(lm, MAXLEN, kind, alpha)
        assert bs == es and abs(bsc - esc) < 1e-9, f"seed {seed} {kind}: beam {bs} {bsc:.4f} vs exact {es} {esc:.4f}"
    assert beam_search(lm, 1, MAXLEN)[0] == greedy_decode(lm, MAXLEN)[0], f"seed {seed}: beam(1) != greedy"
check("full-width beam == exact search (20 LMs x 3 normalizations)", True)
check("beam(1) == greedy (20 LMs)", True)

bs, bsc = beam_search(lm_noeos, beam_width=3, max_len=4)
check("length-capped hypotheses are finished at max_len", len(bs) == 4 and EOS not in bs, f"got {bs}")
check_close("returned score is the sequence's log-prob (kind='none')", bsc, seq_logp(lm_noeos, bs))
for seed in range(20):
    lm = TableLM(seed=seed)
    for B in (1, 2, 3):
        seq, _ = beam_search(lm, B, MAXLEN, alpha=0.6, kind="gnmt")
        assert len(seq) <= MAXLEN and EOS not in seq[:-1], f"seed {seed} B={B}: EOS inside {seq}"
check("EOS only ever appears as the last token", True)

# %% [markdown]
# ### Length bias, measured
# Many random LMs with a mild EOS bias: average length of the chosen output under each scoring rule (beam width 4)
# and how often beam search misses the exact optimum (a *search error*).

# %%
rows = []
for kind, alpha in [("none", 0.0), ("avg", 1.0), ("gnmt", 0.6), ("gnmt", 1.0)]:
    lens, errs = [], 0
    for seed in range(200):
        lm = TableLM(seed=1000 + seed, eos_bias=0.5)
        seq, sc = beam_search(lm, 4, MAXLEN, alpha, kind)
        es, esc = exact_search(lm, MAXLEN, kind, alpha)
        lens.append(len(seq))
        errs += sc < esc - 1e-12
    rows.append((kind, alpha, sum(lens) / len(lens), errs / 200))
    print(f"{kind:>5} alpha={alpha:.1f}: mean |Y| = {rows[-1][2]:.2f}, search-error rate (B=4) = {rows[-1][3]:.1%}")
check("normalizing by length gives longer outputs than raw log-prob", rows[1][2] > rows[0][2] and rows[3][2] > rows[0][2])

# %% [markdown]
# ## Say it out loud
# 1. Why does greedy fail on the tree LM? Why does beam(2) fail? What is a *search error* vs a *model error*?
# 2. Why do unnormalized scores favor short outputs? Write the GNMT length penalty and say what $\alpha=0$ means.
# 3. What exactly happens to a hypothesis that emits EOS? Why does HF take the top `2·num_beams` candidates per step?
# 4. When can you stop early without changing the result, with and without length normalization?
# 5. Cost of beam search per step in forward passes, FLOPs and KV-cache memory. What happens to the cache when beams
#    are reordered?
# 6. Is a bigger beam always better? Why is beam search rarely used for open-ended LLM chat but common for MT/ASR?
# 7. Name one fix for beams collapsing to near-duplicates.

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Greedy commits to the locally best `a` (.40) and every continuation of `a` is flat, so it ends with .06.
#    Beam(2) keeps {a, b} at step 1 and prunes `c` (.25), even though `c` is followed by EOS with prob .97
#    (total .2425 > .198). The model prefers `cE` but the search never finds it: that is a *search error*. A *model
#    error* is when the exact argmax itself is a bad output. Stahlberg & Byrne (arXiv:1908.10090) found that for more
#    than half of WMT15 En-De sentences the exact argmax was the empty translation.
# 2. Each token adds $\log p \le 0$, so longer hypotheses can only lose score. Dividing by a length term rewards
#    length. GNMT: $s(Y) = \log P(Y|X) / \mathrm{lp}(Y)$ with $\mathrm{lp}(Y) = (5+|Y|)^\alpha/(5+1)^\alpha$
#    (plus a coverage penalty); $\alpha=0$ gives plain max-probability beam search, and Wu et al. found
#    $\alpha\in[0.6,0.7]$ best for the simpler $|Y|^\alpha$ form. The demo above measures longer outputs under both normalizations.
# 3. It moves to the finished pool with its score (normalized when compared) and is never extended. It still
#    occupies a rank, so if several of the top-B candidates are EOS the live beam would shrink. HF ranks the top
#    `2·num_beams` candidates so that at least `num_beams` non-EOS continuations remain (at most `num_beams` can
#    be EOS). My rule ("walk the ranking until B live ones are found") has the same effect.
# 4. Without normalization, scores only decrease, so once the best finished score is ≥ the best live raw score,
#    no live beam can win and you can stop (exact w.r.t. the beam). With a length penalty, a longer hypothesis can
#    still overtake. You then need a bound such as the best live score normalized at `max_len` (HF's
#    `early_stopping="never"` heuristic for positive `length_penalty`) or you just run to `max_len`. HF's docstring
#    notes that its default early-stopping is a heuristic rather than canonical beam search.
# 5. Each step runs one batched forward pass over B sequences (B× the FLOPs and B× the KV cache of greedy), then a
#    top-k over B·V scores. After selection, the surviving beams' parents are a permutation/duplication of the old
#    beams, so the KV cache must be gathered along the batch dimension by beam index (`index_select`). A paged
#    cache can share prefix blocks instead of copying them.
# 6. No. Larger beams reduce search error but can lower quality, because the model's argmax is often degenerate
#    (short, repetitive, generic). Meister et al. (arXiv:2010.02650) argue that beam search's bias is part of
#    why it works. For open-ended generation, high-probability text is bland and repetitive (Holtzman et al.
#    arXiv:1904.09751), so sampling is used. Beam search suits tasks with a narrow correct output (MT, ASR,
#    constrained decoding).
# 7. Diverse beam search (Vijayakumar et al., arXiv:1610.02424): split beams into groups and penalize tokens chosen
#    by earlier groups at the same step. Alternatives are n-gram blocking and sampling-based stochastic beam search.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
