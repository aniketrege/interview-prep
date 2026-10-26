# %% [markdown]
# # A3.11 · Sampling: temperature, top-k, top-p, min-p, repetition penalty
#
# **ID:** A3.11 · **Target:** 15 min · **Plan week:** 5 · **Part:** A
#
# **The drill.** Write the next-token sampling stack of an LLM decoder from a blank function body: greedy,
# temperature, top-k, top-p (nucleus), min-p, a CTRL-style repetition penalty, and the combined pipeline in the
# order Hugging Face `generate()` applies them. Every function takes a batch of logits `(B, V)` and returns
# logits of the same shape with filtered entries set to `-inf`, so the steps compose. Guide §A3 ("Top-k / top-p /
# temperature sampling") and §C4 ("greedy/temperature/top-k/top-p/min-p, ... repetition penalties").
#
# **Conventions (they are what the tests check, and they match HF `transformers`):**
# - **top-k** removes a token iff its logit is *strictly below* the k-th largest, so ties with the k-th value survive.
# - **top-p** keeps the *smallest* set of most probable tokens whose mass is $\ge p$ (Holtzman et al., Eq. 2). The token
#   that crosses $p$ is **kept**: keep token $i$ iff the mass of the tokens ranked above it is $< p$. Always keep at
#   least `min_tokens_to_keep` (default 1).
# - **min-p** removes tokens with $p_i < p_\text{base}\cdot p_\max$ (Nguyen et al., Eq. 1-2).
# - **repetition penalty** $\theta$: for every token id already in the context (prompt included, counted once),
#   divide its logit by $\theta$ if positive, multiply by $\theta$ if negative.
# - **pipeline order:** repetition penalty → temperature → top-k → top-p → min-p → softmax → sample.
#   `temperature == 0` means greedy (argmax of the penalized logits), as in vLLM.
#
# **Allowed:** `softmax`, `sort`, `topk`, `cumsum`, `scatter`, `gather`, `masked_fill`, `where`, `torch.multinomial`.
# Fill in every `raise NotImplementedError`; the cell under each function prints `ok` or says what is off.
#
# **Deeper study:** Raschka, *LLMs from scratch* §5.3 "Decoding strategies to control randomness" (temperature, top-k,
# the `generate` function): `../../LLMs-from-scratch/ch05/01_main-chapter-code/ch05.ipynb`. Raschka, *Reasoning from
# scratch* §4.4 (temperature) and §4.5 (top-p, including the "keep the token that crosses p" variant):
# https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch04/01_main-chapter-code/ch04_main.ipynb
#
# ## Sources
# - A. Holtzman, J. Buys, L. Du, M. Forbes, Y. Choi, *The Curious Case of Neural Text Degeneration*, ICLR 2020,
#   arXiv:1904.09751 — https://arxiv.org/abs/1904.09751 (nucleus sampling, Eq. 2-3; flat vs peaked distributions
#   argument against a fixed k, §3.2-3.3).
# - N. S. Keskar, B. McCann, L. R. Varshney, C. Xiong, R. Socher, *CTRL: A Conditional Transformer Language Model for
#   Controllable Generation*, 2019, arXiv:1909.05858 — https://arxiv.org/abs/1909.05858 (penalized sampling, §4.1:
#   $p_i \propto \exp(x_i / (T\cdot I(i\in g)))$, $I = \theta \approx 1.2$ for seen tokens).
# - M. N. Nguyen, A. Baker, C. Neo, A. Roush, A. Kirsch, R. Shwartz-Ziv, *Turning Up the Heat: Min-p Sampling for
#   Creative and Coherent LLM Outputs*, ICLR 2025, arXiv:2407.01082 — https://arxiv.org/abs/2407.01082 (§3.1, Eq. 1-3).
# - Hugging Face `transformers`, `src/transformers/generation/logits_process.py` at commit
#   `8735a4a12bdf3e737978b30520bb22a8a710fd67` (Apache-2.0): `TemperatureLogitsWarper`,
#   `RepetitionPenaltyLogitsProcessor`, `TopPLogitsWarper`, `TopKLogitsWarper`, `MinPLogitsWarper` (lines 238-795) —
#   https://github.com/huggingface/transformers/blob/8735a4a12bdf3e737978b30520bb22a8a710fd67/src/transformers/generation/logits_process.py
#   Processor order from `GenerationMixin._get_logits_processor` in `generation/utils.py` (lines 1410-1650; repetition
#   penalty at 1467, temperature/top-k/top-p/min-p at 1605-1619), and `GenerationConfig` defaults (`top_k=50`) in
#   `generation/configuration_utils.py` — same commit. The `solution_only` cell at the end transcribes these warpers
#   (Apache-2.0) to cross-check the solution.
# - vLLM `vllm/sampling_params.py` (temperature 0 = greedy; presence/frequency/repetition penalties) and
#   `vllm/model_executor/layers/utils.py::apply_penalties` (additive OpenAI-style frequency/presence penalties) at commit
#   `c723f6d3ee5c832f7d4a64aaa8dfda5cecbf9eef` (Apache-2.0) — https://github.com/vllm-project/vllm/blob/c723f6d3ee5c832f7d4a64aaa8dfda5cecbf9eef/vllm/sampling_params.py
# - S. Raschka, *Build a Large Language Model (From Scratch)*, ch. 5 code, commit f9313ea (Apache-2.0); S. Raschka,
#   *Build a Reasoning Model (From Scratch)*, ch. 4 code, commit a788466 (Apache-2.0) — https://github.com/rasbt/reasoning-from-scratch

# %%
import math

import torch

from prep import Session, check, check_close

s = Session("A3.11", target_min=15)
torch.manual_seed(0)
NEG_INF = float("-inf")


def logits_from_probs(p):
    """Logits whose softmax is p (rows of p must sum to 1)."""
    return torch.as_tensor(p, dtype=torch.float64).log()


def kept(z):
    """Boolean mask of entries that survived a filter."""
    return torch.isfinite(z)


def tv(p, q):
    """Total-variation distance between two probability vectors."""
    return 0.5 * (torch.as_tensor(p, dtype=torch.float64) - torch.as_tensor(q, dtype=torch.float64)).abs().sum().item()


# handcrafted distribution, deliberately NOT sorted, so you must map sorted positions back to vocab ids
P6 = torch.tensor([0.10, 0.40, 0.04, 0.25, 0.06, 0.15], dtype=torch.float64)
Z6 = logits_from_probs(P6)[None]            # (1, 6); ranks: id1 .40 > id3 .25 > id5 .15 > id0 .10 > id4 .06 > id2 .04
Zr = torch.randn(64, 100, dtype=torch.float64) * 3   # random batch for property checks

# %% [markdown]
# ## 1. Greedy and temperature
# $\mathrm{softmax}(z/T)$: $T<1$ sharpens, $T>1$ flattens, $T\to 0$ tends to one-hot on the argmax and $T\to\infty$ to
# uniform. Temperature does not change the ranking, so greedy decoding is unaffected by it.

# %%
def greedy(logits):
    """(B, V) -> (B,) argmax token ids."""
    ### BEGIN SOLUTION
    return logits.argmax(dim=-1)
    ### END SOLUTION


def apply_temperature(logits, temperature):
    """Return logits / T for T > 0 (T == 0 is handled as greedy in sample_next)."""
    ### BEGIN SOLUTION
    return logits / temperature
    ### END SOLUTION


check("greedy", torch.equal(greedy(Zr), Zr.max(-1).indices))
check_close("temperature 0.7", apply_temperature(Zr, 0.7).softmax(-1), (Zr / 0.7).softmax(-1))
onehot = torch.nn.functional.one_hot(Zr.argmax(-1), 100).double()
check_close("T -> 0 gives one-hot argmax", apply_temperature(Zr, 1e-6).softmax(-1), onehot, atol=1e-6)
check_close("T -> inf gives uniform", apply_temperature(Zr, 1e6).softmax(-1), torch.full_like(Zr, 0.01), atol=1e-6)

# %% [markdown]
# ## 2. Top-k
# Keep the k largest logits per row. Use the k-th largest **value** as a threshold (remove `z < kth`), not the
# `topk` **indices**: with ties at the boundary the value rule keeps every tied token (HF semantics), the index rule
# keeps an arbitrary subset. `k >= V` must keep everything.

# %%
def top_k_filter(logits, k):
    """(B, V) -> (B, V), entries outside the top-k set to -inf. Ties with the k-th value are kept."""
    ### BEGIN SOLUTION
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]       # (B, 1)
    return logits.masked_fill(logits < kth, NEG_INF)
    ### END SOLUTION


zt = torch.tensor([[3.0, 1.0, 2.0, 2.0, 0.0], [0.0, 5.0, -1.0, 4.0, 4.5]])
want = torch.tensor([[3.0, NEG_INF, 2.0, 2.0, NEG_INF], [NEG_INF, 5.0, NEG_INF, NEG_INF, 4.5]])
check("top-k hand case (tie at the k-th value kept)", torch.equal(top_k_filter(zt, 2), want), f"got {top_k_filter(zt, 2)}")
check("top-k with k >= V keeps all", torch.equal(top_k_filter(zt, 10), zt))
zk = top_k_filter(Zr, 7)
idx = Zr.topk(7, dim=-1).indices
check("top-k random batch: exactly the topk indices survive",
      torch.equal(kept(zk), torch.zeros_like(kept(zk)).scatter(1, idx, True)))
check("top-k leaves kept logits unchanged", torch.equal(zk[kept(zk)], Zr[kept(zk)]))

# %% [markdown]
# ## 3. Top-p (nucleus)
# Sort probabilities descending, let $c_i$ be the cumulative sum and $m_i = c_i - p_{(i)}$ the mass **strictly above**
# rank $i$. Keep rank $i$ iff $m_i < p$. That keeps the token that crosses $p$ and gives the smallest set with mass
# $\ge p$. The common bug `keep c_i <= p` drops the crossing token and can return an empty set (if $p_{(1)} > p$).
# HF computes the same set by sorting ascending and removing `cumsum <= 1 - top_p`.
#
# Exact equality ($m_i = p$ to the last bit) is a measure-zero float coincidence: HF then removes the token, but
# rounding in `softmax`/`cumsum` can flip it either way, so the tests probe just below and just above boundaries.

# %%
def top_p_filter(logits, top_p, min_tokens_to_keep=1):
    """(B, V) -> (B, V). Keep the smallest set of most probable tokens with total probability >= top_p
    (the crossing token is kept), and at least `min_tokens_to_keep` tokens. Entries already -inf stay -inf."""
    ### BEGIN SOLUTION
    probs = logits.softmax(dim=-1)
    sorted_p, order = probs.sort(dim=-1, descending=True)
    mass_above = sorted_p.cumsum(dim=-1) - sorted_p            # mass of strictly higher-ranked tokens
    remove_sorted = mass_above >= top_p
    remove_sorted[..., :min_tokens_to_keep] = False
    remove = remove_sorted.scatter(-1, order, remove_sorted)   # back to vocab order
    return logits.masked_fill(remove, NEG_INF)
    ### END SOLUTION


# cumulative mass in rank order: .40, .65, .80, .90, .96, 1.0
for p, ids in [(0.30, {1}), (0.6499, {1, 3}), (0.6501, {1, 3, 5}), (0.70, {1, 3, 5}), (0.85, {1, 3, 5, 0}),
               (1.0, {0, 1, 2, 3, 4, 5})]:
    got = set(kept(top_p_filter(Z6, p))[0].nonzero().flatten().tolist())
    check(f"top-p={p} keeps ids {sorted(ids)}", got == ids, f"got {sorted(got)}")
check("top-p keeps >= 1 token even for tiny p", kept(top_p_filter(Z6, 1e-9)).sum().item() == 1)
check("top-p min_tokens_to_keep=3", kept(top_p_filter(Z6, 1e-9, min_tokens_to_keep=3)).sum().item() == 3)
check("top-p after top-k: -inf stays -inf", not kept(top_p_filter(top_k_filter(Z6, 2), 1.0))[0, [0, 2, 4, 5]].any())

# properties that pin the nucleus down uniquely (no ties in random float logits)
pr, zp = Zr.softmax(-1), top_p_filter(Zr, 0.9)
keep = kept(zp)
mass = (pr * keep).sum(-1)
min_kept = pr.masked_fill(~keep, 2.0).amin(-1)
max_removed = pr.masked_fill(keep, -1.0).amax(-1)
check("top-p random batch: kept mass >= p", bool((mass >= 0.9).all()))
check("top-p random batch: minimal (dropping the smallest kept token goes below p)", bool((mass - min_kept < 0.9).all()))
check("top-p random batch: kept tokens are the most probable ones", bool((min_kept > max_removed).all()))

# %% [markdown]
# ## 4. Min-p
# Threshold relative to the most likely token: remove $p_i < p_\text{base}\,p_\max$. When the model is confident
# ($p_\max$ large) the filter is aggressive; when it is unsure, many tokens survive. The argmax always survives
# because $p_\text{base} \le 1$.

# %%
def min_p_filter(logits, min_p):
    """(B, V) -> (B, V), tokens with prob < min_p * max_prob set to -inf."""
    ### BEGIN SOLUTION
    probs = logits.softmax(dim=-1)
    return logits.masked_fill(probs < min_p * probs.amax(dim=-1, keepdim=True), NEG_INF)
    ### END SOLUTION


# p_max = .40: threshold .19 * .40 = .076 keeps {.40,.25,.15,.10}; .26 * .40 = .104 keeps {.40,.25,.15}
for mp, ids in [(0.19, {1, 3, 5, 0}), (0.26, {1, 3, 5}), (0.0, set(range(6))), (1.0, {1})]:
    got = set(kept(min_p_filter(Z6, mp))[0].nonzero().flatten().tolist())
    check(f"min-p={mp} keeps ids {sorted(ids)}", got == ids, f"got {sorted(got)}")
pr = Zr.softmax(-1)
check("min-p random batch", torch.equal(kept(min_p_filter(Zr, 0.1)), pr >= 0.1 * pr.amax(-1, keepdim=True)))

# %% [markdown]
# ## 5. Repetition penalty (CTRL §4.1, HF sign convention)
# CTRL divides the logit of every previously generated token by $\theta$ (it folds $\theta$ into the temperature). For a
# **negative** logit, dividing by $\theta > 1$ moves it towards 0 and *raises* its probability, so HF (and vLLM)
# multiply negative logits by $\theta$ instead. Each seen id is penalized once, however often it occurs, and in HF
# the prompt counts as "seen" for decoder-only models. Rows of the batch have their own histories.

# %%
def repetition_penalty(logits, prev_ids, penalty):
    """logits: (B, V); prev_ids: (B, L) long, all context tokens so far (prompt included).
    Seen tokens: z / penalty if z > 0 else z * penalty. Unseen tokens unchanged. Do not modify `logits` in place."""
    ### BEGIN SOLUTION
    seen = torch.zeros_like(logits, dtype=torch.bool).scatter(1, prev_ids, True)
    penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalized, logits)
    ### END SOLUTION


zr = torch.tensor([[2.0, -1.0, 0.5, 3.0, 0.0], [2.0, -1.0, 0.5, 3.0, 0.0]])
prev = torch.tensor([[0, 1, 1, 1], [3, 4, 3, 3]])     # row 0 has seen {0, 1}; row 1 has seen {3, 4}
want = torch.tensor([[1.0, -2.0, 0.5, 3.0, 0.0], [2.0, -1.0, 0.5, 1.5, 0.0]])
zr_copy = zr.clone()
check_close("repetition penalty hand case (theta=2)", repetition_penalty(zr, prev, 2.0), want)
check("repetition penalty does not modify its input", torch.equal(zr, zr_copy))
check("repetition penalty theta=1 is the identity", torch.equal(repetition_penalty(Zr, Zr.argsort(-1)[:, :5], 1.0), Zr))
pr0, pr1 = zr.softmax(-1), repetition_penalty(zr, prev, 1.5).softmax(-1)
check("theta > 1 lowers the probability of every seen token", bool((pr1[0, [0, 1]] < pr0[0, [0, 1]]).all()))

# %% [markdown]
# ## 6. The pipeline
# Order (HF `_get_logits_processor`): repetition penalty → temperature → top-k → top-p → min-p. The order matters:
# temperature before top-p means a hot sample also has a *bigger nucleus*; reversing them gives a different
# distribution. `sample_next` turns the result into probabilities and draws one token per row with
# `torch.multinomial(..., generator=generator)`; `temperature == 0` returns greedy on the penalized logits.

# %%
def process_logits(logits, prev_ids=None, temperature=1.0, top_k=None, top_p=None, min_p=None, rep_penalty=None):
    """Apply the enabled steps in HF order; None (or rep_penalty 1.0) disables a step. Requires temperature > 0."""
    ### BEGIN SOLUTION
    z = logits
    if rep_penalty is not None and rep_penalty != 1.0:
        z = repetition_penalty(z, prev_ids, rep_penalty)
    if temperature != 1.0:
        z = apply_temperature(z, temperature)
    if top_k is not None:
        z = top_k_filter(z, top_k)
    if top_p is not None and top_p < 1.0:
        z = top_p_filter(z, top_p)
    if min_p is not None:
        z = min_p_filter(z, min_p)
    return z
    ### END SOLUTION


def sample_next(logits, prev_ids=None, temperature=1.0, top_k=None, top_p=None, min_p=None, rep_penalty=None,
                generator=None):
    """(B, V) -> (B,) sampled ids. temperature == 0: greedy after the repetition penalty."""
    ### BEGIN SOLUTION
    if temperature == 0:
        z = logits if rep_penalty is None else repetition_penalty(logits, prev_ids, rep_penalty)
        return greedy(z)
    z = process_logits(logits, prev_ids, temperature, top_k, top_p, min_p, rep_penalty)
    return torch.multinomial(z.softmax(dim=-1), 1, generator=generator).squeeze(-1)
    ### END SOLUTION


check("pipeline with everything disabled is the identity", torch.equal(process_logits(Zr), Zr))
# order: probs (.6, .2, .1, .06, .04), top_p=.75. T=1 nucleus = 2 tokens; T=2 first flattens -> 3 tokens
zo = logits_from_probs([0.6, 0.2, 0.1, 0.06, 0.04])[None]
check("top-p alone keeps 2", kept(process_logits(zo, top_p=0.75)).sum().item() == 2)
check("temperature is applied BEFORE top-p (T=2 -> nucleus of 3)",
      kept(process_logits(zo, temperature=2.0, top_p=0.75)).sum().item() == 3)
# repetition penalty comes first and changes the greedy choice; temperature 0 must still apply it
zg = torch.tensor([[2.0, 1.9, -3.0]])
check("temperature 0 is greedy", sample_next(zg, temperature=0).item() == 0)
check("temperature 0 still applies the repetition penalty",
      sample_next(zg, prev_ids=torch.tensor([[0]]), temperature=0, rep_penalty=1.3).item() == 1)

# %% [markdown]
# ### Statistical checks
# Draw $N = 200{,}000$ tokens with a fixed `torch.Generator` and compare empirical frequencies with the
# filtered distribution you can write down by hand. Monte-Carlo TV error here is about 0.002; the threshold is 0.01.
# Filtered-out tokens must have count exactly 0.

# %%
N = 200_000
g = torch.Generator().manual_seed(1234)


def freqs(ids, V):
    return torch.bincount(ids, minlength=V).double() / ids.numel()


# (a) top-p = 0.7 on P6 keeps ids {1, 3, 5} with mass .80 -> renormalized (.5, .3125, .1875)
want_a = torch.tensor([0, 0.40, 0, 0.25, 0, 0.15], dtype=torch.float64) / 0.80
f_a = freqs(sample_next(Z6.expand(N, -1), top_p=0.7, generator=g), 6)
check("sampling top-p=0.7: never samples filtered ids", bool((f_a[[0, 2, 4]] == 0).all()))
check("sampling top-p=0.7: matches the renormalized nucleus", tv(f_a, want_a) < 0.01, f"TV={tv(f_a, want_a):.4f}")
# (b) temperature only: P^(1/T) renormalized
want_b = (P6 ** (1 / 0.5)) / (P6 ** (1 / 0.5)).sum()
f_b = freqs(sample_next(Z6.expand(N, -1), temperature=0.5, generator=g), 6)
check("sampling T=0.5 matches P^(1/T)/Z", tv(f_b, want_b) < 0.01, f"TV={tv(f_b, want_b):.4f}")
# (c) everything on, reference built step by step in plain Python: penalty 2 on ids {1, 2} (all P6 logits are
#     negative, so both get multiplied by 2), T = 0.8, top-k = 4, top-p = 0.8 on the top-k survivors, min-p = 0.55
#     (each of top-k, top-p and min-p removes at least one token here)
prev_c = torch.tensor([[1, 1, 2]])
z_ref = [2 * v if i in (1, 2) else v for i, v in enumerate(Z6[0].tolist())]
w = [math.exp(v / 0.8) for v in z_ref]
order = sorted(range(6), key=lambda i: -w[i])[:4]                   # top-k = 4
tot = sum(w[i] for i in order)
nucleus, mass = [], 0.0
for i in order:                                                     # top-p = 0.8, renormalized over the top-k set
    if mass < 0.8:
        nucleus.append(i)
    mass += w[i] / tot
w_max = max(w[i] for i in nucleus)
final = [i for i in nucleus if w[i] >= 0.55 * w_max]                # min-p = 0.55 (ratio test: normalization-free)
keep_c = torch.zeros(6, dtype=torch.bool)
keep_c[final] = True
p_c = torch.tensor(w, dtype=torch.float64)
want_c = torch.where(keep_c, p_c, 0.0) / (p_c * keep_c).sum()
print("kept ids:", final)
f_c = freqs(sample_next(Z6.expand(N, -1), prev_ids=prev_c.expand(N, -1), temperature=0.8, top_k=4, top_p=0.8,
                        min_p=0.55, rep_penalty=2.0, generator=g), 6)
check("sampling full pipeline matches the hand-built distribution", tv(f_c, want_c) < 0.01, f"TV={tv(f_c, want_c):.4f}")
check("sampling full pipeline: filtered ids never drawn", bool((f_c[~keep_c] == 0).all()))
print("expected:", want_c.numpy().round(4), "\nempirical:", f_c.numpy().round(4))

# %% [markdown] tags=["solution_only"]
# ### Cross-check against Hugging Face's warpers
# Line-for-line transcriptions of the `__call__` bodies of HF's `TopKLogitsWarper`, `TopPLogitsWarper`,
# `MinPLogitsWarper` and `RepetitionPenaltyLogitsProcessor` (2-D path) at commit `8735a4a` (Apache-2.0, see Sources).
# `transformers` is not installed in this venv, so the code is copied rather than imported.

# %% tags=["solution_only"]
def hf_top_k(scores, top_k, min_tokens_to_keep=1):
    top_k = min(max(top_k, min_tokens_to_keep), scores.size(-1))
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    return scores.masked_fill(indices_to_remove, NEG_INF)


def hf_top_p(scores, top_p, min_tokens_to_keep=1):
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return scores.masked_fill(indices_to_remove, NEG_INF)


def hf_min_p(scores, min_p, min_tokens_to_keep=1):
    probs = torch.softmax(scores, dim=-1)
    tokens_to_remove = probs < min_p * probs.amax(dim=-1, keepdim=True)
    sorted_indices = torch.topk(probs, min(min_tokens_to_keep, probs.shape[-1]), dim=-1).indices
    tokens_to_remove.scatter_(-1, sorted_indices, False)
    return scores.masked_fill(tokens_to_remove, NEG_INF)


def hf_rep(input_ids, scores, penalty):
    score = torch.gather(scores, 1, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return scores.scatter(1, input_ids, score)


Zh = torch.randn(256, 50, dtype=torch.float64) * 2
prev_h = torch.randint(0, 50, (256, 12))
for k in (1, 5, 50, 80):
    check(f"== HF top-k k={k}", torch.equal(top_k_filter(Zh, k), hf_top_k(Zh, k)))
for p in (0.05, 0.5, 0.9, 0.99):
    for m in (1, 3):
        check(f"== HF top-p p={p} min_keep={m}", torch.equal(top_p_filter(Zh, p, m), hf_top_p(Zh, p, m)))
for mp in (0.0, 0.05, 0.3, 1.0):
    check(f"== HF min-p {mp}", torch.equal(min_p_filter(Zh, mp), hf_min_p(Zh, mp)))
check("== HF repetition penalty", torch.equal(repetition_penalty(Zh, prev_h, 1.3), hf_rep(prev_h, Zh, 1.3)))

# %% [markdown]
# ## Say it out loud
# 1. What do $T \to 0$ and $T \to \infty$ do to $\mathrm{softmax}(z/T)$? Does temperature change greedy decoding? What
#    happens to the entropy?
# 2. Why did Holtzman et al. argue for top-p over top-k? Give a context where a fixed k is too small and one where it
#    is too large.
# 3. State the nucleus precisely. Which token at the boundary is kept, and what goes wrong with `cumsum <= p`?
# 4. How does min-p differ from top-p, and why is it described as "relative"? Why does HF apply it *after* temperature?
# 5. Why does HF multiply negative logits by the repetition penalty instead of dividing? How does it differ from
#    OpenAI/vLLM presence and frequency penalties? Does HF penalize prompt tokens?
# 6. Give the order of the pipeline and an example where swapping two steps changes the output distribution.
# 7. Cost per decode step of each filter for vocabulary size V? How would you make top-p cheap for V = 150k?
# 8. How can you sample from softmax(z) without computing the softmax (Gumbel-max)? Why pass a `generator`?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. $T\to0$: all mass moves to the argmax (one-hot); $T\to\infty$: uniform. Dividing by $T>0$ preserves the ranking,
#    so greedy is unchanged. Entropy increases monotonically with $T$ (sharpening lowers it). Checked in §1.
# 2. A fixed k ignores the shape of the distribution (Holtzman et al. §3.2-3.3, Fig. 5): for a flat distribution
#    (open-ended story continuation, many plausible words) a small k cuts reasonable options; for a peaked one
#    ("The capital of France is") a large k lets junk tail tokens in. Top-p adapts the candidate-set size to the
#    entropy.
# 3. The smallest set of highest-probability tokens whose mass is $\ge p$ (Eq. 2). The token that crosses $p$ is
#    kept: keep rank $i$ iff the mass strictly above it is $< p$. `cumsum <= p` drops the crossing token, so the kept
#    mass can be $< p$, and if $p_{\max} > p$ the set is empty (all `-inf`, the softmax gives NaN). HF also enforces
#    `min_tokens_to_keep >= 1`.
# 4. Top-p fixes the *mass* to keep; min-p fixes a *probability floor* proportional to $p_\max$ (Nguyen et al.
#    Eq. 1-2), so the floor rises when the model is confident and falls when it is unsure. At high temperature the
#    tail fattens, and a relative floor keeps the tail out better than a nucleus does, which is the paper's case
#    for min-p at high T. HF applies it after temperature (code comment cites a llama.cpp discussion), so the
#    floor is computed on the distribution actually sampled.
# 5. Dividing a negative logit by $\theta>1$ moves it towards 0, which *raises* its probability, the opposite of a
#    penalty. CTRL's formula divides (it was framed as a per-token temperature), and HF's sign rule fixes the negative
#    case. HF's penalty is multiplicative, applied once per seen id, and for decoder-only models it includes the
#    prompt. vLLM/OpenAI `frequency_penalty` and `presence_penalty` are *additive*
#    (`z -= f*count + p*[count>0]`) over generated tokens only. vLLM's `repetition_penalty` uses prompt + output.
# 6. Repetition penalty → temperature → top-k → top-p → min-p (HF `_get_logits_processor`). Example from §6:
#    probs (.6,.2,.1,.06,.04), top_p=.75: top-p then T=2 keeps 2 tokens; T=2 then top-p keeps 3. Also note
#    `GenerationConfig` defaults `top_k=50`, so `do_sample=True` alone is not pure sampling in HF.
# 7. Temperature, min-p and the penalty are O(V). Top-k is O(V) with selection (`topk`). Top-p needs a sort,
#    O(V log V). In practice you apply top-k first (or a partial sort) and run top-p on the survivors, or use a
#    fused sampling kernel. All of this is tiny next to the forward pass, but it runs at every decode step for every sequence.
# 8. Gumbel-max: $\arg\max_i (z_i + g_i)$ with $g_i = -\log(-\log u_i)$, $u_i \sim U(0,1)$, is an exact sample from
#    $\mathrm{softmax}(z)$; filtered tokens at `-inf` never win. A per-request generator makes sampling reproducible
#    and independent of what else is in the batch. Statistical tests like the ones above need it.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
