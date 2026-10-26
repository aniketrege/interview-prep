# %% [markdown]
# # A3.18 · Speculative sampling: the acceptance rule, exactness, and expected speedup
#
# **ID:** A3.18 · **Target:** 20 min · **Plan week:** 5 · **Part:** A
#
# **The drill.** Implement one step of speculative sampling (Leviathan et al. 2023, Algorithm 1; Chen et al. 2023,
# Algorithm 2) and the generation loop around it, then *prove it empirically*: the output distribution equals the
# target model's, and the mean number of tokens per target call matches $(1-\alpha^{\gamma+1})/(1-\alpha)$.
# Guide §C4: "Speculative decoding (draft + verify; implement the acceptance rule)".
#
# **One step, given a prefix:**
# 1. Draft $\gamma$ tokens autoregressively from the small model: $x_i \sim q_i = q(\cdot \mid \text{prefix}, x_{<i})$.
# 2. Run the target once on all $\gamma+1$ prefixes (in a real LLM this is one forward pass over $\gamma$ new
#    positions): $p_1, \dots, p_{\gamma+1}$.
# 3. For $i = 1..\gamma$: accept $x_i$ with probability $\min(1, p_i(x_i)/q_i(x_i))$. At the first rejection, sample
#    a replacement from the **residual** $\mathrm{norm}(\max(0, p_i - q_i))$ and stop.
# 4. If all $\gamma$ were accepted, sample a **bonus** token from $p_{\gamma+1}$.
#
# Every step emits between 1 and $\gamma+1$ tokens for one target call.
#
# **The toy models** are bigram tables over $V=6$ tokens (row $V$ is the start state), so the sequence process is real
# (each draft token conditions the next) and exact sequence probabilities are a product of table entries. No EOS.
#
# **Deeper study:** the KV-cache drill A3.13 (a real system must roll the draft and target caches back to the last
# accepted token) and Raschka's KV-cache bonus `../../LLMs-from-scratch/ch04/03_kv-cache/`. The papers' proofs
# (Leviathan et al. Appendix A.1; Chen et al. Theorem 1) are a short read and a common interview follow-up.
#
# ## Sources
# - Y. Leviathan, M. Kalman, Y. Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023,
#   arXiv:2211.17192 — https://arxiv.org/abs/2211.17192 (Algorithm 1; Corollary 3.6 $\alpha = 1 - D_{LK}(p,q) =
#   \sum_x \min(p(x), q(x))$; Eq. 1 expected tokens per step $(1-\alpha^{\gamma+1})/(1-\alpha)$ under the i.i.d.
#   assumption; Theorem 3.8 walltime factor $(1-\alpha^{\gamma+1})/((1-\alpha)(\gamma c+1))$; numbering as in arXiv v2).
# - C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, J. Jumper, *Accelerating Large Language Model Decoding
#   with Speculative Sampling*, 2023, arXiv:2302.01318 — https://arxiv.org/abs/2302.01318 (Algorithm 2, modified
#   rejection sampling; Theorem 1, the output distribution equals the target's).
# - T. Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, 2024,
#   arXiv:2401.10774 — https://arxiv.org/abs/2401.10774; Y. Li, F. Wei, C. Zhang, H. Zhang, *EAGLE: Speculative Sampling
#   Requires Rethinking Feature Uncertainty*, 2024, arXiv:2401.15077 — https://arxiv.org/abs/2401.15077 (draft heads /
#   tree drafts; mentioned in the answers only).

# %%
import math

import torch

from prep import Session, check, check_close

s = Session("A3.18", target_min=20)
torch.manual_seed(0)
V, BOS = 6, 6
_g = torch.Generator().manual_seed(0)
P_TAB = (torch.randn(V + 1, V, generator=_g, dtype=torch.float64) * 1.5).softmax(-1)   # target  p(next | last)
R_TAB = (torch.randn(V + 1, V, generator=_g, dtype=torch.float64) * 1.5).softmax(-1)
Q_TAB = 0.3 * P_TAB + 0.7 * R_TAB                                                       # a mediocre draft


def target(last):
    """Target model: next-token distribution (V,) given the last token (BOS for an empty prefix)."""
    return P_TAB[last]


def draft(last):
    return Q_TAB[last]


def tv(a, b):
    return 0.5 * (a - b).abs().sum().item()


def sample(probs, gen):
    return int(torch.multinomial(probs, 1, generator=gen))

# %% [markdown]
# ## 1. Acceptance rate and the residual distribution
# Accepting $x \sim q$ with probability $\min(1, p(x)/q(x))$ succeeds overall with probability
# $\alpha = \sum_x q(x)\min(1, p(x)/q(x)) = \sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p, q)$.
# After a rejection, sample from $r = \max(0, p-q)/\sum\max(0,p-q)$; the normalizer equals $1-\alpha$, the
# rejection probability. That is why accepted mass $\min(p,q)$ + rejected mass $(1-\alpha)\,r$ adds back up to exactly $p$.

# %%
def acceptance_rate(p, q):
    """alpha = P(a draft token x ~ q is accepted) for target p. p, q: (V,) probabilities."""
    ### BEGIN SOLUTION
    return torch.minimum(p, q).sum().item()
    ### END SOLUTION


def residual_dist(p, q):
    """norm(max(0, p - q)). Only called after a rejection, which implies p != q."""
    ### BEGIN SOLUTION
    r = (p - q).clamp(min=0)
    return r / r.sum()
    ### END SOLUTION


p_h = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
q_h = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
check_close("alpha hand case = .2 + .3 + .2", acceptance_rate(p_h, q_h), 0.7)
check_close("residual hand case", residual_dist(p_h, q_h), torch.tensor([1.0, 0.0, 0.0]))
check_close("alpha = 1 - TV(p, q)", acceptance_rate(P_TAB[BOS], Q_TAB[BOS]), 1 - tv(P_TAB[BOS], Q_TAB[BOS]))
check_close("alpha(p, p) = 1", acceptance_rate(P_TAB[2], P_TAB[2]), 1.0)
for c in range(V + 1):   # the identity that makes the algorithm exact, for every context of the toy model
    p, q = P_TAB[c], Q_TAB[c]
    recon = q * torch.clamp(p / q, max=1.0) + (1 - acceptance_rate(p, q)) * residual_dist(p, q)
    assert torch.allclose(recon, p), f"context {c}: accepted + rejected mass != p"
check("q*min(1,p/q) + (1-alpha)*residual == p for every context", True)

# %% [markdown]
# ## 2. One speculative step and the generation loop
# `speculative_step(last, target, draft, gamma, gen)` returns `(new_tokens, n_accepted)`. Use `sample(...)` and
# `torch.rand((), generator=gen, dtype=torch.float64)` for all randomness. Accept iff `u < p(x)/q(x)` (equivalent to
# probability $\min(1, p/q)$). `speculative_generate` repeats steps until it has `n_tokens` tokens, truncates the last
# step's overshoot, and returns `(tokens, n_steps)`; `n_steps` is the number of target calls.

# %%
def speculative_step(last, target, draft, gamma, gen):
    """last: last token of the prefix (BOS if empty). -> (list of 1..gamma+1 new tokens, number of accepted drafts)."""
    ### BEGIN SOLUTION
    xs, qs, prev = [], [], last
    for _ in range(gamma):                                  # 1. draft autoregressively
        q = draft(prev)
        prev = sample(q, gen)
        xs.append(prev)
        qs.append(q)
    ps = [target(t) for t in [last] + xs]                   # 2. one "parallel" target pass: gamma + 1 distributions
    out = []
    for i, x in enumerate(xs):                              # 3. accept / reject left to right
        if torch.rand((), generator=gen, dtype=torch.float64) < ps[i][x] / qs[i][x]:
            out.append(x)
        else:
            out.append(sample(residual_dist(ps[i], qs[i]), gen))
            return out, i
    out.append(sample(ps[gamma], gen))                      # 4. bonus token from p_{gamma+1}
    return out, gamma
    ### END SOLUTION


def speculative_generate(target, draft, n_tokens, gamma, gen):
    """-> (tokens: list of exactly n_tokens ints, n_steps: number of speculative steps = target calls)."""
    ### BEGIN SOLUTION
    seq, steps = [], 0
    while len(seq) < n_tokens:
        new, _ = speculative_step(seq[-1] if seq else BOS, target, draft, gamma, gen)
        seq += new
        steps += 1
    return seq[:n_tokens], steps
    ### END SOLUTION


gen = torch.Generator().manual_seed(1)
out, n_acc = speculative_step(BOS, target, target, 4, gen)
check("draft == target: everything accepted, gamma + 1 tokens", n_acc == 4 and len(out) == 5)
for _ in range(200):
    out, n_acc = speculative_step(BOS, target, draft, 3, gen)
    assert 1 <= len(out) <= 4 and len(out) == n_acc + 1, f"step returned {out} with n_accepted={n_acc}"
check("every step emits n_accepted + 1 tokens (1..gamma+1)", True)
toks, steps = speculative_generate(target, draft, 10, 3, gen)
check("generate returns exactly n_tokens", len(toks) == 10 and all(0 <= t < V for t in toks))

# %% [markdown]
# ## 3. Exactness: the output distribution is the target's
# Generate $N = 30{,}000$ sequences of 4 tokens with $\gamma = 3$ (so steps start at different positions) and compare
# the empirical joint distributions of $(x_1, x_2)$ and $(x_3, x_4)$ (36 cells each) with the exact ones,
# $p(x_1\mid\text{BOS})\,p(x_2\mid x_1)$ and $P(x_3)\,p(x_4\mid x_3)$. Monte-Carlo TV error is below about 0.01; the
# threshold is 0.03. A typical bug (resampling from $p$ instead of the residual) is off by about 0.1.

# %%
N, T_LEN, GAMMA = 30_000, 4, 3
gen = torch.Generator().manual_seed(2026)
seqs = torch.tensor([speculative_generate(target, draft, T_LEN, GAMMA, gen)[0] for _ in range(N)])

joint_emp = torch.bincount(seqs[:, 0] * V + seqs[:, 1], minlength=V * V).double() / N
joint_true = (P_TAB[BOS][:, None] * P_TAB[:V]).flatten()
marg3 = P_TAB[BOS] @ torch.linalg.matrix_power(P_TAB[:V], 2)              # law of x3
joint34_true = (marg3[:, None] * P_TAB[:V]).flatten()
joint34_emp = torch.bincount(seqs[:, 2] * V + seqs[:, 3], minlength=V * V).double() / N
print(f"TV(joint x1,x2) = {tv(joint_emp, joint_true):.4f}   TV(joint x3,x4) = {tv(joint34_emp, joint34_true):.4f}")
check("joint of (x1, x2) matches the target", tv(joint_emp, joint_true) < 0.03)
check("joint of (x3, x4) matches the target (x4 is often the bonus token)", tv(joint34_emp, joint34_true) < 0.03)

# a draft with no support where the target has mass: only the residual can produce those tokens
Q_BAD = Q_TAB.clone()
Q_BAD[:, :2] = 0.0
Q_BAD /= Q_BAD.sum(-1, keepdim=True)
gen = torch.Generator().manual_seed(7)
seqs_bad = torch.tensor([speculative_generate(target, lambda t: Q_BAD[t], 2, 2, gen)[0] for _ in range(N)])
jb = torch.bincount(seqs_bad[:, 0] * V + seqs_bad[:, 1], minlength=V * V).double() / N
check("still exact when the draft has zero mass on tokens 0, 1", tv(jb, joint_true) < 0.03, f"TV={tv(jb, joint_true):.4f}")

# %% [markdown]
# ## 4. How many tokens per target call?
# If each draft token is accepted independently with probability $\alpha$ (the i.i.d. assumption of Leviathan
# et al. §3.1), the number of accepted drafts is a truncated geometric and the tokens per step are
# $\mathbb{E}[n_\text{acc}+1] = \sum_{k=0}^{\gamma}\alpha^k = (1-\alpha^{\gamma+1})/(1-\alpha)$. To test that formula
# use a *context-free* toy (the same $p$, $q$ at every position), so $\alpha$ really is constant. With a draft cost
# ratio $c$ (draft step time / target step time) the walltime factor is $(1-\alpha^{\gamma+1})/((1-\alpha)(\gamma c+1))$
# (Theorem 3.8).

# %%
def expected_tokens_per_step(alpha, gamma):
    ### BEGIN SOLUTION
    return gamma + 1.0 if alpha == 1 else (1 - alpha ** (gamma + 1)) / (1 - alpha)
    ### END SOLUTION


def walltime_speedup(alpha, gamma, c):
    """Expected speedup over plain decoding when one draft step costs c target steps."""
    ### BEGIN SOLUTION
    return expected_tokens_per_step(alpha, gamma) / (gamma * c + 1)
    ### END SOLUTION


check_close("E[tokens/step] alpha=.5, gamma=3 = 1 + .5 + .25 + .125", expected_tokens_per_step(0.5, 3), 1.875)
check_close("E[tokens/step] alpha=1 is gamma+1", expected_tokens_per_step(1.0, 4), 5.0)
check_close("E[tokens/step] alpha=0 is 1", expected_tokens_per_step(0.0, 4), 1.0)
check_close("walltime speedup alpha=.8, gamma=4, c=.05", walltime_speedup(0.8, 4, 0.05), (1 - 0.8 ** 5) / (0.2 * 1.2))

p_iid, q_iid = P_TAB[3], Q_TAB[3]
alpha = acceptance_rate(p_iid, q_iid)
gen = torch.Generator().manual_seed(11)
for gamma in (1, 3, 6):
    res = [speculative_step(0, lambda t: p_iid, lambda t: q_iid, gamma, gen) for _ in range(20_000)]
    tok = torch.tensor([len(o) for o, _ in res], dtype=torch.float64)
    p_rej_first = sum(n == 0 for _, n in res) / len(res)
    want = expected_tokens_per_step(alpha, gamma)
    print(f"alpha={alpha:.3f} gamma={gamma}: simulated {tok.mean():.3f} tokens/step, formula {want:.3f}")
    check(f"tokens/step gamma={gamma} matches (1-a^(g+1))/(1-a)", abs(tok.mean().item() - want) < 0.04)
    check(f"first draft rejected with prob 1 - alpha (gamma={gamma})", abs(p_rej_first - (1 - alpha)) < 0.02)

# on the context-dependent toy the formula is only an approximation (alpha varies with the context)
gen = torch.Generator().manual_seed(12)
outs = [speculative_generate(target, draft, 400, 3, gen) for _ in range(20)]
print(f"bigram toy: {sum(len(t) for t, _ in outs) / sum(n for _, n in outs):.2f} tokens per target call (gamma=3); "
      f"alpha ranges {min(acceptance_rate(P_TAB[c], Q_TAB[c]) for c in range(V)):.2f}"
      f"-{max(acceptance_rate(P_TAB[c], Q_TAB[c]) for c in range(V)):.2f} across contexts")
print("best gamma for alpha=.8, c=.05:", max(range(1, 16), key=lambda g: walltime_speedup(0.8, g, 0.05)))

# %% [markdown]
# ## Say it out loud
# 1. Prove in three lines that one accept/reject round outputs a token distributed exactly as $p$.
# 2. What is $\alpha$ in terms of $p$ and $q$? What draft maximizes it?
# 3. Why is the bonus token free, and what goes wrong if you drop it? Which of the tests above would catch that?
# 4. Derive the expected tokens per step. Which assumption does it rely on, and does it hold for real LMs?
# 5. When does speculative decoding *not* speed things up? Think about the cost ratio $c$, batch size, and compute- vs
#    memory-bound decoding.
# 6. What has to happen to the KV caches of both models after a rejection at position $i$?
# 7. How does the rule change for greedy decoding? Name one way to get a draft without a separate model.

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. $P(\text{out}=x) = q(x)\min(1, p(x)/q(x)) + (1-\alpha)\,r(x) = \min(p(x),q(x)) + \max(0, p(x)-q(x)) = p(x)$,
#    using $\sum_x \max(0,p-q) = 1 - \sum_x\min(p,q) = 1-\alpha$, which is the normalizer of $r$ (Chen et al. Thm 1;
#    Leviathan et al. App. A.1). Later positions are exact by induction, because each accepted or replaced token
#    is an exact sample given the prefix, and the next round conditions on it. §1 and §3 check this numerically.
# 2. $\alpha = \sum_x \min(p(x),q(x)) = 1 - \mathrm{TV}(p,q)$ (Leviathan et al. Cor. 3.6). It is maximized by
#    $q=p$ ($\alpha=1$). In practice that means a small model distilled from or aligned with the target, sharing
#    its tokenizer.
# 3. The target already computed $p_{\gamma+1}$ in the same parallel pass, so sampling from it costs nothing extra.
#    Without it, a fully accepted step yields $\gamma$ instead of $\gamma+1$ tokens. Exactness is unaffected (the
#    next step simply starts there), so only the tokens-per-step test (§4) catches it. The mean drops by
#    $\alpha^\gamma$.
# 4. $n_\text{acc}=k$ with prob. $\alpha^k(1-\alpha)$ for $k<\gamma$ and $\alpha^\gamma$ for $k=\gamma$, so
#    $\mathbb{E}[n_\text{acc}+1]=\sum_{k=0}^{\gamma}\alpha^k=(1-\alpha^{\gamma+1})/(1-\alpha)$. It assumes the
#    acceptances are i.i.d. with a fixed $\alpha$. Real acceptance varies by context (boilerplate and code are
#    easy, content words are hard), so treat it as an estimate. The bigram toy above shows the spread.
# 5. When $\alpha$ is low or $c$ is high (Theorem 3.8 needs $\alpha > c$ for some $\gamma$ to help, Cor. 3.9). Also
#    when decoding is already compute-bound, e.g. large batches. Speculation trades extra FLOPs (rejected tokens,
#    $\gamma+1$ positions per target pass) for fewer memory-bound serial steps, which helps only while the target
#    pass over $\gamma+1$ positions costs about the same as over 1. Long $\gamma$ wastes draft time once
#    $\alpha^\gamma$ is small.
# 6. Truncate both caches to the prefix plus accepted tokens (drop entries for rejected drafts). The replacement
#    or bonus token has not been through either model yet, so it is fed in on the next step. With a paged or
#    ring cache, truncation is only a length pointer update.
# 7. Greedy: accept $x_i$ iff it equals $\arg\max p_i$, and on mismatch emit $\arg\max p_i$ (the limit of the
#    rule as both distributions become one-hot). Drafts without a separate model: extra decoding heads on the
#    target (Medusa, arXiv:2401.10774), a light feature-level draft head (EAGLE, arXiv:2401.15077), or n-gram /
#    prompt-lookup matching. Tree drafts verify several candidate continuations in one pass.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
