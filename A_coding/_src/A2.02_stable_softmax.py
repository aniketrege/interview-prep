# %% [markdown]
# # A2.2 · Numerically stable softmax, log-softmax, logsumexp, cross-entropy
#
# **ID:** A2.2 · **Target:** 15 min · **Plan week:** 1 · **Part:** A
#
# **The drill.** Write softmax / log-softmax / logsumexp in NumPy and PyTorch. First run the naive versions on
# concrete inputs and watch them overflow and underflow. Then write CE from logits, the softmax Jacobian
# $\operatorname{diag}(p) - pp^\top$ (checked against `torch.autograd.functional.jacobian`), its cheap
# vector-Jacobian product, temperature, and a **streaming (online) logsumexp** over chunks. The streaming version
# is the scalar core of the online softmax in FlashAttention (G1.04).
#
# Guide A2 "must include": numerically stable softmax, log-softmax, logsumexp, cross-entropy.
#
# **Allowed:** `exp`, `log`, `log1p`, `max`, `sum`, `where`, `isfinite`, indexing. **Not allowed** inside your functions:
# `torch.softmax`, `log_softmax`, `torch.logsumexp`, `scipy.special.*`, `F.cross_entropy` (they are the references).
#
# ## Sources
# - P. Blanchard, D. J. Higham, N. J. Higham, *Accurately computing the log-sum-exp and softmax functions*,
#   IMA J. Numer. Anal. 41(4):2311-2330, 2021, DOI 10.1093/imanum/draa038 — https://doi.org/10.1093/imanum/draa038
#   (shifted formulas, and which of the two softmax forms is more accurate).
# - M. Milakov, N. Gimelshein, *Online normalizer calculation for softmax*, 2018, arXiv:1805.02867 —
#   https://arxiv.org/abs/1805.02867 (the one-pass running-max / running-sum recurrence).
# - T. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022,
#   arXiv:2205.14135 — https://arxiv.org/abs/2205.14135 (online softmax applied to tiles of attention).
# - IEEE-754 limits used below are printed with `np.finfo` in the notebook itself.
# - PyTorch docs: `torch.logsumexp` https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html ·
#   `torch.autograd.functional.jacobian` https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html

# %%
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp as sp_logsumexp, softmax as sp_softmax

from prep import Session, check, check_close, seed_everything

s = Session("A2.2", target_min=15)
seed_everything(0)
np.seterr(all="ignore")  # we *want* to see inf/nan from the naive versions, without warnings
for dt in (np.float32, np.float64):
    fi = np.finfo(dt)
    print(f"{dt.__name__}: max {fi.max:.3e}, exp overflows above ln(max) = {np.log(fi.max):.2f}; "
          f"smallest normal {fi.tiny:.3e} (exp underflows to 0 below about {np.log(fi.tiny * fi.eps):.1f})")

# %% [markdown]
# ## 1. Naive versions fail, with concrete numbers
# $\mathrm{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}$. In float32, `exp` overflows past $z \approx 88.7$ and
# underflows to 0 below about $-103$ (subnormals included), so:

# %%
def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


z_big = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
z_small = np.array([-1000.0, -1001.0, -1002.0], dtype=np.float32)
z_spread = np.array([0.0, -200.0], dtype=np.float32)
print("overflow :", softmax_naive(z_big))                    # inf/inf = nan
print("underflow:", softmax_naive(z_small))                  # 0/0 = nan
print("log(softmax) of a tiny prob:", np.log(softmax_naive(z_spread)))  # [0, -inf]; the true value is -200
print("naive logsumexp(z_big):", np.log(np.exp(z_big).sum()), " (true: 1002.4076)")

# %% [markdown]
# ## 2. Stable NumPy versions
# Shift by $m = \max_j z_j$: $\mathrm{softmax}(z) = \mathrm{softmax}(z - m)$, and
# $\mathrm{lse}(z) = m + \log\sum_j e^{z_j - m}$. Every exponent is $\le 0$ and the largest term is exactly 1, so the
# sum lies in $[1, n]$: no overflow, and no $\log 0$. Then $\log\mathrm{softmax}(z) = z - \mathrm{lse}(z)$, which is
# **not** the same as `log(softmax(z))`: the latter rounds tiny probabilities to 0 first.
# Work along `axis` with `keepdims` so batched inputs broadcast correctly.

# %%
def logsumexp_np(z, axis=-1):
    ### BEGIN SOLUTION
    m = np.max(z, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))).squeeze(axis)
    ### END SOLUTION


def softmax_np(z, axis=-1):
    ### BEGIN SOLUTION
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)
    ### END SOLUTION


def log_softmax_np(z, axis=-1):
    ### BEGIN SOLUTION
    return z - np.expand_dims(logsumexp_np(z, axis), axis)
    ### END SOLUTION


rng = np.random.default_rng(0)
Z = (rng.standard_normal((4, 7, 5)) * 30).astype(np.float64)
for ax in (-1, 0, 1):
    check_close(f"logsumexp axis={ax}", logsumexp_np(Z, ax), sp_logsumexp(Z, axis=ax))
    check_close(f"softmax axis={ax}", softmax_np(Z, ax), sp_softmax(Z, axis=ax))
    check_close(f"log_softmax axis={ax}", log_softmax_np(Z, ax), Z - sp_logsumexp(Z, axis=ax, keepdims=True))
z_rows = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]])  # each row needs its own max
check_close("softmax rows at different scales", softmax_np(z_rows), sp_softmax(z_rows, axis=-1))
check_close("logsumexp rows at different scales", logsumexp_np(z_rows), sp_logsumexp(z_rows, axis=-1))
for name, z in [("big", z_big), ("small", z_small), ("spread", z_spread)]:
    check_close(f"softmax {name}", softmax_np(z), sp_softmax(z.astype(np.float64)), atol=1e-6)
    check_close(f"log_softmax {name}", log_softmax_np(z), z.astype(np.float64) - sp_logsumexp(z.astype(np.float64)), atol=1e-4)

# %% [markdown]
# ## 3. PyTorch versions, and a fully masked row
# Same formulas with `dim`. Extra edge case from attention with padding: a row that is **entirely** `-inf`.
# Then $m = -\infty$ and $z - m$ is $(-\infty) - (-\infty) = \text{nan}$. `torch.logsumexp` returns $-\infty$ for that row.
# Match it by replacing a non-finite max with 0 before shifting. Softmax of an all-`-inf` row has no right answer;
# we only require finite rows to be right.

# %%
def logsumexp(z, dim=-1):
    ### BEGIN SOLUTION
    m = z.amax(dim=dim, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    return (m + (z - m).exp().sum(dim=dim, keepdim=True).log()).squeeze(dim)
    ### END SOLUTION


def softmax(z, dim=-1):
    ### BEGIN SOLUTION
    e = (z - z.amax(dim=dim, keepdim=True)).exp()
    return e / e.sum(dim=dim, keepdim=True)
    ### END SOLUTION


def log_softmax(z, dim=-1):
    ### BEGIN SOLUTION
    return z - logsumexp(z, dim).unsqueeze(dim)
    ### END SOLUTION


torch.manual_seed(0)
zt = torch.randn(6, 9) * 40
zt[0, :3] = 1e4
for d in (0, 1):
    check_close(f"torch logsumexp dim={d}", logsumexp(zt, d), torch.logsumexp(zt, d))
    check_close(f"torch softmax dim={d}", softmax(zt, d), torch.softmax(zt, d))
    check_close(f"torch log_softmax dim={d}", log_softmax(zt, d), torch.log_softmax(zt, d))
zm = torch.randn(3, 4)
zm[1] = float("-inf")       # fully masked row
zm[2, :2] = float("-inf")   # partially masked row
check_close("logsumexp with masked rows", logsumexp(zm, 1), torch.logsumexp(zm, 1))
check_close("softmax partially masked row", softmax(zm[[0, 2]], 1), torch.softmax(zm[[0, 2]], 1))

# %% [markdown]
# ## 4. Cross-entropy from logits
# $\mathrm{CE} = \frac1N\sum_n \big(\mathrm{lse}(z_n) - z_{n, y_n}\big)$. One line on top of your `logsumexp`.

# %%
def cross_entropy(z, y):
    ### BEGIN SOLUTION
    return (logsumexp(z, 1) - z.gather(1, y[:, None]).squeeze(1)).mean()
    ### END SOLUTION


y = torch.randint(0, 9, (6,))
check_close("CE", cross_entropy(zt, y), F.cross_entropy(zt, y))
zb = torch.tensor([[1e4, -1e4, 0.0], [-1e4, 1e4, 3.0]])
check_close("CE extreme logits", cross_entropy(zb, torch.tensor([1, 2])), F.cross_entropy(zb, torch.tensor([1, 2])))

# %% [markdown]
# ## 5. The softmax Jacobian and its vector-Jacobian product
# $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$, i.e. $J = \operatorname{diag}(p) - pp^\top$ (symmetric, rows sum to 0,
# rank $\le n-1$: adding a constant to all logits changes nothing). Backprop never forms $J$:
# $J^\top g = p \odot (g - \langle g, p\rangle)$, which is $O(n)$ instead of $O(n^2)$.
# `softmax_jacobian(p)` takes a batch `(B, n)` and returns `(B, n, n)`; `softmax_vjp(p, g)` returns `(B, n)`.

# %%
def softmax_jacobian(p):
    ### BEGIN SOLUTION
    return torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    ### END SOLUTION


def softmax_vjp(p, g):
    ### BEGIN SOLUTION
    return p * (g - (g * p).sum(-1, keepdim=True))
    ### END SOLUTION


zj = torch.randn(3, 5, dtype=torch.float64)
pj = torch.softmax(zj, -1)
J_ref = torch.stack([torch.autograd.functional.jacobian(lambda v: torch.softmax(v, -1), zj[b]) for b in range(3)])
check_close("softmax Jacobian", softmax_jacobian(pj), J_ref, atol=1e-10)
gj = torch.randn(3, 5, dtype=torch.float64)
zj_ = zj.clone().requires_grad_()
torch.softmax(zj_, -1).backward(gj)
check_close("softmax VJP vs autograd", softmax_vjp(pj, gj), zj_.grad, atol=1e-10)
check("Jacobian rows sum to 0", bool(softmax_jacobian(pj).sum(-1).abs().max() < 1e-12))

# %% [markdown]
# ## 6. Temperature
# $\mathrm{softmax}(z/T)$: $T\to 0$ approaches one-hot on the argmax, $T\to\infty$ approaches uniform, and the entropy
# increases with $T$ (for non-constant $z$). Implement `softmax_t` with your stable `softmax`. It must work for $T = 10^{-3}$.

# %%
def softmax_t(z, T):
    ### BEGIN SOLUTION
    return softmax(z / T, -1)
    ### END SOLUTION


zT = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
check_close("T=1", softmax_t(zT, 1.0), torch.softmax(zT, -1))
check_close("T -> 0 is one-hot", softmax_t(zT, 1e-3), F.one_hot(zT.argmax(-1), 4).float(), atol=1e-6)
check_close("T -> inf is uniform", softmax_t(zT, 1e6), torch.full_like(zT, 0.25), atol=1e-5)
Ts = [0.1, 0.5, 1.0, 2.0, 10.0]
H = [-(softmax_t(zT, t) * log_softmax(zT / t)).sum().item() for t in Ts]
check("entropy increases with T", all(a < b for a, b in zip(H, H[1:])), f"{H}")
print("entropy by T:", {t: round(h, 3) for t, h in zip(Ts, H)})

# %% [markdown]
# ## 7. Streaming (online) logsumexp
# You get the logits in chunks (e.g. a vocab-parallel shard at a time, or tiles of attention scores) and may keep only
# two scalars per row: the running max $m$ and the running sum $s = \sum e^{z - m}$. On a new chunk $c$:
# $m' = \max(m, \max c)$, $s' = s\,e^{m - m'} + \sum e^{c - m'}$. At the end, $\mathrm{lse} = m + \log s$.
# Start from $m = -\infty$, $s = 0$ and take care that the very first rescale $e^{-\infty - m'}$ is 0, not nan.
# FlashAttention also carries a running output $o$ rescaled by the same factor (G1.04). Stretch goal: `online_softmax_matvec`.

# %%
def streaming_logsumexp(chunks):
    """chunks: iterable of (B, n_k) tensors (same B). Returns (B,) logsumexp of their concatenation, in one pass."""
    ### BEGIN SOLUTION
    m = s_ = None
    for c in chunks:
        cm = c.amax(-1)
        if m is None:
            m, s_ = cm, (c - cm[:, None]).exp().sum(-1)
            continue
        m_new = torch.maximum(m, cm)
        s_ = s_ * (m - m_new).exp() + (c - m_new[:, None]).exp().sum(-1)
        m = m_new
    return m + s_.log()
    ### END SOLUTION


def online_softmax_matvec(chunks, values):
    """Stretch. chunks: list of (B, n_k) logits; values: matching list of (n_k, D). Returns softmax(z) @ V, (B, D), in one pass."""
    ### BEGIN SOLUTION
    m = s_ = o = None
    for c, v in zip(chunks, values):
        cm = c.amax(-1)
        m_new = cm if m is None else torch.maximum(m, cm)
        e = (c - m_new[:, None]).exp()
        if m is None:
            s_, o = e.sum(-1), e @ v
        else:
            a = (m - m_new).exp()
            s_, o = s_ * a + e.sum(-1), o * a[:, None] + e @ v
        m = m_new
    return o / s_[:, None]
    ### END SOLUTION


torch.manual_seed(1)
zs = torch.randn(4, 1000, dtype=torch.float64) * 20
zs[0] += 1e4         # would overflow without shifting
zs[1] -= 1e4         # would underflow without shifting
zs[2, 700:] += 50    # the max arrives late: earlier chunks must be rescaled
chunks = list(zs.split(128, dim=1))
check_close("streaming logsumexp", streaming_logsumexp(chunks), torch.logsumexp(zs, 1), atol=1e-8)
check_close("streaming logsumexp, 1 chunk", streaming_logsumexp([zs]), torch.logsumexp(zs, 1), atol=1e-8)
V = torch.randn(1000, 3, dtype=torch.float64)
check_close("online softmax @ V (stretch)", online_softmax_matvec(chunks, list(V.split(128))), torch.softmax(zs, 1) @ V, atol=1e-8)

# %% [markdown]
# ## Say it out loud
# 1. Why is subtracting the max *exact* for softmax? What does it buy for overflow, underflow, and the log?
# 2. Why compute log-softmax as `z - logsumexp(z)` rather than `log(softmax(z))`? Give a concrete float32 input where they differ.
# 3. What is the softmax Jacobian? Why is it singular, and how does backprop avoid forming it?
# 4. What is the gradient of CE w.r.t. the logits, and why does fusing softmax+CE (as `F.cross_entropy` does) help?
# 5. Explain the online logsumexp recurrence. Why is it exact and not an approximation? Where does FlashAttention use it?
# 6. What does temperature do to the distribution and its entropy? What happens numerically as $T \to 0$?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. $e^{z_i - m}/\sum_j e^{z_j - m}$: the factor $e^{-m}$ cancels between numerator and denominator, so the value is unchanged. After the shift
#    every exponent is $\le 0$ (no overflow), and the max term is exactly $e^0 = 1$, so the denominator is $\ge 1$ (no 0/0, and its log is finite).
#    Entries far below the max can still underflow to 0 in the softmax itself, which is fine for probabilities but not for their logs.
# 2. `log(softmax(z))` first rounds tiny probabilities to 0 and then gives `-inf` (and NaN gradients). `z - lse(z)` never forms them. Example:
#    float32 `z = [0, -200]`: the true log-prob of the second entry is about -200, but $e^{-200}$ underflows to 0 in float32 and the log gives -inf
#    (section 1). Blanchard, Higham and Higham (2021) analyze the shifted formulas and their rounding error.
# 3. $J = \operatorname{diag}(p) - pp^\top$. $J\mathbf{1} = p - p(\mathbf{1}^\top p) = 0$, so $\mathbf{1}$ is in the null space: shifting all logits
#    by a constant does not change the output. Backprop needs only $J^\top g = p\odot(g - \langle g,p\rangle)$, which costs $O(n)$ per row and
#    never forms the $n\times n$ matrix.
# 4. $\partial\,\mathrm{CE}/\partial z = \mathrm{softmax}(z) - \mathrm{onehot}(y)$ (divided by N for the mean). Fusing avoids storing the softmax for the
#    backward of a separate log, avoids the `log(0)` problem, and gives this simple, bounded gradient, with entries in $[-1, 1]$.
# 5. Keep $(m, s)$ with $s = \sum e^{z - m}$ over what has been seen. When a larger max arrives, rescale the old sum by $e^{m - m'}$. This is an exact
#    algebraic identity, not an approximation, so the result equals the two-pass value up to rounding (Milakov & Gimelshein 2018).
#    FlashAttention processes K/V in tiles. Per query row it keeps the running max, the running denominator and the unnormalized output
#    $o$, rescaling $o$ by the same factor, so the $T\times T$ score matrix is never written to HBM (Dao et al. 2022).
# 6. Dividing by $T<1$ sharpens the distribution (lower entropy) and $T>1$ flattens it. $T\to 0$ gives the argmax (greedy decoding) and $T\to\infty$
#    gives uniform. Numerically, $z/T$ gets huge as $T\to 0$, so the max-shift is essential: with $T = 10^{-3}$ the logits above are $\sim 2000$,
#    which overflows `exp` without the shift.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
