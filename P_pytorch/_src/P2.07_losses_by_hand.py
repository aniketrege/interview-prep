# %% [markdown]
# # P2.7 · Losses by hand
#
# **ID:** P2.7 · **Target:** 15 min · **Plan week:** 1 (CE, BCE, MSE), 2 (KL, Huber, focal), 5 (full set) · **Part:** P
#
# **The drill.** Implement each loss from primitives in 2-5 lines and `assert allclose` against PyTorch.
# For each one, be ready to say **which likelihood it maximizes**, its **gradient w.r.t. the logits**, and
# **why the stable form is stable** (guide §P1).
#
# **Allowed:** `torch.logsumexp`, `log1p`, `exp`, `clamp`, `abs`, `where`, `gather`, `sigmoid`, reductions.
# **Not allowed inside your functions:** anything from `torch.nn.functional` / `torch.nn` losses,
# `log_softmax`, `softmax`. The tests use the built-ins as the reference.
#
# **Fill in** every `raise NotImplementedError`. Run the cell under each function; it prints `ok` or
# tells you what is off. Plan weeks 1-2 can stop after the sections listed above.
#
# ## Sources
# - PyTorch loss docs (reference semantics the tests check against):
#   `CrossEntropyLoss` https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html ·
#   `BCEWithLogitsLoss` https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html ·
#   `HuberLoss` https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html ·
#   `KLDivLoss` https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html ·
#   `GaussianNLLLoss` https://docs.pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
# - `torchvision.ops.sigmoid_focal_loss` (reference for the focal test; itself from fvcore):
#   https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py
# - T.-Y. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017, arXiv:1708.02002 — https://arxiv.org/abs/1708.02002
# - C. Szegedy et al., *Rethinking the Inception Architecture for Computer Vision* (label smoothing, §7), arXiv:1512.00567 — https://arxiv.org/abs/1512.00567
# - P. Huber, *Robust Estimation of a Location Parameter*, Ann. Math. Statist. 1964 — https://doi.org/10.1214/aoms/1177703732
# - Bishop, *Pattern Recognition and Machine Learning* (2006), §4.3.2 and §4.3.4: BCE/CE as Bernoulli/categorical
#   likelihoods, and the `y - t` gradient form of logistic and softmax regression.
# - R. Müller, S. Kornblith, G. Hinton, *When Does Label Smoothing Help?*, NeurIPS 2019, arXiv:1906.02629 — https://arxiv.org/abs/1906.02629

# %%
import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from prep import Session, check_close

s = Session("P2.7", target_min=15)
torch.manual_seed(0)
N, C = 16, 5
logits = torch.randn(N, C) * 3
y = torch.randint(0, C, (N,))
big = torch.tensor([[1e4, -1e4, 0.0], [-1e4, 1e4, 3.0]])  # stress test: naive formulas give inf/nan
y_big = torch.tensor([0, 2])

# %% [markdown]
# ## 1. Cross-entropy from logits (class-index targets) · week 1
# $\ell_n = -w_{y_n}\big(z_{n,y_n} - \log\sum_c e^{z_{n,c}}\big)$; mean = $\sum_n \ell_n / \sum_n w_{y_n}$ over
# non-ignored $n$. Stability: never form $e^{z}$ directly; use logsumexp (subtracts $\max_c z$).

# %%
def cross_entropy(z, y, weight=None, ignore_index=-100):
    """z: (N, C) logits, y: (N,) long. reduction='mean' semantics of F.cross_entropy."""
    ### BEGIN SOLUTION
    keep = y != ignore_index
    z, y = z[keep], y[keep]
    logp = z - torch.logsumexp(z, dim=1, keepdim=True)          # log-softmax
    nll = -logp.gather(1, y[:, None]).squeeze(1)                 # (N,)
    w = torch.ones_like(nll) if weight is None else weight[y]
    return (w * nll).sum() / w.sum()
    ### END SOLUTION


check_close("CE", cross_entropy(logits, y), F.cross_entropy(logits, y))
check_close("CE large logits", cross_entropy(big, y_big), F.cross_entropy(big, y_big))
w = torch.rand(C) + 0.5
check_close("CE weighted", cross_entropy(logits, y, weight=w), F.cross_entropy(logits, y, weight=w))
y_ign = y.clone(); y_ign[:4] = -100
check_close("CE ignore_index", cross_entropy(logits, y_ign), F.cross_entropy(logits, y_ign))

# %% [markdown]
# ## 2. Binary cross-entropy with logits (+ `pos_weight`) · week 1
# $\ell = -[\,p_w\,y\log\sigma(z) + (1-y)\log(1-\sigma(z))\,]$. Use $\log\sigma(z) = -\mathrm{softplus}(-z)$ and
# $\log(1-\sigma(z)) = -\mathrm{softplus}(z)$, with the stable $\mathrm{softplus}(x) = \max(x,0) + \log(1+e^{-|x|})$.

# %%
def softplus(x):
    ### BEGIN SOLUTION
    return x.clamp(min=0) + torch.log1p(torch.exp(-x.abs()))
    ### END SOLUTION


def bce_with_logits(z, t, pos_weight=None):
    """z, t: same shape, t float in [0, 1]. reduction='mean'."""
    ### BEGIN SOLUTION
    pw = 1.0 if pos_weight is None else pos_weight
    return (pw * t * softplus(-z) + (1 - t) * softplus(z)).mean()
    ### END SOLUTION


zb, tb = torch.randn(N, 3) * 4, torch.randint(0, 2, (N, 3)).float()
check_close("BCE", bce_with_logits(zb, tb), F.binary_cross_entropy_with_logits(zb, tb))
check_close("BCE large logits", bce_with_logits(big, (big > 0).float()), F.binary_cross_entropy_with_logits(big, (big > 0).float()))
pw = torch.tensor([1.0, 5.0, 0.3])
check_close("BCE pos_weight", bce_with_logits(zb, tb, pw), F.binary_cross_entropy_with_logits(zb, tb, pos_weight=pw))

# %% [markdown]
# ## 3. Regression: MSE, L1, Huber · week 1 (MSE), week 2 (Huber)
# Huber($\delta$): $\tfrac12 r^2$ if $|r|<\delta$ else $\delta(|r| - \tfrac12\delta)$.

# %%
def mse(pred, t):
    ### BEGIN SOLUTION
    return ((pred - t) ** 2).mean()
    ### END SOLUTION


def l1(pred, t):
    ### BEGIN SOLUTION
    return (pred - t).abs().mean()
    ### END SOLUTION


def huber(pred, t, delta=1.0):
    ### BEGIN SOLUTION
    r = (pred - t).abs()
    return torch.where(r < delta, 0.5 * r**2, delta * (r - 0.5 * delta)).mean()
    ### END SOLUTION


p_, t_ = torch.randn(N) * 3, torch.randn(N)
check_close("MSE", mse(p_, t_), F.mse_loss(p_, t_))
check_close("L1", l1(p_, t_), F.l1_loss(p_, t_))
for d in (0.5, 1.0, 2.0):
    check_close(f"Huber delta={d}", huber(p_, t_, d), F.huber_loss(p_, t_, delta=d))

# %% [markdown]
# ## 4. KL divergence, `reduction="batchmean"` · week 2
# `F.kl_div(input, target)` takes **input = log q** (model log-probs) and **target = p** (probs) and returns
# $\frac1N\sum_n\sum_c p_{n,c}(\log p_{n,c} - \log q_{n,c}) = \mathrm{KL}(p\,\|\,q)$. Mind the argument order.

# %%
def kl_batchmean(log_q, p):
    ### BEGIN SOLUTION
    return (p * (p.clamp_min(1e-30).log() - log_q)).sum() / p.shape[0]
    ### END SOLUTION


log_q = logits - torch.logsumexp(logits, 1, keepdim=True)
p_t = torch.softmax(torch.randn(N, C), 1)
check_close("KL batchmean", kl_batchmean(log_q, p_t), F.kl_div(log_q, p_t, reduction="batchmean"))

# %% [markdown]
# ## 5. Label smoothing · week 2
# Target $q = (1-\varepsilon)\,\text{onehot}(y) + \varepsilon/C$, loss $= -\sum_c q_c \log p_c$
# $= (1-\varepsilon)\,\text{NLL} + \varepsilon\cdot\text{mean}_c(-\log p_c)$.

# %%
def ce_label_smoothing(z, y, eps=0.1):
    ### BEGIN SOLUTION
    logp = z - torch.logsumexp(z, dim=1, keepdim=True)
    nll = -logp.gather(1, y[:, None]).squeeze(1)
    uniform = -logp.mean(dim=1)
    return ((1 - eps) * nll + eps * uniform).mean()
    ### END SOLUTION


for e in (0.0, 0.1, 0.3):
    check_close(f"CE label_smoothing={e}", ce_label_smoothing(logits, y, e), F.cross_entropy(logits, y, label_smoothing=e))

# %% [markdown]
# ## 6. Focal loss (sigmoid / binary form) · week 2
# $\mathrm{FL} = -\alpha_t(1-p_t)^\gamma\log p_t$ with $p_t = p$ if $t=1$ else $1-p$, and $\alpha_t$ likewise.
# Reuse your stable BCE for $-\log p_t$.

# %%
def focal_loss(z, t, alpha=0.25, gamma=2.0):
    """Elementwise mean, matching torchvision sigmoid_focal_loss(reduction='mean')."""
    ### BEGIN SOLUTION
    p = torch.sigmoid(z)
    ce = t * softplus(-z) + (1 - t) * softplus(z)         # -log p_t, stable
    p_t = p * t + (1 - p) * (1 - t)
    a_t = alpha * t + (1 - alpha) * (1 - t)
    return (a_t * (1 - p_t) ** gamma * ce).mean()
    ### END SOLUTION


for a, g in ((0.25, 2.0), (0.5, 0.0), (0.75, 1.0)):
    check_close(f"focal a={a} g={g}", focal_loss(zb, tb, a, g), sigmoid_focal_loss(zb, tb, a, g, reduction="mean"))

# %% [markdown]
# ## 7. Gradients w.r.t. logits (say these in the interview) · week 1
# For mean CE: $\partial L/\partial z = (\mathrm{softmax}(z) - \mathrm{onehot}(y))/N$. For mean BCE:
# $(\sigma(z) - t)/\text{numel}$. Write them in closed form (you may use `torch.softmax` here) and compare to autograd.

# %%
def ce_grad(z, y):
    ### BEGIN SOLUTION
    g = torch.softmax(z, 1)
    g[torch.arange(len(y)), y] -= 1
    return g / len(y)
    ### END SOLUTION


def bce_grad(z, t):
    ### BEGIN SOLUTION
    return (torch.sigmoid(z) - t) / z.numel()
    ### END SOLUTION


z_ = logits.clone().requires_grad_()
F.cross_entropy(z_, y).backward()
check_close("dCE/dz", ce_grad(logits, y), z_.grad)
zb_ = zb.clone().requires_grad_()
F.binary_cross_entropy_with_logits(zb_, tb).backward()
check_close("dBCE/dz", bce_grad(zb, tb), zb_.grad)

# %% [markdown]
# ## 8. Gaussian NLL (regression with predicted variance) · week 5
# `F.gaussian_nll_loss(mu, t, var, eps=1e-6)` = mean of $\tfrac12(\log v + (\mu - t)^2 / v)$ with $v=\max(\text{var},\varepsilon)$
# (constant $\tfrac12\log 2\pi$ dropped unless `full=True`). In a model, produce `var` with `softplus` to keep it positive.

# %%
def gaussian_nll(mu, t, var, eps=1e-6):
    ### BEGIN SOLUTION
    v = var.clamp(min=eps)
    return (0.5 * (v.log() + (mu - t) ** 2 / v)).mean()
    ### END SOLUTION


var_ = F.softplus(torch.randn(N))
check_close("Gaussian NLL", gaussian_nll(p_, t_, var_), F.gaussian_nll_loss(p_, t_, var_))

# %% [markdown]
# ## Say it out loud
# 1. MSE is the NLL of which distribution? L1? CE? BCE? What does Huber interpolate between?
# 2. Why is `log(softmax(z))` unstable but `z - logsumexp(z)` fine? What exactly does logsumexp subtract?
# 3. With `weight=` in `CrossEntropyLoss`, what is the mean divided by? Why does it matter for imbalanced batches?
# 4. `KLDivLoss(input, target)`: which argument is log-probs? Which KL direction is it, and when would you want the other?
# 5. What do focal loss's $\gamma$ and $\alpha$ each do? What is focal loss at $\gamma=0$?
# 6. Label smoothing: what does it do to the optimal logits, and why can it hurt calibration-sensitive uses (e.g. distillation)?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. MSE: Gaussian with fixed variance (minimizer = conditional mean). L1: Laplace (minimizer = conditional median,
#    robust to outliers). CE: categorical. BCE: Bernoulli. Huber is quadratic near 0 (MSE-like, smooth gradient) and
#    linear in the tails (L1-like, bounded gradient $\pm\delta$).
# 2. `softmax` exponentiates: $e^{1000}$ overflows to inf, and tiny probabilities underflow to 0 so `log` gives -inf.
#    `logsumexp(z) = m + log Σ exp(z - m)` with $m = \max_c z_c$: every exponent is ≤ 0 and the largest term is 1, so the
#    sum is in $[1, C]$. The log-softmax is then a difference of finite numbers.
# 3. By $\sum_n w_{y_n}$ (sum of the weights of the targets present), not by N. So the loss is a weighted *average* whose
#    scale doesn't change with how many rare-class examples land in a batch. `ignore_index` items are excluded from both sums.
# 4. `input` is log q (the model), `target` is p (the data/teacher). It computes KL(p‖q), the *forward* KL. That is
#    mass-covering: q is penalized wherever p > 0 and q is small. Reverse KL(q‖p) is mode-seeking. RL fine-tuning penalties
#    (KL to a reference policy, estimated on samples from q) and variational inference use the reverse direction.
# 5. $(1-p_t)^\gamma$ down-weights easy, well-classified examples ($p_t \to 1$), so training focuses on hard ones.
#    $\alpha$ is a static class-balance weight. At $\gamma=0$ it is $\alpha$-weighted BCE. Lin et al. found $\alpha$=0.25 works
#    with $\gamma$=2: the down-weighting already favors the rare positives, so $\alpha$ ends up *favoring negatives*. This is counterintuitive.
# 6. Without smoothing the optimal logit gap is infinite; with smoothing it is finite (the target puts $\varepsilon/C$ on every
#    class). Penultimate-layer clusters get tighter, and the logits stop carrying "dark knowledge" about class similarity.
#    Müller et al. 2019 showed smoothed teachers distill worse (arXiv:1906.02629).
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
