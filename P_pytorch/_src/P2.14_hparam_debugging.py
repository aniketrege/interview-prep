# %% [markdown]
# # P2.14 · Hyperparameter debugging
#
# **ID:** P2.14 · **Target:** 30 min · **Plan week:** 4 · **Part:** P
#
# **The drill.** Guide §P2 "Must include": LR sweep, batch-size effects, over/underfit diagnosis from curves. You will
# (1) implement Smith's **LR range test** by hand, (2) run a real **LR sweep** and read it the way the Tuning Playbook
# says (is the best point on the boundary?), (3) measure **batch-size effects**: gradient noise ∝ 1/B and the
# **linear scaling rule** with warmup, and (4) **diagnose seven pairs of train/val curves**. The answers there are checked
# against hashes, so reading the check does not give them away.
#
# Workload: a 784-256-10 MLP on a 10k-image MNIST subset kept on the GPU, SGD with momentum 0.9 (LR effects are
# clearest with SGD). Everything runs in about a minute.
#
# ## Sources
# - V. Godbole, G. E. Dahl, J. Gilmer, C. J. Shallue, Z. Nado, *Deep Learning Tuning Playbook*, Google Research, 2023
#   (CC-BY-4.0), commit 7554a51 — https://github.com/google-research/tuning_playbook . Main source for this notebook:
#   "Choosing the batch size", "Changing the batch size requires re-tuning most hyperparameters", "Identifying bad
#   search space boundaries", "Examining the training curves", "How can optimization failures be debugged and mitigated?".
# - L. N. Smith, *Cyclical Learning Rates for Training Neural Networks*, WACV 2017, arXiv:1506.01186 —
#   https://arxiv.org/abs/1506.01186 (§3.3, the "LR range test": increase the LR during a short run and watch the loss).
# - L. N. Smith, *A disciplined approach to neural network hyper-parameters: Part 1*, 2018, arXiv:1803.09820 —
#   https://arxiv.org/abs/1803.09820
# - P. Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, 2017, arXiv:1706.02677 —
#   https://arxiv.org/abs/1706.02677 (linear scaling rule: multiply the LR by k when the batch is multiplied by k; gradual warmup).
# - C. J. Shallue et al., *Measuring the Effects of Data Parallelism on Neural Network Training*, JMLR 2019,
#   arXiv:1811.03600 — https://arxiv.org/abs/1811.03600 (perfect scaling up to a critical batch size, then diminishing returns).
# - S. McCandlish, J. Kaplan, D. Amodei, OpenAI Dota Team, *An Empirical Model of Large-Batch Training*, 2018,
#   arXiv:1812.06162 — https://arxiv.org/abs/1812.06162 (gradient noise scale).
# - A. Karpathy, *A Recipe for Training Neural Networks* (2019): https://karpathy.github.io/2019/04/25/recipe/

# %%
import hashlib
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from prep import DATA, Session, check, seed_everything

s = Session("P2.14", target_min=30)
seed_everything(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnist = torchvision.datasets.MNIST(DATA, train=True, download=False)
X = mnist.data[:12000].reshape(-1, 784).float() / 255.0
mu, sd = X[:10000].mean(), X[:10000].std()
X = ((X - mu) / sd).to(dev)
Y = mnist.targets[:12000].to(dev)
Xtr, Ytr, Xva, Yva = X[:10000], Y[:10000], X[10000:], Y[10000:]


def make_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(dev)


@torch.no_grad()
def val_loss(model):
    return F.cross_entropy(model(Xva), Yva).item()

# %% [markdown]
# ## 1. LR range test (by hand)
# One short run from a fresh model, with the LR increased **geometrically** from `lr_min` to `lr_max` over `steps`
# mini-batches: $\eta_i = \eta_{\min}(\eta_{\max}/\eta_{\min})^{i/(\text{steps}-1)}$. (Smith 2015 increased it linearly
# over a few epochs; the exponential version used by most libraries covers several decades in one short run.) Record a
# **debiased** EMA of the loss at each step *before* the update, and stop once it exceeds `diverge` × its minimum. A common
# pick is about 1/10 of the LR at the minimum of the smoothed loss (where it is still clearly decreasing).

# %%
def lr_range_test(make_model, X, Y, lr_min=1e-5, lr_max=10.0, steps=200, batch_size=128, beta=0.9, diverge=4.0, seed=0):
    """Returns (lrs, smoothed_losses), one entry per step taken."""
    ### BEGIN SOLUTION
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=lr_min, momentum=0.9)
    g = torch.Generator(device=X.device).manual_seed(seed)
    q = (lr_max / lr_min) ** (1 / (steps - 1))
    lrs, losses, avg, best = [], [], 0.0, float("inf")
    for i in range(steps):
        lr = lr_min * q**i
        for group in opt.param_groups:
            group["lr"] = lr
        idx = torch.randint(0, len(X), (batch_size,), device=X.device, generator=g)
        loss = F.cross_entropy(model(X[idx]), Y[idx])
        avg = beta * avg + (1 - beta) * loss.item()
        smooth = avg / (1 - beta ** (i + 1))
        lrs.append(lr); losses.append(smooth)
        best = min(best, smooth)
        if smooth > diverge * best or not math.isfinite(smooth):
            break
        opt.zero_grad(); loss.backward(); opt.step()
    return lrs, losses
    ### END SOLUTION


def suggest_lr(lrs, losses):
    ### BEGIN SOLUTION
    return lrs[int(np.argmin(losses))] / 10
    ### END SOLUTION


lrs_rt, losses_rt = lr_range_test(make_model, Xtr, Ytr)
lr_suggest = suggest_lr(lrs_rt, losses_rt)
ratios = np.array(lrs_rt[1:]) / np.array(lrs_rt[:-1])
check("starts at lr_min", abs(lrs_rt[0] - 1e-5) < 1e-12)
check("geometric schedule hitting lr_max at the last step", np.allclose(ratios, (1e6) ** (1 / 199), rtol=1e-6))
check("smoothed loss is debiased (first value ~ ln 10, not ~0.23)", abs(losses_rt[0] - math.log(10)) < 0.3, f"{losses_rt[0]:.3f}")
check("loss actually decreased (the LR really changed)", min(losses_rt) < 1.0, f"min {min(losses_rt):.3f}")
check("stopped early on divergence", len(lrs_rt) < 200, f"ran {len(lrs_rt)} steps")
check("suggested LR in a sane range for this setup", 1e-3 <= lr_suggest <= 1.0, f"{lr_suggest:.3g}")
check("suggested LR is well below the loss-minimum LR (still stable there)", lr_suggest <= 0.5 * lrs_rt[int(np.argmin(losses_rt))])
plt.figure(figsize=(5, 3.5))
plt.semilogx(lrs_rt, losses_rt)
plt.axvline(lr_suggest, color="r", ls="--", label=f"suggested {lr_suggest:.2g}")
plt.ylim(0, 4 * min(losses_rt) + 1)
plt.xlabel("learning rate"); plt.ylabel("smoothed train loss"); plt.title("LR range test"); plt.legend(); plt.show()

# %% [markdown]
# ## 2. LR sweep, read like the Playbook
# Train a fresh model for a **fixed number of steps** at each LR on a log grid; report the final val loss (`inf` if it
# hits NaN/inf). The Playbook's first question about a sweep: *is the best point on the edge of the search space?* If it is,
# the optimum may lie outside, so widen the range before you conclude anything.

# %%
def train_fixed_steps(lr, steps=300, batch_size=128, seed=0, warmup=0):
    """Fresh model, SGD(momentum=0.9), optional linear warmup over `warmup` steps. Returns final val loss (inf if diverged)."""
    ### BEGIN SOLUTION
    model = make_model(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    g = torch.Generator(device=dev).manual_seed(seed)
    for i in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (i + 1) / warmup) if warmup else lr
        idx = torch.randint(0, len(Xtr), (batch_size,), device=dev, generator=g)
        loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
        if not torch.isfinite(loss):
            return float("inf")
        opt.zero_grad(); loss.backward(); opt.step()
    v = val_loss(model)
    return v if math.isfinite(v) else float("inf")
    ### END SOLUTION


def best_on_boundary(grid, scores):
    """True if the best (lowest) score sits at the first or last grid point."""
    ### BEGIN SOLUTION
    i = int(np.argmin(scores))
    return i == 0 or i == len(grid) - 1
    ### END SOLUTION


narrow = [1e-4, 3e-4, 1e-3]
wide = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
scores_wide = [train_fixed_steps(lr) for lr in wide]
scores_narrow = scores_wide[:3]
best_lr = wide[int(np.argmin(scores_wide))]
print({lr: round(v, 3) for lr, v in zip(wide, scores_wide)})
check("narrow grid: best is on the boundary (expand it!)", best_on_boundary(narrow, scores_narrow))
check("wide grid: best is interior", not best_on_boundary(wide, scores_wide), f"best lr {best_lr}")
check("largest LR blows up (val loss worse than chance)", scores_wide[-1] > math.log(10), f"{scores_wide[-1]:.3g}")
check("range-test suggestion within ~1 decade of the sweep optimum", 0.1 <= lr_suggest / best_lr <= 10,
      f"suggest {lr_suggest:.3g}, sweep best {best_lr:.3g}")
plt.figure(figsize=(5, 3.5))
ok = [(lr, v) for lr, v in zip(wide, scores_wide) if v < math.log(10)]
bad = [lr for lr, v in zip(wide, scores_wide) if v >= math.log(10)]
plt.semilogx(*zip(*ok), "o-")
plt.semilogx(bad, [math.log(10)] * len(bad), "rx", label="diverged / worse than chance")
plt.axvspan(narrow[0], narrow[-1], alpha=0.15, label="narrow grid")
plt.xlabel("learning rate"); plt.ylabel("val loss after 300 steps"); plt.title("LR sweep"); plt.legend(); plt.show()

# %% [markdown]
# ## 3. Batch size: gradient noise and the linear scaling rule
# A mini-batch gradient $g_B$ is an unbiased estimate of the full gradient $g$, with $\mathbb E\|g_B - g\|^2 \approx \operatorname{tr}(\Sigma)/B$
# (exactly $\frac{\operatorname{tr}\Sigma}{B}\frac{N-B}{N-1}$ when sampling without replacement). **Measure** it at a fixed
# model and fit the log-log slope. Then the consequence: with a fixed *epoch* budget, a k× larger batch takes k× fewer
# steps. Goyal et al.'s **linear scaling rule** multiplies the LR by k (with warmup) to compensate, which works up to a critical
# batch size (Shallue et al.). The Playbook adds that every batch-size change needs the LR, and usually the regularization, re-tuned.

# %%
def grad_noise(model, X, Y, B, draws=20, seed=0):
    """Mean over `draws` random mini-batches (without replacement) of ||g_B - g_full||^2, flattened over all params."""
    ### BEGIN SOLUTION
    params = list(model.parameters())
    flat = lambda gs: torch.cat([g.reshape(-1) for g in gs])
    g_full = flat(torch.autograd.grad(F.cross_entropy(model(X), Y), params))
    gen = torch.Generator(device=X.device).manual_seed(seed)
    out = []
    for _ in range(draws):
        idx = torch.randperm(len(X), device=X.device, generator=gen)[:B]
        g_b = flat(torch.autograd.grad(F.cross_entropy(model(X[idx]), Y[idx]), params))
        out.append((g_b - g_full).pow(2).sum().item())
    return float(np.mean(out))
    ### END SOLUTION


probe = make_model(1)
Bs = [16, 32, 64, 128, 256]
noise = [grad_noise(probe, Xtr, Ytr, B, draws=40) for B in Bs]
slope = np.polyfit(np.log(Bs), np.log(noise), 1)[0]
print("E||g_B - g||^2:", [f"{v:.3g}" for v in noise], f"log-log slope {slope:.2f}")
check("gradient noise scales like 1/B (slope ~ -1)", -1.25 < slope < -0.75, f"slope {slope:.2f}")

base_lr, base_B, epochs = 0.01, 32, 3
rows = []
for B in (32, 128, 512):
    steps = epochs * len(Xtr) // B
    same = train_fixed_steps(base_lr, steps=steps, batch_size=B)
    scaled = train_fixed_steps(base_lr * B / base_B, steps=steps, batch_size=B, warmup=max(1, steps // 10))  # warm up over the first 10% of steps
    rows.append((B, steps, same, scaled))
    print(f"B={B:4d} steps={steps:4d}  val loss: same LR {same:.3f} | linearly scaled LR {base_lr * B / base_B:.2f} + warmup {scaled:.3f}")
check("fixed epoch budget: bigger batch with the same LR is worse", rows[-1][2] > rows[0][2] + 0.05)
check("linear scaling + warmup recovers most of it at B=512", rows[-1][3] < rows[-1][2] - 0.05)

# %% [markdown]
# ## 4. Diagnose the curves
# Seven training runs (train and val loss per epoch, 10-class problem, so ln 10 ≈ 2.30 at init). Label each panel
# with exactly one of:
#
# | label | what it looks like |
# | --- | --- |
# | `healthy` | both decrease and flatten, small stable gap |
# | `underfit` | both plateau high and close together: capacity/optimization too weak |
# | `overfit` | train keeps falling, val turns **up** (problematic overfitting, Playbook) |
# | `lr_too_high` | noisy, spiky, plateaus high or blows up |
# | `lr_too_low` | smooth, slow, nearly linear decrease, still improving at the end |
# | `leakage` | val far **better** than train, implausibly good |
# | `bug` | no learning at all: loss pinned at its init value |
#
# Fill in `my_labels`. Don't read the generator. Look at the plots.

# %%
def make_curves(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(50)
    def shape(start, floor, tau, noise, up=0.0, t_up=0, spikes=0.0):
        y = floor + (start - floor) * np.exp(-t / tau) + up * np.clip(t - t_up, 0, None)
        y = y + noise * rng.standard_normal(50) * np.sqrt(y)
        if spikes:
            y = y + spikes * (rng.random(50) < 0.15) * rng.random(50)
        return np.clip(y, 1e-3, None)
    specs = [  # (train spec, val spec)
        (dict(start=2.3, floor=0.02, tau=6, noise=0.01), dict(start=2.3, floor=0.42, tau=6, noise=0.01, up=0.015, t_up=12)),
        (dict(start=2.3, floor=2.3026, tau=5, noise=0.001), dict(start=2.3, floor=2.3026, tau=5, noise=0.001)),
        (dict(start=2.3, floor=0.25, tau=8, noise=0.01), dict(start=2.3, floor=0.31, tau=8, noise=0.01)),
        (dict(start=2.3, floor=0.4, tau=150, noise=0.003), dict(start=2.3, floor=0.45, tau=150, noise=0.003)),
        (dict(start=2.3, floor=1.15, tau=2, noise=0.12, spikes=1.5), dict(start=2.3, floor=1.25, tau=2, noise=0.12, spikes=1.5)),
        (dict(start=2.3, floor=0.45, tau=8, noise=0.01), dict(start=2.3, floor=0.03, tau=3, noise=0.01)),
        (dict(start=2.3, floor=1.42, tau=4, noise=0.01), dict(start=2.3, floor=1.47, tau=4, noise=0.01)),
    ]
    order = rng.permutation(len(specs))
    return {chr(65 + i): (shape(**specs[j][0]), shape(**specs[j][1])) for i, j in enumerate(order)}


curves = make_curves()
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, (name, (tr, va)) in zip(axes.flat, curves.items()):
    ax.plot(tr, label="train"); ax.plot(va, label="val")
    ax.set(title=f"run {name}", xlabel="epoch", ylabel="loss", ylim=(0, 3.5))
axes.flat[0].legend(); axes.flat[-1].axis("off")
plt.tight_layout(); plt.show()

# %%
### BEGIN SOLUTION
# TEMPLATE: my_labels = {"A": None, "B": None, "C": None, "D": None, "E": None, "F": None, "G": None}
my_labels = {"A": "healthy", "B": "lr_too_high", "C": "lr_too_low", "D": "underfit", "E": "leakage", "F": "overfit",
             "G": "bug"}
### END SOLUTION

# %%
ANSWER_HASHES = {  # sha256 of "<panel>:<label>"
    "A": "8bd0148b920542d8acaf3cff37bc9b7af5123db5b3c68c0321575f466a9c97a5",
    "B": "e7ee74a6c5598e3f3acf49a89ccbd47e9871b636b166bef3604ef0298ce79cfa",
    "C": "19d742ad7060544767115e45504dd0e389c22e64c76128d409a7386af879a606",
    "D": "b0e50131b2a8752169825e570b0fe0230b8d8206c889a928118baab5d4bca79d",
    "E": "970cc17dcb1b1aaf0bac52eb0a5c3d54d2ea05692756d2f5b68979ce8b930da1",
    "F": "a6a6ec84f76084d827765fe37c2bce08ab4da8e8ab4598621db3fdc142c7e223",
    "G": "608d24282f2bd4cdf4c9c0db482b4012b6e21b3dff335b60c006e2426359b9e7",
}
wrong = [k for k in curves if hashlib.sha256(f"{k}:{my_labels.get(k)}".encode()).hexdigest() != ANSWER_HASHES[k]]
for k in curves:
    print(f"run {k}: {my_labels.get(k)!s:12s} {'ok' if k not in wrong else 'WRONG'}")
check("all seven curve pairs diagnosed correctly", not wrong, f"wrong: {wrong}")

# %% [markdown]
# ## Say it out loud
# 1. Walk through an LR range test: what do you plot, where do you read the LR off, and why not take the LR at the loss minimum?
# 2. Your sweep's best LR is the largest value you tried. What do you conclude, and what do you run next?
# 3. Why does gradient noise scale like 1/B, and what does that imply for the LR when you scale the batch? Where does the linear rule break?
# 4. Why warm up the LR for large batches?
# 5. Should you tune the batch size for validation accuracy? (Playbook answer.)
# 6. For each of the seven curve types, what would you change next?
# 7. How do you tell "the task is genuinely hard" apart from "there is a bug" or "there is leakage"?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Plot the smoothed training loss vs the LR on a log axis over one short run with exponentially increasing LR.
#    The loss is flat while the LR is too small, falls, then blows up. The minimum sits at the edge of instability,
#    and the LR there was only applied for a few steps from an already-improved model, so it overestimates the LR that is stable
#    for a whole run. Take ~1/10 of the minimum (or the steepest-descent point) as the max LR, then confirm with a short sweep (Smith 2015, 2018).
# 2. The search space boundary is too narrow: the optimum may lie beyond it (Playbook, "Identifying bad search space
#    boundaries"). Extend the grid in that direction until the best point is interior and you see the divergence side.
# 3. $g_B$ is a mean of B i.i.d. per-example gradients, so its covariance is Σ/B (measured slope ≈ −1 above). SGD's update
#    noise is roughly $\eta^2\Sigma/B$ per step. Keeping η/B fixed keeps the noise per epoch similar, which gives the linear rule
#    (Goyal et al. 2017). It breaks past a critical batch size, where curvature, not noise, limits the step size (Shallue et al. 2018; McCandlish et al. 2018
#    estimate it via the gradient noise scale). It also breaks for adaptive optimizers (Adam is closer to √k scaling in some regimes), and
#    early in training, when the loss changes fast.
# 4. At init the curvature is high and the large scaled LR can diverge before the network settles. Ramping the LR from small
#    to the target over the first few epochs avoids that early instability (Goyal et al. used 5 epochs of gradual warmup; the Playbook lists
#    warmup as the first fix for early-training instability).
# 5. No. Batch size mainly sets time and compute. With LR/regularization re-tuned and enough steps, any batch size reaches
#    the same final performance. Pick the largest one that keeps speeding up training, then re-tune the optimizer hyperparameters (Playbook,
#    citing Shallue et al.).
# 6. healthy: stop early or scale up. underfit: bigger model, train longer, higher LR, less regularization. overfit: more data/augmentation,
#    weight decay, dropout, early stopping. LR too high: lower the LR, warmup, clipping. LR too low: raise the LR (range test) or train longer.
#    leakage: audit the splits (duplicates, preprocessing fit on all data, label in features). bug: overfit one batch, check that
#    `zero_grad`/`step` are in the right order, that the parameters are in the optimizer, that the labels are aligned, and the loss at init.
# 7. A bug or leakage shows up in cheap sanity checks: can the model overfit a tiny batch (if not: bug)? Does a shuffled-label
#    run go to chance on val (if not: leakage)? Does a trivial baseline (majority class, linear model) do about as well? Is the val loss below
#    the irreducible noise? A genuinely hard task passes those checks, and its train loss falls with capacity while val stays stuck.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
