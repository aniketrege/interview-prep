# %% [markdown]
# # A2.5 · Manual backprop: Linear, ReLU, softmax+CE, LayerNorm, BatchNorm, a 2-layer MLP
#
# **ID:** A2.5 · **Target:** 40 min · **Plan week:** 1-3 · **Part:** A
#
# **The drill.** Write the backward pass of each layer by hand, on tensors, with no autograd inside your functions.
# Each one is checked against PyTorch autograd in float64. Then chain them into a full 2-layer MLP backward, and
# write a central finite-difference gradient checker that you can use anywhere.
# Guide A2 "must include": "Manual backprop for linear, ReLU, softmax+CE, LayerNorm (gradient check with finite differences)"; A3: "MLP with manual backward".
#
# **Rules.** Inside your functions: no `.backward()`, no `torch.autograd`, no `torch.func`. Use `@`, elementwise ops and
# reductions. For each gradient, say its **shape** before you write it.
#
# **Shape rules (say them out loud).**
# - $\partial L/\partial X$ always has the shape of $X$.
# - $Y = XW^\top + b$ with $X\,(N,D_{in})$, $W\,(D_{out},D_{in})$: $dX = dY\,W$ $(N,D_{in})$, $dW = dY^\top X$ $(D_{out},D_{in})$,
#   $db = \sum_n dY_n$ $(D_{out})$. To get these, write the only matmul whose shapes fit.
# - Broadcasting in the forward becomes a **sum** over the broadcast dims in the backward (bias, LayerNorm's $\gamma,\beta$ over $(B,T)$,
#   BatchNorm's $\gamma,\beta$ over $N$). A reduction in the forward (mean, sum) becomes a broadcast in the backward.
# - Elementwise $y=f(x)$: $dx = dy \odot f'(x)$.
#
# ## Sources
# - A. Karpathy, *llm.c* LayerNorm tutorial: https://github.com/karpathy/llm.c, commit `f1e2ace`, `doc/layernorm/layernorm.md`
#   (derivation) and `doc/layernorm/layernorm.py` (forward with `(mean, rstd)` cache and the 3-term backward; adapted
#   below), MIT license.
# - A. Karpathy, *Neural Networks: Zero to Hero*, Lecture 5 "Building makemore Part 4: Becoming a Backprop Ninja":
#   https://youtu.be/q8SA3rM6ckI, notebook `lectures/makemore/makemore_part4_backprop.ipynb` in
#   https://github.com/karpathy/nn-zero-to-hero, commit `73c3fcc`, MIT license (exercise 2, the fused CE gradient, and
#   exercise 3, the one-line BatchNorm backward; that notebook uses Bessel's $n-1$ in training-mode variance, and here we
#   use PyTorch's biased variance, so the formula has no $n/(n-1)$ factor).
# - S. Ioffe, C. Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*,
#   ICML 2015, arXiv:1502.03167 — https://arxiv.org/abs/1502.03167 (the BN backward equations in §3).
# - J. L. Ba, J. R. Kiros, G. E. Hinton, *Layer Normalization*, 2016, arXiv:1607.06450 — https://arxiv.org/abs/1607.06450
# - PyTorch `F.batch_norm` / `nn.BatchNorm1d` docs (biased variance for normalization in training mode):
#   https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

# %%
import torch
import torch.nn.functional as F

from prep import Session, check, check_close, seed_everything

s = Session("A2.5", target_min=40)
seed_everything(0)
dd = dict(dtype=torch.float64)


def autograd_grads(fn, inputs, dout):
    """Reference: grads of sum(fn(*inputs) * dout) w.r.t. each input, via autograd."""
    leaves = [t.detach().clone().requires_grad_() for t in inputs]
    out = fn(*leaves)
    return torch.autograd.grad((out * dout).sum(), leaves)

# %% [markdown]
# ## 1. Linear and ReLU
# Linear convention as in `nn.Linear`: `W (out, in)`, `y = x @ W.T + b`. ReLU: `dx = dy * (x > 0)`.

# %%
def linear_backward(dy, x, W):
    """dy (N, out), x (N, in), W (out, in) -> dx (N, in), dW (out, in), db (out,)."""
    ### BEGIN SOLUTION
    return dy @ W, dy.T @ x, dy.sum(0)
    ### END SOLUTION


def relu_backward(dy, x):
    ### BEGIN SOLUTION
    return dy * (x > 0).to(dy.dtype)
    ### END SOLUTION


x = torch.randn(6, 4, **dd)
W, b = torch.randn(3, 4, **dd), torch.randn(3, **dd)
dy = torch.randn(6, 3, **dd)
ref = autograd_grads(F.linear, (x, W, b), dy)
for name, got, exp in zip(("dx", "dW", "db"), linear_backward(dy, x, W), ref):
    check_close(f"linear {name}", got, exp, atol=1e-12)
xr, dr = torch.randn(50, 8, **dd), torch.randn(50, 8, **dd)
check_close("relu dx", relu_backward(dr, xr), autograd_grads(torch.relu, (xr,), dr)[0], atol=1e-12)

# %% [markdown]
# ## 2. Fused softmax + cross-entropy
# For the mean CE over $N$ rows: $\partial L/\partial z = (\mathrm{softmax}(z) - \mathrm{onehot}(y))/N$, shape $(N, C)$.
# (You may use `torch.softmax` / `torch.logsumexp` for the forward value here; A2.2 is where you write those.)

# %%
def softmax_ce(z, y):
    """Returns (mean CE loss, dL/dz)."""
    ### BEGIN SOLUTION
    N = z.shape[0]
    loss = (torch.logsumexp(z, 1) - z[torch.arange(N), y]).mean()
    dz = torch.softmax(z, 1)
    dz[torch.arange(N), y] -= 1
    return loss, dz / N
    ### END SOLUTION


z = torch.randn(8, 5, **dd) * 3
yz = torch.randint(0, 5, (8,))
zl = z.clone().requires_grad_()
loss_ref = F.cross_entropy(zl, yz)
loss_ref.backward()
loss_mine, dz_mine = softmax_ce(z, yz)
check_close("softmax_ce loss", loss_mine, loss_ref, atol=1e-12)
check_close("softmax_ce dz", dz_mine, zl.grad, atol=1e-12)

# %% [markdown]
# ## 3. LayerNorm (from llm.c)
# Over the last dim $C$, for each row: $\mu = \bar x$, $\sigma^{-1} = (\overline{(x-\mu)^2} + \epsilon)^{-1/2}$ (`rstd`),
# $\hat x = (x - \mu)\,\sigma^{-1}$, $y = \gamma\odot\hat x + \beta$. Cache `(x, w, mean, rstd)` only.
# Backward: $db = \sum_{B,T} dy$, $dw = \sum_{B,T} dy\odot\hat x$, and with $g = dy\odot\gamma$:
# $$dx = \sigma^{-1}\Big(g - \overline{g} - \hat x\,\overline{g\odot\hat x}\Big)$$
# (means over $C$). The three terms are: the direct path, the path through $\mu$, and the path through $\sigma$.

# %%
def layernorm_forward(x, w, b, eps=1e-5):
    """x (B, T, C), w/b (C,) -> out (B, T, C), cache. Adapted from llm.c doc/layernorm/layernorm.py (MIT)."""
    ### BEGIN SOLUTION
    mean = x.mean(-1, keepdim=True)
    xshift = x - mean
    var = (xshift ** 2).mean(-1, keepdim=True)
    rstd = (var + eps) ** -0.5
    out = xshift * rstd * w + b
    return out, (x, w, mean, rstd)
    ### END SOLUTION


def layernorm_backward(dout, cache):
    """-> dx (B, T, C), dw (C,), db (C,)."""
    ### BEGIN SOLUTION
    x, w, mean, rstd = cache
    norm = (x - mean) * rstd
    db = dout.sum((0, 1))
    dw = (dout * norm).sum((0, 1))
    g = dout * w
    dx = (g - g.mean(-1, keepdim=True) - norm * (g * norm).mean(-1, keepdim=True)) * rstd
    return dx, dw, db
    ### END SOLUTION


xl = torch.randn(2, 3, 8, **dd) * 2 + 1
wl, bl = torch.randn(8, **dd), torch.randn(8, **dd)
dl = torch.randn(2, 3, 8, **dd)
out_l, cache_l = layernorm_forward(xl, wl, bl)
check_close("layernorm forward", out_l, F.layer_norm(xl, (8,), wl, bl, eps=1e-5), atol=1e-10)
ref = autograd_grads(lambda a, w_, b_: F.layer_norm(a, (8,), w_, b_, eps=1e-5), (xl, wl, bl), dl)
for name, got, exp in zip(("dx", "dw", "db"), layernorm_backward(dl, cache_l), ref):
    check_close(f"layernorm {name}", got, exp, atol=1e-10)

# %% [markdown]
# ## 4. BatchNorm, training mode
# Same algebra, but the statistics are over the **batch** dim $N$ for each feature (input $(N, D)$), and the variance is
# biased (divide by $N$), as PyTorch uses for normalization in training mode. With $\hat x$ and $\sigma^{-1}$ per feature:
# $$dx = \frac{\gamma\,\sigma^{-1}}{N}\Big(N\,dy - \textstyle\sum_n dy - \hat x \sum_n (dy\odot\hat x)\Big)$$
# This is makemore part 4's exercise 3 without its $n/(n-1)$ factor (it used Bessel's correction).

# %%
def batchnorm_forward(x, gamma, beta, eps=1e-5):
    """x (N, D), gamma/beta (D,) -> out (N, D), cache."""
    ### BEGIN SOLUTION
    mean = x.mean(0, keepdim=True)
    var = ((x - mean) ** 2).mean(0, keepdim=True)
    rstd = (var + eps) ** -0.5
    xhat = (x - mean) * rstd
    return gamma * xhat + beta, (xhat, gamma, rstd)
    ### END SOLUTION


def batchnorm_backward(dout, cache):
    """-> dx (N, D), dgamma (D,), dbeta (D,)."""
    ### BEGIN SOLUTION
    xhat, gamma, rstd = cache
    N = dout.shape[0]
    dbeta = dout.sum(0)
    dgamma = (dout * xhat).sum(0)
    dx = gamma * rstd / N * (N * dout - dout.sum(0) - xhat * (dout * xhat).sum(0))
    return dx, dgamma, dbeta
    ### END SOLUTION


xbn = torch.randn(16, 5, **dd) * 3 + 2
gbn, bbn = torch.randn(5, **dd), torch.randn(5, **dd)
dbn = torch.randn(16, 5, **dd)
out_bn, cache_bn = batchnorm_forward(xbn, gbn, bbn)
bn_ref = lambda a, g_, b_: F.batch_norm(a, None, None, g_, b_, training=True, eps=1e-5)
check_close("batchnorm forward", out_bn, bn_ref(xbn, gbn, bbn), atol=1e-10)
for name, got, exp in zip(("dx", "dgamma", "dbeta"), batchnorm_backward(dbn, cache_bn), autograd_grads(bn_ref, (xbn, gbn, bbn), dbn)):
    check_close(f"batchnorm {name}", got, exp, atol=1e-10)

# %% [markdown]
# ## 5. A central finite-difference gradient checker
# $\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}$, with error $O(h^2)$ from truncation plus
# $O(u/h)$ from rounding. In float64, $h=10^{-6}$ is a good compromise. Report the max **relative** error
# $|a - n| / \max(|a|, |n|, \text{tiny})$. Around $10^{-7}$ or less is a pass; $10^{-2}$ or more is a bug.
# `numerical_grad(f, x)` takes a scalar function `f` of one float64 tensor and returns a tensor shaped like `x`.

# %%
def numerical_grad(f, x, h=1e-6):
    ### BEGIN SOLUTION
    g = torch.zeros_like(x)
    xf, gf = x.view(-1), g.view(-1)        # views: writing xf[i] perturbs x in place
    for i in range(xf.numel()):
        old = xf[i].item()
        xf[i] = old + h
        fp = float(f(x))
        xf[i] = old - h
        fm = float(f(x))
        xf[i] = old
        gf[i] = (fp - fm) / (2 * h)
    return g
    ### END SOLUTION


def rel_err(a, n):
    return ((a - n).abs() / torch.maximum(a.abs(), n.abs()).clamp(min=1e-12)).max().item()


xq = torch.randn(3, 4, **dd)
check_close("numerical_grad on sum(x^3)", numerical_grad(lambda t: (t ** 3).sum(), xq.clone()), 3 * xq ** 2, atol=1e-6)
xq_copy = xq.clone()
numerical_grad(lambda u: (u ** 2).sum(), xq_copy)
check("numerical_grad leaves x unchanged", torch.equal(xq_copy, xq), "restore x[i] after perturbing it")
x_small = torch.randn(1, 2, 6, **dd)
d_small = torch.randn(1, 2, 6, **dd)
w_small, b_small = torch.randn(6, **dd), torch.randn(6, **dd)
num = numerical_grad(lambda t: (layernorm_forward(t, w_small, b_small)[0] * d_small).sum(), x_small.clone())
ana = layernorm_backward(d_small, layernorm_forward(x_small, w_small, b_small)[1])[0]
e = rel_err(ana, num)
print(f"layernorm dx: analytic vs finite differences, max rel err = {e:.2e}")
check("layernorm dx passes finite-difference check", e < 1e-5, f"rel err {e:.2e}")

# %% [markdown]
# ## 6. Full 2-layer MLP: forward, manual backward, compare to autograd
# `x (N, D) → h1 = x W1ᵀ + b1 (N, H) → a1 = relu(h1) → z = a1 W2ᵀ + b2 (N, C) → mean CE`. Reuse your pieces.
# `mlp_backward` returns a dict with keys `W1, b1, W2, b2` (each shaped like its parameter).

# %%
def mlp_forward(p, x, y):
    ### BEGIN SOLUTION
    h1 = x @ p["W1"].T + p["b1"]
    a1 = h1.clamp(min=0)
    z = a1 @ p["W2"].T + p["b2"]
    loss, dz = softmax_ce(z, y)
    return loss, (x, h1, a1, dz)
    ### END SOLUTION


def mlp_backward(p, cache):
    ### BEGIN SOLUTION
    x, h1, a1, dz = cache
    da1, dW2, db2 = linear_backward(dz, a1, p["W2"])
    dh1 = relu_backward(da1, h1)
    _, dW1, db1 = linear_backward(dh1, x, p["W1"])
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    ### END SOLUTION


N, D, H, C = 32, 10, 16, 4
torch.manual_seed(1)
p = {"W1": torch.randn(H, D, **dd) / D ** 0.5, "b1": torch.randn(H, **dd) * 0.1,
     "W2": torch.randn(C, H, **dd) / H ** 0.5, "b2": torch.randn(C, **dd) * 0.1}
xm, ym = torch.randn(N, D, **dd), torch.randint(0, C, (N,))
loss_m, cache_m = mlp_forward(p, xm, ym)
grads = mlp_backward(p, cache_m)
pa = {k: v.clone().requires_grad_() for k, v in p.items()}
loss_a = F.cross_entropy(F.linear(F.relu(F.linear(xm, pa["W1"], pa["b1"])), pa["W2"], pa["b2"]), ym)
loss_a.backward()
check_close("MLP loss", loss_m, loss_a, atol=1e-12)
for k in p:
    check_close(f"MLP d{k} vs autograd", grads[k], pa[k].grad, atol=1e-10)
num_b1 = numerical_grad(lambda t: mlp_forward({**p, "b1": t}, xm, ym)[0], p["b1"].clone())
print(f"MLP db1 vs finite differences: rel err {rel_err(grads['b1'], num_b1):.2e}")

# %% [markdown]
# A few SGD steps with your gradients should reduce the loss on this batch (overfitting one batch is the standard sanity check).

# %%
p_train = {k: v.clone() for k, v in p.items()}
losses = []
for step in range(200):
    loss_t, cache_t = mlp_forward(p_train, xm, ym)
    g_t = mlp_backward(p_train, cache_t)
    for k in p_train:
        p_train[k] -= 0.5 * g_t[k]
    losses.append(loss_t.item())
print(f"loss {losses[0]:.3f} -> {losses[-1]:.4f}")
check("manual-backprop SGD overfits one batch", losses[-1] < 0.05, f"final loss {losses[-1]:.3f}")

# %% [markdown]
# ## Say it out loud
# 1. For $Y = XW^\top + b$, derive $dX$, $dW$, $db$ from shapes alone. Why is $db$ a sum?
# 2. Why is the softmax+CE gradient $(p - y)/N$? What would you have to store if you backpropagated through softmax and log separately?
# 3. Walk through the three terms of the LayerNorm $dx$. Why does $dx$ sum to zero over $C$ (when $\gamma = 1$)?
# 4. LayerNorm vs BatchNorm backward: what changes? Why does the BatchNorm backward couple examples in a batch, and what does that imply
#    for tiny batches or gradient accumulation?
# 5. How do you choose $h$ in finite differences, and why central differences? Why float64?
# 6. ReLU's derivative at 0 is undefined: what does PyTorch use, and can it matter?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. $dX$ must be $(N, D_{in})$ from $dY\,(N, D_{out})$ and $W\,(D_{out}, D_{in})$, so $dX = dY\,W$. $dW$ must be $(D_{out}, D_{in})$, so
#    $dW = dY^\top X$. $b$ is broadcast over the $N$ rows in the forward, so its gradient sums the per-row contributions, $\sum_n dY_n$.
# 2. $L = \mathrm{lse}(z) - z_y$, and $\partial\,\mathrm{lse}/\partial z = \mathrm{softmax}(z)$, $\partial z_y/\partial z = e_y$. So $p - e_y$ per row, and $/N$
#    for the mean. Separately, you would store $p$ and push $-1/p_y$ through log, which overflows for tiny $p_y$, and then apply the full softmax
#    VJP. The fused form needs only $p$ and is bounded in $[-1, 1]$.
# 3. $g$ is the direct path through $\hat x$. $-\bar g$ comes from the mean: every $x_i$ shifts $\mu$. $-\hat x\,\overline{g\hat x}$ comes from the variance:
#    every $x_i$ changes $\sigma$. The outer $\sigma^{-1}$ is the scaling. Summing over $C$: $\sum(g - \bar g) = 0$, and
#    $\sum \hat x\cdot\text{const} = 0$ because $\sum\hat x = 0$. So $dx$ has zero mean over $C$: LayerNorm's output is invariant to adding a
#    constant to its input row.
# 4. The algebra is the same, but the means run over the batch dim $N$ instead of features $C$, and $\gamma,\beta$ gradients sum over $N$. Since
#    batch statistics depend on every example, $dx_n$ depends on all other examples in the batch. With tiny batches the statistics are noisy.
#    Gradient accumulation over micro-batches is **not** equivalent to one large batch for BN (different statistics), unlike for LayerNorm.
#    At eval, BN uses running statistics, so the train and eval functions differ.
# 5. Central differences have $O(h^2)$ truncation error vs $O(h)$ for one-sided. Rounding adds about $u|f|/h$. Minimizing the sum gives
#    $h \sim u^{1/3}$, about $10^{-5}$ to $10^{-6}$ in float64. In float32 ($u\approx 6\times10^{-8}$) the best achievable error is only about $10^{-3}$
#    relative, too coarse to separate small bugs from noise. Check only a random subset of coordinates for large tensors: each coordinate costs 2 forwards.
# 6. PyTorch uses 0 at exactly 0 (`threshold_backward` passes the gradient only where $x > 0$). It is a subgradient choice; it rarely matters
#    because exact zeros are rare with float inputs. It can matter for finite-difference checks right at a kink, and for inputs that are
#    exactly 0 (e.g. padding after a ReLU).
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
