# %% [markdown]
# # P2.8 · Layers by hand
#
# **ID:** P2.8 · **Target:** 15 min · **Plan week:** 3 · **Part:** P
#
# **The drill.** Re-implement the layers you use every day from `nn.Parameter`, buffers and tensor ops, then
# prove each one matches the `torch.nn` module after copying its weights/state. Guide §P2 "Must include":
# `nn.Linear` via `nn.Parameter`, Dropout, BatchNorm with running stats, manual SGD step under `no_grad`.
# This notebook adds LayerNorm and Embedding because they are the transformer-era versions of the same questions.
#
# **Allowed:** tensor ops (`@`, `mean`, `var`, `rsqrt`, `rand_like`, indexing, `index_add_`), `nn.Parameter`,
# `register_buffer`, `torch.nn.init`. **Not allowed inside your code:** the module you are re-implementing
# or its `torch.nn.functional` twin (`F.linear`, `F.dropout`, `F.batch_norm`, `F.layer_norm`, `F.embedding`),
# and `torch.optim` inside `sgd_step`. The tests use those as the reference.
#
# **Fill in** every `raise NotImplementedError` and run the test cell under it.
#
# ## Sources
# - PyTorch source, torch 2.14.1 `torch/nn/modules/linear.py`, `Linear.reset_parameters` (line 124 in this venv):
#   `init.kaiming_uniform_(self.weight, a=math.sqrt(5))`, bias `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`; the code
#   comment points to https://github.com/pytorch/pytorch/issues/57109 for why `a=√5` means `U(±1/√fan_in)`.
#   The test cell prints that source with `inspect` so you can see it yourself.
# - PyTorch docs (the reference semantics):
#   `Linear` https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html ·
#   `Dropout` https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html ·
#   `BatchNorm1d` (biased var to normalize, unbiased var in `running_var`, momentum convention)
#   https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html ·
#   `LayerNorm` https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html ·
#   `Embedding` https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html ·
#   `SGD` (update pseudo-code) https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html
# - K. He, X. Zhang, S. Ren, J. Sun, *Delving Deep into Rectifiers* (Kaiming init), ICCV 2015, arXiv:1502.01852 — https://arxiv.org/abs/1502.01852
# - N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov, *Dropout: A Simple Way to Prevent Neural
#   Networks from Overfitting*, JMLR 15, 2014 — https://jmlr.org/papers/v15/srivastava14a.html
#   (the paper scales weights by p at test time; PyTorch uses the equivalent "inverted" form, scaling by 1/(1-p) at train time).
# - S. Ioffe, C. Szegedy, *Batch Normalization*, ICML 2015, arXiv:1502.03167 — https://arxiv.org/abs/1502.03167
# - J. L. Ba, J. R. Kiros, G. Hinton, *Layer Normalization*, 2016, arXiv:1607.06450 — https://arxiv.org/abs/1607.06450
# - I. Sutskever, J. Martens, G. Dahl, G. Hinton, *On the importance of initialization and momentum in deep learning*,
#   ICML 2013 (PMLR 28) — https://proceedings.mlr.press/v28/sutskever13.html

# %%
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prep import Session, check, check_close

s = Session("P2.8", target_min=15)
torch.manual_seed(0)

# %% [markdown]
# ## 1. `Linear` from `nn.Parameter`
# $y = xW^\top + b$ with $W \in \mathbb{R}^{\text{out}\times\text{in}}$ (PyTorch stores **out × in**).
# Default init: `kaiming_uniform_(a=√5)` draws $U(-\text{bound}, \text{bound})$ with
# bound $= \text{gain}\sqrt{3/\text{fan\_in}}$ and gain $=\sqrt{2/(1+a^2)} = \sqrt{1/3}$, so
# bound $= 1/\sqrt{\text{fan\_in}}$. Bias: also $U(\pm 1/\sqrt{\text{fan\_in}})$. Variance $= \text{bound}^2/3 = 1/(3\,\text{fan\_in})$,
# a third of what "variance-preserving" LeCun init ($1/\text{fan\_in}$) would give.

# %%
class MyLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        ### BEGIN SOLUTION
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.empty(d_out)) if bias else None
        bound = 1 / math.sqrt(d_in)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)
        ### END SOLUTION

    def forward(self, x):
        ### BEGIN SOLUTION
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y
        ### END SOLUTION


print(inspect.getsource(nn.Linear.reset_parameters))  # the claim above, straight from the installed torch
fan_in = 256
lin, ref = MyLinear(fan_in, 512), nn.Linear(fan_in, 512)
bound = 1 / math.sqrt(fan_in)
check("weight is an nn.Parameter of shape (out, in)",
      isinstance(lin.weight, nn.Parameter) and lin.weight.shape == (512, fan_in))
check("Parameters registered (weight + bias)", len(list(lin.parameters())) == 2)
for name, w in (("MyLinear", lin.weight), ("nn.Linear", ref.weight)):
    check(f"{name} init within ±1/sqrt(fan_in)", w.abs().max().item() <= bound + 1e-7)
    check(f"{name} init std = bound/sqrt(3)", abs(w.std().item() / (bound / math.sqrt(3)) - 1) < 0.03,
          f"std {w.std().item():.4f}, expected {bound / math.sqrt(3):.4f}")
check("bias init within ±1/sqrt(fan_in) and not all zero",
      lin.bias.abs().max().item() <= bound + 1e-7 and lin.bias.abs().max().item() > bound / 2)
with torch.no_grad():
    ref.weight.copy_(lin.weight); ref.bias.copy_(lin.bias)
x = torch.randn(8, fan_in, requires_grad=True)
check_close("Linear forward", lin(x), ref(x))
lin(x).square().sum().backward()
gx = x.grad.clone(); x.grad = None
ref(x).square().sum().backward()
check_close("Linear dL/dx", gx, x.grad)
check_close("Linear dL/dW", lin.weight.grad, ref.weight.grad, atol=1e-4)
check("bias=False works", MyLinear(4, 3, bias=False).bias is None and MyLinear(4, 3, bias=False)(torch.randn(2, 4)).shape == (2, 3))

# %% [markdown]
# ## 2. Inverted dropout
# Train: keep each unit with prob $1-p$ and scale kept units by $1/(1-p)$, so $\mathbb{E}[\tilde x] = x$ and eval is the
# identity (no rescaling needed at test time). Edge case: `p=1` must return zeros, not NaN.

# %%
def dropout(x, p=0.5, training=True):
    ### BEGIN SOLUTION
    if not training or p == 0:
        return x
    if p == 1:
        return torch.zeros_like(x)
    mask = (torch.rand_like(x) >= p).to(x.dtype)
    return x * mask / (1 - p)
    ### END SOLUTION


torch.manual_seed(1)
xd = torch.rand(2000, 500) + 0.5                     # positive, so a zero means "dropped"
for p in (0.1, 0.3, 0.7):
    yd = dropout(xd, p, training=True)
    frac0 = (yd == 0).float().mean().item()
    check(f"p={p}: dropped fraction ~ p", abs(frac0 - p) < 0.01, f"got {frac0:.3f}")
    kept = yd != 0
    check_close(f"p={p}: kept units scaled by 1/(1-p)", yd[kept], xd[kept] / (1 - p))
    check(f"p={p}: expectation preserved", abs(yd.mean().item() / xd.mean().item() - 1) < 0.01)
check_close("eval mode is identity", dropout(xd, 0.3, training=False), xd)
check_close("p=1 gives zeros", dropout(xd[:4], 1.0), torch.zeros_like(xd[:4]))
xg = xd[:4].clone().requires_grad_()
yg = dropout(xg, 0.4)
yg.sum().backward()
check_close("grad = mask / (1-p)", xg.grad, (yg != 0).float() / 0.6)

# %% [markdown]
# ## 3. BatchNorm1d with running statistics
# Train mode, per feature over the batch: $\hat x = (x-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$ with the **biased** $\sigma_B^2$
# (divide by $N$); $y=\gamma\hat x+\beta$. Then, without tracking gradients:
# `running = (1 - momentum) * running + momentum * batch_stat`, using the **unbiased** variance ($N-1$) for
# `running_var`. PyTorch's `momentum=0.1` is the weight on the *new* value, the opposite of the optimizer meaning.
# Eval mode normalizes with the running stats. Buffers (`register_buffer`) go in `state_dict` but not in `parameters()`.

# %%
class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        ### BEGIN SOLUTION
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        ### END SOLUTION

    def forward(self, x):  # x: (N, C)
        ### BEGIN SOLUTION
        if self.training:
            mean, var = x.mean(0), x.var(0, unbiased=False)
            with torch.no_grad():
                n = x.shape[0]
                m = self.momentum
                self.running_mean.mul_(1 - m).add_(m * mean)
                self.running_var.mul_(1 - m).add_(m * var * n / (n - 1))
                self.num_batches_tracked += 1
        else:
            mean, var = self.running_mean, self.running_var
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias
        ### END SOLUTION


torch.manual_seed(2)
C = 6
bn, bn_ref = MyBatchNorm1d(C), nn.BatchNorm1d(C)
check("weight/bias are Parameters, running stats are buffers",
      {n for n, _ in bn.named_parameters()} == {"weight", "bias"}
      and {"running_mean", "running_var"} <= set(bn.state_dict()))
with torch.no_grad():
    for p_ in (bn.weight, bn.bias):
        p_.copy_(torch.randn(C))
bn_ref.load_state_dict(bn.state_dict())
scale = torch.tensor([3.0, 3.0, 1.0, 0.3, 0.01, 0.01])  # tiny-variance features make eps placement matter
for step in range(5):  # small batches so biased vs unbiased variance differ visibly
    xb = torch.randn(8, C) * scale + 2
    xb1, xb2 = xb.clone().requires_grad_(), xb.clone().requires_grad_()
    out, out_ref = bn.train()(xb1), bn_ref.train()(xb2)
    up = torch.randn(8, C)  # random upstream gradient (sum(out) would give dL/dx = 0 for BN)
    (out * up).sum().backward(); (out_ref * up).sum().backward()
check_close("BN train output", out, out_ref)
check_close("BN dL/dx (train)", xb1.grad, xb2.grad, atol=1e-4, rtol=1e-3)
check_close("BN dL/dgamma", bn.weight.grad, bn_ref.weight.grad, atol=1e-4, rtol=1e-3)
check_close("BN running_mean after 5 steps", bn.running_mean, bn_ref.running_mean)
check_close("BN running_var (unbiased) after 5 steps", bn.running_var, bn_ref.running_var)
xe = torch.randn(4, C)
check_close("BN eval uses running stats", bn.eval()(xe), bn_ref.eval()(xe))
check_close("eval does not update running stats", bn.running_mean, bn_ref.running_mean)

# %% [markdown]
# ## 4. LayerNorm
# Normalize each example over its **last** `D` features (no batch statistics, no running stats, same in train and
# eval), biased variance, $\epsilon$ inside the square root, elementwise affine $\gamma,\beta \in \mathbb{R}^D$.

# %%
class MyLayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        ### BEGIN SOLUTION
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        ### END SOLUTION

    def forward(self, x):  # x: (..., d)
        ### BEGIN SOLUTION
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias
        ### END SOLUTION


torch.manual_seed(3)
D = 8
ln, ln_ref = MyLayerNorm(D), nn.LayerNorm(D)
with torch.no_grad():
    ln.weight.copy_(torch.randn(D)); ln.bias.copy_(torch.randn(D))
ln_ref.load_state_dict(ln.state_dict())
xl = (torch.randn(2, 5, D) * 4 + 1).requires_grad_()
check_close("LayerNorm forward (B, T, D)", ln(xl), ln_ref(xl))
g1 = torch.autograd.grad(ln(xl).pow(2).sum(), xl)[0]
g2 = torch.autograd.grad(ln_ref(xl).pow(2).sum(), xl)[0]
check_close("LayerNorm dL/dx", g1, g2, atol=1e-4)

# %% [markdown]
# ## 5. Embedding (and its backward)
# A lookup `W[idx]` is a one-hot matmul, $\text{onehot}(idx)\,W$, so its gradient is a **scatter-add** of the upstream
# gradient into the rows that were used (repeated ids accumulate). Default init $\mathcal N(0,1)$. With `padding_idx`,
# that row starts at zero and never receives gradient.

# %%
class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, dim, padding_idx=None):
        super().__init__()
        self.padding_idx = padding_idx
        ### BEGIN SOLUTION
        self.weight = nn.Parameter(torch.randn(num_embeddings, dim))
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].zero_()
        ### END SOLUTION

    def forward(self, idx):
        ### BEGIN SOLUTION
        out = self.weight[idx]
        if self.padding_idx is not None:
            out = out * (idx != self.padding_idx).unsqueeze(-1)  # no gradient flows to the pad row
        return out
        ### END SOLUTION


def embedding_weight_grad(idx, grad_out, num_embeddings):
    """Gradient of the loss w.r.t. the embedding table, by hand. idx: (...), grad_out: (..., dim)."""
    ### BEGIN SOLUTION
    g = torch.zeros(num_embeddings, grad_out.shape[-1], dtype=grad_out.dtype)
    return g.index_add_(0, idx.reshape(-1), grad_out.reshape(-1, grad_out.shape[-1]))
    ### END SOLUTION


torch.manual_seed(4)
V, E, PAD = 10, 4, 0
emb, emb_ref = MyEmbedding(V, E, padding_idx=PAD), nn.Embedding(V, E, padding_idx=PAD)
check_close("pad row initialized to zero", emb.weight[PAD], torch.zeros(E))
check("init ~ N(0, 1)", abs(MyEmbedding(1000, 100).weight.std().item() - 1) < 0.02)
emb_ref.load_state_dict(emb.state_dict())
ids = torch.tensor([[1, 2, 2, 0], [3, 2, 0, 9]])     # repeated id 2, padding 0
check_close("Embedding forward", emb(ids), emb_ref(ids))
up = torch.randn(2, 4, E)
(emb(ids) * up).sum().backward(); (emb_ref(ids) * up).sum().backward()
check_close("Embedding weight grad (repeats accumulate, pad row zero)", emb.weight.grad, emb_ref.weight.grad)
ids_np = torch.tensor([[1, 2, 2, 5], [3, 2, 7, 9]])
e2 = nn.Embedding(V, E)
(e2(ids_np) * up).sum().backward()
check_close("embedding_weight_grad by hand", embedding_weight_grad(ids_np, up, V), e2.weight.grad)

# %% [markdown]
# ## 6. Manual SGD step (momentum + weight decay) under `torch.no_grad()`
# PyTorch's SGD (see the docs' pseudo-code): $g \leftarrow g + \lambda\theta$ (L2-style decay, added to the gradient,
# so it **goes through momentum**); $b \leftarrow g$ on the first step, else $b \leftarrow \mu b + (1-\tau) g$
# ($\tau$ = dampening); with Nesterov $g \leftarrow g + \mu b$ else $g \leftarrow b$; $\theta \leftarrow \theta - \eta g$.
# Note the learning rate sits **outside** the buffer (Sutskever et al. put it inside); the two agree only for a constant LR.
# Update in place under `no_grad` (otherwise autograd records the update, or errors on a leaf that requires grad).

# %%
@torch.no_grad()
def sgd_step(params, bufs, lr, momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=False):
    """params: list of tensors with .grad. bufs: dict id(p) -> momentum buffer (mutated in place)."""
    ### BEGIN SOLUTION
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        if weight_decay:
            g = g + weight_decay * p
        if momentum:
            b = bufs.get(id(p))
            if b is None:
                b = bufs[id(p)] = g.clone()
            else:
                b.mul_(momentum).add_(g, alpha=1 - dampening)
            g = g + momentum * b if nesterov else b
        p.add_(g, alpha=-lr)
    ### END SOLUTION


def make_net():
    torch.manual_seed(5)
    return nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 3))


torch.manual_seed(6)
batches = [(torch.randn(32, 5), torch.randint(0, 3, (32,))) for _ in range(6)]
configs = [dict(lr=0.1, momentum=0.9, weight_decay=1e-2),
           dict(lr=0.05, momentum=0.9, weight_decay=1e-2, nesterov=True),
           dict(lr=0.1, momentum=0.8, dampening=0.3),
           dict(lr=0.2)]
for cfg in configs:
    net_a, net_b = make_net(), make_net()
    opt = torch.optim.SGD(net_b.parameters(), **cfg)
    bufs = {}
    for xb, yb in batches:
        for net in (net_a, net_b):
            net.zero_grad()
            F.cross_entropy(net(xb), yb).backward()
        sgd_step(list(net_a.parameters()), bufs, **cfg)
        opt.step()
    for (n, pa), pb in zip(net_a.named_parameters(), net_b.parameters()):
        check_close(f"SGD {cfg} param {n} after 6 steps", pa, pb, atol=1e-6)
check("update did not create autograd history", all(p.grad_fn is None and p.is_leaf for p in net_a.parameters()))

# %% [markdown]
# ## Say it out loud
# 1. What exactly changes between `model.train()` and `model.eval()` for Dropout and for BatchNorm? What does neither change?
# 2. Why "inverted" dropout? What would you have to do at test time otherwise?
# 3. BatchNorm: why the biased variance for normalizing but the unbiased one for `running_var`? What does `momentum=0.1` mean here?
# 4. What goes wrong with BatchNorm at batch size 1, or with very small / non-i.i.d. batches? Why do transformers use LayerNorm?
# 5. What is `nn.Linear`'s default init distribution and its variance? Is it variance-preserving for ReLU nets?
# 6. Why does an optimizer step run under `torch.no_grad()`? Why is `weight_decay` in `torch.optim.SGD`/`Adam` not the same as AdamW's decay?
# 7. What is the gradient of an embedding lookup? Why is it sparse, and what happens to repeated ids?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Dropout: train samples a mask and scales by 1/(1-p); eval is the identity. BatchNorm: train normalizes with
#    the current batch's statistics and updates the running buffers; eval normalizes with the running buffers and
#    updates nothing. Neither mode switches gradient tracking off. That is `no_grad`/`inference_mode`, which you still need in eval loops.
# 2. Scaling the kept units by 1/(1-p) during training keeps $\mathbb E[\tilde x]=x$. The eval network is then just the
#    trained network, with nothing to rescale. The original paper instead multiplies weights by the keep probability at test time
#    (Srivastava et al. 2014). It is the same in expectation for a linear layer, but it is easy to forget.
# 3. Normalization uses the maximum-likelihood batch statistics. The batch is all there is, and the
#    gradient formula in the paper uses $1/m$. `running_var` estimates the population variance for use at test time, so it applies Bessel's
#    correction $m/(m-1)$ (Ioffe & Szegedy 2015 use the unbiased estimate for inference; the PyTorch docs state both
#    estimators explicitly). `momentum` is the weight on the **new** batch statistic: `r = 0.9 r + 0.1 batch`.
# 4. With N=1 the per-feature batch variance is 0, so the output is just $\beta$ and the layer carries no information. PyTorch raises
#    "Expected more than 1 value per channel when training" for a (1, C) input. Small or correlated batches give noisy or biased
#    statistics and a train/test mismatch, and the batch elements stop being independent. LayerNorm normalizes each token
#    by itself: no dependence on batch size or sequence padding, and identical behaviour in train and eval. That suits variable-length sequences and tiny per-GPU batches.
# 5. $U(\pm 1/\sqrt{\text{fan\_in}})$ for the weight (`kaiming_uniform_(a=√5)`) and the bias. Variance $1/(3\,\text{fan\_in})$.
#    That is smaller than He init ($2/\text{fan\_in}$, which preserves variance through ReLU) by a factor of 6, so activations shrink with depth
#    in a deep plain ReLU MLP. It works in practice for shallow nets and with normalization layers. PyTorch keeps it for backward compatibility
#    (pytorch/pytorch#57109).
# 6. The update is not part of the model's computation. Without `no_grad`, an in-place update of a leaf that requires grad raises,
#    and an out-of-place update would make the parameter a non-leaf with history. SGD/Adam `weight_decay` adds $\lambda\theta$
#    to the gradient (L2). In Adam that term is then divided by $\sqrt{\hat v}$, so parameters with large gradients get little decay.
#    AdamW applies $\theta \leftarrow \theta - \eta\lambda\theta$ directly, decoupled from the adaptive scaling (Loshchilov & Hutter, arXiv:1711.05101).
#    For plain SGD the two are equivalent up to an LR rescaling of λ.
# 7. $\partial L/\partial W = \text{onehot}(idx)^\top\,\partial L/\partial \text{out}$. Only the looked-up rows get a nonzero gradient. Repeated ids
#    accumulate (scatter-add / `index_add_`). `sparse=True` returns it as a sparse tensor. `padding_idx` rows are excluded.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
