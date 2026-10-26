# %% [markdown]
# # A2.4 · Custom `torch.autograd.Function` with a manual backward
#
# **ID:** A2.4 · **Target:** 25 min · **Plan week:** 1-3 · **Part:** A
#
# **The drill.** Write five custom autograd Functions and verify each one with `torch.autograd.gradcheck` in float64:
# 1. `LinearFn` (`y = x Wᵀ + b`) using `ctx.save_for_backward` and `ctx.needs_input_grad`;
# 2. a numerically stable `SigmoidFn` that saves its **output**;
# 3. a stable `SoftplusFn` whose backward is itself differentiable, so double backward works (`gradgradcheck`);
# 4. a straight-through estimator `STERound` (forward rounds, backward is the identity);
# 5. `ClampFn(x, lo, hi)` with **non-tensor** arguments, whose backward returns `None` for them.
#
# Then see three failure modes: gradcheck in float32, gradcheck catching a wrong backward, and `@once_differentiable`
# refusing double backward. One more demo shows why you sometimes need a custom Function at all: the `torch.where` NaN-gradient trap.
# Guide A2 "must include": "custom `autograd.Function` with manual backward".
#
# **Rules.** `forward` and `backward` are `@staticmethod`s. `backward` returns **one gradient per `forward` input**
# (after `ctx`), `None` for inputs that are not tensors or need no gradient. Allowed: elementwise torch ops, `@`, reductions.
# Not allowed inside your Functions: `F.linear`, `torch.sigmoid`, `F.softplus`, `torch.clamp` (they are the references).
#
# ## Sources
# - PyTorch docs, *Extending PyTorch → Extending torch.autograd* (`save_for_backward` rules, `needs_input_grad`,
#   `None` for non-tensor inputs, `once_differentiable`, `setup_context`, `mark_non_differentiable`, `mark_dirty`,
#   gradcheck): https://docs.pytorch.org/docs/stable/notes/extending.html
# - PyTorch tutorial, *Double Backward with Custom Functions* (when double backward works, saving inputs vs outputs vs
#   intermediates, `gradgradcheck`): https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
# - `torch.autograd.gradcheck` (note: "The default values are designed for input of double precision"):
#   https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html ·
#   `ctx.save_for_backward`: https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html
# - Y. Bengio, N. Léonard, A. Courville, *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional
#   Computation*, 2013, arXiv:1308.3432 — https://arxiv.org/abs/1308.3432 (the straight-through estimator).

# %%
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck, gradgradcheck
from torch.autograd.function import once_differentiable

from prep import Session, check, check_close, seed_everything

s = Session("A2.4", target_min=25)
seed_everything(0)
dd = dict(dtype=torch.float64)

# %% [markdown]
# ## 1. `LinearFn`: `save_for_backward` and `needs_input_grad`
# Shapes: `x (N, in)`, `W (out, in)`, `b (out,)`, `y (N, out)`. With upstream `g = ∂L/∂y (N, out)`:
# `dx = g W (N, in)`, `dW = gᵀ x (out, in)`, `db = Σ_n g (out,)`. Each gradient has the shape of its input.
# Skip any gradient whose `ctx.needs_input_grad[i]` is False (return `None`). This saves a matmul, e.g. `dx` for the first layer.

# %%
class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b):
        ### BEGIN SOLUTION
        ctx.save_for_backward(x, W)
        return x @ W.T + b
        ### END SOLUTION

    @staticmethod
    def backward(ctx, g):
        ### BEGIN SOLUTION
        x, W = ctx.saved_tensors
        dx = g @ W if ctx.needs_input_grad[0] else None
        dW = g.T @ x if ctx.needs_input_grad[1] else None
        db = g.sum(0) if ctx.needs_input_grad[2] else None
        return dx, dW, db
        ### END SOLUTION


x = torch.randn(5, 3, **dd, requires_grad=True)
W = torch.randn(4, 3, **dd, requires_grad=True)
b = torch.randn(4, **dd, requires_grad=True)
check("LinearFn gradcheck", gradcheck(LinearFn.apply, (x, W, b)))
g = torch.randn(5, 4, **dd)
LinearFn.apply(x, W, b).backward(g)
mine = [t.grad.clone() for t in (x, W, b)]
for t in (x, W, b):
    t.grad = None
F.linear(x, W, b).backward(g)
check_close("LinearFn grads == F.linear grads", torch.cat([m.flatten() for m in mine]), torch.cat([t.grad.flatten() for t in (x, W, b)]))


class FakeCtx:  # call backward directly to see what it returns when x needs no grad
    saved_tensors = (x.detach(), W.detach())
    needs_input_grad = (False, True, True)


grads = LinearFn.backward(FakeCtx(), g)
check("LinearFn skips dx when not needed", grads[0] is None and grads[1] is not None, "use ctx.needs_input_grad")


class MyLinear(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        ref = nn.Linear(din, dout)                       # reuse PyTorch's default init
        self.weight, self.bias = nn.Parameter(ref.weight.detach().clone()), nn.Parameter(ref.bias.detach().clone())

    def forward(self, x):
        return LinearFn.apply(x, self.weight, self.bias)


m = MyLinear(3, 4)
xin = torch.randn(8, 3)
check_close("MyLinear forward", m(xin), F.linear(xin, m.weight, m.bias))

# %% [markdown]
# ## 2. `SigmoidFn` (stable forward, save the output)
# $\sigma(x) = 1/(1+e^{-x})$ overflows `exp` for very negative $x$. With $e = e^{-|x|} \in (0, 1]$:
# $\sigma(x) = 1/(1+e)$ if $x\ge 0$, else $e/(1+e)$. Backward: $\sigma'(x) = \sigma(1-\sigma)$, so save the **output** $y$.

# %%
class SigmoidFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ### BEGIN SOLUTION
        e = torch.exp(-x.abs())
        y = torch.where(x >= 0, 1 / (1 + e), e / (1 + e))
        ctx.save_for_backward(y)
        return y
        ### END SOLUTION

    @staticmethod
    def backward(ctx, g):
        ### BEGIN SOLUTION
        (y,) = ctx.saved_tensors
        return g * y * (1 - y)
        ### END SOLUTION


xs = torch.cat([torch.randn(20, **dd) * 5, torch.tensor([-1000.0, -50.0, 0.0, 50.0, 1000.0], **dd)]).requires_grad_()
check("SigmoidFn gradcheck", gradcheck(SigmoidFn.apply, (torch.randn(10, **dd, requires_grad=True),)))
ys = SigmoidFn.apply(xs)
ys.sum().backward()
xr = xs.detach().clone().requires_grad_()
torch.sigmoid(xr).sum().backward()
check_close("SigmoidFn forward incl. |x|=1000", ys, torch.sigmoid(xr), atol=1e-12)
check_close("SigmoidFn grad incl. |x|=1000", xs.grad, xr.grad, atol=1e-12)
check("SigmoidFn finite", bool(torch.isfinite(ys).all() and torch.isfinite(xs.grad).all()))

# %% [markdown]
# **Why not just write it with `torch.where` and autograd?** `where` sends a zero gradient to the branch it did not
# pick, but autograd still differentiates through that branch. Here the unused branch is $1/(1+e^{-x})$ at $x=-1000$.
# There $e^{-x} = \infty$, so its local derivative involves $0\cdot\infty$ = NaN, and $0 \cdot \text{NaN} = \text{NaN}$.
# A custom backward (or computing the branches on safe inputs) avoids that.

# %%
xw = torch.tensor([-1000.0, 0.5, 1000.0], requires_grad=True)
naive = torch.where(xw >= 0, 1 / (1 + torch.exp(-xw)), torch.exp(xw) / (1 + torch.exp(xw)))
naive.sum().backward()
print("forward :", naive.detach().tolist())
print("autograd through torch.where:", xw.grad.tolist(), " <- NaN at both extremes")

# %% [markdown]
# ## 3. `SoftplusFn` with a differentiable backward (double backward)
# $\mathrm{softplus}(x) = \log(1+e^x) = \max(x,0) + \log(1+e^{-|x|})$, with derivative $\sigma(x)$. Save the **input** and
# compute $\sigma(x)$ in `backward` **with ordinary differentiable torch ops** (not your `SigmoidFn`, whose own backward is
# only first-order friendly here). Then autograd can record the backward, and `gradgradcheck` passes:
# $\mathrm{softplus}''(x) = \sigma(x)(1-\sigma(x))$. Careful: the `exp(-|x|)` trick from section 2 is fine for values, but
# autograd differentiates `|x|` at 0 as 0, so the *second* derivative at exactly $x=0$ comes out 0 instead of 1/4. Feed
# each `where` branch a safe input instead: `where(x >= 0, x, 0)` for the $x\ge0$ branch and `where(x < 0, x, 0)` for the other.
# (Not `x.clamp(min=0)`: its gradient at exactly 0 is also 0 in current PyTorch.)

# %%
class SoftplusFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ### BEGIN SOLUTION
        ctx.save_for_backward(x)
        return x.clamp(min=0) + torch.log1p(torch.exp(-x.abs()))
        ### END SOLUTION

    @staticmethod
    def backward(ctx, g):
        ### BEGIN SOLUTION
        (x,) = ctx.saved_tensors
        xp, xn = torch.where(x >= 0, x, 0.0), torch.where(x < 0, x, 0.0)  # safe input per branch, no kink at 0
        sig = torch.where(x >= 0, 1 / (1 + torch.exp(-xp)), torch.exp(xn) / (1 + torch.exp(xn)))
        return g * sig
        ### END SOLUTION


xp = torch.randn(12, **dd, requires_grad=True)
check("SoftplusFn gradcheck", gradcheck(SoftplusFn.apply, (xp,)))
check("SoftplusFn gradgradcheck (double backward)", gradgradcheck(SoftplusFn.apply, (xp,)))
xb = torch.tensor([-800.0, -30.0, 0.0, 30.0, 800.0], **dd, requires_grad=True)
yb = SoftplusFn.apply(xb)
(gb,) = torch.autograd.grad(yb.sum(), xb, create_graph=True)
(hb,) = torch.autograd.grad(gb.sum(), xb)
check_close("softplus at extremes", yb, F.softplus(xb.detach()), atol=1e-12)
check_close("softplus' = sigmoid", gb, torch.sigmoid(xb.detach()), atol=1e-12)
check_close("softplus'' = s(1-s)", hb, torch.sigmoid(xb.detach()) * (1 - torch.sigmoid(xb.detach())), atol=1e-12)

# %% [markdown]
# ## 4. Straight-through estimator for `round`
# `round` has zero gradient almost everywhere, so nothing upstream learns. The STE (Bengio et al. 2013) uses the
# forward `round(x)` but pretends the backward is the identity. This is how quantization-aware training and VQ-VAE codebooks
# pass gradients. Also write the one-line **detach trick** that gives the same forward and backward without a Function.

# %%
class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ### BEGIN SOLUTION
        return x.round()
        ### END SOLUTION

    @staticmethod
    def backward(ctx, g):
        ### BEGIN SOLUTION
        return g
        ### END SOLUTION


def ste_round_detach(x):
    ### BEGIN SOLUTION
    return x + (x.round() - x).detach()
    ### END SOLUTION


xq = (torch.randn(10) * 3).requires_grad_()
gq = torch.randn(10)
yq = STERound.apply(xq)
yq.backward(gq)
check_close("STE forward = round", yq, xq.detach().round())
check_close("STE backward = identity", xq.grad, gq)
xq2 = xq.detach().clone().requires_grad_()
y2 = ste_round_detach(xq2)
y2.backward(gq)
check_close("detach trick forward", y2, xq.detach().round())
check_close("detach trick backward", xq2.grad, gq)
xq3 = xq.detach().clone().requires_grad_()
xq3.round().backward(gq)
print("plain torch.round grad (useless for learning):", xq3.grad[:4].tolist())

# %% [markdown]
# ## 5. Non-tensor arguments: `ClampFn(x, lo, hi)`
# `lo`, `hi` are Python floats. Store non-tensors directly on `ctx` (`save_for_backward` is for tensors only). The
# backward returns **three** values: the gradient for `x`, then `None, None`. Gradient: 1 inside `(lo, hi)`, 0 outside.
# Exactly at `lo`/`hi` the derivative is undefined; `torch.clamp` returns 0 there in torch 2.14, and the test avoids those points.

# %%
class ClampFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lo, hi):
        ### BEGIN SOLUTION
        ctx.save_for_backward(x)
        ctx.lo, ctx.hi = lo, hi
        return torch.where(x < lo, lo, torch.where(x > hi, hi, x))
        ### END SOLUTION

    @staticmethod
    def backward(ctx, g):
        ### BEGIN SOLUTION
        (x,) = ctx.saved_tensors
        return g * ((x > ctx.lo) & (x < ctx.hi)).to(g.dtype), None, None
        ### END SOLUTION


xc = torch.randn(30, **dd) * 2
xc = xc[(xc.abs() - 1).abs() > 1e-3].requires_grad_()   # stay away from the kinks at +-1 for finite differences
check("ClampFn gradcheck", gradcheck(lambda t: ClampFn.apply(t, -1.0, 1.0), (xc,)))
gc = torch.randn_like(xc)
ClampFn.apply(xc, -1.0, 1.0).backward(gc)
xcr = xc.detach().clone().requires_grad_()
torch.clamp(xcr, -1.0, 1.0).backward(gc)
check_close("ClampFn forward", ClampFn.apply(xc.detach(), -1.0, 1.0), torch.clamp(xc.detach(), -1.0, 1.0))
check_close("ClampFn grad == torch.clamp grad", xc.grad, xcr.grad)

# %% [markdown]
# ## 6. Failure modes worth seeing once
# (a) **gradcheck in float32.** The default `eps=1e-6` perturbation is about 8 float32 ulps at 1.0
# (float32 eps ≈ 1.2e-7). The finite-difference quotient then has error of order 1e-1, far above `atol=1e-5`.
# (b) **gradcheck catches a wrong backward**, here `σ(-x)` instead of `σ(x)`. (c) **`@once_differentiable`** makes the
# backward run without graph recording, and asking for a second derivative raises instead of silently returning garbage.

# %%
x32 = torch.randn(6, dtype=torch.float32, requires_grad=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ok32 = gradcheck(SoftplusFn.apply, (x32,), raise_exception=False)
print("(a) float32 gradcheck passes?", ok32)


class BadSoftplus(SoftplusFn):
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.sigmoid(-x)   # wrong sign


print("(b) gradcheck on a wrong backward passes?", gradcheck(BadSoftplus.apply, (xp,), raise_exception=False))


class SoftplusOnce(SoftplusFn):
    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.sigmoid(x)


yo = SoftplusOnce.apply(xp)
(g1,) = torch.autograd.grad((yo ** 2).sum(), xp, create_graph=True)  # upstream grad 2y depends on xp
try:
    g1.sum().backward()
    print("(c) double backward ran (unexpected)")
except RuntimeError as e:
    print("(c) once_differentiable double backward ->", str(e).split("\n")[0][:110])
check("float32 gradcheck is unreliable here", not ok32, "expected float32 gradcheck to fail")

# %% [markdown]
# ## Say it out loud
# 1. What does `ctx.save_for_backward` do that `ctx.x = x` does not? What should you store directly on `ctx`?
# 2. What must `backward` return, and in what order? What if `forward` takes a Python float or a bool flag?
# 3. What is `ctx.needs_input_grad` for? Is skipping a gradient a correctness requirement?
# 4. Why does `gradcheck` want float64? What else makes it fail spuriously?
# 5. When does a custom Function support double backward, and what does `@once_differentiable` do?
# 6. Explain the straight-through estimator and the `x + (f(x) - x).detach()` trick. Where is it used?
# 7. Why can `torch.where` produce NaN gradients even though the forward is fine? Two ways to fix it.

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. `save_for_backward` registers the tensors with autograd. It checks that they were not modified in place before backward (version counter),
#    it lets the engine free them as soon as backward is done, and saved inputs/outputs stay connected for double backward. Storing tensors as
#    `ctx` attributes bypasses all of that (and can create reference cycles). Non-tensors (ints, floats, shapes, flags) go directly on `ctx`
#    (Extending PyTorch docs).
# 2. One value per `forward` argument after `ctx`, in the same order, each with the shape of that input (or `None`). Non-tensor arguments must
#    get `None`. Returning the wrong count raises "returned an incorrect number of gradients".
# 3. A tuple of booleans, one per input, saying whether that input needs a gradient. Checking it is optional and only for efficiency: you can
#    skip expensive work such as `dx = g @ W` for the first layer or for frozen weights. Returning a gradient that was not needed is not an error
#    (Extending PyTorch docs).
# 4. Central differences have truncation error $O(\varepsilon^2)$ and rounding error $O(u/\varepsilon)$, with $u$ the unit roundoff. With float32
#    ($u\approx 6\times10^{-8}$) and the default $\varepsilon=10^{-6}$, the rounding term is about $0.1$, far above the default tolerances. The gradcheck
#    docs say the defaults are designed for double precision. Other spurious failures come from non-differentiable points (kinks: clamp/ReLU at the
#    boundary, hence the filtering above) and from nondeterministic ops.
# 5. When autograd can record the backward: it must be written with differentiable torch ops on tensors that are inputs or outputs, saved via
#    `save_for_backward`. Intermediates must be returned as extra outputs, or saved in `setup_context`. Then `create_graph=True` builds a graph of
#    the backward, which `gradgradcheck` tests (Double Backward tutorial). `@once_differentiable` runs backward under no-grad and makes any attempt
#    to differentiate its result raise. Use it when the backward is not differentiable (e.g. calls NumPy or an in-place kernel).
# 6. In the forward use the non-differentiable op; in the backward pretend it was the identity, optionally clipped to where the input is in range.
#    `x + (f(x) - x).detach()` has value `f(x)` and gradient 1 because the detached part is a constant. Used in quantization-aware training,
#    binary/ternary networks, VQ-VAE codebooks, and hard Gumbel-softmax.
# 7. `where(c, a, b)` backpropagates `g·c` into `a` and `g·(1-c)` into `b`. Both branches are still differentiated. If the unselected branch has
#    an infinite local derivative, `0 · inf = NaN` poisons the gradient. Fixes: make the unselected branch's *inputs* safe first
#    (e.g. `torch.where(x>=0, x, 0)` before the risky op), or write a custom Function with an explicit stable backward (as in section 2).
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
