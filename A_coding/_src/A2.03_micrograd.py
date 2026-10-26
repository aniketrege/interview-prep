# %% [markdown]
# # A2.3 · micrograd: a scalar autograd engine from scratch
#
# **ID:** A2.3 · **Target:** 40 min · **Plan week:** 1-3 (first pass week 1, timed repeats in weeks 2-3) · **Part:** A
#
# **The drill.** Write a `Value` class that records the computation graph as you compute with Python floats and runs
# reverse-mode autodiff with `backward()`. Support `+`, `*`, `**` (constant exponent), negation, subtraction,
# division, `relu`, `tanh`, `exp` and `log`. Then build a tiny MLP on top of it and train it on two moons.
# Guide A2 "must include": "write a micrograd-style scalar engine".
#
# **What the tests check.** The forward values and gradients are compared with PyTorch (float64) on hand-written
# expressions (micrograd's own tests), on **200 random expression graphs** in which nodes are reused many times,
# and on the MLP's gradients for every parameter. Two bugs are targeted specifically: assigning `grad = ...`
# instead of accumulating `grad += ...` (wrong whenever a node feeds two consumers, e.g. `a + a`), and running the
# local backward functions in the wrong order (wrong without a topological sort).
#
# **Rules.** No torch inside `Value`. Write the whole class in one go (as in an interview), then run the test cells in order.
#
# ## Sources
# - A. Karpathy, *micrograd*, https://github.com/karpathy/micrograd, commit `7bc720e`, files `micrograd/engine.py`
#   (the `Value` class and the topological-sort `backward`), `micrograd/nn.py` (`Neuron`/`Layer`/`MLP`) and
#   `test/test_engine.py` (the two sanity expressions below), MIT license. The `tanh`/`exp`/`log` ops and the
#   random-graph tests are additions. The training loop follows micrograd's `demo.ipynb` (same data, model, loss and schedule).
# - A. Karpathy, *Neural Networks: Zero to Hero*, Lecture 1 "The spelled-out intro to neural networks and
#   backpropagation: building micrograd": https://www.youtube.com/watch?v=VMj-3S1tku0 · course page
#   https://karpathy.ai/zerotohero.html · notebooks https://github.com/karpathy/nn-zero-to-hero (MIT).
# - scikit-learn `make_moons`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html

# %%
import math
import random
import time

import numpy as np
import torch
from sklearn.datasets import make_moons

from prep import Session, check, check_close, seed_everything

s = Session("A2.3", target_min=40)
seed_everything(0)

# %% [markdown]
# ## 1. The `Value` class
# Each op creates `out = Value(result, children=(self, other), op)` and sets `out._backward`, a closure that adds
# **this op's** local contribution to the children's grads: `child.grad += (∂out/∂child) * out.grad`.
# `backward()` builds a topological order of the graph (every node after all of its inputs) with a DFS, sets
# `self.grad = 1`, and calls `_backward` on the nodes in **reverse** topological order. By the time a node's
# `_backward` runs, its own `grad` is complete.
#
# | op | local derivative |
# | --- | --- |
# | $a+b$ | 1, 1 |
# | $a\cdot b$ | $b$, $a$ |
# | $a^k$ ($k$ constant) | $k a^{k-1}$ |
# | relu | $\mathbb 1[a>0]$ |
# | $t=\tanh a$ | $1-t^2$ |
# | $e^a$ | $e^a$ (the output) |
# | $\log a$ | $1/a$ |
#
# Derived ops need no backward of their own: `-a = a * -1`, `a - b = a + (-b)`, `a / b = a * b**-1`; the `r`-versions
# (`2 * a`, `1 - a`, `1 / a`) handle a Python number on the left.

# %%
class Value:
    """A scalar that remembers how it was computed. Adapted from karpathy/micrograd@7bc720e engine.py (MIT)."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        ### BEGIN SOLUTION
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
        ### END SOLUTION

    def __mul__(self, other):
        ### BEGIN SOLUTION
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
        ### END SOLUTION

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        ### BEGIN SOLUTION
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out
        ### END SOLUTION

    def relu(self):
        ### BEGIN SOLUTION
        out = Value(max(self.data, 0.0), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
        ### END SOLUTION

    def tanh(self):
        ### BEGIN SOLUTION
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out
        ### END SOLUTION

    def exp(self):
        ### BEGIN SOLUTION
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
        ### END SOLUTION

    def log(self):
        ### BEGIN SOLUTION
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out
        ### END SOLUTION

    def backward(self):
        ### BEGIN SOLUTION
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
        ### END SOLUTION

    # derived ops: one line each, no new _backward
    ### BEGIN SOLUTION
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1
    # TEMPLATE: def __neg__(self): raise NotImplementedError          # -self
    # TEMPLATE: def __radd__(self, other): raise NotImplementedError  # other + self
    # TEMPLATE: def __sub__(self, other): raise NotImplementedError   # self - other
    # TEMPLATE: def __rsub__(self, other): raise NotImplementedError  # other - self
    # TEMPLATE: def __rmul__(self, other): raise NotImplementedError  # other * self
    # TEMPLATE: def __truediv__(self, other): raise NotImplementedError   # self / other
    # TEMPLATE: def __rtruediv__(self, other): raise NotImplementedError  # other / self
    ### END SOLUTION

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

# %% [markdown]
# ## 2. Tests: `+`, `*`, reuse and order
# `a + a` has one node used twice. Its gradient is 2, which you only get if each use **adds** to `a.grad`.
# The "diamond" `d = (a*b) + (a*b)**2` with a shared intermediate `c = a*b` is the smallest graph where running
# `_backward` in DFS order (instead of reverse topological order) propagates `c.grad` before it is complete.

# %%
a = Value(3.0)
b = a + a
b.backward()
check("a + a: grad 2 (accumulate with +=)", a.grad == 2.0, f"got {a.grad}")
a = Value(3.0)
(a * a).backward()
check("a * a: grad 2a", a.grad == 6.0, f"got {a.grad}")
a, b = Value(2.0), Value(-3.0)
c = a * b
d = c + c * c
d.backward()
check("diamond: dd/da = b(1 + 2ab)", abs(a.grad - (-3.0) * (1 + 2 * 2.0 * -3.0)) < 1e-12, f"got {a.grad}")
check("diamond: dd/db = a(1 + 2ab)", abs(b.grad - 2.0 * (1 + 2 * 2.0 * -3.0)) < 1e-12, f"got {b.grad}")

# %% [markdown]
# ## 3. Tests: micrograd's own sanity checks against PyTorch
# Adapted from micrograd `test/test_engine.py` (`test_sanity_check`, `test_more_ops`). The same Python code runs on
# `Value`s and on float64 torch tensors, because both support the same operators and `.relu()`.

# %%
def sanity(x):
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    return h + q + q * x


def more_ops(a, b):
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    return g + 10.0 / f


def tleaf(v):
    return torch.tensor(v, dtype=torch.float64, requires_grad=True)


x_mg, x_pt = Value(-4.0), tleaf(-4.0)
y_mg, y_pt = sanity(x_mg), sanity(x_pt)
y_mg.backward(); y_pt.backward()
check_close("sanity forward", y_mg.data, y_pt.item(), atol=1e-9)
check_close("sanity grad", x_mg.grad, x_pt.grad.item(), atol=1e-9)
a_mg, b_mg, a_pt, b_pt = Value(-4.0), Value(2.0), tleaf(-4.0), tleaf(2.0)
g_mg, g_pt = more_ops(a_mg, b_mg), more_ops(a_pt, b_pt)
g_mg.backward(); g_pt.backward()
check_close("more_ops forward", g_mg.data, g_pt.item(), atol=1e-6)
check_close("more_ops grads", [a_mg.grad, b_mg.grad], [a_pt.grad.item(), b_pt.grad.item()], atol=1e-6)

# %% [markdown]
# ## 4. Tests: `tanh`, `exp`, `log`, division, and 200 random expression graphs
# `random_graph` draws a random DAG over 3 leaves. Each new node picks its operands from **all** earlier nodes, so
# nodes get reused by several consumers. The structure depends only on the seed, so the same graph is built from
# `Value`s and from torch tensors. Operations are kept in safe domains (e.g. `log(x*x + 1)`, `x / (y*y + 1)`).

# %%
def unary_ops(x):
    return [x.tanh(), (x * 0.5).exp(), (x * x + 1).log(), x.relu(), 1 / (x * x + 1), (x * x + 2) ** 0.5]


xs = [0.7, -1.3, 2.1]
for i, name in enumerate(["tanh", "exp", "log", "relu", "rtruediv", "pow 0.5"]):
    for v in xs:
        vm, vt = Value(v), tleaf(v)
        om, ot = unary_ops(vm)[i], unary_ops(vt)[i]
        om.backward(); ot.backward()
        assert abs(om.data - ot.item()) < 1e-9 and abs(vm.grad - vt.grad.item()) < 1e-9, \
            f"{name} at x={v}: value {om.data} vs {ot.item()}, grad {vm.grad} vs {vt.grad.item()}"
    check(f"{name} value + grad", True)


def random_graph(leaves, seed, n_ops=30):
    rng = random.Random(seed)
    pool = list(leaves)
    for _ in range(n_ops):
        x, y = rng.choice(pool), rng.choice(pool)   # with replacement: x and y may be the same node
        op = rng.randrange(9)
        if op == 0:
            z = x + y
        elif op == 1:
            z = (x * y).tanh()
        elif op == 2:
            z = x - y * 0.5
        elif op == 3:
            z = x / (y * y + 1)
        elif op == 4:
            z = (x * 0.5).exp()
        elif op == 5:
            z = (x * x + 1).log()
        elif op == 6:
            z = x.relu() + 0.1
        elif op == 7:
            z = (x * x + 1) ** -0.5 * 3
        else:
            z = 2 - x * y.tanh()
        pool.append(z)
    out = pool[-1]
    for z in pool[-5:-1]:
        out = out + z
    return out


worst = 0.0
for seed in range(200):
    vals = [random.Random(1000 + seed).uniform(-2, 2) for _ in range(3)]
    lm, lt = [Value(v) for v in vals], [tleaf(v) for v in vals]
    om, ot = random_graph(lm, seed), random_graph(lt, seed)
    om.backward(); ot.backward()
    gm, gt = np.array([l.grad for l in lm]), np.array([0.0 if l.grad is None else l.grad.item() for l in lt])  # unused leaf: torch leaves grad None
    scale = max(1.0, np.abs(gt).max())
    assert abs(om.data - ot.item()) <= 1e-9 * max(1.0, abs(ot.item())), f"seed {seed}: forward {om.data} vs {ot.item()}"
    assert np.abs(gm - gt).max() <= 1e-8 * scale, f"seed {seed}: grads {gm} vs torch {gt}"
    worst = max(worst, np.abs(gm - gt).max() / scale)
check("200 random graphs match torch", True)
print(f"worst relative grad error: {worst:.2e}")

# %% [markdown]
# ## 5. A tiny MLP on top of `Value`
# From micrograd `nn.py`: a `Neuron` computes `act = sum(w_i x_i) + b`, followed by ReLU unless it is in the last layer.
# Fill in `Neuron.__call__`. Then the gradients of every parameter are compared with the same network in torch.

# %%
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        ### BEGIN SOLUTION
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
        ### END SOLUTION

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout, **kw):
        self.neurons = [Neuron(nin, **kw) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def hinge_loss(model, X, y, alpha=1e-4):
    """SVM max-margin loss + L2, as in micrograd's demo.ipynb. y in {-1, +1}. Returns (loss Value, accuracy)."""
    y = [float(v) for v in y]  # plain floats: numpy scalars on the left of a Value would hijack the operator
    scores = [model([Value(v) for v in row]) for row in X]
    data_loss = sum((1 + -yi * si).relu() for yi, si in zip(y, scores)) * (1.0 / len(y))
    reg = alpha * sum(p * p for p in model.parameters())
    acc = np.mean([(yi > 0) == (si.data > 0) for yi, si in zip(y, scores)])
    return data_loss + reg, acc


random.seed(1337)
X, y = make_moons(n_samples=100, noise=0.1, random_state=1337)
y = 2 * y - 1
model = MLP(2, [16, 16, 1])
print("number of parameters:", len(model.parameters()))

# torch twin with identical weights
Ws = [torch.tensor([[p.data for p in n.w] for n in L.neurons], dtype=torch.float64, requires_grad=True) for L in model.layers]
bs = [torch.tensor([n.b.data for n in L.neurons], dtype=torch.float64, requires_grad=True) for L in model.layers]
Xt, yt = torch.tensor(X[:20]), torch.tensor(y[:20], dtype=torch.float64)
h = Xt
for i, (W, b_) in enumerate(zip(Ws, bs)):
    h = h @ W.T + b_
    h = h.relu() if i < len(Ws) - 1 else h
loss_pt = (1 - yt * h.squeeze(1)).relu().mean() + 1e-4 * sum((W ** 2).sum() + (b_ ** 2).sum() for W, b_ in zip(Ws, bs))
loss_pt.backward()
loss_mg, _ = hinge_loss(model, X[:20], y[:20])
model.zero_grad()
loss_mg.backward()
check_close("MLP loss vs torch", loss_mg.data, loss_pt.item(), atol=1e-10)
g_mg = [p.grad for p in model.parameters()]
g_pt = [g for W, b_ in zip(Ws, bs) for row_w, row_b in zip(W.grad, b_.grad) for g in [*row_w.tolist(), row_b.item()]]
check_close("MLP grads vs torch (all 337 params)", g_mg, g_pt, atol=1e-10)

# %% [markdown]
# ## 6. Train on two moons
# Full-batch SGD with the learning rate decaying linearly from 1.0 (micrograd's demo, shortened from 100 to 60 steps
# because pure Python is slow: about 1 s per step). Remember
# `zero_grad()` before every `backward()`: the engine accumulates. Target: training accuracy ≥ 95%.

# %%
t0 = time.time()
steps = 60
for k in range(steps):
    ### BEGIN SOLUTION
    loss, acc = hinge_loss(model, X, y)
    model.zero_grad()
    loss.backward()
    lr = 1.0 - 0.9 * k / steps
    for p in model.parameters():
        p.data -= lr * p.grad
    ### END SOLUTION
    if k % 10 == 0 or k == steps - 1:
        print(f"step {k:3d}  loss {loss.data:.4f}  acc {acc:.2f}")
print(f"{time.time() - t0:.0f} s for {steps} full-batch steps in pure Python")
_, acc = hinge_loss(model, X, y)
check("moons train accuracy >= 0.95", acc >= 0.95, f"acc={acc}")

# %% [markdown]
# ## Say it out loud
# 1. Why must `_backward` use `+=`? Give the smallest graph where `=` is wrong.
# 2. Why a topological sort? What goes wrong if you recurse into children right after calling a node's `_backward`?
# 3. What do you have to remember between two `backward()` calls, and why does PyTorch behave the same way (`.grad` accumulates)?
# 4. Why does micrograd store `out.data` for `exp` and `t` for `tanh` in the closure? What is the memory cost of reverse mode in general?
# 5. Reverse mode vs forward mode: when is each cheaper? Why does deep learning use reverse mode?
# 6. How does this scalar engine differ from PyTorch's tensor autograd (granularity, saved tensors, graph freeing, `retain_graph`)?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. A node's gradient is the **sum** over all paths through its consumers (multivariate chain rule). If two consumers each *assign*,
#    the second overwrites the first. Smallest case: `b = a + a`. With `=` you get `a.grad = 1` instead of 2. Same for `a * a` (gives a, not 2a).
# 2. A node's `_backward` must run only after **all** of its consumers have pushed their contributions into its `grad`. Reverse topological
#    order guarantees this. Naive DFS from the output can visit a shared intermediate (the diamond `c` above) through its first consumer and
#    propagate a partial `c.grad` further down. The DFS-with-postorder trick builds the topological order in O(V+E).
# 3. Gradients accumulate across calls, so call `zero_grad()` before each `backward()`. PyTorch does the same on purpose: it makes gradient
#    accumulation over micro-batches free, and you must `optimizer.zero_grad()` (or `set_to_none`) between steps.
# 4. The backward of exp needs $e^a$, which is the output, so saving it avoids recomputation. tanh needs $1-t^2$. Reverse mode must keep
#    whatever each local derivative needs (activations) until the backward pass reaches that node, so memory grows with depth times width.
#    That is why activation checkpointing (recompute instead of store) exists.
# 5. For $f:\mathbb R^n\to\mathbb R^m$, forward mode costs about one pass per *input* direction (JVP), and reverse mode about one pass per
#    *output* (VJP). Losses are scalar ($m=1$) with millions of parameters ($n$ huge), so reverse mode gets the whole gradient in about 2-3x
#    the cost of the forward pass.
# 6. PyTorch's nodes are tensor ops with vectorized backward formulas (VJPs, never full Jacobians). It saves only the tensors each op needs
#    (`save_for_backward`). It frees the graph after `backward()` unless `retain_graph=True`. It runs the backward in a C++ engine, also in
#    dependency order, one thread per device. micrograd builds one Python object per scalar, which is why the 337-parameter MLP above takes seconds.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
