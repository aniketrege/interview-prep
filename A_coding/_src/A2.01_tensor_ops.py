# %% [markdown]
# # A2.1 · Tensor-op puzzles (broadcasting, einsum, gather/scatter, strides)
#
# **ID:** A2.1 · **Target:** 25 min · **Plan week:** 1 · **Part:** A
#
# **The drill.** 23 short puzzles. Each one is a function you write in **one line** (sections D and E allow up
# to 3), with **no Python loops or comprehensions**. Under every puzzle is a loop-based reference (the *spec*)
# and a randomized test that compares your function against it on 30 random inputs of random sizes. The checker
# also parses your function and rejects loops and the banned built-in for that puzzle.
#
# Guide A2 "must include": broadcasting, `einsum`, `gather`/`scatter`, `view` vs `reshape` vs `contiguous`,
# masking, `torch.where`, indexing tricks. Sections:
# A. Broadcasting from first principles (Tensor-Puzzles rules) · B. `einsum`, `gather`, `scatter` ·
# C. Masks and `where` · D. Memory layout: strides, `view` vs `reshape` · E. Stride tricks and cumsum tricks.
#
# **Rules.** No docs. Section A follows Sasha Rush's Tensor-Puzzles: only `arange`, broadcasting, arithmetic,
# comparisons, indexing, `@`, `torch.where`, `.sum`, `torch.ones/zeros`, and your own earlier puzzles. Later
# sections may use any torch op except the one named in the puzzle's `banned=`.
#
# ## Sources
# - S. Rush, *Tensor Puzzles* (puzzles A1-A7 are adapted from #3, #6, #7, #8, #14, #15, #16, #21 and its
#   "one line, no loops, broadcasting only" rules): https://github.com/srush/Tensor-Puzzles, commit `ad5069f`,
#   `README.md` / `Tensor Puzzlers.ipynb`, MIT license.
# - N. Rougier et al., *100 NumPy exercises* (puzzles adapted from #64 repeated-index add, #69 diagonal of a
#   product, #75 moving average, #76/#84 `as_strided` windows, #87 block sum, #52 pairwise distances):
#   https://github.com/rougier/numpy-100, commit `8ca28aa`, `100_Numpy_exercises_with_solutions.md`, MIT license.
# - PyTorch docs: broadcasting semantics https://docs.pytorch.org/docs/stable/notes/broadcasting.html ·
#   tensor views https://docs.pytorch.org/docs/stable/tensor_view.html ·
#   `torch.as_strided` (overlapping views, in-place writes undefined) https://docs.pytorch.org/docs/stable/generated/torch.as_strided.html ·
#   `torch.gather` https://docs.pytorch.org/docs/stable/generated/torch.gather.html ·
#   `Tensor.scatter_add_` https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html ·
#   `torch.einsum` (opt_einsum contraction order) https://docs.pytorch.org/docs/stable/generated/torch.einsum.html ·
#   `torch.use_deterministic_algorithms` (list of nondeterministic CUDA ops such as `scatter_add_`, `index_add`)
#   https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html

# %%
import ast
import inspect
import math
import textwrap

import torch

from prep import Session, check, seed_everything

s = Session("A2.1", target_min=25)
seed_everything(0)


def arange(n):
    "Tensor-Puzzles' only constructor: use it to replace a for-loop."
    return torch.arange(n)


def _lint(fn, banned=(), allow_loops=False):
    """Reject Python loops/comprehensions and banned names (attributes, calls, or the @ operator)."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError):  # source not available (e.g. defined in exec): skip the lint
        return
    for node in ast.walk(tree):
        if not allow_loops and isinstance(node, (ast.For, ast.While, ast.comprehension)):
            raise AssertionError(f"{fn.__name__}: no Python loops or comprehensions allowed")
        name = getattr(node, "attr", None) or getattr(node, "id", None)
        if name in banned:
            raise AssertionError(f"{fn.__name__}: `{name}` is banned in this puzzle")
        if "@" in banned and isinstance(node, ast.MatMult):
            raise AssertionError(f"{fn.__name__}: `@` is banned in this puzzle")


def check_puzzle(name, fn, spec, gen, trials=30, banned=(), allow_loops=False, atol=1e-5):
    """Compare fn(*args) to the loop reference spec(*args) on `trials` random inputs from gen()."""
    _lint(fn, banned, allow_loops)
    for t in range(trials):
        args = gen()
        got, exp = fn(*args), spec(*args)
        same = (isinstance(got, torch.Tensor) and got.shape == exp.shape
                and torch.allclose(got.double(), exp.double(), atol=atol, rtol=1e-4))
        if not same:
            shown = [a if not isinstance(a, torch.Tensor) or a.numel() < 40 else a.shape for a in args]
            raise AssertionError(f"{name}: trial {t} failed\n  args={shown}\n  got={got}\n  expected={exp}")
    check(name, True)


def rint(lo, hi):
    return int(torch.randint(lo, hi, ()).item())

# %% [markdown]
# ## A. Broadcasting from first principles (Tensor-Puzzles rules)
# Broadcasting: align shapes **from the right**; each pair of dims must be equal or one of them 1; size-1 dims
# are virtually repeated (stride 0, no copy). `a[:, None]` inserts a size-1 axis. Most puzzles here are
# "compare an `arange` against another `arange` (or an index vector) and use the boolean grid".
#
# **A1 · outer** (TP #3): `out[i, j] = a[i] * b[j]`.

# %%
def outer(a, b):
    ### BEGIN SOLUTION
    return a[:, None] * b[None, :]
    ### END SOLUTION


def outer_spec(a, b):
    out = torch.zeros(len(a), len(b))
    for i in range(len(a)):
        for j in range(len(b)):
            out[i, j] = a[i] * b[j]
    return out


check_puzzle("A1 outer", outer, outer_spec, lambda: (torch.randn(rint(1, 6)), torch.randn(rint(1, 6))), banned=("outer", "einsum"))

# %% [markdown]
# **A2 · triu** (TP #6): `(n, n)` int64 matrix, 1 where `i <= j` (the diagonal included), else 0.

# %%
def triu(n):
    ### BEGIN SOLUTION
    return (arange(n)[:, None] <= arange(n)[None, :]).long()
    ### END SOLUTION


def triu_spec(n):
    out = torch.zeros(n, n, dtype=torch.long)
    for i in range(n):
        for j in range(n):
            out[i, j] = int(i <= j)
    return out


check_puzzle("A2 triu", triu, triu_spec, lambda: (rint(1, 8),), banned=("triu", "tril", "ones"))

# %% [markdown]
# **A3 · cumsum** (TP #7): running sum of an int vector, using `triu` and `@` (no `cumsum`).
# Think: which entries of `a` contribute to `out[j]`?

# %%
def cumsum(a):
    ### BEGIN SOLUTION
    return a @ triu(len(a))
    ### END SOLUTION


def cumsum_spec(a):
    out, total = torch.zeros_like(a), 0
    for i in range(len(a)):
        total += a[i]
        out[i] = total
    return out


check_puzzle("A3 cumsum", cumsum, cumsum_spec, lambda: (torch.randint(-5, 6, (rint(1, 9),)),), banned=("cumsum",))

# %% [markdown]
# **A4 · diff** (TP #8): `out[0] = a[0]`, `out[i] = a[i] - a[i-1]`. Use `torch.where` to handle position 0.

# %%
def diff(a):
    ### BEGIN SOLUTION
    return a - torch.where(arange(len(a)) > 0, a[arange(len(a)) - 1], 0)
    ### END SOLUTION


def diff_spec(a):
    out = a.clone()
    for i in range(1, len(a)):
        out[i] = a[i] - a[i - 1]
    return out


check_puzzle("A4 diff", diff, diff_spec, lambda: (torch.randint(-5, 6, (rint(1, 9),)),), banned=("diff",))

# %% [markdown]
# **A5 · sequence_mask** (TP #14): keep `values[i, j]` where `j < length[i]`, else 0. This is a padding mask.

# %%
def sequence_mask(values, length):
    ### BEGIN SOLUTION
    return torch.where(arange(values.shape[1])[None, :] < length[:, None], values, 0)
    ### END SOLUTION


def sequence_mask_spec(values, length):
    out = torch.zeros_like(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if j < length[i]:
                out[i, j] = values[i, j]
    return out


def _gen_seq():
    n, t = rint(1, 5), rint(1, 7)
    return torch.randn(n, t), torch.randint(0, t + 1, (n,))


check_puzzle("A5 sequence_mask", sequence_mask, sequence_mask_spec, _gen_seq)

# %% [markdown]
# **A6 · bincount and scatter_add** (TP #15, #16): build the one-hot grid `link[:, None] == arange(j)` and
# contract it with `@`. Note the memory cost: the grid is `(i, j)`, which is why real code uses `scatter_add_`.

# %%
def bincount(a, j):
    ### BEGIN SOLUTION
    return torch.ones_like(a) @ (a[:, None] == arange(j)).long()
    ### END SOLUTION


def scatter_add(values, link, j):
    ### BEGIN SOLUTION
    return values @ (link[:, None] == arange(j)).to(values.dtype)
    ### END SOLUTION


def bincount_spec(a, j):
    out = torch.zeros(j, dtype=torch.long)
    for i in range(len(a)):
        out[a[i]] += 1
    return out


def scatter_add_spec(values, link, j):
    out = torch.zeros(j)
    for i in range(len(values)):
        out[link[i]] += values[i]
    return out


def _gen_bins():
    j = rint(1, 7)
    return torch.randint(0, j, (rint(1, 12),)), j


def _gen_scatter():
    j, i = rint(1, 7), rint(1, 12)
    return torch.randn(i), torch.randint(0, j, (i,)), j


check_puzzle("A6 bincount", bincount, bincount_spec, _gen_bins, banned=("bincount", "scatter_add", "scatter_add_", "index_add_"))
check_puzzle("A6 scatter_add", scatter_add, scatter_add_spec, _gen_scatter, banned=("bincount", "scatter_add", "scatter_add_", "index_add_"))

# %% [markdown]
# **A7 · bucketize** (TP #21): `out[i]` = number of (sorted) boundaries `<= v[i]`, i.e. `torch.bucketize(v, b, right=True)`.

# %%
def bucketize(v, boundaries):
    ### BEGIN SOLUTION
    return (v[:, None] >= boundaries[None, :]).long().sum(1)
    ### END SOLUTION


def bucketize_spec(v, boundaries):
    out = torch.zeros(len(v), dtype=torch.long)
    for i in range(len(v)):
        for j in range(len(boundaries)):
            if v[i] >= boundaries[j]:
                out[i] = j + 1
    return out


def _gen_bucket():
    b = torch.randint(0, 4, (rint(1, 6),)).cumsum(0)  # sorted, with repeats
    return torch.randint(-1, int(b.max()) + 2, (rint(1, 10),)), b


check_puzzle("A7 bucketize", bucketize, bucketize_spec, _gen_bucket, banned=("bucketize", "searchsorted"))

# %% [markdown]
# ## B. `einsum`, `gather`, `scatter`
# `einsum` letters: repeated across inputs and absent from the output means *summed*; in the output means
# *kept*. `x.gather(dim, idx)`: `out[i][j] = x[i][idx[i][j]]` for `dim=1`; `idx` has the **output's** shape
# (same number of dims as `x`). `scatter_add_` is its adjoint: `out[i][idx[i][j]] += src[i][j]`.
#
# **B1 · three einsums.** (a) batched outer product `(B, I), (B, J) -> (B, I, J)`; (b) `diag(A @ B)` for square
# `A, B` **without forming `A @ B`** (numpy-100 #69); (c) attention scores `(B, H, T, D), (B, H, S, D) -> (B, H, T, S)`
# scaled by $1/\sqrt{D}$.

# %%
def batched_outer(a, b):
    ### BEGIN SOLUTION
    return torch.einsum("bi,bj->bij", a, b)
    ### END SOLUTION


def diag_of_product(A, B):
    ### BEGIN SOLUTION
    return torch.einsum("ij,ji->i", A, B)
    ### END SOLUTION


def attn_scores(q, k):
    ### BEGIN SOLUTION
    return torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(q.shape[-1])
    ### END SOLUTION


def batched_outer_spec(a, b):
    return torch.stack([outer_spec(a[i], b[i]) for i in range(len(a))])


def diag_of_product_spec(A, B):
    return torch.tensor([sum(A[i, k] * B[k, i] for k in range(A.shape[1])) for i in range(A.shape[0])])


def attn_scores_spec(q, k):
    B, H, T, D = q.shape
    out = torch.zeros(B, H, T, k.shape[2])
    for b in range(B):
        for h in range(H):
            for t in range(T):
                for s_ in range(k.shape[2]):
                    out[b, h, t, s_] = (q[b, h, t] * k[b, h, s_]).sum() / math.sqrt(D)
    return out


def _gen_bo():
    B = rint(1, 4)
    return torch.randn(B, rint(1, 5)), torch.randn(B, rint(1, 5))


def _gen_sq():
    n = rint(1, 6)
    return torch.randn(n, n), torch.randn(n, n)


def _gen_attn():
    B, H, T, S, D = rint(1, 3), rint(1, 3), rint(1, 4), rint(1, 4), rint(1, 5)
    return torch.randn(B, H, T, D), torch.randn(B, H, S, D)


check_puzzle("B1a batched_outer", batched_outer, batched_outer_spec, _gen_bo)
check_puzzle("B1b diag_of_product", diag_of_product, diag_of_product_spec, _gen_sq, banned=("@", "matmul", "mm", "diag", "diagonal"))
check_puzzle("B1c attn_scores", attn_scores, attn_scores_spec, _gen_attn, trials=10)

# %% [markdown]
# **B2 · pick the label log-prob** with `gather`: `logp (N, C), y (N,) -> (N,)` with `out[n] = logp[n, y[n]]`.
# (This is the core of every NLL / policy-gradient `log π(a|s)` line.)

# %%
def pick(logp, y):
    ### BEGIN SOLUTION
    return logp.gather(1, y[:, None]).squeeze(1)
    ### END SOLUTION


def pick_spec(logp, y):
    return torch.stack([logp[n, y[n]] for n in range(len(y))])


def _gen_pick():
    N, C = rint(1, 7), rint(1, 6)
    return torch.randn(N, C), torch.randint(0, C, (N,))


check_puzzle("B2 pick (gather)", pick, pick_spec, _gen_pick)

# %% [markdown]
# **B3 · per-row top-k gather**: `scores (B, V)`, `feats (B, V, D)`, `k` → `(B, k, D)`: the feature vectors of the
# `k` highest-scoring items of each row, in descending score order (e.g. retrieving top-k memory slots).
# Hint: `gather` needs the index expanded to the output shape.

# %%
def topk_gather(scores, feats, k):
    ### BEGIN SOLUTION
    return feats.gather(1, scores.topk(k, dim=1).indices[..., None].expand(-1, -1, feats.shape[-1]))
    ### END SOLUTION


def topk_gather_spec(scores, feats, k):
    out = torch.zeros(scores.shape[0], k, feats.shape[-1])
    for b in range(scores.shape[0]):
        order = sorted(range(scores.shape[1]), key=lambda v: -scores[b, v].item())
        for r in range(k):
            out[b, r] = feats[b, order[r]]
    return out


def _gen_topk():
    B, V, D = rint(1, 4), rint(2, 7), rint(1, 4)
    return torch.randn(B, V), torch.randn(B, V, D), rint(1, V + 1)


check_puzzle("B3 topk_gather", topk_gather, topk_gather_spec, _gen_topk)

# %% [markdown]
# **B4 · one-hot with `scatter_`**: `y (N,)`, `C` → float `(N, C)`. (No `F.one_hot`, no comparison grid.)

# %%
def one_hot(y, C):
    ### BEGIN SOLUTION
    return torch.zeros(len(y), C).scatter_(1, y[:, None], 1.0)
    ### END SOLUTION


def one_hot_spec(y, C):
    out = torch.zeros(len(y), C)
    for n in range(len(y)):
        out[n, y[n]] = 1.0
    return out


check_puzzle("B4 one_hot (scatter_)", one_hot, one_hot_spec, lambda: (lambda C: (torch.randint(0, C, (rint(1, 8),)), C))(rint(1, 6)),
             banned=("one_hot", "eye"))

# %% [markdown]
# **B5 · segment mean with `scatter_add_` / `index_add_`** (up to 3 lines): `values (N, D)`, segment ids
# `seg (N,)` in `[0, S)` → `(S, D)` per-segment means, with **empty segments = 0** (not NaN). This is the
# graph-pooling / group-by-mean primitive (numpy-100 #68 does the 1-D version with `bincount(weights=)`).

# %%
def segment_mean(values, seg, S):
    ### BEGIN SOLUTION
    sums = torch.zeros(S, values.shape[1]).scatter_add_(0, seg[:, None].expand_as(values), values)
    counts = torch.zeros(S).index_add_(0, seg, torch.ones(len(seg)))
    return sums / counts.clamp(min=1)[:, None]
    ### END SOLUTION


def segment_mean_spec(values, seg, S):
    out = torch.zeros(S, values.shape[1])
    for k in range(S):
        rows = [values[n] for n in range(len(seg)) if seg[n] == k]
        if rows:
            out[k] = torch.stack(rows).mean(0)
    return out


def _gen_seg():
    S, N = rint(2, 8), rint(1, 10)
    return torch.randn(N, rint(1, 4)), torch.randint(0, S, (N,)), S  # some segments stay empty


check_puzzle("B5 segment_mean", segment_mean, segment_mean_spec, _gen_seg)

# %% [markdown]
# **Gotcha (numpy-100 #64).** Advanced-index assignment with repeated indices does **not** accumulate: `x[idx] += 1`
# is `x[idx] = x[idx] + 1`, and duplicates write the same value. Use `index_add_` / `index_put_(accumulate=True)`.

# %%
x = torch.zeros(4)
idx = torch.tensor([0, 0, 0, 2])
x[idx] += 1
print("x[idx] += 1          ->", x.tolist())   # [1, 0, 1, 0], not [3, 0, 1, 0]
print("index_add_           ->", torch.zeros(4).index_add_(0, idx, torch.ones(4)).tolist())
print("index_put_ accumulate->", torch.zeros(4).index_put_((idx,), torch.ones(4), accumulate=True).tolist())

# %% [markdown]
# ## C. Masks and `torch.where`
# **C1 · causal softmax**: `scores (..., T, T)`; position `i` may attend to `j <= i`. Fill the future with `-inf`
# and softmax over the last dim (you may use `torch.softmax`, `masked_fill`, `torch.ones(...).triu(...)` here).

# %%
def causal_softmax(scores):
    ### BEGIN SOLUTION
    T = scores.shape[-1]
    future = torch.ones(T, T, dtype=torch.bool).triu(1)
    return scores.masked_fill(future, float("-inf")).softmax(-1)
    ### END SOLUTION


def causal_softmax_spec(scores):
    out = torch.zeros_like(scores)
    flat, oflat = scores.reshape(-1, *scores.shape[-2:]), out.view(-1, *scores.shape[-2:])
    for b in range(flat.shape[0]):
        for i in range(flat.shape[1]):
            e = torch.exp(flat[b, i, : i + 1] - flat[b, i, : i + 1].max())
            oflat[b, i, : i + 1] = e / e.sum()
    return out


check_puzzle("C1 causal_softmax", causal_softmax, causal_softmax_spec,
             lambda: (lambda T: (torch.randn(rint(1, 3), rint(1, 3), T, T) * 3,))(rint(1, 6)))

# %% [markdown]
# **C2 · masked mean pooling**: `x (B, T, D)`, `lengths (B,)` with `1 <= lengths[b] <= T` → `(B, D)`, the mean over the
# first `lengths[b]` tokens only (padding excluded).

# %%
def masked_mean(x, lengths):
    ### BEGIN SOLUTION
    m = (torch.arange(x.shape[1])[None, :] < lengths[:, None]).to(x.dtype)  # (B, T)
    return (x * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    ### END SOLUTION


def masked_mean_spec(x, lengths):
    return torch.stack([x[b, : lengths[b]].mean(0) for b in range(x.shape[0])])


def _gen_len():
    B, T = rint(1, 5), rint(1, 7)
    return torch.randn(B, T, rint(1, 4)), torch.randint(1, T + 1, (B,))


check_puzzle("C2 masked_mean", masked_mean, masked_mean_spec, _gen_len)

# %% [markdown]
# **C3 · last valid token** (advanced indexing): `x (B, T, D)`, `lengths (B,)` → `(B, D)` with `out[b] = x[b, lengths[b]-1]`.
# Index two dims at once with two integer tensors.

# %%
def last_token(x, lengths):
    ### BEGIN SOLUTION
    return x[torch.arange(x.shape[0]), lengths - 1]
    ### END SOLUTION


def last_token_spec(x, lengths):
    return torch.stack([x[b, lengths[b] - 1] for b in range(x.shape[0])])


check_puzzle("C3 last_token", last_token, last_token_spec, _gen_len, banned=("gather",))

# %% [markdown]
# ## D. Memory layout: strides, `view` vs `reshape` vs `contiguous`
# A tensor is (storage, offset, shape, **strides**). Element `[i, j]` lives at `offset + i*stride[0] + j*stride[1]`.
# `transpose`/`permute`/`expand`/slicing only change strides (views, no copy). `view` needs the new shape to be
# expressible with strides over the *same* memory; after a transpose it usually is not, so `view` raises.
# `reshape` = `view` when possible, else a copy. `contiguous()` copies into row-major order.

# %%
x = torch.arange(12).view(3, 4)
xt = x.t()
print("x.stride() =", x.stride(), " x.t().stride() =", xt.stride(), " contiguous?", xt.is_contiguous())
try:
    xt.view(12)
except RuntimeError as e:
    print("x.t().view(12) fails:", str(e).split(".")[0])
r = xt.reshape(12)
print("reshape copies here:", r.untyped_storage().data_ptr() != x.untyped_storage().data_ptr(), r.tolist())
print("expand is stride 0  :", torch.tensor([1, 2, 3])[:, None].expand(3, 4).stride())

# %% [markdown]
# **D1 · contiguous strides**: given a shape tuple, return the row-major strides (Python ints; a loop is fine here).

# %%
def contiguous_strides(shape):
    ### BEGIN SOLUTION
    strides, acc = [], 1
    for n in reversed(shape):
        strides.append(acc)
        acc *= n
    return tuple(reversed(strides))
    ### END SOLUTION


for shp in [(3, 4), (2, 3, 5), (7,), (2, 1, 4, 3), (5, 1)]:
    assert contiguous_strides(shp) == torch.empty(shp).stride(), (shp, contiguous_strides(shp))
check("D1 contiguous_strides", True)

# %% [markdown]
# **D2 · split / merge attention heads.** `split_heads`: `(B, T, H*D) -> (B, H, T, D)`, returned as a **view**
# (no copy). `merge_heads`: the inverse `(B, H, T, D) -> (B, T, H*D)`. The input of `merge_heads` is usually
# contiguous in `(B, H, T, D)` order (it is the output of `att @ v`), so after your transpose it is not.

# %%
def split_heads(x, H):
    ### BEGIN SOLUTION
    B, T, HD = x.shape
    return x.view(B, T, H, HD // H).transpose(1, 2)
    ### END SOLUTION


def merge_heads(x):
    ### BEGIN SOLUTION
    B, H, T, D = x.shape
    return x.transpose(1, 2).reshape(B, T, H * D)
    ### END SOLUTION


def split_heads_spec(x, H):
    B, T, HD = x.shape
    D = HD // H
    out = torch.zeros(B, H, T, D)
    for b in range(B):
        for h in range(H):
            for t in range(T):
                out[b, h, t] = x[b, t, h * D:(h + 1) * D]
    return out


def _gen_heads():
    H = rint(1, 4)
    return torch.randn(rint(1, 3), rint(1, 5), H * rint(1, 4)), H


check_puzzle("D2 split_heads", split_heads, split_heads_spec, _gen_heads)
xh = torch.randn(2, 5, 12)
check("D2 split_heads is a view", split_heads(xh, 3).untyped_storage().data_ptr() == xh.untyped_storage().data_ptr(),
      "split_heads copied the data; use view + transpose")
check("D2 merge_heads round trip", torch.equal(merge_heads(split_heads(xh, 3)), xh), "merge(split(x)) != x")
yh = torch.randn(2, 3, 5, 4)  # contiguous (B, H, T, D), like the output of att @ v
check("D2 merge_heads on att @ v layout", torch.equal(merge_heads(yh), yh.permute(0, 2, 1, 3).flatten(2)), "wrong merge order")

# %% [markdown]
# ## E. Stride tricks and cumsum tricks
# **E1 · sliding windows with `as_strided`** (numpy-100 #76): 1-D `x (n,)`, window `w` → `(n-w+1, w)` with
# `out[i] = x[i:i+w]`, as a view. Use `x.stride()` (the input may itself be a strided view). Overlapping views
# share memory: never write into them in place (the docs call the result undefined).

# %%
def sliding_windows(x, w):
    ### BEGIN SOLUTION
    return x.as_strided((x.shape[0] - w + 1, w), (x.stride(0), x.stride(0)))
    ### END SOLUTION


def sliding_windows_spec(x, w):
    return torch.stack([x[i:i + w] for i in range(len(x) - w + 1)])


def _gen_win():
    n = rint(1, 12)
    base = torch.randn(2 * n + 3)
    x = base[1::2][:n] if rint(0, 2) else base[:n]  # sometimes a strided, offset view
    return x, rint(1, n + 1)


check_puzzle("E1 sliding_windows", sliding_windows, sliding_windows_spec, _gen_win, banned=("unfold", "sliding_window_view"))

# %% [markdown]
# **E2 · all k×k patches of a 2-D array** (numpy-100 #84): `x (H, W)` → `(H-k+1, W-k+1, k, k)` view with
# `out[i, j] = x[i:i+k, j:j+k]`. The test feeds transposed and sliced inputs, so do not hard-code strides.

# %%
def patches2d(x, k):
    ### BEGIN SOLUTION
    H, W = x.shape
    return x.as_strided((H - k + 1, W - k + 1, k, k), x.stride() + x.stride())
    ### END SOLUTION


def patches2d_spec(x, k):
    H, W = x.shape
    return torch.stack([torch.stack([x[i:i + k, j:j + k] for j in range(W - k + 1)]) for i in range(H - k + 1)])


def _gen_patch():
    H, W = rint(1, 7), rint(1, 7)
    x = torch.randn(H + 1, W + 2)
    x = [x[:H, :W], x.t()[:W, :H].t()[:H, :W], x[1:, 2:]][rint(0, 3)]
    return x, rint(1, min(x.shape) + 1)


check_puzzle("E2 patches2d", patches2d, patches2d_spec, _gen_patch, banned=("unfold", "sliding_window_view"))

# %% [markdown]
# **E3 · moving average via cumsum** (numpy-100 #75): `a (n,)` float64, window `w` → `(n-w+1,)` means, in O(n)
# (no windows tensor). Prepend a zero to the cumsum.

# %%
def moving_average(a, w):
    ### BEGIN SOLUTION
    c = torch.cat([a.new_zeros(1), a.cumsum(0)])
    return (c[w:] - c[:-w]) / w
    ### END SOLUTION


def moving_average_spec(a, w):
    return torch.stack([a[i:i + w].mean() for i in range(len(a) - w + 1)])


check_puzzle("E3 moving_average", moving_average, moving_average_spec,
             lambda: (lambda n: (torch.randn(n, dtype=torch.float64), rint(1, n + 1)))(rint(1, 12)),
             banned=("unfold", "as_strided", "conv1d", "avg_pool1d"))

# %% [markdown]
# **E4 · block sum via reshape** (numpy-100 #87): `Z (H, W)` with `k | H, k | W` → `(H/k, W/k)` sums of
# non-overlapping k×k blocks. One `reshape` + one `sum`: which axes are the within-block ones?

# %%
def block_sum(Z, k):
    ### BEGIN SOLUTION
    H, W = Z.shape
    return Z.reshape(H // k, k, W // k, k).sum((1, 3))
    ### END SOLUTION


def block_sum_spec(Z, k):
    H, W = Z.shape
    out = torch.zeros(H // k, W // k)
    for i in range(H // k):
        for j in range(W // k):
            out[i, j] = Z[i * k:(i + 1) * k, j * k:(j + 1) * k].sum()
    return out


check_puzzle("E4 block_sum", block_sum, block_sum_spec,
             lambda: (lambda k: (torch.randn(k * rint(1, 4), k * rint(1, 4)), k))(rint(1, 4)),
             banned=("unfold", "avg_pool2d", "as_strided"))

# %% [markdown]
# **E5 · pairwise Euclidean distances** (numpy-100 #52, the kNN / k-means / retrieval workhorse):
# `A (N, D)`, `B (M, D)` → `(N, M)`, via $\|a\|^2 + \|b\|^2 - 2\,a\cdot b$ (one matmul, O(NM) memory instead of
# the O(NMD) broadcast). Rounding can make the squared distance slightly negative: clamp at 0 before `sqrt`.

# %%
def pairwise_dist(A, B):
    ### BEGIN SOLUTION
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
    return d2.clamp(min=0).sqrt()
    ### END SOLUTION


def pairwise_dist_spec(A, B):
    return torch.stack([torch.stack([((A[i] - B[j]) ** 2).sum().sqrt() for j in range(len(B))]) for i in range(len(A))])


check_puzzle("E5 pairwise_dist", pairwise_dist, pairwise_dist_spec,
             lambda: (lambda D: (torch.randn(rint(1, 6), D, dtype=torch.float64), torch.randn(rint(1, 6), D, dtype=torch.float64)))(rint(1, 5)),
             banned=("cdist", "pdist", "norm"))
# float32, far from the origin, with duplicate points: the formula cancels catastrophically, so check it stays finite and >= 0
P = torch.randn(64, 16) + 300.0
Dp = pairwise_dist(P, P)
check("E5 no NaN / negative under cancellation", bool(torch.isfinite(Dp).all() and (Dp >= 0).all()), "clamp before sqrt")
print("float32 error on the zero diagonal at offset 300:", Dp.diagonal().max().item(),
      "| after centering:", pairwise_dist(P - P.mean(0), P - P.mean(0)).diagonal().max().item())

# %% [markdown]
# ## Say it out loud
# 1. State the broadcasting rule. What is the shape of `torch.zeros(3, 1, 5) + torch.zeros(4, 1)`? What does `expand` cost in memory?
# 2. `view` vs `reshape` vs `contiguous()`: when does `view` fail, and why does `reshape` sometimes return a copy and sometimes not?
#    Why can silently getting a copy from `reshape` matter?
# 3. `gather` vs advanced indexing `x[arange(N), y]`: what shapes must the index have in each case? Which one generalizes to
#    "pick k items per row"?
# 4. What does `x[idx] += 1` do with repeated indices? How do you accumulate correctly, and what is the GPU determinism caveat?
# 5. Why is $\|a\|^2+\|b\|^2-2ab$ preferred for pairwise distances, what is its numerical weakness, and how do you mitigate it?
# 6. When is `as_strided` dangerous? What does `F.unfold` give you instead?
# 7. How does `einsum` decide the contraction order with three or more operands?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Align shapes from the right; each pair of sizes must be equal or one of them 1 (missing leading dims count as 1); the result takes
#    the max. `(3,1,5) + (4,1)` → `(3,4,5)`. `expand` sets stride 0 on the broadcast dims: no memory is allocated until something
#    materializes it (e.g. `.contiguous()` or an in-place write, which is an error on expanded tensors).
# 2. `view` never copies; it requires that the new shape can be expressed with strides over the existing memory. After `transpose`/`permute`
#    the merged dims are not adjacent in memory, so `view` raises. `reshape` returns a view if possible and otherwise copies (as in
#    `x.t().reshape(-1)` above). It matters because writes into a copy do not propagate to the original. A hidden copy also costs memory
#    bandwidth inside a hot loop. `contiguous()` returns `self` if already row-major, else a row-major copy.
# 3. Advanced indexing takes one integer tensor per indexed dim, and these broadcast together (`x[arange(N), y]` → `(N,)`). `gather(dim, idx)`
#    takes one index tensor with the same number of dims as `x`, and the output has `idx`'s shape. For k items per row, `gather(1, idx (N,k))`
#    is natural. With a trailing feature dim, you `expand` the index (B3), or use `x[arange(B)[:, None], idx]`.
# 4. It is `x[idx] = x[idx] + 1`: the read happens once and duplicates all write the same value, so counts are lost. Use `index_add_`,
#    `index_put_(..., accumulate=True)` or `scatter_add_` (or `bincount`). On CUDA these use atomic float adds, whose summation order varies,
#    so results can differ in the last bits run to run. `torch.use_deterministic_algorithms(True)` lists `scatter_add_` and `index_add`
#    on CUDA as nondeterministic (see its docs).
# 5. It needs one `(N,D)@(D,M)` matmul (tensor cores, O(NM) memory) instead of materializing `(N,M,D)` differences. Weakness: catastrophic
#    cancellation when $\|a\|^2 \approx \|b\|^2 \gg \|a-b\|^2$. The result can come out slightly negative (→ NaN after sqrt) or have a large
#    absolute error (E5 measures it). Mitigate by clamping at 0, centering the data first (distances are translation-invariant), computing in
#    float64, or recomputing the small distances exactly.
# 6. You set strides by hand. Out-of-range strides raise an error, but overlapping windows alias memory, so in-place ops on them are
#    undefined per the docs. The result also depends on the input's current layout (the input may already be a strided view, as tested in
#    E1/E2). `F.unfold` (im2col) and `Tensor.unfold` give safe sliding windows. `F.unfold` copies; `Tensor.unfold` is a view.
# 7. With 2 operands the order is irrelevant. With ≥3, `torch.einsum` uses `opt_einsum` (if installed) to pick a near-optimal pairwise
#    contraction path; otherwise it contracts left to right (torch.einsum docs). Bad orders can blow up the intermediate size, e.g.
#    `(A@B)@v` vs `A@(B@v)`.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
