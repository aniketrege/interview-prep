# %% [markdown]
# # A3.4 · conv2d (and max-pool) forward via `unfold` + matmul
#
# **ID:** A3.4 · **Target:** 20 min · **Plan week:** 3 · **Part:** A
#
# **The drill.** Implement `conv2d` with stride, padding and dilation as **im2col**: `F.unfold` turns every receptive
# field into a column, and the convolution becomes one matmul with the flattened kernel. Then derive the output-size
# formula, write max-pool the same way (with the right padding value), and count a conv layer's parameters and FLOPs.
# Stretch: `groups` (depthwise conv is `groups = C_in`). Everything is checked against `F.conv2d`, `F.max_pool2d`,
# `nn.Conv2d` and `torch.utils.flop_counter`.
# Guide A3: "conv2d forward via `unfold`" (blank-editor classic, under 20 min).
#
# **Rules.** Inside your functions: `F.unfold`, `F.pad`, `view`/`reshape`, `@`/`einsum`, reductions. Not `F.conv2d`,
# `F.max_pool2d`, `nn.Conv2d`.
#
# ## Sources
# - V. Dumoulin, F. Visin, *A guide to convolution arithmetic for deep learning*, 2016, arXiv:1603.07285 —
#   https://arxiv.org/abs/1603.07285 (output-size relationships for padding, stride, dilation).
# - Stanford CS231n notes, *Convolutional Neural Networks*, section "Implementation as Matrix Multiplication" (im2col),
#   https://cs231n.github.io/convolutional-networks/
# - PyTorch docs: `nn.Unfold` (column layout `(N, C·∏k, L)` and the output-size formula)
#   https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html · `nn.Conv2d` (weight shape
#   `(out, in/groups, kH, kW)`, groups semantics) https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html ·
#   `F.max_pool2d` (implicit `-inf` padding) https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html
# - A. G. Howard et al., *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*, 2017,
#   arXiv:1704.04861 — https://arxiv.org/abs/1704.04861 (depthwise-separable cost ratio $1/O + 1/k^2$, §3.1).
# - `torch.utils.flop_counter.FlopCounterMode` (counts 2 FLOPs per multiply-add) — PyTorch source
#   `torch/utils/flop_counter.py`, https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py

# %%
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from prep import Session, check, check_close, seed_everything

s = Session("A3.4", target_min=20)
seed_everything(0)

# %% [markdown]
# ## 1. Output size
# A kernel of size $k$ with dilation $d$ spans $d(k-1)+1$ pixels. With padding $p$ on both sides and stride $s$:
# $$H_{out} = \Big\lfloor \frac{H + 2p - d(k-1) - 1}{s} \Big\rfloor + 1.$$

# %%
def out_size(H, k, stride=1, padding=0, dilation=1):
    ### BEGIN SOLUTION
    return (H + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    ### END SOLUTION


bad = []
for H, k, st, p, dl in itertools.product((5, 7, 8, 13), (1, 2, 3, 5), (1, 2, 3), (0, 1, 2), (1, 2)):
    if H + 2 * p < dl * (k - 1) + 1:
        continue
    ref = nn.Conv2d(1, 1, k, stride=st, padding=p, dilation=dl)(torch.zeros(1, 1, H, H)).shape[-1]
    if out_size(H, k, st, p, dl) != ref:
        bad.append((H, k, st, p, dl))
check("out_size matches nn.Conv2d on 250+ configs", not bad, f"wrong for (H,k,s,p,d) = {bad[:5]}")

# %% [markdown]
# ## 2. conv2d = unfold + matmul
# `cols = F.unfold(x, (kh, kw), dilation, padding, stride)` has shape `(N, C·kh·kw, L)` with `L = H_out·W_out`, and its
# rows are ordered (c, i, j) exactly like `w.view(O, C·kh·kw)`. So `out = w_flat @ cols` → `(N, O, L)`; add the bias; `view`
# to `(N, O, H_out, W_out)`. (Like PyTorch, this is cross-correlation: the kernel is not flipped.)

# %%
def conv2d(x, w, b=None, stride=1, padding=0, dilation=1):
    """x (N, C, H, W), w (O, C, kh, kw), b (O,) or None. Ints for stride/padding/dilation."""
    ### BEGIN SOLUTION
    N, C, H, W = x.shape
    O, _, kh, kw = w.shape
    cols = F.unfold(x, (kh, kw), dilation=dilation, padding=padding, stride=stride)    # (N, C*kh*kw, L)
    out = w.view(O, -1) @ cols                                                          # (N, O, L)
    if b is not None:
        out = out + b[:, None]
    return out.view(N, O, out_size(H, kh, stride, padding, dilation), out_size(W, kw, stride, padding, dilation))
    ### END SOLUTION


torch.manual_seed(0)
for (st, p, dl) in [(1, 0, 1), (2, 1, 1), (1, 2, 2), (3, 1, 2), (2, 0, 1)]:
    x = torch.randn(2, 3, 11, 9)
    w, b = torch.randn(5, 3, 3, 2), torch.randn(5)          # non-square kernel and input catch swapped H/W
    check_close(f"conv2d stride={st} pad={p} dil={dl}", conv2d(x, w, b, st, p, dl), F.conv2d(x, w, b, st, p, dl), atol=1e-4)
w1 = torch.randn(4, 3, 1, 1)
check_close("conv2d 1x1, no bias", conv2d(x, w1), F.conv2d(x, w1), atol=1e-4)

# %% [markdown]
# ## 3. Stretch: `groups`
# With `groups = G`: the input channels split into G blocks of `C/G`, the outputs into G blocks of `O/G`, and block g of the
# output only sees block g of the input. `w` has shape `(O, C/G, kh, kw)`. Reshape `cols` to `(N, G, (C/G)·kh·kw, L)` and `w` to
# `(G, O/G, (C/G)·kh·kw)`, then use one `einsum`. Depthwise conv: `G = C = O`.

# %%
def conv2d_groups(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    ### BEGIN SOLUTION
    N, C, H, W = x.shape
    O, Cg, kh, kw = w.shape
    G = groups
    cols = F.unfold(x, (kh, kw), dilation=dilation, padding=padding, stride=stride)
    cols = cols.view(N, G, Cg * kh * kw, -1)
    out = torch.einsum("gok,ngkl->ngol", w.view(G, O // G, Cg * kh * kw), cols).reshape(N, O, -1)
    if b is not None:
        out = out + b[:, None]
    return out.view(N, O, out_size(H, kh, stride, padding, dilation), out_size(W, kw, stride, padding, dilation))
    ### END SOLUTION


x = torch.randn(2, 6, 10, 10)
for G, O in [(1, 4), (2, 4), (3, 9), (6, 6)]:
    w, b = torch.randn(O, 6 // G, 3, 3), torch.randn(O)
    check_close(f"conv2d groups={G}", conv2d_groups(x, w, b, 2, 1, 1, G), F.conv2d(x, w, b, 2, 1, 1, G), atol=1e-4)

# %% [markdown]
# ## 4. Max-pool via unfold
# Unfold each channel's windows: `(N, C·k·k, L)` → view `(N, C, k·k, L)` → `max` over the window dim. **Padding:** `F.unfold` pads
# with **zeros**, but max-pool must ignore padding (PyTorch pads with $-\infty$). For negative inputs zero padding changes
# the answer, so pad with `F.pad(x, ..., value=-inf)` first and unfold with `padding=0`.

# %%
def max_pool2d(x, k, stride=None, padding=0):
    ### BEGIN SOLUTION
    stride = stride or k
    N, C, H, W = x.shape
    xp = F.pad(x, (padding,) * 4, value=float("-inf"))
    cols = F.unfold(xp, k, stride=stride)                                   # (N, C*k*k, L)
    out = cols.view(N, C, k * k, -1).amax(2)
    return out.view(N, C, out_size(H, k, stride, padding), out_size(W, k, stride, padding))
    ### END SOLUTION


xm = torch.randn(2, 3, 9, 7) - 2.0          # mostly negative: zero padding would win the max at the borders
for k, st, p in [(2, None, 0), (3, 2, 1), (3, 1, 1), (2, 1, 1)]:
    check_close(f"max_pool2d k={k} s={st} p={p}", max_pool2d(xm, k, st, p), F.max_pool2d(xm, k, st, p))

# %% [markdown]
# ## 5. Parameters and FLOPs of a conv layer
# Parameters: $O\cdot\frac{C}{G}\cdot k_h k_w$ (+ $O$ for the bias). Multiply-adds per image: $H_{out}W_{out}\cdot O\cdot\frac{C}{G}k_hk_w$
# (each output element is a dot product of length $\frac{C}{G}k_hk_w$). FLOPs $= 2\times$ MACs, the convention of `FlopCounterMode`
# (which ignores the bias add). Example: a 3×3 conv, 64→128 channels on 56×56 with padding 1.

# %%
def conv_params(C, O, kh, kw, groups=1, bias=True):
    ### BEGIN SOLUTION
    return O * (C // groups) * kh * kw + (O if bias else 0)
    ### END SOLUTION


def conv_flops(N, C, H, W, O, kh, kw, stride=1, padding=0, dilation=1, groups=1):
    """2 * multiply-adds for the whole batch (no bias)."""
    ### BEGIN SOLUTION
    Ho, Wo = out_size(H, kh, stride, padding, dilation), out_size(W, kw, stride, padding, dilation)
    return 2 * N * Ho * Wo * O * (C // groups) * kh * kw
    ### END SOLUTION


for (C, O, k, st, p, G, bias) in [(64, 128, 3, 1, 1, 1, True), (32, 32, 3, 2, 1, 32, False), (6, 9, 5, 1, 2, 3, True)]:
    m = nn.Conv2d(C, O, k, stride=st, padding=p, groups=G, bias=bias)
    check(f"params C={C} O={O} k={k} G={G}", conv_params(C, O, k, k, G, bias) == sum(t.numel() for t in m.parameters()))
    xin = torch.randn(2, C, 20, 18)
    with FlopCounterMode(display=False) as fc:
        m(xin)
    check(f"FLOPs C={C} O={O} k={k} s={st} G={G}", conv_flops(2, C, 20, 18, O, k, k, st, p, 1, G) == fc.get_total_flops(),
          f"{conv_flops(2, C, 20, 18, O, k, k, st, p, 1, G)} vs {fc.get_total_flops()}")
print(f"3x3, 64->128, 56x56: {conv_params(64, 128, 3, 3):,} params, {conv_flops(1, 64, 56, 56, 128, 3, 3, padding=1) / 1e9:.2f} GFLOPs per image")

# %% [markdown]
# ## Say it out loud
# 1. Derive the output-size formula. What padding keeps the size ("same") for stride 1, odd $k$, dilation $d$?
# 2. What exactly is in a column of `F.unfold`'s output, and why does `w.view(O, -1)` line up with it?
# 3. What is the memory cost of im2col, and why do libraries still use it (or avoid it: implicit GEMM, Winograd, FFT)?
# 4. Parameters and FLOPs of a conv layer; how do a depthwise-separable conv's compare to a standard conv?
# 5. Why must max-pool padding be $-\infty$? What padding does average-pool use, and what does `count_include_pad` change?
# 6. Convolution vs cross-correlation: does it matter for learning?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. The padded input has $H+2p$ pixels and a dilated kernel spans $d(k-1)+1$. The number of valid start positions with step $s$ is
#    $\lfloor (H+2p-d(k-1)-1)/s\rfloor + 1$ (Dumoulin & Visin 2016). For $s=1$, "same" needs $2p = d(k-1)$, so $p = d(k-1)/2$ (e.g. 1 for 3×3, 2 for 3×3 with $d=2$).
# 2. Column $l$ (one output location) holds the receptive field for that location, flattened in (channel, kernel-row, kernel-col) order, i.e. the
#    same row-major order in which `w[o]` of shape `(C, kh, kw)` flattens. So each output is the dot product `w_flat[o] · cols[:, l]`.
# 3. The columns tensor is $C k_h k_w \times H_{out}W_{out}$ per image, about $k_hk_w$ times the input (9× for 3×3), which is a lot of memory traffic.
#    The win is that the conv becomes one large GEMM, which is highly optimized and hits tensor cores. cuDNN uses implicit GEMM (it forms the
#    tiles on the fly in shared memory and never writes the column matrix), Winograd for small kernels (fewer multiplies), or FFT for large kernels.
# 4. Params $O\,(C/G)\,k^2 + O$; FLOPs $2\,H_{out}W_{out}\,O\,(C/G)\,k^2$ per image. Depthwise-separable = depthwise $k\times k$ ($G=C$) + pointwise 1×1:
#    MACs $HW(Ck^2 + CO)$ vs $HWCOk^2$, a ratio of $1/O + 1/k^2$, about 8-9× fewer for 3×3 (Howard et al. 2017, §3.1).
# 5. A padded element must never be selected as the max, and $-\infty$ guarantees that; zeros would beat negative activations at the borders
#    (the test uses mostly negative inputs to catch exactly this). Average pooling pads with zeros. `count_include_pad=True` (the default)
#    divides by $k^2$ including the padded zeros; `False` divides by the number of real pixels in the window.
# 6. True convolution flips the kernel. Deep-learning "conv" layers compute cross-correlation. Since the kernel is learned, the flip is absorbed
#    into the weights; it only matters when you load fixed kernels from signal-processing code, or when you derive the backward pass (the input
#    gradient of a cross-correlation is a true convolution with the same kernel, i.e. a transposed conv).
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
