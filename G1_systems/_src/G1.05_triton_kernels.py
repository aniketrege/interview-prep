# %% [markdown]
# # G1.5 · Triton kernels: vector add, fused softmax, matmul, FlashAttention-2 forward
#
# **ID:** G1.5 · **Target:** 120 min · **Plan week:** 15 (drill: Triton softmax) · **Part:** G1
#
# **The lab.** Guide G1: *"Kernels at a glance: a Triton softmax or matmul kernel."* You will write four kernels,
# each checked against PyTorch and benchmarked on this RTX 3090: a vector add (bandwidth), a fused row softmax
# (fusion), a tiled matmul with grouped program ordering (L2 reuse, tensor cores, and the FP16-accumulate rate the
# whitepaper advertises), and a **FlashAttention-2 forward** (non-causal and causal, fp16/bf16, head dim 64) that you
# complete from the PyTorch version of G1.4.
#
# Everything here runs on sm_86 with Triton 3.8: plain pointer arithmetic and `tl.dot`, no Hopper-only features (TMA
# tensor descriptors, warp specialization, `tl.make_tensor_descriptor`). Solution runtime: about 1 min with a cold Triton
# cache (each kernel and config compiles once), well under the ~8 min budget allowed for this lab.
#
# ## Sources
# - Triton tutorials (triton-lang/triton @c1641ab, MIT): `python/tutorials/01-vector-add.py`,
#   `02-fused-softmax.py`, `03-matrix-multiplication.py` (grouped ordering, "L2 Cache Optimizations"),
#   `06-fused-attention.py` (FA2 forward/backward in Triton; backward pointer). Kernels below are adapted from 01-03
#   (structure and names) and written from Algorithm 1 for attention. https://github.com/triton-lang/triton/tree/main/python/tutorials
#   and https://triton-lang.org/main/getting-started/tutorials/index.html
# - P. Tillet, H. T. Kung, D. Cox, *Triton: an intermediate language and compiler for tiled neural network
#   computations*, MAPL 2019, DOI: 10.1145/3315508.3329973 — https://doi.org/10.1145/3315508.3329973
# - Stanford CS336 Spring 2026 Assignment 2 handout §4.2.1-4.2.3 (Triton weighted-sum example, FA2 forward kernel spec:
#   grid `(T_q, batch)`, fp32 on-chip buffers, cast $\tilde P$ to V's dtype, causal flag as `tl.constexpr`; optional
#   Triton backward, Algorithm 2) and `tests/test_attention.py` tolerances — https://github.com/stanford-cs336/assignment2-systems @ca8bc81 (MIT)
# - T. Dao, *FlashAttention-2*, arXiv:2307.08691 — https://arxiv.org/abs/2307.08691
# - NVIDIA GA102 whitepaper v2, Appendix A Table 9 (RTX 3090: FP16 tensor 142 TFLOPS with FP16 accumulate, 71 with FP32
#   accumulate; 936 GB/s) — https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
# - Modal GPU Glossary, "occupancy", "warp", "shared memory": https://modal.com/gpu-glossary

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton.testing import do_bench

from prep import Session, check, check_close, seed_everything

s = Session("G1.5", target_min=120)
seed_everything(0)
dev = torch.device("cuda")
print(torch.cuda.get_device_name(), "triton", triton.__version__)

# %% [markdown]
# ## 0. The Triton programming model in five lines
#
# - You write the code of **one program** (≈ a CUDA thread block). You launch a **grid** of programs:
#   `kernel[grid](args...)`; inside, `tl.program_id(axis)` says which one you are.
# - A program operates on **blocks** (tiles) of values, not on scalars: `offs = pid * BLOCK + tl.arange(0, BLOCK)`
#   is a vector of `BLOCK` indices. Triton maps the block onto the program's threads (`num_warps` × 32) for you,
#   handles vectorized/coalesced loads, and stages data through shared memory for `tl.dot`.
# - Memory is accessed through **pointers**: `tl.load(ptr + offs, mask=offs < n, other=0.0)` and
#   `tl.store(ptr + offs, x, mask=...)`. The **mask** guards the ragged last block.
# - Block shapes must be **compile-time constants** (`BLOCK: tl.constexpr`, powers of 2); each distinct value compiles
#   a new kernel. `num_warps` and `num_stages` (software-pipelining depth for loads in loops) are launch options.
# - `tl.dot(a, b, acc)` is a tile matmul on tensor cores (fp16/bf16/tf32 inputs; accumulate in fp32 by default).
#
# CUDA makes you program a thread and think about the block; Triton lets you program the block and leaves the threads
# to the compiler. You still choose the tiling, the grid, and what stays on chip, which is where performance comes from.

# %% [markdown]
# ## 1. Vector add (Triton tutorial 01)
# **Exercise 1.1.** Each program handles `BLOCK_SIZE` consecutive elements: compute the offsets, a mask for the tail,
# load `x` and `y`, store `x + y`.

# %%
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    ### BEGIN SOLUTION
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
    ### END SOLUTION


def add(x, y, BLOCK_SIZE=1024):
    out = torch.empty_like(x)
    n = out.numel()
    add_kernel[(triton.cdiv(n, BLOCK_SIZE),)](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


xa, ya = torch.rand(98432, device=dev), torch.rand(98432, device=dev)  # not a multiple of BLOCK_SIZE
check_close("vector add (ragged tail)", add(xa, ya), xa + ya, atol=0, rtol=0)
buf = torch.full((98432 + 4096,), -7.0, device=dev)  # out-of-bounds stores would land in the sentinel region
add_kernel[(triton.cdiv(98432, 1024),)](xa, ya, buf, 98432, BLOCK_SIZE=1024)
check("no stores past n_elements (the mask works)", bool((buf[98432:] == -7.0).all()))

sizes = [2**k for k in range(12, 29, 2)]
gbps = {"triton": [], "torch": []}
for n in sizes:
    xa, ya = torch.rand(n, device=dev), torch.rand(n, device=dev)
    for name, fn in (("triton", lambda: add(xa, ya)), ("torch", lambda: xa + ya)):
        gbps[name].append(3 * 4 * n / (do_bench(fn) * 1e-3) / 1e9)  # read x, y; write out
print("GB/s at 2^28 elements:", {k: round(v[-1]) for k, v in gbps.items()}, "(whitepaper peak 936)")
plt.figure(figsize=(5.5, 3.2))
for k, v in gbps.items():
    plt.semilogx(sizes, v, "o-", label=k)
plt.axhline(936, color="k", lw=0.8, label="936 GB/s peak")
plt.xscale("log", base=2); plt.xlabel("elements (fp32)"); plt.ylabel("GB/s"); plt.title("vector add"); plt.legend(); plt.tight_layout(); plt.show()
check("Triton add reaches >= 85% of torch's bandwidth at 2^28", gbps["triton"][-1] > 0.85 * gbps["torch"][-1])

# %% [markdown]
# ## 2. Fused softmax (Triton tutorial 02)
#
# The naive PyTorch softmax `x.max`, `x - max`, `exp`, `sum`, `/` is five kernels that read about $5MN$ and write $3MN$
# elements (tutorial 02's count). Softmax is memory-bound, so one kernel that reads each row **once** into registers,
# does everything on chip, and writes once ($2MN$) should be up to ~4× faster than the naive version, and match or beat
# `torch.softmax`.
#
# **Exercise 2.1.** One program per row. `BLOCK_SIZE` is the next power of two ≥ `n_cols`. Load the row with
# `other=-inf` (why -inf, not 0?), subtract the max, exponentiate, normalize, store.

# %%
@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    ### BEGIN SOLUTION
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)
    ### END SOLUTION


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE <= 2048 else (8 if BLOCK_SIZE <= 8192 else 16)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y


def naive_softmax(x):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    num = torch.exp(z)
    return num / num.sum(dim=1)[:, None]


xs = torch.randn(1823, 781, device=dev) * 4  # tutorial 02's awkward shape
check_close("fused softmax", softmax(xs), torch.softmax(xs, 1), atol=1e-6)
check_close("fused softmax, huge logits", softmax(xs * 1e3), torch.softmax(xs * 1e3, 1), atol=1e-6)

cols_list = [256, 512, 1024, 2048, 4096, 8192, 16384]
sm_gbps = {"triton fused": [], "torch.softmax": [], "naive (5 kernels)": []}
for nc in cols_list:
    xs = torch.randn(4096, nc, device=dev)
    for name, fn in (("triton fused", softmax), ("torch.softmax", lambda t: torch.softmax(t, -1)), ("naive (5 kernels)", naive_softmax)):
        sm_gbps[name].append(2 * xs.numel() * 4 / (do_bench(lambda: fn(xs)) * 1e-3) / 1e9)
plt.figure(figsize=(5.5, 3.2))
for k, v in sm_gbps.items():
    plt.semilogx(cols_list, v, "o-", label=k)
plt.xscale("log", base=2); plt.xlabel("row length (4096 rows, fp32)"); plt.ylabel("effective GB/s (2 x bytes / time)")
plt.title("softmax"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()
print({k: [round(x) for x in v] for k, v in sm_gbps.items()})
check("fused beats naive by >= 2x at 4096 columns", sm_gbps["triton fused"][4] > 2 * sm_gbps["naive (5 kernels)"][4])

# %% [markdown]
# ## 3. Matmul with grouped ordering (Triton tutorial 03)
#
# Program $(m, n)$ computes the $B_M\times B_N$ tile of $C$ by looping over $K$ in steps of $B_K$:
# `acc += tl.dot(A[m-tile, k-tile], B[k-tile, n-tile])`. Arithmetic intensity per tile is
# $\frac{2B_MB_NB_K}{2(B_MB_K + B_KB_N)}$ FLOP/byte of *global* loads ($= 64$ for $128\times128$ tiles in fp16), so the
# loads come from L2 more than from DRAM if neighbouring programs share tiles.
#
# **Grouped ordering.** Programs launch in order of `pid`. Row-major mapping (`pid_m = pid // num_pid_n`) runs a whole
# row of C tiles at once: it reuses one row-panel of A but streams *all* of B. Grouping `GROUP_M` rows and walking
# column-major inside the group makes concurrently running programs share both A and B panels, raising the L2 hit rate.
#
# **Exercise 3.1.** (a) The grouped `pid → (pid_m, pid_n)` mapping. (b) The K loop: load the A and B tiles with masks for
# the ragged K edge, accumulate with `tl.dot(a, b, acc, out_dtype=ACC_DTYPE)`, advance the pointers.

# %%
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                  ACC_DTYPE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    ### BEGIN SOLUTION
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    ### END SOLUTION
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M  # wrap instead of mask on M, N (tutorial 03)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    ### BEGIN SOLUTION
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=ACC_DTYPE)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    ### END SOLUTION
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, acc16=False, num_warps=4, num_stages=4):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
                        ACC_DTYPE=tl.float16 if acc16 else tl.float32, num_warps=num_warps, num_stages=num_stages)
    return c


torch.manual_seed(0)
for (M, N, K) in [(512, 512, 512), (1000, 777, 300), (64, 4096, 1024)]:
    a, b = torch.randn(M, K, device=dev, dtype=torch.float16), torch.randn(K, N, device=dev, dtype=torch.float16)
    check_close(f"Triton matmul {M}x{N}x{K} vs torch", matmul(a, b), a @ b, atol=0.05, rtol=0.01)
# The grouped mapping must be a permutation of all tiles:
nm, nn_, G = 7, 5, 3
seen = set()
for pid in range(nm * nn_):
    grp = pid // (G * nn_); first = grp * G; gs = min(nm - first, G)
    seen.add((first + (pid % (G * nn_)) % gs, (pid % (G * nn_)) // gs))
check("grouped ordering covers every (pid_m, pid_n) once (reference formula)", len(seen) == nm * nn_)

# %%
mm_sizes = [1024, 2048, 4096, 8192]
tfl = {"cuBLAS (torch)": [], "Triton, GROUP_M=8": [], "Triton, GROUP_M=1 (row-major)": [], "Triton, FP16 accumulate": []}
for n in mm_sizes:
    a, b = torch.randn(n, n, device=dev, dtype=torch.float16), torch.randn(n, n, device=dev, dtype=torch.float16)
    for name, fn in (("cuBLAS (torch)", lambda: a @ b), ("Triton, GROUP_M=8", lambda: matmul(a, b)),
                     ("Triton, GROUP_M=1 (row-major)", lambda: matmul(a, b, GROUP_M=1)),
                     ("Triton, FP16 accumulate", lambda: matmul(a, b, acc16=True))):
        tfl[name].append(2 * n**3 / (do_bench(fn) * 1e-3) / 1e12)
for k, v in tfl.items():
    print(f"{k:30s}", "  ".join(f"{n}: {x:5.1f}" for n, x in zip(mm_sizes, v)), "TFLOP/s")
plt.figure(figsize=(6, 3.5))
for k, v in tfl.items():
    plt.plot(mm_sizes, v, "o-", label=k)
plt.axhline(71, color="k", ls=":", lw=1, label="71 (FP32 acc. peak)"); plt.axhline(142, color="k", ls="--", lw=1, label="142 (FP16 acc. peak)")
plt.xscale("log", base=2); plt.xlabel("n (n x n x n, fp16)"); plt.ylabel("TFLOP/s"); plt.legend(fontsize=7); plt.tight_layout(); plt.show()
check("Triton matmul within 15% of cuBLAS at n=4096", tfl["Triton, GROUP_M=8"][2] > 0.85 * tfl["cuBLAS (torch)"][2])
check("FP16 accumulation runs > 1.4x faster than FP32 accumulation (GeForce half-rate FP32 acc.)",
      tfl["Triton, FP16 accumulate"][2] > 1.4 * tfl["Triton, GROUP_M=8"][2])

# %% [markdown]
# Three things the plot shows. (1) A 30-line Triton kernel gets close to cuBLAS. (2) On GeForce Ampere, FP16 tensor-core
# math with **FP16 accumulation** really does run near the 142 TFLOPS row of the whitepaper, twice the FP32-accumulate
# rate, at the cost of accuracy (a length-$K$ sum in fp16 loses bits; never train with it). A100/H100 have no such
# penalty. (3) Grouped vs row-major ordering is within noise here: for square fp16 matrices up to $n=8192$ the 3090's
# 6 MB L2 plus 128-wide tiles already give enough reuse. It matters more for very large or very skinny shapes and on
# GPUs where DRAM bandwidth is scarcer relative to compute. Measure before you believe a folklore optimization.

# %% [markdown]
# ## 4. FlashAttention-2 forward in Triton
#
# Layout: `q, k, v` are `(B*H, T, D)` contiguous, `D = 64`. Grid `(cdiv(T, BLOCK_M), B*H)`: program `(i, bh)` owns query
# rows `[i*BLOCK_M, (i+1)*BLOCK_M)` of head `bh` (CS336 §4.2.2: one Q tile per program, one loop over K tiles). Running
# statistics $m, \ell$ and the accumulator stay in fp32 registers. Two standard tricks:
# - use `exp2`: $e^{x} = 2^{x\log_2 e}$, so fold $\log_2 e$ into the softmax scale once (`qk_scale`) and use `tl.exp2`
#   (one hardware instruction);
# - **causal**: stop the K loop at the end of this Q tile (`hi = min((i+1)*BLOCK_M, T)`), and mask `key > query`
#   inside the tile. Here we apply the mask on every tile for simplicity; production kernels split the loop into
#   "below the diagonal, no mask" and "on the diagonal, masked".
#
# **Exercise 4.1.** Complete the loop body (G1.4 §2 in Triton): scores, masking (`key < T`, and causal), running max,
# $\tilde P$, rescale factor `alpha`, $\ell$, accumulator (cast `p` to `v.dtype` before `tl.dot`), update `m_i`. Then
# write the normalized output and $L$ (natural-log logsumexp) after the loop.

# %%
@triton.jit
def flash_fwd_kernel(Q, K, V, O, L, stride_qb, stride_qt, stride_kb, stride_kt, stride_vb, stride_vt,
                     stride_ob, stride_ot, stride_lb, T, scale,
                     D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + bh * stride_qb + offs_m[:, None] * stride_qt + offs_d[None, :], mask=offs_m[:, None] < T, other=0.0)
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    qk_scale = scale * 1.4426950408889634  # log2(e): we work in base 2
    hi = tl.minimum((pid_m + 1) * BLOCK_M, T) if CAUSAL else T
    for start_n in range(0, hi, BLOCK_N):
        kn = start_n + offs_n
        k = tl.load(K + bh * stride_kb + kn[None, :] * stride_kt + offs_d[:, None], mask=kn[None, :] < T, other=0.0)  # (D, BLOCK_N)
        v = tl.load(V + bh * stride_vb + kn[:, None] * stride_vt + offs_d[None, :], mask=kn[:, None] < T, other=0.0)  # (BLOCK_N, D)
        ### BEGIN SOLUTION
        s_ = tl.dot(q, k) * qk_scale
        valid = kn[None, :] < T
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= kn[None, :])
        s_ = tl.where(valid, s_, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s_, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s_ - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        ### END SOLUTION
    ### BEGIN SOLUTION
    acc = acc / l_i[:, None]
    tl.store(O + bh * stride_ob + offs_m[:, None] * stride_ot + offs_d[None, :], acc.to(O.dtype.element_ty), mask=offs_m[:, None] < T)
    tl.store(L + bh * stride_lb + offs_m, (m_i + tl.log2(l_i)) * 0.6931471805599453, mask=offs_m < T)  # back to natural log
    ### END SOLUTION


def flash_attention(q, k, v, causal=False, BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2):
    """q, k, v: (B, H, T, 64) fp16/bf16. Returns O (same shape/dtype) and L (B, H, T) fp32."""
    B, H, T, D = q.shape
    q3, k3, v3 = (t.contiguous().view(B * H, T, D) for t in (q, k, v))
    o = torch.empty_like(q3)
    L = torch.empty(B * H, T, device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(T, BLOCK_M), B * H)
    flash_fwd_kernel[grid](q3, k3, v3, o, L, q3.stride(0), q3.stride(1), k3.stride(0), k3.stride(1),
                           v3.stride(0), v3.stride(1), o.stride(0), o.stride(1), L.stride(0), T, 1.0 / math.sqrt(D),
                           D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal, num_warps=num_warps, num_stages=num_stages)
    return o.view(B, H, T, D), L.view(B, H, T)


def ref_lse(q, k, causal):
    S = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        T = q.shape[-2]
        S = S.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), -math.inf)
    return torch.logsumexp(S, -1)


torch.manual_seed(0)
for dtype in (torch.float16, torch.bfloat16):
    for causal in (False, True):
        for T in (128, 1000):
            q, k, v = (torch.randn(2, 4, T, 64, device=dev, dtype=dtype) for _ in range(3))
            o, L = flash_attention(q, k, v, causal)
            o_ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            tag = f"{str(dtype)[6:]} causal={causal} T={T}"
            check_close(f"O vs SDPA, {tag}", o, o_ref, atol=1e-2, rtol=1e-2)
            check_close(f"L vs logsumexp, {tag}", L, ref_lse(q, k, causal), atol=1e-3, rtol=1e-3)

# %%
Bb, H, D = 4, 16, 64
Ts = [512, 1024, 2048, 4096, 8192]
att = {"Triton FA2 (ours)": [], "SDPA flash backend": [], "naive (materialized)": []}


def naive_attn(q, k, v, causal):
    S = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        T = q.shape[-2]
        S = S.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return S.softmax(-1) @ v


def sdpa_flash(q, k, v, causal):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


causal = True
for T in Ts:
    q, k, v = (torch.randn(Bb, H, T, D, device=dev, dtype=torch.float16) for _ in range(3))
    flops = 4 * Bb * H * T * T * D / 2  # causal: half of the T x T work
    for name, fn in (("Triton FA2 (ours)", lambda: flash_attention(q, k, v, causal, BLOCK_M=64, BLOCK_N=64)),
                     ("SDPA flash backend", lambda: sdpa_flash(q, k, v, causal)),
                     ("naive (materialized)", lambda: naive_attn(q, k, v, causal))):
        if name.startswith("naive") and T > 4096:  # three 8.6 GB T x T tensors would not fit in 24 GB
            att[name].append(float("nan"))
            continue
        att[name].append(flops / (do_bench(fn) * 1e-3) / 1e12)
for k_, v_ in att.items():
    print(f"{k_:22s}", "  ".join(f"T={t}: {x:5.1f}" for t, x in zip(Ts, v_)), "TFLOP/s (causal, model FLOPs)")
plt.figure(figsize=(6, 3.5))
for k_, v_ in att.items():
    plt.plot(Ts, v_, "o-", label=k_)
plt.xscale("log", base=2); plt.xlabel("sequence length T"); plt.ylabel("TFLOP/s (causal forward)")
plt.title(f"attention forward, B={Bb} H={H} d={D} fp16"); plt.legend(); plt.tight_layout(); plt.show()
check("our kernel is within 30% of PyTorch's flash backend at T=4096", att["Triton FA2 (ours)"][3] > 0.7 * att["SDPA flash backend"][3])
check("our kernel beats naive attention by >= 2x at T=4096", att["Triton FA2 (ours)"][3] > 2 * att["naive (materialized)"][3])

# %% [markdown]
# **Backward (optional, next step).** CS336 A2 §4.2.3 gives the tiled FA2 backward (Algorithm 2): one kernel with K/V
# tiles in the outer loop accumulating $dK, dV$, and one with Q tiles outer accumulating $dQ$, both recomputing
# $P = \exp(S - L)$ and using $D = \mathrm{rowsum}(O\circ dO)$; computing $P$ twice avoids atomics. Triton's
# `06-fused-attention.py` is a reference implementation (it also uses Hopper features when available; on sm_86 it falls
# back to plain pointers). Your G1.4 `flash_attn_bwd` is the PyTorch spec to test against.

# %% [markdown]
# ## Say it out loud
# 1. What is a Triton "program", and how does it relate to a CUDA thread block and warp? What does `num_warps` change?
# 2. Why must block sizes be `tl.constexpr`? Why powers of two? What does the mask do?
# 3. Why is the fused softmax faster than the five-kernel version, and why must padding be `-inf`? What limits it for
#    very long rows?
# 4. Explain grouped program ordering in the matmul and when it matters.
# 5. Your Triton matmul runs at 2× cuBLAS on a 3090. What did you probably change, and is it acceptable for training?
# 6. Walk through your FlashAttention kernel: grid, what is in registers, why fp32 accumulators, why cast `p` before
#    `tl.dot`, how causal masking saves work.
# 7. What would you change to make the kernel fast on an H100?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. One program instance = one CUDA thread block executing the kernel on a block of data; `tl.program_id` is the
#    block index in the grid. Triton distributes the program's tile operations over `num_warps` × 32 threads and decides
#    the per-thread layout. More warps means more parallelism per program (good for big tiles, long rows), but more
#    registers and shared memory per block, so fewer resident blocks (occupancy).
# 2. Tile shapes determine register allocation, shared-memory layout and the tensor-core instruction mapping, all at
#    compile time; each new value is a new compiled kernel. `tl.arange` requires power-of-two lengths, and tensor-core
#    tiles are powers of two. The mask disables out-of-range lanes of the last block for loads (with `other=`) and
#    stores, so no out-of-bounds memory is touched.
# 3. Softmax is memory-bound (a handful of FLOPs per element). The fused kernel reads each row once and writes once
#    ($2MN$ elements) instead of ~$8MN$ for the five kernels, and launches one kernel instead of five. Padding with
#    $-\infty$ makes padded lanes contribute $e^{-\infty} = 0$ to the sum and never win the max; zeros would corrupt both.
#    For very long rows the row no longer fits in registers (spills, low occupancy): use a looped online softmax (G1.4)
#    over column chunks.
# 4. Programs are scheduled roughly in `pid` order, about one wave of 82+ at a time. Row-major order makes the
#    concurrent programs share one A row-panel but each need a different B column-panel, so B is re-read from DRAM.
#    Grouping `GROUP_M` rows and going down columns inside a group makes a wave touch about $\sqrt{\text{wave}}$ panels
#    of each matrix, so more loads hit L2. It matters when A and B are much larger than L2 (6 MB here) (Triton tutorial 03).
# 5. Probably FP16 accumulation (`out_dtype=tl.float16`): GeForce Ampere runs it at 142 vs 71 TFLOPS (GA102 whitepaper).
#    Each output sums $K$ products in fp16, so error grows with $K$ and large values can overflow. Not acceptable for
#    training matmuls; possibly acceptable for some inference with care. (Or the comparison was unfair: wrong FLOP count,
#    timing without sync, or a different shape.)
# 6. Grid = (Q tiles, batch × heads); each program loads its $Q$ tile once and streams $K, V$ tiles. In registers: $Q$
#    tile, the score tile, $\tilde P$, the fp32 accumulator $(B_M\times D)$, $m$, $\ell$. Accumulators are fp32 because
#    the running sums and rescalings would lose precision in fp16, and `tl.dot` accumulates in fp32. $\tilde P$ is cast to
#    V's dtype so the second `tl.dot` runs on fp16/bf16 tensor cores (CS336 §4.2.2 tip). Causal: the loop stops at the
#    diagonal tile (`hi`), so about half the tiles are never loaded or multiplied.
# 7. Use Hopper features: TMA for asynchronous tile loads (tensor descriptors), WGMMA asynchronous warpgroup matmuls,
#    warp specialization (producer/consumer warps) and pipelining so softmax overlaps matmuls, bigger tiles thanks to
#    228 KB shared memory, possibly FP8 (FlashAttention-3). Also split the causal loop into unmasked and masked parts and
#    autotune tile sizes per head dim.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
