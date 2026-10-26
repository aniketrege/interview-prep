# %% [markdown]
# # G1.13 · Tensor and sequence parallelism: Megatron's f/g operators by hand
#
# **ID:** G1.13 · **Target:** 110 min · **Plan week:** 15 · **Part:** G1
#
# **The lab.** Data parallelism (G1.11-G1.12) splits the *batch*. Tensor parallelism (TP) splits the *weight
# matrices* of every layer across $t$ GPUs, so each GPU computes a slice of every matmul. You will:
# 1. derive Megatron-LM's column-parallel → row-parallel MLP and its two conjugate operators $f$ (identity forward,
#    all-reduce backward) and $g$ (all-reduce forward, identity backward), and **implement them as
#    `torch.autograd.Function`s**;
# 2. build a TP MLP, a **head-parallel attention** (slicing tinygpt's fused QKV correctly) and a full TP transformer
#    block, and verify **forward outputs and every gradient** against the unsharded module;
# 3. count the collectives: 2 all-reduces forward + 2 backward per layer;
# 4. add **sequence parallelism** (Korthikanti et al. 2022): replace each all-reduce by reduce-scatter +
#    all-gather so LayerNorm/residual activations are sharded along the sequence. Prove equivalence, including the
#    LayerNorm-gradient all-reduce that is easy to forget;
# 5. time a TP MLP on this PCIe box against one GPU and explain, with numbers, **why TP stays inside an NVLink node**.
#
# ## Sources
# - M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, B. Catanzaro, *Megatron-LM: Training Multi-Billion
#   Parameter Language Models Using Model Parallelism*, 2019, arXiv:1909.08053 — https://arxiv.org/abs/1909.08053
#   (§3: column-parallel $A$, row-parallel $B$, conjugate $f$/$g$, head-parallel attention, "two all-reduces in the
#   forward path and two in the backward path").
# - V. Korthikanti, J. Casper, S. Lym, L. McAfee, M. Andersch, M. Shoeybi, B. Catanzaro, *Reducing Activation
#   Recomputation in Large Transformer Models*, 2022, arXiv:2205.05198 — https://arxiv.org/abs/2205.05198 (§4.2.2
#   sequence parallelism: $g$ = all-gather fwd / reduce-scatter bwd, $\bar g$ = reduce-scatter fwd / all-gather bwd;
#   same bandwidth as TP; Table 2 activation memory $sbh(10 + 24/t + 5as/(ht))$ vs $sbh(34/t + 5as/(ht))$).
#   Note the papers name the operators differently: Megatron's $g$ (all-reduce fwd) is Korthikanti's $\bar f$.
# - D. Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, SC 2021,
#   arXiv:2104.04473 (Takeaway #1: use TP up to the number of GPUs per server, then pipeline parallelism).
# - Hugging Face picotron, `picotron/tensor_parallel/tp_communications.py` and `tensor_parallel.py`
#   (`CopyToModelParallelRegion`, `ReduceFromModelParallelRegion`, `ColumnParallelLinear`, `RowParallelLinear`),
#   huggingface/picotron@59714b1, Apache-2.0. Our autograd functions and parallel linears adapt these.
# - Stanford CS336 Spring 2026, Assignment 2 §8.4 (TP calculations for an FFN). stanford-cs336/assignment2-systems@ca8bc81, MIT.
# - J. Austin et al., *How to Scale Your Model*, 2025, Part 5 "How to Parallelize a Transformer for Training"
#   (TP is comm-bound when $Y > M_Y F/2550$ on TPU v5p; the analogous GPU argument is below)
#   https://jax-ml.github.io/scaling-book/training/ and Part 12 (H100 NVLink 450 GB/s/direction, IB 50 GB/s/GPU)
#   https://jax-ml.github.io/scaling-book/gpus/
# - N. Tazi et al., *The Ultra-Scale Playbook*, Hugging Face 2025, §"Tensor Parallelism" and §"Sequence
#   Parallelism" ("TP+SP is usually done only within a node"): https://huggingface.co/spaces/nanotron/ultrascale-playbook

# %% [markdown]
# ## 1. The math
# MLP: $Y = \mathrm{GeLU}(XA)$, $Z = YB$ with $X\in\mathbb R^{n\times h}$, $A\in\mathbb R^{h\times 4h}$,
# $B\in\mathbb R^{4h\times h}$. Split $A = [A_1,\dots,A_t]$ by **columns** and $B$ by **rows**:
# $$Y_i = \mathrm{GeLU}(XA_i)\ \text{(no communication: GeLU is elementwise)},\qquad Z = \sum_i Y_iB_i\ \text{(one all-reduce)}.$$
# Column-then-row needs one collective per MLP. Row-then-column would need the full $Y$ before GeLU.
#
# ```
#              f                      column-parallel          row-parallel        g
#  X (full) ──[id fwd | AR bwd]──> X ──> X A_i ─GeLU─> Y_i ──> Y_i B_i (partial Z) ──[AR fwd | id bwd]──> Z (full)
#  rank i holds A[:, i-th 4h/t cols], B[i-th 4h/t rows, :]; bias of A sliced, bias of B added ONCE after g
# ```
# **Backward.** $g$ sums partial outputs in the forward pass, so its backward just passes $\partial L/\partial Z$
# to every rank (identity). $X$ feeds every rank's $XA_i$, so $\partial L/\partial X = \sum_i
# \partial L/\partial X^{(i)}$: $f$'s backward is an all-reduce. Weight gradients need **no** communication:
# $\partial L/\partial A_i = X^\top(\cdot)$ only involves rank $i$'s slice.
#
# **Attention** splits the same way by **heads**. The Q, K, V projections are column-parallel (rank $i$ computes
# heads $iH/t .. (i+1)H/t$), attention per head is local, and the output projection is row-parallel. So one $f$/$g$
# pair per attention block and one per MLP: **2 all-reduces in forward, 2 in backward, per layer**, each of
# $b\cdot s\cdot h$ elements. They sit on the critical path: the next op needs the reduced result.
#
# **Your task (inside the script below):**
# 1. `CopyToTP` ($f$) and `ReduceFromTP` ($g$) as `torch.autograd.Function`s. Clone before an in-place all-reduce:
#    autograd may own the incoming gradient buffer;
# 2. `TPMLP`, built from a full `nn.Sequential(Linear, GELU, Linear)` that is identical on all ranks (rank $i$ keeps
#    its slices). `local(x)` returns this rank's **partial** output with no communication, and
#    `forward(x) = g(local(f(x))) + b_2`;
# 3. `TPAttention`, built from tinygpt's `CausalSelfAttention`. Its `qkv` weight is **fused**: rows $[0,D)$ are Q,
#    $[D,2D)$ K, $[2D,3D)$ V, and head $j$ uses rows $j\cdot d_h .. (j+1)d_h$ *within each* of the three. Slicing
#    rows $[r\cdot 3D/t, (r+1)3D/t)$ of the fused matrix is the classic bug;
# 4. `TPBlock`: pre-norm residual block with replicated LayerNorms.

# %%
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from prep import Session, check, check_close

s = Session("G1.13", target_min=110)
NGPU = torch.cuda.device_count()
NPROC = min(4, NGPU) if NGPU >= 2 else 2
BACKEND = "nccl" if NGPU >= 2 else "gloo"
RUN = Path("runs_G1.13")
RUN.mkdir(exist_ok=True)
print(f"{NGPU} GPUs -> NPROC={NPROC}, backend={BACKEND}")


def torchrun(script, *args, nproc=NPROC, timeout=900):
    """Run `torchrun --standalone --nproc_per_node={nproc} script args` inside RUN/; raise if any rank fails."""
    args = [str(a) for a in args]
    print(f"$ torchrun --standalone --nproc_per_node={nproc} {script} {' '.join(args)}")
    env = {**os.environ, "BACKEND": BACKEND, "OMP_NUM_THREADS": "2"}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", script, *args]
    p = subprocess.run(cmd, cwd=RUN, capture_output=True, text=True, timeout=timeout, env=env)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr[-6000:])
        raise RuntimeError(f"{script} exited with code {p.returncode}")


def load(name):
    return json.loads((RUN / name).read_text())

# %%
# %%writefile runs_G1.13/my_tp.py
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

COMM = {"all_reduce": 0, "all_gather": 0, "reduce_scatter": 0}   # call counters (read by the harness)


def _all_reduce(x):
    COMM["all_reduce"] += 1
    x = x.clone()
    dist.all_reduce(x)
    return x


class CopyToTP(torch.autograd.Function):
    """f: identity in forward, all-reduce (sum) of the gradient in backward."""

    @staticmethod
    def forward(ctx, x):
        ### BEGIN SOLUTION
        return x
        ### END SOLUTION

    @staticmethod
    def backward(ctx, grad):
        ### BEGIN SOLUTION
        return _all_reduce(grad)
        ### END SOLUTION


class ReduceFromTP(torch.autograd.Function):
    """g: all-reduce (sum) in forward, identity in backward."""

    @staticmethod
    def forward(ctx, x):
        ### BEGIN SOLUTION
        return _all_reduce(x)
        ### END SOLUTION

    @staticmethod
    def backward(ctx, grad):
        ### BEGIN SOLUTION
        return grad
        ### END SOLUTION


class TPMLP(nn.Module):
    """Column-parallel Linear(h, 4h) -> GELU -> row-parallel Linear(4h, h), from a full nn.Sequential."""

    def __init__(self, mlp, rank, world):
        super().__init__()
        fc1, act, fc2 = mlp
        self.act = act
        ### BEGIN SOLUTION
        n = fc1.out_features // world
        sl = slice(rank * n, (rank + 1) * n)
        self.w1 = nn.Parameter(fc1.weight.detach()[sl].clone())        # (4h/t, h): rows of nn.Linear's (out, in)
        self.b1 = nn.Parameter(fc1.bias.detach()[sl].clone())
        self.w2 = nn.Parameter(fc2.weight.detach()[:, sl].clone())     # (h, 4h/t)
        self.b2 = nn.Parameter(fc2.bias.detach().clone())              # full, added once after the reduce
        ### END SOLUTION

    def local(self, x):
        """This rank's partial output (before the sum over ranks), no communication, no output bias."""
        ### BEGIN SOLUTION
        return F.linear(self.act(F.linear(x, self.w1, self.b1)), self.w2)
        ### END SOLUTION

    def forward(self, x):
        ### BEGIN SOLUTION
        return ReduceFromTP.apply(self.local(CopyToTP.apply(x))) + self.b2
        ### END SOLUTION


class TPAttention(nn.Module):
    """Head-parallel causal self-attention from a tinygpt CausalSelfAttention (fused qkv, no biases)."""

    def __init__(self, attn, rank, world):
        super().__init__()
        D, H = attn.cfg.d_model, attn.cfg.n_heads
        assert H % world == 0, "TP degree must divide the number of heads"
        self.hd, self.hl = D // H, H // world                         # head dim, heads on this rank
        ### BEGIN SOLUTION
        rows = slice(rank * self.hl * self.hd, (rank + 1) * self.hl * self.hd)
        wq, wk, wv = attn.qkv.weight.detach().split(D, dim=0)
        self.qkv = nn.Parameter(torch.cat([wq[rows], wk[rows], wv[rows]]).clone())   # (3D/t, D)
        self.proj = nn.Parameter(attn.proj.weight.detach()[:, rows].clone())        # (D, D/t)
        ### END SOLUTION

    def local(self, x):
        B, T, _ = x.shape
        ### BEGIN SOLUTION
        q, k, v = F.linear(x, self.qkv).split(self.hl * self.hd, dim=-1)
        q, k, v = (t.view(B, T, self.hl, self.hd).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return F.linear(y.transpose(1, 2).reshape(B, T, self.hl * self.hd), self.proj)
        ### END SOLUTION

    def forward(self, x):
        return ReduceFromTP.apply(self.local(CopyToTP.apply(x)))


class TPBlock(nn.Module):
    """x + attn(ln1(x)); x + mlp(ln2(x)) with TP attention/MLP and replicated LayerNorms."""

    def __init__(self, block, rank, world):
        super().__init__()
        self.ln1, self.ln2 = block.ln1, block.ln2                   # replicated (same on all ranks)
        self.attn = TPAttention(block.attn, rank, world)
        self.mlp = TPMLP(block.mlp, rank, world)

    def forward(self, x):
        ### BEGIN SOLUTION
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))
        ### END SOLUTION


# ---------------- sequence parallelism (section 3) ----------------
def _all_gather_seq(x):                    # (B, T/t, D) -> (B, T, D)
    COMM["all_gather"] += 1
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x.contiguous())
    return torch.cat(parts, dim=1)


def _reduce_scatter_seq(x):                # (B, T, D) partial sums -> (B, T/t, D) full sums of my chunk
    COMM["reduce_scatter"] += 1
    parts = [c.contiguous() for c in x.chunk(dist.get_world_size(), dim=1)]
    out = torch.empty_like(parts[0])
    dist.reduce_scatter(out, parts)
    return out


class GatherFromSP(torch.autograd.Function):
    """Enter the TP region: all-gather along the sequence in forward, reduce-scatter in backward."""

    @staticmethod
    def forward(ctx, x):
        ### BEGIN SOLUTION
        return _all_gather_seq(x)
        ### END SOLUTION

    @staticmethod
    def backward(ctx, grad):
        ### BEGIN SOLUTION
        return _reduce_scatter_seq(grad)
        ### END SOLUTION


class ReduceScatterToSP(torch.autograd.Function):
    """Leave the TP region: reduce-scatter along the sequence in forward, all-gather in backward."""

    @staticmethod
    def forward(ctx, x):
        ### BEGIN SOLUTION
        return _reduce_scatter_seq(x)
        ### END SOLUTION

    @staticmethod
    def backward(ctx, grad):
        ### BEGIN SOLUTION
        return _all_gather_seq(grad)
        ### END SOLUTION


class SPBlock(TPBlock):
    """Input and output are sequence shards (B, T/t, D); LayerNorm and residual run on the shard only."""

    def forward(self, x_shard):
        ### BEGIN SOLUTION
        h = self.attn.local(GatherFromSP.apply(self.ln1(x_shard)))
        x_shard = x_shard + ReduceScatterToSP.apply(h)
        h = self.mlp.local(GatherFromSP.apply(self.ln2(x_shard)))
        return x_shard + ReduceScatterToSP.apply(h) + self.mlp.b2
        ### END SOLUTION


def sync_sp_replicated_grads(block):
    """LayerNorm params are replicated but each rank saw only its sequence shard, so their grads are partial."""
    ### BEGIN SOLUTION
    for p in list(block.ln1.parameters()) + list(block.ln2.parameters()) + [block.mlp.b2]:
        dist.all_reduce(p.grad)
    ### END SOLUTION

# %% [markdown]
# The harness builds each module unsharded (same seed on every rank) and as its TP version, feeds identical inputs,
# backpropagates a random linear functional of the output, and compares:
# outputs, $\partial L/\partial x$, and each rank's weight-gradient shards against the matching slices of the full
# gradients.

# %%
# %%writefile runs_G1.13/check_tp.py
import copy
import json
import os

import torch
import torch.distributed as dist
import torch.nn as nn

import my_tp
from my_tp import SPBlock, TPAttention, TPBlock, TPMLP, sync_sp_replicated_grads
from prep.tinygpt import Block, CausalSelfAttention, GPTConfig

backend = os.environ.get("BACKEND", "nccl")
lr_ = int(os.environ["LOCAL_RANK"])
dev = torch.device("cuda", lr_) if backend == "nccl" else torch.device("cpu")
if backend == "nccl":
    torch.cuda.set_device(dev)
dist.init_process_group(backend, device_id=dev if backend == "nccl" else None)
rank, N = dist.get_rank(), dist.get_world_size()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

H = 4 if 4 % N == 0 else N                       # heads must be divisible by the TP degree
cfg = GPTConfig(d_model=16 * H, n_heads=H, n_layers=1, context_length=16 * N)
D, T, B = cfg.d_model, cfg.context_length, 2
res = {"N": N}


def maxerr(a, b):
    e = torch.tensor([(a - b).abs().max().item()], device=dev)
    dist.all_reduce(e, op=dist.ReduceOp.MAX)
    return e.item()


def run_pair(full, tp, x, seq_shard=False):
    """Return (full out, tp out, full x.grad, tp x.grad); loss = <out, R> with shared random R."""
    R = torch.randn(B, T, D, generator=torch.Generator().manual_seed(9)).to(dev)
    xf = x.clone().requires_grad_()
    yf = full(xf)
    (yf * R).sum().backward()
    if seq_shard:
        sl = slice(rank * T // N, (rank + 1) * T // N)
        xt = x[:, sl].clone().requires_grad_()
        yt = tp(xt)
        (yt * R[:, sl]).sum().backward()
        return yf[:, sl], yt, xf.grad[:, sl], xt.grad
    xt = x.clone().requires_grad_()
    yt = tp(xt)
    (yt * R).sum().backward()
    return yf, yt, xf.grad, xt.grad


def qkv_rows(W):                                  # rows of the fused (3D, D) qkv that belong to this rank
    hl, hd = cfg.n_heads // N, D // cfg.n_heads
    rows = slice(rank * hl * hd, (rank + 1) * hl * hd)
    return torch.cat([w[rows] for w in W.split(D, dim=0)])


x = torch.randn(B, T, D, generator=torch.Generator().manual_seed(1)).to(dev)
sl4 = lambda r: slice(r * 4 * D // N, (r + 1) * 4 * D // N)
cols = lambda r: slice(r * D // N, (r + 1) * D // N)

# MLP
torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(approximate="tanh"), nn.Linear(4 * D, D)).to(dev)
tmlp = TPMLP(copy.deepcopy(mlp), rank, N)
for k in my_tp.COMM:
    my_tp.COMM[k] = 0
yf, yt, gf, gt = run_pair(mlp, tmlp, x)
res["mlp_comm"] = dict(my_tp.COMM)
res["mlp"] = {"out": maxerr(yf, yt), "dx": maxerr(gf, gt),
              "dw1": maxerr(mlp[0].weight.grad[sl4(rank)], tmlp.w1.grad),
              "db1": maxerr(mlp[0].bias.grad[sl4(rank)], tmlp.b1.grad),
              "dw2": maxerr(mlp[2].weight.grad[:, sl4(rank)], tmlp.w2.grad),
              "db2": maxerr(mlp[2].bias.grad, tmlp.b2.grad)}

# attention
torch.manual_seed(0)
attn = CausalSelfAttention(cfg).to(dev)
tattn = TPAttention(copy.deepcopy(attn), rank, N)
yf, yt, gf, gt = run_pair(attn, tattn, x)
res["attn"] = {"out": maxerr(yf, yt), "dx": maxerr(gf, gt),
               "dqkv": maxerr(qkv_rows(attn.qkv.weight.grad), tattn.qkv.grad),
               "dproj": maxerr(attn.proj.weight.grad[:, cols(rank)], tattn.proj.grad)}

# full block, TP
torch.manual_seed(0)
blk = Block(cfg).to(dev)
tblk = TPBlock(copy.deepcopy(blk), rank, N)
for k in my_tp.COMM:
    my_tp.COMM[k] = 0
yf, yt, gf, gt = run_pair(blk, tblk, x)
res["block_comm"] = dict(my_tp.COMM)
res["block"] = {"out": maxerr(yf, yt), "dx": maxerr(gf, gt),
                "dln1": maxerr(blk.ln1.weight.grad, tblk.ln1.weight.grad),
                "dqkv": maxerr(qkv_rows(blk.attn.qkv.weight.grad), tblk.attn.qkv.grad),
                "dw1": maxerr(blk.mlp[0].weight.grad[sl4(rank)], tblk.mlp.w1.grad)}

# full block, TP + SP (input/output sharded along the sequence)
torch.manual_seed(0)
blk = Block(cfg).to(dev)
sblk = SPBlock(copy.deepcopy(blk), rank, N)
for k in my_tp.COMM:
    my_tp.COMM[k] = 0
yf, yt, gf, gt = run_pair(blk, sblk, x, seq_shard=True)
res["sp_comm"] = dict(my_tp.COMM)
ln_before = maxerr(blk.ln2.weight.grad, sblk.ln2.weight.grad)
sync_sp_replicated_grads(sblk)
res["sp"] = {"out": maxerr(yf, yt), "dx": maxerr(gf, gt),
             "dln1": maxerr(blk.ln1.weight.grad, sblk.ln1.weight.grad),
             "dln2_b": maxerr(blk.ln2.bias.grad, sblk.ln2.bias.grad),
             "db2": maxerr(blk.mlp[2].bias.grad, sblk.mlp.b2.grad),
             "dqkv": maxerr(qkv_rows(blk.attn.qkv.weight.grad), sblk.attn.qkv.grad),
             "dw2": maxerr(blk.mlp[2].weight.grad[:, sl4(rank)], sblk.mlp.w2.grad),
             "dln2_before_sync": ln_before}
if rank == 0:
    print(json.dumps(res, indent=1))
    json.dump(res, open("check_tp.json", "w"))
dist.destroy_process_group()

# %%
torchrun("check_tp.py")
r = load("check_tp.json")
for part in ("mlp", "attn", "block"):
    for k, v in r[part].items():
        check(f"TP {part}: {k} matches unsharded", v < 1e-5, f"max err {v:.2e}")
for k, v in r["sp"].items():
    if k != "dln2_before_sync":
        check(f"TP+SP block: {k} matches unsharded", v < 1e-5, f"max err {v:.2e}")
check("SP: LayerNorm grads are partial before the extra all-reduce", r["sp"]["dln2_before_sync"] > 1e-4)
check("TP MLP: 1 all-reduce forward + 1 backward", r["mlp_comm"]["all_reduce"] == 2, str(r["mlp_comm"]))
check("TP block: 2 all-reduces forward + 2 backward", r["block_comm"]["all_reduce"] == 4, str(r["block_comm"]))
check("TP+SP block: 4 all-gathers + 4 reduce-scatters, no all-reduce",
      r["sp_comm"] == {"all_reduce": 0, "all_gather": 4, "reduce_scatter": 4}, str(r["sp_comm"]))

# %% [markdown]
# ## 3. Sequence parallelism: the same bytes, $t\times$ less activation memory
# In plain TP the LayerNorms, residual adds and dropouts sit between $g$ and the next $f$, where activations are
# **replicated** on all $t$ ranks. Korthikanti et al. count $10\,sbh$ bytes per layer of such replicated
# activations (Table 2), not divided by $t$. These ops are independent along the sequence, so shard them there:
#
# ```
#   x (B, T/t, D) ─LN─> AG_seq ─> [TP attention, local heads] ─> partial (B,T,D) ─RS_seq─> (B,T/t,D) ─(+x)─> ...
#   forward:  all-gather in, reduce-scatter out      backward: reduce-scatter in, all-gather out (conjugates)
# ```
# An all-reduce *is* a reduce-scatter followed by an all-gather (G1.10), so TP+SP moves exactly the bytes TP moves:
# 4 all-gathers + 4 reduce-scatters per layer instead of 4 all-reduces. Activation memory per layer drops from
# $sbh(10 + 24/t + 5as/(ht))$ to $sbh(34/t + 5as/(ht))$.
#
# The bug the checks above catch: the LayerNorm weights are replicated, but each rank's LN saw only $T/t$ tokens,
# so its gradient is a **partial sum** and must be all-reduced across the TP group (Megatron does this for all
# "sequence-parallel" parameters). The same holds for the row-parallel output bias, which is added on the shard.
#
# ## 4. Why TP stays inside a node
# Per layer and microbatch: compute is $\approx 24\,bsh^2$ FLOPs forward (matmuls only), so $72\,bsh^2$ forward +
# backward, split over $t$ GPUs. Communication is 4 ring all-reduces of $2bsh$ bytes (bf16), each costing
# $2\frac{t-1}{t}\cdot 2bsh/W$ at link bandwidth $W$, and none of them overlaps easily (the next matmul needs the
# result). So
# $$\frac{T_\text{comm}}{T_\text{comp}} = \frac{4\cdot 2\frac{t-1}{t}\cdot 2bsh / W}{72\,bsh^2/(tC)} = \frac{2(t-1)\,C}{9\,h\,W}.$$
# It is independent of batch and sequence length. It grows with $t$ and with $C/W$, and shrinks with model width $h$.
# **Your task:** implement it, then evaluate it for an H100 node (NVLink) vs across nodes (IB) vs this box.

# %%
def tp_comm_compute_ratio(h, t, C, W):
    """T_comm / T_comp for one transformer layer under t-way Megatron TP (bf16 activations)."""
    ### BEGIN SOLUTION
    return 2 * (t - 1) * C / (9 * h * W)
    ### END SOLUTION


def brute(h, t, C, W, b=4, s=4096):          # explicit sums, independent of the closed form
    comp = 3 * (2 * b * s * h * 3 * h + 2 * b * s * h * h + 2 * b * s * h * 4 * h * 2) / t / C   # qkv, proj, mlp (x3 fwd+bwd)
    comm = 4 * (2 * (t - 1) / t) * (2 * b * s * h) / W
    return comm / comp


for args in [(8192, 8, 990e12, 450e9), (4096, 2, 71e12, 3.5e9), (12288, 16, 990e12, 50e9)]:
    check_close(f"ratio formula h={args[0]} t={args[1]}", tp_comm_compute_ratio(*args), brute(*args))
for name, h, t, C, W in [("H100 node, NVLink 450 GB/s", 8192, 8, 990e12, 450e9),
                         ("H100 TP=16 across 2 nodes, IB 50 GB/s", 8192, 16, 990e12, 50e9),
                         ("RTX 3090 x2, PCIe/SHM ~3.5 GB/s (G1.10)", 4096, 2, 71e12, 3.5e9)]:
    print(f"{name:42s} h={h:5d} t={t:2d}: T_comm/T_comp = {tp_comm_compute_ratio(h, t, C, W):6.2f}")

# %% [markdown]
# Inside an H100 node with $h = 8192$ the ratio is ≈ 0.4: communication is significant but smaller than compute,
# and partial overlap (e.g. splitting the matmul and the collective into chunks) hides much of it. Stretch the TP
# group across nodes and the link drops 9×, with ratio ≈ 8 at $t=16$: GPUs mostly wait. This is Narayanan et al.'s
# Takeaway #1: TP degree ≤ GPUs per node, and pipeline or data parallelism across nodes. Now measure it on this box.

# %%
# %%writefile runs_G1.13/bench_tp.py
import copy
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from my_tp import TPMLP

backend = os.environ.get("BACKEND", "nccl")
lr_ = int(os.environ["LOCAL_RANK"])
dev = torch.device("cuda", lr_) if backend == "nccl" else torch.device("cpu")
if backend == "nccl":
    torch.cuda.set_device(dev)
dist.init_process_group(backend, device_id=dev if backend == "nccl" else None)
rank, N = dist.get_rank(), dist.get_world_size()
cuda = dev.type == "cuda"
D, T = (4096, 2048) if cuda else (512, 256)
dt = torch.bfloat16 if cuda else torch.float32


def timeit(fn, iters=10):
    for _ in range(3):
        fn()
    if cuda:
        torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t = torch.tensor([(time.perf_counter() - t0) / iters], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(approximate="tanh"), nn.Linear(4 * D, D)).to(dev, dt)
tp = TPMLP(copy.deepcopy(mlp), rank, N)
x = torch.randn(1, T, D, device=dev, dtype=dt, requires_grad=True)
act = torch.randn(1, T, D, device=dev, dtype=dt)


def step(m):
    def f():
        m(x).sum().backward()
    return f


res = {"N": N, "D": D, "T": T, "t_single": timeit(step(mlp)), "t_tp": timeit(step(tp)),
       "t_allreduce": timeit(lambda: dist.all_reduce(act))}
if rank == 0:
    print(json.dumps(res))
    json.dump(res, open("bench_tp.json", "w"))
dist.destroy_process_group()

# %%
torchrun("bench_tp.py")
bt = load("bench_tp.json")
print(f"MLP h={bt['D']}, {bt['T']} tokens, bf16, fwd+bwd: 1 GPU {bt['t_single']*1e3:.1f} ms | "
      f"TP={bt['N']} {bt['t_tp']*1e3:.1f} ms | one all-reduce of the activation {bt['t_allreduce']*1e3:.2f} ms")
model_t = bt["t_single"] / bt["N"] + 2 * bt["t_allreduce"]
print(f"model: t_single/t + 2 all-reduces = {model_t*1e3:.1f} ms  (measured {bt['t_tp']*1e3:.1f} ms)")
check("TP time is explained by compute/t + 2 blocking all-reduces (within 50%)",
      abs(model_t - bt["t_tp"]) / bt["t_tp"] < 0.5, f"model {model_t:.4f} vs {bt['t_tp']:.4f}")

# %% [markdown]
# On this PCIe box, TP over 2+ GPUs is far from the ideal $t\times$ speedup for this layer (at best a little faster
# than one GPU), because each all-reduce of the 16 MB activation costs several ms at ~3.5 GB/s. The same all-reduce over NVSwitch takes
# ≈ $2\cdot\frac78\cdot 16.8\,\text{MB}/370\,\text{GB/s} \approx 80\,\mu s$. TP exists to fit and speed up layers
# *inside* a high-bandwidth domain.
#
# ## Say it out loud
# 1. Derive why the MLP is split column-then-row, and what $f$ and $g$ do in forward and backward.
# 2. How many all-reduces per transformer layer does Megatron TP need (forward and backward), and of what size?
#    Why can't they easily overlap with compute like DDP's gradient all-reduce?
# 3. How do you split attention under TP? What constraint does it put on the TP degree, and what about GQA with
#    fewer KV heads than $t$?
# 4. Where does the row-parallel bias go, and what goes wrong if every rank adds it?
# 5. What is sequence parallelism, what does it save, and why does it cost no extra bandwidth? Which gradients need
#    an extra all-reduce?
# 6. Derive $T_\text{comm}/T_\text{comp}$ for TP. Why is TP kept within a node? What do you use across nodes instead?
# 7. TP shards weights and activations, ZeRO-3 shards weights. Compare their communication volume and what each
#    one scales with.

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. GeLU is elementwise, so if $A$ is split by columns each rank can apply GeLU to its own slice $XA_i$ without
#    communication. Splitting $A$ by rows would produce partial sums that must be reduced *before* the nonlinearity.
#    $B$ is then split by rows to consume $Y_i$ directly, leaving one sum at the end. $f$ marks where a replicated
#    input enters: identity forward, all-reduce backward (gradient contributions from every rank's use of $X$ add
#    up). $g$ marks the exit: all-reduce forward (sum partial outputs), identity backward (each rank needs the full
#    upstream gradient, which it already has). They are conjugates (Shoeybi et al. 2019 §3).
# 2. Two forward (after attention and after MLP) and two backward (the $f$'s), each an all-reduce of $b\cdot s\cdot h$
#    activations (2 bytes each in bf16). They are on the critical path: the residual add or LayerNorm right after
#    $g$ needs the full sum. DDP's gradient all-reduce has the rest of backward to hide under. Overlap for TP
#    requires splitting the GEMM and the collective into chunks (async TP, collective matmul), which is not free.
# 3. Shard heads: Q/K/V projections column-parallel by head, attention local per head, output projection
#    row-parallel. $t$ must divide the number of heads ($H/t$ heads per rank). With GQA, $t$ should divide the KV
#    head count. If $t$ > KV heads, KV heads are replicated across ranks (more memory and compute for K/V).
# 4. Add it once, after the reduction ($g$). If each rank adds $b$ before the all-reduce, the output gets $t\cdot b$.
#    The forward check catches it. Its gradient is the full upstream gradient sum, identical on all ranks under TP,
#    but partial under SP (sharded tokens), so it must be all-reduced there.
# 5. SP shards the activations outside the TP regions (LayerNorm, dropout, residual) along the sequence: the
#    all-reduce becomes reduce-scatter (exit) + all-gather (entry), and in backward the roles swap. An all-reduce
#    *is* RS + AG, so the bytes are identical. Memory for those activations drops by $t$ (Korthikanti et al.:
#    $10sbh \to 10sbh/t$ per layer). Replicated parameters inside the sequence-sharded region (LayerNorm weights and
#    biases, the row-parallel output bias) get partial gradients and need an all-reduce over the TP group.
# 6. Per layer: $T_\text{comp} = 72bsh^2/(tC)$, $T_\text{comm} = 4\cdot2\frac{t-1}{t}\cdot2bsh/W$, ratio
#    $2(t-1)C/(9hW)$. On H100 NVLink ($W$ = 450 GB/s) at $h = 8192$, $t = 8$ it is ≈ 0.4. Across nodes ($W$ = 50 GB/s
#    IB per GPU) it is ≈ 4, so communication-bound. Across nodes use pipeline parallelism (point-to-point
#    activations at stage boundaries only) and data parallelism / FSDP (overlappable, once per step). Megatron 2021
#    Takeaway #1; Llama 3 used TP=8 within a node.
# 7. ZeRO-3 moves $3\Psi$ parameter-sized traffic per step, independent of batch size, and can be prefetched and
#    overlapped. TP moves activation-sized traffic, $\propto b\cdot s\cdot h$ per layer, on the critical path, and
#    independent of parameter count. FSDP gets relatively cheaper as the per-GPU batch grows. TP is the tool when a
#    single layer's weights or activations are too big, or the per-GPU batch is too small for FSDP to hide its
#    communication.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
