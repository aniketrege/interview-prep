# %% [markdown]
# # G1.15 · Context and expert parallelism: ring attention, Ulysses, and all-to-all MoE dispatch
#
# **ID:** G1.15 · **Target:** 120 min · **Plan week:** 15 · **Part:** G1
#
# **The lab.** Two parallelism axes that exist because of two model features: long sequences and mixture-of-experts.
# 1. **Ring attention (context parallelism).** Shard the sequence across $N$ ranks. Each rank keeps its query
#    block and passes key/value blocks around a ring with `send`/`recv`, combining partial results with the
#    log-sum-exp merge from online softmax (the FlashAttention idea, G1.04). Verify against full attention:
#    non-causal (required), then causal. Measure the causal **load imbalance** and fix it with **zig-zag** sharding.
# 2. **DeepSpeed-Ulysses.** Swap sequence-sharding for head-sharding with one all-to-all, attend locally, swap back.
#    Verify it, and state its constraint (heads divisible by the degree).
# 3. **Expert parallelism.** A top-$k$ router with capacity factor and token dropping, experts sharded across
#    ranks, and **all-to-all dispatch/combine with variable splits** (`dist.all_to_all_single`). Check it against a
#    single-process MoE, and implement Switch's load-balancing loss.
#
# ## Sources
# - H. Liu, M. Zaharia, P. Abbeel, *Ring Attention with Blockwise Transformers for Near-Infinite Context*, ICLR
#   2024, arXiv:2310.01889 — https://arxiv.org/abs/2310.01889 (KV blocks rotate around a ring, overlapped with
#   blockwise attention).
# - M. Milakov, N. Gimelshein, *Online normalizer calculation for softmax*, 2018, arXiv:1805.02867, and T. Dao et
#   al., *FlashAttention*, 2022, arXiv:2205.14135 (the running max / log-sum-exp merge).
# - W. Brandon et al., *Striped Attention: Faster Ring Attention for Causal Transformers*, 2023, arXiv:2311.09431
#   (causal load imbalance; striping). Zig-zag sharding: zhuzilin/ring-flash-attention
#   (https://github.com/zhuzilin/ring-flash-attention), which picotron's context parallelism credits; the Llama 3
#   paper §3.3.2 also splits the sequence into $2\times$CP chunks for load balancing (arXiv:2407.21783).
# - S. A. Jacobs, M. Tanaka, C. Zhang, M. Zhang, S. L. Song, S. Rajbhandari, Y. He, *DeepSpeed Ulysses: System
#   Optimizations for Enabling Training of Extreme Long Sequence Transformer Models*, 2023, arXiv:2309.14509 —
#   https://arxiv.org/abs/2309.14509 (all-to-all sequence↔head re-sharding; constant communication volume when $N$
#   and sequence length grow together).
# - D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, Z. Chen, *GShard: Scaling Giant
#   Models with Conditional Computation and Automatic Sharding*, 2020, arXiv:2006.16668 (top-2 gating, expert
#   capacity, per-group dispatch with all-to-all, local groups).
# - W. Fedus, B. Zoph, N. Shazeer, *Switch Transformers*, JMLR 2022, arXiv:2101.03961 — https://arxiv.org/abs/2101.03961
#   (top-1 routing; expert capacity = tokens per batch / experts × capacity factor; dropped tokens skip via the
#   residual; auxiliary loss $\alpha N\sum_i f_i P_i$, $\alpha = 10^{-2}$).
# - Hugging Face picotron, `picotron/context_parallel/context_parallel.py` (ring attention forward with
#   `update_out_and_lse`) and `cp_communications.py` (`batch_isend_irecv` ring), huggingface/picotron@59714b1,
#   Apache-2.0. Our ring loop and merge adapt these.
# - N. Tazi et al., *The Ultra-Scale Playbook*, Hugging Face 2025, §"Context Parallelism" (ring attention, causal
#   imbalance, zig-zag) and §"Expert parallelism": https://huggingface.co/spaces/nanotron/ultrascale-playbook

# %%
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from prep import Session, check, check_close

s = Session("G1.15", target_min=120)
NGPU = torch.cuda.device_count()
NPROC = min(4, NGPU) if NGPU >= 2 else 2
BACKEND = "nccl" if NGPU >= 2 else "gloo"
RUN = Path("runs_G1.15")
RUN.mkdir(exist_ok=True)


def torchrun(script, *args, nproc=NPROC, timeout=600):
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

# %% [markdown]
# ## 1. Ring attention
# Sequence of length $T$, $N$ ranks, rank $r$ holds rows $Q_r, K_r, V_r$ of length $T/N$. Attention for its
# queries needs **all** keys and values: $O_r = \mathrm{softmax}(Q_rK^\top/\sqrt d)V$. Split $K, V$ into the $N$
# blocks. For each block $j$ compute the block-local result and its log-sum-exp
# $$o_j = \mathrm{softmax}(S_j)V_j,\quad \ell_j = \log\sum_c e^{S_{j,c}} \qquad (S_j = Q_rK_j^\top/\sqrt d,\ \text{row-wise}).$$
# Two partial results merge exactly (online softmax, G1.04):
# $$\ell = \log(e^{\ell_a} + e^{\ell_b}),\qquad o = e^{\ell_a - \ell}\,o_a + e^{\ell_b - \ell}\,o_b.$$
# The ring: at step $t = 0..N-1$ rank $r$ holds the KV block of rank $(r - t) \bmod N$. It **starts sending it to
# $r+1$ and receiving the next one from $r-1$**, computes on the current block while the transfer is in flight, then
# waits. After $N$ steps every query has seen every key. Each rank sends $(N-1)$ KV blocks ($2\cdot\frac{T}{N}d$
# values each per head) and computes $N$ block products. Compute per step scales as $(T/N)^2$ and communication as
# $T/N$, so with long enough blocks the transfer hides under compute (Liu et al.).
#
# **Causal.** With contiguous shards, block $j$ is fully visible to rank $r$ if $j < r$, needs the triangular
# mask if $j = r$, and is fully masked if $j > r$ (skip it). Rank $N-1$ computes $N$ blocks and rank 0 one, so the
# step time is set by the last rank. **Zig-zag**: split the sequence into $2N$ chunks and give rank $r$ chunks $r$
# and $2N-1-r$. Every rank then has one early and one late chunk, and equal work.
#
# **Your task:** `block_attn` (with optional boolean mask, returns `out, lse`), `merge`, and the ring loop.
# Use `float('-inf')` masking and handle rows that are fully masked within a block: their lse is $-\infty$, and the
# merge must ignore them (`torch.logaddexp` and `exp(-inf)=0` do this as long as you never compute `-inf - -inf`).

# %%
# %%writefile runs_G1.15/ring_attn.py
import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

backend = os.environ.get("BACKEND", "nccl")
lr_ = int(os.environ["LOCAL_RANK"])
dev = torch.device("cuda", lr_) if backend == "nccl" else torch.device("cpu")
if backend == "nccl":
    torch.cuda.set_device(dev)
dist.init_process_group(backend, device_id=dev if backend == "nccl" else None)
rank, N = dist.get_rank(), dist.get_world_size()
torch.backends.cuda.matmul.allow_tf32 = False


def block_attn(q, k, v, mask=None):
    """q (B,H,Tq,d), k/v (B,H,Tk,d), mask (Tq,Tk) bool True = keep. Returns out (B,H,Tq,d), lse (B,H,Tq)."""
    ### BEGIN SOLUTION
    s_ = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        s_ = s_.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(s_, dim=-1)
    p = torch.exp(s_ - lse[..., None].clamp_min(-1e30))       # fully masked rows: exp(-inf) = 0
    return p @ v, lse
    ### END SOLUTION


def merge(out, lse, out_b, lse_b):
    """Combine two partial softmax-attention results over disjoint key sets."""
    ### BEGIN SOLUTION
    new = torch.logaddexp(lse, lse_b)
    safe = new.clamp_min(-1e30)                               # rows with no keys yet stay at -inf / 0
    out = torch.exp(lse - safe)[..., None] * out + torch.exp(lse_b - safe)[..., None] * out_b
    return out, new
    ### END SOLUTION


def ring_attention(q, k, v, causal=False):
    """q, k, v: this rank's contiguous sequence block. Returns this rank's output block and #blocks computed."""
    B, H, Tl, d = q.shape
    out = torch.zeros_like(q)
    lse = torch.full((B, H, Tl), float("-inf"), device=q.device, dtype=q.dtype)
    computed = 0
    ### BEGIN SOLUTION
    for t in range(N):
        src = (rank - t) % N                                  # whose KV block we hold now
        if t < N - 1:                                         # start passing it on while we compute
            nk, nv = torch.empty_like(k), torch.empty_like(v)
            reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, k, (rank + 1) % N),
                                           dist.P2POp(dist.isend, v, (rank + 1) % N),
                                           dist.P2POp(dist.irecv, nk, (rank - 1) % N),
                                           dist.P2POp(dist.irecv, nv, (rank - 1) % N)])
        if not causal or src < rank:
            o_b, l_b = block_attn(q, k, v)
        elif src == rank:
            o_b, l_b = block_attn(q, k, v, torch.ones(Tl, Tl, dtype=torch.bool, device=q.device).tril())
        else:
            o_b = None                                        # fully masked: skip the work
        if o_b is not None:
            out, lse = merge(out, lse, o_b, l_b)
            computed += 1
        if t < N - 1:
            for r_ in reqs:
                r_.wait()
            k, v = nk, nv
    ### END SOLUTION
    return out, computed


def seq_to_heads(x):
    """Ulysses all-to-all: (B, H, T/N, d) sequence-sharded -> (B, H/N, T, d) head-sharded."""
    B, H, Tl, d = x.shape
    ### BEGIN SOLUTION
    send = x.reshape(B, N, H // N, Tl, d).transpose(0, 1).contiguous()   # chunk j = heads for rank j
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)                                   # recv[j] = my heads, seq block of rank j
    return recv.permute(1, 2, 0, 3, 4).reshape(B, H // N, N * Tl, d)
    ### END SOLUTION


def heads_to_seq(x):
    """Inverse: (B, H/N, T, d) head-sharded -> (B, H, T/N, d) sequence-sharded."""
    B, Hl, T, d = x.shape
    ### BEGIN SOLUTION
    send = x.reshape(B, Hl, N, T // N, d).permute(2, 0, 1, 3, 4).contiguous()  # chunk j = seq block j
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)                                          # recv[j] = rank j's heads
    return recv.transpose(0, 1).reshape(B, N * Hl, T // N, d)
    ### END SOLUTION


def ulysses_attention(q, k, v, causal=False):
    q, k, v = (seq_to_heads(t) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return heads_to_seq(o)


# ---- harness: full tensors from a shared seed, each rank takes its block ----
B, H, d, Tl = 2, 2 * N, 32, 48
T = N * Tl
g = torch.Generator().manual_seed(0)
Q, K, V = (torch.randn(B, H, T, d, generator=g).to(dev) for _ in range(3))
sl = slice(rank * Tl, (rank + 1) * Tl)
res = {"N": N}
for causal in (False, True):
    ref = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)[:, :, sl]
    o, nblk = ring_attention(Q[:, :, sl].contiguous(), K[:, :, sl].contiguous(), V[:, :, sl].contiguous(), causal)
    e = torch.tensor([(o - ref).abs().max().item()], device=dev)
    dist.all_reduce(e, op=dist.ReduceOp.MAX)
    counts = [torch.zeros(1, device=dev) for _ in range(N)]
    dist.all_gather(counts, torch.tensor([float(nblk)], device=dev))
    res[f"ring_causal={causal}"] = {"err": e.item(), "blocks_per_rank": [int(c.item()) for c in counts]}
    o = ulysses_attention(Q[:, :, sl].contiguous(), K[:, :, sl].contiguous(), V[:, :, sl].contiguous(), causal)
    e = torch.tensor([(o - ref).abs().max().item()], device=dev)
    dist.all_reduce(e, op=dist.ReduceOp.MAX)
    res[f"ulysses_causal={causal}"] = e.item()
o, l = block_attn(Q[:, :, :4], K[:, :, :4], V[:, :, :4], torch.zeros(4, 4, dtype=torch.bool, device=dev))
oa, la = merge(o, l, *block_attn(Q[:, :, :4], K, V))
ref_rows = F.scaled_dot_product_attention(Q[:, :, :4], K, V)
res["merge_with_fully_masked_block"] = (oa - ref_rows).abs().max().item()
if rank == 0:
    print(json.dumps(res, indent=1))
    json.dump(res, open("ring.json", "w"))
dist.destroy_process_group()

# %%
torchrun("ring_attn.py")
r = load("ring.json")
N = r["N"]
check("ring attention (non-causal) == full attention", r["ring_causal=False"]["err"] < 1e-5, str(r["ring_causal=False"]))
check("ring attention (causal) == full causal attention", r["ring_causal=True"]["err"] < 1e-5, str(r["ring_causal=True"]))
check("merging a fully-masked block is a no-op (no NaNs)", r["merge_with_fully_masked_block"] < 1e-5)
check("non-causal: every rank computes N blocks", r["ring_causal=False"]["blocks_per_rank"] == [N] * N)
check("causal + contiguous shards: rank r computes r+1 blocks (imbalance)",
      r["ring_causal=True"]["blocks_per_rank"] == list(range(1, N + 1)))
for c in (False, True):
    check(f"Ulysses (causal={c}) == full attention", r[f"ulysses_causal={c}"] < 1e-5, str(r[f"ulysses_causal={c}"]))

# %% [markdown]
# **Load balance, counted exactly.** For causal attention, the work of rank $r$ is the number of unmasked
# (query, key) pairs among its queries: query position $i$ attends to $i+1$ keys. **Your task:** write the
# zig-zag assignment and compare per-rank work for contiguous vs zig-zag shards.

# %%
def contiguous_positions(T, n, r):
    return np.arange(r * T // n, (r + 1) * T // n)


def zigzag_positions(T, n, r):
    """Split [0, T) into 2n equal chunks; rank r gets chunks r and 2n-1-r."""
    ### BEGIN SOLUTION
    c = T // (2 * n)
    return np.concatenate([np.arange(r * c, (r + 1) * c), np.arange((2 * n - 1 - r) * c, (2 * n - r) * c)])
    ### END SOLUTION


T_, n_ = 4096, 8
work = lambda pos: int((pos + 1).sum())          # causal: query i sees keys 0..i
w_con = [work(contiguous_positions(T_, n_, r)) for r in range(n_)]
w_zz = [work(zigzag_positions(T_, n_, r)) for r in range(n_)]
check("zig-zag covers every position exactly once",
      sorted(np.concatenate([zigzag_positions(T_, n_, r) for r in range(n_)]).tolist()) == list(range(T_)))
check("zig-zag gives every rank the same causal work", len(set(w_zz)) == 1, str(w_zz))
print(f"contiguous: max/min work = {max(w_con)/min(w_con):.1f}x  -> step time set by rank {n_-1}")
print(f"zig-zag:    max/min work = {max(w_zz)/min(w_zz):.1f}x;  speedup of the slowest rank: {max(w_con)/max(w_zz):.2f}x")

# %% [markdown]
# **Ring vs Ulysses vs all-gather KV.**
# - *Ring*: per rank, $(N-1)$ KV blocks of $2\frac{T}{N}h$ values ≈ $2Th$ in total, regardless of $N$.
#   Point-to-point traffic overlaps with block compute. Any number of heads works.
# - *Ulysses*: 4 all-to-alls per attention (q, k, v, out), each moving $\frac{N-1}{N}$ of a $\frac{T}{N}h$
#   tensor per rank, so per-rank volume $\approx 4Th/N$, *decreasing* with $N$ (constant if $T$ grows with $N$:
#   Jacobs et al.). Attention itself is unmodified (any kernel, e.g. FlashAttention). Constraint: $N$ must
#   divide the number of heads (KV heads with GQA), which caps the degree at $H$, and all-to-all needs a
#   high-bisection network (in practice, inside a node).
# - *All-gather KV*: Llama 3's CP gathers K and V (small thanks to GQA) and then computes local attention, also
#   with $2\times$CP chunks for causal balance (Llama 3 §3.3.2).
# These compose: Ulysses inside a node × ring across nodes ("2-D" / hybrid CP).
#
# ## 2. Expert parallelism
# An MoE layer replaces the MLP with $E$ expert MLPs and a router $W_r \in \mathbb R^{h\times E}$:
# $p(x) = \mathrm{softmax}(xW_r)$, each token goes to its top-$k$ experts, and
# $y = \sum_{e\in\text{top-}k} p_e(x)\,\text{expert}_e(x)$. Here the gates are the softmax probabilities without
# renormalization (Switch for $k=1$; some models renormalize the top-$k$ gates).
#
# **Capacity.** To give every expert a static buffer size, expert $e$ accepts at most
# $C = \lceil \text{cf}\cdot k \cdot T_\text{group}/E\rceil$ tokens per group (Switch eq. 3, GShard groups; here a
# group is one rank's local tokens). Assignments beyond $C$ are **dropped**: they contribute nothing, and the token
# passes through the residual connection. Priority: all first choices (in token order) before any second choices,
# as in GShard's top-2.
#
# **Expert parallelism (EP):** rank $r$ owns experts $[rE/N, (r+1)E/N)$. Forward:
# ```
#  route locally ─> sort kept (token, expert) pairs by expert ─> counts per destination rank
#  all_to_all(counts)                   # everyone learns how much it will receive from whom
#  all_to_all_single(x[sorted], in_splits=send_counts, out_splits=recv_counts)       # dispatch
#  run local experts on what arrived (grouped by local expert id)
#  all_to_all_single(results, in_splits=recv_counts, out_splits=send_counts)         # combine (reverse)
#  y.index_add_(token ids, gate * result)
# ```
# The splits are **variable**: they depend on the routing, which is why the counts are exchanged first.
# **Your task:** `route` (top-$k$ + capacity + dropping) and `ep_moe_forward`. The harness checks every rank's
# output against an independent dense single-process implementation (same groups, same capacity rule).

# %%
# %%writefile runs_G1.15/moe_ep.py
import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn

backend = os.environ.get("BACKEND", "nccl")
lr_ = int(os.environ["LOCAL_RANK"])
dev = torch.device("cuda", lr_) if backend == "nccl" else torch.device("cpu")
if backend == "nccl":
    torch.cuda.set_device(dev)
dist.init_process_group(backend, device_id=dev if backend == "nccl" else None)
rank, N = dist.get_rank(), dist.get_world_size()
torch.backends.cuda.matmul.allow_tf32 = False


def route(x, router, k, cf, E):
    """Top-k routing with capacity C = ceil(cf * k * T / E) per expert for these T tokens.
    Priority: slot 0 (all tokens, in order), then slot 1, ... Returns kept (token_idx, expert_idx, gate), probs."""
    T = x.shape[0]
    probs = torch.softmax(router(x), dim=-1)                    # (T, E)
    C = math.ceil(cf * k * T / E)
    ### BEGIN SOLUTION
    gate, expert = probs.topk(k, dim=-1)                        # (T, k)
    tok = torch.arange(T, device=x.device).repeat(k)            # slot-major order: slot 0 for all, then slot 1
    exp_ = expert.t().reshape(-1)
    gat = gate.t().reshape(-1)
    onehot = torch.nn.functional.one_hot(exp_, E)
    pos = (onehot.cumsum(0) * onehot).sum(-1) - 1               # position of each assignment within its expert
    keep = pos < C
    return tok[keep], exp_[keep], gat[keep], probs
    ### END SOLUTION


def ep_moe_forward(x, router, local_experts, k, cf, E):
    """x: (T, h) this rank's tokens. local_experts: experts [rank*E/N, (rank+1)*E/N). Returns (T, h)."""
    El = E // N
    tok, exp_, gate, _ = route(x, router, k, cf, E)
    ### BEGIN SOLUTION
    order = torch.argsort(exp_, stable=True)
    tok, exp_, gate = tok[order], exp_[order], gate[order]
    send_counts = torch.bincount(exp_ // El, minlength=N)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    sc, rc = send_counts.tolist(), recv_counts.tolist()
    x_recv = torch.empty(sum(rc), x.shape[1], device=x.device, dtype=x.dtype)
    dist.all_to_all_single(x_recv, x[tok].contiguous(), output_split_sizes=rc, input_split_sizes=sc)
    e_recv = torch.empty(sum(rc), device=x.device, dtype=exp_.dtype)
    dist.all_to_all_single(e_recv, (exp_ % El).contiguous(), output_split_sizes=rc, input_split_sizes=sc)
    out = torch.zeros_like(x_recv)
    for j, expert in enumerate(local_experts):
        m = e_recv == j
        if m.any():
            out[m] = expert(x_recv[m])
    back = torch.empty(sum(sc), x.shape[1], device=x.device, dtype=x.dtype)
    dist.all_to_all_single(back, out, output_split_sizes=sc, input_split_sizes=rc)
    y = torch.zeros_like(x)
    y.index_add_(0, tok, gate[:, None] * back)
    return y
    ### END SOLUTION


def moe_reference(groups, router, experts, k, cf):
    """Dense single-process MoE: per group, loop over experts; capacity by first-come order (slot-major)."""
    E, outs, stats = len(experts), [], []
    for x in groups:
        T = x.shape[0]
        C = math.ceil(cf * k * T / E)
        p = torch.softmax(router(x), -1)
        gate, ex = p.topk(k, -1)
        y = torch.zeros_like(x)
        used = [0] * E
        dropped = 0
        for slot in range(k):
            for t in range(T):
                e = int(ex[t, slot])
                if used[e] < C:
                    used[e] += 1
                    y[t] += gate[t, slot] * experts[e](x[t:t + 1])[0]
                else:
                    dropped += 1
        outs.append(y)
        stats.append({"tokens_per_expert": used, "dropped": dropped, "capacity": C})
    return outs, stats


h, E, T = 16, 2 * N, 24
torch.manual_seed(0)
router = nn.Linear(h, E, bias=False).to(dev)
experts = nn.ModuleList(nn.Sequential(nn.Linear(h, 32), nn.GELU(), nn.Linear(32, h)) for _ in range(E)).to(dev)
g = torch.Generator().manual_seed(1)
groups = [torch.randn(T, h, generator=g).to(dev) * (1 + r) for r in range(N)]   # different data per rank
mine = groups[rank]
local = experts[rank * (E // N):(rank + 1) * (E // N)]
res = {"N": N, "E": E}
with torch.no_grad():
    for k, cf in [(1, 1.0), (2, 1.25), (2, 0.5), (1, 4.0)]:
        y = ep_moe_forward(mine, router, local, k, cf, E)
        ref, stats = moe_reference(groups, router, experts, k, cf)
        e = torch.tensor([(y - ref[rank]).abs().max().item()], device=dev)
        dist.all_reduce(e, op=dist.ReduceOp.MAX)
        res[f"k={k},cf={cf}"] = {"err": e.item(), "dropped_per_rank": [st["dropped"] for st in stats],
                                 "capacity": stats[0]["capacity"],
                                 "tokens_per_expert_rank0": stats[0]["tokens_per_expert"]}
if rank == 0:
    print(json.dumps(res, indent=1))
    json.dump(res, open("moe.json", "w"))
dist.destroy_process_group()

# %%
torchrun("moe_ep.py")
mo = load("moe.json")
for key, v in mo.items():
    if key.startswith("k="):
        check(f"EP MoE ({key}) == single-process MoE", v["err"] < 1e-5, f"err {v['err']:.2e}")
check("cf=0.5 actually drops tokens (the test exercises dropping)", sum(mo["k=2,cf=0.5"]["dropped_per_rank"]) > 0)
check("cf=4 drops nothing", sum(mo["k=1,cf=4.0"]["dropped_per_rank"]) == 0)
for key in ("k=1,cf=1.0", "k=2,cf=0.5"):
    print(key, "capacity", mo[key]["capacity"], "tokens per expert (rank 0's group):",
          mo[key]["tokens_per_expert_rank0"], "dropped per group:", mo[key]["dropped_per_rank"])

# %% [markdown]
# **Load-balancing loss (Switch eq. 4-6).** With $T$ tokens and $E$ experts, $f_e$ = fraction of tokens whose
# top-1 expert is $e$ (not differentiable), and $P_e$ = mean router probability for $e$ (differentiable):
# $$\mathcal L_\text{aux} = \alpha\, E \sum_{e} f_e P_e,\qquad \alpha = 10^{-2}.$$
# Under uniform routing $f_e = P_e = 1/E$, so $\mathcal L_\text{aux} = \alpha$. The factor $E$ keeps that value
# independent of $E$. If everything collapses onto one expert, it is $\alpha E$. The gradient flows through $P$ and
# pushes probability away from overloaded experts. **Your task:** implement it.

# %%
def switch_aux_loss(probs, alpha=1e-2):
    """probs: (T, E) router softmax. Returns alpha * E * sum_e f_e * P_e with f from the top-1 assignment."""
    ### BEGIN SOLUTION
    T, E = probs.shape
    f = torch.bincount(probs.argmax(-1), minlength=E).float() / T
    P = probs.mean(0)
    return alpha * E * (f * P).sum()
    ### END SOLUTION


E_ = 8
uniform = torch.full((64, E_), 1 / E_) + torch.eye(E_).repeat(8, 1) * 1e-4      # balanced argmax, ~uniform probs
check_close("aux loss under uniform routing = alpha", switch_aux_loss(uniform), 1e-2, rtol=1e-3)
collapsed = torch.zeros(64, E_)
collapsed[:, 3] = 1.0
check_close("aux loss when all tokens go to one expert = alpha * E", switch_aux_loss(collapsed), 1e-2 * E_)
pr = torch.softmax(torch.randn(100, E_), -1)
brute = 1e-2 * E_ * sum((pr.argmax(-1) == e).float().mean() * pr[:, e].mean() for e in range(E_))
check_close("aux loss matches a brute-force loop", switch_aux_loss(pr), brute)
lg = torch.randn(100, E_, requires_grad=True)
switch_aux_loss(torch.softmax(lg, -1)).backward()
check("aux loss is differentiable through P (router logits get a gradient)", lg.grad.abs().sum() > 0)

# %% [markdown]
# **EP cost and placement.** Each MoE layer does 2 all-to-alls forward (dispatch, combine) and 2 backward, each
# moving $\approx k\cdot T\cdot h\cdot\frac{N-1}{N}$ values per rank. Unlike TP, the volume does not shrink with the
# expert size, and all-to-all stresses the network's bisection bandwidth. Hence EP inside a node when possible, or
# node-limited routing across nodes: DeepSeek-V3 routes each token to at most 4 nodes, overlaps the all-to-all with
# compute in DualPipe, and replaces the auxiliary loss with a bias-based balancing scheme (DeepSeek-V3 technical
# report, arXiv:2412.19437). Capacity factor trades dropped tokens (quality) against padded compute and buffer
# memory (Switch uses 1.0-1.25 in training). Dropless MoE implementations use grouped GEMMs with variable sizes
# instead of fixed buffers.
#
# ## Say it out loud
# 1. Derive the log-sum-exp merge of two partial attention results. Why is it exact?
# 2. Walk through ring attention for 4 ranks. What does each rank send and compute per step, and when is the
#    communication hidden?
# 3. Why is causal ring attention load-imbalanced with contiguous shards, and how does zig-zag (or striping) fix it?
# 4. Ulysses vs ring attention: communication pattern, volume, constraints, when to use which?
# 5. How is attention's backward done under ring attention (what circulates)?
# 6. Explain top-$k$ routing, capacity factor and token dropping. What happens to a dropped token?
# 7. Walk through EP dispatch and combine. Why do you need an all-to-all of counts first?
# 8. Write the Switch load-balancing loss. What are its value at uniform routing and at collapse, and why does
#    its gradient balance the load?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. For key sets $A$, $B$: $\mathrm{softmax}$ over $A\cup B$ has normalizer $Z = Z_A + Z_B$ with
#    $Z_X = \sum_{c\in X}e^{s_c}$, and $O = (Z_A O_A + Z_B O_B)/Z$ since $O_X = \sum_{c\in X}e^{s_c}v_c/Z_X$. In log
#    space $\ell = \mathrm{logaddexp}(\ell_A, \ell_B)$ and $O = e^{\ell_A-\ell}O_A + e^{\ell_B-\ell}O_B$. Nothing is
#    approximated, only re-associated (the online-softmax identity of Milakov & Gimelshein / FlashAttention).
# 2. Rank $r$ keeps $Q_r$. At step $t$ it holds $K,V$ of rank $r-t$: it posts isend to $r+1$ and irecv from $r-1$
#    for the next block, computes $Q_rK_{r-t}^\top$ and the block result, merges, then waits. $N$ steps, $N-1$
#    transfers of $2\cdot\frac{T}{N}\cdot d$ per head. Hidden when block compute ($\propto (T/N)^2 d$) exceeds the
#    transfer time ($\propto (T/N)d/W$), i.e. when $T/N \gtrsim$ a hardware-dependent constant (Liu et al. give
#    the condition in terms of FLOPs/bandwidth).
# 3. Under a causal mask the queries late in the sequence see more keys. With contiguous shards rank $r$ has
#    $r+1$ non-empty blocks, so the last rank does $\sim 2\times$ the average work and everyone waits for it.
#    Zig-zag gives rank $r$ chunks $r$ and $2N-1-r$: one cheap early chunk and one expensive late chunk, so equal
#    work (checked above). Striped attention assigns tokens round-robin for the same effect.
# 4. Ring: P2P, ~$2Th$ per rank regardless of $N$, overlappable, no head constraint, works across nodes, but needs
#    a custom attention loop (merge, causal block skipping). Ulysses: 4 all-to-alls of activation size per
#    attention, per-rank volume $\propto Th/N$, uses any attention kernel unchanged, but degree ≤ number of (KV)
#    heads and the all-to-all wants a fast, high-bisection network, so in practice it stays inside a node. They
#    combine: Ulysses intra-node, ring inter-node.
# 5. The backward needs each $(Q_r, dO_r)$ against every KV block again. KV blocks circulate a second time, and
#    each rank computes partial $dK_j, dV_j$ for the block it holds. Those partial gradients also travel around the
#    ring, accumulating, until they reach the owner (picotron's `RingAttentionFunc.backward`). The saved per-row
#    lse lets each block's probabilities be recomputed exactly, as in FlashAttention's backward.
# 6. The router gives each token a distribution over experts; the token is sent to its $k$ highest. Each expert
#    has a fixed buffer $C = \lceil\text{cf}\cdot kT/E\rceil$ per group, needed for static shapes and memory. When
#    more tokens pick an expert than fit, the overflow is dropped (by priority order). A dropped token gets no
#    expert output for that slot and is carried by the residual stream unchanged. Larger cf → fewer drops but more
#    padding, compute and communication.
# 7. Each rank sorts its kept assignments by expert (so by destination rank), sends `x[tokens]` with
#    `all_to_all_single(in_splits=counts_to_each_rank, out_splits=counts_from_each_rank)`, the experts process what
#    arrives, and a second all-to-all with the splits swapped returns results to their source positions. The
#    receive sizes depend on other ranks' routing, which this rank does not know, so the counts are exchanged
#    first (one small all-to-all) to allocate buffers and set the splits.
# 8. $\alpha E\sum_e f_eP_e$. It equals $\alpha$ when balanced and $\alpha E$ when all tokens go to one expert. $f$
#    is a constant in the gradient, so $\partial/\partial P_e \propto f_e$: probability mass is pushed hardest away
#    from the most-loaded experts. It is minimized when the dispatch is uniform (Switch §2.2).
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
