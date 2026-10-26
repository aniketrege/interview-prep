# %% [markdown]
# # G1.16 · Parallelism planning: a memory + communication calculator and interview design problems
#
# **ID:** G1.16 · **Target:** 100 min · **Plan week:** 15-16 · **Part:** G1
#
# **The lab.** Systems interviews (big labs, Modal, Prime Intellect) rarely ask you to write NCCL code. They ask
# "you have X GPUs and a Y-parameter model: how do you lay it out, and what will bound you?". This lab turns
# G1.10-G1.15 into a calculator you write yourself:
# 1. **parameter count** from the architecture (dense, GQA, SwiGLU), checked against GPT-3 175B, Llama-2-70B,
#    Llama-3-405B;
# 2. **memory per GPU**: model states under ZeRO (G1.12) sharded over TP×PP×DP×CP, plus activations from
#    Korthikanti et al.'s Table 2 (TP, SP, selective/full recomputation) and the 1F1B in-flight rule (G1.14);
# 3. **time per step**: compute ($6\Psi$ + attention FLOPs at an assumed MFU), the pipeline bubble, and each
#    collective's time on the link it actually uses (NVLink inside a node, InfiniBand across nodes; group placement
#    in Llama 3's [TP, CP, PP, DP] order);
# 4. worked interview problems with model answers, grounded in the calculator and cited numbers.
#
# Single process; no GPU needed. Every hardware number is cited (or is the value you measured in G1.10).
#
# ## Sources
# - V. Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, 2022, arXiv:2205.05198
#   (Eq. 1-6 and Table 2: activation bytes per layer $sbh(34 + 5as/h)$, TP $sbh(10 + 24/t + 5as/(ht))$, TP+SP
#   $sbh(34/t + 5as/(ht))$, selective recompute $34\,sbh/t$, full recompute $2sbh$; the first 1F1B stage holds $L$
#   layers' worth; interleaving multiplies by $1 + \frac{p-1}{pm}$).
# - S. Rajbhandari et al., *ZeRO*, SC 2020, arXiv:1910.02054 ($16\Psi$ model states, stages, $K=12$).
# - D. Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, SC 2021,
#   arXiv:2104.04473 (Takeaways #1-#3: TP ≤ GPUs per server, PP across servers, DP for the rest; bubble
#   $(p-1)/m$; FLOPs per iteration).
# - A. Grattafiori et al. (Llama Team, Meta), *The Llama 3 Herd of Models*, 2024, arXiv:2407.21783 (§3.3.2, Table 4:
#   405B on 8,192-16,384 H100s with TP=8, CP=1 or 16, PP=16, DP=64/128/8, 16M tokens per batch, 380-430 TFLOPs/GPU
#   = 38-43% BF16 MFU; parallelism order [TP, CP, PP, DP], innermost = highest bandwidth; Table 3 architecture).
# - H. Touvron et al., *Llama 2*, 2023, arXiv:2307.09288 (70B: 80 layers, $d$=8192, 64 heads, 8 KV heads, FFN 28672,
#   vocab 32000). T. Brown et al., *Language Models are Few-Shot Learners*, 2020, arXiv:2005.14165 (GPT-3 175B:
#   96 layers, $d$=12288, 96 heads, context 2048).
# - A. Chowdhery et al., *PaLM*, 2022, arXiv:2204.02311 (App. B: model FLOPs per token $6N + 12LHQT$ for MFU).
# - J. Austin et al., *How to Scale Your Model*, 2025: Part 5 (training parallelism, compute-vs-comm thresholds)
#   https://jax-ml.github.io/scaling-book/training/ and Part 12 (H100: 990 bf16 TFLOP/s; NVLink 450 GB/s per GPU
#   per direction, ~370 GB/s achieved all-reduce in a node; 400 Gb/s = 50 GB/s IB NIC per GPU)
#   https://jax-ml.github.io/scaling-book/gpus/
# - N. Tazi et al., *The Ultra-Scale Playbook*, Hugging Face 2025, §"Finding the Best Training Configuration"
#   (step 1: fit in memory, TP ≤ 8 inside a node, PP/ZeRO-3 across nodes; step 2: reach the target global batch
#   with DP and accumulation; step 3: optimize throughput):
#   https://huggingface.co/spaces/nanotron/ultrascale-playbook
# - Stanford CS336 Spring 2026, Assignment 2 §8 (parallelism analysis). stanford-cs336/assignment2-systems@ca8bc81, MIT.

# %%
import math
from dataclasses import dataclass, replace

from prep import Session, check, check_close

s = Session("G1.16", target_min=100)


@dataclass
class Model:
    name: str
    L: int            # layers
    h: int            # hidden size
    a: int            # attention heads
    V: int            # vocab
    ffn: int          # MLP inner size
    mlp_mats: int = 2  # 2 = GELU MLP (up, down), 3 = SwiGLU (gate, up, down)
    kv: int = None    # KV heads (GQA); None = a
    tied: bool = False


@dataclass
class Plan:
    tp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1
    zero: int = 1            # 0 = DDP, 1/2/3 ZeRO stage over the dp*cp group
    mb: int = 1              # microbatch size (sequences)
    seq: int = 4096
    batch_tokens: float = 4e6
    sp: bool = True          # sequence parallelism with TP
    recompute: str = "selective"   # "none" | "selective" | "full"
    v: int = 1               # interleaved virtual stages

    @property
    def n(self):
        return self.tp * self.pp * self.dp * self.cp


@dataclass
class HW:
    mem: float = 80e9          # H100 80 GB
    flops: float = 990e12      # dense bf16 (scaling-book Part 12)
    mfu: float = 0.40          # assumed; Llama 3 reports 38-43%
    nvlink: float = 370e9      # achieved intra-node all-reduce busbw (scaling-book Part 12; peak 450 GB/s/dir)
    ib: float = 50e9           # 400 Gb/s NIC per GPU
    gpn: int = 8               # GPUs per node


GPT3 = Model("GPT-3 175B", L=96, h=12288, a=96, V=50257, ffn=4 * 12288, tied=True)
LLAMA2_70B = Model("Llama-2 70B", L=80, h=8192, a=64, V=32000, ffn=28672, mlp_mats=3, kv=8)
LLAMA3_405B = Model("Llama-3 405B", L=126, h=16384, a=128, V=128256, ffn=53248, mlp_mats=3, kv=8)
LLAMA3_8B = Model("Llama-3 8B", L=32, h=4096, a=32, V=128256, ffn=14336, mlp_mats=3, kv=8)

# %% [markdown]
# ## 1. Parameters
# Per layer: attention $Q$ and $O$ are $h\times h$, $K$ and $V$ are $h\times (h\cdot kv/a)$ with GQA; the MLP has
# `mlp_mats` matrices of $h\times\text{ffn}$. Embeddings $V\cdot h$, times 2 if the output head is untied. Ignore
# norms and biases (< 0.1%).

# %%
def param_count(m: Model):
    """Total parameters (weights only; norms/biases ignored)."""
    ### BEGIN SOLUTION
    kv = m.kv or m.a
    attn = 2 * m.h * m.h + 2 * m.h * (m.h * kv // m.a)
    mlp = m.mlp_mats * m.h * m.ffn
    emb = m.V * m.h * (1 if m.tied else 2)
    return m.L * (attn + mlp) + emb
    ### END SOLUTION


def layer_params(m: Model):
    return (param_count(m) - m.V * m.h * (1 if m.tied else 2)) / m.L


check_close("GPT-3 = 175B", param_count(GPT3) / 1e9, 175, rtol=0.01)
check_close("Llama-2 70B ~ 69B (reported as 70B)", param_count(LLAMA2_70B) / 1e9, 69.0, rtol=0.01)
check_close("Llama-3 405B", param_count(LLAMA3_405B) / 1e9, 405, rtol=0.01)
check_close("Llama-3 8B", param_count(LLAMA3_8B) / 1e9, 8.0, rtol=0.02)
for m_ in (GPT3, LLAMA2_70B, LLAMA3_405B, LLAMA3_8B):
    print(f"{m_.name:14s} {param_count(m_)/1e9:7.1f} B params")

# %% [markdown]
# ## 2. Memory per GPU
# **Model states.** Each GPU holds the parameters of $L/p$ layers sliced by $t$: $\Psi_\text{gpu} \approx \Psi/(tp)$
# (we spread the embeddings evenly, a simplification: really they sit on the first/last stage). ZeRO then shards
# over the data-parallel group. CP ranks hold the same weights, so the ZeRO group is $d\cdot c$. Bytes:
# $(2+2+12)\Psi_\text{gpu}$ for bf16 params + bf16 grads + fp32 master/Adam (G1.12).
#
# **Activations** (bytes per layer per microbatch; $s$ is the per-rank sequence $s/c$ under CP; $a$ heads):
#
# | configuration | bytes / layer |
# | --- | --- |
# | no parallelism | $sbh(34 + 5as/h)$ |
# | TP | $sbh(10 + 24/t + 5as/(ht))$ |
# | TP + SP | $sbh(34/t + 5as/(ht))$ |
# | TP (+SP) + selective recompute | $sbh(10 + 24/t)$ (resp. $34\,sbh/t$) |
# | full recompute | $2sbh$ (resp. $2sbh/t$ with SP), plus one layer's working set during its backward |
#
# The $5as/h$ term is the stored $s\times s$ softmax and its dropout mask. FlashAttention never stores it, so in
# practice modern stacks are at the "selective" row by default. **1F1B rule:** the first stage holds
# $\min(p, m)$ microbatches × $L/p$ layers, i.e. ≈ $L$ layers' worth when $m \ge p$, times $1 + \frac{p-1}{pv}$
# with $v$-way interleaving.

# %%
def zero_state_bytes(psi, n_d, stage):
    p, g, o = 2 * psi, 2 * psi, 12 * psi
    if stage >= 1:
        o /= n_d
    if stage >= 2:
        g /= n_d
    if stage >= 3:
        p /= n_d
    return p + g + o


def act_bytes_per_layer(s, b, h, a, t=1, sp=False, recompute="none"):
    """Korthikanti et al. Table 2 (16-bit activations, 1-byte dropout masks)."""
    ### BEGIN SOLUTION
    sbh = s * b * h
    if recompute == "full":
        return 2 * sbh / (t if sp else 1)
    attn = 0 if recompute == "selective" else 5 * a * s / h
    if sp:
        return sbh * (34 / t + attn / t)
    return sbh * (10 + 24 / t + attn / t)
    ### END SOLUTION


def memory_per_gpu(m: Model, p: Plan):
    """Return dict of bytes on the most loaded (first) pipeline stage."""
    ### BEGIN SOLUTION
    psi_gpu = param_count(m) / (p.tp * p.pp)
    states = zero_state_bytes(psi_gpu, p.dp * p.cp, p.zero)
    s_loc = p.seq / p.cp
    per_layer = act_bytes_per_layer(s_loc, p.mb, m.h, m.a, p.tp, p.sp, p.recompute)
    m_micro = p.batch_tokens / (p.dp * p.mb * p.seq)
    layers_held = min(p.pp, m_micro) * m.L / p.pp * (1 + (p.pp - 1) / (p.pp * p.v) if p.v > 1 else 1)
    acts = per_layer * layers_held
    if p.recompute == "full":     # one layer is materialized (no recompute savings) during its backward
        acts += act_bytes_per_layer(s_loc, p.mb, m.h, m.a, p.tp, p.sp, "selective")
    if p.zero == 3:               # one gathered layer (+ one prefetched) of bf16 weights
        acts += 2 * 2 * layer_params(m) / p.tp
    return {"states": states, "activations": acts, "total": states + acts}
    ### END SOLUTION


# checks: Korthikanti's GPT-3 numbers, ZeRO Fig. 1, and two hand-computed cases
check_close("GPT-3: 5as/h = 80", 5 * GPT3.a * 2048 / GPT3.h, 80)
sbh = 2048 * 1 * 12288
check_close("act/layer, no parallelism", act_bytes_per_layer(2048, 1, 12288, 96), sbh * (34 + 80))
check_close("act/layer, TP=8", act_bytes_per_layer(2048, 1, 12288, 96, t=8), sbh * (10 + 3 + 10))
check_close("act/layer, TP+SP = 1/t of no-parallel", act_bytes_per_layer(2048, 1, 12288, 96, 8, True), sbh * 114 / 8)
check_close("act/layer, TP+SP+selective", act_bytes_per_layer(2048, 1, 12288, 96, 8, True, "selective"), sbh * 34 / 8)
check_close("act/layer, full recompute + SP", act_bytes_per_layer(2048, 1, 12288, 96, 8, True, "full"), 2 * sbh / 8)
toy = Model("toy", L=4, h=1000, a=10, V=1000, ffn=4000, tied=True)        # 4*12e6 + 1e6 = 49e6 params
mem = memory_per_gpu(toy, Plan(tp=1, pp=1, dp=1, zero=0, mb=1, seq=100, batch_tokens=100, sp=False, recompute="none"))
check_close("toy: model states = 16 Psi", mem["states"], 16 * 49e6)
check_close("toy: activations = L * sbh(34 + 5as/h)", mem["activations"], 4 * 100 * 1000 * (34 + 5 * 10 * 100 / 1000))
mem = memory_per_gpu(toy, Plan(tp=2, pp=2, dp=4, zero=1, mb=1, seq=100, batch_tokens=100 * 4 * 8, sp=True,
                               recompute="selective"))
check_close("toy: TP2 PP2 DP4 ZeRO-1 states", mem["states"], (4 + 12 / 4) * 49e6 / 4)
check_close("toy: 1F1B stage 0 holds L layers (TP+SP+selective)", mem["activations"], 4 * 100 * 1000 * 34 / 2)
mem_v = memory_per_gpu(toy, Plan(tp=2, pp=2, dp=4, zero=1, mb=1, seq=100, batch_tokens=100 * 4 * 8, v=2))
check_close("toy: interleaving (v=2) multiplies by 1 + (p-1)/(pv)", mem_v["activations"], mem["activations"] * 1.25)

# %% [markdown]
# ## 3. Time per step
# **Compute.** Model FLOPs per token $= 6\Psi + 12 L h s$ (PaLM App. B: matmuls forward+backward, plus attention
# scores). Full recomputation adds one forward ($\times 4/3$ on the matmuls). Time $=$ FLOPs × tokens /
# ($N$ × peak × MFU), divided by $1 - $ bubble, where bubble $= \frac{p-1}{vm + p - 1}$ and $m$ is the number
# of microbatches per DP replica.
#
# **Communication.** Which link a group uses depends on placement. In the [TP, CP, PP, DP] order (Llama 3),
# group $X$ with inner stride $S$ (product of the inner degrees) stays inside a node iff $S\cdot X \le$ GPUs per
# node. Bytes (bf16) and patterns:
# - **TP:** 4 all-reduces of $b\,(s/c)\,h$ per layer per microbatch, on the critical path (G1.13). Each costs
#   $2\frac{t-1}{t}\cdot 2b(s/c)h / W$;
# - **CP (ring):** per layer per microbatch, KV blocks of $2b\frac{s}{c}h_{kv}$ values pass $c-1$ times forward and
#   about twice that in backward (KV again + dKV), overlapped with attention;
# - **PP:** per microbatch, one activation forward and one gradient backward of $b(s/c)h/t$ (SP-sharded) per stage
#   boundary, P2P, overlapped;
# - **DP:** gradients of $\Psi_\text{gpu}$: $2\frac{d-1}{d}\cdot 2\Psi_\text{gpu}$ bytes (ZeRO ≤ 2), 1.5× that for
#   ZeRO-3, once per step and overlappable with backward (G1.11-G1.12).
#
# The "exposed" estimate is compute/(1 − bubble) + TP + whatever DP time exceeds the backward pass (≈ 2/3 of
# compute). It is a first-order model meant to rank plans. Real systems differ by tens of percent.

# %%
def group_bw(stride, size, hw: HW):
    return hw.nvlink if stride * size <= hw.gpn else hw.ib


def step_time(m: Model, p: Plan, hw: HW = HW()):
    """Return dict of seconds per optimizer step: compute, bubble, tp, cp, pp, dp, exposed."""
    ### BEGIN SOLUTION
    tokens = p.batch_tokens
    flops = (6 * param_count(m) * (4 / 3 if p.recompute == "full" else 1) + 12 * m.L * m.h * p.seq) * tokens
    compute = flops / (p.n * hw.flops * hw.mfu)
    m_micro = tokens / (p.dp * p.mb * p.seq)
    bubble = (p.pp - 1) / (p.v * m_micro + p.pp - 1)
    s_loc = p.seq / p.cp
    act = 2 * p.mb * s_loc * m.h                                       # bytes of one activation tensor
    layers = m.L / p.pp
    tp = 0.0 if p.tp == 1 else layers * m_micro * 4 * 2 * (p.tp - 1) / p.tp * act / group_bw(1, p.tp, hw)
    kvdim = m.h * (m.kv or m.a) // m.a
    cp = 0.0 if p.cp == 1 else layers * m_micro * 3 * (p.cp - 1) * 2 * 2 * p.mb * s_loc * kvdim / \
        group_bw(p.tp, p.cp, hw)
    pp = 0.0 if p.pp == 1 else m_micro * 2 * act / p.tp / group_bw(p.tp * p.cp, p.pp, hw)
    psi_gpu = param_count(m) / (p.tp * p.pp)
    dp_ = p.dp * p.cp                                                  # ZeRO / gradient-reduction group
    bw = min(group_bw(p.tp, p.cp, hw), group_bw(p.tp * p.cp * p.pp, p.dp, hw))
    dp = 0.0 if dp_ == 1 else (1.5 if p.zero == 3 else 1) * 2 * (dp_ - 1) / dp_ * 2 * psi_gpu / bw
    exposed = compute / (1 - bubble) + tp + max(0.0, dp - 2 / 3 * compute)
    return {"compute": compute, "bubble": bubble, "tp": tp, "cp": cp, "pp": pp, "dp": dp, "exposed": exposed,
            "microbatches": m_micro}
    ### END SOLUTION


# checks against numbers derived in earlier labs
one_b = Model("1B", L=16, h=2304, a=16, V=50000, ffn=4 * 2304, tied=True)
t_ = step_time(one_b, Plan(dp=8, seq=4096, batch_tokens=8 * 4096))
check_close("DP all-reduce of 2 bytes x Psi on 8 GPUs = 2*(7/8)*2Psi/370 GB/s",
            t_["dp"], 2 * 7 / 8 * 2 * param_count(one_b) / 370e9)
t_ = step_time(LLAMA2_70B, Plan(tp=8, dp=1, seq=4096, batch_tokens=4096))
check_close("TP time on NVLink matches 4 all-reduces per layer", t_["tp"],
            80 * 4 * 2 * 7 / 8 * 2 * 4096 * 8192 / 370e9)
t_ = step_time(LLAMA2_70B, Plan(tp=16, dp=1, seq=4096, batch_tokens=4096))
check("TP=16 spills onto InfiniBand (slower link)", t_["tp"] > 7 * step_time(LLAMA2_70B, Plan(tp=8, seq=4096, batch_tokens=4096))["tp"])
check_close("bubble formula (p-1)/(m+p-1)", step_time(GPT3, Plan(pp=8, dp=1, seq=2048, batch_tokens=2048 * 24))["bubble"],
            7 / (24 + 7))

# %% [markdown]
# A sanity check against a published run. Llama 3 405B, Table 4, first row: 8,192 GPUs, TP=8, CP=1, PP=16, DP=64,
# sequence 8,192, 16M tokens per batch, **430 TFLOPs/GPU (43% MFU)**. Our compute model at 43% MFU predicts the
# step time below. The paper doesn't print the step time, but $6\Psi\cdot 16\text{M}/(8192\cdot 430\text{T})$
# follows from its own numbers, so this checks the arithmetic, not the model.

# %%
def report(m: Model, p: Plan, hw: HW = HW(), label=""):
    mem, t = memory_per_gpu(m, p), step_time(m, p, hw)
    fits = "fits" if mem["total"] <= hw.mem * 0.9 else "DOES NOT FIT"
    print(f"{label:34s} N={p.n:5d} tp={p.tp} pp={p.pp} dp={p.dp} cp={p.cp} zero={p.zero} {p.recompute:9s} | "
          f"mem {mem['states']/1e9:6.1f}+{mem['activations']/1e9:6.1f}={mem['total']/1e9:6.1f} GB ({fits}) | "
          f"step {t['exposed']:6.2f}s (compute {t['compute']:.2f}, bubble {t['bubble']:.0%}, tp {t['tp']:.2f}, "
          f"dp {t['dp']:.2f}, pp {t['pp']:.3f}, cp {t['cp']:.3f})")
    return mem, t


llama3 = Plan(tp=8, cp=1, pp=16, dp=64, zero=1, mb=1, seq=8192, batch_tokens=16 * 2**20, recompute="selective")
_, t = report(LLAMA3_405B, llama3, replace(HW(), mfu=0.43), "Llama-3 405B, Table 4 row 1")
manual = 6 * param_count(LLAMA3_405B) * 16 * 2**20 / (8192 * 430e12)
print(f"6*Psi*tokens/(N*430 TFLOP/s) = {manual:.1f} s per step (matmul FLOPs only)")
check("our compute time is within 15% of 6 Psi D / (N x 430 TFLOP/s)", abs(t["compute"] - manual) / manual < 0.15)

# %% [markdown]
# Two things the calculator flags. (1) With plain 1F1B and $m = 32$ microbatches per replica the bubble is
# $15/47 = 32\%$. The reported MFU is end-to-end, so the real schedule must hide most of that (Llama 3 uses
# interleaving and a tunable depth-first/breadth-first schedule). Our `v=1` estimate is pessimistic. (2) The layout
# does not fit with selective recompute alone: holding $L$ layers of activations at $s=8192$ costs ~70 GB on
# stage 0. So Llama 3 must have done more than our model assumes (their text discusses
# a modified, tunable pipeline schedule and memory imbalance from embeddings and warm-up microbatches; the
# calculator only knows plain 1F1B). Treat the memory line as an upper-bound estimate that tells you *where to
# look*: recompute more, reduce in-flight microbatches, or shard more.
#
# ## 4. Worked design problems
# Run each cell, then read the model answer. In an interview, say the numbers out loud the way the calculator
# prints them.
#
# ### Problem A. 70B dense model, 64 H100s (8 nodes × 8, NVSwitch inside, 400 Gb/s IB per GPU), 4k context, 4M tokens per step. Pick a layout.

# %%
m70 = LLAMA2_70B
base = Plan(seq=4096, batch_tokens=4 * 2**20, mb=1)
plans = {
    "A1 FSDP-64 (ZeRO-3), no TP": replace(base, dp=64, zero=3, sp=False, recompute="selective"),
    "A2 FSDP-64 + full recompute": replace(base, dp=64, zero=3, sp=False, recompute="full"),
    "A3 TP8(SP) x DP8 ZeRO-1": replace(base, tp=8, dp=8, zero=1, recompute="selective"),
    "A4 TP8(SP) x DP8 ZeRO-3": replace(base, tp=8, dp=8, zero=3, recompute="selective"),
    "A5 TP8 x PP2 x DP4 ZeRO-1": replace(base, tp=8, pp=2, dp=4, zero=1, recompute="selective"),
    "A6 TP16 (across nodes) x DP4": replace(base, tp=16, dp=4, zero=1, recompute="selective"),
}
resA = {k: report(m70, p, label=k) for k, p in plans.items()}
check("A1 (FSDP, no TP, selective) does not fit", resA["A1 FSDP-64 (ZeRO-3), no TP"][0]["total"] > 0.9 * 80e9)
check("A3 (TP8 x DP8 ZeRO-1) fits", resA["A3 TP8(SP) x DP8 ZeRO-1"][0]["total"] < 0.9 * 80e9)
check("TP across nodes is much slower than inside a node",
      resA["A6 TP16 (across nodes) x DP4"][1]["tp"] > 5 * resA["A3 TP8(SP) x DP8 ZeRO-1"][1]["tp"])

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answer A</summary>
#
# - **Memory first** (Playbook step 1). $16\Psi$ = 1.1 TB of model states, so sharding is mandatory. Per-GPU
#   activations at $s=4096$, $h=8192$ are $34\,sbh \approx 1.1$ GB per layer without TP, ×80 layers ≈ 90 GB. Pure
#   FSDP (A1) does not fit unless you fully recompute (A2: states 17.5 GB + ~5 GB activations).
# - **A3: TP=8 inside each node (with SP) × DP=8 across nodes with ZeRO-1** fits: states ≈ (4 + 12/8)Ψ/8 ≈ 48 GB,
#   plus ~11 GB of activations. TP traffic stays on NVSwitch: ~13 s of blocking all-reduces against ~74 s of
#   compute at 40% MFU, which is the price of TP. The DP all-reduce of the 8.6B-parameter TP shard (17 GB in bf16)
#   goes over IB at 50 GB/s per GPU, ~0.6 s, hidden under a ~50 s backward. A4 (ZeRO-3 over DP) costs 1.5× DP
#   traffic (still < 1 s) and saves ~30 GB per GPU, which you can spend on larger microbatches or less recompute.
# - **A2 (FSDP-64 + full recompute)** fits too and has no TP traffic at all. It pays ~33% extra matmul FLOPs for
#   recompute, and its ZeRO-3 traffic (8 s per step over IB) must be hidden by prefetching. It is a legitimate
#   alternative; compare measured throughput.
# - **A5 (PP=2)** also fits and moves only activations across nodes. Its bubble is tiny here ($m$ = 256
#   microbatches per replica, bubble ≈ 0.4%), but it adds engineering complexity.
# - **A6 shows why not TP=16:** half of every TP all-reduce crosses IB (7-9× slower), and TP time (~200 s)
#   exceeds compute.
# - Pick **A3 or A4** (TP8 + FSDP/ZeRO across nodes), as the Playbook recommends for 10-100B models. Pick A5 if
#   memory is tight at long context. Then tune `mb` for arithmetic intensity, and check the global batch (4M tokens
#   ⇒ 1024 sequences ⇒ 128 per DP replica: enough microbatches).
# </details>

# %% [markdown]
# ### Problem B. "Why not just use TP across nodes?"

# %%
for name, W, t_ in [("NVLink (achieved), t=8", 370e9, 8), ("IB 400 Gb/s per GPU, t=16", 50e9, 16)]:
    h, C = 8192, 990e12 * 0.4
    print(f"{name:26s}: per-layer TP comm/compute = 2(t-1)C/(9hW) = {2 * (t_ - 1) * C / (9 * h * W):.2f}  (h=8192, MFU 40%)")

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answer B</summary>
#
# TP needs 4 blocking all-reduces of the full activation ($b\cdot s\cdot h$) **per layer**, on the critical path.
# The ratio of TP communication to compute is $2(t-1)C/(9hW)$ (G1.13), independent of batch size, so you cannot
# amortize it with a bigger batch. Inside a node, $W$ ≈ 370-450 GB/s and the ratio is well below 1. Across nodes,
# $W$ = 50 GB/s per GPU, 7-9× worse, and the ratio exceeds 1: GPUs wait for the network every layer. Across nodes
# use parallelism whose traffic is either overlappable and once per step (DP/FSDP gradients), or small and
# point-to-point (PP activations at stage boundaries only). Narayanan et al. 2021 Takeaway #1; Llama 3 uses TP=8 =
# GPUs per server.
# </details>

# %% [markdown]
# ### Problem C. "Your DDP scaling efficiency on 8 GPUs is 60%. What do you check?"
# Answer out loud before opening the model answer.

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answer C</summary>
#
# (Not a calculator problem; this is the checklist.)
# 1. **Measure, don't guess**: step time with communication disabled (`no_sync` for every step, or a single GPU at
#    the same per-GPU batch) vs with it, and an `torch.profiler` / Nsight trace: is NCCL overlapped with backward,
#    and is there a long tail after the last gradient?
# 2. **Comm vs compute budget**: grad bytes / busbw (G1.10) vs backward time. If comm > backward, no amount of
#    overlap helps. Then increase per-GPU work (bigger microbatch, `no_sync` accumulation), shrink bytes (bf16/fp16
#    gradient compression hook, PowerSGD), or get faster links.
# 3. **Interconnect**: `nvidia-smi topo -m`, `NCCL_DEBUG=INFO` (are we on NVLink/P2P or falling back to SHM/sockets,
#    as on this PCIe box?), NIC count and placement, NCCL version/algorithm.
# 4. **Stragglers**: data loading (GPU idle at the start of the step), uneven sequence lengths, one slow or
#    thermally throttled GPU, CPU contention (one process per GPU, OMP threads), Python overhead. DDP runs at the
#    speed of the slowest rank.
# 5. **DDP knobs**: `bucket_cap_mb`, `gradient_as_bucket_view=True`, `static_graph=True`, avoid
#    `find_unused_parameters`, check that no host syncs (`.item()`) are inside the step.
# 6. **Then** reconsider the strategy: FSDP/ZeRO to fit a larger per-GPU batch, hierarchical (HSDP) layouts, or
#    low-communication training (G1.21) if the links are fundamentally slow.
# </details>

# %% [markdown]
# ### Problem D. "Train a 1B model on 4 PCIe RTX 3090s (this machine). Data parallel?"

# %%
box = HW(mem=24e9, flops=71e12, mfu=0.4, nvlink=3.5e9, ib=3.5e9, gpn=4)    # busbw measured in G1.10 (SHM path)
m1b = Model("1.3B", L=24, h=2048, a=16, V=50257, ffn=4 * 2048, tied=True)
pD = Plan(dp=4, zero=0, mb=4, seq=2048, batch_tokens=4 * 4 * 2048, sp=False)       # 8k tokens per GPU per step
resD = {}
for label, p in [("D1 DDP, selective", pD),
                 ("D2 ZeRO-1, full recompute", replace(pD, zero=1, recompute="full")),
                 ("D3 ZeRO-1, full, 8x accumulation", replace(pD, zero=1, recompute="full", batch_tokens=8 * pD.batch_tokens)),
                 ("D4 ZeRO-3 (FSDP), selective", replace(pD, zero=3))]:
    resD[label[:2]] = report(m1b, p, box, label)
check("D1 (plain DDP) does not fit in 24 GB", resD["D1"][0]["total"] > 0.9 * 24e9)
check("accumulation shrinks DP comm relative to compute",
      resD["D3"][1]["dp"] / resD["D3"][1]["compute"] < resD["D2"][1]["dp"] / resD["D2"][1]["compute"] / 4)

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answer D</summary>
#
# Memory first: $16\Psi \approx 21$ GB of model states, so plain DDP plus activations (35 GB) does not fit in 24 GB.
# ZeRO-1 is free in communication and with full recompute needs ~11 GB (D2). FSDP with selective recompute needs
# ~19 GB (D4). Communication: the bf16 gradient all-reduce of 2.6 GB at the ~3.5 GB/s busbw measured in G1.10 takes
# ~1.1 s per step, against ~2.6-3.4 s of compute for 32k tokens at 40% MFU. In theory DDP hides it under backward,
# but G1.11 showed overlap on this box is only partial (the last buckets form a tail, and NCCL's SHM path competes
# for PCIe and host memory). So: (1) accumulate with `no_sync` (D3: communication drops to ~4% of compute);
# (2) reduce in bf16; (3) prefer ZeRO-1/2 over ZeRO-3, whose 1.5× traffic (1.7 s) sits on the forward path;
# (4) on links that are slower still (across machines or the internet), use low-communication training that syncs
# every $H$ steps (DiLoCo, G1.21).
# </details>

# %% [markdown]
# ### Problem E. "Fine-tune Llama-3 8B at 128k context on one 8×H100 node."

# %%
m8 = LLAMA3_8B
for label, p in [("TP8, selective", Plan(tp=8, seq=131072, batch_tokens=131072, recompute="selective")),
                 ("TP8, full recompute", Plan(tp=8, seq=131072, batch_tokens=131072, recompute="full")),
                 ("CP8 (ring), ZeRO-3, full recompute", Plan(cp=8, zero=3, seq=131072, batch_tokens=131072,
                                                          sp=False, recompute="full")),
                 ("TP2 x CP4, ZeRO-1, full recompute", Plan(tp=2, cp=4, zero=1, seq=131072, batch_tokens=131072,
                                                          recompute="full"))]:
    report(m8, p, label=label)

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answer E</summary>
#
# At $s = 128$k a single layer's activations are $34\,sbh \approx 18$ GB ($b=1$, $h=4096$), and model states
# (16 × 8B = 128 GB) must be sharded across the node too. TP=8 with SP divides activations by 8, but 32 layers ×
# 2.3 GB still don't fit with selective recompute (89 GB). Two layouts fit:
# - **TP8 + full recompute** (~23 GB). But TP puts ~0.65 s of blocking all-reduces on the critical path, and it
#   splits every GEMM 8 ways ($h/t = 512$-wide shards: low arithmetic intensity, worse MFU than the model assumes).
# - **CP8 (ring attention or Ulysses) + ZeRO-3 + full recompute** (~24 GB). Every per-token activation is divided
#   by $c$, the attention FLOPs are split too (zig-zag balancing for the causal mask, G1.15), and the KV ring traffic
#   (~0.1 s) overlaps with attention compute. Ulysses also works here: 32 query and 8 KV heads are divisible by 8.
#
# Prefer CP (or TP2×CP4) at this context length. Note also that at 128k the attention FLOPs ($12Lhs$ per token,
# ~4× the $6\Psi$ term here) dominate. The calculator applies the recompute overhead only to the matmul term, so it
# understates full recomputation's cost when attention dominates. Llama 3 itself used CP=16 for its 128k stage
# (Table 4).
# </details>

# %% [markdown]
# ### Problem F. "MoE with 64 experts, top-2, $h$ = 4096: where do you put the experts, and what does it cost?"
# Answer out loud before opening the model answer.

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answer F</summary>
#
# Expert parallelism: shard the experts across an EP group (e.g. 8 GPUs = one node, 8 experts per GPU), keep
# attention and everything else data-parallel (or TP) as usual. Each MoE layer does 2 all-to-alls forward
# (dispatch, combine) and 2 backward, each moving ≈ $k\cdot b s\cdot h\cdot 2$ bytes $\times\frac{E_p - 1}{E_p}$ per
# GPU. With $k=2$, 16k tokens per GPU and $h=4096$ that is ≈ 0.25 GB per all-to-all: ~0.7 ms on NVSwitch, ~5 ms
# across IB, per layer per direction. That's why EP stays inside a node, or uses node-limited routing
# (DeepSeek-V3: ≤ 4 nodes per token) plus overlap of the all-to-all with computation of other microbatches.
# Watch load balance: the step waits for the busiest expert. Use the auxiliary loss (Switch, G1.15) or bias-based
# balancing, and size the capacity factor between token drops and wasted padding.
# </details>

# %% [markdown]
# ## Say it out loud
# 1. Recite the order of operations for planning (memory → global batch → throughput) and the default
#    layouts for 1B, 10-100B and 400B dense models.
# 2. Write the memory per GPU of TP×PP×DP with ZeRO-1 and SP+selective recompute.
# 3. What scales with batch size and what doesn't: DP communication, TP communication, PP bubble?
# 4. Given a compute budget, why can't you just increase DP forever? (critical batch size, communication per
#    microbatch, bubble)

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Playbook: (1) fit one step in memory: <10B: ZeRO/FSDP (+ recompute) or TP within a node; 10-100B: TP=8 inside
#    the node plus PP or ZeRO-3 across nodes; hundreds of B: TP=8 × PP across nodes × DP/FSDP, with CP for long
#    context and EP for MoE. (2) Reach the target global batch with DP and gradient accumulation. (3) Optimize
#    throughput: largest microbatch that fits, minimal TP, overlap, fewer PP stages / interleaving.
# 2. $(4 + 12/(dc))\cdot\Psi/(tp)$ bytes of states $+ \min(p, m)\cdot\frac{L}{p}\cdot 34\,\frac{s}{c} b h/t$ bytes of
#    activations (× the interleaving factor).
# 3. DP gradient traffic is once per step, independent of batch size, so bigger batches amortize it. TP traffic is
#    proportional to tokens: same fraction of compute regardless of batch. The PP bubble shrinks with the number of
#    microbatches $m$ = batch per replica / microbatch.
# 4. Beyond the critical batch size, more tokens per step stop reducing the number of steps needed (worse sample
#    efficiency; McCandlish et al. 2018, arXiv:1812.06162). With a fixed global batch, more DP replicas mean fewer microbatches per
#    replica (bigger PP bubble, less to overlap DP communication with) and a smaller per-GPU batch (lower MFU). So
#    beyond a point you add TP/PP/CP instead, or low-communication methods.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
