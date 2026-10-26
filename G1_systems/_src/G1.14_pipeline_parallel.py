# %% [markdown]
# # G1.14 · Pipeline parallelism: schedules, bubbles, activation memory, and a real send/recv pipeline
#
# **ID:** G1.14 · **Target:** 100 min · **Plan week:** 15 · **Part:** G1
#
# **The lab.** Pipeline parallelism (PP) splits the model by **layers** into $p$ stages on $p$ GPUs and streams
# $m$ microbatches through them. Only activations cross stage boundaries (point-to-point, $b\cdot s\cdot h$
# per microbatch), which is why PP is the strategy *across* nodes. Its cost is idle time: the **bubble**. You will:
# 1. write a **schedule simulator** (dependency-driven, any per-stage op order) and implement GPipe
#    (all-forward-all-backward) and 1F1B schedules; plot their timelines and check the bubble fraction against
#    $(p-1)/(m+p-1)$;
# 2. measure **in-flight activations per stage**: GPipe stores $m$ microbatches, 1F1B at most $p - s$;
# 3. build a **real $N$-stage pipeline** of tinygpt with `send`/`recv` of activations and activation-gradients,
#    run both schedules, and verify the gradients equal the unsplit model's on the full batch. Measure stage-0
#    peak memory for both;
# 4. understand interleaved 1F1B and zero-bubble scheduling (split backward into B and W).
#
# ## Sources
# - Y. Huang, Y. Cheng, A. Bapna, O. Firat, M. X. Chen, D. Chen, H. Lee, J. Ngiam, Q. V. Le, Y. Wu, Z. Chen,
#   *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*, NeurIPS 2019,
#   arXiv:1811.06965 — https://arxiv.org/abs/1811.06965 (micro-batching, synchronous flush, bubble
#   $O((p-1)/(m+p-1))$, re-materialization).
# - D. Narayanan, A. Harlap, A. Phanishayee, V. Seshadri, N. R. Devanur, G. R. Ganger, P. B. Gibbons, M. Zaharia,
#   *PipeDream: Generalized Pipeline Parallelism for DNN Training*, SOSP 2019, DOI 10.1145/3341301.3359646 (the
#   1F1B steady state).
# - D. Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, SC 2021,
#   arXiv:2104.04473 — https://arxiv.org/abs/2104.04473 (§2.2: PipeDream-Flush 1F1B, bubble time fraction
#   $(p-1)/m$ relative to ideal; interleaved schedule with $v$ chunks: $\frac1v\cdot\frac{p-1}{m}$, at $v\times$
#   the communication).
# - P. Qi, X. Wan, G. Huang, M. Lin, *Zero Bubble Pipeline Parallelism*, ICLR 2024, arXiv:2401.10241 —
#   https://arxiv.org/abs/2401.10241 (split backward into input-gradient B and weight-gradient W).
# - V. Korthikanti et al., 2022, arXiv:2205.05198 §4.2.3 (with 1F1B the first stage holds $p$ microbatches of
#   activations for $L/p$ layers, i.e. $L$ layers' worth whatever $p$ is).
# - Hugging Face picotron, `picotron/pipeline_parallel/pipeline_parallel.py` (`train_step_pipeline_afab`,
#   `train_step_pipeline_1f1b`, warmup = $p - s - 1$) and `pp_communications.py`, huggingface/picotron@59714b1,
#   Apache-2.0. Our schedules adapt their structure.
# - N. Tazi et al., *The Ultra-Scale Playbook*, Hugging Face 2025, §"Pipeline Parallelism" (bubble
#   $t_{pb}/t_{id} = (p-1)/m$, 1F1B memory, interleaving, zero bubble and DualPipe):
#   https://huggingface.co/spaces/nanotron/ultrascale-playbook

# %% [markdown]
# ## 1. The bubble
# $p$ stages, $m$ microbatches, forward time $t_f$ and backward time $t_b$ per microbatch per stage. Every stage
# does $m(t_f+t_b)$ of work. The last microbatch's forward reaches the last stage only after $(p-1)$ forward slots,
# and the first stage's last backward waits $(p-1)$ backward slots, so with a synchronous flush:
# $$T = (m + p - 1)(t_f + t_b), \qquad \text{bubble fraction} = 1 - \frac{m(t_f+t_b)}{T} = \frac{p-1}{m+p-1}.$$
# Megatron and the Playbook quote the same bubble relative to the *ideal* time, $t_{pb}/t_{id} = (p-1)/m$. Both
# say: use $m \gg p$.
#
# ```
# GPipe (AFAB), p=3, m=4          F=forward  B=backward  .=idle
# s0: F0 F1 F2 F3 .  .  .  .  .  .  B0 B0 B1 B1 B2 B2 B3 B3          (t_b = 2 t_f)
# s1: .  F0 F1 F2 F3 .  .  .  B0 B0 B1 B1 B2 B2 B3 B3 .  .
# s2: .  .  F0 F1 F2 F3 B0 B0 B1 B1 B2 B2 B3 B3 .  .  .  .
# 1F1B: s0 does F0 F1 (warmup p-s-1 = 2), then alternates F2 B0 F3 B1, then B2 B3: same makespan, but at most
#       p - s microbatches of activations alive instead of m.
# ```
# **Your task:** `schedule(kind, p, m)` returns each stage's op order as a list of `("F" | "B", microbatch)`.
# `simulate(orders, tf, tb)` executes them under the dependencies
# - $F(s,j)$ starts after $F(s-1,j)$ ends (activation must arrive),
# - $B(s,j)$ starts after $B(s+1,j)$ ends (gradient must arrive), and after $F(s,j)$,
#
# with each stage running one op at a time in its given order. Return `{(s, kind, j): (start, end)}`. Raise if the
# orders deadlock.

# %%
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from prep import Session, check, check_close

s = Session("G1.14", target_min=100)
NGPU = torch.cuda.device_count()
NPROC = min(4, NGPU) if NGPU >= 2 else 2
BACKEND = "nccl" if NGPU >= 2 else "gloo"
RUN = Path("runs_G1.14")
RUN.mkdir(exist_ok=True)


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


def schedule(kind, p, m):
    """Per-stage op order for 'gpipe' (all F then all B) or '1f1b' (warmup F's, steady F/B pairs, cooldown B's)."""
    ### BEGIN SOLUTION
    orders = []
    for st in range(p):
        if kind == "gpipe":
            ops = [("F", j) for j in range(m)] + [("B", j) for j in range(m)]
        else:
            w = min(p - st - 1, m)
            ops = [("F", j) for j in range(w)]
            for i in range(m - w):
                ops += [("F", w + i), ("B", i)]
            ops += [("B", j) for j in range(m - w, m)]
        orders.append(ops)
    return orders
    ### END SOLUTION


def simulate(orders, tf=1.0, tb=2.0):
    """Execute per-stage op orders under pipeline dependencies. Returns {(stage, kind, mb): (start, end)}."""
    p = len(orders)
    ### BEGIN SOLUTION
    ev, ptr, free = {}, [0] * p, [0.0] * p
    total = sum(len(o) for o in orders)
    while len(ev) < total:
        progress = False
        for st in range(p):
            if ptr[st] == len(orders[st]):
                continue
            kind, j = orders[st][ptr[st]]
            if kind == "F":
                deps = [(st - 1, "F", j)] if st > 0 else []
            else:
                deps = [(st, "F", j)] + ([(st + 1, "B", j)] if st < p - 1 else [])
            if all(d in ev for d in deps):
                start = max([free[st]] + [ev[d][1] for d in deps])
                end = start + (tf if kind == "F" else tb)
                ev[(st, kind, j)] = (start, end)
                free[st], ptr[st], progress = end, ptr[st] + 1, True
        if not progress:
            raise RuntimeError("deadlock: no stage can make progress")
    return ev
    ### END SOLUTION


def bubble_fraction(ev, p, m, tf=1.0, tb=2.0):
    T = max(e for _, e in ev.values())
    return 1 - m * (tf + tb) / T


def peak_inflight(ev, p):
    """Max over time of (#forwards finished - #backwards finished) per stage = microbatches whose activations are stored."""
    out = []
    for st in range(p):
        marks = sorted([(e, +1) for (s_, k, j), (b, e) in ev.items() if s_ == st and k == "F"] +
                       [(e, -1) for (s_, k, j), (b, e) in ev.items() if s_ == st and k == "B"])
        cur = best = 0
        for _, d in marks:
            cur += d
            best = max(best, cur)
        out.append(best)
    return out


for p_, m_ in [(4, 8), (4, 4), (8, 32), (2, 1)]:
    for kind in ("gpipe", "1f1b"):
        for tf, tb in [(1.0, 2.0), (1.0, 1.0)]:
            ev = simulate(schedule(kind, p_, m_), tf, tb)
            check_close(f"{kind} p={p_} m={m_} tb/tf={tb/tf:.0f}: bubble = (p-1)/(m+p-1)",
                        bubble_fraction(ev, p_, m_, tf, tb), (p_ - 1) / (m_ + p_ - 1))
    check("GPipe stores m microbatches on every stage", peak_inflight(simulate(schedule("gpipe", p_, m_)), p_) == [m_] * p_)
    check("1F1B stores min(p - s, m) on stage s",
          peak_inflight(simulate(schedule("1f1b", p_, m_)), p_) == [min(p_ - st, m_) for st in range(p_)])
try:
    simulate([[("B", 0), ("F", 0)], [("F", 0), ("B", 0)]])
    raise AssertionError("simulate should detect the deadlock")
except RuntimeError:
    print("ok  deadlock detected for an impossible order")

# %%
def plot(ev, p, title, ax):
    for (st, kind, j), (b, e) in ev.items():
        ax.broken_barh([(b, e - b)], (p - 1 - st - 0.4, 0.8), color=f"C{j % 10}", alpha=1.0 if kind == "F" else 0.55)
        ax.text((b + e) / 2, p - 1 - st, f"{kind}{j}", ha="center", va="center", fontsize=7)
    ax.set_yticks(range(p), [f"stage {p - 1 - i}" for i in range(p)])
    ax.set_title(title)


p_, m_ = 4, 8
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
for ax, kind in zip(axes, ("gpipe", "1f1b")):
    ev = simulate(schedule(kind, p_, m_))
    plot(ev, p_, f"{kind}: p={p_}, m={m_}, bubble {bubble_fraction(ev, p_, m_):.1%}, "
                 f"peak stored microbatches per stage {peak_inflight(ev, p_)}", ax)
axes[-1].set_xlabel("time (units of t_f; t_b = 2 t_f)")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Same bubble, different memory.** 1F1B does not shrink the bubble. It starts backward passes as early as
# possible, so stage $s$ never holds more than $p-s$ microbatches of activations instead of $m$. That allows
# large $m$ (small bubble) without memory growing with $m$. With 1F1B, stage 0 holds $p$ microbatches × $L/p$
# layers = $L$ layers' worth, independent of $p$ (Korthikanti et al. §4.2.3).
#
# **Interleaved 1F1B** (Narayanan et al. 2021): give each GPU $v$ non-contiguous chunks of layers (GPU 0 gets
# layers 0-1 and 8-9, ...). Each op is $v\times$ shorter, so the fill/drain shrinks: bubble $= \frac1v\frac{p-1}{m}$
# of ideal time, for $v\times$ more point-to-point messages. Llama 3 used interleaving plus its own tunable
# variant for 405B pre-training (Llama 3 paper §3.3.2).
#
# **Zero bubble** (Qi et al. 2024): a backward pass is two different matmuls: $B$ = gradient w.r.t. the input
# (needed by the previous stage, on the critical path) and $W$ = gradient w.r.t. the weights (needed only by the
# optimizer). Schedule $B$ immediately and defer $W$ into the slots that would otherwise be bubbles (ZB-H1 keeps
# 1F1B's memory and roughly a third of its bubble). Handcrafted and searched schedules then approach zero bubble,
# with a post-hoc validation trick to avoid the global sync for gradient clipping and NaN checks in the optimizer
# step. DeepSeek-V3's DualPipe builds on the same split (Playbook §"Zero Bubble and DualPipe").
#
# ## 2. A real pipeline with `send`/`recv`
# Each rank builds the *same* tinygpt (same seed, untied embeddings) and keeps only its stage: rank 0 the token +
# position embeddings and the first blocks, the last rank the last blocks + `ln_f` + head, and the loss. The helpers
# move tensors between neighbors with `p2p(...)`, which posts all requested sends/receives as **one**
# `batch_isend_irecv` group. **Your task:**
# - `fwd`: on stage 0 run the stage on the token ids; elsewhere make the received activation a leaf
#   (`requires_grad_()`, so `x.grad` exists after backward) and run the stage. On the last stage the output is the
#   microbatch's mean loss **divided by $m$**, so the accumulated gradients equal the full-batch mean-loss gradient;
# - `bwd`: `out.backward()` on the last stage, `out.backward(g)` elsewhere; return `x.grad` (None on stage 0);
# - `train_step`: GPipe = $m$ × (receive, forward, send), then $m$ × (receive grad, backward, send grad), FIFO as in
#   picotron. 1F1B = `w = min(p - s - 1, m)` warmup forwards, then $m - w$ steady iterations, then $w$ cooldown
#   backwards.
#
# **Deadlock trap (why `p2p` groups ops).** In the 1F1B steady state stage $s$ wants to *send* activation $j$ and
# then *receive* gradient $i$, while stage $s+1$ wants to *send* gradient $i$ and then *receive* activation $j$. With
# separate blocking calls, each side waits in its send for a receive the other side has not posted. NCCL
# serializes point-to-point ops between a pair of ranks on one stream, so this can hang on the GPU even though
# the Python calls return. Post the pair together: `p2p(send_fwd=out, recv_bwd=True)` on one side matches
# `p2p(send_bwd=dx, recv_fwd=True)` on the other. This is picotron's `bidirectional_pipeline_communicate`.
#
# Tied embeddings would put one weight in stage 0 *and* the last stage, and Megatron then all-reduces that
# gradient between the two. We untie to keep the lab focused.

# %%
# %%writefile runs_G1.14/pp.py
import copy
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

from prep.tinygpt import GPT, GPTConfig

backend = os.environ.get("BACKEND", "nccl")
lr_ = int(os.environ["LOCAL_RANK"])
dev = torch.device("cuda", lr_) if backend == "nccl" else torch.device("cpu")
if backend == "nccl":
    torch.cuda.set_device(dev)
dist.init_process_group(backend, device_id=dev if backend == "nccl" else None)
rank, P = dist.get_rank(), dist.get_world_size()
torch.backends.cuda.matmul.allow_tf32 = False
FIRST, LAST = rank == 0, rank == P - 1


class Stage(nn.Module):
    """This rank's slice of a GPT: (embeddings) + contiguous blocks + (ln_f, head, loss)."""

    def __init__(self, gpt, rank, P):
        super().__init__()
        L = len(gpt.blocks)
        per = [L // P + (i < L % P) for i in range(P)]          # picotron's distribute_layers
        lo = sum(per[:rank])
        self.layer_ids = list(range(lo, lo + per[rank]))
        self.blocks = nn.ModuleList(gpt.blocks[i] for i in self.layer_ids)
        self.gpt = gpt if (rank == 0 or rank == P - 1) else None   # embeddings / head live here
        self.first, self.last = rank == 0, rank == P - 1

    def forward(self, x, targets=None):
        if self.first:                                           # x = token ids
            T = x.shape[1]
            x = self.gpt.tok_emb(x) + self.gpt.pos_emb(torch.arange(T, device=x.device))
        for b in self.blocks:
            x = b(x)
        if self.last:
            logits = self.gpt.head(self.gpt.ln_f(x))
            return nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return x

    def stage_params(self):
        """(name in the full model, parameter) for parameters this stage owns."""
        names = {id(p): n for n, p in self._full.named_parameters()}
        own = [p for b in self.blocks for p in b.parameters()]
        if self.first:
            own += [self._full.tok_emb.weight, self._full.pos_emb.weight]
        if self.last:
            own += list(self._full.ln_f.parameters()) + [self._full.head.weight]
        return [(names[id(p)], p) for p in own]


def p2p(shape, send_fwd=None, send_bwd=None, recv_fwd=False, recv_bwd=False):
    """Post every requested neighbor transfer as ONE batch_isend_irecv group and wait. Requests that make no sense
    on this stage (e.g. recv_fwd on the first stage) are ignored and return None. Returns (x_from_prev, g_from_next)."""
    ops, x, g = [], None, None
    if send_fwd is not None and not LAST:
        ops.append(dist.P2POp(dist.isend, send_fwd.detach().contiguous(), rank + 1))
    if send_bwd is not None and not FIRST:
        ops.append(dist.P2POp(dist.isend, send_bwd.contiguous(), rank - 1))
    if recv_fwd and not FIRST:
        x = torch.empty(shape, device=dev)
        ops.append(dist.P2POp(dist.irecv, x, rank - 1))
    if recv_bwd and not LAST:
        g = torch.empty(shape, device=dev)
        ops.append(dist.P2POp(dist.irecv, g, rank + 1))
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()
    return x, g


def fwd(stage, mb, x, m):
    """Forward of one microbatch. x: activation from the previous stage (None on the first stage).
    Returns (input leaf or None, output). On the last stage the output is that microbatch's loss / m."""
    ### BEGIN SOLUTION
    ids, y = mb
    if FIRST:
        return None, stage(ids)
    x = x.requires_grad_()
    out = stage(x, y if LAST else None)
    return x, (out / m if LAST else out)
    ### END SOLUTION


def bwd(x, out, g):
    """Backward of one microbatch given the gradient g of `out` from the next stage (None on the last stage).
    Returns the gradient to send to the previous stage (None on the first stage)."""
    ### BEGIN SOLUTION
    if LAST:
        out.backward()
    else:
        out.backward(g)
    return None if FIRST else x.grad
    ### END SOLUTION


def train_step(stage, mbs, shape, kind):
    m, store = len(mbs), []          # store: FIFO of (x, out) awaiting their backward
    ### BEGIN SOLUTION
    if kind == "gpipe":
        for j in range(m):
            x, _ = p2p(shape, recv_fwd=True)
            x, out = fwd(stage, mbs[j], x, m)
            p2p(shape, send_fwd=out)
            store.append((x, out))
        for j in range(m):
            _, g = p2p(shape, recv_bwd=True)
            x, out = store.pop(0)
            p2p(shape, send_bwd=bwd(x, out, g))
        return
    w = min(P - rank - 1, m)                                    # warmup forwards
    for j in range(w):
        x, _ = p2p(shape, recv_fwd=True)
        x, out = fwd(stage, mbs[j], x, m)
        p2p(shape, send_fwd=out)
        store.append((x, out))
    x = p2p(shape, recv_fwd=True)[0] if m > w else None
    for i in range(m - w):                                      # steady state: 1 forward, 1 backward
        x, out = fwd(stage, mbs[w + i], x, m)
        store.append((x, out))
        _, g = p2p(shape, send_fwd=out, recv_bwd=True)          # send my output, get the grad of the oldest
        x0, out0 = store.pop(0)
        dx = bwd(x0, out0, g)
        last = i == m - w - 1
        x, _ = p2p(shape, send_bwd=dx, recv_fwd=not last)       # send my input-grad, get the next activation
    for j in range(w):                                          # cooldown
        _, g = p2p(shape, recv_bwd=True)
        x0, out0 = store.pop(0)
        p2p(shape, send_bwd=bwd(x0, out0, g))
    ### END SOLUTION


def build(cfg):
    torch.manual_seed(0)
    gpt = GPT(cfg).to(dev)
    stage = Stage(gpt, rank, P)
    stage._full = gpt
    return gpt, stage


res = {"P": P}
# (1) correctness: tiny model, compare every owned grad with the unsplit model on the full batch
cfg = GPTConfig(vocab_size=128, context_length=16, d_model=32, n_heads=4, n_layers=2 * P, tie_weights=False)
m, mb = 6, 2
g = torch.Generator().manual_seed(3)
data = torch.randint(0, cfg.vocab_size, (m * mb, cfg.context_length + 1), generator=g).to(dev)
full_ref, _ = build(cfg)
full_ref(data[:, :-1], data[:, 1:])[1].backward()
ref_grads = {n: p.grad for n, p in full_ref.named_parameters()}
mbs = [(data[j * mb:(j + 1) * mb, :-1], data[j * mb:(j + 1) * mb, 1:]) for j in range(m)]
for kind in ("gpipe", "1f1b"):
    _, stage = build(cfg)
    train_step(stage, mbs, (mb, cfg.context_length, cfg.d_model), kind)
    err = max((p.grad - ref_grads[n]).abs().max().item() for n, p in stage.stage_params())
    errs = torch.tensor([err], device=dev)
    dist.all_reduce(errs, op=dist.ReduceOp.MAX)
    res[f"grad_err_{kind}"] = errs.item()

# (2) memory: bigger activations, m = 8 microbatches, peak memory on each stage
if dev.type == "cuda":
    cfg = GPTConfig(vocab_size=1024, context_length=256, d_model=512, n_heads=8, n_layers=4 * P, tie_weights=False)
    m, mb = 8, 4
    data = torch.randint(0, cfg.vocab_size, (m * mb, cfg.context_length + 1), generator=g).to(dev)
    mbs = [(data[j * mb:(j + 1) * mb, :-1], data[j * mb:(j + 1) * mb, 1:]) for j in range(m)]
    for kind in ("gpipe", "1f1b"):
        _, stage = build(cfg)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        train_step(stage, mbs, (mb, cfg.context_length, cfg.d_model), kind)
        torch.cuda.synchronize()
        peak = torch.tensor([(torch.cuda.max_memory_allocated() - base) / 2**20], device=dev)
        allp = [torch.zeros_like(peak) for _ in range(P)]
        dist.all_gather(allp, peak)
        res[f"peak_act_MiB_{kind}"] = [round(t.item(), 1) for t in allp]
        del stage
        torch.cuda.empty_cache()
if rank == 0:
    print(json.dumps(res, indent=1))
    json.dump(res, open("pp.json", "w"))
dist.destroy_process_group()

# %%
torchrun("pp.py")
pp = load("pp.json")
for kind in ("gpipe", "1f1b"):
    check(f"{kind}: every stage's grads == unsplit model on the full batch", pp[f"grad_err_{kind}"] < 1e-5,
          f"max err {pp[f'grad_err_{kind}']:.2e}")
if "peak_act_MiB_gpipe" in pp:
    print("activation peak per stage (MiB)  GPipe:", pp["peak_act_MiB_gpipe"], "  1F1B:", pp["peak_act_MiB_1f1b"])
    check("1F1B needs less activation memory than GPipe on stage 0",
          pp["peak_act_MiB_1f1b"][0] < 0.8 * pp["peak_act_MiB_gpipe"][0])
    print(f"stage-0 ratio 1F1B / GPipe = {pp['peak_act_MiB_1f1b'][0] / pp['peak_act_MiB_gpipe'][0]:.2f} "
          f"(simulator predicts {min(pp['P'], 8)}/{8} = {min(pp['P'], 8) / 8:.2f} for the stored activations)")

# %% [markdown]
# The measured stage-0 ratio is close to the simulator's $p/m$ for stored activations. It is not exact: peak memory
# also includes the transient activations of the op that is running. The last stage holds logits. In real systems
# the last stage is often given fewer layers because of the head and loss (Llama 3 paper §3.3.2 discusses this
# imbalance).
#
# ## Say it out loud
# 1. Derive the bubble fraction of GPipe. How does it depend on $p$, $m$ and $t_b/t_f$?
# 2. 1F1B has the same bubble as GPipe. Then why does everyone use it?
# 3. How much activation memory does stage 0 hold under 1F1B, and why is it independent of $p$?
# 4. What does interleaving buy and what does it cost?
# 5. Explain the B/W split behind zero-bubble pipelines. Why can W be delayed?
# 6. Why is PP the parallelism you put across nodes, and TP within? What does PP communicate per microbatch?
# 7. What other problems does PP bring (load balance, embeddings, tied weights, batch-size constraints)?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. With synchronous flushes the critical path is $(m + p - 1)$ forward slots plus $(m+p-1)$ backward slots, so
#    $T = (m+p-1)(t_f+t_b)$ while each stage is busy for $m(t_f+t_b)$: bubble $(p-1)/(m+p-1)$, or $(p-1)/m$ relative
#    to the ideal. It does not depend on $t_b/t_f$ when every stage has the same costs (the simulator checks
#    $t_b = t_f$ and $t_b = 2t_f$). Unequal stage costs add straggler time.
# 2. Memory. GPipe keeps the activations of all $m$ microbatches until the backward phase. 1F1B begins backward as
#    soon as the last stage has its first microbatch, so stage $s$ holds at most $p - s$. You can raise $m$ to shrink
#    the bubble without raising memory.
# 3. $p$ microbatches in flight × $L/p$ layers = activations of $L$ layers for one microbatch. Doubling $p$
#    halves the layers per stage but doubles the microbatches it must hold (Korthikanti et al. 2022 §4.2.3).
# 4. With $v$ model chunks per device the bubble shrinks to $\frac1v\cdot\frac{p-1}{m}$ (the warmup and drain are
#    made of $v\times$ shorter ops), at the cost of $v\times$ more point-to-point communication, a constraint that $m$
#    be a multiple of $p$ (in Megatron's implementation), and slightly more memory.
# 5. The backward of $y = xW$ computes $dx = dy\,W^\top$ (B: the previous stage is waiting for it) and
#    $dW = x^\top dy$ (W: only the optimizer needs it, at the end of the step). Deferring W into idle slots fills the
#    bubble without delaying the critical path. Its inputs ($x$, $dy$) must stay alive longer, which costs memory.
#    Zero-bubble schedules also remove the optimizer-step synchronization (Qi et al. 2024).
# 6. PP sends one activation tensor ($b\cdot s\cdot h$) forward and one gradient backward per microbatch per stage
#    boundary, point-to-point and overlappable. TP needs 4 blocking all-reduces of the same size per *layer*, so it
#    needs NVLink. Across nodes, PP's small, infrequent P2P traffic fits slow IB links (Narayanan et al. 2021
#    Takeaway #1).
# 7. Load balance (embedding/head/loss stages are heavier: give them fewer blocks), tied input/output embeddings
#    live on two stages (all-reduce their gradient), global batch must provide $m \gg p$ microbatches per DP replica
#    (large batch), more complex checkpointing and failure handling, and schedule-dependent memory imbalance
#    (stage 0 holds the most).
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
