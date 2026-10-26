# %% [markdown]
# # G1.12 · ZeRO and FSDP: shard the optimizer yourself, then measure FSDP2 `fully_shard`
#
# **ID:** G1.12 · **Target:** 110 min · **Plan week:** 15 · **Part:** G1
#
# **The lab.** DDP (G1.11) keeps a full copy of parameters, gradients and optimizer states on every GPU. ZeRO
# (Rajbhandari et al. 2020) removes that redundancy in three stages, and PyTorch FSDP implements stage 3. You will:
# 1. derive the ZeRO memory table ($\Psi$ parameters, $K = 12$ optimizer bytes per parameter) and reproduce the
#    paper's Figure 1 numbers;
# 2. **implement a ZeRO-1 sharded AdamW** from collectives: reduce-scatter gradients → each rank runs AdamW on
#    its 1/N shard (with globally correct gradient clipping) → all-gather parameters. Check it against unsharded
#    single-process AdamW and measure that each rank holds 1/N of the optimizer state;
# 3. shard tinygpt with **FSDP2 `fully_shard`** (torch 2.14), look at the DTensor shards, and measure per-rank
#    memory and step time against DDP, with `reshard_after_forward` on/off and a bf16 `MixedPrecisionPolicy`;
# 4. **count FSDP's collectives** by wrapping `dist.all_gather_single` / `dist.reduce_scatter_single`, and confirm
#    that ZeRO-3 moves 1.5× the bytes of DDP.
#
# **Prerequisites:** G1.10 (all-reduce = reduce-scatter + all-gather), G1.11 (DDP).
#
# ## Sources
# - S. Rajbhandari, J. Rasley, O. Ruwase, Y. He, *ZeRO: Memory Optimizations Toward Training Trillion Parameter
#   Models*, SC 2020, arXiv:1910.02054 — https://arxiv.org/abs/1910.02054 (§3 the $2\Psi+2\Psi+K\Psi$ model states,
#   K=12 for mixed-precision Adam; Fig. 1 7.5B / 64-GPU example; §7 communication: $P_{os}$, $P_{os+g}$ = $2\Psi$
#   like DDP, $P_{os+g+p}$ = $3\Psi$ = 1.5×).
# - Y. Zhao, A. Gu, R. Varma, L. Luo, C.-C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott, S. Shleifer et al.,
#   *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*, VLDB 2023, arXiv:2304.11277,
#   DOI 10.14778/3611540.3611569 — https://arxiv.org/abs/2304.11277 (unit wrapping, all-gather before forward and
#   backward, reduce-scatter of gradients, prefetching, rate limiting).
# - PyTorch 2.14 `fully_shard` (FSDP2) API: `torch.distributed.fsdp.fully_shard`, `MixedPrecisionPolicy`,
#   `reshard_after_forward`, DTensor `Shard(0)` parameters:
#   https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html · FSDP2 tutorial:
#   https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
# - Stanford CS336 Spring 2026, Assignment 2 §6 (optimizer state sharding) and §7 (FSDP: all-gather weights in
#   time for forward/backward, reduce-scatter grads, cast to the compute dtype before communicating), and
#   `tests/test_sharded_optimizer.py` (compare with unsharded AdamW). stanford-cs336/assignment2-systems@ca8bc81,
#   MIT. Test idea adapted; code and prose ours. Note: CS336 shards *whole parameters* across ranks; we shard a flat
#   buffer as ZeRO does.
# - N. Tazi et al., *The Ultra-Scale Playbook*, Hugging Face 2025, §"ZeRO" (memory per stage, ZeRO-3 communication
#   $3\Psi$ vs $2\Psi$, prefetching): https://huggingface.co/spaces/nanotron/ultrascale-playbook
# - `torch.nn.utils.clip_grad_norm_` (reference semantics for clipping): https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html

# %% [markdown]
# ## 1. Where the memory goes, and the ZeRO stages
# Mixed-precision Adam for a model with $\Psi$ parameters keeps, per GPU (ZeRO §3):
# bf16/fp16 parameters $2\Psi$ bytes, bf16/fp16 gradients $2\Psi$, and optimizer state $K\Psi$ with $K=12$
# (fp32 master parameters 4, Adam first moment 4, second moment 4). That is **16Ψ bytes** before any activations.
# A 7.5B model needs 120 GB, more than one 80 GB GPU, even though DDP has 64 GPUs' worth of memory.
#
# ZeRO partitions the model states across the $N_d$ data-parallel ranks:
#
# | stage | sharded | bytes per GPU | communication per step |
# | --- | --- | --- | --- |
# | DDP | none | $(2+2+K)\Psi$ | all-reduce grads: $2\Psi$ |
# | ZeRO-1 ($P_{os}$) | optimizer state | $2\Psi + 2\Psi + K\Psi/N_d$ | reduce-scatter grads + all-gather params: $2\Psi$ |
# | ZeRO-2 ($P_{os+g}$) | + gradients | $2\Psi + (2+K)\Psi/N_d$ | same: $2\Psi$ |
# | ZeRO-3 ($P_{os+g+p}$) = FSDP | + parameters | $(2+2+K)\Psi/N_d$ | all-gather params in fwd **and** bwd + reduce-scatter grads: $3\Psi$ |
#
# Communication is in units of "$\Psi$ elements through each rank's link", counting all-reduce as RS + AG
# ($\Psi$ each, up to the $(N-1)/N$ factor). ZeRO-1/2 cost **nothing extra** over DDP: the all-reduce is
# split into its two halves and the optimizer step goes between them. ZeRO-3 re-gathers parameters for the
# backward pass, so 1.5× DDP.
#
# ```
# who holds what, N=4, flat parameter vector split into shards s0..s3
#                params        grads         Adam m,v (+fp32 master)
# DDP      r_i:  s0 s1 s2 s3   s0 s1 s2 s3   s0 s1 s2 s3
# ZeRO-1   r_i:  s0 s1 s2 s3   s0 s1 s2 s3   s_i
# ZeRO-2   r_i:  s0 s1 s2 s3   s_i           s_i
# ZeRO-3   r_i:  s_i (+ one unit gathered)   s_i           s_i
# ```
#
# **Your task:** write both formulas; the checks reproduce ZeRO's Figure 1 ($\Psi = 7.5$B, $N_d = 64$:
# 120 / 31.4 / 16.6 / 1.88 GB, with GB = $10^9$ bytes).

# %%
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from prep import Session, check, check_close

s = Session("G1.12", target_min=110)
NGPU = torch.cuda.device_count()
NPROC = min(4, NGPU) if NGPU >= 2 else 2
BACKEND = "nccl" if NGPU >= 2 else "gloo"
RUN = Path("runs_G1.12")
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
def zero_model_state_bytes(psi, n_d, stage, K=12):
    """Bytes per GPU for params + grads + optimizer states. stage 0 = DDP, 1/2/3 = ZeRO stages."""
    ### BEGIN SOLUTION
    p, g, o = 2 * psi, 2 * psi, K * psi
    if stage >= 1:
        o /= n_d
    if stage >= 2:
        g /= n_d
    if stage >= 3:
        p /= n_d
    return p + g + o
    ### END SOLUTION


def zero_comm_elements(psi, stage):
    """Elements each rank sends+receives per step, counting an all-reduce as RS + AG (Psi each), ignoring (N-1)/N."""
    ### BEGIN SOLUTION
    return 3 * psi if stage == 3 else 2 * psi
    ### END SOLUTION


psi = 7.5e9
for st, gb in [(0, 120), (1, 31.4), (2, 16.6), (3, 1.875)]:
    check_close(f"ZeRO Fig.1 stage {st}", zero_model_state_bytes(psi, 64, st) / 1e9, gb, rtol=5e-3)
check_close("ZeRO-3 comm / DDP comm", zero_comm_elements(psi, 3) / zero_comm_elements(psi, 0), 1.5)
check_close("ZeRO-1 comm / DDP comm", zero_comm_elements(psi, 1) / zero_comm_elements(psi, 0), 1.0)
for n in (8, 64, 1024):
    print(f"N_d={n:5d}: " + "  ".join(f"stage{st} {zero_model_state_bytes(70e9, n, st)/1e9:8.1f} GB"
                                      for st in range(4)) + "   (70B model)")

# %% [markdown]
# For a 70B model, ZeRO-1/2 hit a floor of $4\Psi$ = 280 GB (resp. $2\Psi$) per GPU however many GPUs you
# add: replicated parameters don't fit. Only ZeRO-3/FSDP (or tensor/pipeline parallelism, G1.13-G1.14) gets
# below one GPU's 80 GB. Activations come on top of all of this (G1.16).
#
# ## 2. ZeRO-1 sharded AdamW from collectives
# Flatten all parameters into one vector of length $P$, pad it to a multiple of $N$, and let rank $r$ own the
# contiguous slice $[rP/N, (r+1)P/N)$. One step:
# 1. flatten this rank's local gradients (from its data shard) and **reduce-scatter** them, then divide by $N$:
#    rank $r$ now has the *averaged* gradient of its slice only;
# 2. **gradient clipping needs the global norm**: $\|g\|^2 = \sum_r \|g_r\|^2$, so all-reduce the shard's squared
#    norm (a single scalar), then scale the shard by $\min(1, c/(\|g\|+10^{-6}))$ exactly like `clip_grad_norm_`;
# 3. run an ordinary `torch.optim.AdamW` on a 1-D fp32 `nn.Parameter` holding the slice (AdamW is
#    elementwise, so updating a slice equals slicing the full update; the slice plays the role of ZeRO's fp32
#    master copy);
# 4. **all-gather** the updated slices into the full flat vector and copy them back into the model parameters.
#
# Steps 1 + 4 are exactly an all-reduce split in half, so communication equals DDP's. Our version still keeps full
# gradients (ZeRO-1 memory). ZeRO-2 would reduce-scatter bucket by bucket during backward and free each full
# gradient right away. Return the global (pre-clip) gradient norm from `step()`: the check compares it with the
# reference, which catches a missing $1/N$ that AdamW would hide (G1.11).

# %%
# %%writefile runs_G1.12/my_zero.py
import torch
import torch.distributed as dist
import torch.nn as nn


class ZeRO1AdamW:
    """Each rank keeps AdamW state for 1/N of a flat parameter vector."""

    def __init__(self, params, max_norm=None, **adamw_kwargs):
        self.params = [p for p in params if p.requires_grad]
        self.rank, self.N = dist.get_rank(), dist.get_world_size()
        self.numel = sum(p.numel() for p in self.params)
        self.shard = -(-self.numel // self.N)                 # ceil: every rank's slice has the same length
        self.max_norm = max_norm
        dev, dt = self.params[0].device, self.params[0].dtype
        flat = torch.zeros(self.shard * self.N, device=dev, dtype=dt)
        flat[: self.numel] = torch.cat([p.detach().flatten() for p in self.params])
        lo = self.rank * self.shard
        self.master = nn.Parameter(flat[lo:lo + self.shard].clone())   # this rank's slice
        self.opt = torch.optim.AdamW([self.master], **adamw_kwargs)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        """Returns the global gradient L2 norm (before clipping)."""
        dev, dt = self.master.device, self.master.dtype
        flat_g = torch.zeros(self.shard * self.N, device=dev, dtype=dt)
        flat_g[: self.numel] = torch.cat([p.grad.flatten() for p in self.params])
        ### BEGIN SOLUTION
        g = torch.empty(self.shard, device=dev, dtype=dt)
        dist.reduce_scatter_tensor(g, flat_g, op=dist.ReduceOp.SUM)
        g.div_(self.N)
        sq = g.pow(2).sum()
        dist.all_reduce(sq)
        norm = sq.sqrt()
        if self.max_norm is not None:
            g.mul_(torch.clamp(self.max_norm / (norm + 1e-6), max=1.0))
        self.master.grad = g
        self.opt.step()
        full = torch.empty(self.shard * self.N, device=dev, dtype=dt)
        dist.all_gather_into_tensor(full, self.master.detach())
        off = 0
        for p in self.params:
            p.copy_(full[off:off + p.numel()].view_as(p))
            off += p.numel()
        return norm.item()
        ### END SOLUTION

# %% [markdown]
# The harness: same init on every rank, disjoint data shards, 10 steps of AdamW with clipping (`max_norm` small
# enough that clipping is active). Reference: one process, the whole global batch, `torch.optim.AdamW` +
# `clip_grad_norm_`.

# %%
# %%writefile runs_G1.12/check_zero.py
import json

import torch
import torch.distributed as dist

from my_zero import ZeRO1AdamW
from prep.tinygpt import GPT, GPTConfig

backend = __import__("os").environ.get("BACKEND", "nccl")
lr_ = int(__import__("os").environ["LOCAL_RANK"])
dev = torch.device("cuda", lr_) if backend == "nccl" else torch.device("cpu")
if backend == "nccl":
    torch.cuda.set_device(dev)
dist.init_process_group(backend, device_id=dev if backend == "nccl" else None)
rank, N = dist.get_rank(), dist.get_world_size()

cfg = GPTConfig(vocab_size=256, context_length=32, d_model=64, n_heads=4, n_layers=2)
B, STEPS, MAX_NORM = 4, 10, 0.2
kw = dict(lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def batch(step):
    g = torch.Generator().manual_seed(100 + step)
    x = torch.randint(0, cfg.vocab_size, (N * B, cfg.context_length + 1), generator=g).to(dev)
    return x[:, :-1], x[:, 1:]


torch.manual_seed(0)
ref = GPT(cfg).to(dev)
torch.manual_seed(0)
model = GPT(cfg).to(dev)
opt_ref = torch.optim.AdamW(ref.parameters(), **kw)
opt = ZeRO1AdamW(model.parameters(), max_norm=MAX_NORM, **kw)
norms_ref, norms = [], []
for step in range(STEPS):
    x, y = batch(step)
    opt_ref.zero_grad()
    ref(x, y)[1].backward()
    norms_ref.append(torch.nn.utils.clip_grad_norm_(ref.parameters(), MAX_NORM).item())
    opt_ref.step()
    opt.zero_grad()
    model(x[rank * B:(rank + 1) * B], y[rank * B:(rank + 1) * B])[1].backward()
    norms.append(opt.step())

err = max((p - q).abs().max().item() for p, q in zip(model.parameters(), ref.parameters()))
state_bytes = sum(t.numel() * t.element_size() for st in opt.opt.state.values() for t in st.values() if t.dim() > 0)
ref_bytes = sum(t.numel() * t.element_size() for st in opt_ref.state.values() for t in st.values() if t.dim() > 0)
errs = torch.tensor([err], device=dev)
dist.all_reduce(errs, op=dist.ReduceOp.MAX)
res = {"N": N, "max_norm": MAX_NORM, "param_err": errs.item(), "norms": norms, "norms_ref": norms_ref,
       "state_bytes": state_bytes, "ref_state_bytes": ref_bytes, "numel": opt.numel}
if rank == 0:
    print(json.dumps({k: v for k, v in res.items() if "norm" not in k}))
    print("grad norms (ref):", [round(v, 3) for v in norms_ref])
    json.dump(res, open("check_zero.json", "w"))
dist.destroy_process_group()

# %%
torchrun("check_zero.py")
z = load("check_zero.json")
check("clipping is active in the test (otherwise it would not test anything)", min(z["norms_ref"]) > z["max_norm"],
      f"pre-clip norms {min(z['norms_ref']):.3f}.. must exceed max_norm={z['max_norm']}")
check_close("global grad norm every step == reference clip_grad_norm_", z["norms"], z["norms_ref"], rtol=1e-4)
check("ZeRO-1 params == unsharded AdamW after 10 steps", z["param_err"] < 1e-4, f"max |diff| {z['param_err']:.2e}")
check_close("optimizer state per rank = 1/N of unsharded", z["state_bytes"] / z["ref_state_bytes"], 1 / z["N"], rtol=1e-2)
print(f"Adam state per rank: {z['state_bytes']/1e3:.0f} kB vs {z['ref_state_bytes']/1e3:.0f} kB unsharded")

# %% [markdown]
# ## 3. FSDP2: `fully_shard` on tinygpt
# `fully_shard(module)` turns every parameter of `module` (not already owned by a child FSDP module) into a
# **DTensor sharded on dim 0** across the mesh, and groups them into one communication unit. Typical use:
# `fully_shard` each transformer block, then the root (which picks up embeddings, final norm and head; tied weights
# stay together in the root unit). At runtime, per unit:
# - **before forward:** all-gather the unit's shards into full parameters (prefetched: the next unit's all-gather is
#   issued while the current one computes);
# - **after forward:** free the gathered parameters if `reshard_after_forward=True`. That is ZeRO-3: re-gather in
#   backward. `False` keeps them until backward, which uses more memory and avoids one all-gather. That is ZeRO-2
#   style, and the default for the root, which would be re-gathered immediately anyway;
# - **backward:** all-gather again (if resharded), compute grads, then **reduce-scatter** them so each rank keeps
#   only its gradient shard; the optimizer then updates only local shards. A plain `torch.optim.AdamW` works,
#   because the parameters *are* the shards.
# - `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)` all-gathers and computes in bf16
#   (half the all-gather bytes) and reduces gradients in fp32. The sharded fp32 parameters act as master weights.
#
# **Your task:** write `shard_model` (one `fully_shard` per block, then the root). Then predict the steady-state
# memory per rank after `optimizer.step()` (fp32 params + fp32 grads + AdamW's 2 states) for DDP and for FSDP, and
# compare with the measurement. The harness also wraps `dist.all_gather_single` / `dist.reduce_scatter_single`
# (the functions FSDP2 calls in torch 2.14) and a DDP comm hook to count bytes per step.

# %%
# %%writefile runs_G1.12/fsdp_run.py
import json
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP

from prep.tinygpt import GPT, GPTConfig, num_params

local_rank = int(os.environ["LOCAL_RANK"])
backend = os.environ.get("BACKEND", "nccl")
dev = torch.device("cuda", local_rank) if backend == "nccl" else torch.device("cpu")
if backend == "nccl":
    torch.cuda.set_device(dev)
dist.init_process_group(backend, device_id=dev if backend == "nccl" else None)
rank, N = dist.get_rank(), dist.get_world_size()


def shard_model(model, reshard_after_forward=True, mp_policy=None):
    """FSDP2: fully_shard every transformer block, then the root module. Returns the model."""
    mp_policy = mp_policy or MixedPrecisionPolicy()
    ### BEGIN SOLUTION
    for blk in model.blocks:
        fully_shard(blk, reshard_after_forward=reshard_after_forward, mp_policy=mp_policy)
    fully_shard(model, reshard_after_forward=reshard_after_forward, mp_policy=mp_policy)
    return model
    ### END SOLUTION


# ---- byte counters around the collectives FSDP2 uses (torch 2.14: dist.all_gather_single / reduce_scatter_single)
COUNT = {"all_gather": 0, "reduce_scatter": 0, "all_reduce": 0}
_ag, _rs = dist.all_gather_single, dist.reduce_scatter_single


def ag(output_tensor, input_tensor, *a, **k):
    COUNT["all_gather"] += output_tensor.numel() * output_tensor.element_size()
    return _ag(output_tensor, input_tensor, *a, **k)


def rs(output, input, *a, **k):
    COUNT["reduce_scatter"] += input.numel() * input.element_size()
    return _rs(output, input, *a, **k)


dist.all_gather_single, dist.reduce_scatter_single = ag, rs


def counting_hook(state, bucket):
    COUNT["all_reduce"] += bucket.buffer().numel() * bucket.buffer().element_size()
    return default_hooks.allreduce_hook(None, bucket)


cfg = GPTConfig(vocab_size=8192, context_length=256, d_model=768, n_heads=12, n_layers=8)
B, STEPS = 8, 6
g = torch.Generator().manual_seed(rank)
data = torch.randint(0, cfg.vocab_size, (STEPS, B, cfg.context_length + 1), generator=g).to(dev)


def run(mode):
    torch.manual_seed(0)
    model = GPT(cfg).to(dev)
    psi = num_params(model)
    if mode == "ddp":
        model = DDP(model, device_ids=[local_rank] if dev.type == "cuda" else None, gradient_as_bucket_view=True)
        model.register_comm_hook(None, counting_hook)
    else:
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if mode == "fsdp_bf16" else None
        model = shard_model(model, reshard_after_forward=(mode != "fsdp_no_reshard"), mp_policy=mp)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    losses, times = [], []
    for step in range(STEPS):
        if step == 2 and dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for k in COUNT:
            COUNT[k] = 0
        x, y = data[step, :, :-1], data[step, :, 1:]
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=False)
        loss = model(x, y)[1]
        loss.backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        lt = loss.detach().float().clone()
        dist.all_reduce(lt)
        losses.append(lt.item() / N)
    out = {"psi": psi, "losses": losses, "step_ms": 1e3 * sorted(times[2:])[len(times[2:]) // 2], "bytes": dict(COUNT)}
    if dev.type == "cuda":
        out["static_MB"] = torch.cuda.memory_allocated() / 2**20      # after step: params + grads + Adam states
        out["peak_MB"] = torch.cuda.max_memory_allocated() / 2**20
    if mode == "fsdp":
        w = model.blocks[0].mlp[0].weight
        out["dtensor"] = {"type": type(w).__name__, "placements": str(w.placements),
                          "global_shape": list(w.shape), "local_shape": list(w.to_local().shape)}
    del model, opt, loss
    __import__("gc").collect()          # FSDP modules hold reference cycles; collect before measuring the next mode
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


res = {"N": N}
for mode in ("ddp", "fsdp", "fsdp_no_reshard", "fsdp_bf16"):
    res[mode] = run(mode)
    if rank == 0:
        r = res[mode]
        print(f"{mode:16s} step {r['step_ms']:7.1f} ms  static {r.get('static_MB', 0):7.1f} MiB  "
              f"peak {r.get('peak_MB', 0):7.1f} MiB  bytes {r['bytes']}")
if rank == 0:
    print("DTensor:", res["fsdp"]["dtensor"])
    json.dump(res, open("fsdp.json", "w"), indent=1)
dist.destroy_process_group()

# %%
torchrun("fsdp_run.py")
f = load("fsdp.json")
N, psi = f["N"], f["fsdp"]["psi"]
print(f"N={N}, Psi = {psi/1e6:.1f}M parameters, fp32")

# %%
def predicted_static_bytes(psi, n, mode):
    """fp32 params + fp32 grads + 2 fp32 AdamW states, per rank, after optimizer.step().
    mode 'ddp' (gradient_as_bucket_view=True: grads live in the buckets, no extra copy) or 'fsdp'."""
    ### BEGIN SOLUTION
    total = (4 + 4 + 8) * psi
    return total if mode == "ddp" else total / n
    ### END SOLUTION


check_close("FSDP Shard(0) local shape", f["fsdp"]["dtensor"]["local_shape"][0],
            -(-f["fsdp"]["dtensor"]["global_shape"][0] // N))
print("DTensor parameter:", f["fsdp"]["dtensor"])
if "static_MB" in f["ddp"]:
    for mode, key in (("ddp", "ddp"), ("fsdp", "fsdp")):
        meas = f[key]["static_MB"] * 2**20
        check_close(f"{mode}: measured steady-state memory vs 16 Psi{'' if mode == 'ddp' else '/N'}",
                    meas, predicted_static_bytes(psi, N, mode), rtol=0.10)
    print(f"{'mode':16s} {'step ms':>8s} {'static MiB':>11s} {'peak MiB':>9s}")
    for mode in ("ddp", "fsdp", "fsdp_no_reshard", "fsdp_bf16"):
        print(f"{mode:16s} {f[mode]['step_ms']:8.1f} {f[mode]['static_MB']:11.1f} {f[mode]['peak_MB']:9.1f}")
    check("FSDP peak memory < DDP peak memory", f["fsdp"]["peak_MB"] < f["ddp"]["peak_MB"])
    check("reshard_after_forward=False costs memory (keeps gathered params fwd -> bwd)",
          f["fsdp_no_reshard"]["peak_MB"] > f["fsdp"]["peak_MB"])
check_close("FSDP fp32 losses == DDP losses", f["fsdp"]["losses"], f["ddp"]["losses"], rtol=1e-4)
check_close("FSDP bf16 losses ~ DDP fp32 losses", f["fsdp_bf16"]["losses"], f["ddp"]["losses"], rtol=2e-2)

# %% [markdown]
# **Communication volume.** Convert the counted bytes to per-rank link traffic with the busbw factors from G1.10
# (all-reduce $2\frac{N-1}{N}S$, all-gather / reduce-scatter $\frac{N-1}{N}S$ with $S$ the full size) and compare
# FSDP with DDP.

# %%
def link_traffic(b, n):
    return (2 * b["all_reduce"] + b["all_gather"] + b["reduce_scatter"]) * (n - 1) / n


for mode in ("ddp", "fsdp", "fsdp_no_reshard", "fsdp_bf16"):
    bb = f[mode]["bytes"]
    print(f"{mode:16s} AG {bb['all_gather']/2**20:8.1f} MiB  RS {bb['reduce_scatter']/2**20:8.1f} MiB  "
          f"AR {bb['all_reduce']/2**20:8.1f} MiB  -> link traffic {link_traffic(bb, N)/2**20:8.1f} MiB/step")
ratio = link_traffic(f["fsdp"]["bytes"], N) / link_traffic(f["ddp"]["bytes"], N)
check_close("ZeRO-3 / DDP traffic = 1.5 (ZeRO §7)", ratio, 1.5, rtol=0.05)
check("FSDP all-gathers the params twice per step (fwd + bwd)",
      abs(f["fsdp"]["bytes"]["all_gather"] / f["fsdp"]["bytes"]["reduce_scatter"] - 2) < 0.1)
print(f"no_reshard: {link_traffic(f['fsdp_no_reshard']['bytes'], N) / link_traffic(f['ddp']['bytes'], N):.2f}x DDP; "
      f"bf16 params: {link_traffic(f['fsdp_bf16']['bytes'], N) / link_traffic(f['ddp']['bytes'], N):.2f}x DDP")

# %% [markdown]
# **Reading the results.**
# - Steady-state memory follows $16\Psi$ (DDP) vs $16\Psi/N$ (FSDP) to within a few percent (padding, the data
#   tensors, allocator rounding). Peak memory adds activations and, for FSDP, the gathered parameters of the
#   units in flight. With `reshard_after_forward=False`, every block's gathered parameters survive until its
#   backward: peak memory rises toward DDP's, and traffic drops to ZeRO-2's $2\Psi$ because the backward
#   all-gathers disappear.
# - With the bf16 policy the all-gathers carry half the bytes. The reduce-scatter stays fp32 because we asked for
#   `reduce_dtype=float32`. Losses agree with fp32 to ~1e-3 relative, and the step is faster than fp32 FSDP (bf16
#   tensor cores and half the all-gather bytes).
# - Step time: on this PCIe box FSDP is slower than DDP. It moves 1.5× the bytes, and the all-gathers sit on the
#   forward critical path unless they are prefetched far enough ahead. FSDP2 prefetches one unit by default.
#   `module.set_modules_to_forward_prefetch([...])` / `set_modules_to_backward_prefetch` issue all-gathers earlier,
#   trading memory for overlap. On NVLink the extra traffic is cheap, and FSDP is how you train models whose
#   $16\Psi$ does not fit on one GPU.
#
# ## Say it out loud
# 1. Write the per-GPU memory of DDP and ZeRO-1/2/3 for $\Psi$ parameters with mixed-precision Adam. Where does
#    $K=12$ come from?
# 2. Why does ZeRO-1/2 cost no extra communication over DDP, and why does ZeRO-3 cost 1.5×?
# 3. In your sharded optimizer, why can AdamW be applied to an arbitrary contiguous slice of the flattened parameters?
#    Which common training-loop operations need a collective in a sharded optimizer?
# 4. Walk through one FSDP training step for one transformer block: what is gathered, freed, reduced, and when?
# 5. What does `reshard_after_forward` trade? Why is the root module's default `False`?
# 6. How does FSDP mixed precision work: which copy is the master weight, and what dtypes are the all-gather and
#    the reduce-scatter in?
# 7. FSDP vs DDP on a slow interconnect: which is faster and why? When would you still choose FSDP?
# 8. What is HSDP (2-D mesh) and when do you use it?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. DDP $(2+2+K)\Psi = 16\Psi$; ZeRO-1 $4\Psi + K\Psi/N$; ZeRO-2 $2\Psi + 14\Psi/N$; ZeRO-3 $16\Psi/N$ (ZeRO §3,
#    §5). $K = 12$: fp32 master weights (4 B), Adam $m$ (4 B), Adam $v$ (4 B). The 2+2 are the bf16/fp16
#    working weights and gradients. Pure fp32 training without a separate master copy is $4+4+8 = 16$ bytes/param
#    too.
# 2. An all-reduce is a reduce-scatter followed by an all-gather. ZeRO-1/2 reduce-scatter the gradients (each rank
#    gets the averaged gradient of the shard it owns), update the shard, then all-gather the parameters: the same
#    two halves, $2\Psi$ total. ZeRO-3 doesn't keep full parameters, so it must all-gather them before forward
#    *and again* before backward ($2\Psi$), plus reduce-scatter gradients ($\Psi$): $3\Psi$ (ZeRO §7).
# 3. AdamW (and SGD, momentum, weight decay) is elementwise in the parameters and their state, so slicing commutes
#    with the update. Operations that couple elements need collectives: global gradient-norm clipping (all-reduce
#    the squared shard norms), the global norm for logging, LAMB/LARS-style per-tensor trust ratios (a shard
#    boundary can split a tensor, so per-tensor norms need care), Shampoo/Muon-style matrix preconditioners (need
#    whole matrices), and checkpointing (gather or save sharded).
# 4. Forward: all-gather the block's parameter shards (ideally prefetched during the previous block), compute,
#    free the gathered parameters (`reshard_after_forward=True`). Backward: all-gather again (prefetched during the
#    next-later block's backward), compute input and weight gradients, free the gathered parameters,
#    reduce-scatter the full gradients so each rank keeps its gradient shard (averaged), free the full gradients.
#    Optimizer: update local shards only.
# 5. Memory vs communication: freeing after forward saves $\approx$ the unsharded parameters of every unit
#    between its forward and its backward, at the cost of one extra all-gather per unit ($\Psi$). The root is the
#    last unit in forward and the first in backward, so re-gathering it immediately would be pure waste.
# 6. The sharded fp32 parameters (and fp32 optimizer state) are the master weights. Before each use FSDP casts the
#    shard to `param_dtype` (bf16) and all-gathers bf16, halving all-gather bytes, then computes in bf16. Gradients
#    are produced in bf16 and reduce-scattered in `reduce_dtype` (fp32 here, for accuracy), and land as fp32
#    sharded gradients for the fp32 optimizer step.
# 7. DDP: 2Ψ of traffic, overlapped with backward. FSDP: 3Ψ, with all-gathers on the forward path, so on a slow
#    link it is slower (as measured here). Choose FSDP when $16\Psi$ + activations does not fit one GPU, or when
#    freeing memory allows larger micro-batches (which improves arithmetic intensity and hides the communication).
# 8. Hybrid sharding: a 2-D mesh where parameters are sharded (FSDP) within a group (usually a node, over fast
#    NVLink) and replicated across groups (DDP-style all-reduce of the sharded gradients across nodes). The
#    expensive all-gathers stay on the fast intra-node links, while memory savings are limited to 1/(group size).
#    In FSDP2 you pass a 2-D `DeviceMesh` to `fully_shard`.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
