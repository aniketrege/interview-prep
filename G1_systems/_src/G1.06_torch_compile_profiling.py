# %% [markdown]
# # G1.6 · Profiling, benchmarking hygiene and `torch.compile`
#
# **ID:** G1.6 · **Target:** 90 min · **Plan week:** 15 (Saturday mini-project: "speed up your week-4 GPT") · **Part:** G1
#
# **The lab.** Before optimizing, find out where the time goes. You will profile a `tinygpt` training step with
# `torch.profiler` (CPU vs CUDA time, kernel list, Chrome trace), compute from the trace how busy the GPU is and how
# much of it is GEMM, then learn what `torch.compile` does (Dynamo → AOTAutograd → Inductor → Triton), find and fix a
# graph break, predict and measure the fusion speedup of a pointwise chain, compile a whole training step, and see
# when `mode="reduce-overhead"` (CUDA graphs) pays off.
#
# Guide G1: *"Kernels at a glance: a Triton softmax or matmul kernel; `torch.compile`."* Companion reading:
# S. Raschka's "PyTorch Performance Tips for Faster LLM Training", `../../LLMs-from-scratch/ch05/10_llm-training-speed/`
# (README + `00_orig.py` → `01_opt_single_gpu.py`): tensor cores, fused AdamW, bf16, native LayerNorm/GELU,
# FlashAttention, `torch.compile`, vocabulary padding to a multiple of 64, larger batch: 12.5k → 142k tokens/s on an A100.
#
# Fill in every `raise NotImplementedError`. Solution runtime: about 1.5 min (most of it is compilation).
#
# ## Sources
# - J. Ansel et al., *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph
#   Compilation*, ASPLOS 2024, DOI: 10.1145/3620665.3640366 (TorchDynamo, AOTAutograd, TorchInductor) — https://doi.org/10.1145/3620665.3640366
# - PyTorch docs: `torch.compile` / `torch.compiler` https://docs.pytorch.org/docs/stable/torch.compiler.html ;
#   graph breaks and `torch._dynamo.explain` https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html ;
#   `torch.profiler` https://docs.pytorch.org/docs/stable/profiler.html ; CUDA graphs
#   https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
# - H. He, *Making Deep Learning Go Brrrr From First Principles* (overhead, operator fusion: `x.cos().cos()` costs 2
#   memory round-trips fused vs 4 unfused) — https://horace.io/brrr_intro.html
# - S. Raschka, *LLMs-from-scratch*, `ch05/10_llm-training-speed/README.md` (Apache-2.0, commit f9313ea; local copy
#   `../../LLMs-from-scratch/ch05/10_llm-training-speed/`) — https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/10_llm-training-speed
# - Stanford CS336 Spring 2026 A2 handout §2.1.3-2.1.4 (benchmarking, Nsight Systems, NVTX), §3.1.1 (fusion with
#   `torch.compile`), §4.2 (compiling attention) — https://github.com/stanford-cs336/assignment2-systems @ca8bc81 (MIT)
# - NVIDIA, *Matrix Multiplication Background User's Guide* (dimensions that are multiples of 8/64 for tensor cores) —
#   https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html

# %%
import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.profiler import ProfilerActivity, profile
from triton.testing import do_bench

from prep import Session, check, check_close, seed_everything
from prep.tinygpt import GPT, GPTConfig, random_batch

s = Session("G1.6", target_min=90)
seed_everything(0)
dev = torch.device("cuda")
OUT = tempfile.mkdtemp(prefix="g1_6_")  # traces go here, not into the repo
print("artifacts in", OUT)

# %% [markdown]
# ## 1. Benchmarking hygiene (checklist)
# From G1.1 and CS336 A2 §2.1.3: (1) synchronize, or time with CUDA events / `triton.testing.do_bench` /
# `torch.utils.benchmark.Timer`; (2) warm up, and **never include compilation or autotuning** in the timed region;
# (3) report a median over many iterations and look at the spread; (4) keep inputs on the device and avoid hidden
# syncs (`.item()`, `print(tensor)`, `.cpu()`, data-dependent shapes); (5) mind caches (L2, cuDNN/cuBLAS heuristics,
# `torch.backends.cudnn.benchmark`); (6) quiet machine, stable clocks, note the GPU and versions; (7) measure the
# thing you care about end-to-end (tokens/s of the real step), then drill down with a profiler.

# %% [markdown]
# ## 2. Profile a training step
#
# `torch.profiler.profile(activities=[CPU, CUDA])` records every ATen op on the CPU thread and every kernel on the GPU
# (via CUPTI), with the link between them. `key_averages().table()` aggregates by op: **Self CPU** is time on the CPU
# inside that op excluding children; **CUDA total** is the GPU time of kernels it launched. `export_chrome_trace()`
# writes a timeline you can open in https://ui.perfetto.dev or `chrome://tracing`.
#
# **How to read a trace.** The top rows are CPU threads (Python → `aten::` ops → `cudaLaunchKernel`); below, the GPU
# stream with kernels as boxes. Arrows join each launch to its kernel. If the CPU row is busy and the GPU row has
# **gaps**, you are overhead-bound (launch-bound). If the GPU row is solid, look at *which* kernels dominate: GEMMs
# (`*gemm*`, `cutlass`, `xmma`) are compute; elementwise/reduction kernels are memory-bound and candidates for fusion.

# %%
cfg = GPTConfig()


def make_step(B, model=None):
    torch.manual_seed(0)
    m = model if model is not None else GPT(cfg).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    x, y = random_batch(cfg, B, dev, torch.Generator(device=dev).manual_seed(0))

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = m(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    return step


def profile_steps(step, n=3):
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for _ in range(n):
            step()
        torch.cuda.synchronize()
        wall_us = (time.perf_counter() - t0) * 1e6
    return prof, wall_us


prof8, wall8 = profile_steps(make_step(8))
print(prof8.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=50))
trace = os.path.join(OUT, "tinygpt_step_trace.json")
prof8.export_chrome_trace(trace)
print(f"chrome trace: {trace} ({os.path.getsize(trace) / 1e6:.1f} MB)")

# %% [markdown]
# **Exercise 2.1.** From the raw kernel events, compute (a) the fraction of GPU kernel time in GEMMs, attention, and
# everything else, and (b) how busy the GPU was: the length of the **union** of kernel intervals divided by the wall
# time. Kernel events are the entries of `prof.events()` whose `device_type` is CUDA; each has `.name` and
# `.time_range.start/.end` (µs). Classify a kernel as attention if its lowercased name contains any of
# `("flash", "fmha", "attention")`, **else** as GEMM if it contains any of `("gemm", "cutlass", "xmma", "s16816",
# "s1688")` (test attention first: the flash kernels are built on CUTLASS templates, so their names contain "cutlass").

# %%
def kernel_events(prof):
    return [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def time_by_category(prof):
    """Return {'gemm': f, 'attention': f, 'other': f}: fractions of total kernel time (they sum to 1)."""
    ### BEGIN SOLUTION
    tot = {"gemm": 0.0, "attention": 0.0, "other": 0.0}
    for e in kernel_events(prof):
        n = e.name.lower()
        dur = e.time_range.end - e.time_range.start
        if any(t in n for t in ("flash", "fmha", "attention")):
            tot["attention"] += dur
        elif any(t in n for t in ("gemm", "cutlass", "xmma", "s16816", "s1688")):
            tot["gemm"] += dur
        else:
            tot["other"] += dur
    z = sum(tot.values())
    return {k: v / z for k, v in tot.items()}
    ### END SOLUTION


def gpu_busy_fraction(prof, wall_us):
    """Length of the union of kernel time intervals / wall time."""
    ### BEGIN SOLUTION
    iv = sorted((e.time_range.start, e.time_range.end) for e in kernel_events(prof))
    busy, cur_s, cur_e = 0.0, None, None
    for a, b in iv:
        if cur_e is None or a > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    busy += cur_e - cur_s
    return busy / wall_us
    ### END SOLUTION


cats = time_by_category(prof8)
print({k: f"{v:.0%}" for k, v in cats.items()})
check_close("categories sum to 1", sum(cats.values()), 1.0)
ksum = sum(e.time_range.end - e.time_range.start for e in kernel_events(prof8))
check("total kernel time fits inside the wall time (one stream)", ksum <= 1.02 * wall8)
check("GEMMs are a large share but not everything", 0.3 < cats["gemm"] < 0.95)
check("the flash-attention kernels are found", 0.005 < cats["attention"] < 0.3)
# union test on synthetic intervals
class _E:  # minimal stand-in for a profiler kernel event
    def __init__(self, a, b):
        self.name, self.device_type = "k", torch.autograd.DeviceType.CUDA
        self.time_range = type("R", (), {"start": a, "end": b})()
class _P:
    def __init__(self, ev): self._ev = ev
    def events(self): return self._ev
check_close("busy fraction handles overlaps and gaps", gpu_busy_fraction(_P([_E(0, 4), _E(2, 6), _E(8, 9)]), 10.0), 0.7)

busy = {}
for B in (1, 8, 32):
    p, w = profile_steps(make_step(B)) if B != 8 else (prof8, wall8)
    busy[B] = gpu_busy_fraction(p, w)
    print(f"batch {B:3d}: GPU busy {busy[B]:.0%} of wall time")
check("small batches leave the GPU idle more (overhead-bound)", busy[1] < busy[32])

# %% [markdown]
# At batch 1 the CPU cannot launch the ~600 kernels of a step as fast as the GPU finishes them, so the GPU idles:
# making the GPU faster would not help, removing launches would (fusion, CUDA graphs, bigger batches). Also note in the
# table how much CPU and GPU time `aten::copy_` / `aten::to` take: those are autocast's bf16 casts of weights and
# activations, a cost that a compiled graph can fuse into neighbouring kernels.

# %% [markdown]
# ## 3. What `torch.compile` does
#
# `torch.compile(fn)` (Ansel et al. 2024) is three stages:
# 1. **TorchDynamo** hooks CPython's frame evaluation (PEP 523) and symbolically executes the function's *bytecode*,
#    recording tensor operations into an **FX graph** and installing **guards** (checks on dtypes, shapes, Python
#    values, module attributes) under which the graph is valid. A guard failure triggers a recompile (shapes that vary
#    become *dynamic* symbolic sizes after a recompile). Code Dynamo cannot trace (data-dependent Python control flow on
#    tensor values, `.item()`, `print`, unsupported libraries) causes a **graph break**: the graph ends, that code runs
#    in normal Python, and a new graph starts after it.
# 2. **AOTAutograd** traces the backward ahead of time into a joint forward+backward graph, functionalizes it (no
#    in-place mutation), decomposes ops into a smaller core ATen/prims set, and **partitions** it with a min-cut that
#    decides which intermediates to save for backward and which to recompute (cheap pointwise ops are recomputed).
# 3. **TorchInductor** lowers the graph to a loop-level IR, **fuses** pointwise and reduction ops into as few kernels as
#    possible, and generates **Triton** code for GPUs (C++/OpenMP for CPUs). Matmuls go to cuBLAS (or Triton templates
#    with `mode="max-autotune"`). `mode="reduce-overhead"` additionally records the whole graph into a **CUDA graph** and
#    replays it with a single launch.
#
# ### 3.1 Graph breaks
# **Exercise 3.1.** `f` below has two graph breaks. Write `f_fixed` with the same math for finite inputs and **zero**
# graph breaks (hint: `torch.where` instead of Python `if` on a tensor value; drop the print).

# %%
def f(x):
    y = torch.sin(x) * 2
    if y.sum().item() > 0:  # data-dependent Python control flow: needs the value on the CPU -> graph break
        y = y + 1
    else:
        y = y - 1
    print("debug")  # side effect Dynamo does not trace -> graph break
    return torch.cos(y)


def f_fixed(x):
    ### BEGIN SOLUTION
    y = torch.sin(x) * 2
    y = torch.where(y.sum() > 0, y + 1, y - 1)
    return torch.cos(y)
    ### END SOLUTION


xg = torch.randn(1000, device=dev)
torch._dynamo.reset()
ex = torch._dynamo.explain(f)(xg)
print(f"f: {ex.graph_count} graphs, {ex.graph_break_count} breaks")
for r in ex.break_reasons:
    print("   break:", str(r.reason).splitlines()[0][:110])
torch._dynamo.reset()
ex2 = torch._dynamo.explain(f_fixed)(xg)
check("f_fixed has no graph breaks", ex2.graph_break_count == 0 and ex2.graph_count == 1)
for sign in (1, -1):
    xs_ = sign * xg.abs() + 0.1 * sign
    check_close(f"f_fixed matches f (sum sign {sign:+d})", torch.compile(f_fixed)(xs_), f(xs_), atol=1e-6)
# In a script, `TORCH_LOGS="graph_breaks,recompiles"` prints breaks and recompilations as they happen.

# %% [markdown]
# ### 3.2 Fusion of a pointwise chain
#
# `out = dropout(gelu(x + b)) + r` on a $4096\times4096$ fp32 tensor, under `torch.no_grad()` (so no mask is kept for
# a backward). Eager runs four kernels; each reads its inputs from DRAM and writes its output. Inductor emits **one**
# Triton kernel that reads `x`, `b`, `r` and writes `out`, generating the dropout random numbers in registers.
#
# **Exercise 3.2.** Count DRAM bytes per element (ignore the bias vector $b$, it is tiny): eager = `x + b` (read 4, write
# 4) + GELU (read 4, write 4) + dropout (read 4, write 4, plus a 1-byte bool mask that eager dropout always writes) +
# `+ r` (read 8, write 4); fused = read `x` and `r`, write `out`. **Predict** the speedup as the ratio, then measure.

# %%
def eager_bytes_per_elem():
    ### BEGIN SOLUTION
    return (4 + 4) + (4 + 4) + (4 + 4 + 1) + (8 + 4)
    ### END SOLUTION


def fused_bytes_per_elem():
    ### BEGIN SOLUTION
    return 4 + 4 + 4
    ### END SOLUTION


def chain(x, b, r):
    return F.dropout(F.gelu(x + b, approximate="tanh"), p=0.1, training=True) + r


cx = torch.randn(4096, 4096, device=dev)
cb, cr = torch.randn(4096, device=dev), torch.randn_like(cx)
chain_c = torch.compile(chain)
with torch.no_grad():
    _, codes = run_and_get_code(chain_c, cx, cb, cr)
    t_eager, t_comp = do_bench(lambda: chain(cx, cb, cr)), do_bench(lambda: chain_c(cx, cb, cr))
pred = eager_bytes_per_elem() / fused_bytes_per_elem()
print(f"eager {t_eager * 1e3:.0f} us, compiled {t_comp * 1e3:.0f} us: speedup {t_eager / t_comp:.2f}x (predicted {pred:.2f}x)")
print(f"compiled kernel bandwidth: {fused_bytes_per_elem() * cx.numel() / (t_comp * 1e-3) / 1e9:.0f} GB/s")
check("byte counts", (eager_bytes_per_elem(), fused_bytes_per_elem()) == (37, 12))
check("measured fusion speedup within 25% of the byte-count prediction", abs(t_eager / t_comp / pred - 1) < 0.25)

src = codes[0]
i = src.find("@triton.jit")
print("\n".join(src[i:].splitlines()[:28]))  # the generated Triton kernel (first lines)
with torch.no_grad():  # correctness without randomness: p=0 path
    chain0 = lambda x, b, r: F.gelu(x + b, approximate="tanh") + r
    check_close("compiled pointwise chain matches eager (no dropout)", torch.compile(chain0)(cx, cb, cr), chain0(cx, cb, cr), atol=1e-5)

# %% [markdown]
# Read the generated kernel: one `tl.program_id`, a `tl.load` per input (the bias with `evict_last`, since every
# row reuses it), Philox random numbers (`triton_helpers.rand4x`) for the dropout mask, the tanh-GELU polynomial, and a
# single store. The name, `triton_poi_fused_add_gelu_native_dropout_0`, lists what was fused. Horace He's fusion argument, executed by a
# compiler. The prediction works because the op is memory-bound: time ∝ bytes.

# %% [markdown]
# ## 4. Compile a whole training step
# **Predict** the compile time and the step speedup for tinygpt at batch 8 (bf16 autocast) before running.

# %%
step_eager = make_step(8)
torch.manual_seed(0)
model_c = GPT(cfg).to(dev)
model_compiled = torch.compile(model_c)
step_comp = make_step(8, model=model_compiled)
t0 = time.perf_counter()
loss_c0 = step_comp()
torch.cuda.synchronize()
compile_s = time.perf_counter() - t0
loss_e0 = make_step(8)()  # fresh eager model with the same seed and batch
check_close("compiled first-step loss matches eager", loss_c0, loss_e0, atol=2e-2, rtol=0)


def ms_per_step(step, n=10):
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


ms_e, ms_c = ms_per_step(step_eager), ms_per_step(step_comp)
tok = 8 * cfg.context_length
print(f"compile time {compile_s:.1f} s;  eager {ms_e:.1f} ms/step ({tok / ms_e * 1e3:,.0f} tok/s)  "
      f"compiled {ms_c:.1f} ms/step ({tok / ms_c * 1e3:,.0f} tok/s): {ms_e / ms_c:.2f}x")
print(f"break-even after {compile_s * 1e3 / max(ms_e - ms_c, 1e-9):,.0f} steps")
check("compiled step is faster", ms_c < ms_e)

# %% [markdown]
# The compile time you see depends on caches: the first compile of this model on a fresh machine took ~12 s here,
# repeated runs reuse Inductor's on-disk FX-graph and Triton kernel caches (`/tmp/torchinductor_<user>`) and take about
# a second. In a real job, also count recompiles (new shapes) and the warmup in your tokens/s numbers.

# %% [markdown]
# ## 5. `mode="reduce-overhead"`: CUDA graphs for launch-bound work
#
# A CUDA graph records a fixed sequence of kernel launches (with fixed pointers and shapes) once, and replays it with
# one CPU call. It removes per-kernel CPU overhead (Python, dispatcher, launch), which matters when kernels are tiny:
# small models, batch-1 decoding, many small layers. It does nothing for a GPU that is already busy. Caveats: static
# shapes (a new shape records a new graph), inputs are copied into static buffers, outputs are overwritten by the next
# replay (clone them if you keep them), memory is reserved in a private pool, and CPU-side logic or syncs inside break it.
#
# **Predict:** for a 4-layer, $d=256$ tinygpt forward on a $1\times 64$ input, the per-call time eager vs default compile
# vs reduce-overhead.

# %%
small = GPT(GPTConfig(n_layers=4, d_model=256, n_heads=4, context_length=64)).to(dev).eval()
xi = torch.randint(0, cfg.vocab_size, (1, 64), device=dev)
per_call = {}
with torch.no_grad():
    for name, fn in (("eager", small), ("compile default", torch.compile(small)),
                     ("compile reduce-overhead", torch.compile(small, mode="reduce-overhead"))):
        for _ in range(3):
            fn(xi)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            fn(xi)
        torch.cuda.synchronize()
        per_call[name] = (time.perf_counter() - t0) / 200 * 1e3
        print(f"{name:24s} {per_call[name]:.3f} ms per forward")
check("CUDA graphs beat default compile by > 1.5x on this launch-bound model",
      per_call["compile reduce-overhead"] * 1.5 < per_call["compile default"])

# %% [markdown]
# ## 6. One more shape lesson: pad the vocabulary
# Raschka's tip 9 (from Karpathy/Megatron): $V = 50257$ is odd; $50304 = 64\times786$ lets the LM-head GEMM use full
# tensor-core tiles. Measure the head matmul alone.

# %%
h_ = torch.randn(8 * 256, 384, device=dev, dtype=torch.bfloat16)
for V in (50257, 50304):
    W_ = torch.randn(V, 384, device=dev, dtype=torch.bfloat16)
    t = do_bench(lambda: h_ @ W_.t())
    print(f"V={V}: {t * 1e3:6.1f} us, {2 * h_.shape[0] * 384 * V / (t * 1e-3) / 1e12:5.1f} TFLOP/s")

# %% [markdown]
# ## Say it out loud
# 1. How do you tell from a profile whether a training step is compute-bound, memory-bound or overhead-bound? What do
#    you do in each case?
# 2. Explain Dynamo, guards, recompilation and graph breaks. Name four things that cause graph breaks, and how you find them.
# 3. What does AOTAutograd's partitioner decide, and how does that relate to activation checkpointing?
# 4. Why did fusing four pointwise ops give ~3×? When would fusion give nothing?
# 5. When does `mode="reduce-overhead"` help, and what are its pitfalls?
# 6. Your compiled model is slower than eager on the first 20 steps. Why might that be, and when would it never pay off?
# 7. List the steps you would take to speed up a small GPT training loop (the LLMs-from-scratch progression) and which
#    regime each one attacks.

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Overhead-bound: the GPU timeline has gaps while the CPU is busy launching; GPU busy fraction well below 100%;
#    time barely changes with batch size. Fix: bigger batch, fewer and larger kernels (fusion, `torch.compile`),
#    CUDA graphs, remove syncs. Memory-bound: GPU busy, time dominated by elementwise/reduction/norm kernels with low
#    arithmetic intensity, achieved bandwidth near peak. Fix: fusion, lower precision, fewer passes (e.g.
#    FlashAttention). Compute-bound: GEMMs dominate and run near peak TFLOPS. Fix: tensor cores/lower precision, better
#    shapes (multiples of 64, avoid wave quantization), less redundant compute (no unneeded recomputation).
# 2. Dynamo intercepts Python frames (PEP 523), interprets bytecode symbolically and builds an FX graph of tensor ops,
#    plus guards that must hold for the cached graph to be reused (dtype, device, shapes or symbolic shape relations,
#    Python constants, module identity). A failed guard recompiles (with a recompile limit, after which it falls back to
#    eager). Graph breaks: `.item()`/`.tolist()` and data-dependent `if` on tensor values, `print` and other side effects,
#    calls into unsupported C extensions or libraries (e.g. NumPy on tensors in some cases), data-dependent output shapes
#    (`nonzero`), some generators/closures. Find them with `torch._dynamo.explain(fn)(*args)` or
#    `TORCH_LOGS="graph_breaks"`, or force an error with `fullgraph=True` (Ansel et al. 2024; PyTorch docs).
# 3. With the joint forward+backward graph it chooses which forward intermediates to save and which to recompute in the
#    backward, by a min-cut that minimizes saved memory/traffic subject to recompute cost (it recomputes cheap pointwise
#    ops and keeps matmul outputs). It is automatic, selective activation checkpointing at op granularity; explicit
#    `torch.utils.checkpoint` forces recompute at module granularity.
# 4. The chain is memory-bound, so time ∝ DRAM bytes: eager moves 37 bytes/element over 4 kernels, fused 12, ratio
#    ≈ 3.1, as measured. Fusion gives little for compute-bound ops (a large GEMM is already near peak; fusing an epilogue
#    saves only the epilogue's bytes), for ops already fused (cuDNN, FlashAttention), or when tensors are so small that
#    the problem is launch overhead (then CUDA graphs help more), and when a graph break separates the ops.
# 5. When the CPU cannot keep up with tiny kernels: small models, batch-1 or decode loops, many small layers. We measured
#    several times faster than default compile on a small forward. Pitfalls: needs static shapes (each new shape records
#    and stores a new graph), outputs live in static buffers overwritten by the next replay (clone them), extra memory in
#    a private pool, CPU-side control flow or host syncs cannot be inside, and RNG/mutating inputs need care.
# 6. Compilation (Dynamo tracing, AOTAutograd, Inductor codegen, Triton compilation, maybe autotuning) costs seconds to
#    minutes, and each new input shape or guard failure recompiles. Break-even = compile time ÷ per-step saving (printed
#    above). It never pays off for short runs, constantly changing shapes (dynamic padding without bucketing), or
#    programs that are dominated by already-optimal GEMMs or that graph-break everywhere.
# 7. (Raschka ch05/10) Enable TF32/tensor cores (compute), fused AdamW (fewer kernels: overhead/memory), pinned memory and
#    non-blocking copies (input pipeline), bf16 autocast (compute and memory), native fused LayerNorm/GELU (memory),
#    FlashAttention via SDPA (memory: no T×T), `torch.compile` (fusion: memory + overhead), pad the vocabulary to a
#    multiple of 64 (GEMM shape efficiency), increase batch size (overhead amortization, better GEMM utilization). Measure
#    tokens/s after each change.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
