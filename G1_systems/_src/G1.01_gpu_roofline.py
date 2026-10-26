# %% [markdown]
# # G1.1 · The GPU, arithmetic intensity and the roofline
#
# **ID:** G1.1 · **Target:** 120 min · **Plan week:** 15 (read it once in week 4 alongside FLOP counting) · **Part:** G1
#
# **The lab.** Build a mental model of the GPU from first principles, then test it on this machine. Every
# number we use is either *measured in this notebook* or *cited* (GA102 whitepaper). The loop for every section:
# **explain → predict → measure → compare with theory**.
#
# Guide G1 "Must include": *GPU basics: memory hierarchy, arithmetic intensity, roofline, why attention is
# memory-bound.* We also do "why decode is memory-bound", because every inference interview asks it.
#
# What you will do:
# 1. Read the hardware model (SMs, warps, tensor cores, registers / shared memory / L2 / GDDR6X) and check the
#    numbers against what the driver reports.
# 2. Write a correct GPU timer (CUDA events, warmup, synchronize) and see how wrong timing goes wrong.
# 3. Measure launch overhead, dependent-load latency, and copy bandwidth; fit $t = \alpha + n/\beta$.
# 4. Compute arithmetic intensity (AI) by hand for elementwise ops, matmul and attention.
# 5. Measure matmul throughput vs size for fp32 / tf32 / fp16 / bf16 and compare with the datasheet peaks.
# 6. Put every measured op on a roofline plot of *this* GPU.
# 7. Predict, then measure, where a decode-style GEMV becomes compute-bound, and why naive attention is memory-bound.
#
# **How to work.** Fill in every `raise NotImplementedError`. Make a written prediction (a number) *before* each
# measurement cell; the point is to calibrate your intuition. Runtime of the solution: about 1-2 min on an RTX 3090.
#
# ## Sources
# - NVIDIA, *NVIDIA Ampere GA102 GPU Architecture* whitepaper v2 (2021): SM layout (p. 10-13), GDDR6X 936 GB/s
#   (p. 31), Appendix A Table 9 (RTX 3090 peak rates at 1695 MHz boost: FP32 35.6 TFLOPS; FP16 tensor 142 TFLOPS
#   with FP16 accumulate, **71 with FP32 accumulate**; BF16 tensor 71; TF32 tensor 35.6; 6144 KB L2; 82 SMs).
#   https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
# - S. Williams, A. Waterman, D. Patterson, *Roofline: an insightful visual performance model for multicore
#   architectures*, CACM 52(4), 2009. DOI: 10.1145/1498765.1498785 — https://doi.org/10.1145/1498765.1498785
# - H. He, *Making Deep Learning Go Brrrr From First Principles* (compute / memory / overhead regimes, fusion),
#   https://horace.io/brrr_intro.html
# - Modal, *GPU Glossary*: roofline model, arithmetic intensity, memory-bound, compute-bound, latency hiding,
#   occupancy, streaming multiprocessor, warp, tensor core. https://modal.com/gpu-glossary
#   (e.g. https://modal.com/gpu-glossary/perf/roofline-model, https://modal.com/gpu-glossary/perf/latency-hiding)
# - J. Austin et al., *How to Scale Your Model* (Google DeepMind, 2025), chapters "All About Rooflines" and
#   "How to Think About GPUs": https://jax-ml.github.io/scaling-book/roofline/ , https://jax-ml.github.io/scaling-book/gpus/
# - NVIDIA, *GPU Performance Background User's Guide* (ops:byte ratio, latency/bandwidth) and *Matrix Multiplication
#   Background User's Guide* (tile and wave quantization):
#   https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html ,
#   https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
# - NVIDIA, *CUDA C++ Programming Guide* (thread hierarchy, warps, asynchronous execution, events):
#   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
# - Stanford CS336 Spring 2026, Assignment 2 (systems) handout §2.1.3 "End-to-End Benchmarking" (why
#   `torch.cuda.synchronize()`, warmup) — https://github.com/stanford-cs336/assignment2-systems @ca8bc81 (MIT).
#   Prose here is our own; only the benchmarking checklist structure is borrowed.
# - kipply, *Transformer Inference Arithmetic* (2022), decode is memory-bound below a batch of ~FLOPs/bandwidth:
#   https://kipp.ly/transformer-inference-arithmetic/
# - T. Dao et al., *FlashAttention*, NeurIPS 2022, arXiv:2205.14135 (attention as a memory-bound op) — https://arxiv.org/abs/2205.14135
# - PyTorch notes on TF32 (`torch.backends.cuda.matmul.fp32_precision`) and CUDA semantics:
#   https://docs.pytorch.org/docs/stable/notes/cuda.html
# - Triton `triton.testing.do_bench` (reference timer used in a check): https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html

# %%
import math
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.testing import do_bench

from prep import Session, check, check_close, seed_everything

s = Session("G1.1", target_min=120)
seed_everything(0)
assert torch.cuda.is_available(), "this lab needs a CUDA GPU"
dev = torch.device("cuda")
props = torch.cuda.get_device_properties(dev)
print(props.name, f"sm_{props.major}{props.minor}", torch.__version__, "triton", triton.__version__)

# %% [markdown]
# ## 1. The machine: what a GPU is made of
#
# A GPU is a few dozen independent processors (**streaming multiprocessors, SMs**) sharing one big, slow-ish
# DRAM through an L2 cache. Each SM runs many threads at once, grouped in **warps** of 32 threads that issue the
# same instruction together (SIMT). The programmer launches a **grid** of **thread blocks**; each block lands on
# one SM and can use that SM's fast **shared memory**. Matrix math goes to **tensor cores**, which do a small
# matrix multiply-accumulate per instruction; everything else (adds, exp, max) runs on the ordinary FP32 lanes
# ("CUDA cores").
#
# ```
#   RTX 3090 (GA102, sm_86)                                  capacity            measured below
#   ┌───────────────────────────── 82 SMs ─────────────────────────────┐
#   │  SM: 4 partitions × (1 warp scheduler, 32 FP32 lanes,             │
#   │      1 tensor core, 64 KB registers)                              │  regs: 256 KB/SM   ~1 cycle
#   │      128 KB L1 / shared memory (up to 100 KB shared)              │  L1:  128 KB/SM    "L1" latency
#   └──────────────────────────────────────────────────────────────────┘
#                    │  (all SMs)
#              ┌───────────┐
#              │ L2  6 MB  │                                                         "L2" latency
#              └───────────┘
#                    │  384-bit bus, 19.5 Gbps/pin
#              ┌──────────────────────┐
#              │ GDDR6X DRAM, 24 GB   │  936 GB/s peak (whitepaper)          ~840 GB/s copy, DRAM latency
#              └──────────────────────┘
#                    │  PCIe 4.0 x16 (~32 GB/s per direction) to the CPU and to other GPUs on this box
# ```
#
# Numbers from the GA102 whitepaper (Appendix A, Table 9, RTX 3090 Founders Edition, rates at the 1695 MHz boost clock):
#
# | Quantity | RTX 3090 | Note |
# | --- | --- | --- |
# | SMs / FP32 lanes / tensor cores | 82 / 10496 / 328 | 128 lanes and 4 tensor cores per SM |
# | Register file | 20992 KB total (256 KB per SM) | fastest storage; per-thread |
# | L1 / shared memory | 128 KB per SM (10496 KB total) | shared memory is programmer-managed |
# | L2 cache | 6144 KB | shared by all SMs |
# | DRAM | 24 GB GDDR6X, 384-bit, 19.5 Gbps → **936 GB/s** | "HBM" in papers; GDDR6X here |
# | Peak FP32 (non-tensor) | **35.6 TFLOPS** | also FP16/BF16 non-tensor: 35.6 |
# | Peak TF32 tensor | **35.6** (71 with 2:4 sparsity) | fp32 inputs rounded to 10-bit mantissa |
# | Peak FP16 tensor, FP16 accumulate | **142** | |
# | Peak FP16 / BF16 tensor, **FP32 accumulate** | **71** | GeForce runs FP32 accumulation at half rate; A100/A6000 do not |
# | FP8 tensor cores | none | FP8 arrives with Ada (sm_89) and Hopper (sm_90) |
#
# Two consequences to remember. First, PyTorch's fp16/bf16 matmuls accumulate in FP32, so the relevant peak on
# this card is **71 TFLOPS**, not 142. Second, the GPU clock is not fixed: under load the SM clock often sits
# *above* the 1695 MHz the table assumes, so a measurement can beat the "peak". We read the live clock below.
#
# **Latency hiding.** A DRAM load takes hundreds of cycles. An SM hides this by keeping many warps resident (up to
# 48 warps = 1536 threads per SM on sm_86) and switching to a ready warp every cycle. By Little's law, to sustain
# bandwidth $\beta$ with latency $\lambda$ you need $\beta\lambda$ bytes in flight. **Occupancy** (resident warps /
# maximum) is limited by registers and shared memory per block, which is why "use fewer registers" and
# "bigger tiles" pull in opposite directions (Modal glossary: latency hiding, occupancy).

# %%
# What the driver says. Compare with the table above.
print(f"SMs                      {props.multi_processor_count}")
print(f"L2 cache                 {props.L2_cache_size / 2**20:.0f} MiB")
print(f"shared mem per SM        {props.shared_memory_per_multiprocessor / 1024:.0f} KiB "
      f"(per block, opt-in max {props.shared_memory_per_block_optin / 1024:.0f} KiB)")
print(f"registers per SM         {props.regs_per_multiprocessor} x 4 B = {props.regs_per_multiprocessor * 4 / 1024:.0f} KiB")
print(f"max threads per SM       {props.max_threads_per_multi_processor} = {props.max_threads_per_multi_processor // props.warp_size} warps")
print(f"boost clock (driver)     {props.clock_rate / 1e3:.0f} MHz")
print(f"memory                   {props.total_memory / 2**30:.1f} GiB, {props.memory_bus_width}-bit bus")

check("82 SMs", props.multi_processor_count == 82, "this lab's numbers assume an RTX 3090")
check("6 MiB L2", props.L2_cache_size == 6 * 2**20)
check("256 KiB registers per SM", props.regs_per_multiprocessor * 4 == 256 * 1024)

# Cited peaks (GA102 whitepaper, Table 9), in FLOP/s and bytes/s, at the 1695 MHz boost clock.
PEAK = {"fp32": 35.6e12, "tf32": 35.6e12, "fp16": 71e12, "bf16": 71e12, "fp16_acc16": 142e12}
BW_PEAK = 936e9
BOOST_MHZ = 1695


def gpu_clocks():
    """Live SM and memory clocks of *this* GPU (MHz) from nvidia-smi, matched by UUID."""
    out = subprocess.run(["nvidia-smi", f"--id=GPU-{props.uuid}", "--query-gpu=clocks.sm,clocks.mem,clocks.max.sm",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [int(v) for v in out.strip().split(",")]


print("clocks now (sm, mem, max sm) MHz:", gpu_clocks())

# %% [markdown]
# ## 2. Timing GPU code correctly
#
# A CUDA kernel launch is **asynchronous**: `A @ B` enqueues work on a stream and returns to Python in a few
# microseconds, long before the GPU finishes. So `time.perf_counter()` around a call measures the *launch*, not
# the *kernel*. Correct recipes:
# - **CUDA events**: `start.record(); fn(); end.record(); torch.cuda.synchronize(); start.elapsed_time(end)` (ms,
#   measured on the GPU's timeline), or wall-clock with `torch.cuda.synchronize()` before reading the clock.
# - **Warm up first.** The first call pays one-time costs: CUDA context and cuBLAS handle creation, kernel
#   selection / autotuning, Triton/Inductor compilation, the caching allocator growing its pool.
# - **Repeat and take the median**; report the spread. Keep the GPU otherwise idle.
# - Know what you leave in cache: a 4 MB tensor timed in a loop stays in the 6 MB L2. `triton.testing.do_bench`
#   flushes L2 between runs for this reason.
#
# **Exercise 2.1.** Write `bench(fn)`: run `warmup` calls, then time `iters` calls, each between its own pair of
# CUDA events, and return the **median in milliseconds**.

# %%
def bench(fn, warmup=3, iters=10):
    """Median GPU time of fn() in ms, using one CUDA event pair per iteration."""
    ### BEGIN SOLUTION
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for st, en in zip(starts, ends):
        st.record()
        fn()
        en.record()
    torch.cuda.synchronize()  # events are only readable once the GPU has reached them
    return float(np.median([st.elapsed_time(en) for st, en in zip(starts, ends)]))
    ### END SOLUTION


torch.backends.cuda.matmul.fp32_precision = "ieee"  # plain FP32 (no TF32) for this check
n = 4096
A = torch.randn(n, n, device=dev)
t_bench = bench(lambda: A @ A)
t_ref = do_bench(lambda: A @ A)  # Triton's reference timer (ms)
# Physics check: nobody beats peak at the *maximum* clock (2100 MHz for this card).
t_floor = 2 * n**3 / (PEAK["fp32"] * 2100 / BOOST_MHZ) * 1e3
print(f"bench {t_bench:.3f} ms   do_bench {t_ref:.3f} ms   physical floor {t_floor:.3f} ms")
check("bench is not faster than physics allows", t_bench >= t_floor)
check("bench agrees with triton.testing.do_bench within 15%", abs(t_bench / t_ref - 1) < 0.15,
      f"{t_bench:.3f} vs {t_ref:.3f} ms")

# %%
# The classic mistake: wall clock without synchronize measures only the launch.
torch.cuda.synchronize()
t0 = time.perf_counter(); C = A @ A; t_nosync = (time.perf_counter() - t0) * 1e3
torch.cuda.synchronize()  # drain the queue, or the next timing also waits for this matmul
t0 = time.perf_counter(); C = A @ A; torch.cuda.synchronize(); t_sync = (time.perf_counter() - t0) * 1e3
print(f"no sync: {t_nosync:.3f} ms (just the launch)   with sync: {t_sync:.3f} ms   events: {t_bench:.3f} ms")

# %% [markdown]
# ## 3. Overhead, latency and bandwidth
#
# Horace He splits GPU time into three regimes: **compute** (doing FLOPs), **memory bandwidth** (moving bytes
# to and from DRAM) and **overhead** (everything else: Python, PyTorch dispatch, kernel launch). A useful first
# model of a memory-bound kernel that moves $n$ bytes is
# $$t(n) = \alpha + n/\beta,$$
# with a fixed cost $\alpha$ (launch + DRAM latency + ramp-up/tail of the kernel) and bandwidth $\beta$. The size
# $n_{1/2} = \alpha\beta$ at which you reach half of the peak bandwidth tells you how big a kernel must be before
# bandwidth, not fixed cost, dominates. Below it, the fix is fewer, bigger kernels (fusion, CUDA graphs).
#
# **Predict** before running: (a) the CPU cost of one tiny eager PyTorch op, in µs; (b) the latency of one
# dependent DRAM load, in ns; (c) the fraction of 936 GB/s a large `copy_` reaches.

# %%
# (a) Cost per tiny eager op: enqueue many tiny kernels, synchronize once.
x = torch.zeros(1, device=dev)
for _ in range(100):
    x.add_(1)
torch.cuda.synchronize()
n_ops = 5000
t0 = time.perf_counter()
for _ in range(n_ops):
    x.add_(1)
torch.cuda.synchronize()
us_eager = (time.perf_counter() - t0) / n_ops * 1e6

# The same 100 kernels captured once in a CUDA graph and replayed: no Python, no dispatcher, one launch.
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(100):
        x.add_(1)
us_graph = bench(lambda: g.replay(), warmup=3, iters=20) * 1e3 / 100
print(f"eager: {us_eager:.1f} us per tiny op (CPU-bound: the GPU waits for Python)")
print(f"CUDA graph replay: {us_graph:.1f} us per tiny kernel (what the GPU itself needs)")
check("eager tiny ops are overhead-bound (graph replay is >= 2x faster per op)", us_eager > 2 * us_graph)

# %% [markdown]
# (b) **Dependent-load latency.** A single thread follows a random cycle `i = next[i]` through a buffer, so each
# load must wait for the previous one: no parallelism, no latency hiding. Time per step = latency of the level of
# the hierarchy the buffer fits in (L1 128 KB/SM, L2 6 MB, DRAM beyond). Provided code (Triton, see G1.5).

# %%
@triton.jit
def _pointer_chase(nxt_ptr, pos_ptr, n_steps):
    i = tl.load(pos_ptr)
    for _ in range(n_steps):
        i = tl.load(nxt_ptr + i)  # the next address depends on this load: latency, not bandwidth
    tl.store(pos_ptr, i)  # resume here next call, so repeated calls walk into lines not yet cached


def dependent_load_latency_ns(n_bytes, n_steps=2**16):
    n_el = n_bytes // 8
    perm = torch.randperm(n_el, generator=torch.Generator().manual_seed(0))
    nxt = torch.empty(n_el, dtype=torch.int64)
    nxt[perm] = torch.roll(perm, -1)  # one random cycle through every element
    nxt, pos = nxt.to(dev), torch.zeros(1, dtype=torch.int64, device=dev)
    return bench(lambda: _pointer_chase[(1,)](nxt, pos, n_steps), warmup=2, iters=5) * 1e6 / n_steps


lat = {kb: dependent_load_latency_ns(kb * 1024) for kb in (32, 1024, 4096, 65536, 1048576)}
for kb, ns in lat.items():
    print(f"buffer {kb:>8} KiB: {ns:6.0f} ns per dependent load")
lat_dram = lat[1048576]
check("latency grows down the hierarchy (L1 < L2 < DRAM)", lat[32] < lat[1024] < lat_dram)

# %% [markdown]
# (c) **Bandwidth vs size.** `b.copy_(a)` reads and writes each byte once, so it moves $2n$ bytes for an
# $n$-byte tensor. Then **fit** $\alpha, \beta$.
#
# **Exercise 3.1.** `fit_alpha_beta(nbytes, seconds)`: fit $t = \alpha + n/\beta$ by least squares on the
# **relative** error, $\min \sum_i \big((\alpha + n_i/\beta - t_i)/t_i\big)^2$, so that the µs-scale small sizes
# count as much as the ms-scale large ones. Return `(alpha_seconds, beta_bytes_per_second)`. Hint: the model is
# linear in $(\alpha, 1/\beta)$; divide each row of the design matrix and the target by $t_i$.

# %%
def fit_alpha_beta(nbytes, seconds):
    ### BEGIN SOLUTION
    nbytes, seconds = np.asarray(nbytes, float), np.asarray(seconds, float)
    X = np.stack([np.ones_like(nbytes), nbytes], axis=1) / seconds[:, None]
    (alpha, inv_beta), *_ = np.linalg.lstsq(X, np.ones_like(seconds), rcond=None)
    return float(alpha), float(1 / inv_beta)
    ### END SOLUTION


nb_syn = np.logspace(3, 9, 20)
t_syn = (4e-6 + nb_syn / 8e11) * np.exp(np.random.default_rng(0).normal(0, 0.02, nb_syn.size))  # 2% noise
a_syn, b_syn = fit_alpha_beta(nb_syn, t_syn)
check_close("fit recovers alpha (relative LS)", a_syn, 4e-6, rtol=0.05, atol=0)
check_close("fit recovers beta (relative LS)", b_syn, 8e11, rtol=0.05)

# %%
def bench_graph(fn, reps):
    """GPU time of one fn() (s), with `reps` calls captured in a CUDA graph so CPU launch overhead is excluded."""
    fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    return bench(lambda: g.replay(), warmup=2, iters=5) / 1e3 / reps


sizes = [2**k for k in range(10, 31, 2)]  # 1 KiB ... 1 GiB per tensor
copy_s = []
for nbytes in sizes:
    a = torch.empty(nbytes // 4, device=dev)
    b = torch.empty_like(a)
    copy_s.append(bench_graph(lambda: b.copy_(a), reps=50 if nbytes <= 2**24 else 3))
    del a, b
moved = 2 * np.array(sizes, float)  # read + write
for nbytes, t in zip(sizes, copy_s):
    print(f"tensor {nbytes / 2**20:9.3f} MiB: {t * 1e6:9.1f} us  {2 * nbytes / t / 1e9:7.0f} GB/s")
# Mid sizes whose 2 tensors fit in L2 (a graph replays the same copy, so they stay cached) run faster than DRAM
# allows; the DRAM model t = alpha + n/beta does not describe them. Fit on the tiny and the DRAM-sized points.
use = (moved <= 2**15) | (moved >= 8 * props.L2_cache_size)
alpha, beta = fit_alpha_beta(moved[use], np.array(copy_s)[use])
clk = gpu_clocks()
bw_live = BW_PEAK * clk[1] / 9751  # whitepaper 936 GB/s assumes the 9751 MHz (19.5 Gbps) memory clock
print(f"alpha = {alpha * 1e6:.1f} us   beta = {beta / 1e9:.0f} GB/s = {beta / BW_PEAK:.0%} of 936 GB/s "
      f"(live memory clock {clk[1]} MHz -> peak {bw_live / 1e9:.0f} GB/s)")
print(f"n_1/2 = alpha*beta = {alpha * beta / 2**20:.1f} MiB moved")
print(f"Little's law: bytes in flight = beta * DRAM latency = {beta * lat_dram * 1e-9 / 1024:.0f} KiB "
      f"= {beta * lat_dram * 1e-9 / props.multi_processor_count / 1024:.1f} KiB per SM")
BW = beta  # measured achievable bandwidth: use it for the roofline below

plt.figure(figsize=(6, 3.5))
plt.semilogx(moved, moved / np.array(copy_s) / 1e9, "o-", label="copy_ (measured)")
plt.semilogx(moved, moved / (alpha + moved / beta) / 1e9, "--", label=r"fit $\alpha + n/\beta$")
plt.axhline(BW_PEAK / 1e9, color="k", lw=0.8, label="936 GB/s (whitepaper)")
plt.axvline(2 * props.L2_cache_size, color="gray", ls=":", label="2 x L2 size")
plt.xlabel("bytes moved per call (read + write)"); plt.ylabel("achieved bandwidth (GB/s)")
plt.title("Copy bandwidth vs size"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()

check("large copies reach > 70% of 936 GB/s", beta > 0.7 * BW_PEAK, f"{beta / 1e9:.0f} GB/s")
check("small copies are overhead-bound (< 10% of peak at 4 KiB)", 2 * 4096 / copy_s[1] < 0.1 * BW_PEAK)
print(f"eager PyTorch: effective alpha ~ {us_eager:.1f} us -> n_1/2 ~ {us_eager * 1e-6 * beta / 2**20:.1f} MiB per op")

# %% [markdown]
# Interpretation. On the GPU alone a kernel costs about a microsecond plus bytes / bandwidth, so anything moving
# less than ~1 MB is latency-bound. In eager PyTorch each op also costs several µs of CPU time (Python,
# dispatcher, launch), and if the GPU finishes faster than the CPU can issue, the GPU idles: the program is
# **overhead-bound**, and a bigger GPU will not help. Symptoms: time does not change when you shrink the tensors;
# the profiler shows gaps between kernels (G1.6). Remedies: bigger batches, fusion (`torch.compile`), CUDA graphs
# (`mode="reduce-overhead"`). The mid sizes above 100% of "peak" are L2 hits: the same 1-4 MB copy replayed in a
# graph never leaves the 6 MB L2.

# %% [markdown]
# ## 4. Arithmetic intensity by hand
#
# **Arithmetic intensity** $I = \text{FLOPs} / \text{bytes moved to/from DRAM}$. A kernel takes at least
# $\max(\text{FLOPs}/P,\ \text{bytes}/\beta)$, so it is **memory-bound** when $I < P/\beta$ and **compute-bound**
# when $I > P/\beta$. The threshold $I^* = P/\beta$ is the **ridge point** (NVIDIA calls it the ops:byte ratio).
# On the 3090: $71\text{e}12 / 936\text{e}9 \approx 76$ FLOP/byte for fp16/bf16 tensor-core math, $38$ for FP32.
#
# Count FLOPs as multiply = 1, add = 1 (so a multiply-accumulate is 2). Count the *minimum* DRAM traffic: every
# input read once, every output written once, nothing else (caches are assumed perfect inside one kernel and
# empty between kernels).
#
# **Exercise 4.1.** Implement the three functions (formulas in the docstrings). Write the number for a
# $4096^3$ fp16 matmul on paper first.

# %%
def ai_elementwise(flops_per_elem, n_inputs, n_outputs, bytes_per_elem):
    """N-element pointwise op: each of n_inputs tensors read once, each of n_outputs written once."""
    ### BEGIN SOLUTION
    return flops_per_elem / ((n_inputs + n_outputs) * bytes_per_elem)
    ### END SOLUTION


def ai_matmul(M, N, K, bytes_per_elem):
    """C[M,N] = A[M,K] @ B[K,N]: 2MNK FLOPs; read A and B once, write C once."""
    ### BEGIN SOLUTION
    return 2 * M * N * K / (bytes_per_elem * (M * K + K * N + M * N))
    ### END SOLUTION


def ai_attention(T, d, bytes_per_elem, fused):
    """One head, non-causal. FLOPs: only the two matmuls QK^T and PV (4 T^2 d).
    Unfused: read Q,K; write S; read S, write P; read P,V; write O  -> (4Td + 4T^2) elements.
    Fused (FlashAttention): read Q,K,V, write O                      -> 4Td elements."""
    ### BEGIN SOLUTION
    flops = 4 * T * T * d
    elems = 4 * T * d if fused else 4 * T * d + 4 * T * T
    return flops / (elems * bytes_per_elem)
    ### END SOLUTION


check_close("x*2 + 1 in fp32 (2 FLOPs, 1 in, 1 out)", ai_elementwise(2, 1, 1, 4), 0.25)
check_close("square fp16 matmul n=4096", ai_matmul(4096, 4096, 4096, 2), 4096 / 3)
check_close("GEMV 1x4096 @ 4096x4096 fp16", ai_matmul(1, 4096, 4096, 2), 2 * 4096**2 / (2 * (2 * 4096 + 4096**2)))
check_close("naive attention T=4096 d=64 fp16", ai_attention(4096, 64, 2, False), 4096 * 64 / (2 * (64 + 4096)))
check_close("fused attention T=4096 fp16", ai_attention(4096, 64, 2, True), 4096 / 2)
RIDGE = {k: v / BW_PEAK for k, v in PEAK.items()}
print({k: round(v, 1) for k, v in RIDGE.items()}, "FLOP/byte at the cited peaks")

# %% [markdown]
# Read the numbers: a square matmul has $I \approx n/3$ (fp16), so anything bigger than $n \approx 230$ is
# compute-bound *in principle*. A GEMV ($M=1$) has $I \approx 1$: hopelessly memory-bound. Naive attention at
# $T=4096,d=64$ has $I \approx 31 < 76$: memory-bound, because the $T\times T$ matrices dominate traffic.
# Fusing it (never writing $S, P$) makes $I = T/2$: compute-bound. That is FlashAttention in one line (G1.4).

# %% [markdown]
# ## 5. Matmul throughput vs size and dtype
#
# **Predict:** at $n=8192$, what fraction of the cited peak will FP32, TF32, FP16 and BF16 reach? At $n=512$?
#
# `torch.backends.cuda.matmul.fp32_precision = "tf32"` lets cuBLAS run FP32 matmuls on tensor cores with inputs
# rounded to TF32 (10-bit mantissa); `"ieee"` keeps true FP32. Use only this new-style flag in torch 2.14: mixing
# it with the legacy `allow_tf32` raises an error.

# %%
mm_sizes = [256, 512, 1024, 2048, 4096, 8192]
mm_tflops = {}
for name in ["fp32", "tf32", "fp16", "bf16"]:
    dtype = {"fp32": torch.float32, "tf32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]
    torch.backends.cuda.matmul.fp32_precision = "tf32" if name == "tf32" else "ieee"
    mm_tflops[name] = []
    for n in mm_sizes:
        A, B = torch.randn(n, n, device=dev, dtype=dtype), torch.randn(n, n, device=dev, dtype=dtype)
        t = bench(lambda: A @ B, warmup=3, iters=10 if n >= 4096 else 30) / 1e3
        mm_tflops[name].append(2 * n**3 / t / 1e12)
    clk_sm = gpu_clocks()[0]  # read right after the heaviest matmuls, while the clock is still up
    print(f"{name}: " + "  ".join(f"{n}:{v:5.1f}" for n, v in zip(mm_sizes, mm_tflops[name]))
          + f"  TFLOP/s   (SM clock ~{clk_sm} MHz -> peak x{clk_sm / BOOST_MHZ:.2f})")
torch.backends.cuda.matmul.fp32_precision = "ieee"

fig, ax = plt.subplots(figsize=(6.5, 3.8))
for name, c in zip(mm_tflops, ["C0", "C1", "C2", "C3"]):
    ax.semilogx(mm_sizes, mm_tflops[name], "o-", color=c, label=name)
    ax.axhline(PEAK[name] / 1e12, color=c, ls=":", lw=1)
ax.set_xscale("log", base=2); ax.set_xlabel("n (square matmul n x n x n)"); ax.set_ylabel("TFLOP/s")
ax.set_title("Measured matmul throughput (dotted: whitepaper peak at 1695 MHz)"); ax.legend(); plt.tight_layout(); plt.show()

check("fp16 reaches > 70% of the 71 TFLOPS FP32-accumulate peak", max(mm_tflops["fp16"]) > 0.7 * 71)
check("fp16 stays well below the 142 TFLOPS FP16-accumulate peak (PyTorch accumulates in fp32)",
      max(mm_tflops["fp16"]) < 0.65 * 142)
check("tf32 beats fp32 at n=8192", mm_tflops["tf32"][-1] > 1.2 * mm_tflops["fp32"][-1])

# %% [markdown]
# What you should see, and why:
# - **Small $n$ is far below peak** even though $I = n/3$ is above the ridge. A $512^3$ matmul is 0.27 GFLOP:
#   4 µs at peak, about the fixed cost $\alpha$ measured above. It also produces only $16$ output tiles of
#   $128\times128$ for 82 SMs, so most SMs idle (**wave quantization**: work is issued in waves of one tile per
#   SM; NVIDIA Matrix Multiplication Background guide).
# - **FP16 ≈ BF16 ≈ 71 TFLOPS**: same tensor-core rate with FP32 accumulation. G1.5 shows a Triton matmul with
#   FP16 accumulation running close to the 142 TFLOPS row of the table.
# - **TF32 ≈ 2× FP32 at large $n$**; numerics differ (G1.2).
# - **Measured can exceed "peak"** by a few percent because the whitepaper assumes 1695 MHz while the SM clock
#   under load is often higher (printed above). Always ask "peak at which clock?".

# %% [markdown]
# ## 6. The roofline of this GPU
#
# The roofline model (Williams et al. 2009) plots attainable FLOP/s against arithmetic intensity:
# $$P_{\text{attainable}}(I) = \min(P_{\text{peak}},\ I\cdot\beta).$$
# The slanted part is the memory roof, the flat part the compute roof, and they meet at the ridge $I^*$.
# A kernel plotted far below its roof is limited by something else (overhead, occupancy, bad access pattern).
#
# **Exercise 6.1.** `roofline_time(flops, nbytes, peak, bw)`: the roofline's lower bound on run time (seconds).

# %%
def roofline_time(flops, nbytes, peak, bw):
    ### BEGIN SOLUTION
    return max(flops / peak, nbytes / bw)
    ### END SOLUTION


check_close("compute-bound case", roofline_time(2e12, 1e10, 1e14, 1e12), 2e-2, atol=0, rtol=1e-9)
check_close("memory-bound case", roofline_time(1e11, 1e10, 1e14, 1e12), 1e-2, atol=0, rtol=1e-9)

# %%
# Measure a zoo of ops: (name, fn, flops, bytes moved, dtype family for the roof)
def op_zoo():
    ops = []
    N = 2**26
    x32 = torch.randn(N, device=dev)
    ops.append(("x*2+1 fp32", lambda: x32 * 2 + 1, 2 * N, 8 * N, "fp32"))
    ops.append(("gelu fp32", lambda: F.gelu(x32), 8 * N, 8 * N, "fp32"))  # ~8 FLOPs/elem (erf polynomial etc.): rough
    xs = torch.randn(4096, 8192, device=dev, dtype=torch.float16)
    ops.append(("softmax fp16", lambda: xs.softmax(-1), 5 * xs.numel(), 4 * xs.numel(), "fp16"))
    for n in (512, 1024, 2048, 4096, 8192):
        A, B = torch.randn(n, n, device=dev, dtype=torch.float16), torch.randn(n, n, device=dev, dtype=torch.float16)
        ops.append((f"mm {n}", lambda A=A, B=B: A @ B, 2 * n**3, 2 * 3 * n * n, "fp16"))
    W = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    for m in (1, 8, 64, 512):
        X = torch.randn(m, 8192, device=dev, dtype=torch.float16)
        ops.append((f"gemv-ish M={m}", lambda X=X: X @ W, 2 * m * 8192**2, 2 * (8192**2 + 2 * m * 8192), "fp16"))
    return ops


zoo = []
for name, fn, flops, nbytes, fam in op_zoo():
    t = bench(fn, warmup=3, iters=20) / 1e3
    zoo.append((name, flops / nbytes, flops / t, t, roofline_time(flops, nbytes, PEAK[fam], BW), fam))
    print(f"{name:16s} I={flops / nbytes:8.2f} FLOP/B  achieved {flops / t / 1e12:6.2f} TFLOP/s  "
          f"time {t * 1e6:8.1f} us  roofline bound {zoo[-1][4] * 1e6:8.1f} us")

I = np.logspace(-1.5, 4, 200)
plt.figure(figsize=(7, 4.5))
for fam, c in (("fp16", "C2"), ("fp32", "C0")):
    plt.loglog(I, np.minimum(PEAK[fam], I * BW) / 1e12, color=c, label=f"roof {fam} (peak {PEAK[fam] / 1e12:.1f}, BW {BW / 1e9:.0f} GB/s)")
for name, ai, perf, t, tb, fam in zoo:
    plt.loglog(ai, perf / 1e12, "o", color="C2" if fam == "fp16" else "C0")
    plt.annotate(name, (ai, perf / 1e12), fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.xlabel("arithmetic intensity (FLOP / byte of DRAM traffic)"); plt.ylabel("achieved TFLOP/s")
plt.title("Roofline of this RTX 3090 (measured bandwidth, cited compute peaks)")
plt.legend(fontsize=8, loc="lower right"); plt.grid(True, which="both", alpha=0.2); plt.tight_layout(); plt.show()

check("no measured op beats its roof by more than 15% (clock boost headroom)",
      all(perf <= 1.15 * min(PEAK[fam], ai * BW) for _, ai, perf, _, _, fam in zoo))

# %% [markdown]
# ## 7. Predict, then measure: decode is a GEMV, and GEMVs are memory-bound
#
# In autoregressive decoding each step multiplies a batch of $B$ token vectors by every weight matrix:
# $X\in\mathbb{R}^{B\times d}$ times $W\in\mathbb{R}^{d\times d}$. FLOPs $= 2Bd^2$; bytes $\approx b(d^2 + 2Bd)$
# for $b$ bytes per element. For $B \ll d$ the weight read dominates and $I \approx B$ FLOP/byte (bf16). So the
# step time is *flat in $B$* (you pay to stream $W$ once, whatever the batch) until $B$ reaches the ridge, then
# grows linearly. That is why serving systems batch requests (G1.20) and why decode cost is "bytes of weights
# + KV cache / bandwidth".
#
# Numbers for a 7B-parameter model in bf16 at batch 1: $14$ GB of weights per token. At 936 GB/s that is
# $\geq 15$ ms per token ($\leq 67$ tokens/s), while the $2\times 7\text{e}9 = 14$ GFLOP of math would take
# $0.2$ ms at 71 TFLOPS. Compute sits idle >98% of the time.
#
# **Exercise 7.1.** `crossover_batch(d, peak, bw, b)`: the batch $B^*$ at which $2Bd^2/P = b(d^2+2Bd)/\beta$
# (solve exactly for $B$). Then predict the knee of the measured curve.

# %%
def crossover_batch(d, peak, bw, b):
    ### BEGIN SOLUTION
    # 2 B d^2 / P = b (d^2 + 2 B d) / bw  ->  B (2 d^2 / P - 2 b d / bw) = b d^2 / bw
    return (b * d * d / bw) / (2 * d * d / peak - 2 * b * d / bw)
    ### END SOLUTION


d = 8192
B_star = crossover_batch(d, PEAK["bf16"], BW, 2)
grid_B = np.arange(1, 4096)
brute = grid_B[np.argmin(np.abs(2 * grid_B * d**2 / PEAK["bf16"] - 2 * (d * d + 2 * grid_B * d) / BW))]
check("crossover matches brute force", abs(B_star - brute) <= 1, f"{B_star:.1f} vs {brute}")
print(f"predicted crossover B* = {B_star:.0f}  (cf. ridge P/beta = {PEAK['bf16'] / BW:.0f} FLOP/byte with the measured beta)")

W = torch.randn(d, d, device=dev, dtype=torch.bfloat16)
Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
t_dec = []
for Bsz in Bs:
    X = torch.randn(Bsz, d, device=dev, dtype=torch.bfloat16)
    t_dec.append(bench(lambda: X @ W, warmup=3, iters=20) / 1e3)
t_pred = [roofline_time(2 * Bsz * d * d, 2 * (d * d + 2 * Bsz * d), PEAK["bf16"], BW) for Bsz in Bs]

fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
axs[0].loglog(Bs, np.array(t_dec) * 1e6, "o-", label="measured"); axs[0].loglog(Bs, np.array(t_pred) * 1e6, "--", label="roofline bound")
axs[0].axvline(B_star, color="gray", ls=":", label=f"B* = {B_star:.0f}")
axs[0].set_xlabel("batch B (tokens)"); axs[0].set_ylabel("time per X@W (us)"); axs[0].legend(); axs[0].set_title("8192x8192 bf16 weight")
axs[1].loglog(Bs, np.array(Bs) / np.array(t_dec), "o-"); axs[1].axvline(B_star, color="gray", ls=":")
axs[1].set_xlabel("batch B (tokens)"); axs[1].set_ylabel("tokens / s through this layer"); axs[1].set_title("throughput grows ~linearly until B*")
plt.tight_layout(); plt.show()

check("memory-bound regime: B=32 costs < 1.5x B=1", t_dec[Bs.index(32)] < 1.5 * t_dec[0])
check("compute-bound regime: B=2048 costs > 4x B=1", t_dec[-1] > 4 * t_dec[0])

# %% [markdown]
# ## 8. Why naive attention is memory-bound (measured)
#
# Naive attention runs three kernels: $S = QK^\top/\sqrt d$ (matmul, writes $T^2$ per head), $P=\mathrm{softmax}(S)$
# (reads and writes $T^2$, ~5 FLOPs per element), $O = PV$ (reads $T^2$). The softmax has about $5/(4d)$ of the
# matmul FLOPs, but it moves as many bytes as they do. **Predict** the softmax's share of the time, then run.

# %%
Bb, H, T, dh = 4, 16, 2048, 64
q, k, v = (torch.randn(Bb, H, T, dh, device=dev, dtype=torch.float16) for _ in range(3))
S_ = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
P_ = S_.softmax(-1)
t_qk = bench(lambda: (q @ k.transpose(-1, -2)) / math.sqrt(dh))
t_sm = bench(lambda: S_.softmax(-1))
t_pv = bench(lambda: P_ @ v)
t_fused = bench(lambda: F.scaled_dot_product_attention(q, k, v))
mm_flops = 2 * Bb * H * T * T * dh
print(f"QK^T+scale {t_qk:.3f} ms   softmax {t_sm:.3f} ms   PV {t_pv:.3f} ms   total {t_qk + t_sm + t_pv:.3f} ms")
print(f"softmax: {5 * Bb * H * T * T / (2 * mm_flops + 5 * Bb * H * T * T):.1%} of FLOPs, "
      f"{t_sm / (t_qk + t_sm + t_pv):.0%} of time")
print(f"fused SDPA (FlashAttention kernel): {t_fused:.3f} ms -> {(t_qk + t_sm + t_pv) / t_fused:.1f}x faster, "
      f"{2 * mm_flops / t_fused / 1e9:.1f} TFLOP/s")
print(f"T x T matrices written+read by the naive path: {Bb * H * T * T * 2 * 4 / 2**20:.0f} MiB of traffic")
check("softmax is >= 15% of naive attention time while < 4% of FLOPs", t_sm / (t_qk + t_sm + t_pv) > 0.15)
check("the fused kernel is faster than the three-kernel version", t_fused < t_qk + t_sm + t_pv)
del S_, P_

# %% [markdown]
# ## Say it out loud
# 1. Define arithmetic intensity and the ridge point. What are the ridge points of an RTX 3090 for bf16 tensor-core
#    math and for FP32? Classify: elementwise add, LayerNorm, a $4096^3$ matmul, a GEMV, naive attention, FlashAttention.
# 2. Why is `t0 = time.time(); y = x @ w; print(time.time() - t0)` wrong on a GPU? List three more benchmarking traps.
# 3. Your $512\times512\times512$ fp16 matmul reaches 15% of peak even though $I \approx 170 > 76$. Why?
# 4. Why is LLM decoding memory-bound at small batch? Give the 7B-model numbers on this GPU. Name four ways to make
#    decoding faster and say which term in $t \approx \text{bytes}/\beta$ each one attacks.
# 5. Why is naive attention memory-bound and how does FlashAttention change its arithmetic intensity?
# 6. A colleague says "the 3090 does 142 TFLOPS of fp16". When is that true, and what number should they use for
#    PyTorch training? What changes on an A100?
# 7. What is latency hiding? Use Little's law and the numbers you measured to say how many bytes must be in flight.
#    Why can raising registers per thread make a kernel slower?
# 8. Your elementwise kernel reaches 300 GB/s on a 3090. List what you would check.

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. $I$ = FLOPs / bytes of DRAM traffic. Ridge $I^* = P/\beta$: bf16 tensor (FP32 accumulate) $71\text{e}12/936\text{e}9 \approx 76$
#    FLOP/B; FP32 $35.6\text{e}12/936\text{e}9 \approx 38$ (GA102 whitepaper Table 9). Elementwise add: $1/12$ (fp32), memory-bound.
#    LayerNorm: a few FLOPs per element over 8 bytes of read+write, memory-bound. $4096^3$ fp16 matmul: $I = n/3 \approx 1365$,
#    compute-bound. GEMV: $I \approx 1$, memory-bound. Naive attention $T=4096, d=64$ fp16: $\approx 31$, memory-bound.
#    FlashAttention: $I \approx T/2$, compute-bound.
# 2. Kernels launch asynchronously, so without `torch.cuda.synchronize()` (or CUDA events) you time the enqueue. Other traps:
#    no warmup (context creation, cuBLAS heuristics, autotuning, compile, allocator growth); timing a single noisy run
#    instead of a median; inputs resident in L2 in a loop (flush it, as `do_bench` does); clocks that ramp up and
#    throttle (thermals, power); other processes on the GPU; including host-device copies or `.item()` syncs by accident;
#    timing Python-side overhead and calling it kernel time (or the reverse).
# 3. It is 0.27 GFLOP, about 4 µs at peak, the same order as the fixed per-kernel cost $\alpha$. It also has only
#    $16$ output tiles of $128\times128$ for 82 SMs, so most SMs idle (wave quantization), and each tile's short K-loop
#    cannot amortize pipeline fill. Peak needs enough independent tiles for several waves.
# 4. Each decode step reads every weight (and the KV cache) once to do 2 FLOPs per weight per sequence, so $I \approx B$
#    (bf16) ≪ 76. 7B in bf16 is 14 GB per step: ≥ 15 ms per step on 936 GB/s, so ≤ 67 tokens/s at batch 1 while compute
#    needs 0.2 ms. Fixes: batching (more FLOPs per byte; continuous batching, G1.20), weight quantization int8/int4 (fewer
#    bytes; GPTQ/AWQ), GQA/MQA/MLA (fewer KV bytes), speculative decoding (verify several tokens per weight read,
#    A3.18), and CUDA graphs / fused kernels (remove the overhead term when the model is small).
# 5. Unfused attention writes and re-reads the $T\times T$ score and probability matrices, $O(T^2)$ bytes for
#    $O(T^2 d)$ FLOPs, so $I \approx d/2$ fp16 elements' worth (~31 at $d=64$): memory-bound. The softmax alone is a few
#    percent of the FLOPs but a large share of the time (measured above). FlashAttention keeps $S,P$ tiles in SRAM and
#    uses online softmax, so DRAM traffic is $O(Td)$ per head (plus repeated reads of K/V tiles) and $I$ grows with
#    $T$: compute-bound (Dao et al. 2022).
# 6. 142 TFLOPS is FP16 tensor math *with FP16 accumulation* (GA102 whitepaper Table 9). GeForce Ampere runs FP32
#    accumulation at half rate: 71 TFLOPS for fp16 and bf16, which is what cuBLAS uses for PyTorch matmuls and what you
#    should use for MFU. The A100's 312 TFLOPS dense fp16/bf16 holds with FP32 accumulation.
# 7. Each SM keeps many warps resident and issues from whichever is ready, so one warp's DRAM wait overlaps others'
#    work. Little's law: bytes in flight = bandwidth × latency. With ~840 GB/s and a few hundred ns DRAM latency
#    (measured above) that is a few hundred KB across the GPU, a few KB per SM. More registers per thread means fewer
#    resident warps (the register file is 256 KB per SM), so less latency hiding, unless instruction-level parallelism
#    inside each thread makes up for it.
# 8. Make sure you count bytes correctly (read + write). Check that the access is coalesced: contiguous, aligned,
#    vectorized loads, no strided or transposed reads. Check that the grid is large enough to fill 82 SMs with several
#    waves, and that occupancy is reasonable. Check that the tensor is large enough to amortize $\alpha$ (here
#    $n_{1/2} \approx$ a few MB). Check that you are not timing launch overhead or a sync. Check dtype conversions and
#    extra passes the kernel makes, and the live memory clock.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
