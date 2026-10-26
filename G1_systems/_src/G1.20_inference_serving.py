# %% [markdown]
# # G1.20 · Inference serving: decode is memory-bound, batching, KV-cache memory, prefix caching
#
# **ID:** G1.20 · **Target:** 120 min · **Plan week:** 15 (with F4 "cost accounting" in week 14) · **Part:** G1
#
# **The lab.** Guide G1: *"Inference serving: vLLM/SGLang concepts, batching, prefix caching, throughput vs latency."*
# And F4: *"prefill vs decode cost; caching (KV/prefix caching) effects on agent loops."* You will:
# 1. compute the arithmetic intensity of **prefill** and **decode** (7B model and our GPU) and size the **KV cache** for
#    MHA, GQA and MLA;
# 2. **measure** decode-step latency vs batch size on this GPU (`prep.tinygpt.GPT` with a KV cache), and see the
#    memory-bound regime: throughput scales almost linearly with batch until compute or KV reads take over;
# 3. build a small **discrete-event simulator** of a serving engine and compare **static** vs **continuous** batching
#    (Orca), then **chunked prefill** (Sarathi) and a sketch of **prefill/decode disaggregation** (DistServe);
# 4. implement **PagedAttention's** block-table allocator (with copy-on-write fork) and measure fragmentation vs
#    contiguous preallocation (vLLM);
# 5. implement **prefix caching** with a trie over token blocks (SGLang RadixAttention, vLLM automatic prefix caching)
#    and measure the prefill saved in a multi-turn agent trace.
#
# Related drills elsewhere: KV-cache implementation from scratch is **A3.13**, speculative decoding is **A3.18**.
# Fill in every `raise NotImplementedError`. Solution runtime: about 1 min.
#
# ## Sources
# - W. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023,
#   arXiv:2309.06180 (block tables, copy-on-write; existing systems use only 20.4-38.2% of KV memory for tokens, §3) — https://arxiv.org/abs/2309.06180
# - G.-I. Yu, J. S. Jeong, G.-W. Kim, S. Kim, B.-G. Chun, *Orca: A Distributed Serving System for Transformer-Based
#   Generative Models*, OSDI 2022 (iteration-level scheduling, selective batching) — https://www.usenix.org/conference/osdi22/presentation/yu
# - L. Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, NeurIPS 2024, arXiv:2312.07104
#   (RadixAttention: LRU cache of KV in a radix tree) — https://arxiv.org/abs/2312.07104
# - A. Agrawal et al., *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*, 2023,
#   arXiv:2308.16369 — https://arxiv.org/abs/2308.16369 ; A. Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM
#   Inference with Sarathi-Serve*, OSDI 2024, arXiv:2403.02310 (chunked prefills, stall-free schedules) — https://arxiv.org/abs/2403.02310
# - Y. Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving*, OSDI 2024,
#   arXiv:2401.09670 (TTFT/TPOT, goodput, prefill-decode interference) — https://arxiv.org/abs/2401.09670
# - N. Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (MQA), 2019, arXiv:1911.02150 — https://arxiv.org/abs/1911.02150
# - J. Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023,
#   arXiv:2305.13245 — https://arxiv.org/abs/2305.13245
# - DeepSeek-AI, *DeepSeek-V2*, 2024, arXiv:2405.04434 (MLA: KV cache $(d_c + d^R_h)l$ elements per token, $d_c=512$,
#   $d^R_h=64$, 60 layers; Table 1) — https://arxiv.org/abs/2405.04434
# - H. Touvron et al., *LLaMA*, 2023, arXiv:2302.13971 (Table 2: 6.7B = 32 layers, $d$=4096, 32 heads) — https://arxiv.org/abs/2302.13971 ;
#   Llama Team, *The Llama 3 Herd of Models*, 2024, arXiv:2407.21783 (Table 3: 8B/70B with 8 KV heads) — https://arxiv.org/abs/2407.21783
# - kipply, *Transformer Inference Arithmetic*, 2022 (KV bytes per token $2\cdot 2\cdot n_\text{layers} n_\text{heads} d_\text{head}$;
#   decode memory-bound below batch ≈ FLOPs/bandwidth) — https://kipp.ly/transformer-inference-arithmetic/
# - J. Austin et al., *How to Scale Your Model*, 2025, "All About Transformer Inference" (KV size $2\cdot\text{bytes}\cdot HKLT$,
#   critical batch size) and "How to Think About GPUs" (A100: 2.0e12 B/s, 3.1e14 bf16 FLOP/s) —
#   https://jax-ml.github.io/scaling-book/inference/ , https://jax-ml.github.io/scaling-book/gpus/
# - R. Pope et al., *Efficiently Scaling Transformer Inference*, MLSys 2023, arXiv:2211.05102 — https://arxiv.org/abs/2211.05102
# - E. Frantar et al., *GPTQ*, ICLR 2023, arXiv:2210.17323 — https://arxiv.org/abs/2210.17323 ; J. Lin et al., *AWQ*,
#   MLSys 2024, arXiv:2306.00978 — https://arxiv.org/abs/2306.00978 (weight-only quantization)
# - T. Dao et al., *Flash-Decoding for long-context inference*, Stanford CRFM blog, 2023 — https://crfm.stanford.edu/2023/10/12/flashdecoding.html
# - NVIDIA GA102 whitepaper v2, Table 9 (RTX 3090: 71 TFLOPS bf16, 936 GB/s, 24 GB) — https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf

# %%
import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton.testing import do_bench

from prep import Session, check, check_close, seed_everything
from prep.tinygpt import GPT, GPTConfig, num_params

s = Session("G1.20", target_min=120)
seed_everything(0)
dev = torch.device("cuda")
P_3090, BW_3090 = 71e12, 936e9  # GA102 whitepaper (bf16 tensor, FP32 accumulate; GDDR6X)

# %% [markdown]
# ## 1. Prefill vs decode, by arithmetic intensity
#
# A request has two phases. **Prefill** runs the whole prompt of $T_p$ tokens through the model at once (one big batch
# of matmuls) and fills the KV cache; it determines the **time to first token (TTFT)**. **Decode** generates one token
# per step per sequence, reading every weight and the sequence's whole KV cache; it determines the **time per output
# token (TPOT)**.
#
# For a dense model with $N$ matmul weights in $b_w$ bytes, a step that processes $n$ tokens does $\approx 2Nn$ FLOPs
# and reads $Nb_w$ bytes of weights, so $I \approx 2n/b_w$: **prefill** ($n = T_p$, hundreds to thousands) is
# compute-bound; **decode** at batch $B$ ($n = B$) has $I \approx B$ in bf16, memory-bound until $B \approx P/\beta$
# (76 on a 3090, ~150 on an A100, ~290 on an H100). On top of that, each decoding sequence reads its own KV cache, which
# **no amount of batching amortizes**: the attention part of decode stays memory-bound at any batch size.
#
# **KV cache bytes per token** $= 2\,(\text{K and V}) \times L \times n_{kv} \times d_\text{head} \times b$ (kipply 2022;
# scaling book). MHA has $n_{kv} = n_\text{heads}$; GQA shares each KV head among a group of query heads (Ainslie et al.
# 2023); MQA has $n_{kv}=1$ (Shazeer 2019); MLA caches one compressed latent of $d_c$ plus a decoupled RoPE key of
# $d^R_h$ per layer: $(d_c + d^R_h)\,L$ elements (DeepSeek-V2, Table 1).
#
# **Exercise 1.1.**

# %%
def kv_bytes_per_token(n_layers, n_kv_heads, d_head, bytes_per_el=2):
    ### BEGIN SOLUTION
    return 2 * n_layers * n_kv_heads * d_head * bytes_per_el
    ### END SOLUTION


def mla_kv_bytes_per_token(n_layers, d_c, d_rope, bytes_per_el=2):
    ### BEGIN SOLUTION
    return n_layers * (d_c + d_rope) * bytes_per_el
    ### END SOLUTION


def decode_step_time(n_params, batch, context, kv_per_token, peak, bw, bytes_w=2):
    """Roofline lower bound (s) for one decode step of `batch` sequences that each have `context` cached tokens.
    FLOPs: 2 * n_params * batch (ignore attention FLOPs). Bytes: weights once + every sequence's KV cache."""
    ### BEGIN SOLUTION
    flops = 2 * n_params * batch
    nbytes = n_params * bytes_w + batch * context * kv_per_token
    return max(flops / peak, nbytes / bw)
    ### END SOLUTION


kv7 = kv_bytes_per_token(32, 32, 128)  # LLaMA 6.7B / Llama 2 7B shape: MHA
kv8 = kv_bytes_per_token(32, 8, 128)  # Llama 3 8B: GQA with 8 KV heads
kv70 = kv_bytes_per_token(80, 8, 128)  # Llama 3 70B: GQA
kvds = mla_kv_bytes_per_token(60, 512, 64)  # DeepSeek-V2 MLA
kvds_mha = kv_bytes_per_token(60, 128, 128)  # what MHA with 128 heads of 128 would need
for name, v in (("7B MHA", kv7), ("8B GQA-8", kv8), ("70B GQA-8", kv70), ("DeepSeek-V2 MLA", kvds), ("same, if MHA", kvds_mha)):
    print(f"{name:16s} {v / 1024:7.1f} KiB/token   {v * 4096 / 2**30:6.2f} GiB per 4k-token sequence")
check("7B MHA = 512 KiB per token", kv7 == 512 * 1024)
check("MLA = 576 elements per layer", kvds == 60 * 576 * 2)
check_close("MLA ~ GQA with 2.25 groups (DeepSeek-V2 Table 1)", kvds / kv_bytes_per_token(60, 1, 128), 2.25)

N7 = 6.7e9
for B in (1, 16, 64, 256):
    t = decode_step_time(N7, B, 2048, kv7, P_3090, BW_3090)
    print(f"7B decode on a 3090, context 2048, batch {B:3d}: >= {t * 1e3:6.1f} ms/step, <= {B / t:7.0f} tokens/s, "
          f"KV = {B * 2048 * kv7 / 2**30:5.1f} GiB")
t_pre = 2 * N7 * 2048 / P_3090
print(f"7B prefill of 2048 tokens: >= {t_pre * 1e3:.0f} ms (compute-bound, I ~ {2 * 2048 / 2:.0f} FLOP/byte)")
check_close("batch-1 decode is weight-bandwidth-bound", decode_step_time(N7, 1, 0, kv7, P_3090, BW_3090), N7 * 2 / BW_3090)

# %% [markdown]
# Notice the last column: at batch 16 and 2k context the 7B model's KV cache (16 GiB) no longer fits next to its 13.4 GB
# of weights on a 24 GB card. KV memory, not FLOPs, caps the batch, which is why GQA/MQA/MLA, KV quantization and
# paged allocation (section 4) matter so much for serving throughput.

# %% [markdown]
# ## 2. Measure decode latency vs batch on this GPU
#
# We use `prep.tinygpt.GPT` widened to $d=2048$, 8 layers, 16 heads (508M parameters, bf16) so that a step moves enough
# bytes to be measurable. The KV-cache decode step below is *provided* (writing it is drill A3.13). Two engineering
# details that the measurement taught us:
# - **CUDA graphs.** Eager decode launches ~100 small kernels per step; at batch 1 the CPU launch overhead is about as
#   large as the GPU work (we time both). Serving engines capture decode steps as CUDA graphs for this reason.
# - **Attention kernel choice.** With one query per sequence, the FlashAttention kernel (which parallelizes over query
#   tiles, G1.4) reaches only ~60% of bandwidth here, while two batched GEMVs reach ~100% (measured below). Decode needs
#   split-KV kernels (Flash-Decoding, vLLM's paged-attention kernel).

# %%
def prefill(model, idx):
    """Full forward of the prompt; returns logits and per-layer K, V caches of shape (B, H, T, d_head)."""
    B, T = idx.shape
    H, D = model.cfg.n_heads, model.cfg.d_model
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(T, device=idx.device))
    ks, vs = [], []
    for blk in model.blocks:
        q, k, v = blk.attn.qkv(blk.ln1(x)).split(D, -1)
        q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in (q, k, v))
        ks.append(k); vs.append(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, D))
        x = x + blk.mlp(blk.ln2(x))
    return model.head(model.ln_f(x)), ks, vs


def decode_step(model, tok, pos, kc, vc):
    """One token per sequence. tok: (B, 1); pos: (1,) position tensor; kc/vc: per-layer caches (B, H, Tc, d_head)
    whose last slot is overwritten with this token's K, V (static shapes, so the step is CUDA-graph capturable)."""
    B = tok.shape[0]
    H, D = model.cfg.n_heads, model.cfg.d_model
    x = model.tok_emb(tok) + model.pos_emb(pos)[None]
    for i, blk in enumerate(model.blocks):
        q, k, v = blk.attn.qkv(blk.ln1(x)).split(D, -1)
        q, k, v = (t.view(B, 1, H, D // H).transpose(1, 2) for t in (q, k, v))
        kc[i][:, :, -1:] = k
        vc[i][:, :, -1:] = v
        y = ((q @ kc[i].transpose(-1, -2)) / math.sqrt(D // H)).softmax(-1) @ vc[i]  # two batched GEMVs
        x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, 1, D))
        x = x + blk.mlp(blk.ln2(x))
    return model.head(model.ln_f(x))


torch.manual_seed(0)
wcfg = GPTConfig(d_model=2048, n_heads=16, n_layers=8, context_length=1024)
wide = GPT(wcfg).to(dev).to(torch.bfloat16).eval()
Nw = num_params(wide)
ctx = 128
with torch.no_grad():
    idx = torch.randint(0, wcfg.vocab_size, (2, ctx), device=dev)
    full = wide(idx)[:, -1].float()
    _, ks, vs = prefill(wide, idx[:, :-1])
    kc = [torch.cat([k_, torch.zeros_like(k_[:, :, :1])], 2) for k_ in ks]
    vc = [torch.cat([v_, torch.zeros_like(v_[:, :, :1])], 2) for v_ in vs]
    step_logits = decode_step(wide, idx[:, -1:], torch.tensor([ctx - 1], device=dev), kc, vc)[:, 0].float()
check_close("prefill + one cached decode step = full forward (bf16)", step_logits, full, atol=0.15, rtol=0)

# Decode attention for one query: FlashAttention kernel vs two GEMVs (bandwidth = KV bytes / time)
qd = torch.randn(256, 16, 1, 128, device=dev, dtype=torch.bfloat16)
kd, vd = torch.randn(256, 16, 512, 128, device=dev, dtype=torch.bfloat16), torch.randn(256, 16, 512, 128, device=dev, dtype=torch.bfloat16)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    t_fl = do_bench(lambda: F.scaled_dot_product_attention(qd, kd, vd))
t_gv = do_bench(lambda: ((qd @ kd.transpose(-1, -2)) / math.sqrt(128)).softmax(-1) @ vd)
print(f"decode attention over 512 cached tokens x 256 seqs: flash kernel {2 * kd.numel() * 2 / t_fl / 1e6:.0f} GB/s, "
      f"GEMVs {2 * kd.numel() * 2 / t_gv / 1e6:.0f} GB/s")
del qd, kd, vd

# %%
def time_decode(B, ctx, graph=True):
    kc = [torch.randn(B, wcfg.n_heads, ctx, wcfg.d_model // wcfg.n_heads, device=dev, dtype=torch.bfloat16) for _ in range(wcfg.n_layers)]
    vc = [torch.randn_like(k_) for k_ in kc]
    tok = torch.randint(0, wcfg.vocab_size, (B, 1), device=dev)
    pos = torch.tensor([ctx - 1], device=dev)
    fn = lambda: decode_step(wide, tok, pos, kc, vc)
    if graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        fn = g.replay
    return do_bench(fn) * 1e-3


kvw = kv_bytes_per_token(wcfg.n_layers, wcfg.n_heads, wcfg.d_model // wcfg.n_heads)
Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
with torch.no_grad():
    t_eager1 = time_decode(1, ctx, graph=False)
    t_meas = [time_decode(B, ctx) for B in Bs]
t_pred = [decode_step_time(Nw, B, ctx, kvw, P_3090, BW_3090) for B in Bs]
print(f"batch 1: eager {t_eager1 * 1e3:.2f} ms vs CUDA graph {t_meas[0] * 1e3:.2f} ms vs roofline {t_pred[0] * 1e3:.2f} ms")
for B, tm, tp in zip(Bs, t_meas, t_pred):
    print(f"B={B:4d}: {tm * 1e3:6.2f} ms/step  {B / tm:8.0f} tok/s   (roofline bound {tp * 1e3:5.2f} ms)")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
axs[0].loglog(Bs, np.array(t_meas) * 1e3, "o-", label="measured (CUDA graph)")
axs[0].loglog(Bs, np.array(t_pred) * 1e3, "--", label="roofline bound")
axs[0].set_xlabel("batch (sequences)"); axs[0].set_ylabel("ms per decode step"); axs[0].legend()
axs[1].loglog(Bs, np.array(Bs) / np.array(t_meas), "o-")
axs[1].set_xlabel("batch (sequences)"); axs[1].set_ylabel("generated tokens / s")
axs[0].set_title(f"{Nw / 1e6:.0f}M-param GPT, bf16, {ctx} cached tokens"); plt.tight_layout(); plt.show()
check("memory-bound regime: batch 16 costs < 1.6x batch 1", t_meas[Bs.index(16)] < 1.6 * t_meas[0])
check("throughput at batch 64 is > 25x batch 1", 64 / t_meas[Bs.index(64)] > 25 / t_meas[0])
check("eager batch-1 decode is slower than the CUDA graph (launch overhead)", t_eager1 > 1.3 * t_meas[0])

# %% [markdown]
# ## 3. Static vs continuous batching (a discrete-event simulator)
#
# **Cost model.** One engine iteration processes some prefill tokens and one token for each decoding sequence. We price
# it with the roofline of a 7B MHA model on an A100 (A100 numbers from the scaling book table: 3.1e14 bf16 FLOP/s,
# 2.0e12 B/s), with 60% achievable compute and 80% achievable bandwidth:
# $$t_\text{iter} = \max\Big(\frac{2N(\text{prefill tokens} + \text{decode seqs})}{0.6P},\ \frac{2N + \text{KV tokens read}\cdot 512\text{KiB}}{0.8\beta}\Big) + 1\text{ ms}.$$
#
# **Static batching** (the pre-Orca way): when the engine is free, take up to $B$ waiting requests, prefill them
# together, and decode until the *longest* finishes. Finished sequences idle in their slots, and new arrivals wait for
# the whole batch. **Continuous (iteration-level) batching** (Orca): the batch is re-formed *every iteration*: finished
# sequences leave, waiting ones join (their prefill runs in that iteration), so slots never idle.
#
# **Exercise 3.1.** Write `simulate_continuous(reqs, cost, max_batch, kv_capacity, chunk=None)`. Loop over iterations:
# 1. *Admit* arrived requests in FIFO order while `len(running) < max_batch` and the KV **reservation** of all running
#    requests (prompt + output tokens each; real engines don't know the output length and allocate on demand with
#    preemption, vLLM §4.5) plus the new one fits in `kv_capacity`. If nothing is running, jump `t` to the next arrival.
# 2. *Build the iteration*: for each running request that still has prompt left, prefill up to the remaining prompt
#    (or, if `chunk` is set, as much as fits in the per-iteration prefill-token budget `chunk`, shared by all requests in
#    order). Every request whose prompt is complete *before* this iteration decodes one token.
# 3. `t += cost(prefill_tokens, n_decode, kv_tokens)` where `kv_tokens` = cached tokens of the running requests (prompt
#    tokens prefilled so far + tokens generated).
# 4. *Emit*: a request whose prefill completed in this iteration emits its first token (TTFT); each decoding request
#    emits one token. Record the time of every token in `r["token_times"]`; remove a request when it has `output` tokens.
# Return the list of requests (each dict gets `token_times`). Use the provided `metrics`.

# %%
A100_P, A100_BW = 3.1e14, 2.0e12


def cost_7b(prefill_tokens, n_decode, kv_tokens):
    compute = 2 * N7 * (prefill_tokens + n_decode) / (0.6 * A100_P)
    memory = (2 * N7 + kv_tokens * kv7) / (0.8 * A100_BW)
    return max(compute, memory) + 1e-3


def make_requests(n, rate, prompt=(64, 1024), output=(16, 512), seed=0):
    rng = np.random.default_rng(seed)
    arr = np.cumsum(rng.exponential(1 / rate, n))
    return [dict(id=i, arrival=float(a), prompt=int(rng.integers(*prompt)), output=int(rng.integers(*output)))
            for i, a in enumerate(arr)]


def simulate_static(reqs, cost, batch):
    reqs = [dict(r, token_times=[]) for r in reqs]
    t, i = 0.0, 0
    while i < len(reqs):
        t = max(t, reqs[i]["arrival"])
        group = [r for r in reqs[i:i + batch] if r["arrival"] <= t]
        i += len(group)
        Tp = max(r["prompt"] for r in group)  # pad prompts to the longest
        t += cost(Tp * len(group), 0, 0)
        for r in group:
            r["token_times"].append(t)
        steps = max(r["output"] for r in group) - 1
        for k in range(steps):  # every slot runs until the longest output is done
            t += cost(0, len(group), len(group) * (Tp + k + 1))
            for r in group:
                if len(r["token_times"]) < r["output"]:
                    r["token_times"].append(t)
    return reqs


def simulate_continuous(reqs, cost, max_batch, kv_capacity, chunk=None):
    ### BEGIN SOLUTION
    reqs = [dict(r, token_times=[], done_prompt=0) for r in reqs]
    waiting, running, t = deque(reqs), [], 0.0
    reserved = 0
    while waiting or running:
        while (waiting and waiting[0]["arrival"] <= t and len(running) < max_batch
               and reserved + waiting[0]["prompt"] + waiting[0]["output"] <= kv_capacity):
            r = waiting.popleft()
            reserved += r["prompt"] + r["output"]
            running.append(r)
        if not running:
            t = waiting[0]["arrival"]
            continue
        budget = math.inf if chunk is None else chunk
        prefill_tokens, decoders, finishing_prefill = 0, [], []
        for r in running:
            if r["done_prompt"] < r["prompt"]:
                take = int(min(r["prompt"] - r["done_prompt"], budget - prefill_tokens))
                if take > 0:
                    r["done_prompt"] += take
                    prefill_tokens += take
                    if r["done_prompt"] == r["prompt"]:
                        finishing_prefill.append(r)
            else:
                decoders.append(r)
        kv_tokens = sum(r["done_prompt"] + len(r["token_times"]) for r in running)
        t += cost(prefill_tokens, len(decoders), kv_tokens)
        for r in finishing_prefill + decoders:
            r["token_times"].append(t)
        still = []
        for r in running:
            if len(r["token_times"]) >= r["output"]:
                reserved -= r["prompt"] + r["output"]
            else:
                still.append(r)
        running = still
    return reqs
    ### END SOLUTION


def metrics(done):
    out_tokens = sum(len(r["token_times"]) for r in done)
    span = max(r["token_times"][-1] for r in done) - min(r["arrival"] for r in done)
    ttft = np.array([r["token_times"][0] - r["arrival"] for r in done])
    lat = np.array([r["token_times"][-1] - r["arrival"] for r in done])
    gaps = np.concatenate([np.diff(r["token_times"]) for r in done if len(r["token_times"]) > 1])
    return dict(tok_s=out_tokens / span, ttft_mean=ttft.mean(), ttft_p99=np.percentile(ttft, 99),
                lat_mean=lat.mean(), lat_p99=np.percentile(lat, 99), tbt_p99=np.percentile(gaps, 99), tbt_max=gaps.max())


KV_CAP = int((80e9 - 2 * N7) * 0.9 / kv7)  # tokens of KV that fit on an 80 GB A100 next to the weights
reqs = make_requests(300, rate=4.0)
st = simulate_static(reqs, cost_7b, batch=16)
co = simulate_continuous(reqs, cost_7b, max_batch=64, kv_capacity=KV_CAP)

# checks: conservation and causality
check("every request gets exactly `output` tokens", all(len(r["token_times"]) == r["output"] for r in co))
check("no token before arrival, token times increasing",
      all(r["token_times"][0] > r["arrival"] and np.all(np.diff(r["token_times"]) > 0) for r in co))
tiny = simulate_continuous([dict(id=0, arrival=0.0, prompt=10, output=3)], lambda p, d, k: 1.0, 4, 100)
check("one request alone: prefill iteration emits token 1, then one token per iteration",
      tiny[0]["token_times"] == [1.0, 2.0, 3.0])
cap_test = simulate_continuous(make_requests(20, 100.0, (50, 51), (50, 51)), lambda p, d, k: 1.0, 64, kv_capacity=300)
check("KV reservation limits concurrency (300 tokens / 100 per request = 3 at a time)",
      max(sum(1 for r in cap_test if r["token_times"][0] <= tt < r["token_times"][-1] + 1e-9)
          for tt in np.arange(0, 400, 0.5)) <= 3)

for name, d in (("static, B=16", st), ("continuous, max 64", co)):
    m = metrics(d)
    print(f"{name:20s} {m['tok_s']:7.0f} tok/s   TTFT mean {m['ttft_mean']:6.2f} s   latency mean {m['lat_mean']:6.2f} s "
          f"p99 {m['lat_p99']:6.2f} s")
check("continuous batching: higher throughput and lower mean latency than static",
      metrics(co)["tok_s"] > 1.2 * metrics(st)["tok_s"] and metrics(co)["lat_mean"] < metrics(st)["lat_mean"])

# %% [markdown]
# ### 3.1 Throughput vs latency as load grows
# Sweep the arrival rate. Below saturation both systems keep up and latency is set by the work itself; near
# saturation queues explode. The throughput–latency curve is *the* serving trade-off: a bigger batch raises tokens/s per
# GPU but each step (TPOT) gets slower, and TTFT includes queueing.

# %%
rates = [0.5, 1, 2, 4, 6, 8]
curves = {"static B=16": [], "continuous": []}
for rate in rates:
    rq = make_requests(200, rate, seed=1)
    curves["static B=16"].append(metrics(simulate_static(rq, cost_7b, 16)))
    curves["continuous"].append(metrics(simulate_continuous(rq, cost_7b, 64, KV_CAP)))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
for k, v in curves.items():
    axs[0].plot(rates, [m["tok_s"] for m in v], "o-", label=k)
    axs[1].semilogy(rates, [m["lat_mean"] for m in v], "o-", label=k)
axs[0].set_xlabel("arrival rate (requests/s)"); axs[0].set_ylabel("throughput (output tokens/s)"); axs[0].legend()
axs[1].set_xlabel("arrival rate (requests/s)"); axs[1].set_ylabel("mean request latency (s)"); axs[1].legend()
axs[0].set_title("simulated 7B on one A100"); plt.tight_layout(); plt.show()

# %% [markdown]
# ### 3.2 Chunked prefill (Sarathi-Serve) and disaggregation (DistServe)
#
# In a mixed engine, a long prompt's prefill makes one iteration long, and every decoding sequence in the batch waits:
# a spike in **time between tokens (TBT)**. Sarathi splits prefills into chunks with a per-iteration token budget and
# piggybacks them on decode iterations ("stall-free" schedules). Decode iterations are memory-bound, so a few hundred
# prefill tokens ride along almost for free (the `max(compute, memory)` in our cost model). **Predict** what chunking does
# to the p99/max TBT and to TTFT, then run.

# %%
long_reqs = make_requests(200, rate=2.0, prompt=(512, 6000), output=(64, 256), seed=2)
res_chunk = {}
for chunk in (None, 2048, 512):
    d = simulate_continuous(long_reqs, cost_7b, 64, KV_CAP, chunk=chunk)
    res_chunk[chunk] = metrics(d)
    m = res_chunk[chunk]
    print(f"chunk={str(chunk):5s} TBT p99 {m['tbt_p99'] * 1e3:6.1f} ms  max {m['tbt_max'] * 1e3:6.1f} ms   "
          f"TTFT mean {m['ttft_mean']:5.2f} s   {m['tok_s']:6.0f} tok/s")
check("chunked prefill cuts the worst inter-token stall by > 2x", res_chunk[512]["tbt_max"] * 2 < res_chunk[None]["tbt_max"])

# %% [markdown]
# **Disaggregation** (DistServe) goes further: run prefill on one set of GPUs and decode on another, and ship the KV
# cache between them (here PCIe/NVLink/RDMA). Prefill and decode then never interfere, each phase gets its own batch size
# and parallelism, and you provision the ratio of prefill to decode GPUs for your TTFT and TPOT SLOs. The price is the KV
# transfer (a 4k-token prompt of a 7B MHA model is 2 GiB) and two pools to balance. DistServe optimizes **goodput**: the
# request rate served while meeting both SLOs for, e.g., 90% of requests.

# %% [markdown]
# ## 4. PagedAttention: a block table over a free list
#
# Contiguous KV allocation must reserve `max_len` slots per request up front: internal fragmentation (unused reserved
# slots), external fragmentation (holes between differently sized reservations), and no sharing. vLLM found existing
# systems used only 20.4-38.2% of KV memory for actual tokens (Kwon et al. §3). PagedAttention stores KV in fixed-size
# **blocks** (e.g. 16 tokens) anywhere in a pool; each sequence has a **block table** mapping logical block $i$ to a
# physical block, like OS paging. Waste is < 1 block per sequence. Blocks can be **shared**: a forked sequence (parallel
# sampling, beam search) points to the parent's blocks with reference counts, and a shared block is copied only when
# someone writes into it (**copy-on-write**).
#
# **Exercise 4.1.** Implement the allocator. `append(seq)` adds one token: if the sequence's last block is full (or it
# has none) take a block from the free list (raise `MemoryError` if empty); if the last block is shared (refcount > 1),
# copy-on-write: take a new block, decrement the old one's refcount, and record the copy in `self.copies`. `fork(parent,
# child)` shares all of the parent's blocks. `free(seq)` decrements refcounts and returns blocks that reach 0 to the free list.

# %%
class PagedKVAllocator:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free_list = list(range(num_blocks))[::-1]
        self.refcount = [0] * num_blocks
        self.tables = {}  # seq_id -> list of physical block ids
        self.lengths = {}  # seq_id -> number of tokens
        self.copies = 0  # copy-on-write events

    def _take(self):
        if not self.free_list:
            raise MemoryError("out of KV blocks")
        b = self.free_list.pop()
        self.refcount[b] = 1
        return b

    def add_sequence(self, seq):
        self.tables[seq], self.lengths[seq] = [], 0

    def append(self, seq):
        ### BEGIN SOLUTION
        table, n = self.tables[seq], self.lengths[seq]
        if n % self.block_size == 0:  # no block yet, or the last one is full
            table.append(self._take())
        elif self.refcount[table[-1]] > 1:  # writing into a shared block: copy on write
            old = table[-1]
            table[-1] = self._take()
            self.refcount[old] -= 1
            self.copies += 1
        self.lengths[seq] = n + 1
        ### END SOLUTION

    def fork(self, parent, child):
        ### BEGIN SOLUTION
        self.tables[child] = list(self.tables[parent])
        self.lengths[child] = self.lengths[parent]
        for b in self.tables[child]:
            self.refcount[b] += 1
        ### END SOLUTION

    def free(self, seq):
        ### BEGIN SOLUTION
        for b in self.tables.pop(seq):
            self.refcount[b] -= 1
            if self.refcount[b] == 0:
                self.free_list.append(b)
        del self.lengths[seq]
        ### END SOLUTION


def check_invariants(al):
    used = {b for t in al.tables.values() for b in t}
    ok = len(used) + len(al.free_list) == len(al.refcount) and not (used & set(al.free_list))
    ok &= all(al.refcount[b] == sum(t.count(b) for t in al.tables.values()) for b in used)
    ok &= all(len(al.tables[sq]) == math.ceil(al.lengths[sq] / al.block_size) for sq in al.tables)
    return ok


rng = np.random.default_rng(0)
al = PagedKVAllocator(num_blocks=64, block_size=4)
live, next_id, ok = [], 0, True
for step in range(3000):
    op = rng.random()
    try:
        if op < 0.1 or not live:
            al.add_sequence(next_id); live.append(next_id); next_id += 1
        elif op < 0.15:
            al.fork(live[rng.integers(len(live))], next_id); live.append(next_id); next_id += 1
        elif op < 0.25:
            al.free(live.pop(rng.integers(len(live))))
        else:
            al.append(live[rng.integers(len(live))])
    except MemoryError:
        al.free(live.pop(0))
    ok &= check_invariants(al)
check("allocator invariants hold through 3000 random add/fork/append/free operations", ok)
check("copy-on-write happened", al.copies > 0)
al2 = PagedKVAllocator(8, 4); al2.add_sequence("p")
for _ in range(6):
    al2.append("p")
al2.fork("p", "c"); al2.append("c")
check("fork shares full blocks and copies only the partially filled last block",
      al2.tables["c"][0] == al2.tables["p"][0] and al2.tables["c"][1] != al2.tables["p"][1] and al2.copies == 1)

# %%
# Fragmentation: the same request trace, contiguous max-length reservation vs 16-token blocks.
lens = rng.integers(64, 1024, 500) + rng.integers(16, 512, 500)  # prompt + output tokens actually used
max_len = 2048
util_contig = lens.sum() / (len(lens) * max_len)
util_paged = lens.sum() / (np.ceil(lens / 16).sum() * 16)
budget_tokens = 200_000
print(f"KV memory holding real tokens: contiguous(max_len={max_len}) {util_contig:.0%}   paged(16) {util_paged:.1%}")
print(f"concurrent requests in a {budget_tokens}-token KV budget: contiguous {budget_tokens // max_len}, "
      f"paged ~{int(budget_tokens / (np.ceil(lens / 16).mean() * 16))}")
check("paging wastes < 3% while max-length reservation wastes > 50%", util_paged > 0.97 and util_contig < 0.5)

# %% [markdown]
# ## 5. Prefix caching over token blocks
#
# Many requests share prefixes: a system prompt, few-shot examples, and above all **multi-turn agent loops**, where
# turn $k$'s prompt is turn $k-1$'s prompt + response + tool output. Recomputing the shared prefix's KV every turn
# wastes prefill FLOPs and TTFT. **Prefix caching** keeps KV blocks of finished requests and looks up the longest cached
# prefix of a new prompt. SGLang's **RadixAttention** keeps the cache as a radix tree with LRU eviction; vLLM's
# automatic prefix caching hashes each full block together with its prefix. Both work at block granularity, since the
# KV of a block depends on every token before it.
#
# **Exercise 5.1.** A trie whose edges are full blocks of `block_size` tokens (tuples). `insert(tokens)` adds every
# full block of the sequence along a path from the root; `match(tokens)` returns the number of tokens (a multiple of
# `block_size`) of the longest cached prefix.

# %%
class PrefixCache:
    def __init__(self, block_size):
        self.bs = block_size
        self.root = {}
        self.n_blocks = 0

    def insert(self, tokens):
        ### BEGIN SOLUTION
        node = self.root
        for i in range(0, len(tokens) - self.bs + 1, self.bs):
            key = tuple(tokens[i:i + self.bs])
            if key not in node:
                node[key] = {}
                self.n_blocks += 1
            node = node[key]
        ### END SOLUTION

    def match(self, tokens):
        ### BEGIN SOLUTION
        node, n = self.root, 0
        for i in range(0, len(tokens) - self.bs + 1, self.bs):
            key = tuple(tokens[i:i + self.bs])
            if key not in node:
                break
            node, n = node[key], n + self.bs
        return n
        ### END SOLUTION


def brute_match(cached, tokens, bs):
    best = 0
    for c in cached:
        k = 0
        while k < min(len(c), len(tokens)) and c[k] == tokens[k]:
            k += 1
        best = max(best, (k // bs) * bs)
    return best


rng = np.random.default_rng(1)
system = list(rng.integers(0, 50, 37))
pc, cached, ok = PrefixCache(8), [], True
for _ in range(200):
    base = list(cached[rng.integers(len(cached))]) if cached and rng.random() < 0.6 else system
    seq = base[: rng.integers(0, len(base) + 1)] + list(rng.integers(0, 50, rng.integers(1, 40)))
    ok &= pc.match(seq) == brute_match(cached, seq, 8)
    pc.insert(seq); cached.append(seq)
check("trie match equals brute-force longest common block prefix (200 random sequences)", ok)

# A multi-turn agent trace: system prompt, then each turn appends a response and a tool observation.
pc = PrefixCache(16)
convs, total, saved = 20, 0, 0
for c in range(convs):
    hist = list(range(1000)) + [10_000 + c] * 50  # shared 1000-token system prompt + a per-conversation task
    for turn in range(8):
        total += len(hist)
        saved += pc.match(hist)
        pc.insert(hist)
        hist = hist + list(rng.integers(20_000, 30_000, 300))  # model response + tool output
print(f"agent trace: {saved / total:.0%} of prompt tokens served from the prefix cache "
      f"({pc.n_blocks} blocks = {pc.n_blocks * 16 * kv7 / 2**30:.1f} GiB of 7B KV kept)")
check("prefix caching saves most prefill in multi-turn agent loops", saved / total > 0.7)

# %% [markdown]
# ## 6. Quantization in one paragraph, and the trade-offs
#
# Decode time ≈ bytes moved / bandwidth, so **weight-only quantization** (int8, int4 with per-group scales; GPTQ,
# AWQ) cuts the weight term by 2× or 4× and speeds up small-batch decoding nearly proportionally, as long as the
# dequantize-in-kernel (int → bf16 before the tensor-core math) is fused. It does nothing for compute-bound prefill
# unless the activations are quantized too (W8A8, FP8 on Hopper/Ada; not this 3090). KV-cache quantization (fp8/int8 KV)
# attacks the other memory term and raises the batch that fits.
#
# Summary of the levers: **batch** (continuous batching) raises throughput at a small TPOT cost until compute-bound
# or KV-bound; **chunked prefill** trades a little TTFT for smooth TPOT; **disaggregation** removes interference at the
# cost of KV transfer; **paging** raises the batch that fits; **prefix caching** cuts TTFT and prefill cost for shared
# prefixes; **GQA/MLA/KV quantization** shrink KV; **speculative decoding** (A3.18) spends spare FLOPs to emit several
# tokens per weight read; **CUDA graphs** remove launch overhead at small batch.

# %% [markdown]
# ## Say it out loud
# 1. Why is prefill compute-bound and decode memory-bound? Give the arithmetic intensity of each and the batch at which
#    decode becomes compute-bound on an A100 and on this 3090. Why does batching not help the attention part of decode?
# 2. KV cache size for a 7B MHA model at 4k context and batch 32 in bf16? How do MQA, GQA and MLA change it?
# 3. Static vs continuous batching: what does Orca's iteration-level scheduling fix? What new problem does mixing
#    prefills into decode batches create, and how does Sarathi fix it?
# 4. Explain PagedAttention like an OS person. What is the waste per sequence? How does copy-on-write help beam search
#    and parallel sampling? What does the attention kernel have to do differently?
# 5. How does prefix caching work in vLLM and SGLang, and why must it be block-granular and prefix-dependent? Why does
#    it matter so much for agents?
# 6. TTFT, TPOT, throughput, goodput: define each. You must cut p99 TTFT in half for a chat product without buying
#    GPUs. What do you try?
# 7. When does weight-only int4 quantization speed up serving, and when not?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. A step over $n$ tokens does $2Nn$ FLOPs and reads $2N$ bytes of bf16 weights: $I\approx n$. Prefill has $n=T_p$
#    (hundreds+): compute-bound. Decode has $n=B$: memory-bound until $B \approx P/\beta$: $3.1\text{e}14/2.0\text{e}12 \approx 155$ on
#    an A100, $71\text{e}12/936\text{e}9 \approx 76$ on a 3090 (scaling book quotes ~240-280 for TPU v5e/H100). Each sequence's KV cache
#    is read once per step for $O(1)$ FLOPs per byte and is not shared across the batch, so attention's intensity stays
#    ~1 regardless of $B$; at long context and large batch, KV reads dominate decode time (kipply 2022; Pope et al. 2023).
# 2. $2\cdot 32\cdot 32\cdot 128\cdot 2 = 512$ KiB per token → $512\text{KiB}\times4096\times32 = 64$ GiB. MQA: 1 KV head → 2 GiB
#    (÷32). GQA with 8 KV heads → 16 GiB (÷4). MLA caches a $d_c$-dim latent + a $d^R_h$ RoPE key per layer: DeepSeek-V2's
#    $(512+64)$ elements per layer equals GQA with 2.25 groups (DeepSeek-V2 Table 1).
# 3. Static batching forms a batch, runs it to completion, and pads: short requests idle in slots, and new arrivals wait
#    for the whole batch, so throughput and latency both suffer. Orca re-schedules at every iteration: finished sequences
#    leave and waiting ones join immediately (selective batching handles the attention of different lengths). Mixing a
#    long prefill into an iteration stalls every decode in it (TBT spikes); Sarathi-Serve chunks prefills to a token budget
#    and piggybacks the chunks on decode iterations (stall-free), keeping TPOT smooth. The TTFT cost is small or zero
#    when decode iterations are memory-bound and absorb the chunks for free; in our simulation TTFT even dropped slightly.
# 4. KV memory = physical pages (blocks of e.g. 16 tokens); each sequence has a page table (block table) from logical
#    to physical blocks; a free list hands out blocks on demand; waste is less than one block per sequence instead of
#    `max_len - len` (vLLM measured 20-38% utilization in prior systems). Forks share pages with reference counts;
#    a write to a shared page copies it first (copy-on-write), so beam candidates and parallel samples share their prompt
#    KV. The attention kernel must gather K/V through the block table (non-contiguous), which vLLM's PagedAttention kernel
#    does; preemption can swap or recompute blocks when memory runs out (Kwon et al. 2023).
# 5. Keep KV blocks after a request finishes; index them by their content and position: vLLM hashes (prefix hash, block
#    tokens), SGLang stores a radix tree of token sequences with LRU eviction. A new prompt walks the tree / hashes to find
#    its longest cached prefix and only prefills the rest. Block granularity because KV is stored and shared in blocks;
#    prefix-dependent because a token's K/V depend on all previous tokens (same tokens after a different prefix have
#    different KV). Agents re-send a growing history every turn, so most prompt tokens are cached: we measured most of them saved
#    in a synthetic trace (87%), cutting TTFT and prefill cost proportionally.
# 6. TTFT: arrival → first token (queueing + prefill). TPOT: average time between subsequent tokens. Throughput:
#    tokens/s (or requests/s) per GPU. Goodput: max request rate that meets the latency SLOs for a target fraction of
#    requests (DistServe). To cut p99 TTFT: prefix caching (agent/chat history), chunked prefill with a priority for new
#    prefills, cap the decode batch (or admission) so prefills are not starved, schedule short prompts first, route
#    across replicas by load and cache affinity, disaggregate prefill, speculative decoding does not help TTFT.
# 7. When decode is memory-bound on weights: small to moderate batch, so bytes dominate; int4 cuts weight bytes 4×, giving
#    up to ~2-3× faster TPOT after overheads (scales, dequant), and it frees memory for KV (bigger batch). It does not help
#    compute-bound prefill or large-batch decode unless activations are also quantized, it does not shrink the KV cache,
#    and slow unfused dequant kernels can erase the gain. Check the accuracy cost (GPTQ/AWQ keep it small at 4 bits).
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
