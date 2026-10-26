# %% [markdown]
# # A3.13 · KV cache and the incremental decoding loop
#
# **ID:** A3.13 · **Target:** 25 min · **Plan week:** 5 · **Part:** A
#
# **The drill.** Add a KV cache to a small GPT and write the decode loop: prefill the prompt once, then feed one token
# per step, reading past keys/values from the cache instead of recomputing them. Guide §A3 ("KV-cache decoding loop"),
# §C4 ("KV cache (implement)"), mock checklist ("MHA + KV cache in 20 min": the MHA half is A3.10).
#
# You write:
# 1. `KVCache`: a **preallocated** per-layer buffer (B, H_kv, T_max, d_head) with a fill pointer, and `update`.
# 2. `CachedAttention.forward`: causal attention over cached + new keys, with grouped K/V heads (GQA) and the
#    **bottom-right** causal mask for multi-token chunks.
# 3. `CachedGPT.forward`: the position-offset bookkeeping.
# 4. `generate_greedy` (with and without cache).
# 5. The KV-cache memory formula $2\cdot L\cdot H_{kv}\cdot d_{head}\cdot T\cdot B\cdot\text{bytes}$ and the FLOPs of one decode step.
#
# The model scaffolding (embeddings, MLP, LayerNorms) is given and is weight-compatible with `prep.tinygpt.GPT`, which
# serves as the trusted full-recompute reference. Tests: incremental logits == full-recompute logits (float64, atol 1e-10),
# identical greedy generations, chunked prefill, GQA, measured speedup, and your formulas vs the actual buffers and
# `FlopCounterMode`. You may use `F.scaled_dot_product_attention` or explicit matmuls (explicit matmuls are what
# `FlopCounterMode` counts in §5, so the solution uses them).
#
# **Deeper study:** Raschka, *LLMs from scratch*, KV-cache bonus (concatenating cache, then a preallocated/sliding-window
# version): `../../LLMs-from-scratch/ch04/03_kv-cache/README.md`, `gpt_with_kv_cache.py` and
# `gpt_with_kv_cache_optimized.py` in the same folder; GQA cache sizes: `../../LLMs-from-scratch/ch04/04_gqa/README.md`.
# Raschka, *Reasoning from scratch* §2.8 "Faster inference via KV caching":
# https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch02/01_main-chapter-code/ch02_main.ipynb
# and its preallocated `KVCache` in `reasoning_from_scratch/qwen3_optimized.py` (same commit).
#
# ## Sources
# - S. Raschka, *Build a Large Language Model (From Scratch)*, `ch04/03_kv-cache/gpt_with_kv_cache.py`,
#   `gpt_with_kv_cache_optimized.py`, `ch04/04_gqa/README.md` (KV-storage formula and the 17.18 GB / 4.29 GB example
#   checked below), rasbt/LLMs-from-scratch commit f9313ea, Apache-2.0 — https://github.com/rasbt/LLMs-from-scratch
# - S. Raschka, *Build a Reasoning Model (From Scratch)*, ch02 §2.8 and `reasoning_from_scratch/qwen3_optimized.py`
#   (`KVCache`), rasbt/reasoning-from-scratch commit a788466, Apache-2.0 — https://github.com/rasbt/reasoning-from-scratch
# - `prep/tinygpt.py` (this repo): GPT-2-style reference model, itself following Raschka ch04 `gpt.py` and nanoGPT.
# - N. Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019, arXiv:1911.02150 —
#   https://arxiv.org/abs/1911.02150 (incremental decoding is memory-bandwidth bound; multi-query attention shrinks K/V).
# - J. Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023,
#   arXiv:2305.13245 — https://arxiv.org/abs/2305.13245
# - R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, A. Levskaya, J. Heek, K. Xiao, S. Agrawal, J. Dean,
#   *Efficiently Scaling Transformer Inference*, MLSys 2023, arXiv:2211.05102 — https://arxiv.org/abs/2211.05102
#   (KV-cache memory, why decode is memory-bound and prefill compute-bound).
# - W. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM), SOSP 2023,
#   arXiv:2309.06180 — https://arxiv.org/abs/2309.06180 (paged KV cache; fragmentation of preallocated buffers).
# - H. Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023, arXiv:2302.13971 —
#   https://arxiv.org/abs/2302.13971 (Table 2: the 7B model has d=4096, 32 heads, 32 layers, used in the memory example).
# - Further reading for answer 8 (cache compression): DeepSeek-AI, *DeepSeek-V2* (multi-head latent attention), 2024,
#   arXiv:2405.04434 — https://arxiv.org/abs/2405.04434 · Z. Zhang et al., *H$_2$O: Heavy-Hitter Oracle for Efficient
#   Generative Inference of Large Language Models*, NeurIPS 2023, arXiv:2306.14048 — https://arxiv.org/abs/2306.14048 ·
#   G. Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, ICLR 2024, arXiv:2309.17453 —
#   https://arxiv.org/abs/2309.17453
# - PyTorch `torch.utils.flop_counter.FlopCounterMode` — https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py

# %%
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from prep import Session, check, check_close, seed_everything
from prep.tinygpt import GPT, GPTConfig

s = Session("A3.13", target_min=25)
seed_everything(0)
torch.set_default_dtype(torch.float64)  # float64: cached and recomputed logits agree to ~1e-12
cfg = GPTConfig(vocab_size=512, context_length=128, d_model=128, n_heads=8, n_layers=4)
ref_model = GPT(cfg).eval()  # trusted full-recompute reference (random weights; we only test equality)

# %% [markdown]
# ## 1. The cache: preallocate once, write in place
# Concatenating (`torch.cat([cache_k, k_new])`) every step is the simplest cache, but it reallocates and copies the whole
# cache each step ($O(T^2)$ bytes moved over a generation). Real servers preallocate (or page, as in vLLM) and write the new
# entries at a pointer. Shapes: K and V each (B, H_kv, T_max, d_head) **per layer**.

# %%
class KVCache:
    """One layer's cache. self.k, self.v: (B, n_kv_heads, T_max, d_head) zeros; self.len: filled positions."""

    def __init__(self, B, n_kv_heads, T_max, d_head, dtype=None, device=None):
        ### BEGIN SOLUTION
        self.k = torch.zeros(B, n_kv_heads, T_max, d_head, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.len = 0
        ### END SOLUTION

    def update(self, k_new, v_new):
        """Write k_new, v_new (B, H_kv, T_new, d_head) at positions [len, len+T_new), advance len, and return
        views of everything cached so far: (k, v) each (B, H_kv, len, d_head)."""
        ### BEGIN SOLUTION
        t0, t1 = self.len, self.len + k_new.shape[2]
        assert t1 <= self.k.shape[2], "KV cache overflow"
        self.k[:, :, t0:t1] = k_new
        self.v[:, :, t0:t1] = v_new
        self.len = t1
        return self.k[:, :, :t1], self.v[:, :, :t1]
        ### END SOLUTION


c = KVCache(2, 3, 10, 4)
k1, v1, k2, v2 = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4), torch.randn(2, 3, 1, 4), torch.randn(2, 3, 1, 4)
c.update(k1, v1)
kk, vv = c.update(k2, v2)
check("cache len after 4 + 1 tokens", c.len == 5, f"got {c.len}")
check_close("cached k == cat of the pieces", kk, torch.cat([k1, k2], 2))
check_close("cached v == cat of the pieces", vv, torch.cat([v1, v2], 2))
check("returns views of the preallocated buffer (no copy)", kk.data_ptr() == c.k.data_ptr() and c.k.shape == (2, 3, 10, 4))

# %% [markdown]
# ## 2. Attention that reads the cache
# At a step with $T$ new tokens and $S$ cached positions *after* appending, the new queries are positions
# $S-T, \dots, S-1$, so query $i$ may attend to keys $j \le i + (S - T)$: the **bottom-right** causal mask (A3.10 §2).
# For $T = 1$ that is "attend to everything". `is_causal=True` in SDPA is top-left aligned and would be wrong for $T>1$
# chunks **and** for $T = 1$ (it lets the single query see only key 0).
#
# The fused `qkv` projection has output width $(H + 2H_{kv})\,d_{head}$, laid out as [Q (all heads) | K | V], so with
# $H_{kv}=H$ it is weight-for-weight identical to `prep.tinygpt`. Query head $h$ uses KV head $\lfloor h / (H/H_{kv}) \rfloor$
# (`repeat_interleave`). Cache the **un-repeated** K/V: storing the repeated ones would throw away GQA's memory saving.

# %%
class CachedAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.H, self.H_kv, self.Dh = n_heads, n_kv_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, (n_heads + 2 * n_kv_heads) * self.Dh, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None):
        """x: (B, T, d_model) new tokens. cache: KVCache or None (None = plain causal attention over x).
        Returns (B, T, d_model)."""
        ### BEGIN SOLUTION
        B, T, D = x.shape
        q, k, v = self.qkv(x).split([self.H * self.Dh, self.H_kv * self.Dh, self.H_kv * self.Dh], dim=-1)
        q = q.view(B, T, self.H, self.Dh).transpose(1, 2)              # (B, H, T, Dh)
        k = k.view(B, T, self.H_kv, self.Dh).transpose(1, 2)           # (B, H_kv, T, Dh)
        v = v.view(B, T, self.H_kv, self.Dh).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(k, v)                                  # (B, H_kv, S, Dh), S = past + T
        S = k.shape[2]
        g = self.H // self.H_kv
        k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.Dh)           # (B, H, T, S)
        keep = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)  # bottom-right
        att = att.masked_fill(~keep, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y)
        ### END SOLUTION


# %% [markdown]
# ## 3. The model: position offsets
# Given: blocks and weights in `prep.tinygpt` layout. You write `forward`: new tokens at absolute positions
# `start_pos ... start_pos + T - 1` must get **those** learned position embeddings (forgetting the offset gives every
# decoded token position 0 and is the classic KV-cache bug: no error, wrong text). For RoPE models the same offset goes
# into the rotation angles.

# %%
class Block(nn.Module):
    def __init__(self, cfg, n_kv_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = CachedAttention(cfg.d_model, cfg.n_heads, n_kv_heads)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class CachedGPT(nn.Module):
    def __init__(self, cfg, n_kv_heads=None):
        super().__init__()
        self.cfg, self.n_kv_heads = cfg, n_kv_heads or cfg.n_heads
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, self.n_kv_heads) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied, as in prep.tinygpt
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def new_caches(self, B, T_max=None):
        """One KVCache per layer, same dtype/device as the weights."""
        w = self.tok_emb.weight
        return [KVCache(B, self.n_kv_heads, T_max or self.cfg.context_length, self.cfg.d_model // self.cfg.n_heads,
                        dtype=w.dtype, device=w.device) for _ in self.blocks]

    def forward(self, idx, caches=None, start_pos=0):
        """idx: (B, T) new token ids at absolute positions start_pos..start_pos+T-1. caches: list of KVCache
        (one per layer) or None. Returns logits (B, T, vocab)."""
        ### BEGIN SOLUTION
        T = idx.shape[1]
        assert start_pos + T <= self.cfg.context_length
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        return self.head(self.ln_f(x))
        ### END SOLUTION


model = CachedGPT(cfg).eval()
model.load_state_dict(ref_model.state_dict())  # same parameter names and shapes when n_kv_heads == n_heads
idx = torch.randint(0, cfg.vocab_size, (3, 40))
with torch.no_grad():
    full = ref_model(idx)
    check_close("no-cache forward == prep.tinygpt reference", model(idx), full, atol=1e-10)

    # prefill 16 tokens, then one token per step
    caches = model.new_caches(B=3)
    steps = [model(idx[:, :16], caches, start_pos=0)]
    for t in range(16, 40):
        steps.append(model(idx[:, t:t + 1], caches, start_pos=t))
    inc = torch.cat(steps, dim=1)
    check_close("incremental decode logits == full recompute (all 40 positions)", inc, full, atol=1e-10)
    check("cache filled to 40", all(c.len == 40 for c in caches))

    # chunked prefill: 7, then chunks of 5 and 1 tokens (exercises the bottom-right mask with T > 1)
    caches, chunks, t = model.new_caches(B=3), [], 0
    for n in (7, 5, 1, 5, 1, 1, 20):
        chunks.append(model(idx[:, t:t + n], caches, start_pos=t)); t += n
    check_close("chunked prefill + decode == full recompute", torch.cat(chunks, 1), full, atol=1e-10)

# %% [markdown]
# ## 4. The generation loop
# Greedy decoding, `max_new` tokens. Without a cache: at each step run the whole sequence so far and take the last
# position's logits. With a cache: **prefill** the prompt in one call (fills every layer's cache, gives the first next-token
# logits), then **decode** one token per call at `start_pos = current length`. Do not feed the prompt again.

# %%
@torch.no_grad()
def generate_greedy(model, prompt, max_new, use_cache=True):
    """prompt: (B, P) ids -> (B, P + max_new) ids. Requires P + max_new <= context_length."""
    ### BEGIN SOLUTION
    B, P = prompt.shape
    assert P + max_new <= model.cfg.context_length
    out = prompt
    if not use_cache:
        for _ in range(max_new):
            nxt = model(out)[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
        return out
    caches = model.new_caches(B, T_max=P + max_new)
    logits = model(prompt, caches, start_pos=0)                    # prefill
    for t in range(max_new):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        if t + 1 < max_new:
            logits = model(nxt, caches, start_pos=P + t)            # decode one token
    return out
    ### END SOLUTION


@torch.no_grad()
def reference_greedy(m, prompt, max_new):  # independent loop on prep.tinygpt, no cache
    out = prompt
    for _ in range(max_new):
        out = torch.cat([out, m(out)[:, -1].argmax(-1, keepdim=True)], 1)
    return out


prompt = torch.randint(0, cfg.vocab_size, (2, 10))
g_ref = reference_greedy(ref_model, prompt, 50)
check("greedy without cache == reference", torch.equal(generate_greedy(model, prompt, 50, use_cache=False), g_ref))
check("greedy with cache == reference", torch.equal(generate_greedy(model, prompt, 50, use_cache=True), g_ref))
check("output length P + max_new", generate_greedy(model, prompt, 7).shape == (2, 17))

# %% [markdown]
# ### GQA: the same cache with fewer K/V heads
# Build a GQA model ($H=8$, $H_{kv}=2$). There is no `tinygpt` reference with GQA, so the reference is the same model run
# without a cache (full recompute), whose attention you already validated above with $H_{kv}=H$.

# %%
gqa = CachedGPT(cfg, n_kv_heads=2).eval()
with torch.no_grad():
    att, xa = gqa.blocks[0].attn, torch.randn(3, 11, cfg.d_model)   # head pairing vs SDPA's enable_gqa
    qa, ka, va = att.qkv(xa).split([8 * att.Dh, 2 * att.Dh, 2 * att.Dh], -1)
    qa, ka, va = (t_.view(3, 11, -1, att.Dh).transpose(1, 2) for t_ in (qa, ka, va))
    ya = F.scaled_dot_product_attention(qa, ka, va, is_causal=True, enable_gqa=True)
    check_close("GQA attention (no cache) vs SDPA(enable_gqa=True)", att(xa), att.proj(ya.transpose(1, 2).reshape(3, 11, -1)), atol=1e-10)
    full_g = gqa(idx)
    caches = gqa.new_caches(B=3)
    inc_g = torch.cat([gqa(idx[:, :9], caches, 0)] + [gqa(idx[:, t:t + 1], caches, t) for t in range(9, 40)], 1)
check_close("GQA incremental == GQA full recompute", inc_g, full_g, atol=1e-10)
check("GQA cache stores H_kv=2 heads, not H=8", caches[0].k.shape == (3, 2, cfg.context_length, cfg.d_model // cfg.n_heads),
      str(caches[0].k.shape))
check("GQA greedy: cache == no cache",
      torch.equal(generate_greedy(gqa, prompt, 40), generate_greedy(gqa, prompt, 40, use_cache=False)))

# %% [markdown]
# ## 5. Memory and per-token cost
# **Memory.** Per layer you store K and V, each $H_{kv}\cdot d_{head}$ numbers per token per sequence:
# $$\text{KV bytes} = 2\cdot L\cdot H_{kv}\cdot d_{head}\cdot T\cdot B\cdot\text{bytes/elem}.$$
# **Compute of one decode step** at cache length $S$ (after appending the new token): every weight matrix is used once,
# $2N_{mat}$ FLOPs where $N_{mat}$ counts the parameters of the `Linear` layers (the tied output head included, embeddings
# and LayerNorms excluded), plus attention $QK^\top$ and $PV$ over $S$ keys with $H$ query heads: $2\cdot 2\cdot L\cdot H d_{head}\cdot S$.
# GQA does **not** reduce attention FLOPs (all $H$ query heads still score every key); it reduces bytes read.

# %%
def kv_cache_bytes(n_layers, n_kv_heads, d_head, T, B, bytes_per_el):
    ### BEGIN SOLUTION
    return 2 * n_layers * n_kv_heads * d_head * T * B * bytes_per_el
    ### END SOLUTION


def decode_step_flops(model, S):
    """Matmul FLOPs (multiply-add = 2) of one decode step (B=1, one new token) when the cache holds S positions
    after appending it. Use model.cfg and the model's nn.Linear modules."""
    ### BEGIN SOLUTION
    n_mat = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))  # head counted once (tied)
    c = model.cfg
    return 2 * n_mat + 4 * c.n_layers * c.d_model * S
    ### END SOLUTION


for m_, name in ((model, "MHA"), (gqa, "GQA")):
    cs = m_.new_caches(B=3)
    actual = sum(c.k.nbytes + c.v.nbytes for c in cs)
    check(f"{name}: formula == bytes of the allocated buffers",
          kv_cache_bytes(cfg.n_layers, m_.n_kv_heads, cfg.d_model // cfg.n_heads, cfg.context_length, 3, 8) == actual)
# Raschka ch04/04_gqa README example: d=4096, 32 heads, 32 layers, T=32768, B=1, bf16 -> MHA 17.18 GB, GQA(8 kv) 4.29 GB
check("MHA 7B-class at 32k context = 17.18 GB", round(kv_cache_bytes(32, 32, 128, 32768, 1, 2) / 1e9, 2) == 17.18)
check("GQA with 8 KV heads = 4.29 GB", round(kv_cache_bytes(32, 8, 128, 32768, 1, 2) / 1e9, 2) == 4.29)

with torch.no_grad():
    for S_ in (12, 40):
        cs = model.new_caches(B=1)
        model(idx[:1, :S_ - 1], cs, 0)
        with FlopCounterMode(display=False) as fc:
            model(idx[:1, S_ - 1:S_], cs, S_ - 1)
        check(f"decode_step_flops(S={S_}) == FlopCounterMode", decode_step_flops(model, S_) == fc.get_total_flops(),
              f"{decode_step_flops(model, S_)} vs counted {fc.get_total_flops()}")

per_tok = kv_cache_bytes(32, 32, 128, 1, 1, 2)
print(f"LLaMA-7B-shaped model (L=32, H=32, d_head=128, fp16): {per_tok / 2**20:.2f} MiB of KV cache per token; "
      f"a 4096-token sequence: {4096 * per_tok / 2**30:.1f} GiB, vs ~13.4 GB of fp16 weights for its 6.7B params")

# %% [markdown]
# ## 6. Measured speedup
# float32 on CPU (deterministic enough to time; the GPU is shared). Without a cache, step $t$ re-runs all $t$ tokens, so total
# work grows ~quadratically with the generated length; with a cache each step processes one token.

# %%
torch.set_default_dtype(torch.float32)
cfg_t = GPTConfig(vocab_size=512, context_length=512, d_model=256, n_heads=8, n_layers=4)
m32 = CachedGPT(cfg_t).eval()
p32 = torch.randint(0, cfg_t.vocab_size, (1, 32))
timings = {}
for use_cache in (True, False):
    generate_greedy(m32, p32, 8, use_cache=use_cache)  # warm-up
    t0 = time.perf_counter()
    out = generate_greedy(m32, p32, 384, use_cache=use_cache)
    timings[use_cache] = time.perf_counter() - t0
    print(f"use_cache={use_cache!s:5}: {timings[use_cache]:.2f} s for 384 tokens ({384 / timings[use_cache]:.0f} tok/s)")
speedup = timings[False] / timings[True]
print(f"speedup: {speedup:.1f}x")
check("KV cache is at least 2x faster at 416 tokens", speedup > 2.0, f"{speedup:.2f}x")
check("float32 cached vs uncached greedy agree on the first 32 tokens",
      torch.equal(generate_greedy(m32, p32, 32), generate_greedy(m32, p32, 32, use_cache=False)))

# %% [markdown]
# ## Say it out loud
# 1. What exactly is cached, per layer, and why are queries not cached? Why can't you cache the attention *outputs*?
# 2. Give the KV-cache memory formula. How big is it for a 7B LLaMA-shaped model at 4k context in fp16, per sequence?
#    What does GQA with 8 KV heads change, and what does it not change?
# 3. Prefill vs decode: which is compute-bound and which memory-bandwidth-bound, and why? What does that imply for batching?
# 4. Name three bugs that make cached decoding silently wrong.
# 5. Preallocated buffer vs `torch.cat` vs paged cache (vLLM): trade-offs?
# 6. What happens when generation reaches the context length? (sliding window, and why you can't just drop the oldest
#    entry with absolute positional embeddings)
# 7. How does batched generation with prompts of different lengths interact with the cache? (padding side, masks, positions)
# 8. What other ways are there to make the cache smaller?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Per layer, the keys and values of every past position: K, V each (B, H_kv, T, d_head). A past token's K/V depend only on
#    tokens at or before it (causal), so they never change once computed. Queries are only needed for the *new* token (its
#    query attends to all cached keys); old queries are never used again. Attention outputs of old positions are also fixed,
#    but the *next layer* needs K/V, not outputs. Caching each layer's K/V is what lets every layer skip old positions.
# 2. $2\cdot L\cdot H_{kv}\cdot d_{head}\cdot T\cdot B\cdot$bytes. LLaMA-7B shape (L=32, H=32, d_head=128; Touvron et al. 2023,
#    Table 2) in fp16: $2\cdot32\cdot32\cdot128\cdot2 = 512$ KiB per token, so 2 GiB at T=4096, B=1 (printed above). GQA with 8 KV
#    heads cuts that by 4× (0.5 GiB) and cuts the bytes read per decode step equally. It does not change attention FLOPs or
#    the weight memory (much), and quality is close to MHA when uptrained (Ainslie et al. 2023).
# 3. Prefill processes P tokens in parallel: big matmuls, arithmetic intensity ∝ P, compute-bound. Decode processes one token
#    per sequence: every weight and the whole KV cache are read from HBM to do ~2 FLOPs per weight, so it is
#    memory-bandwidth-bound (Shazeer 2019; Pope et al. 2023). Batching B sequences reuses each weight read B times, so
#    throughput rises almost linearly until compute-bound, but the KV cache grows with B and is read per sequence, which
#    caps the batch size and is why KV memory, not FLOPs, limits serving throughput.
# 4. (a) Not offsetting positions (`start_pos`, RoPE angles) for decoded tokens. (b) Using a top-left causal mask
#    (`is_causal=True`) with cached keys, or masking wrongly for chunked prefill. (c) Forgetting to reset the cache between
#    prompts / reusing it across batch elements. (d) Feeding the whole sequence again with the cache on (keys get appended
#    twice). (e) Caching K/V after `repeat_interleave` (correct but loses GQA's savings), or pairing heads with `repeat`.
#    The test in §3 (incremental logits == full logits) catches all the correctness ones.
# 5. `torch.cat`: simplest, but reallocates and copies the cache every step ($O(T^2)$ bytes moved) and fragments memory.
#    Preallocated `(B, H_kv, T_max, d)`: one allocation and in-place writes, but reserves T_max for every sequence even if it
#    stops early (internal fragmentation). Paged (vLLM): fixed-size blocks allocated on demand plus a block table per
#    sequence, so near-zero waste, sharing of prefix blocks across sequences (parallel sampling, beam search), at the cost of
#    a gather-style attention kernel (Kwon et al. 2023).
# 6. With learned absolute positions you can't go past `context_length`. Dropping the oldest cache entry breaks the model:
#    the cached keys were computed with their absolute position embeddings, and new tokens would need positions beyond the
#    trained range. Options: stop, re-prefill a truncated window (recomputing all K/V), or use a model trained with
#    sliding-window attention (Raschka's `gpt_with_kv_cache_optimized.py` shows a rolling buffer), or relative positions
#    (RoPE) plus attention sinks / context extension.
# 7. Left-pad the prompts so every sequence's next token is at the same tensor index. Pad keys must be masked
#    (a key-padding mask carried with the cache), and each sequence's positions must start at its first real token
#    (per-sequence `position_ids`), otherwise padding shifts the position embeddings. Servers avoid padding entirely with
#    continuous batching over a paged cache.
# 8. GQA/MQA (fewer KV heads); KV quantization (int8/fp8/int4); multi-head latent attention (cache a low-rank latent,
#    DeepSeek-V2, arXiv:2405.04434); sliding-window or local attention layers; cross-layer KV sharing; token eviction
#    (H2O heavy hitters, arXiv:2306.14048; StreamingLLM attention sinks, arXiv:2309.17453). Pointers: `../../LLMs-from-scratch/ch04/05_mla`, `06_swa`, `10_kv-sharing`.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
