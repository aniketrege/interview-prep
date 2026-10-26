# %% [markdown]
# # A3.10 · Attention from scratch: SDPA, masks, MHA, cross-attention, GQA
#
# **ID:** A3.10 · **Target:** 20 min · **Plan week:** 4 · **Part:** A
#
# **The drill.** The single most-asked ML-coding question: write scaled dot-product attention and multi-head
# attention in a blank file, with a causal mask, and answer the complexity follow-ups. Guide §A3: "Scaled
# dot-product + multi-head attention with causal mask"; §C2: "Multi-head, multi-query, grouped-query attention;
# causal masking"; mock checklist: "From a blank file: MHA + KV cache in 20 min" (the KV-cache half is A3.13).
#
# You write, in order:
# 1. `sdpa`: attention by hand, with the exact semantics of `F.scaled_dot_product_attention` (bool and additive
#    masks, `is_causal`, `scale`, and what happens to a query row whose keys are all masked).
# 2. Mask helpers: top-left vs bottom-right causal masks, key-padding mask, bool ↔ additive.
# 3. `MultiHeadAttention` and a weight copier from `nn.MultiheadAttention` (its `in_proj_weight` layout).
# 4. Cross-attention with the same module: queries of length L, keys/values of length S ≠ L and another width.
# 5. `GroupedQueryAttention` (MQA when `n_kv_heads=1`), checked against SDPA's `enable_gqa=True`.
# 6. FLOP and memory formulas, checked with `torch.utils.flop_counter` and measured GPU memory.
#
# **Rules.** Inside your functions use only tensor ops (`@`, `softmax`, `masked_fill`, `view`, `transpose`,
# `repeat_interleave`, ...) and `nn.Linear`. **Do not call** `F.scaled_dot_product_attention` or
# `nn.MultiheadAttention` in your code: the tests use them as the reference. Fill in every
# `raise NotImplementedError`; the cell under each function prints `ok` or says what is off. Interview mode:
# do §1 and §3 from a blank file first, then come back for the rest.
#
# **Deeper study (Raschka, *LLMs from scratch*, local copy):**
# - ch03 §3.3-3.6 (self-attention → causal → multi-head with weight splits):
#   `../../LLMs-from-scratch/ch03/01_main-chapter-code/ch03.ipynb` and `multihead-attention.ipynb` in the same folder.
# - Ten MHA implementations benchmarked (einsum, SDPA, `nn.MultiheadAttention`, FlexAttention):
#   `../../LLMs-from-scratch/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb`.
# - GQA with `repeat_interleave` and the KV-memory formula: `../../LLMs-from-scratch/ch04/04_gqa/README.md` and
#   `../../LLMs-from-scratch/ch04/04_gqa/gpt_with_kv_gqa.py`.
#
# ## Sources
# - A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, *Attention Is
#   All You Need*, NeurIPS 2017, arXiv:1706.03762 — https://arxiv.org/abs/1706.03762 (§3.2.1 scaled dot-product and
#   footnote 4 on why $1/\sqrt{d_k}$; §3.2.2 multi-head; §3.2.3 encoder-decoder (cross) attention; §4 complexity table).
# - N. Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (multi-query attention), 2019,
#   arXiv:1911.02150 — https://arxiv.org/abs/1911.02150
# - J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, S. Sanghai, *GQA: Training Generalized
#   Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023, arXiv:2305.13245 —
#   https://arxiv.org/abs/2305.13245 (§2.1 uptraining by mean-pooling K/V heads; 5% of pre-training compute).
# - T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with
#   IO-Awareness*, NeurIPS 2022, arXiv:2205.14135 — https://arxiv.org/abs/2205.14135 (Theorem 1: $O(N^2 d)$ FLOPs,
#   $O(N)$ extra memory; Theorem 2: $\Theta(N^2 d^2 M^{-1})$ vs $\Theta(Nd + N^2)$ HBM accesses).
# - M. N. Rabe, C. Staats, *Self-attention Does Not Need $O(n^2)$ Memory*, 2021, arXiv:2112.05682 —
#   https://arxiv.org/abs/2112.05682
# - A. Haviv, O. Ram, O. Press, P. Izsak, O. Levy, *Transformer Language Models without Positional Encodings Still Learn
#   Positional Information*, Findings of EMNLP 2022, arXiv:2203.16634 — https://arxiv.org/abs/2203.16634
# - PyTorch docs (reference semantics the tests check):
#   `F.scaled_dot_product_attention` https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html ·
#   `nn.MultiheadAttention` https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html ·
#   `torch.nn.attention.bias` (`causal_upper_left`, `causal_lower_right`) https://docs.pytorch.org/docs/stable/nn.attention.bias.html ·
#   `torch.utils.flop_counter.FlopCounterMode` https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py
# - S. Raschka, *Build a Large Language Model (From Scratch)*, ch03 and ch04/04_gqa code, rasbt/LLMs-from-scratch
#   commit f9313ea, Apache-2.0 — https://github.com/rasbt/LLMs-from-scratch

# %%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right, causal_upper_left
from torch.utils.flop_counter import FlopCounterMode

from prep import Session, check, check_close, seed_everything

s = Session("A3.10", target_min=20)
seed_everything(0)
torch.set_default_dtype(torch.float64)  # float64 on CPU: the tests can use tight tolerances
B, H, L, S, E = 2, 3, 5, 7, 8
q, k, v = torch.randn(B, H, L, E), torch.randn(B, H, S, E), torch.randn(B, H, S, 6)  # Ev = 6 != E on purpose

# %% [markdown]
# ## 1. Scaled dot-product attention by hand
# $$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}} + M\Big)V,\qquad Q\in\mathbb R^{L\times d_k},
# K\in\mathbb R^{S\times d_k}, V\in\mathbb R^{S\times d_v}.$$
# The softmax is over the **key** axis (last dim of the $L\times S$ score matrix). SDPA's mask conventions:
# - **bool** `attn_mask`: `True` = *may attend* (masked-out scores become $-\infty$);
# - **float** `attn_mask`: *added* to the scores (so 0 = keep, $-\infty$ = drop, anything else = a bias such as ALiBi);
# - `is_causal=True`: query $i$ sees keys $j \le i$, aligned to the **top-left** corner even when $L \ne S$.
#
# A query row whose keys are **all** masked has $\mathrm{softmax}(-\infty,\dots,-\infty) = 0/0 =$ NaN. Run the next cell
# to see what PyTorch 2.14 does: SDPA returns **zeros** for such rows, `nn.MultiheadAttention` returns **NaN**.
# Your `sdpa` must match SDPA (zeros, no NaN).

# %%
m = torch.ones(3, 3, dtype=torch.bool); m[1] = False            # row 1 may attend to nothing
x3 = torch.randn(1, 1, 3, 4)
print("SDPA, fully-masked row 1:\n", F.scaled_dot_product_attention(x3, x3, x3, attn_mask=m)[0, 0])
mha_demo = nn.MultiheadAttention(4, 1, batch_first=True)
print("nn.MultiheadAttention (its attn_mask uses True = BLOCK):\n", mha_demo(x3[0], x3[0], x3[0], attn_mask=~m)[0][0].detach())


# %%
def sdpa(q, k, v, attn_mask=None, is_causal=False, scale=None):
    """q: (..., L, E), k: (..., S, E), v: (..., S, Ev); attn_mask bool (True = attend) or float (added),
    broadcastable to (..., L, S). is_causal: top-left aligned, key j allowed iff j <= i. scale defaults to
    1/sqrt(E). Fully masked rows give 0. Returns (out (..., L, Ev), weights (..., L, S))."""
    ### BEGIN SOLUTION
    scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else scale
    scores = (q @ k.transpose(-2, -1)) * scale                    # (..., L, S)
    if is_causal:
        Lq, Sk = scores.shape[-2:]
        keep = torch.ones(Lq, Sk, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~keep, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    w = torch.softmax(scores, dim=-1)
    w = torch.nan_to_num(w, nan=0.0)                              # all -inf row: 0/0 -> 0, as SDPA
    return w @ v, w
    ### END SOLUTION


ref = F.scaled_dot_product_attention
check_close("sdpa no mask", sdpa(q, k, v)[0], ref(q, k, v), atol=1e-10)
check_close("sdpa scale=0.3", sdpa(q, k, v, scale=0.3)[0], ref(q, k, v, scale=0.3), atol=1e-10)
check_close("sdpa causal L=S", sdpa(q, q, q, is_causal=True)[0], ref(q, q, q, is_causal=True), atol=1e-10)
check_close("sdpa causal L<S (top-left)", sdpa(q, k, v, is_causal=True)[0], ref(q, k, v, is_causal=True), atol=1e-10)
bmask = torch.rand(B, 1, L, S) > 0.5
bmask[..., 0] = True                                              # every row keeps >= 1 key
check_close("sdpa bool mask (broadcast over heads)", sdpa(q, k, v, attn_mask=bmask)[0], ref(q, k, v, attn_mask=bmask), atol=1e-10)
fmask = torch.randn(L, S)
check_close("sdpa additive float mask", sdpa(q, k, v, attn_mask=fmask)[0], ref(q, k, v, attn_mask=fmask), atol=1e-10)
bmask[0, 0, 2] = False                                            # one fully masked row
out_fm, w_fm = sdpa(q, k, v, attn_mask=bmask)
check("sdpa fully masked row -> zeros, no NaN", not out_fm.isnan().any() and out_fm[0, :, 2].abs().max() == 0)
check_close("sdpa fully masked row matches SDPA", out_fm, ref(q, k, v, attn_mask=bmask), atol=1e-10)
check_close("weights: rows sum to 1 (0 if fully masked)", w_fm.sum(-1), (bmask.expand(B, H, L, S).any(-1)).double())

# %% [markdown]
# ## 2. Mask helpers (and the top-left vs bottom-right trap)
# With a KV cache (A3.13) or chunked prefill, the $L$ new queries are the **last** $L$ of $S$ positions, so query $i$
# must see keys $j \le i + (S-L)$: a **bottom-right** aligned causal mask. SDPA's `is_causal=True` is **top-left**
# aligned, so `is_causal=True` with one new query lets it see only key 0. This bug produces fluent garbage, not an error.
# `torch.nn.attention.bias.causal_lower_right(L, S)` is the built-in bottom-right mask.
#
# Padding: `nn.MultiheadAttention` takes `key_padding_mask` of shape (B, S) with **True = padding (ignore)**, the
# opposite of SDPA's bool `attn_mask` (True = attend). To feed SDPA you invert it and reshape to (B, 1, 1, S).

# %%
def causal_mask(L, S, align="top_left"):
    """Bool (L, S), True = may attend. top_left: j <= i. bottom_right: j <= i + (S - L)."""
    ### BEGIN SOLUTION
    offset = 0 if align == "top_left" else S - L
    return torch.ones(L, S, dtype=torch.bool).tril(diagonal=offset)
    ### END SOLUTION


def key_padding_to_attn_mask(key_padding_mask):
    """(B, S) bool, True = PAD (nn.MultiheadAttention convention) -> (B, 1, 1, S) bool, True = attend (SDPA)."""
    ### BEGIN SOLUTION
    return (~key_padding_mask)[:, None, None, :]
    ### END SOLUTION


def bool_to_additive(mask, dtype=torch.float64):
    """True -> 0.0, False -> -inf, same shape."""
    ### BEGIN SOLUTION
    return torch.zeros(mask.shape, dtype=dtype).masked_fill(~mask, float("-inf"))
    ### END SOLUTION


# reference: SDPA called with PyTorch's own CausalBias objects (public API: pass them as attn_mask)
for (l_, s_) in ((5, 5), (3, 7), (1, 6)):
    qc, kc = torch.randn(B, H, l_, E), torch.randn(B, H, s_, E)
    for align, bias in (("top_left", causal_upper_left), ("bottom_right", causal_lower_right)):
        cm = causal_mask(l_, s_, align)
        check(f"causal {align} {l_}x{s_}: bool (L, S)", cm.dtype == torch.bool and cm.shape == (l_, s_))
        check_close(f"causal {align} {l_}x{s_} vs torch.nn.attention.bias", ref(qc, kc, kc, attn_mask=cm),
                    ref(qc, kc, kc, attn_mask=bias(l_, s_)), atol=1e-10)
# the decode case: the last 2 rows of full causal attention == 2 new queries attending to all 7 keys, bottom-right
qq = torch.randn(B, H, S, E)
full = ref(qq, qq, qq, is_causal=True)
check_close("bottom-right mask reproduces the last rows of full causal attention",
            sdpa(qq[:, :, -2:], qq, qq, attn_mask=causal_mask(2, S, "bottom_right"))[0], full[:, :, -2:], atol=1e-10)
check("...and is_causal=True (top-left) does NOT",
      not torch.allclose(ref(qq[:, :, -2:], qq, qq, is_causal=True), full[:, :, -2:]))
kpm = torch.zeros(B, S, dtype=torch.bool); kpm[0, 5:] = True; kpm[1, 3:] = True
check_close("key padding -> SDPA mask", sdpa(q, k, v, attn_mask=key_padding_to_attn_mask(kpm))[0],
            ref(q, k, v, attn_mask=(~kpm)[:, None, None, :]), atol=1e-10)
check_close("bool and additive masks agree", sdpa(q, k, v, attn_mask=bool_to_additive(bmask))[0], sdpa(q, k, v, attn_mask=bmask)[0], atol=1e-10)

# %% [markdown]
# ## 3. Multi-head attention, checked against `nn.MultiheadAttention`
# $\mathrm{MHA}(X_q, X_k, X_v) = \mathrm{Concat}(\text{head}_1..\text{head}_h)W^O$, $\text{head}_i = \mathrm{Attn}(X_qW_i^Q, X_kW_i^K, X_vW_i^V)$,
# $d_{head} = d_{model}/h$. In code: one `Linear(d_model, d_model)` per Q/K/V, then
# `view(B, T, H, Dh).transpose(1, 2)` → (B, H, T, Dh), attention per head, `transpose(1, 2).reshape(B, T, d_model)`, `out_proj`.
#
# `nn.MultiheadAttention` stores the three input projections **stacked** in `in_proj_weight` of shape
# (3·d_model, d_model) in the order **[W_q; W_k; W_v]** (and `in_proj_bias` (3·d_model,)) when key/value width equals
# `embed_dim`; with `kdim`/`vdim` ≠ `embed_dim` it uses separate `q_proj_weight`, `k_proj_weight`, `v_proj_weight` and
# still one stacked `in_proj_bias`. `nn.Linear` computes $xW^\top + b$, so these weights copy over *without transposing*.
# Use the attribute names `q_proj`, `k_proj`, `v_proj`, `out_proj` (the copier and later sections rely on them).

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kdim=None, vdim=None, bias=True):
        super().__init__()
        ### BEGIN SOLUTION
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(kdim or d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(vdim or d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        ### END SOLUTION

    def forward(self, query, key, value, key_padding_mask=None, is_causal=False):
        """query (B, L, d_model), key (B, S, kdim), value (B, S, vdim); key_padding_mask (B, S) True = pad.
        Returns (out (B, L, d_model), per-head attention weights (B, H, L, S))."""
        ### BEGIN SOLUTION
        Bq, Lq, D = query.shape
        Sk = key.shape[1]
        Hh, Dh = self.n_heads, self.d_head
        q = self.q_proj(query).view(Bq, Lq, Hh, Dh).transpose(1, 2)    # (B, H, L, Dh)
        k = self.k_proj(key).view(Bq, Sk, Hh, Dh).transpose(1, 2)      # (B, H, S, Dh)
        v = self.v_proj(value).view(Bq, Sk, Hh, Dh).transpose(1, 2)
        mask = None if key_padding_mask is None else key_padding_to_attn_mask(key_padding_mask)
        y, w = sdpa(q, k, v, attn_mask=mask, is_causal=is_causal)
        y = y.transpose(1, 2).reshape(Bq, Lq, D)                       # concat heads
        return self.out_proj(y), w
        ### END SOLUTION


def copy_from_torch(mine, ref_mha):
    """Copy all weights of an nn.MultiheadAttention into your MultiHeadAttention (both layouts)."""
    with torch.no_grad():
        ### BEGIN SOLUTION
        if ref_mha.in_proj_weight is not None:                        # packed [W_q; W_k; W_v]
            wq, wk, wv = ref_mha.in_proj_weight.chunk(3, dim=0)
        else:                                                         # kdim/vdim != embed_dim
            wq, wk, wv = ref_mha.q_proj_weight, ref_mha.k_proj_weight, ref_mha.v_proj_weight
        for lin, w in ((mine.q_proj, wq), (mine.k_proj, wk), (mine.v_proj, wv)):
            lin.weight.copy_(w)
        if ref_mha.in_proj_bias is not None:
            bq, bk, bv = ref_mha.in_proj_bias.chunk(3)
            for lin, b in ((mine.q_proj, bq), (mine.k_proj, bk), (mine.v_proj, bv)):
                lin.bias.copy_(b)
        mine.out_proj.weight.copy_(ref_mha.out_proj.weight)
        if ref_mha.out_proj.bias is not None:
            mine.out_proj.bias.copy_(ref_mha.out_proj.bias)
        ### END SOLUTION


D_MODEL, N_HEADS, T = 32, 4, 9
torch_mha = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
nn.init.normal_(torch_mha.in_proj_bias); nn.init.normal_(torch_mha.out_proj.bias)  # nonzero biases, so a mix-up shows
my_mha = MultiHeadAttention(D_MODEL, N_HEADS)
copy_from_torch(my_mha, torch_mha)
x = torch.randn(B, T, D_MODEL)
with torch.no_grad():
    o_ref, w_ref = torch_mha(x, x, x, need_weights=True, average_attn_weights=False)
    o_me, w_me = my_mha(x, x, x)
    check_close("MHA self-attention output", o_me, o_ref, atol=1e-10)
    check_close("MHA per-head weights (B, H, L, S)", w_me, w_ref, atol=1e-10)
    # causal: nn.MultiheadAttention's bool attn_mask means True = NOT allowed, so pass the complement
    o_ref = torch_mha(x, x, x, attn_mask=~causal_mask(T, T), need_weights=False)[0]
    check_close("MHA causal", my_mha(x, x, x, is_causal=True)[0], o_ref, atol=1e-10)
    kpm9 = torch.zeros(B, T, dtype=torch.bool); kpm9[1, 6:] = True
    o_ref = torch_mha(x, x, x, key_padding_mask=kpm9, need_weights=False)[0]
    check_close("MHA key_padding_mask", my_mha(x, x, x, key_padding_mask=kpm9)[0], o_ref, atol=1e-10)
    o_ref = torch_mha(x, x, x, key_padding_mask=kpm9, attn_mask=~causal_mask(T, T), need_weights=False)[0]
    check_close("MHA causal + padding", my_mha(x, x, x, key_padding_mask=kpm9, is_causal=True)[0], o_ref, atol=1e-10)
check("MHA parameter count = 4 d^2 + 4 d", sum(p.numel() for p in my_mha.parameters()) == 4 * D_MODEL**2 + 4 * D_MODEL)

# %% [markdown]
# ## 4. Cross-attention: queries from one sequence, keys/values from another
# Encoder-decoder attention (Vaswani §3.2.3), and the image-to-text attention of Flamingo-style VLMs: queries come from the
# decoder (length $L$), keys and values from the encoder memory (length $S$, width `kdim`). No causal mask (the decoder may
# look at the whole source), but a padding mask over the *source* keys. The weights are (B, H, L, S).
# Your module from §3 should already handle this; this cell checks it, including the separate-projection weight layout.

# %%
L_DEC, S_ENC, D_ENC = 6, 11, 24
torch_x = nn.MultiheadAttention(D_MODEL, N_HEADS, kdim=D_ENC, vdim=D_ENC, batch_first=True)
nn.init.normal_(torch_x.in_proj_bias); nn.init.normal_(torch_x.out_proj.bias)
my_x = MultiHeadAttention(D_MODEL, N_HEADS, kdim=D_ENC, vdim=D_ENC)
copy_from_torch(my_x, torch_x)
dec, mem = torch.randn(B, L_DEC, D_MODEL), torch.randn(B, S_ENC, D_ENC)
src_pad = torch.zeros(B, S_ENC, dtype=torch.bool); src_pad[0, 8:] = True
with torch.no_grad():
    o_ref, w_ref = torch_x(dec, mem, mem, key_padding_mask=src_pad, average_attn_weights=False)
    o_me, w_me = my_x(dec, mem, mem, key_padding_mask=src_pad)
check("cross-attn weights shape (B, H, L, S)", w_me.shape == (B, N_HEADS, L_DEC, S_ENC), str(w_me.shape))
check_close("cross-attn output vs nn.MultiheadAttention(kdim, vdim)", o_me, o_ref, atol=1e-10)
check_close("cross-attn weights", w_me, w_ref, atol=1e-10)
check("padded source keys get zero weight", w_me[0, :, :, 8:].abs().max().item() == 0)

# %% [markdown]
# ## 5. Grouped-query attention (GQA) and multi-query attention (MQA)
# $H$ query heads share $H_{kv}$ key/value heads ($H_{kv} \mid H$, group size $g = H/H_{kv}$); MQA is $H_{kv}=1$ (Shazeer 2019),
# MHA is $H_{kv}=H$. Query head $h$ uses KV head $\lfloor h/g \rfloor$, which is what `repeat_interleave(g, dim=1)` produces
# (KV heads `[a, b]` → `[a, a, b, b]`). `repeat` would give `[a, b, a, b]`: same shapes, wrong heads, so a checkpoint
# trained one way gives garbage the other way. SDPA's `enable_gqa=True` (present in this torch 2.14) does the `repeat_interleave`
# internally (see its documented reference implementation); it is the reference here.
# What GQA buys: K/V projections shrink from $2d^2$ to $2d\cdot H_{kv}d_{head}$ parameters, and, more importantly, the
# **KV cache** shrinks by $H/H_{kv}$ (A3.13).

# %%
class GroupedQueryAttention(nn.Module):
    """Causal self-attention with n_kv_heads K/V heads. No biases. Attributes: q_proj (d -> d),
    k_proj and v_proj (d -> n_kv_heads * d_head), out_proj (d -> d)."""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        ### BEGIN SOLUTION
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads, self.d_head = n_heads, n_kv_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        ### END SOLUTION

    def forward(self, x):
        """x (B, T, d_model) -> (B, T, d_model), causal."""
        ### BEGIN SOLUTION
        Bx, Tx, D = x.shape
        q = self.q_proj(x).view(Bx, Tx, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(Bx, Tx, self.n_kv_heads, self.d_head).transpose(1, 2)   # (B, Hkv, T, Dh)
        v = self.v_proj(x).view(Bx, Tx, self.n_kv_heads, self.d_head).transpose(1, 2)
        g = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)               # (B, H, T, Dh)
        y, _ = sdpa(q, k, v, is_causal=True)
        return self.out_proj(y.transpose(1, 2).reshape(Bx, Tx, D))
        ### END SOLUTION


def gqa_reference(mod, x):
    """Same projections, attention by SDPA with enable_gqa=True (K/V passed with only H_kv heads)."""
    Bx, Tx, D = x.shape
    Hh, Hkv, Dh = mod.n_heads, mod.n_kv_heads, D // mod.n_heads
    qr = mod.q_proj(x).view(Bx, Tx, Hh, Dh).transpose(1, 2)
    kr = mod.k_proj(x).view(Bx, Tx, Hkv, Dh).transpose(1, 2)
    vr = mod.v_proj(x).view(Bx, Tx, Hkv, Dh).transpose(1, 2)
    y = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True, enable_gqa=True)
    return mod.out_proj(y.transpose(1, 2).reshape(Bx, Tx, D))


with torch.no_grad():
    for hkv in (8, 4, 2, 1):
        gqa = GroupedQueryAttention(D_MODEL, 8, hkv)
        check_close(f"GQA H=8, H_kv={hkv} vs SDPA(enable_gqa=True)", gqa(x), gqa_reference(gqa, x), atol=1e-10)
        n_par = sum(p.numel() for p in gqa.parameters())
        check(f"GQA H_kv={hkv} params = 2d^2 + 2d*H_kv*d_head", n_par == 2 * D_MODEL**2 + 2 * D_MODEL * hkv * (D_MODEL // 8))
    # H_kv = H is plain MHA: copy into the §3 module and compare
    gqa = GroupedQueryAttention(D_MODEL, N_HEADS, N_HEADS)
    mha_nb = MultiHeadAttention(D_MODEL, N_HEADS, bias=False)
    mha_nb.load_state_dict(gqa.state_dict())
    check_close("GQA with H_kv = H equals MHA (causal)", gqa(x), mha_nb(x, x, x, is_causal=True)[0], atol=1e-10)

# %% [markdown]
# ## 6. Cost: FLOPs and memory of one attention call
# Count a multiply-add as 2 FLOPs. $QK^\top$ is a (L×d)(d×S) matmul per head: $2LSd$ FLOPs; $PV$ is (L×S)(S×d): another
# $2LSd$. So attention costs $4BHLSd_{head} = 4BLSd_{model}$ FLOPs, **quadratic in sequence length**, while the four
# projections cost $8BTd_{model}^2$ (self-attention, L=S=T): attention dominates once $T > 2d_{model}$.
# Naive attention also **materializes** the (B, H, L, S) score and probability tensors: $O(BHLS)$ memory, which is what
# FlashAttention removes (its extra memory is $O(N)$, FLOPs unchanged; Dao et al. Theorem 1).

# %%
def attention_flops(B, H, L, S, d_head):
    """Matmul FLOPs of QK^T and PV for one call (multiply-add = 2 FLOPs). Ignore scale/mask/softmax."""
    ### BEGIN SOLUTION
    return 2 * B * H * L * S * d_head * 2
    ### END SOLUTION


def score_matrix_bytes(B, H, L, S, bytes_per_el):
    """Bytes of ONE materialized (B, H, L, S) score tensor."""
    ### BEGIN SOLUTION
    return B * H * L * S * bytes_per_el
    ### END SOLUTION


for shp in ((2, 3, 5, 7, 8), (1, 4, 16, 16, 32), (3, 2, 1, 50, 64)):
    qf, kf, vf = torch.randn(*shp[:3], shp[4]), torch.randn(shp[0], shp[1], shp[3], shp[4]), torch.randn(shp[0], shp[1], shp[3], shp[4])
    with FlopCounterMode(display=False) as fc:
        sdpa(qf, kf, vf)
    check(f"attention_flops{shp} == FlopCounterMode on your sdpa", attention_flops(*shp) == fc.get_total_flops(),
          f"{attention_flops(*shp)} vs counted {fc.get_total_flops()}")
check("score bytes: 1x32x4096x4096 fp16 = 1 GiB", score_matrix_bytes(1, 32, 4096, 4096, 2) == 2**30)


# %% [markdown]
# Measured on the GPU (skipped on CPU): peak memory of your naive `sdpa` vs `F.scaled_dot_product_attention`
# (FlashAttention / memory-efficient kernel on this RTX 3090) at growing T. The naive peak should be at least one score
# matrix above the inputs and grow ~4× per doubling of T; the fused kernel's should grow ~2× (inputs and output only).

# %%
if torch.cuda.is_available():
    dev, Hm, Dm = torch.device("cuda"), 8, 64
    rows = []
    for Tm in (1024, 2048, 4096):
        qm, km, vm = (torch.randn(1, Hm, Tm, Dm, device=dev, dtype=torch.float16) for _ in range(3))
        peaks = []
        for fn in (lambda: sdpa(qm, km, vm, is_causal=True)[0], lambda: F.scaled_dot_product_attention(qm, km, vm, is_causal=True)):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
            out_m = fn(); torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - base); del out_m
        rows.append((Tm, *peaks))
        print(f"T={Tm:5d}  naive peak {peaks[0] / 2**20:8.1f} MiB   fused SDPA peak {peaks[1] / 2**20:6.1f} MiB   "
              f"one score matrix {score_matrix_bytes(1, Hm, Tm, Tm, 2) / 2**20:7.1f} MiB")
        del qm, km, vm
    Tm, naive, fused = rows[-1]
    check("naive peak >= one fp16 score matrix", naive >= score_matrix_bytes(1, Hm, Tm, Tm, 2))
    check("fused SDPA peak < 1/8 of one score matrix", fused < score_matrix_bytes(1, Hm, Tm, Tm, 2) / 8)
    check("naive peak grows ~4x per doubling of T", rows[-1][1] / rows[-2][1] > 3.0, f"{rows[-1][1] / rows[-2][1]:.2f}")
    torch.cuda.empty_cache()
else:
    print("no GPU: skipping the memory measurement")

# %% [markdown]
# ## Say it out loud
# 1. Why divide by $\sqrt{d_k}$? What goes wrong without it, and what would dividing by $d_k$ do?
# 2. Time and memory complexity of self-attention in $T$ and $d$, for training and for one decode step with a KV cache.
#    At what sequence length does attention cost more FLOPs than the Q/K/V/O projections?
# 3. What does FlashAttention change about attention's cost, and what does it not change?
# 4. State the mask conventions of `F.scaled_dot_product_attention` (bool and float), `nn.MultiheadAttention`'s `attn_mask`
#    and `key_padding_mask`. What happens to a fully masked query row in each, and where do such rows appear in practice?
# 5. Why does `is_causal=True` give wrong results when you pass 1 new query and $S$ cached keys?
# 6. Why multiple heads instead of one head of width $d_{model}$, at the same parameter count and FLOPs?
# 7. MQA vs GQA vs MHA: what each saves, what it costs, and why `repeat_interleave` rather than `repeat`. How do you turn an
#    MHA checkpoint into a GQA one?
# 8. Self- vs cross-attention: which shapes change, which masks apply, and what can be cached across decode steps?
# 9. Without positional information, what symmetry does self-attention have? (Permute the input tokens: what happens to the output?)

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. If the entries of $q$ and $k$ are independent with mean 0 and variance 1, $q\cdot k$ has variance $d_k$ (Vaswani et al.,
#    footnote 4). Large logits saturate the softmax into a near one-hot whose gradient is ≈ 0 for everything but the max, so
#    training stalls. Dividing by $\sqrt{d_k}$ restores unit variance; dividing by $d_k$ would shrink logits to variance
#    $1/d_k$, making attention nearly uniform at init (it is only a constant rescale, so learnable, but a worse starting point).
# 2. Training/prefill: $QK^\top$ and $PV$ cost $4BT^2d$ FLOPs ($O(T^2 d)$) and naive memory $O(BHT^2)$ for scores; projections
#    cost $8BTd^2$. Attention FLOPs exceed projection FLOPs when $4T^2d > 8Td^2$, i.e. $T > 2d$ (e.g. $T > 8192$ for
#    $d=4096$; the FFN adds another $16Td^2$, moving the crossover for the whole layer to $T > 6d$). One decode step with a cache:
#    $O(Td)$ FLOPs and it must *read* the whole $O(T\cdot H_{kv}d_{head})$ KV cache, so it is memory-bandwidth bound.
# 3. FlashAttention is exact attention computed in SRAM-sized tiles with an online softmax, so the $T\times T$ matrix is never
#    written to HBM: extra memory $O(T)$ instead of $O(T^2)$, and far fewer HBM accesses ($\Theta(T^2d^2/M)$ vs
#    $\Theta(Td + T^2)$, Dao et al. Theorem 2). FLOPs stay $O(T^2d)$ (slightly more, since the backward recomputes the scores
#    instead of storing them). It is a speedup because attention is memory-bound, not because it does less arithmetic.
# 4. SDPA bool: True = attend. SDPA float: added to the scores. `nn.MultiheadAttention` bool `attn_mask`: True = **not
#    allowed**; `key_padding_mask` (B, S): True = **pad, ignore**. Fully masked row: SDPA in torch 2.14 returns 0 (measured
#    above), `nn.MultiheadAttention` and a naive softmax give NaN, which then poisons the loss and every gradient. They appear
#    for padding *queries* (e.g. left-padded batched generation, where a pad position may see no valid key) and in
#    packed-sequence masks. Fixes: let pad queries attend to themselves, or zero their outputs and exclude them from the loss.
# 5. SDPA aligns the causal mask to the top-left corner: query $i$ sees keys $\le i$. A single new query is row 0, so it sees
#    only key 0. With a cache the new queries are the last $L$ positions and need the bottom-right alignment
#    $j \le i + (S - L)$ (`causal_lower_right`, or no mask at all when $L = 1$).
# 6. Each head gets its own attention pattern, so the layer can attend to several positions for different reasons (syntax,
#    copying, position) in one layer. A single wide head produces one convex combination of values per query. Cost is the same
#    because $H \cdot d_{head} = d_{model}$ (Vaswani §3.2.2).
# 7. MHA: $H$ K/V heads. MQA: one shared K/V head, cutting the KV cache and per-step memory traffic by $H\times$, with some
#    quality loss and fine-tuning instability (Ainslie et al. §1 and App. A). GQA: $H_{kv}$ groups, a middle ground
#    with near-MHA quality and near-MQA speed; the cache shrinks $H/H_{kv}\times$. Query head $h$ must read KV head $\lfloor h/g\rfloor$, which is what
#    `repeat_interleave` produces and what checkpoints expect; `repeat` silently pairs the wrong heads. Converting MHA→GQA:
#    mean-pool the K and V projection heads within each group, then *uptrain* for ~5% of the pre-training compute
#    (Ainslie et al. §2.1).
# 8. Self: Q, K, V all from $X$ ($L = S$), causal mask in a decoder. Cross: Q from the decoder ($L$), K/V from the encoder
#    or image tokens ($S$, maybe a different width `kdim`), weights $(B,H,L,S)$, only a source padding mask. In decoding,
#    the cross-attention K/V of the source are computed once and reused at every step; self-attention K/V grow by one per step
#    (the KV cache).
# 9. Permutation equivariance: permuting the input tokens permutes the outputs the same way (with no causal mask). That is why
#    transformers need positional encodings (sinusoidal, learned, RoPE, ALiBi). A causal mask breaks the symmetry partially,
#    which is why decoder-only models without positional encodings (NoPE) can still learn positions (Haviv et al. 2022,
#    arXiv:2203.16634).
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
