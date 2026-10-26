"""A small, standard GPT-2-style decoder used as a *workload* by the G1 systems labs.

Not a drill: you write attention / KV cache yourself in A3.10 and A3.13. This file exists so
the memory, FLOP, DDP, FSDP and torch.compile labs all measure the same well-understood model.

Architecture follows GPT-2 (Radford et al. 2019) as re-implemented in
  - S. Raschka, LLMs-from-scratch, ch04/01_main-chapter-code/gpt.py (Apache-2.0),
    https://github.com/rasbt/LLMs-from-scratch (local copy: ../LLMs-from-scratch, commit f9313ea)
  - A. Karpathy, nanoGPT model.py (MIT), https://github.com/karpathy/nanoGPT/blob/master/model.py
Differences from Raschka's version: fused QKV projection, `F.scaled_dot_product_attention`
(switchable to an explicit softmax for the memory labs), weight tying, no dropout by default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 256
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.0
    tie_weights: bool = True
    sdpa: bool = True  # False: materialize the (T, T) attention matrix explicitly


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H = self.cfg.n_heads
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in (q, k, v))  # (B, H, T, Dh)
        if self.cfg.sdpa:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.cfg.dropout if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(D // H)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            att = att.masked_fill(mask, float("-inf")).softmax(-1)
            y = att @ v
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * cfg.d_model, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))  # pre-norm residual blocks
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def num_params(model: nn.Module, non_embedding: bool = False) -> int:
    """Unique parameters (tied weights counted once). non_embedding drops token+position tables,
    the convention used for N in Kaplan et al. 2020 scaling laws."""
    n = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    if non_embedding and isinstance(model, GPT):
        n -= model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
    return n


def random_batch(cfg: GPTConfig, batch_size: int, device="cpu", generator=None):
    """Random token ids: fine for systems measurements (FLOPs/memory/time do not depend on content)."""
    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.context_length + 1), device=device, generator=generator)
    return x[:, :-1], x[:, 1:]
