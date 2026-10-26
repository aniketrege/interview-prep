# %% [markdown]
# # A3.15 · Contrastive losses: InfoNCE, CLIP, SigLIP
#
# **ID:** A3.15 · **Target:** 15 min · **Plan week:** 10 · **Part:** A
#
# **The drill.** Guide A3 lists "Contrastive (InfoNCE) loss" and week 9 builds "CLIP/SigLIP loss". Implement, in a
# few lines each and batched (no Python loops):
#
# 1. **InfoNCE** with in-batch negatives and a fixed temperature,
# 2. the **CLIP** symmetric loss with a learnable log-scale $t$ (scale $e^t$),
# 3. the **SigLIP** pairwise sigmoid loss with learnable $t'$ and bias $b$,
# 4. the papers' **initial values** for those parameters, and CLIP's clamp.
#
# **Allowed:** `F.normalize`, `F.cross_entropy`, `F.logsigmoid`, matmul, `torch.eye`, `torch.arange`.
# The checks compare against independent double-loop implementations (plain `math`, float64), against
# `F.binary_cross_entropy_with_logits`, and against analytic limits (τ → ∞, τ → 0, perfectly aligned pairs, SigLIP at init).
#
# **Deeper study:** the CLIP paper's Figure 3 pseudo-code and SigLIP's Algorithm 1 are ten lines each; read both.
# For the distributed versions (all-gather with gradients for CLIP; SigLIP's chunked "pass the texts around the ring"
# loss) read `ClipLoss` and `SigLipLoss` in open_clip's `src/open_clip/loss.py` (link below).
#
# ## Sources
# - A. van den Oord, Y. Li, O. Vinyals, *Representation Learning with Contrastive Predictive Coding*, 2018,
#   arXiv:1807.03748 — https://arxiv.org/abs/1807.03748 (InfoNCE, Eq. 4; the bound $I(x;c) \ge \log N - \mathcal{L}_N$, §2.3).
# - A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, et al., *Learning Transferable Visual Models From Natural Language
#   Supervision* (CLIP), ICML 2021, arXiv:2103.00020 — https://arxiv.org/abs/2103.00020 (Fig. 3 pseudo-code; §2.5:
#   temperature "initialized to the equivalent of 0.07 ... and clipped to prevent scaling the logits by more than 100").
# - X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer, *Sigmoid Loss for Language Image Pre-Training* (SigLIP), ICCV 2023,
#   arXiv:2303.15343 — https://arxiv.org/abs/2303.15343 (§3.2 and Algorithm 1: labels $2I - 1$, loss summed over all
#   pairs and divided by the batch size $n$, $t = e^{t'}$, "We initialize $t'$ and $b$ to $\log 10$ and $-10$").
# - google-research/big_vision @ 0127fb6, `big_vision/trainers/proj/image_text/siglip.py` (`logits * t + b`,
#   `log_sigmoid(m1_diag1 * logits)`, sum over the row then mean), Apache-2.0 —
#   https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/trainers/proj/image_text/siglip.py
# - mlfoundations/open_clip @ 7067b35: `src/open_clip/model.py` (`init_logit_scale = np.log(1 / 0.07)`),
#   `src/open_clip_train/train.py` ("we clamp to 4.6052 = ln(100), as in the original paper"), `src/open_clip/loss.py`
#   (`ClipLoss`, `SigLipLoss`), MIT — https://github.com/mlfoundations/open_clip/tree/7067b350050dc409bfdfdbb3f0cc6c9a1651c000
# - T. Chen, S. Kornblith, M. Norouzi, G. Hinton, *A Simple Framework for Contrastive Learning of Visual
#   Representations* (SimCLR, NT-Xent), ICML 2020, arXiv:2002.05709 — https://arxiv.org/abs/2002.05709

# %%
import math

import torch
import torch.nn.functional as F

from prep import Session, check, check_close

s = Session("A3.15", target_min=15)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)        # analytic limits below need the precision
N, D = 8, 16
img = torch.randn(N, D) * 3                    # unnormalized encoder outputs
txt = img + 0.8 * torch.randn(N, D) * 3        # correlated "captions"


def _loop_cos(a, b):
    """Independent cosine-similarity matrix with Python loops (test reference)."""
    na = [math.sqrt(sum(x * x for x in r)) for r in a.tolist()]
    nb = [math.sqrt(sum(x * x for x in r)) for r in b.tolist()]
    return [[sum(x * y for x, y in zip(ra, rb)) / (na[i] * nb[j]) for j, rb in enumerate(b.tolist())]
            for i, ra in enumerate(a.tolist())]


def _loop_info_nce(q, k, tau):
    S = _loop_cos(q, k)
    out = 0.0
    for i in range(len(S)):
        m = max(S[i]) / tau
        lse = m + math.log(sum(math.exp(sij / tau - m) for sij in S[i]))
        out += lse - S[i][i] / tau
    return out / len(S)

# %% [markdown]
# ## 1. InfoNCE (in-batch negatives)
# Pairs $(q_i, k_i)$ are positives and every other $k_j$ in the batch is a negative. With cosine similarities
# $s_{ij} = \hat q_i^\top \hat k_j$:
# $$\mathcal{L} = -\frac1N\sum_i \log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}},$$
# i.e. an $N$-way cross-entropy whose "class" for row $i$ is $i$.

# %%
def info_nce(q, k, temperature=0.1):
    """q, k: (N, D), not normalized. Return the scalar mean InfoNCE loss with in-batch negatives."""
    ### BEGIN SOLUTION
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    logits = q @ k.T / temperature
    return F.cross_entropy(logits, torch.arange(len(q), device=q.device))
    ### END SOLUTION


for tau in (0.07, 0.5, 2.0):
    check_close(f"InfoNCE vs loop, tau={tau}", info_nce(img, txt, tau), _loop_info_nce(img, txt, tau), atol=1e-8, rtol=1e-8)
check_close("InfoNCE is scale-invariant in q and k (cosine)", info_nce(5 * img, 0.1 * txt, 0.1), info_nce(img, txt, 0.1))
check_close("tau -> inf: loss -> log N", info_nce(img, txt, 1e8), math.log(N), atol=1e-6)
S = torch.tensor(_loop_cos(img, txt))
hinge = (S.masked_fill(torch.eye(N, dtype=torch.bool), -math.inf).amax(1) - S.diag()).clamp(min=0).mean()
check_close("tau -> 0: tau * loss -> mean_i relu(max_{j!=i} s_ij - s_ii)", 1e-4 * info_nce(img, txt, 1e-4), hinge, atol=1e-3)
eye = torch.eye(N, D)                                                     # orthonormal, perfectly aligned
check_close("aligned orthonormal pairs: log(1 + (N-1) e^{-1/tau})", info_nce(eye, eye, 0.1),
            math.log(1 + (N - 1) * math.exp(-1 / 0.1)), atol=1e-8)
perm = torch.randperm(N)
check_close("permuting pairs jointly leaves loss unchanged", info_nce(img[perm], txt[perm], 0.1), info_nce(img, txt, 0.1))
check("shuffling only the keys raises the loss", info_nce(img, txt[perm], 0.1) > info_nce(img, txt, 0.1))

# %% [markdown]
# ## 2. CLIP symmetric loss
# CLIP (Fig. 3): `logits = (I_e @ T_e.T) * exp(t)` on L2-normalized embeddings, then the average of the cross-entropy
# over images→texts (rows) and texts→images (columns). $t$ is the learnable **log** of the logit scale, so
# $\tau = e^{-t}$. Swapping `img` and `txt` must give the same loss.

# %%
def clip_loss(img_emb, txt_emb, t):
    """img_emb, txt_emb: (N, D), not normalized. t: scalar tensor, log logit-scale. Return the scalar loss."""
    ### BEGIN SOLUTION
    x, y = F.normalize(img_emb, dim=-1), F.normalize(txt_emb, dim=-1)
    logits = t.exp() * x @ y.T                                   # (N_img, N_txt)
    labels = torch.arange(len(x), device=x.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    ### END SOLUTION


t0 = torch.tensor(math.log(1 / 0.07))
ref_clip = 0.5 * (_loop_info_nce(img, txt, 0.07) + _loop_info_nce(txt, img, 0.07))
check("the two directions really differ on this data", abs(_loop_info_nce(img, txt, 0.07) - _loop_info_nce(txt, img, 0.07)) > 1e-3)
check_close("CLIP vs loop (mean of both directions)", clip_loss(img, txt, t0), ref_clip, atol=1e-8, rtol=1e-8)
check_close("CLIP symmetric in (img, txt)", clip_loss(txt, img, t0), clip_loss(img, txt, t0))
check_close("CLIP = average of two InfoNCEs at tau = exp(-t)", clip_loss(img, txt, t0),
            0.5 * (info_nce(img, txt, 0.07) + info_nce(txt, img, 0.07)))
t_req = t0.clone().requires_grad_()
clip_loss(img, txt, t_req).backward()
check("gradient flows to the learnable temperature", t_req.grad is not None and t_req.grad.abs() > 0)

# %% [markdown]
# ## 3. SigLIP pairwise sigmoid loss
# Every image-text pair is an independent binary classification: label $z_{ij} = +1$ on the diagonal and $-1$ elsewhere.
# Following Algorithm 1 (and big_vision's `siglip.py`):
# $$\text{logits}_{ij} = e^{t'}\,\hat x_i^\top \hat y_j + b,\qquad
# \mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{n}\log\sigma\!\big(z_{ij}\,\text{logits}_{ij}\big).$$
# Note the normalizer: **divide by the batch size $n$**, not by the $n^2$ pairs. No softmax, so no row/column
# normalization over the batch.

# %%
def siglip_loss(img_emb, txt_emb, t_prime, b):
    """img_emb, txt_emb: (N, D), not normalized. t_prime, b: scalar tensors. Return the scalar loss."""
    ### BEGIN SOLUTION
    x, y = F.normalize(img_emb, dim=-1), F.normalize(txt_emb, dim=-1)
    logits = t_prime.exp() * x @ y.T + b
    labels = 2 * torch.eye(len(x), device=x.device, dtype=logits.dtype) - 1
    return -F.logsigmoid(labels * logits).sum() / len(x)
    ### END SOLUTION


def _loop_siglip(a, c, tp, bias):
    S, out = _loop_cos(a, c), 0.0
    for i in range(len(S)):
        for j in range(len(S)):
            z = 1.0 if i == j else -1.0
            u = z * (math.exp(tp) * S[i][j] + bias)
            out += math.log1p(math.exp(-u)) if u > 0 else -u + math.log1p(math.exp(u))  # -log sigmoid(u), stable
    return out / len(S)


for tp, bb in ((math.log(10), -10.0), (math.log(3), 0.5), (2.5, -4.0)):
    tp_, bb_ = torch.tensor(tp), torch.tensor(bb)
    check_close(f"SigLIP vs loop (t'={tp:.2f}, b={bb})", siglip_loss(img, txt, tp_, bb_), _loop_siglip(img, txt, tp, bb),
                atol=1e-8, rtol=1e-8)
logits_ref = math.exp(2.5) * S + -4.0
check_close("SigLIP = BCE-with-logits(sum) / n", siglip_loss(img, txt, torch.tensor(2.5), torch.tensor(-4.0)),
            F.binary_cross_entropy_with_logits(logits_ref, torch.eye(N), reduction="sum") / N)
check_close("SigLIP symmetric in (img, txt)", siglip_loss(txt, img, torch.tensor(2.5), torch.tensor(-4.0)),
            siglip_loss(img, txt, torch.tensor(2.5), torch.tensor(-4.0)))
sp = lambda u: math.log1p(math.exp(u))                               # softplus; -log sigmoid(u) = softplus(-u)
check_close("at init (t'=log 10, b=-10), aligned orthonormal pairs: log 2 + (n-1) softplus(-10)",
            siglip_loss(eye, eye, torch.tensor(math.log(10)), torch.tensor(-10.0)), math.log(2) + (N - 1) * sp(-10), atol=1e-12)
check_close("same embeddings without the bias: softplus(-10) + (n-1) log 2 (grows with n)",
            siglip_loss(eye, eye, torch.tensor(math.log(10)), torch.tensor(0.0)), sp(-10) + (N - 1) * math.log(2), atol=1e-12)

# %% [markdown]
# ## 4. Learnable temperature / bias: initial values and the CLIP clamp
# CLIP: $t$ starts at $\log(1/0.07)$ (scale ≈ 14.3) and the scale is clamped at 100, i.e. $t \le \ln 100$; open_clip
# clamps the parameter in place after each optimizer step. SigLIP: $t' = \log 10$, $b = -10$.

# %%
def init_temperature_params():
    """Return dict with scalar nn.Parameters: 'clip_t', 'siglip_t_prime', 'siglip_b' at the papers' init values."""
    ### BEGIN SOLUTION
    P = torch.nn.Parameter
    return {"clip_t": P(torch.tensor(math.log(1 / 0.07))),
            "siglip_t_prime": P(torch.tensor(math.log(10.0))),
            "siglip_b": P(torch.tensor(-10.0))}
    ### END SOLUTION


@torch.no_grad()
def clamp_clip_scale_(t):
    """In place: keep the CLIP logit scale exp(t) in [1, 100] (open_clip clamps t to [0, ln 100])."""
    ### BEGIN SOLUTION
    t.clamp_(0, math.log(100))
    ### END SOLUTION


p = init_temperature_params()
check("params are nn.Parameters", all(isinstance(v, torch.nn.Parameter) for v in p.values()))
check_close("CLIP init scale = 1/0.07", p["clip_t"].exp(), 1 / 0.07)
check_close("SigLIP init t = exp(t') = 10", p["siglip_t_prime"].exp(), 10.0)
check_close("SigLIP init b = -10", p["siglip_b"], -10.0)
big = torch.nn.Parameter(torch.tensor(7.0))
clamp_clip_scale_(big)
check_close("clamp: scale <= 100", big.exp(), 100.0)
mid = torch.nn.Parameter(torch.tensor(3.0))
clamp_clip_scale_(mid)
check_close("clamp leaves in-range values alone", mid, 3.0)

# %% [markdown]
# ## Say it out loud
# 1. Write InfoNCE as a cross-entropy. What are the "classes"? What does the loss equal at random init, and what is the
#    link to mutual information?
# 2. What does the temperature do? Describe the τ → 0 and τ → ∞ limits (both checked above).
# 3. Why is the temperature learned in CLIP, and why clamp it?
# 4. Why does CLIP average two cross-entropies? Is the logit matrix symmetric?
# 5. SigLIP vs CLIP: what changes in the loss, in the distributed implementation, and with batch size? What is $b$ for?
# 6. CLIP on 8 GPUs with a per-GPU batch of 4k: what has to be communicated, and what is the memory of the logit matrix?
# 7. Failure modes: false negatives in the batch, small batches, hard negatives. How does SimCLR's NT-Xent differ?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Row $i$ is an $N$-way classification over keys, and the correct class is $k_i$. Logits are $s_{ij}/\tau$.
#    `F.cross_entropy(q̂ k̂ᵀ/τ, arange(N))`. With uninformative similarities the loss is $\approx\log N$. CPC §2.3:
#    $I(x;c) \ge \log N - \mathcal{L}_N$, so the bound can never exceed $\log N$. That is one argument for
#    large batches (more negatives).
# 2. τ sets the sharpness of the softmax over negatives. As τ → ∞ all logits → 0 and the loss → $\log N$ (no signal).
#    As τ → 0, τ·L → $\frac1N\sum_i \max(0, \max_{j\ne i}s_{ij} - s_{ii})$: a zero-margin hinge on the hardest
#    negative. Small τ therefore focuses the gradient on hard negatives and produces more uniform embeddings, but it is
#    noisier and more sensitive to false negatives.
# 3. The right sharpness is hard to tune by hand and changes during training. CLIP makes $t = \log(1/\tau)$ a parameter
#    (initialized to 1/0.07). Without the clamp at 100 the scale keeps growing (the loss can always be reduced by
#    sharpening once pairs are separated), which destabilizes training (CLIP §2.5). The log parametrization keeps the
#    scale positive and makes updates multiplicative.
# 4. The image→text loss normalizes each row over texts, and text→image normalizes each column over images. These are
#    different softmaxes, so both retrieval directions get trained. The logit matrix is $XY^\top$, not symmetric in
#    general. Its transpose is the text→image logit matrix (`logits_per_text`).
# 5. SigLIP uses a per-pair sigmoid instead of a softmax over the batch, so no global normalizer is needed. Each device
#    can compute its block of the loss and pass text embeddings around (the chunked implementation, §3.3), with no
#    all-gather of a full $B\times B$ matrix. It works better at small batch sizes and saturates around 32k (paper
#    abstract). There are $n$ positives and $n^2-n$ negatives. The bias $b=-10$ starts the model at the prior "almost
#    everything is negative". Without it, the initial loss is $\approx(n-1)\log 2$ per image (checked above), and
#    large early updates go into correcting that imbalance (§3.2).
# 6. Each GPU all-gathers the other GPUs' normalized embeddings (with gradients via a custom autograd all-gather; see
#    open_clip `gather_features`) to form its local $4\text{k} \times 32\text{k}$ logit block. In fp32 that is
#    $4096 \cdot 32768 \cdot 4$ B ≈ 0.54 GB per GPU, times a few for softmax and gradients. The full $32\text{k}^2$ matrix
#    is never materialized on one device.
# 7. False negatives: duplicate or near-duplicate captions in the batch get pushed apart. Mitigations are dedup, soft
#    targets, and a larger τ. Small batches give few negatives and a weak bound. Hard negatives (mined or synthetic)
#    give sharper gradients but risk false negatives. SimCLR NT-Xent: $2N$ augmented views, each anchor has 1 positive
#    and $2N-2$ negatives drawn from both views (self-similarity masked out), with a fixed τ.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
