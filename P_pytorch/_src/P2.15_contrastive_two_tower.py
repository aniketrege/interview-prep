# %% [markdown]
# # P2.15 · Contrastive two-tower model (CLIP-style InfoNCE)
#
# **ID:** P2.15 · **Target:** 30 min · **Plan week:** 5 · **Part:** P
#
# **The drill.** Guide §P2 "Must include": two-tower on toy pairs, InfoNCE, recall@k. The toy pairs: each MNIST digit
# is cut into its **top half** (tower A's input) and **bottom half** (tower B's input), like an image and its caption.
# The model must retrieve *the* matching bottom half of a given top half among 2,000 candidates, so this is instance
# retrieval, not just same-digit retrieval. Two separate MLP towers, **L2-normalized** embeddings,
# **symmetric InfoNCE** written by hand (checked against the `F.cross_entropy` formulation used by CLIP / OpenCLIP), a
# **learnable temperature** stored as a log-scale and clamped as in CLIP, and **recall@k** by hand.
#
# **Not allowed inside `clip_loss`:** `F.cross_entropy`, `log_softmax`, `softmax`. Use `logsumexp`. The test uses
# `F.cross_entropy` as the reference.
#
# ## Sources
# - A. van den Oord, Y. Li, O. Vinyals, *Representation Learning with Contrastive Predictive Coding*, 2018,
#   arXiv:1807.03748 — https://arxiv.org/abs/1807.03748 (InfoNCE; $I(x;c) \ge \log N - \mathcal L_N$, §2.3).
# - A. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021,
#   arXiv:2103.00020 — https://arxiv.org/abs/2103.00020 (Fig. 3 pseudo-code: symmetric CE over the similarity matrix;
#   §2.5: learnable temperature initialized to 0.07 and clipped so logits are scaled by at most 100).
# - T. Chen, S. Kornblith, M. Norouzi, G. Hinton, *A Simple Framework for Contrastive Learning of Visual
#   Representations* (SimCLR), ICML 2020, arXiv:2002.05709 — https://arxiv.org/abs/2002.05709 (NT-Xent: 2(N−1) negatives
#   from both views; temperature and batch-size ablations).
# - OpenCLIP v2.24.0 (MIT): `src/open_clip/loss.py` `ClipLoss.forward` (mean of the two cross-entropies),
#   `src/open_clip/model.py` (`logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))`),
#   `src/training/train.py` line 189 (`logit_scale.clamp_(0, math.log(100))` after each step) —
#   https://github.com/mlfoundations/open_clip/tree/v2.24.0 . Read for reference, not copied.
# - MNIST via `torchvision.datasets.MNIST` (offline copy in `prep.DATA`).

# %%
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from prep import DATA, Session, check, check_close, seed_everything

s = Session("P2.15", target_min=30)
seed_everything(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnist = torchvision.datasets.MNIST(DATA, train=True, download=False)
imgs = (mnist.data[:22000].float() / 255.0 - 0.13) / 0.31
top, bottom = imgs[:, :14].reshape(-1, 392).to(dev), imgs[:, 14:].reshape(-1, 392).to(dev)
labels = mnist.targets[:22000].to(dev)
A_tr, B_tr, A_te, B_te = top[:20000], bottom[:20000], top[20000:], bottom[20000:]
y_te = labels[20000:]
print("pairs:", A_tr.shape, B_tr.shape, "| gallery:", len(A_te))

# %% [markdown]
# ## 1. Two towers, normalized embeddings, learnable temperature
# Each tower maps its input to $\mathbb R^{d}$; normalize to unit length so the dot product is a cosine in $[-1,1]$.
# Logits are $s\,z_A z_B^\top$ with $s = e^{\theta}$. Storing $\theta = \log s$ keeps $s>0$ and makes the
# optimization scale-free. CLIP initializes $s = 1/0.07$ and clips $s \le 100$; OpenCLIP does `clamp_(0, log 100)`
# on $\theta$ after every optimizer step.

# %%
class TwoTower(nn.Module):
    def __init__(self, d_in=392, d_hidden=512, d_emb=128, init_temp=0.07):
        super().__init__()
        ### BEGIN SOLUTION
        self.tower_a = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_emb))
        self.tower_b = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_emb))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / init_temp)))
        ### END SOLUTION

    def forward(self, xa, xb):
        """Returns (za, zb, scale): unit-norm embeddings (N, d_emb) and the positive scalar s = exp(logit_scale)."""
        ### BEGIN SOLUTION
        za = F.normalize(self.tower_a(xa), dim=-1)
        zb = F.normalize(self.tower_b(xb), dim=-1)
        return za, zb, self.logit_scale.exp()
        ### END SOLUTION


torch.manual_seed(0)
model = TwoTower().to(dev)
za, zb, scale = model(A_tr[:8], B_tr[:8])
check("embeddings are unit norm", torch.allclose(za.norm(dim=1), torch.ones(8, device=dev), atol=1e-5)
      and torch.allclose(zb.norm(dim=1), torch.ones(8, device=dev), atol=1e-5))
check_close("initial scale = 1/0.07", scale, 1 / 0.07, rtol=1e-5)
check("logit_scale is a learnable scalar Parameter", isinstance(model.logit_scale, nn.Parameter) and model.logit_scale.numel() == 1)
check("towers do not share weights", model.tower_a[0].weight.data_ptr() != model.tower_b[0].weight.data_ptr())

# %% [markdown]
# ## 2. Symmetric InfoNCE by hand
# $L = s\,z_A z_B^\top$ ($N\times N$, positives on the diagonal).
# $$\mathcal L = \tfrac12\Big[\underbrace{\tfrac1N\textstyle\sum_i\big(\log\sum_j e^{L_{ij}} - L_{ii}\big)}_{A\to B\text{ (rows)}}
# + \underbrace{\tfrac1N\textstyle\sum_j\big(\log\sum_i e^{L_{ij}} - L_{jj}\big)}_{B\to A\text{ (columns)}}\Big].$$
# Each row is an N-way classification "which B matches this A?". At $s\to0$ every logit is 0 and $\mathcal L = \ln N$.
# CPC: $I(A;B) \ge \ln N - \mathcal L$, so the bound saturates at $\ln N$. Bigger batches give more negatives and a looser ceiling.

# %%
def clip_loss(za, zb, scale):
    ### BEGIN SOLUTION
    logits = scale * za @ zb.T
    pos = logits.diagonal()
    loss_a = (torch.logsumexp(logits, dim=1) - pos).mean()
    loss_b = (torch.logsumexp(logits, dim=0) - pos).mean()
    return (loss_a + loss_b) / 2
    ### END SOLUTION


def clip_loss_reference(za, zb, scale):  # the CLIP pseudo-code / OpenCLIP ClipLoss formulation
    logits = scale * za @ zb.T
    target = torch.arange(len(za), device=za.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


torch.manual_seed(1)
ua, ub = F.normalize(torch.randn(64, 16), dim=1), F.normalize(torch.randn(64, 16), dim=1)
ub_skew = F.normalize(ub + 0.8 * ua[torch.randperm(64)], dim=1)      # make rows and columns differ
for sc in (1.0, 14.3, 100.0):
    check_close(f"clip_loss == CE formulation (scale={sc})", clip_loss(ua, ub_skew, sc), clip_loss_reference(ua, ub_skew, sc))
check_close("scale -> 0 gives ln N", clip_loss(ua, ub, 0.0), math.log(64))
check_close("symmetric in the two towers", clip_loss(ua, ub_skew, 10.0), clip_loss(ub_skew, ua, 10.0))
check_close("perfect alignment at large scale -> ~0", clip_loss(ua, ua, 100.0), 0.0, atol=1e-3)
th = torch.tensor(2.0, requires_grad=True); th_ref = torch.tensor(2.0, requires_grad=True)
clip_loss(ua, ub_skew, th.exp()).backward(); clip_loss_reference(ua, ub_skew, th_ref.exp()).backward()
check_close("d loss / d logit_scale", th.grad, th_ref.grad)

# %% [markdown]
# ## 3. Training loop (AdamW, clamp the temperature after each step)
# Sample a batch of pairs; the *other* pairs in the batch are the negatives. After `opt.step()`, clamp
# `logit_scale` into $[0, \ln 100]$ under `no_grad`. Do not give the temperature weight decay: it is not a
# weight, and decay would pull $s$ toward 1.

# %%
def train(model, A, B, steps=600, batch_size=512, lr=1e-3, seed=0):
    """Returns history dict with 'loss' and 'temp' (=1/scale) per step."""
    ### BEGIN SOLUTION
    decay = [p for n, p in model.named_parameters() if n != "logit_scale"]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": [model.logit_scale], "weight_decay": 0.0}], lr=lr)
    g = torch.Generator(device=A.device).manual_seed(seed)
    hist = {"loss": [], "temp": []}
    model.train()
    for _ in range(steps):
        idx = torch.randperm(len(A), device=A.device, generator=g)[:batch_size]
        loss = clip_loss(*model(A[idx], B[idx]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.logit_scale.clamp_(0, math.log(100))
        hist["loss"].append(loss.item())
        hist["temp"].append(1 / model.logit_scale.exp().item())
    return hist
    ### END SOLUTION


probe = TwoTower().to(dev)
with torch.no_grad():
    probe.logit_scale.fill_(6.0)                       # e^6 = 403 > 100
train(probe, A_tr, B_tr, steps=1)
check("logit_scale clamped to <= ln 100 after a step", probe.logit_scale.item() <= math.log(100) + 1e-6)

torch.manual_seed(0)
model = TwoTower().to(dev)
with torch.no_grad():
    init_loss = clip_loss(*model(A_tr[:512], B_tr[:512])).item()
print(f"initial loss {init_loss:.3f} vs ln(512) = {math.log(512):.3f}")
hist = train(model, A_tr, B_tr)
print(f"final train loss {sum(hist['loss'][-50:]) / 50:.3f}; learned temperature {hist['temp'][-1]:.4f}; "
      f"CPC MI lower bound on training batches ~ {math.log(512) - sum(hist['loss'][-50:]) / 50:.2f} nats (max ln 512 = {math.log(512):.2f})")
check("training reduced the loss a lot", sum(hist["loss"][-50:]) / 50 < 0.5 * init_loss)

# %% [markdown]
# ## 4. Recall@k by hand
# Queries $Q$ (N, d), gallery $G$ (N, d), query $i$'s match is gallery item $i$. The rank of the match is the number of gallery items
# scoring **strictly higher** than it; recall@k = fraction of queries with rank < k. This is $O(N^2)$ memory for the similarity matrix,
# which is fine for 2,000. For millions you would use top-k over chunks or an ANN index.

# %%
@torch.no_grad()
def recall_at_k(q, g, ks=(1, 5, 10)):
    """Returns {k: recall@k} for matching q[i] <-> g[i]."""
    ### BEGIN SOLUTION
    sims = q @ g.T
    pos = sims.diagonal()[:, None]
    rank = (sims > pos).sum(dim=1)
    return {k: (rank < k).float().mean().item() for k in ks}
    ### END SOLUTION


def recall_bruteforce(q, g, ks):
    top = (q @ g.T).topk(max(ks), dim=1).indices.cpu()
    return {k: sum(i in top[i, :k].tolist() for i in range(len(q))) / len(q) for k in ks}


torch.manual_seed(2)
qa, ga = F.normalize(torch.randn(300, 8), dim=1), F.normalize(torch.randn(300, 8), dim=1)
ga = F.normalize(ga + 1.5 * qa, dim=1)
r_hand, r_bf = recall_at_k(qa, ga, (1, 3, 10)), recall_bruteforce(qa, ga, (1, 3, 10))
for k in (1, 3, 10):
    check_close(f"recall@{k} vs brute force top-k", r_hand[k], r_bf[k])
check_close("identical gallery -> recall@1 = 1", recall_at_k(qa, qa, (1,))[1], 1.0)

model.eval()
with torch.no_grad():
    za_te, zb_te, _ = model(A_te, B_te)
r_ab, r_ba = recall_at_k(za_te, zb_te), recall_at_k(zb_te, za_te)
print(f"top->bottom {r_ab} | bottom->top {r_ba} | random: recall@k = k/{len(A_te)}")
check("top->bottom recall@1 far above chance (> 0.2)", r_ab[1] > 0.2, f"{r_ab[1]:.3f}")
check("recall@10 > 0.6 in both directions", r_ab[10] > 0.6 and r_ba[10] > 0.6, f"{r_ab[10]:.3f}, {r_ba[10]:.3f}")
with torch.no_grad():
    nn_idx = (za_te @ zb_te.T).argmax(1)
same_digit = (y_te[nn_idx] == y_te).float().mean().item()
print(f"top-1 retrieved bottom half has the same digit class {same_digit:.1%} of the time "
      f"(class identity emerges without labels)")

# %%
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ax[0].plot(hist["loss"]); ax[0].axhline(math.log(512), ls="--", c="gray", label="ln N")
ax[0].set(xlabel="step", ylabel="symmetric InfoNCE", title="training loss"); ax[0].legend()
ax[1].plot(hist["temp"]); ax[1].set(xlabel="step", ylabel="τ = 1/s", title="learned temperature")
q = 0
order = (za_te[q] @ zb_te.T).argsort(descending=True)[:5].cpu()
strip = torch.cat([torch.cat([A_te[q].reshape(14, 28).cpu(), B_te[j].reshape(14, 28).cpu()], 0) for j in order], 1)
ax[2].imshow(strip, cmap="gray"); ax[2].axis("off")
ax[2].set(title=f"query top half + top-5 bottoms (true = #{(order == q).nonzero().flatten().tolist() or ['>5']})")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Say it out loud
# 1. Write symmetric InfoNCE from memory. Why is it "cross-entropy with labels `arange(N)`"? Why average both directions?
# 2. Why L2-normalize the embeddings? What happens without normalization, or with normalization but no temperature?
# 3. What does the temperature control? Why learn it as a log, and why did CLIP clip it at 100?
# 4. How does the batch size affect InfoNCE (negatives, the ln N bound)? How do CLIP-scale runs get 32k negatives across GPUs?
# 5. CLIP's loss vs SimCLR's NT-Xent vs SigLIP's sigmoid loss: what is different?
# 6. What are false negatives in this toy task and in real image-text data? What would you do about them, and about hard negatives?
# 7. Recall@k vs mAP/nDCG: when is recall@k the right retrieval metric?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. `logits = s * za @ zb.T; loss = (CE(logits, arange) + CE(logits.T, arange)) / 2` (CLIP Fig. 3). Row i is a softmax
#    classifier over the N candidates, and the correct class is i. One direction trains "given A, find B" and the other "given B, find A". The row and
#    column normalizers differ, so a single direction leaves one retrieval direction (and one tower's geometry) under-constrained.
# 2. Normalizing turns dot products into cosines, so the model cannot lower the loss by inflating norms, and the logit
#    scale becomes a single explicit knob. Without normalization the norms grow and the effective temperature drifts. With normalization but s = 1
#    the logits lie in [−1, 1], the softmax is nearly uniform, and the loss is stuck near ln N with tiny gradients. You need s ≫ 1.
# 3. τ = 1/s sets how peaked the softmax over negatives is. A small τ focuses the gradient on the hardest negatives, and a large τ spreads it out
#    (SimCLR ablates this). The log parameterization keeps s > 0 and makes updates multiplicative. CLIP initialized τ = 0.07 and clipped
#    the logit scale at 100 "to prevent training instability" (Radford et al. 2021, §2.5). OpenCLIP clamps log s to [0, ln 100] after each step.
# 4. A batch of N gives N−1 negatives per positive, and $\ln N - \mathcal L$ lower-bounds the mutual information (van den Oord et al. 2018),
#    so the bound can't exceed ln N. More negatives give a harder task and usually better representations, up to false-negative effects.
#    Distributed training all-gathers embeddings (not inputs) from every GPU and computes the full N×N logits (OpenCLIP's `gather_with_grad`).
#    Gradient accumulation does NOT give more negatives unless you cache embeddings (e.g. GradCache).
# 5. CLIP: two modalities, an N×N matrix, symmetric CE. SimCLR NT-Xent: one encoder, 2N augmented views, each anchor's negatives are the
#    other 2(N−1) views from both "towers", the self-similarity is excluded, and τ is fixed. SigLIP (Zhai et al. 2023, arXiv:2303.15343) replaces the softmax with
#    an independent binary sigmoid loss on every pair, plus a learnable bias. There is no global normalization over the batch, which makes it
#    cheaper to distribute and better at small batch sizes.
# 6. False negatives: off-diagonal pairs that are actually compatible. Here, two different 1s have nearly interchangeable bottoms; in
#    web data, duplicate or near-duplicate captions. The loss pushes them apart anyway. Mitigations: larger, deduplicated datasets, soft
#    targets or label smoothing, removing likely false negatives (very high similarity) from the denominator. Hard negatives (similar but wrong)
#    sharpen the model: mine them from an index or from the batch, but filter them, since hard negatives are where the false negatives hide.
# 7. Recall@k suits the case with one (or few) known relevant items and a downstream stage that consumes the top k (a reranker or an LLM
#    context window), so you need the right item somewhere in the top k. With many graded relevant items, or when the order within
#    the top k matters, use mAP/nDCG/MRR.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
