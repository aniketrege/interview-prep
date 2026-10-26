# %% [markdown]
# # A3.5 · Box IoU, NMS, GIoU, and ranking metrics (P@k, R@k, MRR, nDCG@k)
#
# **ID:** A3.5 · **Target:** 20 min · **Plan week:** 12 (vision / retrieval rounds), useful anytime · **Part:** A
#
# **The drill.** Two small toolkits that come up in vision and retrieval rounds.
# (1) Detection: pairwise box IoU for `xyxy` boxes (vectorized, `(N, M)`), greedy non-maximum suppression, and GIoU
# (optional), checked against `torchvision.ops.box_iou`, `nms` (identical kept indices on random boxes),
# `batched_nms` and `generalized_box_iou`. (2) Ranking: precision@k, recall@k, MRR and nDCG@k for a batch of queries,
# checked against brute-force loops and `sklearn.metrics.ndcg_score`. ROC-AUC via ranking lives in P2.9.
# Guide A3: "AUC/precision@k; NMS and IoU (vision rounds)".
#
# **Rules.** Torch only inside your functions; no `torchvision.ops`, no sklearn. Vectorize over boxes and queries. The only
# Python loop allowed is NMS's outer "pick the next best box" loop.
#
# ## Sources
# - torchvision ops docs: `box_iou` https://docs.pytorch.org/vision/stable/generated/torchvision.ops.box_iou.html ·
#   `nms` (discards boxes with IoU **>** threshold, keeps indices sorted by decreasing score)
#   https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html ·
#   `batched_nms` https://docs.pytorch.org/vision/stable/generated/torchvision.ops.batched_nms.html ·
#   `generalized_box_iou` https://docs.pytorch.org/vision/stable/generated/torchvision.ops.generalized_box_iou.html
# - H. Rezatofighi et al., *Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression*,
#   CVPR 2019, arXiv:1902.09630 — https://arxiv.org/abs/1902.09630
# - N. Bodla, B. Singh, R. Chellappa, L. S. Davis, *Soft-NMS — Improving Object Detection With One Line of Code*, ICCV 2017,
#   arXiv:1704.04503 — https://arxiv.org/abs/1704.04503
# - K. Järvelin, J. Kekäläinen, *Cumulated gain-based evaluation of IR techniques*, ACM TOIS 20(4):422-446, 2002,
#   DOI 10.1145/582415.582418 — https://doi.org/10.1145/582415.582418 (DCG / nDCG).
# - C. D. Manning, P. Raghavan, H. Schütze, *Introduction to Information Retrieval*, Cambridge University Press 2008,
#   ch. 8 "Evaluation in information retrieval" (precision/recall at k, MRR-style rank metrics):
#   https://nlp.stanford.edu/IR-book/
# - `sklearn.metrics.ndcg_score` (linear gain, $\log_2$ discount, `k=`):
#   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html

# %%
import math

import numpy as np
import torch
from sklearn.metrics import ndcg_score
from torchvision.ops import batched_nms as tv_batched_nms
from torchvision.ops import box_iou as tv_box_iou
from torchvision.ops import generalized_box_iou as tv_giou
from torchvision.ops import nms as tv_nms

from prep import Session, check, check_close, seed_everything

s = Session("A3.5", target_min=20)
seed_everything(0)


def random_boxes(n, size=100.0, max_wh=40.0, gen=None):
    xy = torch.rand(n, 2, generator=gen) * size
    wh = torch.rand(n, 2, generator=gen) * max_wh + 1.0
    return torch.cat([xy, xy + wh], 1)   # xyxy with x1 < x2, y1 < y2

# %% [markdown]
# ## 1. Pairwise IoU
# For `a (N, 4)`, `b (M, 4)` in `(x1, y1, x2, y2)`: the intersection is `[max(x1), max(y1), min(x2), min(y2)]` with
# width/height clamped at 0 (disjoint boxes). Then IoU $= I / (A_a + A_b - I)$. Broadcast `a[:, None, :]` against `b[None]`.

# %%
def box_area(b):
    ### BEGIN SOLUTION
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ### END SOLUTION


def box_iou(a, b):
    ### BEGIN SOLUTION
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    return inter / (box_area(a)[:, None] + box_area(b)[None, :] - inter)
    ### END SOLUTION


g = torch.Generator().manual_seed(0)
a, b = random_boxes(30, gen=g), random_boxes(20, gen=g)
check_close("box_iou vs torchvision", box_iou(a, b), tv_box_iou(a, b), atol=1e-6)
far = torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([[5.0, 5.0, 6.0, 7.0], [0.5, 0.5, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
check_close("disjoint / overlapping / identical", box_iou(*far), torch.tensor([[0.0, 0.25 / 3.0, 1.0]]))

# %% [markdown]
# ## 2. Greedy NMS
# Sort by score (descending). Repeatedly keep the best remaining box and drop every remaining box whose IoU with it is
# **greater than** `iou_thr`. Return the kept indices (into the original arrays), ordered by decreasing score, as `int64`.

# %%
def nms(boxes, scores, iou_thr):
    ### BEGIN SOLUTION
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if rest.numel() == 0:
            break
        order = rest[box_iou(boxes[i][None], boxes[rest])[0] <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)
    ### END SOLUTION


for trial in range(30):
    bx = random_boxes(200, gen=g)
    sc = torch.rand(200, generator=g)
    for thr in (0.1, 0.5, 0.8):
        mine, ref = nms(bx, sc, thr), tv_nms(bx, sc, thr)
        assert torch.equal(mine, ref), f"trial {trial} thr {thr}: kept {mine.tolist()[:10]}... vs {ref.tolist()[:10]}..."
check("nms == torchvision.ops.nms (30 random sets x 3 thresholds)", True)
check("nms keeps everything at thr = 1", len(nms(bx, sc, 1.0)) == 200)
two = torch.tensor([[0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 1.0, 1.0]])   # IoU exactly 0.5
check("IoU == thr is kept (strict >)", torch.equal(nms(two, torch.tensor([0.9, 0.8]), 0.5), tv_nms(two, torch.tensor([0.9, 0.8]), 0.5)),
      "suppress only IoU > thr")

# %% [markdown]
# **Class-aware NMS** (`batched_nms`): boxes of different classes must not suppress each other. Trick: shift each class's
# boxes by `class_id × (max coordinate + 1)` so that boxes of different classes never overlap, then run one NMS.

# %%
def batched_nms(boxes, scores, classes, iou_thr):
    ### BEGIN SOLUTION
    offset = classes.to(boxes.dtype)[:, None] * (boxes.max() + 1)
    return nms(boxes + offset, scores, iou_thr)
    ### END SOLUTION


for trial in range(10):
    bx, sc, cl = random_boxes(150, gen=g), torch.rand(150, generator=g), torch.randint(0, 4, (150,), generator=g)
    assert torch.equal(batched_nms(bx, sc, cl, 0.5), tv_batched_nms(bx, sc, cl, 0.5)), f"trial {trial}"
check("batched_nms == torchvision.ops.batched_nms", True)

# %% [markdown]
# ## 3. Optional: GIoU
# $\mathrm{GIoU} = \mathrm{IoU} - \frac{|C| - |A\cup B|}{|C|}$, with $C$ the smallest box enclosing both. It lies in $(-1, 1]$ and,
# unlike IoU, still gives a signal (and a gradient, as a loss $1-\mathrm{GIoU}$) when the boxes do not overlap.

# %%
def generalized_box_iou(a, b):
    ### BEGIN SOLUTION
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    clt = torch.minimum(a[:, None, :2], b[None, :, :2])
    crb = torch.maximum(a[:, None, 2:], b[None, :, 2:])
    hull = (crb - clt).prod(-1)
    return inter / union - (hull - union) / hull
    ### END SOLUTION


check_close("GIoU vs torchvision", generalized_box_iou(a, b), tv_giou(a, b), atol=1e-6)

# %% [markdown]
# ## 4. Ranking metrics
# Inputs for a batch of $Q$ queries over $n$ candidates: `scores (Q, n)` (higher = ranked earlier) and relevance
# `rel (Q, n)`, binary for P/R/MRR and graded (0, 1, 2, ...) for nDCG. Each function returns the **mean over queries**.
# - precision@k = (# relevant in top k) / k; recall@k = (# relevant in top k) / (# relevant). Assume every query has ≥ 1 relevant item.
# - MRR = mean of $1/\text{rank}$ of the first relevant item (rank starts at 1).
# - $\mathrm{DCG@k} = \sum_{r=1}^{k} \frac{rel_{(r)}}{\log_2(r+1)}$ (linear gain, like sklearn); nDCG@k = DCG@k / IDCG@k, where IDCG sorts by the true relevance.
#
# Hint: `idx = scores.topk(k, dim=1).indices` and `rel.gather(1, idx)` give the relevance in ranked order.

# %%
def precision_at_k(scores, rel, k):
    ### BEGIN SOLUTION
    top = rel.gather(1, scores.topk(k, dim=1).indices)
    return ((top > 0).sum(1).double() / k).mean()
    ### END SOLUTION


def recall_at_k(scores, rel, k):
    ### BEGIN SOLUTION
    top = rel.gather(1, scores.topk(k, dim=1).indices)
    return ((top > 0).sum(1).double() / (rel > 0).sum(1)).mean()
    ### END SOLUTION


def mrr(scores, rel):
    ### BEGIN SOLUTION
    ranked = rel.gather(1, scores.argsort(1, descending=True)) > 0
    first = ranked.double().argmax(1)                      # first True (rows with no relevant item get 0 below)
    rr = 1.0 / (first + 1).double()
    return torch.where(ranked.any(1), rr, torch.zeros_like(rr)).mean()
    ### END SOLUTION


def ndcg_at_k(scores, rel, k):
    ### BEGIN SOLUTION
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float64))
    dcg = (rel.gather(1, scores.topk(k, dim=1).indices).double() * disc).sum(1)
    idcg = (rel.double().topk(k, dim=1).values * disc).sum(1)
    return (dcg / idcg).mean()
    ### END SOLUTION


def loops_reference(scores, rel, k):
    P, R, RR, ND = [], [], [], []
    for q in range(len(scores)):
        order = sorted(range(scores.shape[1]), key=lambda i: -scores[q, i].item())
        r = [rel[q, i].item() for i in order]
        hits = sum(1 for x in r[:k] if x > 0)
        P.append(hits / k)
        R.append(hits / sum(1 for x in r if x > 0))
        RR.append(next((1.0 / (j + 1) for j, x in enumerate(r) if x > 0), 0.0))
        dcg = sum(x / math.log2(j + 2) for j, x in enumerate(r[:k]))
        idcg = sum(x / math.log2(j + 2) for j, x in enumerate(sorted(r, reverse=True)[:k]))
        ND.append(dcg / idcg)
    return [float(np.mean(v)) for v in (P, R, RR, ND)]


Q, n = 64, 30
scores = torch.randn(Q, n, generator=g, dtype=torch.float64)
rel_graded = (torch.rand(Q, n, generator=g) ** 3 * 4).floor().long()        # mostly 0, some 1-3
rel_graded[torch.arange(Q), torch.randint(0, n, (Q,), generator=g)] = torch.randint(1, 4, (Q,), generator=g)  # >= 1 relevant
rel_bin = (rel_graded > 0).long()
for k in (1, 5, 10):
    P, R, RR, ND = loops_reference(scores, rel_graded, k)
    check_close(f"precision@{k}", precision_at_k(scores, rel_bin, k), P)
    check_close(f"recall@{k}", recall_at_k(scores, rel_bin, k), R)
    check_close(f"nDCG@{k} vs loops", ndcg_at_k(scores, rel_graded, k), ND)
    check_close(f"nDCG@{k} vs sklearn", ndcg_at_k(scores, rel_graded, k), ndcg_score(rel_graded.numpy(), scores.numpy(), k=k))
check_close("MRR", mrr(scores, rel_bin), RR)
check_close("MRR with a query that has no relevant item", mrr(scores[:2], torch.stack([rel_bin[0], torch.zeros(n, dtype=torch.long)])),
            loops_reference(scores[:1], rel_bin[:1], 1)[2] / 2)
perfect = rel_graded.double() + 1e-3 * torch.randn(Q, n, generator=g, dtype=torch.float64)
check_close("perfect ranking has nDCG = 1", ndcg_at_k(perfect, rel_graded, 10), 1.0)

# %% [markdown]
# ## Say it out loud
# 1. Write IoU for `xyxy` boxes. What breaks if boxes are `xywh` or if you forget the clamp at 0?
# 2. Walk through greedy NMS. Complexity? What do Soft-NMS and class-aware (batched) NMS change?
# 3. Why is NMS's tie/threshold convention (`>` vs `>=`) worth stating? What happens at thresholds 0 and 1?
# 4. When does GIoU help over IoU as a loss?
# 5. Precision@k vs recall@k vs MRR vs nDCG: what does each reward, and which would you report for retrieval-augmented generation?
# 6. How does nDCG handle graded relevance? Linear gain vs $2^{rel}-1$?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Intersection corners: `max` of the top-lefts and `min` of the bottom-rights; width and height clamped at 0; IoU = I / (A + B − I). For `xywh`, convert first
#    (`x2 = x + w`), otherwise the "corners" are wrong. Without the clamp, two disjoint boxes give a product of two negative sides, a *positive* fake intersection.
# 2. Sort by score. Pop the top box, keep it, and remove all remaining boxes with IoU > thr against it. Repeat. Worst case $O(n^2)$ IoUs, and
#    $O(n\log n)$ for the sort. Soft-NMS decays the scores of overlapping boxes instead of deleting them, which helps with crowded objects (Bodla et al. 2017).
#    Class-aware NMS runs NMS per class; the coordinate-offset trick does it in one call (torchvision's `batched_nms` uses it for small inputs, e.g. ≤ 4000 box coordinates on CPU, and loops over classes otherwise).
# 3. torchvision discards boxes with IoU **strictly greater** than the threshold, so at thr = 1 nothing is suppressed except boxes that overlap
#    with IoU > 1 (impossible). At thr = 0 any positive overlap suppresses. Matching a reference implementation exactly (as tested here)
#    depends on this convention and on deterministic tie-breaking of equal scores.
# 4. When the predicted and target boxes do not overlap, IoU is 0 with zero gradient, so the IoU loss gives no signal. GIoU's enclosing-box term
#    still pulls the boxes together (Rezatofighi et al. 2019). DIoU/CIoU add center-distance and aspect-ratio terms.
# 5. P@k: the fraction of the top-k that is useful (what a user sees). R@k: coverage of all relevant items (what a reranker or an LLM context can
#    use); for RAG, recall@k of the retriever matters most, because the generator can only use what is retrieved. MRR: how early the first hit is
#    (QA with one answer). nDCG: graded relevance with a position discount, normalized per query to [0, 1].
# 6. Gains are summed with discount $1/\log_2(r+1)$ and normalized by the ideal ordering's DCG. Linear gain ($rel$, sklearn's choice) vs exponential
#    $2^{rel}-1$ (common in web search, e.g. LETOR) emphasizes highly relevant items more. They coincide for binary relevance. Always say which one you
#    use. The log base does not matter for nDCG (it cancels in DCG/IDCG), only for raw DCG. Ties in predicted scores also need a convention (sklearn averages over tied orderings by default).
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
