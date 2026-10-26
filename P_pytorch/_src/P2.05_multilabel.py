# %% [markdown]
# # P2.5 · Multilabel classifier (per-class BCE, micro vs macro F1, per-class thresholds)
#
# **ID:** P2.5 · **Target:** 20 min · **Plan week:** 2 · **Part:** P (PyTorch first principles)
#
# **The drill.** Each example can carry any subset of C labels. Train an MLP with one logit per class and
# `BCEWithLogitsLoss` on a float 0/1 target matrix, compute micro and macro F1 **by hand** (checked against
# scikit-learn), and tune one threshold per class on val.
#
# **Must include (from the plan):** per-class BCE, micro vs macro F1. Also (guide §P1 traps): targets are a
# **float** 0/1 matrix of shape (N, C), and thresholds are chosen **per class at eval**. Initial-loss check,
# overfit-one-batch, whole-set eval under `eval()` + `no_grad()`, test touched once.
#
# **Rules.** No docs, no LLM. Each solution cell keeps only its spec. The self-check names the variables it expects.
#
# ## Sources
# - scikit-learn `make_multilabel_classification` (generative process: number of labels ~ Poisson, features are
#   word counts): https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_multilabel_classification.html
# - scikit-learn *Multiclass and multioutput algorithms* / `f1_score(average="micro" | "macro" | "samples")`
#   (reference for the checks; `zero_division`): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
# - G. Tsoumakas, I. Katakis, I. Vlahavas, *Mining Multi-label Data*, in Data Mining and Knowledge Discovery Handbook,
#   Springer 2010, pp. 667-685, DOI 10.1007/978-0-387-09823-4_34 — https://doi.org/10.1007/978-0-387-09823-4_34
#   (example-based vs label-based metrics; micro vs macro averaging).
# - Z. C. Lipton, C. Elkan, B. Narayanaswamy, *Thresholding Classifiers to Maximize F1 Score*, ECML-PKDD 2014,
#   arXiv:1402.1892 — https://arxiv.org/abs/1402.1892 (micro/macro F1 thresholding in the multilabel setting).
# - `torch.nn.BCEWithLogitsLoss` docs (element-wise BCE, `pos_weight` of shape (C,) for multilabel):
#   https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prep import Session, check, check_close, seed_everything

s = Session("P2.5", target_min=20)
seed_everything(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% tags=["solution"]
# 1. DATA. make_multilabel_classification(n_samples=8000, n_features=40, n_classes=10, n_labels=2,
#    length=150, allow_unlabeled=True, random_state=0) -> X (counts), Y (N, 10) 0/1.
#    Inspect: label frequency per class (prevalence), mean labels per example, fraction with no label.
#    Split 70/15/15 (random_state=0; plain split, stratifying multilabel data needs iterative stratification).
#    Features: log1p, then standardize with train stats. Tensors: X_* float32; Y_* FLOAT32 (N, 10). Keep Y_*_np.
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

X, Y = make_multilabel_classification(n_samples=8000, n_features=40, n_classes=10, n_labels=2, length=150,
                                      allow_unlabeled=True, random_state=0)
print(X.shape, Y.shape, "prevalence per class", Y.mean(0).round(3))
print(f"labels/example {Y.sum(1).mean():.2f}, no-label rows {(Y.sum(1) == 0).mean():.3f}")
X_train, X_tmp, Y_train_np, Y_tmp = train_test_split(X, Y, test_size=0.30, random_state=0)
X_val, X_test, Y_val_np, Y_test_np = train_test_split(X_tmp, Y_tmp, test_size=0.50, random_state=0)
X_train, X_val, X_test = (np.log1p(a) for a in (X_train, X_val, X_test))
mu, sd = X_train.mean(0), X_train.std(0)
X_train, X_val, X_test = (torch.tensor((a - mu) / sd, dtype=torch.float32) for a in (X_train, X_val, X_test))
Y_train, Y_val, Y_test = (torch.tensor(a, dtype=torch.float32) for a in (Y_train_np, Y_val_np, Y_test_np))
C = Y.shape[1]

# %% tags=["solution"]
# 2. F1 BY HAND. Inputs: 0/1 numpy arrays (N, C).
#    per_class_counts(Y, P) -> tp, fp, fn, each shape (C,).
#    f1_micro(Y, P): pool tp/fp/fn over ALL classes, then F1 = 2TP / (2TP + FP + FN).
#    f1_macro(Y, P): F1 per class (0 when 2TP+FP+FN == 0, sklearn's zero_division=0), then the unweighted mean.
#    f1_samples(Y, P): F1 per EXAMPLE (row), then the mean (same zero-division rule).
def per_class_counts(Y, P):
    Y, P = np.asarray(Y).astype(bool), np.asarray(P).astype(bool)
    return (Y & P).sum(0), (~Y & P).sum(0), (Y & ~P).sum(0)


def _f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def f1_micro(Y, P):
    tp, fp, fn = per_class_counts(Y, P)
    return float(_f1(tp.sum(), fp.sum(), fn.sum()))


def f1_macro(Y, P):
    return float(_f1(*per_class_counts(Y, P)).mean())


def f1_samples(Y, P):
    tp, fp, fn = per_class_counts(np.asarray(Y).T, np.asarray(P).T)  # transpose: rows become "classes"
    return float(_f1(tp, fp, fn).mean())

# %% tags=["solution"]
# 3. MODEL + LOSS. `MLP`: 40 -> 128 -> 128 -> C, ReLU, Dropout(0.1); returns (B, C) logits. make_model() on `dev`.
#    loss_fn = nn.BCEWithLogitsLoss() (mean over all N*C entries). DataLoaders: train batch 128 shuffle, val/test 2048.
#    Sanity: `init_loss` on val (expect ≈ ln 2 for every entry); overfit one batch of 64 (fresh model, eval mode,
#    Adam lr=1e-2, 300 steps) -> `overfit_loss` < 0.05.
class MLP(nn.Module):
    def __init__(self, d_in=40, d_h=128, n_out=10, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(d_h, d_h), nn.ReLU(), nn.Dropout(p), nn.Linear(d_h, n_out))

    def forward(self, x):
        return self.net(x)


def make_model():
    return MLP(n_out=C).to(dev)


loss_fn = nn.BCEWithLogitsLoss()
train_dl = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)
val_dl = DataLoader(TensorDataset(X_val, Y_val), batch_size=2048)
test_dl = DataLoader(TensorDataset(X_test, Y_test), batch_size=2048)

model = make_model()
with torch.no_grad():
    model.eval()
    init_loss = loss_fn(model(X_val.to(dev)), Y_val.to(dev)).item()
print(f"init BCE {init_loss:.3f} vs ln2 {math.log(2):.3f}")
tmp = make_model().eval()
tmp_opt = torch.optim.Adam(tmp.parameters(), lr=1e-2)
xb, yb = X_train[:64].to(dev), Y_train[:64].to(dev)
for _ in range(300):
    loss = loss_fn(tmp(xb), yb)
    tmp_opt.zero_grad()
    loss.backward()
    tmp_opt.step()
overfit_loss = loss.item()
print(f"overfit-one-batch {overfit_loss:.4f}")

# %% tags=["solution"]
# 4. TRAIN + EVAL. AdamW lr=1e-3, wd=1e-2, 30 epochs. `probs(model, dl) -> numpy (N, C)` sigmoid probabilities
#    (eval, no_grad, whole set). Each epoch record train loss and val micro-F1 at threshold 0.5 in `history`
#    ("train_loss", "val_micro"). Keep the best-val-micro state and load it at the end.
@torch.no_grad()
def probs(model, dl):
    model.eval()
    return torch.cat([torch.sigmoid(model(xb.to(dev))).cpu() for xb, _ in dl]).double().numpy()


opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
history = {"train_loss": [], "val_micro": []}
best, best_state = -1.0, None
for epoch in range(30):
    model.train()
    tot, n = 0.0, 0
    for xb, yb in train_dl:
        xb, yb = xb.to(dev), yb.to(dev)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item() * len(yb)
        n += len(yb)
    vm = f1_micro(Y_val_np, probs(model, val_dl) >= 0.5)
    history["train_loss"].append(tot / n)
    history["val_micro"].append(vm)
    if vm > best:
        best, best_state = vm, {k: v.detach().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
print(f"best val micro-F1 @0.5 {best:.4f}")

# %% tags=["solution"]
# 5. PER-CLASS THRESHOLDS. On val, for each class c choose t_c from the grid np.linspace(0.05, 0.95, 19) (it must
#    include 0.5) maximizing that class's F1 -> `thr` (C,). Report on val AND test (test once):
#    micro / macro / samples F1 at 0.5 and at `thr` -> dicts `val_05`, `val_thr`, `test_05`, `test_thr`
#    with keys "micro", "macro", "samples". Also print the per-class F1 + prevalence table for test.
grid = np.linspace(0.05, 0.95, 19)
P_val = probs(model, val_dl)
f1_grid = np.stack([_f1(*per_class_counts(Y_val_np, P_val >= t)) for t in grid])  # (G, C)
thr = grid[f1_grid.argmax(0)]
print("per-class thresholds:", thr.round(2))
P_test = probs(model, test_dl)
metrics = lambda Y, P: {"micro": f1_micro(Y, P), "macro": f1_macro(Y, P), "samples": f1_samples(Y, P)}
val_05, val_thr = metrics(Y_val_np, P_val >= 0.5), metrics(Y_val_np, P_val >= thr)
test_05, test_thr = metrics(Y_test_np, P_test >= 0.5), metrics(Y_test_np, P_test >= thr)
for name, d in [("val @0.5", val_05), ("val @thr", val_thr), ("test @0.5", test_05), ("test @thr", test_thr)]:
    print(f"{name:10s} " + "  ".join(f"{k} {v:.4f}" for k, v in d.items()))
per_class = _f1(*per_class_counts(Y_test_np, P_test >= thr))
for c in range(C):
    print(f"class {c}: prevalence {Y_test_np[:, c].mean():.3f}  F1 {per_class[c]:.3f}  t={thr[c]:.2f}")

# %% [markdown]
# ## Self-check
# Expects: `Y_train`, `f1_micro`, `f1_macro`, `f1_samples`, `model`, `loss_fn`, `init_loss`, `overfit_loss`,
# `thr`, `val_05`, `val_thr`, `test_thr`.

# %%
from sklearn.metrics import f1_score

rng = np.random.default_rng(0)
prev = np.array([0.5, 0.3, 0.1, 0.03, 0.0])          # imbalanced classes, one with NO positives
Yr = (rng.random((400, 5)) < prev).astype(int)
Pr = np.where(rng.random((400, 5)) < 0.8, Yr, 1 - Yr)  # 80 %-accurate predictions
Pr[:, 4] = 0                                           # nothing predicted for the empty class either
for avg, fn in [("micro", f1_micro), ("macro", f1_macro), ("samples", f1_samples)]:
    check_close(f"f1 {avg} == sklearn", fn(Yr, Pr), f1_score(Yr, Pr, average=avg, zero_division=0))
check("micro != macro on imbalanced labels", abs(f1_micro(Yr, Pr) - f1_macro(Yr, Pr)) > 0.05)
check("targets are a FLOAT 0/1 matrix (N, C)", Y_train.dtype == torch.float32 and Y_train.shape[1] == C)
check("loss is BCEWithLogits", isinstance(loss_fn, nn.BCEWithLogitsLoss))
check("model returns (B, C) logits", model(X_train[:4].to(dev)).shape == (4, C))
check("no Sigmoid in the model", not any(isinstance(m, nn.Sigmoid) for m in model.modules()))
check("initial loss ~ ln 2", abs(init_loss - math.log(2)) < 0.1, f"got {init_loss:.3f}")
check("can overfit one batch", overfit_loss < 0.05, f"got {overfit_loss:.4f}")
check("one threshold per class", np.shape(thr) == (C,))
check("per-class thresholds can't hurt val macro-F1", val_thr["macro"] >= val_05["macro"] - 1e-9, f"{val_thr} vs {val_05}")
check("test micro-F1 > 0.70", test_thr["micro"] > 0.70, f"got {test_thr['micro']:.3f}")
check("test macro-F1 > 0.60", test_thr["macro"] > 0.60, f"got {test_thr['macro']:.3f}")

# %% [markdown]
# ## Say it out loud (P3)
# 1. Why BCE per class and not softmax + CE? When would softmax be right even with "many" labels?
# 2. Why must the targets be float, and what shape is the loss before reduction?
# 3. Micro vs macro F1: which one does a frequent class dominate? Which would you report for a long-tailed label set, and why both?
# 4. Why tune thresholds per class, and why on val? What is the risk for a class with 20 val positives?
# 5. What is "samples" F1, and how is it different from subset accuracy (exact match) and Hamming loss?
# 6. Why is the initial loss ln 2 even though most labels are 0? What init would give a lower initial loss?
# 7. How would you handle label imbalance in the loss (shape of `pos_weight`)?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Labels are not mutually exclusive. Softmax forces the probabilities to sum to 1, so raising one label lowers
#    the others. Per-class sigmoids model C independent Bernoulli outputs, one per label (given the shared features).
#    Softmax is right when exactly one label is true (multiclass), however large C is.
# 2. BCE computes y·log σ(z) + (1−y)·log(1−σ(z)) element-wise, so y must be a float tensor shaped like the logits.
#    Integer class indices would be a CE target. With `reduction="none"` the loss is (N, C). The default mean
#    divides by N·C, which matters if you compare loss values across different C.
# 3. Micro F1 pools TP/FP/FN over all labels, so frequent labels dominate it. Macro F1 averages per-label F1 with
#    equal weight, so rare labels count as much as common ones, and it is much noisier. For long-tailed label sets,
#    report both. Macro tells you whether the tail works at all.
# 4. Classes differ in prevalence and in how well calibrated they are, so one global 0.5 is rarely optimal
#    (Lipton et al.: for a calibrated classifier the optimal threshold is F1*/2, and F1* differs per class). Tune
#    on val because thresholds are hyperparameters. With about 20 positives the F1 curve is jagged, so the chosen
#    threshold overfits. Mitigate with a coarse grid, smoothing, cross-validation, or a shared threshold for rare
#    classes.
# 5. Samples F1 is F1 computed per example (over its label set), then averaged (example-based). Subset accuracy
#    requires the whole label vector to be right, which is very strict. Hamming loss is the fraction of the N·C
#    entries that are wrong, which is very lenient because most entries are easy 0s. Gotcha: about 13 % of rows here
#    have no labels at all. For such a row, predicting nothing gives 0/0, and with `zero_division=0` it scores
#    F1 = 0. That is why samples F1 is the lowest number above. Decide (and state) whether such rows should count
#    as 1 (`zero_division=1`).
# 6. At init all logits ≈ 0, so σ = 0.5 for every entry and each entry costs ln 2, whatever its label. Setting each
#    output bias to logit(prevalence_c) starts the model at the label priors. The initial loss then drops to the
#    mean binary entropy of the priors.
# 7. `BCEWithLogitsLoss(pos_weight=w)` with w of shape (C,), e.g. w_c = n_neg,c / n_pos,c, broadcast over the
#    batch. Alternatives are focal loss or resampling. As in P2.4, reweighting shifts the optimal thresholds, so
#    re-tune them afterwards.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")  # score: 1-5 self-rating; notes: what tripped you up
