# %% [markdown]
# # P2.10a · Bug hunt A: an MNIST training script with 6 planted bugs
#
# **ID:** P2.10a · **Target:** 15 min · **Plan week:** 3 · **Part:** P
#
# **The drill.** The cell below is a complete MNIST training script. It **runs without errors**, prints numbers,
# and is wrong in **6 places**. Find and fix all 6 by editing that cell in place, then run the self-check, which
# probes your functions (`compute_loss`, `train_step`, `predict`, `evaluate`) and normalization statistics on
# fresh probe batches and prints which bugs remain. For each one, say out loud *what symptom it would produce*
# and *how you would have caught it* (initial-loss check, overfit one batch, a unit test, a curve).
#
# The model has an auxiliary head: besides the digit (10 logits), it predicts whether the digit is **odd**
# from one logit. Keep the function names and signatures, and keep `Net()` constructible without arguments.
# A fixed script reaches > 95% validation accuracy on both heads in 5 epochs.
#
# **Rules.** No running the probes before you have read the whole script once. Count only real bugs: style
# issues, hyperparameters and "I would have used Adam" are not bugs. The guide's list for this drill is in the
# collapsed hint below. Open it only if you are stuck after 10 minutes.
#
# <details><summary>Hint (the six categories)</summary>
# softmax before cross-entropy · missing <code>zero_grad</code> · wrong <code>dim</code> · evaluating in train mode ·
# float labels for cross-entropy · leakage from normalizing with statistics of all the data.
# </details>
#
# ## Sources
# - Guide §P0 (the 45-minute skeleton) and §P2 drill P2.10; the six bug categories are the guide's.
# - PyTorch `CrossEntropyLoss` docs. Targets are class indices (`long`) **or class probabilities (float, same shape
#   as the input)**; an input of shape `(C)` is an unbatched single example:
#   https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# - PyTorch `BCEWithLogitsLoss` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
# - PyTorch FAQ / autograd mechanics: gradients accumulate in `.grad` until zeroed —
#   https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
# - `Module.eval()` (Dropout/BatchNorm behaviour): https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval
# - A. Karpathy, *A Recipe for Training Neural Networks* (2019), "verify loss @ init", "overfit one batch":
#   https://karpathy.github.io/2019/04/25/recipe/
# - S. Kaufman, S. Rosset, C. Perlich, O. Stitelman, *Leakage in Data Mining: Formulation, Detection, and Avoidance*,
#   ACM TKDD 6(4), 2012, DOI 10.1145/2382577.2382579 — https://doi.org/10.1145/2382577.2382579
# - LeCun, Cortes, Burges, MNIST database: http://yann.lecun.com/exdb/mnist/ (loaded offline via `torchvision.datasets.MNIST`).

# %%
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from prep import DATA, Session, check, seed_everything

s = Session("P2.10a", target_min=15)
seed_everything(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% tags=["template_only"]
# ===== the script under review: find and fix the 6 bugs in this cell =====
mnist = torchvision.datasets.MNIST(DATA, train=True, download=False)
X_raw = mnist.data[:12000].reshape(-1, 784).float() / 255.0
y_all = mnist.targets[:12000]
perm = torch.randperm(len(X_raw), generator=torch.Generator().manual_seed(0))
idx_tr, idx_val = perm[:10000], perm[10000:]

mu, sd = X_raw.mean(), X_raw.std()
X_all = (X_raw - mu) / sd
X_tr, y_tr = X_all[idx_tr], y_all[idx_tr]
X_val, y_val = X_all[idx_val], y_all[idx_val]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.digit_head = nn.Linear(128, 10)
        self.odd_head = nn.Linear(128, 1)          # auxiliary task: is the digit odd?

    def forward(self, x):
        h = self.body(x)
        return self.digit_head(h), self.odd_head(h).squeeze(-1)   # (B, 10), (B,)


def compute_loss(model, xb, yb):
    digit_logits, odd_logit = model(xb)
    loss_digit = F.cross_entropy(F.softmax(digit_logits, dim=1), yb)
    loss_odd = F.cross_entropy(odd_logit, (yb % 2).float())
    return loss_digit, loss_odd


def train_step(model, opt, xb, yb):
    model.train()
    loss_digit, loss_odd = compute_loss(model, xb, yb)
    (loss_digit + loss_odd).backward()
    opt.step()
    return loss_digit.item(), loss_odd.item()


def predict(model, x):
    digit_logits, odd_logit = model(x)
    return F.softmax(digit_logits, dim=0).argmax(dim=1), (odd_logit > 0).long()


def evaluate(model, X, y):
    with torch.no_grad():
        pred_digit, pred_odd = predict(model, X)
    return (pred_digit == y).float().mean().item(), (pred_odd == y % 2).float().mean().item()


torch.manual_seed(0)
model = Net().to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
for epoch in range(5):
    order = torch.randperm(len(X_tr))
    for i in range(0, len(X_tr), 128):
        b = order[i:i + 128]
        ld, lo = train_step(model, opt, X_tr[b].to(dev), y_tr[b].to(dev))
    acc_digit, acc_odd = evaluate(model, X_val.to(dev), y_val.to(dev))
    print(f"epoch {epoch}: last loss digit {ld:.3f} odd {lo:.3f} | val acc digit {acc_digit:.3f} odd {acc_odd:.3f}")

# %% tags=["solution_only"]
# ===== fixed script; every change is marked BUG n =====
mnist = torchvision.datasets.MNIST(DATA, train=True, download=False)
X_raw = mnist.data[:12000].reshape(-1, 784).float() / 255.0
y_all = mnist.targets[:12000]
perm = torch.randperm(len(X_raw), generator=torch.Generator().manual_seed(0))
idx_tr, idx_val = perm[:10000], perm[10000:]

# BUG 6: statistics were computed on ALL rows (train + val) before splitting: val information leaks into
#        preprocessing and the val estimate is optimistic. Fit preprocessing on the training rows only.
mu, sd = X_raw[idx_tr].mean(), X_raw[idx_tr].std()
X_tr, y_tr = (X_raw[idx_tr] - mu) / sd, y_all[idx_tr]
X_val, y_val = (X_raw[idx_val] - mu) / sd, y_all[idx_val]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.digit_head = nn.Linear(128, 10)
        self.odd_head = nn.Linear(128, 1)          # auxiliary task: is the digit odd?

    def forward(self, x):
        h = self.body(x)
        return self.digit_head(h), self.odd_head(h).squeeze(-1)   # (B, 10), (B,)


def compute_loss(model, xb, yb):
    digit_logits, odd_logit = model(xb)
    # BUG 1: softmax before cross-entropy. F.cross_entropy applies log-softmax itself; feeding probabilities
    #        squashes the "logits" into [0, 1], so the loss can never go below ~ln(10) - 1 + small and gradients vanish.
    loss_digit = F.cross_entropy(digit_logits, yb)
    # BUG 5: float labels with cross-entropy. With a (B,) input, F.cross_entropy treats the whole batch as ONE
    #        unbatched example with B "classes" and the float (B,) labels as an (unnormalized) probability target,
    #        i.e. -sum_n t_n log softmax(z)_n over the batch. It runs, but it trains a batch-level ranking, so
    #        "logit > 0 means odd" is meaningless. One logit + float 0/1 labels needs BCE-with-logits.
    loss_odd = F.binary_cross_entropy_with_logits(odd_logit, (yb % 2).float())
    return loss_digit, loss_odd


def train_step(model, opt, xb, yb):
    model.train()
    opt.zero_grad()  # BUG 2: missing zero_grad. .grad accumulates across steps, so each update used the sum of all past gradients
    loss_digit, loss_odd = compute_loss(model, xb, yb)
    (loss_digit + loss_odd).backward()
    opt.step()
    return loss_digit.item(), loss_odd.item()


def predict(model, x):
    digit_logits, odd_logit = model(x)
    # BUG 3: softmax over dim=0 normalizes each class column across the BATCH, which changes the per-row argmax
    #        (and makes a prediction depend on the other examples in the batch). Classes live on dim=1.
    return digit_logits.argmax(dim=1), (odd_logit > 0).long()


def evaluate(model, X, y):
    model.eval()  # BUG 4: evaluated in train mode: Dropout was active (noisy predictions) and BatchNorm used
    #               val-batch statistics AND overwrote its running stats with val data
    with torch.no_grad():
        pred_digit, pred_odd = predict(model, X)
    return (pred_digit == y).float().mean().item(), (pred_odd == y % 2).float().mean().item()


torch.manual_seed(0)
model = Net().to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
for epoch in range(5):
    order = torch.randperm(len(X_tr))
    for i in range(0, len(X_tr), 128):
        b = order[i:i + 128]
        ld, lo = train_step(model, opt, X_tr[b].to(dev), y_tr[b].to(dev))
    acc_digit, acc_odd = evaluate(model, X_val.to(dev), y_val.to(dev))
    print(f"epoch {epoch}: last loss digit {ld:.3f} odd {lo:.3f} | val acc digit {acc_digit:.3f} odd {acc_odd:.3f}")

# %% [markdown]
# ## Self-check
# The code below names the bugs it probes for, so its source is collapsed: **run it, but don't read it** until you are done.
# Probes each function on fresh models and probe batches, independently of the other bugs where possible.
# It prints the status of every bug, then fails if any remain.

# %% jupyter={"source_hidden": true}
def probe_net(seed=1):
    torch.manual_seed(seed)
    return Net().to(dev)


Xp, yp = X_val[:512].to(dev), y_val[:512].to(dev)
status = {}

# BUG 6: normalization statistics must come from the training rows only
tr_raw = X_raw[idx_tr]
mu_t, sd_t = torch.as_tensor(mu).float().cpu(), torch.as_tensor(sd).float().cpu()
ref_mu, ref_sd = (tr_raw.mean(), tr_raw.std()) if mu_t.numel() == 1 else (tr_raw.mean(0), tr_raw.std(0))
big = ref_sd > 1e-3
status["6 leakage: normalization stats from train rows only"] = bool(
    torch.allclose(mu_t, ref_mu, atol=1e-7, rtol=1e-6)
    and torch.allclose(sd_t.expand_as(ref_sd)[big], ref_sd[big], rtol=1e-4))

# BUG 1 and BUG 5: loss values on a probe batch (eval mode, so no dropout noise)
m = probe_net().eval()
with torch.no_grad():
    ld, lo = compute_loss(m, Xp, yp)
    dl, ol = m(Xp)
not_probs = not (dl.min() >= 0 and torch.allclose(dl.sum(1), torch.ones(len(dl), device=dev)))
status["1 softmax before cross-entropy"] = bool(not_probs and torch.isclose(ld, F.cross_entropy(dl, yp), rtol=1e-4))
odd = yp % 2
if ol.shape[-1:] == (2,):
    ref_odd = F.cross_entropy(ol, odd)
else:
    ref_odd = F.binary_cross_entropy_with_logits(ol.reshape(-1), odd.float())
status["5 float labels for cross-entropy"] = bool(torch.isclose(lo, ref_odd, rtol=1e-4))

# BUG 2: the second of two consecutive train_steps must move the params by exactly lr * (that step's gradient)
m = probe_net()
opt_p = torch.optim.SGD(m.parameters(), lr=0.1)
torch.manual_seed(10); train_step(m, opt_p, Xp[:128], yp[:128])
m_ref = copy.deepcopy(m)
before = [p.detach().clone() for p in m.parameters()]
torch.manual_seed(11); train_step(m, opt_p, Xp[128:256], yp[128:256])
torch.manual_seed(11); m_ref.train()
grads = torch.autograd.grad(sum(compute_loss(m_ref, Xp[128:256], yp[128:256])), list(m_ref.parameters()))
status["2 missing zero_grad"] = all(torch.allclose((b - p.detach()) / 0.1, g, rtol=1e-3, atol=1e-5)
                                    for b, p, g in zip(before, m.parameters(), grads))

# BUG 3: predicted class = argmax of the logits along the class dimension, per example
m = probe_net().eval()
with torch.no_grad():
    pd, _ = predict(m, Xp)
    status["3 wrong dim"] = bool(torch.equal(pd.reshape(-1), m(Xp)[0].argmax(1)))

# BUG 4: evaluate must run the model in eval mode and must not touch BatchNorm running stats
m = probe_net().train()
modes = []
hooks = [mod.register_forward_hook(lambda mod_, i, o: modes.append(mod_.training))
         for mod in m.modules() if isinstance(mod, (nn.Dropout, nn.BatchNorm1d))]
rm_before = [b.clone() for n, b in m.named_buffers() if "running" in n]
evaluate(m, Xp, yp)
for h in hooks:
    h.remove()
rm_after = [b for n, b in m.named_buffers() if "running" in n]
status["4 eval in train mode"] = bool(modes) and not any(modes) and all(torch.equal(a, b) for a, b in zip(rm_before, rm_after))

remaining = [k for k, ok in status.items() if not ok]
for k, ok in status.items():
    print(f"{'fixed ' if ok else 'BUG   '} {k}")
print(f"\n{6 - len(remaining)}/6 fixed")
check("all 6 bugs fixed", not remaining, f"still present: {remaining}")
check("end to end: val accuracy > 0.94 on both heads", acc_digit > 0.94 and acc_odd > 0.94,
      f"digit {acc_digit:.3f}, odd {acc_odd:.3f}")

# %% [markdown]
# ## Say it out loud
# 1. For each bug: what symptom would you have seen in the loss curve or metrics, and which sanity check catches it first?
# 2. Why does softmax-before-CE still "train" a bit? What is the lowest loss it can reach with 10 classes?
# 3. `F.cross_entropy(logits_(B,), float_labels_(B,))` runs in torch 2.x. What does it compute? And with `(B, C)` logits?
# 4. When is NOT calling `zero_grad` intentional? What breaks if you then forget to scale the loss?
# 5. Which modules behave differently in `train()` vs `eval()`, and what does BatchNorm do to its buffers if you evaluate in train mode?
# 6. Here the leakage barely moved the numbers. Give examples where the same mistake is catastrophic.

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. (1) softmax→CE: the loss plateaus high and the initial loss is ≈ ln 10 but drops much more slowly than usual; caught by the "loss goes to
#    ~0 when overfitting one batch" test. (2) no `zero_grad`: effective step = sum of all past gradients, so loss oscillates or
#    explodes (with momentum it diverges quickly); caught by overfit-one-batch or by logging grad norms. (3) `dim=0`: train loss
#    fine but val accuracy oddly low and *batch-size dependent*; caught by a unit test `predict(x[:1])` vs `predict(x)[:1]`.
#    (4) train-mode eval: val metrics noisy between identical runs and BN stats polluted; caught by evaluating twice and
#    comparing. (5) float labels: odd-head accuracy ≈ chance even though its loss decreases; caught by a per-head metric and
#    by checking loss at init (it is ≈ ln B, not ln 2). (6) leakage: no visible symptom, only an optimistic val number. You catch it by
#    reading the code: fit every preprocessing step on train only.
# 2. The CE input is in [0, 1], so the largest achievable logit gap is 1. Best case: input = one-hot, loss =
#    −log(e/(e+9)) = log(1 + 9/e) ≈ 1.46. Gradients are also scaled by the softmax Jacobian, which vanishes once predictions are confident.
# 3. Per the `CrossEntropyLoss` docs, a 1-D input is one unbatched example over C = B classes, and a float target of the same
#    shape is a class-probability vector. So it computes −Σₙ tₙ log softmax(z)ₙ, a softmax across the batch. With `(B, C)`
#    logits and float `(B,)` labels it raises "expected target dtype to be Long or Byte" (float targets must then be `(B, C)`
#    probabilities). The silent case only appears with a squeezed single-logit head, as here.
# 4. Gradient accumulation: you sum gradients over k micro-batches and step once. You must divide each micro-batch loss by k
#    (or by the token-count ratio). Otherwise the effective LR is k× larger (see P2.10b and P2.13).
# 5. Dropout (off in eval), BatchNorm (batch stats + running-stat updates in train; running stats in eval), and anything
#    that checks `self.training`. Evaluating in train mode normalizes with val-batch statistics and updates the running
#    mean/var with val data at momentum 0.1, which silently changes the model you later deploy.
# 6. Target/mean encoding with statistics computed on all rows; feature selection or PCA fitted on all data; time series
#    normalized with future statistics or randomly split; duplicate or near-duplicate images across splits; per-patient data
#    split by image rather than by patient. In each case val accuracy can be far above what deployment gives (Kaufman et al. 2012).
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
