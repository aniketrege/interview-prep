# %% [markdown]
# # P2.10b · Bug hunt B: a next-token model with 6 planted bugs
#
# **ID:** P2.10b · **Target:** 15 min · **Plan week:** 4 · **Part:** P
#
# **The drill.** The cell below trains a small causal next-token model on synthetic token sequences, with gradient
# accumulation. It **runs without errors** and the loss even goes down, but it has **6 bugs**. All six differ from P2.10a.
# Fix them in place, then run the self-check. It calls `lm_loss`, `train_step` and `fit` on probe inputs (with a
# recording optimizer and forward hooks) and prints which bugs remain. Keep names and signatures, and keep
# `LM(dropout=...)` constructible.
#
# **The data has a known answer.** Tokens follow $x_t = (x_{t-1} + x_{t-2}) \bmod V$ with probability 0.9, otherwise a
# uniform random token. So the best possible next-token loss is computable (cell below). A loss far *below* that floor
# is as much a red flag as a loss stuck at $\ln V$.
#
# **Rules.** Read the whole script first. Hyperparameters and style are not bugs.
#
# <details><summary>Hint (areas, not answers)</summary>
# one in logging, two in the gradient-accumulation step, two in the loss function, one in the train/eval mode handling.
# </details>
#
# ## Sources
# - PyTorch FAQ, "My model reports cuda runtime error(2): out of memory" (`total_loss += loss` accumulates autograd
#   history; use `loss.item()` / `float(loss)`): https://docs.pytorch.org/docs/stable/notes/faq.html
# - `Optimizer.zero_grad` (default `set_to_none=True`; optimizers skip params whose `.grad` is None):
#   https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
# - PyTorch AMP examples, "Gradient accumulation" (loss divided by `iters_to_accumulate`):
#   https://docs.pytorch.org/docs/stable/notes/amp_examples.html
# - `CrossEntropyLoss` input layout `(N, C, d1, ...)` (class dim second):
#   https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html ·
#   `Tensor.view` vs `permute`/`transpose`: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
# - `Module.train` / `Module.eval`: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
# - Next-token shift as in A. Karpathy, nanoGPT `train.py`/`model.py` (MIT): https://github.com/karpathy/nanoGPT and
#   S. Raschka, LLMs-from-scratch ch. 5 (Apache-2.0), local copy `../../LLMs-from-scratch` (commit f9313ea),
#   `ch05/01_main-chapter-code/ch05.ipynb` (`calc_loss_batch` flattens logits and targets).

# %%
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prep import Session, check, seed_everything

s = Session("P2.10b", target_min=15)
seed_everything(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V, T, P_RULE = 16, 32, 0.9


def make_sequences(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, V, (n, T), generator=g)
    for t in range(2, T):
        rule = (x[:, t - 1] + x[:, t - 2]) % V
        keep = torch.rand(n, generator=g) < P_RULE
        x[:, t] = torch.where(keep, rule, x[:, t])
    return x


train_tokens, val_tokens = make_sequences(4000, 0).to(dev), make_sequences(500, 1).to(dev)
# Best achievable mean next-token CE: target 1 is uniform (ln V); targets 2..T-1 follow the rule w.p. 0.9 + 0.1/V.
p_hit, p_miss = P_RULE + (1 - P_RULE) / V, (1 - P_RULE) / V
h_rule = -(p_hit * math.log(p_hit) + (V - 1) * p_miss * math.log(p_miss))
floor = (math.log(V) + (T - 2) * h_rule) / (T - 1)
print(f"loss at init ~ ln V = {math.log(V):.3f}; best achievable val loss = {floor:.3f}")

# %% tags=["template_only"]
# ===== the script under review: find and fix the 6 bugs in this cell =====
class LM(nn.Module):
    """Causal 3-token-window model: logits at position t depend on tokens t-2, t-1, t only."""

    def __init__(self, vocab=V, d=32, hidden=256, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(3 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, vocab)

    def forward(self, tokens):                       # (B, T) -> (B, T, V) logits
        e = self.emb(tokens)
        ctx = torch.cat([e, F.pad(e, (0, 0, 1, 0))[:, :-1], F.pad(e, (0, 0, 2, 0))[:, :-2]], dim=-1)  # [e_t, e_{t-1}, e_{t-2}]
        return self.head(self.drop(self.mlp(self.drop(ctx))))


def lm_loss(logits, tokens):
    B, T_, V_ = logits.shape
    return F.cross_entropy(logits.view(B, V_, T_), tokens)


def train_step(model, opt, tokens, accum_steps=4):
    total = 0.0
    for chunk in tokens.chunk(accum_steps):
        loss = lm_loss(model(chunk), chunk)
        loss.backward()
        total += loss / accum_steps
    opt.zero_grad()
    opt.step()
    return total


@torch.no_grad()
def evaluate(model, tokens):
    model.eval()
    return lm_loss(model(tokens), tokens).item()


def fit(model, opt, train_tokens, val_tokens, epochs=6, steps_per_epoch=100, batch_size=64):
    history = {"train_loss": [], "val_loss": []}
    model.train()
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            idx = torch.randint(0, len(train_tokens), (batch_size,), device=train_tokens.device)
            history["train_loss"].append(train_step(model, opt, train_tokens[idx]))
        history["val_loss"].append(evaluate(model, val_tokens))
        print(f"epoch {epoch}: train {sum(history['train_loss'][-steps_per_epoch:]) / steps_per_epoch:.3f}"
              f" val {history['val_loss'][-1]:.3f}")
    return history


torch.manual_seed(0)
model = LM().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
history = fit(model, opt, train_tokens, val_tokens)

# %% tags=["solution_only"]
# ===== fixed script; every change is marked BUG n =====
class LM(nn.Module):
    """Causal 3-token-window model: logits at position t depend on tokens t-2, t-1, t only."""

    def __init__(self, vocab=V, d=32, hidden=256, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(3 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, vocab)

    def forward(self, tokens):                       # (B, T) -> (B, T, V) logits
        e = self.emb(tokens)
        ctx = torch.cat([e, F.pad(e, (0, 0, 1, 0))[:, :-1], F.pad(e, (0, 0, 2, 0))[:, :-2]], dim=-1)  # [e_t, e_{t-1}, e_{t-2}]
        return self.head(self.drop(self.mlp(self.drop(ctx))))


def lm_loss(logits, tokens):
    B, T_, V_ = logits.shape
    # BUG 4: no shift. Position t must predict token t+1, so drop the last logit and the first target.
    #        Unshifted, the model sees the token it is asked to predict: it learns to copy, and the loss
    #        falls far below the entropy floor printed above.
    # BUG 5: `view(B, V, T)` to get CE's (N, C, d) layout reinterprets memory and scrambles classes and positions.
    #        It needs permute(0, 2, 1) (or flatten both to (B*T, V) and (B*T,), as here).
    return F.cross_entropy(logits[:, :-1].reshape(-1, V_), tokens[:, 1:].reshape(-1))


def train_step(model, opt, tokens, accum_steps=4):
    model.train()  # BUG 6 (other half of the fix): be in train mode for every step, not just the first epoch
    opt.zero_grad()  # BUG 3: zero_grad sat between backward and step, so step() saw no gradients (.grad=None => no update)
    total = 0.0
    for chunk in tokens.chunk(accum_steps):
        loss = lm_loss(model(chunk), chunk) / accum_steps  # BUG 2: without /accum_steps the grads are k times the
        loss.backward()                                     #        big-batch gradient (LR silently multiplied by k)
        total += loss.item()  # BUG 1: `total += loss` kept every step's autograd graph alive and logged tensors
    opt.step()
    return total


@torch.no_grad()
def evaluate(model, tokens):
    model.eval()
    return lm_loss(model(tokens), tokens).item()


def fit(model, opt, train_tokens, val_tokens, epochs=6, steps_per_epoch=100, batch_size=64):
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()  # BUG 6: model.train() was called once before the loop; evaluate() switched to eval and
        #                 nothing switched back, so dropout was off for every epoch after the first
        for step in range(steps_per_epoch):
            idx = torch.randint(0, len(train_tokens), (batch_size,), device=train_tokens.device)
            history["train_loss"].append(train_step(model, opt, train_tokens[idx]))
        history["val_loss"].append(evaluate(model, val_tokens))
        print(f"epoch {epoch}: train {sum(history['train_loss'][-steps_per_epoch:]) / steps_per_epoch:.3f}"
              f" val {history['val_loss'][-1]:.3f}")
    return history


torch.manual_seed(0)
model = LM().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
history = fit(model, opt, train_tokens, val_tokens)

# %% [markdown]
# ## Self-check
# The code below names the bugs it probes for, so its source is collapsed: **run it, but don't read it** until you are done.

# %% jupyter={"source_hidden": true}
class RecordingOptimizer:
    """Stands in for an optimizer: records the gradients present at every zero_grad() and step() call."""

    def __init__(self, params):
        self.params = list(params)
        self.param_groups = [{"params": self.params, "lr": 0.0}]
        self.events = []

    def _grads(self):
        return [None if p.grad is None else p.grad.detach().clone() for p in self.params]

    def zero_grad(self, set_to_none=True):
        self.events.append(("zero", self._grads()))
        for p in self.params:
            p.grad = None

    def step(self, closure=None):
        self.events.append(("step", self._grads()))


def nonzero(gs):
    return any(g is not None and g.abs().sum() > 0 for g in gs)


status = {}
torch.manual_seed(0)
probe_tokens = make_sequences(32, 7).to(dev)

# BUG 2 and BUG 3: what gradients does step() see, compared to the one-big-batch gradient?
m = LM(dropout=0.0).to(dev)
rec = RecordingOptimizer(m.parameters())
out = train_step(m, rec, probe_tokens, accum_steps=4)
steps = [g for kind, g in rec.events if kind == "step"]
status["3 zero_grad between backward and step"] = len(steps) == 1 and nonzero(steps[0])
acc_grads = steps[0] if steps and nonzero(steps[0]) else next((g for kind, g in reversed(rec.events) if nonzero(g)), None)
m_ref = copy.deepcopy(m)
m_ref.zero_grad()
ref = torch.autograd.grad(lm_loss(m_ref(probe_tokens), probe_tokens), list(m_ref.parameters()))
status["2 grad accumulation without dividing the loss"] = acc_grads is not None and all(
    ((a - r).norm() / r.norm()).item() < 1e-3 for a, r in zip(acc_grads, ref))

# BUG 4 and BUG 5: lm_loss on random logits, compared with the four shift x layout combinations
torch.manual_seed(1)
B_, T_ = 4, 12
lg, tk = torch.randn(B_, T_, V, device=dev), torch.randint(0, V, (B_, T_), device=dev)
got = lm_loss(lg, tk)
cand = {
    "ok": F.cross_entropy(lg[:, :-1].reshape(-1, V), tk[:, 1:].reshape(-1)),
    "noshift": F.cross_entropy(lg.reshape(-1, V), tk.reshape(-1)),
    "view": F.cross_entropy(lg[:, :-1].reshape(B_, V, T_ - 1), tk[:, 1:]),
    "noshift+view": F.cross_entropy(lg.view(B_, V, T_), tk),
}
match = [k for k, v in cand.items() if torch.isclose(got, v, rtol=1e-5)]
if not match:
    print("note: lm_loss matches none of the known variants (check the shift direction and the reduction)")
status["4 off-by-one: targets not shifted"] = bool(match) and match[0] in ("ok", "view")
status["5 view where permute is needed"] = bool(match) and match[0] in ("ok", "noshift")

# BUG 1 and BUG 6: a tiny probe run of fit(), with hooks recording the dropout mode of every forward pass
m = LM().to(dev)
modes = []
m.drop.register_forward_hook(lambda mod, i, o: modes.append((torch.is_grad_enabled(), mod.training)))
hist = fit(m, torch.optim.AdamW(m.parameters(), lr=1e-3), train_tokens[:256], val_tokens[:64],
           epochs=2, steps_per_epoch=2, batch_size=32)
is_plain = lambda v: isinstance(v, (float, int)) or (torch.is_tensor(v) and not v.requires_grad)
status["1 loss accumulated with its autograd graph"] = is_plain(out) and all(is_plain(v) for v in hist["train_loss"])
status["6 eval mode never switched back to train"] = all(train for grad, train in modes if grad) and \
    not any(train for grad, train in modes if not grad)

remaining = [k for k, ok in status.items() if not ok]
for k in sorted(status):
    print(f"{'fixed ' if status[k] else 'BUG   '} {k}")
print(f"\n{6 - len(remaining)}/6 fixed")
check("all 6 bugs fixed", not remaining, f"still present: {remaining}")
check("end to end: val loss within 0.15 nats of the floor", floor - 0.02 < history["val_loss"][-1] < floor + 0.15,
      f"val {history['val_loss'][-1]:.3f}, floor {floor:.3f}")

# %% [markdown]
# ## Say it out loud
# 1. Which of these bugs make the loss go *down faster* than it should? Which make it not move at all? Which are invisible in the curve?
# 2. `total += loss` vs `total += loss.item()` vs `total += loss.detach()`: what does each keep alive, and what does `.item()` cost on a GPU?
# 3. Show that gradient accumulation with `loss / k` gives the same gradient as one big batch. When is it *not* exactly equal?
# 4. Why is `x.view(B, V, T)` on a `(B, T, V)` tensor not a transpose? When does `view` raise, and when does it silently succeed?
# 5. Where exactly should `model.train()` / `model.eval()` live in a loop with periodic evaluation?
# 6. How did the printed entropy floor help? What other "known answer" sanity checks would you build into a new pipeline?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Faster than it should: no shift (copying makes the loss go towards 0, far below the floor) and no `/k` (k× effective LR, which
#    looks faster early and is unstable later). Not moving: `zero_grad` before `step`: `.grad` is None, so AdamW skips every
#    parameter and the loss stays at ≈ ln V. Invisible or subtle: the `view` scramble (it still trains, just to a worse loss),
#    eval-mode training (no dropout after epoch 1, so train loss drops a little and overfitting rises), and the graph-holding logger
#    (a slow memory leak and tensors in `history`).
# 2. `loss` is a tensor attached to its graph. Summing it builds one giant graph that references every step's
#    graph (the PyTorch FAQ's example of accumulating history), so memory grows with the step count. `.item()`
#    returns a Python float but forces a host-device sync every call. `.detach()` keeps a GPU tensor with no graph
#    (no sync, no leak). Sum detached values on the GPU and call `.item()` only when you log.
# 3. With equal-size micro-batches, mean loss over the full batch $= \frac1k\sum_j \bar\ell_j$, and gradients are linear, so the sum of
#    the gradients of $\bar\ell_j/k$ equals the big-batch gradient. It is not exact with unequal chunks (weight each by its size or count
#    tokens, e.g. with padding/`ignore_index`), with BatchNorm (batch statistics differ), and up to floating-point summation order.
# 4. `view` keeps the memory order and only reinterprets the shape, so element `[b, t, v]` lands at some unrelated
#    `[b, v', t']`. A transpose changes the strides. `view` raises when the requested shape is incompatible with the
#    strides (e.g. after `transpose` or a slice like `[:, :-1]`), and silently succeeds on a contiguous tensor of the same
#    numel. That is the dangerous case. `reshape` copies if needed but has the same semantics as `view`, so it does not fix this either.
# 5. Call `model.train()` at the start of each training phase (or inside `train_step`), and `model.eval()` inside `evaluate`
#    together with `no_grad`/`inference_mode`. Never rely on the mode left behind by a previous call. (With a cuDNN
#    `nn.GRU`/`nn.LSTM` on GPU this bug is loud: backward raises "cudnn RNN backward can only be called in training mode".
#    That is why this script uses an MLP over a 3-token window, where it is silent.)
# 6. A loss below the irreducible entropy means the model has access to the answer (leakage, no shift, a label in the input).
#    Others: initial loss ≈ ln V; overfit one batch to ≈ 0; a shuffled-label run should give chance; `predict(x[:1]) == predict(x)[:1]`
#    (no batch dependence); one optimizer step on a known quadratic; gradient check against finite differences.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
