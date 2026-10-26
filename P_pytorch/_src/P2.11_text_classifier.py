# %% [markdown]
# # P2.11 · Text classifier on AG News (vocab, padding + mask, mean pooling, 1-layer transformer)
#
# **ID:** P2.11 · **Target:** 35 min · **Plan week:** 4 · **Part:** P (PyTorch first principles)
#
# **The drill.** Classify news headlines + descriptions into 4 topics (World, Sports, Business, Sci/Tech). Build the
# text pipeline yourself: tokenizer → vocabulary with `<pad>`=0 and `<unk>`=1 → variable-length `Dataset` →
# `collate_fn` that pads and returns a mask → two models sharing the same pipeline:
# (A) `nn.Embedding(padding_idx=0)` + **masked mean pooling** (a fastText-style bag of embeddings), and
# (B) the same plus learned positions and a **1-layer `nn.TransformerEncoder`** with `src_key_padding_mask`.
#
# **Must include (from the plan):** tokenize, `nn.Embedding`, padding + mask, mean-pooling or 1-layer transformer
# encoder (here: both). Also: a unit check that **masked mean pooling is invariant to extra padding** (and that
# naive mean pooling is not), initial loss ≈ ln 4, overfit-one-batch, whole-set eval under `eval()` + `no_grad()`.
#
# **Rules.** No docs, no LLM, no tokenizer libraries. Data: `DATA/"ag_news"/{train,test}.parquet` (columns
# `text`, `label`). Subsample train for speed as specified. Each solution cell keeps only its spec.
#
# ## Sources
# - X. Zhang, J. Zhao, Y. LeCun, *Character-level Convolutional Networks for Text Classification*, NeurIPS 2015,
#   arXiv:1509.01626 — https://arxiv.org/abs/1509.01626 (introduces the AG News benchmark: 4 classes, 120k/7.6k).
# - AG News dataset card (label ids 0 World, 1 Sports, 2 Business, 3 Sci/Tech): https://huggingface.co/datasets/fancyzhx/ag_news
# - A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, *Bag of Tricks for Efficient Text Classification* (fastText),
#   EACL 2017, arXiv:1607.01759 — https://arxiv.org/abs/1607.01759 (averaged word embeddings + linear classifier).
# - A. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017, arXiv:1706.03762 — https://arxiv.org/abs/1706.03762
# - PyTorch docs: `nn.Embedding` (`padding_idx`: that row gets no gradient)
#   https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html ·
#   `nn.TransformerEncoder` (`src_key_padding_mask`: True = ignore; `enable_nested_tensor`)
#   https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html ·
#   `nn.utils.rnn.pad_sequence` https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
# - PyTorch tutorial *Text classification with the torchtext library* (EmbeddingBag on AG News; the pipeline shape
#   this drill rebuilds without torchtext): https://docs.pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

# %%
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from prep import DATA, Session, check, check_close, seed_everything

s = Session("P2.11", target_min=35)
seed_everything(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]

# %% tags=["solution"]
# 1. DATA. Read train/test parquet. Shuffle train with df.sample(frac=1, random_state=0); take the first 20 000 rows as
#    train_df and the next 5 000 as val_df; test_df = full test (7 600). Print class balance and 2 examples.
train_full = pd.read_parquet(DATA / "ag_news" / "train.parquet").sample(frac=1, random_state=0).reset_index(drop=True)
train_df, val_df = train_full.iloc[:20000], train_full.iloc[20000:25000]
test_df = pd.read_parquet(DATA / "ag_news" / "test.parquet")
print(len(train_df), len(val_df), len(test_df), train_df.label.value_counts().sort_index().tolist())
for t, l in train_df[["text", "label"]].values[:2]:
    print(CLASSES[l], "|", t[:120])

# %% tags=["solution"]
# 2. TOKENIZER + VOCAB. tokenize(text) -> list[str]: lowercase, then re.findall(r"[a-z0-9]+(?:'[a-z]+)?", ...).
#    Build the vocab from TRAIN only: Counter over all train tokens; keep tokens with count >= 2, most frequent first,
#    at most 30 000 -> itos = ["<pad>", "<unk>", ...]; stoi = {tok: i}. PAD, UNK = 0, 1.
#    encode(text, max_len=128) -> list[int] (unknown -> UNK, truncate to max_len). Print vocab size, OOV rate on val.
PAD, UNK = 0, 1
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


counts = Counter(tok for t in train_df.text for tok in tokenize(t))
itos = ["<pad>", "<unk>"] + [w for w, c in counts.most_common(30000) if c >= 2]
stoi = {w: i for i, w in enumerate(itos)}


def encode(text, max_len=128):
    return [stoi.get(tok, UNK) for tok in tokenize(text)][:max_len]


val_ids = [encode(t) for t in val_df.text]
oov = sum(i == UNK for ids in val_ids for i in ids) / sum(len(ids) for ids in val_ids)
print(f"vocab {len(itos)}, val OOV rate {oov:.3f}, mean length {np.mean([len(x) for x in val_ids]):.1f}")

# %% tags=["solution"]
# 3. DATASET + COLLATE. `TextDataset(df)`: stores encoded LongTensors and labels; __getitem__ -> (ids_1d, label).
#    collate(batch) -> (ids (B, T) long padded with PAD to the batch's max length, mask (B, T) bool with
#    True = real token, labels (B,) long). Use nn.utils.rnn.pad_sequence(batch_first=True, padding_value=PAD).
#    DataLoaders: train batch 64 shuffle; val/test batch 512; all with collate_fn=collate.
class TextDataset(Dataset):
    def __init__(self, df):
        self.ids = [torch.tensor(encode(t), dtype=torch.long) for t in df.text]
        self.labels = torch.tensor(df.label.values, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate(batch):
    seqs, labels = zip(*batch)
    ids = nn.utils.rnn.pad_sequence(list(seqs), batch_first=True, padding_value=PAD)
    return ids, ids != PAD, torch.stack(labels)


train_ds, val_ds, test_ds = TextDataset(train_df), TextDataset(val_df), TextDataset(test_df)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
val_dl = DataLoader(val_ds, batch_size=512, collate_fn=collate)
test_dl = DataLoader(test_ds, batch_size=512, collate_fn=collate)
ids, mask, yb = next(iter(train_dl))
print(ids.shape, mask.shape, mask.sum(1)[:8].tolist(), yb[:8].tolist())

# %% tags=["solution"]
# 4. MODEL A: `MeanPoolClassifier(vocab_size, d=128, n_classes=4)`. forward(ids, mask) -> (B, 4) logits.
#    nn.Embedding(vocab_size, d, padding_idx=PAD) -> masked mean over time:
#    sum(E * mask[..., None], dim=1) / mask.sum(1, keepdim=True).clamp(min=1)  (an all-pad row must not give NaN)
#    -> Dropout(0.2) -> Linear(d, n_classes). Write the pooling as a standalone `masked_mean(x, mask)`.
def masked_mean(x, mask):
    m = mask.unsqueeze(-1).to(x.dtype)  # (B, T, 1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, d=128, n_classes=4, p=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(d, n_classes)

    def forward(self, ids, mask):
        return self.head(self.drop(masked_mean(self.emb(ids), mask)))

# %% tags=["solution"]
# 5. MODEL B: `TransformerClassifier(vocab_size, d=128, n_heads=4, max_len=128, n_classes=4)`.
#    token Embedding(padding_idx=PAD) + learned position Embedding(max_len, d) (positions 0..T-1);
#    ONE nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=256, dropout=0.1, batch_first=True, norm_first=True)
#    wrapped in nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False);
#    call it with src_key_padding_mask = ~mask (True = PAD = ignore). Then LayerNorm, masked_mean, Linear.
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d=128, n_heads=4, max_len=128, n_classes=4):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=256, dropout=0.1, batch_first=True, norm_first=True)
        # enable_nested_tensor=False: be explicit. (The nested-tensor eval fast path needs norm_first=False anyway; when it
        # runs, outputs at PAD positions come back as exact zeros, unlike train mode. Masked pooling never reads them.)
        self.enc = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, ids, mask):
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.tok(ids) + self.pos(pos)[None]
        h = self.enc(h, src_key_padding_mask=~mask)
        return self.head(masked_mean(self.norm(h), mask))

# %% tags=["solution"]
# 6. TRAIN / EVAL. evaluate(model, dl) -> (mean loss, accuracy) over the whole set (eval, no_grad).
#    fit(model, epochs, lr) -> history {"train_loss", "val_acc"}: AdamW (wd=1e-2), CE, clip grad-norm 1.0,
#    keep the best-val-acc state. Before training each model: record its initial val loss (≈ ln 4).
#    Overfit check: a fresh TransformerClassifier in eval mode, one train batch, Adam lr=1e-3, 200 steps ->
#    `overfit_loss` < 0.05.
#    model_mean = fit(MeanPoolClassifier, 5 epochs, lr=3e-3); model_tf = fit(TransformerClassifier, 5 epochs, lr=1e-3).
#    Store `init_losses = {"mean": ..., "tf": ...}`, `val_accs = {...}` (best), test accuracy `test_accs = {...}` (once).
loss_fn = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    tot, correct, n = 0.0, 0, 0
    for ids, mask, y in dl:
        ids, mask, y = ids.to(dev), mask.to(dev), y.to(dev)
        logits = model(ids, mask)
        tot += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return tot / n, correct / n


def fit(model, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    history, best, best_state = {"train_loss": [], "val_acc": []}, -1.0, None
    for epoch in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for ids, mask, y in train_dl:
            ids, mask, y = ids.to(dev), mask.to(dev), y.to(dev)
            loss = loss_fn(model(ids, mask), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        _, va = evaluate(model, val_dl)
        history["train_loss"].append(tot / n)
        history["val_acc"].append(va)
        print(f"  epoch {epoch + 1} train {tot / n:.3f} val acc {va:.4f}")
        if va > best:
            best, best_state = va, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


V = len(itos)
tmp = TransformerClassifier(V).to(dev).eval()
tmp_opt = torch.optim.Adam(tmp.parameters(), lr=1e-3)
ids, mask, y = (t.to(dev) for t in next(iter(train_dl)))
for _ in range(200):
    loss = loss_fn(tmp(ids, mask), y)
    tmp_opt.zero_grad()
    loss.backward()
    tmp_opt.step()
overfit_loss = loss.item()
print(f"overfit-one-batch (transformer) {overfit_loss:.4f}")

init_losses, val_accs, test_accs, histories = {}, {}, {}, {}
for name, cls, lr in [("mean", MeanPoolClassifier, 3e-3), ("tf", TransformerClassifier, 1e-3)]:
    torch.manual_seed(0)
    m = cls(V).to(dev)
    init_losses[name] = evaluate(m, val_dl)[0]
    print(f"{name}: {sum(p.numel() for p in m.parameters()):,} params, init val loss {init_losses[name]:.3f} (ln4 {math.log(4):.3f})")
    m, histories[name] = fit(m, epochs=5, lr=lr)
    val_accs[name] = max(histories[name]["val_acc"])
    test_accs[name] = evaluate(m, test_dl)[1]
    globals()["model_" + name] = m
print("val", val_accs, "test", test_accs)

# %% [markdown]
# ## Self-check
# Expects: `tokenize`, `itos`, `stoi`, `encode`, `PAD`, `UNK`, `collate`, `train_ds`, `masked_mean`, `model_mean`,
# `model_tf` (both `forward(ids, mask) -> (B, 4)`), `init_losses`, `overfit_loss`, `val_accs`, `test_accs`.
# The padding-invariance check appends 7 PAD columns to a real batch and requires identical logits.

# %%
check("special tokens", itos[PAD] == "<pad>" and itos[UNK] == "<unk>" and PAD == 0 and UNK == 1)
check("unknown words map to <unk>", encode("qwxzvjk blorptastic") == [UNK, UNK])
_train_toks = {t for txt in train_df.text for t in tokenize(txt)}
_test_only = {t for txt in test_df.text for t in tokenize(txt)} - _train_toks
check("vocab built from train only (no test-only words)", len(_test_only) > 0 and not any(w in stoi for w in _test_only))
batch = [train_ds[i] for i in range(5)]
ids, mask, y = collate(batch)
check("collate shapes", ids.shape == mask.shape and ids.shape[0] == 5 and ids.shape[1] == max(len(b[0]) for b in batch))
check("mask marks real tokens", torch.equal(mask.sum(1), torch.tensor([len(b[0]) for b in batch])) and mask.dtype == torch.bool)
check("padding uses PAD", (ids[~mask] == PAD).all().item())
x = torch.randn(3, 6, 4)
mk = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
check_close("masked_mean row 0", masked_mean(x, mk)[0], x[0, :3].mean(0))
check("masked_mean all-pad row is finite", torch.isfinite(masked_mean(x, mk)).all().item())


def pad_more(ids, mask, k=7):
    return (torch.cat([ids, torch.full((ids.shape[0], k), PAD, dtype=ids.dtype)], 1),
            torch.cat([mask, torch.zeros(mask.shape[0], k, dtype=torch.bool)], 1))


ids, mask, _ = next(iter(val_dl))
ids, mask = ids[:64], mask[:64]
ids2, mask2 = pad_more(ids, mask)
for name, m in [("mean-pool", model_mean), ("transformer", model_tf)]:
    m.eval()
    emb = next(mm for mm in m.modules() if isinstance(mm, nn.Embedding))
    check(f"{name}: token Embedding has padding_idx=0", emb.padding_idx == PAD)
    with torch.no_grad():
        a = m(ids.to(dev), mask.to(dev))
        b = m(ids2.to(dev), mask2.to(dev))
    check(f"{name}: logits (B, 4)", a.shape == (64, 4))
    check_close(f"{name}: invariant to extra padding", b, a, atol=1e-4, rtol=1e-4)
with torch.no_grad():
    e = next(mm for mm in model_mean.modules() if isinstance(mm, nn.Embedding))
    naive = lambda i: e(i.to(dev)).mean(1)  # mean over ALL positions, pads included
    check("naive (unmasked) mean is NOT padding-invariant", not torch.allclose(naive(ids), naive(ids2), atol=1e-4))
for k in ("mean", "tf"):
    check(f"{k}: initial loss ~ ln 4", abs(init_losses[k] - math.log(4)) < 0.2, f"got {init_losses[k]:.3f}")
check("can overfit one batch", overfit_loss < 0.05, f"got {overfit_loss:.4f}")
check("mean-pool test acc > 0.85", test_accs["mean"] > 0.85, f"got {test_accs['mean']:.3f}")
check("transformer test acc > 0.80", test_accs["tf"] > 0.80, f"got {test_accs['tf']:.3f}")

# %% [markdown]
# ## Say it out loud (P3)
# 1. Why build the vocabulary on train only? What happens to test words that are not in it?
# 2. What does `padding_idx=0` do (forward and backward)? Is it enough on its own to make pooling padding-safe?
# 3. Why must the mean divide by the number of *real* tokens? What breaks if you divide by T?
# 4. `src_key_padding_mask`: which value means "ignore"? What would go wrong if you passed `mask` instead of `~mask`?
# 5. Why does the transformer need position embeddings while the mean-pool model doesn't? Does the transformer beat the bag of embeddings here, and why might it not?
# 6. Why pad per batch in `collate_fn` instead of padding everything to the global max length? What further trick reduces padding?
# 7. Where would `ignore_index` come in if this were a token-level task (e.g., tagging or LM)?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. The vocabulary is part of the model. Building it on val/test leaks their word distribution and over-states
#    coverage. Unseen words map to `<unk>`, which has its own trained embedding. Train on `<unk>` as well (here
#    words with count 1 become `<unk>`), so the model knows what "rare word" looks like.
# 2. Row 0 is initialized to zeros and gets no gradient, so PAD embeds to the zero vector forever. That alone is not
#    padding-safe: zeros still enter a plain mean (the denominator T grows), attention (unless masked), and the
#    position embeddings (pos + 0 ≠ 0). Masking is what guarantees invariance. `padding_idx` is a convenience.
# 3. Dividing by T shrinks the pooled vector by len/T, so the same sentence gets different features depending on
#    the longest sentence in its batch. Predictions then depend on batching (train/eval mismatch). The "naive mean"
#    check above shows exactly that. Clamp the count at 1 so an empty text gives 0 instead of NaN.
# 4. In PyTorch's `src_key_padding_mask`, `True` means "do not attend to this key". So pass `~mask` where `mask`
#    is True for real tokens. Passing `mask` would let queries attend only to pads. Rows that are all True (no real
#    keys) give NaN from softmax over all −inf. Never read the encoder output at pad positions. In eval, the
#    nested-tensor fast path (post-norm layers) returns exact zeros there, while train mode returns computed values
#    (measured on torch 2.14). Masked pooling makes this irrelevant.
# 5. Self-attention is permutation-equivariant, so without positions the encoder sees a bag of words too. A mean
#    pool is order-invariant by design. On AG News topic labels are mostly decided by keywords, so a bag of
#    embeddings is already strong. fastText (Joulin et al., Table 1) reports 91.5 % on AG News without bigrams and
#    92.5 % with them, trained on the full 120k set. Here, on 20k examples, the 1-layer transformer lost by about
#    4 points (≈ 85 % vs ≈ 89 % test). It has to learn attention and positions from little data, overfits faster,
#    and word order adds little for topic labels.
# 6. Global padding wastes compute on pad tokens (attention is O(T²)). Per-batch padding pads only to the longest
#    sequence in the batch. Bucketing (sorting by length within chunks, then shuffling the batches) cuts padding
#    further. Packing several sequences into one row with a block-diagonal mask removes it entirely.
# 7. For per-token losses, set the targets at pad positions to −100 (`ignore_index`) so they contribute neither loss
#    nor gradient, and the mean is taken over real tokens only. For an LM, also shift targets by one.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")  # score: 1-5 self-rating; notes: what tripped you up
