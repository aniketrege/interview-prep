# %% [markdown]
# # A3.14 · Byte-level BPE tokenizer: train, encode, decode
#
# **ID:** A3.14 · **Target:** 25 min · **Plan week:** 4 · **Part:** A
#
# **The drill.** Guide A3 lists "BPE tokenizer train + encode" as a blank-editor classic, and C1 adds byte-level BPE,
# special tokens and "why tokenization breaks arithmetic". You write, from plain Python lists and dicts:
#
# 1. `get_stats` and `merge` (the two primitives),
# 2. `train_bpe`: repeatedly merge the most frequent adjacent pair of byte-level token ids,
# 3. `encode_bytes` / `decode`: apply learned merges by **rank** (earliest-learned first); bytes back to text,
# 4. GPT-2-style **regex pre-tokenization** (train and encode chunk by chunk) and **special tokens**.
#
# Sections 1-3 are the 25-minute core; section 4 is the follow-up an interviewer asks next.
# **Rules:** plain Python lists, dicts and bytes. No `tiktoken` / `tokenizers` inside your functions; section 4 may
# use the `regex` package (for the `\p{L}` / `\p{N}` classes). The checks compare your merges and token ids,
# **exactly**, with karpathy/minbpe's `BasicTokenizer` / `RegexTokenizer` trained on the same text and vocab size.
# To match exactly, **break ties between equally frequent pairs the way minbpe does**: `max(stats, key=stats.get)`
# returns the first pair in dict insertion order, i.e. the pair whose first occurrence in the current sequence is
# leftmost. Any deterministic rule is a valid BPE; this one just makes the comparison exact.
#
# **Deeper study:** Raschka's from-scratch BPE, simple and tiktoken-faithful versions,
# `../../LLMs-from-scratch/ch02/05_bpe-from-scratch/bpe-from-scratch-simple.ipynb` and
# `../../LLMs-from-scratch/ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb`; the original GPT-2 encoder vs tiktoken in
# `../../LLMs-from-scratch/ch02/02_bonus_bytepair-encoder/compare-bpe-tiktoken.ipynb`; Karpathy's tokenization lecture
# notes, https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/lecture.md.
#
# ## Sources
# - R. Sennrich, B. Haddow, A. Birch, *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016,
#   arXiv:1508.07909 — https://arxiv.org/abs/1508.07909 (BPE for subword vocabularies).
# - A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, *Language Models are Unsupervised Multitask
#   Learners* (GPT-2), OpenAI 2019, §2.2 "Input Representation" (byte-level BPE; "we prevent BPE from merging across
#   character categories ... We add an exception for spaces") —
#   https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
# - karpathy/minbpe @ 1acefe8, `minbpe/base.py` (`get_stats`, `merge`), `minbpe/basic.py` (`BasicTokenizer`),
#   `minbpe/regex.py` (`RegexTokenizer`, `GPT2_SPLIT_PATTERN`, `GPT4_SPLIT_PATTERN`), MIT licence, Copyright (c) 2024
#   Andrej Karpathy — https://github.com/karpathy/minbpe/tree/1acefe89412b20245db5a22d2a02001e547dc602 .
#   Copied (comments trimmed) into the hidden reference cell below; the test-only reference for every check.
# - openai/tiktoken 0.14.0, `tiktoken_ext/openai_public.py` (GPT-2: `explicit_n_vocab=50257`,
#   `<|endoftext|>` = 50256, the r50k split pattern) — https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
# - openai/gpt-2 `src/encoder.py` (original byte-level BPE encoder) — https://github.com/openai/gpt-2/blob/master/src/encoder.py
# - A. K. Singh, DJ Strouse, *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs*, 2024,
#   arXiv:2402.14903 — https://arxiv.org/abs/2402.14903 (GPT-3.5/4 tokenize numbers into 1-3 digit chunks left to
#   right; forcing right-to-left chunking with commas improves arithmetic accuracy substantially).
# - Training text: AG News (X. Zhang, J. Zhao, Y. LeCun, *Character-level Convolutional Networks for Text
#   Classification*, NeurIPS 2015, arXiv:1509.01626), from `prep.DATA / "ag_news"`, plus a hand-written multilingual
#   paragraph.

# %%
import random

import pandas as pd
import regex  # the `regex` package: supports \p{L} (letter), \p{N} (number) Unicode classes; `re` does not

from prep import DATA, Session, check

s = Session("A3.14", target_min=25)
news = pd.read_parquet(DATA / "ag_news" / "train.parquet", columns=["text"]).text
MULTI = ("Größe, façade, naïve café. 東京は日本の首都です。 Привет, мир! नमस्ते दुनिया. "
         "emoji: 🙂🚀👍🏽 and e\u0301 (e + combining acute) vs é. ")
train_text = "\n".join(news[:200]) + "\n" + MULTI * 8       # ~57 KB, mostly ASCII news
held_out = "\n".join(news[200:230]) + "\n" + MULTI
VOCAB_SIZE = 256 + 150                                      # 150 merges: trains in a few seconds in pure Python
print(f"train text: {len(train_text):,} chars, {len(train_text.encode('utf-8')):,} bytes")

# %% [markdown]
# <details><summary>Reference implementation (karpathy/minbpe, MIT). Used by the checks; don't read it before you finish.</summary>
#
# The next cell is collapsed in Jupyter. It defines `RefBasicTokenizer` and `RefRegexTokenizer`.
# </details>

# %% jupyter={"source_hidden": true}
# --- test-only reference: karpathy/minbpe @ 1acefe8 (MIT, (c) 2024 Andrej Karpathy), comments trimmed ---
# minbpe/base.py: get_stats, merge;  minbpe/basic.py: BasicTokenizer;  minbpe/regex.py: RegexTokenizer
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def _ref_get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _ref_merge(ids, pair, idx):
    newids, i = [], 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class RefBasicTokenizer:
    def train(self, text, vocab_size):
        ids = list(text.encode("utf-8"))
        merges, vocab = {}, {idx: bytes([idx]) for idx in range(256)}
        for i in range(vocab_size - 256):
            stats = _ref_get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = _ref_merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merges, self.vocab = merges, vocab

    def decode(self, ids):
        return b"".join(self.vocab[idx] for idx in ids).decode("utf-8", errors="replace")

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = _ref_get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _ref_merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        return self._encode_chunk(text.encode("utf-8"))


class RefRegexTokenizer(RefBasicTokenizer):
    def __init__(self, pattern=GPT2_SPLIT_PATTERN):
        self.pattern, self.compiled_pattern = pattern, regex.compile(pattern)
        self.special_tokens, self.inverse_special_tokens = {}, {}

    def train(self, text, vocab_size):
        ids = [list(ch.encode("utf-8")) for ch in regex.findall(self.compiled_pattern, text)]
        merges, vocab = {}, {idx: bytes([idx]) for idx in range(256)}
        for i in range(vocab_size - 256):
            stats = {}
            for chunk_ids in ids:
                _ref_get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [_ref_merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merges, self.vocab = merges, vocab

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        parts = [self.vocab[i] if i in self.vocab else self.inverse_special_tokens[i].encode("utf-8") for i in ids]
        return b"".join(parts).decode("utf-8", errors="replace")

    def encode_ordinary(self, text):
        out = []
        for chunk in regex.findall(self.compiled_pattern, text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def encode(self, text, allowed_special="none_raise"):
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            return self.encode_ordinary(text)
        out = []
        for part in regex.split("(" + "|".join(regex.escape(k) for k in special) + ")", text):
            out.extend([special[part]] if part in special else self.encode_ordinary(part))
        return out


ref = RefBasicTokenizer()
ref.train(train_text, VOCAB_SIZE)
print("reference trained:", len(ref.merges), "merges")

# %% [markdown]
# ## 1. The two primitives
# Text → UTF-8 bytes → a list of ints in `0..255`. `get_stats` counts adjacent pairs (overlapping: `[1,1,1]` has the
# pair `(1,1)` **twice**). `merge` scans left to right and replaces each **non-overlapping** occurrence of `pair` by the
# new id `idx` (`[1,1,1]` → `[idx, 1]`).

# %%
def get_stats(ids, counts=None):
    """ids: list[int]. Return dict {(a, b): count} of adjacent pairs, inserted in order of first occurrence.
    If `counts` is given, update it in place (and return it) — used to sum stats over regex chunks."""
    ### BEGIN SOLUTION
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
    ### END SOLUTION


def merge(ids, pair, idx):
    """Replace every non-overlapping occurrence of `pair` (scanning left to right) in `ids` by `idx`."""
    ### BEGIN SOLUTION
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
    ### END SOLUTION


check("get_stats basic", get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1})
check("get_stats counts overlapping pairs", get_stats([7, 7, 7]) == {(7, 7): 2})
check("get_stats insertion order = first occurrence", list(get_stats([5, 6, 1, 2, 5, 6])) == [(5, 6), (6, 1), (1, 2), (2, 5)])
acc = get_stats([1, 2]); get_stats([1, 2, 3], acc)
check("get_stats accumulates in place", acc == {(1, 2): 2, (2, 3): 1})
check("merge basic", merge([1, 2, 3, 1, 2], (1, 2), 256) == [256, 3, 256])
check("merge is left-to-right, non-overlapping", merge([7, 7, 7], (7, 7), 256) == [256, 7] and merge([7] * 4, (7, 7), 256) == [256, 256])
check("merge handles pair at the very end / no match", merge([3, 1, 2], (1, 2), 9) == [3, 9] and merge([1], (1, 2), 9) == [1])

# %% [markdown]
# ## 2. Training
# Start from `vocab = {i: bytes([i]) for i in range(256)}`. For `vocab_size - 256` steps: count pairs over the
# **current** ids, take the most frequent pair (ties → first in insertion order, i.e. `max(stats, key=stats.get)`),
# mint id `256 + step`, merge it everywhere, record `merges[pair] = id` and `vocab[id] = vocab[a] + vocab[b]`.
# Cost: $O(M \cdot N)$ for $M$ merges over $N$ bytes (fine here; production trainers update pair counts incrementally).

# %%
def train_bpe(text, vocab_size):
    """Return (merges, vocab): merges dict {(a, b): new_id} in rank order; vocab dict {id: bytes} of size vocab_size."""
    ### BEGIN SOLUTION
    assert vocab_size >= 256
    ids = list(text.encode("utf-8"))
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    for step in range(vocab_size - 256):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + step
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab
    ### END SOLUTION


merges, vocab = train_bpe(train_text, VOCAB_SIZE)
check("vocab size", len(vocab) == VOCAB_SIZE and len(merges) == VOCAB_SIZE - 256)
check("first merge matches minbpe", list(merges.items())[:1] == list(ref.merges.items())[:1],
      f"got {list(merges.items())[:1]}, minbpe {list(ref.merges.items())[:1]}")
check("all merges match minbpe, in order", list(merges.items()) == list(ref.merges.items()),
      next((f"first difference at rank {r}: got {a}, minbpe {b}" for r, (a, b)
            in enumerate(zip(merges.items(), ref.merges.items())) if a != b), "length differs"))
check("vocab bytes match minbpe", vocab == ref.vocab)
print("10 longest tokens:", sorted((vocab[i] for i in vocab), key=len)[-10:])

# %% [markdown]
# ## 3. Encoding and decoding
# **Encode** a byte string: start from its bytes and, while some adjacent pair is in `merges`, merge the pair with
# the **lowest rank** (smallest new id). Not the most frequent pair in this string, and not the leftmost one: the rank
# order is what makes encoding reproduce training. **Decode**: concatenate `vocab[id]` bytes and decode UTF-8 with
# `errors="replace"` (an arbitrary id sequence, e.g. a model sample cut mid-character, need not be valid UTF-8).

# %%
def encode_bytes(b, merges):
    """b: bytes. Return list[int] after applying merges in rank order."""
    ### BEGIN SOLUTION
    ids = list(b)
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids
    ### END SOLUTION


def encode(text, merges):
    return encode_bytes(text.encode("utf-8"), merges)


def decode(ids, vocab):
    """ids: list[int] -> str. Invalid UTF-8 becomes U+FFFD."""
    ### BEGIN SOLUTION
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
    ### END SOLUTION


toy = {(98, 99): 256, (97, 98): 257}                       # rank 0: "bc", rank 1: "ab"
check("encode uses rank, not position", encode("abc", toy) == [97, 256], f"got {encode('abc', toy)}")
check("encode chains merges", encode("aaaa", {(97, 97): 256, (256, 256): 257}) == [257])
check("encode empty / single byte", encode("", merges) == [] and encode("a", merges) == [97])
check("token ids match minbpe on held-out text", encode(held_out, merges) == ref.encode(held_out))
check("encode(train) == ids at the end of training", encode(train_text, merges) == ref.encode(train_text))
for t in ["hello world", "Größe naïve café", "東京は日本の首都です。", "🙂🚀👍🏽", "e\u0301 vs é", "", "  \n\t x  ", MULTI]:
    assert decode(encode(t, merges), vocab) == t, f"round trip failed on {t!r}"
check("round trip on unicode (CJK, emoji + skin tone, combining chars, whitespace)", True)
emoji = list("🙂".encode("utf-8"))                         # 4 bytes
check("decode of a partial character gives U+FFFD", decode(emoji[:2], vocab) == "\N{REPLACEMENT CHARACTER}", f"got {decode(emoji[:2], vocab)!r}")
check("decode matches minbpe on arbitrary ids", decode([0x80, 104, 105, 0xFF] + emoji[:3], vocab) == ref.decode([0x80, 104, 105, 0xFF] + emoji[:3]))
n_bytes = len(held_out.encode("utf-8"))
print(f"held-out compression: {n_bytes} bytes -> {len(encode(held_out, merges))} tokens "
      f"({n_bytes / len(encode(held_out, merges)):.2f} bytes/token)")

# %% [markdown]
# ## 4. GPT-2 regex pre-tokenization and special tokens
# Plain byte BPE learns tokens like `b"e t"` that straddle word boundaries and wastes slots on `dog.`, `dog!`, `dog?`.
# GPT-2 first splits text with a regex and runs BPE **inside each chunk only** (radford2019, §2.2):
# ```
# '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
# ```
# contractions · optional space + letters · optional space + digits · optional space + other symbols ·
# whitespace not followed by non-space (so the last space attaches to the next word: `" world"`) · remaining whitespace.
# Training sums pair counts over all chunks, merging never crosses a chunk boundary, encoding = concatenation of
# per-chunk encodings. (GPT-4's `cl100k` pattern, `GPT4_SPLIT_PATTERN` above, is case-insensitive for contractions and
# splits numbers into chunks of at most 3 digits.)
#
# **Special tokens** (`<|endoftext|>`, chat-template markers) get ids *after* the merges and are never produced by
# merging: you split the text on the special strings first and emit their ids directly. Doing that only when the caller
# opts in (`allowed_special`) is a safety feature: user-supplied text containing `"<|endoftext|>"` must not turn into a
# control token (tiktoken and minbpe raise by default).

# %%
def train_bpe_regex(text, vocab_size, pattern=GPT2_SPLIT_PATTERN):
    """Like train_bpe, but split `text` with regex.findall(pattern, text) first; pairs never cross chunks."""
    ### BEGIN SOLUTION
    chunks = [list(c.encode("utf-8")) for c in regex.findall(pattern, text)]
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    for step in range(vocab_size - 256):
        stats = {}
        for c in chunks:
            get_stats(c, stats)
        pair = max(stats, key=stats.get)
        idx = 256 + step
        chunks = [merge(c, pair, idx) for c in chunks]
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab
    ### END SOLUTION


def encode_regex(text, merges, pattern=GPT2_SPLIT_PATTERN, special_tokens=None, allowed_special="none_raise"):
    """special_tokens: dict {str: id}. allowed_special: "all" | "none" | "none_raise" (assert none present).
    Split on allowed special strings first (they map straight to their id), regex-split the rest, encode per chunk."""
    ### BEGIN SOLUTION
    special_tokens = special_tokens or {}
    if allowed_special == "all":
        special = special_tokens
    elif allowed_special == "none":
        special = {}
    elif allowed_special == "none_raise":
        assert not any(t in text for t in special_tokens), "special token in text; pass allowed_special"
        special = {}
    else:
        raise ValueError(allowed_special)
    parts = regex.split("(" + "|".join(regex.escape(t) for t in special) + ")", text) if special else [text]
    ids = []
    for part in parts:
        if part in special:
            ids.append(special[part])
        else:
            for chunk in regex.findall(pattern, part):
                ids.extend(encode_bytes(chunk.encode("utf-8"), merges))
    return ids
    ### END SOLUTION


rmerges, rvocab = train_bpe_regex(train_text, VOCAB_SIZE)
rref = RefRegexTokenizer(GPT2_SPLIT_PATTERN)
rref.train(train_text, VOCAB_SIZE)
check("regex merges match minbpe RegexTokenizer", list(rmerges.items()) == list(rref.merges.items()))
check("regex encode matches minbpe", encode_regex(held_out, rmerges) == rref.encode(held_out))
cross = [vocab[i] for i in merges.values() if regex.search(rb"[A-Za-z] ", vocab[i])]
check("plain BPE learned word-straddling tokens", len(cross) > 0, "expected some")
check("regex BPE never merges across a word boundary",
      not any(regex.search(rb"[A-Za-z] ", rvocab[i]) for i in rmerges.values()))
print("plain-BPE tokens straddling words:", cross[:6])

EOT = "<|endoftext|>"
special = {EOT: VOCAB_SIZE}                                 # next free id, like GPT-2's 50256
rref.register_special_tokens(special)
doc = "first doc." + EOT + "second doc, 2024."
check("special token -> one id", encode_regex(doc, rmerges, special_tokens=special, allowed_special="all")
      == rref.encode(doc, allowed_special="all") and VOCAB_SIZE in encode_regex(doc, rmerges, special_tokens=special, allowed_special="all"))
check("allowed_special='none' encodes it as ordinary text",
      encode_regex(doc, rmerges, special_tokens=special, allowed_special="none") == rref.encode(doc, allowed_special="none")
      and VOCAB_SIZE not in encode_regex(doc, rmerges, special_tokens=special, allowed_special="none"))
try:
    encode_regex(doc, rmerges, special_tokens=special)
    raised = False
except (AssertionError, ValueError):
    raised = True
check("default refuses text containing a special token", raised)
full_vocab = {**rvocab, VOCAB_SIZE: EOT.encode("utf-8")}
check("decode with special token", decode(encode_regex(doc, rmerges, special_tokens=special, allowed_special="all"), full_vocab) == doc)

# %% [markdown]
# ### Why tokenization breaks arithmetic (look at the chunks)
# How digits become tokens depends on corpus frequencies (GPT-2 pattern) or a fixed left-to-right 3-digit split
# (GPT-4 pattern). Either way, the same digit position lands in different tokens for different numbers, so the model
# cannot learn one carry rule per digit position. Singh & Strouse (arXiv:2402.14903) found GPT-3.5/4 arithmetic improves
# a lot when commas force right-to-left 3-digit groups. LLaMA and PaLM split numbers into single digits.

# %%
def pieces(text, merges, vocab, pattern=GPT2_SPLIT_PATTERN):
    return [vocab[i].decode("utf-8", errors="replace") for i in encode_regex(text, merges, pattern)]


# a numbers-heavy toy corpus (years, prices, counts) so that digit merges get learned
rng = random.Random(0)
num_text = " ".join(str(rng.choice([rng.randint(1990, 2025), rng.randint(0, 99), rng.randint(100, 99999)]))
                    for _ in range(4000))
nmerges, nvocab = train_bpe_regex(num_text, 256 + 60)
for t in ["2004", "2005 + 1996 = 4001", "1234567", "12345 + 67890"]:
    print(f"{t!r:20} GPT-2 split + merges learned on numbers: {pieces(t, nmerges, nvocab)}")
print("GPT-4 pre-split of 1234567:", regex.findall(GPT4_SPLIT_PATTERN, "1234567"), "(left to right)")
print("GPT-4 pre-split of 1,234,567:", regex.findall(GPT4_SPLIT_PATTERN, "1,234,567"))
check("GPT-4 pattern chunks digits left to right in 3s", regex.findall(GPT4_SPLIT_PATTERN, "1234567") == ["123", "456", "7"])

# %% [markdown]
# ## Say it out loud
# 1. Why byte-level BPE rather than character-level? What is the base vocabulary and what can never be out-of-vocabulary?
# 2. Walk through training. What is the complexity of your loop, and how do production trainers make it fast?
# 3. Why must encoding apply merges by rank rather than by frequency in the string being encoded?
# 4. GPT-2's vocabulary is 50,257. Where does that number come from?
# 5. What does the regex pre-tokenizer buy you, and what does `\s+(?!\S)` do?
# 6. How are special tokens handled, and why is "allowed_special" opt-in?
# 7. Name three failure modes caused by tokenization (arithmetic, spelling/character counting, multilingual cost, glitch tokens).
# 8. What happens if you decode the ids of a sample that was cut in the middle of a multi-byte character?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Unicode has >130k code points (GPT-2 paper §2.2), so a character-level base vocabulary is huge and still has
#    unseen characters. Bytes give a base of 256, so every string is encodable and nothing is OOV. The cost is that one
#    character can take up to 4 tokens before merges.
# 2. Loop: count adjacent pairs ($O(N)$), argmax, merge ($O(N)$), repeat $M$ times, so $O(MN)$. Fast trainers
#    (HF `tokenizers`, SentencePiece) count pairs once, keep a heap of counts plus an index from pair to positions, and on
#    each merge update only the counts of the neighbours of the merged occurrences. They also train on unique
#    pre-tokenized words weighted by frequency, not on the raw text.
# 3. Training applied merges in rank order to the whole corpus. A token at rank $r$ only takes part in pairs of rank
#    $> r$. Applying the lowest-rank pair present, repeatedly, therefore replays training, and
#    `encode(train_text)` equals the final training sequence (checked above). Choosing by frequency in the string, or
#    leftmost-first, gives different and inconsistent segmentations of the same word (the `"abc"` check).
# 4. 256 byte tokens + 50,000 learned merges + 1 special `<|endoftext|>` (id 50256). See tiktoken's `gpt2()`:
#    `explicit_n_vocab=50257`.
# 5. Merges can't cross character categories (letters / digits / punctuation / whitespace), so vocabulary slots don't
#    go to `dog.`, `dog!`, and so on. A leading space attaches to the following word (`" the"`), so a word and its
#    space-prefixed form are two tokens. `\s+(?!\S)` matches a whitespace run but leaves the *last* space to prefix the
#    next word. That keeps `" world"` one chunk even after multiple spaces.
# 6. They get reserved ids after the merges and are never learnable by merging. At encode time the text is split on the
#    allowed special strings and each one maps to its id. By default, encountering one is an error (or it is encoded as
#    plain text), so untrusted text can't inject `<|endoftext|>` or chat-role markers (a prompt-injection vector).
# 7. Arithmetic: digits are chunked inconsistently, so place value isn't aligned across numbers (Singh & Strouse 2024).
#    Spelling and counting letters: the model sees `"strawberry"` as 1-3 opaque tokens, not characters. Multilingual
#    cost: non-Latin scripts need 2-4x more tokens for the same content. That means less effective context and higher
#    API cost. Glitch tokens: merges learned from text that is rare in the LM's training data have under-trained
#    embeddings. Also, leading-space variants (`"Hello"` vs `" Hello"`) behave differently at prompt boundaries.
# 8. The concatenated bytes are not valid UTF-8, so `decode(errors="replace")` emits U+FFFD for the dangling bytes.
#    Streaming detokenizers therefore buffer bytes until a complete character is available.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
