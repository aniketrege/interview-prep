# interview-prep

Daily practice material for ML research internship and job interviews: timed PyTorch drills (Part P), coding
fundamentals (Part A), a study map onto Raschka's books (Part C), and first-principles systems labs (Part G1). Every
notebook has a blank **practice** version and an executed **solution**, self-checks against trusted references,
interview questions with model answers, and a `## Sources` list at the top.

It follows a personal 16-week prep guide (not included in this repo). Part letters (P, A, B–G) and "the guide" in the
notebooks refer to that plan, and [schedule.yaml](schedule.yaml) is its day-by-day version.

| Folder | What | Start here |
| --- | --- | --- |
| [P_pytorch/](P_pytorch) | Part P: the "blank Colab" end-to-end drills P0, P2.1–P2.15 | [P0.00_skeleton](P_pytorch/P0.00_skeleton.ipynb), [P2.01](P_pytorch/P2.01_moons_mlp.ipynb) |
| [A_coding/](A_coding) | Part A: A1 DSA patterns + [problem list](A_coding/A1_problem_list.md), A2 tensor fluency, A3 ML-coding classics | [A1.00 session template](A_coding/A1.00_session_template.ipynb), [A3.10 attention](A_coding/A3.10_attention.ipynb) |
| [C_llms/](C_llms/README.md) | Part C: where each C1–C6 topic lives in `LLMs-from-scratch` / `reasoning-from-scratch`, plus gaps | [C_llms/README.md](C_llms/README.md) |
| [G1_systems/](G1_systems/README.md) | Part G1: GPUs, memory/FLOPs, FlashAttention, Triton, compile, collectives, DDP/FSDP/TP/PP/CP/EP, DiLoCo, serving | [G1_systems/README.md](G1_systems/README.md) |
| [practice/](practice) | your dated attempts, `log.csv`, weekly `notes/` (commit this) | |
| [tools/](tools) | **`today.py` (daily entry point)**, new session, progress, build / verify / citation check | |

## Setup (once)

```bash
cd interview-prep     # this repo
./setup.sh            # isolated .venv (torch 2.14.1+cu130, triton 3.8), Jupyter kernel, datasets -> data/
```
Links to `../LLMs-from-scratch` and `../rlhf-book` expect those repos checked out next to this one
([rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) @ f9313ea,
[natolambert/rlhf-book](https://github.com/natolambert/rlhf-book) @ eecc49e). They work locally, not on GitHub.
In VS Code pick the interpreter `interview-prep/.venv/bin/python` (or the kernel "Python 3 (interview-prep)").
The `.venv` keeps the drills independent of any system or conda Python (pinned in `requirements.lock.txt`).

## Start here every day

```bash
cd interview-prep
.venv/bin/python tools/today.py --open
```

That one command prints today's plan (week N of 16, starting Mon Oct 26, 2026). It copies fresh, blank notebooks for
today's drill / build / DSA / lab into `practice/<date>/` and creates this week's notes file
`practice/notes/week-NN.md`. Open the copies in VS Code, work top to bottom, and **run the last cell of each notebook**:
it logs your time, docs lookups and 1–5 self-score to `practice/log.csv`. The log drives the spaced-review line at the
bottom of `today.py` and the 20-min "speed" drill it picks for you from week 7 on.

```bash
.venv/bin/python tools/today.py --week                 # the whole week at a glance (plan Sunday evenings)
.venv/bin/python tools/today.py --date 2026-11-14      # preview any day
.venv/bin/python tools/progress.py --all               # every drill: tries, best time, streak, next review
.venv/bin/python tools/new_session.py A3.10            # extra practice copy of any notebook, any day
```

The plan lives in [schedule.yaml](schedule.yaml): the guide's weekly table, day by day, with the changes below. Edit
it freely, for example when an interview gets booked (guide rule 4). Entries without a notebook (most RL / agents /
retrieval builds in weeks 7–14) name what to study.

**How a day is meant to go** (guide "Daily rhythm", ≈ 2.5–3 h): Part P drill (45–60 min in weeks 1–6, a 20-min speed
round after) → main build (75–90 min) → one DSA problem (25 min) → theory note + 5 flashcards in the week's notes file
(20–30 min) → 10 min redoing one item from the spaced-review line. Saturday = mini-project (3 h); Sunday = G1 lab (≈ 2 h)
+ plan the next week with `--week`; from week 10, Sunday also has a 45-min mock.

- Do drills in the **practice** copy, out loud, with no docs. Open `solutions/` only after the self-check passes,
  or after you've been stuck for 10 minutes.
- **Targets** in each header are *mastery* times (the guide's progress bar: P2.1–P2.4 < 25 min, P2.7–P2.10 < 15 min).
  First attempts typically take 2–3×. That's expected; the log tracks the trend.
- An attempt counts as "clean" when score ≥ 4, lookups = 0, and time ≤ target. Clean attempts push the next review out
  (1 → 2 → 4 → 7 → 14 → 30 → 60 days); anything else brings it back tomorrow.
- G1 labs are study sessions, not timed drills. Re-do their exercises and interview questions as spaced review.

## Is the plan reasonable? Review and changes (Oct 22, 2026)

**Verdict: yes, the structure is sound, but change the timing for this season.** Code-first builds, a daily PyTorch
drill, depth on the two target areas (RL post-training; multimodal agents and search), and the mock checklist match what research-intern loops test. Candidate reports
I found (anonymous and self-reported, so signal rather than spec; the full Glassdoor/1point3acres texts sit behind
login walls, so only search snippets and summaries were readable) consistently mention two things. First,
"implement single/multi-head or cross-attention in PyTorch, then discuss time and memory complexity" as an ML-coding
ask (TikTok/ByteDance research-intern, xAI, and startup MLE screens, e.g. this
[1point3acres MHA phone-screen report](https://www.1point3acres.com/interview/thread/1159880)). Second, Meta
research-scientist loops that combine an online screen of LeetCode mediums, coding rounds, ML fundamentals/design,
research rounds and a 50–60 min research talk.
Recommended changes:

1. **Start DSA now, not in week 7.** Meta-style loops reportedly open with an online screen of LeetCode mediums, and
   interviews can land from November. Do one problem a day (25 min, [A1.00 template](A_coding/A1.00_session_template.ipynb)) from week 1,
   following [A1_problem_list.md](A_coding/A1_problem_list.md) (Blind 75 first), and go to the guide's full DSA block from week 7.
2. **Pull the most-asked ML-coding drills forward to weeks 2–3:** A3.10 attention/MHA/GQA, A3.13 KV cache, A3.11
   sampling, A3.17 DPO/GRPO losses. They're short (15–25 min) and appear across companies. Keep Part P as the main block.
3. **Run G1 as a weekly thread, not one week.** Systems topics are easy to postpone, and research-heavy startups weight them heavily.
   One lab per weekend from week 2 (order in [G1_systems/README.md](G1_systems/README.md)) leaves week 15 for consolidation.
4. **Add two rounds the guide misses.** First, an *AI-assisted coding* round. Meta has piloted one since Oct 2025
   (60 min, CoderPad with a built-in assistant, a multi-file project, judged on how well you direct and check the AI;
   [interviewing.io guide](https://interviewing.io/blog/how-to-use-ai-in-meta-s-ai-assisted-coding-interview-with-real-prompts-and-examples)).
   It's documented for SWE loops, so ask your recruiter whether research loops use it; practise one drill a week with an
   assistant while narrating what you accept or reject. Second, a **45–60 min research talk**: the guide only has 5- and 20-min versions.
5. **Budget realistically.** Five new builds a week plus drill plus notes at 2.5 h/day is tight. Two builds will not
   fit in a day as written: week 4's Saturday "GPT-2 on a tiny corpus (Raschka ch. 2–5)" and week 8's "PPO-RLHF loop with
   KL on a small LM". Spread Raschka ch. 2–4 over the week-4 weekday evenings (see the [C map](C_llms/README.md#suggested-order-fits-plan-weeks-4-6)), and
   start week 8 from `../rlhf-book/code/policy_gradients` instead of from scratch.
6. **Repo convention:** the guide's `impl.py / test_impl.py / NOTES.md` per build becomes one notebook with
   built-in checks plus a separate solution. `today.py --open` creates a weekly `practice/notes/week-NN.md` for the half-page theory notes and flashcards.
7. **Small fixes:** reasoning-from-scratch is a separate repo from LLMs-from-scratch, so clone it next to it (see the
   [C map](C_llms/README.md)). The RTX 3090 reference machine can't run FP8, so FP8 is conceptual in G1.2. Everything in the
   materials table was checked and resolves (links below).

### What changed in the schedule (week 1 = Oct 26, 2026)

All of the above is built into [schedule.yaml](schedule.yaml), which `tools/today.py` reads:

| Weeks | Dates | Focus | Changes vs the guide's table |
| --- | --- | --- | --- |
| 1–3 | Oct 26 – Nov 15 | Part P drills, tensor fluency, classical ML, training mechanics | daily DSA from day 1 (A1.0–A1.5); A3.10 attention, A3.13 KV cache, A3.11 sampling, A3.17 RM/DPO/GRPO pulled forward; Sundays G1.1, G1.2, G1.4 |
| 4–6 | Nov 16 – Dec 6 | transformers, inference, SFT/DPO | Raschka ch. 2–5 spread over week 4; Thanksgiving (Thu Nov 26) is a light day; Sundays G1.3, G1.10, G1.11 |
| 7–8 | Dec 7 – 20 | RL fundamentals, PPO/RLHF/GRPO | PPO-RLHF starts from `rlhf-book/code/policy_gradients`; Sundays G1.12, G1.6 |
| 9 | Dec 21 – 27 | holiday review (guide's week 10) | swapped with the guide's week 9 so the review week covers Christmas |
| 10–14 | Dec 28 – Jan 31 | multimodal, agentic RL, retrieval, agents/search, compute-adaptive | mocks start Sundays in week 10; Sundays G1.5, G1.13, G1.14, G1.20, G1.15 |
| 15–16 | Feb 1 – 14 | systems from memory, mock loops | G1.21 + G1.16 in week 15 |


## Notebook catalog

Generated by `python tools/catalog.py` from the notebook headers; "Verified" is the last `tools/verify.py` result.

<!-- catalog:start -->
#### Part P · PyTorch first principles

| ID | Drill | Target | Plan week | Practice | Solution | Verified |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | The 45-minute skeleton (memorize this order) | 45 min | 1 (Saturday: write it from memory), 6, 9 and 16 (full mock runs on unseen datasets) | [practice](P_pytorch/P0.00_skeleton.ipynb) | [solution](P_pytorch/solutions/P0.00_skeleton.ipynb) | pass 2026-10-26 |
| P2.1 | MLP binary classifier on two moons | 25 min | 1 (and timed repeats) | [practice](P_pytorch/P2.01_moons_mlp.ipynb) | [solution](P_pytorch/solutions/P2.01_moons_mlp.ipynb) | pass 2026-10-22 |
| P2.2 | MLP on FashionMNIST (+ confusion matrix by hand) | 25 min | 1 (and timed repeat in week 2) | [practice](P_pytorch/P2.02_mnist_mlp.ipynb) | [solution](P_pytorch/solutions/P2.02_mnist_mlp.ipynb) | pass 2026-10-22 |
| P2.3 | MLP regression on California housing (MSE vs Huber, RMSE and R² by hand) | 25 min | 1 (timed repeat in week 3) | [practice](P_pytorch/P2.03_housing_regression.ipynb) | [solution](P_pytorch/solutions/P2.03_housing_regression.ipynb) | pass 2026-10-22 |
| P2.4 | Imbalanced binary classification (1 % positives) | 25 min | 2 (timed repeat in week 4) | [practice](P_pytorch/P2.04_imbalanced_binary.ipynb) | [solution](P_pytorch/solutions/P2.04_imbalanced_binary.ipynb) | pass 2026-10-22 |
| P2.5 | Multilabel classifier (per-class BCE, micro vs macro F1, per-class thresholds) | 20 min | 2 | [practice](P_pytorch/P2.05_multilabel.ipynb) | [solution](P_pytorch/solutions/P2.05_multilabel.ipynb) | pass 2026-10-22 |
| P2.6 | Small CNN on CIFAR-10 (conv/pool/BN, augmentation, OneCycle, early stopping) | 35 min | 3 (timed repeat in week 5) | [practice](P_pytorch/P2.06_cifar_cnn.ipynb) | [solution](P_pytorch/solutions/P2.06_cifar_cnn.ipynb) | pass 2026-10-22 |
| P2.7 | Losses by hand | 15 min | 1 (CE, BCE, MSE), 2 (KL, Huber, focal), 5 (full set) | [practice](P_pytorch/P2.07_losses_by_hand.ipynb) | [solution](P_pytorch/solutions/P2.07_losses_by_hand.ipynb) | pass 2026-10-22 |
| P2.8 | Layers by hand | 15 min | 3 | [practice](P_pytorch/P2.08_layers_by_hand.ipynb) | [solution](P_pytorch/solutions/P2.08_layers_by_hand.ipynb) | pass 2026-10-22 |
| P2.9 | Metrics by hand | 15 min | 2 (and again in 5) | [practice](P_pytorch/P2.09_metrics_by_hand.ipynb) | [solution](P_pytorch/solutions/P2.09_metrics_by_hand.ipynb) | pass 2026-10-22 |
| P2.10a | Bug hunt A: an MNIST training script with 6 planted bugs | 15 min | 3 | [practice](P_pytorch/P2.10a_bug_hunt.ipynb) | [solution](P_pytorch/solutions/P2.10a_bug_hunt.ipynb) | pass 2026-10-22 |
| P2.10b | Bug hunt B: a next-token model with 6 planted bugs | 15 min | 4 | [practice](P_pytorch/P2.10b_bug_hunt.ipynb) | [solution](P_pytorch/solutions/P2.10b_bug_hunt.ipynb) | pass 2026-10-22 |
| P2.11 | Text classifier on AG News (vocab, padding + mask, mean pooling, 1-layer transformer) | 35 min | 4 | [practice](P_pytorch/P2.11_text_classifier.ipynb) | [solution](P_pytorch/solutions/P2.11_text_classifier.ipynb) | pass 2026-10-22 |
| P2.12 | Small VAE on MNIST (reconstruction-loss choice, ELBO with reparameterization) | 35 min | 4 | [practice](P_pytorch/P2.12_vae_mnist.ipynb) | [solution](P_pytorch/solutions/P2.12_vae_mnist.ipynb) | pass 2026-10-22 |
| P2.13 | Training utilities | 30 min | 3 | [practice](P_pytorch/P2.13_training_utilities.ipynb) | [solution](P_pytorch/solutions/P2.13_training_utilities.ipynb) | pass 2026-10-22 |
| P2.14 | Hyperparameter debugging | 30 min | 4 | [practice](P_pytorch/P2.14_hparam_debugging.ipynb) | [solution](P_pytorch/solutions/P2.14_hparam_debugging.ipynb) | pass 2026-10-22 |
| P2.15 | Contrastive two-tower model (CLIP-style InfoNCE) | 30 min | 5 | [practice](P_pytorch/P2.15_contrastive_two_tower.ipynb) | [solution](P_pytorch/solutions/P2.15_contrastive_two_tower.ipynb) | pass 2026-10-22 |

#### Part A · Coding fundamentals

| ID | Drill | Target | Plan week | Practice | Solution | Verified |
| --- | --- | --- | --- | --- | --- | --- |
| A1.0 | LeetCode session template (copy for every problem) | 25 min | 1-16 (schedule.yaml: this template on day 1, then one 25-min problem every weekday) | [practice](A_coding/A1.00_session_template.ipynb) | [solution](A_coding/solutions/A1.00_session_template.ipynb) | pass 2026-10-26 |
| A1.1 | Arrays & hashing, two pointers, sliding window, prefix sums | 60 min | 1 (pattern notebook, Tue-Wed); second pass on its problems in week 7 (hashing, two pointers) and 8 (sliding window) | [practice](A_coding/A1.01_arrays_hashing_two_pointers.ipynb) | [solution](A_coding/solutions/A1.01_arrays_hashing_two_pointers.ipynb) | pass 2026-10-26 |
| A1.2 | Stacks, monotonic stacks, queues and deques | 60 min | 2 (pattern notebook, Mon); second pass in week 8 | [practice](A_coding/A1.02_stacks_queues.ipynb) | [solution](A_coding/solutions/A1.02_stacks_queues.ipynb) | pass 2026-10-26 |
| A1.3 | Binary search: bounds with an invariant, rotated arrays, search on the answer | 60 min | 2 (pattern notebook, Wed); second pass in week 8 | [practice](A_coding/A1.03_binary_search.ipynb) | [solution](A_coding/solutions/A1.03_binary_search.ipynb) | pass 2026-10-26 |
| A1.4 | Linked lists: reversal, fast/slow pointers, merge, dummy head | 60 min | 3 (pattern notebook, Mon); second pass in week 10 | [practice](A_coding/A1.04_linked_lists.ipynb) | [solution](A_coding/solutions/A1.04_linked_lists.ipynb) | pass 2026-10-26 |
| A1.5 | Trees and tries: DFS, BFS, BSTs, LCA, serialization, prefix trees | 60 min | 3 (pattern notebook, Wed); second pass in week 10 (trees) and 11 (tries) | [practice](A_coding/A1.05_trees_tries.ipynb) | [solution](A_coding/solutions/A1.05_trees_tries.ipynb) | pass 2026-10-26 |
| A1.6 | Heaps (top-k, k-way merge, two-heap median) and intervals (merge, insert, sweep line) | 60 min | 4 (pattern notebook, Mon); second pass in week 11 | [practice](A_coding/A1.06_heaps_intervals.ipynb) | [solution](A_coding/solutions/A1.06_heaps_intervals.ipynb) | pass 2026-10-26 |
| A1.7 | Graphs: BFS/DFS, topological sort, union-find, Dijkstra, cycle detection | 60 min | 4 (pattern notebook, Thu); second pass in weeks 12 (BFS/DFS, topo sort, union-find) and 13 (Dijkstra) | [practice](A_coding/A1.07_graphs.ipynb) | [solution](A_coding/solutions/A1.07_graphs.ipynb) | pass 2026-10-26 |
| A1.8 | Backtracking: subsets, permutations, combinations, pruning, constraint checking | 60 min | 5 (pattern notebook, Tue); second pass in week 13 | [practice](A_coding/A1.08_backtracking.ipynb) | [solution](A_coding/solutions/A1.08_backtracking.ipynb) | pass 2026-10-26 |
| A1.9 | Dynamic programming: top-down vs bottom-up, 1-D and 2-D | 60 min | 6 (pattern notebook, Mon); second pass in weeks 14 (1-D DP, greedy) and 15 (2-D DP) | [practice](A_coding/A1.09_dynamic_programming.ipynb) | [solution](A_coding/solutions/A1.09_dynamic_programming.ipynb) | pass 2026-10-26 |
| A1.10 | Bit manipulation and a complexity-analysis self-quiz | 60 min | 7 (pattern notebook, Mon); complexity analysis is used in every A1 notebook; timed mixed sets in week 16 | [practice](A_coding/A1.10_bits_complexity.ipynb) | [solution](A_coding/solutions/A1.10_bits_complexity.ipynb) | pass 2026-10-26 |
| A2.1 | Tensor-op puzzles (broadcasting, einsum, gather/scatter, strides) | 25 min | 1 | [practice](A_coding/A2.01_tensor_ops.ipynb) | [solution](A_coding/solutions/A2.01_tensor_ops.ipynb) | pass 2026-10-22 |
| A2.2 | Numerically stable softmax, log-softmax, logsumexp, cross-entropy | 15 min | 1 | [practice](A_coding/A2.02_stable_softmax.ipynb) | [solution](A_coding/solutions/A2.02_stable_softmax.ipynb) | pass 2026-10-22 |
| A2.3 | micrograd: a scalar autograd engine from scratch | 40 min | 1-3 (first pass week 1, timed repeats in weeks 2-3) | [practice](A_coding/A2.03_micrograd.ipynb) | [solution](A_coding/solutions/A2.03_micrograd.ipynb) | pass 2026-10-22 |
| A2.4 | Custom `torch.autograd.Function` with a manual backward | 25 min | 1-3 | [practice](A_coding/A2.04_custom_autograd_function.ipynb) | [solution](A_coding/solutions/A2.04_custom_autograd_function.ipynb) | pass 2026-10-22 |
| A2.5 | Manual backprop: Linear, ReLU, softmax+CE, LayerNorm, BatchNorm, a 2-layer MLP | 40 min | 1-3 | [practice](A_coding/A2.05_manual_backprop.ipynb) | [solution](A_coding/solutions/A2.05_manual_backprop.ipynb) | pass 2026-10-22 |
| A3.1 | Linear, ridge and logistic regression from scratch | 20 min | 2 | [practice](A_coding/A3.01_linear_logistic_regression.ipynb) | [solution](A_coding/solutions/A3.01_linear_logistic_regression.ipynb) | pass 2026-10-22 |
| A3.2 | k-means (k-means++ init, Lloyd) and k-NN from scratch | 20 min | 2 | [practice](A_coding/A3.02_kmeans_knn.ipynb) | [solution](A_coding/solutions/A3.02_kmeans_knn.ipynb) | pass 2026-10-22 |
| A3.3 | PCA via SVD, the covariance eigendecomposition, and Eckart–Young | 15 min | 2 | [practice](A_coding/A3.03_pca_svd.ipynb) | [solution](A_coding/solutions/A3.03_pca_svd.ipynb) | pass 2026-10-22 |
| A3.4 | conv2d (and max-pool) forward via `unfold` + matmul | 20 min | 3 | [practice](A_coding/A3.04_conv2d_unfold.ipynb) | [solution](A_coding/solutions/A3.04_conv2d_unfold.ipynb) | pass 2026-10-22 |
| A3.5 | Box IoU, NMS, GIoU, and ranking metrics (P@k, R@k, MRR, nDCG@k) | 20 min | 12 (vision / retrieval rounds), useful anytime | [practice](A_coding/A3.05_iou_nms_ranking_metrics.ipynb) | [solution](A_coding/solutions/A3.05_iou_nms_ranking_metrics.ipynb) | pass 2026-10-22 |
| A3.10 | Attention from scratch: SDPA, masks, MHA, cross-attention, GQA | 20 min | 4 | [practice](A_coding/A3.10_attention.ipynb) | [solution](A_coding/solutions/A3.10_attention.ipynb) | pass 2026-10-22 |
| A3.11 | Sampling: temperature, top-k, top-p, min-p, repetition penalty | 15 min | 5 | [practice](A_coding/A3.11_sampling.ipynb) | [solution](A_coding/solutions/A3.11_sampling.ipynb) | pass 2026-10-22 |
| A3.12 | Beam search with EOS handling and length normalization | 20 min | 5 | [practice](A_coding/A3.12_beam_search.ipynb) | [solution](A_coding/solutions/A3.12_beam_search.ipynb) | pass 2026-10-22 |
| A3.13 | KV cache and the incremental decoding loop | 25 min | 5 | [practice](A_coding/A3.13_kv_cache_decoding.ipynb) | [solution](A_coding/solutions/A3.13_kv_cache_decoding.ipynb) | pass 2026-10-22 |
| A3.14 | Byte-level BPE tokenizer: train, encode, decode | 25 min | 4 | [practice](A_coding/A3.14_bpe.ipynb) | [solution](A_coding/solutions/A3.14_bpe.ipynb) | pass 2026-10-22 |
| A3.15 | Contrastive losses: InfoNCE, CLIP, SigLIP | 15 min | 10 | [practice](A_coding/A3.15_contrastive_losses.ipynb) | [solution](A_coding/solutions/A3.15_contrastive_losses.ipynb) | pass 2026-10-26 |
| A3.16 | GAE and the PPO loss | 20 min | 7-8 | [practice](A_coding/A3.16_gae_ppo.ipynb) | [solution](A_coding/solutions/A3.16_gae_ppo.ipynb) | pass 2026-10-22 |
| A3.17 | Reward-model, DPO and GRPO losses | 25 min | 6 (RM, DPO), 8 (GRPO and variants) | [practice](A_coding/A3.17_rm_dpo_grpo.ipynb) | [solution](A_coding/solutions/A3.17_rm_dpo_grpo.ipynb) | pass 2026-10-22 |
| A3.18 | Speculative sampling: the acceptance rule, exactness, and expected speedup | 20 min | 5 | [practice](A_coding/A3.18_speculative_decoding.ipynb) | [solution](A_coding/solutions/A3.18_speculative_decoding.ipynb) | pass 2026-10-22 |

#### Part G1 · Training & inference systems

| ID | Drill | Target | Plan week | Practice | Solution | Verified |
| --- | --- | --- | --- | --- | --- | --- |
| G1.1 | The GPU, arithmetic intensity and the roofline | 120 min | 15 (read it once in week 4 alongside FLOP counting) | [practice](G1_systems/G1.01_gpu_roofline.ipynb) | [solution](G1_systems/solutions/G1.01_gpu_roofline.ipynb) | pass 2026-10-22 |
| G1.2 | Floating point, mixed precision and training-memory math | 120 min | 15 (sections 1-3 also fit week 3, "AMP + grad accumulation") | [practice](G1_systems/G1.02_memory_math_mixed_precision.ipynb) | [solution](G1_systems/solutions/G1.02_memory_math_mixed_precision.ipynb) | pass 2026-10-22 |
| G1.3 | Counting FLOPs, measuring MFU, estimating training cost | 90 min | 4 (FLOP/param counting, sections 1-2), 15 (MFU and the 7B estimate) | [practice](G1_systems/G1.03_flops_mfu.ipynb) | [solution](G1_systems/solutions/G1.03_flops_mfu.ipynb) | pass 2026-10-22 |
| G1.4 | Online softmax and FlashAttention, in PyTorch | 120 min | 15 (drill: online softmax from memory in 20 min) | [practice](G1_systems/G1.04_online_softmax_flashattention.ipynb) | [solution](G1_systems/solutions/G1.04_online_softmax_flashattention.ipynb) | pass 2026-10-22 |
| G1.5 | Triton kernels: vector add, fused softmax, matmul, FlashAttention-2 forward | 120 min | 15 (drill: Triton softmax) | [practice](G1_systems/G1.05_triton_kernels.ipynb) | [solution](G1_systems/solutions/G1.05_triton_kernels.ipynb) | pass 2026-10-22 |
| G1.6 | Profiling, benchmarking hygiene and `torch.compile` | 90 min | 15 (Saturday mini-project: "speed up your week-4 GPT") | [practice](G1_systems/G1.06_torch_compile_profiling.ipynb) | [solution](G1_systems/solutions/G1.06_torch_compile_profiling.ipynb) | pass 2026-10-22 |
| G1.10 | Collectives from first principles: semantics, ring all-reduce, α–β cost, measured bandwidth | 100 min | 15 | [practice](G1_systems/G1.10_collectives.ipynb) | [solution](G1_systems/solutions/G1.10_collectives.ipynb) | pass 2026-10-22 |
| G1.11 | Distributed data parallel from scratch: naive → overlapped → bucketed, vs `torch` DDP | 110 min | 15 | [practice](G1_systems/G1.11_ddp_from_scratch.ipynb) | [solution](G1_systems/solutions/G1.11_ddp_from_scratch.ipynb) | pass 2026-10-22 |
| G1.12 | ZeRO and FSDP: shard the optimizer yourself, then measure FSDP2 `fully_shard` | 110 min | 15 | [practice](G1_systems/G1.12_zero_fsdp.ipynb) | [solution](G1_systems/solutions/G1.12_zero_fsdp.ipynb) | pass 2026-10-22 |
| G1.13 | Tensor and sequence parallelism: Megatron's f/g operators by hand | 110 min | 15 | [practice](G1_systems/G1.13_tensor_sequence_parallel.ipynb) | [solution](G1_systems/solutions/G1.13_tensor_sequence_parallel.ipynb) | pass 2026-10-22 |
| G1.14 | Pipeline parallelism: schedules, bubbles, activation memory, and a real send/recv pipeline | 100 min | 15 | [practice](G1_systems/G1.14_pipeline_parallel.ipynb) | [solution](G1_systems/solutions/G1.14_pipeline_parallel.ipynb) | pass 2026-10-22 |
| G1.15 | Context and expert parallelism: ring attention, Ulysses, and all-to-all MoE dispatch | 120 min | 15 | [practice](G1_systems/G1.15_context_expert_parallel.ipynb) | [solution](G1_systems/solutions/G1.15_context_expert_parallel.ipynb) | pass 2026-10-22 |
| G1.16 | Parallelism planning: a memory + communication calculator and interview design problems | 100 min | 15-16 | [practice](G1_systems/G1.16_parallelism_planning.ipynb) | [solution](G1_systems/solutions/G1.16_parallelism_planning.ipynb) | pass 2026-10-22 |
| G1.20 | Inference serving: decode is memory-bound, batching, KV-cache memory, prefix caching | 120 min | 15 (with F4 "cost accounting" in week 14) | [practice](G1_systems/G1.20_inference_serving.ipynb) | [solution](G1_systems/solutions/G1.20_inference_serving.ipynb) | pass 2026-10-22 |
| G1.21 | Low-communication training: from local SGD to DiLoCo (+ compressed pseudo-gradients) | 110 min | 15-16 | [practice](G1_systems/G1.21_diloco_low_comm.ipynb) | [solution](G1_systems/solutions/G1.21_diloco_low_comm.ipynb) | pass 2026-10-22 |
<!-- catalog:end -->

## How the material is verified

- **Solutions are executed end to end** on this machine (4× RTX 3090, torch 2.14.1, CUDA 13.0) by
  `python tools/verify.py`. Results are in [VERIFICATION.md](VERIFICATION.md), and outputs are saved in `solutions/*.ipynb`.
- **Self-checks compare against trusted references**: PyTorch/torchvision/sklearn built-ins, brute force, finite
  differences, or analytic values. They were tested to fail on typical wrong answers.
- **Citations are machine-checked**: `python tools/check_citations.py` resolves every arXiv ID through the arXiv API
  and compares the title with the citing text, checks every URL, and checks every relative link. See [tools/citation_report.md](tools/citation_report.md).
- Adapted code keeps attribution (repo, commit SHA, path, license) in the notebook header.
- To add or change a drill, see [AUTHORING.md](AUTHORING.md). Edit `_src/*.py`, then run `tools/build_notebooks.py` and `tools/verify.py`.

## Materials (guide's table, links checked Oct 22, 2026)

| Part | Code-first | Theory |
| --- | --- | --- |
| A | [Grind 75](https://www.techinterviewhandbook.org/grind75/) · [NeetCode 150](https://neetcode.io/practice) · Karpathy [Zero to Hero](https://karpathy.ai/zero-to-hero.html) · [Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles) | CLRS 4e |
| B | [d2l.ai](https://d2l.ai/) | [Bishop & Bishop, *Deep Learning* (2024)](https://www.bishopbook.com/) · [Murphy, *PML*](https://probml.github.io/pml-book/) · [Goodfellow et al.](https://www.deeplearningbook.org/) |
| C | [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (local) · [nanoGPT](https://github.com/karpathy/nanoGPT) · [minbpe](https://github.com/karpathy/minbpe) | see [C map](C_llms/README.md) |
| C6, D4 | [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) · [TRL](https://github.com/huggingface/trl) | DeepSeekMath arXiv:2402.03300 · DeepSeek-R1 arXiv:2501.12948 · DAPO arXiv:2503.14476 · Dr. GRPO arXiv:2503.20783 |
| D1–D2 | [Spinning Up](https://spinningup.openai.com/en/latest/) · [CleanRL](https://github.com/vwxyzjn/cleanrl) | [Sutton & Barto 2e](http://incompleteideas.net/book/the-book-2nd.html) · PPO arXiv:1707.06347 · GAE arXiv:1506.02438 |
| D3 | [TRL](https://github.com/huggingface/trl) · [HF RLHF blog](https://huggingface.co/blog/rlhf) · `../rlhf-book/code` (local) | InstructGPT arXiv:2203.02155 · DPO arXiv:2305.18290 · [RLHF book](https://rlhfbook.com/) |
| D4–D5 | [verl](https://github.com/volcengine/verl) · [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) · [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | |
| E | [open_clip](https://github.com/mlfoundations/open_clip) · [LLaVA](https://github.com/haotian-liu/LLaVA) · [timm](https://github.com/huggingface/pytorch-image-models) | ViT arXiv:2010.11929 · CLIP arXiv:2103.00020 · SigLIP arXiv:2303.15343 · LLaVA arXiv:2304.08485 · Flamingo arXiv:2204.14198 |
| F1 | [FAISS](https://github.com/facebookresearch/faiss) · [rank_bm25](https://github.com/dorianbrown/rank_bm25) · [sentence-transformers](https://www.sbert.net/) | DPR arXiv:2004.04906 · ColBERT arXiv:2004.12832 · Matryoshka arXiv:2205.13147 · HNSW arXiv:1603.09320 |
| F2–F4 | [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) · [Terminal-Bench](https://github.com/laude-institute/terminal-bench) | ReAct arXiv:2210.03629 · Tree of Thoughts arXiv:2305.10601 · Search-R1 arXiv:2503.09516 · Snell et al. arXiv:2408.03314 · Let's Verify arXiv:2305.20050 |
| G1 | [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) · [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) · [Modal GPU Glossary](https://modal.com/gpu-glossary) · [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) · [CS336](https://cs336.stanford.edu/) | see [G1 README](G1_systems/README.md) |
| G2–G4 | [interviewing.io](https://interviewing.io/) | Keshav, [*How to Read a Paper*](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf) (CCR 2007) |

## License

MIT for the original material ([LICENSE](LICENSE)). Some notebooks copy or adapt code from other open-source projects
(MIT, Apache-2.0, BSD-3-Clause, CC-BY-4.0). Those parts keep their licenses, and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists each one, with the full license texts. No datasets are included.
