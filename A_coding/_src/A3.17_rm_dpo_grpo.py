# %% [markdown]
# # A3.17 · Reward-model, DPO and GRPO losses
#
# **ID:** A3.17 · **Target:** 25 min · **Plan week:** 6 (RM, DPO), 8 (GRPO and variants) · **Part:** A
#
# **The drill.** The post-training losses you will be asked to write in a blank editor: per-sequence log-probs
# from logits (the step everyone gets wrong), the Bradley–Terry reward-model loss, the DPO loss with its implicit
# rewards, and GRPO: group-normalized advantages, the clipped token-level objective with a $k_3$ KL penalty, and the
# three loss-aggregation variants (GRPO / DAPO / Dr. GRPO) that change which tokens get how much gradient.
# The mock-interview checklist asks for **"DPO and GRPO losses in 10 min each"** from a blank file; the log-prob
# helper is shared by both and should take about 3 minutes.
#
# **Conventions.**
# - `input_ids` is prompt + response (+ right padding), shape `(B, T)`. `response_mask[b, t] = 1` iff token `t` is a
#   response token (aligned with `input_ids`, *not* yet shifted). Logits at position `t` predict token `t+1`.
# - Per-token tensors for GRPO (`logp`, `old_logp`, `ref_logp`, `mask`) are `(B, L)` over the **response** tokens
#   only, already shifted. One prompt's group of `G` responses lives in `G` consecutive rows.
# - Every loss is **minimized**.
#
# **Rules.** You may use `F.log_softmax`, `F.logsigmoid`, `gather`, reductions. Not inside your functions:
# `F.cross_entropy`, `F.binary_cross_entropy*` (the checks use them as references).
#
# **Deeper study:** rlhf-book `../../rlhf-book/code/direct_alignment/loss.py` (DPO and IPO/SimPO/ORPO/KTO side by
# side), `../../rlhf-book/code/policy_gradients/loss.py` and `../../rlhf-book/code/policy_gradients/utils.py`
# (GRPO, DAPO, Dr. GRPO, RLOO advantages and losses). LLMs-from-scratch
# `../../LLMs-from-scratch/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb` §4 (DPO loss on a GPT-2 model;
# note it averages log-probs over tokens, see Q1). reasoning-from-scratch ch06 (GRPO from scratch, §6.7-6.9:
# https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch06/01_main-chapter-code/ch06_main.ipynb)
# and ch07 (clipped ratios §7.4, KL term §7.5:
# https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch07/01_main-chapter-code/ch07_main.ipynb).
#
# ## Sources
# - R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, C. Finn, *Direct Preference Optimization: Your Language
#   Model is Secretly a Reward Model*, NeurIPS 2023, arXiv:2305.18290 — https://arxiv.org/abs/2305.18290 (loss: eq. 7;
#   optimal policy and reward reparametrization: eqs. 4-5;
#   implicit reward $\hat r_\theta=\beta\log\pi_\theta/\pi_{ref}$ and its gradient: §4 "What does the DPO update do?").
# - R. A. Bradley, M. E. Terry, *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*,
#   Biometrika 39 (1952) 324-345, doi:10.2307/2334029 — https://doi.org/10.2307/2334029
# - L. Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), NeurIPS 2022,
#   arXiv:2203.02155 — https://arxiv.org/abs/2203.02155 (reward-model loss $-\log\sigma(r_c - r_r)$, eq. 1).
# - Z. Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024,
#   arXiv:2402.03300 — https://arxiv.org/abs/2402.03300 (GRPO objective eq. 3 with $\frac1G\sum_i\frac1{|o_i|}\sum_t$;
#   KL estimator eq. 4; outcome-supervision advantages $(r_i-\mathrm{mean})/\mathrm{std}$ §4.1.2; β=0.04, 64 samples
#   per question §4.2).
# - Q. Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, 2025, arXiv:2503.14476 —
#   https://arxiv.org/abs/2503.14476 (token-level loss §3.3 eq. 12; clip-higher §3.1 with ε_low=0.2, ε_high=0.28 §4.1;
#   dynamic sampling §3.2; no KL term §2.3).
# - Z. Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO), 2025, arXiv:2503.20783 —
#   https://arxiv.org/abs/2503.20783 (length and difficulty biases §3.1; fix §3.2: drop $1/|o_i|$ and the std, use a
#   constant normalizer such as the generation budget).
# - J. Schulman, *Approximating KL Divergence* (blog, 2020) — http://joschu.net/blog/kl-approx.html ($k_3$).
# - A. Pal et al., *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*, 2024, arXiv:2402.13228 —
#   https://arxiv.org/abs/2402.13228 (DPO can lower the likelihood of the preferred response).
# - C. Zheng et al., *Group Sequence Policy Optimization*, 2025, arXiv:2507.18071 — https://arxiv.org/abs/2507.18071
#   (sequence-level ratio; mentioned in Q10).
# - N. Lambert, *RLHF Book* code, natolambert/rlhf-book@eecc49e1e1daf1be670d7242eb090240b5bb0f04 (code under MIT;
#   the policy-gradient losses are credited in-file to Z. Stojanovski, Apache-2.0):
#   `code/direct_alignment/loss.py` (`compute_logprobs`, `DPOLoss`), `code/policy_gradients/loss.py` (`GRPOLoss`,
#   `DAPOLoss`, `approx_kl3`, `masked_mean`), `code/policy_gradients/utils.py` (`compute_standardized_advantages`) —
#   https://github.com/natolambert/rlhf-book/tree/eecc49e1e1daf1be670d7242eb090240b5bb0f04/code . Transcribed into the
#   hidden reference cell.
# - Hugging Face TRL `trl/trainer/grpo_trainer.py` (Apache-2.0), huggingface/trl@ea8cb1b57bd29da6deb1fc2ee9f1f8ff77c7038e —
#   https://github.com/huggingface/trl/blob/ea8cb1b57bd29da6deb1fc2ee9f1f8ff77c7038e/trl/trainer/grpo_trainer.py
#   (group std with Bessel correction and `+1e-4`; per-token $k_3$; `loss_type` "grpo" / "dapo" / "dr_grpo", the last
#   normalized by `B * max_completion_length`). Transcribed into the hidden reference cell.
# - S. Raschka, *Build a Reasoning Model (From Scratch)* code, rasbt/reasoning-from-scratch@a788466 (Apache-2.0),
#   `ch07/03_rlvr_grpo_scripts_advanced/7_5_plus_kl.py` (sequence-level GRPO with clipping and a $k_1$ KL) —
#   https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch07/03_rlvr_grpo_scripts_advanced/7_5_plus_kl.py

# %%
import math

import torch
import torch.nn.functional as F

from prep import Session, check, check_close, seed_everything

s = Session("A3.17", target_min=25)
seed_everything(0)
torch.set_default_dtype(torch.float64)
g = torch.Generator().manual_seed(0)

# a right-padded batch: prompt lengths 2, 3, 1 and response lengths 3, 2, 5 inside T = 7
B, T, V = 3, 7, 11
prompt_len, resp_len = [2, 3, 1], [3, 2, 5]
input_ids = torch.randint(0, V, (B, T), generator=g)
response_mask = torch.zeros(B, T)
for b in range(B):
    response_mask[b, prompt_len[b]:prompt_len[b] + resp_len[b]] = 1
    input_ids[b, prompt_len[b] + resp_len[b]:] = 0  # pad id
logits = torch.randn(B, T, V, generator=g) * 2

# %% [markdown]
# ### Reference implementations (ground truth for the checks; do not read before you finish)
# Collapsed in Jupyter. Transcriptions of rlhf-book@eecc49e (MIT) and TRL@ea8cb1b (Apache-2.0), with config fields
# turned into arguments.

# %% jupyter={"source_hidden": true}
def _rlhf_compute_logprobs(logits, labels, mask, average_log_prob=False):  # rlhf-book direct_alignment/loss.py
    logits, labels, mask = logits[:, :-1, :], labels[:, 1:], mask[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) * mask
    if average_log_prob:
        return per_token_logps.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return per_token_logps.sum(dim=-1)


def _rlhf_dpo(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    losses = -F.logsigmoid(beta * (chosen_logratios - rejected_logratios))
    chosen_rewards, rejected_rewards = beta * chosen_logratios.detach(), beta * rejected_logratios.detach()
    return losses.mean(), chosen_rewards, rejected_rewards


def _trl_advantages(rewards_flat, num_generations, scale_rewards=True):  # TRL "group" scaling
    mean = rewards_flat.view(-1, num_generations).mean(dim=1).repeat_interleave(num_generations)
    std = rewards_flat.view(-1, num_generations).std(dim=1).repeat_interleave(num_generations)  # Bessel, as nanstd
    adv = rewards_flat - mean
    return adv / (std + 1e-4) if scale_rewards else adv


def _rlhf_masked_mean(tensor, mask, dim=None, eps=1e-8):
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)


def _rlhf_approx_kl3(log_probs, log_probs_ref, action_mask):
    log_ratio = (log_probs_ref - log_probs) * action_mask
    return torch.expm1(log_ratio) - log_ratio


def _rlhf_grpo_loss(log_probs, old, ref, adv, mask, eps_lo, eps_hi, beta):  # GRPOLoss.forward
    ratio = (log_probs - old).exp()
    policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - eps_lo, 1 + eps_hi) * adv)
    loss = policy_loss + beta * _rlhf_approx_kl3(log_probs, ref, mask)
    return _rlhf_masked_mean(loss, mask, dim=-1).mean(dim=0)


def _rlhf_dapo_loss(log_probs, old, adv, mask, eps_lo, eps_hi):  # DAPOLoss.forward
    ratio = (log_probs - old).exp()
    per_token_loss = -torch.min(ratio * adv, ratio.clamp(1 - eps_lo, 1 + eps_hi) * adv)
    return (per_token_loss * mask).sum() / mask.sum().clamp(1e-8)


def _trl_per_token_loss(per_token_logps, old_per_token_logps, ref_per_token_logps, advantages, eps_low, eps_high, beta):
    coef_1 = torch.exp(per_token_logps - old_per_token_logps)
    coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
    per_token_loss = -torch.min(coef_1 * advantages[:, None], coef_2 * advantages[:, None])
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    return per_token_loss + beta * per_token_kl


def _trl_dr_grpo(per_token_loss, mask, max_completion_length):
    return (per_token_loss * mask).sum() / (per_token_loss.size(0) * max_completion_length)


# %% [markdown]
# ## 1. Sequence log-prob of the response
# $\log\pi(y\mid x) = \sum_{t\in\text{resp}} \log p(\text{ids}_t \mid \text{ids}_{<t})$. The logit row that scores
# token $t$ is row $t-1$, so use `logits[:, :-1]` against `input_ids[:, 1:]` and **shift the mask the same way**
# (`response_mask[:, 1:]`). The last prompt token's logits predict the first response token; the logits at the last
# response token predict something we never score.

# %%
def token_logprobs(logits, input_ids):
    """(B, T, V), (B, T) -> (B, T-1): log p(input_ids[:, t+1] | prefix up to t)."""
    ### BEGIN SOLUTION
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    return logp.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    ### END SOLUTION


def sequence_logprob(logits, input_ids, response_mask, average=False):
    """Returns (B,): sum (or mean, if average) of response-token log-probs. Padding and prompt excluded."""
    ### BEGIN SOLUTION
    lp = token_logprobs(logits, input_ids)
    m = response_mask[:, 1:]
    tot = (lp * m).sum(-1)
    return tot / m.sum(-1).clamp(min=1) if average else tot
    ### END SOLUTION


ce = F.cross_entropy(logits[:, :-1].transpose(1, 2), input_ids[:, 1:], reduction="none")  # (B, T-1)
check_close("token logprobs vs -cross_entropy", token_logprobs(logits, input_ids), -ce)
check_close("sequence logprob vs -CE on response tokens", sequence_logprob(logits, input_ids, response_mask),
            -(ce * response_mask[:, 1:]).sum(-1))
check_close("sequence logprob vs rlhf-book compute_logprobs", sequence_logprob(logits, input_ids, response_mask),
            _rlhf_compute_logprobs(logits, input_ids, response_mask))
check_close("average variant", sequence_logprob(logits, input_ids, response_mask, average=True),
            _rlhf_compute_logprobs(logits, input_ids, response_mask, average_log_prob=True))
# padding must not matter: recompute each row on its unpadded slice, with garbage logits after the end
for b in range(B):
    n = prompt_len[b] + resp_len[b]
    garbage = logits.clone(); garbage[b, n:] = 1e3
    got = sequence_logprob(garbage[b:b + 1], input_ids[b:b + 1], response_mask[b:b + 1])
    lp_b = F.log_softmax(logits[b, :n - 1], -1).gather(-1, input_ids[b, 1:n, None]).squeeze(-1)
    check_close(f"row {b}: only response tokens, padding ignored", got, lp_b[prompt_len[b] - 1:].sum()[None])

# %% [markdown]
# ## 2. Bradley–Terry reward-model loss
# $P(y_c \succ y_r) = \sigma(r_c - r_r)$, so the NLL is $-\log\sigma(r_c - r_r)$ averaged over pairs (InstructGPT eq. 1).
# Use `F.logsigmoid`: `torch.log(torch.sigmoid(x))` underflows to `-inf` for $x \ll 0$.

# %%
def bt_loss(r_chosen, r_rejected):
    """(P,), (P,) scalar rewards -> scalar mean loss."""
    ### BEGIN SOLUTION
    return -F.logsigmoid(r_chosen - r_rejected).mean()
    ### END SOLUTION


rc, rr = torch.randn(16, generator=g) * 3, torch.randn(16, generator=g) * 3
check_close("BT vs BCE-with-logits(diff, 1)", bt_loss(rc, rr), F.binary_cross_entropy_with_logits(rc - rr, torch.ones(16)))
check_close("BT equal rewards = ln 2", bt_loss(torch.zeros(4), torch.zeros(4)), torch.tensor(math.log(2)))
check_close("BT stable for a badly wrong pair", bt_loss(torch.tensor([-500.0]), torch.tensor([500.0])), torch.tensor(1000.0))
check_close("BT only depends on the difference", bt_loss(rc + 7.0, rr + 7.0), bt_loss(rc, rr))

# %% [markdown]
# ## 3. DPO
# With $h_\theta = \log\frac{\pi_\theta(y_c|x)}{\pi_{ref}(y_c|x)} - \log\frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)}$ (sequence
# log-probs, **summed** over response tokens), $\mathcal L_{DPO} = -\mathbb E\,\log\sigma(\beta h_\theta)$ (eq. 7).
# Implicit rewards: $\hat r = \beta(\log\pi_\theta - \log\pi_{ref})$ per response (detached, for logging). Report
# **reward accuracy** = fraction of pairs with $\hat r_c > \hat r_r$ and **margin** = mean of $\hat r_c - \hat r_r$.

# %%
def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    """All (P,) sequence log-probs. Returns (loss, chosen_rewards (P,), rejected_rewards (P,))."""
    ### BEGIN SOLUTION
    lr_c, lr_r = pi_c - ref_c, pi_r - ref_r
    loss = -F.logsigmoid(beta * (lr_c - lr_r)).mean()
    return loss, beta * lr_c.detach(), beta * lr_r.detach()
    ### END SOLUTION


def dpo_metrics(chosen_rewards, rejected_rewards):
    """Returns (reward_accuracy, reward_margin), both scalars."""
    ### BEGIN SOLUTION
    return (chosen_rewards > rejected_rewards).double().mean(), (chosen_rewards - rejected_rewards).mean()
    ### END SOLUTION


P = 32
ref_c, ref_r = -torch.rand(P, generator=g) * 40, -torch.rand(P, generator=g) * 40
pi_c = ref_c + torch.randn(P, generator=g)
pi_r = ref_r + torch.randn(P, generator=g)
loss0, _, _ = dpo_loss(ref_c, ref_r, ref_c, ref_r, beta=0.1)
check_close("DPO = ln 2 when policy == reference", loss0, torch.tensor(math.log(2)))
for beta in (0.1, 0.5):
    loss, cr, rj = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta)
    l_ref, cr_ref, rj_ref = _rlhf_dpo(pi_c, pi_r, ref_c, ref_r, beta)
    check_close(f"DPO loss vs rlhf-book (beta={beta})", loss, l_ref)
    check_close(f"implicit rewards vs rlhf-book (beta={beta})", torch.stack([cr, rj]), torch.stack([cr_ref, rj_ref]))
_, cr, rj = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.5)
acc, margin = dpo_metrics(cr, rj)
check_close("reward accuracy", acc, ((pi_c - ref_c) > (pi_r - ref_r)).double().mean())
check_close("reward margin (beta=0.5)", margin, 0.5 * ((pi_c - ref_c) - (pi_r - ref_r)).mean())
# gradient: dL/d pi_c = -beta * sigmoid(-beta h) / P  (DPO §4, "What does the DPO update do?")
x = pi_c.clone().requires_grad_()
dpo_loss(x, pi_r, ref_c, ref_r, 0.5)[0].backward()
h = (pi_c - ref_c) - (pi_r - ref_r)
check_close("DPO gradient w.r.t. chosen logp", x.grad, -0.5 * torch.sigmoid(-0.5 * h) / P)

# %% [markdown]
# ## 4. GRPO group-relative advantages
# For one prompt with rewards $r_1..r_G$: $\hat A_i = (r_i - \mathrm{mean}(r)) / (\mathrm{std}(r) + \varepsilon)$, the same value
# for every token of response $i$ (DeepSeekMath §4.1.2). TRL, rlhf-book and reasoning-from-scratch all use the
# **unbiased** (Bessel-corrected) std; TRL and reasoning-from-scratch use ε = 1e-4. `scale=False` gives Dr. GRPO's
# advantages $r_i - \mathrm{mean}(r)$.

# %%
def grpo_advantages(rewards, scale=True, eps=1e-4):
    """rewards: (num_prompts, G). Returns (num_prompts, G) advantages, normalized within each row (group)."""
    ### BEGIN SOLUTION
    adv = rewards - rewards.mean(dim=1, keepdim=True)
    return adv / (rewards.std(dim=1, keepdim=True) + eps) if scale else adv
    ### END SOLUTION


Pq, G = 5, 4
rew = torch.randint(0, 2, (Pq, G), generator=g).double() + 0.1 * torch.randn(Pq, G, generator=g)
rew[0] = 1.0  # a group where every response is equally good
A = grpo_advantages(rew)
check_close("advantages vs TRL (group scaling)", A.flatten(), _trl_advantages(rew.flatten(), G))
check_close("Dr. GRPO advantages vs TRL scale_rewards='none'", grpo_advantages(rew, scale=False).flatten(),
            _trl_advantages(rew.flatten(), G, scale_rewards=False))
check_close("advantages sum to zero per group", A.sum(1), torch.zeros(Pq), atol=1e-10)
check_close("all-equal group -> zero advantage (no gradient)", A[0], torch.zeros(G))
check_close("invariant to per-group shift", grpo_advantages(rew + torch.arange(Pq)[:, None] * 10.0), A, atol=1e-8)
check_close("~invariant to positive scaling (up to eps)", grpo_advantages(rew[1:] * 3.0, eps=1e-12),
            grpo_advantages(rew[1:], eps=1e-12))
check_close("binary rewards, 1 of 4 correct", grpo_advantages(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), eps=0.0),
            torch.tensor([[1.5, -0.5, -0.5, -0.5]]))

# %% [markdown]
# ## 5. Per-token $k_3$ KL to the reference and the clipped GRPO token loss
# Samples come from the policy, so with $\rho = \pi_{ref}/\pi_\theta$ per token, $k_3 = \rho - \log\rho - 1 \ge 0$ estimates
# $\mathrm{KL}[\pi_\theta\,\|\,\pi_{ref}]$ (DeepSeekMath eq. 4). Unlike PPO-RLHF, GRPO puts the KL **in the loss**, not in
# the reward. Per-token loss with $r_t = \exp(\log\pi_\theta - \log\pi_{old})$ and decoupled clip bounds
# (DAPO's clip-higher):
# $\ell_{i,t} = -\min(r_t\hat A_i,\ \mathrm{clip}(r_t, 1-\epsilon_{low}, 1+\epsilon_{high})\hat A_i) + \beta\,k_{3,t}$.

# %%
def kl_k3(logp, ref_logp):
    """Per-token k3 estimate of KL[pi || pi_ref] for tokens sampled from pi. Same shape as inputs."""
    ### BEGIN SOLUTION
    d = ref_logp - logp
    return d.exp() - d - 1
    ### END SOLUTION


def grpo_token_loss(logp, old_logp, ref_logp, adv, eps_low=0.2, eps_high=0.2, beta=0.04):
    """logp, old_logp, ref_logp: (B, L); adv: (B,). Returns the (B, L) per-token loss (unmasked)."""
    ### BEGIN SOLUTION
    ratio = (logp - old_logp).exp()
    a = adv[:, None]
    pg = -torch.min(ratio * a, ratio.clamp(1 - eps_low, 1 + eps_high) * a)
    return pg + beta * kl_k3(logp, ref_logp)
    ### END SOLUTION


# k3 is unbiased for KL[pi || ref] when tokens are sampled from pi
pi = torch.tensor([0.5, 0.25, 0.15, 0.1])
ref = torch.tensor([0.1, 0.2, 0.3, 0.4])
xs = torch.multinomial(pi, 400_000, replacement=True, generator=g)
kl_exact = (pi * (pi / ref).log()).sum()
print(f"KL[pi||ref] = {kl_exact:.3f}, KL[ref||pi] = {(ref * (ref / pi).log()).sum():.3f}")
check_close("k3 mean ~ KL[pi||ref]", kl_k3(pi.log()[xs], ref.log()[xs]).mean(), kl_exact, atol=0.01)
d_ = torch.linspace(-4, 4, 17)
check("k3 >= 0 and = 0 iff equal", bool((kl_k3(torch.zeros(17), d_) >= 0).all()) and kl_k3(torch.zeros(1), torch.zeros(1)).item() == 0)

Bg, L = 2 * G, 6  # two prompts x G responses
lens = torch.tensor([6, 2, 4, 1, 3, 5, 6, 2])
mask = (torch.arange(L)[None] < lens[:, None]).double()
old_lp = -torch.rand(Bg, L, generator=g) * 3
lp = old_lp + 0.3 * torch.randn(Bg, L, generator=g)
ref_lp = old_lp + 0.2 * torch.randn(Bg, L, generator=g)
adv = grpo_advantages(torch.randint(0, 2, (2, G), generator=g).double() + 0.3 * torch.randn(2, G, generator=g)).flatten()
for el, eh, bt in ((0.2, 0.2, 0.04), (0.2, 0.28, 0.0), (0.1, 0.3, 0.1)):
    check_close(f"token loss vs TRL (eps_low={el}, eps_high={eh}, beta={bt})",
                grpo_token_loss(lp, old_lp, ref_lp, adv, el, eh, bt), _trl_per_token_loss(lp, old_lp, ref_lp, adv, el, eh, bt))
# clip-higher: with A > 0, ratio 1.25 is clipped under eps_high=0.2 but not under 0.28
x = torch.log(torch.tensor([[1.25]])).requires_grad_()
grpo_token_loss(x, torch.zeros(1, 1), x.detach(), torch.tensor([1.0]), 0.2, 0.2, 0.0).sum().backward()
g_sym = x.grad.clone(); x.grad = None
grpo_token_loss(x, torch.zeros(1, 1), x.detach(), torch.tensor([1.0]), 0.2, 0.28, 0.0).sum().backward()
check("clip-higher: A>0, r=1.25 -> zero grad at eps_high=0.2, gradient at 0.28", g_sym.item() == 0 and x.grad.item() < 0)

# %% [markdown]
# ## 6. Loss aggregation: GRPO vs DAPO vs Dr. GRPO
# Given the masked `(B, L)` per-token loss:
# - `"grpo"` (DeepSeekMath eq. 3): mean over each response's tokens, then mean over responses. Token weight $\frac{1}{B\,|o_i|}$.
# - `"dapo"` (DAPO eq. 12): sum over all tokens / total number of response tokens. Token weight $\frac{1}{\sum_i |o_i|}$.
# - `"dr_grpo"` (Dr. GRPO §3.2, TRL `loss_type="dr_grpo"`): sum over all tokens / (B · `max_len`), a constant
#   independent of the lengths. Token weight $\frac{1}{B\,L_{max}}$. (Dr. GRPO also drops the std: `scale=False` above.)

# %%
def aggregate_loss(per_token_loss, mask, mode="grpo", max_len=None):
    """per_token_loss, mask: (B, L). mode in {"grpo", "dapo", "dr_grpo"}; max_len required for "dr_grpo"."""
    ### BEGIN SOLUTION
    masked = per_token_loss * mask
    if mode == "grpo":
        return (masked.sum(-1) / mask.sum(-1).clamp(min=1)).mean()
    if mode == "dapo":
        return masked.sum() / mask.sum().clamp(min=1)
    if mode == "dr_grpo":
        return masked.sum() / (per_token_loss.shape[0] * max_len)
    raise ValueError(mode)
    ### END SOLUTION


ptl = grpo_token_loss(lp, old_lp, ref_lp, adv, 0.2, 0.2, 0.04)
ptl = ptl.masked_fill(mask == 0, 123.0)  # garbage at padded positions must not leak
check_close("grpo aggregation vs rlhf-book GRPOLoss", aggregate_loss(ptl, mask, "grpo"),
            _rlhf_grpo_loss(lp, old_lp, ref_lp, adv[:, None], mask, 0.2, 0.2, 0.04))
check_close("dapo aggregation vs rlhf-book DAPOLoss", aggregate_loss(grpo_token_loss(lp, old_lp, ref_lp, adv, 0.2, 0.28, 0.0)
            .masked_fill(mask == 0, 123.0), mask, "dapo"), _rlhf_dapo_loss(lp, old_lp, adv[:, None], mask, 0.2, 0.28))
check_close("dr_grpo aggregation vs TRL", aggregate_loss(ptl, mask, "dr_grpo", max_len=L), _trl_dr_grpo(ptl, mask, L))

# What each variant does to the gradient: at logp = old_logp = ref_logp the ratio is 1 and the k3 gradient is 0,
# so d loss / d logp[i, t] = -A_i * (token weight).
at = old_lp.clone()
weights = {"grpo": 1 / (Bg * lens.double()), "dapo": torch.full((Bg,), 1 / lens.sum().double()),
           "dr_grpo": torch.full((Bg,), 1 / (Bg * L))}
for mode, w in weights.items():
    x = at.clone().requires_grad_()
    aggregate_loss(grpo_token_loss(x, at, at, adv), mask, mode, max_len=L).backward()
    check_close(f"{mode}: per-token gradient = -A_i * weight, 0 on padding", x.grad, -(adv * w)[:, None] * mask)
print("per-token weight by response length (grpo):", dict(zip(lens.tolist(), (1 / (Bg * lens.double())).tolist())))

# %% [markdown] tags=["solution_only"]
# ### Reference check for answer 8: what gradient does the $k_3$ loss give?
# Exact expectation over a 6-token vocabulary (no sampling noise): $\mathbb E_{x\sim\pi_\theta}[\nabla_\theta k_3(x)]$ with $x$ held fixed,
# compared with the gradients of the reverse and the forward KL.

# %% tags=["solution_only"]
theta = torch.randn(6, generator=g, requires_grad=True)
ref6 = torch.softmax(torch.randn(6, generator=g), 0)
logp6 = theta.log_softmax(0)
pi6 = logp6.exp().detach()
grad_k3 = torch.autograd.grad((pi6 * kl_k3(logp6, ref6.log())).sum(), theta, retain_graph=True)[0]  # weights pi6 fixed
grad_rev = torch.autograd.grad((logp6.exp() * (logp6 - ref6.log())).sum(), theta, retain_graph=True)[0]
grad_fwd = torch.autograd.grad((ref6 * (ref6.log() - logp6)).sum(), theta)[0]
check_close("E[grad k3] = grad KL[ref || pi] (forward)", grad_k3, grad_fwd)
check("... and differs from grad KL[pi || ref] (reverse)", not torch.allclose(grad_k3, grad_rev, atol=1e-3))

# %% [markdown]
# ## Say it out loud
# 1. Why `logits[:, :-1]` vs `input_ids[:, 1:]`, and why must the mask be shifted too? Sum or mean over tokens for DPO?
# 2. Derive the DPO loss from $\max_\pi \mathbb E[r] - \beta\,\mathrm{KL}[\pi\|\pi_{ref}]$ in five lines.
# 3. The DPO loss starts at exactly ln 2. Why? What do reward accuracy and margin tell you, and what does it mean when
#    *both* chosen and rejected implicit rewards go down?
# 4. What does β control in DPO, and what happens as β → 0 or large?
# 5. Bradley–Terry: what is invariant about the learned reward, and what does that imply for using it in RL?
# 6. GRPO has no critic. What plays the baseline's role? What happens in a group where every answer is right?
# 7. Why does the std normalization bias training (Dr. GRPO), and why is `std` unbiased vs biased a detail worth knowing?
# 8. Which KL direction does GRPO's $k_3$ estimate, why is it in the loss instead of the reward, and why do DAPO and many
#    RLVR recipes drop it?
# 9. Explain the length bias of per-sequence averaging and what DAPO's and Dr. GRPO's normalizers change.
# 10. reasoning-from-scratch ch07 uses a *sequence-level* ratio $\exp(\sum_t \log\pi_\theta - \sum_t\log\pi_{old})$ and a
#     $k_1$ KL on sequence log-probs. How does that differ from the token-level version here?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. A causal LM's output at position $t$ is the distribution of token $t+1$, so token $t$ is scored by row $t-1$. The
#    mask must select the *targets*, so it is `response_mask[:, 1:]`. With `response_mask[:, :-1]` you are off by one:
#    you score the last prompt token and drop the last response token (often EOS, which carries the "stop" decision).
#    The DPO paper uses $\log\pi(y|x)$, the **sum** over response tokens. Raschka's DPO notebook averages over tokens,
#    a length-normalized variant (rlhf-book's `average_log_prob=True`; SimPO uses it by design). The sum matches the
#    derivation, but it lets longer responses carry larger log-ratios.
# 2. The KL-regularized optimum is $\pi^*(y|x) = \pi_{ref}(y|x)\exp(r(x,y)/\beta)/Z(x)$. Solve for the reward:
#    $r = \beta\log\frac{\pi^*}{\pi_{ref}} + \beta\log Z(x)$. Substitute into Bradley–Terry $\sigma(r_c - r_r)$, where
#    $\beta\log Z(x)$ cancels because both responses share $x$. Maximize the likelihood over a parametric $\pi_\theta$
#    to get $-\log\sigma(\beta h_\theta)$ (DPO §4, eq. 7).
# 3. At init $\pi_\theta = \pi_{ref}$, so $h=0$ and $-\log\sigma(0)=\ln 2$; any other value at step 0 means a bug
#    (wrong reference, dropout on, mismatched masks). Accuracy is how often the implicit reward ranks the pair right,
#    and the margin is how confidently. Both log-ratios falling while the margin grows means the policy lowers the
#    likelihood of the chosen response too, just less. This is a known DPO failure mode that Smaug/DPO-Positive targets
#    (arXiv:2402.13228).
# 4. β is the KL-regularization strength (the inverse temperature of the implicit reward). Small β lets the policy move
#    far from the reference and saturates the loss quickly. Large β keeps it close (the gradient weight
#    $\beta\sigma(-\beta h)$ also changes scale). Typical values are 0.1-0.5.
# 5. Only differences matter, so $r(x,y) + c(x)$ gives the same likelihood: the reward is identified only up to a
#    per-prompt offset (and its scale is set by the data). RL therefore needs a baseline or normalization per prompt,
#    which is exactly what per-group centering in GRPO provides. Reward-model scores are also often whitened.
# 6. The group mean (empirical value of the prompt under the current policy) is the baseline, and the std rescales it.
#    If all G answers get the same reward, every advantage is 0, so that prompt contributes nothing but compute (and
#    KL). DAPO's dynamic sampling (§3.2) over-samples and filters such groups out.
# 7. Dividing by the group std up-weights prompts whose rewards barely vary (very easy or very hard ones), a
#    question-level difficulty bias (Dr. GRPO §3.1). With binary rewards and small G, biased vs unbiased std differs
#    by $\sqrt{G/(G-1)}$ (15% at G=4). That is not catastrophic, but it is enough to fail a reference check, and the
#    common implementations (TRL, rlhf-book, reasoning-from-scratch) use the unbiased one.
# 8. $\mathrm{KL}[\pi_\theta\|\pi_{ref}]$ (reverse KL, tokens sampled from the policy). DeepSeekMath puts it in the
#    loss "instead of adding KL penalty in the reward ... avoiding complicating the calculation of $\hat A_{i,t}$"
#    (§4.1.1): a per-token reward penalty would mix into the group normalization. DAPO drops it (§2.3): a long-CoT
#    policy is expected to move far from the initial model, and the KL mostly slows that down. A subtlety (derived and
#    checked numerically in the cell below): differentiating the per-sample $k_3$ with the samples held fixed gives
#    $(1-\rho)\nabla\log\pi_\theta$, whose expectation under $\pi_\theta$ is $-\sum_x\pi_{ref}\nabla\log\pi_\theta =
#    \nabla\mathrm{KL}[\pi_{ref}\|\pi_\theta]$. The $k_3$ *loss* therefore follows the forward-KL gradient, even though its
#    *value* estimates the reverse KL.
# 9. Per-sequence averaging gives each token weight $1/(G|o_i|)$. A short correct answer gets a big per-token push, and
#    a long wrong answer gets a small per-token penalty, so wrong answers drift longer (Dr. GRPO §3.1). DAPO weights
#    every token equally ($1/\sum|o_i|$, §3.3), so long responses count in proportion to their length, which also
#    penalizes long low-quality outputs. Dr. GRPO uses a constant ($B\cdot L_{max}$), so the per-token weight does not
#    depend on the lengths or on batch composition, like plain REINFORCE with a fixed scale.
# 10. A sequence-level ratio is a product of per-token ratios, so it explodes or vanishes for long responses, and one
#     clip decision covers the whole response. The reasoning-from-scratch scripts default to `clip_eps=10` (argparse
#     help: "DeepSeek uses 10"), which makes the clip almost inactive. GSPO (arXiv:2507.18071) uses a length-normalized
#     sequence ratio instead. A $k_1$ term $\log\pi_\theta - \log\pi_{ref}$ on sampled sequences estimates the KL value
#     without bias, but as a loss its gradient is $\nabla\log\pi_\theta(y)$, and $\mathbb E_{y\sim\pi_\theta}[\nabla\log\pi_\theta(y)] = 0$.
#     In expectation it regularizes nothing; it only adds noise. Token-level ratios with $k_3$, as here, are what
#     DeepSeekMath, TRL and rlhf-book implement.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
