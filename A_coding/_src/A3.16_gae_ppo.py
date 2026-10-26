# %% [markdown]
# # A3.16 · GAE and the PPO loss
#
# **ID:** A3.16 · **Target:** 20 min · **Plan week:** 7-8 · **Part:** A
#
# **The drill.** Write the two pieces of PPO that interviewers ask for most: generalized advantage
# estimation over a rollout with episode boundaries, and the full PPO minibatch loss (clipped surrogate,
# value loss with optional clipping, entropy bonus) plus the two diagnostics you always log (approx-KL and
# clip fraction). The mock-interview checklist asks for **"PPO loss + GAE in 20 min"** from a blank file,
# so after the first pass, redo it in an empty notebook against these checks.
#
# **Conventions (state them before you code; interviewers check them).**
# - Rollout tensors are time-major, shape `(T, N)` for `T` steps of `N` parallel envs.
# - `dones[t] = 1` means the transition at step `t` **ended** the episode (Gymnasium `terminated or
#   truncated` returned by `env.step(a_t)`), so there is no bootstrap from $V(s_{t+1})$ and no advantage
#   flows back across it. `last_value` is $V(s_T)$ for the observation after the last step.
# - Losses are **minimized**: the policy loss is the negative of the PPO objective $L^{CLIP}$.
# - CleanRL stores the flag one step later (`dones[t]` = "obs t is the first obs after a reset") and reads
#   `dones[t+1]`. The reference cell below shows the conversion. Both are correct; mixing them is the classic
#   off-by-one bug.
#
# **Rules.** No `torch.distributions` inside your functions. Loop over time in GAE (that is what every
# real implementation does); vectorize over envs. Fill in every `raise NotImplementedError`; the cell under
# each function prints `ok` or tells you what is off.
#
# **Deeper study:** rlhf-book `../../rlhf-book/code/policy_gradients/loss.py` (`PPOLoss`: the same
# clipped surrogate and clipped value loss applied per token for an LLM) and `../../rlhf-book/code/policy_gradients/utils.py`
# (`compute_gae` for a sparse end-of-sequence reward). CleanRL's full single-file PPO is the best end-to-end read;
# the 37-details post explains every line of it. For the LLM version of these advantages see A3.17 (GRPO) and
# reasoning-from-scratch ch06-07 (link in A3.17).
#
# ## Sources
# - J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, *High-Dimensional Continuous Control Using Generalized
#   Advantage Estimation*, ICLR 2016, arXiv:1506.02438 — https://arxiv.org/abs/1506.02438 (GAE: eq. 16; λ=0 and λ=1
#   limits: eqs. 17-18).
# - J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, *Proximal Policy Optimization Algorithms*, 2017,
#   arXiv:1707.06347 — https://arxiv.org/abs/1707.06347 ($L^{CLIP}$: eq. 7; combined objective with value loss and
#   entropy: eq. 9; truncated GAE: eqs. 11-12).
# - CleanRL `cleanrl/ppo.py` (MIT), vwxyzjn/cleanrl@fe8d8a03c41a7ef5b523e2e354bd01c363e786bb —
#   https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo.py :
#   GAE block lines 217-231, loss block lines 251-285. Transcribed (not executed from the repo) into the hidden
#   reference cell; that transcription is the ground truth for the checks.
# - S. Huang, R. F. J. Dossa, A. Raffin, A. Kanervisto, W. Wang, *The 37 Implementation Details of Proximal Policy
#   Optimization*, ICLR Blog Track 2022 — https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
#   (per-minibatch advantage normalization, value clipping, approx-KL and clipfrac as diagnostics, truncation handling).
# - J. Schulman, *Approximating KL Divergence* (blog, 2020) — http://joschu.net/blog/kl-approx.html (estimators
#   $k_1 = -\log r$, $k_3 = (r-1) - \log r$ with $r = p/q$, $x\sim q$).
# - L. Engstrom et al., *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO*, ICLR 2020,
#   arXiv:2005.12729 — https://arxiv.org/abs/2005.12729 ; M. Andrychowicz et al., *What Matters In On-Policy
#   Reinforcement Learning? A Large-Scale Empirical Study*, ICLR 2021, arXiv:2006.05990 — https://arxiv.org/abs/2006.05990
#   (evidence on value clipping and advantage normalization, as summarized in the 37-details post).
# - N. Lambert, *RLHF Book* code, natolambert/rlhf-book@eecc49e (code under MIT), `code/policy_gradients/loss.py` —
#   https://github.com/natolambert/rlhf-book/blob/eecc49e1e1daf1be670d7242eb090240b5bb0f04/code/policy_gradients/loss.py

# %%
import math

import torch

from prep import Session, check, check_close, seed_everything

s = Session("A3.16", target_min=20)
seed_everything(0)
torch.set_default_dtype(torch.float64)  # exact comparisons against the reference loops

T, N = 32, 3
g = torch.Generator().manual_seed(0)
rewards = torch.randn(T, N, generator=g)
values = torch.randn(T, N, generator=g)
dones = (torch.rand(T, N, generator=g) < 0.15).double()
dones[-1, 0] = 1.0  # env 0 ends exactly at the rollout boundary
last_value = torch.randn(N, generator=g)

# %% [markdown]
# ### Reference implementations (ground truth for the checks; do not read before you finish)
# The cell below is collapsed in Jupyter. It is a line-by-line transcription of CleanRL `ppo.py` (MIT, @fe8d8a0,
# lines 217-231 and 251-285), with `args.*` turned into function arguments.

# %% jupyter={"source_hidden": true}
def _cleanrl_gae(rewards, values, cl_dones, next_value, next_done, gamma, gae_lambda):
    # CleanRL convention: cl_dones[t] = 1 if obs t is the first obs of a new episode; next_done for obs T.
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - cl_dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns


def _to_cleanrl_dones(dones):
    """Our dones[t] ('step t ended an episode') -> CleanRL (dones[t] 'obs t starts one', next_done)."""
    cl = torch.zeros_like(dones)
    cl[1:] = dones[:-1]  # cl[0] belongs to the previous rollout; GAE never reads it
    return cl, dones[-1]


def _cleanrl_loss(newlogprob, b_logprobs, b_advantages, newvalue, b_values, b_returns, entropy,
                  clip_coef, norm_adv, clip_vloss, ent_coef, vf_coef):
    logratio = newlogprob - b_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
    mb_advantages = b_advantages
    if norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    newvalue = newvalue.view(-1)
    if clip_vloss:
        v_loss_unclipped = (newvalue - b_returns) ** 2
        v_clipped = b_values + torch.clamp(newvalue - b_values, -clip_coef, clip_coef)
        v_loss_clipped = (v_clipped - b_returns) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
    return dict(loss=loss, pg_loss=pg_loss, v_loss=v_loss, entropy=entropy_loss,
                old_approx_kl=old_approx_kl, approx_kl=approx_kl, clipfrac=clipfrac)


# %% [markdown]
# ## 1. GAE(γ, λ) with episode boundaries
# $\delta_t = r_t + \gamma(1-d_t)V(s_{t+1}) - V(s_t)$, and backwards in time
# $\hat A_t = \delta_t + \gamma\lambda(1-d_t)\hat A_{t+1}$, with $V(s_T)$ = `last_value`.
# The mask $(1-d_t)$ appears **twice**: no bootstrap across the boundary, and no advantage leaking back from the next
# episode. Value targets ("returns") are $\hat A_t + V(s_t)$, the TD(λ) return.

# %%
def gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """rewards, values, dones: (T, N); last_value: (N,). Returns (advantages, returns), both (T, N)."""
    ### BEGIN SOLUTION
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    next_v = last_value
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        running = delta + gamma * lam * nonterm * running
        adv[t] = running
        next_v = values[t]
    return adv, adv + values
    ### END SOLUTION


# hand-computed: T=3, one env, episode ends after step 1
a, ret = gae(torch.tensor([[1.0], [0.0], [2.0]]), torch.tensor([[0.5], [0.2], [0.1]]),
             torch.tensor([[0.0], [1.0], [0.0]]), torch.tensor([0.3]), gamma=0.9, lam=0.5)
check_close("GAE hand case", a.squeeze(1), torch.tensor([0.59, -0.2, 2.17]))
check_close("returns = adv + V", ret.squeeze(1), torch.tensor([1.09, 0.0, 2.27]))

cl_dones, next_done = _to_cleanrl_dones(dones)
for gm, lm in ((0.99, 0.95), (0.9, 0.5), (1.0, 1.0)):
    a, ret = gae(rewards, values, dones, last_value, gm, lm)
    a_ref, ret_ref = _cleanrl_gae(rewards, values, cl_dones, last_value, next_done, gm, lm)
    check_close(f"GAE vs CleanRL (gamma={gm}, lam={lm})", a, a_ref)
    check_close(f"returns vs CleanRL (gamma={gm}, lam={lm})", ret, ret_ref)

# λ = 0: one-step TD residual (GAE eq. 17)
v_next = torch.cat([values[1:], last_value[None]])
delta = rewards + 0.99 * (1 - dones) * v_next - values
check_close("lam=0 -> TD residual", gae(rewards, values, dones, last_value, 0.99, 0.0)[0], delta)

# λ = 1: discounted Monte-Carlo return (bootstrapped with V(s_T) if the last episode is unfinished) minus V (eq. 18)
G, mc = last_value.clone(), torch.zeros_like(rewards)
for t in reversed(range(T)):
    G = rewards[t] + 0.99 * (1 - dones[t]) * G
    mc[t] = G
check_close("lam=1 -> MC return - V", gae(rewards, values, dones, last_value, 0.99, 1.0)[0], mc - values)

# %% [markdown]
# ## 2. Advantage normalization (per minibatch)
# CleanRL (and SB3, baselines) standardize advantages **inside each minibatch**, with the unbiased std and `+1e-8`:
# $\tilde A = (A - \bar A)/(\mathrm{std}(A) + 10^{-8})$.

# %%
def normalize_adv(adv, eps=1e-8):
    ### BEGIN SOLUTION
    return (adv - adv.mean()) / (adv.std() + eps)
    ### END SOLUTION


x = torch.randn(64, generator=g) * 3 + 2
check_close("normalized mean 0", normalize_adv(x).mean(), torch.tensor(0.0), atol=1e-10)
check_close("normalized std 1 (unbiased)", normalize_adv(x).std(), torch.tensor(1.0), atol=1e-6)
check_close("normalize_adv vs (x-mean)/(std+eps)", normalize_adv(x), (x - x.mean()) / (x.std() + 1e-8))

# %% [markdown]
# ## 3. Clipped surrogate (policy loss)
# $r = \exp(\log\pi_\theta - \log\pi_{old})$. PPO **maximizes** $\mathbb E[\min(rA,\ \mathrm{clip}(r,1-\epsilon,1+\epsilon)A)]$
# (eq. 7), so the loss is $\mathbb E[\max(-rA,\ -\mathrm{clip}(r)A)]$. The min makes it a pessimistic bound: the clip only
# removes the incentive to move $r$ further in the direction that *helps* the objective.

# %%
def ppo_policy_loss(new_logp, old_logp, adv, clip_coef=0.2):
    """All (B,). adv already normalized if you want that. Returns scalar mean loss."""
    ### BEGIN SOLUTION
    ratio = (new_logp - old_logp).exp()
    return torch.max(-adv * ratio, -adv * ratio.clamp(1 - clip_coef, 1 + clip_coef)).mean()
    ### END SOLUTION


# four cases with eps = 0.2: (A>0, r=1.5) clipped; (A>0, r=0.5) not clipped; (A<0, r=0.5) clipped; (A<0, r=1.5) not clipped
r = torch.tensor([1.5, 0.5, 0.5, 1.5])
A = torch.tensor([1.0, 1.0, -1.0, -1.0])
old = torch.log(torch.tensor([0.2, 0.4, 0.3, 0.1]))
new = (old + r.log()).requires_grad_()
loss = ppo_policy_loss(new, old, A, 0.2)
check_close("policy loss hand case", loss, torch.tensor((-1.2 - 0.5 + 0.8 + 1.5) / 4))
loss.backward()
# d loss / d new_logp = -A * r / B where unclipped, 0 where clipped
check_close("policy grad: zero where clipped", new.grad, torch.tensor([0.0, -0.5, 0.0, 1.5]) / 4)

# %% [markdown]
# ## 4. Value loss (optional clipping) and the KL / clip-fraction diagnostics
# Unclipped: $\tfrac12\,\mathbb E[(V_\theta - R)^2]$. Clipped (CleanRL / original baselines):
# $V^{clip} = V_{old} + \mathrm{clip}(V_\theta - V_{old}, -\epsilon, \epsilon)$ and the loss is
# $\tfrac12\,\mathbb E[\max((V_\theta-R)^2, (V^{clip}-R)^2)]$ (again pessimistic).
#
# Approx-KL between the sampling policy $q=\pi_{old}$ and $p=\pi_\theta$ from samples $x\sim q$, with
# $\log r = \log\pi_\theta - \log\pi_{old}$: $k_1 = -\log r$ (unbiased, high variance, can be negative) and
# $k_3 = (r-1) - \log r$ (unbiased, always $\ge 0$, low variance). Both estimate $\mathrm{KL}[\pi_{old}\,\|\,\pi_\theta]$.
# Clip fraction = fraction of samples with $|r-1| > \epsilon$.

# %%
def ppo_value_loss(new_v, old_v, returns, clip_coef=None):
    """All (B,). clip_coef=None: plain 0.5 * MSE."""
    ### BEGIN SOLUTION
    unclipped = (new_v - returns) ** 2
    if clip_coef is None:
        return 0.5 * unclipped.mean()
    v_clip = old_v + (new_v - old_v).clamp(-clip_coef, clip_coef)
    return 0.5 * torch.max(unclipped, (v_clip - returns) ** 2).mean()
    ### END SOLUTION


def approx_kl(new_logp, old_logp):
    """Returns (k1, k3), each a scalar mean over samples drawn from the OLD policy."""
    ### BEGIN SOLUTION
    logr = new_logp - old_logp
    return (-logr).mean(), ((logr.exp() - 1) - logr).mean()
    ### END SOLUTION


def clip_fraction(new_logp, old_logp, clip_coef=0.2):
    ### BEGIN SOLUTION
    return ((new_logp - old_logp).exp().sub(1).abs() > clip_coef).double().mean()
    ### END SOLUTION


# value loss: returns R=1, V_old=0, eps=0.2
nv = torch.tensor([0.5, -0.1, 1.5])
check_close("value loss unclipped", ppo_value_loss(nv, torch.zeros(3), torch.ones(3)),
            0.5 * torch.tensor([0.25, 1.21, 0.25]).mean())
# clipped values: 0.2, -0.1, 0.2 -> squared errors 0.64, 1.21, 0.64; max with unclipped -> 0.64, 1.21, 0.64
check_close("value loss clipped", ppo_value_loss(nv, torch.zeros(3), torch.ones(3), 0.2),
            0.5 * torch.tensor([0.64, 1.21, 0.64]).mean())

# KL estimators: exact KL[q||p] for categoricals vs Monte-Carlo with x ~ q (q = old policy)
q = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05])
p = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])
kl_qp = (q * (q / p).log()).sum()
kl_pq = (p * (p / q).log()).sum()
print(f"KL[q||p] = {kl_qp:.3f}, KL[p||q] = {kl_pq:.3f}  (the direction matters)")
xs = torch.multinomial(q, 400_000, replacement=True, generator=g)
k1, k3 = approx_kl(p.log()[xs], q.log()[xs])
check_close("k1 unbiased for KL[old||new]", k1, kl_qp, atol=0.01)
check_close("k3 unbiased for KL[old||new]", k3, kl_qp, atol=0.01)
lr_ = torch.linspace(-3, 3, 13)
check("k3 >= 0 per sample", bool((((lr_.exp() - 1) - lr_) >= 0).all()) and approx_kl(lr_, torch.zeros(13))[1] > 0)
check_close("clip fraction", clip_fraction(r.log(), torch.zeros(4), 0.2), torch.tensor(1.0))
check_close("clip fraction (mixed)",
            clip_fraction(torch.tensor([1.1, 1.2, 0.95, 1.0]).log(), torch.zeros(4), 0.15), torch.tensor(0.25))

# %% [markdown]
# ## 5. The full minibatch loss
# $L = L_{pg} - c_{ent}\,\mathbb E[H] + c_{vf}\,L_V$ (eq. 9 with the signs flipped for minimization). Defaults are
# CleanRL's: `clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, norm_adv=True, clip_vloss=True`. The value-loss clip uses
# the **same** `clip_coef` (CleanRL), even though values and ratios live on different scales.

# %%
def ppo_loss(new_logp, old_logp, adv, new_v, old_v, returns, entropy,
             clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, norm_adv=True, clip_vloss=True):
    """All inputs (B,). Returns (loss, stats) with stats keys: pg_loss, v_loss, entropy, k1, k3, clipfrac."""
    ### BEGIN SOLUTION
    if norm_adv:
        adv = normalize_adv(adv)
    pg = ppo_policy_loss(new_logp, old_logp, adv, clip_coef)
    vl = ppo_value_loss(new_v, old_v, returns, clip_coef if clip_vloss else None)
    ent = entropy.mean()
    with torch.no_grad():
        k1, k3 = approx_kl(new_logp, old_logp)
        cf = clip_fraction(new_logp, old_logp, clip_coef)
    loss = pg - ent_coef * ent + vf_coef * vl
    return loss, dict(pg_loss=pg, v_loss=vl, entropy=ent, k1=k1, k3=k3, clipfrac=cf)
    ### END SOLUTION


B = 256
mb = dict(new_logp=-torch.rand(B, generator=g) * 2, old_logp=-torch.rand(B, generator=g) * 2,
          adv=torch.randn(B, generator=g) * 2 + 0.5, new_v=torch.randn(B, generator=g),
          old_v=torch.randn(B, generator=g), returns=torch.randn(B, generator=g), entropy=torch.rand(B, generator=g))
for nadv, cv in ((True, True), (False, False), (True, False)):
    loss, st = ppo_loss(**mb, norm_adv=nadv, clip_vloss=cv)
    ref = _cleanrl_loss(mb["new_logp"], mb["old_logp"], mb["adv"], mb["new_v"], mb["old_v"], mb["returns"],
                        mb["entropy"], clip_coef=0.2, norm_adv=nadv, clip_vloss=cv, ent_coef=0.01, vf_coef=0.5)
    tag = f"(norm_adv={nadv}, clip_vloss={cv})"
    check_close(f"total loss vs CleanRL {tag}", loss, ref["loss"])
    check_close(f"pg_loss vs CleanRL {tag}", st["pg_loss"], ref["pg_loss"])
    check_close(f"v_loss vs CleanRL {tag}", st["v_loss"], ref["v_loss"])
check_close("k1 vs CleanRL old_approx_kl", st["k1"], ref["old_approx_kl"])
check_close("k3 vs CleanRL approx_kl", st["k3"], ref["approx_kl"])
check_close("clipfrac vs CleanRL", st["clipfrac"], ref["clipfrac"])

# %% [markdown] tags=["solution_only"]
# ### Reference run: one PPO epoch on a 3-armed bandit
# Sanity check that the pieces work together: sample from the old policy, compute (trivial, one-step) advantages,
# take a few gradient steps on the PPO loss. The best arm's probability must go up and the ratio stays near the
# clip range, which is what `clipfrac` and `k3` report.

# %% tags=["solution_only"]
torch.manual_seed(0)
arm_reward = torch.tensor([0.0, 1.0, 0.2])
logits = torch.zeros(3, requires_grad=True)
opt = torch.optim.Adam([logits], lr=0.06)
for it in range(6):
    with torch.no_grad():
        old_lp_all = logits.log_softmax(-1)
        acts = torch.multinomial(old_lp_all.exp(), 512, replacement=True)
        rew = arm_reward[acts] + 0.1 * torch.randn(512)
        old_lp = old_lp_all[acts]
        adv_b = rew - (old_lp_all.exp() * arm_reward).sum()  # exact baseline V = E_pi[r]
    for _ in range(4):
        lp_all = logits.log_softmax(-1)
        ent = -(lp_all.exp() * lp_all).sum().expand(512)
        loss, st = ppo_loss(lp_all[acts], old_lp, adv_b, torch.zeros(512), torch.zeros(512), torch.zeros(512), ent)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"iter {it}: pi = {logits.softmax(-1).detach().numpy().round(3)}, k3 = {st['k3']:.4f}, clipfrac = {st['clipfrac']:.2f}")
check("PPO prefers the best arm", logits.softmax(-1)[1] > 0.6)

# %% [markdown]
# ## Say it out loud
# 1. What do λ=0 and λ=1 give, and what is the bias/variance trade-off in between? Why are the value targets $A+V$?
# 2. Where exactly does `done` enter GAE, and why twice? What goes wrong if you forget it in the recursive term only?
# 3. Termination vs truncation (time limit): what should GAE do at a truncated step, and what does CleanRL do?
# 4. Walk through the four sign cases of the clipped objective. When is the gradient exactly zero? Does clipping
#    bound how far the ratio can move in an epoch?
# 5. `approx_kl` in CleanRL: which KL direction does it estimate, why is $k_3$ preferred over $k_1$, and what value
#    range is normal? What would you do if it spikes?
# 6. Why normalize advantages per minibatch? Does value clipping help? What does the entropy bonus do, and what sign
#    does it get in the loss?
# 7. In LLM RLHF with PPO, what are the "states", "actions" and per-token rewards, and what γ, λ are typical?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. λ=0: $\hat A_t=\delta_t$, the one-step TD residual: low variance, but biased whenever $V$ is wrong. λ=1:
#    $\sum_l\gamma^l r_{t+l} - V(s_t)$, the Monte-Carlo advantage: unbiased for any $V$ (the baseline does not bias it),
#    high variance (GAE eqs. 17-18). λ in between is an exponentially weighted mix of k-step estimators (eq. 16);
#    0.95 is the usual default. $A+V$ equals the λ-return, so the critic is regressed onto TD(λ) targets.
# 2. In $\delta_t$ (do not bootstrap $V(s_{t+1})$, which belongs to the next episode after an auto-reset) and in
#    the recursion $\gamma\lambda(1-d_t)\hat A_{t+1}$ (do not let the next episode's advantages leak into this one).
#    If you mask only $\delta$, advantages at the end of an episode absorb the start of the next one. The λ=1 check
#    catches it.
# 3. A time-limit truncation is not a real terminal state: the correct target bootstraps $\gamma V(s_{final})$ of the
#    true final observation (Gymnasium returns it in `info`), but the advantage recursion still stops there. CleanRL
#    (like the original baselines) treats `terminated or truncated` as done and never bootstraps. The 37-details post
#    keeps this on purpose for fidelity and notes that the principled version bootstraps.
# 4. With A>0 the min picks the clipped term only when $r>1+\epsilon$: zero gradient, no incentive to push the
#    good action further. When $r<1-\epsilon$ the unclipped term is the smaller one, so the gradient survives and pushes
#    $r$ back up. With A<0 it is mirrored: zero gradient when $r<1-\epsilon$, full gradient when $r>1+\epsilon$ (the bad
#    action is still pushed down). So $L^{CLIP}$ is a pessimistic lower bound on the unclipped surrogate. Clipping does
#    not hard-bound $r$: over several epochs and minibatches, shared parameters can push ratios well outside
#    $[1-\epsilon,1+\epsilon]$. That is why people watch approx-KL and sometimes stop early (`target_kl`).
# 5. Samples come from $\pi_{old}$, $r=\pi_\theta/\pi_{old}$, so both estimate $\mathrm{KL}[\pi_{old}\|\pi_\theta]$.
#    $k_1$ is unbiased but negative for about half the samples (high variance); $k_3$ adds the zero-mean control
#    variate $r-1$, is still unbiased, and is always $\ge 0$ since $\log r\le r-1$ (Schulman's blog). The 37-details
#    post says approx-KL usually stays below about 0.02; a spike means updates are too large (LR, too many
#    epochs, a bug in the ratio such as stale `old_logp`). Lower LR, use fewer epochs, or early-stop on `target_kl`.
# 6. Normalization makes the policy-gradient scale invariant to reward scale and keeps the clip range meaningful.
#    Per-minibatch is the reference implementation's choice, and Andrychowicz et al. found it matters little.
#    Value clipping: Engstrom et al. 2020 found no evidence it helps and Andrychowicz et al. 2021 suggest it can hurt
#    (both as summarized in the 37-details post). It is kept for fidelity. The entropy bonus is *maximized*, so the
#    loss gets $-c_{ent}H$. It slows premature collapse to a deterministic policy.
# 7. State = prompt + tokens so far, action = next token, episode = one response. The reward-model score arrives at
#    the last token, and the reward is penalized by $\beta\log(\pi_\theta/\pi_{ref})$ (Ziegler et al. 2019, arXiv:1909.08593;
#    InstructGPT eq. 2, arXiv:2203.02155), usually applied per token in implementations. Typical choices are γ=1 and λ
#    close to 1: rlhf-book's `configs/ppo.yaml` uses γ=1.0, λ=0.98, and its `compute_gae` puts the scalar reward on the
#    last response token. The PPO loss is the same as here but averaged over response tokens with a mask (A3.17
#    explains why the aggregation matters).
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
