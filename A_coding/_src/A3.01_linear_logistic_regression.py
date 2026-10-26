# %% [markdown]
# # A3.1 · Linear, ridge and logistic regression from scratch
#
# **ID:** A3.1 · **Target:** 20 min · **Plan week:** 2 · **Part:** A
#
# **The drill.** (1) Ordinary least squares by `lstsq` (not `inv`) and by gradient descent; (2) ridge regression in closed
# form with an unpenalized intercept; (3) L2-regularized logistic regression by gradient descent with a **hand-derived
# gradient**, then the same with autograd. Match scikit-learn's `LinearRegression`, `Ridge` and `LogisticRegression`,
# which means getting the **C ↔ λ conversion** right.
# Guide A3: "Linear/logistic regression with gradient descent" (blank-editor classic, under 20 min).
#
# **Rules.** Torch or NumPy only inside your functions (sklearn is the reference). Float64 throughout.
#
# ## Sources
# - scikit-learn User Guide §1.1 *Linear Models*: OLS, the ridge objective $\|Xw-y\|_2^2+\alpha\|w\|_2^2$, and the
#   logistic objective $\frac1S\sum_i s_i\,\ell_i + \frac{r(w)}{S\,C}$ with $r(w)=\frac12\|w\|_2^2$ and an unpenalized
#   intercept (except for liblinear): https://scikit-learn.org/stable/modules/linear_model.html
#   (sections `#ordinary-least-squares`, `#ridge-regression-and-classification`, `#logistic-regression`, `#solvers`).
# - `sklearn.linear_model.LogisticRegression` (C = inverse regularization strength):
#   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# - T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer 2009, §3.2 (least squares),
#   §3.4.1 (ridge closed form and centering the intercept), §4.4 (logistic regression):
#   https://hastie.su.domains/ElemStatLearn/
# - `torch.linalg.lstsq` (QR/SVD-based least squares): https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html
# - scikit-learn `load_breast_cancer` (bundled with sklearn, no download):
#   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

# %%
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from prep import Session, check, check_close, seed_everything

s = Session("A3.1", target_min=20)
seed_everything(0)
dd = dict(dtype=torch.float64)

n, d = 200, 5
X = torch.randn(n, d, **dd) * torch.tensor([1.0, 2.0, 0.5, 3.0, 1.0], **dd) + 1.0
w_true, b_true = torch.tensor([1.5, -2.0, 0.0, 0.7, 3.0], **dd), 3.0
y = X @ w_true + b_true + 0.1 * torch.randn(n, **dd)

# %% [markdown]
# ## 1. OLS with `lstsq`
# Append a column of ones and solve $\min_\theta \|[X, \mathbf 1]\theta - y\|^2$. The normal equations
# $X^\top X\theta = X^\top y$ give the same answer in exact arithmetic, but forming $X^\top X$ **squares the condition
# number**. `lstsq` works on $X$ directly (QR/SVD). Return `(w (d,), b scalar tensor)`.

# %%
def ols_lstsq(X, y):
    ### BEGIN SOLUTION
    A = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    theta = torch.linalg.lstsq(A, y[:, None]).solution.squeeze(1)
    return theta[:-1], theta[-1]
    ### END SOLUTION


w_ols, b_ols = ols_lstsq(X, y)
sk = LinearRegression().fit(X.numpy(), y.numpy())
check_close("OLS w vs sklearn", w_ols, sk.coef_, atol=1e-10)
check_close("OLS b vs sklearn", b_ols, sk.intercept_, atol=1e-10)

# %% [markdown]
# **Why not `inv(XᵀX)`?** A noiseless system with two nearly collinear columns ($\kappa(X)\approx10^{6}$): `lstsq`'s error scales
# with $\kappa(X)\,u$, the normal equations' error with $\kappa(X)^2 u$ ($u\approx10^{-16}$).

# %%
torch.manual_seed(1)
Xc = torch.randn(100, 3, **dd)
Xc[:, 2] = Xc[:, 1] + 1e-6 * torch.randn(100, **dd)
wc = torch.tensor([1.0, 2.0, -1.0], **dd)
yc = Xc @ wc
err_lstsq = (torch.linalg.lstsq(Xc, yc[:, None]).solution.squeeze(1) - wc).abs().max().item()
err_inv = (torch.linalg.inv(Xc.T @ Xc) @ Xc.T @ yc - wc).abs().max().item()
print(f"cond(X) = {torch.linalg.cond(Xc).item():.1e}; max error: lstsq {err_lstsq:.1e} vs inv(X^T X) {err_inv:.1e}")
check("lstsq is more accurate than the normal equations here", err_lstsq < err_inv / 100)

# %% [markdown]
# ## 2. OLS by gradient descent
# $L(w,b) = \frac1n\|Xw + b - y\|^2$, $\nabla_w L = \frac2n X^\top r$, $\partial_b L = \frac2n\sum r$ with $r = Xw+b-y$.
# Start from zeros; run `steps` full-batch steps with learning rate `lr`. It converges to the `lstsq` solution
# (the condition number of $X^\top X/n$, with the features unscaled, decides how many steps that takes).

# %%
def linreg_gd(X, y, lr, steps):
    ### BEGIN SOLUTION
    n = len(X)
    w, b = torch.zeros(X.shape[1], dtype=X.dtype), torch.zeros((), dtype=X.dtype)
    for _ in range(steps):
        r = X @ w + b - y
        w = w - lr * 2 / n * (X.T @ r)
        b = b - lr * 2 / n * r.sum()
    return w, b
    ### END SOLUTION


A1 = torch.cat([X, torch.ones(n, 1, **dd)], 1)
L_smooth = 2 * torch.linalg.eigvalsh(A1.T @ A1 / n).max().item()   # Lipschitz constant of the gradient
w_gd, b_gd = linreg_gd(X, y, lr=1 / L_smooth, steps=20000)
check_close("GD w -> lstsq", w_gd, w_ols, atol=1e-6)
check_close("GD b -> lstsq", b_gd, b_ols, atol=1e-6)
w_1, b_1 = linreg_gd(X, y, lr=0.01, steps=1)   # one step from zero: -lr * grad at 0 = lr * (2/n) [X^T y, sum y]
check_close("one GD step uses the exact gradient", torch.cat([w_1, b_1[None]]), 0.01 * 2 / n * torch.cat([X.T @ y, y.sum()[None]]), atol=1e-12)

# %% [markdown]
# ## 3. Ridge in closed form
# sklearn's `Ridge(alpha)` minimizes $\|Xw + b - y\|_2^2 + \alpha\|w\|_2^2$: a **sum** of squares (not a mean) and **no
# penalty on $b$**. Center $X$ and $y$, solve $(X_c^\top X_c + \alpha I)\,w = X_c^\top y_c$ with `torch.linalg.solve`, then
# $b = \bar y - \bar x^\top w$.

# %%
def ridge(X, y, alpha):
    ### BEGIN SOLUTION
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w = torch.linalg.solve(Xc.T @ Xc + alpha * torch.eye(X.shape[1], dtype=X.dtype), Xc.T @ yc)
    return w, ym - xm @ w
    ### END SOLUTION


for alpha in (0.1, 10.0, 1000.0):
    w_r, b_r = ridge(X, y, alpha)
    skr = Ridge(alpha=alpha).fit(X.numpy(), y.numpy())
    check_close(f"ridge alpha={alpha} w", w_r, skr.coef_, atol=1e-8)
    check_close(f"ridge alpha={alpha} b", b_r, skr.intercept_, atol=1e-8)

# %% [markdown]
# ## 4. Logistic regression, manual gradient
# Data: breast-cancer (569 × 30), standardized. Objective, with $p = \sigma(Xw + b)$:
# $$J(w,b) = \frac1n\sum_i \mathrm{BCE}(p_i, y_i) + \frac\lambda2\|w\|^2,\qquad
# \nabla_w J = \frac1n X^\top(p - y) + \lambda w,\quad \partial_b J = \frac1n\sum_i(p_i - y_i).$$
# Compute the BCE stably from logits $z$: $\mathrm{BCE} = \mathrm{softplus}(z) - yz$.
# `logreg_loss_grad` returns `(J, grad_w, grad_b)`; `logreg_gd` runs GD from zeros.

# %%
def logreg_loss_grad(X, y, w, b, lam):
    ### BEGIN SOLUTION
    z = X @ w + b
    J = (torch.nn.functional.softplus(z) - y * z).mean() + 0.5 * lam * (w @ w)
    r = torch.sigmoid(z) - y
    return J, X.T @ r / len(X) + lam * w, r.mean()
    ### END SOLUTION


def logreg_gd(X, y, lam, lr, steps):
    ### BEGIN SOLUTION
    w, b = torch.zeros(X.shape[1], dtype=X.dtype), torch.zeros((), dtype=X.dtype)
    for _ in range(steps):
        _, gw, gb = logreg_loss_grad(X, y, w, b, lam)
        w, b = w - lr * gw, b - lr * gb
    return w, b
    ### END SOLUTION


data = load_breast_cancer()
Xb = torch.tensor(data.data, **dd)
Xb = (Xb - Xb.mean(0)) / Xb.std(0)
yb = torch.tensor(data.target, **dd)
nb, lam = len(Xb), 1e-2

w0, b0 = torch.randn(30, **dd) * 0.1, torch.tensor(0.3, **dd)
w0a, b0a = w0.clone().requires_grad_(), b0.clone().requires_grad_()
J0, gw0, gb0 = logreg_loss_grad(Xb, yb, w0, b0, lam)
za = Xb @ w0a + b0a
Ja = torch.nn.functional.binary_cross_entropy_with_logits(za, yb) + 0.5 * lam * (w0a @ w0a)
Ja.backward()
check_close("logreg loss vs BCEWithLogits + L2", J0, Ja, atol=1e-12)
check_close("logreg manual grad_w == autograd", gw0, w0a.grad, atol=1e-12)
check_close("logreg manual grad_b == autograd", gb0, b0a.grad, atol=1e-12)

Lb = 0.25 * torch.linalg.eigvalsh(torch.cat([Xb, torch.ones(nb, 1, **dd)], 1).T @ torch.cat([Xb, torch.ones(nb, 1, **dd)], 1) / nb).max().item() + lam
w_lr, b_lr = logreg_gd(Xb, yb, lam, lr=1 / Lb, steps=20000)

# %% [markdown]
# ## 5. Matching sklearn: the C ↔ λ conversion
# sklearn (binary, unit sample weights, L2) minimizes $\frac1n\sum_i \ell_i + \frac{1}{2nC}\|w\|^2$ (User Guide §1.1.11.1:
# $\frac1S\sum s_i\ell_i + \frac{r(w)}{SC}$ with $S = n$). This is the same as our $J$ when $\frac\lambda2 = \frac1{2nC}$, i.e.
# $$C = \frac{1}{n\lambda}.$$
# Equivalently, the classic form $C\sum_i\ell_i + \frac12\|w\|^2$ is $nC$ times $J$, with the same minimizer. The intercept is
# not penalized (lbfgs). Write `C_from_lambda(lam, n)`; the check fits sklearn with it, and also shows that the naive
# `C = 1/λ` gives a visibly different model.

# %%
def C_from_lambda(lam, n):
    ### BEGIN SOLUTION
    return 1.0 / (n * lam)
    ### END SOLUTION


skl = LogisticRegression(C=C_from_lambda(lam, nb), tol=1e-12, max_iter=100_000).fit(Xb.numpy(), yb.numpy())
check_close("logreg GD w vs sklearn", w_lr, skl.coef_[0], atol=1e-5)
check_close("logreg GD b vs sklearn", b_lr, skl.intercept_[0], atol=1e-5)
naive = LogisticRegression(C=1 / lam, tol=1e-12, max_iter=100_000).fit(Xb.numpy(), yb.numpy())
gap = np.abs(naive.coef_[0] - w_lr.numpy()).max()
print(f"C = 1/lambda instead of 1/(n lambda): max |w| difference {gap:.3f}")
check("naive C conversion is visibly wrong", gap > 0.1)
print("train accuracy:", ((Xb @ w_lr + b_lr > 0).double() == yb).double().mean().item())

# %% [markdown]
# ## 6. Same model with autograd
# Write the GD loop again, but get the gradient from `.backward()` on $J$ (use `F.binary_cross_entropy_with_logits`).
# Same init, `lr` and number of steps, so the trajectory should match the manual version to rounding.

# %%
def logreg_autograd(X, y, lam, lr, steps):
    ### BEGIN SOLUTION
    w = torch.zeros(X.shape[1], dtype=X.dtype, requires_grad=True)
    b = torch.zeros((), dtype=X.dtype, requires_grad=True)
    for _ in range(steps):
        J = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 0.5 * lam * (w @ w)
        gw, gb = torch.autograd.grad(J, (w, b))
        with torch.no_grad():
            w -= lr * gw
            b -= lr * gb
    return w.detach(), b.detach()
    ### END SOLUTION


w_ag, b_ag = logreg_autograd(Xb, yb, lam, lr=1 / Lb, steps=2000)
w_m2, b_m2 = logreg_gd(Xb, yb, lam, lr=1 / Lb, steps=2000)
check_close("autograd GD == manual GD (2000 steps)", torch.cat([w_ag, b_ag[None]]), torch.cat([w_m2, b_m2[None]]), atol=1e-10)

# %% [markdown]
# ## Say it out loud
# 1. Derive the logistic-regression gradient. Why does it have the same $X^\top(\text{prediction} - y)$ form as linear regression?
# 2. Why `lstsq`/QR/Cholesky instead of `inv(XᵀX)`? What does ridge do to the condition number?
# 3. Why is the intercept usually not penalized, and how does centering implement that in closed form?
# 4. sklearn's `C` vs your $\lambda$: derive the conversion. What changes if you use sum-loss instead of mean-loss?
# 5. Is the logistic loss convex? Strongly convex? What happens without regularization on linearly separable data?
# 6. How would you pick the GD step size here, and what controls the number of iterations?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. $\ell = \mathrm{softplus}(z) - yz$ with $z = x^\top w + b$, so $\partial\ell/\partial z = \sigma(z) - y$, and the chain rule gives $x(\sigma(z) - y)$.
#    Averaged: $\frac1n X^\top(p-y)$, plus $\lambda w$ from the penalty. It matches linear regression because both are GLMs with their
#    canonical link: the gradient is always features × (mean prediction − target) (Bishop PRML §4.3.2, ESL §4.4).
# 2. $\kappa(X^\top X) = \kappa(X)^2$, so the normal equations lose twice as many digits (the demo above: about $10^{-11}$ vs $10^{-3}$). QR/SVD
#    (`lstsq`) work on $X$. For ridge, $X^\top X + \alpha I$ is SPD with eigenvalues $\ge\alpha$, so the condition number is at most
#    $(\sigma_{max}^2+\alpha)/\alpha$; use `solve`/Cholesky, never `inv`.
# 3. Penalizing $b$ would make the fit depend on where the origin of $y$ is, and shifting $y$ by a constant should only shift $b$. With centered $X$ and $y$
#    the optimal $w$ does not involve $b$; then $b = \bar y - \bar x^\top w$ (ESL §3.4.1).
# 4. sklearn: $\frac1n\sum\ell_i + \frac{1}{2nC}\|w\|^2$; ours: $\frac1n\sum\ell_i + \frac\lambda2\|w\|^2$, so $C = 1/(n\lambda)$. With a summed loss
#    $\sum\ell_i + \frac{\lambda'}2\|w\|^2$ it is $C = 1/\lambda'$. Getting this wrong changes the regularization strength by a factor of $n$.
#    For `Ridge`, alpha multiplies $\|w\|^2$ next to a summed squared error, so a mean-MSE $+\lambda\|w\|^2$ objective corresponds to $\alpha = n\lambda$.
# 5. Convex: the Hessian $\frac1n X^\top\mathrm{diag}(p(1-p))X \succeq 0$. Strongly convex only with $\lambda>0$ (or full-rank $X$ and $p$ bounded away
#    from 0/1). On separable data without regularization the loss keeps decreasing as $\|w\|\to\infty$: no finite minimizer, and GD diverges in
#    norm (while its direction converges to the max-margin separator).
# 6. With an $L$-smooth objective, $\eta = 1/L$ guarantees descent, where $L \le \frac14\lambda_{max}(\tilde X^\top\tilde X/n) + \lambda$ ($\tilde X$ includes
#    the ones column); that is what the code uses. GD needs $O(\kappa\log(1/\varepsilon))$ steps with $\kappa = L/\mu$ and $\mu\ge\lambda$.
#    Standardizing features reduces $\kappa$. Newton/IRLS or L-BFGS converge in a handful of iterations.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
