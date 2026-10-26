# %% [markdown]
# # A3.3 · PCA via SVD, the covariance eigendecomposition, and Eckart–Young
#
# **ID:** A3.3 · **Target:** 15 min · **Plan week:** 2 · **Part:** A
#
# **The drill.** PCA two ways: SVD of the centered data and `eigh` of the covariance. Then explained-variance ratios,
# projection and reconstruction, and agreement with `sklearn.decomposition.PCA` **up to the sign** of each component.
# Finally, check the Eckart–Young theorem numerically: the best rank-$k$ approximation error in Frobenius norm is
# $\sqrt{\sum_{i>k}\sigma_i^2}$, and in spectral norm it is $\sigma_{k+1}$.
# Guide A3: "PCA via SVD" (blank-editor classic, under 20 min).
#
# **Rules.** Torch only inside your functions (`torch.linalg.svd`, `torch.linalg.eigh`). Float64. Components are rows
# (like sklearn's `components_`, shape `(k, D)`).
#
# ## Sources
# - C. Eckart, G. Young, *The approximation of one matrix by another of lower rank*, Psychometrika 1(3):211-218, 1936,
#   DOI 10.1007/BF02288367 — https://doi.org/10.1007/BF02288367
# - J. Shlens, *A Tutorial on Principal Component Analysis*, 2014, arXiv:1404.1100 — https://arxiv.org/abs/1404.1100
#   (PCA via SVD vs covariance eigendecomposition).
# - scikit-learn `PCA` (centering, `explained_variance_` with $n-1$, `explained_variance_ratio_`, sign convention via
#   `svd_flip`): https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# - `torch.linalg.svd` / `torch.linalg.eigh` (ascending eigenvalues):
#   https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html ·
#   https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigh.html
# - N. Halko, P.-G. Martinsson, J. A. Tropp, *Finding structure with randomness: Probabilistic algorithms for constructing
#   approximate matrix decompositions*, SIAM Review 53(2):217-288, 2011, arXiv:0909.4061 — https://arxiv.org/abs/0909.4061
# - scikit-learn `load_digits` (bundled): https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

# %%
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from prep import Session, check, check_close, seed_everything

s = Session("A3.3", target_min=15)
seed_everything(0)
dd = dict(dtype=torch.float64)
X = torch.tensor(load_digits().data, **dd)          # (1797, 64)
n, D = X.shape
k = 10

# %% [markdown]
# ## 1. PCA via SVD
# Center: $X_c = X - \bar x$. Thin SVD $X_c = U S V^\top$. The principal directions are the rows of $V^\top$; the variance
# along direction $i$ is $\sigma_i^2/(n-1)$; the explained-variance ratio is $\sigma_i^2/\sum_j\sigma_j^2$. Return
# `(mean (D,), components (k, D), explained_variance (k,), ratio (k,))`.

# %%
def pca_svd(X, k):
    ### BEGIN SOLUTION
    mean = X.mean(0)
    U, S, Vt = torch.linalg.svd(X - mean, full_matrices=False)
    var = S ** 2 / (X.shape[0] - 1)
    return mean, Vt[:k], var[:k], var[:k] / var.sum()
    ### END SOLUTION


def align_signs(A, B):
    """Flip the rows of A so that each has a non-negative dot product with the matching row of B."""
    return A * torch.sign((A * B).sum(1, keepdim=True))


mean, comps, ev, evr = pca_svd(X, k)
sk = PCA(n_components=k, svd_solver="full").fit(X.numpy())
sk_comps = torch.tensor(sk.components_)
check_close("mean", mean, sk.mean_, atol=1e-10)
check_close("components (up to sign)", align_signs(comps, sk_comps), sk_comps, atol=1e-8)
check_close("explained_variance_", ev, sk.explained_variance_, atol=1e-8)
check_close("explained_variance_ratio_", evr, sk.explained_variance_ratio_, atol=1e-10)
check_close("components orthonormal", comps @ comps.T, torch.eye(k, **dd), atol=1e-10)
print("top-10 explained variance ratio:", [round(v, 3) for v in evr.tolist()], f"(cumulative {evr.sum().item():.3f})")

# %% [markdown]
# ## 2. PCA via the covariance matrix
# $\Sigma = X_c^\top X_c/(n-1)$ is symmetric, so use `eigh`. It returns eigenvalues in **ascending** order: sort them
# descending. The eigenvalues equal $\sigma_i^2/(n-1)$. This is fine for small $D$, but it squares the condition number
# (as with the normal equations in A3.1), and forming $\Sigma$ costs $O(nD^2)$.

# %%
def pca_eig(X, k):
    ### BEGIN SOLUTION
    Xc = X - X.mean(0)
    evals, evecs = torch.linalg.eigh(Xc.T @ Xc / (X.shape[0] - 1))
    order = evals.argsort(descending=True)[:k]
    return evecs[:, order].T, evals[order]
    ### END SOLUTION


comps_e, ev_e = pca_eig(X, k)
check_close("eig components == SVD components (up to sign)", align_signs(comps_e, comps), comps, atol=1e-6)
check_close("eig eigenvalues == explained variance", ev_e, ev, atol=1e-8)

# %% [markdown]
# ## 3. Project and reconstruct
# Scores $Z = (X - \bar x)V_k$ with shape $(n, k)$; reconstruction $\hat X = Z V_k^\top + \bar x$. The squared reconstruction error
# $\|X - \hat X\|_F^2$ equals the variance left out, $(n-1)\sum_{i>k}\lambda_i$.

# %%
def pca_transform(X, mean, comps):
    ### BEGIN SOLUTION
    return (X - mean) @ comps.T
    ### END SOLUTION


def pca_reconstruct(Z, mean, comps):
    ### BEGIN SOLUTION
    return Z @ comps + mean
    ### END SOLUTION


Z = pca_transform(X, mean, comps)
check_close("transform == sklearn (up to sign)", Z * torch.sign((comps * sk_comps).sum(1)), sk.transform(X.numpy()), atol=1e-7)
Xhat = pca_reconstruct(Z, mean, comps)
check_close("reconstruct == sklearn inverse_transform", Xhat, sk.inverse_transform(sk.transform(X.numpy())), atol=1e-7)
_, _, ev_all, _ = pca_svd(X, D)
check_close("||X - Xhat||_F^2 == (n-1) * sum of dropped eigenvalues", ((X - Xhat) ** 2).sum(), (n - 1) * ev_all[k:].sum(), atol=1e-6)

# %% [markdown]
# ## 4. Eckart–Young, numerically
# For any matrix $A$ with singular values $\sigma_1\ge\sigma_2\ge\dots$, the truncated SVD $A_k = U_k S_k V_k^\top$ is the best rank-$k$
# approximation in both norms: $\|A - A_k\|_F = \sqrt{\sum_{i>k}\sigma_i^2}$ and $\|A - A_k\|_2 = \sigma_{k+1}$. Write `best_rank_k`
# and check both identities; then check that a few other rank-$k$ matrices do worse.

# %%
def best_rank_k(A, k):
    ### BEGIN SOLUTION
    U, S, Vt = torch.linalg.svd(A, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]
    ### END SOLUTION


torch.manual_seed(0)
A = torch.randn(40, 25, **dd) @ torch.diag(torch.logspace(0, -2, 25, **dd)) @ torch.randn(25, 25, **dd)
S = torch.linalg.svdvals(A)
for kk in (1, 5, 12):
    Ak = best_rank_k(A, kk)
    check("rank(A_k) == k", torch.linalg.matrix_rank(Ak).item() == kk)
    check_close(f"k={kk}: ||A - A_k||_F == sqrt(sum tail sigma^2)", torch.linalg.matrix_norm(A - Ak), S[kk:].pow(2).sum().sqrt(), atol=1e-9)
    check_close(f"k={kk}: ||A - A_k||_2 == sigma_(k+1)", torch.linalg.matrix_norm(A - Ak, ord=2), S[kk], atol=1e-9)
    best = torch.linalg.matrix_norm(A - Ak).item()
    Qs = [torch.linalg.qr(torch.randn(40, kk, **dd)).Q for _ in range(5)]   # random rank-k projections Q Q^T A
    others = [torch.linalg.matrix_norm(A - Q @ (Q.T @ A)).item() for Q in Qs]
    check(f"k={kk}: random rank-k projections do worse", min(others) > best, f"{min(others):.3f} <= {best:.3f}")

# %% [markdown]
# ## Say it out loud
# 1. Why must you center the data before PCA? What goes wrong if you don't?
# 2. SVD of $X_c$ vs `eigh` of the covariance: same answer? Which do you prefer, and when is the covariance route better?
# 3. Why are components only defined up to sign (and up to rotation when eigenvalues repeat)? How does sklearn fix the sign?
# 4. State Eckart–Young. How does it justify PCA as "best linear dimensionality reduction"?
# 5. How do you pick $k$? What does the explained-variance ratio miss?
# 6. Should you standardize features before PCA? What changes?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. PCA finds directions of maximal **variance**, i.e. the eigenvectors of the covariance, which is defined around the mean. Without
#    centering, the first "component" mostly points at the mean vector (it captures $\|\bar x\|^2$), and the rest are distorted.
# 2. Same directions and variances: $X_c^\top X_c = V S^2 V^\top$. SVD is more accurate, because forming $X_c^\top X_c$ squares the condition number,
#    and small eigenvalues lose relative precision. The covariance route costs $O(nD^2 + D^3)$ and can be cheaper when $n \gg D$ (or streamed in
#    one pass). For a few top components of a huge matrix, use randomized SVD (Halko, Martinsson & Tropp 2011) or Lanczos.
# 3. If $v$ is an eigenvector, so is $-v$. With repeated eigenvalues, any rotation within the eigenspace also works. sklearn applies `svd_flip`,
#    with `u_based_decision=False` in sklearn 1.9's `_pca.py`: the largest-|value| entry of each component row is made positive. Compare
#    components only after aligning signs.
# 4. Among all matrices of rank $\le k$, the truncated SVD minimizes $\|A-B\|_F$ (error $\sqrt{\sum_{i>k}\sigma_i^2}$) and $\|A-B\|_2$ (error $\sigma_{k+1}$)
#    (Eckart & Young 1936). Applied to $X_c$: projecting onto the top-$k$ principal subspace gives the smallest squared reconstruction error of
#    any rank-$k$ linear projection, equivalently it maximizes the retained variance. A linear autoencoder with MSE loss learns the same subspace.
# 5. Cumulative explained-variance threshold (e.g. 90-95%), a scree/elbow plot, cross-validated reconstruction or downstream performance.
#    Variance is not relevance: a low-variance direction can be the discriminative one (PCA is unsupervised), and outliers can dominate variance.
# 6. If features have different units/scales, yes: otherwise the largest-scale feature dominates the first component. Standardizing means PCA on
#    the correlation matrix instead of the covariance. For homogeneous features (pixels, as here) you usually keep the raw scale.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
