# %% [markdown]
# # A3.2 · k-means (k-means++ init, Lloyd) and k-NN from scratch
#
# **ID:** A3.2 · **Target:** 20 min · **Plan week:** 2 · **Part:** A
#
# **The drill.** Vectorized squared distances; k-means++ seeding by $D^2$ sampling; Lloyd iterations with empty-cluster
# handling and inertia; a k-NN classifier with majority vote and a defined tie-break. Checks: k-means matches
# `sklearn.cluster.KMeans` **exactly** (centers, labels, inertia) when started from the same initial centers, and k-NN
# predictions match `KNeighborsClassifier(algorithm="brute")` on every test point.
# Guide A3: "k-means; k-NN" (blank-editor classic, under 20 min).
#
# **Rules.** Torch only inside your functions. No `torch.cdist`, no sklearn. Python loops are fine over Lloyd iterations and
# over the k centers of k-means++, never over data points.
#
# ## Sources
# - D. Arthur, S. Vassilvitskii, *k-means++: The Advantages of Careful Seeding*, Proc. 18th ACM-SIAM Symposium on
#   Discrete Algorithms (SODA 2007), pp. 1027-1035 — https://dl.acm.org/doi/10.5555/1283383.1283494
#   ($D^2$ sampling, and its $O(\log k)$-competitive guarantee in expectation).
# - S. P. Lloyd, *Least squares quantization in PCM*, IEEE Trans. Inf. Theory 28(2):129-137, 1982, DOI 10.1109/TIT.1982.1056489 —
#   https://doi.org/10.1109/TIT.1982.1056489
# - T. Cover, P. Hart, *Nearest neighbor pattern classification*, IEEE Trans. Inf. Theory 13(1):21-27, 1967,
#   DOI 10.1109/TIT.1967.1053964 — https://doi.org/10.1109/TIT.1967.1053964
# - scikit-learn `KMeans` (inertia, `init=` array, `n_init`, Lloyd algorithm; relocation of empty clusters to far points):
#   https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html ·
#   `KNeighborsClassifier`: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# - Squared-distance expansion: N. Rougier et al., *100 NumPy exercises* #52, https://github.com/rougier/numpy-100 (MIT);
#   see also A2.1 puzzle E5.

# %%
import torch
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

from prep import Session, check, check_close, seed_everything

s = Session("A3.2", target_min=20)
seed_everything(0)
dd = dict(dtype=torch.float64)

# %% [markdown]
# ## 1. Squared distances
# $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2a^\top b$: one matmul, $(N, M)$ memory. Clamp at 0 (rounding can go negative).

# %%
def sqdist(A, B):
    """A (N, D), B (M, D) -> (N, M) squared Euclidean distances."""
    ### BEGIN SOLUTION
    return ((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T).clamp(min=0)
    ### END SOLUTION


A_, B_ = torch.randn(7, 3, **dd), torch.randn(5, 3, **dd)
check_close("sqdist vs cdist^2", sqdist(A_, B_), torch.cdist(A_, B_) ** 2, atol=1e-10)
P_ = torch.randn(64, 16, **dd) + 1e7   # far from the origin: the expansion cancels and can go negative
check("sqdist >= 0 under cancellation", bool((sqdist(P_, P_) >= 0).all()), "clamp at 0")

# %% [markdown]
# ## 2. k-means++ seeding
# Pick the first center uniformly at random among the data points. Then, repeatedly, pick the next center with
# probability proportional to $D(x)^2$, the squared distance from $x$ to its **nearest already-chosen** center.
# Use `torch.multinomial(..., generator=gen)` so runs are reproducible. Return the chosen **indices** too.

# %%
def next_center_probs(X, C):
    """X (N, D), C (j, D) chosen centers -> (N,) D^2 sampling probabilities."""
    ### BEGIN SOLUTION
    d2 = sqdist(X, C).min(1).values
    return d2 / d2.sum()
    ### END SOLUTION


def kmeans_pp_init(X, k, gen):
    """-> (centers (k, D), indices (k,))."""
    ### BEGIN SOLUTION
    idx = [int(torch.randint(len(X), (1,), generator=gen))]
    for _ in range(1, k):
        p = next_center_probs(X, X[idx])
        idx.append(int(torch.multinomial(p, 1, generator=gen)))
    idx = torch.tensor(idx)
    return X[idx].clone(), idx
    ### END SOLUTION


Xs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0], [-1.0, 0.0]], **dd)
p_ref = torch.tensor([0.0, 1.0, 4.0, 18.0, 1.0], **dd) / 24
check_close("D^2 probabilities, one center", next_center_probs(Xs, Xs[:1]), p_ref)
p_ref2 = torch.tensor([0.0, 1.0, 4.0, 0.0, 1.0], **dd) / 6
check_close("D^2 probabilities use the NEAREST center", next_center_probs(Xs, Xs[[0, 3]]), p_ref2)

# statistical check: 5 tight, well-separated blobs. k-means++ almost always seeds one center per blob; uniform seeding rarely does.
Xb, yb = make_blobs(n_samples=500, centers=[[0, 0], [10, 0], [0, 10], [10, 10], [5, 20]], cluster_std=0.5, random_state=0)
Xb, yb = torch.tensor(Xb, **dd), torch.tensor(yb)
gen = torch.Generator().manual_seed(0)
hits_pp = hits_unif = 0
for t in range(200):
    _, idx = kmeans_pp_init(Xb, 5, gen)
    hits_pp += len(set(yb[idx].tolist())) == 5
    hits_unif += len(set(yb[torch.randperm(500, generator=gen)[:5]].tolist())) == 5
print(f"one center per blob: k-means++ {hits_pp / 200:.2f}, uniform {hits_unif / 200:.2f} (theory for uniform: 5!/5^5 = 0.04)")
# exact D^2 law of the second center, given the first: compare empirical frequencies on the 5-point toy set
second = torch.zeros(5, **dd)
for t in range(10000):
    _, idx = kmeans_pp_init(Xs, 2, gen)
    if idx[0] == 0:
        second[idx[1]] += 1
tv = 0.5 * (second / second.sum() - p_ref).abs().sum().item()
check("second center follows D^2 sampling (total variation < 0.05)", tv < 0.05, f"TV = {tv:.3f}; sample, do not take the argmax")
_, idx = kmeans_pp_init(Xb, 5, gen)
check("k-means++ returns k distinct data points", len(set(idx.tolist())) == 5 and torch.equal(kmeans_pp_init(Xb, 5, torch.Generator().manual_seed(3))[0],
      Xb[kmeans_pp_init(Xb, 5, torch.Generator().manual_seed(3))[1]]))
check("k-means++ seeds one center per blob >= 80% of the time", hits_pp / 200 >= 0.8, f"{hits_pp / 200}")

# %% [markdown]
# ## 3. Lloyd's algorithm with empty-cluster handling
# Repeat: assign each point to its nearest center; move each center to the mean of its points. Stop when the
# assignment no longer changes (or after `max_iter`). Return `(centers, labels, inertia)` with inertia
# $= \sum_i \|x_i - c_{\ell_i}\|^2$ for the **returned** centers and labels.
# **Empty cluster:** a center with no points would become 0/0. Move it to the point that is currently farthest from its
# assigned center (sklearn does the same kind of relocation), so you keep k clusters.

# %%
def kmeans(X, C0, max_iter=300):
    ### BEGIN SOLUTION
    C = C0.clone()
    labels = None
    for _ in range(max_iter):
        d2 = sqdist(X, C)
        new = d2.argmin(1)
        if labels is not None and torch.equal(new, labels):
            break
        labels = new
        counts = torch.bincount(labels, minlength=len(C)).to(X.dtype)
        sums = torch.zeros_like(C).index_add_(0, labels, X)
        C = sums / counts.clamp(min=1)[:, None]
        far = d2.gather(1, labels[:, None]).squeeze(1)
        for j in torch.nonzero(counts == 0).flatten().tolist():   # empty clusters (rare)
            i = int(far.argmax())
            C[j], far[i] = X[i], -1.0
    d2 = sqdist(X, C)
    labels = d2.argmin(1)
    return C, labels, d2.gather(1, labels[:, None]).sum()
    ### END SOLUTION


Xk, _ = make_blobs(n_samples=600, centers=6, cluster_std=2.5, random_state=1)   # overlapping: many Lloyd iterations
Xk = torch.tensor(Xk, **dd)
C0, _ = kmeans_pp_init(Xk, 6, torch.Generator().manual_seed(1))
C, labels, inertia = kmeans(Xk, C0)
sk = KMeans(n_clusters=6, init=C0.numpy(), n_init=1, algorithm="lloyd", max_iter=300, tol=0.0).fit(Xk.numpy())
check_close("k-means centers == sklearn (same init)", C, sk.cluster_centers_, atol=1e-8)
check("k-means labels == sklearn", bool((labels.numpy() == sk.labels_).all()))
check_close("k-means inertia == sklearn", inertia, sk.inertia_, atol=1e-6)
print(f"inertia {inertia.item():.2f} after {sk.n_iter_} sklearn iterations")

# empty cluster: the third center starts far from all data and gets no points in the first assignment
X2 = torch.cat([torch.randn(50, 2, **dd) + 5, torch.randn(50, 2, **dd) + 13])
C02 = torch.tensor([[5.0, 5.0], [13.0, 13.0], [100.0, 100.0]], **dd)
C2, l2, in2 = kmeans(X2, C02)
check("empty cluster: centers finite", bool(torch.isfinite(C2).all()), f"{C2}")
check("empty cluster: all k clusters used", len(l2.unique()) == 3, f"labels used: {l2.unique().tolist()}")
check("empty cluster: center moved into the data", bool((C2 > 0).all() and (C2 < 20).all()), f"{C2}")

# %% [markdown]
# ## 4. k-NN classifier
# For each test point, take the `k` nearest training points (`topk(..., largest=False)` on squared distances) and return
# the majority label. **Tie-break:** the smallest label among the tied classes (this is what sklearn's uniform-weight vote
# does). Vectorize the vote with a one-hot count: `argmax` returns the first maximum.

# %%
def knn_predict(Xtr, ytr, Xte, k, n_classes):
    ### BEGIN SOLUTION
    nn_idx = sqdist(Xte, Xtr).topk(k, dim=1, largest=False).indices      # (M, k)
    votes = torch.zeros(len(Xte), n_classes).scatter_add_(1, ytr[nn_idx], torch.ones(nn_idx.shape))
    return votes.argmax(1)
    ### END SOLUTION


Xa, ya = make_blobs(n_samples=400, centers=4, cluster_std=3.0, random_state=2)
Xa, ya = torch.tensor(Xa, **dd), torch.tensor(ya)
Xtr, ytr, Xte, yte = Xa[:300], ya[:300], Xa[300:], ya[300:]
for k in (1, 4, 7, 10):   # even k forces vote ties
    pred = knn_predict(Xtr, ytr, Xte, k, 4)
    skp = KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(Xtr.numpy(), ytr.numpy()).predict(Xte.numpy())
    check(f"kNN k={k} == sklearn on all {len(Xte)} test points", bool((pred.numpy() == skp).all()),
          f"{(pred.numpy() != skp).sum()} mismatches")
    print(f"   k={k}: accuracy {(pred == yte).double().mean().item():.3f}")
Xt_tie = torch.tensor([[0.0], [1.0], [-1.1], [1.2]], **dd)
check("kNN tie -> smallest label", knn_predict(Xt_tie, torch.tensor([2, 1, 1, 2]), torch.tensor([[0.0]], **dd), 4, 3).item() == 1)

# %% [markdown]
# ## Say it out loud
# 1. What objective does Lloyd's algorithm decrease, and why does each step not increase it? Does it find the global optimum?
# 2. What does k-means++ guarantee, and why sample proportional to $D^2$ rather than pick the farthest point?
# 3. How do you handle an empty cluster? Name two strategies.
# 4. Complexity of one Lloyd iteration and of a k-NN query batch? How do you scale k-NN to millions of points?
# 5. How does k-NN behave as $k$ grows? What about high dimensions? Why do feature scales matter for both algorithms?
# 6. How do you choose $k$ for k-means?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Inertia $\sum_i\|x_i - c_{\ell_i}\|^2$. The assignment step minimizes it over labels with the centers fixed, and the update step over centers
#    with the labels fixed (the mean minimizes the sum of squared distances). So it is coordinate descent, monotone, and terminates (finitely
#    many partitions). It reaches only a local optimum (k-means is NP-hard in general), hence good seeding and `n_init` restarts.
# 2. The expected inertia right after seeding is at most $8(\ln k + 2)$ times the optimal inertia (Arthur & Vassilvitskii 2007). Farthest-point
#    seeding is deterministic and picks outliers; $D^2$ sampling favors far regions in proportion to how much inertia they would remove, yet
#    rarely wastes a center on a single outlier. sklearn adds "greedy" local trials ($2 + \ln k$ candidates per step).
# 3. (a) Re-seed it at the point farthest from its current center, i.e. the one contributing the most inertia (as here; sklearn does a version of
#    this). (b) Split the largest cluster, or re-seed at a random point / by $D^2$ sampling. Never leave it as NaN or silently drop it.
# 4. One Lloyd iteration costs $O(nkd)$ (the distance matmul) plus $O(nd)$ for the means. Brute-force k-NN for $m$ queries costs $O(mnd)$ time and
#    $O(mn)$ memory (chunk the queries). To scale: approximate nearest neighbors (IVF/product quantization as in FAISS, graph indices like HNSW),
#    or KD/ball trees in low dimension.
# 5. Small $k$: low bias, high variance (k=1 has training error 0). Large $k$: smoother and more biased, tending to the majority class.
#    Cover & Hart (1967): as $n\to\infty$ the 1-NN error is at most twice the Bayes error. In high dimensions distances concentrate (nearest and
#    farthest become similar), so raw Euclidean k-NN degrades; use learned embeddings or cosine distance. Both methods use Euclidean distance,
#    so a feature with a large scale dominates: standardize first.
# 6. Elbow in inertia vs $k$ (inertia always decreases with $k$), silhouette score, gap statistic, BIC under a GMM view, or the downstream task metric.
# </details>

# %%
s.finish(lookups=0, score=None, notes="")
