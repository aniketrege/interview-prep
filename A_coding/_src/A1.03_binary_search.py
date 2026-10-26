# %% [markdown]
# # A1.3 · Binary search: bounds with an invariant, rotated arrays, search on the answer
#
# **ID:** A1.3 · **Target:** 60 min · **Plan week:** 2 (pattern notebook, Wed); second pass in week 8 · **Part:** A
#
# **The drill.** This is a 60-minute study notebook. Binary search is short to write and easy to get off by one.
# The cure is to write **one** template with an explicit loop invariant and reduce everything else to it. Fill in
# `lower_bound`, `upper_bound` and the general `first_true(lo, hi, pred)`, then do five warm-ups (10-15 min each):
# rotated arrays, first/last position, integer square root and a "minimum feasible rate" search on the answer space.
# The tests compare against Python's `bisect`, `math.isqrt` and brute force. Every call runs under a 2-second
# time limit, because the classic binary-search bug is an infinite loop (`lo = mid`).
#
# **Rules.** Python standard library only. Do not call `bisect` inside your functions; the tests use it as the
# reference. Fill in every `raise NotImplementedError`.
#
# ## Sources
# - Python docs, `bisect` (the half-open invariant for `bisect_left`: `all(e < x for e in a[lo:ip])` and
#   `all(e >= x for e in a[ip:hi])`; the `key=` parameter needs Python ≥ 3.10):
#   https://docs.python.org/3/library/bisect.html ; `math.isqrt`: https://docs.python.org/3/library/math.html#math.isqrt
# - J. Bloch, *Extra, Extra - Read All About It: Nearly All Binary Searches and Mergesorts are Broken*, Google
#   Research blog, 2006 (the `(low + high) / 2` overflow in Java/C, including in the binary search of Bentley's
#   *Programming Pearls* Ch. 5): https://research.google/blog/extra-extra-read-all-about-it-nearly-all-binary-searches-and-mergesorts-are-broken/
# - T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, 4th ed., MIT Press, 2022
#   (ISBN 9780262046305): Ch. 2 "Getting Started" (loop invariants: initialization, maintenance, termination).
#   https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
# - Tech Interview Handbook, *Sorting and searching* cheatsheet, https://www.techinterviewhandbook.org/algorithms/sorting-searching/
#   (yangshun/tech-interview-handbook @ e1d28e8, MIT).
# - NeetCode 150 problem data (practice list): https://neetcode.io/practice ,
#   https://github.com/neetcode-gh/leetcode @ d5ebb3e `.problemSiteData.json`.

# %%
import bisect
import contextlib
import copy
import math
import random
import signal

from prep import Session, check

s = Session("A1.3", target_min=60)
random.seed(0)


@contextlib.contextmanager
def time_limit(sec=2):
    """Raise TimeoutError if the block runs longer than sec (catches `lo = mid` infinite loops). Unix only."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f"took more than {sec}s: infinite loop? (check that lo/hi always move)")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, sec)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def run_tests(fn, cases, name=None):
    """cases: list of (args_tuple, expected). Stops at the first failing case and prints it."""
    name = name or fn.__name__
    with time_limit():
        for i, (args, want) in enumerate(cases):
            got = fn(*copy.deepcopy(args))
            assert got == want, f"{name} case {i}: args={args!r}\n  got  {got!r}\n  want {want!r}"
    print(f"ok  {name}: {len(cases)} cases")


def stress(fn, brute, gen, n=300, seed=0, name=None):
    """Compare fn against a trusted brute force on n random small inputs; gen(rng) returns an args tuple."""
    name = name or fn.__name__
    rng = random.Random(seed)
    with time_limit():
        for t in range(n):
            args = gen(rng)
            want = brute(*copy.deepcopy(args))
            got = fn(*copy.deepcopy(args))
            assert got == want, f"{name} vs brute force, trial {t}: args={args!r}\n  got  {got!r}\n  want {want!r}"
    print(f"ok  {name} vs brute force: {n} random cases")


def sorted_list(rng, max_len=9, lo=0, hi=6):
    return sorted(rng.randint(lo, hi) for _ in range(rng.randint(0, max_len)))

# %% [markdown]
# ## 1. One template with an invariant: the half-open lower bound
# **Intuition.** Keep a **half-open** range `[lo, hi)` of "undecided" indices with the invariant
#
# $$a[i] < x \;\; \forall i < lo, \qquad a[i] \ge x \;\; \forall i \ge hi.$$
#
# It starts true with `lo = 0, hi = len(a)`: nothing is decided yet. `mid = (lo + hi) // 2` is always in `[lo, hi)`.
# If `a[mid] < x`, set `lo = mid + 1`, otherwise set `hi = mid`. The range strictly shrinks every step, so the loop
# ends when `lo == hi`. At that point the invariant says `lo` is the first index with `a[i] >= x`: the insertion
# point, exactly `bisect_left`. Every other variant changes **only the comparison**.
#
# **Signals:** sorted input; "first/last position", "insertion point", "count of values in a range", "closest
# value"; $O(\log n)$ is required; the input is huge or given only as a function.

# %%
def lower_bound(a, x):
    """First index i with a[i] >= x (len(a) if none). Same as bisect.bisect_left."""
    ### BEGIN SOLUTION
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2           # Python ints do not overflow; in Java/C++ write lo + (hi - lo) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
    ### END SOLUTION


def upper_bound(a, x):
    """First index i with a[i] > x (len(a) if none). Same as bisect.bisect_right."""
    ### BEGIN SOLUTION
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo
    ### END SOLUTION


run_tests(lower_bound, [(([1, 2, 2, 2, 5], 2), 1), (([1, 2, 2, 2, 5], 6), 5), (([1, 3], 0), 0), (([], 1), 0)])
run_tests(upper_bound, [(([1, 2, 2, 2, 5], 2), 4), (([1, 2, 2, 2, 5], 6), 5), (([1, 3], 0), 0), (([], 1), 0)])
gen_bound = lambda r: (sorted_list(r), r.randint(-1, 7))
stress(lower_bound, bisect.bisect_left, gen_bound, n=1000)
stress(upper_bound, bisect.bisect_right, gen_bound, n=1000)

# %% [markdown]
# ## 2. Binary search on a monotone predicate (and on the answer space)
# **Intuition.** Both bounds are special cases of one question: on `[lo, hi)`, `pred` is False...False True...True.
# Where is the first True? `lower_bound(a, x)` is `first_true(0, len(a), lambda i: a[i] >= x)`. The range need
# not be indices. It can be **candidate answers** (a speed, a capacity, a day, a threshold) whenever feasibility is
# monotone: if rate $r$ works, every rate $> r$ works too.
#
# **Signals:** "minimum X such that ...", "maximum X such that ...", "smallest capacity/speed/time/largest
# minimum distance", the answer is an integer in a known range, and checking one candidate is easy ($O(n)$).
# Cost: $O(\log(\text{range}) \cdot \text{cost(pred)})$.
#
# **Template: first True of a monotone predicate.**

# %%
def first_true(lo, hi, pred):
    """Smallest t in [lo, hi) with pred(t) True, assuming pred is monotone (F..F T..T). Returns hi if none."""
    ### BEGIN SOLUTION
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
    ### END SOLUTION


def first_true_brute(lo, hi, pred):
    return next((t for t in range(lo, hi) if pred(t)), hi)


def _gen_ft(r):
    lo = r.randint(-5, 5)
    hi = lo + r.randint(0, 10)
    thr = r.randint(lo - 2, hi + 2)
    return lo, hi, (lambda t, thr=thr: t >= thr)


run_tests(first_true, [((0, 10, lambda t: t * t >= 50), 8), ((0, 5, lambda t: False), 5), ((3, 3, lambda t: True), 3),
                       ((-4, 4, lambda t: True), -4)])
stress(first_true, first_true_brute, _gen_ft, n=1000)

# %% [markdown]
# ### Warm-up 2.1: integer square root (~8 min)
# Return $\lfloor\sqrt{n}\rfloor$ for an integer $n \ge 0$ without floating point. Express it with `first_true`:
# find the first `r` with `r * r > n` and subtract 1. It must be fast for $n \approx 10^{18}$.

# %%
def int_sqrt(n):
    ### BEGIN SOLUTION
    return first_true(0, n + 2, lambda r: r * r > n) - 1
    ### END SOLUTION


run_tests(int_sqrt, [((0,), 0), ((1,), 1), ((15,), 3), ((16,), 4), ((10**18,), 10**9), ((10**18 - 1,), 10**9 - 1)])
stress(int_sqrt, math.isqrt, lambda r: (r.choice([r.randint(0, 200), r.randint(0, 10**30)]),), n=1000)

# %% [markdown]
# ### Warm-up 2.2: slowest sufficient eating rate (classic "Koko"; ~12 min)
# There are piles of bananas, `piles[i] > 0`, and `h` hours, with `h >= len(piles)`. Each hour you pick one pile
# and eat up to `r` bananas from it; leftovers wait for the next hour. Return the smallest integer `r >= 1`
# that finishes everything within `h` hours. *Hint:* hours needed at rate `r` is $\sum_i \lceil p_i / r \rceil$,
# which is non-increasing in `r`. What are the smallest and largest candidate rates?

# %%
def min_eating_rate(piles, h):
    ### BEGIN SOLUTION
    hours = lambda r: sum((p + r - 1) // r for p in piles)     # ceil without floats
    return first_true(1, max(piles) + 1, lambda r: hours(r) <= h)
    ### END SOLUTION


def min_eating_rate_brute(piles, h):
    r = 1
    while sum(math.ceil(p / r) for p in piles) > h:
        r += 1
    return r


def _gen_koko(r):
    piles = [r.randint(1, 20) for _ in range(r.randint(1, 6))]
    return piles, r.randint(len(piles), 3 * len(piles) + 5)


run_tests(min_eating_rate, [(([3, 6, 7, 11], 8), 4), (([30, 11, 23, 4, 20], 5), 30), (([30, 11, 23, 4, 20], 6), 23),
                            (([1], 1), 1), (([10**9], 2), 5 * 10**8)])
stress(min_eating_rate, min_eating_rate_brute, _gen_koko)

# %% [markdown]
# ### Warm-up 2.3: first and last position of a value (~8 min)
# Given a sorted list (duplicates allowed) and `x`, return `(first, last)`, the indices of the first and last
# occurrence of `x`, or `(-1, -1)` if it is absent. Use your two bounds: $O(\log n)$.

# %%
def first_last(a, x):
    ### BEGIN SOLUTION
    i = lower_bound(a, x)
    if i == len(a) or a[i] != x:
        return (-1, -1)
    return (i, upper_bound(a, x) - 1)
    ### END SOLUTION


def first_last_brute(a, x):
    idx = [i for i, v in enumerate(a) if v == x]
    return (idx[0], idx[-1]) if idx else (-1, -1)


run_tests(first_last, [(([5, 7, 7, 8, 8, 10], 8), (3, 4)), (([5, 7, 7, 8, 8, 10], 6), (-1, -1)), (([], 0), (-1, -1)),
                       (([2, 2], 2), (0, 1)), (([1], 3), (-1, -1))])
stress(first_last, first_last_brute, gen_bound, n=1000)

# %% [markdown]
# ## 3. Rotated sorted arrays
# **Intuition.** A sorted array of **distinct** values rotated at an unknown pivot (for example `[4,5,6,0,1,2]`) is
# still "sorted enough". The predicate `a[i] <= a[-1]` is False on the left (large) part and True on the right
# (small) part, so the minimum is `first_true` of it. To search for a target, compare `a[lo]` with `a[mid]`: one of
# the two halves around `mid` is sorted, so check whether the target lies in that half's range.
#
# **Signals:** "rotated", "shifted", "circularly sorted", "find the pivot/minimum", "search in $O(\log n)$" on such data.
#
# ### Warm-up 3.1: minimum of a rotated sorted array (distinct values, non-empty; ~8 min)

# %%
def min_rotated(a):
    ### BEGIN SOLUTION
    return a[first_true(0, len(a), lambda i: a[i] <= a[-1])]
    ### END SOLUTION


def rotated(r, max_len=9):
    vals = sorted(r.sample(range(-10, 20), r.randint(1, max_len)))
    k = r.randrange(len(vals))
    return vals[k:] + vals[:k]


run_tests(min_rotated, [(([4, 5, 6, 0, 1, 2],), 0), (([1, 2, 3],), 1), (([2, 1],), 1), (([7],), 7)])
stress(min_rotated, min, lambda r: (rotated(r),), n=1000)

# %% [markdown]
# ### Warm-up 3.2: search a rotated sorted array (distinct values; ~15 min)
# Return the index of `target`, or -1. *Target:* $O(\log n)$. You may write it as one closed-interval loop
# (`while lo <= hi`) that decides which half is sorted, or as "find the pivot, then `lower_bound` the correct side".
# The two-element case (`lo == mid`) is where most bugs live.

# %%
def search_rotated(a, target):
    ### BEGIN SOLUTION
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == target:
            return mid
        if a[lo] <= a[mid]:                       # left half [lo, mid] is sorted (<= handles lo == mid)
            if a[lo] <= target < a[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                     # right half [mid, hi] is sorted
            if a[mid] < target <= a[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
    ### END SOLUTION


def search_rotated_brute(a, target):
    return a.index(target) if target in a else -1


run_tests(search_rotated, [(([4, 5, 6, 7, 0, 1, 2], 0), 4), (([4, 5, 6, 7, 0, 1, 2], 3), -1), (([1], 0), -1),
                           (([3, 1], 1), 1), (([], 5), -1), (([1, 3], 3), 1)])
stress(search_rotated, search_rotated_brute, lambda r: (rotated(r), r.randint(-11, 21)), n=2000)

# %% [markdown]
# ## Say it out loud (complexity and correctness)
# 1. State the invariant of your `lower_bound`. Why does `hi = len(a)` (not `len(a) - 1`) matter?
# 2. Why does `lo = mid` (instead of `mid + 1`) loop forever? Which `mid` rounding goes with `hi = mid - 1` / `lo = mid`?
# 3. What is Bloch's overflow bug, and why does it not bite in Python?
# 4. Koko: total complexity in terms of $n$ and $M = \max p_i$? Why is the predicate monotone?
# 5. Rotated search with **duplicates** (e.g. `[1, 0, 1, 1, 1]`): what breaks, and what is the worst case?
# 6. "Binary search on the answer" vs "binary search on the index": give one problem of each kind.
# 7. When is a binary search over a *real* range used, and how do you stop it?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. `a[:lo] < x` and `a[hi:] >= x` (half-open). It holds at the start, every step keeps it, and at termination
#    `lo == hi` is the first index with `a[i] >= x` (the CLRS Ch. 2 loop-invariant pattern). With `hi = len(a)`,
#    "x is larger than everything" correctly gives `len(a)`. With `len(a) - 1`, that case returns `len(a) - 1`.
#    This is the same contract as `bisect_left` (Python docs).
# 2. With two elements left, `mid = (lo + hi) // 2 == lo`, so `lo = mid` does not shrink the range and the loop never
#    ends. Rule: **floor** mid pairs with `lo = mid + 1, hi = mid`. If you need `lo = mid`, use the **ceiling**
#    `mid = (lo + hi + 1) // 2` with `hi = mid - 1` (the "last True" form).
# 3. `(low + high) / 2` overflows a 32-bit `int` when `low + high > 2^31 - 1` (arrays of about a billion elements),
#    giving a negative index. Fix: `low + (high - low) / 2` (Bloch 2006). Python integers are arbitrary precision, so
#    the sum cannot overflow, but mention it if the interviewer asks about Java/C++.
# 4. $O(n \log M)$: there are $\log_2 M$ predicate calls, each an $O(n)$ sum. Each term $\lceil p/r \rceil$ is
#    non-increasing in $r$, so the total is too: if $r$ finishes in time, any $r' > r$ does as well.
# 5. With duplicates, `a[lo] == a[mid] == a[hi]` gives no information about which side is sorted. The standard fix
#    shrinks `lo += 1; hi -= 1` in that case, and the worst case degrades to $O(n)$ (e.g. all 1s and a single 0). No
#    $O(\log n)$ comparison algorithm exists for this case, because an adversary can hide the 0 anywhere.
# 6. Index: first/last position, search insert, rotated search, peak element. Answer: Koko's rate, ship capacity
#    within D days, split array largest sum, minimum days to make bouquets, aggressive cows / magnetic balls
#    (maximize the minimum distance).
# 7. Continuous answers, such as a square root to $10^{-6}$ or "minimum time such that ...". Loop a fixed number of
#    times (for example 100 halvings, enough for doubles) or until `hi - lo < eps`, rather than testing floats
#    for equality.
# </details>

# %% [markdown]
# <!-- practice:start -->
# ## Practice list (solve these on LeetCode)
#
# NeetCode 150 problems for this topic, generated from `neetcode-gh/leetcode@d5ebb3e` `.problemSiteData.json` (https://neetcode.io/practice). **B75** = also in Blind 75. Tick them off in your session copy; the full checklist is `A_coding/A1_problem_list.md`. Grind 75 (https://www.techinterviewhandbook.org/grind75/) is an alternative ordering of a similar set.
#
# **Binary Search** · plan week 8 · 7 problems
#
# - [ ] [Binary Search](https://leetcode.com/problems/binary-search/) · Easy
# - [ ] [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/) · Medium
# - [ ] [Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/) · Medium
# - [ ] [Find Minimum In Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) · Medium · **B75**
# - [ ] [Search In Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) · Medium · **B75**
# - [ ] [Time Based Key Value Store](https://leetcode.com/problems/time-based-key-value-store/) · Medium
# - [ ] [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) · Hard
#
# **Extras (not in NeetCode 150; close to the warm-ups above)**
#
# - [ ] [Search Insert Position](https://leetcode.com/problems/search-insert-position/) · Easy
# - [ ] [Find First And Last Position of Element In Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/) · Medium
# - [ ] [Sqrt(x)](https://leetcode.com/problems/sqrtx/) · Easy
# - [ ] [Capacity to Ship Packages](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/) · Medium
# <!-- practice:end -->

# %%
s.finish(lookups=0, score=None, notes="")
