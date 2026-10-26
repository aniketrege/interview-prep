# %% [markdown]
# # A1.9 · Dynamic programming: top-down vs bottom-up, 1-D and 2-D
#
# **ID:** A1.9 · **Target:** 60 min · **Plan week:** 6 (pattern notebook, Mon); second pass in weeks 14 (1-D DP, greedy) and 15 (2-D DP) · **Part:** A
#
# **The drill.** This is a 60-minute study notebook. DP is "backtracking that remembers": define a **state**, write the
# **recurrence**, fix the **base cases** and an **evaluation order**, then read off the answer. Fill in one problem both
# ways (memoized recursion and a table), then the 1-D classics (house robber, coin change, longest increasing
# subsequence in $O(n^2)$ and $O(n\log n)$) and the 2-D classics (grid paths with obstacles, longest common
# subsequence, edit distance, 0/1 knapsack with a rolling array). Each fill-in takes 4-10 minutes. The brute forces
# enumerate every choice exponentially, on small inputs.
#
# **Rules.** Python standard library only (`functools.cache` and `bisect` are fine). Fill in every
# `raise NotImplementedError`.
#
# **State-definition checklist (say it before coding).**
# 1. **State:** what does `dp[i]` / `dp[i][j]` mean, *in words*? (For example, "the best answer using only the first i
#    items" or "the edit distance between `a[:i]` and `b[:j]`".) Prefix-based states with an extra row and column for
#    the empty prefix remove most off-by-ones.
# 2. **Transition:** what is the *last decision*? Take or skip item i, which coin comes last, match or edit the last
#    characters. Write the recurrence over those options.
# 3. **Base cases:** the empty prefix, amount 0, the first row and column. Use an "impossible" value (`inf` / -1) where
#    nothing is feasible.
# 4. **Order:** every state must be computed after the states it reads (increasing i, or decreasing w for 0/1 knapsack).
# 5. **Answer:** which cell holds it (`dp[n]`, `max(dp)`, `dp[m][n]`)? Can you **roll** the table to $O(\text{row})$ space?
#
# ## Sources
# - T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, 4th ed., MIT Press, 2022
#   (ISBN 9780262046305), Ch. 14 "Dynamic Programming" (optimal substructure, overlapping subproblems, memoization vs
#   bottom-up, LCS) and Ch. 15 "Greedy Algorithms". https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
# - R. A. Wagner, M. J. Fischer, *The String-to-String Correction Problem*, Journal of the ACM 21(1), 1974,
#   DOI 10.1145/321796.321811 (the edit-distance DP).
# - M. L. Fredman, *On computing the length of longest increasing subsequences*, Discrete Mathematics 11(1), 1975,
#   DOI 10.1016/0012-365X(75)90103-X ($O(n\log n)$ LIS).
# - Python docs, `functools.cache` (an unbounded `lru_cache`): https://docs.python.org/3/library/functools.html#functools.cache ;
#   `bisect`: https://docs.python.org/3/library/bisect.html ; recursion limit: https://docs.python.org/3/library/sys.html#sys.getrecursionlimit
# - Tech Interview Handbook, *Dynamic programming* cheatsheet, https://www.techinterviewhandbook.org/algorithms/dynamic-programming/
#   (yangshun/tech-interview-handbook @ e1d28e8, MIT).
# - NeetCode 150 problem data (practice list): https://neetcode.io/practice ,
#   https://github.com/neetcode-gh/leetcode @ d5ebb3e `.problemSiteData.json`.

# %%
import bisect
import copy
import itertools
import random
from collections import deque
from functools import cache

from prep import Session, check

s = Session("A1.9", target_min=60)
random.seed(0)


def run_tests(fn, cases, name=None):
    """cases: list of (args_tuple, expected). Stops at the first failing case and prints it."""
    name = name or fn.__name__
    for i, (args, want) in enumerate(cases):
        got = fn(*copy.deepcopy(args))
        assert got == want, f"{name} case {i}: args={args!r}\n  got  {got!r}\n  want {want!r}"
    print(f"ok  {name}: {len(cases)} cases")


def stress(fn, brute, gen, n=300, seed=0, name=None):
    """Compare fn against a trusted brute force on n random small inputs; gen(rng) returns an args tuple."""
    name = name or fn.__name__
    rng = random.Random(seed)
    for t in range(n):
        args = gen(rng)
        want = brute(*copy.deepcopy(args))
        got = fn(*copy.deepcopy(args))
        assert got == want, f"{name} vs brute force, trial {t}: args={args!r}\n  got  {got!r}\n  want {want!r}"
    print(f"ok  {name} vs brute force: {n} random cases")


def rand_list(rng, lo=0, hi=9, max_len=8, min_len=0):
    return [rng.randint(lo, hi) for _ in range(rng.randint(min_len, max_len))]

# %% [markdown]
# ## 1. Top-down (memoized) vs bottom-up (tabulated)
# **Intuition.** If a plain recursion solves the same subproblem many times (overlapping subproblems) and an optimal
# answer is built from optimal sub-answers (optimal substructure), caching turns exponential time into
# (number of states) × (work per state) (CLRS Ch. 14). **Top-down:** write the recursion and add `@cache`. It only
# visits reachable states, but it uses stack depth equal to the dependency chain. **Bottom-up:** fill a table in
# dependency order. There is no recursion limit, and the table can often be rolled to a few variables.
#
# **Signals:** "number of ways", "minimum/maximum cost", "is it possible", "longest/shortest subsequence"; choices at
# each step whose consequences depend only on a small state (index, remaining budget, last choice).
#
# **Template (one problem, both ways): ways to climb n stairs taking 1 or 2 steps.** `ways(0) = 1` (the empty climb).

# %%
def climb_memo(n):
    ### BEGIN SOLUTION
    @cache
    def f(i):
        return 1 if i <= 1 else f(i - 1) + f(i - 2)
    return f(n)
    ### END SOLUTION


def climb_tab(n):
    ### BEGIN SOLUTION
    a, b = 1, 1                       # ways(0), ways(1); rolled table
    for _ in range(n - 1):
        a, b = b, a + b
    return b if n >= 1 else 1
    ### END SOLUTION


def climb_brute(n):
    return 1 if n <= 1 else climb_brute(n - 1) + climb_brute(n - 2)


for f in (climb_memo, climb_tab):
    run_tests(f, [((0,), 1), ((1,), 1), ((2,), 2), ((3,), 3), ((5,), 8)])
    stress(f, climb_brute, lambda r: (r.randint(0, 18),), n=50)
check("climb_tab(300) == climb_memo(300) (big ints, fast)", climb_tab(300) == climb_memo(300))

# %% [markdown]
# ## 2. 1-D DP
# ### Warm-up 2.1: rob non-adjacent houses (classic; ~6 min)
# Each house holds a non-negative amount. Pick a set of houses, no two adjacent, maximizing the total. *State:*
# `best[i]` is the best total using houses `[:i]`. *Last decision:* rob house `i-1` or not.

# %%
def house_robber(nums):
    ### BEGIN SOLUTION
    prev2, prev1 = 0, 0               # best[i-2], best[i-1]
    for x in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + x)
    return prev1
    ### END SOLUTION


def house_robber_brute(nums):
    n, best = len(nums), 0
    for mask in range(1 << n):
        if mask & (mask >> 1) == 0:                       # no two adjacent bits
            best = max(best, sum(nums[i] for i in range(n) if mask >> i & 1))
    return best


run_tests(house_robber, [(([1, 2, 3, 1],), 4), (([2, 7, 9, 3, 1],), 12), (([],), 0), (([5],), 5), (([2, 1, 1, 2],), 4)])
stress(house_robber, house_robber_brute, lambda r: (rand_list(r, 0, 9, 10),))

# %% [markdown]
# ### Warm-up 2.2: fewest coins (classic "coin change"; ~8 min)
# Given coin denominations (unlimited supply of each) and an `amount`, return the fewest coins that sum to it, or -1
# if it is impossible. *State:* `dp[a]` is the fewest coins for amount `a`. *Last decision:* which coin is last.

# %%
def coin_change(coins, amount):
    ### BEGIN SOLUTION
    INF = float("inf")
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] != INF else -1
    ### END SOLUTION


def coin_change_brute(coins, amount):         # BFS over amounts: level = number of coins used
    seen, q = {0}, deque([(0, 0)])
    while q:
        a, k = q.popleft()
        if a == amount:
            return k
        for c in coins:
            if a + c <= amount and a + c not in seen:
                seen.add(a + c)
                q.append((a + c, k + 1))
    return -1


run_tests(coin_change, [(([1, 2, 5], 11), 3), (([2], 3), -1), (([1], 0), 0), (([1, 3, 4], 6), 2),   # greedy gives 3
                        (([7], 14), 2)])
stress(coin_change, coin_change_brute, lambda r: (r.sample(range(1, 10), r.randint(1, 3)), r.randint(0, 30)))

# %% [markdown]
# ### Warm-up 2.3: longest strictly increasing subsequence, twice (~12 min)
# (a) $O(n^2)$: `dp[i]` is the length of the LIS **ending at** `i` (the answer is `max(dp)`, not `dp[-1]`).
# (b) $O(n\log n)$ (Fredman 1975, "patience sorting"): `tails[L]` is the smallest possible last value of an increasing
# subsequence of length `L+1`. For each `x`, replace the first tail `>= x` (`bisect_left`, because the sequence is
# strict) or append. `len(tails)` is the answer. `tails` is *not* itself an LIS.

# %%
def lis_quadratic(nums):
    ### BEGIN SOLUTION
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp, default=0)
    ### END SOLUTION


def lis_nlogn(nums):
    ### BEGIN SOLUTION
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)       # strict increase -> left
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)
    ### END SOLUTION


def lis_brute(nums):
    return max((k for k in range(len(nums) + 1) for c in itertools.combinations(nums, k)
                if all(a < b for a, b in zip(c, c[1:]))), default=0)


for f in (lis_quadratic, lis_nlogn):
    run_tests(f, [(([10, 9, 2, 5, 3, 7, 101, 18],), 4), (([0, 1, 0, 3, 2, 3],), 4), (([7, 7, 7],), 1), (([],), 0),
                  (([3, 1, 2],), 2)])
    stress(f, lis_brute, lambda r: (rand_list(r, 0, 6, 9),))

# %% [markdown]
# ## 3. 2-D DP
# ### Warm-up 3.1: grid paths around obstacles (~6 min)
# `grid` is a list of lists of 0 (free) and 1 (blocked). Count the paths from the top-left to the bottom-right that
# move only right or down. A blocked start or end gives 0. One row of `dp` is enough.

# %%
def grid_paths(grid):
    ### BEGIN SOLUTION
    w = len(grid[0])
    dp = [0] * w
    dp[0] = 1
    for row in grid:
        for j in range(w):
            if row[j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j - 1]            # from above (old dp[j]) + from the left (new dp[j-1])
    return dp[-1]
    ### END SOLUTION


def grid_paths_brute(grid):
    h, w = len(grid), len(grid[0])

    def go(r, c):
        if r >= h or c >= w or grid[r][c] == 1:
            return 0
        if (r, c) == (h - 1, w - 1):
            return 1
        return go(r + 1, c) + go(r, c + 1)

    return go(0, 0)


run_tests(grid_paths, [(([[0, 0, 0], [0, 1, 0], [0, 0, 0]],), 2), (([[0, 1], [0, 0]],), 1), (([[1]],), 0), (([[0]],), 1),
                       (([[0, 0], [0, 1]],), 0)])
stress(grid_paths, grid_paths_brute,
       lambda r: ([[int(r.random() < 0.25) for _ in range(w)] for w in [r.randint(1, 5)] for _ in range(r.randint(1, 5))],))

# %% [markdown]
# ### Warm-up 3.2: longest common subsequence (~8 min)
# `dp[i][j]` is the LCS length of `a[:i]` and `b[:j]`. If the last characters match, the value is
# `dp[i-1][j-1] + 1`. Otherwise it is `max(dp[i-1][j], dp[i][j-1])` (CLRS Ch. 14).

# %%
def lcs(a, b):
    ### BEGIN SOLUTION
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
    ### END SOLUTION


def lcs_brute(a, b):
    def is_subseq(sub, t):
        it = iter(t)
        return all(ch in it for ch in sub)
    return max((k for k in range(len(a) + 1) for c in itertools.combinations(a, k) if is_subseq(c, b)), default=0)


rand_str = lambda r, n=6: "".join(r.choice("abc") for _ in range(r.randint(0, n)))
run_tests(lcs, [(("abcde", "ace"), 3), (("abc", "def"), 0), (("", "abc"), 0), (("aaaa", "aa"), 2)])
stress(lcs, lcs_brute, lambda r: (rand_str(r, 7), rand_str(r, 7)))

# %% [markdown]
# ### Warm-up 3.3: edit distance (~10 min)
# The minimum number of single-character insertions, deletions and substitutions that turn `a` into `b`
# (Wagner and Fischer 1974). `dp[i][j]` is the distance between `a[:i]` and `b[:j]`, with `dp[i][0] = i` and
# `dp[0][j] = j`.

# %%
def edit_distance(a, b):
    ### BEGIN SOLUTION
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],        # delete a[i-1]
                                   dp[i][j - 1],        # insert b[j-1]
                                   dp[i - 1][j - 1])    # substitute
    return dp[m][n]
    ### END SOLUTION


def edit_brute(a, b):                         # plain exponential recursion, no memo
    if not a or not b:
        return len(a) + len(b)
    if a[0] == b[0]:
        return edit_brute(a[1:], b[1:])
    return 1 + min(edit_brute(a[1:], b), edit_brute(a, b[1:]), edit_brute(a[1:], b[1:]))


run_tests(edit_distance, [(("horse", "ros"), 3), (("intention", "execution"), 5), (("", ""), 0), (("", "ab"), 2),
                          (("abc", "abc"), 0), (("ab", "ba"), 2)])
stress(edit_distance, edit_brute, lambda r: (rand_str(r, 5), rand_str(r, 5)), n=200)

# %% [markdown]
# ### Warm-up 3.4: 0/1 knapsack with a rolling array (~10 min)
# Items have positive integer weights and non-negative values. Each item can be taken **at most once**. Return the
# best total value with total weight `<= W`. The 2-D state is `dp[i][w]` (best value using the first `i` items with
# capacity `w`). Roll it into 1-D `dp[w]` and loop `w` **downwards**, so that `dp[w - wt]` still refers to the
# previous item row. Looping upwards would let one item be used many times: that is the unbounded knapsack.

# %%
def knapsack_01(weights, values, W):
    ### BEGIN SOLUTION
    dp = [0] * (W + 1)
    for wt, val in zip(weights, values):
        for w in range(W, wt - 1, -1):          # downwards: each item at most once
            dp[w] = max(dp[w], dp[w - wt] + val)
    return dp[W]
    ### END SOLUTION


def knapsack_brute(weights, values, W):
    n = len(weights)
    return max(sum(values[i] for i in range(n) if m >> i & 1) for m in range(1 << n)
               if sum(weights[i] for i in range(n) if m >> i & 1) <= W)


def _gen_knap(r):
    n = r.randint(0, 7)
    return [r.randint(1, 6) for _ in range(n)], [r.randint(0, 9) for _ in range(n)], r.randint(0, 15)


run_tests(knapsack_01, [(([1, 3, 4, 5], [1, 4, 5, 7], 7), 9), (([5], [10], 4), 0), (([], [], 10), 0),
                        (([2], [3], 4), 3)])                     # an unbounded (upward) loop returns 6 here
stress(knapsack_01, knapsack_brute, _gen_knap)

# %% [markdown]
# ## Say it out loud (complexity and correctness)
# 1. For each problem above, give the state, the number of states, the work per state, the total time and the space after rolling.
# 2. Top-down vs bottom-up: when do you prefer each? What is the Python-specific risk of top-down?
# 3. Why does greedy fail for coin change with `[1, 3, 4]` and amount 6? When is greedy optimal?
# 4. LIS in $O(n\log n)$: what invariant does `tails` maintain? Why `bisect_left` for strict and `bisect_right` for non-decreasing?
# 5. Edit distance: how do you reconstruct the actual edits? How do you reduce the space to $O(\min(m, n))$?
# 6. 0/1 knapsack is $O(nW)$. Is that polynomial? What is it called?
# 7. How do you recognise a "DP on intervals" or "DP on subsets (bitmask)" problem?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Climbing: $n$ states, $O(1)$ each, $O(n)$ time, $O(1)$ space. Robber: the same. Coin change: `amount` states ×
#    `len(coins)` work, $O(A\,k)$ time, $O(A)$ space. LIS: $n$ states × $O(n)$ work $= O(n^2)$, or $O(n\log n)$ with
#    `tails`. Grid: $RC$ states, $O(1)$ each, $O(C)$ space rolled. LCS and edit distance: $mn$ states, $O(1)$ each, so
#    $O(mn)$ time, and $O(\min(m,n))$ space rolled (two rows). Knapsack: $nW$ states, $O(nW)$ time, $O(W)$ space.
# 2. Top-down is quicker to write from the recursion, and it touches only reachable states (sparse spaces such as
#    `(index, remaining)` with large ranges). Bottom-up has no recursion overhead and no depth limit, and it makes
#    rolling easy. In Python, a memoized recursion of depth more than about 1000 raises `RecursionError` at the default
#    limit (`sys.getrecursionlimit()`), so `climb_memo(5000)` fails while `climb_tab(5000)` is instant.
# 3. Greedy takes 4 + 1 + 1 (3 coins), but 3 + 3 uses 2. Greedy is optimal for "canonical" coin systems such as
#    1/5/10/25, and in general when the problem has the greedy-choice property (exchange argument, CLRS Ch. 15). For
#    arbitrary denominations, use DP.
# 4. `tails` is strictly increasing, and `tails[L]` is the minimum possible tail of an increasing subsequence of length
#    $L+1$ among the elements seen so far. Replacing the first tail $\ge x$ keeps both properties (Fredman 1975). For a
#    non-decreasing subsequence, equal values may extend it, so replace the first tail $> x$ with `bisect_right`.
# 5. Keep the full table and walk back from `(m, n)`. At each cell, follow the move that produced it (diagonal with
#    equal characters, diagonal + 1 for a substitution, up for a delete, left for an insert). For distance only, keep
#    two rows of length $\min(m,n)+1$. Hirschberg's divide-and-conquer recovers the alignment in linear space.
# 6. It is *pseudo-polynomial*: polynomial in the **value** of $W$, but exponential in its bit length $\log W$.
#    Knapsack is NP-hard in general.
# 7. Intervals: the answer for `s[i..j]` combines answers for sub-intervals (burst balloons, palindrome partitioning,
#    matrix-chain order, CLRS Ch. 14). Iterate by interval length, $O(n^3)$ typically. Bitmask: $n \le 20$ and the state
#    is "which items are used so far" (TSP, assignment), $O(2^n n)$ states. Bit tricks are in A1.10.
# </details>

# %% [markdown]
# <!-- practice:start -->
# ## Practice list (solve these on LeetCode)
#
# NeetCode 150 problems for this topic, generated from `neetcode-gh/leetcode@d5ebb3e` `.problemSiteData.json` (https://neetcode.io/practice). **B75** = also in Blind 75. Tick them off in your session copy; the full checklist is `A_coding/A1_problem_list.md`. Grind 75 (https://www.techinterviewhandbook.org/grind75/) is an alternative ordering of a similar set.
#
# **1-D Dynamic Programming** · plan week 14 · 12 problems
#
# - [ ] [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) · Easy · **B75**
# - [ ] [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/) · Easy
# - [ ] [House Robber](https://leetcode.com/problems/house-robber/) · Medium · **B75**
# - [ ] [House Robber II](https://leetcode.com/problems/house-robber-ii/) · Medium · **B75**
# - [ ] [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/) · Medium · **B75**
# - [ ] [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/) · Medium · **B75**
# - [ ] [Decode Ways](https://leetcode.com/problems/decode-ways/) · Medium · **B75**
# - [ ] [Coin Change](https://leetcode.com/problems/coin-change/) · Medium · **B75**
# - [ ] [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/) · Medium · **B75**
# - [ ] [Word Break](https://leetcode.com/problems/word-break/) · Medium · **B75**
# - [ ] [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/) · Medium · **B75**
# - [ ] [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/) · Medium
#
# **Greedy** · plan week 14 · 8 problems
#
# - [ ] [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) · Medium · **B75**
# - [ ] [Jump Game](https://leetcode.com/problems/jump-game/) · Medium · **B75**
# - [ ] [Jump Game II](https://leetcode.com/problems/jump-game-ii/) · Medium
# - [ ] [Gas Station](https://leetcode.com/problems/gas-station/) · Medium
# - [ ] [Hand of Straights](https://leetcode.com/problems/hand-of-straights/) · Medium
# - [ ] [Merge Triplets to Form Target Triplet](https://leetcode.com/problems/merge-triplets-to-form-target-triplet/) · Medium
# - [ ] [Partition Labels](https://leetcode.com/problems/partition-labels/) · Medium
# - [ ] [Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/) · Medium
#
# **2-D Dynamic Programming** · plan week 15 · 11 problems
#
# - [ ] [Unique Paths](https://leetcode.com/problems/unique-paths/) · Medium · **B75**
# - [ ] [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/) · Medium · **B75**
# - [ ] [Best Time to Buy And Sell Stock With Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) · Medium
# - [ ] [Coin Change II](https://leetcode.com/problems/coin-change-ii/) · Medium
# - [ ] [Target Sum](https://leetcode.com/problems/target-sum/) · Medium
# - [ ] [Interleaving String](https://leetcode.com/problems/interleaving-string/) · Medium
# - [ ] [Longest Increasing Path In a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/) · Hard
# - [ ] [Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/) · Hard
# - [ ] [Edit Distance](https://leetcode.com/problems/edit-distance/) · Medium
# - [ ] [Burst Balloons](https://leetcode.com/problems/burst-balloons/) · Hard
# - [ ] [Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/) · Hard
#
# **Extras (not in NeetCode 150; close to the warm-ups above)**
#
# - [ ] [Unique Paths II](https://leetcode.com/problems/unique-paths-ii/) · Medium
# <!-- practice:end -->

# %%
s.finish(lookups=0, score=None, notes="")
