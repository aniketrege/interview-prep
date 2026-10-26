# %% [markdown]
# # A1.1 · Arrays & hashing, two pointers, sliding window, prefix sums
#
# **ID:** A1.1 · **Target:** 60 min · **Plan week:** 1 (pattern notebook, Tue-Wed); second pass on its problems in week 7 (hashing, two pointers) and 8 (sliding window) · **Part:** A
#
# **The drill.** This is a 60-minute study notebook. For each pattern, read the intuition and the "signals", fill in
# the canonical template, then do the warm-ups (each one 10-20 min, with the A1.0 steps condensed: clarify,
# brute force, optimize, code, test). Every function has hand-written cases (including edge cases) and a
# randomized **stress test** against a brute force that is given and trusted. Do not edit the brute forces or the tests.
# Afterwards, answer the complexity questions out loud and pick problems from the practice list at the bottom.
#
# **Rules.** Python standard library only (`collections`, `itertools` are fine). Fill in every
# `raise NotImplementedError`. Each test cell prints `ok ...` or shows the smallest failing input.
#
# ## Sources
# - T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, 4th ed., MIT Press,
#   2022 (ISBN 9780262046305), Ch. 11 "Hash Tables" (expected $O(1)$ operations under uniform hashing).
#   https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
# - Python docs, `collections.Counter` / `defaultdict`: https://docs.python.org/3/library/collections.html ;
#   average and worst-case costs of `dict`/`set`/`list` operations: https://wiki.python.org/moin/TimeComplexity
# - Tech Interview Handbook, *Array* and *Hash table* cheatsheets (techniques: sliding window, two pointers,
#   precomputation; corner cases), https://www.techinterviewhandbook.org/algorithms/array/ ,
#   https://www.techinterviewhandbook.org/algorithms/hash-table/ (repo yangshun/tech-interview-handbook @ e1d28e8,
#   MIT; paraphrased).
# - A. Gajentaan, M. H. Overmars, *On a class of O(n²) problems in computational geometry*, Computational
#   Geometry 5(3), 1995, DOI 10.1016/0925-7721(95)00022-2 (3SUM as a hardness baseline).
# - A. Grønlund, S. Pettie, *Threesomes, Degenerates, and Love Triangles*, FOCS 2014, arXiv:1404.0799
#   (3SUM is solvable in slightly sub-quadratic time, by log factors only).
# - NeetCode 150 problem data (practice list): https://neetcode.io/practice ,
#   https://github.com/neetcode-gh/leetcode @ d5ebb3e `.problemSiteData.json`.

# %%
import copy
import itertools
import random
from collections import Counter, defaultdict

from prep import Session, check

s = Session("A1.1", target_min=60)
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


def rand_list(rng, lo=-4, hi=4, max_len=8, min_len=0):
    return [rng.randint(lo, hi) for _ in range(rng.randint(min_len, max_len))]

# %% [markdown]
# ## 1. Hashing: trade memory for $O(1)$ lookups
# **Intuition.** A brute force often asks "have I seen a matching value before?" with an inner loop. Keep a
# `set`/`dict`/`Counter` of what you have seen and the inner loop becomes an expected $O(1)$ lookup (CLRS Ch. 11).
#
# **Signals:** "pair/complement", "duplicate", "count occurrences/frequencies", "group items that are
# equivalent" (canonical key), "first unique", $O(n)$ required on unsorted input.
#
# **Template: complement lookup in one pass.** For each `x`, query the complement in what you have seen
# **so far**, then insert `x`. The order matters: querying first means an element never pairs with itself.

# %%
def count_pairs_with_sum(nums, target):
    """Number of index pairs i < j with nums[i] + nums[j] == target. One pass, O(n) expected time."""
    ### BEGIN SOLUTION
    seen = Counter()
    count = 0
    for x in nums:
        count += seen[target - x]   # pairs (earlier index, current index)
        seen[x] += 1
    return count
    ### END SOLUTION


def count_pairs_brute(nums, target):
    return sum(1 for a, b in itertools.combinations(nums, 2) if a + b == target)


run_tests(count_pairs_with_sum, [
    (([1, 5, 3, 3, 7], 6), 2),     # (1,5), (3,3)
    (([3], 6), 0),                 # an element must not pair with itself
    (([], 0), 0),
    (([2, 2, 2, 2], 4), 6),        # duplicates: C(4,2)
    (([-1, 1, 0, 0], 0), 2),
])
stress(count_pairs_with_sum, count_pairs_brute, lambda r: (rand_list(r), r.randint(-4, 4)))

# %% [markdown]
# ### Warm-up 1.1: group words by anagram (classic; ~10 min)
# You get a list of lowercase words. Put words that are rearrangements of each other into the same group.
# Return the groups in a **canonical form**: each group sorted, and the list of groups sorted. The test then
# does not depend on the order you produce them in. *Think:* what is a good hash key for "same multiset of letters"?

# %%
def group_anagrams(words):
    ### BEGIN SOLUTION
    groups = defaultdict(list)
    for w in words:
        groups[tuple(sorted(w))].append(w)   # or a 26-length count tuple: O(L) instead of O(L log L)
    return sorted(sorted(g) for g in groups.values())
    ### END SOLUTION


def group_anagrams_brute(words):
    groups = []
    for w in words:
        for g in groups:
            if Counter(g[0]) == Counter(w):
                g.append(w)
                break
        else:
            groups.append([w])
    return sorted(sorted(g) for g in groups)


run_tests(group_anagrams, [
    ((["eat", "tea", "tan", "ate", "nat", "bat"],), [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]),
    (([],), []),
    (([""],), [[""]]),
    ((["", ""],), [["", ""]]),
    ((["aab", "abb", "bab", "aba"],), [["aab", "aba"], ["abb", "bab"]]),   # same letter *set*, different counts
])
stress(group_anagrams, group_anagrams_brute,
       lambda r: (["".join(r.choice("ab") for _ in range(r.randint(0, 3))) for _ in range(r.randint(0, 6))],))

# %% [markdown]
# ## 2. Two pointers on sorted input
# **Intuition.** On a sorted array with `i` at the left and `j` at the right, if `a[i] + a[j]` is too small then
# no pair using `a[i]` can work, because every other partner is $\le a[j]$. So drop `i`. If the sum is too big, drop `j`. Each
# step discards one candidate, so the scan takes $O(n)$ after an $O(n\log n)$ sort.
#
# **Signals:** sorted input (or sorting does not hurt), "pair/triple with sum", "closest sum", "remove
# duplicates in place", palindromes, "container" problems, merging two sorted sequences.
#
# **Template: converging pointers.**

# %%
def pair_sum_sorted(a, target):
    """a is sorted ascending. True iff some i < j has a[i] + a[j] == target. O(n) time, O(1) space."""
    ### BEGIN SOLUTION
    i, j = 0, len(a) - 1
    while i < j:
        s_ = a[i] + a[j]
        if s_ == target:
            return True
        if s_ < target:
            i += 1
        else:
            j -= 1
    return False
    ### END SOLUTION


def pair_sum_brute(a, target):
    return count_pairs_brute(a, target) > 0


run_tests(pair_sum_sorted, [
    (([1, 2, 4, 7, 11], 9), True),
    (([1, 2, 4, 7, 11], 10), False),
    (([3], 6), False),             # same element twice is not a pair
    (([], 0), False),
    (([-3, -1, 0, 2], -4), True),
])
stress(pair_sum_sorted, pair_sum_brute, lambda r: (sorted(rand_list(r)), r.randint(-6, 6)))

# %% [markdown]
# ### Warm-up 2.1: all zero-sum triples (classic "3-sum"; ~20 min)
# Given a list of integers, return every distinct triple of values $(x, y, z)$ with $x \le y \le z$ that comes
# from three different positions and sums to 0. Return a sorted list of tuples with no duplicates.
# *Target:* $O(n^2)$ time. Fix the smallest element, run the two-pointer template on the rest, and skip equal
# neighbours so that no triple appears twice.

# %%
def three_sum_zero(nums):
    ### BEGIN SOLUTION
    a = sorted(nums)
    res = []
    for i in range(len(a) - 2):
        if i > 0 and a[i] == a[i - 1]:
            continue                                   # same first value -> same triples
        lo, hi = i + 1, len(a) - 1
        while lo < hi:
            t = a[i] + a[lo] + a[hi]
            if t < 0:
                lo += 1
            elif t > 0:
                hi -= 1
            else:
                res.append((a[i], a[lo], a[hi]))
                lo, hi = lo + 1, hi - 1
                while lo < hi and a[lo] == a[lo - 1]:
                    lo += 1                            # skip duplicate second values
    return res
    ### END SOLUTION


def three_sum_brute(nums):
    return sorted({tuple(sorted(t)) for t in itertools.combinations(nums, 3) if sum(t) == 0})


run_tests(three_sum_zero, [
    (([-1, 0, 1, 2, -1, -4],), [(-1, -1, 2), (-1, 0, 1)]),
    (([0, 0, 0, 0],), [(0, 0, 0)]),
    (([0, 1],), []),
    (([],), []),
    (([-2, 1, 1, 1],), [(-2, 1, 1)]),
])
stress(three_sum_zero, three_sum_brute, lambda r: (rand_list(r, -3, 3, 9),))

# %% [markdown]
# ## 3. Sliding window (fixed and variable)
# **Intuition.** For a property of **contiguous** ranges, do not recompute each window from scratch. Move
# the window and update an aggregate (sum, counts) in $O(1)$. A *fixed* window adds one element and removes one per
# step. A *variable* window grows `right` every step and shrinks `left` while the window is invalid. Each index
# enters and leaves at most once, so the total is $O(n)$ even though there is a nested `while`.
#
# **Signals:** "contiguous subarray/substring", "longest/shortest window such that ...", "at most k
# distinct/replacements", "window of size k". The variable-window version needs **monotonicity**: if a window is
# valid, its sub-windows are valid too (or, for "shortest" problems, growing a valid window keeps it valid).
# Negative numbers usually break this for sum constraints. Use prefix sums then (section 4).
#
# **Template A: fixed window of size k.**

# %%
def max_window_sum(nums, k):
    """Largest sum of a length-k contiguous window, 1 <= k <= len(nums). O(n)."""
    ### BEGIN SOLUTION
    w = sum(nums[:k])
    best = w
    for i in range(k, len(nums)):
        w += nums[i] - nums[i - k]
        best = max(best, w)
    return best
    ### END SOLUTION


def max_window_sum_brute(nums, k):
    return max(sum(nums[i:i + k]) for i in range(len(nums) - k + 1))


def _gen_window(r):
    a = rand_list(r, -5, 5, 8, 1)
    return a, r.randint(1, len(a))


run_tests(max_window_sum, [
    (([1, 3, -1, -3, 5, 3, 6, 7], 3), 16),
    (([-5, -2, -7], 1), -2),                 # all negative: do not start best at 0
    (([4, -1, 2], 3), 5),                    # k == n
    (([2], 1), 2),
])
stress(max_window_sum, max_window_sum_brute, _gen_window)

# %% [markdown]
# **Template B: variable window ("longest valid").** Grow `right`. While the window is invalid, shrink
# `left`. Record the best after restoring validity. Example property: at most `k` distinct characters.

# %%
def longest_at_most_k_distinct(s_, k):
    """Length of the longest substring of s_ with at most k distinct characters (k >= 0). O(n)."""
    ### BEGIN SOLUTION
    count = Counter()
    left = best = 0
    for right, ch in enumerate(s_):
        count[ch] += 1
        while len(count) > k:
            count[s_[left]] -= 1
            if count[s_[left]] == 0:
                del count[s_[left]]          # otherwise len(count) overcounts distinct letters
            left += 1
        best = max(best, right - left + 1)
    return best
    ### END SOLUTION


def longest_k_distinct_brute(s_, k):
    return max([j - i for i in range(len(s_) + 1) for j in range(i, len(s_) + 1)
                if len(set(s_[i:j])) <= k], default=0)


run_tests(longest_at_most_k_distinct, [
    (("eceba", 2), 3),
    (("aa", 1), 2),
    (("", 2), 0),
    (("abc", 0), 0),
    (("abaccc", 2), 4),
    (("abc", 5), 3),
])
stress(longest_at_most_k_distinct, longest_k_distinct_brute,
       lambda r: ("".join(r.choice("abc") for _ in range(r.randint(0, 9))), r.randint(0, 3)))

# %% [markdown]
# ### Warm-up 3.1: shortest window reaching a sum (classic; ~10 min)
# Given a list of **positive** integers and a target, return the length of the shortest contiguous window
# whose sum is at least the target, or 0 if none exists. *Variant of Template B:* this time you shrink
# **while the window is valid** and record the best inside the loop.

# %%
def shortest_subarray_at_least(nums, target):
    ### BEGIN SOLUTION
    left = run = 0
    best = float("inf")
    for right, x in enumerate(nums):
        run += x
        while run >= target:
            best = min(best, right - left + 1)
            run -= nums[left]
            left += 1
    return 0 if best == float("inf") else best
    ### END SOLUTION


def shortest_at_least_brute(nums, target):
    return min([j - i for i in range(len(nums)) for j in range(i + 1, len(nums) + 1)
                if sum(nums[i:j]) >= target], default=0)


run_tests(shortest_subarray_at_least, [
    (([2, 3, 1, 2, 4, 3], 7), 2),
    (([1, 4, 4], 4), 1),
    (([1, 1, 1, 1], 11), 0),
    (([], 1), 0),
    (([1, 1, 1, 10], 10), 1),      # one-step shrinking (`if` instead of `while`) returns 4 here
])
stress(shortest_subarray_at_least, shortest_at_least_brute,
       lambda r: (rand_list(r, 1, 6, 8), r.randint(1, 15)))

# %% [markdown]
# ## 4. Prefix sums (+ hash map)
# **Intuition.** With $P[0]=0$ and $P[i+1]=P[i]+a_i$, any range sum is a difference: $\sum a[l:r] = P[r]-P[l]$
# (half-open). "Subarray sum equals k" then becomes "how many earlier prefixes equal $P[r]-k$?", which is the
# hash-map complement template from section 1 applied to prefixes. This works with negative numbers, where
# a sliding window does not.
#
# **Signals:** many range-sum queries on a static array, "subarray sum equals / divisible by k" with negatives,
# "equal number of 0s and 1s" (map 0 to -1), 2-D region sums (2-D prefix table), difference arrays for range updates.
#
# **Template: prefix array (length n+1).**

# %%
def prefix_sums(nums):
    """Return P with len(P) == len(nums) + 1, P[0] == 0 and sum(nums[l:r]) == P[r] - P[l]."""
    ### BEGIN SOLUTION
    P = [0]
    for x in nums:
        P.append(P[-1] + x)
    return P
    ### END SOLUTION


def _check_prefix(nums):
    P = prefix_sums(nums)
    return len(P) == len(nums) + 1 and all(P[r] - P[l] == sum(nums[l:r])
                                           for l in range(len(nums) + 1) for r in range(l, len(nums) + 1))


run_tests(prefix_sums, [(([3, -1, 2],), [0, 3, 2, 4]), (([],), [0])])
stress(_check_prefix, lambda nums: True, lambda r: (rand_list(r),), name="prefix_sums range identity")

# %% [markdown]
# ### Warm-up 4.1: count subarrays with sum exactly k (classic; ~15 min)
# Given integers (negatives allowed) and `k`, count the contiguous non-empty subarrays whose sum is `k`.
# *Target:* $O(n)$. Keep a running prefix and a `Counter` of prefixes seen so far. Decide what the counter
# must contain **before** the first element.

# %%
def count_subarrays_with_sum(nums, k):
    ### BEGIN SOLUTION
    seen = Counter({0: 1})        # the empty prefix: subarrays that start at index 0
    run = count = 0
    for x in nums:
        run += x
        count += seen[run - k]
        seen[run] += 1
    return count
    ### END SOLUTION


def count_subarrays_brute(nums, k):
    return sum(1 for i in range(len(nums)) for j in range(i + 1, len(nums) + 1) if sum(nums[i:j]) == k)


run_tests(count_subarrays_with_sum, [
    (([1, 1, 1], 2), 2),
    (([1, 2, 3], 3), 2),
    (([1, -1, 0], 0), 3),          # negatives and zeros: a sliding window fails here
    (([], 0), 0),                  # no non-empty subarray
    (([5], 5), 1),
])
stress(count_subarrays_with_sum, count_subarrays_brute, lambda r: (rand_list(r, -3, 3), r.randint(-3, 3)))

# %% [markdown]
# ## Say it out loud (complexity and correctness)
# 1. `dict`/`set` lookups are "O(1)". Average or worst case? What makes them degrade, and what is the worst case?
# 2. The variable sliding window has a `while` inside a `for`. Why is it still $O(n)$?
# 3. Why does a sliding window fail for "count subarrays with sum k" when negatives are allowed? What replaces it?
# 4. 3-sum: what is your time and space complexity? Is $o(n^2)$ possible?
# 5. Group anagrams: compare a sorted-string key with a 26-count key. What are the time per word and the assumptions?
# 6. Prove the two-pointer pair-sum scan never skips a valid pair.
# 7. Python-specific: what do `nums[i:j]`, `x in some_list` and `s += ch` in a loop cost?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Average/expected $O(1)$ under a good hash function (CLRS Ch. 11). The worst case is $O(n)$ per operation when
#    many keys collide. Adversarial inputs against a known hash can trigger it, which is why CPython randomizes
#    `str` hashing per process. The Python wiki lists dict get/set as average $O(1)$, amortized worst $O(n)$.
# 2. Each index is added once (when `right` passes it) and removed at most once (when `left` passes it). The
#    total work over the whole loop is at most $2n$ pointer moves (aggregate/amortized argument), so $O(n)$.
# 3. Shrinking assumes that removing elements cannot make a too-big sum bigger. With negatives, removing an element can
#    *increase* the sum, so validity is no longer monotone in the window. Prefix sums with a hash map of prefix counts
#    give $O(n)$ for any integers.
# 4. $O(n^2)$ time: an $O(n\log n)$ sort plus $n$ two-pointer scans. Extra space is $O(1)$ beyond the output and the
#    sort's buffer. 3SUM is the canonical "believed roughly quadratic" problem (Gajentaan and Overmars 1995). Grønlund and
#    Pettie (arXiv:1404.0799) showed it can be beaten only by log factors. In an interview $O(n^2)$ is the expected answer.
# 5. A sorted key costs $O(L\log L)$ per word of length $L$. A count tuple costs $O(L + 26)$ but assumes a fixed small
#    alphabet. Both give total $O(nL\log L)$ or $O(nL)$ time and $O(nL)$ space for the groups.
# 6. Invariant: every valid pair has indices in $[i, j]$. If $a_i + a_j < t$, then $a_i + a_m \le a_i + a_j < t$ for every
#    $m \le j$, so $a_i$ is in no valid pair inside $[i,j]$ and dropping it keeps the invariant. The case $> t$ is symmetric.
# 7. Slicing `nums[i:j]` copies, costing $O(j-i)$. `x in list` is a linear scan, $O(n)$. Repeated `s += ch` can be
#    $O(n^2)$ overall (CPython sometimes optimizes it in place, but do not rely on that). Build a list and `"".join` it.
#    Source: https://wiki.python.org/moin/TimeComplexity .
# </details>

# %% [markdown]
# <!-- practice:start -->
# ## Practice list (solve these on LeetCode)
#
# NeetCode 150 problems for this topic, generated from `neetcode-gh/leetcode@d5ebb3e` `.problemSiteData.json` (https://neetcode.io/practice). **B75** = also in Blind 75. Tick them off in your session copy; the full checklist is `A_coding/A1_problem_list.md`. Grind 75 (https://www.techinterviewhandbook.org/grind75/) is an alternative ordering of a similar set.
#
# **Arrays & Hashing** · plan week 7 · 9 problems
#
# - [ ] [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/) · Easy · **B75**
# - [ ] [Valid Anagram](https://leetcode.com/problems/valid-anagram/) · Easy · **B75**
# - [ ] [Two Sum](https://leetcode.com/problems/two-sum/) · Easy · **B75**
# - [ ] [Group Anagrams](https://leetcode.com/problems/group-anagrams/) · Medium · **B75**
# - [ ] [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) · Medium · **B75**
# - [ ] [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/) · Medium · **B75**
# - [ ] [Valid Sudoku](https://leetcode.com/problems/valid-sudoku/) · Medium
# - [ ] [Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/) · Medium · **B75**
# - [ ] [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) · Medium · **B75**
#
# **Two Pointers** · plan week 7 · 5 problems
#
# - [ ] [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/) · Easy · **B75**
# - [ ] [Two Sum II Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) · Medium
# - [ ] [3Sum](https://leetcode.com/problems/3sum/) · Medium · **B75**
# - [ ] [Container With Most Water](https://leetcode.com/problems/container-with-most-water/) · Medium · **B75**
# - [ ] [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) · Hard
#
# **Sliding Window** · plan week 8 · 6 problems
#
# - [ ] [Best Time to Buy And Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) · Easy · **B75**
# - [ ] [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) · Medium · **B75**
# - [ ] [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/) · Medium · **B75**
# - [ ] [Permutation In String](https://leetcode.com/problems/permutation-in-string/) · Medium
# - [ ] [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) · Hard · **B75**
# - [ ] [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) · Hard
#
# **Extras (not in NeetCode 150; close to the warm-ups above)**
#
# - [ ] [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) · Medium
# - [ ] [Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/) · Medium
# - [ ] [Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/) · Easy
# - [ ] [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/) · Medium
# - [ ] [Sort Colors](https://leetcode.com/problems/sort-colors/) · Medium
# <!-- practice:end -->

# %%
s.finish(lookups=0, score=None, notes="")
