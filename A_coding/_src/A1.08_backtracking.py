# %% [markdown]
# # A1.8 · Backtracking: subsets, permutations, combinations, pruning, constraint checking
#
# **ID:** A1.8 · **Target:** 60 min · **Plan week:** 5 (pattern notebook, Tue); second pass in week 13 · **Part:** A
#
# **The drill.** This is a 60-minute study notebook. Every backtracking solution follows the same skeleton: *choose,
# recurse, un-choose*, plus a rule for **what to try next** and a rule for **when to stop early** (pruning). Fill in the
# three canonical enumerations (subsets, permutations, combinations), then five warm-ups (8-15 min each): subsets
# with duplicates, combination sum, generating balanced parentheses, N-Queens and word search on a grid. Results are
# compared as **canonical sorted lists**, so you may produce them in any order, but duplicates count as errors. The
# references are `itertools`, filter-everything brute forces and OEIS counts.
#
# **Rules.** Python standard library only. Do not use `itertools` inside your solutions; the tests use it as the
# reference. Fill in every `raise NotImplementedError`.
#
# ## Sources
# - Python docs, `itertools.combinations` / `permutations` / `product` (reference enumerations and their counts):
#   https://docs.python.org/3/library/itertools.html
# - T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, 4th ed., MIT Press, 2022
#   (ISBN 9780262046305), Ch. 4 "Divide-and-Conquer" (recursion trees, for counting the nodes of a backtracking tree).
#   https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
# - OEIS A000170, *Number of ways of placing n nonattacking queens on an n × n board*: https://oeis.org/A000170 ;
#   OEIS A000108, *Catalan numbers* (number of balanced strings of n bracket pairs): https://oeis.org/A000108
# - Tech Interview Handbook, *Recursion* cheatsheet (backtracking, memoization, recursion-depth corner cases):
#   https://www.techinterviewhandbook.org/algorithms/recursion/ (yangshun/tech-interview-handbook @ e1d28e8, MIT).
# - NeetCode 150 problem data (practice list): https://neetcode.io/practice ,
#   https://github.com/neetcode-gh/leetcode @ d5ebb3e `.problemSiteData.json`.

# %%
import copy
import itertools
import math
import random

from prep import Session, check

s = Session("A1.8", target_min=60)
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


def canon(f, inner_sort=True):
    """Wrap an enumerator: returns the sorted list of results (each as a tuple). Duplicates are kept, so they fail."""
    def w(*args):
        return sorted(tuple(sorted(x)) if inner_sort else tuple(x) for x in f(*args))
    w.__name__ = f.__name__
    return w


def distinct_list(r, max_len=6):
    return r.sample(range(-5, 10), r.randint(0, max_len))

# %% [markdown]
# ## 1. The skeleton: choose, recurse, un-choose
# **Intuition.** Backtracking is a DFS over a **decision tree**. Each level makes one choice, a leaf is a complete
# candidate, and you undo the choice when you return (`path.pop()`), so a single `path` list is shared by the whole search.
# The work is (number of tree nodes) × (work per node). Pruning removes whole subtrees as early as possible.
#
# **Signals:** "return **all** subsets/permutations/combinations/partitions/placements", "generate every valid ...",
# small input ($n \le 10$-$20$), constraint satisfaction (N-Queens, Sudoku), path search on a grid with no revisits.
#
# ```python
# def backtrack(state):
#     if is_complete(state): record(copy of state); return
#     for choice in candidates(state):        # in an order that makes pruning and de-duplication easy
#         if not feasible(state, choice): continue   # (or `break` if the candidates are sorted)
#         apply(choice); backtrack(state); undo(choice)
# ```
#
# **Template A: all subsets of distinct values.** At each index, the next element to add is chosen from
# `nums[start:]`. Every node of the tree is a subset, so record the path at every node.

# %%
def subsets(nums):
    ### BEGIN SOLUTION
    res, path = [], []

    def bt(start):
        res.append(path[:])                  # copy! path keeps changing
        for i in range(start, len(nums)):
            path.append(nums[i])
            bt(i + 1)
            path.pop()

    bt(0)
    return res
    ### END SOLUTION


def subsets_brute(nums):
    return sorted(tuple(sorted(c)) for k in range(len(nums) + 1) for c in itertools.combinations(nums, k))


run_tests(canon(subsets), [(([1, 2, 3],), subsets_brute([1, 2, 3])), (([],), [()])])
check("subsets count is 2^n", len(subsets([1, 2, 3, 4, 5])) == 32)
stress(canon(subsets), subsets_brute, lambda r: (distinct_list(r),))

# %% [markdown]
# **Template B: all permutations of distinct values.** At each depth, try every unused element. Track the used ones
# with a boolean list, which is $O(1)$ per check.

# %%
def permutations(nums):
    ### BEGIN SOLUTION
    res, path, used = [], [], [False] * len(nums)

    def bt():
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i, x in enumerate(nums):
            if not used[i]:
                used[i] = True
                path.append(x)
                bt()
                path.pop()
                used[i] = False

    bt()
    return res
    ### END SOLUTION


perm_brute = lambda nums: sorted(itertools.permutations(nums))
run_tests(canon(permutations, inner_sort=False), [(([1, 2, 3],), perm_brute([1, 2, 3])), (([],), [()]), (([7],), [(7,)])])
stress(canon(permutations, inner_sort=False), perm_brute, lambda r: (distinct_list(r, 5),))

# %% [markdown]
# **Template C: all k-element combinations of `1..n`.** Like subsets, but record only at depth k, and **prune**:
# when fewer elements remain than you still need, stop the loop early.

# %%
def combinations(n, k):
    ### BEGIN SOLUTION
    res, path = [], []

    def bt(start):
        if len(path) == k:
            res.append(path[:])
            return
        need = k - len(path)
        for i in range(start, n - need + 2):      # prune: leave room for `need` numbers in [i, n]
            path.append(i)
            bt(i + 1)
            path.pop()

    bt(1)
    return res
    ### END SOLUTION


comb_brute = lambda n, k: sorted(itertools.combinations(range(1, n + 1), k))
run_tests(canon(combinations), [((4, 2), comb_brute(4, 2)), ((3, 0), [()]), ((3, 3), [(1, 2, 3)]), ((2, 3), [])])
stress(canon(combinations), comb_brute, lambda r: (lambda n: (n, r.randint(0, n + 1)))(r.randint(0, 7)))

# %% [markdown]
# ## 2. De-duplication and pruning
# ### Warm-up 2.1: subsets when values repeat (~10 min)
# `nums` may contain duplicates. Return every **distinct** sub-multiset exactly once. *Hint:* sort first. At a given
# depth, skip `nums[i]` if it equals `nums[i-1]` and `i > start`: choosing the second copy at the same depth would
# rebuild a subtree you already explored.

# %%
def subsets_with_dup(nums):
    ### BEGIN SOLUTION
    nums = sorted(nums)
    res, path = [], []

    def bt(start):
        res.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue                          # same value at the same depth -> duplicate subtree
            path.append(nums[i])
            bt(i + 1)
            path.pop()

    bt(0)
    return res
    ### END SOLUTION


def subsets_dup_brute(nums):
    return sorted({tuple(sorted(c)) for k in range(len(nums) + 1) for c in itertools.combinations(nums, k)})


run_tests(canon(subsets_with_dup), [(([1, 2, 2],), [(), (1,), (1, 2), (1, 2, 2), (2,), (2, 2)]), (([0],), [(), (0,)]),
                                    (([2, 2, 2],), [(), (2,), (2, 2), (2, 2, 2)])])
stress(canon(subsets_with_dup), subsets_dup_brute, lambda r: ([r.randint(0, 3) for _ in range(r.randint(0, 7))],))

# %% [markdown]
# ### Warm-up 2.2: combination sum with unlimited reuse (~12 min)
# `cands` are **distinct positive** integers. Return every multiset of them (each value may be used any number of
# times) that sums to `target`, each multiset once. *Hint:* sort, and pass `start = i` (not `i + 1`) to allow reuse.
# Once `cands[i]` exceeds the remaining amount, `break`: every later candidate is bigger.

# %%
def combination_sum(cands, target):
    ### BEGIN SOLUTION
    cands = sorted(cands)
    res, path = [], []

    def bt(start, remaining):
        if remaining == 0:
            res.append(path[:])
            return
        for i in range(start, len(cands)):
            if cands[i] > remaining:
                break                             # sorted -> prune the rest of this level
            path.append(cands[i])
            bt(i, remaining - cands[i])           # i, not i + 1: reuse allowed
            path.pop()

    bt(0, target)
    return res
    ### END SOLUTION


def combination_sum_brute(cands, target):
    out = set()
    for k in range(target // min(cands) + 1 if cands else 1):
        for c in itertools.combinations_with_replacement(sorted(cands), k):
            if sum(c) == target:
                out.add(c)
    return sorted(out)


run_tests(canon(combination_sum), [(([2, 3, 6, 7], 7), [(2, 2, 3), (7,)]), (([2, 3, 5], 8), [(2, 2, 2, 2), (2, 3, 3), (3, 5)]),
                                   (([2], 1), []), (([1], 2), [(1, 1)])])
stress(canon(combination_sum), combination_sum_brute,
       lambda r: (r.sample(range(1, 8), r.randint(1, 4)), r.randint(1, 10)))

# %% [markdown]
# ### Warm-up 2.3: all balanced strings of n bracket pairs (~8 min)
# Return every string of `n` `(` and `n` `)` that is balanced. *Pruning rule:* add `(` while fewer than `n` are open,
# and add `)` only while it would not close more than were opened. The tree then contains only valid prefixes, and
# the count is the Catalan number $C_n$ (OEIS A000108).

# %%
def gen_parens(n):
    ### BEGIN SOLUTION
    res, path = [], []

    def bt(opened, closed):
        if len(path) == 2 * n:
            res.append("".join(path))
            return
        if opened < n:
            path.append("(")
            bt(opened + 1, closed)
            path.pop()
        if closed < opened:
            path.append(")")
            bt(opened, closed + 1)
            path.pop()

    bt(0, 0)
    return res
    ### END SOLUTION


def gen_parens_brute(n):
    def ok(s_):
        bal = 0
        for ch in s_:
            bal += 1 if ch == "(" else -1
            if bal < 0:
                return False
        return bal == 0
    return sorted("".join(p) for p in itertools.product("()", repeat=2 * n) if ok(p))


sorted_out = lambda f: (lambda *a: sorted(f(*a)))
run_tests(sorted_out(gen_parens), [((1,), ["()"]), ((2,), ["(())", "()()"]), ((0,), [""])], name="gen_parens")
stress(sorted_out(gen_parens), gen_parens_brute, lambda r: (r.randint(0, 6),), n=30, name="gen_parens")
catalan = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862]              # OEIS A000108
check("gen_parens counts are Catalan numbers", [len(gen_parens(n)) for n in range(10)] == catalan)

# %% [markdown]
# ## 3. Constraint checking
# ### Warm-up 3.1: count N-Queens placements (~12 min)
# Count the ways to place `n` queens on an `n × n` board so that no two share a row, column or diagonal. Place one queen
# per row and keep three sets: columns, diagonals `r - c` and anti-diagonals `r + c`. Each conflict test is then $O(1)$.

# %%
def n_queens(n):
    ### BEGIN SOLUTION
    cols, diag, anti = set(), set(), set()

    def bt(r):
        if r == n:
            return 1
        count = 0
        for c in range(n):
            if c in cols or (r - c) in diag or (r + c) in anti:
                continue
            cols.add(c); diag.add(r - c); anti.add(r + c)
            count += bt(r + 1)
            cols.remove(c); diag.remove(r - c); anti.remove(r + c)
        return count

    return bt(0)
    ### END SOLUTION


def n_queens_brute(n):                        # one queen per row and column = a permutation; check diagonals only
    return sum(1 for p in itertools.permutations(range(n))
               if len({r - c for r, c in enumerate(p)}) == n and len({r + c for r, c in enumerate(p)}) == n)


oeis_a000170 = [1, 1, 0, 0, 2, 10, 4, 40, 92]                   # n = 0..8
run_tests(n_queens, [((n,), v) for n, v in enumerate(oeis_a000170)])
stress(n_queens, n_queens_brute, lambda r: (r.randint(0, 7),), n=20)

# %% [markdown]
# ### Warm-up 3.2: find a word on a letter grid (~15 min)
# `board` is a list of strings. Can `word` be spelled by a path of 4-adjacent cells that uses each cell at most once?
# Mark a cell as used **before** you recurse and unmark it **after**. Prune immediately on a letter mismatch. An empty
# word is always found.

# %%
def word_search(board, word):
    ### BEGIN SOLUTION
    h, w = len(board), len(board[0])
    used = [[False] * w for _ in range(h)]

    def bt(r, c, i):
        if i == len(word):
            return True
        if not (0 <= r < h and 0 <= c < w) or used[r][c] or board[r][c] != word[i]:
            return False
        used[r][c] = True
        found = any(bt(r + dr, c + dc, i + 1) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        used[r][c] = False                    # un-choose, even on success paths you abandon
        return found

    return any(bt(r, c, 0) for r in range(h) for c in range(w)) if word else True
    ### END SOLUTION


def word_search_brute(board, word):           # explicit-stack search over (cell, index, visited set)
    if not word:
        return True
    h, w = len(board), len(board[0])
    stack = [((r, c), 1, frozenset([(r, c)])) for r in range(h) for c in range(w) if board[r][c] == word[0]]
    while stack:
        (r, c), i, seen = stack.pop()
        if i == len(word):
            return True
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and board[nr][nc] == word[i]:
                stack.append(((nr, nc), i + 1, seen | {(nr, nc)}))
    return False


B = ["ABCE", "SFCS", "ADEE"]
run_tests(word_search, [((B, "ABCCED"), True), ((B, "SEE"), True), ((B, "ABCB"), False),   # ABCB would reuse B
                        ((["a"], "a"), True), ((["a"], "aa"), False), ((["ab"], ""), True)])
stress(word_search, word_search_brute,
       lambda r: (["".join(r.choice("ab") for _ in range(w)) for w in [r.randint(1, 4)] for _ in range(r.randint(1, 4))],
                  "".join(r.choice("ab") for _ in range(r.randint(1, 6)))), n=500)

# %% [markdown]
# ## Say it out loud (complexity and correctness)
# 1. Give the output size and the total time for subsets, permutations and k-combinations of $n$ items. Why is the
#    time "output size × n"?
# 2. Why must you append `path[:]` (a copy), not `path`?
# 3. Subsets with duplicates: why is the skip condition `i > start` and not `i > 0`?
# 4. How many balanced strings of n pairs are there, asymptotically? How does that compare with the $4^n$ brute force?
# 5. N-Queens: what is a simple upper bound on the work, and what do the three sets buy you?
# 6. Word search: time complexity for an $R \times C$ board and a word of length $L$? Which pruning helps most in practice?
# 7. When does backtracking turn into DP (memoization)? Give an example from this notebook's neighbours.

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Subsets: $2^n$ outputs, $O(n\,2^n)$ time. Permutations: $n!$ outputs, $O(n \cdot n!)$ time. Combinations: $\binom nk$
#    outputs, $O(k\binom nk)$ time. Each recorded output costs $O(n)$ (or $O(k)$) to copy. For subsets every tree node
#    is an output. The permutation tree has $\sum_k n!/(n-k)! \approx e\cdot n!$ nodes, which is still $O(n!)$. Extra
#    space is $O(n)$ for the recursion and the path, excluding the output.
# 2. `path` is a single list that keeps being mutated. Appending the object stores the same reference many times, and
#    every entry ends up as the final, empty state.
# 3. `i > start` skips a duplicate only as an alternative **at the same depth** (a sibling). A duplicate deeper in the
#    tree, such as the second 2 in `[2, 2]`, is still allowed. With `i > 0`, `[2, 2]` could never be built.
# 4. $C_n = \frac{1}{n+1}\binom{2n}{n} \sim \frac{4^n}{n^{3/2}\sqrt\pi}$ (OEIS A000108). That is only a polynomial
#    factor below the $4^n$ strings of the brute force, but pruning also means no invalid prefix is ever extended.
#    The total work is $O(n\,C_n)$.
# 5. One queen per row with distinct columns gives at most $n!$ leaves, so $O(n!)$ with $O(1)$ checks. Diagonal
#    pruning cuts far below that: there are 92 solutions for $n = 8$ out of $8! = 40320$ permutations (OEIS A000170).
#    The sets make each conflict check $O(1)$ instead of $O(n)$ scans.
# 6. $O(RC \cdot 4 \cdot 3^{L-1})$: any start cell, 4 first moves, then at most 3 (you cannot go back). Pruning on the
#    first mismatching letter helps most. Other tricks: count letters first (the board must contain the word's
#    multiset), and start from the rarer end of the word.
# 7. When different paths reach the **same subproblem state** and you only need a count, an optimum or a yes/no
#    answer, not the list of solutions. Combination sum asks for the list (backtracking). Counting the combinations
#    is coin change II (DP over `(index, remaining)`, A1.9). Word break is "can I split this string": memoize on the
#    start index.
# </details>

# %% [markdown]
# <!-- practice:start -->
# ## Practice list (solve these on LeetCode)
#
# NeetCode 150 problems for this topic, generated from `neetcode-gh/leetcode@d5ebb3e` `.problemSiteData.json` (https://neetcode.io/practice). **B75** = also in Blind 75. Tick them off in your session copy; the full checklist is `A_coding/A1_problem_list.md`. Grind 75 (https://www.techinterviewhandbook.org/grind75/) is an alternative ordering of a similar set.
#
# **Backtracking** · plan week 13 · 9 problems
#
# - [ ] [Subsets](https://leetcode.com/problems/subsets/) · Medium
# - [ ] [Combination Sum](https://leetcode.com/problems/combination-sum/) · Medium · **B75**
# - [ ] [Permutations](https://leetcode.com/problems/permutations/) · Medium
# - [ ] [Subsets II](https://leetcode.com/problems/subsets-ii/) · Medium
# - [ ] [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/) · Medium
# - [ ] [Word Search](https://leetcode.com/problems/word-search/) · Medium · **B75**
# - [ ] [Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/) · Medium
# - [ ] [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) · Medium
# - [ ] [N Queens](https://leetcode.com/problems/n-queens/) · Hard
#
# **Extras (not in NeetCode 150; close to the warm-ups above)**
#
# - [ ] [Combinations](https://leetcode.com/problems/combinations/) · Medium
# - [ ] [N Queens II](https://leetcode.com/problems/n-queens-ii/) · Hard
# <!-- practice:end -->

# %%
s.finish(lookups=0, score=None, notes="")
