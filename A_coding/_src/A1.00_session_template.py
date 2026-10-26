# %% [markdown]
# # A1.0 · LeetCode session template (copy for every problem)
#
# **ID:** A1.0 · **Target:** 25 min · **Plan week:** 1-16 (schedule.yaml: this template on day 1, then one 25-min problem every weekday) · **Part:** A (coding fundamentals)
#
# **The drill.** One problem, 25 minutes, out loud, with a timer. Pick a problem from the practice list
# at the bottom (or from a topic notebook), run `python tools/new_session.py A1.0`, and work through
# the eight steps below in order. The minute budgets add up to 25. Target from the guide: *solve a
# LeetCode medium in 25 min*. The mock checklist asks for *3 unseen mediums in 25 min each, talking aloud*.
#
# **Why from week 1.** The guide starts LeetCode in week 7. Anonymous candidate reports from 2025-26
# (self-reported, e.g. on Glassdoor, so formats vary) describe Meta research-scientist and intern loops
# that open with a timed online screen of several LeetCode-medium problems. That screen can come before any
# ML round, so keep a light daily problem going from week 1.
#
# **The steps** follow CodePath's UMPIRE method (Understand, Match, Plan, Implement, Review, Evaluate)
# and the "during the interview" checklist of the Tech Interview Handbook. This template adds an explicit
# brute-force step and a randomized stress test, because a stress test is the fastest way to find your own
# bug before the interviewer does.
#
# **Rules.** Use Python only, with no docs and no autocomplete. Name things `cases`, `brute`, `solve` and
# `gen` so the test cells run unchanged. Write the LeetCode problem in your own words in step 0; do not
# paste the statement. The solution notebook works one example (product of the array except self) through
# every step, so you can see what a finished session looks like.
#
# ## Sources
# - CodePath, *UMPIRE Interview Strategy* (CodePath Guides, Computer Science): six steps, Understand /
#   Match / Plan / Implement / Review / Evaluate. https://guides.codepath.org/compsci/UMPIRE-Interview-Strategy
# - Y. Tay et al., *Tech Interview Handbook*: "Coding interview cheatsheet: best practices before, during and
#   after" (clarify, discuss approaches + complexity before coding, talk while coding, test without being
#   asked) https://www.techinterviewhandbook.org/coding-interview-cheatsheet/ ; "Coding interview
#   techniques" (how to optimize) https://www.techinterviewhandbook.org/coding-interview-techniques/ ;
#   "Coding interview rubrics" (communication, problem solving, technical competency, testing)
#   https://www.techinterviewhandbook.org/coding-interview-rubrics/ . Source repo:
#   https://github.com/yangshun/tech-interview-handbook @ e1d28e8 `apps/website/contents/coding-interview-*.md`
#   (MIT). The checklists here are paraphrased.
# - NeetCode 150 problem data: https://neetcode.io/practice , file `.problemSiteData.json` in
#   https://github.com/neetcode-gh/leetcode @ d5ebb3e (used to generate the practice list).
# - Grind 75 (alternative ordering): https://www.techinterviewhandbook.org/grind75/
# - Python time complexity of built-ins (the hidden costs in step 8): https://wiki.python.org/moin/TimeComplexity

# %%
import copy
import random

from prep import Session, check

s = Session("A1.0", target_min=25)
random.seed(0)


def run_tests(fn, cases, name=None):
    """cases: list of (args_tuple, expected). Stops at the first failing case and prints it."""
    name = name or fn.__name__
    for i, (args, want) in enumerate(cases):
        got = fn(*copy.deepcopy(args))  # deepcopy: an in-place solution must not corrupt the next case
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


# the helpers must catch a wrong answer: a "max" that forgets negative numbers
def _max_wrong(a):
    best = 0
    for x in a:
        best = max(best, x)
    return best


try:
    stress(_max_wrong, max, lambda rng: ([rng.randint(-5, 5) for _ in range(rng.randint(1, 4))],))
    caught = False
except AssertionError as e:
    caught = True
    print("stress caught the bug, as it should:\n ", str(e).splitlines()[0])
check("stress catches an all-negative bug", caught)

# %% [markdown]
# ## 0. Problem (paste the link, restate it in one sentence) · 0.5 min

# %% [markdown] tags=["template_only"]
# **Link:** https://leetcode.com/problems/...
#
# **In my words:** ...

# %% [markdown] tags=["solution_only"]
# **Worked example (classic; LeetCode 238 "Product of Array Except Self",
# https://leetcode.com/problems/product-of-array-except-self/).** Given a list of integers, return a list
# whose entry $i$ is the product of every entry except the one at $i$. Do not use division, and run in $O(n)$.

# %% [markdown]
# ## 1. Understand: clarify before you think about code · 3 min
# Say the problem back in your own words, then ask 2-3 questions (Tech Interview Handbook). Defaults to run through:
# - **Size and range:** how big is $n$? What range are the values in? (This tells you the target complexity:
#   $n \le 20$ allows exponential time, $n \le 10^3$ allows $O(n^2)$, and $n \le 10^5$ means $O(n\log n)$ or better.)
# - **Values:** negatives? zeros? duplicates? floats? Can the answer overflow? (Python ints do not, but say so.)
# - **Shape:** can the input be empty or have one element? Is it sorted? Is the graph really a tree?
# - **Output:** what format? If several answers are valid, which one? What do I return when there is none?
# - **Constraints:** may I modify the input? Is there an extra-space limit? Will there be many queries
#   (precompute) or one?

# %% [markdown] tags=["solution_only"]
# **Worked example.** "So for every position I return the product of all the other entries. Can values be 0
# or negative?" Yes. "Is the length at least 2?" Assume $n \ge 2$, but also handle $n=1$ by returning `[1]`
# (the empty product). "Does the output list count as extra space?" No, so the target is $O(1)$ extra space
# besides the output. "No division, because a 0 would break it?" Yes.

# %% [markdown]
# ## 2. Examples and edge cases, written as tests · 2 min
# Work one example by hand first. Then add the cases the interviewer will not give you: empty or
# minimal input, duplicates, negatives or zeros, all-equal values, already-sorted or reversed input, and the extreme values.

# %% tags=["solution"]
# cases = [((args...), expected), ...]
# at least: 2 typical cases worked by hand + empty/one-element + duplicates + negatives/zeros + extremes
cases = [
    (([1, 2, 3, 4],), [24, 12, 8, 6]),
    (([2, 3],), [3, 2]),
    (([5],), [1]),                     # empty product
    (([0, 4, 5],), [20, 0, 0]),        # one zero: only its own slot is non-zero
    (([0, 4, 0],), [0, 0, 0]),         # two zeros: everything is zero
    (([-1, 2, -3],), [-6, 3, -2]),
    (([7, 7, 7],), [49, 49, 49]),
]

# %% [markdown]
# ## 3. Match: which pattern? · 1 min
# Rate a few of these as likely or unlikely, out loud (UMPIRE's "Match" step):
#
# | Signal in the problem | Pattern | Notebook |
# | --- | --- | --- |
# | "pair/triple with sum", "seen before", counting, anagrams | hash map / set; sort + two pointers | A1.1 |
# | contiguous subarray/substring, "longest/shortest window with property" | sliding window; prefix sums + hash map | A1.1 |
# | "next greater/smaller", matching brackets, nested structure | stack / monotonic stack | A1.2 |
# | sorted input, "minimum X such that ...", monotone yes/no | binary search (on index or answer) | A1.3 |
# | pointers, cycles, "k-th from the end", merge | fast/slow pointers, dummy head | A1.4 |
# | hierarchy, "ancestor", prefix of strings | tree DFS/BFS, BST invariants, trie | A1.5 |
# | "top k", "k-th largest", streaming median, overlapping ranges | heap; sort by start + sweep | A1.6 |
# | grid, dependencies, connectivity, shortest path | BFS/DFS, topo sort, union-find, Dijkstra | A1.7 |
# | "all combinations/permutations/placements", $n \le 20$ | backtracking with pruning | A1.8 |
# | "number of ways", "min/max cost", choices with overlapping subproblems | DP (1-D/2-D) | A1.9 |
# | "single/missing number", subsets as masks, powers of two | bit tricks | A1.10 |

# %% [markdown] tags=["solution_only"]
# **Worked example.** It is an array problem that combines "everything to the left" with "everything to the
# right", so **prefix/suffix aggregates** (A1.1) are likely. A hash map is unlikely, and so is a sliding window.

# %% [markdown]
# ## 4. Brute force first, with its complexity · 3 min
# State it, give its time and space complexity, and code it if it takes under about 3 minutes. It is your test oracle.

# %% tags=["solution"]
# def brute(...): the obviously-correct slow solution (it is the oracle for the stress test in step 7)
def brute(nums):
    out = []
    for i in range(len(nums)):
        p = 1
        for j, x in enumerate(nums):
            if j != i:
                p *= x
        out.append(p)
    return out                                     # O(n^2) time, O(1) extra space


# %%
run_tests(brute, cases)

# %% [markdown]
# ## 5. Plan / optimize, then agree before coding · 4 min
# Find the bottleneck in the brute force. Look for **repeated work** (cache it: hash map, prefix sums,
# memo), **the wrong data structure** (heap, sorted order, deque), and **unnecessary work** (stop early,
# prune). Say the best conceivable complexity, for example "I have to read every element, so $\Omega(n)$".
# Write 3-6 lines of pseudocode and state the target time and space. Ask "Shall I code this?" before you start.

# %% [markdown] tags=["solution_only"]
# **Worked example.** The brute force recomputes the same left products for every $i$.
# `answer[i] = (product of nums[:i]) * (product of nums[i+1:])`. One left-to-right pass writes the prefix
# products into `out`. One right-to-left pass multiplies in a running suffix product. That is $O(n)$ time and $O(1)$
# extra space besides the output. Division would fail on zeros, which is why the problem bans it.

# %% [markdown]
# ## 6. Implement while talking · 8 min
# Say each block's purpose before you write it ("this loop maintains the invariant ..."). Use clear names and
# no premature micro-optimizations.

# %% tags=["solution"]
# def solve(...): the optimized solution, same signature as brute
def solve(nums):
    n = len(nums)
    out = [1] * n
    for i in range(1, n):                # out[i] = product of nums[:i]
        out[i] = out[i - 1] * nums[i - 1]
    suffix = 1
    for i in range(n - 1, -1, -1):       # multiply in product of nums[i+1:]
        out[i] *= suffix
        suffix *= nums[i]
    return out


# %% [markdown]
# ## 7. Review and test: trace one example by hand, then run · 3 min
# Read your code as if someone else wrote it, looking for off-by-ones, an empty input and a forgotten
# `return`. Then run the hand cases and a stress test against `brute` on small random inputs. Keep the inputs small
# (sizes 0-8, values in a small range including negatives and 0). Collisions and edge cases then come up
# often, and a failing case is short enough to debug.

# %%
run_tests(solve, cases)

# %% tags=["solution"]
# def gen(rng): returns ONE random args tuple, small sizes, values that hit the edge cases
def gen(rng):
    return ([rng.randint(-3, 3) for _ in range(rng.randint(1, 7))],)


stress(solve, brute, gen)

# %% [markdown]
# ## 8. Evaluate: final complexity and trade-offs · 1 min
# Restate time and space complexity, and count Python's hidden costs: slicing, `x in list`, `list.pop(0)`
# and string `+=` in a loop are all $O(n)$ (https://wiki.python.org/moin/TimeComplexity). Then say what you
# would change for follow-ups: streaming input, many queries, a memory limit, or a parallel version.

# %% [markdown] tags=["solution_only"]
# **Worked example.** $O(n)$ time (two passes), $O(1)$ extra space besides the output. Follow-up: with
# division allowed, count the zeros. With 0 zeros use `total // x`. With 1 zero, only the zero's slot is non-zero.
# With 2 or more zeros, everything is 0. Products can grow huge: Python's big ints hide this, but in C++ or Java you
# would need to ask about overflow.

# %% [markdown]
# ## Talk while coding: prompts to say out loud
# - "Let me restate the problem... Can I assume ...?" (then **wait** for an answer)
# - "A brute force is ..., which is $O(\cdot)$. The repeated work is ..., so I can ..."
# - "The invariant of this loop is ..." / "This pointer always points to ..."
# - "I'm cutting a corner here (no input validation). In production I would ..."
# - "Let me trace `[1, 2, 3]` through this: after the first loop `out` is ..."
# - "Edge cases: empty, one element, all equal, negatives. Let me check each one against the code."
# - "Final complexity: time ..., space ... (not counting the output)."
# - When stuck: "Let me try a smaller example by hand" or "What if the input were sorted?", "What if I
#   process it from the right?", "What do I know after seeing the first $i$ elements?"

# %% [markdown]
# ## Say it out loud (after the session)
# 1. Which clarifying question would have changed your approach the most for this problem?
# 2. What was the brute force's complexity, and which step removed the repeated work?
# 3. Which edge case did you almost miss? Add it to your personal default list.
# 4. Did the stress test find anything? If it did, what kind of bug was it (off-by-one, empty input, duplicates)?
# 5. Rate yourself 1-5 on the four rubric axes (communication, problem solving, technical competency,
#    testing) and log the lowest one in `s.finish(notes=...)`.
# 6. Why stress-test with *small* random inputs rather than large ones?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers (worked example)</summary>
#
# 1. "Is division allowed?" and "Can there be zeros?" Allowed division with no zeros is a one-liner. Zeros make
#    the division approach need special cases, and banning division forces prefix/suffix products.
# 2. The brute force is $O(n^2)$, because for each $i$ it multiplies the other $n-1$ values. The left products shared between
#    neighbouring $i$ are recomputed every time. A prefix pass and a suffix pass compute each product once: $O(n)$.
# 3. Two zeros (every output is 0) and $n = 1$ (the empty product is 1).
# 4. In a scratch run, initializing `suffix = nums[-1]` instead of 1 fails both the hand cases and the stress test.
#    Off-by-one in the range of the prefix loop is the usual bug in this problem.
# 5. The four axes follow the Tech Interview Handbook rubric. Most people score themselves lowest on testing:
#    they announce "done" instead of tracing a case and listing edge cases.
# 6. Small inputs reach the edge cases (empty, duplicates, zeros, ties) with high probability, run thousands of
#    trials in under a second, and give a failing case short enough to trace by hand. Large inputs test speed,
#    not correctness. Time a separate large case if speed is in question.
# </details>

# %% [markdown]
# <!-- practice:start -->
# ## Practice list: a light daily ladder for weeks 1-6
#
# One problem a day with this template, before the topic notebooks start in week 7. First the 28 NeetCode 150 Easy problems, then the first Blind 75 Medium of each NeetCode pattern (generated from `neetcode-gh/leetcode@d5ebb3e` `.problemSiteData.json`, https://neetcode.io/practice). **B75** = Blind 75. Full checklist by topic: `A_coding/A1_problem_list.md`; alternative ordering: Grind 75 (https://www.techinterviewhandbook.org/grind75/).
#
# **Easy (28)**
#
# - [ ] [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/) · Easy · **B75** · Arrays & Hashing
# - [ ] [Valid Anagram](https://leetcode.com/problems/valid-anagram/) · Easy · **B75** · Arrays & Hashing
# - [ ] [Two Sum](https://leetcode.com/problems/two-sum/) · Easy · **B75** · Arrays & Hashing
# - [ ] [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/) · Easy · **B75** · Two Pointers
# - [ ] [Best Time to Buy And Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) · Easy · **B75** · Sliding Window
# - [ ] [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) · Easy · **B75** · Stack
# - [ ] [Binary Search](https://leetcode.com/problems/binary-search/) · Easy · Binary Search
# - [ ] [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) · Easy · **B75** · Linked List
# - [ ] [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) · Easy · **B75** · Linked List
# - [ ] [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/) · Easy · **B75** · Linked List
# - [ ] [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/) · Easy · **B75** · Trees
# - [ ] [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/) · Easy · **B75** · Trees
# - [ ] [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/) · Easy · Trees
# - [ ] [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/) · Easy · Trees
# - [ ] [Same Tree](https://leetcode.com/problems/same-tree/) · Easy · **B75** · Trees
# - [ ] [Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/) · Easy · **B75** · Trees
# - [ ] [Kth Largest Element In a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/) · Easy · Heap / Priority Queue
# - [ ] [Last Stone Weight](https://leetcode.com/problems/last-stone-weight/) · Easy · Heap / Priority Queue
# - [ ] [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) · Easy · **B75** · 1-D Dynamic Programming
# - [ ] [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/) · Easy · 1-D Dynamic Programming
# - [ ] [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/) · Easy · **B75** · Intervals
# - [ ] [Happy Number](https://leetcode.com/problems/happy-number/) · Easy · Math & Geometry
# - [ ] [Plus One](https://leetcode.com/problems/plus-one/) · Easy · Math & Geometry
# - [ ] [Single Number](https://leetcode.com/problems/single-number/) · Easy · Bit Manipulation
# - [ ] [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/) · Easy · **B75** · Bit Manipulation
# - [ ] [Counting Bits](https://leetcode.com/problems/counting-bits/) · Easy · **B75** · Bit Manipulation
# - [ ] [Reverse Bits](https://leetcode.com/problems/reverse-bits/) · Easy · **B75** · Bit Manipulation
# - [ ] [Missing Number](https://leetcode.com/problems/missing-number/) · Easy · **B75** · Bit Manipulation
#
# **First Mediums (15)**
#
# - [ ] [Group Anagrams](https://leetcode.com/problems/group-anagrams/) · Medium · **B75** · Arrays & Hashing
# - [ ] [3Sum](https://leetcode.com/problems/3sum/) · Medium · **B75** · Two Pointers
# - [ ] [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) · Medium · **B75** · Sliding Window
# - [ ] [Find Minimum In Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) · Medium · **B75** · Binary Search
# - [ ] [Reorder List](https://leetcode.com/problems/reorder-list/) · Medium · **B75** · Linked List
# - [ ] [Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) · Medium · **B75** · Trees
# - [ ] [Implement Trie Prefix Tree](https://leetcode.com/problems/implement-trie-prefix-tree/) · Medium · **B75** · Tries
# - [ ] [Combination Sum](https://leetcode.com/problems/combination-sum/) · Medium · **B75** · Backtracking
# - [ ] [Number of Islands](https://leetcode.com/problems/number-of-islands/) · Medium · **B75** · Graphs
# - [ ] [House Robber](https://leetcode.com/problems/house-robber/) · Medium · **B75** · 1-D Dynamic Programming
# - [ ] [Unique Paths](https://leetcode.com/problems/unique-paths/) · Medium · **B75** · 2-D Dynamic Programming
# - [ ] [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) · Medium · **B75** · Greedy
# - [ ] [Insert Interval](https://leetcode.com/problems/insert-interval/) · Medium · **B75** · Intervals
# - [ ] [Rotate Image](https://leetcode.com/problems/rotate-image/) · Medium · **B75** · Math & Geometry
# - [ ] [Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/) · Medium · **B75** · Bit Manipulation
# <!-- practice:end -->

# %%
s.finish(lookups=0, score=None, notes="")
