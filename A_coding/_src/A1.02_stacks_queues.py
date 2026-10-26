# %% [markdown]
# # A1.2 · Stacks, monotonic stacks, queues and deques
#
# **ID:** A1.2 · **Target:** 60 min · **Plan week:** 2 (pattern notebook, Mon); second pass in week 8 · **Part:** A
#
# **The drill.** This is a 60-minute study notebook. You fill in three canonical templates: next-greater element with a
# monotonic stack, sliding-window maximum with a monotonic deque, and a queue built from two stacks. Then you do four
# warm-ups (10-20 min each). Every function is tested on hand-written edge cases and **stress-tested** against a
# trusted brute force. Data-structure classes are tested by replaying random operation sequences against a slow
# reference class. Finish with the complexity questions and the practice list.
#
# **Rules.** Python standard library only. Fill in every `raise NotImplementedError`; do not edit the brute forces or tests.
#
# ## Sources
# - T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, 4th ed., MIT Press, 2022
#   (ISBN 9780262046305): Ch. 10 "Elementary Data Structures" (stacks, queues) and Ch. 16 "Amortized Analysis"
#   (aggregate / accounting arguments used for the monotonic stack and the two-stack queue).
#   https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
# - Python docs, `collections.deque` ("approximately the same O(1) performance in either direction"; indexed
#   access is $O(n)$ in the middle): https://docs.python.org/3/library/collections.html#collections.deque ;
#   `list.pop(0)` is $O(n)$: https://wiki.python.org/moin/TimeComplexity
# - Tech Interview Handbook, *Stack* and *Queue* cheatsheets, https://www.techinterviewhandbook.org/algorithms/stack/ ,
#   https://www.techinterviewhandbook.org/algorithms/queue/ (yangshun/tech-interview-handbook @ e1d28e8, MIT).
# - Sliding-window maximum (the monotonic-deque problem): https://leetcode.com/problems/sliding-window-maximum/
# - NeetCode 150 problem data (practice list): https://neetcode.io/practice ,
#   https://github.com/neetcode-gh/leetcode @ d5ebb3e `.problemSiteData.json`.

# %%
import copy
import random
from collections import deque

from prep import Session, check

s = Session("A1.2", target_min=60)
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


def replay(cls, ops):
    """Run [(method, *args), ...] on a fresh cls() and return the list of return values."""
    obj, out = cls(), []
    for name, *args in ops:
        out.append(getattr(obj, name)(*args))
    return out


def rand_list(rng, lo=0, hi=5, max_len=9, min_len=0):
    return [rng.randint(lo, hi) for _ in range(rng.randint(min_len, max_len))]

# %% [markdown]
# ## 1. Stack basics: matching and nesting
# **Intuition.** A stack remembers the "still open" items in LIFO order. That is exactly what nesting needs: the
# most recent unmatched opener must be closed first.
#
# **Signals:** brackets and nested structure, "undo", evaluating expressions (RPN, calculators), simulating
# recursion iteratively (DFS), "remove adjacent duplicates", collisions (asteroids).
#
# ### Warm-up 1.1: balanced brackets (classic; ~8 min)
# A string contains only `()[]{}`. Return whether every closer matches the most recent unmatched opener of the
# same type, and nothing is left open at the end.

# %%
def valid_brackets(s_):
    ### BEGIN SOLUTION
    pair = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s_:
        if ch in pair:
            if not stack or stack.pop() != pair[ch]:
                return False
        else:
            stack.append(ch)
    return not stack                     # leftover openers -> invalid
    ### END SOLUTION


def valid_brackets_brute(s_):
    prev = None
    while prev != s_:
        prev, s_ = s_, s_.replace("()", "").replace("[]", "").replace("{}", "")
    return s_ == ""


run_tests(valid_brackets, [
    (("()[]{}",), True), (("([{}])",), True), (("(]",), False), (("([)]",), False),
    (("",), True), (("((",), False), ((")",), False), (("]()",), False),
])
stress(valid_brackets, valid_brackets_brute,
       lambda r: ("".join(r.choice("()[]") for _ in range(r.randint(0, 8))),), n=1000)

# %% [markdown]
# ## 2. Monotonic stack: "next greater / smaller"
# **Intuition.** Scan left to right and keep a stack of indices that are **still waiting** for a greater
# element. Their values are non-increasing from bottom to top. When `x` arrives, it is the answer for every waiting
# index with a smaller value. Pop those and record `x`. Each index is pushed once and popped at most once, so the
# scan is $O(n)$ (aggregate analysis, CLRS Ch. 16).
#
# **Signals:** "next/previous greater or smaller element", "days until ...", "span", "how far can I see",
# largest rectangle / trapping water, "remove k digits to make the number smallest".
#
# **Template: next strictly greater value to the right (or -1).**

# %%
def next_greater(a):
    ### BEGIN SOLUTION
    res = [-1] * len(a)
    stack = []                                   # indices; a[stack] is non-increasing
    for i, x in enumerate(a):
        while stack and a[stack[-1]] < x:        # strictly smaller -> x is their next greater
            res[stack.pop()] = x
        stack.append(i)
    return res
    ### END SOLUTION


def next_greater_brute(a):
    return [next((y for y in a[i + 1:] if y > x), -1) for i, x in enumerate(a)]


run_tests(next_greater, [
    (([2, 1, 2, 4, 3],), [4, 2, 4, -1, -1]),
    (([2, 2, 3],), [3, 3, -1]),                  # equal is not greater
    (([5, 4, 3],), [-1, -1, -1]),
    (([],), []),
])
stress(next_greater, next_greater_brute, lambda r: (rand_list(r),))

# %% [markdown]
# ### Warm-up 2.1: days until a warmer day (classic; ~10 min)
# You get daily temperatures. For each day, return how many days you must wait for a strictly warmer
# temperature, or 0 if it never comes. This is the template with distances instead of values.

# %%
def days_until_warmer(temps):
    ### BEGIN SOLUTION
    res = [0] * len(temps)
    stack = []
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
    ### END SOLUTION


def days_until_warmer_brute(temps):
    return [next((j - i for j in range(i + 1, len(temps)) if temps[j] > t), 0) for i, t in enumerate(temps)]


run_tests(days_until_warmer, [
    (([73, 74, 75, 71, 69, 72, 76, 73],), [1, 1, 4, 2, 1, 1, 0, 0]),
    (([30, 30, 31],), [2, 1, 0]),
    (([1],), [0]),
    (([],), []),
])
stress(days_until_warmer, days_until_warmer_brute, lambda r: (rand_list(r, 0, 4),))

# %% [markdown]
# ### Warm-up 2.2: largest rectangle under a histogram (classic hard; ~20 min)
# Bars of width 1 have the given non-negative heights. Return the area of the largest axis-aligned rectangle that
# fits under the bars. *Hint:* each bar extends left and right until the first strictly shorter bar. Keep a stack of
# increasing heights. When a bar is popped, the current index is its right boundary and the new stack top is its left
# boundary. Remember to flush the stack at the end (a sentinel height 0 does it).

# %%
def largest_rectangle(heights):
    ### BEGIN SOLUTION
    best = 0
    stack = []                                         # indices with increasing heights
    for i, h in enumerate(heights + [0]):              # sentinel 0 flushes the stack
        while stack and heights[stack[-1]] >= h:
            top = stack.pop()
            left = stack[-1] + 1 if stack else 0
            best = max(best, heights[top] * (i - left))
        stack.append(i)
    return best
    ### END SOLUTION


def largest_rectangle_brute(heights):
    best = 0
    for i in range(len(heights)):
        m = heights[i]
        for j in range(i, len(heights)):
            m = min(m, heights[j])
            best = max(best, m * (j - i + 1))
    return best


run_tests(largest_rectangle, [
    (([2, 1, 5, 6, 2, 3],), 10),
    (([1, 2, 3],), 4),             # increasing: only the final flush finds the answer
    (([2, 2, 2],), 6),
    (([0],), 0),
    (([],), 0),
    (([4],), 4),
])
stress(largest_rectangle, largest_rectangle_brute, lambda r: (rand_list(r, 0, 5),))

# %% [markdown]
# ## 3. Stack with an extra invariant
# ### Warm-up 3.1: min-stack (classic; ~10 min)
# Design a stack with `push(x)`, `pop()` (returns the removed value), `top()` and `get_min()`, each $O(1)$. Calls
# are only made when they are valid. *Hint:* store, next to every element, the minimum of the stack up to and
# including that element.

# %%
class MinStack:
    def __init__(self):
        ### BEGIN SOLUTION
        self.st = []                       # (value, min of stack up to here)
        ### END SOLUTION

    def push(self, x):
        ### BEGIN SOLUTION
        m = x if not self.st else min(x, self.st[-1][1])
        self.st.append((x, m))
        ### END SOLUTION

    def pop(self):
        ### BEGIN SOLUTION
        return self.st.pop()[0]
        ### END SOLUTION

    def top(self):
        ### BEGIN SOLUTION
        return self.st[-1][0]
        ### END SOLUTION

    def get_min(self):
        ### BEGIN SOLUTION
        return self.st[-1][1]
        ### END SOLUTION


class MinStackBrute:
    def __init__(self):
        self.a = []

    def push(self, x):
        self.a.append(x)

    def pop(self):
        return self.a.pop()

    def top(self):
        return self.a[-1]

    def get_min(self):
        return min(self.a)


def gen_stack_ops(r):
    ops, size = [], 0
    for _ in range(r.randint(1, 12)):
        if size == 0 or r.random() < 0.45:
            ops.append(("push", r.randint(-3, 3)))
            size += 1
        else:
            op = r.choice(["pop", "top", "get_min"])
            ops.append((op,))
            size -= op == "pop"
    return (ops,)


ops_min = [("push", 2), ("push", 1), ("push", 1), ("get_min",), ("pop",), ("get_min",), ("pop",), ("get_min",), ("top",)]
run_tests(lambda ops: replay(MinStack, ops), [((ops_min,), [None, None, None, 1, 1, 1, 1, 2, 2])], name="MinStack")
stress(lambda ops: replay(MinStack, ops), lambda ops: replay(MinStackBrute, ops), gen_stack_ops, n=500,
       name="MinStack")

# %% [markdown]
# ## 4. Queues and deques
# **Intuition.** A FIFO queue is the frontier for BFS (A1.7). In Python use `collections.deque`: `popleft` is $O(1)$,
# while `list.pop(0)` shifts every element and costs $O(n)$. A deque is also the basis of the **monotonic deque**, which
# keeps a window's candidates in decreasing order.
#
# **Signals:** BFS / level order, "process in arrival order", rate limiters and "last k seconds", sliding-window
# max/min, "queue from stacks" design questions.
#
# **Template A: queue from two stacks (amortized $O(1)$).** `push` goes onto `inbox`. `pop`/`peek` read from `outbox`.
# Only when `outbox` is empty, move **all** of `inbox` over, which reverses the order. Each element is moved at most once,
# so the cost is amortized $O(1)$ per operation (accounting method: charge 2 extra units when an element is pushed).

# %%
class TwoStackQueue:
    def __init__(self):
        self.inbox, self.outbox = [], []

    def push(self, x):
        ### BEGIN SOLUTION
        self.inbox.append(x)
        ### END SOLUTION

    def _shift(self):
        ### BEGIN SOLUTION
        if not self.outbox:                # only when empty, or the order breaks
            while self.inbox:
                self.outbox.append(self.inbox.pop())
        ### END SOLUTION

    def pop(self):
        self._shift()
        return self.outbox.pop()

    def peek(self):
        self._shift()
        return self.outbox[-1]

    def empty(self):
        return not self.inbox and not self.outbox


class QueueBrute:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)

    def pop(self):
        return self.q.popleft()

    def peek(self):
        return self.q[0]

    def empty(self):
        return not self.q


def gen_queue_ops(r):
    ops, size = [], 0
    for _ in range(r.randint(1, 14)):
        if size == 0 or r.random() < 0.5:
            ops.append(("push", r.randint(0, 9)))
            size += 1
        else:
            op = r.choice(["pop", "peek", "empty"])
            ops.append((op,))
            size -= op == "pop"
    return (ops,)


run_tests(lambda ops: replay(TwoStackQueue, ops),
          [(([("push", 1), ("push", 2), ("pop",), ("push", 3), ("pop",), ("peek",), ("empty",), ("pop",), ("empty",)],),
            [None, None, 1, None, 2, 3, False, 3, True])], name="TwoStackQueue")
stress(lambda ops: replay(TwoStackQueue, ops), lambda ops: replay(QueueBrute, ops), gen_queue_ops, n=500,
       name="TwoStackQueue")

# %% [markdown]
# **Template B: sliding-window maximum with a monotonic deque.** Keep indices whose values are **strictly
# decreasing** from front to back. A new element evicts every smaller-or-equal element from the back: those can never
# be a maximum again while the new one is in the window. Drop the front when it leaves the window. The front is the window max.

# %%
def sliding_window_max(a, k):
    """Max of every length-k window, 1 <= k <= len(a) (or [] if a is empty). O(n)."""
    ### BEGIN SOLUTION
    dq, res = deque(), []
    for i, x in enumerate(a):
        while dq and a[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:                 # front index fell out of the window
            dq.popleft()
        if i >= k - 1:
            res.append(a[dq[0]])
    return res
    ### END SOLUTION


def sliding_window_max_brute(a, k):
    return [max(a[i:i + k]) for i in range(len(a) - k + 1)]


def _gen_sw(r):
    a = rand_list(r, 0, 6, 10, 1)
    return a, r.randint(1, len(a))


run_tests(sliding_window_max, [
    (([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7]),
    (([4, 2], 1), [4, 2]),
    (([9, 8, 7, 6], 2), [9, 8, 7]),        # decreasing: the front must expire
    (([1, 2], 2), [2]),
])
stress(sliding_window_max, sliding_window_max_brute, _gen_sw)

# %% [markdown]
# ## Say it out loud (complexity and correctness)
# 1. The monotonic stack loop has a `while` inside a `for`. Prove it is $O(n)$.
# 2. Next-greater: what changes in the code for next *smaller*, *previous* greater, or a *circular* array?
# 3. Largest rectangle: why is the popped bar's width `i - stack[-1] - 1`? What does the sentinel do?
# 4. Two-stack queue: worst case of a single `pop`? Amortized cost? Give the accounting argument.
# 5. Sliding-window max: why is evicting elements $\le x$ from the back safe? What is the space complexity?
# 6. Why not use a Python `list` as a queue? When is `deque` indexing slow?
# 7. A heap also gives sliding-window max. What does it cost, and how do you deal with expired elements?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. Each index is pushed exactly once and popped at most once, so the inner `while` runs at most $n$ times **in
#    total** over the whole scan. Total work is $O(n)$ (aggregate analysis, CLRS Ch. 16). Space is $O(n)$ for an
#    increasing input, where nothing is ever popped.
# 2. Next smaller: flip the comparison (`a[top] > x`). Previous greater: scan right-to-left, or read the stack top
#    *after* popping when you push `i`. Circular: iterate `i` over `range(2n)` with `a[i % n]` and push only in the
#    first pass.
# 3. Bars between the new stack top and `i` are all $\ge$ the popped height. The new top is the first strictly
#    shorter bar on the left (or there is none, so the left edge is 0), and `i` is the first shorter-or-equal bar on
#    the right. The width is `i - left`. Equal heights get popped early, but the last bar of an equal run sees the
#    full width, so the maximum is still found. The sentinel 0 is shorter than everything, so it flushes the stack at the end.
# 4. Worst case for one `pop` is $O(n)$, when it moves everything. Amortized it is $O(1)$. Charge each `push` 3 units:
#    one for the push itself, one prepaid for moving the element to `outbox` later, and one for popping it from
#    `outbox`. Each element is moved at most once, so the credit never goes negative.
# 5. If `j < i` and `a[j] <= a[i]`, then `j` leaves the window before `i`, and while both are in the window `a[i]`
#    is at least as large. So `j` can never be the answer again. Each index enters and leaves the deque once: $O(n)$
#    time, $O(k)$ space.
# 6. `list.pop(0)` / `insert(0, x)` shift all elements: $O(n)$ (Python wiki TimeComplexity). `deque` has $O(1)$ at both
#    ends, but indexing `dq[i]` slows to $O(n)$ in the middle (collections docs).
# 7. Push `(-value, index)` into a heap and lazily pop the top while its index is outside the window. That is
#    $O(n\log n)$ time and $O(n)$ space in the worst case. The deque is $O(n)$, but the heap version is easier to
#    get right under time pressure.
# </details>

# %% [markdown]
# <!-- practice:start -->
# ## Practice list (solve these on LeetCode)
#
# NeetCode 150 problems for this topic, generated from `neetcode-gh/leetcode@d5ebb3e` `.problemSiteData.json` (https://neetcode.io/practice). **B75** = also in Blind 75. Tick them off in your session copy; the full checklist is `A_coding/A1_problem_list.md`. Grind 75 (https://www.techinterviewhandbook.org/grind75/) is an alternative ordering of a similar set.
#
# **Stack** · plan week 8 · 7 problems
#
# - [ ] [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) · Easy · **B75**
# - [ ] [Min Stack](https://leetcode.com/problems/min-stack/) · Medium
# - [ ] [Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/) · Medium
# - [ ] [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/) · Medium
# - [ ] [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/) · Medium
# - [ ] [Car Fleet](https://leetcode.com/problems/car-fleet/) · Medium
# - [ ] [Largest Rectangle In Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) · Hard
#
# **Extras (not in NeetCode 150; close to the warm-ups above)**
#
# - [ ] [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/) · Easy
# - [ ] [Implement Stack Using Queues](https://leetcode.com/problems/implement-stack-using-queues/) · Easy
# <!-- practice:end -->

# %%
s.finish(lookups=0, score=None, notes="")
