# %% [markdown]
# # A1.6 · Heaps (top-k, k-way merge, two-heap median) and intervals (merge, insert, sweep line)
#
# **ID:** A1.6 · **Target:** 60 min · **Plan week:** 4 (pattern notebook, Mon); second pass in week 11 · **Part:** A
#
# **The drill.** This is a 60-minute study notebook. Fill in four templates: top-k with a size-k min-heap, k-way
# merge, a streaming median from two heaps, and merging intervals. Then do three warm-ups (10-15 min each): a
# max-heap simulation, inserting into a sorted interval list, and the minimum number of meeting rooms with a sweep
# line. All tests compare against sorting-based brute forces on random inputs.
#
# **Rules.** Python standard library only. Use `heapq` (a min-heap on a plain list). For a max-heap, push
# negated keys. `heapq.heappush_max` and the other max-heap functions only exist from Python 3.14, and this repo's venv
# and most interview pads run older versions. Fill in every `raise NotImplementedError`.
#
# ## Sources
# - Python docs, `heapq` (min-heap invariant `heap[k] <= heap[2k+1], heap[2k+2]`; `heapify` is linear time;
#   `nlargest`/`nsmallest` are best for small n; `merge` assumes sorted inputs; tuple tie-breaking with an entry count;
#   max-heap functions added in 3.14): https://docs.python.org/3/library/heapq.html
# - T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, 4th ed., MIT Press, 2022
#   (ISBN 9780262046305), Ch. 6 "Heapsort" (binary heaps, priority queues, `BUILD-MAX-HEAP` in $O(n)$).
#   https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
# - Tech Interview Handbook, *Heap* and *Interval* cheatsheets (sort by start; ask whether touching counts; strict overlap test
#   `a[0] < b[1] and b[0] < a[1]`), https://www.techinterviewhandbook.org/algorithms/heap/ ,
#   https://www.techinterviewhandbook.org/algorithms/interval/ (yangshun/tech-interview-handbook @ e1d28e8, MIT).
# - NeetCode 150 problem data (practice list): https://neetcode.io/practice ,
#   https://github.com/neetcode-gh/leetcode @ d5ebb3e `.problemSiteData.json`.

# %%
import copy
import heapq
import itertools
import random
import statistics

from prep import Session, check

s = Session("A1.6", target_min=60)
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
    obj, out = cls(), []
    for name, *args in ops:
        out.append(getattr(obj, name)(*args))
    return out


def rand_list(rng, lo=0, hi=9, max_len=9, min_len=0):
    return [rng.randint(lo, hi) for _ in range(rng.randint(min_len, max_len))]

# %% [markdown]
# ## 1. Heaps: the k best so far
# **Intuition.** A binary heap gives the min in $O(1)$ and push/pop in $O(\log n)$ (CLRS Ch. 6). To keep the **k
# largest** items of a stream, use a **min**-heap of size k. Its root is the weakest of the current top k, so each
# new item either beats it (`heapreplace`) or is discarded. That is $O(n\log k)$ time and $O(k)$ memory, better
# than sorting when $k \ll n$ or when the data is a stream.
#
# **Signals:** "k largest/smallest/closest/most frequent", "k-th largest", "repeatedly take the best/cheapest"
# (simulation, greedy scheduling), "merge k sorted", "median of a stream", Dijkstra (A1.7).
#
# **Template A: top-k largest, returned in descending order.**

# %%
def top_k_largest(nums, k):
    """The k largest values (0 <= k <= len(nums)), largest first. O(n log k)."""
    ### BEGIN SOLUTION
    if k == 0:
        return []
    heap = []
    for x in nums:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)            # pop the weakest, push x
    return sorted(heap, reverse=True)
    ### END SOLUTION


def _gen_topk(r):
    a = rand_list(r)
    return a, r.randint(0, len(a))


run_tests(top_k_largest, [(([3, 2, 1, 5, 6, 4], 2), [6, 5]), (([3, 3, 3], 2), [3, 3]), (([], 0), []), (([1], 1), [1])])
stress(top_k_largest, lambda a, k: sorted(a, reverse=True)[:k], _gen_topk)

# %% [markdown]
# ### Warm-up 1.1: smash the two heaviest (classic "last stone"; ~8 min)
# You have stones with positive weights. While at least two remain, take the two heaviest, $x \le y$. If they are
# equal, both disappear. Otherwise one stone of weight $y - x$ remains. Return the last weight, or 0 if none is
# left. Use `heapq` as a **max**-heap via negation.

# %%
def last_stone(weights):
    ### BEGIN SOLUTION
    heap = [-w for w in weights]
    heapq.heapify(heap)                             # O(n)
    while len(heap) > 1:
        y, x = -heapq.heappop(heap), -heapq.heappop(heap)
        if y != x:
            heapq.heappush(heap, -(y - x))
    return -heap[0] if heap else 0
    ### END SOLUTION


def last_stone_brute(weights):
    w = sorted(weights)
    while len(w) > 1:
        y, x = w.pop(), w.pop()
        if y != x:
            w.append(y - x)
            w.sort()
    return w[0] if w else 0


run_tests(last_stone, [(([2, 7, 4, 1, 8, 1],), 1), (([1],), 1), (([3, 3],), 0), (([],), 0), (([10, 4],), 6)])
stress(last_stone, last_stone_brute, lambda r: (rand_list(r, 1, 9),))

# %% [markdown]
# **Template B: k-way merge.** Seed the heap with the first element of each list as `(value, list_index,
# elem_index)`. Pop the smallest and push its successor from the same list. With the indices in the tuple, ties never
# fall through to comparing un-orderable payloads (`heapq` docs). That is $O(N\log k)$ for $N$ total elements.

# %%
def k_way_merge(lists):
    ### BEGIN SOLUTION
    heap = [(lst[0], i, 0) for i, lst in enumerate(lists) if lst]
    heapq.heapify(heap)
    out = []
    while heap:
        v, i, j = heapq.heappop(heap)
        out.append(v)
        if j + 1 < len(lists[i]):
            heapq.heappush(heap, (lists[i][j + 1], i, j + 1))
    return out
    ### END SOLUTION


run_tests(k_way_merge, [(([[1, 4, 5], [1, 3, 4], [2, 6]],), [1, 1, 2, 3, 4, 4, 5, 6]), (([],), []), (([[], []],), []),
                        (([[], [1]],), [1])])
stress(k_way_merge, lambda lists: sorted(itertools.chain(*lists)),
       lambda r: ([sorted(rand_list(r, max_len=5)) for _ in range(r.randint(0, 5))],))
check("k_way_merge agrees with heapq.merge", k_way_merge([[1, 3], [2, 2, 9], [0]]) == list(heapq.merge([1, 3], [2, 2, 9], [0])))

# %% [markdown]
# **Template C: streaming median with two heaps.** Keep a max-heap `lo` (stored negated) with the smaller half
# and a min-heap `hi` with the larger half. Invariants: every element of `lo` is $\le$ every element of `hi`, and
# `len(lo) - len(hi)` is 0 or 1. `add` is $O(\log n)$ and `median` is $O(1)$. For an even count, return the mean of
# the two middle values.

# %%
class MedianFinder:
    def __init__(self):
        self.lo, self.hi = [], []           # lo: max-heap via negation; hi: min-heap

    def add(self, x):
        ### BEGIN SOLUTION
        heapq.heappush(self.lo, -x)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))     # move lo's max to hi: keeps the ordering invariant
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))  # rebalance sizes
        ### END SOLUTION

    def median(self):
        ### BEGIN SOLUTION
        if len(self.lo) > len(self.hi):
            return float(-self.lo[0])
        return (-self.lo[0] + self.hi[0]) / 2
        ### END SOLUTION


class MedianBrute:
    def __init__(self):
        self.a = []

    def add(self, x):
        self.a.append(x)

    def median(self):
        return float(statistics.median(self.a))


def _gen_median_ops(r):
    ops = [("add", r.randint(-5, 5))]
    for _ in range(r.randint(0, 14)):
        ops.append(("add", r.randint(-5, 5)) if r.random() < 0.6 else ("median",))
    return (ops,)


run_tests(lambda ops: replay(MedianFinder, ops),
          [(([("add", 1), ("add", 2), ("median",), ("add", 3), ("median",), ("add", -10), ("median",)],),
            [None, None, 1.5, None, 2.0, None, 1.5])], name="MedianFinder")
stress(lambda ops: replay(MedianFinder, ops), lambda ops: replay(MedianBrute, ops), _gen_median_ops, n=500,
       name="MedianFinder")

# %% [markdown]
# ## 2. Intervals: sort by start, then sweep
# **Intuition.** After sorting by start, overlaps only happen between **neighbours in the sweep**. An interval
# overlaps the current merged block iff it starts before (or, for closed intervals, at) the block's end. More
# generally, turn intervals into **events** (+1 at start, -1 at end), sort them, and keep a running count. The
# maximum count is the peak overlap.
#
# **Signals:** "meetings/bookings/ranges", "overlap", "merge", "free time", "minimum rooms/arrows/removals to make
# non-overlapping" (greedy by end time), "how many active at time t", calendar problems.
#
# **Template D: merge closed intervals** `[start, end]`, where touching intervals (`[1, 2]` and `[2, 3]`) merge.
# Return them sorted by start.

# %%
def merge_intervals(iv):
    ### BEGIN SOLUTION
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:             # overlaps (or touches) the current block
            out[-1][1] = max(out[-1][1], b)     # max: the new interval may be nested inside
        else:
            out.append([a, b])
    return out
    ### END SOLUTION


def merge_brute(iv):
    """Mark covered half-unit points on a doubled grid, then read off the maximal runs."""
    if not iv:
        return []
    cov = set()
    for a, b in iv:
        cov.update(range(2 * a, 2 * b + 1))
    pts, out = sorted(cov), []
    start = prev = pts[0]
    for p in pts[1:] + [None]:
        if p is None or p != prev + 1:
            out.append([start // 2, prev // 2])
            if p is not None:
                start = p
        if p is not None:
            prev = p
    return out


def rand_intervals(r, n_max=6, t_max=12):
    out = []
    for _ in range(r.randint(0, n_max)):
        a = r.randint(0, t_max)
        out.append([a, a + r.randint(0, 4)])
    return out


run_tests(merge_intervals, [(([[1, 3], [2, 6], [8, 10], [15, 18]],), [[1, 6], [8, 10], [15, 18]]),
                            (([[1, 4], [4, 5]],), [[1, 5]]), (([[1, 10], [2, 3]],), [[1, 10]]), (([],), []),
                            (([[5, 5], [1, 2]],), [[1, 2], [5, 5]])])
stress(merge_intervals, merge_brute, lambda r: (rand_intervals(r),), n=1000)

# %% [markdown]
# ### Warm-up 2.1: insert into a sorted, non-overlapping list (~12 min)
# `iv` is sorted by start and has no overlaps (closed intervals). Insert `new` and merge as needed, in $O(n)$:
# copy everything that ends before `new` starts, absorb everything that overlaps it, then copy the rest.
# Calling `merge_intervals(iv + [new])` gives the same result but costs $O(n\log n)$.

# %%
def insert_interval(iv, new):
    ### BEGIN SOLUTION
    a, b = new
    out, i, n = [], 0, len(iv)
    while i < n and iv[i][1] < a:               # strictly before
        out.append(iv[i])
        i += 1
    while i < n and iv[i][0] <= b:              # overlaps or touches
        a, b = min(a, iv[i][0]), max(b, iv[i][1])
        i += 1
    out.append([a, b])
    out.extend(iv[i:])
    return out
    ### END SOLUTION


def _gen_insert(r):
    base = merge_brute(rand_intervals(r))
    a = r.randint(0, 14)
    return base, [a, a + r.randint(0, 5)]


run_tests(insert_interval, [(([[1, 3], [6, 9]], [2, 5]), [[1, 5], [6, 9]]),
                            (([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]), [[1, 2], [3, 10], [12, 16]]),
                            (([], [5, 7]), [[5, 7]]), (([[1, 5]], [6, 8]), [[1, 5], [6, 8]]),
                            (([[3, 5]], [1, 2]), [[1, 2], [3, 5]]), (([[1, 5]], [5, 7]), [[1, 7]])])
stress(insert_interval, lambda iv, new: merge_brute(iv + [new]), _gen_insert, n=1000)

# %% [markdown]
# ### Warm-up 2.2: minimum number of meeting rooms (sweep line; ~12 min)
# Meetings are **half-open** `[start, end)`, so one that ends at 10 frees its room for one that starts at 10. Return the
# minimum number of rooms needed, which is the maximum number of meetings running at the same time. Use the sweep: sort
# events, and at equal times process ends **before** starts. Alternatively, sort by start and keep a min-heap of end times.

# %%
def min_meeting_rooms(iv):
    ### BEGIN SOLUTION
    events = sorted([(a, 1) for a, b in iv] + [(b, -1) for a, b in iv])   # (t, -1) sorts before (t, +1)
    best = cur = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best
    ### END SOLUTION


def rooms_brute(iv):
    times = range(0, 20)
    return max([sum(a <= t < b for a, b in iv) for t in times], default=0)


run_tests(min_meeting_rooms, [(([[0, 30], [5, 10], [15, 20]],), 2), (([[7, 10], [2, 4]],), 1), (([],), 0),
                              (([[1, 5], [5, 10]],), 1),                  # back-to-back: one room
                              (([[1, 5], [1, 5], [1, 5]],), 3), (([[3, 3]],), 0)])   # empty meeting needs no room
stress(min_meeting_rooms, rooms_brute, lambda r: (rand_intervals(r, 7, 12),), n=1000)

# %% [markdown]
# ## Say it out loud (complexity and correctness)
# 1. Top-k: compare a min-heap of size k, sorting, `heapq.nlargest` and quickselect, in time and space.
#    When would you pick each?
# 2. Why is `heapify` $O(n)$ and not $O(n\log n)$?
# 3. k-way merge: complexity in $N$ and $k$? Why put the list index in the tuple?
# 4. Two-heap median: state both invariants. What is the cost per `add` and per `median`? What if you also need deletions
#    (sliding-window median)?
# 5. Merge intervals: why is sorting by start enough? Why `max(out[-1][1], b)` rather than `b`?
# 6. Meeting rooms: why must ends sort before starts at equal times for half-open intervals? What changes for closed intervals?
# 7. Non-overlapping intervals (fewest removals): why does the greedy sort **by end** work?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. A size-k heap costs $O(n\log k)$ time and $O(k)$ space, and works on a stream. Sorting costs $O(n\log n)$ time and $O(n)$
#    space, and is simplest when $k$ is close to $n$. `heapq.nlargest(k, a)` is the library version of the size-k heap. The
#    docs recommend it for small $k$, `sorted` for large $k$, and `max` for $k=1$. Quickselect is expected $O(n)$ but
#    worst case $O(n^2)$, and the k items come out unsorted. It works in place, offline.
# 2. Sift-down from the bottom up: a node at height $h$ costs $O(h)$, and there are about $n/2^{h+1}$ nodes at height
#    $h$. The sum $\sum_h h\,n/2^{h+1} = O(n)$ (CLRS Ch. 6, `BUILD-MAX-HEAP`). The `heapq` docs also state linear time.
# 3. $O(N\log k)$ time, $O(k)$ heap space. Equal values would otherwise make Python compare the next tuple field, which
#    might be a `ListNode` (no `<`, so a `TypeError`). A unique index breaks ties deterministically, the trick the `heapq`
#    docs describe with an entry count.
# 4. (a) $\max(lo) \le \min(hi)$. (b) $|lo| = |hi|$ or $|lo| = |hi| + 1$. `add` costs $O(\log n)$ (a constant number of
#    pushes and pops), and `median` is $O(1)$. Deletions are not supported by `heapq`. Use lazy deletion (a counter of
#    to-be-deleted values, cleaned when they reach a top, tracking the "real" sizes) or a balanced BST / sorted container:
#    $O(\log n)$ amortized per step, so $O(n\log k)$ for a sliding window.
# 5. After sorting, an interval can only overlap the current block or later ones. It cannot overlap an earlier block,
#    because that block ended before the current block started. `max` handles a nested interval like `[2, 3]` inside `[1, 10]`.
# 6. With half-open intervals, a meeting ending at $t$ and one starting at $t$ do not overlap. Processing the $-1$
#    first keeps the count from peaking falsely. With closed intervals they do overlap, so starts must come first
#    (sort key `(t, -delta)`).
# 7. Exchange argument: among intervals compatible with the ones already kept, the one that ends first leaves the most
#    room for the rest. Any optimal solution can swap its first interval for this one without losing anything (CLRS
#    Ch. 15, activity selection).
# </details>

# %% [markdown]
# <!-- practice:start -->
# ## Practice list (solve these on LeetCode)
#
# NeetCode 150 problems for this topic, generated from `neetcode-gh/leetcode@d5ebb3e` `.problemSiteData.json` (https://neetcode.io/practice). **B75** = also in Blind 75. Tick them off in your session copy; the full checklist is `A_coding/A1_problem_list.md`. Grind 75 (https://www.techinterviewhandbook.org/grind75/) is an alternative ordering of a similar set.
#
# **Heap / Priority Queue** · plan week 11 · 7 problems
#
# - [ ] [Kth Largest Element In a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/) · Easy
# - [ ] [Last Stone Weight](https://leetcode.com/problems/last-stone-weight/) · Easy
# - [ ] [K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/) · Medium
# - [ ] [Kth Largest Element In An Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) · Medium
# - [ ] [Task Scheduler](https://leetcode.com/problems/task-scheduler/) · Medium
# - [ ] [Design Twitter](https://leetcode.com/problems/design-twitter/) · Medium
# - [ ] [Find Median From Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) · Hard · **B75**
#
# **Intervals** · plan week 11 · 6 problems
#
# - [ ] [Insert Interval](https://leetcode.com/problems/insert-interval/) · Medium · **B75**
# - [ ] [Merge Intervals](https://leetcode.com/problems/merge-intervals/) · Medium · **B75**
# - [ ] [Non Overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/) · Medium · **B75**
# - [ ] [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/) · Easy · **B75**
# - [ ] [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/) · Medium · **B75**
# - [ ] [Minimum Interval to Include Each Query](https://leetcode.com/problems/minimum-interval-to-include-each-query/) · Hard
# <!-- practice:end -->

# %%
s.finish(lookups=0, score=None, notes="")
