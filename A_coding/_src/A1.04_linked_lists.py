# %% [markdown]
# # A1.4 · Linked lists: reversal, fast/slow pointers, merge, dummy head
#
# **ID:** A1.4 · **Target:** 60 min · **Plan week:** 3 (pattern notebook, Mon); second pass in week 10 · **Part:** A
#
# **The drill.** This is a 60-minute study notebook. Linked-list questions test pointer bookkeeping, not algorithms.
# Fill in three templates (iterative reversal, middle via fast/slow, merge with a dummy head), then four warm-ups
# (10-15 min each): cycle start (Floyd), remove the n-th node from the end, reorder in place, and add two numbers.
# A tiny `ListNode` with `from_list` / `to_list` is given. `to_list` fails loudly if your code accidentally creates
# a cycle. Every function is stress-tested against a brute force that works on Python lists.
#
# **Rules.** Python standard library only. Work on the nodes. Do not convert to a Python list inside your
# solutions, except where a warm-up says so. Fill in every `raise NotImplementedError`.
#
# ## Sources
# - T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, 4th ed., MIT Press, 2022
#   (ISBN 9780262046305), Ch. 10 "Elementary Data Structures" (linked lists, sentinels, the same idea as the
#   dummy head). https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
# - Floyd's "tortoise and hare" cycle detection and the proof that resetting one pointer to the head finds the
#   cycle start: *Cycle detection*, Wikipedia. Knuth (TAOCP Vol. II, 1969, p. 7, Ex. 6-7) credits it to Floyd, and
#   the article notes this attribution may be folklore: https://en.wikipedia.org/wiki/Cycle_detection#Floyd's_tortoise_and_hare
# - Tech Interview Handbook, *Linked list* cheatsheet (dummy node, two pointers, corner cases):
#   https://www.techinterviewhandbook.org/algorithms/linked-list/ (yangshun/tech-interview-handbook @ e1d28e8, MIT).
# - NeetCode 150 problem data (practice list): https://neetcode.io/practice ,
#   https://github.com/neetcode-gh/leetcode @ d5ebb3e `.problemSiteData.json`.

# %%
import copy
import random

from prep import Session, check

s = Session("A1.4", target_min=60)
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


class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

    def __repr__(self):
        return f"ListNode({self.val})"


def from_list(vals):
    dummy = tail = ListNode()
    for v in vals:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def to_list(head, limit=10_000):
    out = []
    while head:
        out.append(head.val)
        head = head.next
        assert len(out) <= limit, "to_list: list longer than limit; did you create a cycle (forgot to cut a .next)?"
    return out


def nodes_of(head):
    out = []
    while head:
        out.append(head)
        head = head.next
    return out


def rand_vals(r, max_len=8, lo=0, hi=9, min_len=0):
    return [r.randint(lo, hi) for _ in range(r.randint(min_len, max_len))]


check("ListNode helpers round-trip", to_list(from_list([1, 2, 3])) == [1, 2, 3] and from_list([]) is None)

# %% [markdown]
# ## 1. Reversal
# **Intuition.** Walk the list once and point each node back at its predecessor. You need three names, `prev`,
# `cur` and `nxt`. Save `nxt` **before** you overwrite `cur.next`. When the loop ends, `prev` is the new head.
#
# **Signals:** "reverse" (the whole list, a sublist, groups of k), palindrome check in $O(1)$ space, reorder,
# adding numbers stored most-significant-digit first.
#
# **Template: iterative reversal, $O(n)$ time, $O(1)$ space.**

# %%
def reverse_list(head):
    ### BEGIN SOLUTION
    prev, cur = None, head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev, cur = cur, nxt
    return prev
    ### END SOLUTION


def _rev(vals):
    head = from_list(vals)
    before = {id(n) for n in nodes_of(head)}
    new = reverse_list(head)
    assert {id(n) for n in nodes_of(new)} == before, "the returned list must hold exactly the original nodes (relinked; none lost, none new)"
    return to_list(new)


run_tests(_rev, [(([1, 2, 3],), [3, 2, 1]), (([],), []), (([7],), [7]), (([1, 2],), [2, 1])], name="reverse_list")
stress(_rev, lambda vals: vals[::-1], lambda r: (rand_vals(r),), name="reverse_list")

# %% [markdown]
# ## 2. Fast/slow pointers
# **Intuition.** Move `slow` one step and `fast` two steps at a time. When `fast` reaches the end, `slow` is at the
# middle. If there is a cycle, `fast` laps `slow` and they meet inside it. A fixed gap of `n` between two pointers
# finds the n-th node from the end in one pass.
#
# **Signals:** "middle", "cycle", "k-th from the end", "intersection of two lists", palindrome, and "find the
# duplicate in an array of 1..n" (indices viewed as a linked list).
#
# **Template: middle node.** For even length, return the **second** of the two middles: `[1,2,3,4]` gives 3.

# %%
def middle_node(head):
    ### BEGIN SOLUTION
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    return slow
    ### END SOLUTION


def _mid(vals):
    m = middle_node(from_list(vals))
    return None if m is None else m.val


run_tests(_mid, [(([1, 2, 3, 4, 5],), 3), (([1, 2, 3, 4],), 3), (([1],), 1), (([],), None), (([1, 2],), 2)],
          name="middle_node")
stress(_mid, lambda vals: vals[len(vals) // 2] if vals else None,
       lambda r: (list(range(r.randint(0, 9))),), name="middle_node")

# %% [markdown]
# ### Warm-up 2.1: where does the cycle start? (Floyd; ~15 min)
# The list may end in a cycle (the last node points back to some earlier node). Return the node where the cycle
# begins, or `None` if there is no cycle. *Target:* $O(n)$ time, $O(1)$ space. *Hint:* after `slow` and `fast` meet,
# a pointer from the head and a pointer from the meeting point, both moving one step at a time, meet at the start.

# %%
def cycle_start(head):
    ### BEGIN SOLUTION
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:                       # met inside the cycle
            p = head
            while p is not slow:
                p, slow = p.next, slow.next
            return p
    return None
    ### END SOLUTION


def make_cycle(vals, pos):
    """List from vals whose tail points to node index pos (pos = -1: no cycle). Returns (head, nodes)."""
    head = from_list(vals)
    nodes = nodes_of(head)
    if pos >= 0:
        nodes[-1].next = nodes[pos]
    return head, nodes


def _cycle(vals, pos):
    head, nodes = make_cycle(vals, pos)
    got = cycle_start(head)
    return -1 if got is None else next(i for i, n in enumerate(nodes) if n is got)


def _cycle_brute(vals, pos):           # visited-set solution: O(n) extra space
    head, nodes = make_cycle(vals, pos)
    seen = set()
    while head and id(head) not in seen:
        seen.add(id(head))
        head = head.next
    return -1 if head is None else next(i for i, n in enumerate(nodes) if n is head)


def _gen_cycle(r):
    vals = rand_vals(r, 10)
    return vals, (r.randint(-1, len(vals) - 1) if vals else -1)


run_tests(_cycle, [(([3, 2, 0, -4], 1), 1), (([1, 2], 0), 0), (([1], -1), -1), (([1], 0), 0), (([], -1), -1),
                   (([1, 2, 3, 4, 5, 6], 5), 5)], name="cycle_start")
stress(_cycle, _cycle_brute, _gen_cycle, n=500, name="cycle_start")

# %% [markdown]
# ### Warm-up 2.2: remove the n-th node from the end (~10 min)
# Remove the `n`-th node from the end (`1 <= n <= length`) in **one pass** and return the new head. *Hint:* put
# a dummy node before the head so that removing the head is not a special case. Move `fast` `n` steps ahead, then
# move both until `fast.next` is `None`.

# %%
def remove_nth_from_end(head, n):
    ### BEGIN SOLUTION
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n):
        fast = fast.next
    while fast.next:
        fast, slow = fast.next, slow.next
    slow.next = slow.next.next             # slow is just before the node to delete
    return dummy.next
    ### END SOLUTION


def _gen_rm(r):
    vals = rand_vals(r, 8, min_len=1)
    return vals, r.randint(1, len(vals))


run_tests(lambda v, n: to_list(remove_nth_from_end(from_list(v), n)),
          [(([1, 2, 3, 4, 5], 2), [1, 2, 3, 5]), (([1], 1), []), (([1, 2], 2), [2]), (([1, 2], 1), [1])],
          name="remove_nth_from_end")
stress(lambda v, n: to_list(remove_nth_from_end(from_list(v), n)), lambda v, n: v[:len(v) - n] + v[len(v) - n + 1:],
       _gen_rm, name="remove_nth_from_end")

# %% [markdown]
# ## 3. Merging with a dummy head
# **Intuition.** Build the output behind a dummy node `dummy`, keeping a `tail` pointer. Then the first node is
# not a special case, and you return `dummy.next`. When one input runs out, attach the rest of the other in $O(1)$.
#
# **Signals:** "merge", "partition around x", "split odd/even", "remove all nodes with value v", any time the head
# may change.
#
# **Template: merge two sorted lists by relinking nodes (stable: ties take from `a` first).**

# %%
def merge_two(a, b):
    ### BEGIN SOLUTION
    dummy = tail = ListNode()
    while a and b:
        if a.val <= b.val:
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b                     # attach the remainder
    return dummy.next
    ### END SOLUTION


run_tests(lambda x, y: to_list(merge_two(from_list(x), from_list(y))),
          [(([1, 2, 4], [1, 3, 4]), [1, 1, 2, 3, 4, 4]), (([], []), []), (([], [0]), [0]), (([5], [1, 2, 3]), [1, 2, 3, 5])],
          name="merge_two")
stress(lambda x, y: to_list(merge_two(from_list(x), from_list(y))), lambda x, y: sorted(x + y),
       lambda r: (sorted(rand_vals(r, 6)), sorted(rand_vals(r, 6))), name="merge_two")

# %% [markdown]
# ### Warm-up 3.1: reorder in place, first/last alternating (classic; ~15 min)
# Rearrange $L_0 \to L_1 \to \dots \to L_{n-1}$ into $L_0 \to L_{n-1} \to L_1 \to L_{n-2} \to \dots$ **in place**
# (relink the nodes; return nothing). *Plan:* find the middle, cut, reverse the second half, then interleave.
# All three templates appear here.

# %%
def reorder_list(head):
    ### BEGIN SOLUTION
    if not head or not head.next:
        return
    slow, fast = head, head.next                 # slow ends at the end of the first half
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    second = reverse_list(slow.next)
    slow.next = None                             # cut, or the result has a cycle
    first = head
    while second:
        n1, n2 = first.next, second.next
        first.next, second.next = second, n1
        first, second = n1, n2
    ### END SOLUTION


def _reorder(vals):
    head = from_list(vals)
    before = {id(n) for n in nodes_of(head)}
    reorder_list(head)
    out = to_list(head)
    assert {id(n) for n in nodes_of(head)} == before, "reorder must relink the existing nodes"
    return out


def _reorder_brute(vals):
    out, i, j = [], 0, len(vals) - 1
    while i <= j:
        out.append(vals[i])
        if i != j:
            out.append(vals[j])
        i, j = i + 1, j - 1
    return out


run_tests(_reorder, [(([1, 2, 3, 4],), [1, 4, 2, 3]), (([1, 2, 3, 4, 5],), [1, 5, 2, 4, 3]), (([],), []),
                     (([1],), [1]), (([1, 2],), [1, 2])], name="reorder_list")
stress(_reorder, _reorder_brute, lambda r: (list(range(r.randint(0, 10))),), name="reorder_list")

# %% [markdown]
# ### Warm-up 3.2: add two numbers stored as digit lists (classic; ~10 min)
# Two non-negative integers are stored as linked lists of decimal digits, **least significant digit first**
# (`2 -> 4 -> 3` is 342). Return their sum as a new list in the same format. Do not convert to `int`.

# %%
def add_two_numbers(a, b):
    ### BEGIN SOLUTION
    dummy = tail = ListNode()
    carry = 0
    while a or b or carry:                       # `or carry`: 5 + 5 = 10 needs a new node
        v = carry + (a.val if a else 0) + (b.val if b else 0)
        carry, digit = divmod(v, 10)
        tail.next = ListNode(digit)
        tail = tail.next
        a, b = (a.next if a else None), (b.next if b else None)
    return dummy.next
    ### END SOLUTION


def _digits(n):
    return [int(c) for c in str(n)[::-1]]


def _add_brute(x, y):
    to_int = lambda d: int("".join(map(str, d[::-1])))
    return _digits(to_int(x) + to_int(y))


run_tests(lambda x, y: to_list(add_two_numbers(from_list(x), from_list(y))),
          [(([2, 4, 3], [5, 6, 4]), [7, 0, 8]), (([0], [0]), [0]), (([9, 9, 9, 9], [9, 9]), [8, 9, 0, 0, 1]),
           (([5], [5]), [0, 1])], name="add_two_numbers")
stress(lambda x, y: to_list(add_two_numbers(from_list(x), from_list(y))), _add_brute,
       lambda r: (_digits(r.randint(0, 10**r.randint(1, 6))), _digits(r.randint(0, 10**r.randint(1, 6)))),
       name="add_two_numbers")

# %% [markdown]
# ## Say it out loud (complexity and correctness)
# 1. Recursive reversal: write it in words. What is its space complexity, and what breaks for a 100k-node list in Python?
# 2. Why does Floyd's algorithm find the cycle **start** when one pointer restarts from the head? (Sketch the algebra.)
# 3. What does the dummy head buy you, concretely, in `remove_nth_from_end` and `merge_two`?
# 4. Merge **k** sorted lists: give the complexity of (a) merging one by one and (b) using a heap. (See A1.6.)
# 5. Reorder list: time and space? Could you do it with $O(n)$ extra space more simply? Would the interviewer accept that?
# 6. Arrays vs linked lists: when is a linked list actually the right tool? (Think `OrderedDict` / LRU cache.)

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. `rev(head)`: if `head` is `None` or a single node, return it. Otherwise `new = rev(head.next)`, then
#    `head.next.next = head; head.next = None; return new`. Time is $O(n)$, but it uses $O(n)$ **stack** space. CPython's
#    default recursion limit is 1000 (`sys.getrecursionlimit()`, https://docs.python.org/3/library/sys.html), so a
#    100k-node list raises `RecursionError`. Prefer the iterative version.
# 2. Let $\mu$ be the tail length before the cycle and $\lambda$ the cycle length. When they meet, slow has walked $\mu + x$
#    and fast $2(\mu + x)$, and the difference $\mu + x$ is a multiple of $\lambda$. So from the meeting point (offset $x$
#    in the cycle), walking $\mu$ more steps lands at offset $x + \mu \equiv 0$: the cycle start. A pointer from the head
#    also reaches the start after $\mu$ steps, so the two meet there (Wikipedia, *Cycle detection*).
# 3. The node before the target always exists, even when the target is the head. You never write
#    `if prev is None: head = ...` branches, and you always return `dummy.next`. In a merge, the first appended node
#    needs no special case.
# 4. With $N$ total nodes in $k$ lists: (a) merging one by one costs $O(kN)$ in the worst case, because the growing
#    result is re-scanned each time. (b) A min-heap of the $k$ current heads costs $O(N\log k)$ time and $O(k)$ space.
#    Pairwise divide-and-conquer merging is also $O(N\log k)$.
# 5. $O(n)$ time, $O(1)$ extra space. The simpler version copies the nodes into a Python list and relinks them with two
#    indices: $O(n)$ space. It is a fine first answer, but the follow-up will ask for $O(1)$ space.
# 6. When you need $O(1)$ insert/delete at a known node, plus a hash map from key to node. An LRU cache is a hash map
#    plus a doubly linked list: Python's `OrderedDict.move_to_end` / `popitem(last=False)`. Otherwise arrays win on
#    cache locality and $O(1)$ indexing.
# </details>

# %% [markdown]
# <!-- practice:start -->
# ## Practice list (solve these on LeetCode)
#
# NeetCode 150 problems for this topic, generated from `neetcode-gh/leetcode@d5ebb3e` `.problemSiteData.json` (https://neetcode.io/practice). **B75** = also in Blind 75. Tick them off in your session copy; the full checklist is `A_coding/A1_problem_list.md`. Grind 75 (https://www.techinterviewhandbook.org/grind75/) is an alternative ordering of a similar set.
#
# **Linked List** · plan week 9 · 11 problems
#
# - [ ] [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) · Easy · **B75**
# - [ ] [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) · Easy · **B75**
# - [ ] [Reorder List](https://leetcode.com/problems/reorder-list/) · Medium · **B75**
# - [ ] [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) · Medium · **B75**
# - [ ] [Copy List With Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/) · Medium
# - [ ] [Add Two Numbers](https://leetcode.com/problems/add-two-numbers/) · Medium
# - [ ] [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/) · Easy · **B75**
# - [ ] [Find The Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/) · Medium
# - [ ] [LRU Cache](https://leetcode.com/problems/lru-cache/) · Medium
# - [ ] [Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) · Hard · **B75**
# - [ ] [Reverse Nodes In K Group](https://leetcode.com/problems/reverse-nodes-in-k-group/) · Hard
#
# **Extras (not in NeetCode 150; close to the warm-ups above)**
#
# - [ ] [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/) · Easy
# - [ ] [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/) · Easy
# - [ ] [Design Circular Queue](https://leetcode.com/problems/design-circular-queue/) · Medium
# <!-- practice:end -->

# %%
s.finish(lookups=0, score=None, notes="")
