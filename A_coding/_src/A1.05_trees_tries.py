# %% [markdown]
# # A1.5 · Trees and tries: DFS, BFS, BSTs, LCA, serialization, prefix trees
#
# **ID:** A1.5 · **Target:** 60 min · **Plan week:** 3 (pattern notebook, Wed); second pass in week 10 (trees) and 11 (tries) · **Part:** A
#
# **The drill.** This is a 60-minute study notebook. Fill in the traversal templates: inorder DFS (recursive and
# iterative) and BFS level order. Then do BST search and validation, lowest common ancestor, serialize/deserialize,
# a trie, and the k-th smallest element of a BST (10-15 min each). Trees are given in LeetCode's level-order list
# format (`[3, 9, 20, None, None, 15, 7]`) and converted with the given `from_level` / `to_level`. All tests use
# random trees and compare against a brute force. One test uses a 3000-deep chain, which a recursive traversal
# cannot handle in CPython.
#
# **Rules.** Python standard library only. Fill in every `raise NotImplementedError`.
#
# ## Sources
# - T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, 4th ed., MIT Press, 2022
#   (ISBN 9780262046305): Ch. 12 "Binary Search Trees" (BST property, inorder walk, search) and Ch. 20
#   "Elementary Graph Algorithms" (BFS/DFS). https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
# - E. Fredkin, *Trie Memory*, Communications of the ACM 3(9), 1960, DOI 10.1145/367390.367400.
# - Python docs: `sys.getrecursionlimit` (CPython default 1000): https://docs.python.org/3/library/sys.html#sys.getrecursionlimit ;
#   `collections.deque`: https://docs.python.org/3/library/collections.html#collections.deque
# - Tech Interview Handbook, *Tree* and *Trie* cheatsheets, https://www.techinterviewhandbook.org/algorithms/tree/ ,
#   https://www.techinterviewhandbook.org/algorithms/trie/ (yangshun/tech-interview-handbook @ e1d28e8, MIT).
# - LeetCode's level-order tree encoding (used by the helpers): e.g. https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
# - NeetCode 150 problem data (practice list): https://neetcode.io/practice ,
#   https://github.com/neetcode-gh/leetcode @ d5ebb3e `.problemSiteData.json`.

# %%
import copy
import random
import sys
from collections import deque

from prep import Session, check

s = Session("A1.5", target_min=60)
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


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

    def __repr__(self):
        return f"TreeNode({self.val})"


def from_level(vals):
    """LeetCode level-order list (None = missing child) -> root. Iterative."""
    if not vals or vals[0] is None:
        return None
    it = iter(vals)
    root = TreeNode(next(it))
    q = deque([root])
    while q:
        node = q.popleft()
        for side in ("left", "right"):
            v = next(it, "END")
            if v == "END":
                return root
            if v is not None:
                child = TreeNode(v)
                setattr(node, side, child)
                q.append(child)
    return root


def to_level(root):
    """root -> LeetCode level-order list with trailing Nones stripped. Iterative."""
    out, q = [], deque([root])
    while q:
        node = q.popleft()
        out.append(None if node is None else node.val)
        if node:
            q.extend([node.left, node.right])
    while out and out[-1] is None:
        out.pop()
    return out


def rand_tree(r, max_n=9, lo=0, hi=9, distinct=False):
    """Random binary-tree shape with random values, as a level-order list."""
    n = r.randint(0, max_n)
    vals = r.sample(range(lo, max(hi, lo + n) + 1), n) if distinct else [r.randint(lo, hi) for _ in range(n)]
    if not vals:
        return []
    root = TreeNode(vals[0])
    for v in vals[1:]:
        node = root
        while True:
            side = r.choice(("left", "right"))
            if getattr(node, side) is None:
                setattr(node, side, TreeNode(v))
                break
            node = getattr(node, side)
    return to_level(root)


def rand_bst(r, max_n=9, lo=0, hi=15):
    vals = r.sample(range(lo, hi + 1), r.randint(0, min(max_n, hi - lo + 1)))
    root = None
    for v in vals:
        if root is None:
            root = TreeNode(v)
            continue
        node = root
        while True:
            side = "left" if v < node.val else "right"
            if getattr(node, side) is None:
                setattr(node, side, TreeNode(v))
                break
            node = getattr(node, side)
    return to_level(root)


def _inorder_ref(root):               # trusted reference (iterative)
    out, stack, node = [], [], root
    while True:
        while node:
            stack.append(node)
            node = node.left
        if not stack:
            return out
        node = stack.pop()
        out.append(node.val)
        node = node.right


check("tree helpers round-trip", to_level(from_level([3, 9, 20, None, None, 15, 7])) == [3, 9, 20, None, None, 15, 7])
chain = [0] + [x for i in range(1, 3000) for x in (None, i)]   # right-leaning chain, depth 3000
print("recursion limit:", sys.getrecursionlimit())

# %% [markdown]
# ## 1. DFS: recursive and iterative
# **Intuition.** Most tree problems are "compute something for each subtree from its children's answers"
# (post-order) or "pass something down from the parent" (pre-order, e.g. bounds or a path sum). Recursion is
# shortest. The iterative version with an explicit stack avoids CPython's recursion limit (1000 frames by default),
# which skewed trees from an online judge can exceed.
#
# **Signals:** "depth/height/diameter/balanced", "path sum", "same/symmetric/subtree", anything defined
# recursively on subtrees. On a BST, inorder means **sorted order**.
#
# **Template A: inorder, recursive.**

# %%
def inorder_recursive(root):
    ### BEGIN SOLUTION
    out = []

    def go(node):
        if node:
            go(node.left)
            out.append(node.val)
            go(node.right)

    go(root)
    return out
    ### END SOLUTION


wrap = lambda f: (lambda lv: f(from_level(lv)))
run_tests(wrap(inorder_recursive), [(([2, 1, 3],), [1, 2, 3]), (([],), []), (([1, None, 2, 3],), [1, 3, 2])],
          name="inorder_recursive")
stress(wrap(inorder_recursive), wrap(_inorder_ref), lambda r: (rand_tree(r),), name="inorder_recursive")

# %% [markdown]
# **Template B: inorder, iterative.** Go left as far as possible, pushing nodes. Pop one, visit it, then move
# into its right subtree. Each node is pushed and popped once: $O(n)$ time, $O(h)$ space.

# %%
def inorder_iterative(root):
    ### BEGIN SOLUTION
    out, stack, node = [], [], root
    while stack or node:
        if node:
            stack.append(node)
            node = node.left
        else:
            node = stack.pop()
            out.append(node.val)
            node = node.right
    return out
    ### END SOLUTION


run_tests(wrap(inorder_iterative), [(([2, 1, 3],), [1, 2, 3]), (([],), []), (([1, None, 2, 3],), [1, 3, 2])],
          name="inorder_iterative")
stress(wrap(inorder_iterative), wrap(_inorder_ref), lambda r: (rand_tree(r),), name="inorder_iterative")
check("inorder_iterative on a 3000-deep chain", inorder_iterative(from_level(chain)) == list(range(3000)))

# %% [markdown]
# ## 2. BFS: level order
# **Intuition.** A queue visits nodes in order of depth. To group them by level, record `len(q)` at the start of
# each round and pop exactly that many nodes.
#
# **Signals:** "level by level", "right side view", "minimum depth" (BFS stops at the first leaf), "zigzag",
# "average per level", "connect next pointers", shortest path in an unweighted tree or graph.
#
# **Template C: level order as a list of lists.**

# %%
def level_order(root):
    ### BEGIN SOLUTION
    if not root:
        return []
    res, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):              # exactly the nodes of this level
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level)
    return res
    ### END SOLUTION


def level_order_brute(lv):
    res = []

    def go(node, d):
        if node:
            if d == len(res):
                res.append([])
            res[d].append(node.val)
            go(node.left, d + 1)
            go(node.right, d + 1)

    go(from_level(lv), 0)
    return res


run_tests(wrap(level_order), [(([3, 9, 20, None, None, 15, 7],), [[3], [9, 20], [15, 7]]), (([],), []), (([1],), [[1]])],
          name="level_order")
stress(wrap(level_order), level_order_brute, lambda r: (rand_tree(r),), name="level_order")
check("level_order on a 3000-deep chain", len(level_order(from_level(chain))) == 3000)

# %% [markdown]
# ## 3. Binary search trees
# **Intuition.** BST property (CLRS Ch. 12, here with distinct keys): every key in the left subtree is less than the
# node, and every key in the right subtree is greater. **Every** key in the subtree must satisfy this, not only the
# children. So validation passes an allowed open interval `(lo, hi)` down the tree. Search, insert and LCA walk one
# root-to-leaf path: $O(h)$, which is $O(\log n)$ only if the tree is balanced.
#
# **Signals:** "BST" in the statement; "sorted order", "k-th smallest", "closest value", "range sum", "successor".
#
# **Template D: BST search (iterative).**

# %%
def bst_search(root, x):
    """Return the node with value x, or None."""
    ### BEGIN SOLUTION
    node = root
    while node and node.val != x:
        node = node.left if x < node.val else node.right
    return node
    ### END SOLUTION


def _bst_search(lv, x):
    node = bst_search(from_level(lv), x)
    return None if node is None else node.val


run_tests(_bst_search, [(([4, 2, 7, 1, 3], 2), 2), (([4, 2, 7, 1, 3], 5), None), (([], 1), None)], name="bst_search")
stress(_bst_search, lambda lv, x: x if x in lv else None, lambda r: (rand_bst(r), r.randint(-1, 16)), name="bst_search")

# %% [markdown]
# ### Warm-up 3.1: is it a valid BST? (strict: no duplicates; ~10 min)

# %%
def is_valid_bst(root):
    ### BEGIN SOLUTION
    stack = [(root, float("-inf"), float("inf"))]     # iterative pre-order with open bounds
    while stack:
        node, lo, hi = stack.pop()
        if not node:
            continue
        if not (lo < node.val < hi):
            return False
        stack.append((node.left, lo, node.val))
        stack.append((node.right, node.val, hi))
    return True
    ### END SOLUTION


def is_valid_bst_brute(lv):
    vals = _inorder_ref(from_level(lv))
    return all(a < b for a, b in zip(vals, vals[1:]))


def _gen_maybe_bst(r):
    lv = rand_bst(r)
    if lv and r.random() < 0.5:                      # corrupt one value half the time
        idx = r.choice([i for i, v in enumerate(lv) if v is not None])
        lv[idx] = r.randint(0, 15)
    return (lv,)


run_tests(wrap(is_valid_bst), [
    (([2, 1, 3],), True),
    (([5, 1, 4, None, None, 3, 6],), False),
    (([5, 4, 6, None, None, 3, 7],), False),        # 3 < 5 deep in the right subtree: children-only checks miss it
    (([2, 2],), False),                              # duplicates are not allowed
    (([],), True),
], name="is_valid_bst")
stress(wrap(is_valid_bst), is_valid_bst_brute, _gen_maybe_bst, n=1000, name="is_valid_bst")

# %% [markdown]
# ### Warm-up 3.2: lowest common ancestor in a binary tree (~12 min)
# Values are distinct, and `p` and `q` are values that are both in the tree (`p == q` is allowed). Return the
# **value** of their deepest common ancestor. A node counts as its own ancestor. The tree is a general binary tree,
# not a BST. *Hint:* post-order. A node is the answer if `p` and `q` are found in different subtrees, or if it is
# one of them and the other is below it.

# %%
def lca(root, p, q):
    ### BEGIN SOLUTION
    def go(node):                       # returns the node p or q found in this subtree (or the LCA), else None
        if node is None or node.val in (p, q):
            return node
        left, right = go(node.left), go(node.right)
        if left and right:
            return node
        return left or right

    return go(root).val
    ### END SOLUTION


def lca_brute(lv, p, q):
    root = from_level(lv)
    parent, stack = {root.val: None}, [root]
    while stack:
        node = stack.pop()
        for c in (node.left, node.right):
            if c:
                parent[c.val] = node.val
                stack.append(c)
    anc = set()
    while p is not None:
        anc.add(p)
        p = parent[p]
    while q not in anc:
        q = parent[q]
    return q


def _gen_lca(r):
    lv = rand_tree(r, 10, distinct=True)
    while not lv:
        lv = rand_tree(r, 10, distinct=True)
    vals = [v for v in lv if v is not None]
    return lv, r.choice(vals), r.choice(vals)


run_tests(lambda lv, p, q: lca(from_level(lv), p, q), [
    (([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4], 5, 1), 3),
    (([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4], 5, 4), 5),   # p is an ancestor of q
    (([1, 2], 2, 2), 2),
], name="lca")
stress(lambda lv, p, q: lca(from_level(lv), p, q), lca_brute, _gen_lca, n=1000, name="lca")

# %% [markdown]
# ### Warm-up 3.3: k-th smallest in a BST (~8 min)
# Return the k-th smallest value (1-indexed, `1 <= k <= n`). *Target:* $O(h + k)$ with an iterative inorder that
# **stops early**.

# %%
def kth_smallest(root, k):
    ### BEGIN SOLUTION
    stack, node = [], root
    while True:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        k -= 1
        if k == 0:
            return node.val
        node = node.right
    ### END SOLUTION


def _gen_kth(r):
    lv = rand_bst(r)
    while not lv:
        lv = rand_bst(r)
    return lv, r.randint(1, sum(v is not None for v in lv))


run_tests(lambda lv, k: kth_smallest(from_level(lv), k), [(([3, 1, 4, None, 2], 1), 1), (([5, 3, 6, 2, 4, None, None, 1], 3), 3),
                                                          (([1], 1), 1)], name="kth_smallest")
stress(lambda lv, k: kth_smallest(from_level(lv), k), lambda lv, k: sorted(v for v in lv if v is not None)[k - 1],
       _gen_kth, name="kth_smallest")

# %% [markdown]
# ## 4. Serialization
# ### Warm-up 4.1: serialize and deserialize a binary tree (classic hard; ~15 min)
# Write `serialize(root) -> str` and `deserialize(str) -> root` so that any tree with integer values (negative
# values, duplicates and the empty tree included) round-trips exactly. You choose the format. *Hint:* pre-order with
# a marker such as `#` for missing children is unambiguous. Without the markers, pre-order alone is not.

# %%
def serialize(root):
    ### BEGIN SOLUTION
    out, stack = [], [root]
    while stack:                               # iterative pre-order
        node = stack.pop()
        if node is None:
            out.append("#")
        else:
            out.append(str(node.val))
            stack.append(node.right)
            stack.append(node.left)
    return ",".join(out)
    ### END SOLUTION


def deserialize(data):
    ### BEGIN SOLUTION
    tokens = iter(data.split(","))
    first = next(tokens)
    if first == "#":
        return None
    root = TreeNode(int(first))
    stack = [(root, "left")]                   # slots waiting to be filled, in pre-order
    for tok in tokens:
        node, side = stack.pop()
        if side == "left":
            stack.append((node, "right"))
        if tok != "#":
            child = TreeNode(int(tok))
            setattr(node, side, child)
            stack.append((child, "left"))
    return root
    ### END SOLUTION


def _roundtrip(lv):
    data = serialize(from_level(lv))
    assert isinstance(data, str), "serialize must return a str"
    return to_level(deserialize(data))


run_tests(_roundtrip, [(([1, 2, 3, None, None, 4, 5],), [1, 2, 3, None, None, 4, 5]), (([],), []),
                       (([-1, None, -1],), [-1, None, -1]), (([1, 1],), [1, 1]), (([1, None, 1],), [1, None, 1])],
          name="serialize/deserialize")
stress(_roundtrip, lambda lv: lv, lambda r: (rand_tree(r, 10, -12, 12),), n=1000, name="serialize/deserialize")
try:                                       # informational: recursive versions are fine in an interview
    check("serialize/deserialize on a 3000-deep chain", _roundtrip(chain) == chain)
except RecursionError:
    print("note: your serializer is recursive and hits the recursion limit on a 3000-deep chain (OK in an interview; say so)")

# %% [markdown]
# ## 5. Tries (prefix trees)
# **Intuition.** A trie (Fredkin 1960) stores strings character by character along root-to-node paths. Shared
# prefixes share nodes. `insert`, `search` and `starts_with` cost $O(L)$ for a key of length $L$, independent of how
# many words are stored. A node needs a `children` dict and an **end-of-word flag**: without the flag, `search`
# cannot tell a stored word from a prefix of one.
#
# **Signals:** many prefix queries, "autocomplete", "word search on a grid with a dictionary" (trie plus
# backtracking, A1.8), "longest common prefix", "replace words by their shortest root", XOR-maximization (binary trie).
#
# ### Warm-up 5.1: implement a trie (~12 min)

# %%
class Trie:
    def __init__(self):
        ### BEGIN SOLUTION
        self.root = {}                         # char -> child dict; "$" key marks end of word
        ### END SOLUTION

    def insert(self, word):
        ### BEGIN SOLUTION
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node["$"] = True
        ### END SOLUTION

    def _walk(self, s_):
        ### BEGIN SOLUTION
        node = self.root
        for ch in s_:
            if ch not in node:
                return None
            node = node[ch]
        return node
        ### END SOLUTION

    def search(self, word):
        ### BEGIN SOLUTION
        node = self._walk(word)
        return node is not None and "$" in node
        ### END SOLUTION

    def starts_with(self, prefix):
        ### BEGIN SOLUTION
        return self._walk(prefix) is not None
        ### END SOLUTION


class TrieBrute:
    def __init__(self):
        self.words = set()

    def insert(self, word):
        self.words.add(word)

    def search(self, word):
        return word in self.words

    def starts_with(self, prefix):
        return any(w.startswith(prefix) for w in self.words)


def replay(cls, ops):
    obj, out = cls(), []
    for name, *args in ops:
        out.append(getattr(obj, name)(*args))
    return out


def _gen_trie_ops(r):
    word = lambda: "".join(r.choice("abc") for _ in range(r.randint(1, 4)))
    return ([(r.choice(["insert", "search", "starts_with"]), word()) for _ in range(r.randint(1, 15))],)


run_tests(lambda ops: replay(Trie, ops), [(([("insert", "apple"), ("search", "apple"), ("search", "app"),
                                            ("starts_with", "app"), ("insert", "app"), ("search", "app"),
                                            ("starts_with", "b")],), [None, True, False, True, None, True, False])],
          name="Trie")
stress(lambda ops: replay(Trie, ops), lambda ops: replay(TrieBrute, ops), _gen_trie_ops, n=500, name="Trie")

# %% [markdown]
# ## Say it out loud (complexity and correctness)
# 1. Time and space of recursive DFS on a tree with $n$ nodes and height $h$? Best and worst $h$?
# 2. Why is "check each node against its two children" not enough for BST validation? Give a counterexample.
# 3. LCA: what is the complexity of your version, and how does the BST version differ? What if `p` or `q` might be missing?
# 4. Why is pre-order **with null markers** enough to rebuild a tree, while pre-order alone is not? Which pairs of
#    traversals (without markers, distinct values) determine a tree?
# 5. Trie vs hash set of all words, for prefix queries: time and memory?
# 6. BFS vs DFS on a tree: which uses less memory on a complete binary tree, and which on a long path?
# 7. Iterative vs recursive in a Python interview: when would you switch, and how do you say it?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. $O(n)$ time and $O(h)$ stack space. A balanced tree has $h = \Theta(\log n)$. A degenerate (path-like) tree has
#    $h = n$, and a 3000-deep chain then raises `RecursionError` at CPython's default recursion limit of 1000
#    (`sys.getrecursionlimit`).
# 2. The BST property concerns whole subtrees (CLRS Ch. 12). In `[5, 4, 6, None, None, 3, 7]`, every parent-child
#    pair is ordered correctly, but 3 sits in 5's right subtree. Pass the bounds `(lo, hi)` down, or check that the
#    inorder sequence is strictly increasing.
# 3. $O(n)$ time and $O(h)$ space: post-order visits each node once. In a BST, walk from the root. If both values are
#    smaller go left, if both are larger go right, otherwise this node is the LCA: $O(h)$. If a value might be absent,
#    the shortcut "return the node when it equals p" gives a wrong answer. Count how many of p and q were actually
#    found, and return None unless both were.
# 4. With markers, each token says whether a node exists, so the parser knows exactly when a subtree ends. Without them,
#    `[1, 2]` could mean 2 is the left or the right child. With distinct values, inorder plus pre-order (or
#    inorder plus post-order) determines the tree. Pre-order plus post-order does not, unless every node has 0 or 2 children.
# 5. Hash set: `search` is expected $O(L)$ (to hash the string), but a prefix query scans all $n$ words, $O(nL)$.
#    A trie answers both in $O(L)$. Its memory is $O(\text{total characters})$ nodes in the worst case, with a large
#    constant for dicts, and less when prefixes are shared.
# 6. A complete tree has about $n/2$ leaves on the last level, so BFS's queue holds $\Theta(n)$ nodes while DFS's
#    stack holds $O(\log n)$. On a path, BFS holds $O(1)$ and DFS holds $O(n)$.
# 7. When depth can exceed about 1000 (skewed trees, linked-list-like inputs, big grids in A1.7). Say: "Recursion is
#    clearer, but CPython's default limit is 1000, so for adversarial depth I'll use an explicit stack." Raising
#    `sys.setrecursionlimit` is a hack that can still crash the C stack.
# </details>

# %% [markdown]
# <!-- practice:start -->
# ## Practice list (solve these on LeetCode)
#
# NeetCode 150 problems for this topic, generated from `neetcode-gh/leetcode@d5ebb3e` `.problemSiteData.json` (https://neetcode.io/practice). **B75** = also in Blind 75. Tick them off in your session copy; the full checklist is `A_coding/A1_problem_list.md`. Grind 75 (https://www.techinterviewhandbook.org/grind75/) is an alternative ordering of a similar set.
#
# **Trees** · plan week 9 · 15 problems
#
# - [ ] [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/) · Easy · **B75**
# - [ ] [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/) · Easy · **B75**
# - [ ] [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/) · Easy
# - [ ] [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/) · Easy
# - [ ] [Same Tree](https://leetcode.com/problems/same-tree/) · Easy · **B75**
# - [ ] [Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/) · Easy · **B75**
# - [ ] [Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) · Medium · **B75**
# - [ ] [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/) · Medium · **B75**
# - [ ] [Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/) · Medium
# - [ ] [Count Good Nodes In Binary Tree](https://leetcode.com/problems/count-good-nodes-in-binary-tree/) · Medium
# - [ ] [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/) · Medium · **B75**
# - [ ] [Kth Smallest Element In a Bst](https://leetcode.com/problems/kth-smallest-element-in-a-bst/) · Medium · **B75**
# - [ ] [Construct Binary Tree From Preorder And Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) · Medium · **B75**
# - [ ] [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/) · Hard · **B75**
# - [ ] [Serialize And Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) · Hard · **B75**
#
# **Tries** · plan week 11 · 3 problems
#
# - [ ] [Implement Trie Prefix Tree](https://leetcode.com/problems/implement-trie-prefix-tree/) · Medium · **B75**
# - [ ] [Design Add And Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/) · Medium · **B75**
# - [ ] [Word Search II](https://leetcode.com/problems/word-search-ii/) · Hard · **B75**
#
# **Extras (not in NeetCode 150; close to the warm-ups above)**
#
# - [ ] [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/) · Easy
# <!-- practice:end -->

# %%
s.finish(lookups=0, score=None, notes="")
