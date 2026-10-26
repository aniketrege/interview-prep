# %% [markdown]
# # A1.7 · Graphs: BFS/DFS, topological sort, union-find, Dijkstra, cycle detection
#
# **ID:** A1.7 · **Target:** 60 min · **Plan week:** 4 (pattern notebook, Thu); second pass in weeks 12 (BFS/DFS, topo sort, union-find) and 13 (Dijkstra) · **Part:** A
#
# **The drill.** This is a 60-minute study notebook. Fill in five templates: BFS shortest path on a grid, topological
# sort two ways (Kahn's in-degree queue and DFS with three colours, both of which detect cycles), union-find with path
# compression and union by rank, and Dijkstra with `heapq` and lazy deletion. Then do three warm-ups (10-15 min each):
# counting islands, multi-source BFS ("rotting"), and the redundant edge with union-find. Brute forces are
# deliberately different algorithms: repeated relaxation, Bellman-Ford, transitive closure, and BFS for connectivity.
#
# **Rules.** Python standard library only. Graphs come as `n` plus an edge list (nodes `0..n-1` unless a warm-up says
# otherwise). Build the adjacency list yourself. Fill in every `raise NotImplementedError`.
#
# ## Sources
# - T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, 4th ed., MIT Press, 2022
#   (ISBN 9780262046305): Ch. 20 "Elementary Graph Algorithms" (BFS, DFS, white/gray/black colouring, topological sort),
#   Ch. 19 "Data Structures for Disjoint Sets" (union by rank + path compression, $O(m\,\alpha(n))$),
#   Ch. 22 "Single-Source Shortest Paths" (Bellman-Ford, Dijkstra). https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
# - A. B. Kahn, *Topological sorting of large networks*, Communications of the ACM 5(11), 1962, DOI 10.1145/368996.369025.
# - R. E. Tarjan, *Depth-First Search and Linear Graph Algorithms*, SIAM J. Computing 1(2), 1972, DOI 10.1137/0201010.
# - R. E. Tarjan, *Efficiency of a Good But Not Linear Set Union Algorithm*, Journal of the ACM 22(2), 1975,
#   DOI 10.1145/321879.321884 (the inverse-Ackermann bound).
# - E. W. Dijkstra, *A note on two problems in connexion with graphs*, Numerische Mathematik 1, 1959, DOI 10.1007/BF01386390.
# - Python docs, `heapq` (priority-queue notes: mark entries removed instead of deleting = lazy deletion):
#   https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes ; `collections.deque`:
#   https://docs.python.org/3/library/collections.html#collections.deque
# - Tech Interview Handbook, *Graph* and *Matrix* cheatsheets, https://www.techinterviewhandbook.org/algorithms/graph/ ,
#   https://www.techinterviewhandbook.org/algorithms/matrix/ (yangshun/tech-interview-handbook @ e1d28e8, MIT).
# - NeetCode 150 problem data (practice list): https://neetcode.io/practice ,
#   https://github.com/neetcode-gh/leetcode @ d5ebb3e `.problemSiteData.json`.

# %%
import contextlib
import copy
import heapq
import math
import random
import signal
from collections import defaultdict, deque

from prep import Session, check

s = Session("A1.7", target_min=60)
random.seed(0)


@contextlib.contextmanager
def time_limit(sec=20):
    """Raise TimeoutError if the block runs longer than sec (e.g. a Dijkstra that re-pushes forever). Unix only."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f"took more than {sec}s: infinite loop? (visited set / stale-entry check?)")

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


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def rand_grid(r, chars, weights, max_side=6):
    h, w = r.randint(1, max_side), r.randint(1, max_side)
    return ["".join(r.choices(chars, weights, k=w)) for _ in range(h)]

# %% [markdown]
# ## 1. BFS: shortest paths when every edge costs 1
# **Intuition.** BFS visits nodes in order of distance from the source, so the first time you reach a node is along
# a shortest path (CLRS Ch. 20). Mark nodes **when you enqueue them**, not when you pop them, or a node can be queued
# many times. Grids are implicit graphs: neighbours are the 4 (or 8) adjacent in-bounds, non-wall cells.
#
# **Signals:** "minimum number of steps/moves/transformations", unweighted shortest path, "nearest X" from many
# sources (multi-source BFS), level-by-level spreading ("minutes until ..."), word ladder.
#
# **Template A: BFS on a grid.** `grid` is a list of strings, `#` is a wall. Return the fewest 4-directional steps
# from `start` to `goal` (both `(row, col)` on open cells), or -1.

# %%
def bfs_grid(grid, start, goal):
    ### BEGIN SOLUTION
    h, w = len(grid), len(grid[0])
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return dist[(r, c)]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != "#" and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1      # mark on enqueue
                q.append((nr, nc))
    return -1
    ### END SOLUTION


def bfs_grid_brute(grid, start, goal):        # repeated relaxation until nothing changes (Bellman-Ford style)
    h, w = len(grid), len(grid[0])
    INF = math.inf
    d = {(r, c): INF for r in range(h) for c in range(w) if grid[r][c] != "#"}
    d[start] = 0
    changed = True
    while changed:
        changed = False
        for (r, c), v in list(d.items()):
            for dr, dc in DIRS:
                u = (r + dr, c + dc)
                if u in d and d[u] + 1 < v:
                    d[(r, c)] = v = d[u] + 1
                    changed = True
    return -1 if d[goal] == INF else d[goal]


def _gen_bfs(r):
    g = rand_grid(r, ".#", (3, 1))
    cells = [(i, j) for i, row in enumerate(g) for j, ch in enumerate(row) if ch == "."]
    if not cells:
        g, cells = ["."], [(0, 0)]
    return g, r.choice(cells), r.choice(cells)


run_tests(bfs_grid, [((["...", ".#.", "..."], (0, 0), (2, 2)), 4), (([".#.", ".#.", ".#."], (0, 0), (0, 2)), -1),
                     (([".."], (0, 1), (0, 1)), 0), ((["....", "##.#", "...."], (0, 0), (2, 0)), 6)])
stress(bfs_grid, bfs_grid_brute, _gen_bfs, n=500)

# %% [markdown]
# ### Warm-up 1.1: count islands (~10 min)
# `grid` is a list of strings of `'1'` (land) and `'0'` (water). Islands are groups of land connected
# horizontally or vertically. Return how many there are. Flood-fill each new land cell with BFS or an iterative DFS.
# A recursive flood fill also works, but a 60×60 all-land grid exceeds CPython's default recursion limit (1000).

# %%
def num_islands(grid):
    ### BEGIN SOLUTION
    h, w = len(grid), len(grid[0]) if grid else 0
    seen, count = set(), 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] == "1" and (i, j) not in seen:
                count += 1
                seen.add((i, j))
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    for dr, dc in DIRS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == "1" and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
    return count
    ### END SOLUTION


def num_islands_brute(grid):                # union-find over land cells (a different algorithm)
    parent = {}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch == "1":
                parent[(i, j)] = (i, j)
                for u in ((i - 1, j), (i, j - 1)):
                    if u in parent:
                        parent[find(u)] = find((i, j))
    return len({find(x) for x in parent})


run_tests(num_islands, [((["11000", "11000", "00100", "00011"],), 3), ((["0"],), 0), ((["1"],), 1),
                        ((["101", "010", "101"],), 5),           # diagonals do not connect
                        ((["111", "101", "111"],), 1)])
stress(num_islands, num_islands_brute, lambda r: (rand_grid(r, "01", (1, 1)),), n=500)
try:                                          # informational: only recursive flood fills fail here
    check("num_islands on a 60x60 all-land grid", num_islands(["1" * 60] * 60) == 1)
except RecursionError:
    print("note: recursive flood fill hit the recursion limit on 60x60; say so, or switch to BFS/explicit stack")

# %% [markdown]
# ### Warm-up 1.2: minutes until everything rots (multi-source BFS; ~12 min)
# Grid cells: 0 = empty, 1 = fresh, 2 = rotten. Every minute, fresh cells 4-adjacent to a rotten cell become
# rotten. Return the minutes until no fresh cell is left, or -1 if that never happens. *Hint:* start BFS from
# **all** rotten cells at once (distance 0), and count the fresh cells you convert.

# %%
def rotting_minutes(grid):
    ### BEGIN SOLUTION
    h, w = len(grid), len(grid[0])
    q = deque((i, j, 0) for i in range(h) for j in range(w) if grid[i][j] == 2)
    fresh = sum(row.count(1) for row in grid)
    g = [row[:] for row in grid]
    minutes = 0
    while q:
        r, c, t = q.popleft()
        minutes = max(minutes, t)
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr][nc] == 1:
                g[nr][nc] = 2
                fresh -= 1
                q.append((nr, nc, t + 1))
    return minutes if fresh == 0 else -1
    ### END SOLUTION


def rotting_brute(grid):                     # literal minute-by-minute simulation
    g, t = [row[:] for row in grid], 0
    while True:
        fresh = [(i, j) for i, row in enumerate(g) for j, v in enumerate(row) if v == 1]
        if not fresh:
            return t
        rot = [(i, j) for i, j in fresh if any(0 <= i + a < len(g) and 0 <= j + b < len(g[0]) and g[i + a][j + b] == 2
                                               for a, b in DIRS)]
        if not rot:
            return -1
        for i, j in rot:
            g[i][j] = 2
        t += 1


run_tests(rotting_minutes, [(([[2, 1, 1], [1, 1, 0], [0, 1, 1]],), 4), (([[2, 1, 1], [0, 1, 1], [1, 0, 1]],), -1),
                            (([[0, 2]],), 0), (([[0]],), 0), (([[1]],), -1), (([[2, 1, 1, 1, 2]],), 2)])
def _gen_rot(r):
    h, w = r.randint(1, 5), r.randint(1, 5)
    return ([r.choices((0, 1, 2), (1, 3, 1), k=w) for _ in range(h)],)


stress(rotting_minutes, rotting_brute, _gen_rot, n=500)

# %% [markdown]
# ## 2. Topological sort and directed cycle detection
# **Intuition.** A topological order puts every edge $u \to v$ with $u$ before $v$. It exists iff the graph is a DAG.
# - **Kahn (1962):** repeatedly output a node with in-degree 0 and decrement its successors. If fewer than $n$ nodes come
#   out, the remaining ones lie on or behind a cycle.
# - **DFS (Tarjan 1972, CLRS Ch. 20):** colour nodes white/gray/black. Finding a **gray** node again means a back edge,
#   which means a cycle. Appending each node when it turns black and reversing the list gives a topological order.
#
# **Signals:** "prerequisites", "build order", "can all courses be finished", "dependencies", "alien dictionary"
# (derive edges from adjacent words), "longest path in a DAG" (DP over the topological order).
#
# **Template B: Kahn's algorithm.** Return a valid order of `0..n-1`, or `None` if there is a cycle. `edges` holds
# `(u, v)` pairs meaning `u` must come before `v`. Self-loops and duplicate edges may occur.

# %%
def topo_kahn(n, edges):
    ### BEGIN SOLUTION
    adj, indeg = defaultdict(list), [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else None
    ### END SOLUTION


# %% [markdown]
# **Template C: DFS with three colours** (same contract). A recursive helper is fine at these sizes.

# %%
def topo_dfs(n, edges):
    ### BEGIN SOLUTION
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    WHITE, GRAY, BLACK = 0, 1, 2
    color, post = [WHITE] * n, []

    def visit(u):
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:              # back edge -> cycle
                return False
            if color[v] == WHITE and not visit(v):
                return False
        color[u] = BLACK
        post.append(u)
        return True

    for u in range(n):
        if color[u] == WHITE and not visit(u):
            return None
    return post[::-1]
    ### END SOLUTION


def topo_valid(fn):
    """Wrap a topo-sort: None stays None; an order becomes True iff it is a valid topological order."""
    def w(n, edges):
        order = fn(n, edges)
        if order is None:
            return None
        pos = {v: i for i, v in enumerate(order)}
        return sorted(order) == list(range(n)) and all(pos[u] < pos[v] for u, v in edges)
    w.__name__ = fn.__name__
    return w


def topo_brute(n, edges):
    """None if there is a directed cycle (transitive closure: some node reaches itself), else True."""
    reach = [[False] * n for _ in range(n)]
    for u, v in edges:
        reach[u][v] = True
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    reach[i][j] = reach[i][j] or reach[k][j]
    return None if any(reach[i][i] for i in range(n)) else True


def _gen_digraph(r):
    n = r.randint(1, 7)
    perm = r.sample(range(n), n)
    edges = []
    for _ in range(r.randint(0, 2 * n)):
        a, b = sorted(r.sample(range(n), 2)) if n > 1 else (0, 0)
        if n > 1:
            edges.append((perm[a], perm[b]))          # forward edge in a hidden order: stays a DAG
    if r.random() < 0.4:                              # sometimes add a random edge (may create a cycle or self-loop)
        edges.append((r.randrange(n), r.randrange(n)))
    return n, edges


cases_topo = [((4, [(1, 0), (2, 0), (3, 1), (3, 2)]), True), ((2, [(0, 1), (1, 0)]), None), ((1, []), True),
              ((1, [(0, 0)]), None), ((3, [(0, 1), (0, 1)]), True), ((3, [(0, 1), (1, 2), (2, 1)]), None)]
for f in (topo_kahn, topo_dfs):
    run_tests(topo_valid(f), cases_topo)
    stress(topo_valid(f), topo_brute, _gen_digraph, n=1000)

# %% [markdown]
# ## 3. Union-find (disjoint sets)
# **Intuition.** Keep a forest in which each set is a tree identified by its root. `find` follows parents to the root,
# and **path compression** re-points every visited node directly to the root. `union` links the root of lower
# **rank** (an upper bound on tree height) under the higher one. Together they give $O(\alpha(n))$ amortized per
# operation, where $\alpha$ is the inverse Ackermann function, at most 4 for any practical $n$ (Tarjan 1975; CLRS Ch. 19).
#
# **Signals:** "connected components" under incremental edge additions, "are x and y connected", "redundant
# edge", "is it a valid tree", Kruskal's MST, "accounts merge", equivalence classes ("similar strings").
#
# **Template D.** `union` returns True if it merged two different sets. `count` is the number of sets.

# %%
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x):
        ### BEGIN SOLUTION
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:             # path compression (iterative, no recursion limit)
            self.parent[x], x = root, self.parent[x]
        return root
        ### END SOLUTION

    def union(self, a, b):
        ### BEGIN SOLUTION
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra                      # lower rank under higher rank
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.count -= 1
        return True
        ### END SOLUTION

    def connected(self, a, b):
        return self.find(a) == self.find(b)


class UnionFindBrute:                             # adjacency sets + BFS on every query
    def __init__(self, n):
        self.adj = [set() for _ in range(n)]

    def _comp(self, a):
        seen, q = {a}, [a]
        while q:
            for v in self.adj[q.pop()]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen

    def union(self, a, b):
        merged = b not in self._comp(a)
        self.adj[a].add(b)
        self.adj[b].add(a)
        return merged

    def connected(self, a, b):
        return b in self._comp(a)

    @property
    def count(self):
        seen, c = set(), 0
        for v in range(len(self.adj)):
            if v not in seen:
                seen |= self._comp(v)
                c += 1
        return c


def replay_uf(cls, n, ops):
    uf, out = cls(n), []
    for op, *args in ops:
        out.append(uf.count if op == "count" else getattr(uf, op)(*args))
    return out


def _gen_uf(r):
    n = r.randint(1, 8)
    ops = [(r.choice(["union", "union", "connected"]), r.randrange(n), r.randrange(n)) if r.random() < 0.85
           else ("count",) for _ in range(r.randint(1, 15))]
    return n, ops


run_tests(lambda n, ops: replay_uf(UnionFind, n, ops),
          [((5, [("union", 0, 1), ("union", 1, 2), ("union", 0, 2), ("connected", 0, 2), ("connected", 0, 3), ("count",)]),
            [True, True, False, True, False, 3])], name="UnionFind")
stress(lambda n, ops: replay_uf(UnionFind, n, ops), lambda n, ops: replay_uf(UnionFindBrute, n, ops), _gen_uf,
       n=1000, name="UnionFind")

uf = UnionFind(1024)
rng = random.Random(1)
for _ in range(3000):
    uf.union(rng.randrange(1024), rng.randrange(1024))
check("union by rank keeps rank <= log2(n)", max(uf.rank) <= 10)
uf = UnionFind(3)
uf.union(0, 1)
r01 = uf.find(0)
uf.union(2, 0)                                    # singleton (rank 0) must go under the rank-1 root
check("union by rank: lower-rank root goes under higher-rank root", uf.parent[2] == r01 and uf.parent[r01] == r01)
uf = UnionFind(5)
uf.parent = [0, 0, 1, 2, 3]                       # hand-built chain 4 -> 3 -> 2 -> 1 -> 0
uf.find(4)
check("path compression: every node on the path now points at the root", uf.parent == [0, 0, 0, 0, 0],
      "this check expects full compression; path halving is also O(alpha(n)) but leaves [0, 0, 0, 2, 2]")

# %% [markdown]
# ### Warm-up 3.1: the redundant edge (~10 min)
# Nodes are labelled `1..n`. The edge list was a tree on these nodes plus **one** extra edge, so there is exactly one
# cycle. Return the edge that can be removed to leave a tree; if several can, return the one that appears **last** in
# the input. *Hint:* add the edges in order with union-find. The first edge whose endpoints are already connected is
# the answer.

# %%
def redundant_edge(edges):
    ### BEGIN SOLUTION
    uf = UnionFind(len(edges) + 1)
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]
    ### END SOLUTION


def redundant_brute(edges):
    n = len(edges)

    def is_tree(es):
        adj = defaultdict(list)
        for u, v in es:
            adj[u].append(v)
            adj[v].append(u)
        seen, st = {1}, [1]
        while st:
            for v in adj[st.pop()]:
                if v not in seen:
                    seen.add(v)
                    st.append(v)
        return len(seen) == n

    for i in range(n - 1, -1, -1):
        if is_tree(edges[:i] + edges[i + 1:]):
            return list(edges[i])


def _gen_redundant(r):
    n = r.randint(3, 8)
    edges = [[r.randint(1, v - 1), v] for v in range(2, n + 1)]     # random tree
    a, b = r.sample(range(1, n + 1), 2)
    while [min(a, b), max(a, b)] in [sorted(e) for e in edges]:
        a, b = r.sample(range(1, n + 1), 2)
    edges.append([a, b])
    r.shuffle(edges)
    return ([e if r.random() < 0.5 else e[::-1] for e in edges],)


run_tests(redundant_edge, [(([[1, 2], [1, 3], [2, 3]],), [2, 3]), (([[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]],), [1, 4])])
stress(redundant_edge, redundant_brute, _gen_redundant, n=500)

# %% [markdown]
# ## 4. Dijkstra (non-negative weights)
# **Intuition.** Dijkstra is BFS with a priority queue ordered by tentative distance. When a node is popped with its
# final (smallest) distance, its outgoing edges are relaxed (Dijkstra 1959; CLRS Ch. 22). `heapq` has no decrease-key,
# so push a new `(dist, node)` entry on every improvement, and on pop **skip entries that are stale**
# (`d > dist[u]`). This is lazy deletion, as in the `heapq` docs. The heap then holds up to $O(m)$ entries, and the
# running time is $O(m\log m) = O(m\log n)$.
#
# **Signals:** weighted shortest path with non-negative weights; "minimum cost/time/effort"; "network delay"; grid
# with costs; a "minimax" path (minimize the maximum edge, e.g. swim in rising water) where `+` becomes `max`.
# Negative edges need Bellman-Ford. 0/1 weights can use a deque (0-1 BFS).
#
# **Template E.** `edges` holds directed `(u, v, w)` triples with `w >= 0`. Return the list of distances from `src`
# (`math.inf` if unreachable).

# %%
def dijkstra(n, edges, src):
    ### BEGIN SOLUTION
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [math.inf] * n
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue                               # stale entry (lazy deletion)
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    return dist
    ### END SOLUTION


def bellman_ford(n, edges, src):
    dist = [math.inf] * n
    dist[src] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist


def _gen_weighted(r):
    n = r.randint(1, 7)
    edges = [(r.randrange(n), r.randrange(n), r.randint(0, 9)) for _ in range(r.randint(0, 3 * n))]
    return n, edges, r.randrange(n)


run_tests(dijkstra, [((4, [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1)], 0), [0, 3, 1, 4]),
                     ((3, [(0, 1, 5)], 0), [0, 5, math.inf]), ((1, [], 0), [0]),
                     ((3, [(0, 1, 0), (1, 2, 0)], 0), [0, 0, 0])])
stress(dijkstra, bellman_ford, _gen_weighted, n=1000)

# %% [markdown]
# ## Say it out loud (complexity and correctness)
# 1. BFS/DFS complexity on an adjacency list and on an $R \times C$ grid? Why mark visited on enqueue?
# 2. Kahn vs DFS topological sort: complexity, how each detects a cycle, and which is easier to make lexicographically smallest?
# 3. For undirected graphs, how do you detect a cycle with DFS, and with union-find?
# 4. What do path compression and union by rank each give alone, and together?
# 5. Why does Dijkstra fail with a negative edge? Give a 3-node example. What do you use instead?
# 6. Lazy deletion: how many heap entries can there be, and what is the final complexity?
# 7. When do you choose BFS vs Dijkstra vs 0-1 BFS vs Bellman-Ford vs Floyd-Warshall?

# %% [markdown] tags=["solution_only"]
# <details><summary>Model answers</summary>
#
# 1. $O(V + E)$ time and $O(V)$ space. A grid has $V = RC$ and $E \le 4RC$, so $O(RC)$. Marking on enqueue means each node
#    enters the queue once. Marking on pop lets a node be enqueued once per neighbour: the result is still correct, but
#    the extra work can push it past time limits.
# 2. Both are $O(V + E)$. Kahn finds a cycle when it outputs fewer than $V$ nodes. DFS finds one by meeting a **gray**
#    node, which is a back edge (CLRS Ch. 20). For the lexicographically smallest order, run Kahn with a min-heap instead
#    of a queue, at $O((V + E)\log V)$.
# 3. DFS: a visited neighbour that is not the node's parent means a cycle. Parallel edges need care: track the edge id,
#    not the parent. Union-find: an edge whose endpoints already have the same root closes a cycle.
# 4. Union by rank alone keeps trees $O(\log n)$ high, so each operation is $O(\log n)$. Path compression alone gives
#    $O(\log n)$ amortized. Together the cost is $O(m\,\alpha(n))$ for $m$ operations, where $\alpha(n) \le 4$ in
#    practice (Tarjan 1975; CLRS Ch. 19).
# 5. Dijkstra finalizes a node when it is popped, assuming no later path can be shorter. A negative edge breaks that
#    assumption. Example: edges $s \to a$ (2), $s \to b$ (3), $b \to a$ (-2). Dijkstra pops $a$ with distance 2, but the
#    true distance is 1. Lazy versions that re-push may recover on some inputs, but they lose the complexity guarantee.
#    Use Bellman-Ford, $O(VE)$, which also detects negative cycles (CLRS Ch. 22).
# 6. One entry per successful relaxation, so at most $E + 1$ entries. That gives $O(E\log E) = O(E\log V)$ time and
#    $O(E)$ heap space.
# 7. Unit weights: BFS. Weights in {0, 1}: 0-1 BFS with a deque (push 0-edges to the front), $O(V + E)$. Non-negative
#    weights: Dijkstra. Negative edges: Bellman-Ford. All pairs with small $V$ (about 400 or fewer): Floyd-Warshall, $O(V^3)$.
#    Shortest paths in a DAG: relax in topological order, $O(V + E)$, negative weights allowed.
# </details>

# %% [markdown]
# <!-- practice:start -->
# ## Practice list (solve these on LeetCode)
#
# NeetCode 150 problems for this topic, generated from `neetcode-gh/leetcode@d5ebb3e` `.problemSiteData.json` (https://neetcode.io/practice). **B75** = also in Blind 75. Tick them off in your session copy; the full checklist is `A_coding/A1_problem_list.md`. Grind 75 (https://www.techinterviewhandbook.org/grind75/) is an alternative ordering of a similar set.
#
# **Graphs** · plan week 12 · 13 problems
#
# - [ ] [Number of Islands](https://leetcode.com/problems/number-of-islands/) · Medium · **B75**
# - [ ] [Clone Graph](https://leetcode.com/problems/clone-graph/) · Medium · **B75**
# - [ ] [Max Area of Island](https://leetcode.com/problems/max-area-of-island/) · Medium
# - [ ] [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/) · Medium · **B75**
# - [ ] [Surrounded Regions](https://leetcode.com/problems/surrounded-regions/) · Medium
# - [ ] [Rotting Oranges](https://leetcode.com/problems/rotting-oranges/) · Medium
# - [ ] [Walls And Gates](https://leetcode.com/problems/walls-and-gates/) · Medium
# - [ ] [Course Schedule](https://leetcode.com/problems/course-schedule/) · Medium · **B75**
# - [ ] [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/) · Medium
# - [ ] [Redundant Connection](https://leetcode.com/problems/redundant-connection/) · Medium
# - [ ] [Number of Connected Components In An Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/) · Medium · **B75**
# - [ ] [Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/) · Medium · **B75**
# - [ ] [Word Ladder](https://leetcode.com/problems/word-ladder/) · Hard
#
# **Advanced Graphs** · plan week 13 · 6 problems
#
# - [ ] [Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/) · Hard
# - [ ] [Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/) · Medium
# - [ ] [Network Delay Time](https://leetcode.com/problems/network-delay-time/) · Medium
# - [ ] [Swim In Rising Water](https://leetcode.com/problems/swim-in-rising-water/) · Hard
# - [ ] [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/) · Hard · **B75**
# - [ ] [Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/) · Medium
#
# **Extras (not in NeetCode 150; close to the warm-ups above)**
#
# - [ ] [Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/) · Medium
# - [ ] [Path with Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/) · Medium
# <!-- practice:end -->

# %%
s.finish(lookups=0, score=None, notes="")
