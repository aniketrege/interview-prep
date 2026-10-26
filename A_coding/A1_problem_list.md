# A1 problem list: NeetCode 150 by topic

All 150 NeetCode 150 problems, grouped by the guide's A1 topics (one notebook each), with difficulty, a **B75** flag for Blind 75 problems, and the second-pass week from the guide's "DSA focus" column (shifted for week 1 = Oct 26, 2026). Generated from the NeetCode problem data ([neetcode.io/practice](https://neetcode.io/practice); file `.problemSiteData.json` in [neetcode-gh/leetcode@d5ebb3e](https://github.com/neetcode-gh/leetcode/tree/d5ebb3e)). Links go to LeetCode.

**How to use it.** `tools/today.py` assigns one problem per weekday from week 1 (schedule.yaml, week 1 = Oct 26, 2026): weeks 1-7 follow the A1 pattern notebooks in order (one pattern every 2-3 days; pick Easy or familiar Medium problems from that section, Blind 75 first), and weeks 7-16 are a second pass with the remaining Mediums and Hards in the weeks listed below. Use `A1.00_session_template` for every problem. Anonymous, self-reported candidate reports (2025-26, e.g. Glassdoor) describe research-intern loops that open with a timed online screen of LeetCode-medium problems, which is why this starts on day 1. [Grind 75](https://www.techinterviewhandbook.org/grind75/) (Tech Interview Handbook) is an alternative ordering with a customizable schedule ("A better Blind 75 you can customize").

Second-pass weeks: 7 arrays/hashing/two pointers · 8 sliding window, stacks, binary search · 9 holiday light review · 10 linked lists, trees · 11 tries, heaps, intervals · 12 graphs · 13 advanced graphs (Dijkstra), backtracking · 14 1-D DP (+ greedy) · 15 2-D DP · 16 timed mixed sets (bit manipulation and math: first pass in week 7, more in week 16).

Totals: 150 problems (28 Easy, 101 Medium, 21 Hard), 75 in Blind 75.

## A1.1 · Arrays & hashing, two pointers, sliding window, prefix sums

Notebook: `A_coding/A1.01_arrays_hashing_two_pointers.ipynb`

### Arrays & Hashing · week 7 (9)

- [ ] [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/) · Easy · **B75** · wk 7
- [ ] [Valid Anagram](https://leetcode.com/problems/valid-anagram/) · Easy · **B75** · wk 7
- [ ] [Two Sum](https://leetcode.com/problems/two-sum/) · Easy · **B75** · wk 7
- [ ] [Group Anagrams](https://leetcode.com/problems/group-anagrams/) · Medium · **B75** · wk 7
- [ ] [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) · Medium · **B75** · wk 7
- [ ] [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/) · Medium · **B75** · wk 7
- [ ] [Valid Sudoku](https://leetcode.com/problems/valid-sudoku/) · Medium · wk 7
- [ ] [Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/) · Medium · **B75** · wk 7
- [ ] [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) · Medium · **B75** · wk 7

### Two Pointers · week 7 (5)

- [ ] [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/) · Easy · **B75** · wk 7
- [ ] [Two Sum II Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) · Medium · wk 7
- [ ] [3Sum](https://leetcode.com/problems/3sum/) · Medium · **B75** · wk 7
- [ ] [Container With Most Water](https://leetcode.com/problems/container-with-most-water/) · Medium · **B75** · wk 7
- [ ] [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) · Hard · wk 7

### Sliding Window · week 8 (6)

- [ ] [Best Time to Buy And Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) · Easy · **B75** · wk 8
- [ ] [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) · Medium · **B75** · wk 8
- [ ] [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/) · Medium · **B75** · wk 8
- [ ] [Permutation In String](https://leetcode.com/problems/permutation-in-string/) · Medium · wk 8
- [ ] [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) · Hard · **B75** · wk 8
- [ ] [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) · Hard · wk 8

## A1.2 · Stacks, monotonic stacks, queues, deques

Notebook: `A_coding/A1.02_stacks_queues.ipynb`

### Stack · week 8 (7)

- [ ] [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) · Easy · **B75** · wk 8
- [ ] [Min Stack](https://leetcode.com/problems/min-stack/) · Medium · wk 8
- [ ] [Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/) · Medium · wk 8
- [ ] [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/) · Medium · wk 8
- [ ] [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/) · Medium · wk 8
- [ ] [Car Fleet](https://leetcode.com/problems/car-fleet/) · Medium · wk 8
- [ ] [Largest Rectangle In Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) · Hard · wk 8

## A1.3 · Binary search (incl. on the answer space)

Notebook: `A_coding/A1.03_binary_search.ipynb`

### Binary Search · week 8 (7)

- [ ] [Binary Search](https://leetcode.com/problems/binary-search/) · Easy · wk 8
- [ ] [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/) · Medium · wk 8
- [ ] [Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/) · Medium · wk 8
- [ ] [Find Minimum In Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) · Medium · **B75** · wk 8
- [ ] [Search In Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) · Medium · **B75** · wk 8
- [ ] [Time Based Key Value Store](https://leetcode.com/problems/time-based-key-value-store/) · Medium · wk 8
- [ ] [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) · Hard · wk 8

## A1.4 · Linked lists

Notebook: `A_coding/A1.04_linked_lists.ipynb`

### Linked List · week 10 (11)

- [ ] [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) · Easy · **B75** · wk 9
- [ ] [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) · Easy · **B75** · wk 9
- [ ] [Reorder List](https://leetcode.com/problems/reorder-list/) · Medium · **B75** · wk 9
- [ ] [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) · Medium · **B75** · wk 9
- [ ] [Copy List With Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/) · Medium · wk 9
- [ ] [Add Two Numbers](https://leetcode.com/problems/add-two-numbers/) · Medium · wk 9
- [ ] [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/) · Easy · **B75** · wk 9
- [ ] [Find The Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/) · Medium · wk 9
- [ ] [LRU Cache](https://leetcode.com/problems/lru-cache/) · Medium · wk 9
- [ ] [Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) · Hard · **B75** · wk 9
- [ ] [Reverse Nodes In K Group](https://leetcode.com/problems/reverse-nodes-in-k-group/) · Hard · wk 9

## A1.5 · Trees (BST, traversal, LCA) and tries

Notebook: `A_coding/A1.05_trees_tries.ipynb`

### Trees · week 10 (15)

- [ ] [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/) · Easy · **B75** · wk 9
- [ ] [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/) · Easy · **B75** · wk 9
- [ ] [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/) · Easy · wk 9
- [ ] [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/) · Easy · wk 9
- [ ] [Same Tree](https://leetcode.com/problems/same-tree/) · Easy · **B75** · wk 9
- [ ] [Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/) · Easy · **B75** · wk 9
- [ ] [Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) · Medium · **B75** · wk 9
- [ ] [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/) · Medium · **B75** · wk 9
- [ ] [Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/) · Medium · wk 9
- [ ] [Count Good Nodes In Binary Tree](https://leetcode.com/problems/count-good-nodes-in-binary-tree/) · Medium · wk 9
- [ ] [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/) · Medium · **B75** · wk 9
- [ ] [Kth Smallest Element In a Bst](https://leetcode.com/problems/kth-smallest-element-in-a-bst/) · Medium · **B75** · wk 9
- [ ] [Construct Binary Tree From Preorder And Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) · Medium · **B75** · wk 9
- [ ] [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/) · Hard · **B75** · wk 9
- [ ] [Serialize And Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) · Hard · **B75** · wk 9

### Tries · week 11 (3)

- [ ] [Implement Trie Prefix Tree](https://leetcode.com/problems/implement-trie-prefix-tree/) · Medium · **B75** · wk 11
- [ ] [Design Add And Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/) · Medium · **B75** · wk 11
- [ ] [Word Search II](https://leetcode.com/problems/word-search-ii/) · Hard · **B75** · wk 11

## A1.6 · Heaps / top-k / k-way merge, intervals

Notebook: `A_coding/A1.06_heaps_intervals.ipynb`

### Heap / Priority Queue · week 11 (7)

- [ ] [Kth Largest Element In a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/) · Easy · wk 11
- [ ] [Last Stone Weight](https://leetcode.com/problems/last-stone-weight/) · Easy · wk 11
- [ ] [K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/) · Medium · wk 11
- [ ] [Kth Largest Element In An Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) · Medium · wk 11
- [ ] [Task Scheduler](https://leetcode.com/problems/task-scheduler/) · Medium · wk 11
- [ ] [Design Twitter](https://leetcode.com/problems/design-twitter/) · Medium · wk 11
- [ ] [Find Median From Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) · Hard · **B75** · wk 11

### Intervals · week 11 (6)

- [ ] [Insert Interval](https://leetcode.com/problems/insert-interval/) · Medium · **B75** · wk 11
- [ ] [Merge Intervals](https://leetcode.com/problems/merge-intervals/) · Medium · **B75** · wk 11
- [ ] [Non Overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/) · Medium · **B75** · wk 11
- [ ] [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/) · Easy · **B75** · wk 11
- [ ] [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/) · Medium · **B75** · wk 11
- [ ] [Minimum Interval to Include Each Query](https://leetcode.com/problems/minimum-interval-to-include-each-query/) · Hard · wk 11

## A1.7 · Graphs: BFS/DFS, topological sort, union-find, Dijkstra

Notebook: `A_coding/A1.07_graphs.ipynb`

### Graphs · week 12 (13)

- [ ] [Number of Islands](https://leetcode.com/problems/number-of-islands/) · Medium · **B75** · wk 12
- [ ] [Clone Graph](https://leetcode.com/problems/clone-graph/) · Medium · **B75** · wk 12
- [ ] [Max Area of Island](https://leetcode.com/problems/max-area-of-island/) · Medium · wk 12
- [ ] [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/) · Medium · **B75** · wk 12
- [ ] [Surrounded Regions](https://leetcode.com/problems/surrounded-regions/) · Medium · wk 12
- [ ] [Rotting Oranges](https://leetcode.com/problems/rotting-oranges/) · Medium · wk 12
- [ ] [Walls And Gates](https://leetcode.com/problems/walls-and-gates/) · Medium · wk 12
- [ ] [Course Schedule](https://leetcode.com/problems/course-schedule/) · Medium · **B75** · wk 12
- [ ] [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/) · Medium · wk 12
- [ ] [Redundant Connection](https://leetcode.com/problems/redundant-connection/) · Medium · wk 12
- [ ] [Number of Connected Components In An Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/) · Medium · **B75** · wk 12
- [ ] [Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/) · Medium · **B75** · wk 12
- [ ] [Word Ladder](https://leetcode.com/problems/word-ladder/) · Hard · wk 12

### Advanced Graphs · week 13 (6)

- [ ] [Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/) · Hard · wk 13
- [ ] [Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/) · Medium · wk 13
- [ ] [Network Delay Time](https://leetcode.com/problems/network-delay-time/) · Medium · wk 13
- [ ] [Swim In Rising Water](https://leetcode.com/problems/swim-in-rising-water/) · Hard · wk 13
- [ ] [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/) · Hard · **B75** · wk 13
- [ ] [Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/) · Medium · wk 13

## A1.8 · Backtracking

Notebook: `A_coding/A1.08_backtracking.ipynb`

### Backtracking · week 13 (9)

- [ ] [Subsets](https://leetcode.com/problems/subsets/) · Medium · wk 13
- [ ] [Combination Sum](https://leetcode.com/problems/combination-sum/) · Medium · **B75** · wk 13
- [ ] [Permutations](https://leetcode.com/problems/permutations/) · Medium · wk 13
- [ ] [Subsets II](https://leetcode.com/problems/subsets-ii/) · Medium · wk 13
- [ ] [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/) · Medium · wk 13
- [ ] [Word Search](https://leetcode.com/problems/word-search/) · Medium · **B75** · wk 13
- [ ] [Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/) · Medium · wk 13
- [ ] [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) · Medium · wk 13
- [ ] [N Queens](https://leetcode.com/problems/n-queens/) · Hard · wk 13

## A1.9 · 1-D / 2-D dynamic programming (+ greedy)

Notebook: `A_coding/A1.09_dynamic_programming.ipynb`

### 1-D Dynamic Programming · week 14 (12)

- [ ] [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) · Easy · **B75** · wk 14
- [ ] [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/) · Easy · wk 14
- [ ] [House Robber](https://leetcode.com/problems/house-robber/) · Medium · **B75** · wk 14
- [ ] [House Robber II](https://leetcode.com/problems/house-robber-ii/) · Medium · **B75** · wk 14
- [ ] [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/) · Medium · **B75** · wk 14
- [ ] [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/) · Medium · **B75** · wk 14
- [ ] [Decode Ways](https://leetcode.com/problems/decode-ways/) · Medium · **B75** · wk 14
- [ ] [Coin Change](https://leetcode.com/problems/coin-change/) · Medium · **B75** · wk 14
- [ ] [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/) · Medium · **B75** · wk 14
- [ ] [Word Break](https://leetcode.com/problems/word-break/) · Medium · **B75** · wk 14
- [ ] [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/) · Medium · **B75** · wk 14
- [ ] [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/) · Medium · wk 14

### Greedy · week 14 (8)

- [ ] [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) · Medium · **B75** · wk 14
- [ ] [Jump Game](https://leetcode.com/problems/jump-game/) · Medium · **B75** · wk 14
- [ ] [Jump Game II](https://leetcode.com/problems/jump-game-ii/) · Medium · wk 14
- [ ] [Gas Station](https://leetcode.com/problems/gas-station/) · Medium · wk 14
- [ ] [Hand of Straights](https://leetcode.com/problems/hand-of-straights/) · Medium · wk 14
- [ ] [Merge Triplets to Form Target Triplet](https://leetcode.com/problems/merge-triplets-to-form-target-triplet/) · Medium · wk 14
- [ ] [Partition Labels](https://leetcode.com/problems/partition-labels/) · Medium · wk 14
- [ ] [Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/) · Medium · wk 14

### 2-D Dynamic Programming · week 15 (11)

- [ ] [Unique Paths](https://leetcode.com/problems/unique-paths/) · Medium · **B75** · wk 15
- [ ] [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/) · Medium · **B75** · wk 15
- [ ] [Best Time to Buy And Sell Stock With Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) · Medium · wk 15
- [ ] [Coin Change II](https://leetcode.com/problems/coin-change-ii/) · Medium · wk 15
- [ ] [Target Sum](https://leetcode.com/problems/target-sum/) · Medium · wk 15
- [ ] [Interleaving String](https://leetcode.com/problems/interleaving-string/) · Medium · wk 15
- [ ] [Longest Increasing Path In a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/) · Hard · wk 15
- [ ] [Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/) · Hard · wk 15
- [ ] [Edit Distance](https://leetcode.com/problems/edit-distance/) · Medium · wk 15
- [ ] [Burst Balloons](https://leetcode.com/problems/burst-balloons/) · Hard · wk 15
- [ ] [Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/) · Hard · wk 15

## A1.10 · Bit manipulation (+ math & geometry), complexity analysis

Notebook: `A_coding/A1.10_bits_complexity.ipynb`

### Bit Manipulation · week 16 (7)

- [ ] [Single Number](https://leetcode.com/problems/single-number/) · Easy · wk 16
- [ ] [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/) · Easy · **B75** · wk 16
- [ ] [Counting Bits](https://leetcode.com/problems/counting-bits/) · Easy · **B75** · wk 16
- [ ] [Reverse Bits](https://leetcode.com/problems/reverse-bits/) · Easy · **B75** · wk 16
- [ ] [Missing Number](https://leetcode.com/problems/missing-number/) · Easy · **B75** · wk 16
- [ ] [Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/) · Medium · **B75** · wk 16
- [ ] [Reverse Integer](https://leetcode.com/problems/reverse-integer/) · Medium · wk 16

### Math & Geometry · week 16 (8)

- [ ] [Rotate Image](https://leetcode.com/problems/rotate-image/) · Medium · **B75** · wk 16
- [ ] [Spiral Matrix](https://leetcode.com/problems/spiral-matrix/) · Medium · **B75** · wk 16
- [ ] [Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/) · Medium · **B75** · wk 16
- [ ] [Happy Number](https://leetcode.com/problems/happy-number/) · Easy · wk 16
- [ ] [Plus One](https://leetcode.com/problems/plus-one/) · Easy · wk 16
- [ ] [Pow(x, n)](https://leetcode.com/problems/powx-n/) · Medium · wk 16
- [ ] [Multiply Strings](https://leetcode.com/problems/multiply-strings/) · Medium · wk 16
- [ ] [Detect Squares](https://leetcode.com/problems/detect-squares/) · Medium · wk 16
