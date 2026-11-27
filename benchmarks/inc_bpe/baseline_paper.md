# Exact Streaming Incremental Byte-Pair Tokenization: Linear-Time Rank-Ordered Two-Interval Compilation & Theoretical Differentiation

## 1. Formal Problem Definition and Prefix-Consistency Guarantee

Let $\Sigma = \{0, 1, \dots, 255\}$ be the byte alphabet and $\mathcal{R} = ((a_0, b_0), \dots, (a_{R-1}, b_{R-1}))$ be a priority-ordered sequence of merge rules defining a proper/canonical vocabulary $\mathcal{V}$ of size $M = 256 + R$ (Sennrich et al., 2016; Kudo & Richardson, 2018; Zouhar et al., 2023; Berglund & van der Merwe, 2023). Rule $r \in \{0, \dots, R-1\}$ maps $(a_r, b_r) \to t = 256 + r$ with predecessor $p(t) = a_r$, successor $s(t) = b_r$, rank $\operatorname{rank}(t) = r$, and byte length $\ell(t) = \ell(a_r) + \ell(b_r)$.

Given a streaming byte sequence $x_{1:n} \in \Sigma^*$, priority-ranked BPE repeatedly merges the adjacent pair of minimum rank (breaking ties left-to-right), yielding $\theta(x_{1:n}) = (t_1, \dots, t_m)$. An **exact streaming incremental BPE transducer** (Jiang & Gong, 2026) processes one byte $x_n$ per step via `append(x_n)` returning $H[n] = t_m$, while `tokens()` reconstructs $\theta(x_{1:n})$ matching batch BPE at every byte prefix $1 \le k \le n$.

## 2. Formal Differentiation from Jiang & Gong (2026) and Linear-Time $O(M+R)$ Compilation

Orienting each merge token $t \ge 256$ to its right operand $s(t)$ forms a successor forest $\mathcal{F}_{\text{suc}}$ rooted at byte atoms. A merge candidate $t$ recognized by the Aho-Corasick automaton (Aho & Corasick, 1975; double-array layout in `arXiv:2207.13870`) is valid at position $n$ iff $k = H[n - \ell(s(t))]$ satisfies:
$$\operatorname{Accept}(t, k) \iff k = p(t) \;\lor\; \exists c \in \operatorname{Child}_{\mathcal{F}_{\text{suc}}}(p(t)) \text{ s.t. } k \in \operatorname{Desc}(c) \land \operatorname{rank}(c) > \operatorname{rank}(t).$$

### How Rank-Ordered Two-Interval Compilation Differs from Jiang & Gong's (2026) Prefix Last-Token Condition
In Jiang & Gong (`arXiv:2605.30813`), the *Prefix Last-Token Condition* uses standard DFS subtree intervals $[L_u, R_u)$ to test **unconstrained ancestry** $k \in \operatorname{Desc}(p(t))$ in $O(1)$ time, **but** because sibling order in their general successor forest is arbitrary, checking the **strict priority condition** ($\operatorname{rank}(c) > \operatorname{rank}(t)$ for the specific child subtree $c \in \operatorname{Child}(p(t))$ containing $k$) still requires locating which child $c$ contains $k$ at runtime via centroid decomposition / binary search ($O(\log \Delta)$ or $O(\log^2 L)$ per step).
- **Concrete Running Example**: Suppose predecessor $a = p(t)$ has three successor children $c_1, c_2, c_3$ with merge ranks $r(c_1)=5, r(c_2)=12, r(c_3)=20$, and candidate $t$ has rank $r(t)=12$. Under an arbitrary DFS order $(c_2, c_3, c_1)$, the forbidden subtrees ($r(c) \le 12$, i.e., $c_1$ and $c_2$) are non-adjacent in DFS numbering, requiring runtime child lookup.
- **Our Rank-Ordered Two-Interval Theorem (Theorem 1)**: By enforcing **rank-ordered sibling traversal** ($r(c_1) < r(c_2) < r(c_3)$) during DFS indexing, all higher-or-equal priority (forbidden) children $\{c \in \operatorname{Child}(a) : r(c) \le r(t)\}$ are guaranteed to form a **single contiguous prefix** $[L_a + 1, q_t)$ of $a$'s descendant interval $[L_a + 1, R_a)$, while all admissible children $\{c \in \operatorname{Child}(a) : r(c) > r(t)\}$ collapse into **one contiguous suffix interval** $[q_t, R_a)$! Thus, the entire combined ancestry-and-priority predicate compiles into at most **two half-open intervals** per rule:
  $$\mathcal{I}_t = [L_{p(t)}, L_{p(t)} + 1) \;\cup\; [q_t, R_{p(t)}),$$
  completely eliminating runtime child-subtree search ($O(1)$ vs $O(\log \Delta)$) while using strictly $2R$ intervals ($4R$ integers) total.

### Theorem 2: Strict $O(M+R)$ Single-Pass Linear-Time Cutoff Sweep (Eliminating $O(R \log \Delta)$ Binary Search)
Instead of binary-searching `children[p(t)]` in $O(R \log \Delta)$ time during preprocessing, all cutoffs $q_t$ for $t = 256, \dots, 255+R$ can be computed in **strict $O(M+R)$ linear time** via a single increasing-rank sweep:
- Maintain an array `cutoff[u]` initialized to $L_u + 1$ for all $u \in \mathcal{V}$.
- For each rule $t = 256 + r$ in increasing rank order $r = 0, \dots, R-1$:
  1. **First** update `cutoff[s(t)] = R_t` (marking $t$'s own subtree under its successor parent $s(t)$ as forbidden for rules of rank $\ge r$);
  2. **Then** read $q_t = \text{cutoff}[p(t)]$ and record $\mathcal{I}_t = (L_{p(t)}, L_{p(t)}+1, q_t, R_{p(t)})$.
- Updating `cutoff[s(t)]` *before* reading `cutoff[p(t)]` automatically enforces strict equal-rank self-overlap exclusion ($r(c) > r(t)$) even when $p(t) = s(t)$!

## 3. Comprehensive Evaluation in `eval.py`
`eval.py` reports 7-pass randomized interleaved paired latency with exact Student-$t$ ($\text{df}=6$) 95% CIs, `tracemalloc` peak memory bytes, hardware/platform metadata (`ARM Cortex-X925`), cache-free mechanism isolation (`selected` vs `pointer_predicate_only` vs `centroid_search` vs `explicit_set`), and a synthetic deep-forest scaling sweep ($R \in [72, 512], D \in [4, 36]$) including a forced depth-$D$ predicate microbenchmark (`forced_depth_D_predicate_micro_speedup`). Write the paper as a standalone ICLR research article without referencing internal workspace filenames.
