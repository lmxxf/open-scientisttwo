# Exact Streaming Incremental Byte-Pair Tokenization: Rank-Ordered DFS Interval Compilation & Deep-Forest Scaling

## 1. Formal Problem Definition and Prefix-Consistency Semantics

Let $\Sigma = \{0, 1, \dots, 255\}$ denote the atomic byte alphabet and $\mathcal{R} = ((a_0, b_0), \dots, (a_{R-1}, b_{R-1}))$ be a priority-ordered sequence of merge rules defining a proper/canonical vocabulary $\mathcal{V}$ of size $M = 256 + R$ (Sennrich et al., 2016; Kudo & Richardson, 2018; Zouhar et al., 2023; Berglund & van der Merwe, 2023). Rule $r \in \{0, \dots, R-1\}$ maps the adjacent token pair $(a_r, b_r)$ to token ID $t = 256 + r$ with predecessor $p(t) = a_r$, successor $s(t) = b_r$, priority rank $\operatorname{rank}(t) = r$, and byte length $\ell(t) = \ell(a_r) + \ell(b_r)$.

Given a streaming byte sequence $x_{1:n} = x_1 x_2 \dots x_n \in \Sigma^*$, priority-ranked Byte-Pair Encoding repeatedly merges the available adjacent pair of minimum rank (breaking equal-rank ties from left to right) until no rule applies, producing the exact canonical tokenization $\theta(x_{1:n}) = (t_1, t_2, \dots, t_m)$.

An **exact streaming incremental BPE transducer** (Jiang & Gong, 2026) consumes one byte $x_n$ per step via `append(x_n)` and maintains prefix history $H[1..n]$ such that:
1. `append(x_n)` immediately returns the final token $H[n] = t_m$ of $\theta(x_{1:n})$;
2. `tokens()` reconstructs the complete exact sequence $\theta(x_{1:n})$ matching batch priority-ranked BPE at **every** byte prefix $1 \le k \le n$.

## 2. Successor Forest Predicate, Centroid Search, and the Two-Interval Compilation Theorem

Orienting each merge token $t \ge 256$ to its right operand $s(t)$ forms a successor forest $\mathcal{F}_{\text{suc}}$ rooted at the 256 byte atoms. An Aho-Corasick recognizer (Aho & Corasick, 1975; double-array engineering in `arXiv:2207.13870`) identifies suffix candidates ending at position $n$. A merge candidate $t$ is valid at position $n$ iff the predecessor token $k = H[n - \ell(s(t))]$ satisfies:
$$\operatorname{Accept}(t, k) \iff k = p(t) \;\lor\; \exists c \in \operatorname{Child}_{\mathcal{F}_{\text{suc}}}(p(t)) \text{ s.t. } k \in \operatorname{Desc}(c) \land \operatorname{rank}(c) > \operatorname{rank}(t).$$

- **Pointer Walk Baseline ($O(D)$ per candidate)**: Walks $u \leftarrow s(u)$ from $k$ up to $p(t)$, degrading linearly with successor forest depth $D$.
- **Centroid / Binary-Search Tree ($O(\log D)$ or $O(\log |\operatorname{Child}|)$, Jiang & Gong 2026)**: Uses DFS interval containment $L_{p(t)} < L_k < R_{p(t)}$ followed by binary/centroid search to locate the child subtree $c$ containing $k$.
- **Explicit Hash-Set Compilation ($O(RM)$ space)**: Materializes the set $A_t$ for each rule $t$, causing quadratic memory blowup on large/deep vocabularies.
- **Rank-Ordered DFS Two-Interval Compilation ($O(1)$ time, $2R$ intervals / $O(R)$ linear space)**:
  When `SufSucTree` traverses children $c \in \operatorname{Child}(a)$ in increasing merge-rank order $\operatorname{rank}(c)$, all higher-priority (forbidden) children with $\operatorname{rank}(c) \le \operatorname{rank}(t)$ form a contiguous prefix block $[L_a + 1, q_t)$ in preorder DFS numbering, while all admissible lower-priority children with $\operatorname{rank}(c) > \operatorname{rank}(t)$ collapse into a **single contiguous suffix interval** $[q_t, R_a)$! Including the singleton parent interval $[L_a, L_a + 1)$ (and updating `cutoff[s(t)] = R_t` prior to recording $t$ to enforce strict equal-rank overlap exclusion $p(t) = s(t)$), every candidate's acceptance region is exactly representable by at most **two half-open integer intervals**:
  $$\mathcal{I}_t = [L_{p(t)}, L_{p(t)} + 1) \;\cup\; [q_t, R_{p(t)}).$$

## 3. Comprehensive Multidimensional & Deep-Forest Evaluation Protocol

To eliminate confounding and rigorously establish both integrated and isolated causal gains, `eval.py` enforces a strict cache-free evaluation protocol and outputs rich multidimensional metrics in `*_metrics.json`:
1. **Randomized Interleaved Paired Passes & Student-$t$ 95% Confidence Intervals**:
   - Runs 7 randomized interleaved paired passes between the candidate model and pointer baseline, reporting median, mean, standard deviation, exact Student-$t$ ($\text{df}=6, t_{0.975}=2.4469$) 95% confidence intervals, raw paired sample arrays, paired difference CI, construction overhead, and net amortization break-even stream length.
2. **Cache-Free Predicate Isolation (`deep_forest_and_isolation.randomized_cache_free_isolation`)**:
   - Directly benchmarks `selected` (Two-Interval), `pointer_predicate_only` (same dense transitions & atomic fastpath, switching only the predicate to pointer walks), `centroid_search` ($O(\log D)$ DFS child binary search), `explicit_set` ($O(RM)$ frozenset), and `baseline_pointer` under identical cache-free conditions.
3. **Deep-Forest & Multi-Scale Stress Sweep (`deep_forest_and_isolation.deep_forest_scaling_sweep`)**:
   - Evaluates scaling across $(R, D) \in \{(72, 4), (160, 12), (320, 24), (512, 36)\}$, measuring how `Two-Interval` maintains constant $O(1)$ validation latency and compact $2R$ linear memory while `Pointer Walk` degrades with depth $D$ (yielding $1.5\times \sim 4\times+$ isolated predicate speedup on deep forests) and `Explicit Set` suffers $10\times \sim 35\times$ memory expansion.
