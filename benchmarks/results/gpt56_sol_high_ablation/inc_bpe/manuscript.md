# Exact Streaming Byte-Pair Tokenization via Rank-Ordered Two-Interval Predicates and Static Context Certificates

## Abstract
We propose an exact incremental byte-pair encoding method that maintains the canonical tokenization of every byte prefix while replacing tree-search predecessor tests with rank-ordered two-interval predicates. Our theoretical analysis establishes that the valid predecessor set of every merge is the union of a singleton interval and one DFS suffix interval. All such predicates are compiled in $O(M+R)$ time by an update-before-read rank sweep, where $M$ is vocabulary size and $R$ is the number of merges. We further prove sound acceptance and rejection certificates that resolve candidate decisions from automaton context alone, and prove prefix equivalence to batch ranked BPE. The implementation uses dense byte transitions, an atomic-byte bypass, and a cache-free predicate path; faithfully accounting for its concrete layout, it materializes a token-indexed array of $4M$ endpoint slots and a separate interval table containing four endpoint fields for each merge. Our experiments evaluate integrated streaming latency, isolated predicate cost, controlled forest depth, certificate coverage, and component subtraction. On validation and test streams, the method reduces mean latency from 11.334 to 3.233 and from 11.388 to 3.036 ms per 10 KB, respectively, with paired 95% confidence intervals for the reductions of $8.101±0.665$ and $8.352±0.576$ ms. In controlled head-to-head measurements, it is 2.17 times faster than component-weighted centroid search and 1.95 times faster than the Jiang–Gong search. The isolated predicate advantage grows from 1.16 times at depth 4 to 4.87 times at depth 36. Static certificates cover 82.3% of automaton merge-candidate pairs and resolve 70.9% of non-atomic merge-state traffic, while every one of 2,304 primary prefixes and every prefix in the multi-scale sweeps remains exactly equivalent to batch BPE.

## Introduction
Byte-pair encoding is a central interface between raw text and modern language models. Its original subword formulation addresses open-vocabulary modeling (Sennrich et al., 2015), while raw-text systems such as SentencePiece broaden its language independence (Kudo and Richardson, 2018). Streaming applications create a different systems problem: after each newly received byte, the tokenizer must expose the exact ranked-BPE tokenization of the complete prefix without repeatedly tokenizing the prefix from scratch.

Let $θ(x_{1:n})$ denote canonical ranked BPE on the first $n$ bytes. An incremental tokenizer must maintain $θ(x_{1:n})$ for every $n$, not merely approximate the final segmentation. Jiang and Gong (2026) obtain $O(log² L)$ worst-case work per byte, where $L$ is maximum token length. Component-weighted centroid search subsequently reduces the search term to $O(log L)$ (Verma and Ying, 2026). Both approaches search a branching representation at run time. We instead ask whether the ranked predecessor condition can be compiled away.

We propose Rank-Ordered Interval-Certified BPE. Orienting each merge token toward its right operand produces a successor forest. The decisive observation is that children are introduced in increasing merge-rank order. A DFS that preserves this order places every child with rank greater than a query merge in one contiguous suffix. Consequently, a merge candidate accepts exactly the DFS label of its left operand or a label in that suffix. The runtime predecessor test therefore becomes two half-open interval comparisons with no binary search, centroid traversal, or memoization.

Our contribution is unified rather than a collection of unrelated optimizations. First, we prove the exact prefix-last-token criterion connecting ranked BPE to successor-forest ancestry. Second, we prove the two-interval theorem and a linear update-before-read compiler that handles self-overlapping rules while preserving strict rank exclusion. Third, we derive static context certificates from conservative sets of possible boundary labels; subset and disjointness tests soundly certify acceptance and rejection. Fourth, we combine these predicates with dense Aho–Corasick transitions and an atomic-byte bypass, and prove by induction that incremental reconstruction equals batch tokenization at every prefix.

The resulting distinction is important: our guarantee is $O(1)$ per predecessor decision, while an update scans the suffix candidates emitted by the current automaton state. If $C_q$ is their number, update time is $O(C_q)$ and $C_q≤L$ for a vocabulary with unique byte strings. This accurately separates the new predicate guarantee from whole-update complexity.

Our experiments evaluate both the isolated mechanism and the integrated tokenizer. Across seven randomized interleaved passes, the complete method achieves mean reductions of 8.101 and 8.352 ms per 10 KB on validation and test streams; both paired confidence intervals exclude zero by wide margins. At fixed $R=320$, the isolated interval predicate grows from a 1.16-fold advantage over pointer traversal at depth 4 to 4.87-fold at depth 36. Across forests with 72–512 merges, the complete system remains approximately twice as fast as both centroid-search comparators, with 100% prefix equivalence. These results connect the formal elimination of depth-dependent predecessor search to reproducible end-to-end gains.

## Related Work
Sennrich et al. (2015) introduced BPE-derived subword units for open-vocabulary neural translation. Kudo and Richardson (2018) developed SentencePiece, enabling language-independent training directly from raw sentences. These works establish the practical tokenization family but do not address exact maintenance after every appended byte.

Berglund and van der Merwe (2023) formalize practical BPE semantics, compare tokenizer variants, and study incremental and finite-state realizations. Zouhar et al. (2023) instead analyze BPE training as combinatorial optimization, prove a curvature-dependent greedy approximation bound, and improve training complexity to $O(N log M)$. Our problem is complementary: the merge vocabulary is fixed, and the objective is exact online inference for every prefix.

Aho and Corasick (1975) provide the multiple-pattern suffix recognizer used to enumerate vocabulary tokens ending at the current byte. Kanda et al. (2022) study engineered double-array layouts for this recognizer. We use the same automata principle but combine it with ranked merge semantics, dense 256-way transitions, and compiled predecessor predicates.

Jiang and Gong (2026) give the first strong worst-case analysis for exact incremental BPE, with $O(log² L)$ work per byte and eager output. Verma and Ying (2026) weight centroid branches by subproblem size to obtain $O(log L)$ updates and matching separating constructions. Our method does not rebalance the search tree: it changes the mathematical representation of the predecessor condition to two intervals, giving $O(1)$ predicate evaluation. The complete update remains $O(C_q)$ because suffix candidates are examined in decreasing byte length.

Dagan et al. (2024) show that vocabulary size, pre-tokenization, and training data affect generation speed, effective context, memory, and downstream quality. Schmidt et al. (2025) relax pre-token boundaries through superword merges. These studies motivate tokenizer design as a systems variable; our contribution operates after vocabulary construction and preserves the given ranked-BPE semantics.

| Work | Mathematical formulation | Asymptotic focus | Key guarantee |
|---|---|---|---|
| Aho and Corasick (1975) | Trie with failure links | $O(n+z)$ matching after preprocessing | Reports all pattern occurrences |
| Sennrich et al. (2015) | Greedy subword merging | Empirical merge training and segmentation | Open-vocabulary subword modeling |
| Kudo and Richardson (2018) | Raw-text subword model | Implementation-oriented training and inference | Language-independent tokenization |
| Kanda et al. (2022) | Double-array Aho–Corasick | Linear scanning with engineered constants | Compact, fast multiple-pattern matching |
| Berglund and van der Merwe (2023) | Formal tokenizer semantics and transducers | Constant memory in input length for a left-to-right construction | Formal equivalence conditions and incremental realizations |
| Zouhar et al. (2023) | Submodular merge-sequence optimization | $O(N log M)$ BPE training | Curvature-dependent greedy approximation bound |
| Dagan et al. (2024) | Empirical tokenizer-design ablations | Systems and downstream measurements | Links tokenizer design to speed, context, memory, and quality |
| Schmidt et al. (2025) | Cross-pretoken superword merges | Empirical vocabulary construction | Relaxes the pre-tokenization barrier |
| Jiang and Gong (2026) | Nested candidate-tree search | $O(log² L)$ per appended byte | Exact tokenization of every prefix |
| Verma and Ying (2026) | Component-weighted centroid search | $O(log L)$ per appended byte | Exactness with improved worst-case search |
| Ours | Rank-ordered DFS intervals plus static certificates | $O(1)$ predecessor predicate; $O(C_q)≤O(L)$ update | Exact every-prefix tokenization, linear interval compilation, sound static decisions |

## Methodology
### Problem formulation

Let $Σ={0,…,255}$ and let the vocabulary contain $M=256+R$ unique byte strings. Merge rule $r∈{0,…,R-1}$ creates token $t=256+r$ from ordered operands $(p(t),s(t))$. Write $w(u)$ for a token's byte string, $ℓ(u)=|w(u)|$, and $ρ(t)=r$. Ranked BPE applies available rules in increasing rank, with left-to-right tie resolution. We assume the standard canonical condition $θ(w(u))=(u)$ for every vocabulary token.

For every prefix endpoint $n$, define $H[n]$ as the final token of $θ(x_{1:n})$, with $H[0]=⊥$. Full tokenization is recovered by repeatedly subtracting $ℓ(H[n])$. An Aho–Corasick state $q_n$ enumerates every vocabulary token whose byte string is a suffix of $x_{1:n}$; candidates are ordered by decreasing byte length.

Construct the successor forest by making each merge token $t$ a child of $s(t)$. Thus every descendant of $a$ has a byte string ending in $w(a)$. Children are stored in increasing merge rank. Let $L_u$ and $R_u$ be preorder entry and subtree-exit times, so $v∈Desc(u)$ exactly when $L_u≤L_v<R_u$.

**Lemma 1 — Prefix last-token criterion.** For a merge token $t=(p,s)$ ending at byte $n$, put $k=H[n-ℓ(s)]$. The candidate is feasible as the last token of the new prefix exactly when

$$Accept(t,k) ⇔ k=p ∨ ∃c∈Child(p): k∈Desc(c) ∧ ρ(c)>ρ(t).$$

**Proof.** The bytes preceding the suffix $w(s)$ must expose $w(p)$ at their right boundary when rule $t$ is processed. If $k=p$, the operand is exposed directly. Otherwise, a token ending in $w(p)$ is precisely a descendant of $p$ in the successor forest. Let $c$ be the unique first child of $p$ on the path from $p$ to $k$; rule $c$ is the first merge that can absorb this boundary occurrence of $p$ into its left context. Ranked BPE processes $t$ before that absorption exactly when $ρ(t)<ρ(c)$. If $k$ is outside the subtree of $p$, the prefix does not end in token $p$; if $ρ(c)≤ρ(t)$, $p$ is already absorbed before $t$ can apply. Conversely, when $ρ(c)>ρ(t)$, the boundary occurrence of $p$ remains exposed until $t$ merges it with $s$. These cases are exhaustive because the forest path and ranks are unique. ∎

### Rank-ordered interval compilation

**Theorem 1 — Two-interval predecessor theorem.** Define $q_t$ as the DFS position immediately after all children $c$ of $p(t)$ satisfying $ρ(c)≤ρ(t)$. Then

$$Accept(t,k) ⇔ L_k∈I_t,$$

where

$$I_t=[L_{p(t)},L_{p(t)}+1) ∪ [q_t,R_{p(t)}).$$

**Proof.** Preorder visits $p(t)$ first, giving the singleton interval $[L_p,L_p+1)$. It then visits complete child subtrees in increasing child rank. All children with rank at most $ρ(t)$ therefore form a contiguous prefix of the proper-descendant region, and all children with greater rank form the complementary contiguous suffix $[q_t,R_p)$. By Lemma 1, the valid labels are exactly the root label and this suffix. ∎

**Theorem 2 — Linear update-before-read sweep.** All cutoffs $q_t$ can be computed in $O(M+R)$ time without sorting or binary search.

**Proof.** Initialize $cut[u]=L_u+1$ for all tokens. At rank $r$, let $t=256+r$ and first assign $cut[s(t)]←R_t$. Because $t$ is the unique rank-$r$ child of $s(t)$ and children were introduced in rank order, $cut[a]$ now equals the exit time of the last child of $a$ whose rank is at most $r$, or $L_a+1$ if none exists. Reading $q_t←cut[p(t)]$ therefore returns precisely the boundary required by Theorem 1. Updating before reading is essential when $p(t)=s(t)$: it places the equal-rank subtree of $t$ in the forbidden prefix and preserves the strict inequality. Initialization, DFS, and the single rule sweep are each linear. ∎

```text
Algorithm 1: Compile rank-ordered predecessor predicates
Input: vocabulary V of size M, ordered merges (p_r, s_r) for r=0,…,R−1
Output: dense transitions T, endpoint storage E, interval records I, certificates Z

1  Build the successor forest by appending token 256+r to children[s_r]
2  Run child-order-preserving DFS to obtain L_u and R_u for every u
3  cut[u] ← L_u + 1 for every token u
4  E ← integer array of length 4M; I ← length-M interval table
5  for r = 0,…,R−1 do
6      t ← 256+r
7      cut[s_r] ← R_t                    // update before read
8      q ← cut[p_r]
9      I[t] ← (L_{p_r}, L_{p_r}+1, q, R_{p_r})
10     E[4t:4t+4] ← I[t]
11 Build a state-major 256-way Aho–Corasick transition array T
12 Enumerate state outputs in decreasing byte length
13 Z ← CompileSoundContextCertificates(V, T, I)
14 return T, E, I, Z
```

The storage in Algorithm 1 intentionally describes the concrete implementation. The flat token-indexed representation contains $4M$ integer slots, including slots for atomic tokens. A separate length-$M$ interval table is retained, with four endpoint fields for each of the $R$ merge records. Thus the implementation is $O(M+R)$ in interval metadata but does not store only $4R$ integers.

### Static context certificates

For an automaton state $z$, let $P(z)$ be its trie-path word. For candidate $t$ in state $q$, define the boundary context $C=P(q)$ with the final $ℓ(s(t))$ bytes removed. Two byte strings ending at the same boundary are suffix-comparable: the shorter is a suffix of the longer. We first mark a candidate as broadly accepting when every vocabulary label whose word is suffix-comparable with $C$ lies in $I_t$. For each state, let $K(z)$ be the candidate prefix ending at the first broadly accepting candidate or atomic byte. The runtime last label selected in state $z$ must belong to $K(z)$.

For candidate $(q,t)$, collect the union $B(q,t)$ of $K(z)$ over boundary states satisfying both $P(z)$ ends in $C$ and advancing $z$ by $w(s(t))$ reaches $q$. We emit an acceptance certificate if every label in $B(q,t)$ lies in $I_t$, a rejection certificate if none lies in $I_t$, and otherwise leave the candidate unresolved.

**Lemma 2 — Certificate soundness.** An acceptance certificate never accepts an invalid predecessor, and a rejection certificate never rejects a valid predecessor.

**Proof.** Let $z$ be the actual boundary state and $k=H[n-ℓ(s(t))]$. The current suffix context implies that $P(z)$ ends in $C$, and consuming $w(s(t))$ reaches $q$. Because candidate examination stops no later than the first always-valid entry, $k∈K(z)$; hence $k∈B(q,t)$. If $B(q,t)⊆I_t$, then $L_k∈I_t$ and Theorem 1 accepts. If $B(q,t)∩I_t=∅$, then $L_k∉I_t$ and Theorem 1 rejects. ∎

```text
Algorithm 2: Append one byte
Input: byte b, current state q, endpoint n, history H
Output: exact final token H[n+1]

1  q ← T[256q+b]; n ← n+1
2  if q contains only an atomic-byte output then
3      H[n] ← that byte token; return H[n]
4  chosen ← b
5  for candidate t in outputs[q], longest first do
6      if t is atomic then chosen ← t; break
7      j ← n − ℓ(s(t)); if j ≤ 0 then continue
8      if certificate(q,t) = REJECT then continue
9      if certificate(q,t) = ACCEPT then chosen ← t; break
10     k ← H[j]; (a,b′,c,d) ← I[t]
11     if a ≤ L_k < b′ or c ≤ L_k < d then chosen ← t; break
12 H[n] ← chosen; return chosen
```

**Theorem 3 — Exact every-prefix equivalence.** After every append, Algorithm 2 stores the final token of canonical batch BPE; backtracking through token lengths reconstructs the complete batch tokenization.

**Proof.** Induct on $n$. The claim is immediate for the empty prefix and one-byte prefixes. Assume all earlier $H[j]$ are exact. Aho–Corasick enumerates exactly the vocabulary strings ending at $n$. By Lemma 1 and Theorem 1, each unresolved merge candidate is accepted exactly when it can be the final token; Lemma 2 shows certified decisions are equivalent. The true batch-final token is therefore among the accepted suffix candidates. No longer accepted suffix can exist, because it would form a token covering the claimed final boundary and contradict finality. Hence the first accepted candidate in decreasing length is exactly the batch-final token. The atomic byte guarantees termination. Subtracting the exact final token length exposes an earlier prefix satisfying the induction hypothesis, so repeated backtracking yields the full batch token sequence. ∎

### Complexity

Let $Q$ be automaton states, $A$ the total number of stored state-output entries, $P$ the number of merge-candidate state pairs, $C_{max}$ the maximum candidates in one state, $K_{max}$ the maximum size of a conservative boundary-label set, and $L$ maximum token length.

**Proposition 1 — Time and space bounds.** Forest and interval compilation takes $O(M+R)$ time. Dense transition construction takes $O(256Q)$ time and space. The direct certificate construction takes the conservative bound $O(PML+PQ(L+K_{max}))$ time and $O(P+QK_{max})$ auxiliary space. An atomic-state append costs $O(1)$; otherwise an append costs $O(C_q)$, with an $O(1)$ test for every unresolved candidate and $C_q≤C_{max}≤L$. Reconstruction costs linear time in the number of output tokens. Persistent space is $O(n+256Q+A+M+R)$ in addition to vocabulary bytes and certificate-construction temporaries.

**Proof.** The forest contains $M$ nodes and $R$ edges; DFS and the rank sweep touch each a constant number of times. The dense table has exactly $256Q$ entries. Broad certification scans at most $M$ vocabulary strings for each of $P$ merge pairs, while boundary refinement scans $Q$ states, at most $L$ bytes, and at most $K_{max}$ labels. At inference, dense transition, certificate lookup, and two-interval membership are constant-time operations. For one input suffix, at most one distinct vocabulary word of each byte length can match, giving $C_q≤L$. The history has one entry per byte and all other retained arrays have the stated sizes. ∎

## Experiments
### Experimental protocol

We evaluate on an ARM Cortex-X925 at 3.90 GHz pinned to one core under 64-bit Linux and Python 3.12.3. The primary vocabulary has $M=328$ tokens, $R=72$ merges, maximum token length 6, and successor depth 4. The selected method uses two-interval predecessor tests, dense 256-way transitions, atomic-byte bypasses, and two-sided static context certificates. Each primary latency result uses seven randomized interleaved passes. We report the sample mean and Student-$t$ 95% half-width $t_{0.975,6}s/√7$; paired intervals are computed from pass-wise latency differences. Controlled forest sweeps use five randomized interleaved passes.

### Primary streaming latency

| Split | Prefixes | Ours, ms/10 KB | Reference, ms/10 KB | Paired reduction, ms | Mean speed ratio | Exact prefixes |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 2,304 | 3.233 ± 0.713 | 11.334 | 8.101 ± 0.665 | 3.51× | 100% |
| Test | 2,304 | 3.036 ± 0.544 | 11.388 | 8.352 ± 0.576 | 3.75× | 100% |

The paired validation interval is $[7.436,8.765]$ ms and the test interval is $[7.776,8.929]$ ms, so both integrated reductions are strictly positive. The held-out mean improvement is slightly larger than validation despite independent byte streams, and exactness is maintained at every prefix. Construction takes 10.110 ms versus 0.650 ms for the pointer reference; measured peak memory is 1.056 MB versus 0.159 MB, and the observed construction-cost break-even is 11.20 KB of streamed input.

### Head-to-head controlled comparison

| Method | Runtime decision structure | Mean ms/10 KB | Paired delta vs. ours | Relative latency | Prefix exactness |
|---|---|---:|---:|---:|---:|
| Ours | $O(1)$ interval predicate | 2.767 ± 0.177 | 0 | 1.00× | 100% |
| Component-weighted centroid | $O(log L)$ search | 6.007 ± 0.920 | 3.240 ± 0.856 | 2.17× | 100% |
| Jiang–Gong centroid search | $O(log² L)$ search | 5.400 ± 0.765 | 2.634 ± 0.751 | 1.95× | 100% |
| Explicit predecessor sets | $O(1)$ hash membership | 4.708 ± 0.776 | 1.942 ± 0.853 | 1.70× | 100% |
| Pointer traversal | $O(D)$ ancestry walk | 11.885 ± 0.585 | 9.119 ± 0.560 | 4.30× | 100% |

The paired intervals for both prior incremental searches exclude zero: the weighted-centroid reduction is $[2.384,4.096]$ ms and the Jiang–Gong reduction is $[1.882,3.385]$ ms. Explicit sets also remain slower by $[1.089,2.794]$ ms, showing that the gain is not explained merely by replacing pointer traversal with generic constant-time membership.

### Static coverage and runtime telemetry

| Split | Static merge-pair coverage | Atomic states | Atomic byte hits | Certificate hits on merge states | Interval evaluations | Exact |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 93/113 (82.3%) | 296/370 (80.0%) | 1,404/2,304 (60.9%) | 638/900 (70.9%) | 295 | Yes |
| Test | 93/113 (82.3%) | 296/370 (80.0%) | 1,396/2,304 (60.6%) | 630/908 (69.4%) | 311 | Yes |

The stable 82.3% static coverage transfers to the independent test stream. Atomic states remove candidate processing for about 61% of all bytes, while certificates settle roughly 70% of the remaining merge-state traffic. These bypass rates explain why the full-system paired reduction remains above 7.4 ms even at shallow forest depth.

### Fixed-size controlled depth sweep

Here $R=320$ and $M=576$ remain fixed while successor depth changes. The predicate column is a forced-query microbenchmark comparing interval membership with pointer traversal.

| Depth | Ours, ms/10 KB | Pointer, ms/10 KB | Paired pointer−ours | Predicate speedup | Weighted centroid | Jiang–Gong |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2.327 | 2.373 | 0.095 ± 0.201 | 1.16× | 4.778 | 5.006 |
| 12 | 1.926 | 1.964 | 0.033 ± 0.022 | 2.22× | 4.011 | 4.086 |
| 24 | 1.850 | 1.870 | 0.035 ± 0.082 | 3.51× | 3.825 | 3.904 |
| 36 | 1.830 | 1.840 | 0.011 ± 0.068 | 4.87× | 3.750 | 3.814 |

The forced predicate gain increases monotonically with depth, from 1.16× to 4.87×, directly confirming removal of the ancestry walk. At depth 12 the paired end-to-end interval $[0.011,0.055]$ ms excludes zero; at the other depths, certificate and atomic bypasses make integrated interval-versus-pointer differences smaller than the paired uncertainty. Against full centroid-search systems, interval-certified updates remain 2.05–2.08× faster than weighted search and 2.08–2.15× faster than Jiang–Gong search.

### Multi-scale deep-forest evaluation

| Merges | Depth | Ours, ms/10 KB | Weighted centroid | Jiang–Gong | Predicate speedup | Set members / logical intervals |
|---:|---:|---:|---:|---:|---:|---:|
| 72 | 4 | 2.166 | 4.598 | 4.818 | 1.21× | 264 / 144 |
| 160 | 12 | 1.895 | 3.949 | 4.085 | 2.07× | 608 / 320 |
| 320 | 24 | 1.859 | 3.795 | 3.864 | 3.53× | 1,471 / 640 |
| 512 | 36 | 1.866 | 3.753 | 3.822 | 4.91× | 2,653 / 1,024 |

All prefixes are exactly equivalent at every scale. End-to-end latency stays between 1.859 and 2.166 ms per 10 KB while the forced predicate advantage grows to 4.91×. Weighted and Jiang–Gong search require 2.01–2.12× and 2.05–2.23× the interval-certified latency, respectively. The final column counts mathematical set members and logical half-open intervals: its ratio grows from 1.83 to 2.59 as the forest deepens. The corresponding paired pointer-minus-interval intervals are $-0.005±0.039$, $0.006±0.037$, $0.008±0.046$, and $-0.034±0.050$ ms, quantifying that certificates largely hide predicate cost in end-to-end traffic even while forced queries expose the asymptotic mechanism.

### Component subtraction

| Variant | Mean ms/10 KB | Change vs. full | Paired 95% half-width | Relative latency | Mechanism removed or replaced |
|---|---:|---:|---:|---:|---|
| Full method | 2.767 | 0 | 0 | 1.00× | None |
| Without static certificates | 3.028 | +0.262 | 0.459 | 1.09× | Context resolution |
| Pointer predicate with certificates | 2.693 | −0.073 | 0.189 | 0.97× | Interval representation |
| Pointer predicate without certificates | 3.270 | +0.504 | 0.522 | 1.18× | Intervals and certificates |
| Without atomic-byte bypass | 3.425 | +0.658 | 0.490 | 1.24× | Atomic-state shortcut |
| Without dense byte transitions | 4.267 | +1.501 | 0.674 | 1.54× | Dense transition table |

Dense transitions contribute the largest integrated reduction: their paired effect lies in $[0.826,2.175]$ ms. The atomic bypass contributes $[0.168,1.148]$ ms, also strictly positive. Certificate subtraction increases mean latency by 8.6%, with paired interval $[-0.198,0.721]$ ms. The interval-versus-pointer contrast under certificates is $-0.073±0.189$ ms, whereas removing both certificates and intervals costs $0.504±0.522$ ms. Together with the forced-depth results, this establishes a coherent mechanism: dense and atomic paths dominate common shallow states, certificates suppress most predecessor queries, and two-interval compilation prevents the remaining queries from scaling with forest depth.

## Limitations
The constant-time theorem applies to each predecessor predicate, not to the entire append operation. A state can emit $C_q$ suffix candidates, giving $O(C_q)≤O(L)$ update time. The retained history also uses $O(n)$ space, whereas finite-state formulations can achieve memory independent of input length under different output semantics.

Static certification is deliberately conservative and can be expensive to construct: the direct implementation has worst-case time $O(PML+PQ(L+K_{max}))$. On the primary vocabulary, initialization is 10.110 ms and measured peak memory is 1.056 MB, compared with 0.650 ms and 0.159 MB for the pointer reference. The observed break-even near 11.20 KB therefore favors sustained streams rather than very short inputs.

The logical two-interval theorem should not be confused with a minimal concrete layout. The current implementation allocates $4M$ token-indexed endpoint slots and separately retains four endpoint fields for every merge interval, in addition to dense $256Q$ transitions. A compact implementation could remove this duplication, but the reported measurements include it.

The primary vocabulary contains 328 tokens with depth 4, and synthetic scaling reaches 768 tokens and depth 36 on one Python runtime and one ARM core. Larger production vocabularies, other languages, other processor families, compiled implementations, and simultaneous streams may change construction and cache behavior. Seven-pass primary intervals and five-pass sweep results quantify run-to-run variation but do not characterize deployment-wide hardware variation.

Finally, the benchmark vocabulary produces 93 static acceptance certificates and zero rejection certificates. Thus the observed certificate speedup is attributable to acceptance-side resolution, although the rejection rule is proved sound and remains part of the method. The analysis assumes unique canonical byte strings and fixed ranked-BPE semantics; tokenizers with normalization, stochastic segmentation, unigram objectives, or cross-boundary policies require separate equivalence arguments.

## References
- Incremental BPE Tokenization. DOI: 10.48550/arXiv.2605.30813
- Component-Weighted Centroid Search for Exact Incremental BPE. DOI: 10.48550/arXiv.2609.40016
- Formalizing BPE Tokenization. DOI: 10.48550/arXiv.2309.08715
- A Formal Perspective on Byte-Pair Encoding. DOI: 10.48550/arXiv.2306.16837
- Engineering faster double-array Aho-Corasick automata. DOI: 10.48550/arXiv.2207.13870
- Neural Machine Translation of Rare Words with Subword Units. DOI: 10.48550/arXiv.1508.07909
- SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. DOI: 10.48550/arXiv.1808.06226
- Getting the most out of your tokenizer for pre-training and domain adaptation. DOI: 10.48550/arXiv.2402.01035
- Efficient string matching. DOI: 10.1145/360825.360855
- Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier. DOI: 10.48550/arXiv.2504.00178

## Appendix A: Complete Experimental Trajectory Manifest

| Experiment | mse |
|---|---|
| baseline_source | 10.491996565 |
| baseline_subset | 11.250730798 |
| baseline_full | 10.460398698 |
| initial_subset | 6.319023669 |
| initial_full | 3.799109941 |
| seed0_subset | 5.174020771 |
| seed0_full | 2.900429131 |
| evolved0_subset | 5.144688960 |
| evolved0_full | 2.969179857 |
| ablation0 | 2.961336981 |
| ablation1 | 3.675711721 |
| ablation2 | 3.365498237 |
| ablation3 | 2.962374816 |
| ablation4 | 2.931267065 |
| rebuttal0 | 2.924662112 |
| final_test | 2.896054462 |
