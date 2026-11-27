# Compiling Exact Streaming BPE into Rank-Ordered Intervals and Static Context Certificates

## Abstract
We propose an exact streaming byte-pair encoding algorithm that compiles rank-constrained predecessor acceptance into constant-time interval tests and specializes selected tests using static context certificates. Our theoretical analysis establishes that a rank-ordered depth-first traversal represents every admissible predecessor region by at most two intervals, and that all regions can be compiled in $O(M+R)$ time for $M$ vocabulary tokens and $R$ merge rules. We additionally prove that conservative context certificates safely eliminate acceptance checks without runtime memoization. Combined with dense byte transitions, precomputed candidate sequences, and an atomic-token fast path, the method achieves a held-out median latency of 2.806 ms per 10,000 bytes against 11.425 ms for a paired pointer-based baseline, a 4.072-fold integrated speedup. A cache-free acceptance-mode control isolates a 1.119-fold improvement, while forced-depth predicate measurements reach 13.71-fold acceleration at successor depth 36. Every evaluated primary-workload prefix matches independent batch BPE exactly. These results establish a unified compilation approach whose predicate guarantees are independent of forest depth, while distinguishing its isolated mechanism gains from the larger benefits of integrated dispatch engineering.

## Introduction
Streaming tokenization requires more than recognizing vocabulary strings. Appending one byte can change the segmentation of the current suffix, so an incremental tokenizer must respect the same merge priorities and overlap decisions as batch byte-pair encoding (BPE). We study the exact-prefix problem: after every appended byte, the maintained representation must reconstruct precisely the batch tokenization of the entire current prefix.

The central computation is a rank-constrained ancestry query. For a candidate merge token, the tokenizer inspects a previously computed prefix's last token and determines whether it is the required predecessor or belongs to an admissible successor subtree. Pointer traversal makes this decision depend on forest depth; searching child subtrees retains a runtime search operation. We instead choose the traversal order to make the priority constraint geometrically contiguous.

We propose rank-ordered interval compilation with static context acceptance certificates. Our theoretical analysis establishes two results. First, ordering successor children by increasing merge rank makes the accepted child subtrees a single suffix of a depth-first interval. Together with the predecessor itself, the acceptance region therefore requires at most two intervals. Second, a forward rank sweep computes all interval cutoffs in $O(M+R)$ time, including strict equal-rank exclusion for self-overlapping merges. We further prove that a conservative vocabulary-level context test can certify selected state–candidate pairs before streaming begins.

Our experiments evaluate these mechanisms within an integrated implementation. On the held-out workload, the proposed method reduces median streaming latency from 11.425 to 2.806 ms per 10,000 bytes, yielding a 4.072-fold speedup. Holding the selected implementation's other switches fixed while replacing interval acceptance with pointer acceptance yields a smaller, isolated 1.119-fold improvement. At successor depth 36, a direct predicate stress test measures a 13.71-fold speedup. This distinction is essential: interval compilation establishes the depth-independent predicate guarantee, whereas dense transitions, precomputed dispatch, static certificates, and atomic specialization contribute jointly to end-to-end performance.

Our contribution is an exact compilation strategy for a fixed canonical ranked vocabulary. We do not change vocabulary training or merge semantics. We explicitly account for candidate enumeration, automaton storage, certificate construction, and initialization amortization rather than extending a linear forest-compilation theorem to the entire tokenizer.

## Related Work
We position our method relative to eight prior works spanning BPE semantics, incremental tokenization, vocabulary design, and string-matching implementation.

**Incremental BPE.** Jiang and Gong's *Incremental BPE Tokenization* (2026; [arXiv:2605.30813](https://arxiv.org/abs/2605.30813)) is the closest comparison. It maintains exact tokenizations for all input prefixes, reports $O(n\log^2 B)$ processing for maximum token length $B$, and additionally studies eager emission of irrevocable token boundaries. We retain the exact-prefix objective and build on the predecessor/successor interpretation of candidate validity. Our formal differentiation concerns the acceptance representation: rank-ordered traversal makes all admissible child subtrees contiguous, eliminating child identification from the predicate. The forward sweep also eliminates a separate binary search for each cutoff. This is a constant-time predicate result; our implementation still enumerates suffix candidates and therefore does not establish constant-time byte processing independent of $B$. Our search comparator implements child-interval binary search, not the full published algorithm, and our measurements do not constitute a direct comparison with that work's released implementation.

**Formal tokenization semantics.** *Formalizing BPE Tokenization* ([arXiv:2309.08715](https://arxiv.org/abs/2309.08715)) and *A Formal Perspective on Byte-Pair Encoding* ([arXiv:2306.16837](https://arxiv.org/abs/2306.16837)) place BPE on a precise mathematical footing. Their relevance is the distinction between ranked merging and unrestricted longest-string matching. Our recognizer supplies matching suffixes, but acceptance additionally depends on earlier prefix states and merge ranks. Formally, matching a spelling is necessary but insufficient: a candidate $t$ must satisfy $H[n-\ell(s(t))]\in\mathcal A_t$. Our contribution compiles this contextual predicate while preserving the ranked semantics. It does not replace BPE with greedy longest-match segmentation or claim a new vocabulary-learning objective.

**Subword learning and tokenizer systems.** Sennrich et al.'s *Neural Machine Translation of Rare Words with Subword Units* ([arXiv:1508.07909](https://arxiv.org/abs/1508.07909)) motivates BPE as a subword representation for rare words. Its primary design variable is the learned segmentation vocabulary; ours is the execution of a fixed ranked merge system. Kudo and Richardson's *SentencePiece* ([arXiv:1808.06226](https://arxiv.org/abs/1808.06226)) provides language-independent subword tokenization and detokenization. We address a narrower computational contract: exact byte-prefix maintenance under supplied merge rules. Normalization, training choices, and non-BPE segmentation objectives remain outside our algorithm. *Getting the Most out of Your Tokenizer for Pre-training and Domain Adaptation* ([arXiv:2402.01035](https://arxiv.org/abs/2402.01035)) concerns tokenizer choices for model training and adaptation. Such changes alter the vocabulary and potentially downstream statistical behavior. Our transformation instead preserves $\theta(x)$ for a fixed vocabulary, so its evaluation target is latency, memory, and exactness rather than language-model quality.

**Multi-pattern recognition and representation engineering.** Aho and Corasick's *Efficient String Matching* ([DOI:10.1145/360825.360855](https://doi.org/10.1145/360825.360855)) supplies the multi-pattern recognition foundation. Failure-inherited outputs identify vocabulary suffixes, but they do not encode BPE priority compatibility. We therefore separate recognition from acceptance. *Engineering Faster Double-Array Aho–Corasick Automata* ([arXiv:2207.13870](https://arxiv.org/abs/2207.13870)) studies efficient automaton representation. Our implementation uses a failure-completed dense 256-way transition array rather than a double-array layout. This contributes predictable transition lookup at a substantial space cost. The interval theorem is independent of this representation choice, and the transition ablation measures its practical contribution separately.

These comparisons identify three distinct operations: learning a token inventory, recognizing matching strings, and validating ranked contextual merges. Our theoretical contribution targets the third; our implementation combines it with established recognition and dispatch techniques.

## Methodology
### Problem and notation

Let $\Sigma=\{0,\ldots,255\}$ and let $\mathcal V$ contain the byte atoms and $R$ merge tokens, so $M=256+R$. Rule $r$ creates token $t=256+r$ from $(a_r,b_r)$, with predecessor $p(t)=a_r$, successor $s(t)=b_r$, rank $\rho(t)=r$, spelling $w(t)=w(a_r)w(b_r)$, and length $\ell(t)=|w(t)|$. Operands precede their merge token, spellings are unique, and the vocabulary is canonical under the specified merge procedure. Batch BPE repeatedly applies the currently available minimum-rank adjacent merge, resolving equal-rank occurrences from left to right.

For input $x_{1:n}$, write $\theta(x_{1:n})$ for the resulting token sequence and $H[n]$ for its last token. Set $H[0]=\bot$. An append operation returns $H[n]$; reconstruction repeatedly reads $H[j]$ and replaces $j$ by $j-\ell(H[j])$.

The successor forest has edge $t\rightarrow s(t)$ for every merge token. Its roots are byte atoms. Let $\operatorname{Desc}(u)$ include $u$. The ranked predecessor region is

$$\mathcal A_t=\{p(t)\}\;\cup\!\bigcup_{\substack{c\in\operatorname{Child}(p(t))\\\rho(c)>\rho(t)}}\operatorname{Desc}(c).$$

We use the standard prefix-last-token characterization of canonical ranked BPE: among vocabulary suffixes of the current input, the longest admissible candidate is the last batch token. An atom is always admissible. A merge candidate $t$ is admissible exactly when $j=n-\ell(s(t))>0$ and $H[j]\in\mathcal A_t$. This characterization is the semantic basis inherited from incremental BPE; our new results concern its compilation and specialization.

### Rank-ordered two-interval compilation

Traverse each root's successor tree in preorder, visiting children by increasing merge rank. Let $L_u$ be the preorder index of $u$ and $U_u$ the exclusive end of its subtree. Since rule identifiers increase with rank, inserting successor children in rule order already supplies the required sibling order.

**Theorem 1 (two-interval representation).** For every merge token $t$, there exists $q_t\in[L_{p(t)}+1,U_{p(t)}]$ such that

$$k\in\mathcal A_t\iff L_k\in[L_{p(t)},L_{p(t)}+1)\cup[q_t,U_{p(t)}).$$

**Proof.** Preorder gives every child subtree a contiguous interval. Increasing-rank sibling traversal places all children with $\rho(c)\le\rho(t)$ before all children with $\rho(c)>\rho(t)$. The forbidden children therefore occupy a prefix of the proper descendants, and the admissible children occupy the remaining suffix, beginning at $q_t$. The predecessor itself occupies the singleton interval $[L_{p(t)},L_{p(t)}+1)$. These are precisely the two components of $\mathcal A_t$. If no child is admissible, the second interval is empty. $\square$

For example, suppose the predecessor's children have ranks 5, 12, and 20, and the candidate rank is 12. The first two subtrees are excluded and the third is accepted. Sorting the traversal by rank makes this decision a single cutoff rather than a runtime child search.

**Theorem 2 (linear cutoff sweep).** Given rank-ordered forest intervals, all $q_t$ can be constructed in $O(M+R)$ time and $O(M+R)$ space, including the forest traversal and stored acceptance regions.

**Proof.** Initialize a conceptual cutoff $C[u]=L_u+1$. Process merge tokens in increasing rank. Before recording token $t$'s region, set $C[s(t)]=U_t$ and then read $q_t=C[p(t)]$. After processing rank $r$, the cutoff at node $u$ equals the exclusive end of the last child subtree whose rank is at most $r$, or $L_u+1$ if there is no such child. This invariant follows because children are visited in increasing rank and each update advances exactly one parent's cutoff to the corresponding subtree end. Reading after updating includes the current child in the forbidden prefix when $p(t)=s(t)$, enforcing the strict inequality. Each forest node and rule requires constant work. $\square$

```text
Algorithm 1: Compile ranked acceptance regions
Input: canonical vocabulary V and increasing-rank rules
Build successor children in rule order
Run preorder DFS to obtain L[u], U[u]
For every token u: C[u] <- L[u] + 1
For each rule r = 0, ..., R-1:
    t <- 256 + r
    a <- p(t); b <- s(t)
    C[b] <- U[t]                 # update before reading
    q <- C[a]
    I[t] <- (L[a], L[a]+1, q, U[a])
    offset[t] <- length(b)
Return I, offset, L
```

Each rule stores four endpoint scalars, representing two intervals. Thus interval metadata occupies $4R$ scalar entries, excluding token-indexed container overhead. A packed representation would require $16R$ bytes with 32-bit endpoints; the evaluated implementation stores Python tuples and integers, so this packed byte count is an accounting reference rather than its measured allocation.

### Static context acceptance certificates

The suffix recognizer is an Aho–Corasick automaton over vocabulary spellings. For state $z$, let $W_z$ be its trie-path spelling. Candidate outputs include failure-inherited matches and are stably sorted by decreasing byte length.

For an output merge candidate $t$, define

$$C_{z,t}=W_z[:-\ell(s(t))],$$

and the conservative compatibility domain

$$\mathcal K(C)=\{k\in\mathcal V: w(k)\text{ ends with }C\ \lor\ C\text{ ends with }w(k)\}.$$

We mark $(z,t)$ certified when every token in $\mathcal K(C_{z,t})$ belongs to $\mathcal A_t$. This is universal testing over a conservative domain, not enumeration of reachable tokenization states.

**Proposition 3 (certificate soundness).** Whenever state $z$ is active, $t$ is an output candidate, and $j=n-\ell(s(t))>0$, a certified pair $(z,t)$ may be accepted without reading $H[j]$.

**Proof.** The current input ends in $W_z$, so the prefix ending at $j$ ends in $C_{z,t}$. Its last token $k=H[j]$ and this context end at the same position. If $\ell(k)\ge|C_{z,t}|$, then $w(k)$ ends with the context. Otherwise the context ends with $w(k)$. Hence $k\in\mathcal K(C_{z,t})$. Certification guarantees $k\in\mathcal A_t$, which is the exact acceptance condition. The argument includes tokens extending beyond the state's left boundary. $\square$

The selected method enables certificates and interval acceptance. It examines at most 64 eligible states and at most 256 state–merge-candidate pairs. States are considered in automaton index order and candidates in their precomputed order; each attempted pair consumes one pair-budget unit, whether certification succeeds or fails. Certification scans the vocabulary and stops at the first compatible token outside the acceptance region. Uncertified candidates use the exact interval test. A generic constructor defaults certification to disabled; the proposed selected configuration explicitly enables it. There is no runtime memoization or query-dependent acceptance cache.

### Streaming execution

We compile failure-completed transitions into a flat unsigned-integer array indexed by $256z+b$. Dense transitions and the atomic fast path are enabled. A state receives an atomic shortcut only when its complete output list contains exactly one token and that token is a byte atom. Candidate rows store $(t,\ell(s(t)),L_{p(t)},L_{p(t)}+1,q_t,U_{p(t)},\mathrm{certified})$. The history stores token identifiers alone.

```text
Algorithm 2: Append one byte b
z <- transition[256*z + b]
n <- n + 1
If complete_output[z] contains exactly one byte atom a:
    H.append(a); return a
For (t, off, lo, end, q, hi, certified) in candidates[z]:
    If t is a byte atom:
        H.append(t); return t
    j <- n - off
    If j > 0:
        If certified:
            H.append(t); return t
        x <- L[H[j]]
        If lo <= x < end or q <= x < hi:
            H.append(t); return t
Raise an error if no atomic suffix exists

Algorithm 3: Reconstruct the current tokenization
result <- empty list; j <- n
While j > 0:
    t <- H[j]
    result.append(t)
    j <- j - length(t)
Return reverse(result)
```

The standalone validity operation accepts atoms immediately, rejects a merge query if its predecessor position is out of history bounds or contains $\bot$, and otherwise applies the same interval predicate. It does not consult certificates. Pointer and explicit-set acceptance modes provide alternative implementations for controlled comparisons. The selected method contains neither candidate truncation after a certified output nor static rejection certificates.

**Proposition 4 (semantic preservation).** Under the canonical prefix-last-token characterization, Algorithms 1–3 preserve batch BPE at every prefix.

**Proof.** Induct on prefix length. The empty history is correct. Assuming all earlier last-token entries are correct, Theorem 1 makes every ordinary interval decision identical to the ranked predecessor predicate. Proposition 3 makes every certificate shortcut identical to an accepted predicate decision. An atomic shortcut has no competing merge output. Thus the first admissible candidate in decreasing-length order is the same last token prescribed by the characterization. Reconstruction removes this token and follows an already correct earlier prefix, yielding the full batch segmentation. $\square$

### Time and space complexity

Let $S$ be the number of automaton states, $B$ the maximum spelling length, $T=\sum_{t\in\mathcal V}\ell(t)$, $Q$ the total number of materialized output entries, and $K_n$ the number of candidates inspected at byte $n$. The acceptance compiler alone takes $O(M+R)$ time and space. Dense transition construction takes $O(256S)$ time and storage. With four-byte unsigned entries, its array payload is $1024S$ bytes.

Output ordering costs $O(\sum_z d_z\log(d_z+1))$, where $d_z$ is state $z$'s output count. Because spellings are unique, at most one matching suffix has each length, giving $d_z\le B$. Candidate rows require $O(Q)$ space. A conservative bound for the recognizer construction used here is $O(T+SB+Q)$ time, accounting for failure-pointer traversal and inherited outputs.

Constructing state spellings for certification costs $O(SB)$ time and temporary space. With $P\le256$ attempted pairs, certificate testing costs at most $O(PMB)$ time. These costs are separate from linear interval compilation. A conservative total initialization bound is

$$O\!\left(T+SB+Q+\sum_z d_z\log(d_z+1)+256S+M+R+PMB\right).$$

Dense transitions make recognition lookup constant-time per byte. Each interval decision is worst-case $O(1)$, so streaming costs

$$O\!\left(n+\sum_{i=1}^n K_i\right)\subseteq O(nB).$$

Pointer acceptance instead permits $O(K_iD)$ predicate work at successor depth $D$. Certificates reduce executed checks without changing the candidate-count worst case. Reconstruction takes $O(m)$ time and output space for $m$ current tokens. Including vocabulary spellings, automaton structures, candidates, temporary certification strings, and history, peak asymptotic space is $O(T+256S+Q+M+R+SB+n)$. History itself grows as $O(n)$.

## Experiments
### Evaluation protocol

Our experiments evaluate exactness, integrated streaming latency, acceptance-mode isolation, synthetic depth scaling, component removal, and initialization–memory tradeoffs. The primary vocabulary contains 256 byte atoms and 72 learned merges: $M=328$, maximum token length 6, and maximum successor depth 4. Its learning corpus contains English prose, Chinese text, and Python code. The full validation and held-out workloads each contain three 768-byte streams, totaling 2,304 bytes; preliminary subsets contain three 160-byte streams, totaling 480 bytes.

Validation and held-out generation use distinct pseudorandom seeds, 410 and 991. They draw from the same small source corpus used to learn the vocabulary. Thus held-out denotes separately generated sequences, not an independent-domain corpus. An additional validation generation uses a seed offset of 10,000.

For every primary-workload prefix, the tokenizer's reconstructed tokens are compared with an independent linked-list, priority-heap batch implementation, and decoded bytes are compared with the input prefix. All 20 recorded evaluations pass exactness. This establishes tested agreement, while the formal guarantees depend on the canonical-vocabulary assumptions stated above.

Streaming timings exclude model construction and token reconstruction. Seven passes randomize candidate/baseline order; model instances are constructed before their timed streaming loops. Latencies are normalized to 10,000 bytes. We report medians and arithmetic means separately. For seven measurements, the reported two-sided 95% mean-confidence halfwidth is $2.4469s/\sqrt7$. Paired differences use the seven within-pass baseline-minus-candidate differences. Confidence intervals describe repeated timing variability on the fixed workload rather than variation across independently sampled corpora.

### Primary held-out comparison

| Method | Median ms/10,000 bytes | Mean ms/10,000 bytes | 95% mean-CI halfwidth | Exact prefixes |
|---|---:|---:|---:|---:|
| Paired pointer baseline | 11.4253 | 11.3919 | — | Reference comparison |
| Proposed method | 2.8061 | 2.7895 | 0.0611 | 2,304/2,304 |

The median ratio is 4.0715, corresponding to a 75.44% latency reduction. The mean paired saving is $8.6024\pm0.1023$ ms per 10,000 bytes. Median-derived throughput is 3.5636 MB/s using decimal units. The selected method's full-validation median is 2.7824 ms, with mean $2.7766\pm0.0793$ ms and paired median speedup 4.0556. An additional validation generation records 2.8030 ms and a paired speedup of 4.0205.

These integrated comparisons include several improvements over the pointer baseline: candidate sorting moves out of the byte loop, transitions become dense, predicate metadata is pre-unpacked, static certificates bypass selected checks, and atomic-only states have a direct path. The entire fourfold gain therefore cannot be assigned to interval membership alone.

### Cache-free acceptance-mode isolation

The following held-out diagnostic uses seven randomized passes over five variants. The pointer-mode control retains the selected implementation's other switches, including static certification, dense transitions, and atomic dispatch. Changing acceptance mode also changes its predicate-dispatch branch; it is therefore a close acceptance-mode control rather than a perfectly identical instruction-level loop.

| Variant | Median ms/10,000 bytes | Mean ± 95% CI halfwidth |
|---|---:|---:|
| Proposed interval mode | 2.8131 | 2.8082 ± 0.0376 |
| Pointer acceptance in selected implementation | 3.1481 | 3.1474 ± 0.1142 |
| Child-interval binary-search comparator | 4.9372 | 4.9343 ± 0.0361 |
| Explicit acceptance-set comparator | 4.2961 | 4.4043 ± 0.2821 |
| Pointer baseline | 11.4156 | 11.4968 ± 0.2713 |

Interval mode is 1.1191-fold faster than the pointer-mode control, a 10.64% median latency reduction. It is 1.7551-fold faster than child-interval search and 1.5272-fold faster than the explicit-set comparator. The latter two comparisons include representation and execution differences beyond the acceptance predicate. The corresponding validation acceptance-mode ratio is 1.0918. No variant of the selected implementation uses runtime acceptance memoization.

### Multi-scale successor-forest evaluation

Synthetic vocabularies vary merge count and successor depth together. Each synthetic stream is truncated to at most 2,400 bytes. Streaming entries below are medians of five repetitions; the variants execute in a fixed order within each repetition. These measurements are descriptive scaling observations, not randomized paired estimates with confidence intervals.

| Merges $R$ | Tokens $M$ | Depth $D$ | Interval ms | Pointer-mode ms | Child-search ms | Set ms | Pointer/interval | Forced-depth predicate speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 72 | 328 | 4 | 2.4465 | 2.3911 | 4.8375 | 3.8627 | 0.977 | 2.66 |
| 160 | 416 | 12 | 2.2261 | 2.7137 | 4.0433 | 3.3454 | 1.219 | 5.51 |
| 320 | 576 | 24 | 2.1183 | 2.8265 | 3.9668 | 3.3301 | 1.334 | 9.98 |
| 512 | 768 | 36 | 2.1390 | 2.7500 | 3.8260 | 3.2624 | 1.286 | 13.71 |

All streaming latencies use ms per 10,000 bytes. At depth 4, interval mode is slightly slower than the pointer-mode control, despite its better worst-case predicate bound. At larger depths, its observed streaming advantage ranges from 1.219 to 1.334. Depth alone does not determine end-to-end latency: candidate frequency, context, certificate coverage, and transition behavior also matter.

The direct predicate microbenchmark invokes validity 3,000 times with a deliberately constructed history containing a deepest-chain token. It forces depth-sensitive ancestry work without representing a naturally occurring stream history. Its 2.66-to-13.71-fold progression supports the predicted depth sensitivity of pointer traversal and depth independence of interval checks; it is not an end-to-end speedup claim. The synthetic timing study does not record the same exhaustive prefix verification as the primary evaluations.

### Component ablations

Each ablation changes the indicated component relative to the selected configuration. These full-validation evaluations are separate runs, each with its own interleaved pointer baseline; they do not provide paired confidence intervals for differences between ablation rows.

| Configuration | Median ms/10,000 bytes | Mean ± 95% CI halfwidth | Construction ms | Traced peak bytes |
|---|---:|---:|---:|---:|
| Selected method | 2.7824 | 2.7766 ± 0.0793 | 5.9705 | 572,901 |
| Remove static context certificates | 2.8726 | 2.8394 ± 0.0902 | 0.8152 | 557,044 |
| Remove atomic fast path | 3.1704 | 3.1489 ± 0.0852 | 5.9421 | 572,901 |
| Remove dense transitions | 3.5699 | 3.5616 ± 0.0999 | 5.7360 | 191,005 |
| Replace intervals with explicit sets | 3.0468 | 3.0289 ± 0.0932 | 7.0851 | 592,969 |
| Replace intervals with pointer acceptance | 3.1420 | 3.1121 ± 0.0848 | 5.9237 | 572,901 |

Removing certificates increases median streaming latency by 3.24%, while reducing construction time substantially. This is evidence of a modest steady-state benefit, not a large independently established certificate effect. Removing the atomic path increases median latency by 13.94%, and removing dense transitions increases it by 28.31%. The dense-transition result makes clear that established automaton engineering contributes materially to the integrated gain. Precomputed candidate ordering and tuple unpacking lack separate removal measurements, so their individual contributions remain unquantified.

Exploratory alternatives also inform selection. A dual token/coordinate-history variant records 3.6897 ms on full validation, a rank-sweep variant 3.0204 ms, a candidate-pruning variant 2.8152 ms, and a static-rejection variant 2.9374 ms. None surpasses the selected method's 2.7824 ms in its recorded full evaluation. Coordinate history, candidate truncation, and rejection certificates are therefore absent from the proposed selected implementation. These observations do not imply that every variant differs in only one component.

### Memory footprint and initialization amortization

| Measured configuration | Current traced bytes | Peak traced bytes | Construction ms |
|---|---:|---:|---:|
| Original pointer implementation, validation | 158,376 | 158,960 | 0.3104 |
| Proposed method, validation | 554,916 | 572,901 | 5.9705 |
| Proposed method, held-out | 554,916 | 572,901 | 5.9695 |
| Proposed method without certificates | 554,804 | 557,044 | 0.8152 |
| Proposed method without dense transitions | 175,928 | 191,005 | 5.7360 |
| Proposed method with explicit sets | 574,984 | 592,969 | 7.0851 |

Memory tracing covers construction and processing of the first stream with Python allocation tracing; it is not process resident memory or total machine memory. The selected method's held-out peak is approximately 559.5 KiB. Its dense array alone requires $1024S$ bytes when entries occupy four bytes. Interval metadata contains 288 endpoint scalars for 72 rules; at four bytes per scalar, a packed endpoint-only representation would occupy 1,152 bytes, excluding all Python overhead.

The synthetic comparison below reports logical acceptance storage rather than measured allocation bytes.

| $R$ | Explicit-set memberships | Interval records, at most $2R$ | Endpoint scalars, $4R$ | Memberships / interval records |
|---:|---:|---:|---:|---:|
| 72 | 264 | 144 | 288 | 1.83 |
| 160 | 608 | 320 | 640 | 1.90 |
| 320 | 1,471 | 640 | 1,280 | 2.30 |
| 512 | 2,653 | 1,024 | 2,048 | 2.59 |

A set membership and an interval record have different representations, so the final column is not a byte-memory reduction factor. On the primary vocabulary, explicit acceptance sets contain only 99 memberships, illustrating that interval representation need not minimize logical item count for every small instance. Its guarantee is linear-size rank-constrained representation, rather than universal byte superiority.

Held-out construction takes 5.9695 ms versus 0.3625 ms for the paired baseline constructor. The recorded streaming-only amortization estimate is 6.5053 kB. Token reconstruction takes 63.50 microseconds in aggregate across the three held-out streams. These measurements favor reusing a constructed tokenizer over sufficiently long streams. Disabling certificates reduces construction to 0.8152 ms, making the selected certificate configuration less attractive when initialization dominates.

### Hardware and reproducibility

| Item | Recorded setting |
|---|---|
| Processor metadata | ARM Cortex-X925, 3.90 GHz, pinned single core |
| Architecture | aarch64 |
| Operating system | Linux 6.17.0-1014-nvidia |
| C library | glibc 2.39 |
| Python | 3.12.3 |
| Primary timing | Seven randomized interleaved passes |
| Isolation timing | Seven passes, variant-order seed 77 |
| Primary order seeds | Validation 101; held-out 202 |
| Constructor timing | Median of five measurements |
| Scaling timing | Median of five repetitions per variant |
| Predicate stress test | 3,000 direct validity calls per mode |
| Certificate limits | 64 eligible states; 256 attempted pairs |
| Selected acceptance | Two intervals; static certificates enabled |
| Selected dispatch | Dense transitions; atomic shortcut enabled |
| Runtime acceptance memoization | Absent |

Hardware entries reproduce the recorded environment description; the timing records do not independently establish clock stability or affinity enforcement. No accelerator measurement is reported.

## Limitations
The empirical scope is small. The primary vocabulary has 328 tokens, and synthetic scaling reaches 768 tokens with successor depth 36. These experiments do not establish performance for production vocabularies containing tens or hundreds of thousands of tokens. Held-out streams use a different generation seed but share the vocabulary-learning source corpus, so the results do not demonstrate independent-domain generalization.

The strongest unconditional algorithmic result is the constant-time acceptance predicate after linear forest compilation. The streaming implementation enumerates matching candidates and has an $O(nB)$ worst-case bound under the stated uniqueness assumptions. It does not inherit the published $O(n\log^2 B)$ guarantee merely by sharing an incremental-tokenization objective. Nor does it implement eager emission of permanently finalized token boundaries.

Static certification is sound but conservative. Its construction can require $O(SB+PMB)$ work and temporary state-spelling storage, and successful certificates need not cover a large fraction of runtime queries. Certificate counts and hit rates are not reported. The observed 3.24% streaming advantage over certificate removal comes from separate evaluations with overlapping mean-confidence intervals; stronger paired evidence is needed to quantify that component precisely. The selected configuration's substantially higher construction cost also matters for short-lived tokenizers.

Dense transitions trade memory for execution speed, costing $1024S$ payload bytes with four-byte entries. The implementation uses unsigned array indices without a separate large-state capacity study. The successor-forest traversal is recursive, so exceptionally deep vocabularies can encounter interpreter recursion limits. Stored prefix history grows linearly with stream length; bounded-memory operation would require an additional retention or commitment mechanism.

The child-search and explicit-set comparators are local algorithmic controls, not reproductions of complete external tokenizer libraries. Acceptance-mode isolation retains most selected components but changes dispatch structure as well as predicate logic. Consequently, neither those controls nor the integrated baseline comparison provide a perfectly isolated machine-instruction attribution. Candidate preordering and tuple unpacking also lack individual ablations.

Primary-prefix correctness is checked against an independent batch procedure, whereas synthetic scaling measurements do not record equally exhaustive prefix validation. Forced-depth predicate tests deliberately construct histories and should be interpreted as computational stress tests. Timing confidence intervals characterize repeated execution on fixed streams, not uncertainty across datasets, processors, or trained vocabularies. Memory measurements cover traced Python allocations rather than resident memory.

## References
- Incremental BPE Tokenization. DOI: 10.48550/arXiv.2605.30813
- Formalizing BPE Tokenization. DOI: 10.48550/arXiv.2309.08715
- A Formal Perspective on Byte-Pair Encoding. DOI: 10.48550/arXiv.2306.16837
- Engineering faster double-array Aho-Corasick automata. DOI: 10.48550/arXiv.2207.13870
- Neural Machine Translation of Rare Words with Subword Units. DOI: 10.48550/arXiv.1508.07909
- SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. DOI: 10.48550/arXiv.1808.06226
- Getting the most out of your tokenizer for pre-training and domain adaptation. DOI: 10.48550/arXiv.2402.01035
- Efficient string matching. DOI: 10.1145/360825.360855

## Appendix A: Complete Experimental Trajectory Manifest

| Experiment | mse |
|---|---|
| baseline_source | 10.480728937 |
| baseline_subset | 11.250042007 |
| baseline_full | 10.554896208 |
| initial_subset | 3.835336732 |
| initial_full | 2.782369645 |
| seed0_subset | 4.615668634 |
| seed0_full | 3.689735119 |
| evolved0_subset | 3.803686316 |
| evolved0_full | 2.815151877 |
| seed1_subset | 4.262018289 |
| seed1_full | 3.020360156 |
| evolved1_subset | 4.001354682 |
| evolved1_full | 2.937443545 |
| ablation0 | 2.872577372 |
| ablation1 | 3.170360085 |
| ablation2 | 3.569943551 |
| ablation3 | 3.046814325 |
| ablation4 | 3.141957576 |
| rebuttal0 | 2.803007091 |
| final_test | 2.806132721 |
