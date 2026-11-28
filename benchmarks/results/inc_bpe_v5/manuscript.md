# Exact Streaming Byte-Pair Tokenization through Rank-Ordered Interval Compilation and Static Context Certificates

## Abstract
We propose an exact streaming byte-pair encoding algorithm that compiles rank-constrained predecessor acceptance into two intervals per merge rule and supplements these predicates with sound, positive-only context certificates. Our theoretical analysis establishes constant-time acceptance queries, strict linear-time construction of the interval representation, and certificate soundness, including self-overlapping rules and beginning-of-stream boundaries. The complete implementation combines these mechanisms with dense automaton transitions and an atomic-token bypass while preserving batch tokenization at every input prefix. Our experiments evaluate exactness, randomized interleaved paired latency, component subtraction, and controlled forest-depth scaling. The integrated method achieves median speedups of 4.05× on validation and 4.01× on held-out streams against a pointer-based baseline. Without certificates, replacing pointer traversal with interval predicates reduces mean validation latency from 3.4861 to 2.9653 ms per 10 KB; adding certificates further reduces it to 2.8095 ms. Static analysis certifies 78.76% of automaton-state/merge-candidate pairs, and certificate hits cover 66.56% and 64.43% of non-atomic bytes on validation and test streams, respectively. At fixed vocabulary size, forced-depth predicate speedups reach 7.85×. These results demonstrate an algorithmic improvement in predecessor acceptance alongside complementary implementation gains, with explicit accounting for preprocessing, memory, and experimental scope.

## Introduction
Byte-pair encoding (BPE) is usually described as a sequence of ranked adjacent-token merges. Streaming introduces a different requirement: after each arriving byte, the tokenizer must recover the exact result of applying those merges to the entire current prefix. A newly arriving byte can alter a suffix containing several earlier token boundaries, making simple append-only token emission insufficient.

We propose rank-ordered interval compilation with static context certificates for exact incremental BPE. The central observation concerns the successor forest, in which each merge token points to its right operand. A predecessor acceptance query combines ancestry with a strict rank condition on the first descendant branch. If sibling subtrees are visited in increasing merge rank, the admissible branches occupy one contiguous DFS suffix. Together with the predecessor itself, this gives a representation containing at most two intervals per rule.

Our theoretical analysis establishes three guarantees. First, the compiled predicate is exactly equivalent to the rank-constrained successor-forest condition and takes constant time independently of forest depth. Second, all interval cutoffs can be constructed in strict $O(M+R)$ time for vocabulary size $M$ and $R$ rules, using an update-before-read sweep that correctly handles identical left and right operands. Third, suffix-compatible context domains yield sound positive acceptance certificates: whenever all compatible predecessors satisfy the compiled predicate, the runtime predecessor lookup can be omitted. Unknown contexts retain the exact predicate check.

The complete tokenizer evaluates suffix candidates in decreasing byte length using an Aho–Corasick recognizer, dense byte transitions, and an atomic-token bypass. It stores the last token of every processed prefix and reconstructs a current tokenization by following token lengths backward. It does not memoize runtime acceptance queries. Our constant-time theorem concerns each predecessor predicate; the implemented candidate scan retains a dependence on the number of recognized suffix candidates.

Our experiments evaluate both isolated mechanisms and integrated performance. The integrated method reduces median latency from 11.3880 to 2.8085 ms per 10 KB on validation and from 11.3379 to 2.8260 on held-out streams. In a separate randomized cache-free comparison, interval predicates without certificates improve mean validation latency by approximately 14.9% over pointer predicates with the same dense transitions and atomic bypass. Certificates provide another 5.3% reduction. Static certificates cover 89 of 113 state/candidate pairs; their runtime hit rates among non-atomic bytes are 66.56% on validation and 64.43% on test. Controlled sweeps distinguish vocabulary growth from forest depth and show that the direct predicate advantage increases strongly with depth, while whole-stream gains remain moderated by other costs.

## Related Work
Incremental BPE. Jiang and Gong's Incremental BPE Tokenization [1] establishes exact maintenance of BPE results for every input prefix, with a stated worst-case per-byte bound of $O(\log^2 L)$ for maximum token length $L$. Its suffix-recognition and successor-forest formulation supplies the setting for our predecessor predicate. We address the rank-constrained acceptance subproblem by changing the forest numbering: an ancestry interval alone does not encode the rank of the first child on a descendant path, whereas rank-ordered numbering makes the admissible child subtrees contiguous. Our implementation scans recognized candidates, so we do not transfer the logarithmic candidate-search guarantee of [1] to the complete implemented tokenizer.

Component-weighted search. Component-Weighted Centroid Search for Exact Incremental BPE [2] targets candidate-search traversal through component weighting. That objective is formally distinct from compiling the binary acceptance relation $A(t,k)$. A search structure determines which candidate to inspect next; our intervals determine whether a specified candidate accepts a specified predecessor. The techniques are therefore conceptually composable. We evaluate local weighted and unweighted search baselines, but do not claim an implemented composition or equate local baseline timings with published end-to-end results.

Formal BPE semantics. Formalizing BPE Tokenization [3] and A Formal Perspective on Byte-Pair Encoding [4] concern the mathematical treatment of merge systems and tokenization behavior. Our contribution assumes a fixed canonical ranked vocabulary and preserves its greedy merge semantics. In particular, the acceptance rule uses strict rank inequality, and the stream boundary has no predecessor token. These details are essential for equivalence; replacing the acceptance relation with unconstrained ancestry would define a different algorithm. Our results add a compact representation and construction theorem for this relation rather than a new BPE training objective or a different segmentation semantics.

String recognition and automaton engineering. Aho and Corasick's Efficient String Matching [9] provides the failure-link automaton that recognizes vocabulary suffixes. Engineering Faster Double-Array Aho–Corasick Automata [5] studies an alternative engineering direction for automaton layout. We use a dense table with 256 transitions per state, rather than a double-array representation. The distinction is quantitative: dense transitions take $1024S$ bytes with 32-bit state identifiers and $S$ states, while our acceptance representation requires four integer endpoints per merge. Recognition layout and acceptance compilation address separate costs. Dense transitions alone cannot remove the rank-constrained predecessor query.

Subword modeling and tokenizer systems. Neural Machine Translation of Rare Words with Subword Units [6] motivates BPE as a way to represent rare words through subword units. SentencePiece [7] provides language-independent tokenizer and detokenizer infrastructure. Getting the Most Out of Your Tokenizer for Pre-training and Domain Adaptation [8] addresses tokenizer utility in model preparation and adaptation. Our method holds vocabulary and merge rules fixed: it changes the execution of exact prefix tokenization, without changing learned subword units, compression objectives, or model training. Consequently, we measure exactness and computational cost rather than downstream language-model quality.

## Methodology
Problem and notation. Let $\Sigma=\{0,\ldots,255\}$ and let $\mathcal V$ contain all byte atoms followed by $R$ merge tokens, so $M=256+R$. Rule $r$ creates token $t=256+r$ from operands $(p(t),s(t))$, with spelling $v_t=v_{p(t)}v_{s(t)}$ and length $\ell(t)=\ell(p(t))+\ell(s(t))$. Operands precede the token they create. We assume a canonical vocabulary: applying ranked greedy BPE to $v_t$ yields the singleton token $t$. Greedy BPE selects the adjacent pair with minimum rule rank, breaking equal-rank occurrence ties from left to right.

For stream $x_{1:n}$, let $H[n]$ denote the last token in its exact BPE tokenization. The maintained history starts with $H[0]=\bot$. A merge candidate $t$ recognized as a suffix uses predecessor $k=H[n-\ell(s(t))]$, provided $n>\ell(s(t))$. In the successor forest, each merge token has parent $s(t)$. The acceptance relation is

$$A(t,k)\iff k=p(t)\ \lor\ \exists c\in\operatorname{Child}(p(t)):\ k\in\operatorname{Desc}(c)\ \land\ \operatorname{rank}(c)>\operatorname{rank}(t).$$

This is the prefix-last-token criterion for canonical ranked BPE. At each position, the last token is the longest recognized suffix satisfying this criterion, with byte atoms always admissible.

Rank-ordered compilation. We append children while visiting rules in increasing rank, so child lists are already ordered. An iterative preorder DFS assigns entry and exit positions $L_u,U_u$, with subtree interval $[L_u,U_u)$. The interval compiler stores

$$I_t=[L_{p(t)},L_{p(t)}+1)\ \cup\ [q_t,U_{p(t)}).$$

Theorem 1 (exact two-interval predicate). Define $q_t$ as the end of the last child subtree of $p(t)$ whose rank is at most $\operatorname{rank}(t)$, or $L_{p(t)}+1$ if no such child exists. Then, for every vocabulary token $k$,

$$A(t,k)\iff L_k\in I_t.$$

Proof. DFS places $p(t)$ first, followed by its child subtrees, each contiguous and ordered by increasing child rank. The child subtrees with rank at most that of $t$ therefore form precisely the prefix $[L_{p(t)}+1,q_t)$. Those with larger rank form the suffix $[q_t,U_{p(t)})$. The first interval admits $p(t)$ itself; the second admits exactly the descendants entering through a permitted child. Tokens outside the subtree satisfy neither condition. $\square$

Theorem 2 (linear cutoff construction). Initialize $\operatorname{cut}[u]=L_u+1$. For each rule token $t$ in increasing rank, first assign $\operatorname{cut}[s(t)]=U_t$, then read $q_t=\operatorname{cut}[p(t)]$. This computes all exact intervals in $O(M+R)$ time and $O(M+R)$ auxiliary words, including rules with $p(t)=s(t)$.

Proof. After processing rank $r$, $\operatorname{cut}[u]$ is the exit position of the last child of $u$ with rank at most $r$, or its initialized value if no such child exists. This invariant follows because each processed token is the next child in its parent's rank order. Reading after the update includes the current child when both operands coincide, excluding the equal-rank subtree as required by the strict inequality. DFS visits each vertex and edge once; the sweep performs constant work per rule. No child sorting or binary search is needed. $\square$

```text
Algorithm 1: Compile predecessor intervals
Input: canonical vocabulary V and ranked rules
Build successor-forest child lists in increasing rule order
Compute preorder entries L and subtree exits U by iterative DFS
for each token u:
    cut[u] <- L[u] + 1
for each merge token t in increasing rank:
    a <- p(t); b <- s(t)
    cut[b] <- U[t]
    I[t] <- (L[a], L[a] + 1, cut[a], U[a])
    offset[t] <- length[b]
return I, offset, L
```

Positive context certificates. Let $W_s$ be the trie spelling of automaton state $s$. For candidate $t$ at that state, define $C=W_s[:-\ell(s(t))]$. For nonempty $C$, the compatibility domain is

$$K(C)=\{u\in\mathcal V:\ v_u\text{ ends with }C\ \lor\ C\text{ ends with }v_u\}.$$

We sort the DFS positions of compatible tokens and coalesce consecutive positions into disjoint half-open regions. A candidate receives ACCEPT only if the domain is nonempty and every coalesced region is wholly contained in one of the two acceptance intervals. Otherwise it receives CHECK. Requiring each region to fit inside an individual interval is conservative when the intervals touch, but cannot create an incorrect certificate.

The selected configuration uses interval acceptance, dense transitions, the atomic bypass, and positive certificates; negative certificates are disabled. Certificate compilation permits at most $B=2{,}000{,}000$ vocabulary-token compatibility tests by default. Each newly considered nonempty context requires a complete scan of all $M$ vocabulary tokens. If the remaining budget is insufficient, that context remains CHECK. Empty and unconstructed domains also remain CHECK. Context interning exists only during construction and is discarded before streaming.

Theorem 3 (sound positive certificate). Suppose candidate $t$ is recognized at state $s$, $n>\ell(s(t))$, and its nonempty compatibility domain is certified as above. Then $A(t,H[n-\ell(s(t))])$ holds.

Proof. The spelling $C$ is a known suffix of the input ending immediately before the right operand of $t$. The actual predecessor token also ends at that position. Two strings ending at the same position are suffix-comparable, so the actual predecessor belongs to $K(C)$. Complete region coverage places its DFS position in $I_t$. Theorem 1 implies acceptance. The guard $n>\ell(s(t))$ ensures that a predecessor exists; a certificate cannot supply one at the beginning of the stream. $\square$

Streaming execution. We build an Aho–Corasick vocabulary recognizer and precompute a dense transition table. State $s$ and byte $b$ use index $256s+b$. Each state stores candidate tuples $(t,o,l,h,q,e,z)$ in decreasing token length, where $o=\ell(s(t))$, $(l,h,q,e)$ are interval endpoints, and $z$ is ACCEPT or CHECK. A state whose only candidate is an atom receives a direct bypass. The interval predicate is inlined in the streaming loop; direct predicate queries use the same intervals.

```text
Algorithm 2: Append one byte b
state <- transition[256 * state + b]
n <- n + 1
if state has exactly one candidate and it is an atom:
    append that atom to H; return it
for (t, offset, l, h, q, e, verdict) in candidates[state]:
    if t is an atom:
        append t to H; return t
    if n <= offset:
        continue
    if verdict == ACCEPT:
        append t to H; return t
    x <- L[H[n - offset]]
    if l <= x < h or q <= x < e:
        append t to H; return t
raise an error if no byte-covering candidate exists
```

Proposition 4 (preservation of prefix tokenization). Under the canonical prefix-last-token criterion, the compiled algorithm produces the exact last token for every prefix. Starting from position $n$, repeatedly outputting $H[n]$ and replacing $n$ by $n-\ell(H[n])$, then reversing the result, reconstructs the exact current tokenization.

Proof. Induct on the input length. Previous history entries are exact by the induction hypothesis. CHECK evaluates the original acceptance relation by Theorem 1; ACCEPT has the same outcome by Theorem 3. The atomic bypass returns the sole recognized candidate. Thus the descending-length scan selects exactly the candidate specified by the prefix-last-token criterion. Its boundary precedes a prefix whose last token is already stored, so repeated length subtraction reconstructs the segmentation. The empty stream is the base case. $\square$

Complexity. Let $S$ be the automaton-state count, $P=\sum_s c_s$ the total number of state/candidate entries, $L_{\max}$ the maximum token length, and $c_{\max}=\max_s c_s$. Interval compilation alone takes $O(M+R)$ time and four endpoint words per merge. Each predicate takes worst-case $O(1)$ time. Dense transitions require $O(256S)$ construction work and $256S$ unsigned 32-bit cells, or $1024S$ bytes when that integer type occupies four bytes. Candidate ordering takes $O(\sum_s c_s\log c_s)$ time; candidate tuples occupy $O(P)$ words.

Certificate compilation is additional to the linear interval compiler. If $Q$ contexts receive complete compatibility scans, then $QM\le B$. With byte-string suffix comparisons, scanning costs $O(QML_{\max})$, sorting compatible positions costs at most $O(QM\log M)$, and checking region coverage costs $O(PM)$ in the worst case. Temporary state spellings occupy $O(SL_{\max})$ bytes; temporary domains occupy at most $O(QM)$ words. Context slicing and construction of state spellings add up to $O((S+P)L_{\max})$ byte work in the straightforward implementation. These bounds explicitly preclude describing the entire preprocessing procedure as linear.

After compilation, appending byte $i$ costs $O(1+c_{s_i})$, giving $O(n+\sum_{i=1}^n c_{s_i})\subseteq O(nc_{\max})$ streaming time. Distinct canonical token spellings allow at most one recognized suffix of each length, so $c_{\max}\le\min(M,L_{\max})$. Persistent storage, excluding input vocabulary spellings, is $O(M+256S+P+n)$ words, including the prefix history. Reconstructing $z$ output tokens takes $O(z)$ time and output space.

## Experiments
Experimental setting. We evaluate a fixed 328-token canonical vocabulary containing 256 byte atoms and 72 learned merges. The vocabulary is constructed from English prose, Chinese text, and Python-like source text. Maximum token length is six bytes and maximum successor depth is four. Validation and held-out evaluation each contain three 768-byte streams, totaling 2,304 bytes, generated from the same source families with different randomization. Thus held-out evaluation tests unseen generated streams rather than domain transfer. Exactness is checked against an independent global greedy implementation using linked-list updates and a rank-priority heap at every byte prefix.

Timing protocol. Primary streaming timings use seven randomized interleaved paired passes, excluding model construction. Latencies are normalized to 10,000 bytes. For paired differences $d_i$, we report $\bar d\pm t_{0.975,6}s_d/\sqrt7$, with $t_{0.975,6}=2.4469$. Mean-latency intervals use the analogous formula on individual-method samples. A confidence interval on a difference is paired; intervals beside individual means are marginal. Median ratios are point estimates and are not confidence intervals. Scaling studies use five randomized interleaved passes and paired Student-t intervals with four degrees of freedom.

Primary integrated comparison. Both validation and held-out evaluation match all 2,304 reference prefixes exactly.

| Split | Proposed median | Pointer baseline median | Proposed mean ± 95% CI | Paired baseline-minus-proposed mean ± 95% CI | Median speedup |
|---|---:|---:|---:|---:|---:|
| Validation | 2.8085 | 11.3880 | 2.9279 ± 0.2296 | 8.4780 ± 0.3362 | 4.05× |
| Held-out test | 2.8260 | 11.3379 | 2.9348 ± 0.2658 | 8.3659 ± 0.2566 | 4.01× |

All latency entries are ms per 10 KB. The paired intervals exclude zero on both splits. The test median is approximately 0.6% above the validation median, indicating similar performance on these two generated stream sets.

Comparison against all implemented baselines. A separate seven-pass randomized cache-free study compares the complete method, an interval-only variant, a pointer-predicate variant, weighted and unweighted centroid-search implementations, explicit acceptance sets, and the basic pointer tokenizer. The interval-only and pointer-predicate variants retain dense transitions and the atomic bypass while disabling certificates. Entries below show mean latency with its marginal 95% half-width, followed by the paired excess latency relative to the complete method.

| Method | Validation mean ± CI | Validation paired excess ± CI | Test mean ± CI | Test paired excess ± CI |
|---|---:|---:|---:|---:|
| Proposed complete method | 2.8095 ± 0.0720 | 0 | 2.7908 ± 0.0688 | 0 |
| Intervals without certificates | 2.9653 ± 0.0462 | 0.1558 ± 0.0908 | 2.9500 ± 0.0653 | 0.1592 ± 0.1244 |
| Pointer predicate without certificates | 3.4861 ± 0.0300 | 0.6765 ± 0.0650 | 3.5184 ± 0.0560 | 0.7276 ± 0.0779 |
| Weighted centroid search | 4.8683 ± 0.1110 | 2.0588 ± 0.1502 | 4.9665 ± 0.2039 | 2.1757 ± 0.2096 |
| Unweighted centroid search | 4.9144 ± 0.1504 | 2.1049 ± 0.1741 | 4.9867 ± 0.2001 | 2.1959 ± 0.2039 |
| Explicit acceptance sets | 4.5297 ± 0.2315 | 1.7202 ± 0.2137 | 4.6554 ± 0.2372 | 1.8646 ± 0.2243 |
| Basic pointer tokenizer | 11.6209 ± 0.3877 | 8.8114 ± 0.3792 | 11.6238 ± 0.4611 | 8.8330 ± 0.4720 |

All paired excess-latency intervals exclude zero. These measurements compare the evaluated implementations under a common protocol; they are not reproductions of the published systems' external benchmark claims.

Orthogonal mechanism isolation. We distinguish predecessor representation from context certification. The available cache-free factorial cells are shown explicitly; an unmeasured cell is not inferred from another timing study.

| Predecessor predicate | Certificates disabled: validation / test mean | Positive certificates enabled: validation / test mean |
|---|---:|---:|
| Pointer traversal | 3.4861 / 3.5184 | Not measured in this paired study |
| Two intervals | 2.9653 / 2.9500 | 2.8095 / 2.7908 |

With certificates disabled, intervals reduce mean latency by 0.5208 ms on validation and 0.5684 ms on test, corresponding to 14.9% and 16.2%. These differences isolate predecessor evaluation from certificate benefits, although a paired interval for this particular contrast is not reported. With intervals fixed, certificates save 0.1558 ± 0.0908 ms and 0.1592 ± 0.1244 ms, or approximately 5.3% and 5.4%. The incomplete fourth cell prevents estimation of a full factorial interaction. No interaction coefficient is claimed.

Five-component subtraction. We evaluate removal of interval predicates, dense transitions, the atomic bypass, context certificates, and negative certificates. The reference for the first four removals permits both positive and negative certificates; the fifth removal yields the positive-only design. Consequently, these are component-subtraction measurements around a common certificate-enabled reference, rather than five removals from the final positive-only configuration.

| Removed component | Validation mean ± marginal 95% CI | Change from certificate-enabled reference |
|---|---:|---:|
| None: reference with positive and negative certificates | 2.9015 ± 0.2259 | — |
| Constant-time interval predicate; use pointer traversal | 3.2183 ± 0.2655 | +0.3168 |
| Dense transition table; use sparse failure-link stepping | 3.5242 ± 0.0965 | +0.6227 |
| Atomic-token bypass | 3.1310 ± 0.2660 | +0.2295 |
| Context certificates | 3.1205 ± 0.2424 | +0.2190 |
| Negative certificates; retain positive certificates only | 2.9127 ± 0.2536 | +0.0112 |

The subtraction measurements identify dense transitions as a substantial contributor to integrated latency. They also show that the complete speedup cannot be attributed solely to interval compilation. Removing negative certificates changes the mean by only 0.0112 ms in these measurements; no statistically established benefit supports retaining that component. The final method therefore uses positive certificates only. Because subtraction runs were separate, the displayed changes are differences of means, without paired confidence intervals or claims of significance for each removal.

Multi-scale forest sweep. We vary merge count and maximum successor depth jointly. The following validation results report median streaming latency, the paired mean pointer-minus-interval difference with its 95% half-width, and a direct predicate microbenchmark that forces traversal depth. Certificates are disabled for the isolated interval-versus-pointer comparison.

| Merges / depth | Intervals | Pointer | Weighted search | Unweighted search | Explicit sets | Paired pointer-minus-interval ± CI | Forced-depth predicate speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 72 / 4 | 2.5507 | 3.0249 | 4.7387 | 4.8570 | 3.8204 | 0.4663 ± 0.0494 | 1.65× |
| 160 / 12 | 2.1980 | 2.6666 | 3.9650 | 4.1306 | 3.3656 | 0.4577 ± 0.0310 | 3.24× |
| 320 / 24 | 2.1695 | 2.6318 | 3.8222 | 3.9558 | 3.3468 | 0.4824 ± 0.0843 | 5.67× |
| 512 / 36 | 2.1350 | 2.5988 | 3.7661 | 3.8810 | 3.3044 | 0.4636 ± 0.0361 | 7.44× |

Every evaluated prefix matches the reference at every scale. Median whole-stream interval-versus-pointer speedups are 1.186×–1.217×. The much larger forced-depth ratios demonstrate that a depth-sensitive predicate can be eliminated without implying equally large end-to-end gains. Decreasing streaming latency across these synthetic scales should not be interpreted as a favorable complexity exponent: the vocabularies and tokenization workloads also change.

Controlled vocabulary-size and depth sweep. To separate those factors, we fix $R=320$ and $M=576$ while varying maximum depth.

| Depth | Intervals | Pointer | Weighted search | Unweighted search | Explicit sets | Paired pointer-minus-interval ± CI | Direct predicate speedup | Set elements / stored intervals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2.8001 | 3.4125 | 4.8199 | 4.9773 | 3.8565 | 0.6394 ± 0.0723 | 1.65× | 944 / 640 |
| 12 | 2.2926 | 2.8345 | 4.0114 | 4.1348 | 3.3984 | 0.5506 ± 0.0586 | 3.30× | 1309 / 640 |
| 24 | 2.1482 | 2.7392 | 3.8276 | 3.9332 | 3.3796 | 0.4856 ± 0.1670 | 5.49× | 1471 / 640 |
| 36 | 2.0920 | 2.5051 | 3.7822 | 3.8378 | 3.3892 | 0.3959 ± 0.0799 | 7.85× | 1536 / 640 |

All prefixes remain exact. The interval representation stays at $2R=640$ interval records, or $4R=1280$ endpoint scalars, while explicit membership counts increase with depth. Counts of set elements and interval records are structural quantities, not byte-accurate memory ratios. The controlled direct-predicate results support the depth-independent acceptance theorem; whole-stream performance also reflects changing candidate and token distributions.

Certificate coverage and memory. The coverage study independently enumerates suffix-compatible predecessors and replays the positive acceptance rule. It reports a domain-based coverage diagnostic rather than an instruction-level profile of the compiled loop. In particular, its predicate-attempt counter can include candidates rejected by the stream-boundary guard.

| Quantity | Validation | Held-out test |
|---|---:|---:|
| Automaton states | 370 | 370 |
| Atomic-only states | 296 (80.0%) | 296 (80.0%) |
| Certified state/merge pairs | 89 / 113 (78.76%) | 89 / 113 (78.76%) |
| Runtime atomic bypasses | 1404 / 2304 (60.94%) | 1396 / 2304 (60.59%) |
| Runtime positive certificate hits | 599 | 585 |
| Certificate hits / non-atomic bytes | 599 / 900 (66.56%) | 585 / 908 (64.43%) |
| Remaining predicate attempts in coverage replay | 334 | 356 |
| Traced current allocation | 507276 bytes | 507276 bytes |
| Traced peak allocation | 556245 bytes | 556245 bytes |
| Dense transition payload | 378880 bytes | 378880 bytes |
| Acceptance endpoint scalars | 288 | 288 |
| Construction latency | 5.773 ms | 5.771 ms |
| Pointer-baseline construction latency | 0.357 ms | 0.359 ms |
| Aggregate token reconstruction latency | 64.129 μs | 63.568 μs |
| Estimated amortization length | 6.312 KB | 6.358 KB |

The dense payload follows directly from $370\times256\times4$ bytes. Traced allocation includes Python objects and one tokenizer after processing a 768-byte stream; it is not process resident memory or a vocabulary-independent bound. The 288 endpoints represent four scalars for each of 72 merge rules. Although independent domain analysis demonstrates substantial certificate coverage, the reported measurements do not separately enumerate all ACCEPT verdicts produced by the budgeted conservative region compiler.

Hardware and reproducibility. Measurements report Python 3.12.3, Linux 6.17.0-1014-nvidia-aarch64 with glibc 2.39, and an ARM Cortex-X925 at 3.90 GHz pinned to one core. Construction results are medians of five measurements. Streaming uses seven paired passes, and structural sweeps use five interleaved passes. Memory is measured with Python allocation tracing. No GPU computation is involved.

## Limitations
The empirical evidence covers a small canonical vocabulary, short streams, and synthetic successor-forest families. Validation and held-out streams share English, Chinese, and source-code generation families. Exactness on every evaluated prefix establishes correctness for those inputs, while the formal results establish equivalence of the compiled acceptance relation under the stated canonical assumptions. These experiments do not establish performance on production vocabularies, unrestricted merge systems, or shifted domains.

The complete implementation performs a descending-length candidate scan. Constant-time acceptance therefore does not imply constant-time byte processing, and we do not claim the logarithmic candidate-search bounds of centroid-based incremental algorithms. Dense transitions consume 1024 bytes per automaton state on the measured platform; their favorable latency may become a memory bottleneck for substantially larger vocabularies. Prefix history also grows linearly with stream length.

Only interval construction has a strict linear preprocessing guarantee. Certificate generation performs budgeted vocabulary scans, string comparisons, position sorting, and region checks. Its budget counts compatibility tests rather than total byte operations. Conservative region coverage can miss valid positive certificates, and exhausted budgets preserve correctness by retaining runtime checks. Independent certificate coverage measurements should not be interpreted as an exact profile of budgeted compiler verdicts.

The factorial study lacks a pointer-with-positive-certificates cell under the same paired protocol, preventing a complete interaction analysis. Component-subtraction runs provide marginal timing intervals but do not supply paired uncertainty for every removal. The direct predicate advantage is isolated without certificates, but the larger integrated speedup also depends on dense transitions and the atomic bypass. Negative certificates have no established retained benefit and are excluded from the final design.

Seven timing passes on one processor characterize local variability rather than variation across machines, compilers, or operating systems. Student-t intervals describe these repeated measurements and are not adjusted for multiple comparisons. Construction overhead makes very short independent streams less favorable unless preprocessing is reused. No downstream model-quality, energy-consumption follow from these results.

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

## Appendix A: Complete Experimental Trajectory Manifest

| Experiment | mse |
|---|---|
| baseline_source | 10.602365769 |
| baseline_subset | 11.308477648 |
| baseline_full | 10.470968037 |
| initial_subset | 3.740710478 |
| initial_full | 2.817947057 |
| seed0_subset | 5.387730198 |
| seed0_full | 3.932682375 |
| evolved0_subset | 3.862374191 |
| evolved0_full | 2.935578070 |
| ablation0 | 3.227525869 |
| ablation1 | 3.556627538 |
| ablation2 | 3.020789639 |
| ablation3 | 2.963423766 |
| ablation4 | 2.762382994 |
| pruned | 2.808493365 |
| rebuttal0 | 2.811597773 |
| final_test | 2.826045602 |
