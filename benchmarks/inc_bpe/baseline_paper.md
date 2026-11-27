# Incremental BPE Tokenization: pointer traversal reference workload

This is a deterministic CPU/NumPy-compatible reimplementation of the *task family*
in Jiang & Gong, arXiv:2605.30813, and ScientistTwo Fig.2/Table9. It is not the
released authors' implementation or the original 12 production tokenizer benchmark.

Byte tokens 0..255 are atomic. An ordered dictionary maps adjacent token pairs to
IDs 256+rank. Reference BPE repeatedly merges the lowest-rank adjacent pair, with
leftmost tie breaking. Longest-match tokenization is NOT an equivalent algorithm.
Every prefix must reconstruct identical byte strings and identical token IDs.

A canonical token t has predecessor pre(t) and successor suc(t). Edges t->suc(t)
form a forest rooted at byte atoms. Aho-Corasick maintains suffix candidates for
each new byte; the suffix-induced successor subtree is SufSucTree(tau). For each
candidate t, obtain k=theta(prefix minus suc(t)); walk successor pointers until
pre(t) is reached. If the intervening child has a higher-priority merge than t,
t is invalid. Choose the longest valid candidate. Prefix-consistency lets the
whole tokenization be reconstructed by walking the stored last-token history.

DFS gives [L_t,R_t): k descends from t iff L_t<=L_k<R_t. data.SufSucTree stores
both pointer children and DFS intervals, but the baseline deliberately performs
pointer walks and Python branching. This exposes legitimate optimization room:
flat interval tests, cached transitions, atomic paths, compact candidate arrays.
All optimized versions must still pass exactness for EVERY byte prefix.

Vocabulary: 72 frequency-trained merges on fixed English, UTF-8 Chinese and code
strings. Validation/test/extra use disjoint deterministic stream RNG seeds, with
160/768 bytes per category in subset/full. Every prefix is checked by independent
ranked-merge reference OUTSIDE timing. Median of 7 streaming append passes,
excluding construction and tokens() reconstruction, gives latency_ms_per_10kb
=elapsed_seconds/bytes*1e7 (named mse for the default scalar minimization contract).
throughput_mb_s=bytes/elapsed_seconds/1e6. These are wall-clock measurements, not
numerically deterministic outputs; the manifest declares an explicit 50% rerun
relative tolerance, alongside exact untoleranced token equality. Pin CPU/threads
and repeat runs for publishable speed comparisons. Initialization is excluded
and must be measured separately if changed.

Source-provided targets, NOT local measurements: ScientistTwo Fig.2 reports a
10.9% relative improvement and a Stanford reviewer score of 6.5. Table9 describes
further 9.6% and 8.2% relative improvements. The generated VD-STrans manuscript
separately reports 1.123x geometric-mean throughput across 12 tokenizers and
AVX2 kernels. Those denominators and x86-specific SIMD do not transfer to this
small Python/ARM64 workload. Optimizing it does not reproduce those claims.
