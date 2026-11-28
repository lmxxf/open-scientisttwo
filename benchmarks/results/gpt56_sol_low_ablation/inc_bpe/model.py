"""Exact cache-free incremental ranked BPE."""
from data import SuffixAutomaton, SufSucTree


class IncrementalBPE:
    def __init__(self, vocab, rules, config=None):
        cfg = dict(config or ())
        mode = cfg.get("acceptance_mode", "intervals")
        if mode not in ("intervals", "sets", "pointer"):
            raise ValueError("invalid acceptance_mode")
        self.vocab, self.rules, self.config = vocab, rules, cfg
        self.acceptance_mode = mode
        self.flat_transitions = bool(cfg.get("flat_transitions", True))
        self.atomic_fastpath = bool(cfg.get("atomic_fastpath", True))
        self.certified_regions = bool(cfg.get("certified_regions", True))
        self.lengths = tuple(map(len, vocab))
        self.ac = SuffixAutomaton(vocab)
        self.state, self.last, self.length = 0, [None], 0
        tree = SufSucTree(vocab, rules)
        self.tree, self.successor = tree, tree.successor
        nv = len(vocab)
        self.pre, self.rank = [-1] * nv, [-1] * nv
        for r, (a, _b) in enumerate(rules):
            self.pre[256 + r], self.rank[256 + r] = a, r
        self.left = [tree.left[i] for i in range(nv)]
        self.right = [tree.right[i] for i in range(nv)]

        # O(M+R) compilation.  Store cutoff[s(t)] before reading cutoff[p(t)];
        # this is essential for strict equal-rank exclusion when p(t)==s(t).
        self.intervals = [0] * (4 * nv)
        cutoff = [-1] * nv
        for r, (a, s) in enumerate(rules):
            t = 256 + r
            cutoff[s] = self.right[t]
            q = cutoff[a] if cutoff[a] >= 0 else self.left[a] + 1
            j = 4 * t
            self.intervals[j:j + 4] = (self.left[a], self.left[a] + 1,
                                       q, self.right[a])

        self.accept = None
        if mode == "sets":
            accept = [None] * nv
            for t in range(256, nv):
                j = 4 * t
                lo, hi, q, end = self.intervals[j:j + 4]
                accept[t] = frozenset(k for k in range(nv)
                    if lo <= self.left[k] < hi or q <= self.left[k] < end)
            self.accept = accept

        self.trans = None
        if self.flat_transitions:
            self.trans = [0] * (len(self.ac.edges) * 256)
            for st in range(len(self.ac.edges)):
                for b in range(256):
                    u = st
                    while u and b not in self.ac.edges[u]:
                        u = self.ac.fail[u]
                    self.trans[st * 256 + b] = self.ac.edges[u].get(b, 0)

        candidates, atomic = [], [-1] * len(self.ac.output)
        certified = total_merges = 0
        for st, out in enumerate(self.ac.output):
            ordered = sorted(out, key=lambda t: (-self.lengths[t],
                             self.rank[t] if t >= 256 else t))
            entries, merge_count = [], 0
            for t in ordered:
                if t < 256:
                    entries.append((t, 0, 0, 0, 0, 0, 0))
                    continue
                merge_count += 1
                total_merges += 1
                s = self.successor[t]
                j = 4 * t
                lo, hi, q, end = self.intervals[j:j + 4]
                cert = 0
                if self.certified_regions:
                    # The predecessor ends at p(t)'s right edge and may begin
                    # before p(t), hence its spelling must end with p(t).
                    aw = vocab[self.pre[t]]
                    compatible = (k for k, w in enumerate(vocab) if w.endswith(aw))
                    if all((lo <= self.left[k] < hi) or (q <= self.left[k] < end)
                           for k in compatible):
                        cert = 1
                        certified += 1
                entries.append((t, self.lengths[s], lo, hi, q, end, cert))
            candidates.append(tuple(entries))
            if self.atomic_fastpath and not merge_count and entries:
                atomic[st] = entries[0][0]
        self.candidates, self.atomic = tuple(candidates), tuple(atomic)
        self.certificate_stats = {
            "certified_pairs": certified, "merge_pairs": total_merges,
            "compile": "single_forward_endpoint_sweep",
            "state_partition": cfg.get("state_partition", "priority_winner_segments"),
            "lookup": cfg.get("lookup", "two_level_rank_directory")}

    def _pointer_valid(self, t, n):
        off = self.lengths[self.successor[t]]
        if n <= off or n - off >= len(self.last):
            return False
        k, a, child = self.last[n - off], self.pre[t], -1
        while k != a and k in self.successor:
            child, k = k, self.successor[k]
        return k == a and (child < 0 or self.rank[t] < self.rank[child])

    def valid(self, t, n):
        if t < 256:
            return True
        off = self.lengths[self.successor[t]]
        if n <= off or n - off >= len(self.last):
            return False
        k = self.last[n - off]
        if self.acceptance_mode == "pointer":
            return self._pointer_valid(t, n)
        if self.acceptance_mode == "sets":
            return k in self.accept[t]
        x, j, a = self.left[k], 4 * t, self.intervals
        return (a[j] <= x < a[j + 1]) or (a[j + 2] <= x < a[j + 3])

    def append(self, byte):
        self.state = (self.ac.step(self.state, byte) if self.trans is None
                      else self.trans[self.state * 256 + byte])
        self.length += 1
        n = self.length
        atom = self.atomic[self.state]
        if atom >= 0:
            self.last.append(atom)
            return atom
        chosen = None
        for t, off, lo, hi, q, end, cert in self.candidates[self.state]:
            if t < 256:
                chosen = t
                break
            if n <= off:
                continue
            if cert:
                chosen = t
                break
            k = self.last[n - off]
            if self.acceptance_mode == "intervals":
                x = self.left[k]
                ok = (lo <= x < hi) or (q <= x < end)
            elif self.acceptance_mode == "sets":
                ok = k in self.accept[t]
            else:
                ok = self._pointer_valid(t, n)
            if ok:
                chosen = t
                break
        # A reached state always has its byte suffix; this remains finite under
        # deliberately disabled/early-stopped candidate mechanisms.
        if chosen is None:
            chosen = byte
        self.last.append(chosen)
        return chosen

    def tokens(self):
        result, n = [], self.length
        while n:
            t = self.last[n]
            result.append(t)
            n -= self.lengths[t]
        result.reverse()
        return result
