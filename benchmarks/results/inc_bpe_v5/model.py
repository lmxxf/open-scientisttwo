"""Exact incremental ranked BPE with static, three-way context certificates."""
from array import array
from collections import deque
from data import SuffixAutomaton


class IncrementalBPE:
    def __init__(self, vocab, rules, config=None):
        cfg = config or {}
        self.vocab, self.rules = vocab, rules
        self.mode = cfg.get('acceptance_mode', 'intervals')
        if self.mode not in ('intervals', 'sets', 'pointer'):
            raise ValueError('unknown acceptance_mode')
        self.flat = cfg.get('flat_transitions', True)
        self.atomic_fastpath = cfg.get('atomic_fastpath', True)
        self.certified_regions = cfg.get('certified_regions', True)
        self.rejection_certificates = cfg.get('rejection_certificates', True)
        self.lengths = tuple(map(len, vocab))
        m = len(vocab)
        self.successor = [-1] * m
        self.pre = [-1] * m
        children = [[] for _ in vocab]
        for t, (a, b) in enumerate(rules, 256):
            self.pre[t], self.successor[t] = a, b
            children[b].append(t)
        # Iterative DFS avoids recursion limits. Children are already rank ordered.
        left, right = [0] * m, [0] * m
        tick = 0
        for root in range(256):
            stack = [(root, False)]
            while stack:
                u, exiting = stack.pop()
                if exiting:
                    right[u] = tick
                else:
                    left[u] = tick
                    tick += 1
                    stack.append((u, True))
                    stack.extend((c, False) for c in reversed(children[u]))
        self.left, self.right = left, right
        cutoff = [x + 1 for x in left]
        self.intervals = [()] * m
        self.offsets = [0] * m
        # Strict O(M+R): update BEFORE lookup, including when a == b.
        for t, (a, b) in enumerate(rules, 256):
            cutoff[b] = right[t]
            self.intervals[t] = (left[a], left[a] + 1, cutoff[a], right[a])
            self.offsets[t] = self.lengths[b]
        if self.mode == 'sets':
            self.accept = [None] * m
            for t in range(256, m):
                lo, hi, q, end = self.intervals[t]
                self.accept[t] = frozenset(u for u in range(m)
                                          if lo <= left[u] < hi or q <= left[u] < end)

        self.ac = SuffixAutomaton(vocab)
        ac = self.ac
        count = len(ac.edges)
        words = [b''] * count
        queue = deque([0])
        bfs = []
        while queue:
            s = queue.popleft()
            bfs.append(s)
            for c, ns in ac.edges[s].items():
                words[ns] = words[s] + bytes((c,))
                queue.append(ns)
        # 256 * states uint32 cells, i.e. exactly 1024 bytes/state.
        self.trans = None
        if self.flat:
            trans = array('I', [0]) * (256 * count)
            for s in bfs:
                base = s * 256
                if s:
                    fbase = ac.fail[s] * 256
                    trans[base:base + 256] = trans[fbase:fbase + 256]
                for c, ns in ac.edges[s].items():
                    trans[base + c] = ns
            self.trans = trans

        # K(C) overapproximates the predecessor token at the boundary before
        # s(t): its spelling and the known context must be suffix-comparable.
        # Store K as disjoint DFS intervals. Compilation is explicitly budgeted;
        # unconstructed/empty domains always remain CHECK, never ACCEPT.
        budget = max(0, int(cfg.get('certificate_budget', 2000000)))
        self.certificate_stats = dict(compatibility_tests=0, representation_intervals=0,
                                      accept=0, reject=0, check=0,
                                      incomplete_contexts=0)
        domains = {}  # construction-only interning, discarded before streaming
        stats = self.certificate_stats
        self.candidates = []
        self.atomic = [-1] * count
        for s, out in enumerate(ac.output):
            ordered = sorted(out, key=lambda t: -self.lengths[t])
            if len(ordered) == 1 and ordered[0] < 256:
                self.atomic[s] = ordered[0]
            entries = []
            for t in ordered:
                if t < 256:
                    entries.append((t, 0, 0, 0, 0, 0, 0))
                    continue
                off = self.offsets[t]
                lo, hi, q, end = self.intervals[t]
                verdict = 0
                if self.certified_regions:
                    ctx = words[s][:-off]
                    if ctx and ctx not in domains:
                        if stats['compatibility_tests'] + m <= budget:
                            positions = []
                            for u, word in enumerate(vocab):
                                if word.endswith(ctx) or ctx.endswith(word):
                                    positions.append(left[u])
                            stats['compatibility_tests'] += m
                            positions.sort()
                            regions = []
                            for pos in positions:
                                if regions and regions[-1][1] == pos:
                                    regions[-1] = (regions[-1][0], pos + 1)
                                else:
                                    regions.append((pos, pos + 1))
                            domains[ctx] = regions
                            stats['representation_intervals'] += len(regions)
                        else:
                            domains[ctx] = None
                            stats['incomplete_contexts'] += 1
                    regions = domains.get(ctx)
                    if regions:
                        covered = all((lo <= a and b <= hi) or
                                      (q <= a and b <= end) for a, b in regions)
                        disjoint = all(not (a < hi and lo < b) and
                                       not (a < end and q < b) for a, b in regions)
                        if covered:
                            verdict = 1
                        elif disjoint and self.rejection_certificates:
                            verdict = -1
                stats[{1: 'accept', -1: 'reject', 0: 'check'}[verdict]] += 1
                entries.append((t, off, lo, hi, q, end, verdict))
            self.candidates.append(tuple(entries))
        self.state = 0
        self.length = 0
        self.last = [None]

    def valid(self, t, n):
        if t < 256:
            return True
        idx = n - self.offsets[t]
        if idx <= 0 or idx >= len(self.last):
            return False
        k = self.last[idx]
        if k is None:
            return False
        if self.mode == 'intervals':
            x = self.left[k]
            lo, hi, q, end = self.intervals[t]
            return lo <= x < hi or q <= x < end
        if self.mode == 'sets':
            return k in self.accept[t]
        a, child = self.pre[t], -1
        while k != a and self.successor[k] != -1:
            child, k = k, self.successor[k]
        return k == a and (child == -1 or t < child)

    def append(self, byte):
        s = (self.trans[self.state * 256 + byte] if self.flat
             else self.ac.step(self.state, byte))
        self.state = s
        self.length += 1
        if self.atomic_fastpath and self.atomic[s] >= 0:
            t = self.atomic[s]
            self.last.append(t)
            return t
        n = self.length
        for t, off, lo, hi, q, end, verdict in self.candidates[s]:
            if t < 256:
                self.last.append(t)
                return t
            # BOS has no predecessor token and cannot be certified positive.
            if n <= off or verdict == -1:
                continue
            if verdict == 1:
                self.last.append(t)
                return t
            if self.mode == 'intervals':
                x = self.left[self.last[n - off]]
                ok = lo <= x < hi or q <= x < end
            else:
                ok = self.valid(t, n)
            if ok:
                self.last.append(t)
                return t
        raise ValueError('vocabulary does not cover input byte')

    def tokens(self):
        result = []
        n = self.length
        while n:
            t = self.last[n]
            result.append(t)
            n -= self.lengths[t]
        return result[::-1]
