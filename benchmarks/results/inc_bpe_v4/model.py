"""Exact streaming BPE; linear rank-sweep intervals and static certificates.

Interval compilation: O(M+R). Dense transitions: O(256*S), 1024*S bytes.
Optional certification: O(S*B + P*M*B), potentially superlinear, with
bounded P selected state/candidate pairs and maximum spelling length B.
No runtime acceptance memoization is used.
"""
from array import array
from collections import deque
from data import SuffixAutomaton, SufSucTree


class IncrementalBPE:
    def __init__(self, vocab, rules, config=None):
        config = config or {}
        self.vocab, self.rules = vocab, rules
        self.acceptance_mode = config.get('acceptance_mode', 'intervals')
        if self.acceptance_mode not in ('intervals', 'sets', 'pointer'):
            raise ValueError('unknown acceptance_mode')
        self.atomic_fastpath = config.get('atomic_fastpath', True)
        self.certified_regions = config.get('certified_regions', False)
        self.lengths = tuple(map(len, vocab))
        self.ac = SuffixAutomaton(vocab)
        self.state = 0
        self.last = [None]
        self.length = 0
        self.tree = SufSucTree(vocab, rules)
        self.successor = self.tree.successor
        self.pre = {256+i: a for i, (a, b) in enumerate(rules)}
        self.rank = {256+i: i for i in range(len(rules))}
        self.left = tuple(self.tree.left[t] for t in range(len(vocab)))
        self.intervals = [None] * len(vocab)
        self.offsets = [0] * len(vocab)
        cutoff = {}
        # Children are rank ordered and their DFS subtrees are contiguous.
        # Update first to exclude the equal-rank child even when a == b.
        for i, (a, b) in enumerate(rules):
            t = 256+i
            cutoff[b] = self.tree.right[t]
            lo = self.left[a]
            q = cutoff.get(a, lo+1)
            self.intervals[t] = (lo, lo+1, q, self.tree.right[a])
            self.offsets[t] = self.lengths[b]
        if self.acceptance_mode == 'sets':
            self.accept = [None] * len(vocab)
            for t in range(256, len(vocab)):
                lo, end, q, hi = self.intervals[t]
                self.accept[t] = frozenset(k for k, x in enumerate(self.left)
                                          if lo <= x < end or q <= x < hi)
        self.trans = None
        if config.get('flat_transitions', True):
            self.trans = array('I', [0]) * (256*len(self.ac.edges))
            for c, v in self.ac.edges[0].items():
                self.trans[c] = v
            queue = deque(self.ac.edges[0].values())
            while queue:
                u = queue.popleft()
                base, fail = u*256, self.ac.fail[u]*256
                self.trans[base:base+256] = self.trans[fail:fail+256]
                for c, v in self.ac.edges[u].items():
                    self.trans[base+c] = v
                    queue.append(v)
        outputs = [tuple(sorted(out, key=lambda t: -self.lengths[t]))
                   for out in self.ac.output]
        # State word W fixes context C = W[:-len(s(t))] before s(t).
        # Any predecessor K must end in C or be a suffix of C. This includes
        # long predecessors crossing the state's left boundary. Enumerating
        # all such vocabulary tokens is conservative, not a reachability claim.
        words = None
        if self.certified_regions:
            words = [b''] * len(outputs)
            stack = [0]
            while stack:
                u = stack.pop()
                for c, v in self.ac.edges[u].items():
                    words[v] = words[u] + bytes((c,))
                    stack.append(v)
        state_budget = max(0, int(config.get('certificate_max_states', 64)))
        pair_budget = max(0, int(config.get('certificate_max_pairs', 256)))
        selected_states = 0
        self.certificate_count = 0
        self.candidates = []
        for state, out in enumerate(outputs):
            eligible = (self.certified_regions and selected_states < state_budget
                        and pair_budget > 0 and any(t >= 256 for t in out))
            if eligible:
                selected_states += 1
            row = []
            for t in out:
                if t < 256:
                    row.append((t, 0, 0, 0, 0, 0, False))
                    continue
                lo, end, q, hi = self.intervals[t]
                off = self.offsets[t]
                certified = False
                if eligible and pair_budget:
                    pair_budget -= 1
                    context = words[state][:-off]
                    certified = True
                    for k, word in enumerate(vocab):
                        if word.endswith(context) or context.endswith(word):
                            x = self.left[k]
                            if not (lo <= x < end or q <= x < hi):
                                certified = False
                                break
                    self.certificate_count += int(certified)
                row.append((t, off, lo, end, q, hi, certified))
            self.candidates.append(tuple(row))
        self.atomic = tuple(out[0] if len(out) == 1 and out[0] < 256 else -1
                            for out in outputs)

    def valid(self, t, n):
        if t < 256:
            return True
        pos = n-self.offsets[t]
        if pos < 0 or pos >= len(self.last):
            return False
        k = self.last[pos]
        if k is None:
            return False
        mode = self.acceptance_mode
        if mode == 'intervals':
            x = self.left[k]
            lo, end, q, hi = self.intervals[t]
            return lo <= x < end or q <= x < hi
        if mode == 'sets':
            return k in self.accept[t]
        pre = self.pre[t]
        child = None
        while k != pre and k in self.successor:
            child = k
            k = self.successor[k]
        return k == pre and (child is None or self.rank[t] < self.rank[child])

    def append(self, byte):
        state = (self.trans[self.state*256+byte] if self.trans is not None
                 else self.ac.step(self.state, byte))
        self.state = state
        self.length += 1
        last = self.last
        if self.atomic_fastpath:
            token = self.atomic[state]
            if token >= 0:
                last.append(token)
                return token
        n = self.length
        if self.acceptance_mode == 'intervals':
            left = self.left
            for t, off, lo, end, q, hi, certified in self.candidates[state]:
                if t < 256:
                    last.append(t)
                    return t
                pos = n-off
                if pos > 0:
                    if certified:
                        last.append(t)
                        return t
                    x = left[last[pos]]
                    if lo <= x < end or q <= x < hi:
                        last.append(t)
                        return t
        else:
            for t, off, lo, end, q, hi, certified in self.candidates[state]:
                if (t < 256 or (certified and n-off > 0)
                        or self.valid(t, n)):
                    last.append(t)
                    return t
        raise ValueError('automaton has no atomic suffix')

    def tokens(self):
        result = []
        n = self.length
        while n:
            t = self.last[n]
            result.append(t)
            n -= self.lengths[t]
        return result[::-1]
