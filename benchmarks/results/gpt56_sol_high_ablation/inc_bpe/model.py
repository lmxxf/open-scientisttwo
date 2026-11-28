"""Exact incremental ranked byte BPE with compiled predecessor predicates.

For a merge token ``t = (p, s)``, valid predecessor labels are ``p`` itself
and descendants of children of ``p`` whose merge rank is greater than the
rank of ``t``. The successor tree's children occur in increasing rank order,
so one forward rank sweep compiles that set into two DFS intervals. No
query-time cache is used.
"""

from collections import deque

from data import SuffixAutomaton, SufSucTree


class IncrementalBPE:
    """Maintain the exact priority-ranked BPE of every byte prefix."""

    def __init__(self, vocab, rules, config=None):
        cfg = {} if config is None else dict(config)
        mode = cfg.get("acceptance_mode", "intervals")
        if mode not in ("intervals", "sets", "pointer"):
            raise ValueError("acceptance_mode must be intervals, sets, or pointer")

        certificate_mode = cfg.get("certificate_mode", "accept_and_reject")
        aliases = {
            "accept": "accept_only",
            "acceptance": "accept_only",
            "both": "accept_and_reject",
            "off": "none",
        }
        certificate_mode = aliases.get(certificate_mode, certificate_mode)
        if certificate_mode not in (
            "none", "accept_only", "reject_only", "accept_and_reject"
        ):
            raise ValueError(
                "certificate_mode must be none, accept_only, reject_only, "
                "or accept_and_reject"
            )

        self.vocab = vocab
        self.rules = rules
        self.acceptance_mode = mode
        self.flat_transitions = bool(cfg.get("flat_transitions", True))
        self.atomic_fastpath = bool(cfg.get("atomic_fastpath", True))
        self.certified_regions = bool(cfg.get("certified_regions", True))
        self.certificate_mode = certificate_mode
        # Compatibility with the evaluator's independent rejection ablation.
        if "rejection_certificates" in cfg and not cfg["rejection_certificates"]:
            if certificate_mode == "accept_and_reject":
                self.certificate_mode = "accept_only"
            elif certificate_mode == "reject_only":
                self.certificate_mode = "none"

        self.lengths = tuple(map(len, vocab))
        size = len(vocab)
        self.ac = SuffixAutomaton(vocab)
        self.tree = SufSucTree(vocab, rules)
        self.state = 0
        self.last = [None]
        self.length = 0

        # Dense token-indexed metadata avoids dictionary work in the hot path.
        successor = [-1] * size
        predecessor = [-1] * size
        rank = [-1] * size
        for r, (p, s) in enumerate(rules):
            token = 256 + r
            predecessor[token] = p
            successor[token] = s
            rank[token] = r
        self.successor = successor
        self.pre = predecessor
        self.rank = rank
        self.left = tuple(self.tree.left[token] for token in range(size))
        self.right = tuple(self.tree.right[token] for token in range(size))

        # Strict O(M + R), single, forward, rank-ordered cutoff sweep. The
        # update MUST precede the read: if p(t) == s(t), using R[t] excludes
        # t's equal-rank subtree, preserving the strict rank inequality.
        cutoff = [self.left[token] + 1 for token in range(size)]
        intervals = [()] * size
        interval_data = [0] * (size * 4)
        for r, (p, s) in enumerate(rules):
            token = 256 + r
            cutoff[s] = self.right[token]
            q = cutoff[p]
            lo = self.left[p]
            region = (lo, lo + 1, q, self.right[p])
            intervals[token] = region
            base = token * 4
            interval_data[base] = region[0]
            interval_data[base + 1] = region[1]
            interval_data[base + 2] = region[2]
            interval_data[base + 3] = region[3]
        self.intervals = tuple(intervals)
        self.interval_data = interval_data

        # The explicit-set backend is an ablation of only the predicate
        # representation; the interval compiler above remains identical.
        self.accept = None
        if mode == "sets":
            accepted = [None] * size
            for token in range(256, size):
                lo, hi, q, end = intervals[token]
                accepted[token] = frozenset(
                    label for label, x in enumerate(self.left)
                    if (lo <= x < hi) or (q <= x < end)
                )
            self.accept = tuple(accepted)

        self.trans = (
            self._compile_dense_transitions() if self.flat_transitions else None
        )
        state_words = self._state_words()
        self.candidates, self.atomic, self.certificate_stats = (
            self._compile_candidates(state_words)
        )

    def _compile_dense_transitions(self):
        """Materialize the AC DFA as one state-major 256-way flat array."""
        ac = self.ac
        trans = [0] * (len(ac.edges) * 256)
        for byte, child in ac.edges[0].items():
            trans[byte] = child
        queue = deque(ac.edges[0].values())
        while queue:
            state = queue.popleft()
            base = state * 256
            fail_base = ac.fail[state] * 256
            trans[base:base + 256] = trans[fail_base:fail_base + 256]
            for byte, child in ac.edges[state].items():
                trans[base + byte] = child
                queue.append(child)
        return trans

    def _state_words(self):
        """Recover the trie word represented by each AC state."""
        words = [b""] * len(self.ac.edges)
        queue = deque((0,))
        while queue:
            state = queue.popleft()
            prefix = words[state]
            for byte, child in self.ac.edges[state].items():
                words[child] = prefix + bytes((byte,))
                queue.append(child)
        return tuple(words)

    @staticmethod
    def _inside(x, region):
        lo, hi, q, end = region
        return (lo <= x < hi) or (q <= x < end)

    def _broad_acceptance_certificate(self, context, region):
        """Soundly accept when every lexically compatible label is valid."""
        if not context:
            return False
        clen = len(context)
        left = self.left
        for token, word in enumerate(self.vocab):
            compatible = (
                (len(word) <= clen and context.endswith(word))
                or (len(word) > clen and word.endswith(context))
            )
            if compatible and not self._inside(left[token], region):
                return False
        return True

    def _advance_state(self, state, word):
        trans = self.trans
        if trans is not None:
            for byte in word:
                state = trans[state * 256 + byte]
            return state
        ac = self.ac
        for byte in word:
            state = ac.step(state, byte)
        return state

    def _compile_candidates(self, state_words):
        """Preorder candidates and compile bidirectional context certificates.

        Broad acceptance certificates first identify an always-valid stopping
        point in each boundary state. Thus K(C), the labels that can be selected
        in C, is conservatively the candidate prefix through that point. For a
        candidate t in current state Q, all boundary states C whose trie word
        ends in Q[:-len(s(t))] and which transition to Q on s(t) are enumerated
        statically. If every label in their K(C) lies in I_t, t is accepted; if
        none does, t is rejected. Both conclusions quantify over a sound
        superset and neither reads runtime predecessor history.
        """
        state_count = len(self.ac.output)
        ordered = [
            tuple(sorted(output, key=lambda token: -self.lengths[token]))
            for output in self.ac.output
        ]

        seed_accept = [set() for _ in range(state_count)]
        if self.certified_regions and self.certificate_mode != "none":
            for state, output in enumerate(ordered):
                path = state_words[state]
                for token in output:
                    if token < 256:
                        continue
                    off = self.lengths[self.successor[token]]
                    context = path[:-off]
                    if self._broad_acceptance_certificate(
                        context, self.intervals[token]
                    ):
                        seed_accept[state].add(token)

        # Any unresolved candidate before the first always-valid one can be
        # selected; later candidates cannot. Atomic bytes are always valid.
        possible_labels = []
        for state, output in enumerate(ordered):
            labels = []
            for token in output:
                labels.append(token)
                if token < 256 or token in seed_accept[state]:
                    break
            possible_labels.append(tuple(labels))

        want_accept = self.certified_regions and self.certificate_mode in (
            "accept_only", "accept_and_reject"
        )
        want_reject = self.certified_regions and self.certificate_mode in (
            "reject_only", "accept_and_reject"
        )
        all_candidates = []
        atomic = [-1] * state_count
        merge_pairs = accepted_pairs = rejected_pairs = unresolved_pairs = 0

        for state, output in enumerate(ordered):
            if (
                self.atomic_fastpath
                and output
                and not any(token >= 256 for token in output)
            ):
                atomic[state] = output[0]

            entries = []
            path = state_words[state]
            for token in output:
                if token < 256:
                    entries.append((token, 0, 0, 0, 0, 0, 1))
                    continue

                merge_pairs += 1
                off = self.lengths[self.successor[token]]
                region = self.intervals[token]
                verdict = 0
                context = path[:-off]

                if context and (want_accept or want_reject):
                    successor_word = self.vocab[self.successor[token]]
                    saw_label = False
                    any_inside = False
                    all_inside = True
                    for boundary_state, boundary_word in enumerate(state_words):
                        if not boundary_word.endswith(context):
                            continue
                        if self._advance_state(
                            boundary_state, successor_word
                        ) != state:
                            continue
                        for label in possible_labels[boundary_state]:
                            saw_label = True
                            inside = self._inside(self.left[label], region)
                            any_inside = any_inside or inside
                            all_inside = all_inside and inside
                    if saw_label and all_inside and want_accept:
                        verdict = 1
                    elif saw_label and not any_inside and want_reject:
                        verdict = -1

                if verdict == 1:
                    accepted_pairs += 1
                elif verdict == -1:
                    rejected_pairs += 1
                else:
                    unresolved_pairs += 1
                lo, hi, q, end = region
                entries.append((token, off, lo, hi, q, end, verdict))
            all_candidates.append(tuple(entries))

        stats = {
            "merge_candidate_pairs": merge_pairs,
            "accepted_candidate_pairs": accepted_pairs,
            "rejected_candidate_pairs": rejected_pairs,
            "unresolved_candidate_pairs": unresolved_pairs,
            "certified_candidate_pairs": accepted_pairs,
            "certified_fraction": (
                accepted_pairs / merge_pairs if merge_pairs else 0.0
            ),
            "bidirectional_certified_fraction": (
                (accepted_pairs + rejected_pairs) / merge_pairs
                if merge_pairs else 0.0
            ),
        }
        return tuple(all_candidates), tuple(atomic), stats

    def valid(self, token, n):
        """Test a candidate in O(1) for intervals/sets, or O(depth) for pointer."""
        if token < 256:
            return True
        off = self.lengths[self.successor[token]]
        index = n - off
        if index <= 0 or index >= len(self.last):
            return False
        previous = self.last[index]
        if previous is None:
            return False

        if self.acceptance_mode == "intervals":
            base = token * 4
            data = self.interval_data
            x = self.left[previous]
            return (
                (data[base] <= x < data[base + 1])
                or (data[base + 2] <= x < data[base + 3])
            )
        if self.acceptance_mode == "sets":
            return previous in self.accept[token]

        ancestor = previous
        child = -1
        wanted = self.pre[token]
        successor = self.successor
        while ancestor != wanted and successor[ancestor] >= 0:
            child = ancestor
            ancestor = successor[ancestor]
        return (
            ancestor == wanted
            and (child < 0 or self.rank[token] < self.rank[child])
        )

    def append(self, byte):
        """Append one byte and return the last token ID of the new prefix."""
        state = self.state
        if self.trans is not None:
            state = self.trans[state * 256 + byte]
        else:
            state = self.ac.step(state, byte)
        self.state = state
        n = self.length + 1
        self.length = n

        atomic = self.atomic[state]
        if atomic >= 0:
            self.last.append(atomic)
            return atomic

        mode = self.acceptance_mode
        last = self.last
        left = self.left
        chosen = byte
        for token, off, lo, hi, q, end, certificate in self.candidates[state]:
            if token < 256:
                chosen = token
                break
            index = n - off
            if index <= 0:
                continue
            if certificate < 0:
                continue
            if certificate > 0:
                chosen = token
                break

            previous = last[index]
            if mode == "intervals":
                x = left[previous]
                if (lo <= x < hi) or (q <= x < end):
                    chosen = token
                    break
            elif mode == "sets":
                if previous in self.accept[token]:
                    chosen = token
                    break
            else:
                ancestor = previous
                child = -1
                wanted = self.pre[token]
                successor = self.successor
                while ancestor != wanted and successor[ancestor] >= 0:
                    child = ancestor
                    ancestor = successor[ancestor]
                if (
                    ancestor == wanted
                    and (child < 0 or self.rank[token] < self.rank[child])
                ):
                    chosen = token
                    break

        # A byte atom is a finite exact fallback for every canonical byte
        # vocabulary, including ablations that disable all fast mechanisms.
        last.append(chosen)
        return chosen

    def tokens(self):
        """Return all token IDs in the exact BPE of the current prefix."""
        result = []
        n = self.length
        lengths = self.lengths
        last = self.last
        while n:
            token = last[n]
            result.append(token)
            n -= lengths[token]
        result.reverse()
        return result
