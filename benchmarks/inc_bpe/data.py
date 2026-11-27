"""Ranked byte BPE vocabulary, independent reference and AC suffix recognizer."""
from collections import Counter, deque
import random

CORPUS = ['the quick brown fox and the tokenizer stream. ', '增量分词保持每个前缀完全一致。中文编码测试。',
          'def encode(data):\n    return [token for token in data]\n']

def vocabulary():
    encoded=[list(t.encode()) for t in CORPUS]
    vocab=[bytes([i]) for i in range(256)];rules=[]
    for _ in range(72):
        counts=Counter(pair for seq in encoded for pair in zip(seq,seq[1:]))
        valid=[p for p in counts if vocab[p[0]]+vocab[p[1]] not in vocab]
        if not valid:break
        pair=min(valid,key=lambda p:(-counts[p],p));new=len(vocab)
        vocab.append(vocab[pair[0]]+vocab[pair[1]]);rules.append(pair)
        for seq in encoded:
            i=0
            while i<len(seq)-1:
                if tuple(seq[i:i+2])==pair:seq[i:i+2]=[new]
                else:i+=1
    # Reject noncanonical tokens rather than silently changing dictionary semantics.
    if any(reference(v,rules)!=[i] for i,v in enumerate(vocab)):
        raise ValueError('noncanonical vocabulary')
    return vocab,rules

class SuffixAutomaton:
    def __init__(self,vocab):
        self.edges=[{}];self.fail=[0];self.output=[[]]
        for token,word in enumerate(vocab):
            node=0
            for c in word:
                if c not in self.edges[node]:
                    self.edges[node][c]=len(self.edges);self.edges.append({});self.fail.append(0);self.output.append([])
                node=self.edges[node][c]
            self.output[node].append(token)
        q=deque(self.edges[0].values())
        while q:
            u=q.popleft()
            for c,v in self.edges[u].items():
                f=self.fail[u]
                while f and c not in self.edges[f]:f=self.fail[f]
                self.fail[v]=self.edges[f].get(c,0)
                self.output[v]+=self.output[self.fail[v]];q.append(v)
    def step(self,node,c):
        while node and c not in self.edges[node]:node=self.fail[node]
        return self.edges[node].get(c,0)

def streams(mode,split,extra=False):
    rng=random.Random((410 if split=='validation' else 991)+(10000 if extra else 0))
    size=160 if mode=='subset' else 768
    return [((''.join(rng.choice(CORPUS[i].splitlines(keepends=True)) for _ in range(80))).encode()[:size]) for i in range(3)]


class SufSucTree:
    """Successor forest, pointer ancestry and independently usable DFS intervals."""
    def __init__(self,vocab,rules):
        self.successor={256+i:b for i,(a,b) in enumerate(rules)}
        self.children={i:[] for i in range(len(vocab))}
        for t,parent in self.successor.items():self.children[parent].append(t)
        self.left={};self.right={};clock=0
        def visit(t):
            nonlocal clock
            self.left[t]=clock;clock+=1
            for child in self.children[t]:visit(child)
            self.right[t]=clock
        for t in range(256):visit(t)
    def contains(self,root,node):
        return self.left[root]<=self.left[node]<self.right[root]

# Independent global greedy reference: linked list + priority heap, never successor trees.
def reference(raw,rules):
    import heapq
    ids=list(raw);n=len(ids)
    if not n:return []
    nxt=list(range(1,n))+[-1];prev=[-1]+list(range(n-1));alive=[True]*n;heap=[]
    ranks={pair:(i,256+i) for i,pair in enumerate(rules)}
    def push(i):
        if i<0 or not alive[i]:return
        j=nxt[i]
        if j>=0 and (ids[i],ids[j]) in ranks:
            rank,t=ranks[(ids[i],ids[j])];heapq.heappush(heap,(rank,i,j,ids[i],ids[j],t))
    for i in range(n):push(i)
    while heap:
        rank,i,j,a,b,t=heapq.heappop(heap)
        if not alive[i] or not alive[j] or nxt[i]!=j or ids[i]!=a or ids[j]!=b:continue
        ids[i]=t;alive[j]=False;nxt[i]=nxt[j]
        if nxt[j]>=0:prev[nxt[j]]=i
        push(prev[i]);push(i)
    return [t for i,t in enumerate(ids) if alive[i]]
