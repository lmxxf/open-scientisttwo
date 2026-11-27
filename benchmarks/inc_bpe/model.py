"""Pointer-based suffix-successor traversal; exact incremental ranked BPE."""
from data import SuffixAutomaton,SufSucTree

class IncrementalBPE:
    def __init__(self,vocab,rules,config=None):
        self.vocab,self.rules=vocab,rules
        self.ac=SuffixAutomaton(vocab);self.state=0;self.last=[None];self.length=0
        self.tree=SufSucTree(vocab,rules);self.successor=self.tree.successor
        self.pre={256+i:a for i,(a,b) in enumerate(rules)}
        self.rank={256+i:i for i in range(len(rules))}
    def valid(self,t,n):
        if t<256:return True
        k=self.last[n-len(self.vocab[self.successor[t]])]
        pre=self.pre[t];child=None
        while k!=pre and k in self.successor:
            child=k;k=self.successor[k]
        return k==pre and (child is None or self.rank[t]<self.rank[child])
    def append(self,byte):
        self.state=self.ac.step(self.state,byte);self.length+=1
        candidates=sorted(self.ac.output[self.state],key=lambda t:-len(self.vocab[t]))
        token=next(t for t in candidates if self.valid(t,self.length))
        self.last.append(token)
        return token
    def tokens(self):
        result=[];n=self.length
        while n:
            t=self.last[n];result.append(t);n-=len(self.vocab[t])
        return result[::-1]
