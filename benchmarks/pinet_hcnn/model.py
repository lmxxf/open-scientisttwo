"""Unpreconditioned lifted forward Douglas--Rachford projection baseline.

z=(y,s), A z=(b,0) with A=[[M,0],[I,-I]]. Box projection acts on s;
forward objective gradient acts on y. No nullspace reduction or row equilibration.
"""
import numpy as np

def solve(problem,config):
    Q,q,M,b=(problem[k] for k in ('Q','q','M','b'));d=len(q)
    A=np.block([[M,np.zeros_like(M)],[np.eye(d),-np.eye(d)]])
    target=np.r_[b,np.zeros(d)];z=np.zeros(2*d)
    # Redundant lifted solve intentionally remains inside each iteration.
    for _ in range(int(config.get('iterations',40))):
        x=z.copy();x[d:]=np.clip(x[d:],problem['lower'],problem['upper'])
        grad=np.r_[Q@x[:d]+q+problem['alpha']*np.cos(x[:d]),np.zeros(d)]
        v=2*x-z-float(config.get('step',.08))*grad
        projected=v-A.T@np.linalg.solve(A@A.T,A@v-target)
        z+=projected-x
    return projected[:d] if int(config.get('iterations',40)) else z[:d]
