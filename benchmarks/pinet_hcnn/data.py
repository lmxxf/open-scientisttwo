"""Context-dependent dense QPs with independently certified KKT reference solutions."""
import numpy as np

def objective(p,y):
    return float(.5*y@p['Q']@y+p['q']@y+p['alpha']*np.sin(y).sum())

def instances(mode,split,extra=False):
    rng=np.random.default_rng((103 if split=='validation' else 701)+(10000 if extra else 0))
    dimensions=[12,24] if mode=='subset' else [24,48,64]
    result=[]
    for d in dimensions:
        for alpha in [0.,.8]:
            U,_=np.linalg.qr(rng.normal(size=(d,d)))
            Q=(U*np.linspace(.3,3.,d))@U.T
            M=rng.normal(size=(d//3,d))*np.geomspace(.1,10,d//3)[:,None]
            lower=np.full(d,-.25);upper=np.full(d,.25)
            optimum=rng.uniform(-.18,.18,d);optimum[0]=lower[0];optimum[1]=upper[1]
            normal=np.zeros(d);normal[0]=-.12;normal[1]=.12
            lam=rng.normal(size=d//3)*.05
            q=-Q@optimum-alpha*np.cos(optimum)-M.T@lam-normal
            p=dict(Q=Q,q=q,M=M,b=M@optimum,lower=lower,upper=upper,alpha=alpha)
            # Positive curvature on the box proves the KKT point is globally optimal there.
            certificate=float(np.linalg.eigvalsh(Q)[0]-alpha*np.sin(.25))
            if certificate<=0:raise ValueError('reference convexity certificate failed')
            result.append((p,optimum,dict(curvature_lower_bound=certificate,lambda_=lam,normal=normal)))
    return result
