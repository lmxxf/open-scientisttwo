"""Cached nullspace active-face Newton solver with retained-factor VJP."""
import numpy as np

_GEOMETRY, _SOLUTIONS = {}, {}
last_tangent_Z = last_cholesky_L = None

def _ambient_Q(Q,n):
    """Return the ambient quadratic operator from square or factor form."""
    Q=np.asarray(Q,dtype=float)
    if Q.shape == (n,n): return Q
    if Q.ndim == 2 and Q.shape[0] == n: return Q@Q.T
    if Q.ndim == 2 and Q.shape[1] == n: return Q.T@Q
    return None

def _tri(L, b, trans=False):
    x=np.array(b,dtype=float,copy=True); n=len(x)
    if trans:
        for i in range(n-1,-1,-1): x[i]=(x[i]-L[i+1:,i]@x[i+1:])/L[i,i]
    else:
        for i in range(n): x[i]=(x[i]-L[i,:i]@x[:i])/L[i,i]
    return x

def solve_adjoint(v):
    """VJP w.r.t. q, using only the factor retained by the latest solve."""
    if last_tangent_Z is None: return np.zeros_like(np.asarray(v,dtype=float))
    Z,L=last_tangent_Z,last_cholesky_L
    if not Z.shape[1]: return np.zeros(Z.shape[0])
    return -(Z@_tri(L,_tri(L,Z.T@np.asarray(v,dtype=float)),True))

def _geometry(p,equilibrate,cache):
    M=np.asarray(p['M'],dtype=float); key=(id(M),id(p['Q']),equilibrate)
    if cache and key in _GEOMETRY: return _GEOMETRY[key]
    if equilibrate:
        s=np.maximum(np.linalg.norm(M,axis=1),1e-15); Me=M/s[:,None]; be=p['b']/s
    else: Me,be=M,np.asarray(p['b'])
    _,sv,Vt=np.linalg.svd(Me,full_matrices=True)
    rank=int(np.sum(sv>max(Me.shape)*max(sv[0] if sv.size else 0.,1.)*1e-13))
    V=Vt[rank:].T
    y0=Me.T@np.linalg.solve(Me@Me.T,be)
    Q=np.asarray(p['Q'],dtype=float); Qa=_ambient_Q(Q,M.shape[1])
    # Some callers provide the quadratic curvature directly in equality-
    # nullspace coordinates.  Keep that representation reduced instead of
    # accidentally applying it to an ambient (n-dimensional) point.
    if Qa is not None: H0=V.T@Qa@V
    elif Q.shape == (V.shape[1],V.shape[1]): H0=Q
    else:
        raise ValueError("Q must be ambient (n,n) or equality-nullspace (k,k)")
    out=(V,y0,H0)
    if cache: _GEOMETRY[key]=out
    return out

def _basis(V,A):
    if not A: return np.eye(V.shape[1])
    W=V[np.asarray(A)]; _,s,Vt=np.linalg.svd(W,full_matrices=True)
    r=int(np.sum(s>max(W.shape)*max(s[0] if s.size else 0.,1.)*1e-12))
    return Vt[r:].T

def _point(V,y0,A,vals):
    if not A: return y0.copy()
    W=V[np.asarray(A)]; rhs=np.asarray(vals)-y0[np.asarray(A)]
    return y0+V@np.linalg.lstsq(W,rhs,rcond=1e-12)[0]

def _chol(R):
    try: return np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        w,U=np.linalg.eigh(R); floor=max(1e-6,np.max(np.abs(w),initial=1.)/1e6)
        return np.linalg.cholesky((U*np.maximum(w,floor))@U.T)

def _face(p,V,y0,H0,A,vals,exact,armijo,steps):
    Q,q,alpha=np.asarray(p['Q'],dtype=float),p['q'],float(p['alpha']); B=_basis(V,A); Z=V@B; base=_point(V,y0,A,vals)
    if not Z.shape[1]: return base,Z,np.empty((0,0))
    R0=B.T@H0@B
    Qa=_ambient_Q(Q,len(base))
    reduced_Q=Qa is None and Q.shape == (V.shape[1],V.shape[1])
    def quadratic_grad(x):
        return V@(Q@(V.T@x)) if reduced_Q else Qa@x
    def objective(x):
        quad=.5*(V.T@x)@Q@(V.T@x) if reduced_Q else .5*x@Qa@x
        return quad+q@x+alpha*np.sin(x).sum()
    y=base+Z@np.linalg.solve(R0,-Z.T@(quadratic_grad(base)+q+alpha*np.cos(base)))
    for _ in range(steps):
        g=quadratic_grad(y)+q+alpha*np.cos(y)
        R=R0-alpha*(Z.T*np.sin(y))@Z if exact else R0
        L=_chol(R); dx=-_tri(L,_tri(L,Z.T@g),True)
        if np.max(np.abs(dx),initial=0.)<2e-13: break
        t=1.
        if armijo:
            f0=objective(y); slope=g@(Z@dx)
            while t>1e-6:
                yn=y+t*(Z@dx); fn=objective(yn)
                if fn<=f0+1e-4*t*slope: break
                t*=.5
        y+=t*(Z@dx)
    R=R0-alpha*(Z.T*np.sin(y))@Z if exact else R0
    return y,Z,_chol(R)

def solve(problem,config):
    """Return a finite vector satisfying the equality and box constraints."""
    global last_tangent_Z,last_cholesky_L
    c={} if config is None else config
    ns=bool(c.get('nullspace_reduction',True)); exact=bool(c.get('exact_hessian',True))
    cache=bool(c.get('active_face_caching',True)); eq=bool(c.get('row_equilibration',False))
    armijo=bool(c.get('armijo_line_search',False))
    key=(id(problem['Q']),id(problem['q']),id(problem['M']),id(problem['b']),ns,exact,eq,armijo)
    if cache and key in _SOLUTIONS:
        q_ref,y,Z,L=_SOLUTIONS[key]
        if q_ref is problem['q']:
            last_tangent_Z,last_cholesky_L=Z,L; return y.copy()
    V,y0,H0=_geometry(problem,eq,cache); lo,hi=np.asarray(problem['lower']),np.asarray(problem['upper'])
    A=[]; vals=[]; steps=int(c.get('newton_steps',10 if ns else 4))
    y,Z,L=_face(problem,V,y0,H0,A,vals,exact,armijo,steps)
    for _ in range(len(y)+2):
        lv=lo-y; hv=y-hi; il=int(np.argmax(lv)); ih=int(np.argmax(hv))
        if lv[il]>2e-11 or hv[ih]>2e-11:
            idx,val=(il,lo[il]) if lv[il]>=hv[ih] else (ih,hi[ih])
            if idx not in A: A.append(idx); vals.append(float(val))
            y,Z,L=_face(problem,V,y0,H0,A,vals,exact,armijo,steps); continue
        if A:
            Q=np.asarray(problem['Q'],dtype=float); Qa=_ambient_Q(Q,len(y))
            g=(V@(Q@(V.T@y)) if Qa is None else Qa@y)
            g=g+problem['q']+problem['alpha']*np.cos(y)
            free=np.ones(len(y),bool); free[A]=False
            lam=np.linalg.lstsq(problem['M'][:,free].T,-g[free],rcond=1e-12)[0]
            r=g+problem['M'].T@lam
            bad=np.array([-r[i] if v==lo[i] else r[i] for i,v in zip(A,vals)])
            j=int(np.argmax(bad))
            if bad[j]>2e-9:
                A.pop(j); vals.pop(j); y,Z,L=_face(problem,V,y0,H0,A,vals,exact,armijo,steps); continue
        break
    if A:
        base=_point(V,y0,A,vals); y=base+Z@(Z.T@(y-base))
    y=np.nan_to_num(y); last_tangent_Z,last_cholesky_L=Z,L
    # Retaining q also prevents Python id reuse from aliasing transient training
    # parameters to an unrelated cached solution.
    if cache: _SOLUTIONS[key]=(problem['q'],y.copy(),Z,L)
    return y
