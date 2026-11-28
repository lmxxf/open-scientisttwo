"""Feasible active-face Newton solver and retained-factor implicit derivatives.

solve_adjoint returns the q-VJP for one adjacent-face generalized Jacobian.
At weakly active bounds it is a selection, not a unique classical derivative.
projection_directional_derivative implements the critical-cone derivative of
Euclidean projection, with the projection normal z-x supplied explicitly.
"""
import numpy as np

def lstsq(A, b, check_finite=False):
    return np.linalg.lstsq(A, b, rcond=None)


def solve_triangular(A, b, lower=False, check_finite=False):
    out = np.array(b, dtype=float, copy=True)
    indices = range(len(A)) if lower else range(len(A)-1, -1, -1)
    for i in indices:
        if lower:
            out[i] = (out[i]-A[i,:i]@out[:i])/A[i,i]
        else:
            out[i] = (out[i]-A[i,i+1:]@out[i+1:])/A[i,i]
    return out


def cho_solve(factor, rhs, check_finite=False):
    L = factor[0]
    return solve_triangular(L.T, solve_triangular(L, rhs, lower=True), lower=False)


last_tangent_Z = None
last_cholesky_L = None
last_status = None
_last_problem = None
_last_y = None
_last_critical = True
_geometry_cache = []


def _null(A, n):
    if not A.shape[0]:
        return np.eye(n)
    U, R = np.linalg.qr(A.T, mode='complete')
    diag = np.abs(np.diag(R))
    threshold = max(A.shape)*np.finfo(float).eps*diag.max(initial=0)
    rank = int(np.sum(diag > threshold))
    if rank < min(A.shape):
        _, singular, vt = np.linalg.svd(A, full_matrices=True)
        rank = int(np.sum(singular > max(A.shape)*np.finfo(float).eps*singular.max(initial=0)))
        return vt[rank:].T
    return U[:, rank:]


def _factor(H):
    # Curvature is positive for the supported convex-on-the-box problems.
    return np.linalg.cholesky(H), True


def _run(Q, q, alpha, V, H0, y, lower, upper, iterations,
         exact, caching, armijo, face_cache, base_constraints=None):
    k = V.shape[1]
    active = np.flatnonzero((y <= lower+1e-10) | (y >= upper-1e-10)).tolist()
    converged = False
    B = np.eye(k)
    Z = V
    factor = None
    for iteration in range(iterations):
        key = tuple(sorted(active))
        if caching and key in face_cache:
            B, Z, quadratic = face_cache[key]
        else:
            if base_constraints is None:
                B = _null(V[list(key)], k) if key else np.eye(k)
            else:
                C = np.vstack((base_constraints, np.eye(len(y))[list(key)]))
                B = _null(C, k)
            Z = V @ B if key or base_constraints is not None else V
            quadratic = B.T @ H0 @ B
            if caching:
                face_cache[key] = (B, Z, quadratic)
        g = Q@y + q + alpha*np.cos(y)
        reduced_g = Z.T@g
        if np.max(np.abs(reduced_g), initial=0) < 2e-10:
            if active:
                if base_constraints is None:
                    multipliers = lstsq(V[active].T, -(V.T@g), check_finite=False)[0]
                else:
                    C = np.vstack((base_constraints, np.eye(len(y))[active]))
                    multipliers = lstsq(C.T, -g, check_finite=False)[0][-len(active):]
                signs = np.where(y[active] <= lower[active]+1e-8, -1., 1.)
                bad = multipliers*signs
                j = int(np.argmin(bad))
                if bad[j] < -2e-9:
                    active.pop(j)
                    continue
            converged = True
            break
        H = quadratic - (Z.T*(alpha*np.sin(y)))@Z if exact and alpha else quadratic
        factor = _factor(H)
        step = -Z@cho_solve(factor, reduced_g, check_finite=False)
        ratios = np.full(len(y), np.inf)
        pos = step > 1e-13
        neg = step < -1e-13
        ratios[pos] = (upper[pos]-y[pos])/step[pos]
        ratios[neg] = (lower[neg]-y[neg])/step[neg]
        if active:
            ratios[active] = np.inf
        hit = int(np.argmin(ratios))
        length = min(1., max(0., ratios[hit]))
        if armijo:
            old = .5*y@(Q@y)+q@y+alpha*np.sin(y).sum()
            slope = g@step
            while length > 1e-14:
                trial = y+length*step
                value = .5*trial@(Q@trial)+q@trial+alpha*np.sin(trial).sum()
                if value <= old+1e-4*length*slope:
                    break
                length *= .5
        y = y+length*step
        if ratios[hit] <= 1. and length >= ratios[hit]-1e-14:
            active.append(hit)
    # Always retain factors at the returned candidate, including early stopping.
    key = tuple(sorted(active))
    if caching and key in face_cache:
        B, Z, quadratic = face_cache[key]
    else:
        C = V[list(key)] if base_constraints is None else np.vstack((base_constraints, np.eye(len(y))[list(key)]))
        B = _null(C, k) if C.shape[0] else np.eye(k)
        Z = V@B
        quadratic = B.T@H0@B
        if caching:
            face_cache[key] = B, Z, quadratic
    if caching:
        face_cache["terminal_active"] = key
    H = quadratic-(Z.T*(alpha*np.sin(y)))@Z if exact and alpha else quadratic
    factor = _factor(H)
    return y, Z, factor, converged


def solve(problem, config):
    global last_tangent_Z, last_cholesky_L, last_status, _last_problem, _last_y, _last_critical
    Q, q, M, b = (np.asarray(problem[t], dtype=float) for t in ('Q','q','M','b'))
    lower = np.broadcast_to(problem['lower'], q.shape)
    upper = np.broadcast_to(problem['upper'], q.shape)
    reduced = bool(config.get('nullspace_reduction', True))
    caching = bool(config.get('active_face_caching', True))
    equilibrate = bool(config.get('row_equilibration', False))
    entry = None
    if caching:
        for e in _geometry_cache:
            if e[0] is problem['Q'] and e[1] is problem['M'] and e[2] == (reduced, equilibrate):
                entry = e
                break
    if entry is None:
        A = M/np.maximum(np.linalg.norm(M,axis=1), 1e-300)[:,None] if equilibrate else M
        V = _null(A, len(q)) if reduced else np.eye(len(q))
        H0 = V.T@Q@V
        entry = (problem['Q'], problem['M'], (reduced,equilibrate), A, V, H0, {})
        if caching:
            _geometry_cache.append(entry)
            del _geometry_cache[:-8]
    _, _, _, A, V, H0, faces = entry
    rhs = b/np.maximum(np.linalg.norm(M,axis=1),1e-300) if equilibrate else b
    center = (lower+upper)*.5
    y = center+lstsq(A, rhs-A@center, check_finite=False)[0]
    if np.any(y < lower-1e-12) or np.any(y > upper+1e-12):
        # Dykstra phase I: projection onto the affine set and box.
        affine_inverse = np.linalg.pinv(A)
        u = np.zeros_like(y)
        v = np.zeros_like(y)
        for _ in range(20000):
            z = y+u
            affine = z-affine_inverse@(A@z-rhs)
            u = z-affine
            z = affine+v
            y = np.clip(z, lower, upper)
            v = z-y
            if np.max(np.abs(A@y-rhs), initial=0) < 1e-12:
                break

    if caching and reduced and faces.get('terminal_active'):
        active = list(faces['terminal_active'])
        targets = np.where(y[active] < (lower[active]+upper[active])*.5,
                           lower[active], upper[active])
        # The cached face records bound sides, independently of q and b.
        sides = faces.get('terminal_sides')
        if sides is not None:
            targets = np.where(sides, lower[active], upper[active])
        trial = y+V@lstsq(V[active], targets-y[active], check_finite=False)[0]
        if np.all(trial >= lower-1e-12) and np.all(trial <= upper+1e-12):
            y = trial
    y, Z, factor, converged = _run(
        Q,q,float(problem['alpha']),V,H0,y,lower,upper,
        max(0,int(config.get('iterations',80))), bool(config.get('exact_hessian',True)),
        caching,bool(config.get('armijo_line_search',False)),faces,
        None if reduced else A)
    if caching:
        active = list(faces.get("terminal_active", ()))
        faces["terminal_sides"] = y[active] <= lower[active]+1e-8
    last_tangent_Z = Z
    last_cholesky_L = np.tril(factor[0])
    last_status = 'converged' if converged else 'iteration_limit'
    _last_problem, _last_y = problem, y.copy()
    _last_critical = bool(config.get('critical_cone_directional_differentiation', True))
    solve.last_tangent_Z = last_tangent_Z
    solve.last_cholesky_L = last_cholesky_L
    return y


def solve_adjoint(v):
    """q-VJP: -Z R^{-1} Z.T v, using ONLY retained triangular factors.

    Tight weak bounds are fixed: this selects an adjacent-face generalized
    Jacobian. Directional derivatives at degeneracies need not equal this VJP.
    """
    if last_tangent_Z is None:
        raise RuntimeError('Call solve before solve_adjoint')
    rhs = last_tangent_Z.T@np.asarray(v)
    w = solve_triangular(last_cholesky_L, rhs, lower=True, check_finite=False)
    w = solve_triangular(last_cholesky_L.T, w, lower=False, check_finite=False)
    return -last_tangent_Z@w


def projection_directional_derivative(direction, projection_normal):
    """D P_C(z)[direction] = P_{T_C(x) intersect (z-x)^perp}(direction).

    Call solve for x first; supply the actual Euclidean projection normal z-x.
    The critical-cone toggle off fixes every tight bound, providing the stated
    ablation. This helper concerns projection input, whereas solve_adjoint
    concerns the objective's linear coefficient q. No classical derivative is
    claimed at weak activity. Cone subproblems cache their face factors locally.
    """
    if _last_y is None:
        raise RuntimeError('Call solve first')
    p, x = _last_problem, _last_y
    d = np.asarray(direction, dtype=float)
    n = np.asarray(projection_normal, dtype=float)
    tight_low = x <= np.asarray(p['lower'])+1e-8
    tight_high = x >= np.asarray(p['upper'])-1e-8
    constraints = [np.asarray(p['M'])]
    if np.linalg.norm(n) > 1e-12:
        constraints.append(n[None,:]/np.linalg.norm(n))
    if not _last_critical:
        constraints.append(np.eye(len(x))[tight_low | tight_high])
    V = _null(np.vstack(constraints), len(x))
    if not _last_critical:
        return V@(V.T@d)
    lo = np.where(tight_low, 0., -np.inf)
    hi = np.where(tight_high, 0., np.inf)
    result, _, _, _ = _run(np.eye(len(x)), -d, 0., V, np.eye(V.shape[1]),
                          np.zeros(len(x)),lo,hi,4*len(x)+20,True,True,False,{})
    return result
