"""Feasible active-face Newton solver and factor-reused local adjoint.

The adjoint differentiates q on the identified face (strict complementarity
and nonsingular face curvature are needed for a classical derivative).
"""
import numpy as np
def cholesky(a, lower=True, check_finite=False):
    return np.ascontiguousarray(np.linalg.cholesky(a))


# NumPy's linked LAPACK supplies a triangular factor solve without a SciPy
# dependency. Its symbol and integer ABI are selected explicitly at import.
import ctypes
_lapack = ctypes.CDLL(np._core._multiarray_umath.__file__)
if hasattr(_lapack, 'scipy_dpotrs_64_'):
    _potrs = _lapack.scipy_dpotrs_64_
    _integer = ctypes.c_int64
elif hasattr(_lapack, 'dpotrs_64_'):
    _potrs = _lapack.dpotrs_64_
    _integer = ctypes.c_int64
elif hasattr(_lapack, 'dpotrs_'):
    _potrs = _lapack.dpotrs_
    _integer = ctypes.c_int
else:
    raise ImportError('NumPy linked LAPACK must expose dpotrs')
_potrs.restype = None


def cho_solve(factor, rhs, check_finite=False):
    L = factor[0]
    x = np.array(rhs, dtype=float, order='F', copy=True)
    if not len(L):
        return x
    n = _integer(len(L))
    nrhs = _integer(1 if x.ndim == 1 else x.shape[1])
    info = _integer(0)
    # C-order lower L is the F-order upper factor L.T.
    _potrs(ctypes.c_char_p(b'U'), ctypes.byref(n), ctypes.byref(nrhs),
           ctypes.c_void_p(L.ctypes.data), ctypes.byref(n),
           ctypes.c_void_p(x.ctypes.data), ctypes.byref(n), ctypes.byref(info))
    if info.value:
        raise ArithmeticError('LAPACK dpotrs argument error')
    return x

_FINAL = None


def adjoint(cotangent):
    """Return (dy/dq).T @ cotangent using the last solve's exact factor."""
    if _FINAL is None:
        raise RuntimeError('solve must precede adjoint')
    Z, factor = _FINAL
    v = np.asarray(cotangent, dtype=float)
    return -Z @ cho_solve((factor, True), Z.T @ v, check_finite=False)


def solve(problem, config):
    global _FINAL
    _FINAL = None
    Q, q, M, b = (np.asarray(problem[n], dtype=float) for n in ('Q', 'q', 'M', 'b'))
    lo, hi = (np.broadcast_to(problem[n], q.shape) for n in ('lower', 'upper'))
    alpha = float(problem['alpha'])
    d = q.size
    reduction = config.get('nullspace_reduction', True)
    exact = config.get('exact_hessian', True)
    caching = config.get('active_face_caching', True)
    armijo = config.get('armijo_line_search', False)
    if config.get('row_equilibration', False):
        scale = np.linalg.norm(M, axis=1)
        scale = np.where(scale > 0, scale, 1.)
        M, b = M / scale[:, None], b / scale
    # A complete QR costs O(d*m^2); use SVD only for dependent rows.
    U, triangular = np.linalg.qr(M.T, mode='complete')
    diag = np.abs(np.diag(triangular))
    rank = int(np.sum(diag > max(M.shape) * np.finfo(float).eps * max(diag.max(initial=0.), 1.)))
    if rank == M.shape[0]:
        E = U[:, :rank]
        coeff = np.linalg.solve(triangular[:rank, :rank].T, b)
    else:
        Um, diag, Ut = np.linalg.svd(M, full_matrices=True)
        U = Ut.T
        rank = int(np.sum(diag > max(M.shape) * np.finfo(float).eps * max(diag.max(initial=0.), 1.)))
        E = U[:, :rank]
        coeff = (Um[:, :rank].T @ b) / diag[:rank]
    center = (lo + hi) * .5
    y = center + E @ (coeff - E.T @ center)
    V = U[:, rank:] if reduction else np.eye(d)
    H0 = V.T @ Q @ V if reduction else Q
    base = np.empty((0, V.shape[1])) if reduction else M
    # Phase I is only needed when the affine projection of the box center
    # lies outside the box. It uses no objective or reference information.
    if np.any(y < lo - 1e-12) or np.any(y > hi + 1e-12):
        correction = np.zeros(d)
        for phase_iteration in range(20000):
            clipped = np.clip(y + correction, lo, hi)
            correction += y - clipped
            y = clipped + E @ (coeff - E.T @ clipped)
            if max(np.max(lo - y), np.max(y - hi)) < 2e-13:
                break
        else:
            raise ValueError('Feasibility phase did not converge')
    active = list(np.flatnonzero((y <= lo + 1e-11) | (y >= hi - 1e-11)))
    cache = {}
    rank_failures = 0
    previous = None
    incremental_updates = 0

    def geometry(indices):
        nonlocal rank_failures, previous, incremental_updates
        key = tuple(indices)
        if caching and key in cache:
            previous = (key, cache[key])
            return cache[key]
        if caching and previous is not None and key[:-1] == previous[0] and len(key) == len(previous[0]) + 1:
            old_B, old_Z, old_R = previous[1]
            row = V[indices[-1]] @ old_B
            norm = np.linalg.norm(row)
            if norm > 1e-11:
                u = row.copy()
                u[0] += np.copysign(norm, u[0])
                u /= np.linalg.norm(u)
                B = (old_B - 2 * np.outer(old_B @ u, u))[:, 1:]
                Z = (old_Z - 2 * np.outer(old_Z @ u, u))[:, 1:]
                ru = old_R @ u
                R0 = (old_R - 2*np.outer(u, ru) - 2*np.outer(ru, u)
                      + 4*(u @ ru)*np.outer(u, u))[1:, 1:]
                # Rebuild from the small active-row matrix if accumulated
                # roundoff fails the newly imposed tangent constraint.
                if np.max(np.abs(V[indices[-1]] @ B), initial=0.) <= 1e-11:
                    value = (B, Z, R0)
                    cache[key] = value
                    previous = (key, value)
                    incremental_updates += 1
                    return value
            rank_failures += 1
        rows = np.vstack((base, V[indices])) if indices else base
        if not len(rows):
            B = np.eye(V.shape[1])
        else:
            W, triangular = np.linalg.qr(rows.T, mode='complete')
            dd = np.abs(np.diag(triangular))
            rr = int(np.sum(dd > 1e-11 * max(dd.max(initial=0.), 1.)))
            if rr < len(rows):
                _, dd, Wt = np.linalg.svd(rows, full_matrices=True)
                W = Wt.T
                rr = int(np.sum(dd > 1e-11 * max(dd.max(initial=0.), 1.)))
                rank_failures += 1
            B = W[:, rr:]
        Z = V @ B if reduction else B
        R0 = B.T @ H0 @ B
        value = (B, Z, R0)
        if caching:
            cache[key] = value
        previous = (key, value)
        return value

    # Predict the face in batches from the equality-constrained minimizer.
    # This avoids one full Newton factorization per bound hit. Only a feasible
    # predictor is accepted; otherwise the feasible primal iteration starts
    # from its original affine-box point.
    if caching and reduction and V.shape[1] and int(config.get('iterations', 100)) > 0:
        trial = y.copy()
        for _ in range(3):
            grad = V.T @ (Q @ trial + q + alpha * np.cos(trial))
            curvature = H0 - alpha * (V.T * np.sin(trial)) @ V if exact and alpha else H0
            direction = -V @ cho_solve((cholesky(curvature), True), grad)
            length = 1.
            if armijo:
                value = .5 * trial @ Q @ trial + q @ trial + alpha * np.sin(trial).sum()
                slope = (Q @ trial + q + alpha * np.cos(trial)) @ direction
                while length > 1e-14:
                    candidate = trial + length * direction
                    objective = .5 * candidate @ Q @ candidate + q @ candidate + alpha * np.sin(candidate).sum()
                    if objective <= value + 1e-4 * length * slope:
                        break
                    length *= .5
            trial += length * direction
        predicted = list(np.flatnonzero((trial < lo) | (trial > hi)))
        sides = np.where(trial < lo, lo, hi)
        for _ in range(8):
            Bp, Zp, Rp = geometry(predicted)
            if predicted:
                rows = V[predicted]
                correction = np.linalg.lstsq(rows, sides[predicted] - trial[predicted], rcond=1e-11)[0]
                trial += V @ correction
            for _ in range(4):
                grad = Zp.T @ (Q @ trial + q + alpha * np.cos(trial))
                if np.max(np.abs(grad), initial=0.) < 2e-11:
                    break
                curvature = Rp - alpha * (Zp.T * np.sin(trial)) @ Zp if exact and alpha else Rp
                direction = -Zp @ cho_solve((cholesky(curvature), True), grad)
                length = 1.
                if armijo:
                    value = .5 * trial @ Q @ trial + q @ trial + alpha * np.sin(trial).sum()
                    slope = (Q @ trial + q + alpha * np.cos(trial)) @ direction
                    while length > 1e-14:
                        candidate = trial + length * direction
                        objective = .5 * candidate @ Q @ candidate + q @ candidate + alpha * np.sin(candidate).sum()
                        if objective <= value + 1e-4 * length * slope:
                            break
                        length *= .5
                trial += length * direction
            violated = np.flatnonzero((trial < lo - 1e-11) | (trial > hi + 1e-11))
            additions = [int(i) for i in violated if i not in predicted]
            if additions:
                for i in additions:
                    sides[i] = lo[i] if trial[i] < lo[i] else hi[i]
                predicted.extend(additions)
                continue
            if not len(violated):
                y = trial
                active = predicted
            break
    converged = False
    for iteration in range(max(0, int(config.get('iterations', 100)))):
        B, Z, R0 = geometry(active)
        g = Q @ y + q + alpha * np.cos(y)
        rg = Z.T @ g
        if rg.size and np.max(np.abs(rg)) > 2e-11:
            R = R0 - alpha * (Z.T * np.sin(y)) @ Z if exact and alpha else R0
            L = cholesky(R, lower=True, check_finite=False)
            step = -Z @ cho_solve((L, True), rg, check_finite=False)
        else:
            step = np.zeros(d)
        if np.max(np.abs(step), initial=0.) < 1e-10:
            if active:
                # Multipliers in reduced coordinates avoid ill-scaled M.
                A = V[active] if reduction else np.vstack((M, np.eye(d)[active]))
                multipliers = np.linalg.lstsq(A.T, -(V.T @ g), rcond=1e-11)[0]
                multipliers = multipliers[-len(active):]
                signs = np.where(y[active] <= lo[active] + 1e-8, -1., 1.)
                dual = multipliers * signs
                if dual.min(initial=0.) < -1e-9:
                    active.pop(int(np.argmin(dual)))
                    continue
            converged = True
            break
        ratios = np.full(d, np.inf)
        pos = step > 1e-13
        neg = step < -1e-13
        ratios[pos] = (hi[pos] - y[pos]) / step[pos]
        ratios[neg] = (lo[neg] - y[neg]) / step[neg]
        ratios[active] = np.inf
        hit = int(np.argmin(ratios))
        length = min(1., max(0., ratios[hit]))
        if armijo:
            value = .5 * y @ Q @ y + q @ y + alpha * np.sin(y).sum()
            slope = g @ step
            while length > 1e-14:
                trial = y + length * step
                obj = .5 * trial @ Q @ trial + q @ trial + alpha * np.sin(trial).sum()
                if obj <= value + 1e-4 * length * slope:
                    break
                length *= .5
        y = y + length * step
        if ratios[hit] <= 1. and abs(length - ratios[hit]) < 1e-13:
            active.append(hit)
    # Always cache curvature at the returned point, including early stopping
    # and the approximate-Hessian ablation. The derivative itself is exact.
    B, Z, R0 = geometry(active)
    R = R0 - alpha * (Z.T * np.sin(y)) @ Z if alpha else R0
    L = cholesky(R, lower=True, check_finite=False)
    _FINAL = (Z, L)
    solve.diagnostics = dict(converged=converged, iterations=iteration + 1 if int(config.get('iterations', 100)) > 0 else 0,
                             rank_failures=rank_failures, incremental_updates=incremental_updates,
                             cache_bytes=sum(sum(a.nbytes for a in entry) for entry in cache.values()))
    return y
