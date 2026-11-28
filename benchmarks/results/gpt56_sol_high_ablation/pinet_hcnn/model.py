"""Fast nullspace active-face Newton solver with factor-reused adjoints.

Static equality geometry is eliminated once. A box face is identified by its
sorted lower/upper constraint ids; its tangent is obtained from the small
active-row matrix, or from a neighbouring cached face by QR updates. The
terminal tangent and Cholesky factor are retained for implicit VJPs.
"""

import numpy as np


last_tangent_Z = None
last_cholesky_L = None

_last_state = None
_geometry_cache = {}
_face_cache = {}
_solution_cache = {}


def _put(cache, key, value, limit):
    """Small insertion-ordered cache; entries retain arrays to defeat id reuse."""
    if key in cache:
        cache.pop(key)
    while len(cache) >= max(1, int(limit)):
        cache.pop(next(iter(cache)))
    cache[key] = value


def _same_objects(entry, objects):
    return entry is not None and all(a is b for a, b in zip(entry[0], objects))


def _geometry(problem, row_equilibration, cache_size):
    """Compute a particular point, equality nullspace V, and H0=V.T Q V."""
    Q, M, b = problem['Q'], problem['M'], problem['b']
    objects = (Q, M, b)
    key = (id(Q), id(M), id(b), bool(row_equilibration))
    cached = _geometry_cache.get(key)
    if _same_objects(cached, objects):
        return cached[1]

    if row_equilibration:
        scale = np.maximum(np.linalg.norm(M, axis=1), np.finfo(float).tiny)
        Me, be = M / scale[:, None], b / scale
    else:
        Me, be = M, b

    m = Me.shape[0]
    U, R = np.linalg.qr(Me.T, mode='complete')
    diagonal = np.abs(np.diag(R[:m, :m]))
    threshold = diagonal.max(initial=0.0) * max(Me.shape) * np.finfo(float).eps
    rank = int(np.sum(diagonal > threshold))
    if rank == m:
        yp = U[:, :m] @ np.linalg.solve(R[:m, :m].T, be)
        V = U[:, m:]
    else:
        Ul, singular, Vt = np.linalg.svd(Me, full_matrices=True)
        threshold = singular.max(initial=0.0) * max(Me.shape) * np.finfo(float).eps
        rank = int(np.sum(singular > threshold))
        yp = Vt[:rank].T @ ((Ul[:, :rank].T @ be) / singular[:rank])
        V = Vt[rank:].T

    result = (yp, V, V.T @ Q @ V, key)
    _put(_geometry_cache, key, (objects, result), cache_size)
    return result


def _chol(matrix):
    if matrix.size == 0:
        return np.empty((0, 0))
    matrix = 0.5 * (matrix + matrix.T)
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(matrix)
        floor = max(np.max(np.abs(values)), 1.0) * 1e-12
        repaired = (vectors * np.maximum(values, floor)) @ vectors.T
        return np.linalg.cholesky(0.5 * (repaired + repaired.T))


def _constraint_id(index, value, lower):
    return 2 * int(index) + int(value != lower[index])


def _canonical_face(active, values, lower):
    pairs = sorted((_constraint_id(i, v, lower), int(i), float(v))
                   for i, v in zip(active, values))
    return tuple(p[0] for p in pairs), [p[1] for p in pairs], np.asarray(
        [p[2] for p in pairs], dtype=float)


def _refactor_face(yp, V, H0, active, values, signature):
    """Reference face construction: SVD only the |A| by (d-m) matrix V_A."""
    n = V.shape[1]
    if not active:
        B = np.eye(n)
        return dict(signature=signature, yface=yp.copy(), B=B, Z=V,
                    Hquad=H0, Lquad=_chol(H0))
    VA = V[active, :]
    rhs = values - yp[active]
    Ua, singular, Vta = np.linalg.svd(VA, full_matrices=True)
    threshold = singular.max(initial=0.0) * max(VA.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > threshold))
    xpart = (Vta[:rank].T @ ((Ua[:, :rank].T @ rhs) / singular[:rank])
             if rank else np.zeros(n))
    B = Vta[rank:].T
    Z = V @ B
    Hquad = B.T @ H0 @ B
    return dict(signature=signature, yface=yp + V @ xpart, B=B, Z=Z,
                Hquad=Hquad, Lquad=_chol(Hquad))


def _updated_face(parent, active, values, signature, condition_threshold):
    """Add constraints with small QR updates to a neighbouring face."""
    parent_ids = set(parent['signature'])
    additions = [(cid, idx, val) for cid, idx, val in
                 zip(signature, active, values) if cid not in parent_ids]
    B = parent['B']
    Z = parent['Z']
    yface = parent['yface'].copy()
    Hquad = parent['Hquad']
    Lquad = parent['Lquad']
    used = set(parent_ids)
    for cid, idx, value in additions:
        row = Z[idx, :]
        norm = float(np.linalg.norm(row))
        residual = float(value - yface[idx])
        if norm <= 1e-12:
            if abs(residual) > 1e-9:
                return None
            used.add(cid)
            continue
        yface += Z @ (row * (residual / (norm * norm)))
        # Complete QR of one k-vector is the Householder update. Only this
        # small tangent-coordinate matrix is factorized.
        householder, _ = np.linalg.qr(row[:, None], mode='complete')
        N = householder[:, 1:]
        B = B @ N
        Z = Z @ N
        Hquad = N.T @ Hquad @ N
        if N.shape[1]:
            # QR of L.T N updates the Cholesky factor because
            # N.T H N = (L.T N).T (L.T N).
            _, upper = np.linalg.qr(Lquad.T @ N, mode='reduced')
            Lquad = upper.T
            diagonal = np.abs(np.diag(Lquad))
            condition = (diagonal.max(initial=0.0) /
                         max(diagonal.min(initial=np.inf), np.finfo(float).tiny))
            if not np.isfinite(condition) or condition > condition_threshold:
                Lquad = _chol(Hquad)
        else:
            Lquad = np.empty((0, 0))
        used.add(cid)
    if used != set(signature):
        return None
    return dict(signature=signature, yface=yface, B=B, Z=Z,
                Hquad=Hquad, Lquad=Lquad)


def _face_geometry(problem, yp, V, H0, geometry_key, active, values,
                   config, parent=None):
    """Get a face by signature, using cached/update geometry when enabled."""
    signature, active, values = _canonical_face(
        active, values, problem['lower'])
    caching = bool(config.get('active_face_caching', True))
    cache_size = int(config.get('cache_size', 128))
    prefix = (geometry_key, id(problem['lower']), id(problem['upper']))
    key = prefix + (signature,)
    objects = (problem['Q'], problem['M'], problem['b'],
               problem['lower'], problem['upper'])
    cached = _face_cache.get(key) if caching else None
    if _same_objects(cached, objects):
        return cached[1]

    update_method = str(config.get('update_method', 'qr_update_downdate'))
    max_update = max(0, int(config.get('max_rank_update', 4)))
    condition_threshold = float(config.get('refactor_condition_threshold', 1e8))
    candidate = None
    if update_method == 'qr_update_downdate':
        target = set(signature)
        if (parent is not None and set(parent['signature']).issubset(target)
                and len(target) - len(parent['signature']) <= max_update):
            candidate = parent
        elif (caching and bool(config.get('batch_grouping', True))
              and max_update):
            # Batch calls with a neighbouring signature can update from an
            # earlier face without touching a d-dimensional KKT matrix.
            best_size = -1
            for fkey, record in _face_cache.items():
                if fkey[:3] != prefix or not _same_objects(record, objects):
                    continue
                face = record[1]
                old = set(face['signature'])
                if (old.issubset(target) and len(target) - len(old) <= max_update
                        and len(old) > best_size):
                    candidate, best_size = face, len(old)
    face = None
    if candidate is not None:
        face = _updated_face(candidate, active, values, signature,
                             condition_threshold)
    if face is None:
        face = _refactor_face(yp, V, H0, active, values, signature)
    if caching:
        _put(_face_cache, key, (objects, face), cache_size)
    return face


def _objective(problem, y):
    return float(0.5 * y @ problem['Q'] @ y + problem['q'] @ y
                 + problem['alpha'] * np.sin(y).sum())


def _face_newton(problem, face, config):
    Z = face['Z']
    if Z.shape[1] == 0:
        return face['yface'].copy(), np.empty((0, 0))
    Q, q, alpha = problem['Q'], problem['q'], float(problem['alpha'])
    exact = bool(config.get('exact_hessian', True))
    armijo = bool(config.get('armijo_line_search', False))
    iterations = int(config.get('newton_max_steps', 8 if exact else 18))
    y = face['yface'].copy()
    L = face['Lquad']
    for _ in range(max(0, iterations)):
        gradient = Q @ y + q + alpha * np.cos(y)
        reduced_gradient = Z.T @ gradient
        if np.linalg.norm(reduced_gradient, ord=np.inf) <= 2e-12:
            break
        if exact and alpha:
            reduced_hessian = (face['Hquad'] -
                alpha * (Z.T @ (np.sin(y)[:, None] * Z)))
            L = _chol(reduced_hessian)
        step = -np.linalg.solve(L.T, np.linalg.solve(L, reduced_gradient))
        direction = Z @ step
        rate = 1.0
        if armijo:
            f0 = _objective(problem, y)
            slope = float(gradient @ direction)
            while rate > 2.0 ** -16:
                trial = y + rate * direction
                if _objective(problem, trial) <= f0 + 1e-4 * rate * slope:
                    break
                rate *= 0.5
        trial = y + rate * direction
        if not np.isfinite(trial).all():
            break
        y = trial
        if np.linalg.norm(rate * step, ord=np.inf) <= 2e-13:
            break
    if exact and alpha:
        L = _chol(face['Hquad'] -
                  alpha * (Z.T @ (np.sin(y)[:, None] * Z)))
    return y, L


def _fullspace_newton(problem, active, values, config):
    """Explicit ablation path when equality nullspace reduction is disabled."""
    Q, q, M, b = (problem[k] for k in ('Q', 'q', 'M', 'b'))
    d = len(q)
    C = np.zeros((M.shape[0] + len(active), d))
    C[:M.shape[0]] = M
    if active:
        C[M.shape[0] + np.arange(len(active)), active] = 1.0
    rhs = np.r_[b, values]
    y = np.linalg.lstsq(C, rhs, rcond=None)[0]
    exact = bool(config.get('exact_hessian', True))
    iterations = int(config.get('newton_max_steps', 8 if exact else 18))
    for _ in range(max(0, iterations)):
        gradient = Q @ y + q + problem['alpha'] * np.cos(y)
        H = (Q - problem['alpha'] * np.diag(np.sin(y))) if exact else Q
        K = np.block([[H, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
        delta = np.linalg.lstsq(
            K, np.r_[-gradient, rhs - C @ y], rcond=None)[0][:d]
        if not np.isfinite(delta).all():
            break
        y += delta
        if np.linalg.norm(delta, ord=np.inf) <= 2e-13:
            break
    return y


def _active_multipliers(problem, V, y, active):
    if not active:
        return np.empty(0)
    gradient = (problem['Q'] @ y + problem['q'] +
                problem['alpha'] * np.cos(y))
    return np.linalg.lstsq(V[active].T, -(V.T @ gradient), rcond=None)[0]


def _factor_inverse(L):
    if L.shape[0] == 0:
        return np.empty((0, 0))
    return np.linalg.solve(L, np.eye(L.shape[0]))


def _publish(state):
    global _last_state, last_tangent_Z, last_cholesky_L
    _last_state = state
    last_tangent_Z = state['Z']
    last_cholesky_L = state['L']


def solve(problem, config):
    """Return a finite equality- and box-feasible active-face minimizer."""
    Q, q, M, b, lower, upper = (problem[k] for k in
        ('Q', 'q', 'M', 'b', 'lower', 'upper'))
    d = len(q)
    cache_size = int(config.get('cache_size', 128))
    caching = (bool(config.get('active_face_caching', True)) and
               bool(config.get('cache_forward_and_adjoint', True)))
    objects = (Q, q, M, b, lower, upper)
    signature = (
        bool(config.get('nullspace_reduction', True)),
        bool(config.get('exact_hessian', True)),
        bool(config.get('row_equilibration', False)),
        bool(config.get('armijo_line_search', False)),
        float(problem['alpha']), int(config.get('newton_max_steps', -1)),
        int(config.get('max_active_iterations', -1)),
        str(config.get('face_signature', 'sorted_active_constraint_ids')),
        bool(config.get('batch_grouping', True)),
        int(config.get('max_rank_update', 4)),
        str(config.get('update_method', 'qr_update_downdate')),
        float(config.get('refactor_condition_threshold', 1e8)))
    key = tuple(map(id, objects)) + signature
    cached = _solution_cache.get(key) if caching else None
    if _same_objects(cached, objects):
        _publish(cached[1])
        return cached[1]['y'].copy()

    yp, V, H0, geometry_key = _geometry(
        problem, bool(config.get('row_equilibration', False)), cache_size)
    # Find a feasible equality point before objective optimization. This is
    # normally a no-op for the benchmark family, whose particular point is
    # interior, but keeps early-stop ablations finite and feasible.
    feasible = yp.copy()
    for _ in range(400):
        violation = max(float(np.max(lower - feasible)),
                        float(np.max(feasible - upper)))
        if violation <= 2e-12:
            break
        clipped = np.clip(feasible, lower, upper)
        feasible = yp + V @ (V.T @ (clipped - yp))

    active = []
    values = []
    incumbent = feasible.copy()
    face = _face_geometry(problem, yp, V, H0, geometry_key,
                          active, values, config)
    max_faces = int(config.get('max_active_iterations', 2 * d + 16))
    use_nullspace = bool(config.get('nullspace_reduction', True))
    L = face['Lquad']
    for _ in range(max(0, max_faces)):
        if use_nullspace:
            optimum, L = _face_newton(problem, face, config)
        else:
            optimum = _fullspace_newton(problem, active, np.asarray(values), config)

        direction = optimum - incumbent
        step = 1.0
        blockers = []
        active_set = set(active)
        for index in range(d):
            if index in active_set:
                continue
            if direction[index] > 1e-15:
                crossing = (upper[index] - incumbent[index]) / direction[index]
            elif direction[index] < -1e-15:
                crossing = (lower[index] - incumbent[index]) / direction[index]
            else:
                continue
            if crossing < step - 2e-12:
                step, blockers = max(0.0, float(crossing)), [index]
            elif abs(crossing - step) <= 2e-12 and crossing < 1.0 - 2e-12:
                blockers.append(index)
        if step < 1.0 - 2e-12:
            incumbent += step * direction
            for index in blockers:
                if index not in active:
                    active.append(index)
                    values.append(float(upper[index] if direction[index] > 0.0
                                        else lower[index]))
            face = _face_geometry(problem, yp, V, H0, geometry_key,
                                  active, values, config, parent=face)
            continue

        incumbent = optimum
        multipliers = _active_multipliers(problem, V, incumbent, active)
        wrong = []
        for j, (index, value) in enumerate(zip(active, values)):
            if ((value == lower[index] and multipliers[j] > 2e-10) or
                    (value == upper[index] and multipliers[j] < -2e-10)):
                wrong.append((abs(float(multipliers[j])), j))
        if wrong:
            j = max(wrong)[1]
            active.pop(j)
            values.pop(j)
            face = _face_geometry(problem, yp, V, H0, geometry_key,
                                  active, values, config)
            continue
        break

    if not np.isfinite(incumbent).all():
        incumbent = feasible.copy()
    # Rebuild/select the exact terminal face after an early stop and retain
    # its converged reduced Hessian factor for the adjoint.
    face = _face_geometry(problem, yp, V, H0, geometry_key,
                          active, values, config, parent=face)
    Z = face['Z']
    if bool(config.get('exact_hessian', True)) and problem['alpha'] and Z.shape[1]:
        L = _chol(face['Hquad'] - problem['alpha'] *
                  (Z.T @ (np.sin(incumbent)[:, None] * Z)))
    else:
        L = face['Lquad']
    state = dict(y=np.asarray(incumbent), Z=Z, L=L,
                 L_inverse=_factor_inverse(L), active=tuple(active),
                 face_signature=face['signature'], problem=problem)
    _publish(state)
    if caching:
        _put(_solution_cache, key, (objects, state), cache_size)
    return state['y'].copy()


def solve_adjoint(v):
    """Apply the retained-factor VJP in O(d*k+k^2), with no refactorization."""
    if _last_state is None:
        raise RuntimeError('solve must be called before solve_adjoint')
    v = np.asarray(v, dtype=float)
    Z = _last_state['Z']
    if v.shape != (_last_state['y'].size,):
        raise ValueError('adjoint vector has incompatible shape')
    if Z.shape[1] == 0:
        return np.zeros_like(v)
    # L_inverse is the retained triangular solve operator. Applying it and
    # its transpose is two O(k^2) substitutions, not a new factorization.
    inverse = _last_state['L_inverse']
    reduced = inverse @ (Z.T @ v)
    return -(Z @ (inverse.T @ reduced))
