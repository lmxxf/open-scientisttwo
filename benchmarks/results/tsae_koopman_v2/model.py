"""Sparse dictionary learning with frozen-scale robust prediction-validated edges."""
import numpy as np
from functools import lru_cache


def _topk(z, k, enabled):
    z = np.maximum(z, 0.0)
    if enabled and k <= 0:
        return np.zeros_like(z)
    if enabled and k < z.shape[1]:
        idx = np.argpartition(z, -k, axis=1)[:, -k:]
        out = np.zeros_like(z)
        np.put_along_axis(out, idx, np.take_along_axis(z, idx, axis=1), axis=1)
        return out
    return z


def _koopman(zs, m, rho):
    a = rho * np.eye(m)
    b = rho * np.eye(m)
    for z in zs:
        a += z[:-1].T @ z[:-1]
        b += z[1:].T @ z[:-1]
    return np.linalg.solve(a, b.T).T


def _graph(z, d, K, cfg):
    n = len(z)
    L = np.zeros((n, n))
    q = min(max(0, int(cfg.get('candidate_neighbors', 8))), n - 1)
    if q <= 0:
        return L, np.empty(0)
    x = z @ d.T
    dist = np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(x*x, axis=1)[None, :] - 2*x@x.T, 0)
    mode = cfg.get('graph_mode', 'prediction_validated')
    ranking = dist.copy()
    if mode == 'temporal':
        ranking = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    elif mode not in ('prediction_validated', 'similarity_only'):
        raise ValueError('Unknown graph_mode: ' + str(mode))
    np.fill_diagonal(ranking, np.inf)
    j = np.argpartition(ranking, q-1, axis=1)[:, :q]
    i = np.broadcast_to(np.arange(n)[:, None], j.shape)
    dx = dist[i, j]
    sx = max(float(np.median(dx)), 1e-10)
    w = np.exp(-dx / sx)
    if mode == 'prediction_validated':
        pred = z @ K.T
        residual = (np.sum((z[j]-pred[i])**2, axis=2) + np.sum((z[i]-pred[j])**2, axis=2)) / 2
        sr = max(float(np.median(residual)), 1e-10)
        w *= np.exp(-residual / sr)
    W = np.zeros((n, n))
    W[i, j] = w
    W = (W + W.T) / 2
    degree = np.sum(W, axis=1)
    inv = 1 / np.sqrt(np.maximum(degree, 1e-30))
    L = np.diag((degree > 0).astype(float)) - W * inv[:, None] * inv[None, :]
    # Preserve symmetrization and normalized edge weights. Difference penalties
    # use diag(W 1)-W, so constant latent states have zero edge penalty.
    adjacency = W * inv[:, None] * inv[None, :]
    return adjacency, w.ravel()


@lru_cache(maxsize=64)
def _temporal_inverse(n, alpha):
    diag = np.full(n, 1 + 2*alpha)
    diag[[0, -1]] -= alpha
    # Thomas factorization with all identity columns as right-hand sides.
    out = np.eye(n)
    for t in range(1, n):
        factor = -alpha/diag[t-1]
        diag[t] += alpha*factor
        out[t] -= factor*out[t-1]
    out[-1] /= diag[-1]
    for t in range(n-2, -1, -1):
        out[t] = (out[t]+alpha*out[t+1])/diag[t]
    return out


def _tridiagonal(rhs, alpha):
    """Apply cached Thomas inverse; avoids repeated Python time-axis sweeps."""
    if len(rhs) < 2 or alpha == 0:
        return rhs.copy()
    return _temporal_inverse(len(rhs), float(alpha))@rhs


def _prox(u, K, L, a, b, g, sweeps):
    """Matrix-free CG for the joint quadratic proximal, tridiagonal preconditioned.

    The transition adjoint contributes to both endpoints (bidirectional inference).
    Coefficients include the factor two from the squared-norm penalties.
    """
    if a == b == g == 0 or len(u) < 2:
        return u.copy()
    def operator(z):
        out = z.copy()
        delta = z[1:] - z[:-1]
        out[:-1] -= a*delta
        out[1:] += a*delta
        if b:
            r = z[1:] - z[:-1] @ K.T
            out[1:] += b*r
            out[:-1] -= b*r@K
        if g:
            out += g*(L@z)
        return out
    z = _tridiagonal(u, a)
    r = u - operator(z)
    p = _tridiagonal(r, a)
    rz = np.sum(r*p)
    for _ in range(max(0, sweeps)):
        if rz < 1e-20:
            break
        ap = operator(p)
        step = rz / max(float(np.sum(p*ap)), 1e-30)
        z += step*p
        r -= step*ap
        pre = _tridiagonal(r, a)
        nxt = np.sum(r*pre)
        p = pre + (nxt / max(rz, 1e-30))*p
        rz = nxt
    return z


def _robust_graph_prox(u, adjacency, strength, delta, cfg, diagnostics):
    """Nonnegative IRLS with original-objective monitoring and finite early exits.

    Objective: .5||z-u||² + strength sum_(i<j) w_ij phi_delta(z_i-z_j).
    The quadratic ablation uses the identical edges and projected solver.
    """
    i, j = np.nonzero(np.triu(adjacency, 1))
    if not len(i) or strength <= 0:
        return np.maximum(u, 0)
    w = adjacency[i, j]
    robust = cfg.get('robust_graph_penalty', False) and delta > 0
    def objective(z):
        sq = np.sum((z[i]-z[j])**2, axis=1)
        # Rationalized pseudo-Huber expression avoids subtractive cancellation.
        penalty = sq/(np.sqrt(1+sq/delta**2)+1) if robust else 0.5*sq
        return float(0.5*np.sum((z-u)**2)+strength*np.dot(w, penalty))
    z = np.maximum(u, 0)
    old = initial = objective(z)
    converged = False
    tolerance = max(float(cfg.get('graph_tolerance', 1e-5)), 0.0)
    for _ in range(max(0, int(cfg.get('graph_irls_iterations', 4)))):
        sq = np.sum((z[i]-z[j])**2, axis=1)
        attenuation = 1/np.sqrt(1+sq/delta**2) if robust else np.ones_like(w)
        A = np.zeros_like(adjacency)
        A[i, j] = w*attenuation
        A[j, i] = w*attenuation
        degree = A.sum(axis=1)
        step = 1/(1+2*strength*float(degree.max()))
        candidate = z.copy()
        for _ in range(max(0, int(cfg.get('graph_quadratic_iterations', 12)))):
            grad = candidate-u+strength*(degree[:, None]*candidate-A@candidate)
            nxt = np.maximum(candidate-step*grad, 0)
            change = np.linalg.norm(nxt-candidate)
            candidate = nxt
            if change <= tolerance*max(1.0, np.linalg.norm(candidate)):
                break
        value = objective(candidate)
        if not np.isfinite(value):
            diagnostics['nonfinite_proposals'] += 1
            break
        for _ in range(20):
            if value <= old+1e-12*max(1.0, abs(old)):
                break
            candidate = (candidate+z)/2
            value = objective(candidate)
        if value > old+1e-12*max(1.0, abs(old)):
            diagnostics['objective_failures'] += 1
            break
        z, old = candidate, value
        sq = np.sum((z[i]-z[j])**2, axis=1)
        attenuation = 1/np.sqrt(1+sq/delta**2) if robust else np.ones_like(w)
        A.fill(0)
        A[i, j] = w*attenuation
        A[j, i] = w*attenuation
        grad = z-u+strength*(A.sum(axis=1)[:, None]*z-A@z)
        residual = np.linalg.norm(z-np.maximum(z-grad, 0))
        if residual <= tolerance*max(1.0, np.linalg.norm(z)):
            converged = True
            break
    diagnostics['calls'] += 1
    diagnostics['not_converged'] += int(not converged)
    diagnostics['objective_failures'] += int(old > initial+1e-10)
    diagnostics['objective_initial_sum'] += initial
    diagnostics['objective_final_sum'] += old
    sq = np.sum((z[i]-z[j])**2, axis=1)
    attenuation = 1/np.sqrt(1+sq/delta**2) if robust else np.ones_like(w)
    diagnostics['attenuation_sum'] += float(np.sum(attenuation))
    diagnostics['edge_count'] += len(w)
    return z


def _infer(x, D, K, k, cfg, initial=None, graph=None, delta=0.0, diagnostics=None):
    gram = D.T@D
    step = 1 / max(float(np.linalg.norm(gram, 2)), 1e-12)
    cross = x@D
    z = np.zeros_like(cross) if initial is None else initial.copy()
    y = z.copy()
    momentum = 1.0
    lap = float(cfg.get('mu_lap', 0.12)) if cfg.get('temporal_laplacian_smoothing', True) else 0.0
    koop = float(cfg.get('mu_koop', 0.10)) if cfg.get('koopman_dynamical_prior', True) else 0.0
    strength = float(cfg.get('lambda_graph', 0.06)) if cfg.get('nonlocal_graph', True) else 0.0
    if graph is None:
        graph = np.zeros((len(x), len(x)))
        strength = 0.0
    lam = float(cfg.get('l1_penalty', 0.07))
    for _ in range(max(0, int(cfg.get('ista_iterations', 45)))):
        u = y - step*(y@gram - cross)
        u = _prox(u, K, graph, 2*step*lap, 2*step*koop, 0.0,
                  int(cfg.get('proximal_sweeps', 6)))
        nxt = (_robust_graph_prox(u-step*lam, graph, step*strength, delta,
                                  cfg, diagnostics) if strength
               else np.maximum(u-step*lam, 0))
        # Convex warm-up allows correlated atoms to exchange support before projection.
        t = (1+np.sqrt(1+4*momentum**2))/2
        y = nxt + ((momentum-1)/t)*(nxt-z) if cfg.get('fista_momentum', True) else nxt.copy()
        z, momentum = nxt, t
    z = _topk(z, k, cfg.get('topk_support_projection', True))
    # Restricted projected iterations refine amplitudes without adding support.
    support = z > 0
    refit_steps = int(cfg.get('support_refit_iterations', 8)) if cfg.get('support_refit', True) else 0
    for _ in range(max(0, refit_steps)):
        u = z - step*(z@gram-cross)
        u = _prox(u, K, graph, 2*step*lap, 2*step*koop, 0.0,
                  int(cfg.get('proximal_sweeps', 6)))
        z = (_robust_graph_prox(u-step*lam, graph, step*strength, delta,
                                cfg, diagnostics) if strength
             else np.maximum(u-step*lam, 0))
        if cfg.get('topk_support_projection', True):
            z *= support
    return z


def run_tsae(bundle, config=None):
    cfg = {} if config is None else dict(config)
    D = np.asarray(bundle['D_init'], dtype=float).copy()
    D /= np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)
    m = D.shape[1]
    k = min(m, max(0, int(bundle['k_active'])))
    train = [np.asarray(x, dtype=float) for x in bundle['train_sequences']]
    observed = [np.asarray(case['X_obs'], dtype=float) for case in bundle['eval_cases']]
    sequences = train + observed
    K = np.eye(m)
    zs = [_infer(x, D, K, k, cfg) for x in sequences]
    # Freeze scale before dictionary updates, using training-only baseline codes.
    positive_distances = []
    for z in zs[:len(train)]:
        adjacency, _ = _graph(z, D, K, cfg)
        i, j = np.nonzero(np.triu(adjacency, 1))
        distances = np.linalg.norm(z[i]-z[j], axis=1)
        positive_distances.extend(distances[distances > 0].tolist())
    delta = float(np.median(positive_distances)) if positive_distances else 0.0
    diagnostics = dict(calls=0, not_converged=0, objective_failures=0,
                       nonfinite_proposals=0, objective_initial_sum=0.0,
                       objective_final_sum=0.0, attenuation_sum=0.0, edge_count=0)
    rounds = max(0, int(cfg.get('alternating_iterations', 6)))
    graphs = [None]*len(sequences)
    edge_statistics = []
    for iteration in range(rounds):
        if cfg.get('dictionary_refinement', True) and len(zs):
            Z = np.concatenate(zs, axis=0)
            X = np.concatenate(sequences, axis=0)
            residual = X - Z@D.T
            grad = residual.T@Z
            # Remove radial components: each update lies in the column sphere's tangent plane.
            grad -= D*np.sum(D*grad, axis=0, keepdims=True)
            curvature = max(float(np.linalg.norm(Z.T@Z, 2)), 1e-12)
            D += float(cfg.get('dictionary_step', 0.8))*grad/curvature
            D /= np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)
        if cfg.get('koopman_dynamical_prior', True):
            K = _koopman(zs, m, max(float(cfg.get('koopman_ridge', 0.5)), 1e-10))
        if cfg.get('nonlocal_graph', True) and iteration % max(1, int(cfg.get('graph_refresh_interval', 2))) == 0:
            graphs = []
            edge_statistics = []
            for z in zs:
                L, weights = _graph(z.copy(), D, K.copy(), cfg)
                graphs.append(L)
                if weights.size:
                    edge_statistics.append(dict(count=int(weights.size), minimum=float(weights.min()),
                                                median=float(np.median(weights)), maximum=float(weights.max())))
        zs = [_infer(x, D, K, k, cfg, z, graph, delta, diagnostics) for x, z, graph in zip(sequences, zs, graphs)]
    if cfg.get('koopman_dynamical_prior', True):
        K = _koopman(zs, m, max(float(cfg.get('koopman_ridge', 0.5)), 1e-10))
    result = zs[len(train):]
    diagnostics['graph_laplacian'] = 'degree-normalized parent edge weights; combinatorial difference Laplacian'
    diagnostics['robust_graph_penalty'] = bool(cfg.get('robust_graph_penalty', False))
    diagnostics['delta'] = delta
    diagnostics['delta_degenerate'] = not bool(positive_distances)
    diagnostics['degeneracy_policy'] = ('no positive training-edge distances; quadratic limit used'
                                         if not positive_distances else 'none')
    diagnostics['mean_edge_attenuation'] = (diagnostics['attenuation_sum']/diagnostics['edge_count']
                                           if diagnostics['edge_count'] else None)
    diagnostics['support_turnover'] = [float(np.mean((z[1:]>0)!=(z[:-1]>0)))
                                        if len(z)>1 else 0.0 for z in result]
    return dict(D_est=D, K_est=K, reconstructions=[z@D.T for z in result], latents=result,
                graph_edge_statistics=edge_statistics, robust_graph_diagnostics=diagnostics)
