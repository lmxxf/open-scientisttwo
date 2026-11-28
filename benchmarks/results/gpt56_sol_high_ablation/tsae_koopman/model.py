"""Temporal non-negative sparse coding with learned predictive feature graphs.

Only observed trajectories are used here. The dictionary, graph, sparse-code,
and dynamical updates are separable so each mechanism can be ablated cleanly.
"""
import numpy as np


_EPS = 1e-10


def _normalise_dictionary(D):
    D = np.asarray(D, dtype=float).copy()
    norms = np.linalg.norm(D, axis=0)
    good = norms > _EPS
    D[:, good] /= norms[good]
    if np.any(~good):
        D[:, ~good] = 0.0
        rows = np.flatnonzero(~good) % D.shape[0]
        D[rows, np.flatnonzero(~good)] = 1.0
    return D


def _topk_nonnegative(Z, k, soft=0.0):
    """Non-negative hard Top-K projection with optional support shrinkage."""
    Z = np.maximum(np.asarray(Z, dtype=float), 0.0)
    if soft > 0.0:
        Z = np.maximum(Z - soft, 0.0)
    k = min(max(int(k), 1), Z.shape[1])
    if k == Z.shape[1]:
        return Z
    idx = np.argpartition(Z, -k, axis=1)[:, -k:]
    out = np.zeros_like(Z)
    rows = np.arange(Z.shape[0])[:, None]
    out[rows, idx] = Z[rows, idx]
    return out


def _solve_temporal_tridiagonal(B, weight):
    """Solve (I + weight * Delta.T Delta) Z = B in O(T*m)."""
    if weight <= 0.0 or len(B) < 2:
        return B.copy()
    n = len(B)
    diag = np.full(n, 1.0 + 2.0 * weight, dtype=float)
    diag[0] = diag[-1] = 1.0 + weight
    off = -weight
    c = np.empty(n - 1, dtype=float)
    rhs = B.copy()
    c[0] = off / diag[0]
    rhs[0] /= diag[0]
    for i in range(1, n):
        den = diag[i] - off * c[i - 1]
        if i < n - 1:
            c[i] = off / den
        rhs[i] = (rhs[i] - off * rhs[i - 1]) / den
    out = rhs
    for i in range(n - 2, -1, -1):
        out[i] -= c[i] * out[i + 1]
    return out


def _temporal_prox(Z, K, mu_lap, mu_koop, lap_on, koop_on, sweeps):
    """Laplacian solve and bidirectional block Gauss--Seidel Koopman prox."""
    out = _solve_temporal_tridiagonal(Z, 2.0 * mu_lap if lap_on else 0.0)
    if not koop_on or mu_koop <= 0.0 or len(out) < 2:
        return np.maximum(out, 0.0)
    w = float(mu_koop)
    m = out.shape[1]
    eye = np.eye(m)
    # For row codes, an outgoing residual has Hessian K.T K while an
    # incoming residual has identity Hessian.
    inv_first = np.linalg.pinv(eye + w * (K.T @ K), rcond=1e-9)
    inv_mid = np.linalg.pinv(eye + w * (eye + K.T @ K), rcond=1e-9)
    inv_last = eye / (1.0 + w)
    target = out.copy()
    for _ in range(max(1, int(sweeps))):
        for direction in (range(len(out)), range(len(out) - 1, -1, -1)):
            for t in direction:
                rhs = target[t].copy()
                if t > 0:
                    rhs += w * (out[t - 1] @ K.T)
                if t + 1 < len(out):
                    rhs += w * (out[t + 1] @ K)
                inv = inv_first if t == 0 else inv_last if t == len(out) - 1 else inv_mid
                out[t] = np.maximum(rhs @ inv, 0.0)
    return np.maximum(out, 0.0)


def _innovation_gates(Z, K, cfg):
    """Robust persistence weights; large dynamical innovations open the gate."""
    if len(Z) < 2:
        return np.empty(0, dtype=float)
    if not cfg["innovation_gating"]:
        return np.ones(len(Z) - 1, dtype=float)
    residual = np.linalg.norm(Z[1:] - Z[:-1] @ K.T, axis=1)
    row_norm = np.linalg.norm(Z, axis=1)
    nonzero = row_norm[row_norm > _EPS]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    tau = max(float(cfg["innovation_tau"]), _EPS)
    gate = np.exp(-0.5 * (residual / max(scale, _EPS) / tau) ** 2)
    return np.clip(gate, float(cfg["innovation_gate_min"]), 1.0)


def _standardize_rows(A):
    return ((A - np.mean(A, axis=1, keepdims=True)) /
            np.maximum(np.std(A, axis=1, keepdims=True), 1e-6))


def _support_consensus(proposal, initial, K, affinity, cfg,
                       unary_proposal=None):
    """Alternating forward/backward Top-K minimization of support switching."""
    k = min(max(int(cfg["topk"]), 1), proposal.shape[1])
    support = initial > 0.0
    score_source = proposal if unary_proposal is None else unary_proposal
    unary = _standardize_rows(np.maximum(score_source, 0.0))
    if affinity is not None and float(cfg["support_graph_evidence"]) > 0.0:
        A = np.maximum(affinity, 0.0)
        A /= np.maximum(np.sum(A, axis=1, keepdims=True), _EPS)
        evidence = support.astype(float) @ A
        unary += float(cfg["support_graph_evidence"]) * _standardize_rows(evidence)
    gates = _innovation_gates(initial, K, cfg)
    switch = max(float(cfg["support_switch_weight"]), 0.0)
    for _ in range(max(0, int(cfg["consensus_sweeps"]))):
        directions = (range(len(support)), range(len(support) - 1, -1, -1))
        for direction in directions:
            for t in direction:
                score = unary[t].copy()
                if t > 0:
                    score += switch * gates[t - 1] * support[t - 1]
                if t + 1 < len(support):
                    score += switch * gates[t] * support[t + 1]
                idx = np.argpartition(score, -k)[-k:]
                support[t] = False
                support[t, idx] = True
    return np.maximum(proposal, 0.0) * support, support


def _amplitude_debias(X, D, Z, support, cfg):
    """Refit amplitudes by projected gradient without changing support."""
    iterations = max(0, int(cfg["amplitude_debias_iterations"]))
    if iterations == 0:
        return Z
    lipschitz = float(np.linalg.norm(D, 2) ** 2)
    step = (float(cfg["amplitude_debias_step_scale"])
            * min(float(cfg["fista_step_cap"]), 0.98 / max(lipschitz, _EPS)))
    l1 = max(0.0, float(cfg["amplitude_debias_l1"]))
    out = np.maximum(Z, 0.0) * support
    for _ in range(iterations):
        grad = (out @ D.T - X) @ D
        out = np.maximum(out - step * grad - step * l1, 0.0) * support
    return out


def _infer_codes(X, D, cfg, K=None, graph_affinity=None,
                 graph_laplacian=None, temporal=True, postprocess=True):
    """FISTA, temporal prox, discrete consensus, and amplitude debiasing."""
    X = np.asarray(X, dtype=float)
    m = D.shape[1]
    k = int(cfg["topk"])
    iterations = max(1, int(cfg["fista_iterations"]))
    l1 = max(0.0, float(cfg["l1_penalty"]))
    lipschitz = float(np.linalg.norm(D, 2) ** 2)
    graph_on = bool(cfg["adaptive_feature_graph"] and cfg["graph_code_smoothing"])
    if graph_on and graph_laplacian is not None:
        lipschitz += 2.0 * float(cfg["mu_graph_code"]) * float(np.linalg.norm(graph_laplacian, 2))
    step = min(float(cfg["fista_step_cap"]), 0.98 / max(lipschitz, _EPS))
    Z = np.maximum(X @ D, 0.0)
    if cfg["topk_support_projection"]:
        Z = _topk_nonnegative(Z, k)
    Y = Z.copy()
    fista_t = 1.0
    dense = Z.copy()
    for i in range(iterations):
        grad = (Y @ D.T - X) @ D
        if graph_on and graph_laplacian is not None:
            grad += 2.0 * float(cfg["mu_graph_code"]) * (Y @ graph_laplacian)
        dense = np.maximum(Y - step * grad - step * l1, 0.0)
        new_Z = dense
        if cfg["topk_support_projection"] and (i + 1) % int(cfg["support_projection_interval"]) == 0:
            new_Z = _topk_nonnegative(new_Z, k)
        if cfg["fista_momentum"]:
            new_t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * fista_t * fista_t))
            Y = new_Z + ((fista_t - 1.0) / new_t) * (new_Z - Z)
            if np.sum((Y - new_Z) * (new_Z - Z)) > 0.0:
                Y = new_Z.copy()
                new_t = 1.0
            fista_t = new_t
        else:
            Y = new_Z
        Z = new_Z
    proposal = Z
    active_K = np.eye(m) if K is None else K
    if temporal:
        proposal = _temporal_prox(
            proposal, active_K,
            float(cfg["mu_lap"]), float(cfg["mu_koop"]),
            bool(cfg["temporal_laplacian_smoothing"]),
            bool(cfg["koopman_dynamical_prior"] and K is not None),
            int(cfg["temporal_sweeps"]),
        )
    if cfg["topk_support_projection"]:
        Z = _topk_nonnegative(proposal, k, float(cfg["topk_soft_threshold"]))
    else:
        Z = np.maximum(proposal, 0.0)
    support = Z > 0.0
    if (temporal and postprocess and cfg["support_consensus"]
            and cfg["topk_support_projection"]):
        Z, support = _support_consensus(
            proposal, Z, active_K, graph_affinity, cfg)
    if temporal and postprocess and cfg["amplitude_debias"]:
        Z = _amplitude_debias(X, D, Z, support, cfg)
    return np.nan_to_num(np.maximum(Z, 0.0), nan=0.0, posinf=1e6, neginf=0.0)


def _feature_graph(code_sequences, K, previous, cfg):
    """Sparse affinity from conditional coactivation and predictive residuals."""
    m = K.shape[0]
    coact = np.zeros((m, m), dtype=float)
    pred = np.zeros((m, m), dtype=float)
    for Z in code_sequences:
        scale = np.sqrt(np.sum(Z * Z, axis=0) + _EPS)
        C = (Z.T @ Z) / (scale[:, None] * scale[None, :] + _EPS)
        np.fill_diagonal(C, 0.0)
        coact += np.maximum(C, 0.0)
        if len(Z) > 1:
            R = Z[1:] - Z[:-1] @ K.T
            rscale = np.sqrt(np.sum(R * R, axis=0) + _EPS)
            P = np.abs(R.T @ R) / (rscale[:, None] * rscale[None, :] + _EPS)
            np.fill_diagonal(P, 0.0)
            pred += P
    n = max(len(code_sequences), 1)
    coact /= n
    pred /= n
    use_coact = bool(cfg["coactivation_affinity"])
    use_pred = bool(cfg["predictive_residual_affinity"])
    if use_coact and use_pred:
        alpha = float(cfg["predictive_affinity_blend"])
        A = (1.0 - alpha) * coact + alpha * pred
    elif use_pred:
        A = pred
    elif use_coact:
        A = coact
    else:
        A = np.zeros_like(coact)
    A = np.maximum(0.5 * (A + A.T), 0.0)
    np.fill_diagonal(A, 0.0)
    neighbors = min(max(int(cfg["top_neighbors_per_feature"]), 1), m - 1)
    sparse = np.zeros_like(A)
    idx = np.argpartition(A, -neighbors, axis=1)[:, -neighbors:]
    rows = np.arange(m)[:, None]
    sparse[rows, idx] = A[rows, idx]
    A = np.maximum(sparse, sparse.T)
    positive = A[A > 0.0]
    floor_weight = 0.05 * (float(np.median(positive)) if positive.size else 1.0)
    floor_degree = max(0, int(cfg["graph_degree_floor"]))
    for offset in range(1, floor_degree // 2 + 1):
        j = (np.arange(m) + offset) % m
        A[np.arange(m), j] = np.maximum(A[np.arange(m), j], floor_weight)
        A[j, np.arange(m)] = np.maximum(A[j, np.arange(m)], floor_weight)
    if previous is not None:
        ema = float(np.clip(cfg["graph_ema"], 0.0, 1.0))
        A = ema * previous + (1.0 - ema) * A
    degree = np.sum(A, axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, _EPS))
    L = np.eye(m) - inv_sqrt[:, None] * A * inv_sqrt[None, :]
    if cfg["graph_trace_normalization"]:
        L *= m / max(float(np.trace(L)), _EPS)
    return A, 0.5 * (L + L.T)


def _estimate_koopman(code_sequences, rho, previous_K, cfg, use_weights=None):
    """Exact ridge cross-covariance, optionally innovation weighted."""
    m = code_sequences[0].shape[1]
    gram = np.zeros((m, m), dtype=float)
    cross = np.zeros((m, m), dtype=float)
    for Z in code_sequences:
        if len(Z) > 1:
            X0, X1 = Z[:-1], Z[1:]
            weighted = cfg["weighted_koopman_fit"] if use_weights is None else use_weights
            weights = (_innovation_gates(Z, previous_K, cfg)
                       if weighted else np.ones(len(X0)))
            gram += X0.T @ (weights[:, None] * X0)
            cross += X0.T @ (weights[:, None] * X1)
    rho = max(float(rho), _EPS)
    eye = np.eye(m)
    # W = K.T in X1 = X0 W.
    W = np.linalg.solve(gram + rho * eye, cross + rho * eye)
    return np.nan_to_num(W.T, nan=0.0, posinf=0.0, neginf=0.0)


def _dictionary_step(D, sequences, codes, graph_laplacian, cfg):
    """One tangent/preconditioned proximal dictionary update."""
    grad = np.zeros_like(D)
    curvature = np.full(D.shape[1], float(cfg["dictionary_ridge"]), dtype=float)
    for X, Z in zip(sequences, codes):
        grad += (Z @ D.T - X).T @ Z
        curvature += np.sum(Z * Z, axis=0)
    grad /= curvature[None, :]
    if (cfg["adaptive_feature_graph"] and cfg["graph_dictionary_smoothing"]
            and graph_laplacian is not None):
        grad += 2.0 * float(cfg["mu_graph_dictionary"]) * (D @ graph_laplacian)
    # Tangent projection is the available orthogonalisation for m > d.
    grad -= D * np.sum(D * grad, axis=0, keepdims=True)
    return _normalise_dictionary(D - float(cfg["dictionary_learning_rate"]) * grad)


def _config(user, bundle):
    cfg = {
        "dictionary_refinement": True,
        "temporal_laplacian_smoothing": True,
        "koopman_dynamical_prior": True,
        "topk_support_projection": True,
        "fista_momentum": True,
        "adaptive_feature_graph": True,
        "coactivation_affinity": True,
        "predictive_residual_affinity": True,
        "graph_code_smoothing": True,
        "graph_dictionary_smoothing": True,
        "support_consensus": True,
        "innovation_gating": True,
        "weighted_koopman_fit": True,
        "amplitude_debias": True,
        "stop_gradient_through_innovation_gate": True,
        "support_postprocess_during_dictionary": False,
        "use_unlabelled_eval_for_dictionary": True,
        "dictionary_iterations": 4,
        "dictionary_learning_rate": 0.10,
        "dictionary_ridge": 0.08,
        "fista_iterations": 24,
        "fista_step_cap": 1.0,
        "l1_penalty": 0.07,
        "topk": int(bundle["k_active"]),
        "topk_soft_threshold": 0.0,
        "support_projection_interval": 1,
        "mu_lap": 1.0,
        "mu_koop": 0.20,
        "koopman_ridge": 1.5,
        "temporal_sweeps": 2,
        "mu_graph_code": 0.035,
        "mu_graph_dictionary": 0.002,
        "predictive_affinity_blend": 0.90,
        "top_neighbors_per_feature": 16,
        "graph_ema": 0.95,
        "graph_degree_floor": 2,
        "graph_trace_normalization": True,
        "graph_update_interval": 500,
        "consensus_sweeps": 3,
        "support_switch_weight": 0.06,
        "support_graph_evidence": 0.01,
        "innovation_tau": 1.5,
        "innovation_gate_min": 0.05,
        "amplitude_debias_iterations": 6,
        "amplitude_debias_l1": 0.0,
        "amplitude_debias_step_scale": 0.01,
    }
    cfg.update(user or {})
    cfg["support_projection_interval"] = max(1, int(cfg["support_projection_interval"]))
    return cfg


def run_tsae(bundle, config=None):
    """Fit using D_init and observed train/evaluation trajectories only."""
    cfg = _config(config or {}, bundle)
    D = _normalise_dictionary(bundle["D_init"])
    train = [np.asarray(X, dtype=float) for X in bundle["train_sequences"]]
    observed_eval = [np.asarray(case["X_obs"], dtype=float) for case in bundle["eval_cases"]]
    fit_sequences = (train + observed_eval
                     if cfg["use_unlabelled_eval_for_dictionary"] else train)
    m = int(bundle["m_dict"])
    K = np.eye(m, dtype=float)
    graph_affinity = None
    graph_laplacian = None
    rounds = (max(0, int(cfg["dictionary_iterations"]))
              if cfg["dictionary_refinement"] else 1)
    for round_idx in range(rounds):
        codes = [_infer_codes(X, D, cfg, K, graph_affinity,
                              graph_laplacian, temporal=False)
                 for X in fit_sequences]
        K = (_estimate_koopman(codes, cfg["koopman_ridge"], K, cfg,
                               use_weights=False)
             if cfg["koopman_dynamical_prior"] else np.eye(m, dtype=float))
        if cfg["adaptive_feature_graph"] and round_idx % max(1, int(cfg["graph_update_interval"])) == 0:
            graph_affinity, graph_laplacian = _feature_graph(codes, K, graph_affinity, cfg)
        if cfg["dictionary_refinement"]:
            codes = [_infer_codes(X, D, cfg, K, graph_affinity,
                                  graph_laplacian, temporal=True,
                                  postprocess=bool(cfg["support_postprocess_during_dictionary"]))
                     for X in fit_sequences]
            D = _dictionary_step(D, fit_sequences, codes, graph_laplacian, cfg)
    train_codes = [_infer_codes(X, D, cfg, K, graph_affinity,
                                graph_laplacian, temporal=True)
                   for X in fit_sequences]
    K = (_estimate_koopman(train_codes, cfg["koopman_ridge"], K, cfg)
         if cfg["koopman_dynamical_prior"] else np.eye(m, dtype=float))
    reconstructions, latents = [], []
    for X in observed_eval:
        Z = _infer_codes(X, D, cfg, K, graph_affinity,
                         graph_laplacian, temporal=True)
        latents.append(Z)
        reconstructions.append(Z @ D.T)
    return {
        "D_est": D,
        "K_est": K,
        "reconstructions": reconstructions,
        "latents": latents,
    }

