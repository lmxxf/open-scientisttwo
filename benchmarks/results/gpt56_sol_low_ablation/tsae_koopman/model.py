"""Temporal sparse coding with learned dictionary and Koopman dynamics."""
import numpy as np


def _normalize(D):
    return D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)


def _topk(Z, k):
    Z = np.maximum(Z, 0.0)
    if k >= Z.shape[1]:
        return Z
    jj = np.argpartition(Z, -k, axis=1)[:, -k:]
    out = np.zeros_like(Z)
    out[np.arange(len(Z))[:, None], jj] = Z[np.arange(len(Z))[:, None], jj]
    return out


def _laplacian_prox(Z, strength):
    """Solve (I + strength L)Y=Z for the path-graph Laplacian."""
    if strength <= 0.0 or len(Z) < 2:
        return Z
    n = len(Z)
    diag = np.full(n, 1.0 + 2.0 * strength)
    diag[[0, -1]] = 1.0 + strength
    off = np.full(n - 1, -strength)
    rhs = Z.copy()
    for i in range(1, n):
        q = off[i - 1] / diag[i - 1]
        diag[i] -= q * off[i - 1]
        rhs[i] -= q * rhs[i - 1]
    out = np.empty_like(rhs)
    out[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        out[i] = (rhs[i] - off[i] * out[i + 1]) / diag[i]
    return out


def _infer(X, D, cfg, K=None, iterations=None):
    topk_on = bool(cfg.get("topk_support_projection", True))
    momentum_on = bool(cfg.get("fista_momentum", True))
    k = min(D.shape[1], max(1, int(cfg.get("support_size", cfg["k_active"] + 1))))
    lam = float(cfg.get("l1_penalty", 0.055))
    n_iter = max(1, int(iterations if iterations is not None else cfg.get("fista_iterations", 24)))
    lipschitz = float(np.linalg.norm(D, 2) ** 2) + 1e-8
    step = min(float(cfg.get("fista_step", 0.24)), 0.98 / lipschitz)
    Z = np.maximum(0.0, X @ D)
    if topk_on:
        Z = _topk(Z, k)
    Y, tk = Z.copy(), 1.0
    for _ in range(n_iter):
        nxt = np.maximum(0.0, Y - step * ((Y @ D.T - X) @ D + lam))
        if topk_on:
            nxt = _topk(nxt, k)
        if momentum_on:
            nt = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
            Y = nxt + ((tk - 1.0) / nt) * (nxt - Z)
            tk = nt
        else:
            Y = nxt
        Z = nxt
    if bool(cfg.get("temporal_laplacian_smoothing", True)):
        Z = _laplacian_prox(Z, float(cfg.get("mu_lap", 0.10)))
    if K is not None and bool(cfg.get("koopman_dynamical_prior", True)) and len(Z) > 1:
        mu = float(cfg.get("mu_koop", 0.07))
        A = np.eye(K.shape[0]) + mu * (np.eye(K.shape[0]) + K.T @ K)
        for _ in range(max(1, int(cfg.get("temporal_sweeps", 2)))):
            old = Z.copy()
            rhs = old.copy()
            rhs[1:] += mu * (old[:-1] @ K.T)
            rhs[:-1] += mu * (old[1:] @ K)
            Z = np.maximum(0.0, np.linalg.solve(A, rhs.T).T)
            if bool(cfg.get("temporal_laplacian_smoothing", True)):
                Z = _laplacian_prox(Z, float(cfg.get("mu_lap", 0.10)) * 0.5)
    if topk_on:
        Z = _topk(Z, k)
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_koopman(latents, rho):
    m = latents[0].shape[1]
    cov = rho * np.eye(m)
    cross = rho * np.eye(m)
    for Z in latents:
        cov += Z[:-1].T @ Z[:-1]
        cross += Z[:-1].T @ Z[1:]
    K = np.linalg.solve(cov, cross).T
    norm = np.linalg.norm(K, 2)
    if norm > 1.05:
        K *= 1.05 / norm
    return K


def run_tsae(bundle, config=None):
    cfg = dict(config or {})
    cfg["k_active"] = int(bundle["k_active"])
    D = _normalize(np.asarray(bundle["D_init"], dtype=float).copy())
    train = [np.asarray(x, dtype=float) for x in bundle["train_sequences"]]
    observed = train + [np.asarray(c["X_obs"], dtype=float) for c in bundle["eval_cases"]]
    rounds = max(0, int(cfg.get("dictionary_rounds", 4))) if bool(cfg.get("dictionary_refinement", True)) else 0
    K = np.eye(D.shape[1])
    for _ in range(rounds):
        zs = [_infer(X, D, cfg, iterations=cfg.get("dictionary_fista_iterations", 12)) for X in observed]
        grad = np.zeros_like(D)
        curvature = np.full(D.shape[1], float(cfg.get("dictionary_ridge", 0.08)))
        for X, Z in zip(observed, zs):
            grad += (X - Z @ D.T).T @ Z
            curvature += np.sum(Z * Z, axis=0)
        D = _normalize(D + float(cfg.get("dictionary_step", 0.32)) * grad / curvature[None, :])
    zs = [_infer(X, D, cfg) for X in observed]
    if bool(cfg.get("koopman_dynamical_prior", True)):
        K = _fit_koopman(zs, float(cfg.get("koopman_ridge", 2.0)))
        zs = [_infer(X, D, cfg, K=K) for X in observed]
        K = _fit_koopman(zs, float(cfg.get("koopman_ridge", 2.0)))
    recons, latents = [], []
    for case in bundle["eval_cases"]:
        Z = _infer(np.asarray(case["X_obs"], dtype=float), D, cfg, K=K)
        latents.append(Z)
        recons.append(Z @ D.T)
    return {"D_est": D, "K_est": K, "reconstructions": recons, "latents": latents}
