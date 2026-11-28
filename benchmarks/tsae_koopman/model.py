"""Baseline token-independent framewise ReLU-ISTA Sparse Autoencoder (ignoring temporal/Koopman structure)."""
import numpy as np


def run_tsae(bundle, config=None):
    cfg = config or {}
    D = np.asarray(bundle["D_init"], dtype=float).copy()
    D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    m_dict = bundle["m_dict"]
    lam = float(cfg.get("l1_penalty", 0.22))
    step = float(cfg.get("ista_step", 0.18))
    iters = max(1, int(cfg.get("ista_iterations", 15)))

    # Standard framewise ISTA without temporal graph-Laplacian smoothing or Koopman operator learning
    K_est = np.eye(m_dict, dtype=float)
    reconstructions = []
    latents = []
    for case in bundle["eval_cases"]:
        X = np.asarray(case["X_obs"], dtype=float)
        Z = np.maximum(0.0, X @ D)
        for _ in range(iters):
            grad = (Z @ D.T - X) @ D
            Z = np.maximum(0.0, Z - step * grad - step * lam)
        reconstructions.append(Z @ D.T)
        latents.append(Z)

    return {
        "D_est": D,
        "K_est": K_est,
        "reconstructions": reconstructions,
        "latents": latents,
    }
