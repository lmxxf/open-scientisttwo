"""Deterministic high-dimensional Quality-Diversity (DMSQD / CMA-MAE) benchmark."""
import numpy as np


def evaluate_candidates(X, task):
    """Evaluate batch of candidate parameters X (N x d) -> qualities f in [0, 1] and 2D behavior descriptors B in [-1, 1]^2."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    d = task["dim"]
    if X.shape[1] != d or not np.isfinite(X).all():
        raise ValueError("invalid candidate parameter dimension or nonfinite values")
    # Clip search domain to [-2.5, 2.5]^d
    Xc = np.clip(X, -2.5, 2.5)
    diff = Xc - task["opt_center"][None, :]
    # Anisotropic rotated quadratic + multi-modal Rastrigin ripple
    z = diff @ task["rot_Q"]
    quad_loss = np.sum(task["eig_weights"][None, :] * (z ** 2), axis=1) / float(d)
    ripple = float(task["ripple_amp"]) * np.mean(1.0 - np.cos(2.0 * np.pi * z[:, :6]), axis=1)
    boundary_pen = 0.25 * np.mean(np.maximum(0.0, np.abs(X) - 2.5) ** 2, axis=1)
    # Quality score in (0, 1]
    qualities = np.exp(-0.85 * (quad_loss + ripple + boundary_pen))

    # Nonlinear 2D behavior descriptor projection in [-1, 1]^2
    lin_b = Xc @ task["proj_B"]  # N x 2
    b0 = np.tanh(0.75 * lin_b[:, 0] + 0.20 * np.sin(lin_b[:, 1]))
    b1 = np.tanh(0.75 * lin_b[:, 1] - 0.20 * np.cos(lin_b[:, 0]))
    descriptors = np.stack([b0, b1], axis=1)
    return qualities, descriptors


def archive_cell_indices(descriptors, grid_size):
    clipped = np.clip(descriptors, -0.999999, 0.999999)
    scaled = (clipped + 1.0) * 0.5 * grid_size
    ij = np.floor(scaled).astype(int)
    return np.clip(ij[:, 0], 0, grid_size - 1), np.clip(ij[:, 1], 0, grid_size - 1)


def tasks(mode="full", split="validation", extra=False):
    seed_base = (710 if split == "validation" else 1120) + (7000 if extra else 0)
    rng = np.random.default_rng(seed_base)
    configs = [(24, 10, 0.12, 0), (36, 10, 0.12, 1), (48, 10, 0.12, 2)] if mode == "subset" else [
        (d, 12, 0.15, rep) for d in (24, 36, 48, 64) for rep in range(3)
    ]
    out = []
    for t_idx, (d, grid_sz, rip, rep) in enumerate(configs):
        U, _ = np.linalg.qr(rng.normal(size=(d, d)))
        V_b, _ = np.linalg.qr(rng.normal(size=(d, 2)))
        eigs = np.geomspace(0.4, 4.5 * (1.2 if extra else 1.0), d)
        center = rng.uniform(-0.8, 0.8, size=d)
        out.append({
            "task_id": t_idx,
            "name": f"QD_dim{d}_rep{rep}",
            "dim": d,
            "replicate": rep,
            "grid_size": grid_sz,
            "budget_generations": 20 if mode == "subset" else 28,
            "batch_size": 24,
            "ripple_amp": rip,
            "rot_Q": U,
            "proj_B": V_b * np.sqrt(d / 2.5),
            "eig_weights": eigs,
            "opt_center": center,
            "seed": int(seed_base + t_idx * 37),
        })
    return out
