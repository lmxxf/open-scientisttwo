"""Baseline isotropic random-emitter MAP-Elites without covariance adaptation or archive-annealing thresholds."""
import numpy as np
from data import archive_cell_indices, evaluate_candidates


def optimize_qd(task, config=None):
    cfg = config or {}
    rng = np.random.default_rng(task["seed"])
    d = task["dim"]
    g = task["grid_size"]
    gens = task["budget_generations"]
    batch = task["batch_size"]
    sigma = float(cfg.get("sigma", 0.35))

    archive_q = np.full((g, g), -np.inf, dtype=float)
    archive_x = np.zeros((g, g, d), dtype=float)
    occupied = []

    for gen in range(gens):
        if not occupied or gen == 0:
            X = rng.uniform(-1.2, 1.2, size=(batch, d))
        else:
            picks = [occupied[int(i)] for i in rng.integers(0, len(occupied), size=batch)]
            parents = np.array([archive_x[r, c] for r, c in picks], dtype=float)
            X = parents + rng.normal(0.0, sigma, size=(batch, d))
        quals, desc = evaluate_candidates(X, task)
        rows, cols = archive_cell_indices(desc, g)
        for idx in range(batch):
            r, c = int(rows[idx]), int(cols[idx])
            if quals[idx] > archive_q[r, c]:
                if not np.isfinite(archive_q[r, c]):
                    occupied.append((r, c))
                archive_q[r, c] = float(quals[idx])
                archive_x[r, c] = X[idx]

    return {
        "archive_q": archive_q,
        "archive_x": archive_x,
    }
