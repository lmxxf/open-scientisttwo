"""Baseline TS-RAG forecaster: raw z-score cosine top-k retrieval averaging + last-value level shift."""
import numpy as np


def forecast(case, config=None):
    cfg = config or {}
    x = np.asarray(case["query"], dtype=float)
    keys = np.asarray(case["bank_keys"], dtype=float)
    vals = np.asarray(case["bank_vals"], dtype=float)
    top_k = max(1, int(cfg.get("top_k", 5)))
    blend = float(cfg.get("retrieval_blend", 0.7))

    # Standard TS-RAG baseline: global window z-score normalization without trend decoupling or spectral filtering
    mu_x = float(np.mean(x))
    std_x = float(np.std(x)) + 1e-6
    x_norm = (x - mu_x) / std_x

    mu_k = np.mean(keys, axis=1, keepdims=True)
    std_k = np.std(keys, axis=1, keepdims=True) + 1e-6
    k_norm = (keys - mu_k) / std_k

    sims = (k_norm @ x_norm) / (np.linalg.norm(k_norm, axis=1) * (np.linalg.norm(x_norm) + 1e-8) + 1e-8)
    idx = np.argsort(sims)[-top_k:]
    v_norm = (vals[idx] - mu_k[idx]) / std_k[idx]
    retrieved = mu_x + std_x * np.mean(v_norm, axis=0)

    # Naive persistence baseline
    naive = np.full(case["horizon"], x[-1], dtype=float)
    return blend * retrieved + (1.0 - blend) * naive
