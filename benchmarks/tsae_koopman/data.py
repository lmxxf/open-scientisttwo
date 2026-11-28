"""Deterministic temporal sparse autoencoder (T-SAE) & Koopman latent dynamics benchmark."""
import numpy as np


def _generate_sequence(rng, T, d_in, m_dict, D_true, K_true, k_active, noise_std):
    Z_true = np.zeros((T, m_dict), dtype=float)
    # Partition dictionary into 4 semantic topic blocks of size m_dict // 4
    block_size = m_dict // 4
    active_block = int(rng.integers(0, 4))
    support = rng.choice(np.arange(active_block * block_size, (active_block + 1) * block_size), size=k_active, replace=False)
    z = np.zeros(m_dict, dtype=float)
    z[support] = rng.uniform(0.7, 1.6, size=k_active)
    Z_true[0] = z

    for t in range(1, T):
        # Smooth Koopman evolution within semantic block + rare topic transition
        if t % 16 == 0 and rng.random() < 0.5:
            active_block = (active_block + 1) % 4
            support = rng.choice(np.arange(active_block * block_size, (active_block + 1) * block_size), size=k_active, replace=False)
            z_new = np.zeros(m_dict, dtype=float)
            z_new[support] = rng.uniform(0.7, 1.5, size=k_active)
            z = 0.35 * (K_true @ z) + 0.65 * z_new
        else:
            z = K_true @ z + rng.normal(0.0, 0.04, size=m_dict)
        # Enforce exact k_active + 2 sparsity support on true semantic factors
        top_idx = np.argsort(np.abs(z))[-(k_active + 1):]
        z_sparse = np.zeros(m_dict, dtype=float)
        z_sparse[top_idx] = np.maximum(0.0, z[top_idx])
        Z_true[t] = z_sparse

    # High-frequency syntactic nuisance perturbation orthogonal/isotropic in activation space
    X_clean = Z_true @ D_true.T
    X_obs = X_clean + rng.normal(0.0, noise_std, size=X_clean.shape)
    return X_obs, X_clean, Z_true


def dataset(mode="full", split="validation", extra=False):
    seed_base = (610 if split == "validation" else 1020) + (6000 if extra else 0)
    rng = np.random.default_rng(seed_base)
    d_in = 32
    m_dict = 64
    k_active = 5
    T = 36 if mode == "subset" else 48
    n_seqs = 3 if mode == "subset" else 8

    # Construct coherent overcomplete dictionary D_true (d_in x m_dict) with unit columns
    raw_D = rng.normal(0.0, 1.0, size=(d_in, m_dict))
    D_true = raw_D / (np.linalg.norm(raw_D, axis=0, keepdims=True) + 1e-12)

    # Construct stable block-structured Koopman transition operator K_true (m_dict x m_dict)
    K_true = 0.82 * np.eye(m_dict, dtype=float)
    block_size = m_dict // 4
    for b in range(4):
        sl = slice(b * block_size, (b + 1) * block_size)
        W_b = rng.normal(0.0, 0.08, size=(block_size, block_size))
        W_b = 0.5 * (W_b - W_b.T)  # skew-symmetric generator for oscillatory semantic persistence
        K_true[sl, sl] += W_b

    noise_std = 0.14 * (1.25 if extra else 1.0)
    train_seqs = []
    for _ in range(n_seqs):
        Xo, _, _ = _generate_sequence(rng, T, d_in, m_dict, D_true, K_true, k_active, noise_std)
        train_seqs.append(Xo)

    eval_cases = []
    for s_i in range(n_seqs):
        Xo, Xc, Zt = _generate_sequence(rng, T, d_in, m_dict, D_true, K_true, k_active, noise_std)
        eval_cases.append({
            "seq_id": s_i,
            "X_obs": Xo,
            "X_clean": Xc,
            "Z_true": Zt,
        })

    return {
        "d_in": d_in,
        "m_dict": m_dict,
        "k_active": k_active,
        "D_init": (D_true + rng.normal(0.0, 0.18, size=D_true.shape)),
        "D_true": D_true,
        "K_true": K_true,
        "train_sequences": train_seqs,
        "eval_cases": eval_cases,
    }
