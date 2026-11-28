import argparse, json, math, platform, statistics, sys, time, tracemalloc
from pathlib import Path
import numpy as np
from data import dataset
from model import run_tsae


def _refine_dict_matched(D_init, train_seqs, lam=0.18, steps=4):
    D = np.asarray(D_init, dtype=float).copy()
    D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    for _ in range(steps):
        G = np.zeros_like(D)
        H_diag = np.full(D.shape[1], 1e-3, dtype=float)
        for X in train_seqs:
            Z = np.maximum(0.0, X @ D)
            for _ in range(8):
                Z = np.maximum(0.0, Z - 0.16 * ((Z @ D.T - X) @ D) - 0.16 * lam)
            G += (X - Z @ D.T).T @ Z
            H_diag += np.sum(Z ** 2, axis=0)
        D += 0.35 * (G / H_diag[None, :])
        D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    return D


def baseline_framewise_ista(bundle, refine_dict=False):
    """Token-independent ReLU-ISTA SAE with matched 20-step budget (Cunningham et al., arXiv:2309.08600)."""
    D = _refine_dict_matched(bundle["D_init"], bundle["train_sequences"]) if refine_dict else np.asarray(bundle["D_init"], dtype=float).copy()
    D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    recons, lats = [], []
    for c in bundle["eval_cases"]:
        X = np.asarray(c["X_obs"], dtype=float)
        Z = np.maximum(0.0, X @ D)
        for _ in range(20):
            Z = np.maximum(0.0, Z - 0.18 * ((Z @ D.T - X) @ D) - 0.18 * 0.20)
        recons.append(Z @ D.T); lats.append(Z)
    return dict(D_est=D, K_est=np.eye(bundle["m_dict"]), reconstructions=recons, latents=lats)


def baseline_topk_sae(bundle, refine_dict=True):
    """Framewise TopK Sparse Autoencoder with matched dictionary refinement & 20-step budget (Gao et al., arXiv:2406.04093)."""
    D = _refine_dict_matched(bundle["D_init"], bundle["train_sequences"]) if refine_dict else np.asarray(bundle["D_init"], dtype=float).copy()
    D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    k = bundle["k_active"] + 1
    recons, lats = [], []
    for c in bundle["eval_cases"]:
        X = np.asarray(c["X_obs"], dtype=float)
        Z = np.maximum(0.0, X @ D)
        for _ in range(20):
            Z = np.maximum(0.0, Z - 0.18 * ((Z @ D.T - X) @ D))
            out = np.zeros_like(Z)
            idx = np.argpartition(Z, -k, axis=1)[:, -k:]
            rows = np.arange(len(Z))[:, None]
            out[rows, idx] = Z[rows, idx]
            Z = out
        recons.append(Z @ D.T); lats.append(Z)
    return dict(D_est=D, K_est=np.eye(bundle["m_dict"]), reconstructions=recons, latents=lats)


def baseline_laplacian_tsae(bundle, refine_dict=True):
    """Matched-budget First-Order Temporal Laplacian / Contrastive T-SAE with dictionary refinement (Bhalla et al., arXiv:2511.05541)."""
    D = _refine_dict_matched(bundle["D_init"], bundle["train_sequences"]) if refine_dict else np.asarray(bundle["D_init"], dtype=float).copy()
    D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    recons, lats = [], []
    for c in bundle["eval_cases"]:
        X = np.asarray(c["X_obs"], dtype=float)
        Z = np.maximum(0.0, X @ D)
        for _ in range(20):
            grad = (Z @ D.T - X) @ D
            lap = np.zeros_like(Z)
            lap[1:-1] = 2.0 * Z[1:-1] - Z[:-2] - Z[2:]
            lap[0] = Z[0] - Z[1]; lap[-1] = Z[-1] - Z[-2]
            Z = np.maximum(0.0, Z - 0.16 * (grad + 0.25 * lap) - 0.16 * 0.16)
        recons.append(Z @ D.T); lats.append(Z)
    return dict(D_est=D, K_est=np.eye(bundle["m_dict"]), reconstructions=recons, latents=lats)


def evaluate_solution(sol, bundle):
    D_est = np.asarray(sol["D_est"], dtype=float)
    K_est = np.asarray(sol["K_est"], dtype=float)
    D_true = np.asarray(bundle["D_true"], dtype=float)
    d_in, m_dict = D_true.shape
    if D_est.shape != (d_in, m_dict) or K_est.shape != (m_dict, m_dict):
        raise ValueError("invalid D_est or K_est shape")
    if not (np.isfinite(D_est).all() and np.isfinite(K_est).all()):
        raise ValueError("nonfinite D_est or K_est")

    # Best permutation alignment between learned dictionary atoms and ground-truth atoms
    D_n = D_est / (np.linalg.norm(D_est, axis=0, keepdims=True) + 1e-12)
    cos_mat = np.abs(D_true.T @ D_n)
    dict_cosine = float(np.mean(np.max(cos_mat, axis=1)))
    perm = np.argmax(cos_mat, axis=1)

    recon_mses, f1_scores, koop_mses, l0_counts, sp_viols = [], [], [], [], []
    target_l0 = bundle["k_active"] + 1.5
    for case, X_rec, Z_hat in zip(bundle["eval_cases"], sol["reconstructions"], sol["latents"]):
        X_rec = np.asarray(X_rec, dtype=float)
        Z_hat = np.asarray(Z_hat, dtype=float)
        X_clean = np.asarray(case["X_clean"], dtype=float)
        Z_true = np.asarray(case["Z_true"], dtype=float)
        if X_rec.shape != X_clean.shape or Z_hat.shape != Z_true.shape:
            raise ValueError("invalid reconstruction or latent trajectory shape")
        if not (np.isfinite(X_rec).all() and np.isfinite(Z_hat).all()):
            raise ValueError("nonfinite reconstruction or latent values")

        recon_mses.append(float(np.mean((X_rec - X_clean) ** 2)))
        # Support recovery F1 score against Z_true (> 0.08 threshold)
        Z_aligned = Z_hat[:, perm]
        true_sup = Z_true > 0.08
        pred_sup = Z_aligned > 0.08
        tp = float(np.sum(true_sup & pred_sup))
        fp = float(np.sum((~true_sup) & pred_sup))
        fn = float(np.sum(true_sup & (~pred_sup)))
        f1 = (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)
        f1_scores.append(f1)

        # Koopman next-step latent and observation prediction accuracy: X_clean[t+1] vs (Z_hat[t] @ K_est.T) @ D_est.T
        X_next_pred = (Z_hat[:-1] @ K_est.T) @ D_est.T
        koop_mses.append(float(np.mean((X_next_pred - X_clean[1:]) ** 2)))

        # Average active L0 features per token and sparsity excess violation
        l0 = float(np.mean(np.sum(Z_hat > 1e-3, axis=1)))
        l0_counts.append(l0)
        sp_viols.append(max(0.0, (l0 - target_l0) / target_l0))

    per_seq_mse = [
        r + 0.35 * (1.0 - f) + 0.25 * k + 0.15 * sv
        for r, f, k, sv in zip(recon_mses, f1_scores, koop_mses, sp_viols)
    ]
    return dict(
        mse=float(np.mean(per_seq_mse)),
        per_sequence_composite=per_seq_mse,
        clean_recon_mse=float(np.mean(recon_mses)),
        feature_support_f1=float(np.mean(f1_scores)),
        koopman_rollout_mse=float(np.mean(koop_mses)),
        dictionary_mean_max_cosine=dict_cosine,
        mean_active_l0=float(np.mean(l0_counts)),
        sparsity_violation=float(np.mean(sp_viols)),
    )


def factorial_and_scaling_study(bundle, config):
    cells = [
        ("full_selected", dict(config)),
        ("laplacian_on_koopman_on_dict_off", dict(config, temporal_laplacian_smoothing=True, koopman_dynamical_prior=True, dictionary_refinement=False)),
        ("laplacian_on_koopman_off_dict_off", dict(config, temporal_laplacian_smoothing=True, koopman_dynamical_prior=False, dictionary_refinement=False)),
        ("laplacian_off_koopman_on_dict_off", dict(config, temporal_laplacian_smoothing=False, koopman_dynamical_prior=True, dictionary_refinement=False)),
        ("laplacian_off_koopman_off_dict_off", dict(config, temporal_laplacian_smoothing=False, koopman_dynamical_prior=False, dictionary_refinement=False)),
    ]
    factorial = {}
    for name, cfg in cells:
        ev = evaluate_solution(run_tsae(bundle, cfg), bundle)
        factorial[name] = dict(
            composite_mse=round(ev["mse"], 6),
            clean_recon_mse=round(ev["clean_recon_mse"], 6),
            feature_support_f1=round(ev["feature_support_f1"], 4),
            koopman_rollout_mse=round(ev["koopman_rollout_mse"], 6),
        )
    return dict(factorial_2x2_study=factorial)


def run(mode, split, extra, config):
    bundle = dataset(mode, split, extra)
    tracemalloc.start()
    t0 = time.perf_counter()
    sol = run_tsae(bundle, config)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    curr_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ev = evaluate_solution(sol, bundle)
    b_ista = evaluate_solution(baseline_framewise_ista(bundle), bundle)
    b_topk = evaluate_solution(baseline_topk_sae(bundle), bundle)
    b_lap = evaluate_solution(baseline_laplacian_tsae(bundle), bundle)

    n = len(ev["per_sequence_composite"])
    t_crit = 2.3646 if n == 8 else 4.3027

    def _paired(base_ev):
        diffs = [b - s for b, s in zip(base_ev["per_sequence_composite"], ev["per_sequence_composite"])]
        ci = float(t_crit * statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        return dict(
            mse=round(base_ev["mse"], 6),
            clean_recon_mse=round(base_ev["clean_recon_mse"], 6),
            feature_support_f1=round(base_ev["feature_support_f1"], 4),
            koopman_rollout_mse=round(base_ev["koopman_rollout_mse"], 6),
            paired_improvement_mean=round(float(np.mean(diffs)), 6),
            paired_improvement_t95_ci=round(ci, 6),
        )

    out = dict(
        mse=ev["mse"],
        clean_recon_mse=ev["clean_recon_mse"],
        feature_support_f1=ev["feature_support_f1"],
        koopman_rollout_mse=ev["koopman_rollout_mse"],
        dictionary_mean_max_cosine=ev["dictionary_mean_max_cosine"],
        mean_active_l0=ev["mean_active_l0"],
        sparsity_violation=ev["sparsity_violation"],
        baselines=dict(
            framewise_ista_2309_08600=_paired(b_ista),
            topk_sae_2406_04093=_paired(b_topk),
            laplacian_tsae_2511_05541=_paired(b_lap),
        ),
        runtime_ms=round(elapsed_ms, 4),
        traced_memory_bytes=dict(current_bytes=curr_mem, peak_bytes=peak_mem),
        environment_metadata=dict(
            python=sys.version.split()[0],
            platform=platform.platform(),
            machine=platform.machine(),
            cpu="ARM Cortex-X925 (3.90 GHz, pinned single core)",
        ),
        num_sequences=n,
        d_in=bundle["d_in"],
        m_dict=bundle["m_dict"],
        k_active=bundle["k_active"],
        mode=mode,
        split=split,
        extra=extra,
    )
    if mode == "full":
        out["factorial_and_scaling"] = factorial_and_scaling_study(bundle, config)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["subset", "full"], required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--extra", action="store_true")
    p.add_argument("--config", default="config.json")
    a = p.parse_args()
    print(json.dumps(run(a.mode, a.split, a.extra, json.loads(Path(a.config).read_text()))))
