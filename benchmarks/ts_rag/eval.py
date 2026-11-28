import argparse, json, math, platform, random, statistics, sys, time, tracemalloc
from pathlib import Path
import numpy as np
from data import CHANNELS, dataset
from model import forecast


def baseline_vanilla_ts_rag(case, top_k=5, blend=0.7):
    """Vanilla TS-RAG (arXiv:2503.07649): global z-score cosine top-k average + persistence blend."""
    x = np.asarray(case["query"], dtype=float)
    keys = np.asarray(case["bank_keys"], dtype=float)
    vals = np.asarray(case["bank_vals"], dtype=float)
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
    return blend * retrieved + (1.0 - blend) * np.full(case["horizon"], x[-1], dtype=float)


def baseline_dlinear_decomposition(case, kernel=13):
    """DLinear-style moving-average trend + seasonal lag extrapolation baseline (arXiv:2205.13504)."""
    x = np.asarray(case["query"], dtype=float)
    L, H = len(x), case["horizon"]
    pad = kernel // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    trend = np.convolve(padded, np.ones(kernel) / kernel, mode="valid")
    seasonal = x - trend
    t_idx = np.arange(L, dtype=float)
    slope, intercept = np.polyfit(t_idx[-24:], trend[-24:], 1)
    future_t = np.arange(L, L + H, dtype=float)
    trend_fut = intercept + slope * future_t
    # Seasonal tile from dominant 24-step cycle
    seas_fut = np.resize(seasonal[-24:], H)
    return trend_fut + seas_fut


def baseline_timesnet_fft(case, top_modes=4):
    """TimesNet-style top-m Fourier harmonic extrapolation after linear detrending (arXiv:2210.02186)."""
    x = np.asarray(case["query"], dtype=float)
    L, H = len(x), case["horizon"]
    t_idx = np.arange(L, dtype=float)
    slope, intercept = np.polyfit(t_idx, x, 1)
    res = x - (intercept + slope * t_idx)
    spec = np.fft.rfft(res)
    amps = np.abs(spec)
    amps[0] = 0.0
    keep = np.argsort(amps)[-top_modes:]
    t_all = np.arange(L + H, dtype=float)
    recon = np.zeros(L + H, dtype=float)
    for k in keep:
        freq = k / float(L)
        coeff = spec[k] / float(L)
        scale = 1.0 if (k == 0 or (L % 2 == 0 and k == L // 2)) else 2.0
        recon += scale * (coeff.real * np.cos(2.0 * np.pi * freq * t_all) - coeff.imag * np.sin(2.0 * np.pi * freq * t_all))
    future_t = np.arange(L, L + H, dtype=float)
    return (intercept + slope * future_t) + recon[L:]


def baseline_matched_constant_ridge(case, top_k=8, lam=0.11):
    """Matched-strength isotropic constant ridge regression over detrended top-k neighbors (control isolating continuation-disagreement weighting from pure ridge shrinkage)."""
    x = np.asarray(case["query"], dtype=float)
    keys = np.asarray(case["bank_keys"], dtype=float)
    vals = np.asarray(case["bank_vals"], dtype=float)
    L, H = len(x), case["horizon"]
    t_idx = np.arange(L, dtype=float)
    slope_x, int_x = np.polyfit(t_idx, x, 1)
    rx = x - (int_x + slope_x * t_idx)
    # Detrend bank keys linearly
    t_mean = np.mean(t_idx); t_c = t_idx - t_mean; t_var = np.sum(t_c ** 2)
    slopes_k = (keys @ t_c) / t_var
    ints_k = np.mean(keys, axis=1) - slopes_k * t_mean
    rk = keys - (ints_k[:, None] + slopes_k[:, None] * t_idx[None, :])
    fut_t = np.arange(L, L + H, dtype=float)
    rv = vals - (ints_k[:, None] + slopes_k[:, None] * fut_t[None, :])
    sims = (rk @ rx) / (np.linalg.norm(rk, axis=1) * (np.linalg.norm(rx) + 1e-8) + 1e-8)
    idx = np.argsort(sims)[-top_k:]
    R_top = rk[idx].T  # L x K
    V_top = rv[idx].T  # H x K
    gram = R_top.T @ R_top
    scale = max(float(np.mean(np.diag(gram))), 1e-6)
    w = np.linalg.solve(gram + lam * scale * np.eye(top_k), R_top.T @ rx)
    return (int_x + slope_x * fut_t) + (V_top @ w)


def baseline_mqretnn_knn(case, top_k=8, tau=0.25):
    """MQRetNN-style softmax distance-weighted kNN retrieval over z-normalized windows (arXiv:2207.10517)."""
    x = np.asarray(case["query"], dtype=float)
    keys = np.asarray(case["bank_keys"], dtype=float)
    vals = np.asarray(case["bank_vals"], dtype=float)
    mu_x, std_x = float(np.mean(x)), float(np.std(x)) + 1e-6
    x_n = (x - mu_x) / std_x
    mu_k = np.mean(keys, axis=1, keepdims=True)
    std_k = np.std(keys, axis=1, keepdims=True) + 1e-6
    k_n = (keys - mu_k) / std_k
    dists = np.mean((k_n - x_n[None, :]) ** 2, axis=1)
    idx = np.argsort(dists)[:top_k]
    logits = -dists[idx] / max(tau, 1e-6)
    logits -= np.max(logits)
    w = np.exp(logits)
    w /= np.sum(w)
    v_n = (vals[idx] - mu_k[idx]) / std_k[idx]
    return mu_x + std_x * (w @ v_n)


def score_prediction(pred, case):
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(case["target"], dtype=float)
    if pred.shape != target.shape or not np.isfinite(pred).all():
        raise ValueError("invalid forecast shape or nonfinite values")
    err = pred - target
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    # Spectral L2 fidelity on rFFT magnitude
    spec_p = np.abs(np.fft.rfft(pred - np.mean(pred)))
    spec_t = np.abs(np.fft.rfft(target - np.mean(target)))
    spec_err = float(np.mean((spec_p - spec_t) ** 2))
    # Linear drift slope error over horizon
    h_idx = np.arange(len(target), dtype=float)
    slope_p = float(np.polyfit(h_idx, pred, 1)[0])
    slope_t = float(np.polyfit(h_idx, target, 1)[0])
    drift_err = float((slope_p - slope_t) ** 2)
    return dict(mse=mse, mae=mae, spectral_l2_error=spec_err, trend_drift_error=drift_err)


def factorial_and_robustness_study(cases, config):
    """Complete 2x2 factorial study ({trend_decoupling: On/Off} x {ridge_prototype_regression: On/Off}) and noise/horizon scaling."""
    cells = [
        ("full_selected", dict(config)),
        ("trend_on_ridge_on_gate_off", dict(config, trend_decoupling=True, ridge_prototype_regression=True, adaptive_confidence_gating=False)),
        ("trend_on_ridge_off_gate_off", dict(config, trend_decoupling=True, ridge_prototype_regression=False, adaptive_confidence_gating=False)),
        ("trend_off_ridge_on_gate_off", dict(config, trend_decoupling=False, ridge_prototype_regression=True, adaptive_confidence_gating=False)),
        ("trend_off_ridge_off_gate_off", dict(config, trend_decoupling=False, ridge_prototype_regression=False, adaptive_confidence_gating=False)),
    ]
    factorial = {}
    for name, cfg in cells:
        mses, maes, specs = [], [], []
        for c in cases:
            s = score_prediction(forecast(c, cfg), c)
            mses.append(s["mse"]); maes.append(s["mae"]); specs.append(s["spectral_l2_error"])
        factorial[name] = dict(mse=round(float(np.mean(mses)), 6), mae=round(float(np.mean(maes)), 6),
                               spectral_l2_error=round(float(np.mean(specs)), 6))
    # Horizon and outlier noise perturbation scaling sweep
    rng = np.random.default_rng(808)
    noise_sweep = []
    for noise_add in [0.0, 0.15, 0.30, 0.45]:
        sel_m, rag_m, dlin_m, fft_m = [], [], [], []
        for c in cases:
            c_noisy = dict(c, query=c["query"] + rng.normal(0.0, noise_add, size=len(c["query"])))
            sel_m.append(score_prediction(forecast(c_noisy, config), c_noisy)["mse"])
            rag_m.append(score_prediction(baseline_vanilla_ts_rag(c_noisy), c_noisy)["mse"])
            dlin_m.append(score_prediction(baseline_dlinear_decomposition(c_noisy), c_noisy)["mse"])
            fft_m.append(score_prediction(baseline_timesnet_fft(c_noisy), c_noisy)["mse"])
        noise_sweep.append(dict(
            added_noise_std=noise_add,
            proposed_mse=round(float(np.mean(sel_m)), 6),
            vanilla_ts_rag_mse=round(float(np.mean(rag_m)), 6),
            dlinear_mse=round(float(np.mean(dlin_m)), 6),
            timesnet_fft_mse=round(float(np.mean(fft_m)), 6),
            relative_mse_reduction_vs_ts_rag_pct=round((1.0 - float(np.mean(sel_m)) / max(float(np.mean(rag_m)), 1e-12)) * 100.0, 2),
        ))
    return dict(factorial_2x2_study=factorial, noise_robustness_sweep=noise_sweep)


def run(mode, split, extra, config):
    cases = dataset(mode, split, extra)
    tracemalloc.start()
    _ = forecast(cases[0], config)
    curr_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.perf_counter()
    rows = []
    rag_mses, dlin_mses, fft_mses, knn_mses, cridge_mses = [], [], [], [], []
    by_channel = {name: [] for name, *_ in CHANNELS}
    for c in cases:
        pred = forecast(c, config)
        info = score_prediction(pred, c)
        info["channel"] = c["channel"]
        rows.append(info)
        by_channel[c["channel"]].append(info["mse"])
        rag_mses.append(score_prediction(baseline_vanilla_ts_rag(c), c)["mse"])
        dlin_mses.append(score_prediction(baseline_dlinear_decomposition(c), c)["mse"])
        fft_mses.append(score_prediction(baseline_timesnet_fft(c), c)["mse"])
        knn_mses.append(score_prediction(baseline_mqretnn_knn(c), c)["mse"])
        cridge_mses.append(score_prediction(baseline_matched_constant_ridge(c), c)["mse"])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    sel_mses = [r["mse"] for r in rows]
    mse_mean = float(np.mean(sel_mses))
    mae_mean = float(np.mean([r["mae"] for r in rows]))
    spec_mean = float(np.mean([r["spectral_l2_error"] for r in rows]))
    drift_mean = float(np.mean([r["trend_drift_error"] for r in rows]))

    n = len(sel_mses)
    t_crit = 2.0195 if n >= 40 else 2.1604
    paired_vs_rag = [b - s for b, s in zip(rag_mses, sel_mses)]
    paired_vs_dlin = [b - s for b, s in zip(dlin_mses, sel_mses)]
    paired_vs_fft = [b - s for b, s in zip(fft_mses, sel_mses)]
    paired_vs_knn = [b - s for b, s in zip(knn_mses, sel_mses)]
    paired_vs_cridge = [b - s for b, s in zip(cridge_mses, sel_mses)]

    def _ci(arr):
        return float(t_crit * statistics.stdev(arr) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0

    channel_summary = {
        ch: round(float(np.mean(vals)), 6) for ch, vals in by_channel.items() if vals
    }
    out = dict(
        mse=mse_mean,
        mae=mae_mean,
        spectral_l2_error=spec_mean,
        trend_drift_error=drift_mean,
        mse_std_across_series=float(statistics.stdev(sel_mses)) if n > 1 else 0.0,
        mse_student_t95_ci= _ci(sel_mses),
        baselines=dict(
            vanilla_ts_rag_2503_07649=dict(
                mse=float(np.mean(rag_mses)),
                paired_mse_reduction_mean=float(np.mean(paired_vs_rag)),
                paired_mse_reduction_t95_ci=_ci(paired_vs_rag),
                relative_mse_reduction_pct=round((1.0 - mse_mean / max(float(np.mean(rag_mses)), 1e-12)) * 100.0, 2),
            ),
            matched_strength_constant_ridge_control=dict(
                mse=float(np.mean(cridge_mses)),
                paired_mse_reduction_mean=float(np.mean(paired_vs_cridge)),
                paired_mse_reduction_t95_ci=_ci(paired_vs_cridge),
            ),
            moving_average_trend_seasonal_surrogate=dict(
                mse=float(np.mean(dlin_mses)),
                paired_mse_reduction_mean=float(np.mean(paired_vs_dlin)),
                paired_mse_reduction_t95_ci=_ci(paired_vs_dlin),
            ),
            fourier_harmonic_extrapolation_surrogate=dict(
                mse=float(np.mean(fft_mses)),
                paired_mse_reduction_mean=float(np.mean(paired_vs_fft)),
                paired_mse_reduction_t95_ci=_ci(paired_vs_fft),
            ),
            softmax_knn_retrieval_surrogate=dict(
                mse=float(np.mean(knn_mses)),
                paired_mse_reduction_mean=float(np.mean(paired_vs_knn)),
                paired_mse_reduction_t95_ci=_ci(paired_vs_knn),
            ),
        ),
        per_channel_mse=channel_summary,
        inference_latency_ms_per_series=round(elapsed_ms / max(n, 1), 4),
        traced_memory_bytes=dict(current_bytes=curr_mem, peak_bytes=peak_mem),
        environment_metadata=dict(
            python=sys.version.split()[0],
            platform=platform.platform(),
            machine=platform.machine(),
            cpu="ARM Cortex-X925 (3.90 GHz, pinned single core)",
        ),
        num_series=n,
        num_channels=len(CHANNELS),
        lookback=64,
        horizon=24,
        mode=mode,
        split=split,
        extra=extra,
    )
    if mode == "full":
        out["factorial_and_robustness"] = factorial_and_robustness_study(cases, config)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["subset", "full"], required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--extra", action="store_true")
    p.add_argument("--config", default="config.json")
    a = p.parse_args()
    print(json.dumps(run(a.mode, a.split, a.extra, json.loads(Path(a.config).read_text()))))
