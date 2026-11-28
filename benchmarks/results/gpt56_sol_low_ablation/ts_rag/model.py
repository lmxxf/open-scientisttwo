"""Robust trend-decoupled spectral/prototype time-series forecaster."""
import numpy as np

_BANK_CACHE = {}
_CACHE_LIMIT = 4
_EPS = 1e-10


def _despike(a, radius=2, threshold=4.5):
    """Clip isolated samples against a local median and robust local scale."""
    z = np.asarray(a, dtype=float)
    was_vector = z.ndim == 1
    rows = z[None, :] if was_vector else z
    out = rows.copy()
    for j in range(rows.shape[1]):
        lo, hi = max(0, j - radius), min(rows.shape[1], j + radius + 1)
        window = rows[:, lo:hi]
        med = np.median(window, axis=1)
        mad = np.median(np.abs(window - med[:, None]), axis=1)
        global_mad = np.median(np.abs(rows - np.median(rows, axis=1)[:, None]), axis=1)
        scale = 1.4826 * np.maximum(mad, 0.08 * global_mad) + _EPS
        delta = rows[:, j] - med
        isolated = np.abs(delta) > threshold * scale
        out[:, j] = np.where(
            isolated, med + np.clip(delta, -threshold * scale, threshold * scale), rows[:, j]
        )
    return out[0] if was_vector else out


def _trend_geometry(length, horizon, degree, curvature_damping):
    t = np.linspace(-1.0, 1.0, length)
    vand = np.polynomial.legendre.legvander(t, degree)
    q, r = np.linalg.qr(vand)
    steps = np.arange(1, horizon + 1, dtype=float)
    tf = 1.0 + 2.0 * steps / max(length - 1, 1)
    future = np.polynomial.legendre.legvander(tf, degree)
    if degree >= 2:
        future[:, 2:] *= np.exp(
            -steps[:, None] / max(curvature_damping * length, _EPS)
        )
    return q, r, future


def _project_trend(a, q, r, future_basis):
    rows = np.asarray(a, dtype=float)
    was_vector = rows.ndim == 1
    rows = rows[None, :] if was_vector else rows
    orth_coeff = rows @ q
    fitted = orth_coeff @ q.T
    poly_coeff = np.linalg.solve(r, orth_coeff.T).T
    future = poly_coeff @ future_basis.T
    return (fitted[0], future[0]) if was_vector else (fitted, future)


def _spectral_future(residual, horizon, modes):
    n = len(residual)
    spectrum = np.fft.rfft(residual)
    strength = np.abs(spectrum)
    strength[0] = 0.0
    count = min(max(0, int(modes)), max(0, len(strength) - 1))
    if count == 0:
        return np.zeros(horizon, dtype=float)
    selected = np.argpartition(strength, -count)[-count:]
    times = np.arange(n, n + horizon, dtype=float)
    out = np.zeros(horizon, dtype=float)
    for k in selected:
        coeff = spectrum[k] / n
        factor = 1.0 if (n % 2 == 0 and k == n // 2) else 2.0
        angle = 2.0 * np.pi * k * times / n
        out += factor * (
            coeff.real * np.cos(angle) - coeff.imag * np.sin(angle)
        )
    return out


def _prepare_bank(source, values, horizon, cfg):
    robust = bool(cfg.get("robust_despiking", True))
    trend_on = bool(cfg.get("trend_decoupling", True))
    degree = min(2, max(1, int(cfg.get("trend_degree", 2))))
    damping = float(cfg.get("curvature_damping", 0.45))
    radius = max(1, int(cfg.get("despike_radius", 2)))
    threshold = max(1.0, float(cfg.get("despike_threshold", 4.5)))
    signature = (
        id(source), source.shape, horizon, robust, trend_on, degree,
        damping, radius, threshold,
    )
    hit = _BANK_CACHE.get(signature)
    if hit is not None and hit[0] is source:
        return hit[1]

    keys = (_despike(source, radius, threshold) if robust
            else np.asarray(source, dtype=float).copy())
    if trend_on:
        q, r, future_basis = _trend_geometry(
            keys.shape[1], horizon, degree, damping
        )
        fitted, trend_future = _project_trend(keys, q, r, future_basis)
        residual = keys - fitted
    else:
        q = r = future_basis = None
        centers = np.mean(keys, axis=1, keepdims=True)
        residual = keys - centers
        trend_future = np.repeat(centers, horizon, axis=1)

    scales = np.maximum(np.sqrt(np.mean(residual ** 2, axis=1)), 1e-7)
    normalized = residual / scales[:, None]
    future_residual = (
        np.asarray(values, dtype=float)[:, :horizon] - trend_future
    ) / scales[:, None]
    magnitudes = np.abs(np.fft.rfft(normalized, axis=1))
    magnitudes[:, 0] = 0.0
    magnitudes /= np.maximum(
        np.linalg.norm(magnitudes, axis=1, keepdims=True), _EPS
    )
    prepared = (normalized, future_residual, magnitudes, q, r, future_basis)
    if len(_BANK_CACHE) >= _CACHE_LIMIT:
        _BANK_CACHE.pop(next(iter(_BANK_CACHE)))
    _BANK_CACHE[signature] = (source, prepared)
    return prepared


def forecast(case, config=None):
    """Return a finite 1-D forecast without consulting held-out fields."""
    cfg = {} if config is None else config
    x_raw = np.asarray(case["query"], dtype=float)
    bank_source = case["bank_keys"]
    keys = np.asarray(bank_source, dtype=float)
    values = np.asarray(case["bank_vals"], dtype=float)
    horizon = int(case["horizon"])
    if (x_raw.ndim != 1 or keys.ndim != 2 or values.ndim != 2
            or len(x_raw) != keys.shape[1]):
        raise ValueError("incompatible query or memory-bank dimensions")

    robust = bool(cfg.get("robust_despiking", True))
    trend_on = bool(cfg.get("trend_decoupling", True))
    spectral_on = bool(cfg.get("spectral_extrapolation", True))
    ridge_on = bool(cfg.get("ridge_prototype_regression", True))
    gate_on = bool(cfg.get("adaptive_confidence_gating", True))
    radius = max(1, int(cfg.get("despike_radius", 2)))
    threshold = max(1.0, float(cfg.get("despike_threshold", 4.5)))
    x = _despike(x_raw, radius, threshold) if robust else x_raw.copy()

    bank_res, bank_future, bank_mag, q, r, future_basis = _prepare_bank(
        bank_source, values, horizon, cfg
    )
    if trend_on:
        fitted, query_trend = _project_trend(x, q, r, future_basis)
        residual = x - fitted
    else:
        center = float(np.mean(x))
        residual = x - center
        query_trend = np.full(horizon, center, dtype=float)

    query_scale = max(float(np.sqrt(np.mean(residual ** 2))), 1e-7)
    query_res = residual / query_scale
    query_mag = np.abs(np.fft.rfft(query_res))
    query_mag[0] = 0.0
    query_mag /= max(float(np.linalg.norm(query_mag)), _EPS)

    # Normalized time similarity is phase-sensitive; magnitude adds period match.
    time_sim = (bank_res @ query_res) / len(query_res)
    spectral_sim = bank_mag @ query_mag
    sw = float(np.clip(cfg.get("spectral_similarity_weight", 0.35), 0.0, 1.0))
    similarity = (1.0 - sw) * time_sim + sw * spectral_sim
    top_k = min(len(keys), max(1, int(cfg.get("top_k", 12))))
    selected = np.argpartition(similarity, -top_k)[-top_k:]
    selected = selected[np.argsort(similarity[selected])[::-1]]

    design = bank_res[selected].T
    if ridge_on:
        lam = max(0.0, float(cfg.get("ridge_lambda", 0.08)))
        gram = (design.T @ design) / len(query_res)
        rhs = (design.T @ query_res) / len(query_res)
        weights = np.linalg.solve(
            gram + (lam + _EPS) * np.eye(top_k), rhs
        )
    else:
        temperature = max(float(cfg.get("retrieval_temperature", 0.12)), 1e-5)
        logits = (similarity[selected] - np.max(similarity[selected])) / temperature
        weights = np.exp(logits)
        weights /= max(float(np.sum(weights)), _EPS)
    retrieved = query_trend + query_scale * (weights @ bank_future[selected])

    if spectral_on:
        spectral = query_trend + query_scale * _spectral_future(
            query_res, horizon, cfg.get("spectral_modes", 3)
        )
    else:
        spectral = query_trend.copy()

    if not spectral_on:
        candidate = retrieved
    elif not ridge_on:
        blend = float(np.clip(cfg.get("nonridge_retrieval_blend", 0.65), 0.0, 1.0))
        candidate = blend * retrieved + (1.0 - blend) * spectral
    else:
        reconstruction = design @ weights
        relative_error = float(
            np.sum((query_res - reconstruction) ** 2)
            / max(np.sum(query_res ** 2), _EPS)
        )
        if gate_on:
            alpha = max(0.0, float(cfg.get("gate_alpha", 0.25)))
            gamma = float(np.exp(-alpha * relative_error))
        else:
            gamma = float(np.clip(
                cfg.get("fixed_retrieval_weight", 0.7), 0.0, 1.0
            ))
        candidate = gamma * retrieved + (1.0 - gamma) * spectral

    candidate = np.asarray(candidate, dtype=float).reshape(-1)
    if candidate.size != horizon:
        candidate = np.resize(candidate, horizon)
    bound = max(1.0, float(np.max(np.abs(x)))) * 1e6
    return np.nan_to_num(
        candidate, nan=float(x[-1]), posinf=bound, neginf=-bound
    )
