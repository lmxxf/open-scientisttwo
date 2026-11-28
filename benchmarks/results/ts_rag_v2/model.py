"""Orthogonal trend/residual forecasting with observed-prefix gate calibration."""
import numpy as np
from collections import OrderedDict

_BANK_CACHE = OrderedDict()
_PREFIX_CACHE = OrderedDict()
_EPS = 1e-12


def _despike(a, cfg):
    a = np.asarray(a, dtype=float)
    if not cfg.get('robust_despiking', True):
        return a.copy()
    radius = max(1, int(cfg.get('despike_radius', 3)))
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(a, [(0, 0)] * (a.ndim - 1) + [(radius, radius)], mode='edge'),
        2 * radius + 1, axis=-1)
    med = np.median(windows, axis=-1)
    mad = 1.4826 * np.median(np.abs(windows - med[..., None]), axis=-1)
    floor = np.median(np.abs(np.diff(a, axis=-1)), axis=-1, keepdims=True) * .5
    limit = float(cfg.get('despike_threshold', 4.5)) * np.maximum(mad, floor + _EPS)
    # Only isolated excursions are clipped; contiguous level changes survive.
    bad = np.abs(a - med) > limit
    isolated = bad.copy()
    isolated[..., 1:] &= ~bad[..., :-1]
    isolated[..., :-1] &= ~bad[..., 1:]
    return np.where(isolated, med + np.clip(a - med, -limit, limit), a)


def _decompose(a, h, cfg):
    n = a.shape[-1]
    if not cfg.get('trend_decoupling', True):
        level = np.mean(a, axis=-1, keepdims=True)
        return a - level, np.repeat(level, h, axis=-1)
    degree = min(max(1, int(cfg.get('trend_degree', 2))), n - 1)
    t = np.linspace(-1., 1., n)
    v = np.polynomial.legendre.legvander(t, degree)
    q, r = np.linalg.qr(v)
    coeff = a @ q
    residual = a - coeff @ q.T
    poly = np.linalg.solve(r, coeff.reshape(-1, degree + 1).T).T
    u = 2. * np.arange(1, h + 1) / max(n - 1, 1)
    # Continue the endpoint tangent, damping additional quadratic curvature.
    future_basis = np.polynomial.legendre.legvander(1. + u, degree)
    if degree == 2:
        damp = float(cfg.get('curvature_damping', .25))
        future_basis[:, 2] = 1. + damp * (3. * u + 1.5 * damp * u ** 2)
    future = (poly @ future_basis.T).reshape(a.shape[:-1] + (h,))
    return residual, future


def _bank(keys, vals, h, cfg):
    signature = (id(keys), id(vals), h, cfg.get('robust_despiking', True),
                 cfg.get('despike_radius', 3), cfg.get('despike_threshold', 4.5),
                 cfg.get('trend_decoupling', True), cfg.get('trend_degree', 2),
                 cfg.get('curvature_damping', .5))
    if signature in _BANK_CACHE:
        _BANK_CACHE.move_to_end(signature)
        return _BANK_CACHE[signature][2:]
    clean = _despike(keys, cfg)
    residual, trend = _decompose(clean, h, cfg)
    scale = np.maximum(np.sqrt(np.mean(residual ** 2, axis=1)), 1e-8)
    norm = residual / scale[:, None]
    future = (np.asarray(vals)[:, :h] - trend) / scale[:, None]
    mag = np.abs(np.fft.rfft(norm, axis=1))
    mag[:, 0] = 0.
    mag /= np.maximum(np.linalg.norm(mag, axis=1, keepdims=True), _EPS)
    result = (norm, future, mag)
    # Strong references prevent id reuse; bound retained memory across datasets.
    _BANK_CACHE[signature] = (keys, vals) + result
    while len(_BANK_CACHE) > 24:
        _BANK_CACHE.popitem(last=False)
    return result


def _components(x, keys, vals, h, cfg, exclude=None):
    clean = _despike(x, cfg)
    rx, trend = _decompose(clean, h, cfg)
    n = len(rx)
    scale = max(float(np.sqrt(np.mean(rx ** 2))), 1e-8)
    query = rx / scale
    norms, futures, mags = _bank(keys, vals, h, cfg)
    fft = np.fft.rfft(query)
    magnitude = np.abs(fft)
    magnitude[0] = 0.
    magnitude /= max(float(np.linalg.norm(magnitude)), _EPS)
    spectral_weight = float(cfg.get('spectral_similarity_weight', .25))
    similarity = (1. - spectral_weight) * (norms @ query / n) + spectral_weight * (mags @ magnitude)
    if exclude is not None:
        similarity = np.where(exclude, -np.inf, similarity)
    available = np.flatnonzero(np.isfinite(similarity))
    k = min(max(1, int(cfg.get('top_k', 12))), len(available))
    if not k:
        return trend.copy(), trend.copy(), 0.
    idx = available[np.argsort(similarity[available])[-k:]]
    design = norms[idx].T
    if cfg.get('ridge_prototype_regression', True):
        penalty = np.eye(k)
        if cfg.get('ridge_penalty', 'prototype_continuation_disagreement') == 'prototype_continuation_disagreement':
            neighbors = min(max(1, int(cfg.get('disagreement_neighbors', 4))), max(k - 1, 1))
            distance = np.mean((design[:, :, None] - design[:, None, :]) ** 2, axis=0)
            np.fill_diagonal(distance, np.inf)
            near = np.argsort(distance, axis=1)[:, :neighbors]
            disagreement = np.mean((futures[idx, None, :] - futures[idx][near]) ** 2, axis=(1, 2))
            if cfg.get('penalty_normalization', 'mean_diagonal') == 'mean_diagonal':
                disagreement /= max(float(np.mean(disagreement)), _EPS)
            penalty += float(cfg.get('continuation_penalty', .1)) * np.diag(disagreement) / max(float(cfg.get('ridge_lambda', .01)), _EPS)
        gram = design.T @ design / n
        regularizer = max(float(cfg.get('ridge_lambda', .01)), 1e-10)
        weights = np.linalg.solve(gram + regularizer * penalty, design.T @ query / n)
    else:
        weights = np.full(k, 1. / k)
    retrieval = trend + scale * (weights @ futures[idx])
    error = float(np.mean((query - design @ weights) ** 2))
    gamma = float(np.exp(-float(cfg.get('gate_alpha', 3.)) * error / max(float(np.mean(query ** 2)), _EPS)))
    amplitudes = np.abs(fft)
    amplitudes[0] = 0.
    keep = np.argsort(amplitudes)[-min(max(1, int(cfg.get('spectral_modes', 4))), len(fft) - 1):]
    time = np.arange(n, n + h)
    factors = np.where((keep == 0) | ((n % 2 == 0) & (keep == n // 2)), 1., 2.)
    spectral = trend + scale * np.sum((factors * fft[keep] / n)[:, None] * np.exp(2j * np.pi * keep[:, None] * time / n), axis=0).real
    return retrieval, spectral, gamma


def _calibrated_gate(x, keys, h, cfg, original):
    blocks = max(1, int(cfg.get('gate_horizon_blocks', 3)))
    block_ids = np.minimum(np.arange(h) * blocks // h, blocks - 1)
    errors = [[] for _ in range(blocks)]
    n = len(x)
    # Bank timestamps are unavailable: only bank observations before each
    # pseudo-origin are keys; continuations come exclusively from bank history.
    origins = sorted(set([n - h, n - 2 * h, n - max(1, h // 2)]))
    for origin in origins:
        if origin < max(24, n // 2):
            continue
        length = min(h, n - origin)
        signature = (id(keys), origin, length)
        if signature not in _PREFIX_CACHE:
            _PREFIX_CACHE[signature] = (keys, np.asarray(keys)[:, :origin].copy(),
                                       np.asarray(keys)[:, origin:origin + length].copy())
            while len(_PREFIX_CACHE) > 16:
                _PREFIX_CACHE.popitem(last=False)
        _PREFIX_CACHE.move_to_end(signature)
        _, prefix_keys, prefix_values = _PREFIX_CACHE[signature]
        # Exclude duplicate/overlapping query histories, independent of metadata.
        overlap = np.zeros(len(keys), dtype=bool)
        for start in range(n - origin + 1):
            candidates = np.flatnonzero(np.abs(prefix_keys[:, 0] - x[start]) < 1e-6)
            if len(candidates):
                overlap[candidates] |= np.mean((prefix_keys[candidates] - x[start:start + origin]) ** 2, axis=1) < 1e-14
        r, s, _ = _components(x[:origin], prefix_keys, prefix_values, length, cfg, overlap)
        observed = _despike(x[origin:origin + length], cfg)
        er, es = r - observed, s - observed
        for b in range(blocks):
            mask = block_ids[:length] == b
            if np.any(mask):
                errors[b].extend(zip(er[mask], es[mask]))
    gate = np.full(h, original)
    shrink = np.clip(float(cfg.get('gate_covariance_shrinkage', 0.)), 0., 1.)
    bounds = cfg.get('gate_weight_constraint', [0., 1.])
    for b, pairs in enumerate(errors):
        if len(pairs) < int(cfg.get('gate_min_samples', 8)):
            continue  # Unsupported blocks explicitly retain reconstruction gate.
        e = np.asarray(pairs)
        second = e.T @ e / len(e)
        cross = (1. - shrink) * second[0, 1]
        weight = (second[1, 1] - cross) / max(second[0, 0] + second[1, 1] - 2. * cross, _EPS)
        gate[block_ids == b] = np.clip(weight, float(bounds[0]), float(bounds[1]))
    return gate


def forecast(case, config=None):
    cfg = config or {}
    x = np.asarray(case['query'], dtype=float)
    keys, vals = case['bank_keys'], case['bank_vals']
    h = int(case['horizon'])
    retrieval, spectral, gamma = _components(x, keys, vals, h, cfg)
    if not cfg.get('spectral_extrapolation', True):
        return np.asarray(retrieval, dtype=float).reshape(h)
    if cfg.get('adaptive_confidence_gating', True):
        if cfg.get('gate_calibration', 'rolling_origin_observed_history') == 'rolling_origin_observed_history':
            gamma = _calibrated_gate(x, keys, h, cfg, gamma)
    else:
        gamma = float(cfg.get('retrieval_blend', .7))
    return np.asarray(gamma * retrieval + (1. - gamma) * spectral, dtype=float).reshape(h)
