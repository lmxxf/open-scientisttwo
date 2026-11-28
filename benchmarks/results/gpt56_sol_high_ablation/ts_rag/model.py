"""Robust adaptive-trend spectral retrieval forecaster."""

import numpy as np


_BANK_CACHE = {}
_BASIS_CACHE = {}


def _boolean(config, name, default=True):
    value = config.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "off", "no"}
    return bool(value)


def _candidate_names(config):
    allowed = {"linear", "quadratic", "piecewise_linear_1knot"}
    requested = config.get(
        "trend_candidates", ["linear", "quadratic", "piecewise_linear_1knot"]
    )
    if isinstance(requested, str):
        requested = [requested]
    names = []
    for name in requested:
        normalized = str(name).strip().lower()
        if normalized in allowed and normalized not in names:
            names.append(normalized)
    return names or ["linear", "quadratic", "piecewise_linear_1knot"]


def _basis(length, name, knot=0.0):
    """Return a raw candidate basis and its orthogonal QR representation."""
    cache_key = (int(length), str(name), float(knot))
    cached = _BASIS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    z = np.linspace(-1.0, 1.0, int(length), dtype=float)
    if name == "linear":
        raw = np.column_stack((np.ones(length), z))
    elif name == "piecewise_linear_1knot":
        raw = np.column_stack((np.ones(length), z, np.maximum(z - knot, 0.0)))
    else:
        raw = np.column_stack((np.ones(length), z, 0.5 * (3.0 * z * z - 1.0)))
    q, r = np.linalg.qr(raw, mode="reduced")
    cached = (raw, q, r)
    _BASIS_CACHE[cache_key] = cached
    return cached


def _basis_at(z, name, knot=0.0):
    z = np.asarray(z, dtype=float)
    if name == "linear":
        return np.column_stack((np.ones_like(z), z))
    if name == "piecewise_linear_1knot":
        return np.column_stack((np.ones_like(z), z, np.maximum(z - knot, 0.0)))
    return np.column_stack((np.ones_like(z), z, 0.5 * (3.0 * z * z - 1.0)))


def _despike_rows(values, threshold):
    """Winsorize isolated samples against a rolling neighbour median/MAD."""
    array = np.asarray(values, dtype=float)
    rows = array[None, :] if array.ndim == 1 else array
    output = rows.copy()
    if rows.shape[1] < 5:
        return output[0] if array.ndim == 1 else output
    windows = np.lib.stride_tricks.sliding_window_view(rows, 5, axis=1)
    neighbours = windows[..., [0, 1, 3, 4]]
    median = np.median(neighbours, axis=2)
    mad = 1.4826 * np.median(np.abs(neighbours - median[..., None]), axis=2)
    local_step = np.median(np.abs(np.diff(neighbours, axis=2)), axis=2)
    scale = np.maximum(mad, 0.35 * local_step + 1e-8)
    delta = rows[:, 2:-2] - median
    limit = threshold * scale
    clipped = median + np.clip(delta, -limit, limit)
    output[:, 2:-2] = np.where(np.abs(delta) > limit, clipped, rows[:, 2:-2])
    return output[0] if array.ndim == 1 else output


def _solve(matrix, rhs):
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _blocked_prequential_scores(rows, candidates, knot):
    """Select trend geometry using only prefix-to-next-block prediction errors."""
    count, length = rows.shape
    block = max(4, length // 8)
    # The center-knot candidate must have observations on both sides of its
    # knot before it can be compared fairly with the global polynomials.
    first_train = max(5 * block, length // 2 + block)
    folds = []
    train_end = first_train
    while train_end + block <= length:
        folds.append((train_end, train_end + block))
        train_end += block
    if not folds:
        folds = [(max(3, length - block), length)]

    z = np.linspace(-1.0, 1.0, length, dtype=float)
    row_center = np.median(rows, axis=1)
    robust_scale = 1.4826 * np.median(
        np.abs(rows - row_center[:, None]), axis=1
    ) + 1e-8
    scores = np.zeros((count, len(candidates)), dtype=float)
    for candidate_index, name in enumerate(candidates):
        fold_losses = []
        for train_end, valid_end in folds:
            train_basis = _basis_at(z[:train_end], name, knot)
            q, r = np.linalg.qr(train_basis, mode="reduced")
            orthogonal_coefficients = rows[:, :train_end] @ q
            raw_coefficients = _solve(r, orthogonal_coefficients.T).T
            prediction = raw_coefficients @ _basis_at(
                z[train_end:valid_end], name, knot
            ).T
            error = rows[:, train_end:valid_end] - prediction
            scaled = np.abs(error) / robust_scale[:, None]
            huber = np.where(scaled <= 1.5, 0.5 * scaled * scaled, 1.5 * scaled - 1.125)
            fold_losses.append(np.mean(huber, axis=1))
        scores[:, candidate_index] = np.mean(fold_losses, axis=0)
    return scores


def _causal_smooth(rows, window):
    """Leakage-safe trailing mean used only for trend-model selection."""
    window = max(1, int(window))
    if window == 1:
        return rows
    cumulative = np.cumsum(rows, axis=1)
    output = cumulative.copy()
    output[:, window:] -= cumulative[:, :-window]
    denominators = np.minimum(np.arange(1, rows.shape[1] + 1), window)
    return output / denominators[None, :]


def _orthogonal_robust_fit(rows, name, knot, robust, huber_delta, iterations):
    """Fit in an orthogonal candidate basis, optionally by Huber IRLS."""
    _, q, r = _basis(rows.shape[1], name, knot)
    coefficients_q = rows @ q
    if robust:
        for _ in range(max(1, iterations)):
            fitted = coefficients_q @ q.T
            error = rows - fitted
            scale = 1.4826 * np.median(np.abs(error), axis=1) + 1e-8
            standardized = np.abs(error) / scale[:, None]
            weights = np.minimum(1.0, huber_delta / np.maximum(standardized, 1e-12))
            for row_index in range(len(rows)):
                weighted_q = q * weights[row_index, :, None]
                gram = q.T @ weighted_q + 1e-10 * np.eye(q.shape[1])
                rhs = q.T @ (weights[row_index] * rows[row_index])
                coefficients_q[row_index] = _solve(gram, rhs)
    raw_coefficients = _solve(r, coefficients_q.T).T
    fitted = coefficients_q @ q.T
    return fitted, raw_coefficients


def _trend_residual(rows, horizon, enabled, config):
    """Adaptively project histories and continue the selected trend models."""
    array = np.asarray(rows, dtype=float)
    one_row = array.ndim == 1
    matrix = array[None, :] if one_row else array
    count, length = matrix.shape
    if not enabled:
        level = np.mean(matrix, axis=1)
        fitted = np.broadcast_to(level[:, None], matrix.shape).copy()
        future = np.broadcast_to(level[:, None], (count, horizon)).copy()
        selected = np.zeros(count, dtype=int)
        result = (matrix - fitted, future, selected)
        return tuple(item[0] if one_row else item for item in result)

    candidates = _candidate_names(config)
    knot = float(np.clip(config.get("trend_knot", 0.0), -0.75, 0.75))
    adaptive = _boolean(config, "adaptive_trend_basis", True)
    prequential = _boolean(config, "blocked_prequential_selection", True)
    selection_rule = str(config.get("trend_selection", "blocked_prequential_cv")).lower()
    if adaptive and prequential and selection_rule == "blocked_prequential_cv" and len(candidates) > 1:
        selection_rows = matrix
        if _boolean(config, "trend_selection_smoothing", True):
            selection_rows = _causal_smooth(
                matrix, max(1, int(config.get("trend_selection_window", 5)))
            )
        scores = _blocked_prequential_scores(selection_rows, candidates, knot)
        selected = np.argmin(scores, axis=1)
    else:
        fixed = str(config.get("fixed_trend_basis", "quadratic")).lower()
        chosen = candidates.index(fixed) if fixed in candidates else min(1, len(candidates) - 1)
        selected = np.full(count, chosen, dtype=int)

    robust_setting = str(config.get("robust_fit", "huber_irls")).lower()
    robust_fit = _boolean(config, "robust_trend_fit", True) and robust_setting == "huber_irls"
    huber_delta = max(float(config.get("huber_delta", 2.0)), 0.1)
    iterations = max(1, min(10, int(config.get("huber_iterations", 4))))
    curvature_damping = float(np.clip(config.get("curvature_damping", 0.5), 0.0, 1.5))
    slope_damping = float(np.clip(config.get("slope_damping", 0.85), 0.0, 1.5))
    delta_z = 2.0 * np.arange(1, horizon + 1, dtype=float) / max(length - 1, 1)

    fitted = np.empty_like(matrix)
    future = np.empty((count, horizon), dtype=float)
    for candidate_index, name in enumerate(candidates):
        row_indices = np.flatnonzero(selected == candidate_index)
        if not len(row_indices):
            continue
        local_fit, coefficients = _orthogonal_robust_fit(
            matrix[row_indices], name, knot, robust_fit, huber_delta, iterations
        )
        fitted[row_indices] = local_fit
        endpoint = coefficients @ _basis_at(np.array([1.0]), name, knot).reshape(-1)
        if name == "linear":
            derivative = coefficients[:, 1]
            curvature = np.zeros(len(row_indices))
        elif name == "piecewise_linear_1knot":
            derivative = coefficients[:, 1] + coefficients[:, 2]
            curvature = np.zeros(len(row_indices))
        else:
            derivative = coefficients[:, 1] + 3.0 * coefficients[:, 2]
            curvature = 1.5 * coefficients[:, 2]
        future[row_indices] = (
            endpoint[:, None]
            + slope_damping * derivative[:, None] * delta_z[None, :]
            + curvature_damping * curvature[:, None] * delta_z[None, :] ** 2
        )
    result = (matrix - fitted, future, selected)
    return tuple(item[0] if one_row else item for item in result)


def _spectral_features(normalized_residuals):
    spectrum = np.fft.rfft(normalized_residuals, axis=-1)
    magnitude = np.abs(spectrum)
    magnitude[..., 0] = 0.0
    norm = np.linalg.norm(magnitude, axis=-1, keepdims=True)
    return spectrum, magnitude / np.maximum(norm, 1e-12)


def _harmonic_future(residual, horizon, modes):
    """Continue the dominant DFT bins with their observed phase."""
    residual = np.asarray(residual, dtype=float)
    length = len(residual)
    spectrum = np.fft.rfft(residual)
    amplitude = np.abs(spectrum)
    amplitude[0] = 0.0
    count = min(max(0, int(modes)), max(0, len(amplitude) - 1))
    if count == 0 or not np.any(amplitude[1:] > 0.0):
        return np.zeros(horizon, dtype=float)
    bins = np.argpartition(amplitude, -count)[-count:]
    time = np.arange(length, length + horizon, dtype=float)
    output = np.zeros(horizon, dtype=float)
    for frequency_bin in bins:
        coefficient = spectrum[frequency_bin] / float(length)
        factor = 1.0 if length % 2 == 0 and frequency_bin == length // 2 else 2.0
        angle = 2.0 * np.pi * frequency_bin * time / float(length)
        output += factor * (
            coefficient.real * np.cos(angle) - coefficient.imag * np.sin(angle)
        )
    return output


def _bank_entry(
    keys, values, horizon, config, robust_despiking, threshold, trend_enabled,
    source_keys, source_values,
):
    candidates = tuple(_candidate_names(config))
    cache_key = (
        id(source_keys), id(source_values), keys.shape, values.shape, horizon, robust_despiking,
        float(threshold), trend_enabled, _boolean(config, "adaptive_trend_basis", True),
        _boolean(config, "blocked_prequential_selection", True),
        _boolean(config, "robust_trend_fit", True), candidates,
        _boolean(config, "trend_selection_smoothing", True),
        int(config.get("trend_selection_window", 5)),
        str(config.get("trend_selection", "blocked_prequential_cv")),
        str(config.get("robust_fit", "huber_irls")),
        float(config.get("trend_knot", 0.0)), float(config.get("huber_delta", 2.0)),
        int(config.get("huber_iterations", 4)), float(config.get("curvature_damping", 0.5)),
        float(config.get("slope_damping", 0.85)),
    )
    cached = _BANK_CACHE.get(cache_key)
    if (
        cached is not None
        and cached["source"] is source_keys
        and cached["source_values"] is source_values
    ):
        return cached
    clean = _despike_rows(keys, threshold) if robust_despiking else keys.copy()
    residual, trend_future, selected = _trend_residual(
        clean, horizon, trend_enabled, config
    )
    scale = np.maximum(np.sqrt(np.mean(residual * residual, axis=1)), 1e-8)
    normalized = residual / scale[:, None]
    _, spectral_unit = _spectral_features(normalized)
    future_residual = (values[:, :horizon] - trend_future) / scale[:, None]
    cached = {
        "source": source_keys,
        "source_values": source_values,
        "normalized": normalized,
        "future_residual": future_residual,
        "spectral_unit": spectral_unit,
        "trend_selection": selected,
    }
    if len(_BANK_CACHE) >= 8:
        _BANK_CACHE.pop(next(iter(_BANK_CACHE)))
    _BANK_CACHE[cache_key] = cached
    return cached


def _phase_align(query, histories, futures, max_shift, strength):
    """Apply each candidate's best circular phase shift to key and future residual."""
    shifts = np.arange(-max_shift, max_shift + 1, dtype=int)
    best_keys = histories.copy()
    best_futures = futures.copy()
    best_scores = np.full(len(histories), -np.inf)
    query_norm = np.linalg.norm(query) + 1e-12
    for shift in shifts:
        shifted = np.roll(histories, shift, axis=1)
        scores = shifted @ query / (np.linalg.norm(shifted, axis=1) * query_norm + 1e-12)
        improved = scores > best_scores
        if np.any(improved):
            best_scores[improved] = scores[improved]
            best_keys[improved] = shifted[improved]
            joined = np.concatenate((histories[improved], futures[improved]), axis=1)
            best_futures[improved] = np.roll(joined, shift, axis=1)[:, -futures.shape[1]:]
    strength = float(np.clip(strength, 0.0, 1.0))
    aligned_keys = (1.0 - strength) * histories + strength * best_keys
    aligned_futures = (1.0 - strength) * futures + strength * best_futures
    scores = aligned_keys @ query / (
        np.linalg.norm(aligned_keys, axis=1) * query_norm + 1e-12
    )
    return aligned_keys, aligned_futures, scores


def _condition_number(design):
    """Condition number of the prototype Gram matrix (without forming it)."""
    if design.shape[1] <= 1:
        return 1.0
    singular_values = np.linalg.svd(design, compute_uv=False)
    if not len(singular_values) or singular_values[-1] <= 1e-12:
        return np.inf
    return float((singular_values[0] / singular_values[-1]) ** 2)


def _prune_prototypes(design, scores, config):
    """Remove redundant prototypes while retaining the best-scoring dictionary."""
    count = design.shape[1]
    minimum = min(count, max(1, int(config.get("min_effective_neighbors", 3))))
    if count <= minimum or not _boolean(config, "influence_pruning", True):
        return np.arange(count, dtype=int)

    column_norms = np.linalg.norm(design, axis=0)
    unit = design / np.maximum(column_norms[None, :], 1e-12)
    correlation = np.abs(unit.T @ unit)
    np.fill_diagonal(correlation, 0.0)
    threshold = float(np.clip(config.get("collinearity_threshold", 0.9975), 0.8, 1.0))
    retained = []
    for index in np.argsort(scores)[::-1]:
        if len(retained) < minimum or not retained or np.max(correlation[index, retained]) < threshold:
            retained.append(int(index))

    # A hard condition ceiling is meaningful only before ridge regularization.
    # Remove the more redundant member of the worst pair, never crossing the
    # configured effective-neighbour floor.
    maximum_condition = max(float(config.get("max_condition_number", 10000.0)), 1.0)
    while len(retained) > minimum and _condition_number(design[:, retained]) > maximum_condition:
        local_correlation = correlation[np.ix_(retained, retained)]
        first, second = np.unravel_index(np.argmax(local_correlation), local_correlation.shape)
        if first == second:
            break
        left, right = retained[first], retained[second]
        remove = left if scores[left] < scores[right] else right
        retained.remove(remove)
    return np.asarray(sorted(retained), dtype=int)


def _gcv_lambda(design, target, base_lambda, config):
    """Select ridge strength by generalized cross-validation on the lookback."""
    if str(config.get("lambda_selection", "gcv")).strip().lower() != "gcv":
        return base_lambda
    sample_count = max(design.shape[0], 1)
    u, singular_values, _ = np.linalg.svd(design, full_matrices=False)
    projected = u.T @ target
    orthogonal_energy = max(
        float(target @ target - projected @ projected), 0.0
    )
    low = max(float(config.get("gcv_lambda_min_factor", 0.05)), 1e-4)
    # GCV is restricted to no stronger shrinkage than the already conservative
    # base ridge.  The upper bound prevents a noisy, short lookback from
    # selecting the over-smoothed edge of an unbounded logarithmic grid.
    high = max(float(config.get("gcv_lambda_max_factor", 0.6)), low)
    grid_size = max(5, min(41, int(config.get("gcv_grid_size", 17))))
    candidates = base_lambda * np.geomspace(low, high, grid_size)
    candidates = np.unique(np.append(candidates, base_lambda))
    eigenvalues = singular_values * singular_values / float(sample_count)
    best_lambda = base_lambda
    best_score = np.inf
    for candidate in candidates:
        shrinkage = eigenvalues / (eigenvalues + candidate)
        residual_energy = orthogonal_energy + float(
            np.sum(((1.0 - shrinkage) * projected) ** 2)
        )
        denominator = max(float(sample_count - np.sum(shrinkage)), 1e-8)
        score = sample_count * residual_energy / (denominator * denominator)
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_lambda = float(candidate)
    return best_lambda


def _affine_capped_weights(weights, negative_cap):
    """Form an affine prototype mixture with bounded negative coordinates.

    Ridge also estimates an overall residual gain.  Factoring that gain from
    the prototype mixture lets the normalized mixture obey sum(a)=1 without
    incorrectly forcing every query to unit oscillation amplitude.
    """
    weights = np.asarray(weights, dtype=float)
    gain = float(np.sum(weights))
    if abs(gain) <= 1e-10:
        return weights
    affine = weights / gain
    cap = max(float(negative_cap), 0.0)
    affine = np.maximum(affine, -cap)
    negative = affine < 0.0
    positive = ~negative
    positive_mass = float(np.sum(affine[positive]))
    target_positive_mass = 1.0 - float(np.sum(affine[negative]))
    if positive_mass > 1e-12:
        affine[positive] *= target_positive_mass / positive_mass
    else:
        affine[:] = 1.0 / max(len(affine), 1)
    return gain * affine


def _stability_aware_ridge(design, target, scores, config):
    """SVD ridge with GCV, affine calibration, and bounded extrapolation."""
    retained = _prune_prototypes(design, scores, config)
    stable_design = design[:, retained]
    sample_count = max(stable_design.shape[0], 1)
    base_lambda = max(float(config.get("ridge_lambda", 0.008)), 1e-10)
    ridge_lambda = _gcv_lambda(stable_design, target, base_lambda, config)

    # The SVD representation whitens the correlated prototype coordinates and
    # avoids explicitly inverting an ill-conditioned Gram matrix.
    u, singular_values, vt = np.linalg.svd(stable_design, full_matrices=False)
    eigenvalues = singular_values * singular_values / float(sample_count)
    rhs_coordinates = singular_values * (u.T @ target) / float(sample_count)
    weights = vt.T @ (rhs_coordinates / (eigenvalues + ridge_lambda))

    constraint = str(config.get("weight_constraint", "sum_to_one")).strip().lower()
    if constraint == "sum_to_one":
        # Decompose coefficients into a scalar residual gain and affine
        # prototype coordinates.  The latter are sum-to-one by construction.
        weights = _affine_capped_weights(
            weights, float(config.get("negative_weight_cap", 0.25))
        )

    full_weights = np.zeros(design.shape[1], dtype=float)
    full_weights[retained] = weights
    return full_weights


def forecast(case, config=None):
    """Return a finite one-dimensional forecast of ``case['horizon']`` samples."""
    config = {} if config is None else config
    horizon = int(case["horizon"])
    query_history = np.asarray(case["query"], dtype=float).reshape(-1)
    source_keys = case["bank_keys"]
    source_values = case["bank_vals"]
    keys = np.asarray(source_keys, dtype=float)
    values = np.asarray(source_values, dtype=float)

    trend_enabled = _boolean(config, "trend_decoupling", True)
    spectral_enabled = _boolean(config, "spectral_extrapolation", True)
    ridge_enabled = _boolean(config, "ridge_prototype_regression", True)
    gate_enabled = _boolean(config, "adaptive_confidence_gating", True)
    robust_despiking = _boolean(config, "robust_despiking", True)
    phase_alignment = _boolean(config, "phase_aligned_retrieval", True)
    threshold = max(float(config.get("despike_threshold", 8.0)), 1.0)

    clean_query = (
        _despike_rows(query_history, threshold) if robust_despiking else query_history.copy()
    )
    residual, trend_future, _ = _trend_residual(
        clean_query, horizon, trend_enabled, config
    )
    query_scale = max(float(np.sqrt(np.mean(residual * residual))), 1e-8)
    normalized_query = residual / query_scale
    bank = _bank_entry(
        keys, values, horizon, config, robust_despiking, threshold, trend_enabled,
        source_keys, source_values,
    )

    _, query_spectral = _spectral_features(normalized_query)
    temporal_similarity = bank["normalized"] @ normalized_query / max(len(normalized_query), 1)
    spectral_similarity = bank["spectral_unit"] @ query_spectral.reshape(-1)
    spectral_weight = float(np.clip(config.get("spectral_similarity_weight", 0.3), 0.0, 1.0))
    similarity = (
        (1.0 - spectral_weight) * temporal_similarity
        + spectral_weight * spectral_similarity
    )
    bank_size = len(bank["normalized"])
    shortlist_size = min(bank_size, max(1, int(config.get("spectral_shortlist_k", 32))))
    shortlist = np.argpartition(similarity, -shortlist_size)[-shortlist_size:]
    top_k = min(shortlist_size, max(1, int(config.get("top_k", 12))))

    candidate_keys = bank["normalized"][shortlist]
    candidate_futures = bank["future_residual"][shortlist]
    if phase_alignment:
        max_shift = min(
            len(normalized_query) // 4,
            max(0, int(config.get("phase_max_shift", 12))),
        )
        candidate_keys, candidate_futures, temporal_aligned = _phase_align(
            normalized_query,
            candidate_keys,
            candidate_futures,
            max_shift,
            float(config.get("phase_alignment_strength", 0.2)),
        )
        local_scores = (
            (1.0 - spectral_weight) * temporal_aligned
            + spectral_weight * spectral_similarity[shortlist]
        )
    else:
        local_scores = similarity[shortlist]
    selected = np.argsort(local_scores)[-top_k:][::-1]
    selected_keys = candidate_keys[selected]
    selected_futures = candidate_futures[selected]
    selected_scores = local_scores[selected]

    design = selected_keys.T
    if ridge_enabled:
        stability_aware = _boolean(config, "stability_aware_ridge", True)
        whitening = str(config.get("ridge_whitening", "svd")).strip().lower()
        if stability_aware and whitening == "svd":
            weights = _stability_aware_ridge(
                design, normalized_query, selected_scores, config
            )
        else:
            ridge_lambda = max(float(config.get("ridge_lambda", 0.01)), 1e-10)
            gram = design.T @ design / max(len(normalized_query), 1)
            rhs = design.T @ normalized_query / max(len(normalized_query), 1)
            weights = _solve(gram + ridge_lambda * np.eye(top_k), rhs)
    else:
        temperature = max(float(config.get("retrieval_temperature", 0.2)), 1e-6)
        logits = (selected_scores - np.max(selected_scores)) / temperature
        weights = np.exp(np.clip(logits, -60.0, 0.0))
        weights /= max(float(np.sum(weights)), 1e-12)

    retrieved_residual = weights @ selected_futures
    reconstruction = design @ weights
    relative_error = float(
        np.sum((normalized_query - reconstruction) ** 2)
        / max(float(np.sum(normalized_query * normalized_query)), 1e-12)
    )
    spectral_residual = (
        _harmonic_future(
            normalized_query, horizon, max(0, int(config.get("spectral_modes", 5)))
        )
        if spectral_enabled
        else np.zeros(horizon, dtype=float)
    )
    gamma = (
        float(np.exp(-max(float(config.get("gate_alpha", 0.2)), 0.0) * relative_error))
        if gate_enabled
        else 1.0
    )
    fused_residual = gamma * retrieved_residual + (1.0 - gamma) * spectral_residual
    prediction = np.asarray(trend_future + query_scale * fused_residual, dtype=float)

    if prediction.shape != (horizon,) or not np.isfinite(prediction).all():
        trend_only = np.asarray(trend_future, dtype=float).reshape(-1)
        if trend_only.shape == (horizon,) and np.isfinite(trend_only).all():
            return trend_only
        last = float(clean_query[-1]) if len(clean_query) and np.isfinite(clean_query[-1]) else 0.0
        return np.full(horizon, last, dtype=float)
    return prediction

