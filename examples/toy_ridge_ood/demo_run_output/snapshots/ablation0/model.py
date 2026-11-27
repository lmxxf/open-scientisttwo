import numpy as np


def fit_predict(x, y, test, config):
    """Least squares with optional residual-MAD Huber IRLS and intercept."""
    a = np.column_stack([np.ones(len(x)), x])
    b = np.column_stack([np.ones(len(test)), test])
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    method = config.get("method", "ols")
    if method == "ols":
        return b @ coef
    if method != "huber_irls":
        raise ValueError("Unsupported regression method")
    iterations = config.get("iterations", 0)
    if not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if iterations == 0:
        return b @ coef
    if np.ndim(y) != 1:
        raise ValueError("Huber IRLS requires a one-dimensional target")
    if config.get("scale", "residual_mad") != "residual_mad":
        raise ValueError("Unsupported residual scale")
    delta = float(config.get("delta", 1.345))
    scale_floor = float(config.get("scale_floor", 1e-8))
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    if not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")
    for _ in range(iterations):
        residual = y - a @ coef
        scale = max(
            float(np.median(np.abs(residual - np.median(residual))))
            / 0.6744897501960817,
            scale_floor,
        )
        magnitude = np.abs(residual)
        cutoff = delta * scale
        weights = np.ones_like(residual, dtype=float)
        large = magnitude > cutoff
        weights[large] = cutoff / magnitude[large]
        root_weights = np.sqrt(weights)
        coef = np.linalg.lstsq(
            a * root_weights[:, None], y * root_weights, rcond=None
        )[0]
    return b @ coef
