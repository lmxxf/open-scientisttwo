import numpy as np


def fit_predict(x, y, test, config):
    """Huber IRLS with residual MAD scale and an unpenalized intercept."""
    a = np.column_stack([np.ones(len(x)), x])
    b = np.column_stack([np.ones(len(test)), test])
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    iterations = config.get("iterations", 0)
    if int(iterations) != iterations or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if iterations == 0:
        return b @ coef
    delta = float(config.get("delta", 1.345))
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    for _ in range(int(iterations)):
        residual = y - a @ coef
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if scale <= np.finfo(float).eps:
            break
        magnitude = np.abs(residual)
        weights = np.ones_like(magnitude, dtype=float)
        outside = magnitude > delta * scale
        weights[outside] = delta * scale / magnitude[outside]
        root_weights = np.sqrt(weights)
        coef = np.linalg.lstsq(
            a * root_weights[:, None], y * root_weights, rcond=None
        )[0]
    return b @ coef
