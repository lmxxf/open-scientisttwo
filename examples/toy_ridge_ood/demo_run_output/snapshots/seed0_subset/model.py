import numpy as np


def fit_predict(x, y, test, config):
    """Least squares with optional Huber initialization and residual trimming."""
    a = np.column_stack([np.ones(len(x)), x])
    b = np.column_stack([np.ones(len(test)), test])
    y = np.asarray(y)
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    iterations = config.get("iterations", 0)
    if not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if iterations == 0:
        return b @ coef
    if config.get("method") != "trimmed_least_squares":
        raise ValueError("unsupported method")
    fraction = float(config.get("retain_fraction", 0.8))
    if not 0 < fraction <= 1:
        raise ValueError("retain_fraction must be in (0, 1]")
    initialization = config.get("initialization", "ols")
    if initialization not in ("ols", "huber_irls"):
        raise ValueError("unsupported initialization")

    if initialization == "huber_irls":
        for _ in range(iterations):
            residual = y - a @ coef
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            if scale <= np.finfo(float).eps:
                break
            weights = np.minimum(
                1.0, (1.345 * scale) / np.maximum(np.abs(residual), np.finfo(float).eps)
            )
            root_weights = np.sqrt(weights)
            coef = np.linalg.lstsq(
                a * root_weights[:, None], y * root_weights, rcond=None
            )[0]

    retained_count = max(1, int(np.ceil(fraction * len(a))))
    previous = None
    for _ in range(iterations):
        residual = np.abs(y - a @ coef)
        retained = np.sort(np.argsort(residual, kind="stable")[:retained_count])
        if previous is not None and np.array_equal(retained, previous):
            break
        coef = np.linalg.lstsq(a[retained], y[retained], rcond=None)[0]
        previous = retained
    return b @ coef
