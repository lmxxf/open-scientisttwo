import numpy as np


def fit_predict(x, y, test, config):
    """Immutable example baseline: ordinary least squares, intercept included."""
    a = np.column_stack([np.ones(len(x)), x])
    b = np.column_stack([np.ones(len(test)), test])
    return b @ np.linalg.lstsq(a, y, rcond=None)[0]
