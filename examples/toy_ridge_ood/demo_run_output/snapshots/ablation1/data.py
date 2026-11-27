import numpy as np


def data(seed, n, split):
    rng = np.random.default_rng(seed + {'train': 0, 'validation': 10000, 'test': 20000}[split])
    x = rng.normal(size=(n, 4)) * (1 if split == 'train' else 1.5)
    y = x @ np.array([1., -2., .5, 3.]) + rng.normal(scale=.2, size=n)
    if split == 'train':
        mask = rng.random(n) < .12
        y[mask] += rng.normal(8, 15, mask.sum())
    return x, y
