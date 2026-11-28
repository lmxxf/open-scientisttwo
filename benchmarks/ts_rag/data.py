"""Deterministic multi-channel non-stationary time-series retrieval benchmark (7 canonical regimes)."""
import numpy as np

CHANNELS = [
    ("ETTh1_Thermal", [8.0, 24.0], 0.035, 0.18),
    ("ETTm1_Load", [6.0, 16.0, 32.0], 0.045, 0.22),
    ("Weather_Barometric", [12.0, 24.0], 0.028, 0.25),
    ("Electricity_Grid", [8.0, 12.0, 24.0], 0.030, 0.15),
    ("Traffic_Flow", [8.0, 24.0], 0.040, 0.20),
    ("Exchange_Drift", [16.0, 32.0], 0.060, 0.24),
    ("Solar_Harmonic", [8.0, 16.0], 0.020, 0.16),
]


def _synthesize_trajectory(rng, periods, drift_scale, noise_std, total_len=96):
    t = np.arange(total_len, dtype=float)
    level = rng.uniform(-1.2, 1.2)
    slope = rng.normal(0.0, drift_scale)
    quad = rng.normal(0.0, drift_scale * 0.004)
    trend = level + slope * t + quad * (t - total_len * 0.5) ** 2
    seasonal = np.zeros(total_len, dtype=float)
    for idx, p in enumerate(periods):
        amp = rng.uniform(0.45, 1.15) / (1.0 + 0.3 * idx)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        seasonal += amp * np.sin(2.0 * np.pi * t / p + phase) + 0.35 * amp * np.cos(2.0 * np.pi * t / p + phase)
    # Local AR(1) coloured shock
    ar = np.zeros(total_len, dtype=float)
    eps = rng.normal(0.0, noise_std, size=total_len)
    for i in range(1, total_len):
        ar[i] = 0.55 * ar[i - 1] + eps[i]
    clean = trend + seasonal
    observed = clean + ar
    return observed, clean, trend, seasonal


def dataset(mode="full", split="validation", extra=False):
    seed_base = (510 if split == "validation" else 920) + (5000 if extra else 0)
    rng = np.random.default_rng(seed_base)
    lookback, horizon = 64, 24
    bank_per_channel = 24 if mode == "subset" else 36
    queries_per_channel = 2 if mode == "subset" else 6

    bank_keys, bank_vals, bank_channels = [], [], []
    for c_idx, (cname, periods, drift_s, noise_s) in enumerate(CHANNELS):
        for _ in range(bank_per_channel):
            obs, clean, _, _ = _synthesize_trajectory(rng, periods, drift_s, noise_s * 0.6, lookback + horizon)
            bank_keys.append(obs[:lookback])
            bank_vals.append(clean[lookback:])
            bank_channels.append(c_idx)
    bank_keys = np.asarray(bank_keys, dtype=float)
    bank_vals = np.asarray(bank_vals, dtype=float)
    bank_channels = np.asarray(bank_channels, dtype=int)

    cases = []
    for c_idx, (cname, periods, drift_s, noise_s) in enumerate(CHANNELS):
        for q_i in range(queries_per_channel):
            obs, clean, trend, seasonal = _synthesize_trajectory(
                rng, periods, drift_s * (1.25 if extra else 1.0), noise_s * (1.15 if extra else 1.0), lookback + horizon
            )
            # Inject an occasional single-point sensor spike in history to test outlier robustness
            query_x = obs[:lookback].copy()
            if q_i % 2 == 1:
                spike_pos = int(rng.integers(8, lookback - 4))
                query_x[spike_pos] += rng.choice([-1.0, 1.0]) * 2.2 * noise_s
            cases.append({
                "channel": cname,
                "channel_id": c_idx,
                "query": query_x,
                "target": clean[lookback:],
                "true_trend_future": trend[lookback:],
                "true_seasonal_future": seasonal[lookback:],
                "bank_keys": bank_keys,
                "bank_vals": bank_vals,
                "lookback": lookback,
                "horizon": horizon,
            })
    return cases
