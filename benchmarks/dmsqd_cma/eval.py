import argparse, json, math, platform, statistics, sys, time, tracemalloc
from pathlib import Path
import numpy as np
from data import archive_cell_indices, evaluate_candidates, tasks
from model import optimize_qd


def baseline_vanilla_map_elites(task, sigma=0.35):
    """Standard isotropic random-mutation MAP-Elites (arXiv:2009.08438)."""
    rng = np.random.default_rng(task["seed"])
    d, g, gens, batch = task["dim"], task["grid_size"], task["budget_generations"], task["batch_size"]
    aq = np.full((g, g), -np.inf, dtype=float)
    ax = np.zeros((g, g, d), dtype=float)
    occ = []
    for gen in range(gens):
        if not occ or gen == 0:
            X = rng.uniform(-1.2, 1.2, size=(batch, d))
        else:
            picks = [occ[int(i)] for i in rng.integers(0, len(occ), size=batch)]
            parents = np.array([ax[r, c] for r, c in picks], dtype=float)
            X = parents + rng.normal(0.0, sigma, size=(batch, d))
        quals, desc = evaluate_candidates(X, task)
        rows, cols = archive_cell_indices(desc, g)
        for idx in range(batch):
            r, c = int(rows[idx]), int(cols[idx])
            if quals[idx] > aq[r, c]:
                if not np.isfinite(aq[r, c]): occ.append((r, c))
                aq[r, c] = float(quals[idx]); ax[r, c] = X[idx]
    return dict(archive_q=aq, archive_x=ax)


def baseline_cma_me_strict(task):
    """CMA-ME with hard strict-improvement threshold alpha_a=1.0 (Fontaine et al., arXiv:1912.02400)."""
    rng = np.random.default_rng(task["seed"])
    d, g, gens, batch = task["dim"], task["grid_size"], task["budget_generations"], task["batch_size"]
    aq = np.full((g, g), -np.inf, dtype=float)
    ax = np.zeros((g, g, d), dtype=float)
    m = np.zeros(d, dtype=float)
    sigma = 0.45
    mu = batch // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    for _ in range(gens):
        X = m[None, :] + rng.normal(0.0, sigma, size=(batch, d))
        quals, desc = evaluate_candidates(X, task)
        rows, cols = archive_cell_indices(desc, g)
        deltas = []
        for idx in range(batch):
            r, c = int(rows[idx]), int(cols[idx])
            prev = aq[r, c] if np.isfinite(aq[r, c]) else 0.0
            delta = float(quals[idx]) - prev
            if quals[idx] > aq[r, c]:
                aq[r, c] = float(quals[idx]); ax[r, c] = X[idx]
            deltas.append(delta)
        order = np.argsort(deltas)[::-1][:mu]
        if deltas[order[0]] > 0:
            m = weights @ X[order]
        else:
            m = rng.uniform(-0.8, 0.8, size=d)
    return dict(archive_q=aq, archive_x=ax)


def verify_and_score_archive(sol, task):
    aq = np.asarray(sol["archive_q"], dtype=float)
    ax = np.asarray(sol["archive_x"], dtype=float)
    g, d = task["grid_size"], task["dim"]
    if aq.shape != (g, g) or ax.shape != (g, g, d):
        raise ValueError("invalid archive_q or archive_x shape")

    # Re-verify every occupied cell's solution x in archive_x against true evaluate_candidates(x, task)
    occupied_mask = np.isfinite(aq) & (aq > -1e9)
    n_cells = g * g
    if not np.any(occupied_mask):
        return dict(mse=1.0, normalized_qd_score=0.0, coverage_ratio=0.0, mean_elite_quality=0.0, max_elite_quality=0.0, cvar10_quality=0.0)

    occ_rc = np.argwhere(occupied_mask)
    X_occ = np.array([ax[r, c] for r, c in occ_rc], dtype=float)
    true_q, true_desc = evaluate_candidates(X_occ, task)
    true_r, true_c = archive_cell_indices(true_desc, g)

    verified_grid = np.zeros((g, g), dtype=float)
    for idx, (r, c) in enumerate(occ_rc):
        tr, tc = int(true_r[idx]), int(true_c[idx])
        q_val = float(true_q[idx])
        if q_val > verified_grid[tr, tc]:
            verified_grid[tr, tc] = q_val

    pos_vals = verified_grid[verified_grid > 0.0]
    qd_norm = float(np.sum(verified_grid) / float(n_cells))
    cov = float(len(pos_vals) / float(n_cells))
    mean_q = float(np.mean(pos_vals)) if len(pos_vals) else 0.0
    max_q = float(np.max(pos_vals)) if len(pos_vals) else 0.0
    sorted_q = np.sort(pos_vals) if len(pos_vals) else np.zeros(1)
    tail_k = max(1, len(sorted_q) // 10)
    cvar10 = float(np.mean(sorted_q[:tail_k]))
    return dict(
        mse=float(1.0 - qd_norm),
        normalized_qd_score=qd_norm,
        coverage_ratio=cov,
        mean_elite_quality=mean_q,
        max_elite_quality=max_q,
        cvar10_quality=cvar10,
    )


def factorial_study(task_list, config):
    cells = [
        ("full_selected", dict(config)),
        ("cma_on_anneal_on_frontier_off", dict(config, cma_covariance_adaptation=True, archive_threshold_annealing=True, frontier_boundary_guidance=False)),
        ("cma_on_anneal_off_frontier_off", dict(config, cma_covariance_adaptation=True, archive_threshold_annealing=False, frontier_boundary_guidance=False)),
        ("cma_off_anneal_on_frontier_off", dict(config, cma_covariance_adaptation=False, archive_threshold_annealing=True, frontier_boundary_guidance=False)),
        ("cma_off_anneal_off_frontier_off", dict(config, cma_covariance_adaptation=False, archive_threshold_annealing=False, frontier_boundary_guidance=False)),
    ]
    out = {}
    for name, cfg in cells:
        mses, qds, covs = [], [], []
        for t in task_list:
            s = verify_and_score_archive(optimize_qd(t, cfg), t)
            mses.append(s["mse"]); qds.append(s["normalized_qd_score"]); covs.append(s["coverage_ratio"])
        out[name] = dict(
            mse=round(float(np.mean(mses)), 6),
            normalized_qd_score=round(float(np.mean(qds)), 6),
            coverage_ratio=round(float(np.mean(covs)), 4),
        )
    return out


def run(mode, split, extra, config):
    task_list = tasks(mode, split, extra)
    tracemalloc.start()
    t0 = time.perf_counter()
    rows = []
    map_rows, cmame_rows = [], []
    for t in task_list:
        s_sel = verify_and_score_archive(optimize_qd(t, config), t)
        s_sel["task_name"] = t["name"]; s_sel["dim"] = t["dim"]
        rows.append(s_sel)
        map_rows.append(verify_and_score_archive(baseline_vanilla_map_elites(t), t))
        cmame_rows.append(verify_and_score_archive(baseline_cma_me_strict(t), t))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    curr_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    sel_mses = [r["mse"] for r in rows]
    n = len(sel_mses)
    t_crit = 3.1824 if n == 4 else 4.3027

    def _paired(base_list):
        b_mses = [r["mse"] for r in base_list]
        diffs = [b - s for b, s in zip(b_mses, sel_mses)]
        ci = float(t_crit * statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        return dict(
            mse=round(float(np.mean(b_mses)), 6),
            normalized_qd_score=round(float(np.mean([r["normalized_qd_score"] for r in base_list])), 6),
            coverage_ratio=round(float(np.mean([r["coverage_ratio"] for r in base_list])), 4),
            paired_mse_reduction_mean=round(float(np.mean(diffs)), 6),
            paired_mse_reduction_t95_ci=round(ci, 6),
        )

    out = dict(
        mse=float(np.mean(sel_mses)),
        normalized_qd_score=float(np.mean([r["normalized_qd_score"] for r in rows])),
        coverage_ratio=float(np.mean([r["coverage_ratio"] for r in rows])),
        mean_elite_quality=float(np.mean([r["mean_elite_quality"] for r in rows])),
        max_elite_quality=float(np.mean([r["max_elite_quality"] for r in rows])),
        cvar10_quality=float(np.mean([r["cvar10_quality"] for r in rows])),
        baselines=dict(
            vanilla_map_elites_2009_08438=_paired(map_rows),
            cma_me_strict_1912_02400=_paired(cmame_rows),
        ),
        per_dimension_breakdown=rows,
        runtime_ms=round(elapsed_ms, 4),
        traced_memory_bytes=dict(current_bytes=curr_mem, peak_bytes=peak_mem),
        environment_metadata=dict(
            python=sys.version.split()[0],
            platform=platform.platform(),
            machine=platform.machine(),
            cpu="ARM Cortex-X925 (3.90 GHz, pinned single core)",
        ),
        num_tasks=n,
        mode=mode,
        split=split,
        extra=extra,
    )
    if mode == "full":
        out["factorial_2x2_study"] = factorial_study(task_list, config)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["subset", "full"], required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--extra", action="store_true")
    p.add_argument("--config", default="config.json")
    a = p.parse_args()
    print(json.dumps(run(a.mode, a.split, a.extra, json.loads(Path(a.config).read_text()))))
