"""Budgeted CMA-MAE with evidence-gated descriptor frontier proposals."""
import numpy as np
from data import archive_cell_indices, evaluate_candidates


def _ridge(a, b, strength):
    if strength <= 1e-12:
        return np.linalg.lstsq(a, b, rcond=1e-10)[0]
    return np.linalg.solve(a.T @ a + strength * np.eye(a.shape[1]), a.T @ b)


def _ranks(x):
    # Equal observations receive equal percentile ranks.
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts)[inv] - 0.5 * counts[inv]) / max(1, len(x))


def optimize_qd(task, config=None):
    cfg = config or {}
    cov_on = bool(cfg.get('cma_covariance_adaptation', True))
    anneal = bool(cfg.get('archive_threshold_annealing', True))
    frontier = bool(cfg.get('frontier_boundary_guidance', True))
    surrogate = bool(cfg.get('local_surrogate_refinement', True))
    smooth = bool(cfg.get('dimension_adaptive_smoothing', True))
    gate_on = bool(cfg.get('reachability_gate', True))
    rng = np.random.default_rng(task['seed'])
    d, g, batch = task['dim'], task['grid_size'], task['batch_size']
    generations = min(task['budget_generations'], max(0, int(cfg.get('max_generations', task['budget_generations']))))
    aq = np.full((g, g), -np.inf)
    ax = np.zeros((g, g, d))
    ad = np.zeros((g, g, 2))
    t_min = float(cfg.get('t_min', 0.0))
    thresholds = np.full((g, g), t_min)
    alpha = float(np.clip(cfg.get('alpha_a', 0.15), 0, 1))
    sigma = float(cfg.get('sigma', 0.45))
    mean = np.zeros(d)
    variance = np.ones(d)
    pc, ps = np.zeros(d), np.zeros(d)
    mu = max(1, batch // 2)
    weights = np.log(mu + .5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    meff = 1 / np.sum(weights ** 2)
    cc = (4 + meff / d) / (d + 4 + 2 * meff / d)
    cs = (meff + 2) / (d + meff + 5)
    c1 = 2 / ((d + 1.3) ** 2 + meff)
    cmu = min(1 - c1, 2 * (meff - 2 + 1 / meff) / ((d + 2) ** 2 + meff))
    damping = 1 + cs + 2 * max(0, np.sqrt((meff - 1) / (d + 1)) - 1)
    hx, hq = [], []
    tx, td, tg, tu = [], [], [], []
    crossings = 0
    active_generations = 0
    ridge = float(cfg.get('transition_ridge', 0.1)) * (np.sqrt(d / 24) if smooth else 1.)
    for gen in range(generations):
        occupied = np.argwhere(np.isfinite(aq))
        parent = np.repeat(mean[None], batch, axis=0)
        pdesc = np.zeros((batch, 2))
        paired = np.zeros(batch, dtype=bool)
        target = np.zeros((batch, 2))
        directed = np.zeros(batch, dtype=bool)
        X = mean + sigma * rng.normal(size=(batch, d)) * np.sqrt(variance)
        jac = None
        reachable = not gate_on
        # Each fold holds out entire generations; compare directional error to
        # predicting zero descriptor displacement, with no held-out refitting.
        if len(tx) >= 2 * batch:
            dx, db, blocks = np.asarray(tx[-384:]), np.asarray(td[-384:]), np.asarray(tg[-384:])
            jac = _ridge(dx, db, ridge)
            if gate_on and len(np.unique(blocks)) >= 3:
                held = blocks % 3 == gen % 3
                if held.any() and (~held).sum() >= d:
                    fit = _ridge(dx[~held], db[~held], ridge)
                    pred = dx[held] @ fit
                    direction = np.asarray(tu[-384:])[held]
                    actual = np.sum(db[held] * direction, axis=1)
                    directional = np.sum(pred * direction, axis=1)
                    reachable = bool(np.mean((actual - directional) ** 2) < np.mean(actual ** 2))
        if len(occupied):
            ex = ax[occupied[:, 0], occupied[:, 1]]
            ed = ad[occupied[:, 0], occupied[:, 1]]
            eq = aq[occupied[:, 0], occupied[:, 1]]
            n = len(ex)
            # Archive mutations keep the emitter connected to diverse niches.
            count = batch // 3
            ids = rng.integers(n, size=count)
            parent[:count], pdesc[:count], paired[:count] = ex[ids], ed[ids], True
            X[:count] = ex[ids] + sigma * rng.normal(size=(count, d)) * np.sqrt(variance)
            if frontier and n >= 4 and reachable:
                active_generations += 1
                dist = np.linalg.norm(ed[:, None] - ed[None, :], axis=2)
                nn = np.argsort(dist, axis=1)[:, 1:min(n, 16)]
                radius = np.median(np.take_along_axis(dist, nn, axis=1), axis=1)
                novelty = np.mean(np.take_along_axis(dist, nn, axis=1), axis=1) / max(2 / g, np.median(radius))
                probs = novelty + .1
                probs /= probs.sum()
                for i in range(count):
                    e = rng.choice(n, p=probs)
                    near = nn[e]
                    away = ed[e] - ed[near].mean(axis=0)
                    away /= max(np.linalg.norm(away), 1e-8)
                    goal = np.clip(ed[e] + radius[e] * away, -.995, .995)
                    rr, ccells = archive_cell_indices(goal[None], g)
                    if np.isfinite(aq[rr[0], ccells[0]]):
                        empty = np.argwhere(~np.isfinite(aq))
                        if len(empty):
                            centers = (empty + .5) * 2 / g - 1
                            goal = centers[np.argmin(np.linalg.norm(centers - goal, axis=1))]
                    # Finite differences from neighboring elites give a local
                    # Jacobian; transition evidence regularizes this estimate.
                    local = _ridge(ex[near] - ex[e], ed[near] - ed[e], ridge)
                    if jac is not None:
                        local = .5 * (local + jac)
                    step = (goal - ed[e]) @ np.linalg.solve(local.T @ local + .02 * np.eye(2), local.T)
                    step *= min(1., 1.5 / max(np.linalg.norm(step), 1e-8))
                    parent[i], pdesc[i], paired[i] = ex[e], ed[e], True
                    target[i], directed[i] = goal, True
                    X[i] = ex[e] + step + .12 * rng.normal(size=d) * np.sqrt(variance)
            if surrogate and len(hx) >= max(2 * d, batch * 3):
                # Local ridge interpolation of negative log quality using a
                # diagonal quadratic; positive curvature enforces a trust step.
                history = np.asarray(hx[-384:])
                quality = np.asarray(hq[-384:])
                anchor = ex[np.argmax(eq)]
                nearest = np.argsort(np.linalg.norm(history - anchor, axis=1))[:max(2*d+8, 100)]
                u = history[nearest] - anchor
                features = np.column_stack((np.ones(len(u)), u, .5 * u*u))
                beta = _ridge(features, -np.log(np.maximum(quality[nearest], 1e-12)), .05 * (np.sqrt(d/24) if smooth else 1.))
                curvature = np.maximum(beta[1+d:], .05)
                for j in range(max(1, batch // 4)):
                    i = batch - j - 1
                    e = np.argmax(eq) if j == 0 else rng.integers(n)
                    gradient = beta[1:1+d] + curvature * (ex[e] - anchor)
                    step = -gradient / curvature
                    step *= min(1., .9 / max(np.linalg.norm(step), 1e-8))
                    parent[i], pdesc[i], paired[i] = ex[e], ed[e], True
                    X[i] = ex[e] + step + .06 * rng.normal(size=d)
        X = np.clip(X, -2.5, 2.5)
        quality, desc = evaluate_candidates(X, task)
        rows, cols = archive_cell_indices(desc, g)
        old_threshold = thresholds[rows, cols].copy()
        delta = quality - old_threshold
        progress = np.zeros(batch)
        if jac is not None and directed.any():
            unit = target - pdesc
            unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-8)
            progress[directed] = np.sum(((X - parent) @ jac)[directed] * unit[directed], axis=1)
        ranking = _ranks(delta)
        if directed.any():
            ranking = .75 * ranking + .25 * _ranks(progress)
        # New cells take priority independent of within-batch ordering.
        ranking += (~np.isfinite(aq[rows, cols])).astype(float)
        for i in range(batch):
            r, c = rows[i], cols[i]
            empty = not np.isfinite(aq[r, c])
            if quality[i] > aq[r, c]:
                aq[r, c], ax[r, c], ad[r, c] = quality[i], X[i], desc[i]
                crossings += int(empty and directed[i])
            if anneal:
                if quality[i] > thresholds[r, c]:
                    thresholds[r, c] = (1 - alpha) * thresholds[r, c] + alpha * quality[i]
            else:
                thresholds[r, c] = aq[r, c]
        selected = np.argsort(ranking)[::-1][:mu]
        step = (X[selected] - mean) / sigma
        weighted = weights @ step
        mean = weights @ X[selected]
        if cov_on:
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * meff) * weighted / np.sqrt(variance)
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * meff) * weighted
            variance = (1 - c1 - cmu) * variance + c1 * pc**2 + cmu * (weights @ (step**2))
            variance = np.clip(variance, .03, 12.)
            expected = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d*d))
            sigma *= np.exp(np.clip(cs / damping * (np.linalg.norm(ps) / expected - 1), -.12, .12))
            sigma = float(np.clip(sigma, .08, .8))
        hx.extend(X.copy()); hq.extend(quality.tolist())
        if paired.any():
            tx.extend((X[paired] - parent[paired]).copy())
            td.extend((desc[paired] - pdesc[paired]).copy())
            tg.extend([gen] * int(paired.sum()))
            directions = np.where(directed[:, None], target - pdesc, pdesc)
            directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
            tu.extend(directions[paired].copy())
    return dict(archive_q=aq, archive_x=ax,
                diagnostics=dict(evaluations=generations * batch,
                                 guided_generations=active_generations,
                                 frontier_crossings=crossings,
                                 frontier_crossings_per_evaluation=crossings / max(1, generations * batch)))
