"""CMA-MAE with learned frontier and local-surrogate emitters.

The optimizer deliberately learns every geometric quantity from evaluated
``(x, quality, descriptor)`` triples. No benchmark internals are inspected.
"""

import numpy as np

from data import archive_cell_indices, evaluate_candidates


def _as_bool(cfg, name, default=True):
    value = cfg.get(name, default)
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "off", "no"}
    return bool(value)


def _cell_center(r, c, g):
    return np.array([-1.0 + (2.0 * r + 1.0) / g,
                     -1.0 + (2.0 * c + 1.0) / g])


def _ridge_jacobian(xs, bs, center_x, center_b, ridge, k):
    """Return a local descriptor Jacobian from observed finite differences."""
    if len(xs) < 3:
        return None
    dist = np.sum((bs - center_b) ** 2, axis=1)
    ids = np.argsort(dist)[: min(k, len(xs))]
    dx = xs[ids] - center_x
    db = bs[ids] - center_b
    scale = max(float(np.median(np.sqrt(dist[ids] + 1e-12))), 1e-3)
    w = np.exp(-0.5 * dist[ids] / (scale * scale))
    a = dx.T @ (w[:, None] * dx) + ridge * np.eye(xs.shape[1])
    j_t = np.linalg.solve(a, dx.T @ (w[:, None] * db))
    return j_t.T


def _frontier_candidates(rng, archive_q, archive_x, archive_b, n, xs, bs,
                         cfg, base_sigma, smoothing):
    """Sparse-neighborhood boundary discovery with normal/tangent emitters."""
    g = archive_q.shape[0]
    d = archive_x.shape[2]
    occupied = np.argwhere(np.isfinite(archive_q))
    if n <= 0 or len(occupied) < 4:
        return np.empty((0, d))

    k = max(4, int(cfg.get("k", 32)))
    ridge = float(cfg.get("frontier_ridge", 2e-3))
    normal_scale = float(cfg.get("normal_step_scale", 1.5))
    tangent_scale = float(cfg.get("tangent_step_scale", 0.5))
    normal_weight = max(0.0, float(cfg.get("normal_emitter_fraction", 0.6)))
    tangent_weight = max(0.0, float(cfg.get("tangent_emitter_fraction", 0.4)))
    normal_fraction = normal_weight / max(normal_weight + tangent_weight, 1e-12)
    density_norm = _as_bool(cfg, "frontier_density_normalization", True)
    local_pca = _as_bool(cfg, "frontier_local_pca", True)
    pca_energy = float(cfg.get("intrinsic_dimension_energy", 0.95))

    empty_edges = []
    for r0, c0 in occupied:
        local_occ = 0
        local_empty = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                rr, cc = int(r0 + dr), int(c0 + dc)
                if 0 <= rr < g and 0 <= cc < g:
                    if np.isfinite(archive_q[rr, cc]):
                        local_occ += 1
                    else:
                        local_empty += 1
                        cardinal = 1.25 if abs(dr) + abs(dc) == 1 else 0.75
                        density = (1.0 + local_empty) / (1.0 + local_occ)
                        score = cardinal * (0.15 + archive_q[r0, c0])
                        if density_norm:
                            score *= np.sqrt(density)
                        empty_edges.append((int(r0), int(c0), rr, cc, score))
    edges = empty_edges
    # Once support is complete, keep the same finite-difference mechanism
    # useful: transport a better adjacent elite toward a weaker cell.  This is
    # interpolation inside the discovered boundary, not blind global novelty.
    if not edges:
        edges = []
        for r0, c0 in occupied:
            q0 = archive_q[r0, c0]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = int(r0 + dr), int(c0 + dc)
                if 0 <= rr < g and 0 <= cc < g and np.isfinite(archive_q[rr, cc]):
                    advantage = q0 - archive_q[rr, cc]
                    if advantage > 0.0:
                        score = (0.02 + advantage) * (0.2 + q0)
                        edges.append((int(r0), int(c0), rr, cc, score))
    if not edges:
        return np.empty((0, d))

    scores = np.asarray([e[4] for e in edges], dtype=float)
    scores = np.maximum(scores, 1e-12)
    scores /= scores.sum()
    chosen = rng.choice(len(edges), size=n, replace=True, p=scores)
    out = []
    for out_idx, edge_idx in enumerate(chosen):
        r0, c0, rt, ct, _ = edges[int(edge_idx)]
        x0 = archive_x[r0, c0]
        b0 = archive_b[r0, c0]
        target_b = _cell_center(rt, ct, g)
        jac = _ridge_jacobian(xs, bs, x0, b0, ridge, k)
        if jac is None:
            out.append(x0 + rng.normal(0.0, base_sigma, d))
            continue

        jj = jac @ jac.T + ridge * np.eye(2)
        outward = jac.T @ np.linalg.solve(jj, target_b - b0)
        max_out = normal_scale * (0.75 + 0.10 * np.sqrt(d))
        out_norm = np.linalg.norm(outward)
        if out_norm > max_out:
            outward *= max_out / out_norm

        z = rng.normal(size=d)
        if local_pca:
            local_dist = np.sum((bs - b0) ** 2, axis=1)
            local_ids = np.argsort(local_dist)[: min(k, len(xs))]
            local_dx = xs[local_ids] - x0
            _, singular, axes = np.linalg.svd(local_dx, full_matrices=False)
            energy = np.cumsum(singular * singular)
            if len(energy) and energy[-1] > 0.0:
                rank = 1 + int(np.searchsorted(
                    energy / energy[-1], np.clip(pca_energy, 0.0, 1.0)))
                basis = axes[: min(rank, len(axes))]
                z = basis.T @ (basis @ z)
        tangent = z - jac.T @ np.linalg.solve(jj, jac @ z)
        tn = np.linalg.norm(tangent)
        if tn > 1e-12:
            tangent /= tn
        if (out_idx + rng.random()) / max(n, 1) < normal_fraction:
            step = outward + 0.20 * tangent_scale * smoothing * tangent
        else:
            step = 0.45 * outward + tangent_scale * smoothing * tangent
        out.append(x0 + step)
    return np.asarray(out)


def _surrogate_candidates(rng, archive_q, archive_x, n, xs, qs,
                          base_sigma, cfg, smoothing):
    """Interpolate local log-quality with a ridge diagonal-quadratic model."""
    d = archive_x.shape[2]
    occ = np.argwhere(np.isfinite(archive_q))
    if n <= 0 or len(occ) == 0 or len(xs) < max(8, d // 2):
        return np.empty((0, d))

    vals = np.asarray([archive_q[r, c] for r, c in occ])
    cutoff = np.quantile(vals, float(cfg.get("surrogate_elite_quantile", 0.55)))
    pool = occ[vals >= cutoff]
    local_k = min(len(xs), max(20, int(cfg.get("surrogate_neighbors", 3 * d))))
    ridge = float(cfg.get("surrogate_ridge", 0.08))
    trust = float(cfg.get("surrogate_trust_radius", 0.25)) * smoothing
    out = []
    for _ in range(n):
        r0, c0 = pool[int(rng.integers(len(pool)))]
        x0 = archive_x[r0, c0]
        dist = np.sum((xs - x0) ** 2, axis=1)
        ids = np.argsort(dist)[:local_k]
        dx = xs[ids] - x0
        radius = max(float(np.median(np.sqrt(dist[ids] + 1e-12))), base_sigma)
        u = dx / radius
        phi = np.concatenate([u, 0.5 * u * u], axis=1)
        y = (np.log(np.maximum(qs[ids], 1e-12)) -
             np.log(max(archive_q[r0, c0], 1e-12)))
        weights = np.exp(-0.5 * dist[ids] / (radius * radius))
        lhs = phi.T @ (weights[:, None] * phi) + ridge * np.eye(2 * d)
        coef = np.linalg.solve(lhs, phi.T @ (weights * y))
        grad, curve = coef[:d], coef[d:]
        step_u = np.zeros(d)
        good = curve < -1e-3
        step_u[good] = -grad[good] / curve[good]
        step_u[~good] = grad[~good]
        norm = np.linalg.norm(step_u)
        if norm > 1e-12:
            step_u *= min(trust / norm, 1.0)
        jitter = rng.normal(0.0, 0.06 * smoothing, d)
        out.append(x0 + radius * (step_u + jitter))
    return np.asarray(out)


def optimize_qd(task, config=None):
    """Optimize a 2-D QD archive within exactly the supplied evaluation budget."""
    cfg = {} if config is None else dict(config)
    rng = np.random.default_rng(task["seed"])
    d = int(task["dim"])
    g = int(task["grid_size"])
    generations = int(task["budget_generations"])
    batch = int(task["batch_size"])

    use_cma = _as_bool(cfg, "cma_covariance_adaptation", True)
    use_anneal = _as_bool(cfg, "archive_threshold_annealing", True)
    use_frontier = _as_bool(cfg, "frontier_boundary_guidance", True)
    use_surrogate = _as_bool(cfg, "local_surrogate_refinement", True)
    use_smoothing = _as_bool(cfg, "dimension_adaptive_smoothing", True)

    smoothing = np.sqrt(32.0 / d) if use_smoothing else 1.0
    sigma = float(cfg.get("sigma", 0.23)) * smoothing
    alpha_a = float(cfg.get("alpha_a", 0.03))
    t_min = float(cfg.get("t_min", 0.0))
    frontier_fraction = float(cfg.get("frontier_fraction", 0.20))
    surrogate_fraction = float(cfg.get("surrogate_fraction", 0.20))

    archive_q = np.full((g, g), -np.inf, dtype=float)
    archive_x = np.zeros((g, g, d), dtype=float)
    archive_b = np.zeros((g, g, 2), dtype=float)
    thresholds = np.full((g, g), t_min, dtype=float)

    mean = np.zeros(d)
    diag_c = np.ones(d)
    path_c = np.zeros(d)
    mu = max(2, batch // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mu_eff = 1.0 / np.sum(weights * weights)
    c_mean = float(cfg.get("cma_mean_rate", 0.75))
    c_path = float(cfg.get("cma_path_rate", 2.0 / (d + 5.0)))
    c_cov = float(cfg.get("cma_covariance_rate", 2.0 / (d + 7.0)))
    rank_path = float(cfg.get("cma_rank_path", 0.16))

    history_x = []
    history_q = []
    history_b = []

    for gen in range(generations):
        if gen == 0:
            x_batch = rng.uniform(-1.15, 1.15, size=(batch, d))
            cma_count = batch
        else:
            n_frontier = int(round(batch * frontier_fraction)) if use_frontier else 0
            n_surrogate = int(round(batch * surrogate_fraction)) if use_surrogate else 0
            if n_frontier + n_surrogate > batch - 8:
                scale = (batch - 8) / max(n_frontier + n_surrogate, 1)
                n_frontier = int(n_frontier * scale)
                n_surrogate = int(n_surrogate * scale)
            cma_count = batch - n_frontier - n_surrogate

            z = rng.normal(size=(cma_count, d))
            y = z * np.sqrt(diag_c)[None, :]
            if use_cma and np.linalg.norm(path_c) > 1e-12:
                rank_z = rng.normal(size=(cma_count, 1))
                y += rank_z * (np.sqrt(rank_path) * path_c[None, :] /
                               max(np.linalg.norm(path_c), 1e-12))
            cma_x = mean[None, :] + sigma * y

            hx = np.asarray(history_x)
            hq = np.asarray(history_q)
            hb = np.asarray(history_b)
            frontier_x = _frontier_candidates(
                rng, archive_q, archive_x, archive_b, n_frontier,
                hx, hb, cfg, sigma, smoothing)
            surrogate_x = _surrogate_candidates(
                rng, archive_q, archive_x, n_surrogate,
                hx, hq, sigma, cfg, smoothing)
            parts = [cma_x]
            if len(frontier_x):
                parts.append(frontier_x)
            if len(surrogate_x):
                parts.append(surrogate_x)
            x_batch = np.concatenate(parts, axis=0)
            if len(x_batch) < batch:
                extra_z = rng.normal(size=(batch - len(x_batch), d))
                x_batch = np.concatenate(
                    [x_batch, mean + sigma * extra_z * np.sqrt(diag_c)], axis=0)
            x_batch = x_batch[:batch]

        x_batch = np.clip(np.nan_to_num(x_batch, nan=0.0, posinf=2.5, neginf=-2.5),
                          -2.5, 2.5)
        qualities, descriptors = evaluate_candidates(x_batch, task)
        rows, cols = archive_cell_indices(descriptors, g)

        deltas = np.empty(batch)
        for i in range(batch):
            r, c = int(rows[i]), int(cols[i])
            q = float(qualities[i])
            old_t = thresholds[r, c]
            # CMA-MAE improvement score.  New cells retain t_min while visited
            # cells anneal upward, so discoveries receive their intended
            # priority without an unrelated novelty bonus.
            deltas[i] = q - old_t
            if q > archive_q[r, c]:
                archive_q[r, c] = q
                archive_x[r, c] = x_batch[i]
                archive_b[r, c] = descriptors[i]
            if use_anneal:
                if q > old_t:
                    thresholds[r, c] = (1.0 - alpha_a) * old_t + alpha_a * q
            else:
                thresholds[r, c] = archive_q[r, c]

        history_x.extend(x_batch)
        history_q.extend(qualities)
        history_b.extend(descriptors)

        if gen > 0 and cma_count > 0:
            n_select = min(mu, cma_count)
            order = np.argsort(deltas[:cma_count])[::-1][:n_select]
            w = weights[:n_select].copy()
            w /= w.sum()
            selected_steps = (x_batch[order] - mean) / max(sigma, 1e-12)
            step = w @ selected_steps
            old_mean = mean.copy()
            mean = old_mean + c_mean * sigma * step
            if use_cma:
                path_c = ((1.0 - c_path) * path_c +
                          np.sqrt(c_path * (2.0 - c_path) * mu_eff) * step)
                rank_mu = w @ (selected_steps * selected_steps)
                diag_c = ((1.0 - c_cov) * diag_c + c_cov * rank_mu +
                          c_cov * rank_path * (path_c * path_c))
                diag_c = np.clip(diag_c, 0.08, 6.0)
            else:
                diag_c.fill(1.0)
                path_c.fill(0.0)
        elif gen == 0:
            order = np.argsort(deltas)[::-1][:mu]
            mean = weights @ x_batch[order]

    return {"archive_q": archive_q, "archive_x": archive_x}
