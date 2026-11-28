"""CMA-MAE with adaptive frontier, surrogate, and elite emitters."""
import numpy as np
from data import archive_cell_indices, evaluate_candidates


def _allocation(cfg, saturated, reward, enabled, n):
    pre = {"frontier_secant": .25, "local_surrogate": .20, "archive_elite_mutation": .55}
    post = {"frontier_secant": .05, "local_surrogate": .15, "archive_elite_mutation": .80}
    names = ("frontier_secant", "local_surrogate", "archive_elite_mutation")
    base = cfg.get("post_saturation_shares" if saturated else "initial_shares",
                   post if saturated else pre)
    p = np.array([max(0., float(base.get(k, (post if saturated else pre)[k])))
                  for k in names]) * enabled
    if cfg.get("adaptive_emitter_allocation", True) and np.any(reward > 0):
        u = reward / (np.mean(reward[reward > 0]) + 1e-12)
        p *= np.exp(np.clip(.35*u, 0., 1.5))
    p = np.maximum(p, float(cfg.get("minimum_share", .05))*enabled)
    if p.sum() == 0: p[2] = 1.
    raw = n*p/p.sum()
    out = np.floor(raw).astype(int)
    for i in np.argsort(raw-out)[::-1][:n-int(out.sum())]: out[i] += 1
    return out


def optimize_qd(task, config=None):
    cfg = {} if config is None else dict(config)
    rng = np.random.default_rng(task["seed"])
    d, g = int(task["dim"]), int(task["grid_size"])
    gens, batch = int(task["budget_generations"]), int(task["batch_size"])
    cma = bool(cfg.get("cma_covariance_adaptation", True))
    anneal = bool(cfg.get("archive_threshold_annealing", True))
    frontier = bool(cfg.get("frontier_boundary_guidance", True))
    surrogate = bool(cfg.get("local_surrogate_refinement", True))
    smooth = bool(cfg.get("dimension_adaptive_smoothing", True))
    sigma, alpha = float(cfg.get("sigma", .25)), float(cfg.get("alpha_a", .15))
    tmin = float(cfg.get("t_min", 0.))

    aq = np.full((g, g), -np.inf)
    ax = np.zeros((g, g, d))
    threshold = np.full((g, g), tmin)
    occupied = []
    mean, diag, path = np.zeros(d), np.ones(d), np.zeros(d)
    cc, cp = 2./(d+6.), 2./(d+2.)
    history_x, history_q = [], []
    reward = np.zeros(3)
    decay = float(cfg.get("reward_ema_decay", .9))
    cov_gains, saturated = [], False

    for gen in range(gens):
        old_n = len(occupied)
        kinds = np.full(batch, 2, int)
        if gen == 0 or not occupied:
            X = rng.uniform(-1.2, 1.2, (batch, d))
        else:
            counts = _allocation(cfg, saturated, reward,
                                 np.array([frontier, surrogate, True]), batch)
            xs, ks = [], []
            # Secants between descriptor-adjacent elites approximate the local
            # descriptor Jacobian; extrapolation aims through an empty neighbor.
            boundary = []
            for r, c in occupied:
                empty, full = [], []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, co = r+dr, c+dc
                        if (dr or dc) and 0 <= rr < g and 0 <= co < g:
                            (full if np.isfinite(aq[rr, co]) else empty).append((rr, co))
                if empty: boundary.append((r, c, empty, full))
            for _ in range(counts[0]):
                if boundary:
                    r, c, empty, full = boundary[int(rng.integers(len(boundary)))]
                    tr, tc = empty[int(rng.integers(len(empty)))]
                    if full:
                        nr, nc = min(full, key=lambda z:(z[0]-tr)**2+(z[1]-tc)**2)
                        x = ax[r,c] + rng.uniform(.45,.95)*(ax[r,c]-ax[nr,nc])
                    else: x = ax[r,c] + rng.normal(0., .75*sigma, d)
                else:
                    r,c = occupied[int(rng.integers(len(occupied)))]
                    x = ax[r,c] + rng.normal(0., sigma, d)
                xs.append(x); ks.append(0)
            # Ridge linear + diagonal-quadratic response surface around an elite.
            for _ in range(counts[1]):
                r,c = occupied[int(rng.integers(len(occupied)))]
                x0, x = ax[r,c], None
                if len(history_x) >= max(8, d//2):
                    HX, hq = np.asarray(history_x), np.asarray(history_q)
                    ids = np.argsort(np.sum((HX-x0)**2, axis=1))[:min(len(HX), max(12,d+4))]
                    D = (HX[ids]-x0)/(sigma+1e-12)
                    A = np.c_[np.ones(len(ids)), D, .5*D**2]
                    ridge = float(cfg.get("surrogate_ridge", .08))
                    coef = np.linalg.solve(A.T@A+ridge*np.eye(A.shape[1]), A.T@hq[ids])
                    step = coef[1:1+d]/(np.abs(coef[1+d:])+.35)
                    norm = np.linalg.norm(step)
                    if norm > 0:
                        step *= min(1., 1.5/norm)
                        x = x0 + sigma*step + rng.normal(0., .08*sigma, d)
                if x is None: x = x0+rng.normal(0., .55*sigma, d)
                xs.append(x); ks.append(1)
            oq = np.array([aq[r,c] for r,c in occupied])
            topn = max(1, int(np.ceil(float(cfg.get("cvar_parent_fraction",.25))*len(occupied))))
            top = np.argsort(oq)[-topn:]
            for _ in range(counts[2]):
                j = int(top[int(rng.integers(topn))]) if rng.random()<.55 else int(rng.integers(len(occupied)))
                r,c = occupied[j]; parent = ax[r,c]
                z = rng.normal(size=d)
                if cma:
                    z = np.sqrt(diag)*z + np.sqrt(float(cfg.get("low_rank_path_strength",.18)))*path*rng.normal()
                    center = .55*parent+.45*mean
                else: center = parent
                xs.append(center+sigma*z); ks.append(2)
            X, kinds = np.clip(np.asarray(xs), -2.5, 2.5), np.asarray(ks)

        q, desc = evaluate_candidates(X, task)
        rows, cols = archive_cell_indices(desc, g)
        before = aq[rows, cols].copy()
        delta = q-threshold[rows, cols]
        for i in range(batch):
            r,c,qi = int(rows[i]),int(cols[i]),float(q[i])
            if qi > aq[r,c]:
                if not np.isfinite(aq[r,c]): occupied.append((r,c))
                aq[r,c], ax[r,c] = qi, X[i]
            if anneal and qi > threshold[r,c]:
                threshold[r,c] = (1-alpha)*threshold[r,c]+alpha*qi
            elif not anneal:
                threshold[r,c] = aq[r,c] if np.isfinite(aq[r,c]) else tmin

        history_x.extend(X.copy()); history_q.extend(q.tolist())
        if len(history_x)>12*d:
            history_x, history_q = history_x[-12*d:], history_q[-12*d:]

        # Rank-mu natural-gradient update, diagonal plus rank-one path sampling.
        if cma:
            mu = max(1,batch//2); order=np.argsort(delta)[::-1][:mu]
            w=np.log(mu+.5)-np.log(np.arange(1,mu+1)); w/=w.sum()
            old=mean.copy(); mean=w@X[order]
            y=(mean-old)/max(sigma,1e-12)
            path=(1-cp)*path+np.sqrt(cp*(2-cp))*y
            rankmu=w@(((X[order]-old)/max(sigma,1e-12))**2)
            diag=np.clip((1-cc)*diag+cc*rankmu, .08 if smooth else .02, 3.5 if smooth else 8.)
            if smooth:
                geom=np.exp(np.mean(np.log(diag+1e-12)))
                shrink=min(.2,2/np.sqrt(d)); diag=(1-shrink)*diag+shrink*geom

        # Per-emitter realized normalized QD gain per evaluation.
        for k in range(3):
            ids=np.flatnonzero(kinds==k)
            if len(ids):
                gain=np.maximum(0.,q[ids]-np.where(np.isfinite(before[ids]),before[ids],0.))
                reward[k]=decay*reward[k]+(1-decay)*float(gain.sum()/len(ids))
        cov_gains.append((len(occupied)-old_n)/float(g*g))
        win=int(cfg.get("coverage_saturation_window",3))
        if len(cov_gains)>=win:
            saturated=max(cov_gains[-win:]) < float(cfg.get("coverage_gain_threshold",.005))
    return {"archive_q": aq, "archive_x": ax}
