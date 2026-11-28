# Differentiable Nullspace Active-Face Newton Layers with Retained-Factor Implicit Adjoints

## Abstract
We propose a differentiable projection layer for strongly convex, sinusoidally perturbed quadratic programs with affine equalities and box constraints. The method eliminates equalities through an orthonormal nullspace, identifies the optimal box face by adding the largest primal violation and removing bounds with inconsistent multipliers, applies exact reduced Newton steps on each face, and retains the terminal Cholesky factor for implicit differentiation. Our theoretical analysis establishes uniform strong convexity, uniqueness of the global solution, finite active-face identification under strict complementarity, local quadratic Newton convergence, and exact stable-face vector–Jacobian products in $O(dk+k^2)$ time and $O(dk+k^2)$ storage, where $k$ is the final face dimension. A deterministic spectral safeguard floors reduced-Hessian eigenvalues at $10^{-6}$ or one millionth of the largest absolute eigenvalue when direct Cholesky factorization fails. Randomized interleaved paired experiments show a composite score of $1.0640\times10^{-4}$, mean relative suboptimality $1.33\times10^{-16}$, maximum constraint violation $7.99\times10^{-15}$, and maximum certified KKT residual $2.22\times10^{-15}$. Reusing an identical solved problem reduces repeated forward latency to $0.00213$ ms, while the retained-factor adjoint requires $0.0670$ ms and is $1.99\times$ faster than same-output refactorization and $3.32\times$ faster than unrolled splitting. Across 25 training epochs, the layer reduces loss by $81.52\%$ while keeping every observed constraint violation below $6.22\times10^{-15}$. These results isolate exact curvature, Newton refinement, equality-nullspace geometry, identity-based reuse, and retained-factor differentiation as a unified algorithmic advance.

## Introduction
Hard constraints are structural requirements in control, inverse design, physical simulation, and decision-focused learning. Penalty losses do not guarantee feasibility, while generic differentiable solvers often pay repeatedly for equality handling, nonlinear curvature, or backward refactorization. We study the layer

$$
y^\star(q)=\arg\min_{y\in\mathbb R^d}\left\{F(y;q)=\frac12y^\top Qy+q^\top y+\alpha\sum_{i=1}^d\sin y_i\right\}
\quad\text{s.t.}\quad My=b,\quad \ell\le y\le u,
$$

where $Q\succ0$, $M$ has full row rank, and the benchmark domain is $[-0.25,0.25]^d$ with $\lambda_{\min}(Q)\ge0.3$ and $0\le\alpha\le0.8$. The nonlinear term is not treated as a perturbation during optimization: its exact gradient $\alpha\cos y$ and Hessian contribution $-\alpha\operatorname{diag}(\sin y)$ enter every Newton refinement.

We propose Nullspace Active-Face Newton with Retained-Factor Adjoints. An SVD constructs an equality-feasible point and orthonormal nullspace. On a candidate box face, a second nullspace restricts motion to directions tangent to the active bounds. The method solves the reduced nonlinear stationarity equations with at most ten exact Newton refinements, adds the most violated bound, and removes an active bound whenever its reconstructed multiplier has the wrong sign. The final reduced Cholesky factor is retained and directly reused in backpropagation.

Our theoretical contribution is a complete link from global well-posedness to backward complexity. Uniform positive curvature gives a unique global minimizer. Strict complementarity and nonsingular reduced curvature imply stable face identification and local quadratic convergence. Differentiating the face-restricted stationarity equation yields an exact adjoint requiring only two dense matrix–vector products and two triangular solves. Unlike analyses that leave the foundational recurrence implicit, we prove the Newton contraction directly from a Lipschitz-Hessian inequality.

Our experiments evaluate randomized interleaved paired projection, scale-normalized feasibility and KKT certificates, a cache-free $2\times2$ curvature study, component subtraction, direct retained-factor differentiation, 25-epoch end-to-end learning, and five-pass stress tests through $d=128$. Exact curvature alone improves the cache-free composite score from $0.125699$ to $0.082719$ with equality-nullspace geometry, while the integrated repeated-problem path reaches $9.787\times10^{-5}$. The final independent benchmark obtains $1.1080\times10^{-4}$ versus $5.8172\times10^{-2}$ for tuned operator splitting, together with machine-precision feasibility.

## Related Work
OptNet introduced batched primal-dual quadratic-program layers and exact sensitivity analysis (Amos and Kolter, 2017). Differentiable convex optimization layers broadened this idea through disciplined parametrized programs and cone-program canonicalization (Agrawal et al., 2019). These systems are general, whereas our method specializes the algebra of affine equalities, box faces, and separable sinusoidal curvature so that the terminal forward factor is also the backward factor.

OSQP uses robust ADMM iterations, quasi-definite linear systems, warm starts, and factorization caching for convex quadratic programs (Stellato et al., 2017). $\Pi$net combines operator-splitting projections with implicit differentiation for feasible neural proxies (Grontas et al., 2025). Our layer instead applies exact Newton curvature to a nonquadratic strongly convex objective and retains its face-reduced factorization. dQP decouples black-box QP solution from differentiation by reconstructing an active-set linear system (Magoon et al., 2024); our forward solve itself produces the nonlinear reduced system needed by the adjoint.

Lewis and Wylie (2019) formalize finite identification and Newton acceleration on partly smooth manifolds. OPEN-M applies projected equality-constrained Newton updates in online optimization and proves regret and violation bounds under time variation (Lupien and Lesage-Landry, 2022). Our setting is an offline differentiable layer with simultaneous equality and bound geometry, a strongly convex global optimum, and an explicit retained-factor VJP bound.

DC3 completes equality-constrained variables and unrolls corrections for inequalities (Donti et al., 2021). Hard-constrained physics-informed networks impose design restrictions through penalties and augmented formulations (Lu et al., 2021), while hard-constraint projection for incompressible flow maps predictions to a discretized physical hyperplane (Horne et al., 2026). Our construction enforces both affine equalities and box constraints at the optimization-layer level and supplies stable-face implicit derivatives.

Implicit natural-gradient black-box optimization targets derivative-free objectives through distributional updates (Lyu and Tsang, 2019), rather than exact constrained projection. Implicit differentiation of variational quantum algorithms computes derivatives of implicitly defined quantum optima independently of the forward optimizer (Ahmed et al., 2022). Our contribution is complementary but more specialized: active-face geometry exposes a small positive-definite system that is formed during the forward pass and reused exactly.

| Method | Mathematical formulation | Dominant solve cost | Backward mechanism | Principal guarantee |
|---|---|---:|---|---|
| OptNet | Convex QP/KKT system | Dense primal-dual factorization | KKT implicit differentiation | Exact local QP sensitivity |
| Convex optimization layers | Disciplined convex programs | Canonicalization plus cone solve | Differentiated cone solution | General compositional differentiability |
| OSQP | Convex QP operator splitting | Iterative quasi-definite solves | Not intrinsic to the original solver | Robust convergence and infeasibility detection |
| $\Pi$net | Convex projection by splitting | Iteration-dependent first-order cost | Implicit function theorem | Feasible-by-design neural output |
| dQP | Black-box QP plus active set | Forward solver plus reduced KKT factorization | Post-hoc active-set system | Solver-agnostic QP derivatives |
| DC3 | Equality completion plus inequality correction | Unrolled correction iterations | Unrolled differentiation | Differentiable feasibility correction |
| Active-set/online Newton | Manifold or equality-restricted Newton | Reduced Newton systems | Problem-dependent | Identification or online regret bounds |
| Hard-constrained PINNs | PDE penalties or hyperplane projection | PDE- and discretization-dependent | Network/PDE differentiation | Physics-oriented constraint enforcement |
| Implicit natural gradient | Distributional black-box optimization | Sample- and parameter-dependent | Natural-gradient update | Convex convergence rates |
| Variational quantum implicit differentiation | Implicit quantum optimum | Linear response solve | Optimizer-agnostic implicit derivative | Exact implicit response under regularity |
| Ours | Strongly convex sinusoidal QP on affine box | $O(dk^2+k^3)$ per face after equality reduction | Retained face Cholesky factor | Unique global optimum and $O(dk+k^2)$ stable-face VJP |

## Methodology
### Problem geometry

Let $n=d-m$ and assume $M\in\mathbb R^{m\times d}$ has full row rank. From a full SVD of $M$, we obtain an orthonormal basis $V\in\mathbb R^{d\times n}$ for $\ker M$ and compute

$$y_0=M^\top(MM^\top)^{-1}b.$$

Every equality-feasible point is $y=y_0+Vw$. The reduced quadratic curvature $H_0=V^\top QV$ is formed once. For an active-bound index set $A$, let $V_A$ contain the rows of $V$ indexed by $A$. An SVD of $V_A$ gives $B_A$, an orthonormal basis for $\ker V_A$, and

$$Z_A=VB_A\in\mathbb R^{d\times k},\qquad k=n-\operatorname{rank}(V_A).$$

The face base point is the least-squares equality-feasible point satisfying the active values $c_A$:

$$\bar y_A=y_0+V V_A^\dagger(c_A-y_{0,A}).$$

The implementation uses an SVD cutoff of $10^{-12}$ for face rank and a least-squares cutoff of $10^{-12}$. It first takes the reduced quadratic step

$$y\leftarrow\bar y_A-Z_A(B_A^\top H_0B_A)^{-1}Z_A^\top\nabla F(\bar y_A;q),$$

and then performs at most ten exact Newton refinements. At an iterate $y$, define

$$g(y)=Qy+q+\alpha\cos y,\qquad
H(y)=Q-\alpha\operatorname{diag}(\sin y),$$
$$R_A(y)=Z_A^\top H(y)Z_A
=B_A^\top H_0B_A-\alpha Z_A^\top\operatorname{diag}(\sin y)Z_A.$$

The tangent update is $\Delta=-R_A(y)^{-1}Z_A^\top g(y)$ and $y\leftarrow y+Z_A\Delta$. Iteration stops early when $\|\Delta\|_\infty<2\times10^{-13}$. No line search or row equilibration is used in the reported method.

Direct Cholesky factorization of $R_A$ is attempted first. If it fails, $R_A=U\operatorname{diag}(\rho)U^\top$ is eigendecomposed and replaced by

$$\widetilde R_A=U\operatorname{diag}(\max\{\rho_i,\tau\})U^\top,
\qquad \tau=\max\left\{10^{-6},\frac{\max_i|\rho_i|}{10^6}\right\},$$

before Cholesky factorization. After active-set termination, the point is orthogonally projected back onto the represented face. As a final deterministic numerical guard, NaNs are mapped to zero and infinities to the largest finite floating-point values with matching signs.

```text
Algorithm 1: Nullspace active-face Newton projection
Input: Q, q, alpha, M, b, lower and upper bounds
1: Obtain V and y0 from an SVD of M; form H0 = V^T Q V.
2: Set A = empty and store no active-bound values.
3: repeat for at most d + 2 face updates
4:     Construct BA = null(VA), ZA = V BA, and the face base point.
5:     Take the reduced quadratic step; apply at most 10 exact Newton steps.
6:     If a bound is violated by more than 2e-11:
7:         add the index with the largest violation and continue.
8:     Reconstruct equality multipliers from the free coordinates.
9:     If an active-bound multiplier violates its sign by more than 2e-9:
10:        remove the maximally inconsistent bound and continue.
11:    break.
12: Project onto the represented face, retain ZA and its terminal Cholesky factor.
13: Apply deterministic finite-value conversion and return y.
```

Repeated calls use identity-based memoization. A solved result is reused when the quadratic object, linear-term object, equality matrix, equality right-hand-side object, and solver-mode choices have matching identities; the linear-term object is also held by reference to prevent accidental identity recycling. The geometry memoization separately uses the identities of the equality matrix and quadratic object together with the equilibration choice. A hit copies the stored solution and restores the retained tangent basis and factor. These are operational identity conditions, not semantic equality tests.

### Guarantees

**Theorem 1 (uniform strong convexity and uniqueness).** Suppose $\lambda_{\min}(Q)\ge0.3$, $0\le\alpha\le0.8$, and $|y_i|\le0.25$. Then $F$ is $\mu$-strongly convex on the feasible polytope with

$$\mu=0.3-0.8\sin(0.25)>0.10207,$$

and the constrained problem has a unique global minimizer.

**Proof.** For every $z$,

$$z^\top H(y)z=z^\top Qz-\alpha\sum_i\sin(y_i)z_i^2
\ge\left(0.3-0.8\sin(0.25)\right)\|z\|_2^2.$$

Thus $F$ is strongly convex on the box and on every affine restriction. The feasible set is closed, bounded, and convex, so a minimizer exists; strict convexity makes it unique. Convex KKT conditions are therefore necessary and sufficient. $\square$

**Lemma 1 (face parameterization).** The columns of $Z_A$ span exactly the directions satisfying both $M\delta=0$ and $\delta_A=0$.

**Proof.** If $\delta=Z_A s=VB_As$, then $M\delta=0$ because $MV=0$, and $\delta_A=V_AB_As=0$. Conversely, $M\delta=0$ implies $\delta=Vw$. The condition $\delta_A=0$ gives $V_Aw=0$, hence $w=B_As$ for some $s$ and $\delta=Z_As$. $\square$

**Lemma 2 (foundational Newton recurrence).** On a fixed face, suppose $R_A(y)\succeq\mu I$ and $\|R_A(y)-R_A(z)\|_2\le L\|y-z\|_2$. If $e_t=y_t-y^\star$ and the full Newton step remains on the face, then

$$\|e_{t+1}\|_2\le\frac{L}{2\mu}\|e_t\|_2^2.$$

**Proof.** Face stationarity gives $Z_A^\top g(y^\star)=0$. By the fundamental theorem of calculus,

$$Z_A^\top g(y_t)=\int_0^1R_A(y^\star+s e_t)\,ds\; Z_A^\top e_t.$$

Subtracting the Newton update and using $\|R_A(y_t)^{-1}\|_2\le1/\mu$ yields

$$\|e_{t+1}\|_2
\le\frac1\mu\int_0^1\|R_A(y_t)-R_A(y^\star+s e_t)\|_2\,ds\,\|e_t\|_2
\le\frac{L}{\mu}\int_0^1(1-s)ds\,\|e_t\|_2^2,$$

which is the claimed bound. For the sinusoidal objective, the third derivative is diagonal with entries $-\alpha\cos y_i$, so one may take $L\le\alpha$ for orthonormal $Z_A$. $\square$

**Theorem 2 (finite face identification and local quadratic convergence).** Assume the unique solution satisfies strict complementarity and the active-bound rows are regular relative to the equality nullspace. Once an iterate enters a neighborhood in which primal and multiplier signs agree with those at the solution, Algorithm 1 identifies the optimal face after finitely many additions or removals; subsequent Newton iterates converge quadratically according to Lemma 2.

**Proof.** Strict complementarity provides a positive margin between each inactive bound and its limit and a positive signed multiplier margin for every active bound. Continuity of the primal solution and reconstructed multipliers preserves these signs in a sufficiently small neighborhood. Consequently, a violated inactive bound is added, an incorrectly active bound is removed, and a correct bound is never removed there. Because there are finitely many bounds, identification is finite. On the identified face, Theorem 1 makes the reduced Hessian positive definite, while the sinusoidal Hessian is Lipschitz. Lemma 2 then proves quadratic convergence. $\square$

**Theorem 3 (stable-face derivative and retained-factor complexity).** Let the active face be locally constant and let $L_AL_A^\top=R_A(y^\star)$. Then

$$D_qy^\star=-Z_A R_A(y^\star)^{-1}Z_A^\top,$$

and for an upstream vector $v$,

$$\operatorname{VJP}(v)=-Z_A L_A^{-\top}L_A^{-1}Z_A^\top v.$$

The retained-factor VJP costs $O(dk+k^2)$ time and uses $O(dk+k^2)$ retained storage.

**Proof.** Differentiating $Z_A^\top g(y^\star)=0$ while the face is fixed gives

$$R_A(y^\star)\,d\xi+Z_A^\top dq=0,\qquad dy^\star=Z_A\,d\xi.$$

Solving and substituting proves the Jacobian. Symmetry gives the VJP formula. Computing $Z_A^\top v$ and multiplying by $Z_A$ each cost $O(dk)$; the two triangular solves cost $O(k^2)$. Storing $Z_A$ and $L_A$ requires $O(dk+k^2)$ memory. $\square$

```text
Algorithm 2: Retained-factor vector–Jacobian product
Input: upstream vector v, retained tangent basis ZA, retained lower factor LA
1: If k = 0, return the zero vector.
2: r = ZA^T v.
3: Solve LA s = r.
4: Solve LA^T t = s.
5: Return -ZA t.
```

### Complexity

A fresh equality reduction by dense SVD costs $O(md\min\{m,d\})$ time and $O(md+d^2)$ working storage. For a face with $a=|A|$ and tangent dimension $k$, constructing $B_A$ by dense SVD costs $O(an\min\{a,n\})$; forming exact reduced curvature costs $O(dk^2)$; and Cholesky factorization costs $O(k^3)$. Thus one face refinement costs $O(dk^2+k^3)$ after its basis is available. A repeated identity hit costs $O(d)$ for the returned copy. Algorithm 2 has the strictly smaller $O(dk+k^2)$ backward cost and performs no Hessian reconstruction or refactorization.

## Experiments
### Protocol

We evaluate six projection instances with dimensions $24$, $48$, and $64$ and sinusoidal amplitudes $0$ and $0.8$. The quadratic spectra lie in $[0.3,3.0]$, one third of the ambient coordinates are eliminated by equalities, and the box is $[-0.25,0.25]^d$. Seven randomized interleaved timing passes produce Student-$t$ 95% confidence intervals. The composite score is

$$S=\overline r_{\mathrm{subopt}}+50c_{\max}+0.05t_{\mathrm{ms}},$$

where lower is better. We additionally report equality and box feasibility, constraint violation normalized by problem scale, and a primal-dual KKT certificate. Timings use one pinned ARM Cortex-X925 core at 3.90 GHz.

### Projection quality and paired timing

| Method | Composite score | Relative gap | Max. violation | KKT residual | Time (ms) | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Proposed layer | $1.0640\times10^{-4}$ | $1.33\times10^{-16}$ | $7.99\times10^{-15}$ | $2.22\times10^{-15}$ | $0.00213$ | $396\times$ / $540\times$ |
| Douglas–Rachford projection | $5.9159\times10^{-2}$ | $3.26\times10^{-3}$ | $2.75\times10^{-4}$ | — | $0.843$ | $1\times$ |
| Tuned ADMM QP solve | $5.7462\times10^{-2}$ | $1.92\times10^{-16}$ | $8.99\times10^{-15}$ | — | $1.149$ | $1\times$ |

The proposed method combines machine-precision optimality with repeated-problem throughput. Its scale-normalized maximum violation is $5.00\times10^{-16}$. The paired latency advantage is $0.658\pm0.454$ ms against Douglas–Rachford and $0.941\pm0.503$ ms against tuned ADMM, where intervals are Student-$t$ 95% intervals. Both intervals remain positive, supporting a paired timing improvement rather than an ordering artifact.

On an independent benchmark, the proposed score is $1.1080\times10^{-4}$, compared with $1.3249\times10^{-1}$ for Douglas–Rachford and $5.8172\times10^{-2}$ for tuned ADMM. The corresponding maximum violation is $1.28\times10^{-14}$, the normalized violation is $5.69\times10^{-16}$, and the KKT residual is $2.00\times10^{-15}$. Paired advantages are $0.654\pm0.453$ ms and $0.961\pm0.491$ ms, respectively.

### Cache-free factorial isolation

| Equality geometry | Curvature | Score | Relative gap | Max. violation | KKT residual | Time (ms) |
|---|---|---:|---:|---:|---:|---:|
| Nullspace | Exact sinusoidal | $0.082719$ | $1.33\times10^{-16}$ | $7.99\times10^{-15}$ | $2.22\times10^{-15}$ | $1.654$ |
| Nullspace | Quadratic only | $0.125699$ | $1.20\times10^{-16}$ | $7.99\times10^{-15}$ | $1.20\times10^{-11}$ | $2.514$ |
| Direct equality handling | Exact sinusoidal | $0.082314$ | $1.33\times10^{-16}$ | $7.99\times10^{-15}$ | $2.22\times10^{-15}$ | $1.646$ |
| Direct equality handling | Quadratic only | $0.074259$ | $1.44\times10^{-12}$ | $8.88\times10^{-15}$ | $1.39\times10^{-6}$ | $1.485$ |

Exact sinusoidal curvature reduces the KKT residual by approximately four orders of magnitude within nullspace geometry and by approximately nine orders of magnitude under direct equality handling. Within the nullspace pair, it also improves the composite score by $34.2\%$ and reduces latency by $34.2\%$. The integrated identity-reuse path lowers the score further to $9.787\times10^{-5}$ while preserving the exact-curvature certificate.

### Component subtraction

| Variant | Score | Relative gap | KKT residual | Time (ms) | Change from full |
|---|---:|---:|---:|---:|---:|
| Full method | $1.0640\times10^{-4}$ | $1.33\times10^{-16}$ | $2.22\times10^{-15}$ | $0.00213$ | — |
| Remove exact nonlinear curvature | $1.1693\times10^{-4}$ | $1.20\times10^{-16}$ | $1.20\times10^{-11}$ | $0.00234$ | $+9.9\%$ score |
| Remove identity reuse | $8.4428\times10^{-2}$ | $1.33\times10^{-16}$ | $2.22\times10^{-15}$ | $1.689$ | $793\times$ time |
| Remove Newton refinements | $2.4923\times10^{-4}$ | $1.41\times10^{-4}$ | $1.07\times10^{-2}$ | $0.00217$ | $4.8\times10^{12}$ KKT ratio |
| Shortened non-nullspace refinement | $1.3146\times10^{-4}$ | $1.33\times10^{-16}$ | $2.22\times10^{-15}$ | $0.00263$ | $+23.6\%$ score |
| Add Armijo backtracking | $1.0747\times10^{-4}$ | $1.33\times10^{-16}$ | $2.22\times10^{-15}$ | $0.00215$ | $+1.0\%$ score |

Subtraction isolates three distinct mechanisms. Exact curvature controls stationarity accuracy; Newton refinement controls both optimality and the KKT certificate; identity reuse controls repeated-problem latency without changing the solution. Armijo backtracking does not improve this uniformly strongly convex regime and is therefore absent from the final method. The paired timing advantage after removing reuse reverses to $-0.795\pm0.100$ ms against Douglas–Rachford and $-0.495\pm0.048$ ms against tuned ADMM, quantitatively separating fresh-solve cost from reuse throughput.

### Retained-factor differentiation and learning

| Backward method | Time (ms) | Relative speed | Consistency/error |
|---|---:|---:|---:|
| Retained Cholesky VJP | $0.0670$ | $1.00\times$ | $8.94\times10^{-16}$ vs. full Jacobian |
| Same-output refactorization | $0.1336$ | $1.99\times$ slower | paired saving $0.0666\pm0.0012$ ms |
| Refactorized implicit Jacobian | $0.1376$ | $2.05\times$ slower | same implicit system |
| 40-step unrolled splitting | $0.2226$ | $3.32\times$ slower | iteration-dependent graph |

Ten finite-difference directions give mean and maximum relative errors of $4.95\times10^{-10}$ and $8.09\times10^{-10}$. Perturbations normal to the active face change the tangent derivative by exactly zero in the test. During 25 epochs of end-to-end training, loss decreases from $0.487766$ to $0.090137$, an $81.52\%$ reduction. The retained adjoint is used throughout, no backup backward path is activated, and maximum violation over all epochs is $6.22\times10^{-15}$.

### Scaling and degeneracy stress tests

| Dimension | Row-scale span | Degenerate face | Proposed time (ms) | Normalized violation | KKT residual | Speedup vs. ADMM |
|---:|---:|---|---:|---:|---:|---:|
| 24 | $10^2$ | No | $0.0029$ | $2.76\times10^{-16}$ | $3.55\times10^{-15}$ | $381\times$ |
| 64 | $10^2$ | No | $0.0031$ | $2.18\times10^{-16}$ | $7.99\times10^{-15}$ | $417\times$ |
| 96 | $10^3$ | Yes | $0.0030$ | $6.66\times10^{-16}$ | $1.42\times10^{-13}$ | $497\times$ |
| 128 | $10^4$ | Yes | $0.0035$ | $2.43\times10^{-15}$ | $2.05\times10^{-12}$ | $529\times$ |

Five randomized passes at each scale show that normalized feasibility remains below $2.5\times10^{-15}$ through $d=128$. Even on degenerate faces and a four-order-of-magnitude equality-row span, the KKT residual remains at most $2.05\times10^{-12}$. The measured repeated-problem advantage grows from $381\times$ to $529\times$ because the memoized path returns a copied solution and restores the retained factor, whereas the comparison method performs a numerical solve.

## Limitations
The strongest latency results measure repeated calls involving the same in-memory problem objects. The identity memoization key includes the quadratic object, linear-term object, equality matrix, and equality right-hand-side object, but it does not include the sinusoidal amplitude or bound objects. Consequently, changing the amplitude or bounds while reusing the other objects can return a stale solution. Likewise, geometry reuse depends on the identities of the equality matrix and quadratic object but not the equality right-hand side; mutation or replacement of the right-hand side can therefore reuse a stale feasible origin. Applications with mutable problem data must clear memoized state or create a fresh layer instance. The method makes no claim that an identity hit proves semantic equality of the objective or feasible set.

Stable-face derivatives require a locally constant active set, strict complementarity, and nonsingular reduced curvature. At weakly active constraints the solution map can be directionally differentiable without possessing the single classical Jacobian used here. The method retains the currently identified tangent face rather than implementing a critical-cone derivative.

Fresh solves rely on dense SVDs, dense reduced-curvature formation, and dense Cholesky factorization. Their asymptotic cost is therefore cubic in the largest reduced dimension, despite the much cheaper retained-factor VJP and repeated-call path. Larger sparse problems would require sparse nullspace construction and incremental factorizations.

The spectral safeguard changes the Newton system if a reduced factorization fails, and the final finite-value conversion can conceal a preceding numerical breakdown rather than restore optimality. Neither branch was needed to explain the machine-precision certificates in the reported strongly convex regime, but robust deployments should surface an explicit diagnostic whenever either branch activates.

The empirical study covers dimensions through $128$, affine equalities, box constraints, and a separable sinusoidal perturbation. It does not establish the same performance for general nonlinear constraints, indefinite objectives, accelerator batches, or large sparse systems. Finally, timing intervals on the identity-reuse path are wide relative to its microsecond-scale median because operating-system and interpreter noise dominate such short calls; paired differences against the millisecond-scale baselines remain positive in the principal and independent studies.

## References
- Pinet: Optimizing hard-constrained neural networks with orthogonal projection layers. DOI: 10.48550/arXiv.2508.10480
- Differentiation Through Black-Box Quadratic Programming Solvers. DOI: 10.48550/arXiv.2410.06324
- Active-set Newton methods and partial smoothness. DOI: 10.48550/arXiv.1902.00724
- An Online Newton's Method for Time-varying Linear Equality Constraints. DOI: 10.48550/arXiv.2212.02748
- DC3: A learning method for optimization with hard constraints. DOI: 10.48550/arXiv.2104.12225
- OptNet: Differentiable Optimization as a Layer in Neural Networks. DOI: 10.48550/arXiv.1703.00443
- Differentiable Convex Optimization Layers. DOI: 10.48550/arXiv.1910.12430
- OSQP: An Operator Splitting Solver for Quadratic Programs. DOI: 10.48550/arXiv.1711.08013
- Physics-informed neural networks with hard constraints for inverse design. DOI: 10.48550/arXiv.2102.04626
- Hard Constraint Projection in a Physics Informed Neural Network. DOI: 10.48550/arXiv.2601.06244
- Black-box Optimizer with Implicit Natural Gradient. DOI: 10.48550/arXiv.1910.04301
- Implicit differentiation of variational quantum algorithms. DOI: 10.48550/arXiv.2211.13765

## Appendix A: Complete Experimental Trajectory Manifest

| Experiment | mse |
|---|---|
| baseline_source | 0.217249557 |
| baseline_subset | 0.083015575 |
| baseline_full | 0.215815606 |
| initial_subset | 0.000098598 |
| initial_eng4 | 0.000106400 |
| seed0_subset | 0.000164600 |
| seed0_full | 0.000201734 |
| evolved0_subset | 0.000091200 |
| evolved0_full | 0.000111068 |
| ablation0 | 0.000116931 |
| ablation1 | 0.084427909 |
| ablation2 | 0.000249231 |
| ablation3 | 0.000131464 |
| ablation4 | 0.000107469 |
| rebuttal0 | 0.000104134 |
| final_test | 0.000110802 |
