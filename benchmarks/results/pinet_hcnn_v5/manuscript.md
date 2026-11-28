# Active-Face Newton Layers with Retained-Factor Adjoints and Critical-Cone Directional Differentiation

## Abstract
We propose a differentiable constrained optimization layer that unifies equality-nullspace reduction, exact active-face Newton steps, reusable constraint geometry, and retained-factor implicit adjoints. Our theoretical analysis establishes uniform strong convexity for bounded quadratic–sinusoidal objectives, feasibility preservation from feasible initialization, local quadratic convergence on an identified optimal face, and exact stable-face vector–Jacobian products with an arithmetic upper bound of $O(dk+k^2)$. For Euclidean projection at weakly active constraints, we characterize directional sensitivity through the critical cone, separately from reverse-mode face selections. Our experiments evaluate accuracy, randomized paired latency, orthogonal mechanism isolation, component subtraction, and dense scaling through dimension 128. Across six held-out problems, the integrated solver achieves mean relative objective gap $5.36\times10^{-16}$ and maximum constraint violation $2.13\times10^{-14}$, with latency ratios of $2.14\times$ and $2.66\times$ against measured Douglas–Rachford and OSQP-style ADMM implementations. A cache-free factorial study isolates a $1.88\times$ combined improvement from reduction and exact curvature, followed by a further $2.05\times$ improvement from geometry reuse. A factor-available backward microbenchmark measures a $5.67\times$ advantage over full implicit-Jacobian reconstruction; its scope differs from direct timing of the retained-factor interface. These results establish the computational value of coupling face geometry and curvature across constrained forward and backward computation.

## Introduction
Hard-constrained neural outputs arise when predictions must obey conservation equations, resource limits, or prescribed bounds. A constrained output layer must simultaneously produce a feasible point, optimize the intended objective, and provide useful derivatives. These requirements share mathematical structure: the tangent space and reduced curvature used to compute an optimum also determine its local sensitivity. Reconstructing them independently in the backward pass duplicates computation.

We propose an active-face Newton layer for affine equalities and box inequalities. An orthonormal equality-nullspace basis removes redundant primal degrees of freedom. Active bounds define a smaller tangent space on which exact Newton steps exploit the curvature of a smooth nonquadratic objective. Repeated evaluations reuse equality and face geometry, while the terminal tangent basis and Cholesky factor are retained for subsequent adjoint queries. The layer has no trainable internal parameters and can receive its linear objective coefficient from a neural network.

Our theoretical contribution is an explicit account of the guarantees and computational boundaries of this construction. We establish a uniform curvature bound exceeding $0.102$, prove feasibility preservation under tangent blocking steps, derive local quadratic convergence on an identified optimal face, and establish an $O(dk+k^2)$ upper bound for stable-face adjoints. We also distinguish two sensitivity objects that coincide only under additional regularity: a linear reverse-mode face selection and the generally nonlinear directional derivative at a weakly active projection point. The latter is projection onto the critical cone.

Our experiments evaluate the integrated method and isolate its mechanisms. Held-out median-based latency is $0.221897$ ms, compared with $0.474579$ ms for the measured Douglas–Rachford implementation and $0.590074$ ms for OSQP-style ADMM. Randomized interleaved paired Student-$t$ intervals establish positive mean latency reductions against both comparators. Without geometry caching, reduction and exact curvature jointly improve latency by $1.88\times$; geometry reuse supplies another $2.05\times$. A separate backward diagnostic measures $0.00925$ ms for a factor-available VJP versus $0.0524$ ms for full Jacobian reconstruction. We report that measurement according to its actual computation rather than identify it with a direct retained-interface or complete training-throughput benchmark.

The resulting contribution is a specialized constrained layer with aligned forward geometry, backward algebra, and explicit treatment of degeneracy. Its empirical scope is small dense synthetic optimization; its theoretical statements specify the conditions needed for exact optimization sensitivity.

## Related Work
Differentiable optimization layers. OptNet, Amos and Kolter, introduces quadratic programs as differentiable neural components through optimality-system differentiation ([OptNet](https://arxiv.org/abs/1703.00443)). Our quadratic special case shares this foundation: differentiating a reduced optimality system yields the optimizer Jacobian. Our specialization adds bounded nonquadratic curvature and an explicit terminal-factor interface. Differentiable Convex Optimization Layers, Agrawal et al., provides a broader construction through convex-program canonicalization and solver differentiation ([Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430)). We exchange modeling generality for affine-and-box structure, obtaining the direct coefficient sensitivity $-Z(Z^THZ)^{-1}Z^T$. Convexity of our sinusoidal objective is established on its restricted domain; we do not assume that this expression is immediately representable in a particular disciplined convex modeling system.

Orthogonal projection layers. Grontas, Terpin, Balta, D'Andrea, and Lygeros introduce $\Pi$net, combining operator splitting for convex projection with implicit differentiation, including optimization proxies, batching, and motion planning ([Pinet: Optimizing hard-constrained neural networks with orthogonal projection layers](https://arxiv.org/abs/2508.10480)). We share its feasible-output objective and implicit-function foundation. Our method instead uses active-face Newton iterations for a restricted strongly convex objective and retains reduced curvature for backward queries. Formally, Euclidean projection is the special case $Q=I$, $\alpha=0$, and $q=-z$; arbitrary supported $Q$ and $\alpha$ define a constrained minimizer rather than an orthogonal projector. Our measured CPU splitting comparator does not establish a ranking against the published JAX implementation or batched accelerator execution.

Black-box quadratic-program differentiation. Differentiation Through Black-Box Quadratic Programming Solvers separates forward QP solution from active-set-based differentiation ([Differentiation Through Black-Box Quadratic Programming Solvers](https://arxiv.org/abs/2410.06324)). Both constructions reduce sensitivity to an active optimality system. Our distinction is computational coupling: the forward layer exposes its terminal tangent basis and factor, and the nonquadratic reduced Hessian includes $-Z^T\operatorname{diag}(\alpha\sin y)Z$. This is a specialization of implicit differentiation, not a new implicit-function identity. At changing active sets, both the underlying solution map and the interpretation of its derivative require care.

Active manifolds and equality-constrained Newton methods. Lewis and Wylie's Active-set Newton methods and partial smoothness explains how manifold identification enables fast local Newton convergence ([Active-set Newton methods and partial smoothness](https://arxiv.org/abs/1902.00724)). Our local-rate result specializes this geometry to affine equalities and active box coordinates; it does not establish a general finite-identification theorem. An Online Newton's Method for Time-varying Linear Equality Constraints studies Newton updates under evolving equalities ([An Online Newton's Method for Time-varying Linear Equality Constraints](https://arxiv.org/abs/2212.02748)). Our method additionally activates and releases bounds and targets repeated layer evaluation. Cached geometry requires unchanged matrix objects and valid matrix contents; it provides no tracking guarantee under arbitrary changes to the equality system.

Operator splitting. OSQP develops operator splitting for quadratic programs ([OSQP: An Operator Splitting Solver for Quadratic Programs](https://arxiv.org/abs/1711.08013)). Our measured comparator is an OSQP-style over-relaxed ADMM implementation with an explicit inverse, fixed iteration budget, and a sinusoidal gradient update. It is distinct from the optimized OSQP package. The proposed layer instead terminates using reduced stationarity and bound-multiplier signs, and factors the exact reduced nonlinear curvature. These differences are relevant to the measured dense regime; sparse structure and batching can change the computational comparison.

Learning feasibility and physical constraints. DC3 combines learning with constraint completion and correction ([DC3: A learning method for optimization with hard constraints](https://arxiv.org/abs/2104.12225)). We instead solve a specified strongly convex objective, preserve feasibility through tangent steps and bound blocking, and differentiate local optimizer sensitivity. Physics-informed neural networks with hard constraints for inverse design addresses hard constraints in physics-informed learning ([Physics-informed neural networks with hard constraints for inverse design](https://arxiv.org/abs/2102.04626)). Hard Constraint Projection in a Physics Informed Neural Network likewise places projection within physics-informed learning ([Hard Constraint Projection in a Physics Informed Neural Network](https://arxiv.org/abs/2601.06244)). Our finite-dimensional results provide a potential output-layer primitive when constraints have affine-and-box form, but establish neither PDE accuracy nor superiority to application-specific boundary constructions.

Together, these ten works motivate three separate questions: whether outputs satisfy constraints, whether they minimize the intended objective, and whether the reported backward operator differentiates the relevant solution map. Our construction answers these questions under explicit geometric assumptions and evaluates their numerical manifestations separately.

## Methodology
Problem formulation. Let a network produce $q\in\mathbb R^d$. We define
$$y^\star(q)=\arg\min_{y\in C} f(y;q),\qquad f(y;q)=\frac12y^TQy+q^Ty+\alpha\sum_{i=1}^d\sin y_i,$$
where $C=\{y:My=b,\ l\le y\le u\}$ is nonempty, $l_i=-0.25$, $u_i=0.25$, $Q$ is symmetric with spectrum in $[0.3,3]$, and $0\le\alpha\le0.8$. All parameters except $q$ are fixed for the coefficient derivative. Write $n=d-\operatorname{rank}(M)$ and let $k$ denote the current face-tangent dimension.

Proposition 1: uniform strong convexity. The objective is uniformly $\mu$-strongly convex on the box, with
$$\mu=0.3-0.8\sin(0.25)>0.102.$$
The constrained minimizer exists, is unique, and is characterized by $0\in\nabla f(y;q)+N_C(y)$.

Proof. Its Hessian is $H(y)=Q-\alpha\operatorname{diag}(\sin y)$. Since $|y_i|\le0.25$, every vector $v$ satisfies
$$v^TH(y)v\ge[0.3-0.8\sin(0.25)]\|v\|_2^2.$$
Integration along a feasible line segment establishes strong convexity. Compactness and nonemptiness give existence, and strong convexity gives uniqueness. The normal-cone condition is necessary and sufficient for minimizing a differentiable convex function over a closed convex set. For this polyhedron it is equivalent to equality multipliers and sign-compatible bound normals. $\square$

Equality geometry and initialization. We compute an orthonormal basis $V\in\mathbb R^{d\times n}$ of $\ker M$ and precompute $H_0=V^TQV$. Nullspace construction uses complete QR with rank threshold equal to the largest absolute diagonal entry of its triangular factor multiplied by machine precision and the maximum matrix dimension. If QR indicates rank deficiency, SVD determines the nullspace using the analogous singular-value threshold.

With $c=(l+u)/2$, initialization solves a least-squares correction,
$$y_0=c+M^\dagger(b-Mc).$$
If a bound is violated by more than $10^{-12}$, Dykstra iterations alternate affine and box projections using the affine pseudoinverse. The cap is 20,000 iterations and the stopping threshold is absolute equality residual below $10^{-12}$. The finite initialization procedure does not certify feasibility for arbitrary inputs.

Active-face Newton steps. Let $A$ contain active coordinates. Compute orthonormal $B_A$ spanning $\ker V_A$, and define
$$Z_A=VB_A,\qquad K_A=B_A^TH_0B_A.$$
Then $MZ_A=0$ and $(Z_A)_A=0$. At a current point $y$,
$$g=Qy+q+\alpha\cos y,\qquad g_A=Z_A^Tg,$$
$$R_A=K_A-Z_A^T\operatorname{diag}(\alpha\sin y)Z_A,\qquad p=-Z_AR_A^{-1}g_A.$$
We factor $R_A=L_AL_A^T$ and apply two triangular substitutions. The step length is the smaller of one and the first nonactive bound-intersection ratio. Components with magnitude at most $10^{-13}$ do not define blocking ratios. A reached blocking coordinate is added to $A$.

Initial active coordinates are within $10^{-10}$ of a bound. If $\|g_A\|_\infty<2\times10^{-10}$, active normal coefficients are estimated by least squares from $V_A^T\lambda_A=-V^Tg$. Multiplying each coefficient by $-1$ at a lower bound and $+1$ at an upper bound gives its sign test. The coordinate with the most negative signed coefficient is released if that value is below $-2\times10^{-9}$; otherwise convergence is recorded. Bound sides for this check are classified with tolerance $10^{-8}$.

Geometry reuse and selected settings. Repeated evaluations with identical $Q$ and $M$ objects reuse $V$, $H_0$, and face tuples $(B_A,Z_A,K_A)$. The cache retains at most eight matrix-geometry entries, with no explicit cap on the number of faces within an entry. A previous terminal active set and its stored bound sides define a nullspace least-squares warm start; it is accepted only if all bounds hold within $10^{-12}$. The selected method enables nullspace reduction, exact Hessians, face caching, and critical-cone directional differentiation. Row equilibration and Armijo backtracking are disabled, and the Newton cap is 80. Disabling reduction retains ambient coordinates and constructs face tangents from the stacked equality and active-coordinate constraints. Disabling exact curvature replaces $R_A$ by $K_A$. The optional, unused equilibration normalizes equality rows and right-hand sides, while optional Armijo backtracking uses coefficient $10^{-4}$ and step halving.

```text
Algorithm 1: Active-face Newton layer
Input: Q, q, alpha, M, b, lower, upper
Obtain cached or newly constructed V and H0 = V^T Q V
Initialize y by affine least squares; apply finite Dykstra phase if needed
Accept a cached-face warm start when it is bound-feasible
Initialize A from coordinates within 1e-10 of their bounds
converged = false
For at most 80 iterations:
    Obtain B, Z = V B, and K = B^T H0 B for A
    g = Q y + q + alpha cos(y)
    If max_abs(Z^T g) < 2e-10:
        Estimate active normal coefficients by least squares
        If the minimum signed coefficient is below -2e-9:
            Release that coordinate and continue
        converged = true
        Break
    R = K - Z^T diag(alpha sin(y)) Z
    L = Cholesky(R)
    p = -Z backward_substitute(L^T, forward_substitute(L, Z^T g))
    Compute the first nonactive bound-intersection ratio
    Take the feasible step with length at most one
    Add the blocking coordinate if its bound is reached
Reconstruct terminal Z and factor terminal curvature R = L L^T
Retain Z, L, terminal active coordinates, and bound sides
Set last_status to 'converged' or 'iteration_limit'
Return y
```

The terminal factor is formed at the returned point, including when the iteration budget is exhausted. The public solve operation returns only $y$; convergence status is stored separately in last_status.

Proposition 2: feasibility preservation and local quadratic convergence. Suppose initialization is feasible and active coordinates equal their assigned bounds in exact arithmetic. Each bound-blocked tangent step remains feasible. If the optimal face is identified, the iterate is sufficiently close to its relative-interior optimum, and exact full Newton steps are taken, then in orthonormal face coordinates $y=y_A+Z_Aw$,
$$\|w_{t+1}-w^\star\|_2\le\frac{\alpha}{2\mu}\|w_t-w^\star\|_2^2.$$

Proof. The identities $MZ_A=0$ and $(Z_A)_A=0$ preserve the equalities and active coordinates. Clipping at the first inactive bound preserves all remaining bounds. The reduced Hessian is bounded below by $\mu I$. Because sine is 1-Lipschitz and $Z_A$ is orthonormal, this Hessian is $\alpha$-Lipschitz in $w$. The integral Taylor remainder for the reduced gradient is bounded by $(\alpha/2)\|w_t-w^\star\|_2^2$. Multiplication by the inverse reduced Hessian, whose norm is at most $1/\mu$, proves the result. Near the relative-interior optimum, inactive bounds do not block the local full step. $\square$

This proposition is conditional on feasible initialization and correct face identification. It does not establish global convergence of the finite-budget undamped iteration.

Theorem 3: exact stable-face adjoint and computational upper bound. Suppose a neighborhood of $q$ has a fixed optimal face, with orthonormal tangent basis $Z\in\mathbb R^{d\times k}$. Let $LL^T=Z^TH(y^\star)Z$. Then
$$D_qy^\star=-ZL^{-T}L^{-1}Z^T,$$
and the cotangent $v$ has VJP
$$\operatorname{VJP}(v)=-ZL^{-T}\bigl(L^{-1}Z^Tv\bigr).$$
Using retained $Z,L$, this computation has an arithmetic upper bound of $O(dk+k^2)$ and requires $O(k)$ working storage beyond the retained factors and output.

Proof. Differentiate reduced stationarity $Z^T\nabla f(y^\star;q)=0$ on the fixed face:
$$Z^TH(y^\star)Z\,dw+Z^T\,dq=0,\qquad dy=Z\,dw.$$
Proposition 1 makes the reduced Hessian invertible. Solving and substituting gives the symmetric Jacobian above. Two tangent matrix–vector products require $O(dk)$ arithmetic, and two triangular substitutions require $O(k^2)$. The intermediate vectors have length $k$. $\square$

```text
Algorithm 2: Retained-factor coefficient VJP
Input: cotangent v; terminal tangent Z and Cholesky factor L
a = Z^T v
b = forward_substitute(L, a)
c = backward_substitute(L^T, b)
Return -Z c
```

Exactness requires a converged optimum and exact terminal curvature. At an iteration-limited point the computation is a local curvature operator. At weak activity, fixing the retained face defines a linear reverse-mode selection, not necessarily a classical derivative. Interpreting it as an adjacent-face generalized-Jacobian element additionally requires a face realized by nearby regular solutions.

Theorem 4: directional derivative of polyhedral Euclidean projection. Let $C$ be a nonempty polyhedron, $x=P_C(z)$, and $n=z-x$. Define $K=T_C(x)\cap n^\perp$. Then
$$DP_C(z)[h]=P_K(h).$$
For affine equalities and box bounds, the tangent cone imposes $Mv=0$, $v_i\ge0$ at lower bounds, and $v_i\le0$ at upper bounds.

Proof. Let $x_t=P_C(z+th)$ and $v_t=(x_t-x)/t$. Nonexpansiveness bounds $v_t$. Polyhedral feasibility locally agrees with the tangent inequalities. Expanding the projection objective relative to $x$, with constants removed, gives
$$-t n^Tv+\frac12t^2\|v-h\|_2^2.$$
The projection normal satisfies $n^Tv\le0$ for every tangent direction. Comparing the optimal $v_t$ with zero shows $n^Tv_t\to0$, so every cluster point belongs to $K$. Comparing with any fixed $v\in K$, dividing by $t^2$, and dropping the nonnegative term $-n^Tv_t/t$ gives $\|v_t-h\|_2^2\le\|v-h\|_2^2$. Every cluster point is therefore the unique projection of $h$ onto $K$, proving the directional limit. $\square$

The directional helper requires the actual Euclidean projection normal, supplied explicitly. It classifies tight bounds with tolerance $10^{-8}$ and includes the normalized normal as an equality if its norm exceeds $10^{-12}$. It solves the cone projection with quadratic data $Q=I$, $q=-h$, zero sinusoidal coefficient, zero initial direction, and at most $4d+20$ active-face iterations, caching geometry locally. With critical-cone treatment disabled, all tight coordinates become equalities and the helper returns a linear tangent projector. At $z=x=0$ for $C=[0,\infty)$, the correct directional derivative is $\max(h,0)$, whereas the tight-equality approximation is zero. This theorem concerns projection input $z$, not general coefficient sensitivity for the nonquadratic objective.

Complexity. Dense equality setup has a conservative $O(d^3)$ time and $O(d^2)$ storage bound. New faces use complete QR, with SVD when rank deficiency is detected; no rank-one QR update is assumed. For $a$ active rows, complete QR of the $n\times a$ transpose costs conservatively $O(na\min(n,a)+n^2\min(n,a))$. Forming $Z_A$ and $K_A$ costs $O(dnk+n^2k+nk^2)$. With geometry available, a Newton iteration costs $O(d^2+dk^2+k^3)$, including the dense gradient, nonlinear reduced curvature, factorization, and multiplier work. A conservative bound over $T$ iterations is $O((T+1)d^3)$ before phase I. After pseudoinverse setup, $P$ Dykstra iterations add $O(Pmd)$ work, with $P\le20{,}000$. Across cached faces, storage is $O(d^2+\sum_A(nk_A+dk_A+k_A^2))$ per matrix geometry, with at most eight such geometries. The terminal adjoint state alone occupies $O(dk+k^2)$ storage. Each subsequent VJP satisfies the upper bound established in Theorem 3; this is not a lower-bound or optimality claim.

## Experiments
Evaluation protocol. Primary validation and held-out evaluations each contain six synthetic problems: $d\in\{24,48,64\}$ crossed with $\alpha\in\{0,0.8\}$. Reference solutions have independently checked stationarity residual at most $10^{-10}$. We report nonnegative relative objective gap $r=\max\{0,[f(y)-f(y^\star)]/\max(1,|f(y^\star)|)\}$, maximum absolute equality-or-box violation $c$, and composite score $S=\bar r+50c+0.05\tau$, where $\tau$ is latency in milliseconds. A further $1000c$ penalty applies when $c>10^{-2}$ and is inactive in the primary comparisons. This score combines optimization accuracy, feasibility, and computation.

Each primary timing comparison uses seven passes with randomized method order within each problem. Displayed latency is the mean of per-problem median timings. Paired inference uses differences of pass means: for $\Delta_j=\tau_{B,j}-\tau_{N,j}$, the Student-$t$ 95% interval is $\bar\Delta\pm2.4469s_\Delta/\sqrt7$, with six degrees of freedom. These are different estimands, so paired mean savings need not equal differences between displayed point latencies. Confidence intervals characterize repeated timing, not uncertainty over a large sampled problem population.

Baselines. DR denotes a 40-iteration Douglas–Rachford implementation with step size $0.08$. OSQP-style ADMM uses 80 iterations, penalty $\rho=2$, and relaxation $1.6$; its sinusoidal gradient is updated explicitly. Both construct their projection or linear-system setup inside each solve call. A separate uncached DR implementation is available on validation. These are measured implementations rather than benchmarks of the published packages.

| Split | Method | Latency ms | Composite score | Mean relative gap | Maximum violation | Baseline/proposed latency |
|---|---|---:|---:|---:|---:|---:|
| Validation | Proposed | 0.224272 | 0.011214 | 9.25e-17 | 1.14e-14 | 1.00 |
| Validation | DR | 0.462533 | 0.040124 | 3.26e-3 | 2.75e-4 | 2.06 |
| Validation | OSQP-style ADMM | 0.579025 | 0.028951 | 1.92e-16 | 8.99e-15 | 2.58 |
| Validation | Uncached DR | 1.469298 | 0.090462 | 3.26e-3 | 2.75e-4 | 6.55 |
| Held-out | Proposed | 0.221897 | 0.011095 | 5.36e-16 | 2.13e-14 | 1.00 |
| Held-out | DR | 0.474579 | 0.114115 | 2.52e-3 | 1.76e-3 | 2.14 |
| Held-out | OSQP-style ADMM | 0.590074 | 0.029504 | 5.38e-16 | 2.04e-14 | 2.66 |

The uncached DR row comes from a separate validation measurement; its latency ratio is descriptive, with no direct paired interval against the proposed method. No held-out measurement for that implementation is available.

| Split | Paired comparator | Mean saving ms | Paired Student-$t$ 95% CI ms |
|---|---|---:|---:|
| Validation | DR | 0.214456 | [0.162667, 0.266246] |
| Validation | OSQP-style ADMM | 0.330460 | [0.270355, 0.390566] |
| Held-out | DR | 0.228803 | [0.163300, 0.294306] |
| Held-out | OSQP-style ADMM | 0.343036 | [0.280900, 0.405171] |

All four paired intervals exclude zero. Against ADMM, both methods attain essentially machine-precision objective gaps, and the proposed score improvement is primarily computational. Against fixed-budget DR, objective accuracy and feasibility also improve. Proposed pass-mean timing standard deviations are $0.073492$ ms on validation and $0.077591$ ms on held-out evaluation; timing variability remains material.

Orthogonal mechanism isolation. We independently vary equality reduction and exact curvature with geometry caching disabled, then enable caching in an additional integrated cell. Each entry is the median of five randomized pass-average timings. Paired intervals are unavailable for this study.

| Equality reduction | Exact curvature | Geometry reuse | Validation ms | Held-out ms | Held-out maximum KKT residual |
|---|---|---|---:|---:|---:|
| Off | Off | Off | 0.841201 | 0.838694 | 2.15e-10 |
| Off | On | Off | 0.582822 | 0.578045 | 1.39e-15 |
| On | Off | Off | 0.654418 | 0.648229 | 2.15e-10 |
| On | On | Off | 0.449573 | 0.445208 | 1.78e-15 |
| On | On | On | 0.219981 | 0.217695 | 1.78e-15 |

On held-out problems, reduction alone improves the inexact-curvature uncached cell by $1.29\times$, while exact curvature alone improves the unreduced cell by $1.45\times$. Their combination gives $1.88\times$, and caching supplies another $2.05\times$. Reduction saves $0.132837$ ms with exact curvature and $0.190465$ ms without it; the interaction contrast is $-0.057628$ ms. Thus the effects are not identical additive savings. Every cell retains mean relative gap below $6.2\times10^{-16}$ and maximum violation below $2.2\times10^{-14}$, separating computational gains from a substantial loss of solution accuracy.

Five-component subtraction ablations. Each row removes one mechanism. Disabling Newton updates sets the iteration budget to zero and serves as an optimization control. CI half-widths below are computed from seven pass means; they are not intervals centered on the displayed median-based latency estimand.

| Removed component | Latency ms | Pass-mean 95% CI half-width ms | Composite score | Mean relative gap | Maximum KKT residual |
|---|---:|---:|---:|---:|---:|
| None | 0.224272 | 0.067968 | 0.011214 | 9.25e-17 | 4.03e-11 |
| Critical-cone directional treatment | 0.225762 | 0.066668 | 0.011288 | 9.25e-17 | 4.03e-11 |
| Active-face geometry reuse | 0.462783 | 0.010670 | 0.023139 | 1.20e-16 | 6.77e-15 |
| Equality-nullspace reduction | 0.342856 | 0.059075 | 0.017143 | 1.11e-16 | 6.83e-15 |
| Exact sinusoidal curvature | 0.366302 | 0.066068 | 0.018315 | 1.02e-16 | 8.46e-11 |
| Newton optimization updates | 0.057320 | 0.021324 | 0.367752 | 3.65e-1 | 6.08e-1 |

Geometry reuse, equality reduction, and exact curvature each improve observed latency point estimates. Removing Newton updates preserves near-feasibility but destroys optimization accuracy, demonstrating that the objective gains require actual optimization. Critical-cone treatment affects a separate directional helper, so its removal leaves forward results unchanged. This subtraction provides no empirical training or directional-accuracy benefit for that helper; its supported contribution is the mathematical distinction established in Theorem 4.

Multi-scale and boundary-structure stress. A separate controlled construction uses $\alpha=0.8$, $m=\lfloor d/3\rfloor$, prescribed boundary optima, and five randomized interleaved repetitions per scale. Larger cases introduce more active bounds and a structured equality row. These are boundary-structure stress cases, not an isolated demonstration of weak complementarity.

| Dimension | Equality-row scale span | Boundary stress | Proposed median ms | Proposed SD ms | ADMM median ms | ADMM SD ms | Speedup |
|---:|---:|---|---:|---:|---:|---:|---:|
| 24 | 100 | No | 0.1758 | 0.0642 | 0.5418 | 0.0126 | 3.083 |
| 64 | 100 | No | 0.3596 | 0.1356 | 0.6610 | 0.0206 | 1.838 |
| 96 | 1,000 | Yes | 0.5457 | 1.6642 | 1.0325 | 0.0527 | 1.892 |
| 128 | 10,000 | Yes | 0.8399 | 3.5687 | 1.4366 | 0.0725 | 1.710 |

| Dimension | Proposed relative gap | Absolute violation | Normalized violation | ADMM normalized violation | Proposed KKT residual |
|---:|---:|---:|---:|---:|---:|
| 24 | 0 | 4.26e-14 | 1.06e-15 | 8.11e-16 | 9.49e-12 |
| 64 | 1.05e-15 | 1.99e-13 | 1.83e-15 | 9.61e-16 | 5.36e-11 |
| 96 | 4.77e-15 | 1.59e-12 | 2.23e-14 | 6.30e-14 | 1.66e-11 |
| 128 | 0 | 4.14e-11 | 2.08e-13 | 1.99e-12 | 1.82e-12 |

Normalized equality residuals divide each row residual by $\max(\|M_i\|_2+|b_i|,1)$ and are combined with box violation. This separates changes of equality units from numerical feasibility. All proposed median latencies are below $1.1$ ms. However, the standard deviations at dimensions 96 and 128 substantially exceed the medians, so these results do not establish tail-latency guarantees. Dimension, row scaling, and boundary structure vary together; their individual causal effects are not isolated.

Backward computation. The derivative diagnostic uses $d=36$, 12 equalities, $\alpha=0.8$, and four prescribed strictly active coordinates. It compares a factor-available VJP with reconstruction of a tangent basis, reduced Hessian, Cholesky factor, and full Jacobian. VJP timing uses 50 measurements and reconstruction uses 15; reported values are medians without speedup confidence intervals.

| Backward diagnostic | Validation | Held-out evaluation run |
|---|---:|---:|
| Factor-available VJP ms | 0.00936 | 0.00925 |
| Full implicit-Jacobian reconstruction ms | 0.0529 | 0.0524 |
| Reconstruction/VJP latency ratio | 5.65 | 5.67 |
| Forty affine-projector applications with setup ms | 0.0852 | 0.0837 |
| Projector-diagnostic/VJP ratio | 9.10 | 9.05 |
| Relative VJP/full-Jacobian discrepancy | 4.38e-16 | 4.38e-16 |
| Mean central-difference relative error | 3.95e-10 | 3.95e-10 |
| Maximum central-difference relative error | 6.42e-10 | 6.42e-10 |

The factor-available benchmark uses factors constructed inside the diagnostic and general dense linear solves. It does not directly invoke the retained triangular-substitution interface of Algorithm 2, so the measured speedup supports factor availability but does not experimentally establish that interface's asymptotic complexity. Full Jacobian reconstruction also returns more information than one VJP. The forty-projector comparison omits the complete reverse recurrence through objective and bound projections; its ratio is not an end-to-end unrolled-backpropagation speedup.

Finite differences use ten random tangent directions and central step $10^{-6}$. Active-coordinate invariance under a small normal perturbation has error $1.67\times10^{-16}$. A 25-update coefficient-learning diagnostic with learning rate $0.45$ minimizes $\frac12\|y-y_{\mathrm{target}}\|_2^2$. Recorded loss decreases from $0.487766$ to $0.090137$, an 81.52% reduction, while maximum constraint violation remains $4.44\times10^{-15}$. No fallback update activates. Factors are reconstructed during these learning updates, so this experiment supports sensitivity and feasibility rather than retained-factor training throughput. The fixed derivative and learning problem is repeated across evaluation runs; identical values are not independent replication.

Numerical certificate coverage and memory. We define a numerical pass as absolute primal violation and measured stationarity residual both below $10^{-8}$.

| Quantity | Validation | Held-out |
|---|---:|---:|
| Primary numerical passes | 6/6 | 6/6 |
| Maximum primary stationarity residual | 4.03e-11 | 1.96e-10 |
| Maximum normalized primal violation | 1.86e-15 | 3.97e-15 |
| Auxiliary stress numerical passes | 4/4 | 4/4 |
| Traced current allocation, bytes | 30,954 | 31,000 |
| Traced peak allocation, bytes | 42,692 | 42,738 |

Stationarity is evaluated using estimated equality multipliers and bound-sign residuals. Memory tracing surrounds one initial solve and excludes complete process resident memory and long-lived cache growth.

Hardware reproducibility. Recorded measurements use Python 3.12.3, Linux 6.17.0-1014-nvidia-aarch64, glibc 2.39, and an ARM Cortex-X925 at 3.90 GHz, reported as pinned to one core. Solver inputs use NumPy double precision. Randomization seeds are 101 for validation timing, 202 for held-out timing, 303 for factorial ordering, 909 for stress construction and ordering, and 2026 for derivative diagnostics. NumPy version, BLAS vendor, BLAS thread count, and frequency-governor state are not recorded. The reported environment therefore specifies the measured platform without claiming exact cross-system latency reproducibility.

## Limitations
Our guarantees have distinct scopes. Uniform strong convexity ensures a unique global minimizer, but the selected finite-budget Newton iteration has no established global convergence guarantee. Feasibility preservation assumes feasible initialization, while the capped initialization procedure does not raise a final feasibility failure. Terminal factors retained after an iteration limit do not certify sensitivity of the exact optimizer.

The adjoint theorem assumes a locally stable optimal face. At weak activity, a fixed-face reverse-mode operator is not generally the nonlinear directional derivative; generalized-Jacobian membership additionally requires an adjacent realizable face. Critical-cone differentiation is mathematically characterized for Euclidean projection, but its finite numerical subproblem has an iteration cap and no dedicated measured weak-activity accuracy sweep. The forward subtraction experiment does not establish a learning benefit from this helper.

Empirical results concern small dense synthetic problems, six instances per primary split, and one CPU environment. Timing confidence intervals quantify repeated execution rather than broad task generalization. The scaling construction jointly changes dimension, row scales, and boundary structure, and its largest timing standard deviations preclude worst-case latency conclusions. Comparisons concern specific local splitting implementations rather than the published optimization-layer packages.

The backward microbenchmark uses available factors with dense solves rather than directly timing the retained triangular-substitution interface. Full Jacobian reconstruction has a larger output requirement than one VJP, the projector diagnostic is not complete unrolled differentiation, and coefficient learning reconstructs factors. Consequently, direct retained-interface speedups and full neural-training throughput remain unmeasured.

Complete QR construction can remain cubic, individual face caches are uncapped, and terminal state belongs to the most recent solve. Multiple outstanding or concurrent layer calls require explicit state ownership. Matrix-object caching also assumes that cached matrix contents remain valid. Single-call traced allocation does not bound process memory or long-run cache storage. Sparse and accelerator scaling, and derivatives with respect to equality matrices or right-hand sides are not evaluated.

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

## Appendix A: Complete Experimental Trajectory Manifest

| Experiment | mse |
|---|---|
| baseline_source | 0.090502407 |
| baseline_subset | 0.045309988 |
| baseline_full | 0.090462447 |
| initial_subset | 0.006258476 |
| initial_full | 0.011213589 |
| seed0_subset | 0.007985663 |
| seed0_full | 0.011794500 |
| evolved0_subset | 0.008778460 |
| evolved0_full | 0.012562525 |
| ablation0 | 0.011288113 |
| ablation1 | 0.023139134 |
| ablation2 | 0.017142800 |
| ablation3 | 0.018315083 |
| ablation4 | 0.367751919 |
| rebuttal0 | 0.010740751 |
| final_test | 0.011094873 |
