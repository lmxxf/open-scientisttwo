# Predictive Active-Face Newton Layers with Exact Reduced Curvature and Reusable Implicit Adjoints

## Abstract
We propose a differentiable optimization layer that couples equality-nullspace elimination, predictive active-face Newton steps, and reusable terminal curvature factors for affine-and-box-constrained nonlinear optimization. Although the objective contains a sinusoidal perturbation, our theoretical analysis establishes uniform strong convexity on the prescribed feasible box, uniqueness of the constrained optimum, quadratic local convergence on an identified face, and an exact stable-face sensitivity formula. Retaining the terminal tangent basis and Cholesky factor gives subsequent adjoint queries an $O(dk+k^2)$ cost without matrix reconstruction or refactorization, where $d$ is the output dimension and $k$ is the face tangent dimension. Our experiments evaluate both isolated mechanisms and the integrated algorithm. With prediction and caching disabled, exact nonlinear curvature reduces validation latency by 29.73%, while equality elimination reduces latency by 27.08% under exact curvature. The integrated method solves six held-out problems in 0.198641 ms on average, achieving 2.27× and 2.89× forward speedups over local cached splitting and OSQP-style ADMM comparators, respectively, with maximum feasibility violation $1.15\times10^{-14}$. A separate stress study reaches dimension 128 and equality-row scale span $10^4$, with 1.0497 ms latency and approximately 2,027× smaller violation than the ADMM comparator. These results support a unified geometry-and-curvature advance; the adjoint complexity guarantee is theoretical and implemented, while auxiliary derivative measurements do not directly benchmark retained-factor reuse.

## Introduction
Hard-constrained prediction requires more than a small objective value: the output must satisfy the constraints, and downstream learning requires a useful derivative of the constrained solution. Consider a network producing coefficients $q_\theta(x)$ for a constrained optimization layer. A finite splitting iteration can be inexpensive but leave residual infeasibility. A high-accuracy constrained solve can instead spend substantial work reconstructing linear systems, identifying bounds sequentially, and rebuilding derivative information after convergence. We address these costs together by making active-face geometry the shared representation of optimization and differentiation.

We propose predictive active-face Newton layers for objectives consisting of a positive-definite quadratic and a separable sinusoidal term, subject to affine equalities and coordinate bounds. Equality elimination restricts all optimization directions to the equality nullspace. A batch predictor estimates active bounds before the main feasible iteration. Within a face, exact reduced curvature supplies Newton directions, while ordered-face caching and inexpensive one-bound Householder updates reuse geometric information. At the returned point, the method explicitly constructs exact terminal curvature and retains its Cholesky factor for subsequent adjoints.

Our theoretical analysis establishes four complementary properties. First, the objective is uniformly strongly convex on the specified box despite its nonlinear perturbation, yielding a unique constrained optimum. Second, the geometric updates preserve the equality and active-coordinate tangent constraints in exact arithmetic. Third, Newton iteration converges quadratically once the correct face is identified and full steps remain feasible. Fourth, stable-face differentiation reduces to a positive-definite tangent system, and retained factors give an $O(dk+k^2)$ adjoint query. These statements distinguish global optimality of a certified stationary point from local convergence of the algorithm: we do not assert global convergence of the predictor or finite identification for every degenerate configuration.

Our experiments evaluate isolated mechanisms before interpreting integrated performance. With the entire cache-and-prediction package disabled, exact nonlinear curvature reduces latency from 0.412311 to 0.289726 ms, a 29.73% improvement. Under exact curvature and the same disabled package, equality elimination reduces latency from 0.397327 to 0.289726 ms, a 27.08% improvement. These controlled comparisons establish benefits beyond generic caching. They are complementary interventions, rather than additive contributions.

The integrated method achieves 0.198641 ms mean latency on an independently held-out six-instance workload, compared with 0.450600 ms for cached lifted splitting and 0.574959 ms for a local OSQP-style ADMM comparator. Its maximum feasibility violation is $1.15\times10^{-14}$ and its maximum measured stationarity residual is $1.55\times10^{-15}$. Separate scaling experiments evaluate dimensions through 128, equality-row scales through $10^4$, and increased active-coordinate coupling. Our contribution is the demonstrated coupling of predictive geometry and exact reduced curvature with a formally inexpensive adjoint representation. Neither implicit differentiation nor hard-constrained layers alone are new, and our empirical comparisons do not establish superiority over official implementations of the broader prior methods.

## Related Work
Differentiable constrained optimization methods differ in their objective family, feasibility mechanism, forward linear algebra, and derivative representation. We compare against all seven cited works along these axes.

**Orthogonal projection layers.** Grontas et al., [Pinet: Optimizing hard-constrained neural networks with orthogonal projection layers](https://arxiv.org/abs/2508.10480), introduce $\Pi$net, combining operator-splitting projections with implicit-function-based differentiation. Their formulation includes feasible optimization proxies and emphasizes single-instance and batched deployment, including a GPU-ready implementation. This is the closest conceptual predecessor. Our distinction is the nonlinear forward solve and its retained state: we predict active bounds, optimize using $Z_A^\top H(y)Z_A$, and retain the terminal reduced factor for adjoints. The operator-splitting representation uses a fixed projection mechanism; our representation changes with the identified face and current nonlinear curvature. Both approaches use implicit differentiation. Moreover, our map is generally an objective-minimizing constrained layer, becoming an ordinary Euclidean projection only in special quadratic cases. We do not evaluate the published $\Pi$net implementation or its batching advantage.

**Quadratic optimization layers.** Amos and Kolter, [OptNet: Differentiable Optimization as a Layer in Neural Networks](https://arxiv.org/abs/1703.00443), establish differentiable quadratic programming as a neural-network component. When $\alpha=0$, our layer also solves a quadratic program. For $\alpha\ne0$, its Hessian depends on the solution: $H(y)=Q-\alpha\operatorname{diag}(\sin y)$. A single quadratic program therefore does not express the complete nonlinear forward problem. Our derivative follows the same general principle of differentiating constrained optimality, but represents the sensitivity through the tangent-space Schur reduction $-Z(Z^\top HZ)^{-1}Z^\top$. Our contribution is a specialized forward solver and factor representation, rather than the introduction of differentiable optimization.

**General differentiable convex layers.** Agrawal et al., [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430), provide a general framework for constructing and differentiating convex optimization problems. We exchange that modeling breadth for explicit affine-and-box geometry and an analytical restricted-domain curvature bound. The sinusoidal expression is not asserted to satisfy a standard disciplined-convex modeling grammar. Its convexity here follows directly from the spectrum of $Q$ and the prescribed box. This restriction exposes an explicit tangent basis and positive-definite reduced Hessian, allowing subsequent adjoints to reuse a dense Cholesky factor.

**Operator splitting for quadratic programs.** Stellato et al., [OSQP: An Operator Splitting Solver for Quadratic Programs](https://arxiv.org/abs/1711.08013), motivate fixed-system reuse in splitting-based constrained optimization. Our local comparator uses an OSQP-style over-relaxed ADMM construction, a retained inverse of an augmented quadratic system, and explicit treatment of the sinusoidal gradient. Our method instead solves a changing reduced system, $Z_A^\top H(y)Z_A\delta u=-Z_A^\top g(y)$, on an evolving face. Thus our measurements contrast fixed-system splitting with exact reduced Newton correction; they are not official OSQP package benchmarks. At $\alpha=0$, the optimization family overlaps directly with quadratic programming, whereas the nonlinear cases require additional treatment.

**Completion and correction.** Donti et al., [DC3: A learning method for optimization with hard constraints](https://arxiv.org/abs/2104.12225), use completion and correction to construct constrained predictions. Equality elimination is algebraically related to completion: both reduce the independent degrees of freedom. Our layer, however, solves the remaining objective for every supplied coefficient vector and manages bounds using boundary truncation and multiplier-based release. Formally, its target is the solution of constrained stationarity, rather than a learned substitute for the optimization policy. We do not measure the application-level tradeoff between this per-instance solve and DC3.

**Hard-constrained physics-informed learning.** Lu et al., [Physics-informed neural networks with hard constraints for inverse design](https://arxiv.org/abs/2102.04626), and [Hard Constraint Projection in a Physics Informed Neural Network](https://arxiv.org/abs/2601.06244) study hard feasibility in physics-informed learning. Our formal setting is finite-dimensional: $My=b$, coordinate bounds, and analytically positive reduced curvature. For an explicit constraint-preserving parameterization $y=T_\theta(x)$, differentiation follows the parameterization; our sensitivity follows reduced stationarity, $Z^\top g(y,q)=0$. This distinguishes the mathematical object being differentiated, without implying a specific implementation for every physics-informed approach. We provide neither a PDE discretization nor an inverse-design benchmark, and the available descriptions do not support more detailed implementation-level comparisons with the latter work.

Across these precedents, hard feasibility, projection layers, active constraints, and implicit differentiation are established ingredients. Our specific advance is their geometry-and-curvature coupling: batch prediction reduces sequential face discovery, exact reduced curvature improves optimization even without caching, and terminal factor retention provides an exact stable-face adjoint with explicit query complexity. The first two effects have controlled empirical support; the third has a proof and implementation but no dedicated retained-factor runtime benchmark.

## Methodology
### Problem formulation and restricted-domain convexity

We study
$$y^\star(q)=\arg\min_{y\in\mathcal F}f(y;q),\qquad f(y;q)=\frac12y^\top Qy+q^\top y+\alpha\sum_{i=1}^d\sin y_i,$$
where $\mathcal F=\{y:My=b,\ l\le y\le u\}$. We assume $Q=Q^\top$, $0.3I\preceq Q\preceq3I$, $\alpha\in[0,0.8]$, $l_i=-0.25$, $u_i=0.25$, and a nonempty feasible set. Equalities are assumed consistent, including when their rows are dependent. Sensitivities are with respect to $q$ only; all other quantities are fixed. Coordinatewise differentiation gives
$$g(y)=Qy+q+\alpha\cos y,\qquad H(y)=Q-\alpha\operatorname{diag}(\sin y).$$

**Theorem 1 (uniform curvature and unique optimum).** On the prescribed box,
$$\mu I\preceq H(y)\preceq LI,\qquad \mu=0.3-0.8\sin(0.25)>0.10207,\qquad L=3+0.8\sin(0.25)<3.19793.$$
Consequently, the constrained minimizer exists and is unique. A feasible point satisfying $g(y)+M^\top\lambda+\nu=0$, with $\nu\in N_{[l,u]}(y)$, is that global minimizer.

**Proof.** For every vector $v$,
$$v^\top H(y)v=v^\top Qv-\alpha\sum_i\sin(y_i)v_i^2.$$
The bound $|\sin y_i|\le\sin(0.25)$ gives the asserted spectral inequalities. Integrating the lower Hessian bound along feasible line segments establishes strong convexity. Compactness gives existence, and strong convexity gives uniqueness. The normal-cone condition implies $g(y)^\top(z-y)\ge0$ for all feasible $z$; strong convexity then proves global optimality. $\square$

### Equality elimination and initialization

A complete QR factorization of $M^\top$ produces orthonormal equality and nullspace bases $E$ and $V$. Let $r=\operatorname{rank}(M)$ and $n=d-r$. In the full-row-rank case, write $M^\top=ET$ and solve $T^\top c=b$. A rank-deficient case uses an SVD to obtain the corresponding consistent coefficients and bases. Equality rank uses cutoff $\max(m,d)\epsilon\max(\max_i|T_{ii}|,1)$, with the analogous singular-value cutoff for SVD.

With $h=(l+u)/2$, initialize
$$y_0=h+E(c-E^\top h),\qquad MV=0,\quad V^\top V=I,$$
and precompute $H_0=V^\top QV$. If box violation exceeds $10^{-12}$, initialize $p=0$ and apply corrected alternating projections:
$$z=\operatorname{clip}(y+p,l,u),\qquad p\leftarrow p+y-z,\qquad y\leftarrow z+E(c-E^\top z).$$
This phase stops below $2\times10^{-13}$ box violation and raises an error after 20,000 unsuccessful iterations. Initial active coordinates lie within $10^{-11}$ of either bound.

### Active-face geometry and incremental updates

For an ordered active-coordinate list $A$, construct an orthonormal basis $B_A$ of $\ker(V_A)$ and set
$$Z_A=VB_A,\qquad R_{0,A}=B_A^\top H_0B_A.$$
Thus $MZ_A=0$ and $(Z_A)_A=0$. A complete QR of $V_A^\top$ supplies the complement; detected dependence triggers SVD reconstruction. The face-rank cutoff is $10^{-11}$ times the larger of the maximum factor magnitude and one. Geometry triples $(B_A,Z_A,R_{0,A})$ are cached within a solve under ordered-list signatures.

If a single independent bound $j$ is appended to the previously constructed face, let $a=V_jB_A$. Construct $u$ by adding $\operatorname{copysign}(\|a\|_2,a_1)$ to the first coordinate of $a$ and normalizing. With $P=I-2uu^\top$ and $D$ containing all identity columns except the first,
$$B_{A+j}=B_APD,\quad Z_{A+j}=Z_APD,\quad R_{0,A+j}=D^\top PR_{0,A}PD.$$
Matrix-vector products and outer products implement these transformations. Geometry is rebuilt if $\|a\|_2\le10^{-11}$ or the newly imposed tangent residual exceeds $10^{-11}$. Removing a bound retrieves cached geometry or constructs it afresh.

**Proposition 2 (tangent preservation).** In exact arithmetic, the independent-bound update preserves orthonormality and annihilates every old and new active row. A displacement $Z_A\delta u$ preserves equalities and active-coordinate values.

**Proof.** Since $P$ is orthogonal and $D^\top D=I$, the columns of $B_APD$ are orthonormal. Construction gives $aPD=0$, while old active rows already annihilate $B_A$. The displacement claim follows from $MZ_A=0$ and $(Z_A)_A=0$. $\square$

### Selected predictor and Newton iteration

The selected method enables equality reduction, exact nonlinear forward curvature, and active-face caching. Row equilibration and Armijo backtracking are disabled. The cache control also enables batch prediction and incremental geometry, so its empirical intervention is an acceleration bundle. The main iteration limit is 100.

Starting from the feasible initialization, the predictor performs three full equality-nullspace Newton steps. Strictly out-of-box coordinates define predicted active indices and bound sides. Up to eight expansion rounds follow. In each round, a least-squares correction with relative cutoff $10^{-11}$ moves predicted coordinates toward their selected bounds, followed by at most four tangent Newton steps. Tangent correction stops early below reduced-gradient infinity norm $2\times10^{-11}$. Newly violated coordinates are appended together. A prediction is accepted only if no coordinate violates its box by more than $10^{-11}$; otherwise the original feasible initialization is retained. Predictor trials may leave the box, so their curvature is not covered by Theorem 1, and a failed factorization raises an error rather than silently substituting another method.

At a main iterate, construct
$$R_A(y)=R_{0,A}-\alpha Z_A^\top\operatorname{diag}(\sin y)Z_A,$$
and compute
$$R_A(y)\delta u=-Z_A^\top g(y),\qquad s=Z_A\delta u$$
using Cholesky factorization. If the reduced-gradient infinity norm is at most $2\times10^{-11}$, set $s=0$. Otherwise, truncate the step at the first inactive bound:
$$\eta=\max\!\left(0,\min\!\left\{1,\min_{i\notin A:s_i>10^{-13}}\frac{u_i-y_i}{s_i},\min_{i\notin A:s_i<-10^{-13}}\frac{l_i-y_i}{s_i}\right\}\right).$$
The update is $y\leftarrow y+\eta s$. Append the first blocking coordinate if its ratio is at most one and agrees with the accepted length within $10^{-13}$.

When $\|s\|_\infty<10^{-10}$, solve $V_A^\top\lambda_A=-V^\top g$ by least squares with relative cutoff $10^{-11}$. A coordinate is classified as lower-active when within $10^{-8}$ of its lower bound. Signed multipliers are $-\lambda_i$ for lower bounds and $+\lambda_i$ for upper bounds. Release the most negative signed multiplier if it is below $-10^{-9}$; otherwise terminate. For dependent active rows, the least-squares multiplier representation need not find every possible sign-feasible representation.

```text
Algorithm 1: Predictive active-face Newton
Input: Q, q, M, b, lower bounds, upper bounds, alpha
Construct equality bases E, V and affine coefficients c
Project the box center onto the equalities
If required, run corrected box/affine feasibility projections
Precompute H0 = V^T Q V; initialize active list A and face cache
Perform three full equality-nullspace Newton predictor steps
For at most eight predictor rounds:
    Retrieve or construct predicted-face geometry
    Correct predicted bound coordinates by least squares
    Perform at most four full tangent Newton corrections
    Append all newly violated coordinates, if any
    If no violation exceeds tolerance, accept prediction
    Stop if no new coordinate can be appended
Retain the original initialization if no prediction is accepted
For at most 100 main iterations:
    Retrieve, construct, or incrementally update (B, Z, R0)
    Form reduced gradient and exact reduced Newton direction
    If the direction is below tolerance:
        Compute signed active multipliers
        Release the worst incorrect bound, or terminate
    Otherwise:
        Truncate at the first inactive bound and update y
        Append a blocking bound when reached
Reconstruct exact reduced curvature at the returned y
Factor R = L L^T and retain (Z, L)
Return y and convergence diagnostics
```

The mathematical description treats diagnostics as associated solver state; the output vector is the constrained prediction.

**Theorem 3 (local quadratic convergence).** Suppose the correct optimal face has been identified and iterates lie in a neighborhood where full Newton steps remain feasible. Before numerical stopping, face-coordinate errors satisfy
$$\|e_{t+1}\|_2\le\frac{\alpha}{2\mu}\|e_t\|_2^2.$$

**Proof.** The reduced Hessian is bounded below by $\mu I$. Since sine is one-Lipschitz and $Z$ has orthonormal columns,
$$\|Z^\top[H(y)-H(z)]Z\|_2\le\alpha\|y-z\|_2.$$
The integral remainder for the reduced gradient, followed by multiplication by the inverse reduced Hessian, gives the stated Newton error bound. Feasibility of full steps ensures that boundary truncation does not alter this local update. $\square$

### Exact terminal-factor differentiation

**Theorem 4 (stable-face sensitivity and reusable adjoint).** Assume the returned point is optimal and its active face remains locally constant under perturbations of $q$. With $R=Z^\top H(y^\star)Z=LL^\top$,
$$\frac{\partial y^\star}{\partial q}=-ZR^{-1}Z^\top,$$
and, for cotangent $v$,
$$\left(\frac{\partial y^\star}{\partial q}\right)^\top v=-ZL^{-\top}L^{-1}Z^\top v.$$

**Proof.** Parameterize the fixed face by $y=y_f+Zu$. Differentiating reduced stationarity $Z^\top g(y,q)=0$ yields $R\,du+Z^\top dq=0$. Theorem 1 makes $R$ positive definite. Therefore $dy=-ZR^{-1}Z^\top dq$, and symmetry gives the adjoint. For a zero-dimensional tangent the derivative is zero. $\square$

Independent active constraints and strict complementarity provide a sufficient regular setting for face stability. At face transitions, a classical derivative may fail to exist. Exact curvature is reconstructed and factorized at the returned point even in controls using approximate forward Hessians. Reuse therefore removes reconstruction and factorization from subsequent adjoint queries, while the terminal factorization remains part of forward work. After early stopping, retained curvature alone does not certify an exact optimum-map derivative or differentiate the finite iteration sequence.

```text
Algorithm 2: Retained-factor adjoint
Input: cotangent v; retained tangent basis Z and Cholesky factor L
a = Z^T v
Solve L t = a
Solve L^T w = t
Return -Z w
```

### Time and space complexity

Let $a=|A|$, $n=d-r$, and $k=\dim\ker(V_A)$. Complete dense equality QR, including its explicit orthogonal complement, costs $O(d^2m)$ for $m\le d$. Constructing $H_0$ costs $O(d^2n+dn^2)$. A fresh complete face QR costs $O(n^2a)$ for $a\le n$; constructing $Z$ and $R_0$ costs $O(dnk+n^2k+nk^2)$. These explicit-complement costs are larger than thin-factor costs. Rank-deficient reconstruction retains dense cubic worst-case work when dimensions grow together.

An accepted one-bound Householder update costs $O((n+d)k+k^2)$. With fixed geometry, a Newton iteration costs $O(d^2+dk^2+k^3)$, including dense gradient evaluation, curvature assembly, and factorization. Each feasibility iteration costs $O(dr+d)$ and the phase is capped at 20,000 iterations. Total forward work is setup plus all predictor, geometry, main-iteration, and terminal-factorization costs; it is not uniformly $O(dk^2)$.

An adjoint query costs $O(dk+k^2)$ time and $O(d+k)$ temporary storage. Retained float64 tangent and factor arrays occupy $8(dk+k^2)$ bytes. For $S$ cached geometries with tangent dimensions $k_s$, geometry payload is
$$8\sum_{s=1}^{S}\bigl(nk_s+dk_s+k_s^2\bigr)\ \text{bytes}.$$
Overall storage is $O(d^2+\sum_s[(n+d)k_s+k_s^2])$, excluding library workspaces and object overhead. The within-solve cache has no explicit size cap.

## Experiments
### Evaluation protocol

Our experiments evaluate synthetic problems with constructed reference optima and checked reference stationarity. Screening comprises four validation instances at dimensions 12 and 24. Primary validation comprises six instances at dimensions 24, 48, and 64, each paired with $\alpha\in\{0,0.8\}$. Additional validation and a separate six-instance held-out assessment evaluate the selected configuration. We distinguish these assessments from the fixed-construction scaling and auxiliary differentiation studies.

For instance $i$, define clipped relative suboptimality and violation as
$$r_i=\max\left(0,\frac{f(y_i)-f(y_i^\star)}{\max(|f(y_i^\star)|,1)}\right),$$
$$c_i=\max\{\|My_i-b\|_\infty,\|(l-y_i)_+\|_\infty,\|(y_i-u)_+\|_\infty\}.$$
The composite score, with latency in milliseconds, is
$$S=\bar r+50c_{\max}+0.05\bar t+1000c_{\max}\mathbf1\{c_{\max}>10^{-2}\}.$$
No reported assessment activates the severe-infeasibility term. Once gaps and violations reach floating-point scale, this score is effectively latency dominated. Clipping negative objective differences means a zero reported gap does not establish exact-arithmetic equality.

The stationarity diagnostic reconstructs equality multipliers and uses 30 sign-correction iterations. We report the resulting residual rather than claiming globally minimized certificate norms. Solver timing includes setup: five repeats yield a median for each instance, and instance medians are averaged. Comparators use three repeats. Cached lifted splitting uses 40 iterations and step size 0.08. The local OSQP-style ADMM comparator uses 80 iterations, penalty 2.0, relaxation 1.6, and an explicitly inverted fixed augmented system. Neither comparator is an official package implementation.

### Primary held-out comparison

| Method | Composite score ↓ | Mean relative gap ↓ | Maximum violation ↓ | Mean latency, ms ↓ |
|---|---:|---:|---:|---:|
| Cached lifted splitting | 0.11291649 | 2.5204e-3 | 1.7573e-3 | 0.450600 |
| OSQP-style ADMM | 0.02874796 | 5.3832e-16 | 2.0428e-14 | 0.574959 |
| Proposed | **0.00993207** | **2.4980e-16** | **1.1546e-14** | **0.198641** |

The proposed method achieves 2.2684× and 2.8945× forward speedups over cached splitting and ADMM, respectively. Composite scores improve by 91.20% and 65.45%. Maximum box violation is zero, and maximum measured stationarity residual is $1.5543\times10^{-15}$. Thus the method improves both latency and finite-iteration accuracy relative to splitting, while improving latency over the accurate ADMM control.

| Held-out dimension | $\alpha$ | Latency, ms | Equality residual | Stationarity residual |
|---:|---:|---:|---:|---:|
| 24 | 0.0 | 0.148657 | 3.553e-15 | 4.163e-16 |
| 24 | 0.8 | 0.170513 | 8.882e-16 | 3.348e-16 |
| 48 | 0.0 | 0.158993 | 4.441e-15 | 7.772e-16 |
| 48 | 0.8 | 0.218914 | 5.329e-15 | 1.124e-15 |
| 64 | 0.0 | 0.202338 | 7.105e-15 | 1.554e-15 |
| 64 | 0.8 | 0.292434 | 1.155e-14 | 1.110e-15 |

Nonlinear instances require more work at larger dimensions, while both regimes maintain small stationarity residuals.

### Cross-workload consistency and uncertainty

| Workload | Latency, ms | Pooled SD, ms | Nominal 95% timing interval, ms | Speedup over cached splitting | Speedup over ADMM |
|---|---:|---:|---:|---:|---:|
| Primary validation | 0.201990 | 0.052008 | [0.182572, 0.221408] | 2.1951× | 2.8326× |
| Additional validation | 0.197004 | 0.054451 | [0.176674, 0.217335] | 2.2594× | 2.9285× |
| Held-out | 0.198641 | 0.053705 | [0.178590, 0.218693] | 2.2684× | 2.8945× |

Nominal interval half-widths use $2.045s/\sqrt{30}$, pooling raw timings over heterogeneous instances. Their center is the mean of instance medians. They are descriptive timing summaries, not calibrated problem-distribution confidence intervals or paired speedup intervals. Additional validation obtains score 0.00985022 and maximum violation $8.88\times10^{-15}$. Screening obtains score 0.00748081 and latency 0.149616 ms.

For development context, uncached lifted splitting obtains score 0.09567968 and latency 1.573643 ms on primary validation. The selected method improves that score by 89.44% and is approximately 7.79× faster. Contemporaneous cached splitting and ADMM have scores 0.03916722 and 0.02860743; these stronger controls are more informative than the uncached baseline alone.

### Cache-free mechanism isolation

The cache control also enables prediction and incremental reuse. We disable that whole package to isolate curvature and equality reduction.

| Equality reduction | Exact nonlinear forward curvature | Cache and prediction | Latency, ms | Score | Maximum stationarity residual |
|---|---|---|---:|---:|---:|
| Enabled | Disabled | Disabled | 0.412311 | 0.02061554 | 3.435e-11 |
| Enabled | Enabled | Disabled | **0.289726** | **0.01448631** | 6.661e-15 |
| Disabled | Enabled | Disabled | 0.397327 | 0.01986634 | 6.606e-15 |
| Disabled | Disabled | Disabled | Unmeasured | Unmeasured | Unmeasured |

At fixed reduced coordinates, exact nonlinear curvature reduces latency by 29.73%, approximately 1.42×, and improves measured stationarity by roughly four orders of magnitude. At fixed exact curvature, equality reduction reduces latency by 27.08%. These comparisons establish mechanisms beyond caching. The joint-off cell is absent, so interaction effects are unidentified and the percentages should not be added.

### Component ablations

| Configuration | Latency, ms | Score | Maximum violation | Maximum stationarity residual |
|---|---:|---:|---:|---:|
| Integrated method | **0.201990** | **0.01009951** | 6.217e-15 | 1.110e-15 |
| Remove equality reduction | 0.282409 | 0.01412046 | 5.329e-15 | 6.661e-15 |
| Omit nonlinear forward curvature | 0.236529 | 0.01182644 | 6.217e-15 | 8.066e-11 |
| Remove cache and prediction | 0.289726 | 0.01448631 | 6.217e-15 | 6.661e-15 |
| Remove reduction, cache, and prediction | 0.397327 | 0.01986634 | 7.105e-15 | 6.606e-15 |
| Omit nonlinear curvature, cache, and prediction | 0.412311 | 0.02061554 | 5.329e-15 | 3.435e-11 |

Removing reduction, nonlinear curvature, or the acceleration bundle increases latency by 39.81%, 17.10%, and 43.44%, respectively. Removing reduction also disables prediction, making that intervention bundled. Among unreduced configurations, prediction is absent in both, and caching reduces latency from 0.397327 to 0.282409 ms, a 28.92% reduction. Prediction and incremental geometry are not individually isolated. Approximate-forward-curvature controls retain exact terminal curvature for differentiation.

Adding Armijo backtracking gives 0.237398 ms latency and score 0.01186989 without improving the primary objective or feasibility summaries. Combining row equilibration and Armijo gives 0.294481 ms and score 0.01472403; this does not isolate equilibration. Earlier development candidates have full-workload latencies 0.250398 and 0.326796 ms. These are method-development configurations, not independent random-seed replications.

### Multi-scale and difficult-face evaluation

The stress study uses $\alpha=0.8$, $m=\lfloor d/3\rfloor$, and row-scale spans through $10^4$. Larger cases increase active-coordinate counts and modify equality coupling. This stresses face geometry without establishing algebraic rank deficiency.

| $d$ | Row-scale span | Face regime | Proposed, ms | ADMM, ms | Speedup | Proposed violation | ADMM violation | Proposed stationarity residual |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 24 | 100 | Standard | 0.1901 | 0.5556 | 2.92× | 1.776e-14 | 4.086e-14 | 1.776e-15 |
| 64 | 100 | Standard | 0.3186 | 0.6887 | 2.16× | 4.263e-14 | 6.253e-13 | 6.217e-15 |
| 96 | 1,000 | Increased active coupling | 0.5967 | 1.0880 | 1.82× | 1.137e-12 | 1.241e-10 | 1.137e-13 |
| 128 | 10,000 | Increased active coupling | 1.0497 | 1.5070 | 1.44× | 1.000e-11 | 2.028e-8 | 2.501e-12 |

The stress construction uses a repeated fixed seed and single timing measurements per configuration. It is separate from independent held-out sampling. Relative objective gaps are zero after clipping except for $1.51\times10^{-16}$ at dimension 64. Absolute equality residual increases with row scaling: the largest case has approximately $10^{-11}$ violation, not $10^{-15}$. Nevertheless, its violation is approximately 2,027× smaller than ADMM's.

| Development configuration | $d=96$ latency, ms | $d=128$ latency, ms |
|---|---:|---:|
| Integrated method | **0.5848** | 1.0498 |
| Omit nonlinear forward curvature | 0.6192 | **1.0058** |
| Remove reduction | 2.7490 | 5.1925 |
| Remove cache and prediction | 2.5351 | 5.3023 |
| Remove reduction, cache, and prediction | 4.4844 | 10.8600 |
| Omit nonlinear curvature, cache, and prediction | 2.6213 | 5.1529 |

The acceleration bundle becomes particularly consequential at larger dimensions. Exact curvature is not uniformly fastest: its omission is slightly faster at dimension 128, while the selected configuration has better aggregate validation performance and more accurate stationarity.

| Scaling axis | Evidence |
|---|---|
| Optimization dimension | Four dimensions through 128 |
| Equality-row scale | Spans through 10,000 |
| Active-face structure | Two larger constructed cases |
| Objective regime | Quadratic and sinusoidal held-out instances |


### Auxiliary sensitivity and coefficient optimization

An auxiliary study uses $d=36$, $m=12$, $\alpha=0.8$, seed 2026, and central-difference step $10^{-6}$ along one tangent direction.

| Auxiliary measurement | Result |
|---|---:|
| Directional finite-difference relative error | 4.2964e-10 |
| Independently reconstructed implicit Jacobian time | 0.0820 ms |
| Repeated affine-propagation comparator time | 0.3465 ms |
| Recorded timing ratio | 4.22× |
| Direct coefficient-update steps | 25 |
| Learning rate | 0.45 |
| Initial target loss | 0.313248 |
| Final recorded target loss | 0.095658 |
| Recorded loss reduction | 69.46% |
| Maximum update-trajectory violation | 5.3291e-15 |

The derivative experiment independently constructs a tangent basis and full Jacobian rather than invoking the retained-factor adjoint. Its propagation comparator omits complete clipping and objective-gradient derivatives of splitting. Accordingly, the timing ratio does not measure retained-factor acceleration or fully differentiated splitting.

The learning demonstration updates $q$ directly for one synthetic target; it is not neural-network training over a dataset. Its auxiliary update substitutes $-\nabla_y\mathcal L$ when too few free coordinates remain for its tangent construction, and activation of this branch is not separately reported. These observations support local directional consistency and feasible coefficient optimization, not neural-training acceleration. The approximate splitting baseline records a smaller final target loss, 0.008271, together with violation $9.12\times10^{-4}$ and derivative error approximately 0.349; we therefore do not claim superior final learning loss.

### Memory footprint

The following analytical float64 payloads assume one face with upper-envelope tangent dimension $k=n$. Geometry payload is $8(nk+dk+k^2)$ bytes; terminal adjoint payload is $8(dk+k^2)$ bytes.

| $d$ | $m$ | $n$ | One geometry entry, bytes | Terminal adjoint arrays, bytes |
|---:|---:|---:|---:|---:|
| 24 | 8 | 16 | 7,168 | 5,120 |
| 64 | 21 | 43 | 51,600 | 36,808 |
| 96 | 32 | 64 | 114,688 | 81,920 |
| 128 | 42 | 86 | 206,400 | 147,232 |

These are array payloads, excluding problem storage, decomposition workspaces, object overhead, and additional faces. Tangent arrays can be shared between cached geometry and terminal state, so the columns should not be summed as independent allocations. No measured peak resident-memory result is available.

### Hardware and reproducibility

| Item | Specification or protocol |
|---|---|
| Documented reproduction architecture | 64-bit ARM aarch64 |
| Documented operating system | Linux 6.17.0-1014-nvidia; glibc 2.39 |
| Python | 3.12.3 |
| NumPy | 2.5.5 |
| Numerical precision | Float64 |
| BLAS threading | OPENBLAS_NUM_THREADS=1 |
| Factor solves | NumPy-linked LAPACK dpotrs with explicit integer-ABI selection |
| Proposed timing | Mean of five-repeat instance medians |
| Comparator timing | Mean of three-repeat instance medians |
| Stress timing | One measurement per configuration |
| Solver GPU computation | None |
| Historical processor assignment, affinity, and load | Unrecorded |

These environment details describe the documented reproduction host rather than complete historical provenance for every timing. The solver raises an import error if the required LAPACK symbol is unavailable.

## Limitations
Our independent held-out comparison comprises six synthetic instances. It provides evidence beyond validation selection but does not establish performance over large problem distributions, real applications, PDE inverse design, or GPU batches. Scaling uses repeated fixed constructions and single stress timings. Comparators are local cached splitting and OSQP-style ADMM implementations; we do not directly benchmark official implementations of $\Pi$net, OptNet, DC3, differentiable convex layers, or OSQP.

The theoretical guarantees have explicit domains. Uniform strong convexity holds inside the prescribed box, whereas predictor trials can leave it. We establish neither global convergence of the complete algorithm nor finite identification of every optimal face. Quadratic convergence assumes the correct face and feasible full steps. Numerical rank decisions, dependent active constraints, and least-squares multiplier representations can complicate release decisions. Equality consistency is assumed, and absolute equality residuals increase with row scaling.

The adjoint is exact for an optimum on a locally stable face. Classical differentiability may fail at face transitions. Exact terminal curvature after early stopping does not certify the optimum-map derivative or differentiate the finite algorithm. Only sensitivities with respect to $q$ are implemented. Retained state belongs to the most recent solve, requiring additional state management for interleaved or concurrent calls.

Attribution is incomplete because caching also enables prediction and incremental geometry. Cache-free comparisons support curvature and equality-elimination benefits, but the factorial design lacks the joint-off cell. Dense setup and factorization retain cubic worst-case work, and cached geometry has no explicit memory cap. Reported memory values are analytical payloads rather than measured process peaks.

Auxiliary derivative timings reconstruct a full Jacobian and do not benchmark the retained-factor adjoint. Their propagation comparator is incomplete, and coefficient optimization is not a neural-network training benchmark. The $O(dk+k^2)$ adjoint result therefore establishes computational structure rather than a measured practical backward speedup. Pooled timing intervals do not quantify paired-speedup uncertainty, and historical hardware provenance is incomplete.

## References
- Pinet: Optimizing hard-constrained neural networks with orthogonal projection layers. DOI: 10.48550/arXiv.2508.10480
- DC3: A learning method for optimization with hard constraints. DOI: 10.48550/arXiv.2104.12225
- OptNet: Differentiable Optimization as a Layer in Neural Networks. DOI: 10.48550/arXiv.1703.00443
- Differentiable Convex Optimization Layers. DOI: 10.48550/arXiv.1910.12430
- OSQP: An Operator Splitting Solver for Quadratic Programs. DOI: 10.48550/arXiv.1711.08013
- Physics-informed neural networks with hard constraints for inverse design. DOI: 10.48550/arXiv.2102.04626
- Hard Constraint Projection in a Physics Informed Neural Network. DOI: 10.48550/arXiv.2601.06244

## Appendix A: Complete Experimental Trajectory Manifest

| Experiment | mse |
|---|---|
| baseline_source | 0.095711802 |
| baseline_subset | 0.045580683 |
| baseline_full | 0.095679676 |
| initial_subset | 0.010260838 |
| initial_full | 0.016339800 |
| seed0_subset | 0.008783213 |
| seed0_full | 0.012519919 |
| evolved0_subset | 0.007480814 |
| evolved0_full | 0.010099508 |
| seed1_subset | 0.010513439 |
| seed1_full | 0.014724033 |
| evolved1_subset | 0.009091836 |
| evolved1_full | 0.011869889 |
| ablation0 | 0.014120457 |
| ablation1 | 0.011826439 |
| ablation2 | 0.014486305 |
| ablation3 | 0.019866343 |
| ablation4 | 0.020615539 |
| rebuttal0 | 0.009850224 |
| final_test | 0.009932074 |
