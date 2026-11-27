# Differentiable Hard-Constrained Neural Projection Layers: Fast Nullspace Active-Face Newton & Factor-Reused Implicit Differentiation

## 1. Problem Formulation and Uniform Strong Convexity on the Feasible Polytope

Differentiable optimization layers and hard-constrained neural architectures—including OptNet (Amos & Kolter, 2017), CvxpyLayers (Agrawal et al., 2019), OSQP operator splitting (Stellato et al., 2020), DC3 (Donti et al., 2021), hard-constrained PINNs (Lu et al., 2021; `arXiv:2601.06244`), and $\Pi\text{net}$ orthogonal projection layers (Grontas et al., 2026)—map unconstrained network predictions $q_\theta(x)$ onto context-dependent affine equalities and box inequalities:
$$\min_{y \in \mathbb{R}^d} J(y; x) = \frac{1}{2} y^\top Q(x) y + q_\theta(x)^\top y + \alpha \sum_{i=1}^d \sin(y_i) \quad \text{s.t.} \quad M(x) y = b(x), \;\; l(x) \le y \le u(x)$$
where $Q(x) \in \mathbb{S}_{++}^d$ has spectrum in $[0.3, 3.0]$, $M(x) \in \mathbb{R}^{m \times d}$ ($m = d/3$) has row scales spanning up to $10^4\times$, and $\mathcal{B} = [-0.25, 0.25]^d$.

**Theorem 1 (Uniform Strong Convexity & Unique Global KKT Optimum)**:
For $\alpha \in [0, 0.8]$ and $y \in \mathcal{B}$, the Hessian $H(y) = Q(x) - \alpha \operatorname{diag}(\sin(y))$ satisfies $\lambda_{\min}(H(y)) \ge 0.3 - 0.8 \sin(0.25) = 0.102077 > 0$. Thus $J(y; x)$ is $\mu$-strongly convex on $\mathcal{F}(x) = \{y \in \mathcal{B} : M(x)y = b(x)\}$, guaranteeing a unique global minimizer $y^\star(q)$ characterized by primal feasibility, complementary slackness, and dual-feasible stationarity $\nabla J(y^\star) + M^\top \lambda^\star + \nu^\star = 0$.

## 2. Fast Nullspace Active-Face Newton & Zero-Refactorization Implicit Differentiation

1. **Algebraic Nullspace Elimination ($y = y_p + V w$)**:
   - Orthonormal QR/SVD factorization of $M$ yields the minimum-norm particular solution $y_p = M^\dagger b$ and orthonormal nullspace basis $V \in \mathbb{R}^{d \times (d-m)}$ ($MV = 0, V^\top V = I$), enforcing $\|My - b\|_\infty \sim 10^{-15}$ at every step.
2. **Fast $O(|A| n \min(|A|, n))$ Active-Face Tangent Basis $Z$ & Cholesky Newton Steps**:
   - For active bound set $A \subset \{1, \dots, d\}$ of size $|A| \ll n = d-m$, the nullspace of the small $|A| \times n$ submatrix $V_A$ is computed via QR decomposition of $V_A^\top$ in $O(|A|^2 n)$ time (avoiding full $n \times n$ dense SVD), yielding $Z = V B_A \in \mathbb{R}^{d \times k_{\text{free}}}$ cached by active set signature.
   - Reduced Newton steps factorize the positive-definite reduced Hessian $R = Z^\top H(y) Z = L_R L_R^\top$ via Cholesky decomposition and solve $R \Delta u = -Z^\top \nabla J(y)$, achieving quadratic local convergence ($10^{-16}$ suboptimality and $10^{-14}$ constraint residual) while scaling cleanly to $d=128$ in $<1\text{ ms}$ (outperforming OSQP/ADMM in both speed and accuracy across all dimensions).
3. **Factor-Reused Implicit Differentiation Backward Pass (Theorem 2)**:
   - On the identified active face at $y^\star$, differentiating the reduced stationarity condition $Z^\top \nabla J(y^\star; q) = 0$ yields the exact Jacobian and adjoint:
     $$\frac{\partial y^\star}{\partial q} = -Z R^{-1} Z^\top = -Z L_R^{-\top} L_R^{-1} Z^\top, \qquad \frac{\partial \mathcal{L}}{\partial q} = -Z L_R^{-\top} \bigl(L_R^{-1} (Z^\top \frac{\partial \mathcal{L}}{\partial y^\star})\bigr).$$
   - By caching $(Z, L_R)$ from the terminal forward Newton step, the backward pass executes in $O(d k_{\text{free}} + k_{\text{free}}^2)$ time via two triangular solves with zero matrix reconstruction or refactorization, matching central finite differences to $\sim 10^{-10}$ relative error and accelerating backpropagation by $\sim 5\times$ over unrolled splitting while driving end-to-end differentiable layer training with $10^{-15}$ constraint satisfaction.

## 3. Comprehensive Evaluation in `eval.py`
`eval.py` reports composite score `mse`, `rs_mean`, `eq_cv_max`, `box_cv_max`, `cv_max`, minimized primal-dual KKT certificate `kkt_certified_residual_max`, Student-$t$ 95% CIs, head-to-head comparisons against `cached_dr_baseline` and `tuned_osqp_admm_baseline`, `implicit_differentiation_and_e2e_training` (backward latency, speedup, finite-difference agreement, 25-epoch training), and `scaling_and_degeneracy_stress_sweep` ($d \in \{24, 64, 96, 128\}$ with row scales up to $10^4\times$ and degenerate faces). Write the paper as a standalone ICLR research article without referencing internal workspace filenames.
