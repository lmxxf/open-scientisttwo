# Frontier-Annealed Covariance MAP-Elites for High-Dimensional Quality-Diversity Optimization

## Abstract
Quality-diversity optimization seeks a repertoire that is simultaneously high-performing and behaviorally diverse, but finite-budget search becomes difficult when decision vectors are high-dimensional, objective curvature is anisotropic, and descriptor-space frontiers are sparsely sampled. We propose Frontier-Annealed Covariance MAP-Elites, a unified black-box method combining rank-weighted diagonal covariance adaptation, persistent soft archive thresholds, ridge-estimated descriptor geometry, frontier-directed emission, and dimension-adaptive smoothing. Our theoretical analysis establishes four properties: soft thresholds retain an exact geometric amount of improvement pressure; the rank-weighted mean update is an empirical natural gradient; the covariance state remains uniformly positive definite; and regularized frontier steps have nonnegative first-order alignment with the target descriptor displacement, with a quantitative bound on null-space perturbation leakage. The method explicitly handles its first generation, insufficient archive support, unavailable Jacobians, full-archive refinement, and zero-norm tangent directions.

Across twelve controlled tasks with dimensions 24, 36, 48, and 64, the proposed method attains a normalized QD-Score of 0.7847, 100% archive coverage, mean elite quality of 0.7847, and worst-decile quality of 0.6854. It reduces QD loss by 0.4886 ± 0.0312 relative to Vanilla MAP-Elites, 0.2251 ± 0.0409 relative to CMA-ME, and 0.2118 ± 0.0445 relative to fixed-annealing CMA-MAE, where intervals are paired Student-t 95% confidence intervals over twelve tasks. Controlled subtraction attributes significant gains to archive annealing and frontier guidance, while the complete covariance-by-annealing factorial study demonstrates complementary exploitation and persistence effects. These results support frontier-aware annealed covariance search as an effective and theoretically interpretable approach to high-dimensional QD optimization.

## Introduction
Quality-diversity optimization replaces the single-solution objective of conventional optimization with a repertoire objective. Let $x\in[-2.5,2.5]^d$ be a decision vector, $f(x)\in(0,1]$ its quality, and $b(x)\in[-1,1]^2$ its behavioral descriptor. A grid $\mathcal G$ partitions descriptor space into $M$ cells, and the archive stores the best observed solution $x_e$ in every occupied cell $e$. Its normalized score is

$$\mathrm{QD}(\mathcal A)=\frac{1}{M}\sum_{e\in\mathcal G}q_e,\qquad q_e=\begin{cases}f(x_e),&e\text{ occupied},\\0,&e\text{ empty},\end{cases}$$

with loss $\mathcal L_{\mathrm{QD}}=1-\mathrm{QD}(\mathcal A)$. This objective couples two demands: discovering cells and improving their elites. The coupling is especially severe in high-dimensional decision spaces. Isotropic mutation expends most displacement in directions unrelated to descriptor progress, while strict archive-improvement ranking rapidly suppresses repeated visits that are necessary to climb quality within a cell. Covariance adaptation addresses anisotropic decision geometry, but it does not by itself identify which search directions approach unoccupied descriptor cells.

We propose Frontier-Annealed Covariance MAP-Elites, abbreviated FAC-MAE. FAC-MAE divides each post-initialization batch between a rank-weighted covariance emitter and a frontier emitter. The covariance emitter learns coordinate-wise scale and an evolution-path direction from archive-improvement ranks. The archive maintains a soft threshold per cell, updated by exponential annealing rather than immediate replacement; consequently, repeated high-quality visits retain a precisely characterized cumulative reward. The frontier emitter identifies occupied-to-empty grid edges, estimates the local descriptor Jacobian exclusively from previously evaluated decision–descriptor pairs, and applies a regularized minimum-norm displacement toward the neighboring cell center. A small regularized null-space perturbation preserves local decision-space exploration. Search scales as $\sqrt{32/d}$, preventing step energy from growing unchecked with dimension.

Our contributions are:

1. We introduce a single finite-budget algorithm integrating soft archive annealing, rank-weighted covariance adaptation, and learned frontier transport without gradients or access to latent benchmark parameters.
2. Our theoretical analysis proves the exact threshold-persistence law, the natural-gradient interpretation of the mean update, bounded positive definiteness of the covariance state, well-posedness of the local Jacobian estimator, and positive first-order descriptor alignment of frontier steps.
3. We specify all exceptional execution paths—including first-generation initialization, sparse-archive Gaussian completion, unavailable-Jacobian sampling, full-archive neighbor refinement, and zero-norm tangent handling—so the algorithm is mathematically and operationally complete.
4. Our experiments evaluate twelve tasks over $d\in\{24,36,48,64\}$ under 672 black-box evaluations per task. FAC-MAE obtains full coverage at every dimension and improves normalized QD-Score by 0.2118 ± 0.0445 over fixed-annealing CMA-MAE.
5. Complete factorial and subtractive studies isolate the mechanisms: frontier guidance contributes 0.0332 ± 0.0142 QD-Score, threshold annealing contributes 0.0472 ± 0.0149 in the integrated method, and their combination with covariance adaptation yields an 0.0800 ± 0.0198 advantage over search using neither covariance adaptation nor annealing.

## Related Work
CMA-ME combines covariance-matrix adaptation with behavior-space archives and established emitters as a central abstraction for continuous QD search [Fontaine et al., 2019, arXiv:1912.02400]. CMA-MAE subsequently replaces strict archive thresholds with annealed thresholds to retain objective pressure on flat, coarse, or repeatedly visited regions [Fontaine and Nikolaidis, 2022, arXiv:2205.10752]. FAC-MAE inherits this persistence principle but couples it to an explicit, locally learned map from decision displacement to descriptor displacement.

Discount Model Search addresses distortion in high-dimensional measure spaces by learning a continuous model of discount values rather than assigning one value to an entire histogram cell [Tjanaka et al., 2026, arXiv:2601.01082]. Its focus is continuous discount representation in high-dimensional descriptors. FAC-MAE instead considers a two-dimensional archive reached through high-dimensional decisions and learns local descriptor geometry for boundary transport. The two approaches therefore regularize different objects: discount values versus the decision-to-descriptor map.

Differentiable Quality Diversity and MEGA exploit first-order objective and measure gradients when these are available [Fontaine and Nikolaidis, 2021, arXiv:2106.03894]. Model-Based Quality-Diversity with Gradients combines learned dynamics, imagined perturbations, and gradients for sample-efficient exploration [Lim et al., 2022, arXiv:2211.12610]. FAC-MAE remains entirely black-box: its Jacobian is a ridge finite-difference estimate constructed from evaluated samples, and its covariance emitter uses rank information rather than analytic gradients.

Multi-Emitter MAP-Elites dynamically selects heterogeneous emitter types with a bandit mechanism [Cully, 2020, arXiv:2007.05352]. FAC-MAE uses a fixed, interpretable allocation between two complementary emitters and derives the frontier emitter directly from archive topology. Multi-Task Multi-Behavior MAP-Elites transfers information across related tasks while maintaining many task-conditioned repertoires [Anne and Mouret, 2023, arXiv:2305.01264]; FAC-MAE instead concentrates on one archive per task and adapts its search geometry within that task.

MAP-Elites has demonstrated competitive and robust locomotion-controller discovery relative to proximal policy optimization [Brych and Cully, 2020, arXiv:2009.08438]. QD has also produced varied model-rocket designs in simulation and physical launches, with domain-dependent differences among MAP-Elites, CMA-ME, and CMA-MAE [Schrum and Crosby, 2025, arXiv:2504.02177]. These studies motivate strong black-box baselines and emphasize that archive coverage alone is insufficient; our experiments consequently report score, coverage, mean quality, and tail quality.

Surrogate-Assisted MAP-Elites trains a deep predictor online to reduce expensive Hearthstone evaluations [Zhang et al., 2021, arXiv:2112.03534]. The YAHPO-derived QD suite similarly uses fast surrogate predictions to formulate machine-learning hyperparameter problems with behavioral features such as interpretability and resource use [Schneider et al., 2022, arXiv:2204.14061]. FAC-MAE does not predict objective quality. Its regression is confined to a local two-output descriptor Jacobian, thereby avoiding a global quality-modeling assumption.

Finally, the Venus exploration strategy study organizes a broad scientific portfolio across missions and international opportunities [VEXAG Exploration Strategy Study Analysis Workgroup, 2024, arXiv:2412.06830]. Although it is not a QD optimizer, it exemplifies the portfolio-level motivation behind repertoire construction: decision-makers benefit from diverse, high-value alternatives rather than a single nominal optimum.

Let $d$ denote decision dimension, $B$ the batch size, $M$ the number of archive cells, $E$ the number of emitters, $K$ the number of tasks, and $N$ the accumulated sample count. Costs below exclude the domain-specific function evaluation.

| Work | Mathematical formulation | Dominant update complexity | Principal guarantee or capability |
|---|---|---:|---|
| Discount Model Search | Continuous learned discount over measure space | $O(C_{\rm fit}(N)+BC_{\rm model})$ | Distinguishes nearby high-dimensional measures |
| CMA-MAE | Annealed cell thresholds plus CMA emitters | $O(E(Bd^2+d^3))$ for full covariance | Persistent objective pressure and exploration |
| CMA-ME | Improvement-ranked CMA emitters and archive | $O(E(Bd^2+d^3))$ for full covariance | Self-adapted continuous-space illumination |
| Differentiable QD / MEGA | Objective and measure gradients form an arborescence | $O(Bd+BC_{\rm grad})$ | Exploits exact first-order information |
| Multi-Emitter MAP-Elites | Bandit allocation over heterogeneous emitters | $O(\sum_e C_e+E\log E)$ | Adaptive emitter selection |
| Multi-Task Multi-Behavior MAP-Elites | Joint task and behavior repertoire | $O(KBd+KM)$ | Cross-task transfer and many solutions per task |
| QD model rockets | Archive search coupled to physical simulation | $O(BC_{\rm sim}+Bd)$ | Diverse manufacturable designs |
| MAP-Elites locomotion | Isotropic mutation and cellwise elitism | $O(Bd+B)$ | Robust repertoire construction without gradients |
| Surrogate-Assisted MAP-Elites | Online deep outcome model plus archive search | $O(C_{\rm train}(N)+BC_{\rm pred}+Bd)$ | Reduced use of expensive game simulation |
| Model-Based QD with Gradients | Learned dynamics, imagination, and gradients | $O(C_{\rm train}(N)+B(C_{\rm model}+C_{\rm grad}))$ | Simultaneous model-based exploration and exploitation |
| YAHPO-derived QD problems | QD over surrogate hyperparameter objectives | $O(BC_{\rm surrogate}+B)$ | Fast, reproducible QD benchmarking |
| Venus exploration strategy | Community portfolio synthesis | Community-study workflow without an iterative QD update | Mission-level diversity of scientific opportunities |
| FAC-MAE | Soft thresholds, rank moments, and local ridge frontier transport | $O(Bd+F(N\log N+d^3+kd^2)+M)$ | Threshold persistence, positive-definite search, and aligned frontier motion |

## Methodology
### Problem setting

For generation $t\in\{0,\ldots,G-1\}$, FAC-MAE evaluates a batch of $B$ decisions and stores the best quality observed in every cell of a $g\times g$ archive, so $M=g^2$. Each cell also carries a threshold $\tau_e$, initialized to zero. The full experiments use $G=28$, $B=24$, $g=12$, and hence 672 function evaluations per task.

Define the dimension scale, global step size, and frontier count by

$$s_d=\sqrt{32/d},\qquad \sigma=0.23s_d,\qquad F=\operatorname{round}(0.20B).$$

The covariance emitter uses $B-F$ samples. With $\mu=B/2$, its positive log weights are

$$\widetilde w_i=\log(\mu+1/2)-\log i,\qquad w_i=\frac{\widetilde w_i}{\sum_{j=1}^{\mu}\widetilde w_j},\qquad \mu_{\rm eff}=\left(\sum_iw_i^2\right)^{-1}.$$

The state comprises a mean $m_t\in\mathbb R^d$, diagonal variance $v_t\in[0.08,6]^d$, and evolution path $p_t\in\mathbb R^d$. Initially $m_0=0$, $v_0=\mathbf1$, and $p_0=0$.

### Rank-weighted covariance emission and archive annealing

At generation zero, all $B$ points are sampled uniformly from $[-1.15,1.15]^d$. After scoring them by $\Delta_i=f(x_i)-\tau_{e_i}$, the mean is initialized as the weighted average of the best $\mu$ points. The covariance and path deliberately remain unchanged during this initialization.

For $t>0$, the covariance samples are

$$x_i=m_t+\sigma y_i,\qquad y_i=D_t^{1/2}z_i+\mathbf1_{\{\|p_t\|>10^{-12}\}}\sqrt{\rho}\,\xi_i\frac{p_t}{\|p_t\|},$$

where $D_t=\operatorname{diag}(v_t)$, $z_i\sim\mathcal N(0,I_d)$, $\xi_i\sim\mathcal N(0,1)$, and $\rho=0.16$. Thus the conditional sampling covariance is $D_t+\rho p_tp_t^\top/\|p_t\|^2$ when the path is nonzero and $D_t$ otherwise.

Candidates are ranked by $\Delta_i$. If $s_i=(x_i-m_t)/\sigma$ and $\bar s=\sum_{i=1}^{\mu}w_is_{(i)}$, the state update is

$$m_{t+1}=m_t+0.75\sigma\bar s,$$

$$p_{t+1}=(1-c_p)p_t+\sqrt{c_p(2-c_p)\mu_{\rm eff}}\,\bar s,\qquad c_p=\frac{2}{d+5},$$

$$v_{t+1}=\operatorname{clip}_{[0.08,6]}\left((1-c_v)v_t+c_v\sum_{i=1}^{\mu}w_i(s_{(i)}\odot s_{(i)})+c_v\rho(p_{t+1}\odot p_{t+1})\right),$$

where $c_v=2/(d+7)$ and $\odot$ denotes componentwise multiplication.

Whenever a candidate has $f(x)>\tau_e$, its threshold is updated with $\alpha=0.03$:

$$\tau_e\leftarrow(1-\alpha)\tau_e+\alpha f(x).$$

Elite replacement is a separate operation and occurs exactly when $f(x)>q_e$.

```text
Algorithm 1: Frontier-Annealed Covariance MAP-Elites
Input: generations G, batch B, dimension d, grid width g
Initialize archive qualities to empty, thresholds τe ← 0
Initialize m ← 0, v ← 1, p ← 0, evaluated history H ← ∅
Set sd ← sqrt(32/d), σ ← 0.23 sd, F ← round(0.20B)
for t = 0,...,G−1 do
    if t = 0 then
        X ← B uniform samples from [−1.15,1.15]^d
        covariance sample count C ← B
    else
        C ← B−F
        draw C covariance samples using m, v, p, σ, and ρ = 0.16
        XF ← FrontierBatch(archive, H, F, σ, sd)
        X ← concatenation of covariance samples and XF
        if |X| < B then
            append Gaussian samples m + σ diag(sqrt(v)) z until |X| = B
        truncate X to B points
    end if
    replace nonfinite coordinates by zero and clip X to [−2.5,2.5]^d
    evaluate quality and descriptor for every x in X
    for every evaluated x assigned to cell e do
        Δ(x) ← f(x) − τe
        if f(x) > qe then replace the elite of e by x
        if f(x) > τe then τe ← 0.97τe + 0.03f(x)
        append the evaluated triple to H
    end for
    if t = 0 then
        m ← weighted mean of the best μ points by Δ
        leave v = 1 and p = 0
    else
        rank only the first C covariance samples by Δ
        update m, p, and v using the displayed equations
    end if
end for
return the archive
```

### Learned frontier transport

An occupied cell is a frontier source if at least one of its eight neighboring cells is empty. A cardinal edge receives multiplier 1.25 and a diagonal edge multiplier 0.75. The edge score is

$$a(e\rightarrow e')=c(e,e')\,[0.15+q_e].$$

Edges are sampled proportionally to these scores. Once every cell is occupied, FAC-MAE continues the same finite-difference mechanism by transporting a better elite toward a weaker cardinal neighbor. Such an edge has score

$$a(e\rightarrow e')=[0.02+q_e-q_{e'}][0.2+q_e],\qquad q_e>q_{e'}.$$

For a chosen source $(x_0,b_0)$, the $k=32$ historical descriptors nearest to $b_0$ are selected. Let $X$ contain their centered decision vectors as rows and $Y$ their centered descriptors. With descriptor distances $r_i$, scale $h=\max(10^{-3},\operatorname{median}_i\sqrt{r_i+10^{-12}})$, weights $W_{ii}=\exp[-r_i/(2h^2)]$, and ridge $\lambda=0.002$, the local Jacobian is

$$J=\left[(X^\top WX+\lambda I)^{-1}X^\top WY\right]^\top.$$

For target cell center $b_\star$, set $\delta=b_\star-b_0$ and

$$u=J^\top(JJ^\top+\lambda I_2)^{-1}\delta.$$

Its norm is capped at $1.5(0.75+0.10\sqrt d)$. For $z\sim\mathcal N(0,I_d)$, define

$$r=z-J^\top(JJ^\top+\lambda I_2)^{-1}Jz.$$

If $\|r\|>10^{-12}$, set $t=r/\|r\|$; otherwise set $t=0$. The emitted point is

$$x_{\rm front}=x_0+u+0.10s_dt.$$

If fewer than four archive cells are occupied, the frontier batch is empty and Algorithm 1 fills the missing positions from the covariance Gaussian. If fewer than three historical observations are available for a selected source, it instead emits $x_0+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma^2I_d)$.

```text
Algorithm 2: FrontierBatch
Input: archive, evaluated history H, requested count F, σ, sd
if F = 0 or fewer than four cells are occupied then return empty batch
construct occupied-to-empty eight-neighbor edges and their scores
if no such edge exists then
    construct better-to-weaker cardinal edges and their scores
end if
if no edge exists then return empty batch
for j = 1,...,F do
    sample an edge proportionally to its score
    obtain source elite (x0,b0) and target-cell center b★
    estimate the ridge Jacobian J from the 32 nearest historical descriptors
    if J is unavailable then append x0 + Normal(0,σ²I) and continue
    u ← Jᵀ(JJᵀ + 0.002I)⁻¹(b★−b0)
    cap ||u|| at 1.5(0.75 + 0.10 sqrt(d))
    r ← z − Jᵀ(JJᵀ + 0.002I)⁻¹Jz, z ∼ Normal(0,I)
    if ||r|| > 10⁻¹² then t ← r/||r|| else t ← 0
    append x0 + u + 0.10 sd t
end for
return the batch
```

### Theoretical analysis

**Theorem 1 — Exact persistence of soft archive improvement.** Suppose a cell begins at threshold $\tau_0$ and receives $n$ accepted visits of constant quality $q>\tau_0$. Under $\tau_{j+1}=(1-\alpha)\tau_j+\alpha q$ with $0<\alpha\leq1$,

$$\tau_j=q-(q-\tau_0)(1-\alpha)^j,$$

and its cumulative ranking advantage is

$$\sum_{j=0}^{n-1}(q-\tau_j)=(q-\tau_0)\frac{1-(1-\alpha)^n}{\alpha}.$$

Consequently, thresholds increase monotonically to $q$ without overshoot, and every finite visit retains positive improvement advantage when $\alpha<1$.

**Proof.** Subtract the update from $q$:

$$q-\tau_{j+1}=q-(1-\alpha)\tau_j-\alpha q=(1-\alpha)(q-\tau_j).$$

Starting from $q-\tau_0$, repeated substitution gives $q-\tau_j=(q-\tau_0)(1-\alpha)^j$, which yields the threshold formula. Summing this finite geometric series gives the cumulative expression. Since $0\leq1-\alpha<1$, the gap is nonnegative, decreases monotonically, and converges to zero. ∎

**Lemma 2 — Existence and uniqueness of the local descriptor estimator.** For any sample matrix $X$, nonnegative diagonal $W$, and $\lambda>0$, the estimator

$$A=(X^\top WX+\lambda I)^{-1}X^\top WY$$

is the unique minimizer of $\|W^{1/2}(XA-Y)\|_F^2+\lambda\|A\|_F^2$.

**Proof.** The objective is a quadratic with gradient $2X^\top W(XA-Y)+2\lambda A$. Setting it to zero gives $(X^\top WX+\lambda I)A=X^\top WY$. For every nonzero $h$, $h^\top(X^\top WX+\lambda I)h=\|W^{1/2}Xh\|^2+\lambda\|h\|^2>0$. The Hessian is therefore positive definite, the linear system is invertible, and its stationary point is the unique global minimizer. ∎

**Proposition 3 — Regularized frontier alignment.** Let $u=J^\top(JJ^\top+\lambda I)^{-1}\delta$ with $\lambda>0$. Then $u$ uniquely minimizes

$$\|Ju-\delta\|^2+\lambda\|u\|^2,$$

and

$$\delta^\top Ju=\delta^\top JJ^\top(JJ^\top+\lambda I)^{-1}\delta\geq0.$$

The inequality is strict whenever $\delta$ has a nonzero component in the range of $J$.

**Proof.** Differentiation yields $(J^\top J+\lambda I)u=J^\top\delta$. The coefficient matrix is positive definite, so the minimizer is unique. Multiplying the claimed expression by $J^\top J+\lambda I$ verifies the push-through identity $(J^\top J+\lambda I)^{-1}J^\top=J^\top(JJ^\top+\lambda I)^{-1}$. Let $JJ^\top=U\operatorname{diag}(s_i^2)U^\top$. Then

$$\delta^\top Ju=\sum_i\frac{s_i^2}{s_i^2+\lambda}(u_i^\top\delta)^2,$$

which is nonnegative and is positive exactly when at least one component with $s_i>0$ is present. ∎

**Proposition 4 — Bounded descriptor leakage of the stabilizing perturbation.** For

$$R_\lambda=I-J^\top(JJ^\top+\lambda I)^{-1}J,$$

one has $\|JR_\lambda z\|\leq\sqrt\lambda\|z\|/2$. Thus, before normalization, the auxiliary perturbation approaches the Jacobian null space as $\lambda\to0$.

**Proof.** With singular-value decomposition $J=U\Sigma V^\top$, the nonzero singular directions of $JR_\lambda$ have gains $\lambda s/(s^2+\lambda)$. For $s\geq0$, differentiation or $(s-\sqrt\lambda)^2\geq0$ gives $\lambda s/(s^2+\lambda)\leq\sqrt\lambda/2$. The spectral norm bound follows. The explicit zero-norm rule prevents division by zero when $R_\lambda z$ vanishes numerically. ∎

**Theorem 5 — Natural-gradient mean direction and covariance stability.** Condition on a generation's rank utilities $w_i$. For a Gaussian $\mathcal N(m,\sigma^2D)$ with diagonal $D\succ0$, the Fisher natural gradient of expected rank utility with respect to $m$ is proportional to $\mathbb E[w(X)(X-m)]$. The empirical FAC-MAE direction $\sigma\bar s=\sum_iw_i(x_{(i)}-m)$ is its rank-weighted Monte Carlo estimator. Furthermore, every maintained diagonal variance lies in $[0.08,6]$, and the actual sampling covariance is positive definite with minimum eigenvalue at least $0.08$.

**Proof.** The mean score function is $\nabla_m\log p(X)=\sigma^{-2}D^{-1}(X-m)$, while the mean block of the Fisher information is $\mathcal I_m=\sigma^{-2}D^{-1}$. Hence

$$\mathcal I_m^{-1}\mathbb E[w(X)\nabla_m\log p(X)]=\mathbb E[w(X)(X-m)].$$

Replacing the expectation by the weighted selected samples gives $\sigma\bar s$. The diagonal update before clipping is a sum of three nonnegative vectors: $(1-c_v)v_t$, a weighted second moment, and a squared-path term. Coordinatewise clipping then places every entry in $[0.08,6]$. When $p=0$, the sampling covariance is $D$; otherwise it is $D+\rho pp^\top/\|p\|^2$. The latter adds a positive-semidefinite matrix to $D$, so its smallest eigenvalue is at least $\min_jv_j\geq0.08$. Because ranking depends only on order, the update is also invariant to strictly increasing transformations of the improvement scores; the diagonal representation is equivariant to signed coordinate permutations. ∎

### Complexity

Let $N_t=tB$ be the history size and $F=\operatorname{round}(0.20B)$. Constructing archive edges costs $O(M)$. Each frontier point computes descriptor distances and an ordering in $O(N_t\log N_t)$, forms a $d\times d$ ridge system in $O(kd^2)$, and solves it in $O(d^3)$. The remaining sampling, ranking, archive, and covariance operations cost $O(Bd+B\log B+M)$. Excluding black-box function calls, one generation therefore costs

$$O\!\left(Bd+B\log B+M+F(N_t\log N_t+kd^2+d^3)\right).$$

Across $G$ generations, the bound is

$$O\!\left(GBd+GM+G F(kd^2+d^3)+FBG^2\log(BG)\right).$$

The archive requires $O(Md)$ space, retained observations require $O(GBd)$, and the ridge system requires $O(d^2)$ working memory, giving $O(Md+GBd+d^2)$ total space.

## Experiments
### Experimental protocol

We study twelve deterministic black-box tasks: three independent instances at each $d\in\{24,36,48,64\}$. Every task uses a $12\times12$ descriptor grid, 28 generations, a batch of 24, and 672 total evaluations. Quality is generated by a rotated anisotropic quadratic with eigenvalues geometrically spanning 0.4 to 4.5, augmented by a multimodal periodic term of amplitude 0.15. Descriptors are nonlinear two-dimensional projections bounded to $[-1,1]^2$. FAC-MAE observes only evaluated decisions, qualities, and descriptors.

We compare with Vanilla MAP-Elites using isotropic mutation scale 0.35, diagonal CMA-ME using strict thresholds, and diagonal fixed-annealing CMA-MAE with threshold rate 0.15. Each stored elite is re-executed through the black-box task before aggregation. We report normalized QD-Score, coverage, mean quality over occupied cells, and the mean quality of the worst 10% of occupied elites. Paired confidence intervals use the twelve taskwise differences, $\bar d\pm t_{0.975,11}s_d/\sqrt{12}$ with $t_{0.975,11}=2.201$.

### Results by dimension

| Method | $d$ | QD-Score | Coverage (%) | Mean elite | Worst 10% |
|---|---:|---:|---:|---:|---:|
| Vanilla MAP-Elites | 24 | 0.3582 | 94.44 | 0.3792 | 0.2186 |
| CMA-ME | 24 | 0.6430 | 98.84 | 0.6505 | 0.5484 |
| Fixed-Annealing CMA-MAE | 24 | 0.6647 | 98.15 | 0.6772 | 0.5628 |
| FAC-MAE | 24 | **0.7814** | **100.00** | **0.7814** | **0.6727** |
| Vanilla MAP-Elites | 36 | 0.2984 | 88.66 | 0.3368 | 0.1691 |
| CMA-ME | 36 | 0.5712 | 95.37 | 0.5994 | 0.4931 |
| Fixed-Annealing CMA-MAE | 36 | 0.5871 | 96.53 | 0.6079 | 0.5058 |
| FAC-MAE | 36 | **0.7866** | **100.00** | **0.7866** | **0.6875** |
| Vanilla MAP-Elites | 48 | 0.2792 | 84.95 | 0.3293 | 0.1892 |
| CMA-ME | 48 | 0.5475 | 93.06 | 0.5883 | 0.4743 |
| Fixed-Annealing CMA-MAE | 48 | 0.5653 | 93.29 | 0.6060 | 0.4967 |
| FAC-MAE | 48 | **0.7978** | **100.00** | **0.7978** | **0.7062** |
| Vanilla MAP-Elites | 64 | 0.2486 | 83.80 | 0.2967 | 0.1619 |
| CMA-ME | 64 | 0.4768 | 85.42 | 0.5583 | 0.4595 |
| Fixed-Annealing CMA-MAE | 64 | 0.4745 | 85.42 | 0.5557 | 0.4490 |
| FAC-MAE | 64 | **0.7729** | **100.00** | **0.7729** | **0.6754** |

FAC-MAE fills all 144 cells at every dimension. Its QD-Score varies by only 0.0249 between the strongest and weakest dimensional aggregates, whereas Vanilla MAP-Elites loses 0.1096 from $d=24$ to $d=64$. At $d=64$, FAC-MAE exceeds the strongest baseline QD-Score by 0.2962 and worst-decile quality by 0.2158. The simultaneous tail-quality and coverage improvements show that the score gain is distributed through the archive rather than concentrated in a small group of elites.

### Aggregate benchmark comparison

| Method | QD loss | QD-Score | Coverage (%) | Mean elite | Worst 10% | QD gain, paired 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla MAP-Elites | 0.7039 | 0.2961 | 87.96 | 0.3355 | 0.1847 | — |
| CMA-ME | 0.4404 | 0.5596 | 93.17 | 0.5991 | 0.4938 | — |
| Fixed-Annealing CMA-MAE | 0.4271 | 0.5729 | 93.34 | 0.6117 | 0.5036 | — |
| FAC-MAE | **0.2153** | **0.7847** | **100.00** | **0.7847** | **0.6854** | — |
| FAC-MAE minus Vanilla MAP-Elites | — | — | — | — | — | **+0.4886 ± 0.0312** |
| FAC-MAE minus CMA-ME | — | — | — | — | — | **+0.2251 ± 0.0409** |
| FAC-MAE minus Fixed-Annealing CMA-MAE | — | — | — | — | — | **+0.2118 ± 0.0445** |

All three paired intervals are strictly positive. Their lower endpoints are 0.4574, 0.1842, and 0.1673, respectively. Relative to the baseline losses, FAC-MAE reduces QD loss by 69.4% against Vanilla MAP-Elites, 51.1% against CMA-ME, and 49.6% against fixed-annealing CMA-MAE. Compared with the strongest aggregate baseline, it also raises coverage by 6.66 percentage points, mean elite quality by 0.1730, and worst-decile quality by 0.1819.

### Component subtraction

Each row below changes exactly one active mechanism while retaining all other settings.

| Condition | QD loss | QD-Score | Coverage (%) | Mean elite | Worst 10% | QD drop, paired 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| FAC-MAE | **0.2153** | **0.7847** | **100.00** | **0.7847** | **0.6854** | — |
| Without frontier guidance | 0.2485 | 0.7515 | 96.35 | 0.7797 | 0.6435 | 0.0332 ± 0.0142 |
| Without covariance adaptation | 0.2231 | 0.7769 | 100.00 | 0.7769 | 0.6705 | 0.0078 ± 0.0106 |
| Without threshold annealing | 0.2625 | 0.7375 | 100.00 | 0.7375 | 0.6464 | 0.0472 ± 0.0149 |
| Without dimension-adaptive scaling | 0.2249 | 0.7751 | 100.00 | 0.7751 | 0.6646 | 0.0096 ± 0.0094 |

Frontier guidance is the only component whose removal reduces both score and coverage: it contributes 3.65 coverage points, 0.0419 worst-decile quality, and a significant 0.0332 QD-Score. Threshold annealing produces the largest isolated score contribution, 0.0472, with a confidence interval bounded away from zero. Dimension-adaptive scaling contributes 0.0096 with a narrowly positive lower endpoint. Covariance adaptation has a positive mean contribution of 0.0078 and improves worst-decile quality by 0.0149, although its isolated score interval crosses zero; the factorial study below shows that its effect depends on the threshold regime.

### Complete covariance-by-annealing factorial study

To isolate covariance and annealing without directed frontier transport, we evaluate all four combinations while keeping initialization, batch allocation, archive rules, and dimensional scaling fixed.

| Covariance adaptation | Soft threshold annealing | QD loss | QD-Score | Coverage (%) |
|---|---|---:|---:|---:|
| On | On | **0.2485** | **0.7515** | 96.35 |
| On | Off | 0.2878 | 0.7122 | 96.99 |
| Off | On | 0.2504 | 0.7496 | 96.01 |
| Off | Off | 0.2953 | 0.7047 | 96.70 |

Averaging over covariance states, annealing contributes 0.0421 QD-Score. Averaging over threshold states, covariance adaptation contributes 0.0047. Covariance contributes 0.0075 when annealing is absent and 0.0020 when it is present, giving an interaction contrast of −0.0056: annealing supplies most of the persistence benefit, while covariance is most useful under strict thresholds. Restoring frontier guidance to the strongest factorial cell adds 0.0332 ± 0.0142 and reaches the integrated score of 0.7847. Relative to the cell using neither mechanism, the complete method gains 0.0800 ± 0.0198 QD-Score; this paired interval is strictly positive.

## Limitations
The present evidence concerns deterministic synthetic objectives, two-dimensional descriptors, a $12\times12$ grid, dimensions 24–64, and a budget of 672 evaluations. Performance on noisy objectives, discontinuous descriptors, higher-dimensional behavior spaces, substantially larger archives, or nonstationary tasks remains to be established.

FAC-MAE maintains a diagonal covariance plus one evolution-path direction. This representation is computationally lighter than a full covariance but is invariant only to signed coordinate permutations, not arbitrary rotations. The natural-gradient result applies exactly to the rank-weighted mean moment and to the base moment interpretation of covariance adaptation; the added path term is a positive-semidefinite temporal-memory correction rather than a full affine-invariant natural-gradient update.

The frontier guarantees are local and first-order. Proposition 3 proves alignment under the estimated Jacobian, not guaranteed entry into the target cell under an arbitrarily nonlinear descriptor map. Ridge regularization bounds instability but introduces bias, and normalization of the residual direction can amplify small residuals above the numerical tolerance. The explicit zero-residual rule avoids numerical division by zero but cannot eliminate model error.

The local Jacobian solve costs $O(d^3)$ per frontier proposal and retained history grows as $O(GBd)$. These costs are modest at the evaluated dimensions but may dominate at thousands of parameters or much larger budgets; low-rank solvers, incremental sufficient statistics, and bounded replay buffers are natural extensions. Finally, covariance adaptation has a positive but statistically unresolved isolated contribution in the subtraction study, whereas threshold annealing and frontier guidance have confidence intervals bounded away from zero. Broader seed counts are needed to characterize the covariance interaction more precisely.

## References
- Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces. DOI: 10.48550/arXiv.2601.01082
- Covariance Matrix Adaptation MAP-Annealing. DOI: 10.48550/arXiv.2205.10752
- Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space. DOI: 10.48550/arXiv.1912.02400
- Differentiable Quality Diversity. DOI: 10.48550/arXiv.2106.03894
- Multi-Emitter MAP-Elites: Improving quality, diversity and convergence speed with heterogeneous sets of emitters. DOI: 10.48550/arXiv.2007.05352
- Multi-Task Multi-Behavior MAP-Elites. DOI: 10.48550/arXiv.2305.01264
- A Quality Diversity Approach to Evolving Model Rockets. DOI: 10.48550/arXiv.2504.02177
- Competitiveness of MAP-Elites against Proximal Policy Optimization on locomotion tasks in deterministic simulations. DOI: 10.48550/arXiv.2009.08438
- Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding. DOI: 10.48550/arXiv.2112.03534
- Efficient Exploration using Model-Based Quality-Diversity with Gradients. DOI: 10.48550/arXiv.2211.12610
- A Collection of Quality Diversity Optimization Problems Derived from Hyperparameter Optimization of Machine Learning Models. DOI: 10.48550/arXiv.2204.14061
- A New Strategy for the Exploration of Venus. DOI: 10.48550/arXiv.2412.06830

## Appendix A: Complete Experimental Trajectory Manifest

| Experiment | mse |
|---|---|
| baseline_source | 0.712111014 |
| baseline_subset | 0.674305967 |
| baseline_full | 0.712111014 |
| initial_subset | 0.254810539 |
| initial_full | 0.243970300 |
| seed0_eng5 | 0.319712909 |
| seed0_full | 0.338438296 |
| evolved0_subset | 0.309768658 |
| evolved0_full | 0.297698941 |
| ablation0 | 0.244228101 |
| ablation1 | 0.311766559 |
| ablation2 | 0.240601718 |
| ablation3 | 0.240795742 |
| ablation4 | 0.267098084 |
| pruned | 0.241194496 |
| pruned_ablation0 | 0.267098084 |
| pruned_ablation1 | 0.254322887 |
| pruned_ablation2 | 0.236431391 |
| pruned_ablation3 | 0.283870423 |
| pruned_ablation4 | 0.244689084 |
| pruned_r1 | 0.236431391 |
| pruned_r1_ablation0 | 0.267098084 |
| pruned_r1_ablation1 | 0.234892729 |
| pruned_r1_ablation2 | 0.261284531 |
| pruned_r1_ablation3 | 0.225702079 |
| pruned_r1_ablation4 | 0.246184784 |
| pruned_r2 | 0.225702079 |
| rebuttal0 | 0.233845998 |
| final_test | 0.215320679 |
