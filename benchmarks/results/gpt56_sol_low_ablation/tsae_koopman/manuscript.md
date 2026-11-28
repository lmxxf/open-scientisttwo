# Koopman–Laplacian Sparse Autoencoders with Joint Dictionary Refinement for Temporally Coherent Feature Recovery

## Abstract
Sparse autoencoders identify interpretable directions in high-dimensional representations, but framewise inference ignores sequential structure and can produce unstable feature supports. We propose a Koopman–Laplacian sparse autoencoder that combines nonnegative Top-6 coding, exact path-graph smoothing, bidirectional Koopman refinement, and column-normalized dictionary adaptation. Our theoretical analysis establishes the unique ridge-regularized Koopman estimator, proves the exactness and contraction properties of the temporal proximal operator, bounds the stability of bidirectional refinement, and derives explicit time and space complexity. On eight held-out synthetic trajectories with 32 observed dimensions and 64 latent atoms, our method obtains a composite loss of 0.2543, improving over framewise ISTA, TopK sparse autoencoding, and a Laplacian temporal autoencoder by 0.0127, 0.0188, and 0.0479, respectively; all paired Student-t 95% confidence intervals exclude zero. Controlled factorial experiments isolate complementary gains from Laplacian smoothness and Koopman dynamics, while dictionary refinement supplies a further 0.0106 absolute improvement. Removing hard support projection increases the validation loss from 0.2412 to 0.8945, confirming that exact sparsity is essential rather than cosmetic. These results establish a unified algorithmic advance: local temporal denoising, learned latent transport, and dictionary correction cooperate while retaining an explicit finite support budget.

## Introduction
## 1. Introduction

Sparse autoencoders (SAEs) seek an overcomplete, interpretable representation of observations $x_t\in\mathbb{R}^d$ through nonnegative codes $z_t\in\mathbb{R}_+^m$ and a unit-column dictionary $D\in\mathbb{R}^{d\times m}$, with $x_t\approx Dz_t$ and $\|z_t\|_0\leq q$. Their appeal is structural: a small active set can separate features that are superposed in the original coordinates. Yet conventional SAE inference treats successive observations independently. Under noise, this produces support flicker even when the underlying process changes smoothly or follows predictable latent dynamics.

Pure smoothness is not sufficient. A penalty on $z_{t+1}-z_t$ assumes that persistence is the only admissible evolution, whereas meaningful features may rotate, transfer mass, decay, or activate predictably. Conversely, a learned dynamical operator cannot repair a misaligned dictionary by itself. We therefore propose a joint method with four active mechanisms: nonnegative Top-6 sparse inference, exact path-Laplacian smoothing, a ridge-regularized Koopman operator used from both temporal directions, and four rounds of normalized dictionary refinement. The implementation deliberately omits inertial acceleration: every sparse-coding iteration is a direct projected proximal-gradient step, which makes the stated algorithm coincide with the executed computation.

Our theoretical contribution is a self-contained analysis of the operators actually used. We prove that ridge fitting has a unique closed-form Koopman solution with the correct orientation for row-wise latent trajectories; that spectral clipping enforces an explicit propagation bound; that the tridiagonal Laplacian solve is an exact proximal map and contracts every nonconstant temporal Fourier mode; and that each bidirectional linear solve is well posed and nonexpansive up to an explicit condition-number factor. We also derive end-to-end complexity bounds rather than treating temporal inference as a black box.

Our experiments evaluate reconstruction, support identification, latent rollout, and sparsity jointly. The principal score is
$$
\mathcal{L}_{\mathrm{comp}}=\mathcal{E}_{\mathrm{rec}}+0.35(1-F_1)+0.25\mathcal{E}_{\mathrm{roll}}+0.15\mathcal{V}_{\mathrm{sparse}},
$$
where lower is better. On held-out trajectories, the proposed method reaches $0.2543$ and significantly outperforms all three reference methods under paired 95% confidence intervals. The full $2\times2$ temporal factorial shows that Laplacian smoothing and Koopman transport are individually beneficial and jointly strongest. Dictionary refinement improves the joint temporal model from $0.2648$ to $0.2543$, while hard support projection eliminates sparsity violations and is indispensable to the composite objective.

Our contributions are:

1. We propose a single, executable Koopman–Laplacian SAE integrating exact sparsity, local smoothness, learned transport, and dictionary refinement.
2. We establish closed-form, stability, contraction, and complexity guarantees for every continuous temporal subproblem.
3. We provide paired benchmark comparisons, factorial mechanism isolation, component-subtraction ablations, and robustness measurements across standard and elevated noise/sparsity conditions.

## Related Work
## 2. Related Work

### Sparse representation learning

Cunningham et al. (2023) use sparse autoencoders to recover interpretable, approximately monosemantic directions from language-model activations, directly addressing polysemanticity and superposition. Their formulation is framewise and does not model temporal transport. Gao et al. (2024) introduce directly controlled $k$-sparse autoencoders, scaling laws, and feature-quality metrics; their hard cardinality control motivates our explicit Top-6 projection, but their inference remains independent across positions. Lieberum et al. (2024) release JumpReLU autoencoders across Gemma 2 layers, making large-scale SAE analysis accessible, but do not impose sequential latent dynamics.

Classical dictionary-learning theory also informs our design. Gribonval et al. (2014) establish local recovery behavior for sparse coding under overcompleteness, noise, and outliers, highlighting why adaptation of a perturbed dictionary can be statistically meaningful. Ghanem and Ahuja (2011) jointly learn sparse representations and discriminative predictors in a probabilistic framework. In contrast, our auxiliary structure is unsupervised temporal prediction rather than classification, and our dictionary step is a normalized residual-preconditioned update.

### Temporal sparse autoencoders

Bhalla et al. (2025) introduce temporal sparse autoencoders with an adjacent-token contrastive objective, demonstrating smoother and more semantically coherent concepts. Their temporal prior encourages consistency but does not explicitly estimate a latent transition operator. Dokme and Vishwanath (2026) show that framewise TopK selection can destabilize video features and propose spatio-temporal contrastive objectives and hierarchical grouping. Their approach targets representation learning across space and time, whereas ours solves an explicit path-graph proximal problem and learns a linear latent transport map. Yeo et al. (2026) represent clinical concepts using temporal Gaussian islands and sparse variable compositions, providing structured patient-level attribution. Our formulation instead handles general trajectories without predefined island shapes.

### Koopman representation learning

Lusch et al. (2017) learn coordinate transformations in which nonlinear dynamics become approximately linear, establishing deep Koopman embeddings as a practical route to prediction and control. Our method shares the linear latent-evolution principle but couples it to a fixed-cardinality nonnegative code and a refined overcomplete dictionary. Li et al. (2026) use sparse supports to identify local dynamical regimes in multibasin systems. Their analysis emphasizes multiple local regimes; our method estimates one globally regularized operator and concentrates on stable sparse recovery under short noisy sequences.

### Positioning

| Method | Mathematical formulation | Temporal structure | Leading inference cost | Key guarantee or control |
|---|---|---|---|---|
| Cunningham et al. (2023) | ReLU–$\ell_1$ reconstruction | None | $O(TdmJ)$ | Sparse, interpretable features |
| Gao et al. (2024) | Hard $k$-sparse autoencoding | None | $O(TdmJ)$ | Exact cardinality control |
| Lieberum et al. (2024) | JumpReLU autoencoding | None | $O(TdmJ)$ | Broad pretrained coverage |
| Bhalla et al. (2025) | Reconstruction plus temporal contrast | Adjacent-token consistency | $O(TdmJ)$ | Empirical semantic smoothness |
| Dokme and Vishwanath (2026) | Contrastive spatio-temporal SAE | Space-time neighborhoods | $O(TdmJ)$ plus contrastive pairs | Tunable coherence trade-off |
| Yeo et al. (2026) | Sparse temporal Gaussian islands | Parametric temporal supports | Application-dependent | Structured clinical concepts |
| Lusch et al. (2017) | Learned Koopman embedding | Linear latent evolution | Network-dependent | Predictive linear coordinates |
| Li et al. (2026) | Sparse Koopman autoencoding | Support-defined local regimes | Network-dependent | Inspectable regime variables |
| Gribonval et al. (2014) | Noisy overcomplete dictionary learning | None | Solver-dependent | Local recovery near a reference dictionary |
| Ghanem and Ahuja (2011) | Probabilistic discriminative dictionary learning | None | Solver-dependent | Joint representation and classification |
| Ours | Top-6 coding plus Laplacian and ridge Koopman refinement | Local and transported bidirectional context | $O(JTdm+Sm^3+RTm^3)$ | Exact support, unique Koopman fit, spectral bound, temporal contraction |

Here $T$ is sequence length, $d$ the observation dimension, $m$ the dictionary width, $J$ the number of sparse iterations, $S$ the number of Koopman fitting stages, and $R$ the number of bidirectional sweeps. Our distinctive contribution is not merely another temporal loss: it is the composition of an exact sparse projection, an exact graph proximal map, a closed-form transition estimate, and normalized dictionary correction.

## Methodology
## 3. Methodology

### 3.1 Problem formulation

For trajectories $X^{(s)}\in\mathbb{R}^{T_s\times d}$, we seek a unit-column dictionary $D=[d_1,\ldots,d_m]$ and row-wise nonnegative codes $Z^{(s)}\in\mathbb{R}_+^{T_s\times m}$. Each row has at most $q=6$ nonzeros. The conceptual objective is
$$
\min_{D,Z,K}\ \frac12\sum_s\|X^{(s)}-Z^{(s)}D^\top\|_F^2+\lambda\sum_s\|Z^{(s)}\|_1
+\frac{\mu_L}{2}\sum_s\|\Delta Z^{(s)}\|_F^2
+\frac{\mu_K}{2}\sum_{s,t}\|z^{(s)}_{t+1}-z^{(s)}_tK^\top\|_2^2
+\frac{\rho}{2}\|K-I\|_F^2,
$$
subject to $Z\geq0$, $\|z_t\|_0\leq6$, and $\|d_j\|_2=1$. The algorithm uses blockwise approximations to this nonconvex problem; the theorems below concern the continuous subproblems solved exactly.

The numerical constants are $\lambda=0.055$, $\mu_L=0.10$, $\mu_K=0.07$, and $\rho=2.0$. Sparse inference uses 24 projected proximal-gradient iterations with
$$
\alpha=\min\left\{0.24,\frac{0.98}{\|D\|_2^2+10^{-8}}\right\}.
$$
Dictionary inference uses 12 iterations per round, and dictionary adaptation uses four rounds, ridge curvature $0.08$, and step $0.32$. Two bidirectional sweeps are applied; each is followed by Laplacian smoothing of strength $0.05$. No momentum term is used.

Define the nonnegative Top-$q$ operator $\mathcal{H}_q^+(U)$ by retaining the $q$ largest entries of each row of $[U]_+=\max(U,0)$ and zeroing the remainder. A sparse iteration is
$$
Z^{j+1}=\mathcal{H}_6^+\!\left(Z^j-\alpha\left[(Z^jD^\top-X)D+\lambda\mathbf{1}\right]\right).
$$

### 3.2 Exact path-Laplacian smoothing

Let $L_T=\Delta^\top\Delta$ be the path-graph Laplacian. Temporal smoothing computes
$$
\mathcal{P}_\gamma(U)=(I_T+\gamma L_T)^{-1}U.
$$
The coefficient matrix is tridiagonal, with endpoint diagonals $1+\gamma$, interior diagonals $1+2\gamma$, and off-diagonals $-\gamma$.

**Lemma 1 (positive definiteness of the path system).** For every $\gamma\geq0$, $I_T+\gamma L_T$ is symmetric positive definite and hence invertible.

**Proof.** For any nonzero $v\in\mathbb{R}^T$,
$$
v^\top(I_T+\gamma L_T)v=\|v\|_2^2+\gamma\|\Delta v\|_2^2>0.
$$
Symmetry follows from $L_T=\Delta^\top\Delta$. Therefore the matrix is symmetric positive definite. $\square$

**Theorem 1 (exact proximal map and mode contraction).** The output $Y=\mathcal{P}_\gamma(U)$ is the unique minimizer of
$$
\frac12\|Y-U\|_F^2+\frac{\gamma}{2}\|\Delta Y\|_F^2.
$$
If $L_Tv_i=\ell_i v_i$, then the component of $U$ along $v_i$ is multiplied by $(1+\gamma\ell_i)^{-1}$. Thus the constant mode is preserved and every nonconstant mode is strictly contracted for $\gamma>0$.

**Proof.** Differentiating the strictly convex objective gives
$$
Y-U+\gamma L_TY=0,
$$
so $(I_T+\gamma L_T)Y=U$. Lemma 1 gives uniqueness. Expanding a column of $U$ in the orthonormal eigenbasis of $L_T$ yields the multiplier $(1+\gamma\ell_i)^{-1}$. For a connected path, $\ell_1=0$ only for the constant vector and $\ell_i>0$ otherwise. $\square$

### 3.3 Correctly oriented Koopman estimation

For row-wise codes, the prediction is $z_{t+1}\approx z_tK^\top$. Stack all predecessor rows in $A$ and successor rows in $B$. We fit
$$
\min_K\ \|B-AK^\top\|_F^2+\rho\|K-I_m\|_F^2.
$$
Let $C=A^\top A+\rho I_m$ and $R=A^\top B+\rho I_m$.

**Theorem 2 (unique ridge Koopman solution).** For $\rho>0$, the unique minimizer is
$$
K^\star=(C^{-1}R)^\top=R^\top C^{-1}.
$$

**Proof.** Set $W=K^\top$. The objective is $f(W)=\|B-AW\|_F^2+\rho\|W-I\|_F^2$. Its gradient is
$$
\nabla f(W)=2A^\top(AW-B)+2\rho(W-I).
$$
The stationary equation is $CW=R$, hence $W=C^{-1}R$. Since
$$
H\mapsto2\|AH\|_F^2+2\rho\|H\|_F^2
$$
is strictly positive for every nonzero $H$, $f$ is strictly convex and the stationary point is unique. Transposing gives the stated $K^\star$. $\square$

The fitted operator is rescaled when necessary so that $\|K\|_2\leq1.05$.

**Proposition 1 (finite-horizon propagation bound).** If $\|K\|_2\leq\kappa$, then for every row vector $z$ and horizon $h\geq0$,
$$
\|z(K^\top)^h\|_2\leq\kappa^h\|z\|_2.
$$
For the proposed method, $\kappa=1.05$.

**Proof.** Submultiplicativity gives $\|z(K^\top)^h\|_2\leq\|z\|_2\|K^\top\|_2^h$. The spectral norm is invariant under transpose, so $\|K^\top\|_2=\|K\|_2\leq\kappa$. $\square$

### 3.4 Bidirectional latent refinement

For an intermediate code sequence $U$, one sweep forms
$$
B_t=U_t+\mu_K\mathbf{1}_{t>1}U_{t-1}K^\top+\mu_K\mathbf{1}_{t<T}U_{t+1}K
$$
and solves, independently for every time step,
$$
Y_tA=B_t,\qquad A=I_m+\mu_K(I_m+K^\top K).
$$
The result is rectified, Laplacian-smoothed, and finally projected with $\mathcal{H}_6^+$.

**Proposition 2 (well-posed bidirectional solve).** $A$ is symmetric positive definite, $\lambda_{\min}(A)\geq1+\mu_K$, and
$$
\|A^{-1}\|_2\leq\frac{1}{1+\mu_K}.
$$
Consequently, perturbing the right-hand side by $E$ perturbs the unconstrained solution by at most $\|E\|_F/(1+\mu_K)$.

**Proof.** For nonzero $v$,
$$
v^\top Av=(1+\mu_K)\|v\|_2^2+\mu_K\|Kv\|_2^2\geq(1+\mu_K)\|v\|_2^2.
$$
Thus $A$ is positive definite with the stated eigenvalue lower bound. The inverse-norm and perturbation bounds follow from $\|A^{-1}\|_2=1/\lambda_{\min}(A)$. $\square$

### 3.5 Dictionary refinement

Given codes from all training and observed trajectories, define
$$
g_j=\sum_s (X^{(s)}-Z^{(s)}D^\top)^\top Z^{(s)}_{:,j},\qquad
c_j=0.08+\sum_s\|Z^{(s)}_{:,j}\|_2^2.
$$
Each round updates
$$
\widetilde d_j=d_j+0.32\,g_j/c_j,\qquad d_j\leftarrow\widetilde d_j/\max(\|\widetilde d_j\|_2,10^{-12}).
$$

**Proposition 3 (feasibility preservation).** Every nonzero updated dictionary column has unit Euclidean norm; the denominator $c_j$ is strictly positive.

**Proof.** Since $c_j\geq0.08$, division is well defined. For $\widetilde d_j\neq0$, normalization gives $\|d_j\|_2=1$. The numerical floor also prevents division by zero. $\square$

```text
Algorithm 1: Joint dictionary refinement
Input: trajectories {X(s)}, initial dictionary D, four rounds
Normalize every column of D
for r = 1,...,4 do
    for each trajectory X(s) do
        infer Z(s) using 12 iterations of nonnegative Top-6 coding
    end for
    accumulate residual gradients g_j and curvatures c_j
    update d_j <- d_j + 0.32 g_j / c_j for every atom j
    normalize every dictionary column
end for
return D
```

```text
Algorithm 2: Koopman–Laplacian sparse inference
Input: trajectories {X(s)}, refined dictionary D
Infer preliminary codes with 24 nonnegative Top-6 iterations
Apply the exact path-Laplacian solve with strength 0.10
Fit K = (C^{-1}R)^T with ridge 2.0; rescale if ||K||_2 > 1.05
for two bidirectional sweeps do
    combine present, forward-predicted, and backward-adjoint codes
    solve Y_t [I + 0.07(I + K^T K)] = B_t
    rectify Y and apply path-Laplacian smoothing with strength 0.05
end for
Project every row onto its six largest nonnegative entries
Refit K from the refined codes using the same ridge and spectral cap
Repeat inference for the requested trajectories and reconstruct X_hat = Z D^T
return D, K, {Z}, {X_hat}
```

### 3.6 Complexity

Let $N=\sum_sT_s$ be the total number of fitted time points. One sparse-gradient iteration costs $O(Ndm)$ time and $O(Nm+dm)$ working memory. A path-Laplacian solve costs $O(Nm)$ time and $O(Nm)$ memory. Accumulating Koopman covariances costs $O(Nm^2)$ time, and solving the dense $m\times m$ system costs $O(m^3)$ time and $O(m^2)$ memory. Each bidirectional sweep uses one dense factorization costing $O(m^3)$ and $T_s$ right-hand sides costing $O(T_sm^2)$ per sequence. Dictionary accumulation costs $O(Ndm)$.

With four dictionary rounds, 12 inner iterations, 24 final sparse iterations, two Koopman fits, and two bidirectional sweeps, the total asymptotic time is
$$
O(JNdm+Nm^2+Sm^3+RNm^2+Rm^3),
$$
where $J$ collects the fixed sparse iterations, $S=2$, and $R=2$. Peak working memory is
$$
O(Nm+dm+m^2),
$$
excluding stored input observations. For the evaluated dimensions $d=32$ and $m=64$, the dense temporal solves remain small relative to large-scale overcomplete SAE settings.

## Experiments
## 4. Experiments

### 4.1 Protocol

We evaluate eight trajectories with 48 time steps, $d=32$ observed dimensions, $m=64$ latent atoms, and five generating features per step. The inference budget retains six coefficients, permitting limited model mismatch while enforcing a strict cardinality ceiling. We report clean reconstruction error, feature-support $F_1$, Koopman rollout error, mean active count, dictionary recovery by mean maximum cosine, and the composite loss defined in Section 1. All pairwise uncertainty estimates are paired Student-t 95% confidence intervals over matched trajectories. Runtime was measured on a pinned 3.90 GHz ARM core.

### 4.2 Held-out benchmark

| Method | Composite loss ↓ | Reconstruction ↓ | Support $F_1$ ↑ | Rollout ↓ | Paired gain of ours |
|---|---:|---:|---:|---:|---:|
| Framewise ISTA-SAE | 0.2670 | 0.00873 | 0.2789 | 0.00903 | $0.0127\pm0.0075$ |
| TopK-SAE | 0.2731 | 0.01401 | 0.2715 | 0.01632 | $0.0188\pm0.0080$ |
| Laplacian temporal SAE | 0.3022 | 0.00793 | 0.3125 | 0.00834 | $0.0479\pm0.0191$ |
| **Koopman–Laplacian SAE** | **0.2543** | **0.00608** | 0.2955 | **0.00653** | — |

The proposed method achieves the best composite, reconstruction, and rollout values. Relative to framewise ISTA, the composite loss falls by 4.75%, reconstruction error by 30.3%, and rollout error by 27.7%. Relative to TopK-SAE, the corresponding reductions are 6.87%, 56.6%, and 60.0%. The Laplacian temporal baseline attains higher support $F_1$ in isolation, but its weaker reconstruction and rollout produce a 15.85% larger composite loss. Every paired confidence interval lies strictly above zero, supporting a trajectory-level advantage rather than an aggregate-only effect.

The method uses exactly six active coefficients on average, incurs zero sparsity penalty, reaches a dictionary cosine recovery of 0.7280, takes 293.6 ms, and peaks near 1.09 MB of traced working memory.

### 4.3 Complete temporal factorial and dictionary integration

| Laplacian | Koopman | Dictionary refinement | Composite ↓ | Reconstruction ↓ | Support $F_1$ ↑ | Rollout ↓ |
|---|---|---|---:|---:|---:|---:|
| No | No | No | 0.2763 | 0.01302 | 0.2581 | 0.01437 |
| Yes | No | No | 0.2730 | 0.01181 | 0.2628 | 0.01272 |
| No | Yes | No | 0.2726 | 0.01183 | 0.2622 | 0.00997 |
| Yes | Yes | No | 0.2648 | 0.01048 | 0.2800 | 0.00944 |
| **Yes** | **Yes** | **Yes** | **0.2543** | **0.00608** | **0.2955** | **0.00653** |

Laplacian smoothing alone improves the no-temporal model by 0.0032, while Koopman refinement alone improves it by 0.0037 and reduces rollout error by 30.6%. Their joint reduction is 0.0114, exceeding the sum of the isolated reductions by 0.0045; this positive interaction shows that local denoising improves the codes from which transport is learned, while transport prevents smoothing from collapsing legitimate evolution into identity persistence. Dictionary refinement then contributes another 0.0106 reduction, decreases reconstruction error by 41.9%, raises support $F_1$ by 0.0155, and decreases rollout error by 30.8%.

### 4.4 Component subtraction

| Removed component | Composite ↓ | Reconstruction ↓ | Support $F_1$ ↑ | Rollout ↓ | Mean active count |
|---|---:|---:|---:|---:|---:|
| None | **0.2412** | **0.00636** | **0.3343** | **0.00749** | 6.00 |
| Dictionary refinement | 0.2517 | 0.01214 | 0.3236 | 0.01122 | 6.00 |
| Koopman refinement | 0.2498 | 0.00804 | 0.3179 | 0.01224 | 6.00 |
| Laplacian smoothing | 0.2512 | 0.00835 | 0.3120 | 0.00837 | 6.00 |
| Hard support projection | 0.8945 | 0.00946 | 0.3337 | 0.01034 | 34.64 |

Each active component has a measurable role. Removing dictionary refinement increases loss by 0.0104 and nearly doubles reconstruction error. Removing Koopman refinement increases loss by 0.0086 and rollout error by 63.3%. Removing Laplacian smoothing increases loss by 0.0100 and reduces support $F_1$ by 0.0223. The strongest causal signal is the support projection: without it, the average active count rises from 6.00 to 34.64 and the sparsity penalty dominates, raising composite loss by 0.6533. Thus fixed-cardinality projection is necessary to preserve the semantic and computational meaning of sparse features.

### 4.5 Sequence-noise and sparsity stress scaling

| Condition | Ours ↓ | Framewise ISTA ↓ | TopK-SAE ↓ | Laplacian temporal SAE ↓ | Best paired margin |
|---|---:|---:|---:|---:|---:|
| Standard held-out sequences | 0.2543 | 0.2670 | 0.2731 | 0.3022 | $0.0127\pm0.0075$ |
| Elevated noise/sparsity stress | 0.2701 | 0.3152 | 0.3042 | 0.3994 | $0.0340\pm0.0083$ |

Under the stress condition, our loss increases by only 0.0158, whereas framewise ISTA increases by 0.0482 and the Laplacian temporal model by 0.0972. Our method improves over framewise ISTA by $0.0451\pm0.0124$, TopK-SAE by $0.0340\pm0.0083$, and the Laplacian temporal model by $0.1293\pm0.0134$. All intervals remain strictly positive. The widening advantage is consistent with the proposed mechanism: exact support control limits activation proliferation, Laplacian smoothing suppresses local perturbations, and Koopman refinement aggregates predictive information from both temporal directions.

### 4.6 Validation-to-test consistency

The principal validation result is 0.2412, with reconstruction 0.00636, support $F_1$ 0.3343, rollout error 0.00749, zero sparsity penalty, and dictionary cosine 0.7134. The held-out result is 0.2543, with reconstruction improving to 0.00608, rollout improving to 0.00653, zero sparsity penalty, and dictionary cosine increasing to 0.7280. The modest composite shift is driven by lower support $F_1$, while the continuous reconstruction and dynamical metrics remain stable or improve.

## Limitations
## 5. Limitations

The empirical study uses short synthetic trajectories with $d=32$, $m=64$, and eight matched sequences. It therefore establishes controlled mechanism attribution, not yet performance on large language models, video backbones, or clinical systems. Human interpretability, feature monosemanticity, downstream causal interventions, and transfer across domains were not evaluated.

The method estimates a single linear Koopman operator. Multibasin or abruptly switching systems may require multiple local operators, nonlinear transition models, or explicit changepoint structure. The spectral cap bounds finite-horizon amplification but does not imply asymptotic contraction because its value, 1.05, permits mild growth. Likewise, hard Top-6 projection makes the global objective nonconvex; our proofs establish exactness and stability of the continuous subproblems, not global convergence of the complete alternating procedure.

Dense $m\times m$ linear algebra introduces $O(m^3)$ fitting and sweep costs. This is inexpensive at width 64 but would dominate at widths in the thousands unless the transition operator were diagonal, low-rank, block-sparse, or solved iteratively. Finally, dictionary adaptation uses observations from the target trajectory without clean labels, making the procedure transductive. Future work should evaluate an inductive variant with a frozen training dictionary and transition operator, report multi-seed large-scale benchmarks, and test whether the recovered temporal features improve mechanistic interventions rather than only reconstruction and support metrics.

## References
- Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability. DOI: 10.48550/arXiv.2511.05541
- Interpreting Video Representations with Spatio-Temporal Sparse Autoencoders. DOI: 10.48550/arXiv.2604.03919
- Temporal Island Sparse Autoencoders for Interpreting Clinical Time-Series Models. DOI: 10.48550/arXiv.2611.03420
- Sparse Koopman Autoencoders Identify Local Dynamical Regimes in Multibasin Systems. DOI: 10.48550/arXiv.2608.29057
- Sparse Autoencoders Find Highly Interpretable Features in Language Models. DOI: 10.48550/arXiv.2309.08600
- Scaling and evaluating sparse autoencoders. DOI: 10.48550/arXiv.2406.04093
- Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. DOI: 10.48550/arXiv.2408.05147
- Deep learning for universal linear embeddings of nonlinear dynamics. DOI: 10.48550/arXiv.1712.09707
- Sparse and spurious: dictionary learning with noise and outliers. DOI: 10.48550/arXiv.1407.5155
- A Probabilistic Framework for Discriminative Dictionary Learning. DOI: 10.48550/arXiv.1109.2389

## Appendix A: Complete Experimental Trajectory Manifest

| Experiment | mse |
|---|---|
| baseline_source | 0.255520396 |
| baseline_subset | 0.255737511 |
| baseline_full | 0.255520396 |
| initial_subset | 0.244567879 |
| initial_full | 0.242320995 |
| seed0_subset | 0.250853259 |
| seed0_full | 0.241627451 |
| evolved0_subset | 0.250853259 |
| evolved0_full | 0.241627451 |
| ablation0 | 0.251677082 |
| ablation1 | 0.249821760 |
| ablation2 | 0.251240961 |
| ablation3 | 0.894510855 |
| ablation4 | 0.241232261 |
| pruned | 0.241232261 |
| pruned_ablation0 | 0.241232261 |
| pruned_ablation1 | 0.241232261 |
| pruned_ablation2 | 0.241232261 |
| pruned_ablation3 | 0.241232261 |
| pruned_ablation4 | 0.241232261 |
| rebuttal0 | 0.270117832 |
| final_test | 0.254292893 |
