# Innovation-Gated Koopman Support Consensus for Predictive-Graph Temporal Sparse Autoencoders

## Abstract
Sparse autoencoders ordinarily infer each observation independently, making latent supports vulnerable to high-frequency noise, while first-order temporal smoothing can blur genuine semantic transitions. We propose an innovation-gated temporal sparse autoencoder that jointly refines a unit-norm overcomplete dictionary, learns a predictive feature graph, estimates ridge-regularized latent Koopman dynamics, performs bidirectional temporal proximal inference, and coordinates exact-cardinality support masks across time. Koopman innovations attenuate persistence at poorly predicted transitions, while fixed-support projected debiasing improves amplitudes without changing the inferred support mask. Our theoretical analysis establishes positive semidefiniteness of the predictive graph, a valid smooth-step bound, uniqueness and exact spectral attenuation of the temporal Laplacian solve, uniqueness of the innovation-weighted Koopman estimator, monotone coordinate improvement of the discrete consensus energy, exact mask cardinality, and monotone fixed-support amplitude refinement. On eight controlled trajectories with 32-dimensional observations, 64 atoms, and five inferred active features, the method obtains 0.2191 composite loss, improving over framewise ISTA, framewise TopK, and a Laplacian temporal sparse autoencoder by 17.0%, 22.4%, and 33.0%, respectively; all paired Student-$t$ 95% confidence intervals exclude zero. Factorial isolation attributes loss reductions of 7.39% to Laplacian smoothing alone, 1.17% to Koopman inference alone, 7.69% to their combination, and a further 9.94% to integration with dictionary, graph, consensus, and amplitude refinement. On an independently generated test suite not used for model selection, the method improves over the same comparators by 10.9%, 12.9%, and 21.3%, again with strictly positive paired confidence intervals. Length, noise, and sparsity sweeps preserve the exact five-, three-, or seven-feature mask budget and zero sparsity violation throughout.

## Introduction
## 1 Introduction

Sparse autoencoders (SAEs) represent an observation $x_t\in\mathbb{R}^d$ as $x_t\approx Dz_t$, where $D\in\mathbb{R}^{d\times m}$ is an overcomplete dictionary and $z_t\in\mathbb{R}_+^m$ is sparse. This decomposition is useful only when a recovered feature remains identifiable across nearby observations. Framewise inference does not encode this requirement: noise can exchange similarly aligned atoms between consecutive frames even when the underlying state evolves smoothly.

A first-order temporal penalty partially addresses this instability but assumes persistence around the identity map. That assumption is inappropriate when amplitudes rotate, decay, or transfer predictably between features. Dense temporal smoothing can also suppress authentic regime changes. We therefore separate three questions: how amplitudes evolve, when supports should persist, and when a transition provides enough evidence to change them.

We propose the Innovation-Gated Koopman Support-Consensus Temporal Sparse Autoencoder. The method combines five coupled mechanisms. First, accelerated nonnegative sparse inference enforces a Top-$k$ mask at every iteration. Second, a normalized predictive graph blends feature coactivation with correlation between Koopman residuals. Third, an exact temporal Laplacian solve and bidirectional Koopman block sweeps denoise continuous amplitudes. Fourth, a detached innovation gate controls both discrete support persistence and the influence of each transition on the final Koopman estimate. Fifth, tangent-preconditioned dictionary refinement improves the observation basis while preserving unit-norm atoms. A final projected-gradient stage debiases amplitudes on the fixed support.

The contribution is a unified inference algorithm rather than an undifferentiated collection of penalties. The continuous stage proposes amplitudes; the discrete stage performs cardinality-constrained support-coordinate minimization; Koopman innovations determine when neighboring support evidence should be trusted; and amplitude debiasing cannot introduce a new feature. This decomposition directly controls the principal failure modes of temporal sparse coding.

Our theoretical contributions are self-contained. We prove that the normalized predictive graph is positive semidefinite; derive the exact Lipschitz constant used by the accelerated proposal step; prove uniqueness and frequency-selective contraction of the linear-time temporal solve; establish the unique closed-form innovation-weighted Koopman update; prove exact mask cardinality and monotone coordinate descent for support consensus; and prove monotone reconstruction improvement for fixed-support amplitude debiasing. We also establish exact feasibility and first-order tangency of dictionary normalization.

Our experiments evaluate reconstruction, support identification, latent rollout, dictionary recovery, cardinality, runtime, and memory. The method reaches 0.2191 composite loss, 0.00693 clean reconstruction error, 0.3992 support F1, and 0.00738 rollout error while activating exactly five features per frame. Compared with framewise ISTA, framewise TopK, and a Laplacian temporal SAE, the loss reductions are 17.0%, 22.4%, and 33.0%. The complete $2\times2$ temporal factorial study isolates positive contributions from Laplacian smoothing and learned Koopman dynamics, while component subtraction confirms positive contributions from support consensus, innovation gating, transition-weighted dynamics, graph-neighbor evidence, and amplitude debiasing.

The independently generated test suite provides a stronger distribution-replication test. Without using it for model selection, the method achieves 0.2379 composite loss and improves over the three comparators by 10.9%, 12.9%, and 21.3%. The corresponding paired 95% intervals remain strictly positive. Multi-scale sweeps further show stable finite inference across sequence lengths 24–96, noise standard deviations 0.07–0.21, and active-feature budgets 3–7.

## Related Work
## 2 Related Work

### Sparse autoencoders and dictionary learning

[Cunningham et al. (2023)](https://arxiv.org/abs/2309.08600) use sparse autoencoders to recover monosemantic directions from language-model activations, establishing the interpretability motivation for overcomplete sparse decomposition. Their inference is framewise. [Gao et al. (2024)](https://arxiv.org/abs/2406.04093) introduce explicit $k$-sparse autoencoders, simplify sparsity control, reduce dead latents, and document scaling laws. We retain exact Top-$k$ control but make support inference trajectory-aware and innovation-gated. [Lieberum et al. (2024)](https://arxiv.org/abs/2408.05147) release JumpReLU SAEs across layers and sublayers of Gemma 2 models. Their contribution is broad model coverage, whereas ours changes temporal inference, learned dynamics, and dictionary refinement.

Classical dictionary-learning theory provides complementary foundations. [Gribonval et al. (2014)](https://arxiv.org/abs/1407.5155) prove the existence of local minima near a generating dictionary under overcompleteness, noise, and outliers. Our results instead give deterministic guarantees for the graph, temporal, Koopman, support, and normalization operations executed by the proposed algorithm. [Ghanem and Ahuja (2011)](https://arxiv.org/abs/1109.2389) combine sparse representation and classification through joint maximum-a-posteriori estimation. Our setting is unsupervised and dynamical: the auxiliary structure is a predictive graph and linear latent evolution rather than a label predictor.

### Temporal sparse representations

[Bhalla et al. (2025)](https://arxiv.org/abs/2511.05541) introduce a temporal SAE whose contrastive loss encourages adjacent-token consistency and improves semantic coherence. Our method replaces identity-centered persistence with learned $z_{t+1}\approx z_tK^\top$ evolution, uses forward and backward temporal evidence, and reduces persistence when the learned dynamics report a large innovation.

[Dokme and Vishwanath (2026)](https://arxiv.org/abs/2604.03919) show that framewise TopK selection destabilizes video features and propose spatio-temporal contrastive losses with Matryoshka grouping. Their hierarchy targets video representation structure; ours derives a sparse predictive graph, closed-form dynamics, and a discrete innovation-gated support objective. [Yeo et al. (2026)](https://arxiv.org/abs/2611.03420) represent clinical concepts with temporal Gaussian islands and sparse variable compositions, separating shared concept construction from patient-specific selection. Our method instead learns atom dynamics and performs per-frame cardinality-constrained inference without prescribing Gaussian temporal shapes.

### Koopman representation learning

[Lusch et al. (2017)](https://arxiv.org/abs/1712.09707) learn coordinates in which nonlinear dynamics become approximately linear, demonstrating the value of Koopman embeddings for prediction and control. [Li et al. (2026)](https://arxiv.org/abs/2608.29057) introduce sparse Koopman autoencoders whose latent supports expose local regimes in multibasin systems. We combine Koopman evolution with overcomplete nonnegative sparse coding, but use a ridge-anchored closed-form operator and turn its residual into an explicit confidence signal for support persistence and transition weighting.

The following comparison reports representative dominant per-trajectory inference costs for length $T$, observation dimension $d$, dictionary width $m$, and Koopman dimension $q$.

| Work | Mathematical formulation | Representative time | Principal guarantee or evidence |
|---|---|---:|---|
| Cunningham et al. | Reconstruction plus sparse activation penalty | $O(Tdm)$ per pass | Empirical monosemantic recovery |
| Gao et al. | Hard $k$-sparse bottleneck | $O(Tdm+Tm)$ | Exact activation budget; empirical scaling laws |
| Lieberum et al. | JumpReLU sparse bottleneck | $O(Tdm)$ | Broad empirical model and layer coverage |
| Bhalla et al. | Adjacent-token contrastive regularization | $O(Tdm+Tm)$ | Empirical semantic coherence |
| Dokme and Vishwanath | Spatio-temporal contrast and hierarchical grouping | $O(Tdm+Tm)$ | Coherence, classification, and retrieval gains |
| Yeo et al. | Gaussian temporal islands and sparse variable composition | $O(Tdm)$ | Structured patient-shared concepts |
| Li et al. | Sparse latent Koopman autoencoder | $O(Tdm+Tq^2)$ | Empirical local-regime identification |
| Lusch et al. | Learned Koopman eigenfunction coordinates | Encoder cost $+O(Tq^2)$ | Approximate linear latent evolution |
| Gribonval et al. | Noisy overcomplete sparse dictionary learning | $O(Tdm)$ per alternating pass | High-probability local minimum near the generating dictionary |
| Ghanem and Ahuja | MAP sparse coding plus discriminative prediction | $O(Tdm)$ plus classifier updates | Unified representation-classification objective |
| Ours | Graph-regularized Top-$k$ coding, Laplacian–Koopman proximal inference, gated consensus | $O(q_fT(dm+m^2)+q_TTm^2+m^3)$ per trajectory | Unique dynamics update, exact temporal response, exact mask budget, coordinate energy descent |

Unlike prior methods, our formulation makes learned dynamics serve three coordinated roles: continuous amplitude prediction, transition confidence estimation, and discrete support persistence.

## Methodology
## 3 Methodology

### 3.1 Problem formulation

For observed trajectories $X^{(s)}\in\mathbb{R}^{T_s\times d}$, we seek a dictionary $D=[d_1,\ldots,d_m]\in\mathbb{R}^{d\times m}$ and codes $Z^{(s)}\in\mathbb{R}_+^{T_s\times m}$ satisfying $\|d_j\|_2=1$ and an inferred support-mask budget $k$. Let $\Delta\in\mathbb{R}^{(T-1)\times T}$ be the first-difference matrix. The continuous proposal stage is organized by

$$
\mathcal{J}(Z;D,K,L)=\frac12\|X-ZD^\top\|_F^2+\lambda\|Z\|_1+\mu_g\operatorname{tr}(ZLZ^\top)+\mu_\Delta\|\Delta Z\|_F^2+\frac{\mu_K}{2}\sum_{t=1}^{T-1}\|z_{t+1}-z_tK^\top\|_2^2.
$$

We use $\lambda=0.07$, $\mu_g=0.035$, $\mu_\Delta=1.0$, and $\mu_K=0.20$. Operator splitting applies 24 accelerated reconstruction/graph steps, an exact Laplacian solve, two alternating forward/backward Koopman block sweeps, and Top-$k$ projection. Three support-consensus sweeps and six fixed-support amplitude updates follow.

### 3.2 Predictive feature graph

For each preliminary code trajectory, define normalized coactivation and predictive-residual similarities

$$
C_{ij}=\frac{Z_{:i}^\top Z_{:j}}{\|Z_{:i}\|_2\|Z_{:j}\|_2+\varepsilon},\qquad
P_{ij}=\frac{|R_{:i}^\top R_{:j}|}{\|R_{:i}\|_2\|R_{:j}\|_2+\varepsilon},
$$

where $R_{t:}=z_{t+1}-z_tK^\top$. We remove diagonal entries, average across trajectories, form $0.1C+0.9P$, symmetrize, retain the 16 strongest neighbors per feature, and take the symmetric union. A weak nearest-neighbor ring edge with weight 5% of the median positive affinity prevents isolated nodes. With affinity $A=A^\top\ge0$ and degree matrix $Q=\operatorname{diag}(A\mathbf{1})$, the normalized Laplacian is

$$
L_0=I-Q^{-1/2}AQ^{-1/2},\qquad L=\frac{m}{\operatorname{tr}(L_0)}L_0.
$$

The graph is constructed after the first coding pass. Continuous codes use $0.035\operatorname{tr}(ZLZ^\top)$, dictionary refinement uses $0.002\operatorname{tr}(DLD^\top)$, and graph-neighbor support evidence enters the discrete unary score with weight 0.01.

**Lemma 1 (positive-semidefinite graph regularization).** If $A=A^\top\ge0$ and every degree is positive, then $L_0\succeq0$. Positive trace normalization preserves positive semidefiniteness.

**Proof.** For any $v\in\mathbb{R}^m$,

$$
v^\top L_0v=\frac12\sum_{i,j}A_{ij}\left(\frac{v_i}{\sqrt{Q_{ii}}}-\frac{v_j}{\sqrt{Q_{jj}}}\right)^2\ge0.
$$

Thus every quadratic form is nonnegative. The graph contains an edge and has no isolated node, so $\operatorname{tr}(L_0)>0$. Multiplication by the positive scalar $m/\operatorname{tr}(L_0)$ cannot change the sign of any quadratic form. $\square$

### 3.3 Accelerated sparse proposal

Let

$$
f(Z)=\frac12\|X-ZD^\top\|_F^2+\mu_g\operatorname{tr}(ZLZ^\top).
$$

One proposal step is

$$
Z^+=\mathcal{P}_{k,+}\left(Y-\eta\nabla f(Y)-\eta\lambda\mathbf{1}\right),
$$

where $\mathcal{P}_{k,+}$ clips negative entries and retains the $k$ largest values in every row. Momentum uses $a^+=(1+\sqrt{1+4a^2})/2$ and extrapolation $Y^+=Z^+ + ((a-1)/a^+)(Z^+-Z)$, with restart when the extrapolation and most recent displacement have a positive inner product. Top-$k$ projection is applied at all 24 iterations. The step size is

$$
\eta=\min\left\{1,\frac{0.98}{\|D\|_2^2+2\mu_g\|L\|_2}\right\}.
$$

**Proposition 1 (valid smooth-step bound).** The gradient of $f$ is Lipschitz in Frobenius norm with constant

$$
L_f=\|D\|_2^2+2\mu_g\|L\|_2,
$$

and the chosen step satisfies $0<\eta<1/L_f$.

**Proof.** For a perturbation $H$,

$$
\nabla f(Z+H)-\nabla f(Z)=HD^\top D+2\mu_gHL.
$$

Submultiplicativity yields

$$
\|HD^\top D\|_F\le\|D\|_2^2\|H\|_F,\qquad
\|HL\|_F\le\|L\|_2\|H\|_F,
$$

which proves the stated constant. If $L_f\ge0.98$, then $\eta=0.98/L_f<1/L_f$. If $L_f<0.98$, then $\eta=1<1/L_f$. Positivity follows from finite $D$ and $L$. $\square$

### 3.4 Laplacian and bidirectional Koopman proximal inference

For proposal $B$, temporal Laplacian smoothing solves

$$
Z_\Delta=\arg\min_Z\frac12\|Z-B\|_F^2+\mu_\Delta\|\Delta Z\|_F^2
=(I+2\mu_\Delta\Delta^\top\Delta)^{-1}B.
$$

**Theorem 1 (unique temporal solve and exact spectral attenuation).** The Laplacian proximal problem has a unique solution. If $\Delta^\top\Delta=U\operatorname{diag}(\xi_1,\ldots,\xi_T)U^\top$, temporal mode $j$ is multiplied exactly by $(1+2\mu_\Delta\xi_j)^{-1}$. The solution requires $O(Tm)$ time and $O(Tm)$ output space.

**Proof.** Differentiation gives the normal equation

$$
(I+2\mu_\Delta\Delta^\top\Delta)Z=B.
$$

Because $\Delta^\top\Delta\succeq0$, all eigenvalues of the coefficient matrix equal $1+2\mu_\Delta\xi_j\ge1$. The matrix is therefore positive definite, so the minimizer is unique. Substituting the eigendecomposition gives

$$
Z=U\operatorname{diag}\left((1+2\mu_\Delta\xi_j)^{-1}\right)U^\top B,
$$

which proves the exact attenuation. The coefficient matrix is tridiagonal: its endpoint diagonal is $1+2\mu_\Delta$, its interior diagonal is $1+4\mu_\Delta$, and its off-diagonal is $-2\mu_\Delta$. Forward elimination and backward substitution each visit every time index once and operate on $m$ right-hand sides, giving $O(Tm)$ time. Positive definiteness guarantees nonzero elimination pivots. $\square$

The Koopman proximal stage minimizes

$$
\frac12\|Z-Z_\Delta\|_F^2+\frac{\mu_K}{2}\sum_{t=1}^{T-1}\|z_{t+1}-z_tK^\top\|_2^2.
$$

For an interior row, the unconstrained block minimizer is

$$
z_t\leftarrow\left[z_{\Delta,t}+\mu_K(z_{t-1}K^\top+z_{t+1}K)\right]
\left[I+\mu_K(I+K^\top K)\right]^{-1}.
$$

At the first time index the matrix is $I+\mu_KK^\top K$; at the last it is $(1+\mu_K)I$. Every block update is clipped to the nonnegative orthant. We perform two complete forward/backward sweeps before Top-$k$ projection.

### 3.5 Innovation-weighted Koopman estimation

Given the current operator $K_0$, define

$$
r_t=\frac{\|z_{t+1}-z_tK_0^\top\|_2}{\operatorname{median}_{u:\|z_u\|_2>0}\|z_u\|_2+\varepsilon},\qquad
g_t=\operatorname{clip}\left(\exp\left[-\frac12(r_t/1.5)^2\right],0.05,1\right).
$$

Large innovations make $g_t$ small, weakening persistence and reducing that transition's influence on the final dynamics fit. Gates are treated as constants during the update. With $W=K^\top$, the estimator solves

$$
\min_W\sum_{s,t}g_t^{(s)}\|z_{t+1}^{(s)}-z_t^{(s)}W\|_2^2+\rho\|W-I\|_F^2,\qquad \rho=1.5.
$$

**Theorem 2 (unique innovation-weighted Koopman estimator).** Define

$$
G=\sum_{s,t}g_t^{(s)}(z_t^{(s)})^\top z_t^{(s)},\qquad
C=\sum_{s,t}g_t^{(s)}(z_t^{(s)})^\top z_{t+1}^{(s)}.
$$

For $\rho>0$, the unique minimizer is

$$
W^*=(G+\rho I)^{-1}(C+\rho I),\qquad K^*=(W^*)^\top.
$$

**Proof.** Expanding the quadratic and differentiating gives

$$
\nabla_W=2(GW-C)+2\rho(W-I).
$$

The stationary condition is $(G+\rho I)W=C+\rho I$. For every nonzero $v$,

$$
v^\top(G+\rho I)v=\sum_{s,t}g_t^{(s)}\|z_t^{(s)}v\|_2^2+\rho\|v\|_2^2>0.
$$

Thus $G+\rho I$ is positive definite and invertible. The Hessian is $2(G+\rho I)\otimes I\succ0$, so the stationary point is the unique global minimizer. $\square$

### 3.6 Innovation-gated support consensus

Let $p_t\ge0$ be the temporal proposal and let $s_t\in\{0,1\}^m$ satisfy $\mathbf{1}^\top s_t=k$. The unary vector $u_t$ is the row-standardized proposal plus 0.01 times standardized activation evidence propagated through the row-normalized feature graph. We minimize by coordinate sweeps

$$
E(S)=-\sum_tu_t^\top s_t-\gamma\sum_{t=1}^{T-1}g_ts_t^\top s_{t+1},\qquad \gamma=0.12.
$$

Because all masks have cardinality $k$, this is equivalent up to a constant to the gated switching penalty $(\gamma/2)\sum_tg_t\|s_{t+1}-s_t\|_1$. Thus the implemented overlap reward is $0.12$, and its exactly equivalent binary-support switching coefficient is $0.06$.

**Proposition 2 (exact mask cardinality and coordinate energy descent).** With all other masks fixed, choosing the $k$ largest entries of

$$
u_t+\gamma g_{t-1}s_{t-1}+\gamma g_ts_{t+1}
$$

is a global minimizer of $E$ with respect to $s_t$. Every coordinate update is non-increasing in $E$, and every mask has exactly $k$ active entries.

**Proof.** The terms involving $s_t$ equal the negative inner product between $s_t$ and the displayed vector. Among binary vectors with exactly $k$ ones, the inner product is maximized by assigning ones to the $k$ largest entries: if a chosen entry were smaller than an unchosen entry, swapping them would increase the inner product. Therefore the update globally minimizes the coordinate subproblem and cannot increase $E$. Constructing the update with exactly $k$ indices preserves $\mathbf{1}^\top s_t=k$. Ties may be resolved arbitrarily without changing the objective. $\square$

After three forward/backward sweeps, amplitudes are $z_t=p_t\odot s_t$. The mask therefore guarantees at most $k$ nonzero amplitudes and exactly $k$ admissible coordinates.

### 3.7 Fixed-support amplitude debiasing

For six iterations we update amplitudes on the fixed mask by

$$
Z^+=\Pi_{\mathcal{C}_S}\left(Z-\alpha(ZD^\top-X)D\right),
$$

where $\mathcal{C}_S=\{Z:Z\ge0,\;Z_{tj}=0\text{ when }s_{tj}=0\}$ and

$$
\alpha=0.01\min\left\{1,\frac{0.98}{\|D\|_2^2}\right\}.
$$

**Proposition 3 (support invariance and monotone amplitude refinement).** The update cannot introduce an index outside $S$. Moreover, for $h(Z)=\frac12\|X-ZD^\top\|_F^2$, every update satisfies $h(Z^+)\le h(Z)$.

**Proof.** Projection onto $\mathcal{C}_S$ clips negative supported coordinates and sets every unsupported coordinate to zero, proving support invariance. The gradient of $h$ has Lipschitz constant $L_h=\|D\|_2^2$. The stated step obeys $0<\alpha\le1/L_h$: if $L_h\ge0.98$, then $\alpha=0.0098/L_h$; otherwise $\alpha=0.01<1/L_h$. Let $Y=Z^+$ and $G=\nabla h(Z)$. Optimality of Euclidean projection gives

$$
\langle Z-\alpha G-Y,Z-Y\rangle\le0,
$$

hence $\langle G,Y-Z\rangle\le-\|Y-Z\|_F^2/\alpha$. Smoothness then yields

$$
h(Y)\le h(Z)+\langle G,Y-Z\rangle+\frac{L_h}{2}\|Y-Z\|_F^2
\le h(Z)-\left(\frac1\alpha-\frac{L_h}{2}\right)\|Y-Z\|_F^2\le h(Z).
$$

Thus reconstruction error cannot increase. $\square$

### 3.8 Dictionary refinement

Four outer rounds refine the dictionary. For atom $j$, define curvature $H_j=0.08+\sum_{s,t}(z_{tj}^{(s)})^2$. The reconstruction field is divided columnwise by $H_j$, after which the graph field $0.004DL$ is added. For the resulting column field $g_j$, we use the tangent component

$$
p_j=g_j-d_j(d_j^\top g_j),\qquad d_j^+=\frac{d_j-0.10p_j}{\|d_j-0.10p_j\|_2}.
$$

**Proposition 4 (exact dictionary feasibility and first-order tangency).** If $\|d_j\|_2=1$, then $\|d_j^+\|_2=1$, $d_j^\top p_j=0$, and $d_j^+=d_j-0.10p_j+O(0.10^2\|p_j\|_2^2)$.

**Proof.** Normalization directly gives unit norm. Furthermore,

$$
d_j^\top p_j=d_j^\top g_j-(d_j^\top d_j)(d_j^\top g_j)=0.
$$

Consequently $\|d_j-\eta p_j\|_2^2=1+\eta^2\|p_j\|_2^2$. Expanding $(1+\eta^2\|p_j\|_2^2)^{-1/2}$ around $\eta=0$ proves the first-order expression. $\square$

### 3.9 Algorithms

```text
Algorithm 1: Fit the innovation-gated temporal sparse autoencoder
Input: observed trajectories, initial dictionary D, cardinality k
1: Normalize every column of D; set K = I
2: for outer round r = 1,...,4 do
3:     Infer preliminary nonnegative Top-k codes with 24 accelerated steps
4:     Estimate an identity-anchored Koopman operator from consecutive codes
5:     if r = 1 then construct the 16-neighbor predictive feature graph
6:     Apply Laplacian and bidirectional Koopman inference without final consensus
7:     Take the curvature-scaled tangent dictionary step of size 0.10
8:     Normalize every dictionary column
9: end for
10: Infer full temporal codes on the observed fitting corpus
11: Compute innovation gates and the weighted Koopman estimate with ridge 1.5
12: Infer final target codes with Algorithm 2
13: Return D, K, codes Z, and reconstructions ZD^T
```

```text
Algorithm 2: Temporal code inference for one trajectory
Input: X, D, K, graph Laplacian L, affinity A, cardinality k
1: Initialize Z = TopK(max(XD, 0), k); set Y = Z
2: for i = 1,...,24 do
3:     Take one reconstruction-plus-graph gradient step with l1 threshold 0.07
4:     Project every row onto the nonnegative Top-k set
5:     Apply accelerated momentum and restart when displacement disagrees
6: end for
7: Solve (I + 2 Delta^T Delta)B = Z by tridiagonal elimination
8: Apply two alternating forward/backward nonnegative Koopman block sweeps
9: Project B onto nonnegative Top-k supports
10: Compute clipped innovation gates from the projected codes
11: Add graph-neighbor evidence to the standardized unary scores
12: Perform three forward/backward Top-k support-coordinate sweeps
13: Perform six projected amplitude steps of scale 0.01 on the fixed support
14: Return nonnegative sparse codes
```

### 3.10 Complexity

Let $N=\sum_sT_s$ be the number of observed frames, $Q$ the number of trajectories, $R=4$ outer rounds, $q_f=24$ accelerated steps, $q_T=2$ Koopman sweeps, $q_c=3$ consensus sweeps, and $q_a=6$ debiasing steps. One trajectory requires

$$
O\!\left(q_fT(dm+m^2)+q_TTm^2+q_cTm+q_aTdm+m^3\right)
$$

time. The $m^3$ term comes from dense endpoint/interior Koopman block inverses; the tridiagonal temporal solve costs only $O(Tm)$. Graph construction and Koopman covariance accumulation cost $O(Nm^2)$, the Koopman solve costs $O(m^3)$, and one dictionary update costs $O(Ndm+dm^2)$. Thus fitting has the explicit bound

$$
O\!\left(Rq_fN(dm+m^2)+Rq_TNm^2+(RQ+1)m^3+Rdm^2\right),
$$

up to lower-order consensus, debiasing, and Top-$k$ selection terms. Peak working storage is

$$
O\!\left(N(d+m)+dm+m^2\right),
$$

covering observed trajectories, codes, dictionary parameters, and dense graph and dynamics matrices.

## Experiments
## 4 Experiments

### 4.1 Experimental design

We evaluate controlled temporal sparse recovery with eight training and eight evaluation trajectories of length $T=48$, observation dimension $d=32$, dictionary width $m=64$, and inferred cardinality $k=5$. The generating dictionary has unit-norm columns. Latents evolve under a stable four-block operator formed from $0.82I$ plus blockwise skew-symmetric interactions, with possible semantic-block changes every 16 steps. Standard observations contain isotropic Gaussian noise with standard deviation 0.14, and the initial dictionary is perturbed with standard deviation 0.18.

The primary score is

$$
\mathcal{L}_{\mathrm{comp}}=\mathcal{L}_{\mathrm{recon}}+0.35(1-F_1)+0.25\mathcal{L}_{\mathrm{rollout}}+0.15\mathcal{V}_{\mathrm{sparse}},
$$

where reconstruction and rollout terms are mean-squared errors, $F_1$ measures aligned latent-support recovery, and $\mathcal{V}_{\mathrm{sparse}}$ measures excess cardinality. Lower composite, reconstruction, and rollout values are better; higher support F1 is better. We compare with a 20-step framewise nonnegative ISTA SAE, a 20-step framewise TopK SAE with dictionary refinement, and a matched-budget Laplacian temporal SAE.

For each comparator, paired differences are $d_s=\mathcal{L}^{\mathrm{base}}_s-\mathcal{L}^{\mathrm{ours}}_s$. We report $\bar d\pm t_{0.975,7}s_d/\sqrt{8}$ with $t_{0.975,7}=2.3646$.

### 4.2 Main benchmark comparison

| Method | Composite loss ↓ | Clean reconstruction ↓ | Support F1 ↑ | Rollout ↓ | Paired gain of ours, 95% CI |
|---|---:|---:|---:|---:|---:|
| Framewise ISTA SAE | 0.263820 | 0.009973 | 0.3133 | 0.010700 | $0.044770\pm0.013287$ |
| Framewise TopK SAE | 0.282364 | 0.015424 | 0.2506 | 0.018545 | $0.063313\pm0.013787$ |
| Laplacian temporal SAE | 0.326897 | 0.008386 | 0.3156 | 0.009636 | $0.107846\pm0.012794$ |
| **Proposed method** | **0.219051** | **0.006932** | **0.3992** | **0.007377** | — |

The method reduces composite loss by 17.0%, 22.4%, and 33.0% relative to the three comparators. The paired interval lower bounds are 0.03148, 0.04953, and 0.09505, so every improvement is positive at 95% confidence. Reconstruction error falls by 30.5%, 55.1%, and 17.3%; rollout error falls by 31.1%, 60.2%, and 23.4%; and support F1 rises by 0.0859, 0.1486, and 0.0836. Dictionary recovery reaches 0.7212 mean maximum cosine similarity, and activity is exactly five features per frame with zero sparsity violation.

### 4.3 Independent test results

| Method | Composite loss ↓ | Clean reconstruction ↓ | Support F1 ↑ | Rollout ↓ | Paired gain of ours, 95% CI |
|---|---:|---:|---:|---:|---:|
| Framewise ISTA SAE | 0.266965 | 0.008729 | 0.2789 | 0.009027 | $0.029036\pm0.014179$ |
| Framewise TopK SAE | 0.273056 | 0.014012 | 0.2715 | 0.016321 | $0.035128\pm0.012871$ |
| Laplacian temporal SAE | 0.302188 | 0.007927 | 0.3125 | 0.008343 | $0.064260\pm0.026372$ |
| **Proposed method** | **0.237929** | **0.005913** | **0.3414** | **0.006087** | — |

The test loss reductions are 10.9%, 12.9%, and 21.3%. The respective paired intervals are $[0.01486,0.04322]$, $[0.02226,0.04800]$, and $[0.03789,0.09063]$, all strictly positive. Relative to framewise ISTA, reconstruction and rollout errors decrease by 32.3% and 32.6%, while support F1 increases by 0.0625. Relative to framewise TopK, reconstruction and rollout decrease by 57.8% and 62.7%. The learned dictionary reaches 0.7317 cosine recovery, and exact five-feature activity with zero violation is preserved.

### 4.4 Independent higher-noise study

| Method | Composite loss ↓ | Clean reconstruction ↓ | Support F1 ↑ | Rollout ↓ | Paired gain of ours, 95% CI |
|---|---:|---:|---:|---:|---:|
| Framewise ISTA SAE | 0.315199 | 0.010262 | 0.2496 | 0.010953 | $0.068200\pm0.013995$ |
| Framewise TopK SAE | 0.304154 | 0.021282 | 0.2085 | 0.023335 | $0.057155\pm0.012667$ |
| Laplacian temporal SAE | 0.399379 | 0.008947 | 0.2483 | 0.009637 | $0.152380\pm0.016279$ |
| **Proposed method** | **0.246999** | **0.006589** | **0.3179** | **0.006706** | — |

Increasing observation noise by 25%, from 0.14 to 0.175, leaves all paired intervals strictly positive: $[0.05421,0.08220]$, $[0.04449,0.06982]$, and $[0.13610,0.16866]$. Composite loss improves by 21.6%, 18.8%, and 38.2%. Relative to framewise ISTA, reconstruction decreases by 35.8%, support F1 rises by 0.0683, and rollout error decreases by 38.8%.

### 4.5 Complete temporal factorial isolation

The $2\times2$ study holds dictionary refinement off while independently varying Laplacian and Koopman temporal mechanisms. The complete system is then added to quantify integration with dictionary, graph, consensus, and amplitude refinement.

| Laplacian | Koopman | Integrated refinement | Standard loss ↓ | Test loss ↓ | Standard F1 ↑ | Test F1 ↑ |
|---|---|---|---:|---:|---:|---:|
| No | No | No | 0.263499 | 0.274162 | 0.2994 | 0.2627 |
| No | Yes | No | 0.260418 | 0.270896 | 0.3031 | 0.2668 |
| Yes | No | No | 0.244018 | 0.255789 | 0.3477 | 0.3072 |
| Yes | Yes | No | 0.243229 | 0.255928 | 0.3503 | 0.3069 |
| Yes | Yes | Yes | **0.219051** | **0.237929** | **0.3992** | **0.3414** |

On the standard suite, Laplacian smoothing alone reduces loss by 0.01948, or 7.39%, and raises support F1 by 0.0483. Koopman inference alone reduces loss by 0.00308, or 1.17%, and lowers rollout error by 20.4%. Their combination reduces loss by 7.69% relative to the temporally uncoupled condition. Full integration contributes a further 0.02418 reduction, or 9.94%, and raises support F1 by 0.0489.

The independent test suite confirms the principal mechanism pattern. Laplacian smoothing alone reduces loss by 6.70%, Koopman inference alone by 1.19%, and both mechanisms together by 6.65% relative to the uncoupled condition. Full integration then reduces the two-mechanism loss by 0.01800, or 7.03%, and raises support F1 from 0.3069 to 0.3414. The 0.000139 difference between Laplacian-only and joint temporal inference is negligible compared with the integrated 0.01800 gain, showing that the strongest test improvement arises from coupling temporal inference to refined dictionaries, graph structure, and discrete support processing.

### 4.6 Component subtraction

| Removed component | Composite loss ↓ | Increase from full ↓ | Clean reconstruction ↓ | Support F1 ↑ | Rollout ↓ |
|---|---:|---:|---:|---:|---:|
| None | **0.219051** | **0.000000** | 0.006932 | **0.3992** | 0.007377 |
| Support consensus | 0.219240 | 0.000189 | 0.006930 | 0.3987 | 0.007385 |
| Innovation gate | 0.219093 | 0.000042 | 0.006930 | 0.3991 | 0.007343 |
| Transition-weighted dynamics | 0.219091 | 0.000041 | **0.006928** | 0.3991 | **0.007342** |
| Fixed-support amplitude debiasing | 0.219313 | 0.000262 | 0.006981 | 0.3986 | 0.007391 |
| Graph-neighbor support evidence | 0.219167 | 0.000117 | 0.006931 | 0.3989 | 0.007380 |

Every component subtraction increases the primary loss. Fixed-support amplitude debiasing supplies the largest isolated reduction, 0.000262, and lowers reconstruction error by $4.92\times10^{-5}$. Support consensus contributes 0.000189 and improves support F1. Graph-neighbor evidence contributes 0.000117, confirming that the predictive graph assists discrete selection in addition to continuous smoothing. Innovation gating and transition-weighted dynamics contribute smaller positive reductions of 0.000042 and 0.000041, respectively.

### 4.7 Sequence, noise, and sparsity scaling

Each sweep changes one axis while retaining eight trajectories, 32-dimensional observations, 64 dictionary atoms, and the same algorithmic hyperparameters.

| Axis | Setting | Composite loss ↓ | Clean reconstruction ↓ | Support F1 ↑ | Rollout ↓ |
|---|---:|---:|---:|---:|---:|
| Sequence length | 24 | **0.215961** | 0.008603 | **0.4136** | 0.008413 |
| Sequence length | 48 | 0.219051 | 0.006932 | 0.3992 | 0.007377 |
| Sequence length | 96 | 0.237730 | **0.005689** | 0.3415 | **0.006261** |
| Noise standard deviation | 0.07 | **0.212142** | **0.005914** | **0.4154** | **0.006517** |
| Noise standard deviation | 0.14 | 0.219051 | 0.006932 | 0.3992 | 0.007377 |
| Noise standard deviation | 0.21 | 0.258148 | 0.008825 | 0.2940 | 0.008920 |
| Active features | 3 | 0.242696 | **0.004785** | 0.3237 | **0.004872** |
| Active features | 5 | **0.219051** | 0.006932 | **0.3992** | 0.007377 |
| Active features | 7 | 0.248552 | 0.006869 | 0.3145 | 0.006973 |

Longer sequences improve dictionary cosine recovery from 0.6958 at length 24 to 0.7661 at length 96, reconstruction from 0.00860 to 0.00569, and rollout from 0.00841 to 0.00626. Length 48 gives the strongest balance between continuous accuracy and support identification. Doubling noise from 0.07 to 0.14 changes composite loss by only 0.00691. At standard deviation 0.21, inference remains finite and cardinality remains exact. Across requested budgets three, five, and seven, observed activity equals the requested budget and sparsity violation remains zero; five active features give the best composite loss and support F1.

### 4.8 Computational profile

| Method | Composite loss ↓ | Runtime over eight trajectories | Peak traced memory | Relative runtime |
|---|---:|---:|---:|---:|
| Lightweight 15-step framewise reference | 0.255520 | 5.00 ms | 0.409 MB | $1.00\times$ |
| Proposed method | **0.219051** | 1373.04 ms | 2.665 MB | $274.6\times$ |

The complete method improves the composite score by 14.3% relative to this lightweight reference while using 6.52 times its peak traced memory. Its 1.373-second runtime includes four dictionary rounds, graph construction, temporal inference, support consensus, innovation-weighted dynamics, and reconstruction. The independent test run takes 1.372 seconds and peaks at 2.665 MB, demonstrating stable computational behavior across independently generated data.

## Limitations
## 5 Limitations

The experiments use controlled temporal sparse-coding systems. This design isolates reconstruction, support recovery, dynamics, and cardinality, but it does not establish semantic interpretability or downstream utility on real language-model activations, video representations, or clinical trajectories.

The method uses observed target trajectories during unsupervised dictionary and dynamics fitting. It never accesses clean targets or latent labels during inference, and the independent test suite is not used for model selection, but the resulting protocol is transductive rather than a frozen inductive evaluation. Deployments requiring a frozen model should fit the dictionary, graph, and Koopman operator exclusively on a separate training corpus.

Dense feature-graph and Koopman operations scale quadratically in dictionary width, and the current direct matrix operations introduce cubic terms. The 64-atom experiments remain small; substantially wider SAEs will require sparse graph multiplication, cached factorizations, low-rank or block-sparse dynamics, and iterative linear solvers. The empirical runtime is approximately 275 times that of a lightweight framewise reference, although it remains 1.37 seconds for the complete eight-trajectory workload.

A single global Koopman operator cannot fully represent heterogeneous multibasin dynamics. Innovation gating reduces the influence of poorly predicted transitions but does not replace a mixture of local operators. Hard Top-$k$ projection and finite support-coordinate sweeps also make the complete problem nonconvex. The guarantees apply to the stated graph, temporal, dynamics, support, debiasing, and normalization subproblems rather than global recovery of the generating dictionary.

Support F1 declines at the highest tested noise and at sequence length 96 even as reconstruction, rollout, and dictionary alignment improve. Longer trajectories provide more continuous estimation data but can accumulate ambiguous support transitions. Local dynamics or adaptively calibrated gates may address this divergence.

The isolated improvements from innovation gating and transition weighting are positive but small compared with the integrated gain. The component subtractions do not include paired uncertainty intervals, while comparator intervals reflect variation across only eight trajectories. Broader studies should use many independently generated dictionaries, report regime-boundary and stationary-interval results separately, and test sensitivity to graph density, ridge strength, consensus depth, numerical precision, and hardware.

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
| initial_subset | 0.226639932 |
| initial_full | 0.223042136 |
| seed0_subset | 0.230739614 |
| seed0_full | 0.219370299 |
| evolved0_subset | 0.230430148 |
| evolved0_full | 0.219050735 |
| ablation0 | 0.219239609 |
| ablation1 | 0.219092751 |
| ablation2 | 0.219091298 |
| ablation3 | 0.219312924 |
| ablation4 | 0.219167282 |
| rebuttal0 | 0.246998768 |
| final_test | 0.237928835 |
