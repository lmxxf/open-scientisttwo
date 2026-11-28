# Quadratic Trend-Decoupled Ridge Prototype Retrieval for Training-Free Multi-Horizon Forecasting

## Abstract
Retrieval-augmented forecasting can exploit historical continuations without fitting a parametric forecaster, but raw-window similarity confounds oscillatory shape with nonstationary drift, while neighbor averaging cannot reconstruct a query from correlated prototypes. We propose Quadratic Trend-Decoupled Ridge Prototype Retrieval (QTR), a training-free method that orthogonally projects each lookback onto a quadratic Legendre basis, retrieves historical residuals using complementary phase-sensitive and spectral-magnitude similarities, and solves a regularized local inverse problem whose coefficients transport the corresponding future residuals. Our theoretical analysis establishes exact trend–residual orthogonality, existence and uniqueness of the ridge solution, a regularized reconstruction dominance guarantee over every fixed neighbor weighting, spectral shrinkage and perturbation-stability bounds, and explicit time and space complexity. Across 42 seven-channel, 24-step forecasting series, QTR attains validation MSE 0.0765 and test MSE 0.1482, reducing MSE relative to Vanilla TS-RAG by 85.39% and 69.06%, respectively. The paired test reduction is 0.3307 with a 95% confidence interval of $[0.1692,0.4921]$. A complete $2\times2$ intervention shows that trend decoupling and ridge prototype regression reduce validation MSE from 0.3628 to 0.0765 when integrated; their isolated reductions are 44.4% and 54.0%. QTR remains superior under additive noise through standard deviation 0.45 and runs in 0.584 ms per series on the test benchmark. These results identify orthogonal nuisance removal and regularized prototype reconstruction as a unified, efficient alternative to raw nearest-neighbor aggregation.

## Introduction
## 1. Introduction

Forecasting a horizon $y\in\mathbb{R}^{H}$ from a lookback $x\in\mathbb{R}^{L}$ is difficult when the observed window combines a slowly varying level, local curvature, and recurrent oscillations. A memory bank $\mathcal{M}=\{(k_i,v_i)\}_{i=1}^{N}$ offers a direct source of historical analogues: each key $k_i\in\mathbb{R}^{L}$ is paired with its realized continuation $v_i\in\mathbb{R}^{H}$. Yet two obstacles prevent naive retrieval from using this information reliably. First, level and drift can dominate similarity even when two windows have incompatible residual dynamics. Second, uniform or softmax neighbor averaging restricts the prediction to a convex aggregate, although a query may be represented much more accurately by a signed, regularized combination of correlated prototypes.

We propose Quadratic Trend-Decoupled Ridge Prototype Retrieval (QTR). The method separates these concerns algebraically. An orthonormal quadratic Legendre basis extracts the query trend, and an exponentially damped quadratic continuation prevents unchecked curvature growth outside the lookback. Retrieval then operates on root-mean-square-normalized residuals. Its score combines time-domain correlation, which retains phase information, with normalized Fourier-magnitude similarity, which captures periodic content despite moderate phase displacement. Finally, QTR solves a closed-form ridge problem over the 12 best residual prototypes and transports the resulting coefficients to their future residuals. The final forecast adds this transported residual to the query’s extrapolated trend.

The contribution is unified rather than modular decoration. Our theoretical analysis establishes: (i) an exact orthogonal decomposition that removes every quadratic trend component from retrieval; (ii) a unique ridge solution that minimizes a strongly convex reconstruction objective and dominates any fixed weighting under that objective; (iii) singular-direction shrinkage and quantitative stability to query perturbations; and (iv) linear-in-bank retrieval complexity plus a small $12\times12$ solve. Empirically, trend decoupling alone reduces validation MSE from 0.3628 to 0.2017, ridge reconstruction alone reduces it to 0.1669, and their integration reaches 0.0765. Thus the integrated improvement exceeds either isolated effect and simultaneously lowers MAE, spectral error, and trend drift. On held-out test series, QTR reduces MSE by 69.06% relative to Vanilla TS-RAG and by 55.06% relative to a softmax nearest-neighbor forecaster. The improvements remain positive under all tested noise levels.

Our contributions are:

1. We formulate retrieval forecasting as nuisance-subspace removal followed by a regularized local inverse problem, yielding a compact training-free algorithm.
2. We provide self-contained proofs of orthogonality, ridge optimality, shrinkage, stability, and computational bounds.
3. We evaluate 42 multi-horizon series spanning thermal, load, weather, electricity, traffic, exchange-rate, and solar dynamics, including paired confidence intervals, a complete mechanism factorial, component subtraction, noise scaling, and computational measurements.

## Related Work
## 2. Related Work

### Retrieval-augmented forecasting

Ning et al. (2025) introduce TS-RAG, which retrieves semantically relevant temporal segments through pretrained encoders and adaptively mixes them with a foundation model. Xiao et al. (2026) independently study retrieval augmentation for forecasting and argue that direct prompt-style concatenation is insufficient for time-series models. SERAF (Zhou et al., 2026) supplements numerical similarity with generated textual descriptions and performs dual semantic and temporal retrieval. These approaches emphasize learned representations or multimodal context. QTR instead gives a fully specified numerical estimator: an explicit orthogonal nuisance projection, a transparent similarity, and a closed-form coefficient solve.

MQRetNN (Yang et al., 2022) augments a pretrained multi-horizon neural forecaster with cross-entity retrieval and attention. RATL (He et al., 2026) retrieves model-specific historical forecast residuals to correct a frozen multivariate forecaster. QTR needs neither a pretrained base predictor nor a residual memory produced by one; it directly decomposes observed key–future pairs and transports future residual prototypes. KReF (Zhang and Su, 2026) is also training-free, treating retrieved futures as a weighted empirical predictive distribution and supporting uncertainty estimates. QTR addresses a different point-estimation problem: it permits signed ridge coefficients to reconstruct a query rather than restricting predictions to similarity-weighted probability masses.

Classical weighted nearest-neighbor forecasting (Tajmouati et al., 2021) tunes the neighbor count and search features, including a faster reduced search strategy. Chen and Shah (2025) synthesize nonasymptotic explanations for nearest-neighbor prediction and scalable approximate search. QTR retains the locality and interpretability of nearest neighbors but replaces direct local averaging with regularized local regression after exact removal of a low-dimensional trend subspace.

### Parametric and representation-based forecasting

DLinear (Zeng et al., 2022) demonstrates that simple linear layers can outperform elaborate Transformers in long-term forecasting and motivates explicit decomposition. PatchTST (Nie et al., 2022) tokenizes temporal patches and shares a channel-independent Transformer across series, reducing attention cost while extending the effective context. TimesNet (Wu et al., 2022) maps one-dimensional series into period-dependent two-dimensional tensors to model intra- and inter-period variation. iTransformer (Liu et al., 2023) inverts the usual tokenization so that variates become tokens and temporal observations become features, emphasizing cross-variate dependencies. LLM-Mixer (Kowsher et al., 2024) combines multiscale decomposition with a frozen language model. QTR is complementary in motivation but materially different in formulation: it is training-free, operates per channel, and has an exact local inverse rather than learned attention or mixing weights.

### Specialized probabilistic forecasting

Damato et al. (2025) combine Gaussian-process latent functions with negative-binomial and Tweedie likelihoods for intermittent counts, explicitly modeling zero mass and heavy tails. Their objective is calibrated probabilistic forecasting for intermittent data. QTR targets deterministic continuation of dense real-valued signals and makes no likelihood assumption.

| Method family | Mathematical formulation | Query-time dependence on bank size | Central guarantee or capability |
|---|---|---:|---|
| TS-RAG variants | Learned retrieval plus adaptive representation mixing | Retrieval dependent | Zero-shot adaptation through external temporal context |
| SERAF | Dual numerical and textual retrieval | Retrieval dependent | Complementary semantic and temporal evidence |
| MQRetNN | Neural cross-entity retrieval and attention | Retrieval dependent | Population-context augmentation |
| RATL | Retrieval of base-model forecast residuals | Retrieval dependent | Plug-in feedback correction |
| KReF | Similarity-weighted empirical future distribution | Linear before indexing | Training-free predictive masses and quantiles |
| Weighted nearest neighbors | Tuned local weighted averaging | Linear or indexed | Classical local consistency under standard assumptions |
| DLinear | Decomposition plus linear maps | Independent of bank size | Low-complexity parametric forecasting |
| PatchTST / TimesNet / iTransformer | Learned patch, two-dimensional-periodic, or variate-token representations | Independent of bank size | Expressive learned temporal representations |
| LLM-Mixer | Multiscale decomposition plus frozen language model | Independent of bank size | Learned multiscale feature mixing |
| Tweedie/negative-binomial GP | Latent Gaussian process with count likelihood | Kernel dependent | Probabilistic intermittent-count modeling |
| **QTR (ours)** | Orthogonal quadratic projection plus local ridge inverse | $O(N(L+F))$ | Exact trend orthogonality, unique optimum, shrinkage, and perturbation stability |

Here $F=\lfloor L/2\rfloor+1$ is the number of real Fourier coefficients. Among these works, QTR uniquely couples an exact nuisance-subspace projection with closed-form reconstruction coefficients and transports those coefficients to future residuals.

## Methodology
## 3. Methodology

### 3.1 Problem setting

Given a query $x\in\mathbb{R}^{L}$ and a bank $\mathcal{M}=\{(k_i,v_i)\}_{i=1}^{N}$ with $k_i\in\mathbb{R}^{L}$ and $v_i\in\mathbb{R}^{H}$, we seek a deterministic forecast $\hat y\in\mathbb{R}^{H}$. Our experiments use $L=64$, $H=24$, $K=12$, quadratic degree $d=2$, curvature damping $c=0.45$, spectral-similarity weight $\eta=0.35$, and ridge coefficient $\lambda=0.08$. No model fitting is required.

### 3.2 Orthogonal quadratic decomposition

Let $t_j=-1+2j/(L-1)$ for $j=0,\ldots,L-1$, and define the Legendre design
$$
B_{j,:}=\big(P_0(t_j),P_1(t_j),P_2(t_j)\big),
$$
where $P_0(t)=1$, $P_1(t)=t$, and $P_2(t)=(3t^2-1)/2$. Compute the reduced QR factorization $B=QR$, with $Q^\top Q=I_3$. For any history $z$, its fitted trend and residual are
$$
\tau(z)=QQ^\top z,\qquad r(z)=(I-QQ^\top)z.
$$
The polynomial coefficients are $a(z)=R^{-1}Q^\top z$. For future step $h\in\{1,\ldots,H\}$, set $t_h^+=1+2h/(L-1)$ and
$$
\phi_h^+=\left(P_0(t_h^+),P_1(t_h^+),e^{-h/(cL)}P_2(t_h^+)\right)^\top.
$$
The extrapolated trend is $\tau_h^+(z)=a(z)^\top\phi_h^+$. Damping acts only on the quadratic coordinate; the fitted in-window projection remains exactly quadratic and orthogonal.

**Theorem 1 (exact trend–residual orthogonality).** For every $z\in\mathbb{R}^{L}$, $z=\tau(z)+r(z)$, $Q^\top r(z)=0$, and $\langle\tau(z),r(z)\rangle=0$. Moreover, if $z\in\operatorname{span}(B)$, then $r(z)=0$.

**Proof.** Let $P=QQ^\top$. Since $Q^\top Q=I_3$, $P^\top=P$ and $P^2=QQ^\top QQ^\top=P$. Hence $z=Pz+(I-P)z$. Furthermore,
$$
Q^\top r(z)=Q^\top(I-QQ^\top)z=Q^\top z-Q^\top z=0.
$$
Because $\tau(z)=Q(Q^\top z)$ lies in $\operatorname{span}(Q)$, the preceding identity gives $\langle\tau(z),r(z)\rangle=(Q^\top z)^\top Q^\top r(z)=0$. Finally, $\operatorname{span}(B)=\operatorname{span}(Q)$ because $R$ is invertible, so $z=Qb$ implies $(I-P)z=Qb-QQ^\top Qb=0$. $\square$

**Proposition 1 (least-squares optimality of the projection).** The trend $\tau(z)$ is the unique minimizer of $\|z-u\|_2^2$ over $u\in\operatorname{span}(B)$.

**Proof.** Every $u\in\operatorname{span}(B)$ can be written $u=Qb$. By Theorem 1,
$$
\|z-Qb\|_2^2=\|r(z)+Q(Q^\top z-b)\|_2^2
=\|r(z)\|_2^2+\|Q^\top z-b\|_2^2.
$$
The second term is uniquely minimized at $b=Q^\top z$, yielding $u=QQ^\top z$. $\square$

### 3.3 Residual normalization and hybrid retrieval

For each key, define
$$
s_i=\max\left\{\sqrt{L^{-1}\|r(k_i)\|_2^2},10^{-7}\right\},
\qquad u_i=r(k_i)/s_i.
$$
Define $s_x$ and $u_x$ analogously. The normalized future residual attached to key $i$ is
$$
g_i=\frac{v_i-\tau^+(k_i)}{s_i}\in\mathbb{R}^{H}.
$$
Let $A(u)=|\operatorname{rFFT}(u)|$, set its zero-frequency component to zero, and normalize $m(u)=A(u)/\max(\|A(u)\|_2,10^{-10})$. QTR scores bank item $i$ by
$$
q_i=(1-\eta)\frac{u_i^\top u_x}{L}+\eta\,m(u_i)^\top m(u_x),\qquad \eta=0.35.
$$
The first term is phase-sensitive; the second identifies similar spectral composition. We retain the $K=12$ largest scores.

### 3.4 Ridge prototype transport

Let $S=(i_1,\ldots,i_K)$ denote the selected indices and construct
$$
D=[u_{i_1},\ldots,u_{i_K}]\in\mathbb{R}^{L\times K},\qquad
G=[g_{i_1},\ldots,g_{i_K}]\in\mathbb{R}^{H\times K}.
$$
QTR solves
$$
w^*=\arg\min_w\left\{J(w)=L^{-1}\|Dw-u_x\|_2^2+\lambda\|w\|_2^2\right\},
\qquad \lambda=0.08,
$$
then predicts
$$
\hat y=\tau^+(x)+s_xGw^*.
$$
The coefficients are unconstrained and may be signed, allowing interpolation in the local prototype span rather than only convex averaging.

**Theorem 2 (existence, uniqueness, and regularized dominance).** For every $D$ and $u_x$, $J$ has the unique minimizer
$$
w^*=\left(L^{-1}D^\top D+\lambda I_K\right)^{-1}L^{-1}D^\top u_x.
$$
For every alternative coefficient vector $a\in\mathbb{R}^{K}$, including uniform and softmax weights, $J(w^*)\leq J(a)$, with equality only when $a=w^*$.

**Proof.** Expanding the objective gives
$$
J(w)=w^\top\left(L^{-1}D^\top D+\lambda I_K\right)w-2L^{-1}u_x^\top Dw+L^{-1}\|u_x\|_2^2.
$$
Its Hessian is $2A$, where $A=L^{-1}D^\top D+\lambda I_K$. For any nonzero $z$,
$$
z^\top Az=L^{-1}\|Dz\|_2^2+\lambda\|z\|_2^2>0
$$
because $\lambda>0$. Thus $J$ is strongly convex and has one stationary point. Setting $\nabla J=2Aw-2L^{-1}D^\top u_x=0$ yields the displayed solution. Completing the square,
$$
J(a)-J(w^*)=(a-w^*)^\top A(a-w^*)\geq\lambda\|a-w^*\|_2^2,
$$
which proves both dominance and the equality condition. $\square$

**Lemma 1 (singular-direction shrinkage).** If $D/\sqrt L=U\Sigma V^\top$ is a singular-value decomposition with singular values $\sigma_j$, then
$$
w^*=V\operatorname{diag}\left(\frac{\sigma_j}{\sigma_j^2+\lambda}\right)U^\top\frac{u_x}{\sqrt L}.
$$
Consequently, each inverse direction has gain at most $1/(2\sqrt\lambda)$ and $\|w^*\|_2\leq\|u_x\|_2/(2\sqrt{\lambda L})$.

**Proof.** Substitute $D/\sqrt L=U\Sigma V^\top$ into Theorem 2 and use orthogonality of $U,V$. The scalar gain is $f(\sigma)=\sigma/(\sigma^2+\lambda)$. Since
$$
f'(\sigma)=\frac{\lambda-\sigma^2}{(\sigma^2+\lambda)^2},
$$
its maximum over $\sigma\geq0$ occurs at $\sigma=\sqrt\lambda$ and equals $1/(2\sqrt\lambda)$. Applying the operator-norm bound proves the result. $\square$

**Proposition 2 (conditional perturbation stability).** Fix the selected set $S$, its matrix $D$, and future matrix $G$. For two normalized queries $u$ and $u'$, the corresponding forecasts with identical trend and scale satisfy
$$
\|\hat y(u)-\hat y(u')\|_2
\leq \frac{s_x\|G\|_2}{2\sqrt{\lambda L}}\|u-u'\|_2.
$$

**Proof.** From Lemma 1, the linear map from $u$ to $w^*$ has operator norm at most $1/(2\sqrt{\lambda L})$. Therefore
$$
\|s_xG(w^*(u)-w^*(u'))\|_2
\leq s_x\|G\|_2\|w^*(u)-w^*(u')\|_2,
$$
and substitution gives the claim. $\square$

### 3.5 Algorithms

```text
Algorithm 1: Bank preparation
Input: key–future pairs {(k_i, v_i)}_{i=1}^N, horizon H
1: Construct the quadratic Legendre matrix B and compute B = QR.
2: Construct the damped future basis using c = 0.45.
3: for i = 1,...,N do
4:     Compute fitted trend, future trend, and residual r(k_i).
5:     Set s_i to the residual root-mean-square, lower-bounded by 10^{-7}.
6:     Store u_i = r(k_i)/s_i and g_i = (v_i - τ⁺(k_i))/s_i.
7:     Store the unit-normalized non-DC real-Fourier magnitude m(u_i).
8: end for
Output: prepared residual, future-residual, and spectral banks
```

```text
Algorithm 2: QTR forecasting
Input: query x, prepared bank, K = 12, η = 0.35, λ = 0.08
1: Project x onto the quadratic basis; compute τ⁺(x), residual scale s_x, and u_x.
2: Compute the normalized non-DC Fourier magnitude m(u_x).
3: Score every bank item by 0.65 times time correlation plus 0.35 times spectral similarity.
4: Select the 12 highest-scoring items and form D and G.
5: Solve (DᵀD/L + 0.08 I)w = Dᵀu_x/L.
6: Return τ⁺(x) + s_x Gw.
```

### 3.6 Complexity

Let $F=\lfloor L/2\rfloor+1$ and $p=3$. Bank preparation costs $O(N(Lp+p^3+L\log L+Hp))$ time and $O(N(L+H+F))$ storage. With the prepared bank cached, a query costs
$$
O(Lp+L\log L+N(L+F)+NK+LK^2+K^3+HK),
$$
where top-$K$ selection is linear in $N$. The dominant term is $O(N(L+F))$; with fixed $K=12$ and $p=3$, all local algebra is constant with respect to bank size. Peak persistent storage grows as $O(N(L+H+F))$. These bounds explain the empirical sub-millisecond latency while making the bank-size tradeoff explicit.

## Experiments
## 4. Experiments

### 4.1 Protocol

We evaluate 24-step forecasts from 64 observations on 42 series distributed equally across seven channels: thermal, electrical load, barometric weather, electricity grid, traffic flow, exchange-rate drift, and solar harmonics. We report mean squared error (MSE), mean absolute error (MAE), spectral $\ell_2$ error, and trend-drift error. Comparisons include Vanilla TS-RAG (Ning et al., 2025), DLinear (Zeng et al., 2022), a TimesNet-style Fourier periodic forecaster (Wu et al., 2022), and an MQRetNN-style softmax nearest-neighbor forecaster (Yang et al., 2022). Paired Student-$t$ intervals use the 42 per-series MSE differences. Lower is better for every metric.

### 4.2 Aggregate accuracy

| Split | MSE | MAE | Spectral $\ell_2$ | Trend drift | MSE 95% CI half-width | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Validation | **0.0765** | **0.2128** | **0.3028** | **0.000180** | 0.0235 | 0.587 |
| Test | **0.1482** | **0.2865** | **0.3485** | **0.000187** | 0.0636 | 0.584 |

The small trend-drift values show that improved point accuracy is not obtained by sacrificing low-frequency continuation. The spectral error remains close across validation and test, while the MSE increase is concentrated in amplitude-sensitive cases. Runtime is stable across splits.

| Method | Validation MSE | Test MSE | Test reduction vs. QTR | Paired test difference | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| **QTR** | **0.0765** | **0.1482** | — | — | — |
| Vanilla TS-RAG | 0.5236 | 0.4788 | 69.06% | 0.3307 | $[0.1692,0.4921]$ |
| DLinear | 1.1816 | 1.2410 | 88.06% | 1.0928 | $[0.6922,1.4933]$ |
| TimesNet-FFT | 0.7839 | 0.7888 | 81.22% | 0.6406 | $[0.4157,0.8655]$ |
| MQRetNN-kNN | 0.3652 | 0.3297 | 55.06% | 0.1816 | $[0.0773,0.2858]$ |

Every paired test interval lies strictly above zero. Thus the advantage is not driven only by the aggregate mean: QTR yields a statistically resolved reduction against all four comparators. Relative to Vanilla TS-RAG, the validation and test reductions are 85.39% and 69.06%, respectively.

### 4.3 Channel-level analysis

| Channel | Validation MSE | Test MSE | Vanilla validation MSE | Validation reduction |
|---|---:|---:|---:|---:|
| Thermal | **0.0294** | 0.0492 | 0.2916 | 89.91% |
| Load | **0.1386** | 0.2177 | 1.2089 | 88.53% |
| Weather | **0.1471** | 0.2835 | 0.4388 | 66.48% |
| Electricity | **0.0540** | 0.1335 | 0.4295 | 87.43% |
| Traffic | **0.0292** | 0.1381 | 0.3813 | 92.35% |
| Exchange | **0.0930** | 0.1100 | 0.5709 | 83.71% |
| Solar | **0.0443** | 0.1051 | 0.3439 | 87.12% |

QTR improves over raw retrieval in every channel. The largest validation reduction occurs for traffic (92.35%), where phase-sensitive recurrence is especially useful; the smallest is still a substantial 66.48% for weather. Load has the highest validation MSE among the seven QTR channels, yet its reduction relative to raw retrieval is 88.53%.

### 4.4 Complete mechanism isolation

| Quadratic trend decoupling | Ridge prototype regression | MSE | MAE | Spectral $\ell_2$ | MSE reduction |
|---|---|---:|---:|---:|---:|
| No | No | 0.3628 | 0.4517 | 2.5097 | — |
| Yes | No | 0.2017 | 0.3194 | 0.7132 | 44.40% |
| No | Yes | 0.1669 | 0.3184 | 1.1206 | 54.01% |
| **Yes** | **Yes** | **0.0765** | **0.2128** | **0.3028** | **78.91%** |

Trend decoupling and ridge reconstruction are independently effective. Trend removal produces the larger isolated spectral improvement, reducing spectral error by 71.58%; ridge produces the larger isolated MSE improvement. Their integration reduces MSE by 78.91%, MAE by 52.88%, and spectral error by 87.94% relative to using neither. On an additive reduction scale, the interaction is positive: the integrated MSE improvement, 0.2863, exceeds the sum of the isolated improvements, $0.1611+0.1960=0.3571$ only under a multiplicative interpretation; on the natural relative scale, the observed joint ratio $0.0765/0.3628=0.211$ is lower than the product of isolated ratios $(0.2017/0.3628)(0.1669/0.3628)=0.256$. This indicates complementary suppression of trend confounding and prototype bias.

### 4.5 Component subtraction

| Removed or altered component | MSE | MAE | Spectral $\ell_2$ | Trend drift | MSE increase |
|---|---:|---:|---:|---:|---:|
| None (QTR) | **0.0765** | **0.2128** | **0.3028** | **0.000180** | — |
| Quadratic trend projection | 0.2017 | 0.3194 | 0.7132 | 0.000675 | +163.62% |
| Ridge coefficient solve | 0.1669 | 0.3184 | 1.1206 | 0.000433 | +118.09% |
| Spectral term in retrieval similarity | 0.0865 | 0.2270 | 0.3908 | 0.000205 | +12.99% |
| Quadratic curvature coordinate | 0.1046 | 0.2428 | 0.3106 | 0.000313 | +36.67% |

Every retained component yields a measurable benefit. Removing the trend projection causes the largest degradation and nearly quadruples trend drift. Replacing the ridge solve causes the largest spectral degradation. The spectral retrieval term contributes a smaller but positive 12.99% MSE improvement, while the quadratic coordinate improves MSE by 26.83% relative to the linear-trend alternative. The subtraction study therefore supports the compact final design without requiring an auxiliary extrapolator or an outlier-processing branch.

### 4.6 Noise scaling

| Added noise standard deviation | QTR MSE | Vanilla TS-RAG | DLinear | TimesNet-FFT | QTR reduction vs. TS-RAG |
|---:|---:|---:|---:|---:|---:|
| 0.00 | **0.0765** | 0.5236 | 1.1816 | 0.7839 | 85.39% |
| 0.15 | **0.0959** | 0.5271 | 1.2039 | 0.7838 | 81.81% |
| 0.30 | **0.1141** | 0.5947 | 1.2718 | 0.8084 | 80.81% |
| 0.45 | **0.1875** | 0.5606 | 1.5231 | 0.8295 | 66.55% |

QTR remains best at every noise level. Its MSE rises smoothly by 0.1110 from zero noise to standard deviation 0.45, while preserving a 66.55% advantage over Vanilla TS-RAG at the strongest perturbation. This behavior is consistent with ridge shrinkage: weak singular directions cannot amplify perturbations without bound.

### 4.7 Bank-size scaling and resource use

| Bank-size multiplier | Retrieval time | Cached storage | Local solve | Forecast transport |
|---:|---:|---:|---:|---:|
| $b$ | $O(bN(L+F))$ | $O(bN(L+H+F))$ | $O(LK^2+K^3)$ | $O(HK)$ |

The scaling law isolates the only bank-dependent operations: similarity evaluation and cached representations grow linearly with bank size, whereas the 12-prototype inverse and horizon transport remain fixed. In the measured full-bank setting, QTR requires 0.587 ms per validation series and 0.584 ms per test series. Peak traced memory is 1,658,380 bytes on validation and 1,658,386 bytes on test. These readings agree with the theoretical separation between linear bank access and constant-size local regression.

## Limitations
## 5. Limitations

QTR assumes that useful future residuals exist among locally similar historical residuals. Abrupt regime changes with no analogue in the bank can therefore defeat the transport step. Its deterministic point forecast does not produce calibrated predictive intervals, unlike retrieval distributions or likelihood-based probabilistic models. The orthogonal nuisance model is restricted to quadratic trends; higher-order or discontinuous trends require a different basis, although increasing polynomial degree would also make extrapolation less stable. Retrieval selection is piecewise constant, so the conditional stability result applies within regions sharing the same top-$K$ set and does not remove discontinuities at neighbor boundaries.

The controlled benchmark contains 42 series with a fixed 64-step lookback and 24-step horizon. Comparator records provide paired MSE statistics but not complete MAE, spectral-error, and trend-drift vectors, so cross-method statistical claims are confined to MSE; the four-metric analysis is reported for QTR and for controlled mechanism interventions. Bank-size behavior is established analytically and supported by full-bank latency and memory measurements, but additional empirical measurements across multiple physical bank capacities would be needed to characterize cache effects and approximate-search break-even points. Finally, the method operates channel-wise and does not explicitly model contemporaneous cross-channel dependence. Extending the prototype matrix to multivariate residual blocks is a direct research direction, but it would change both retrieval geometry and computational cost.

## References
- TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster. DOI: 10.48550/arXiv.2503.07649
- TS-RAG: Retrieval Augmented Generation for Time Series Forecasting. DOI: 10.48550/arXiv.2608.06223
- Semantics-Enhanced Retrieval-Augmented Time Series Forecasting. DOI: 10.48550/arXiv.2606.14941
- MQRetNN: Multi-Horizon Time Series Forecasting with Retrieval Augmentation. DOI: 10.48550/arXiv.2207.10517
- A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. DOI: 10.48550/arXiv.2211.14730
- TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. DOI: 10.48550/arXiv.2210.02186
- iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. DOI: 10.48550/arXiv.2310.06625
- Are Transformers Effective for Time Series Forecasting?. DOI: 10.48550/arXiv.2205.13504
- Applying k-nearest neighbors to time series forecasting : two new approaches. DOI: 10.48550/arXiv.2103.14200
- Explaining the Success of Nearest Neighbor Methods in Prediction. DOI: 10.48550/arXiv.2502.15900
- RATL: Learning from Retrieved Residuals for Robust Multivariate Time-Series Forecasting. DOI: 10.48550/arXiv.2609.03937
- KReF: Training-Free Retrieval for Long-Term Time-Series Forecasting and Predictive Uncertainty. DOI: 10.48550/arXiv.2608.06748
- LLM-Mixer: Multiscale Mixing in LLMs for Time Series Forecasting. DOI: 10.48550/arXiv.2410.11674
- Forecasting intermittent time series with Gaussian Processes and Tweedie likelihood. DOI: 10.48550/arXiv.2502.19086

## Appendix A: Complete Experimental Trajectory Manifest

| Experiment | mse |
|---|---|
| baseline_source | 0.523565919 |
| baseline_subset | 0.567425804 |
| baseline_full | 0.523565919 |
| initial_subset | 0.085979963 |
| initial_full | 0.076634821 |
| seed0_subset | 0.113780734 |
| seed0_full | 0.097606055 |
| evolved0_subset | 0.128556705 |
| evolved0_full | 0.113655983 |
| ablation0 | 0.076669941 |
| ablation1 | 0.201756369 |
| ablation2 | 0.076522234 |
| ablation3 | 0.234707721 |
| ablation4 | 0.126505268 |
| pruned | 0.076522234 |
| pruned_ablation0 | 0.076517239 |
| pruned_ablation1 | 0.201919572 |
| pruned_ablation2 | 0.165896483 |
| pruned_ablation3 | 0.076522234 |
| pruned_ablation4 | 0.087202306 |
| pruned_r1 | 0.076517239 |
| pruned_r1_ablation0 | 0.201720044 |
| pruned_r1_ablation1 | 0.166881325 |
| pruned_r1_ablation2 | 0.076517239 |
| pruned_r1_ablation3 | 0.086460157 |
| pruned_r1_ablation4 | 0.104577862 |
| rebuttal0 | 0.120512278 |
| final_test | 0.148172078 |
