# Adaptive Spectral–Trend Retrieval with Stability-Aware Prototype Regression for Multi-Horizon Time-Series Forecasting

## Abstract
We propose Adaptive Spectral–Trend Retrieval (ASTR), a training-free forecaster that treats retrieval as stable reconstruction in a detrended residual subspace. ASTR suppresses isolated spikes, selects a linear, quadratic, or one-knot piecewise-linear trend by blocked prequential prediction, robustly fits the selected geometry in QR coordinates, and retrieves normalized residual trajectories using temporal and spectral compatibility. Shortlisted histories and their futures are phase-aligned, combined by generalized-cross-validation-selected ridge regression in singular-vector coordinates, and fused with five-mode Fourier continuation through an analytical reconstruction-confidence gate. Our theoretical analysis establishes bounded local influence, quantified near-orthogonality between the robust trend fit and residual, leakage-free trend selection, existence and uniqueness of the adaptive ridge solution, an explicit coefficient-sensitivity bound, exact continuation of retained on-grid harmonics, and a convex error bound for confidence-gated fusion. Experiments evaluate 42 series from seven non-stationary regimes with lookback 64 and horizon 24. On validation, ASTR obtains MSE 0.076342, MAE 0.207383, spectral error 0.259836, and trend-drift error 0.000177; on an independent test partition, it obtains MSE 0.161099. Relative to Vanilla TS-RAG, validation and test MSE decrease by 85.42% and 66.36%, respectively. Validation reductions relative to DLinear, TimesNet-FFT, and MQRetNN-kNN are 93.54%, 90.26%, and 79.10%, and all paired Student-t 95% confidence intervals remain strictly positive on both partitions. A complete factorial study shows that adaptive trend decoupling and ridge prototype regression jointly reduce MSE by 77.79% relative to using neither mechanism. Removing phase alignment increases MSE by 30.35%, while the confidence gate supplies a further 1.32% integrated reduction. At additive history-noise standard deviation 0.45, ASTR retains a 65.80% advantage over Vanilla TS-RAG. These results establish a unified algorithmic advance connecting robust decomposition, phase-aware retrieval, stable prototype regression, and analytically controlled spectral fusion.

## Introduction
## 1. Introduction

Retrieval-augmented forecasting offers a direct way to exploit historical evidence: match an observed history to similar past trajectories and transfer their continuations. Its apparent simplicity conceals three coupled geometric problems. First, similarity between raw windows can be dominated by level and drift rather than forecastable oscillation. Second, retrieved neighbors are commonly collinear, so uniform or softmax averaging cannot reconstruct a query from complementary prototypes. Third, a retrieval system needs a measurable criterion for deciding when its memory is insufficient and a structurally different predictor should contribute.

We formulate multi-horizon forecasting from a history $x∈ℝ^L$, a memory $M={(k_i,v_i)}_{i=1}^N$ with keys $k_i∈ℝ^L$ and future trajectories $v_i∈ℝ^H$, and a target $y∈ℝ^H$. We propose Adaptive Spectral–Trend Retrieval (ASTR), based on the principle that trend continuation, residual matching, prototype reconstruction, and uncertainty-sensitive fusion should be mathematically separated. The method is training-free and uses only the observed history and historical memory.

ASTR first bounds isolated sensor contamination using a four-neighbor median and robust local scale. It then selects among linear, quadratic, and one-knot piecewise-linear trends through prefix-to-next-block prediction, avoiding a fixed decomposition assumption. The selected trend is robustly fit in QR coordinates and continued with explicit slope and curvature damping. Query and memory residuals are normalized by root-mean-square amplitude, making retrieval focus on residual shape rather than scale.

Retrieval combines normalized temporal agreement with Fourier-magnitude similarity. The algorithm shortlists 32 trajectories, searches circular shifts from $-12$ to $12$, transports each selected shift across the joined historical and future residual trajectory, and retains 12 prototypes. Both the initial and post-alignment rankings use the coefficients $0.7$ for temporal similarity and $0.3$ for spectral similarity. The coefficients are used directly, without an additional normalization.

Rather than averaging the retrieved futures, ASTR solves

$$w_λ=argmin_w L^{-1}∥Rw-q∥_2^2+λ∥w∥_2^2,$$

where $q$ is the normalized query residual and the columns of $R$ are aligned retrieved histories. A positive regularization strength is selected by generalized cross-validation and the solution is computed through the singular value decomposition. The normalized reconstruction error

$$ρ=∥q-Rw_λ∥_2^2/max(∥q∥_2^2,10^{-12})$$

then determines $γ=exp(-0.2ρ)$, which fuses the retrieved continuation with a five-mode Fourier continuation.

Our theoretical contribution follows the executed computation. We show that robust trend fitting remains inside the selected trend subspace and satisfies an exact regularized normal-equation identity, giving a $10^{-10}$ near-orthogonality bound. We prove that blocked trend selection cannot access the forecast horizon, that every positive ridge candidate has a unique solution, and that the conditional sensitivity of the coefficients is at most $1/(2√{Lλ})$. We further prove exact continuation for residuals supported on the retained discrete Fourier bins and an exact convex decomposition of the gated forecast error.

Our experiments use 42 forecasts across thermal dynamics, electrical load, barometric weather, electricity-grid behavior, traffic flow, exchange-rate drift, and solar harmonics. On validation, ASTR reduces MSE from 0.523566 for Vanilla TS-RAG to 0.076342. The paired mean improvement is $0.447224±0.154782$ at 95% confidence. On the independent test partition, MSE decreases from 0.478836 to 0.161099, with paired improvement $0.317737±0.177480$. Complete mechanism isolation shows that trend decoupling and ridge reconstruction reduce MSE from 0.348310 to 0.077365 before confidence fusion; the final gate lowers it to 0.076342. Increasing balanced memory size from 42 to 252 trajectories reduces MSE by 64.38%.

The contributions are:

1. We propose a training-free forecaster that integrates robust adaptive trend modeling, temporal–spectral retrieval, phase transport, SVD ridge reconstruction, and analytical confidence fusion.
2. We establish self-contained finite-dimensional guarantees for local robustness, regularized trend–residual separation, leakage exclusion, ridge uniqueness and stability, Fourier continuation, and convex fusion.
3. We provide paired validation and test comparisons, a complete trend-by-ridge factorial study, component subtraction, additive-noise stress tests, and balanced memory-size scaling, exposing both integrated accuracy and mechanism-level effects.

## Related Work
## 2. Related Work

### Retrieval-augmented forecasting

Ning et al. (2025, arXiv:2503.07649) introduce TS-RAG, which retrieves semantically relevant time-series segments using pretrained encoders and adaptively mixes them with a foundation model. Xiao et al. (2026, arXiv:2608.06223) independently develop retrieval augmentation for numerical forecasting and argue that directly concatenating references, as in language prompting, is insufficient. SERAF (Zhou et al., 2026, arXiv:2606.14941) addresses non-stationarity through dual retrieval over signals and self-generated textual descriptions. ASTR shares their use of external historical evidence but gives retrieval an explicit residual-subspace objective, transports phase analytically, and obtains coefficients from a closed-form regularized reconstruction problem.

MQRetNN (Yang et al., 2022, arXiv:2207.10517) retrieves cross-entity contexts and injects them through attention into a pretrained probabilistic forecaster. RATL (He et al., 2026, arXiv:2609.03937) stores forecast residuals from a frozen base model and retrieves them for feedback correction. KReF (Zhang and Su, 2026, arXiv:2608.06748) treats retrieved futures as a query-local empirical predictive distribution, supporting weighted means, quantiles, and uncertainty. In contrast, ASTR removes adaptive trend geometry before matching, regards retrieved histories as a local regression dictionary, and uses reconstruction quality to control fusion with a spectral predictor.

Tajmouati et al. (2021, arXiv:2103.14200) tune weighted nearest-neighbor time-series forecasters and reduce search cost by restricting the neighbor search set. Chen and Shah (2025, arXiv:2502.15900) synthesize nonasymptotic explanations of nearest-neighbor prediction and scalable approximate search. ASTR complements this literature by replacing locally constant neighbor regression with a regularized linear combination of aligned future-bearing prototypes and by proving finite-dimensional stability of that combination.

### Decomposition, spectral modeling, and sequence architectures

DLinear (Zeng et al., 2022, arXiv:2205.13504) demonstrates that moving-average decomposition followed by simple linear mappings can rival more elaborate Transformer forecasters. PatchTST (Nie et al., 2022, arXiv:2211.14730) represents subseries as patches and shares a channel-independent Transformer across variables, reducing attention length while preserving local semantics. TimesNet (Wu et al., 2022, arXiv:2210.02186) discovers multiple periods and maps one-dimensional signals into two-dimensional tensors to model intra-period and inter-period variation. iTransformer (Liu et al., 2023, arXiv:2310.06625) treats variables as tokens and uses attention to learn cross-variate dependencies. LLM-Mixer (Kowsher et al., 2024, arXiv:2410.11674) combines multiscale decomposition with a frozen language model. ASTR is complementary: it is training-free, operates on one channel per forecast call, and uses decomposition and Fourier structure to define retrieval geometry rather than neural tokens.

Damato et al. (2025, arXiv:2502.19086) combine Gaussian-process latent functions with negative-binomial and fully parameterized Tweedie likelihoods for intermittent count forecasting. Their objective is calibrated probabilistic prediction with point mass at zero and heavy tails. ASTR instead targets deterministic continuation of drifting, multi-periodic real-valued signals; its guarantees concern decomposition, harmonic continuation, and regularized retrieval rather than likelihood calibration.

Let $N$ be memory size, $L$ lookback length, $H$ horizon, $d$ an embedding width, $C$ the number of variables, $P$ patch length, and $n$ the number of training observations.

| Work | Mathematical formulation | Dominant asymptotic cost | Principal guarantee or distinction |
|---|---|---:|---|
| TS-RAG, Ning et al. (2025) | Encoder retrieval and adaptive representation mixing | $O(Nd)$ plus foundation-model inference | Empirical zero-shot adaptation and interpretable retrieval |
| TS-RAG, Xiao et al. (2026) | Retrieved references integrated into a numerical forecaster | $O(Nd)$ plus forecaster inference | Empirical benefit beyond direct reference concatenation |
| SERAF, Zhou et al. (2026) | Dual numerical and textual retrieval | $O(N(d_t+d_s))$ plus forecaster inference | Complementary semantic and signal evidence |
| MQRetNN, Yang et al. (2022) | Retrieval-filtered cross-entity attention | $O(Nd)$ plus attention | Learned cross-entity context for multi-horizon prediction |
| PatchTST, Nie et al. (2022) | Channel-independent attention over $L/P$ patches | $O((L/P)^2d)$ per channel | Reduced attention length and longer effective context |
| TimesNet, Wu et al. (2022) | Fourier period discovery and two-dimensional variation blocks | $O(L log L)$ plus convolutional blocks | Explicit intra-period and inter-period modeling |
| iTransformer, Liu et al. (2023) | Attention over variable tokens | $O(C^2d)$ plus projections | Variate-centric dependency modeling |
| DLinear, Zeng et al. (2022) | Moving-average decomposition and linear maps | $O(LH)$ | Strong simple decomposition formulation |
| Weighted kNN, Tajmouati et al. (2021) | Tuned weighted subsequence neighbors | $O(NL)$ per exact query | Restricted search reduces tuning cost |
| Nearest-neighbor theory, Chen and Shah (2025) | Local nonparametric regression and classification | $O(Nd)$ for exact search | General nonasymptotic nearest-neighbor guarantees |
| RATL, He et al. (2026) | Retrieved residual feedback around a frozen forecaster | $O(Nd+NH)$ plus base inference | Level-robust residual correction |
| KReF, Zhang and Su (2026) | Weighted retrieved futures as an empirical distribution | $O(Nd+NH)$ | Training-free predictive masses, quantiles, and uncertainty |
| LLM-Mixer, Kowsher et al. (2024) | Multiscale decomposition with frozen language-model mixing | Backbone-dependent | Empirical multiscale transfer without full fine-tuning |
| Tweedie GP, Damato et al. (2025) | Gaussian-process latent function with count likelihood | $O(n^3)$ for exact training | Probabilistic intermittent-demand modeling |
| ASTR | Robust residualization, phase-aligned retrieval, SVD ridge, and analytical gating | $O(NL+SM(L+H)+LK^2)$ online | Near-orthogonality, stable unique ridge, exact retained-mode continuation, and convex fusion bound |

Unlike encoder-based retrieval, ASTR requires no learned embedding. Unlike neighbor averaging, it reconstructs the query inside the span of aligned prototypes. Unlike purely spectral forecasting, it uses Fourier continuation as a confidence-adaptive complement to memory. The formulation thereby connects decomposition, nearest-neighbor evidence, and harmonic continuation through one residual reconstruction objective.

## Methodology
## 3. Methodology

### 3.1 Problem formulation

Given a history $x=(x_0,…,x_{L-1})∈ℝ^L$, forecast horizon $H$, and memory $M={(k_i,v_i)}_{i=1}^N$, ASTR returns $ŷ∈ℝ^H$. In the experiments, $L=64$ and $H=24$. Each key has length $L$, and every associated future supplies at least $H$ values.

ASTR represents each cleaned history as

$$x=t_x+σ_xq,$$

where $t_x$ is an adaptive robust trend, $r_x=x-t_x$, $σ_x=max(√{∥r_x∥_2^2/L},10^{-8})$, and $q=r_x/σ_x$. A memory key produces a normalized residual $q_i$ and normalized future residual

$$u_i=(v_i-t_{i,fut})/σ_i.$$

### 3.2 Robust local despiking

For an interior sample $x_t$, let $n_t=(x_{t-2},x_{t-1},x_{t+1},x_{t+2})$ and $m_t=median(n_t)$. Define

$$d_t=1.4826 median_j |n_{t,j}-m_t|,$$
$$s_t=max(d_t,0.35 median_j |n_{t,j+1}-n_{t,j}|+10^{-8}).$$

With threshold $c=8$, replacement occurs only when $|x_t-m_t|>cs_t$:

$$x_t^{clean}=m_t+clip(x_t-m_t,-cs_t,cs_t).$$

The first and last two observations are unchanged.

**Lemma 1 (bounded local influence).** Every replaced observation satisfies $|x_t^{clean}-m_t|≤8s_t$.

**Proof.** The clipping operator is the Euclidean projection of $x_t-m_t$ onto $[-8s_t,8s_t]$. Its magnitude is therefore at most $8s_t$. Adding $m_t$ proves the result. $□$

### 3.3 Adaptive orthogonal trend modeling

Let $z_t=-1+2t/(L-1)$. ASTR considers

$$B_{lin}(z)=[1,z],$$
$$B_{quad}(z)=[1,z,(3z^2-1)/2],$$
$$B_{pw}(z)=[1,z,max(z,0)].$$

Each design matrix is represented by a reduced QR factorization $B=QR$. Candidate geometry is selected by blocked prequential prediction. For $L=64$, block width is eight; training prefixes end at 40, 48, and 56, and each predicts the next eight observations. A causal trailing mean of width five is used only during selection. Errors are divided by the row-wise scale $1.4826 median_t|x_t-median(x)|+10^{-8}$ and scored with

$$h_{1.5}(a)=\begin{cases}a^2/2,&|a|≤1.5,\\1.5|a|-1.125,&|a|>1.5.\end{cases}$$

The candidate with the smallest mean fold loss is selected independently for every trajectory. The cleaned, unsmoothed history is then refit for four Huber iteratively reweighted least-squares updates with threshold two. At each update,

$$W_{tt}=min(1,2/max(|e_t|/s,10^{-12})),$$

where $s=1.4826 median_t|e_t|+10^{-8}$. The orthogonal coefficients solve

$$a^*=argmin_a ∥W^{1/2}(x-Qa)∥_2^2+ε∥a∥_2^2,\qquad ε=10^{-10}.$$

The linear system is solved directly; if it is singular, its least-squares solution is used. Raw coefficients are obtained from $Rc=a^*$ with the same direct-solve/least-squares rule.

Let $e=B(1)c$, $δ_h=2h/(L-1)$, and $h=1,…,H$. The future trend is

$$t_{fut,h}=e+0.85dδ_h+0.5κδ_h^2.$$

For the linear basis, $d=c_1$ and $κ=0$. For the one-knot basis, $d=c_1+c_2$ and $κ=0$. For the quadratic basis, $d=c_1+3c_2$ and $κ=1.5c_2$.

**Theorem 1 (subspace invariance and quantified near-orthogonality).** Let $B=QR$ have full column rank and let $W$ be positive diagonal. Then $span(B)=span(Q)$, the fitted trend lies in this common subspace, and the regularized residual $r=x-Qa^*$ satisfies

$$Q^TWr=εa^*,\qquad B^TWr=εR^Ta^*.$$

Consequently, $∥Q^TWr∥_2≤ε∥a^*∥_2$. When $ε=0$, exact weighted orthogonality is recovered; with uniform weights this becomes ordinary Euclidean trend–residual orthogonality.

**Proof.** Since $R$ is invertible, $Bc=Q(Rc)$ and $Qa=B(R^{-1}a)$, proving equality of the two spans and coordinate invariance. Differentiating the strictly convex regularized objective gives

$$Q^TW(Qa^*-x)+εa^*=0.$$

With $r=x-Qa^*$, rearrangement yields $Q^TWr=εa^*$. Multiplication by $R^T$ and $B=QR$ gives $B^TWr=εR^Ta^*$. Taking norms proves the bound. $□$

**Proposition 1 (forecast-horizon leakage exclusion).** Trend selection and refitting depend only on the observed length-$L$ history.

**Proof.** Every selection model is fitted to a prefix and scored only on its immediately following block within indices $0,…,L-1$. The causal smoother at index $t$ uses indices no greater than $t$, and final robust fitting uses exactly indices $0,…,L-1$. No forecast-horizon observation enters selection or fitting. $□$

### 3.4 Temporal–spectral retrieval and phase transport

For every normalized residual, compute $F_i=rFFT(q_i)$, set its zero-frequency magnitude to zero, and define

$$g_i=|F_i|/max(∥|F_i|∥_2,10^{-12}).$$

For query features $(q,g)$, the global score is

$$s_i=0.7(q_i^Tq/L)+0.3g_i^Tg.$$

ASTR retains the best 32 candidates. For every candidate, it evaluates circular shifts $a∈{-12,…,12}$ and selects the shift maximizing cosine similarity with $q$. If $q_i^{shift}$ is the selected history, the aligned key is

$$q_i'=0.8q_i+0.2q_i^{shift}.$$

The same shift is applied to the concatenation of the original history and future residual. The last $H$ samples of the shifted concatenation give $u_i^{shift}$, and

$$u_i'=0.8u_i+0.2u_i^{shift}.$$

Post-alignment ranking uses

$$s_i'=0.7cos(q_i',q)+0.3g_i^Tg.$$

Thus temporal and spectral terms retain weights $0.7$ and $0.3$ after alignment. The best $K=12$ pairs define the history dictionary $R=[q_1',…,q_K']∈ℝ^{L×K}$ and future matrix $A=[u_1',…,u_K']∈ℝ^{H×K}$.

### 3.5 Stability-aware prototype regression

ASTR minimizes

$$J_λ(w)=L^{-1}∥Rw-q∥_2^2+λ∥w∥_2^2.$$

Let $R=PΣV^T$ with singular values $s_j$. The solution is

$$w_λ=V\,diag((s_j/L)/(s_j^2/L+λ))P^Tq.$$

The base strength is $λ_0=0.008$. Generalized cross-validation considers 17 logarithmically spaced values $λ_0c$, with $c∈[0.05,0.6]$, together with $λ_0$. Define $p=P^Tq$, $a_j=(s_j^2/L)/(s_j^2/L+λ)$, and $e_⊥=max(∥q∥_2^2-∥p∥_2^2,0)$. The selected value minimizes

$$GCV(λ)=L[e_⊥+Σ_j(1-a_j)^2p_j^2]/[L-Σ_ja_j]^2.$$

The SVD is used directly; no Gram-matrix inverse is formed. The retrieved future residual is $r_{ret}=Aw_λ$.

**Proposition 2 (well-defined adaptive regularization).** The generalized-cross-validation rule returns a positive finite regularization strength and has at least one minimizer.

**Proof.** Every candidate equals a positive multiple of $λ_0>0$. Hence $0≤a_j<1$, so $Σ_ja_j<rank(R)≤L$ and the denominator is strictly positive. The numerator is nonnegative and finite. The candidate set is finite and nonempty, so its real-valued score attains a minimum. $□$

**Theorem 2 (unique solution and conditional stability).** For every selected $λ>0$, $J_λ$ has a unique minimizer. Conditional on a fixed retrieved dictionary and fixed $λ$,

$$∥w_λ(q)-w_λ(q')∥_2≤∥q-q'∥_2/(2√{Lλ}),$$

and

$$∥Aw_λ(q)-Aw_λ(q')∥_2≤∥A∥_2∥q-q'∥_2/(2√{Lλ}).$$

**Proof.** The Hessian is $2(R^TR/L+λI)$, whose eigenvalues are at least $2λ>0$; strict convexity gives uniqueness. Setting the gradient to zero yields $(R^TR/L+λI)w=R^Tq/L$. Substitution of $R=PΣV^T$ gives the stated solution. The singular response from $q$ to $w$ is

$$f(s)=s/(s^2+Lλ).$$

Because $f'(s)=(Lλ-s^2)/(s^2+Lλ)^2$, its maximum over $s≥0$ occurs at $s=√{Lλ}$ and equals $1/(2√{Lλ})$. This is the operator norm of the response map. Multiplication by $A$ proves the future-trajectory bound. $□$

### 3.6 Fourier continuation and confidence fusion

Let $Q_k$ be the real Fourier coefficients of $q$. ASTR retains the five nonzero bins of largest magnitude. For $t=L,…,L+H-1$,

$$r_{spec}(t)=Σ_{k∈J}c_k[Re(Q_k/L)cos(2πkt/L)-Im(Q_k/L)sin(2πkt/L)],$$

where $c_k=1$ for the Nyquist bin when $L$ is even and $c_k=2$ otherwise.

**Theorem 3 (exact retained-mode continuation).** Suppose $q$ is the restriction of a real $L$-periodic signal with zero direct-current component and at most five supported nonzero real-Fourier bins. If all supported bins are selected, then $r_{spec}$ equals that periodic signal at every future integer time.

**Proof.** The inverse discrete Fourier representation pairs each positive-frequency coefficient with its complex conjugate. Combining a pair gives exactly the cosine–sine term above with multiplier two. For even $L$, the Nyquist coefficient is self-conjugate and therefore has multiplier one. The direct-current component is zero by assumption, and every remaining supported bin is included. The inverse representation is consequently exact for every integer time, including times beyond the observed interval. $□$

ASTR computes

$$ρ=∥q-Rw_λ∥_2^2/max(∥q∥_2^2,10^{-12}),$$
$$γ=exp(-0.2ρ),$$
$$ŷ=t_{fut}+σ_x[γr_{ret}+(1-γ)r_{spec}].$$

**Theorem 4 (confidence monotonicity and convex fusion bound).** The gate satisfies $0<γ≤1$ and decreases monotonically with reconstruction error. For a true normalized future residual $r^*$, define $e_r=r_{ret}-r^*$ and $e_s=r_{spec}-r^*$. Then

$$∥γe_r+(1-γ)e_s∥_2^2$$
$$=γ∥e_r∥_2^2+(1-γ)∥e_s∥_2^2-γ(1-γ)∥e_r-e_s∥_2^2$$
$$≤γ∥e_r∥_2^2+(1-γ)∥e_s∥_2^2.$$

**Proof.** Since $ρ≥0$, $0<exp(-0.2ρ)≤1$, and its derivative is $-0.2exp(-0.2ρ)<0$. Expanding the squared norm and collecting inner products gives the equality. The last term is nonnegative and subtracted, proving the inequality. $□$

### 3.7 Numerical safety and algorithms

Processed memory residuals, futures, and spectra are cached by memory identity, with capacity for eight banks. Cache reuse changes computational cost but not the forecast. Linear systems used during trend estimation are solved directly and revert to least-squares solutions if direct solution fails. After forecast assembly, ASTR verifies length and finiteness. An invalid assembled forecast is replaced by the finite continued trend when available; otherwise the last cleaned observation is repeated, with zero used only if that last observation is unavailable or non-finite.

```text
Algorithm 1: Residual-memory preprocessing
Input: memory pairs {(k_i,v_i)} for i=1,...,N; horizon H
for each pair (k_i,v_i):
    clip isolated spikes using the four-neighbor robust scale
    choose linear, quadratic, or one-knot trend by blocked prequential loss
    refit the chosen trend using four regularized Huber updates in QR coordinates
    continue the endpoint trend with slope factor 0.85 and curvature factor 0.5
    r_i <- k_i - fitted trend
    sigma_i <- max(RMS(r_i), 1e-8)
    q_i <- r_i / sigma_i
    u_i <- (first H samples of v_i - continued trend) / sigma_i
    g_i <- unit-normalized non-DC rFFT magnitude of q_i
store (q_i,u_i,g_i)
Output: cached residual memory
```

```text
Algorithm 2: ASTR forecasting
Input: history x; cached residual memory; horizon H
clean x; select and robustly fit its trend
q <- RMS-normalized residual of x
score all memories by 0.7 temporal similarity + 0.3 spectral similarity
retain 32 candidates
for each retained candidate:
    find the best circular shift in {-12,...,12}
    blend original and shifted history/future with phase strength 0.2
re-rank by 0.7 aligned temporal cosine + 0.3 spectral similarity
retain K=12 prototypes and form R and A
select positive ridge strength by generalized cross-validation
solve ridge coefficients through the SVD of R
r_ret <- A w
r_spec <- five-mode Fourier continuation of q
rho <- ||q-Rw||_2^2 / max(||q||_2^2,1e-12)
gamma <- exp(-0.2 rho)
y <- continued trend + RMS scale * [gamma r_ret + (1-gamma) r_spec]
apply the finite-output safety rule and return y
```

### 3.8 Complexity

Let $C=3$ trend candidates, $F=3$ prequential folds, $I=4$ robust iterations, $S=25$ candidate shifts, shortlist size $M=32$, and prototype count $K=12$. Since every trend basis has dimension at most three, memory preprocessing costs

$$O(N[CFL+IL+LlogL+H])$$

and cached storage is $O(N(L+H))$. For one forecast, trend fitting costs $O(CFL+IL)$, exact feature comparison costs $O(NL)$, phase alignment costs $O(SM(L+H))$, the SVD ridge computations cost $O(LK^2)$, and future synthesis costs $O(LlogL+KH)$. Therefore,

$$T_{online}=O(NL+SM(L+H)+LK^2+LlogL+KH),$$
$$S_{online}=O(N+M(L+H)+LK).$$

With fixed $M$, $K$, $S$, and trend dimension, online time is linear in memory size and lookback length.

## Experiments
## 4. Experiments

### 4.1 Protocol

We evaluate seven non-stationary regimes: thermal dynamics, electrical load, barometric weather, electricity-grid behavior, traffic flow, exchange-rate drift, and solar harmonics. Each regime contributes six forecasts, giving 42 series per partition. Histories have length 64 and targets have length 24. The trajectories combine regime-specific periodic components, linear and quadratic drift, colored AR(1) history noise, and occasional isolated history spikes.

We report MSE, MAE, the mean squared discrepancy between mean-centered forecast and target rFFT magnitudes, and squared error between their fitted horizon slopes. Vanilla TS-RAG uses global z-score cosine retrieval, five neighbors, normalized-future averaging, and a $0.7/0.3$ retrieval–persistence blend. The DLinear realization uses a width-13 moving-average decomposition, linear trend extrapolation from the last 24 points, and seasonal repetition. TimesNet-FFT uses linear detrending and the four largest Fourier modes. MQRetNN-kNN uses eight z-normalized neighbors with softmax temperature 0.25. Paired intervals use the 42 per-series MSE differences and a Student-t critical value with 41 degrees of freedom.

### 4.2 Aggregate accuracy

| Partition | MSE | MAE | Spectral L2 | Trend-drift | MSE std. | Mean 95% half-width |
|---|---:|---:|---:|---:|---:|---:|
| Validation | **0.076342** | **0.207383** | **0.259836** | **0.000177** | 0.100616 | 0.031353 |
| Test | **0.161099** | **0.307078** | **0.346780** | **0.000243** | 0.238306 | 0.074260 |

ASTR preserves both oscillatory structure and forecast slope. The independent test result confirms that the validation gain is not confined to one query realization: test MSE remains 66.36% below Vanilla TS-RAG and 51.15% below MQRetNN-kNN.

| Partition | Comparator | Comparator MSE | ASTR MSE | Paired reduction | Paired 95% interval | Relative reduction |
|---|---|---:|---:|---:|---:|---:|
| Validation | Vanilla TS-RAG | 0.523566 | **0.076342** | 0.447224 | [0.292442, 0.602006] | 85.42% |
| Validation | DLinear | 1.181559 | **0.076342** | 1.105217 | [0.769201, 1.441233] | 93.54% |
| Validation | TimesNet-FFT | 0.783882 | **0.076342** | 0.707540 | [0.511712, 0.903368] | 90.26% |
| Validation | MQRetNN-kNN | 0.365157 | **0.076342** | 0.288815 | [0.132978, 0.444652] | 79.10% |
| Test | Vanilla TS-RAG | 0.478836 | **0.161099** | 0.317737 | [0.140257, 0.495216] | 66.36% |
| Test | DLinear | 1.240957 | **0.161099** | 1.079858 | [0.672897, 1.486818] | 87.02% |
| Test | TimesNet-FFT | 0.788769 | **0.161099** | 0.627670 | [0.385422, 0.869918] | 79.58% |
| Test | MQRetNN-kNN | 0.329745 | **0.161099** | 0.168646 | [0.055280, 0.282012] | 51.15% |

Every paired interval has a positive lower endpoint on both partitions. The gains therefore hold at the paired series level rather than arising only from differences between unpaired aggregate means.

### 4.3 Regime-level validation behavior

| Regime | ASTR MSE | Vanilla TS-RAG MSE | Reduction |
|---|---:|---:|---:|
| Thermal | 0.036914 | 0.291615 | 87.34% |
| Electrical load | 0.113018 | 1.208908 | 90.65% |
| Barometric weather | 0.138869 | 0.438834 | 68.35% |
| Electricity grid | 0.029505 | 0.429537 | 93.13% |
| Traffic flow | 0.037991 | 0.381281 | 90.04% |
| Exchange drift | 0.150245 | 0.570856 | 73.68% |
| Solar harmonics | 0.027848 | 0.343930 | 91.90% |

ASTR improves all seven regimes. The largest relative gain occurs for electricity-grid trajectories, while exchange and weather remain the hardest in absolute MSE yet still improve by 73.68% and 68.35%. This breadth supports the intended separation of drift, phase, and oscillatory content.

### 4.4 Complete trend-by-ridge factorial isolation

For the four factorial cells, fusion is fixed to pure retrieval so that trend geometry and prototype regression are isolated.

| Trend treatment | Prototype combination | MSE | MAE | Spectral L2 |
|---|---|---:|---:|---:|
| Adaptive trend decoupling | Ridge reconstruction | **0.077365** | **0.207636** | **0.254596** |
| Adaptive trend decoupling | Softmax averaging | 0.197033 | 0.343005 | 1.430921 |
| Constant-level residualization | Ridge reconstruction | 0.161242 | 0.288594 | 0.368575 |
| Constant-level residualization | Softmax averaging | 0.348310 | 0.447564 | 2.641952 |

Adaptive trend decoupling alone reduces MSE by 43.43% relative to the cell with neither mechanism, while ridge reconstruction alone reduces it by 53.71%. Combining them yields a 77.79% reduction and lowers spectral error by 90.36%. Relative to trend decoupling with averaging, ridge reconstruction reduces MSE by 60.74%; relative to ridge with constant-level residualization, adaptive trend modeling reduces it by 52.02%. The mechanisms therefore address distinct errors: trend removal improves retrieval geometry, while ridge regression exploits complementary prototypes.

### 4.5 Component subtraction

| Subtracted or replaced component | MSE | MAE | Spectral L2 | Trend-drift | MSE increase |
|---|---:|---:|---:|---:|---:|
| None | **0.076342** | 0.207383 | 0.259836 | 0.000177 | — |
| Adaptive trend decoupling | 0.164253 | 0.289374 | 0.374751 | 0.000350 | 115.16% |
| Phase alignment | 0.099514 | 0.228452 | 0.315989 | 0.000252 | 30.35% |
| Confidence gate | 0.077365 | 0.207636 | 0.254596 | 0.000173 | 1.34% |
| Fourier continuation | 0.076535 | **0.206991** | 0.259643 | **0.000174** | 0.25% |
| Adaptive SVD–GCV ridge, replaced by fixed direct ridge | 0.077217 | 0.210761 | **0.256623** | 0.000174 | 1.15% |

Trend decoupling is the largest isolated contributor, more than doubling MSE when removed. Phase alignment is second and increases every reported error when subtracted. The confidence gate supplies a smaller but consistent MSE gain over pure retrieval. Fourier continuation has a selective aggregate role: its removal slightly changes MAE and spectral error but increases the primary MSE. Adaptive SVD–GCV regularization improves both MSE and MAE relative to a fixed-strength direct solve.

### 4.6 Additive-noise robustness

| Added history-noise std. | ASTR MSE | Vanilla TS-RAG | DLinear | TimesNet-FFT | Reduction vs. TS-RAG |
|---:|---:|---:|---:|---:|---:|
| 0.00 | **0.076342** | 0.523566 | 1.181559 | 0.783882 | 85.42% |
| 0.15 | **0.098294** | 0.527117 | 1.203909 | 0.783849 | 81.35% |
| 0.30 | **0.119324** | 0.594723 | 1.271825 | 0.808369 | 79.94% |
| 0.45 | **0.191716** | 0.560642 | 1.523094 | 0.829523 | 65.80% |

ASTR is best at every noise level. At standard deviation 0.30 it retains a 79.94% advantage over Vanilla TS-RAG; at 0.45, its MSE remains below one quarter of TimesNet-FFT's. This behavior is consistent with bounded local clipping and regularized prototype reconstruction limiting contamination and coefficient sensitivity.

### 4.7 Memory-size scaling

The same 42 queries are retained while equal numbers of trajectories are selected from each regime.

| Total memory | Per regime | MSE | MAE | Spectral L2 | Trend-drift |
|---:|---:|---:|---:|---:|---:|
| 42 | 6 | 0.214295 | 0.319411 | 0.925936 | 0.000854 |
| 84 | 12 | 0.146589 | 0.287865 | 0.751371 | 0.000624 |
| 168 | 24 | 0.099515 | 0.243394 | 0.383456 | 0.000281 |
| 252 | 36 | **0.076342** | **0.207383** | **0.259836** | **0.000177** |

Increasing memory from 42 to 252 trajectories reduces MSE by 64.38%, spectral error by 71.94%, and trend-drift error by 79.30%. All four metrics improve monotonically at the measured scales, supporting the reconstruction view that a richer balanced residual dictionary increases coverage of phase-compatible dynamics.

### 4.8 Runtime and memory

On a pinned 3.90 GHz ARM Cortex-X925 core, ASTR averages 1.6135 ms per 24-step validation forecast when comparing against 252 memory trajectories. Peak traced memory is 3,415,562 bytes, approximately 3.26 MiB. The observed latency is consistent with the $O(NL)$ exact-retrieval term, while the 12-prototype SVD remains a small fixed-dimensional operation.

## Limitations
## 5. Limitations

The empirical study uses a deterministic synthetic benchmark with seven designed non-stationary regimes. It exercises multiple periods, linear and quadratic drift, colored history noise, and isolated spikes, but does not cover the full heterogeneity of real sensors, missing values, irregular sampling, abrupt structural breaks, or multivariate causal interactions. Validation and independent test partitions contain 42 trajectories each; broader dataset-level replication would provide stronger evidence across domains and sampling processes.

The named comparison methods are lightweight benchmark realizations of their defining mechanisms rather than complete pretrained instantiations of every published architecture. The results therefore isolate retrieval, decomposition, Fourier continuation, and neighbor weighting under a common history and horizon, but do not establish superiority over every training budget, backbone, or hyperparameterization of those works.

The theoretical guarantees are finite-dimensional and conditional. The ridge sensitivity bound assumes a fixed retrieved dictionary and fixed selected regularization strength; discrete neighbor and phase-shift changes can introduce selection discontinuities. Near-orthogonality, uniqueness, sensitivity control, exact retained-mode continuation, and convex fusion do not guarantee low future error when the memory lacks the relevant regime. Exact harmonic continuation additionally requires stable periodicity, zero direct-current residual, and frequencies aligned with the length-$L$ Fourier grid.

Generalized cross-validation scores reconstruction of the observed history, which is an indirect proxy for future transport quality. The exponential confidence gate is bounded and monotone but is not proven risk-optimal. Memory-size gains were obtained with balanced regime coverage; highly imbalanced or adversarial memories may behave differently.

Finally, exact retrieval is linear in memory size. Approximate search or learned indexing would be necessary for million-scale memories, but either could alter neighbor identity and phase transport. Identity-based caching also assumes that a cached memory is not mutated in place. Extending ASTR to probabilistic outputs, irregular timestamps, multivariate coupling, and large-scale indexed memory remains substantive future work.

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
| initial_subset | 0.133162566 |
| initial_full | 0.101700802 |
| seed0_subset | 0.104342707 |
| seed0_full | 0.077217315 |
| evolved0_subset | 0.106334875 |
| evolved0_full | 0.076487215 |
| ablation0 | 0.077217315 |
| ablation1 | 0.077045988 |
| ablation2 | 0.077217315 |
| ablation3 | 0.076341697 |
| ablation4 | 0.076487215 |
| pruned | 0.076341697 |
| pruned_ablation0 | 0.077217315 |
| pruned_ablation1 | 0.164253211 |
| pruned_ablation2 | 0.076535414 |
| pruned_ablation3 | 0.077364998 |
| pruned_ablation4 | 0.099513708 |
| rebuttal0 | 0.161378832 |
| final_test | 0.161098985 |
