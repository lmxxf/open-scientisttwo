# Damped Orthogonal Trend Projection and Spectral–Temporal Ridge Retrieval for Time-Series Forecasting

## Abstract
We propose a nonparametric forecaster that separates local polynomial drift from oscillatory retrieval features. The method clips isolated historical spikes, projects query and memory histories onto an orthogonal quadratic basis, retrieves twelve prototypes using temporal correlation and Fourier-magnitude similarity, and estimates their continuation weights by constant-penalty ridge regression. A damped polynomial continuation restores the query trend. Our theoretical analysis establishes orthogonal residualization, uniqueness of the regularized reconstruction, and a precise reconstruction–regularization tradeoff. Across 42 validation and 42 held-out test trajectories spanning seven regimes, the method achieves MSEs of 0.078471 and 0.134549, respectively, reducing error against the benchmark's vanilla retrieval baseline by 85.01% and 71.90%. The held-out paired MSE reduction is 0.344287 with Student-t 95% confidence interval [0.183436, 0.505138]. A complete factorial study isolates trend projection and prototype regression, while five individual component removals each increase validation MSE. Fourier features support retrieval; the evaluated predictor uses neither spectral forecast blending nor adaptive confidence gating.

## Introduction
We study forecasting a future vector y ∈ ℝᴴ from a history x ∈ ℝᴸ and a memory of N historical key–continuation pairs. Our central question is how to transfer oscillatory continuations when histories differ in level, drift, amplitude, and occasional sensor spikes. The evaluated setting uses L = 64 and H = 24.

Global normalization removes a constant offset and rescales amplitude, but leaves polynomial drift inside the retrieval representation. Consequently, histories can appear similar because their ramps align rather than because their oscillations match. A second issue arises after retrieval: averaging continuations imposes equal coefficients even when the retrieved histories are correlated or contribute unequally to reconstructing the query.

We address these issues with an explicit decomposition. Orthogonal quadratic projection defines the oscillatory retrieval space; damping controls polynomial continuation beyond the observation boundary. Temporal residual correlation retains phase information, while Fourier magnitudes add complementary frequency information. Ridge regression then fits the query using the retrieved residual histories and transfers the resulting signed coefficients to their continuations.

Our contributions are an implementation-specified estimator, mathematical guarantees for its projection and reconstruction steps, and an evaluation combining paired baseline comparisons, complete mechanism isolation, five component subtraction experiments, and noise perturbations. The final estimator retains robust preprocessing, orthogonal trend separation, damped continuation, spectral retrieval features, and constant ridge regression.

## Related Work
Retrieval-augmented forecasting takes several forms. Ning et al.'s [TS-RAG](https://arxiv.org/abs/2503.07649) retrieves segments with pretrained time-series encoders and combines retrieved evidence with a foundation model through an Adaptive Retrieval Mixer. Xiao et al.'s [TS-RAG](https://arxiv.org/abs/2608.06223) motivates forecasting-specific integration of reference sequences, observing that language-model-style concatenation need not transfer directly. [SERAF](https://arxiv.org/abs/2606.14941) augments time-series retrieval with self-generated textual descriptions and selectively combines the resulting evidence. Our estimator instead specifies a numerical retrieval geometry and a closed-form continuation operator.

[MQRetNN](https://arxiv.org/abs/2207.10517) retrieves cross-entity contexts and incorporates them through attention over pretrained representations. [RATL](https://arxiv.org/abs/2609.03937) freezes a base forecaster and retrieves historical forecast-error trajectories from model-specific memory. Our residuals have a different definition: they are histories and continuations after polynomial trend removal, rather than errors produced by a trained base forecaster.

Several architectural studies motivate the representation choices. [PatchTST](https://arxiv.org/abs/2211.14730) uses subseries patches and shared channel-independent Transformer weights. [TimesNet](https://arxiv.org/abs/2210.02186) expresses temporal variation through period-dependent two-dimensional tensors. [iTransformer](https://arxiv.org/abs/2310.06625) reconsiders temporal tokenization to emphasize variate-centric representations. Zeng et al.'s [LTSF-Linear study](https://arxiv.org/abs/2205.13504) demonstrates the strength of simple linear forecasting models. Our method combines an explicit polynomial decomposition with Fourier retrieval descriptors and a small linear reconstruction problem.

Classical nearest-neighbor forecasting also informs our design. [Tajmouati et al.](https://arxiv.org/abs/2103.14200) study weighted-neighbor parameter tuning and reduced candidate search sets. [Chen and Shah](https://arxiv.org/abs/2502.15900) discuss statistical guarantees, approximate search, and learned neighborhood structure. Our contribution concerns the representation and continuation fit within a retrieved neighborhood.

Table 1 contrasts prior retrieval-augmented and decomposition-based forecasting frameworks against our estimator.


Among training-free retrieval methods, [KReF](https://arxiv.org/abs/2608.06748) combines robust preprocessing and handcrafted time/Fourier summary features with kernel-weighted historical continuation transfer and conformal intervals. While KReF represents windows via handcrafted summary statistics or frozen random Fourier features and aggregates retrieved futures using nonnegative similarity-derived probability weights, our estimator operates directly on orthonormal degree-2 Legendre residuals with damped quadratic continuation and solves the closed-form Tikhonov linear system $(G + \lambda I_k)w^* = D^\top q / n$ to account for inter-prototype Gram cross-correlations. By contrast, our estimator projects histories onto an orthonormal degree-2 Legendre basis with damped quadratic horizon continuation and solves the closed-form Tikhonov gram system $(G + \lambda I_k)w^* = D^\top q / n$, accounting for inter-prototype Gram cross-correlations while allowing signed de-biasing weights.

| Work | Representation or memory | Forecasting mechanism | Relation to our method |
|---|---|---|---|
| Ning et al., TS-RAG | Pretrained time-series embeddings | Foundation model with adaptive retrieval mixer | We use explicit residual features and ridge coefficients |
| Xiao et al., TS-RAG | Retrieved reference sequences | Forecasting-specific retrieval integration | We specify the numerical transfer operator |
| SERAF | Series and generated text | Selective dual-retrieval fusion | We retrieve from numerical features alone |
| MQRetNN | Encoded cross-entity contexts | Retrieval with cross-entity attention | We reconstruct histories by regularized linear fitting |
| PatchTST | Channel-independent patches | Transformer forecasting | We use a local polynomial and memory estimator |
| TimesNet | Period-based two-dimensional tensors | Temporal variation modeling | We use Fourier magnitudes for neighbor ranking |
| iTransformer | Variate-centric representations | Inverted Transformer architecture | We forecast each query from explicit historical prototypes |
| LTSF-Linear | Ordered historical values | Simple linear forecasting models | We fit a query-specific retrieved basis |
| Tajmouati et al. | Historical subsequences | Tuned weighted nearest neighbors | We combine residual geometry with ridge transfer |
| Chen and Shah | Nearest-neighbor structures | Statistical and computational analysis | Our analysis concerns projection and local reconstruction |
| RATL | Base-model forecast residuals | Retrieved error correction | Our memory residuals remove polynomial trends |
| Our method | Orthogonal residuals and Fourier magnitudes | Damped trend plus ridge-weighted continuations | Explicit numerical decomposition and transfer |
| KReF (2608.06748) | Training-free kernel retrieval + conformal intervals | Nonnegative kernel probability weights over handcrafted/Fourier features | Orthonormal degree-2 Legendre damped continuation + Gram-regularized signed Tikhonov ridge solution |

## Methodology
### Inputs and evaluated configuration

The predictor receives only the query history, memory keys, memory continuations, and horizon. Computation uses NumPy float64 arrays. The evaluated configuration uses quadratic trend separation, damping d = 0.25, isolated-spike preprocessing, spectral similarity weight η = 0.25, K = 12, and isotropic ridge coefficient λ = 0.05. The continuation-disagreement penalty is zero. Spectral forecast blending and adaptive confidence gating are disabled; the output is the retrieval prediction described below.

### Isolated-spike preprocessing

For each history a, form edge-padded windows of radius three, giving seven observations per window. Let mₜ be the window median and let MADₜ = 1.4826 medianⱼ|aⱼ − mₜ| within that window. Define the history-wide floor f = 0.5 medianₜ|aₜ₊₁ − aₜ| and threshold bₜ = 4.5 max(MADₜ, f + 10⁻¹²). Mark positions satisfying |aₜ − mₜ| > bₜ. A marked position is modified only when neither adjacent position is marked, with boundaries checked where neighbors exist. Replace such a position by mₜ + clip(aₜ − mₜ, −bₜ, bₜ). Apply this operator independently to the query and every memory key. Memory continuation values remain unchanged.

### Orthogonal decomposition and damped continuation

For a cleaned history a of length L, let tₜ = −1 + 2t/(L − 1), t = 0,…,L − 1. Construct the Legendre design V with columns 1, t, and (3t² − 1)/2, and compute its reduced QR factorization V = QR. With column-vector notation, define c = Qᵀa, r(a) = a − Qc, and β = R⁻¹c.

For future step j = 1,…,H, set uⱼ = 2j/(L − 1). The continuation basis is

bⱼ = [1, 1 + uⱼ, 1 + d(3uⱼ + 1.5d uⱼ²)]ᵀ, with d = 0.25,

and Tⱼ(a) = βᵀbⱼ. This expression damps both the quadratic coefficient's linear contribution beyond the endpoint and its additional curvature. The linear basis column itself is not damped.

**Proposition 1: orthogonal residualization.** Qᵀr(a) = 0, and a = QQᵀa + r(a) is an orthogonal decomposition.

**Proof.** Since QᵀQ = I, Qᵀ(a − QQᵀa) = Qᵀa − Qᵀa = 0. Thus the fitted component and residual have zero inner product. This statement concerns the observed interval; future continuation is defined separately by bⱼ.

### Normalized spectral–temporal retrieval

For each cleaned history a, define s(a) = max(sqrt(mean(r(a)²)), 10⁻⁸) and z(a) = r(a)/s(a). For memory pair (kᵢ,vᵢ), store zᵢ = z(kᵢ) and fᵢ = [vᵢ − T(kᵢ)]/s(kᵢ), using the first H continuation entries.

Let F(z) be the real FFT magnitude vector with its DC entry set to zero, divided by max(its Euclidean norm, 10⁻¹²). Rank memory entries by

Sᵢ = 0.75 zᵢᵀz(x)/L + 0.25 F(zᵢ)ᵀF(z(x)).

Take the largest min(12,N) scores using a complete ascending sort followed by selection of the final entries. The temporal term carries phase information; the magnitude term compares frequency content.

### Ridge continuation transfer

Let A ∈ ℝᴸˣᴷ contain the selected normalized memory histories as columns, let B ∈ ℝᴴˣᴷ contain their normalized continuations, and let q = z(x). Solve

w* = argmin_w {‖Aw − q‖₂²/L + 0.05‖w‖₂²}

by the direct linear system

(AᵀA/L + 0.05I)w* = Aᵀq/L.

The final prediction is ŷ = T(x) + s(x)Bw*. Coefficients may be negative and need not sum to one. The implementation uses a linear solve, not a matrix inverse or a Moore–Penrose pseudoinverse. Its regularization coefficient is lower-bounded by 10⁻¹⁰ when supplied through configuration; the evaluated value is 0.05.

**Proposition 2: uniqueness and comparator inequality.** The fitted coefficients are unique. For every comparator u,

‖Aw* − q‖₂²/L ≤ ‖Au − q‖₂²/L + λ(‖u‖₂² − ‖w*‖₂²),

where λ = 0.05.

**Proof.** AᵀA/L is positive semidefinite, so adding λI makes the normal matrix positive definite. The stated inequality follows by comparing the minimized objective at w* and u and rearranging. For uniform coefficients u = 1/K, the additive term is λ(1/K − ‖w*‖₂²). This is a reconstruction-objective guarantee, without an assumption equating historical reconstruction error with future forecast risk.

### Prediction procedure

```text
Input: query x, memory keys k[1:N], continuations v[1:N], horizon H
1. Clip isolated spikes in x and each memory key.
2. Project cleaned histories onto the quadratic Legendre basis.
3. Compute residual RMS scales and damped trend continuations.
4. Normalize residual histories and memory continuations.
5. Compute DC-free normalized real-FFT magnitudes.
6. Rank keys by 0.75 temporal correlation + 0.25 spectral similarity.
7. Select at most 12 keys and solve the constant-ridge normal equations.
8. Return query trend continuation + query scale × weighted continuations.
```

### Computational cost and caching

With fixed polynomial degree, bank preparation costs O(NL log L + NH), apart from the constant-width median operations. A cached query costs O(L log L + NL + N log N + LK² + K³ + HK); the N log N term reflects the implemented full sort. Normalized bank histories, continuations, and Fourier magnitudes require approximately 8N[L + H + floor(L/2) + 1] bytes. For N = 252, L = 64, H = 24, this is 243,936 bytes of feature payload. Retaining the original keys and continuations accounts for another 177,408 bytes. These are array-payload calculations, excluding working arrays and object overhead. The bank cache retains strong references, keys entries by array identity and preprocessing settings, and holds at most 24 entries.

## Experiments
### Protocol and metrics

Each split contains 42 query trajectories, six from each of seven named regimes: ETTh1 Thermal, ETTm1 Load, Weather Barometric, Electricity Grid, Traffic Flow, Exchange Drift, and Solar Harmonic. The benchmark synthesizes polynomial drift, multiple harmonic components, AR(1) observation noise, and occasional isolated historical spikes. Validation and held-out test generation use seeds 510 and 920. Each full split has a shared memory of 252 key–continuation pairs. Forecasts are compared with clean future trajectories at L = 64 and H = 24.

MSE and MAE are averaged across horizons and then series. Spectral error is the mean squared difference between the real-FFT magnitudes of mean-centered predictions and targets. Drift error is the squared difference between their fitted future linear slopes. Reported comparisons use the supplied evaluation results. The evaluated validation and test implementations have identical source hashes and identical parameter settings.

For baseline b and series i, define Δᵢ = MSEᵢ(b) − MSEᵢ(ours). The paired Student-t interval is mean(Δ) ± 2.0195 sd(Δ)/sqrt(42). Positive differences favor our method. The table reports means with their 95% confidence intervals; these intervals apply to paired differences, not to individual baseline means.

### Primary validation and held-out comparison

| Method | Validation MSE | Validation paired Δ [95% CI] | Test MSE | Test paired Δ [95% CI] |
|---|---:|---|---:|---|
| Our method | 0.078471 | Reference | 0.134549 | Reference |
| Vanilla TS-RAG benchmark baseline | 0.523566 | 0.445095 [0.293091, 0.597099] | 0.478836 | 0.344287 [0.183436, 0.505138] |
| Matched constant-ridge control | 0.102766 | 0.024295 [−0.010436, 0.059027] | 0.252176 | 0.117628 [0.055280, 0.179975] |
| Moving-average trend/seasonal surrogate | 1.181559 | 1.103088 [0.761638, 1.444537] | 1.240957 | 1.106408 [0.706974, 1.505842] |
| Fourier harmonic surrogate | 0.783882 | 0.705411 [0.511763, 0.899058] | 0.788769 | 0.654220 [0.428832, 0.879608] |
| Softmax-kNN retrieval surrogate | 0.365157 | 0.286686 [0.133562, 0.439809] | 0.329745 | 0.195197 [0.092883, 0.297510] |

The vanilla baseline uses five z-normalized cosine neighbors, uniform continuation averaging, and a 0.7 retrieval/persistence blend. The matched control uses quadratic Legendre projection, twelve temporal neighbors, λ = 0.05, and undamped polynomial continuation. The moving-average surrogate uses a 13-point trend filter, a linear fit over the last 24 trend samples, and repeated seasonal residuals. The Fourier surrogate retains four modes after linear detrending. The softmax-kNN surrogate uses eight neighbors and temperature 0.25.

Our relative MSE reductions against the vanilla baseline are 85.01% on validation and 71.90% on test. The test paired intervals exclude zero for all five comparisons. The validation interval against the matched ridge control includes zero.

| Split | Our MSE [95% CI] | MAE | Spectral error | Drift error |
|---|---|---:|---:|---:|
| Validation | 0.078471 [0.053122, 0.103819] | 0.213580 | 0.295230 | See note below |
| Held-out test | 0.134549 [0.077963, 0.191134] | 0.277409 | 0.360720 | 0.000215717 |

The validation MSE interval reports the across-series Student-t 95% confidence half-width. 

### Complete 2×2 mechanism isolation

The four factorial conditions below preserve the exact recorded condition names. Adaptive gating is off throughout; the evaluated predictor also has spectral forecast blending off. The selected model equals the first row.

| Exact factorial condition | Validation MSE | Validation MAE | Validation spectral error | Test MSE | Test MAE | Test spectral error |
|---|---:|---:|---:|---:|---:|---:|
| Trend Decoupling: On, Ridge Regression: On (Full) | 0.078471 | 0.213580 | 0.295230 | 0.134549 | 0.277409 | 0.360720 |
| Trend Decoupling: On, Ridge Regression: Off (Uniform Top-K) | 0.206692 | 0.354215 | 1.787539 | 0.244646 | 0.401069 | 1.695570 |
| Trend Decoupling: Off (Mean Level), Ridge Regression: On | 0.212990 | 0.331425 | 0.674055 | 0.139041 | 0.292250 | 0.649488 |
| Trend Decoupling: Off (Mean Level), Ridge Regression: Off | 0.453281 | 0.516020 | 3.380377 | 0.346592 | 0.468408 | 3.139653 |

Disabling trend separation replaces polynomial projection with mean subtraction and constant-level continuation. Disabling ridge replaces fitted coefficients with uniform neighbor weights. On validation, enabling ridge reduces MSE by 0.128221 with trend separation and by 0.240291 without it. Enabling trend separation reduces MSE by 0.134519 with ridge and by 0.246589 without it. The difference-in-differences is −0.112070 when defined as the ridge improvement with trend separation minus the ridge improvement without it. Both mechanisms improve the selected validation configuration.

### Five single-component subtraction experiments

Each experiment changes one component relative to the selected estimator. All five removals strictly increase validation MSE.

| Component removed | Exact intervention | Validation MSE | MSE increase | Validation MAE |
|---|---|---:|---:|---:|
| None | Full evaluated estimator | 0.078471 | 0 | 0.213580 |
| Orthogonal trend separation | Mean subtraction and constant continuation | 0.212990 | 0.134519 | 0.331425 |
| Ridge prototype regression | Uniform neighbor coefficients | 0.206692 | 0.128221 | 0.354215 |
| Continuation damping | Set d from 0.25 to 1.0 | 0.095299 | 0.017829 | 0.231530 |
| Spectral retrieval contribution | Set η from 0.25 to 0 | 0.088680 | 0.011210 | 0.231190 |
| Isolated-spike preprocessing | Use unmodified histories | 0.078982 | 0.000511 | 0.214267 |

The largest observed subtraction effects concern trend separation and fitted prototype coefficients. Spectral retrieval and damping provide additional improvements. Spike preprocessing yields a smaller positive difference in the recorded aggregate MSE.

### Noise robustness and scaling

The robustness sweep adds Gaussian noise to query histories using perturbation seed 808. Memory and evaluation targets remain fixed. Each split is evaluated at four added-noise standard deviations.

| Added noise SD | Our validation MSE | Vanilla validation MSE | Validation reduction | Our test MSE | Vanilla test MSE | Test reduction |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.078471 | 0.523566 | 85.01% | 0.134549 | 0.478836 | 71.90% |
| 0.15 | 0.099398 | 0.527117 | 81.14% | 0.136083 | 0.469582 | 71.02% |
| 0.30 | 0.116927 | 0.594723 | 80.34% | 0.148967 | 0.502729 | 70.37% |
| 0.45 | 0.196022 | 0.560642 | 65.04% | 0.201591 | 0.456705 | 55.86% |

The accompanying surrogate results are:

| Added noise SD | Moving-average validation MSE | Fourier validation MSE | Moving-average test MSE | Fourier test MSE |
|---|---:|---:|---:|---:|
| 0.00 | 1.181559 | 0.783882 | 1.240957 | 0.788769 |
| 0.15 | 1.203909 | 0.783849 | 1.295265 | 0.783513 |
| 0.30 | 1.271825 | 0.808369 | 1.363317 | 0.828091 |
| 0.45 | 1.523094 | 0.829523 | 1.596119 | 0.841208 |

For computational scaling, the implemented operations imply the following dependencies at fixed polynomial degree. These entries are analytical costs.

| Operation | Time scaling | Principal storage |
|---|---|---|
| Bank residual and Fourier preparation | O(NL log L + NH) | O(N(L + H)) |
| Query similarity computation | O(NL) | O(N) scores |
| Full neighbor sorting | O(N log N) | O(N) indices |
| Ridge fitting and continuation | O(LK² + K³ + HK) | O(LK + HK + K²) |

### Runtime, memory, and reproducibility

The recorded environment is Python 3.12.3 on aarch64 Linux 6.17.0-1014-nvidia with glibc 2.39. The evaluator identifies an ARM Cortex-X925 at 3.90 GHz with a pinned single core. It warms the predictor once before the timed evaluation loop. That loop includes our prediction, metric computation, and all five baseline predictions; the reported per-series value therefore measures the combined evaluation loop. Memory is measured by tracemalloc around the first predictor call, after dataset construction.

| Split | Series | Bank entries | Evaluation-loop ms/series | Traced current bytes | Traced peak bytes |
|---|---:|---:|---:|---:|---:|
| Validation | 42 | 252 | 2.0310 | 2,050,103 | 4,031,540 |
| Held-out test | 42 | 252 | 2.1275 | 2,049,814 | 4,031,276 |


## Limitations
The benchmark contains synthetic trajectories with domain-inspired names; these results do not establish performance on the corresponding public real-world datasets. Memory continuations and evaluation targets are clean latent trajectories, while observed histories contain noise. This differs from deployment settings with noisy historical future values.

The vanilla retrieval implementation and the moving-average, Fourier, and softmax-kNN surrogates are local numerical baselines. They are not full reproductions of the published TS-RAG foundation-model architecture, DLinear, TimesNet, or MQRetNN. PatchTST, iTransformer, SERAF, and RATL are contextual comparisons rather than evaluated competitors. Several supplied reference abstracts are truncated, so the literature comparison is restricted to the mechanisms explicitly described in those materials.

Our empirical evaluation covers one validation split and one held-out test split, with six queries per regime and shared memory within each split. Student-t intervals use series-level paired differences without a hierarchical dependence model or multiple-comparison correction. The validation difference against the matched ridge control is not statistically resolved by its interval. Individual subtraction effects, especially the small preprocessing effect, have no supplied paired confidence intervals.

Our robustness experiments evaluate additive query noise across four intensity levels. They do not include controlled bank-size, lookback-length, or forecast-horizon sweeps; the scaling table gives computational dependencies rather than measured scaling experiments. The subset setting changes both memory size and query composition, so it cannot isolate bank-size effects. Runtime values include baseline execution and scoring, and tracemalloc measurements exclude preconstructed dataset storage and do not represent total process resident memory. Repeated timing distributions, numerical-library versions, and independent affinity verification are unavailable.

The projection and ridge propositions guarantee algebraic properties of observed-history reconstruction, not future-risk dominance. Damping is a fixed continuation assumption. The evaluated model does not use a spectral forecast mixture or adaptive confidence gate, so the results provide no evidence for those mechanisms.

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

## Appendix A: Complete Experimental Trajectory Manifest

| Experiment | mse |
|---|---|
| baseline_source | 0.115334534 |
| baseline_subset | 0.139684685 |
| baseline_full | 0.115334534 |
| initial_subset | 0.104673044 |
| initial_full | 0.089908764 |
| seed0_subset | 0.104673044 |
| seed0_full | 0.089908764 |
| evolved0_subset | 0.107273115 |
| evolved0_full | 0.091247156 |
| ablation0 | 0.212990091 |
| ablation1 | 0.206691948 |
| ablation2 | 0.095299466 |
| ablation3 | 0.088680420 |
| ablation4 | 0.078982044 |
| pruned | 0.078471168 |
| rebuttal0 | 0.128116095 |
| final_test | 0.134548677 |
