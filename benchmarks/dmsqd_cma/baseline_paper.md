# High-Dimensional Quality-Diversity Optimization via Natural-Gradient CMA-MAE and Frontier Guidance (DMSQD Benchmark)

## 1. Problem Formulation and Degeneracies of Standard MAP-Elites and Strict CMA-ME

Quality-Diversity (QD) optimization algorithms—including `MAP-Elites` (`arXiv:2009.08438`), `Multi-Emitter MAP-Elites` (Cully, `arXiv:2007.05352`), `Multi-Task MAP-Elites` (`arXiv:2305.01264`), `CMA-ME` (Fontaine et al., `arXiv:1912.02400`), `DQD / CMA-MEGA` (Fontaine & Nikolaidis, `arXiv:2106.03894`), `CMA-MAE` (Fontaine & Nikolaidis, `arXiv:2205.10752`), rocket evolution QD (`arXiv:2504.02177`), and `DMSQD` (*Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces*, Tjanaka et al., `arXiv:2601.01082`)—seek an archive $\mathcal{A}$ over a discretized behavior descriptor tessellation $\mathcal{G} = \{e_1, \dots, e_M\} \subset [-1, 1]^2$ maximizing the normalized QD-Score:
$$\operatorname{QD}(\mathcal{A}) = \frac{1}{M} \sum_{e \in \mathcal{G}} \max_{x \in \mathcal{A}_e} f(x), \qquad \mathcal{L}_{\text{QD}}(\mathcal{A}) = 1 - \operatorname{QD}(\mathcal{A})$$
where $x \in \mathbb{R}^d$ ($d \in \{24, 36, 48, 64\}$) is evaluated under a finite evaluation budget $T_{\text{gen}} \times B$.

1. **Isotropic Random-Mutation Slowdown in High Dimensions**: Standard `MAP-Elites` perturbs elites via isotropic Gaussian noise $\mathcal{N}(0, \sigma^2 I_d)$. On ill-conditioned anisotropic objectives with condition number $\kappa \gg 1$, the probability of an isotropic step simultaneously improving $f(x)$ and expanding descriptor coverage decays exponentially as $O(\kappa^{-d/2})$.
2. **Strict-Threshold Stagnation in `CMA-ME` ($\alpha_a = 1$)**: `CMA-ME` ranks offspring by $\Delta_i = f(x_i) - f_{\mathcal{A}}(e(x_i))$, which equals $f(x_i)$ on newly discovered cells but drops abruptly to $O(\sigma)$ on occupied cells, causing the natural-gradient mean $m$ to abandon high-performing regions before ascending their local quality peak.

## 2. Theoretical Framework: Soft-Threshold Archive Annealing, Natural-Gradient CMA, and Frontier Guidance

1. **Theorem 1 (Archive Soft-Threshold Discount Annealing — `CMA-MAE` Hysteresis Principle)**: Maintaining per-cell soft acceptance thresholds $t_e^{(k)}$ initialized to $t_{\min}$ and updated via $t_e^{(k+1)} = (1 - \alpha_a) t_e^{(k)} + \alpha_a f(x)$ ($\alpha_a \in (0, 1)$) upon each accepted improvement $f(x) > t_e^{(k)}$ ensures that the cumulative improvement reward over $m$ visits to cell $e$ at constant quality $f$ is geometric:
   $$\sum_{j=0}^{m-1} \bigl(f - t_e^{(j)}\bigr) = (f - t_{\min}) \frac{1 - (1 - \alpha_a)^m}{\alpha_a},$$
   interpolating smoothly between pure objective optimization ($\alpha_a \to 0$) and strict novelty exploration ($\alpha_a = 1$).
2. **Theorem 2 (Rank-$\mu$ Natural-Gradient Covariance Adaptation)**: Ranking offspring $x_i = m + \sigma C^{1/2} z_i$ by $\Delta_i = f(x_i) - t_{e(x_i)}$ and updating $(m, C)$ via recombination weights $w_1 \ge \dots \ge w_\mu > 0$ ($\sum w_i = 1$) follows the Fisher-information natural gradient of expected archive improvement on $(\mathbb{R}^d, \mathcal{S}_{++}^d)$, achieving invariance to orthogonal search-space rotations $Q$.
3. **Proposition 3 (Frontier Boundary Directional Exploration)**: Directing a fraction of offspring along the empirical descriptor finite-difference secant $v_{\text{front}} = (x_{e_{\text{edge}}} - x_{e_{\text{int}}}) / \|x_{e_{\text{edge}}} - x_{e_{\text{int}}}\|_2$ from interior elites toward low-density boundary elites increases boundary-crossing probability into adjacent unoccupied cells.

## 3. Comprehensive Evaluation in `eval.py`
`eval.py` independently re-verifies every returned elite solution $x \in \mathcal{A}$ through the true black-box oracle `evaluate_candidates(X, task)` across dimensions $d \in \{24, 36, 48, 64\}$, reporting `mse = 1 - normalized_qd_score`, `normalized_qd_score`, `coverage_ratio`, `mean_elite_quality`, `cvar10_quality`, paired Student-$t$ 95% CIs against `Vanilla MAP-Elites` (`2009.08438`) and `CMA-ME` (`1912.02400`), and a complete $2\times 2$ factorial study.
