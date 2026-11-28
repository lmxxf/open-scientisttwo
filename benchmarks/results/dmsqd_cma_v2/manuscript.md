# Frontier-Guided Quality Diversity through Descriptor Inversion and Local Quadratic Refinement

## Abstract
We propose Frontier-Guided Quality Diversity (FGQD), a derivative-free optimizer that combines diagonal covariance adaptation, descriptor-directed exploration, and local quadratic quality refinement under a fixed evaluation budget. FGQD estimates descriptor displacement from neighboring archive elites and observed parent–offspring transitions, then approximately inverts this model to target sparse behavior regions. A separate regularized quadratic model of negative log quality generates bounded refinement proposals. Strict archive thresholds, predicted directional progress, and explicit empty-cell priority determine candidate selection. Our theoretical analysis establishes monotonic archive improvement, stability and residual contraction of regularized descriptor inversion, and descent for the deterministic quadratic refinement operator. Our experiments evaluate 24–64-dimensional decision spaces with two-dimensional descriptors and 672 search evaluations per task. Across 12 validation and 12 held-out tasks, FGQD achieves normalized quality-diversity scores of 0.755878 and 0.743752, respectively. Held-out scores for isotropic MAP-Elites, diagonal CMA-ME, and fixed-annealing diagonal CMA-MAE are 0.296115, 0.559610, and 0.572905. Paired Student-t 95% confidence intervals favor FGQD against all three comparators. A complete covariance-by-annealing factorial experiment, five subtraction comparisons, dimension scaling, and an independently generated higher-conditioning suite characterize the contributions and computational footprint of the method.

## Introduction
Quality-diversity optimization seeks a repertoire of solutions that are simultaneously high-performing and behaviorally distinct. An archive partitions behavior space into niches and retains the best solution discovered in each niche. With a limited number of evaluations, an optimizer must allocate effort between discovering empty niches and improving occupied ones. These objectives interact: a mutation that improves quality can remain behaviorally redundant, whereas a large behavior change can substantially reduce quality.

We study a black-box oracle returning quality $f(x)\in(0,1]$ and descriptor $b(x)\in[-1,1]^2$ for a decision vector $x\in[-2.5,2.5]^d$. A regular grid contains $M=g^2$ descriptor cells. For archive $\mathcal A$, define the zero-filled cell quality and normalized quality-diversity score as
$$q_e=\max\bigl(\{f(x):x\in\mathcal A_e\}\cup\{0\}\bigr),\qquad Q(\mathcal A)=\frac{1}{M}\sum_{e=1}^{M}q_e.$$
The optimization loss is $L=1-Q$. Both increased coverage and improved quality within occupied cells increase $Q$.

Our approach separates two uses of local observations. Descriptor observations identify directions that change behavior, while quality observations identify directions that improve an elite. We propose FGQD, which combines these two model-directed operators with an adaptive stochastic emitter. Its descriptor model uses both secants between nearby archive elites and displacements observed during evaluated transitions. A regularized inverse maps a sparse-region target into a bounded decision-space step. Its quality model fits a diagonal quadratic to negative log quality and proposes bounded curvature-scaled refinement steps.

The archive remains strictly elitist. Candidate ranking combines improvement over the incumbent cell quality, predicted progress toward a descriptor target, and priority for cells empty at the start of the batch. The resulting method shares one evaluation budget across central stochastic sampling, archive-centered exploration, frontier proposals, and quality refinement.

Our contributions are threefold. First, we specify a compact black-box optimizer that couples descriptor inversion with local quality refinement and diagonal adaptation. Second, we provide self-contained guarantees for archive accounting and the deterministic proposal operators. Third, our experiments evaluate matched-budget baseline comparisons, a complete factorial mechanism study, subtraction effects around the proposed configuration, scaling across decision dimensions, and robustness to independently generated landscapes with stronger quadratic conditioning. Together, these analyses distinguish gains in behavior coverage from gains in elite quality.

## Related Work
Adaptive archive search provides the foundation for FGQD. Fontaine et al. (2019), in [Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space](https://arxiv.org/abs/1912.02400), combine CMA-style adaptation with diversity-preserving archives. FGQD retains adaptive stochastic search and adds explicit descriptor inversion and quadratic refinement. Fontaine and Nikolaidis (2022), in [Covariance Matrix Adaptation MAP-Annealing](https://arxiv.org/abs/2205.10752), address premature objective abandonment, flat objectives, and low-resolution archives through adaptive acceptance thresholds. FGQD uses strict incumbent thresholds; our factorial experiment evaluates annealing as a separate intervention.

Tjanaka et al. (2026), in [Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](https://arxiv.org/abs/2601.01082), identify distortion in high-dimensional measure spaces and replace histogram-based discount representations with a smooth continuous model. Their experiments include image-valued measures. FGQD models a different quantity: decision-to-descriptor displacement. It retains a discrete archive and constructs proposals through a regularized inverse of the displacement model.

Several approaches exploit directional information or heterogeneous search operators. Fontaine and Nikolaidis (2021), in [Differentiable Quality Diversity](https://arxiv.org/abs/2106.03894), formulate QD with differentiable objectives and measures and introduce gradient-arborescence exploration. FGQD estimates local directions from black-box observations. Cully (2020), in [Multi-Emitter MAP-Elites](https://arxiv.org/abs/2007.05352), combines heterogeneous emitters with bandit-based allocation. FGQD also combines proposal types, using fixed batch allocations and one shared diagonal adaptation state. Lim, Flageat, and Cully (2022), in [Efficient Exploration using Model-Based Quality-Diversity with Gradients](https://arxiv.org/abs/2211.12610), use model-based gradients for exploitation and imagined perturbations for exploration. FGQD instead fits small regression models directly to evaluated parameter, descriptor, and quality observations.

Surrogate-assisted QD provides a closely related route to evaluation efficiency. Zhang et al. (2021), in [Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding](https://arxiv.org/abs/2112.03534), train an online deep outcome predictor and use the evolving archive to improve its training data. FGQD uses a local diagonal quadratic quality model and a separate linear descriptor-displacement model; every proposed candidate receives a true oracle evaluation before archive insertion.

Other work expands the task structure and application domains of QD. Anne and Mouret (2023), in [Multi-Task Multi-Behavior MAP-Elites](https://arxiv.org/abs/2305.01264), combine behavioral diversity with transfer across related tasks and evaluate humanoid fault recovery. FGQD constructs each task's repertoire independently. Brych and Cully (2020), in [Competitiveness of MAP-Elites against Proximal Policy Optimization on locomotion tasks in deterministic simulations](https://arxiv.org/abs/2009.08438), compare evolutionary archive search with reinforcement learning and study robustness across seeds and hyperparameters. Their work motivates reporting repertoire performance separately from single-policy quality. Schrum and Crosby (2025), in [A Quality Diversity Approach to Evolving Model Rockets](https://arxiv.org/abs/2504.02177), compare MAP-Elites, CMA-ME, and CMA-MAE in rocket design and evaluate manufactured rockets. Their reported differences in design diversity motivate our separate coverage and quality analyses. Schneider et al. (2022), in [A Collection of Quality Diversity Optimization Problems Derived from Hyperparameter Optimization of Machine Learning Models](https://arxiv.org/abs/2204.14061), formulate QD problems with measures such as interpretability and resource usage and use surrogate benchmark evaluations for efficiency. FGQD addresses the optimizer rather than the construction of an application-specific benchmark.

The following comparison summarizes all eleven works and the proposed method. Each row identifies the central mechanism described by the corresponding work.

| Work | Central mechanism or setting | Relation to FGQD |
|---|---|---|
| Tjanaka et al. (2026), Discount Model Search | Continuous discount modeling for high-dimensional measures | FGQD models descriptor displacement and inverts it to generate proposals |
| Fontaine and Nikolaidis (2022), CMA-MAE | Annealed archive thresholds | FGQD uses strict incumbent thresholds and evaluates annealing factorially |
| Fontaine et al. (2019), CMA-ME | Covariance-adapting archive emitters | FGQD adds frontier inversion and local quality refinement to diagonal adaptation |
| Fontaine and Nikolaidis (2021), Differentiable QD | Objective and measure gradients; gradient arborescences | FGQD estimates directions from oracle observations |
| Cully (2020), Multi-Emitter MAP-Elites | Heterogeneous emitters with bandit allocation | FGQD uses fixed proposal allocations and shared adaptation |
| Anne and Mouret (2023), Multi-Task Multi-Behavior MAP-Elites | Transfer across tasks and within-task diversity | FGQD optimizes tasks independently |
| Schrum and Crosby (2025), Evolving Model Rockets | Simulated and physical rocket-design repertoires | FGQD evaluates quality and coverage on controlled continuous landscapes |
| Brych and Cully (2020), MAP-Elites versus PPO | Locomotion performance and robustness | FGQD studies evaluation-budget allocation in continuous archive search |
| Zhang et al. (2021), Deep Surrogate Assisted MAP-Elites | Online deep outcome prediction for deckbuilding | FGQD uses local quadratic quality and linear displacement models |
| Lim et al. (2022), Model-Based QD with Gradients | Model-based exploitation and imagined exploration | FGQD constructs bounded steps from directly fitted local regressions |
| Schneider et al. (2022), QD Hyperparameter Benchmarks | Interpretability and resource measures with surrogate evaluations | FGQD provides an optimizer for evaluated decision–behavior mappings |
| FGQD, ours | Descriptor inversion, quadratic refinement, diagonal adaptation | Strict true-evaluation archive with shared finite-budget proposal generation |

## Methodology
### Archive and batch structure

FGQD maintains elite decisions $x_e$, qualities $q_e$, and observed descriptors $b_e$ in a $g\times g$ archive. Internally, empty cells have quality $-\infty$; they contribute zero to the reported score. The emitter maintains mean $m\in\mathbb R^d$, diagonal variance $v\in\mathbb R_{>0}^d$, evolution paths $p_c,p_\sigma$, and scalar step size $\sigma$. Initialization is $m=p_c=p_\sigma=0$, $v=\mathbf1$, and $\sigma=0.45$.

For batch size $B$, each generation first draws $B$ central proposals $m+\sigma\sqrt v\odot z$, with independent standard Gaussian $z$. When elites exist, the first $\lfloor B/3\rfloor$ proposals are replaced by mutations of uniformly sampled archive elites, using the same diagonal sampling scale. Once at least four elites exist, these positions are replaced by frontier proposals. Once enough evaluated history exists, the last $\max(1,\lfloor B/4\rfloor)$ positions are replaced by quadratic-refinement proposals. In our experiments, $B=24$, giving eight frontier positions and six refinement positions when both operators are active. All remaining positions retain central proposals. Candidate vectors are clipped coordinatewise to $[-2.5,2.5]$ before evaluation.

### Regularized regression

For an input matrix $A$ and target matrix or vector $Y$, define
$$\mathcal R_\lambda(A,Y)=\arg\min_W\bigl\{\|AW-Y\|_F^2+\lambda\|W\|_F^2\bigr\}.$$
For $\lambda>10^{-12}$, FGQD computes
$$\mathcal R_\lambda(A,Y)=(A^\top A+\lambda I)^{-1}A^\top Y$$
by a linear solve. For $\lambda\le10^{-12}$, it computes the minimum-norm Moore–Penrose least-squares solution using a relative singular-value cutoff of $10^{-10}$. Thus removing descriptor regularization changes the estimator to minimum-norm least squares. The proposed method uses constant descriptor regularization $\lambda_b=0.1$ and constant quality regularization $\lambda_f=0.05$ across dimensions. Every regression coefficient, including the quality-model intercept, is regularized.

### Descriptor-directed frontier proposals

Let $n$ be the number of occupied cells. For elite $e$, let $N_e$ contain its $k=\min(n-1,15)$ nearest other elites in descriptor distance. Write $r_e$ for the median neighbor distance and define
$$\nu_e=\frac{k^{-1}\sum_{j\in N_e}\|b_j-b_e\|_2}{\max\{2/g,\operatorname{median}_{\ell}r_\ell\}}.$$
Each frontier parent is sampled independently with probability proportional to $\nu_e+0.1$. Its outward direction and initial descriptor target are
$$a_e=\frac{b_e-k^{-1}\sum_{j\in N_e}b_j}{\max\{\|b_e-k^{-1}\sum_{j\in N_e}b_j\|_2,10^{-8}\}},\qquad t_e=\operatorname{clip}_{[-0.995,0.995]^2}(b_e+r_ea_e).$$
If this target lies in an occupied cell and empty cells remain, the target becomes the center of the empty cell nearest to $t_e$.

A local displacement model is
$$J_e=\mathcal R_{0.1}\bigl([x_j-x_e]_{j\in N_e},[b_j-b_e]_{j\in N_e}\bigr)\in\mathbb R^{d\times2}.$$
After at least $2B$ paired transitions have been observed, we also fit $J_T=\mathcal R_{0.1}(D_x,D_b)$ using the latest at most 384 evaluated parent–offspring displacements. We use $J=(J_e+J_T)/2$ when this transition estimate is available and $J=J_e$ otherwise. Frontier generation is activated by archive occupancy without an additional statistical activation test.

Using column-vector notation for $r=t_e-b_e$, the inverse step is
$$s_b=J(J^\top J+0.02I_2)^{-1}r,\qquad \widetilde s_b=s_b\min\{1,1.5/\max(\|s_b\|_2,10^{-8})\}.$$
The proposal is $x_e+\widetilde s_b+0.12\sqrt v\odot z$. Each evaluated archive-centered proposal supplies a transition observation with its actual parent descriptor.

### Local quadratic quality refinement

Refinement begins after at least $\max(2d,3B)$ evaluated candidates have been accumulated. Let $x_\star$ be the highest-quality archive elite. From the latest at most 384 evaluated candidates, select the nearest $\min\{H,\max(2d+8,100)\}$ candidates to $x_\star$, where $H$ is the available window length. For $u=x-x_\star$, construct features
$$\phi(u)=[1,u_1,\ldots,u_d,\tfrac12u_1^2,\ldots,\tfrac12u_d^2].$$
Fit $\beta=\mathcal R_{0.05}(\Phi,y)$ with $y=-\log\max(f,10^{-12})$. Let $a$ be the fitted linear coefficients and $h_j=\max(\beta_{d+j},0.05)$ the curvature coefficients, using one-based indexing after the intercept. Define $H_f=\operatorname{diag}(h)$.

The first refinement parent is $x_\star$; subsequent refinement parents are sampled uniformly from occupied elites. For parent $x_e$, the gradient of the curvature-adjusted model is
$$g_e=a+H_f(x_e-x_\star),\qquad s_f=-H_f^{-1}g_e.$$
The proposed point is
$$x_e+s_f\min\{1,0.9/\max(\|s_f\|_2,10^{-8})\}+0.06z.$$
The quality-noise scale is isotropic and independent of $\sigma$ and $v$.

### Candidate ranking and archive update

Let $\tau_e$ equal the incumbent elite quality in occupied cells and zero in empty cells. Thresholds are strict: after processing a candidate, $\tau_e$ becomes the resulting archive elite quality. Before any within-batch archive updates, compute $\Delta_i=f(x_i)-\tau_{e(x_i)}$ and record whether the destination cell is empty.

For a vector $z\in\mathbb R^B$, define the tie-aware percentile rank
$$R_i(z)=\frac{|\{j:z_j<z_i\}|+\tfrac12|\{j:z_j=z_i\}|}{B}.$$
For directed candidates, predicted directional progress is
$$p_i=\left\langle J_T^\top(x_i-x_{\mathrm{parent},i}),\frac{t_i-b_{\mathrm{parent},i}}{\max(\|t_i-b_{\mathrm{parent},i}\|_2,10^{-8})}\right\rangle.$$
Progress is zero for other candidates and for all candidates before $J_T$ is available. If directed proposals occur, the selection score is
$$S_i=0.75R_i(\Delta)+0.25R_i(p)+\mathbf1\{e(x_i)\text{ was empty before the batch}\}.$$
Otherwise, $S_i=R_i(\Delta)+\mathbf1\{e(x_i)\text{ was empty before the batch}\}$. This uses a common pre-update occupancy state for every candidate. Archive insertion subsequently processes the evaluated batch in order and replaces an elite only when the new quality is strictly larger.

### Diagonal adaptation

Let $\mu=\max(1,\lfloor B/2\rfloor)$, and select the $\mu$ largest scores. Define normalized positive recombination weights
$$w_i=\frac{\log(\mu+0.5)-\log i}{\sum_{j=1}^{\mu}[\log(\mu+0.5)-\log j]},\qquad \mu_{\mathrm{eff}}=(\sum_iw_i^2)^{-1}.$$
For selected candidates, use the old mean and scale to form $y_i=(x_i-m)/\sigma$ and $\bar y=\sum_iw_iy_i$. The updated mean is $m^+=\sum_iw_ix_i$. Set
$$c_c=\frac{4+\mu_{\mathrm{eff}}/d}{d+4+2\mu_{\mathrm{eff}}/d},\quad c_s=\frac{\mu_{\mathrm{eff}}+2}{d+\mu_{\mathrm{eff}}+5},\quad c_1=\frac{2}{(d+1.3)^2+\mu_{\mathrm{eff}}},$$
$$c_\mu=\min\left\{1-c_1,\frac{2(\mu_{\mathrm{eff}}-2+1/\mu_{\mathrm{eff}})}{(d+2)^2+\mu_{\mathrm{eff}}}\right\},$$
$$d_s=1+c_s+2\max\left\{0,\sqrt{\frac{\mu_{\mathrm{eff}}-1}{d+1}}-1\right\}.$$
The path and variance updates are
$$p_s^+=(1-c_s)p_s+\sqrt{c_s(2-c_s)\mu_{\mathrm{eff}}}\,\bar y\oslash\sqrt v,$$
$$p_c^+=(1-c_c)p_c+\sqrt{c_c(2-c_c)\mu_{\mathrm{eff}}}\,\bar y,$$
$$v^+=\operatorname{clip}_{[0.03,12]}\left((1-c_1-c_\mu)v+c_1(p_c^+)^{\odot2}+c_\mu\sum_iw_iy_i^{\odot2}\right).$$
With $\chi_d=\sqrt d(1-1/(4d)+1/(21d^2))$, update
$$\sigma^+=\operatorname{clip}_{[0.08,0.8]}\left[\sigma\exp\left(\operatorname{clip}_{[-0.12,0.12]}\left\{\frac{c_s}{d_s}\left(\frac{\|p_s^+\|_2}{\chi_d}-1\right)\right\}\right)\right].$$
Disabling adaptation in controlled comparisons fixes $v=\mathbf1$ and $\sigma=0.45$, while retaining mean recombination.

```text
Algorithm 1: Frontier-Guided Quality Diversity
Input: oracle, decision dimension d, grid width g, generations G, batch B
Initialize empty elite archive, zero thresholds, emitter, and histories
For generation = 1,...,G:
    Draw B central Gaussian proposals
    If paired history has at least 2B observations:
        Fit transition displacement model on its latest 384 observations
    If archive is nonempty:
        Draw archive parents and overwrite the first floor(B/3) proposals
        If archive contains at least four elites:
            Overwrite those positions using Algorithm 2 frontier proposals
        If evaluated history contains at least max(2d,3B) observations:
            Fit the local quadratic model around the best elite
            Overwrite the last max(1,floor(B/4)) positions with refinement proposals
    Clip all candidates to the decision domain
    Evaluate exactly B candidates with the true oracle
    Compute batch scores using pre-update thresholds and occupancy
    Insert strict elite improvements and set thresholds to incumbent qualities
    Recombine the highest-ranked half and update diagonal adaptation state
    Append evaluated candidates and paired transitions to histories
Return elite decisions and qualities
```

```text
Algorithm 2: Frontier Proposal by Descriptor Inversion
Input: elite archive, diagonal variance v, optional transition model J_T
Compute descriptor distances and up to 15 neighbors per elite
Compute local radii and sparsity-weighted parent probabilities
Sample an elite parent
Construct its outward descriptor target
If the target cell is occupied and empty cells remain:
    Replace target with the nearest empty-cell center
Fit local descriptor secants with ridge strength 0.1
Average local and transition models when the latter is available
Solve the two-dimensional regularized inverse with strength 0.02
Cap the decision-step norm at 1.5
Add Gaussian noise with coordinate scales 0.12 sqrt(v)
Return proposal, parent, parent descriptor, and target
```

### Theoretical analysis

**Proposition 1: Monotone archive improvement and exact search budget.** Suppose oracle evaluation is deterministic and returns $f(x)\in[0,1]$. For any random proposal sequence, the archive score after each insertion is nondecreasing and belongs to $[0,1]$. Running $G$ complete generations uses exactly $GB$ search evaluations. Independently verifying the final archive uses at most $M$ additional evaluations.

**Proof.** In an occupied cell, replacement occurs only when the new quality exceeds the incumbent. In an empty cell, insertion changes the zero-filled contribution from zero to a nonnegative value. Consequently every cell contribution is nondecreasing, and so is their normalized sum. Since every contribution lies in $[0,1]$, the normalized sum also lies in that interval. Each generation evaluates one batch of size $B$; all regressions use observations already collected. Thus search requires exactly $GB$ evaluations. There is at most one stored elite per cell, yielding the verification bound. $\square$

**Theorem 1: Stability of regularized displacement fitting and contraction of the predicted residual.** Let $\lambda>0$. For any perturbation $E$ to regression targets,
$$\|\mathcal R_\lambda(A,Y+E)-\mathcal R_\lambda(A,Y)\|_F\le\frac{\|E\|_F}{2\sqrt\lambda}.$$
For any $J\in\mathbb R^{d\times2}$, $r\in\mathbb R^2$, and $\eta>0$, let $s=J(J^\top J+\eta I)^{-1}r$. If the step is scaled by $0<\theta\le1$, then
$$r-J^\top(\theta s)=\left[(1-\theta)I+\theta\eta(J^\top J+\eta I)^{-1}\right]r.$$
Hence its predicted residual norm does not exceed $\|r\|_2$. If $J$ has full column rank and smallest singular value $a>0$, then
$$\|r-J^\top(\theta s)\|_2\le\left(1-\theta+\frac{\theta\eta}{a^2+\eta}\right)\|r\|_2.$$

**Proof.** An SVD of $A$ shows that the target-to-coefficient linear operator has singular values $s_j/(s_j^2+\lambda)$. For $s_j\ge0$, these are bounded above by $1/(2\sqrt\lambda)$, proving the stability inequality. Next, write $K=J^\top J$. Since $K(K+\eta I)^{-1}=I-\eta(K+\eta I)^{-1}$, substitution gives the residual identity. The residual operator has eigenvalues $1-\theta+\theta\eta/(\kappa_j+\eta)$, where $\kappa_j\ge0$ are eigenvalues of $K$. These belong to $[0,1]$. Under full column rank, $\kappa_j\ge a^2$, giving the stated bound. $\square$

For a differentiable descriptor map satisfying $b(x+\theta s)-b(x)=J^\top\theta s+e$ with $\|e\|_2\le\varepsilon$, the true target residual is bounded by the corresponding predicted bound plus $\varepsilon$. This follows directly from the triangle inequality.

**Proposition 2: Descent of curvature-adjusted quadratic refinement.** Let
$$\widehat\ell(u)=\beta_0+a^\top u+\tfrac12u^\top H_fu,\qquad H_f\succeq0.05I.$$
At any $u$, define $g=a+H_fu$, $s=-H_f^{-1}g$, and $0<\theta\le1$. Then
$$\widehat\ell(u+\theta s)-\widehat\ell(u)=-\theta(1-\theta/2)g^\top H_f^{-1}g\le0.$$
If the difference between the true negative-log-quality change and this predicted change has magnitude at most $\varepsilon$, true quality strictly improves whenever $\theta(1-\theta/2)g^\top H_f^{-1}g>\varepsilon$.

**Proof.** Exact quadratic expansion gives $\theta g^\top s+(\theta^2/2)s^\top H_fs$. Substituting $s=-H_f^{-1}g$ yields the identity. Positive definiteness implies nonnegativity of $g^\top H_f^{-1}g$, and $\theta(1-\theta/2)>0$. The error assumption gives the sufficient condition for a negative true log-loss change; exponentiation then gives strictly increased quality. $\square$

### Time and space complexity

Let $n\le M$ be archive occupancy, $H\le384$ the fitting window, and $k\le15$ the local neighbor count. Pairwise descriptor distances cost $O(n^2)$ time and space, while sorting every neighbor list costs $O(n^2\log n)$. A dense primal descriptor regression costs $O(Hd^2+d^3)$; each of $O(B)$ local regressions costs $O(kd^2+d^3)$. The inverse solve is only two-dimensional after forming its matrices. Quality regression has $2d+1$ features and costs $O(Hd^2+d^3)$. Searching empty-cell centers costs $O(BM)$, and candidate ranking costs $O(B\log B)$.

A conservative per-generation bound, excluding oracle work, is
$$O\bigl(M^2\log M+(H+Bk)d^2+Bd^3+BM+Bd+B\log B\bigr).$$
Total search work multiplies this bound by $G$ and adds $GB$ oracle evaluations. The benchmark oracle uses a dense rotation, costing $O(d^2)$ arithmetic per candidate.

The implementation retains accumulated evaluated and transition histories even though fitting uses bounded windows. Its total space is therefore
$$O(Md+M^2+GBd+d^2+Bd).$$
Diagonal adaptation itself requires $O(d)$ state. With float64 storage, elite decisions require $8Md$ bytes: 73,728 bytes at $M=144,d=64$. A full $144\times144$ descriptor-distance matrix requires 165,888 bytes. A dense quality-regression normal matrix at $d=64$ has $129^2$ entries, requiring 133,128 bytes before factorization workspaces.

## Experiments
### Benchmark, evaluation, and statistical protocol

Our experiments evaluate four decision dimensions, $d\in\{24,36,48,64\}$, with three independently generated tasks per dimension in each primary evaluation set. Every task uses a $12\times12$ descriptor archive, $G=28$ generations, and $B=24$ candidates, giving 672 search evaluations per optimizer. Final elites are independently reevaluated and reassigned to their true descriptor cells before scoring.

The objective combines a rotated anisotropic quadratic and a periodic ripple. Let $x_c=\operatorname{clip}_{[-2.5,2.5]^d}(x)$, let $U$ be orthogonal, and define $z=U^\top(x_c-c)$. The oracle returns
$$f(x)=\exp\left[-0.85\left(\frac1d\sum_{j=1}^{d}\lambda_jz_j^2+\frac{0.15}{6}\sum_{j=1}^{6}[1-\cos(2\pi z_j)]+\frac{0.25}{d}\sum_{j=1}^{d}\max(0,|x_j|-2.5)^2\right)\right].$$
The eigenvalues are geometrically spaced from 0.4 to 4.5. Centers are sampled uniformly from $[-0.8,0.8]^d$. For an orthonormal two-column matrix $V$, let $P=\sqrt{d/2.5}\,V$ and $\ell=P^\top x_c$. The descriptors are
$$b_1(x)=\tanh(0.75\ell_1+0.20\sin\ell_2),\qquad b_2(x)=\tanh(0.75\ell_2-0.20\cos\ell_1).$$
Optimization uses only oracle-returned qualities and descriptors. The validation and held-out landscape generators use seeds 710 and 1120, respectively. Tasks are ordered by increasing dimension and then replicate; the optimizer seed for zero-based task index $i$ is the corresponding generator seed plus $37i$. The higher-conditioning suite uses generator seed 7710 and raises the upper quadratic eigenvalue to 5.4 while preserving the remaining experimental settings.

We compare isotropic MAP-Elites, strict diagonal CMA-ME, and fixed-annealing diagonal CMA-MAE. MAP-Elites initializes uniformly in $[-1.2,1.2]^d$ and subsequently mutates uniformly sampled archive elites with Gaussian standard deviation 0.35. The diagonal CMA comparators initialize at zero with step size 0.42, use logarithmic half-batch recombination weights, and update diagonal sampling factors at rate $2/(d+6)$ with bounds $[0.15,4]$. Strict and annealed variants use threshold update rates 1 and 0.15, respectively; an unproductive batch triggers mean reinitialization. All comparators receive the same search budget as FGQD.

We report $L=1-Q$, normalized QD score, coverage, mean occupied-cell quality, and CVaR-10% quality. CVaR-10% is the mean of the lowest $\max(1,\lfloor n/10\rfloor)$ occupied-cell qualities. Each aggregate is the arithmetic mean of its task-level metric.

For comparator $b$, paired differences are $D_i=L_{b,i}-L_{\mathrm{FGQD},i}$. Their Student-t 95% interval is
$$\overline D\pm2.2010\,s_D/\sqrt{12}.$$
Positive differences also equal QD-score gains. Subtraction intervals use the same expression with the modified method as comparator.

### Primary validation and held-out comparisons

The final column reports FGQD's paired loss reduction relative to the row's comparator, followed by its 95% interval.

| Set | Method | Loss ↓ | QD score ↑ | Coverage (%) ↑ | FGQD gain [95% CI] |
|---|---|---:|---:|---:|---|
| Validation | MAP-Elites | 0.712111 | 0.287889 | 86.28 | 0.467989 [0.434870, 0.501108] |
| Validation | Diagonal CMA-ME | 0.449596 | 0.550404 | 92.36 | 0.205474 [0.164273, 0.246675] |
| Validation | Diagonal CMA-MAE | 0.435906 | 0.564094 | 92.77 | 0.191783 [0.150385, 0.233181] |
| Validation | FGQD | 0.244122 | 0.755878 | 98.90 | Reference |
| Held-out | MAP-Elites | 0.703885 | 0.296115 | 87.96 | 0.447637 [0.420566, 0.474708] |
| Held-out | Diagonal CMA-ME | 0.440390 | 0.559610 | 93.17 | 0.184142 [0.149548, 0.218736] |
| Held-out | Diagonal CMA-MAE | 0.427095 | 0.572905 | 93.34 | 0.170846 [0.130874, 0.210818] |
| Held-out | FGQD | 0.256248 | 0.743752 | 99.13 | Reference |

Every primary paired interval lies strictly above zero. Relative to the annealed diagonal comparator, FGQD increases held-out QD by 0.170846 and coverage by 5.79 percentage points. Its held-out score is 0.012126 below its validation score, while coverage increases slightly. The gains therefore persist across independently generated landscapes and combine broader illumination with stronger retained solutions.

| Evaluation set | QD score ↑ | Coverage (%) ↑ | Mean elite ↑ | Mean task maximum ↑ | CVaR-10% ↑ |
|---|---:|---:|---:|---:|---:|
| Validation | 0.755878 | 98.90 | 0.764073 | 0.900682 | 0.598128 |
| Held-out | 0.743752 | 99.13 | 0.750166 | 0.886984 | 0.586209 |
| Higher conditioning | 0.740763 | 98.96 | 0.748377 | 0.886519 | 0.561141 |

### Complete covariance-by-annealing factorial study

The following four rows form the complete $2\times2$ experiment. Frontier proposals are absent in every factorial cell; quality refinement and the remaining settings are held fixed. Annealing uses update rate 0.15. The condition identifiers preserve the exact experimental conditions. The integrated FGQD result appears separately below the four cells.

| Condition | Validation loss ↓ | Held-out loss ↓ | Validation QD ↑ | Held-out QD ↑ | Validation coverage (%) ↑ |
|---|---:|---:|---:|---:|---:|
| Covariance Adaptation: On, Soft Annealing: On | 0.271029 | 0.274723 | 0.728971 | 0.725277 | 97.34 |
| Covariance Adaptation: On, Soft Annealing: Off | 0.276919 | 0.276283 | 0.723081 | 0.723717 | 96.93 |
| Covariance Adaptation: Off, Soft Annealing: On | 0.406532 | 0.411416 | 0.593468 | 0.588584 | 91.61 |
| Covariance Adaptation: Off, Soft Annealing: Off | 0.415484 | 0.409439 | 0.584516 | 0.590561 | 91.61 |
| Integrated FGQD | 0.244122 | 0.256248 | 0.755878 | 0.743752 | 98.90 |

With strict thresholds, adaptation increases validation QD by 0.138565 and held-out QD by 0.133156. With annealing, the corresponding gains are 0.135503 and 0.136693. Averaging across threshold conditions gives adaptation main effects of 0.137034 on validation and approximately 0.134925 on held-out tasks.

Annealing increases validation QD by 0.005890 with adaptation and by 0.008952 without adaptation. On held-out tasks it changes QD by +0.001560 and −0.001977, respectively. The covariance-by-annealing difference-in-differences is −0.003062 on validation and +0.003537 on held-out tasks. Adaptation thus contributes the dominant factorial effect, while the effect of annealing changes with the evaluation set and adaptation condition.

Restoring frontier proposals to the adapted, strict-threshold cell adds 0.032797 validation QD and 0.020035 held-out QD. On the higher-conditioning suite, the same frontier contrast adds 0.022736 QD. These contrasts retain quality refinement and isolate the integrated effect of frontier proposal generation and its associated ranking behavior.

### Five subtraction comparisons

Each comparison below uses the proposed configuration as its reference. The table reports the actual intervention, including the joint removal in the final row. Positive loss increases indicate degraded mean validation performance.

| Intervention | Validation loss ↓ | Loss increase [95% CI] | Coverage (%) ↑ | CVaR-10% ↑ | Interpretation |
|---|---:|---|---:|---:|---|
| Full FGQD | 0.244122 | Reference | 98.90 | 0.598128 | Integrated method |
| Remove frontier guidance | 0.276919 | 0.032797 [0.020530, 0.045065] | 96.93 | 0.558927 | Descriptor-directed exploration |
| Remove quadratic refinement | 0.302821 | 0.058699 [0.041977, 0.075420] | 98.55 | 0.575982 | Local quality improvement |
| Remove adaptive distribution updates | 0.324685 | 0.080563 [0.063138, 0.097988] | 97.28 | 0.527726 | Variance and step-size adaptation |
| Remove descriptor ridge penalty | 0.244844 | 0.000721 [−0.009075, 0.010518] | 98.90 | 0.592859 | Minimum-norm displacement fitting |
| Remove frontier and refinement jointly | 0.346169 | 0.102046 [0.085256, 0.118837] | 96.93 | 0.543931 | Combined model-directed operators |

All five interventions strictly increase mean validation loss. Removing refinement reduces mean elite quality from 0.764073 to 0.707328 while coverage remains 98.55%, identifying quality improvement as its main observed contribution. Removing frontier guidance reduces both coverage and tail quality. Removing adaptive distribution updates causes the largest single-module loss increase and lowers CVaR-10% to 0.527726. Descriptor regularization has a small positive retention point estimate, with an interval spanning zero.

The combined frontier-and-refinement loss increase, 0.102046, exceeds either individual increase. The individual increases sum to 0.091496; their difference from the joint increase is 0.010551. This quantifies the observed nonadditivity of the two operators under the common reference configuration.

### Dimension scaling and conditioning robustness

Each dimension row averages three tasks. The higher-conditioning suite increases the quadratic condition number from $4.5/0.4=11.25$ to $5.4/0.4=13.5$ and uses independently generated landscapes.

| Suite | Decision dimension | QD score ↑ | Coverage (%) ↑ | Mean elite ↑ | CVaR-10% ↑ |
|---|---:|---:|---:|---:|---:|
| Validation | 24 | 0.755865 | 100.00 | 0.755865 | 0.538556 |
| Validation | 36 | 0.784586 | 99.31 | 0.790073 | 0.628149 |
| Validation | 48 | 0.767266 | 99.07 | 0.774431 | 0.631314 |
| Validation | 64 | 0.715794 | 97.22 | 0.735922 | 0.594493 |
| Held-out | 24 | 0.755197 | 100.00 | 0.755197 | 0.572871 |
| Held-out | 36 | 0.738249 | 99.07 | 0.745008 | 0.575688 |
| Held-out | 48 | 0.767855 | 99.07 | 0.774948 | 0.614692 |
| Held-out | 64 | 0.713705 | 98.38 | 0.725509 | 0.581585 |
| Higher conditioning | 24 | 0.795555 | 99.77 | 0.797394 | 0.584670 |
| Higher conditioning | 36 | 0.725076 | 99.31 | 0.730188 | 0.541033 |
| Higher conditioning | 48 | 0.720167 | 98.61 | 0.730171 | 0.543021 |
| Higher conditioning | 64 | 0.722254 | 98.15 | 0.735754 | 0.575842 |

FGQD attains complete coverage at dimension 24 in both primary sets. At dimension 64, held-out coverage remains 98.38% and QD remains above 0.713. The higher-conditioning suite attains aggregate QD 0.740763 with 98.96% coverage, and every dimension retains more than 98% coverage. The descriptor-regularization contrast additionally evaluates two estimator regimes, ridge strength 0.1 and zero-penalty minimum-norm fitting, with respective validation losses 0.244122 and 0.244844.

### Runtime, memory, and reproducibility

The recorded environment is Python 3.12.3 on Linux 6.17.0, aarch64, glibc 2.39, with an ARM Cortex-X925 at 3.90 GHz and a reported pinned single core. Numerical arrays use NumPy float64 and random draws use NumPy's default generator. The timings below cover the complete 12-task primary evaluation loop: FGQD or its intervention, all three baseline optimizers, and final archive verification. The separate factorial experiment occurs after this timing interval.

| Evaluation workload | Tasks | Search evaluations per optimizer/task | Wall time (s) | Peak traced bytes | Current traced bytes |
|---|---:|---:|---:|---:|---:|
| Validation | 12 | 672 | 5.9880 | 3,899,272 | 1,035,096 |
| Held-out | 12 | 672 | 5.9973 | 3,915,183 | 1,037,540 |
| Higher conditioning | 12 | 672 | 5.9737 | 3,971,656 | 1,046,884 |
| Without frontier | 12 | 672 | 3.6376 | 1,996,859 | 59,454 |
| Without refinement | 12 | 672 | 4.7123 | 3,248,857 | 1,045,932 |
| Without adaptive distribution updates | 12 | 672 | 5.7706 | 3,893,410 | 1,039,167 |
| Without descriptor ridge penalty | 12 | 672 | 6.0899 | 3,935,398 | 1,038,181 |
| Without frontier and refinement | 12 | 672 | 2.3746 | 1,112,222 | 62,193 |

The integrated primary workloads take approximately six seconds with peak traced allocations below four million bytes. Frontier and refinement removal reduce workload time alongside their measured quality losses. These results expose the computational cost of the model-directed operators while preserving an identical oracle budget.

## Limitations
Our evidence concerns one synthetic landscape family with 24–64 decision variables, two-dimensional descriptors, a 144-cell archive, and 672 search evaluations per task. It does not establish performance for high-dimensional measure spaces, image-valued descriptors, physical systems, large archives, or substantially larger evaluation budgets. The available scaling study varies decision dimension, and the robustness study changes both landscape realizations and quadratic conditioning. It is therefore not a paired intervention isolating conditioning alone. Beyond the two descriptor-regularization settings, no broad sensitivity sweep over budgets, resolutions, noise levels, or regularization strengths is available.

The named CMA-ME and CMA-MAE comparators are specified diagonal implementations rather than complete reproductions of all mechanisms in the published algorithms. Their adaptation, initialization, and restart rules differ from FGQD. Consequently, the primary comparisons establish performance against these matched-budget implementations, while the factorial study provides the more controlled evidence about adaptation and annealing. We do not empirically compare against Discount Model Search, differentiable QD, heterogeneous bandit emitters, or deep surrogate-assisted methods. The literature comparison identifies methodological relationships rather than demonstrating superiority over all cited work.

There are four single-component interventions and one joint frontier-and-refinement intervention in the five subtraction comparisons. Calling all five independent single-component removals would misidentify the final intervention. Although every subtraction increases mean validation loss, the descriptor-regularization interval includes zero, so its small point estimate does not establish a reliable benefit. Removing adaptive distribution updates disables both diagonal variance adaptation and scalar step-size adaptation; its effect cannot be attributed exclusively to covariance adaptation. Frontier removal also changes proposal-dependent ranking behavior. Thus that intervention measures the complete frontier mechanism rather than the inverse proposal operator alone.

The paired confidence intervals pool 12 matched tasks spanning four dimensions. They quantify variation across this task collection rather than repeated optimizer seeds on one fixed landscape. Factorial summaries support point contrasts but do not provide task-level observations for uncertainty estimates on every interaction. These intervals are unadjusted for multiple comparisons. Mean and tail-quality measurements for FGQD and its interventions are available, but the primary baseline summaries do not support equivalent paired distributional analyses.

Our theoretical results establish archive elitism, evaluation accounting, regression stability, linear-model residual contraction, and descent of a curvature-adjusted quadratic. They do not establish global convergence or guaranteed discovery of an empty cell. Model approximation error, added Gaussian noise, and decision-domain clipping can alter descriptor progress or quality improvement. The curvature floor changes the fitted quadratic, so the descent guarantee concerns the explicitly curvature-adjusted model. Diagonal adaptation is coordinate-dependent and does not imply general rotational invariance. Because selected candidates combine central, elite-centered, and model-directed proposals, the complete update is not established to be an exact Fisher natural-gradient step.

Complete pairwise neighbor sorting and repeated dense primal regression solves can become expensive as archives or decision dimensions grow. Although fitting windows contain at most 384 observations, the retained histories grow with the evaluation budget. Recorded runtime measures the combined evaluation workload rather than isolated optimizer latency, and traced allocations do not represent complete process resident memory. Exact timing reproduction also depends on numerical-library versions, linear algebra backends, and machine scheduling beyond the reported environment.

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

## Appendix A: Complete Experimental Trajectory Manifest

| Experiment | mse |
|---|---|
| baseline_source | 0.220737953 |
| baseline_subset | 0.227024839 |
| baseline_full | 0.220737953 |
| initial_subset | 0.262446139 |
| initial_full | 0.253030415 |
| seed0_subset | 0.262446139 |
| seed0_full | 0.253030415 |
| evolved0_subset | 0.262446139 |
| evolved0_full | 0.253030415 |
| ablation0 | 0.276919474 |
| ablation1 | 0.302820774 |
| ablation2 | 0.324684866 |
| ablation3 | 0.244843739 |
| ablation4 | 0.346168715 |
| pruned | 0.244122267 |
| rebuttal0 | 0.259237033 |
| final_test | 0.256248418 |
