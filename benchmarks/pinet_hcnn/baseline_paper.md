# Pinet / HCNN lifted projection-layer reference workload

Task-family reimplementation corresponding to ScientistTwo Table16 first task
and Grontas et al., arXiv:2508.10480. This is a standalone projection/optimization
kernel benchmark; it does not reproduce training a 2x200 neural network,
implicit differentiation, the original DC3 dataset, or end-to-end ANSE training.

For each context x, solve min_y 0.5*y^T Q y + q^T y + alpha*sum(sin(y)), subject
to M y=b and lower<=y<=upper. Q, q, M and b change with each context. Row scales
of M span 0.1..10. There are active lower/upper constraints. Subset dimensions
12/24, full dimensions 24/48/64, each with alpha=0 and alpha=0.8. Splits and extra
use independent deterministic seeds. No hundred-GB download or GPU is required.

Reference solutions are constructed independently from primal feasibility,
stationarity and signed bound normals: q=-Q*y_star-alpha*cos(y_star)-M^T*lambda-n.
Normals are negative at active lower bounds and positive at upper bounds.
Q has eigenvalues 0.3..3; boxes are [-0.25,0.25]. Thus
lambda_min(H)>=0.3-0.8*sin(0.25)>0 throughout the box: KKT proves a UNIQUE GLOBAL
minimum for these instances, including the sinusoidal cases. The sinusoidal
objective can be nonconvex outside that box; this suite does NOT claim to solve
arbitrary nonconvex global optimization. Reference vectors/multipliers are
withheld from solve(problem,config); the evaluator checks the KKT residual.

Baseline: lift z=(y,s), A=[[M,0],[I,-I]], Az=(b,0). Box prox clips s. A fixed
forward Douglas-Rachford / Davis-Yin three-operator step takes the objective
gradient on y, reflects the box iterate and projects onto Az=(b,0), then updates
z. This is a forward-DR variant, not an exact implementation of the original
trained Pinet layer. Each step redundantly solves A A^T in lifted space, with no
row equilibration/nullspace reduction, 40 steps and gamma=0.08. The returned
affine projection satisfies equality accurately but may violate the box.

Optimization room: y=y_p+Vw enforces equalities algebraically, equilibrate rows,
cache context factorizations, accelerate splitting and restore feasible points.
Do not replace the objective, bypass constraints or access reference solutions.

rs=max(0,(J(y)-J_ref)/max(1,abs(J_ref))). Also retain signed gap because infeasible
y may beat the feasible optimum's objective. eq_cv_max=max ||M y-b||_inf;
cv_max includes equality AND box violations. Median of 3 timings per context,
then mean solve_time_ms. Primary mse=rs_mean+50*cv_max+0.05*solve_time_ms;
if cv_max>1e-2 add 1000*cv_max, explicitly disclosed as severe infeasibility
penalty. NaN, wrong shape or corrupted KKT reference fails outright. Timing
makes this scalar noisy; use fixed CPU threads and repeated paired runs.

Source-provided ANSE targets: generated manuscript claims 2.9x-3.2x training
speed, 1e-13..1e-14 constraint residuals and 100% feasibility on its DC3 grid
(d=100/1000, 10k contexts); these are NOT measured here. ScientistTwo compares
against AutoSOTA in Table16; no numeric AutoSOTA target is asserted without its
specific row, evaluation setting and denominator. Our forward-only CPU kernel
cannot support claims about their training speed or gradient quality.
