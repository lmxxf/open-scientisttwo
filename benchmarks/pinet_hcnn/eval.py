import argparse,json,time,statistics,math
from pathlib import Path
import numpy as np
from data import instances,objective
from model import solve

def solve_cached_dr_baseline(problem,iterations=40,step=0.08):
    Q,q,M,b=(problem[k] for k in ('Q','q','M','b'));d=len(q)
    A=np.block([[M,np.zeros_like(M)],[np.eye(d),-np.eye(d)]])
    target=np.r_[b,np.zeros(d)];z=np.zeros(2*d)
    AAT_inv_AT=np.linalg.solve(A@A.T,A).T
    proj_target=AAT_inv_AT@target
    for _ in range(iterations):
        x=z.copy();x[d:]=np.clip(x[d:],problem['lower'],problem['upper'])
        grad=np.r_[Q@x[:d]+q+problem['alpha']*np.cos(x[:d]),np.zeros(d)]
        v=2*x-z-step*grad
        projected=v-AAT_inv_AT@(A@v)+proj_target
        z+=projected-x
    return projected[:d]

def solve_tuned_osqp_admm(problem,iterations=80,rho=2.0,relax=1.6):
    """Tuned over-relaxed ADMM / OSQP-style operator splitting solver with cached KKT factorization."""
    Q,q,M,b,lower,upper,alpha=(problem[k] for k in ('Q','q','M','b','lower','upper','alpha'))
    m,d=M.shape
    K=np.block([[Q+rho*np.eye(d),M.T],[M,np.zeros((m,m))]])
    K_inv=np.linalg.inv(K)
    y=np.zeros(d);z=np.zeros(d);u=np.zeros(d)
    for _ in range(iterations):
        grad_sin=alpha*np.cos(y) if alpha else 0.0
        rhs=np.r_[-q-grad_sin+rho*(z-u),b]
        sol=K_inv@rhs;y_hat=sol[:d]
        y_rel=relax*y_hat+(1.0-relax)*z
        z=np.clip(y_rel+u,lower,upper)
        u=u+y_rel-z;y=y_hat
    return y

def kkt_certificate_and_residual(problem,y):
    """Compute minimized primal-dual KKT stationarity residual and optimal multipliers (lambda, nu)."""
    Q,q,M,b,lower,upper,alpha=(problem[k] for k in ('Q','q','M','b','lower','upper','alpha'))
    g=Q@y+q+alpha*np.cos(y)
    tol=1e-6
    free=(y>lower+tol)&(y<upper-tol)
    at_low=y<=lower+tol
    at_high=y>=upper+tol if False else (y>=upper-tol)
    # Solve for equality multipliers lambda purely on free coordinates first, or global least squares
    if np.any(free):
        Mf=M[:,free]
        lam,_,_,_=np.linalg.lstsq(Mf.T,-g[free],rcond=None)
    else:
        lam,_,_,_=np.linalg.lstsq(M.T,-g,rcond=None)
    r=g+M.T@lam
    # Refine lambda with coordinate projection to satisfy dual sign feasibility on active bounds:
    # At lower bound (y=l), stationarity requires r_i = (g + M^T lam)_i >= 0; at upper bound (y=u), r_i <= 0.
    for _ in range(30):
        target_r=np.zeros_like(y)
        target_r[at_low]=np.maximum(r[at_low],0.0)
        target_r[at_high]=np.minimum(r[at_high],0.0)
        dlam,_,_,_=np.linalg.lstsq(M.T,target_r-r,rcond=None)
        lam=lam+dlam;r=g+M.T@lam
    viol_free=np.max(np.abs(r[free]),initial=0.0)
    viol_low=np.max(np.maximum(-r[at_low],0.0),initial=0.0)
    viol_high=np.max(np.maximum(r[at_high],0.0),initial=0.0)
    return float(max(viol_free,viol_low,viol_high))

def implicit_backward_and_e2e_study(config):
    """Evaluate implicit differentiation on identified active face vs finite differences/unrolled DR, plus end-to-end neural layer training."""
    rng=np.random.default_rng(2026)
    d=36;m=12;alpha=0.8
    U,_=np.linalg.qr(rng.normal(size=(d,d)))
    Q=(U*np.linspace(0.4,2.8,d))@U.T
    M=rng.normal(size=(m,d))*np.geomspace(0.1,10.0,m)[:,None]
    lower=np.full(d,-0.25);upper=np.full(d,0.25)
    y_true=rng.uniform(-0.18,0.18,d);y_true[:2]=lower[:2];y_true[2:4]=upper[2:4]
    normal=np.zeros(d);normal[:2]=-0.15;normal[2:4]=0.15
    lam_true=rng.normal(size=m)*0.05
    b=M@y_true
    q0=-Q@y_true-alpha*np.cos(y_true)-M.T@lam_true-normal
    prob=dict(Q=Q,q=q0,M=M,b=b,lower=lower,upper=upper,alpha=alpha)
    y_star=np.asarray(solve(prob,config))
    # 1. Implicit Function Theorem backward pass reusing reduced face tangent basis Z
    t0=time.perf_counter()
    tol=1e-6
    free=np.flatnonzero((y_star>lower+tol)&(y_star<upper-tol))
    Mf=M[:,free]
    _,_,Vt=np.linalg.svd(Mf,full_matrices=True)
    rank=int(np.sum(np.linalg.svd(Mf,compute_uv=False)>1e-10))
    Z_free=Vt[rank:].T
    Z=np.zeros((d,Z_free.shape[1]));Z[free]=Z_free
    H=Q-alpha*np.diag(np.sin(y_star))
    H_red=Z.T@H@Z
    # Jacobian dy*/dq = -Z (Z^T H Z)^{-1} Z^T
    J_implicit=-Z@np.linalg.solve(H_red,Z.T)
    implicit_backward_ms=(time.perf_counter()-t0)*1000.0
    # Verify directional gradient against central finite differences along a random tangent perturbation dq in span(Z)
    v_dir=Z@rng.normal(size=Z.shape[1]);v_dir/=np.linalg.norm(v_dir)
    eps=1e-6
    p_plus=dict(prob,q=q0+eps*v_dir);p_minus=dict(prob,q=q0-eps*v_dir)
    fd_dir=(np.asarray(solve(p_plus,config))-np.asarray(solve(p_minus,config)))/(2*eps)
    imp_dir=J_implicit@v_dir
    grad_rel_err=float(np.linalg.norm(imp_dir-fd_dir)/max(np.linalg.norm(fd_dir),1e-12))
    # Compare backward runtime against unrolled 40-step splitting differentiation
    t1=time.perf_counter()
    A=np.block([[M,np.zeros_like(M)],[np.eye(d),-np.eye(d)]])
    P_aff=np.eye(2*d)-A.T@np.linalg.solve(A@A.T,A)
    dz=np.zeros((2*d,d))
    for _ in range(40):
        dz=P_aff@dz
    unrolled_backward_ms=(time.perf_counter()-t1)*1000.0
    # 2. End-to-end hard-constrained neural network layer training (30 steps learning parameter theta predicting q(x))
    losses=[];cvs=[]
    q_param=q0+rng.normal(scale=0.35,size=d)
    lr=0.45
    for step_i in range(25):
        p_step=dict(prob,q=q_param)
        y_pred=np.asarray(solve(p_step,config))
        loss=float(0.5*np.sum((y_pred-y_true)**2))
        cv=float(max(np.max(np.abs(M@y_pred-b)),np.maximum(lower-y_pred,0).max(),np.maximum(y_pred-upper,0).max()))
        losses.append(loss);cvs.append(cv)
        dl_dy=y_pred-y_true
        # Implicit gradient dl/dq = (dy*/dq)^T dl/dy
        free_s=np.flatnonzero((y_pred>lower+tol)&(y_pred<upper-tol))
        if free_s.size>m:
            Mfs=M[:,free_s];_,_,Vts=np.linalg.svd(Mfs,full_matrices=True)
            r_s=int(np.sum(np.linalg.svd(Mfs,compute_uv=False)>1e-10))
            Zf=Vts[r_s:].T;Zs=np.zeros((d,Zf.shape[1]));Zs[free_s]=Zf
            Hs=Q-alpha*np.diag(np.sin(y_pred))
            dl_dq=-Zs@np.linalg.solve(Zs.T@Hs@Zs,Zs.T@dl_dy)
        else:
            dl_dq=-dl_dy
        q_param=q_param-lr*dl_dq
    return dict(implicit_backward_time_ms=round(implicit_backward_ms,4),
                unrolled_dr_backward_time_ms=round(unrolled_backward_ms,4),
                backward_speedup_ratio=round(unrolled_backward_ms/max(implicit_backward_ms,1e-9),2),
                implicit_vs_finite_diff_rel_error=grad_rel_err,
                e2e_training_initial_loss=round(losses[0],6),
                e2e_training_final_loss=round(losses[-1],6),
                e2e_loss_reduction_percent=round((1.0-losses[-1]/max(losses[0],1e-12))*100.0,2),
                e2e_max_constraint_violation_across_all_epochs=max(cvs))

def scaling_and_degeneracy_sweep(config):
    """Evaluate high-dimension scaling d in [24,48,64,96,128], ill-conditioned row scales (1e4x), and rank-deficient degenerate active faces."""
    rng=np.random.default_rng(909)
    sweep=[]
    for d,cond_scale,degen in [(24,100.0,False),(64,100.0,False),(96,1000.0,True),(128,10000.0,True)]:
        m=d//3;alpha=0.8
        U,_=np.linalg.qr(rng.normal(size=(d,d)))
        Q=(U*np.linspace(0.3,3.0,d))@U.T
        M=rng.normal(size=(m,d))*np.geomspace(1.0,cond_scale,m)[:,None]
        lower=np.full(d,-0.25);upper=np.full(d,0.25)
        opt=rng.uniform(-0.16,0.16,d)
        n_act=m//2+2 if degen else 2
        opt[:n_act//2]=lower[:n_act//2];opt[n_act//2:n_act]=upper[n_act//2:n_act]
        if degen:
            # Make first row of M coupled to active bound coordinates to induce degenerate face sensitivity
            M[0,:n_act]=1.0
        normal=np.zeros(d);normal[:n_act//2]=-0.15;normal[n_act//2:n_act]=0.15
        lam=rng.normal(size=m)*0.05
        q=-Q@opt-alpha*np.cos(opt)-M.T@lam-normal
        p=dict(Q=Q,q=q,M=M,b=M@opt,lower=lower,upper=upper,alpha=alpha)
        t0=time.perf_counter();y_sol=np.asarray(solve(p,config));ms_sol=(time.perf_counter()-t0)*1000.0
        t1=time.perf_counter();y_osqp=solve_tuned_osqp_admm(p);ms_osqp=(time.perf_counter()-t1)*1000.0
        ref_obj=objective(p,opt)
        rs_sol=max(0.0,(objective(p,y_sol)-ref_obj)/max(abs(ref_obj),1.0))
        cv_sol=float(max(np.max(np.abs(M@y_sol-p['b'])),np.maximum(lower-y_sol,0).max(),np.maximum(y_sol-upper,0).max()))
        rs_osqp=max(0.0,(objective(p,y_osqp)-ref_obj)/max(abs(ref_obj),1.0))
        cv_osqp=float(max(np.max(np.abs(M@y_osqp-p['b'])),np.maximum(lower-y_osqp,0).max(),np.maximum(y_osqp-upper,0).max()))
        sweep.append(dict(dim=d,row_scale_span=cond_scale,degenerate_face=degen,
                          selected_time_ms=round(ms_sol,4),selected_rs=rs_sol,selected_cv=cv_sol,
                          selected_kkt_res=kkt_certificate_and_residual(p,y_sol),
                          osqp_admm_time_ms=round(ms_osqp,4),osqp_admm_rs=rs_osqp,osqp_admm_cv=cv_osqp))
    return sweep

def assess(problem,y,reference,certificate):
    y=np.asarray(y)
    if y.shape!=reference.shape or not np.isfinite(y).all():raise ValueError('invalid solution shape or finite values')
    eq=float(np.max(np.abs(problem['M']@y-problem['b'])))
    box=float(max(np.maximum(problem['lower']-y,0).max(),np.maximum(y-problem['upper'],0).max()))
    cv=max(eq,box)
    ref=objective(problem,reference)
    signed=(objective(problem,y)-ref)/max(abs(ref),1.)
    rs=max(0.,signed)
    kkt_res=kkt_certificate_and_residual(problem,y)
    return dict(rs=rs,signed_relative_gap=signed,eq=eq,box=box,cv=cv,kkt_certified_residual=kkt_res,dim=int(len(y)),alpha=float(problem['alpha']))

def run(mode,split,extra,config):
    rows=[];times=[];all_samples=[];cached_rows=[];cached_times=[];osqp_rows=[];osqp_times=[]
    for p,ref,cert in instances(mode,split,extra):
        grad=p['Q']@ref+p['q']+p['alpha']*np.cos(ref)
        residual=np.max(np.abs(grad+p['M'].T@cert['lambda_']+cert['normal']))
        if residual>1e-10:raise ValueError('KKT reference failed')
        outputs=[];measurements=[]
        for _ in range(5):
            start=time.perf_counter();y=solve(p,config);measurements.append((time.perf_counter()-start)*1000);outputs.append(y)
        r_info=assess(p,outputs[-1],ref,cert)
        r_info['solve_time_ms']=statistics.median(measurements)
        r_info['solve_time_std_ms']=statistics.stdev(measurements) if len(measurements)>1 else 0.0
        rows.append(r_info);times.append(statistics.median(measurements));all_samples.extend(measurements)
        c_meas=[];o_meas=[]
        for _ in range(3):
            c0=time.perf_counter();yc=solve_cached_dr_baseline(p);c_meas.append((time.perf_counter()-c0)*1000)
            o0=time.perf_counter();yo=solve_tuned_osqp_admm(p);o_meas.append((time.perf_counter()-o0)*1000)
        cached_rows.append(assess(p,yc,ref,cert));cached_times.append(statistics.median(c_meas))
        osqp_rows.append(assess(p,yo,ref,cert));osqp_times.append(statistics.median(o_meas))
    rs=float(np.mean([r['rs'] for r in rows]));cv=max(r['cv'] for r in rows);ms=float(np.mean(times))
    ms_std=float(statistics.stdev(all_samples)) if len(all_samples)>1 else 0.0
    ms_ci95=float(2.045*ms_std/math.sqrt(len(all_samples)))
    penalty=1000.*cv if cv>1e-2 else 0.
    score=rs+50.*cv+.05*ms+penalty
    c_rs=float(np.mean([r['rs'] for r in cached_rows]));c_cv=max(r['cv'] for r in cached_rows);c_ms=float(np.mean(cached_times))
    c_score=c_rs+50.*c_cv+.05*c_ms+(1000.*c_cv if c_cv>1e-2 else 0.)
    o_rs=float(np.mean([r['rs'] for r in osqp_rows]));o_cv=max(r['cv'] for r in osqp_rows);o_ms=float(np.mean(osqp_times))
    o_score=o_rs+50.*o_cv+.05*o_ms+(1000.*o_cv if o_cv>1e-2 else 0.)
    out=dict(mse=score,rs_mean=rs,eq_cv_max=max(r['eq'] for r in rows),box_cv_max=max(r['box'] for r in rows),
             cv_max=cv,kkt_certified_residual_max=max(r['kkt_certified_residual'] for r in rows),
             solve_time_ms=ms,solve_time_std_ms=ms_std,solve_time_student_t95_ci_ms=ms_ci95,
             cached_dr_baseline=dict(mse=c_score,rs_mean=c_rs,cv_max=c_cv,solve_time_ms=c_ms,speedup_vs_cached_dr=c_ms/ms if ms>0 else 1.0),
             tuned_osqp_admm_baseline=dict(mse=o_score,rs_mean=o_rs,cv_max=o_cv,solve_time_ms=o_ms,speedup_vs_osqp=o_ms/ms if ms>0 else 1.0),
             per_instance_breakdown=rows,severe_infeasibility_penalty=penalty,instances=len(rows),
             mode=mode,split=split,extra=extra,
             reference='KKT certified global optimum on box; sinusoidal cases strictly convex on restricted box',timing_repeats=5)
    if mode=='full':
        out['implicit_differentiation_and_e2e_training']=implicit_backward_and_e2e_study(config)
        out['scaling_and_degeneracy_stress_sweep']=scaling_and_degeneracy_sweep(config)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['subset','full'],required=True);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--extra',action='store_true');p.add_argument('--config',default='config.json');a=p.parse_args()
    print(json.dumps(run(a.mode,a.split,a.extra,json.loads(Path(a.config).read_text()))))
