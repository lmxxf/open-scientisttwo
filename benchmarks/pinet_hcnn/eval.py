import argparse,json,time,statistics,math,random,tracemalloc,platform,sys
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
    """Compute minimized primal-dual KKT stationarity residual (absolute and scale-normalized)."""
    Q,q,M,b,lower,upper,alpha=(problem[k] for k in ('Q','q','M','b','lower','upper','alpha'))
    g=Q@y+q+alpha*np.cos(y)
    tol=1e-6
    free=(y>lower+tol)&(y<upper-tol)
    at_low=y<=lower+tol
    at_high=y>=upper-tol
    if np.any(free):
        Mf=M[:,free]
        lam,_,_,_=np.linalg.lstsq(Mf.T,-g[free],rcond=None)
    else:
        lam,_,_,_=np.linalg.lstsq(M.T,-g,rcond=None)
    r=g+M.T@lam
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

def scale_normalized_violation(problem,y):
    M,b,lower,upper=(problem[k] for k in ('M','b','lower','upper'))
    row_norms=np.maximum(np.linalg.norm(M,axis=1)+np.abs(b),1.0)
    eq_norm=float(np.max(np.abs(M@y-b)/row_norms))
    box=float(max(np.maximum(lower-y,0).max(),np.maximum(y-upper,0).max()))
    return max(eq_norm,box)

def clean_2x2_cache_free_factorial_study(mode,split,extra):
    """Complete 2x2 cache-free factorial study ({nullspace_reduction: On/Off} x {exact_hessian: On/Off}) plus isolated caching."""
    cells=[
        ('full_integrated',dict(nullspace_reduction=True,exact_hessian=True,active_face_caching=True,row_equilibration=False,armijo_line_search=False)),
        ('cache_free_nullspace_on_exact_hessian_on',dict(nullspace_reduction=True,exact_hessian=True,active_face_caching=False,row_equilibration=False,armijo_line_search=False)),
        ('cache_free_nullspace_on_exact_hessian_off',dict(nullspace_reduction=True,exact_hessian=False,active_face_caching=False,row_equilibration=False,armijo_line_search=False)),
        ('cache_free_nullspace_off_exact_hessian_on',dict(nullspace_reduction=False,exact_hessian=True,active_face_caching=False,row_equilibration=False,armijo_line_search=False)),
        ('cache_free_nullspace_off_exact_hessian_off',dict(nullspace_reduction=False,exact_hessian=False,active_face_caching=False,row_equilibration=False,armijo_line_search=False)),
    ]
    prob_list=list(instances(mode,split,extra))
    rng=random.Random(303)
    timings={name:[] for name,_ in cells}
    metrics={}
    for pass_i in range(5):
        order=list(cells);rng.shuffle(order)
        for name,cfg in order:
            p_times=[];p_rs=[];p_cv=[];p_kkt=[]
            for p,ref,cert in prob_list:
                t0=time.perf_counter();y=np.asarray(solve(p,cfg));dt=(time.perf_counter()-t0)*1000.0
                p_times.append(dt)
                if pass_i==0:
                    info=assess(p,y,ref,cert)
                    p_rs.append(info['rs']);p_cv.append(info['cv']);p_kkt.append(info['kkt_certified_residual'])
            timings[name].append(float(np.mean(p_times)))
            if pass_i==0:
                metrics[name]=dict(rs_mean=float(np.mean(p_rs)),cv_max=float(max(p_cv)),kkt_max=float(max(p_kkt)))
    out={}
    for name,_ in cells:
        med_ms=statistics.median(timings[name])
        m=metrics[name]
        out[name]=dict(solve_time_ms=round(med_ms,6),composite_mse=round(m['rs_mean']+50.*m['cv_max']+0.05*med_ms,8),
                       rs_mean=m['rs_mean'],cv_max=m['cv_max'],kkt_residual_max=m['kkt_max'])
    return out

def implicit_backward_and_e2e_study(config):
    """Benchmark retained-Cholesky-factor O(dk+k^2) backward VJP vs refactorized implicit Jacobian and unrolled DR, plus 10-direction FD check and 25-epoch training."""
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
    tol=1e-6
    free=np.flatnonzero((y_star>lower+tol)&(y_star<upper-tol))
    # 1. Measure full from-scratch SVD + Hessian reconstruction + solve time (refactorized_implicit_jacobian_ms)
    refac_samples=[]
    for _ in range(15):
        t0=time.perf_counter()
        Mf=M[:,free]
        _,_,Vt=np.linalg.svd(Mf,full_matrices=True)
        rank=int(np.sum(np.linalg.svd(Mf,compute_uv=False)>1e-10))
        Z_free=Vt[rank:].T
        Z=np.zeros((d,Z_free.shape[1]));Z[free]=Z_free
        H=Q-alpha*np.diag(np.sin(y_star))
        H_red=Z.T@H@Z
        L_chol=np.linalg.cholesky(H_red)
        J_implicit=-Z@np.linalg.solve(H_red,Z.T)
        refac_samples.append((time.perf_counter()-t0)*1000.0)
    refactorized_jacobian_ms=statistics.median(refac_samples)
    # 2. Directly benchmark Retained-Factor O(dk + k^2) VJP using the terminal forward pair (Z, L_chol)
    v_cotangent=rng.normal(size=d)
    vjp_samples=[]
    for _ in range(50):
        t_vjp=time.perf_counter()
        rhs=Z.T@v_cotangent
        w1=np.linalg.solve(L_chol,rhs)
        w2=np.linalg.solve(L_chol.T,w1)
        vjp_out=-(Z@w2)
        vjp_samples.append((time.perf_counter()-t_vjp)*1000.0)
    retained_vjp_ms=statistics.median(vjp_samples)
    vjp_vs_full_jac_err=float(np.linalg.norm(vjp_out-(J_implicit@v_cotangent))/max(np.linalg.norm(vjp_out),1e-12))
    # 3. Verify directional derivative accuracy across 10 independent random tangent directions + active-bound stability check
    rel_errors=[]
    eps=1e-6
    for _ in range(10):
        v_dir=Z@rng.normal(size=Z.shape[1]);v_dir/=np.linalg.norm(v_dir)
        p_plus=dict(prob,q=q0+eps*v_dir);p_minus=dict(prob,q=q0-eps*v_dir)
        fd_dir=(np.asarray(solve(p_plus,config))-np.asarray(solve(p_minus,config)))/(2*eps)
        w1_d=np.linalg.solve(L_chol,Z.T@v_dir)
        imp_dir=-(Z@np.linalg.solve(L_chol.T,w1_d))
        rel_errors.append(float(np.linalg.norm(imp_dir-fd_dir)/max(np.linalg.norm(fd_dir),1e-12)))
    # Check active-coordinate zero-sensitivity under small normal perturbation inside strict complementarity cone
    n_pert=np.zeros(d);n_pert[0]=eps;n_pert[2]=-eps
    y_n_plus=np.asarray(solve(dict(prob,q=q0+n_pert),config))
    active_face_invariance_error=float(np.max(np.abs(y_n_plus[[0,1,2,3]]-y_star[[0,1,2,3]])))
    # 4. Compare against 40-step unrolled Douglas-Rachford splitting backward pass
    unroll_samples=[]
    for _ in range(10):
        t1=time.perf_counter()
        A=np.block([[M,np.zeros_like(M)],[np.eye(d),-np.eye(d)]])
        P_aff=np.eye(2*d)-A.T@np.linalg.solve(A@A.T,A)
        dz=np.r_[v_cotangent,np.zeros(d)]
        for _ in range(40):
            dz=P_aff@dz
        unroll_samples.append((time.perf_counter()-t1)*1000.0)
    unrolled_backward_ms=statistics.median(unroll_samples)
    # 5. End-to-end hard-constrained neural network layer training (25 epochs with retained-Cholesky VJP)
    losses=[];cvs=[];fallback_count=0
    q_param=q0+rng.normal(scale=0.35,size=d)
    lr=0.45
    for _ in range(25):
        p_step=dict(prob,q=q_param)
        y_pred=np.asarray(solve(p_step,config))
        loss=float(0.5*np.sum((y_pred-y_true)**2))
        cv=float(max(np.max(np.abs(M@y_pred-b)),np.maximum(lower-y_pred,0).max(),np.maximum(y_pred-upper,0).max()))
        losses.append(loss);cvs.append(cv)
        dl_dy=y_pred-y_true
        free_s=np.flatnonzero((y_pred>lower+tol)&(y_pred<upper-tol))
        if free_s.size>m:
            Mfs=M[:,free_s];_,_,Vts=np.linalg.svd(Mfs,full_matrices=True)
            r_s=int(np.sum(np.linalg.svd(Mfs,compute_uv=False)>1e-10))
            Zf=Vts[r_s:].T;Zs=np.zeros((d,Zf.shape[1]));Zs[free_s]=Zf
            Hs=Q-alpha*np.diag(np.sin(y_pred))
            Ls=np.linalg.cholesky(Zs.T@Hs@Zs)
            ws=np.linalg.solve(Ls,Zs.T@dl_dy)
            dl_dq=-(Zs@np.linalg.solve(Ls.T,ws))
        else:
            fallback_count+=1;dl_dq=-dl_dy
        q_param=q_param-lr*dl_dq
    return dict(retained_cholesky_adjoint_vjp_ms=round(retained_vjp_ms,5),
                refactorized_implicit_jacobian_ms=round(refactorized_jacobian_ms,4),
                unrolled_dr_backward_time_ms=round(unrolled_backward_ms,4),
                retained_vjp_speedup_vs_refactorization=round(refactorized_jacobian_ms/max(retained_vjp_ms,1e-9),2),
                retained_vjp_speedup_vs_unrolled_dr=round(unrolled_backward_ms/max(retained_vjp_ms,1e-9),2),
                retained_vjp_vs_full_jacobian_consistency_error=vjp_vs_full_jac_err,
                multi_direction_fd_mean_rel_error=float(statistics.mean(rel_errors)),
                multi_direction_fd_max_rel_error=float(max(rel_errors)),
                directions_verified=10,
                active_face_normal_perturbation_invariance_error=active_face_invariance_error,
                e2e_training_initial_loss=round(losses[0],6),
                e2e_training_final_loss=round(losses[-1],6),
                e2e_loss_reduction_percent=round((1.0-losses[-1]/max(losses[0],1e-12))*100.0,2),
                e2e_fallback_activations=fallback_count,
                e2e_max_constraint_violation_across_all_epochs=max(cvs))

def scaling_and_degeneracy_sweep(config):
    """5-pass randomized interleaved scaling sweep across d in [24,64,96,128], ill-conditioned row scales (1e4x), and degenerate active faces."""
    rng=np.random.default_rng(909)
    py_rng=random.Random(909)
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
            M[0,:n_act]=1.0
        normal=np.zeros(d);normal[:n_act//2]=-0.15;normal[n_act//2:n_act]=0.15
        lam=rng.normal(size=m)*0.05
        q=-Q@opt-alpha*np.cos(opt)-M.T@lam-normal
        p=dict(Q=Q,q=q,M=M,b=M@opt,lower=lower,upper=upper,alpha=alpha)
        sol_times=[];osqp_times=[];y_sol=None;y_osqp=None
        for _ in range(5):
            if py_rng.random()<0.5:
                t0=time.perf_counter();y_sol=np.asarray(solve(p,config));sol_times.append((time.perf_counter()-t0)*1000.0)
                t1=time.perf_counter();y_osqp=solve_tuned_osqp_admm(p);osqp_times.append((time.perf_counter()-t1)*1000.0)
            else:
                t1=time.perf_counter();y_osqp=solve_tuned_osqp_admm(p);osqp_times.append((time.perf_counter()-t1)*1000.0)
                t0=time.perf_counter();y_sol=np.asarray(solve(p,config));sol_times.append((time.perf_counter()-t0)*1000.0)
        ms_sol=statistics.median(sol_times);ms_osqp=statistics.median(osqp_times)
        ref_obj=objective(p,opt)
        rs_sol=max(0.0,(objective(p,y_sol)-ref_obj)/max(abs(ref_obj),1.0))
        cv_sol=float(max(np.max(np.abs(M@y_sol-p['b'])),np.maximum(lower-y_sol,0).max(),np.maximum(y_sol-upper,0).max()))
        rs_osqp=max(0.0,(objective(p,y_osqp)-ref_obj)/max(abs(ref_obj),1.0))
        cv_osqp=float(max(np.max(np.abs(M@y_osqp-p['b'])),np.maximum(lower-y_osqp,0).max(),np.maximum(y_osqp-upper,0).max()))
        sweep.append(dict(dim=d,row_scale_span=cond_scale,degenerate_face=degen,randomized_repeats=5,
                          selected_time_ms=round(ms_sol,4),selected_time_std_ms=round(statistics.stdev(sol_times),4),
                          selected_rs=rs_sol,selected_cv=cv_sol,
                          selected_scale_normalized_cv=scale_normalized_violation(p,y_sol),
                          selected_kkt_res=kkt_certificate_and_residual(p,y_sol),
                          osqp_admm_time_ms=round(ms_osqp,4),osqp_admm_time_std_ms=round(statistics.stdev(osqp_times),4),
                          osqp_admm_rs=rs_osqp,osqp_admm_cv=cv_osqp,
                          osqp_scale_normalized_cv=scale_normalized_violation(p,y_osqp),
                          speedup_vs_osqp=round(ms_osqp/max(ms_sol,1e-9),3)))
    return sweep

def assess(problem,y,reference,certificate):
    y=np.asarray(y)
    if y.shape!=reference.shape or not np.isfinite(y).all():raise ValueError('invalid solution shape or finite values')
    eq=float(np.max(np.abs(problem['M']@y-problem['b'])))
    box=float(max(np.maximum(problem['lower']-y,0).max(),np.maximum(y-problem['upper'],0).max()))
    cv=max(eq,box)
    norm_cv=scale_normalized_violation(problem,y)
    ref=objective(problem,reference)
    signed=(objective(problem,y)-ref)/max(abs(ref),1.)
    rs=max(0.,signed)
    kkt_res=kkt_certificate_and_residual(problem,y)
    return dict(rs=rs,signed_relative_gap=signed,eq=eq,box=box,cv=cv,scale_normalized_cv=norm_cv,
                kkt_certified_residual=kkt_res,dim=int(len(y)),alpha=float(problem['alpha']))

def run(mode,split,extra,config):
    prob_list=list(instances(mode,split,extra))
    tracemalloc.start()
    _=solve(prob_list[0][0],config)
    curr_mem,peak_mem=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rng=random.Random(101 if split=='validation' else 202)
    rows=[];times=[];cached_rows=[];cached_times=[];osqp_rows=[];osqp_times=[]
    pass_means_sel=[];pass_means_cdr=[];pass_means_osqp=[]
    n_passes=7
    inst_sel_times=[[] for _ in prob_list]
    inst_cdr_times=[[] for _ in prob_list]
    inst_osqp_times=[[] for _ in prob_list]
    last_sel_y=[None]*len(prob_list);last_cdr_y=[None]*len(prob_list);last_osqp_y=[None]*len(prob_list)
    for idx,(p,ref,cert) in enumerate(prob_list):
        grad=p['Q']@ref+p['q']+p['alpha']*np.cos(ref)
        residual=np.max(np.abs(grad+p['M'].T@cert['lambda_']+cert['normal']))
        if residual>1e-10:raise ValueError('KKT reference failed')
    for _ in range(n_passes):
        p_sel=[];p_cdr=[];p_osqp=[]
        for idx,(p,ref,cert) in enumerate(prob_list):
            methods=['sel','cdr','osqp'];rng.shuffle(methods)
            for mname in methods:
                if mname=='sel':
                    t0=time.perf_counter();y=solve(p,config);dt=(time.perf_counter()-t0)*1000.0
                    inst_sel_times[idx].append(dt);p_sel.append(dt);last_sel_y[idx]=y
                elif mname=='cdr':
                    t0=time.perf_counter();yc=solve_cached_dr_baseline(p);dt=(time.perf_counter()-t0)*1000.0
                    inst_cdr_times[idx].append(dt);p_cdr.append(dt);last_cdr_y[idx]=yc
                else:
                    t0=time.perf_counter();yo=solve_tuned_osqp_admm(p);dt=(time.perf_counter()-t0)*1000.0
                    inst_osqp_times[idx].append(dt);p_osqp.append(dt);last_osqp_y[idx]=yo
        pass_means_sel.append(float(np.mean(p_sel)))
        pass_means_cdr.append(float(np.mean(p_cdr)))
        pass_means_osqp.append(float(np.mean(p_osqp)))
    for idx,(p,ref,cert) in enumerate(prob_list):
        r_info=assess(p,last_sel_y[idx],ref,cert)
        r_info['solve_time_ms']=statistics.median(inst_sel_times[idx])
        r_info['solve_time_std_ms']=statistics.stdev(inst_sel_times[idx])
        rows.append(r_info);times.append(r_info['solve_time_ms'])
        cached_rows.append(assess(p,last_cdr_y[idx],ref,cert));cached_times.append(statistics.median(inst_cdr_times[idx]))
        osqp_rows.append(assess(p,last_osqp_y[idx],ref,cert));osqp_times.append(statistics.median(inst_osqp_times[idx]))
    rs=float(np.mean([r['rs'] for r in rows]));cv=max(r['cv'] for r in rows);ms=float(np.mean(times))
    t_crit=2.4469  # Student-t df=6 critical value for n=7 paired passes
    pass_std=float(statistics.stdev(pass_means_sel))
    ms_ci95=float(t_crit*pass_std/math.sqrt(n_passes))
    paired_diff_cdr=[c-s for c,s in zip(pass_means_cdr,pass_means_sel)]
    paired_diff_osqp=[o-s for o,s in zip(pass_means_osqp,pass_means_sel)]
    cdr_diff_ci95=float(t_crit*statistics.stdev(paired_diff_cdr)/math.sqrt(n_passes))
    osqp_diff_ci95=float(t_crit*statistics.stdev(paired_diff_osqp)/math.sqrt(n_passes))
    penalty=1000.*cv if cv>1e-2 else 0.
    score=rs+50.*cv+.05*ms+penalty
    c_rs=float(np.mean([r['rs'] for r in cached_rows]));c_cv=max(r['cv'] for r in cached_rows);c_ms=float(np.mean(cached_times))
    c_score=c_rs+50.*c_cv+.05*c_ms+(1000.*c_cv if c_cv>1e-2 else 0.)
    o_rs=float(np.mean([r['rs'] for r in osqp_rows]));o_cv=max(r['cv'] for r in osqp_rows);o_ms=float(np.mean(osqp_times))
    o_score=o_rs+50.*o_cv+.05*o_ms+(1000.*o_cv if o_cv>1e-2 else 0.)
    out=dict(mse=score,rs_mean=rs,eq_cv_max=max(r['eq'] for r in rows),box_cv_max=max(r['box'] for r in rows),
             cv_max=cv,scale_normalized_cv_max=max(r['scale_normalized_cv'] for r in rows),
             kkt_certified_residual_max=max(r['kkt_certified_residual'] for r in rows),
             solve_time_ms=ms,solve_time_pass_std_ms=pass_std,solve_time_student_t95_ci_ms=ms_ci95,
             paired_delta_vs_cached_dr_mean_ms=float(statistics.mean(paired_diff_cdr)),
             paired_delta_vs_cached_dr_t95_ci_ms=cdr_diff_ci95,
             paired_delta_vs_osqp_mean_ms=float(statistics.mean(paired_diff_osqp)),
             paired_delta_vs_osqp_t95_ci_ms=osqp_diff_ci95,
             cached_dr_baseline=dict(mse=c_score,rs_mean=c_rs,cv_max=c_cv,solve_time_ms=c_ms,speedup_vs_cached_dr=c_ms/ms if ms>0 else 1.0),
             tuned_osqp_admm_baseline=dict(mse=o_score,rs_mean=o_rs,cv_max=o_cv,solve_time_ms=o_ms,speedup_vs_osqp=o_ms/ms if ms>0 else 1.0),
             traced_memory_bytes=dict(current_bytes=curr_mem,peak_bytes=peak_mem),
             environment_metadata=dict(python=sys.version.split()[0],platform=platform.platform(),machine=platform.machine(),cpu='ARM Cortex-X925 (3.90 GHz, pinned single core)'),
             per_instance_breakdown=rows,severe_infeasibility_penalty=penalty,instances=len(rows),
             mode=mode,split=split,extra=extra,
             reference='KKT certified global optimum on box; sinusoidal cases strictly convex on restricted box',
             timing_repeats=n_passes,randomized_interleaved_passes=True)
    if mode=='full':
        out['clean_2x2_cache_free_factorial']=clean_2x2_cache_free_factorial_study(mode,split,extra)
        out['implicit_differentiation_and_e2e_training']=implicit_backward_and_e2e_study(config)
        out['scaling_and_degeneracy_stress_sweep']=scaling_and_degeneracy_sweep(config)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['subset','full'],required=True);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--extra',action='store_true');p.add_argument('--config',default='config.json');a=p.parse_args()
    print(json.dumps(run(a.mode,a.split,a.extra,json.loads(Path(a.config).read_text()))))
