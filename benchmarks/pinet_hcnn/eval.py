import argparse,json,time,statistics
from pathlib import Path
import numpy as np
from data import instances,objective
from model import solve

def assess(problem,y,reference,certificate):
    y=np.asarray(y)
    if y.shape!=reference.shape or not np.isfinite(y).all():raise ValueError('invalid solution shape or finite values')
    eq=float(np.max(np.abs(problem['M']@y-problem['b'])))
    cv=max(eq,float(np.maximum(problem['lower']-y,0).max()),float(np.maximum(y-problem['upper'],0).max()))
    ref=objective(problem,reference)
    signed=(objective(problem,y)-ref)/max(abs(ref),1.)
    # Infeasible low objectives are never credited as better than a feasible optimum.
    rs=max(0.,signed)
    return dict(rs=rs,signed_relative_gap=signed,eq=eq,cv=cv)

def run(mode,split,extra,config):
    rows=[];times=[]
    for p,ref,cert in instances(mode,split,extra):
        grad=p['Q']@ref+p['q']+p['alpha']*np.cos(ref)
        residual=np.max(np.abs(grad+p['M'].T@cert['lambda_']+cert['normal']))
        if residual>1e-10:raise ValueError('KKT reference failed')
        outputs=[];measurements=[]
        for _ in range(3):
            start=time.perf_counter();y=solve(p,config);measurements.append((time.perf_counter()-start)*1000);outputs.append(y)
        rows.append(assess(p,outputs[-1],ref,cert));times.append(statistics.median(measurements))
    rs=float(np.mean([r['rs'] for r in rows]));cv=max(r['cv'] for r in rows);ms=float(np.mean(times))
    penalty=1000.*cv if cv>1e-2 else 0.
    score=rs+50.*cv+.05*ms+penalty
    return dict(mse=score,rs_mean=rs,eq_cv_max=max(r['eq'] for r in rows),cv_max=cv,solve_time_ms=ms,
                severe_infeasibility_penalty=penalty,instances=len(rows),mode=mode,split=split,extra=extra,
                reference='KKT certified global optimum on box; sinusoidal cases convex on this restricted box',timing_repeats=3)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['subset','full'],required=True);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--extra',action='store_true');p.add_argument('--config',default='config.json');a=p.parse_args()
    print(json.dumps(run(a.mode,a.split,a.extra,json.loads(Path(a.config).read_text()))))
