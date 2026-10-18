#!/usr/bin/env python3
"""Independent formulations for arithmetic review, not an independent AI reviewer."""
import argparse,csv,itertools,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,minimize
import final_round as D
F,E,A=D.F,D.E,D.A

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    rows=E.load(args.results/'seed_rows.json.gz');jrows=E.load(args.results/'joint_rows.json.gz');normdiff=[];lpcheck=0;failed_solver=[];imax=0.;proof=[]
    for r in rows:
        a,b=E.arr(r['old_state']),E.arr(r['new_state']);n=b['n'];k0,k1=F.rateints(a['w']),F.rateints(b['w'])
        l0,u0=F.cell(k0,'truncate');l1,u1=F.cell(k1,'truncate');P=F.projector(n,D.K);M=P@b['Q']
        eq=linprog(np.zeros(6),A_eq=np.column_stack((M,-np.eye(3))),b_eq=np.zeros(3),bounds=list(zip(l0,u0))+list(zip(l1,u1)),method='highs')
        expected=r['scores']['raw']['lp_slack_cells']<=1e-7
        if bool(eq.success)!=expected:raise AssertionError('Six-variable equality LP disagrees')
        lpcheck+=1
        # Explicit expression differs from the active-set matrix implementation.
        def fun(y):return (float(y@y)-(1-D.K**2)*float(y@n)**2)/D.K**2
        def jac(y):return (2*y-2*(1-D.K**2)*float(y@n)*n)/D.K**2
        opt=minimize(fun,(l1+u1)/2,jac=jac,bounds=list(zip(l1,u1)),method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10,'maxiter':1000})
        if not opt.success:failed_solver.append(dict(uid=r['uid'],tick=r['tick1'],message=str(opt.message)))
        lower=np.sqrt(max(0.,float(opt.fun)));upper=np.sqrt(max(fun(np.array(z)) for z in itertools.product(*zip(l1,u1))))
        exp_lo,exp_hi=r['invariant']['inverse_output_norm_interval_cells'];delta=max(abs(lower-exp_lo),abs(upper-exp_hi));normdiff.append(delta)
        if delta>1e-6:raise AssertionError('Norm interval optimizer mismatch '+str(delta))
        invariance=abs(np.linalg.norm(np.linalg.solve(P,P@b['Q']@a['w']))-np.linalg.norm(a['w']));imax=max(imax,float(invariance))
        if r['family']=='R-77-1' and not expected:
            T=(b['tick']-a['tick'])/48;bar=(a['vm']+b['vm'])/2;dp=b['p']-a['p'];gg=D.geo(a['Ps'],a['Vs'],a['p'],a['vm'])
            proof.append(dict(replay=r['replay'],eid=r['eid'],uid=r['uid'],tick0=r['tick0'],tick1=r['tick1'],old_index=a['index'],new_index=b['index'],old_source_sha=a['raw_sha'],new_source_sha=b['raw_sha'],old_rate_int=k0,new_rate_int=k1,old_rate_deg_s=a['w']*A.RAD,new_rate_deg_s=b['w']*A.RAD,old_timer=a['timer'],new_timer=b['timer'],old_flags=a['flags'] if 'flags' in a else a['flag'],new_flags=b['flags'] if 'flags' in b else b['flag'],old_pointing_deg=a['angles']*A.RAD,new_pointing_deg=b['angles']*A.RAD,old_direction_vs_Ps_deg=A.angle(a['n'],A.unit(a['Ps']-a['p']))*A.RAD,old_rate_vs_geometry_deg_s=float(np.linalg.norm(a['w']-gg))*A.RAD,own_displacement_error_m=float(np.linalg.norm(dp-T*bar)),own_motion_time_ratio=float(dp@bar/(bar@bar)/T),rate_qnorm_gap_cells=r['invariant']['interval_gap_cells'],Q_alone_cannot_repair=not r['invariant']['arbitrary_Q_necessary_feasible_with_n'],raw_rate_error=r['scores']['raw']['rate_deg_s'],half_geometry_rate_error=r['scores']['delta_geometry_half']['rate_deg_s'],half_geometry_cell_feasible=r['scores']['delta_geometry_half']['lp_slack_cells']<=1e-7,linear_rate_min_slack_cells=r['scores']['raw']['lp_slack_cells']))
    cohort=[r for r in rows if r['family']=='R-77-1'];base_bad={(r['replay'],r['eid'],r['tick1']) for r in cohort if r['scores']['raw']['lp_slack_cells']>1e-7};half_bad={(r['replay'],r['eid'],r['tick1']) for r in cohort if r['scores']['delta_geometry_half']['lp_slack_cells']>1e-7}
    jr={ (r['replay'],r['eid'],r['tick1']):r for r in jrows}
    for r in proof:
        j=jr[(r['replay'],r['eid'],r['tick1'])];r['joint_after_feedback_error']=j['scores']['n_is_after_feedback']['rate_deg_s'];r['joint_stage_rate_prediction_difference']=j['rate_difference_between_stages_deg_s']
    report=dict(scope='Same fixed actual cases; alternate numerical formulations, not new independent observations',equality_LP_comparisons=lpcheck,norm_minimum_comparisons=len(normdiff),norm_interval_max_abs_difference_cells=max(normdiff),norm_optimizer_status_warnings=failed_solver,rotation_invariance_max_abs_error_rad_s=imax,baseline_R77_failures=len(base_bad),half_geometry_R77_failures=len(half_bad),repaired_prior_failures=len(base_bad-half_bad),introduced_failures=len(half_bad-base_bad),remaining_failures=len(base_bad&half_bad),case_details=proof)
    A.save(args.out/'independent_review.json',report)
    lines=['# 六个固定反例：接手索引','','下列诊断使用同一批原始导出字段，没有新增目标真值；模长不相容结论限于固定kappa=.8、声明的rate截断单元，以及目标方向允许.0031°的敏感性范围。','', '| Session | UID / EID | INS→TRK tick | 仅换Q能否修复 | 原误差 °/s | 补半区间误差 °/s | 补后单元相容 |','|---|---|---|---|---:|---:|---|']
    for r in proof:lines.append(f"| {r['replay']} | {r['uid']} / {r['eid']} | {r['tick0']}→{r['tick1']} | {'不能' if r['Q_alone_cannot_repair'] else '必要条件未排除'} | {r['raw_rate_error']:.6f} | {r['half_geometry_rate_error']:.6f} | {r['half_geometry_cell_feasible']} |")
    lines+=['','## 读取原始字段','', 'final-results/six_exceptions.json包含完整old_state/new_state；independent_review.json补充原始记录index、源SHA、rate整数、字段timer及自身运动检查。当前数据中旧Ps/Vs的几何rate与保存rate并非每例相容，因此几何增量只能是候选，不能无条件替代原状态。','', '模长区间仅为必要条件。通过该检验不表示实际Q一定有解；未通过才可排除任意Q的纯旋转补救。区间优化不相容不能推广为任何非线性、额外反馈或其他量化编码都失败。']
    (args.out/'ADVERSE_CASES.md').write_text('\n'.join(lines),encoding='utf-8')
    with (args.out/'CASE_INDEX.csv').open('w',newline='',encoding='utf-8') as f:
        cols=['replay','eid','uid','tick0','tick1','old_index','new_index','old_source_sha','new_source_sha','old_rate_vs_geometry_deg_s','own_displacement_error_m','own_motion_time_ratio','Q_alone_cannot_repair','raw_rate_error','half_geometry_rate_error','half_geometry_cell_feasible']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in proof)
    print('INDEPENDENT_REVIEW',json.dumps(A.serial({k:v for k,v in report.items() if k!='case_details'})),flush=True)
    for r in proof:print('CASE_REVIEW',json.dumps(A.serial(r)),flush=True)
if __name__=='__main__':main()
