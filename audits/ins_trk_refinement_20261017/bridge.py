#!/usr/bin/env python3
"""Cross-mode conditional consistency; ordinary fits are frozen, capture rates are labels."""
import argparse,json,math
from pathlib import Path
import numpy as np
import refine as F
import followup_review as V
E,A=F.E,F.A

def transfer(n,wold,Q,alpha):
    s=Q@wold;g=1/(F.TAU+F.H/2);kappa=1-alpha*F.H*g
    wlinear=F.projector(n,kappa)@s
    p=A.rotate(n,s,F.H);nn=A.blend(p,n,alpha);wexact=s+A.correction(p,nn)*g
    return kappa,wlinear,nn,wexact

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--lag',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    caps=E.load(a.input/'data/main/evidence/capture_events.json.gz');lag=E.load(a.lag);fold={f['test_scene']:f['models'] for f in lag['folds']}
    selected=[e for e in caps if e['tick1']==F.phase_after(e['tick0']) and e['new']['timer']==0 and e['old']['flag']==e['new']['flag']==0];assert len(selected)==244
    rows=[];algebra_max=0;gh=1/(F.TAU+F.H/2);constants=[]
    for scene,mods in fold.items():
        c={name:x['value'] for name,x in mods.items()}
        alph={'constant_alpha':c['constant_alpha'],'fractional_fixed_lag':1-c['fractional_fixed_lag']/F.H,'rational_lag':F.H/(F.H+c['rational_lag']),'implicit_velocity':c['implicit_velocity']/(1-(1-c['implicit_velocity'])*F.H*gh),'fixed_0p8':.2/(F.H*gh)}
        constants.append(dict(held_scene=scene,alpha_h=alph,kappa_from_ordinary={name:1-v*F.H*gh for name,v in alph.items()}))
    table={r['held_scene']:r for r in constants}
    for e in selected:
        old,new=E.arr(e['old']),E.arr(e['new']);n,Q=new['n'],new['Q'];k0,k1=F.rateints(old['w']),F.rateints(new['w']);scores={}
        for model,alpha in table[e['replay']]['alpha_h'].items():
            k,wl,nn,we=transfer(n,old['w'],Q,alpha);M=F.projector(n,k)@Q
            equivalent=Q@F.projector(Q.T@n,k);diff=float(np.max(np.abs(M-equivalent)));algebra_max=max(algebra_max,diff);assert diff<1e-12
            scores[model]=dict(kappa=k,linear_rate_error_deg_s=float(np.linalg.norm(wl-new['w']))*A.RAD,exact_rate_error_deg_s=float(np.linalg.norm(we-new['w']))*A.RAD,exact_vs_linear_deg_s=float(np.linalg.norm(we-wl))*A.RAD,reset_ray_to_predicted_direction_deg=A.angle(n,nn)*A.RAD,linear_truncation_cells_feasible=V.equal_lp(M,k0,k1,'truncate'))
        rows.append(F.ident(e)|dict(scores=scores))
    groups={}
    keys=('linear_rate_error_deg_s','exact_rate_error_deg_s','exact_vs_linear_deg_s','reset_ray_to_predicted_direction_deg','kappa')
    for group,rr in [('all244',rows),('R77_79',[r for r in rows if r['family']=='R-77-1'])]:
        groups[group]={model:{**{key:A.metric([r['scores'][model][key] for r in rr]) for key in keys},'linear_cell_feasible':sum(r['scores'][model]['linear_truncation_cells_feasible'] for r in rr),'n':len(rr)} for model in rows[0]['scores']}
    result=dict(scope='Current target direction is a conditional reset/projection input; coefficients come from ordinary rows in other scenes. No capture-rate fitting. Tau previously fitted, not newly blinded data.',constants=constants,groups=groups,covariant_matrix_identity_max_abs_error=algebra_max,anchors=[r for r in rows if (r['replay'],r['eid']) in A.ANCHORS])
    A.save(a.out/'bridge_summary.json',result);A.save(a.out/'bridge_rows.json.gz',rows)
    lines=['# Cross-mode numerical receipt','',result['scope'],'','## Ordinary-fit coefficient transferred to capture','','| Held scene | alpha(h), fixed-lag family | Predicted kappa |','|---|---:|---:|']
    for r in constants:lines.append(f"| {r['held_scene']} | {r['alpha_h']['fractional_fixed_lag']:.10f} | {r['kappa_from_ordinary']['fractional_fixed_lag']:.10f} |")
    lines+=['','## Same 79 R77 conditional capture cases','','| Model | Linear complete-rate RMS (deg/s) | Exact feedback complete-rate RMS (deg/s) | Linear truncation-cell feasible |','|---|---:|---:|---:|']
    for name,r in groups['R77_79'].items():lines.append(f"| {name} | {r['linear_rate_error_deg_s']['rms']:.10f} | {r['exact_rate_error_deg_s']['rms']:.10f} | {r['linear_cell_feasible']}/79 |")
    lines+=['','The current n is not independently predicted. Cell checks are for the linear map only at fixed Q/n. The exact spherical step is evaluated at decoded rates; its difference from the linear map is reported in JSON. P(n)Q = QP(Q^T n), so the tested order/axis alternatives cannot uniquely determine internal call order.','']
    (a.out/'CROSS_MODE_RESULT.md').write_text('\n'.join(lines))
    print('BRIDGE_CONSTANTS',json.dumps(A.serial(constants)),flush=True)
    print('BRIDGE_SCORES',json.dumps(A.serial(groups)),flush=True)
    print('BRIDGE_ALGEBRA',algebra_max,flush=True)

if __name__=='__main__':main()
