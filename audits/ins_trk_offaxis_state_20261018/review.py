#!/usr/bin/env python3
"""Alternate arithmetic review of frozen game replay state diagnostics."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import compare_states as B
A,C=B.A,B.C

def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))
def norm(x):return math.sqrt(dot(x,x))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def sub(a,b):return [float(x)-float(y) for x,y in zip(a,b)]
def angle(a,b):return math.atan2(norm(cross(a,b)),dot(a,b))*B.DEG
def rotate(n,w,t):
    mag=norm(w)
    if mag<1e-14:return list(n)
    u=[x/mag for x in w];co=math.cos(mag*t);si=math.sin(mag*t);v=cross(n,u);p=dot(u,n)
    return [n[i]*co+v[i]*si+u[i]*p*(1-co) for i in range(3)]
def correction(pred,n):
    c=cross(n,pred);q=norm(c);theta=math.atan2(q,dot(pred,n))
    return [x*theta/q for x in c] if q>1e-14 else c

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--windows',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    events=C.read(a.input/'input/prior/evidence/data/main/evidence/capture_events.json.gz');results=C.read(a.results/'events.json.gz');by={B.ident(e):e for e in events};ws=C.read(a.windows/'windows.json.gz');summary=C.read(a.results/'summary.json');wsummary=C.read(a.windows/'summary.json')
    diffs={'offaxis_deg':0.,'rate_error_deg_s':0.,'axial_error_deg_s':0.,'ordinary_relation_deg_s':0.};paircount=0
    for r in results:
        e=by[B.ident(r)];old,new=e['old'],e['new'];ray=sub(old['Ps'],old['p']);forward=[old['Q'][i][0] for i in range(3)];eta=angle(ray,forward)
        diffs['offaxis_deg']=max(diffs['offaxis_deg'],abs(eta-r['offaxis_deg']))
        s=[dot(row,old['w']) for row in new['Q']];par=dot(s,new['n']);v=[.8*s[i]+.2*par*new['n'][i] for i in range(3)];err=sub(v,new['w'])
        diffs['rate_error_deg_s']=max(diffs['rate_error_deg_s'],abs(norm(err)*B.DEG-r['baseline_error_deg_s']))
        diffs['axial_error_deg_s']=max(diffs['axial_error_deg_s'],abs(dot(err,new['n'])*B.DEG-r['baseline_axial_deg_s']))
    unique={}
    for w in ws:
        states=w['states']
        for i,s in enumerate(states):
            if s.get('unsupported'):continue
            key=(w['identity'][0],w['identity'][1],s['index'])
            if key in unique:
                for field in ('tick','raw_sha','n','w','Ps','Vs'):assert unique[key][field]==s[field]
            else:unique[key]=s
            if 'preceding_pair' not in s or i==0 or states[i-1].get('unsupported'):continue
            prev=states[i-1];pair=s['preceding_pair'];T=(s['tick']-prev['tick'])/48
            pred=rotate(prev['n'],prev['w'],T);c=correction(pred,s['n']);dw=sub(s['w'],prev['w']);ee=[dw[j]-c[j]/(B.TAU+T/2) for j in range(3)]
            calc=norm(ee)*B.DEG;diffs['ordinary_relation_deg_s']=max(diffs['ordinary_relation_deg_s'],abs(calc-pair['ordinary_relation_residual_deg_s']));paircount+=1
    assert max(diffs.values())<1e-9,diffs
    counts={'events':len(results),'native_windows':len(ws),'unique_native_states_in_windows':len(unique),'ordinary_relation_pair_checks_including_overlaps':paircount}
    rawgroup=B.__name__
    # Recalculate window summaries from the saved raw-window states; preserve exact inclusion criteria.
    import raw_windows as W
    assert W.describe(ws)==wsummary['groups']
    for name,gg in summary['groups'].items():
        for group,s in gg.items():
            if s['n']:
                assert s['feasible']<=s['n']
    rr=[r for r in results if r['family']=='R-77-1']
    # Quantify tail leverage without dropping any observations in primary results.
    tails={}
    for label,sel in [('R77_all_large',[r for r in rr if r['angle_group']=='large']),('R77_late_large',[r for r in rr if r['angle_group']=='large' and r['fresh00'] and r['candidate_reacq_lag']==1])]:
        ordered=sorted(sel,key=lambda r:r['baseline_error_deg_s'],reverse=True);total=math.fsum(r['baseline_error_deg_s']**2 for r in ordered)
        tails[label]=dict(n=len(sel),top_four_squared_error_fraction=math.fsum(r['baseline_error_deg_s']**2 for r in ordered[:4])/total if total else None,top_four=[{k:r[k] for k in ('replay','eid','uid','tick0','tick1','offaxis_deg','baseline_error_deg_s','baseline_axial_deg_s','baseline_transverse_deg_s','prior_vs_new_direction_deg','old_geo_error_deg_s','candidate_reacq_lag')} for r in ordered[:4]])
    notes={'scope':'Independent scalar arithmetic and Rodrigues rotation, same data not independent new trials; no parameter estimation','counts':counts,'maximum_absolute_differences':diffs,'window_summary_recomputed_exactly':True,'tail_diagnostics':tails,'parameters':summary['parameters']}
    C.save(a.out/'verification.json',notes)
    lines=['# Verified small/large off-axis comparison','','Units: full-vector rate errors in deg/s. Group by last INS target-estimate ray: small <=20deg, large >=50deg. Current n is a condition in the baseline, not an independently predicted measurement.','','| Group | Angle | Count | RMS rate | Median rate | RMS axial | RMS transverse | Cell compatible |','|---|---|---:|---:|---:|---:|---:|---:|']
    for name in ('R77_all','R77_first','R77_near','R77_reacq_fresh00_lag0','R77_reacq_fresh00_lag1'):
        for g in ('small','large'):
            d=summary['groups'][name][g];m=d['metrics']
            if d['n']:lines.append(f"| {name} | {g} | {d['n']} | {m['baseline_error_deg_s']['rms']:.9f} | {m['baseline_error_deg_s']['median']:.9f} | {m['baseline_axial_deg_s']['rms']:.9f} | {m['baseline_transverse_deg_s']['rms']:.9f} | {d['feasible']}/{d['n']} |")
    lines+=['','## Direct window checks','','| Group | Angle | Eligible next fresh pair | RMS next rate increment | RMS ordinary-relation residual |','|---|---|---:|---:|---:|']
    for name in ('R77_all','R77_reacq_lag0','R77_reacq_lag1'):
        for g in ('small','large'):
            d=wsummary['groups'][name][g]
            if d['next_first_eligible']:lines.append(f"| {name} | {g} | {d['next_first_eligible']} | {d['next_first_rate_increment']['rms']:.9f} | {d['next_first_ordinary_residual']['rms']:.9f} |")
    lines+=['','## Exact anchor source decomposition','','| UID / tick | Prior off-axis | INS n vs Ps ray | Propagated old Ps vs TRK n | Saved n temporal jump | Fixed baseline rate error |','|---|---:|---:|---:|---:|---:|']
    for r in summary['anchor_events'][:3]:lines.append(f"| {r['uid']} / {r['tick1']} | {r['offaxis_deg']:.9f} | {r['n0_vs_Ps_deg']:.9f} | {r['prior_vs_new_direction_deg']:.9f} | {r['n_temporal_jump_deg']:.9f} | {r['baseline_error_deg_s']:.9f} |")
    (a.out/'TABLES.md').write_text('\n'.join(lines)+'\n')
    print('REVIEW',json.dumps(notes),flush=True)
    print('\n'.join(lines),flush=True)
    for name,ss in summary['by_scene'].items():
        print('SCENE_DIRECTIONAL_CHECK',name,json.dumps({g:dict(n=v['n'],baseline=v['metrics']['baseline_error_deg_s']) for g,v in ss.items()}),flush=True)
if __name__=='__main__':main()
