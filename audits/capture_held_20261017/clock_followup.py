#!/usr/bin/env python3
"""Second-stage diagnostics motivated by boundary results; no parameter fitting.
Timer-continuity stratum was preregistered. Additional return-clock and anchor
propagation candidates are exploratory and reported separately from first-pass tests.
"""
import argparse,gzip,json,math
from pathlib import Path
import numpy as np
import audit as A

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def arrays(r):
    r=r.copy()
    for k in ('n','w','Ps','Vs','unknown','p','vm','Q','angles'):r[k]=np.asarray(r[k],dtype=float)
    r['m']=r['Q']@A.ray(r['angles'])
    return r

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--results',type=Path,required=True);cli.add_argument('--out',type=Path,required=True);a=cli.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    held=read(a.results/'held_pairs.json.gz');returns=read(a.results/'fresh_return_events.json.gz');hist=read(a.results/'anchor_full_histories.json.gz');caps=read(a.results/'capture_events.json.gz')
    eps=2*10/32767
    def hm(rr):
        return dict(n=len(rr),cache_equal=sum(r['cache_equal'] for r in rr),rate_equal=sum(r['rate_equal'] for r in rr),metrics={k:A.metric([r[k] for r in rr]) for k in ('rotate_deg','rate_error_deg_s','R_frozen_error_m','R_CV_error_m')})
    out={'scope':'Timer-defined strata plus explicitly exploratory split-clock and anchor propagation checks','held_timer_groups':{}}
    for name,rr in [('all',held),('R77',[r for r in held if r['family']=='R-77-1']),('flag00',[r for r in held if r['flag0']==r['flag1']==0]),('flag11',[r for r in held if r['flag0']==r['flag1']==1])]:
        out['held_timer_groups'][name]={'continuous_timer':hm([r for r in rr if r['timer_increment_error_s']<eps]),'discontinuous_timer':hm([r for r in rr if r['timer_increment_error_s']>=eps])}
    fr=[]
    for r in returns:
        x,y=arrays(r['old']),arrays(r['new']);T=r['gap']/48;dtobs=T+x['timer'];p=A.rotate(x['n'],x['w'],T)
        preds={}
        for label,alpha in [('alpha_frozen',A.ALPHA),('alpha_one',1.)]:
            n=A.blend(p,y['m'],alpha);c=A.correction(p,n)
            for clock,delta in [('saved_dt',T),('measurement_dt',dtobs)]:
                w=x['w']+c/(A.TAU+delta/2);preds[label+'/'+clock]=A.errors(n,w,y)
        preds['snap_angle_keep_rate']=A.errors(y['m'],x['w'],y)
        fr.append({k:r[k] for k in ('replay','eid','uid','family','age0','age1','gap','flag0','flag1','limited')}|{'timer0':x['timer'],'predictions':preds})
    def fm(rr):
        names=list(rr[0]['predictions']) if rr else []
        return {name:{k:A.metric([r['predictions'][name][k] for r in rr]) for k in ('n_deg','w_deg_s','lambda_error_deg_s','P_m')} for name in names}
    out['fresh_return_split_clock']={name:fm(rr) for name,rr in [('all',fr),('R77_unlimited',[r for r in fr if r['family']=='R-77-1' and not r['limited']]),('R77_unlimited_flag00',[r for r in fr if r['family']=='R-77-1' and not r['limited'] and r['flag0']==r['flag1']==0]),('R77_unlimited_flag11',[r for r in fr if r['family']=='R-77-1' and not r['limited'] and r['flag0']==r['flag1']==1]),('R77_unlimited_flagchange',[r for r in fr if r['family']=='R-77-1' and not r['limited'] and r['flag0']!=r['flag1']])]}
    epout=[];dts=[];boundaries=[]
    for uid,hh in hist.items():
        rr=[arrays(r) for r in hh];start=None;errs={};preindex=None
        epoch=next(r['tick0']-round(r['age0']*48) for r in caps if r['uid']==int(uid) and (r['replay'],r['eid']) in A.ANCHORS)
        def finish():
            nonlocal start,errs
            if start is not None and errs:
                epout.append(dict(uid=int(uid),flag=start['flag'],start_age_s=(start['tick']-epoch)/48,end_age_s=(preindex-epoch)/48,steps=len(next(iter(errs.values()))),metrics={k:A.metric(v) for k,v in errs.items()}))
            start=None;errs={}
        for x,y in zip(rr,rr[1:]):
            gap=y['tick']-x['tick'];T=gap/48
            coherent=0<gap<=8 and x['code']==y['code']==31 and y['timer']>0 and x['flag']==y['flag'] and abs(y['timer']-x['timer']-T)<eps
            if x['code']!=y['code'] or x['flag']!=y['flag'] or ((x['timer']==0)!=(y['timer']==0)):
                if 0<gap<=8:boundaries.append(dict(uid=int(uid),age=(y['tick']-epoch)/48,code=[x['code'],y['code']],flag=[x['flag'],y['flag']],timer=[x['timer'],y['timer']],angles_deg=y['angles']*A.RAD,n_body_deg=A.angles(y['Q'].T@y['n'])*A.RAD,w_old=x['w']*A.RAD,w_new=y['w']*A.RAD))
            if not coherent:
                finish();continue
            if start is None:start=x
            st=start;dt=(y['tick']-st['tick'])/48;n0=st['n'];w0=st['w'];u0=st['vm']+(st['d']-st['vm']@n0)*n0+st['R']*np.cross(n0,w0)
            predictions={
                'full_rate_exponential':A.rotate(n0,w0,dt),
                'transverse_rate_exponential':A.rotate(n0,w0-(w0@n0)*n0,dt),
                'linear_tangent_from_start':A.unit(n0+dt*np.cross(n0,w0)),
                'cached_Ps_Vs_CV':A.unit(st['Ps']+st['Vs']*(st['timer']+dt)-y['p']),
                'angular_state_CV':A.unit(st['p']+st['R']*n0+u0*dt-y['p']),
                'cached_Ps_no_velocity':A.unit(st['Ps']-y['p'])}
            for name,n in predictions.items():errs.setdefault(name,[]).append(A.angle(n,y['n'])*A.RAD)
            errs.setdefault('complete_rate_change_deg_s',[]).append(float(np.linalg.norm(y['w']-w0))*A.RAD)
            preindex=y['tick']
            mag=np.linalg.norm(x['w'])
            if mag*T>.001:
                axis=x['w']/mag;v0=x['n']-(x['n']@axis)*axis;v1=y['n']-(y['n']@axis)*axis
                if np.linalg.norm(v0)>.01 and np.linalg.norm(v1)>.01:
                    tf=-math.atan2(axis@np.cross(v0,v1),v0@v1)/mag
                    dts.append(dict(uid=int(uid),flag=y['flag'],age=(y['tick']-epoch)/48,ratio=tf/T,axial_direction_change=float(y['n']@axis-x['n']@axis)))
        finish()
    out['anchor_held_episodes']=epout
    out['anchor_effective_time_ratios']={f'{uid}_flag{flag}':{k:A.metric([r[k] for r in dts if r['uid']==uid and r['flag']==flag]) for k in ('ratio','axial_direction_change')} for uid in (118,148) for flag in (0,1)}
    out['anchor_boundaries']=boundaries
    A.save(a.out/'followup.json',out);A.save(a.out/'fresh_return_predictions.json.gz',fr);A.save(a.out/'anchor_time_diagnostics.json.gz',dts)
    for k,v in out.items():print('FOLLOWUP',k,json.dumps(A.serial(v),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
