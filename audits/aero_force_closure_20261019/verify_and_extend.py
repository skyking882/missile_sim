#!/usr/bin/env python3
"""Independent grid check plus fixed discrete update hypotheses for a game."""
from __future__ import annotations
import argparse,copy,json,math
from pathlib import Path
import numpy as np
import compare_candidates as B
E=B.E


def discrete(rt,forcing,end,kind,pose_fraction,substeps=1):
    start=.25;p,v,q,_=forcing.at(start);x=np.r_[p,v];h=B.H/substeps;N=round((end-start)/h);samples=[];maxiter=0;maxeq=0.
    baseline=B.candidates()[0]
    for k in range(N):
        t=start+k*h;tm=t+h/2;old=x.copy();q=forcing.at(t+pose_fraction*h)[2];f=np.array(E.axes(tuple(q))[0]);mass,thrust=rt.mass_thrust(tm);external=thrust*f/mass+np.array([0.,-E.G,0.])
        def acc(p,v):return B.force(rt,tm,p,v,q,baseline)[0]
        if kind=='implicit_velocity':
            nv=old[3:]+h*acc(old[:3],old[3:])
            for it in range(80):
                pp=old[:3]+h*(old[3:]+nv)/2
                nxt=old[3:]+h*acc(pp,nv)
                if np.linalg.norm(nxt-nv)<1e-10:
                    nv=nxt;break
                nv=nxt
            else:raise ValueError('Implicit fixed-point solve did not converge')
            maxiter=max(maxiter,it+1);pp=old[:3]+h*(old[3:]+nv)/2
            eq=np.linalg.norm(nv-old[3:]-h*acc(pp,nv));maxeq=max(maxeq,float(eq));x=np.r_[pp,nv]
        elif kind=='thrust_then_aero':
            vv=old[3:]+h*external;aa=acc(old[:3]+h*old[3:]/2,vv)-external;nv=vv+h*aa;x=np.r_[old[:3]+h*(old[3:]+nv)/2,nv]
        elif kind=='aero_then_thrust':
            aa=acc(old[:3]+h*old[3:]/2,old[3:])-external;nv=old[3:]+h*aa+h*external;x=np.r_[old[:3]+h*(old[3:]+nv)/2,nv]
        else:raise ValueError(kind)
        now=start+(k+1)*h;p0,v0,_,_=forcing.at(now);dp=x[:3]-p0;dv=x[3:]-v0;vh=v0/np.linalg.norm(v0);along=float(dp@vh)
        samples.append(dict(t=now,position_error_m=float(np.linalg.norm(dp)),position_error_xyz_m=dp.tolist(),horizontal_error_m=float(np.hypot(dp[0],dp[2])),height_error_m=float(dp[1]),speed_error_m_s=float(np.linalg.norm(x[3:])-np.linalg.norm(v0)),velocity_error_m_s=float(np.linalg.norm(dv)),along_track_m=along,cross_track_m=float(np.linalg.norm(dp-along*vh)),model_p=x[:3].tolist(),model_v=x[3:].tolist()))
    return dict(kind=kind,pose_fraction=pose_fraction,substeps=substeps,steps=N,maximum_iteration_count=maxiter,maximum_implicit_equation_residual_m_s=maxeq,**B.metrics(samples),samples=samples)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--first',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    for r in E.read(a.source/'MANIFEST.json'):
        z=a.source/r['path'];assert z.stat().st_size==r['bytes'] and E.digest(z)==r['sha256']
    histories=E.read(a.source/'input/anchor_full_histories.json.gz');infos=E.read(a.source/'input/infos.json');profile=E.read(B.ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json');defaults=E.read(B.ROOT/'config/effective_surface_defaults.json');old=E.read(a.source/'results/attitude/summary.json')
    objects={};grids={};force=B.force
    for uid,(rep,eid) in E.ANCHORS.items():
        info=next(i for i in infos if i['replay']==rep and i['eid']==eid);s=E.Series(histories[uid],info['epoch_tick']);rt=E.make_runtime(profile,defaults,histories[uid][0]);forcing=B.Forcing(s);objects[uid]=(rt,forcing);end=12.5 if uid=='118' else 15.
        saved=E.read(a.first/f'validation_{uid}_baseline_sub4.json.gz')
        def native(rt,t,p,v,q,c,omega=None):
            d,z=rt.plant(t,tuple(p)+tuple(v)+tuple(q)+(0.,0.,0.),(0.,0.));return np.array(d[3:6]),z
        B.force=native
        try:direct=B.run(rt,forcing,end,B.candidates()[0])
        finally:B.force=force
        diff=max(np.linalg.norm(np.array(x['model_p'])-y['model_p']) for x,y in zip(direct['samples'],saved['samples']));dv=max(np.linalg.norm(np.array(x['model_v'])-y['model_v']) for x,y in zip(direct['samples'],saved['samples']));assert diff<1e-7 and dv<1e-7,(uid,diff,dv)
        prior=old['cases'][uid]['primary']['checkpoints'][-1];dp=float(np.linalg.norm(np.array(direct['final']['model_p'])-prior['model_position']));ds=abs(direct['final']['speed_error_m_s']-prior['speed_error_m_s']);pos_bound=old['cases'][uid]['max_checkpoint_position_convergence_m'];vel_bound=old['cases'][uid]['max_checkpoint_velocity_convergence_m_s'];assert dp<pos_bound and ds<vel_bound
        grids[uid]=dict(native_vs_reconstructed_same_grid_position_m=diff,native_vs_reconstructed_same_grid_velocity_m_s=dv,previous_checkpoint_grid_position_m=dp,previous_checkpoint_grid_speed_m_s=ds,previous4to8_position_change_m=pos_bound,previous4to8_velocity_change_m_s=vel_bound)
    first=[]
    for c in B.candidates():
        r=E.read(a.first/('train118_'+c['name']+'.json.gz'));first.append({k:v for k,v in r.items() if k!='samples'})
    fixed=[]
    for r in first:
        if r['candidate']['q_shift']==0 and r['candidate']['method']=='midpoint':fixed.append(r)
    eligible=[r for r in fixed if r['speed_target_pass']];first_select=min(eligible or fixed,key=lambda r:r['max_position_error_m'])['candidate'];firstheld=E.read(a.first/f'validation_148_{first_select["name"]}_sub8.json.gz')
    train=[]
    for kind in ('implicit_velocity','thrust_then_aero','aero_then_thrust'):
        for frac in (0.,.5,1.):
            r=discrete(*objects['118'],12.5,kind,frac);E.save(a.out/f'train118_{kind}_Q{frac:g}.json.gz',r);sm={k:v for k,v in r.items() if k!='samples'};train.append(sm);print('DISCRETE118',json.dumps(E.clean(sm)),flush=True)
    good=[r for r in train if r['speed_target_pass']];choice=min(good or train,key=lambda r:r['max_position_error_m']);print('FROZEN_DISCRETE_SELECTION',choice['kind'],choice['pose_fraction'],flush=True)
    tests=[]
    for uid,end in [('118',12.5),('148',15.)]:
        for sub in (1,2):
            r=discrete(*objects[uid],end,choice['kind'],choice['pose_fraction'],sub);E.save(a.out/f'validation_{uid}_sub{sub}.json.gz',r);sm={k:v for k,v in r.items() if k!='samples'};sm['uid']=uid;tests.append(sm);print('DISCRETE_VALIDATION',json.dumps(E.clean(sm)),flush=True)
    out=dict(scope='Nine unretuned48Hz discrete-map candidates selected on#118 then checked on#148. Not a causal prescribed-Q production patch or an autonomous flight. No coefficient calibration.',grid_verification=grids,first_pass=dict(candidates=first,selected_from118=first_select,held_out148={k:v for k,v in firstheld.items() if k!='samples'}),second_pass=dict(train118=train,selected=dict(kind=choice['kind'],pose_fraction=choice['pose_fraction']),validation=tests))
    E.save(a.out/'summary.json',out);print('GRID_VERIFICATION',json.dumps(grids),flush=True)
    lines=['# Fixed mechanism attempts — strict error targets','','First pass selects force convention from#118 only; second pass selects an operator from#118 only. No numeric coefficients fitted. Results use externally prescribed game attitude and continuous translation from one initial state.','','## Nine operator-order hypotheses','#118 only; all coefficients unchanged.','','|Operator|Q fraction|Max3D m|Max speed m/s|','|---|---:|---:|---:|']
    for r in train:lines.append(f'|{r["kind"]}|{r["pose_fraction"]}|{r["max_position_error_m"]:.6f}|{r["max_abs_speed_error_m_s"]:.6f}|')
    lines+=['','## Frozen validation','','|Case|Substeps|Max3D m|Max speed m/s|Max cross-track m|Final xyz m|','|---|---:|---:|---:|---:|---|']
    for r in tests:lines.append(f'|{r["uid"]}|{r["substeps"]}|{r["max_position_error_m"]:.6f}|{r["max_abs_speed_error_m_s"]:.6f}|{r["max_cross_track_error_m"]:.6f}|{r["final"]["position_error_xyz_m"]}|')
    (a.out/'TABLES.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
