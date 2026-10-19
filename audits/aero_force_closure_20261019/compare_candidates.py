#!/usr/bin/env python3
"""Fixed-equation game replay experiment. No coefficient optimizer or feedback."""
from __future__ import annotations
import argparse,copy,itertools,json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'audits/aero_energy_20261019'))
import check_energy as E

DRAG=('total_sine2','total_angle2','total_sine_tangent','planar_sine2','planar_angle2')
LIFT=('isotropic','planar_normalized','planar_projected')
H=1/48

def candidates():
    out=[dict(name=d+'__'+l,drag=d,lift=l,station='cg',q_shift=0.,method='midpoint') for d,l in itertools.product(DRAG,LIFT)]
    out[0]['name']='baseline'
    for name,st in [('local_incidence_drag','local_drag'),('local_all_aero','local_all')]:
        out.append(dict(name=name,drag='total_sine2',lift='isotropic',station=st,q_shift=0.,method='midpoint'))
    for count in (-1.,-.5,.5,1.):
        out.append(dict(name='diagnostic_Q_shift_'+str(count)+'tick',drag='total_sine2',lift='isotropic',station='cg',q_shift=count*H,method='midpoint'))
    for method in ('euler_velocity_trapezoid_position','semiimplicit_euler'):
        out.append(dict(name='diagnostic_'+method,drag='total_sine2',lift='isotropic',station='cg',q_shift=0.,method=method))
    return out

def features(v,f,up,right):
    speed=float(np.linalg.norm(v));t=v/speed if speed>1e-12 else f
    co=float(np.dot(f,t));tr=f-co*t;sn=float(np.linalg.norm(tr));alpha=math.atan2(sn,co)
    ap=math.atan2(-float(t@up),co);ay=math.atan2(-float(t@right),co)
    return speed,t,tr,sn,alpha,ap,ay

def force(rt,t,p,v,q,c,omega=None):
    f,up,right=(np.array(x) for x in E.axes(tuple(q)))
    speed,tangent,tr,sn,alpha,ap,ay=features(v,f,up,right)
    atm=rt.atmosphere.sample(float(p[1]));dynamic=.5*atm.density_kg_m3*speed**2;mach=speed/atm.speed_of_sound_mps
    cd,oldinc,cl=rt.coefficients(mach,alpha)
    va=v
    if c['station']!='cg':
        if omega is None:raise ValueError('Local station requires body angular velocity input')
        va=v+np.cross(omega,-rt.ell*f)
    ls,lt,ltr,lsn,la,lap,lay=features(va,f,up,right)
    da=la if c['station']!='cg' else alpha
    dp=lap if c['station']!='cg' else ap;dy=lay if c['station']!='cg' else ay
    if c['drag']=='total_sine2':b=math.sin(da)**2
    elif c['drag']=='total_angle2':b=da*da
    elif c['drag']=='total_sine_tangent':
        if math.cos(da)<=0:raise ValueError('Sine-tangent candidate left forward flow')
        b=math.sin(da)*math.tan(da)
    elif c['drag']=='planar_sine2':b=math.sin(dp)**2+math.sin(dy)**2
    elif c['drag']=='planar_angle2':b=dp*dp+dy*dy
    else:raise ValueError(c['drag'])
    inc=rt.p['cd_incidence_ratio']*cd*b
    if c['station']=='local_all':
        dynamic=.5*atm.density_kg_m3*ls**2
        tangent,tr,sn,alpha,ap,ay=lt,ltr,lsn,la,lap,lay
        cl=rt.coefficients(ls/atm.speed_of_sound_mps,la)[2]
    if c['lift']=='isotropic':lift=dynamic*rt.area*cl*tr/sn if sn>1e-12 else np.zeros(3)
    else:
        rolloff=math.exp(-(max(alpha-rt.p['lift_stall_rad'],0.)/rt.p['lift_rolloff_width_rad'])**2)
        lift=np.zeros(3)
        for a,axis in [(ap,up),(ay,right)]:
            n=axis-float(axis@tangent)*tangent
            if c['lift']=='planar_normalized':
                size=float(np.linalg.norm(n));n=n/size if size>1e-12 else np.zeros(3)
            lift+=dynamic*rt.area*rt.p['lift_sine_slope']*rolloff*math.sin(a)*n
    drag=-dynamic*rt.area*(cd+inc)*tangent
    mass,thrust=rt.mass_thrust(t)
    a=(lift+drag+thrust*f)/mass+np.array([0.,-E.G,0.])
    return a,dict(alpha_deg=math.degrees(alpha),drag_n=float(np.linalg.norm(drag)),lift_n=float(np.linalg.norm(lift)),lift_power=float(lift@v),cd_total=cd+inc)

class Forcing:
    def __init__(self,series):self.s=series;self.cache={}
    def at(self,t):
        key=round(t,12)
        if key not in self.cache:self.cache[key]=self.s.at(t)
        return self.cache[key]
    def omega(self,t):
        j,a,b,x=self.s.bracket(t);dt=(b['tick']-a['tick'])/48
        wn=Rotation.from_matrix(np.array(b['Q'])@np.array(a['Q']).T).as_rotvec()/dt
        # Angular velocity is axial under the native/runtime reflection.
        return -E.C@wn

def metrics(samples):
    pe=np.array([r['position_error_m'] for r in samples]);se=np.array([r['speed_error_m_s'] for r in samples]);ve=np.array([r['velocity_error_m_s'] for r in samples]);ce=np.array([r['cross_track_m'] for r in samples])
    im=int(np.argmax(pe));iv=int(np.argmax(abs(se)))
    return dict(max_position_error_m=float(pe.max()),max_position_error_time_s=samples[im]['t'],max_abs_speed_error_m_s=float(abs(se).max()),max_speed_error_time_s=samples[iv]['t'],max_velocity_vector_error_m_s=float(ve.max()),max_cross_track_error_m=float(ce.max()),position_rms_m=float(np.sqrt(np.mean(pe**2))),speed_error_rms_m_s=float(np.sqrt(np.mean(se**2))),final=samples[-1],speed_target_pass=bool(abs(se).max()<5),position_target_pass=bool(pe.max()<10),both_targets_pass=bool(abs(se).max()<5 and pe.max()<10))

def run(rt,forcing,end,c,substeps=4):
    start=.25;p,v,q,_=forcing.at(start);x=np.r_[p,v];samples=[];actual_substeps=1 if c['method']!='midpoint' else substeps
    step=H/actual_substeps;N=round((end-start)/step);max_lift_power=0.;forces=0
    def rhs(t,x):
        q=forcing.at(t+c['q_shift'])[2]
        omega=forcing.omega(t) if c['station']!='cg' else None
        a,diag=force(rt,t,x[:3],x[3:],q,c,omega)
        return np.r_[x[3:],a]
    for k in range(N):
        t=start+k*step;d=rhs(t,x)
        if c['method']=='midpoint':x=x+rhs(t+step/2,x+d*step/2)*step
        elif c['method']=='euler_velocity_trapezoid_position':x=np.r_[x[:3]+x[3:]*step+.5*d[3:]*step**2,x[3:]+d[3:]*step]
        elif c['method']=='semiimplicit_euler':
            vv=x[3:]+d[3:]*step;x=np.r_[x[:3]+vv*step,vv]
        else:raise ValueError(c['method'])
        now=start+(k+1)*step;p0,v0,_,_=forcing.at(now);dp=x[:3]-p0;dv=x[3:]-v0;vh=v0/np.linalg.norm(v0);along=float(dp@vh);ct=dp-along*vh
        # Every integration substep is scored; position interpolation is part
        # of the declared reference, not a continuous-time proof between nodes.
        samples.append(dict(t=now,position_error_m=float(np.linalg.norm(dp)),position_error_xyz_m=dp.tolist(),horizontal_error_m=float(np.hypot(dp[0],dp[2])),height_error_m=float(dp[1]),speed_error_m_s=float(np.linalg.norm(x[3:])-np.linalg.norm(v0)),velocity_error_m_s=float(np.linalg.norm(dv)),along_track_m=along,cross_track_m=float(np.linalg.norm(ct)),model_p=x[:3].tolist(),model_v=x[3:].tolist()))
    return dict(candidate=c,substeps=actual_substeps,steps=N,**metrics(samples),samples=samples)

def verify_force(rt,series):
    checks=0;maxdiff=0.;baseline=candidates()[0]
    for t in (.5,1.,2.,3.,5.,6.5,7.5,10.,12.):
        p,v,q,_=series.at(t);a,d=force(rt,t,p,v,q,baseline);v0,diag=rt.plant(t,tuple(p)+tuple(v)+tuple(q)+(0.,0.,0.),(0.,0.));maxdiff=max(maxdiff,float(np.max(np.abs(a-v0[3:6]))));checks+=1
        assert abs(d['lift_power'])<1e-5
        for c in candidates()[:15]:
            aa,dd=force(rt,t,p,v,q,c);assert np.all(np.isfinite(aa)) and dd['drag_n']>=0 and abs(dd['lift_power'])<1e-5;checks+=1
    assert maxdiff<1e-8,maxdiff
    return dict(force_checks=checks,baseline_force_max_difference=maxdiff)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    for r in E.read(a.source/'MANIFEST.json'):
        x=a.source/r['path'];assert x.stat().st_size==r['bytes'] and E.digest(x)==r['sha256']
    histories=E.read(a.source/'input/anchor_full_histories.json.gz');infos=E.read(a.source/'input/infos.json');profile=E.read(ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json');defaults=E.read(ROOT/'config/effective_surface_defaults.json');old=E.read(a.source/'results/attitude/summary.json')
    objects={};verification={}
    for uid,(replay,eid) in E.ANCHORS.items():
        info=next(i for i in infos if i['replay']==replay and i['eid']==eid);s=E.Series(histories[uid],info['epoch_tick']);rt=E.make_runtime(profile,defaults,histories[uid][0]);objects[uid]=(rt,Forcing(s));verification[uid]=verify_force(rt,s)
    train=[];family=candidates()
    for c in family:
        r=run(*objects['118'],12.5,c);E.save(a.out/('train118_'+c['name']+'.json.gz'),r);small={k:v for k,v in r.items() if k!='samples'};train.append(small);print('CANDIDATE118',json.dumps(E.clean(small)),flush=True)
    eligible=[r for r in train if r['candidate']['q_shift']==0 and r['candidate']['method']=='midpoint']
    speedok=[r for r in eligible if r['speed_target_pass']]
    selected=min(speedok or eligible,key=lambda r:r['max_position_error_m'])
    print('FROZEN_SELECTION_FROM_118',json.dumps(selected['candidate']),flush=True)
    tests=[]
    for c in [family[0],selected['candidate']]:
        for sub in (4,8):
            for uid,end in [('118',12.5),('148',15.)]:
                r=run(*objects[uid],end,c,sub);E.save(a.out/f'validation_{uid}_{c["name"]}_sub{sub}.json.gz',r);small={k:v for k,v in r.items() if k!='samples'};small['uid']=uid;tests.append(small);print('VALIDATION',json.dumps(E.clean(small)),flush=True)
    for uid in ('118','148'):
        r=next(t for t in tests if t['uid']==uid and t['candidate']['name']=='baseline' and t['substeps']==4);ref=old['cases'][uid]['primary']['checkpoints'][-1]
        d=abs(r['final']['speed_error_m_s']-ref['speed_error_m_s']);pd=np.linalg.norm(np.array(r['final']['model_p'])-ref['model_position']);assert d<1e-5 and pd<1e-4,(uid,d,pd);verification[uid]['previous_baseline_speed_difference']=d;verification[uid]['previous_baseline_position_difference']=pd
    out=dict(scope='Unfitted fixed-equation selection on #118 then frozen check on #148; recorded prescribed-Q p/v integration, not autonomous flight',source_dev='ab93a013dbe5ed8010e3f925a4bcbc1fa83d6f8d',source_coefficients=defaults['effective_parameters'],targets=dict(max_speed_difference_m_s=5.,max_3d_position_difference_m=10.),train118=train,selected_from_118=selected['candidate'],validation=tests,verification=verification)
    E.save(a.out/'summary.json',out)
    lines=['# Unretuned force-convention comparison','','Selection uses #118 only. All errors are maxima over integration evaluation nodes, not only RMSE. All runs prescribe recorded Q(t); no autonomous flight claim.','','|Candidate|Case|Substeps|Max 3D position m|Max abs speed m/s|Max cross-track m|Final xyz m|','|---|---|---:|---:|---:|---:|---|']
    for r in tests:lines.append('|'+r['candidate']['name']+'|'+r['uid']+'|'+str(r['substeps'])+'|'+f'{r["max_position_error_m"]:.6f}|{r["max_abs_speed_error_m_s"]:.6f}|{r["max_cross_track_error_m"]:.6f}|'+str(r['final']['position_error_xyz_m'])+'|')
    (a.out/'TABLES.md').write_text('\n'.join(lines)+'\n');print('VERIFICATION',json.dumps(E.clean(verification)),flush=True)
if __name__=='__main__':main()
