#!/usr/bin/env python3
"""Prescribed-Q translational test of current game code, not autonomous flight.
Initial p/v are read once at .25s. Later p/v evolve only under current plant
forces. Q(t) is deliberately prescribed from recorded own attitude. No seeker
or control output enters the calculation, and no parameter is estimated.
"""
from __future__ import annotations
import argparse,copy,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import check_energy as E

CHECKPOINTS=[2.,2.5,6.5,7.,7.5,8.5,10.5,12.5,15.]

def run(rt,s,end,substeps):
    start=.25;p,v,q,_=s.at(start);x=np.r_[p,v];t=start
    checkpoints=[a for a in CHECKPOINTS if a<=end];h=1/(48*substeps);out=[];samples=[];stepcount=0
    def rhs(t,x):
        q=s.at(t)[2]
        d,_=rt.plant(t,tuple(x)+tuple(q)+(0.,0.,0.),(0.,0.))
        return np.asarray(d[:6])
    for stop in checkpoints:
        while t<stop-1e-12:
            dt=min(h,stop-t);mid=x+rhs(t,x)*dt/2;x=x+rhs(t+dt/2,mid)*dt;t+=dt;stepcount+=1
            # Primary output cadence48Hz, independent of numerical substeps.
            if stepcount%substeps==0 or abs(t-stop)<1e-11:
                p0,v0,q0,_=s.at(t);f=np.asarray(E.axes(tuple(q0))[0]);sv=np.linalg.norm(x[3:]);rv=np.linalg.norm(v0)
                am=math.degrees(math.atan2(np.linalg.norm(np.cross(f,x[3:])),np.dot(f,x[3:])))
                ar=math.degrees(math.atan2(np.linalg.norm(np.cross(f,v0)),np.dot(f,v0)))
                samples.append(dict(t=t,position_error_m=float(np.linalg.norm(x[:3]-p0)),speed_error_m_s=float(sv-rv),alpha_error_deg=am-ar))
        p0,v0,q0,_=s.at(stop);dp=x[:3]-p0;dv=x[3:]-v0
        out.append(dict(time_s=stop,position_error_m=float(np.linalg.norm(dp)),horizontal_error_m=float(np.hypot(dp[0],dp[2])),height_error_m=float(dp[1]),speed_error_m_s=float(np.linalg.norm(x[3:])-np.linalg.norm(v0)),velocity_vector_error_m_s=float(np.linalg.norm(dv)),model_velocity=x[3:],reference_velocity=v0,model_position=x[:3],reference_position=p0))
    return dict(substeps=substeps,stepcount=stepcount,checkpoints=out,position_rms_m=float(np.sqrt(np.mean([r['position_error_m']**2 for r in samples]))),speed_error_rms_m_s=float(np.sqrt(np.mean([r['speed_error_m_s']**2 for r in samples]))),max_abs_speed_error_m_s=max(abs(r['speed_error_m_s']) for r in samples),max_abs_alpha_error_deg=max(abs(r['alpha_error_deg']) for r in samples),samples=samples)


def summarize_bins(main):
    result={}
    for uid,case in main['cases'].items():
        groups={}
        for name,(lo,hi) in {'pre_capture':(.5,2.),'powered_turn':(2.5,6.5),'coast':(7.5,15.)}.items():
            rows=[r for r in case['half_second_bins'] if r['status']=='ok' and lo<=r['start_s'] and r['end_s']<=hi]
            groups[name]=dict(n=len(rows),sum_net_energy_excess_J_kg=sum(r['energy_excess_J_kg'] for r in rows),sum_absolute_energy_excess_J_kg=sum(abs(r['energy_excess_J_kg']) for r in rows),required_drag_work_J_kg=sum(r['required_drag_work_J_kg'] for r in rows),modeled_drag_work_J_kg=sum(r['predicted_base_drag_work_J_kg']+r['predicted_incidence_drag_work_J_kg'] for r in rows),min_drag_work_ratio=min((r['modeled_to_required_drag_work'] for r in rows if r['modeled_to_required_drag_work'] is not None),default=None),max_drag_work_ratio=max((r['modeled_to_required_drag_work'] for r in rows if r['modeled_to_required_drag_work'] is not None),default=None),max_normal_accel_residual_g=max((r['normal_accel_residual_g'] for r in rows),default=None),max_pos_time_residual_m=max((r['position_versus_integrated_velocity_m'] for r in rows),default=None))
        result[uid]=groups
    return result

def checks(rt):
    count=0
    # Rotation around vertical must rotate forces without changing their power.
    for speed in (400.,800.,1200.):
        for alpha in (0.,math.radians(10),math.radians(30)):
            p=np.array([0.,7000.,0.]);v=np.array([speed,0.,0.]);q=Rotation.from_rotvec([0.,0.,alpha]).as_quat();a=E.scalar_forces(rt,8.,p,v,q)
            rot=Rotation.from_rotvec([0.,.8,0.]);b=E.scalar_forces(rt,8.,rot.apply(p),rot.apply(v),(rot*Rotation.from_quat(q)).as_quat())
            np.testing.assert_allclose(rot.apply(a['a']),b['a'],rtol=0,atol=1e-10)
            assert abs(a['lift_power'])<1e-7 and abs(a['thrust_power'])<1e-10
            assert a['drag0_power']>0 and a['draginc_power']>=0
            if alpha==0:assert a['draginc_power']==0 and np.linalg.norm(a['lift_accel'])==0
            count+=1
    # No claim that the game uses this burn law; guard the actual profile contract.
    assert rt.mass_thrust(0.)==(190.,23835.)
    assert rt.mass_thrust(7.)==(119.,0.)
    assert rt.mass_thrust(20.)==(119.,0.)
    return count+3


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--main',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    histories=E.read(a.input/'anchor_full_histories.json.gz');infos=E.read(a.input/'infos.json');main=E.read(a.main/'summary.json');profile=E.read(E.ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json');defaults=E.read(E.ROOT/'config/effective_surface_defaults.json')
    output=dict(scope='Recorded-Q forced translational integration, initial p/v once at.25s. This deliberately uses future attitude input, not an autonomous flight or regenerated controller. No relock analysis or coefficient fitting.',cases={},half_second_summaries=summarize_bins(main),contract_checks=0)
    for uid,(rep,eid) in E.ANCHORS.items():
        info=next(i for i in infos if (i['replay'],i['eid'])==(rep,eid));s=E.Series(histories[uid],info['epoch_tick']);rt=E.make_runtime(profile,defaults,histories[uid][0]);output['contract_checks']+=checks(rt)
        end=12.5 if uid=='118' else 15.;coarse=run(rt,s,end,4);fine=run(rt,s,end,8)
        conv=max(np.linalg.norm(np.array(a['model_position'])-np.array(b['model_position'])) for a,b in zip(coarse['checkpoints'],fine['checkpoints']))
        sconv=max(np.linalg.norm(np.array(a['model_velocity'])-np.array(b['model_velocity'])) for a,b in zip(coarse['checkpoints'],fine['checkpoints']))
        output['cases'][uid]=dict(primary=coarse,refined=fine,max_checkpoint_position_convergence_m=conv,max_checkpoint_velocity_convergence_m_s=sconv)
        print('PRESCRIBED_Q',uid,json.dumps(E.clean({k:v for k,v in coarse.items() if k!='samples'})),flush=True)
        print('STEP_REFINEMENT',uid,conv,sconv,flush=True)
    print('HALF_SECOND_RESIDUALS',json.dumps(output['half_second_summaries']),flush=True)
    E.save(a.out/'summary.json',output)
    # Tables include small bins so a long interval's net cannot hide sign changes.
    lines=['# Fixed-state force/work and prescribed-Q translation','','These are diagnostic conditions, not autonomous flight accuracy. Positive energy/speed excess means the model retains too much energy.','','## Fixed own-state energy windows','','|Case|Age interval (s)|Mean alpha (deg)|Model/required drag work|Energy excess (J/kg)|Equivalent end-speed excess (m/s)|','|---|---|---:|---:|---:|---:|']
    for uid,c in main['cases'].items():
        for r in c['windows']:
            if r['status']!='ok':continue
            lines.append(f"|{uid}|{r['start_s']}-{r['end_s']}|{r['alpha_mean_deg']:.3f}|{r['modeled_to_required_drag_work']:.6f}|{r['energy_excess_J_kg']:.3f}|{r['equivalent_endpoint_speed_excess_m_s']:.4f}|")
    lines+=['','## Prescribed-Q continuous translation from0.25s','','|Case|Age(s)|Horizontal error(m)|Height error(m)|Speed error(m/s)|','|---|---:|---:|---:|---:|']
    for uid,c in output['cases'].items():
        for r in c['primary']['checkpoints']:lines.append(f"|{uid}|{r['time_s']}|{r['horizontal_error_m']:.4f}|{r['height_error_m']:.4f}|{r['speed_error_m_s']:.4f}|")
    lines+=['','## Every predeclared half-second bin','','|Case|Age(s)|Mean alpha|Drag-work ratio|Energy residual J/kg|Normal impulse residual g|','|---|---|---:|---:|---:|---:|']
    for uid,c in main['cases'].items():
        for r in c['half_second_bins']:
            if r['status']=='ok':lines.append(f"|{uid}|{r['start_s']}-{r['end_s']}|{r['alpha_mean_deg']:.3f}|{r['modeled_to_required_drag_work']:.6f}|{r['energy_excess_J_kg']:.3f}|{r['normal_accel_residual_g']:.4f}|")
    (a.out/'TABLES.md').write_text('\n'.join(lines)+'\n')
    print('CONTRACT_CHECKS',output['contract_checks'],flush=True)
if __name__=='__main__':main()
