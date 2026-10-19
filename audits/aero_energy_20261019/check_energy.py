#!/usr/bin/env python3
"""Read-only fixed-state audit of a commercial game's modeled dynamics.
The source own trajectory is a conditioning input, not a forward prediction.
No seeker answers, PN, target association or control tuning enter the scores.
"""
from __future__ import annotations
import argparse,bisect,copy,gzip,hashlib,json,math,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from aim120_model.surface_runtime import SurfaceRuntime
from aim120_model.effective_surface_core import axes,G
from aim120_model.target import TargetState

C=np.diag([1.,1.,-1.])
WINDOWS=[(.25,2.),(2.5,6.5),(7.5,8.5),(8.5,10.5),(10.5,12.5),(12.5,15.)]
ANCHORS={'118':('2026.10.08_11.49.38',2439170),'148':('2026.10.08_12.11.04',1612803)}

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(p,'wt') if p.suffix=='.gz' else p.open('w')) as f:json.dump(clean(d),f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def make_runtime(profile,defaults,initial):
    q=Rotation.from_matrix(C@np.asarray(initial['Q'])@C).as_quat()
    vel=C@np.asarray(initial['vm']);pos=C@np.asarray(initial['p'])
    scenario=dict(loft_enabled=False,launch_pitch_deg=0.,launch_heading_deg=0.,launch_altitude_m=float(pos[1]),launch_speed_kmh=float(np.linalg.norm(vel))*3.6,simulation_dt_s=1/48,observation_mode='ideal_truth')
    target=SimpleNamespace(state_at=lambda t:TargetState((0.,10000.,0.),(0.,0.,0.)))
    return SurfaceRuntime(profile,scenario,{},defaults,target=target,launch_position_m=tuple(pos),launch_velocity_mps=tuple(vel),launch_quaternion_xyzw=tuple(q))

def scalar_forces(rt,t,pos,v,q):
    """Alternate scalar implementation; exact same declared parameter law."""
    basis=axes(tuple(q));f=np.asarray(basis[0]);speed=math.sqrt(math.fsum(float(x*x) for x in v));vh=v/speed
    co=float(np.dot(f,vh));s=math.sqrt(math.fsum(float(x*x) for x in np.cross(f,vh)));alpha=math.atan2(s,co)
    atm=rt.atmosphere.sample(float(pos[1]));dynamic=.5*atm.density_kg_m3*speed**2;mach=speed/atm.speed_of_sound_mps
    c0,c1,c2=rt.p['cd_base_inverse_mach'];me=max(mach,-2*c2/c1);cd=c0+c1/me+c2/me**2
    cdinc=rt.p['cd_incidence_ratio']*cd*math.sin(alpha)**2
    stall=max(alpha-rt.p['lift_stall_rad'],0.)/rt.p['lift_rolloff_width_rad'];cl=rt.p['lift_sine_slope']*math.sin(alpha)*math.exp(-stall*stall)
    mass=float(rt.raw['mass']);thrust=0.;clock=0.
    for st in rt.stages:
        start=clock+float(st.get('fire_delay_s') or 0.);end=start+st['duration_s']
        mass-=st['mass_lost_kg']*min(1.,max(0.,(t-start)/st['duration_s']))
        if start<=t<end:thrust+=st['thrust_n']
        clock=end
    normal=f-co*vh;ln=float(np.linalg.norm(normal));lift=normal*(dynamic*rt.area*cl/ln) if ln>1e-12 else np.zeros(3)
    drag=-dynamic*rt.area*(cd+cdinc)*vh
    a=(thrust*f+lift+drag)/mass+np.array([0.,-G,0.])
    return dict(a=a,thrust_accel=thrust*f/mass,lift_accel=lift/mass,drag_accel=drag/mass,mass=mass,thrust=thrust,speed=speed,mach=mach,alpha=alpha,cd=cd,cdinc=cdinc,cl=cl,dynamic=dynamic,
                thrust_power=float(thrust*np.dot(f,v)/mass),drag0_power=dynamic*rt.area*cd*speed/mass,draginc_power=dynamic*rt.area*cdinc*speed/mass,lift_power=float(np.dot(lift,v)/mass))

class Series:
    def __init__(self,rows,epoch):
        self.rows=rows;self.epoch=epoch;self.ts=np.array([(r['tick']-epoch)/48 for r in rows]);self.breaks=[]
        for i,(a,b) in enumerate(zip(rows,rows[1:])):
            gap=b['tick']-a['tick']
            if not 0<gap<=8:self.breaks.append(dict(index=i,t0=self.ts[i],t1=self.ts[i+1],raw_index0=a['index'],raw_index1=b['index'],gap=gap))
        # Duplicates beyond the fixed15s analysis window must not spoil earlier
        # interpolation; any issue inside a requested window remains a failure.
        self.relevant=[b for b in self.breaks if min(b['t0'],b['t1'])<=15]
    def bracket(self,t):
        if t<self.ts[0]-1e-10 or t>self.ts[-1]+1e-10:raise ValueError('Unbracketed own time')
        j=bisect.bisect_right(self.ts,t)-1;j=max(0,min(j,len(self.rows)-2));a,b=self.rows[j:j+2]
        dt=self.ts[j+1]-self.ts[j]
        if not 0<dt<=8/48+1e-12:raise ValueError('Broken own time interval')
        x=(t-self.ts[j])/dt;return j,a,b,x
    def at(self,t):
        j,a,b,x=self.bracket(t);p=(1-x)*np.asarray(a['p'])+x*np.asarray(b['p']);v=(1-x)*np.asarray(a['vm'])+x*np.asarray(b['vm'])
        qa=np.asarray(a['Q']);r=Rotation.from_matrix(np.asarray(b['Q'])@qa.T).as_rotvec();q=Rotation.from_rotvec(x*r).as_matrix()@qa
        return C@p,C@v,Rotation.from_matrix(C@q@C).as_quat(),((1-x)*a['packet_ms']+x*b['packet_ms'])/1000
    def valid(self,lo,hi):
        if lo<self.ts[0] or hi>self.ts[-1]:return False
        return not any(min(b['t0'],b['t1'])<=hi and max(b['t0'],b['t1'])>=lo for b in self.breaks)


def evaluate(rt,series,t,controls):
    p,v,q,packet=series.at(t);state=tuple(p)+tuple(v)+tuple(q)+(0.,0.,0.)
    deriv,diag=rt.plant(t,state,(0.,0.));z=scalar_forces(rt,t,p,v,q)
    controls['force_checks']+=1;controls['max_acceleration_difference']=max(controls['max_acceleration_difference'],float(np.max(np.abs(np.array(deriv[3:6])-z['a']))))
    power=float(np.dot(v,np.array(deriv[3:6]))+G*v[1]);expected=z['thrust_power']-z['drag0_power']-z['draginc_power']
    controls['max_power_identity_difference']=max(controls['max_power_identity_difference'],abs(power-expected));controls['max_lift_power_abs']=max(controls['max_lift_power_abs'],abs(z['lift_power']))
    # Order: acceleration, thrust impulse, modeled aero impulse, velocity,
    # thrust work, base drag work, incidence drag work, angle, speed, Mach.
    values=np.r_[z['a'],z['thrust_accel'],z['lift_accel']+z['drag_accel'],v,z['thrust_power'],z['drag0_power'],z['draginc_power'],z['alpha'],z['speed'],z['mach']]
    return values,z


def integrate(rt,series,lo,hi,nodes,clock,controls):
    cuts=[lo]+[float(t) for t in series.ts if lo<t<hi]+[hi]
    # Split at actual stage edges instead of smearing a motor cutoff.
    end=0.
    for st in rt.stages:
        start=end+float(st.get('fire_delay_s') or 0.);end=start+st['duration_s']
        cuts.extend(x for x in (start,end) if lo<x<hi)
    cuts=sorted(set(cuts));ab,weights=np.polynomial.legendre.leggauss(nodes);out=np.zeros(18);dur=0.;amax=0.;mmin=math.inf;mmax=0.
    for a,b in zip(cuts,cuts[1:]):
        mid=(a+b)/2;half=(b-a)/2
        if clock=='state':fac=1.
        else:
            pa=series.at(a)[3];pb=series.at(b)[3];fac=(pb-pa)/(b-a)
            if fac<=0:raise ValueError('Nonpositive packet-time segment')
        dur+=(b-a)*fac
        for x,w in zip(ab,weights):
            t=mid+half*x;values,z=evaluate(rt,series,t,controls);out+=half*w*fac*values
            amax=max(amax,z['alpha']);mmin=min(mmin,z['mach']);mmax=max(mmax,z['mach'])
    return out,dur,dict(alpha_max_deg=math.degrees(amax),mach_min=mmin,mach_max=mmax)


def window(rt,s,lo,hi,controls):
    if not s.valid(lo,hi):return dict(start_s=lo,end_s=hi,status='unsupported_clock_or_coverage')
    integ,dt,extra=integrate(rt,s,lo,hi,3,'state',controls);fine,_,_=integrate(rt,s,lo,hi,5,'state',controls);pack,pdt,_=integrate(rt,s,lo,hi,3,'packet',controls)
    p0,v0,_,_=s.at(lo);p1,v1,_,_=s.at(hi);dv=v1-v0;delta_e=.5*float(v1@v1-v0@v0)+G*(p1[1]-p0[1])
    model_de=integ[12]-integ[13]-integ[14];eres=model_de-delta_e;power_req=integ[12]-delta_e;work_pred=integ[13]+integ[14]
    vm=s.at((lo+hi)/2)[1];vh=vm/np.linalg.norm(vm);imerr=integ[:3]-dv;parallel=float(imerr@vh);normal=imerr-parallel*vh
    areq=dv-np.array([0.,-G*dt,0.])-integ[3:6];pred=integ[6:9];rq=areq-float(areq@vh)*vh;pn=pred-float(pred@vh)*vh
    peak_sign=float(pn@rq)
    packet_de=pack[12]-pack[13]-pack[14]
    n=float(v1@v1+2*eres)
    return dict(start_s=lo,end_s=hi,status='ok',state_dt_s=dt,packet_dt_s=pdt,burn_active=rt.mass_thrust((lo+hi)/2)[1]>0,
        speed0_m_s=float(np.linalg.norm(v0)),speed1_m_s=float(np.linalg.norm(v1)),height_delta_m=float(p1[1]-p0[1]),alpha_mean_deg=math.degrees(integ[15]/dt),**extra,
        observed_specific_energy_change=delta_e,modeled_specific_energy_change=model_de,energy_excess_J_kg=eres,mean_power_excess_W_kg=eres/dt,equivalent_endpoint_speed_excess_m_s=math.sqrt(n)-float(np.linalg.norm(v1)) if n>0 else None,
        thrust_work_J_kg=integ[12],predicted_base_drag_work_J_kg=integ[13],predicted_incidence_drag_work_J_kg=integ[14],required_drag_work_J_kg=power_req,modeled_to_required_drag_work=work_pred/power_req if power_req>0 else None,
        velocity_impulse_residual_m_s=imerr,tangential_accel_residual_m_s2=parallel/dt,normal_accel_residual_g=float(np.linalg.norm(normal))/dt/G,
        required_mean_transverse_aero_g=float(np.linalg.norm(rq))/dt/G,modeled_mean_transverse_aero_g=float(np.linalg.norm(pn))/dt/G,transverse_direction_agrees=peak_sign>=0,
        position_versus_integrated_velocity_m=float(np.linalg.norm(p1-p0-integ[9:12])),position_versus_packet_velocity_m=float(np.linalg.norm(p1-p0-pack[9:12])),
        quadrature3_vs5_max_abs=float(np.max(np.abs(integ-fine))),quadrature_energy_difference=float((fine[12]-fine[13]-fine[14])-model_de),
        packet_energy_excess_J_kg=float(packet_de-delta_e),packet_model_to_required_drag=(pack[13]+pack[14])/(pack[12]-delta_e) if pack[12]>delta_e else None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    manifest=read(a.input/'SOURCE_MANIFEST.json')
    for r in manifest:
        p=a.input/r['path'];assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],str(p)
    histories=read(a.input/'anchor_full_histories.json.gz');infos=read(a.input/'infos.json')
    profile=read(ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json');defaults=read(ROOT/'config/effective_surface_defaults.json')
    controls=dict(force_checks=0,max_acceleration_difference=0.,max_power_identity_difference=0.,max_lift_power_abs=0.,zero_control_dependence_checks=0,source_pose_checks=0)
    summary=dict(scope='Fixed WT own p/v/Q conditional force and energy audit; no fitting, guidance or autonomous rerun. Units energyJ/kg powerW/kg, positive excess=model retains too much energy.',source_dev='ab93a013dbe5ed8010e3f925a4bcbc1fa83d6f8d',inputs=manifest,code_sha256={p:digest(ROOT/p) for p in ['src/aim120_model/effective_surface_core.py','src/aim120_model/surface_runtime.py','config/effective_surface_defaults.json','data/profile_sets/2.59.0.27/missiles/su_r_77_1.json']},cases={})
    for uid,(rep,eid) in ANCHORS.items():
        info=next(z for z in infos if z['replay']==rep and z['eid']==eid);rs=histories[uid]
        assert all(r.get('eid',eid)==eid for r in rs)
        s=Series(rs,info['epoch_tick']);rt=make_runtime(profile,defaults,rs[0]);rows=[]
        for r in rs:
            t=(r['tick']-info['epoch_tick'])/48
            if t>15:break
            p=C@np.asarray(r['p']);v=C@np.asarray(r['vm']);q=Rotation.from_matrix(C@np.asarray(r['Q'])@C).as_quat();state=tuple(p)+tuple(v)+tuple(q)+(0.,0.,0.)
            d,z=rt.plant(t,state,(0.,0.));d2,_=rt.plant(t,state[:10]+(.3,-.7,.9),(.75,-.6));np.testing.assert_array_equal(d[3:6],d2[3:6]);controls['zero_control_dependence_checks']+=1
            np.testing.assert_allclose(np.column_stack(axes(tuple(q))),C@np.asarray(r['Q'])@C,rtol=0,atol=1e-13);controls['source_pose_checks']+=1
            rows.append(dict(age_s=t,index=r['index'],tick=r['tick'],packet_ms=r['packet_ms'],code=r['code'],timer=r['timer'],raw_sha=r['raw_sha'],p=r['p'],vm=r['vm'],Q=r['Q'],**z))
        fixed=[window(rt,s,lo,hi,controls) for lo,hi in WINDOWS]
        bins=[window(rt,s,float(lo),float(lo+.5),controls) for lo in np.arange(.5,15,.5)]
        entry=dict(identity=[rep,eid],epoch_tick=info['epoch_tick'],record_start_s=s.ts[0],record_end_s=s.ts[-1],time_breaks=s.breaks,model_parameters=rt.p,reference_area_m2=rt.area,propulsion=rt.stages,windows=fixed,half_second_bins=bins)
        summary['cases'][uid]=entry;save(a.out/f'samples_{uid}.json.gz',rows)
        print('CASE',uid,json.dumps(clean({k:v for k,v in entry.items() if k not in ('windows','half_second_bins')})),flush=True)
        for w in fixed:print('ENERGY_WINDOW',uid,json.dumps(clean(w)),flush=True)
    assert controls['max_acceleration_difference']<1e-8,controls
    assert controls['max_power_identity_difference']<1e-7,controls
    summary['verification']=controls;save(a.out/'summary.json',summary)
    print('FORCE_VERIFICATION',json.dumps(controls),flush=True)
if __name__=='__main__':main()
